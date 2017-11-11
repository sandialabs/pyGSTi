import unittest
import os
import numpy as np

import pygsti
from pygsti.construction import std1Q_XY as std
import pygsti.io.jsoncodec as jc

from ..testutils import BaseTestCase, compare_files, temp_files

class CodecsTestCase(BaseTestCase):

    def setUp(self):
        super(CodecsTestCase, self).setUp()
        self.gateset = std.gs_target

        self.germs = pygsti.construction.gatestring_list( [('Gx',), ('Gy',) ] ) #abridged for speed
        self.fiducials = std.fiducials
        self.maxLens = [1,2]
        self.gateLabels = list(self.gateset.gates.keys())

        self.lsgstStrings = pygsti.construction.make_lsgst_lists(
            self.gateLabels, self.fiducials, self.fiducials, self.germs, self.maxLens )

        self.datagen_gateset = self.gateset.depolarize(gate_noise=0.05, spam_noise=0.1)
        self.ds = pygsti.construction.generate_fake_data(
            self.datagen_gateset, self.lsgstStrings[-1],
            nSamples=1000,sampleError='binomial', seed=100)

        self.results = self.runSilent(pygsti.do_long_sequence_gst,
                                     self.ds, std.gs_target, self.fiducials, self.fiducials,
                                     self.germs, self.maxLens)

        #make a confidence region factory
        estLbl = "default"
        crfact = self.results.estimates[estLbl].add_confidence_region_factory('go0', 'final')
        crfact.compute_hessian(comm=None)
        crfact.project_hessian('std')

        #create a Workspace object
        self.ws = pygsti.report.create_standard_report(self.results, None, 
                                                       title="GST Codec TEST Report",
                                                       confidenceLevel=95)


class TestCodecs(CodecsTestCase):
    
    def test_json_codec(self):

        #string list
        s = jc.dumps(self.lsgstStrings)
        x = jc.loads(s)
        self.assertEqual(x, self.lsgstStrings)

        # DataSet
        s = jc.dumps(self.ds)
        x = jc.loads(s)
        self.assertEqual(list(x.keys()), list(self.ds.keys()))
        self.assertEqual(x[('Gx',)].as_dict(), self.ds[('Gx',)].as_dict())

        # GateSet
        s = jc.dumps(self.datagen_gateset)
        with open(temp_files + "/gateset.json",'w') as f:
            jc.dump(self.datagen_gateset, f)
        with open(temp_files + "/gateset.json",'r') as f:
            x = jc.load(f)
        self.assertAlmostEqual(self.datagen_gateset.frobeniusdist(x),0)

        # Results (containing confidence region)
        with open(temp_files + "/results.json",'w') as f:
            jc.dump(self.results, f)
        with open(temp_files + "/results.json",'r') as f:
            x = jc.load(f)
        self.assertEqual(list(x.estimates.keys()), list(self.results.estimates.keys()))
        self.assertEqual(list(x.estimates['default'].confidence_region_factories.keys()),
                         list(self.results.estimates['default'].confidence_region_factories.keys()))

        # Workspace
        s = jc.dumps(self.ws)
        x = jc.loads(s)
         #TODO: comparison (?)


if __name__ == "__main__":
    unittest.main(verbosity=2)

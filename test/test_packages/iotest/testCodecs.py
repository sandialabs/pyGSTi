import unittest
import os
import numpy as np
import pickle

import pygsti
from pygsti.construction import std1Q_XY as std
import pygsti.io.json as json
import pygsti.io.msgpack as msgpack

from ..testutils import BaseTestCase, compare_files, temp_files

class CodecsTestCase(BaseTestCase):

    def setUp(self):
        std.gs_target._check_paramvec()
        super(CodecsTestCase, self).setUp()
        self.gateset = std.gs_target

        self.germs = pygsti.construction.gatestring_list( [('Gx',), ('Gy',) ] ) #abridged for speed
        self.fiducials = std.fiducials
        self.maxLens = [1,2]
        self.gateLabels = list(self.gateset.gates.keys())

        self.lsgstStrings = pygsti.construction.make_lsgst_lists(
            self.gateLabels, self.fiducials, self.fiducials, self.germs, self.maxLens )

        self.datagen_gateset = self.gateset.depolarize(gate_noise=0.05, spam_noise=0.1)
        test = self.datagen_gateset.copy()
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
        std.gs_target._check_paramvec()


class TestCodecs(CodecsTestCase):
    
    def test_json(self):
        
        #string list
        s = json.dumps(self.lsgstStrings)
        x = json.loads(s)
        self.assertEqual(x, self.lsgstStrings)

        # DataSet
        s = json.dumps(self.ds)
        x = json.loads(s)
        self.assertEqual(list(x.keys()), list(self.ds.keys()))
        self.assertEqual(x[('Gx',)].as_dict(), self.ds[('Gx',)].as_dict())

        # GateSet
        s = json.dumps(self.datagen_gateset)
        with open(temp_files + "/gateset.json",'w') as f:
            json.dump(self.datagen_gateset, f)
        with open(temp_files + "/gateset.json",'r') as f:
            x = json.load(f)

        #print(s)
        x._check_paramvec(True)
        self.assertAlmostEqual(self.datagen_gateset.frobeniusdist(x),0)

        # Results (containing confidence region)
        std.gs_target._check_paramvec()
        print("gs_target = ",id(std.gs_target))
        print("rho0 parent = ",id(std.gs_target.preps['rho0'].parent))
        with open(temp_files + "/results.json",'w') as f:
            json.dump(self.results, f)
        print("gs_target2 = ",id(std.gs_target))
        print("rho0 parent2 = ",id(std.gs_target.preps['rho0'].parent))
        std.gs_target._check_paramvec()            
        with open(temp_files + "/results.json",'r') as f:
            x = json.load(f)
        self.assertEqual(list(x.estimates.keys()), list(self.results.estimates.keys()))
        self.assertEqual(list(x.estimates['default'].confidence_region_factories.keys()),
                         list(self.results.estimates['default'].confidence_region_factories.keys()))

        
        # Workspace
        s = json.dumps(self.ws)
        x = json.loads(s)
         #TODO: comparison (?)


    def test_msgpack(self):

        #string list
        s = msgpack.dumps(self.lsgstStrings)
        x = msgpack.loads(s)
        self.assertEqual(x, self.lsgstStrings)

        # DataSet
        s = msgpack.dumps(self.ds)
        x = msgpack.loads(s)
        self.assertEqual(list(x.keys()), list(self.ds.keys()))
        self.assertEqual(x[('Gx',)].as_dict(), self.ds[('Gx',)].as_dict())

        # GateSet
        s = msgpack.dumps(self.datagen_gateset)
        with open(temp_files + "/gateset.mpk",'wb') as f:
            msgpack.dump(self.datagen_gateset, f)
        with open(temp_files + "/gateset.mpk",'rb') as f:
            x = msgpack.load(f)
        self.assertAlmostEqual(self.datagen_gateset.frobeniusdist(x),0)

        # Results (containing confidence region)
        with open(temp_files + "/results.mpk",'wb') as f:
            msgpack.dump(self.results, f)
        with open(temp_files + "/results.mpk",'rb') as f:
            x = msgpack.load(f)
        self.assertEqual(list(x.estimates.keys()), list(self.results.estimates.keys()))
        self.assertEqual(list(x.estimates['default'].confidence_region_factories.keys()),
                         list(self.results.estimates['default'].confidence_region_factories.keys()))

        # Workspace
        s = msgpack.dumps(self.ws)
        x = msgpack.loads(s)
         #TODO: comparison (?)


    def test_pickle(self):

        #string list
        s = pickle.dumps(self.lsgstStrings)
        x = pickle.loads(s)
        self.assertEqual(x, self.lsgstStrings)

        # DataSet
        s = pickle.dumps(self.ds)
        x = pickle.loads(s)
        self.assertEqual(list(x.keys()), list(self.ds.keys()))
        self.assertEqual(x[('Gx',)].as_dict(), self.ds[('Gx',)].as_dict())

        # GateSet
        s = pickle.dumps(self.datagen_gateset)
        with open(temp_files + "/gateset.pickle",'wb') as f:
            pickle.dump(self.datagen_gateset, f)
        with open(temp_files + "/gateset.pickle",'rb') as f:
            x = pickle.load(f)
        self.assertAlmostEqual(self.datagen_gateset.frobeniusdist(x),0)

        # Results (containing confidence region)
        with open(temp_files + "/results.pickle",'wb') as f:
            pickle.dump(self.results, f)
        with open(temp_files + "/results.pickle",'rb') as f:
            x = pickle.load(f)
        self.assertEqual(list(x.estimates.keys()), list(self.results.estimates.keys()))
        self.assertEqual(list(x.estimates['default'].confidence_region_factories.keys()),
                         list(self.results.estimates['default'].confidence_region_factories.keys()))

        # Workspace
        s = pickle.dumps(self.ws)
        x = pickle.loads(s)
         #TODO: comparison (?)


    


if __name__ == "__main__":
    unittest.main(verbosity=2)

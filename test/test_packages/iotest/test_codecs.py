
import logging
mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.WARNING)

import unittest
import sys
import numpy as np
import pickle
import collections

import pygsti
from pygsti.modelpacks.legacy import std1Q_XY as std
from pygsti.baseobjs.label import CircuitLabel

from ..testutils import BaseTestCase, temp_files


class ObjDerivedFromStdType(list):
    def __init__(self,listInit):
        self.extra = "Hello"
        super(ObjDerivedFromStdType,self).__init__(listInit)


testObj = ObjDerivedFromStdType( (1,2,3) )
testObj.__class__.__module__ = "pygsti.baseobjs" # make object look like a pygsti-native object so it gets special serialization treatment.
sys.modules['pygsti.baseobjs'].ObjDerivedFromStdType = ObjDerivedFromStdType

class CodecsTestCase(BaseTestCase):

    def setUp(self):
        std.target_model()._check_paramvec()
        super(CodecsTestCase, self).setUp()
        self.model = std.target_model()

        self.germs = pygsti.circuits.to_circuits([('Gx',), ('Gy',)]) #abridged for speed
        self.fiducials = std.fiducials
        self.maxLens = [1,2]
        self.op_labels = list(self.model.operations.keys())

        self.lsgstStrings = pygsti.circuits.create_lsgst_circuit_lists(
            self.op_labels, self.fiducials, self.fiducials, self.germs, self.maxLens )

        self.datagen_gateset = self.model.depolarize(op_noise=0.05, spam_noise=0.1)
        test = self.datagen_gateset.copy()
        self.ds = pygsti.data.simulate_data(
            self.datagen_gateset, self.lsgstStrings[-1],
            num_samples=1000,sample_error='binomial', seed=100)

        #Make an model with instruments
        E = self.datagen_gateset.povms['Mdefault']['0']
        Erem = self.datagen_gateset.povms['Mdefault']['1']
        Gmz_plus = np.dot(E,E.T)
        Gmz_minus = np.dot(Erem,Erem.T)
        self.mdl_withInst = self.datagen_gateset.copy()
        self.mdl_withInst.instruments['Iz'] = pygsti.modelmembers.instruments.Instrument({'plus': Gmz_plus, 'minus': Gmz_minus})
        self.mdl_withInst.instruments['Iztp'] = pygsti.modelmembers.instruments.TPInstrument({'plus': Gmz_plus, 'minus': Gmz_minus})
        self.mdl_withInst.num_params  # rebuilds parameter vector

        self.results = self.runSilent(pygsti.run_long_sequence_gst,
                                      self.ds, std.target_model(), self.fiducials, self.fiducials,
                                      self.germs, self.maxLens)

        #make a confidence region factory
        estLbl = self.results.name
        crfact = self.results.estimates[estLbl].add_confidence_region_factory('stdgaugeopt', 'final')
        crfact.compute_hessian(comm=None)
        crfact.project_hessian('std')

        #create a Workspace object
        self.ws = pygsti.report.create_standard_report(self.results, None,
                                                       title="GST Codec TEST Report",
                                                       confidence_level=95)
        std.target_model()._check_paramvec()

        #create miscellaneous other objects
        self.miscObjects = []
        self.miscObjects.append( pygsti.baseobjs.OutcomeLabelDict(
            [( ('0',), 90 ), ( ('1',), 10)]) )


class TestCodecs(CodecsTestCase):

    def test_pickle(self):

        #basic types
        s = pickle.dumps(range(10))
        x = pickle.loads(s)
        s = pickle.dumps(4+3.0j)
        x = pickle.loads(s)
        s = pickle.dumps(np.array([1,2,3,4],'d'))
        x = pickle.loads(s)
        s = pickle.dumps( testObj ) #b/c we've messed with its __module__ this won't work...
        x = pickle.loads(s)

        #string list
        s = pickle.dumps(self.lsgstStrings)
        x = pickle.loads(s)
        self.assertEqual(x, self.lsgstStrings)

        # DataSet
        s = pickle.dumps(self.ds)
        x = pickle.loads(s)
        self.assertEqual(list(x.keys()), list(self.ds.keys()))
        self.assertEqual(x[('Gx',)].to_dict(), self.ds[('Gx',)].to_dict())

        # Model
        s = pickle.dumps(self.datagen_gateset)
        with open(temp_files + "/model.pickle",'wb') as f:
            pickle.dump(self.datagen_gateset, f)
        with open(temp_files + "/model.pickle",'rb') as f:
            x = pickle.load(f)
        self.assertAlmostEqual(self.datagen_gateset.frobeniusdist(x),0)
        s = pickle.dumps(self.mdl_withInst)
        x = pickle.loads(s)
        self.assertAlmostEqual(self.mdl_withInst.frobeniusdist(x),0)

        # Results (containing confidence region)
        with open(temp_files + "/results.pickle",'wb') as f:
            pickle.dump(self.results, f)
        with open(temp_files + "/results.pickle",'rb') as f:
            x = pickle.load(f)
        self.assertEqual(list(x.estimates.keys()), list(self.results.estimates.keys()))
        self.assertEqual(list(x.estimates[x.name].confidence_region_factories.keys()),
                         list(self.results.estimates[self.results.name].confidence_region_factories.keys()))

        # Workspace
        pygsti.report.workspace.enable_plotly_pickling() # b/c workspace cache may contain plotly figures
        s = pickle.dumps(self.ws)
        x = pickle.loads(s)
        pygsti.report.workspace.disable_plotly_pickling()
         #TODO: comparison (?)

        #Misc other objects
        for obj in self.miscObjects:
            s = pickle.dumps(obj)
            x = pickle.loads(s)

    def test_std_decode(self):
        # test _decode_std_base function since it isn't easily reached/covered:
        binary = False

        mock_json_obj = {'__tuple__': True}
        with self.assertRaises(AssertionError):
            pygsti.serialization.jsoncodec._decode_std_base(mock_json_obj, "", binary)

        mock_json_obj = {'__list__': ['a','b']}
        pygsti.serialization.jsoncodec._decode_std_base(mock_json_obj, [], binary)

        mock_json_obj = {'__set__': ['a','b']}
        pygsti.serialization.jsoncodec._decode_std_base(mock_json_obj, set(), binary)

        mock_json_obj = {'__ndict__': [('key1','val1'),('key2','val2')]}
        pygsti.serialization.jsoncodec._decode_std_base(mock_json_obj, {}, binary)

        mock_json_obj = {'__odict__': [('key1','val1'),('key2','val2')]}
        pygsti.serialization.jsoncodec._decode_std_base(mock_json_obj, collections.OrderedDict(), binary)

        mock_json_obj = {'__uuid__': True}
        with self.assertRaises(AssertionError):
            pygsti.serialization.jsoncodec._decode_std_base(mock_json_obj, "", binary)

        mock_json_obj = {'__ndarray__': True}
        with self.assertRaises(AssertionError):
            pygsti.serialization.jsoncodec._decode_std_base(mock_json_obj, "", binary)

        mock_json_obj = {'__npgeneric__': True}
        with self.assertRaises(AssertionError):
            pygsti.serialization.jsoncodec._decode_std_base(mock_json_obj, "", binary)

        mock_json_obj = {'__complex__': True}
        with self.assertRaises(AssertionError):
            pygsti.serialization.jsoncodec._decode_std_base(mock_json_obj, "", binary)

        mock_json_obj = {'__counter__': True}
        with self.assertRaises(AssertionError):
            pygsti.serialization.jsoncodec._decode_std_base(mock_json_obj, "", binary)

        mock_json_obj = {'__slice__': True}
        with self.assertRaises(AssertionError):
            pygsti.serialization.jsoncodec._decode_std_base(mock_json_obj, "", binary)


    def test_helpers(self):
        pygsti.serialization.jsoncodec._tostr("Hi")
        pygsti.serialization.jsoncodec._tostr(b"Hi")
        pygsti.serialization.jsoncodec._tobin("Hi")
        pygsti.serialization.jsoncodec._tobin(b"Hi")

    def test_pickle_dataset_with_circuitlabels(self):
        #A later-added test checking whether Circuits containing CiruitLabels
        # are correctly pickled within a DataSet.  In particular correct
        # preservation of the circuit's .str property
        pygsti.circuits.Circuit.default_expand_subcircuits = False # so exponentiation => CircuitLabels
        ds = pygsti.data.DataSet(outcome_labels=('0', '1'))
        c0 = pygsti.circuits.Circuit(None, stringrep="[Gx:0Gy:1]")
        c = c0**2
        self.assertTrue(isinstance(c.tup[0], CircuitLabel))
        self.assertEqual(c.str, "([Gx:0Gy:1])^2")
        ds.add_count_dict(c, {'0': 50, '1': 50})
        s = pickle.dumps(ds)
        ds2 = pickle.loads(s)
        c2 = list(ds2.keys())[0]
        self.assertEqual(c2.str, "([Gx:0Gy:1])^2")
        pygsti.circuits.Circuit.default_expand_subcircuits = True


if __name__ == "__main__":
    unittest.main(verbosity=2)

import unittest
import os,sys
import numpy as np
import pickle
import collections

import pygsti
from pygsti.construction import std1Q_XY as std
import pygsti.io.json as json
import pygsti.io.msgpack as msgpack

from ..testutils import BaseTestCase, compare_files, temp_files


class ObjDerivedFromStdType(list):
    def __init__(self,listInit):
        self.extra = "Hello"
        super(ObjDerivedFromStdType,self).__init__(listInit)
testObj = ObjDerivedFromStdType( (1,2,3) )
testObj.__class__.__module__ = "pygsti.objects" # make object look like a pygsti-native object so it gets special serialization treatment.
sys.modules['pygsti.objects'].ObjDerivedFromStdType = ObjDerivedFromStdType

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
        
        #Make an gateset with instruments
        E = self.datagen_gateset.povms['Mdefault']['0']
        Erem = self.datagen_gateset.povms['Mdefault']['1']
        Gmz_plus = np.dot(E,E.T)
        Gmz_minus = np.dot(Erem,Erem.T)
        self.gs_withInst = self.datagen_gateset.copy()
        self.gs_withInst.instruments['Iz'] = pygsti.obj.Instrument({'plus': Gmz_plus, 'minus': Gmz_minus})
        self.gs_withInst.instruments['Iztp'] = pygsti.obj.TPInstrument({'plus': Gmz_plus, 'minus': Gmz_minus})
        
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
        
        #create miscellaneous other objects
        self.miscObjects = []
        self.miscObjects.append( pygsti.objects.labeldicts.OutcomeLabelDict(
            [( ('0',), 90 ), ( ('1',), 10)]) )


class TestCodecs(CodecsTestCase):
    
    def test_json(self):
        
        #basic types
        s = json.dumps(range(10))
        x = json.loads(s)
        s = json.dumps(4+3.0j)
        x = json.loads(s)
        s = json.dumps(np.array([1,2,3,4],'d'))
        x = json.loads(s)
        s = json.dumps( testObj )
        x = json.loads(s)
        
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
        s = json.dumps(self.gs_withInst)
        x = json.loads(s)
        self.assertAlmostEqual(self.gs_withInst.frobeniusdist(x),0)

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

        #Misc other objects
        for obj in self.miscObjects:
            s = json.dumps(obj)
            x = json.loads(s)



    def test_msgpack(self):

        #basic types
        s = msgpack.dumps(range(10))
        x = msgpack.loads(s)
        s = msgpack.dumps(4+3.0j)
        x = msgpack.loads(s)
        s = msgpack.dumps(np.array([1,2,3,4],'d'))
        x = msgpack.loads(s)
        s = msgpack.dumps( testObj )
        x = msgpack.loads(s)

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
        s = msgpack.dumps(self.gs_withInst)
        x = msgpack.loads(s)
        self.assertAlmostEqual(self.gs_withInst.frobeniusdist(x),0)

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

        #Misc other objects
        for obj in self.miscObjects:
            s = msgpack.dumps(obj)
            x = msgpack.loads(s)



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
        self.assertEqual(x[('Gx',)].as_dict(), self.ds[('Gx',)].as_dict())

        # GateSet
        s = pickle.dumps(self.datagen_gateset)
        with open(temp_files + "/gateset.pickle",'wb') as f:
            pickle.dump(self.datagen_gateset, f)
        with open(temp_files + "/gateset.pickle",'rb') as f:
            x = pickle.load(f)
        self.assertAlmostEqual(self.datagen_gateset.frobeniusdist(x),0)
        s = pickle.dumps(self.gs_withInst)
        x = pickle.loads(s)
        self.assertAlmostEqual(self.gs_withInst.frobeniusdist(x),0)

        # Results (containing confidence region)
        with open(temp_files + "/results.pickle",'wb') as f:
            pickle.dump(self.results, f)
        with open(temp_files + "/results.pickle",'rb') as f:
            x = pickle.load(f)
        self.assertEqual(list(x.estimates.keys()), list(self.results.estimates.keys()))
        self.assertEqual(list(x.estimates['default'].confidence_region_factories.keys()),
                         list(self.results.estimates['default'].confidence_region_factories.keys()))

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
        # test decode_std_base function since it isn't easily reached/covered:
        binary = False
        
        mock_json_obj = {'__tuple__': True}
        with self.assertRaises(AssertionError):
            pygsti.io.jsoncodec.decode_std_base(mock_json_obj,"",binary)

        mock_json_obj = {'__list__': ['a','b']}
        pygsti.io.jsoncodec.decode_std_base(mock_json_obj,[],binary)

        mock_json_obj = {'__set__': ['a','b']}
        pygsti.io.jsoncodec.decode_std_base(mock_json_obj,set(),binary)

        mock_json_obj = {'__ndict__': [('key1','val1'),('key2','val2')]}
        pygsti.io.jsoncodec.decode_std_base(mock_json_obj,{},binary)

        mock_json_obj = {'__odict__': [('key1','val1'),('key2','val2')]}
        pygsti.io.jsoncodec.decode_std_base(mock_json_obj,collections.OrderedDict(),binary)

        mock_json_obj = {'__uuid__': True}
        with self.assertRaises(AssertionError):
            pygsti.io.jsoncodec.decode_std_base(mock_json_obj,"",binary)

        mock_json_obj = {'__ndarray__': True}
        with self.assertRaises(AssertionError):
            pygsti.io.jsoncodec.decode_std_base(mock_json_obj,"",binary)

        mock_json_obj = {'__npgeneric__': True}
        with self.assertRaises(AssertionError):
            pygsti.io.jsoncodec.decode_std_base(mock_json_obj,"",binary)

        mock_json_obj = {'__complex__': True}
        with self.assertRaises(AssertionError):
            pygsti.io.jsoncodec.decode_std_base(mock_json_obj,"",binary)

        mock_json_obj = {'__counter__': True}
        with self.assertRaises(AssertionError):
            pygsti.io.jsoncodec.decode_std_base(mock_json_obj,"",binary)

        mock_json_obj = {'__slice__': True}
        with self.assertRaises(AssertionError):
            pygsti.io.jsoncodec.decode_std_base(mock_json_obj,"",binary)

    
    def test_helpers(self):
        pygsti.io.jsoncodec.tostr("Hi")
        pygsti.io.jsoncodec.tostr(b"Hi")
        pygsti.io.jsoncodec.tobin("Hi")
        pygsti.io.jsoncodec.tobin(b"Hi")
    


if __name__ == "__main__":
    unittest.main(verbosity=2)

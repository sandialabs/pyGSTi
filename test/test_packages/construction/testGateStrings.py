import unittest
import copy
import pygsti
import os

from ..testutils import BaseTestCase, compare_files, temp_files
from pygsti.construction import std1Q_XY
from pygsti.objects import Label as L

class TestGateStringMethods(BaseTestCase):
    def test_string_compression(self):
        mdl = pygsti.objects.Circuit(None, stringrep="Gx^100")
        comp_gs = pygsti.objects.circuit.CompressedCircuit.compress_op_label_tuple(tuple(mdl))
        exp_gs = pygsti.objects.circuit.CompressedCircuit.expand_op_label_tuple(comp_gs)
        self.assertEqual(tuple(mdl), exp_gs)

    def test_python_string_conversion(self):
        mdl = pygsti.obj.Circuit(None, stringrep="Gx^3Gy^2GxGz")

        op_labels = (L('Gx'),L('Gy'),L('Gz'))
        pystr = mdl.to_pythonstr( op_labels )
        self.assertEqual( pystr, "AAABBAC" )

        gs2_tup = pygsti.obj.Circuit.from_pythonstr( pystr, op_labels )
        self.assertEqual( gs2_tup, tuple(mdl) )

    def test_std_lists_and_structs(self):
        opLabels = [L('Gx'),L('Gy')]
        strs = pygsti.construction.circuit_list( [('Gx',),('Gy',),('Gx','Gx')] )
        germs = pygsti.construction.circuit_list( [('Gx','Gy'),('Gy','Gy')] )
        testFidPairs = [(0,1)]
        testFidPairsDict = { (L('Gx'),L('Gy')): [(0,0),(0,1)], (L('Gy'),L('Gy')): [(0,0)] }

        # LSGST
        maxLens = [1,2]
        lsgstLists = pygsti.construction.make_lsgst_lists(
            std1Q_XY.target_model(), strs, strs, germs, maxLens, fidPairs=None,
            truncScheme="whole germ powers") #also try a Model as first arg
        lsgstStructs = pygsti.construction.make_lsgst_structs(
            std1Q_XY.target_model(), strs, strs, germs, maxLens, fidPairs=None,
            truncScheme="whole germ powers") #also try a Model as first arg
        self.assertEqual(set(lsgstLists[-1]), set(lsgstStructs[-1].allstrs))

        lsgstLists2 = pygsti.construction.make_lsgst_lists(
            opLabels, strs, strs, germs, maxLens, fidPairs=None,
            truncScheme="truncated germ powers")
        lsgstStructs2 = pygsti.construction.make_lsgst_structs(
            opLabels, strs, strs, germs, maxLens, fidPairs=None,
            truncScheme="truncated germ powers")
        self.assertEqual(set(lsgstLists2[-1]), set(lsgstStructs2[-1].allstrs))

        lsgstLists3 = pygsti.construction.make_lsgst_lists(
            opLabels, strs, strs, germs, maxLens, fidPairs=None,
            truncScheme="length as exponent")
        lsgstStructs3 = pygsti.construction.make_lsgst_structs(
            opLabels, strs, strs, germs, maxLens, fidPairs=None,
            truncScheme="length as exponent")
        self.assertEqual(set(lsgstLists3[-1]), set(lsgstStructs3[-1].allstrs))


        maxLens = [1,2]
        lsgstLists4 = pygsti.construction.make_lsgst_lists(
            opLabels, strs, strs, germs, maxLens, fidPairs=None,
            truncScheme="whole germ powers", nest=False)
        lsgstStructs4 = pygsti.construction.make_lsgst_structs(
            opLabels, strs, strs, germs, maxLens, fidPairs=None,
            truncScheme="whole germ powers", nest=False)
        self.assertEqual(set(lsgstLists4[-1]), set(lsgstStructs4[-1].allstrs))

        lsgstLists5 = pygsti.construction.make_lsgst_lists(
            opLabels, strs, strs, germs, maxLens, fidPairs=testFidPairs,
            truncScheme="whole germ powers")
        lsgstStructs5 = pygsti.construction.make_lsgst_structs(
            opLabels, strs, strs, germs, maxLens, fidPairs=testFidPairs,
            truncScheme="whole germ powers")
        self.assertEqual(set(lsgstLists5[-1]), set(lsgstStructs5[-1].allstrs))

        lsgstLists6 = pygsti.construction.make_lsgst_lists(
            opLabels, strs, strs, germs, maxLens, fidPairs=testFidPairsDict,
            truncScheme="whole germ powers")
        lsgstStructs6 = pygsti.construction.make_lsgst_structs(
            opLabels, strs, strs, germs, maxLens, fidPairs=testFidPairsDict,
            truncScheme="whole germ powers")
        self.assertEqual(set(lsgstLists6[-1]), set(lsgstStructs6[-1].allstrs))

        lsgstExpList = pygsti.construction.make_lsgst_experiment_list(
            opLabels, strs, strs, germs, maxLens, fidPairs=None,
            truncScheme="whole germ powers")
        lsgstExpListb = pygsti.construction.make_lsgst_experiment_list(
            std1Q_XY.target_model(), strs, strs, germs, maxLens, fidPairs=None,
            truncScheme="whole germ powers") # with Model as first arg

        with self.assertRaises(ValueError):
            pygsti.construction.make_lsgst_lists(
                opLabels, strs, strs, germs, maxLens, fidPairs=None,
                truncScheme="foobar")
        with self.assertRaises(ValueError):
            pygsti.construction.make_lsgst_structs(
                opLabels, strs, strs, germs, maxLens, fidPairs=None,
                truncScheme="foobar")

        lsgstLists7 = pygsti.construction.make_lsgst_lists(
            opLabels, strs, strs, germs, maxLens, fidPairs=None,
            truncScheme="whole germ powers", keepFraction=0.5, keepSeed=1234)
        lsgstStructs7 = pygsti.construction.make_lsgst_structs(
            opLabels, strs, strs, germs, maxLens, fidPairs=None,
            truncScheme="whole germ powers", keepFraction=0.5, keepSeed=1234)
        self.assertEqual(set(lsgstLists7[-1]), set(lsgstStructs7[-1].allstrs))

        lsgstLists8 = pygsti.construction.make_lsgst_lists(
            opLabels, strs, strs, germs, maxLens, fidPairs=testFidPairs,
            truncScheme="whole germ powers", keepFraction=0.7, keepSeed=1234)
        lsgstStructs8 = pygsti.construction.make_lsgst_structs(
            opLabels, strs, strs, germs, maxLens, fidPairs=testFidPairs,
            truncScheme="whole germ powers", keepFraction=0.7, keepSeed=1234)
        self.assertEqual(set(lsgstLists8[-1]), set(lsgstStructs8[-1].allstrs))

        # empty max-lengths ==> no output
        lsgstStructs9 = pygsti.construction.make_lsgst_structs(
            opLabels, strs, strs, germs, [] )
        self.assertEqual(len(lsgstStructs9), 0)

        # checks against datasets
        ds = pygsti.objects.DataSet(outcomeLabels=['0','1']) # a dataset that is missing
        ds.add_count_dict( ('Gx',), {'0': 10, '1': 90} )     # almost all our strings...
        ds.done_adding_data()
        lgst_strings = pygsti.construction.list_lgst_circuits(strs,strs,opLabels)
        lsgstStructs10 = pygsti.construction.make_lsgst_structs(
            opLabels, strs, strs, germs, maxLens, dscheck=ds, actionIfMissing="drop", verbosity=4 )
        self.assertEqual([pygsti.obj.Circuit(('Gx',))], lsgstStructs10[-1].allstrs)

        with self.assertRaises(ValueError):
            pygsti.construction.make_lsgst_structs(
                opLabels, strs, strs, germs, maxLens, dscheck=ds ) #missing sequences
        with self.assertRaises(ValueError):
            pygsti.construction.make_lsgst_structs(
                opLabels, strs, strs, germs, maxLens, dscheck=ds, actionIfMissing="foobar" ) #invalid action



        # ELGST
        maxLens = [1,2]
        elgstLists = pygsti.construction.make_elgst_lists(
            opLabels, germs, maxLens, truncScheme="whole germ powers")

        maxLens = [1,2]
        elgstLists2 = pygsti.construction.make_elgst_lists(
            opLabels, germs, maxLens, truncScheme="whole germ powers",
            nest=False, includeLGST=False)
        elgstLists2b = pygsti.construction.make_elgst_lists(
            std1Q_XY.target_model(), germs, maxLens, truncScheme="whole germ powers",
            nest=False, includeLGST=False) #with a Model as first arg


        elgstExpLists = pygsti.construction.make_elgst_experiment_list(
            opLabels, germs, maxLens, truncScheme="whole germ powers")

        with self.assertRaises(ValueError):
            pygsti.construction.make_elgst_lists(
                opLabels, germs, maxLens, truncScheme="foobar")





        #TODO: check values here



    def test_gatestring_object(self):
        s1 = pygsti.obj.Circuit( ('Gx','Gx'), stringrep="Gx^2" )
        s2 = pygsti.obj.Circuit( s1, stringrep="Gx^2" )
        s3 = s1 + s2
        s4 = s1**3
        s5 = s4
        s6 = copy.copy(s1)
        s7 = copy.deepcopy(s1)

        self.assertEqual( s1, ('Gx','Gx') )
        self.assertEqual( s2, ('Gx','Gx') )
        self.assertEqual( s3, ('Gx','Gx','Gx','Gx') )
        self.assertEqual( s4, ('Gx','Gx','Gx','Gx','Gx','Gx') )
        self.assertEqual( s5, s4 )
        self.assertEqual( s1, s6 )
        self.assertEqual( s1, s7 )


        b1 = s1.__lt__(s2)
        b2 = s1.__gt__(s2)
        #b1 = s1 < s2
        #b2 = s1 > s2

        with self.assertRaises(AssertionError):
            s1[0] = 'Gx' #cannot set items - like a tuple they're read-only
        with self.assertRaises(ValueError):
            bad = s1 + ("Gx",) #can't add non-Circuit to circuit
        with self.assertRaises(ValueError):
            pygsti.obj.Circuit( ('Gx','Gx'), stringrep="GxGy", check=True) #mismatch
        with self.assertRaises(ValueError):
            pygsti.obj.Circuit( None )
        with self.assertRaises(ValueError):
            pygsti.obj.Circuit( ('foobar',), stringrep="foobar", check=True ) # lexer illegal character

        #REMOVED: WeightedOpString
        #w1 = pygsti.obj.WeightedOpString( ('Gx','Gy'), "GxGy", weight=0.5)
        #w2 = pygsti.obj.WeightedOpString( ('Gy',), "Gy", weight=0.5)
        #w3 = w1 + w2
        #w4 = w2**2
        #w5 = s1 + w2
        #w6 = w2 + s1
        #w7 = copy.copy(w1)
        #
        #with self.assertRaises(ValueError):
        #    w1 + ('Gx',) #can only add to other Circuits
        #with self.assertRaises(ValueError):
        #    ('Gx',) + w1 #can only add to other Circuits
        #
        #w1_str = str(w1)
        #w1_repr = repr(w1)
        #x = w1[0]
        #x2 = w1[0:2]
        #
        #self.assertEqual( w1, ('Gx','Gy') ); self.assertEqual(w1.weight, 0.5)
        #self.assertEqual( w2, ('Gy',) ); self.assertEqual(w2.weight, 0.5)
        #self.assertEqual( w3, ('Gx','Gy','Gy') ); self.assertEqual(w3.weight, 1.0)
        #self.assertEqual( w4, ('Gy','Gy') ); self.assertEqual(w4.weight, 0.5)
        #self.assertEqual( w5, ('Gx','Gx','Gy') ); self.assertEqual(w5.weight, 0.5)
        #self.assertEqual( w6, ('Gy','Gx','Gx') ); self.assertEqual(w6.weight, 0.5)
        #self.assertEqual( x, 'Gx' )
        #self.assertEqual( x2, ('Gx','Gy') )
        #self.assertEqual( w1, w7)

        c1 = pygsti.objects.circuit.CompressedCircuit(s1)
        s1_expanded = c1.expand()
        self.assertEqual(s1,s1_expanded)

        with self.assertRaises(ValueError):
            pygsti.objects.circuit.CompressedCircuit( ('Gx',) ) #can only create from Circuits


if __name__ == "__main__":
    unittest.main(verbosity=2)

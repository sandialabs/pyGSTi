import unittest
import copy
import pygsti
import os

from ..testutils import BaseTestCase, compare_files, temp_files
from pygsti.construction import std1Q_XY
from pygsti.objects import Label as L

class TestGateStringMethods(BaseTestCase):
    def test_simple(self):
        #The workhorse function is pygsti.construction.create_circuit_list, which executes its positional arguments within a nested
        # loop given by iterable keyword arguments.  That's a mouthful, so let's look at a few examples:
        As = [('a1',),('a2',)]
        Bs = [('b1',), ('b2',)]

        def rep2(x):
            return x+x

        def asserter(x):
            assert(False)

        def samestr(x):
            return "Gx" #to test string processing

        def sametup(x):
            return "Gx" #to test string processing


        list0 = pygsti.construction.create_circuit_list("")
        list1 = pygsti.construction.create_circuit_list("a", a=As)
        list2 = pygsti.construction.create_circuit_list("a+b", a=As, b=Bs, order=['a','b'])
        list3 = pygsti.construction.create_circuit_list("a+b", a=As, b=Bs, order=['b','a'])
        list4 = pygsti.construction.create_circuit_list("R(a)+c", a=As, c=[('c',)], R=rep2, order=['a','c'])
        list5 = pygsti.construction.create_circuit_list("Ast(a)", a=As, Ast=asserter)
        list6 = pygsti.construction.create_circuit_list("SS(a)", a=As, SS=samestr)
        list7 = pygsti.construction.circuit_list(list1)

        self.assertEqual(list0, pygsti.construction.circuit_list([ () ] )) #special case: get the empty operation sequence
        self.assertEqual(list1, pygsti.construction.circuit_list(As))
        self.assertEqual(list2, pygsti.construction.circuit_list([('a1','b1'),('a1','b2'),('a2','b1'),('a2','b2')]))
        self.assertEqual(list3, pygsti.construction.circuit_list([('a1','b1'),('a2','b1'),('a1','b2'),('a2','b2')]))
        self.assertEqual(list4, pygsti.construction.circuit_list([('a1','a1','c'),('a2','a2','c')]))
        self.assertEqual(list5, []) # failed assertions cause item to be skipped
        self.assertEqual(list6, pygsti.construction.circuit_list([('Gx',), ('Gx',)])) #strs => parser => Circuits
        self.assertEqual(list7, list1)

        with self.assertRaises(ValueError):
            pygsti.construction.circuit_list( [ {'foo': "Bar"} ] ) #cannot convert dicts to Circuits...


    def test_truncate_methods(self):
        self.assertEqual( pygsti.construction.repeat_and_truncate(('A','B','C'),5), ('A','B','C','A','B'))
        self.assertEqual( pygsti.construction.repeat_with_max_length(('A','B','C'),5), ('A','B','C'))
        self.assertEqual( pygsti.construction.repeat_count_with_max_length(('A','B','C'),5), 1)

    def test_fiducials_germs(self):
        fids  = pygsti.construction.circuit_list( [ ('Gf0',), ('Gf1',)    ] )
        germs = pygsti.construction.circuit_list( [ ('G0',), ('G1a','G1b')] )

        gateStrings1 = pygsti.construction.create_circuit_list("f0+germ*e+f1", f0=fids, f1=fids,
                                                                  germ=germs, e=2, order=["germ","f0","f1"])
        expected1 = ["Gf0(G0)^2Gf0",
                     "Gf0(G0)^2Gf1",
                     "Gf1(G0)^2Gf0",
                     "Gf1(G0)^2Gf1",
                     "Gf0(G1aG1b)^2Gf0",
                     "Gf0(G1aG1b)^2Gf1",
                     "Gf1(G1aG1b)^2Gf0",
                     "Gf1(G1aG1b)^2Gf1" ]
        self.assertEqual( [ x.str for x in gateStrings1], expected1 )


        gateStrings2 = pygsti.construction.create_circuit_list("f0+T(germ,N)+f1", f0=fids, f1=fids,
                                                                  germ=germs, N=3, T=pygsti.construction.repeat_and_truncate,
                                                                  order=["germ","f0","f1"])
        expected2 = ["Gf0G0G0G0Gf0",
                     "Gf0G0G0G0Gf1",
                     "Gf1G0G0G0Gf0",
                     "Gf1G0G0G0Gf1",
                     "Gf0G1aG1bG1aGf0",
                     "Gf0G1aG1bG1aGf1",
                     "Gf1G1aG1bG1aGf0",
                     "Gf1G1aG1bG1aGf1" ]
        self.assertEqual( [ x.str for x in gateStrings2], expected2 )


        gateStrings3 = pygsti.construction.create_circuit_list("f0+T(germ,N)+f1", f0=fids, f1=fids,
                                                                  germ=germs, N=3,
                                                                  T=pygsti.construction.repeat_with_max_length,
                                                                  order=["germ","f0","f1"])
        expected3 = [ "Gf0(G0)^3Gf0",
                      "Gf0(G0)^3Gf1",
                      "Gf1(G0)^3Gf0",
                      "Gf1(G0)^3Gf1",
                      "Gf0(G1aG1b)Gf0",
                      "Gf0(G1aG1b)Gf1",
                      "Gf1(G1aG1b)Gf0",
                      "Gf1(G1aG1b)Gf1" ]
        self.assertEqual( [ x.str for x in gateStrings3], expected3 )

    def test_string_compression(self):
        mdl = pygsti.objects.Circuit(None, stringrep="Gx^100")
        comp_gs = pygsti.objects.circuit.CompressedCircuit.compress_op_label_tuple(tuple(mdl))
        exp_gs = pygsti.objects.circuit.CompressedCircuit.expand_op_label_tuple(comp_gs)
        self.assertEqual(tuple(mdl), exp_gs)

    def test_repeat(self):
        mdl = pygsti.objects.Circuit( ('Gx','Gx','Gy') )

        gs2 = pygsti.construction.repeat(mdl, 2)
        self.assertEqual( gs2, pygsti.obj.Circuit( ('Gx','Gx','Gy','Gx','Gx','Gy') ))

        gs3 = pygsti.construction.repeat_with_max_length(mdl, 7)
        self.assertEqual( gs3, pygsti.obj.Circuit( ('Gx','Gx','Gy','Gx','Gx','Gy') ))

        gs4 = pygsti.construction.repeat_and_truncate(mdl, 4)
        self.assertEqual( gs4, pygsti.obj.Circuit( ('Gx','Gx','Gy','Gx') ))

        gs5 = pygsti.construction.repeat_remainder_for_truncation(mdl, 4)
        self.assertEqual( gs5, pygsti.obj.Circuit( ('Gx',) ))

    def test_simplify(self):
        s = "{}Gx^1Gy{}Gz^1"
        self.assertEqual( pygsti.construction.simplify_str(s), "GxGyGz" )

        s = "{}Gx^1(Gy)^2{}Gz^1"
        self.assertEqual( pygsti.construction.simplify_str(s), "Gx(Gy)^2Gz" )

        s = "{}{}^1{}"
        self.assertEqual( pygsti.construction.simplify_str(s), "{}" )


    def test_lists(self):
        expected_allStrs = set( pygsti.construction.circuit_list(
                [(),('Gx',),('Gy',),('Gx','Gx'),('Gx','Gy'),('Gy','Gx'),('Gy','Gy')] ))
        allStrs = pygsti.construction.list_all_circuits( ('Gx','Gy'), 0,2 )
        self.assertEqual( set(allStrs), expected_allStrs)

        expected_onelenStrs = set( pygsti.construction.circuit_list(
            [('Gx','Gx'),('Gx','Gy'),('Gy','Gx'),('Gy','Gy')] ))
        onelenStrs = pygsti.construction.list_all_circuits_onelen(('Gx','Gy'), 2)
        self.assertEqual( set(onelenStrs), expected_onelenStrs )

        allStrs = list(pygsti.construction.gen_all_circuits( ('Gx','Gy'), 0,2 ))
        #self.assertEqual( set(allStrs), set([(),('Gx',),('Gy',),('Gx','Gx'),('Gx','Gy'),('Gy','Gx'),('Gy','Gy')]))
        #self.assertEqual( set(allStrs), set([(),('Gx',),('Gy',),('Gx','Gy'),('Gy','Gx')]))

        randStrs = pygsti.construction.list_random_circuits_onelen( ('Gx','Gy','Gz'), 2, 3)
        self.assertEqual( len(randStrs), 3 )
        self.assertTrue( all( [len(s)==2 for s in randStrs] ) )

        partialStrs = pygsti.construction.list_partial_strings( ('G1','G2','G3') )
        self.assertEqual( partialStrs, [ (), ('G1',), ('G1','G2'), ('G1','G2','G3') ] )


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


    def test_alias_manips(self):
        orig_list = pygsti.construction.circuit_list(
            [ ('Gx','Gx'), ('Gx','Gy'), ('Gx','Gx','Gx'), ('Gy','Gy'), ('Gi',) ] )

        list0 = pygsti.construction.translate_circuit_list(orig_list, None)
        self.assertEqual(list0, orig_list)
        
        list1 = pygsti.construction.translate_circuit_list(orig_list, {L('Gx'): (L('Gx2'),), L('Gy'): (L('Gy'),)} )
        list2 = pygsti.construction.translate_circuit_list(orig_list, {L('Gi'): (L('Gx'),L('Gx'),L('Gx'),L('Gx'))} )
        print(list1)
        
        expected_list1 = pygsti.construction.circuit_list(
                    [ ('Gx2','Gx2'), ('Gx2','Gy'), ('Gx2','Gx2','Gx2'), ('Gy','Gy'), ('Gi',) ] )
        expected_list2 = pygsti.construction.circuit_list(
                    [ ('Gx','Gx'), ('Gx','Gy'), ('Gx','Gx','Gx'), ('Gy','Gy'), ('Gx','Gx','Gx','Gx') ] )

        self.assertEqual(list1, expected_list1)
        self.assertEqual(list2, expected_list2)

        aliasDict1 = { 'A': ('B','B') }
        aliasDict2 = { 'B': ('C','C') }
        aliasDict3 = pygsti.construction.compose_alias_dicts(aliasDict1, aliasDict2)
        self.assertEqual(aliasDict3, { 'A': ('C','C','C','C') } )

    def test_manipulate_strings(self):
        sequenceRules = [
            (('A', 'B'), ('A', 'B\'')),
            (('B', 'A'), ('B\'\'', 'A')),
            (('C', 'A'), ('C', 'A\'')),
            (('B', 'C'), ('B', 'C\'')),
            (('D',), ('E',)),
            (('A','A'), ('A','B','C',))]

        result = pygsti.construction.manipulate_circuit(tuple('BAB'), sequenceRules)
        self.assertEqual(result, ("B''","A","B'"))

        result = pygsti.construction.manipulate_circuit(tuple('ABA'), sequenceRules)
        self.assertEqual(result, ("A","B'","A"))

        result = pygsti.construction.manipulate_circuit(tuple('CAB'), sequenceRules)
        self.assertEqual(result, ("C","A'","B'"))

        result = pygsti.construction.manipulate_circuit(tuple('ABC'), sequenceRules)
        self.assertEqual(result, ("A","B'","C'"))

        result = pygsti.construction.manipulate_circuit(tuple('DD'), sequenceRules)
        self.assertEqual(result, ("E","E"))

        result = pygsti.construction.manipulate_circuit(tuple('AA'), sequenceRules)
        self.assertEqual(result, ("A","B","C"))

        result = pygsti.construction.manipulate_circuit(tuple('AAAA'), sequenceRules)
        self.assertEqual(result, ("A","B","C","B","C","B","C"))

        results = pygsti.construction.manipulate_circuit_list([tuple('ABC'),tuple('GHI')], sequenceRules)
        results_trivial = pygsti.construction.manipulate_circuit_list([tuple('ABC'),tuple('GHI')], None) #special case


if __name__ == "__main__":
    unittest.main(verbosity=2)

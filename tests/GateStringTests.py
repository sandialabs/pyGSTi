import unittest
import GST
import numpy as np

class GateStringTestCase(unittest.TestCase):

    def setUp(self):
        pass

class TestGateStringMethods(GateStringTestCase):
    def test_simple(self):
        #The workhorse function is GST.create_gatestring_list, which executes its positional arguments within a nested
        # loop given by iterable keyword arguments.  That's a mouthful, so let's look at a few examples:
        As = [('a1',),('a2',)]
        Bs = [('b1',), ('b2',)]

        def rep2(x):
            return x+x

        list1 = GST.create_gatestring_list("a", a=As)
        list2 = GST.create_gatestring_list("a+b", a=As, b=Bs, order=['a','b'])
        list3 = GST.create_gatestring_list("a+b", a=As, b=Bs, order=['b','a'])
        list4 = GST.create_gatestring_list("R(a)+c", a=As, c=[('c',)], R=rep2, order=['a','c'])

        self.assertEqual(list1, GST.gatestring_list(As))
        self.assertEqual(list2, GST.gatestring_list([('a1','b1'),('a1','b2'),('a2','b1'),('a2','b2')]))
        self.assertEqual(list3, GST.gatestring_list([('a1','b1'),('a2','b1'),('a1','b2'),('a2','b2')]))
        self.assertEqual(list4, GST.gatestring_list([('a1','a1','c'),('a2','a2','c')]))

    def test_truncate_methods(self):
        self.assertEqual( GST.GateStringTools.repeat_and_truncate(('A','B','C'),5), ('A','B','C','A','B'))
        self.assertEqual( GST.GateStringTools.repeat_with_max_length(('A','B','C'),5), ('A','B','C'))
        self.assertEqual( GST.GateStringTools.repeat_count_with_max_length(('A','B','C'),5), 1)

    def test_fiducials_germs(self):
        fids  = GST.gatestring_list( [ ('Gf0',), ('Gf1',)    ] )
        germs = GST.gatestring_list( [ ('G0',), ('G1a','G1b')] )

        gateStrings1 = GST.create_gatestring_list("f0+germ*e+f1", f0=fids, f1=fids,
                                                germ=germs, e=2, order=["germ","f0","f1"])
        expected1 = ["Gf0(G0)^2Gf0",
                     "Gf0(G0)^2Gf1",
                     "Gf1(G0)^2Gf0",
                     "Gf1(G0)^2Gf1",
                     "Gf0(G1aG1b)^2Gf0",
                     "Gf0(G1aG1b)^2Gf1",
                     "Gf1(G1aG1b)^2Gf0",
                     "Gf1(G1aG1b)^2Gf1" ]
        self.assertEqual( map(str,gateStrings1), expected1 )

        
        gateStrings2 = GST.create_gatestring_list("f0+T(germ,N)+f1", f0=fids, f1=fids,
                                                germ=germs, N=3, T=GST.GateStringTools.repeat_and_truncate,
                                                order=["germ","f0","f1"])                                                
        expected2 = ["Gf0G0G0G0Gf0",
                     "Gf0G0G0G0Gf1",
                     "Gf1G0G0G0Gf0",
                     "Gf1G0G0G0Gf1",
                     "Gf0G1aG1bG1aGf0",
                     "Gf0G1aG1bG1aGf1",
                     "Gf1G1aG1bG1aGf0",
                     "Gf1G1aG1bG1aGf1" ]
        self.assertEqual( map(str,gateStrings2), expected2 )
        

        gateStrings3 = GST.create_gatestring_list("f0+T(germ,N)+f1", f0=fids, f1=fids,
                                                germ=germs, N=3, T=GST.GateStringTools.repeat_with_max_length,
                                                order=["germ","f0","f1"])
        expected3 = [ "Gf0(G0)^3Gf0",
                      "Gf0(G0)^3Gf1",
                      "Gf1(G0)^3Gf0",
                      "Gf1(G0)^3Gf1",
                      "Gf0(G1aG1b)Gf0",
                      "Gf0(G1aG1b)Gf1",
                      "Gf1(G1aG1b)Gf0",
                      "Gf1(G1aG1b)Gf1" ] 
        self.assertEqual( map(str,gateStrings3), expected3 )

    def test_string_compression(self):
        gs = GST.GateString(None, stringRepresentation="Gx^100")
        comp_gs = GST.GateStringTools.compress_gate_label_tuple(tuple(gs))
        exp_gs = GST.GateStringTools.expand_gate_label_tuple(comp_gs)
        self.assertEqual(tuple(gs), exp_gs)

    def test_repeat(self):
        gs = GST.GateString( ('Gx','Gx','Gy') )
        
        gs2 = GST.GateStringTools.repeat(gs, 2)
        self.assertEqual( gs2, GST.GateString( ('Gx','Gx','Gy','Gx','Gx','Gy') ))

        gs3 = GST.GateStringTools.repeat_with_max_length(gs, 7)
        self.assertEqual( gs3, GST.GateString( ('Gx','Gx','Gy','Gx','Gx','Gy') ))

        gs4 = GST.GateStringTools.repeat_and_truncate(gs, 4)
        self.assertEqual( gs4, GST.GateString( ('Gx','Gx','Gy','Gx') ))

        gs5 = GST.GateStringTools.repeat_remainder_for_truncation(gs, 4)
        self.assertEqual( gs5, GST.GateString( ('Gx',) ))

    def test_simplify(self):
        s = "{}Gx^1Gy{}Gz^1"
        self.assertEqual( GST.GateStringTools.simplify_str(s), "GxGyGz" )

        s = "{}Gx^1(Gy)^2{}Gz^1"
        self.assertEqual( GST.GateStringTools.simplify_str(s), "Gx(Gy)^2Gz" )

        s = "{}{}^1{}"
        self.assertEqual( GST.GateStringTools.simplify_str(s), "{}" )


    def test_lists(self):
        expected_allStrs = set( GST.gatestring_list( [(),('Gx',),('Gy',),('Gx','Gx'),('Gx','Gy'),('Gy','Gx'),('Gy','Gy')] ))
        allStrs = GST.GateStringTools.list_all_gatestrings( ('Gx','Gy'), 0,2 )
        self.assertEqual( set(allStrs), expected_allStrs)

        allStrs = list(GST.GateStringTools.gen_all_gatestrings( ('Gx','Gy'), 0,2 ))
        #self.assertEqual( set(allStrs), set([(),('Gx',),('Gy',),('Gx','Gx'),('Gx','Gy'),('Gy','Gx'),('Gy','Gy')]))
        #self.assertEqual( set(allStrs), set([(),('Gx',),('Gy',),('Gx','Gy'),('Gy','Gx')]))

        randStrs = GST.GateStringTools.list_random_gatestrings_onelen( ('Gx','Gy','Gz'), 2, 3)
        self.assertEqual( len(randStrs), 3 )
        self.assertTrue( all( [len(s)==2 for s in randStrs] ) )

        partialStrs = GST.GateStringTools.list_partial_strings( ('G1','G2','G3') )
        self.assertEqual( partialStrs, [ (), ('G1',), ('G1','G2'), ('G1','G2','G3') ] )


    def test_python_string_conversion(self):
        gs = GST.GateString(None, stringRepresentation="Gx^3Gy^2GxGz")

        pystr = GST.GateStringTools.gatestr_to_pythonstr( gs, ('Gx','Gy','Gz') )
        self.assertEqual( pystr, "AAABBAC" )
        
        gs2_tup = GST.GateStringTools.pythonstr_to_gatestr( pystr, ('Gx','Gy','Gz') )
        self.assertEqual( gs2_tup, tuple(gs) )
        
        
if __name__ == "__main__":
    unittest.main(verbosity=2)

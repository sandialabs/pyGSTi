import unittest
import GST
import numpy as np

class StdInputParserTestCase(unittest.TestCase):

    def setUp(self):
        pass

    def assertArraysAlmostEqual(self,a,b):
        self.assertAlmostEqual( np.linalg.norm(a-b), 0 )


class TestStdInputParser(StdInputParserTestCase):

    def test_strings(self):
        lkup = { '1': ('G1',),
                 '2': ('G1','G2'),
                 '3': ('G1','G2','G3','G4','G5','G6','G7','G8','G9','G10') }

        string_tests = [ ("{}", ()),
                         ("G1", ('G1',)),
                         ("G1G2G3", ('G1','G2','G3')),
                         ("G1(G2)G3", ('G1','G2','G3')),
                         ("G1(G2)^3G3", ('G1','G2','G2','G2','G3')),
                         ("G1(G2G3)^2", ('G1','G2','G3','G2','G3')),
                         ("G1*G2*G3", ('G1','G2','G3')),
                         ("G1 * G2", ('G1','G2')),
                         ("S[1]",('G1',)),
                         ("S[2]",('G1','G2')),
                         ("G1S[1]G3",('G1','G1','G3')),
                         ("S[3][0:4]",('G1', 'G2', 'G3', 'G4')),
                         ("G_my_xG_my_y", ('G_my_x', 'G_my_y')),
                         ("G_my_x*G_my_y", ('G_my_x', 'G_my_y')),
                         ("G_my_x G_my_y", ('G_my_x', 'G_my_y')) ]
        
        std = GST.StdInputParser.StdInputParser()

        #print "String Tests:"
        for s,expected in string_tests:
            #print "%s ==> " % s, result
            result = std.parse_gatestring(s, lookup=lkup)
            self.assertEqual(result, expected)


    def test_lines(self):
        dataline_tests = [ "G1G2G3           0.1 100", 
                           "G1 G2 G3         0.798 100",
                           "G1 (G2 G3)^2 G4  1.0 100" ]

        dictline_tests = [ "1  G1G2G3",
                           "MyFav (G1G2)^3" ]

        std = GST.StdInputParser.StdInputParser()

        self.assertEqual( std.parse_dataline(dataline_tests[0]), (('G1', 'G2', 'G3'), 'G1G2G3', [0.1, 100.0]))
        self.assertEqual( std.parse_dataline(dataline_tests[1]), (('G1', 'G2', 'G3'), 'G1 G2 G3', [0.798, 100.0]))
        self.assertEqual( std.parse_dataline(dataline_tests[2]), (('G1', 'G2', 'G3', 'G2', 'G3', 'G4'), 'G1 (G2 G3)^2 G4', [1.0, 100.0]))

        self.assertEqual( std.parse_dictline(dictline_tests[0]), ('1', ('G1', 'G2', 'G3'), 'G1G2G3'))
        self.assertEqual( std.parse_dictline(dictline_tests[1]), ('MyFav', ('G1', 'G2', 'G1', 'G2', 'G1', 'G2'), '(G1G2)^3'))

        #print "Dataline Tests:"
        #for dl in dataline_tests:
        #    print "%s ==> " % dl, std.parse_dataline(dl)
        #print " Dictline Tests:"
        #for dl in dictline_tests:
        #    print "%s ==> " % dl, std.parse_dictline(dl)


    def test_files(self):
        stringfile_test = \
"""#My string file
G1
G1G2
G1(G2G3)^2
""" 
        f = open("temp_test_files/sip_test.list","w")
        f.write(stringfile_test)
        f.close()


        dictfile_test = \
"""#My Dictionary file
# You can't use lookups within this file.
1 G1
2 G1G2
3 G1G2G3G4G5G6
MyFav1 G1G1G1
MyFav2 G2^3
this1  G3*G3*G3
thatOne G1 G2 * G3
""" 
        f = open("temp_test_files/sip_test.dict","w")
        f.write(dictfile_test)
        f.close()

        datafile_test = \
"""#My Data file
#Get string lookup data from the file test.dict
## Lookup = sip_test.dict
## Columns = plus frequency, count total
# OLD Columns = plus count, minus count

#empty string
{}            1.0 100

#simple sequences
G1G2          0.098  100
G2 G3         0.2    100
(G1)^4        0.1   1000

#using lookups
G1 S[1]       0.9999 100
S[MyFav1]G2   0.23   100
G1S[2]^2      0.5     20
S[3][0:4]     0.2      5
G1G2G3G4      0.2      5

#different ways to concatenate gates
G_my_xG_my_y  0.5 24.0
G_my_x*G_my_y 0.5 24.0
G_my_x G_my_y 0.5 24.0
""" 
        f = open("temp_test_files/sip_test.data","w")
        f.write(datafile_test)
        f.close()

        std = GST.StdInputParser.StdInputParser()

        import pprint
        pp = pprint.PrettyPrinter(indent=4)

        #print " Stringfile Test:"
        strlist = std.parse_stringfile("temp_test_files/sip_test.list")
        #print " ==> String list:"
        #pp.pprint(strlist)
    
        #print " Dictfile Test:"
        lkupDict = std.parse_dictfile("temp_test_files/sip_test.dict")
        #print " ==> Lookup dictionary:"
        #pp.pprint(lkupDict)

        #print " Datafile Test:"
        ds = std.parse_datafile("temp_test_files/sip_test.data")
        #print " ==> DataSet:\n", ds

        #TODO: add asserts


    def test_GateSetFile(self):

        gatesetfile_test = \
"""#My Gateset file

rho 
PauliVec
1.0/sqrt(2) 0 0 1.0/sqrt(2)

E 
PauliVec
1.0/sqrt(2) 0 0 -1.0/sqrt(2)

G1
PauliMx
1 0 0 0
0 1 0 0
0 0 0 -1
0 0 1 0

G2
PauliMx
1 0 0 0
0 0 0 1
0 0 1 0
0 -1 0 0
"""

        gatesetfile_test2 = \
"""#My Gateset file specified using non-Pauli format

rho_up
StateVec
1 0

rho_dn
DensityMx
0 0
0 1

E 
StateVec
0 1

#G1 = X(pi/2)
G1
UnitaryMx
 1/sqrt(2)   -1j/sqrt(2)
-1j/sqrt(2)   1/sqrt(2)

#G2 = Y(pi/2)
G2
UnitaryMxExp
0           -1j*pi/4.0
1j*pi/4.0  0

#G3 = X(pi)
G3
UnitaryMxExp
0          pi/2
pi/2      0


SPAMLABEL plus = rho_up E
"""

        f = open("temp_test_files/sip_test.gateset1","w")
        f.write(gatesetfile_test)
        f.close()
        
        f = open("temp_test_files/sip_test.gateset2","w")
        f.write(gatesetfile_test2)
        f.close()

        gs1 = GST.StdInputParser.readGateset("temp_test_files/sip_test.gateset1")
        gs2 = GST.StdInputParser.readGateset("temp_test_files/sip_test.gateset2")
        #print " ==> gateset1:\n", gs1
        #print " ==> gateset2:\n", gs2

        rotXPi   = GST.buildGate( [2],[('Q0',)], "X(pi,Q0)").matrix
        rotXPiOv2   = GST.buildGate( [2],[('Q0',)], "X(pi/2,Q0)").matrix        
        rotYPiOv2   = GST.buildGate( [2],[('Q0',)], "Y(pi/2,Q0)").matrix        


        self.assertArraysAlmostEqual(gs1['G1'],rotXPiOv2)
        self.assertArraysAlmostEqual(gs1['G2'],rotYPiOv2)
        self.assertArraysAlmostEqual(gs1.rhoVecs[0], 1/np.sqrt(2)*np.array([1,0,0,1]).reshape(-1,1) )
        self.assertArraysAlmostEqual(gs1.EVecs[0], 1/np.sqrt(2)*np.array([1,0,0,-1]).reshape(-1,1) )

        self.assertArraysAlmostEqual(gs2['G1'],rotXPiOv2)
        self.assertArraysAlmostEqual(gs2['G2'],rotYPiOv2)
        self.assertArraysAlmostEqual(gs2['G3'],rotXPi)
        self.assertArraysAlmostEqual(gs2.rhoVecs[0], 1/np.sqrt(2)*np.array([1,0,0,1]).reshape(-1,1) )
        self.assertArraysAlmostEqual(gs2.EVecs[0], 1/np.sqrt(2)*np.array([1,0,0,-1]).reshape(-1,1) )



            




if __name__ == "__main__":
    unittest.main(verbosity=2)

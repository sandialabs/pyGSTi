import unittest
import warnings
import pygsti
import numpy as np
import os
from ..testutils import BaseTestCase, compare_files, temp_files


class TestStdInputParser(BaseTestCase):

    def test_strings(self):
        lkup = { '1': ('G1',),
                 '2': ('G1','G2'),
                 '3': ('G1','G2','G3','G4','G5','G6','G7','G8','G9','G10'),
                 'G12': ('G1', 'G2'),
                 'S23': ('G2', 'G3')}

        string_tests = [ ("{}", ()),
                         ("{}^127", ()),
                         ("{}^0002", ()),
                         ("G1", ('G1',)),
                         ("G1G2G3", ('G1','G2','G3')),
                         ("G1(G2)G3", ('G1','G2','G3')),
                         ("G1(G2)^3G3", ('G1','G2','G2','G2','G3')),
                         ("G1(G2G3)^2", ('G1','G2','G3','G2','G3')),
                         ("G1*G2*G3", ('G1','G2','G3')),
                         ("G1^02", ('G1', 'G1')),
                         ("G1*((G2G3)^2G4G5)^2G7", ('G1', 'G2', 'G3', 'G2', 'G3', 'G4', 'G5', 'G2', 'G3', 'G2', 'G3', 'G4', 'G5', 'G7')),
                         ("G1(G2^2(G3G4)^2)^2", ('G1', 'G2', 'G2', 'G3', 'G4', 'G3', 'G4', 'G2', 'G2', 'G3', 'G4', 'G3', 'G4')),
                         ("G1*G2", ('G1','G2')),
                         #("S<1>",('G1',)),
                         #("S<2>",('G1','G2')),
                         #("G1S<2>^2G3", ('G1', 'G1', 'G2', 'G1', 'G2', 'G3')),
                         #("G1S<1>G3",('G1','G1','G3')),
                         #("S<3>[0:4]",('G1', 'G2', 'G3', 'G4')),
                         ("G_my_xG_my_y", ('G_my_x', 'G_my_y')),
                         ("G_my_x*G_my_y", ('G_my_x', 'G_my_y')),
                         ("GsG___", ('Gs', 'G___')),
                         #("S<2>G3", ('G1', 'G2', 'G3')),
                         #("S<G12>", ('G1', 'G2')),
                         #("S<S23>", ('G2', 'G3')),
                         ("G1G2", ('G1', 'G2')),
                         ("rho0*Gx", ('rho0','Gx')),
                         ("rho0*Gx*Mdefault", ('rho0','Gx','Mdefault'))]

        std = pygsti.io.StdInputParser()

        #print "String Tests:"
        for s,expected in string_tests:
            #print("%s ==> " % s, expected)
            result,line_labels = std.parse_circuit(s, lookup=lkup)
            self.assertEqual(line_labels, None)
            circuit_result = pygsti.obj.Circuit(result,line_labels="auto",expand_subcircuits=True)
              #use "auto" line labels since none are parsed.
            self.assertEqual(circuit_result.tup, expected)


        with self.assertRaises(ValueError):
            std.parse_circuit("FooBar")

        with self.assertRaises(ValueError):
            std.parse_circuit("G1G2^2^2")

        with self.assertRaises(ValueError):
            std.parse_circuit("(G1")

    def test_parse_circuit_with_time_and_args(self):
        std = pygsti.io.StdInputParser()
        
        cstr = "Gx;pi/1.2:0:2!1.0"
        firstLbl = std.parse_circuit(cstr)[0][0]
        self.assertEqual(firstLbl.time, 1.0)
        self.assertEqual(firstLbl.args, ('pi/1.2',))
        self.assertEqual(firstLbl.sslbls, (0, 2))
        self.assertEqual(firstLbl.name, 'Gx')
        self.assertEqual(tuple(firstLbl), ('Gx', 3, 'pi/1.2', 0, 2))

        cstr = "rho0!1.21{}"
        firstLbl = std.parse_circuit(cstr)[0][0]
        self.assertEqual(firstLbl.time, 1.21)
        self.assertEqual(firstLbl.args, ())
        self.assertEqual(firstLbl.sslbls, None)
        self.assertEqual(firstLbl.name, 'rho0')
        self.assertEqual(str(firstLbl), 'rho0!1.21')

        cstr = "{}M0!1.22"
        firstLbl = std.parse_circuit(cstr)[0][0]
        self.assertEqual(firstLbl.time, 1.22)
        self.assertEqual(firstLbl.args, ())
        self.assertEqual(firstLbl.sslbls, None)
        self.assertEqual(firstLbl.name, 'M0')
        self.assertEqual(str(firstLbl), 'M0!1.22')

    def test_string_exception(self):
        """Test lookup failure and Syntax error"""
        std = pygsti.io.StdInputParser()
        with self.assertRaises(ValueError):
            std.parse_circuit("G1 S[test]")
        with self.assertRaises(ValueError):
            std.parse_circuit("G1 SS")


    def test_lines(self):
        dataline_tests = [ "G1G2G3           0.1 100",
                           "G1*G2*G3         0.798 100",
                           "G1*(G2*G3)^2*G4  1.0 100" ]

        dictline_tests = [ "1  G1G2G3",
                           "MyFav (G1G2)^3" ]

        std = pygsti.io.StdInputParser()

        from pygsti.objects import Label as L
        from pygsti.objects import CircuitLabel as CL

        self.assertEqual( std.parse_dataline(dataline_tests[0],expected_counts=2), (['G1', 'G2', 'G3'], 'G1G2G3', None, [0.1, 100.0]))
        self.assertEqual( std.parse_dataline(dataline_tests[1],expected_counts=2), (['G1', 'G2', 'G3'], 'G1*G2*G3', None, [0.798, 100.0]))
        self.assertEqual( std.parse_dataline(dataline_tests[2],expected_counts=2), (['G1', CL('',('G2', 'G3'),None,2), 'G4'], 'G1*(G2*G3)^2*G4', None, [1.0, 100.0]))
        self.assertEqual( std.parse_dataline("G1G2G3 0.1 100 2.0", expected_counts=2),
                          (['G1', 'G2', 'G3'], 'G1G2G3', None, [0.1, 100.0])) #extra col ignored

        with self.assertRaises(ValueError):
            std.parse_dataline("G1G2G3  1.0", expected_counts=2) #too few cols == error
        with self.assertRaises(ValueError):
            std.parse_dataline("1.0 2.0") #just data cols (no circuit col!)


        self.assertEqual( std.parse_dictline(dictline_tests[0]), ('1', ['G1', 'G2', 'G3'], 'G1G2G3', None))
        self.assertEqual( std.parse_dictline(dictline_tests[1]), ('MyFav', [CL('',('G1', 'G2'),None,3),] , '(G1G2)^3', None))
          # OLD (before subcircuit parsing) the above result should have been: ('G1', 'G2', 'G1', 'G2', 'G1', 'G2')

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
        f = open(temp_files + "/sip_test.list","w")
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
thatOne G1G2*G3
"""
        f = open(temp_files + "/sip_test.dict","w")
        f.write(dictfile_test)
        f.close()

        datafile_test = \
"""#My Data file
#Get string lookup data from the file test.dict
## Lookup = sip_test.dict
## Columns = 0 frequency, count total
# OLD Columns = 0 count, 1 count

#empty string
{}            1.0 100

#simple sequences
G1G2          0.098  100
G2G3          0.2    100
(G1)^4        0.1   1000

#using lookups
#G1 S<1>       0.9999 100
#S<MyFav1>G2   0.23   100
#G1S<2>^2      0.5     20
#S<3>[0:4]     0.2      5
G1G2G3G4      0.2      5

#different ways to concatenate gates
G_my_xG_my_y  0.5 24.0
G_my_x*G_my_y 0.5 24.0
G_my_xG_my_y 0.5 24.0
"""
        f = open(temp_files + "/sip_test.data","w")
        f.write(datafile_test)
        f.close()

        datafile_test = \
"""#Data File without Header
{}            1.0 100
"""
        f = open(temp_files + "/sip_test2.data","w")
        f.write(datafile_test)
        f.close()

        datafile_test = \
"""#Data File with bad syntax
## Columns = 0 frequency, count total
{}            1.0 100
G1            0.0 100
FooBar        0.4 100
G3            0.2 100
"""
        f = open(temp_files + "/sip_test3.data","w")
        f.write(datafile_test)
        f.close()

        datafile_test = \
"""#Data File with zero counts
## Columns = 0 frequency, count total
{}            1.0 100
G1            0.0 100
G2            0   0
G3            0.2 100
"""
        f = open(temp_files + "/sip_test4.data","w")
        f.write(datafile_test)
        f.close()

        datafile_test = \
"""#Data File with bad columns
## Columns = 0 frequency, 1 frequency
{}            1.0 0.0
G1            0.0 1.0
G2            0   1.0
G3            0.2 0.8
"""
        f = open(temp_files + "/sip_test5.data","w")
        f.write(datafile_test)
        f.close()

        datafile_test = \
"""#Data File with bad frequency
## Columns = 1 frequency, count total
{}            1.0 100
G1            0.0 100
G2            3.4 100
G3            0.2 100
"""
        f = open(temp_files + "/sip_test6.data","w")
        f.write(datafile_test)
        f.close()

        datafile_test = \
"""#Data File with bad counts
## Columns = 0 count, count total
{}            30  100
G1            10  100
G2            0.2 100
G3            0.1 100
"""
        f = open(temp_files + "/sip_test7.data","w")
        f.write(datafile_test)
        f.close()

        datafile_test = \
"""#Data File with bad syntax
## Columns = 0 count, count total
{xx}            10  100
"""
        f = open(temp_files + "/sip_test8.data","w")
        f.write(datafile_test)
        f.close()



        multidatafile_test = \
"""#Multi Data File
## Lookup = sip_test.dict
## Columns = ds1 0 count, ds1 count total, ds2 0 count, ds2 count total
{}            30  100  20 200
G1            10  100  10 200
G2            20  100  5  200
G3            10  100  80 200
"""
        f = open(temp_files + "/sip_test.multidata","w")
        f.write(multidatafile_test)
        f.close()

        multidatafile_test = \
"""#Multi Data File with default cols
{}            30  100
G1            10  100
G2            20  100
G3            10  100
"""
        f = open(temp_files + "/sip_test2.multidata","w")
        f.write(multidatafile_test)
        f.close()

        multidatafile_test = \
"""#Multi Data File syntax error
{}            30  100
FooBar        10  100
G2            20  100
"""
        f = open(temp_files + "/sip_test3.multidata","w")
        f.write(multidatafile_test)
        f.close()

        multidatafile_test = \
"""#Multi Data File bad columns
## Columns = ds1 0 frequency, ds1 1 frequency, ds2 1 count, ds2 count total
{}            0.3  0.4  20 200
G1            0.1  0.5  10 200
G2            0.2  0.3  5  200
"""
        f = open(temp_files + "/sip_test4.multidata","w")
        f.write(multidatafile_test)
        f.close()

        multidatafile_test = \
"""#Multi Data File frequency out of range and count before frequency
## Columns = ds1 count total, ds1 0 frequency, ds2 0 count, ds2 count total
{}            100  0.3  20 200
G1            100  10   10 200
G2            100  0.2  5  200
"""
        f = open(temp_files + "/sip_test5.multidata","w")
        f.write(multidatafile_test)
        f.close()

        multidatafile_test = \
"""#Multi Data File count out of range
## Columns = ds1 0 count, ds1 count total, ds2 0 count, ds2 count total
{}            0.3  100  20 200
G1            0.1   100  10 200
G2            20  100  5  200
"""
        f = open(temp_files + "/sip_test6.multidata","w")
        f.write(multidatafile_test)
        f.close()

        multidatafile_test = \
"""#Multi Data File with bad syntax
## Columns = ds1 0 count, ds1 count total, ds2 0 count, ds2 count total
{xxx}         0.3  100  20 200
"""
        f = open(temp_files + "/sip_test7.multidata","w")
        f.write(multidatafile_test)
        f.close()


        std = pygsti.io.StdInputParser()

        import pprint
        pp = pprint.PrettyPrinter(indent=4)

        #print " Stringfile Test:"
        strlist = std.parse_stringfile(temp_files + "/sip_test.list")
        #print " ==> String list:"
        #pp.pprint(strlist)

        #print " Dictfile Test:"
        lkupDict = std.parse_dictfile(temp_files + "/sip_test.dict")
        #print " ==> Lookup dictionary:"
        #pp.pprint(lkupDict)

        #print " Datafile Test:"
        ds = std.parse_datafile(temp_files + "/sip_test.data")
        #print " ==> DataSet:\n", ds

        #test file with no header
        ds = std.parse_datafile(temp_files + "/sip_test2.data")

        #test file with bad data
        with self.assertRaises(ValueError):
            std.parse_datafile(temp_files + "/sip_test3.data")

        #test file with line(s) containing all zeros => ignore with warning
        self.assertWarns( std.parse_datafile, temp_files + "/sip_test4.data" )

        #test file with frequency columns but no count total
        with self.assertRaises(ValueError):
            std.parse_datafile(temp_files + "/sip_test5.data")

        #test file with out-of-range frequency
        #OLD with self.assertRaises(ValueError):
        self.assertWarns(std.parse_datafile, temp_files + "/sip_test6.data")
            
        #test file with out-of-range counts
        self.assertWarns(std.parse_datafile, temp_files + "/sip_test7.data")

        #test file with bad syntax
        with self.assertRaises(ValueError):
            std.parse_datafile(temp_files + "/sip_test8.data")



        #Multi-dataset tests
        mds = std.parse_multidatafile(temp_files + "/sip_test.multidata")

        #test file with no header
        mds = std.parse_multidatafile(temp_files + "/sip_test2.multidata")

        #test file with bad data
        with self.assertRaises(ValueError):
            std.parse_multidatafile(temp_files + "/sip_test3.multidata")

        #test file with frequency columns but no count total
        with self.assertRaises(ValueError):
            std.parse_multidatafile(temp_files + "/sip_test4.multidata")

        #test file with out-of-range frequency
        with self.assertRaises(ValueError):
            std.parse_multidatafile(temp_files + "/sip_test5.multidata")

        #test file with out-of-range counts
        with self.assertRaises(ValueError):
            std.parse_multidatafile(temp_files + "/sip_test6.multidata")

        #test file with bad syntax
        with self.assertRaises(ValueError):
            std.parse_multidatafile(temp_files + "/sip_test7.multidata")


        #TODO: add asserts


    def test_GateSetFile(self):

        gatesetfile_test = \
"""#My Model file

PREP: rho
LiouvilleVec
1.0/sqrt(2) 0 0 1.0/sqrt(2)

POVM: Mdefault

EFFECT: 0
LiouvilleVec
1.0/sqrt(2) 0 0 -1.0/sqrt(2)

END POVM

GATE: G1
LiouvilleMx
1 0 0 0
0 1 0 0
0 0 0 -1
0 0 1 0

GATE: G2
LiouvilleMx
1 0 0 0
0 0 0 1
0 0 1 0
0 -1 0 0

BASIS: pp 4
"""

        gatesetfile_test2 = \
"""#My Model file specified using non-Liouville format

PREP: rho_up
StateVec
1 0

PREP: rho_dn
DensityMx
0 0
0 1

POVM: Mdefault

EFFECT: 0
StateVec
1 0

END POVM

#G1 = X(pi/2)
GATE: G1
UnitaryMx
 1/sqrt(2)   -1j/sqrt(2)
-1j/sqrt(2)   1/sqrt(2)

#G2 = Y(pi/2)
GATE: G2
UnitaryMxExp
0           -1j*pi/4.0
1j*pi/4.0  0

#G3 = X(pi)
GATE: G3
UnitaryMxExp
0          pi/2
pi/2      0

BASIS: pp 4
GAUGEGROUP: Full
"""

        gatesetfile_test3 = \
"""#My Model file with bad StateVec size

PREP: rho_up
StateVec
1 0 0

"""

        gatesetfile_test4 = \
"""#My Model file with bad DensityMx size

PREP: rho_dn
DensityMx
0 0 0
0 1 0
0 0 1

BASIS: pp 4
"""

        gatesetfile_test5 = \
"""#My Model file with bad UnitaryMx size

#G1 = X(pi/2)
GATE: G1
UnitaryMx
 1/sqrt(2)   -1j/sqrt(2)

BASIS: pp 4
"""

        gatesetfile_test6 = \
"""#My Model file with bad UnitaryMxExp size

#G2 = Y(pi/2)
GATE: G2
UnitaryMxExp
0           -1j*pi/4.0 0.0
1j*pi/4.0  0           0.0

BASIS: pp 4
"""

        gatesetfile_test7 = \
"""#My Model file with bad format spec

GATE: G2
FooBar
0   1
1   0

BASIS: pp 4
"""

        gatesetfile_test8 = \
"""#My Model file specifying 2-Qubit gates using non-Lioville format

PREP: rho_up
DensityMx
1 0 0 0
0 0 0 0
0 0 0 0
0 0 0 0

POVM: Mdefault

EFFECT: 00
DensityMx
0 0 0 0
0 0 0 0
0 0 0 0
0 0 0 1

EFFECT: 11
DensityMx
1 0 0 0
0 0 0 0
0 0 0 0
0 0 0 0

END POVM

GATE: G1
UnitaryMx
 1/sqrt(2)   -1j/sqrt(2) 0 0
-1j/sqrt(2)   1/sqrt(2)  0 0
 0                0      1 0
 0                0      0 1

GATE: G2
UnitaryMxExp
0           -1j*pi/4.0 0 0
1j*pi/4.0  0           0 0
0          0           1 0
0          0           0 1

BASIS: pp 16
GAUGEGROUP: Full
"""


        gatesetfile_test9 = \
"""#My Model file with TP gates and no basis dim specified

TP-PREP: rho
LiouvilleVec
1.0/sqrt(2) 0 0 1.0/sqrt(2)

TP-POVM: Mdefault

EFFECT: 0
LiouvilleVec
1.0/sqrt(2) 0 0 1.0/sqrt(2)

EFFECT: 1
LiouvilleVec
1.0/sqrt(2) 0 0 -1.0/sqrt(2)

END POVM

TP-GATE: G1
LiouvilleMx
1 0 0 0
0 1 0 0
0 0 0 -1
0 0 1 0

CPTP-GATE: G2
LiouvilleMx
1 0 0 0
0 0 0 1
0 0 1 0
0 -1 0 0

BASIS: pp
GAUGEGROUP: TP
"""

        gatesetfile_test10 = \
"""#My Model file with instrument and POVM at end

PREP: rho
LiouvilleVec
1.0/sqrt(2) 0 0 1.0/sqrt(2)

GATE: G1
LiouvilleMx
1 0 0 0
0 1 0 0
0 0 0 -1
0 0 1 0

GATE: G2
LiouvilleMx
1 0 0 0
0 0 0 1
0 0 1 0
0 -1 0 0

Instrument: Iz

IGATE: minus
LiouvilleMx
      0.50000000               0               0     -0.50000000
               0               0               0               0
               0               0               0               0
     -0.50000000               0               0      0.50000000


IGATE: plus
LiouvilleMx
      0.50000000               0               0      0.50000000
               0               0               0               0
               0               0               0               0
      0.50000000               0               0      0.50000000


END Instrument

BASIS: pp 4
GAUGEGROUP: full

POVM: Mdefault

EFFECT: 0
LiouvilleVec
1.0/sqrt(2) 0 0 -1.0/sqrt(2)

END POVM
"""
        
        gatesetfile_test11 = \
"""# Invalid gauge group

GATE: G1
UnitaryMx
 1 0
 0 1

BASIS: pp 4
GAUGEGROUP: Foobar
"""


        gatesetfile_test12 = \
"""# Invalid item type

FOOBARGATE: G1
UnitaryMx
 1 0
 0 1

BASIS: pp 4
GAUGEGROUP: full
"""

        gatesetfile_test13 = \
"""# No basis dimension
BASIS: pp
"""



        f = open(temp_files + "/sip_test.model1","w")
        f.write(gatesetfile_test); f.close()

        f = open(temp_files + "/sip_test.model2","w")
        f.write(gatesetfile_test2); f.close()

        f = open(temp_files + "/sip_test.gateset3","w")
        f.write(gatesetfile_test3); f.close()

        f = open(temp_files + "/sip_test.gateset4","w")
        f.write(gatesetfile_test4); f.close()

        f = open(temp_files + "/sip_test.gateset5","w")
        f.write(gatesetfile_test5); f.close()

        f = open(temp_files + "/sip_test.gateset6","w")
        f.write(gatesetfile_test6); f.close()

        f = open(temp_files + "/sip_test.gateset7","w")
        f.write(gatesetfile_test7); f.close()

        f = open(temp_files + "/sip_test.gateset8","w")
        f.write(gatesetfile_test8); f.close()

        f = open(temp_files + "/sip_test.gateset9","w")
        f.write(gatesetfile_test9); f.close()

        f = open(temp_files + "/sip_test.gateset10","w")
        f.write(gatesetfile_test10); f.close()

        f = open(temp_files + "/sip_test.gateset11","w")
        f.write(gatesetfile_test11); f.close()

        f = open(temp_files + "/sip_test.gateset12","w")
        f.write(gatesetfile_test12); f.close()

        f = open(temp_files + "/sip_test.gateset13","w")
        f.write(gatesetfile_test13); f.close()

        gs1 = pygsti.io.read_model(temp_files + "/sip_test.model1")
        gs2 = pygsti.io.read_model(temp_files + "/sip_test.model2")

        with self.assertRaises(ValueError):
            pygsti.io.read_model(temp_files + "/sip_test.gateset3")
        with self.assertRaises(ValueError):
            pygsti.io.read_model(temp_files + "/sip_test.gateset4")
        with self.assertRaises(AssertionError):
            pygsti.io.read_model(temp_files + "/sip_test.gateset5")
        with self.assertRaises(ValueError):
            pygsti.io.read_model(temp_files + "/sip_test.gateset6")
        with self.assertRaises(ValueError):
            pygsti.io.read_model(temp_files + "/sip_test.gateset7")

        gs8 = pygsti.io.read_model(temp_files + "/sip_test.gateset8")
        #gs9 = pygsti.io.read_model(temp_files + "/sip_test.gateset9") # to test inferred basis dim, which isn't supported anymore (12/20/18)
        gs10 = pygsti.io.read_model(temp_files + "/sip_test.gateset10")

        self.assertWarns(pygsti.io.read_model, temp_files + "/sip_test.gateset11") #invalid gauge group = warning
        with self.assertRaises(ValueError):
            pygsti.io.read_model(temp_files + "/sip_test.gateset12") # invalid item type
        with self.assertRaises(ValueError):
            pygsti.io.read_model(temp_files + "/sip_test.gateset13") # cannot infer basis dim


        #print " ==> model1:\n", gs1
        #print " ==> model2:\n", gs2

        rotXPi   = pygsti.construction.build_operation( [(4,)],[('Q0',)], "X(pi,Q0)")
        rotXPiOv2   = pygsti.construction.build_operation( [(4,)],[('Q0',)], "X(pi/2,Q0)")
        rotYPiOv2   = pygsti.construction.build_operation( [(4,)],[('Q0',)], "Y(pi/2,Q0)")

        self.assertArraysAlmostEqual(gs1.operations['G1'],rotXPiOv2)
        self.assertArraysAlmostEqual(gs1.operations['G2'],rotYPiOv2)
        self.assertArraysAlmostEqual(gs1.preps['rho'], 1/np.sqrt(2)*np.array([1,0,0,1]).reshape(-1,1) )
        self.assertArraysAlmostEqual(gs1.povms['Mdefault']['0'], 1/np.sqrt(2)*np.array([1,0,0,-1]).reshape(-1,1) )

        self.assertArraysAlmostEqual(gs2.operations['G1'],rotXPiOv2)
        self.assertArraysAlmostEqual(gs2.operations['G2'],rotYPiOv2)
        self.assertArraysAlmostEqual(gs2.operations['G3'],rotXPi)
        self.assertArraysAlmostEqual(gs2.preps['rho_up'], 1/np.sqrt(2)*np.array([1,0,0,1]).reshape(-1,1) )
        self.assertArraysAlmostEqual(gs2.povms['Mdefault']['0'], 1/np.sqrt(2)*np.array([1,0,0,1]).reshape(-1,1) )

    def test_parse_complicated_circuits(self):
        #Test that a bunch of weird nested single layers can be parsed in,
        # and that this matches what is parsed in when a circuit object is
        # given to Circuit.__init__ and the parsing just checks for consistency:

        for expand in [False, True]:
            print("Expand = ",expand)
            for s in ["(Gx:0)Gy:1", "(Gx:0)^4Gy:1", "[Gx:0Gy:1]","[Gx:0Gy:1]^2","[Gx:0[Gz:2Gy:1]]Gz:0",
                      "[Gx:0(Gz:2Gy:1)]Gz:0", "[Gx:0[Gz:2Gy:1]^2]", "[Gx:0([Gz:2Gy:1]^2)]"]:
                print("FROM ",s,":")
                c = pygsti.obj.Circuit(None,stringrep=s, expand_subcircuits=expand)
                print(c)
                # c._print_labelinfo() #DEBUG - TODO: could check this structure as part of this test
                c2 = pygsti.obj.Circuit(c, stringrep=c.str, expand_subcircuits=expand)
                self.assertEqual(c, c2)
                print("\n\n")


if __name__ == "__main__":
    unittest.main(verbosity=2)

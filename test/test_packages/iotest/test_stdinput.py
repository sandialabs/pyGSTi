from pathlib import Path

import numpy as np

import pygsti.io.stdinput as stdin
from pygsti import io
from pygsti.modelpacks.legacy import std1Q_XYI as std
from pygsti.objects import Circuit, CircuitLabel
from . import IOBase, with_temp_path, with_temp_file


class StdInputBase:
    def setUp(self):
        self.std = stdin.StdInputParser()


class ParserTester(StdInputBase, IOBase):
    def test_parse_circuit(self):
        lkup = {'1': ('G1',),
                '2': ('G1', 'G2'),
                '3': ('G1', 'G2', 'G3', 'G4', 'G5', 'G6', 'G7', 'G8', 'G9', 'G10'),
                'G12': ('G1', 'G2'),
                'S23': ('G2', 'G3')}

        string_tests = [("{}", ()),
                        ("{}^127", ()),
                        ("{}^0002", ()),
                        ("G1", ('G1',)),
                        ("G1G2G3", ('G1', 'G2', 'G3')),
                        ("G1(G2)G3", ('G1', 'G2', 'G3')),
                        ("G1(G2)^3G3", ('G1', 'G2', 'G2', 'G2', 'G3')),
                        ("G1(G2G3)^2", ('G1', 'G2', 'G3', 'G2', 'G3')),
                        ("G1*G2*G3", ('G1', 'G2', 'G3')),
                        ("G1^02", ('G1', 'G1')),
                        ("G1*((G2G3)^2G4G5)^2G7", ('G1', 'G2', 'G3', 'G2', 'G3',
                                                   'G4', 'G5', 'G2', 'G3', 'G2', 'G3', 'G4', 'G5', 'G7')),
                        ("G1(G2^2(G3G4)^2)^2", ('G1', 'G2', 'G2', 'G3', 'G4', 'G3', 'G4', 'G2', 'G2', 'G3', 'G4', 'G3', 'G4')),
                        ("G1*G2", ('G1', 'G2')),
                        #("S<1>", ('G1',)),
                        #("S<2>", ('G1', 'G2')),
                        #("G1S<2>^2G3", ('G1', 'G1', 'G2', 'G1', 'G2', 'G3')),
                        #("G1S<1>G3", ('G1', 'G1', 'G3')),
                        #("S<3>[0:4]", ('G1', 'G2', 'G3', 'G4')),
                        ("G_my_xG_my_y", ('G_my_x', 'G_my_y')),
                        ("G_my_x*G_my_y", ('G_my_x', 'G_my_y')),
                        ("G_my_x*G_my_y", ('G_my_x', 'G_my_y')),
                        ("GsG___", ('Gs', 'G___')),
                        #("S < 2 >G3", ('G1', 'G2', 'G3')),
                        #("S<G12>", ('G1', 'G2')),
                        #("S<S23>", ('G2', 'G3')),
                        ("G1G2", ('G1', 'G2')),
                        ("rho0*Gx", ('rho0', 'Gx')),
                        ("rho0*Gx*Mdefault", ('rho0', 'Gx', 'Mdefault'))]

        for s, expected in string_tests:
            result, line_labels, occurrence_id = self.std.parse_circuit_raw(s, lookup=lkup)
            self.assertEqual(line_labels, None)
            circuit_result = Circuit(result, line_labels="auto", expand_subcircuits=True)
            #use "auto" line labels since none are parsed.
            self.assertEqual(circuit_result.tup, expected)

    def test_parse_circuit_raises_on_syntax_error(self):
        with self.assertRaises(ValueError):
            self.std.parse_circuit("FooBar")

        with self.assertRaises(ValueError):
            self.std.parse_circuit("G1G2^2^2")

        with self.assertRaises(ValueError):
            self.std.parse_circuit("(G1")

        with self.assertRaises(ValueError):
            self.std.parse_circuit("G1*S[test]")

        with self.assertRaises(ValueError):
            self.std.parse_circuit("G1*SS")

    def test_parse_dataline(self):
        dataline_tests = ["G1G2G3           0.1 100",
                          "G1*G2*G3         0.798 100",
                          "G1*(G2*G3)^2*G4  1.0 100"]

        self.assertEqual(
            self.std.parse_dataline(dataline_tests[0], expected_counts=2),
            (['G1', 'G2', 'G3'], [0.1, 100.0])
        )
        self.assertEqual(
            self.std.parse_dataline(dataline_tests[1], expected_counts=2),
            (['G1', 'G2', 'G3'], [0.798, 100.0])
        )
        self.assertEqual(
            self.std.parse_dataline(dataline_tests[2], expected_counts=2),
            (['G1', CircuitLabel('', ('G2', 'G3'), None, 2), 'G4'], [1.0, 100.0])
        )
        self.assertEqual(
            self.std.parse_dataline("G1G2G3 0.1 100 2.0", expected_counts=2),
            (['G1', 'G2', 'G3'], [0.1, 100.0])
        )  # extra col ignored

    def test_parse_dataline_raises_on_syntax_error(self):
        with self.assertRaises(ValueError):
            self.std.parse_dataline("G1G2G3  1.0", expected_counts=2)  # too few cols == error
        with self.assertRaises(ValueError):
            self.std.parse_dataline("1.0 2.0")  # just data cols (no circuit col!)

    def test_parse_dictline(self):
        dictline_tests = ["1  G1G2G3",
                          "MyFav (G1G2)^3"]
        self.assertEqual(
            self.std.parse_dictline(dictline_tests[0]),
            ('1', ('G1', 'G2', 'G3'), 'G1G2G3', None, None)
        )
        self.assertEqual(
            self.std.parse_dictline(dictline_tests[1]),
            ('MyFav', (CircuitLabel('', ('G1', 'G2'), None, 3),), '(G1G2)^3', None, None)
        )


class FileInputTester(StdInputBase, IOBase):
    def _write_dictfile(self, file_path):
        """Helper used for dict tests"""
        contents = """#My Dictionary file
# You can't use lookups within this file.
1 G1
2 G1G2
3 G1G2G3G4G5G6
MyFav1 G1G1G1
MyFav2 G2^3
this1  G3*G3*G3
thatOne G1G2*G3
"""
        with open(file_path, 'w') as f:
            f.write(contents)

    @with_temp_path
    def test_parse_dictfile(self, tmp_path):
        self._write_dictfile(tmp_path)
        lkupDict = self.std.parse_dictfile(tmp_path)
        # TODO assert correctness

    @with_temp_path
    def test_parse_datafile(self, tmp_path):
        # write lookup dict
        dict_path = str(Path(tmp_path).parent / "sip_test.dict")
        self._write_dictfile(dict_path)

        contents = """#My Data file
#Get string lookup data from the file test.dict
## Lookup = {dict_path}
## Columns = 0 count, 1 count
# OLD Columns = 0 frequency, count total

#empty string
{{}}            100 0

#simple sequences
G1G2          9.8  90.2
G2G3          20    80
(G1)^4        100   900

#using lookups
#G1 S<1>       100   0
#S<MyFav1>G2   23   77
#G1S<2>^2      10   10
#S<3>[0:4]     2     3
#G1G2G3G4      2     3

#different ways to concatenate gates
G_my_xG_my_y  0.5 24.0
G_my_x*G_my_y 0.5 24.0
G_my_xG_my_y 0.5 24.0
""".format(dict_path=dict_path)
        with open(tmp_path, 'w') as f:
            f.write(contents)

        ds = self.std.parse_datafile(tmp_path)
        # TODO assert correctness

    @with_temp_path
    def test_parse_multidatafile(self, tmp_path):
        # write lookup dict
        dict_path = str(Path(tmp_path).parent / "sip_test.dict")
        self._write_dictfile(dict_path)

        contents = """#Multi Data File
## Lookup = {dict_path}
## Columns = ds1 0 count, ds1 1 count, ds2 0 count, ds2 1 count
{{}}            30  70  20 180
G1            10  90  10 190
G2            20  80  5  195
G3            10  90  80 120
""".format(dict_path=dict_path)
        with open(tmp_path, 'w') as f:
            f.write(contents)

        mds = self.std.parse_multidatafile(tmp_path)
        # TODO assert correctness

    @with_temp_file("""#My string file
G1
G1G2
G1(G2G3)^2
""")
    def test_parse_stringfile(self, tmp_path):
        strlist = self.std.parse_stringfile(tmp_path)
        # TODO assert correctness

    @with_temp_file("""#Data File without Header
{}            100 0
""")
    def test_parse_datafile_no_header(self, tmp_path):
        ds = self.std.parse_datafile(tmp_path)
        # TODO assert correctness

    @with_temp_file("""#Data File with bad syntax
## Columns = 0 count, 1 count
{}            100  0
G1            0  100
FooBar        40  60
G3            20  80
""")
    def test_parse_datafile_raises_on_bad_data(self, tmp_path):
        with self.assertRaises(ValueError):
            self.std.parse_datafile(tmp_path)

    @with_temp_file("""#Data File with bad syntax
## Columns = 0 count, 1 count
{xx}            10  90
""")
    def test_parse_datafile_raises_on_syntax_error(self, tmp_path):
        with self.assertRaises(ValueError):
            self.std.parse_datafile(tmp_path)

    @with_temp_file("""#Data File with zero counts
## Columns = 0 count, 1 count
{}            100 0
G1            0.0 100
G2            0   0
G3            20 80
""")
    def test_parse_datafile_warns_on_missing_counts(self, tmp_path):
        self.assertWarns(self.std.parse_datafile, tmp_path)
        # TODO assert correctness

#    @with_temp_file("""#Data File with bad columns
### Columns = 0 frequency, 1 frequency
#{}            1.0 0.0
#G1            0.0 1.0
#G2            0   1.0
#G3            0.2 0.8
#""")
#    def test_parse_datafile_raises_on_bad_columns(self, tmp_path):
#        with self.assertRaises(ValueError):
#            self.std.parse_datafile(tmp_path)
#
#    @with_temp_file("""#Data File with bad frequency
### Columns = 1 frequency, count total
#{}            1.0 100
#G1            0.0 100
#G2            3.4 100
#G3            0.2 100
#""")
#    def test_parse_datafile_warns_on_frequency_out_of_range(self, tmp_path):
#        self.assertWarns(self.std.parse_datafile, tmp_path)
#        # TODO assert correctness
#
#    @with_temp_file("""#Data File with bad counts
### Columns = 0 count, count total
#{}            30  100
#G1            10  100
#G2            0.2 100
#G3            0.1 100
#""")
#    def test_parse_datafile_warns_on_counts_out_of_range(self, tmp_path):
#        self.assertWarns(self.std.parse_datafile, tmp_path)
#        # TODO assert correctness

    @with_temp_file("""#Multi Data File with default cols
{}            30  100
G1            10  100
G2            20  100
G3            10  100
""")
    def test_parse_multidatafile_no_header(self, tmp_path):
        mds = self.std.parse_multidatafile(tmp_path)
        # TODO assert correctness

    @with_temp_file("""#Multi Data File syntax error
{}            30  100
FooBar        10  100
G2            20  100
""")
    def test_parse_multidatafile_raises_on_bad_data(self, tmp_path):
        with self.assertRaises(ValueError):
            self.std.parse_multidatafile(tmp_path)

#    @with_temp_file("""#Multi Data File bad columns
### Columns = ds1 0 frequency, ds1 1 frequency, ds2 1 count, ds2 count total
#{}            0.3  0.4  20 200
#G1            0.1  0.5  10 200
#G2            0.2  0.3  5  200
#""")
#    def test_parse_multidatafile_raises_on_bad_columns(self, tmp_path):
#        with self.assertRaises(ValueError):
#            self.std.parse_multidatafile(tmp_path)
#
#    @with_temp_file("""#Multi Data File frequency out of range and count before frequency
### Columns = ds1 count total, ds1 0 frequency, ds2 0 count, ds2 count total
#{}            100  0.3  20 200
#G1            100  10   10 200
#G2            100  0.2  5  200
#""")
#    def test_parse_multidatafile_raises_on_frequency_out_of_range(self, tmp_path):
#        with self.assertRaises(ValueError):
#            self.std.parse_multidatafile(tmp_path)
#
#    @with_temp_file("""#Multi Data File count out of range
### Columns = ds1 0 count, ds1 count total, ds2 0 count, ds2 count total
#{}            0.3  100  20 200
#G1            0.1   100  10 200
#G2            20  100  5  200
#""")
#    def test_parse_multidatafile_raises_on_counts_out_of_range(self, tmp_path):
#        with self.assertRaises(ValueError):
#            self.std.parse_multidatafile(tmp_path)

    @with_temp_file("""#Multi Data File with bad syntax
## Columns = ds1 0 count, ds1 count total, ds2 0 count, ds2 count total
{xxx}         0.3  100  20 200
""")
    def test_parse_multidatafile_raises_on_syntax_error(self, tmp_path):
        with self.assertRaises(ValueError):
            self.std.parse_multidatafile(tmp_path)

    @with_temp_file("""#My Model file

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
""")
    def test_read_model(self, tmp_path):
        gs1 = stdin.parse_model(tmp_path)
        rotXPiOv2 = pygsti.models.modelconstruction._create_operation([(4,)], [('Q0',)], "X(pi/2,Q0)")
        rotYPiOv2 = pygsti.models.modelconstruction._create_operation([(4,)], [('Q0',)], "Y(pi/2,Q0)")

        self.assertArraysAlmostEqual(gs1.operations['G1'], rotXPiOv2)
        self.assertArraysAlmostEqual(gs1.operations['G2'], rotYPiOv2)
        self.assertArraysAlmostEqual(gs1.preps['rho'], 1 / np.sqrt(2) * np.array([1, 0, 0, 1]).reshape(-1, 1))
        self.assertArraysAlmostEqual(gs1.povms['Mdefault']['0'], 1 / np.sqrt(2)
                                     * np.array([1, 0, 0, -1]).reshape(-1, 1))

    @with_temp_file("""#My Model file specified using non-Liouville format

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
""")
    def test_read_model_non_liouville(self, tmp_path):
        gs2 = stdin.parse_model(tmp_path)
        rotXPi = pygsti.models.modelconstruction._create_operation([(4,)], [('Q0',)], "X(pi,Q0)")
        rotXPiOv2 = pygsti.models.modelconstruction._create_operation([(4,)], [('Q0',)], "X(pi/2,Q0)")
        rotYPiOv2 = pygsti.models.modelconstruction._create_operation([(4,)], [('Q0',)], "Y(pi/2,Q0)")
        self.assertArraysAlmostEqual(gs2.operations['G1'], rotXPiOv2)
        self.assertArraysAlmostEqual(gs2.operations['G2'], rotYPiOv2)
        self.assertArraysAlmostEqual(gs2.operations['G3'], rotXPi)
        self.assertArraysAlmostEqual(gs2.preps['rho_up'], 1 / np.sqrt(2) * np.array([1, 0, 0, 1]).reshape(-1, 1))
        self.assertArraysAlmostEqual(gs2.povms['Mdefault']['0'], 1 / np.sqrt(2) * np.array([1, 0, 0, 1]).reshape(-1, 1))

    @with_temp_file("""#My Model file specifying 2-Qubit gates using non-Lioville format

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
""")
    def test_read_model_2q(self, tmp_path):
        gs8 = stdin.parse_model(tmp_path)
        # TODO assert correctness

    @with_temp_file("""#My Model file with instrument and POVM at end

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
""")
    def test_read_model_with_instrument_and_povm(self, tmp_path):
        gs10 = stdin.parse_model(tmp_path)
        # TODO assert correctness

    @with_temp_file("""#My Model file with bad StateVec size

PREP: rho_up
StateVec
1 0 0

""")
    def test_read_model_raises_on_bad_statevec(self, tmp_path):
        with self.assertRaises(ValueError):
            stdin.parse_model(tmp_path)

    @with_temp_file("""#My Model file with bad DensityMx size

PREP: rho_dn
DensityMx
0 0 0
0 1 0
0 0 1

BASIS: pp 4
""")
    def test_read_model_raises_on_bad_densitymx(self, tmp_path):
        with self.assertRaises(ValueError):
            stdin.parse_model(tmp_path)

    @with_temp_file("""#My Model file with bad UnitaryMx size

#G1 = X(pi/2)
GATE: G1
UnitaryMx
 1/sqrt(2)   -1j/sqrt(2)

BASIS: pp 4
""")
    def test_read_model_raises_on_bad_unitarymx(self, tmp_path):
        with self.assertRaises(AssertionError):
            stdin.parse_model(tmp_path)

    @with_temp_file("""#My Model file with bad UnitaryMxExp size

#G2 = Y(pi/2)
GATE: G2
UnitaryMxExp
0           -1j*pi/4.0 0.0
1j*pi/4.0  0           0.0

BASIS: pp 4
""")
    def test_read_model_raises_on_bad_unitarymxexp(self, tmp_path):
        with self.assertRaises(ValueError):
            stdin.parse_model(tmp_path)

    @with_temp_file("""#My Model file with bad format spec

GATE: G2
FooBar
0   1
1   0

BASIS: pp 4
""")
    def test_read_model_raises_on_bad_format_spec(self, tmp_path):
        with self.assertRaises(ValueError):
            stdin.parse_model(tmp_path)

    @with_temp_file("""# Invalid gauge group

GATE: G1
UnitaryMx
 1 0
 0 1

BASIS: pp 4
GAUGEGROUP: Foobar
""")
    def test_read_model_warns_on_invalid_gauge_group(self, tmp_path):
        self.assertWarns(stdin.parse_model, tmp_path)

    @with_temp_file("""# Invalid item type

FOOBARGATE: G1
UnitaryMx
 1 0
 0 1

BASIS: pp 4
GAUGEGROUP: full
""")
    def test_read_model_raises_on_bad_item_type(self, tmp_path):
        with self.assertRaises(ValueError):
            stdin.parse_model(tmp_path)

    @with_temp_file("""# No basis dimension
BASIS: pp
""")
    def test_read_model_raises_on_no_basis_dimension(self, tmp_path):
        with self.assertRaises(ValueError):
            stdin.parse_model(tmp_path)

    @with_temp_path
    def _test_gateset_writeload(self, tmp_path, param):
        mdl = std.target_model()
        mdl.set_all_parameterizations(param)
        io.write_model(mdl, tmp_path)

        gs2 = stdin.parse_model(tmp_path)
        self.assertAlmostEqual(mdl.frobeniusdist(gs2), 0.0)
        for lbl in mdl.operations:
            self.assertEqual(type(mdl.operations[lbl]), type(gs2.operations[lbl]))
        for lbl in mdl.preps:
            self.assertEqual(type(mdl.preps[lbl]), type(gs2.preps[lbl]))
        for lbl in mdl.povms:
            self.assertEqual(type(mdl.povms[lbl]), type(gs2.povms[lbl]))
        for lbl in mdl.instruments:
            self.assertEqual(type(mdl.instruments[lbl]), type(gs2.instruments[lbl]))

    def test_read_model_full_param(self):
        self._test_gateset_writeload('full')

    def test_read_model_TP_param(self):
        self._test_gateset_writeload('TP')

    def test_read_model_CPTP_param(self):
        self._test_gateset_writeload('CPTP')

    def test_read_model_static_param(self):
        self._test_gateset_writeload('static')

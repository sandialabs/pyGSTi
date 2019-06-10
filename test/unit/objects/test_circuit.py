import copy

from ..util import BaseCase

from pygsti.baseobjs.label import Label
from pygsti.objects import circuit


class CircuitTester(BaseCase):
    def test_to_pythonstr(self):
        mdl = circuit.Circuit(None, stringrep="Gx^3Gy^2GxGz")

        op_labels = (Label('Gx'), Label('Gy'), Label('Gz'))
        pystr = mdl.to_pythonstr(op_labels)
        self.assertEqual(pystr, "AAABBAC")

        gs2_tup = circuit.Circuit.from_pythonstr(pystr, op_labels)
        self.assertEqual(gs2_tup, tuple(mdl))

    def test_raise_on_bad_construction(self):
        with self.assertRaises(ValueError):
            circuit.Circuit(('Gx', 'Gx'), stringrep="GxGy", check=True)  # mismatch
        with self.assertRaises(ValueError):
            circuit.Circuit(None)
        with self.assertRaises(ValueError):
            circuit.Circuit(('foobar',), stringrep="foobar", check=True)  # lexer illegal character


class CircuitOperationTester(BaseCase):
    def setUp(self):
        self.s1 = circuit.Circuit(('Gx', 'Gx'), stringrep="Gx^2")
        self.s2 = circuit.Circuit(self.s1, stringrep="Gx^2")

    def test_eq(self):
        self.assertEqual(self.s1, ('Gx', 'Gx'))
        self.assertEqual(self.s2, ('Gx', 'Gx'))
        self.assertTrue(self.s1 == self.s2)

    def test_add(self):
        s3 = self.s1 + self.s2
        self.assertEqual(s3, ('Gx', 'Gx', 'Gx', 'Gx'))

    def test_pow(self):
        s4 = self.s1**3
        self.assertEqual(s4, ('Gx', 'Gx', 'Gx', 'Gx', 'Gx', 'Gx'))

    def test_copy(self):
        s5 = self.s1
        s6 = copy.copy(self.s1)
        s7 = copy.deepcopy(self.s1)

        self.assertEqual(self.s1, s5)
        self.assertEqual(self.s1, s6)
        self.assertEqual(self.s1, s7)

    def test_lt_gt(self):
        self.assertFalse(self.s1 < self.s2)
        self.assertFalse(self.s1 > self.s2)

        s3 = self.s1 + self.s2
        self.assertTrue(self.s1 < s3)
        self.assertTrue(s3 > self.s1)

    def test_read_only(self):
        with self.assertRaises(AssertionError):
            self.s1[0] = 'Gx'  # cannot set items - like a tuple they're read-only

    def test_raise_on_add_non_circuit(self):
        with self.assertRaises(ValueError):
            self.s1 + ("Gx",)  # can't add non-Circuit to circuit


class CompressedCircuitTester(BaseCase):
    def test_compress_op_label(self):
        mdl = circuit.Circuit(None, stringrep="Gx^100")
        comp_gs = circuit.CompressedCircuit.compress_op_label_tuple(tuple(mdl))
        # TODO assert correctness

        exp_gs = circuit.CompressedCircuit.expand_op_label_tuple(comp_gs)
        self.assertEqual(tuple(mdl), exp_gs)

    def test_compress_expand(self):
        s1 = circuit.Circuit(('Gx', 'Gx'), stringrep="Gx^2")
        c1 = circuit.CompressedCircuit(s1)
        s1_expanded = c1.expand()
        self.assertEqual(s1, s1_expanded)

    def test_raise_on_construct_from_non_circuit(self):
        with self.assertRaises(ValueError):
            circuit.CompressedCircuit(('Gx',))  # can only create from Circuits

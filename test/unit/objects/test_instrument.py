import numpy as np

from pygsti.modelmembers import instruments as inst
from pygsti.modelpacks.legacy import std1Q_XYI as std
from pygsti.models.gaugegroup import FullGaugeGroupElement
from ..util import BaseCase


class InstrumentMethodBase(object):
    def test_num_elements(self):
        self.assertEqual(self.instrument.num_elements, self.n_elements)

    def test_copy(self):
        inst_copy = self.instrument.copy()
        # TODO assert correctness

    def test_to_string(self):
        inst_str = str(self.instrument)
        # TODO assert correctness

    def test_transform(self):
        T = FullGaugeGroupElement(
            np.array([[1, 0, 0, 0],
                      [0, 1, 0, 0],
                      [0, 0, 0, 1],
                      [0, 0, 1, 0]], 'd'))
        self.instrument.transform_inplace(T)
        # TODO assert correctness

    def test_simplify_operations(self):
        gates = self.instrument.simplify_operations(prefix="ABC")
        # TODO assert correctness

    def test_constructor_raises_on_non_none_param_conflict(self):
        with self.assertRaises(AssertionError):
            self.constructor(["Non-none-matrices"], 'default', None, ["Non-none-items"])  # can't both be non-None

    def test_constructor_raises_on_bad_op_matrices_type(self):
        with self.assertRaises(ValueError):
            self.constructor("foobar")  # op_matrices must be a list or dict

    def test_convert_raises_on_unknown_basis(self):
        with self.assertRaises(ValueError):
            inst.convert(self.instrument, "foobar", self.model.basis)


class InstrumentInstanceBase(object):
    def setUp(self):
        # Initialize standard target model for instruments
        # XXX can instruments be tested independently of a model?  EGN: yes, I was just lazy; but they should also be tested within a model.
        self.n_elements = 32

        self.model = std.target_model()
        E = self.model.povms['Mdefault']['0']
        Erem = self.model.povms['Mdefault']['1']
        self.Gmz_plus = np.dot(E, E.T)
        self.Gmz_minus = np.dot(Erem, Erem.T)
        # XXX is this used?
        self.povm_ident = self.model.povms['Mdefault']['0'] + self.model.povms['Mdefault']['1']
        self.instrument = self.constructor({'plus': self.Gmz_plus, 'minus': self.Gmz_minus})
        self.model.instruments['Iz'] = self.instrument
        super(InstrumentInstanceBase, self).setUp()


class InstrumentInstanceTester(InstrumentMethodBase, InstrumentInstanceBase, BaseCase):
    constructor = inst.Instrument


class TPInstrumentInstanceTester(InstrumentMethodBase, InstrumentInstanceBase, BaseCase):
    constructor = inst.TPInstrument

    def test_raise_on_modify(self):
        with self.assertRaises(ValueError):
            self.instrument['plus'] = None  # can't set value of a TP Instrument element

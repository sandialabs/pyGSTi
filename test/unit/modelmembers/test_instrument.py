import numpy as np

import pygsti.modelmembers.operations as op
from pygsti.modelmembers import instruments as inst
from pygsti.modelmembers.instruments import cptp_instrument
from pygsti.modelmembers.instruments import Instrument, TPInstrument
from pygsti.modelmembers.operations import RootConjOperator, SummedOperator
from pygsti.modelmembers import povms as pv
from pygsti.modelpacks.legacy import std1Q_XYI as std
from pygsti.models.gaugegroup import FullGaugeGroupElement
from ..util import BaseCase
from .test_operation import ImmutableDenseOpBase


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
            self.constructor(["Non-none-matrices"], 'default', None, False, ["Non-none-items"])  # can't both be non-None

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


class CPTPInstrumentTester(BaseCase):
    """Tests for cptp_instrument and the CPTR operation primitives."""

    def setUp(self):
        model = std.target_model()
        E = model.povms['Mdefault']['0']
        Erem = model.povms['Mdefault']['1']
        self.Gmz_plus  = np.dot(E, E.T)
        self.Gmz_minus = np.dot(Erem, Erem.T)
        self.basis = model.basis
        self.op_arrays = {'plus': self.Gmz_plus, 'minus': self.Gmz_minus}
        self.instrument = cptp_instrument(self.op_arrays, self.basis)

    # --- cptp_instrument factory tests ---

    def test_construction_returns_instrument(self):
        self.assertIsInstance(self.instrument, Instrument)

    def test_construction_keys(self):
        self.assertSetEqual(set(self.instrument.keys()), {'plus', 'minus'})

    def test_to_from_vector_roundtrip(self):
        v0 = self.instrument.to_vector()
        self.assertEqual(len(v0), self.instrument.num_params)
        # Perturb and restore; dense output should change then return.
        v_perturbed = v0 + 1e-3 * np.random.default_rng(0).standard_normal(len(v0))
        self.instrument.from_vector(v_perturbed)
        v1 = self.instrument.to_vector()
        self.assertFalse(np.allclose(v0, v1))
        self.instrument.from_vector(v0)
        v2 = self.instrument.to_vector()
        np.testing.assert_allclose(v0, v2, atol=1e-12)

    def test_convert_to_cptplnd(self):
        plain = inst.Instrument({'plus': self.Gmz_plus, 'minus': self.Gmz_minus})
        converted = inst.convert(plain, 'CPTPLND', self.basis)
        self.assertIsInstance(converted, Instrument)
        self.assertSetEqual(set(converted.keys()), {'plus', 'minus'})

    def test_member_dense_shapes(self):
        dim = self.basis.dim
        for op in self.instrument.values():
            mx = op.to_dense('HilbertSchmidt')
            self.assertEqual(mx.shape, (dim, dim))

    def test_sum_of_members_is_cptp_like(self):
        # The sum of all instrument members' superoperators should equal the
        # superoperator of the total channel (which is CPTP), i.e. the first
        # row of the sum must be [1, 0, 0, 0] in the pp basis.
        total = sum(op.to_dense('HilbertSchmidt') for op in self.instrument.values())
        # First row encodes trace preservation: tr(rho) is preserved.
        np.testing.assert_allclose(total[0, 0], 1.0, atol=1e-6)
        np.testing.assert_allclose(total[0, 1:], 0.0, atol=1e-6)

    # --- RootConjOperator unit tests ---

    def test_root_conj_operator_to_dense_shape(self):
        # Build a real StaticPOVMEffect from the plus POVM effect
        effect = pv.StaticPOVMEffect(self.Gmz_plus[:, 0])
        rcop = RootConjOperator(effect, self.basis)
        mx = rcop.to_dense('HilbertSchmidt')
        dim = self.basis.dim
        self.assertEqual(mx.shape, (dim, dim))

    def test_root_conj_operator_num_params(self):
        effect = pv.StaticPOVMEffect(self.Gmz_plus[:, 0])
        rcop = RootConjOperator(effect, self.basis)
        # StaticPOVMEffect has zero parameters; RootConjOperator inherits them.
        self.assertEqual(rcop.num_params, 0)

    def test_root_conj_operator_has_nonzero_hessian(self):
        effect = pv.StaticPOVMEffect(self.Gmz_plus[:, 0])
        rcop = RootConjOperator(effect, self.basis)
        self.assertTrue(rcop.has_nonzero_hessian())

    # --- SummedOperator unit tests ---

    def test_summed_operator_to_dense_equals_sum(self):
        e1 = pv.StaticPOVMEffect(self.Gmz_plus[:, 0])
        e2 = pv.StaticPOVMEffect(self.Gmz_minus[:, 0])
        r1 = RootConjOperator(e1, self.basis)
        r2 = RootConjOperator(e2, self.basis)
        sop = SummedOperator([r1, r2], self.basis)
        expected = r1.to_dense() + r2.to_dense()
        np.testing.assert_allclose(sop.to_dense(), expected, atol=1e-12)

    def test_summed_operator_num_params(self):
        e1 = pv.StaticPOVMEffect(self.Gmz_plus[:, 0])
        e2 = pv.StaticPOVMEffect(self.Gmz_minus[:, 0])
        r1 = RootConjOperator(e1, self.basis)
        r2 = RootConjOperator(e2, self.basis)
        sop = SummedOperator([r1, r2], self.basis)
        self.assertEqual(sop.num_params, 0)

    def test_summed_operator_deriv_raises(self):
        e1 = pv.StaticPOVMEffect(self.Gmz_plus[:, 0])
        r1 = RootConjOperator(e1, self.basis)
        sop = SummedOperator([r1], self.basis)
        with self.assertRaises(NotImplementedError):
            sop.deriv_wrt_params()


class TPInstrumentOpTester(ImmutableDenseOpBase, BaseCase):
    n_params = 28

    @staticmethod
    def build_gate():
        Gmz_plus = np.array([[0.5, 0, 0, 0.5],
                             [0, 0, 0, 0],
                             [0, 0, 0, 0],
                             [0.5, 0, 0, 0.5]])
        Gmz_minus = np.array([[0.5, 0, 0, -0.5],
                              [0, 0, 0, 0],
                              [0, 0, 0, 0],
                              [-0.5, 0, 0, 0.5]])
        evotype = 'default'
        instrument = TPInstrument({
            'plus':  op.FullArbitraryOp(Gmz_plus, 'pp', evotype),
            'minus': op.FullArbitraryOp(Gmz_minus, 'pp', evotype)
        })
        return instrument['plus']

    def test_vector_conversion(self):
        self.gate.to_vector()  # now to_vector is allowed

    def test_deriv_wrt_params_shape(self):
        super(TPInstrumentOpTester, self).test_deriv_wrt_params()
        deriv = self.gate.deriv_wrt_params([0])
        self.assertEqual(deriv.shape[1], 1)

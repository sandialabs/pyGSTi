import numpy as np

import pygsti
import pygsti.modelmembers.operations as op
from pygsti.modelmembers import instruments as inst
from pygsti.modelmembers.instruments import Instrument, TPInstrument, convert
from pygsti.modelmembers.operations import RootConjOperator, SummedOperator
from pygsti.modelmembers import povms as pv
from pygsti.modelpacks.legacy import std1Q_XYI as std
from pygsti.models.gaugegroup import FullGaugeGroupElement
from ..util import BaseCase
from .test_operation import ImmutableDenseOpBase


def z_measurement_projectors(model):
    """Ideal Z-measurement effect projectors built from `model`'s Mdefault POVM: each returned
    array is the outer product of a POVM effect with itself, i.e. a rank-1 Kraus operator for
    that outcome (used across this file, and by test_forwardsim.py's instrument fixtures)."""
    E0 = np.asarray(model.povms['Mdefault']['0'].to_dense()).reshape(-1)
    E1 = np.asarray(model.povms['Mdefault']['1'].to_dense()).reshape(-1)
    return E0, E1, np.outer(E0, E0), np.outer(E1, E1)


def _noisy_z_op_arrays(gamma=0.05, theta=np.deg2rad(10), eps=0.02):
    """Dense CPTR superops for a noisy Z measurement whose members have Kraus rank 2.

    Amplitude damping composed with a weak Z measurement gives each "true" outcome
    Kraus rank 2; a classical assignment-error mix keeps every member completely
    positive and the pair jointly trace-preserving.
    """
    A0 = np.array([[1, 0], [0, np.sqrt(1 - gamma)]], dtype=complex)
    A1 = np.array([[0, np.sqrt(gamma)], [0, 0]], dtype=complex)
    M0 = np.array([[np.cos(theta), 0], [0, np.sin(theta)]], dtype=complex)
    M1 = np.array([[np.sin(theta), 0], [0, np.cos(theta)]], dtype=complex)
    S0 = op.FullArbitraryOp.from_kraus_operators([M0 @ A0, M0 @ A1], 'pp').to_dense()
    S1 = op.FullArbitraryOp.from_kraus_operators([M1 @ A0, M1 @ A1], 'pp').to_dense()
    R0 = (1 - eps) * S0 + eps * S1
    R1 = eps * S0 + (1 - eps) * S1
    return {'p0': R0, 'p1': R1}


def _weak_z_op_arrays(off_eigenvalue):
    """Dense CPTR superops for a weak Z measurement with diagonal effects.

    The effects are ``diag(1 - off_eigenvalue, off_eigenvalue)`` and its complement,
    so `off_eigenvalue` controls how close each effect is to a rank-1 projector
    (and whether the small eigenvalue falls below the kernel ``trunc_tol``).
    """
    s = np.sqrt(off_eigenvalue)
    c = np.sqrt(1 - off_eigenvalue)
    K0 = np.diag([c, s]).astype(complex)
    K1 = np.diag([s, c]).astype(complex)
    I0 = op.FullArbitraryOp.from_kraus_operators([K0], 'pp').to_dense()
    I1 = op.FullArbitraryOp.from_kraus_operators([K1], 'pp').to_dense()
    return {'p0': I0, 'p1': I1}


class InstrumentTestBase(BaseCase):
    """Shared setUp and test methods for Instrument and TPInstrument.

    Subclasses must define a class attribute ``constructor`` pointing to the
    instrument class under test.
    """
    __test__ = False
    constructor: type

    def setUp(self):
        self.n_elements = 32
        self.model = std.target_model()
        _, _, self.Gmz_plus, self.Gmz_minus = z_measurement_projectors(self.model)
        self.instrument: Instrument = self.constructor({'plus': self.Gmz_plus, 'minus': self.Gmz_minus})
        self.model.instruments['Iz'] = self.instrument

    def test_num_elements(self):
        self.assertEqual(self.instrument.num_elements, self.n_elements)

    def test_copy(self):
        inst_copy = self.instrument.copy()
        self.assertIsInstance(inst_copy, type(self.instrument))
        self.assertEqual(list(inst_copy.keys()), list(self.instrument.keys()))
        for key in self.instrument.keys():
            actual = inst_copy[key].to_dense()
            expected = self.instrument[key].to_dense()
            self.assertArraysEqual(actual, expected)

    def test_to_string(self):
        inst_str = str(self.instrument)
        self.assertIsInstance(inst_str, str)
        self.assertIn("Instrument with elements:", inst_str)
        for key in self.instrument.keys():
            self.assertIn(str(key), inst_str)

    def test_transform(self):
        T = FullGaugeGroupElement(
            np.array([[1, 0, 0, 0],
                      [0, 1, 0, 0],
                      [0, 0, 0, 1],
                      [0, 0, 1, 0]], 'd'))
        T_mx = T.transform_matrix
        T_inv = T.transform_matrix_inverse
        originals = {k: v.to_dense().copy() for k, v in self.instrument.items()}
        self.instrument.transform_inplace(T)
        for key, E_orig in originals.items():
            expected = T_mx @ E_orig @ T_inv
            self.assertArraysAlmostEqual(self.instrument[key].to_dense(), expected)

    def test_simplify_operations(self):
        gates = self.instrument.simplify_operations(prefix="ABC")
        self.assertEqual(len(gates), len(self.instrument))
        expected_keys = ["ABC_" + k for k in self.instrument.keys()]
        self.assertEqual(list(gates.keys()), expected_keys)
        for gate, orig in zip(gates.values(), self.instrument.values()):
            self.assertIs(gate, orig)


class InstrumentInstanceTester(InstrumentTestBase):
    __test__ = True
    constructor = inst.Instrument


class TPInstrumentInstanceTester(InstrumentTestBase):
    __test__ = True
    constructor = inst.TPInstrument

    def test_raise_on_modify(self):
        with self.assertRaises(ValueError):
            self.instrument['plus'] = None


class FromCptrSuperopsTester(BaseCase):
    """Tests for Instrument.from_cptr_superops and the CPTR operation primitives."""

    def setUp(self):
        model = std.target_model()
        self.E0, self.E1, self.Gmz_plus, self.Gmz_minus = z_measurement_projectors(model)
        self.basis = model.basis
        # The default fixture is an ideal projective Z measurement: each effect is a
        # rank-1 projector (singular E_k), which exercises the gate kernel completion.
        self.op_arrays = {'plus': self.Gmz_plus, 'minus': self.Gmz_minus}
        self.instrument = Instrument.from_cptr_superops(self.op_arrays, self.basis)

    def _members(self, instrument):
        return {k: v.to_dense('HilbertSchmidt') for k, v in instrument.items()}

    def _assert_cp(self, member_superop, atol=1e-8):
        viol = pygsti.tools.sum_of_negative_choi_eigenvalues_gate(member_superop, 'pp')
        self.assertGreaterEqual(viol, -atol)

    # --- factory / structure tests ---

    def test_construction_returns_instrument(self):
        self.assertIsInstance(self.instrument, Instrument)

    def test_construction_keys(self):
        self.assertSetEqual(set(self.instrument.keys()), {'plus', 'minus'})

    def test_member_dense_shapes(self):
        dim = self.basis.dim
        for member in self.instrument.values():
            mx = member.to_dense('HilbertSchmidt')
            self.assertEqual(mx.shape, (dim, dim))

    def test_each_member_is_single_composed_op(self):
        # Every member is one ComposedOp([RootConjOperator, gate]) -- never a
        # SummedOperator, regardless of Kraus rank.
        for member in self.instrument.values():
            self.assertIsInstance(member, op.ComposedOp)
            self.assertNotIsInstance(member, SummedOperator)
            self.assertIsInstance(member.factorops[0], RootConjOperator)

    def test_shared_povm_has_n_effects(self):
        # Regression against the old Kraus-polar proliferation (which produced one
        # effect per Kraus term): there must be exactly one effect per outcome.
        effects = {id(member.factorops[0].submembers()[0]) for member in self.instrument.values()}
        self.assertEqual(len(effects), len(self.instrument))

    def test_to_from_vector_roundtrip(self):
        v0 = self.instrument.to_vector()
        self.assertEqual(len(v0), self.instrument.num_params)
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

    def test_convert_to_glnd(self):
        plain = inst.Instrument({'plus': self.Gmz_plus, 'minus': self.Gmz_minus})
        converted = inst.convert(plain, 'GLND', self.basis)
        self.assertIsInstance(converted, Instrument)

    def test_sum_of_members_is_tp(self):
        # The sum of all members' superoperators is the (CPTP) total channel; in the
        # pp basis the TP constraint means first rows sum to [1, 0, 0, 0].
        total = sum(self._members(self.instrument).values())
        np.testing.assert_allclose(total[0, 0], 1.0, atol=1e-6)
        np.testing.assert_allclose(total[0, 1:], 0.0, atol=1e-6)

    # --- numerical correctness: effect-then-CPTP-gate reconstruction ---

    def test_ideal_projector_singular_effect(self):
        # Singular (rank-1) effects: the recovered gate is the canonical CPTP
        # completion G_k = Q_k + P_k; the members reproduce the ideal projectors and
        # are completely positive.
        members = self._members(self.instrument)
        self.assertArraysAlmostEqual(members['plus'], self.Gmz_plus)
        self.assertArraysAlmostEqual(members['minus'], self.Gmz_minus)
        for mx in members.values():
            self._assert_cp(mx)

    def test_rank2_members_stay_single_composed_op(self):
        # A Kraus-rank-2 member still yields a single ComposedOp (no SummedOperator)
        # and a single shared effect per outcome, while reproducing the input.
        arrays = _noisy_z_op_arrays()
        instrument = Instrument.from_cptr_superops(arrays, self.basis)
        members = self._members(instrument)
        for k, member in instrument.items():
            self.assertIsInstance(member, op.ComposedOp)
            self.assertNotIsInstance(member, SummedOperator)
            self.assertIsInstance(member.factorops[0], RootConjOperator)
            self.assertArraysAlmostEqual(members[k], arrays[k])
            self._assert_cp(members[k])
        effects = {id(member.factorops[0].submembers()[0]) for member in instrument.values()}
        self.assertEqual(len(effects), len(instrument))

    def test_perturbed_projector_kernel_threshold(self):
        # A small effect eigenvalue *above* trunc_tol is kept (exact reconstruction);
        # one *below* trunc_tol is treated as kernel and completed (reconstruction
        # then degrades only in that barely-measured direction).  Both stay CP and TP.
        for off, recon_atol in [(1e-5, 1e-6), (1e-9, 1e-3)]:
            arrays = _weak_z_op_arrays(off)
            instrument = Instrument.from_cptr_superops(arrays, self.basis)
            members = self._members(instrument)
            total = sum(members.values())
            np.testing.assert_allclose(total[0], [1, 0, 0, 0], atol=1e-9)
            for k, mx in members.items():
                self._assert_cp(mx)
                np.testing.assert_allclose(mx, arrays[k], atol=recon_atol)

    def test_raises_when_members_not_tp(self):
        # Effects E_k = I_k^dagger(I) that do not sum to I -> not a TP channel.
        with self.assertRaises(ValueError):
            Instrument.from_cptr_superops({'plus': self.Gmz_plus, 'minus': self.Gmz_plus}, self.basis)

    def test_raises_on_non_cp_member(self):
        # A member with a negative Choi spectrum is not completely positive.
        non_cp = 1.5 * self.Gmz_plus - 0.5 * self.Gmz_minus
        complement = np.eye(self.basis.dim) - non_cp  # keeps the pair trace-preserving
        with self.assertRaises(ValueError):
            Instrument.from_cptr_superops({'plus': non_cp, 'minus': complement}, self.basis)

    # --- RootConjOperator unit tests ---

    def test_root_conj_operator_to_dense_shape(self):
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


class FromEffectsTester(BaseCase):
    """Tests for the Instrument.from_effects convenience constructor."""

    def setUp(self):
        model = std.target_model()
        self.E0, self.E1, self.Gmz_plus, self.Gmz_minus = z_measurement_projectors(model)
        self.basis = model.basis

    def _members(self, instrument):
        return {k: v.to_dense('HilbertSchmidt') for k, v in instrument.items()}

    def test_returns_instrument_with_keys(self):
        instrument = Instrument.from_effects({'p0': self.E0, 'p1': self.E1}, self.basis)
        self.assertIsInstance(instrument, Instrument)
        self.assertSetEqual(set(instrument.keys()), {'p0', 'p1'})
        for mx in self._members(instrument).values():
            self.assertEqual(mx.shape, (self.basis.dim, self.basis.dim))

    def test_effect_only_reproduces_projectors(self):
        # With the gate defaulting to the identity, the members are the ideal
        # projective measurement maps rho -> E^1/2 rho E^1/2.
        members = self._members(Instrument.from_effects({'p0': self.E0, 'p1': self.E1}, self.basis))
        self.assertArraysAlmostEqual(members['p0'], self.Gmz_plus)
        self.assertArraysAlmostEqual(members['p1'], self.Gmz_minus)

    def test_identity_gate_matches_effect_only(self):
        bare = self._members(Instrument.from_effects({'p0': self.E0, 'p1': self.E1}, self.basis))
        with_gate = self._members(Instrument.from_effects(
            {'p0': (self.E0, np.eye(4)), 'p1': (self.E1, None)}, self.basis))
        for k in bare.keys():
            self.assertArraysAlmostEqual(with_gate[k], bare[k])

    def test_total_channel_is_trace_preserving(self):
        members = self._members(Instrument.from_effects({'p0': self.E0, 'p1': self.E1}, self.basis))
        total = sum(members.values())
        np.testing.assert_allclose(total[0], [1, 0, 0, 0], atol=1e-9)

    def test_members_are_cp(self):
        instrument = Instrument.from_effects({'p0': self.E0, 'p1': self.E1}, self.basis)
        for mx in self._members(instrument).values():
            viol = pygsti.tools.sum_of_negative_choi_eigenvalues_gate(mx, 'pp')
            self.assertGreaterEqual(viol, -1e-8)

    def test_unitary_gate_and_matrix_effect(self):
        # A post-measurement Z on outcome p1, and effects given as 2x2 matrices.
        from pygsti.tools import optools as ot
        proj0 = np.array([[1, 0], [0, 0]], dtype=complex)
        proj1 = np.array([[0, 0], [0, 1]], dtype=complex)
        Z = np.array([[1, 0], [0, -1]], dtype=complex)
        members = self._members(Instrument.from_effects(
            {'p0': (proj0, np.eye(2)), 'p1': (proj1, Z)}, self.basis))
        expected_p1 = ot.unitary_to_superop(Z, self.basis) @ ot.rootconj_superop(self.E1, self.basis)
        self.assertArraysAlmostEqual(members['p0'], self.Gmz_plus)
        self.assertArraysAlmostEqual(members['p1'], expected_p1)

    def test_relaxed_cp_full_tp_gate(self):
        # 'full TP' relaxes the per-member CP constraint while keeping the instrument
        # jointly trace-preserving; each gate factor is a FullTPOp.
        instrument = Instrument.from_effects(
            {'p0': self.E0, 'p1': self.E1}, self.basis, gate_parameterization='full TP')
        for member in instrument.values():
            self.assertIsInstance(member, op.ComposedOp)
            self.assertIsInstance(member.factorops[1], op.FullTPOp)
        total = sum(self._members(instrument).values())
        np.testing.assert_allclose(total[0], [1, 0, 0, 0], atol=1e-9)

    def test_rejects_full_gate_parameterization(self):
        # 'full' is not trace-preserving and would break the instrument's joint TP.
        with self.assertRaises(ValueError):
            Instrument.from_effects(
                {'p0': self.E0, 'p1': self.E1}, self.basis, gate_parameterization='full')

    def test_feeds_plain_instrument(self):
        # The same (effect, gate) data is also valid for the plain dense constructor.
        superops = {'p0': self.Gmz_plus, 'p1': self.Gmz_minus}
        self.assertIsInstance(Instrument(superops), Instrument)
        self.assertIsInstance(convert(Instrument(superops), 'full TP', self.basis), TPInstrument)

    def test_raises_when_effects_incomplete(self):
        with self.assertRaises(ValueError):
            Instrument.from_effects({'p0': self.E0, 'p1': self.E0}, self.basis)

    def test_raises_on_non_tp_gate(self):
        with self.assertRaises(ValueError):
            Instrument.from_effects({'p0': (self.E0, 0.9 * np.eye(4)), 'p1': self.E1}, self.basis)


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

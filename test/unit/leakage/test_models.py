import numpy as np

from pygsti.baseobjs import Label
from pygsti.baseobjs.statespace import ExplicitStateSpace
from pygsti.circuits import Circuit
from pygsti.tools.basistools import stdmx_to_vec, vec_to_stdmx
from pygsti.leakage.models import (
    leaky_qubit_model_from_pspec,
    promote_bb_to_bt,
    random_unitary_excitation,
    _lift_unitary_bb_to_bt,
)
from ..util import BaseCase


# A few reusable 2x2 unitaries.
_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
_Z = np.diag([1.0, -1.0]).astype(complex)
_H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
_I2 = np.eye(2, dtype=complex)
# CNOT with the first register as control -- a genuinely entangling (non-kron) 4x4 unitary.
_CNOT = np.array([[1, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 0, 1],
                  [0, 0, 1, 0]], dtype=complex)


def _haar_unitary(dim, seed):
    """A reproducible Haar-random unitary via QR of a complex Ginibre matrix."""
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((dim, dim)) + 1j * rng.standard_normal((dim, dim))
    q, r = np.linalg.qr(z)
    ph = np.diagonal(r).copy()
    ph /= np.abs(ph)
    return q * ph


class LeakyQubitModelTester(BaseCase):

    @classmethod
    def setUpClass(cls):
        from pygsti.processors import QubitProcessorSpec
        pspec = QubitProcessorSpec(1, ['Gxpi2', 'Gypi2'], qubit_labels=['Q0'])
        cls.pspec    = pspec
        cls.model    = leaky_qubit_model_from_pspec(pspec)
        cls.mx_basis = cls.model.basis

    # ---- structural properties -------------------------------------------------

    def test_state_space_udim(self):
        self.assertEqual(self.model.state_space.udim, 3)

    def test_gauge_group_is_direct_sum(self):
        from pygsti.models.gaugegroup import DirectSumUnitaryGroup
        self.assertIsInstance(self.model.default_gauge_group, DirectSumUnitaryGroup)

    def test_gauge_group_decomposition(self):
        # The gauge group is a block-diagonal subgroup of U(3): a nontrivial unitary
        # factor on the 2-dimensional computational subspace, plus a (by default
        # trivial) factor on the 1-dimensional leakage level.
        from pygsti.models.gaugegroup import UnitaryGaugeGroup, TrivialGaugeGroup
        subs = self.model.default_gauge_group.subgroups
        self.assertEqual(len(subs), 2)
        self.assertIsInstance(subs[0], UnitaryGaugeGroup)
        self.assertIsInstance(subs[1], TrivialGaugeGroup)

    def test_povm_has_two_effects(self):
        povm = self.model.povms['Mdefault']
        self.assertIn('0', povm)
        self.assertIn('1', povm)

    def test_rho0_is_ground_projector(self):
        rho_vec = self.model.preps['rho0'].to_dense().ravel()
        rho_mx  = vec_to_stdmx(rho_vec, self.mx_basis)
        expected = np.zeros((3, 3), dtype=complex)
        expected[0, 0] = 1.0
        self.assertArraysAlmostEqual(rho_mx, expected)

    def test_povm_effect0_matches_readout_levels(self):
        eff0_vec = self.model.povms['Mdefault']['0'].to_dense().ravel()
        eff0_mx  = vec_to_stdmx(eff0_vec, self.mx_basis)
        expected = np.diag([1.0, 0.0, 0.0]).astype(complex)  # default levels_readout_zero=(0,)
        self.assertArraysAlmostEqual(eff0_mx, expected)

    def test_povm_effects_sum_to_identity(self):
        povm = self.model.povms['Mdefault']
        eff0 = povm['0'].to_dense().ravel()
        eff1 = povm['1'].to_dense().ravel()
        expected = stdmx_to_vec(np.eye(3, dtype=complex), self.mx_basis).ravel()
        self.assertArraysAlmostEqual(eff0 + eff1, expected)

    def test_gate_superop_dimensions(self):
        for op in self.model.operations.values():
            mx = op.to_dense()
            self.assertEqual(mx.shape, (9, 9))

    # ---- gate action / forward-simulation correctness --------------------------

    def test_empty_circuit_reads_zero(self):
        probs = self.model.probabilities(Circuit([], line_labels=['Q0']))
        self.assertAlmostEqual(probs['0'], 1.0, places=9)
        self.assertAlmostEqual(probs['1'], 0.0, places=9)

    def test_single_gxpi2_is_balanced(self):
        c = Circuit([('Gxpi2', 'Q0')], line_labels=['Q0'])
        probs = self.model.probabilities(c)
        self.assertAlmostEqual(probs['0'], 0.5, places=9)
        self.assertAlmostEqual(probs['1'], 0.5, places=9)

    def test_double_gxpi2_flips(self):
        c = Circuit([('Gxpi2', 'Q0'), ('Gxpi2', 'Q0')], line_labels=['Q0'])
        probs = self.model.probabilities(c)
        self.assertAlmostEqual(probs['0'], 0.0, places=9)
        self.assertAlmostEqual(probs['1'], 1.0, places=9)

    def test_gate_is_identity_on_leakage_level(self):
        # Each ideal gate promotes a 2x2 unitary to a 3x3 unitary that fixes level 2.
        # After changing back to the standard superoperator basis, the leakage
        # population should be untouched: feeding in |2><2| returns |2><2|.
        from pygsti.tools import basistools as pgbt
        leak_state = np.zeros((3, 3), dtype=complex)
        leak_state[2, 2] = 1.0
        vin = stdmx_to_vec(leak_state, self.mx_basis).ravel()
        for lbl, op in self.model.operations.items():
            vout = op.to_dense() @ vin
            mout = vec_to_stdmx(vout, self.mx_basis)
            self.assertArraysAlmostEqual(mout, leak_state)

    # ---- variety: alternate bases ---------------------------------------------

    def test_alternate_bases_build_and_agree(self):
        # Probabilities are basis-independent; models built in different HS bases
        # should give identical outcome probabilities. We vary over *Hermitian* bases
        # ('gm', 'qt') -- like the default 'l2p1' -- so the superkets/superoperators stay
        # real, matching the model's real parameter vector. (The matrix-unit 'std' basis
        # is non-Hermitian: complex gate parameters would be silently truncated by the
        # real paramvec, so it is deliberately not exercised here.)
        c = Circuit([('Gxpi2', 'Q0'), ('Gypi2', 'Q0')], line_labels=['Q0'])
        ref = self.model.probabilities(c)
        for basis_name in ('gm', 'qt'):
            with self.subTest(basis=basis_name):
                m = leaky_qubit_model_from_pspec(self.pspec, mx_basis=basis_name)
                self.assertEqual(m.basis.name, basis_name)
                self.assertEqual(m.state_space.udim, 3)
                probs = m.probabilities(c)
                for outcome in ('0', '1'):
                    self.assertAlmostEqual(probs[outcome], ref[outcome], places=9)

    # ---- variety: readout-level assignment ------------------------------------

    def test_levels_readout_zero_variation(self):
        # Route the leakage level (index 2) into the "0" outcome alongside the ground
        # level. E0 should then project onto levels {0, 2}.
        m = leaky_qubit_model_from_pspec(self.pspec, levels_readout_zero=(0, 2))
        eff0_vec = m.povms['Mdefault']['0'].to_dense().ravel()
        eff0_mx  = vec_to_stdmx(eff0_vec, m.basis)
        expected = np.diag([1.0, 0.0, 1.0]).astype(complex)
        self.assertArraysAlmostEqual(eff0_mx, expected)

    # ---- variety: different gate sets ------------------------------------------

    def test_nonhermitian_basis_rejected(self):
        # The matrix-unit 'std' basis is non-Hermitian; building in it would silently
        # truncate complex gate parameters against the real paramvec, so it is rejected.
        with self.assertRaises(ValueError):
            leaky_qubit_model_from_pspec(self.pspec, mx_basis='std')

    def test_alternate_gateset_builds(self):
        from pygsti.processors import QubitProcessorSpec
        pspec = QubitProcessorSpec(1, ['Gxpi2', 'Gzpi2', 'Ghtest'],
                                   nonstd_gate_unitaries={'Ghtest': _H},
                                   qubit_labels=['Q0'])
        m = leaky_qubit_model_from_pspec(pspec)
        self.assertEqual(m.state_space.udim, 3)
        for gname in ('Gxpi2', 'Gzpi2', 'Ghtest'):
            self.assertIn(Label((gname, 'Q0')), m.operations)


class LiftUnitaryBBtoBTTester(BaseCase):
    """
    Direct tests for the qubit-qubit -> qubit-qutrit unitary lift.

    The lift adds a leakage level to the *second* register only. It acts as the input
    unitary on the computational subspace (second register in {0, 1}) and as the
    identity on the leakage subspace (second register at level 2) -- i.e. the
    block-diagonal unitary ``u ⨁ I2``. This is unitary for *every* unitary input, with
    no restriction on the second-register factor.
    """

    def test_identity_lifts_to_identity(self):
        self.assertArraysAlmostEqual(_lift_unitary_bb_to_bt(np.eye(4)), np.eye(6))

    def test_gate_on_first_register(self):
        # kron(A, I2) lifts to a unitary for *any* A. (The lift fixes the second
        # register's leakage level, so the result is the block-diagonal u ⨁ I2 rather
        # than kron(A, I3); we only assert unitarity here.)
        for A in (_X, _Y, _H):
            with self.subTest(A=A.round(3).tolist()):
                lifted = _lift_unitary_bb_to_bt(np.kron(A, _I2))
                self.assertArraysAlmostEqual(lifted @ lifted.conj().T, np.eye(6))

    def test_gate_on_second_register_fixes_leakage_level(self):
        # kron(I2, M) lifts to kron(I2, M_trit) for *any* 2x2 unitary M, where M_trit
        # acts as M on levels {0,1} and as the identity on the leakage level (index 2).
        # This holds beyond the Paulis: a Hadamard on the second register works too.
        for M in (_X, _Y, _Z, _H):
            with self.subTest(M=M.round(3).tolist()):
                M_trit = np.eye(3, dtype=complex)
                M_trit[:2, :2] = M
                lifted = _lift_unitary_bb_to_bt(np.kron(_I2, M))
                self.assertArraysAlmostEqual(lifted, np.kron(_I2, M_trit))

    def test_product_lifts_are_unitary(self):
        for A, P in ((_X, _X), (_H, _Y), (_Y, _Z), (_I2, _H)):
            with self.subTest():
                lifted = _lift_unitary_bb_to_bt(np.kron(A, P))
                self.assertArraysAlmostEqual(lifted @ lifted.conj().T, np.eye(6))

    def test_nonpauli_on_second_register_is_unitary(self):
        # A Hadamard on the second register mixes two Pauli components, but the lift is
        # still exactly unitary (block-diagonal u ⨁ I2) and never warns.
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('error')  # any warning becomes a failure
            lifted = _lift_unitary_bb_to_bt(np.kron(_I2, _H))
        self.assertArraysAlmostEqual(lifted @ lifted.conj().T, np.eye(6))

    def test_output_shape(self):
        self.assertEqual(_lift_unitary_bb_to_bt(np.kron(_X, _Z)).shape, (6, 6))

    def test_general_unitary_block_structure(self):
        # The defining contract, checked on *entangling / non-kron* inputs: the lift acts
        # as u on the computational subspace (second register in {0,1}) and as the exact
        # identity on the leakage complement (second register at level 2), with no coupling
        # between the two. Flat index k = 3*a + b, so comp = {0,1,3,4}, leak = {2,5}.
        comp = [3 * a + b for a in range(2) for b in range(2)]  # [0,1,3,4]
        leak = [3 * a + 2 for a in range(2)]                    # [2,5]
        for u in (_CNOT, _haar_unitary(4, 7), _haar_unitary(4, 99)):
            with self.subTest():
                L = _lift_unitary_bb_to_bt(u)
                self.assertArraysAlmostEqual(L[np.ix_(comp, comp)], u)                  # nominal gate
                self.assertArraysAlmostEqual(L[np.ix_(leak, leak)], np.eye(2))          # identity on leakage
                self.assertArraysAlmostEqual(L[np.ix_(comp, leak)], np.zeros((4, 2)))   # no leak->comp coupling
                self.assertArraysAlmostEqual(L[np.ix_(leak, comp)], np.zeros((2, 4)))   # no comp->leak coupling
                self.assertArraysAlmostEqual(L @ L.conj().T, np.eye(6))

    def test_lift_respects_composition(self):
        # The lift is u -> u (+) I2 on a fixed subspace decomposition, hence a
        # homomorphism: lifting a product equals the product of the lifts. This is what
        # lets a circuit of lifted gates agree with lifting the composed unitary.
        u1 = _haar_unitary(4, 1)
        u2 = _haar_unitary(4, 2)
        self.assertArraysAlmostEqual(
            _lift_unitary_bb_to_bt(u1 @ u2),
            _lift_unitary_bb_to_bt(u1) @ _lift_unitary_bb_to_bt(u2),
        )

    def test_wrong_input_shape_raises(self):
        with self.assertRaises(AssertionError):
            _lift_unitary_bb_to_bt(np.eye(2))


class PromoteBBToBTTester(BaseCase):

    @classmethod
    def setUpClass(cls):
        from pygsti.processors import QubitProcessorSpec
        from pygsti.models import create_explicit_model
        pspec_2q  = QubitProcessorSpec(2, ['Gcnot', 'Gxpi2', 'Gypi2'], qubit_labels=['Q0', 'Q1'], geometry='line')
        cls.model_2q = create_explicit_model(pspec_2q)
        cls.model_6  = promote_bb_to_bt(cls.model_2q)

    # ---- structural properties -------------------------------------------------

    def test_state_space_udim(self):
        self.assertEqual(self.model_6.state_space.udim, 6)

    def test_gauge_group_is_unitary(self):
        from pygsti.models.gaugegroup import UnitaryGaugeGroup
        self.assertIsInstance(self.model_6.default_gauge_group, UnitaryGaugeGroup)

    def test_povm_has_4_effects(self):
        povm = self.model_6.povms['Mdefault']
        for key in ('00', '01', '10', '11'):
            self.assertIn(key, povm)

    def test_povm_effects_sum_to_identity(self):
        mx_basis = self.model_6.basis
        total = sum(
            eff.to_dense().ravel() for eff in self.model_6.povms['Mdefault'].values()
        )
        expected = stdmx_to_vec(np.eye(6, dtype=complex), mx_basis).ravel()
        self.assertArraysAlmostEqual(total, expected)

    def test_povm_effects_match_expected_projectors(self):
        # Effect "ab" is kron(qubit-projector a, qutrit-readout-projector b).
        mx_basis = self.model_6.basis
        E0_b = np.diag([1.0, 0.0]).astype(complex)
        E1_b = np.diag([0.0, 1.0]).astype(complex)
        E0_t = np.diag([1.0, 0.0, 0.0]).astype(complex)  # levels_readout_zero=(0,)
        E1_t = np.diag([0.0, 1.0, 1.0]).astype(complex)
        expected = {
            '00': np.kron(E0_b, E0_t), '01': np.kron(E0_b, E1_t),
            '10': np.kron(E1_b, E0_t), '11': np.kron(E1_b, E1_t),
        }
        povm = self.model_6.povms['Mdefault']
        for key, exp_mx in expected.items():
            with self.subTest(effect=key):
                eff_mx = vec_to_stdmx(povm[key].to_dense().ravel(), mx_basis)
                self.assertArraysAlmostEqual(eff_mx, exp_mx)

    def test_rho0_is_ground_projector(self):
        mx_basis = self.model_6.basis
        rho_mx = vec_to_stdmx(self.model_6.preps['rho0'].to_dense().ravel(), mx_basis)
        expected = np.zeros((6, 6), dtype=complex)
        expected[0, 0] = 1.0
        self.assertArraysAlmostEqual(rho_mx, expected)

    def test_operations_match_source(self):
        # Every non-idle gate in the 2-qubit model should be present in the 6-level model.
        non_idle = [k for k in self.model_2q.operations.keys() if k != Label(())]
        for op_lbl in non_idle:
            self.assertIn(op_lbl, self.model_6.operations)

    def test_gate_superop_dimensions(self):
        for op in self.model_6.operations.values():
            mx = op.to_dense()
            self.assertEqual(mx.shape, (36, 36))

    # ---- forward-simulation correctness ----------------------------------------

    def test_empty_circuit_reads_zero(self):
        probs = self.model_6.probabilities(Circuit([], line_labels=['Q0', 'Q1']))
        self.assertAlmostEqual(probs['00'], 1.0, places=9)

    def test_single_qubit_gate_marginal(self):
        # A pi/2 X on Q0 splits population between "00" and "10"; Q1 stays in "0".
        c = Circuit([('Gxpi2', 'Q0')], line_labels=['Q0', 'Q1'])
        probs = self.model_6.probabilities(c)
        self.assertAlmostEqual(probs['00'], 0.5, places=9)
        self.assertAlmostEqual(probs['10'], 0.5, places=9)
        self.assertAlmostEqual(probs['01'] + probs['11'], 0.0, places=9)

    def test_cnot_entangles(self):
        # Prepare |+> on Q0 then CNOT -> Bell state; readout is balanced over 00/11.
        c = Circuit([('Gxpi2', 'Q0'), ('Gxpi2', 'Q0'), ('Gcnot', 'Q0', 'Q1')],
                    line_labels=['Q0', 'Q1'])
        # (Gxpi2 twice == X, so this drives Q0 to |1> and CNOT flips Q1 -> "11".)
        probs = self.model_6.probabilities(c)
        self.assertAlmostEqual(probs['11'], 1.0, places=9)

    # ---- variety: different 2-qubit source models ------------------------------

    def test_nonhermitian_basis_rejected(self):
        # A non-Hermitian component ('std') in either register basis is rejected.
        with self.assertRaises(ValueError):
            promote_bb_to_bt(self.model_2q, sys1_basis='std')

    def test_variety_of_source_models(self):
        from pygsti.processors import QubitProcessorSpec
        from pygsti.models import create_explicit_model
        gate_sets = [
            ['Gxpi2', 'Gcphase'],
            ['Gxpi2', 'Gypi2', 'Gcnot'],
        ]
        for gates in gate_sets:
            with self.subTest(gates=gates):
                ps = QubitProcessorSpec(2, gates, qubit_labels=['Q0', 'Q1'], geometry='line')
                m6 = promote_bb_to_bt(create_explicit_model(ps))
                self.assertEqual(m6.state_space.udim, 6)
                probs = m6.probabilities(Circuit([], line_labels=['Q0', 'Q1']))
                self.assertAlmostEqual(probs['00'], 1.0, places=9)


class RandomUnitaryExcitationTester(BaseCase):

    # A variety of state spaces: bare qutrit, bare ququart, and qubit (X) qutrit.
    QUTRIT   = ExplicitStateSpace(['Q0'], [3])
    QUQUART  = ExplicitStateSpace(['Q0'], [4])
    QB_QT    = ExplicitStateSpace(['A', 'B'], [2, 3])

    def test_return_types(self):
        from pygsti.modelmembers.operations import EmbeddedOp
        G, p = random_unitary_excitation(self.QUTRIT, 'Q0', 'gm', 0, 0.3, rng_seed=1)
        self.assertIsInstance(G, EmbeddedOp)
        self.assertIsInstance(p, np.ndarray)

    def test_p_is_unit_norm_and_supported_on_two_levels(self):
        cases = [
            (self.QUTRIT, 'Q0', 'gm', 0, 3),
            (self.QUQUART, 'Q0', 'gm', 2, 4),
            (self.QB_QT, 'B', 'gm', 1, 3),
        ]
        for ss, sub, basis, glevel, udim in cases:
            with self.subTest(state_space=str(ss), ground_level=glevel):
                _, p = random_unitary_excitation(ss, sub, basis, glevel, 0.5, rng_seed=2)
                self.assertEqual(p.shape, (udim,))
                self.assertAlmostEqual(np.linalg.norm(p), 1.0, places=12)
                nz = set(np.nonzero(np.abs(p) > 1e-12)[0].tolist())
                self.assertTrue(nz.issubset({glevel, glevel + 1}))

    def test_zero_strength_is_identity(self):
        for ss, sub, udim in [(self.QUTRIT, 'Q0', 3), (self.QB_QT, 'B', 6)]:
            with self.subTest(state_space=str(ss)):
                G, _ = random_unitary_excitation(ss, sub, 'gm', 0, 0.0, rng_seed=3)
                self.assertArraysAlmostEqual(G.to_dense(), np.eye(udim ** 2))

    def test_superop_is_unitary_channel(self):
        # A unitary channel's superoperator in a normalized Hermitian basis is real
        # orthogonal (S S^T = I).
        for ss, sub in [(self.QUTRIT, 'Q0'), (self.QUQUART, 'Q0'), (self.QB_QT, 'B')]:
            with self.subTest(state_space=str(ss)):
                glevel = 2 if ss is self.QUQUART else (1 if ss is self.QB_QT else 0)
                G, _ = random_unitary_excitation(ss, sub, 'gm', glevel, 0.7, rng_seed=4)
                S = G.to_dense()
                self.assertArraysAlmostEqual(S @ S.T, np.eye(S.shape[0]))

    def test_embedded_superop_dimensions(self):
        G, _ = random_unitary_excitation(self.QB_QT, 'B', 'gm', 1, 0.4, rng_seed=5)
        self.assertEqual(G.to_dense().shape, (36, 36))  # (2*3)^2

    def test_determinism_with_seed(self):
        G1, p1 = random_unitary_excitation(self.QUTRIT, 'Q0', 'gm', 0, 0.6, rng_seed=42)
        G2, p2 = random_unitary_excitation(self.QUTRIT, 'Q0', 'gm', 0, 0.6, rng_seed=42)
        self.assertArraysAlmostEqual(p1, p2)
        self.assertArraysAlmostEqual(G1.to_dense(), G2.to_dense())

    def test_different_seeds_differ(self):
        _, p1 = random_unitary_excitation(self.QUTRIT, 'Q0', 'gm', 0, 0.6, rng_seed=1)
        _, p2 = random_unitary_excitation(self.QUTRIT, 'Q0', 'gm', 0, 0.6, rng_seed=2)
        self.assertGreater(np.linalg.norm(p1 - p2), 1e-6)

    def test_ground_level_out_of_range_raises(self):
        # ground_level+1 must be a valid level of the subsystem.
        with self.assertRaises(AssertionError):
            random_unitary_excitation(self.QUTRIT, 'Q0', 'gm', 2, 0.5, rng_seed=0)

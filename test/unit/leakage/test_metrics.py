import numpy as np

import warnings

from pygsti.baseobjs.basis import Basis, BuiltinBasis
from pygsti.tools import optools as pgot, basistools as pgbt
from pygsti.tools.basistools import vec_to_stdmx
from pygsti.leakage.core import computational_effect
from pygsti.leakage.metrics import (
    choi_state,
    subspace_jtracedist,
    subspace_superop_fro_dist,
    subspace_entanglement_fidelity,
    gate_leakage_profile,
    gate_seepage_profile,
    pop_transport_profile,
)
from ..util import BaseCase, needs_cvxpy


def _make_superop(u3x3, basis):
    """Convert a 3x3 unitary to its superoperator in `basis`."""
    superop_std = pgot.unitary_to_std_process_mx(u3x3)
    return pgbt.change_basis(superop_std, 'std', basis)


def _xpi2_superop(basis):
    """Xπ/2 acting on the computational subspace; leakage level is untouched."""
    c, s = np.cos(np.pi / 4), np.sin(np.pi / 4)
    u = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=complex)
    return _make_superop(u, basis)


def _leaky_superop(basis, theta=np.pi / 6):
    """Rotation by `theta` mixing |0⟩ and the leakage level |2⟩."""
    u = np.eye(3, dtype=complex)
    u[0, 0] =  np.cos(theta)
    u[0, 2] = -np.sin(theta)
    u[2, 0] =  np.sin(theta)
    u[2, 2] =  np.cos(theta)
    return _make_superop(u, basis)


class ChoistateTester(BaseCase):

    def setUp(self):
        self.basis   = BuiltinBasis('l2p1', 9)
        self.op_id   = np.eye(9)
        self.op_xpi2 = _xpi2_superop(self.basis)

    def test_choi_state_id(self):
        C = choi_state(self.op_id, self.basis)
        self.assertArraysAlmostEqual(C, C.conj().T)
        eigvals = np.linalg.eigvalsh(C)
        self.assertTrue(np.all(eigvals >= -1e-10), f"Negative eigenvalue: {eigvals.min():.3e}")
        self.assertAlmostEqual(np.trace(C).real, 1.0, places=10)

    def test_choi_state_xpi2(self):
        C = choi_state(self.op_xpi2, self.basis)
        self.assertArraysAlmostEqual(C, C.conj().T)
        eigvals = np.linalg.eigvalsh(C)
        self.assertTrue(np.all(eigvals >= -1e-10), f"Negative eigenvalue: {eigvals.min():.3e}")
        self.assertAlmostEqual(np.trace(C).real, 1.0, places=10)


class DistanceMetricsTester(BaseCase):

    def setUp(self):
        self.basis   = BuiltinBasis('l2p1', 9)
        self.op_id   = np.eye(9)
        self.op_xpi2 = _xpi2_superop(self.basis)

    # subspace_jtracedist

    def test_jtracedist_self_is_zero(self):
        d = subspace_jtracedist(self.op_id, self.op_id, self.basis)
        self.assertAlmostEqual(d, 0.0, places=10)

    def test_jtracedist_id_vs_xpi2(self):
        d_ab = subspace_jtracedist(self.op_id, self.op_xpi2, self.basis)
        self.assertGreater(d_ab, 0.0)
        self.assertAlmostEqual(d_ab, subspace_jtracedist(self.op_xpi2, self.op_id, self.basis), places=10)

    # subspace_superop_fro_dist

    def test_fro_dist_self_is_zero(self):
        d = subspace_superop_fro_dist(self.op_id, self.op_id, self.basis)
        self.assertAlmostEqual(d, 0.0, places=10)

    def test_fro_dist_id_vs_xpi2(self):
        d_ab = subspace_superop_fro_dist(self.op_id, self.op_xpi2, self.basis)
        self.assertGreater(d_ab, 0.0)
        self.assertAlmostEqual(d_ab, subspace_superop_fro_dist(self.op_xpi2, self.op_id, self.basis), places=10)

    # subspace_entanglement_fidelity

    def test_entanglement_fidelity_self_is_one(self):
        f = subspace_entanglement_fidelity(self.op_id, self.op_id, self.basis)
        self.assertAlmostEqual(f, 1.0, places=10)

    def test_entanglement_fidelity_id_vs_xpi2(self):
        f_ab = subspace_entanglement_fidelity(self.op_id, self.op_xpi2, self.basis)
        self.assertLess(f_ab, 1.0)
        self.assertAlmostEqual(f_ab, subspace_entanglement_fidelity(self.op_xpi2, self.op_id, self.basis), places=10)

    # subspace_diamonddist (requires cvxpy)

    @needs_cvxpy
    def test_diamonddist_self_is_zero(self):
        from pygsti.leakage.metrics import subspace_diamonddist
        d = subspace_diamonddist(self.op_id, self.op_id, self.basis)
        self.assertAlmostEqual(d, 0.0, places=6)

    @needs_cvxpy
    def test_diamonddist_positive_for_different_ops(self):
        from pygsti.leakage.metrics import subspace_diamonddist
        d = subspace_diamonddist(self.op_id, self.op_xpi2, self.basis)
        self.assertGreater(d, 0.0)


class TransportProfileTester(BaseCase):

    def setUp(self):
        self.basis    = BuiltinBasis('l2p1', 9)
        self.op_id    = np.eye(9)
        self.op_comp  = _xpi2_superop(self.basis)   # purely computational gate
        self.op_leaky = _leaky_superop(self.basis)  # mixes |0⟩ and leakage |2⟩

    def test_leakage_profile_identity_is_zero(self):
        rates, _ = gate_leakage_profile(self.op_id, self.basis)
        self.assertAlmostEqual(float(np.max(np.abs(rates))), 0.0, places=10)

    def test_seepage_profile_identity_is_zero(self):
        rates, _ = gate_seepage_profile(self.op_id, self.basis)
        self.assertAlmostEqual(float(np.max(np.abs(rates))), 0.0, places=10)

    def test_leakage_profile_computational_gate_is_zero(self):
        # A gate that acts within the computational subspace and fixes |2⟩
        # cannot transport population out of the computational subspace.
        rates, _ = gate_leakage_profile(self.op_comp, self.basis)
        self.assertAlmostEqual(float(np.max(np.abs(rates))), 0.0, places=10)

    def test_leakage_profile_nonzero_for_leaky_gate(self):
        rates, _ = gate_leakage_profile(self.op_leaky, self.basis)
        self.assertGreater(float(np.max(np.abs(rates))), 1e-6)

    def test_seepage_profile_nonzero_for_leaky_gate(self):
        rates, _ = gate_seepage_profile(self.op_leaky, self.basis)
        self.assertGreater(float(np.max(np.abs(rates))), 1e-6)

    def test_pop_transport_profile_non_projector_raises(self):
        E_bad = 0.5 * np.eye(3)  # Hermitian but not a projector
        with self.assertRaises(ValueError):
            pop_transport_profile(E_bad, self.op_id, self.basis)

    def test_pop_transport_profile_leaky(self):
        E = computational_effect(self.basis)
        rates, states = pop_transport_profile(E, self.op_leaky, self.basis)
        # Count: one pair per dimension of E_sub (rank(E) = 2 for l2p1).
        expected_count = int(round(np.trace(E).real))
        self.assertEqual(len(rates), expected_count)
        self.assertEqual(len(states), expected_count)
        # Eigenvalue bounds: transport rates are probabilities.
        self.assertTrue(np.all(rates >= -1e-10), f"Negative transport rate: {rates.min():.3e}")
        self.assertTrue(np.all(rates <= 1.0 + 1e-10), f"Transport rate > 1: {rates.max():.3e}")
        # Eigenvector equation: E_transport @ state ≈ rate * state.
        n = E.shape[0]
        E_perp_vec = pgbt.stdmx_to_vec(np.eye(n) - E, self.basis)
        transport_mat = vec_to_stdmx(self.op_leaky.T @ E_perp_vec, self.basis, keep_complex=True)
        transport_mat = E @ transport_mat @ E
        for rate, state in zip(rates, states):
            self.assertArraysAlmostEqual(transport_mat @ state, rate * state)


class NonLeakageBasisTester(BaseCase):
    """
    Tests for metric functions called with a non-leakage basis (pp).

    When implies_leakage_modeling is False several functions degrade gracefully:
    - tensorized_teststate_density: uses the full identity as the computational
      effect (lines 40-41).
    - subspace_superop_fro_dist: uses IdentityOperator instead of the projector
      (line 187).
    - gate_leakage_profile / gate_seepage_profile: warn and return empty arrays
      when the computational subspace equals the whole Hilbert space (lines
      339-345, 361-367).
    """

    def setUp(self):
        self.pp_basis = Basis.cast('pp', 4)
        self.op_id    = np.eye(4)

    # Lines 40-41: non-leakage path in tensorized_teststate_density, reached
    # through subspace_jtracedist (which calls apply_tensorized_to_teststate
    # which calls tensorized_teststate_density).
    def test_jtracedist_nonleakage_basis_self_is_zero(self):
        d = subspace_jtracedist(self.op_id, self.op_id, self.pp_basis)
        self.assertAlmostEqual(d, 0.0, places=10)

    # Line 187: non-leakage path in subspace_superop_fro_dist.
    def test_fro_dist_nonleakage_basis_self_is_zero(self):
        d = subspace_superop_fro_dist(self.op_id, self.op_id, self.pp_basis)
        self.assertAlmostEqual(d, 0.0, places=10)

    # Line 240: non-leakage path in subspace_diamonddist.
    @needs_cvxpy
    def test_diamonddist_nonleakage_basis_self_is_zero(self):
        from pygsti.leakage.metrics import subspace_diamonddist
        d = subspace_diamonddist(self.op_id, self.op_id, self.pp_basis)
        self.assertAlmostEqual(d, 0.0, places=6)

    # Lines 339-345: warn + return empty in gate_leakage_profile.
    def test_leakage_profile_fullspace_warns_and_empty(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            rates, states = gate_leakage_profile(self.op_id, self.pp_basis)
        self.assertTrue(len(w) > 0, "Expected a warning but none were raised")
        self.assertEqual(rates.shape, (0,))
        self.assertEqual(states, [])

    # Lines 361-367: warn + return empty in gate_seepage_profile.
    def test_seepage_profile_fullspace_warns_and_empty(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            rates, states = gate_seepage_profile(self.op_id, self.pp_basis)
        self.assertTrue(len(w) > 0, "Expected a warning but none were raised")
        self.assertEqual(rates.shape, (0,))
        self.assertEqual(states, [])

import numpy as np

from pygsti.baseobjs.basis import BuiltinBasis
from pygsti.tools import optools as pgot, basistools as pgbt
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
    """Convert a 3×3 unitary to its superoperator in `basis`."""
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

    def test_choi_state_hermitian(self):
        C = choi_state(self.op_id, self.basis)
        self.assertArraysAlmostEqual(C, C.conj().T)

    def test_choi_state_psd(self):
        C = choi_state(self.op_id, self.basis)
        eigvals = np.linalg.eigvalsh(C)
        self.assertTrue(
            np.all(eigvals >= -1e-10),
            f"Choi state has negative eigenvalue: {eigvals.min():.3e}"
        )

    def test_choi_state_xpi2_hermitian(self):
        C = choi_state(self.op_xpi2, self.basis)
        self.assertArraysAlmostEqual(C, C.conj().T)

    def test_choi_state_xpi2_psd(self):
        C = choi_state(self.op_xpi2, self.basis)
        eigvals = np.linalg.eigvalsh(C)
        self.assertTrue(
            np.all(eigvals >= -1e-10),
            f"Choi state has negative eigenvalue: {eigvals.min():.3e}"
        )


class DistanceMetricsTester(BaseCase):

    def setUp(self):
        self.basis   = BuiltinBasis('l2p1', 9)
        self.op_id   = np.eye(9)
        self.op_xpi2 = _xpi2_superop(self.basis)

    # subspace_jtracedist

    def test_jtracedist_self_is_zero(self):
        d = subspace_jtracedist(self.op_id, self.op_id, self.basis)
        self.assertAlmostEqual(d, 0.0, places=10)

    def test_jtracedist_positive_for_different_ops(self):
        d = subspace_jtracedist(self.op_id, self.op_xpi2, self.basis)
        self.assertGreater(d, 0.0)

    # subspace_superop_fro_dist

    def test_fro_dist_self_is_zero(self):
        d = subspace_superop_fro_dist(self.op_id, self.op_id, self.basis)
        self.assertAlmostEqual(d, 0.0, places=10)

    def test_fro_dist_positive_for_different_ops(self):
        d = subspace_superop_fro_dist(self.op_id, self.op_xpi2, self.basis)
        self.assertGreater(d, 0.0)

    # subspace_entanglement_fidelity

    def test_entanglement_fidelity_self_is_one(self):
        f = subspace_entanglement_fidelity(self.op_id, self.op_id, self.basis)
        self.assertAlmostEqual(f, 1.0, places=10)

    def test_entanglement_fidelity_different_ops_less_than_one(self):
        f = subspace_entanglement_fidelity(self.op_id, self.op_xpi2, self.basis)
        self.assertLess(f, 1.0)

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

    def test_pop_transport_profile_eigenvalues_in_0_1(self):
        E = computational_effect(self.basis)
        rates, _ = pop_transport_profile(E, self.op_leaky, self.basis)
        self.assertTrue(
            np.all(rates >= -1e-10),
            f"Negative transport rate: {rates.min():.3e}"
        )
        self.assertTrue(
            np.all(rates <= 1.0 + 1e-10),
            f"Transport rate > 1: {rates.max():.3e}"
        )

    def test_pop_transport_profile_non_projector_raises(self):
        E_bad = 0.5 * np.eye(3)  # Hermitian but not a projector
        with self.assertRaises(ValueError):
            pop_transport_profile(E_bad, self.op_id, self.basis)

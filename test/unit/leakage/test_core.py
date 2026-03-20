import numpy as np

from pygsti.baseobjs.basis import BuiltinBasis, Basis
from pygsti.tools.matrixtools import is_projector
from pygsti.tools.basistools import stdmx_to_vec
from pygsti.leakage.core import (
    computational_effect, computational_superkets, computational_projector,
)
from ..util import BaseCase


class BasisLeakagePropertiesTester(BaseCase):

    def setUp(self):
        self.l2p1 = BuiltinBasis('l2p1', 9)
        self.pp   = Basis.cast('pp', 4)
        self.std  = Basis.cast('std', 4)

    def test_implies_leakage_modeling_l2p1(self):
        self.assertTrue(self.l2p1.implies_leakage_modeling)

    def test_implies_leakage_modeling_pp(self):
        # pp basis has a full-rank 'I' element (identity), so no leakage
        self.assertFalse(self.pp.implies_leakage_modeling)

    def test_implies_leakage_modeling_std(self):
        # std basis has no 'I*'-labeled element at all
        self.assertFalse(self.std.implies_leakage_modeling)

    def test_is_hermitian_pp(self):
        self.assertTrue(self.pp.is_hermitian())

    def test_is_hermitian_std(self):
        self.assertFalse(self.std.is_hermitian())

    def test_implies_leakage_is_cached(self):
        # First call populates the cache; second call returns the same value.
        val1 = self.l2p1.implies_leakage_modeling
        self.assertTrue(hasattr(self.l2p1, '_implies_leakage'))
        val2 = self.l2p1.implies_leakage_modeling
        self.assertEqual(val1, val2)


class IsProjectorTester(BaseCase):

    def test_rank1_projector(self):
        P = np.diag([1., 0., 0.])
        self.assertTrue(is_projector(P))

    def test_rank2_projector(self):
        P = np.diag([1., 1., 0.])
        self.assertTrue(is_projector(P))

    def test_identity_is_projector(self):
        self.assertTrue(is_projector(np.eye(3)))

    def test_not_hermitian(self):
        M = np.array([[0., 1.], [0., 0.]])
        self.assertFalse(is_projector(M))

    def test_hermitian_not_idempotent(self):
        # 0.5 * I is Hermitian but (0.5 I)^2 = 0.25 I ≠ 0.5 I
        M = 0.5 * np.eye(3)
        self.assertFalse(is_projector(M))


class ComputationalEffectTester(BaseCase):

    def setUp(self):
        self.basis = BuiltinBasis('l2p1', 9)

    def test_is_projector(self):
        E = computational_effect(self.basis)
        self.assertTrue(is_projector(E))

    def test_rank_equals_2(self):
        # l2p1 has a 2-dimensional computational subspace
        E = computational_effect(self.basis)
        self.assertEqual(np.linalg.matrix_rank(E), 2)

    def test_idempotent(self):
        E = computational_effect(self.basis)
        self.assertArraysAlmostEqual(E @ E, E)


class ComputationalSuperketsTester(BaseCase):

    def setUp(self):
        self.basis    = BuiltinBasis('l2p1', 9)
        self.pp_basis = Basis.cast('pp', 4)

    def test_column_orthonormality(self):
        U = computational_superkets(self.basis)
        # For k=2 computational levels, we expect k^2 = 4 columns
        k_sq = U.shape[1]
        self.assertArraysAlmostEqual(U.T @ U, np.eye(k_sq))

    def test_nonleakage_returns_identity(self):
        U = computational_superkets(self.pp_basis)
        self.assertArraysAlmostEqual(U, np.eye(4))

    def test_span_contains_comp_state(self):
        # A density matrix with support only on the computational subspace
        # must lie in the column span of the computational superkets.
        basis = self.basis
        rho = np.zeros((3, 3), complex)
        rho[0, 0] = 0.7
        rho[1, 1] = 0.3
        rho_vec = stdmx_to_vec(rho, basis).real
        U   = computational_superkets(basis)
        proj = U @ (U.T @ rho_vec)
        self.assertArraysAlmostEqual(proj, rho_vec)


class ComputationalProjectorTester(BaseCase):

    def setUp(self):
        self.basis    = BuiltinBasis('l2p1', 9)
        self.pp_basis = Basis.cast('pp', 4)

    # --- 1-arg (basis-aware) form ---

    def test_idempotent_1arg(self):
        P = computational_projector(self.basis)
        self.assertArraysAlmostEqual(P @ P, P)

    def test_symmetric_1arg(self):
        P = computational_projector(self.basis)
        self.assertArraysAlmostEqual(P, P.T)

    def test_nonleakage_is_identity_1arg(self):
        P = computational_projector(self.pp_basis)
        self.assertArraysAlmostEqual(P, np.eye(4))

    # --- 3-arg (explicit dimensions) form ---

    def test_idempotent_3arg(self):
        P = computational_projector(2, 3, self.basis)
        self.assertArraysAlmostEqual(P @ P, P)

    def test_symmetric_3arg(self):
        P = computational_projector(2, 3, self.basis)
        self.assertArraysAlmostEqual(P, P.T)

    def test_d_equals_n_returns_identity(self):
        P = computational_projector(3, 3, self.basis)
        self.assertArraysAlmostEqual(P, np.eye(9))

    # --- consistency between forms ---

    def test_1arg_3arg_agree(self):
        P1 = computational_projector(self.basis)
        P3 = computational_projector(2, 3, self.basis)
        self.assertArraysAlmostEqual(P1, P3)

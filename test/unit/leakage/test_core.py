import numpy as np

from pygsti.baseobjs.basis import BuiltinBasis, Basis
from pygsti.tools.matrixtools import is_projector
from pygsti.tools import basistools as pgbt
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

    def test_l2p1_implies_leakage(self):
        val = self.l2p1.implies_leakage_modeling
        self.assertTrue(val)
        # Result is cached after the first access.
        self.assertTrue(hasattr(self.l2p1, '_implies_leakage'))
        self.assertEqual(self.l2p1.implies_leakage_modeling, val)

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

    def test_nondiagonal_rank1_projector(self):
        # |+><+| in C^2 — rank-1, mixes both standard-basis vectors
        P = 0.5 * np.array([[1., 1.], [1., 1.]])
        self.assertTrue(is_projector(P))

    def test_nondiagonal_rank2_projector(self):
        # Projector onto span{|+>, |2>} in C^3, where |+> = (|0>+|1>)/sqrt(2).
        # Off-diagonal in the standard basis.
        P = np.array([[0.5, 0.5, 0.],
                      [0.5, 0.5, 0.],
                      [0.,  0.,  1.]])
        self.assertTrue(is_projector(P))


class ComputationalEffectTester(BaseCase):

    def setUp(self):
        self.basis = BuiltinBasis('l2p1', 9)

    def test_l2p1_effect_properties(self):
        # l2p1 has a 2-dimensional computational subspace.
        E = computational_effect(self.basis)
        self.assertArraysAlmostEqual(E, E.conj().T)       # Hermitian
        self.assertArraysAlmostEqual(E @ E, E)             # idempotent
        self.assertEqual(np.linalg.matrix_rank(E), 2)     # rank 2
        self.assertAlmostEqual(np.trace(E).real, 2.0, places=10)  # tr = rank
        self.assertTrue(is_projector(E))


class ComputationalSuperketsTester(BaseCase):

    def setUp(self):
        self.basis    = BuiltinBasis('l2p1', 9)
        self.pp_basis = Basis.cast('pp', 4)

    def test_l2p1_superkets_properties(self):
        U = computational_superkets(self.basis)
        # k=2 computational levels → k^2=4 columns; must be column-orthonormal.
        self.assertArraysAlmostEqual(U.T @ U, np.eye(U.shape[1]))
        # A density matrix with support only on the computational subspace
        # must lie in the column span of U.
        rho = np.zeros((3, 3), complex)
        rho[0, 0] = 0.7
        rho[1, 1] = 0.3
        rho_vec = stdmx_to_vec(rho, self.basis).real
        self.assertArraysAlmostEqual(U @ (U.T @ rho_vec), rho_vec)

    def test_nonleakage_returns_identity(self):
        U = computational_superkets(self.pp_basis)
        self.assertArraysAlmostEqual(U, np.eye(4))

    def test_nondiagonal_explicit_projector(self):
        # Projector onto span{(|0>+|2>)/sqrt(2), |1>} in C^3 — rank-2, non-diagonal.
        # Exercises the QR path in computational_superkets with an input that mixes
        # standard-basis columns.
        E_nd = np.array([[0.5, 0., 0.5],
                         [0.,  1., 0. ],
                         [0.5, 0., 0.5]])
        U = computational_superkets(self.basis, E=E_nd)
        # k=2 computational levels → k^2=4 columns; columns must be orthonormal.
        self.assertArraysAlmostEqual(U.T @ U, np.eye(U.shape[1]))
        # A density matrix with support inside the computational subspace must lie
        # in the column span of U.  Use |+><+| where |+> = (|0>+|2>)/sqrt(2).
        rho = np.array([[0.5, 0., 0.5],
                        [0.,  0., 0. ],
                        [0.5, 0., 0.5]], dtype=complex)
        rho_vec = pgbt.stdmx_to_vec(rho, self.basis).real
        proj = U @ (U.T @ rho_vec)
        self.assertArraysAlmostEqual(proj, rho_vec)

    def test_explicit_nonprojector_E_raises(self):
        # diag(1, 0.5, 0) is not a scalar multiple of any projector: after the
        # internal normalization E *= (rank/trace) = (2/1.5), it becomes
        # diag(4/3, 2/3, 0), which is Hermitian but not idempotent.
        E_bad = np.diag([1., 0.5, 0.])
        with self.assertRaises(ValueError):
            computational_superkets(self.basis, E=E_bad)
        # Note: line 102 (non-Hermitian leakage basis) is unreachable from the
        # public API — all standard leakage bases (e.g. 'l2p1') are Hermitian.


class ComputationalProjectorTester(BaseCase):

    def setUp(self):
        self.basis    = BuiltinBasis('l2p1', 9)
        self.pp_basis = Basis.cast('pp', 4)

    def test_l2p1_projector_properties(self):
        # l2p1 with k=2 computational levels: P projects the 9-dim superop space
        # onto the k^2=4-dimensional subspace S[C].
        P = computational_projector(self.basis)
        self.assertArraysAlmostEqual(P @ P, P)             # idempotent
        self.assertArraysAlmostEqual(P, P.T)               # symmetric
        self.assertEqual(np.linalg.matrix_rank(P), 4)     # rank 4
        self.assertAlmostEqual(np.trace(P).real, 4.0, places=10)  # tr = rank

    def test_nonleakage_is_identity(self):
        P = computational_projector(self.pp_basis)
        self.assertArraysAlmostEqual(P, np.eye(4))

    def test_rank_and_trace(self):
        # For l2p1 (k=2 computational levels), P projects the 9-dim superop space
        # onto the k^2=4-dimensional subspace S[C].  So rank(P) = tr(P) = 4.
        P = computational_projector(self.basis)
        k_sq = 4
        self.assertEqual(np.linalg.matrix_rank(P), k_sq)
        self.assertAlmostEqual(np.trace(P).real, float(k_sq), places=10)


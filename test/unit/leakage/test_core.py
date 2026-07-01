import numpy as np

from pygsti.baseobjs.basis import BuiltinBasis, Basis, ExplicitBasis, TensorProdBasis
from pygsti.baseobjs.basisconstructors import lf_matrices
from pygsti.tools.matrixtools import is_projector
from pygsti.tools import basistools as pgbt
from pygsti.tools.basistools import stdmx_to_vec
import scipy.linalg as la

from pygsti.leakage.core import (
    computational_effect, computational_superkets, computational_projector,
    augment_for_leakage_modeling,
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

    def test_l2p1_labels(self):
        self.assertEqual(tuple(self.l2p1.labels),
                         ('C[I]', 'C[X]', 'C[Y]', 'C[Z]', 'L[X_02]', 'L[X_12]', 'L[Y_02]', 'L[Y_12]', 'L[I]'))

    def test_tensor_product_implies_leakage(self):
        # TensorProdBasis concatenates factor labels, so the identity-candidate label of
        # pp ⨂ l2p1 is 'IC[I]'.  Detection must handle this mixed form.
        tp = TensorProdBasis((BuiltinBasis('pp', 4), BuiltinBasis('l2p1', 9)))
        self.assertTrue(tp.implies_leakage_modeling)
        E = computational_effect(tp)
        self.assertArraysAlmostEqual(E, np.diag([1., 1., 0., 1., 1., 0.]))


class LegacyLabelBackCompatTester(BaseCase):
    """
    Bases labeled under the pre-'C[...]' convention (computational identity labeled by a
    string of 'I' characters) must still be recognized.  The label set below is committed
    to git in docs/markdown/examples/Leakage-manual.md.
    """

    def setUp(self):
        self.legacy = ExplicitBasis(lf_matrices(3),
                                    labels=['I', 'X', 'Y', 'Z', 'LX0', 'LX1', 'LY0', 'LY1', 'L'],
                                    name='LegacyLeakageBasis')

    def test_implies_leakage_modeling(self):
        self.assertTrue(self.legacy.implies_leakage_modeling)

    def test_computational_effect(self):
        E = computational_effect(self.legacy)
        self.assertArraysAlmostEqual(E, np.diag([1., 1., 0.]))

    def test_matches_builtin_l2p1(self):
        # Same elements as the (relabeled) builtin, so all derived leakage objects agree.
        l2p1 = BuiltinBasis('l2p1', 9)
        U_legacy = computational_superkets(self.legacy)
        U_new    = computational_superkets(l2p1)
        P_legacy = U_legacy @ U_legacy.T
        P_new    = U_new @ U_new.T
        self.assertArraysAlmostEqual(P_legacy, P_new)

    def test_legacy_tensor_product(self):
        tp = TensorProdBasis((BuiltinBasis('pp', 4), self.legacy))
        self.assertTrue(tp.implies_leakage_modeling)
        E = computational_effect(tp)
        self.assertArraysAlmostEqual(E, np.diag([1., 1., 0., 1., 1., 0.]))



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


class AugmentForLeakageModelingTester(BaseCase):

    def setUp(self):
        self.l2p1  = BuiltinBasis('l2p1', 9)
        self.gm9   = Basis.cast('gm', 9)
        self.E     = computational_effect(self.l2p1)   # rank-2 projector in 3×3 space
        self.k     = 2
        self.b_aug = augment_for_leakage_modeling(self.gm9, self.E)

    # --- structural / happy-path tests ---

    def test_implies_leakage_modeling(self):
        self.assertTrue(self.b_aug.implies_leakage_modeling)

    def test_element_count(self):
        self.assertEqual(len(self.b_aug.elements), self.gm9.dim)

    def test_first_element_proportional_to_E(self):
        E_norm = self.E / la.norm(self.E)
        self.assertArraysAlmostEqual(self.b_aug.elements[0], E_norm)

    def test_first_label_is_bracketed_all_I(self):
        # The computational identity is labeled 'C[I...I]'; for a gm source basis
        # (identity label 'I') that's exactly 'C[I]'.
        lbl = self.b_aug.labels[0]
        self.assertTrue(lbl.startswith('C[') and lbl.endswith(']'))
        inner = lbl[2:-1]
        self.assertEqual(inner.strip('I'), '')
        self.assertGreater(len(inner), 0)
        self.assertEqual(lbl, 'C[I]')

    def test_cs_labels_format(self):
        # Elements 1 ... k²−1 are labeled 'C[lbl]' with lbl a label of the source basis.
        for lbl in self.b_aug.labels[1:self.k**2]:
            self.assertTrue(lbl.startswith('C[') and lbl.endswith(']'),
                            msg=f"expected 'C[...]' label, got {lbl!r}")
            self.assertIn(lbl[2:-1], self.gm9.labels)

    def test_elements_are_normalized(self):
        for elem in self.b_aug.elements:
            self.assertAlmostEqual(la.norm(elem), 1.0, places=12)

    def test_cs_elements_span_mc(self):
        # Any matrix E A E (supported on C) must lie in the span of the first k² elements.
        k = self.k
        V = np.column_stack([e.ravel() for e in self.b_aug.elements[:k**2]])
        A = self.E @ np.diag([1., 0., 0.]) @ self.E
        a_vec = A.ravel()
        proj = V @ np.linalg.lstsq(V, a_vec, rcond=None)[0]
        self.assertArraysAlmostEqual(proj, a_vec)

    def test_oc_elements_span_mc_perp(self):
        # The complement projector I - E lives entirely in M[C]^⊥.
        k = self.k
        W = np.column_stack([e.ravel() for e in self.b_aug.elements[k**2:]])
        E_comp = np.eye(self.E.shape[0]) - self.E
        a_vec  = E_comp.ravel()
        proj   = W @ np.linalg.lstsq(W, a_vec, rcond=None)[0]
        self.assertArraysAlmostEqual(proj, a_vec)

    def test_leakage_labels_format(self):
        # Every label after the first k² must start with 'L'.
        for lbl in self.b_aug.labels[self.k**2:]:
            self.assertTrue(lbl.startswith('L'), msg=f"expected 'L...' label, got {lbl!r}")

    def test_final_label_is_LI(self):
        self.assertEqual(self.b_aug.labels[-1], 'L[I]')

    def test_output_name(self):
        self.assertIn('Leakage augmented', self.b_aug.name)

    # --- non-diagonal projector (closes the commented-out test above) ---

    def test_nondiagonal_projector(self):
        # Rank-2 projector onto span{(|0>+|2>)/sqrt(2), |1>} in C^3 — non-diagonal.
        E_nd = np.array([[0.5, 0., 0.5],
                         [0.,  1., 0. ],
                         [0.5, 0., 0.5]])
        b_nd = augment_for_leakage_modeling(self.gm9, E_nd)
        self.assertTrue(b_nd.implies_leakage_modeling)
        self.assertArraysAlmostEqual(computational_effect(b_nd), E_nd)
        U = computational_superkets(b_nd)
        self.assertArraysAlmostEqual(U.T @ U, np.eye(4))

    # --- input validation ---

    def test_invalid_complex_E_raises(self):
        E_complex = self.E.astype(complex)
        E_complex[0, 1] += 1e-5j
        with self.assertRaises(ValueError):
            augment_for_leakage_modeling(self.gm9, E_complex)

    def test_invalid_non_projector_raises(self):
        # Diagonal matrix with eigenvalues 1, 0.5, 0 — Hermitian but not a projector at
        # any scale (eigenvalues can't be simultaneously scaled to lie in {0, 1}).
        E_bad = np.diag([1., 0.5, 0.])
        with self.assertRaises(ValueError):
            augment_for_leakage_modeling(self.gm9, E_bad)

    def test_invalid_non_hermitian_raises(self):
        E_bad = np.array([[1., 1., 0.],
                          [0., 1., 0.],
                          [0., 0., 0.]])
        with self.assertRaises(ValueError):
            augment_for_leakage_modeling(self.gm9, E_bad)

import numpy as np
import scipy.sparse as sps

from pygsti.tools import matrixmod2
from ..util import BaseCase


def _is_invertible_mod2_exact(m):
    """Reference implementation: exact GF(2) invertibility via Gaussian elimination."""
    m = np.array(m, dtype='int')
    n = m.shape[0]
    r = matrixmod2.gaussian_elimination_mod2(m)
    return np.array_equal(r, np.eye(n, dtype='int'))


class MatrixMod2BasicOpsTester(BaseCase):
    def test_dot_mod2(self):
        m1 = np.array([[1, 1], [1, 0]])
        m2 = np.array([[1, 0], [1, 1]])
        expected = np.array([[0, 1], [1, 0]])
        self.assertArraysEqual(matrixmod2.dot_mod2(m1, m2), expected)

    def test_multidot_mod2(self):
        m1 = np.array([[1, 1], [1, 0]])
        m2 = np.array([[1, 0], [1, 1]])
        m3 = np.eye(2, dtype=int)
        expected = np.array([[0, 1], [1, 0]])
        self.assertArraysEqual(matrixmod2.multidot_mod2([m1, m2, m3]), expected)

    def test_det_mod2(self):
        invertible = np.array([[1, 1], [0, 1]])
        singular = np.array([[1, 1], [1, 1]])
        even_real_det = np.array([[2, 0], [0, 1]])  # real det = 2, reduces to 0 mod 2
        self.assertAlmostEqual(matrixmod2.det_mod2(invertible), 1.0)
        self.assertAlmostEqual(matrixmod2.det_mod2(singular), 0.0)
        self.assertAlmostEqual(matrixmod2.det_mod2(even_real_det), 0.0)

    def test_det_mod2_large_matrices(self):
        # Regression guard: det_mod2 previously used np.linalg.det (floating-point
        # LU decomposition), which silently produced incorrect results for n >~ 40-50
        # due to floating-point overflow/rounding on the astronomically large real
        # determinants of binary matrices. Verify against an exact reference
        # (Gaussian elimination mod 2) at sizes where the old implementation was
        # known to fail.
        rand_state = np.random.RandomState(2024)
        for n in (50, 60, 70, 80):
            for _ in range(5):
                M = rand_state.randint(0, 2, size=(n, n))
                expected = 1.0 if _is_invertible_mod2_exact(M) else 0.0
                self.assertAlmostEqual(matrixmod2.det_mod2(M), expected)

    def test_matrix_directsum(self):
        m1 = np.array([[1]])
        m2 = np.array([[0, 1], [1, 0]])
        expected = np.array([
            [1, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
        ])
        self.assertArraysEqual(matrixmod2.matrix_directsum(m1, m2), expected)

    def test_diagonal_as_vec(self):
        m = np.array([[1, 2], [3, 4]])
        self.assertArraysEqual(matrixmod2.diagonal_as_vec(m), np.array([1, 4]))

    def test_diagonal_as_vec_rejects_sparse(self):
        m = sps.csr_matrix(np.array([[1, 2], [3, 4]]))
        with self.assertRaises(TypeError):
            matrixmod2.diagonal_as_vec(m)

    def test_diagonal_as_matrix(self):
        m = np.array([[1, 2], [3, 4]])
        expected = np.array([[1, 0], [0, 4]])
        self.assertArraysEqual(matrixmod2.diagonal_as_matrix(m), expected)

    def test_diagonal_as_matrix_rejects_sparse(self):
        m = sps.csr_matrix(np.array([[1, 2], [3, 4]]))
        with self.assertRaises(TypeError):
            matrixmod2.diagonal_as_matrix(m)

    def test_strictly_upper_triangle(self):
        m = np.array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ])
        expected = np.array([
            [0, 2, 3],
            [0, 0, 6],
            [0, 0, 0],
        ])
        self.assertArraysEqual(matrixmod2.strictly_upper_triangle(m), expected)

    def test_strictly_upper_triangle_rejects_sparse(self):
        m = sps.csr_matrix(np.array([[1, 2], [3, 4]]))
        with self.assertRaises(TypeError):
            matrixmod2.strictly_upper_triangle(m)


class MatrixMod2LinearAlgebraTester(BaseCase):
    def test_gaussian_elimination_mod2(self):
        m = np.array([[1, 1], [0, 1]])
        expected = np.eye(2, dtype=int)
        self.assertArraysEqual(matrixmod2.gaussian_elimination_mod2(m), expected)

        # An augmented matrix [A | I] should reduce to [I | A^-1]
        A = np.array([[1, 1], [0, 1]])
        augmented = np.append(A, np.eye(2, dtype=int), 1)
        expected_augmented = np.array([[1, 0, 1, 1], [0, 1, 0, 1]])
        self.assertArraysEqual(matrixmod2.gaussian_elimination_mod2(augmented), expected_augmented)

    def test_inv_mod2(self):
        A = np.array([[1, 1], [0, 1]])
        Ainv = matrixmod2.inv_mod2(A)
        identity = np.eye(2, dtype=int)
        self.assertArraysEqual(matrixmod2.dot_mod2(A, Ainv), identity)
        self.assertArraysEqual(matrixmod2.dot_mod2(Ainv, A), identity)

        # Identity matrix is its own inverse
        I = np.eye(3, dtype=int)
        self.assertArraysEqual(matrixmod2.inv_mod2(I), I)

    def test_Axb_mod2(self):
        A = np.array([[1, 1], [0, 1]])
        b = np.array([1, 0])
        x = matrixmod2.Axb_mod2(A, b)
        check = matrixmod2.dot_mod2(A, x).flatten() % 2
        self.assertArraysEqual(check, b)


class MatrixMod2RandomTester(BaseCase):
    def test_random_bitstring(self):
        for parity in (0, 1):
            rand_state = np.random.RandomState(1234)
            bitstring = matrixmod2.random_bitstring(5, parity, rand_state=rand_state)
            self.assertEqual(len(bitstring), 5)
            self.assertEqual(np.sum(bitstring) % 2, parity)

    def test_random_invertable_matrix(self):
        rand_state = np.random.RandomState(2)
        M = matrixmod2.random_invertable_matrix(4, rand_state=rand_state)
        self.assertEqual(M.shape, (4, 4))
        self.assertAlmostEqual(matrixmod2.det_mod2(M), 1.0)

    def test_random_symmetric_invertable_matrix(self):
        rand_state = np.random.RandomState(3)
        S = matrixmod2.random_symmetric_invertable_matrix(4, rand_state=rand_state)
        self.assertArraysEqual(S, S.T)
        self.assertAlmostEqual(matrixmod2.det_mod2(S), 1.0)


class AlbertFactorizationTester(BaseCase):
    def test_onesify(self):
        rand_state = np.random.RandomState(5)
        a = matrixmod2.random_symmetric_invertable_matrix(3, rand_state=rand_state)
        N = matrixmod2.onesify(a, rand_state=rand_state)

        # N should be invertible mod 2
        self.assertAlmostEqual(matrixmod2.det_mod2(N), 1.0)

        # N a N.T should have ones along the diagonal
        aa = matrixmod2.multidot_mod2([N, a, N.T])
        self.assertArraysEqual(matrixmod2.diagonal_as_vec(aa), np.ones(3, dtype=int))

    def test_permute_top(self):
        a = np.array([
            [1, 0, 1],
            [0, 1, 0],
            [1, 0, 0],
        ])
        aa, P = matrixmod2.permute_top(a, 2)

        # P should be a valid permutation matrix that swaps rows/cols 0 and 2
        expected_P = np.array([
            [0, 0, 1],
            [0, 1, 0],
            [1, 0, 0],
        ])
        self.assertArraysEqual(P, expected_P)

        expected_aa = np.array([
            [0, 0, 1],
            [0, 1, 0],
            [1, 0, 1],
        ])
        self.assertArraysEqual(aa, expected_aa)

    def test_fix_top(self):
        # trivial 1x1 case returns the identity
        self.assertArraysEqual(matrixmod2.fix_top(np.array([[1]])), np.eye(1, dtype=int))

        rand_state = np.random.RandomState(7)
        a = matrixmod2.random_symmetric_invertable_matrix(3, rand_state=rand_state)
        N = matrixmod2.onesify(a, rand_state=rand_state)
        aa = matrixmod2.multidot_mod2([N, a, N.T])

        P = matrixmod2.fix_top(aa)
        permuted = matrixmod2.multidot_mod2([P, aa, P.T])
        B = permuted[1:, 1:]
        self.assertAlmostEqual(matrixmod2.det_mod2(B), 1.0)

    def test_proper_permutation(self):
        rand_state = np.random.RandomState(7)
        a = matrixmod2.random_symmetric_invertable_matrix(3, rand_state=rand_state)
        N = matrixmod2.onesify(a, rand_state=rand_state)
        aa = matrixmod2.multidot_mod2([N, a, N.T])

        P = matrixmod2.proper_permutation(aa)
        A = matrixmod2.multidot_mod2([P, aa, P.T])
        self.assertTrue(matrixmod2._check_proper_permutation(A))

    def test_albert_factor(self):
        for n in (2, 3, 4):
            rand_state = np.random.RandomState(100 + n)
            d = matrixmod2.random_symmetric_invertable_matrix(n, rand_state=rand_state)
            L = matrixmod2.albert_factor(d, rand_state=rand_state)
            product = matrixmod2.multidot_mod2([L, L.T])
            self.assertArraysEqual(product, d)

    def test_albert_factor_large_n(self):
        n = 100
        rand_state = np.random.RandomState(1001)
        d = matrixmod2.random_symmetric_invertable_matrix(n, rand_state=rand_state)
        L = matrixmod2.albert_factor(d, rand_state=rand_state)
        product = matrixmod2.multidot_mod2([L, L.T])
        self.assertArraysEqual(product, d)

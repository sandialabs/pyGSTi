import numpy as np
import scipy.linalg as la
import scipy.sparse as spar

import pygsti.tools.matrixtools as mt
from ..util import BaseCase


class MatrixToolsTester(BaseCase):

    def test_eign(self):

        gen = np.random.default_rng(0)
        n = 5
        TOL = 1e-14

        def reconstruct(_eigvals, _eigvecs):
            return _eigvecs @ np.diag(_eigvals) @ _eigvecs.T.conj()

        def check_matrix(mat):
            mat_norm = la.norm(mat, ord=2)
            eigvals, eigvecs = mt.eign(mat)
            mat_recon = reconstruct(eigvals, eigvecs)
            error = mat - mat_recon
            self.assertLessEqual(la.norm(error), np.sqrt(n) * TOL * mat_norm)
            return

        G_real     = gen.standard_normal(size=(n, n))
        orthogonal = la.qr(G_real)[0]
        symmetric  = G_real + G_real.T

        G_imaginary = gen.standard_normal(size=(n, n))
        G_complex   = G_real + 1j * G_imaginary
        unitary     = la.qr(G_complex)[0]
        hermitian   = G_complex + G_complex.T.conj()
        normal      = reconstruct(np.diag(G_complex), unitary)

        check_matrix(orthogonal)
        check_matrix(symmetric)
        check_matrix(unitary)
        check_matrix(hermitian)
        check_matrix(normal)

        return

    def test_is_hermitian(self):
        herm_mx = np.array([[ 1, 1+2j],
                            [1-2j, 3]], 'complex')
        non_herm_mx = np.array([[ 1, 4+2j],
                                [1+2j, 3]], 'complex')
        self.assertTrue(mt.is_hermitian(herm_mx))
        self.assertFalse(mt.is_hermitian(non_herm_mx))

    def test_is_pos_def(self):
        pos_mx = np.array([[ 4, 0.2],
                           [0.1, 3]], 'complex')
        non_pos_mx = np.array([[ 0, 1],
                               [1, 0]], 'complex')
        self.assertTrue(mt.is_pos_def(pos_mx))
        self.assertFalse(mt.is_pos_def(non_pos_mx))

    def test_mx_to_string(self):
        mx = np.array([[ 1, 1+2j],
                       [1-2j, 3]], 'complex')

        s = mt.mx_to_string(mx)

        ls = s.split('\n')[:-1] # trim empty last line
        mx2 = np.zeros_like(mx)
        for i, row in enumerate(ls):
            entries = row.split()
            for j in range(len(entries) // 2):
                mx2[i, j] = float(entries[2*j]) + 1j*float(entries[2*j+1][:-1]) # trim 'j'

        self.assertArraysAlmostEqual(mx, mx2)

    def test_is_valid_density_mx(self):
        density_mx = np.array([[ 0.9,   0],
                               [   0, 0.1]], 'complex')
        non_density_mx = np.array([[ 2.0, 1.0],
                                   [-1.0,   0]], 'complex')
        self.assertTrue(mt.is_valid_density_mx(density_mx))
        self.assertFalse(mt.is_valid_density_mx(non_density_mx))

    def test_nullspace(self):
        a = np.array([[1, 1], [1, 1]])
        #print("Nullspace = ", mt.nullspace(a))
        expected = np.array(
            [[ 0.70710678],
             [-0.70710678]]
        )

        diff1 = np.linalg.norm(mt.nullspace(a) - expected)
        diff2 = np.linalg.norm(mt.nullspace(a) + expected)  # -1*expected is OK too (just an eigenvector)
        self.assertTrue(np.isclose(diff1, 0) or np.isclose(diff2, 0))

        diff1 = np.linalg.norm(mt.nullspace_qr(a) - expected)
        diff2 = np.linalg.norm(mt.nullspace_qr(a) + expected)  # -1*expected is OK too (just an eigenvector)
        self.assertTrue(np.isclose(diff1, 0) or np.isclose(diff2, 0))

        #mt.print_mx(a)

    def test_matrix_log(self):
        M = np.array([[-1, 0], [0, -1]], 'complex')  # degenerate negative evals
        logM = mt.real_matrix_log(M, action_if_imaginary="raise", tol=1e-6)
        self.assertArraysAlmostEqual(la.expm(logM), M)

        M = np.array([[-1, 1e-10], [1e-10, -1]], 'complex')  # degenerate negative evals, but will generate complex evecs
        logM = mt.real_matrix_log(M, action_if_imaginary="raise", tol=1e-6)
        self.assertArraysAlmostEqual(la.expm(logM), M)

        with self.assertRaises(ValueError):
            M = np.array([[1, 0], [0, -1]], 'd')  # a negative *unparied* eigenvalue => log may be imaginary
            mt.real_matrix_log(M, action_if_imaginary="raise", tol=1e-6)

        M = np.array([[1, 0], [0, -1]], 'd')  # a negative *unparied* eigenvalue => log may be imaginary
        logM = mt.real_matrix_log(M, action_if_imaginary="ignore", tol=1e-6)
        self.assertArraysAlmostEqual(la.expm(logM), M)

    def test_matrix_log_warns_on_imaginary(self):
        M = np.array([[1, 0], [0, -1]], 'd')
        self.assertWarns(Warning, mt.real_matrix_log, M, action_if_imaginary="warn", tol=1e-6)

    def test_matrix_log_raises_on_imaginary(self):
        M = np.array([[1, 0], [0, -1]], 'd')
        with self.assertRaises(ValueError):
            mt.real_matrix_log(M, action_if_imaginary="raise", tol=1e-6)

    def test_matrix_log_raises_on_invalid_action(self):
        M = np.array([[1, 0], [0, -1]], 'd')
        with self.assertRaises(AssertionError):
            mt.real_matrix_log(M, action_if_imaginary="foobar", tol=1e-6)

    def test_matrix_log_raise_on_no_real_log(self):
        a = np.array([[1, 1], [1, 1]])
        with self.assertRaises(AssertionError):
            mt.real_matrix_log(a)

    def test_minweight_match(self):
        a = np.array([1, 2, 3, 4], 'd')
        b = np.array([3.1, 2.1, 4.1, 1.1], 'd')
        expectedPairs = [(0, 3), (1, 1), (2, 0), (3, 2)]  # (i,j) indices into a & b

        wts = mt.minweight_match(a, b, metricfn=None, return_pairs=False,
                                 pass_indices_to_metricfn=False)
        wts, pairs = mt.minweight_match(a, b, metricfn=None, return_pairs=True,
                                        pass_indices_to_metricfn=False)
        self.assertEqual(set(pairs), set(expectedPairs))

        def fn(x, y): return abs(x - y)
        wts, pairs = mt.minweight_match(a, b, metricfn=fn, return_pairs=True,
                                        pass_indices_to_metricfn=False)
        self.assertEqual(set(pairs), set(expectedPairs))

        def fn(i, j): return abs(a[i] - b[j])
        wts, pairs = mt.minweight_match(a, b, metricfn=fn, return_pairs=True,
                                        pass_indices_to_metricfn=True)
        self.assertEqual(set(pairs), set(expectedPairs))

    def test_fancy_assignment(self):
        a = np.zeros((4, 4, 4), 'd')
        twoByTwo = np.ones((2, 2), 'd')

        #NOTEs from commit message motivating why we need this:
        # a = np.zeros((3,3,3))
        # a[:,1:2,1:3].shape == (3,1,2) # good!
        # a[0,:,1:3].shape == (3,2) #good!
        # a[0,:,[1,2]].shape == (2,3) # ?? (broacasting ':' makes this like a[0,[1,2]])
        # a[:,[1,2],[1,2]].shape == (3,2) # ?? not (3,2,2) b/c lists broadcast
        # a[:,[1],[1,2]].shape == (3,2) # ?? not (3,1,2) b/c lists broadcast
        # a[:,[1,2],[0,1,2]].shape == ERROR b/c [1,2] can't broadcast to [0,1,2]!

        #simple integer indices
        mt._fas(a, (0, 0, 0), 4.5)  # a[0,0,0] = 4.5
        self.assertAlmostEqual(a[0, 0, 0], 4.5)

        mt._fas(a, (0, 0, 0), 4.5, add=True)  # a[0,0,0] += 4.5
        self.assertAlmostEqual(a[0, 0, 0], 9.0)

        #still simple: mix of slices and integers
        mt._fas(a, (slice(0, 2), slice(0, 2), 0), twoByTwo)  # a[0:2,0:2,0] = twoByTwo
        self.assertArraysAlmostEqual(a[0:2, 0:2, 0], twoByTwo)

        #complex case: some/all indices are integer arrays
        mt._fas(a, ([0, 1], [0, 1], 0), twoByTwo[:, :])  # a[0:2,0:2,0] = twoByTwo - but a[[0,1],[0,1],0] wouldn't do this!
        self.assertArraysAlmostEqual(a[0:2, 0:2, 0], twoByTwo)

        mt._fas(a, ([0, 1], [0, 1], 0), twoByTwo[:, :], add=True)  # a[0:2,0:2,0] = twoByTwo - but a[[0,1],[0,1],0] wouldn't do this!
        self.assertArraysAlmostEqual(a[0:2, 0:2, 0], 2 * twoByTwo)

        # Fancy indexing (without assignment)
        self.assertEqual(mt._findx(a, (0, 0, 0)).shape, ())  # (1,1,1))
        self.assertEqual(mt._findx(a, (slice(0, 2), slice(0, 2), slice(0, 2))).shape, (2, 2, 2))
        self.assertEqual(mt._findx(a, (slice(0, 2), slice(0, 2), 0)).shape, (2, 2))
        self.assertEqual(mt._findx(a, ([0, 1], [0, 1], 0)).shape, (2, 2))
        self.assertEqual(mt._findx(a, ([], [0, 1], 0)).shape, (0, 2))

    def test_safe_ops(self):
        mx = np.array([[1+1j, 0],
                       [2+2j, 3+3j]], 'complex')
        smx = spar.csr_matrix(mx)
        smx_lil = spar.lil_matrix(mx)  # currently unsupported

        r = mt.safe_real(mx, inplace=False)
        self.assertArraysAlmostEqual(r, np.real(mx))
        i = mt.safe_imag(mx, inplace=False)
        self.assertArraysAlmostEqual(i, np.imag(mx))

        r = mt.safe_real(smx, inplace=False)
        self.assertArraysAlmostEqual(r.toarray(), np.real(mx))
        i = mt.safe_imag(smx, inplace=False)
        self.assertArraysAlmostEqual(i.toarray(), np.imag(mx))

        with self.assertRaises(NotImplementedError):
            mt.safe_real(smx_lil, inplace=False)
        with self.assertRaises(NotImplementedError):
            mt.safe_imag(smx_lil, inplace=False)

        with self.assertRaises(AssertionError):
            mt.safe_real(mx, check=True)
        with self.assertRaises(AssertionError):
            mt.safe_imag(mx, check=True)

        M = mx.copy(); M = mt.safe_real(M, inplace=True)
        self.assertArraysAlmostEqual(M, np.real(mx))
        M = mx.copy(); M = mt.safe_imag(M, inplace=True)
        self.assertArraysAlmostEqual(M, np.imag(mx))

        M = smx.copy(); M = mt.safe_real(M, inplace=True)
        self.assertArraysAlmostEqual(M.toarray(), np.real(mx))
        M = smx.copy(); M = mt.safe_imag(M, inplace=True)
        self.assertArraysAlmostEqual(M.toarray(), np.imag(mx))

    def test_fast_expm(self):
        mx = np.array([[1, 2],
                       [2, 3]], 'd')
        A = spar.csr_matrix(mx)
        A, mu, m_star, s, eta = mt.expm_multiply_prep(A)
        tol = 1e-6

        B = np.array([1, 1], 'd')
        expA = mt._custom_expm_multiply_simple_core(A, B, mu, m_star, s, tol, eta)

        sp_expA = np.inner(la.expm(mx), B)
        self.assertArraysAlmostEqual(expA, sp_expA)

    def test_fast_expm_raises_on_non_square(self):
        nonSq = np.array([[1, 2, 4],
                          [2, 3, 5]], 'd')
        N = spar.csr_matrix(nonSq)

        with self.assertRaises(ValueError):
            mt.expm_multiply_prep(N)

    def test_complex_compare(self):
        self.assertEqual(mt.complex_compare(1.0 + 2.0j, 1.0 + 2.0j), 0)  # ==
        self.assertEqual(mt.complex_compare(1.0 + 2.0j, 2.0 + 2.0j), -1)  # real a < real b
        self.assertEqual(mt.complex_compare(1.0 + 2.0j, 0.5 + 2.0j), +1)  # real a > real b
        self.assertEqual(mt.complex_compare(1.0 + 2.0j, 1.0 + 3.0j), -1)  # imag a < imag b
        self.assertEqual(mt.complex_compare(1.0 + 2.0j, 1.0 + 1.0j), +1)  # imag a > imag b

    def test_prime_factors(self):
        self.assertEqual(mt.prime_factors(7), [7])
        self.assertEqual(mt.prime_factors(10), [2, 5])
        self.assertEqual(mt.prime_factors(12), [2, 2, 3])

import numpy as np
import scipy.linalg as spl
import scipy.sparse as sps

import pygsti.tools.matrixtools as mt
from ..util import BaseCase


class MatrixToolsTester(BaseCase):

    def test_is_hermitian(self):
        herm_mx = np.array([[ 1, 1+2j],
                            [1-2j, 3]], 'complex')
        non_herm_mx = np.array([[ 1, 4+2j],
                                [1+2j, 3]], 'complex')
        self.assertTrue(mt.is_hermitian(herm_mx))
        self.assertFalse(mt.is_hermitian(non_herm_mx))

    def test_is_pos_def(self):
        pos_mx = np.array([[ 4.0, 0.2],
                            [0.2, 3.0]], 'complex')
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
        self.assertArraysAlmostEqual(spl.expm(logM), M)

        M = np.array([[-1, 1e-10], [1e-10, -1]], 'complex')  # degenerate negative evals, but will generate complex evecs
        logM = mt.real_matrix_log(M, action_if_imaginary="raise", tol=1e-6)
        self.assertArraysAlmostEqual(spl.expm(logM), M)

        with self.assertRaises(ValueError):
            M = np.array([[1, 0], [0, -1]], 'd')  # a negative *unparied* eigenvalue => log may be imaginary
            mt.real_matrix_log(M, action_if_imaginary="raise", tol=1e-6)

        M = np.array([[1, 0], [0, -1]], 'd')  # a negative *unparied* eigenvalue => log may be imaginary
        logM = mt.real_matrix_log(M, action_if_imaginary="ignore", tol=1e-6)
        self.assertArraysAlmostEqual(spl.expm(logM), M)

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
        import warnings
        a = np.array([[1, 1], [1, 1]])
        with self.assertRaises(AssertionError):
            with warnings.catch_warnings():
                warnings.filterwarnings(action="ignore", 
                    message="divide by zero encountered in log", category=RuntimeWarning
                )
                warnings.filterwarnings(action='ignore',
                    message='invalid value encountered in dot', category=RuntimeWarning
                )
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

    def test_fast_expm(self):
        mx = np.array([[1, 2],
                       [2, 3]], 'd')
        A = sps.csr_matrix(mx)
        A, mu, m_star, s, eta = mt.expm_multiply_prep(A)
        tol = 1e-6

        B = np.array([1, 1], 'd')
        expA = mt._custom_expm_multiply_simple_core(A, B, mu, m_star, s, tol, eta)

        sp_expA = np.inner(spl.expm(mx), B)
        self.assertArraysAlmostEqual(expA, sp_expA)

    def test_fast_expm_raises_on_non_square(self):
        nonSq = np.array([[1, 2, 4],
                          [2, 3, 5]], 'd')
        N = sps.csr_matrix(nonSq)

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

    def test_eigendecomposition_assume_normal(self):
        # A non-Hermitian but normal matrix: a unitary conjugate of a diagonal matrix.
        rng = np.random.default_rng(0)
        diag = np.array([1 + 2j, 3 - 1j, 0.5j])
        Q = spl.qr(rng.standard_normal((3, 3)) + 1j * rng.standard_normal((3, 3)))[0]
        M = Q @ np.diag(diag) @ Q.conj().T
        evecs, evals, inv_evecs = mt.eigendecomposition(M, assume_normal=True)
        self.assertArraysAlmostEqual(evecs @ np.diag(evals) @ inv_evecs, M)
        self.assertArraysAlmostEqual(evecs @ evecs.conj().T, np.eye(3))
        self.assertArraysAlmostEqual(inv_evecs, evecs.conj().T)

    def test_eigendecomposition_assume_normal_matches_assume_hermitian_for_hermitian_input(self):
        rng = np.random.default_rng(0)
        A = rng.standard_normal((4, 4)) + 1j * rng.standard_normal((4, 4))
        H = A + A.conj().T
        evecs_n, evals_n, _ = mt.eigendecomposition(H, assume_normal=True)
        recon_n = evecs_n @ np.diag(evals_n) @ evecs_n.conj().T
        self.assertArraysAlmostEqual(recon_n, H)

    def test_eigendecomposition_assume_normal_raises_on_non_normal_input(self):
        N = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0.5]], dtype=complex)
        # N is not normal: N @ N.conj().T != N.conj().T @ N
        self.assertFalse(np.allclose(N @ N.conj().T, N.conj().T @ N))
        with self.assertRaises(ValueError):
            mt.eigendecomposition(N, assume_normal=True)

    def test_eigenvalues_assume_normal_matches_full_eig(self):
        # Regression test: eigenvalues(assume_normal=True)'s Schur-decomposition path
        # used to read diag(Z) (the Schur decomposition's unitary factor) instead of
        # diag(T) (the triangular factor whose diagonal actually holds the
        # eigenvalues) -- a pre-existing bug on develop, independent of the
        # assume_hermitian-vs-assume_normal inference fixed in eigendecomposition
        # above. Use a normal, non-Hermitian matrix so assume_normal=True actually
        # exercises the Schur path (a Hermitian input would take the eigh path
        # instead, per eigenvalues' own Hermiticity-first inference).
        rng = np.random.default_rng(1)
        diag = np.array([1 + 2j, 3 - 1j, 0.5j])
        Q = spl.qr(rng.standard_normal((3, 3)) + 1j * rng.standard_normal((3, 3)))[0]
        M = Q @ np.diag(diag) @ Q.conj().T
        self.assertFalse(np.allclose(M, M.conj().T))

        actual = mt.eigenvalues(M, assume_normal=True)
        expected = np.linalg.eigvals(M)

        def sort_key(z):
            return (round(z.real, 8), round(z.imag, 8))

        self.assertArraysAlmostEqual(
            np.array(sorted(actual, key=sort_key)), np.array(sorted(expected, key=sort_key)))


def _pp_computational_superket(zvals):
    """
    Independent reference for the Pauli-product super-ket of a computational
    basis state, built by explicit Kronecker product.

    |z><z| = (I + (-1)**z Z) / 2, whose Pauli-product coefficient vector
    (normalized so the PP basis is orthonormal) is [1, 0, 0, (-1)**z]/sqrt(2).
    Deliberately does not share any code with the routines under test.
    """
    vec = np.array([1.0])
    for z in zvals:
        vec = np.kron(vec, np.array([1, 0, 0, 1 - 2 * z], 'd') / np.sqrt(2))
    return vec


def _zvals_to_int(zvals):
    """Encode z-values little-endian, matching EffectRepComputational."""
    return sum(z << i for i, z in enumerate(zvals))


class ZvalsInt64Tester(BaseCase):
    """
    Cover ``zvals_int64_to_dense`` / ``zvals_int64_probability``.

    ``zvals_int64_probability`` is an O(2**N) sparse inner product that replaces
    densifying the (mostly zero) effect vector to length 4**N and taking a full
    dense dot product. These tests check that optimization against an
    independently constructed dense reference, since being fast is not by itself
    evidence of being right.
    """

    ALL_ZVALS = [(), (0,), (1,), (0, 0), (0, 1), (1, 0), (1, 1),
                 (0, 1, 0), (1, 1, 0), (1, 0, 1), (1, 1, 1)]

    def test_to_dense_matches_kron_reference(self):
        for zvals in self.ALL_ZVALS:
            with self.subTest(zvals=zvals):
                actual = mt.zvals_int64_to_dense(_zvals_to_int(zvals), len(zvals))
                self.assertArraysAlmostEqual(actual, _pp_computational_superket(zvals))

    def test_probability_matches_kron_reference(self):
        rng = np.random.default_rng(0)
        for zvals in self.ALL_ZVALS:
            with self.subTest(zvals=zvals):
                state = rng.standard_normal(4 ** len(zvals))
                actual = mt.zvals_int64_probability(_zvals_to_int(zvals), len(zvals), state)
                expected = np.dot(_pp_computational_superket(zvals), state)
                self.assertAlmostEqual(actual, expected)

    def test_probability_agrees_with_densifying_then_dotting(self):
        # The specific claim the optimization makes: same answer as the O(4**N)
        # densify-then-dot it replaced.
        rng = np.random.default_rng(1)
        for zvals in self.ALL_ZVALS:
            with self.subTest(zvals=zvals):
                zvals_int, nq = _zvals_to_int(zvals), len(zvals)
                state = rng.standard_normal(4 ** nq)
                dense = mt.zvals_int64_to_dense(zvals_int, nq)
                self.assertAlmostEqual(mt.zvals_int64_probability(zvals_int, nq, state),
                                       np.dot(dense, state))

    def test_zero_qubits_is_the_scalar_case(self):
        # N == 0 is special-cased in both routines: the super-ket is the length-1
        # vector [1] and the "inner product" is just the single state element.
        self.assertArraysAlmostEqual(mt.zvals_int64_to_dense(0, 0), np.array([1.0]))
        self.assertAlmostEqual(mt.zvals_int64_probability(0, 0, np.array([2.5])), 2.5)

    def test_zero_qubits_honors_abs_elval(self):
        self.assertArraysAlmostEqual(mt.zvals_int64_to_dense(0, 0, None, False, 0.25),
                                     np.array([0.25]))
        self.assertAlmostEqual(mt.zvals_int64_probability(0, 0, np.array([2.0]), 0.25), 0.5)

    def test_omitted_abs_elval_matches_explicit_value(self):
        # abs_elval is a caller-supplied cache of 1/sqrt(2)**nqubits; omitting it
        # must compute the same thing.
        zvals = (1, 0, 1)
        zvals_int, nq = _zvals_to_int(zvals), len(zvals)
        state = np.random.default_rng(2).standard_normal(4 ** nq)
        abs_elval = 1 / (np.sqrt(2) ** nq)
        self.assertArraysAlmostEqual(mt.zvals_int64_to_dense(zvals_int, nq),
                                     mt.zvals_int64_to_dense(zvals_int, nq, None, False, abs_elval))
        self.assertAlmostEqual(mt.zvals_int64_probability(zvals_int, nq, state),
                               mt.zvals_int64_probability(zvals_int, nq, state, abs_elval))

    def test_to_dense_allocates_a_new_array_when_outvec_is_none(self):
        out = mt.zvals_int64_to_dense(0b01, 2, None)
        self.assertEqual(out.shape, (16,))
        # Successive calls must not alias one another (the internal index cache is
        # shared across calls, but the output buffer must not be).
        other = mt.zvals_int64_to_dense(0b10, 2, None)
        self.assertIsNot(out, other)
        self.assertFalse(np.allclose(out, other))

    def test_to_dense_clears_a_dirty_caller_supplied_outvec(self):
        # Without trust_outvec_sparsity, leftover data in the caller's buffer must
        # be zeroed; otherwise stale values survive at the structurally-zero indices.
        expected = mt.zvals_int64_to_dense(0b01, 2)
        dirty = np.full(16, 99.0)
        out = mt.zvals_int64_to_dense(0b01, 2, dirty, False)
        self.assertIs(out, dirty)
        self.assertArraysAlmostEqual(out, expected)

    def test_to_dense_trusts_sparsity_when_asked(self):
        expected = mt.zvals_int64_to_dense(0b01, 2)
        zeroed = np.zeros(16)
        out = mt.zvals_int64_to_dense(0b01, 2, zeroed, True)
        self.assertIs(out, zeroed)
        self.assertArraysAlmostEqual(out, expected)

    def test_index_cache_reuse_across_calls_is_correct(self):
        # The (r, final_indices) arrays are cached by nqubits and reused, so a
        # second call with the same nqubits but different zvals must not inherit
        # the first call's signs.
        mt._zvals_int64_to_dense_cache.clear()
        first = mt.zvals_int64_to_dense(0b01, 2)
        second = mt.zvals_int64_to_dense(0b10, 2)
        self.assertArraysAlmostEqual(first, _pp_computational_superket((1, 0)))
        self.assertArraysAlmostEqual(second, _pp_computational_superket((0, 1)))

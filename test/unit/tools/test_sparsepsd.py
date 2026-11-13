import importlib
import importlib.util
import numpy as np
from pygsti.tools import sparsepsd as sp
from pygsti.tools.graphs import sparsechol
from ..util import BaseCase


class SparsePsdExtractionTester(BaseCase):
    def test_numerical_module_exists(self):
        self.assertIsNotNone(importlib.util.find_spec('pygsti.tools.sparsepsd'))

    def test_component_masks_reject_nonmatrices(self):
        for mask in (np.array(0), np.array([0])):
            with self.subTest(shape=mask.shape), self.assertRaisesRegex(ValueError, 'square'):
                sp._sample_psd(np.zeros((1, 1)), np.random.default_rng(0), real_pattern=mask)

    def test_component_masks_validate_shape_and_containment(self):
        sp = importlib.import_module('pygsti.tools.sparsepsd')
        pattern = np.zeros((3, 3), dtype=bool)
        pattern[0, 1] = pattern[1, 0] = True
        for mask in (np.zeros((2, 2)), np.zeros((3, 4))):
            with self.assertRaisesRegex(ValueError, 'shape|square'):
                sp._sample_psd(pattern, np.random.default_rng(0), real_pattern=mask)
        with self.assertRaisesRegex(ValueError, 'contained|outside'):
            sp._sample_psd(pattern, np.random.default_rng(0), imag_pattern=np.ones((3, 3)))

    def test_independent_component_masks(self):
        sp = importlib.import_module('pygsti.tools.sparsepsd')
        pattern = ~np.eye(4, dtype=bool)
        real, imag = pattern.copy(), pattern.copy()
        real[0, 1] = real[1, 0] = False
        imag[2, 3] = imag[3, 2] = False
        k = sp._sample_psd(pattern, np.random.default_rng(19), real_pattern=real, imag_pattern=imag)
        self.assertArraysAlmostEqual(k, k.conj().T)
        self.assertEqual(k[0, 1].real, 0)
        self.assertEqual(k[2, 3].imag, 0)
        self.assertGreater(np.linalg.norm(k.imag), 0)
        self.assertGreaterEqual(np.linalg.eigvalsh(k).min(), -1e-12)

    def test_singular_anchor_policy_is_conservative(self):
        sp = importlib.import_module('pygsti.tools.sparsepsd')
        b, o = np.diag([1., 0.]), np.diag([1., 0.])
        self.assertGreaterEqual(np.linalg.eigvalsh(b + o).min(), 0)
        self.assertEqual(sp._psd_step_size(b, o), 0)


def _weight_pattern(dims, max_S_weight, max_CA_weight):
    """Graph on nonempty coordinate supports, with edges limited by their union size."""
    import itertools
    supports = [frozenset(i for i, x in enumerate(t) if x)
                for t in itertools.product(*[range(d * d) for d in dims])]
    supports = [s for s in supports if 0 < len(s) <= max_S_weight]
    n = len(supports)
    P = np.zeros((n, n), dtype=bool)
    for i, j in itertools.combinations(range(n), 2):
        P[i, j] = P[j, i] = len(supports[i] | supports[j]) <= max_CA_weight
    return P


class PsdSamplingTester(BaseCase):

    # (pattern, is chordal): complete, weight-limited (chordal), weight-3 C/A on four qubits (not chordal)
    PATTERNS = [(~np.eye(3, dtype=bool), True),
                (_weight_pattern((2, 2), 2, 1), True),
                (_weight_pattern((2, 3), 2, 1), True),
                (_weight_pattern((2, 2, 2, 2), 2, 3), False)]

    def _check_psd(self, K):
        self.assertArraysAlmostEqual(K, K.conj().T)
        self.assertGreaterEqual(np.linalg.eigvalsh(K)[0], -1e-12 * np.max(np.abs(np.diag(K))))

    def test_psd_and_support(self):
        for pattern, _ in self.PATTERNS:
            outside = ~pattern & ~np.eye(len(pattern), dtype=bool)
            for offdiag in ('complex', 'real', 'imag'):
                for seed in range(5):
                    K = sp._sample_psd(pattern, np.random.default_rng(seed), offdiag, scale=0.1)
                    self._check_psd(K)
                    self.assertEqual(np.count_nonzero(K[outside]), 0)
                    self.assertTrue(np.all(np.real(np.diag(K)) > 0))
                    self.assertEqual(np.count_nonzero(np.imag(np.diag(K))), 0)
                    offd = K[~np.eye(len(K), dtype=bool)]
                    if offdiag == 'real':
                        self.assertEqual(np.count_nonzero(np.imag(offd)), 0)
                    if offdiag == 'imag':
                        self.assertEqual(np.count_nonzero(np.real(offd)), 0)
                        self.assertGreater(np.count_nonzero(np.imag(offd)), 0)

    def test_legacy_seed_2_sample_is_now_cp(self):
        # random_CPTP_error_generator_rates(1, seed=2) produced an indefinite coefficient matrix.
        K = sp._sample_psd(~np.eye(3, dtype=bool), np.random.default_rng(2))
        self._check_psd(K)

    def test_chordal_patterns_need_no_shrinking(self):
        # A chordal pattern is ordered without fill, so K = L L^dag exactly and the shrink never runs;
        # a non-chordal pattern (and A-only sampling) needs it.
        from unittest import mock
        from pygsti.tools.graphs import sparsechol
        for pattern, chordal in self.PATTERNS:
            self.assertEqual(sparsechol.perfect_elimination_ordering(pattern) is not None, chordal)
            with mock.patch.object(sp, '_psd_step_size', side_effect=AssertionError('shrink called')) as shrink:
                for offdiag in ('complex', 'real'):
                    try:
                        sp._sample_psd(pattern, np.random.default_rng(0), offdiag)
                    except AssertionError:
                        pass
                self.assertEqual(shrink.called, not chordal)

    def test_dense_pattern_matches_wishart_moments(self):
        # A complete pattern with delta = 1 gives W ~ Wishart(n, I); normalization divides by n, so
        # E[K] = I and Var(K_ii) = 2/n for real K (1/n for complex).
        n, N = 4, 20000
        st = sparsechol.CholeskyStructure(~np.eye(n, dtype=bool))
        for offdiag, var in (('real', 2 / n), ('complex', 1 / n)):
            rng = np.random.default_rng(11)
            Ks = np.array([sp._sample_psd(st, rng, offdiag) for _ in range(N)])
            self.assertArraysAlmostEqual(Ks.mean(axis=0), np.eye(n), places=1)
            diag_var = np.var(np.real(np.einsum('kii->ki', Ks)), axis=0)
            np.testing.assert_allclose(diag_var, var, rtol=0.1)

    def test_bartlett_draw_is_order_invariant_and_normalized(self):
        pattern = np.zeros((5, 5), dtype=bool)
        for i, j in [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (3, 4)]:
            pattern[i, j] = pattern[j, i] = True
        N = 20000
        moments = []
        for perm in ([4, 0, 3, 1, 2], [4, 3, 2, 1, 0]):  # two perfect elimination orderings
            rng = np.random.default_rng(12)
            st = sparsechol.CholeskyStructure(pattern, ordering=np.array(perm))
            X = np.array([sp._sample_psd(st, rng, 'complex').ravel() for _ in range(N)])
            moments.append((X.mean(axis=0), np.real(X.conj().T @ X) / N))
            np.testing.assert_allclose(np.real(np.diag(X.mean(axis=0).reshape(5, 5))), 1, rtol=0.05)
        np.testing.assert_allclose(moments[0][0], moments[1][0], atol=0.03)
        np.testing.assert_allclose(moments[0][1], moments[1][1], atol=0.05)

    def test_psd_step_size_matches_bisection(self):
        rng = np.random.default_rng(3)
        for _ in range(20):
            n = 6
            G = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
            B = G @ G.conj().T + 0.1 * np.eye(n)
            H = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
            O = 3 * (H + H.conj().T)
            t = sp._psd_step_size(B, O)
            self.assertGreaterEqual(np.linalg.eigvalsh(B + t * O)[0], 0)
            lo, hi = 0.0, 1.0
            if np.linalg.eigvalsh(B + O)[0] >= 0:
                lo = 1.0
            for _ in range(60):
                mid = (lo + hi) / 2
                lo, hi = (mid, hi) if np.linalg.eigvalsh(B + mid * O)[0] >= 0 else (lo, mid)
            self.assertAlmostEqual(t, lo, places=6)
        with self.assertRaises(ValueError):
            sp._psd_step_size(np.diag([1.0, -1.0]), np.zeros((2, 2)))
        self.assertEqual(sp._psd_step_size(np.diag([1.0, 0.0]), np.ones((2, 2))), 0.0)

    def test_psd_step_size_boundary_cases(self):
        # diag(2, 8) + t*[[0, 8], [8, 0]] is PSD exactly when |t| <= 1/2.
        b = np.diag([2., 8.])
        direction = np.array([[0., 8.], [8., 0.]])
        t = sp._psd_step_size(b, direction)
        self.assertAlmostEqual(t, 0.5, places=8)
        self.assertLess(t, 0.5)
        self.assertGreaterEqual(np.linalg.eigvalsh(b + t * direction).min(), 0)
        self.assertLess(np.linalg.eigvalsh(b + (t + 1e-6) * direction).min(), 0)
        self.assertEqual(sp._psd_step_size(b, direction / 2), 1.0)
        self.assertEqual(sp._psd_step_size(b, np.zeros((2, 2))), 1.0)

    def test_impose_diagonal(self):
        pattern = _weight_pattern((2, 2), 2, 1)
        K = sp._sample_psd(pattern, np.random.default_rng(4))
        K2 = sp._impose_diagonal(K, [0, 5], [0.3, 0.02])
        self.assertAlmostEqual(np.real(K2[0, 0]), 0.3, places=14)
        self.assertAlmostEqual(np.real(K2[5, 5]), 0.02, places=14)
        self.assertArraysAlmostEqual(np.diag(K2)[1:5], np.diag(K)[1:5])
        self.assertTrue(np.array_equal(K2 != 0, K != 0))
        self._check_psd(K2)
        budget = 0.01 * K / np.real(np.trace(K))  # a budget is a scalar congruence
        self.assertAlmostEqual(np.real(np.trace(budget)), 0.01)

    def test_impose_offdiagonals(self):
        K = sp._sample_psd(~np.eye(4, dtype=bool), np.random.default_rng(5))
        s = np.real(np.diag(K))
        v = 0.5 * np.sqrt(s[0] * s[2]) * np.exp(0.3j)
        K2 = sp._impose_offdiagonals(K, real={(0, 2): v.real}, imag={(0, 2): v.imag})
        self.assertEqual(K2[0, 2], v)
        self.assertEqual(K2[2, 0], np.conj(v))
        self.assertArraysAlmostEqual(np.diag(K2), np.diag(K))
        self._check_psd(K2)
        with self.assertRaisesRegex(ValueError, 'completion'):
            sp._impose_offdiagonals(K, real={(0, 2): 2 * np.sqrt(s[0] * s[2])})

    def test_global_random_state_is_untouched(self):
        state = np.random.get_state()
        sp._sample_psd(_weight_pattern((2, 2, 2, 2), 2, 3), np.random.default_rng(0))
        after = np.random.get_state()
        self.assertTrue(np.array_equal(state[1], after[1]) and state[2] == after[2])

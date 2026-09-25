import numpy as np
import scipy.sparse as sps
import scipy.linalg as spl
from pygsti.tools import generator_infidelity
from pygsti.tools import lindbladtools as lt
from pygsti.modelmembers.operations import LindbladErrorgen
from pygsti.baseobjs import Basis, QubitSpace
from pygsti.baseobjs.errorgenlabel import GlobalElementaryErrorgenLabel, LocalElementaryErrorgenLabel
from ..util import BaseCase


class LindbladToolsTester(BaseCase):
    def test_hamiltonian_to_lindbladian(self):
        expectedLindbladian = np.array([
            [ 0,  0,  0,  0],
            [ 0,  0,  0,  0],
            [ 0,  0,  0,  0],
            [ 0,  0,  0,  0]
        ])
        self.assertArraysAlmostEqual(lt.create_elementary_errorgen('H', np.zeros(shape=(2, 2))),
                                     expectedLindbladian)
        sparse = sps.csr_matrix(np.zeros(shape=(2, 2)))
        #spL = lt.hamiltonian_to_lindbladian(sparse, True)
        spL = lt.create_elementary_errorgen('H', sparse, sparse=True)
        self.assertArraysAlmostEqual(spL.toarray(),
                                     expectedLindbladian)

    def test_stochastic_lindbladian(self):
        a = np.array([[1, 2], [3, 4]], 'd')
        expected = np.array([
            [ 1,  2,  2,  4],
            [ 3,  4,  6,  8],
            [ 3,  6,  4,  8],
            [ 9, 12, 12, 16]
        ], 'd')
        dual_eg, norm = lt.create_elementary_errorgen_dual('S', a, normalization_factor='auto_return')
        self.assertArraysAlmostEqual(
            dual_eg * norm, expected)
        sparse = sps.csr_matrix(a)
        spL = lt.create_elementary_errorgen_dual('S', sparse, sparse=True)
        self.assertArraysAlmostEqual(spL.toarray() * norm, expected)

    def test_nonham_lindbladian(self):
        a = np.array([[1, 2], [3, 4]], 'd')
        b = np.array([[1, 2], [3, 4]], 'd')
        expected = np.array([
            [ -9,  -5,  -5,  4],
            [ -4, -11,   6,  1],
            [ -4,   6, -11,  1],
            [  9,   5,   5, -4]
        ], 'd')
        self.assertArraysAlmostEqual(lt.create_lindbladian_term_errorgen('O', a, b), expected)
        sparsea = sps.csr_matrix(a)
        sparseb = sps.csr_matrix(b)
        spL = lt.create_lindbladian_term_errorgen('O', sparsea, sparseb, sparse=True)
        self.assertArraysAlmostEqual(spL.toarray(),
                                     expected)

    def test_elementary_errorgen_bases(self):

        bases = [Basis.cast('gm', 4),
                 Basis.cast('pp', 4),
                 Basis.cast('PP', 4)]

        for basis in bases:
            print(basis)

            primals = []; duals = []; lbls = []
            for lbl, bel in zip(basis.labels[1:], basis.elements[1:]):
                lbls.append("H_%s" % lbl)
                primals.append(lt.create_elementary_errorgen('H', bel))
                duals.append(lt.create_elementary_errorgen_dual('H', bel))
            for lbl, bel in zip(basis.labels[1:], basis.elements[1:]):
                lbls.append("S_%s" % lbl)
                primals.append(lt.create_elementary_errorgen('S', bel))
                duals.append(lt.create_elementary_errorgen_dual('S', bel))
            for i, (lbl, bel) in enumerate(zip(basis.labels[1:], basis.elements[1:])):
                for lbl2, bel2 in zip(basis.labels[1+i+1:], basis.elements[1+i+1:]):
                    lbls.append("C_%s_%s" % (lbl, lbl2))
                    primals.append(lt.create_elementary_errorgen('C', bel, bel2))
                    duals.append(lt.create_elementary_errorgen_dual('C', bel, bel2))
            for i, (lbl, bel) in enumerate(zip(basis.labels[1:], basis.elements[1:])):
                for lbl2, bel2 in zip(basis.labels[1+i+1:], basis.elements[1+i+1:]):
                    lbls.append("A_%s_%s" % (lbl, lbl2))
                    primals.append(lt.create_elementary_errorgen('A', bel, bel2))
                    duals.append(lt.create_elementary_errorgen_dual('A', bel, bel2))

            dot_mx = np.empty((len(duals), len(primals)), complex)
            for i, dual in enumerate(duals):
                for j, primal in enumerate(primals):
                    dot_mx[i,j] = np.vdot(dual, primal)

            self.assertTrue(np.allclose(dot_mx, np.identity(len(lbls), 'd')))

class RandomErrorgenRatesTester(BaseCase):

    def test_default_settings(self):
        random_errorgen_rates = lt.random_CPTP_error_generator_rates(num_qubits=2, seed=1234, label_type='local')

        #make sure that we get the expected number of rates:
        self.assertEqual(len(random_errorgen_rates), 240)

        #also make sure this is CPTP, do so by constructing an error generator and confirming it doesn't fail
        #with CPTP parameterization. This should fail if the error generator dictionary is not CPTP.
        errorgen = LindbladErrorgen.from_elementary_errorgens(random_errorgen_rates, parameterization='CPTPLND', truncate=False, state_space=QubitSpace(2))

    def test_sector_restrictions(self):
        #H-only:
        random_errorgen_rates = lt.random_CPTP_error_generator_rates(num_qubits=2, errorgen_types=('H',), seed=1234)
        #make sure that we get the expected number of rates:
        self.assertEqual(len(random_errorgen_rates), 15)
        #also make sure this is CPTP, do so by constructing an error generator and confirming it doesn't fail
        #with CPTP parameterization. This should fail if the error generator dictionary is not CPTP.
        errorgen = LindbladErrorgen.from_elementary_errorgens(random_errorgen_rates, parameterization='CPTPLND', truncate=False, state_space=QubitSpace(2))

        #S-only
        random_errorgen_rates = lt.random_CPTP_error_generator_rates(num_qubits=2, errorgen_types=('S',), seed=1234)
        #make sure that we get the expected number of rates:
        self.assertEqual(len(random_errorgen_rates), 15)
        #also make sure this is CPTP, do so by constructing an error generator and confirming it doesn't fail
        #with CPTP parameterization. This should fail if the error generator dictionary is not CPTP.
        errorgen = LindbladErrorgen.from_elementary_errorgens(random_errorgen_rates, parameterization='CPTPLND', truncate=False, state_space=QubitSpace(2))

        #H+S
        random_errorgen_rates = lt.random_CPTP_error_generator_rates(num_qubits=2, errorgen_types=('H','S'), seed=1234)
        #make sure that we get the expected number of rates:
        self.assertEqual(len(random_errorgen_rates), 30)
        #also make sure this is CPTP, do so by constructing an error generator and confirming it doesn't fail
        #with CPTP parameterization. This should fail if the error generator dictionary is not CPTP.
        errorgen = LindbladErrorgen.from_elementary_errorgens(random_errorgen_rates, parameterization='CPTPLND', truncate=False, state_space=QubitSpace(2))

        #H+S+A
        random_errorgen_rates = lt.random_CPTP_error_generator_rates(num_qubits=2, errorgen_types=('H','S','A'), seed=1234)
        #make sure that we get the expected number of rates:
        self.assertEqual(len(random_errorgen_rates), 135)
        #also make sure this is CPTP, do so by constructing an error generator and confirming it doesn't fail
        #with CPTP parameterization. This should fail if the error generator dictionary is not CPTP.
        errorgen = LindbladErrorgen.from_elementary_errorgens(random_errorgen_rates, parameterization='CPTPLND', truncate=False, state_space=QubitSpace(2))

    def test_error_metric_restrictions(self):
        #test generator_infidelity
        random_errorgen_rates = lt.random_CPTP_error_generator_rates(num_qubits=2, errorgen_types=('H','S'),
                                                                error_metric= 'generator_infidelity',
                                                                error_metric_value=0.99, seed=1234)
        #confirm this has the correct generator infidelity.
        gen_infdl = 0
        for coeff, rate in random_errorgen_rates.items():
            if coeff.errorgen_type == 'H':
                gen_infdl+=rate**2
            elif coeff.errorgen_type == 'S':
                gen_infdl+=rate

        assert abs(gen_infdl-0.99)<1e-5

        #test generator_error
        random_errorgen_rates = lt.random_CPTP_error_generator_rates(num_qubits=2, errorgen_types=('H','S'),
                                                                error_metric= 'total_generator_error',
                                                                error_metric_value=0.99, seed=1234)
        #confirm this has the correct generator infidelity.
        gen_error = 0
        for coeff, rate in random_errorgen_rates.items():
            if coeff.errorgen_type == 'H':
                gen_error+=abs(rate)
            elif coeff.errorgen_type == 'S':
                gen_error+=rate

        assert abs(gen_error-0.99)<1e-5

        #test relative_HS_contribution:
        random_errorgen_rates = lt.random_CPTP_error_generator_rates(num_qubits=2, errorgen_types=('H','S'),
                                                                error_metric= 'generator_infidelity',
                                                                error_metric_value=0.99,
                                                                relative_HS_contribution=(0.5, 0.5), seed=1234)
        #confirm this has the correct generator infidelity contributions.
        gen_infdl_H = 0
        gen_infdl_S = 0
        for coeff, rate in random_errorgen_rates.items():
            if coeff.errorgen_type == 'H':
                gen_infdl_H+=rate**2
            elif coeff.errorgen_type == 'S':
                gen_infdl_S+=rate

        assert abs(gen_infdl_S - gen_infdl_H)<1e-5

        random_errorgen_rates = lt.random_CPTP_error_generator_rates(num_qubits=2, errorgen_types=('H','S'),
                                                                error_metric= 'total_generator_error',
                                                                error_metric_value=0.99,
                                                                relative_HS_contribution=(0.5, 0.5), seed=1234)
        #confirm this has the correct generator error contributions.
        gen_error_H = 0
        gen_error_S = 0
        for coeff, rate in random_errorgen_rates.items():
            if coeff.errorgen_type == 'H':
                gen_error_H+=abs(rate)
            elif coeff.errorgen_type == 'S':
                gen_error_S+=rate

        assert abs(gen_error_S - gen_error_H)<1e-5

    def test_fixed_errorgen_rates(self):
        fixed_rates_dict = {GlobalElementaryErrorgenLabel('H', ('X',), (0,)): 1}
        random_errorgen_rates = lt.random_CPTP_error_generator_rates(num_qubits=2, errorgen_types=('H','S'),
                                                                fixed_errorgen_rates=fixed_rates_dict,
                                                                seed=1234)

        self.assertEqual(random_errorgen_rates[GlobalElementaryErrorgenLabel('H', ('X',), (0,))], 1)

    def test_label_type(self):

        random_errorgen_rates = lt.random_CPTP_error_generator_rates(num_qubits=2, errorgen_types=('H','S'),
                                                                label_type='local', seed=1234)
        assert isinstance(next(iter(random_errorgen_rates)), LocalElementaryErrorgenLabel)

    def test_sslbl_overlap(self):
        random_errorgen_rates = lt.random_CPTP_error_generator_rates(num_qubits=2, errorgen_types=('H','S'),
                                                                sslbl_overlap=(0,),
                                                                seed=1234)
        for coeff in random_errorgen_rates:
            assert 0 in coeff.sslbls

    def test_weight_restrictions(self):
        random_errorgen_rates = lt.random_CPTP_error_generator_rates(num_qubits=2, errorgen_types=('H','S','C','A'),
                                                                label_type='local', seed=1234,
                                                                max_weights={'H':1, 'S':1, 'C':1, 'A':1})
        assert len(random_errorgen_rates) == 24
        #confirm still CPTP
        errorgen = LindbladErrorgen.from_elementary_errorgens(random_errorgen_rates, parameterization='CPTPLND', truncate=False, state_space=QubitSpace(2))

        random_errorgen_rates = lt.random_CPTP_error_generator_rates(num_qubits=2, errorgen_types=('H','S','C','A'),
                                                                label_type='local', seed=1234,
                                                                max_weights={'H':2, 'S':2, 'C':1, 'A':1})
        assert len(random_errorgen_rates) == 42
        errorgen = LindbladErrorgen.from_elementary_errorgens(random_errorgen_rates, parameterization='CPTPLND', truncate=False, state_space=QubitSpace(2))

    def test_global_labels(self):
        random_errorgen_rates = lt.random_CPTP_error_generator_rates(num_qubits=2, seed=1234, label_type='global')

        #make sure that we get the expected number of rates:
        self.assertEqual(len(random_errorgen_rates), 240)

        #also make sure this is CPTP, do so by constructing an error generator and confirming it doesn't fail
        #with CPTP parameterization. This should fail if the error generator dictionary is not CPTP.
        errorgen = LindbladErrorgen.from_elementary_errorgens(random_errorgen_rates, parameterization='CPTPLND', truncate=False, state_space=QubitSpace(2))

    def test_respect_fidelity_constraint_issue_716(self):
        # This test is meant to confirm that the issue described in
        # https://github.com/sandialabs/pyGSTi/issues/716.
        #
        random_Honly_errors = lt.random_CPTP_error_generator_rates(
            1, ('H',),
            error_metric="generator_infidelity",
            error_metric_value=1e-2,
            label_type="local")
        errgen = LindbladErrorgen.from_elementary_errorgens(random_Honly_errors, state_space=["Q0"])
        noisy_ptm = spl.expm(errgen.to_dense("HilbertSchmidt"))
        actual = generator_infidelity(noisy_ptm, np.eye(4))
        self.assertAlmostEqual(actual, 1e-2, places=5)
        return


def _weight_pattern(dims, max_S_weight, max_CA_weight):
    """ Allowed C/A pairs among the S directions of weight <= max_S_weight, by union-support weight. """
    import itertools
    supports = [frozenset(i for i, x in enumerate(t) if x)
                for t in itertools.product(*[range(d * d) for d in dims])]
    supports = [s for s in supports if 0 < len(s) <= max_S_weight]
    n = len(supports)
    P = np.zeros((n, n), dtype=bool)
    for i, j in itertools.combinations(range(n), 2):
        P[i, j] = P[j, i] = len(supports[i] | supports[j]) <= max_CA_weight
    return P


class KossakowskiSamplingTester(BaseCase):

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
                    K = lt._sample_psd_kossakowski(pattern, np.random.default_rng(seed), offdiag, scale=0.1)
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
        K = lt._sample_psd_kossakowski(~np.eye(3, dtype=bool), lt._as_generator(2))
        self._check_psd(K)

    def test_chordal_patterns_need_no_shrinking(self):
        # A chordal pattern is ordered without fill, so K = L L^dag exactly and the shrink never runs;
        # a non-chordal pattern (and A-only sampling) needs it.
        from unittest import mock
        from pygsti.tools import sparsechol
        for pattern, chordal in self.PATTERNS:
            self.assertEqual(sparsechol.perfect_elimination_ordering(pattern) is not None, chordal)
            with mock.patch.object(lt, '_psd_shrink_factor', side_effect=AssertionError('shrink called')) as shrink:
                for offdiag in ('complex', 'real'):
                    try:
                        lt._sample_psd_kossakowski(pattern, np.random.default_rng(0), offdiag)
                    except AssertionError:
                        pass
                self.assertEqual(shrink.called, not chordal)

    def test_dense_pattern_matches_wishart_moments(self):
        # A complete pattern with delta = 1 gives W ~ Wishart(n, I); normalization divides by n, so
        # E[K] = I and Var(K_ii) = 2/n for real K (1/n for complex).
        n, N = 4, 20000
        st = lt._KossakowskiStructure(~np.eye(n, dtype=bool))
        for offdiag, var in (('real', 2 / n), ('complex', 1 / n)):
            rng = np.random.default_rng(11)
            Ks = np.array([lt._sample_psd_kossakowski(st, rng, offdiag) for _ in range(N)])
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
            st = lt._KossakowskiStructure(pattern, _perm=np.array(perm))
            X = np.array([lt._sample_psd_kossakowski(st, rng, 'complex').ravel() for _ in range(N)])
            moments.append((X.mean(axis=0), np.real(X.conj().T @ X) / N))
            np.testing.assert_allclose(np.real(np.diag(X.mean(axis=0).reshape(5, 5))), 1, rtol=0.05)
        np.testing.assert_allclose(moments[0][0], moments[1][0], atol=0.03)
        np.testing.assert_allclose(moments[0][1], moments[1][1], atol=0.05)

    def test_psd_shrink_factor_matches_bisection(self):
        rng = np.random.default_rng(3)
        for _ in range(20):
            n = 6
            G = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
            B = G @ G.conj().T + 0.1 * np.eye(n)
            H = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
            O = 3 * (H + H.conj().T)
            t = lt._psd_shrink_factor(B, O)
            self.assertGreaterEqual(np.linalg.eigvalsh(B + t * O)[0], 0)
            lo, hi = 0.0, 1.0
            if np.linalg.eigvalsh(B + O)[0] >= 0:
                lo = 1.0
            for _ in range(60):
                mid = (lo + hi) / 2
                lo, hi = (mid, hi) if np.linalg.eigvalsh(B + mid * O)[0] >= 0 else (lo, mid)
            self.assertAlmostEqual(t, lo, places=6)
        with self.assertRaises(ValueError):
            lt._psd_shrink_factor(np.diag([1.0, -1.0]), np.zeros((2, 2)))
        self.assertEqual(lt._psd_shrink_factor(np.diag([1.0, 0.0]), np.ones((2, 2))), 0.0)

    def test_impose_diagonal(self):
        pattern = _weight_pattern((2, 2), 2, 1)
        K = lt._sample_psd_kossakowski(pattern, np.random.default_rng(4))
        K2 = lt._impose_diagonal(K, [0, 5], [0.3, 0.02])
        self.assertAlmostEqual(np.real(K2[0, 0]), 0.3, places=14)
        self.assertAlmostEqual(np.real(K2[5, 5]), 0.02, places=14)
        self.assertArraysAlmostEqual(np.diag(K2)[1:5], np.diag(K)[1:5])
        self.assertTrue(np.array_equal(K2 != 0, K != 0))
        self._check_psd(K2)
        budget = 0.01 * K / np.real(np.trace(K))  # a budget is a scalar congruence
        self.assertAlmostEqual(np.real(np.trace(budget)), 0.01)

    def test_impose_offdiagonals(self):
        K = lt._sample_psd_kossakowski(~np.eye(4, dtype=bool), np.random.default_rng(5))
        s = np.real(np.diag(K))
        v = 0.5 * np.sqrt(s[0] * s[2]) * np.exp(0.3j)
        K2 = lt._impose_offdiagonals(K, {(0, 2): v})
        self.assertEqual(K2[0, 2], v)
        self.assertEqual(K2[2, 0], np.conj(v))
        self.assertArraysAlmostEqual(np.diag(K2), np.diag(K))
        self._check_psd(K2)
        with self.assertRaisesRegex(ValueError, 'completion'):
            lt._impose_offdiagonals(K, {(0, 2): 2 * np.sqrt(s[0] * s[2])})

    def test_as_generator(self):
        g = np.random.default_rng(5)
        self.assertIs(lt._as_generator(g), g)
        self.assertEqual(lt._as_generator(7).random(), np.random.default_rng(7).random())
        self.assertEqual(lt._as_generator(None).bit_generator.__class__, np.random.PCG64)
        bg = np.random.Philox(3)
        self.assertIs(lt._as_generator(bg).bit_generator, bg)
        rs = np.random.RandomState(4)
        gen = lt._as_generator(rs)
        before = rs.get_state()[1].copy()
        gen.random()
        self.assertFalse(np.array_equal(before, rs.get_state()[1]))  # shares the bit generator state
        with self.assertRaises((TypeError, ValueError)):
            lt._as_generator('not a seed')

    def test_global_random_state_is_untouched(self):
        state = np.random.get_state()
        lt._sample_psd_kossakowski(_weight_pattern((2, 2, 2, 2), 2, 3), lt._as_generator(0))
        after = np.random.get_state()
        self.assertTrue(np.array_equal(state[1], after[1]) and state[2] == after[2])

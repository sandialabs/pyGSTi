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
        random_errorgen_rates = lt.random_cptp_errorgen_rates(QubitSpace(2), seed=1234, label_type='local')

        #make sure that we get the expected number of rates:
        self.assertEqual(len(random_errorgen_rates), 240)

        #also make sure this is CPTP, do so by constructing an error generator and confirming it doesn't fail
        #with CPTP parameterization. This should fail if the error generator dictionary is not CPTP.
        errorgen = LindbladErrorgen.from_elementary_errorgens(random_errorgen_rates, parameterization='CPTPLND', truncate=False, state_space=QubitSpace(2))

    def test_sector_restrictions(self):
        #H-only:
        random_errorgen_rates = lt.random_cptp_errorgen_rates(QubitSpace(2), errorgen_types=('H',), seed=1234)
        #make sure that we get the expected number of rates:
        self.assertEqual(len(random_errorgen_rates), 15)
        #also make sure this is CPTP, do so by constructing an error generator and confirming it doesn't fail
        #with CPTP parameterization. This should fail if the error generator dictionary is not CPTP.
        errorgen = LindbladErrorgen.from_elementary_errorgens(random_errorgen_rates, parameterization='CPTPLND', truncate=False, state_space=QubitSpace(2))

        #S-only
        random_errorgen_rates = lt.random_cptp_errorgen_rates(QubitSpace(2), errorgen_types=('S',), seed=1234)
        #make sure that we get the expected number of rates:
        self.assertEqual(len(random_errorgen_rates), 15)
        #also make sure this is CPTP, do so by constructing an error generator and confirming it doesn't fail
        #with CPTP parameterization. This should fail if the error generator dictionary is not CPTP.
        errorgen = LindbladErrorgen.from_elementary_errorgens(random_errorgen_rates, parameterization='CPTPLND', truncate=False, state_space=QubitSpace(2))

        #H+S
        random_errorgen_rates = lt.random_cptp_errorgen_rates(QubitSpace(2), errorgen_types=('H','S'), seed=1234)
        #make sure that we get the expected number of rates:
        self.assertEqual(len(random_errorgen_rates), 30)
        #also make sure this is CPTP, do so by constructing an error generator and confirming it doesn't fail
        #with CPTP parameterization. This should fail if the error generator dictionary is not CPTP.
        errorgen = LindbladErrorgen.from_elementary_errorgens(random_errorgen_rates, parameterization='CPTPLND', truncate=False, state_space=QubitSpace(2))

        #H+S+A
        random_errorgen_rates = lt.random_cptp_errorgen_rates(QubitSpace(2), errorgen_types=('H','S','A'), seed=1234)
        #make sure that we get the expected number of rates:
        self.assertEqual(len(random_errorgen_rates), 135)
        #also make sure this is CPTP, do so by constructing an error generator and confirming it doesn't fail
        #with CPTP parameterization. This should fail if the error generator dictionary is not CPTP.
        errorgen = LindbladErrorgen.from_elementary_errorgens(random_errorgen_rates, parameterization='CPTPLND', truncate=False, state_space=QubitSpace(2))

    def test_error_metric_restrictions(self):
        #test generator_infidelity
        random_errorgen_rates = lt.random_cptp_errorgen_rates(QubitSpace(2), errorgen_types=('H','S'),
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
        random_errorgen_rates = lt.random_cptp_errorgen_rates(QubitSpace(2), errorgen_types=('H','S'),
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
        random_errorgen_rates = lt.random_cptp_errorgen_rates(QubitSpace(2), errorgen_types=('H','S'),
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

        random_errorgen_rates = lt.random_cptp_errorgen_rates(QubitSpace(2), errorgen_types=('H','S'),
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
        random_errorgen_rates = lt.random_cptp_errorgen_rates(QubitSpace(2), errorgen_types=('H','S'),
                                                                fixed_errorgen_rates=fixed_rates_dict,
                                                                seed=1234)

        self.assertEqual(random_errorgen_rates[GlobalElementaryErrorgenLabel('H', ('X',), (0,))], 1)

    def test_label_type(self):

        random_errorgen_rates = lt.random_cptp_errorgen_rates(QubitSpace(2), errorgen_types=('H','S'),
                                                                label_type='local', seed=1234)
        assert isinstance(next(iter(random_errorgen_rates)), LocalElementaryErrorgenLabel)

    def test_sslbl_overlap(self):
        random_errorgen_rates = lt.random_cptp_errorgen_rates(QubitSpace(2), errorgen_types=('H','S'),
                                                                sslbl_overlap=(0,),
                                                                seed=1234)
        for coeff in random_errorgen_rates:
            assert 0 in coeff.sslbls

    def test_weight_restrictions(self):
        random_errorgen_rates = lt.random_cptp_errorgen_rates(QubitSpace(2), errorgen_types=('H','S','C','A'),
                                                                label_type='local', seed=1234,
                                                                max_weights={'H':1, 'S':1, 'C':1, 'A':1})
        assert len(random_errorgen_rates) == 24
        #confirm still CPTP
        errorgen = LindbladErrorgen.from_elementary_errorgens(random_errorgen_rates, parameterization='CPTPLND', truncate=False, state_space=QubitSpace(2))

        random_errorgen_rates = lt.random_cptp_errorgen_rates(QubitSpace(2), errorgen_types=('H','S','C','A'),
                                                                label_type='local', seed=1234,
                                                                max_weights={'H':2, 'S':2, 'C':1, 'A':1})
        assert len(random_errorgen_rates) == 42
        errorgen = LindbladErrorgen.from_elementary_errorgens(random_errorgen_rates, parameterization='CPTPLND', truncate=False, state_space=QubitSpace(2))

    def test_global_labels(self):
        random_errorgen_rates = lt.random_cptp_errorgen_rates(QubitSpace(2), seed=1234, label_type='global')

        #make sure that we get the expected number of rates:
        self.assertEqual(len(random_errorgen_rates), 240)

        #also make sure this is CPTP, do so by constructing an error generator and confirming it doesn't fail
        #with CPTP parameterization. This should fail if the error generator dictionary is not CPTP.
        errorgen = LindbladErrorgen.from_elementary_errorgens(random_errorgen_rates, parameterization='CPTPLND', truncate=False, state_space=QubitSpace(2))

    def test_respect_fidelity_constraint_issue_716(self):
        # This test is meant to confirm that the issue described in
        # https://github.com/sandialabs/pyGSTi/issues/716.
        #
        random_Honly_errors = lt.random_cptp_errorgen_rates(
            QubitSpace(1), errorgen_types=('H',),
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
        K2 = lt._impose_offdiagonals(K, real={(0, 2): v.real}, imag={(0, 2): v.imag})
        self.assertEqual(K2[0, 2], v)
        self.assertEqual(K2[2, 0], np.conj(v))
        self.assertArraysAlmostEqual(np.diag(K2), np.diag(K))
        self._check_psd(K2)
        with self.assertRaisesRegex(ValueError, 'completion'):
            lt._impose_offdiagonals(K, real={(0, 2): 2 * np.sqrt(s[0] * s[2])})

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


def _space(sslbls, dims):
    from pygsti.baseobjs import ExplicitStateSpace
    return ExplicitStateSpace([tuple(sslbls)], [tuple(dims)])


def _kossakowski_from_rates(rates, basis):
    """ K_ii = S_i and K_ij = C_ij - 1j A_ij (i < j), indexed by the non-identity elements of basis. """
    index = {bel: a for a, bel in enumerate(basis.labels[1:])}
    n = len(index)
    K = np.zeros((n, n), dtype=complex)
    for lbl, v in rates.items():
        idx = [index[bel] for bel in lbl.basis_element_labels]
        if lbl.errorgen_type == 'S':
            K[idx[0], idx[0]] += v
        elif lbl.errorgen_type in ('C', 'A'):
            (i, j), sign = sorted(idx), (1 if idx[0] < idx[1] else -1)
            z = v if lbl.errorgen_type == 'C' else -1j * sign * v
            K[i, j] += z
            K[j, i] += np.conj(z)
    return K


class RandomCptpErrorgenRatesTester(BaseCase):

    SPACES = [(('Q0', 'Q1'), (2, 2)), (('T0',), (3,)), (('Q0', 'T1'), (2, 3))]

    def _check_cp(self, rates, state_space, basis=None):
        from pygsti.baseobjs import canonical_errorgen_basis
        basis = canonical_errorgen_basis(state_space) if basis is None else basis
        local = {LocalElementaryErrorgenLabel.cast(k, sslbls=state_space.sole_tensor_product_block_labels): v
                 for k, v in rates.items()}
        K = _kossakowski_from_rates(local, basis)
        self.assertGreaterEqual(np.linalg.eigvalsh(K)[0], -1e-12 * max(1.0, np.max(np.abs(K))))
        # pyGSTi's own CPTP parameterization rejects a non-CP generator (truncate=False). Constructing
        # it takes about a minute on four qubits, so it only checks the small spaces.
        if state_space.dim <= 36:
            LindbladErrorgen.from_elementary_errorgens(rates, parameterization='CPTPLND', truncate=False,
                                                       state_space=state_space, elementary_errorgen_basis=basis,
                                                       mx_basis=basis)
        return K

    def _support(self, bel):
        from pygsti.baseobjs.errorgenlabel import _bel_tokens
        return {i for i, t in enumerate(_bel_tokens(bel)) if t != 'I'}

    def test_counts_labels_and_cp_on_qubits_qutrits_and_mixtures(self):
        for sslbls, dims in self.SPACES:
            ss = _space(sslbls, dims)
            n = int(np.prod(dims)) ** 2 - 1
            for label_type, cls in (('local', LocalElementaryErrorgenLabel), ('global', GlobalElementaryErrorgenLabel)):
                rates = lt.random_cptp_errorgen_rates(ss, label_type=label_type, seed=1)
                counts = {t: sum(k.errorgen_type == t for k in rates) for t in 'HSCA'}
                self.assertEqual(counts, {'H': n, 'S': n, 'C': n * (n - 1) // 2, 'A': n * (n - 1) // 2})
                self.assertTrue(all(isinstance(k, cls) for k in rates))
                self.assertTrue(all(v > 0 for k, v in rates.items() if k.errorgen_type == 'S'))
                self._check_cp(rates, ss)
            glob = lt.random_cptp_errorgen_rates(ss, errorgen_types=('H', 'S'), seed=2)
            self.assertTrue(all(set(k.sslbls) <= set(sslbls) for k in glob))

    def test_cp_under_restrictions_and_sectors(self):
        cases = [  # (space, errorgen_types, max_weights, sslbl_overlap)
            (_space(range(4), (2,) * 4), ('S', 'C', 'A'), {'S': 2, 'C': 3, 'A': 3}, None),  # not chordal
            (_space(range(3), (2, 3, 2)), ('H', 'S', 'C', 'A'), {'S': 2, 'C': 2, 'A': 1}, None),  # C and A differ
            (_space(('Q0', 'T1'), (2, 3)), ('S', 'A'), None, None),  # A without C
            (_space(('Q0', 'T1'), (2, 3)), ('S', 'C'), None, None),
            (_space(('Q0', 'T1', 'Q2'), (2, 3, 2)), ('H', 'S', 'C', 'A'), {'S': 1, 'C': 2, 'A': 2}, ('T1',)),
        ]
        for ss, types, mw, overlap in cases:
            for seed in range(3):
                rates = lt.random_cptp_errorgen_rates(ss, errorgen_types=types, max_weights=mw, sslbl_overlap=overlap,
                                                      label_type='local', seed=seed)
                self.assertEqual({k.errorgen_type for k in rates}, set(types))
                self._check_cp(rates, ss)
                for k in rates:
                    w = len(set().union(*map(self._support, k.basis_element_labels)))
                    if mw is not None:
                        self.assertLessEqual(w, mw.get(k.errorgen_type, np.inf))
                    if overlap is not None:
                        g = GlobalElementaryErrorgenLabel.cast(k, sslbls=ss.sole_tensor_product_block_labels)
                        self.assertIn('T1', g.sslbls)

    def test_weight_limits_count_pairs_by_union_support(self):
        # On two qubits with weight-1 S directions (3 per qubit), C pairs of union weight <= 1 stay on one qubit.
        ss = QubitSpace(2)
        rates = lt.random_cptp_errorgen_rates(ss, errorgen_types=('S', 'C'), max_weights={'S': 1, 'C': 1}, seed=0)
        self.assertEqual(sum(k.errorgen_type == 'C' for k in rates), 2 * 3)
        rates = lt.random_cptp_errorgen_rates(ss, errorgen_types=('S', 'C'), max_weights={'S': 1, 'C': 2}, seed=0)
        self.assertEqual(sum(k.errorgen_type == 'C' for k in rates), 15)  # all pairs of the 6 S directions

    def test_error_metrics_on_a_qutrit(self):
        ss = _space(('T0',), (3,))
        for metric, hp in (('generator_infidelity', lambda h: h ** 2), ('total_generator_error', abs)):
            rates = lt.random_cptp_errorgen_rates(ss, error_metric=metric, error_metric_value=0.02, seed=4)
            H = sum(hp(v) for k, v in rates.items() if k.errorgen_type == 'H')
            S = sum(v for k, v in rates.items() if k.errorgen_type == 'S')
            self.assertAlmostEqual(H + S, 0.02, places=12)
            rates = lt.random_cptp_errorgen_rates(ss, error_metric=metric, error_metric_value=0.02,
                                                  relative_HS_contribution=(0.25, 0.75), seed=4)
            H = sum(hp(v) for k, v in rates.items() if k.errorgen_type == 'H')
            S = sum(v for k, v in rates.items() if k.errorgen_type == 'S')
            self.assertAlmostEqual(H, 0.005, places=12)
            self.assertAlmostEqual(S, 0.015, places=12)
            self._check_cp(rates, ss)

    def test_generator_infidelity_budget_matches_leading_order_infidelity(self):
        # On the canonical scale, sum(h^2) + sum(s) is the leading-order entanglement infidelity
        # 1 - Tr(exp(L)) / D^2 on qudits too (the trace of a superoperator is basis-independent).
        from pygsti.baseobjs import canonical_errorgen_basis
        ss = _space(('Q0', 'T1'), (2, 3))
        b = canonical_errorgen_basis(ss)
        for types in (('H',), ('S',)):
            rates = lt.random_cptp_errorgen_rates(ss, errorgen_types=types, error_metric='generator_infidelity',
                                                  error_metric_value=1e-4, seed=5)
            eg = LindbladErrorgen.from_elementary_errorgens(rates, state_space=ss, elementary_errorgen_basis=b,
                                                            mx_basis=b)
            actual = 1 - np.real(np.trace(spl.expm(eg.to_dense()))) / 36
            self.assertAlmostEqual(actual / 1e-4, 1.0, places=2)

    def test_fixed_rates(self):
        ss = _space(('Q0', 'T1'), (2, 3))
        L, G = LocalElementaryErrorgenLabel, GlobalElementaryErrorgenLabel
        fixed = {G('H', ('X_{0,1}',), ('T1',)): 0.003,
                 L('S', ('XZ_{1}',)): 0.002,  # outside the weight limit below: included anyway
                 G('S', ('X',), ('Q0',)): 0.001,
                 L('S', ('IZ_{1}',)): 0.001,
                 L('S', ('IY_{0,2}',)): 0.001,
                 L('A', ('XI', 'IY_{0,2}')): 1e-5,
                 L('C', ('IZ_{1}', 'XI')): -2e-5}  # reversed order is accepted
        for label_type in ('local', 'global'):
            rates = lt.random_cptp_errorgen_rates(ss, max_weights={'H': 1, 'S': 1, 'C': 1, 'A': 1},
                                                  fixed_errorgen_rates=fixed, label_type=label_type, seed=6)
            cls = L if label_type == 'local' else G
            for k, v in fixed.items():
                self.assertEqual(rates[cls.cast(k, sslbls=('Q0', 'T1'))], v)
            self._check_cp(rates, ss)

        with_budget = lt.random_cptp_errorgen_rates(ss, errorgen_types=('H', 'S'), fixed_errorgen_rates=fixed_H_S(),
                                                    error_metric='generator_infidelity', error_metric_value=0.01,
                                                    label_type='local', seed=7)
        total = sum(v ** 2 if k.errorgen_type == 'H' else v for k, v in with_budget.items())
        self.assertAlmostEqual(total, 0.01, places=12)
        self.assertEqual(with_budget[L('S', ('XZ_{1}',))], 0.002)

    def test_fixed_rates_errors(self):
        ss = _space(('Q0', 'T1'), (2, 3))
        L = LocalElementaryErrorgenLabel
        with self.assertRaisesRegex(ValueError, 'completion'):
            lt.random_cptp_errorgen_rates(ss, fixed_errorgen_rates={L('S', ('XI',)): 1e-4, L('S', ('IX_{0,1}',)): 1e-4,
                                                                    L('C', ('XI', 'IX_{0,1}')): 1e-3}, seed=0)
        with self.assertRaisesRegex(ValueError, 'S rates'):
            lt.random_cptp_errorgen_rates(ss, errorgen_types=('H',), fixed_errorgen_rates={L('C', ('XI', 'IX_{0,1}')): 1e-5})
        with self.assertRaisesRegex(ValueError, 'not non-identity labels'):
            lt.random_cptp_errorgen_rates(ss, fixed_errorgen_rates={L('H', ('XX',)): 1e-3})  # 'XX' is not a qubit-qutrit label
        with self.assertRaisesRegex(ValueError, 'exceed'):
            lt.random_cptp_errorgen_rates(ss, fixed_errorgen_rates={L('S', ('XI',)): 0.1},
                                          error_metric='generator_infidelity', error_metric_value=0.01)
        with self.assertRaises(TypeError):
            lt.random_cptp_errorgen_rates(ss, fixed_errorgen_rates={'H_XI': 0.1})

    def test_bases(self):
        from pygsti.baseobjs import canonical_errorgen_basis
        from pygsti.baseobjs.basis import BuiltinBasis, TensorProdBasis
        ss = _space(('Q0', 'T1'), (2, 3))
        canonical = lt.random_cptp_errorgen_rates(ss, seed=8)
        self.assertEqual(lt.random_cptp_errorgen_rates(ss, elementary_errorgen_basis=canonical_errorgen_basis(ss),
                                                       seed=8), canonical)
        gm = lt.random_cptp_errorgen_rates(ss, elementary_errorgen_basis='GM', seed=8)
        self.assertEqual(len(gm), len(canonical))
        self._check_cp(gm, ss, Basis.cast('GM', ss))
        # An explicit orthonormal basis keeps its normalization and labels.
        gm_orth = TensorProdBasis([BuiltinBasis('gm', 4), BuiltinBasis('gm', 9)])
        rates = lt.random_cptp_errorgen_rates(ss, elementary_errorgen_basis=gm_orth, label_type='local', seed=8)
        self._check_cp(rates, ss, gm_orth)
        # A basis without per-subsystem structure only supports local labels and no support restrictions.
        flat = BuiltinBasis('GM', 36)
        rates = lt.random_cptp_errorgen_rates(ss, elementary_errorgen_basis=flat, label_type='local', seed=8)
        self.assertEqual(len(rates), 2 * 35 + 35 * 34)
        with self.assertRaisesRegex(ValueError, 'per subsystem'):
            lt.random_cptp_errorgen_rates(ss, elementary_errorgen_basis=flat, seed=8)
        with self.assertRaisesRegex(ValueError, 'not available'):
            lt.random_cptp_errorgen_rates(ss, elementary_errorgen_basis='PP')
        with self.assertRaisesRegex(ValueError, 'identity'):
            lt.random_cptp_errorgen_rates(ss, elementary_errorgen_basis='std', label_type='local')
        with self.assertRaisesRegex(ValueError, 'dimension'):
            lt.random_cptp_errorgen_rates(ss, elementary_errorgen_basis=BuiltinBasis('PP', 16))
        with self.assertRaises(TypeError):
            lt.random_cptp_errorgen_rates(ss, elementary_errorgen_basis=3)

    def test_argument_validation(self):
        from pygsti.baseobjs import ExplicitStateSpace
        ss = QubitSpace(1)
        bad = [dict(errorgen_types=('H', 'C')), dict(errorgen_types=('H', 'X')), dict(label_type='both'),
               dict(error_metric='generator_infidelity'), dict(error_metric_value=0.1),
               dict(error_metric='fidelity', error_metric_value=0.1), dict(relative_HS_contribution=(0.5, 0.5)),
               dict(error_metric='generator_infidelity', error_metric_value=0.1, relative_HS_contribution=(0.5, 0.6)),
               dict(SCA_params=(0.1, 0.01)), dict(H_params=(0.1, 0.01), error_metric='generator_infidelity',
                                                  error_metric_value=0.1),
               dict(max_weights={'X': 1}), dict(sslbl_overlap=('nope',))]
        for kwargs in bad:
            with self.assertRaises(ValueError, msg=str(kwargs)):
                lt.random_cptp_errorgen_rates(ss, **kwargs)
        with self.assertRaises(TypeError):
            lt.random_cptp_errorgen_rates(1)
        with self.assertRaises(TypeError):
            lt.random_cptp_errorgen_rates(ss, ('H',))  # keyword-only
        with self.assertRaises(ValueError):
            lt.random_cptp_errorgen_rates(ExplicitStateSpace([('Q0',), ('L',)], [(2,), (1,)]))  # direct sum

    def test_rng_contract(self):
        ss = _space(('T0',), (3,))
        self.assertEqual(lt.random_cptp_errorgen_rates(ss, seed=9), lt.random_cptp_errorgen_rates(ss, seed=9))
        self.assertEqual(lt.random_cptp_errorgen_rates(ss, seed=9),
                         lt.random_cptp_errorgen_rates(ss, seed=np.random.default_rng(9)))
        gen = np.random.default_rng(10)
        first = lt.random_cptp_errorgen_rates(ss, seed=gen)
        self.assertNotEqual(first, lt.random_cptp_errorgen_rates(ss, seed=gen))  # the generator advanced
        state = np.random.get_state()[1].copy()
        lt.random_cptp_errorgen_rates(ss, seed=None)
        self.assertTrue(np.array_equal(state, np.random.get_state()[1]))

    def test_deprecated_wrapper(self):
        import inspect
        from unittest import mock
        from pygsti.tools.exceptions import pyGSTiDeprecationWarning
        self.assertEqual(list(inspect.signature(lt.random_CPTP_error_generator_rates).parameters),
                         ['num_qubits', 'errorgen_types', 'max_weights', 'H_params', 'SCA_params', 'error_metric',
                          'error_metric_value', 'relative_HS_contribution', 'fixed_errorgen_rates', 'sslbl_overlap',
                          'label_type', 'seed', 'qubit_labels'])
        with self.assertWarns(pyGSTiDeprecationWarning):
            old = lt.random_CPTP_error_generator_rates(2, ('H', 'S'), seed=11)
        self.assertEqual(old, lt.random_cptp_errorgen_rates(QubitSpace(2), errorgen_types=('H', 'S'),
                                                            elementary_errorgen_basis='PP', seed=11))
        with mock.patch.object(lt, 'random_cptp_errorgen_rates', return_value={}) as new, \
                self.assertWarns(pyGSTiDeprecationWarning):
            lt.random_CPTP_error_generator_rates(3, ('H',), {'H': 1}, error_metric_value=0.1, sslbl_overlap=[1],
                                                 label_type='local', seed=5, qubit_labels=['a', 'b', 'c'])
        (ss,), kwargs = new.call_args
        self.assertEqual(ss, QubitSpace(3))
        self.assertEqual(kwargs['elementary_errorgen_basis'], 'PP')
        self.assertEqual((kwargs['errorgen_types'], kwargs['max_weights'], kwargs['sslbl_overlap'],
                          kwargs['label_type'], kwargs['seed']), (('H',), {'H': 1}, [1], 'local', 5))
        self.assertIsNone(kwargs['error_metric_value'])  # dropped without a metric, as it used to be ignored
        with self.assertWarns(pyGSTiDeprecationWarning):
            relabeled = lt.random_CPTP_error_generator_rates(2, ('H',), seed=12, qubit_labels=['Q0', 'Q1'])
        self.assertTrue(all(set(k.sslbls) <= {'Q0', 'Q1'} for k in relabeled))

    def test_legacy_a_sector_regression(self):
        # random_CPTP_error_generator_rates(1, seed=2) used to return a non-CP coefficient matrix.
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            rates = lt.random_CPTP_error_generator_rates(1, seed=2)
        self._check_cp(rates, QubitSpace(1))


def fixed_H_S():
    L = LocalElementaryErrorgenLabel
    return {L('H', ('XI',)): 0.03, L('S', ('XZ_{1}',)): 0.002}

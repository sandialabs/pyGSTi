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


class GeneratorConversionTester(BaseCase):
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


class BasisFactorValidationTester(BaseCase):
    def test_builtin_factors_skip_numerical_validation(self):
        from unittest import mock
        for name in ('pp', 'PP', 'gm', 'GM', 'gm_unnormalized'):
            for sparse in (False, True):
                basis = Basis.cast(name, 4, sparse=sparse)
                with self.subTest(name=name, sparse=sparse), mock.patch.object(
                        lt._np, 'allclose', side_effect=AssertionError('numerical validation invoked')):
                    lt._check_basis_factor(basis)

    def test_custom_GM_basis_is_numerically_validated(self):
        from pygsti.baseobjs import ExplicitBasis
        elements = Basis.cast('GM', 4).elements.copy()
        elements[1, 0, 1] += .2j
        invalid = ExplicitBasis(elements, labels=['I', 'X', 'Y', 'Z'], name='GM')
        with self.assertRaisesRegex(ValueError, 'Hermitian'):
            lt._check_basis_factor(invalid)
        elements = Basis.cast('GM', 4).elements.copy()
        elements[2] += .1 * elements[1]
        invalid = ExplicitBasis(elements, labels=['I', 'X', 'Y', 'Z'], name='GM')
        with self.assertRaisesRegex(ValueError, 'orthogonal'):
            lt._check_basis_factor(invalid)

    def test_mixed_tensor_product_checks_custom_factor(self):
        from pygsti.baseobjs import ExplicitBasis, TensorProdBasis
        builtin = Basis.cast('GM', 4)
        custom = ExplicitBasis(builtin.elements, labels=builtin.labels, name='GM')
        space = QubitSpace(2)
        mixed = TensorProdBasis([Basis.cast('PP', 4), custom])
        self.assertIs(lt._resolve_errorgen_basis(space, mixed), mixed)
        elements = builtin.elements.copy()
        elements[1] += np.eye(2)
        invalid = ExplicitBasis(elements, labels=builtin.labels, name='GM')
        with self.assertRaisesRegex(ValueError, 'traceless'):
            lt._resolve_errorgen_basis(space, TensorProdBasis([Basis.cast('PP', 4), invalid]))

    def test_lazy_invalid_builtin_dimensions_still_fail(self):
        from pygsti.baseobjs import BuiltinBasis
        with self.assertRaises((ValueError, AssertionError)):
            lt._check_basis_factor(BuiltinBasis('PP', 9))
        with self.assertRaisesRegex(ValueError, 'dimension'):
            lt._resolve_errorgen_basis(QubitSpace(1), BuiltinBasis('GM', 9))

    def test_nested_builtin_tensor_product_skips_numerical_validation(self):
        from unittest import mock
        from pygsti.baseobjs import TensorProdBasis
        pair = TensorProdBasis([Basis.cast('PP', 4), Basis.cast('GM', 4)])
        basis = TensorProdBasis([pair, Basis.cast('PP', 4)])
        with mock.patch.object(lt._np, 'allclose', side_effect=AssertionError('numerical validation invoked')):
            self.assertIs(lt._resolve_errorgen_basis(QubitSpace(3), basis), basis)

from itertools import chain, product
from math import factorial
import random
from unittest import mock
import warnings

import numpy as np
from scipy.linalg import expm, logm

from pygsti.algorithms.randomcircuit import create_random_circuit
from pygsti.baseobjs import QubitSpace
from pygsti.baseobjs.errorgenbasis import CompleteElementaryErrorgenBasis
from pygsti.errorgenpropagation.errorpropagator import ErrorGeneratorPropagator
from pygsti.errorgenpropagation.localstimerrorgen import LocalStimErrorgenLabel as _LSE, bel_less_than
from pygsti.models.modelconstruction import create_crosstalk_free_model
from pygsti.processors import QubitProcessorSpec
from pygsti.tools import errorgenpropagation as _eprop
from pygsti.tools.exceptions import pyGSTiDeprecationWarning

from ...util import BaseCase
from .errgenprop_test_utils import (
    _all_sector_rate_subset,
    sample_error_rates_dict,
    _dense_zassenhaus_generators_analytic,
)


class ErrgenExpansionsTester(BaseCase):

    def setUp(self):
        num_qubits = 4
        gate_names = ['Gcphase', 'Gxpi2', 'Gypi2']
        availability = {'Gcphase': [(0, 1), (1, 2), (2, 3), (3, 0)]}
        pspec = QubitProcessorSpec(num_qubits, gate_names, availability=availability)
        self.target_model = create_crosstalk_free_model(processor_spec=pspec)
        self.circuit = create_random_circuit(pspec, 4, sampler='edgegrab', samplerargs=[0.4], rand_state=12345)
        max_strengths = {1: {'S': 0, 'H': 0.0001},
                         2: {'S': 0, 'H': 0.0001}}
        error_rates_dict = sample_error_rates_dict(pspec, max_strengths, seed=12345)
        self.error_model = create_crosstalk_free_model(pspec, lindblad_error_coeffs=error_rates_dict)
        self.errorgen_propagator = ErrorGeneratorPropagator(self.error_model.copy())
        self.propagated_errorgen_layers = self.errorgen_propagator.propagate_errorgens(self.circuit)

        # 4-sector GLND model used by Taylor expansion tests and empty layer test
        rates = {
            'Gxpi2': _all_sector_rate_subset(1, 1, seed=100),
            'Gypi2': _all_sector_rate_subset(1, 1, seed=101),
            'Gcphase': _all_sector_rate_subset(2, 1, seed=200),
        }
        self.error_model_glnd = create_crosstalk_free_model(pspec, lindblad_error_coeffs=rates, lindblad_parameterization='GLND')
        self.error_propagator = ErrorGeneratorPropagator(self.error_model_glnd.copy())
        self.propagated_errorgen_layer = self.error_propagator.propagate_errorgens_bch(self.circuit, bch_order=1)

    def _random_errorgen_dicts(self, num_qubits, sizes, seed):
        #random all-sector error generator dictionaries over the complete basis, with the basis' matrix
        #dictionary for dense checks.
        basis = CompleteElementaryErrorgenBasis('PP', QubitSpace(num_qubits), default_label_type='local')
        matrix_dict = {lbl: mat for lbl, mat in zip(basis.labels, basis.elemgen_matrices)}
        rng = random.Random(seed)
        by_type = {t: [lbl for lbl in basis.labels if lbl.errorgen_type == t] for t in 'HSCA'}
        dicts = []
        for size in sizes:
            #`size` labels of each type (or all of them, for the small H/S sectors)
            labels = [lbl for t in 'HSCA' for lbl in rng.sample(by_type[t], min(size, len(by_type[t])))]
            dicts.append({_LSE.cast(lbl): rng.uniform(-1e-2, 1e-2) for lbl in labels})
        return matrix_dict, dicts

    def test_layer_pairwise_commutator_skips_disjoint_support(self):
        # Error generators supported on disjoint sets of qubits commute exactly.
        # error_generator_commutator does not special-case them (its emitted terms cancel only
        # once aggregated), whereas _accumulate_layer_pairwise_commutators skips such pairs
        # outright using the labels' support masks. Check, on the weight-1 3-qubit basis, that
        # (a) the per-pair terms of every disjoint pair indeed sum to zero label by label (which
        # is what makes the skip exact) while every overlapping pair matches the numerical
        # commutator, and (b) the layerwise accumulation equals a direct sum over all pairs.
        errorgen_basis = CompleteElementaryErrorgenBasis('PP', QubitSpace(3), default_label_type='local', max_weights={'H': 1, 'S': 1, 'C': 1, 'A': 1})
        errorgen_lbls = errorgen_basis.labels
        errorgen_lbl_matrix_dict = {lbl: mat for lbl, mat in zip(errorgen_lbls, errorgen_basis.elemgen_matrices)}
        stim_lbls = [_LSE.cast(lbl) for lbl in errorgen_lbls]
        num_disjoint = 0
        for (lbl1, slbl1), (lbl2, slbl2) in product(zip(errorgen_lbls, stim_lbls), repeat=2):
            analytic = _eprop.error_generator_commutator(slbl1, slbl2)
            if slbl1.support_mask & slbl2.support_mask == 0:
                num_disjoint += 1
                summed = {}
                for lbl, rate in analytic:
                    summed[lbl] = summed.get(lbl, 0) + rate
                self.assertTrue(all(abs(rate) < 1e-12 for rate in summed.values()))
            else:
                # (the cancelling terms of disjoint pairs are weight-2 labels, outside this basis)
                numeric = _eprop.error_generator_commutator_numerical(lbl1, lbl2, errorgen_lbl_matrix_dict)
                analytic_mat = _eprop.errorgen_layer_to_matrix(analytic, 3, errorgen_lbl_matrix_dict)
                self.assertLess(np.linalg.norm(numeric - analytic_mat), 1e-10)
        self.assertGreater(num_disjoint, 0)

        rng = np.random.default_rng(1)
        layer_1 = {lbl: r for lbl, r in zip(stim_lbls, rng.normal(size=len(stim_lbls)))}
        layer_2 = {lbl: r for lbl, r in zip(stim_lbls, rng.normal(size=len(stim_lbls)))}
        accumulated = {}
        _eprop._accumulate_layer_pairwise_commutators(accumulated, layer_1, layer_2, 'III', addl_weight=0.5)
        expected = {}
        for lbl1, r1 in layer_1.items():
            for lbl2, r2 in layer_2.items():
                for lbl, rate in _eprop.error_generator_commutator(lbl1, lbl2, weight=0.5 * r1 * r2, identity='III'):
                    expected[lbl] = expected.get(lbl, 0) + rate
        # the direct sum also holds the exactly-cancelled labels of the skipped pairs
        self.assertTrue(set(accumulated) <= set(expected))
        for lbl, rate in expected.items():
            self.assertAlmostEqual(accumulated.get(lbl, 0), rate, places=12)

    def test_commuting_product(self):
        #_commuting_product(L, M) must equal the dense product L @ M whenever [L, M] = 0: check the square
        #L o L (same object passed twice), the cube L o L^2 (different objects) and the reversed L^2 o L,
        #against the dense matrices, for random all-sector 2- and 3-qubit dictionaries.
        for num_qubits, sizes, seed in ((2, (6, 15), 101), (3, (8, 18), 102)):
            matrix_dict, dicts = self._random_errorgen_dicts(num_qubits, sizes, seed)
            identity = 'I' * num_qubits
            for errorgen_dict in dicts:
                self.assertTrue({lbl.errorgen_type for lbl in errorgen_dict} == set('HSCA'))
                dense = _eprop.errorgen_layer_to_matrix(errorgen_dict, num_qubits, matrix_dict)
                square = _eprop._commuting_product(errorgen_dict, errorgen_dict, identity)
                self.assertLess(np.linalg.norm(_eprop.errorgen_layer_to_matrix(square, num_qubits, matrix_dict) - dense @ dense), 1e-12)
                cube = _eprop._commuting_product(errorgen_dict, square, identity)
                self.assertLess(np.linalg.norm(_eprop.errorgen_layer_to_matrix(cube, num_qubits, matrix_dict) - dense @ dense @ dense), 1e-12)
                cube_reversed = _eprop._commuting_product(square, errorgen_dict, identity)
                self.assertEqual(set(cube), set(cube_reversed))
                for lbl in cube:
                    self.assertAlmostEqual(cube[lbl], cube_reversed[lbl], places=14)
                #every two-index key of the result is in canonical order (the bleed terms are added on the
                #operands' own labels, which are canonical by the label contract).
                for lbl in chain(square, cube):
                    if len(lbl.basis_element_labels) == 2:
                        self.assertTrue(bel_less_than(*lbl.basis_element_labels), f'{lbl} not canonical')

    def test_error_generator_taylor_expansion_all_sectors(self):
        #order-k term == L^k / k! from the dense matrix, at threshold 0, on random all-sector dictionaries;
        #and the default threshold only removes terms below it.
        for num_qubits, seed in ((2, 104), (3, 105)):
            matrix_dict, (errorgen_dict,) = self._random_errorgen_dicts(num_qubits, (10,), seed)
            dense = _eprop.errorgen_layer_to_matrix(errorgen_dict, num_qubits, matrix_dict)
            terms = _eprop.error_generator_taylor_expansion(errorgen_dict, order=3, truncation_threshold=0.0)
            self.assertEqual(len(terms), 3)
            for k, order_dict in enumerate(terms, start=1):
                expected = np.linalg.matrix_power(dense, k) / factorial(k)
                self.assertLess(np.linalg.norm(_eprop.errorgen_layer_to_matrix(order_dict, num_qubits, matrix_dict) - expected), 1e-12)
            truncated = _eprop.error_generator_taylor_expansion(errorgen_dict, order=3, truncation_threshold=1e-9)
            for full, trunc in zip(terms, truncated):
                self.assertTrue(set(trunc) <= set(full))
                for lbl, rate in full.items():
                    if abs(rate) > 1e-9:
                        self.assertIn(lbl, trunc)
                        self.assertEqual(trunc[lbl], rate)
                    else:
                        self.assertNotIn(lbl, trunc)

    def test_pairwise_mode_deprecated(self):
        with self.assertWarns(pyGSTiDeprecationWarning) as cm:
            pairwise = self.errorgen_propagator.propagate_errorgens_bch(self.circuit, bch_order=1, mode='pairwise')
        self.assertIn("mode='magnus'", str(cm.warning))
        magnus = self.errorgen_propagator.propagate_errorgens_bch(self.circuit, bch_order=1)
        self.assertEqual(set(pairwise), set(magnus))
        for lbl, rate in magnus.items():
            self.assertAlmostEqual(pairwise[lbl], rate, places=14)
        with self.assertRaises(ValueError):
            self.errorgen_propagator.propagate_errorgens_bch(self.circuit, bch_order=1, mode='not_a_mode')

    def test_bch_approximation(self):
        # exercises the deprecated 'pairwise' mode (bch_approximation itself is not deprecated).
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', pyGSTiDeprecationWarning)
            self._check_bch_approximation()

    def _check_bch_approximation(self):
        first_order_bch_numerical = _eprop.bch_numerical(self.propagated_errorgen_layers, self.errorgen_propagator, bch_order=1)
        propagated_errorgen_layers_bch_order_1 = self.errorgen_propagator.propagate_errorgens_bch(self.circuit, bch_order=1, mode='pairwise')
        first_order_bch_analytical = self.errorgen_propagator.errorgen_layer_dict_to_errorgen(propagated_errorgen_layers_bch_order_1,mx_basis='pp')
        assert np.linalg.norm(first_order_bch_analytical-first_order_bch_numerical) < 1e-14
        
        propagated_errorgen_layers_bch_order_2 = self.errorgen_propagator.propagate_errorgens_bch(self.circuit, bch_order=2, mode='pairwise')
        second_order_bch_numerical = _eprop.bch_numerical(self.propagated_errorgen_layers, self.errorgen_propagator, bch_order=2)
        second_order_bch_analytical = self.errorgen_propagator.errorgen_layer_dict_to_errorgen(propagated_errorgen_layers_bch_order_2, mx_basis='pp')
        assert np.linalg.norm(second_order_bch_analytical-second_order_bch_numerical) < 1e-14

        third_order_bch_numerical = _eprop.bch_numerical(self.propagated_errorgen_layers, self.errorgen_propagator, bch_order=3)
        propagated_errorgen_layers_bch_order_3 = self.errorgen_propagator.propagate_errorgens_bch(self.circuit, bch_order=3, mode='pairwise')
        third_order_bch_analytical = self.errorgen_propagator.errorgen_layer_dict_to_errorgen(propagated_errorgen_layers_bch_order_3, mx_basis='pp')
        assert np.linalg.norm(third_order_bch_analytical-third_order_bch_numerical) < 1e-14

        fourth_order_bch_numerical = _eprop.bch_numerical(self.propagated_errorgen_layers, self.errorgen_propagator, bch_order=4)
        propagated_errorgen_layers_bch_order_4 = self.errorgen_propagator.propagate_errorgens_bch(self.circuit, bch_order=4, mode='pairwise')
        fourth_order_bch_analytical = self.errorgen_propagator.errorgen_layer_dict_to_errorgen(propagated_errorgen_layers_bch_order_4, mx_basis='pp')
        assert np.linalg.norm(fourth_order_bch_analytical-fourth_order_bch_numerical) < 1e-14

        fifth_order_bch_numerical = _eprop.bch_numerical(self.propagated_errorgen_layers, self.errorgen_propagator, bch_order=5)
        propagated_errorgen_layers_bch_order_5 = self.errorgen_propagator.propagate_errorgens_bch(self.circuit, bch_order=5, truncation_threshold=0, mode='pairwise')
        fifth_order_bch_analytical = self.errorgen_propagator.errorgen_layer_dict_to_errorgen(propagated_errorgen_layers_bch_order_5, mx_basis='pp')
        assert np.linalg.norm(fifth_order_bch_analytical-fifth_order_bch_numerical) < 1e-14

        exact_errorgen = logm(self.errorgen_propagator.eoc_error_channel(self.circuit))
        exact_vs_first_order_norm  = np.linalg.norm(first_order_bch_analytical-exact_errorgen)
        exact_vs_second_order_norm = np.linalg.norm(second_order_bch_analytical-exact_errorgen)
        exact_vs_third_order_norm  = np.linalg.norm(third_order_bch_analytical-exact_errorgen)
        exact_vs_fourth_order_norm = np.linalg.norm(fourth_order_bch_analytical-exact_errorgen)
        exact_vs_fifth_order_norm  = np.linalg.norm(fifth_order_bch_analytical-exact_errorgen)
        
        self.assertTrue((exact_vs_first_order_norm > exact_vs_second_order_norm) and (exact_vs_second_order_norm > exact_vs_third_order_norm)
                        and (exact_vs_third_order_norm > exact_vs_fourth_order_norm) and (exact_vs_fourth_order_norm > exact_vs_fifth_order_norm))

    def test_merged_orders_are_truncated(self):
        # The per-order terms of bch_approximation and magnus_expansion are each truncated, but a
        # label's contributions from different orders can cancel when they are summed. Arrange an
        # expansion where a near-cancellation drops the merged rate below the truncation threshold:
        # the result must not retain this sub-threshold label.
        x_lbl = _LSE.cast(('H', ('X',)))
        y_lbl = _LSE.cast(('H', ('Y',)))
        z_lbl = _LSE.cast(('H', ('Z',)))
        # [H_X, H_Y] = 2 H_Z. In bch_approximation(L1, L2, bch_order=2) the second-order term is
        # (1/2) [L1, L2] = H_Z. If L1 also contains -H_Z with rate slightly smaller than -1.0, the
        # merged rate for H_Z is just above zero but below the truncation threshold.
        eps = 1e-8
        l1 = {x_lbl: 1.0, z_lbl: -1.0 + eps}
        l2 = {y_lbl: 1.0}
        # With threshold > eps the merged H_Z term must be pruned
        res = _eprop.bch_approximation(l1, l2, bch_order=2, truncation_threshold=1e-6)
        self.assertNotIn(z_lbl, res)
        # With threshold < eps it is preserved
        res_kept = _eprop.bch_approximation(l1, l2, bch_order=2, truncation_threshold=1e-10)
        self.assertIn(z_lbl, res_kept)
        self.assertAlmostEqual(res_kept[z_lbl], eps, places=12)

    def test_magnus_expansion(self):
        first_order_magnus_numerical = _eprop.magnus_numerical(self.propagated_errorgen_layers, self.errorgen_propagator, magnus_order=1)
        first_order_magnus_analytical = _eprop.magnus_expansion(self.propagated_errorgen_layers, magnus_order=1)
        first_order_magnus_analytical = self.errorgen_propagator.errorgen_layer_dict_to_errorgen(first_order_magnus_analytical, mx_basis='pp')
        assert np.linalg.norm(first_order_magnus_analytical-first_order_magnus_numerical) < 1e-14
        
        second_order_magnus_numerical = _eprop.magnus_numerical(self.propagated_errorgen_layers, self.errorgen_propagator, magnus_order=2)
        second_order_magnus_analytical = _eprop.magnus_expansion(self.propagated_errorgen_layers, magnus_order=2)
        second_order_magnus_analytical = self.errorgen_propagator.errorgen_layer_dict_to_errorgen(second_order_magnus_analytical, mx_basis='pp')
        assert np.linalg.norm(second_order_magnus_analytical-second_order_magnus_numerical) < 1e-14

        third_order_magnus_numerical = _eprop.magnus_numerical(self.propagated_errorgen_layers, self.errorgen_propagator, magnus_order=3)
        third_order_magnus_analytical = _eprop.magnus_expansion(self.propagated_errorgen_layers, magnus_order=3)
        third_order_magnus_analytical = self.errorgen_propagator.errorgen_layer_dict_to_errorgen(third_order_magnus_analytical, mx_basis='pp')
        assert np.linalg.norm(third_order_magnus_analytical-third_order_magnus_numerical) < 1e-14

        exact_errorgen = logm(self.errorgen_propagator.eoc_error_channel(self.circuit))
        exact_vs_first_order_norm  = np.linalg.norm(first_order_magnus_analytical-exact_errorgen)
        exact_vs_second_order_norm = np.linalg.norm(second_order_magnus_analytical-exact_errorgen)
        exact_vs_third_order_norm  = np.linalg.norm(third_order_magnus_analytical-exact_errorgen)
        self.assertTrue((exact_vs_first_order_norm > exact_vs_second_order_norm) and (exact_vs_second_order_norm > exact_vs_third_order_norm))

    def test_second_order_magnus_decision_uses_all_layers(self):
        # Regression: the parallelization decision previously averaged only the first two layers,
        # so an empty (or small) state-prep layer disabled parallelism for the whole expansion.
        layers = [{}] + list(self.propagated_errorgen_layers)
        expected_avg = sum(len(l) for l in layers) // len(layers)
        with mock.patch.object(_eprop.parallelutils, '_should_parallelize_magnus', return_value=False) as decision:
            _eprop.magnus_expansion(layers, magnus_order=2, num_workers=2)
        self.assertEqual(decision.call_args.args[1], expected_avg)

    def test_magnus_expansion_all_sectors(self):
        # The model of setUp has H errors only, so magnus_expansion's commutator terms exercise
        # a single type-pair branch there and the third-order term is ~1e-12. This model has
        # seeded random rates ~1e-2 from all four sectors on every gate (a subset of a random
        # CPTP rate dictionary; not itself CP, which is irrelevant for propagation), giving a
        # third-order term of norm ~2e-4 that involves every commutator branch and running-sum
        # bookkeeping path. With truncation disabled the analytic expansion must match the
        # dense oracle to machine precision at every order; with the default threshold the
        # error is set by the dropped sub-threshold terms and must stay small relative to the
        # third-order term.
        pspec = QubitProcessorSpec(4, ['Gcphase', 'Gxpi2', 'Gypi2'], availability={'Gcphase': [(0, 1), (1, 2), (2, 3), (3, 0)]})
        rates = {'Gxpi2': _all_sector_rate_subset(1, 2, seed=100), 'Gypi2': _all_sector_rate_subset(1, 2, seed=101),
                 'Gcphase': _all_sector_rate_subset(2, 2, seed=200)}
        model = create_crosstalk_free_model(pspec, lindblad_error_coeffs=rates, lindblad_parameterization='GLND')
        circuit = create_random_circuit(pspec, 4, sampler='edgegrab', samplerargs=[0.4, ], rand_state=12345)
        propagator = ErrorGeneratorPropagator(model.copy())
        layers = propagator.propagate_errorgens(circuit)
        self.assertEqual(len(layers), 6)
        self.assertEqual({lbl.errorgen_type for layer in layers for lbl in layer}, set('HSCA'))

        numerical = [_eprop.magnus_numerical(layers, propagator, magnus_order=order) for order in (1, 2, 3)]
        for order, numerical_mat in zip((1, 2, 3), numerical):
            analytic = _eprop.magnus_expansion(layers, magnus_order=order, truncation_threshold=0)
            analytic_mat = propagator.errorgen_layer_dict_to_errorgen(analytic, mx_basis='pp')
            self.assertLess(np.linalg.norm(analytic_mat - numerical_mat), 1e-14)
        third_order_term_norm = np.linalg.norm(numerical[2] - numerical[1])
        self.assertGreater(third_order_term_norm, 1e-5)
        analytic = _eprop.magnus_expansion(layers, magnus_order=3)
        analytic_mat = propagator.errorgen_layer_dict_to_errorgen(analytic, mx_basis='pp')
        self.assertLess(np.linalg.norm(analytic_mat - numerical[2]), 1e-6 * third_order_term_norm)

    def test_magnus_expansion_third_order_bookkeeping(self):
        # Check the third-order Magnus term against its row/column form, evaluated explicitly
        # for n = 4 layers with the pairwise commutators P_ik = [A(i), A(k)] (i > k):
        #   T1 = (1/6) sum_i [A(i), sum_{j<i} row_j + row_i/2],   row_i = sum_{k<i} P_ik
        #   T2 = -(1/6) sum_k [A(k), sum_{j>k} col_j + col_k/2],  col_k = sum_{i>k} P_ik
        # (the 1/2 is the boundary weight of the discretized time-ordered integral). This is
        # independent of how magnus_expansion organizes the running sums, and cheap enough to
        # run at full precision on two qubits with random labels from all sectors.
        rng = np.random.default_rng(7)
        basis = CompleteElementaryErrorgenBasis('PP', QubitSpace(2), default_label_type='local')
        stim_lbls = [_LSE.cast(lbl) for lbl in basis.labels]
        layers = []
        for _ in range(4):
            chosen = rng.choice(len(stim_lbls), size=6, replace=False)
            layers.append({stim_lbls[i]: 1e-2 * rng.standard_normal() for i in chosen})
        n, identity = len(layers), 'II'

        def commutator(layer_1, layer_2, weight=1.0):
            out = {}
            _eprop._accumulate_layer_pairwise_commutators(out, layer_1, layer_2, identity, addl_weight=weight, truncation_threshold=0)
            return out

        def add(target, source, scale=1.0):
            for lbl, rate in source.items():
                target[lbl] = target.get(lbl, 0) + scale * rate

        P = {(i, k): commutator(layers[i], layers[k]) for i in range(n) for k in range(i)}
        rows = [{} for _ in range(n)]
        cols = [{} for _ in range(n)]
        for (i, k), p in P.items():
            add(rows[i], p)
            add(cols[k], p)
        expected = {}
        for m in range(n):
            row_sum, col_sum = {}, {}
            for j in range(m):
                add(row_sum, rows[j])
            add(row_sum, rows[m], 0.5)
            for j in range(m + 1, n):
                add(col_sum, cols[j])
            add(col_sum, cols[m], 0.5)
            add(expected, commutator(layers[m], row_sum), 1 / 6)
            add(expected, commutator(layers[m], col_sum), -1 / 6)
        expected = {lbl: rate.real for lbl, rate in expected.items() if abs(rate) > 1e-20}

        second = _eprop.magnus_expansion(layers, magnus_order=2, truncation_threshold=0)
        third = _eprop.magnus_expansion(layers, magnus_order=3, truncation_threshold=0)
        actual = {lbl: third.get(lbl, 0) - second.get(lbl, 0) for lbl in set(third) | set(second)}
        actual = {lbl: rate for lbl, rate in actual.items() if abs(rate) > 1e-20}
        self.assertGreater(len(expected), 10)
        self.assertEqual(set(actual), set(expected))
        scale = max(abs(rate) for rate in expected.values())
        for lbl, rate in expected.items():
            self.assertLess(abs(actual[lbl] - rate), 1e-11 * scale)

    def test_zassenhaus_formula(self):
        first_order_zassenhaus_numerical = _eprop.zassenhaus_formula_numerical(self.propagated_errorgen_layers, self.errorgen_propagator, zassenhaus_order=1)
        first_order_zassenhaus_analytical = _dense_zassenhaus_generators_analytic(self.propagated_errorgen_layers, self.errorgen_propagator, zassenhaus_order=1)
        assert all([np.linalg.norm(analytic-numerical) < 1e-14 for analytic, numerical in zip(first_order_zassenhaus_analytical,first_order_zassenhaus_numerical)])
        
        second_order_zassenhaus_numerical = _eprop.zassenhaus_formula_numerical(self.propagated_errorgen_layers, self.errorgen_propagator, zassenhaus_order=2)
        second_order_zassenhaus_analytical = _dense_zassenhaus_generators_analytic(self.propagated_errorgen_layers, self.errorgen_propagator, zassenhaus_order=2)
        assert all([np.linalg.norm(analytic-numerical) < 1e-14 for analytic, numerical in zip(second_order_zassenhaus_analytical,second_order_zassenhaus_numerical)])

        exact_channel = self.errorgen_propagator.eoc_error_channel(self.circuit, use_bch=True, bch_kwargs={'bch_order':1})
        exact_vs_first_order_norm  = np.linalg.norm(np.linalg.multi_dot([expm(gen) for gen in first_order_zassenhaus_analytical])-exact_channel)
        exact_vs_second_order_norm = np.linalg.norm(np.linalg.multi_dot([expm(gen) for gen in second_order_zassenhaus_analytical])-exact_channel)
        
        self.assertTrue(exact_vs_first_order_norm > exact_vs_second_order_norm)

    def test_magnus_expansion_empty_first_layer(self):
        # Empty first layer should correctly infer n_q from subsequent layer
        res = _eprop.magnus_expansion([{}, self.propagated_errorgen_layer], magnus_order=2)
        self.assertIsInstance(res, dict)
        self.assertGreater(len(res), 0)

    def test_magnus_expansion_all_empty_layers(self):
        # All empty layers should short-circuit and return empty dicts without error
        res_m2 = _eprop.magnus_expansion([{}, {}], magnus_order=2)
        self.assertEqual(res_m2, {})
        res_m3 = _eprop.magnus_expansion([{}, {}], magnus_order=3)
        self.assertEqual(res_m3, {})
        res_s2 = _eprop._second_order_magnus_term([{}, {}], identity=None)
        self.assertEqual(res_s2, {})
        res_s3_2, res_s3_3 = _eprop._second_and_third_order_magnus_terms([{}, {}], identity=None)
        self.assertEqual(res_s3_2, {})
        self.assertEqual(res_s3_3, {})

    def test_taylor_expansion_with_truncated_powers_matches_untruncated_powers(self):
        # The powers used to build the higher orders are truncated at the threshold. Compare with the
        # reported terms computed from untruncated powers, at several overall scales of the generator.
        thr = 1e-14
        for scale in (1.0, 30.0):
            rates = {lbl: scale * rate for lbl, rate in self.propagated_errorgen_layer.items()}
            taylor = _eprop.error_generator_taylor_expansion(rates, order=4, truncation_threshold=thr)
            power = rates
            for k in range(2, 5):
                power = _eprop._raw_commuting_product(rates, power, 'IIII')  # untruncated powers
                reference = {lbl: rate / factorial(k) for lbl, rate in power.items() if abs(rate) / factorial(k) > thr}
                got = taylor[k - 1]
                for lbl in set(reference) | set(got):
                    self.assertLessEqual(abs(reference.get(lbl, 0) - got.get(lbl, 0)), 0.1 * thr + 1e-12 * abs(reference.get(lbl, 0)))
                self.assertTrue(all(abs(v) > thr for v in got.values()))

    def test_error_generator_taylor_expansion(self):
        #the propagated layer's expansion against the dense numerical one (sum of L^k / k!) at order 2; the
        #order-3, all-sector check on random 2- and 3-qubit dictionaries is in ErrgenCompositionCommutationTester
        #(the dense 4-qubit conversion of every label of the result is what makes this test expensive).
        taylor_terms = _eprop.error_generator_taylor_expansion(self.propagated_errorgen_layer, order=2, truncation_threshold=0.0)
        self.assertEqual(len(taylor_terms), 2)
        total = {}
        for order_dict in taylor_terms:
            for lbl, rate in order_dict.items():
                total[lbl] = total.get(lbl, 0) + rate
        numerical = _eprop.error_generator_taylor_expansion_numerical(self.propagated_errorgen_layer, self.error_propagator, order=2)
        analytic = self.error_propagator.errorgen_layer_dict_to_errorgen(total, 'pp')
        self.assertLess(np.linalg.norm(numerical - analytic), 1e-12)

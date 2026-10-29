import copy
from itertools import combinations, product
import math
import sys
from unittest import mock

import numpy as np
import stim

from pygsti.algorithms.randomcircuit import create_random_circuit
from pygsti.errorgenpropagation.errorpropagator import ErrorGeneratorPropagator
from pygsti.errorgenpropagation.localstimerrorgen import LocalStimErrorgenLabel as _LSE
from pygsti.models.modelconstruction import create_crosstalk_free_model
from pygsti.processors import QubitProcessorSpec
from pygsti.tools import errorgenpropagation as _eprop

from ...util import BaseCase
from ..mptools_test_utils import PoolDispatchSpy as _PoolDispatchSpy
from .errgenprop_test_utils import _all_sector_rate_subset, parallel_path_exercised


class ErrgenParallelUtilsTester(BaseCase):

    def setUp(self):
        num_qubits = 4
        gate_names = ['Gcphase', 'Gxpi2', 'Gypi2']
        availability = {'Gcphase': [(0, 1), (1, 2), (2, 3), (3, 0)]}
        pspec = QubitProcessorSpec(num_qubits, gate_names, availability=availability)
        self.pspec = pspec
        self.circuit = create_random_circuit(pspec, 4, sampler='edgegrab', samplerargs=[0.4], rand_state=12345)

        rates = {
            'Gxpi2': _all_sector_rate_subset(1, 1, seed=100),
            'Gypi2': _all_sector_rate_subset(1, 1, seed=101),
            'Gcphase': _all_sector_rate_subset(2, 1, seed=200),
        }
        self.error_model = create_crosstalk_free_model(pspec, lindblad_error_coeffs=rates, lindblad_parameterization='GLND')
        self.errorgen_propagator = ErrorGeneratorPropagator(self.error_model.copy())
        self.propagated_errorgen_layers = self.errorgen_propagator.propagate_errorgens(self.circuit)
        self.propagated_errorgen_layer = self.errorgen_propagator.propagate_errorgens_bch(self.circuit, bch_order=1)
        self.circuit_tableau = self.circuit.convert_to_stim_tableau()

    def test_magnus_expansion_parallel(self):
        seq = _eprop.magnus_expansion(self.propagated_errorgen_layers, magnus_order=2, num_workers=None)
        with parallel_path_exercised(self, ['magnus'], ['_MagnusPairEvaluator']):
            par = _eprop.magnus_expansion(self.propagated_errorgen_layers, magnus_order=2, num_workers=2)
        self.assertEqual(len(seq), len(par))
        self.assertEqual(set(seq.keys()), set(par.keys()))
        for k in seq.keys():
            self.assertAlmostEqual(seq[k], par[k], places=12)

    def test_magnus_expansion_parallel_order3(self):
        pspec = QubitProcessorSpec(4, ['Gcphase', 'Gxpi2', 'Gypi2'], availability={'Gcphase': [(0, 1), (1, 2), (2, 3), (3, 0)]})
        rates = {'Gxpi2': _all_sector_rate_subset(1, 1, seed=100), 'Gypi2': _all_sector_rate_subset(1, 1, seed=101),
                 'Gcphase': _all_sector_rate_subset(2, 1, seed=200)}
        model = create_crosstalk_free_model(pspec, lindblad_error_coeffs=rates, lindblad_parameterization='GLND')
        circuit = create_random_circuit(pspec, 4, sampler='edgegrab', samplerargs=[0.4, ], rand_state=12345)
        propagator = ErrorGeneratorPropagator(model.copy())
        layers = propagator.propagate_errorgens(circuit)

        seq = _eprop.magnus_expansion(layers, magnus_order=3, num_workers=None)
        with parallel_path_exercised(self, ['magnus'], ['_MagnusPhase1PairEvaluator', '_MagnusPhase2Evaluator']):
            par = _eprop.magnus_expansion(layers, magnus_order=3, num_workers=2)
        self.assertEqual(set(seq.keys()), set(par.keys()))
        for k in seq.keys():
            self.assertAlmostEqual(seq[k], par[k], places=12)

    def test_in_place_magnus_merge_functions(self):
        # Reference implementations that copy (the previous behaviour)
        def ref_merge(d1, d2):
            res = dict(d1)
            for k, v in d2.items():
                res[k] = res.get(k, 0.0) + v
            return res

        def ref_indexed_merge(d1, d2):
            res = {k: dict(v) for k, v in d1.items()}
            for pk, rates in d2.items():
                sub = res.setdefault(pk, {})
                for lbl, r in rates.items():
                    sub[lbl] = sub.get(lbl, 0.0) + r
            return res

        a = {'HXI': 1.0, 'SZZ': 2.0}
        b = {'HXI': 0.5, 'CXIYZ': -3.0}
        c = {'SZZ': 1.0, 'AXIZY': 4.0}
        expected = ref_merge(ref_merge(a, b), c)
        a_in, b_in, c_in = dict(a), dict(b), dict(c)
        out = _eprop._merge_string_dictionaries(_eprop._merge_string_dictionaries(a_in, b_in), c_in)
        self.assertEqual(out, expected)
        self.assertIs(out, a_in)             # updated in place and returned
        self.assertEqual(b_in, b)            # the second argument is never modified
        self.assertEqual(c_in, c)

        d1 = {(1, 0): {'HXI': 1.0}, (2, 0): {'SZZ': 2.0}}
        d2 = {(1, 0): {'HXI': 0.5, 'CXIYZ': 1.0}, (3, 1): {'AXIZY': 4.0}}  # a repeated pair (sliced) and a new one
        d3 = {(2, 0): {'SZZ': -1.0}, (3, 1): {'AXIZY': 1.0, 'HYI': 2.0}}
        expected = ref_indexed_merge(ref_indexed_merge(d1, d2), d3)
        d1_in, d2_in, d3_in = copy.deepcopy(d1), copy.deepcopy(d2), copy.deepcopy(d3)
        out = _eprop._merge_pair_indexed_string_dictionaries(
            _eprop._merge_pair_indexed_string_dictionaries(d1_in, d2_in), d3_in)
        self.assertEqual(out, expected)
        self.assertIs(out, d1_in)
        self.assertEqual(d3_in, d3)  # the second argument is never modified
        self.assertEqual(d2_in[(1, 0)], d2[(1, 0)])
        # A per-pair dictionary of the second argument that is new to the total is stored in it by reference, so
        # it is then updated by later merges: this is why a result must not be reused after being merged in.
        self.assertIs(d1_in[(3, 1)], d2_in[(3, 1)])

    def test_magnus_evaluators_return_fresh_objects(self):
        # The in-place merges (and `parallel_pair_reduce`'s documented contract) need every evaluator call
        # to return a new object that shares no inner dictionaries with a previous result or with its input.
        layers = [l for l in self.propagated_errorgen_layers if l][:3]
        identity = 'IIII'
        tasks, _ = _eprop._generate_balanced_magnus_pairs(layers, 2, include_indices=False)
        tasks3, _ = _eprop._generate_balanced_magnus_pairs(layers, 2, include_indices=True)
        pairs = [(_eprop._MagnusPairEvaluator(identity, 1e-14), tasks[0]),
                 (_eprop._MagnusPhase1PairEvaluator(identity, 1e-14), tasks3[0]),
                 (_eprop._MagnusPhase2Evaluator(identity, 1e-14), (layers[0], layers[1], 1.0))]
        for evaluator, task in pairs:
            r1, r2 = evaluator(task), evaluator(task)
            self.assertIsNot(r1, r2)
            self.assertEqual(r1, r2)
            if isinstance(next(iter(r1.values()), None), dict):  # phase 1: {(i, k): {label: rate}}
                for key in r1:
                    self.assertIsNot(r1[key], r2[key])

    def test_zassenhaus_formula_parallel(self):
        seq = _eprop.zassenhaus_formula(self.propagated_errorgen_layers, zassenhaus_order=2, num_workers=None)
        with parallel_path_exercised(self, ['magnus'], ['_MagnusPairEvaluator']):
            par = _eprop.zassenhaus_formula(self.propagated_errorgen_layers, zassenhaus_order=2, num_workers=2)
        self.assertEqual(len(seq), len(par))
        for s_dict, p_dict in zip(seq, par):
            self.assertEqual(set(s_dict.keys()), set(p_dict.keys()))
            for k in s_dict.keys():
                self.assertAlmostEqual(s_dict[k], p_dict[k], places=12)

    def test_bulk_alpha_parallel(self):
        tableau = self.circuit_tableau
        errorgens = list(self.propagated_errorgen_layer.keys())
        bitstrings = ['0000', '0001', '1010', '1111']
        seq = _eprop.bulk_alpha(errorgens, tableau, bitstrings, num_workers=None)
        with parallel_path_exercised(self, ['alpha'], ['_grid_task_worker']):
            par = _eprop.bulk_alpha(errorgens, tableau, bitstrings, num_workers=2)
        self.assertArraysAlmostEqual(par, seq)

    def test_bulk_alpha_pauli_parallel(self):
        tableau = self.circuit_tableau
        errorgens = list(self.propagated_errorgen_layer.keys())
        paulis = [stim.PauliString('XXXX'), stim.PauliString('ZIII'), stim.PauliString('YYYY')]
        seq = _eprop.bulk_alpha_pauli(errorgens, tableau, paulis, num_workers=None)
        with parallel_path_exercised(self, ['alpha'], ['_grid_task_worker']):
            par = _eprop.bulk_alpha_pauli(errorgens, tableau, paulis, num_workers=2)
        self.assertArraysAlmostEqual(par, seq)

    def test_should_parallelize_alpha(self):
        self.assertFalse(_eprop._should_parallelize_alpha(100, 100, 1, num_workers=1))
        self.assertTrue(_eprop._should_parallelize_alpha(100, 10000, 16, num_workers=4))

    def test_should_parallelize_alpha_dynamic_fallback_on_missing_param(self):
        # This test modifies module-level calibration state: always restore the defaults, even on failure.
        self.addCleanup(_eprop.configure_errorgen_propagation_parallelization, reset_to_defaults=True)

        # Set dynamic_calibrated=True but omit alpha_a / alpha_b
        _eprop._ACTIVE_PARALLEL_THRESHOLDS["dynamic_calibrated"] = True
        _eprop._ACTIVE_PARALLEL_THRESHOLDS.pop("alpha_a", None)
        _eprop._ACTIVE_PARALLEL_THRESHOLDS.pop("alpha_b", None)

        # Falls back to the default table: for n_qubits=4 ('small' scale) the threshold is 1000
        small_thresh = _eprop.DEFAULT_PARALLEL_THRESHOLDS['small']['min_km_alpha']
        self.assertEqual(small_thresh, 1000)
        self.assertFalse(_eprop._should_parallelize_alpha(4, small_thresh - 100, 1, num_workers=2))
        self.assertTrue(_eprop._should_parallelize_alpha(4, small_thresh + 100, 1, num_workers=2))

    def test_configure_errorgen_propagation_parallelization(self):
        self.addCleanup(_eprop.configure_errorgen_propagation_parallelization, reset_to_defaults=True)
        # Override min_km_alpha to high value
        _eprop.configure_errorgen_propagation_parallelization(min_km_alpha=50000)
        self.assertFalse(_eprop._should_parallelize_alpha(100, 1000, 4, num_workers=4))

        # Force serial
        _eprop.configure_errorgen_propagation_parallelization(force_serial=True)
        self.assertFalse(_eprop._should_parallelize_alpha(100, 100000, 16, num_workers=4))

        # Force parallel
        _eprop.configure_errorgen_propagation_parallelization(force_serial=False, force_parallel=True)
        self.assertTrue(_eprop._should_parallelize_alpha(100, 10, 1, num_workers=4))

        # Reset to defaults
        res = _eprop.configure_errorgen_propagation_parallelization(reset_to_defaults=True)
        self.assertEqual(res['status'], 'reset_to_defaults')

    def test_public_parallelization_constants_are_read_only(self):
        table = _eprop.DEFAULT_PARALLEL_THRESHOLDS
        self.assertEqual(set(table), {'small', 'medium', 'large'})
        with self.assertRaises(TypeError):
            table['small'] = {}
        with self.assertRaises(TypeError):
            table['small']['min_km_alpha'] = -1
        self.assertIsInstance(_eprop.EGPROP_PREWARM_PACKAGES, tuple)
        self.assertIn('stim', _eprop.EGPROP_PREWARM_PACKAGES)
        # every tier defines the same cutoffs
        self.assertEqual(len({frozenset(t) for t in table.values()}), 1)

    def test_configure_parallelization_validation(self):
        configure = _eprop.configure_errorgen_propagation_parallelization
        self.addCleanup(configure, reset_to_defaults=True)

        # A bare operation name is treated as a single-element sequence.
        configure(force_parallel='magnus')
        self.assertTrue(_eprop._should_parallelize_magnus(3, 1, num_workers=4, n_qubits=4))
        configure(force_parallel=False)

        # Unknown operation names and invalid types are rejected, leaving the configuration unchanged.
        with self.assertRaises(ValueError):
            configure(force_parallel=['magnu'])
        with self.assertRaises(ValueError):
            configure(force_serial='not_an_operation')
        with self.assertRaises(TypeError):
            configure(force_serial=3)
        self.assertIsNone(_eprop._forced_parallelism_mode('magnus'))

        # An operation cannot be forced both parallel and serial.
        configure(force_serial=['magnus'])
        with self.assertRaises(ValueError):
            configure(force_parallel=['magnus'])
        with self.assertRaises(ValueError):
            configure(force_parallel=True)
        self.assertFalse(_eprop._forced_parallelism_mode('magnus'))
        configure(reset_to_defaults=True)

        # min_terms_per_magnus_pair takes effect on its own (min_magnus_pairs left at its default).
        configure(min_terms_per_magnus_pair=10**9)
        self.assertFalse(_eprop._should_parallelize_magnus(100, 500, num_workers=4, n_qubits=4))
        configure(min_terms_per_magnus_pair=1)
        self.assertTrue(_eprop._should_parallelize_magnus(100, 2, num_workers=4, n_qubits=4))
        configure(reset_to_defaults=True)

        # reset_to_defaults returns a deep copy: mutating it does not alter the defaults.
        res = configure(reset_to_defaults=True)
        res['defaults']['small']['min_km_alpha'] = -1
        self.assertNotEqual(_eprop.DEFAULT_PARALLEL_THRESHOLDS['small']['min_km_alpha'], -1)

    def test_optimize_errorgen_propagation_parallelization(self):
        self.addCleanup(_eprop.configure_errorgen_propagation_parallelization, reset_to_defaults=True)
        # Missing all three inputs raises ValueError
        with self.assertRaises(ValueError):
            _eprop.optimize_errorgen_propagation_parallelization()

        with self.assertRaises(ValueError):
            _eprop.optimize_errorgen_propagation_parallelization(num_qubits=0)

        # Invalid argument types raise TypeError
        with self.assertRaises(TypeError):
            _eprop.optimize_errorgen_propagation_parallelization(model="not_a_model")
        with self.assertRaises(TypeError):
            _eprop.optimize_errorgen_propagation_parallelization(circuit="not_a_circuit")
        with self.assertRaises(TypeError):
            _eprop.optimize_errorgen_propagation_parallelization(num_qubits="not_an_int")

        # Call with explicit circuit
        res_circ = _eprop.optimize_errorgen_propagation_parallelization(circuit=self.circuit, num_workers=2, set_as_default=False)
        self.assertIsInstance(res_circ, dict)
        self.assertEqual(res_circ['num_qubits'], self.circuit.num_lines)

        res = _eprop.optimize_errorgen_propagation_parallelization(num_qubits=4, num_workers=2, set_as_default=True)
        self.assertIsInstance(res, dict)
        self.assertIn('tau_alpha_us', res)
        self.assertIn('threshold_KM', res)
        # Clean up
        _eprop.configure_errorgen_propagation_parallelization(reset_to_defaults=True)

    def test_alpha_calibration_profiling_inputs_for_all_qubit_counts(self):
        # The profiling stabilizer state has exactly the requested number of qubits (including odd
        # counts and a single qubit), and calibration runs end to end for each.
        for n in (1, 2, 3, 5, 8):
            tableau, gens = _eprop._calibration_tableau_and_errorgens(n, num_errorgens=20)
            self.assertEqual(len(tableau), n)
            self.assertEqual(len(gens), 20)
            self.assertTrue(all(len(g.basis_element_labels[0]) == n for g in gens))
            self.assertTrue(_eprop._measure_alpha_throughput_us(n, repeats=1) >= 0.1)
        self.addCleanup(_eprop.configure_errorgen_propagation_parallelization, reset_to_defaults=True)
        for n in (1, 3):
            res = _eprop.optimize_errorgen_propagation_parallelization(num_qubits=n, num_workers=2, set_as_default=False)
            self.assertEqual(res['num_qubits'], n)
            self.assertGreater(res['tau_alpha_us'], 0)
            self.assertGreaterEqual(res['alpha_b'], 0)
            self.assertGreaterEqual(res['threshold_KM'], 1)

    def test_fit_alpha_scaling(self):
        a, b = _eprop._fit_alpha_scaling(4, 2.0, 20, 2.8)
        self.assertAlmostEqual(b, 0.05)
        self.assertAlmostEqual(a, 1.8)
        # the fitted line is a local one; extrapolating it to other sizes cannot give a non-positive throughput
        self.assertEqual(_eprop._alpha_tau_us(-1.7, 0.3, 5), _eprop._MIN_ALPHA_TAU_US)
        self.assertAlmostEqual(_eprop._alpha_tau_us(2.0, 0.05, 10), 2.5)
        # a negative slope (measurement noise) is clamped to a constant model through the mean
        self.assertEqual(_eprop._fit_alpha_scaling(4, 3.0, 20, 2.0), (2.5, 0.0))

    def test_alpha_calibration_values_and_decision_threshold(self):
        configure = _eprop.configure_errorgen_propagation_parallelization
        self.addCleanup(configure, reset_to_defaults=True)
        n, t_ipc = 10, 1e-4
        fake_tau = lambda m: 2.0 + 0.05 * m  # microseconds per cell at m qubits
        with mock.patch.object(_eprop.parallelutils, '_measure_alpha_throughput_us', side_effect=fake_tau) as timing, \
                mock.patch.object(_eprop.parallelutils, '_measure_ipc_latency', return_value=t_ipc):
            res = _eprop.optimize_errorgen_propagation_parallelization(num_qubits=n, num_workers=2)
            # measured at the target size and `_ALPHA_CALIBRATION_QUBIT_SPAN` qubits above it
            self.assertEqual([c.args[0] for c in timing.call_args_list], [n, n + _eprop._ALPHA_CALIBRATION_QUBIT_SPAN])
            self.assertAlmostEqual(res['alpha_a'], 2.0)
            self.assertAlmostEqual(res['alpha_b'], 0.05)
            self.assertAlmostEqual(res['tau_alpha_us'], fake_tau(n))
            beta = _eprop._ACTIVE_PARALLEL_THRESHOLDS['safety_factor']
            expected = beta * t_ipc / (fake_tau(n) * 1e-6)  # 400 for the default safety factor of 10
            self.assertEqual(res['threshold_KM'], math.ceil(expected))
            km = res['threshold_KM']

            # the decision flips exactly at the reported threshold
            self.assertFalse(_eprop._should_parallelize_alpha(n, km - 1, 1, num_workers=2))
            self.assertTrue(_eprop._should_parallelize_alpha(n, km, 1, num_workers=2))
            self.assertFalse(_eprop._should_parallelize_alpha(n, 1, km - 1, num_workers=2))
            self.assertTrue(_eprop._should_parallelize_alpha(n, 1, km, num_workers=2))

            # the configured safety factor is used (not a hard-coded value)
            with mock.patch.dict(_eprop._ACTIVE_PARALLEL_THRESHOLDS, {'safety_factor': 2 * beta}):
                res2 = _eprop.optimize_errorgen_propagation_parallelization(num_qubits=n, num_workers=2,
                                                                            set_as_default=False)
            self.assertAlmostEqual(res2['threshold_KM'] / res['threshold_KM'], 2.0, places=2)

        # reset clears all of the calibration state, restoring the table-based decision
        configure(reset_to_defaults=True)
        self.assertEqual(set(_eprop._ACTIVE_PARALLEL_THRESHOLDS), {'safety_factor'})
        table_km = _eprop.DEFAULT_PARALLEL_THRESHOLDS['small']['min_km_alpha']
        self.assertFalse(_eprop._should_parallelize_alpha(n, table_km - 1, 1, num_workers=2))
        self.assertTrue(_eprop._should_parallelize_alpha(n, table_km, 1, num_workers=2))

    def test_alpha_calibration_with_single_worker_does_not_record_ipc_latency(self):
        self.addCleanup(_eprop.configure_errorgen_propagation_parallelization, reset_to_defaults=True)
        with mock.patch.object(_eprop.parallelutils, '_measure_alpha_throughput_us', return_value=1.0):
            _eprop.optimize_errorgen_propagation_parallelization(num_qubits=4, num_workers=1)
        self.assertTrue(_eprop._ACTIVE_PARALLEL_THRESHOLDS['dynamic_calibrated'])
        self.assertNotIn('t_ipc_s', _eprop._ACTIVE_PARALLEL_THRESHOLDS)
        # a later multi-worker decision measures the latency instead of using a latency of zero (always parallel)
        with mock.patch.object(_eprop.parallelutils, '_measure_ipc_latency', return_value=1e-3) as ipc:
            self.assertFalse(_eprop._should_parallelize_alpha(4, 1, 1, num_workers=2))
            ipc.assert_called_once()

    def test_estimate_magnus_peak_memory(self):
        # Order 1 < Order 2 < Order 3 for multi-layer circuits
        mem_o1 = _eprop.estimate_magnus_peak_memory(4, 10, 20, num_workers=2, magnus_order=1)
        mem_o2 = _eprop.estimate_magnus_peak_memory(4, 10, 20, num_workers=2, magnus_order=2)
        mem_o3 = _eprop.estimate_magnus_peak_memory(4, 10, 20, num_workers=2, magnus_order=3)
        self.assertGreater(mem_o2, mem_o1)
        self.assertGreater(mem_o3, mem_o2)

        # Sequential mode (workers=1) has lower memory footprint than parallel (workers=2)
        mem_seq = _eprop.estimate_magnus_peak_memory(4, 10, 20, num_workers=1, magnus_order=2)
        self.assertLess(mem_seq, mem_o2)

        # Invalid worker count raises ValueError
        with self.assertRaises(ValueError):
            _eprop.estimate_magnus_peak_memory(4, 10, 20, num_workers=0)

        # Invalid magnus_order raises ValueError
        with self.assertRaises(ValueError):
            _eprop.estimate_magnus_peak_memory(4, 10, 20, num_workers=2, magnus_order=4)

    def test_measure_lse_entry_size(self):
        s1 = _eprop.measure_lse_entry_size(1)
        s4 = _eprop.measure_lse_entry_size(4)
        s16 = _eprop.measure_lse_entry_size(16)
        s64 = _eprop.measure_lse_entry_size(64)
        self.assertGreater(s4, s1)
        self.assertGreater(s16, s4)
        self.assertGreater(s64, s16)
        # cached per qubit count
        self.assertGreaterEqual(_eprop.measure_lse_entry_size.cache_info().hits, 0)
        before = _eprop.measure_lse_entry_size.cache_info().hits
        _eprop.measure_lse_entry_size(64)
        self.assertEqual(_eprop.measure_lse_entry_size.cache_info().hits, before + 1)
        # accounts for the stim.PauliStrings and attribute dictionary of the label, not just the label object
        lbl = _LSE('C', [stim.PauliString('X' * 64), stim.PauliString('Z' * 64)])
        self.assertGreater(s64, sys.getsizeof(lbl.__dict__) + 2 * sys.getsizeof(lbl.basis_element_labels[0]))

    def test_distinct_errorgen_label_counts(self):
        # exact counts by enumeration: H/S labels are non-identity Paulis, C/A labels unordered pairs of distinct ones
        for n in (1, 2, 3):
            paulis = [''.join(p) for p in product('IXYZ', repeat=n)][1:]
            exact = 2 * len(paulis) + 2 * (len(paulis) * (len(paulis) - 1) // 2)
            self.assertEqual(_eprop._num_distinct_errorgen_labels(n), exact)
        # support-limited count: exact, by enumeration, for every support limit (the C/A pairs are the
        # unordered pairs of distinct non-identity Paulis, in the sorted order the labels store them in)
        def support(pauli):
            return {q for q, c in enumerate(pauli) if c != 'I'}
        for n in (1, 2, 3):
            paulis = [''.join(p) for p in product('IXYZ', repeat=n)][1:]
            for max_support in range(n + 1):
                exact = 2 * sum(len(support(p)) <= max_support for p in paulis) \
                    + 2 * sum(len(support(p) | support(q)) <= max_support for p, q in combinations(paulis, 2))
                self.assertEqual(_eprop._num_labels_with_support_at_most(n, max_support), exact, (n, max_support))
        self.assertEqual(_eprop._num_labels_with_support_at_most(3, 3), _eprop._num_distinct_errorgen_labels(3))
        self.assertEqual(_eprop._num_labels_with_support_at_most(3, 1), 3 * (15 - 3))
        self.assertEqual(_eprop._num_labels_with_support_at_most(3, 7), _eprop._num_distinct_errorgen_labels(3))
        self.assertEqual(_eprop._num_labels_with_support_at_most(10, 10), _eprop._num_distinct_errorgen_labels(10))
        self.assertEqual(_eprop._num_labels_with_support_at_most(5, 0), 0)
        vals = [_eprop._num_labels_with_support_at_most(10, s) for s in range(11)]
        self.assertEqual(vals, sorted(vals))
        # the worst-case Magnus estimate no longer caps the term count at the number of H labels (4^n - 1)
        entry = _eprop.measure_lse_entry_size(2)
        self.assertGreaterEqual(_eprop.estimate_magnus_peak_memory(2, 30, 50, num_workers=1, magnus_order=2),
                                _eprop._num_distinct_errorgen_labels(2) * entry)

    def test_estimate_taylor_peak_memory(self):
        mem_o1 = _eprop.estimate_taylor_peak_memory(4, 20, order=1, num_workers=2)
        mem_o2 = _eprop.estimate_taylor_peak_memory(4, 20, order=2, num_workers=2)
        mem_o3 = _eprop.estimate_taylor_peak_memory(4, 20, order=3, num_workers=2)
        self.assertGreater(mem_o2, mem_o1)
        self.assertGreater(mem_o3, mem_o2)

        with self.assertRaises(ValueError):
            _eprop.estimate_taylor_peak_memory(4, 20, order=0)
        with self.assertRaises(ValueError):
            _eprop.estimate_taylor_peak_memory(4, 20, order=1, num_workers=0)

    def test_estimate_magnus_peak_memory_from_layers(self):
        # Upper-bounds the dictionaries actually built, and is far tighter than the worst-case scalar estimate.
        layers = ErrorGeneratorPropagator(self.error_model.copy()).propagate_errorgens(self.circuit)
        n = 4
        entry = _eprop.measure_lse_entry_size(n)
        identity = 'I' * n
        second = _eprop._second_order_magnus_term(layers, identity)
        second3, third3 = _eprop._second_and_third_order_magnus_terms(layers, identity)
        first = sum(len(l) for l in layers)
        est2 = _eprop._estimate_magnus_peak_memory_from_layers(layers, n, num_workers=1, magnus_order=2)
        est3 = _eprop._estimate_magnus_peak_memory_from_layers(layers, n, num_workers=1, magnus_order=3)
        self.assertGreaterEqual(est2, (first + len(second)) * entry)
        self.assertGreaterEqual(est3, (first + len(second3) + len(third3)) * entry)
        self.assertGreaterEqual(est3, est2)
        # workers add their baseline footprint and buffers
        self.assertGreater(_eprop._estimate_magnus_peak_memory_from_layers(layers, n, num_workers=4, magnus_order=2), est2)
        # nothing to estimate for first order or a single layer
        self.assertEqual(_eprop._estimate_magnus_peak_memory_from_layers(layers, n, magnus_order=1), 0)

    def test_should_parallelize_magnus_qubit_validation(self):
        # n_qubits must be provided and >= 1
        with self.assertRaises(ValueError):
            _eprop._should_parallelize_magnus(4, 20, num_workers=2, n_qubits=None)
        with self.assertRaises(ValueError):
            _eprop._should_parallelize_magnus(4, 20, num_workers=2, n_qubits=0)
        with self.assertRaises(ValueError):
            _eprop._should_parallelize_magnus(4, 20, num_workers=2, n_qubits=-2)

        # Valid n_qubits returns bool
        res = _eprop._should_parallelize_magnus(4, 20, num_workers=2, n_qubits=4)
        self.assertIsInstance(res, bool)

    def test_approximate_stabilizer_probabilities_parallel(self):
        seq = _eprop.approximate_stabilizer_probabilities(self.propagated_errorgen_layer, self.circuit, order=1, num_workers=None)
        with parallel_path_exercised(self, ['alpha'], ['_grid_task_worker']):
            par = _eprop.approximate_stabilizer_probabilities(self.propagated_errorgen_layer, self.circuit, order=1,
                                                              num_workers=2)
        self.assertArraysAlmostEqual(par, seq)
        self.assertAlmostEqual(np.sum(par), 1.0)

    def test_bulk_alpha_single_outcome_column_splitting(self):
        tableau = self.circuit_tableau
        errorgens = list(self.propagated_errorgen_layer.keys())
        bitstrings = ['0000']  # M = 1 forces 1D column (generator) splitting
        seq = _eprop.bulk_alpha(errorgens, tableau, bitstrings, num_workers=None)
        with parallel_path_exercised(self, ['alpha'], ['_grid_task_worker']):
            par = _eprop.bulk_alpha(errorgens, tableau, bitstrings, num_workers=2)
        self.assertEqual(par.shape, (1, len(errorgens)))
        self.assertArraysAlmostEqual(par, seq)

    def test_should_parallelize_commuting_product(self):
        # Validation
        with self.assertRaises(ValueError):
            _eprop._should_parallelize_commuting_product(None, 10, 10)
        with self.assertRaises(ValueError):
            _eprop._should_parallelize_commuting_product(0, 10, 10)
        with self.assertRaises(ValueError):
            _eprop._should_parallelize_commuting_product(4, 10, 10, num_workers=0)

        # 1 worker -> False
        self.assertFalse(_eprop._should_parallelize_commuting_product(4, 100, 100, num_workers=1))

        # Under default threshold (1000 for small)
        self.assertFalse(_eprop._should_parallelize_commuting_product(4, 10, 10, num_workers=2))
        # Over threshold
        self.assertTrue(_eprop._should_parallelize_commuting_product(4, 40, 40, num_workers=2))

        # Configuration overrides
        try:
            _eprop.configure_errorgen_propagation_parallelization(min_terms_commuting_product=5000)
            self.assertFalse(_eprop._should_parallelize_commuting_product(4, 40, 40, num_workers=2))

            _eprop.configure_errorgen_propagation_parallelization(force_serial=["commuting_product"])
            self.assertFalse(_eprop._should_parallelize_commuting_product(4, 100, 100, num_workers=2))

            _eprop.configure_errorgen_propagation_parallelization(force_serial=False, force_parallel=["commuting_product"])
            self.assertTrue(_eprop._should_parallelize_commuting_product(4, 2, 2, num_workers=2))
        finally:
            _eprop.configure_errorgen_propagation_parallelization(reset_to_defaults=True)

    def test_generate_balanced_magnus_pairs(self):
        # Empty or single layer -> empty tasks
        self.assertEqual(_eprop._generate_balanced_magnus_pairs([], 4), ([], []))
        self.assertEqual(_eprop._generate_balanced_magnus_pairs([{}], 4), ([], []))

        # Check order-preserving splitting
        l0 = {f"k{i}": 0.1 for i in range(100)}
        l1 = {f"k{i}": 0.1 for i in range(2000)}

        # With 16 workers, l1*l0 = 200,000, if alpha is small, it splits l1
        tasks, weights = _eprop._generate_balanced_magnus_pairs([l0, l1], num_workers=16, alpha=0.3)
        self.assertGreater(len(tasks), 1)
        # Check that operand order is strictly preserved: l1 is layer 1 (first arg), l0 is layer 0 (second arg)
        for t in tasks:
            sub_l1, intact_l0 = t
            self.assertIs(intact_l0, l0)
            self.assertTrue(set(sub_l1.keys()).issubset(set(l1.keys())))

        # If earlier layer l0 is larger than later layer l1:
        tasks_rev, _ = _eprop._generate_balanced_magnus_pairs([l1, l0], num_workers=16, alpha=0.3)
        self.assertGreater(len(tasks_rev), 1)
        for t in tasks_rev:
            intact_l0, sub_l1 = t
            self.assertIs(intact_l0, l0)
            self.assertTrue(set(sub_l1.keys()).issubset(set(l1.keys())))

    def test_error_generator_taylor_expansion_parallel(self):
        prod_seq = _eprop._commuting_product(self.propagated_errorgen_layer, self.propagated_errorgen_layer, 'IIII', num_workers=1)
        with parallel_path_exercised(self, ['commuting_product'], ['_CommutingProductSliceWorker']):
            prod_par = _eprop._commuting_product(self.propagated_errorgen_layer, self.propagated_errorgen_layer, 'IIII', num_workers=2)
        for k in set(prod_seq.keys()) | set(prod_par.keys()):
            self.assertAlmostEqual(prod_seq.get(k, 0), prod_par.get(k, 0), places=12)

        t_seq = _eprop.error_generator_taylor_expansion(self.propagated_errorgen_layer, order=2, num_workers=1)
        with parallel_path_exercised(self, ['commuting_product'], ['_CommutingProductSliceWorker']):
            t_par = _eprop.error_generator_taylor_expansion(self.propagated_errorgen_layer, order=2, num_workers=2)
        self.assertEqual(len(t_seq), len(t_par))
        for s_dict, p_dict in zip(t_seq, t_par):
            # (serial and parallel truncation of the powers agree to within about the threshold)
            for k in set(s_dict) | set(p_dict):
                self.assertAlmostEqual(s_dict.get(k, 0), p_dict.get(k, 0), places=12)

    def test_commuting_product_parallel_truncation_is_consistent_with_serial(self):
        # Both evaluations truncate at the threshold. They are not exactly equal: workers drop partial sums
        # below threshold / n_chunks, so merged values can differ from the serial ones by up to ~threshold,
        # and labels whose value is within that band of the threshold may be kept by only one evaluation.
        thr = 1e-14
        layer = self.propagated_errorgen_layer
        self.addCleanup(_eprop.configure_errorgen_propagation_parallelization, reset_to_defaults=True)
        _eprop.configure_errorgen_propagation_parallelization(force_parallel=['commuting_product'])
        raw = _eprop._raw_commuting_product(layer, layer, 'IIII')
        self.assertTrue(any(abs(v) <= thr for v in raw.values()))  # the fixture produces terms to truncate
        for workers in (2, 3):
            seq = _eprop._commuting_product(layer, layer, 'IIII', truncation_threshold=thr, num_workers=1)
            with _PoolDispatchSpy() as spy:
                par = _eprop._commuting_product(layer, layer, 'IIII', truncation_threshold=thr, num_workers=workers)
            self.assertIn('_CommutingProductSliceWorker', spy.dispatched)  # the parallel branch ran
            for result in (seq, par):
                self.assertTrue(all(abs(v) > thr for v in result.values()))  # truncated at the threshold
            # serial truncation is exactly that of the raw product
            self.assertEqual(set(seq), {k for k, v in raw.items() if abs(v) > thr})
            for k in set(seq) & set(par):
                self.assertLessEqual(abs(seq[k] - par[k]), thr + 1e-15 * abs(seq[k])) # allow small roundoff error from FP
            for k in set(seq) ^ set(par):  # a label kept by only one evaluation must be near the threshold
                self.assertLessEqual(abs(raw[k]), 2 * thr)

        # a product whose terms are all below the threshold is empty in both evaluations
        tiny = {lbl: 1e-9 * rate / abs(rate) for lbl, rate in layer.items()}  # products ~1e-18
        self.assertEqual(_eprop._commuting_product(tiny, tiny, 'IIII', num_workers=1), {})
        self.assertEqual(_eprop._commuting_product(tiny, tiny, 'IIII', num_workers=2), {})

        # a label whose total is above the threshold is never lost entirely: with every chunk's entry
        # at most threshold / n_chunks, nothing is sent, so the worker threshold is what bounds this
        worker = _eprop._CommutingProductSliceWorker(layer, 'IIII', 0.0)
        everything = worker(layer)
        thr_worker = _eprop._CommutingProductSliceWorker(layer, 'IIII', thr / 3)
        kept = thr_worker(layer)
        self.assertEqual(set(kept), {k for k, v in everything.items() if abs(v) > thr / 3})
        self.assertLess(len(kept), len(everything))

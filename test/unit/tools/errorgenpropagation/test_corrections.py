from unittest import mock

import numpy as np
import stim

from pygsti.algorithms.randomcircuit import create_random_circuit
from pygsti.baseobjs.errorgenlabel import LocalElementaryErrorgenLabel as LEEL
from pygsti.errorgenpropagation.errorpropagator import ErrorGeneratorPropagator
from pygsti.models.modelconstruction import create_crosstalk_free_model
from pygsti.processors import QubitProcessorSpec
from pygsti.tools import errorgenpropagation as _eprop

from ...util import BaseCase
from .errgenprop_test_utils import (
    _all_sector_rate_subset,
    _mock_taylor_expansion,
    parallel_path_exercised,
    probabilities_errorgen_prop,
)


class ErrgenCorrectionsTester(BaseCase):

    def setUp(self):
        num_qubits = 4
        gate_names = ['Gcphase', 'Gxpi2', 'Gypi2']
        availability = {'Gcphase': [(0, 1), (1, 2), (2, 3), (3, 0)]}
        pspec = QubitProcessorSpec(num_qubits, gate_names, availability=availability)
        self.target_model = create_crosstalk_free_model(processor_spec=pspec)
        self.circuit = create_random_circuit(pspec, 4, sampler='edgegrab', samplerargs=[0.4], rand_state=12345)
        self.circuit_alt = create_random_circuit(pspec, 4, sampler='edgegrab', samplerargs=[0.4], rand_state=12345)
        rates = {
            'Gxpi2': _all_sector_rate_subset(1, 1, seed=100),
            'Gypi2': _all_sector_rate_subset(1, 1, seed=101),
            'Gcphase': _all_sector_rate_subset(2, 1, seed=200),
        }
        self.error_model = create_crosstalk_free_model(pspec, lindblad_error_coeffs=rates, lindblad_parameterization='GLND')
        self.error_propagator = ErrorGeneratorPropagator(self.error_model.copy())
        self.propagated_errorgen_layer = self.error_propagator.propagate_errorgens_bch(self.circuit, bch_order=1)
        self.circuit_tableau = self.circuit.convert_to_stim_tableau()
        self.circuit_tableau_alt = self.circuit_alt.convert_to_stim_tableau()

    def _leel_dict_with_out_of_order_pairs(self):
        # LocalElementaryErrorgenLabel-keyed rates whose C/A basis element label pairs are not in
        # the canonical (string-sorted) order used by LocalStimErrorgenLabel.
        return {LEEL('H', ('XIII',)): 2e-3, LEEL('S', ('ZYII',)): 1e-3,
                LEEL('A', ('ZIII', 'XIII')): 1e-3, LEEL('C', ('ZXII', 'XYII')): 5e-4}

    #This unit test for tableau fidelity is straight out of Craig Gidney's stackexchange post.
    def test_tableau_fidelity(self):
        def _assert_correct_tableau_fidelity(u, v):
            expected = abs(np.dot(u, np.conj(v)))**2
            ut = stim.Tableau.from_state_vector(u, endian='little')
            vt = stim.Tableau.from_state_vector(v, endian='little')
            actual = _eprop.tableau_fidelity(ut, vt)
            np.testing.assert_allclose(actual, expected, atol=1e-14, rtol=1e-5)

        s = 0.5**0.5
        _assert_correct_tableau_fidelity([1, 0], [0, 1])
        _assert_correct_tableau_fidelity([1, 0], [1, 0])
        _assert_correct_tableau_fidelity([0, 1], [1, 0])
        _assert_correct_tableau_fidelity([s, s], [s, s])
        _assert_correct_tableau_fidelity([s, s], [s, -s])
        _assert_correct_tableau_fidelity([s, -s], [s, s])
        _assert_correct_tableau_fidelity([s, 1j * s], [s, s])
        _assert_correct_tableau_fidelity([s, s], [s, s])
        _assert_correct_tableau_fidelity([1, 0], [s, s])
        _assert_correct_tableau_fidelity([0, 1], [s, s])
        _assert_correct_tableau_fidelity([1, 0, 0, 0], [0, 0, s, s])
        _assert_correct_tableau_fidelity([0, 0, 1, 0], [0, 0, s, s])
        _assert_correct_tableau_fidelity([0, 0, 1, 0], [0, 0, 1j * s, s])
        for n in range(6):
            for _ in range(10):
                _assert_correct_tableau_fidelity(
                    stim.Tableau.random(n).to_state_vector(),
                    stim.Tableau.random(n).to_state_vector(),
                )

    def test_bitstring_to_tableau(self):
        tableau = _eprop.bitstring_to_tableau('1010')
        self.assertEqual(tableau, stim.PauliString('XIXI').to_tableau())

    def test_approximate_stabilizer_probabilities_bitstrings(self):
        # Full distribution
        with parallel_path_exercised(self, ['alpha', 'commuting_product'], ['_grid_task_worker', '_CommutingProductSliceWorker']):
            probs_full = _eprop.approximate_stabilizer_probabilities(
                self.propagated_errorgen_layer, self.circuit, order=2, num_workers=2
            )
        # Subset of bitstrings (forces subset path in bulk_alpha)
        subset_bitstrings = ['0000', '0101', '1111']
        with parallel_path_exercised(self, ['alpha', 'commuting_product'], ['_grid_task_worker', '_CommutingProductSliceWorker']):
            probs_subset = _eprop.approximate_stabilizer_probabilities(
                self.propagated_errorgen_layer, self.circuit, bitstrings=subset_bitstrings, order=2, num_workers=2
            )
        # Compute subset probabilities individually for comparison
        probs_individual = [
            _eprop.approximate_stabilizer_probability(self.propagated_errorgen_layer, self.circuit, bs, order=2)
            for bs in subset_bitstrings
        ]
        self.assertArraysAlmostEqual(probs_subset, probs_individual)

    def test_tableau_fidelities_and_stabilizer_probabilities(self):
        tableau = self.circuit_tableau
        
        # Empty bitstrings returns empty array
        self.assertEqual(len(_eprop.stabilizer_probabilities(tableau, [])), 0)

        # Mismatched candidate qubit count raises ValueError
        with self.assertRaises(ValueError):
            bad_cand = stim.Tableau(len(tableau) + 1)
            _eprop.tableau_fidelities(tableau, [bad_cand])

    def test_approximate_stabilizer_probabilities_positional_and_underflow(self):
        # Positional bitstrings argument: (errorgen_dict, circuit, bitstrings, order)
        subset = ['0000', '0101']
        probs_pos = _eprop.approximate_stabilizer_probabilities(
            self.propagated_errorgen_layer, self.circuit, subset, 2
        )
        probs_kw = _eprop.approximate_stabilizer_probabilities(
            self.propagated_errorgen_layer, self.circuit, bitstrings=subset, order=2
        )
        self.assertArraysAlmostEqual(probs_pos, probs_kw)

        # Underflow check: mock random_support returning > 2148
        with mock.patch.object(_eprop.sensitivities, "random_support", return_value=2200):
            with self.assertRaises(RuntimeError):
                _eprop.approximate_stabilizer_probabilities(
                    self.propagated_errorgen_layer, self.circuit, subset, 1
                )

    def test_approximate_stabilizer_pauli_expectations_bulk(self):
        paulis = [stim.PauliString('XXXX'), stim.PauliString('ZIII'), stim.PauliString('YYYY')]
        with parallel_path_exercised(self, ['alpha', 'commuting_product'], ['_grid_task_worker', '_CommutingProductSliceWorker']):
            bulk_vals = _eprop.approximate_stabilizer_pauli_expectations(
                self.propagated_errorgen_layer, self.circuit, paulis, order=2, num_workers=2
            )
        indiv_vals = [
            _eprop.approximate_stabilizer_pauli_expectation(self.propagated_errorgen_layer, self.circuit, p, order=2)
            for p in paulis
        ]
        self.assertArraysAlmostEqual(bulk_vals, indiv_vals)

    def test_approximate_stabilizer_probabilities_leel_keys(self):
        leel_dict = self._leel_dict_with_out_of_order_pairs()
        bitstrings = ['0000', '0101', '1111', '1010']
        for order in (1, 2):
            bulk = _eprop.approximate_stabilizer_probabilities(leel_dict, self.circuit, bitstrings, order=order)
            for b, p in zip(bitstrings, bulk):
                single = _eprop.approximate_stabilizer_probability(leel_dict, self.circuit, b, order=order)
                self.assertAlmostEqual(p, single, places=14)

    def test_approximate_stabilizer_pauli_expectations_leel_keys(self):
        leel_dict = self._leel_dict_with_out_of_order_pairs()
        paulis = ['ZZZZ', 'XIII', 'YYII', 'IZIZ']
        for order in (1, 2):
            bulk = _eprop.approximate_stabilizer_pauli_expectations(leel_dict, self.circuit, paulis, order=order)
            for p, e in zip(paulis, bulk):
                single = _eprop.approximate_stabilizer_pauli_expectation(leel_dict, self.circuit, p, order=order)
                self.assertAlmostEqual(e, single, places=14)

    def test_correction_functions_forward_parallel_args_to_taylor_expansion(self):
        tableau = self.circuit_tableau
        with _mock_taylor_expansion() as taylor_mock:
            _eprop.stabilizer_probability_correction(self.propagated_errorgen_layer, tableau, '0000', order=2,
                                                     num_workers=3)
            self.assertEqual(taylor_mock.call_args.kwargs.get('num_workers'), 3)
        with _mock_taylor_expansion() as taylor_mock:
            _eprop.stabilizer_pauli_expectation_correction(self.propagated_errorgen_layer, tableau,
                                                           stim.PauliString('ZIII'), order=2, num_workers=3)
            self.assertEqual(taylor_mock.call_args.kwargs.get('num_workers'), 3)

    def test_stabilizer_probability_correction(self):
        #The corrections testing here will just be integration testing, we'll
        #check for correctness with the probability functions instead.
        bitstrings = ['0000', '1000']
        orders = [1,2,3]
        for bitstring in bitstrings:
            for order in orders:
                _eprop.stabilizer_probability_correction(self.propagated_errorgen_layer, self.circuit_tableau, bitstring, order)

    def test_stabilizer_pauli_expectation_correction(self):
        #The corrections testing here will just be integration testing, we'll
        #check for correctness with the full expecation functions instead.
        paulis = [stim.PauliString('XXXX'), stim.PauliString('ZIII')]
        orders = [1,2,3]
        for pauli in paulis:
            for order in orders:
                _eprop.stabilizer_pauli_expectation_correction(self.propagated_errorgen_layer, self.circuit_tableau, pauli, order)

    def test_approximate_stabilizer_probability(self):
        exact_prop_probs = probabilities_errorgen_prop(self.error_propagator, self.target_model, 
                                                       self.circuit, use_bch=True, bch_order=1)
        first_order_diff = exact_prop_probs[1] - _eprop.approximate_stabilizer_probability(self.propagated_errorgen_layer, self.circuit_tableau, '0001')
        second_order_diff = exact_prop_probs[1] - _eprop.approximate_stabilizer_probability(self.propagated_errorgen_layer, self.circuit_tableau, '0001', order=2)
        third_order_diff = exact_prop_probs[1] - _eprop.approximate_stabilizer_probability(self.propagated_errorgen_layer, self.circuit_tableau, '0001', order=3)

        assert abs(first_order_diff) > abs(second_order_diff)
        assert abs(second_order_diff) > abs(third_order_diff)

        first_order_diff = exact_prop_probs[-1] - _eprop.approximate_stabilizer_probability(self.propagated_errorgen_layer, self.circuit_tableau, '1111')
        second_order_diff = exact_prop_probs[-1] - _eprop.approximate_stabilizer_probability(self.propagated_errorgen_layer, self.circuit_tableau, '1111', order=2)

        assert abs(first_order_diff) > abs(second_order_diff)

    def test_approximate_stabilizer_probabilities(self):
        exact_prop_probs = probabilities_errorgen_prop(self.error_propagator, self.target_model, 
                                                       self.circuit, use_bch=True, bch_order=1)
        approx_stab_prob_vec_order_1 = _eprop.approximate_stabilizer_probabilities(self.propagated_errorgen_layer, self.circuit_tableau)
        approx_stab_prob_vec_order_2 = _eprop.approximate_stabilizer_probabilities(self.propagated_errorgen_layer, self.circuit_tableau, order=2)
        
        tvd_order_1 = np.linalg.norm(exact_prop_probs-approx_stab_prob_vec_order_1, ord=1)
        tvd_order_2 = np.linalg.norm(exact_prop_probs-approx_stab_prob_vec_order_2, ord=1)

        assert tvd_order_1 > tvd_order_2
        
        exact_prop_probs = probabilities_errorgen_prop(self.error_propagator, self.target_model, 
                                                       self.circuit_alt, use_bch=True, bch_order=1)
        approx_stab_prob_vec_order_1 = _eprop.approximate_stabilizer_probabilities(self.propagated_errorgen_layer, self.circuit_tableau_alt)
        approx_stab_prob_vec_order_2 = _eprop.approximate_stabilizer_probabilities(self.propagated_errorgen_layer, self.circuit_tableau_alt, order=2)
        
        tvd_order_1 = np.linalg.norm(exact_prop_probs-approx_stab_prob_vec_order_1, ord=1)
        tvd_order_2 = np.linalg.norm(exact_prop_probs-approx_stab_prob_vec_order_2, ord=1)

        assert tvd_order_1 > tvd_order_2

    def test_approximate_stabilizer_pauli_expectation(self):
        rng = np.random.default_rng(seed=12345)
        paulis_4Q = list(stim.PauliString.iter_all(4))
        random_4Q_pauli_indices = rng.choice(len(paulis_4Q), 3, replace=False)
        random_4Q_paulis = [paulis_4Q[idx] for idx in random_4Q_pauli_indices]

        for pauli in random_4Q_paulis:
            first_order_diff  = _eprop.approximate_stabilizer_pauli_expectation_numerical(self.propagated_errorgen_layer, self.error_propagator, self.circuit, pauli, order=1) -\
                                _eprop.approximate_stabilizer_pauli_expectation(self.propagated_errorgen_layer, self.circuit_tableau, pauli, order=1)
            second_order_diff = _eprop.approximate_stabilizer_pauli_expectation_numerical(self.propagated_errorgen_layer, self.error_propagator, self.circuit, pauli, order=2) -\
                                _eprop.approximate_stabilizer_pauli_expectation(self.propagated_errorgen_layer, self.circuit_tableau, pauli, order=2)
            third_order_diff  = _eprop.approximate_stabilizer_pauli_expectation_numerical(self.propagated_errorgen_layer, self.error_propagator, self.circuit, pauli, order=3) -\
                                _eprop.approximate_stabilizer_pauli_expectation(self.propagated_errorgen_layer, self.circuit_tableau, pauli, order=3)

            assert abs(first_order_diff)  < 1e-6, f'{pauli=}'
            assert abs(second_order_diff) < 1e-8, f'{pauli=}'
            assert abs(third_order_diff)  < 5e-8, f'{pauli=}'

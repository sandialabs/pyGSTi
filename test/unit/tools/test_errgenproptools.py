import ast
import re
import sys
import numpy as np
from scipy.linalg import logm, expm
from pygsti.baseobjs import Label, QubitSpace, BuiltinBasis
from pygsti.baseobjs.errorgenbasis import CompleteElementaryErrorgenBasis
from pygsti.algorithms.randomcircuit import create_random_circuit
from pygsti.models.modelconstruction import create_crosstalk_free_model
from pygsti.baseobjs.errorgenlabel import LocalElementaryErrorgenLabel as LEEL
from pygsti.baseobjs.errorgenlabel import GlobalElementaryErrorgenLabel as GEEL
from pygsti.errorgenpropagation.localstimerrorgen import LocalStimErrorgenLabel as _LSE
from pygsti.tools import errgenproptools as _eprop
from pygsti.tools.matrixtools import print_mx
from pygsti.tools.basistools import change_basis
from ..util import BaseCase
from itertools import product, chain
import random
try:
    import stim
except ImportError:
    stim = None
import unittest
from pygsti.processors import QubitProcessorSpec
from pygsti.errorgenpropagation.errorpropagator import ErrorGeneratorPropagator

#TODO: errorgen_layer_to_matrix, stim_pauli_string_less_than 

@unittest.skipIf(stim is None, "stim is not installed")
class ErrgenCompositionCommutationTester(BaseCase):

    def setUp(self):
        num_qubits = 4
        gate_names = ['Gcphase', 'Gxpi2', 'Gypi2']
        availability = {'Gcphase':[(0,1), (1,2), (2,3), (3,0)]}
        pspec = QubitProcessorSpec(num_qubits, gate_names, availability=availability)
        self.target_model = create_crosstalk_free_model(processor_spec = pspec)
        self.circuit = create_random_circuit(pspec, 4, sampler='edgegrab', samplerargs=[0.4,], rand_state=12345)
        max_strengths = {1: {'S': 0, 'H': 0.0001},
                         2: {'S': 0, 'H': 0.0001}}
        error_rates_dict = sample_error_rates_dict(pspec, max_strengths, seed=12345)
        self.error_model = create_crosstalk_free_model(pspec, lindblad_error_coeffs=error_rates_dict)
        self.errorgen_propagator = ErrorGeneratorPropagator(self.error_model.copy())
        self.propagated_errorgen_layers = self.errorgen_propagator.propagate_errorgens(self.circuit)

    def test_errorgen_commutators(self):
        #confirm we get the correct analytic commutators by comparing to numerics.

        #create an error generator basis.
        errorgen_basis = CompleteElementaryErrorgenBasis('PP', QubitSpace(2), default_label_type='local')

        #use this basis to construct a dictionary from error generator labels to their
        #matrices.
        errorgen_lbls = errorgen_basis.labels
        errorgen_lbl_matrix_dict = {lbl: mat for lbl, mat in zip(errorgen_lbls, errorgen_basis.elemgen_matrices)}

        #loop through all of the pairs of indices.
        errorgen_label_pairs = list(product(errorgen_lbls, repeat=2))

        #also get a version of this list where the labels are local stim ones
        local_stim_errorgen_lbls = [_LSE.cast(lbl) for lbl in errorgen_lbls]
        stim_errorgen_label_pairs = list(product(local_stim_errorgen_lbls, repeat=2))

        #for each pair compute the commutator directly and compute it analytically (then converting it to
        #a numeric array) and see how they compare.
        for pair1, pair2 in zip(errorgen_label_pairs, stim_errorgen_label_pairs):
            numeric_commutator = _eprop.error_generator_commutator_numerical(pair1[0], pair1[1], errorgen_lbl_matrix_dict)
            analytic_commutator = _eprop.error_generator_commutator(pair2[0], pair2[1])
            analytic_commutator_mat = _eprop.errorgen_layer_to_matrix(analytic_commutator, 2, errorgen_lbl_matrix_dict)        

            norm_diff = np.linalg.norm(numeric_commutator-analytic_commutator_mat)
            if norm_diff > 1e-10:
                print(f'Difference in commutators for pair {pair1} is greater than 1e-10.')
                print(f'{np.linalg.norm(numeric_commutator-analytic_commutator_mat)=}')
                print('numeric_commutator=')
                print_mx(numeric_commutator)
                
                #Decompose the numerical commutator into rates.
                for lbl, dual in zip(errorgen_lbls, errorgen_basis.elemgen_dual_matrices):
                    rate = np.trace(dual.conj().T@numeric_commutator)
                    if abs(rate) >1e-3:
                        print(f'{lbl}: {rate}')
                
                print(f'{analytic_commutator=}')
                print('analytic_commutator_mat=')
                print_mx(analytic_commutator_mat)
                raise ValueError()
                
    def test_errorgen_composition(self):
        
        #create an error generator basis.
        complete_errorgen_basis_2Q = CompleteElementaryErrorgenBasis('PP', QubitSpace(2), default_label_type='local')
        complete_errorgen_basis_3Q = CompleteElementaryErrorgenBasis('PP', QubitSpace(3), default_label_type='local')
        
        #use this basis to construct a dictionary from error generator labels to their
        #matrices.
        errorgen_lbls_2Q = complete_errorgen_basis_2Q.labels
        errorgen_lbl_matrix_dict_2Q = {lbl: mat for lbl, mat in zip(errorgen_lbls_2Q, complete_errorgen_basis_2Q.elemgen_matrices)}
        
        #augment testing with random selection of 3Q labels (some commutation relations for C and A terms require a minimum of 3 qubits).
        errorgen_lbls_3Q, errorgen_mats_3Q = select_random_items_from_multiple_lists(
            [complete_errorgen_basis_3Q.labels, complete_errorgen_basis_3Q.elemgen_matrices], 50, seed=12345
        )
        errorgen_lbl_matrix_dict_3Q = {lbl: mat for lbl, mat in zip(errorgen_lbls_3Q, errorgen_mats_3Q)}
            
        complete_errorgen_lbl_matrix_dict_3Q = {lbl: mat for lbl, mat in zip(complete_errorgen_basis_3Q.labels, complete_errorgen_basis_3Q.elemgen_matrices)}

        #loop through all of the pairs of indices.
        errorgen_label_pairs_2Q = list(product(errorgen_lbls_2Q, repeat=2))
        errorgen_label_pairs_3Q = list(product(errorgen_lbls_3Q, repeat=2))
        
        #also get a version of this list where the labels are local stim ones
        local_stim_errorgen_lbls_2Q = [_LSE.cast(lbl) for lbl in errorgen_lbls_2Q]
        local_stim_errorgen_lbls_3Q = [_LSE.cast(lbl) for lbl in errorgen_lbls_3Q]
        
        stim_errorgen_label_pairs_2Q = list(product(local_stim_errorgen_lbls_2Q, repeat=2))
        stim_errorgen_label_pairs_3Q = list(product(local_stim_errorgen_lbls_3Q, repeat=2))
                
        #for each pair compute the composition directly and compute it analytically (then converting it to
        #a numeric array) and see how they compare.
        for pair1, pair2 in zip(errorgen_label_pairs_2Q, stim_errorgen_label_pairs_2Q):
            numeric_composition = _eprop.error_generator_composition_numerical(pair1[0], pair1[1], errorgen_lbl_matrix_dict_2Q)
            analytic_composition = _eprop.error_generator_composition(pair2[0], pair2[1])
            analytic_composition_mat = _eprop.errorgen_layer_to_matrix(analytic_composition, 2, errorgen_matrix_dict = errorgen_lbl_matrix_dict_2Q)        
            norm_diff = np.linalg.norm(numeric_composition-analytic_composition_mat)
            self.assertLess(norm_diff, 1e-10, f"Numeric and analytic error generator compositions differed for pair {pair1}: {norm_diff}")

        for pair1, pair2 in zip(errorgen_label_pairs_3Q, stim_errorgen_label_pairs_3Q):
            numeric_composition = _eprop.error_generator_composition_numerical(pair1[0], pair1[1], errorgen_lbl_matrix_dict_3Q)
            analytic_composition = _eprop.error_generator_composition(pair2[0], pair2[1])
            analytic_composition_mat = _eprop.errorgen_layer_to_matrix(analytic_composition, 3, errorgen_matrix_dict = complete_errorgen_lbl_matrix_dict_3Q)        
            norm_diff = np.linalg.norm(numeric_composition-analytic_composition_mat)
            self.assertLess(norm_diff, 1e-10, f"Numeric and analytic error generator compositions differed for pair {pair1}: {norm_diff}")    
    
    def test_iterative_error_generator_composition(self):
        test_labels = [(_LSE('H', [stim.PauliString('X')]), _LSE('H', [stim.PauliString('X')]), _LSE('H', [stim.PauliString('X')])), 
                       (_LSE('H', [stim.PauliString('IX')]), _LSE('H', [stim.PauliString('IX')]), _LSE('H', [stim.PauliString('XI')])),
                       (_LSE('S', [stim.PauliString('YY')]), _LSE('H', [stim.PauliString('IX')]), _LSE('H', [stim.PauliString('XI')]))]
        rates = [(1,1,1), (1,1,1), (1,1,1)]
    
        correct_iterative_compositions = [[(_LSE('H', (stim.PauliString("+X"),)), (-4-0j))],
                                          [(_LSE('H', (stim.PauliString("+X_"),)), (-2+0j)), (_LSE('A', (stim.PauliString("+_X"), stim.PauliString("+XX"))), (2+0j))],
                                          [(_LSE('C', (stim.PauliString("+YZ"), stim.PauliString("+ZY"))), (1+0j)), (_LSE('C', (stim.PauliString("+YY"), stim.PauliString("+ZZ"))), (1+0j)),
                                           (_LSE('C', (stim.PauliString("+_X"), stim.PauliString("+X_"))), -1)]                                          
                                        ]
        
        for lbls, rates, correct_lbls in zip(test_labels, rates, correct_iterative_compositions):
            iterated_composition = _eprop.iterative_error_generator_composition(lbls, rates)
            self.assertEqual(iterated_composition, correct_lbls)

        _compare_analytic_numeric_iterative_composition(2)
        

    def test_bch_approximation(self):
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
        

    def test_magnus_expansion(self):
        first_order_magnus_numerical = _eprop.magnus_numerical(self.propagated_errorgen_layers, self.errorgen_propagator, magnus_order=1)
        propagated_errorgen_layers_magnus_order_1 = self.errorgen_propagator.propagate_errorgens_bch(self.circuit, bch_order=1)
        first_order_magnus_analytical = self.errorgen_propagator.errorgen_layer_dict_to_errorgen(propagated_errorgen_layers_magnus_order_1,mx_basis='pp')
        assert np.linalg.norm(first_order_magnus_analytical-first_order_magnus_numerical) < 1e-14
        
        propagated_errorgen_layers_magnus_order_2 = self.errorgen_propagator.propagate_errorgens_bch(self.circuit, bch_order=2)
        second_order_magnus_numerical = _eprop.magnus_numerical(self.propagated_errorgen_layers, self.errorgen_propagator, magnus_order=2)
        second_order_magnus_analytical = self.errorgen_propagator.errorgen_layer_dict_to_errorgen(propagated_errorgen_layers_magnus_order_2, mx_basis='pp')
        assert np.linalg.norm(second_order_magnus_analytical-second_order_magnus_numerical) < 1e-14

        third_order_magnus_numerical = _eprop.magnus_numerical(self.propagated_errorgen_layers, self.errorgen_propagator, magnus_order=3)
        propagated_errorgen_layers_magnus_order_3 = self.errorgen_propagator.propagate_errorgens_bch(self.circuit, bch_order=3)
        third_order_magnus_analytical = self.errorgen_propagator.errorgen_layer_dict_to_errorgen(propagated_errorgen_layers_magnus_order_3, mx_basis='pp')
        assert np.linalg.norm(third_order_magnus_analytical-third_order_magnus_numerical) < 1e-14

        exact_errorgen = logm(self.errorgen_propagator.eoc_error_channel(self.circuit))
        exact_vs_first_order_norm  = np.linalg.norm(first_order_magnus_analytical-exact_errorgen)
        exact_vs_second_order_norm = np.linalg.norm(second_order_magnus_analytical-exact_errorgen)
        exact_vs_third_order_norm  = np.linalg.norm(third_order_magnus_analytical-exact_errorgen)
        
        self.assertTrue((exact_vs_first_order_norm > exact_vs_second_order_norm) and (exact_vs_second_order_norm > exact_vs_third_order_norm))

    def test_error_generator_pauli_action(self):
        egbasis_HS = CompleteElementaryErrorgenBasis('PP', QubitSpace(3), default_label_type='local', elementary_errorgen_types=('H','S'))
        egbasis_CA = CompleteElementaryErrorgenBasis('PP', QubitSpace(3), default_label_type='local', elementary_errorgen_types=('C','A'))
        rng = np.random.default_rng()
        paulis = np.fromiter(stim.PauliString.iter_all(3), dtype=object)
        random_paulis = rng.choice(paulis, size=10, replace=False)
        random_errorgens_HS = rng.choice(np.fromiter(egbasis_HS.labels, dtype=object), size=10, replace=False)
        random_errorgens_CA = rng.choice(np.fromiter(egbasis_CA.labels, dtype=object), size=10, replace=False)
        for pauli in random_paulis:
            for eglbl in chain(random_errorgens_HS, random_errorgens_CA):
                pauli_action = _eprop.errorgen_pauli_action(_LSE.cast(eglbl), pauli)
                pauli_action_dense = pauli_action[0]*pauli_action[1].to_unitary_matrix(endian='big') \
                                            if pauli_action is not None else np.zeros((2**len(pauli),2**len(pauli)))
                pauli_action_numerical = _eprop.errorgen_pauli_action_numerical(eglbl, pauli)
                assert np.linalg.norm(pauli_action_dense-pauli_action_numerical) < 1e-14, f'Numerical and analytical results differ, {eglbl=}, {pauli=}'

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

@unittest.skipIf(stim is None, "stim is not installed")
class ApproxStabilizerMethodTester(BaseCase):
    def setUp(self):
        num_qubits = 4
        gate_names = ['Gcphase', 'Gxpi2', 'Gypi2']
        availability = {'Gcphase':[(0,1), (1,2), (2,3), (3,0)]}
        pspec = QubitProcessorSpec(num_qubits, gate_names, availability=availability)
        self.target_model = create_crosstalk_free_model(processor_spec = pspec)
        self.circuit = create_random_circuit(pspec, 4, sampler='edgegrab', samplerargs=[0.4,], rand_state=12345)
        self.circuit_alt = create_random_circuit(pspec, 4, sampler='edgegrab', samplerargs=[0.4,], rand_state=12345)
        max_strengths = {1: {'S': 0.0005, 'H': 0.0001},
                         2: {'S': 0.0005, 'H': 0.0001}}
        error_rates_dict = sample_error_rates_dict(pspec, max_strengths, seed=12345)
        self.error_model = create_crosstalk_free_model(pspec, lindblad_error_coeffs=error_rates_dict)
        self.error_propagator = ErrorGeneratorPropagator(self.error_model.copy())
        self.propagated_errorgen_layer = self.error_propagator.propagate_errorgens_bch(self.circuit, bch_order=1)
        self.circuit_tableau = self.circuit.convert_to_stim_tableau()
        self.circuit_tableau_alt = self.circuit_alt.convert_to_stim_tableau()

        #also create a 3-qubit pspec for making some tests faster.
        num_qubits = 3
        gate_names = ['Gcphase', 'Gxpi2', 'Gypi2']
        availability = {'Gcphase':[(0,1), (1,2)]}
        pspec = QubitProcessorSpec(num_qubits, gate_names, availability=availability)
        self.target_model_3Q = create_crosstalk_free_model(processor_spec = pspec)
        self.circuit_3Q = create_random_circuit(pspec, 4, sampler='edgegrab', samplerargs=[0.4,], rand_state=12345)
        self.circuit_tableau_3Q = self.circuit_3Q.convert_to_stim_tableau()

    
    def test_random_support(self):
        num_random = _eprop.random_support(self.circuit_tableau)
        self.assertEqual(num_random, 3)

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
    
    def test_amplitude_of_state(self):
        amp0000 = _eprop.amplitude_of_state(self.circuit_tableau, '0000', False)
        amp1111 = _eprop.amplitude_of_state(self.circuit_tableau, '1111', False)
        self.assertTrue(abs(amp0000)<1e-7)
        self.assertTrue(abs(amp1111 -(-1j*np.sqrt(0.125)))<1e-7)
        
        amp0000 = _eprop.amplitude_of_state(self.circuit_tableau_alt, '0000', False)
        amp1111 = _eprop.amplitude_of_state(self.circuit_tableau_alt, '1111', False)
        
        self.assertTrue(abs(amp0000)<1e-7)
        self.assertTrue(abs(amp1111 - (-1j*np.sqrt(0.125)))<1e-7)

    def test_bitstring_to_tableau(self):
        tableau = _eprop.bitstring_to_tableau('1010')
        self.assertEqual(tableau, stim.PauliString('XIXI').to_tableau())

    def test_pauli_phase_update(self):
        test_paulis = ['YII', 'ZII', str(stim.PauliString('XYZ')), str(stim.PauliString('+iIII'))]
        test_bitstring = '100'

        correct_phase_updates_standard = [-1j, -1, 1j, 1j]
        correct_phase_updates_dual = [1j, -1, -1j, 1j]
        correct_output_bitstrings = ['000', '100', '010', '100']

        for i, test_pauli in enumerate(test_paulis):
            print(i)
            phase_update, output_bitstring = _eprop.pauli_phase_update(test_pauli, test_bitstring)
            self.assertEqual(phase_update, correct_phase_updates_standard[i])
            self.assertEqual(output_bitstring, correct_output_bitstrings[i])
            
        for i, test_pauli in enumerate(test_paulis):
            print(i)
            phase_update, output_bitstring = _eprop.pauli_phase_update(test_pauli, test_bitstring, dual=True)
            self.assertEqual(phase_update, correct_phase_updates_dual[i])
            self.assertEqual(output_bitstring, correct_output_bitstrings[i])

    def test_pauli_phase_update_all_zeros(self):
        test_paulis = ['YII', 'ZII', str(stim.PauliString('XYZ')), str(stim.PauliString('+iIII'))]

        correct_phase_updates_standard = [1j, 1, 1j, 1j]
        correct_phase_updates_dual = [-1j, 1, -1j, 1j]
        correct_output_bitstrings = ['100', '000', '110', '000']

        for i, test_pauli in enumerate(test_paulis):
            print(i)
            phase_update, output_bitstring = _eprop.pauli_phase_update_all_zeros(test_pauli)
            self.assertEqual(phase_update, correct_phase_updates_standard[i])
            self.assertEqual(output_bitstring, correct_output_bitstrings[i])
            
        for i, test_pauli in enumerate(test_paulis):
            print(i)
            phase_update, output_bitstring = _eprop.pauli_phase_update_all_zeros(test_pauli, dual=True)
            self.assertEqual(phase_update, correct_phase_updates_dual[i])
            self.assertEqual(output_bitstring, correct_output_bitstrings[i])

    def test_phi(self):
        bit_strings_3Q = list(product(['0','1'], repeat=3))
        rng = np.random.default_rng()
        paulis = np.fromiter(stim.PauliString.iter_all(3), dtype=object)
        random_paulis = rng.choice(paulis, size=10, replace=False)
        for bit_string in bit_strings_3Q:
            for pauli_1, pauli_2 in product(random_paulis, random_paulis):
                phi_num = _eprop.phi_numerical(self.circuit_tableau_3Q, bit_string, pauli_1, pauli_2)
                phi_analytic = _eprop.phi(self.circuit_tableau_3Q, bit_string, pauli_1, pauli_2)
                if abs(phi_num-phi_analytic) > 1e-4:
                    _eprop.phi(self.circuit_tableau_3Q, bit_string, pauli_1, pauli_2, debug=True)
                    raise ValueError(f'{pauli_1}, {pauli_2}, {bit_string}, {phi_num=}, {phi_analytic=}')
    
    def test_bulk_phi(self):
        bit_strings_3Q = list(product(['0','1'], repeat=3))
        bit_strings_3Q = [''.join(bitstring) for bitstring in bit_strings_3Q]
        rng = np.random.default_rng()
        paulis = np.fromiter(stim.PauliString.iter_all(3), dtype=object)
        random_paulis = list(rng.choice(paulis, size=5, replace=False))

        def _compute_phis(tableau, bitstring, Ps, Qs):
            phis = []
            for P, Q in zip(Ps, Qs):
                phis.append(_eprop.phi(tableau, bitstring, P, Q))
            return phis

        for bitstring in bit_strings_3Q:
            if not np.allclose(_eprop.bulk_phi(self.circuit_tableau_3Q, bitstring, random_paulis, random_paulis), 
                               np.array(_compute_phis(self.circuit_tableau_3Q, bitstring, random_paulis, random_paulis), dtype=np.complex128)):
                print(f'{bitstring=}')
                print(f'{_eprop.bulk_phi(self.circuit_tableau_3Q, bitstring, random_paulis, random_paulis)=}')
                print(f'{_compute_phis(self.circuit_tableau_3Q, bitstring, random_paulis, random_paulis)=}')
                raise ValueError('Bulk and individually computed phi values are different.')        

    def test_alpha(self):
        bit_strings_3Q = list(product(['0','1'], repeat=3))
        complete_errorgen_basis_3Q = CompleteElementaryErrorgenBasis('PP', QubitSpace(3), default_label_type='local')
        rng = np.random.default_rng()
        random_errorgens = rng.choice(np.fromiter(complete_errorgen_basis_3Q.labels, dtype=object), size=100, replace=False)
        for bit_string in bit_strings_3Q:
            for lbl in random_errorgens:
                alpha_num = _eprop.alpha_numerical(lbl, self.circuit_tableau_3Q, bit_string)
                assert abs(alpha_num - _eprop.alpha(lbl, self.circuit_tableau_3Q, bit_string)) <1e-4
    
    def test_bulk_alpha(self):
        from pygsti.modelpacks import smq2Q_XYCPHASE
        pspec_2Q = smq2Q_XYCPHASE.processor_spec()
        random_circuits_2Q = [create_random_circuit(pspec_2Q, 4, sampler='edgegrab', samplerargs=[0.4,], rand_state=12345+i) for i in range(5)]
        random_circuit_tableaus_2Q = [ckt.convert_to_stim_tableau() for ckt in random_circuits_2Q]
    
        def _compute_alphas(errorgens, tableau, bitstring):
            alphas = []
            for errgen in errorgens:
                alphas.append(_eprop.alpha(errgen, tableau, bitstring))
            return alphas

        bitstrings_2Q = ['00', '01', '10', '11']
        rng = np.random.default_rng()
        errorgen_basis = CompleteElementaryErrorgenBasis('PP', QubitSpace(2), default_label_type='local')
        random_errorgens = rng.choice(np.fromiter(errorgen_basis.labels, dtype=object), size=10, replace=False)
        errorgen_labels = [_LSE.cast(lbl) for lbl in random_errorgens]
        
        for i, ckt_tableau in enumerate(random_circuit_tableaus_2Q):
            for bitstring in bitstrings_2Q:
                if not np.allclose(_eprop.bulk_alpha(errorgen_labels, ckt_tableau, [bitstring]), 
                                   np.array(_compute_alphas(errorgen_labels, ckt_tableau, bitstring), dtype=np.double)):
                    print(f'circuit = {random_circuits_2Q[i]}')
                    print(f'{bitstring=}')
                    print(f'{_eprop.bulk_alpha(errorgen_labels, ckt_tableau, [bitstring])=}')
                    print(f'{_compute_alphas(errorgen_labels, ckt_tableau, bitstring)=}')
                    raise ValueError('Bulk and individually computed alpha values are different.')

    def test_alpha_pauli(self):
        from pygsti.modelpacks import smq2Q_XYCPHASE
        pspec_2Q = smq2Q_XYCPHASE.processor_spec()
        random_circuits_2Q = [create_random_circuit(pspec_2Q, 4, sampler='edgegrab', samplerargs=[0.4,], rand_state=12345+i) for i in range(5)]
        random_circuit_tableaus_2Q = [ckt.convert_to_stim_tableau() for ckt in random_circuits_2Q]
        def _compare_alpha_pauli_analytic_numeric(num_qubits, tableau):
            #loop through all error generators and all paulis
            errorgen_basis = CompleteElementaryErrorgenBasis('PP', QubitSpace(num_qubits), default_label_type='local')
            rng = np.random.default_rng()
            random_errorgens = rng.choice(np.fromiter(errorgen_basis.labels, dtype=object), size=10, replace=False)
            errorgen_labels = [_LSE.cast(lbl) for lbl in random_errorgens]
            pauli_list = list(stim.PauliString.iter_all(num_qubits))
            for lbl in errorgen_labels:
                for pauli in pauli_list:
                    alpha_analytic = _eprop.alpha_pauli(lbl, tableau, pauli)
                    alpha_numerical = _eprop.alpha_pauli_numerical(lbl, tableau, pauli)
                    
                    if abs(alpha_analytic - alpha_numerical)>1e-5:
                        print(f'{alpha_analytic=}')
                        print(f'{alpha_numerical=}')
                        print(f'error generator label: {lbl}')
                        print(f'pauli: {pauli}')
                        raise ValueError('Analytic and numerically computed alpha pauli values differ by more than 1e-5')
        for ckt_tableau in random_circuit_tableaus_2Q:
            _compare_alpha_pauli_analytic_numeric(2, ckt_tableau)

    def test_bulk_alpha_pauli(self):
        from pygsti.modelpacks import smq2Q_XYCPHASE
        pspec_2Q = smq2Q_XYCPHASE.processor_spec()
        random_circuits_2Q = [create_random_circuit(pspec_2Q, 4, sampler='edgegrab', samplerargs=[0.4,], rand_state=12345+i) for i in range(5)]
        random_circuit_tableaus_2Q = [ckt.convert_to_stim_tableau() for ckt in random_circuits_2Q]

        def _compute_alphas_pauli(errorgens, tableau, pauli):
            alphas = []
            for errgen in errorgens:
                alphas.append(_eprop.alpha_pauli(errgen, tableau, pauli))
            return alphas

        pauli_list = list(stim.PauliString.iter_all(2))
        rng = np.random.default_rng()
        errorgen_basis = CompleteElementaryErrorgenBasis('PP', QubitSpace(2), default_label_type='local')
        random_errorgens = rng.choice(np.fromiter(errorgen_basis.labels, dtype=object), size=10, replace=False)
        errorgen_labels = [_LSE.cast(lbl) for lbl in random_errorgens]
        random_paulis = rng.choice(np.fromiter(pauli_list, dtype=object), size=5, replace=False)
        
        for i, ckt_tableau in enumerate(random_circuit_tableaus_2Q):
            for pauli in random_paulis:
                if not np.allclose(_eprop.bulk_alpha_pauli(errorgen_labels, ckt_tableau, [pauli]), 
                                   np.array(_compute_alphas_pauli(errorgen_labels, ckt_tableau, pauli), dtype=np.double)):
                    print(f'circuit = {random_circuits_2Q[i]}')
                    print(f'{pauli=}')
                    print(f'{_eprop.bulk_alpha_pauli(errorgen_labels, ckt_tableau, [pauli])=}')
                    print(f'{_compute_alphas_pauli(errorgen_labels, ckt_tableau, pauli)=}')
                    raise ValueError('Bulk and individually computed alpha_pauli values are different.')

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
        #skip second test of third order for now to save on unit test runtime
        #third_order_diff = exact_prop_probs[-1] - _eprop.approximate_stabilizer_probability(self.propagated_errorgen_layer, self.circuit_tableau, '1111', order=3)

        assert abs(first_order_diff) > abs(second_order_diff)
        #assert abs(second_order_diff) > abs(third_order_diff)
        
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


    def test_error_generator_taylor_expansion(self):
        #this is just an integration test atm.
        _eprop.error_generator_taylor_expansion(self.propagated_errorgen_layer, order=2)

@unittest.skipIf(stim is None, "stim is not installed")
class ErrorGenPropUtilsTester(BaseCase):
    pass
#helper functions

def select_random_items_from_multiple_lists(input_lists, num_items, seed=None):
    """
    Select a specified number of items at random from multiple lists without replacement.

    Parameters:
    input_lists (list of lists): The lists from which to select items.
    num_items (int): The number of items to select.
    seed (int, optional): The seed for the random number generator. Defaults to None.

    Returns:
    list of lists: A list of lists containing the randomly selected items from each input list.
    """
    if not input_lists:
        raise ValueError("input_lists cannot be empty")
    
    list_length = len(input_lists[0])
    for lst in input_lists:
        if len(lst) != list_length:
            raise ValueError("All input lists must have the same length")
    
    if num_items > list_length:
        raise ValueError("num_items cannot be greater than the length of the input lists")
    
    if seed is not None:
        random.seed(seed)
    
    indices = random.sample(range(list_length), num_items)
    
    return [[lst[i] for i in indices] for lst in input_lists]

def sample_error_rates_dict(pspec, strengths, seed=None):
    """
    For example:
        strengths = {1: {'S':0.001, 'H':0.01}, 
                    2: {'S':0.01,'H':0.1}}

    The 'S' and 'H' entries in the strengths dictionary give 
    the maximum possible contribution to the infidelity from a given gate.
    """
    qubits = pspec.qubit_labels
    errors_rates_dict = {}
    for gate, availability in pspec.availability.items():
        n = pspec.gate_num_qubits(gate)
        if availability == 'all-edges':
            assert(n == 1), "Currently require all 2-qubit gates have a specified availability!"
            qubits_for_gate = qubits
        else:
            qubits_for_gate = availability  
        for qs in qubits_for_gate:
            label = Label(gate, qs)
            # First, check if there's a strength specified for this specific gate.
            max_stength = strengths.get(label, None) # to get highly biased errors can set generic error rates to be low, then set it to be high for one or two particular gates.
            # Next, check if there's a strength specified for all gates with this name
            if max_stength is None:
                max_stength = strengths.get(gate, None)
            # Finally, get error rate for all gates on this number of qubits.
            if max_stength is None:
                max_stength = strengths[n]
            # Sample error rates.
            errors_rates_dict[label] = sample_error_rates(max_stength, n, seed)
    return errors_rates_dict

def sample_error_rates(strengths, n, seed = None):
    '''
    Samples an error rates dictionary for dependent gates.
    '''
    error_rates_dict = {}
    
    #create a basis to get the basis element labels.
    basis = BuiltinBasis('pp', 4**n)
    
    #set the rng
    rng = np.random.default_rng(seed)
    
    # Sample stochastic error rates. First we sample the overall stochastic error rate.
    # Then we sample (and normalize) the individual stochastic error rates
    stochastic_strength = strengths['S'] * rng.random()
    s_error_rates = rng.random(4 ** n - 1)
    s_error_rates = s_error_rates / np.sum(s_error_rates) * stochastic_strength

    hamiltonian_strength = strengths['H'] * rng.random()
    h_error_rates = rng.random(4 ** n - 1)
    h_error_rates = h_error_rates * np.sqrt(hamiltonian_strength) / np.sqrt(np.sum(h_error_rates**2))

    error_rates_dict.update({('S', basis.labels[i + 1]): s_error_rates[i] for i in range(4 ** n - 1)})
    error_rates_dict.update({('H', basis.labels[i + 1]): h_error_rates[i] for i in range(4 ** n - 1)})

    return error_rates_dict

def probabilities_errorgen_prop(error_propagator, target_model, circuit, use_bch=False, bch_order=1, truncation_threshold=1e-14):
    #get the eoc error channel, and the process matrix for the ideal circuit:
    if use_bch:
        eoc_channel = error_propagator.eoc_error_channel(circuit, include_spam=True, use_bch=use_bch,
                                                        bch_kwargs={'bch_order':bch_order,
                                                                    'truncation_threshold':truncation_threshold})
    else:
        eoc_channel = error_propagator.eoc_error_channel(circuit, include_spam=True)
    ideal_channel = target_model.sim.product(circuit)
    #also get the ideal state prep and povm:
    ideal_prep = target_model.circuit_layer_operator(Label('rho0'), typ='prep').copy()
    ideal_meas = target_model.circuit_layer_operator(Label('Mdefault'), typ='povm').copy()
    #calculate the probabilities.
    prob_vec = np.zeros(len(ideal_meas))
    for i, effect in enumerate(ideal_meas.values()):
        dense_effect = effect.to_dense().copy()
        dense_prep = ideal_prep.to_dense().copy()
        prob_vec[i] = np.linalg.multi_dot([dense_effect.reshape((1, -1)), eoc_channel, ideal_channel, dense_prep.reshape((-1, 1))]).item()
    return prob_vec

def pauli_expectation_errorgen_prop(error_propagator, target_model, circuit, pauli, use_bch=False, bch_order=1, truncation_threshold=1e-14):
    #get the eoc error channel, and the process matrix for the ideal circuit:
    if use_bch:
        eoc_channel = error_propagator.eoc_error_channel(circuit, include_spam=True, use_bch=use_bch,
                                                        bch_kwargs={'bch_order':bch_order,
                                                                    'truncation_threshold':truncation_threshold})
    else:
        eoc_channel = error_propagator.eoc_error_channel(circuit, include_spam=True)
    ideal_channel = target_model.sim.product(circuit)
    #also get the ideal state prep and povm:
    ideal_prep = target_model.circuit_layer_operator(Label('rho0'), typ='prep').copy()
    
    #finally need the superoperator for the selected pauli.
    pauli_unitary = pauli.to_unitary_matrix(endian='big')
    #flatten this row-wise
    pauli_vec = np.ravel(pauli_unitary)
    pauli_vec.reshape((len(pauli_vec),1))
    #put this in pp basis (since these are paulis themselves I could just read this off directly).
    pauli_vec = change_basis(pauli_vec, 'std', 'pp')
    #print(pauli_vec)
    dense_prep = ideal_prep.to_dense().copy()
    expectation = np.linalg.multi_dot([pauli_vec.reshape((1,len(pauli_vec))), eoc_channel, ideal_channel, dense_prep.reshape((len(dense_prep),1))]).item()
    return expectation

#helper function for iterative composition testing
def _compare_analytic_numeric_iterative_composition(num_qubits):
    #create an error generator basis.
    complete_errorgen_basis = CompleteElementaryErrorgenBasis('PP', QubitSpace(num_qubits), default_label_type='local')
    complete_errorgen_lbls = complete_errorgen_basis.labels
    complete_errorgen_lbl_matrix_dict = {lbl: mat for lbl, mat in zip(complete_errorgen_lbls, complete_errorgen_basis.elemgen_matrices)}

    #loop through all triples.
    errorgen_label_triples = list(product(complete_errorgen_lbls,repeat=3))
    
    #select a random subset of these
    rng = np.random.default_rng(seed=1234)
    random_indices = rng.choice(len(errorgen_label_triples), 10000)
    random_triples = [errorgen_label_triples[idx] for idx in random_indices]
    
    #create local stim error gen label versions:
    random_triples_stim = [(_LSE.cast(a), _LSE.cast(b), _LSE.cast(c)) for a,b,c in random_triples]
    
    #for each triple compute the composition directly and compute it analytically (then converting it to
    #a numeric array) and see how they compare.
    for i, (triple_1, triple_2) in enumerate(zip(random_triples, random_triples_stim)):
        numeric_composition = _eprop.iterative_error_generator_composition_numerical(triple_1, (1,1,1), complete_errorgen_lbl_matrix_dict)
        analytic_composition = _eprop.iterative_error_generator_composition(triple_2, (1,1,1))
        analytic_composition_dict = dict()
        for lbl, rate in analytic_composition:
            local_lbl = lbl.to_local_eel()
            if analytic_composition_dict.get(local_lbl, None) is None:
                analytic_composition_dict[local_lbl] = rate
            else:
                analytic_composition_dict[local_lbl] += rate
        analytic_composition = analytic_composition_dict
        try:
            analytic_composition_mat = _eprop.errorgen_layer_to_matrix(analytic_composition, num_qubits, errorgen_matrix_dict = complete_errorgen_lbl_matrix_dict)        
        except KeyError:
            print(f'{analytic_composition=}')
        norm_diff = np.linalg.norm(numeric_composition-analytic_composition_mat)
        if norm_diff > 1e-10:
            print(f'Difference in compositions for triple {triple_1} is greater than 1e-10.')
            print(f'{triple_2=}')
            print(f'Error encountered on iteration {i}')
            print(f'{np.linalg.norm(numeric_composition-analytic_composition_mat)=}')
            print('numeric_composition=')
            print_mx(numeric_composition)
            
            #Decompose the numerical composition into rates.
            for lbl, dual in zip(complete_errorgen_basis.labels, complete_errorgen_basis.elemgen_dual_matrices):
                rate = np.trace(dual.conj().T@numeric_composition)
                if abs(rate) >1e-3:
                    print(f'{lbl}: {rate}')
            
            print(f'{analytic_composition=}')
            print('analytic_composition_mat=')
            print_mx(analytic_composition_mat)
            raise ValueError('Numeric and analytic error generator compositions were not found to be identical!')

#helper function for zassenhaus formula testing
def _dense_zassenhaus_generators_analytic(errorgen_groups, errogen_propagator, zassenhaus_order=1):
    zassenhaus_gens= _eprop.zassenhaus_formula(errorgen_groups, zassenhaus_order)
    dense_zassenhaus_generators = [errogen_propagator.errorgen_layer_dict_to_errorgen(gen) for gen in zassenhaus_gens]    
    return dense_zassenhaus_generators


# ---------------------------------------------------------------------------
# Independent reference implementation for the H,H / H,S / S,H composition
# blocks of `error_generator_composition`.
#
# These helpers deliberately avoid `_eprop.pauli_product` and
# `_eprop.stim_pauli_string_less_than` (and stim itself) so that the reference
# is not merely a restatement of the code under test. The closed forms are
# derived from the Lindblad elementary error generator definitions
#     H_P[rho]     = -i[P, rho]
#     S_P[rho]     = P rho P - rho
#     C_{P,Q}[rho] = P rho Q + Q rho P - 0.5{{P,Q}, rho}
#     A_{P,Q}[rho] = i(P rho Q - Q rho P + 0.5{[P,Q], rho})
# and are additionally cross-checked numerically in the tests below against
# the dense superoperator product.
# ---------------------------------------------------------------------------

_SINGLE_PAULI_PRODUCT = {
    ('I', 'I'): (1, 'I'), ('I', 'X'): (1, 'X'), ('I', 'Y'): (1, 'Y'), ('I', 'Z'): (1, 'Z'),
    ('X', 'I'): (1, 'X'), ('Y', 'I'): (1, 'Y'), ('Z', 'I'): (1, 'Z'),
    ('X', 'X'): (1, 'I'), ('Y', 'Y'): (1, 'I'), ('Z', 'Z'): (1, 'I'),
    ('X', 'Y'): (1j, 'Z'), ('Y', 'X'): (-1j, 'Z'),
    ('Y', 'Z'): (1j, 'X'), ('Z', 'Y'): (-1j, 'X'),
    ('Z', 'X'): (1j, 'Y'), ('X', 'Z'): (-1j, 'Y'),
}


def _ref_pauli_product(pauli_1, pauli_2):
    """Return (phase, unsigned_product_string) for two equal-length Pauli strings."""
    phase = 1
    letters = []
    for a, b in zip(pauli_1, pauli_2):
        ph, letter = _SINGLE_PAULI_PRODUCT[(a, b)]
        phase *= ph
        letters.append(letter)
    return phase, ''.join(letters)


def _ref_pauli_commutes(pauli_1, pauli_2):
    """Two Pauli strings commute iff they anticommute on an even number of sites."""
    num_anticommuting = sum(1 for a, b in zip(pauli_1, pauli_2)
                            if a != 'I' and b != 'I' and a != b)
    return num_anticommuting % 2 == 0


def _ref_bels_C(pauli_1, pauli_2, first_ident, second_ident, paulis_eq):
    """C-type label canonicalization. Returns None when the generator vanishes."""
    if first_ident or second_ident:
        return None
    if paulis_eq:
        return ('S', (pauli_1,), 2)
    ordered = (pauli_1, pauli_2) if pauli_1 < pauli_2 else (pauli_2, pauli_1)
    return ('C', ordered, 1)


def _ref_bels_A(pauli_1, pauli_2, first_ident, second_ident, paulis_eq):
    """A-type label canonicalization. Returns None when the generator vanishes."""
    if paulis_eq:
        return None
    if first_ident:
        if second_ident:
            return None
        return ('H', (pauli_2,), 1)
    if second_ident:
        return ('H', (pauli_1,), -1)
    if pauli_1 < pauli_2:
        return ('A', (pauli_1, pauli_2), 1)
    return ('A', (pauli_2, pauli_1), -1)


def _ref_compose_HH(P, Q, weight=1.0):
    """Reference for H_P[H_Q[.]]."""
    terms = []
    phase, PQ = _ref_pauli_product(P, Q)
    if not _ref_pauli_commutes(P, Q):
        terms.append((('H', (PQ,)), -1j * weight * phase))
    entry = _ref_bels_C(P, Q, False, False, P == Q)
    terms.append(((entry[0], entry[1]), entry[2] * weight))
    return terms


def _ref_compose_HS(P, Q, weight=1.0):
    """Reference for H_P[S_Q[.]]."""
    terms = []
    identity = 'I' * len(P)
    phase, PQ = _ref_pauli_product(P, Q)
    PQ_ident = (PQ == identity)
    PQ_eq_Q = (PQ == Q)
    if _ref_pauli_commutes(P, Q):
        entry = _ref_bels_A(PQ, Q, PQ_ident, False, PQ_eq_Q)
        if entry is not None:
            terms.append(((entry[0], entry[1]), -phase * entry[2] * weight))
    else:
        entry = _ref_bels_C(PQ, Q, PQ_ident, False, PQ_eq_Q)
        if entry is not None:
            terms.append(((entry[0], entry[1]), -1j * phase * entry[2] * weight))
    terms.append((('H', (P,)), -weight))
    return terms


def _ref_compose_SH(P, Q, weight=1.0):
    """Reference for S_P[H_Q[.]]."""
    terms = []
    identity = 'I' * len(P)
    phase, PQ = _ref_pauli_product(P, Q)
    PQ_ident = (PQ == identity)
    PQ_eq_Q = (PQ == Q)
    if _ref_pauli_commutes(P, Q):
        entry = _ref_bels_A(PQ, P, PQ_ident, False, PQ_eq_Q)
        if entry is not None:
            terms.append(((entry[0], entry[1]), -phase * entry[2] * weight))
    else:
        entry = _ref_bels_C(PQ, P, PQ_ident, False, PQ_eq_Q)
        if entry is not None:
            terms.append(((entry[0], entry[1]), -1j * phase * entry[2] * weight))
    terms.append((('H', (Q,)), -weight))
    return terms


def _ref_compose_SS(P, Q, weight=1.0):
    """
    Reference for S_P[S_Q[.]].

    Derivation: S_P[S_Q[rho]] = P Q rho Q P - Q rho Q - P rho P + rho. Since
    (PQ)^dagger = QP for Hermitian Paulis, the leading term is R rho R with R the
    *unsigned* product PQ (the phases cancel), giving S_P S_Q = S_R - S_P - S_Q,
    where S_I == 0 (so the first term is dropped when P == Q).
    """
    terms = []
    identity = 'I' * len(P)
    _, PQ = _ref_pauli_product(P, Q)
    if PQ != identity:
        terms.append((('S', (PQ,)), weight))
    terms.append((('S', (P,)), -weight))
    terms.append((('S', (Q,)), -weight))
    return terms


def _non_identity_paulis(num_qubits):
    """All non-identity Pauli strings on `num_qubits` qubits, lexicographically ordered."""
    return [''.join(p) for p in product('IXYZ', repeat=num_qubits) if set(p) != {'I'}]


def _c_type_pauli_pairs(num_qubits):
    """All valid (P, Q) basis element pairs for a C-type label (distinct, canonically ordered)."""
    paulis = _non_identity_paulis(num_qubits)
    return [(p, q) for p, q in product(paulis, repeat=2) if p < q]


# ---------------------------------------------------------------------------
# Matrix-free oracle for the composition of two elementary error generators.
#
# `_string_oracle_composition` decomposes a composition onto the elementary error
# generator basis using nothing but Pauli-string arithmetic, via the "sandwich"
# representation of a superoperator as a sum of maps  rho -> M rho N  with M, N
# Pauli strings. Composition is then just (M1,N1) o (M2,N2) = (M1 M2, N2 N1), so
# an elementary-generator composition needs at most 6*6 = 36 string products.
#
# The decomposition is unique and derives only from the definitions of H, S, C and
# A, never from `error_generator_composition`, so it independently pins every
# emitted label and rate. Its cost is independent of 4^n (~23 us even at 3 qubits,
# versus ~180 ms for a dense 64x64 projection onto 4032 dual matrices), which is
# what makes exhaustive C/A coverage tractable.
# ---------------------------------------------------------------------------


def _sandwich_expansion(errorgen_type, bels, num_qubits):
    """
    Expand an elementary error generator as {(M, N): coeff} meaning rho -> M rho N.

        H_P     = -i P rho + i rho P
        S_P     = P rho P - rho
        C_{P,Q} = P rho Q + Q rho P - 0.5 ({P,Q} rho + rho {P,Q})
        A_{P,Q} = i (P rho Q - Q rho P + 0.5 ([P,Q] rho + rho [P,Q]))
    """
    identity = 'I' * num_qubits
    out = {}

    def acc(M, N, coeff):
        out[(M, N)] = out.get((M, N), 0) + coeff

    if errorgen_type == 'H':
        P = bels[0]
        acc(P, identity, -1j)
        acc(identity, P, 1j)
    elif errorgen_type == 'S':
        P = bels[0]
        acc(P, P, 1)
        acc(identity, identity, -1)
    else:
        P, Q = bels
        phase_pq, R = _ref_pauli_product(P, Q)
        phase_qp, _ = _ref_pauli_product(Q, P)
        if errorgen_type == 'C':
            acc(P, Q, 1)
            acc(Q, P, 1)
            anticomm = -0.5 * (phase_pq + phase_qp)
            acc(R, identity, anticomm)
            acc(identity, R, anticomm)
        elif errorgen_type == 'A':
            acc(P, Q, 1j)
            acc(Q, P, -1j)
            comm = 0.5j * (phase_pq - phase_qp)
            acc(R, identity, comm)
            acc(identity, R, comm)
        else:
            raise ValueError(f'unknown error generator type {errorgen_type!r}')
    return {k: v for k, v in out.items() if abs(v) > 1e-15}


def _compose_sandwich(first, second, tol=1e-12):
    """Compose two sandwich expansions: first[second[.]] -> (M1 M2, N2 N1)."""
    out = {}
    for (M1, N1), c1 in first.items():
        for (M2, N2), c2 in second.items():
            phase_m, M = _ref_pauli_product(M1, M2)
            phase_n, N = _ref_pauli_product(N2, N1)
            key = (M, N)
            out[key] = out.get(key, 0) + c1 * c2 * phase_m * phase_n
    return {k: v for k, v in out.items() if abs(v) > tol}


def _decompose_sandwich(expansion, num_qubits, tol=1e-9):
    """
    Invert `_sandwich_expansion` for a whole superoperator.

    The decomposition is read off directly:
      * (P, P) with P non-identity arises only from S_P.
      * (P, Q) and (Q, P) with P != Q both non-identity arise only from C_{P,Q}
        (coefficient 1 each) and A_{P,Q} (coefficients +i and -i), so
        rate_C = (c_PQ + c_QP)/2 and rate_A = (c_PQ - c_QP)/2i.
      * (R, identity) picks up -i*rate_H(R) plus the anticommutator/commutator
        tails of the C and A rates already recovered; subtracting those leaves H.
    """
    identity = 'I' * num_qubits
    rates = {}

    for (M, N), coeff in expansion.items():
        if M == N and M != identity and abs(coeff) > tol:
            rates[('S', (M,))] = coeff

    handled = set()
    for (M, N) in list(expansion):
        if M == identity or N == identity or M == N:
            continue
        key = (M, N) if M < N else (N, M)
        if key in handled:
            continue
        handled.add(key)
        P, Q = key
        c_pq = expansion.get((P, Q), 0)
        c_qp = expansion.get((Q, P), 0)
        rate_c = (c_pq + c_qp) / 2
        rate_a = (c_pq - c_qp) / 2j
        if abs(rate_c) > tol:
            rates[('C', (P, Q))] = rate_c
        if abs(rate_a) > tol:
            rates[('A', (P, Q))] = rate_a

    residual = {}
    for (M, N), coeff in expansion.items():
        if N == identity and M != identity:
            residual[M] = residual.get(M, 0) + coeff
    for (egtype, bels), rate in list(rates.items()):
        if egtype in ('C', 'A'):
            P, Q = bels
            phase_pq, R = _ref_pauli_product(P, Q)
            phase_qp, _ = _ref_pauli_product(Q, P)
            tail = (-0.5 * (phase_pq + phase_qp) * rate if egtype == 'C'
                    else 0.5j * (phase_pq - phase_qp) * rate)
            residual[R] = residual.get(R, 0) - tail
    for R, coeff in residual.items():
        rate_h = coeff / -1j
        if abs(rate_h) > tol:
            rates[('H', (R,))] = rate_h

    return rates


def _string_oracle_composition(label_1, label_2, num_qubits, weight=1.0):
    """
    Oracle 2: compose two elementary error generators using only Pauli strings.

    `label_1`/`label_2` are (errorgen_type, bel_strings) tuples; returns
    {LocalElementaryErrorgenLabel: rate} with duplicate labels already summed.
    """
    expansion = _compose_sandwich(_sandwich_expansion(*label_1, num_qubits),
                                  _sandwich_expansion(*label_2, num_qubits))
    if weight != 1.0:
        expansion = {k: v * weight for k, v in expansion.items()}
    return {LEEL(egtype, tuple(bels)): rate
            for (egtype, bels), rate in _decompose_sandwich(expansion, num_qubits).items()}


def _composition_block_max_terms():
    """
    Return {(type_1, type_2): max_appends_on_any_path} for each dispatch block.

    Derived from the AST rather than hard-coded, so the bound stays correct if the
    implementation is refactored. The number of `composed_errorgens.append` calls
    along a single path is an exact upper bound on the returned list length.

    Loops are counted by unrolling: a `for` over a literal tuple or list runs a known
    number of times, so its body counts once per element. Any loop whose trip count
    can't be read off the AST raises, since silently treating it as zero would make
    the bound unsound.
    """
    tree = ast.parse(open(_eprop.__file__).read())
    # The public `error_generator_composition` is a thin caching wrapper; the dispatch
    # chain this walks lives in the uncached implementation behind it.
    fn = next(n for n in tree.body
              if isinstance(n, ast.FunctionDef) and n.name == '_error_generator_composition_impl')
    module_fns = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}

    def is_append(stmt):
        return (isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call)
                and isinstance(stmt.value.func, ast.Attribute)
                and stmt.value.func.attr == 'append'
                and isinstance(stmt.value.func.value, ast.Name)
                and stmt.value.func.value.id == 'composed_errorgens')

    def called_helper(value, scope):
        """Name of a known function being called, if this expression is such a call."""
        if isinstance(value, ast.Call) and isinstance(value.func, ast.Name):
            return value.func.id if value.func.id in scope else None
        return None

    def max_appends(stmts, depth=0, scope=None):
        # Blocks delegate to shared module-level helpers (e.g.
        # `_error_generator_composition_cc`), and those helpers in turn use nested
        # closures to emit terms, so follow calls into both; otherwise the bound
        # silently reads far too low.
        scope = dict(module_fns if scope is None else scope)
        for stmt in stmts:
            if isinstance(stmt, ast.FunctionDef):
                scope[stmt.name] = stmt

        best = 0
        for stmt in stmts:
            if is_append(stmt):
                best += 1
            elif isinstance(stmt, ast.FunctionDef):
                continue                      # counted at its call sites instead
            elif isinstance(stmt, ast.If):
                best += max(max_appends(stmt.body, depth, scope),
                            max_appends(stmt.orelse, depth, scope) if stmt.orelse else 0)
            elif isinstance(stmt, ast.For):
                if not isinstance(stmt.iter, (ast.Tuple, ast.List)):
                    raise AssertionError(
                        'cannot statically bound the trip count of the loop at line '
                        f'{stmt.lineno} (iterating over {ast.unparse(stmt.iter)}); '
                        '_composition_block_max_terms needs updating')
                best += len(stmt.iter.elts) * max_appends(stmt.body, depth, scope)
                best += max_appends(stmt.orelse, depth, scope) if stmt.orelse else 0
            elif isinstance(stmt, ast.While):
                raise AssertionError(
                    f'cannot statically bound the while loop at line {stmt.lineno}; '
                    '_composition_block_max_terms needs updating')
            elif isinstance(stmt, (ast.Assign, ast.Return, ast.Expr)) and depth < 6:
                helper = called_helper(stmt.value, scope) if stmt.value is not None else None
                if helper is not None:
                    best += max_appends(scope[helper].body, depth + 1, scope)
        return best

    head = "errorgen_1_type == 'H' and errorgen_2_type == 'H'"
    node = next(s for s in fn.body if isinstance(s, ast.If) and head in ast.unparse(s.test))
    bounds = {}
    while node is not None:
        types = re.findall(r"errorgen_[12]_type == '(\w)'", ast.unparse(node.test))
        if len(types) == 2:
            bounds[(types[0], types[1])] = max_appends(node.body)
        nested = [s for s in node.orelse if isinstance(s, ast.If)]
        node = nested[0] if (len(node.orelse) == 1 and nested) else None
    return bounds


def _commutation_signature(A, B, P, Q):
    """
    The six commutation bits the C/A-vs-C/A blocks branch on.

    These blocks dispatch on [A,B], then [P,Q], then the four cross relations, so
    this tuple identifies which of the 64 structural sub-cases an input lands in.
    """
    return (_ref_pauli_commutes(A, B), _ref_pauli_commutes(P, Q),
            _ref_pauli_commutes(A, P), _ref_pauli_commutes(A, Q),
            _ref_pauli_commutes(B, P), _ref_pauli_commutes(B, Q))


def _signature_representatives(num_qubits):
    """
    One deterministic (A, B, P, Q) representative per reachable commutation
    signature, found by an in-order scan (all 64 appear within ~4.7k pairs at 3
    qubits, so this is fast even though the full space is ~3.8M pairs).
    """
    pair_labels = _c_type_pauli_pairs(num_qubits)
    reps = {}
    for A, B in pair_labels:
        for P, Q in pair_labels:
            reps.setdefault(_commutation_signature(A, B, P, Q), (A, B, P, Q))
        if len(reps) == 64:
            break
    return reps


def _composition_dispatch_block_ranges():
    """
    Return {(type_1, type_2): (first_line, last_line)} for each of the 16 blocks in
    the `_error_generator_composition_impl` elif chain, read from the installed source.

    Line numbers are absolute so they can be compared directly against the line
    numbers reported by `sys.monitoring` for the function's code object.
    """
    tree = ast.parse(open(_eprop.__file__).read())
    fn = next(n for n in tree.body
              if isinstance(n, ast.FunctionDef) and n.name == '_error_generator_composition_impl')
    head = "errorgen_1_type == 'H' and errorgen_2_type == 'H'"
    node = next(s for s in fn.body if isinstance(s, ast.If) and head in ast.unparse(s.test))

    ranges = {}
    while node is not None:
        types = re.findall(r"errorgen_[12]_type == '(\w)'", ast.unparse(node.test))
        if len(types) == 2:
            lo = min(s.lineno for s in node.body)
            hi = max(getattr(s, 'end_lineno', s.lineno) for s in node.body)
            ranges[(types[0], types[1])] = (lo, hi)
        nested = [s for s in node.orelse if isinstance(s, ast.If)]
        node = nested[0] if (len(node.orelse) == 1 and nested) else None
    return ranges


def _merge_terms(terms):
    """Sum duplicate labels in an (LSE, rate) list, returning a {LEEL: rate} dict."""
    merged = {}
    for lbl, rate in terms:
        key = lbl.to_local_eel()
        merged[key] = merged.get(key, 0) + rate
    return {k: v for k, v in merged.items() if abs(v) > 1e-12}


def _ref_terms_to_lse(terms):
    """Convert reference (type, bel_strings, rate) triples into (LSE, rate) tuples."""
    return [(_LSE(egtype, [stim.PauliString(b) for b in bels]), rate)
            for (egtype, bels), rate in terms]


@unittest.skipIf(stim is None, "stim is not installed")
class ErrgenCompositionRegressionTester(BaseCase):

    def test_property_weight_linearity(self):
        """Verify composition(a, b, w) == w * composition(a, b, 1.0)."""
        basis_2q = CompleteElementaryErrorgenBasis('PP', QubitSpace(2), default_label_type='local')
        labels = [_LSE.cast(lbl) for lbl in basis_2q.labels[:20]]
        weights = [0.0, 2.5, -1.0, 1j, 2.0 - 3.0j]
        for l1, l2 in zip(labels, labels[::-1]):
            base_res = _eprop.error_generator_composition(l1, l2, weight=1.0)
            for w in weights:
                scaled_res = _eprop.error_generator_composition(l1, l2, weight=w)
                self.assertEqual(len(scaled_res), len(base_res))
                for (lbl1, r1), (lbl2, r2) in zip(base_res, scaled_res):
                    self.assertEqual(lbl1, lbl2)
                    self.assertAlmostEqual(r2, w * r1, places=10)

    def test_property_explicit_identity_invariance(self):
        """Verify passing identity=stim.PauliString('II') matches identity=None."""
        ident_2q = stim.PauliString('II')
        basis_2q = CompleteElementaryErrorgenBasis('PP', QubitSpace(2), default_label_type='local')
        labels = [_LSE.cast(lbl) for lbl in basis_2q.labels[:20]]
        for l1, l2 in zip(labels, labels[::-1]):
            res_default = _eprop.error_generator_composition(l1, l2, identity=None)
            res_explicit = _eprop.error_generator_composition(l1, l2, identity=ident_2q)
            self.assertEqual(res_default, res_explicit)

    def test_property_label_well_formedness(self):
        """Verify returned LSE labels are well-formed."""
        basis_2q = CompleteElementaryErrorgenBasis('PP', QubitSpace(2), default_label_type='local')
        labels = [_LSE.cast(lbl) for lbl in basis_2q.labels[:30]]
        for l1 in labels[:10]:
            for l2 in labels:
                res = _eprop.error_generator_composition(l1, l2)
                for out_lbl, r in res:
                    self.assertIn(out_lbl.errorgen_type, ('H', 'S', 'C', 'A'))
                    bels = out_lbl.basis_element_labels
                    if out_lbl.errorgen_type in ('H', 'S'):
                        self.assertEqual(len(bels), 1)
                    elif out_lbl.errorgen_type in ('C', 'A'):
                        self.assertEqual(len(bels), 2)
                        self.assertNotEqual(bels[0], bels[1])
                        self.assertTrue(_eprop.stim_pauli_string_less_than(bels[0], bels[1]))

    def test_pep669_all_dispatch_blocks_are_reachable(self):
        """
        PEP 669 line-coverage gate: drive one composition per (type_1, type_2) pair
        and assert that every one of the 16 dispatch blocks actually executed.

        This guards against a refactor silently making a block unreachable (e.g. by
        reordering the elif chain so an earlier predicate shadows a later one), which
        no value-based test would catch -- a shadowed block just never runs.

        `sys.monitoring` is used rather than `sys.settrace` to avoid the ~40x
        interpreter slowdown. It is skipped only when the COVERAGE_ID tool slot is
        genuinely already claimed (i.e. coverage really is collecting), not merely
        when pytest-cov happens to be installed.
        """
        if sys.version_info < (3, 12):
            self.skipTest("PEP 669 sys.monitoring requires Python 3.12+")

        tool_id = sys.monitoring.COVERAGE_ID
        if sys.monitoring.get_tool(tool_id) is not None:
            self.skipTest(f"COVERAGE_ID slot already claimed by "
                          f"{sys.monitoring.get_tool(tool_id)!r}; coverage is active")
        try:
            sys.monitoring.use_tool_id(tool_id, "errgencomp_gate")
        except ValueError:
            self.skipTest("Could not acquire the sys.monitoring COVERAGE_ID tool slot")

        # Monitor the uncached implementation: the public entry point is a caching
        # wrapper and contains none of the dispatch chain. Clearing the cache is
        # essential here -- a warm entry from an earlier test would short-circuit the
        # implementation entirely and the block would look unreached.
        _eprop.error_generator_composition.cache_clear()
        target_code = _eprop._error_generator_composition_impl.__code__
        executed_lines = set()

        def line_callback(code, line_no):
            if code is target_code:
                executed_lines.add(line_no)
            return sys.monitoring.DISABLE

        try:
            sys.monitoring.register_callback(tool_id, sys.monitoring.events.LINE, line_callback)
            sys.monitoring.set_events(tool_id, sys.monitoring.events.LINE)

            basis = CompleteElementaryErrorgenBasis('PP', QubitSpace(2), default_label_type='local')
            by_type = {}
            for lbl in basis.labels:
                by_type.setdefault(lbl.errorgen_type, []).append(_LSE.cast(lbl))

            for type_1 in ('H', 'S', 'C', 'A'):
                for type_2 in ('H', 'S', 'C', 'A'):
                    first = by_type[type_1]
                    second = by_type[type_2]
                    for i in range(min(10, len(first))):
                        _eprop.error_generator_composition(first[i], second[(i * 7) % len(second)])
        finally:
            sys.monitoring.set_events(tool_id, 0)
            sys.monitoring.register_callback(tool_id, sys.monitoring.events.LINE, None)
            sys.monitoring.free_tool_id(tool_id)

        self.assertGreater(len(executed_lines), 0, "sys.monitoring recorded no lines at all.")

        block_ranges = _composition_dispatch_block_ranges()
        self.assertEqual(len(block_ranges), 16,
                         f'expected 16 dispatch blocks, found {sorted(block_ranges)}')
        unreached = [f'{t1},{t2} (lines {lo}-{hi})'
                     for (t1, t2), (lo, hi) in sorted(block_ranges.items())
                     if not any(lo <= ln <= hi for ln in executed_lines)]
        self.assertEqual(unreached, [],
                         f'dispatch blocks never executed: {unreached}')


@unittest.skipIf(stim is None, "stim is not installed")
class ErrgenCompositionHSBlockTester(BaseCase):
    """
    Term-level (label *and* rate *and* ordering) tests for the H,H / H,S and S,H
    dispatch blocks of `error_generator_composition`.

    The pre-existing `test_errorgen_composition` only compares the *summed* dense
    superoperator, which cannot detect a mis-ordered output list, a term that was
    split into two duplicates, or two errors that cancel in the sum. These tests
    pin the exact returned list, and separately cross-check it numerically.
    """

    WEIGHTS = (1.0, 2.5, -1.0, 1j, 2.0 - 3.0j)

    def _errorgen_matrix_dict(self, num_qubits):
        basis = CompleteElementaryErrorgenBasis('PP', QubitSpace(num_qubits), default_label_type='local')
        return {lbl: mat for lbl, mat in zip(basis.labels, basis.elemgen_matrices)}

    def _assert_terms_equal(self, actual, expected, context):
        self.assertEqual(len(actual), len(expected),
                         f"{context}: expected {len(expected)} terms, got {len(actual)}. "
                         f"actual={actual} expected={expected}")
        for idx, ((act_lbl, act_rate), (exp_lbl, exp_rate)) in enumerate(zip(actual, expected)):
            self.assertEqual(act_lbl, exp_lbl,
                             f"{context}: term {idx} label mismatch: {act_lbl} != {exp_lbl}")
            self.assertAlmostEqual(act_rate, exp_rate, places=12,
                                   msg=f"{context}: term {idx} ({act_lbl}) rate {act_rate} != {exp_rate}")

    def _run_block(self, type_1, type_2, ref_fn, num_qubits, expected_lengths):
        """Exhaustively compare one dispatch block against the independent reference."""
        paulis = _non_identity_paulis(num_qubits)
        identity = stim.PauliString('I' * num_qubits)
        errorgen_matrix_dict = self._errorgen_matrix_dict(num_qubits)
        observed_lengths = set()
        num_checked = 0

        for P, Q in product(paulis, repeat=2):
            lbl_1 = _LSE(type_1, [stim.PauliString(P)])
            lbl_2 = _LSE(type_2, [stim.PauliString(Q)])
            for weight in self.WEIGHTS:
                actual = _eprop.error_generator_composition(lbl_1, lbl_2, weight=weight, identity=identity)
                expected = _ref_terms_to_lse(ref_fn(P, Q, weight))
                context = f'{type_1}_{P} o {type_2}_{Q} (weight={weight}, {num_qubits}Q)'
                self._assert_terms_equal(actual, expected, context)
                observed_lengths.add(len(actual))
                num_checked += 1

            # Independent numerical oracle: the dense superoperator product must
            # equal the dense form of the analytic term list.
            leel_1 = LEEL(type_1, (P,))
            leel_2 = LEEL(type_2, (Q,))
            numeric = _eprop.error_generator_composition_numerical(leel_1, leel_2, errorgen_matrix_dict)
            analytic = _eprop.error_generator_composition(lbl_1, lbl_2, identity=identity)
            analytic_mat = _eprop.errorgen_layer_to_matrix(analytic, num_qubits,
                                                           errorgen_matrix_dict=errorgen_matrix_dict)
            norm_diff = np.linalg.norm(numeric - analytic_mat)
            self.assertLess(norm_diff, 1e-10,
                            f'{type_1}_{P} o {type_2}_{Q} ({num_qubits}Q): numeric vs analytic differ by {norm_diff}')

        self.assertGreater(num_checked, 0, 'No compositions were exercised.')
        self.assertEqual(observed_lengths, expected_lengths,
                         f'{type_1},{type_2} ({num_qubits}Q): observed output lengths {sorted(observed_lengths)} '
                         f'!= expected {sorted(expected_lengths)}')

    # ------------------------------------------------------------------ H,H

    def test_HH_block_golden_cases(self):
        """Hand-checked H,H results, guarding against reference/implementation co-drift."""
        cases = [
            # P == Q  ->  C_{P,P} collapses to 2*S_P (single term)
            (('X',), ('X',), [(_LSE('S', [stim.PauliString('X')]), 2.0)]),
            # distinct but commuting -> single C term
            (('IX',), ('XI',), [(_LSE('C', [stim.PauliString('IX'), stim.PauliString('XI')]), 1.0)]),
            # anticommuting -> H term (from the commutator) followed by the C term
            (('X',), ('Y',), [(_LSE('H', [stim.PauliString('Z')]), 1.0 + 0j),
                              (_LSE('C', [stim.PauliString('X'), stim.PauliString('Y')]), 1.0)]),
        ]
        for (P,), (Q,), expected in cases:
            actual = _eprop.error_generator_composition(_LSE('H', [stim.PauliString(P)]),
                                                        _LSE('H', [stim.PauliString(Q)]))
            self._assert_terms_equal(actual, expected, f'golden H_{P} o H_{Q}')

    def test_HH_block_exhaustive_1q(self):
        # 1 qubit: distinct non-identity Paulis always anticommute, so only the
        # P==Q (1 term) and anticommuting (2 term) cases are reachable.
        self._run_block('H', 'H', _ref_compose_HH, 1, expected_lengths={1, 2})

    def test_HH_block_exhaustive_2q(self):
        self._run_block('H', 'H', _ref_compose_HH, 2, expected_lengths={1, 2})

    def test_HH_case_coverage(self):
        """Ensure all three H,H structural cases are actually reachable on 2 qubits."""
        seen = set()
        for P, Q in product(_non_identity_paulis(2), repeat=2):
            if P == Q:
                seen.add('equal')
            elif _ref_pauli_commutes(P, Q):
                seen.add('commuting_distinct')
            else:
                seen.add('anticommuting')
        self.assertEqual(seen, {'equal', 'commuting_distinct', 'anticommuting'})

    # ------------------------------------------------------------------ H,S

    def test_HS_block_golden_cases(self):
        cases = [
            # P == Q: the A-term degenerates to an H term, duplicating H_P (total -2 H_X)
            (('X',), ('X',), [(_LSE('H', [stim.PauliString('X')]), -1.0),
                              (_LSE('H', [stim.PauliString('X')]), -1.0)]),
            # anticommuting -> C term then H_P
            (('X',), ('Y',), [(_LSE('C', [stim.PauliString('Y'), stim.PauliString('Z')]), 1.0 + 0j),
                              (_LSE('H', [stim.PauliString('X')]), -1.0)]),
            # commuting and distinct -> A term then H_P
            (('IX',), ('XI',), [(_LSE('A', [stim.PauliString('XI'), stim.PauliString('XX')]), 1.0),
                                (_LSE('H', [stim.PauliString('IX')]), -1.0)]),
        ]
        for (P,), (Q,), expected in cases:
            actual = _eprop.error_generator_composition(_LSE('H', [stim.PauliString(P)]),
                                                        _LSE('S', [stim.PauliString(Q)]))
            self._assert_terms_equal(actual, expected, f'golden H_{P} o S_{Q}')

    def test_HS_block_exhaustive_1q(self):
        # H,S always emits exactly two terms (the H_P term is unguarded).
        self._run_block('H', 'S', _ref_compose_HS, 1, expected_lengths={2})

    def test_HS_block_exhaustive_2q(self):
        self._run_block('H', 'S', _ref_compose_HS, 2, expected_lengths={2})

    def test_HS_duplicate_label_terms_are_not_merged(self):
        """H_P[S_P] returns two separate H_P entries; callers must sum duplicates."""
        lbl = _LSE('H', [stim.PauliString('X')])
        actual = _eprop.error_generator_composition(lbl, _LSE('S', [stim.PauliString('X')]))
        self.assertEqual(len(actual), 2)
        self.assertEqual(actual[0][0], actual[1][0],
                         'Expected H_X[S_X] to emit the same label twice (unmerged).')

    # ------------------------------------------------------------------ S,H

    def test_SH_block_golden_cases(self):
        cases = [
            (('X',), ('X',), [(_LSE('H', [stim.PauliString('X')]), -1.0),
                              (_LSE('H', [stim.PauliString('X')]), -1.0)]),
            # anticommuting -> C term built from (PQ, P), then H_Q
            (('X',), ('Y',), [(_LSE('C', [stim.PauliString('X'), stim.PauliString('Z')]), 1.0 + 0j),
                              (_LSE('H', [stim.PauliString('Y')]), -1.0)]),
            (('IX',), ('XI',), [(_LSE('A', [stim.PauliString('IX'), stim.PauliString('XX')]), 1.0),
                                (_LSE('H', [stim.PauliString('XI')]), -1.0)]),
        ]
        for (P,), (Q,), expected in cases:
            actual = _eprop.error_generator_composition(_LSE('S', [stim.PauliString(P)]),
                                                        _LSE('H', [stim.PauliString(Q)]))
            self._assert_terms_equal(actual, expected, f'golden S_{P} o H_{Q}')

    def test_SH_block_exhaustive_1q(self):
        self._run_block('S', 'H', _ref_compose_SH, 1, expected_lengths={2})

    def test_SH_block_exhaustive_2q(self):
        self._run_block('S', 'H', _ref_compose_SH, 2, expected_lengths={2})

    def test_HS_and_SH_are_distinct(self):
        """H_P[S_Q] and S_P[H_Q] must not be conflated (they differ in both terms)."""
        P, Q = stim.PauliString('IX'), stim.PauliString('XI')
        hs = _eprop.error_generator_composition(_LSE('H', [P]), _LSE('S', [Q]))
        sh = _eprop.error_generator_composition(_LSE('S', [P]), _LSE('H', [Q]))
        self.assertNotEqual([lbl for lbl, _ in hs], [lbl for lbl, _ in sh])


@unittest.skipIf(stim is None, "stim is not installed")
class ErrgenCompositionSSBlockTester(BaseCase):
    """Term-level tests for the S,S dispatch block."""

    WEIGHTS = (1.0, 2.5, -1.0, 1j, 2.0 - 3.0j)

    def _assert_terms_equal(self, actual, expected, context):
        self.assertEqual(len(actual), len(expected),
                         f"{context}: expected {len(expected)} terms, got {len(actual)}. "
                         f"actual={actual} expected={expected}")
        for idx, ((act_lbl, act_rate), (exp_lbl, exp_rate)) in enumerate(zip(actual, expected)):
            self.assertEqual(act_lbl, exp_lbl,
                             f"{context}: term {idx} label mismatch: {act_lbl} != {exp_lbl}")
            self.assertAlmostEqual(act_rate, exp_rate, places=12,
                                   msg=f"{context}: term {idx} ({act_lbl}) rate {act_rate} != {exp_rate}")

    def test_SS_block_golden_cases(self):
        cases = [
            # P == Q: PQ is the identity, so the S_{PQ} term is dropped -> -2*S_X
            ('X', 'X', [(_LSE('S', [stim.PauliString('X')]), -1.0),
                        (_LSE('S', [stim.PauliString('X')]), -1.0)]),
            ('X', 'Y', [(_LSE('S', [stim.PauliString('Z')]), 1.0),
                        (_LSE('S', [stim.PauliString('X')]), -1.0),
                        (_LSE('S', [stim.PauliString('Y')]), -1.0)]),
            ('IX', 'XI', [(_LSE('S', [stim.PauliString('XX')]), 1.0),
                          (_LSE('S', [stim.PauliString('IX')]), -1.0),
                          (_LSE('S', [stim.PauliString('XI')]), -1.0)]),
        ]
        for P, Q, expected in cases:
            actual = _eprop.error_generator_composition(_LSE('S', [stim.PauliString(P)]),
                                                        _LSE('S', [stim.PauliString(Q)]))
            self._assert_terms_equal(actual, expected, f'golden S_{P} o S_{Q}')

    def _run_SS(self, num_qubits):
        paulis = _non_identity_paulis(num_qubits)
        identity = stim.PauliString('I' * num_qubits)
        basis = CompleteElementaryErrorgenBasis('PP', QubitSpace(num_qubits), default_label_type='local')
        errorgen_matrix_dict = {lbl: mat for lbl, mat in zip(basis.labels, basis.elemgen_matrices)}
        observed_lengths = set()

        for P, Q in product(paulis, repeat=2):
            lbl_1 = _LSE('S', [stim.PauliString(P)])
            lbl_2 = _LSE('S', [stim.PauliString(Q)])
            for weight in self.WEIGHTS:
                actual = _eprop.error_generator_composition(lbl_1, lbl_2, weight=weight, identity=identity)
                expected = _ref_terms_to_lse(_ref_compose_SS(P, Q, weight))
                self._assert_terms_equal(actual, expected,
                                         f'S_{P} o S_{Q} (weight={weight}, {num_qubits}Q)')
                observed_lengths.add(len(actual))

            numeric = _eprop.error_generator_composition_numerical(LEEL('S', (P,)), LEEL('S', (Q,)),
                                                                   errorgen_matrix_dict)
            analytic = _eprop.error_generator_composition(lbl_1, lbl_2, identity=identity)
            analytic_mat = _eprop.errorgen_layer_to_matrix(analytic, num_qubits,
                                                           errorgen_matrix_dict=errorgen_matrix_dict)
            norm_diff = np.linalg.norm(numeric - analytic_mat)
            self.assertLess(norm_diff, 1e-10,
                            f'S_{P} o S_{Q} ({num_qubits}Q): numeric vs analytic differ by {norm_diff}')

        # 2 terms when P == Q (the S_{PQ} term vanishes), 3 otherwise.
        self.assertEqual(observed_lengths, {2, 3},
                         f'S,S ({num_qubits}Q): unexpected output lengths {sorted(observed_lengths)}')

    def test_SS_block_exhaustive_1q(self):
        self._run_SS(1)

    def test_SS_block_exhaustive_2q(self):
        self._run_SS(2)


class _CompositionOracleMixin:
    """
    Shared machinery for blocks whose operands are one single-Pauli generator
    (H or S) and one Pauli-pair generator (C or A).

    For these blocks there is no compact closed form worth re-deriving by hand, so
    the expected result is produced by `_string_oracle_composition`: the two
    generators are expanded in the Pauli "sandwich" basis, composed with pure
    string arithmetic, and decomposed back onto the elementary error generator
    basis. That decomposition is unique and depends only on the generator
    definitions, never on `error_generator_composition`, so it independently pins
    every emitted label and rate.

    The string oracle is used rather than a dense dual-basis projection because its
    cost is independent of 4^n, which keeps 3-qubit coverage tractable.

    Because the oracle is order- and duplicate-agnostic, ordering and duplication
    behavior is covered separately by pinned golden cases and structural tests in
    each concrete test class.
    """

    WEIGHTS = (1.0, -2.0, 1j)

    def _assert_dicts_close(self, actual, expected, context):
        for key in set(actual) | set(expected):
            a, e = actual.get(key, 0), expected.get(key, 0)
            self.assertAlmostEqual(a, e, places=8,
                                   msg=f'{context}: rate for {key} was {a}, expected {e}')

    def _assert_labels_well_formed(self, terms, context):
        for lbl, _ in terms:
            bels = lbl.basis_element_labels
            self.assertIn(lbl.errorgen_type, ('H', 'S', 'C', 'A'))
            self.assertEqual(len(bels), 1 if lbl.errorgen_type in ('H', 'S') else 2)
            if lbl.errorgen_type in ('C', 'A'):
                self.assertNotEqual(bels[0], bels[1])
                self.assertTrue(_eprop.stim_pauli_string_less_than(bels[0], bels[1]),
                                f'{context}: non-canonical bel order in {lbl}')

    def _make_operands(self, single_type, pair_type, pair_first, A, P, Q):
        """Build the (LSE, LSE) and (LEEL, LEEL) operand pairs in dispatch order."""
        single_lse = _LSE(single_type, [stim.PauliString(A)])
        pair_lse = _LSE(pair_type, [stim.PauliString(P), stim.PauliString(Q)])
        single_leel = LEEL(single_type, (A,))
        pair_leel = LEEL(pair_type, (P, Q))
        if pair_first:
            return (pair_lse, single_lse), (pair_leel, single_leel)
        return (single_lse, pair_lse), (single_leel, pair_leel)

    def _run_block_against_oracle(self, num_qubits, single_type, pair_type, pair_first,
                                  expected_lengths):
        """Exhaustively validate one dispatch block against the string oracle."""
        identity = stim.PauliString('I' * num_qubits)
        observed_lengths = set()
        num_checked = 0
        block = (f'{pair_type},{single_type}' if pair_first else f'{single_type},{pair_type}')

        for A in _non_identity_paulis(num_qubits):
            for P, Q in _c_type_pauli_pairs(num_qubits):
                (lse_1, lse_2), (leel_1, leel_2) = self._make_operands(
                    single_type, pair_type, pair_first, A, P, Q)
                key_1 = (leel_1.errorgen_type, tuple(leel_1.basis_element_labels))
                key_2 = (leel_2.errorgen_type, tuple(leel_2.basis_element_labels))
                for weight in self.WEIGHTS:
                    actual = _eprop.error_generator_composition(lse_1, lse_2, weight=weight,
                                                                identity=identity)
                    context = f'{block}: A={A} P={P} Q={Q} (weight={weight}, {num_qubits}Q)'
                    expected = _string_oracle_composition(key_1, key_2, num_qubits, weight)
                    self._assert_dicts_close(_merge_terms(actual), expected, context)
                    self._assert_labels_well_formed(actual, context)
                    observed_lengths.add(len(actual))
                    num_checked += 1

        self.assertGreater(num_checked, 0)
        self.assertEqual(observed_lengths, expected_lengths,
                         f'{block} ({num_qubits}Q): observed lengths {sorted(observed_lengths)} '
                         f'!= expected {sorted(expected_lengths)}')

    def _assert_all_subcases_reachable(self, num_qubits=2):
        """The blocks branch on [P,Q] and then on ([A,P], [A,Q]): 8 sub-cases total."""
        seen = set()
        for A in _non_identity_paulis(num_qubits):
            for P, Q in _c_type_pauli_pairs(num_qubits):
                outer = 'commuting' if _ref_pauli_commutes(P, Q) else 'anticommuting'
                sub = {(True, True): 'a', (False, False): 'b',
                       (True, False): 'c', (False, True): 'd'}[(_ref_pauli_commutes(A, P),
                                                                _ref_pauli_commutes(A, Q))]
                seen.add(f'{outer}-{sub}')
        self.assertEqual(len(seen), 8, f'Only reached sub-cases {sorted(seen)}')


@unittest.skipIf(stim is None, "stim is not installed")
class ErrgenCompositionHCCHBlockTester(_CompositionOracleMixin, BaseCase):
    """Tests for the H,C and C,H dispatch blocks (lines 1423-1539 and 1919-2034)."""

    def test_HC_block_exhaustive_1q(self):
        self._run_block_against_oracle(1, 'H', 'C', pair_first=False, expected_lengths={2})

    def test_HC_block_exhaustive_2q(self):
        self._run_block_against_oracle(2, 'H', 'C', pair_first=False, expected_lengths={0, 2, 3, 4})

    def test_CH_block_exhaustive_1q(self):
        self._run_block_against_oracle(1, 'H', 'C', pair_first=True, expected_lengths={2})

    def test_CH_block_exhaustive_2q(self):
        self._run_block_against_oracle(2, 'H', 'C', pair_first=True, expected_lengths={0, 2, 3, 4})

    def test_HC_CH_subcase_coverage(self):
        self._assert_all_subcases_reachable()

    def test_HC_golden_cases(self):
        """Pinned ordered outputs, including the H-term that only cases 1a/1b emit."""
        A, P, Q = 'ZZ', 'IX', 'XI'
        expected = [(_LSE('C', [stim.PauliString('XI'), stim.PauliString('ZY')]), 1.0 + 0j),
                    (_LSE('C', [stim.PauliString('IX'), stim.PauliString('YZ')]), 1.0 + 0j),
                    (_LSE('A', [stim.PauliString('XX'), stim.PauliString('ZZ')]), -1.0),
                    (_LSE('H', [stim.PauliString('YY')]), 1.0 + 0j)]
        actual = _eprop.error_generator_composition(
            _LSE('H', [stim.PauliString(A)]),
            _LSE('C', [stim.PauliString(P), stim.PauliString(Q)]))
        self.assertEqual(len(actual), 4)
        for (a_lbl, a_rate), (e_lbl, e_rate) in zip(actual, expected):
            self.assertEqual(a_lbl, e_lbl)
            self.assertAlmostEqual(a_rate, e_rate, places=12)

    def test_CH_differs_from_HC_only_by_C_slot_sign(self):
        """
        C,H emits exactly the same labels in the same order as H,C; the terms coming
        from the two C-type slots are negated while the A-slot and Hamiltonian terms
        are unchanged.

        Note the C-type slots emit an 'S' label (not 'C') whenever their two Paulis
        coincide, so the sign flip is keyed on `type in ('C', 'S')`. Keying it on
        'C' alone is wrong for 60 of the 1575 two-qubit triples, e.g.
        H_IZ[C_{IX,IY}] = 2*S_IY - 2*S_IX  vs  C_{IX,IY}[H_IZ] = -2*S_IY + 2*S_IX.
        """
        identity = stim.PauliString('II')
        num_checked = 0
        saw_flipped_S = False
        for A in _non_identity_paulis(2):
            h_lbl = _LSE('H', [stim.PauliString(A)])
            for P, Q in _c_type_pauli_pairs(2):
                c_lbl = _LSE('C', [stim.PauliString(P), stim.PauliString(Q)])
                hc = _eprop.error_generator_composition(h_lbl, c_lbl, identity=identity)
                ch = _eprop.error_generator_composition(c_lbl, h_lbl, identity=identity)
                context = f'A={A} P={P} Q={Q}'
                self.assertEqual([l for l, _ in hc], [l for l, _ in ch],
                                 f'{context}: H,C and C,H emitted different label sequences')
                for (lbl, hc_rate), (_, ch_rate) in zip(hc, ch):
                    if lbl.errorgen_type in ('C', 'S'):
                        self.assertAlmostEqual(ch_rate, -hc_rate, places=12,
                                               msg=f'{context}: {lbl} should flip sign')
                        if lbl.errorgen_type == 'S':
                            saw_flipped_S = True
                    else:
                        self.assertAlmostEqual(ch_rate, hc_rate, places=12,
                                               msg=f'{context}: {lbl} should keep its sign')
                num_checked += 1
        self.assertGreater(num_checked, 0)
        self.assertTrue(saw_flipped_S,
                        'Expected to exercise at least one S-from-C-slot sign flip.')

    def test_HC_can_return_empty_list(self):
        """H_A[C_{P,Q}] genuinely vanishes for some inputs; callers must handle []."""
        actual = _eprop.error_generator_composition(
            _LSE('H', [stim.PauliString('XX')]),
            _LSE('C', [stim.PauliString('IX'), stim.PauliString('XI')]))
        self.assertEqual(actual, [])

    def test_HC_emits_unmerged_duplicate_terms(self):
        """H_IX[C_{IX,XI}] emits each of its two distinct labels twice (unmerged)."""
        actual = _eprop.error_generator_composition(
            _LSE('H', [stim.PauliString('IX')]),
            _LSE('C', [stim.PauliString('IX'), stim.PauliString('XI')]))
        self.assertEqual(len(actual), 4)
        self.assertEqual(len(_merge_terms(actual)), 2,
                         f'expected 2 distinct labels after merging, got {_merge_terms(actual)}')


@unittest.skipIf(stim is None, "stim is not installed")
class ErrgenCompositionHAAHBlockTester(_CompositionOracleMixin, BaseCase):
    """Tests for the H,A and A,H dispatch blocks (lines 1541-1655 and 4172-4286)."""

    def test_HA_block_exhaustive_1q(self):
        self._run_block_against_oracle(1, 'H', 'A', pair_first=False, expected_lengths={0, 2})

    def test_HA_block_exhaustive_2q(self):
        self._run_block_against_oracle(2, 'H', 'A', pair_first=False, expected_lengths={0, 1, 2, 3, 4})

    def test_AH_block_exhaustive_1q(self):
        self._run_block_against_oracle(1, 'H', 'A', pair_first=True, expected_lengths={0, 2})

    def test_AH_block_exhaustive_2q(self):
        self._run_block_against_oracle(2, 'H', 'A', pair_first=True, expected_lengths={0, 1, 2, 3, 4})

    def test_HA_AH_subcase_coverage(self):
        self._assert_all_subcases_reachable()

    def test_AH_matches_HA_labels_with_per_term_sign_only(self):
        """
        A,H emits exactly the same labels in the same order as H,A, and each rate is
        either identical or exactly negated -- never anything else.

        Deliberately *not* asserted: a rule predicting which terms flip from the
        emitted label type alone. No such rule exists here. The terms from the P*A
        and Q*A slots invert while the term from the P*Q slot does not, and both can
        surface as type 'A', so type 'A' is observed both flipping and not flipping.
        (Contrast S,C / C,S below, where a clean type-keyed rule does hold.)
        """
        identity = stim.PauliString('II')
        saw_flip = saw_same = False
        for A in _non_identity_paulis(2):
            h_lbl = _LSE('H', [stim.PauliString(A)])
            for P, Q in _c_type_pauli_pairs(2):
                a_lbl = _LSE('A', [stim.PauliString(P), stim.PauliString(Q)])
                ha = _eprop.error_generator_composition(h_lbl, a_lbl, identity=identity)
                ah = _eprop.error_generator_composition(a_lbl, h_lbl, identity=identity)
                context = f'A={A} P={P} Q={Q}'
                self.assertEqual([l for l, _ in ha], [l for l, _ in ah],
                                 f'{context}: H,A and A,H emitted different label sequences')
                for (lbl, ha_rate), (_, ah_rate) in zip(ha, ah):
                    if abs(ah_rate - ha_rate) < 1e-12:
                        saw_same = True
                    elif abs(ah_rate + ha_rate) < 1e-12:
                        saw_flip = True
                    else:
                        self.fail(f'{context}: {lbl} rate {ah_rate} is neither {ha_rate} nor its negation')
        self.assertTrue(saw_flip and saw_same,
                        'Expected to observe both sign-preserving and sign-flipping terms.')

    def test_HA_golden_cases(self):
        cases = [
            # anticommuting P,Q with A commuting both -> two A-type terms
            ('ZZ', 'IX', 'XI',
             [(_LSE('A', [stim.PauliString('XI'), stim.PauliString('ZY')]), -1.0 + 0j),
              (_LSE('A', [stim.PauliString('IX'), stim.PauliString('YZ')]), 1.0 + 0j)]),
            # mixed commutation -> one A term and one C term
            ('IZ', 'IX', 'XI',
             [(_LSE('A', [stim.PauliString('IY'), stim.PauliString('XI')]), 1.0 + 0j),
              (_LSE('C', [stim.PauliString('IX'), stim.PauliString('XZ')]), -1.0)]),
        ]
        for A, P, Q, expected in cases:
            actual = _eprop.error_generator_composition(
                _LSE('H', [stim.PauliString(A)]),
                _LSE('A', [stim.PauliString(P), stim.PauliString(Q)]))
            self.assertEqual(len(actual), len(expected), f'H_{A} o A_({P},{Q}): {actual}')
            for (a_lbl, a_rate), (e_lbl, e_rate) in zip(actual, expected):
                self.assertEqual(a_lbl, e_lbl)
                self.assertAlmostEqual(a_rate, e_rate, places=12)

    def test_HA_can_return_empty_list(self):
        """H_IZ[A_{IX,IY}] vanishes identically."""
        actual = _eprop.error_generator_composition(
            _LSE('H', [stim.PauliString('IZ')]),
            _LSE('A', [stim.PauliString('IX'), stim.PauliString('IY')]))
        self.assertEqual(actual, [])

    def test_HA_can_return_single_term(self):
        """H,A is the only block covered so far that can emit exactly one term."""
        actual = _eprop.error_generator_composition(
            _LSE('H', [stim.PauliString('IX')]),
            _LSE('A', [stim.PauliString('IX'), stim.PauliString('XI')]))
        self.assertEqual(len(actual), 1)
        self.assertEqual(actual[0][0],
                         _LSE('C', [stim.PauliString('IX'), stim.PauliString('XX')]))
        self.assertAlmostEqual(actual[0][1], -1.0, places=12)


@unittest.skipIf(stim is None, "stim is not installed")
class ErrgenCompositionSCCSBlockTester(_CompositionOracleMixin, BaseCase):
    """Tests for the S,C and C,S dispatch blocks (lines 1688-1799 and 2036-2146)."""

    def test_SC_block_exhaustive_1q(self):
        self._run_block_against_oracle(1, 'S', 'C', pair_first=False, expected_lengths={2})

    def test_SC_block_exhaustive_2q(self):
        self._run_block_against_oracle(2, 'S', 'C', pair_first=False, expected_lengths={2, 3})

    def test_CS_block_exhaustive_1q(self):
        self._run_block_against_oracle(1, 'S', 'C', pair_first=True, expected_lengths={2})

    def test_CS_block_exhaustive_2q(self):
        self._run_block_against_oracle(2, 'S', 'C', pair_first=True, expected_lengths={2, 3})

    def test_SC_CS_subcase_coverage(self):
        self._assert_all_subcases_reachable()

    def test_CS_differs_from_SC_by_sign_on_antisymmetric_terms(self):
        """
        C,S emits the same labels in the same order as S,C; the A-type and H-type
        terms are negated while the C-type terms are unchanged.

        Here the rule *is* keyable on the emitted type: the H terms arise only from
        the antisymmetric slots (via `_ordered_new_bels_A` returning 'H' when one
        product is the identity), and those slots are exactly the ones that flip.
        """
        identity = stim.PauliString('II')
        seen_types = set()
        for A in _non_identity_paulis(2):
            s_lbl = _LSE('S', [stim.PauliString(A)])
            for P, Q in _c_type_pauli_pairs(2):
                c_lbl = _LSE('C', [stim.PauliString(P), stim.PauliString(Q)])
                sc = _eprop.error_generator_composition(s_lbl, c_lbl, identity=identity)
                cs = _eprop.error_generator_composition(c_lbl, s_lbl, identity=identity)
                context = f'A={A} P={P} Q={Q}'
                self.assertEqual([l for l, _ in sc], [l for l, _ in cs],
                                 f'{context}: S,C and C,S emitted different label sequences')
                for (lbl, sc_rate), (_, cs_rate) in zip(sc, cs):
                    seen_types.add(lbl.errorgen_type)
                    if lbl.errorgen_type in ('A', 'H'):
                        self.assertAlmostEqual(cs_rate, -sc_rate, places=12,
                                               msg=f'{context}: {lbl} should flip sign')
                    else:
                        self.assertAlmostEqual(cs_rate, sc_rate, places=12,
                                               msg=f'{context}: {lbl} should keep its sign')
        self.assertTrue({'A', 'C'} <= seen_types,
                        f'Expected both A- and C-type terms to be exercised, saw {seen_types}')

    def test_SC_golden_cases(self):
        cases = [
            ('ZZ', 'IX', 'XI',
             [(_LSE('C', [stim.PauliString('YZ'), stim.PauliString('ZY')]), 1.0 + 0j),
              (_LSE('C', [stim.PauliString('YY'), stim.PauliString('ZZ')]), 1.0 + 0j),
              (_LSE('C', [stim.PauliString('IX'), stim.PauliString('XI')]), -1.0)]),
            ('IZ', 'IX', 'XI',
             [(_LSE('A', [stim.PauliString('IY'), stim.PauliString('XZ')]), 1.0 + 0j),
              (_LSE('A', [stim.PauliString('IZ'), stim.PauliString('XY')]), 1.0 + 0j),
              (_LSE('C', [stim.PauliString('IX'), stim.PauliString('XI')]), -1.0)]),
        ]
        for A, P, Q, expected in cases:
            actual = _eprop.error_generator_composition(
                _LSE('S', [stim.PauliString(A)]),
                _LSE('C', [stim.PauliString(P), stim.PauliString(Q)]))
            self.assertEqual(len(actual), len(expected), f'S_{A} o C_({P},{Q}): {actual}')
            for (a_lbl, a_rate), (e_lbl, e_rate) in zip(actual, expected):
                self.assertEqual(a_lbl, e_lbl)
                self.assertAlmostEqual(a_rate, e_rate, places=12)

    def test_SC_emits_unmerged_duplicate_terms(self):
        """S_IZ[C_{IX,IY}] emits C_{IX,IY} twice; merged it is -2*C_{IX,IY}."""
        actual = _eprop.error_generator_composition(
            _LSE('S', [stim.PauliString('IZ')]),
            _LSE('C', [stim.PauliString('IX'), stim.PauliString('IY')]))
        self.assertEqual(len(actual), 2)
        self.assertEqual(actual[0][0], actual[1][0])
        merged = _merge_terms(actual)
        self.assertEqual(len(merged), 1)
        self.assertAlmostEqual(next(iter(merged.values())), -2.0, places=12)

    def test_SC_never_emits_S_type_terms(self):
        """
        Every C-slot in S,C receives two distinct non-identity Paulis, so the block
        can never collapse to an S-type label (unlike H,C).
        """
        identity = stim.PauliString('II')
        for A in _non_identity_paulis(2):
            s_lbl = _LSE('S', [stim.PauliString(A)])
            for P, Q in _c_type_pauli_pairs(2):
                c_lbl = _LSE('C', [stim.PauliString(P), stim.PauliString(Q)])
                for lbl, _ in _eprop.error_generator_composition(s_lbl, c_lbl, identity=identity):
                    self.assertNotEqual(lbl.errorgen_type, 'S',
                                        f'unexpected S-type term for A={A} P={P} Q={Q}')


@unittest.skipIf(stim is None, "stim is not installed")
class ErrgenCompositionSAASBlockTester(_CompositionOracleMixin, BaseCase):
    """Tests for the S,A and A,S dispatch blocks (lines 1802-1916 and 4288-4402)."""

    def test_SA_block_exhaustive_1q(self):
        self._run_block_against_oracle(1, 'S', 'A', pair_first=False, expected_lengths={2})

    def test_SA_block_exhaustive_2q(self):
        self._run_block_against_oracle(2, 'S', 'A', pair_first=False, expected_lengths={2, 3})

    def test_AS_block_exhaustive_1q(self):
        self._run_block_against_oracle(1, 'S', 'A', pair_first=True, expected_lengths={2})

    def test_AS_block_exhaustive_2q(self):
        self._run_block_against_oracle(2, 'S', 'A', pair_first=True, expected_lengths={2, 3})

    def test_SA_AS_subcase_coverage(self):
        self._assert_all_subcases_reachable()

    def test_AS_matches_SA_labels_with_per_term_sign_only(self):
        """
        A,S emits the same labels in the same order as S,A, and each rate is either
        identical or exactly negated.

        As with H,A / A,H -- and unlike S,C / C,S -- there is deliberately no
        type-keyed sign rule asserted here, because none exists. The sign flip
        tracks the *sub-case* (it happens only in the mixed-commutation cases c and
        d, and there only for the P*A/Q*A and A*P*Q slots), not the emitted label
        type. Both 'A' and 'C' terms are observed flipping in some inputs and not
        flipping in others.
        """
        identity = stim.PauliString('II')
        saw_flip = saw_same = False
        for A in _non_identity_paulis(2):
            s_lbl = _LSE('S', [stim.PauliString(A)])
            for P, Q in _c_type_pauli_pairs(2):
                a_lbl = _LSE('A', [stim.PauliString(P), stim.PauliString(Q)])
                sa = _eprop.error_generator_composition(s_lbl, a_lbl, identity=identity)
                as_ = _eprop.error_generator_composition(a_lbl, s_lbl, identity=identity)
                context = f'A={A} P={P} Q={Q}'
                self.assertEqual([l for l, _ in sa], [l for l, _ in as_],
                                 f'{context}: S,A and A,S emitted different label sequences')
                for (lbl, sa_rate), (_, as_rate) in zip(sa, as_):
                    if abs(as_rate - sa_rate) < 1e-12:
                        saw_same = True
                    elif abs(as_rate + sa_rate) < 1e-12:
                        saw_flip = True
                    else:
                        self.fail(f'{context}: {lbl} rate {as_rate} is neither {sa_rate} nor its negation')
        self.assertTrue(saw_flip and saw_same,
                        'Expected to observe both sign-preserving and sign-flipping terms.')

    def test_SA_golden_cases(self):
        cases = [
            # A commutes with both P and Q -> A-type term, then the trailing A_(P,Q)
            ('ZZ', 'IX', 'XI',
             [(_LSE('A', [stim.PauliString('YZ'), stim.PauliString('ZY')]), -1.0 + 0j),
              (_LSE('A', [stim.PauliString('IX'), stim.PauliString('XI')]), -1.0)]),
            # mixed commutation -> C-type term, then the trailing A_(P,Q)
            ('IZ', 'IX', 'XI',
             [(_LSE('C', [stim.PauliString('IY'), stim.PauliString('XZ')]), -1.0 + 0j),
              (_LSE('A', [stim.PauliString('IX'), stim.PauliString('XI')]), -1.0)]),
        ]
        for A, P, Q, expected in cases:
            actual = _eprop.error_generator_composition(
                _LSE('S', [stim.PauliString(A)]),
                _LSE('A', [stim.PauliString(P), stim.PauliString(Q)]))
            self.assertEqual(len(actual), len(expected), f'S_{A} o A_({P},{Q}): {actual}')
            for (a_lbl, a_rate), (e_lbl, e_rate) in zip(actual, expected):
                self.assertEqual(a_lbl, e_lbl)
                self.assertAlmostEqual(a_rate, e_rate, places=12)

    def test_SA_always_emits_the_input_A_term(self):
        """
        Every S,A output contains an A_(P,Q) term echoing the input pair: the
        trailing `_ordered_new_bels_A(P, Q, False, False, False)` slot is
        unconditional and can never vanish for a valid A-type input label.
        """
        identity = stim.PauliString('II')
        for A in _non_identity_paulis(2):
            s_lbl = _LSE('S', [stim.PauliString(A)])
            for P, Q in _c_type_pauli_pairs(2):
                expected = _LSE('A', [stim.PauliString(P), stim.PauliString(Q)])
                actual = _eprop.error_generator_composition(
                    s_lbl, _LSE('A', [stim.PauliString(P), stim.PauliString(Q)]),
                    identity=identity)
                self.assertIn(expected, [lbl for lbl, _ in actual],
                              f'A={A} P={P} Q={Q}: missing the A_(P,Q) term in {actual}')

    def test_SA_never_returns_empty_or_single_term(self):
        """
        S,A always emits 2 or 3 terms. Contrast H,A, which can emit 0 or 1 -- worth
        pinning because the two blocks are otherwise structurally similar.
        """
        identity = stim.PauliString('II')
        for A in _non_identity_paulis(2):
            s_lbl = _LSE('S', [stim.PauliString(A)])
            for P, Q in _c_type_pauli_pairs(2):
                actual = _eprop.error_generator_composition(
                    s_lbl, _LSE('A', [stim.PauliString(P), stim.PauliString(Q)]),
                    identity=identity)
                self.assertIn(len(actual), (2, 3), f'A={A} P={P} Q={Q}: got {actual}')

    def test_SA_emits_unmerged_duplicate_terms(self):
        """S_IX[A_{IX,IY}] emits A_{IX,IY} twice; merged it is -2*A_{IX,IY}."""
        actual = _eprop.error_generator_composition(
            _LSE('S', [stim.PauliString('IX')]),
            _LSE('A', [stim.PauliString('IX'), stim.PauliString('IY')]))
        self.assertEqual(len(actual), 2)
        self.assertEqual(actual[0][0], actual[1][0])
        merged = _merge_terms(actual)
        self.assertEqual(len(merged), 1)
        self.assertAlmostEqual(next(iter(merged.values())), -2.0, places=12)

    def test_SA_can_emit_terms_that_cancel_exactly(self):
        """
        S_IZ[A_{IX,IY}] returns two terms that sum to zero. The composition vanishes
        but the returned list is *not* empty, so callers must not treat a non-empty
        result as a non-zero generator.
        """
        actual = _eprop.error_generator_composition(
            _LSE('S', [stim.PauliString('IZ')]),
            _LSE('A', [stim.PauliString('IX'), stim.PauliString('IY')]))
        self.assertEqual(len(actual), 2)
        self.assertEqual(actual[0][0], actual[1][0])
        self.assertAlmostEqual(actual[0][1] + actual[1][1], 0.0, places=12)
        self.assertEqual(_merge_terms(actual), {})

    def test_SA_single_qubit_emits_only_A_type_terms(self):
        """On one qubit the S,A block degenerates: every emitted term is A-type."""
        identity = stim.PauliString('I')
        seen = set()
        for A in _non_identity_paulis(1):
            s_lbl = _LSE('S', [stim.PauliString(A)])
            for P, Q in _c_type_pauli_pairs(1):
                actual = _eprop.error_generator_composition(
                    s_lbl, _LSE('A', [stim.PauliString(P), stim.PauliString(Q)]),
                    identity=identity)
                seen.update(lbl.errorgen_type for lbl, _ in actual)
        self.assertEqual(seen, {'A'}, f'expected only A-type terms on 1 qubit, saw {seen}')


class ErrgenCompositionStringOracleTester(BaseCase):
    """
    Regression guard for `_string_oracle_composition` itself.

    The string oracle is the sole verifier for every block whose operands include a
    C- or A-type generator, so a silent defect in it would make those tests check
    against a wrong reference while still passing. This pins it against the
    independently hand-derived closed forms used by the H/S block testers
    (`_ref_compose_HH` / `_HS` / `_SH` / `_SS`), which are themselves derived from
    the Lindblad definitions rather than from the sandwich representation.

    Runs at every weight the block testers use, so the oracle's weight scaling --
    which the closed forms exercise independently -- is covered too. No dense
    superoperators are involved, so this is fast; it needs no stim.
    """

    WEIGHTS = (1.0, -2.0, 1j)

    CLOSED_FORMS = ((('H', 'H'), _ref_compose_HH),
                    (('H', 'S'), _ref_compose_HS),
                    (('S', 'H'), _ref_compose_SH),
                    (('S', 'S'), _ref_compose_SS))

    def _expected(self, ref_fn, P, Q, weight):
        """Closed-form reference as a {LEEL: rate} dict with duplicates summed."""
        merged = {}
        for (egtype, bels), rate in ref_fn(P, Q, weight):
            key = LEEL(egtype, tuple(bels))
            merged[key] = merged.get(key, 0) + rate
        return {k: v for k, v in merged.items() if abs(v) > 1e-12}

    def _check(self, num_qubits):
        checked = 0
        for P in _non_identity_paulis(num_qubits):
            for Q in _non_identity_paulis(num_qubits):
                for (type_1, type_2), ref_fn in self.CLOSED_FORMS:
                    for weight in self.WEIGHTS:
                        expected = self._expected(ref_fn, P, Q, weight)
                        actual = _string_oracle_composition((type_1, (P,)), (type_2, (Q,)),
                                                            num_qubits, weight)
                        context = (f'{type_1}_{P} o {type_2}_{Q} '
                                   f'(weight={weight}, {num_qubits}Q)')
                        for key in set(actual) | set(expected):
                            a, e = actual.get(key, 0), expected.get(key, 0)
                            self.assertAlmostEqual(
                                a, e, places=10,
                                msg=f'{context}: rate for {key} was {a} from the string '
                                    f'oracle, {e} from the closed form')
                        checked += 1
        return checked

    def test_string_oracle_matches_closed_forms_1q(self):
        self.assertEqual(self._check(1), 3 * 3 * 4 * 3)

    def test_string_oracle_matches_closed_forms_2q(self):
        self.assertEqual(self._check(2), 15 * 15 * 4 * 3)

    # The closed forms above only take H- and S-type *inputs*, so they never
    # exercise the C or A branches of `_sandwich_expansion` -- the very branches the
    # C/A block testers depend on. The remaining tests anchor those branches without
    # needing any dense superoperators.

    def test_sandwich_round_trip_recovers_every_generator(self):
        """
        Expanding a single elementary generator and decomposing it again must return
        that generator with unit rate, for all four types.

        This pins `_sandwich_expansion` and `_decompose_sandwich` as mutual inverses
        across C and A as well as H and S.
        """
        for num_qubits in (1, 2):
            basis = CompleteElementaryErrorgenBasis('PP', QubitSpace(num_qubits),
                                                    default_label_type='local')
            seen_types = set()
            for lbl in basis.labels:
                egtype = lbl.errorgen_type
                bels = tuple(lbl.basis_element_labels)
                seen_types.add(egtype)
                expansion = _sandwich_expansion(egtype, bels, num_qubits)
                recovered = _decompose_sandwich(expansion, num_qubits)
                self.assertEqual(set(recovered), {(egtype, bels)},
                                 f'{num_qubits}Q {lbl}: round trip produced {recovered}')
                self.assertAlmostEqual(recovered[(egtype, bels)], 1.0, places=12,
                                       msg=f'{num_qubits}Q {lbl}: rate was not 1')
            self.assertEqual(seen_types, {'H', 'S', 'C', 'A'})

    def test_sandwich_round_trip_is_linear(self):
        """A weighted sum of generators must decompose back to the same weights."""
        num_qubits = 2
        basis = CompleteElementaryErrorgenBasis('PP', QubitSpace(num_qubits),
                                                default_label_type='local')
        labels = list(basis.labels)
        chosen = [labels[i] for i in (0, 20, 100, 200)]
        weights = (2.0, -1.5, 1j, 0.25)
        combined = {}
        for lbl, weight in zip(chosen, weights):
            for key, coeff in _sandwich_expansion(lbl.errorgen_type,
                                                  tuple(lbl.basis_element_labels),
                                                  num_qubits).items():
                combined[key] = combined.get(key, 0) + weight * coeff
        recovered = _decompose_sandwich(combined, num_qubits)
        expected = {(lbl.errorgen_type, tuple(lbl.basis_element_labels)): w
                    for lbl, w in zip(chosen, weights)}
        self.assertEqual(set(recovered), set(expected))
        for key, want in expected.items():
            self.assertAlmostEqual(recovered[key], want, places=12, msg=f'{key}')

    def test_C_expansion_with_repeated_pauli_is_twice_S(self):
        """
        C_{P,P} = 2 S_P follows directly from the definition, so it anchors the C
        branch of the expansion against the S branch (which the closed forms cover).
        """
        for num_qubits in (1, 2):
            for P in _non_identity_paulis(num_qubits):
                c_exp = _sandwich_expansion('C', (P, P), num_qubits)
                s_exp = _sandwich_expansion('S', (P,), num_qubits)
                doubled = {k: 2 * v for k, v in s_exp.items()}
                self.assertEqual(set(c_exp), set(doubled), f'C_({P},{P}) vs 2*S_{P}')
                for key in doubled:
                    self.assertAlmostEqual(c_exp[key], doubled[key], places=12,
                                           msg=f'C_({P},{P}) vs 2*S_{P} at {key}')

    def test_A_expansion_with_repeated_pauli_vanishes(self):
        """A_{P,P} = 0 identically, anchoring the antisymmetric branch."""
        for num_qubits in (1, 2):
            for P in _non_identity_paulis(num_qubits):
                self.assertEqual(_sandwich_expansion('A', (P, P), num_qubits), {},
                                 f'A_({P},{P}) should vanish')

    def test_weight_is_applied_by_the_string_oracle(self):
        """
        The block testers call the oracle at several weights, so a dropped or
        mis-applied weight would corrupt their expectations at every weight but 1.
        """
        # distinct Pauli pairs for the two operands: reusing the same pair makes some
        # of these compositions vanish, leaving nothing to scale.
        for (type_1, type_2) in (('H', 'C'), ('C', 'H'), ('S', 'A'),
                                 ('A', 'C'), ('C', 'C'), ('A', 'A')):
            key_1 = ((type_1, ('IX', 'XI')) if type_1 in ('C', 'A') else (type_1, ('IZ',)))
            key_2 = ((type_2, ('IY', 'ZZ')) if type_2 in ('C', 'A') else (type_2, ('IZ',)))
            base = _string_oracle_composition(key_1, key_2, 2)
            self.assertGreater(len(base), 0, f'{type_1},{type_2} produced nothing to scale')
            for weight in (2.5, -1.0, 1j):
                scaled = _string_oracle_composition(key_1, key_2, 2, weight=weight)
                self.assertEqual(set(scaled), set(base))
                for key, rate in base.items():
                    self.assertAlmostEqual(scaled[key], weight * rate, places=10,
                                           msg=f'{type_1},{type_2} at weight {weight}: {key}')


@unittest.skipIf(stim is None, "stim is not installed")
class ErrgenCompositionPairPairBlockTester(BaseCase):
    """
    Tests for the four blocks where *both* operands are Pauli-pair generators:
    C,C, C,A, A,C and A,A. Together these are ~81% of `error_generator_composition`
    and ~97% of its static branch paths.

    Two things make these blocks different from the ones covered above:

    * They branch on six commutation bits -- [A,B], [P,Q] and the four cross
      relations -- giving 64 structural sub-cases. 62 are reachable on two qubits;
      the remaining two require three, so 2-qubit coverage alone leaves a gap.
    * They can emit up to 8 terms, versus at most 4 elsewhere.

    Correctness is checked against `_string_oracle_composition`. Per-key
    `assertAlmostEqual` would dominate the runtime at this scale, so discrepancies
    are accumulated and asserted once.
    """

    EXHAUSTIVE_WEIGHTS = (1.0, 1j)
    SAMPLED_WEIGHTS = (1.0, -2.0, 1j)
    MAX_REPORTED_FAILURES = 10

    def _sweep(self, num_qubits, type_1, type_2, quadruples, weights):
        """Compare the function against the string oracle over the given inputs."""
        identity = stim.PauliString('I' * num_qubits)
        max_terms = _composition_block_max_terms()[(type_1, type_2)]
        failures, lengths, signatures = [], set(), set()
        checked = 0

        for A, B, P, Q in quadruples:
            lse_1 = _LSE(type_1, [stim.PauliString(A), stim.PauliString(B)])
            lse_2 = _LSE(type_2, [stim.PauliString(P), stim.PauliString(Q)])
            key_1, key_2 = (type_1, (A, B)), (type_2, (P, Q))
            signatures.add(_commutation_signature(A, B, P, Q))
            label = f'{type_1}_({A},{B}) o {type_2}_({P},{Q})'

            for weight in weights:
                actual = _eprop.error_generator_composition(lse_1, lse_2, weight=weight,
                                                            identity=identity)
                expected = _string_oracle_composition(key_1, key_2, num_qubits, weight)
                merged = _merge_terms(actual)
                for key in set(merged) | set(expected):
                    got, want = merged.get(key, 0), expected.get(key, 0)
                    if abs(got - want) > 1e-8:
                        failures.append(f'{label} w={weight}: {key} was {got}, oracle says {want}')

                if len(actual) > max_terms:
                    failures.append(f'{label} w={weight}: {len(actual)} terms exceeds the '
                                    f'static bound of {max_terms}')
                for lbl, _ in actual:
                    bels = lbl.basis_element_labels
                    if lbl.errorgen_type in ('C', 'A'):
                        if (len(bels) != 2 or bels[0] == bels[1]
                                or not _eprop.stim_pauli_string_less_than(bels[0], bels[1])):
                            failures.append(f'{label} w={weight}: non-canonical label {lbl}')
                    elif len(bels) != 1:
                        failures.append(f'{label} w={weight}: malformed label {lbl}')

                lengths.add(len(actual))
                checked += 1

        return failures, lengths, signatures, checked

    def _assert_clean(self, failures, context):
        if failures:
            shown = '\n  '.join(failures[:self.MAX_REPORTED_FAILURES])
            self.fail(f'{context}: {len(failures)} discrepancies vs the string oracle:\n  {shown}')

    def _run_exhaustive_2q(self, type_1, type_2):
        pair_labels = _c_type_pauli_pairs(2)
        quads = ((A, B, P, Q) for A, B in pair_labels for P, Q in pair_labels)
        failures, lengths, signatures, checked = self._sweep(
            2, type_1, type_2, quads, self.EXHAUSTIVE_WEIGHTS)
        self._assert_clean(failures, f'{type_1},{type_2} 2Q exhaustive')
        self.assertEqual(checked, len(pair_labels) ** 2 * len(self.EXHAUSTIVE_WEIGHTS))
        self.assertEqual(len(signatures), 62,
                         f'{type_1},{type_2}: expected 62 reachable 2-qubit signatures, '
                         f'got {len(signatures)}')
        self.assertEqual(lengths, self.EXPECTED_LENGTHS_2Q[(type_1, type_2)],
                         f'{type_1},{type_2} (2Q): observed lengths {sorted(lengths)} != '
                         f'expected {sorted(self.EXPECTED_LENGTHS_2Q[(type_1, type_2)])}')
        return lengths

    def _run_signature_reps_3q(self, type_1, type_2):
        reps = _signature_representatives(3)
        self.assertEqual(len(reps), 64, f'expected all 64 signatures at 3 qubits, got {len(reps)}')
        failures, lengths, signatures, checked = self._sweep(
            3, type_1, type_2, list(reps.values()), self.SAMPLED_WEIGHTS)
        self._assert_clean(failures, f'{type_1},{type_2} 3Q signature representatives')
        self.assertEqual(len(signatures), 64)
        self.assertEqual(checked, 64 * len(self.SAMPLED_WEIGHTS))
        self.assertEqual(lengths, self.EXPECTED_LENGTHS_2Q[(type_1, type_2)],
                         f'{type_1},{type_2} (3Q reps): observed lengths {sorted(lengths)} != '
                         f'expected {sorted(self.EXPECTED_LENGTHS_2Q[(type_1, type_2)])}')
        return lengths

    def _errorgen_matrix_dict(self, num_qubits):
        basis = CompleteElementaryErrorgenBasis('PP', QubitSpace(num_qubits), default_label_type='local')
        return {lbl: mat for lbl, mat in zip(basis.labels, basis.elemgen_matrices)}

    # Exact sets of output list lengths, per block and qubit count. Obtained by
    # interpreting each dispatch block's AST against a pure-Python reimplementation of
    # `_ordered_new_bels_A`/`_C` over every operand pair. Term *count* cannot depend on
    # Pauli ordering (no block calls `stim_pauli_string_less_than` directly, and
    # `_ordered_new_bels_*` returns non-None regardless of which order it picks), so
    # these are exact rather than approximate.
    EXPECTED_LENGTHS_1Q = {('C', 'C'): {1, 2}, ('C', 'A'): {0, 2},
                           ('A', 'C'): {0, 2}, ('A', 'A'): {6, 8}}
    EXPECTED_LENGTHS_2Q = {k: {0, 1, 2, 3, 4, 6, 7, 8}
                           for k in (('C', 'C'), ('C', 'A'), ('A', 'C'), ('A', 'A'))}

    def _run_exhaustive_1q(self, type_1, type_2):
        """
        One qubit admits only three Pauli pairs, so this is 9 operand combinations --
        cheap, but it is where the degenerate paths live (products collapsing to the
        identity, PA == QA, and so on) that are comparatively rare at two qubits.
        """
        pair_labels = _c_type_pauli_pairs(1)
        self.assertEqual(len(pair_labels), 3, f'expected 3 one-qubit pairs, got {pair_labels}')
        quads = [(A, B, P, Q) for A, B in pair_labels for P, Q in pair_labels]
        failures, lengths, _, checked = self._sweep(1, type_1, type_2, quads,
                                                    self.EXHAUSTIVE_WEIGHTS)
        self._assert_clean(failures, f'{type_1},{type_2} 1Q exhaustive')
        self.assertEqual(checked, 9 * len(self.EXHAUSTIVE_WEIGHTS))
        self.assertEqual(lengths, self.EXPECTED_LENGTHS_1Q[(type_1, type_2)],
                         f'{type_1},{type_2} (1Q): observed lengths {sorted(lengths)} != '
                         f'expected {sorted(self.EXPECTED_LENGTHS_1Q[(type_1, type_2)])}')

    def _run_against_dense_oracle(self, type_1, type_2, stride):
        """
        Cross-check against `error_generator_composition_numerical`, which builds the
        dense superoperators and multiplies them.

        The other block testers all have a check of this kind; the pair-pair blocks
        previously relied solely on `_string_oracle_composition`, so a fault in the
        oracle would have gone unnoticed in exactly the four blocks that carry ~96% of
        the function's remaining complexity. Dense products cost ~4^n per call, so this
        samples rather than sweeping exhaustively -- the string oracle handles breadth.
        """
        num_qubits = 2
        identity = stim.PauliString('I' * num_qubits)
        matrix_dict = self._errorgen_matrix_dict(num_qubits)
        pair_labels = _c_type_pauli_pairs(num_qubits)
        worst, worst_case, checked = 0.0, None, 0

        for L1 in pair_labels[::stride]:
            for L2 in pair_labels[::stride]:
                lse_1 = _LSE(type_1, [stim.PauliString(s) for s in L1])
                lse_2 = _LSE(type_2, [stim.PauliString(s) for s in L2])
                analytic = _eprop.error_generator_composition(lse_1, lse_2, identity=identity)
                analytic_mat = _eprop.errorgen_layer_to_matrix(
                    analytic, num_qubits, errorgen_matrix_dict=matrix_dict)
                numeric = _eprop.error_generator_composition_numerical(
                    LEEL(type_1, L1), LEEL(type_2, L2), matrix_dict)
                diff = np.linalg.norm(numeric - analytic_mat)
                if diff > worst:
                    worst, worst_case = diff, f'{type_1}_{L1} o {type_2}_{L2}'
                checked += 1

        self.assertGreater(checked, 100,
                           f'{type_1},{type_2}: only {checked} dense comparisons; stride too coarse')
        self.assertLess(worst, 1e-10,
                        f'{type_1},{type_2}: worst dense-vs-analytic disagreement {worst} '
                        f'at {worst_case}')

    # ---------------------------------------------------------------- C,C

    def test_CC_exhaustive_1q(self):
        self._run_exhaustive_1q('C', 'C')

    def test_CC_exhaustive_2q(self):
        self._run_exhaustive_2q('C', 'C')

    def test_CC_signature_representatives_3q(self):
        self._run_signature_reps_3q('C', 'C')

    def test_all_pair_pair_blocks_reach_eight_terms_and_never_emit_five(self):
        """
        Two facts about the pair-pair output lengths, both worth pinning.

        First, all four blocks attain the eight-term maximum at two qubits, so the
        AST-derived bound is tight rather than merely an upper limit -- previously
        only C,C had this confirmed.

        Second, and less obvious: length 5 is never produced by any of the four, even
        though 0,1,2,3,4,6,7,8 all occur. The widest branches emit their terms in
        groups, so once a composition is large enough to need more than four terms it
        jumps to at least six. A refactor that flattened those groups would likely
        start producing 5 and trip this test.
        """
        expected = {0, 1, 2, 3, 4, 6, 7, 8}
        for block in (('C', 'C'), ('C', 'A'), ('A', 'C'), ('A', 'A')):
            self.assertEqual(self.EXPECTED_LENGTHS_2Q[block], expected)
            self.assertIn(8, self.EXPECTED_LENGTHS_2Q[block],
                          f'{block} no longer reaches its eight-term maximum')
            self.assertNotIn(5, self.EXPECTED_LENGTHS_2Q[block],
                             f'{block} now emits five terms')
            self.assertEqual(max(self.EXPECTED_LENGTHS_2Q[block]),
                             _composition_block_max_terms()[block],
                             f'{block}: observed maximum disagrees with the static bound')

    def test_CC_reaches_the_eight_term_maximum(self):
        """
        C_{IX,XI}[C_{IX,YI}] lands in the branch that emits all eight terms
        (com_AB and com_PQ true, com_AP/com_AQ/com_BP true, com_BQ false), with none
        of the seven `is not None` guards firing and the trailing Hamiltonian term
        present. This pins the widest path in the function as genuinely reachable.
        """
        actual = _eprop.error_generator_composition(
            _LSE('C', [stim.PauliString('IX'), stim.PauliString('XI')]),
            _LSE('C', [stim.PauliString('IX'), stim.PauliString('YI')]),
            identity=stim.PauliString('II'))
        self.assertEqual(len(actual), 8, f'expected 8 terms, got {actual}')
        self.assertEqual(_composition_block_max_terms()[('C', 'C')], 8)

    # ---------------------------------------------------------------- A,A

    def test_AA_exhaustive_1q(self):
        self._run_exhaustive_1q('A', 'A')

    def test_AA_exhaustive_2q(self):
        self._run_exhaustive_2q('A', 'A')

    def test_AA_signature_representatives_3q(self):
        self._run_signature_reps_3q('A', 'A')

    # ---------------------------------------------------------------- C,A

    def test_CA_exhaustive_1q(self):
        self._run_exhaustive_1q('C', 'A')

    def test_CA_exhaustive_2q(self):
        self._run_exhaustive_2q('C', 'A')

    def test_CA_signature_representatives_3q(self):
        self._run_signature_reps_3q('C', 'A')

    # ---------------------------------------------------------------- A,C

    def test_AC_exhaustive_1q(self):
        self._run_exhaustive_1q('A', 'C')

    def test_AC_exhaustive_2q(self):
        self._run_exhaustive_2q('A', 'C')

    def test_AC_signature_representatives_3q(self):
        self._run_signature_reps_3q('A', 'C')

    # --------------------------------------- independent dense cross-checks

    def test_CC_matches_dense_numerical_oracle(self):
        self._run_against_dense_oracle('C', 'C', stride=9)

    def test_AA_matches_dense_numerical_oracle(self):
        self._run_against_dense_oracle('A', 'A', stride=9)

    def test_CA_matches_dense_numerical_oracle(self):
        self._run_against_dense_oracle('C', 'A', stride=9)

    def test_AC_matches_dense_numerical_oracle(self):
        self._run_against_dense_oracle('A', 'C', stride=9)

    def _ca_ac_pair(self, L1, L2):
        """Merged, zero-pruned results of C_{L1}[A_{L2}] and A_{L2}[C_{L1}]."""
        identity = stim.PauliString('II')
        c_lbl = _LSE('C', [stim.PauliString(s) for s in L1])
        a_lbl = _LSE('A', [stim.PauliString(s) for s in L2])
        prune = lambda d: {k: v for k, v in d.items() if abs(v) > 1e-12}
        return (prune(_merge_terms(_eprop.error_generator_composition(c_lbl, a_lbl,
                                                                      identity=identity))),
                prune(_merge_terms(_eprop.error_generator_composition(a_lbl, c_lbl,
                                                                      identity=identity))))

    def test_CA_and_AC_agree_on_labels_only_when_both_are_nonzero(self):
        """
        The other ordered pairs (H/C, H/A, S/C, S/A) satisfy "same labels, rates equal
        or negated", which is what makes a shared helper with a single sign flag
        natural. C,A and A,C are weaker, and the exact shape of the weakening matters
        for anyone planning to unify them.

        Sweeping all 11,025 two-qubit operand pairs through the oracle gives:

            both orderings empty                 735
            only C,A non-empty                  1200
            only A,C non-empty                   360
            both non-empty, same label set      8730
            both non-empty, different label set    0

        So the two orderings never disagree about *which* labels appear while both are
        alive -- every difference is one ordering vanishing outright, and it vanishes
        in both directions. Where both survive the rates are related by +1 or -1, but
        with no rule keyed on error generator type (unlike H,C/C,H, where the C- and
        S-type terms flip). This test pins one example of each vanishing direction.
        """
        ca, ac = self._ca_ac_pair(('IX', 'IY'), ('IX', 'IZ'))
        self.assertTrue(ca, 'expected C_{IX,IY}[A_{IX,IZ}] to be non-zero')
        self.assertFalse(ac, f'expected A_{{IX,IZ}}[C_{{IX,IY}}] to vanish, got {ac}')

        ca, ac = self._ca_ac_pair(('IX', 'XI'), ('IY', 'XX'))
        self.assertFalse(ca, f'expected C_{{IX,XI}}[A_{{IY,XX}}] to vanish, got {ca}')
        self.assertTrue(ac, 'expected A_{IY,XX}[C_{IX,XI}] to be non-zero')

    def test_CA_and_AC_share_labels_up_to_sign_when_both_survive(self):
        """
        The complement of the case above: when neither ordering vanishes they agree on
        labels, and each shared rate is equal or negated. Checked over a spread of
        operand pairs rather than exhaustively, since the exhaustive oracle-agreement
        sweeps already cover correctness; this is about the cross-block relationship.
        """
        pair_labels = _c_type_pauli_pairs(2)
        compared = 0
        for L1 in pair_labels[::7]:
            for L2 in pair_labels[::5]:
                ca, ac = self._ca_ac_pair(L1, L2)
                if not ca or not ac:
                    continue
                compared += 1
                self.assertEqual(set(ca), set(ac),
                                 f'C_{L1}[A_{L2}] and A_{L2}[C_{L1}] disagree on labels')
                for key in ca:
                    self.assertAlmostEqual(abs(ca[key]), abs(ac[key]), places=8,
                                           msg=f'C_{L1}[A_{L2}] vs A_{L2}[C_{L1}]: |rate| '
                                               f'differs for {key}')
        self.assertGreater(compared, 100,
                           f'only {compared} operand pairs had both orderings non-zero; '
                           'the sampling stride needs revisiting')

    # ------------------------------------------------- cross-cutting checks

    def test_two_signatures_require_three_qubits(self):
        """
        Documents why the 3-qubit sweeps above exist: these two sub-cases cannot be
        reached with two qubits, so 2-qubit coverage alone would leave the
        corresponding branches of every pair-pair block untested.
        """
        two = set(_signature_representatives(2))
        three = set(_signature_representatives(3))
        self.assertEqual(len(two), 62)
        self.assertEqual(len(three), 64)
        self.assertEqual(three - two,
                         {(False, True, True, True, True, True),
                          (True, False, True, True, True, True)})

    def test_pair_pair_blocks_dominate_the_function(self):
        """Guard the assumption motivating this class; fails loudly if refactoring shifts it."""
        bounds = _composition_block_max_terms()
        for block in (('C', 'C'), ('C', 'A'), ('A', 'C'), ('A', 'A')):
            self.assertEqual(bounds[block], 8, f'{block} no longer emits up to 8 terms')
        for block, bound in bounds.items():
            if block not in (('C', 'C'), ('C', 'A'), ('A', 'C'), ('A', 'A')):
                self.assertLessEqual(bound, 4, f'{block} unexpectedly emits {bound} terms')


@unittest.skipIf(stim is None, "stim is not installed")
class ErrgenCompositionDeadPathTester(BaseCase):
    """
    Pins the composition paths that coverage reports as never taken, by proving they
    are *unreachable* rather than merely untested.

    A guard that can never fire is not a coverage gap to be closed with a cleverer
    input; it is either dead code or an invariant worth stating. These tests assert
    the invariants, so if a future change makes one of these paths live, the test
    fails and points at the guard rather than leaving it silently uncovered.
    """

    def test_ABPQ_is_never_the_identity_when_the_parity_guard_passes(self):
        """
        All four pair-pair blocks end with

            if sum((com_AP, com_AQ, com_BP, com_BQ)) % 2 == 1:
                ABPQ = ...
                if ABPQ[1] != identity:   # <- False branch never taken

        The inner guard is dead, and provably so. Writing x_UV = 1 when U and V
        anticommute, x is additive in each slot, so x_{A,PQ} = x_AP + x_AQ and
        x_{B,PQ} = x_BP + x_BQ (mod 2). If ABPQ == I then PQ == AB up to phase, and
        since x_{A,A} = 0 we get x_{A,PQ} = x_{A,AB} = x_AB, likewise for B. Hence

            x_AP + x_AQ + x_BP + x_BQ = x_AB + x_AB = 0  (mod 2),

        i.e. an *even* number of the four cross relations anticommute, so an even
        number commute. The outer parity guard therefore excludes ABPQ == I.

        Verified exhaustively here for 1-3 qubits across all four family conditions.
        """
        for num_qubits in (1, 2, 3):
            identity = stim.PauliString('I' * num_qubits)
            pair_labels = _c_type_pauli_pairs(num_qubits)
            checked = 0
            for A, B in pair_labels:
                sA, sB = stim.PauliString(A), stim.PauliString(B)
                AB = _eprop.pauli_product(sA, sB)
                for P, Q in pair_labels:
                    sP, sQ = stim.PauliString(P), stim.PauliString(Q)
                    coms = (sA.commutes(sP), sA.commutes(sQ),
                            sB.commutes(sP), sB.commutes(sQ))
                    if sum(coms) % 2 != 1:
                        continue
                    PQ = _eprop.pauli_product(sP, sQ)
                    ABPQ = _eprop.pauli_product(AB[0] * AB[1], PQ[0] * PQ[1])
                    self.assertNotEqual(
                        ABPQ[1], identity,
                        f'{num_qubits}Q: A={A} B={B} P={P} Q={Q} has odd commuting parity '
                        f'but A*B*P*Q == I, so the guard in the pair-pair blocks is live '
                        f'after all')
                    checked += 1
            self.assertGreater(checked, 0,
                               f'{num_qubits}Q: parity guard never passed; nothing checked')

    def test_ordered_new_bels_A_double_identity_branch_is_unreachable(self):
        """
        `_ordered_new_bels_A` returns (None, None, None) for `pauli_eq`, and again for
        `first_pauli_ident and second_pauli_ident`. The second is unreachable from the
        composition helpers: both Paulis being the identity makes them equal, so the
        `pauli_eq` guard fires first.

        (`_ordered_new_bels_C` folds the two identity cases into one test and has no
        such orphan branch.)
        """
        ident = stim.PauliString('II')
        # Reaching the branch at all requires lying about pauli_eq.
        self.assertEqual(_eprop._ordered_new_bels_A(ident, ident, True, True, False),
                         (None, None, None))
        # ...and with the flags computed honestly, pauli_eq short-circuits first.
        self.assertEqual(_eprop._ordered_new_bels_A(ident, ident, True, True, True),
                         (None, None, None))

        # Every call site derives the identity flags and pauli_eq from the same pair of
        # Paulis, so `ident_1 and ident_2` implies `pauli_eq`.
        for P in (stim.PauliString('II'), stim.PauliString('XI'), stim.PauliString('IZ')):
            for Q in (stim.PauliString('II'), stim.PauliString('XI')):
                ident_1, ident_2, pauli_eq = (P == ident), (Q == ident), (P == Q)
                if ident_1 and ident_2:
                    self.assertTrue(pauli_eq,
                                    'both-identity should imply pauli_eq for every caller')

    def test_impl_builds_its_own_identity_when_not_supplied(self):
        """
        `_error_generator_composition_impl` has an `identity is None` fallback that the
        caching entry point never exercises, since it always passes a cached identity.
        Call the implementation directly to pin that the fallback agrees.
        """
        l1 = _LSE('C', [stim.PauliString('XI'), stim.PauliString('IZ')])
        l2 = _LSE('A', [stim.PauliString('IX'), stim.PauliString('ZI')])
        without = _eprop._error_generator_composition_impl(l1, l2, 1, None)
        with_ident = _eprop._error_generator_composition_impl(l1, l2, 1,
                                                              stim.PauliString('II'))
        self.assertEqual(without, with_ident)
        self.assertTrue(without, 'expected a non-empty composition for this pair')

    def test_impl_returns_empty_for_an_unrecognized_errorgen_type(self):
        """
        The trailing `return composed_errorgens` is reachable only for an error
        generator type outside {H, S, C, A}. Pin the current (silent, empty) behavior
        so that a future decision to raise instead is a deliberate, visible change.
        """
        bogus = _LSE('H', [stim.PauliString('XI')])
        bogus.errorgen_type = 'Z'   # not a real elementary error generator type
        good = _LSE('H', [stim.PauliString('IZ')])
        self.assertEqual(
            _eprop._error_generator_composition_impl(bogus, good, 1, stim.PauliString('II')),
            [])
        self.assertEqual(
            _eprop._error_generator_composition_impl(good, bogus, 1, stim.PauliString('II')),
            [])


@unittest.skipIf(stim is None, "stim is not installed")
class ErrgenCompositionOracleHelperTester(BaseCase):
    """
    Tests for `errorgen_layer_to_matrix` and `error_generator_composition_numerical`.

    These two are the numerical oracle the composition tests are checked against, yet
    they were the least covered code on the composition surface (53% and 56%). A fault
    in either would weaken every dense cross-check in this file while looking like a
    passing suite, so they deserve direct tests rather than being exercised only
    incidentally.
    """

    NQ = 2

    def setUp(self):
        self.basis = CompleteElementaryErrorgenBasis('PP', QubitSpace(self.NQ),
                                                     default_label_type='local')
        self.local_dict = {lbl: mat for lbl, mat
                           in zip(self.basis.labels, self.basis.elemgen_matrices)}
        self.sslbls = tuple(range(self.NQ))
        self.global_dict = {GEEL.cast(lbl, sslbls=self.sslbls): mat
                            for lbl, mat in self.local_dict.items()}
        self.lse_dict = {_LSE.cast(lbl): mat for lbl, mat in self.local_dict.items()}
        self.leel = LEEL('H', ('XI',))
        self.expected = self.local_dict[self.leel]

    # ------------------------------------------------ errorgen_layer_to_matrix

    def test_layer_to_matrix_empty_layer_is_all_zeros(self):
        for empty in ([], (), {}):
            mat = _eprop.errorgen_layer_to_matrix(empty, self.NQ,
                                                  errorgen_matrix_dict=self.local_dict)
            self.assertEqual(mat.shape, (4**self.NQ, 4**self.NQ))
            self.assertEqual(np.linalg.norm(mat), 0.0)

    def test_layer_to_matrix_builds_its_own_basis_when_dict_is_omitted(self):
        """The `errorgen_matrix_dict is None` path must agree with the explicit dict."""
        layer = [(self.leel, 1.0)]
        self.assertLess(np.linalg.norm(_eprop.errorgen_layer_to_matrix(layer, self.NQ)
                                       - self.expected), 1e-12)

    def test_layer_to_matrix_accepts_list_tuple_and_dict_layers(self):
        as_list = [(self.leel, 2.0)]
        for layer in (as_list, tuple(as_list), dict(as_list)):
            mat = _eprop.errorgen_layer_to_matrix(layer, self.NQ,
                                                  errorgen_matrix_dict=self.local_dict)
            self.assertLess(np.linalg.norm(mat - 2.0*self.expected), 1e-12,
                            f'layer supplied as {type(layer).__name__} disagreed')

    def test_layer_to_matrix_all_label_and_dict_type_combinations(self):
        """
        Nine of these combinations are legal: {LSE, LEEL, GEEL} labels crossed with
        {local, global} dict keys (plus LSE against an LSE-keyed dict). Each is a
        separate branch and all must produce the same matrix.
        """
        lse = _LSE.cast(self.leel)
        geel = GEEL.cast(self.leel, sslbls=self.sslbls)
        cases = [
            ('LSE  / local ', [(lse, 1.0)], self.local_dict, None),
            ('LSE  / global', [(lse, 1.0)], self.global_dict, None),
            ('LEEL / local ', [(self.leel, 1.0)], self.local_dict, None),
            ('LEEL / global', [(self.leel, 1.0)], self.global_dict, self.sslbls),
            ('GEEL / local ', [(geel, 1.0)], self.local_dict, self.sslbls),
            ('GEEL / global', [(geel, 1.0)], self.global_dict, self.sslbls),
        ]
        for name, layer, matrix_dict, sslbls in cases:
            mat = _eprop.errorgen_layer_to_matrix(layer, self.NQ,
                                                  errorgen_matrix_dict=matrix_dict,
                                                  sslbls=sslbls)
            self.assertLess(np.linalg.norm(mat - self.expected), 1e-12,
                            f'{name}: disagreed with the reference matrix')

    def test_layer_to_matrix_requires_sslbls_when_label_kinds_are_mixed(self):
        """Local labels against a global dict (and vice versa) need `sslbls`."""
        geel = GEEL.cast(self.leel, sslbls=self.sslbls)
        with self.assertRaises(ValueError):
            _eprop.errorgen_layer_to_matrix([(self.leel, 1.0)], self.NQ,
                                            errorgen_matrix_dict=self.global_dict)
        with self.assertRaises(ValueError):
            _eprop.errorgen_layer_to_matrix([(geel, 1.0)], self.NQ,
                                            errorgen_matrix_dict=self.local_dict)

    def test_layer_to_matrix_rejects_unsupported_dict_key_type(self):
        """
        An unsupported `errorgen_matrix_dict` key must raise *with* its message. This
        previously raised a bare `ValueError()`, discarding the explanation it had just
        built -- a bug no test caught because the branch was never executed.
        """
        bad_dict = {'H(XI)': self.expected}
        with self.assertRaises(ValueError) as ctx:
            _eprop.errorgen_layer_to_matrix([(self.leel, 1.0)], self.NQ,
                                            errorgen_matrix_dict=bad_dict)
        self.assertIn('is not supported as a key', str(ctx.exception))

    def test_layer_to_matrix_rejects_empty_dict_for_a_nonempty_layer(self):
        """
        An empty dict is distinct from `None`: `None` means "build the basis for me",
        whereas `{}` is an explicit dict with nothing in it and cannot be used.
        """
        with self.assertRaises(ValueError) as ctx:
            _eprop.errorgen_layer_to_matrix([(self.leel, 1.0)], self.NQ,
                                            errorgen_matrix_dict={})
        self.assertIn('errorgen_matrix_dict is empty', str(ctx.exception))

    def test_layer_to_matrix_rejects_a_layer_that_is_not_a_list_tuple_or_dict(self):
        with self.assertRaises(ValueError) as ctx:
            _eprop.errorgen_layer_to_matrix(iter([(self.leel, 1.0)]), self.NQ,
                                            errorgen_matrix_dict=self.local_dict)
        self.assertIn('should be either a list, tuple or dict', str(ctx.exception))

    def test_layer_to_matrix_rejects_unsupported_coefficient_label_type(self):
        with self.assertRaises(ValueError) as ctx:
            _eprop.errorgen_layer_to_matrix([('H(XI)', 1.0)], self.NQ,
                                            errorgen_matrix_dict=self.local_dict)
        self.assertIn('should be either', str(ctx.exception))

    def test_layer_to_matrix_is_linear_in_the_rates(self):
        """Guards the accumulation loop, which every dense cross-check depends on."""
        other = LEEL('S', ('IZ',))
        combined = _eprop.errorgen_layer_to_matrix(
            [(self.leel, 1.5), (other, -2.0)], self.NQ,
            errorgen_matrix_dict=self.local_dict)
        separate = (1.5*self.local_dict[self.leel] - 2.0*self.local_dict[other])
        self.assertLess(np.linalg.norm(combined - separate), 1e-12)

    def test_layer_to_matrix_sums_duplicate_labels_rather_than_overwriting(self):
        """
        The composition routines return *unmerged* duplicate terms, so the oracle must
        accumulate them. If it overwrote instead, several block tests would silently
        compare against the wrong matrix.
        """
        mat = _eprop.errorgen_layer_to_matrix([(self.leel, 1.0), (self.leel, 1.0)],
                                              self.NQ,
                                              errorgen_matrix_dict=self.local_dict)
        self.assertLess(np.linalg.norm(mat - 2.0*self.expected), 1e-12)

    # --------------------------------- error_generator_composition_numerical

    def test_composition_numerical_all_label_and_dict_type_combinations(self):
        l1, l2 = LEEL('H', ('XI',)), LEEL('S', ('IZ',))
        reference = self.local_dict[l1] @ self.local_dict[l2]
        cases = [
            ('LEEL / LEEL-keyed', l1, l2, self.local_dict),
            ('LSE  / LEEL-keyed', _LSE.cast(l1), _LSE.cast(l2), self.local_dict),
            ('LSE  / LSE-keyed ', _LSE.cast(l1), _LSE.cast(l2), self.lse_dict),
            ('LEEL / LSE-keyed ', l1, l2, self.lse_dict),
        ]
        for name, a, b, matrix_dict in cases:
            got = _eprop.error_generator_composition_numerical(a, b, matrix_dict)
            self.assertLess(np.linalg.norm(got - reference), 1e-12,
                            f'{name}: disagreed with the reference product')

    def test_composition_numerical_builds_its_own_basis_when_dict_is_omitted(self):
        l1, l2 = LEEL('H', ('XI',)), LEEL('S', ('IZ',))
        got = _eprop.error_generator_composition_numerical(l1, l2, None, num_qubits=self.NQ)
        reference = self.local_dict[l1] @ self.local_dict[l2]
        self.assertLess(np.linalg.norm(got - reference), 1e-12)

    def test_composition_numerical_rejects_mismatched_or_unsupported_label_types(self):
        l1 = LEEL('H', ('XI',))
        with self.assertRaises(AssertionError):
            _eprop.error_generator_composition_numerical(l1, _LSE.cast(l1), self.local_dict)
        with self.assertRaises(AssertionError):
            _eprop.error_generator_composition_numerical('H(XI)', l1, self.local_dict)

    def test_composition_numerical_is_not_symmetric(self):
        """
        A sanity check on argument order, which the block tests rely on. H(XI) and
        H(YI) are chosen because their superoperators genuinely fail to commute; many
        elementary pairs (e.g. H(XI) with C(IX,IZ)) act on disjoint qubits and would
        make this check vacuous.
        """
        l1, l2 = LEEL('H', ('XI',)), LEEL('H', ('YI',))
        forward = _eprop.error_generator_composition_numerical(l1, l2, self.local_dict)
        backward = _eprop.error_generator_composition_numerical(l2, l1, self.local_dict)
        self.assertGreater(np.linalg.norm(forward - backward), 1e-10)

    # ------------------------------ iterative_error_generator_composition

    def test_iterative_composition_with_a_single_label_is_the_identity(self):
        """The `len(errorgen_labels) == 1` early return was never exercised."""
        lbl = _LSE('H', [stim.PauliString('XI')])
        self.assertEqual(_eprop.iterative_error_generator_composition((lbl,), (2.5,)),
                         [(lbl, 2.5)])

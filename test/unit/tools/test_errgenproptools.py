import numpy as np
from scipy.linalg import logm
from pygsti.baseobjs import Label, QubitSpace, BuiltinBasis
from pygsti.baseobjs.errorgenbasis import CompleteElementaryErrorgenBasis
from pygsti.algorithms.randomcircuit import create_random_circuit
from pygsti.models.modelconstruction import create_crosstalk_free_model
from pygsti.baseobjs.errorgenlabel import LocalElementaryErrorgenLabel as LEEL
from pygsti.errorgenpropagation.localstimerrorgen import LocalStimErrorgenLabel as _LSE
from pygsti.tools import errgenproptools as _eprop
from pygsti.tools.matrixtools import print_mx
from pygsti.tools.basistools import change_basis
from ..util import BaseCase
from itertools import product
import random
import stim
from pygsti.processors import QubitProcessorSpec
from pygsti.errorgenpropagation.errorpropagator import ErrorGeneratorPropagator

#TODO: errorgen_layer_to_matrix, stim_pauli_string_less_than 

class ErrgenCompositionCommutationTester(BaseCase):

    def setUp(self):
        num_qubits = 4
        gate_names = ['Gcphase', 'Gxpi2', 'Gypi2']
        availability = {'Gcphase':[(0,1), (1,2), (2,3), (3,0)]}
        pspec = QubitProcessorSpec(num_qubits, gate_names, availability=availability)
        self.target_model = create_crosstalk_free_model(processor_spec = pspec)
        self.circuit = create_random_circuit(pspec, 4, sampler='edgegrab', samplerargs=[0.4,], rand_state=12345)
        max_strengths = {1: {'S': 0, 'H': .0001},
                         2: {'S': 0, 'H': .0001}}
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
        errorgen_lbls_3Q, errorgen_mats_3Q = select_random_items_from_multiple_lists([complete_errorgen_basis_3Q.labels, complete_errorgen_basis_3Q.elemgen_matrices], 1000, seed= 1234)
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
            try:
                analytic_composition_mat = _eprop.errorgen_layer_to_matrix(analytic_composition, 2, errorgen_matrix_dict = errorgen_lbl_matrix_dict_2Q)        
            except KeyError:
                print(f'{analytic_composition=}')
            norm_diff = np.linalg.norm(numeric_composition-analytic_composition_mat)
            if norm_diff > 1e-10:
                print(f'Difference in compositions for pair {pair1} is greater than 1e-10.')
                print(f'{np.linalg.norm(numeric_composition-analytic_composition_mat)=}')
                print('numeric_composition=')
                print_mx(numeric_composition)
                
                #Decompose the numerical composition into rates.
                for lbl, dual in zip(complete_errorgen_basis_2Q.labels, complete_errorgen_basis_2Q.elemgen_dual_matrices):
                    rate = np.trace(dual.conj().T@numeric_composition)
                    if abs(rate) >1e-3:
                        print(f'{lbl}: {rate}')
                
                print(f'{analytic_composition=}')
                print('analytic_composition_mat=')
                print_mx(analytic_composition_mat)
                raise ValueError('Numeric and analytic error generator compositions were not found to be identical!')

        for pair1, pair2 in zip(errorgen_label_pairs_3Q, stim_errorgen_label_pairs_3Q):
            numeric_composition = _eprop.error_generator_composition_numerical(pair1[0], pair1[1], errorgen_lbl_matrix_dict_3Q)
            analytic_composition = _eprop.error_generator_composition(pair2[0], pair2[1])
            try:
                analytic_composition_mat = _eprop.errorgen_layer_to_matrix(analytic_composition, 3, errorgen_matrix_dict = complete_errorgen_lbl_matrix_dict_3Q)        
            except KeyError:
                print(f'{analytic_composition=}')
            norm_diff = np.linalg.norm(numeric_composition-analytic_composition_mat)
            if norm_diff > 1e-10:
                print(f'Difference in compositions for pair {pair1} is greater than 1e-10.')
                print(f'{np.linalg.norm(numeric_composition-analytic_composition_mat)=}')
                print('numeric_composition=')
                print_mx(numeric_composition)
                
                #Decompose the numerical composition into rates.
                for lbl, dual in zip(complete_errorgen_basis_3Q.labels, complete_errorgen_basis_3Q.elemgen_dual_matrices):
                    rate = np.trace(dual.conj().T@numeric_composition)
                    if abs(rate) >1e-3:
                        print(f'{lbl}: {rate}')
                
                print(f'{analytic_composition=}')
                print('analytic_composition_mat=')
                print_mx(analytic_composition_mat)
                raise ValueError('Numeric and analytic error generator compositions were not found to be identical!')    
    
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
        propagated_errorgen_layers_bch_order_1 = self.errorgen_propagator.propagate_errorgens_bch(self.circuit, bch_order=1)
        first_order_bch_analytical = self.errorgen_propagator.errorgen_layer_dict_to_errorgen(propagated_errorgen_layers_bch_order_1,mx_basis='pp', return_dense=True)
        assert np.linalg.norm(first_order_bch_analytical-first_order_bch_numerical) < 1e-14
        
        propagated_errorgen_layers_bch_order_2 = self.errorgen_propagator.propagate_errorgens_bch(self.circuit, bch_order=2)
        second_order_bch_numerical = _eprop.bch_numerical(self.propagated_errorgen_layers, self.errorgen_propagator, bch_order=2)
        second_order_bch_analytical = self.errorgen_propagator.errorgen_layer_dict_to_errorgen(propagated_errorgen_layers_bch_order_2, mx_basis='pp', return_dense=True)
        assert np.linalg.norm(second_order_bch_analytical-second_order_bch_numerical) < 1e-14

        third_order_bch_numerical = _eprop.bch_numerical(self.propagated_errorgen_layers, self.errorgen_propagator, bch_order=3)
        propagated_errorgen_layers_bch_order_3 = self.errorgen_propagator.propagate_errorgens_bch(self.circuit, bch_order=3)
        third_order_bch_analytical = self.errorgen_propagator.errorgen_layer_dict_to_errorgen(propagated_errorgen_layers_bch_order_3, mx_basis='pp', return_dense=True)
        assert np.linalg.norm(third_order_bch_analytical-third_order_bch_numerical) < 1e-14

        fourth_order_bch_numerical = _eprop.bch_numerical(self.propagated_errorgen_layers, self.errorgen_propagator, bch_order=4)
        propagated_errorgen_layers_bch_order_4 = self.errorgen_propagator.propagate_errorgens_bch(self.circuit, bch_order=4)
        fourth_order_bch_analytical = self.errorgen_propagator.errorgen_layer_dict_to_errorgen(propagated_errorgen_layers_bch_order_4, mx_basis='pp', return_dense=True)
        assert np.linalg.norm(fourth_order_bch_analytical-fourth_order_bch_numerical) < 1e-14

        fifth_order_bch_numerical = _eprop.bch_numerical(self.propagated_errorgen_layers, self.errorgen_propagator, bch_order=5)
        propagated_errorgen_layers_bch_order_5 = self.errorgen_propagator.propagate_errorgens_bch(self.circuit, bch_order=5, truncation_threshold=0)
        fifth_order_bch_analytical = self.errorgen_propagator.errorgen_layer_dict_to_errorgen(propagated_errorgen_layers_bch_order_5, mx_basis='pp', return_dense=True)
        assert np.linalg.norm(fifth_order_bch_analytical-fifth_order_bch_numerical) < 1e-14

        exact_errorgen = logm(self.errorgen_propagator.eoc_error_channel(self.circuit))
        exact_vs_first_order_norm  = np.linalg.norm(first_order_bch_analytical-exact_errorgen)
        exact_vs_second_order_norm = np.linalg.norm(second_order_bch_analytical-exact_errorgen)
        exact_vs_third_order_norm  = np.linalg.norm(third_order_bch_analytical-exact_errorgen)
        exact_vs_fourth_order_norm = np.linalg.norm(fourth_order_bch_analytical-exact_errorgen)
        exact_vs_fifth_order_norm  = np.linalg.norm(fifth_order_bch_analytical-exact_errorgen)
        
        self.assertTrue((exact_vs_first_order_norm > exact_vs_second_order_norm) and (exact_vs_second_order_norm > exact_vs_third_order_norm)
                        and (exact_vs_third_order_norm > exact_vs_fourth_order_norm) and (exact_vs_fourth_order_norm > exact_vs_fifth_order_norm))
        
class ApproxStabilizerMethodTester(BaseCase):
    def setUp(self):
        num_qubits = 4
        gate_names = ['Gcphase', 'Gxpi2', 'Gypi2']
        availability = {'Gcphase':[(0,1), (1,2), (2,3), (3,0)]}
        pspec = QubitProcessorSpec(num_qubits, gate_names, availability=availability)
        self.target_model = create_crosstalk_free_model(processor_spec = pspec)
        self.circuit = create_random_circuit(pspec, 4, sampler='edgegrab', samplerargs=[0.4,], rand_state=12345)
        self.circuit_alt = create_random_circuit(pspec, 4, sampler='edgegrab', samplerargs=[0.4,], rand_state=12345)
        max_strengths = {1: {'S': .0005, 'H': .0001},
                         2: {'S': .0005, 'H': .0001}}
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
            np.testing.assert_allclose(actual, expected, rtol=1e-5)

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
        amp0000 = _eprop.amplitude_of_state(self.circuit_tableau, '0000')
        amp1111 = _eprop.amplitude_of_state(self.circuit_tableau, '1111')
        self.assertTrue(abs(amp0000)<1e-7)
        self.assertTrue(abs(amp1111 -(-1j*np.sqrt(.125)))<1e-7)
        
        amp0000 = _eprop.amplitude_of_state(self.circuit_tableau_alt, '0000')
        amp1111 = _eprop.amplitude_of_state(self.circuit_tableau_alt, '1111')
        
        self.assertTrue(abs(amp0000)<1e-7)
        self.assertTrue(abs(amp1111 - (-1j*np.sqrt(.125)))<1e-7)

    def test_bitstring_to_tableau(self):
        tableau = _eprop.bitstring_to_tableau('1010')
        self.assertEqual(tableau, stim.PauliString('XIXI').to_tableau())

    def test_pauli_phase_update(self):
        test_paulis = ['YII', 'ZII', stim.PauliString('XYZ'), stim.PauliString('+iIII')]
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
            phase_update, output_bitstring = _eprop.pauli_phase_update(test_pauli, test_bitstring, dual=True)
            self.assertEqual(phase_update, correct_phase_updates_dual[i])
            self.assertEqual(output_bitstring, correct_output_bitstrings[i])

    def test_phi(self):
        bit_strings_3Q = list(product(['0','1'], repeat=3))
        for bit_string in bit_strings_3Q:
            for pauli_1, pauli_2 in product(stim.PauliString.iter_all(3), stim.PauliString.iter_all(3)):
                phi_num = _eprop.phi_numerical(self.circuit_tableau_3Q, bit_string, pauli_1, pauli_2)
                phi_analytic = _eprop.phi(self.circuit_tableau_3Q, bit_string, pauli_1, pauli_2)
                if abs(phi_num-phi_analytic) > 1e-4:
                    _eprop.phi(self.circuit_tableau_3Q, bit_string, pauli_1, pauli_2, debug=True)
                    raise ValueError(f'{pauli_1}, {pauli_2}, {bit_string}, {phi_num=}, {phi_analytic=}')
    
    def test_alpha(self):
        bit_strings_3Q = list(product(['0','1'], repeat=3))
        complete_errorgen_basis_3Q = CompleteElementaryErrorgenBasis('PP', QubitSpace(3), default_label_type='local')
        for bit_string in bit_strings_3Q:
            for lbl in complete_errorgen_basis_3Q.labels:
                alpha_num = _eprop.alpha_numerical(lbl, self.circuit_tableau_3Q, bit_string)
                assert abs(alpha_num - _eprop.alpha(lbl, self.circuit_tableau_3Q, bit_string)) <1e-4

    def test_alpha_pauli(self):
        from pygsti.modelpacks import smq2Q_XYCPHASE
        pspec_2Q = smq2Q_XYCPHASE.processor_spec()
        random_circuits_2Q = [create_random_circuit(pspec_2Q, 4, sampler='edgegrab', samplerargs=[0.4,], rand_state=12345+i) for i in range(5)]
        random_circuit_tableaus_2Q = [ckt.convert_to_stim_tableau() for ckt in random_circuits_2Q]
        def _compare_alpha_pauli_analytic_numeric(num_qubits, tableau):
            #loop through all error generators and all paulis
            errorgen_basis = CompleteElementaryErrorgenBasis('PP', QubitSpace(num_qubits), default_label_type='local')
            errorgen_labels = [_LSE.cast(lbl) for lbl in errorgen_basis.labels]
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
        third_order_diff = exact_prop_probs[-1] - _eprop.approximate_stabilizer_probability(self.propagated_errorgen_layer, self.circuit_tableau, '1111', order=3)

        assert abs(first_order_diff) > abs(second_order_diff)
        assert abs(second_order_diff) > abs(third_order_diff)
        
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
        #random_4Q_pauli_indices = rng.choice(len(paulis_4Q), 10, replace=False)
        #random_4Q_paulis = [paulis_4Q[idx] for idx in random_4Q_pauli_indices]

        for pauli in paulis_4Q:#random_4Q_paulis:
            exact_pauli_expectation = pauli_expectation_errorgen_prop(self.error_propagator, self.target_model, 
                                                                      self.circuit, pauli, use_bch=True, bch_order=1)
            first_order_diff  = exact_pauli_expectation - _eprop.approximate_stabilizer_pauli_expectation(self.propagated_errorgen_layer, self.circuit_tableau, pauli, order=1)
            second_order_diff = exact_pauli_expectation - _eprop.approximate_stabilizer_pauli_expectation(self.propagated_errorgen_layer, self.circuit_tableau, pauli, order=2)
            third_order_diff  = exact_pauli_expectation - _eprop.approximate_stabilizer_pauli_expectation(self.propagated_errorgen_layer, self.circuit_tableau, pauli, order=3)

            if abs(first_order_diff) < abs(second_order_diff):
                print(f'{first_order_diff=}')
                print(f'{second_order_diff=}')
                print(f'{pauli=}')
                raise ValueError('Going to higher order made the expectation value worse!')
            if abs(second_order_diff) < abs(third_order_diff):
                print(f'{second_order_diff=}')
                print(f'{third_order_diff=}')
                print(f'{pauli=}')
                raise ValueError('Going to higher order made the expectation value worse!')

    def test_error_generator_taylor_expansion(self):
        #this is just an integration test atm.
        _eprop.error_generator_taylor_expansion(self.propagated_errorgen_layer, order=2)

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
        prob_vec[i] = np.linalg.multi_dot([dense_effect.reshape((1,len(dense_effect))), eoc_channel, ideal_channel, dense_prep.reshape((len(dense_prep),1))])
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
        numeric_composition = _eprop.iterative_error_generator_composition_numeric(triple_1, (1,1,1), complete_errorgen_lbl_matrix_dict)
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

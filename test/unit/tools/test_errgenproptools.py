import numpy as np
from scipy.linalg import logm, expm
from pygsti.baseobjs import Label, QubitSpace, BuiltinBasis
from pygsti.baseobjs.errorgenbasis import CompleteElementaryErrorgenBasis
from pygsti.algorithms.randomcircuit import create_random_circuit
from pygsti.models.modelconstruction import create_crosstalk_free_model
from pygsti.baseobjs.errorgenlabel import LocalElementaryErrorgenLabel as LEEL
from pygsti.errorgenpropagation.localstimerrorgen import LocalStimErrorgenLabel as _LSE, bel_less_than
from pygsti.tools import errgenproptools as _eprop
from pygsti.tools.matrixtools import print_mx
from pygsti.tools.basistools import change_basis
from pygsti.tools.lindbladtools import create_elementary_errorgen, random_CPTP_error_generator_rates
from pygsti.tools.exceptions import pyGSTiDeprecationWarning
import warnings
from ..util import BaseCase
from itertools import product, chain
import random
import stim
from pygsti.processors import QubitProcessorSpec
from pygsti.errorgenpropagation.errorpropagator import ErrorGeneratorPropagator

#TODO: errorgen_layer_to_matrix 

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

    def test_errorgen_commutator_S_A_degenerate_case(self):
        # Regression test for a bug in error_generator_commutator's 'S'-'A' branch: it referenced
        # an undefined local variable `ptup` (should have been `ptup1`) inside the
        # `if ptup1[1] == ptup2[1]:` branch (two occurrences), causing an UnboundLocalError. This
        # branch is only reached when, for single-qubit Paulis P (the S generator's own Pauli) and
        # the A generator's basis elements Q1,Q2, the (unsigned) products P*Q1 and Q2*P coincide --
        # which, since unsigned single-qubit Pauli multiplication is abelian and P is invertible,
        # requires Q1 == Q2. A well-formed A_{Q1,Q2} generator requires Q1 != Q2 (see "A Taxonomy of
        # Small Errors", Sec. V.D), so this is only reachable for a malformed/degenerate A(Q,Q)
        # label -- which nothing in LocalStimErrorgenLabel's constructor currently prevents one from
        # constructing. This test confirms the fix (the typo'd `ptup` -> `ptup1`) so this code path
        # no longer crashes with an UnboundLocalError.
        s_label = _LSE('S', [stim.PauliString('Z')])
        degenerate_a_label = _LSE('A', [stim.PauliString('X'), stim.PauliString('X')])
        # Should not raise UnboundLocalError.
        result = _eprop.error_generator_commutator(s_label, degenerate_a_label)
        self.assertIsInstance(result, list)

        # Also confirm well-formed (non-degenerate) S,A pairs still work correctly (regression
        # check against the numerical ground truth, since this exact pairing wasn't necessarily
        # hit by test_errorgen_commutators's 2-qubit complete-basis sweep above).
        errorgen_basis = CompleteElementaryErrorgenBasis('PP', QubitSpace(2), default_label_type='local')
        errorgen_lbl_matrix_dict = {lbl: mat for lbl, mat in zip(errorgen_basis.labels, errorgen_basis.elemgen_matrices)}
        s_lbl = LEEL('S', ('XI',))
        a_lbl = LEEL('A', ('IX', 'YY'))
        numeric_commutator = _eprop.error_generator_commutator_numerical(s_lbl, a_lbl, errorgen_lbl_matrix_dict)
        analytic_commutator = _eprop.error_generator_commutator(_LSE.cast(s_lbl), _LSE.cast(a_lbl))
        analytic_commutator_mat = _eprop.errorgen_layer_to_matrix(analytic_commutator, 2, errorgen_lbl_matrix_dict)
        self.assertLess(np.linalg.norm(numeric_commutator - analytic_commutator_mat), 1e-10)

    def test_errorgen_composition(self):
        
        #create an error generator basis.
        complete_errorgen_basis_2Q = CompleteElementaryErrorgenBasis('PP', QubitSpace(2), default_label_type='local')
        complete_errorgen_basis_3Q = CompleteElementaryErrorgenBasis('PP', QubitSpace(3), default_label_type='local')
        
        #use this basis to construct a dictionary from error generator labels to their
        #matrices.
        errorgen_lbls_2Q = complete_errorgen_basis_2Q.labels
        errorgen_lbl_matrix_dict_2Q = {lbl: mat for lbl, mat in zip(errorgen_lbls_2Q, complete_errorgen_basis_2Q.elemgen_matrices)}
        
        #augment testing with random selection of 3Q labels (some commutation relations for C and A terms require a minimum of 3 qubits).
        errorgen_lbls_3Q, errorgen_mats_3Q = select_random_items_from_multiple_lists([complete_errorgen_basis_3Q.labels, complete_errorgen_basis_3Q.elemgen_matrices], 50)
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

    def _label_pairs_2Q_and_3Q(self, num_3Q_labels=50, seed=1234):
        """The (basis, matrix dict, ordered label pairs) fixtures shared by the conjugation tests
        every pair of 2-qubit labels and a random selection of 3-qubit labels
        (some relations for C and A terms need a third qubit)."""
        fixtures = []
        for num_qubits, num_labels in ((2, None), (3, num_3Q_labels)):
            basis = CompleteElementaryErrorgenBasis('PP', QubitSpace(num_qubits), default_label_type='local')
            matrix_dict = {lbl: mat for lbl, mat in zip(basis.labels, basis.elemgen_matrices)}
            labels = list(basis.labels)
            if num_labels is not None:
                labels = random.Random(seed).sample(labels, num_labels)
            stim_labels = [_LSE.cast(lbl) for lbl in labels]
            fixtures.append((num_qubits, basis, matrix_dict, list(product(labels, repeat=2)), list(product(stim_labels, repeat=2))))
        return fixtures

    def test_pauli_conjugation_composition(self):
        #confirm the analytic action of the Pauli conjugation superoperator Q rho Q on every type of
        #elementary error generator against numerics (Q = S_Q + 1 in matrix form), for every
        #(Pauli, label) pair at 2 qubits and a random selection at 3 qubits.
        for num_qubits, basis, matrix_dict, label_pairs, stim_label_pairs in self._label_pairs_2Q_and_3Q():
            labels = sorted({pair[0] for pair in label_pairs}, key=str)
            stim_labels = [_LSE.cast(lbl) for lbl in labels]
            paulis = sorted({lbl.basis_element_labels[0] for lbl in basis.labels if lbl.errorgen_type == 'S'})
            if num_qubits == 3:
                paulis = random.Random(4321).sample(paulis, 12)
            for pauli_str in paulis:
                pauli = stim.PauliString(pauli_str)
                for lbl, stim_lbl in zip(labels, stim_labels):
                    numeric = _eprop.pauli_conjugation_composition_numerical(pauli_str, lbl, matrix_dict)
                    analytic = _eprop.pauli_conjugation_composition(pauli, stim_lbl)
                    analytic_mat = _eprop.errorgen_layer_to_matrix(analytic, num_qubits, matrix_dict)
                    norm_diff = np.linalg.norm(numeric - analytic_mat)
                    if norm_diff > 1e-10:
                        print(f'Difference in conjugation of {lbl} by {pauli_str} is greater than 1e-10.')
                        print(f'{norm_diff=}')
                        print(f'{analytic=}')
                        raise ValueError('Numeric and analytic Pauli conjugation compositions were not found to be identical!')
        #the sign of the conjugating Pauli is irrelevant
        lbl = _LSE('C', [stim.PauliString('XY'), stim.PauliString('ZI')])
        self.assertEqual(_eprop.pauli_conjugation_composition(stim.PauliString('-XZ'), lbl),
                         _eprop.pauli_conjugation_composition(stim.PauliString('XZ'), lbl))

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
        
        def aggregate(label_rate_pairs):
            # the returned order is not guaranteed, and a label may appear more than once.
            totals = {}
            for lbl, rate in label_rate_pairs:
                totals[lbl] = totals.get(lbl, 0) + rate
            return {lbl: rate for lbl, rate in totals.items() if abs(rate) > 1e-12}

        for lbls, rates, correct_lbls in zip(test_labels, rates, correct_iterative_compositions):
            iterated_composition = aggregate(_eprop.iterative_error_generator_composition(lbls, rates))
            correct = aggregate(correct_lbls)
            self.assertEqual(set(iterated_composition), set(correct))
            for lbl, rate in correct.items():
                self.assertAlmostEqual(iterated_composition[lbl], rate)

        _compare_analytic_numeric_iterative_composition(2)
        

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
        # exact cancellation: the second-order term (1/2)[H_X a, H_Y b] has an H_Z component of rate
        # r; giving H_Z the first-order rate -r must then remove H_Z from the merged output rather
        # than leave a zero (or sub-threshold) entry behind.
        H_X, H_Y, H_Z = (_LSE.cast(LEEL('H', [p])) for p in ('X', 'Y', 'Z'))
        layer_1 = {H_X: 1e-2}
        layer_2 = {H_Y: 3e-3}
        r = _eprop.bch_approximation(layer_1, layer_2, bch_order=2)[H_Z]
        self.assertGreater(abs(r), 1e-14)
        combined = _eprop.bch_approximation(layer_1, {H_Y: 3e-3, H_Z: -r}, bch_order=2)
        self.assertNotIn(H_Z, combined)
        self.assertTrue(all(abs(rate) > 1e-14 for rate in combined.values()))
        # magnus: layers ordered so that (1/2)[A(2), A(1)] reproduces the same second-order term.
        r = _eprop.magnus_expansion([layer_2, layer_1], magnus_order=2)[H_Z]
        self.assertGreater(abs(r), 1e-14)
        combined = _eprop.magnus_expansion([{H_Y: 3e-3, H_Z: -r}, layer_1], magnus_order=2)
        self.assertNotIn(H_Z, combined)
        self.assertTrue(all(abs(rate) > 1e-14 for rate in combined.values()))

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
            bracket_1, bracket_2 = {}, {}
            for j in range(m):
                add(bracket_1, rows[j])
            add(bracket_1, rows[m], 0.5)
            for j in range(m + 1, n):
                add(bracket_2, cols[j])
            add(bracket_2, cols[m], 0.5)
            add(expected, commutator(layers[m], bracket_1), 1 / 6)
            add(expected, commutator(layers[m], bracket_2), -1 / 6)
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

    def test_term_emitters(self):
        """
        Check the four term emitters `_H`, `_S`, `_C`, `_A` against the "Extended elementary
        error generator conventions" of the module docstring: for every signed (possibly
        identity or repeated) index combination the emitted canonical terms must sum to the
        superoperator obtained by substituting the signed Paulis literally into the defining
        sandwich expressions, and every emitted label must be canonical.
        """
        from pygsti.errorgenpropagation.localstimerrorgen import bel_less_than, bel_str

        def sandwich(M, N):  # rho -> M rho N on row-stacked vec(rho), pyGSTi's convention
            return np.kron(M, N.T)

        def ext_H(M, I):
            return -1j * (sandwich(M, I) - sandwich(I, M))

        def ext_S(M, I):
            return sandwich(M, M.conj().T) - sandwich(I, I)

        def ext_C(M, N, I):
            anti = M @ N + N @ M
            return sandwich(M, N) + sandwich(N, M) - 0.5 * (sandwich(anti, I) + sandwich(I, anti))

        def ext_A(M, N, I):
            comm = M @ N - N @ M
            return 1j * (sandwich(M, N) - sandwich(N, M) + 0.5 * (sandwich(comm, I) + sandwich(I, comm)))

        def label_matrix(lbl, I):
            mats = [p.to_unitary_matrix(endian='big') for p in lbl.basis_element_labels]
            fn = {'H': ext_H, 'S': ext_S, 'C': ext_C, 'A': ext_A}[lbl.errorgen_type]
            return fn(*mats, I)

        phases = [1, -1, 1j, -1j]
        for num_qubits in (1, 2):
            dim = 2**num_qubits
            I = np.eye(dim)
            paulis = [stim.PauliString(''.join(p)) for p in product('IXYZ', repeat=num_qubits)]
            identity = stim.PauliString(num_qubits)
            identity_str = 'I' * num_qubits
            # tie the sandwich definitions to pyGSTi's standard elementary error generators
            # for the ordinary (Hermitian, non-identity) case.
            P0, Q0 = paulis[1], paulis[-1]
            m0, m1 = P0.to_unitary_matrix(endian='big'), Q0.to_unitary_matrix(endian='big')
            for typ, mats in [('H', (m0,)), ('S', (m0,)), ('C', (m0, m1)), ('A', (m0, m1))]:
                self.assertArraysAlmostEqual(label_matrix(_LSE(typ, (P0, Q0)[:len(mats)]), I),
                                             create_elementary_errorgen(typ, *mats))

            for P in paulis:
                MP = P.to_unitary_matrix(endian='big')
                for w in phases:
                    for c in (0.7, -0.3j):
                        for emitter, expected in [(_eprop._H, c * ext_H(w * MP, I)), (_eprop._S, c * ext_S(w * MP, I))]:
                            terms = []
                            emitter(terms, (w, P), c, identity_str)
                            # the pre-rendered-string form must give the identical result.
                            terms_pre = []
                            emitter(terms_pre, (w, P, bel_str(P)), c, identity_str)
                            self.assertEqual(terms_pre, terms)
                            total = sum((rate * label_matrix(lbl, I) for lbl, rate in terms), np.zeros((dim**2, dim**2), complex))
                            self.assertArraysAlmostEqual(total, expected)
                            for lbl, rate in terms:
                                self.assertNotEqual(lbl.basis_element_labels[0], identity)
                                self.assertEqual(lbl._hashable_basis_element_labels, lbl.bel_to_strings())
                            self.assertLessEqual(len(terms), 1)
                        # a None index (vanishing commutator / anticommutator) is a zero term.
                        terms = []
                        _eprop._H(terms, None, c, identity_str)
                        _eprop._S(terms, None, c, identity_str)
                        _eprop._C(terms, None, (w, P), c, identity_str)
                        _eprop._A(terms, (w, P), None, c, identity_str)
                        self.assertEqual(terms, [])

            for P, Q in product(paulis, repeat=2):
                MP, MQ = P.to_unitary_matrix(endian='big'), Q.to_unitary_matrix(endian='big')
                for w, v in product(phases, repeat=2):
                    c = 0.7 - 0.3j
                    for emitter, expected in [(_eprop._C, c * ext_C(w * MP, v * MQ, I)),
                                              (_eprop._A, c * ext_A(w * MP, v * MQ, I))]:
                        terms = []
                        emitter(terms, (w, P), (v, Q), c, identity_str)
                        terms_pre = []
                        emitter(terms_pre, (w, P, bel_str(P)), (v, Q, bel_str(Q)), c, identity_str)
                        self.assertEqual(terms_pre, terms)
                        self.assertEqual([r for _, r in terms_pre], [r for _, r in terms])
                        total = sum((rate * label_matrix(lbl, I) for lbl, rate in terms), np.zeros((dim**2, dim**2), complex))
                        self.assertArraysAlmostEqual(total, expected)
                        self.assertLessEqual(len(terms), 1)
                        for lbl, rate in terms:
                            self.assertEqual(lbl._hashable_basis_element_labels, lbl.bel_to_strings())
                            self.assertNotIn(identity, lbl.basis_element_labels)
                            if lbl.errorgen_type in ('C', 'A'):
                                self.assertTrue(bel_less_than(*lbl.basis_element_labels))
                            else:  # degenerate outcomes of the two-index emitters
                                self.assertEqual(len(lbl.basis_element_labels), 1)

    def test_CA_label_ordering_preserved_by_tableau_propagation(self):
        """
        Regression test: `propagate_error_gen_tableau` must return C/A labels whose basis
        element label pair is still in canonical (sorted) order.

        A Clifford need not preserve the relative order of the two transformed Paulis (SWAP
        maps (IX, XI) -> (XI, IX)). Without a re-sort the propagated label (1) represents the
        negated generator for 'A', since A_{P,Q} = -A_{Q,P}, and (2) raises a KeyError against
        any canonically-keyed dict, e.g. in `errorgen_layer_to_matrix`.
        """
     
        # The motivating example: SWAP reverses the order of (IX, XI).
        swap_tableau = stim.Tableau.from_named_gate('SWAP')
        P, Q = stim.PauliString('+IX'), stim.PauliString('+XI')
        self.assertTrue(bel_less_than(P, Q))  # (P, Q) starts canonical
        C_lbl = _LSE('C', [P,Q])
        SWAP_C_lbl = C_lbl.propagate_error_gen_tableau(swap_tableau, weight=1)
        assert SWAP_C_lbl[0].basis_element_labels == (P,Q), "propagated bels not in canonical ordering."
        assert SWAP_C_lbl[1] == 1, "Incorrect weight following C subscript swap."

        A_lbl = _LSE('A', [P,Q])
        SWAP_A_lbl = A_lbl.propagate_error_gen_tableau(swap_tableau, weight=1)
        assert SWAP_A_lbl[0].basis_element_labels == (P,Q), "propagated bels not in canonical ordering."
        assert SWAP_A_lbl[1] == -1, "Incorrect weight following A subscript swap."


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
        bit_strings_3Q =  [''.join(bit_tup) for bit_tup in product(['0','1'], repeat=3)]
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

    def test_slow_bulk_alpha_weight_1_active_generators(self):
        # Regression test for a bug in the pure-Python `slow_bulk_alpha` fallback (the function
        # used whenever the Cython `fasterrgencalc` extension is unavailable -- `bulk_alpha` is
        # simply aliased to the Cython `fast_bulk_alpha` when it *is* available, which silently
        # masks this bug in the common case). The "A1" branch (weight-1 'A'-type generators whose
        # two Paulis anticommute -- i.e. essentially *every* weight-1 A-type generator, since any
        # two distinct single-qubit Paulis anticommute) assigned to `sensitivities_by_bitstring`
        # (the function's entire return array) instead of the per-iteration local `sensitivity`,
        # which was then read on the very next line -- an UnboundLocalError. This test calls
        # `slow_bulk_alpha` directly (bypassing the Cython alias) to exercise the actual
        # pure-Python code path, regardless of whether the Cython extension happens to be built
        # in the current environment.
        tableau = stim.PauliString('XI').to_tableau()
        errgen = _LSE('A', [stim.PauliString('X'), stim.PauliString('Y')])
        bitstrings = ['00', '01', '10', '11']
        for bitstring in bitstrings:
            expected = _eprop.alpha(errgen, tableau, bitstring)
            result = _eprop.slow_bulk_alpha([errgen], tableau, [bitstring])
            self.assertAlmostEqual(result[0, 0], expected)

        # Also check a weight-2+ random sample from a complete 3-qubit basis, restricted to A-type
        # generators, for good measure (both anticommuting and commuting P,Q sub-cases).
        errorgen_basis = CompleteElementaryErrorgenBasis('PP', QubitSpace(3), default_label_type='local',
                                                          elementary_errorgen_types=('A',))
        rng = np.random.default_rng(0)
        random_errorgens = rng.choice(np.fromiter(errorgen_basis.labels, dtype=object), size=20, replace=False)
        errorgen_labels = [_LSE.cast(lbl) for lbl in random_errorgens]
        tableau_3q = stim.Tableau.random(3)
        for bitstring in [''.join(p) for p in product(['0', '1'], repeat=3)]:
            expected = [_eprop.alpha(eg, tableau_3q, bitstring) for eg in errorgen_labels]
            result = _eprop.slow_bulk_alpha(errorgen_labels, tableau_3q, [bitstring])
            np.testing.assert_allclose(result[0, :], expected, atol=1e-10)

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

def _all_sector_rate_subset(num_qubits, num_per_sector, seed):
    """
    A seeded random subset of a random CPTP error generator rate dictionary on `num_qubits`
    qubits, keeping `num_per_sector` labels from each of the H, S, C and A sectors. The subset
    is generally not CP (S terms supporting the kept C/A terms may be dropped), which is
    irrelevant for error generator propagation.
    """
    full = random_CPTP_error_generator_rates(num_qubits, errorgen_types=('H', 'S', 'C', 'A'), H_params=(0, .01),
                                             SCA_params=(0, .01), label_type='local', seed=seed)
    rng = np.random.default_rng(seed + 1)
    subset = {}
    for sector in 'HSCA':
        lbls = [lbl for lbl in full if lbl.errorgen_type == sector]
        for i in sorted(rng.choice(len(lbls), min(num_per_sector, len(lbls)), replace=False)):
            subset[lbls[i]] = float(full[lbls[i]])
    return subset

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

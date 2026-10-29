import copy
from itertools import chain, product
import pickle
import random

import numpy as np
import stim

from pygsti.baseobjs import QubitSpace
from pygsti.baseobjs.errorgenbasis import CompleteElementaryErrorgenBasis
from pygsti.baseobjs.errorgenlabel import LocalElementaryErrorgenLabel as LEEL
from pygsti.errorgenpropagation.localstimerrorgen import LocalStimErrorgenLabel as _LSE, bel_less_than, bel_str
from pygsti.tools import errorgenpropagation as _eprop
from pygsti.tools.lindbladtools import create_elementary_errorgen
from pygsti.tools.matrixtools import print_mx

from ...util import BaseCase
from .errgenprop_test_utils import (
    select_random_items_from_multiple_lists,
    _compare_analytic_numeric_iterative_composition,
)


class ErrgenAlgebraTester(BaseCase):

    def _label_pairs_2Q_and_3Q(self, num_3Q_labels=50, seed=1234):
        """The (basis, matrix dict, ordered label pairs) fixtures shared by the anticommutator and
        conjugation tests: every pair of 2-qubit labels and a random selection of 3-qubit labels
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

    def test_errorgen_anticommutators(self):
        #confirm we get the correct analytic anticommutators by comparing to numerics, in the same
        #way as the commutator and composition tests above.
        for num_qubits, basis, matrix_dict, label_pairs, stim_label_pairs in self._label_pairs_2Q_and_3Q():
            for pair1, pair2 in zip(label_pairs, stim_label_pairs):
                numeric_anticommutator = _eprop.error_generator_anticommutator_numerical(pair1[0], pair1[1], matrix_dict)
                analytic_anticommutator = _eprop.error_generator_anticommutator(pair2[0], pair2[1])
                analytic_anticommutator_mat = _eprop.errorgen_layer_to_matrix(analytic_anticommutator, num_qubits, matrix_dict)
                norm_diff = np.linalg.norm(numeric_anticommutator - analytic_anticommutator_mat)
                if norm_diff > 1e-10:
                    print(f'Difference in anticommutators for pair {pair1} is greater than 1e-10.')
                    print(f'{norm_diff=}')
                    print('numeric_anticommutator=')
                    print_mx(numeric_anticommutator)
                    #Decompose the numerical anticommutator into rates.
                    for lbl, dual in zip(basis.labels, basis.elemgen_dual_matrices):
                        rate = np.trace(dual.conj().T@numeric_anticommutator)
                        if abs(rate) > 1e-3:
                            print(f'{lbl}: {rate}')
                    print(f'{analytic_anticommutator=}')
                    raise ValueError('Numeric and analytic error generator anticommutators were not found to be identical!')

    def test_errorgen_anticommutator_matches_compositions(self):
        #the anticommutator is the sum of the two compositions, and the composition is the mean of the
        #commutator and the anticommutator, term by term (same labels, same rates once aggregated); the
        #anticommutator is also symmetric in its arguments.
        def aggregate(terms):
            out = {}
            for lbl, rate in terms:
                out[lbl] = out.get(lbl, 0) + rate
            return {lbl: rate for lbl, rate in out.items() if abs(rate) > 1e-12}

        def assert_same(d1, d2, msg):
            self.assertEqual(set(d1), set(d2), msg)
            for lbl in d1:
                self.assertAlmostEqual(d1[lbl], d2[lbl], places=12, msg=msg)

        weight = 0.7
        _, _, _, _, stim_label_pairs = self._label_pairs_2Q_and_3Q()[0]
        identity = 'II'
        for lbl1, lbl2 in stim_label_pairs:
            anticommutator = aggregate(_eprop.error_generator_anticommutator(lbl1, lbl2, weight, identity))
            comp_12 = _eprop.error_generator_composition(lbl1, lbl2, weight, identity)
            comp_21 = _eprop.error_generator_composition(lbl2, lbl1, weight, identity)
            assert_same(anticommutator, aggregate(comp_12 + comp_21), f'{lbl1}, {lbl2}: {{e1, e2}} != e1 o e2 + e2 o e1')
            assert_same(anticommutator, aggregate(_eprop.error_generator_anticommutator(lbl2, lbl1, weight, identity)),
                        f'{lbl1}, {lbl2}: anticommutator not symmetric')
            half_sum = aggregate([(lbl, 0.5 * rate) for lbl, rate in
                                  _eprop.error_generator_commutator(lbl1, lbl2, weight, identity)
                                  + _eprop.error_generator_anticommutator(lbl1, lbl2, weight, identity)])
            assert_same(aggregate(comp_12), half_sum, f'{lbl1}, {lbl2}: e1 o e2 != ([e1, e2] + {{e1, e2}}) / 2')

    def test_pauli_conjugation_composition(self):
        #confirm the analytic action of the Pauli conjugation superoperator Q rho Q on every type of
        #elementary error generator against numerics (Q = S_Q + 1 in matrix form), for every
        #(Pauli, label) pair at 2 qubits and a random selection at 3 qubits.
        for num_qubits, basis, matrix_dict, label_pairs, stim_label_pairs in self._label_pairs_2Q_and_3Q():
            labels = sorted({pair[0] for pair in label_pairs}, key=str)
            stim_labels = [_LSE.cast(lbl) for lbl in labels]
            paulis = sorted({lbl.basis_element_labels[0] for lbl in basis.labels if lbl.errorgen_type == 'S'})
            if num_qubits == 3:
                paulis = random.Random().sample(paulis, 12)
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

    def test_pauli_conjugation_bleed_rule(self):
        #{S_Q, X} = (1 + (-1)^omega(Q, T_X)) Q[X] - 2X with T_X the total Pauli of X (P for H_P, I for S_P,
        #PR for C_{P,R} and A_{P,R}): the anticommutator is pure bleed (-2X) exactly when Q anticommutes
        #with T_X, and otherwise -2X + 2 Q[X].
        def aggregate(terms):
            out = {}
            for lbl, rate in terms:
                out[lbl] = out.get(lbl, 0) + rate
            return {lbl: rate for lbl, rate in out.items() if abs(rate) > 1e-12}

        def total_pauli(lbl):
            bels = lbl.basis_element_labels
            if lbl.errorgen_type == 'S':
                return None
            return bels[0] if len(bels) == 1 else bels[0] * bels[1]

        _, _, _, _, stim_label_pairs = self._label_pairs_2Q_and_3Q()[0]
        stim_labels = sorted({pair[0] for pair in stim_label_pairs}, key=str)
        s_labels = [lbl for lbl in stim_labels if lbl.errorgen_type == 'S']
        num_pure_bleed = 0
        for s_lbl in s_labels:
            Q = s_lbl.basis_element_labels[0]
            for x_lbl in stim_labels:
                anticommutator = aggregate(_eprop.error_generator_anticommutator(s_lbl, x_lbl, identity='II'))
                T = total_pauli(x_lbl)
                expected = [(x_lbl, -2.0)]
                if T is None or Q.commutes(T):
                    expected += _eprop.pauli_conjugation_composition(Q, x_lbl, weight=2.0, identity='II')
                else:
                    num_pure_bleed += 1
                expected = aggregate(expected)
                self.assertEqual(set(anticommutator), set(expected), f'{s_lbl}, {x_lbl}')
                for lbl in expected:
                    self.assertAlmostEqual(anticommutator[lbl], expected[lbl], places=12, msg=f'{s_lbl}, {x_lbl}')
        self.assertGreater(num_pure_bleed, 0)
        #conjugation composed with S is the only case producing an S of the conjugating Pauli itself
        self.assertEqual(aggregate(_eprop.pauli_conjugation_composition(stim.PauliString('XI'), _LSE('S', [stim.PauliString('XI')]))),
                         {_LSE('S', [stim.PauliString('XI')]): -1.0})
        self.assertEqual(aggregate(_eprop.pauli_conjugation_composition(stim.PauliString('XI'), _LSE('S', [stim.PauliString('YI')]))),
                         {_LSE('S', [stim.PauliString('ZI')]): 1.0, _LSE('S', [stim.PauliString('XI')]): -1.0})

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

        # The propagated label is handed its strings pre-rendered (`pauli_str_reps`); they must
        # be the strings of its (reordered) Paulis, so that hashing and ordering stay consistent.
        tableau = stim.Tableau.random(6)
        for typ, bels in [('H', ['XIIIII']), ('S', ['IYIIZI']), ('C', ['IIZIIY', 'XIIIII']), ('A', ['IXIIII', 'ZIIIYI'])]:
            lbl = _LSE.cast((typ, bels))
            new_lbl, _ = lbl.propagate_error_gen_tableau(tableau, weight=1)
            self.assertEqual(new_lbl._hashable_basis_element_labels, tuple(bel_str(p) for p in new_lbl.basis_element_labels))
            self.assertEqual(new_lbl, _LSE(typ, new_lbl.basis_element_labels))
            self.assertEqual(hash(new_lbl), hash(_LSE(typ, new_lbl.basis_element_labels)))
            if len(bels) == 2:
                self.assertTrue(bel_less_than(*new_lbl.basis_element_labels))

    def test_local_stim_errorgen_label_pickle_fidelity(self):
        h = _LSE('H', (stim.PauliString('XIIX'),), pauli_str_reps=('XIIX',))
        s = _LSE('S', (stim.PauliString('YIIY'),), pauli_str_reps=('YIIY',))
        c = _LSE('C', (stim.PauliString('XIII'), stim.PauliString('IIZI')), pauli_str_reps=('XIII', 'IIZI'))
        a = _LSE('A', (stim.PauliString('YIII'), stim.PauliString('IIZI')), pauli_str_reps=('YIII', 'IIZI'))
        for lbl in (h, s, c, a):
            data = pickle.dumps(lbl, protocol=pickle.HIGHEST_PROTOCOL)
            rec = pickle.loads(data)
            self.assertEqual(rec, lbl)
            self.assertEqual(hash(rec), hash(lbl))
            self.assertEqual(rec.errorgen_type, lbl.errorgen_type)
            self.assertEqual(rec.basis_element_labels, lbl.basis_element_labels)
            self.assertEqual(rec._hashable_basis_element_labels, lbl._hashable_basis_element_labels)

    def test_bel_product_str(self):
        """
        `bel_product_str` (the string-level Pauli product the emitters use in place of a stim
        product plus rendering) must agree with stim on every pair of single-qubit Paulis and
        on random multi-qubit pairs, including the 'I'-padded strings of 100-qubit labels and
        strings longer than the cython implementation's stack buffer. Both the pure-python
        implementation and the cython one (when built) are checked.
        """
        from pygsti.errorgenpropagation import localstimerrorgen as lse
        impls = {'default': lse.bel_product_str, 'python': lse._slow_bel_product_str}
        try:
            from pygsti.tools.errorgenpropagation.fasterrgencalc import fast_bel_product_str
            impls['cython'] = fast_bel_product_str
        except ImportError:
            pass

        def reference(s1, s2):
            PQ = stim.PauliString(s1) * stim.PauliString(s2)
            return lse.bel_str(PQ / PQ.sign)

        rng = np.random.default_rng(2024)
        cases = [(a, b) for a, b in product('IXYZ', repeat=2)]
        for num_qubits in (2, 3, 5, 100, 257, 300):
            for _ in range(100):
                cases.append(tuple(''.join(rng.choice(list('IXYZ'), size=num_qubits, p=[0.7, 0.1, 0.1, 0.1]))
                                   for _ in range(2)))
        cases += [('IXYZ', 'IIII'), ('IXYZ', 'IXYZ')]
        for name, impl in impls.items():
            for s1, s2 in cases:
                self.assertEqual(impl(s1, s2), reference(s1, s2), f'{name}: {s1} * {s2}')

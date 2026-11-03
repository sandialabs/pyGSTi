from itertools import product

import numpy as np
import stim

from pygsti.algorithms.randomcircuit import create_random_circuit
from pygsti.baseobjs import QubitSpace
from pygsti.baseobjs.errorgenbasis import CompleteElementaryErrorgenBasis
from pygsti.errorgenpropagation.localstimerrorgen import LocalStimErrorgenLabel as _LSE
from pygsti.models.modelconstruction import create_crosstalk_free_model
from pygsti.processors import QubitProcessorSpec
from pygsti.tools import errorgenpropagation as _eprop

from ...util import BaseCase


class ErrgenSensitivitiesTester(BaseCase):

    def setUp(self):
        num_qubits = 4
        gate_names = ['Gcphase', 'Gxpi2', 'Gypi2']
        availability = {'Gcphase': [(0, 1), (1, 2), (2, 3), (3, 0)]}
        pspec = QubitProcessorSpec(num_qubits, gate_names, availability=availability)
        self.circuit = create_random_circuit(pspec, 4, sampler='edgegrab', samplerargs=[0.4], rand_state=12345)
        self.circuit_alt = create_random_circuit(pspec, 4, sampler='edgegrab', samplerargs=[0.4], rand_state=12345)
        self.circuit_tableau = self.circuit.convert_to_stim_tableau()
        self.circuit_tableau_alt = self.circuit_alt.convert_to_stim_tableau()

        # Also create a 3-qubit pspec for making some tests faster.
        num_qubits = 3
        gate_names = ['Gcphase', 'Gxpi2', 'Gypi2']
        availability = {'Gcphase': [(0, 1), (1, 2)]}
        pspec = QubitProcessorSpec(num_qubits, gate_names, availability=availability)
        self.circuit_3Q = create_random_circuit(pspec, 4, sampler='edgegrab', samplerargs=[0.4], rand_state=12345)
        self.circuit_tableau_3Q = self.circuit_3Q.convert_to_stim_tableau()

    def test_random_support(self):
        num_random = _eprop.random_support(self.circuit_tableau)
        self.assertEqual(num_random, 3)

    def test_amplitude_of_state(self):
        amp0000 = _eprop.amplitude_of_state(self.circuit_tableau, '0000', False)
        amp1111 = _eprop.amplitude_of_state(self.circuit_tableau, '1111', False)
        self.assertTrue(abs(amp0000) < 1e-7)
        self.assertTrue(abs(amp1111 - (-1j * np.sqrt(0.125))) < 1e-7)

        amp0000 = _eprop.amplitude_of_state(self.circuit_tableau_alt, '0000', False)
        amp1111 = _eprop.amplitude_of_state(self.circuit_tableau_alt, '1111', False)

        self.assertTrue(abs(amp0000) < 1e-7)
        self.assertTrue(abs(amp1111 - (-1j * np.sqrt(0.125))) < 1e-7)

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
        bit_strings_3Q = list(product(['0', '1'], repeat=3))
        rng = np.random.default_rng()
        paulis = np.fromiter(stim.PauliString.iter_all(3), dtype=object)
        random_paulis = rng.choice(paulis, size=10, replace=False)
        for bit_string in bit_strings_3Q:
            for pauli_1, pauli_2 in product(random_paulis, random_paulis):
                phi_num = _eprop.phi_numerical(self.circuit_tableau_3Q, bit_string, pauli_1, pauli_2)
                phi_analytic = _eprop.phi(self.circuit_tableau_3Q, bit_string, pauli_1, pauli_2)
                if abs(phi_num - phi_analytic) > 1e-4:
                    _eprop.phi(self.circuit_tableau_3Q, bit_string, pauli_1, pauli_2, debug=True)
                    raise ValueError(f'{pauli_1}, {pauli_2}, {bit_string}, {phi_num=}, {phi_analytic=}')

    def test_bulk_phi(self):
        bit_strings_3Q = list(product(['0', '1'], repeat=3))
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
        bit_strings_3Q = [''.join(bit_tup) for bit_tup in product(['0', '1'], repeat=3)]
        complete_errorgen_basis_3Q = CompleteElementaryErrorgenBasis('PP', QubitSpace(3), default_label_type='local')
        rng = np.random.default_rng()
        random_errorgens = rng.choice(np.fromiter(complete_errorgen_basis_3Q.labels, dtype=object), size=100, replace=False)
        for bit_string in bit_strings_3Q:
            for lbl in random_errorgens:
                alpha_num = _eprop.alpha_numerical(lbl, self.circuit_tableau_3Q, bit_string)
                assert abs(alpha_num - _eprop.alpha(lbl, self.circuit_tableau_3Q, bit_string)) < 1e-4

    def test_bulk_alpha(self):
        from pygsti.modelpacks import smq2Q_XYCPHASE
        pspec_2Q = smq2Q_XYCPHASE.processor_spec()
        random_circuits_2Q = [create_random_circuit(pspec_2Q, 4, sampler='edgegrab', samplerargs=[0.4], rand_state=12345 + i) for i in range(5)]
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
        random_circuits_2Q = [create_random_circuit(pspec_2Q, 4, sampler='edgegrab', samplerargs=[0.4], rand_state=12345 + i) for i in range(5)]
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

                    if abs(alpha_analytic - alpha_numerical) > 1e-5:
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
        random_circuits_2Q = [create_random_circuit(pspec_2Q, 4, sampler='edgegrab', samplerargs=[0.4], rand_state=12345 + i) for i in range(5)]
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

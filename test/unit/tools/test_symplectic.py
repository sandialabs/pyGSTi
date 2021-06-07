import numpy as np
import unittest

from ..util import BaseCase

import pygsti
from pygsti.tools import matrixmod2
from pygsti.objects.label import Label
import pygsti.tools.symplectic as symplectic

try:
    from pygsti.tools import fastcalc as _fastcalc
except ImportError:
    _fastcalc = None


class SymplecticBase(object):
    # Tests that check the symplectic random sampler, check_symplectic, the convention converter,
    # and the symplectic form constructing function are working.

    # All tests should be run for both even and odd dimensions, so this base
    # class defines tests which are inherited by the actual test cases.

    def test_symplectic_form(self):
        omega_S = symplectic.symplectic_form(self.n, convention='standard')
        omega_DS = symplectic.symplectic_form(self.n, convention='directsum')
        omega_DStoS = symplectic.change_symplectic_form_convention(omega_DS)
        self.assertArraysEqual(omega_S, omega_DStoS)
        omega_StoDS = symplectic.change_symplectic_form_convention(omega_S, outconvention='directsum')
        self.assertArraysEqual(omega_DS, omega_StoDS)

    def test_random_symplectic_matrix(self):
        # Pick a random symplectic matrix in the standard convention and check it is symplectic
        s_S = symplectic.random_symplectic_matrix(self.n)
        self.assertTrue(symplectic.check_symplectic(s_S))

        # Pick a random symplectic matrix in the directsum convention and check it is symplectic
        s_DS = symplectic.random_symplectic_matrix(self.n, convention='directsum')
        self.assertTrue(symplectic.check_symplectic(s_DS, convention='directsum'))

    def test_change_symplectic_form_convention(self):
        # Convert the directsum convention to the standard convention and check the output is symplectic in new convention
        s_DS = symplectic.random_symplectic_matrix(self.n, convention='directsum')
        s_DStoS = symplectic.change_symplectic_form_convention(s_DS)
        self.assertTrue(symplectic.check_symplectic(s_DStoS, convention='standard'))

        # Convert back to the directsum convention, and check the original matrix is recovered.
        s_DStoStoDS = symplectic.change_symplectic_form_convention(s_DStoS, outconvention='directsum')
        self.assertArraysEqual(s_DS, s_DStoStoDS)

    def test_inverse_symplectic(self):
        # Check that the inversion function is working.
        s_S = symplectic.random_symplectic_matrix(self.n)
        sin = symplectic.inverse_symplectic(s_S)
        self.assertArraysEqual(matrixmod2.dot_mod2(sin, s_S), np.identity(2 * self.n, int))
        self.assertArraysEqual(matrixmod2.dot_mod2(s_S, sin), np.identity(2 * self.n, int))

    def test_random_clifford(self):
        # Check the Clifford sampler runs.
        s, p = symplectic.random_clifford(self.n)

        # Check that a randomly sampled Clifford is a valid Clifford
        self.assertTrue(symplectic.check_valid_clifford(s, p))

    def test_inverse_clifford(self):
        # Check the inverse Clifford function runs, and gives a valid Clifford
        s, p = symplectic.random_clifford(self.n)
        sin, pin = symplectic.inverse_clifford(s, p)
        self.assertTrue(symplectic.check_valid_clifford(sin, pin))

        # Check the symplectic matrix part of the inverse Clifford works
        self.assertArraysEqual(matrixmod2.dot_mod2(sin, s), np.identity(2 * self.n, int))
        self.assertArraysEqual(matrixmod2.dot_mod2(s, sin), np.identity(2 * self.n, int))

    def test_compose_cliffords(self):
        # Check that the composite Clifford function runs, and works correctly in the special case whereby
        # one Clifford is the inverse of the other.
        s, p = symplectic.random_clifford(self.n)
        sin, pin = symplectic.inverse_clifford(s, p)
        scomp, pcomp = symplectic.compose_cliffords(s, p, sin, pin)
        self.assertArraysEqual(scomp, np.identity(2 * self.n, int))
        self.assertArraysEqual(pcomp, np.zeros(2 * self.n, int))

    def test_construct_valid_phase_vector(self):
        # Check the p returned is unchanged when the seed is valid.
        s, p = symplectic.random_clifford(self.n)
        pvalid = symplectic.construct_valid_phase_vector(s, p)
        self.assertArraysEqual(p, pvalid)

        # Check that p returned is a valid Clifford when the input pseed is not
        pseed = (p - 1) % 2
        pvalid = symplectic.construct_valid_phase_vector(s, pseed)
        self.assertTrue(symplectic.check_valid_clifford(s, pvalid))

    def test_internal_gate_symplectic_representations(self):
        # Basic tests of the symp. rep. dictionary
        srep_dict = symplectic.compute_internal_gate_symplectic_representations()

        H = (1 / np.sqrt(2)) * np.array([[1., 1.], [1., -1.]], complex)
        s, p = symplectic.unitary_to_symplectic(H)
        self.assertArraysEqual(s, srep_dict['H'][0])
        self.assertArraysEqual(p, srep_dict['H'][1])

        CNOT = np.array([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 0., 1.], [0., 0., 1., 0.]], complex)
        s, p = symplectic.unitary_to_symplectic(CNOT)
        self.assertArraysEqual(s, srep_dict['CNOT'][0])
        self.assertArraysEqual(p, srep_dict['CNOT'][1])
    
    def test_circuit_symplectic_representation(self):
        srep_dict = symplectic.compute_internal_gate_symplectic_representations()

        # Dummy circuit where HZHX = I
        # Running on qubit 1 of two to ensure proper indexing of operations into full matrix
        HZHcirc = pygsti.obj.Circuit([('H', 1), ('Z', 1), ('H', 1), ('X', 1)], num_lines=2)
        s, p  = symplectic.symplectic_rep_of_clifford_circuit(HZHcirc)
        self.assertArraysAlmostEqual(s, np.eye(4))
        self.assertArraysAlmostEqual(p, np.zeros(4))

        # Also test with non-hardcoded names
        HZHcirc = pygsti.obj.Circuit([('Gh', 1), ('Gzpi', 1), ('Gh', 1), ('Gxpi', 1)], num_lines=2)
        srep_custom = {'Gh': srep_dict['H'], 'Gzpi': srep_dict['Z'], 'Gxpi': srep_dict['X']}
        s, p  = symplectic.symplectic_rep_of_clifford_circuit(HZHcirc, srep_dict=srep_custom)
        self.assertArraysAlmostEqual(s, np.eye(4))
        self.assertArraysAlmostEqual(p, np.zeros(4))
    
    @unittest.skipIf(_fastcalc is None, "Skipping fast compose test since no fastcalc compiled")
    def test_fast_compose_cliffords(self):
        srep_dict = symplectic.compute_internal_gate_symplectic_representations()

        # Seems like a good idea to test different embeddings
        # This creates the StateSpaceLabels so that symplectic_rep_of_clifford_layer can work properly
        def sp_offset_embedding(gate, offset, gate_qubits, total_qubits):
            layer_lbl = Label(gate, state_space_labels=list(range(offset, offset+gate_qubits)))
            return symplectic.symplectic_rep_of_clifford_layer(layer_lbl, total_qubits, add_internal_sreps=False)

        # Do all pairwise composes
        for g1 in srep_dict.keys():
            s1, p1 = srep_dict[g1]
            g1qubits = len(p1) // 2

            for g2 in srep_dict.keys():
                s2, p2 = srep_dict[g2]
                g2qubits = len(p2) // 2

                # Try all pairwise offsets in larger space
                for offset1 in range(self.n - g1qubits):
                    s1_embed, p1_embed = sp_offset_embedding(g1, offset1, g1qubits, self.n)

                    for offset2 in range(self.n - g2qubits):
                        s2_embed, p2_embed = sp_offset_embedding(g2, offset2, g2qubits, self.n)

                        # Actually try and compare clifford composes
                        s12_slow, p12_slow = symplectic.compose_cliffords(s1_embed, p1_embed,
                                                                          s2_embed, p2_embed,
                                                                          do_checks=False)
                        s12_fast, p12_fast = _fastcalc.fast_compose_cliffords(s1_embed, p1_embed,
                                                                              s2_embed, p2_embed)
                        
                        # Guard output just so for easier debugging...
                        if not np.allclose(s12_slow, s12_fast) or not np.allclose(p12_slow, p12_fast):
                            print(f'Error detected in {g1} x {g2} with offsets {offset1} and {offset2}')

                            print(f'{g1} s:\n{s1}')
                            print(f'{g1} s embedded with offset {offset1}:\n{s1_embed}')
                            print(f'{g2} s:\n{s2}')
                            print(f'{g2} s embedded with offset {offset2}:\n{s2_embed}')

                            print(f'{g1} p: {p1}')
                            print(f'{g1} p embedded with offset {offset1}: {p1_embed}')
                            print(f'{g2} p : {p2}')
                            print(f'{g2} p embedded with offset {offset2}: {p2_embed}')

                            if not np.allclose(s12_slow, s12_fast):
                                print('\nError in s matrices!')
                                print(f'Python s1_embed x s2_embed:\n{s12_slow}')
                                print(f'Cython s1_embed x s2_embed:\n{s12_fast}')
                            
                            if not np.allclose(p12_slow, p12_fast):
                                print('\nError in p vectors')
                                print(f'Python s1_embed x s2_embed: {p12_slow}')
                                print(f'Cython s1_embed x s2_embed: {p12_fast}')
                        
                        self.assertArraysAlmostEqual(s12_slow, s12_fast)
                        self.assertArraysAlmostEqual(p12_slow, p12_fast)

class SymplecticEvenDimTester(SymplecticBase, BaseCase):
    n = 4


class SymplecticOddDimTester(SymplecticBase, BaseCase):
    n = 5

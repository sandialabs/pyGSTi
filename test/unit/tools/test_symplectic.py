import numpy as np

from ..util import BaseCase

from pygsti.tools import matrixmod2
import pygsti.tools.symplectic as symplectic


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


class SymplecticEvenDimTester(SymplecticBase, BaseCase):
    n = 4


class SymplecticOddDimTester(SymplecticBase, BaseCase):
    n = 5

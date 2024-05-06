import numpy as np
import scipy.sparse as sps

from pygsti.tools import lindbladtools as lt
from pygsti.baseobjs import Basis
from ..util import BaseCase


class LindbladToolsTester(BaseCase):
    def test_hamiltonian_to_lindbladian(self):
        expectedLindbladian = np.array([
            [ 0,  0,  0,  0],
            [ 0,  0,  0,  0],
            [ 0,  0,  0,  0],
            [ 0,  0,  0,  0]
        ])
        self.assertArraysAlmostEqual(lt.create_elementary_errorgen('H', np.zeros(shape=(2, 2))),
                                     expectedLindbladian)
        sparse = sps.csr_matrix(np.zeros(shape=(2, 2)))
        #spL = lt.hamiltonian_to_lindbladian(sparse, True)
        spL = lt.create_elementary_errorgen('H', sparse, sparse=True)
        self.assertArraysAlmostEqual(spL.toarray(),
                                     expectedLindbladian)

    def test_stochastic_lindbladian(self):
        a = np.array([[1, 2], [3, 4]], 'd')
        expected = np.array([
            [ 1,  2,  2,  4],
            [ 3,  4,  6,  8],
            [ 3,  6,  4,  8],
            [ 9, 12, 12, 16]
        ], 'd')
        dual_eg, norm = lt.create_elementary_errorgen_dual('S', a, normalization_factor='auto_return')
        self.assertArraysAlmostEqual(
            dual_eg * norm, expected)
        sparse = sps.csr_matrix(a)
        spL = lt.create_elementary_errorgen_dual('S', sparse, sparse=True)
        self.assertArraysAlmostEqual(spL.toarray() * norm, expected)

    def test_nonham_lindbladian(self):
        a = np.array([[1, 2], [3, 4]], 'd')
        b = np.array([[1, 2], [3, 4]], 'd')
        expected = np.array([
            [ -9,  -5,  -5,  4],
            [ -4, -11,   6,  1],
            [ -4,   6, -11,  1],
            [  9,   5,   5, -4]
        ], 'd')
        self.assertArraysAlmostEqual(lt.create_lindbladian_term_errorgen('O', a, b), expected)
        sparsea = sps.csr_matrix(a)
        sparseb = sps.csr_matrix(b)
        spL = lt.create_lindbladian_term_errorgen('O', sparsea, sparseb, sparse=True)
        self.assertArraysAlmostEqual(spL.toarray(),
                                     expected)

    def test_elementary_errorgen_bases(self):

        bases = [Basis.cast('gm', 4),
                 Basis.cast('pp', 4),
                 Basis.cast('PP', 4)]

        for basis in bases:
            print(basis)

            primals = []; duals = []; lbls = []
            for lbl, bel in zip(basis.labels[1:], basis.elements[1:]):
                lbls.append("H_%s" % lbl)
                primals.append(lt.create_elementary_errorgen('H', bel)) 
                duals.append(lt.create_elementary_errorgen_dual('H', bel)) 
            for lbl, bel in zip(basis.labels[1:], basis.elements[1:]):
                lbls.append("S_%s" % lbl)
                primals.append(lt.create_elementary_errorgen('S', bel))
                duals.append(lt.create_elementary_errorgen_dual('S', bel)) 
            for i, (lbl, bel) in enumerate(zip(basis.labels[1:], basis.elements[1:])):
                for lbl2, bel2 in zip(basis.labels[1+i+1:], basis.elements[1+i+1:]):
                    lbls.append("C_%s_%s" % (lbl, lbl2))
                    primals.append(lt.create_elementary_errorgen('C', bel, bel2))
                    duals.append(lt.create_elementary_errorgen_dual('C', bel, bel2)) 
            for i, (lbl, bel) in enumerate(zip(basis.labels[1:], basis.elements[1:])):
                for lbl2, bel2 in zip(basis.labels[1+i+1:], basis.elements[1+i+1:]):
                    lbls.append("A_%s_%s" % (lbl, lbl2))
                    primals.append(lt.create_elementary_errorgen('A', bel, bel2))
                    duals.append(lt.create_elementary_errorgen_dual('A', bel, bel2)) 

            dot_mx = np.empty((len(duals), len(primals)), complex)
            for i, dual in enumerate(duals):
                for j, primal in enumerate(primals):
                    dot_mx[i,j] = np.vdot(dual.flatten(), primal.flatten())

            self.assertTrue(np.allclose(dot_mx, np.identity(len(lbls), 'd')))
    
    def test_paulistr_tools(self):
        # Multiplication
        self.assertTrue(lt.paulistr_multiply("X", "Y") == (1j, "Z")) # Weight-1 product...
        self.assertTrue(lt.paulistr_multiply("Y", "X") == (-1j, "Z")) # ... and its cyclic counterpart
        self.assertTrue(lt.paulistr_multiply("I", "Z") == (1, "Z")) # Product with I leaves string unchanged
        self.assertTrue(lt.paulistr_multiply("Y", "Y") == (1, "I")) # and self-product gives I
        self.assertTrue(lt.paulistr_multiply("YZ", "YX") == (1j, "IY")) # One Pauli difference
        self.assertTrue(lt.paulistr_multiply("XY", "ZI") == (-1j, "YY")) # Single I formula
        self.assertTrue(lt.paulistr_multiply("XI", "ZI") == (-1j, "YI")) # I in single place in both strings
        self.assertTrue(lt.paulistr_multiply("XX", "II") == (1, "XX")) # Weight-2 product with I
        self.assertTrue(lt.paulistr_multiply("IX", "XI") == (1, "XX")) # ...and with I in both places
        self.assertTrue(lt.paulistr_multiply("XYZ", "XZX") == (-1, "IXY")) # Product in two places = 1j^2 = -1
        self.assertTrue(lt.paulistr_multiply("XYZ", "YZX") == (-1j, "ZXY")) # Product in three places => 1j^3 = -1j
        self.assertTrue(lt.paulistr_multiply("XYX", "YZZ") == (1j, "ZXY")) # Product in three places, but one is flipped => 1j^1 = 1j
        self.assertTrue(lt.paulistr_multiply("YZZ", "XYX") == (-1j, "ZXY")) # Product in three places, flipped other way => 1j^-1 = -1j
        self.assertTrue(lt.paulistr_multiply("XYZXI", "YZXYZ") == (1, "ZXYZZ")) # Product in four places => 1j^4 = 1
        self.assertTrue(lt.paulistr_multiply("XYZXY", "YZXYZ") == (1j, "ZXYZX")) # Product in five places => 1j^5 = 1j

        # Commutator
        self.assertTrue(lt.paulistr_commutator("X", "Y") == (2j, "Z")) # Weight-1 commutator
        self.assertTrue(lt.paulistr_commutator("Y", "X") == (-2j, "Z")) # Weight-1 commutator
        self.assertTrue(lt.paulistr_commutator("I", "Z") == (0, "Z")) # Commutes with I
        self.assertTrue(lt.paulistr_commutator("YZ", "YX") == (2j, "IY")) # One Pauli difference
        self.assertTrue(lt.paulistr_commutator("XY", "ZI") == (-2j, "YY")) # Single I formula
        self.assertTrue(lt.paulistr_commutator("XI", "ZI") == (-2j, "YI")) # I in single place in both strings
        self.assertTrue(lt.paulistr_commutator("XX", "II") == (0, "XX")) # II commutes...
        self.assertTrue(lt.paulistr_commutator("IX", "XI") == (0, "XX")) # ...so does I in both places
        self.assertTrue(lt.paulistr_commutator("XYZ", "XZX") == (0, "IXY")) # Difference in two places, commutes
        self.assertTrue(lt.paulistr_commutator("XYZ", "YZX") == (-2j, "ZXY")) # Difference in three places => 2 * 1j^3 = -2j
        self.assertTrue(lt.paulistr_commutator("XYX", "YZZ") == (2j, "ZXY")) # Difference in three places, but one is flipped => 2 * 1j^1 = 2j
        self.assertTrue(lt.paulistr_commutator("YZZ", "XYX") == (-2j, "ZXY")) # Difference in three places, flipped other way => 2 * 1j^-1 = -2j
        self.assertTrue(lt.paulistr_commutator("XYZXY", "YZXYZ") == (2j, "ZXYZX")) # Difference in five places => 2 * 1j^5 = 2j

        # Anticommutator
        self.assertTrue(lt.paulistr_anticommutator("Z", "Z") == (2, "I")) # Weight-1 anticommutator
        self.assertTrue(lt.paulistr_anticommutator("Z", "I") == (2, "Z")) # Weight-1 anticommutator
        self.assertTrue(lt.paulistr_anticommutator("X", "Y") == (0, "Z")) # and those that commute do not anticommute
        self.assertTrue(lt.paulistr_anticommutator("XX", "II") == (2, "XX")) # Anticommutations for X 
        self.assertTrue(lt.paulistr_anticommutator("XX", "XI") == (2, "IX")) 
        self.assertTrue(lt.paulistr_anticommutator("XX", "XX") == (2, "II")) 
        self.assertTrue(lt.paulistr_anticommutator("IX", "XI") == (2, "XX"))
        self.assertTrue(lt.paulistr_anticommutator("YZ", "YX") == (0, "IY")) # One Pauli difference, does not anticommute
        self.assertTrue(lt.paulistr_anticommutator("XY", "ZI") == (0, "YY")) # Commutes so does not anticommute
        self.assertTrue(lt.paulistr_anticommutator("XI", "ZI") == (0, "YI"))

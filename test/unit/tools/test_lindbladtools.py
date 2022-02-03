import numpy as np
import scipy.sparse as sps

from pygsti.tools import lindbladtools as lt
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
        spL = lt.create_elementary_errorgen('H', sparse, True)
        self.assertArraysAlmostEqual(spL.toarray(),
                                     expectedLindbladian)

    def test_stochastic_lindbladian(self):
        a = np.array([[1, 2], [3, 4]])
        expected = 2.0 * np.array([
            [ 1,  2,  2,  4],
            [ 3,  4,  6,  8],
            [ 3,  6,  4,  8],
            [ 9, 12, 12, 16]
        ])
        self.assertArraysAlmostEqual(
            lt.create_elementary_errorgen('S', a),
            expected)
        sparse = sps.csr_matrix(a)
        spL = lt.create_elementary_errorgen('S', sparse, True)
        self.assertArraysAlmostEqual(spL.toarray(),
                                     expected)

    def test_nonham_lindbladian(self):
        a = np.array([[1, 2], [3, 4]])
        b = np.array([[1, 2], [3, 4]])
        expected = 2.0 * np.array([
            [ -9,  -5,  -5,  4],
            [ -4, -11,   6,  1],
            [ -4,   6, -11,  1],
            [  9,   5,   5, -4]
        ])
        self.assertArraysAlmostEqual(lt.create_lindbladian_term_errorgen('O', a, b), expected)
        sparsea = sps.csr_matrix(a)
        sparseb = sps.csr_matrix(b)
        spL = lt.create_lindbladian_term_errorgen('O', sparsea, sparseb, True)
        self.assertArraysAlmostEqual(spL.toarray(),
                                     expected)

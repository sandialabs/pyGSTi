from ..testutils import BaseTestCase, compare_files, temp_files
import unittest
import numpy as np
import scipy
import scipy.sparse as sps
import pygsti

import pygsti.tools.lindbladtools as lt

class LindbladianToolsBaseTestCase(BaseTestCase):
    def test_hamiltonian_to_lindbladian(self):
        expectedLindbladian = np.array([[ 0,  0,  0,  0],
                                        [ 0,  0,  0,  0,],
                                        [ 0,  0,  0,  0,],
                                        [ 0,  0,  0,  0]]
                                       )

        self.assertArraysAlmostEqual(lt.hamiltonian_to_lindbladian(np.zeros(shape=(2,2))),
                                     expectedLindbladian)
        sparse = sps.csr_matrix(np.zeros(shape=(2,2)))
        spL = lt.hamiltonian_to_lindbladian(sparse, True)
        self.assertArraysAlmostEqual(spL.toarray(),
                                     expectedLindbladian)
                
        

    def test_stochastic_lindbladian(self):
        a = np.array([[1,2],[3,4]])
        expected = np.array(
            [[ 1,  2,  2,  4],
             [ 3,  4,  6,  8],
             [ 3,  6,  4,  8],
             [ 9, 12, 12, 16]])
        self.assertArraysAlmostEqual(
            lt.stochastic_lindbladian(a),
            expected)
        sparse = sps.csr_matrix(a)
        spL = lt.stochastic_lindbladian(sparse, True)
        self.assertArraysAlmostEqual(spL.toarray(),
                                     expected)
                


    def test_nonham_lindbladian(self):
        a = np.array([[1,2],[3,4]])
        b = np.array([[1,2],[3,4]])
        expected = np.array(
                [[ -9,  -5,  -5,  4],
                 [ -4, -11,   6,  1],
                 [ -4,   6, -11,  1],
                 [  9,   5,   5, -4]])
        self.assertArraysAlmostEqual(
                lt.nonham_lindbladian(a, b),
                expected)
        sparsea = sps.csr_matrix(a)
        sparseb = sps.csr_matrix(b)
        spL = lt.nonham_lindbladian(sparsea, sparseb, True)
        self.assertArraysAlmostEqual(spL.toarray(),
                                     expected)

        


if __name__ == '__main__':
    unittest.main(verbosity=2)

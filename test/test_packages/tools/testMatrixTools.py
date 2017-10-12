from ..testutils import BaseTestCase, compare_files, temp_files
import numpy as np
import pygsti
import unittest

import pygsti.tools.matrixtools as mt


class MatrixBaseTestCase(BaseTestCase):

    def test_matrixtools(self):
        herm_mx = np.array( [[ 1, 1+2j],
                             [1-2j, 3]], 'complex' )
        non_herm_mx = np.array( [[ 1, 4+2j],
                                 [1+2j, 3]], 'complex' )
        self.assertTrue( pygsti.is_hermitian(herm_mx) )
        self.assertFalse( pygsti.is_hermitian(non_herm_mx) )

        pos_mx = np.array( [[ 4, 0.2],
                             [0.1, 3]], 'complex' )
        non_pos_mx = np.array( [[ 0, 1],
                                 [1, 0]], 'complex' )
        self.assertTrue( pygsti.is_pos_def(pos_mx) )
        self.assertFalse( pygsti.is_pos_def(non_pos_mx) )

        density_mx = np.array( [[ 0.9,   0],
                                [   0, 0.1]], 'complex' )
        non_density_mx = np.array( [[ 2.0, 1.0],
                                    [-1.0,   0]], 'complex' )
        self.assertTrue( pygsti.is_valid_density_mx(density_mx) )
        self.assertFalse( pygsti.is_valid_density_mx(non_density_mx) )

        s1 = pygsti.mx_to_string(density_mx)
        s2 = pygsti.mx_to_string(non_herm_mx)

    def test_all(self):
        a = np.array([[1,1], [1,1]])
        print("Nullspace = ",mt.nullspace(a))
        expected = np.array(
            [[ 0.70710678],
             [-0.70710678]] )

        diff1 = np.linalg.norm(mt.nullspace(a) - expected)
        diff2 = np.linalg.norm(mt.nullspace(a) + expected) # -1*expected is OK too (just an eigenvector)
        self.assertTrue( np.isclose(diff1,0) or np.isclose(diff2,0) )


        diff1 = np.linalg.norm(mt.nullspace_qr(a) - expected)
        diff2 = np.linalg.norm(mt.nullspace_qr(a) + expected) # -1*expected is OK too (just an eigenvector)
        self.assertTrue( np.isclose(diff1,0) or np.isclose(diff2,0) )

        mt.print_mx(a)

        b = np.array([[1,2],[3,4]],dtype='complex')
        with self.assertRaises(ValueError): 
            mt.real_matrix_log(b)
        with self.assertRaises(AssertionError): 
            mt.real_matrix_log(a)

if __name__ == '__main__':
    unittest.main(verbosity=2)

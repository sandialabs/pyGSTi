from ..testutils import BaseTestCase, compare_files, temp_files
import numpy as np
import pygsti
import unittest


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
        #print "\n>%s<" % s1
        #print "\n>%s<" % s2

        #TODO: go through matrixtools.py and add tests for every function

if __name__ == '__main__':
    unittest.main(verbosity=2)

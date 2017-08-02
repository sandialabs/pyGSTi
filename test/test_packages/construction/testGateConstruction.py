from ..testutils import BaseTestCase, compare_files, temp_files

import numpy as np
import unittest
import pygsti.construction as pc

class GateConstructionTestCase(BaseTestCase):

    def test_single_qubit_gate_matrix(self):
        expected = np.array([[1.00000000e+00, 2.77555756e-16, -2.28983499e-16, 0.00000000e+00],
                            [ -3.53885261e-16, -8.09667193e-01, 5.22395269e-01, -2.67473774e-01],
                            [ -3.92523115e-17, 5.22395269e-01, 8.49200550e-01, 7.72114534e-02],
                            [ 1.66533454e-16, 2.67473774e-01, -7.72114534e-02, -9.60466643e-01]]
                            )
        mx = pc.single_qubit_gate(24.0, 83.140134, 0.0000)
        self.assertArraysAlmostEqual(expected, mx)

    def test_two_qubit_gate_mx(self):
        gate = pc.two_qubit_gate()
        expected = np.array([
         [ 1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,],
         [ 0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,],
         [ 0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,],
         [ 0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,],
         [ 0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,],
         [ 0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,],
         [ 0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,],
         [ 0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,],
         [ 0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,],
         [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,],
         [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,],
         [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,],
         [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,],
         [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,],
         [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,],
         [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,]])
        self.assertArraysAlmostEqual(gate,expected)

if __name__ == '__main__':
    unittest.main(verbosity=2)

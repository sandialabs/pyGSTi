import unittest
import pygsti
import numpy as np

from  pygsti.objects import TPParameterizedGate

from ..testutils import BaseTestCase, compare_files, temp_files

class TPParameterizedGateTestCase(BaseTestCase):

    def setUp(self):
        super(TPParameterizedGateTestCase, self).setUp()
        self.gate = TPParameterizedGate([[1,0], [0,0]])

    def test_bad_first_row(self):
        with self.assertRaises(ValueError):
            gate = TPParameterizedGate([[0,0], [0,0]])

    def test_bad(self):
        gate = self.gate.copy()
        with self.assertRaises(ValueError):
            gate.set_value([])
        gate.set_value([[1, 0],[1, 2]])

        gate.dim = 'adfadsflkj'
        with self.assertRaises(TypeError):
            gate.set_value([[1, 0],[1, 2]])
        with self.assertRaises(ValueError):
            gate.set_value([[[1, 0]],[], []])

    def test_to_vector(self):
        self.assertArraysAlmostEqual(self.gate.to_vector(), np.array([0,0]))

    def test_from_vector(self):
        gatecopy = self.gate.copy()
        self.gate.from_vector(np.array([0, 0]))
        self.assertArraysAlmostEqual(gatecopy, self.gate)

    def test_deriv_wrt_params(self):
        self.assertArraysAlmostEqual(self.gate.deriv_wrt_params(0), np.array([0,0,1,0]))

if __name__ == '__main__':
    unittest.main(verbosity=2)

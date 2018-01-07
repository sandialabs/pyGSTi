import unittest
import pygsti
import numpy as np

from  pygsti.objects import FullyParameterizedGate

from ..testutils import BaseTestCase, compare_files, temp_files

class FullyParameterizedGateTestCase(BaseTestCase):

    def setUp(self):
        super(FullyParameterizedGateTestCase, self).setUp()
        self.gate = FullyParameterizedGate([[0,0],[0,0]])


    def test_bad(self):
        gate = self.gate.copy()
        with self.assertRaises(ValueError):
            gate.set_value([])
        gate.set_value([[1, 2],[1, 2]])

        gate.dim = 'adfadsflkj'
        with self.assertRaises(TypeError):
            gate.set_value([[1, 2],[1, 2]])
        with self.assertRaises(ValueError):
            gate.set_value([[[1, 2]],[], []])


    def test_derive_wrt_params(self):
        self.assertArraysAlmostEqual(self.gate.deriv_wrt_params(0), np.array([1, 0, 0, 0]))

    def test_str(self):
        str(self.gate)

    def test_compose(self):
        gate = self.gate.compose(self.gate)

if __name__ == '__main__':
    unittest.main(verbosity=2)

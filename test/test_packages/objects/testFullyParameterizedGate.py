import unittest
import pygsti
import numpy as np

from  pygsti.objects import FullyParameterizedGate

from ..testutils import BaseTestCase, compare_files, temp_files

class FullyParameterizedGateTestCase(BaseTestCase):

    def setUp(self):
        super(FullyParameterizedGateTestCase, self).setUp()
        self.gate = FullyParameterizedGate([[0,0], [0,0]])

    def test_bad(self):
        gate = self.gate.copy()
        with self.assertRaises(ValueError):
            gate.set_matrix([])
        gate.set_matrix([[1, 2],[1, 2]])

        gate.dim = 'adfadsflkj'
        with self.assertRaises(TypeError):
            gate.set_matrix([[1, 2],[1, 2]])
        with self.assertRaises(ValueError):
            gate.set_matrix([[[1, 2]],[], []])



if __name__ == '__main__':
    unittest.main(verbosity=2)

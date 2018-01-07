import unittest
import pygsti
import numpy as np

from  pygsti.objects import StaticGate

from ..testutils import BaseTestCase, compare_files, temp_files

class StaticGateTestCase(BaseTestCase):

    def setUp(self):
        super(StaticGateTestCase, self).setUp()
        self.gate = StaticGate([[0,0], [0,0]])

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

        self.assertEqual(gate.num_params(), 0)
        self.assertArraysAlmostEqual(gate.to_vector(), np.array([], 'd'))
        with self.assertRaises(AssertionError):
            gate.from_vector([1])

    def test_deriv_wrt_params(self):
        self.assertArraysAlmostEqual(self.gate.deriv_wrt_params(), np.array([]))

    def test_transform(self):
        with self.assertRaises(ValueError):
            elT = pygsti.objects.FullGaugeGroupElement([[1,0],[0,1]])
            self.gate.transform(elT) # can't transform a static gate - no params!

    def test_compose(self):
        self.gate.compose(self.gate)

    def test_str(self):
        str(self.gate)

    def test_reduce(self):
        self.gate.__reduce__()











if __name__ == '__main__':
    unittest.main(verbosity=2)

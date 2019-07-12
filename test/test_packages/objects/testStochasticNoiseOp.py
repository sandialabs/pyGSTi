import unittest
import pygsti
import numpy as np

from ..testutils import BaseTestCase, compare_files, temp_files

class StochasticOpTestCase(BaseTestCase):

    def setUp(self):
        super(StochasticOpTestCase, self).setUp()

    def test_stochastic_noise_op(self):
        rho = pygsti.construction.build_vector([4], ['Q0'], "0", 'pp')
        sop = pygsti.obj.StochasticNoiseOp(4)
        sop.from_vector(np.array([0.1, 0.0, 0.0]))
        self.assertArraysAlmostEqual( sop.to_vector(), np.array([0.1, 0., 0.]))
        expected_mx = np.identity(4); expected_mx[2,2] = expected_mx[3,3] = 0.98 # = 2*(0.1^2)
        self.assertArraysAlmostEqual(sop.todense(), expected_mx)
        self.assertAlmostEqual( float(np.dot(rho.T,np.dot(sop.todense(),rho))), 0.99) # b/c X dephasing w/rate is 0.1^2 = 0.01

    def test_depol_noise_op(self):
        rho = pygsti.construction.build_vector([4], ['Q0'], "0", 'pp')
        dop = pygsti.obj.DepolarizeOp(4)
        dop.from_vector(np.array([0.1]))
        self.assertArraysAlmostEqual(dop.to_vector(), np.array([0.1]))
        expected_mx = np.identity(4); expected_mx[1,1] = expected_mx[2,2] = expected_mx[3,3] = 0.96 # = 4*(0.1^2)
        self.assertArraysAlmostEqual(dop.todense(), expected_mx)
        self.assertAlmostEqual( float(np.dot(rho.T,np.dot(dop.todense(),rho))), 0.98 ) # b/c both X and Y dephasing rates => 0.01 reduction

if __name__ == '__main__':
    unittest.main(verbosity=2)


import unittest
from .algorithmsTestCase import AlgorithmTestCase
import numpy as np

class TestAlgorithmMethods(AlgorithmTestCase):

    def test_strict(self):
        #test strict mode, which forbids all these accesses
        with self.assertRaises(KeyError):
            self.gs_target_noisy['identity'] = [1,0,0,0]
        with self.assertRaises(KeyError):
            self.gs_target_noisy['Gx'] = np.identity(4,'d')
        with self.assertRaises(KeyError):
            self.gs_target_noisy['E0'] = [1,0,0,0]
        with self.assertRaises(KeyError):
            self.gs_target_noisy['rho0'] = [1,0,0,0]

        with self.assertRaises(KeyError):
            x = self.gs_target_noisy['identity']
        with self.assertRaises(KeyError):
            x = self.gs_target_noisy['Gx']
        with self.assertRaises(KeyError):
            x = self.gs_target_noisy['E0']
        with self.assertRaises(KeyError):
            x = self.gs_target_noisy['rho0']





if __name__ == '__main__':
    unittest.main(verbosity=2)

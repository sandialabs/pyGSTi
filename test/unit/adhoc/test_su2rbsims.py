import unittest
import numpy as np
import scipy.linalg as la

from pygsti.tools.su2tools import Spin72, SU2
from pygsti.adhoc.su2rbsims import SU2CharacterRBDesign, SU2CharacterRBSim, SU2RBDesign, SU2RBSim, default_povm


class TestSU2RBSim(unittest.TestCase):

    ABSTOL = 1e-12

    def test_smoke_spin72(self):
        lengths = np.arange(1, 16, 3)
        N = 10
        rbd = SU2RBDesign(Spin72, N, lengths, default_povm(8,'std'), default_povm(8,'std'), seed=0)
        rbm = SU2RBSim(rbd)
        rbm.set_error_channel_gaussian(0.025)
        rbm.compute_probabilities()
        probs = rbm.probs

        self.assertTrue(np.all(probs >= -self.ABSTOL))
        self.assertTrue(np.all(probs <= 1 + self.ABSTOL))
        total_probs = np.sum(probs, axis=3)  # should be right about 1.
        self.assertTrue(np.all(np.abs(total_probs - 1) <= self.ABSTOL))
        return


class TestSU2CharacterRBSim(unittest.TestCase):

    ABSTOL = 1e-12

    def test_smoke_spin72(self):
        lengths = np.arange(1, 16, 3)
        N = 10

        rbd = SU2CharacterRBDesign(Spin72, N, lengths, default_povm(8,'std'), default_povm(8,'std'), seed=0)
        rbm = SU2CharacterRBSim(rbd)
        rbm.set_error_channel_gaussian(0.02)
        rbm.compute_probabilities()
        probs = rbm.probs

        self.assertTrue(np.all(probs >= -self.ABSTOL))
        self.assertTrue(np.all(probs <= 1 + self.ABSTOL))
        total_probs = np.sum(probs, axis=3)  # should be right about 1.
        self.assertTrue(np.all(np.abs(total_probs - 1) <= self.ABSTOL))
        return

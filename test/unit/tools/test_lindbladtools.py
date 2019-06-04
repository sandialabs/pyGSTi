import numpy as np

from ..util import BaseCase

from pygsti.tools import lindbladtools as lt


class LindbladToolsTester(BaseCase):
    def test_hamiltonian_to_lindbladian(self):
        expectedLindbladian = np.array([
            [ 0,  0,  0,  0],
            [ 0,  0,  0,  0],
            [ 0,  0,  0,  0],
            [ 0,  0,  0,  0]
        ])

        self.assertArraysAlmostEqual(lt.hamiltonian_to_lindbladian(np.zeros(shape=(2, 2))),
                                     expectedLindbladian)

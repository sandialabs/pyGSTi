from pygsti.construction import std1Q_XYI as std
import pygsti
import unittest

import numpy as np
import sys, os

from ..testutils import BaseTestCase, temp_files, compare_files

class GraspTestCase(BaseTestCase):

    def setUp(self):
        super(GraspTestCase, self).setUp()

    def test_grasp(self):
        import pygsti.algorithms.germselection as germsel

        threshold             = 1e6
        randomizationStrength = 1e-3
        neighborhoodSize      = 5
        gatesetNeighborhood   = germsel.randomizeGatesetList([std.gs_target],
                                  randomizationStrength=randomizationStrength,
                                  numCopies=neighborhoodSize, seed=2014)

        max_length   = 6
        gates        = std.gs_target.gates.keys()
        superGermSet = pygsti.construction.list_all_gatestrings_without_powers_and_cycles(gates, max_length)
        germsel.grasp_germ_set_optimization(gatesetList=gatesetNeighborhood, germsList=superGermSet,
                                            alpha=0.1, randomize=False, seed=2014, scoreFunc='all',
                                            threshold=threshold, verbosity=1, iterations=1,
                                            l1Penalty=1.0, returnAll=True)



if __name__ == '__main__':
    unittest.main(verbosity=2)

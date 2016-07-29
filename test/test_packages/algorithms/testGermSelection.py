from pygsti.construction import std1Q_XYI as std
import pygsti

import numpy as np
import sys, os

from .algorithmsTestCase import AlgorithmTestCase

class GermSelectionTestCase(AlgorithmTestCase):

    def test_germSelection(self):
        germsToTest = pygsti.construction.list_all_gatestrings_without_powers_and_cycles(
            list(std.gs_target.gates.keys()), 2)


        bSuccess, eigvals_finiteL = pygsti.alg.test_germ_list_finitel(
            self.gs_target_noisy, germsToTest, L=16, returnSpectrum=True, tol=1e-3)
        self.assertFalse(bSuccess)

        bSuccess,eigvals_infiniteL = pygsti.alg.test_germ_list_infl(
            self.gs_target_noisy, germsToTest, returnSpectrum=True, check=True)
        self.assertFalse(bSuccess)

        germsToTest = pygsti.construction.list_all_gatestrings_without_powers_and_cycles(
            list(std.gs_target.gates.keys()), 3)

        germsToTest2 = pygsti.construction.list_all_gatestrings_without_powers_and_cycles(
            list(std.gs_target.gates.keys()), 4) + std.germs

        finalGerms = pygsti.alg.optimize_integer_germs_slack(
            self.gs_target_noisy, germsToTest, initialWeights=None,
            fixedSlack=0.1, slackFrac=False, returnAll=False, tol=1e-6, verbosity=4)

        finalGerms, wts, scoreDict = pygsti.alg.optimize_integer_germs_slack(
            self.gs_target_noisy, germsToTest2, initialWeights=np.ones( len(germsToTest2), 'd' ),
            fixedSlack=False, slackFrac=0.1, returnAll=True, tol=1e-6, verbosity=4)

        self.runSilent(pygsti.alg.optimize_integer_germs_slack,
                       self.gs_target_noisy, germsToTest,
                       initialWeights=np.ones( len(germsToTest), 'd' ),
                       fixedSlack=False, slackFrac=0.1,
                       returnAll=True, tol=1e-6, verbosity=4, maxIter=1)
                       # test hitting max iterations

        with self.assertRaises(ValueError):
            pygsti.alg.optimize_integer_germs_slack(
                self.gs_target_noisy, germsToTest,
                initialWeights=np.ones( len(germsToTest), 'd' ),
                returnAll=True, tol=1e-6, verbosity=4)
                # must specify either fixedSlack or slackFrac

import unittest
import pygsti
from pygsti.construction import std1Q_XYI as std

import numpy as np
import sys

from .algorithmsTestCase import AlgorithmTestCase

class FiducialSelectionTestCase(AlgorithmTestCase):
    def test_fiducialSelection(self):

        prepFidList = pygsti.alg.optimize_integer_fiducials_slack(
            std.gs_target, std.fiducials, prepOrMeas = "prep",
            initialWeights=None, maxIter=100,
            fixedSlack=False, slackFrac=0.1,
            returnAll=False, verbosity=4)

        measFidList, wts, scoredict = pygsti.alg.optimize_integer_fiducials_slack(
            std.gs_target, std.fiducials, prepOrMeas = "meas",
            initialWeights=np.ones( len(std.fiducials), 'i' ), maxIter=100,
            fixedSlack=0.1, slackFrac=False,
            returnAll=True, verbosity=4)


        fiducials_to_try = pygsti.construction.list_all_gatestrings(list(std.gs_target.gates.keys()), 0, 2)
        prepFidList2 = pygsti.alg.optimize_integer_fiducials_slack(
            std.gs_target, fiducials_to_try, prepOrMeas = "prep",
            initialWeights=None, scoreFunc='worst', maxIter=100,
            fixedSlack=False, slackFrac=0.1,
            returnAll=False, verbosity=4)

        prepFidList3 = pygsti.alg.optimize_integer_fiducials_slack(
            std.gs_target, fiducials_to_try, prepOrMeas = "prep",
            initialWeights=None, scoreFunc='all', maxIter=100,
            fixedSlack=False, slackFrac=0.1, fixedNum=4,
            returnAll=False, verbosity=4)
        pygsti.alg.build_bitvec_mx(3,1)

        prepFidList4 = pygsti.alg.optimize_integer_fiducials_slack(
            std.gs_target, fiducials_to_try, prepOrMeas = "prep",
            initialWeights=None, scoreFunc='all', maxIter=100,
            fixedSlack=False, slackFrac=0.1, fixedNum=4, forceEmpty=False,
            returnAll=True, verbosity=4) #fixedNum with forceEmpty=False (& returnAll=True for more coverage)

        self.runSilent(pygsti.alg.optimize_integer_fiducials_slack,
            std.gs_target, fiducials_to_try, prepOrMeas = "prep",
            initialWeights=None, maxIter=1,
            fixedSlack=False, slackFrac=0.1,
            returnAll=False, verbosity=4) #check max iterations

        insuff_fids = pygsti.construction.gatestring_list([('Gx',)])
        ret = self.runSilent(pygsti.alg.optimize_integer_fiducials_slack,
            std.gs_target, insuff_fids, prepOrMeas = "prep",
            initialWeights=np.ones( len(insuff_fids), 'i' ), maxIter=100,
            fixedSlack=0.1, slackFrac=False,
            returnAll=True, verbosity=4)
        self.assertTrue(ret is None) # insufficient fiducials -> returns None


        with self.assertRaises(ValueError):
            pygsti.alg.optimize_integer_fiducials_slack(
            std.gs_target, std.fiducials, prepOrMeas = "meas") #neither fixedSlack nor slackFrac given

        with self.assertRaises(Exception):
            pygsti.alg.optimize_integer_fiducials_slack(
                std.gs_target, std.fiducials, fixedSlack=0.1) #invalid (or missing) prepOrMeas


        print("prepFidList = ",prepFidList)
        print("measFidList = ",measFidList)
        print("wts = ",wts)
        print("scoredict = ",scoredict)

        self.assertTrue(pygsti.alg.test_fiducial_list(
                std.gs_target,prepFidList,"prep",
                scoreFunc='all',returnAll=False))

        self.assertTrue(pygsti.alg.test_fiducial_list(
                std.gs_target,measFidList,"meas",
                scoreFunc='worst',returnAll=False))

        bResult, spectrum, score = pygsti.alg.test_fiducial_list(
            std.gs_target,measFidList,"meas",
            scoreFunc='all',returnAll=True)

        with self.assertRaises(Exception):
            pygsti.alg.test_fiducial_list(
            std.gs_target,measFidList,"foobar",
            scoreFunc='all',returnAll=False)

    def test_grasp_fidsel(self):
        prepFidList = pygsti.alg.grasp_fiducial_optimization(
            std.gs_target, std.fiducials, prepOrMeas = "prep",
            alpha = 0.0, verbosity=4)
        
        measFidList = pygsti.alg.grasp_fiducial_optimization(
            std.gs_target, std.fiducials, prepOrMeas = "meas",
            alpha = 1.0, verbosity=4)
        
        with self.assertRaises(ValueError):
            pygsti.alg.grasp_fiducial_optimization(
            std.gs_target, std.fiducials, prepOrMeas = "foobar",
            alpha = 0.5, verbosity=4)
        

if __name__ == '__main__':
    unittest.main(verbosity = 2)

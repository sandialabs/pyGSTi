import unittest
import pygsti
from pygsti.construction import std1Q_XYI as std

import numpy as np
from scipy import polyfit
import sys

class AlgorithmTestCase(unittest.TestCase):

    def setUp(self):
        #Set GateSet objects to "strict" mode for testing
        pygsti.objects.GateSet._strict = True

        self.gs_target_noisy = std.gs_target.randomize_with_unitary(0.001, seed=1234)

    def runSilent(self, callable, *args, **kwds):
        orig_stdout = sys.stdout
        sys.stdout = open("temp_test_files/silent.txt","w")
        result = callable(*args, **kwds)
        sys.stdout.close()
        sys.stdout = orig_stdout
        return result


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


        fiducials_to_try = pygsti.construction.list_all_gatestrings(std.gs_target.gates.keys(), 0, 2)
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
        pygsti.alg.write_fixed_hamming_weight_code(3,1)

        self.runSilent(pygsti.alg.optimize_integer_fiducials_slack,
            std.gs_target, fiducials_to_try, prepOrMeas = "prep",
            initialWeights=None, maxIter=1, 
            fixedSlack=False, slackFrac=0.1, 
            returnAll=False, verbosity=4) #check max iterations

        with self.assertRaises(ValueError):
            pygsti.alg.optimize_integer_fiducials_slack(
            std.gs_target, std.fiducials, prepOrMeas = "meas") #neither fixedSlack nor slackFrac given

        print "prepFidList = ",prepFidList
        print "measFidList = ",measFidList
        print "wts = ",wts
        print "scoredict = ",scoredict

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


    def test_fiducialPairReduction(self):
        self.runSilent(pygsti.alg.find_sufficient_fiducial_pairs,
                       std.gs_target, std.fiducials, std.fiducials,
                       std.germs, testPairList=[(0,0),(0,1),(1,0)], verbosity=4)
        
        suffPairs = self.runSilent(pygsti.alg.find_sufficient_fiducial_pairs,
            std.gs_target, std.fiducials, std.fiducials, std.germs, verbosity=4)

        small_fiducials = pygsti.construction.gatestring_list([('Gx',)])
        small_germs = pygsti.construction.gatestring_list([('Gx',),('Gy',)])
        self.runSilent(pygsti.alg.find_sufficient_fiducial_pairs,
                       std.gs_target, small_fiducials, small_fiducials,
                       small_germs, searchMode="sequential", verbosity=2)

        self.runSilent(pygsti.alg.find_sufficient_fiducial_pairs,
                       std.gs_target, std.fiducials, std.fiducials,
                       std.germs, searchMode="random", nRandom=3,
                       seed=1234, verbosity=2)
        self.runSilent(pygsti.alg.find_sufficient_fiducial_pairs,
                       std.gs_target, std.fiducials, std.fiducials,
                       std.germs, searchMode="random", nRandom=300,
                       seed=1234, verbosity=2)
        
        self.assertEqual(suffPairs, [(0, 0), (0, 1), (1, 0)])


    def test_germSelection(self):
        germsToTest = pygsti.construction.list_all_gatestrings_without_powers_and_cycles(
            std.gs_target.gates.keys(), 2)

        
        bSuccess, eigvals_finiteL = pygsti.alg.test_germ_list_finitel(
            self.gs_target_noisy, germsToTest, L=16, returnSpectrum=True, tol=1e-3)
        self.assertFalse(bSuccess)

        bSuccess,eigvals_infiniteL = pygsti.alg.test_germ_list_infl(
            self.gs_target_noisy, germsToTest, returnSpectrum=True, check=True)
        self.assertFalse(bSuccess)

        germsToTest = pygsti.construction.list_all_gatestrings_without_powers_and_cycles(
            std.gs_target.gates.keys(), 3)

        germsToTest2 = pygsti.construction.list_all_gatestrings_without_powers_and_cycles(
            std.gs_target.gates.keys(), 4)

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

        
    
      
if __name__ == "__main__":
    unittest.main(verbosity=2)

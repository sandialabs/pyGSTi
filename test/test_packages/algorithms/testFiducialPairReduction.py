import unittest
import pygsti
from pygsti.construction import std1Q_XYI as std
from ..testutils import compare_files, temp_files

import numpy as np
import pickle

from .algorithmsTestCase import AlgorithmTestCase

class FiducialPairReductionTestCase(AlgorithmTestCase):
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

        self.assertEqual(suffPairs, [(0, 0), (0, 1), (0, 2)])

    def test_memlimit(self):
        # A very low memlimit
        pygsti.alg.find_sufficient_fiducial_pairs(std.gs_target, std.fiducials, std.fiducials,
                                                  std.germs, testPairList=[(0,0),(0,1),(1,0)],
                                                  verbosity=0, memLimit=8192)
        # A significantly higher one
        pygsti.alg.find_sufficient_fiducial_pairs(std.gs_target, std.fiducials, std.fiducials,
                                                  std.germs, testPairList=[(0,0),(0,1),(1,0)],
                                                  verbosity=0, memLimit=128000)


    def test_intelligentFiducialPairReduction(self):

        prepStrs = std.fiducials
        effectStrs = std.fiducials
        germList = std.germs
        targetGateset = std.gs_target

        fidPairs = self.runSilent(
            pygsti.alg.find_sufficient_fiducial_pairs_per_germ,
                       std.gs_target, std.fiducials, std.fiducials,
                       std.germs, prepovmTuples="first",
                       searchMode="sequential",
                       constrainToTP=True,
                       nRandom=100, seed=None, verbosity=3,
                       memLimit=None)

        vs = self.versionsuffix
        cmpFilenm = compare_files + "/IFPR_fidPairs_dict%s.pkl" % vs
        #Uncomment to save reference fidPairs dictionary
        #with open(cmpFilenm,"wb") as pklfile:
        #    pickle.dump(fidPairs, pklfile)

        with open(cmpFilenm,"rb") as pklfile:
            fidPairs_cmp = pickle.load(pklfile)

        #On other machines (eg TravisCI) these aren't equal, due to randomness, so don't test
        #self.assertEqual(fidPairs, fidPairs_cmp)

        #test out some additional code paths: mem limit, random mode, & no good pair list
        fidPairs2 = self.runSilent(
            pygsti.alg.find_sufficient_fiducial_pairs_per_germ,
                       std.gs_target, std.fiducials, std.fiducials,
                       std.germs, prepovmTuples="first",
                       searchMode="random",
                       constrainToTP=True,
                       nRandom=3, seed=None, verbosity=3,
                       memLimit=1024*10)

        fidPairs3 = self.runSilent( #larger nRandom
            pygsti.alg.find_sufficient_fiducial_pairs_per_germ,
                       std.gs_target, std.fiducials, std.fiducials,
                       std.germs, prepovmTuples="first",
                       searchMode="random",
                       constrainToTP=True,
                       nRandom=100, seed=None, verbosity=3,
                       memLimit=1024*10)

        fidPairs3b = self.runSilent( #huge nRandom (should cap to all pairs)
            pygsti.alg.find_sufficient_fiducial_pairs_per_germ,
                       std.gs_target, std.fiducials, std.fiducials,
                       std.germs, prepovmTuples="first",
                       searchMode="random",
                       constrainToTP=True,
                       nRandom=1000000, seed=None, verbosity=3,
                       memLimit=1024*10)


        insuff_fids = pygsti.construction.gatestring_list([('Gx',)])
        with self.assertRaises(ValueError):
            fidPairs4 = self.runSilent( #insufficient fiducials
                pygsti.alg.find_sufficient_fiducial_pairs_per_germ,
                std.gs_target, insuff_fids, insuff_fids,
                std.germs, prepovmTuples="first",
                searchMode="random",
                constrainToTP=True,
                nRandom=100, seed=None, verbosity=3,
                memLimit=1024*10)

    def test_FPR_test_pairs(self):
        gs_target = std.gs_target
        prep_fiducials = std.fiducials
        meas_fiducials = std.fiducials
        germs = std.germs
        maxLengths = [1,2,4,8,16]
        
        gateLabels = list(gs_target.gates.keys())
        
        fidPairs = pygsti.alg.find_sufficient_fiducial_pairs(
            gs_target, prep_fiducials, meas_fiducials, germs,
            searchMode="random", nRandom=100, seed=1234,
            verbosity=1, memLimit=int(2*(1024)**3), minimumPairs=2)
        
        # fidPairs is a list of (prepIndex,measIndex) 2-tuples, where
        # prepIndex indexes prep_fiducials and measIndex indexes meas_fiducials
        print("Global FPR says we only need to keep the %d pairs:\n %s\n"
              % (len(fidPairs),fidPairs))
        
        nAmplified = pygsti.alg.test_fiducial_pairs(fidPairs, gs_target, prep_fiducials,
                                                    meas_fiducials, germs,
                                                    verbosity=3, memLimit=None)
        
        #Note: can't amplify SPAM params, so don't count them
        nTotal = pygsti.alg.removeSPAMVectors(gs_target).num_nongauge_params()
        self.assertEqual(nTotal, 34)
        
        print("GFPR: %d AMPLIFIED out of %d total (non-spam non-gauge) params" % (nAmplified, nTotal))
        self.assertEqual(nAmplified, 34)
        
        fidPairsDict = pygsti.alg.find_sufficient_fiducial_pairs_per_germ(
            gs_target, prep_fiducials, meas_fiducials, germs,
            searchMode="random", constrainToTP=True,
            nRandom=100, seed=1234, verbosity=1,
            memLimit=int(2*(1024)**3))
        
        nAmplified = pygsti.alg.test_fiducial_pairs(fidPairsDict, gs_target, prep_fiducials,
                                         meas_fiducials, germs,
                                         verbosity=3, memLimit=None)
        
        print("PFPR: %d AMPLIFIED out of %d total (non-spam non-gauge) params" % (nAmplified, nTotal))
        self.assertEqual(nAmplified, 34)




if __name__ == '__main__':
    unittest.main(verbosity=2)

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

        self.assertEqual(suffPairs, [(0, 0), (0, 1), (1, 0)])

    def test_memlimit(self):
        # A very low memlimit
        pygsti.alg.find_sufficient_fiducial_pairs(std.gs_target, std.fiducials, std.fiducials,
                                                  std.germs, testPairList=[(0,0),(0,1),(1,0)],
                                                  verbosity=0, memLimit=4096)
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
                       std.germs, spamLabels="all",
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

        self.assertEqual(fidPairs, fidPairs_cmp)

        #test out some additional code paths: mem limit, random mode, & no good pair list
        fidPairs2 = self.runSilent(
            pygsti.alg.find_sufficient_fiducial_pairs_per_germ,
                       std.gs_target, std.fiducials, std.fiducials,
                       std.germs, spamLabels="all",
                       searchMode="random",
                       constrainToTP=True,
                       nRandom=3, seed=None, verbosity=3,
                       memLimit=1024*10)

        fidPairs3 = self.runSilent( #larger nRandom
            pygsti.alg.find_sufficient_fiducial_pairs_per_germ,
                       std.gs_target, std.fiducials, std.fiducials,
                       std.germs, spamLabels="all",
                       searchMode="random",
                       constrainToTP=True,
                       nRandom=100, seed=None, verbosity=3,
                       memLimit=1024*10)

        insuff_fids = pygsti.construction.gatestring_list([('Gx',)])
        with self.assertRaises(ValueError):
            fidPairs4 = self.runSilent( #insufficient fiducials
                pygsti.alg.find_sufficient_fiducial_pairs_per_germ,
                std.gs_target, insuff_fids, insuff_fids,
                std.germs, spamLabels="all",
                searchMode="random",
                constrainToTP=True,
                nRandom=100, seed=None, verbosity=3,
                memLimit=1024*10)



if __name__ == '__main__':
    unittest.main(verbosity=2)

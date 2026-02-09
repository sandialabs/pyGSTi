import pickle
import unittest

import pygsti
from pygsti.algorithms import germselection
from pygsti.modelpacks import smq1Q_XYI as std
from .algorithmsTestCase import AlgorithmTestCase
from ..testutils import compare_files, regenerate_references


class FiducialPairReductionTestCase(AlgorithmTestCase):
    def test_memlimit(self):
        with self.assertRaises(MemoryError):
            # A very low memlimit
            pygsti.alg.find_sufficient_fiducial_pairs(std.target_model(), std.prep_fiducials(), std.meas_fiducials(),
                                                      std.germs(lite=True), test_pair_list=[(0,0),(0,1),(1,0)],
                                                      verbosity=0, mem_limit=100)  # 100 bytes!
    
    #Two out of the three tests that were in the following function were superfluous, and taking
    #n_random out to very large values takes a long time to run, so I don't think it is worth the time
    #from a testing standpoint.
#    def test_intelligentFiducialPairReduction(self):
#
#        #test out some additional code paths: random mode, very large n_random
#
#        fidPairs = self.runSilent( #huge n_random (should cap to all pairs)
#            pygsti.alg.find_sufficient_fiducial_pairs_per_germ,
#                       std.target_model(), std.prep_fiducials(), std.meas_fiducials(),
#                       std.germs(lite=True), prep_povm_tuples="first",
#                       search_mode="random",
#                       constrain_to_tp=True,
#                       n_random=1000000, seed=None, verbosity=0,
#                       mem_limit=1024*256)

    def test_FPR_test_pairs(self):
        target_model = std.target_model()
        prep_fiducials = std.prep_fiducials()
        meas_fiducials = std.meas_fiducials()
        germs = std.germs(lite = False)

        op_labels = list(target_model.operations.keys())

        fidPairs = pygsti.alg.find_sufficient_fiducial_pairs(
            target_model, prep_fiducials, meas_fiducials, germs,
            search_mode="random", n_random=10, seed=1234,
            verbosity=1, mem_limit=int(2*(1024)**3), minimum_pairs=2, 
            test_lengths = (64, 512))

        # fidPairs is a list of (prepIndex,measIndex) 2-tuples, where
        # prepIndex indexes prep_fiducials and measIndex indexes meas_fiducials
        print("Global FPR says we only need to keep the %d pairs:\n %s\n"
              % (len(fidPairs),fidPairs))

        nAmplified = pygsti.alg.test_fiducial_pairs(fidPairs, target_model, prep_fiducials,
                                                    meas_fiducials, germs,
                                                    verbosity=3, mem_limit=None, test_lengths=(64, 512),
                                                    tol = 0.5)

        #Note: can't amplify SPAM params, so don't count them

        nTotal = germselection._remove_spam_vectors(target_model).num_nongauge_params
        self.assertEqual(nTotal, 34)

        print("GFPR: %d AMPLIFIED out of %d total (non-spam non-gauge) params" % (nAmplified, nTotal))
        self.assertEqual(nAmplified, 34)

        fidPairsDict = pygsti.alg.find_sufficient_fiducial_pairs_per_germ(
            target_model, prep_fiducials, meas_fiducials, germs,
            search_mode="random", constrain_to_tp=True,
            n_random=10, seed=1234, verbosity=1,
            mem_limit=int(2*(1024)**3))

        nAmplified = pygsti.alg.test_fiducial_pairs(fidPairsDict, target_model, prep_fiducials,
                                                    meas_fiducials, germs,
                                                    verbosity=3, mem_limit=None,
                                                    test_lengths=(64, 512),
                                                    tol = 0.5)

        print("PFPR: %d AMPLIFIED out of %d total (non-spam non-gauge) params" % (nAmplified, nTotal))
        self.assertEqual(nAmplified, 34)

if __name__ == '__main__':
    unittest.main(verbosity=2)

import pickle
import unittest

import pygsti
from pygsti.algorithms import germselection
from pygsti.modelpacks.legacy import std1Q_XYI as std
from .algorithmsTestCase import AlgorithmTestCase
from ..testutils import compare_files, regenerate_references


class FiducialPairReductionTestCase(AlgorithmTestCase):
    def test_memlimit(self):
        with self.assertRaises(MemoryError):
            # A very low memlimit
            pygsti.alg.find_sufficient_fiducial_pairs(std.target_model(), std.fiducials, std.fiducials,
                                                      std.germs, test_pair_list=[(0,0),(0,1),(1,0)],
                                                      verbosity=0, mem_limit=100)  # 100 bytes!
        # A low memlimit
        pygsti.alg.find_sufficient_fiducial_pairs(std.target_model(), std.fiducials, std.fiducials,
                                                  std.germs, test_pair_list=[(0,0),(0,1),(1,0)],
                                                  verbosity=0, mem_limit=40 * 1024**2)  # 10MB
        # A higher limit
        pygsti.alg.find_sufficient_fiducial_pairs(std.target_model(), std.fiducials, std.fiducials,
                                                  std.germs, test_pair_list=[(0,0),(0,1),(1,0)],
                                                  verbosity=0, mem_limit=80 * 1024**2)  # 80MB


    def test_intelligentFiducialPairReduction(self):
        fidPairs = self.runSilent(
            pygsti.alg.find_sufficient_fiducial_pairs_per_germ,
                       std.target_model(), std.fiducials, std.fiducials,
                       std.germs, pre_povm_tuples="first",
                       search_mode="sequential",
                       constrain_to_tp=True,
                       n_random=100, seed=None, verbosity=3,
                       mem_limit=None)

        cmpFilenm = compare_files + "/IFPR_fidPairs_dict.pkl"
        # Run to SAVE reference fidPairs dictionary
        if regenerate_references():
            with open(cmpFilenm,"wb") as pklfile:
                pickle.dump(fidPairs, pklfile)

        with open(cmpFilenm,"rb") as pklfile:
            fidPairs_cmp = pickle.load(pklfile)

        #On other machines (eg TravisCI) these aren't equal, due to randomness, so don't test
        #self.assertEqual(fidPairs, fidPairs_cmp)

        #test out some additional code paths: mem limit, random mode, & no good pair list
        fidPairs2 = self.runSilent(
            pygsti.alg.find_sufficient_fiducial_pairs_per_germ,
                       std.target_model(), std.fiducials, std.fiducials,
                       std.germs, pre_povm_tuples="first",
                       search_mode="random",
                       constrain_to_tp=True,
                       n_random=3, seed=None, verbosity=3,
                       mem_limit=1024*256)

        fidPairs3 = self.runSilent( #larger n_random
            pygsti.alg.find_sufficient_fiducial_pairs_per_germ,
                       std.target_model(), std.fiducials, std.fiducials,
                       std.germs, pre_povm_tuples="first",
                       search_mode="random",
                       constrain_to_tp=True,
                       n_random=100, seed=None, verbosity=3,
                       mem_limit=1024*256)

        fidPairs3b = self.runSilent( #huge n_random (should cap to all pairs)
            pygsti.alg.find_sufficient_fiducial_pairs_per_germ,
                       std.target_model(), std.fiducials, std.fiducials,
                       std.germs, pre_povm_tuples="first",
                       search_mode="random",
                       constrain_to_tp=True,
                       n_random=1000000, seed=None, verbosity=3,
                       mem_limit=1024*256)

    def test_FPR_test_pairs(self):
        target_model = std.target_model()
        prep_fiducials = std.fiducials
        meas_fiducials = std.fiducials
        germs = std.germs
        maxLengths = [1,2,4,8,16]

        op_labels = list(target_model.operations.keys())

        fidPairs = pygsti.alg.find_sufficient_fiducial_pairs(
            target_model, prep_fiducials, meas_fiducials, germs,
            search_mode="random", n_random=100, seed=1234,
            verbosity=1, mem_limit=int(2*(1024)**3), minimum_pairs=2)

        # fidPairs is a list of (prepIndex,measIndex) 2-tuples, where
        # prepIndex indexes prep_fiducials and measIndex indexes meas_fiducials
        print("Global FPR says we only need to keep the %d pairs:\n %s\n"
              % (len(fidPairs),fidPairs))

        nAmplified = pygsti.alg.test_fiducial_pairs(fidPairs, target_model, prep_fiducials,
                                                    meas_fiducials, germs,
                                                    verbosity=3, mem_limit=None)

        #Note: can't amplify SPAM params, so don't count them

        nTotal = germselection._remove_spam_vectors(target_model).num_nongauge_params
        self.assertEqual(nTotal, 34)

        print("GFPR: %d AMPLIFIED out of %d total (non-spam non-gauge) params" % (nAmplified, nTotal))
        self.assertEqual(nAmplified, 34)

        fidPairsDict = pygsti.alg.find_sufficient_fiducial_pairs_per_germ(
            target_model, prep_fiducials, meas_fiducials, germs,
            search_mode="random", constrain_to_tp=True,
            n_random=100, seed=1234, verbosity=1,
            mem_limit=int(2*(1024)**3))

        nAmplified = pygsti.alg.test_fiducial_pairs(fidPairsDict, target_model, prep_fiducials,
                                                    meas_fiducials, germs,
                                                    verbosity=3, mem_limit=None)

        print("PFPR: %d AMPLIFIED out of %d total (non-spam non-gauge) params" % (nAmplified, nTotal))
        self.assertEqual(nAmplified, 34)




if __name__ == '__main__':
    unittest.main(verbosity=2)

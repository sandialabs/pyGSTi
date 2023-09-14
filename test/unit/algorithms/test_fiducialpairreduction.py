import pygsti.algorithms.fiducialpairreduction as fpr
from pygsti.algorithms.germselection import germ_set_spanning_vectors
import pygsti.circuits as pc
from pygsti.circuits import Circuit
import numpy as _np
from . import fixtures
from ..util import BaseCase

_SEED = 1234


class FiducialPairReductionStdData(object):
    def setUp(self):
        super(FiducialPairReductionStdData, self).setUp()
        self.model = fixtures.model
        self.preps = fixtures.fiducials
        self.effects = fixtures.fiducials
        self.germs = fixtures.germs
        self.fiducial_pairs = [(0, 0), (0, 1), (0, 2)]
        
        self.fiducial_pairs_per_germ= {
        Circuit(('Gi',)): [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)], 
        Circuit(('Gx',)): [(0, 0), (0, 1), (2, 2)], 
        Circuit(('Gy',)): [(0, 0), (0, 2), (1, 1)], 
        Circuit(('Gx','Gy')): [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)], 
        Circuit(('Gx','Gx','Gy')): [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1)], 
        Circuit(('Gx','Gy','Gy')): [(0, 0), (0, 1), (0, 2), (2, 0), (2, 2)], 
        Circuit(('Gx','Gy','Gi')): [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)], 
        Circuit(('Gx','Gi','Gy')): [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)], 
        Circuit(('Gx','Gi','Gi')): [(0, 0), (0, 1), (2, 2)], 
        Circuit(('Gy','Gi','Gi')): [(0, 0), (0, 2), (1, 1)], 
        Circuit(('Gx','Gy','Gy','Gi')): [(0, 0), (0, 1), (0, 2), (2, 0), (2, 2)], 
        Circuit(('Gx','Gx','Gy','Gx','Gy','Gy')): [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)]
        }
        
        self.fiducial_pairs_per_germ_windows_38= {
        Circuit(('Gi',)): [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)],
        Circuit(('Gx',)): [(0, 0), (0, 1), (0, 2)],
        Circuit(('Gy',)): [(0, 0), (0, 1), (0, 2)],
        Circuit(('Gx','Gy')): [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],
        Circuit(('Gx','Gx','Gy')): [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1)],
        Circuit(('Gx','Gy','Gy')): [(0, 0), (0, 1), (0, 2), (2, 0), (2, 2)],
        Circuit(('Gx','Gy','Gi')): [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],
        Circuit(('Gx','Gi','Gy')): [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],
        Circuit(('Gx','Gi','Gi')): [(0, 0), (0, 1), (0, 2)],
        Circuit(('Gy','Gi','Gi')): [(0, 0), (0, 1), (0, 2)],
        Circuit(('Gx','Gy','Gy','Gi')): [(0, 0), (0, 1), (0, 2), (2, 0), (2, 2)],
        Circuit(('Gx','Gx','Gy','Gx','Gy','Gy')): [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)]
        }
        
class FiducialPairReductionSmallData(FiducialPairReductionStdData):
    def setUp(self):
        super(FiducialPairReductionSmallData, self).setUp()
        self.preps = pc.to_circuits([('Gx',)])
        self.effects = self.preps
        self.germs = pc.to_circuits([('Gx',), ('Gy',)])
        self.fiducial_pairs = [(0, 0)]

# TODO optimize!!!!


class FindSufficientFiducialPairsBase(object):
    def test_find_sufficient_fiducial_pairs_sequential(self):
        fiducial_pairs = fpr.find_sufficient_fiducial_pairs(
            self.model, self.preps, self.effects, self.germs,
            search_mode='sequential'
        )
        self.assertEqual(fiducial_pairs, self.fiducial_pairs)

    def test_find_sufficient_fiducial_pairs_random(self):
        fiducial_pairs = fpr.find_sufficient_fiducial_pairs(
            self.model, self.preps, self.effects, self.germs,
            search_mode='random', n_random=300, seed=_SEED
        )
        # TODO assert correctness


class FindSufficientFiducialPairsPerGermBase(object):
    def test_find_sufficient_fiducial_pairs_per_germ_sequential(self):
        fiducial_pairs = fpr.find_sufficient_fiducial_pairs_per_germ(
            self.model, self.preps, self.effects, self.germs,
            search_mode='sequential', retry_for_smaller=False, 
            min_iterations=1, verbosity=0
        )
        #print("Found per-germ pairs:\n", fiducial_pairs)
        self.assertTrue((fiducial_pairs == self.fiducial_pairs_per_germ) or (fiducial_pairs == self.fiducial_pairs_per_germ_windows_38))

    def test_find_sufficient_fiducial_pairs_per_germ_random(self):
        fiducial_pairs = fpr.find_sufficient_fiducial_pairs_per_germ(
            self.model, self.preps, self.effects, self.germs,
            search_mode='random', n_random=100, seed=_SEED
        )
        # TODO assert correctness


class FindSufficientFiducialPairsPerGermGreedy(object):
    def test_find_sufficient_fiducial_pairs_per_germ_greedy_random(self):
        fiducial_pairs = fpr.find_sufficient_fiducial_pairs_per_germ_greedy(
            self.model, self.preps, self.effects, self.germs,
            initial_seed_mode='random', seed=_SEED, check_complete_fid_set=False)
        #TODO assert correctness


    def test_find_sufficient_fiducial_pairs_per_germ_greedy_greedy(self):
        fiducial_pairs = fpr.find_sufficient_fiducial_pairs_per_germ_greedy(
            self.model, self.preps, self.effects, self.germs,
            initial_seed_mode='greedy', seed=_SEED, check_complete_fid_set=False)
        #TODO assert correctness

class FindSufficientFiducialPairsPerGermGlobal(object):

    def test_germ_set_spanning_vectors_greedy(self):
        spanning_vec_set = germ_set_spanning_vectors(self.model, self.germs, 
                                                     assume_real=True, float_type=_np.double, 
                                                     verbosity=0, mode = 'greedy', final_test = True)
        
        #TODO assert correctness

    def test_germ_set_spanning_vectors_rrqr(self):
        spanning_vec_set = germ_set_spanning_vectors(self.model, self.germs, 
                                                     assume_real=True, float_type=_np.double, 
                                                     verbosity=0, mode = 'rrqr', final_test = True)


    def test_find_sufficient_fiducial_pairs_per_germ_global(self):
        fiducial_pairs = fpr.find_sufficient_fiducial_pairs_per_germ_global(
            self.model, self.preps, self.effects, germs= self.germs,
            initial_seed_mode='greedy')
        #TODO assert correctness

class StdDataFindSufficientFiducialPairsTester(FindSufficientFiducialPairsBase,
                                               FindSufficientFiducialPairsPerGermBase,
                                               FiducialPairReductionStdData,
                                               BaseCase):
    def test_find_sufficient_fiducial_pairs_with_test_pair_list(self):
        test_pair_list = [(0, 0), (0, 1), (1, 0)]
        fiducial_pairs = fpr.find_sufficient_fiducial_pairs(
            self.model, self.preps, self.effects, self.germs,
            test_pair_list=test_pair_list
        )
        # TODO assert correctness


class SmallDataFindSufficientFiducialPairsTester(FindSufficientFiducialPairsBase,
                                                 FiducialPairReductionSmallData,
                                                 BaseCase):
    pass


class FindSufficientFiducialPairsExceptionTester(FiducialPairReductionStdData, BaseCase):
    def test_find_sufficient_fiducial_pairs_per_germ_raises_on_insufficient_fiducials(self):
        insuff_fids = pc.to_circuits([('Gx',)])
        with self.assertRaises(ValueError):
            fpr.find_sufficient_fiducial_pairs_per_germ(
                self.model, insuff_fids, insuff_fids, self.germs
            )


# TODO optimize
class _TestFiducialPairsBase(object):
    def test_test_fiducial_pairs_from_list(self):
        n_amplified = fpr.test_fiducial_pairs(
            self.fiducial_pairs, self.model, self.preps, self.effects,
            self.germs
        )
        self.assertEqual(n_amplified, self.expected_amplified)

    def test_test_fiducial_pairs_from_dict(self):
        n_amplified = fpr.test_fiducial_pairs(
            self.fiducial_pairs_per_germ, self.model, self.preps, self.effects,
            self.germs
        )
        self.assertEqual(n_amplified, self.expected_amplified)


class StdDataTestFiducialPairsTester(_TestFiducialPairsBase, FiducialPairReductionStdData, BaseCase):
    expected_amplified = 34

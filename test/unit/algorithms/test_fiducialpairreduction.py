import pygsti.algorithms.fiducialpairreduction as fpr
from pygsti.algorithms.germselection import germ_set_spanning_vectors
import pygsti.circuits as pc
from pygsti.circuits import Circuit
from pygsti.baseobjs import Label
import numpy as _np
from . import fixtures
from ..util import BaseCase

_SEED = 1234


class FiducialPairReductionStdData(object):
    def setUp(self):
        super(FiducialPairReductionStdData, self).setUp()
        self.model = fixtures.fullTP_model
        self.preps = fixtures.prep_fids
        self.effects = fixtures.meas_fids
        self.germs = fixtures.germs
        self.fiducial_pairs_per_germ = {Circuit([Label('Gxpi2',0)], line_labels=(0,)): [(0, 0), (0, 1), (2, 2)], 
            Circuit([Label('Gypi2',0)], line_labels=(0,)): [(0, 0), (0, 2), (1, 1)], 
            Circuit([Label('Gxpi2',0), Label('Gypi2',0)], line_labels=(0,)): [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)], 
            Circuit([Label('Gxpi2',0), Label('Gxpi2',0), Label('Gypi2',0)], line_labels=(0,)): [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1)]}
        
        self.fiducial_pairs_per_germ_random = {Circuit([Label('Gxpi2',0)], line_labels=(0,)): [(0, 0), (0, 1), (5, 2)], 
            Circuit([Label('Gypi2',0)], line_labels=(0,)): [(0, 0), (0, 5), (1, 1)], 
            Circuit([Label('Gxpi2',0), Label('Gypi2',0)], line_labels=(0,)): [(0, 2), (0, 4), (0, 5), (2, 5), (5, 2)], 
            Circuit([Label('Gxpi2',0), Label('Gxpi2',0), Label('Gypi2',0)], line_labels=(0,)): [(2, 0), (2, 5), (3, 4), (4, 4), (4, 5)]}
        
        #Sometimes on windows it different final results are obtained (I think primarily due to minor rounding differences coming
        #slightly different linear algebra implementations).
        
        self.fiducial_pairs_per_germ_random_alt_win = {Circuit([Label('Gxpi2',0)], line_labels=(0,)): [(0, 0), (0, 1), (3, 3)], 
        Circuit([Label('Gypi2',0)], line_labels=(0,)): [(2, 3), (5, 1), (5, 2)], 
        Circuit([Label('Gxpi2',0), Label('Gypi2',0)], line_labels=(0,)): [(0, 2), (0, 4), (0, 5), (2, 5), (5, 2)], 
        Circuit([Label('Gxpi2',0), Label('Gxpi2',0), Label('Gypi2',0)], line_labels=(0,)): [(2, 0), (2, 5), (3, 4), (4, 4), (4, 5)]}
        
        self.fiducial_pairs_per_germ_alt_win = {Circuit([Label('Gxpi2',0)], line_labels=(0,)): [(0, 0), (0, 1), (0, 2)], 
        Circuit([Label('Gypi2',0)], line_labels=(0,)): [(0, 0), (0, 1), (0, 2)], 
        Circuit([Label('Gxpi2',0), Label('Gypi2',0)], line_labels=(0,)): [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)], 
        Circuit([Label('Gxpi2',0), Label('Gxpi2',0), Label('Gypi2',0)], line_labels=(0,)): [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1)]}

        # And the macos-latest shifted to the M1 chips and we have another set of difference
        self.fiducial_pairs_per_germ_random_alt_mac = {Circuit([("Gxpi2", 0)]): [(0, 0), (0, 1), (5, 2)],
        Circuit([("Gypi2", 0)]): [(2, 3), (5, 1), (5, 2)],
        Circuit([("Gxpi2", 0), ("Gypi2", 0)]): [(0, 2), (0, 4), (0, 5), (2, 5), (5, 2)],
        Circuit([("Gxpi2", 0), ("Gxpi2", 0), ("Gypi2", 0)]): [(2, 0), (2, 5), (3, 4), (4, 4), (4, 5)]}

        self.fiducial_pairs_per_germ_alt_mac = {Circuit([("Gxpi2", 0)]): [(0, 0), (0, 1), (2, 2)],
        Circuit([("Gypi2", 0)]): [(0, 0), (0, 1), (0, 2)],
        Circuit([("Gxpi2", 0), ("Gypi2", 0)]): [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],
        Circuit([("Gxpi2", 0), ("Gxpi2", 0), ("Gypi2", 0)]): [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1)]}
        
        #self.fiducial_pairs_global = [(0, 0), (0, 1), (0, 2), (1, 3)]
        self.fiducial_pairs_global =  [(0, 0), (0, 1), (0, 2), (1, 0)]

class FindSufficientFiducialPairsBase(object):
    def test_find_sufficient_fiducial_pairs_sequential(self):
        fiducial_pairs = fpr.find_sufficient_fiducial_pairs(
            self.model, self.preps, self.effects, self.germs,
            search_mode='sequential', minimum_pairs=4,
            test_lengths = (64, 512), tol = 0.5
        )

        self.assertTrue(fiducial_pairs == self.fiducial_pairs_global)

    def test_find_sufficient_fiducial_pairs_random(self):
        fiducial_pairs = fpr.find_sufficient_fiducial_pairs(
            self.model, self.preps, self.effects, self.germs,
            search_mode='random', n_random=5, seed=_SEED, minimum_pairs=4,
            test_lengths = (64, 512), tol = 0.5
        )
        # TODO assert correctness


class FindSufficientFiducialPairsPerGermBase(object):
    def test_find_sufficient_fiducial_pairs_per_germ_sequential(self):
        fiducial_pairs = fpr.find_sufficient_fiducial_pairs_per_germ(
            self.model, self.preps, self.effects, self.germs,
            search_mode='sequential', retry_for_smaller=False, 
            min_iterations=1, verbosity=0
        )

        self.assertTrue(fiducial_pairs == self.fiducial_pairs_per_germ
                        or fiducial_pairs == self.fiducial_pairs_per_germ_alt_win
                        or fiducial_pairs == self.fiducial_pairs_per_germ_alt_mac)

    def test_find_sufficient_fiducial_pairs_per_germ_random(self):
        fiducial_pairs = fpr.find_sufficient_fiducial_pairs_per_germ(
            self.model, self.preps, self.effects, self.germs,
            search_mode='random', n_random=10, seed=_SEED
        )
        
        self.assertTrue(fiducial_pairs == self.fiducial_pairs_per_germ_random
                        or fiducial_pairs == self.fiducial_pairs_per_germ_random_alt_win
                        or fiducial_pairs == self.fiducial_pairs_per_germ_random_alt_mac)


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
                                                     verbosity=0, mode = 'greedy', evd_tol= 1e-6, final_test = True)
        
        self.spanning_vec_set_greedy = spanning_vec_set
        
        #TODO assert correctness

    def test_germ_set_spanning_vectors_rrqr(self):
        spanning_vec_set = germ_set_spanning_vectors(self.model, self.germs, 
                                                     assume_real=True, float_type=_np.double, 
                                                     verbosity=0, mode = 'rrqr', final_test = True)

        self.spanning_vec_set_rrqr = spanning_vec_set
        
    def test_find_sufficient_fiducial_pairs_per_germ_global(self):
        fiducial_pairs = fpr.find_sufficient_fiducial_pairs_per_germ_global(
            self.model, self.preps, self.effects, germ_vector_spanning_set= self.spanning_vec_set_greedy,
            initial_seed_mode='greedy', float_type = _np.double, evd_tol=1e-6, inv_trace_tol= 30)
        #TODO assert correctness

class StdDataFindSufficientFiducialPairsTester(FindSufficientFiducialPairsBase,
                                               FindSufficientFiducialPairsPerGermBase,
                                               FiducialPairReductionStdData,
                                               BaseCase):
    def test_find_sufficient_fiducial_pairs_with_test_pair_list(self):
        test_pair_list = [(0, 0), (0, 1), (1, 0)]
        fiducial_pairs = fpr.find_sufficient_fiducial_pairs(
            self.model, self.preps, self.effects, self.germs,
            test_pair_list=test_pair_list, test_lengths = (64, 512), 
            tol = 0.5
        )
        # TODO assert correctness


class FindSufficientFiducialPairsExceptionTester(FiducialPairReductionStdData, BaseCase):
    def test_find_sufficient_fiducial_pairs_per_germ_raises_on_insufficient_fiducials(self):
        insuff_fids = [Circuit([Label('Gxpi2',0)], line_labels = (0,))]
        with self.assertRaises(ValueError):
            fpr.find_sufficient_fiducial_pairs_per_germ(
                self.model, insuff_fids, insuff_fids, self.germs
            )


# TODO optimize
class _TestFiducialPairsBase(object):
    def test_test_fiducial_pairs_from_list(self):
        n_amplified = fpr.test_fiducial_pairs(
            self.fiducial_pairs_global, self.model, self.preps, self.effects,
            self.germs, test_lengths=(64, 512), tol = 0.5
        )
        self.assertEqual(n_amplified, self.expected_amplified_global)

    def test_test_fiducial_pairs_from_dict(self):
        n_amplified = fpr.test_fiducial_pairs(
            self.fiducial_pairs_per_germ, self.model, self.preps, self.effects,
            self.germs, test_lengths=(64, 512), tol = 0.5
        )
        self.assertEqual(n_amplified, self.expected_amplified_per_germ)


class StdDataTestFiducialPairsTester(_TestFiducialPairsBase, FiducialPairReductionStdData, BaseCase):
    expected_amplified_global = 13
    expected_amplified_per_germ = 13

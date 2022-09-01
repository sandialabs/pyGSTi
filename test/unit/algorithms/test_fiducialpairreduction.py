import pygsti.algorithms.fiducialpairreduction as fpr
import pygsti.circuits as pc
from pygsti.circuits import Circuit
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
        Circuit(Gi): [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)], 
        Circuit(Gx): [(0, 0), (0, 1), (2, 2)], 
        Circuit(Gy): [(0, 0), (0, 2), (1, 1)], 
        Circuit(GxGy): [(0, 0), (0, 1), (0, 2), (0, 3), (1, 3)], 
        Circuit(GxGxGy): [(0, 0), (0, 1), (0, 2), (1, 0), (1, 4)], 
        Circuit(GxGyGy): [(0, 0), (0, 1), (0, 2), (2, 0), (2, 2)], 
        Circuit(GxGyGi): [(0, 0), (0, 1), (0, 2), (0, 3), (1, 2)], 
        Circuit(GxGiGy): [(0, 0), (0, 1), (0, 2), (0, 3), (1, 2)], 
        Circuit(GxGiGi): [(0, 0), (0, 1), (2, 2)], 
        Circuit(GyGiGi): [(0, 0), (0, 2), (1, 1)], 
        Circuit(GxGyGyGi): [(0, 0), (0, 1), (0, 2), (2, 0), (2, 2)], 
        Circuit(GxGxGyGxGyGy): [(0, 0), (0, 1), (0, 2), (0, 3), (1, 3)]
        }
        
        
        #self.fiducial_pairs_per_germ_old = {
        #    Circuit(('Gi',)): [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)],
        #    Circuit(('Gx',)): [(0, 0), (0, 1), (2, 2)],
        #    Circuit(('Gy',)): [(0, 0), (0, 2), (1, 4)],
        #    Circuit(('Gx', 'Gy')): [(0, 0), (0, 1), (1, 3)],
        #    Circuit(('Gx', 'Gx', 'Gy')): [(0, 0), (0, 1), (0, 2), (1, 0), (1, 4)],
        #    Circuit(('Gx', 'Gy', 'Gy')): [(0, 0), (0, 1), (0, 2), (2, 0), (2, 2)],
        #    Circuit(('Gx', 'Gy', 'Gi')): [(0, 0), (0, 1), (1, 3)],
        #    Circuit(('Gx', 'Gi', 'Gy')): [(0, 0), (0, 1), (1, 3)],
        #    Circuit(('Gx', 'Gi', 'Gi')): [(0, 0), (0, 1), (2, 2)],
        #    Circuit(('Gy', 'Gi', 'Gi')): [(0, 0), (0, 2), (1, 4)],
        #    Circuit(('Gx', 'Gy', 'Gy', 'Gi')): [(0, 0), (0, 1), (0, 2), (2, 0), (2, 2)],
        #    Circuit(('Gx', 'Gx', 'Gy', 'Gx', 'Gy', 'Gy')): [(0, 0), (0, 1), (1, 3)]
        #}

        #self.fiducial_pairs_per_germ_old1 = {
        #    Circuit(('Gi',)): [(0, 0), (1, 1), (2, 2)],
        #    Circuit(('Gx',)): [(0, 0), (0, 1), (0, 2), (2, 2)],
        #    Circuit(('Gy',)): [(0, 0), (0, 1), (0, 2), (1, 1)],
        #    Circuit(('Gx', 'Gy')): [(0, 0), (0, 1), (0, 2), (2, 0)],
        #    Circuit(('Gx', 'Gx', 'Gy')): [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)],
        #    Circuit(('Gx', 'Gy', 'Gy')): [(0, 0), (0, 1), (0, 2), (2, 0), (2, 1), (2, 2)],
        #    Circuit(('Gx', 'Gy', 'Gi')): [(0, 0), (0, 1), (0, 2), (2, 0)],
        #    Circuit(('Gx', 'Gi', 'Gy')): [(0, 0), (0, 1), (0, 2), (2, 0)],
        #    Circuit(('Gx', 'Gi', 'Gi')): [(0, 0), (0, 1), (0, 2), (2, 2)],
        #    Circuit(('Gy', 'Gi', 'Gi')): [(0, 0), (0, 1), (0, 2), (1, 1)],
        #    Circuit(('Gx', 'Gy', 'Gy', 'Gi')): [(0, 0), (0, 1), (0, 2), (2, 0), (2, 1), (2, 2)],
        #    Circuit(('Gx', 'Gx', 'Gy', 'Gx', 'Gy', 'Gy')): [(0, 0), (0, 1), (0, 2), (1, 0)]
        #}
        #
        ##I think this alternate set is due to a slightly different search ordering - it
        ## still looks like a correct output (there are many), so we'll call this OK for now:
        #self.fiducial_pairs_per_germ_old2 = {
        #    Circuit(('Gi',)): [(0, 0), (1, 1), (2, 2)],
        #    Circuit(('Gx',)): [(0, 0), (0, 1), (0, 2), (2, 0)],  # 2,2 => 2,0
        #    Circuit(('Gy',)): [(0, 0), (0, 1), (0, 2), (1, 0)],  # 1,1 => 1,0
        #    Circuit(('Gx', 'Gy')): [(0, 0), (0, 1), (0, 2), (2, 0)],
        #    Circuit(('Gx', 'Gx', 'Gy')): [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)],
        #    Circuit(('Gx', 'Gy', 'Gy')): [(0, 0), (0, 1), (0, 2), (2, 0), (2, 1), (2, 2)],
        #    Circuit(('Gx', 'Gy', 'Gi')): [(0, 0), (0, 1), (0, 2), (2, 0)],
        #    Circuit(('Gx', 'Gi', 'Gy')): [(0, 0), (0, 1), (0, 2), (2, 0)],
        #    Circuit(('Gx', 'Gi', 'Gi')): [(0, 0), (0, 1), (0, 2), (2, 0)],  # 2,2 => 2,0
        #    Circuit(('Gy', 'Gi', 'Gi')): [(0, 0), (0, 1), (0, 2), (1, 0)],  # 1,1 => 1,0
        #    Circuit(('Gx', 'Gy', 'Gy', 'Gi')): [(0, 0), (0, 1), (0, 2), (2, 0), (2, 1), (2, 2)],
        #    Circuit(('Gx', 'Gx', 'Gy', 'Gx', 'Gy', 'Gy')): [(0, 0), (0, 1), (0, 2), (1, 0)]
        #}


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
            search_mode='sequential'
        )
        print("Found per-germ pairs:\n", fiducial_pairs)
        self.assertTrue(fiducial_pairs == self.fiducial_pairs_per_germ)
            # or fiducial_pairs == self.fiducial_pairs_per_germ_alt)

    def test_find_sufficient_fiducial_pairs_per_germ_random(self):
        fiducial_pairs = fpr.find_sufficient_fiducial_pairs_per_germ(
            self.model, self.preps, self.effects, self.germs,
            search_mode='random', n_random=100, seed=_SEED
        )
        # TODO assert correctness


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

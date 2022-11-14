import pygsti
import numpy as _np
from pygsti.modelpacks.legacy import std1Q_XYI as std
from ..algorithms.algorithmsTestCase import AlgorithmTestCase


class GermSelectionTestCase(AlgorithmTestCase):
    def test_germsel_grasp(self):
        threshold             = 1e6
        randomizationStrength = 1e-3
        neighborhoodSize      = 5
        gatesetNeighborhood   = pygsti.alg.randomize_model_list([std.target_model()],
                                                                randomization_strength=randomizationStrength,
                                                                num_copies=neighborhoodSize, seed=2014)

        # max_length   = 6
        gates        = list(std.target_model().operations.keys())
        superGermSet = [] #OLD: pygsti.construction.list_all_circuits_without_powers_and_cycles(gates, max_length)
        superGermSet.extend( pygsti.circuits.list_all_circuits_without_powers_and_cycles(
            gates, max_length=3) )
        superGermSet.extend( pygsti.circuits.list_random_circuits_onelen(
            gates, 4, 10, seed=2017)) # add 10 random candidates of length 4
        superGermSet.extend( pygsti.circuits.list_random_circuits_onelen(
            gates, 5, 10, seed=2017)) # add 10 random candidates of length 5
        superGermSet.extend( pygsti.circuits.list_random_circuits_onelen(
            gates, 6, 10, seed=2017)) # add 10 random candidates of length 6
        superGermSet.extend(std.germs) #so we know we have enough good ones!

        soln = pygsti.alg.find_germs_grasp(model_list=gatesetNeighborhood, germs_list=superGermSet,
                                           alpha=0.1, randomize=False, seed=2014, score_func='all',
                                           threshold=threshold, verbosity=1, iterations=1,
                                           l1_penalty=1.0, return_all=False)

        forceStrs = pygsti.circuits.to_circuits([('Gx',), ('Gy')])
        bestSoln, initialSolns, localSolns = \
            pygsti.alg.find_germs_grasp(model_list=gatesetNeighborhood, germs_list=superGermSet,
                                        alpha=0.1, randomize=False, seed=2014, score_func='all',
                                        threshold=threshold, verbosity=1, iterations=1,
                                        l1_penalty=1.0, return_all=True, force=forceStrs)

        # try case with incomplete initial germ set
        incompleteSet = pygsti.circuits.to_circuits([('Gx',), ('Gy')])
        soln = pygsti.alg.find_germs_grasp(model_list=gatesetNeighborhood, germs_list=incompleteSet,
                                           alpha=0.1, randomize=False, seed=2014, score_func='worst',
                                           threshold=threshold, verbosity=1, iterations=1,
                                           l1_penalty=1.0)

    def test_germsel_greedy(self):
        threshold             = 1e6
        randomizationStrength = 1e-3
        neighborhoodSize      = 5
        gatesetNeighborhood   = pygsti.alg.randomize_model_list([std.target_model()],
                                                                randomization_strength=randomizationStrength,
                                                                num_copies=neighborhoodSize, seed=2014)

        max_length   = 6
        gates        = std.target_model().operations.keys()
        superGermSet = pygsti.circuits.list_all_circuits_without_powers_and_cycles(gates, max_length)

          # with small memory limit
        with self.assertRaises(MemoryError):
            pygsti.alg.find_germs_breadthfirst(gatesetNeighborhood, superGermSet,
                                               randomize=False, seed=2014, score_func='all',
                                               threshold=threshold, verbosity=1, op_penalty=1.0,
                                               mem_limit=1024)

        pygsti.alg.find_germs_breadthfirst(gatesetNeighborhood, superGermSet,
                                           randomize=False, seed=2014, score_func='all',
                                           threshold=threshold, verbosity=1, op_penalty=1.0,
                                           mem_limit=1024000)
                                           

    def test_germsel_low_rank(self):
        #test greedy search algorithm using low-rank updates

        soln = pygsti.algorithms.germselection.find_germs(std.target_model(), candidate_germ_counts={4:'all upto'},
                                           randomize=False, algorithm='greedy', mode='compactEVD',
                                           assume_real=True, float_type=_np.double,  verbosity=0)


    def test_germsel_driver(self):
        #GREEDY
        options = {'threshold': 1e6 }
        germs = pygsti.alg.find_germs(std.target_model(), randomize=True, randomization_strength=1e-3,
                                      num_gs_copies=5, seed=2017, candidate_germ_counts={3: 'all upto', 4: 10, 5:10, 6:10},
                                      candidate_seed=2017, force="singletons", algorithm='greedy',
                                      algorithm_kwargs=options, mem_limit=None, comm=None,
                                      profiler=None, verbosity=1)
                                      
        #Greedy Low-Rank Updates
        germs = pygsti.algorithms.germselection.find_germs(std.target_model(), seed=2017, 
                                   candidate_germ_counts={3: 'all upto', 4: 10, 5:10, 6:10},
                                   randomize=False, algorithm='greedy', mode='compactEVD',
                                   assume_real=True, float_type=_np.double,  verbosity=1)
        
        
        #GRASP
        options = dict(l1_penalty=1e-2,
                       op_penalty=0.1,
                       score_func='all',
                       tol=1e-6, threshold=1e6,
                       iterations=2)
        germs = pygsti.alg.find_germs(std.target_model(), randomize=True, randomization_strength=1e-3,
                                      num_gs_copies=2, seed=2017, candidate_germ_counts={3: 'all upto', 4: 10, 5:10, 6:10},
                                      candidate_seed=2017, force="singletons", algorithm='grasp',
                                      algorithm_kwargs=options, mem_limit=None, comm=None,
                                      profiler=None, verbosity=1)

        #more args
        options['return_all'] = True #but doesn't change find_germs return value
        germs2 = pygsti.alg.find_germs(std.target_model(), randomize=True, randomization_strength=1e-3,
                                       num_gs_copies=2, seed=2017, candidate_germ_counts={3: 'all upto', 4: 10, 5:10, 6:10},
                                       candidate_seed=2017, force="singletons", algorithm='grasp',
                                       algorithm_kwargs=options, mem_limit=None, comm=None,
                                       profiler=None, verbosity=1)


        #SLACK
        options = dict(fixed_slack=False, slack_frac=0.1)
        germs = pygsti.alg.find_germs(std.target_model(), randomize=True, randomization_strength=1e-3,
                                      num_gs_copies=2, seed=2017, candidate_germ_counts={3: 'all upto', 4: 10, 5:10, 6:10},
                                      candidate_seed=2017, force="singletons", algorithm='slack',
                                      algorithm_kwargs=options, mem_limit=None, comm=None,
                                      profiler=None, verbosity=1)

        #no options -> use defaults
        options = {}
        germs = pygsti.alg.find_germs(std.target_model(), randomize=True, randomization_strength=1e-3,
                                      num_gs_copies=2, seed=2017, candidate_germ_counts={3: 'all upto', 4: 10, 5:10, 6:10},
                                      candidate_seed=2017, force="singletons", algorithm='slack',
                                      algorithm_kwargs=options, mem_limit=None, comm=None,
                                      profiler=None, verbosity=1)

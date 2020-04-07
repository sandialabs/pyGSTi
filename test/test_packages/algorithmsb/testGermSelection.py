from pygsti.modelpacks.legacy import std1Q_XYI as std
import pygsti
import pygsti.algorithms.germselection as germsel
import pygsti.algorithms.scoring as scoring

import numpy as np
import sys, os

from ..algorithms.algorithmsTestCase import AlgorithmTestCase


class GermSelectionTestCase(AlgorithmTestCase):
    def test_germsel_grasp(self):
        threshold             = 1e6
        randomizationStrength = 1e-3
        neighborhoodSize      = 5
        gatesetNeighborhood   = pygsti.alg.randomize_model_list([std.target_model()],
                                  randomizationStrength=randomizationStrength,
                                  numCopies=neighborhoodSize, seed=2014)

        # max_length   = 6
        gates        = list(std.target_model().operations.keys())
        superGermSet = [] #OLD: pygsti.construction.list_all_circuits_without_powers_and_cycles(gates, max_length)
        superGermSet.extend( pygsti.construction.list_all_circuits_without_powers_and_cycles(
            gates, maxLength=3) )
        superGermSet.extend( pygsti.construction.list_random_circuits_onelen(
            gates, 4, 10, seed=2017)) # add 10 random candidates of length 4
        superGermSet.extend( pygsti.construction.list_random_circuits_onelen(
            gates, 5, 10, seed=2017)) # add 10 random candidates of length 5
        superGermSet.extend( pygsti.construction.list_random_circuits_onelen(
            gates, 6, 10, seed=2017)) # add 10 random candidates of length 6
        superGermSet.extend(std.germs) #so we know we have enough good ones!

        soln = pygsti.alg.grasp_germ_set_optimization(modelList=gatesetNeighborhood, germsList=superGermSet,
                                            alpha=0.1, randomize=False, seed=2014, scoreFunc='all',
                                            threshold=threshold, verbosity=1, iterations=1,
                                            l1Penalty=1.0, returnAll=False)

        forceStrs = pygsti.construction.circuit_list([ ('Gx',), ('Gy') ])
        bestSoln, initialSolns, localSolns = \
            pygsti.alg.grasp_germ_set_optimization(modelList=gatesetNeighborhood, germsList=superGermSet,
                                               alpha=0.1, randomize=False, seed=2014, scoreFunc='all',
                                               threshold=threshold, verbosity=1, iterations=1,
                                               l1Penalty=1.0, returnAll=True, force=forceStrs)

        # try case with incomplete initial germ set
        incompleteSet = pygsti.construction.circuit_list([ ('Gx',), ('Gy') ])
        soln = pygsti.alg.grasp_germ_set_optimization(modelList=gatesetNeighborhood, germsList=incompleteSet,
                                               alpha=0.1, randomize=False, seed=2014, scoreFunc='worst',
                                               threshold=threshold, verbosity=1, iterations=1,
                                               l1Penalty=1.0)



    def test_germsel_greedy(self):
        threshold             = 1e6
        randomizationStrength = 1e-3
        neighborhoodSize      = 5
        gatesetNeighborhood   = pygsti.alg.randomize_model_list([std.target_model()],
                                  randomizationStrength=randomizationStrength,
                                  numCopies=neighborhoodSize, seed=2014)

        max_length   = 6
        gates        = std.target_model().operations.keys()
        superGermSet = pygsti.construction.list_all_circuits_without_powers_and_cycles(gates, max_length)

          # with small memory limit
        with self.assertRaises(MemoryError):
            pygsti.alg.build_up_breadth(gatesetNeighborhood, superGermSet,
                                        randomize=False, seed=2014, scoreFunc='all',
                                        threshold=threshold, verbosity=1, opPenalty=1.0,
                                        memLimit=1024)

        pygsti.alg.build_up_breadth(gatesetNeighborhood, superGermSet,
                                    randomize=False, seed=2014, scoreFunc='all',
                                    threshold=threshold, verbosity=1, opPenalty=1.0,
                                    memLimit=1024000)


    def test_germsel_driver(self):
        #GREEDY
        options = {'threshold': 1e6 }
        germs = pygsti.alg.generate_germs(std.target_model(), randomize=True, randomizationStrength=1e-3,
                               numGSCopies=5, seed=2017, candidateGermCounts={3: 'all upto', 4: 10, 5:10, 6:10},
                               candidateSeed=2017, force="singletons", algorithm='greedy',
	                       algorithm_kwargs=options, memLimit=None, comm=None,
		               profiler=None, verbosity=1)

        #GRASP
        options = dict(l1Penalty=1e-2,
                       opPenalty=0.1,
                       scoreFunc='all',
                       tol=1e-6, threshold=1e6,
                       iterations=2)
        germs = pygsti.alg.generate_germs(std.target_model(), randomize=True, randomizationStrength=1e-3,
                               numGSCopies=2, seed=2017, candidateGermCounts={3: 'all upto', 4: 10, 5:10, 6:10},
                               candidateSeed=2017, force="singletons", algorithm='grasp',
	                       algorithm_kwargs=options, memLimit=None, comm=None,
		               profiler=None, verbosity=1)

        #more args
        options['returnAll'] = True #but doesn't change generate_germs return value
        germs2 = pygsti.alg.generate_germs(std.target_model(), randomize=True, randomizationStrength=1e-3,
                                           numGSCopies=2, seed=2017, candidateGermCounts={3: 'all upto', 4: 10, 5:10, 6:10},
                                           candidateSeed=2017, force="singletons", algorithm='grasp',
	                                   algorithm_kwargs=options, memLimit=None, comm=None,
		                           profiler=None, verbosity=1)


        #SLACK
        options = dict(fixedSlack=False, slackFrac=0.1)
        germs = pygsti.alg.generate_germs(std.target_model(), randomize=True, randomizationStrength=1e-3,
                               numGSCopies=2, seed=2017, candidateGermCounts={3: 'all upto', 4: 10, 5:10, 6:10},
                               candidateSeed=2017, force="singletons", algorithm='slack',
	                       algorithm_kwargs=options, memLimit=None, comm=None,
		               profiler=None, verbosity=1)

        #no options -> use defaults
        options = {}
        germs = pygsti.alg.generate_germs(std.target_model(), randomize=True, randomizationStrength=1e-3,
                                          numGSCopies=2, seed=2017, candidateGermCounts={3: 'all upto', 4: 10, 5:10, 6:10},
                                          candidateSeed=2017, force="singletons", algorithm='slack',
	                                  algorithm_kwargs=options, memLimit=None, comm=None,
		                          profiler=None, verbosity=1)

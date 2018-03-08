from pygsti.construction import std1Q_XYI as std
import pygsti
import pygsti.algorithms.germselection as germsel
import pygsti.algorithms.scoring as scoring

import numpy as np
import sys, os

from ..algorithms.algorithmsTestCase import AlgorithmTestCase

class GermSelectionTestCase(AlgorithmTestCase):

    def test_germsel_tests(self):
        germsToTest = pygsti.construction.list_all_gatestrings_without_powers_and_cycles(
            list(std.gs_target.gates.keys()), 2)

        bSuccess, eigvals_finiteL = pygsti.alg.test_germ_list_finitel(
            self.gs_target_noisy, germsToTest, L=16, returnSpectrum=True, tol=1e-3)
        self.assertFalse(bSuccess)

        bSuccess,eigvals_infiniteL = pygsti.alg.test_germ_list_infl(
            self.gs_target_noisy, germsToTest, returnSpectrum=True, check=True)
        self.assertFalse(bSuccess)


    def test_germsel_slack(self):
      
        germsToTest = pygsti.construction.list_all_gatestrings_without_powers_and_cycles(
            list(std.gs_target.gates.keys()), 3)

        germsToTest2 = pygsti.construction.list_all_gatestrings_without_powers_and_cycles(
            list(std.gs_target.gates.keys()), 4) + std.germs

        finalGerms = pygsti.alg.optimize_integer_germs_slack(
            self.gs_target_noisy, germsToTest, initialWeights=None,
            fixedSlack=0.1, slackFrac=False, returnAll=False, tol=1e-6, verbosity=4)

        forceStrs = pygsti.construction.gatestring_list([ ('Gx',), ('Gy') ])
        finalGerms, wts, scoreDict = pygsti.alg.optimize_integer_germs_slack(
            self.gs_target_noisy, germsToTest2, initialWeights=np.ones( len(germsToTest2), 'd' ),
            fixedSlack=False, slackFrac=0.1, returnAll=True, tol=1e-6, verbosity=4,
            force=forceStrs)

        finalGerms = pygsti.alg.optimize_integer_germs_slack(
            self.gs_target_noisy, germsToTest2, initialWeights=np.ones( len(germsToTest2), 'd' ),
            fixedSlack=False, slackFrac=0.1, returnAll=True, tol=1e-6, verbosity=4,
            force=False) #don't force any strings (default would have been "singletons"

        with self.assertRaises(ValueError):
            pygsti.alg.optimize_integer_germs_slack(
                self.gs_target_noisy, germsToTest2, initialWeights=np.ones( len(germsToTest2)+10, 'd' ),
                fixedSlack=False, slackFrac=0.1, returnAll=True, tol=1e-6, verbosity=4,
                force=forceStrs) # initialWeights has bad length (+10)

        self.runSilent(pygsti.alg.optimize_integer_germs_slack,
                       self.gs_target_noisy, germsToTest2,
                       initialWeights=np.ones( len(germsToTest2), 'd' ),
                       fixedSlack=False, slackFrac=0.1,
                       returnAll=True, tol=1e-6, verbosity=4, maxIter=1)
                       # test hitting max iterations

        with self.assertRaises(ValueError):
            pygsti.alg.optimize_integer_germs_slack(
                self.gs_target_noisy, germsToTest,
                initialWeights=np.ones( len(germsToTest), 'd' ),
                returnAll=True, tol=1e-6, verbosity=4)
                # must specify either fixedSlack or slackFrac

        with self.assertRaises(ValueError):
            pygsti.alg.optimize_integer_germs_slack(
                self.gs_target_noisy, germsToTest,
                initialWeights=np.ones( 1, 'd' ),
                returnAll=True, tol=1e-6, verbosity=4)
                # length of initialWeights must match length of germs list


    def test_germsel_grasp(self):
        threshold             = 1e6
        randomizationStrength = 1e-3
        neighborhoodSize      = 5
        gatesetNeighborhood   = pygsti.alg.randomizeGatesetList([std.gs_target],
                                  randomizationStrength=randomizationStrength,
                                  numCopies=neighborhoodSize, seed=2014)

        # max_length   = 6
        gates        = list(std.gs_target.gates.keys())
        superGermSet = [] #OLD: pygsti.construction.list_all_gatestrings_without_powers_and_cycles(gates, max_length)
        superGermSet.extend( pygsti.construction.list_all_gatestrings_without_powers_and_cycles(
            gates, maxLength=3) )
        superGermSet.extend( pygsti.construction.list_random_gatestrings_onelen(
            gates, 4, 10, seed=2017)) # add 10 random candidates of length 4
        superGermSet.extend( pygsti.construction.list_random_gatestrings_onelen(
            gates, 5, 10, seed=2017)) # add 10 random candidates of length 5
        superGermSet.extend( pygsti.construction.list_random_gatestrings_onelen(
            gates, 6, 10, seed=2017)) # add 10 random candidates of length 6
        superGermSet.extend(std.germs) #so we know we have enough good ones!
            
        soln = pygsti.alg.grasp_germ_set_optimization(gatesetList=gatesetNeighborhood, germsList=superGermSet,
                                            alpha=0.1, randomize=False, seed=2014, scoreFunc='all',
                                            threshold=threshold, verbosity=1, iterations=1,
                                            l1Penalty=1.0, returnAll=False)

        forceStrs = pygsti.construction.gatestring_list([ ('Gx',), ('Gy') ])
        bestSoln, initialSolns, localSolns = \
            pygsti.alg.grasp_germ_set_optimization(gatesetList=gatesetNeighborhood, germsList=superGermSet,
                                               alpha=0.1, randomize=False, seed=2014, scoreFunc='all',
                                               threshold=threshold, verbosity=1, iterations=1,
                                               l1Penalty=1.0, returnAll=True, force=forceStrs)

        # try case with incomplete initial germ set
        incompleteSet = pygsti.construction.gatestring_list([ ('Gx',), ('Gy') ])
        soln = pygsti.alg.grasp_germ_set_optimization(gatesetList=gatesetNeighborhood, germsList=incompleteSet,
                                               alpha=0.1, randomize=False, seed=2014, scoreFunc='worst',
                                               threshold=threshold, verbosity=1, iterations=1,
                                               l1Penalty=1.0)



    def test_germsel_greedy(self):
        threshold             = 1e6
        randomizationStrength = 1e-3
        neighborhoodSize      = 5
        gatesetNeighborhood   = pygsti.alg.randomizeGatesetList([std.gs_target],
                                  randomizationStrength=randomizationStrength,
                                  numCopies=neighborhoodSize, seed=2014)
        forceStrs = pygsti.construction.gatestring_list([ ('Gx',), ('Gy') ])

        max_length   = 6
        gates        = std.gs_target.gates.keys()
        superGermSet = pygsti.construction.list_all_gatestrings_without_powers_and_cycles(gates, max_length)
        tinyGermSet = pygsti.construction.list_all_gatestrings_without_powers_and_cycles(gates, 1)

        #Depth first
        pygsti.alg.build_up(gatesetNeighborhood, superGermSet,
                                    randomize=False, seed=2014, scoreFunc='all',
                                    threshold=threshold, verbosity=1, gatePenalty=1.0)
          # with forced gate strings
        pygsti.alg.build_up(gatesetNeighborhood, superGermSet,
                            randomize=False, seed=2014, scoreFunc='all',
                            threshold=threshold, verbosity=1, gatePenalty=1.0,
                            force=forceStrs)
        tiny = pygsti.alg.build_up(gatesetNeighborhood, tinyGermSet,
                                   randomize=False, seed=2014, scoreFunc='all',
                                   threshold=threshold, verbosity=1, gatePenalty=1.0) #incomplete initial set
        self.assertTrue(tiny is None)


        #Breadth first
        pygsti.alg.build_up_breadth(gatesetNeighborhood, superGermSet,
                                    randomize=False, seed=2014, scoreFunc='all',
                                    threshold=threshold, verbosity=1, gatePenalty=1.0)
          # with forced gate strings
        pygsti.alg.build_up_breadth(gatesetNeighborhood, superGermSet,
                                    randomize=False, seed=2014, scoreFunc='all',
                                    threshold=threshold, verbosity=1, gatePenalty=1.0,
                                    force=forceStrs)

          # with small memory limit
        with self.assertRaises(MemoryError):
            pygsti.alg.build_up_breadth(gatesetNeighborhood, superGermSet,
                                        randomize=False, seed=2014, scoreFunc='all',
                                        threshold=threshold, verbosity=1, gatePenalty=1.0,
                                        memLimit=1024)

        pygsti.alg.build_up_breadth(gatesetNeighborhood, superGermSet,
                                    randomize=False, seed=2014, scoreFunc='all',
                                    threshold=threshold, verbosity=1, gatePenalty=1.0,
                                    memLimit=1024000)


    def test_germsel_driver(self):
        #GREEDY
        options = {'threshold': 1e6 }
        germs = pygsti.alg.generate_germs(std.gs_target, randomize=True, randomizationStrength=1e-3,
                               numGSCopies=5, seed=2017, candidateGermCounts={3: 'all upto', 4: 10, 5:10, 6:10},
                               candidateSeed=2017, force="singletons", algorithm='greedy',
	                       algorithm_kwargs=options, memLimit=None, comm=None,
		               profiler=None, verbosity=1)

        #GRASP
        options = dict(l1Penalty=1e-2,
                       gatePenalty=0.1,
                       scoreFunc='all',
                       tol=1e-6, threshold=1e6,
                       iterations=2)
        germs = pygsti.alg.generate_germs(std.gs_target, randomize=True, randomizationStrength=1e-3,
                               numGSCopies=2, seed=2017, candidateGermCounts={3: 'all upto', 4: 10, 5:10, 6:10},
                               candidateSeed=2017, force="singletons", algorithm='grasp',
	                       algorithm_kwargs=options, memLimit=None, comm=None,
		               profiler=None, verbosity=1)

        #more args
        options['returnAll'] = True #but doesn't change generate_germs return value
        germs2 = pygsti.alg.generate_germs(std.gs_target, randomize=True, randomizationStrength=1e-3,
                                           numGSCopies=2, seed=2017, candidateGermCounts={3: 'all upto', 4: 10, 5:10, 6:10},
                                           candidateSeed=2017, force="singletons", algorithm='grasp',
	                                   algorithm_kwargs=options, memLimit=None, comm=None,
		                           profiler=None, verbosity=1)


        #SLACK
        options = dict(fixedSlack=False, slackFrac=0.1)
        germs = pygsti.alg.generate_germs(std.gs_target, randomize=True, randomizationStrength=1e-3,
                               numGSCopies=2, seed=2017, candidateGermCounts={3: 'all upto', 4: 10, 5:10, 6:10},
                               candidateSeed=2017, force="singletons", algorithm='slack',
	                       algorithm_kwargs=options, memLimit=None, comm=None,
		               profiler=None, verbosity=1)

        #no options -> use defaults
        options = {}
        germs = pygsti.alg.generate_germs(std.gs_target, randomize=True, randomizationStrength=1e-3,
                                          numGSCopies=2, seed=2017, candidateGermCounts={3: 'all upto', 4: 10, 5:10, 6:10},
                                          candidateSeed=2017, force="singletons", algorithm='slack',
	                                  algorithm_kwargs=options, memLimit=None, comm=None,
		                          profiler=None, verbosity=1)


        #INVALID
        with self.assertRaises(ValueError):
            pygsti.alg.generate_germs(std.gs_target, randomize=True, randomizationStrength=1e-3,
                                      numGSCopies=2, seed=2017, candidateGermCounts={3: 'all upto', 4: 10, 5:10, 6:10},
                                      candidateSeed=2017, force="singletons", algorithm='foobar',
	                              algorithm_kwargs=options, memLimit=None, comm=None,
		                      profiler=None, verbosity=1)


    def test_scoring(self):
        eigenvalue_array = np.array([1e-6,1e-4,1.0,2.0])
        s0 = scoring.list_score(eigenvalue_array)
        self.assertEqual(s0, 1010001.5)

        s1 = scoring.list_score(eigenvalue_array, 'all')
        self.assertEqual(s1, 1010001.5)

        s2 = scoring.list_score(eigenvalue_array, 'worst')
        self.assertEqual(s2, 1000000)

        with self.assertRaises(ValueError):
            scoring.list_score(eigenvalue_array, 'foobar')

    def test_randomize_gateset(self):
        #with numCopies and a single gate set
        gatesetNeighborhood = pygsti.alg.randomizeGatesetList([std.gs_target],
                                                              randomizationStrength=1e-3,
                                                              numCopies=3, seed=2014)

        #with multiple gate sets
        gatesetNeighborhood   = pygsti.alg.randomizeGatesetList(
            [std.gs_target,std.gs_target], numCopies=None,
            randomizationStrength=1e-3, seed=2014)

        #cannot specify both:
        with self.assertRaises(ValueError):
            pygsti.alg.randomizeGatesetList([std.gs_target,std.gs_target],
                                            numCopies=3, randomizationStrength=1e-3, seed=2014)

    def test_num_nonspam_params(self):
        gs_reduced = pygsti.alg.removeSPAMVectors(std.gs_target)
        N = pygsti.alg.num_non_spam_gauge_params(std.gs_target)
        self.assertEqual(gs_reduced.num_gauge_params(), N)
        self.assertNotEqual(std.gs_target.num_gauge_params(), N)

    def test_helper_functions(self):
        """ Mostly test boundary cases of helper fns """
        germsToTest = pygsti.construction.list_all_gatestrings_without_powers_and_cycles(
            list(std.gs_target.gates.keys()), 2)

        maxscore = pygsti.alg.calculate_germset_score(germsToTest, std.gs_target,
                                                      neighborhoodSize=5, 
                                                      randomizationStrength=1e-2, scoreFunc='all', 
                                                      gatePenalty=0.0, l1Penalty=0.0)

        #Test failures in get_gateset_params:
        gs1 = std.gs_target.copy()
        gs2 = std.gs_target.copy()
        gs3 = std.gs_target.copy()
        gs4 = std.gs_target.copy()
        gs1.set_all_parameterizations("full")
        gs2.set_all_parameterizations("TP")
        gs3.set_all_parameterizations("full")
        gs4.set_all_parameterizations("full")
        gs3.gates['Gi2'] = np.identity(4,'d') #adds non-gauge params but not gauge params
        gs4.gates['Gi2'] = pygsti.obj.StaticGate(np.identity(4,'d')) # keeps param counts the same but adds gate

        with self.assertRaises(ValueError):
            germsel.get_gateset_params([gs1,gs2]) # different number of gauge params
        with self.assertRaises(ValueError):
            germsel.get_gateset_params([gs1,gs3]) # different number of non-gauge params
        with self.assertRaises(ValueError):
            germsel.get_gateset_params([gs1,gs4]) # different number of gates


        self.assertWarns(germsel.setup_gateset_list, [gs1,gs1], randomize=False, randomizationStrength=0, 
                         numCopies=3, seed=0) # warns b/c list len > 1 and numCopies specified

        scoreFn = lambda x: scoring.list_score(x, scoreFunc="all")
        with self.assertRaises(ValueError):
            germsel.compute_composite_germ_score(scoreFn) #need to provide partialDerivDaggerDeriv or partialGermsList

        germsel.compute_composite_germ_score(scoreFn, gateset=gs1, partialGermsList=germsToTest,
                                             eps=1e-5, germLengths=np.array([len(g) for g in germsToTest]))
        
        pDDD = np.zeros( (len(germsToTest),gs1.num_params(),gs1.num_params()), 'd')
        with self.assertRaises(ValueError):
            germsel.compute_composite_germ_score(scoreFn, partialDerivDaggerDeriv=pDDD) # must specify gateset ro numGaugeParams
        with self.assertRaises(ValueError):
            germsel.compute_composite_germ_score(scoreFn, gateset=gs1, partialDerivDaggerDeriv=pDDD,
                                                 gatePenalty=1.0) # must specify partialGermLists or germLengths
        germsel.compute_composite_germ_score(scoreFn, gateset=gs1, partialDerivDaggerDeriv=pDDD,
                                             partialGermsList=germsToTest, gatePenalty=1.0) # OK

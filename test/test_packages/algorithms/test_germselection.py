import pygsti
import numpy as _np
from pygsti.circuits import Circuit
from pygsti.baseobjs import Label
from pygsti.modelpacks import smq1Q_XY as std
from ..algorithms.algorithmsTestCase import AlgorithmTestCase

class GermSelectionTestData(object):
    germs_greedy = {Circuit([Label('Gxpi2',0)]), 
                            Circuit([Label('Gypi2',0)]), 
                            Circuit([Label('Gxpi2',0),Label('Gypi2',0)]), 
                            Circuit([Label('Gxpi2',0),Label('Gxpi2',0),Label('Gypi2',0)]), 
                            Circuit([Label('Gxpi2',0),Label('Gypi2',0),Label('Gypi2',0)]), 
                            Circuit([Label('Gxpi2',0),Label('Gypi2',0),Label('Gypi2',0),Label('Gypi2',0)]), 
                            Circuit([Label('Gxpi2',0),Label('Gxpi2',0),Label('Gxpi2',0),Label('Gypi2',0)]), 
                            Circuit([Label('Gxpi2',0),Label('Gxpi2',0),Label('Gypi2',0),Label('Gypi2',0)]), 
                            Circuit([Label('Gxpi2',0),Label('Gxpi2',0),Label('Gxpi2',0),Label('Gypi2',0),Label('Gypi2',0)]), 
                            Circuit([Label('Gxpi2',0),Label('Gxpi2',0),Label('Gypi2',0),Label('Gypi2',0),Label('Gypi2',0)]), 
                            Circuit([Label('Gxpi2',0),Label('Gypi2',0),Label('Gypi2',0),Label('Gypi2',0),Label('Gypi2',0)]), 
                            Circuit([Label('Gxpi2',0),Label('Gxpi2',0),Label('Gxpi2',0),Label('Gxpi2',0),Label('Gypi2',0)]), 
                            Circuit([Label('Gxpi2',0),Label('Gypi2',0),Label('Gxpi2',0),Label('Gypi2',0),Label('Gypi2',0)]), 
                            Circuit([Label('Gxpi2',0),Label('Gxpi2',0),Label('Gypi2',0),Label('Gxpi2',0),Label('Gypi2',0)]), 
                            Circuit([Label('Gxpi2',0),Label('Gxpi2',0),Label('Gypi2',0),Label('Gxpi2',0),Label('Gypi2',0),Label('Gypi2',0)])}

    germs_driver_greedy = {Circuit([Label('Gxpi2',0)], line_labels=(0,)), 
                           Circuit([Label('Gypi2',0)], line_labels=(0,)), 
                           Circuit([Label('Gxpi2',0),Label('Gypi2',0)], line_labels=(0,)), 
                           Circuit([Label('Gxpi2',0),Label('Gxpi2',0),Label('Gypi2',0),Label('Gxpi2',0),Label('Gypi2',0),Label('Gypi2',0)], line_labels=(0,)), 
                           Circuit([Label('Gxpi2',0),Label('Gxpi2',0),Label('Gypi2',0),Label('Gxpi2',0)], line_labels=(0,))}
    
    germs_driver_greedy_alt = {Circuit([Label('Gxpi2',0)], line_labels=(0,)), 
                           Circuit([Label('Gypi2',0)], line_labels=(0,)), 
                           Circuit([Label('Gxpi2',0),Label('Gypi2',0)], line_labels=(0,)), 
                           Circuit([Label('Gxpi2',0),Label('Gxpi2',0),Label('Gypi2',0),Label('Gxpi2',0),Label('Gypi2',0),Label('Gypi2',0)], line_labels=(0,)), 
                           Circuit([Label('Gxpi2',0),Label('Gxpi2',0),Label('Gxpi2',0),Label('Gypi2',0)], line_labels=(0,))}

    germs_driver_grasp = ({Circuit([Label('Gxpi2',0)]), 
                                Circuit([Label('Gypi2',0)]), 
                                Circuit([Label('Gxpi2',0),Label('Gypi2',0)]), 
                                Circuit([Label('Gxpi2',0),Label('Gxpi2',0),Label('Gxpi2',0),Label('Gypi2',0)]), 
                                Circuit([Label('Gxpi2',0),Label('Gxpi2',0),Label('Gypi2',0),Label('Gxpi2',0),Label('Gypi2',0),Label('Gypi2',0)])}, 
                        [[Circuit([Label('Gxpi2',0)]), Circuit([Label('Gypi2',0)]), 
                            Circuit([Label('Gxpi2',0),Label('Gypi2',0),Label('Gxpi2',0),Label('Gxpi2',0)]), 
                            Circuit([Label('Gypi2',0),Label('Gxpi2',0),Label('Gxpi2',0),Label('Gxpi2',0),Label('Gypi2',0),Label('Gypi2',0)]), 
                            Circuit([Label('Gxpi2',0),Label('Gxpi2',0),Label('Gypi2',0),Label('Gxpi2',0),Label('Gypi2',0),Label('Gypi2',0)])], 
                            [Circuit([Label('Gxpi2',0)]), Circuit([Label('Gypi2',0)]), Circuit([Label('Gxpi2',0),Label('Gypi2',0),Label('Gxpi2',0),Label('Gxpi2',0)]), 
                            Circuit([Label('Gypi2',0),Label('Gxpi2',0),Label('Gxpi2',0),Label('Gxpi2',0),Label('Gypi2',0),Label('Gypi2',0)]), 
                            Circuit([Label('Gxpi2',0),Label('Gxpi2',0),Label('Gypi2',0),Label('Gxpi2',0),Label('Gypi2',0),Label('Gypi2',0)])], 
                            [Circuit([Label('Gxpi2',0)]), Circuit([Label('Gypi2',0)]), Circuit([Label('Gxpi2',0),Label('Gypi2',0),Label('Gxpi2',0),Label('Gxpi2',0)]), 
                            Circuit([Label('Gypi2',0),Label('Gxpi2',0),Label('Gxpi2',0),Label('Gxpi2',0),Label('Gypi2',0),Label('Gypi2',0)]), 
                            Circuit([Label('Gxpi2',0),Label('Gxpi2',0),Label('Gypi2',0),Label('Gxpi2',0),Label('Gypi2',0),Label('Gypi2',0)])], 
                            [Circuit([Label('Gxpi2',0)]), Circuit([Label('Gypi2',0)]), Circuit([Label('Gxpi2',0),Label('Gypi2',0),Label('Gxpi2',0),Label('Gxpi2',0)]), 
                            Circuit([Label('Gypi2',0),Label('Gxpi2',0),Label('Gxpi2',0),Label('Gxpi2',0),Label('Gypi2',0),Label('Gypi2',0)]), 
                            Circuit([Label('Gxpi2',0),Label('Gxpi2',0),Label('Gypi2',0),Label('Gxpi2',0),Label('Gypi2',0),Label('Gypi2',0)])], 
                            [Circuit([Label('Gxpi2',0)]), Circuit([Label('Gypi2',0)]), Circuit([Label('Gxpi2',0),Label('Gypi2',0),Label('Gxpi2',0),Label('Gxpi2',0)]), 
                            Circuit([Label('Gypi2',0),Label('Gxpi2',0),Label('Gxpi2',0),Label('Gxpi2',0),Label('Gypi2',0),Label('Gypi2',0)]), 
                            Circuit([Label('Gxpi2',0),Label('Gxpi2',0),Label('Gypi2',0),Label('Gxpi2',0),Label('Gypi2',0),Label('Gypi2',0)])]], 
                        [[Circuit([Label('Gxpi2',0)]), Circuit([Label('Gypi2',0)]), Circuit([Label('Gxpi2',0),Label('Gypi2',0)]), 
                            Circuit([Label('Gxpi2',0),Label('Gxpi2',0),Label('Gxpi2',0),Label('Gypi2',0)]), 
                            Circuit([Label('Gxpi2',0),Label('Gxpi2',0),Label('Gypi2',0),Label('Gxpi2',0),Label('Gypi2',0),Label('Gypi2',0)])], 
                            [Circuit([Label('Gxpi2',0)]), Circuit([Label('Gypi2',0)]), Circuit([Label('Gxpi2',0),Label('Gypi2',0)]), 
                            Circuit([Label('Gxpi2',0),Label('Gxpi2',0),Label('Gxpi2',0),Label('Gypi2',0)]), 
                            Circuit([Label('Gxpi2',0),Label('Gxpi2',0),Label('Gypi2',0),Label('Gxpi2',0),Label('Gypi2',0),Label('Gypi2',0)])], 
                            [Circuit([Label('Gxpi2',0)]), Circuit([Label('Gypi2',0)]), Circuit([Label('Gxpi2',0),Label('Gypi2',0)]), 
                            Circuit([Label('Gxpi2',0),Label('Gxpi2',0),Label('Gxpi2',0),Label('Gypi2',0)]), 
                            Circuit([Label('Gxpi2',0),Label('Gxpi2',0),Label('Gypi2',0),Label('Gxpi2',0),Label('Gypi2',0),Label('Gypi2',0)])], 
                            [Circuit([Label('Gxpi2',0)]), Circuit([Label('Gypi2',0)]), Circuit([Label('Gxpi2',0),Label('Gypi2',0)]), 
                            Circuit([Label('Gxpi2',0),Label('Gxpi2',0),Label('Gxpi2',0),Label('Gypi2',0)]), 
                            Circuit([Label('Gxpi2',0),Label('Gxpi2',0),Label('Gypi2',0),Label('Gxpi2',0),Label('Gypi2',0),Label('Gypi2',0)])], 
                            [Circuit([Label('Gxpi2',0)]), Circuit([Label('Gypi2',0)]), Circuit([Label('Gxpi2',0),Label('Gypi2',0)]), 
                            Circuit([Label('Gxpi2',0),Label('Gxpi2',0),Label('Gxpi2',0),Label('Gypi2',0)]), 
                            Circuit([Label('Gxpi2',0),Label('Gxpi2',0),Label('Gypi2',0),Label('Gxpi2',0),Label('Gypi2',0),Label('Gypi2',0)])]])

    germs_driver_grasp_alt ={Circuit([Label('Gxpi2',0)]), 
                                Circuit([Label('Gypi2',0)]), 
                                Circuit([Label('Gxpi2',0),Label('Gypi2',0)]), 
                                Circuit([Label('Gxpi2',0),Label('Gxpi2',0),Label('Gypi2',0),Label('Gxpi2',0)]), 
                                Circuit([Label('Gxpi2',0),Label('Gxpi2',0),Label('Gypi2',0),Label('Gxpi2',0),Label('Gypi2',0),Label('Gypi2',0)])}
    
    germs_driver_slack = {Circuit([Label('Gxpi2',0)]), 
                                Circuit([Label('Gypi2',0)]), 
                                Circuit([Label('Gxpi2',0),Label('Gypi2',0)]), 
                                Circuit([Label('Gxpi2',0),Label('Gxpi2',0),Label('Gxpi2',0),Label('Gypi2',0)]), 
                                Circuit([Label('Gxpi2',0),Label('Gxpi2',0),Label('Gypi2',0),Label('Gxpi2',0)]), 
                                Circuit([Label('Gxpi2',0),Label('Gxpi2',0),Label('Gypi2',0),Label('Gxpi2',0),Label('Gypi2',0),Label('Gypi2',0)])}


class GermSelectionTestCase(AlgorithmTestCase, GermSelectionTestData):

    #test with worst score_func
    def test_germsel_greedy(self):
        threshold             = 1e6
        randomizationStrength = 1e-3
        neighborhoodSize      = 2
        gatesetNeighborhood   = pygsti.alg.randomize_model_list([std.target_model()],
                                                                randomization_strength=randomizationStrength,
                                                                num_copies=neighborhoodSize, seed=2014)

        max_length   = 6
        gates        = std.target_model().operations.keys()
        superGermSet = pygsti.circuits.list_all_circuits_without_powers_and_cycles(gates, max_length)

        germs = pygsti.alg.find_germs_breadthfirst(gatesetNeighborhood, superGermSet,
                                           randomize=False, seed=2014, score_func='worst',
                                           threshold=threshold, verbosity=1, op_penalty=1.0,
                                           mem_limit=2*1024000)
        
        self.assertTrue(self.germs_greedy == set(germs))
                                           
    def test_germsel_driver_greedy(self):
        #GREEDY
        options = {'threshold': 1e6 }
        germs = pygsti.alg.find_germs(std.target_model(), randomize=True, randomization_strength=1e-3,
                                      num_gs_copies=2, seed=2017, candidate_germ_counts={3: 'all upto', 4: 10, 5:10, 6:10},
                                      candidate_seed=2017, force="singletons", algorithm='greedy',
                                      algorithm_kwargs=options, mem_limit=None, comm=None,
                                      profiler=None, verbosity=1)
        
        self.assertTrue(self.germs_driver_greedy == set(germs) or self.germs_driver_greedy_alt == set(germs) )
          
    def test_germsel_driver_grasp(self):
        #more args
        options = {'threshold': 1e6 , 'return_all': True}
        germs = pygsti.alg.find_germs(std.target_model(), randomize=True, randomization_strength=1e-3,
                                       num_gs_copies=2, seed=2017, candidate_germ_counts={3: 'all upto', 4: 10, 5:10, 6:10},
                                       candidate_seed=2017, force="singletons", algorithm='grasp',
                                       algorithm_kwargs=options, mem_limit=None,
                                       profiler=None, verbosity=1)
        
        self.assertTrue(self.germs_driver_grasp[0] == set(germs[0]) or self.germs_driver_grasp_alt == set(germs[0]))
        self.assertTrue(self.germs_driver_grasp[1] == germs[1])
        self.assertTrue(self.germs_driver_grasp[2] == germs[2])

    def test_germsel_driver_slack(self):
        #SLACK
        options = dict(fixed_slack=False, slack_frac=0.1)
        germs = pygsti.alg.find_germs(std.target_model(), randomize=True, randomization_strength=1e-3,
                                      num_gs_copies=2, seed=2017, candidate_germ_counts={3: 'all upto', 4: 10, 5:10, 6:10},
                                      candidate_seed=2017, force="singletons", algorithm='slack',
                                      algorithm_kwargs=options, mem_limit=None, comm=None,
                                      profiler=None, verbosity=1)
        
        self.assertTrue(self.germs_driver_slack == set(germs))

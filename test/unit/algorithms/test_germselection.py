from __future__ import annotations
import numpy as np
import pygsti.circuits as pc
from pygsti.circuits import Circuit
from pygsti.baseobjs import Label
from pygsti.algorithms import germselection as germsel
from pygsti.modelmembers.operations import StaticArbitraryOp
from . import fixtures
from pygsti.modelpacks.legacy import std1Q_XYI as std
from ..util import BaseCase
from typing import Any

_SEED = 2019


class GermSelectionData(object):
    @classmethod
    def setUpClass(cls):
        super(GermSelectionData, cls).setUpClass()
        # XXX are these acceptible test fixtures?
        cls.good_germs = fixtures.robust_germs
        cls.germ_set = cls.good_germs + \
            pc.list_random_circuits_onelen(fixtures.opLabels, 4, 1, seed=_SEED) + \
            pc.list_random_circuits_onelen(fixtures.opLabels, 5, 1, seed=_SEED) + \
            pc.list_random_circuits_onelen(fixtures.opLabels, 6, 1, seed=_SEED)
        cls.target_model = fixtures.fullTP_model

    def setUp(self):
        super(GermSelectionData, self).setUp()
        self.mdl_target_noisy = fixtures.mdl_target_noisy.copy()


class GermSelectionWithNeighbors(GermSelectionData):
    @classmethod
    def setUpClass(cls):
        super(GermSelectionWithNeighbors, cls).setUpClass()
        cls.neighbors = germsel.randomize_model_list(
            [fixtures.model], randomization_strength=1e-3, num_copies=2, seed=_SEED
        )


class GermSelectionTester(GermSelectionData, BaseCase):
    def test_test_germ_list_finitel(self):
        bSuccess, eigvals_finiteL = germsel.test_germ_set_finitel(
            self.mdl_target_noisy, self.germ_set, length=16, return_spectrum=True, tol=1e-3
        )
        self.assertTrue(bSuccess)
        # TODO assert correctness

    def test_test_germ_list_infl(self):
        bSuccess, eigvals_infiniteL = germsel.test_germ_set_infl(
            self.mdl_target_noisy, self.germ_set, return_spectrum=True, check=True
        )
        self.assertTrue(bSuccess)
        # TODO assert correctness

    def test_num_non_spam_gauge_params(self):
        # XXX hey why is this under germselection? EGN: probabaly b/c it was/is used exclusively here - could move it to a tools module?
        N = germsel._num_non_spam_gauge_params(self.mdl_target_noisy)
        # TODO assert correctness

    def test_calculate_germset_score(self):
        # TODO remove randomness
        maxscore = germsel.compute_germ_set_score(
            self.germ_set, self.mdl_target_noisy
        )
        # TODO assert correctness

    def test_compute_composite_germ_score(self):
        score = germsel.compute_composite_germ_set_score(
            np.sum, model=self.mdl_target_noisy, partial_germs_list=self.germ_set,
            eps=1e-5, germ_lengths=np.array([len(g) for g in self.germ_set])
        )
        # TODO assert correctness

        pDDD = np.zeros((len(self.germ_set), self.mdl_target_noisy.num_params,
                         self.mdl_target_noisy.num_params), 'd')
        score_pDDD = germsel.compute_composite_germ_set_score(
            np.sum, model=self.mdl_target_noisy, partial_germs_list=self.germ_set,
            partial_deriv_dagger_deriv=pDDD, op_penalty=1.0
        )
        # TODO assert correctness

    def test_randomize_model_list(self):
        # XXX does this need coverage?  EGN: does it take a long time?
        neighborhood = germsel.randomize_model_list(
            [fixtures.model], num_copies=3, randomization_strength=1e-3,
            seed=_SEED
        )
        # TODO assert consistency
        # XXX num_copies and randomization_strength really should be optional kwargs
        neighborhood_2 = germsel.randomize_model_list(
            [fixtures.model, fixtures.model], num_copies=None,
            randomization_strength=1e-3, seed=_SEED
        )
        # TODO assert consistency

    def test_get_model_params_raises_on_model_param_mismatch(self):
        gs1 = fixtures.model.copy()
        gs2 = fixtures.model.copy()
        gs3 = fixtures.model.copy()
        gs4 = fixtures.model.copy()
        gs1.set_all_parameterizations("full")
        gs2.set_all_parameterizations("full TP")
        gs3.set_all_parameterizations("full")
        gs4.set_all_parameterizations("full")
        gs3.operations['Gi2'] = np.identity(4, 'd')  # adds non-gauge params but not gauge params
        gs4.operations['Gi2'] = StaticArbitraryOp(np.identity(4, 'd'))  # keeps param counts the same but adds gate

        with self.assertRaises(ValueError):
            germsel._get_model_params([gs1, gs2])  # different number of gauge params
        with self.assertRaises(ValueError):
            germsel._get_model_params([gs1, gs3])  # different number of non-gauge params
        with self.assertRaises(ValueError):
            germsel._get_model_params([gs1, gs4])  # different number of gates

    def test_setup_model_list_warns_on_list_copy(self):
        with self.assertWarns(Warning):
            models = germsel._setup_model_list(
                [self.mdl_target_noisy, self.mdl_target_noisy], num_copies=3,
                randomize=False, randomization_strength=0, seed=_SEED
            )
            # TODO assert correctness

    def test_compute_composite_germ_score_raises_on_missing_data(self):
        with self.assertRaises(ValueError):
            germsel.compute_composite_germ_set_score(np.sum)

        pDDD = np.zeros((len(self.germ_set), self.mdl_target_noisy.num_params,
                         self.mdl_target_noisy.num_params), 'd')
        with self.assertRaises(ValueError):
            germsel.compute_composite_germ_set_score(np.sum, partial_deriv_dagger_deriv=pDDD)
        with self.assertRaises(ValueError):
            germsel.compute_composite_germ_set_score(
                np.sum, model=self.mdl_target_noisy,
                partial_deriv_dagger_deriv=pDDD, op_penalty=1.0
            )

    def test_randomize_model_list_raises_on_conflicting_arg_spec(self):
        with self.assertRaises(ValueError):
            germsel.randomize_model_list(
                [fixtures.model, fixtures.model], num_copies=3,
                randomization_strength=0, seed=_SEED
            )


class GenerateGermsTester(GermSelectionData, BaseCase):
    def test_generate_germs_with_candidate_germ_counts(self):
        germs = germsel.find_germs(
            self.mdl_target_noisy, randomize=False,
            candidate_germ_counts={3: 'all upto', 4: 10, 5: 10, 6: 10},
            candidate_seed=1234
        )
        # TODO assert correctness

    def test_generate_germs_raises_on_bad_algorithm(self):
        with self.assertRaises(ValueError):
            germsel.find_germs(self.mdl_target_noisy, algorithm='foobar')


class SlackGermSetOptimizationTester(GermSelectionData, BaseCase):
    def test_optimize_integer_germs_slack_with_fixed_slack(self):
        finalGerms = germsel.find_germs_integer_slack(
            self.mdl_target_noisy, self.germ_set, fixed_slack=0.1,
            verbosity=4
        )
        # TODO assert correctness

        finalGerms_all, weights, scores = germsel.find_germs_integer_slack(
            self.mdl_target_noisy, self.germ_set, fixed_slack=0.1,
            return_all=True, verbosity=4
        )
        self.assertEqual(finalGerms, finalGerms_all)

        algorithm_kwargs = dict(
            germs_list=self.germ_set,
            fixed_slack=0.1
        )
        finalGerms_driver = germsel.find_germs(
            self.mdl_target_noisy, randomize=False, algorithm='slack',
            algorithm_kwargs=algorithm_kwargs, verbosity=4
        )
        self.assertEqual(finalGerms_driver, finalGerms)

    def test_optimize_integer_germs_slack_with_slack_fraction(self):
        finalGerms = germsel.find_germs_integer_slack(
            self.mdl_target_noisy, self.germ_set, slack_frac=0.1,
            verbosity=4
        )
        # TODO assert correctness

    def test_optimize_integer_germs_slack_with_initial_weights(self):
        finalGerms = germsel.find_germs_integer_slack(
            self.mdl_target_noisy, self.germ_set,
            initial_weights=np.ones(len(self.germ_set), 'd'),
            fixed_slack=0.1, verbosity=4
        )
        # TODO assert correctness

    def test_optimize_integer_germs_slack_force_strings(self):
        forceStrs = [Circuit([Label('Gxpi2',0)], line_labels = (0,)),
                    Circuit([Label('Gypi2',0)], line_labels = (0,))]
        finalGerms = germsel.find_germs_integer_slack(
            self.mdl_target_noisy, self.germ_set, fixed_slack=0.1,
            force=forceStrs, verbosity=4,
        )

    def test_optimize_integer_germs_slack_max_iterations(self):
        finalGerms = germsel.find_germs_integer_slack(
            self.mdl_target_noisy, self.germ_set, fixed_slack=0.1,
            max_iter=1, verbosity=4
        )
        self.assertEqual(finalGerms, self.germ_set)

    def test_optimize_integer_germs_slack_raises_on_initial_weight_length_mismatch(self):
        with self.assertRaises(ValueError):
            germsel.find_germs_integer_slack(
                self.mdl_target_noisy, self.germ_set,
                initial_weights=np.ones(1, 'd'),
                fixed_slack=0.1, verbosity=4
            )

    def test_optimize_integer_germs_slack_raises_on_missing_param(self):
        # XXX is this a useful test?  EGN: Probably not.
        with self.assertRaises(ValueError):
            germsel.find_germs_integer_slack(self.mdl_target_noisy, self.germ_set)


class GRASPGermSetOptimizationTester(GermSelectionWithNeighbors, BaseCase):
    def setUp(self):
        super(GRASPGermSetOptimizationTester, self).setUp()
        self.options = dict(
            randomize=False,
            threshold=1e6,
            l1_penalty=1.0,
            iterations=1,
            seed=_SEED,
            verbosity=1
        )

    def test_grasp_germ_set_optimization(self):
        soln = germsel.find_germs_grasp(
            self.neighbors, self.germ_set, alpha=0.1, **self.options
        )
        # TODO assert correctness

        best, initial, local = germsel.find_germs_grasp(
            self.neighbors, self.germ_set, alpha=0.1, return_all=True,
            **self.options
        )
        # TODO shouldn't this pass?
        # self.assertEqual(soln, best)

        algorithm_kwargs = dict(
            germs_list=self.germ_set,
            **self.options
        )
        soln_driver = germsel.find_germs(
            self.mdl_target_noisy, randomize=False, algorithm='grasp',
            algorithm_kwargs=algorithm_kwargs
        )
        # TODO assert correctness

    def test_grasp_germ_set_optimization_force_strings(self):
        forceStrs = [Circuit([Label('Gxpi2',0)], line_labels = (0,)),
                    Circuit([Label('Gypi2',0)], line_labels = (0,))]
        soln = germsel.find_germs_grasp(
            self.neighbors, self.germ_set, alpha=0.1, force=forceStrs,
            **self.options
        )
        for string in forceStrs:
            self.assertIn(string, soln)

    def test_grasp_germ_set_optimization_returns_none_on_incomplete_initial_set(self):
        soln = germsel.find_germs_grasp(
            self.neighbors, self.good_germs[:-1], alpha=0.1, **self.options
        )
        self.assertIsNone(soln)


class GreedyGermSelectionTester(GermSelectionWithNeighbors, BaseCase):
    def setUp(self):
        super(GreedyGermSelectionTester, self).setUp()
        self.options = dict(
            randomize=False,
            threshold=1e6,
            op_penalty=1.0,
            seed=_SEED,
            verbosity=1
        )

    def test_build_up(self):
        germs = germsel.find_germs_depthfirst(self.neighbors, self.germ_set, **self.options)
        # TODO assert correctness

    def test_build_up_force_strings(self):
        forceStrs = [Circuit([Label('Gxpi2',0)], line_labels = (0,)),
                    Circuit([Label('Gypi2',0)], line_labels = (0,))]
        germs = germsel.find_germs_depthfirst(
            self.neighbors, self.germ_set, force=forceStrs, **self.options
        )
        # TODO assert correctness

    def test_build_up_returns_none_on_incomplete_initial_set(self):
        germs = germsel.find_germs_depthfirst(
            self.neighbors, self.good_germs[:-1], **self.options
        )
        self.assertIsNone(germs)

    def test_build_up_breadth(self):
        germs = germsel.find_germs_breadthfirst(self.neighbors, self.germ_set, **self.options)
        # TODO assert correctness

        algorithm_kwargs = dict(
            germs_list=self.germ_set,
            **self.options
        )
        germs_driver = germsel.find_germs(
            self.mdl_target_noisy, randomize=False, algorithm='greedy',
            algorithm_kwargs=algorithm_kwargs
        )
        # TODO assert correctness

    def test_build_up_breadth_force_strings(self):
        forceStrs = [Circuit([Label('Gxpi2',0)], line_labels = (0,)),
                    Circuit([Label('Gypi2',0)], line_labels = (0,))]
        germs = germsel.find_germs_breadthfirst(
            self.neighbors, self.germ_set, force=forceStrs, **self.options
        )
        # TODO assert correctness

    def test_build_up_breadth_returns_none_on_incomplete_initial_set(self):
        germs = germsel.find_germs_breadthfirst(
            self.neighbors, self.good_germs[:-1], **self.options
        )
        self.assertIsNone(germs)

    def test_build_up_breadth_raises_on_out_of_memory(self):
        with self.assertRaises(MemoryError):
            germsel.find_germs_breadthfirst(
                self.neighbors, self.germ_set, mem_limit=1024,
                **self.options
            )
    
    def test_greedy_low_rank_update(self):
        # TODO assert correctness
        germs = germsel.find_germs(self.target_model, seed=2017, 
                                   candidate_germ_counts={3: 'all upto', 4: 10, 5:10, 6:10},
                                   randomize=False, algorithm='greedy', mode='compactEVD',
                                   assume_real=True, float_type=np.double,  verbosity=1)
                                   
    def test_forced_germs_none(self):
        # TODO assert correctness
        #make sure that the germ selection doesn't die with force is None
        germs_compactEVD = germsel.find_germs(self.target_model, seed=2017, 
                                   candidate_germ_counts={3: 'all upto', 4: 10, 5:10, 6:10},
                                   randomize=False, algorithm='greedy', mode='compactEVD',
                                   assume_real=True, float_type=np.double,  verbosity=1, force=None)
        germs_allJac = germsel.find_germs(self.target_model, seed=2017, 
                                   candidate_germ_counts={3: 'all upto', 4: 10, 5:10, 6:10},
                                   randomize=False, algorithm='greedy', mode='all-Jac',
                                   assume_real=True, float_type=np.double,  verbosity=1, force=None)
        germs_singleJac = germsel.find_germs(self.target_model, seed=2017, 
                                   candidate_germ_counts={3: 'all upto', 4: 10, 5:10, 6:10},
                                   randomize=False, algorithm='greedy', mode='single-Jac',
                                   assume_real=True, float_type=np.double,  verbosity=1, force=None)
    
    def test_force_germs_outside_candidate_set(self):
        #TODO assert correctness
        #make sure that the germ selection doesn't die when the list of forced germs includes circuits
        #outside the initially specified candidate set.
        germs = germsel.find_germs(self.target_model, seed=2017, 
                                   candidate_germ_counts={3: 'all upto', 4: 10, 5:10, 6:10},
                                   randomize=False, algorithm='greedy', mode='compactEVD',
                                   assume_real=True, float_type=np.double,  verbosity=1, 
                                   force=pc.list_random_circuits_onelen(fixtures.opLabels, length=7, count=2, seed=_SEED))
                                   
class GermSelectionPenaltyTester(GermSelectionData, BaseCase):

    common_kwargs  : dict[str, Any] = dict(
        randomize=True, seed=1234, candidate_germ_counts={7:'all upto'},
        assume_real=True, float_type=np.double, mode='compactEVD', algorithm='greedy'
    )

    def setUp(self):
        super(GermSelectionData, self).setUp()

    def test_op_penalty_greedy(self):
        germs_no_penalty_cevd = germsel.find_germs(self.target_model, **self.common_kwargs)
        germs_penalty_cevd    = germsel.find_germs(self.target_model, **self.common_kwargs, algorithm_kwargs={'op_penalty':.1})        
        assert count_ops(germs_no_penalty_cevd) > count_ops(germs_penalty_cevd)


    def test_gate_penalty_greedy(self):
        kwargs = self.common_kwargs.copy()
        kwargs.pop('mode')
        germs_no_penalty_alljac = germsel.find_germs(self.target_model, **kwargs, mode='all-Jac')     
        germs_no_penalty_cevd   = germsel.find_germs(self.target_model, **kwargs, mode='compactEVD')
        germs_penalty_alljac = germsel.find_germs(
            self.target_model, **kwargs, mode='all-Jac',    algorithm_kwargs={'gate_penalty':{'Gxpi2':.1}}
        )        
        germs_penalty_cevd = germsel.find_germs(
            self.target_model, **kwargs, mode='compactEVD', algorithm_kwargs={'gate_penalty':{'Gxpi2':.1}}
        )
        assert count_gate(germs_no_penalty_alljac, 'Gxpi2') > count_gate(germs_penalty_alljac, 'Gxpi2')
        assert count_gate(germs_no_penalty_cevd,   'Gxpi2') > count_gate(germs_penalty_cevd,   'Gxpi2')
                                           
    def test_gate_penalty_grasp(self):
        kwargs = self.common_kwargs.copy()
        kwargs['mode']      = 'all-Jac'
        kwargs['algorithm'] = 'grasp'
        germs_gate_penalty_grasp = germsel.find_germs(
            self.target_model, **kwargs, algorithm_kwargs={'seed':1234, 'iterations':1, 'gate_penalty':{'Gxpi2':.2}}
        )
        germs_default_grasp = germsel.find_germs(
            self.target_model, **kwargs, algorithm_kwargs={'seed':1234, 'iterations':1}
        )
        assert count_gate(germs_gate_penalty_grasp, 'Gxpi2') < count_gate(germs_default_grasp, 'Gxpi2')

class EndToEndGermSelectionTester(GermSelectionData, BaseCase):

    #This line from our tutorial notebook previously revealed some numerical precision
    #related bugs, and so should be a worthwhile addition to the test suite since it has_key
    #previously proven to be useful as such.
    def lite_germ_selection_end_to_end_test(self):
        liteGerms = germsel.find_germs(self.target_model, randomize=False, algorithm='greedy', verbosity=1,
                                       assume_real=True, float_type=np.double)
        # TODO assert correctness
        
    def robust_germ_selection_end_to_end_test(self):
        robust_germs = germsel.find_germs(self.target_model, seed=2017)
        #todo assert correctness

#helper function
def count_gate(circuits, gate_name):
    num_gate=0
    for circuit in circuits:
        num_gate+=circuit.str.count(gate_name)
    return num_gate

def count_ops(circuits):
    num_ops = 0
    for circuit in circuits:
        num_ops+= circuit.num_gates
    return num_ops

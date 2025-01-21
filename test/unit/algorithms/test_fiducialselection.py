import numpy as np

import pygsti.algorithms.fiducialselection as fs
import pygsti.circuits as pc
from pygsti.circuits import Circuit
from pygsti.baseobjs import Label
import pygsti.models.modelconstruction as mc
from . import fixtures
from ..util import BaseCase


class FiducialSelectionUtilTester(BaseCase):
    def test_build_bitvec_mx(self):
        mx = fs.build_bitvec_mx(3, 1)
        # TODO assert correctness


class FiducialSelectionStdModel(object):
    def setUp(self):
        super(FiducialSelectionStdModel, self).setUp()
        self.model = fixtures.model.copy()
        self.prep_fids = fixtures.prep_fids
        self.meas_fids = fixtures.meas_fids
        self.cand_fiducials = self.prep_fids + self.meas_fids
        
###
# _find_fiducials_integer_slack
#

class OptimizeIntegerFiducialsBase(object):
    def setUp(self):
        super(OptimizeIntegerFiducialsBase, self).setUp()
        self.options = dict(
            verbosity=4
        )

    def test_optimize_integer_fiducials_slack_frac(self):
        fiducials = fs._find_fiducials_integer_slack(
            self.model, self.cand_fiducials, slack_frac=0.1, **self.options
        )
        # TODO assert correctness

    def test_optimize_integer_fiducials_slack_fixed(self):
        fiducials = fs._find_fiducials_integer_slack(
            self.model, self.cand_fiducials, fixed_slack=0.1, **self.options
        )
        # TODO assert correctness

    def test_optimize_integer_fiducials_slack_initial_weights(self):
        weights = np.ones(len(self.cand_fiducials), 'i')
        fiducials = fs._find_fiducials_integer_slack(
            self.model, self.cand_fiducials, fixed_slack=0.1,
            initial_weights=weights, **self.options
        )
        # TODO assert correctness

    def test_optimize_integer_fiducials_slack_return_all(self):
        fiducials, weights, scores = fs._find_fiducials_integer_slack(
            self.model, self.cand_fiducials, slack_frac=0.1, return_all=True,
            **self.options
        )
        # TODO assert correctness

    def test_optimize_integer_fiducials_slack_worst_score_func(self):
        fiducials = fs._find_fiducials_integer_slack(
            self.model, self.cand_fiducials, slack_frac=0.1,
            score_func='worst', **self.options
        )
        # TODO assert correctness

    def test_optimize_integer_fiducials_slack_fixed_num(self):
        fiducials = fs._find_fiducials_integer_slack(
            self.model, self.cand_fiducials, slack_frac=0.1, fixed_num=4,
            **self.options
        )
        # TODO assert correctness

    def test_optimize_integer_fiducials_slack_force_empty(self):
        fiducials = fs._find_fiducials_integer_slack(
            self.model, self.cand_fiducials, slack_frac=0.1, fixed_num=4,
            force_empty=False, **self.options
        )
        # TODO assert correctness

    def test_optimize_integer_fiducials_slack_low_max_iterations(self):
        fiducials = fs._find_fiducials_integer_slack(
            self.model, self.cand_fiducials, slack_frac=0.1, max_iter=1,
            **self.options
        )
        # TODO assert correctness

    def test_optimize_integer_fiducials_slack_insufficient_fiducials(self):
        insuff_fids = [Circuit([Label('Gxpi2',0)], line_labels = (0,))]
        weights = np.ones(len(insuff_fids), 'i')
        fiducials = fs._find_fiducials_integer_slack(
            self.model, insuff_fids, fixed_slack=0.1,
            initial_weights=weights, **self.options
        )
        self.assertIsNone(fiducials)

    def test_optimize_integer_fiducials_slack_raises_on_missing_slack_param(self):
        with self.assertRaises(ValueError):
            fs._find_fiducials_integer_slack(self.model, self.cand_fiducials, **self.options)


class OptimizeIntegerFiducialsExceptionTester(FiducialSelectionStdModel, BaseCase):
    def test_optimize_integer_fiducials_slack_raises_on_missing_method(self):
        with self.assertRaises(Exception):
            fs._find_fiducials_integer_slack(self.model, self.cand_fiducials, fixed_slack=0.1)


class PrepOptimizeIntegerFiducialsStdModelTester(OptimizeIntegerFiducialsBase, FiducialSelectionStdModel, BaseCase):
    def setUp(self):
        super(PrepOptimizeIntegerFiducialsStdModelTester, self).setUp()
        self.options.update(
            prep_or_meas="prep"
        )


class MeasOptimizeIntegerFiducialsStdModelTester(OptimizeIntegerFiducialsBase, FiducialSelectionStdModel, BaseCase):
    def setUp(self):
        super(MeasOptimizeIntegerFiducialsStdModelTester, self).setUp()
        self.options.update(
            prep_or_meas="meas"
        )

###
# test_fiducial_list
#

# XXX class names prefixed with "Test" will be picked up by pytest
class _TestFiducialListBase(object):
    def setUp(self):
        super(_TestFiducialListBase, self).setUp()
        self.fiducials_list = fs._find_fiducials_integer_slack(
            self.model, self.cand_fiducials,
            prep_or_meas=self.prep_or_meas, slack_frac=0.1
        )

    def test_test_fiducial_list(self):
        self.assertTrue(fs.test_fiducial_list(
            self.model, self.fiducials_list, self.prep_or_meas
        ))

    def test_test_fiducial_list_worst_score_func(self):
        self.assertTrue(fs.test_fiducial_list(
            self.model, self.fiducials_list, self.prep_or_meas,
            score_func='worst'
        ))

    def test_test_fiducial_list_return_all(self):
        result, spectrum, score = fs.test_fiducial_list(
            self.model, self.fiducials_list, self.prep_or_meas,
            return_all=True
        )
        # TODO assert correctness


class PrepTestFiducialListTester(_TestFiducialListBase, FiducialSelectionStdModel, BaseCase):
    prep_or_meas = 'prep'


class MeasTestFiducialListTester(_TestFiducialListBase, FiducialSelectionStdModel, BaseCase):
    prep_or_meas = 'meas'


class TestFiducialListExceptionTester(FiducialSelectionStdModel, BaseCase):
    def test_test_fiducial_list_raises_on_bad_method(self):
        with self.assertRaises(ValueError):
            fs.test_fiducial_list(self.model, None, "foobar")

###
# _find_fiducials_grasp
#

class GraspFiducialOptimizationTester(FiducialSelectionStdModel, BaseCase):
    def test_grasp_fiducial_optimization_prep(self):
        fiducials = fs._find_fiducials_grasp(
            self.model, self.cand_fiducials, prep_or_meas="prep", alpha=0.5,
            verbosity=4
        )
        # TODO assert correctness

    def test_grasp_fiducial_optimization_meas(self):
        fiducials = fs._find_fiducials_grasp(
            self.model, self.cand_fiducials, prep_or_meas="meas", alpha=0.5,
            verbosity=4
        )
        # TODO assert correctness

    def test_grasp_fiducial_optimization_raises_on_bad_method(self):
        with self.assertRaises(ValueError):
            fs._find_fiducials_grasp(
                self.model, self.cand_fiducials, prep_or_meas="foobar",
                alpha=0.5, verbosity=4
            )
            
###
# _find_fiducials_greedy
#

class GreedyFiducialOptimizationTester(FiducialSelectionStdModel, BaseCase):
    def test_greedy_fiducial_optimization_prep(self):
        fiducials = fs._find_fiducials_greedy(
            self.model, self.cand_fiducials, prep_or_meas="prep", evd_tol=1e-6,
            verbosity=4
        )
        # TODO assert correctness

    def test_greedy_fiducial_optimization_meas(self):
        fiducials = fs._find_fiducials_greedy(
            self.model, self.cand_fiducials, prep_or_meas="meas", evd_tol=1e-6,
            verbosity=4
        )
        # TODO assert correctness

#End-to-end tests include the candidate list creation and deduping routines.
#For that reason the particular algorithm doesn't really matter so test using greedy.
class EndToEndFiducialOptimizationTester(FiducialSelectionStdModel, BaseCase):
    def test_find_fiducials_non_clifford_dedupe(self):
        fiducials, _ = fs.find_fiducials(
            self.model, candidate_fid_counts = {3:'all upto'}, 
            algorithm = 'greedy', algorithm_kwargs= {'evd_tol':1e-6},
            assume_clifford = False, prep_fids=True, meas_fids=False,
            verbosity=4
        )
        # TODO assert correctness
        # for now at least check it is not None
        self.assertTrue(fiducials is not None)


    def test_find_fiducials_clifford_dedupe(self):
        fiducials, _ = fs.find_fiducials(
            self.model, candidate_fid_counts= {3:'all upto'}, 
            algorithm = 'greedy', algorithm_kwargs= {'evd_tol':1e-6},
            assume_clifford = True, prep_fids=True, meas_fids=False,
            verbosity=4
        )
        # TODO assert correctness
        # for now at least check it is not None
        self.assertTrue(fiducials is not None)
        
    def test_find_fiducials_end_to_end_default(self):
        prepFiducials, measFiducials = fs.find_fiducials(self.model)
        
    def find_fiducials_omit_operations(self):
        target_model_idle = mc.create_explicit_model_from_expressions([('Q0',)], ['Gi','Gx','Gy'],
                                                                     ["I(Q0)", "X(pi/2,Q0)", "Y(pi/2,Q0)"])
        omitIdentityPrepFids, omitIdentityMeasFids = fs.find_fiducials(target_model_idle, omit_identity=False,
                                                                           ops_to_omit=['Gi'])

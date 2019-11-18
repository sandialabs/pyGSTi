import numpy as np

from ..util import BaseCase
from . import fixtures

import pygsti.construction as pc
import pygsti.algorithms.fiducialselection as fs


class FiducialSelectionUtilTester(BaseCase):
    def test_build_bitvec_mx(self):
        mx = fs.build_bitvec_mx(3, 1)
        # TODO assert correctness


class FiducialSelectionStdModel(object):
    def setUp(self):
        super(FiducialSelectionStdModel, self).setUp()
        self.model = fixtures.model.copy()
        self.fiducials = fixtures.fiducials


class FiducialSelectionExtendedModel(FiducialSelectionStdModel):
    def setUp(self):
        super(FiducialSelectionExtendedModel, self).setUp()
        self.fiducials = pc.list_all_circuits(fixtures.opLabels, 0, 2)


###
# optimize_integer_fiducials_slack
#

class OptimizeIntegerFiducialsBase(object):
    def setUp(self):
        super(OptimizeIntegerFiducialsBase, self).setUp()
        self.options = dict(
            verbosity=4
        )

    def test_optimize_integer_fiducials_slack_frac(self):
        fiducials = fs.optimize_integer_fiducials_slack(
            self.model, self.fiducials, slackFrac=0.1, **self.options
        )
        # TODO assert correctness

    def test_optimize_integer_fiducials_slack_fixed(self):
        fiducials = fs.optimize_integer_fiducials_slack(
            self.model, self.fiducials, fixedSlack=0.1, **self.options
        )
        # TODO assert correctness

    def test_optimize_integer_fiducials_slack_initial_weights(self):
        weights = np.ones(len(self.fiducials), 'i')
        fiducials = fs.optimize_integer_fiducials_slack(
            self.model, self.fiducials, fixedSlack=0.1,
            initialWeights=weights, **self.options
        )
        # TODO assert correctness

    def test_optimize_integer_fiducials_slack_return_all(self):
        fiducials, weights, scores = fs.optimize_integer_fiducials_slack(
            self.model, self.fiducials, slackFrac=0.1, returnAll=True,
            **self.options
        )
        # TODO assert correctness

    def test_optimize_integer_fiducials_slack_worst_score_func(self):
        fiducials = fs.optimize_integer_fiducials_slack(
            self.model, self.fiducials, slackFrac=0.1,
            scoreFunc='worst', **self.options
        )
        # TODO assert correctness

    def test_optimize_integer_fiducials_slack_fixed_num(self):
        fiducials = fs.optimize_integer_fiducials_slack(
            self.model, self.fiducials, slackFrac=0.1, fixedNum=4,
            **self.options
        )
        # TODO assert correctness

    def test_optimize_integer_fiducials_slack_force_empty(self):
        fiducials = fs.optimize_integer_fiducials_slack(
            self.model, self.fiducials, slackFrac=0.1, fixedNum=4,
            forceEmpty=False, **self.options
        )
        # TODO assert correctness

    def test_optimize_integer_fiducials_slack_low_max_iterations(self):
        fiducials = fs.optimize_integer_fiducials_slack(
            self.model, self.fiducials, slackFrac=0.1, maxIter=1,
            **self.options
        )
        # TODO assert correctness

    def test_optimize_integer_fiducials_slack_insufficient_fiducials(self):
        insuff_fids = pc.circuit_list([('Gx',)])
        weights = np.ones(len(insuff_fids), 'i')
        fiducials = fs.optimize_integer_fiducials_slack(
            self.model, insuff_fids, fixedSlack=0.1,
            initialWeights=weights, **self.options
        )
        self.assertIsNone(fiducials)

    def test_optimize_integer_fiducials_slack_raises_on_missing_slack_param(self):
        with self.assertRaises(ValueError):
            fs.optimize_integer_fiducials_slack(self.model, self.fiducials, **self.options)


class OptimizeIntegerFiducialsExceptionTester(FiducialSelectionStdModel, BaseCase):
    def test_optimize_integer_fiducials_slack_raises_on_missing_method(self):
        with self.assertRaises(Exception):
            fs.optimize_integer_fiducials_slack(self.model, self.fiducials, fixedSlack=0.1)


class PrepOptimizeIntegerFiducialsStdModelTester(OptimizeIntegerFiducialsBase, FiducialSelectionStdModel, BaseCase):
    def setUp(self):
        super(PrepOptimizeIntegerFiducialsStdModelTester, self).setUp()
        self.options.update(
            prepOrMeas="prep"
        )


class MeasOptimizeIntegerFiducialsStdModelTester(OptimizeIntegerFiducialsBase, FiducialSelectionStdModel, BaseCase):
    def setUp(self):
        super(MeasOptimizeIntegerFiducialsStdModelTester, self).setUp()
        self.options.update(
            prepOrMeas="meas"
        )


# LOL explicit is better than implicit, right?
class PrepOptimizeIntegerFiducialsExtendedModelTester(
        PrepOptimizeIntegerFiducialsStdModelTester, FiducialSelectionExtendedModel):
    pass


class MeasOptimizeIntegerFiducialsExtendedModelTester(
        MeasOptimizeIntegerFiducialsStdModelTester, FiducialSelectionExtendedModel):
    pass


###
# test_fiducial_list
#

# XXX class names prefixed with "Test" will be picked up by nose
class _TestFiducialListBase(object):
    def setUp(self):
        super(_TestFiducialListBase, self).setUp()
        self.fiducials_list = fs.optimize_integer_fiducials_slack(
            self.model, self.fiducials,
            prepOrMeas=self.prep_or_meas, slackFrac=0.1
        )

    def test_test_fiducial_list(self):
        self.assertTrue(fs.test_fiducial_list(
            self.model, self.fiducials_list, self.prep_or_meas
        ))

    def test_test_fiducial_list_worst_score_func(self):
        self.assertTrue(fs.test_fiducial_list(
            self.model, self.fiducials_list, self.prep_or_meas,
            scoreFunc='worst'
        ))

    def test_test_fiducial_list_return_all(self):
        result, spectrum, score = fs.test_fiducial_list(
            self.model, self.fiducials_list, self.prep_or_meas,
            returnAll=True
        )
        # TODO assert correctness


class PrepTestFiducialListTester(_TestFiducialListBase, FiducialSelectionStdModel, BaseCase):
    prep_or_meas = 'prep'


class MeasTestFiducialListTester(_TestFiducialListBase, FiducialSelectionStdModel, BaseCase):
    prep_or_meas = 'prep'


class TestFiducialListExceptionTester(FiducialSelectionStdModel, BaseCase):
    def test_test_fiducial_list_raises_on_bad_method(self):
        with self.assertRaises(ValueError):
            fs.test_fiducial_list(self.model, None, "foobar")


###
# grasp_fiducial_optimization
#

class GraspFiducialOptimizationTester(FiducialSelectionStdModel, BaseCase):
    def test_grasp_fiducial_optimization_prep(self):
        fiducials = fs.grasp_fiducial_optimization(
            self.model, self.fiducials, prepOrMeas="prep", alpha=0.5,
            verbosity=4
        )
        # TODO assert correctness

    def test_grasp_fiducial_optimization_meas(self):
        fiducials = fs.grasp_fiducial_optimization(
            self.model, self.fiducials, prepOrMeas="meas", alpha=0.5,
            verbosity=4
        )
        # TODO assert correctness

    def test_grasp_fiducial_optimization_raises_on_bad_method(self):
        with self.assertRaises(ValueError):
            fs.grasp_fiducial_optimization(
                self.model, self.fiducials, prepOrMeas="foobar",
                alpha=0.5, verbosity=4
            )

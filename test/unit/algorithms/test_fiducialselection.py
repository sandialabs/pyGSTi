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
            self.model, self.fiducials, slack_frac=0.1, **self.options
        )
        # TODO assert correctness

    def test_optimize_integer_fiducials_slack_fixed(self):
        fiducials = fs._find_fiducials_integer_slack(
            self.model, self.fiducials, fixed_slack=0.1, **self.options
        )
        # TODO assert correctness

    def test_optimize_integer_fiducials_slack_initial_weights(self):
        weights = np.ones(len(self.fiducials), 'i')
        fiducials = fs._find_fiducials_integer_slack(
            self.model, self.fiducials, fixed_slack=0.1,
            initial_weights=weights, **self.options
        )
        # TODO assert correctness

    def test_optimize_integer_fiducials_slack_return_all(self):
        fiducials, weights, scores = fs._find_fiducials_integer_slack(
            self.model, self.fiducials, slack_frac=0.1, return_all=True,
            **self.options
        )
        # TODO assert correctness

    def test_optimize_integer_fiducials_slack_worst_score_func(self):
        fiducials = fs._find_fiducials_integer_slack(
            self.model, self.fiducials, slack_frac=0.1,
            score_func='worst', **self.options
        )
        # TODO assert correctness

    def test_optimize_integer_fiducials_slack_fixed_num(self):
        fiducials = fs._find_fiducials_integer_slack(
            self.model, self.fiducials, slack_frac=0.1, fixed_num=4,
            **self.options
        )
        # TODO assert correctness

    def test_optimize_integer_fiducials_slack_force_empty(self):
        fiducials = fs._find_fiducials_integer_slack(
            self.model, self.fiducials, slack_frac=0.1, fixed_num=4,
            force_empty=False, **self.options
        )
        # TODO assert correctness

    def test_optimize_integer_fiducials_slack_low_max_iterations(self):
        fiducials = fs._find_fiducials_integer_slack(
            self.model, self.fiducials, slack_frac=0.1, max_iter=1,
            **self.options
        )
        # TODO assert correctness

    def test_optimize_integer_fiducials_slack_insufficient_fiducials(self):
        insuff_fids = pc.to_circuits([('Gx',)])
        weights = np.ones(len(insuff_fids), 'i')
        fiducials = fs._find_fiducials_integer_slack(
            self.model, insuff_fids, fixed_slack=0.1,
            initial_weights=weights, **self.options
        )
        self.assertIsNone(fiducials)

    def test_optimize_integer_fiducials_slack_raises_on_missing_slack_param(self):
        with self.assertRaises(ValueError):
            fs._find_fiducials_integer_slack(self.model, self.fiducials, **self.options)


class OptimizeIntegerFiducialsExceptionTester(FiducialSelectionStdModel, BaseCase):
    def test_optimize_integer_fiducials_slack_raises_on_missing_method(self):
        with self.assertRaises(Exception):
            fs._find_fiducials_integer_slack(self.model, self.fiducials, fixed_slack=0.1)


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
        self.fiducials_list = fs._find_fiducials_integer_slack(
            self.model, self.fiducials,
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
    prep_or_meas = 'prep'


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
            self.model, self.fiducials, prep_or_meas="prep", alpha=0.5,
            verbosity=4
        )
        # TODO assert correctness

    def test_grasp_fiducial_optimization_meas(self):
        fiducials = fs._find_fiducials_grasp(
            self.model, self.fiducials, prep_or_meas="meas", alpha=0.5,
            verbosity=4
        )
        # TODO assert correctness

    def test_grasp_fiducial_optimization_raises_on_bad_method(self):
        with self.assertRaises(ValueError):
            fs._find_fiducials_grasp(
                self.model, self.fiducials, prep_or_meas="foobar",
                alpha=0.5, verbosity=4
            )

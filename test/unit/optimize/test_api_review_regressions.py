"""Adversarial extension-boundary regressions found during optimizer API review."""
import numpy as np
import pytest

from pygsti.algorithms import core
from pygsti.circuits import Circuit
from pygsti.data import simulate_data
from pygsti.modelpacks import smq1Q_XY, smq1Q_XYI
from pygsti.models.modelparaminterposer import LinearInterposer
from pygsti.optimize import (EvaluationDomain, EvaluationDomainError, OptimizationProblem,
                             OptimizerOutcome, OptimizationStatus)
from .test_optimizer_api import CallbackOptimizer, _ScalarObjective


def test_final_returned_point_must_satisfy_evaluation_domain():
    objective = _ScalarObjective()
    initial = objective.model.to_vector()
    optimizer = CallbackOptimizer(
        lambda problem: OptimizerOutcome([4.], OptimizationStatus.CONVERGED))
    optimizer.supports_evaluation_domain = True
    domain = EvaluationDomain(lambda x: x[0] < 3., generation='fixed-paths')
    # The solver never queried its returned point. Finalization must not bypass
    # domain validity just because scalar statistics happen to be finite there.
    with pytest.raises(EvaluationDomainError):
        optimizer.run(objective, None, None, domain=domain)
    np.testing.assert_array_equal(objective.model.to_vector(), initial)


def test_final_outcome_is_revalidated_after_solver_mutation():
    objective = _ScalarObjective()
    initial = objective.model.to_vector()
    def solve(problem):
        outcome = OptimizerOutcome([1.], OptimizationStatus.CONVERGED)
        outcome.x = np.array([1. + 1.j])
        return outcome
    with pytest.raises(ValueError, match='must be real'):
        CallbackOptimizer(solve).run(objective, None, None)
    np.testing.assert_array_equal(objective.model.to_vector(), initial)


@pytest.mark.parametrize('invalid_value', [np.nan, np.inf, 1.j])
def test_invalid_final_statistics_restore_entry_model(invalid_value):
    objective = _ScalarObjective()
    initial = objective.model.to_vector()
    def terms(x):
        objective.model.from_vector(x)
        return np.array([invalid_value])
    objective.terms = terms
    optimizer = CallbackOptimizer(
        lambda problem: OptimizerOutcome([4.], OptimizationStatus.CONVERGED))
    with pytest.raises(ValueError, match='finite and real'):
        optimizer.run(objective, None, None)
    np.testing.assert_array_equal(objective.model.to_vector(), initial)


def test_finite_terms_with_overflowing_sum_restore_entry_model():
    objective = _ScalarObjective()
    initial = objective.model.to_vector()
    def terms(x):
        objective.model.from_vector(x)
        return np.array([1e308, 1e308])
    objective.terms = terms
    optimizer = CallbackOptimizer(
        lambda problem: OptimizerOutcome([4.], OptimizationStatus.CONVERGED))
    with pytest.raises(ValueError, match='statistics overflowed'):
        optimizer.run(objective, None, None)
    np.testing.assert_array_equal(objective.model.to_vector(), initial)


def test_torch_rejects_unimplemented_model_parameter_interposer():
    pytest.importorskip('torch')
    model = smq1Q_XYI.target_model('full TP', simulator='matrix')
    # A square map is important: all global-index/dimension checks still pass,
    # but treating model coordinates as op coordinates changes probabilities.
    model.param_interposer = LinearInterposer(2. * np.eye(model.num_params))
    circuits = [Circuit([('Gxpi2', 0)], line_labels=(0,))]
    layout = model.sim.create_layout(circuits, array_types=('e', 'ep'))
    with pytest.raises(NotImplementedError, match='interposer'):
        model.sim.probability_function(layout, backend='torch')


def test_custom_gradient_factory_can_plan_scalar_only_legacy_storage():
    model = smq1Q_XY.target_model('full TP', simulator='matrix').depolarize(
        op_noise=.02, spam_noise=.03)
    circuits = smq1Q_XY.prep_fiducials()[:3]
    dataset = simulate_data(model, circuits, 1000, sample_error='none')
    observed = []

    def factory(context, operations):
        objective = context.objective
        assert objective.jac is None
        observed.append(objective)
        # An external solver may explicitly choose numerical differentiation.
        # It uses only scalar legacy evaluations and needs no dense host Jacobian.
        def gradient(x):
            h = 1e-6
            directions = np.eye(len(x)) * h
            return np.array([(objective.fn(x+d) - objective.fn(x-d)) / (2*h)
                             for d in directions])
        return OptimizationProblem(objective.model.to_vector(), objective.fn, gradient=gradient)

    def solve(problem):
        assert np.isfinite(problem.gradient(problem.initial_point)).all()
        return OptimizerOutcome(problem.initial_point, OptimizationStatus.CONVERGED)

    optimizer = CallbackOptimizer(solve, required_operations=('value', 'gradient'),
                                 problem_factory=factory, objective_methods=('fn',))
    core.run_gst_fit_simple(dataset, model, circuits, optimizer, 'chi2', None)
    assert len(observed) == 1
    assert observed[0].jac is None

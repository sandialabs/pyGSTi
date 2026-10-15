"""The canonical LM entry preserves tuned GST options and solves numeric problems."""
import numpy as np
import pytest

from pygsti.optimize import Optimizer, LMOptimizer, SimplerLMOptimizer, OptimizationProblem, OptimizationStatus


def test_default_cast_and_legacy_option_roundtrip():
    optimizer = Optimizer.cast(dict(maxiter=7, tol={'f': 1e-12},
                                   lsvec_mode='percircuit', linesearch={'mode': 'always'}))
    assert isinstance(optimizer, LMOptimizer)
    assert optimizer.legacy.maxiter == 7
    assert optimizer.legacy.lsvec_mode == 'percircuit'
    restored = Optimizer.loads(optimizer.dumps())
    assert isinstance(restored, LMOptimizer)
    assert restored.legacy.tol == optimizer.legacy.tol
    assert restored.legacy.linesearch == optimizer.legacy.linesearch
    assert restored.legacy.lsvec_mode == 'percircuit'
    assert restored.array_types == optimizer.legacy.array_types
    assert restored.called_objective_methods == optimizer.legacy.called_objective_methods
    legacy = SimplerLMOptimizer()
    assert Optimizer.cast(legacy) is legacy


def test_tuned_lm_on_independent_rank_deficient_problem():
    def residual(x):
        return np.array([x[0] ** 2 - 1, x[1] - 2])
    problem = OptimizationProblem(
        [1.2, 0., 7.], lambda x: np.dot(residual(x), residual(x)),
        residual=residual,
        residual_jacobian=lambda x: np.array([[2 * x[0], 0., 0.], [0., 1., 0.]]))
    result = LMOptimizer(tol={'f': 1e-16, 'jac': 1e-10}).solve(problem)
    assert result.status == OptimizationStatus.CONVERGED
    assert problem.value(result.x) < 1e-14
    np.testing.assert_allclose(result.x, [1., 2., 7.], atol=1e-7)


def test_gst_only_controls_are_not_silently_used_for_numeric_problem():
    problem = OptimizationProblem([1.], lambda x: x[0] ** 2,
                                  residual=lambda x: x,
                                  residual_jacobian=lambda x: np.ones((1, 1)))
    with pytest.raises(ValueError, match='apply to GST'):
        LMOptimizer(lsvec_mode='percircuit').solve(problem)

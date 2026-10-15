"""Term-path restarts use explicit domain failures and solver-owned incumbents."""
from types import SimpleNamespace
from unittest import mock
import numpy as np
import pytest

from pygsti.algorithms.core import _do_term_runopt
from pygsti.baseobjs import VerbosityPrinter
from pygsti.baseobjs.profiler import DummyProfiler
from pygsti.optimize import (Optimizer, OptimizerOutcome, OptimizationStatus,
                             OptimizerResult, EvaluationDomainError, StaleEvaluationDomainError)


class Model:
    parameter_bounds = None

    def __init__(self):
        self.vector = np.array([1.])
        self.sim = Paths()

    def to_vector(self):
        return self.vector.copy()

    def from_vector(self, x):
        self.vector[:] = x


class Paths:
    max_term_stages = 3
    path_fraction_threshold = .5
    oob_check_interval = 1
    limit = 2.

    def bulk_test_if_paths_are_sufficient(self, layout, probabilities, verbosity=0):
        return bool(probabilities[0] <= self.limit)

    def find_minimal_paths_set(self, layout):
        return SimpleNamespace(allowed_path_fraction=1., num_failures=0)

    def select_paths_set(self, layout, paths):
        layout.pathset = paths
        self.limit = 4.


class Objective:
    ex = local_ex = 0

    def __init__(self):
        self.model = Model()
        self.probs = np.zeros(1)
        self.resource_alloc = SimpleNamespace(comm=None, profiler=DummyProfiler())
        self.layout = SimpleNamespace(
            pathset=SimpleNamespace(allowed_path_fraction=.1, num_failures=0),
            allgather_local_array=lambda kind, values, **kwargs: values,
            allsum_local_quantity=lambda kind, value, **kwargs: value)

    def lsvec(self, x=None):
        if x is not None:
            self.model.from_vector(x)
        self.probs[:] = self.model.to_vector()
        return self.model.to_vector() - 1.5

    def dlsvec(self, x):
        self.lsvec(x)
        return np.ones((1, 1))

    def terms(self, x):
        return self.lsvec(x) ** 2

    def fn(self, x=None):
        return float(np.dot(self.lsvec(x), self.lsvec(x)))

    def chi2k_distributed_qty(self, value):
        return value


def test_domain_restart_keeps_solver_incumbent_and_invalidates_old_linearization():
    objective = Objective()
    class External(Optimizer):
        supports_evaluation_domain = True
        required_operations = ('value', 'residual', 'residual_jacobian')

        def __init__(self):
            super().__init__()
            self.calls = 0
            self.old_model = None

        def solve(self, problem):
            self.calls += 1
            if self.calls == 1:
                self.old_model = problem.linearize(problem.initial_point)
                problem.value([1.75])  # valid trial, deliberately NOT accepted
                with pytest.raises(EvaluationDomainError):
                    problem.value([3.])
                return OptimizerOutcome(problem.initial_point, OptimizationStatus.DOMAIN_LIMIT)
            np.testing.assert_array_equal(problem.initial_point, [1.])
            with pytest.raises(StaleEvaluationDomainError):
                self.old_model.jvp([1.])
            assert problem.domain.generation == 1
            return OptimizerOutcome([1.5], OptimizationStatus.CONVERGED)
    optimizer = External()
    result = _do_term_runopt(objective, optimizer, VerbosityPrinter(0))
    assert optimizer.calls == 2
    assert result.outcome.status == OptimizationStatus.CONVERGED
    np.testing.assert_array_equal(objective.model.to_vector(), [1.5])


def test_unsupported_external_domain_fails_before_path_processing():
    with pytest.raises(NotImplementedError, match='evaluation-domain support'):
        _do_term_runopt(Objective(), Optimizer(), VerbosityPrinter(0))


def test_exact_term_simulation_does_not_require_an_approximation_domain():
    objective = Objective()
    objective.layout.pathset = None
    class External(Optimizer):
        def solve(self, problem):
            assert not hasattr(problem, 'domain')
            return OptimizerOutcome([1.5], OptimizationStatus.CONVERGED)
    result = _do_term_runopt(objective, External(), VerbosityPrinter(0))
    assert result.f == 0


@pytest.mark.parametrize('legacy', [False, True])
@pytest.mark.parametrize('finality', ['stage_budget', 'path_fraction'])
def test_final_domain_stop_preserves_evaluated_pathset_and_statistics(legacy, finality):
    objective = Objective()
    if finality == 'stage_budget':
        objective.model.sim.max_term_stages = 1
    else:
        objective.layout.pathset.allowed_path_fraction = .8
    final_paths = objective.layout.pathset
    class External(Optimizer):
        supports_evaluation_domain = True

        def solve(self, problem):
            problem.value([1.75])  # valid but not accepted
            return OptimizerOutcome([1.5], OptimizationStatus.DOMAIN_LIMIT)
    class LegacyStop(Optimizer):
        uses_problem_api = False

        def run(self, objective, profiler, printer):
            x = np.array([1.5])
            return OptimizerResult(
                objective, x, objective.fn(x),
                optimizer_specific_qtys={'msg': 'Objective function out-of-bounds! STOP',
                                         'mu': 1., 'nu': 2.})
    simulator = objective.model.sim
    with mock.patch.object(simulator, 'find_minimal_paths_set',
                           wraps=simulator.find_minimal_paths_set) as rebuild:
        result = _do_term_runopt(objective, LegacyStop() if legacy else External(), VerbosityPrinter(0))
    rebuild.assert_not_called()
    assert objective.layout.pathset is final_paths
    assert result.f == objective.fn(result.x)
    np.testing.assert_array_equal(objective.model.to_vector(), [1.5])
    if not legacy:
        assert result.outcome.status == OptimizationStatus.DOMAIN_LIMIT

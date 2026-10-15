"""Validity-domain failures differ from bounds, solver state, and code errors."""
import unittest
import pickle
import numpy as np

from pygsti.optimize.domain import (
    EvaluationDomain, EvaluationDomainError, StaleEvaluationDomainError, guard_problem)
from pygsti.optimize.problem import OptimizationProblem


class EvaluationDomainTester(unittest.TestCase):
    def test_typed_failure_contains_owned_trial_without_acceptance_state(self):
        trial = np.array([3.])
        domain = EvaluationDomain(lambda x: abs(x[0]) < 2, 'insufficient paths', generation=4)
        with self.assertRaises(EvaluationDomainError) as caught:
            domain.require(trial)
        trial[:] = 9
        np.testing.assert_array_equal(caught.exception.point, [3.])
        self.assertEqual(caught.exception.generation, 4)
        self.assertEqual(str(caught.exception), 'insufficient paths')
        self.assertFalse(caught.exception.point.flags.writeable)
        restored = pickle.loads(pickle.dumps(caught.exception))
        np.testing.assert_array_equal(restored.point, [3.])
        self.assertEqual(restored.generation, 4)
        self.assertTrue(domain.check([1.]))
        self.assertFalse(domain.check([3.]))

    def test_unrelated_exception_is_not_a_domain_failure(self):
        def broken_check(x):
            raise ValueError('simulation programming error')
        domain = EvaluationDomain(broken_check)
        with self.assertRaisesRegex(ValueError, 'simulation programming error'):
            domain.require([0.])
        with self.assertRaises(TypeError):
            EvaluationDomain(lambda x: None).require([0.])

    def test_guarded_oracle_never_evaluates_invalid_trial(self):
        evaluations = []
        def value(x):
            evaluations.append(x.copy())
            return x[0] ** 2
        problem = OptimizationProblem([0.], value, bounds=([-10.], [10.]))
        domain = EvaluationDomain(lambda x: abs(x[0]) < 2)
        guarded = guard_problem(problem, domain)
        self.assertEqual(guarded.value([1.]), 1.)
        with self.assertRaises(EvaluationDomainError):
            guarded.value([3.])  # inside box bounds but outside approximation domain
        self.assertEqual(len(evaluations), 1)
        self.assertEqual(guarded.capabilities, problem.capabilities)
        np.testing.assert_array_equal(guarded.initial_point, [0.])

    def test_new_generation_and_product_linearization_invalidation(self):
        problem = OptimizationProblem(
            [1.], lambda x: x[0] ** 4, residual=lambda x: x ** 2,
            residual_jvp=lambda x, v: 2 * x * v,
            residual_vjp=lambda x, w: 2 * x * w)
        old = EvaluationDomain(lambda x: abs(x[0]) < 2, generation=0)
        old_problem = guard_problem(problem, old)
        linearization = old_problem.linearize([1.])
        np.testing.assert_array_equal(linearization.jvp([1.]), [2.])
        old.invalidate()
        with self.assertRaises(StaleEvaluationDomainError):
            old_problem.value([1.])
        with self.assertRaises(StaleEvaluationDomainError):
            linearization.jvp([1.])
        new = EvaluationDomain(lambda x: abs(x[0]) < 4, generation=1)
        self.assertEqual(guard_problem(problem, new).value([3.]), 81.)

    def test_native_structure_requires_explicit_domain_handling(self):
        problem = OptimizationProblem([0.], lambda x: 0., structure=object())
        with self.assertRaises(NotImplementedError):
            guard_problem(problem, EvaluationDomain(lambda x: True))

    def test_dense_cached_linearization_is_invalidated(self):
        problem = OptimizationProblem(
            [1.], lambda x: x[0] ** 2, residual=lambda x: x,
            residual_jacobian=lambda x: np.ones((1, 1)))
        domain = EvaluationDomain(lambda x: True)
        linearization = guard_problem(problem, domain).linearize([1.])
        domain.invalidate()
        for operation in (lambda: linearization.jvp([1.]),
                          lambda: linearization.vjp([1.]), linearization.to_dense):
            with self.assertRaises(StaleEvaluationDomainError):
                operation()

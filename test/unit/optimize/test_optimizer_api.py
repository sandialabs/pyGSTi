"""Integration checks for independent optimizers and the GST adapter lifecycle."""
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

from pygsti.algorithms import core
from pygsti.data import simulate_data
from pygsti.modelpacks import smq1Q_XY
from pygsti.objectivefns import ObjectiveFunctionBuilder
from pygsti.optimize import Optimizer, OptimizerOutcome, OptimizationStatus
from pygsti.optimize.gstproblem import default_problem_factory
from pygsti.protocols import gst


class CallbackOptimizer(Optimizer):
    """An external implementation using only the documented numerical problem."""

    def __init__(self, callback, **kwargs):
        super().__init__(**kwargs)
        self.callback = callback
        self.calls = 0

    def solve(self, problem):
        self.calls += 1
        return self.callback(problem)


class _ScalarModel:
    parameter_bounds = None

    def __init__(self):
        self.vector = np.array([2.])

    def to_vector(self):
        return self.vector.copy()

    def from_vector(self, x):
        self.vector[:] = x


class _ScalarObjective:
    """Penalty-bearing scalar objective with no residual/derivative interface."""

    def __init__(self):
        self.model = _ScalarModel()
        self.ex = self.local_ex = 1
        self.resource_alloc = SimpleNamespace(comm=None)
        self.layout = SimpleNamespace(
            allsum_local_quantity=lambda kind, value, **kwargs: value)

    def terms(self, x):
        self.model.from_vector(x)
        return np.array([(x[0] - 1) ** 2, 7.])

    def fn(self, x):
        return self.terms(x).sum()

    def chi2k_distributed_qty(self, value):
        return 2 * value


class OptimizerLifecycleTester(unittest.TestCase):
    def test_scalar_only_final_point_penalties_and_status(self):
        objective = _ScalarObjective()
        accepted = np.array([np.nextafter(1., 2.)])
        def solve(problem):
            self.assertEqual(problem.capabilities, frozenset({'value'}))
            self.assertEqual(problem.value(accepted), 7.)
            problem.value([3.])  # rejected trial left installed in the model
            return OptimizerOutcome(accepted, OptimizationStatus.ITERATION_LIMIT,
                                    'budget exhausted', {'iterations': 1})
        optimizer = CallbackOptimizer(solve)
        result = optimizer.run(objective, None, None)
        np.testing.assert_array_equal(objective.model.to_vector(), accepted)
        np.testing.assert_array_equal(result.x, accepted)
        self.assertEqual(result.f, 7.)
        self.assertEqual(result.f_no_penalties, (accepted[0] - 1) ** 2)
        self.assertEqual(result.chi2_k_distributed_qty, 14.)
        self.assertEqual(result.outcome.status, OptimizationStatus.ITERATION_LIMIT)
        self.assertEqual(result.optimizer_specific_qtys['iterations'], 1)
        self.assertIsNone(result.jtj)

    def test_exception_restores_entry_model(self):
        objective = _ScalarObjective()
        initial = objective.model.to_vector()
        def solve(problem):
            problem.value([9.])
            raise ArithmeticError('external solver failed')
        with self.assertRaisesRegex(ArithmeticError, 'external solver failed'):
            CallbackOptimizer(solve).run(objective, None, None)
        np.testing.assert_array_equal(objective.model.to_vector(), initial)

    def test_invalid_outcome_restores_entry_model(self):
        for invalid in (None, OptimizerOutcome([1., 2.], OptimizationStatus.CONVERGED)):
            with self.subTest(invalid=invalid):
                objective = _ScalarObjective()
                initial = objective.model.to_vector()
                def solve(problem):
                    problem.value([9.])
                    return invalid
                with self.assertRaises((TypeError, ValueError)):
                    CallbackOptimizer(solve).run(objective, None, None)
                np.testing.assert_array_equal(objective.model.to_vector(), initial)

    def test_final_evaluation_failure_restores_entry_model(self):
        objective = _ScalarObjective()
        initial = objective.model.to_vector()
        original_terms = objective.terms
        def terms(x):
            result = original_terms(x)
            if x[0] == 4.:
                raise ValueError('final statistics failed')
            return result
        objective.terms = terms
        optimizer = CallbackOptimizer(
            lambda problem: OptimizerOutcome([4.], OptimizationStatus.CONVERGED))
        with self.assertRaisesRegex(ValueError, 'final statistics failed'):
            optimizer.run(objective, None, None)
        np.testing.assert_array_equal(objective.model.to_vector(), initial)

    def test_factory_receives_configured_objective_and_declared_operations(self):
        objective = _ScalarObjective()
        captured = []
        def factory(context, operations):
            captured.append((context.objective, context.model, tuple(operations)))
            return default_problem_factory(context, operations)
        optimizer = CallbackOptimizer(
            lambda problem: OptimizerOutcome(problem.initial_point, OptimizationStatus.CONVERGED),
            problem_factory=factory)
        optimizer.run(objective, None, None)
        self.assertEqual(captured, [(objective, objective.model, ('value',))])


class RealGSTOptimizerTester(unittest.TestCase):
    def setUp(self):
        self.model = smq1Q_XY.target_model('full TP', simulator='matrix').depolarize(
            op_noise=0.02, spam_noise=0.03)
        self.circuits = smq1Q_XY.prep_fiducials()[:3]
        self.dataset = simulate_data(self.model, self.circuits, 1000, sample_error='none')

    def test_external_scalar_optimizer_through_real_gst_kernel(self):
        captured = {}
        initial = self.model.to_vector().copy()
        accepted = initial.copy()
        accepted[0] = np.nextafter(accepted[0], np.inf)
        trial = accepted.copy()
        trial[0] += 1e-12  # deliberately allclose, but not equal, to accepted
        def factory(context, operations):
            captured['objective'] = context.objective
            return default_problem_factory(context, operations)
        def solve(problem):
            self.assertEqual(problem.capabilities, frozenset({'value'}))
            self.assertTrue(np.isfinite(problem.value(accepted)))
            problem.value(trial)
            return OptimizerOutcome(accepted, OptimizationStatus.CONVERGED)
        optimizer = CallbackOptimizer(solve, problem_factory=factory)
        result, fitted = core.run_gst_fit_simple(
            self.dataset, self.model, self.circuits, optimizer, 'chi2', None)
        self.assertEqual(optimizer.calls, 1)
        np.testing.assert_array_equal(result.x, accepted)
        np.testing.assert_array_equal(fitted.to_vector(), accepted)
        self.assertEqual(result.f, captured['objective'].fn(accepted))
        self.assertIsNone(captured['objective'].jac)

    def test_real_configured_callable_penalty_is_preserved(self):
        penalty = 2.75
        builder = ObjectiveFunctionBuilder.create_from(('chi2', lambda model: penalty))
        seen_values = []
        def solve(problem):
            seen_values.append(problem.value(problem.initial_point))
            return OptimizerOutcome(problem.initial_point, OptimizationStatus.CONVERGED)
        result, _ = core.run_gst_fit_simple(
            self.dataset, self.model, self.circuits, CallbackOptimizer(solve), builder, None)
        self.assertAlmostEqual(result.f - result.f_no_penalties, penalty)
        self.assertAlmostEqual(seen_values[0], result.f)

    def test_protocol_constructor_and_override_lists_preserve_external_objects(self):
        optimizer = CallbackOptimizer(
            lambda problem: OptimizerOutcome(problem.initial_point, OptimizationStatus.CONVERGED))
        second = CallbackOptimizer(
            lambda problem: OptimizerOutcome(problem.initial_point, OptimizationStatus.ITERATION_LIMIT))
        protocol = gst.StandardGST(modes=('full TP',), target_model=self.model,
                                   optimizer=optimizer, verbosity=0)
        self.assertIs(protocol.optimizer, optimizer)
        self.assertEqual(core.validate_and_extend_optimizer([optimizer], 2), [optimizer, optimizer])
        self.assertEqual(gst._validate_and_extend_optimizers(
            [optimizer, second], 2, self.model), [optimizer, second])

    def test_protocol_run_uses_external_override_and_stores_fitted_model(self):
        from pygsti.protocols.protocol import ProtocolData
        design = gst.GateSetTomographyDesign(
            smq1Q_XY.processor_spec(), [self.circuits], qubit_labels=(0,))
        data = ProtocolData(design, self.dataset)
        default = CallbackOptimizer(lambda problem: self.fail('Default solver was not overridden'))
        seen = []
        def solve(problem):
            seen.append(problem.value(problem.initial_point))
            return OptimizerOutcome(problem.initial_point, OptimizationStatus.ITERATION_LIMIT,
                                    'Explicit zero-step demonstration')
        override = CallbackOptimizer(solve)
        protocol = gst.GateSetTomography(
            initial_model=self.model, optimizer=default, gaugeopt_suite={},
            verbosity=0, name='external')
        results = protocol.run(data, optimizers=override, disable_checkpointing=True)
        fitted = results.estimates['external'].models['final iteration estimate']
        assert seen and all(np.isfinite(seen))
        assert default.calls == 0
        assert override.calls == len(seen)
        np.testing.assert_array_equal(fitted.to_vector(), self.model.to_vector())

    def test_later_gradient_optimizer_has_storage_in_iterative_gst(self):
        # The first stage is scalar-only. Planning from only that stage leaves
        # the later objective without derivative storage and makes this fail.
        gradient_norms = []
        def scalar_solve(problem):
            problem.value(problem.initial_point)
            return OptimizerOutcome(problem.initial_point, OptimizationStatus.CONVERGED)
        def gradient_solve(problem):
            gradient = problem.gradient(problem.initial_point)
            self.assertEqual(gradient.shape, problem.initial_point.shape)
            self.assertTrue(np.all(np.isfinite(gradient)))
            gradient_norms.append(np.linalg.norm(gradient))
            return OptimizerOutcome(problem.initial_point, OptimizationStatus.CONVERGED)
        first = CallbackOptimizer(scalar_solve)
        second = CallbackOptimizer(gradient_solve, required_operations=('value', 'gradient'))
        builder = ObjectiveFunctionBuilder.create_from('chi2')
        observed_methods = []
        original_compute = builder.compute_array_types
        def compute_array_types(methods, simulator):
            observed_methods.append(tuple(methods))
            return original_compute(methods, simulator)
        with mock.patch.object(builder, 'compute_array_types', side_effect=compute_array_types):
            results = list(core.iterative_gst_generator(
                self.dataset, self.model, [self.circuits[:2], self.circuits],
                [first, second], [builder], [], None, verbosity=0))
        self.assertEqual(len(results), 2)
        self.assertEqual(first.calls, 1)
        self.assertEqual(second.calls, 1)
        self.assertEqual(len(gradient_norms), 1)
        self.assertTrue(any('dterms' in methods for methods in observed_methods))

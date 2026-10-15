"""Levenberg–Marquardt entry point and compatibility with the tuned GST solver."""

import numpy as np

from .optimizer import Optimizer
from .simplerlm import SimplerLMOptimizer, simplish_leastsq
from .arraysinterface import UndistributedArraysInterface
from .outcome import OptimizerOutcome, OptimizationStatus


class LMOptimizer(Optimizer):
    """Use the existing tuned LM implementation through the optimizer interface.

    All keyword options accepted by SimplerLMOptimizer remain available. GST
    ``run`` delegates to that implementation, preserving distributed linear
    algebra and its specialized domain controls without global-array gathering.
    ``solve`` also supports an independent numerical OptimizationProblem.
    """

    required_operations = ('value', 'residual', 'residual_jacobian')
    uses_problem_api = False

    def __init__(self, **options):
        super().__init__()
        self.legacy = SimplerLMOptimizer(**options)
        # Replicated entry snapshot and owned outcome, beyond legacy workspace.
        self.array_types = self.legacy.array_types + ('P', 'P')
        self.called_objective_methods = self.legacy.called_objective_methods

    def run(self, objective, profiler, printer):
        initial = np.array(objective.model.to_vector(), copy=True)
        try:
            return self._run_gst(objective, profiler, printer)
        except BaseException:
            objective.model.from_vector(initial)
            raise

    def _run_gst(self, objective, profiler, printer):
        result = self.legacy.run(objective, profiler, printer)
        # The historical adapter uses allclose here. The public result owns an
        # exact accepted point, even after a nearby rejected trial evaluation.
        objective.model.from_vector(result.x)
        diagnostics = result.optimizer_specific_qtys
        message = diagnostics['msg']
        if message.startswith('Maximum iterations'):
            status = OptimizationStatus.ITERATION_LIMIT
        elif message == 'Objective function out-of-bounds! STOP':
            status = OptimizationStatus.DOMAIN_LIMIT
        elif message == 'Keyboard interrupt!':
            status = OptimizationStatus.INTERRUPTED
        else:
            status = OptimizationStatus.CONVERGED
        result.outcome = OptimizerOutcome(result.x, status, message, {'value': result.f},
                                          {'mu': diagnostics['mu'], 'nu': diagnostics['nu']})
        return result

    def solve(self, problem):
        options = self.legacy
        if options.lsvec_mode != 'normal' or options.oob_check_interval != 0:
            raise ValueError('Per-circuit residual selection and legacy path checks apply to GST run(), '
                             'not an independent numerical problem')
        x0 = problem.initial_point
        residual = problem.residual(x0)
        bounds = problem.bounds
        limits = None if bounds is None else np.column_stack(bounds)
        x, converged, message, mu, nu, value, _ = simplish_leastsq(
            problem.residual, problem.residual_jacobian, x0,
            max_iter=options.maxiter,
            f_norm2_tol=options.tol['f'], jac_norm_tol=options.tol['jac'],
            rel_ftol=options.tol['relf'], rel_xtol=options.tol['relx'],
            max_dx_scale=options.tol['maxdx'], init_munu=options.init_munu,
            linesearch=options.linesearch,
            arrays_interface=UndistributedArraysInterface(residual.size, x0.size),
            x_limits=limits)
        if message.startswith('Maximum iterations'):
            status = OptimizationStatus.ITERATION_LIMIT
        elif message == 'Keyboard interrupt!':
            status = OptimizationStatus.INTERRUPTED
        else:
            status = OptimizationStatus.CONVERGED if converged else OptimizationStatus.FAILED
        return OptimizerOutcome(x, status, message, {'value': value}, {'mu': mu, 'nu': nu})

    # The legacy term-simulator orchestration uses these controls; expose the
    # same behavior on this wrapper until that orchestration adopts the generic
    # evaluation-domain contract.
    @property
    def oob_check_interval(self):
        return self.legacy.oob_check_interval

    @oob_check_interval.setter
    def oob_check_interval(self, value):
        self.legacy.oob_check_interval = value

    @property
    def oob_action(self):
        return self.legacy.oob_action

    @oob_action.setter
    def oob_action(self, value):
        self.legacy.oob_action = value

    @property
    def init_munu(self):
        return self.legacy.init_munu

    @init_munu.setter
    def init_munu(self, value):
        self.legacy.init_munu = value

    def _to_nice_serialization(self):
        state = super()._to_nice_serialization()
        state['legacy_options'] = self.legacy.to_nice_serialization()
        return state

    @classmethod
    def _from_nice_serialization(cls, state):
        result = cls()
        result.legacy = SimplerLMOptimizer.from_nice_serialization(state['legacy_options'])
        result.array_types = result.legacy.array_types + ('P', 'P')
        result.called_objective_methods = result.legacy.called_objective_methods
        return result

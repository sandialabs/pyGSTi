"""Adapt model/data/circuit objectives to the numerical optimizer contract.

The default adapter returns owned, global NumPy arrays. Its derivatives are
explicitly dense: it does not advertise matrix-free simulation. Alternative
factories can supply native backends or additional mathematical structure.
"""

from dataclasses import dataclass
import numpy as np

from .problem import OptimizationProblem


# These names refer to the *existing* objective implementation, not solver policy.
_OBJECTIVE_METHODS = {
    'value': ('fn',),
    'gradient': ('dterms',),
    'residual': ('lsvec',),
    'residual_jacobian': ('dlsvec',),
    'residual_jvp': ('dlsvec',),
    'residual_vjp': ('dlsvec',),
    'hessian_vector_product': ('hessian',),
}


def objective_methods(operations):
    """Legacy objective storage needed by explicitly requested operations."""
    methods = ['fn']  # final statistics must also be included in planning
    for operation in operations:
        if operation not in _OBJECTIVE_METHODS:
            raise ValueError("Unknown optimization operation: %s" % operation)
        methods.extend(_OBJECTIVE_METHODS[operation])
    return tuple(dict.fromkeys(methods))


def replicated_array_types(operations):
    """Global-array workspace required by the default dense adapter.

    Uppercase dimensions are global dimensions in layout memory estimates.
    This is in addition to the objective's local simulation storage and an
    optimizer's declared work arrays.
    """
    operations = set(operations)
    arrays = ('P',) * 3
    if operations.intersection(('gradient', 'residual_jacobian', 'residual_jvp', 'residual_vjp')):
        arrays += ('EP',) * 2  # gather workspace plus owned returned data
    if 'residual' in operations:
        arrays += ('E',) * 2
    if 'hessian_vector_product' in operations:
        arrays += ('PP',)
    return arrays


@dataclass(frozen=True)
class GSTOptimizationContext:
    """The documented construction context passed to an external problem factory.

    ``objective`` is the configured MDC objective, including regularization,
    penalties, outcome layout and resource allocation. It owns mutable buffers
    and its model. A factory may reuse it or build a different numerical
    representation; the fit orchestrator remains responsible for committing the
    returned parameter vector to this model and evaluating final statistics.
    """

    objective: object
    profiler: object = None
    printer: object = None
    domain: object = None

    @property
    def model(self):
        return self.objective.model

    @property
    def dataset(self):
        return self.objective.dataset

    @property
    def circuits(self):
        return self.objective.circuits

    @property
    def resource_alloc(self):
        return self.objective.resource_alloc


def default_problem_factory(context, operations):
    """Create an owned global-array problem from an existing MDC objective.

    The supplied ``operations`` must be declared before layout construction.
    Product requests use the existing dense Jacobian implementation and expose
    that capability explicitly. All callbacks are collective when the source
    objective is distributed. They must therefore be called on all ranks.
    """
    objective = context.objective
    operations = frozenset(operations) | {'value'}
    objective_methods(operations)  # validate before any evaluation
    layout = objective.layout
    extra = objective.ex

    def gather(kind, array):
        return np.array(layout.allgather_local_array(kind, array, extra_elements=extra), copy=True)

    def gradient(x):
        # MDCObjectiveFunction.jacobian only sums *local* terms. Gather both
        # element and parameter axes, respecting shared replicas/penalty rows.
        return gather('ep', objective.dterms(x)).sum(axis=0)

    residual_derivatives = operations.intersection(
        ('residual_jacobian', 'residual_jvp', 'residual_vjp'))
    callbacks = {}
    if 'gradient' in operations:
        callbacks['gradient'] = gradient
    if 'residual' in operations:
        callbacks['residual'] = lambda x: gather('e', objective.lsvec(x))
    if residual_derivatives:
        callbacks['residual_jacobian'] = lambda x: gather('ep', objective.dlsvec(x))
    if 'hessian_vector_product' in operations:
        def hessian_vector_product(x, v):
            hessian = objective.hessian(x)
            comm = objective.resource_alloc.comm
            if comm is not None:
                hessian = comm.bcast(hessian, root=0)
            return hessian @ v
        callbacks['hessian_vector_product'] = hessian_vector_product
    bounds = objective.model.parameter_bounds
    return OptimizationProblem(
        objective.model.to_vector(), objective.fn,
        bounds=None if bounds is None else (bounds[:, 0], bounds[:, 1]),
        **callbacks)


def run_problem_optimizer(optimizer, objective, profiler, printer, *, domain=None):
    """Own the GST solve lifecycle, including failures and final model state."""
    from .outcome import OptimizerOutcome
    from .optimizer import OptimizerResult
    from .collective import solve_on_root, collective_phase

    initial = np.array(objective.model.to_vector(), copy=True)
    context = GSTOptimizationContext(objective, profiler, printer, domain)
    factory = optimizer.problem_factory or default_problem_factory
    comm = objective.resource_alloc.comm

    def prepare_problem():
        if domain is not None and not optimizer.supports_evaluation_domain:
            raise NotImplementedError('This optimizer does not declare evaluation-domain support')
        problem = factory(context, optimizer.required_operations)
        if not isinstance(problem, OptimizationProblem):
            raise TypeError("A problem factory must return an OptimizationProblem")
        available = set(problem.capabilities)
        if 'residual_jacobian' in available:
            available.update(('residual_jvp', 'residual_vjp'))
        missing = set(optimizer.required_operations) - available
        if missing:
            raise ValueError("The problem factory does not supply requested operations: %s" % sorted(missing))
        if domain is not None:
            from .domain import guard_problem
            problem = guard_problem(problem, domain)
        return problem

    try:
        problem = collective_phase(comm, prepare_problem, 'Optimization problem construction')
        outcome = (solve_on_root(optimizer.solve, problem, comm)
                   if comm is not None and comm.size > 1 else optimizer.solve(problem))

        def validate_outcome():
            if not isinstance(outcome, OptimizerOutcome):
                raise TypeError("solve(problem) must return an OptimizerOutcome")
            normalized = OptimizerOutcome(outcome.x, outcome.status, outcome.message,
                                          outcome.diagnostics, outcome.restart_state)
            if normalized.x.shape != initial.shape:
                raise ValueError("The returned point has the wrong parameter dimension")
            if problem.bounds is not None:
                lower, upper = problem.bounds
                if np.any(normalized.x < lower) or np.any(normalized.x > upper):
                    raise ValueError("The returned point violates the parameter bounds")
            return normalized
        outcome = collective_phase(comm, validate_outcome, 'Optimizer result validation')
        if domain is not None:
            collective_phase(comm, lambda: domain.require(outcome.x), 'Final evaluation-domain validation')

        # Commit exactly, even when the last oracle call was a rejected trial.
        # Statistics need scalar terms only; never demand a final Jacobian.
        collective_phase(comm, lambda: objective.model.from_vector(outcome.x), 'Final model commit')
        def local_statistics():
            terms = np.array(objective.terms(outcome.x), copy=True)
            if np.iscomplexobj(terms) or not np.all(np.isfinite(terms)):
                raise ValueError('Final objective terms must be finite and real')
            local_extra = getattr(objective, 'local_ex', objective.ex)
            unpenalized = terms[:-local_extra] if local_extra else terms
            with np.errstate(over='ignore', invalid='ignore'):
                statistics = float(terms.sum()), float(unpenalized.sum())
            if not np.all(np.isfinite(statistics)):
                raise ValueError('Final local objective statistics overflowed')
            return statistics
        local_value, local_unpenalized = collective_phase(
            comm, local_statistics, 'Final objective evaluation')
        value = float(objective.layout.allsum_local_quantity('e', local_value, use_shared_mem=False))
        unpenalized_value = float(objective.layout.allsum_local_quantity(
            'e', local_unpenalized, use_shared_mem=False))
        if not np.all(np.isfinite((value, unpenalized_value))):
            raise ValueError("The returned point has nonfinite objective statistics")
        result = OptimizerResult(
            objective, outcome.x.copy(), value, None, unpenalized_value,
            objective.chi2k_distributed_qty(value),
            {'msg': outcome.message, **outcome.diagnostics})
        result.outcome = outcome
        return result
    except BaseException:
        # Restore the caller's entry model on failure, rather than leaving a
        # rejected trial or half-completed numerical derivative installed.
        objective.model.from_vector(initial)
        raise

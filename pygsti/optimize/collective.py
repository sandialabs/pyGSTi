"""Root-driven solver control with collective numerical evaluations.

Only rank zero enters an external solver. Other ranks serve the same problem
oracles, so stochastic or opaque solver branching cannot reorder collectives.
This coordinates exceptions at oracle boundaries; as with the simulator itself,
an oracle must not abandon an internal collective on just one rank.
"""

from .problem import OptimizationProblem
from .domain import EvaluationDomain, EvaluationDomainError, StaleEvaluationDomainError


class CollectiveEvaluationError(RuntimeError):
    """At least one rank failed while evaluating a collective problem oracle."""


def collective_phase(comm, function, description):
    """Synchronize rank-local construction/validation failures at a phase boundary.

    The function may return different local Python objects. Its own internal
    collectives must still be entered consistently on all participating ranks.
    """
    if comm is None or comm.size <= 1:
        return function()
    value = None
    error = None
    try:
        value = function()
    except Exception as exc:
        error = '%s: %s' % (type(exc).__name__, exc)
    errors = comm.allgather(error)
    if any(error is not None for error in errors):
        raise CollectiveEvaluationError('%s failed: %s' % (description, errors))
    return value


def solve_on_root(solver, problem, comm):
    """Run ``solver(problem)`` on root and return its outcome on every rank.

    Native/composite ``structure`` is deliberately rejected here: such backends
    need their own explicit distributed execution model. This service operates
    on the standard real NumPy scalar/residual oracles only.
    """
    from mpi4py import MPI

    signatures = comm.allgather((tuple(sorted(problem.capabilities)), problem.num_parameters,
                                 problem.structure is not None))
    if any(signature[2] for signature in signatures):
        raise NotImplementedError("MPI dispatch of backend-native problem structure is not supported")
    if any(signature[:2] != signatures[0][:2] for signature in signatures):
        raise CollectiveEvaluationError('MPI problem dimensions and capabilities must agree on all ranks')

    def evaluate(name, args):
        result = None
        error = None
        try:
            result = (problem.domain.check(*args) if name == 'check_domain'
                      else getattr(problem, name)(*args))
        except EvaluationDomainError as exc:
            error = ('domain', exc.point, exc.description, exc.generation)
        except StaleEvaluationDomainError as exc:
            error = ('stale', str(exc))
        except Exception as exc:
            error = ('error', "%s: %s" % (type(exc).__name__, exc))
        try:
            MPI.pickle.dumps(error)
        except Exception as exc:
            # Serialization must fail locally before the collective starts.
            # In particular, external domain errors can carry opaque metadata.
            error = ('error', 'Unserializable evaluation error metadata: %s: %s'
                     % (type(exc).__name__, exc))
        errors = comm.allgather(error)
        failures = [e for e in errors if e is not None]
        if failures:
            if all(e[0] == 'domain' for e in failures):
                raise EvaluationDomainError(*failures[0][1:])
            if all(e[0] == 'stale' for e in failures):
                raise StaleEvaluationDomainError(failures[0][1])
            raise CollectiveEvaluationError("Collective %s failed: %s" % (name, errors))
        if name == 'check_domain':
            # Domain predicates may be rank-local; validity requires agreement
            # from every participant, rather than only the solver rank.
            return all(comm.allgather(bool(result)))
        return result

    if comm.rank != 0:
        while True:
            command, payload = comm.bcast(None, root=0)
            if command == 'finish':
                return payload
            if command == 'abort':
                raise RuntimeError("Root optimizer failed: %s" % payload)
            name, args = payload
            try:
                evaluate(name, args)
            except (CollectiveEvaluationError, EvaluationDomainError, StaleEvaluationDomainError):
                # Root receives the same error and may choose to recover, or
                # sends an abort command when the solver exception escapes.
                pass

    active = True
    def remote(name):
        def call(*args):
            if not active:
                raise StaleEvaluationDomainError('The collective optimization service has ended')
            comm.bcast(('evaluate', (name, args)), root=0)
            return evaluate(name, args)
        return call

    callbacks = {name: remote(name) for name in problem.capabilities if name != 'value'}
    proxy = OptimizationProblem(problem.initial_point, remote('value'),
                                bounds=problem.bounds, **callbacks)
    if hasattr(problem, 'domain'):
        proxy.domain = EvaluationDomain(remote('check_domain'), problem.domain.description,
                                        problem.domain.generation)
    try:
        outcome = solver(proxy)
        from .outcome import OptimizerOutcome
        if not isinstance(outcome, OptimizerOutcome):
            raise TypeError('solve(problem) must return an OptimizerOutcome')
        # Normalize owned arrays before serializing the outcome to workers.
        outcome = OptimizerOutcome(outcome.x, outcome.status, outcome.message,
                                   outcome.diagnostics, outcome.restart_state)
        MPI.pickle.dumps(outcome)
    except BaseException as exc:
        comm.bcast(('abort', "%s: %s" % (type(exc).__name__, exc)), root=0)
        raise
    finally:
        active = False
        if hasattr(proxy, 'domain'):
            proxy.domain.invalidate()
    comm.bcast(('finish', outcome), root=0)
    return outcome

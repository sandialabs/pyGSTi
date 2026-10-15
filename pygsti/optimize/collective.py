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


def solve_on_root(solver, problem, comm):
    """Run ``solver(problem)`` on root and return its outcome on every rank.

    Native/composite ``structure`` is deliberately rejected here: such backends
    need their own explicit distributed execution model. This service operates
    on the standard real NumPy scalar/residual oracles only.
    """
    if problem.structure is not None:
        raise NotImplementedError("MPI dispatch of backend-native problem structure is not supported")

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
        errors = comm.allgather(error)
        failures = [e for e in errors if e is not None]
        if failures:
            if all(e[0] == 'domain' for e in failures):
                raise EvaluationDomainError(*failures[0][1:])
            if all(e[0] == 'stale' for e in failures):
                raise StaleEvaluationDomainError(failures[0][1])
            raise CollectiveEvaluationError("Collective %s failed: %s" % (name, errors))
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
    except BaseException as exc:
        comm.bcast(('abort', "%s: %s" % (type(exc).__name__, exc)), root=0)
        raise
    finally:
        active = False
        if hasattr(proxy, 'domain'):
            proxy.domain.invalidate()
    comm.bcast(('finish', outcome), root=0)
    return outcome

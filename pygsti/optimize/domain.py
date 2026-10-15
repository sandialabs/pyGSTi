"""Explicit validity domains for approximate numerical objective functions.

A domain describes where one fixed approximation (for example a selected term
path set) is valid. It is independent of parameter bounds and solver acceptance.
"""

from functools import wraps
import numpy as np

from .problem import OptimizationProblem, ResidualLinearization


class EvaluationDomainError(RuntimeError):
    """A requested point is outside this approximation's validity domain.

    This exception is recoverable by a solver that explicitly handles validity
    domains. It does not identify an accepted incumbent or a replacement point.
    Programming errors and unrelated oracle exceptions must retain their types.
    """

    def __init__(self, point, description='Evaluation outside approximation domain', generation=None):
        self.point = np.array(point, dtype=float, copy=True)
        self.point.setflags(write=False)
        self.description = str(description)
        self.generation = generation
        super().__init__(self.description)

    def __reduce__(self):
        return type(self), (self.point, self.description, self.generation)


class StaleEvaluationDomainError(RuntimeError):
    """An evaluation refers to an approximation that has been invalidated."""


class EvaluationDomain:
    """Validity predicate for a fixed approximation generation.

    ``check(point)`` must return a bool, possibly after collective simulation.
    It may mutate its own simulation state. Its exceptions are never translated
    into domain failures. ``generation`` is an opaque identifier, not a mutable
    simulation version; supply an MPI-serializable identifier for MPI dispatch.
    Construct a fresh domain after changing the approximation and invalidate the
    previous domain, so cached derivative closures cannot use obsolete paths.
    """

    def __init__(self, check, description='Evaluation outside approximation domain', generation=None):
        if not callable(check):
            raise TypeError('The domain check must be callable')
        self._check = check
        self.description = str(description)
        self.generation = generation
        self._active = True

    def invalidate(self):
        """Permanently invalidate this approximation generation."""
        self._active = False

    def _require_active(self):
        if not self._active:
            raise StaleEvaluationDomainError('The approximation domain has been invalidated')

    def check(self, point):
        """Return whether the point is valid, without recording solver state."""
        self._require_active()
        if np.iscomplexobj(point):
            raise ValueError('A domain check requires real coordinates')
        point = np.array(point, dtype=float, copy=True)
        if point.ndim != 1 or not np.all(np.isfinite(point)):
            raise ValueError('A domain check requires a finite real parameter vector')
        valid = self._check(point)
        if not isinstance(valid, (bool, np.bool_)):
            raise TypeError('The domain predicate must return a bool')
        return bool(valid)

    def require(self, point):
        """Raise EvaluationDomainError only when the predicate returns False."""
        if not self.check(point):
            raise EvaluationDomainError(point, self.description, self.generation)

    def guard(self, oracle):
        """Guard a point-first oracle before evaluating its numerical result."""
        @wraps(oracle)
        def guarded(point, *args):
            self.require(point)
            return oracle(point, *args)
        return guarded


class _DomainLinearization(ResidualLinearization):
    def jvp(self, v):
        self._problem.domain._require_active()
        return super().jvp(v)

    def vjp(self, w):
        self._problem.domain._require_active()
        return super().vjp(w)

    def to_dense(self):
        self._problem.domain._require_active()
        return super().to_dense()


class _DomainGuardedProblem(OptimizationProblem):
    def linearize(self, point):
        return _DomainLinearization(self, point)


def guard_problem(problem, domain):
    """Guard standard NumPy problem oracles for one approximation generation.

    No derivative, accepted point, or fallback approximation is invented. A
    backend-native ``structure`` can expose callables bypassing these oracles,
    so it requires an explicit native domain implementation and is rejected.
    Captured linearizations reject derivative actions after invalidation, even
    when their dense Jacobian is cached. Arrays already copied out cannot be
    revoked; solvers must discard cached models when restarting a generation.
    """
    if not isinstance(problem, OptimizationProblem):
        raise TypeError('Expected an OptimizationProblem')
    if not isinstance(domain, EvaluationDomain):
        raise TypeError('Expected an EvaluationDomain')
    if problem.structure is not None:
        raise NotImplementedError('Backend-native structure requires explicit domain handling')
    callbacks = {name: domain.guard(getattr(problem, name))
                 for name in problem.capabilities if name != 'value'}
    guarded = _DomainGuardedProblem(problem.initial_point, domain.guard(problem.value),
                                   bounds=problem.bounds, **callbacks)
    # Public diagnostic metadata; this does not expand mathematical capabilities.
    guarded.domain = domain
    return guarded

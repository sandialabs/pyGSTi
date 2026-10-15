"""Small numerical problem contract for independently implemented optimizers.

The default coordinates and numerical oracles use real NumPy arrays. Backend-native
or composite representations may be supplied through ``structure``; the core does
not translate array backends or approximate missing derivatives.
"""

import numpy as _np


def _owned_vector(value, size=None):
    if _np.iscomplexobj(value):
        raise ValueError("Optimization coordinates and derivatives must be real")
    array = _np.array(value, dtype=float, copy=True)
    if array.ndim != 1 or (size is not None and array.size != size):
        raise ValueError("Expected a one-dimensional vector of the declared size")
    return array


class OptimizationProblem:
    """A scalar objective with optional, explicitly supplied derivative structure.

    ``value`` is required and includes all configured terms. A supplied residual
    represents ``value(x) = residual(x).T @ residual(x)`` (no factor of one half).
    Gradient and exact Hessian products are independent capabilities: they are
    never inferred from residuals. Residual products may use an explicitly supplied
    dense residual Jacobian, but never finite differences. Callers can inspect
    ``capabilities`` to distinguish dense Jacobians from supplied product oracles.

    Oracles may mutate their own reusable buffers. Returned arrays are copied at
    this boundary. One problem's oracles need not support concurrent calls.
    Bounds are metadata, not clipping instructions; solvers choose how to enforce
    them. Invalid trial points may produce nonfinite values or an oracle exception.
    """

    def __init__(self, initial_point, value, gradient=None, residual=None,
                 residual_jacobian=None, residual_jvp=None, residual_vjp=None,
                 hessian_vector_product=None, bounds=None, structure=None):
        self._initial_point = _owned_vector(initial_point)
        if not _np.all(_np.isfinite(self._initial_point)):
            raise ValueError("The initial point must be finite")
        if not callable(value):
            raise TypeError("value must be callable")
        self._oracles = dict(value=value, gradient=gradient, residual=residual,
                             residual_jacobian=residual_jacobian,
                             residual_jvp=residual_jvp, residual_vjp=residual_vjp,
                             hessian_vector_product=hessian_vector_product)
        for name, oracle in self._oracles.items():
            if oracle is not None and not callable(oracle):
                raise TypeError("%s must be callable or None" % name)
        self.capabilities = frozenset(name for name, fn in self._oracles.items() if fn is not None)
        self.structure = structure
        self._bounds = None
        if bounds is not None:
            lower, upper = (_owned_vector(b, self.num_parameters) for b in bounds)
            if _np.any(_np.isnan(lower)) or _np.any(_np.isnan(upper)) or _np.any(lower > upper):
                raise ValueError("Bounds must be ordered and contain no NaNs")
            self._bounds = lower, upper

    @property
    def num_parameters(self):
        """Dimension of the real parameter space."""
        return self._initial_point.size

    @property
    def initial_point(self):
        """An owned copy of the starting point."""
        return self._initial_point.copy()

    @property
    def bounds(self):
        """Owned lower and upper bound vectors, or None."""
        return None if self._bounds is None else tuple(b.copy() for b in self._bounds)

    def _point(self, x):
        return _owned_vector(x, self.num_parameters)

    def _oracle(self, name):
        fn = self._oracles[name]
        if fn is None:
            raise NotImplementedError("Problem does not supply %s" % name)
        return fn

    def value(self, x):
        """Evaluate the full scalar objective."""
        result = _np.asarray(self._oracle('value')(self._point(x)))
        if result.ndim != 0 or _np.iscomplexobj(result):
            raise ValueError("The objective value must be a real scalar")
        return float(result)

    def gradient(self, x):
        """Evaluate the scalar objective gradient; no residual-based fallback."""
        return _owned_vector(self._oracle('gradient')(self._point(x)), self.num_parameters)

    def residual(self, x):
        """Evaluate the supplied sum-of-squares residual representation."""
        return _owned_vector(self._oracle('residual')(self._point(x)))

    def residual_jacobian(self, x):
        """Evaluate an explicitly supplied dense residual Jacobian."""
        raw = self._oracle('residual_jacobian')(self._point(x))
        if _np.iscomplexobj(raw):
            raise ValueError("Residual Jacobian must be real")
        result = _np.array(raw, dtype=float, copy=True)
        if result.ndim != 2 or result.shape[1] != self.num_parameters:
            raise ValueError("Residual Jacobian must have shape (num_residuals, num_parameters)")
        return result

    def residual_jvp(self, x, v):
        """Apply the residual derivative, using a supplied product or dense Jacobian."""
        x, v = self._point(x), _owned_vector(v, self.num_parameters)
        fn = self._oracles['residual_jvp']
        return _owned_vector(fn(x, v)) if fn is not None else self.residual_jacobian(x) @ v

    def residual_vjp(self, x, w):
        """Apply the Euclidean transpose of the residual derivative."""
        x, w = self._point(x), _owned_vector(w)
        fn = self._oracles['residual_vjp']
        if fn is not None:
            return _owned_vector(fn(x, w), self.num_parameters)
        return self.residual_jacobian(x).T @ w

    def hessian_vector_product(self, x, v):
        """Apply the exact scalar Hessian; never substitute Gauss–Newton."""
        return _owned_vector(self._oracle('hessian_vector_product')(
            self._point(x), _owned_vector(v, self.num_parameters)), self.num_parameters)

    def gauss_newton_vector_product(self, x, v):
        """Apply 2 J.T J for the supplied residual representation."""
        return self.residual_vjp(x, 2 * self.residual_jvp(x, v))

    def linearize(self, x):
        """Capture residual information at an owned fixed point."""
        return ResidualLinearization(self, x)


class ResidualLinearization:
    """A point-bound residual and derivative with stable array ownership.

    Dense Jacobians are copied once when available. Product-only oracles are
    invoked at the captured point on each call, so they may recompute simulation.
    No global Jacobian is created when only product oracles are supplied.
    """

    def __init__(self, problem, point):
        self._problem = problem
        self._point = problem._point(point)
        self._residual = problem.residual(self._point)
        self._jacobian = (problem.residual_jacobian(self._point)
                          if 'residual_jacobian' in problem.capabilities else None)
        if self._jacobian is not None and self._jacobian.shape[0] != self._residual.size:
            raise ValueError("Residual and Jacobian row counts differ")
        self.storage = 'dense' if self._jacobian is not None else 'operator'

    @property
    def point(self):
        return self._point.copy()

    @property
    def residual(self):
        return self._residual.copy()

    def jvp(self, v):
        """Apply the derivative at the captured point."""
        v = _owned_vector(v, self._point.size)
        result = (self._jacobian @ v if self._jacobian is not None
                  else self._problem.residual_jvp(self._point, v))
        return _owned_vector(result, self._residual.size)

    def vjp(self, w):
        """Apply the derivative transpose at the captured point."""
        w = _owned_vector(w, self._residual.size)
        return (self._jacobian.T @ w if self._jacobian is not None
                else self._problem.residual_vjp(self._point, w))

    def to_dense(self):
        """Return the supplied dense Jacobian; never materialize product columns."""
        if self._jacobian is None:
            raise NotImplementedError("No dense residual Jacobian was supplied")
        return self._jacobian.copy()

    def gauss_newton_vector_product(self, v):
        """Apply 2 J.T J at the captured point."""
        return self.vjp(2 * self.jvp(v))

"""
Shared fixtures for pygsti.optimize characterization tests.

These fixtures are intentionally self-contained and documented so they can also
serve as illustrative examples for users who want to understand how to feed
custom problems into pygsti's optimization routines.

Design choices
--------------
* All problems have closed-form solutions (so tests can assert against exact
  analytic ground truth with tight tolerances, not just "looks about right").
* No randomness — every value is deterministic so tests reproduce identically
  on any platform.
* Dependencies are limited to numpy and scipy so the fixtures work even in
  environments that lack mpi4py, cvxpy or deap.
"""

import numpy as np
import scipy.linalg

from pygsti.optimize.arraysinterface import UndistributedArraysInterface


# ---------------------------------------------------------------------------
# Problem 1: Linear Least-Squares
# ---------------------------------------------------------------------------
# Minimize ||A x - b||^2.  The analytic solution is the Moore-Penrose
# pseudo-inverse: x* = (A^T A)^{-1} A^T b.
#
# We deliberately use a tall (overdetermined) matrix so the LM solver treats
# the residual vector f = A x - b just like it would a physics objective
# function (many "measurements", few parameters).

LINEAR_A = np.array([
    [2.0,  1.0],
    [1.0,  3.0],
    [0.0,  1.0],
    [1.0, -1.0],
], dtype='d')

LINEAR_B = np.array([5.0, 10.0, 3.0, 0.0], dtype='d')

# Analytic solution: x* = (A^T A)^{-1} A^T b
LINEAR_X_STAR = np.linalg.lstsq(LINEAR_A, LINEAR_B, rcond=None)[0]


def linear_obj_fn(x):
    """Residual vector f(x) = A x - b  (shape: (4,))."""
    return LINEAR_A @ x - LINEAR_B


def linear_jac_fn(x):
    """Jacobian df/dx = A  (shape: (4, 2), constant)."""
    return LINEAR_A.copy()


def make_linear_ari():
    """Return an UndistributedArraysInterface sized for the linear problem."""
    n_elements, n_params = LINEAR_A.shape
    return UndistributedArraysInterface(n_elements, n_params)


# ---------------------------------------------------------------------------
# Problem 2: Bounded Scalar Quadratic
# ---------------------------------------------------------------------------
# Minimise f(x)^2 = x[0]^4  subject to x[0] >= 1.
# True minimum inside the feasible region: x* = 1.
#
# A scalar, highly nonlinear example that stresses x-limits handling.

BOUNDED_X_LIMITS = np.array([[1.0, 10.0]], dtype='d')
BOUNDED_X_STAR = np.array([1.0], dtype='d')


def bounded_obj_fn(x):
    """Residual f(x) = x[0]^2  (shape: (1,)).  Minimiser at x=0, clamped to 1."""
    return np.array([x[0] ** 2], dtype='d')


def bounded_jac_fn(x):
    """Jacobian of bounded_obj_fn  (shape: (1, 1))."""
    return np.array([[2.0 * x[0]]], dtype='d')


def make_bounded_ari():
    """Return an UndistributedArraysInterface sized for the bounded problem."""
    return UndistributedArraysInterface(1, 1)


# ---------------------------------------------------------------------------
# Problem 3: Mildly Nonlinear Least-Squares
# ---------------------------------------------------------------------------
# Fit a 2-parameter exponential model y_i = p0 * exp(p1 * t_i) to
# noiseless data generated from p0=2, p1=-0.5.
#
# True parameters: NONLINEAR_P_STAR = [2.0, -0.5].
# This exercises the Jacobian more seriously than the linear problem while
# still having an exact ground truth.

NONLINEAR_T = np.linspace(0.0, 2.0, 6)          # 6 time points → overdetermined
NONLINEAR_P_STAR = np.array([2.0, -0.5], dtype='d')
NONLINEAR_Y = NONLINEAR_P_STAR[0] * np.exp(NONLINEAR_P_STAR[1] * NONLINEAR_T)


def nonlinear_obj_fn(p):
    """
    Residual vector f(p) = p[0]*exp(p[1]*t) - y_data  (shape: (6,)).

    Example usage for users
    -----------------------
    This is the kind of function you pass to ``simplish_leastsq`` or
    ``custom_leastsq`` as ``obj_fn``.  It should return a *vector* of
    residuals (not a scalar sum-of-squares).  The optimizer minimises
    ``sum(f(p)**2)``.
    """
    return p[0] * np.exp(p[1] * NONLINEAR_T) - NONLINEAR_Y


def nonlinear_jac_fn(p):
    """
    Jacobian df/dp  (shape: (6, 2)).

    Row i, column 0: d/dp0 [p0*exp(p1*t_i)] = exp(p1*t_i)
    Row i, column 1: d/dp1 [p0*exp(p1*t_i)] = p0*t_i*exp(p1*t_i)

    Example usage for users
    -----------------------
    This is the function you pass as ``jac_fn``.  It must return the matrix
    of partial derivatives df_i/dp_j (shape: n_residuals × n_params).
    """
    exp_terms = np.exp(p[1] * NONLINEAR_T)
    col0 = exp_terms                         # d/dp0
    col1 = p[0] * NONLINEAR_T * exp_terms    # d/dp1
    return np.column_stack([col0, col1])


def make_nonlinear_ari():
    """Return an UndistributedArraysInterface sized for the nonlinear problem."""
    n_elements = len(NONLINEAR_T)
    n_params = 2
    return UndistributedArraysInterface(n_elements, n_params)


# ---------------------------------------------------------------------------
# Problem 4: Small SPD linear system for custom_solve / customsolve tests
# ---------------------------------------------------------------------------
# Ax = b where A is symmetric positive definite and the exact solution is known.

SOLVE_A = np.array([
    [4.0, 1.0, 0.5],
    [1.0, 3.0, 0.5],
    [0.5, 0.5, 2.0],
], dtype='d')

SOLVE_B = np.array([1.0, 2.0, 3.0], dtype='d')

# Analytic solution
SOLVE_X_STAR = scipy.linalg.solve(SOLVE_A, SOLVE_B, assume_a='pos')


# ---------------------------------------------------------------------------
# Synthetic Newton-solve objective for wildcardopt tests
# ---------------------------------------------------------------------------
# Minimise f(x) = 0.5 * ||x - x_star||^2  with analytic gradient and Hessian.
# True minimiser: NEWTON_X_STAR.
# This is deliberately trivial so NewtonSolve converges in one iteration.

NEWTON_X_STAR = np.array([1.5, 2.5], dtype='d')


def newton_fn(x):
    """Objective:  0.5 * ||x - x*||^2.  Minimiser at NEWTON_X_STAR."""
    diff = x - NEWTON_X_STAR
    return 0.5 * float(np.dot(diff, diff))


def newton_fn_with_derivs(x):
    """Returns (f, grad, hess) for the quadratic objective."""
    diff = x - NEWTON_X_STAR
    f = 0.5 * float(np.dot(diff, diff))
    grad = diff.copy()
    hess = np.eye(len(x), dtype='d')
    return f, grad, hess

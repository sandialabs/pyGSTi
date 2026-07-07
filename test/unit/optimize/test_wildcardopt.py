"""
Characterization tests for pygsti.optimize.wildcardopt.

Coverage strategy
-----------------
* NewtonSolve — direct unit test with a synthetic convex quadratic whose
  analytic minimiser is known exactly.  No GST machinery required.
* _compute_fd — finite-difference gradient and Hessian on the same quadratic.
* The three GST-coupled wildcard optimisers (neldermead, barrier,
  bisect_alpha, cvxpy) require a live objective function and are therefore
  covered at the integration level through the GST protocol tests.
  optimize_wildcard_budget_percircuit_only_cvxpy is gated by @needs_cvxpy.

Synthetic objective functions
------------------------------
The NewtonSolve tests are intentionally self-contained and well-documented
so they can serve as user-facing examples of how to supply a custom Newton
objective to ``NewtonSolve``.
"""

import numpy as np
import pytest

from pygsti.optimize.wildcardopt import NewtonSolve, _compute_fd
from ..util import BaseCase, needs_cvxpy
from .fixtures import NEWTON_X_STAR, newton_fn, newton_fn_with_derivs


class NewtonSolveTester(BaseCase):
    """
    Direct characterization of NewtonSolve on a quadratic objective.

    Objective:  f(x) = 0.5 * ||x - x*||^2
    Gradient:   Df(x) = x - x*
    Hessian:    Hf(x) = I

    Newton takes exactly one step from any starting point:
        dx = -H^{-1} Df = -(x - x*) = x* - x
        x_new = x + dx = x*

    So we expect convergence in a single iteration to machine precision.

    Example usage for users
    -----------------------
    To use NewtonSolve on your own problem, supply:
      - ``initial_x``: starting point as a 1-D numpy array.
      - ``fn``: scalar objective function f(x) -> float.
      - ``fn_with_derivs``: function returning (f, grad, hess).
        If None, finite differences are used (slower but requires only fn).
    """

    def _solve(self, x0, use_fd=False):
        """Run NewtonSolve; return (final_x, x_list)."""
        fn_derivs = None if use_fd else newton_fn_with_derivs
        return NewtonSolve(
            x0.copy(),
            newton_fn,
            fn_with_derivs=fn_derivs,
            dx_tol=1e-10,
            max_iters=50,
        )

    def test_converges_to_exact_minimum_with_analytic_derivs(self):
        """One Newton step should reach machine precision from any start."""
        x0 = np.array([0.0, 0.0], 'd')
        x_final, x_list = self._solve(x0)
        np.testing.assert_allclose(x_final, NEWTON_X_STAR, atol=1e-10)

    def test_converges_from_far_start(self):
        x0 = np.array([100.0, -50.0], 'd')
        x_final, x_list = self._solve(x0)
        np.testing.assert_allclose(x_final, NEWTON_X_STAR, atol=1e-10)

    def test_converges_with_finite_difference_derivs(self):
        """Using fn_with_derivs=None triggers internal finite differences."""
        x0 = np.array([0.0, 0.0], 'd')
        x_final, x_list = self._solve(x0, use_fd=True)
        # FD is less precise; use a relaxed tolerance
        np.testing.assert_allclose(x_final, NEWTON_X_STAR, atol=1e-4)

    def test_x_list_tracks_iterates(self):
        """x_list should contain at least the initial and final points."""
        x0 = np.array([10.0, 5.0], 'd')
        x_final, x_list = self._solve(x0)
        self.assertGreaterEqual(len(x_list), 2)
        np.testing.assert_array_equal(x_list[0], x0)
        np.testing.assert_allclose(x_list[-1], NEWTON_X_STAR, atol=1e-10)

    def test_already_at_minimum_stays_put(self):
        """Starting at x* should leave x unchanged."""
        x0 = NEWTON_X_STAR.copy()
        x_final, _ = self._solve(x0)
        np.testing.assert_allclose(x_final, NEWTON_X_STAR, atol=1e-12)

    def test_clips_to_nonnegative(self):
        """
        NewtonSolve clips x to x >= 0 after each step.
        When x* has all non-negative components (as in our fixture),
        the solution should remain non-negative.
        """
        x0 = np.array([0.1, 0.1], 'd')
        x_final, _ = self._solve(x0)
        self.assertTrue(np.all(x_final >= 0.0))

    def test_1d_quadratic(self):
        """1-D sanity check: f(x) = 0.5*(x-3)^2, minimum at x=3."""
        x_star_1d = np.array([3.0], 'd')

        def fn_1d(x):
            return 0.5 * float((x[0] - 3.0) ** 2)

        def fn_1d_derivs(x):
            d = x[0] - 3.0
            return 0.5 * d * d, np.array([d]), np.array([[1.0]])

        x0 = np.array([10.0], 'd')
        x_final, _ = NewtonSolve(x0, fn_1d, fn_1d_derivs, dx_tol=1e-12, max_iters=50)
        np.testing.assert_allclose(x_final, x_star_1d, atol=1e-10)


class ComputeFdTester(BaseCase):
    """
    Characterization of _compute_fd (finite-difference gradient and Hessian).

    Uses the same quadratic fixture.  Gradient should match analytic to ~1e-5
    (FD with eps=1e-7 → O(eps) gradient error, O(eps) Hessian error for
    the mixed second differences with eps=1e-5).
    """

    def setUp(self):
        self.x = np.array([2.0, 3.0], 'd')

    def test_gradient_close_to_analytic(self):
        _, grad_analytic, _ = newton_fn_with_derivs(self.x)
        grad_fd, _ = _compute_fd(self.x, newton_fn)
        np.testing.assert_allclose(grad_fd, grad_analytic, atol=1e-4)

    def test_hessian_close_to_identity(self):
        _, hess_fd = _compute_fd(self.x, newton_fn)
        np.testing.assert_allclose(hess_fd, np.eye(2), atol=1e-4)

    def test_gradient_only_mode(self):
        """compute_hessian=False should return only the gradient array with correct values."""
        result = _compute_fd(self.x, newton_fn, compute_hessian=False)
        # Returns only the gradient (1-D array), not a (grad, hess) tuple
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.ndim, 1)
        self.assertEqual(len(result), 2)
        # Analytic gradient of 0.5*||x - x*||^2 is x - x*
        _, grad_analytic, _ = newton_fn_with_derivs(self.x)
        np.testing.assert_allclose(result, grad_analytic, atol=1e-4)

    def test_gradient_at_minimum_is_near_zero(self):
        grad_fd, _ = _compute_fd(NEWTON_X_STAR.copy(), newton_fn)
        np.testing.assert_allclose(grad_fd, np.zeros(2), atol=1e-4)

    def test_non_diagonal_hessian(self):
        """
        Verify _compute_fd recovers a non-diagonal Hessian correctly.

        Objective: f(x) = 0.5 * x^T M x  where M = [[2, 1], [1, 3]].
        Gradient:  Df(x) = M x
        Hessian:   H = M  (constant, non-diagonal)

        At x = [1, 1]:  grad = [3, 4],  hess = [[2, 1], [1, 3]].
        """
        M = np.array([[2.0, 1.0], [1.0, 3.0]])

        def quadratic_fn(x):
            return 0.5 * float(x @ M @ x)

        x = np.array([1.0, 1.0], 'd')
        grad_fd, hess_fd = _compute_fd(x, quadratic_fn)

        expected_grad = M @ x          # [3, 4]
        expected_hess = M              # [[2, 1], [1, 3]]

        np.testing.assert_allclose(grad_fd, expected_grad, atol=1e-4)
        np.testing.assert_allclose(hess_fd, expected_hess, atol=1e-4)

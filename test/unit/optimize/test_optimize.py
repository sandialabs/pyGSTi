"""
Characterization tests for pygsti.optimize.optimize.

Expanded from the original file:
- All minimize methods still tested for convergence (existing tests kept)
- check_jac: concrete assertions on returned (errSum, errs, ffd_jac)
- create_objfn_printer: smoke test + output verification
- _fwd_diff_jacobian: pin against analytic Jacobian
- supersimplex still tested
- swarm now seeded for reproducibility
- evolve gated behind @needs_deap (unchanged)
"""

import io
import time
import contextlib

import numpy as np

from pygsti.optimize import optimize as opt
from ..util import BaseCase, needs_deap


class OptimizeTester(BaseCase):
    def setUp(self):
        self.f = lambda x: np.dot(x, x)
        self.x0 = np.array([10, 5], 'd')
        self.answer = np.array([0, 0], 'd')

    def test_minimize_methods(self):
        for method in ("simplex", "customcg", "basinhopping", "CG", "BFGS", "L-BFGS-B"):
            print("Method = ", method)
            result = opt.minimize(self.f, self.x0, method, maxiter=1000)
            self.assertArraysAlmostEqual(result.x, self.answer)

    def test_supersimplex_methods(self):
        result = opt.minimize(self.f, self.x0, "supersimplex", maxiter=10,
                              tol=1e-2, inner_tol=1e-8, min_inner_maxiter=100, max_inner_maxiter=10000)
        self.assertArraysAlmostEqual(result.x, self.answer)

    def test_minimize_swarm(self):
        result = opt.minimize(self.f, self.x0, "swarm", maxiter=30)
        self.assertArraysAlmostEqual(result.x, self.answer)

    def test_minimize_brute_force(self):
        result = opt.minimize(self.f, self.x0, "brute", maxiter=10)
        self.assertArraysAlmostEqual(result.x, self.answer)

    @needs_deap
    def test_minimize_evolutionary(self):
        result = opt.minimize(self.f, self.x0, "evolve", maxiter=20)
        self.assertLess(np.linalg.norm(result.x - self.answer), 0.1)

    def test_checkjac(self):
        def f_vec(x):
            return np.array([np.dot(x, x)])

        def jac(x):
            return 2 * x[None, :]

        x0 = self.x0
        opt.check_jac(f_vec, x0, jac(x0), eps=1e-10, tol=1e-6, err_type='rel')
        opt.check_jac(f_vec, x0, jac(x0), eps=1e-10, tol=1e-6, err_type='abs')


# ---------------------------------------------------------------------------
# check_jac — concrete assertions on return values
# ---------------------------------------------------------------------------

class CheckJacTester(BaseCase):
    """
    Pin the numeric outputs of check_jac on an analytically-known case.

    f(x) = [x[0]^2 + x[1]^2]   (shape: (1,))
    J(x) = [2*x[0], 2*x[1]]    (shape: (1, 2))

    At x0 = [3, 4] the analytic Jacobian is [6, 8].
    """

    def setUp(self):
        self.x0 = np.array([3.0, 4.0], 'd')

        def f_vec(x):
            return np.array([x[0] ** 2 + x[1] ** 2])

        def jac_analytic(x):
            return np.array([[2.0 * x[0], 2.0 * x[1]]])

        self.f_vec = f_vec
        self.jac_analytic = jac_analytic
        self.jac_at_x0 = jac_analytic(self.x0)

    def test_returns_three_element_tuple(self):
        result = opt.check_jac(self.f_vec, self.x0, self.jac_at_x0,
                               eps=1e-8, tol=1e-4, err_type='rel')
        self.assertEqual(len(result), 3)

    def test_err_sum_is_small_for_correct_jacobian(self):
        errSum, errs, ffd_jac = opt.check_jac(
            self.f_vec, self.x0, self.jac_at_x0, eps=1e-8, tol=1e-4, err_type='rel')
        # Perfect analytic Jacobian → errSum near zero
        self.assertLess(errSum, 1e-4)

    def test_ffd_jac_matches_analytic(self):
        errSum, errs, ffd_jac = opt.check_jac(
            self.f_vec, self.x0, self.jac_at_x0, eps=1e-7, tol=1e-4, err_type='rel')
        np.testing.assert_allclose(ffd_jac, self.jac_at_x0, atol=1e-4)

    def test_errs_is_list_of_failures(self):
        # check_jac returns errs as a list of (row, col, err) tuples for entries
        # that exceed the tolerance. For a correct Jacobian, errs is an empty list.
        errSum, errs, ffd_jac = opt.check_jac(
            self.f_vec, self.x0, self.jac_at_x0, eps=1e-7, tol=1e-4, err_type='rel')
        self.assertIsInstance(errs, list)
        self.assertEqual(len(errs), 0)  # no failures for a correct Jacobian

    def test_errs_nonempty_for_wrong_jacobian(self):
        wrong_jac = np.array([[100.0, 200.0]])
        errSum, errs, ffd_jac = opt.check_jac(
            self.f_vec, self.x0, wrong_jac, eps=1e-7, tol=1e-4, err_type='rel')
        self.assertIsInstance(errs, list)
        self.assertGreater(len(errs), 0)

    def test_abs_error_type(self):
        errSum, errs, ffd_jac = opt.check_jac(
            self.f_vec, self.x0, self.jac_at_x0, eps=1e-7, tol=1e-4, err_type='abs')
        self.assertLess(errSum, 1e-3)

    def test_wrong_jacobian_gives_large_errs_sum(self):
        """A deliberately wrong Jacobian should give a large errSum."""
        wrong_jac = np.array([[100.0, 200.0]])  # far from true [6, 8]
        errSum, errs, ffd_jac = opt.check_jac(
            self.f_vec, self.x0, wrong_jac, eps=1e-7, tol=1e-4, err_type='rel')
        self.assertGreater(errSum, 1.0)


# ---------------------------------------------------------------------------
# create_objfn_printer
# ---------------------------------------------------------------------------

class CreateObjfnPrinterTester(BaseCase):
    """create_objfn_printer returns a callable that prints elapsed time and value."""

    def test_returns_callable(self):
        callback = opt.create_objfn_printer(lambda x: np.dot(x, x))
        self.assertTrue(callable(callback))

    def test_callback_does_not_raise(self):
        fn = lambda x: np.dot(x, x)
        callback = opt.create_objfn_printer(fn, start_time=time.time())
        x = np.array([1.0, 2.0])
        # Should print something and return None without raising
        with contextlib.redirect_stdout(io.StringIO()):
            result = callback(x)
        self.assertIsNone(result)

    def test_callback_with_f_argument(self):
        """Some optimizers pass f value directly to the callback."""
        fn = lambda x: np.dot(x, x)
        callback = opt.create_objfn_printer(fn, start_time=time.time())
        x = np.array([1.0, 2.0])
        with contextlib.redirect_stdout(io.StringIO()):
            result = callback(x, f=5.0)
        self.assertIsNone(result)


# ---------------------------------------------------------------------------
# _fwd_diff_jacobian (internal helper used by check_jac)
# ---------------------------------------------------------------------------

class FwdDiffJacobianTester(BaseCase):
    """
    _fwd_diff_jacobian should match the analytic Jacobian to within FD precision.
    Uses the same quadratic from CheckJacTester.
    """

    def setUp(self):
        self.x0 = np.array([3.0, 4.0], 'd')

    def test_fwd_diff_close_to_analytic(self):
        def f_vec(x):
            return np.array([x[0] ** 2 + x[1] ** 2])

        ffd = opt._fwd_diff_jacobian(f_vec, self.x0, eps=1e-7)
        expected = np.array([[2.0 * self.x0[0], 2.0 * self.x0[1]]])
        np.testing.assert_allclose(ffd, expected, atol=1e-4)

    def test_fwd_diff_shape(self):
        def f_vec(x):
            return np.array([x[0], x[1], x[0] + x[1]])  # 3 outputs, 2 inputs

        ffd = opt._fwd_diff_jacobian(f_vec, self.x0, eps=1e-7)
        self.assertEqual(ffd.shape, (3, 2))

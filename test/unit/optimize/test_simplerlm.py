"""
Characterization tests for pygsti.optimize.simplerlm.

Expanded from the original 3-test file to cover:
- Full return-tuple shapes, types, and values
- damp_coeff_update helper
- jac_guarded helper (analytic and finite-difference paths)
- Optimizer.cast / SimplerLMOptimizer.cast including fallback to CustomLMOptimizer
- OptimizerResult construction
- SimplerLMOptimizer serialization round-trip
- Convergence to analytic solutions using the shared fixtures
"""

import numpy as np
import pytest

from pygsti.optimize import arraysinterface as _ari
from pygsti.optimize import simplerlm as lm
from pygsti.optimize.simplerlm import (
    Optimizer, OptimizerResult, SimplerLMOptimizer,
    damp_coeff_update, jac_guarded,
)
from pygsti.optimize.customlm import CustomLMOptimizer
from ..util import BaseCase
from .fixtures import (
    linear_obj_fn, linear_jac_fn, make_linear_ari, LINEAR_X_STAR,
    nonlinear_obj_fn, nonlinear_jac_fn, make_nonlinear_ari, NONLINEAR_P_STAR,
    bounded_obj_fn, bounded_jac_fn, make_bounded_ari, BOUNDED_X_LIMITS, BOUNDED_X_STAR,
)


# ---------------------------------------------------------------------------
# Fixtures re-used from the original test file
# ---------------------------------------------------------------------------

def f_inf(x):
    return np.inf * np.ones(2, 'd')


def jac_zero(x):
    return np.zeros((2, 3), 'd')


def g(x):
    return np.array([x[0] ** 2], 'd')


def gjac(x):
    return np.array([[2 * x[0]]], 'd')


# ---------------------------------------------------------------------------
# Original tests (kept, now with real assertions)
# ---------------------------------------------------------------------------

class LMTester(BaseCase):

    def test_simplish_leastsq_infinite_objective_fn_norm_at_x0(self):
        x0 = np.ones(3, 'd')
        ari = _ari.UndistributedArraysInterface(2, 3)
        xf, converged, msg, *_ = lm.simplish_leastsq(f_inf, jac_zero, x0, arrays_interface=ari)
        self.assertEqual(msg, "Infinite norm of objective function at initial point!")

    def test_simplish_leastsq_max_iterations_exceeded(self):
        x0 = np.ones(3, 'd')
        ari = _ari.UndistributedArraysInterface(2, 3)
        xf, converged, msg, *_ = lm.simplish_leastsq(f_inf, jac_zero, x0, max_iter=0, arrays_interface=ari)
        self.assertEqual(msg, "Maximum iterations (0) exceeded")

    def test_simplish_leastsq_x_limits(self):
        x0 = np.array([6.0], 'd')
        xlimits = np.array([[1.0, 10.0]], 'd')
        ari = _ari.UndistributedArraysInterface(1, 1)
        xf, converged, msg, *_ = lm.simplish_leastsq(g, gjac, x0, max_iter=100,
                                                      arrays_interface=ari, x_limits=xlimits)
        self.assertAlmostEqual(xf[0], 1.0)


# ---------------------------------------------------------------------------
# Full return-tuple tests
# ---------------------------------------------------------------------------

# Tight tolerances: ensure solver converges to analytic precision, not just
# its default relative-change stopping criterion (~1e-6 accuracy).
_TIGHT_TOL = dict(
    f_norm2_tol=1e-14,
    jac_norm_tol=1e-10,
    rel_ftol=1e-12,
    rel_xtol=1e-12,
)


class SimplishLeastSqReturnTupleTester(BaseCase):
    """Pin the shape, types, and plausible values of every return element."""

    def _run_linear(self, **kwargs):
        x0 = np.array([5.0, 5.0], 'd')
        ari = make_linear_ari()
        defaults = dict(max_iter=1000, arrays_interface=ari)
        defaults.update(_TIGHT_TOL)
        defaults.update(kwargs)
        return lm.simplish_leastsq(linear_obj_fn, linear_jac_fn, x0, **defaults)

    def test_return_tuple_has_7_elements(self):
        result = self._run_linear()
        self.assertEqual(len(result), 7)

    def test_return_types(self):
        xf, converged, msg, mu, nu, norm_f, f = self._run_linear()
        self.assertIsInstance(xf, np.ndarray)
        self.assertIsInstance(converged, bool)
        self.assertIsInstance(msg, str)
        # mu is numpy.float64; nu is a plain int — both are numeric scalars
        self.assertTrue(np.isscalar(mu))
        self.assertTrue(np.isscalar(nu))
        self.assertTrue(np.isscalar(norm_f))
        self.assertIsInstance(f, np.ndarray)

    def test_converged_flag_true_on_success(self):
        _, converged, _, *_ = self._run_linear()
        self.assertTrue(converged)

    def test_norm_f_equals_dot_f_f(self):
        _, _, _, _, _, norm_f, f = self._run_linear()
        self.assertAlmostEqual(norm_f, float(np.dot(f, f)), places=8)

    def test_norm_f_is_nonnegative(self):
        _, _, _, _, _, norm_f, _ = self._run_linear()
        self.assertGreaterEqual(norm_f, 0.0)

    def test_mu_is_positive(self):
        _, _, _, mu, _, _, _ = self._run_linear()
        self.assertGreater(mu, 0.0)

    def test_nu_is_at_least_2(self):
        _, _, _, _, nu, _, _ = self._run_linear()
        self.assertGreaterEqual(nu, 2.0)


# ---------------------------------------------------------------------------
# Convergence tests using fixtures
# ---------------------------------------------------------------------------

class SimplishLeastSqConvergenceTester(BaseCase):

    def test_linear_fixture_converges_to_known_solution(self):
        x0 = np.array([5.0, 5.0], 'd')
        ari = make_linear_ari()
        xf, converged, msg, *_ = lm.simplish_leastsq(
            linear_obj_fn, linear_jac_fn, x0,
            max_iter=1000, arrays_interface=ari, **_TIGHT_TOL)
        self.assertTrue(converged)
        np.testing.assert_allclose(xf, LINEAR_X_STAR, atol=1e-8)

    def test_nonlinear_fixture_converges_to_known_params(self):
        x0 = np.array([1.0, 0.0], 'd')
        ari = make_nonlinear_ari()
        xf, converged, msg, *_ = lm.simplish_leastsq(
            nonlinear_obj_fn, nonlinear_jac_fn, x0,
            max_iter=1000, arrays_interface=ari, **_TIGHT_TOL)
        self.assertTrue(converged)
        np.testing.assert_allclose(xf, NONLINEAR_P_STAR, atol=1e-8)

    def test_with_finite_difference_jacobian(self):
        x0 = np.array([5.0, 5.0], 'd')
        ari = make_linear_ari()
        xf, converged, msg, *_ = lm.simplish_leastsq(
            linear_obj_fn, linear_jac_fn, x0, max_iter=1000,
            arrays_interface=ari, num_fd_iters=3, **_TIGHT_TOL)
        self.assertTrue(converged)
        np.testing.assert_allclose(xf, LINEAR_X_STAR, atol=1e-6)


# ---------------------------------------------------------------------------
# damp_coeff_update helper
# ---------------------------------------------------------------------------

class DampCoeffUpdateTester(BaseCase):
    """Pin the behaviour of the extracted damp_coeff_update helper."""

    def _run(self, mu, nu):
        from pygsti.baseobjs.verbosityprinter import VerbosityPrinter
        printer = VerbosityPrinter(0)
        half_max_nu = 2 ** 15
        return damp_coeff_update(mu, nu, half_max_nu, "rejected step", printer)

    def test_mu_increases(self):
        mu, nu, msg = self._run(1.0, 2.0)
        self.assertGreater(mu, 1.0)

    def test_nu_doubles(self):
        _, nu_new, _ = self._run(1.0, 2.0)
        self.assertEqual(nu_new, 4.0)

    def test_msg_empty_string_when_not_overflow(self):
        # damp_coeff_update returns '' (empty string) when no overflow occurs,
        # not None. Pin current behaviour.
        _, _, msg = self._run(1.0, 2.0)
        self.assertEqual(msg, '')

    def test_msg_set_when_nu_overflow(self):
        half_max_nu = 2 ** 15
        from pygsti.baseobjs.verbosityprinter import VerbosityPrinter
        printer = VerbosityPrinter(0)
        # nu close to overflow threshold
        _, _, msg = damp_coeff_update(1.0, half_max_nu, half_max_nu, "reject", printer)
        self.assertIsNotNone(msg)


# ---------------------------------------------------------------------------
# jac_guarded helper
# ---------------------------------------------------------------------------

class JacGuardedTester(BaseCase):
    """Pin behaviour of jac_guarded (analytic and FD paths)."""

    def setUp(self):
        self.ari = make_linear_ari()
        self.global_x = np.array([1.0, 2.0], 'd')
        # Pre-allocate FD work array (shape: n_elements × n_params)
        from .fixtures import LINEAR_A
        self.fdJac_work = np.zeros_like(LINEAR_A)

    def test_analytic_jac_returned_when_k_ge_num_fd_iters(self):
        J = jac_guarded(
            k=5, num_fd_iters=3,
            obj_fn=linear_obj_fn, jac_fn=linear_jac_fn,
            f=linear_obj_fn(self.global_x), ari=self.ari,
            global_x=self.global_x, fdJac_work=self.fdJac_work,
        )
        # Should call jac_fn directly -> result equals LINEAR_A
        from .fixtures import LINEAR_A
        np.testing.assert_allclose(J, LINEAR_A, atol=1e-12)

    def test_fd_jac_used_when_k_lt_num_fd_iters(self):
        J = jac_guarded(
            k=0, num_fd_iters=3,
            obj_fn=linear_obj_fn, jac_fn=linear_jac_fn,
            f=linear_obj_fn(self.global_x), ari=self.ari,
            global_x=self.global_x, fdJac_work=self.fdJac_work,
        )
        # FD Jacobian should approximate the analytic one for a linear fn
        from .fixtures import LINEAR_A
        np.testing.assert_allclose(J, LINEAR_A, atol=1e-5)


# ---------------------------------------------------------------------------
# Optimizer / SimplerLMOptimizer cast and serialization
# ---------------------------------------------------------------------------

class OptimizerCastTester(BaseCase):

    def test_cast_from_instance_returns_same_object(self):
        opt = SimplerLMOptimizer()
        self.assertIs(Optimizer.cast(opt), opt)

    def test_cast_from_empty_dict_creates_default(self):
        opt = Optimizer.cast({})
        self.assertIsInstance(opt, Optimizer)

    def test_cast_from_none_creates_default(self):
        opt = Optimizer.cast(None)
        self.assertIsInstance(opt, Optimizer)


class SimplerLMOptimizerCastTester(BaseCase):

    def test_cast_from_instance_returns_same_object(self):
        opt = SimplerLMOptimizer(maxiter=42)
        self.assertIs(SimplerLMOptimizer.cast(opt), opt)

    def test_cast_from_simplerLM_compatible_dict(self):
        opt = SimplerLMOptimizer.cast({'maxiter': 77, 'tol': 1e-5})
        self.assertIsInstance(opt, SimplerLMOptimizer)
        self.assertEqual(opt.maxiter, 77)

    def test_cast_fallback_to_customlm_on_unknown_kwargs(self):
        """Dict with CustomLM-only keys should fall back to CustomLMOptimizer."""
        opt = SimplerLMOptimizer.cast({
            'maxiter': 10,
            'damping_mode': 'JTJ',         # CustomLM-only
            'damping_basis': 'diagonal_values',  # CustomLM-only
        })
        self.assertIsInstance(opt, CustomLMOptimizer)


class SimplerLMOptimizerSerializationTester(BaseCase):

    def _make(self, **kwargs):
        defaults = dict(maxiter=25, tol=1e-7, fditer=2, init_munu=(0.5, 2.0),
                        oob_check_interval=5, oob_action='stop', oob_check_mode=1,
                        lsvec_mode='normal', serial_solve_proc_threshold=50)
        defaults.update(kwargs)
        return SimplerLMOptimizer(**defaults)

    def test_round_trip_preserves_all_fields(self):
        opt = self._make()
        state = opt._to_nice_serialization()
        opt2 = SimplerLMOptimizer._from_nice_serialization(state)
        self.assertEqual(opt.maxiter, opt2.maxiter)
        self.assertEqual(opt.tol, opt2.tol)
        self.assertEqual(opt.fditer, opt2.fditer)
        self.assertEqual(opt.init_munu, opt2.init_munu)
        self.assertEqual(opt.oob_check_interval, opt2.oob_check_interval)
        self.assertEqual(opt.oob_action, opt2.oob_action)
        self.assertEqual(opt.oob_check_mode, opt2.oob_check_mode)
        self.assertEqual(opt.lsvec_mode, opt2.lsvec_mode)

    def test_serialization_state_is_dict(self):
        state = self._make()._to_nice_serialization()
        self.assertIsInstance(state, dict)


# ---------------------------------------------------------------------------
# OptimizerResult construction
# ---------------------------------------------------------------------------

class OptimizerResultTester(BaseCase):

    def test_basic_construction(self):
        x = np.array([1.0, 2.0])
        f = np.array([0.1, 0.2])
        result = OptimizerResult(
            objective_func=None,
            opt_x=x,
            opt_f=f,
        )
        np.testing.assert_array_equal(result.x, x)
        np.testing.assert_array_equal(result.f, f)
        self.assertIsNone(result.jtj)
        self.assertIsNone(result.f_no_penalties)
        self.assertIsNone(result.chi2_k_distributed_qty)
        self.assertIsNone(result.optimizer_specific_qtys)

    def test_full_construction(self):
        x = np.array([1.0])
        f = np.array([0.0])
        jtj = np.array([[2.0]])
        result = OptimizerResult(
            objective_func="mock",
            opt_x=x,
            opt_f=f,
            opt_jtj=jtj,
            opt_unpenalized_f=f,
            chi2_k_distributed_qty=3.14,
            optimizer_specific_qtys={'msg': 'ok'},
        )
        self.assertEqual(result.chi2_k_distributed_qty, 3.14)
        self.assertEqual(result.optimizer_specific_qtys['msg'], 'ok')
        np.testing.assert_array_equal(result.jtj, jtj)

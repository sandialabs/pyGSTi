import numpy as np

from pygsti.optimize import arraysinterface as _ari
from pygsti.optimize import simplerlm as lm
from ..util import BaseCase


def f(x):
    return np.inf * np.ones(2, 'd')


def jac(x):
    return np.zeros((2, 3), 'd')

def g(x):
    return np.array([x[0] ** 2],'d')

def gjac(x):
    return np.array([2 * x[0]],'d')


class LMTester(BaseCase):
    def test_simplish_leastsq_infinite_objective_fn_norm_at_x0(self):
        x0 = np.ones(3, 'd')
        ari = _ari.UndistributedArraysInterface(2, 3)
        xf, converged, msg, *_ = lm.simplish_leastsq(f, jac, x0, arrays_interface=ari)
        self.assertEqual(msg, "Infinite norm of objective function at initial point!")

    def test_simplish_leastsq_max_iterations_exceeded(self):
        x0 = np.ones(3, 'd')
        ari = _ari.UndistributedArraysInterface(2, 3)
        xf, converged, msg, *_ = lm.simplish_leastsq(f, jac, x0, max_iter=0, arrays_interface=ari)
        self.assertEqual(msg, "Maximum iterations (0) exceeded")

    def test_simplish_leastsq_x_limits(self):
        #perform optimization of g(x), which has minimum at 0, using limits so x must be 10 > x > 1
        # and check that the optimal x is near 1.0:
        x0 = np.array([6.0],'d')
        xlimits = np.array([[1.0, 10.0]], 'd')
        ari = _ari.UndistributedArraysInterface(1, 1)
        xf, converged, msg, *_ = lm.simplish_leastsq(g, gjac, x0, max_iter=100, arrays_interface=ari,
                                                   x_limits=xlimits)
        self.assertAlmostEqual(xf[0], 1.0)

    def test_linesearch_wrong_basin_witness(self):
        # r(x) = x + sin(10*x) - 0.5 is oscillatory with many local minima of the sum-of-squares.
        # From x0=0.8 the full LM step overshoots the nearby true root (x~0.994) into a
        # neighboring-but-worse local minimum (x~1.718, norm_f~0.05). The line search should
        # backtrack along the same step ray and land at the good root instead.
        target = 0.5

        def obj(x, oob_check=False):
            return np.array([x[0] + np.sin(10 * x[0]) - target])

        def jac(x):
            return np.array([[1 + 10 * np.cos(10 * x[0])]])

        x0 = np.array([0.8])

        ari = _ari.UndistributedArraysInterface(1, 1)
        xf_none, _, _, _, _, norm_f_none, _ = lm.simplish_leastsq(
            obj, jac, x0.copy(), max_iter=100, arrays_interface=ari, linesearch={'mode': 'none'})
        # This assertion is the witness that the test still captures the pathology: if a future
        # change makes 'none' converge to the good basin too, this should fail loudly rather than
        # let the test go vacuous.
        self.assertGreater(norm_f_none, 0.01)

        for mode in ('guarded', 'always'):
            ari = _ari.UndistributedArraysInterface(1, 1)
            xf, _, _, _, _, norm_f, _ = lm.simplish_leastsq(
                obj, jac, x0.copy(), max_iter=100, arrays_interface=ari, linesearch={'mode': mode})
            self.assertLess(norm_f, 1e-6, msg="linesearch mode %s failed to reach the good basin" % mode)
            self.assertAlmostEqual(xf[0], 0.99416, places=4)

    def test_linesearch_nonfinite_rescue(self):
        # r(x) = exp(1.5*x) - 2 has a near-zero Jacobian far in the negative tail, so an
        # undamped full LM step from x0=-3.7 overshoots into positive territory so far that
        # exp overflows to inf. A shrunk step along the same ray stays finite and converges.
        k = 1.5

        def obj(x, oob_check=False):
            with np.errstate(over='ignore'):
                return np.array([np.exp(k * x[0]) - 2.0])

        def jac(x):
            with np.errstate(over='ignore'):
                return np.array([[k * np.exp(k * x[0])]])

        x0 = np.array([-3.7])

        with np.errstate(over='ignore', invalid='ignore'):
            ari = _ari.UndistributedArraysInterface(1, 1)
            xf_none, converged_none, msg_none, *_ = lm.simplish_leastsq(
                obj, jac, x0.copy(), max_iter=20, max_dx_scale=None, arrays_interface=ari,
                linesearch={'mode': 'none'})
        self.assertFalse(converged_none)
        self.assertEqual(msg_none, "Infinite norm of objective function!")

        ari = _ari.UndistributedArraysInterface(1, 1)
        xf, converged, msg, _, _, norm_f, _ = lm.simplish_leastsq(
            obj, jac, x0.copy(), max_iter=20, max_dx_scale=None, arrays_interface=ari,
            linesearch={'mode': 'guarded'})
        self.assertTrue(converged)
        self.assertLess(norm_f, 1e-6)

    def test_linesearch_guarded_quiet_on_healthy_problem(self):
        # On a well-conditioned problem the guarded trigger should never fire, so 'guarded'
        # and 'none' must produce identical results.
        x0 = np.array([6.0], 'd')

        ari = _ari.UndistributedArraysInterface(1, 1)
        xf_none, _, _, _, _, norm_f_none, _ = lm.simplish_leastsq(
            g, gjac, x0.copy(), max_iter=100, arrays_interface=ari, linesearch={'mode': 'none'})

        ari = _ari.UndistributedArraysInterface(1, 1)
        xf_guarded, _, _, _, _, norm_f_guarded, _ = lm.simplish_leastsq(
            g, gjac, x0.copy(), max_iter=100, arrays_interface=ari, linesearch={'mode': 'guarded'})

        self.assertEqual(norm_f_none, norm_f_guarded)
        np.testing.assert_array_equal(xf_none, xf_guarded)

    def test_simplerlm_optimizer_linesearch_serialization_roundtrip(self):
        opt = lm.SimplerLMOptimizer(linesearch={'mode': 'always', 'beta': 0.5, 'max_evals': 3, 'kappa': 2.0})
        state = opt._to_nice_serialization()
        opt2 = lm.SimplerLMOptimizer._from_nice_serialization(state)
        self.assertEqual(opt2.linesearch, {'mode': 'always', 'beta': 0.5, 'max_evals': 3, 'kappa': 2.0})

        # A state dict without the 'linesearch' key (as produced by code predating this feature)
        # should deserialize to the defaults rather than raising a KeyError.
        del state['linesearch']
        opt3 = lm.SimplerLMOptimizer._from_nice_serialization(state)
        self.assertEqual(opt3.linesearch, {'mode': 'guarded', 'beta': 0.25, 'max_evals': 6, 'kappa': 1.0})

    def test_tol_normalization(self):
        # A partial `tol` dict should have its missing keys filled with the values a float
        # tol=1e-6 would have produced, so that `run()`'s direct indexing never KeyErrors.
        opt = lm.SimplerLMOptimizer(tol={'relf': 1e-5})
        self.assertEqual(opt.tol, {'relx': 1e-8, 'relf': 1e-5, 'f': 1.0, 'jac': 1e-6, 'maxdx': 1.0})

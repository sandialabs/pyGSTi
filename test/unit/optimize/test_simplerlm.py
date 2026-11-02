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


class DegenerateStartTester(BaseCase):
    # r(x) = [x0^2 - 0.04, x1 - 1]: at x = 0 the x0 column of the Jacobian is zero (as for a
    # Lindblad rate parameterized by its square root), and x2 never enters r at all.

    def setUp(self):
        self.jac_points = []

    def obj(self, x, oob_check=False):
        return np.array([x[0] ** 2 - 0.04, x[1] - 1.0])

    def jac(self, x):
        self.jac_points.append(np.array(x, copy=True))
        return np.array([[2 * x[0], 0.0, 0.0], [0.0, 1.0, 0.0]])

    def test_zero_columns_escape_without_finite_differences(self):
        ari = _ari.UndistributedArraysInterface(2, 3)
        xf, converged, msg, *_ = lm.simplish_leastsq(self.obj, self.jac, np.zeros(3), max_iter=100,
                                                     arrays_interface=ari)
        self.assertTrue(converged)
        np.testing.assert_allclose(xf, [0.2, 1.0, 0.0], atol=1e-3)
        self.assertEqual(xf[2], 0.0)  # an untouched parameter is never moved by the shift
        np.testing.assert_array_equal(self.jac_points[0], np.zeros(3))
        np.testing.assert_array_equal(self.jac_points[1], [lm.JAC_SHIFT, 0.0, lm.JAC_SHIFT])

    def test_no_shift_without_zero_columns(self):
        # Add r_2 = x2 so that no column is zero at x0; x2 is already optimal, so it stays at
        # exactly 0 unless something shifts it.
        def obj(x, oob_check=False):
            return np.array([x[0] ** 2 - 0.04, x[1] - 1.0, x[2]])

        def jac(x):
            self.jac_points.append(np.array(x, copy=True))
            return np.array([[2 * x[0], 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

        ari = _ari.UndistributedArraysInterface(3, 3)
        xf, converged, msg, *_ = lm.simplish_leastsq(obj, jac, np.array([0.5, 0.0, 0.0]), max_iter=100,
                                                     arrays_interface=ari, linesearch={'mode': 'none'})
        self.assertTrue(converged)
        np.testing.assert_allclose(xf, [0.2, 1.0, 0.0], atol=1e-3)
        self.assertTrue(all(p[2] == 0.0 for p in self.jac_points))


class FiniteDifferenceDeprecationTester(BaseCase):

    def test_fditer_arguments_warn_and_are_ignored(self):
        from pygsti.tools.exceptions import pyGSTiDeprecationWarning
        for kwargs in ({'fditer': 1}, {'first_fditer': 1}):
            with self.assertWarns(pyGSTiDeprecationWarning):
                opt = lm.SimplerLMOptimizer(**kwargs)
            self.assertFalse(hasattr(opt, 'fditer') or hasattr(opt, 'first_fditer'))

    def test_old_state_with_fditer_keys_loads(self):
        state = lm.SimplerLMOptimizer(maxiter=7)._to_nice_serialization()
        self.assertNotIn('number_of_finite_difference_iterations', state)
        self.assertNotIn('number_of_first_stage_finite_difference_iterations', state)
        state['number_of_finite_difference_iterations'] = 2  # as saved by older versions
        state['number_of_first_stage_finite_difference_iterations'] = 1
        opt = lm.SimplerLMOptimizer._from_nice_serialization(state)
        self.assertEqual(opt.maxiter, 7)

    def test_cast_does_not_fall_back_to_customlm(self):
        with self.assertRaises(TypeError):
            lm.SimplerLMOptimizer.cast({'damping_mode': 'identity'})
        self.assertIsInstance(lm.SimplerLMOptimizer.cast({'maxiter': 7}), lm.SimplerLMOptimizer)


class CoefficientStepTester(BaseCase):
    # r(x) = [A b(C C^dagger) - y, x_4 - 1]: one 2x2 Cholesky-parameterized block C (parameters 0-3,
    # layout of lindbladcoefficients.cholesky_params_to_factor) and one ordinary parameter (4). The
    # residual is linear in B's coefficient coordinates b, and its derivative vanishes at C = 0.

    def setUp(self):
        from pygsti.modelmembers.operations import lindbladcoefficients as lc
        self.lc = lc
        self.A = np.random.default_rng(3).standard_normal((6, 4))
        u = np.array([1.0, 0.5 - 0.5j])
        self.set_target(np.outer(u, u.conj()))  # rank one: on the boundary of the PSD cone
        self.blocks = [(np.arange(4), 2)]

    def set_target(self, B):
        self.B_target = B
        self.y = self.A @ self.lc.hermitian_to_coefficient_coords(B)

    def B_of(self, x):
        C = self.lc.cholesky_params_to_factor(x[:4], 2)
        return C @ C.conj().T

    def obj(self, x, oob_check=False):
        return np.concatenate([self.A @ self.lc.hermitian_to_coefficient_coords(self.B_of(x)) - self.y, [x[4] - 1.0]])

    def jac(self, x):
        J = np.zeros((7, 5))
        J[:6, :4] = self.A @ self.lc.cholesky_coords_jacobian(self.lc.cholesky_params_to_factor(x[:4], 2))
        J[6, 4] = 1.0
        return J

    def solve(self, **kwargs):
        ari = _ari.UndistributedArraysInterface(7, 5)
        return lm.simplish_leastsq(self.obj, self.jac, np.zeros(5), max_iter=100, f_norm2_tol=1e-20,
                                   arrays_interface=ari, **kwargs)

    def test_coefficient_steps_reach_a_boundary_solution_from_zero(self):
        xf, converged, msg, *_ = self.solve(coefficient_blocks=self.blocks, coefficient_iters=10)
        self.assertTrue(converged)
        np.testing.assert_allclose(self.B_of(xf), self.B_target, atol=1e-6)
        self.assertAlmostEqual(xf[4], 1.0, places=6)

    def test_coefficient_steps_keep_the_block_psd(self):
        u = np.array([1.0, 0.5 - 0.5j])
        self.set_target(np.outer(u, u.conj()) - 0.3 * np.eye(2))  # indefinite: not reachable
        xf, converged, msg, *_ = self.solve(coefficient_blocks=self.blocks, coefficient_iters=100)
        self.assertTrue(converged)
        self.assertGreater(np.linalg.eigvalsh(self.B_of(xf)).min(), -1e-12)

    def test_inactive_or_empty_coefficient_blocks_change_nothing(self):
        ref = self.solve()
        for kwargs in ({'coefficient_blocks': self.blocks, 'coefficient_iters': 0},
                       {'coefficient_blocks': [], 'coefficient_iters': 10}):
            out = self.solve(**kwargs)
            np.testing.assert_array_equal(out[0], ref[0])
            self.assertEqual(out[2], ref[2])

    def test_steps_from_a_shifted_point_are_line_searched(self):
        # The block starts at C = 0, so the first step is computed at a shifted point. Even with
        # linesearch mode 'none' it is line-searched: after the full-step trial comes one at t = beta.
        trial_x4 = []

        def obj(x, oob_check=False):
            trial_x4.append(x[4])
            return self.obj(x)

        ari = _ari.UndistributedArraysInterface(7, 5)
        lm.simplish_leastsq(obj, self.jac, np.zeros(5), max_iter=1, f_norm2_tol=1e-20, arrays_interface=ari,
                            coefficient_blocks=self.blocks, coefficient_iters=1, linesearch={'mode': 'none'})
        moved = [v for v in trial_x4 if v != 0.0]  # x[4] moves linearly along the step
        self.assertGreaterEqual(len(moved), 2)
        self.assertAlmostEqual(moved[1], 0.25 * moved[0])

    def test_coefficient_blocks_reject_x_limits(self):
        with self.assertRaises(ValueError):
            self.solve(coefficient_blocks=self.blocks, coefficient_iters=1,
                       x_limits=np.array([[-10.0, 10.0]] * 5))


class CoefficientStepOptionTester(BaseCase):

    def test_option_normalization_and_validation(self):
        defaults = {'blocks': 'instruments', 'shift': 1e-2, 'iters': 10}
        self.assertEqual(lm.SimplerLMOptimizer().coefficient_steps, defaults)  # on by default
        self.assertEqual(lm.SimplerLMOptimizer(coefficient_steps=True).coefficient_steps, defaults)
        self.assertIsNone(lm.SimplerLMOptimizer(coefficient_steps=False).coefficient_steps)
        self.assertEqual(lm.SimplerLMOptimizer(coefficient_steps={'iters': 3}).coefficient_steps,
                         {'blocks': 'instruments', 'shift': 1e-2, 'iters': 3})
        for bad in ({'blocks': 'all'}, {'shift': 0.0}, {'iters': -1}, {'iters': 1.5}, {'speed': 2}):
            with self.assertRaises(ValueError):
                lm.SimplerLMOptimizer(coefficient_steps=bad)

    def test_serialization_roundtrip(self):
        for value in ({'shift': 1e-3}, False):
            opt = lm.SimplerLMOptimizer(coefficient_steps=value)
            state = opt._to_nice_serialization()
            self.assertEqual(lm.SimplerLMOptimizer._from_nice_serialization(state).coefficient_steps,
                             opt.coefficient_steps)
        del state['coefficient_steps']  # as saved before this option existed: take the default
        self.assertEqual(lm.SimplerLMOptimizer._from_nice_serialization(state).coefficient_steps,
                         lm.SimplerLMOptimizer().coefficient_steps)

    def test_instrument_cholesky_blocks(self):
        from pygsti.modelmembers.instruments import Instrument
        from pygsti.modelpacks import smq1Q_XYI as std
        mdl = std.target_model()
        E0, E1 = mdl.effects['0'].to_dense(), mdl.effects['1'].to_dense()
        mdl[('Iz', 0)] = Instrument({'p0': np.outer(E0, E0), 'p1': np.outer(E1, E1)})

        cp = mdl.copy()
        cp.set_all_parameterizations('CPTPLND')
        blocks = lm._instrument_cholesky_blocks(cp)
        self.assertEqual([n for _, n in blocks], [3, 3, 3])
        inst = cp.instruments[('Iz', 0)]
        errgens = [inst['p0'].factorops[0]._effect.error_map.errorgen,  # the shared POVM error map
                   inst['p0'].factorops[1].factorops[1].errorgen, inst['p1'].factorops[1].factorops[1].errorgen]
        # each errorgen holds 3 Hamiltonian parameters, then its 9 Cholesky parameters
        expected = sorted(int(i) for eg in errgens for i in eg.gpindices_as_array()[3:])
        self.assertEqual(sorted(int(i) for idx, _ in blocks for i in idx), expected)

        tp = mdl.copy()
        tp.set_all_parameterizations('full TP')
        self.assertEqual(lm._instrument_cholesky_blocks(tp), [])
        no_inst = std.target_model()
        no_inst.set_all_parameterizations('CPTPLND')
        self.assertEqual(lm._instrument_cholesky_blocks(no_inst), [])

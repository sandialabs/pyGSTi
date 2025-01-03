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

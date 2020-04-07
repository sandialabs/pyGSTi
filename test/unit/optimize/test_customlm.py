import numpy as np

from ..util import BaseCase

from pygsti.optimize import customlm as lm


def f(x):
    return np.inf * np.ones(2, 'd')


def jac(x):
    return np.zeros((2, 3), 'd')


class CustomLMTester(BaseCase):
    def test_custom_leastsq_infinite_objective_fn_norm_at_x0(self):
        x0 = np.ones(3, 'd')
        xf, converged, msg, mu, nu = lm.custom_leastsq(f, jac, x0)
        self.assertEqual(msg, "Infinite norm of objective function at initial point!")

    def test_custom_leastsq_max_iterations_exceeded(self):
        x0 = np.ones(3, 'd')
        xf, converged, msg, mu, nu = lm.custom_leastsq(f, jac, x0, max_iter=0)
        self.assertEqual(msg, "Maximum iterations (0) exceeded")

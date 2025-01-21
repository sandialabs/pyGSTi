from pygsti.optimize import customcg as cg
from ..util import BaseCase


class CustomCGTester(BaseCase):
    def test_maximize1D_closed_domain(self):
        def g(x):  # a function with tricky boundaries (|x| only defined in [-2,2]
            if x < -2.0 or x > 2.0: return None
            else: return abs(x)

        start = -2.0
        guess = 4.0  # None
        cg._maximize_1d(g, start, guess, g(start))
        # TODO assert correctness

        start = -3.0  # None
        guess = -1.5
        cg._maximize_1d(g, start, guess, g(start))
        # TODO assert correctness

    def test_maximize1D_open_domain(self):
        def g(x):  # a bad function with a gap between it's boundaries
            if x > -2.0 and x < 2.0: return None
            else: return 1.0

        start = -4.0
        guess = 4.0
        cg._maximize_1d(g, start, guess, g(start))
        # TODO assert correctness

import unittest
import pygsti
import numpy as np
import sys
import warnings
import os

from ..testutils import BaseTestCase, temp_files, compare_files

def f(x):
    return np.dot(x,x)

def f_vec(x):
    return np.array( [np.dot(x,x)] )

def jac(x):
    return 2*x[None,:]

class TestOptimizeMethods(BaseTestCase):

    def setUp(self):
        super(TestOptimizeMethods, self).setUp()

        self.x0 = np.array( [10,5], 'd')
        self.answer = np.array( [0,0], 'd')

    def test_optimize(self):
        old_stdout = sys.stdout
        sys.stdout = open(temp_files + "/optimize.out","w")

        for method in ("simplex","supersimplex","customcg","basinhopping","CG","BFGS","L-BFGS-B"): #"homebrew"
            result = pygsti.optimize.minimize(f, self.x0, method, maxiter=1000)
            self.assertArraysAlmostEqual(result.x, self.answer)

        for method in ("simplex","supersimplex","customcg","basinhopping","CG","BFGS","L-BFGS-B"): #"homebrew"
            result = pygsti.optimize.minimize(f, self.x0, method, maxiter=1) #max iterations exceeded...

        result = pygsti.optimize.minimize(f, self.x0, "swarm", maxiter=30)
        self.assertArraysAlmostEqual(result.x, self.answer)

        result = pygsti.optimize.minimize(f, self.x0, "brute", maxiter=10)
        self.assertArraysAlmostEqual(result.x, self.answer)

        try:
            import deap
            doTest = True
        except ImportError:
            warnings.warn("**** IMPORT: Cannot import deap, and so evolutionary"
                          + " optimization test has been skipped")
            doTest = False

        if doTest:
            result = pygsti.optimize.minimize(f, self.x0, "evolve", maxiter=20)
            self.assertLess(np.linalg.norm(result.x-self.answer), 0.1)
              #takes too long to converge...

        sys.stdout.close()
        sys.stdout = old_stdout


    def test_checkjac(self):
        x0 = self.x0
        pygsti.optimize.check_jac(f_vec, x0, jac(x0), eps=1e-10, tol=1e-6, errType='rel')
        pygsti.optimize.check_jac(f_vec, x0, jac(x0), eps=1e-10, tol=1e-6, errType='abs')

    def test_customcg_helpers(self):
        #Run helper routines to customcg to make sure they at least execute:
        def g(x): # a function with tricky boundaries (|x| only defined in [-2,2]
            if x < -2.0 or x > 2.0: return None
            else: return abs(x)

        start = -2.0
        guess = 4.0 # None
        pygsti.optimize.customcg._maximize1D(g,start,guess,g(start))

        start = -3.0 #None
        guess = -1.5
        pygsti.optimize.customcg._maximize1D(g,start,guess,g(start))


        def g(x): # a bad function with a gap between it's boundaries
            if x > -2.0 and x < 2.0: return None
            else: return 1.0

        start = -4.0
        guess = 4.0
        pygsti.optimize.customcg._maximize1D(g,start,guess,g(start))

    def test_customlm(self):
        #Test a few boundary cases
        def f(x):
            return np.inf * np.ones(2,'d')
        def jac(x):
            return np.zeros( (2,3), 'd')

        x0 = np.ones(3,'d')
        xf, converged, msg = pygsti.optimize.customlm.custom_leastsq(f, jac, x0)
        self.assertEqual(msg, "Infinite norm of objective function at initial point!")

        xf, converged, msg = pygsti.optimize.customlm.custom_leastsq(f, jac, x0, max_iter=0)
        self.assertEqual(msg, "Maximum iterations (0) exceeded")



if __name__ == "__main__":
    unittest.main(verbosity=2)

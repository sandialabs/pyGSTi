import unittest
import pygsti
import numpy as np
import sys
import warnings

def f(x):
    return np.dot(x,x)

def f_vec(x):
    return np.array( [np.dot(x,x)] )

def jac(x):
    return 2*x[None,:]

class OptimizeTestCase(unittest.TestCase):

    def setUp(self):
        #Set GateSet objects to "strict" mode for testing
        pygsti.objects.GateSet._strict = True

        self.x0 = np.array( [10,5], 'd')
        self.answer = np.array( [0,0], 'd')

    def assertArraysAlmostEqual(self,a,b):
        self.assertAlmostEqual( np.linalg.norm(a-b), 0 )


class TestOptimizeMethods(OptimizeTestCase):
    
    def test_optimize(self):
        old_stdout = sys.stdout
        sys.stdout = open("temp_test_files/optimize.out","w")
        
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

      
if __name__ == "__main__":
    unittest.main(verbosity=2)

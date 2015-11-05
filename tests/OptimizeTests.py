import unittest
import GST
import numpy as np
import sys

def f(x):
    return np.dot(x,x)

class OptimizeTestCase(unittest.TestCase):

    def setUp(self):
        self.x0 = np.array( [10,5], 'd')
        self.answer = np.array( [0,0], 'd')

    def assertArraysAlmostEqual(self,a,b):
        self.assertAlmostEqual( np.linalg.norm(a-b), 0 )


class TestOptimizeMethods(OptimizeTestCase):
    
    def test_optimize(self):
        old_stdout = sys.stdout
        sys.stdout = open("temp_test_files/optimize.out","w")
        
        for method in ("simplex","supersimplex","customcg","basinhopping","CG","BFGS","L-BFGS-B"): #"homebrew"
            result = GST.Optimize.minimize(f, self.x0, method, maxiter=1000)
            self.assertArraysAlmostEqual(result.x, self.answer)

        result = GST.Optimize.minimize(f, self.x0, "swarm", maxiter=30)
        self.assertArraysAlmostEqual(result.x, self.answer)

        result = GST.Optimize.minimize(f, self.x0, "brute", maxiter=10)
        self.assertArraysAlmostEqual(result.x, self.answer)

        result = GST.Optimize.minimize(f, self.x0, "evolve", maxiter=20)
        self.assertLess(np.linalg.norm(result.x-self.answer), 0.01) #takes too long to converge...
        
        sys.stdout.close()
        sys.stdout = old_stdout


      
if __name__ == "__main__":
    unittest.main(verbosity=2)

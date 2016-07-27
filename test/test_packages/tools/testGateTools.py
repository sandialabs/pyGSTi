from .toolsBaseCase import ToolsTestCase
import pygsti
import numpy as np
import unittest

class GateToolsTestCase(ToolsTestCase):

    def test_gate_tools(self):
        oneRealPair = np.array( [[1+1j, 0, 0, 0],
                             [ 0, 1-1j,0, 0],
                             [ 0,   0, 2, 0],
                             [ 0,   0,  0, 2]], 'complex')
        decomp = pygsti.decompose_gate_matrix(oneRealPair)
            #decompose gate mx whose eigenvalues have a real but non-unit pair

        dblRealPair = np.array( [[3, 0, 0, 0],
                             [ 0, 3,0, 0],
                             [ 0,   0, 2, 0],
                             [ 0,   0,  0, 2]], 'complex')
        decomp = pygsti.decompose_gate_matrix(dblRealPair)
            #decompose gate mx whose eigenvalues have two real but non-unit pairs


        unpairedMx = np.array( [[1+1j, 0, 0, 0],
                                [ 0, 2-1j,0, 0],
                                [ 0,   0, 2+2j, 0],
                                [ 0,   0,  0,  1.0+3j]], 'complex')
        decomp = pygsti.decompose_gate_matrix(unpairedMx)
            #decompose gate mx which has all complex eigenvalue -> bail out
        self.assertFalse(decomp['isValid'])

        largeMx = np.identity(16,'d')
        decomp = pygsti.decompose_gate_matrix(largeMx) #can only handle 1Q mxs
        self.assertFalse(decomp['isValid'])

        A = np.array( [[0.9, 0, 0.1j, 0],
                       [ 0,  0, 0,    0],
                       [ -0.1j, 0, 0, 0],
                       [ 0,  0,  0,  0.1]], 'complex')

        B = np.array( [[0.5, 0, 0, -0.2j],
                       [ 0,  0.25, 0,  0],
                       [ 0, 0, 0.25,   0],
                       [ 0.2j,  0,  0,  0.1]], 'complex')

        self.assertAlmostEqual( pygsti.frobeniusdist(A,A), 0.0 )
        self.assertAlmostEqual( pygsti.jtracedist(A,A,mxBasis="std"), 0.0 )
        self.assertAlmostEqual( pygsti.diamonddist(A,A,mxBasis="std"), 0.0 )
        self.assertAlmostEqual( pygsti.frobeniusdist(A,B), (0.430116263352+0j) )
        self.assertAlmostEqual( pygsti.jtracedist(A,B,mxBasis="std"), 0.260078105936)
        self.assertAlmostEqual( pygsti.diamonddist(A,B,mxBasis="std"), 0.614258836298)

        self.assertAlmostEqual( pygsti.frobeniusdist(A,B), pygsti.frobeniusnorm(A-B) )
        self.assertAlmostEqual( pygsti.frobeniusdist(A,B), np.sqrt( pygsti.frobeniusnorm2(A-B) ) )

if __name__ == '__main__':
    unittest.main(verbosity=2)

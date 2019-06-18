import unittest
import pygsti
import numpy as np
import pickle

from  pygsti.objects import DenseOperator
import pygsti.construction as pc
import scipy.sparse as sps

from ..testutils import BaseTestCase, compare_files, temp_files

class GateTestCase(BaseTestCase):

    def setUp(self):
        super(GateTestCase, self).setUp()

    def test_convert(self):
        densemx = np.array([[1,0,0,0],
                            [0,1,0,0],
                            [0,0,0,1],
                            [0,0,-1,0]],'d')

        basis = pygsti.obj.Basis.cast("pp",4)
        lndgate = pygsti.objects.LindbladDenseOp.from_operation_matrix(
            densemx,unitaryPostfactor=densemx,
            ham_basis=basis, nonham_basis=basis, param_mode="cptp",
            nonham_mode="all", truncate=True, mxBasis=basis)
        g = pygsti.objects.operation.convert(lndgate,"CPTP",basis)
        self.assertTrue(g is lndgate) #should be trivial (no) conversion

    def test_eigenvalue_param_gate(self):
        mx = np.array( [[ 1,   0,     0,       0],
                        [ 0,   1,     0,       0],
                        [ 0,   0,     -1, -1e-10],
                        [ 0,   0,  1e-10,     -1]], 'd')
        # degenerate (to tol) -1 evals will generate *complex* evecs
        g1 = pygsti.objects.EigenvalueParamDenseOp(
            mx,includeOffDiagsInDegen2Blocks=False,
            TPconstrainedAndUnital=False)

        mx = np.array( [[ 1,   0,     0,       0],
                        [ 0,   1,     0,       0],
                        [ 0,   0,     -1,      0],
                        [ 0,   0,     0,      -1]], 'complex')
        # 2 degenerate real pairs of evecs => should add off-diag els
        g2 = pygsti.objects.EigenvalueParamDenseOp(
            mx,includeOffDiagsInDegen2Blocks=True,
            TPconstrainedAndUnital=False)
        self.assertEqual(g2.params, [[(1.0, (0, 0))], [(1.0, (1, 1))],
                                     [(1.0, (0, 1))], [(1.0, (1, 0))], # off diags blk 1
                                     [(1.0, (2, 2))], [(1.0, (3, 3))],
                                     [(1.0, (2, 3))], [(1.0, (3, 2))]]) # off diags blk 2


        mx = np.array( [[ 1,   -0.1,     0,      0],
                        [ 0.1,    1,     0,      0],
                        [ 0,      0,     1+1,   -0.1],
                        [ 0,      0,   0.1,      1+1]], 'complex')
        # complex pairs of evecs => make sure combined parameters work
        g3 = pygsti.objects.EigenvalueParamDenseOp(
            mx,includeOffDiagsInDegen2Blocks=True,
            TPconstrainedAndUnital=False)
        self.assertEqual(g3.params, [
            [(1.0, (0, 0)), (1.0, (1, 1))], # single param that is Re part of 0,0 and 1,1 els
            [(1j, (0, 0)), (-1j, (1, 1))],  # Im part of 0,0 and 1,1 els
            [(1.0, (2, 2)), (1.0, (3, 3))], # Re part of 2,2 and 3,3 els
            [(1j, (2, 2)), (-1j, (3, 3))]   # Im part of 2,2 and 3,3 els
        ])


        mx = np.array( [[ 1,   -0.1,     0,      0],
                        [ 0.1,    1,     0,      0],
                        [ 0,      0,     1,   -0.1],
                        [ 0,      0,   0.1,      1]], 'complex')
        # 2 degenerate complex pairs of evecs => should add off-diag els
        g4 = pygsti.objects.EigenvalueParamDenseOp(
            mx,includeOffDiagsInDegen2Blocks=True,
            TPconstrainedAndUnital=False)
        self.assertArraysAlmostEqual(g4.evals, [1.+0.1j, 1.+0.1j, 1.-0.1j, 1.-0.1j]) # Note: evals are sorted!
        self.assertEqual(g4.params,[
            [(1.0, (0, 0)), (1.0, (2, 2))], # single param that is Re part of 0,0 and 2,2 els (conj eval pair, since sorted)
            [(1j, (0, 0)), (-1j, (2, 2))],  # Im part of 0,0 and 2,2 els
            [(1.0, (1, 1)), (1.0, (3, 3))], # Re part of 1,1 and 3,3 els
            [(1j, (1, 1)), (-1j, (3, 3))],  # Im part of 1,1 and 3,3 els
            [(1.0, (0, 1)), (1.0, (2, 3))], # Re part of 0,1 and 2,3 els (upper triangle)
            [(1j, (0, 1)), (-1j, (2, 3))],  # Im part of 0,1 and 2,3 els (upper triangle); (0,1) and (2,3) must be conjugates
            [(1.0, (1, 0)), (1.0, (3, 2))], # Re part of 1,0 and 3,2 els (lower triangle)
            [(1j, (1, 0)), (-1j, (3, 2))]   # Im part of 1,0 and 3,2 els (lower triangle); (1,0) and (3,2) must be conjugates
        ])




if __name__ == '__main__':
    unittest.main(verbosity=2)

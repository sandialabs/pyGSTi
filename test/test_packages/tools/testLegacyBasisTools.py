from ..testutils import BaseTestCase, compare_files, temp_files
import unittest
import numpy as np
import scipy
import pygsti

from pygsti.objects.basis import change_basis, basis_matrices
from pygsti.objects.basis import transform_matrix as basis_transform_matrix

from . import legacy_basis_tools as legacy

from pprint import pprint

class BasisBaseTestCase(BaseTestCase):

    def test_transforms(self):
        mxStd = np.array([[1,0,0,0],
                          [0,1,0,0],
                          [0,0,1,0],
                          [0,0,0,1]], 'complex')
        vecStd = np.array([1,0,0,0], 'complex')
        items = [mxStd, vecStd]
        bases = ['std', 'gm', 'pp']
        dims  = [2]

        for item in items:
            for basisA in bases:
                itemBasisA       = change_basis(item, 'std', basisA)
                itemBasisALegacy = legacy.change_basis(item, 'std', basisA)
                self.assertArraysAlmostEqual(itemBasisA, itemBasisALegacy)
                for basisB in bases:
                    itemBasisB       = change_basis(itemBasisA, basisA, basisB)
                    itemBasisBLegacy = legacy.change_basis(itemBasisALegacy, basisA, basisB)
                    self.assertArraysAlmostEqual(itemBasisB, itemBasisBLegacy)

                    for dim in dims:
                        modernTransform = basis_transform_matrix(basisA, basisB, dim)
                        legacyTransform = basis_transform_matrix(basisA, basisB, dim)
                        self.assertArraysAlmostEqual(modernTransform, legacyTransform)

                    itemBasisC       = change_basis(itemBasisB, basisB, basisA)
                    itemBasisCLegacy = legacy.change_basis(itemBasisBLegacy, basisB, basisA)
                    self.assertArraysAlmostEqual(itemBasisC, itemBasisCLegacy)

                    itemBasisD       = change_basis(itemBasisC, basisA, 'std')
                    itemBasisDLegacy = legacy.change_basis(itemBasisCLegacy, basisA, 'std')
                    self.assertArraysAlmostEqual(itemBasisD, item)
                    self.assertArraysAlmostEqual(itemBasisDLegacy, item)

    def test_other(self):
        mxGM = np.array([[ 0.5       ,  0.        ,  0.        , -0.5        , 0.70710678],
                         [ 0.        ,  0.        ,  0.        ,  0.         , 0.        ],
                         [ 0.        ,  0.        ,  0.        ,  0.         , 0.        ],
                         [-0.5       ,  0.        ,  0.        ,  0.5        , 0.70710678],
                         [ 0.70710678,  0.        ,  0.        ,  0.70710678 , 0.        ]])
        mxStd       = change_basis(mxGM, 'gm', 'std', [2,1])
        mxStdLegacy = legacy.change_basis(mxGM, 'gm', 'std', [2,1])
        print(mxGM)
        print(mxStd)
        print(mxStdLegacy)
        self.assertArraysAlmostEqual(mxStd, mxStdLegacy)

    def test_block_dims(self):
        bases = ['gm', 'std']
        for basisA in bases:
            for basisB in bases:
                dim = [2, 1]
                modernTransform = basis_transform_matrix(basisA, basisB, dim)
                legacyTransform = basis_transform_matrix(basisA, basisB, dim)
                self.assertArraysAlmostEqual(modernTransform, legacyTransform)

    def assertBasesAlmostEqual(self, a, b):
        for mxA, mxB in zip(a, b):
            self.assertArraysAlmostEqual(mxA, mxB)

    def test_matrices(self):
        basisDimPairs = [
                ('std', [2]),
                ('gm',  [2]),#, [2,1]]),
                ('pp',  [2]),
                ('qt',  [3])]
        for basis, dims in basisDimPairs:
            for dim in dims:
                print(basis, dim)
                modernMxs = basis_matrices(basis, dim)
                legacyMxs = legacy.basis_matrices(basis, dim)
                pprint(modernMxs)
                pprint(legacyMxs)
                self.assertBasesAlmostEqual(modernMxs, legacyMxs)

if __name__ == '__main__':
    unittest.main(verbosity=2)

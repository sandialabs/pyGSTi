from ..testutils import BaseTestCase, compare_files, temp_files
import unittest
import numpy as np
import scipy
import pygsti

import pygsti.tools.basistools as basistools
from pygsti.tools.basistools import change_basis, basis_matrices, basis_transform_matrix

from . import legacy_basis_tools as legacy

from pprint import pprint

class BasisBaseTestCase(BaseTestCase):

    def test_transforms(self):
        mxStd = np.array([[1,0,0,0],
                          [0,1,0,0],
                          [0,0,1,0],
                          [0,0,0,1]], 'complex')
        vecStd = np.array([1,0,0,0], 'complex')

        bases = ['std', 'gm', 'pp']
        dims  = [2]

        for basisA in bases:
            mxBasisA = change_basis(mxStd, 'std', basisA)
            mxBasisALegacy = legacy.change_basis(mxStd, 'std', basisA)
            self.assertArraysAlmostEqual(mxBasisA, mxBasisALegacy)
            for basisB in bases:
                mxBasisB       = change_basis(mxBasisA, basisA, basisB)
                mxBasisBLegacy = legacy.change_basis(mxBasisA, basisA, basisB)
                self.assertArraysAlmostEqual(mxBasisB, mxBasisBLegacy)

                for dim in dims:
                    modernTransform = basis_transform_matrix(basisA, basisB, dim)
                    legacyTransform = basis_transform_matrix(basisA, basisB, dim)
                    self.assertArraysAlmostEqual(modernTransform, legacyTransform)

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
                ('gm',  [2, [2,1]]),
                ('pp',  [2]),
                ('qt',  [3])]
        for basis, dims in basisDimPairs:
            for dim in dims:
                print(basis, dim)
                modernMxs = basis_matrices(basis, dim)
                legacyMxs = legacy.basis_matrices(basis, dim)
                pprint(modernMxs.matrices)
                pprint(legacyMxs)
                self.assertBasesAlmostEqual(modernMxs, legacyMxs)

if __name__ == '__main__':
    unittest.main(verbosity=2)

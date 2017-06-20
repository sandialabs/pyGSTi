from ..testutils import BaseTestCase, compare_files, temp_files
import unittest
import numpy as np
import scipy
import pygsti

import pygsti.tools.basistools as basistools
from pygsti.tools.basistools import change_basis, basis_matrices, basis_transform_matrix

from . import legacy_basis_tools as legacy

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
                mxBasisB = change_basis(mxBasisA, basisA, basisB)
                mxBasisBLegacy = legacy.change_basis(mxStd, basisA, basisB)
                self.assertArraysAlmostEqual(mxBasisB, mxBasisBLegacy)

                for dim in dims:
                    modernTransform = basis_transform_matrix(basisA, basisB, dim)
                    legacyTransform = basis_transform_matrix(basisA, basisB, dim)
                    self.assertArraysAlmostEqual(modernTransform, legacyTransform)

if __name__ == '__main__':
    unittest.main(verbosity=2)

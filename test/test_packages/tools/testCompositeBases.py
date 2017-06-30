from ..testutils import BaseTestCase, compare_files, temp_files
import unittest
import numpy as np
import scipy
import pygsti

import pygsti.tools.basistools       as basistools

class CompositeBasisTestCase(BaseTestCase):

    def test_build_composite_basis(self):
        a = basistools.Basis([('std', 2), ('std', 2)])
        b = basistools.Basis('std', [2,2])
        self.assertArraysAlmostEqual(np.array(a.matrices), np.array(b.matrices))

if __name__ == '__main__':
    unittest.main(verbosity=2)

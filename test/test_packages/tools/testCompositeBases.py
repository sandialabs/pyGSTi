from ..testutils import BaseTestCase, compare_files, temp_files
import unittest
import numpy as np
import scipy
import pygsti

import pygsti.tools.basistools       as basistools

class CompositeBasisTestCase(BaseTestCase):

    def test(self):
        std = basistools.basis_matrices('std', [2,2])
        print(std)
        print('_' * 80)
        print(std.get_to_std())
        print(std.get_from_std())

    def test_build_composite_basis(self):
        a = basistools.build_composite_basis([('std', 2), ('std', 2)])
        b = basistools.build_basis('std', [2,2])
        print(a)
        print(b)
        print(a.matrices)
        print(b.matrices)
        1/0


if __name__ == '__main__':
    unittest.main(verbosity=2)

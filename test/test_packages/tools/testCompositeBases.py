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
        raise Exception()

if __name__ == '__main__':
    unittest.main(verbosity=2)

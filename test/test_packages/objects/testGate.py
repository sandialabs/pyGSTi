import unittest
import pygsti
import numpy as np

from  pygsti.objects import Gate

from ..testutils import BaseTestCase, compare_files, temp_files

class GateTestCase(BaseTestCase):

    def setUp(self):
        super(GateTestCase, self).setUp()
        self.gate = Gate(np.zeros(2))

    def test_slice(self):
        self.gate[:]

    def test_bad_mx(self):
        bad_mxs = ['akdjsfaksdf',
                  [[], [1, 2]],
                  [[[]], [[1, 2]]]]
        for bad_mx in bad_mxs:
            with self.assertRaises(ValueError):
                Gate.convert_to_matrix(bad_mx)



if __name__ == '__main__':
    unittest.main(verbosity=2)

from ..testutils import BaseTestCase, compare_files, temp_files
import unittest

import numpy as np
import scipy

from functools import partial

import pygsti

from copy import deepcopy

import pygsti.tools.listtools as lt

class ListToolsBaseTestCase(BaseTestCase):

    def test_all(self):
        l = [1,2,2,3]

        l2 = deepcopy(l)
        lt.remove_duplicates_in_place(l2)
        self.assertEqual(l2, [1,2,3])
        self.assertEqual(l2, lt.remove_duplicates(l))

        letters = list('ABCCA')
        self.assertEqual(lt.compute_occurrence_indices(letters), [0, 0, 0, 1, 1])

        begin = ('A', 'B', 'C')
        self.assertEqual(lt.find_replace_tuple(begin, {'B' : 'C'}), ('A', 'C', 'C'))

if __name__ == '__main__':
    unittest.main(verbosity=2)

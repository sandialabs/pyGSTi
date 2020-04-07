from ..util import BaseCase

from copy import deepcopy

import pygsti.tools.listtools as lt


class ListToolsTester(BaseCase):
    def test_remove_duplicates_in_place(self):
        l = [1, 2, 2, 3]
        l2 = deepcopy(l)
        lt.remove_duplicates_in_place(l2)
        self.assertEqual(l2, [1, 2, 3])
        self.assertEqual(l2, lt.remove_duplicates(l))

    def test_compute_occurrence_indices(self):
        letters = list('ABCCA')
        self.assertEqual(lt.compute_occurrence_indices(letters), [0, 0, 0, 1, 1])

    def test_find_replace_tuple(self):
        begin = ('A', 'B', 'C')
        self.assertEqual(lt.find_replace_tuple(begin, {'B': 'C'}), ('A', 'C', 'C'))

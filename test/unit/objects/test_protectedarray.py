import numpy as np

from ..util import BaseCase

import pygsti.objects.protectedarray as pa


class ProtectedArrayTester(BaseCase):
    # TODO actually test functionality?
    def test_construction(self):
        pa1 = pa.ProtectedArray(np.zeros((3, 3), 'd'))  # nothing protected
        pa2 = pa.ProtectedArray(np.zeros((3, 3), 'd'), 0)
        # protect first row (index 0 in 1st dimension) but no cols - so nothing protected
        pa3 = pa.ProtectedArray(np.zeros((3, 3), 'd'), (0, 0))  # protect (0,0) element
        pa4 = pa.ProtectedArray(np.zeros((3, 3), 'd'), (0, slice(None, None, None)))
        #protect first row
        # TODO assert correctness

        pa5 = pa.ProtectedArray(np.zeros((3, 3), 'd'), (0, [0, 1]))
        #protect (0,0) and (0,1) elements

        s1 = pa5[0, :]  # slice s1 should have first two elements protected:
        self.assertEqual(s1.indicesToProtect, ([0, 1],))

    def test_raises_on_index_out_of_range(self):
        pa5 = pa.ProtectedArray(np.zeros((3, 3), 'd'), (0, [0, 1]))
        with self.assertRaises(IndexError):
            pa5[10, 0] = 4

    def test_raises_on_bad_index_type(self):
        pa5 = pa.ProtectedArray(np.zeros((3, 3), 'd'), (0, [0, 1]))
        with self.assertRaises(TypeError):
            pa5["str"] = 4

    def test_raises_on_construct_index_out_of_range(self):
        with self.assertRaises(IndexError):
            pa.ProtectedArray(np.zeros((3, 3), 'd'), (0, 10))

    def test_raises_on_construct_bad_index_type(self):
        with self.assertRaises(TypeError):
            pa.ProtectedArray(np.zeros((3, 3), 'd'), (0, "str"))

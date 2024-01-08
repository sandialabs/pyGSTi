import numpy as np
import pygsti.baseobjs.protectedarray as pa

from ..util import BaseCase


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

        pa5 = pa.ProtectedArray(np.zeros((3, 3), 'd'), ((0,0), (0, 1)))
        #protect (0,0) and (0,1) elements

        s1 = pa5[0, :]  # slice s1 should have first two elements protected:
        self.assertTrue(np.all(s1.protected_index_mask == np.array([1, 1, 0])))

    def test_construction_from_mask_and_invalid_set(self):
        mask = np.eye(2, dtype=np.bool_)
        pa1 = pa.ProtectedArray(np.zeros((3,3)), protected_index_mask= mask)
        #check that accessing a protected element of this raises an
        #exception
        
        with self.assertRaises(ValueError):
            pa1[0,0] = 1
        
    def test_raises_on_index_out_of_range(self):
        pa5 = pa.ProtectedArray(np.zeros((3, 3), 'd'), (0, [0, 1]))
        with self.assertRaises(IndexError):
            pa5[10, 0] = 4

    def test_raises_on_bad_index_type(self):
        pa5 = pa.ProtectedArray(np.zeros((3, 3), 'd'), (0, [0, 1]))
        with self.assertRaises(IndexError):
            pa5["str"] = 4

    def test_raises_on_construct_index_out_of_range(self):
        with self.assertRaises(IndexError):
            pa.ProtectedArray(np.zeros((3, 3), 'd'), (0, 10))

    def test_raises_on_construct_bad_index_type(self):
        with self.assertRaises(IndexError):
            pa.ProtectedArray(np.zeros((3, 3), 'd'), (0, "str"))

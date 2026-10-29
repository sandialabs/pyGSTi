import copy
import pickle

import numpy as np
import pygsti.baseobjs.protectedarray as pa

from ..util import BaseCase


class ProtectedArrayTester(BaseCase):
    # TODO actually test functionality?
    def test_construction(self):
        pa1 = pa.ProtectedArray(np.zeros((3, 3), 'd'))  # nothing protected
        pa1[0, 0] = 5

        pa3 = pa.ProtectedArray(np.zeros((3, 3), 'd'), (0, 0))  # protect (0,0) element
        with self.assertRaises(ValueError):
            pa3[0, 0] = 5.0

        pa4 = pa.ProtectedArray(np.zeros((3, 3), 'd'), (0, slice(None, None, None)))
        #protect first row
        for i in range(3):
            with self.assertRaises(ValueError):
                pa4[0, i] = 100
            pa4[1,i] = 1

        pa5 = pa.ProtectedArray(np.zeros((3, 3), 'd'), ((0,0), (0, 1)))
        with self.assertRaises(ValueError):
            pa5[0, 0] = 1
        with self.assertRaises(ValueError):
            pa5[0, 1] = 1
        #protect (0,0) and (0,1) elements

        s1 = pa5[0, :]  # slice s1 should have first two elements protected:
        self.assertTrue(np.all(s1.protected_index_mask == np.array([1, 1, 0])))

    def test_construction_matrix_but_only_indicate_a_row_to_protect(self):
        with self.assertWarns(RuntimeWarning):
            pa2 = pa.ProtectedArray(np.zeros((3, 3), 'd'), 0)
        # protect first row (index 0 in 1st dimension) but no cols - so nothing protected
        pa2[0, 0] = 5

    def test_construction_from_mask_and_invalid_set(self):
        mask = np.eye(3, dtype=np.bool_)
        pa1 = pa.ProtectedArray(np.zeros((3,3)), protected_index_mask= mask)
        #check that accessing a protected element of this raises an
        #exception
        
        with self.assertRaises(ValueError):
            pa1[0,0] = 1
        
    def test_raises_on_index_out_of_range(self):
        pa5 = pa.ProtectedArray(np.zeros((3, 3), 'd'), ([0, 1]))
        with self.assertRaises(IndexError):
            pa5[10, 0] = 4

    def test_raises_on_bad_index_type(self):
        pa5 = pa.ProtectedArray(np.zeros((3, 3), 'd'), ([0, 1]))
        with self.assertRaises(IndexError):
            pa5["str"] = 4

    def test_raises_on_construct_index_out_of_range(self):
        with self.assertRaises(IndexError):
            pa.ProtectedArray(np.zeros((3, 3), 'd'), ([0, 10],))

    def test_raises_on_construct_bad_index_type(self):
        with self.assertRaises(IndexError):
            pa.ProtectedArray(np.zeros((3, 3), 'd'), ([0, "str"],))

    def test_raises_on_iadd(self):
        pa1 = pa.ProtectedArray(np.zeros(3, 'd'), [0])

        with self.assertRaises(ValueError):
            pa1 += 3

    def test_raises_on_imul(self):
        pa1 = pa.ProtectedArray(np.zeros(3, 'd'), [0])

        with self.assertRaises(ValueError):
            pa1 *= 3

    def test_raises_on_idiv(self):
        pa1 = pa.ProtectedArray(np.zeros(3, 'd'), [0])

        with self.assertRaises(ValueError):
            pa1.__ifloordiv__(3)

        with self.assertRaises(ValueError):
            pa1.__itruediv__(3)

    def test_raises_on_isub(self):
        pa1 = pa.ProtectedArray(np.zeros(3, 'd'), [0])

        with self.assertRaises(ValueError):
            pa1 -= 3

    def test_raises_on_ipow(self):
        pa1 = pa.ProtectedArray(np.zeros(3, 'd'), [0])

        with self.assertRaises(ValueError):
            pa1 **= 3

    def test_raises_on_imod(self):
        pa1 = pa.ProtectedArray(np.zeros(3, 'd'), [0])

        with self.assertRaises(ValueError):
            pa1 %= 3

    def test_raises_on_iand(self):
        pa1 = pa.ProtectedArray(np.zeros(3, 'i'), [0])

        with self.assertRaises(ValueError):
            pa1 &= 3

    def test_raises_on_ior(self):
        pa1 = pa.ProtectedArray(np.zeros(3, 'i'), [0])

        with self.assertRaises(ValueError):
            pa1 |= 3

    def test_raises_on_ixor(self):
        pa1 = pa.ProtectedArray(np.zeros(3, 'i'), [0])

        with self.assertRaises(ValueError):
            pa1 ^= 3

    def test_raises_on_ilshift(self):
        pa1 = pa.ProtectedArray(np.ones(3, 'i'), [0])

        with self.assertRaises(ValueError):
            pa1 <<= 1

    def test_raises_on_irshift(self):
        pa1 = pa.ProtectedArray(np.ones(3, 'i'), [0])

        with self.assertRaises(ValueError):
            pa1 >>= 1

    def test_raises_on_imatmul(self):
        pa1 = pa.ProtectedArray(np.ones((3,3), 'i'), [0,0])

        with self.assertRaises(ValueError):
            pa1 @= pa1

    def test_blocked_write_leaves_memory_unchanged(self):
        # Demonstrate blocked memory access in action: a rejected assignment to a
        # protected element must NOT modify the underlying array, while writes to
        # unprotected elements go through normally.
        arr = np.arange(9, dtype='d').reshape((3, 3))
        original = arr.copy()
        parr = pa.ProtectedArray(arr, (1, 1))  # protect the center element (1,1)

        # A write to the protected element is blocked...
        with self.assertRaises(ValueError):
            parr[1, 1] = 999.0
        # ...and crucially leaves the memory (both the view and the base) untouched.
        self.assertEqual(parr[1, 1], original[1, 1])
        self.assertEqual(parr.base[1, 1], original[1, 1])
        self.assertTrue(np.array_equal(parr.base, original))

        # Unprotected elements remain writable and the write actually lands.
        parr[0, 0] = 42.0
        parr[2, 2] = -7.0
        self.assertEqual(parr.base[0, 0], 42.0)
        self.assertEqual(parr.base[2, 2], -7.0)
        # The protected element is still at its original value after other writes.
        self.assertEqual(parr.base[1, 1], original[1, 1])

    def test_pickle_roundtrip_preserves_data_and_protection(self):
        # A ProtectedArray should survive a pickle round-trip with its data,
        # protection mask, and read-only behavior intact.
        arr = np.arange(9, dtype='d').reshape((3, 3))
        parr = pa.ProtectedArray(arr, (1, 1))  # protect the center element

        restored = pickle.loads(pickle.dumps(parr))

        # Data and mask are preserved...
        self.assertTrue(np.array_equal(restored.base, parr.base))
        self.assertTrue(np.array_equal(restored.protected_index_mask,
                                       parr.protected_index_mask))
        # ...and so is the protection: the center is still blocked.
        with self.assertRaises(ValueError):
            restored[1, 1] = 999.0
        # Unprotected elements remain writable in the restored copy.
        restored[0, 0] = 42.0
        self.assertEqual(restored.base[0, 0], 42.0)

    def test_deepcopy_preserves_data_and_protection_and_is_independent(self):
        # deepcopy must produce an independent object that keeps the same data
        # and protection behavior, without aliasing the original's memory.
        arr = np.arange(9, dtype='d').reshape((3, 3))
        parr = pa.ProtectedArray(arr, (1, 1))  # protect the center element

        clone = copy.deepcopy(parr)

        # Same contents...
        self.assertTrue(np.array_equal(clone.base, parr.base))
        self.assertTrue(np.array_equal(clone.protected_index_mask,
                                       parr.protected_index_mask))
        # ...but independent memory.
        self.assertIsNot(clone.base, parr.base)
        self.assertIsNot(clone.protected_index_mask, parr.protected_index_mask)

        # Protection is preserved on the clone.
        with self.assertRaises(ValueError):
            clone[1, 1] = 999.0

        # Mutating an unprotected element of the clone does not affect the original.
        clone[0, 0] = 42.0
        self.assertEqual(clone.base[0, 0], 42.0)
        self.assertEqual(parr.base[0, 0], 0.0)

    def test_blocked_row_write_leaves_memory_unchanged(self):
        # A protected row: writes anywhere in that row are blocked and leave the
        # memory unchanged, while the rest of the array stays writable.
        arr = np.zeros((3, 3), 'd')
        parr = pa.ProtectedArray(arr, (0, slice(None, None, None)))  # protect row 0

        for col in range(3):
            with self.assertRaises(ValueError):
                parr[0, col] = 100.0
        # Entire protected row is still zeros - no partial writes leaked through.
        self.assertTrue(np.array_equal(parr.base[0, :], np.zeros(3)))

        # Writes to unprotected rows succeed.
        parr[1, :] = 5.0
        parr[2, 0] = 9.0
        self.assertTrue(np.array_equal(parr.base[1, :], np.full(3, 5.0)))
        self.assertEqual(parr.base[2, 0], 9.0)
        self.assertTrue(np.array_equal(parr.base[0, :], np.zeros(3)))

    def test_construction_mixed_top_level_int_and_tuple(self):
        # A top-level list mixing a full index spec (a tuple) with a bare int:
        # each is normalized independently and both should be protected.
        parr = pa.ProtectedArray(np.zeros(3, 'd'), [(0,), 2])
        self.assertTrue(np.array_equal(parr.protected_index_mask,
                                       np.array([1, 0, 1], dtype=bool)))
        for i in (0, 2):
            with self.assertRaises(ValueError):
                parr[i] = 9.0
        # The un-listed middle element remains writable.
        parr[1] = 5.0
        self.assertEqual(parr.base[1], 5.0)

    def test_raises_typeerror_on_bad_indices_to_protect_type(self):
        # `indices_to_protect` must be an int or a (nested) sequence of ints/slices.
        with self.assertRaises(TypeError):
            pa.ProtectedArray(np.zeros(3, 'd'), 3.5)

    def test_getslice_returns_protectedarray(self):
        # __getslice__ (legacy slicing hook) delegates to __getitem__ and returns
        # a ProtectedArray view over the requested range.
        parr = pa.ProtectedArray(np.arange(5, dtype='d'), (0,))
        sl = parr.__getslice__(1, 4)
        self.assertIsInstance(sl, pa.ProtectedArray)
        self.assertTrue(np.array_equal(sl.base, np.array([1., 2., 3.])))
        # None of elements 1..3 were protected, so the slice is fully writable.
        sl[0] = 99.0
        self.assertEqual(sl.base[0], 99.0)

    def test_getitem_scalar_returns_raw_value(self):
        # Indexing down to a scalar returns the raw numpy scalar, not a
        # ProtectedArray.
        parr = pa.ProtectedArray(np.arange(3, dtype='d'))
        val = parr[1]
        self.assertTrue(np.isscalar(val))
        self.assertEqual(val, 1.0)

    def test_getitem_fully_protected_subarray_is_readonly(self):
        # When every element of a selected subarray is protected, __getitem__
        # returns a fully read-only ProtectedArray.
        parr = pa.ProtectedArray(np.zeros((3, 3), 'd'), (0, slice(None, None, None)))
        sub = parr[0, :]  # entire protected row -> all selected indices masked
        self.assertIsInstance(sub, pa.ProtectedArray)
        self.assertTrue(np.all(sub.protected_index_mask))
        self.assertFalse(sub.base.flags.writeable)
        with self.assertRaises(ValueError):
            sub[0] = 1.0

    def test_getattr_delegates_to_underlying_array(self):
        # Attribute access falls through to the wrapped ndarray.
        parr = pa.ProtectedArray(np.arange(6, dtype='d').reshape((2, 3)))
        self.assertEqual(parr.shape, (2, 3))
        self.assertEqual(parr.dtype, np.dtype('d'))
        self.assertEqual(parr.sum(), 15.0)

    def test_getattr_view_of_base_is_returned_readonly(self):
        # When an attribute returns an ndarray that is a *view* aliasing our base
        # memory (e.g. `.T` on an array that owns its data), __getattr__ must hand
        # back an independent, read-only copy so callers can't bypass protection
        # by mutating the view.
        owner = np.zeros((2, 3), 'd')  # a true data owner (owner.base is None)
        parr = pa.ProtectedArray(owner)

        view = parr.T  # ret.base is owner is parr.base -> copy branch fires
        self.assertIsInstance(view, np.ndarray)
        self.assertFalse(view.flags.writeable)   # returned copy is read-only
        self.assertTrue(view.flags.owndata)      # and owns its own memory
        with self.assertRaises(ValueError):
            view[0, 0] = 1.0  # read-only: cannot mutate
        # Mutating attempt did not touch the original memory.
        self.assertEqual(owner[0, 0], 0.0)

    def test_repr_matches_base_array(self):
        arr = np.arange(4, dtype='d')
        parr = pa.ProtectedArray(arr)
        self.assertEqual(repr(parr), np.array2string(arr))
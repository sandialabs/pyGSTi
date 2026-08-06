import numpy as np
import warnings
from pygsti.tools import group
from pygsti.tools.exceptions import pyGSTiDeprecationWarning
from ..util import BaseCase


class MatrixGroupTester(BaseCase):

    def test_construct_1q_clifford_group(self):
        g = group.construct_1q_clifford_group()
        self.assertEqual(len(g), 24)
        self.assertFalse(-1 in g.product_table)
        self.assertFalse(-1 in g.inverse_table)

        # Latin square property: each row and column contains each element exactly once
        N = len(g)
        for i in range(N):
            self.assertEqual(len(set(g.product_table[i])), N)
            self.assertEqual(len(set(g.product_table[:, i])), N)

        # Inverses property: g * g^-1 = identity
        for i in range(N):
            self.assertEqual(g.product_table[i, g.inverse_table[i]], 0)

    def test_is_integer_deprecated(self):
        # Call directly under catch_warnings to ensure pyGSTiDeprecationWarning is raised
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            res_int = group.is_integer(5)
            res_np_int = group.is_integer(np.int32(5))
            res_float = group.is_integer(5.5)

            self.assertTrue(res_int)
            self.assertTrue(res_np_int)
            self.assertFalse(res_float)

            # Assert at least one warning was emitted and it is pyGSTiDeprecationWarning
            dep_warnings = [warning for warning in w if issubclass(warning.category, pyGSTiDeprecationWarning)]
            self.assertGreaterEqual(len(dep_warnings), 1)

    def test_matrix_group_arguments(self):
        # Test that labels and integer indices both work for accessors
        g = group.construct_1q_clifford_group()
        self.assertIsNotNone(g.labels)
        assert g.labels is not None  # for mypy

        # Try np.int32 and check it works with matrix() without raising KeyError
        idx_np = np.int32(1)
        m1 = g.matrix(idx_np)
        m2 = g.matrix(1)
        self.assertArraysEqual(m1, m2)

        lbl = g.labels[1]
        m3 = g.matrix(lbl)
        self.assertArraysEqual(m1, m3)

        # Check inverse accessors
        inv_idx1 = g.inverse_index(idx_np)
        assert isinstance(inv_idx1, (int, np.integer))
        inv_idx2 = g.inverse_index(1)
        self.assertEqual(inv_idx1, inv_idx2)

        inv_lbl = g.inverse_index(lbl)
        self.assertEqual(inv_lbl, g.labels[inv_idx1])

        # Check product
        prod_idx = g.product([1, 2])
        assert isinstance(prod_idx, (int, np.integer))
        prod_lbl = g.product([g.labels[1], g.labels[2]])
        self.assertEqual(prod_lbl, g.labels[prod_idx])

    def test_matrix_group_coercion_and_asserts(self):
        class MockOp:
            def __init__(self, arr):
                self._arr = arr
            @property
            def shape(self):
                return self._arr.shape
            def to_dense(self):
                return self._arr

        id2 = np.identity(2)
        # Mock operations as objects with to_dense()
        mock_ops = [MockOp(id2)]
        g = group.MatrixGroup(mock_ops)
        self.assertEqual(len(g), 1)
        self.assertArraysEqual(g.matrix(0), id2)

        # Non-identity first should assert failure
        non_id = np.array([[0.0, 1.0], [1.0, 0.0]])
        with self.assertRaises(AssertionError):
            group.MatrixGroup([non_id])

    def test_equivalence_guard(self):
        # Build cyclic rotation group Z_16 (irrational entries)
        n = 16
        th = 2 * np.pi / n
        mxs = [np.array([[np.cos(k * th), -np.sin(k * th)],
                          [np.sin(k * th), np.cos(k * th)]]) for k in range(n)]
        
        # Build reference tables using the exact brute force O(N^3) logic from pre-optimization
        N = len(mxs)
        ref_product_table = -1 * np.ones([N, N], dtype=int)
        for i in range(N):
            for j in range(N):
                ij_product = np.dot(mxs[j], mxs[i])
                for k in range(N):
                    if np.isclose(np.linalg.norm(ij_product - mxs[k]), 0):
                        ref_product_table[i, j] = k
                        break
                        
        ref_inverse_table = -1 * np.ones(N, dtype=int)
        for i in range(N):
            for j in range(N):
                if ref_product_table[i, j] == 0:
                    ref_inverse_table[i] = j
                    break

        # Now build using optimized MatrixGroup
        g = group.MatrixGroup(mxs)
        
        # Assert exact agreement
        self.assertArraysEqual(g.product_table, ref_product_table)
        self.assertArraysEqual(g.inverse_table, ref_inverse_table)

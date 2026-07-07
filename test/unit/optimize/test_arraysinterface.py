"""
Characterization tests for pygsti.optimize.arraysinterface.

Coverage
--------
* UndistributedArraysInterface — every public method, pinned to current
  numeric behaviour so refactors can verify they changed nothing.
* Two xfail tests document *intended* but currently-broken behaviour:
  - norm2_jac should return the *squared* Frobenius norm (not the raw norm).
  - min_x should exist on UndistributedArraysInterface (API parity with Distributed).

DistributedArraysInterface is covered by test/integration/test_optimize_mpi.py
which requires mpi4py and runs under CI.
"""

import numpy as np
import pytest

from pygsti.optimize.arraysinterface import UndistributedArraysInterface
from ..util import BaseCase


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_ari(n_elements=4, n_params=3):
    return UndistributedArraysInterface(n_elements, n_params)


class UndistributedAllocationsTester(BaseCase):
    """Shape and type checks for allocate_* methods."""

    def setUp(self):
        self.ari = make_ari(4, 3)

    def test_allocate_jtf_shape(self):
        arr = self.ari.allocate_jtf()
        self.assertEqual(arr.shape, (3,))
        self.assertEqual(arr.dtype, np.float64)

    def test_allocate_jtj_shape(self):
        arr = self.ari.allocate_jtj()
        self.assertEqual(arr.shape, (3, 3))
        self.assertEqual(arr.dtype, np.float64)

    def test_allocate_jac_shape(self):
        arr = self.ari.allocate_jac()
        self.assertEqual(arr.shape, (4, 3))
        self.assertEqual(arr.dtype, np.float64)

    def test_deallocate_methods_are_noops(self):
        # deallocate_* should return None without raising
        self.assertIsNone(self.ari.deallocate_jtf(self.ari.allocate_jtf()))
        self.assertIsNone(self.ari.deallocate_jtj(self.ari.allocate_jtj()))
        self.assertIsNone(self.ari.deallocate_jac(self.ari.allocate_jac()))

    def test_allocate_jtj_shared_mem_buf_returns_none_tuple(self):
        buf, shm = self.ari.allocate_jtj_shared_mem_buf()
        self.assertIsNone(buf)
        self.assertIsNone(shm)

    def test_deallocate_jtj_shared_mem_buf_is_noop(self):
        self.assertIsNone(self.ari.deallocate_jtj_shared_mem_buf((None, None)))


class UndistributedSlicesAndInfoTester(BaseCase):
    """jac_param_slice, jtf_param_slice, global_num_elements, param_fine_info."""

    def setUp(self):
        self.ari = make_ari(4, 3)

    def test_global_num_elements(self):
        self.assertEqual(self.ari.global_num_elements(), 4)

    def test_jac_param_slice_returns_full_range(self):
        s = self.ari.jac_param_slice()
        self.assertEqual(s, slice(0, 3))

    def test_jac_param_slice_only_if_leader_still_full(self):
        # Serial case: single proc is always the leader
        s = self.ari.jac_param_slice(only_if_leader=True)
        self.assertEqual(s, slice(0, 3))

    def test_jtf_param_slice_returns_full_range(self):
        s = self.ari.jtf_param_slice()
        self.assertEqual(s, slice(0, 3))

    def test_param_fine_info_structure(self):
        slices_by_host, owner_map = self.ari.param_fine_info()
        # One host entry
        self.assertEqual(len(slices_by_host), 1)
        # Each param index maps to (host=0, rank=0)
        for i in range(3):
            self.assertEqual(owner_map[i], (0, 0))


class UndistributedGatherScatterTester(BaseCase):
    """allgather_x, allscatter_x, scatter_x, allgather_f are identity operations."""

    def setUp(self):
        self.ari = make_ari(4, 3)

    def test_allgather_x(self):
        x = np.array([1.0, 2.0, 3.0])
        global_x = np.zeros(3)
        self.ari.allgather_x(x, global_x)
        np.testing.assert_array_equal(global_x, x)

    def test_allscatter_x(self):
        global_x = np.array([1.0, 2.0, 3.0])
        x = np.zeros(3)
        self.ari.allscatter_x(global_x, x)
        np.testing.assert_array_equal(x, global_x)

    def test_scatter_x(self):
        global_x = np.array([1.0, 2.0, 3.0])
        x = np.zeros(3)
        self.ari.scatter_x(global_x, x)
        np.testing.assert_array_equal(x, global_x)

    def test_allgather_f(self):
        f = np.array([4.0, 5.0, 6.0, 7.0])
        global_f = np.zeros(4)
        self.ari.allgather_f(f, global_f)
        np.testing.assert_array_equal(global_f, f)

    def test_gather_jtj_returns_same_array(self):
        jtj = np.eye(3)
        result = self.ari.gather_jtj(jtj)
        np.testing.assert_array_equal(result, jtj)

    def test_gather_jtj_return_shared_mode(self):
        jtj = np.eye(3)
        arr, shm = self.ari.gather_jtj(jtj, return_shared=True)
        np.testing.assert_array_equal(arr, jtj)
        self.assertIsNone(shm)

    def test_scatter_jtj(self):
        global_jtj = np.array([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]])
        jtj = np.zeros((3, 3))
        self.ari.scatter_jtj(global_jtj, jtj)
        np.testing.assert_array_equal(jtj, global_jtj)

    def test_gather_jtf_returns_same_array(self):
        jtf = np.array([1.0, 2.0, 3.0])
        result = self.ari.gather_jtf(jtf)
        np.testing.assert_array_equal(result, jtf)

    def test_gather_jtf_return_shared_mode(self):
        jtf = np.array([1.0, 2.0, 3.0])
        arr, shm = self.ari.gather_jtf(jtf, return_shared=True)
        np.testing.assert_array_equal(arr, jtf)
        self.assertIsNone(shm)

    def test_scatter_jtf(self):
        global_jtf = np.array([10.0, 20.0, 30.0])
        jtf = np.zeros(3)
        self.ari.scatter_jtf(global_jtf, jtf)
        np.testing.assert_array_equal(jtf, global_jtf)


class UndistributedVectorOpsTester(BaseCase):
    """dot_x, norm2_x, infnorm_x, max_x, norm2_f, norm2_jtj."""

    def setUp(self):
        self.ari = make_ari(4, 3)

    def test_dot_x(self):
        x1 = np.array([1.0, 2.0, 3.0])
        x2 = np.array([4.0, 5.0, 6.0])
        self.assertAlmostEqual(self.ari.dot_x(x1, x2), 32.0)

    def test_norm2_x_squared(self):
        x = np.array([3.0, 4.0])
        ari2 = UndistributedArraysInterface(2, 2)
        # norm2_x should return dot(x, x) = 9 + 16 = 25 (squared norm)
        self.assertAlmostEqual(ari2.norm2_x(x), 25.0)

    def test_infnorm_x(self):
        x = np.array([-5.0, 1.0, 3.0])
        self.assertAlmostEqual(self.ari.infnorm_x(x), 5.0)

    def test_max_x(self):
        x = np.array([1.0, 7.0, 3.0])
        self.assertAlmostEqual(self.ari.max_x(x), 7.0)

    def test_norm2_f_squared(self):
        f = np.array([3.0, 0.0, 4.0, 0.0])
        # norm2_f = dot(f,f) = 9+16 = 25 (squared)
        self.assertAlmostEqual(self.ari.norm2_f(f), 25.0)

    def test_norm2_jtj_squared(self):
        jtj = np.array([[1.0, 0.0], [0.0, 1.0]])
        ari2 = UndistributedArraysInterface(2, 2)
        # Frobenius norm of identity = sqrt(2), squared = 2
        self.assertAlmostEqual(ari2.norm2_jtj(jtj), 2.0)


class UndistributedNorm2JacTester(BaseCase):
    """
    Characterisation tests for norm2_jac.

    norm2_jac currently returns np.linalg.norm(j)  (NOT squared).
    This is inconsistent with the docstring, with norm2_f, norm2_x, and
    with DistributedArraysInterface.norm2_jac which returns norm(j)**2.

    The first test PINS THE CURRENT BEHAVIOUR.
    The second test (xfail, strict=True) documents the INTENDED behaviour.
    When the bug is fixed the xfail will flip to passing and should be removed.
    """

    def setUp(self):
        self.ari = make_ari(4, 3)
        # 3×4 Jacobian whose Frobenius norm is 5.0 exactly
        self.j = np.array([
            [3.0, 0.0, 0.0, 0.0],
            [0.0, 4.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ], dtype='d').T  # shape (4, 3)

    def test_norm2_jac_current_behaviour_returns_raw_norm(self):
        """
        Pin current (possibly buggy) behaviour: returns norm(j) = 5.0, not 25.0.

        NOTE: latent discrepancy — UndistributedArraysInterface returns the
        raw Frobenius norm while DistributedArraysInterface returns the squared
        norm.  See test_norm2_jac_intended_behaviour_is_squared (xfail) below.
        """
        result = self.ari.norm2_jac(self.j)
        self.assertAlmostEqual(result, 5.0, places=10)

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "norm2_jac should return the *squared* Frobenius norm (25.0) to "
            "match its docstring, norm2_f/norm2_x, and DistributedArraysInterface. "
            "Currently returns the raw norm (5.0).  Fix during refactor."
        )
    )
    def test_norm2_jac_intended_behaviour_is_squared(self):
        result = self.ari.norm2_jac(self.j)
        self.assertAlmostEqual(result, 25.0, places=10)


class UndistributedMinXTester(BaseCase):
    """
    Characterisation tests for min_x API parity.

    DistributedArraysInterface has a min_x method.
    UndistributedArraysInterface does NOT.  This is an API asymmetry — code
    that calls ari.min_x() will crash in serial but not distributed mode.

    The first test PINS THE CURRENT BEHAVIOUR (AttributeError).
    The second test (xfail, strict=True) documents the INTENDED behaviour
    (min_x returns the minimum element).
    """

    def setUp(self):
        self.ari = make_ari(4, 3)
        self.x = np.array([-1.0, 2.0, 0.5])

    def test_min_x_is_missing_current_behaviour(self):
        """
        Pin current behaviour: min_x does not exist on UndistributedArraysInterface.

        NOTE: latent discrepancy — min_x is present on DistributedArraysInterface
        but absent here.  See test_min_x_intended_behaviour (xfail) below.
        """
        self.assertFalse(hasattr(self.ari, 'min_x'))

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "UndistributedArraysInterface should have a min_x method for API "
            "parity with DistributedArraysInterface.  Currently missing.  "
            "Add during refactor: return np.min(x)."
        )
    )
    def test_min_x_intended_behaviour(self):
        result = self.ari.min_x(self.x)
        self.assertAlmostEqual(result, -1.0, places=10)


class UndistributedFillOpsTester(BaseCase):
    """fill_jtf, fill_jtj, global_svd_dot, fill_dx_svd."""

    def setUp(self):
        self.ari = make_ari(4, 3)

    def test_fill_jtf(self):
        j = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 3.0],
            [1.0, 1.0, 1.0],
        ])
        f = np.array([1.0, 1.0, 1.0, 2.0])
        jtf = np.zeros(3)
        self.ari.fill_jtf(j, f, jtf)
        expected = j.T @ f
        np.testing.assert_allclose(jtf, expected, atol=1e-12)

    def test_fill_jtj(self):
        j = np.arange(12, dtype='d').reshape(4, 3)
        jtj = np.zeros((3, 3))
        self.ari.fill_jtj(j, jtj)
        expected = j.T @ j
        np.testing.assert_allclose(jtj, expected, atol=1e-12)

    def test_fill_jtj_with_shared_mem_buf_ignored(self):
        j = np.arange(12, dtype='d').reshape(4, 3)
        jtj = np.zeros((3, 3))
        self.ari.fill_jtj(j, jtj, shared_mem_buf=(None, None))
        expected = j.T @ j
        np.testing.assert_allclose(jtj, expected, atol=1e-12)

    def test_global_svd_dot(self):
        # jac_v is a (3,3) jtj-type; minus_jtf is a (3,) jtf-type
        jac_v = np.eye(3) * 2.0
        minus_jtf = np.array([1.0, 2.0, 3.0])
        result = self.ari.global_svd_dot(jac_v, minus_jtf)
        expected = jac_v.T @ minus_jtf
        np.testing.assert_allclose(result, expected, atol=1e-12)

    def test_fill_dx_svd(self):
        jac_v = np.eye(3) * 3.0
        global_vec = np.array([1.0, 0.0, -1.0])
        dx = np.zeros(3)
        self.ari.fill_dx_svd(jac_v, global_vec, dx)
        expected = jac_v @ global_vec
        np.testing.assert_allclose(dx, expected, atol=1e-12)


class UndistributedRegularizationTester(BaseCase):
    """jtj_diag_indices, jtj_pre_regularization_data, jtj_update_regularization, jtj_max_diagonal_element."""

    def setUp(self):
        self.ari = make_ari(3, 3)
        self.jtj = np.array([
            [1.0, 0.1, 0.2],
            [0.1, 2.0, 0.3],
            [0.2, 0.3, 3.0],
        ])

    def test_jtj_diag_indices_matches_numpy(self):
        rows, cols = self.ari.jtj_diag_indices(self.jtj)
        expected_rows, expected_cols = np.diag_indices_from(self.jtj)
        np.testing.assert_array_equal(rows, expected_rows)
        np.testing.assert_array_equal(cols, expected_cols)

    def test_jtj_pre_regularization_data_returns_diagonal_copy(self):
        prd = self.ari.jtj_pre_regularization_data(self.jtj)
        np.testing.assert_array_equal(prd, np.diag(self.jtj))
        # Must be a copy, not a view
        prd[0] = 999.0
        self.assertAlmostEqual(self.jtj[0, 0], 1.0)

    def test_jtj_max_diagonal_element(self):
        result = self.ari.jtj_max_diagonal_element(self.jtj)
        self.assertAlmostEqual(result, 3.0)

    def test_jtj_update_regularization(self):
        jtj = self.jtj.copy()
        prd = np.array([1.0, 2.0, 3.0])
        mu = 5.0
        self.ari.jtj_update_regularization(jtj, prd, mu)
        expected_diag = prd + mu
        np.testing.assert_array_equal(np.diag(jtj), expected_diag)
        # Off-diagonals should be unchanged
        np.testing.assert_array_equal(jtj[0, 1], self.jtj[0, 1])
        np.testing.assert_array_equal(jtj[1, 2], self.jtj[1, 2])

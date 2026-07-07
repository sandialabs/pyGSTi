"""
Characterization tests for pygsti.optimize.customsolve.

Serial path only (no mpi4py required here).  The distributed (MPI) path is
covered by test/integration/test_optimize_mpi.py.

When comm is None or the number of processors is below proc_threshold,
custom_solve delegates to scipy.linalg.solve.  We test that this delegation
produces results that are numerically identical to calling scipy directly.
"""

import numpy as np
import scipy.linalg
import pytest

from pygsti.optimize.customsolve import custom_solve
from pygsti.optimize.arraysinterface import UndistributedArraysInterface
from pygsti.baseobjs.resourceallocation import ResourceAllocation
from ..util import BaseCase
from .fixtures import SOLVE_A, SOLVE_B, SOLVE_X_STAR


def _make_resource_alloc():
    """Return a serial (single-proc) ResourceAllocation."""
    return ResourceAllocation()


class CustomSolveSerialTester(BaseCase):
    """custom_solve on the serial path must match scipy.linalg.solve exactly."""

    def setUp(self):
        self.ralloc = _make_resource_alloc()

    def _solve(self, a, b):
        """Helper: call custom_solve and return x."""
        n = a.shape[1]
        ari = UndistributedArraysInterface(n, n)
        x = np.zeros(n, 'd')
        custom_solve(a.copy(), b.copy(), x, ari, self.ralloc, proc_threshold=100)
        return x

    def test_spd_3x3_matches_scipy(self):
        """SPD 3×3 system from fixtures — exact match with scipy."""
        x = self._solve(SOLVE_A, SOLVE_B)
        np.testing.assert_allclose(x, SOLVE_X_STAR, atol=1e-10)

    def test_spd_3x3_residual_is_zero(self):
        """Verify A x = b to machine precision."""
        x = self._solve(SOLVE_A, SOLVE_B)
        residual = SOLVE_A @ x - SOLVE_B
        np.testing.assert_allclose(residual, np.zeros(3), atol=1e-10)

    def test_identity_system(self):
        """A = I  =>  x = b."""
        n = 4
        a = np.eye(n, dtype='d')
        b = np.array([1.0, -2.0, 3.0, 0.5])
        x = self._solve(a, b)
        np.testing.assert_allclose(x, b, atol=1e-12)

    def test_diagonal_system(self):
        """Diagonal A  =>  x[i] = b[i] / a[ii]."""
        d = np.array([2.0, 4.0, 8.0])
        a = np.diag(d)
        b = np.array([1.0, 2.0, 4.0])
        x = self._solve(a, b)
        expected = b / d
        np.testing.assert_allclose(x, expected, atol=1e-12)

    def test_2x2_spd(self):
        """Minimal 2×2 SPD system."""
        a = np.array([[3.0, 1.0], [1.0, 2.0]], dtype='d')
        b = np.array([5.0, 5.0], dtype='d')
        x_ref = scipy.linalg.solve(a, b, assume_a='pos')
        x = self._solve(a, b)
        np.testing.assert_allclose(x, x_ref, atol=1e-12)

    def test_larger_spd_system(self):
        """5×5 random-but-reproducible SPD system."""
        rng = np.random.default_rng(seed=42)
        m = rng.random((5, 5))
        a = m.T @ m + 5.0 * np.eye(5)  # guaranteed SPD
        b = rng.random(5)
        x_ref = scipy.linalg.solve(a, b, assume_a='pos')
        x = self._solve(a, b)
        np.testing.assert_allclose(x, x_ref, atol=1e-10)

    def test_does_not_mutate_a_or_b(self):
        """custom_solve must restore a and b to their input values."""
        a = SOLVE_A.copy()
        b = SOLVE_B.copy()
        a_orig = a.copy()
        b_orig = b.copy()
        n = a.shape[1]
        ari = UndistributedArraysInterface(n, n)
        x = np.zeros(n, 'd')
        custom_solve(a, b, x, ari, self.ralloc, proc_threshold=100)
        np.testing.assert_array_equal(a, a_orig)
        np.testing.assert_array_equal(b, b_orig)

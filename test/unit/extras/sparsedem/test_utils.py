import numpy as np
import pytest
import scipy.linalg
from pygsti.extras.sparsedem.utils import build_masked_hadamard


def test_build_masked_hadamard_full():
    """
    Test that the full Hadamard matrix is returned when row_masks and col_masks
    cover the full range of 2^n.
    """
    # 2-bit Hadamard matrix
    expected = scipy.linalg.hadamard(4)
    row_masks = [0, 1, 2, 3]
    col_masks = [0, 1, 2, 3]
    result = build_masked_hadamard(row_masks, col_masks)
    assert np.array_equal(result, expected)


def test_build_masked_hadamard_partial():
    """
    Test that a submatrix of the Hadamard matrix is returned correctly.
    """
    # 1-bit Hadamard matrix
    expected = scipy.linalg.hadamard(2)
    row_masks = [0, 2]
    col_masks = [0, 2]
    result = build_masked_hadamard(row_masks, col_masks)
    assert np.array_equal(result, expected)


def test_build_masked_hadamard_symmetric_default():
    """
    Test that when col_masks is None, it defaults to row_masks.
    """
    row_masks = [0, 1, 2, 3]
    result = build_masked_hadamard(row_masks)
    expected = build_masked_hadamard(row_masks, row_masks)
    assert np.array_equal(result, expected)

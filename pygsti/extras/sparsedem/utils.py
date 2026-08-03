"""
In-memory utilities used across sparseDEM.

These helpers are pure computations or small data wrangling utilities. Any
parsing/serialization of DEMs or external formats lives in `sparsedem.io`.

A note on convention:
--------------------
Stim samples detectors in increasing index order, but sparsedem represents
bitstrings in decreasing order (the sample array is reversed).

Example: the stim event
    error(0.01) D0 D1 D4
flips detectors 0, 1, and 4. If this is the only event, stim would record:
    [1, 1, 0, 1]
sparseDEM takes as input a dictionary of events keyed to the *reversed* bitstring:
    {'1011': 1}
and may also represent it as:
    integer 11
    list [1, 0, 1, 1]
"""

import numpy as np
import scipy.linalg
from typing import Iterable, Union
from collections import Counter

def counts_from_samples(samples: np.ndarray) -> dict:
    """
    Convert a sample matrix into a Counter-like dict of bitstring keys.

    Parameters:
        samples: np.ndarray
            Sample matrix with rows in {0,1}.

    Returns:
        counts: dict
            Mapping from bitstring keys to counts.
    """
    bitstrings = ["".join(map(str, reversed(row))) for row in samples]
    return Counter(bitstrings)


def counts_to_arrays(counts: dict) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert a Counter-like dict of bitstring keys into aligned arrays.

    Parameters:
        counts: dict
            Mapping from bitstring keys to counts.

    Returns:
        keys: np.ndarray
            Array of bitstrings of shape (K, n).
        values: np.ndarray
            Counts aligned with keys.
    """
    if not counts:
        raise ValueError("counts must be non-empty.")
    keys_list = []
    for key in counts.keys():
        if isinstance(key, str):
            keys_list.append([int(bit) for bit in key])
        else:
            raise TypeError("counts keys must be bitstring strings.")
    keys = np.array(keys_list, dtype=np.uint8)
    values = np.fromiter(counts.values(), dtype=np.int64, count=len(counts))
    return keys, values

def estimate_polarizations(
    counts: dict,
    masks: Union[list[list[int]], np.ndarray],
) -> np.ndarray:
    """
    Compute observed polarizations for specific masks.

    Parameters:
        counts: dict
            Mapping from bitstring keys to counts.
        masks: list[list[int]] or np.ndarray
            Masks specified as rows of {0,1} bits.

    Returns:
        polarizations: np.ndarray
            Polarizations E[(-1)^{s·m}] for each mask.
    """
    if not counts:
        raise ValueError("counts must be non-empty.")
    masks_arr = np.asarray(masks, dtype=np.uint8)
    if masks_arr.ndim == 1:
        masks_arr = masks_arr[None, :]

    keys_list = []
    for key in counts.keys():
        if not isinstance(key, str):
            raise TypeError("counts keys must be bitstring strings.")
        keys_list.append([int(bit) for bit in key])
    samples = np.array(keys_list, dtype=np.uint8)
    values = np.fromiter(counts.values(), dtype=np.int64, count=len(counts))

    if samples.shape[1] != masks_arr.shape[1]:
        raise ValueError("counts and masks must have matching bit-lengths.")

    total = values.sum()
    polarizations = np.zeros(masks_arr.shape[0], dtype=float)
    for i, mask in enumerate(masks_arr):
        parities = parity_dot(samples, mask)
        polarizations[i] = np.sum(values * (1.0 - 2.0 * parities.astype(np.float64))) / total
    return polarizations

def bits_to_binary_number(bits: Iterable[int]) -> int:
    """
    Convert a list of bits (most-significant bit first) into an integer.

    Parameters:
        bits: Iterable[int]
            Bits ordered most-significant bit first.

    Returns:
        value: int
            Integer encoded by the bit list.
    """
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def binary_number_to_bits(integer: int, num_bits: int) -> list[int]:
    """
    Convert an integer to a list of bits (most-significant bit first).

    Parameters:
        integer: int
            Non-negative integer to convert.
        num_bits: int
            Length of the output bit list.

    Returns:
        bits: list[int]
            Bits ordered most-significant bit first.
    """
    if integer >= (1 << num_bits):
        raise ValueError(f"binary representation of {integer} is longer than num_bits")
    if num_bits < 0:
        raise ValueError("num_bits must be non-negative.")
    value = int(integer)
    if value < 0:
        raise ValueError("integer must be non-negative.")
    if num_bits == 0:
        return []
    binary_string = bin(value)[2:].zfill(num_bits)
    if len(binary_string) > num_bits:
        binary_string = binary_string[-num_bits:]
    return [int(bit) for bit in binary_string]


def rows_to_tuples(Y: np.ndarray) -> list[tuple[int, ...]]:
    """
    Convert rows of a {0,1} array into tuples for hashing.

    Parameters:
        Y: np.ndarray
            Sample matrix with rows in {0,1}.

    Returns:
        tuples: list[tuple[int, ...]]
            Row tuples corresponding to Y.
    """
    return [tuple(row.tolist()) for row in Y]


def parity_dot(batch_bits: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Compute (batch_bits @ mask) mod 2 for batch_bits in {0,1}^{m x n}, mask in {0,1}^n.
    Returns {0,1}^m.

    Parameters:
        batch_bits: np.ndarray
            Array of shape (m, n) with entries in {0,1}.
        mask: np.ndarray
            Mask vector of shape (n,) in {0,1}.

    Returns:
        parities: np.ndarray
            Parity values in {0,1} for each row.
    """
    return (batch_bits @ (mask % 2)) % 2

def build_masked_hadamard(row_masks, col_masks=None):
    """
    Build a submatrix of the (unnormalized) Hadamard matrix.

    Parameters:
        row_masks: list[int] or np.ndarray
            Row indices (bitmask integers) to include.
        col_masks: list[int] or np.ndarray, optional
            Column indices (bitmask integers) to include. If None, uses row_masks.

    Returns:
        H_submatrix: np.ndarray
            Submatrix of Hadamard matrix with shape (len(row_masks), len(col_masks))
    """
    if col_masks is None:
        col_masks = row_masks

    largest_mask = max(max(row_masks), max(col_masks))
    n_bits = len(f"{largest_mask:0b}")

    row_mask_bits = [[int(bit) for bit in f"{m:0{n_bits}b}"] for m in row_masks]
    col_mask_bits = [[int(bit) for bit in f"{n:0{n_bits}b}"] for n in col_masks]

    H_submatrix = [
        [(-1) ** np.dot(row, col) for col in col_mask_bits]
        for row in row_mask_bits
    ]

    return np.array(H_submatrix, dtype=int)

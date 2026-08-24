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


def counts_to_detector_arrays(counts: dict) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert a Counter-like dict of bitstring keys into detector-order arrays.

    The keys follow the sparsedem convention (reversed stim rows, so the
    string reads as a binary number whose bit d is detector d); the returned
    sample matrix is back in stim column order: column d is detector d.

    Parameters:
        counts: dict
            Mapping from bitstring keys to counts.

    Returns:
        samples: np.ndarray
            (K, n) uint8 matrix of distinct syndromes, column d = detector d.
        weights: np.ndarray
            Counts aligned with the rows of `samples`.
    """
    keys, values = counts_to_arrays(counts)
    return np.ascontiguousarray(keys[:, ::-1]), values


def pack_detector_samples(samples: np.ndarray) -> np.ndarray:
    """
    Pack a (K, n) {0,1} sample matrix into (K, ceil(n/64)) uint64 words.

    Bit d of a row lives in word d // 64 at bit position d % 64, matching
    the integer-bitmask convention (bit d = detector d).

    Parameters:
        samples: np.ndarray
            (K, n) array with entries in {0,1}, column d = detector d.

    Returns:
        packed: np.ndarray
            (K, ceil(n/64)) uint64 array.
    """
    samples = np.asarray(samples, dtype=np.uint8) % 2
    if samples.ndim != 2:
        raise ValueError("samples must be a 2D (rows, bits) array.")
    n = samples.shape[1]
    n_words = max((n + 63) // 64, 1)
    padded = np.zeros((samples.shape[0], n_words * 64), dtype=np.uint8)
    padded[:, :n] = samples
    # Little-endian within each 64-bit word so bit d sits at position d % 64.
    packed_bytes = np.packbits(padded, axis=1, bitorder="little")
    words = packed_bytes.reshape(samples.shape[0], n_words, 8).astype(np.uint64)
    shifts = (np.uint64(8) * np.arange(8, dtype=np.uint64))
    return (words << shifts).sum(axis=2, dtype=np.uint64)


def masks_to_packed(masks, n_bits: int) -> np.ndarray:
    """
    Pack integer bitmasks (arbitrary-precision Python ints) into uint64 words.

    Parameters:
        masks: Iterable[int]
            Integer bitmasks (bit d = detector d); may exceed 64 bits.
        n_bits: int
            Number of bits spanned by the masks.

    Returns:
        packed: np.ndarray
            (len(masks), ceil(n_bits/64)) uint64 array, aligned with
            `pack_detector_samples`.
    """
    masks = list(masks)
    n_words = max((int(n_bits) + 63) // 64, 1)
    packed = np.zeros((len(masks), n_words), dtype=np.uint64)
    word_mask = (1 << 64) - 1
    for i, m in enumerate(masks):
        m = int(m)
        w = 0
        while m:
            packed[i, w] = np.uint64(m & word_mask)
            m >>= 64
            w += 1
    return packed


def weighted_odd_counts(packed_samples: np.ndarray, weights: np.ndarray,
                        packed_masks: np.ndarray) -> np.ndarray:
    """
    Weighted number of odd-parity rows for each mask, on packed bit data.

    For each mask M, returns sum of weights[r] over rows r with
    popcount(row_r & M) odd. This is the exact integer numerator of a Walsh
    polarization: pol(M) = (total - 2 * odd) / total.

    Parameters:
        packed_samples: np.ndarray
            (K, W) uint64 packed samples (see `pack_detector_samples`).
        weights: np.ndarray
            (K,) integer weights (counts) per row.
        packed_masks: np.ndarray
            (M, W) uint64 packed masks (see `masks_to_packed`).

    Returns:
        odd: np.ndarray
            (M,) int64 weighted odd-parity counts.
    """
    packed_samples = np.asarray(packed_samples, dtype=np.uint64)
    packed_masks = np.asarray(packed_masks, dtype=np.uint64)
    weights = np.asarray(weights, dtype=np.int64)
    out = np.empty(len(packed_masks), dtype=np.int64)
    for i, mask in enumerate(packed_masks):
        parity = np.bitwise_count(packed_samples & mask).sum(
            axis=1, dtype=np.int64) & 1
        out[i] = int(weights[parity.astype(bool)].sum())
    return out


def packed_parity_matrix(packed_row_masks: np.ndarray,
                         packed_col_masks: np.ndarray,
                         chunk: int = 256) -> np.ndarray:
    """
    Parity-of-overlap matrix between two packed mask collections.

    Entry (i, j) is popcount(row_mask_i & col_mask_j) mod 2 -- the exponent
    in the masked Hadamard matrix: H[i, j] = (-1)**parity.

    Parameters:
        packed_row_masks: np.ndarray
            (R, W) uint64 packed masks.
        packed_col_masks: np.ndarray
            (C, W) uint64 packed masks.
        chunk: int
            Row chunk size bounding peak memory.

    Returns:
        parity: np.ndarray
            (R, C) uint8 array in {0, 1}.
    """
    rows = np.asarray(packed_row_masks, dtype=np.uint64)
    cols = np.asarray(packed_col_masks, dtype=np.uint64)
    out = np.empty((rows.shape[0], cols.shape[0]), dtype=np.uint8)
    for i0 in range(0, rows.shape[0], chunk):
        block = rows[i0:i0 + chunk, None, :] & cols[None, :, :]
        out[i0:i0 + chunk] = (np.bitwise_count(block).sum(
            axis=2, dtype=np.int64) & 1).astype(np.uint8)
    return out


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

def masked_hadamard_dot(row_masks, col_masks, vector):
    """
    Compute build_masked_hadamard(row_masks, col_masks) @ vector without
    materializing the (rows x cols) submatrix.

    Parameters:
        row_masks: list[int] or np.ndarray
            Row bitmask integers.
        col_masks: list[int] or np.ndarray
            Column bitmask integers, aligned with vector.
        vector: np.ndarray
            Values to contract against the columns.

    Returns:
        product: np.ndarray
            Length len(row_masks) result of the streamed matrix-vector
            product.
    """
    vector = np.asarray(vector, dtype=float)
    largest_mask = max(max(row_masks), max(col_masks))
    n_bits = len(f"{largest_mask:0b}")
    row_bits = np.array([[int(bit) for bit in f"{m:0{n_bits}b}"]
                         for m in row_masks], dtype=np.int32)
    result = np.zeros(len(row_bits))
    step = max(1, 2 ** 24 // max(len(row_bits), 1))
    for j0 in range(0, len(col_masks), step):
        col_bits = np.array([[int(bit) for bit in f"{m:0{n_bits}b}"]
                             for m in col_masks[j0:j0 + step]], dtype=np.int32)
        signs = 1.0 - 2.0 * ((row_bits @ col_bits.T) & 1)
        result += signs @ vector[j0:j0 + step]
    return result


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

    def _bit_matrix(masks):
        return np.array([[int(bit) for bit in f"{m:0{n_bits}b}"] for m in masks],
                        dtype=np.int32)

    row_bits = _bit_matrix(row_masks)
    col_bits = _bit_matrix(col_masks)

    # (-1)**(row . col) = 1 - 2 * parity(row . col), computed blockwise so the
    # integer product never exceeds ~2**24 elements at a time.
    H_submatrix = np.empty((len(row_bits), len(col_bits)), dtype=np.int8)
    step = max(1, 2 ** 24 // max(len(row_bits), 1))
    for j0 in range(0, len(col_bits), step):
        parity = (row_bits @ col_bits[j0:j0 + step].T) & 1
        H_submatrix[:, j0:j0 + step] = (1 - 2 * parity).astype(np.int8)
    return H_submatrix

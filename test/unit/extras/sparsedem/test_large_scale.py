"""
Tests for the bit-packed / large-detector-count paths of sparsedem.

These cover behavior the original dense implementations could not reach:
DEMs with more than 63 detectors (whose integer event masks overflow int64),
the vectorized lattice-pruning checks, the restricted polarization-mask fit,
and the packed + bit-flip-refined GF(2) solver.
"""

import numpy as np
import pytest

from pygsti.extras.sparsedem import utils as sdutils
from pygsti.extras.sparsedem import io as sdio
from pygsti.extras.sparsedem.estimation import (
    default_polarization_masks,
    fit_specified_dem,
)
from pygsti.extras.sparsedem.lattice import (
    bitmask_trie_search,
    check_event_mask,
    lattice_pruning_dem_estimation,
    make_fast_event_check,
)
from pygsti.extras.sparsedem.logical_decoration import (
    _gf2_eliminate,
    _gf2_eliminate_packed,
    dem_to_check_matrix,
    solve_gf2_robust,
)


def _simulate(masks, probs, n_detectors, n_shots, rng):
    """Fire events independently and XOR their patterns into syndromes."""
    H = np.zeros((n_detectors, len(masks)), dtype=np.uint8)
    for j, mask in enumerate(masks):
        mm = int(mask)
        while mm:
            low = mm & -mm
            H[low.bit_length() - 1, j] = 1
            mm ^= low
    occurrences = (rng.random((n_shots, len(masks))) < probs).astype(np.uint8)
    return ((occurrences @ H.T) % 2).astype(np.uint8)


# ---------------------------------------------------------------------------
# Packing utilities
# ---------------------------------------------------------------------------

def test_pack_and_odd_counts_match_bruteforce():
    rng = np.random.default_rng(5)
    n_bits = 130  # > 2 words
    samples = (rng.random((40, n_bits)) < 0.3).astype(np.uint8)
    weights = rng.integers(1, 5, size=40)
    packed = sdutils.pack_detector_samples(samples)
    assert packed.shape == (40, (n_bits + 63) // 64)

    masks = [1, 1 << 129, (1 << 3) | (1 << 70) | (1 << 129)]
    packed_masks = sdutils.masks_to_packed(masks, n_bits)
    odd = sdutils.weighted_odd_counts(packed, weights, packed_masks)
    for i, m in enumerate(masks):
        bits = [d for d in range(n_bits) if (m >> d) & 1]
        parity = samples[:, bits].sum(axis=1) % 2
        assert odd[i] == weights[parity == 1].sum()


def test_packed_parity_matrix_matches_masked_hadamard():
    masks_a = [1, 3, 6, 12]
    masks_b = [2, 5, 7]
    parity = sdutils.packed_parity_matrix(
        sdutils.masks_to_packed(masks_a, 4),
        sdutils.masks_to_packed(masks_b, 4),
    )
    H = sdutils.build_masked_hadamard(masks_a, masks_b)
    assert np.array_equal(1 - 2 * parity.astype(int), H)


# ---------------------------------------------------------------------------
# Estimation: packed fit path
# ---------------------------------------------------------------------------

def test_fit_specified_dem_packed_matches_legacy_small():
    """On a small problem, the packed path with the legacy all-pairs
    polarization masks reproduces the legacy fit."""
    rng = np.random.default_rng(7)
    n_bits = 6
    true_masks = [1, 2, 4, 8, 16, 32, 3, 12, 48]
    true_probs = np.full(len(true_masks), 0.02)
    samples = _simulate(true_masks, true_probs, n_bits, 30_000, rng)
    counts = dict(sdutils.counts_from_samples(samples))

    legacy_masks, legacy_probs, legacy_cov = fit_specified_dem(
        counts, true_masks, return_probs=True, return_covariance=True)

    all_pairs = set(true_masks)
    all_pairs.update(1 << i for i in range(n_bits))
    all_pairs.update((1 << i) | (1 << j)
                     for i in range(1, n_bits) for j in range(i))
    packed_masks, packed_probs, packed_cov = fit_specified_dem(
        counts, true_masks, return_probs=True, return_covariance=True,
        pol_masks=sorted(all_pairs))

    assert np.array_equal(legacy_masks, packed_masks)
    assert np.allclose(legacy_probs, packed_probs, atol=1e-10)
    assert np.allclose(legacy_cov, packed_cov, atol=1e-12)


def test_fit_specified_dem_large_detector_count():
    """Masks beyond bit 63 fit and are recovered on a 70-detector problem."""
    rng = np.random.default_rng(11)
    n_bits = 70
    true_masks = [1 << d for d in range(0, n_bits, 7)]
    true_masks += [(1 << 65) | (1 << 69), (1 << 2) | (1 << 64)]
    true_probs = np.full(len(true_masks), 0.03)
    samples = _simulate(true_masks, true_probs, n_bits, 40_000, rng)
    counts = dict(sdutils.counts_from_samples(samples))

    with pytest.warns(UserWarning, match="restricted"):
        masks_out, probs_out, cov = fit_specified_dem(
            counts, sorted(true_masks), return_probs=True,
            return_covariance=True)
    fitted = dict(zip([int(m) for m in masks_out], probs_out))
    for m, p in zip(true_masks, true_probs):
        assert fitted[int(m)] == pytest.approx(p, abs=0.01)
    assert cov.shape == (len(true_masks), len(true_masks))
    assert np.all(np.diag(cov) > 0)


# ---------------------------------------------------------------------------
# Lattice pruning: fast checks and end-to-end at large n
# ---------------------------------------------------------------------------

def test_fast_event_check_matches_legacy():
    rng = np.random.default_rng(13)
    n_bits = 8
    true_masks = [1, 2, 3, 24, 129 % 256]
    samples = _simulate(true_masks, [0.05] * len(true_masks), n_bits,
                        5_000, rng)
    counts = dict(sdutils.counts_from_samples(samples))
    fast = make_fast_event_check(counts, alpha=0.01)
    for mask in list(range(1, 64)) + [128, 192]:
        assert fast(mask) == check_event_mask(mask, counts, alpha=0.01), mask


def test_lattice_pruning_large_detector_count():
    rng = np.random.default_rng(17)
    n_bits = 70
    true_masks = sorted(
        [1 << d for d in range(0, n_bits, 5)]
        + [(1 << d) | (1 << (d + 9)) for d in range(0, n_bits - 9, 13)]
        + [(1 << 60) | (1 << 68)]
    )
    true_probs = np.full(len(true_masks), 0.04)
    samples = _simulate(true_masks, true_probs, n_bits, 60_000, rng)
    counts = dict(sdutils.counts_from_samples(samples))

    dem, masks, probs, cov = lattice_pruning_dem_estimation(
        counts, confidence=1 - 1e-7, return_covariance=True)
    learned = sdio.dem_to_dict(dem)
    for m, p in zip(true_masks, true_probs):
        assert int(m) in learned, f"missed event {m:#x}"
        assert learned[int(m)] == pytest.approx(p, abs=0.015)
    spurious = set(learned) - set(int(m) for m in true_masks)
    assert len(spurious) <= 2


def test_trie_search_iterative_matches_semantics():
    """The explicit-stack trie visits exactly the downward-closed extensions."""
    valid = {0b001, 0b010, 0b011, 0b110}

    def check(mask):
        return mask in valid

    found = bitmask_trie_search(3, check)
    assert found == {0b001, 0b010, 0b011, 0b110}


# ---------------------------------------------------------------------------
# Logical decoration: packed GF(2) and bit-flip refinement
# ---------------------------------------------------------------------------

def test_packed_elimination_matches_dense():
    rng = np.random.default_rng(19)
    for _ in range(10):
        m, n = rng.integers(3, 40), rng.integers(2, 30)
        A = (rng.random((m, n)) < 0.4).astype(np.uint8)
        y = (rng.random(m) < 0.5).astype(np.uint8)
        R, z, pivots = _gf2_eliminate(A, y)
        Rp = sdutils.pack_detector_samples(A)
        zp = y.copy()
        pivots_p = _gf2_eliminate_packed(Rp, zp, n)
        assert pivots == pivots_p
        # Unpack Rp and compare with the dense RREF.
        unpacked = np.zeros((m, n), dtype=np.uint8)
        for c in range(n):
            w, b = divmod(c, 64)
            unpacked[:, c] = (Rp[:, w] >> np.uint64(b)) & np.uint64(1)
        assert np.array_equal(unpacked, R)
        assert np.array_equal(zp, z)


def test_bitflip_refinement_recovers_flags_at_high_corruption():
    """At ~10% corrupted rows every elimination pivots on bad rows and
    RANSAC alone cannot recover the flags; bit-flip refinement can."""
    rng = np.random.default_rng(23)
    n_events, n_shots = 60, 6_000
    L_true = rng.integers(0, 2, size=n_events).astype(np.uint8)
    B = (rng.random((n_shots, n_events)) < 0.15).astype(np.uint8)
    B[:n_events] = np.eye(n_events, dtype=np.uint8)  # ensure full rank
    Y = ((B @ L_true.astype(np.int64)) % 2).astype(np.uint8)
    corrupted = rng.choice(n_shots, size=n_shots // 10, replace=False)
    Y[corrupted] ^= 1

    L_plain, res_plain, _ = solve_gf2_robust(
        B, Y, residual_threshold=0.15, max_ransac_iterations=5, seed=1,
        refine=False)
    L_ref, res_ref, diag = solve_gf2_robust(
        B, Y, residual_threshold=0.15, max_ransac_iterations=5, seed=1,
        refine=True)

    assert not np.array_equal(L_plain, L_true)
    assert np.array_equal(L_ref, L_true)
    assert res_ref == pytest.approx(len(corrupted) / n_shots)
    assert res_ref < res_plain
    assert diag["refine_flips"] > 0
    assert diag["converged"]


def test_confidence_row_order_reported():
    B = np.eye(4, dtype=np.uint8)
    Y = np.array([1, 0, 1, 0], dtype=np.uint8)
    _, _, diag = solve_gf2_robust(B, Y, row_order="confidence")
    assert diag["row_order"] == "confidence"
    _, _, diag = solve_gf2_robust(B, Y)
    assert diag["row_order"] == "given"


def test_dem_to_check_matrix_beyond_63_detectors():
    dem = sdio.dem_from_str("""
        error(0.01) D0 D1
        error(0.02) D64 D80
        error(0.03) D99
    """)
    H, probs, masks = dem_to_check_matrix(dem)
    assert H.shape == (100, 3)
    assert masks.dtype == object
    assert int(masks[-1]) == 1 << 99
    assert H[64, list(masks).index((1 << 64) | (1 << 80))] == 1
    assert H.sum() == 5


def test_default_polarization_masks_contents():
    dem_masks = [0b11, 0b1100, 0b10000]
    pol = default_polarization_masks(dem_masks, 5)
    for i in range(5):
        assert (1 << i) in pol
    assert 0b11 in pol and 0b1100 in pol and 0b10000 in pol
    assert sorted(pol) == pol

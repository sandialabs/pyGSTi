import random

import numpy as np
import stim
from collections import Counter

from pygsti.extras.sparsedem.compressed_sensing import (
    CSConfig,
    _all_low_weight_masks,
    _random_low_weight_masks,
    estimate_sparse_wh,
)
from pygsti.extras.sparsedem.io import dem_from_str, dem_to_dict


def test_random_low_weight_masks_respects_weight_and_uniqueness():
    rng = random.Random(0)
    masks = _random_low_weight_masks(n_bits=4, weight=2, num=5, rng=rng)
    assert len(masks) == 5
    assert len(set(masks)) == 5
    for m in masks:
        assert bin(m).count("1") == 2


def test_all_low_weight_masks_enumeration():
    masks = _all_low_weight_masks(n_bits=3, max_weight=2)
    # Weight 1: 100, 010, 001; Weight 2: 110, 101, 011
    expected = [0b100, 0b010, 0b001, 0b110, 0b101, 0b011]
    assert masks == expected


def test_estimate_sparse_wh_trivial_zero_polarizations():
    # All syndromes are 00, so polarizations are 1 and depolarizations are 0.
    counts = {"00": 50}
    probs, atts, candidate_masks, used_masks = estimate_sparse_wh(
        counts,
        CSConfig(max_weight=2, budget=3, l1_penalty=1e-3, seed=1),
    )
    assert len(candidate_masks) == 3  # weight-1 and weight-2 for 2 bits
    assert len(used_masks) <= 3
    np.testing.assert_allclose(probs, 0.0, atol=1e-8)
    np.testing.assert_allclose(atts, 0.0, atol=1e-8)


def test_budget_and_sample_per_weight_limit_measurements():
    counts = {"000": 10}
    cfg = CSConfig(max_weight=3, budget=10, sample_per_weight=1, seed=2)
    _, _, _, used_masks = estimate_sparse_wh(counts, cfg)
    # At most one mask per weight up to max_weight
    assert len(used_masks) <= 3
    for m in used_masks:
        assert bin(m).count("1") <= cfg.max_weight


def test_estimate_sparse_wh_recovers_simple_dem():
    dem_str = """
    error(0.05) D0
    error(0.02) D1
    error(0.01) D0 D1
    """
    dem = dem_from_str(dem_str)
    sampler = dem.compile_sampler()
    n_shots = 2**16
    samples = np.array(sampler.sample(n_shots)[0], dtype=int)
    bitstrings = [''.join(map(str, reversed(row))) for row in samples]
    counts = Counter(bitstrings)

    cfg = CSConfig(max_weight=2, budget=20, l1_penalty=1e-4, seed=123)
    probs, _, candidate_masks, _ = estimate_sparse_wh(counts, cfg)

    true_dict = dem_to_dict(dem)
    for mask, est_p in zip(candidate_masks, probs):
        expected = true_dict.get(mask, 0.0)
        assert np.isclose(est_p, expected, atol=0.015)

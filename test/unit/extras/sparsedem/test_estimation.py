import numpy as np
import pytest
import stim

from pygsti.extras.sparsedem.io import dem_from_str, dem_to_dict, dem_to_event_probabilities
from pygsti.extras.sparsedem.estimation import (
    dense_dem_estimation,
    estimate_dem_and_covariance,
    fit_specified_dem,
    threshold_probabilities,
    compute_outcome_distribution_from_dem,
    compute_outcome_distribution_from_high_rank_dem,
)


def test_dense_dem_estimation():
    # Generate a small DEM with known probabilities
    p0 = np.random.random() / 10
    p1 = np.random.random() / 10
    p01 = np.random.random() / 10
    dem_str = f"""
    error({p0}) D0
    error({p1}) D1
    error({p01}) D0 D1
    """
    dem = dem_from_str(dem_str)

    # Compute exact outcome distribution
    probs = compute_outcome_distribution_from_dem(dem)
    prob_dict = {f"{i:02b}": p for i, p in enumerate(probs)}

    # Estimate event probabilities from outcome distribution
    estimated = dense_dem_estimation(prob_dict)
    expected = np.array([0, p0, p1, p01])

    assert np.allclose(estimated, expected, atol=1e-6)


def test_estimate_dem_and_covariance():
    # Create a DEM and sample from it
    dem_str = """
    error(0.01) D0 D1
    error(0.02) D1 D2
    error(0.03) D2 D3
    error(0.005) D0 D2 D4
    """
    dem = dem_from_str(dem_str)
    sampler = dem.compile_sampler()
    n_shots = 2**14
    samples = np.array(sampler.sample(n_shots)[0], dtype=int)
    bitstrings = [''.join(map(str, reversed(row))) for row in samples]
    syndrome_counts = dict()
    for s in bitstrings:
        syndrome_counts[s] = syndrome_counts.get(s, 0) + 1

    # Estimate probabilities and covariance
    estimated_probs, cov = estimate_dem_and_covariance(syndrome_counts)
    assert estimated_probs.shape[0] == cov.shape[0] == cov.shape[1]
    assert np.all(np.diag(cov) >= 0)


def test_threshold_probabilities():
    # Create synthetic data
    n = 4
    estimated = np.array([0.0, 0.01, 0.02, 0.0])
    cov = np.diag([1e-6, 1e-6, 1e-6, 1e-6])

    thresholded, mask = threshold_probabilities(estimated, cov, alpha=0.05)

    assert np.all(thresholded[mask] > 0)
    assert np.all(thresholded[~mask] == 0)


def test_compute_outcome_distribution_from_dem():
    # Use known probabilities to compute expected outcome distribution
    p0 = np.random.random() / 10
    p1 = np.random.random() / 10
    p01 = np.random.random() / 10

    P00 = (1 - p0) * (1 - p1) * (1 - p01) + p0 * p1 * p01
    P01 = p0 * (1 - p1) * (1 - p01) + (1 - p0) * p1 * p01
    P10 = (1 - p0) * p1 * (1 - p01) + p0 * (1 - p1) * p01
    P11 = p0 * p1 * (1 - p01) + (1 - p0) * (1 - p1) * p01

    dem_str = f"""
    error({p0}) D0
    error({p1}) D1
    error({p01}) D0 D1
    """
    dem = dem_from_str(dem_str)
    probs = compute_outcome_distribution_from_dem(dem)

    assert np.allclose(probs, [P00, P01, P10, P11], atol=1e-6)


def test_fit_specified_dem_covariance():
    dem_str = """
    error(0.02) D0
    error(0.015) D1
    error(0.01) D2
    error(0.005) D0 D1
    """
    dem = dem_from_str(dem_str)
    masks = sorted(dem_to_dict(dem).keys())

    n_shots = 2**12
    sampler = dem.compile_sampler()
    samples = np.array(sampler.sample(n_shots)[0], dtype=int)
    bitstrings = [''.join(map(str, reversed(row))) for row in samples]
    syndrome_counts = {}
    for s in bitstrings:
        syndrome_counts[s] = syndrome_counts.get(s, 0) + 1

    dem_masks, probs, cov = fit_specified_dem(
        syndrome_counts,
        masks,
        return_probs=True,
        return_covariance=True,
    )

    assert np.array_equal(dem_masks, np.array(masks))
    assert probs.shape[0] == len(masks)
    assert cov.shape == (len(masks), len(masks))
    assert np.all(np.diag(cov) >= 0)


def _brute_force_block_distribution(blocks, n):
    """Enumerate all block-branch combinations to get the exact distribution."""
    import itertools

    probs = np.zeros(2 ** n)
    for combo in itertools.product(*(range(len(b) + 1) for b in blocks)):
        p = 1.0
        outcome = 0
        for choice, block in zip(combo, blocks):
            if choice < len(block):
                branch_p, dets = block[choice]
                p *= branch_p
                for d in dets:
                    outcome ^= 1 << d
            else:
                p *= 1 - sum(branch_p for branch_p, _ in block)
        probs[outcome] += p
    return probs


def test_high_rank_distribution_matches_independent_case():
    dem = dem_from_str("""
    error(0.01) D0
    error(0.02) D1
    error(0.03) D0 D1 D2
    """)
    probs = compute_outcome_distribution_from_high_rank_dem(dem)
    expected = compute_outcome_distribution_from_dem(dem)
    assert np.allclose(probs, expected)


def test_high_rank_distribution_exclusive_blocks():
    blocks = [
        [(0.01, (0, 1))],                       # independent event
        [(0.05, (0,)), (0.07, (1, 2))],         # rank-3 exclusive event
        [(0.1, (2,)), (0.2, (0, 2)), (0.15, (1,))],  # rank-4 exclusive event
    ]
    probs = compute_outcome_distribution_from_high_rank_dem(blocks, num_detectors=3)
    expected = _brute_force_block_distribution(blocks, 3)
    assert np.allclose(probs, expected)
    assert probs.sum() == pytest.approx(1.0)


def test_high_rank_distribution_exclusivity_zeroes_outcomes():
    # Only event source is one exclusive block, so D0 and D1 can never both fire.
    blocks = [[(0.3, (0,)), (0.15, (1,))]]
    probs = compute_outcome_distribution_from_high_rank_dem(blocks, num_detectors=2)
    assert probs[0b01] == pytest.approx(0.3)
    assert probs[0b10] == pytest.approx(0.15)
    assert probs[0b11] == pytest.approx(0.0, abs=1e-12)
    assert probs[0b00] == pytest.approx(0.55)


def test_high_rank_distribution_rejects_odd_overlap_mass_over_half():
    # The log-space (attenuation) pipeline requires q(s) < 1/2 for every
    # block and parity mask.
    blocks = [[(0.6, (0,)), (0.3, (1,))]]
    with pytest.raises(ValueError, match="q\\(s\\)"):
        compute_outcome_distribution_from_high_rank_dem(blocks, num_detectors=2)


def test_high_rank_distribution_rejects_bad_blocks():
    with pytest.raises(ValueError):
        compute_outcome_distribution_from_high_rank_dem(
            [[(0.7, (0,)), (0.4, (1,))]], num_detectors=2
        )

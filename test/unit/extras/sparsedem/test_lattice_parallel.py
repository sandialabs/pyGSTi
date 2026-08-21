import numpy as np
import pytest
import stim

from pygsti.extras.sparsedem.lattice import (
    bitmask_frontier_search,
    bitmask_trie_search,
    check_event_mask,
    check_event_masks_batch,
    lattice_pruning_dem_estimation,
)
from pygsti.extras.sparsedem.estimation import (
    estimate_dem_and_covariance,
    fit_specified_dem,
)
from pygsti.extras.sparsedem.lattice import marginalize_syndrome_counts
from pygsti.extras.sparsedem.io import dem_to_dict
from pygsti.extras.sparsedem.utils import counts_from_samples, counts_to_arrays
from pygsti.extras.sparsedem.validation import sample_dem


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def synthetic_counts(seed, num_detectors=6, num_shots=20000):
    """Sample syndrome counts from a random small graph-like DEM."""
    rng = np.random.default_rng(seed)
    lines = []
    for d in range(num_detectors):
        lines.append(f"error({rng.uniform(0.02, 0.10):.4f}) D{d}")
    for d in range(num_detectors - 1):
        lines.append(f"error({rng.uniform(0.02, 0.08):.4f}) D{d} D{d + 1}")
    dem = stim.DetectorErrorModel("\n".join(lines))
    det, _ = sample_dem(dem, num_shots, seed=seed)
    return counts_from_samples(det)


def sequential_z(mask, syndrome_counts):
    """The z statistic exactly as check_event_mask computes it."""
    n_bits = len(next(iter(syndrome_counts)))
    bitmask = f"{mask:0{n_bits}b}"
    sub = marginalize_syndrome_counts(syndrome_counts, bitmask)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        probs, cov = estimate_dem_and_covariance(sub)
        return probs[-1] / np.sqrt(cov[-1, -1])


# ---------------------------------------------------------------------------
# Batch checker vs sequential checker
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed", [0, 1, 2])
def test_batch_checker_matches_sequential(seed):
    counts = synthetic_counts(seed)
    keys, values = counts_to_arrays(counts)
    n = keys.shape[1]
    for k in range(1, n + 1):
        masks = [m for m in range(1, 2 ** n) if m.bit_count() == k]
        passed, z = check_event_masks_batch(masks, keys, values.astype(float),
                                            alpha=0.05)
        for m, ok, z_m in zip(masks, passed, z):
            assert ok == check_event_mask(m, counts, alpha=0.05), f"mask {m}"
            z_ref = sequential_z(m, counts)
            if np.isfinite(z_ref) and np.isfinite(z_m):
                assert abs(z_m - z_ref) < 1e-8 * max(1.0, abs(z_ref)), f"mask {m}"
            else:
                assert not (np.isfinite(z_ref) or np.isfinite(z_m)), f"mask {m}"


def test_batch_checker_input_validation():
    counts = synthetic_counts(3)
    keys, values = counts_to_arrays(counts)
    passed, z = check_event_masks_batch([], keys, values)
    assert passed.size == 0 and z.size == 0
    with pytest.raises(ValueError):
        check_event_masks_batch([1, 3], keys, values)  # mixed weights
    with pytest.raises(ValueError):
        check_event_masks_batch([0], keys, values)  # weight 0


# ---------------------------------------------------------------------------
# Frontier search vs trie search
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_frontier_search_equals_trie_search(seed):
    counts = synthetic_counts(seed)
    keys, values = counts_to_arrays(counts)
    n = keys.shape[1]
    trie = bitmask_trie_search(
        n, lambda m: check_event_mask(m, counts, alpha=0.05))

    def checker(masks):
        return check_event_masks_batch(masks, keys, values.astype(float),
                                       alpha=0.05)[0]

    frontier = bitmask_frontier_search(n, checker, closure="prefix")
    assert frontier == trie


def test_full_closure_is_subset_of_prefix():
    counts = synthetic_counts(5)
    keys, values = counts_to_arrays(counts)
    n = keys.shape[1]

    def checker(masks):
        return check_event_masks_batch(masks, keys, values.astype(float),
                                       alpha=0.05)[0]

    prefix = bitmask_frontier_search(n, checker, closure="prefix")
    full = bitmask_frontier_search(n, checker, closure="full")
    assert full <= prefix
    with pytest.raises(ValueError):
        bitmask_frontier_search(n, checker, closure="bogus")


# ---------------------------------------------------------------------------
# End-to-end and process-parallel equivalence
# ---------------------------------------------------------------------------

def legacy_lattice_pruning(syndrome_counts, confidence=0.95):
    """The pre-parallel implementation, verbatim."""
    n_bits = len(next(iter(syndrome_counts)))
    check_flip = lambda m: check_event_mask(m, syndrome_counts,
                                            alpha=1 - confidence)
    masks = bitmask_trie_search(n_bits, check_flip)
    return fit_specified_dem(syndrome_counts, masks)


def test_end_to_end_matches_legacy():
    counts = synthetic_counts(7, num_detectors=7, num_shots=30000)
    old = dem_to_dict(legacy_lattice_pruning(counts, confidence=0.99))
    new = dem_to_dict(lattice_pruning_dem_estimation(counts, confidence=0.99))
    assert set(old) == set(new)
    for mask in old:
        assert abs(old[mask] - new[mask]) < 1e-10


def test_n_jobs_equivalence():
    counts = synthetic_counts(11)
    serial = dem_to_dict(lattice_pruning_dem_estimation(counts, n_jobs=1))
    parallel = dem_to_dict(lattice_pruning_dem_estimation(counts, n_jobs=2))
    assert set(serial) == set(parallel)
    for mask in serial:
        assert abs(serial[mask] - parallel[mask]) < 1e-12

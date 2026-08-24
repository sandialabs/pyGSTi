import numpy as np
import pytest
import stim
from collections import Counter

from pygsti.extras.sparsedem.io import dem_from_str, dem_to_dict
from pygsti.extras.sparsedem.lattice import (
    marginalize_syndrome_counts,
    check_event_mask,
    bitmask_trie_search,
    fit_specified_dem,
    lattice_pruning_dem_estimation,
)
from pygsti.extras.sparsedem.estimation import compute_outcome_distribution_from_dem


def test_marginalize_syndrome_counts():
    syndrome_counts = {'00': 100, '01': 60, '10': 30, '11': 5}
    result_01 = marginalize_syndrome_counts(syndrome_counts, '01')
    result_10 = marginalize_syndrome_counts(syndrome_counts, '10')
    assert result_01 == {'0': 130, '1': 65}
    assert result_10 == {'0': 160, '1': 35}


def test_check_event_mask():
    # Should detect D0 D1 D2 D3
    dem_str_flip = """
    error(0.01) D0 D1
    error(0.01) D1 D2
    error(0.01) D2 D3
    error(0.01) D0 D1 D2 D3
    """
    dem = dem_from_str(dem_str_flip)
    sampler = dem.compile_sampler()
    samples = np.array(sampler.sample(2**14)[0], dtype=int)
    bitstrings = [''.join(map(str, reversed(row))) for row in samples]
    counts = Counter(bitstrings)

    assert check_event_mask('0111', counts)
    assert check_event_mask(7, counts)

    # Should NOT detect D0 D1 D2
    dem_str_noflip = """
    error(0.01) D0 D1
    error(0.01) D1 D2
    error(0.01) D2 D3
    error(0.01) D1 D2 D3
    """
    dem = dem_from_str(dem_str_noflip)
    sampler = dem.compile_sampler()
    samples = np.array(sampler.sample(2**14)[0], dtype=int)
    bitstrings = [''.join(map(str, reversed(row))) for row in samples]
    counts = Counter(bitstrings)

    assert not check_event_mask('0111', counts)
    assert not check_event_mask(7, counts)


def test_bitmask_trie_search():
    # Define a fake check_flip function with known valid masks
        # DEM events    : valid bitmask subsets (sum 2^n for all subsets of flipped detectors)
        # D0 D1     : 1,2,3
        # D1 D2     : 2, 4, 6

    valid_masks = {1, 2, 3, 4, 6}

    def check_flip(mask):
        return mask in valid_masks

    result = bitmask_trie_search(3, check_flip)
    assert result == valid_masks


def test_bitmask_trie_search_edge_case():
    # Define a fake check_flip function with known valid masks
        # DEM events    : valid bitmask subsets (sum 2^n for all subsets of flipped detectors)
        # D0 D1     : 1,2,3
        # D1 D2     : 2, 4, 6

    valid_masks = {1, 2, 3, 4, 6}

    def check_flip(mask):
        return mask in valid_masks

    result = bitmask_trie_search(0, check_flip)
    assert result == set()


def test_fit_specified_dem():
    # Create a 6-bit DEM with known events
    n_bits = 6
    dem_str = ""
    for i in range(n_bits - 1):
        prob = 0.01 + 0.002 * i
        dem_str += f"error({prob:.4f}) D{i} D{i+1}\n"
    dem_str += "error(0.1) D0\n"
    dem_str += "error(0.002) D2 D4 D0\n"

    dem = dem_from_str(dem_str)
    true_probs = compute_outcome_distribution_from_dem(dem)
    syndrome_counts = {
        f"{i:0{n_bits}b}": p for i, p in enumerate(true_probs)
    }

    masks = [21, 1] + [2**n + 2**(n - 1) for n in range(1, n_bits)]
    fit_dem = fit_specified_dem(syndrome_counts, masks, atol=1e-5)
    fit_dict = dem_to_dict(fit_dem)
    original_dict = dem_to_dict(dem)

    for k in original_dict:
        assert np.isclose(fit_dict.get(k, 0), original_dict[k], atol=1e-3)


def test_lattice_pruning_dem_estimation():
    # Create a 6-bit DEM with a mix of low- and high-weight events
    n_bits = 6
    dem_str = ""
    for i in range(n_bits - 2):
        prob = 0.01 + 0.002 * i
        dem_str += f"error({prob:.4f}) D{i} D{i+1}\n"
    dem_str += "error(0.002) D2 D4 D0\n"
    dem_str += "error(0.003) D2 D5 D0\n"
    dem_str += "error(0.004) D1 D2 D3\n"

    dem = dem_from_str(dem_str)
    sampler = dem.compile_sampler()
    samples = np.array(sampler.sample(2**14)[0], dtype=int)
    bitstrings = [''.join(map(str, reversed(row))) for row in samples]
    syndrome_counts = Counter(bitstrings)

    estimated_dem = lattice_pruning_dem_estimation(syndrome_counts)
    estimated_dict = dem_to_dict(estimated_dem)
    original_dict = dem_to_dict(dem)

    for key in original_dict:
        assert np.isclose(estimated_dict.get(key, 0), original_dict[key], atol=1e-2)


def test_lattice_pruning_dem_estimation_covariance():
    n_bits = 6
    dem_str = ""
    for i in range(n_bits - 2):
        prob = 0.01 + 0.002 * i
        dem_str += f"error({prob:.4f}) D{i} D{i+1}\n"
    dem_str += "error(0.002) D2 D4 D0\n"
    dem_str += "error(0.003) D2 D5 D0\n"
    dem_str += "error(0.004) D1 D2 D3\n"

    dem = dem_from_str(dem_str)
    sampler = dem.compile_sampler()
    samples = np.array(sampler.sample(2**14)[0], dtype=int)
    bitstrings = [''.join(map(str, reversed(row))) for row in samples]
    syndrome_counts = Counter(bitstrings)

    dem_est, dem_masks, event_probs, cov = lattice_pruning_dem_estimation(
        syndrome_counts,
        return_covariance=True,
    )

    assert isinstance(dem_est, stim.DetectorErrorModel)
    assert len(dem_masks) == len(event_probs)
    assert cov.shape == (len(event_probs), len(event_probs))
    assert np.all(np.diag(cov) >= 0)

import numpy as np
import pytest
import stim
from collections import Counter

from pygsti.extras.sparsedem.core import SparseDEMEstimator
from pygsti.extras.sparsedem.io import dem_from_str, dem_to_event_probabilities
def generate_syndrome_counts_from_dem(dem: stim.DetectorErrorModel, n_shots: int = 2**14) -> dict:
    sampler = dem.compile_sampler()
    samples = np.array(sampler.sample(n_shots)[0], dtype=int)
    bitstrings = [''.join(map(str, reversed(row))) for row in samples]
    return dict(Counter(bitstrings))


def test_estimate_dense():
    dem_str = """
    error(0.01) D0
    error(0.02) D1
    error(0.03) D0 D1
    """
    dem = dem_from_str(dem_str)
    counts = generate_syndrome_counts_from_dem(dem)
    estimator = SparseDEMEstimator(counts)
    estimated_dem = estimator.estimate_dense()
    estimated_probs = dem_to_event_probabilities(estimated_dem)
    expected_probs = dem_to_event_probabilities(dem)
    assert np.allclose(estimated_probs, expected_probs, atol=1e-2)


def test_estimate_with_covariance():
    dem_str = """
    error(0.01) D0 D1
    error(0.02) D1 D2
    """
    dem = dem_from_str(dem_str)
    counts = generate_syndrome_counts_from_dem(dem)
    estimator = SparseDEMEstimator(counts)
    dem_est, cov = estimator.estimate_with_covariance()
    assert isinstance(dem_est, stim.DetectorErrorModel)
    assert cov.shape[0] == cov.shape[1]


def test_threshold():
    dem_str = """
    error(0.01) D0
    error(0.02) D1
    error(0.03) D0 D1
    """
    dem = dem_from_str(dem_str)
    counts = generate_syndrome_counts_from_dem(dem)
    estimator = SparseDEMEstimator(counts)
    thresholded_dem = estimator.threshold(alpha=0.05)
    assert isinstance(thresholded_dem, stim.DetectorErrorModel)


def test_estimate_lasso():
    dem_str = """
    error(0.05) D0
    error(0.07) D1
    error(0.06) D0 D1
    error(0.03) D1 D2
    """
    dem = dem_from_str(dem_str)
    counts = generate_syndrome_counts_from_dem(dem)
    estimator = SparseDEMEstimator(counts)

    # Should be None before estimation
    assert estimator.get_lasso_info() is None
    assert estimator.get_last_method() is None

    lasso_dem = estimator.estimate_lasso()
    assert isinstance(lasso_dem, stim.DetectorErrorModel)
    assert estimator.get_last_method() == "lasso"
    assert set(estimator.get_last_masks()) == {1, 2, 3, 6}
    assert estimator.get_last_covariance() is not None
    assert estimator.get_lasso_info() is not None
    # Dense probs and covariance were computed lazily
    assert estimator.get_dense_probabilities() is not None
    assert estimator.get_covariance_matrix() is not None


def test_estimate_lattice_pruned():
    dem_str = """
    error(0.01) D0 D1
    error(0.02) D1 D2
    error(0.005) D0 D2 D3
    """
    dem = dem_from_str(dem_str)
    counts = generate_syndrome_counts_from_dem(dem)
    estimator = SparseDEMEstimator(counts)
    pruned_dem = estimator.estimate_lattice_pruned(confidence=0.95)
    assert isinstance(pruned_dem, stim.DetectorErrorModel)


def test_fit_custom_masks():
    dem_str = """
    error(0.01) D0 D1
    error(0.02) D1 D2
    """
    dem = dem_from_str(dem_str)
    counts = generate_syndrome_counts_from_dem(dem)
    estimator = SparseDEMEstimator(counts)
    masks = [3, 6]  # D0 D1 = 1+2=3, D1 D2 = 2+4=6
    custom_dem = estimator.fit_custom_masks(masks)
    assert isinstance(custom_dem, stim.DetectorErrorModel)

import numpy as np
import stim
from collections import Counter

from pygsti.extras.sparsedem.core import SparseDEMEstimator
from pygsti.extras.sparsedem.io import dem_from_str


def generate_syndrome_counts(dem: stim.DetectorErrorModel, n_shots: int = 2**12) -> dict:
    sampler = dem.compile_sampler()
    samples = np.array(sampler.sample(n_shots)[0], dtype=int)
    bitstrings = [''.join(map(str, reversed(row))) for row in samples]
    return dict(Counter(bitstrings))


def test_get_dense_probabilities():
    dem_str = """
    error(0.01) D0
    error(0.02) D1
    error(0.03) D0 D1
    """
    dem = dem_from_str(dem_str)
    counts = generate_syndrome_counts(dem)
    estimator = SparseDEMEstimator(counts)

    # Should be None before estimation
    assert estimator.get_dense_probabilities() is None

    estimator.estimate_dense()
    dense_probs = estimator.get_dense_probabilities()
    assert isinstance(dense_probs, np.ndarray)
    assert dense_probs.shape[0] == 2 ** dem.num_detectors


def test_get_covariance_matrix():
    dem_str = """
    error(0.01) D0 D1
    error(0.02) D1 D2
    """
    dem = dem_from_str(dem_str)
    counts = generate_syndrome_counts(dem)
    estimator = SparseDEMEstimator(counts)

    # Should be None before estimation
    assert estimator.get_covariance_matrix() is None

    estimator.estimate_with_covariance()
    cov = estimator.get_covariance_matrix()
    assert isinstance(cov, np.ndarray)
    assert cov.shape[0] == cov.shape[1]


def test_get_threshold_mask():
    dem_str = """
    error(0.01) D0
    error(0.02) D1
    error(0.03) D0 D1
    """
    dem = dem_from_str(dem_str)
    counts = generate_syndrome_counts(dem)
    estimator = SparseDEMEstimator(counts)

    # Should be None before thresholding
    assert estimator.get_threshold_mask() is None

    estimator.threshold(alpha=0.05)
    mask = estimator.get_threshold_mask()
    assert isinstance(mask, np.ndarray)
    assert mask.dtype == bool


def test_estimate_cp_recovers_support_and_caches():
    dem_str = """
    error(0.02) D0 D1 D2
    error(0.01) D0
    error(0.015) D1 D2
    error(0.01) D2 D3
    """
    dem = dem_from_str(dem_str)
    counts = generate_syndrome_counts(dem, n_shots=200000)
    estimator = SparseDEMEstimator(counts)

    learned, info = estimator.estimate_cp(return_info=True)
    assert isinstance(learned, stim.DetectorErrorModel)
    assert sorted(info["masks"]) == [0b0001, 0b0110, 0b0111, 0b1100]
    assert estimator.get_last_method() == "cp"
    assert estimator.get_last_dem() is learned


def test_log_likelihood_prefers_true_dem():
    dem_str = """
    error(0.02) D0 D1 D2
    error(0.01) D0
    error(0.015) D1 D2
    error(0.01) D2 D3
    """
    dem = dem_from_str(dem_str)
    counts = generate_syndrome_counts(dem, n_shots=50000)
    estimator = SparseDEMEstimator(counts)

    with pytest.raises(ValueError):
        estimator.log_likelihood()

    ll_true = estimator.log_likelihood(dem)
    assert np.isfinite(ll_true)
    # Dropping the hyperedge should make the data less likely.
    ll_missing = estimator.log_likelihood(dem_from_str("error(0.01) D0\nerror(0.015) D1 D2\nerror(0.01) D2 D3"))
    assert ll_missing < ll_true
    # Default argument scores the last estimated DEM.
    estimator.estimate_lattice_pruned()
    assert np.isfinite(estimator.log_likelihood())

import numpy as np
import pytest
import stim

from pygsti.extras.sparsedem.io import dem_from_str
from pygsti.extras.sparsedem.utils import counts_from_samples
from pygsti.extras.sparsedem.model_selection import (
    _pinv_sqrt,
    _nn_lasso,
    lasso_select_events,
    lasso_dem_selection,
)


def generate_syndrome_counts(dem, n_shots=2**14):
    sampler = dem.compile_sampler()
    samples = np.array(sampler.sample(n_shots)[0], dtype=int)
    return counts_from_samples(samples)


def test_pinv_sqrt():
    # Covariance with an exact null direction
    rng = np.random.default_rng(0)
    B = rng.standard_normal((4, 3))
    sigma = B @ B.T  # rank 3, one null direction
    rcond = 1e-8

    W = _pinv_sqrt(sigma, rcond=rcond)

    assert np.allclose(W, W.T)
    # W Sigma W should be the identity on the retained eigenspace and
    # zero on the truncated null direction
    evals, V = np.linalg.eigh(sigma)
    keep = evals > evals.max() * rcond
    P = V[:, keep]
    assert np.allclose(P.T @ W @ sigma @ W @ P, np.eye(keep.sum()), atol=1e-6)
    null_vec = V[:, ~keep]
    assert np.allclose(W @ null_vec, 0.0, atol=1e-6)


def test_pinv_sqrt_rejects_nonpositive():
    with pytest.raises(ValueError):
        _pinv_sqrt(np.zeros((3, 3)))


def test_nn_lasso_kkt():
    # Orthogonal design: closed form x* = max(0, c - lam)
    c = np.array([0.5, 0.05, -0.3, 1.2])
    lam = 0.1
    x = _nn_lasso(np.eye(4), c, lam, np.zeros(4), 1.0)
    assert np.allclose(x, np.maximum(0.0, c - lam), atol=1e-6)

    # KKT conditions on a random PSD Q
    rng = np.random.default_rng(1)
    B = rng.standard_normal((6, 6))
    Q = B @ B.T + 0.1 * np.eye(6)
    c = rng.standard_normal(6)
    x = _nn_lasso(Q, c, lam, np.zeros(6), np.linalg.eigvalsh(Q)[-1], tol=1e-12)
    grad = Q @ x - c + lam
    assert np.all(grad[x == 0] >= -1e-5)
    assert np.allclose(grad[x > 0], 0.0, atol=1e-5)


def test_lasso_select_events_synthetic():
    # Sparse truth plus small Gaussian noise, diagonal covariance
    rng = np.random.default_rng(2)
    size = 16
    sigma = 1e-3
    truth = np.zeros(size)
    true_masks = [1, 4, 6]
    truth[true_masks] = [0.05, 0.03, 0.04]
    p_hat = truth + sigma * rng.standard_normal(size)
    p_hat[0] = sigma * rng.standard_normal()  # index 0 perturbed but never selectable
    covariance = sigma**2 * np.eye(size)

    masks, info = lasso_select_events(p_hat, covariance, n_shots=2**14)

    assert set(masks) == set(true_masks)
    assert 0 not in masks
    assert info["lambda_best"] is not None


def test_lasso_dem_selection_end_to_end():
    dem_str = """
    error(0.05) D0
    error(0.07) D1
    error(0.06) D0 D1
    error(0.03) D1 D2
    """
    dem = dem_from_str(dem_str)
    syndrome_counts = generate_syndrome_counts(dem)

    fitted_dem, dem_masks, event_probs, covariance = lasso_dem_selection(syndrome_counts)

    assert isinstance(fitted_dem, stim.DetectorErrorModel)
    assert set(dem_masks) == {1, 2, 3, 6}
    expected = {1: 0.05, 2: 0.07, 3: 0.06, 6: 0.03}
    for mask, prob in zip(dem_masks, event_probs):
        assert abs(prob - expected[mask]) < 1e-2
    assert covariance.shape == (len(dem_masks), len(dem_masks))
    assert np.all(np.diag(covariance) >= 0)


def test_lasso_fixed_lambda():
    dem_str = """
    error(0.05) D0
    error(0.07) D1
    """
    dem = dem_from_str(dem_str)
    syndrome_counts = generate_syndrome_counts(dem)

    # Huge lambda: nothing selected, empty DEM without crashing
    empty_dem, masks, probs, cov = lasso_dem_selection(syndrome_counts, lam=1e12)
    assert len(empty_dem) == 0
    assert len(masks) == 0
    assert probs.shape == (0,)
    assert cov.shape == (0, 0)

    # Small lambda: support contains the truth
    _, masks, _, _ = lasso_dem_selection(syndrome_counts, lam=1e-6)
    assert {1, 2}.issubset(set(masks))


def test_lasso_return_path():
    dem_str = """
    error(0.05) D0
    error(0.07) D1
    error(0.06) D0 D1
    """
    dem = dem_from_str(dem_str)
    syndrome_counts = generate_syndrome_counts(dem)

    _, _, _, _, info = lasso_dem_selection(syndrome_counts, return_path=True)

    lambdas = info["lambdas"]
    assert np.all(np.diff(lambdas) < 0)
    assert len(info["bics"]) == len(info["supports"])
    assert info["lambda_best"] in lambdas

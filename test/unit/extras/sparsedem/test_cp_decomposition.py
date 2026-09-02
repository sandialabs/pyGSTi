import numpy as np
import pytest
import stim
from scipy.optimize import check_grad

from pygsti.extras.sparsedem import cp_decomposition as cpd
from pygsti.extras.sparsedem.estimation import compute_outcome_distribution_from_dem
from pygsti.extras.sparsedem.io import dem_from_str, dem_to_dict
from pygsti.extras.sparsedem.lattice import lattice_pruning_dem_estimation
from pygsti.extras.sparsedem.utils import counts_from_samples


MIXED_DEM_STR = """
error(0.01) D0 D1 D2
error(0.012) D1 D3
error(0.008) D2 D3
error(0.015) D0
error(0.01) D3 D4
error(0.02) D4
"""


def _scaled_dem(dem_str, scale):
    lines = []
    for line in dem_str.strip().splitlines():
        head, rest = line.split(")", 1)
        p = float(head.split("(")[1]) * scale
        lines.append(f"error({p}){rest}")
    return dem_from_str("\n".join(lines))


def _sample_counts(dem, shots, seed):
    samples = dem.compile_sampler(seed=seed).sample(shots)[0].astype(int)
    return counts_from_samples(samples)


def _rel_err(a, b):
    return np.linalg.norm(a - b) / np.linalg.norm(b)


# ---------------------------------------------------------------------------
# Leading-order structure and sample cumulants
# ---------------------------------------------------------------------------

def test_exact_cumulants_match_leading_order_cp_to_second_order():
    errs = {}
    for scale in (1.0, 0.25):
        dem = _scaled_dem(MIXED_DEM_STR, scale)
        for k in (2, 3):
            exact = cpd.exact_cumulant_tensor_from_dem(dem, k)
            lead = cpd.leading_order_cumulant_tensor(dem, k)
            # symmetric
            assert np.allclose(exact, np.swapaxes(exact, 0, 1))
            errs[(scale, k)] = _rel_err(exact, lead)
    assert errs[(1.0, 2)] < 0.1 and errs[(1.0, 3)] < 0.15
    # O(p) relative error: shrinks (about) linearly when p is scaled by 1/4.
    for k in (2, 3):
        assert errs[(0.25, k)] < errs[(1.0, k)] / 2.5


def test_exact_cumulants_agree_with_outcome_distribution_moments():
    # Direct check of a few entries against the 2^n distribution.
    dem = dem_from_str(MIXED_DEM_STR)
    n = dem.num_detectors
    probs = compute_outcome_distribution_from_dem(dem)
    idx = np.arange(2 ** n)
    bits = ((idx[:, None] >> np.arange(n)) & 1).astype(float)
    mu = probs @ bits
    C = bits - mu
    k3 = np.einsum("s,sa,sb,sc->abc", probs, C, C, C)
    k2 = np.einsum("s,sa,sb->ab", probs, C, C)
    assert np.allclose(cpd.exact_cumulant_tensor_from_dem(dem, 3), k3)
    assert np.allclose(cpd.exact_cumulant_tensor_from_dem(dem, 2), k2)
    # Repeated-index identity: kappa(Y_a, Y_a, Y_b) = (1 - 2 mu_a) Cov(Y_a, Y_b).
    for a in range(n):
        for b in range(n):
            assert np.isclose(k3[a, a, b], (1 - 2 * mu[a]) * k2[a, b])


def test_order4_cumulant_of_single_event():
    p = 0.05
    dem = dem_from_str(f"error({p}) D0\nerror(0.01) D1")
    k4 = cpd.exact_cumulant_tensor_from_dem(dem, 4)
    expected = p * (1 - p) * (1 - 6 * p + 6 * p ** 2)
    assert np.isclose(k4[0, 0, 0, 0], expected)
    assert np.isclose(k4[0, 0, 1, 1], 0.0, atol=1e-12)  # independent detectors


def test_sample_cumulants_converge_to_exact():
    dem = dem_from_str(MIXED_DEM_STR)
    counts = _sample_counts(dem, 200_000, seed=3)
    ct = cpd.cumulant_tensors(counts, order=3)
    assert ct.n_shots == 200_000
    for k in (2, 3):
        exact = cpd.exact_cumulant_tensor_from_dem(dem, k)
        se = np.maximum(ct.stderrs[k], 1.0 / ct.n_shots)
        z = (ct.tensors[k] - exact) / se
        assert np.abs(z).max() < 5.0
        # Most entries within 2 standard errors.
        assert np.mean(np.abs(z) < 2.0) > 0.8
    # Dict and array inputs agree.
    samples, weights = cpd._samples_and_weights(counts)
    T_arr, se_arr = cpd.joint_cumulant_tensor(samples, order=3, weights=weights, return_stderr=True)
    assert np.allclose(T_arr, ct.tensors[3]) and np.allclose(se_arr, ct.stderrs[3])


def test_detector_subset_is_a_subblock():
    dem = dem_from_str(MIXED_DEM_STR)
    counts = _sample_counts(dem, 20_000, seed=4)
    full = cpd.joint_cumulant_tensor(counts, order=3)
    sub = cpd.joint_cumulant_tensor(counts, order=3, detectors=[4, 1, 2])
    assert sub.shape == (3, 3, 3)
    assert np.allclose(sub, full[np.ix_([4, 1, 2], [4, 1, 2], [4, 1, 2])])


# ---------------------------------------------------------------------------
# Symmetric CP solver
# ---------------------------------------------------------------------------

def test_cp_gradient_matches_finite_differences():
    rng = np.random.default_rng(1)
    m, R = 4, 3
    T = rng.normal(size=(m, m, m))
    T = sum(np.transpose(T, perm) for perm in [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]) / 6
    T2 = rng.normal(size=(m, m))
    T2 = T2 + T2.T
    W3 = cpd._prepare_weights(T, np.abs(T) + 0.1, None)
    W2 = cpd._prepare_weights(T2, None, None)
    prob = cpd._CPProblem([(T, W3, 3), (T2, W2, 2)], m, R)
    x0 = rng.uniform(size=R + m * R)
    err = check_grad(lambda x: prob(x)[0], lambda x: prob(x)[1], x0)
    assert err < 1e-5


def _planted():
    B = np.array([[1, 1, 1, 0, 0, 0],
                  [1, 0, 0, 1, 0, 0],
                  [0, 1, 0, 1, 1, 0],
                  [0, 0, 0, 0, 1, 1]], dtype=float).T  # (6, 4)
    w = np.array([0.01, 0.02, 0.005, 0.03])
    return B, w


def test_symmetric_cp_recovers_planted_factors_noiseless():
    B, w = _planted()
    T = cpd.cp_reconstruct(w, B, 3)
    lam, F, info = cpd.symmetric_cp(T, 4, n_restarts=4, seed=0, max_iter=1000, tol=1e-15)
    assert info["relative_residual"] < 1e-3
    masks, weights = cpd.factors_to_masks(F, lam)
    true_masks, _ = cpd.factors_to_masks(B, w)
    assert sorted(masks) == sorted(true_masks)
    # Weights match up to permutation.
    assert np.allclose(sorted(weights), sorted(w), rtol=0.05)


def test_symmetric_cp_recovers_planted_factors_noisy():
    B, w = _planted()
    rng = np.random.default_rng(5)
    T = cpd.cp_reconstruct(w, B, 3)
    noise = rng.normal(scale=2e-4, size=T.shape)
    noise = sum(np.transpose(noise, perm) for perm in
                [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]) / 6
    se = np.full(T.shape, 2e-4)
    lam, F, info = cpd.symmetric_cp(T + noise, 4, stderr=se, n_restarts=4, seed=0)
    masks, weights = cpd.factors_to_masks(F, lam)
    true_masks, _ = cpd.factors_to_masks(B, w)
    assert sorted(masks) == sorted(true_masks)
    assert info["chi2"] / info["dof"] < 2.0


def test_factors_to_masks_maps_subset_and_dedupes():
    F = np.array([[0.9, 0.1, 0.8, 0.0],
                  [0.7, 0.2, 0.9, 0.0],
                  [0.0, 0.6, 0.0, 0.0]])
    w = np.array([0.01, 0.02, 0.005, 0.03])
    masks, weights = cpd.factors_to_masks(F, w, detectors=[5, 2, 7])
    # column 0 and 2 both round to rows {0, 1} -> detectors {5, 2}; column 3 is empty.
    assert masks == [(1 << 5) | (1 << 2), 1 << 7] or masks == [1 << 7, (1 << 5) | (1 << 2)]
    d = dict(zip(masks, weights))
    assert np.isclose(d[(1 << 5) | (1 << 2)], 0.015) and np.isclose(d[1 << 7], 0.02)
    with pytest.raises(ValueError):
        cpd.factors_to_masks(F, w, detectors=[1, 2])


# ---------------------------------------------------------------------------
# Hyperedge vs triangle: why order 3 is needed
# ---------------------------------------------------------------------------

def test_hyperedge_vs_triangle_covariances_agree_but_order3_differs():
    p = 0.01
    hyper = dem_from_str(f"error({p}) D0 D1 D2")
    tri = dem_from_str(f"error({p}) D0 D1\nerror({p}) D1 D2\nerror({p}) D0 D2")
    cov_h = cpd.exact_cumulant_tensor_from_dem(hyper, 2)
    cov_t = cpd.exact_cumulant_tensor_from_dem(tri, 2)
    # Off-diagonal covariances agree to leading order (relative difference O(p)).
    off = ~np.eye(3, dtype=bool)
    assert np.all(np.abs(cov_h[off] - cov_t[off]) / cov_h[off] < 6 * p)  # difference is 4 p^2
    k3_h = cpd.exact_cumulant_tensor_from_dem(hyper, 3)[0, 1, 2]
    k3_t = cpd.exact_cumulant_tensor_from_dem(tri, 3)[0, 1, 2]
    assert np.isclose(k3_h, p * (1 - p) * (1 - 2 * p))
    # kappa3 = -6 p^2 (1-p)^2 + 16 p^3 (1-p)^3 for the triangle: second-order small and negative.
    assert np.isclose(k3_t, -6 * p ** 2 * (1 - p) ** 2 + 16 * p ** 3 * (1 - p) ** 3)
    assert abs(k3_t) < 8 * p ** 2 and k3_t < 0
    # Sampled pipeline discriminates the two supports.
    counts_h = _sample_counts(hyper, 100_000, seed=11)
    counts_t = _sample_counts(tri, 100_000, seed=12)
    dem_h, info_h = cpd.cp_dem_estimation(counts_h, return_info=True)
    dem_t, info_t = cpd.cp_dem_estimation(counts_t, return_info=True)
    assert sorted(info_h["masks"]) == [0b111]
    assert sorted(info_t["masks"]) == [0b011, 0b101, 0b110]
    assert np.isclose(dem_to_dict(dem_h)[0b111], p, rtol=0.15)
    # Order 2 alone cannot tell them apart: same candidate pairs on both.
    pairs_h = cpd.candidate_supports([(cov_h, np.full((3, 3), 1e-4))])
    pairs_t = cpd.candidate_supports([(cov_t, np.full((3, 3), 1e-4))])
    assert pairs_h == pairs_t


# ---------------------------------------------------------------------------
# End-to-end
# ---------------------------------------------------------------------------

def test_cp_dem_estimation_recovers_mixed_support_and_probabilities():
    dem = dem_from_str(MIXED_DEM_STR)
    counts = _sample_counts(dem, 200_000, seed=21)
    est, info = cpd.cp_dem_estimation(counts, return_info=True)
    truth = dem_to_dict(dem)
    assert sorted(info["masks"]) == sorted(truth)
    est_dict = dem_to_dict(est)
    for mask, p in truth.items():
        assert abs(est_dict[mask] - p) < 0.2 * p + 5e-4
    assert isinstance(est, stim.DetectorErrorModel)
    assert info["rank"] >= len(truth)


def test_cp_dem_estimation_merges_into_high_weight_events():
    dem = dem_from_str("""
    error(0.01) D0 D1 D2 D3
    error(0.008) D2 D3 D4 D5 D6
    error(0.012) D1 D4
    error(0.01) D5
    error(0.015) D6 D7
    error(0.01) D0 D3 D7
    """)
    counts = _sample_counts(dem, 200_000, seed=7)
    est, info = cpd.cp_dem_estimation(counts, return_info=True)
    truth = dem_to_dict(dem)
    # Candidate supports have at most 3 detectors; the CP refinement merges them.
    assert max(len(s) for s in info["init_supports"]) <= 3
    assert sorted(info["masks"]) == sorted(truth)


def test_cp_dem_estimation_on_circuit_dem_matches_truth_and_lattice():
    circuit = stim.Circuit.generated("repetition_code:memory", distance=3, rounds=3,
                                     after_clifford_depolarization=0.02,
                                     before_measure_flip_probability=0.01)
    dem = circuit.detector_error_model(decompose_errors=False).flattened()
    assert dem.num_detectors == 8
    truth = dem_to_dict(dem)
    # This circuit DEM has no hyperedges (all events have weight <= 2).
    assert max(bin(m).count("1") for m in truth) == 2
    counts = _sample_counts(dem, 200_000, seed=5)
    est, info = cpd.cp_dem_estimation(counts, return_info=True)
    rec = set(info["masks"])
    true_set = set(truth)
    jaccard = len(rec & true_set) / len(rec | true_set)
    assert jaccard >= 0.9
    est_dict = dem_to_dict(est)
    for mask in rec & true_set:
        assert abs(est_dict[mask] - truth[mask]) < 0.25 * truth[mask] + 1e-3
    lat = lattice_pruning_dem_estimation(counts)
    lat_set = set(dem_to_dict(lat))
    assert len(rec & lat_set) / len(true_set) >= 0.9


def test_cp_dem_estimation_screened_mode_and_order2():
    dem = dem_from_str(MIXED_DEM_STR)
    counts = _sample_counts(dem, 100_000, seed=31)
    truth = sorted(dem_to_dict(dem))
    _, info = cpd.cp_dem_estimation(counts, return_info=True,
                                    config=cpd.CPConfig(screen=True, n_restarts=2))
    assert sorted(info["masks"]) == truth
    assert info["entry_mask"] is not None and info["entry_mask"].dtype == bool
    # Order 2 runs (the non-unique p_ij setting) and still finds the pairs and singles.
    _, info2 = cpd.cp_dem_estimation(counts, order=2, return_info=True,
                                     config=cpd.CPConfig(n_restarts=2, rank_max=8))
    assert set(info2["masks"]) & {1 << 0, 1 << 4}


def test_cp_dem_estimation_is_deterministic():
    dem = dem_from_str(MIXED_DEM_STR)
    counts = _sample_counts(dem, 50_000, seed=41)
    cfg = cpd.CPConfig(seed=123, init="random", rank_max=6, n_restarts=2, max_iter=120)
    _, a = cpd.cp_dem_estimation(counts, return_info=True, config=cfg)
    _, b = cpd.cp_dem_estimation(counts, return_info=True, config=cfg)
    assert a["masks"] == b["masks"]
    assert np.array_equal(a["cp_weights"], b["cp_weights"])
    assert np.array_equal(a["cp_factors"], b["cp_factors"])
    assert a["rank"] == b["rank"]

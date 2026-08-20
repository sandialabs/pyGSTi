import numpy as np
import pytest
import scipy.stats
import stim

from pygsti.extras.sparsedem.validation import (
    ValidationResult,
    ValidationSuiteResult,
    polarization_from_dem,
    sample_dem,
)
from pygsti.extras.sparsedem.validation import (
    click_rate_drift_test,
    event_aligned_masks,
    hamming_weight,
    hamming_weight_test,
    polarization_drift_test,
    polarization_tests,
    predicted_polarizations,
    random_masks,
    run_polarization_battery,
    run_stationarity_battery,
    scalar_distribution_test,
    shot_autocorrelation_test,
    triple_masks,
    weight1_masks,
    weight2_masks,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

SMALL_DEM_STR = """
error(0.08) D0
error(0.12) D1
error(0.05) D2
error(0.04) D0 D1
error(0.06) D1 D2
error(0.03) D2 D3
error(0.05) D3 D4
error(0.02) D4
"""


def small_dem():
    return stim.DetectorErrorModel(SMALL_DEM_STR)


def uniformity_ok(pvalues, alpha=0.05):
    """The p-values are consistent with Uniform(0,1): the fraction below
    alpha stays within generous binomial slack and a KS test does not reject
    at the 1e-3 level."""
    p = np.asarray(pvalues, dtype=float)
    frac = float(np.mean(p < alpha))
    slack = alpha + 4.0 * np.sqrt(alpha * (1 - alpha) / p.size) + 0.02
    ks_p = scipy.stats.kstest(p, "uniform").pvalue
    return frac <= slack and ks_p > 1e-3


# ---------------------------------------------------------------------------
# Predicted polarizations and mask builders
# ---------------------------------------------------------------------------

def test_predicted_polarizations_match_scalar_reference():
    dem = small_dem()
    masks = [(0,), (1,), (4,), (0, 1), (1, 3), (0, 2, 4), (0, 1, 2, 3, 4)]
    vec = predicted_polarizations(dem, masks)
    ref = np.array([polarization_from_dem(dem, m) for m in masks])
    assert np.allclose(vec, ref, atol=1e-12)


def test_predicted_polarizations_edge_cases():
    assert predicted_polarizations(small_dem(), []).size == 0
    empty = stim.DetectorErrorModel("detector D2")
    assert np.allclose(predicted_polarizations(empty, [(0,), (1, 2)]), 1.0)
    half = stim.DetectorErrorModel("error(0.5) D0")
    assert predicted_polarizations(half, [(0,)])[0] == 0.0


def test_weight1_and_weight2_masks():
    assert weight1_masks(4) == [(0,), (1,), (2,), (3,)]
    pairs = weight2_masks(5)
    assert len(pairs) == 10 and all(i < j for i, j in pairs)
    with pytest.warns(UserWarning):
        sub = weight2_masks(40, max_masks=25, seed=11)
    assert len(sub) == 25 and len(set(sub)) == 25
    assert weight2_masks(40, max_masks=25, seed=11) == sub


def test_random_masks_distinct_and_bounded():
    masks = random_masks(30, max_weight=4, num_masks=50, seed=3)
    assert len(masks) == len(set(masks)) == 50
    assert all(1 <= len(m) <= 4 and max(m) < 30 for m in masks)
    assert random_masks(30, max_weight=4, num_masks=50, seed=3) == masks
    # Exhausted space: only 3 distinct weight-1 masks exist on 3 detectors.
    assert len(random_masks(3, max_weight=1, num_masks=10, seed=0)) == 3


def test_event_aligned_masks():
    masks = event_aligned_masks(small_dem())
    assert (0, 1) in masks and (1,) in masks and (3, 4) in masks
    assert len(masks) == 8


def test_triple_masks_connected_and_random():
    dem = small_dem()
    masks = triple_masks(dem, num_masks=20, seed=5)
    assert all(len(m) == 3 and len(set(m)) == 3 for m in masks)
    # D1 has neighbors {0, 2} through events, so (0,1,2) is a connected
    # triple and must be reachable; with 5 detectors the full triple space
    # is C(5,3)=10, all of which fit in the request.
    assert (0, 1, 2) in masks
    assert triple_masks(stim.DetectorErrorModel("error(0.1) D0 D1"),
                        num_masks=10, seed=0) == []


# ---------------------------------------------------------------------------
# Polarization tests: correctness, calibration, power
# ---------------------------------------------------------------------------

def test_polarization_tests_validation_errors():
    dem = small_dem()
    det, _ = sample_dem(dem, 10, seed=0)
    with pytest.raises(ValueError):
        polarization_tests(dem, det, [()])
    with pytest.raises(ValueError):
        polarization_tests(dem, det, [(7,)])
    with pytest.raises(ValueError):
        polarization_tests(dem, det[:0], [(0,)])
    assert polarization_tests(dem, det, []).results == []


def test_polarization_tests_result_fields():
    dem = small_dem()
    det, _ = sample_dem(dem, 4000, seed=1)
    suite = polarization_tests(dem, det, [(0,), (1, 2)], name_prefix="pol")
    assert len(suite.results) == 2
    r = suite.results[0]
    assert r.name == "pol[(0)]"
    assert r.null_model == "dem"
    assert r.num_shots == 4000
    assert r.effect_size == r.statistic
    assert "observed polarization" in r.effect_description
    assert abs(r.details["observed_polarization"]
               - (1 - 2 * det[:, 0].mean())) < 1e-12


def test_polarization_calibration_on_model():
    dem = small_dem()
    masks = [(0,), (2,), (1, 2), (0, 3), (1, 2, 3)]
    pvals = []
    for rep in range(120):
        det, _ = sample_dem(dem, 1500, seed=1000 + rep)
        suite = polarization_tests(dem, det, masks)
        pvals.extend(r.pvalue for r in suite.results)
    assert uniformity_ok(pvals)


def test_polarization_exact_binomial_branch():
    # Detector 5 is untouched by any event: pol0 = +1 exactly.
    dem = stim.DetectorErrorModel("error(0.1) D0\ndetector D5")
    det = np.zeros((200, 6), dtype=np.uint8)
    suite = polarization_tests(dem, det, [(5,)])
    r = suite.results[0]
    assert r.details["method"] == "binomial"
    assert r.pvalue == 1.0
    # A click on an impossible detector must be flagged decisively.
    det_bad = det.copy()
    det_bad[:5, 5] = 1
    r_bad = polarization_tests(dem, det_bad, [(5,)]).results[0]
    assert r_bad.pvalue < 1e-6


def test_polarization_power_doubled_event():
    truth = stim.DetectorErrorModel("""
        error(0.20) D0 D1
        error(0.05) D1 D2
        error(0.05) D0
    """)
    candidate = stim.DetectorErrorModel("""
        error(0.10) D0 D1
        error(0.05) D1 D2
        error(0.05) D0
    """)
    det, _ = sample_dem(truth, 5000, seed=7)
    suite = polarization_tests(candidate, det, event_aligned_masks(candidate))
    by_mask = {r.details["mask"]: r for r in suite.results}
    # A mask only feels events it overlaps oddly: doubling D0 D1 moves the
    # polarization of (0,) (overlap 1) but leaves its own support (0, 1)
    # (overlap 2, even) untouched.
    assert by_mask[(0,)].pvalue < 1e-8
    assert by_mask[(0, 1)].pvalue > 1e-3


def test_polarization_power_missing_hyperedge():
    # Truth contains a weight-3 event the graphlike candidate lacks; the
    # pairwise events make (0,1,2) a connected triple of the candidate.
    common = "error(0.04) D0 D1\nerror(0.04) D1 D2\nerror(0.04) D0 D2\n"
    truth = stim.DetectorErrorModel(common + "error(0.05) D0 D1 D2")
    candidate = stim.DetectorErrorModel(common)
    det, _ = sample_dem(truth, 6000, seed=9)
    masks = triple_masks(candidate, num_masks=20, seed=0)
    assert (0, 1, 2) in masks
    suite = polarization_tests(candidate, det, masks, name_prefix="polarization_w3")
    by_mask = {r.details["mask"]: r for r in suite.results}
    assert by_mask[(0, 1, 2)].pvalue < 1e-6


def test_run_polarization_battery():
    dem = small_dem()
    det, _ = sample_dem(dem, 2000, seed=3)
    suite = run_polarization_battery(dem, det, seed=0)
    names = [r.name for r in suite.results]
    assert any(n.startswith("polarization_w1") for n in names)
    assert any(n.startswith("polarization_w2") for n in names)
    assert any(n.startswith("polarization_event") for n in names)
    assert any(n.startswith("polarization_w3") for n in names)
    with pytest.raises(ValueError):
        run_polarization_battery(dem, det, collections=("nope",))


# ---------------------------------------------------------------------------
# Scalar-function distribution tests
# ---------------------------------------------------------------------------

def test_hamming_weight_function():
    det = np.array([[0, 1, 1], [0, 0, 0], [1, 1, 1]], dtype=np.uint8)
    assert hamming_weight(det).tolist() == [2, 0, 3]


@pytest.mark.parametrize("method", ["chi2", "ks", "mean"])
def test_scalar_calibration_on_model(method):
    dem = small_dem()
    pvals = []
    for rep in range(80):
        det, _ = sample_dem(dem, 800, seed=2000 + rep)
        r = hamming_weight_test(dem, det, method=method,
                                num_null_shots=6000, seed=999 + rep)
        pvals.append(r.pvalue)
    # The KS test is conservative under ties (documented), so only guard
    # against anti-conservatism for it; chi2/mean should be near-uniform.
    frac = float(np.mean(np.asarray(pvals) < 0.05))
    if method == "ks":
        assert frac <= 0.10
    else:
        assert uniformity_ok(pvals)


def test_scalar_power_scaled_model():
    truth = small_dem()
    # Candidate with all probabilities halved: Hamming weight runs high.
    halved = stim.DetectorErrorModel()
    for inst in truth:
        halved.append("error", inst.args_copy()[0] / 2, inst.targets_copy())
    det, _ = sample_dem(truth, 4000, seed=21)
    for method in ("chi2", "mean"):
        r = hamming_weight_test(halved, det, method=method,
                                num_null_shots=40000, seed=5)
        assert r.pvalue < 1e-8
        assert r.effect_size > 0.3  # runs high, in null-SD units
        assert "high" in r.effect_description


def test_scalar_auto_method_selection():
    dem = small_dem()
    det, _ = sample_dem(dem, 500, seed=2)
    r = hamming_weight_test(dem, det, num_null_shots=4000, seed=3)
    assert r.details["method_used"] == "chi2"  # few distinct integer values
    continuous = scalar_distribution_test(
        dem, det, lambda s: s.sum(axis=1) + np.linspace(0, 1e-6, s.shape[0]),
        num_null_shots=4000, seed=3, name="jittered")
    assert continuous.details["method_used"] == "ks"


def test_scalar_single_shot_tail_fallback():
    dem = small_dem()
    det_typical, _ = sample_dem(dem, 1, seed=4)
    r = hamming_weight_test(dem, det_typical, num_null_shots=4000, seed=5)
    assert r.details["method_used"] == "tail"
    assert r.pvalue > 0.01
    det_extreme = np.ones((1, 5), dtype=np.uint8)
    r_bad = hamming_weight_test(dem, det_extreme, num_null_shots=4000, seed=5)
    assert r_bad.pvalue < 0.05


def test_scalar_constant_function():
    dem = small_dem()
    det, _ = sample_dem(dem, 100, seed=6)
    r = scalar_distribution_test(dem, det, lambda s: np.zeros(s.shape[0]),
                                 num_null_shots=2000, seed=7, name="const")
    assert r.pvalue == 1.0
    assert r.effect_size == 0.0
    assert "constant under the model" in r.effect_description


def test_scalar_argument_validation():
    dem = small_dem()
    det, _ = sample_dem(dem, 10, seed=8)
    with pytest.raises(ValueError):
        scalar_distribution_test(dem, det, hamming_weight, method="bogus")
    with pytest.raises(ValueError):
        scalar_distribution_test(dem, det[:0], hamming_weight)
    r = scalar_distribution_test(dem, det, hamming_weight, seed=0)
    assert r.details["num_null_shots"] == 20000
    assert r.name == "scalar[hamming_weight]"


# ---------------------------------------------------------------------------
# Stationarity tests
# ---------------------------------------------------------------------------

def test_stationarity_calibration_on_iid_data():
    dem = small_dem()
    counts = {"click": [], "pol": [], "auto": []}
    for rep in range(50):
        det, _ = sample_dem(dem, 2000, seed=3000 + rep)
        counts["click"].append(click_rate_drift_test(det).pvalue)
        counts["pol"].append(polarization_drift_test(det).pvalue)
        counts["auto"].append(shot_autocorrelation_test(
            det, num_permutations=100, seed=rep).pvalue)
    for key, pvals in counts.items():
        frac = float(np.mean(np.asarray(pvals) < 0.05))
        assert frac <= 0.20, f"{key}: fraction of small p-values {frac}"


def test_click_rate_drift_power():
    low = stim.DetectorErrorModel("error(0.05) D0\nerror(0.05) D1")
    high = stim.DetectorErrorModel("error(0.15) D0\nerror(0.15) D1")
    det = np.vstack([sample_dem(low, 2000, seed=1)[0],
                     sample_dem(high, 2000, seed=2)[0]])
    r = click_rate_drift_test(det)
    assert r.null_model == "iid"
    assert r.pvalue < 1e-8
    assert r.effect_size > 0  # rate rises with shot index
    assert abs(r.details["trend_z"]) > 4


def test_polarization_drift_power_rate_conserving():
    # Rate moves from D0 to D1 with the total conserved: the click-rate test
    # has little to see, the polarization drift test must reject.
    a = stim.DetectorErrorModel("error(0.20) D0\nerror(0.05) D1")
    b = stim.DetectorErrorModel("error(0.05) D0\nerror(0.20) D1")
    det = np.vstack([sample_dem(a, 3000, seed=3)[0],
                     sample_dem(b, 3000, seed=4)[0]])
    r = polarization_drift_test(det, masks=[(0,), (1,)])
    assert r.pvalue < 1e-8
    assert "sigma" in r.effect_description
    assert click_rate_drift_test(det).pvalue > 1e-4


def test_autocorrelation_power_duplicated_shots():
    dem = small_dem()
    det, _ = sample_dem(dem, 1500, seed=5)
    doubled = np.repeat(det, 2, axis=0)
    r = shot_autocorrelation_test(doubled, num_permutations=200, seed=0)
    assert r.pvalue < 0.01
    assert r.effect_size > 0.2
    r_iid = shot_autocorrelation_test(det, num_permutations=200, seed=0)
    assert r_iid.pvalue > 0.01


def test_stationarity_degenerate_inputs():
    one = np.array([[0, 1, 0]], dtype=np.uint8)
    for result in (click_rate_drift_test(one), polarization_drift_test(one),
                   shot_autocorrelation_test(one)):
        assert result.pvalue == 1.0 and result.effect_size == 0.0
    zeros = np.zeros((500, 3), dtype=np.uint8)
    assert click_rate_drift_test(zeros).pvalue == 1.0
    assert polarization_drift_test(zeros).pvalue == 1.0
    assert shot_autocorrelation_test(zeros).pvalue == 1.0


def test_run_stationarity_battery():
    dem = small_dem()
    det, _ = sample_dem(dem, 1000, seed=6)
    suite = run_stationarity_battery(det, seed=0)
    assert len(suite.results) == 3
    assert all(r.null_model == "iid" for r in suite.results)
    names = {r.name for r in suite.results}
    assert names == {"stationarity_click_rate",
                     "stationarity_polarization_drift",
                     "stationarity_autocorrelation"}

import numpy as np
import pytest

from pygsti.extras.sparsedem.io import dem_from_str
from pygsti.extras.sparsedem import validation
from pygsti.extras.sparsedem.validation import sample_dem
from pygsti.extras.sparsedem.validation import (
    decode_logical_predictions,
    logical_error_rate_test,
    matching_weight_function,
    complementary_gap_function,
    matching_weight_test,
    complementary_gap_test,
)

pymatching = pytest.importorskip("pymatching")

DECODERS = ["pymatching", "tesseract"]


def _require_decoder(name):
    """Skip cleanly when the requested decoder backend is not installed."""
    if name == "tesseract":
        pytest.importorskip("tesseract_decoder")


def _hand_dem():
    """3-event repetition line whose decodings are enumerable by hand."""
    return dem_from_str("""
    error(0.01) D0 L0
    error(0.02) D0 D1
    error(0.03) D1
    """)


#: log((1-p)/p) weights of the three _hand_dem events, in the file's order.
_HAND_W = [np.log(0.99 / 0.01), np.log(0.98 / 0.02), np.log(0.97 / 0.03)]


def _line_dem(p, n=5):
    """Decorated repetition-code line: logical flag on the left boundary."""
    lines = [f"error({p}) D0 L0"]
    for d in range(n - 1):
        lines.append(f"error({p}) D{d} D{d + 1}")
    lines.append(f"error({p}) D{n - 1}")
    return dem_from_str("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Matching-weight scalar function
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("decoder", DECODERS)
def test_matching_weight_hand_values(decoder):
    _require_decoder(decoder)
    func = matching_weight_function(_hand_dem(), decoder=decoder)
    shots = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.uint8)
    weights = func(shots)
    # Zero syndrome -> weight 0; single-event syndromes -> that event's
    # log((1-p)/p); [1,1] -> the D0 D1 pair event.
    expected = [0.0, _HAND_W[0], _HAND_W[2], _HAND_W[1]]
    assert weights.shape == (4,)
    np.testing.assert_allclose(weights, expected, rtol=1e-5, atol=1e-9)


def test_matching_weight_rejects_1d_input():
    func = matching_weight_function(_hand_dem())
    with pytest.raises(ValueError, match="2D"):
        func(np.array([0, 1], dtype=np.uint8))


def test_matching_weight_large_batch():
    dem = _line_dem(0.12)
    det, _ = sample_dem(dem, 200000, seed=7)
    weights = matching_weight_function(dem)(det)
    assert weights.shape == (200000,)
    assert np.all(weights >= 0)
    assert weights[np.all(det == 0, axis=1)].max(initial=0.0) == 0.0


# ---------------------------------------------------------------------------
# Complementary-gap scalar function
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("decoder", DECODERS)
def test_complementary_gap_hand_values(decoder):
    _require_decoder(decoder)
    dem = _hand_dem()
    gap = complementary_gap_function(dem, decoder=decoder)
    signed = complementary_gap_function(dem, decoder=decoder, sign=True)
    shots = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.uint8)
    w0, w1, w2 = _HAND_W
    # Hand enumeration of both logical classes per syndrome:
    #   [0,0]: class0 {} = 0            vs class1 {e0,e1,e2} = w0+w1+w2
    #   [1,0]: class0 {e1,e2} = w1+w2   vs class1 {e0} = w0
    #   [0,1]: class0 {e2} = w2         vs class1 {e0,e1} = w0+w1
    #   [1,1]: class0 {e1} = w1         vs class1 {e0,e2} = w0+w2
    expected_signed = [
        w0 + w1 + w2,
        w0 - (w1 + w2),
        (w0 + w1) - w2,
        (w0 + w2) - w1,
    ]
    np.testing.assert_allclose(signed(shots), expected_signed, rtol=1e-5)
    np.testing.assert_allclose(gap(shots), np.abs(expected_signed), rtol=1e-5)
    # The [1,0] syndrome is cheapest to explain WITH a logical flip, so the
    # signed gap is negative there (decoder chooses class 1) and positive on
    # the class-0-preferred syndromes.
    assert signed(shots)[1] < 0 < signed(shots)[0]


@pytest.mark.parametrize("decoder", DECODERS)
def test_complementary_gap_properties_low_p(decoder):
    _require_decoder(decoder)
    dem = _line_dem(0.01)
    gap = complementary_gap_function(dem, decoder=decoder)
    signed = complementary_gap_function(dem, decoder=decoder, sign=True)
    det, _ = sample_dem(dem, 100 if decoder == "tesseract" else 2000, seed=3)
    g, s = gap(det), signed(det)
    assert np.all(g >= 0)
    np.testing.assert_allclose(np.abs(s), g, rtol=1e-6, atol=1e-9)
    # Trivial syndrome at low p: flipping the logical costs a full logical
    # operator's worth of weight, 6 * log(0.99/0.01) here.
    trivial_gap = gap(np.zeros((1, dem.num_detectors), dtype=np.uint8))[0]
    assert trivial_gap == pytest.approx(6 * np.log(0.99 / 0.01), rel=1e-5)
    assert trivial_gap > 5


def test_complementary_gap_requires_single_observable():
    undecorated = dem_from_str("error(0.01) D0\nerror(0.01) D0 D1\n")
    with pytest.raises(ValueError, match="observable"):
        complementary_gap_function(undecorated)
    two_obs = dem_from_str("error(0.01) D0 L0\nerror(0.01) D0 D1 L1\n")
    with pytest.raises(ValueError, match="observable"):
        complementary_gap_function(two_obs)


def test_complementary_gap_requires_logical_flipping_event():
    dem = dem_from_str("error(0.01) D0\nlogical_observable L0\n")
    assert dem.num_observables == 1
    with pytest.raises(ValueError, match="flips the logical"):
        complementary_gap_function(dem)


def test_complementary_gap_bulk_logical_edge():
    # An event flipping two detectors AND the logical gives a weight-3 column
    # in the augmented check matrix: pymatching must refuse, tesseract
    # decodes the hyperedge natively.
    dem = dem_from_str("""
    error(0.05) D0
    error(0.05) D1
    error(0.05) D0 D1 L0
    """)
    with pytest.raises(ValueError, match="augmentation"):
        complementary_gap_function(dem, decoder="pymatching")

    pytest.importorskip("tesseract_decoder")
    gap = complementary_gap_function(dem, decoder="tesseract")
    shots = np.array([[0, 0], [1, 1]], dtype=np.uint8)
    w = np.log(0.95 / 0.05)
    # [0,0]: class1 needs all three events -> gap 3w.
    # [1,1]: class0 {D0}{D1} = 2w vs class1 {D0 D1 L0} = w -> gap w.
    np.testing.assert_allclose(gap(shots), [3 * w, w], rtol=1e-6)


# ---------------------------------------------------------------------------
# Logical-error-rate consistency test
# ---------------------------------------------------------------------------

def test_decode_logical_predictions_validation():
    undecorated = dem_from_str("error(0.01) D0\n")
    shots = np.zeros((2, 1), dtype=np.uint8)
    with pytest.raises(ValueError, match="observables"):
        decode_logical_predictions(undecorated, shots)
    with pytest.raises(ValueError, match="decoder"):
        decode_logical_predictions(_hand_dem(), np.zeros((2, 2), dtype=np.uint8),
                                   decoder="unionfind")


def test_logical_error_rate_input_validation():
    dem = _line_dem(0.12)
    det, obs = sample_dem(dem, 10, seed=0)
    with pytest.raises(ValueError, match="one row per shot"):
        logical_error_rate_test(dem, det, obs[:5])
    with pytest.raises(ValueError, match="observables"):
        logical_error_rate_test(dem, det, np.zeros((10, 3), dtype=np.uint8))
    with pytest.raises(ValueError, match="predicted_ler"):
        logical_error_rate_test(dem, det, obs, predicted_ler=1.5)
    undecorated = dem_from_str("error(0.01) D0\n")
    with pytest.raises(ValueError, match="decorated"):
        logical_error_rate_test(undecorated, np.zeros((3, 1), dtype=np.uint8),
                                np.zeros(3, dtype=np.uint8))


def test_logical_error_rate_calibration():
    # Data sampled from the DEM itself: across repetitions the p-values must
    # not concentrate below 0.05 (allowing binomial slack on 20 draws).
    dem = _line_dem(0.12)
    pvals = []
    for rep in range(20):
        det, obs = sample_dem(dem, 2000, seed=100 + rep)
        result = logical_error_rate_test(dem, det, obs, num_mc_shots=20000,
                                         seed=999 + rep)
        assert result.null_model == "dem"
        assert result.num_shots == 2000
        assert result.details["predicted_source"] == "dem_monte_carlo"
        pvals.append(result.pvalue)
    pvals = np.array(pvals)
    assert np.mean(pvals < 0.05) <= 0.25
    assert pvals.max() > 0.3


@pytest.mark.parametrize("decoder", DECODERS)
def test_logical_error_rate_power(decoder):
    # Data from a DEM with substantially inflated error rates is rejected.
    _require_decoder(decoder)
    dem = _line_dem(0.12)
    num_shots, num_mc = (800, 4000) if decoder == "tesseract" else (4000, 20000)
    det, obs = sample_dem(_line_dem(0.30), num_shots, seed=5)
    result = logical_error_rate_test(dem, det, obs, decoder=decoder,
                                     num_mc_shots=num_mc, seed=6)
    assert result.pvalue < 1e-6
    assert result.effect_size > 2  # observed/predicted LER ratio
    assert result.details["ler_observed"] > result.details["ler_predicted"]
    assert "ratio" in result.effect_description
    assert result.details["decoder"] == decoder


@pytest.mark.parametrize("decoder", DECODERS)
def test_logical_error_rate_calibrated_smoke(decoder):
    _require_decoder(decoder)
    dem = _line_dem(0.12)
    det, obs = sample_dem(dem, 1000, seed=42)
    result = logical_error_rate_test(dem, det, obs, decoder=decoder,
                                     num_mc_shots=4000, seed=43)
    assert result.pvalue > 1e-3
    assert result.name == f"logical_error_rate[{decoder}]"


def test_predicted_ler_exact_match():
    dem = _line_dem(0.12)
    det, obs = sample_dem(dem, 2000, seed=21)
    preds = decode_logical_predictions(dem, det)
    ler = float(np.mean(np.any(preds != obs, axis=1)))
    result = logical_error_rate_test(dem, det, obs, predicted_ler=ler)
    assert result.details["predicted_source"] == "predicted_ler"
    assert result.pvalue > 0.9
    assert result.effect_size == pytest.approx(1.0)


def test_predicted_ler_wrong_small_uses_exact_binomial():
    dem = _line_dem(0.12)
    det, obs = sample_dem(dem, 2000, seed=22)
    result = logical_error_rate_test(dem, det, obs, predicted_ler=1e-4)
    # Expected failures 0.2 << 10: documented switch to the exact binomial.
    assert result.details["test_method"] == "binomial_exact"
    assert result.pvalue < 1e-6
    assert result.effect_size > 10


def test_ler_estimator_hook():
    # The estimator hook is THE extension point for external LER estimators;
    # exercise both return conventions and check it receives the DEM.
    dem = _line_dem(0.12)
    det, obs = sample_dem(dem, 2000, seed=23)
    calls = []

    def estimator_with_stderr(model):
        calls.append(model)
        return (0.013, 0.003)

    result = logical_error_rate_test(dem, det, obs,
                                     ler_estimator=estimator_with_stderr)
    assert calls == [dem]
    assert result.details["predicted_source"] == "ler_estimator"
    assert result.details["ler_predicted"] == pytest.approx(0.013)
    assert result.details["predicted_stderr"] == pytest.approx(0.003)
    assert result.pvalue > 0.01  # stderr widens the null; roughly consistent

    result_flat = logical_error_rate_test(dem, det, obs,
                                          ler_estimator=lambda model: 0.013)
    assert result_flat.details["predicted_stderr"] == 0.0
    # predicted_ler takes priority over the estimator when both are given.
    result_prio = logical_error_rate_test(
        dem, det, obs, predicted_ler=0.013,
        ler_estimator=lambda model: pytest.fail("estimator should be unused"))
    assert result_prio.details["predicted_source"] == "predicted_ler"


def test_single_shot():
    dem = _line_dem(0.12)
    result = logical_error_rate_test(
        dem, np.zeros((1, 5), dtype=np.uint8), np.zeros(1, dtype=np.uint8),
        predicted_ler=0.5)
    assert result.num_shots == 1
    assert result.details["test_method"] == "binomial_exact"
    assert result.pvalue == pytest.approx(1.0)


def test_zero_rate_guards():
    dem = _line_dem(0.12)
    det = np.zeros((50, 5), dtype=np.uint8)
    zeros = np.zeros(50, dtype=np.uint8)
    ones = np.ones(50, dtype=np.uint8)

    # Observed 0 vs predicted 0: perfectly consistent, no division by zero.
    r = logical_error_rate_test(dem, det, zeros, predicted_ler=0.0)
    assert r.pvalue == 1.0 and r.effect_size == pytest.approx(1.0)

    # Observed > 0 vs predicted 0: decisive rejection, ratio +inf.
    r = logical_error_rate_test(dem, det, ones, predicted_ler=0.0)
    assert r.pvalue == 0.0
    assert np.isinf(r.effect_size)
    assert "inf" in r.effect_description

    # Observed 0 against a Monte Carlo prediction: exact-test branch.
    r = logical_error_rate_test(dem, det, zeros, num_mc_shots=2000, seed=1)
    assert r.details["test_method"] == "fisher_exact"
    assert 0.0 <= r.pvalue <= 1.0
    assert r.effect_size == pytest.approx(0.0)


def test_mc_seed_reproducibility():
    dem = _line_dem(0.12)
    det, obs = sample_dem(dem, 500, seed=31)
    r1 = logical_error_rate_test(dem, det, obs, num_mc_shots=5000, seed=77)
    r2 = logical_error_rate_test(dem, det, obs, num_mc_shots=5000, seed=77)
    assert r1.pvalue == r2.pvalue
    assert r1.details["predicted_failures"] == r2.details["predicted_failures"]


# ---------------------------------------------------------------------------
# scalar_distribution_test wrappers (post-merge integration)
# ---------------------------------------------------------------------------

_needs_engine = pytest.mark.skipif(
    not hasattr(validation, "scalar_distribution_test"),
    reason="scalar_distribution_test not yet merged into validation.py",
)


@_needs_engine
def test_matching_weight_test_wrapper():
    dem = _line_dem(0.12)
    det, _ = sample_dem(dem, 3000, seed=51)
    result = matching_weight_test(dem, det, seed=52)
    assert result.name == "scalar[matching_weight[pymatching]]"
    assert 0.0 <= result.pvalue <= 1.0
    assert result.num_shots == 3000


@_needs_engine
def test_complementary_gap_test_wrapper():
    dem = _line_dem(0.12)
    det, _ = sample_dem(dem, 3000, seed=53)
    result = complementary_gap_test(dem, det, seed=54)
    assert result.name == "scalar[complementary_gap[pymatching]]"
    assert 0.0 <= result.pvalue <= 1.0

    # The gap distribution should flag data from a much noisier device.
    det_bad, _ = sample_dem(_line_dem(0.30), 3000, seed=55)
    result_bad = complementary_gap_test(dem, det_bad, seed=56)
    assert result_bad.pvalue < result.pvalue or result_bad.pvalue < 0.01

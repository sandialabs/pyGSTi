import math

import numpy as np
import pytest
import scipy.stats
import stim

from pygsti.extras.sparsedem.validation import (
    MAX_MARGINAL_SIZE,
    ValidationResult,
    ValidationSuiteResult,
    marginal_distribution,
    sample_dem,
)
from pygsti.extras.sparsedem.validation import (
    all_weight_k_subsets,
    build_marginal_subsets,
    detector_graph,
    distant_subsets,
    graph_neighborhood_subsets,
    marginal_likelihood_test,
    random_subsets,
    run_marginal_tests,
    space_slice_subsets,
    spacetime_ball_subsets,
    time_column_subsets,
    _bfs_ball,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

SMALL_DEM_STR = """
error(0.10) D0
error(0.15) D1
error(0.20) D2
error(0.05) D0 D1
error(0.05) D1 D2
"""


def small_dem():
    return stim.DetectorErrorModel(SMALL_DEM_STR)


def chain_dem(n=12, p=0.02):
    """DEM whose detector graph is a path 0 - 1 - ... - (n-1)."""
    text = "\n".join(f"error({p}) D{i} D{i + 1}" for i in range(n - 1))
    return stim.DetectorErrorModel(text)


def rep_code_circuit(rounds=5, distance=5):
    return stim.Circuit.generated(
        "repetition_code:memory", rounds=rounds, distance=distance,
        before_round_data_depolarization=0.01,
        before_measure_flip_probability=0.01,
    )


def error_only_dem(circuit):
    """Flattened DEM keeping only error instructions (io.dem_to_dict does not
    digest repeat blocks or detector-coordinate instructions)."""
    flat = circuit.detector_error_model().flattened()
    return stim.DetectorErrorModel(
        "\n".join(str(inst) for inst in flat if inst.type == "error")
    )


def assert_valid_subsets(subsets, num_detectors=None):
    seen = set()
    for s in subsets:
        assert isinstance(s, tuple)
        assert 1 <= len(s) <= MAX_MARGINAL_SIZE
        assert list(s) == sorted(set(s))
        assert s not in seen
        seen.add(s)
        if num_detectors is not None:
            assert all(0 <= d < num_detectors for d in s)


# ---------------------------------------------------------------------------
# all_weight_k_subsets
# ---------------------------------------------------------------------------

def test_all_weight_k_full_enumeration():
    subsets = all_weight_k_subsets(6, 2)
    assert len(subsets) == math.comb(6, 2)
    assert_valid_subsets(subsets, 6)
    assert all(len(s) == 2 for s in subsets)
    assert subsets[0] == (0, 1)


def test_all_weight_k_sampling_moderate():
    # C(10,4) = 210 > 50 but small enough to enumerate-and-sample.
    with pytest.warns(UserWarning, match="exceed max_subsets"):
        subsets = all_weight_k_subsets(10, 4, max_subsets=50, seed=7)
    assert len(subsets) == 50
    assert_valid_subsets(subsets, 10)
    assert all(len(s) == 4 for s in subsets)
    with pytest.warns(UserWarning):
        again = all_weight_k_subsets(10, 4, max_subsets=50, seed=7)
    assert again == subsets


def test_all_weight_k_astronomical():
    # C(200,10) ~ 2.2e16: must not materialize the combinatorial list.
    with pytest.warns(UserWarning, match="uniform random sample"):
        subsets = all_weight_k_subsets(200, 10, max_subsets=100, seed=3)
    assert len(subsets) == 100
    assert_valid_subsets(subsets, 200)
    assert all(len(s) == 10 for s in subsets)


def test_all_weight_k_validation():
    with pytest.raises(ValueError, match="MAX_MARGINAL_SIZE"):
        all_weight_k_subsets(100, MAX_MARGINAL_SIZE + 1)
    with pytest.raises(ValueError):
        all_weight_k_subsets(3, 5)
    with pytest.raises(ValueError):
        all_weight_k_subsets(3, 0)


# ---------------------------------------------------------------------------
# random_subsets
# ---------------------------------------------------------------------------

def test_random_subsets_sizes_and_seed():
    subsets = random_subsets(50, 5, 40, seed=11, min_size=2)
    assert len(subsets) == 40
    assert_valid_subsets(subsets, 50)
    sizes = {len(s) for s in subsets}
    assert sizes <= {2, 3, 4, 5}
    assert random_subsets(50, 5, 40, seed=11, min_size=2) == subsets


def test_random_subsets_fixed_size():
    subsets = random_subsets(30, 4, 25, seed=0, min_size=4)
    assert all(len(s) == 4 for s in subsets)
    assert len(subsets) == 25


def test_random_subsets_exhaustion_warns():
    # Only C(4,2) = 6 distinct subsets exist.
    with pytest.warns(UserWarning, match="distinct subsets"):
        subsets = random_subsets(4, 2, 10, seed=1, min_size=2)
    assert len(subsets) == 6
    assert_valid_subsets(subsets, 4)


def test_random_subsets_validation():
    with pytest.raises(ValueError, match="min_size"):
        random_subsets(10, 3, 5, min_size=4)
    with pytest.raises(ValueError, match="MAX_MARGINAL_SIZE"):
        random_subsets(100, MAX_MARGINAL_SIZE + 1, 5)


# ---------------------------------------------------------------------------
# detector graph and graph-based builders
# ---------------------------------------------------------------------------

def test_detector_graph():
    dem = stim.DetectorErrorModel("""
        error(0.1) D0 D1
        error(0.1) D1 D2
        error(0.1) D3
    """)
    adj = detector_graph(dem)
    assert adj == {0: {1}, 1: {0, 2}, 2: {1}, 3: set()}
    # dict input matches DEM input.
    from pygsti.extras.sparsedem.io import dem_to_dict
    assert detector_graph(dem_to_dict(dem)) == adj


def test_graph_neighborhood_subsets():
    dem = chain_dem(8)
    subsets = graph_neighborhood_subsets(dem, radius=1)
    assert_valid_subsets(subsets, 8)
    assert (0, 1) in subsets            # ball around 0
    assert (0, 1, 2) in subsets         # ball around 1
    # Deduplication: 8 centers but interior balls are distinct.
    assert len(subsets) == len(set(subsets))
    # radius=2 around a single center.
    subsets2 = graph_neighborhood_subsets(dem, radius=2, centers=[3])
    assert subsets2 == [(1, 2, 3, 4, 5)]


def test_graph_neighborhood_truncation():
    dem = chain_dem(15)
    with pytest.warns(UserWarning, match="truncated"):
        subsets = graph_neighborhood_subsets(dem, radius=4, centers=[7],
                                             max_size=5)
    (subset,) = subsets
    assert len(subset) == 5
    # Graph-closest to 7: 7 (d0), then 6, 8 (d1), then 5, 9 (d2).
    assert subset == (5, 6, 7, 8, 9)


def test_distant_subsets():
    n = 12
    dem = chain_dem(n)
    subsets = distant_subsets(dem, size=3, num_subsets=5, min_distance=3,
                              seed=5)
    assert 1 <= len(subsets) <= 5
    assert_valid_subsets(subsets, n)
    adj = detector_graph(dem)
    for s in subsets:
        for a in s:
            ball = _bfs_ball(adj, a, 2)  # distance <= min_distance - 1
            for b in s:
                if b != a:
                    assert b not in ball
    # On a chain, graph distance is index distance.
    for s in subsets:
        assert min(np.diff(s)) >= 3


def test_distant_subsets_unsatisfiable():
    dem = chain_dem(12)
    with pytest.raises(ValueError, match="unsatisfiable"):
        distant_subsets(dem, size=5, num_subsets=3, min_distance=10, seed=0)
    # Satisfiable but scarce: warns and returns fewer than requested.
    with pytest.warns(UserWarning, match="were found"):
        few = distant_subsets(dem, size=3, num_subsets=50, min_distance=5,
                              seed=0)
    assert 1 <= len(few) < 50


# ---------------------------------------------------------------------------
# Circuit-based spacetime builders
# ---------------------------------------------------------------------------

def test_time_column_subsets():
    circuit = rep_code_circuit(rounds=5, distance=5)  # 24 dets, 4 cols x 6 t
    coords = circuit.get_detector_coordinates()
    subsets = time_column_subsets(circuit)
    assert len(subsets) == 4
    assert_valid_subsets(subsets, circuit.num_detectors)
    for s in subsets:
        assert len(s) == 6
        # All members share the spatial (first) coordinate.
        assert len({coords[d][0] for d in s}) == 1


def test_time_column_subsets_window():
    circuit = rep_code_circuit(rounds=5, distance=5)
    coords = circuit.get_detector_coordinates()
    subsets = time_column_subsets(circuit, window=2)
    assert len(subsets) == 4 * 5  # 4 columns x (6 - 2 + 1) window positions
    for s in subsets:
        assert len(s) == 2
        times = sorted(coords[d][-1] for d in s)
        assert times[1] - times[0] == 1


def test_space_slice_subsets():
    circuit = rep_code_circuit(rounds=5, distance=5)
    coords = circuit.get_detector_coordinates()
    subsets = space_slice_subsets(circuit)
    assert len(subsets) == 6
    for s in subsets:
        assert len(s) == 4
        assert len({coords[d][-1] for d in s}) == 1
    subsets2 = space_slice_subsets(circuit, window=2)
    assert len(subsets2) == 5
    assert all(len(s) == 8 for s in subsets2)


def test_space_slice_split_oversized():
    # distance=30 -> 29 detectors per round > MAX_MARGINAL_SIZE.
    circuit = rep_code_circuit(rounds=2, distance=30)
    coords = circuit.get_detector_coordinates()
    with pytest.warns(UserWarning, match="splitting"):
        subsets = space_slice_subsets(circuit)
    assert_valid_subsets(subsets, circuit.num_detectors)
    # Each time slice is fully covered by its chunks.
    times = sorted({c[-1] for c in coords.values()})
    for t in times:
        slice_dets = {d for d, c in coords.items() if c[-1] == t}
        covered = set()
        for s in subsets:
            if set(s) <= slice_dets:
                covered |= set(s)
        assert covered == slice_dets


def test_spacetime_ball_subsets():
    circuit = rep_code_circuit(rounds=5, distance=5)
    subsets = spacetime_ball_subsets(circuit, space_radius=2, time_radius=1,
                                     centers=[5])
    # Detector 5 sits at [3, 1]; the ball is x in {1,3,5}, t in {0,1,2}.
    assert subsets == [(0, 1, 2, 4, 5, 6, 8, 9, 10)]
    with pytest.warns(UserWarning, match="truncated"):
        trunc = spacetime_ball_subsets(circuit, space_radius=2, time_radius=1,
                                       centers=[5], max_size=5)
    assert trunc == [(1, 4, 5, 6, 9)]  # closest by (space dist, |dt|, index)


def test_spacetime_ball_default_centers_dedup():
    circuit = rep_code_circuit(rounds=5, distance=5)
    subsets = spacetime_ball_subsets(circuit, space_radius=0, time_radius=100)
    # One ball per spatial column (identical across centers in a column).
    assert len(subsets) == 4
    assert all(len(s) == 6 for s in subsets)


def test_time_axis_override():
    circuit = rep_code_circuit(rounds=5, distance=5)
    # With time_axis=0, "space" becomes the round coordinate: 6 columns of 4.
    subsets = time_column_subsets(circuit, time_axis=0)
    assert len(subsets) == 6
    assert all(len(s) == 4 for s in subsets)


# ---------------------------------------------------------------------------
# build_marginal_subsets front door
# ---------------------------------------------------------------------------

def test_build_marginal_subsets_dispatch():
    dem = chain_dem(10)
    circuit = rep_code_circuit(rounds=5, distance=5)
    assert build_marginal_subsets("all_weight_k", num_detectors=6, k=2) == \
        all_weight_k_subsets(6, 2)
    assert build_marginal_subsets("random", num_detectors=20, k=3,
                                  num_subsets=5, seed=2) == \
        random_subsets(20, 3, 5, seed=2)
    assert build_marginal_subsets("neighborhood", dem=dem, radius=1) == \
        graph_neighborhood_subsets(dem, radius=1)
    assert build_marginal_subsets("distant", dem=dem, size=2, num_subsets=3,
                                  min_distance=3, seed=4) == \
        distant_subsets(dem, size=2, num_subsets=3, min_distance=3, seed=4)
    assert build_marginal_subsets("time", circuit=circuit) == \
        time_column_subsets(circuit)
    assert build_marginal_subsets("space", circuit=circuit) == \
        space_slice_subsets(circuit)
    assert build_marginal_subsets("spacetime", circuit=circuit,
                                  space_radius=2, time_radius=1,
                                  centers=[5]) == \
        spacetime_ball_subsets(circuit, 2, 1, centers=[5])


def test_build_marginal_subsets_infers_num_detectors():
    dem = chain_dem(10)
    subsets = build_marginal_subsets("all_weight_k", dem=dem, k=2)
    assert len(subsets) == math.comb(10, 2)
    circuit = rep_code_circuit(rounds=5, distance=5)
    subsets = build_marginal_subsets("random", circuit=circuit, k=2,
                                     num_subsets=5, seed=0)
    assert_valid_subsets(subsets, circuit.num_detectors)


def test_build_marginal_subsets_errors():
    with pytest.raises(ValueError, match="Unknown method"):
        build_marginal_subsets("bogus")
    with pytest.raises(ValueError, match="circuit"):
        build_marginal_subsets("time")
    with pytest.raises(ValueError, match="detector graph"):
        build_marginal_subsets("neighborhood")
    with pytest.raises(ValueError, match="num_detectors"):
        build_marginal_subsets("all_weight_k", k=2)


# ---------------------------------------------------------------------------
# marginal_likelihood_test
# ---------------------------------------------------------------------------

def test_marginal_test_on_model_data():
    dem = small_dem()
    det, _ = sample_dem(dem, 5000, seed=1234)
    res = marginal_likelihood_test(dem, det, (0, 1, 2))
    assert isinstance(res, ValidationResult)
    assert res.name == "marginal_g[0,1,2]"
    assert res.num_shots == 5000
    assert res.null_model == "dem"
    assert 0.0 <= res.pvalue <= 1.0
    assert res.pvalue > 1e-4          # model data should not be rejected hard
    assert 0.0 <= res.effect_size <= 1.0
    assert res.effect_size < 0.05     # TVD small on model data
    assert "detectors (0, 1, 2)" in res.effect_description
    assert "observed" in res.effect_description
    assert res.details["counts"].sum() == 5000
    assert res.details["method"] == "asymptotic"
    assert len(res.details["std_residuals"]) == 8


def test_marginal_test_calibration():
    # On data sampled from the model, p-values are ~uniform.
    dem = small_dem()
    pvals = []
    for rep in range(100):
        det, _ = sample_dem(dem, 2000, seed=10_000 + rep)
        pvals.append(marginal_likelihood_test(dem, det, (0, 1, 2)).pvalue)
    pvals = np.array(pvals)
    frac_small = np.mean(pvals < 0.05)
    # Binomial(100, 0.05) rarely exceeds 12 successes.
    assert frac_small <= 0.12
    assert scipy.stats.kstest(pvals, "uniform").pvalue > 1e-3


def test_marginal_test_power_doubled_event():
    model = small_dem()
    truth = stim.DetectorErrorModel(
        SMALL_DEM_STR.replace("error(0.05) D0 D1", "error(0.25) D0 D1")
    )
    det, _ = sample_dem(truth, 5000, seed=99)
    res = marginal_likelihood_test(model, det, (0, 1, 2))
    assert res.pvalue < 1e-8
    assert res.effect_size > 0.02


def test_marginal_test_power_removed_event():
    model = stim.DetectorErrorModel(SMALL_DEM_STR + "error(0.10) D0 D2\n")
    truth = small_dem()  # data lack the D0 D2 correlation
    det, _ = sample_dem(truth, 5000, seed=100)
    res = marginal_likelihood_test(model, det, (0, 1, 2))
    assert res.pvalue < 1e-6


def test_bootstrap_agrees_with_asymptotic_at_large_n():
    dem = small_dem()
    det, _ = sample_dem(dem, 20000, seed=42)
    asym = marginal_likelihood_test(dem, det, (0, 1, 2))
    boot = marginal_likelihood_test(dem, det, (0, 1, 2), bootstrap=3000,
                                    seed=7)
    assert boot.details["method"] == "bootstrap"
    assert boot.details["num_bootstrap"] == 3000
    assert abs(boot.pvalue - asym.pvalue) < 0.08
    assert boot.statistic == pytest.approx(asym.statistic)


def test_bootstrap_small_n():
    dem = small_dem()
    det, _ = sample_dem(dem, 30, seed=8)
    res = marginal_likelihood_test(dem, det, (0, 1, 2), bootstrap=500, seed=9)
    assert 0.0 < res.pvalue <= 1.0
    assert res.num_shots == 30
    # Reproducible with the same seed.
    res2 = marginal_likelihood_test(dem, det, (0, 1, 2), bootstrap=500, seed=9)
    assert res2.pvalue == res.pvalue


def test_bootstrap_detects_gross_mismatch():
    model_dict = {1: 0.01}  # detector 0 fires 1% of the time
    samples = np.zeros((200, 1), dtype=np.uint8)
    samples[:100, 0] = 1    # observed 50% firing rate
    res = marginal_likelihood_test(model_dict, samples, (0,), bootstrap=999,
                                   seed=0)
    assert res.pvalue == pytest.approx(1.0 / 1000.0)


def test_single_shot_and_empty():
    dem = small_dem()
    det = np.array([[1, 0, 1]], dtype=np.uint8)
    res = marginal_likelihood_test(dem, det, (0, 1, 2))
    assert res.num_shots == 1
    assert 0.0 <= res.pvalue <= 1.0
    with pytest.raises(ValueError, match="at least one shot"):
        marginal_likelihood_test(dem, np.zeros((0, 3), dtype=np.uint8),
                                 (0, 1, 2))


def test_effect_description_points_at_worst_cell():
    model_dict = {1: 0.01}
    samples = np.zeros((100, 1), dtype=np.uint8)
    samples[:50, 0] = 1
    res = marginal_likelihood_test(model_dict, samples, (0,))
    assert res.pvalue < 1e-10
    assert res.effect_size == pytest.approx(0.49)
    # Both cells are ~49 sigma off (k=1 residuals tie in magnitude); the
    # description must point at one of them with correct numbers.
    assert res.effect_description in (
        "outcome 0 on detectors (0,): observed 50, expected 99 (-49.2 sigma)",
        "outcome 1 on detectors (0,): observed 50, expected 1 (+49.2 sigma)",
    )


def test_marginal_test_matches_exact_distribution():
    # Sanity: with counts exactly proportional to the model marginal,
    # the G statistic is ~0 and p ~ 1.
    dem_dict = {1: 0.25, 2: 0.25}
    probs = marginal_distribution(dem_dict, (0, 1))
    n = 1600
    counts = np.round(probs * n).astype(int)
    samples = []
    for outcome, c in enumerate(counts):
        row = [(outcome >> j) & 1 for j in range(2)]
        samples.extend([row] * c)
    samples = np.array(samples, dtype=np.uint8)
    res = marginal_likelihood_test(dem_dict, samples, (0, 1))
    assert res.statistic < 1e-6
    assert res.pvalue > 0.999


# ---------------------------------------------------------------------------
# run_marginal_tests
# ---------------------------------------------------------------------------

def test_run_marginal_tests_suite():
    circuit = rep_code_circuit(rounds=5, distance=5)
    dem = error_only_dem(circuit)
    det, _ = sample_dem(dem, 5000, seed=77)
    subsets = graph_neighborhood_subsets(dem, radius=1)[:10] + \
        [(0, 4), (3, 7, 11)]
    suite = run_marginal_tests(dem, det, subsets, seed=1)
    assert isinstance(suite, ValidationSuiteResult)
    assert len(suite.results) == len(subsets)
    assert all(r.name.startswith("marginal_g[") for r in suite.results)
    assert all(r.num_shots == 5000 for r in suite.results)
    # Model data: nothing (or nearly nothing) rejected after FDR correction.
    assert len(suite.rejected(alpha=0.05)) <= 1
    assert "tests" in suite.summary()


def test_run_marginal_tests_rejects_bad_model():
    circuit = rep_code_circuit(rounds=5, distance=5)
    truth = error_only_dem(circuit)
    det, _ = sample_dem(truth, 5000, seed=78)
    # Corrupt the model: halve every probability.
    bad_lines = []
    for inst in truth:
        p = inst.args_copy()[0]
        targets = " ".join(str(t) for t in inst.targets_copy())
        bad_lines.append(f"error({p / 2}) {targets}")
    bad = stim.DetectorErrorModel("\n".join(bad_lines))
    subsets = graph_neighborhood_subsets(bad, radius=1)[:10]
    suite = run_marginal_tests(bad, det, subsets)
    assert len(suite.rejected(alpha=0.05)) >= len(subsets) // 2


def test_run_marginal_tests_empty_and_bootstrap_seeded():
    dem = small_dem()
    det, _ = sample_dem(dem, 500, seed=3)
    empty = run_marginal_tests(dem, det, [])
    assert empty.results == []
    s1 = run_marginal_tests(dem, det, [(0, 1), (1, 2)], bootstrap=300, seed=5)
    s2 = run_marginal_tests(dem, det, [(0, 1), (1, 2)], bootstrap=300, seed=5)
    assert [r.pvalue for r in s1.results] == [r.pvalue for r in s2.results]

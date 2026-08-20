import numpy as np
import pytest
import stim

from pygsti.extras.sparsedem.io import dem_from_str, dem_to_dict
from pygsti.extras.sparsedem.logical_decoration import (
    dem_to_check_matrix,
    decorate_dem_with_logical_flags,
    solve_gf2_robust,
    build_matcher,
    decode_event_indicators,
    assign_logical_flags,
)

pymatching = pytest.importorskip("pymatching")


def _random_solvable_system(rng, n_shots, n_events, density=0.3):
    L_true = rng.integers(0, 2, size=n_events).astype(np.uint8)
    B = (rng.random((n_shots, n_events)) < density).astype(np.uint8)
    # Guarantee full rank by embedding an identity block.
    B[:n_events, :] = np.eye(n_events, dtype=np.uint8)
    Y = (B @ L_true.astype(np.int64)) % 2
    return B, Y.astype(np.uint8), L_true


def test_dem_to_check_matrix():
    dem = dem_from_str("""
    error(0.01) D0
    error(0.02) D0 D2
    error(0.03) D1 D2
    """)
    H, probs, masks = dem_to_check_matrix(dem)
    assert list(masks) == [1, 5, 6]
    assert np.allclose(probs, [0.01, 0.02, 0.03])
    expected = np.array([
        [1, 1, 0],
        [0, 0, 1],
        [0, 1, 1],
    ], dtype=np.uint8)
    assert np.array_equal(H, expected)


def test_dem_to_check_matrix_extra_detectors():
    dem = dem_from_str("error(0.01) D0 D1\n")
    H, _, _ = dem_to_check_matrix(dem, num_detectors=4)
    assert H.shape == (4, 1)
    assert np.array_equal(H[:, 0], [1, 1, 0, 0])


def test_decorate_dem_with_logical_flags():
    dem = dem_from_str("""
    error(0.01) D0
    error(0.02) D0 D2
    error(0.03) D1 D2
    """)
    decorated = decorate_dem_with_logical_flags(dem, [1, 0, 1])
    assert decorated.num_observables == 1
    # Probabilities and detector targets are unchanged.
    assert dem_to_dict(decorated) == dem_to_dict(dem)
    flagged = []
    for event in decorated:
        targets = event.targets_copy()
        has_logical = any(t.is_logical_observable_id() for t in targets)
        detector_mask = sum(1 << t.val for t in targets if t.is_relative_detector_id())
        if has_logical:
            flagged.append(detector_mask)
    assert sorted(flagged) == [1, 6]


def test_decorate_flag_length_mismatch():
    dem = dem_from_str("error(0.01) D0\n")
    with pytest.raises(ValueError):
        decorate_dem_with_logical_flags(dem, [1, 0])


def test_solve_gf2_exact():
    rng = np.random.default_rng(0)
    B, Y, L_true = _random_solvable_system(rng, 200, 12)
    L, residual, diag = solve_gf2_robust(B, Y)
    assert np.array_equal(L, L_true)
    assert residual == 0.0
    assert diag["rank"] == 12
    assert diag["undetermined_indices"] == []
    assert diag["method"] == "gaussian_elimination"
    assert diag["converged"]


def test_solve_gf2_with_corrupted_rows():
    rng = np.random.default_rng(1)
    B, Y, L_true = _random_solvable_system(rng, 500, 10)
    # Corrupt a few rows, including early rows that naive Gaussian elimination
    # is likely to use as pivots.
    Y = Y.copy()
    for i in [0, 1, 250, 400]:
        Y[i] ^= 1
    L, residual, diag = solve_gf2_robust(B, Y, seed=123)
    assert np.array_equal(L, L_true)
    assert residual == pytest.approx(4 / 500)
    assert diag["method"] == "ransac"
    assert diag["ransac_iterations"] >= 1


def test_solve_gf2_underdetermined():
    rng = np.random.default_rng(2)
    n_events = 8
    L_true = rng.integers(0, 2, size=n_events).astype(np.uint8)
    B = (rng.random((100, n_events)) < 0.4).astype(np.uint8)
    B[:, 5] = 0  # event 5 never selected by the decoder
    Y = ((B @ L_true.astype(np.int64)) % 2).astype(np.uint8)
    with pytest.warns(UserWarning, match="undetermined"):
        L, residual, diag = solve_gf2_robust(B, Y)
    assert residual == 0.0
    assert diag["rank"] == n_events - 1
    assert diag["undetermined_indices"] == [5]
    assert diag["null_space"].shape == (1, n_events)
    assert np.array_equal(diag["null_space"][0], np.eye(n_events, dtype=np.uint8)[5])
    determined = [i for i in range(n_events) if i != 5]
    assert np.array_equal(L[determined], L_true[determined])


def test_solve_gf2_high_residual_warns():
    B = np.vstack([np.eye(4, dtype=np.uint8)] * 3)
    # Three copies of each unit row; columns 0-2 have one dissenting copy each,
    # so no L can achieve a residual below 3/12.
    Y = np.zeros(12, dtype=np.uint8)
    Y[0] = Y[5] = Y[10] = 1
    with pytest.warns(UserWarning, match="residual"):
        _, residual, diag = solve_gf2_robust(B, Y, residual_threshold=1e-3, seed=0)
    assert residual == pytest.approx(3 / 12)
    assert not diag["converged"]


def test_build_matcher_rejects_nongraphlike():
    dem = dem_from_str("error(0.01) D0 D1 D2\n")
    with pytest.raises(ValueError, match="graph-like"):
        build_matcher(dem)


def test_decode_event_indicators():
    dem = dem_from_str("""
    error(0.01) D0
    error(0.01) D0 D1
    error(0.01) D1 D2
    error(0.01) D2
    """)
    matcher, masks = build_matcher(dem)
    assert list(masks) == [1, 3, 4, 6]
    syndromes = np.array([
        [0, 0, 0],
        [1, 0, 0],  # -> D0
        [1, 1, 0],  # -> D0 D1
        [0, 1, 1],  # -> D1 D2
    ], dtype=np.uint8)
    B = decode_event_indicators(matcher, syndromes)
    expected = np.array([
        [0, 0, 0, 0],
        [1, 0, 0, 0],  # mask 1 = D0
        [0, 1, 0, 0],  # mask 3 = D0 D1
        [0, 0, 0, 1],  # mask 6 = D1 D2
    ], dtype=np.uint8)
    assert np.array_equal(B, expected)


def _simulate_shots(rng, H, probs, flags, n_shots):
    occurrences = (rng.random((n_shots, len(probs))) < probs).astype(np.uint8)
    syndromes = (occurrences @ H.T) % 2
    y = (occurrences @ flags.astype(np.int64)) % 2
    return syndromes.astype(np.uint8), y.astype(np.uint8)


def test_assign_logical_flags_end_to_end():
    # Line of 5 detectors: boundary (single-detector) events on each detector
    # plus nearest-neighbor pair events. Logical flags form a cut between
    # detectors {0, 1} and {2, 3, 4} + boundary.
    dem_str = ""
    masks, flags = [], []
    for d in range(5):
        dem_str += f"error(0.01) D{d}\n"
        masks.append(1 << d)
        flags.append(1 if d < 2 else 0)
    for d in range(4):
        dem_str += f"error(0.01) D{d} D{d + 1}\n"
        masks.append((1 << d) | (1 << (d + 1)))
        flags.append(1 if d == 1 else 0)
    dem = dem_from_str(dem_str)
    H, probs, sorted_masks = dem_to_check_matrix(dem)
    true_flags = np.array(flags, dtype=np.uint8)[np.argsort(masks)]

    rng = np.random.default_rng(7)
    syndromes, y = _simulate_shots(rng, H, probs, true_flags, 3000)

    decorated, recovered, residual, diag = assign_logical_flags(
        dem, syndromes, y, seed=11
    )
    assert diag["undetermined_indices"] == []
    assert residual < 1e-2
    assert np.array_equal(recovered, true_flags)
    assert decorated.num_observables == 1

    # The decorated DEM predicts the observable flip that matches the flags.
    for event in decorated:
        targets = event.targets_copy()
        detector_mask = sum(1 << t.val for t in targets if t.is_relative_detector_id())
        has_logical = any(t.is_logical_observable_id() for t in targets)
        idx = list(sorted_masks).index(detector_mask)
        assert has_logical == bool(true_flags[idx])


def test_assign_logical_flags_drops_nongraphlike():
    dem = dem_from_str("""
    error(0.01) D0
    error(0.01) D1
    error(0.01) D2
    error(0.01) D0 D1 D2
    """)
    H, probs, sorted_masks = dem_to_check_matrix(dem)
    true_flags = np.array([1, 0, 0, 0], dtype=np.uint8)  # flag on D0 only
    rng = np.random.default_rng(5)
    # Simulate only the graph-like events so the decoder is consistent.
    graphlike = H.sum(axis=0) <= 2
    syndromes, y = _simulate_shots(
        rng, H[:, graphlike], probs[graphlike], true_flags[graphlike], 4000
    )
    with pytest.warns(UserWarning, match="non-graph-like"):
        _, recovered, residual, diag = assign_logical_flags(dem, syndromes, y, seed=3)
    assert residual == 0.0
    assert 7 in diag["dropped_masks"]
    assert 7 in diag["undetermined_masks"]
    assert np.array_equal(recovered[graphlike], true_flags[graphlike])
    assert recovered[~graphlike].sum() == 0

    with pytest.raises(ValueError, match="graph-like"):
        assign_logical_flags(dem, syndromes, y, on_nongraphlike="raise")

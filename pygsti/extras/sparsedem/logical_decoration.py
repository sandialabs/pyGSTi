"""
Decorate a learned (undecorated) detector error model with logical-flip flags.

Given a learned DEM (a set of events, each flipping a subset of detectors,
with estimated probabilities) plus per-shot detector data and per-shot
observed logical outcomes, this module infers, for each DEM event, a binary
flag indicating whether that event flips the logical degree of freedom.

Algorithm (valid in the low-logical-error regime):
  1. Build a minimum-weight-perfect-matching decoder (pymatching) from the
     learned DEM. This requires the DEM to be graph-like: every event flips
     at most two detectors.
  2. Decode each experimental shot. The matching indicates which DEM events
     the decoder believes occurred, giving a binary indicator row b over the
     learned events. Stacking shots gives a matrix B (shots x events).
  3. Each shot has an observed logical outcome y in {0, 1}. Stacking gives Y.
  4. Solve Y = B L over GF(2) for the per-event flag vector L. The system is
     (typically) heavily overdetermined; a small fraction of rows is expected
     to be inconsistent (decoder logical errors, imperfections of the learned
     DEM), so the solve is made robust via a residual check and a
     RANSAC-style loop over row orderings.

Under-determined flags (events never selected by the decoder, or only
selected in combinations that leave a GF(2) null space) are reported
explicitly through the diagnostics rather than silently guessed.
"""

import warnings

import numpy as np
import scipy.sparse
import stim

from .io import dem_to_dict, dem_from_str

try:
    import pymatching as _pymatching
except ImportError:  # pragma: no cover - exercised only without pymatching
    _pymatching = None


def dem_to_check_matrix(dem, num_detectors=None):
    """
    Convert a stim DetectorErrorModel to a detector check matrix.

    Unlike `io.dem_to_matrix` (MSB-first rows), the rows here are in
    detector-index order: H[d, j] = 1 iff event j flips detector d. This
    matches the column ordering of stim detector sample arrays, and the
    format expected by pymatching.

    Parameters:
        dem: stim.DetectorErrorModel
        num_detectors: int, optional
            Number of detector rows. Defaults to dem.num_detectors; may be
            larger if the sample data has more detectors than the DEM touches.

    Returns:
        H: np.ndarray
            (num_detectors x n_events) binary check matrix.
        probs: np.ndarray
            n_events vector of event probabilities.
        masks: np.ndarray
            n_events vector of integer event bitmasks (bit d <-> detector d),
            sorted in increasing order. Columns of H follow this order.
    """
    dem_dict = dem_to_dict(dem)
    if num_detectors is None:
        num_detectors = dem.num_detectors
    if not dem_dict:
        return (
            np.zeros((num_detectors, 0), dtype=np.uint8),
            np.array([], dtype=float),
            np.array([], dtype=np.int64),
        )
    items = sorted(dem_dict.items())
    masks = np.array([m for m, _ in items], dtype=np.int64)
    if masks[-1] >= (1 << num_detectors):
        raise ValueError("DEM events touch detectors beyond num_detectors.")
    probs = np.array([p for _, p in items], dtype=float)
    H = np.zeros((num_detectors, len(items)), dtype=np.uint8)
    for j, mask in enumerate(masks):
        for d in range(num_detectors):
            H[d, j] = (int(mask) >> d) & 1
    return H, probs, masks


def decorate_dem_with_logical_flags(dem, flags, atol=0.0):
    """
    Return a copy of a DEM in which flagged events also flip logical L0.

    Parameters:
        dem: stim.DetectorErrorModel
            Undecorated DEM.
        flags: array-like
            Binary flags, one per event, aligned with the sorted-bitmask
            event order used throughout sparsedem (see `dem_to_check_matrix`).
        atol: float
            Events with probability <= atol are skipped.

    Returns:
        stim.DetectorErrorModel
            DEM with `L0` appended to the targets of each flagged event.
    """
    dem_dict = dem_to_dict(dem)
    items = sorted(dem_dict.items())
    flags = np.asarray(flags, dtype=np.uint8) % 2
    if len(flags) != len(items):
        raise ValueError(
            f"Got {len(flags)} flags for a DEM with {len(items)} distinct events."
        )
    n_bits = dem.num_detectors
    dem_str = ""
    for (mask, p), flag in zip(items, flags):
        if p <= atol:
            continue
        targets = " ".join(f"D{d}" for d in range(n_bits) if (int(mask) >> d) & 1)
        if flag:
            targets += " L0"
        dem_str += f"error({p}) {targets}\n"
    return dem_from_str(dem_str)


def _gf2_eliminate(A, y):
    """
    Fully row-reduce the augmented system [A | y] over GF(2).

    Parameters:
        A: np.ndarray
            (m x n) binary matrix (modified copy is returned).
        y: np.ndarray
            m-vector of binary values.

    Returns:
        R: np.ndarray
            Row-reduced echelon form of A.
        z: np.ndarray
            y after the same row operations.
        pivot_cols: list[int]
            Columns containing pivots; row i has its pivot in pivot_cols[i].
    """
    R = np.array(A, dtype=np.uint8, copy=True) % 2
    z = np.array(y, dtype=np.uint8, copy=True) % 2
    m, n = R.shape
    pivot_cols = []
    r = 0
    for c in range(n):
        if r >= m:
            break
        nonzero = np.nonzero(R[r:, c])[0]
        if len(nonzero) == 0:
            continue
        pr = r + nonzero[0]
        if pr != r:
            R[[r, pr]] = R[[pr, r]]
            z[[r, pr]] = z[[pr, r]]
        elim = R[:, c].astype(bool)
        elim[r] = False
        R[elim] ^= R[r]
        z[elim] ^= z[r]
        pivot_cols.append(c)
        r += 1
    return R, z, pivot_cols


def _gf2_particular_solution(R, z, pivot_cols, n):
    """Extract the free-variables-zero solution from a reduced system."""
    x = np.zeros(n, dtype=np.uint8)
    for i, c in enumerate(pivot_cols):
        x[c] = z[i]
    return x


def _gf2_null_space(R, pivot_cols, n):
    """
    Null space basis of a row-reduced binary matrix.

    Returns an array of shape (n - rank, n); each row v satisfies R v = 0.
    """
    free_cols = [c for c in range(n) if c not in set(pivot_cols)]
    basis = np.zeros((len(free_cols), n), dtype=np.uint8)
    for k, f in enumerate(free_cols):
        basis[k, f] = 1
        for i, c in enumerate(pivot_cols):
            basis[k, c] = R[i, f]
    return basis


def _residual_fraction(B, Y, x):
    """Fraction of rows where (B x) mod 2 != Y."""
    if B.shape[0] == 0:
        return 0.0
    mismatch = ((B @ x.astype(np.int64)) % 2).astype(np.uint8) ^ Y
    return float(np.mean(mismatch))


def solve_gf2_robust(
    B,
    Y,
    residual_threshold=1e-3,
    max_ransac_iterations=200,
    seed=None,
):
    """
    Solve Y = B L over GF(2), robustly against a small fraction of bad rows.

    First computes a candidate solution by GF(2) Gaussian elimination and
    checks the Hamming-weight residual of B L xor Y over all rows. If the
    residual is nonzero (the naive elimination may then have pivoted on a
    corrupted row), a RANSAC-style loop re-solves from random row orderings
    (equivalent to solving from random independent row subsets) and keeps the
    solution minimizing the residual.

    Parameters:
        B: array-like
            (n_shots x n_events) binary indicator matrix.
        Y: array-like
            n_shots binary vector of observed logical outcomes.
        residual_threshold: float
            Fraction of inconsistent rows considered acceptable; used only to
            set the `converged` diagnostic (the minimizer is always returned).
        max_ransac_iterations: int
            Maximum number of RANSAC re-solves.
        seed: int, optional
            Seed for the RANSAC row shuffles.

    Returns:
        L: np.ndarray
            n_events binary flag vector (free variables set to 0; see
            diagnostics['undetermined_indices']).
        residual_fraction: float
            Fraction of rows with (B L) mod 2 != Y at the returned solution.
        diagnostics: dict
            Keys: 'rank', 'n_events', 'n_shots', 'undetermined_indices',
            'null_space', 'initial_residual_fraction', 'ransac_iterations',
            'method', 'converged'.
    """
    B = np.asarray(B, dtype=np.uint8) % 2
    if B.ndim != 2:
        raise ValueError("B must be a 2D array of shape (n_shots, n_events).")
    Y = np.asarray(Y, dtype=np.uint8).ravel() % 2
    n_shots, n_events = B.shape
    if len(Y) != n_shots:
        raise ValueError("Y must have one entry per row of B.")

    R, z, pivot_cols = _gf2_eliminate(B, Y)
    rank = len(pivot_cols)
    null_space = _gf2_null_space(R, pivot_cols, n_events)
    undetermined = sorted(set(range(n_events)) - set(pivot_cols))

    best = _gf2_particular_solution(R, z, pivot_cols, n_events)
    best_residual = _residual_fraction(B, Y, best)
    initial_residual = best_residual
    method = "gaussian_elimination"

    ransac_iterations = 0
    if best_residual > 0 and n_shots > rank and max_ransac_iterations > 0:
        rng = np.random.default_rng(seed)
        method = "ransac"
        for _ in range(max_ransac_iterations):
            ransac_iterations += 1
            perm = rng.permutation(n_shots)
            Rp, zp, pivots_p = _gf2_eliminate(B[perm], Y[perm])
            candidate = _gf2_particular_solution(Rp, zp, pivots_p, n_events)
            residual = _residual_fraction(B, Y, candidate)
            if residual < best_residual:
                best = candidate
                best_residual = residual
            if best_residual == 0:
                break

    diagnostics = {
        "rank": rank,
        "n_events": n_events,
        "n_shots": n_shots,
        "undetermined_indices": undetermined,
        "null_space": null_space,
        "initial_residual_fraction": initial_residual,
        "ransac_iterations": ransac_iterations,
        "method": method,
        "converged": best_residual <= residual_threshold,
    }
    if not diagnostics["converged"]:
        warnings.warn(
            f"solve_gf2_robust: best residual fraction {best_residual:.3g} exceeds "
            f"threshold {residual_threshold:.3g}; flags may be unreliable."
        )
    if undetermined:
        warnings.warn(
            f"solve_gf2_robust: rank(B) = {rank} < {n_events} events; flags for "
            f"event indices {undetermined} are undetermined (reported flag 0)."
        )
    return best, best_residual, diagnostics


def build_matcher(dem, num_detectors=None, min_probability=1e-12):
    """
    Build a pymatching matcher from a graph-like DEM.

    The matcher is configured with an identity faults matrix, so that
    `matcher.decode(syndrome)` returns the per-event indicator vector b
    (which learned events the decoder believes occurred), in the
    sorted-bitmask event order.

    Parameters:
        dem: stim.DetectorErrorModel
            Graph-like DEM (every event flips at most two detectors).
        num_detectors: int, optional
            Number of detectors in the syndrome data (default dem.num_detectors).
        min_probability: float
            Probabilities are clipped to [min_probability, 0.5 - min_probability]
            before conversion to matching weights log((1-p)/p).

    Returns:
        matcher: pymatching.Matching
        masks: np.ndarray
            Integer event bitmasks aligned with decode output columns.
    """
    if _pymatching is None:
        raise ImportError(
            "pymatching is required for logical decoration; pip install pymatching."
        )
    H, probs, masks = dem_to_check_matrix(dem, num_detectors=num_detectors)
    col_weights = H.sum(axis=0)
    if np.any(col_weights > 2):
        bad = masks[col_weights > 2].tolist()
        raise ValueError(
            f"DEM is not graph-like: events {bad} flip more than two detectors."
        )
    p = np.clip(probs, min_probability, 0.5 - min_probability)
    weights = np.log((1.0 - p) / p)
    n_events = H.shape[1]
    matcher = _pymatching.Matching(
        scipy.sparse.csc_matrix(H),
        weights=weights,
        faults_matrix=scipy.sparse.eye(n_events, dtype=np.uint8, format="csc"),
    )
    return matcher, masks


def decode_event_indicators(matcher, detector_samples):
    """
    Decode a batch of shots into per-event indicator rows.

    Parameters:
        matcher: pymatching.Matching
            Matcher built with an identity faults matrix (see `build_matcher`).
        detector_samples: np.ndarray
            (n_shots x n_detectors) binary array; column d is detector d
            (stim sampling convention).

    Returns:
        B: np.ndarray
            (n_shots x n_events) binary indicator matrix.
    """
    shots = np.asarray(detector_samples, dtype=np.uint8) % 2
    return np.asarray(matcher.decode_batch(shots), dtype=np.uint8) % 2


def assign_logical_flags(
    dem,
    detector_samples,
    logical_outcomes,
    num_detectors=None,
    on_nongraphlike="drop",
    residual_threshold=1e-3,
    max_ransac_iterations=200,
    seed=None,
):
    """
    Decorate a learned DEM with logical-flip flags inferred from shot data.

    Builds a pymatching matcher from the learned DEM, decodes every shot to
    obtain the event indicator matrix B, and solves Y = B L over GF(2)
    (robustly, via `solve_gf2_robust`) for the per-event logical flags L.

    Parameters:
        dem: stim.DetectorErrorModel
            Learned, undecorated DEM.
        detector_samples: np.ndarray
            (n_shots x n_detectors) binary detector data; column d is
            detector d (stim sampling convention).
        logical_outcomes: array-like
            n_shots binary vector of observed logical outcomes
            (0 = no logical flip, 1 = logical flip).
        num_detectors: int, optional
            Defaults to detector_samples.shape[1].
        on_nongraphlike: str
            What to do with events flipping more than two detectors:
            'drop' (default; they are excluded from matching and left
            undecorated) or 'raise'.
        residual_threshold: float
            Acceptable fraction of inconsistent rows (see `solve_gf2_robust`).
        max_ransac_iterations: int
            Maximum RANSAC re-solves in the GF(2) solver.
        seed: int, optional
            Seed for the RANSAC row shuffles.

    Returns:
        decorated_dem: stim.DetectorErrorModel
            The input DEM with `L0` appended to flagged events.
        flags: np.ndarray
            Binary flag per event, aligned with diagnostics['masks'].
        residual_fraction: float
            Fraction of shots inconsistent with the returned flags.
        diagnostics: dict
            GF(2) solver diagnostics plus 'masks' (all event bitmasks, sorted),
            'decoded_masks' (events used in matching), 'dropped_masks'
            (non-graph-like events excluded from matching and flag inference),
            and 'undetermined_masks' (events whose flag the data does not
            determine).
    """
    detector_samples = np.asarray(detector_samples, dtype=np.uint8) % 2
    if detector_samples.ndim != 2:
        raise ValueError("detector_samples must be 2D (n_shots x n_detectors).")
    if num_detectors is None:
        num_detectors = detector_samples.shape[1]
    Y = np.asarray(logical_outcomes, dtype=np.uint8).ravel() % 2
    if len(Y) != detector_samples.shape[0]:
        raise ValueError("logical_outcomes must have one entry per shot.")

    H, probs, masks = dem_to_check_matrix(dem, num_detectors=num_detectors)
    col_weights = H.sum(axis=0)
    graphlike = col_weights <= 2
    dropped_masks = masks[~graphlike]
    if len(dropped_masks) > 0:
        if on_nongraphlike == "raise":
            raise ValueError(
                f"DEM is not graph-like: events {dropped_masks.tolist()} flip "
                "more than two detectors."
            )
        if on_nongraphlike != "drop":
            raise ValueError("on_nongraphlike must be 'drop' or 'raise'.")
        warnings.warn(
            f"assign_logical_flags: dropping {len(dropped_masks)} non-graph-like "
            f"event(s) {dropped_masks.tolist()} from matching; they are left "
            "undecorated."
        )

    if _pymatching is None:
        raise ImportError(
            "pymatching is required for logical decoration; pip install pymatching."
        )
    kept = np.nonzero(graphlike)[0]
    p = np.clip(probs[kept], 1e-12, 0.5 - 1e-12)
    matcher = _pymatching.Matching(
        scipy.sparse.csc_matrix(H[:, kept]),
        weights=np.log((1.0 - p) / p),
        faults_matrix=scipy.sparse.eye(len(kept), dtype=np.uint8, format="csc"),
    )
    B = np.asarray(matcher.decode_batch(detector_samples), dtype=np.uint8) % 2

    L_kept, residual_fraction, diagnostics = solve_gf2_robust(
        B,
        Y,
        residual_threshold=residual_threshold,
        max_ransac_iterations=max_ransac_iterations,
        seed=seed,
    )

    flags = np.zeros(len(masks), dtype=np.uint8)
    flags[kept] = L_kept
    decoded_masks = masks[kept]
    diagnostics["masks"] = masks
    diagnostics["decoded_masks"] = decoded_masks
    diagnostics["dropped_masks"] = dropped_masks
    diagnostics["undetermined_masks"] = np.concatenate([
        decoded_masks[diagnostics["undetermined_indices"]].astype(np.int64)
        if diagnostics["undetermined_indices"] else np.array([], dtype=np.int64),
        dropped_masks.astype(np.int64),
    ])
    diagnostics["event_indicator_matrix"] = B

    decorated_dem = decorate_dem_with_logical_flags(dem, flags)
    return decorated_dem, flags, residual_fraction, diagnostics

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
from .utils import pack_detector_samples

try:
    import pymatching as _pymatching
except ImportError:  # pragma: no cover - exercised only without pymatching
    _pymatching = None

try:
    from tesseract_decoder import tesseract as _tesseract
except ImportError:  # pragma: no cover - exercised only without tesseract
    _tesseract = None


def _mask_array(masks):
    """
    Integer bitmasks as a numpy array: int64 when every mask fits, object
    dtype otherwise (masks of DEMs with more than 63 detectors overflow
    int64, which previously raised on large devices).
    """
    masks = [int(m) for m in masks]
    if masks and max(masks) > np.iinfo(np.int64).max:
        return np.array(masks, dtype=object)
    return np.array(masks, dtype=np.int64)


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
    masks = _mask_array([m for m, _ in items])
    if int(masks[-1]) >= (1 << num_detectors):
        raise ValueError("DEM events touch detectors beyond num_detectors.")
    probs = np.array([p for _, p in items], dtype=float)
    H = np.zeros((num_detectors, len(items)), dtype=np.uint8)
    for j, mask in enumerate(masks):
        mm = int(mask)
        while mm:
            low = mm & -mm
            H[low.bit_length() - 1, j] = 1
            mm ^= low
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


def _gf2_eliminate_packed(R_packed, y, n_cols):
    """
    Row-reduce a bit-packed augmented system [A | y] over GF(2) in place.

    Same pivot selection and row operations as `_gf2_eliminate`, but rows are
    packed 64 bits per uint64 word, which makes the elimination practical for
    tens of thousands of shots by thousands of events.

    Parameters:
        R_packed: np.ndarray
            (m, ceil(n/64)) uint64 packed rows; MODIFIED IN PLACE.
        y: np.ndarray
            m-vector of binary values; MODIFIED IN PLACE.
        n_cols: int
            Number of (unpacked) columns.

    Returns:
        pivot_cols: list[int]
            Columns containing pivots; row i has its pivot in pivot_cols[i].
    """
    m = R_packed.shape[0]
    pivot_cols = []
    r = 0
    for c in range(n_cols):
        if r >= m:
            break
        w, b = divmod(c, 64)
        bit = np.uint64(1) << np.uint64(b)
        below = (R_packed[r:, w] & bit) != 0
        nonzero = np.nonzero(below)[0]
        if len(nonzero) == 0:
            continue
        pr = r + int(nonzero[0])
        if pr != r:
            R_packed[[r, pr]] = R_packed[[pr, r]]
            y[[r, pr]] = y[[pr, r]]
        elim = (R_packed[:, w] & bit) != 0
        elim[r] = False
        if elim.any():
            R_packed[elim] ^= R_packed[r]
            y[elim] ^= y[r]
        pivot_cols.append(c)
        r += 1
    return pivot_cols


def _packed_column_bits(R_packed, col, num_rows):
    """Extract (unpacked) column `col` of the first `num_rows` packed rows."""
    w, b = divmod(col, 64)
    return ((R_packed[:num_rows, w] >> np.uint64(b)) & np.uint64(1)).astype(np.uint8)


def _packed_residual_fraction(B_packed, Y, x_packed):
    """Fraction of rows where parity(row & x) != Y, on packed data."""
    if B_packed.shape[0] == 0:
        return 0.0
    parity = (np.bitwise_count(B_packed & x_packed).sum(axis=1,
                                                        dtype=np.int64)
              & 1).astype(np.uint8)
    return float(np.mean(parity ^ Y))


def _bitflip_refine(B_csc, Y, x, col_totals, max_flips=None):
    """
    Greedy single-bit descent on the number of violated rows of Y = B x.

    Flipping flag j toggles the consistency of every row that contains event
    j, so the change in violated-row count is (satisfied_j - violated_j).
    Repeatedly flip the flag with the largest strictly positive improvement
    until none remains. The true flag vector is a strong local minimum in
    the low-logical-error regime (most rows containing an event agree with
    its correct flag), which is what makes this effective where plain
    elimination cannot avoid pivoting on corrupted rows.

    Parameters:
        B_csc: scipy.sparse.csc_matrix
            (n_shots x n_events) indicator matrix.
        Y: np.ndarray
            n_shots binary outcomes.
        x: np.ndarray
            Starting flag vector; MODIFIED IN PLACE.
        col_totals: np.ndarray
            Number of rows containing each event (column sums of B).
        max_flips: int, optional
            Safety cap (default 10 * n_events).

    Returns:
        flips: int
            Number of flips applied.
    """
    n_shots, n_events = B_csc.shape
    if n_shots == 0 or n_events == 0:
        return 0
    if max_flips is None:
        max_flips = 10 * n_events
    viol = ((B_csc @ x.astype(np.int64)) % 2).astype(np.uint8) ^ Y
    flips = 0
    while flips < max_flips:
        violated_per_event = B_csc.T @ viol.astype(np.int64)
        improvement = 2 * violated_per_event - col_totals
        j = int(np.argmax(improvement))
        if improvement[j] <= 0:
            break
        x[j] ^= 1
        rows = B_csc.indices[B_csc.indptr[j]:B_csc.indptr[j + 1]]
        viol[rows] ^= 1
        flips += 1
    return flips


def solve_gf2_robust(
    B,
    Y,
    residual_threshold=1e-3,
    max_ransac_iterations=200,
    seed=None,
    row_order=None,
    refine=True,
):
    """
    Solve Y = B L over GF(2), robustly against a fraction of bad rows.

    First computes a candidate solution by GF(2) Gaussian elimination
    (bit-packed, so tens of thousands of shots by thousands of events are
    practical) and checks the Hamming-weight residual of B L xor Y over all
    rows. When the residual exceeds `residual_threshold`, two escalations
    engage:

      * a greedy **bit-flip refinement** (`refine=True`, the default): flip
        the single flag that most reduces the number of violated rows, until
        no flip helps. In the regime where a non-negligible fraction of rows
        is corrupted (e.g. decoder logical errors at the percent level or
        above), every Gaussian elimination pivots on some corrupted rows and
        RANSAC re-orderings cannot avoid them either; the bit-flip descent
        repairs the resulting flag errors because each event's flag is
        vouched for by the majority of the (many) rows containing it.
      * a **RANSAC-style loop** over random row orderings (equivalent to
        solving from random independent row subsets), keeping the solution
        minimizing the residual; each candidate is also refined when
        `refine=True`.

    Parameters:
        B: array-like
            (n_shots x n_events) binary indicator matrix.
        Y: array-like
            n_shots binary vector of observed logical outcomes.
        residual_threshold: float
            Fraction of inconsistent rows considered acceptable: below it the
            solve is `converged` and no RANSAC re-solves are attempted. Set
            it to the decoder logical-error rate you expect from the data.
        max_ransac_iterations: int
            Maximum number of RANSAC re-solves.
        seed: int, optional
            Seed for the RANSAC row shuffles.
        row_order: str, optional
            None (default): eliminate rows in the order given. "confidence":
            eliminate rows in order of increasing row weight (shots whose
            correction involves fewer events are less likely to be decoder
            logical errors, so pivots land on more trustworthy rows).
        refine: bool
            Enable the bit-flip refinement stage (default True). Refinement
            only ever lowers the residual; it is skipped when the initial
            solve is already consistent.

    Returns:
        L: np.ndarray
            n_events binary flag vector (free variables set to 0; see
            diagnostics['undetermined_indices']).
        residual_fraction: float
            Fraction of rows with (B L) mod 2 != Y at the returned solution.
        diagnostics: dict
            Keys: 'rank', 'n_events', 'n_shots', 'undetermined_indices',
            'null_space', 'initial_residual_fraction', 'ransac_iterations',
            'method', 'converged', 'refine_flips', 'row_order'.
    """
    B = np.asarray(B, dtype=np.uint8) % 2
    if B.ndim != 2:
        raise ValueError("B must be a 2D array of shape (n_shots, n_events).")
    Y = np.asarray(Y, dtype=np.uint8).ravel() % 2
    n_shots, n_events = B.shape
    if len(Y) != n_shots:
        raise ValueError("Y must have one entry per row of B.")
    if row_order not in (None, "given", "confidence"):
        raise ValueError("row_order must be None, 'given' or 'confidence'.")

    B_packed = pack_detector_samples(B) if n_events else \
        np.zeros((n_shots, 1), dtype=np.uint64)

    if row_order == "confidence" and n_shots:
        order = np.argsort(B.sum(axis=1), kind="stable")
    else:
        order = np.arange(n_shots)

    R_packed = B_packed[order].copy()
    z = Y[order].copy()
    pivot_cols = _gf2_eliminate_packed(R_packed, z, n_events)
    rank = len(pivot_cols)

    # Null space and free variables of the reduced system.
    undetermined = sorted(set(range(n_events)) - set(pivot_cols))
    null_space = np.zeros((len(undetermined), n_events), dtype=np.uint8)
    for k, f in enumerate(undetermined):
        null_space[k, f] = 1
        null_space[k, pivot_cols] = _packed_column_bits(R_packed, f, rank)

    best = np.zeros(n_events, dtype=np.uint8)
    best[pivot_cols] = z[:rank]
    best_packed = pack_detector_samples(best[None, :])[0] if n_events else \
        np.zeros(1, dtype=np.uint64)
    best_residual = _packed_residual_fraction(B_packed, Y, best_packed)
    initial_residual = best_residual
    method = "gaussian_elimination"

    B_csc = None
    col_totals = None
    refine_flips = 0

    def _refine_inplace(x):
        nonlocal B_csc, col_totals
        if B_csc is None:
            B_csc = scipy.sparse.csc_matrix(B)
            col_totals = np.asarray(B_csc.sum(axis=0)).ravel().astype(np.int64)
        return _bitflip_refine(B_csc, Y, x, col_totals)

    if refine and best_residual > 0:
        refine_flips += _refine_inplace(best)
        best_packed = pack_detector_samples(best[None, :])[0]
        best_residual = _packed_residual_fraction(B_packed, Y, best_packed)

    ransac_iterations = 0
    if (best_residual > residual_threshold and n_shots > rank
            and max_ransac_iterations > 0):
        rng = np.random.default_rng(seed)
        method = "ransac"
        for _ in range(max_ransac_iterations):
            ransac_iterations += 1
            perm = rng.permutation(n_shots)
            Rp = B_packed[perm].copy()
            zp = Y[perm].copy()
            pivots_p = _gf2_eliminate_packed(Rp, zp, n_events)
            candidate = np.zeros(n_events, dtype=np.uint8)
            candidate[pivots_p] = zp[:len(pivots_p)]
            if refine:
                refine_flips += _refine_inplace(candidate)
            cand_packed = pack_detector_samples(candidate[None, :])[0]
            residual = _packed_residual_fraction(B_packed, Y, cand_packed)
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
        "refine_flips": refine_flips,
        "row_order": row_order or "given",
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


def _dem_for_decoding(masks, probs, num_detectors, min_probability=1e-12):
    """
    Build a canonical stim DEM (events in sorted-bitmask order, probabilities
    clipped away from 0 and 0.5) padded to span `num_detectors` detectors.
    """
    p = np.clip(probs, min_probability, 0.5 - min_probability)
    lines = []
    for mask, pi in zip(masks, p):
        targets = " ".join(f"D{d}" for d in range(num_detectors) if (int(mask) >> d) & 1)
        lines.append(f"error({pi}) {targets}")
    # Declare the highest detector so dem.num_detectors matches the data even
    # when the learned events do not touch it.
    lines.append(f"detector D{num_detectors - 1}")
    return dem_from_str("\n".join(lines) + "\n")


def _error_detectors(error):
    """Best-effort extraction of the detector indices of a tesseract error."""
    symptom = getattr(error, "symptom", None)
    if symptom is not None:
        detectors = getattr(symptom, "detectors", None)
        if detectors is not None:
            return [int(d) for d in detectors]
    import re
    found = re.findall(r"D(\d+)", str(error))
    if found:
        return [int(d) for d in found]
    return None


class _TesseractEventDecoder:
    """
    Wraps a compiled tesseract decoder so that decoding a shot yields the
    per-event indicator row over the DEM events (sorted-bitmask order).
    """

    def __init__(self, decoder, column_map, n_events):
        self._decoder = decoder
        self._column_map = column_map
        self.n_events = n_events

    def decode_event_indicators(self, detector_samples):
        shots = np.asarray(detector_samples, dtype=bool)
        B = np.zeros((shots.shape[0], self.n_events), dtype=np.uint8)
        for i in range(shots.shape[0]):
            self._decoder.decode_to_errors(shots[i])
            for error_index in self._decoder.predicted_errors_buffer:
                col = self._column_map[error_index]
                if col >= 0:
                    B[i, col] ^= 1
        return B


def build_tesseract_decoder(dem, num_detectors=None, min_probability=1e-12):
    """
    Build a tesseract (most-likely-error) decoder from a DEM.

    Unlike the pymatching path, the DEM does not need to be graph-like:
    tesseract natively decodes hyperedge events (any number of detectors).

    Parameters:
        dem: stim.DetectorErrorModel
        num_detectors: int, optional
            Number of detectors in the syndrome data (default dem.num_detectors).
        min_probability: float
            Probabilities are clipped to [min_probability, 0.5 - min_probability].

    Returns:
        decoder: _TesseractEventDecoder
            Object whose `decode_event_indicators(detector_samples)` returns
            the (n_shots x n_events) indicator matrix.
        masks: np.ndarray
            Integer event bitmasks aligned with the indicator columns.
    """
    if _tesseract is None:
        raise ImportError(
            "tesseract-decoder is required for decoder='tesseract'; "
            "pip install tesseract-decoder (no aarch64-linux wheels exist on "
            "PyPI as of 2026-08; a source build via bazel may be required)."
        )
    H, probs, masks = dem_to_check_matrix(dem, num_detectors=num_detectors)
    n_det = H.shape[0]
    decode_dem = _dem_for_decoding(masks, probs, n_det, min_probability)
    config = _tesseract.TesseractConfig(dem=decode_dem)
    decoder = config.compile_decoder()

    # Map tesseract's internal error indices to our event columns. The DEM is
    # built in sorted-bitmask order, so positional alignment is the fallback;
    # when the decoder exposes each error's detectors we verify explicitly.
    mask_to_col = {int(m): j for j, m in enumerate(masks)}
    errors = getattr(decoder, "errors", None)
    if errors is not None:
        column_map = []
        for k, error in enumerate(errors):
            detectors = _error_detectors(error)
            if detectors is None:
                column_map = None
                break
            error_mask = sum(1 << d for d in detectors)
            column_map.append(mask_to_col.get(error_mask, -1))
        if column_map is not None and len(column_map) != len(errors):
            column_map = None
    else:
        column_map = None
    if column_map is None:
        column_map = list(range(len(masks)))

    return _TesseractEventDecoder(decoder, column_map, len(masks)), masks


def build_decoder(dem, decoder="pymatching", num_detectors=None, min_probability=1e-12):
    """
    Build a decoder backend from a DEM.

    Parameters:
        dem: stim.DetectorErrorModel
        decoder: str
            'pymatching' (minimum-weight perfect matching; DEM must be
            graph-like) or 'tesseract' (most-likely-error; supports
            hyperedge events).
        num_detectors: int, optional
            Number of detectors in the syndrome data (default dem.num_detectors).
        min_probability: float
            Probability clipping used when converting to decoder weights.

    Returns:
        decoder_object: pymatching.Matching or _TesseractEventDecoder
        masks: np.ndarray
            Integer event bitmasks aligned with decode output columns.
    """
    if decoder == "pymatching":
        return build_matcher(dem, num_detectors=num_detectors,
                             min_probability=min_probability)
    if decoder == "tesseract":
        return build_tesseract_decoder(dem, num_detectors=num_detectors,
                                       min_probability=min_probability)
    raise ValueError("decoder must be 'pymatching' or 'tesseract'.")


def decode_event_indicators(decoder_object, detector_samples):
    """
    Decode a batch of shots into per-event indicator rows.

    Parameters:
        decoder_object: pymatching.Matching or _TesseractEventDecoder
            A pymatching matcher built with an identity faults matrix (see
            `build_matcher`) or a tesseract decoder (see
            `build_tesseract_decoder` / `build_decoder`).
        detector_samples: np.ndarray
            (n_shots x n_detectors) binary array; column d is detector d
            (stim sampling convention).

    Returns:
        B: np.ndarray
            (n_shots x n_events) binary indicator matrix.
    """
    if isinstance(decoder_object, _TesseractEventDecoder):
        return decoder_object.decode_event_indicators(detector_samples)
    shots = np.asarray(detector_samples, dtype=np.uint8) % 2
    return np.asarray(decoder_object.decode_batch(shots), dtype=np.uint8) % 2


def assign_logical_flags(
    dem,
    detector_samples,
    logical_outcomes,
    num_detectors=None,
    decoder="pymatching",
    on_nongraphlike="drop",
    residual_threshold=1e-3,
    max_ransac_iterations=200,
    seed=None,
    row_order=None,
    refine=True,
):
    """
    Decorate a learned DEM with logical-flip flags inferred from shot data.

    Builds a decoder from the learned DEM, decodes every shot to obtain the
    event indicator matrix B, and solves Y = B L over GF(2) (robustly, via
    `solve_gf2_robust`) for the per-event logical flags L.

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
        decoder: str
            'pymatching' (default; minimum-weight perfect matching, requires
            graph-like events) or 'tesseract' (most-likely-error decoder;
            handles hyperedge events, so nothing is dropped).
        on_nongraphlike: str
            pymatching only: what to do with events flipping more than two
            detectors: 'drop' (default; they are excluded from matching and
            left undecorated) or 'raise'.
        residual_threshold: float
            Acceptable fraction of inconsistent rows (see `solve_gf2_robust`).
            Set this to roughly the decoder logical-error rate you expect on
            this data; rows are inconsistent exactly when the decoder makes a
            logical error.
        max_ransac_iterations: int
            Maximum RANSAC re-solves in the GF(2) solver.
        seed: int, optional
            Seed for the RANSAC row shuffles.
        row_order: str, optional
            Row ordering for the GF(2) elimination; see `solve_gf2_robust`.
            "confidence" is recommended when the logical error rate is high.
        refine: bool
            Enable bit-flip refinement in the GF(2) solver (default True).

    Returns:
        decorated_dem: stim.DetectorErrorModel
            The input DEM with `L0` appended to flagged events.
        flags: np.ndarray
            Binary flag per event, aligned with diagnostics['masks'].
        residual_fraction: float
            Fraction of shots inconsistent with the returned flags.
        diagnostics: dict
            GF(2) solver diagnostics plus 'decoder', 'masks' (all event
            bitmasks, sorted), 'decoded_masks' (events used in decoding),
            'dropped_masks' (non-graph-like events excluded by the pymatching
            path; always empty for tesseract), and 'undetermined_masks'
            (events whose flag the data does not determine).
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

    if decoder == "pymatching":
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
                "pymatching is required for decoder='pymatching'; "
                "pip install pymatching."
            )
        kept = np.nonzero(graphlike)[0]
        p = np.clip(probs[kept], 1e-12, 0.5 - 1e-12)
        matcher = _pymatching.Matching(
            scipy.sparse.csc_matrix(H[:, kept]),
            weights=np.log((1.0 - p) / p),
            faults_matrix=scipy.sparse.eye(len(kept), dtype=np.uint8, format="csc"),
        )
        B = np.asarray(matcher.decode_batch(detector_samples), dtype=np.uint8) % 2
    elif decoder == "tesseract":
        # Tesseract decodes hyperedges natively: keep every event.
        kept = np.arange(len(masks))
        dropped_masks = masks[:0]
        tesseract_decoder, _ = build_tesseract_decoder(
            dem, num_detectors=num_detectors
        )
        B = tesseract_decoder.decode_event_indicators(detector_samples)
    else:
        raise ValueError("decoder must be 'pymatching' or 'tesseract'.")

    L_kept, residual_fraction, diagnostics = solve_gf2_robust(
        B,
        Y,
        residual_threshold=residual_threshold,
        max_ransac_iterations=max_ransac_iterations,
        seed=seed,
        row_order=row_order,
        refine=refine,
    )

    flags = np.zeros(len(masks), dtype=np.uint8)
    flags[kept] = L_kept
    decoded_masks = masks[kept]
    diagnostics["decoder"] = decoder
    diagnostics["masks"] = masks
    diagnostics["decoded_masks"] = decoded_masks
    diagnostics["dropped_masks"] = dropped_masks
    diagnostics["undetermined_masks"] = _mask_array(
        ([int(m) for m in decoded_masks[diagnostics["undetermined_indices"]]]
         if diagnostics["undetermined_indices"] else [])
        + [int(m) for m in dropped_masks]
    )
    diagnostics["event_indicator_matrix"] = B

    decorated_dem = decorate_dem_with_logical_flags(dem, flags)
    return decorated_dem, flags, residual_fraction, diagnostics

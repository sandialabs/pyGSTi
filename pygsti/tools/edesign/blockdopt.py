"""
Greedy block D-optimal selection of experiment-design candidates
"""
#***************************************************************************************************
# Copyright 2015, 2019, 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

import numpy as _np
import scipy.linalg as _spl

__all__ = [
    'block_linear_dopt',
    'greedy_candidate_scores',
    'greedy_path_log_volumes',
]


def _validate_inputs(A, block_size, max_blocks):
    """Check the arguments of :func:`block_linear_dopt` and return `A` as an ndarray."""
    A = _np.asarray(A)
    if A.ndim != 2:
        raise ValueError("A must be a 2-D array.")
    if A.dtype not in (_np.float32, _np.float64):
        raise TypeError("A must have dtype float32 or float64.")
    m, n = A.shape
    if m <= 0 or n <= 0:
        raise ValueError("block_linear_dopt: m and n must be positive.")
    if block_size <= 0:
        raise ValueError("block_linear_dopt: block_size must be positive.")
    if n % block_size != 0:
        raise ValueError("block_linear_dopt: require n divisible by block_size.")
    if max_blocks < 0:
        raise ValueError("block_linear_dopt: max_blocks must be nonnegative.")
    return A


def _candidate_blocks(A, block_size):
    """`(m, n_candidates * b)` -> `(n_candidates, m, b)` stack with `out[i] = A_i`.

    A view when `A` is C-contiguous; otherwise `reshape` copies.  `A` is never
    written to either way.
    """
    m, n = A.shape
    return A.reshape(m, n // block_size, block_size).transpose(1, 0, 2)


def _scipy_qr_is_batched():
    """Whether this SciPy's `linalg.qr` accepts a stack of matrices.

    Batched (gufunc-style) input landed in SciPy 1.18; pyGSTi does not pin a
    SciPy version, so :func:`_batched_qr_r` falls back to a Python loop when
    this is False.  Probed by trying it rather than by comparing version
    strings, so a backport or a vendored build is picked up too.
    """
    probe = _np.zeros((2, 3, 2))
    probe[:, 0, 0] = 1.0
    probe[:, 1, 1] = 1.0
    try:
        R, = _spl.qr(probe, mode='r', check_finite=False)
    except Exception:
        return False
    return getattr(R, 'shape', None) == (2, 3, 2)


_SCIPY_QR_IS_BATCHED = _scipy_qr_is_batched()


def _batched_qr_r(W):
    """R factors of a `(k, r, c)` stack of matrices, as a `(k, r, c)` stack.

    One `scipy.linalg.qr` call over the whole stack where SciPy supports it,
    otherwise a loop over the stack.  Both paths call LAPACK `geqrf` once per
    matrix and both leave the strict lower triangle filled with the Householder
    reflectors, so the two agree entry for entry; only the Python-level looping
    differs.  `check_finite` is off in both, so NaN/Inf flows into `R` to be
    caught by the caller's singularity test.
    """
    if _SCIPY_QR_IS_BATCHED:
        R, = _spl.qr(W, mode='r', check_finite=False)
        return R
    out = _np.empty_like(W)
    for i in range(W.shape[0]):
        out[i], = _spl.qr(W[i], mode='r', check_finite=False)
    return out


def block_linear_dopt(A, block_size, max_blocks, *, return_scores=False):
    """Deterministic greedy block D-optimal selection.

    Selects column blocks of `A` one at a time, each time taking the block that
    most increases `0.5 * logdet(I + sum_{i in S} A_i A_i^T)`.

    Parameters
    ----------
    A : numpy.ndarray
        The **non-augmented** `J^T` matrix, shape `(m, n)` with `m` = number of
        model parameters and `n = n_candidates * block_size`.  The `i`-th
        candidate is the `m x block_size` column block
        `A_i = A[:, i*block_size:(i+1)*block_size]`.  dtype must be float32 or
        float64; all arithmetic is done in that dtype.  `A` is not modified.

    block_size : int
        Number of columns per candidate block.  For a Jacobian this is the
        number of outcomes per circuit.

    max_blocks : int
        Maximum number of blocks to select; clamped to `n_candidates`.

    return_scores : bool, optional (default False)
        If true, also return the winning score of each greedy step.

    Returns
    -------
    block_pivots : numpy.ndarray
        int64, length `min(n_candidates, max_blocks)`.  0-based candidate
        indices in selection order.

    scores : numpy.ndarray
        Only returned if `return_scores` is true.  float64, same length.
        `scores[k]` is the objective value of the block chosen at step `k`,
        i.e. `sum_j log|R_jj|` of the workspace of that block, which equals
        `0.5 * logdet(I_m + sum_{i in S_k} A_i A_i^T)` with `S_k` the first
        `k+1` selected blocks.  The curve is nondecreasing.

    Notes
    -----
    Every candidate `i` owns an `(m + block_size) x m` workspace

        W_i = [ A_i^T ]
              [  I_m  ].

    All workspaces of the not-yet-selected candidates are kept stacked in one
    `(n_remaining, m + block_size, m)` array `W`, in ascending candidate order.
    Each iteration:

    1. Householder-QR (`geqrf`) every workspace in the stack.
    2. Score each by `sum_j log|R_jj|`; a non-positive (or NaN) diagonal entry
       marks the block singular and it is skipped.  Pick the first block
       attaining the maximum score (ties resolve to the lowest index).  If no
       non-singular block remains, raise `RuntimeError`.
    3. Drop the winner from the stack and, for every *remaining* candidate,
       zero the strict lower triangle of the leading `m x m` part (leaving
       `R_i`) and overwrite the bottom `block_size` rows with `A_{i*}^T` of the
       newly selected block, so that

            W_i <- [ R_i      ]
                   [ A_{i*}^T ].

    Because orthogonal transformations preserve the Gram matrix, after step `k`
    each `W_i` is (up to signs) the R factor of
    `[A_i^T ; I_m ; A_{s_1}^T ; ... ; A_{s_k}^T]`, so its score is exactly the
    log-volume of the design `S_k U {i}`.  That is what makes this greedy
    D-optimal selection: one QR per remaining candidate per step scores every
    candidate exactly, with no rank-one update bookkeeping.

    The identity block in `W_i` is a fixed unit ridge on the information
    matrix.  Callers who want a different ridge `lambda` should scale `A` by
    `lambda**-0.5` before calling, which turns the objective into
    `0.5 * logdet(lambda * I + J_S^T J_S)` up to an additive constant.  The
    ridge also means a rank-deficient -- even all-zero -- candidate block is
    still a non-singular workspace, scoring whatever the current design already
    scores.  So on finite input the `RuntimeError` below cannot fire; it is
    reachable only when NaN or Inf reaches every remaining candidate.

    This is a port of the pure numpy/scipy reference implementation in the
    `bled` package (`bled.reference_impls.block_linear_dopt`), which is itself
    a transliteration of that package's C++ kernel.
    """
    A = _validate_inputs(A, block_size, max_blocks)
    T = A.dtype
    m, n = A.shape
    b_sz = int(block_size)
    n_candidates = n // b_sz
    num_blocks = min(n_candidates, int(max_blocks))

    block_pivs = _np.zeros(num_blocks, dtype=_np.int64)
    best_scores = _np.zeros(num_blocks, dtype=_np.float64)
    if num_blocks == 0:
        return (block_pivs, best_scores) if return_scores else block_pivs

    # blocks_T[i] = A_i^T, shape (n_candidates, b_sz, m).
    blocks_T = _candidate_blocks(A, b_sz).transpose(0, 2, 1)

    # Initialise all W_i = [ A_i^T ; I_m ] in one stacked array. Row r of the
    # stack belongs to candidate cand[r]; cand stays ascending throughout.
    W = _np.empty((n_candidates, m + b_sz, m), dtype=T)
    W[:, :b_sz, :] = blocks_T
    W[:, b_sz:, :] = _np.eye(m, dtype=T)
    cand = _np.arange(n_candidates)

    for it in range(num_blocks):
        n_rem = W.shape[0]

        # ---- Step 1: QR of every remaining workspace. ----------------------
        # mode='r' wraps LAPACK geqrf. It returns the full (m + b_sz) x m
        # factor; the leading m x m upper triangle is R_i.
        R = _batched_qr_r(W)[:, :m, :]                              # (n_rem, m, m)

        # ---- Step 2: score = sum_j log|R_jj|; pick the first maximiser. ----
        d = _np.abs(_np.diagonal(R, axis1=-2, axis2=-1))            # (n_rem, m)
        valid = _np.all(d > 0, axis=1)                              # False for NaN too
        scores = _np.full(n_rem, -_np.inf, dtype=T)
        scores[valid] = _np.log(d[valid]).sum(axis=1)               # working precision
        j = int(_np.argmax(scores))                                 # first maximum
        if not valid[j]:
            raise RuntimeError(
                "block_linear_dopt: no valid (non-singular) candidate block "
                "remains (input may be rank-deficient or contain NaN/Inf)."
            )
        i_star = int(cand[j])
        block_pivs[it] = i_star
        best_scores[it] = scores[j]

        if it == num_blocks - 1:
            break

        # ---- Step 3: W_i <- [ triu(R_i) ; A_{i*}^T ] for the remaining i. ---
        keep = _np.ones(n_rem, dtype=bool)
        keep[j] = False
        W = _np.empty((n_rem - 1, m + b_sz, m), dtype=T)
        W[:, :m, :] = _np.triu(R[keep])                             # stacked triu
        W[:, m:, :] = blocks_T[i_star]
        cand = cand[keep]

    return (block_pivs, best_scores) if return_scores else block_pivs


# --------------------------------------------------------------------------- #
#  Independent (QR-free) scorers
# --------------------------------------------------------------------------- #

def _gram_blocks(A, block_size):
    """float64 `(n_candidates, m, m)` stack of `A_i A_i^T` (one stacked matmul)."""
    A = _np.asarray(A, dtype=_np.float64)
    blocks = _candidate_blocks(A, block_size)                       # (n_cand, m, b)
    return blocks @ blocks.transpose(0, 2, 1)


def _half_logdet(M):
    """Batched `0.5 * logdet(M)` over a stack; `-inf` where `det(M) <= 0`."""
    sign, logabsdet = _np.linalg.slogdet(M)
    return _np.where(sign > 0, 0.5 * logabsdet, -_np.inf)


def greedy_candidate_scores(A, block_size, selected=()):
    """Score of every candidate block given an already-selected prefix.

    This is the objective :func:`block_linear_dopt` maximises at each step,
    evaluated in float64 via `slogdet` rather than via QR.  It shares no
    arithmetic with the kernel, so it is an independent yardstick for judging
    the kernel's choices, and it is the natural way to ask "what would this
    circuit add?" without re-running a selection.

    Parameters
    ----------
    A : numpy.ndarray
        The non-augmented `J^T`, as for :func:`block_linear_dopt`.

    block_size : int
        Number of columns per candidate block.

    selected : sequence of int, optional
        Candidate indices already in the design.

    Returns
    -------
    numpy.ndarray
        float64, length `n_candidates`, with

            `s[i] = 0.5 * logdet(I_m + sum_{j in selected U {i}} A_j A_j^T)`

        for unselected `i`, and `-inf` for `i` already in `selected`.
    """
    grams = _gram_blocks(A, block_size)                             # (n_cand, m, m)
    m = grams.shape[-1]
    selected = _np.asarray(list(selected), dtype=_np.int64)
    base = _np.eye(m) + grams[selected].sum(axis=0)
    scores = _half_logdet(base[None] + grams)                       # (n_cand,)
    scores[selected] = -_np.inf
    return scores


def greedy_path_log_volumes(A, block_size, block_pivots):
    """Cumulative log-volume curve of a selection order, evaluated independently.

    Parameters
    ----------
    A : numpy.ndarray
        The non-augmented `J^T`, as for :func:`block_linear_dopt`.

    block_size : int
        Number of columns per candidate block.

    block_pivots : sequence of int
        Candidate indices in selection order, e.g. the output of
        :func:`block_linear_dopt`.

    Returns
    -------
    numpy.ndarray
        float64, length `len(block_pivots) + 1`, with

            `out[k] = 0.5 * logdet(I_m + sum_{i in block_pivots[:k]} A_i A_i^T)`

        so `out[0] == 0` and `out[1:]` is what `block_linear_dopt`'s
        `return_scores` reports for the same order.  Use it to see where a
        budget stops buying information, or to score a selection the kernel did
        not produce (a random subset, say) on the same footing.
    """
    grams = _gram_blocks(A, block_size)
    m = grams.shape[-1]
    piv = _np.asarray(list(block_pivots), dtype=_np.int64)
    out = _np.zeros(len(piv) + 1)
    if len(piv) == 0:
        return out
    M = _np.eye(m) + _np.cumsum(grams[piv], axis=0)                 # (k, m, m)
    out[1:] = _half_logdet(M)
    return out

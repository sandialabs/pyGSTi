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
    if not _np.isfinite(A).all():
        raise ValueError("block_linear_dopt: A must be finite; it contains NaN or Inf.")
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
    differs.  `check_finite` is off in both because
    :func:`block_linear_dopt` has already rejected non-finite input, which
    makes SciPy's own scan redundant.
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
        float64 and every entry must be finite; all arithmetic is done in that
        dtype.  `A` is not modified.

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
    Scoring goes through the matrix determinant lemma, so a candidate costs a QR
    with `block_size` columns rather than `num_params` columns.

    Write `M_S = I_m + sum_{i in S} A_i A_i^T` for the ridged information matrix
    of the design chosen so far, and `R_S` for its upper-triangular factor,
    `R_S^T R_S = M_S`.  Then

        0.5*logdet(M_{S+i}) = 0.5*logdet(M_S) + 0.5*logdet(I_b + G_i G_i^T),

    where `G_i = A_i^T R_S^-1` is candidate `i`'s row block expressed in the
    current factor's coordinates.  The second term is candidate `i`'s *gain*, and
    `sum_j log|rho_jj|` of the R factor of the `(m + b_sz) x b_sz` panel

        P_i = [ G_i^T ]
              [  I_b  ]

    is exactly that gain.  So one QR per remaining candidate per step still
    scores every candidate exactly, with no rank-one update bookkeeping -- but
    the panels have `b_sz` columns instead of `m`, and the running total of the
    gains is the same log-volume curve.

    `R_S` is never built by a Cholesky factorization, and in fact never built at
    all.  The algorithm carries `G` instead of `A` and updates it in place: when
    candidate `i*` is selected, `R_new = C R_S` with `C^T C = I_m + G_{i*}^T
    G_{i*}`, so `C` is the R factor of the `(b_sz + m) x m` matrix
    `[G_{i*} ; I_m]` -- one QR per *step*, not per candidate -- and every
    surviving `G_i` becomes `G_i C^-1`.  Since `C^T C >= I_m`, `||C^-1|| <= 1`
    and the updates are contractions, so the in-place arithmetic does not
    amplify earlier rounding.

    Storage is a single row-major buffer holding the augmented candidate matrix:
    row block `i` is `[ G_i | I_b ]`, of shape `b_sz x (m + b_sz)`, so
    transposing it yields the panels `P_i` in the column-major order LAPACK's
    `geqrf` wants, with no copy.  The trailing identity columns are constant,
    which makes the update `G <- G C^-1` one in-place triangular solve of the
    survivors' rows against `diag(C, I_b)`; a selection retires the winner by
    swapping its row block past the active boundary.  Nothing is reallocated
    inside the loop.

    Those swaps permute the survivors, so ties are resolved by the lowest
    *original* candidate index explicitly rather than by argmax position.

    The identity block in `P_i` is a fixed unit ridge on the information matrix.
    Callers who want a different ridge `lambda` should scale `A` by
    `lambda**-0.5` before calling, which turns the objective into
    `0.5 * logdet(lambda * I + J_S^T J_S)` up to an additive constant.  Because
    `P_i^T P_i = I_b + G_i G_i^T` is at least `I_b`, every candidate has a finite
    nonnegative gain: a rank-deficient block gains less than a full-rank one and
    an all-zero block gains exactly zero, so blocks that carry no information are
    simply ranked after every block that carries some.  There is no singular
    case to handle, and non-finite input is rejected up front.

    The `bled` reference implementation
    (`bled.reference_impls.block_linear_dopt`) instead QRs an `(m + b_sz) x m`
    workspace per candidate.  The two are algebraically the same greedy
    selection, and agree on selections and log-volumes to rounding; this
    formulation is about 5x faster and uses a third of the memory at 1926
    candidates with 42 parameters and 8 outcomes.
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

    # The one persistent workspace: the augmented candidate matrix, row-major.
    # Row block i is [ G_i | I_b ], so buf[:k].transpose(0, 2, 1) is a stack of
    # column-major panels P_i without copying.  G starts at A^T because the
    # empty design has R_S = I_m.  A itself is never written to.
    buf = _np.empty((n_candidates, b_sz, m + b_sz), dtype=T)
    buf[:, :, :m] = _candidate_blocks(A, b_sz).transpose(0, 2, 1)
    buf[:, :, m:] = _np.eye(b_sz, dtype=T)
    rows = buf.reshape(n_candidates * b_sz, m + b_sz)               # a view

    # Scratch for the two per-step factorizations, also allocated once.
    # cwork.T is the column-major [ G_{i*} ; I_m ] handed to the QR; dwork is
    # diag(C, I_b), the triangular factor the survivors are solved against.
    cwork = _np.empty((m, m + b_sz), dtype=T)
    cwork[:, b_sz:] = _np.eye(m, dtype=T)
    dwork = _np.zeros((m + b_sz, m + b_sz), dtype=T)
    dwork[m:, m:] = _np.eye(b_sz, dtype=T)

    # cand[r] is the original index of the candidate now in row block r. The
    # swaps below permute it, so it is not ascending after the first step.
    cand = _np.arange(n_candidates)
    n_rem = n_candidates
    log_volume = 0.0

    for it in range(num_blocks):
        # ---- Step 1: QR of every remaining panel, b_sz columns each. -------
        # mode='r' wraps LAPACK geqrf and returns the full (m + b_sz) x b_sz
        # factor; the leading b_sz x b_sz upper triangle is rho_i.
        R = _batched_qr_r(buf[:n_rem].transpose(0, 2, 1))[:, :b_sz, :]

        # ---- Step 2: gain = sum_j log|rho_jj|; lowest-index maximiser. -----
        # Every panel has full column rank because of its identity block, so
        # every |rho_jj| is at least 1 and every gain is finite and >= 0.
        d = _np.abs(_np.diagonal(R, axis1=-2, axis2=-1))            # (n_rem, b_sz)
        gains = _np.log(d).sum(axis=1)                              # working precision
        best = gains.max()
        tied = _np.flatnonzero(gains == best)
        j = int(tied[_np.argmin(cand[tied])])
        block_pivs[it] = int(cand[j])
        log_volume += float(best)
        best_scores[it] = log_volume

        if it == num_blocks - 1:
            break

        # ---- Step 3: C, with C^T C = I_m + G_{i*}^T G_{i*}. ----------------
        cwork[:, :b_sz] = buf[j, :, :m].T
        C, = _spl.qr(cwork.T, mode='r', check_finite=False)

        # ---- Step 4: retire the winner past the active boundary. -----------
        last = n_rem - 1
        if j != last:
            winner = buf[j].copy()
            buf[j] = buf[last]
            buf[last] = winner
            cand[j], cand[last] = cand[last], cand[j]
        n_rem = last

        # ---- Step 5: G <- G C^-1 for the survivors, in place. --------------
        # rows[:k] is a contiguous prefix, so its transpose is genuinely
        # column-major and `overwrite_b` writes straight back into buf. The
        # trailing identity columns are solved against I_b, i.e. left alone.
        dwork[:m, :m] = C[:m, :]
        _spl.solve_triangular(dwork, rows[:n_rem * b_sz].T, trans='T', lower=False,
                              overwrite_b=True, check_finite=False)

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
    """Batched `0.5 * logdet(M)` over a stack of ridged information matrices.

    Every `M` handed here is `I_m` plus a sum of Gram matrices, so its
    eigenvalues are at least 1 and `logdet` is nonnegative; there is no
    sign to test.
    """
    return 0.5 * _np.linalg.slogdet(M)[1]


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

        for unselected `i`.  Entries of `selected` are marked `-inf`, as a
        sentinel meaning "already taken" rather than a statement about the
        block: every finite block scores at least as much as `selected` does
        on its own.
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

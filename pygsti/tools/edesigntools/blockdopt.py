"""
Greedy block D-optimal selection of experiment-design candidates

The first half of this module is the selection kernel, `block_linear_dopt`, with
two independent scorers for its output; it is plain numpy and scipy and can be
read without any experiment-design context.  The second half applies it to a
pyGSTi model and design: `rank_circuits_by_dopt` and `BlockDoptReducer`, the
`DesignReducer` that `design.reduce_with` accepts.
"""
#***************************************************************************************************
# Copyright 2015, 2019, 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

from __future__ import annotations

import warnings as _warnings
from typing import TYPE_CHECKING, Any, Mapping, Optional, Sequence, Union

import numpy as _np
import scipy.linalg as _spl
from numpy.typing import ArrayLike as _ArrayLike, DTypeLike as _DTypeLike

from ._reduction import CircuitSelection as _CircuitSelection
from ._reduction import DesignReducer as _DesignReducer

if TYPE_CHECKING:
    from pygsti.circuits.circuit import Circuit
    from pygsti.models.model import Model
    from pygsti.protocols.protocol import ExperimentDesign

__all__ = [
    'BlockDoptReducer',
    'block_linear_dopt',
    'greedy_candidate_scores',
    'greedy_path_log_volumes',
    'perturb_errorgen_rates',
    'rank_circuits_by_dopt',
]


def _validate_inputs(A: _ArrayLike, block_size: int, max_blocks: int) -> _np.ndarray:
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


def _candidate_blocks(A: _np.ndarray, block_size: int) -> _np.ndarray:
    """`(m, n_candidates * b)` -> `(n_candidates, m, b)` stack with `out[i] = A_i`.

    A view when `A` is C-contiguous; otherwise `reshape` copies.  `A` is never
    written to either way.
    """
    m, n = A.shape
    return A.reshape(m, n // block_size, block_size).transpose(1, 0, 2)


def _scipy_qr_is_batched() -> bool:
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


def _batched_qr_r(W: _np.ndarray) -> _np.ndarray:
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


def block_linear_dopt(A: _ArrayLike, block_size: int, max_blocks: int) -> tuple[_np.ndarray, _np.ndarray]:
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

    Returns
    -------
    block_pivots : numpy.ndarray
        int64, length `min(n_candidates, max_blocks)`.  0-based candidate
        indices in selection order.

    scores : numpy.ndarray
        float64, same length.  `scores[k]` is the objective value after the
        block chosen at step `k` is added, i.e.
        `0.5 * logdet(I_m + sum_{i in S_k} A_i A_i^T)` with `S_k` the first
        `k+1` selected blocks.  The curve is nondecreasing.

    Notes
    -----
    Write `M_S = I_m + sum_{i in S} A_i A_i^T` for the regularized Gram matrix
    of the design chosen so far, and `R_S` for its upper-triangular factor,
    `R_S^T R_S = M_S`.  By the matrix determinant lemma,

        0.5*logdet(M_{S+i}) = 0.5*logdet(M_S) + 0.5*logdet(I_b + G_i G_i^T),

    where `G_i = A_i^T R_S^-1` is candidate `i`'s row block expressed in the
    current factor's coordinates.  The second term is candidate `i`'s *gain*,
    and it equals `sum_j log|rho_jj|` for the R factor `rho_i` of the
    `(m + b_sz) x b_sz` panel

        P_i = [ G_i^T ]
              [  I_b  ]

    Each greedy step therefore QRs one such panel per remaining candidate and
    takes the largest gain.  The running total of the winning gains is the
    returned score curve.

    The algorithm carries `G` rather than `A` and updates it in place.  When
    candidate `i*` is selected, `R_new = C R_S` with `C^T C = I_m + G_{i*}^T
    G_{i*}`, so `C` is the R factor of the `(b_sz + m) x m` matrix
    `[G_{i*} ; I_m]` (one QR per step), and every surviving `G_i` becomes
    `G_i C^-1`.  Since `C^T C >= I_m`, `||C^-1|| <= 1`, so these updates are
    contractions and do not amplify earlier rounding.

    Storage is a single row-major buffer holding the augmented candidate matrix:
    row block `i` is `[ G_i | I_b ]`, of shape `b_sz x (m + b_sz)`, so
    transposing it yields the panels `P_i` in the column-major order LAPACK's
    `geqrf` wants.  The trailing identity columns are constant, which makes the
    update `G <- G C^-1` one in-place triangular solve of the survivors' rows
    against `diag(C, I_b)`; a selection retires the winner by swapping its row
    block past the active boundary.  All buffers are allocated once, before the
    loop.

    Those swaps permute the survivors, so ties are resolved by the lowest
    *original* candidate index.

    The identity block in `P_i` is a unit ridge on the Gram matrix.
    Callers who want a different ridge `lambda` should scale `A` by
    `lambda**-0.5` before calling, which turns the objective into
    `0.5 * logdet(lambda * I + J_S^T J_S)` up to an additive constant.  Because
    `P_i^T P_i = I_b + G_i G_i^T` is at least `I_b`, every candidate has a finite
    nonnegative gain: a rank-deficient block gains less than a full-rank one and
    an all-zero block gains exactly zero.
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
        return block_pivs, best_scores

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

        # ---- Step 2: gain = sum_j log|rho_jj|; lowest-index maximizer. -----
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

    return block_pivs, best_scores


# --------------------------------------------------------------------------- #
#  Independent (QR-free) scorers
# --------------------------------------------------------------------------- #

def _gram_blocks(A: _ArrayLike, block_size: int) -> _np.ndarray:
    """float64 `(n_candidates, m, m)` stack of `A_i A_i^T` (one stacked matmul)."""
    A = _np.asarray(A, dtype=_np.float64)
    blocks = _candidate_blocks(A, block_size)                       # (n_cand, m, b)
    return blocks @ blocks.transpose(0, 2, 1)


def _half_logdet(M: _np.ndarray) -> _np.ndarray:
    """Batched `0.5 * logdet(M)` over a stack of regularized Gram matrices.

    Every `M` handed here is `I_m` plus a sum of Gram matrices, so it is
    symmetric positive definite with eigenvalues at least 1.  With `L` its
    Cholesky factor, `0.5 * logdet(M) = sum_j log(L_jj)`.
    """
    L = _np.linalg.cholesky(M)
    return _np.log(_np.diagonal(L, axis1=-2, axis2=-1)).sum(axis=-1)


def greedy_candidate_scores(A: _ArrayLike, block_size: int, selected: Sequence[int] = ()) -> _np.ndarray:
    """Score of every candidate block given an already-selected prefix.

    This is the objective :func:`block_linear_dopt` maximizes at each step,
    evaluated in float64 via a Cholesky factorization rather than via QR.  It shares no
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


def greedy_path_log_volumes(A: _ArrayLike, block_size: int, block_pivots: Sequence[int]) -> _np.ndarray:
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

        so `out[0] == 0` and `out[1:]` is the score curve `block_linear_dopt`
        returns for the same order.  Use it to see where
        objective gains become small, or to score a selection the kernel did
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


# --------------------------------------------------------------------------- #
#  Applying the kernel to a model and an experiment design
# --------------------------------------------------------------------------- #

def _jacobian_dict_to_array(jac_dict: Mapping[Circuit, Mapping[Any, _np.ndarray]]) -> tuple[_np.ndarray, int]:
    """Flatten a `bulk_dprobs` result into a Fortran-ordered Jacobian and its block size.

    Row blocks follow `jac_dict` key order, which `bulk_dprobs` may have deduplicated and
    reordered: map block `i` back through `list(jac_dict)[i]`, not the input list.
    """
    if not jac_dict:
        raise ValueError("_jacobian_dict_to_array: got an empty Jacobian dict.")
    outcome_counts = {len(per_circuit) for per_circuit in jac_dict.values()}
    if len(outcome_counts) != 1:
        raise ValueError(
            "_jacobian_dict_to_array requires a uniform outcome count per circuit, so that "
            f"one block is one circuit; got varying counts {sorted(outcome_counts)}."
        )
    rowblocks = [_np.vstack(list(per_circuit.values())) for per_circuit in jac_dict.values()]
    return _np.asfortranarray(_np.vstack(rowblocks)), outcome_counts.pop()


def perturb_errorgen_rates(model: Model, scale: float = 1e-3,
                           seed: Union[int, _np.random.Generator, None] = None) -> Model:
    """A copy of `model` with seeded Hamiltonian and stochastic error-generator coefficients.

    D-optimal selection needs the Jacobian at a plausible noisy point, not at a target
    model. Stochastic coefficients use `param_mode='cholesky'` (`coefficient = theta**2`),
    so `d(coefficient)/d(theta) = 2*theta` is exactly zero at the target: every
    stochastic column of the Jacobian vanishes and the selection ignores those parameters.

    For each member, Hamiltonian coefficients are sampled independently and all
    stochastic (S) coefficients share one positive sample; C and A coefficients are set to
    zero. This keeps CPTPLND members physical, respects tied depolarizing parameters, and
    keeps the original parameterization. Members with no error generator are left alone.

    Parameters
    ----------
    model : Model
        Not modified; a copy is returned.

    scale : float, optional (default 1e-3)
        Positive, finite coefficient magnitude. Hamiltonian coefficients are drawn from
        `[0, scale)` and the stochastic coefficient from `(0, scale]`.

    seed : int or numpy.random.Generator, optional
        Anything `numpy.random.default_rng` accepts.  Pass one, or the selection
        is not reproducible.

    Returns
    -------
    Model
    """
    if not _np.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be positive and finite.")
    perturbed = model.copy()
    rng = _np.random.default_rng(seed)
    for _, member in perturbed._iter_parameterized_objs():
        getter = getattr(member, 'errorgen_coefficients', None)
        setter = getattr(member, 'set_errorgen_coefficients', None)
        if getter is None or setter is None:
            continue
        coefficients = getter()
        if not coefficients:
            continue
        stochastic = scale * (1.0 - rng.random())
        sampled = {}
        for lbl in coefficients:
            if lbl.errorgen_type == 'H':
                sampled[lbl] = scale * rng.random()
            elif lbl.errorgen_type == 'S':
                sampled[lbl] = stochastic
            else:  # C and A are off-diagonal non-Hamiltonian coefficients.
                sampled[lbl] = 0.0
        setter(sampled, truncate=False)
    return perturbed


def rank_circuits_by_dopt(model: Model, circuits: Sequence[Circuit], max_circuits: Optional[int] = None, *,
                          ridge: float = 1.0, dtype: _DTypeLike = _np.float64) -> tuple[list[Circuit], _np.ndarray]:
    """Order circuits greedily by the objective :class:`BlockDoptReducer` documents.

    Parameters
    ----------
    model : Model
        A model at a noisy point; see :func:`perturb_errorgen_rates`.

    circuits : list of Circuit
        Candidates.  Duplicates are dropped, keeping first occurrence.

    max_circuits : int, optional
        How many to rank.  None (the default) ranks all of them.

    ridge : float, optional (default 1.0)
        Weight of the identity regularizer; see :func:`block_linear_dopt` for how it is applied.

    dtype : numpy dtype, optional (default numpy.float64)
        Working precision.  float32 halves the memory and changes selections at
        rounding level, which matters only among near-tied candidates.

    Returns
    -------
    ranked : list of Circuit
        In selection order, so `ranked[:k]` is the greedy choice of `k`.

    scores : numpy.ndarray
        `0.5 * logdet(I + J_S^T J_S / ridge)` after each pick; nondecreasing.

    Notes
    -----
    The greedy loop's QRs are small enough that it is dispatch-bound rather
    than flop-bound, so it can run *faster* under a small BLAS thread pool than
    an unrestricted one.  If ranking time matters, measure before assuming more
    threads will help.
    """
    if ridge <= 0:
        raise ValueError(f"rank_circuits_by_dopt: ridge must be positive, got {ridge}.")
    if max_circuits is not None and max_circuits < 0:
        raise ValueError(f"rank_circuits_by_dopt: max_circuits must be nonnegative, got {max_circuits}.")
    unique = list(dict.fromkeys(circuits))
    if not unique or max_circuits == 0:
        return [], _np.empty(0, dtype=_np.float64)
    jac_dict = model.sim.bulk_dprobs(unique)
    jacobian, block_size = _jacobian_dict_to_array(jac_dict)
    # bulk_dprobs may reorder and deduplicate, so block i is jac_keys[i], not unique[i].
    jac_keys = list(jac_dict)

    A = _np.asarray(jacobian, dtype=dtype).T
    if ridge != 1.0:
        A = _np.asarray(A * (ridge ** -0.5), dtype=dtype)
    if max_circuits is None:
        max_circuits = len(jac_keys)

    pivots, scores = block_linear_dopt(A, block_size, max_circuits)
    ranked = [jac_keys[int(i)] for i in pivots]
    return ranked, scores


# --------------------------------------------------------------------------- #
#  The reducer object
# --------------------------------------------------------------------------- #

def _looks_like_a_target_model(model: Model) -> bool:
    """Whether `model` has error-generator coefficients and all of them are zero.

    Only drives a warning, so any failure to inspect the model reads as False.
    """
    try:
        members = list(model._iter_parameterized_objs())
    except Exception:
        return False

    saw_one = False
    for _, member in members:
        getter = getattr(member, 'errorgen_coefficients', None)
        if getter is None:
            continue
        try:
            coefficients = getter()
        except Exception:
            continue
        for value in coefficients.values():
            saw_one = True
            if value != 0:
                return False
    return saw_one


class BlockDoptReducer(_DesignReducer):
    """Select circuits by regularized, unweighted probability-Jacobian sensitivity.

    Greedy block D-optimal selection: each step takes the circuit that most increases
    `0.5 * logdet(ridge * I + J_S^T J_S)`, where `J_S` stacks the Jacobian rows of the
    circuits chosen so far.  Outcomes are weighted equally and sensitivity is measured in
    the model's parameter coordinates, so this is not multinomial Fisher information,
    which would also weight by inverse probabilities and shot counts.

    Parameters
    ----------
    model : Model
        A noisy model; see :func:`perturb_errorgen_rates`.  Its parameterization defines
        the objective.

    ridge : float, optional (default 1.0)
        Weight of the identity regularizer on the Jacobian Gram matrix.

    dtype : numpy dtype, optional (default numpy.float64)
        Working precision; see :func:`rank_circuits_by_dopt`.

    warn_on_target_model : bool, optional (default True)
        Warn if every error-generator coefficient of `model` is zero.
    """

    def __init__(self, model: Model, *, ridge: float = 1.0, dtype: _DTypeLike = _np.float64,
                 warn_on_target_model: bool = True) -> None:
        super().__init__()
        if ridge <= 0:
            raise ValueError(f"BlockDoptReducer: ridge must be positive, got {ridge}.")
        self.model = model
        self.ridge = float(ridge)
        self.dtype = _np.dtype(dtype)
        if warn_on_target_model and _looks_like_a_target_model(model):
            _warnings.warn(
                "BlockDoptReducer was given a model whose error-generator coefficients are "
                "all zero, where stochastic Jacobian columns vanish. Use "
                "perturb_errorgen_rates(model, seed=...), or pass warn_on_target_model=False.")

    def _select(self, design: ExperimentDesign, num_circuits: Optional[int]) -> _CircuitSelection:
        candidates = list(design.all_circuits_needing_data)
        ranked, scores = rank_circuits_by_dopt(
            self.model, candidates,
            len(candidates) if num_circuits is None else num_circuits,
            ridge=self.ridge, dtype=self.dtype)
        return _CircuitSelection(
            ranked, scores=scores,
            score_name='0.5*logdet(I + J_S^T J_S / ridge)',
            metadata={'ridge': self.ridge, 'dtype': str(self.dtype)})

    def _to_nice_serialization(self) -> dict[str, Any]:
        state = super()._to_nice_serialization()
        state.update({'model': self.model.to_nice_serialization(),
                      'ridge': self.ridge,
                      'dtype': str(self.dtype)})
        return state

    @classmethod
    def _from_nice_serialization(cls, state: dict[str, Any]) -> BlockDoptReducer:
        from pygsti.models.model import Model as _Model
        # The model came from a prior instance, so it has already been vetted (or the
        # warning already issued); re-warning on every load would be noise.
        return cls(_Model.from_nice_serialization(state['model']),
                   ridge=state['ridge'], dtype=_np.dtype(state['dtype']),
                   warn_on_target_model=False)

"""
Support discovery for detector error models from symmetric CP decompositions
of joint cumulant tensors.

Model
-----
A detector error model (DEM) with R independent events says that the detector
indicator vector Y in {0,1}^n is

    Y = B Z  (mod 2),      Z_j ~ Bernoulli(p_j) independent,

where B is the binary n x R event matrix whose column b_j is the *signature*
of event j (the set of detectors it flips).  Support discovery means finding
the columns of B from syndrome data; the probabilities are then refit with
`estimation.fit_specified_dem`.

Cumulant tensors are (leading-order) symmetric CP decompositions
----------------------------------------------------------------
Let K_k[d_1, ..., d_k] = kappa(Y_{d_1}, ..., Y_{d_k}) be the order-k joint
cumulant tensor of the detector indicators (repeated indices allowed).  To
leading order in the event probabilities,

    K_k  ~=  sum_j  p_j  b_j (x) b_j (x) ... (x) b_j        (k copies),

a symmetric rank-R CP decomposition whose factors are the binary event
signatures and whose weights are the event probabilities.  Sketch of the
argument: for a single event, Y = b_j Z_j exactly, and by multilinearity
kappa(b_{j,d_1} Z_j, ..., b_{j,d_k} Z_j) = (prod_i b_{j,d_i}) kappa_k(Z_j)
with kappa_k(Z_j) = p_j + O(p_j^2) for a Bernoulli variable.  For several
events, Y_d = XOR_j b_{j,d} Z_j = sum_j b_{j,d} Z_j - 2 (pairwise products)
+ ..., so Y differs from the *real* sum B Z only by terms that are products
of two or more independent Bernoullis, i.e. by O(p^2).  Cumulants are
additive over independent summands, so K_k(B Z) = sum_j kappa_k(Z_j)
b_j^{(x)k} exactly for the real sum, and the mod-2 correction perturbs every
entry by O(p^2).  Entries with repeated indices are included in this
statement because for a binary Y_a, (Y_a - mu_a)^2 = (1 - 2 mu_a) Y_a +
mu_a^2, so e.g. kappa(Y_a, Y_a, Y_b) = (1 - 2 mu_a) Cov(Y_a, Y_b) ~=
sum_{j contains a, b} p_j, again to leading order.

Why order 3
-----------
For k = 2 the tensor is the covariance matrix used by the "p_ij" method
(Google's pairwise correlation analysis).  A symmetric matrix factorization
is *not* unique: a weight-3 hyperedge {D0 D1 D2} with probability p and the
triangle of pairs {D0 D1}, {D1 D2}, {D0 D2} each with probability p produce
the same covariance to leading order.  For k = 3, Kruskal's theorem gives
uniqueness of the CP decomposition (up to permutation/scaling) whenever the
k-ranks of the three (identical) factor matrices satisfy 3 k_B >= 2R + 2,
which holds for generic, sufficiently different binary signatures.  In the
hyperedge/triangle example kappa_3(Y_0, Y_1, Y_2) ~= p for the hyperedge
but is O(p^2) (in fact -6 p^2 + O(p^3)) for the triangle, because two pair
events can never flip an odd number of the three detectors.

Honesty about the approximation
-------------------------------
All of the above holds to *leading order* in the event probabilities.  The
CP fit is a support-discovery device: rounded factors become candidate
event masks, and exact probabilities come from `fit_specified_dem`, which
uses the full (non-perturbative) product formula for the outcome
distribution.  A dense order-3 tensor has m^3 entries for m detectors, so
the practical limit of the dense path is m of a few hundred; use the
`detectors` argument to work on subsets.

Pipeline (`cp_dem_estimation`)
------------------------------
1. `cumulant_tensors`: exact sample cumulants of orders 2..k (k <= 4) from the
   syndrome counts, plus per-entry standard errors.
2. `symmetric_cp`: whitened, bound-constrained symmetric CP fit
   (L-BFGS-B on weights >= 0 and factors in [0, 1], several seeded restarts,
   optionally coupling the order-2 and order-3 tensors through shared
   factors).
3. Initialization and rank selection (`structured_init`, `select_rank`):
   the fit starts from the nonnegative least-squares solution on the binary
   dictionary of supports suggested by significant tensor entries (sets of
   at most k detectors); the continuous CP refinement can merge these into
   higher-weight signatures.  Components are then added greedily from the
   most under-explained entry until an added component is no longer
   significant by a likelihood-ratio test (with the noise scale inflated by
   the reduced chi^2, since at large shot counts the O(p^2) model error,
   not the sampling noise, dominates the residual) or the residual is
   consistent with the entry standard errors.
4. `factors_to_masks`: round factors to binary signatures, deduplicate, refit
   the weights by nonnegative least squares on the rounded dictionary and
   drop components whose weight is not significant.
5. `estimation.fit_specified_dem` on the recovered masks, followed by a
   z-test on the refit probabilities (delta-method covariance) that removes
   the spurious low-weight events the leading-order model can produce from
   O(p^2) structure, and a final refit.

Regime of validity: the continuous CP relaxation is identifiable only while
R is below the generic symmetric rank (about C(m+2, 3) / m for order 3);
the binary constraint, the order-2 coupling and the structured
initialization extend recovery beyond that (e.g. 21 events on 8 detectors),
but very dense DEMs on few detectors are the unfavourable case for this
method.

Bit conventions follow `sparsedem.utils`: sample matrices have column d =
detector d and integer masks have bit d = detector d.
"""

from __future__ import annotations

import itertools
import math
import warnings
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import scipy.optimize
import scipy.sparse
import scipy.stats
import stim

from .estimation import (
    _ALL_PAIRS_POL_LIMIT,
    compute_outcome_distribution_from_dem,
    fit_specified_dem,
)
from .io import dem_to_matrix
from .utils import counts_to_detector_arrays

_MAX_ORDER = 4
_EINSUM_LETTERS = "abcdefgh"


# ---------------------------------------------------------------------------
# Data wrangling
# ---------------------------------------------------------------------------

def _samples_and_weights(data, weights=None):
    """Normalize the universal input to (samples, weights).

    `data` is either a sparsedem `syndrome_counts` dict (bitstring keys in the
    reversed/decreasing convention) or a (K, n) {0,1} array with column d =
    detector d.  Weights default to 1 per row for arrays.
    """
    if isinstance(data, dict):
        if weights is not None:
            raise ValueError("weights cannot be combined with a counts dict.")
        samples, weights = counts_to_detector_arrays(data)
        return samples, weights.astype(float)
    samples = np.asarray(data)
    if samples.ndim != 2:
        raise ValueError("samples must be a 2-D {0,1} array.")
    samples = np.ascontiguousarray(samples, dtype=np.uint8)
    if weights is None:
        weights = np.ones(samples.shape[0], dtype=float)
    else:
        weights = np.asarray(weights, dtype=float)
        if weights.shape != (samples.shape[0],):
            raise ValueError("weights must have one entry per sample row.")
    return samples, weights


def _select_detectors(n_detectors: int, detectors) -> np.ndarray:
    if detectors is None:
        return np.arange(n_detectors)
    detectors = np.asarray(list(detectors), dtype=int)
    if detectors.ndim != 1 or len(detectors) == 0:
        raise ValueError("detectors must be a non-empty 1-D sequence.")
    if detectors.min() < 0 or detectors.max() >= n_detectors:
        raise ValueError("detectors out of range.")
    if len(np.unique(detectors)) != len(detectors):
        raise ValueError("detectors must be distinct.")
    return detectors


# ---------------------------------------------------------------------------
# Raw / central moments and cumulants
# ---------------------------------------------------------------------------

def _raw_tensor(Xs: scipy.sparse.csr_matrix, w: np.ndarray, order: int) -> np.ndarray:
    """Unnormalized raw moment tensor sum_s w_s x_s^{(x) order} (sparse rows)."""
    m = Xs.shape[1]
    if order == 1:
        return np.asarray(Xs.T @ w).ravel()
    Xw = scipy.sparse.diags(w) @ Xs
    if order == 2:
        return np.asarray((Xs.T @ Xw).todense())
    out = np.zeros((m,) * order, dtype=float)
    Xc = Xs.tocsc()
    for a in range(m):
        rows = Xc.indices[Xc.indptr[a]:Xc.indptr[a + 1]]
        if len(rows) == 0:
            continue
        out[a] = _raw_tensor(Xs[rows], w[rows], order - 1)
    return out


def _raw_moment_tensors(samples: np.ndarray, weights: np.ndarray, order: int) -> list:
    """Normalized raw moment tensors [1, E[X], E[X (x) X], ...] up to `order`.

    Only shots with fired detectors contribute, so the cost is
    sum_s (fired_s)^order plus the m^order dense output.
    """
    total = float(weights.sum())
    if total <= 0:
        raise ValueError("total weight must be positive.")
    Xs = scipy.sparse.csr_matrix(samples.astype(float))
    raws = [np.asarray(1.0)]
    for k in range(1, order + 1):
        raws.append(_raw_tensor(Xs, weights, k) / total)
    return raws


def _subset_expansion(raws: list, u: np.ndarray, v: np.ndarray, order: int) -> np.ndarray:
    """Evaluate E[prod_i (u_{a_i} + v_{a_i} X_{a_i})] as an order-k tensor.

    Expanding the product over subsets S of the k slots gives
    sum_S (prod_{i not in S} u_{a_i}) (prod_{i in S} v_{a_i}) R_{|S|}[a_S],
    where R_j are the normalized raw moment tensors.  Binary X makes the
    repeated-index entries automatically right (R_2[a, a] = R_1[a], ...).
    """
    m = len(u)
    out = np.zeros((m,) * order, dtype=float)
    for r in range(order + 1):
        for S in itertools.combinations(range(order), r):
            shape = [m if i in S else 1 for i in range(order)]
            term = np.asarray(raws[r]).reshape(shape) if r > 0 else np.full([1] * order, float(raws[0]))
            for i in range(order):
                vec = v if i in S else u
                sh = [1] * order
                sh[i] = m
                term = term * vec.reshape(sh)
            out = out + term
    return out


def _symmetrized_pair_products(M2: np.ndarray) -> np.ndarray:
    """M2_ab M2_cd + M2_ac M2_bd + M2_ad M2_bc (the order-4 cumulant correction)."""
    return (np.einsum("ab,cd->abcd", M2, M2)
            + np.einsum("ac,bd->abcd", M2, M2)
            + np.einsum("ad,bc->abcd", M2, M2))


@dataclass
class CumulantTensors:
    """Sample joint cumulant tensors of detector indicators.

    Attributes:
        detectors: np.ndarray
            Global detector indices; axis i of every tensor indexes
            `detectors[i]`.
        n_shots: float
            Total weight (number of shots).
        mean: np.ndarray
            Detector click frequencies over the selected detectors.
        tensors: dict[int, np.ndarray]
            Cumulant tensors keyed by order (2 .. order).  For order 3 the
            joint central moment equals the joint cumulant; order 4 carries
            the moment-to-cumulant correction.
        stderrs: dict[int, np.ndarray]
            Per-entry standard errors: the sample standard deviation of the
            per-shot centered products divided by sqrt(n_shots).  This
            ignores the (higher-order) contribution from estimating the
            means, which is negligible for rare events; for order 4 it is
            the standard error of the central moment.
    """

    detectors: np.ndarray
    n_shots: float
    mean: np.ndarray
    tensors: dict = field(default_factory=dict)
    stderrs: dict = field(default_factory=dict)

    @property
    def num_detectors(self) -> int:
        return len(self.detectors)


def cumulant_tensors(data, order: int = 3, detectors=None, weights=None) -> CumulantTensors:
    """
    Compute joint cumulant tensors of orders 2..`order` from syndrome data.

    Parameters:
        data: dict or np.ndarray
            Either a sparsedem `syndrome_counts` dict, or a (K, n) {0,1}
            sample matrix with column d = detector d (use `weights` for
            counts or for an exact outcome distribution).
        order: int
            Highest cumulant order (2, 3 or 4).
        detectors: Optional[Sequence[int]]
            Subset of detector indices to include (default all).  Dense
            order-3 tensors need m^3 floats, so keep m below a few hundred.
        weights: Optional[np.ndarray]
            Row weights for array input (counts or probabilities).

    Returns:
        CumulantTensors
    """
    if order < 2 or order > _MAX_ORDER:
        raise ValueError(f"order must be between 2 and {_MAX_ORDER}.")
    samples, weights = _samples_and_weights(data, weights)
    dets = _select_detectors(samples.shape[1], detectors)
    X = np.ascontiguousarray(samples[:, dets])
    total = float(weights.sum())

    raws = _raw_moment_tensors(X, weights, order)
    mu = raws[1]
    ones = np.ones_like(mu)
    result = CumulantTensors(detectors=dets, n_shots=total, mean=mu)
    central = {}
    for k in range(2, order + 1):
        Mk = _subset_expansion(raws, -mu, ones, k)
        central[k] = Mk
        second = _subset_expansion(raws, mu ** 2, ones - 2.0 * mu, k)
        var = np.maximum(second - Mk ** 2, 0.0)
        result.stderrs[k] = np.sqrt(var / total)
        if k <= 3:
            result.tensors[k] = Mk
        else:
            result.tensors[k] = Mk - _symmetrized_pair_products(central[2])
    return result


def joint_cumulant_tensor(data, order: int = 3, detectors=None, weights=None,
                          return_stderr: bool = False):
    """
    Dense symmetric joint cumulant tensor of the detector indicators.

    Parameters:
        data: dict or np.ndarray
            `syndrome_counts` dict or (K, n) sample matrix (column d =
            detector d).
        order: int
            Cumulant order (2, 3 or 4).
        detectors: Optional[Sequence[int]]
            Detector subset (default all); the output has shape (m,) * order.
        weights: Optional[np.ndarray]
            Row weights for array input.
        return_stderr: bool
            Also return the per-entry standard-error tensor.

    Returns:
        tensor: np.ndarray, or (tensor, stderr) if `return_stderr`.
    """
    ct = cumulant_tensors(data, order=order, detectors=detectors, weights=weights)
    if return_stderr:
        return ct.tensors[order], ct.stderrs[order]
    return ct.tensors[order]


def _dem_event_matrix(dem: stim.DetectorErrorModel):
    """(n, R) binary event matrix in detector order (row d = detector d) and probabilities."""
    B_msb, probs = dem_to_matrix(dem)
    return np.ascontiguousarray(B_msb[::-1, :]).astype(float), probs


def leading_order_cumulant_tensor(dem: stim.DetectorErrorModel, order: int = 3,
                                  detectors=None) -> np.ndarray:
    """
    The leading-order model tensor sum_j p_j b_j^{(x) order} of a DEM.

    Parameters:
        dem: stim.DetectorErrorModel
        order: int
        detectors: Optional[Sequence[int]]
            Detector subset (default all `dem.num_detectors`).

    Returns:
        np.ndarray of shape (m,) * order.
    """
    B, probs = _dem_event_matrix(dem)
    dets = _select_detectors(B.shape[0], detectors)
    return cp_reconstruct(probs, B[dets, :], order)


def exact_cumulant_tensor_from_dem(dem: stim.DetectorErrorModel, order: int = 3,
                                   detectors=None) -> np.ndarray:
    """
    Exact population cumulant tensor of a (small) DEM via its 2^n outcome
    distribution (`compute_outcome_distribution_from_dem`).

    Parameters:
        dem: stim.DetectorErrorModel
        order: int
        detectors: Optional[Sequence[int]]

    Returns:
        np.ndarray of shape (m,) * order.
    """
    n = dem.num_detectors
    probs = compute_outcome_distribution_from_dem(dem)
    idx = np.arange(2 ** n)
    # Outcome index i has bit d set iff detector d fired.
    X = ((idx[:, None] >> np.arange(n)[None, :]) & 1).astype(np.uint8)
    return joint_cumulant_tensor(X, order=order, detectors=detectors, weights=probs)


# ---------------------------------------------------------------------------
# Symmetric CP model
# ---------------------------------------------------------------------------

def cp_reconstruct(weights: np.ndarray, factors: np.ndarray, order: int) -> np.ndarray:
    """
    Evaluate the symmetric CP model sum_r weights[r] factors[:, r]^{(x) order}.

    Parameters:
        weights: np.ndarray
            (R,) component weights.
        factors: np.ndarray
            (m, R) factor matrix.
        order: int

    Returns:
        np.ndarray of shape (m,) * order.
    """
    weights = np.asarray(weights, dtype=float)
    factors = np.asarray(factors, dtype=float)
    letters = _EINSUM_LETTERS[:order]
    if factors.shape[1] == 0:
        return np.zeros((factors.shape[0],) * order)
    subscripts = "r," + ",".join(L + "r" for L in letters) + "->" + letters
    return np.einsum(subscripts, weights, *([factors] * order), optimize=True)


def _symmetry_multiplicity(m: int, order: int) -> np.ndarray:
    """Number of distinct index permutations of every entry of an order-k tensor."""
    if order == 1:
        return np.ones(m)
    idx = [np.arange(m).reshape([m if i == j else 1 for i in range(order)]) for j in range(order)]
    n_eq = np.zeros((m,) * order, dtype=int)
    for i, j in itertools.combinations(range(order), 2):
        n_eq = n_eq + (idx[i] == idx[j])
    if order == 2:
        table = {0: 2, 1: 1}
    elif order == 3:
        table = {0: 6, 1: 3, 3: 1}
    elif order == 4:
        table = {0: 24, 1: 12, 2: 6, 3: 4, 6: 1}
    else:
        raise ValueError("order must be <= 4")
    out = np.zeros_like(n_eq, dtype=float)
    for key, val in table.items():
        out[n_eq == key] = val
    return out


def _unique_entry_mask(m: int, order: int) -> np.ndarray:
    """Boolean tensor selecting the entries with non-decreasing indices."""
    idx = [np.arange(m).reshape([m if i == j else 1 for i in range(order)]) for j in range(order)]
    mask = np.ones((m,) * order, dtype=bool)
    for j in range(order - 1):
        mask = mask & (idx[j] <= idx[j + 1])
    return mask


class _CPProblem:
    """Whitened least-squares objective for one or more coupled symmetric tensors."""

    def __init__(self, tensors, m: int, rank: int):
        # tensors: list of (T, W2, order) with W2 = per-entry squared weight
        self.tensors = tensors
        self.m = m
        self.rank = rank

    def unpack(self, x):
        lam = x[:self.rank]
        F = x[self.rank:].reshape(self.m, self.rank)
        return lam, F

    def __call__(self, x):
        lam, F = self.unpack(x)
        obj = 0.0
        g_lam = np.zeros_like(lam)
        g_F = np.zeros_like(F)
        for T, W2, k in self.tensors:
            model = cp_reconstruct(lam, F, k)
            resid = model - T
            E = W2 * resid
            obj += 0.5 * float(np.sum(E * resid))
            letters = _EINSUM_LETTERS[:k]
            g_lam += np.einsum(letters + "," + ",".join(L + "r" for L in letters) + "->r",
                               E, *([F] * k), optimize=True)
            if k == 1:
                G = E[:, None] * np.ones((1, self.rank))
            else:
                G = np.einsum(letters + "," + ",".join(L + "r" for L in letters[1:]) + "->" + letters[0] + "r",
                              E, *([F] * (k - 1)), optimize=True)
            g_F += k * lam[None, :] * G
        return obj, np.concatenate([g_lam, g_F.ravel()])


def _prepare_weights(T, stderr, mask, se_floor_rel=1e-3):
    """Per-entry squared weights for the whitened objective (unique-entry counting)."""
    m = T.shape[0]
    k = T.ndim
    mult = _symmetry_multiplicity(m, k)
    if stderr is None:
        W2 = np.ones_like(T)
    else:
        se = np.asarray(stderr, dtype=float)
        floor = se_floor_rel * se.max() if se.max() > 0 else 1.0
        W2 = 1.0 / np.maximum(se, floor) ** 2
    if mask is not None:
        W2 = W2 * np.asarray(mask, dtype=float)
    return W2 / mult


def symmetric_cp(tensor: np.ndarray, rank: int, *, stderr=None, mask=None, coupled=None,
                 weights_nonneg: bool = True, factor_bounds=(0.0, 1.0), n_restarts: int = 3,
                 seed: Optional[int] = 0, max_iter: int = 300, tol: float = 1e-10, init=None):
    """
    Fit a symmetric CP decomposition T ~= sum_r w_r a_r^{(x) k}.

    The fit minimizes the whitened least squares 1/2 sum_unique ((model -
    T) / stderr)^2 over the weights and the shared factor matrix with
    L-BFGS-B under box constraints (weights >= 0, factors in
    `factor_bounds`).  Each unique entry (sorted index tuple) counts once.
    Several seeded random restarts are run and the best objective is kept,
    so the result is deterministic for a given `seed`.  The returned
    components are gauge-fixed so that the largest entry of each factor is
    1 (the scale ambiguity lam a^{(x)k} = (lam s^k)(a/s)^{(x)k} is otherwise
    unresolved by the box constraints); with `factor_bounds=(0, 1)` this can
    only increase factor entries, so the factors stay inside the bounds.

    Parameters:
        tensor: np.ndarray
            Symmetric array of shape (m,) * k, k in {2, 3, 4}.
        rank: int
            Number of components.
        stderr: Optional[np.ndarray]
            Per-entry standard errors used as inverse weights (None: unweighted).
        mask: Optional[np.ndarray]
            Boolean array; entries with False get zero weight (screened fits).
        coupled: Optional[list]
            Additional tensors sharing the same factors and weights, given as
            (tensor, stderr) or (tensor, stderr, mask) tuples of possibly
            different order (e.g. the covariance next to the order-3 tensor).
        weights_nonneg: bool
            Constrain the weights to be nonnegative.
        factor_bounds: tuple
            (lower, upper) bounds on every factor entry; None for unbounded.
        n_restarts: int
            Number of random restarts (the first one uses `init` if given).
        seed: Optional[int]
            Seed for the restart initializations.
        max_iter: int
            L-BFGS-B iteration cap per restart.
        tol: float
            L-BFGS-B `ftol`.
        init: Optional[tuple]
            (weights, factors) warm start; missing columns are filled randomly.

    Returns:
        weights: np.ndarray (R,), sorted decreasing.
        factors: np.ndarray (m, R).
        info: dict with 'objective', 'chi2', 'dof', 'n_unique', 'restart_objectives',
              'relative_residual', 'n_iter'.  'dof' is n_unique - rank
              (weights only; supports count as discrete).
    """
    T = np.asarray(tensor, dtype=float)
    k = T.ndim
    m = T.shape[0]
    if k < 2 or k > _MAX_ORDER or any(s != m for s in T.shape):
        raise ValueError("tensor must be a symmetric array of order 2..4.")
    rank = int(rank)
    if rank < 1:
        raise ValueError("rank must be >= 1.")

    problems = [(T, stderr, mask, k)]
    if coupled:
        for item in coupled:
            Tc = np.asarray(item[0], dtype=float)
            sc = item[1] if len(item) > 1 else None
            mc = item[2] if len(item) > 2 else None
            if Tc.shape[0] != m:
                raise ValueError("coupled tensors must share the detector dimension.")
            problems.append((Tc, sc, mc, Tc.ndim))

    scale = float(np.max(np.abs(T)))
    if scale <= 0:
        scale = 1.0
    scaled = []
    n_unique_total = 0
    for Tt, st, mk, kk in problems:
        W2 = _prepare_weights(Tt, st, mk)
        # Rescale so that the whitened residual (chi^2) is unchanged.
        scaled.append((Tt / scale, W2 * scale ** 2, kk))
        n_unique_total += int(np.count_nonzero(W2[_unique_entry_mask(m, kk)]))
    problem = _CPProblem(scaled, m, rank)

    lo_f, hi_f = factor_bounds if factor_bounds is not None else (None, None)
    bounds = [(0.0 if weights_nonneg else None, None)] * rank + [(lo_f, hi_f)] * (m * rank)
    lo = 0.0 if lo_f is None else lo_f
    hi = 1.0 if hi_f is None else hi_f

    rng = np.random.default_rng(seed)
    best = None
    restart_objs = []
    for r in range(max(1, n_restarts)):
        lam0 = rng.uniform(0.1, 1.0, size=rank)
        F0 = rng.uniform(lo, hi, size=(m, rank))
        if r == 0 and init is not None:
            lam_i = np.asarray(init[0], dtype=float) / scale
            F_i = np.asarray(init[1], dtype=float)
            R0 = min(rank, len(lam_i))
            lam0[:R0] = lam_i[:R0]
            F0[:, :R0] = F_i[:, :R0]
            lam0[R0:] *= 1e-2  # keep the warm start close to the previous optimum
        x0 = np.concatenate([lam0, F0.ravel()])
        res = scipy.optimize.minimize(problem, x0, jac=True, method="L-BFGS-B", bounds=bounds,
                                      options={"maxiter": max_iter, "ftol": tol, "gtol": 1e-10,
                                               "maxfun": 10 * max_iter})
        restart_objs.append(float(res.fun))
        if best is None or res.fun < best.fun:
            best = res

    lam, F = problem.unpack(best.x)
    lam = lam * scale
    # Fix the scale ambiguity lam a^{(x)k} = (lam s^k) (a / s)^{(x)k} in the gauge
    # natural for binary signatures: the largest entry of every factor is 1.
    peak = np.max(np.abs(F), axis=0)
    nz = peak > 0
    F = np.where(nz[None, :], F / np.where(nz, peak, 1.0)[None, :], F)
    lam = np.where(nz, lam * peak ** k, lam)
    order_idx = np.argsort(-lam, kind="stable")
    lam = lam[order_idx]
    F = F[:, order_idx]

    chi2 = 2.0 * float(best.fun)
    model = cp_reconstruct(lam, F, k)
    rel = float(np.linalg.norm(model - T) / max(np.linalg.norm(T), 1e-300))
    # Degrees of freedom count the weights only: the (rounded) supports are
    # discrete choices, and the continuous factor entries would otherwise
    # exceed the entry count in the over-parametrized regime.
    n_params = rank
    info = {
        "objective": float(best.fun),
        "chi2": chi2,
        "n_unique": n_unique_total,
        "dof": max(n_unique_total - n_params, 1),
        "restart_objectives": restart_objs,
        "relative_residual": rel,
        "n_iter": int(best.nit),
        "success": bool(best.success),
    }
    return lam, F, info


# ---------------------------------------------------------------------------
# Rounding, dictionary refit, masks
# ---------------------------------------------------------------------------

def round_factors(factors: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Round a factor matrix to {0,1} at `threshold`."""
    return (np.asarray(factors, dtype=float) >= threshold).astype(np.uint8)


def _dedupe_binary_columns(B: np.ndarray, weights: np.ndarray):
    """Merge identical nonempty columns (summing weights); returns (B, weights)."""
    cols = {}
    order = []
    for j in range(B.shape[1]):
        key = tuple(int(b) for b in B[:, j])
        if not any(key):
            continue
        if key in cols:
            cols[key] += float(weights[j])
        else:
            cols[key] = float(weights[j])
            order.append(key)
    if not order:
        return np.zeros((B.shape[0], 0), dtype=np.uint8), np.zeros(0)
    Bd = np.array(order, dtype=np.uint8).T
    wd = np.array([cols[key] for key in order])
    return Bd, wd


def refit_dictionary_weights(tensors, B: np.ndarray):
    """
    Nonnegative least-squares weights for fixed binary signatures.

    Parameters:
        tensors: list
            (tensor, stderr) or (tensor, stderr, mask) tuples sharing the
            detector axis; the fit is jointly whitened over all of them.
        B: np.ndarray
            (m, J) binary factor matrix.

    Returns:
        weights: np.ndarray (J,)
        stderr: np.ndarray (J,)
            Linearized standard errors from the whitened normal equations.
        chi2: float
    """
    B = np.asarray(B, dtype=float)
    m, J = B.shape
    rows = []
    rhs = []
    for item in tensors:
        T = np.asarray(item[0], dtype=float)
        se = item[1] if len(item) > 1 else None
        mk = item[2] if len(item) > 2 else None
        k = T.ndim
        W = np.sqrt(_prepare_weights(T, se, mk))
        sel = _unique_entry_mask(m, k) & (W > 0)
        # Columns: whitened vec(b_j^{(x)k}) restricted to unique entries.
        D = np.stack([cp_reconstruct(np.ones(1), B[:, [j]], k)[sel] for j in range(J)], axis=1) \
            if J else np.zeros((int(sel.sum()), 0))
        rows.append(D * W[sel][:, None])
        rhs.append((T * W)[sel])
    A = np.concatenate(rows, axis=0)
    y = np.concatenate(rhs)
    if J == 0:
        return np.zeros(0), np.zeros(0), float(y @ y)
    w, _ = scipy.optimize.nnls(A, y)
    resid = A @ w - y
    cov = np.linalg.pinv(A.T @ A)
    se_w = np.sqrt(np.maximum(np.diag(cov), 0.0))
    return w, se_w, float(resid @ resid)


def factors_to_masks(factors: np.ndarray, weights: np.ndarray, detectors=None,
                     threshold: float = 0.5, weight_min: float = 0.0):
    """
    Round CP factors to binary event signatures encoded as integer bitmasks.

    Parameters:
        factors: np.ndarray
            (m, R) factor matrix over the selected detectors.
        weights: np.ndarray
            (R,) component weights.
        detectors: Optional[Sequence[int]]
            Global detector index of each factor row (default identity), so
            that bit `detectors[i]` of a mask corresponds to row i.
        threshold: float
            Rounding threshold for factor entries.
        weight_min: float
            Drop components whose (merged) weight is below this value.

    Returns:
        masks: list[int]
            Distinct nonempty masks (bit d = detector d), sorted by
            decreasing weight.
        weights: np.ndarray
            Merged weights aligned with `masks`.
    """
    factors = np.asarray(factors, dtype=float)
    weights = np.asarray(weights, dtype=float)
    m = factors.shape[0]
    dets = _select_detectors(m if detectors is None else int(np.max(detectors)) + 1, detectors)
    if len(dets) != m:
        raise ValueError("detectors must have one entry per factor row.")
    B, w = _dedupe_binary_columns(round_factors(factors, threshold), weights)
    keep = w > weight_min
    B, w = B[:, keep], w[keep]
    order = np.argsort(-w, kind="stable")
    masks = []
    for j in order:
        mask = 0
        for i in np.nonzero(B[:, j])[0]:
            mask |= 1 << int(dets[i])
        masks.append(mask)
    return masks, w[order]


def masks_to_factors(masks: Sequence[int], detectors) -> np.ndarray:
    """(m, J) binary factor matrix of `masks` restricted to `detectors` (rows)."""
    dets = np.asarray(detectors, dtype=int)
    B = np.zeros((len(dets), len(masks)), dtype=np.uint8)
    for j, mask in enumerate(masks):
        for i, d in enumerate(dets):
            B[i, j] = (int(mask) >> int(d)) & 1
    return B


# ---------------------------------------------------------------------------
# Structured initialization, rank selection and pipeline
# ---------------------------------------------------------------------------

def _unique_entries_z(T, stderr, mask=None):
    """Iterate (index tuple, value, z) over unique entries with positive weight."""
    m = T.shape[0]
    k = T.ndim
    se = np.asarray(stderr, dtype=float)
    for idx in itertools.combinations_with_replacement(range(m), k):
        if mask is not None and not mask[idx]:
            continue
        s = se[idx]
        z = T[idx] / s if s > 0 else (np.inf if T[idx] > 0 else 0.0)
        yield idx, float(T[idx]), float(z)


def candidate_supports(tensors, confidence: float = 0.95) -> list:
    """
    Candidate event supports suggested by significant tensor entries.

    Every unique entry whose z-score T / stderr exceeds the one-sided
    Bonferroni threshold (over all unique entries of all tensors) contributes
    the set of its distinct indices, so a significant kappa(Y_a, Y_a, Y_b)
    proposes {a, b} and kappa(Y_a, Y_b, Y_c) proposes {a, b, c}.  Supports of
    size larger than the tensor order can only arise later, by the continuous
    CP refinement merging components.

    Parameters:
        tensors: list
            (tensor, stderr) or (tensor, stderr, mask) tuples.
        confidence: float

    Returns:
        list of sorted index tuples (local detector indices), sorted by size.
    """
    n_total = 0
    for item in tensors:
        T = np.asarray(item[0])
        n_total += math.comb(T.shape[0] + T.ndim - 1, T.ndim)
    z_thr = scipy.stats.norm.isf((1.0 - confidence) / max(n_total, 1))
    supports = set()
    for item in tensors:
        T = np.asarray(item[0], dtype=float)
        se = item[1]
        mk = item[2] if len(item) > 2 else None
        for idx, _, z in _unique_entries_z(T, se, mk):
            if z > z_thr:
                supports.add(tuple(sorted(set(idx))))
    return sorted(supports, key=lambda s: (len(s), s))


def structured_init(tensors, confidence: float = 0.95, weight_min: float = 0.0):
    """
    Data-driven CP initialization: nonnegative least squares on the binary
    dictionary of `candidate_supports`, keeping the components with positive
    weight.

    Returns:
        weights: np.ndarray (R0,)
        factors: np.ndarray (m, R0) binary
        supports: list of the kept index tuples
    """
    m = np.asarray(tensors[0][0]).shape[0]
    supports = candidate_supports(tensors, confidence)
    if not supports:
        return np.zeros(0), np.zeros((m, 0)), []
    B = np.zeros((m, len(supports)))
    for j, s in enumerate(supports):
        B[list(s), j] = 1.0
    w, _, _ = refit_dictionary_weights(tensors, B)
    keep = w > weight_min
    return w[keep], B[:, keep], [s for s, k in zip(supports, keep) if k]


def _greedy_new_component(T, stderr, mask, lam, F):
    """Factor/weight proposal from the most under-explained unique entry."""
    k = T.ndim
    model = cp_reconstruct(lam, F, k)
    resid = T - model
    best = None
    for idx, _, _ in _unique_entries_z(T, stderr, mask):
        s = stderr[idx]
        z = resid[idx] / s if s > 0 else 0.0
        if best is None or z > best[0]:
            best = (z, idx)
    if best is None:
        return None
    _, idx = best
    f = np.zeros(T.shape[0])
    f[list(set(idx))] = 1.0
    return max(float(resid[idx]), 1e-3 * float(np.max(np.abs(T)))), f


def select_rank(tensor, stderr, *, coupled=None, mask=None, init=None, rank_max: int = 10,
                chi2_tolerance: float = 3.0, growth_alpha: float = 0.05,
                n_restarts: int = 3, seed: Optional[int] = 0, max_iter: int = 300,
                tol: float = 1e-10):
    """
    Increase the CP rank until an added component is no longer justified.

    Starting from rank R0 = len(init weights) (or 1), ranks R0, R0+1, ... are
    fitted in turn.  Each fit is warm-started from the previous solution plus
    a component proposed from the most under-explained entry (the remaining
    restarts are random).  Selection stops

    * as soon as the reduced chi^2 of the whitened residual is below
      1 + chi2_tolerance * sqrt(2 / dof) (residual consistent with the entry
      standard errors), or
    * when going from R-1 to R decreases chi^2 by less than
      reduced_chi2(R-1) * chi2_{m+1}^{-1}(1 - growth_alpha), i.e. the extra
      component is not significant by a likelihood-ratio test whose noise
      scale is inflated by the previous fit's reduced chi^2 (this accounts for
      the O(p^2) model error that dominates the residual at large shot
      counts); rank R-1 is kept.

    Otherwise `rank_max` is used.  Without `stderr` the relative residual
    replaces chi^2 (stop at 1e-6, or when the relative residual decreases by
    less than 1%).

    Returns:
        weights, factors, info (with 'rank' and a per-rank 'path' list).
    """
    T = np.asarray(tensor, dtype=float)
    m = T.shape[0]
    se = None if stderr is None else np.asarray(stderr, dtype=float)
    path = []
    prev = None
    chosen = None
    R0 = 1 if init is None or len(init[0]) == 0 else len(init[0])
    rank_max = max(rank_max, R0)
    q_growth = scipy.stats.chi2.isf(growth_alpha, df=m + 1)
    warm = init
    for R in range(R0, rank_max + 1):
        lam, F, info = symmetric_cp(T, R, stderr=se, mask=mask, coupled=coupled,
                                    n_restarts=n_restarts, seed=None if seed is None else seed + R,
                                    max_iter=max_iter, tol=tol, init=warm)
        stats = {"rank": R, "chi2": info["chi2"], "dof": info["dof"],
                 "reduced_chi2": info["chi2"] / info["dof"],
                 "relative_residual": info["relative_residual"]}
        path.append(stats)
        if se is not None:
            good = stats["reduced_chi2"] <= 1.0 + chi2_tolerance * math.sqrt(2.0 / info["dof"])
        else:
            good = stats["relative_residual"] <= 1e-6
        if good:
            chosen = (lam, F, info, R)
            break
        if prev is not None:
            if se is not None:
                threshold = prev[2]["chi2"] / prev[2]["dof"] * q_growth
                improved = (prev[2]["chi2"] - info["chi2"]) > threshold
            else:
                improved = (prev[2]["relative_residual"] - info["relative_residual"]) > 0.01 * prev[2]["relative_residual"]
            if not improved:
                chosen = (prev[0], prev[1], prev[2], R - 1)
                break
        prev = (lam, F, info)
        proposal = _greedy_new_component(T, se if se is not None else np.ones_like(T), mask, lam, F)
        if proposal is None:
            warm = (lam, F)
        else:
            warm = (np.append(lam, proposal[0]), np.column_stack([F, proposal[1]]))
    if chosen is None:
        chosen = (prev[0], prev[1], prev[2], path[-1]["rank"])
    lam, F, info, R = chosen
    info = dict(info)
    info["rank"] = R
    info["path"] = path
    return lam, F, info


@dataclass
class CPConfig:
    """Configuration for `cp_dem_estimation`.

    Attributes:
        detectors: subset of detectors to analyse (default all).
        init: "entries" (default) starts the CP fit from `structured_init`
            (NNLS on supports of significant entries); "random" uses random
            restarts from rank 1.
        rank_max: largest rank tried during rank selection (default 3 m, and
            never below the structured-init rank).
        coupled: also fit the order-2 tensor with shared factors when
            order >= 3 (better signal-to-noise; the order-3 tensor keeps the
            fit identifiable).
        chi2_tolerance, growth_alpha: rank-selection rules, see `select_rank`.
        n_restarts, seed, max_iter, tol: passed to `symmetric_cp`.
        round_threshold: factor rounding threshold.
        confidence: significance level used for candidate supports and for
            dropping components whose refit weight is not significant
            (one-sided z-test, Bonferroni over the candidates, standard
            errors inflated by the reduced chi^2 of the dictionary refit).
        weight_min: absolute floor on refit weights.
        se_floor: floor for entry standard errors before whitening
            (default 1 / n_shots, the error of a single count).
        prune_refit: after the exact refit (`fit_specified_dem`), drop events
            whose refit probability is not significant (one-sided z-test on
            the delta-method covariance, Bonferroni over the masks) and refit
            once more.  This removes the spurious low-weight events that the
            leading-order CP model produces from O(p^2) structure (typically
            weight-1 events on busy detectors).
        screen: screened mode: only order-3 entries whose detector pairs all
            have a significant covariance are fitted (others get zero
            weight); the order-2 tensor is always coupled in this mode.
    """

    detectors: Optional[Sequence[int]] = None
    init: str = "entries"
    rank_max: Optional[int] = None
    coupled: bool = True
    chi2_tolerance: float = 3.0
    growth_alpha: float = 0.05
    n_restarts: int = 3
    seed: Optional[int] = 0
    max_iter: int = 300
    tol: float = 1e-10
    round_threshold: float = 0.5
    confidence: float = 0.95
    weight_min: float = 1e-5
    se_floor: Optional[float] = None
    prune_refit: bool = True
    screen: bool = False


def _floor_stderr(se: np.ndarray, floor: float) -> np.ndarray:
    return np.maximum(np.asarray(se, dtype=float), floor)


def covariance_screen_mask(cov: np.ndarray, cov_stderr: np.ndarray, order: int,
                           confidence: float = 0.95) -> tuple:
    """
    Screening mask for higher-order entries from significant covariances.

    A detector pair (a, b) is significant if cov[a, b] / stderr[a, b] exceeds
    the one-sided Bonferroni z threshold over all pairs (shared events make
    covariances positive).  An order-k entry is kept iff every pair of
    distinct indices in it is significant.

    Returns:
        pair_mask: (m, m) boolean matrix (diagonal True).
        entry_mask: boolean array of shape (m,) * order.
    """
    m = cov.shape[0]
    n_pairs = max(m * (m - 1) // 2, 1)
    z_thr = scipy.stats.norm.isf((1.0 - confidence) / n_pairs)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(cov_stderr > 0, cov / np.where(cov_stderr > 0, cov_stderr, 1.0), 0.0)
    pair_mask = z > z_thr
    np.fill_diagonal(pair_mask, True)
    pair_mask = pair_mask | pair_mask.T
    entry_mask = np.ones((m,) * order, dtype=bool)
    for i, j in itertools.combinations(range(order), 2):
        shape = [1] * order
        shape[i] = m
        shape[j] = m
        entry_mask = entry_mask & pair_mask.reshape(shape)
    return pair_mask, entry_mask


def cp_dem_estimation(syndrome_counts: dict, *, order: int = 3, rank: Optional[int] = None,
                      config: Optional[CPConfig] = None, return_info: bool = False):
    """
    Estimate a DEM by symmetric CP decomposition of the joint cumulant tensor.

    Parameters:
        syndrome_counts: dict
            Mapping bitstrings (sparsedem convention) to counts.
        order: int
            Cumulant order used for support discovery (3 recommended; 2 is
            the non-unique covariance/"p_ij" setting; 4 is available but slow).
        rank: Optional[int]
            Fixed CP rank; None selects the rank automatically (`select_rank`).
        config: Optional[CPConfig]
        return_info: bool
            Also return a dict with the cumulant tensors, CP weights and
            factors, residual statistics, and the recovered masks.

    Returns:
        stim.DetectorErrorModel, or (dem, info) if `return_info`.
    """
    cfg = config or CPConfig()
    samples, counts = counts_to_detector_arrays(syndrome_counts)
    counts = counts.astype(float)
    n_bits = samples.shape[1]
    n_shots = float(counts.sum())
    dets = _select_detectors(n_bits, cfg.detectors)
    m = len(dets)

    ct = cumulant_tensors(samples, order=order, detectors=dets, weights=counts)
    se_floor = cfg.se_floor if cfg.se_floor is not None else 1.0 / n_shots
    T = ct.tensors[order]
    SE = _floor_stderr(ct.stderrs[order], se_floor)

    mask = None
    pair_mask = None
    use_coupled = cfg.coupled or cfg.screen
    if cfg.screen and order > 2:
        pair_mask, mask = covariance_screen_mask(
            ct.tensors[2], _floor_stderr(ct.stderrs[2], se_floor), order, cfg.confidence)
    coupled = None
    if use_coupled and order > 2:
        coupled = [(ct.tensors[2], _floor_stderr(ct.stderrs[2], se_floor))]
    fit_tensors = [(T, SE, mask)] + (list(coupled) if coupled else [])

    init = None
    init_supports = []
    if cfg.init == "entries":
        w0, F0, init_supports = structured_init(fit_tensors, cfg.confidence, cfg.weight_min)
        if len(w0):
            init = (w0, F0)
    elif cfg.init != "random":
        raise ValueError("config.init must be 'entries' or 'random'.")

    rank_max = cfg.rank_max if cfg.rank_max is not None else 3 * m
    if rank is None:
        lam, F, cp_info = select_rank(T, SE, coupled=coupled, mask=mask, init=init,
                                      rank_max=max(1, rank_max),
                                      chi2_tolerance=cfg.chi2_tolerance,
                                      growth_alpha=cfg.growth_alpha,
                                      n_restarts=cfg.n_restarts, seed=cfg.seed,
                                      max_iter=cfg.max_iter, tol=cfg.tol)
    else:
        lam, F, cp_info = symmetric_cp(T, rank, stderr=SE, mask=mask, coupled=coupled,
                                       n_restarts=cfg.n_restarts, seed=cfg.seed,
                                       max_iter=cfg.max_iter, tol=cfg.tol, init=init)
        cp_info = dict(cp_info)
        cp_info["rank"] = int(rank)

    # Round, deduplicate, and refit weights on the binary dictionary.
    B, _ = _dedupe_binary_columns(round_factors(F, cfg.round_threshold), lam)
    w_dict, se_dict, chi2_dict = refit_dictionary_weights(fit_tensors, B)
    n_unique = sum(int(np.count_nonzero(_unique_entry_mask(m, t[0].ndim)
                                        & (_prepare_weights(t[0], t[1], t[2] if len(t) > 2 else None) > 0)))
                   for t in fit_tensors)
    dof = max(n_unique - B.shape[1], 1)
    inflation = max(1.0, math.sqrt(chi2_dict / dof))
    n_cand = max(B.shape[1], 1)
    z_thr = scipy.stats.norm.isf((1.0 - cfg.confidence) / n_cand)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(se_dict > 0, w_dict / np.where(se_dict > 0, se_dict * inflation, 1.0), np.inf)
    keep = (w_dict > cfg.weight_min) & (z > z_thr)
    masks, cp_weights = factors_to_masks(B[:, keep], w_dict[keep], detectors=dets, threshold=0.5,
                                         weight_min=cfg.weight_min)

    info = {
        "detectors": dets,
        "n_shots": n_shots,
        "order": order,
        "cumulants": ct,
        "init_supports": init_supports,
        "rank": cp_info["rank"],
        "cp_weights": lam,
        "cp_factors": F,
        "cp_info": cp_info,
        "dictionary_factors": B,
        "dictionary_weights": w_dict,
        "dictionary_stderr": se_dict,
        "dictionary_z": z,
        "dictionary_chi2": chi2_dict,
        "dictionary_reduced_chi2": chi2_dict / dof,
        "masks": masks,
        "mask_weights": cp_weights,
        "pair_mask": pair_mask,
        "entry_mask": mask,
    }

    if not masks:
        warnings.warn("cp_dem_estimation: no significant components found; returning an empty DEM.")
        dem = stim.DetectorErrorModel()
        return (dem, info) if return_info else dem

    pol_masks = None
    if m < n_bits or n_bits > _ALL_PAIRS_POL_LIMIT:
        # Restrict the polarization constraints to the analysed detectors.
        pol = set(int(x) for x in masks)
        pol.update(1 << int(d) for d in dets)
        pol.update((1 << int(a)) | (1 << int(b)) for a, b in itertools.combinations(dets.tolist(), 2))
        pol_masks = sorted(pol)
    if cfg.prune_refit:
        _, probs, cov = fit_specified_dem(syndrome_counts, masks, return_probs=True,
                                          return_covariance=True, pol_masks=pol_masks)
        se_p = np.sqrt(np.maximum(np.diag(cov), 0.0))
        z_refit = np.where(se_p > 0, probs / np.where(se_p > 0, se_p, 1.0), np.inf)
        z_thr_refit = scipy.stats.norm.isf((1.0 - cfg.confidence) / len(masks))
        keep_refit = (probs > cfg.weight_min) & (z_refit > z_thr_refit)
        info["refit_probs_before_pruning"] = probs
        info["refit_z"] = z_refit
        info["masks_dropped_by_refit"] = [mk for mk, k in zip(masks, keep_refit) if not k]
        if not np.any(keep_refit):
            warnings.warn("cp_dem_estimation: all events pruned by the refit z-test; returning an empty DEM.")
            dem = stim.DetectorErrorModel()
            info["masks"] = []
            return (dem, info) if return_info else dem
        masks = [mk for mk, k in zip(masks, keep_refit) if k]
        cp_weights = cp_weights[keep_refit]
        info["masks"] = masks
        info["mask_weights"] = cp_weights
        if pol_masks is not None:
            pol_masks = sorted(set(pol_masks) | set(int(x) for x in masks))
    dem = fit_specified_dem(syndrome_counts, masks, pol_masks=pol_masks)
    info["dem"] = dem
    return (dem, info) if return_info else dem

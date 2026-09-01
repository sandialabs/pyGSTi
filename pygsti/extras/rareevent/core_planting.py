"""Core-planting importance sampling for the failure spectrum f(w).

f(w) = P(fail | W = w) is the failure fraction of weight-w fault sets under
the exact conditional distribution pi_w(E) = prod_{i in E} odds_i / Z_w over
|E| = w (odds_i = q_i / (1 - q_i), Z_w = e_w(odds) the elementary symmetric
polynomial), where q = error_model.probabilities(p_ref). Naive rejection
sampling from pi_w (as in ``failure_spectrum.sample_fixed_weight_failure_fraction``)
wastes almost all draws once f(w) is tiny, because it never deliberately
visits the (rare) failing region.

Core planting fixes this by harvesting a list of small "malignant cores" --
weight-<=w subsets known to cause failure -- and building a mixture proposal
Q(E) that, for each weight-w sample, first plants a core c (chosen with
probability alpha_c) and then fills the remaining w - |c| slots by drawing
from the exact conditional distribution over the complement of c. Because Q
deliberately concentrates mass near known failures, the resulting importance
sampling (IS) estimator

    f_hat(w) = (1/M) * sum_m 1[oracle.fails(E_m)] * pi_w(E_m) / Q(E_m)

has much lower variance than plain rejection sampling once cores cover a
sizeable share of the failure probability mass.

f_hat(w) is unbiased for P(fail AND E contains some harvested core | W = w),
which is at most f(w) (failure is *not* monotone in mechanism sets -- a
weight-w set can contain a malignant core and still not fail if the extra
mechanisms happen to cancel it out at the decoder, so the failure indicator
is always evaluated on the full planted-and-filled set, never assumed). This
makes f_hat(w) a certified lower bound on f(w) for *any* core list, including
an empty or wildly incomplete one (in which case f_hat(w) -> 0). The bound
becomes tight as the core list approaches completeness (covers all minimal
failing configurations of weight <= w).

See ``failure_spectrum.py`` for the shared conditional-sampling primitives
(``tilted_probabilities``, ``poisson_binomial_pmf``) that this module reuses.
"""

from __future__ import annotations

import dataclasses
import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from scipy.special import logsumexp

from .failure_spectrum import poisson_binomial_pmf, tilted_probabilities
from .interfaces import ErrorModel, Estimator, ForwardSimulator
from .weight_points import WeightPoint

ALPHA_MODES = ("mass", "uniform")
PEEL_METHODS = ("single", "subset")


class CountingOracle:
    """`ForwardSimulator` wrapper that counts calls to `fails`, for oracle-cost accounting.

    Every `fails` call is delegated unchanged to the wrapped oracle, so wrapping
    never affects results -- only `calls` is updated. Used by benchmarks (and
    tests) to compare the oracle-call cost of peeling strategies.
    """

    def __init__(self, inner: ForwardSimulator) -> None:
        self.inner = inner
        self.calls = 0

    def fails(self, active: set[int]) -> bool:
        self.calls += 1
        return self.inner.fails(active)


# ---------------------------------------------------------------------------
# Exact conditional fixed-weight sampling (shared by harvesting and filling)
# ---------------------------------------------------------------------------


def sample_fixed_weight_sets(
    probs: np.ndarray,
    weight: int,
    num_samples: int,
    rng: np.random.Generator,
    batch_size: int = 4096,
) -> list[frozenset[int]]:
    """Draw `num_samples` iid sets from the exact conditional P(E | |E| = weight).

    Uses exponential tilting (`tilted_probabilities`) plus rejection on the
    exact weight, as in `failure_spectrum.sample_fixed_weight_failure_fraction`,
    but returns the sampled index sets themselves rather than a failure count.
    """
    if num_samples <= 0:
        return []
    q_t = tilted_probabilities(probs, weight)
    n = len(q_t)
    out: list[frozenset[int]] = []
    while len(out) < num_samples:
        draws = rng.random((batch_size, n)) < q_t
        counts = draws.sum(axis=1)
        for row in np.flatnonzero(counts == weight):
            out.append(frozenset(np.flatnonzero(draws[row]).tolist()))
            if len(out) >= num_samples:
                break
    return out


# ---------------------------------------------------------------------------
# Core harvesting
# ---------------------------------------------------------------------------


def _peel_to_minimal(
    oracle: ForwardSimulator,
    active: set[int] | frozenset[int],
    rng: np.random.Generator,
) -> frozenset[int]:
    """Greedily peel a failing set down to a 1-minimal failing set.

    Repeatedly makes a pass over the current elements in rng-shuffled order,
    permanently dropping any element whose removal leaves the set still
    failing. Passes repeat (with fresh shuffles) until a full pass drops
    nothing, at which point removing any single remaining element un-fails
    the set (1-minimality; not necessarily globally minimal).
    """
    current = set(active)
    if not oracle.fails(current):
        raise ValueError("Cannot peel a set that does not fail.")
    while True:
        order = list(current)
        rng.shuffle(order)
        removed_any = False
        for i in order:
            if i not in current:
                continue
            candidate = current - {i}
            if candidate and oracle.fails(candidate):
                current = candidate
                removed_any = True
        if not removed_any:
            break
    return frozenset(current)


def peel_to_minimal_subset(
    oracle: ForwardSimulator,
    active: set[int] | frozenset[int],
    rng: np.random.Generator,
    *,
    max_rounds: int = 128,
    max_stall_rounds: int = 8,
) -> frozenset[int]:
    """Peel a failing set to a 1-minimal failing set via random-subset removal.

    Implements the random-subset peel of Mullan, Weippert & Brown
    (arXiv:2607.27153, Algorithm 1): each round draws a subset size s and a
    uniform random s-subset S of the current elements, tests E \\ S with a
    single oracle call, and commits the removal iff the reduced set is
    nonempty and still fails. On heavy failing patterns this strips many
    "fluff" mechanisms per oracle call, where single-element peeling pays one
    call per element per pass.

    Subset-size scheme: s is uniform on [1, max(1, |E| // 2)] with |E| the
    *current* set size. This is scale-free -- large draws remove many fluff
    elements per call while the set is heavy, and the cap shrinks with the
    set, degrading gracefully to single-element moves near the core; capping
    at |E| // 2 keeps the per-round acceptance probability practical, since
    removing more than half the set rarely leaves a failing configuration.
    All randomness comes from the caller-supplied `rng`, so results are
    deterministic per seed.

    The subset phase stops after `max_rounds` total rounds, after
    `max_stall_rounds` consecutive uncommitted rounds, or when a single
    element remains. A final single-element polish pass (the `_peel_to_minimal`
    loop) then guarantees the returned set is 1-minimal: removing any single
    remaining element un-fails it.

    Raises ValueError if `active` does not fail.
    """
    current = set(active)
    if not oracle.fails(current):
        raise ValueError("Cannot peel a set that does not fail.")
    stall = 0
    for _ in range(max_rounds):
        if len(current) <= 1 or stall >= max_stall_rounds:
            break
        size = int(rng.integers(1, max(1, len(current) // 2) + 1))
        elems = sorted(current)
        picked = rng.choice(len(elems), size=size, replace=False)
        candidate = current - {elems[int(i)] for i in picked}
        if candidate and oracle.fails(candidate):
            current = candidate
            stall = 0
        else:
            stall += 1
    # `current` still fails (only failing candidates are committed), so this
    # single-element polish never raises; it guarantees 1-minimality.
    return _peel_to_minimal(oracle, current, rng)


def prune_antichain(cores: Sequence[frozenset[int]]) -> list[frozenset[int]]:
    """Deduplicate and drop any core that is a superset of another core.

    Coverage (whether a weight-w set contains some core) is unchanged by
    dropping supersets: if c is kept and c <= c', every set containing c'
    also contains c, so c' contributes no additional coverage.
    """
    uniq = sorted(set(cores), key=len)
    kept: list[frozenset[int]] = []
    for c in uniq:
        if not any(k <= c for k in kept):
            kept.append(c)
    return kept


def harvest_cores(
    oracle: ForwardSimulator,
    probs: np.ndarray,
    *,
    weights: Sequence[int],
    target_failures_per_weight: int,
    max_trials_per_weight: int,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
    peel_method: str = "single",
    subset_max_rounds: int = 128,
    subset_max_stall_rounds: int = 8,
) -> list[frozenset[int]]:
    """Harvest malignant cores by exact conditional sampling at a few weights.

    At each weight in `weights`, draws fixed-weight sets from the exact
    conditional distribution (tilting + rejection) until either
    `target_failures_per_weight` failures are found or `max_trials_per_weight`
    sets have been evaluated. Every failing set is peeled to a 1-minimal
    failing set. The combined list is deduplicated and pruned to an
    inclusion-minimal antichain.

    `peel_method` selects the peeling strategy: "single" (default) is the
    greedy single-element peel (`_peel_to_minimal`; byte-identical results and
    RNG stream to before this option existed), "subset" is the random-subset
    peel (`peel_to_minimal_subset`, cheaper in oracle calls on heavy failing
    sets; `subset_max_rounds` / `subset_max_stall_rounds` are forwarded to it).
    """
    if peel_method not in PEEL_METHODS:
        raise ValueError(f"peel_method must be one of {PEEL_METHODS}; got {peel_method!r}.")
    if rng is None:
        rng = np.random.default_rng(seed)
    q = np.asarray(probs, dtype=np.float64)
    raw_cores: list[frozenset[int]] = []
    for w in weights:
        q_t = tilted_probabilities(q, w)
        n = len(q_t)
        trials = 0
        found = 0
        while trials < max_trials_per_weight and found < target_failures_per_weight:
            batch = min(4096, max_trials_per_weight - trials)
            draws = rng.random((batch, n)) < q_t
            counts = draws.sum(axis=1)
            for row in np.flatnonzero(counts == w):
                active = set(np.flatnonzero(draws[row]).tolist())
                trials += 1
                if oracle.fails(active):
                    found += 1
                    if peel_method == "subset":
                        raw_cores.append(
                            peel_to_minimal_subset(
                                oracle,
                                active,
                                rng,
                                max_rounds=subset_max_rounds,
                                max_stall_rounds=subset_max_stall_rounds,
                            )
                        )
                    else:
                        raw_cores.append(_peel_to_minimal(oracle, active, rng))
                if trials >= max_trials_per_weight or found >= target_failures_per_weight:
                    break
    return prune_antichain(raw_cores)


def load_cores_from_jsonl(
    path: str | Path,
    d: int,
    oracle: ForwardSimulator,
    rng: np.random.Generator,
) -> list[frozenset[int]]:
    """Load and peel cores from `failing_states` records in the anchor_mc.jsonl format.

    Silently returns an empty list if `path` does not exist. Records with a
    different `d` are ignored. States that (under this pipeline's `oracle`)
    do not actually fail are skipped defensively.
    """
    path = Path(path)
    cores: list[frozenset[int]] = []
    if not path.exists():
        return cores
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("d") != d:
                continue
            for state in record.get("failing_states", []):
                active = set(state)
                if not active or not oracle.fails(active):
                    continue
                cores.append(_peel_to_minimal(oracle, active, rng))
    return prune_antichain(cores)


# ---------------------------------------------------------------------------
# Mixture proposal: plant a core, fill the rest from the exact conditional
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class _PreparedCores:
    """Precomputed per-core mixture-proposal terms for a fixed (cores, weight, alpha)."""

    cores: tuple[frozenset[int], ...]
    log_alpha: np.ndarray  # log(alpha_c), aligned with `cores`
    log_z_c: np.ndarray  # log(Z^{(c)}) = log e_{w-|c|}(odds restricted to complement of c)
    log_terms: np.ndarray  # L_c = log(alpha_c) - log(Z^{(c)}) - sum_{i in c} log(odds_i)


def _core_indices(core: frozenset[int]) -> np.ndarray:
    return np.fromiter(core, dtype=np.int64, count=len(core))


def _prepare_core_weights(
    cores: Sequence[frozenset[int]],
    q: np.ndarray,
    weight: int,
    alpha: str,
) -> _PreparedCores:
    if alpha not in ALPHA_MODES:
        raise ValueError(f"alpha must be one of {ALPHA_MODES}; got {alpha!r}.")
    support = q > 0
    log_odds = np.full_like(q, -np.inf)
    log_odds[support] = np.log(q[support]) - np.log1p(-q[support])
    total_log1p = float(np.sum(np.log1p(-q)))

    valid: list[frozenset[int]] = []
    sum_log_odds: list[float] = []
    log_z_c: list[float] = []
    for c in cores:
        if len(c) > weight or not all(support[i] for i in c):
            continue
        remaining = weight - len(c)
        idx = _core_indices(c)
        q_masked = q.copy()
        if idx.size:
            q_masked[idx] = 0.0
        pmf_c = poisson_binomial_pmf(q_masked, max_weight=remaining)
        if remaining >= len(pmf_c) or pmf_c[remaining] <= 0.0:
            continue  # complement of c cannot be filled to this weight
        sum_log1p_c = float(np.sum(np.log1p(-q[idx]))) if idx.size else 0.0
        valid.append(c)
        sum_log_odds.append(float(np.sum(log_odds[idx])) if idx.size else 0.0)
        log_z_c.append(math.log(pmf_c[remaining]) - (total_log1p - sum_log1p_c))

    if not valid:
        empty = np.zeros(0, dtype=np.float64)
        return _PreparedCores(cores=(), log_alpha=empty, log_z_c=empty, log_terms=empty)

    slo_arr = np.asarray(sum_log_odds, dtype=np.float64)
    lzc_arr = np.asarray(log_z_c, dtype=np.float64)

    if alpha == "uniform":
        log_alpha = np.full(len(valid), -math.log(len(valid)))
    else:
        log_unnorm = slo_arr + lzc_arr
        log_alpha = log_unnorm - logsumexp(log_unnorm)

    log_terms = log_alpha - lzc_arr - slo_arr
    return _PreparedCores(cores=tuple(valid), log_alpha=log_alpha, log_z_c=lzc_arr, log_terms=log_terms)


def _log_z_w(q: np.ndarray, weight: int) -> float:
    nonzero = int(np.count_nonzero(q))
    if weight < 0 or weight > nonzero:
        raise ValueError(f"weight must be in [0, {nonzero}] (number of nonzero-probability mechanisms); got {weight}.")
    pmf = poisson_binomial_pmf(q, max_weight=weight)
    if weight >= len(pmf) or pmf[weight] <= 0.0:
        raise ValueError(f"No probability mass at weight {weight} (Z_w = 0).")
    return float(math.log(pmf[weight]) - np.sum(np.log1p(-q)))


def _contained_core_indices(cores: tuple[frozenset[int], ...], active: frozenset[int]) -> list[int]:
    return [i for i, c in enumerate(cores) if c <= active]


def log_mixture_density(
    cores: Sequence[frozenset[int]],
    probs: np.ndarray,
    weight: int,
    active: frozenset[int] | set[int],
    *,
    alpha: str = "mass",
) -> float:
    """log Q(E) for the core-planting mixture proposal, in the stable log-space form.

    Q(E) = sum_{c' subset of E, c' in cores} alpha_{c'} * prod_{i in E \\ c'} odds_i / Z^{(c')}.
    Returns -inf if no core in `cores` (restricted to |c| <= weight and the
    q > 0 support) is a subset of `active`, i.e. Q(E) = 0.
    """
    q = np.asarray(probs, dtype=np.float64)
    active = frozenset(active)
    prepared = _prepare_core_weights(cores, q, weight, alpha)
    contained = _contained_core_indices(prepared.cores, active)
    if not contained:
        return -math.inf
    idx = _core_indices(active)
    sum_log_odds_active = float(np.sum(np.log(q[idx]) - np.log1p(-q[idx])))
    return sum_log_odds_active + float(logsumexp(prepared.log_terms[contained]))


# ---------------------------------------------------------------------------
# The importance-sampling estimator
# ---------------------------------------------------------------------------


def core_planting_estimate_f_w(
    error_model: ErrorModel,
    oracle: ForwardSimulator,
    cores: Sequence[frozenset[int]],
    *,
    weight: int,
    p_ref: float,
    num_samples: int,
    alpha: str = "mass",
    seed: int = 1,
    verbose: bool = False,
) -> WeightPoint:
    """Estimate f(weight) = P(fail | W = weight) by core-planting importance sampling.

    Returns a certified lower bound on f(weight): unbiased for
    P(fail AND E contains some core in `cores` | W = weight) <= f(weight).
    See the module docstring for why containing a core does not imply
    failure (the indicator `oracle.fails` is always evaluated).
    """
    if weight < 0:
        raise ValueError(f"weight must be nonnegative; got {weight}.")
    if num_samples < 1:
        raise ValueError(f"num_samples must be at least 1; got {num_samples}.")

    q = np.asarray(error_model.probabilities(p_ref), dtype=np.float64)
    log_z_w = _log_z_w(q, weight)
    prepared = _prepare_core_weights(cores, q, weight, alpha)

    histogram: dict[int, int] = {}
    for c in prepared.cores:
        histogram[len(c)] = histogram.get(len(c), 0) + 1
    meta_base: dict[str, Any] = {
        "num_cores": len(prepared.cores),
        "core_size_histogram": histogram,
        "alpha": alpha,
        "M": num_samples,
    }

    if not prepared.cores:
        if verbose:
            print(f"w={weight} | no valid cores (|c| <= w inside q>0 support) -> estimate=0")
        return WeightPoint(
            method="core_planting",
            kind="f_w",
            weight=weight,
            estimate=0.0,
            rel_err=0.0,
            exact=False,
            lower_bound=True,
            meta={**meta_base, "ess": 0.0, "fail_fraction": 0.0, "no_valid_cores": True},
        )

    rng = np.random.default_rng(seed)
    n_cores = len(prepared.cores)
    alpha_probs = np.exp(prepared.log_alpha)
    alpha_probs = alpha_probs / alpha_probs.sum()
    choices = rng.choice(n_cores, size=num_samples, p=alpha_probs)
    counts = np.bincount(choices, minlength=n_cores)

    weighted = np.zeros(num_samples, dtype=np.float64)
    num_fail = 0
    pos = 0
    for j in range(n_cores):
        k = int(counts[j])
        if k == 0:
            continue
        core = prepared.cores[j]
        remaining = weight - len(core)
        idx = _core_indices(core)
        q_masked = q.copy()
        if idx.size:
            q_masked[idx] = 0.0
        for fill in sample_fixed_weight_sets(q_masked, remaining, k, rng):
            active = core | fill
            if oracle.fails(set(active)):
                num_fail += 1
                contained = _contained_core_indices(prepared.cores, active)
                log_w = -log_z_w - float(logsumexp(prepared.log_terms[contained]))
                weighted[pos] = math.exp(log_w)
            pos += 1
    assert pos == num_samples

    estimate = float(np.mean(weighted))
    se = float(np.std(weighted, ddof=1) / math.sqrt(num_samples)) if num_samples > 1 else 0.0
    rel_err = se / estimate if estimate > 0 else 0.0
    sum_w = float(np.sum(weighted))
    sum_w2 = float(np.sum(np.square(weighted)))
    ess = (sum_w * sum_w / sum_w2) if sum_w2 > 0 else 0.0
    fail_fraction = num_fail / num_samples

    if verbose:
        print(
            f"w={weight} | estimate={estimate:.6g} | rel_err={rel_err:.3g} | "
            f"ESS={ess:.1f} | fail_fraction={fail_fraction:.4g} | num_cores={n_cores}"
        )

    return WeightPoint(
        method="core_planting",
        kind="f_w",
        weight=weight,
        estimate=estimate,
        rel_err=rel_err,
        exact=False,
        lower_bound=True,
        meta={**meta_base, "ess": ess, "fail_fraction": fail_fraction},
    )


def core_planting_estimate_many(
    error_model: ErrorModel,
    oracle: ForwardSimulator,
    cores: Sequence[frozenset[int]],
    *,
    weights: Sequence[int],
    p_ref: float,
    num_samples: int,
    alpha: str = "mass",
    seed: int = 1,
    verbose: bool = False,
) -> list[WeightPoint]:
    """Run `core_planting_estimate_f_w` over several weights (distinct seeds per weight)."""
    return [
        core_planting_estimate_f_w(
            error_model,
            oracle,
            cores,
            weight=w,
            p_ref=p_ref,
            num_samples=num_samples,
            alpha=alpha,
            seed=seed + i,
            verbose=verbose,
        )
        for i, w in enumerate(weights)
    ]


class CorePlantingEstimator(Estimator):
    """`Estimator`-protocol wrapper around `core_planting_estimate_many`.

    Requires an explicit `cores` list (see `harvest_cores` /
    `load_cores_from_jsonl` to build one) since this estimator only ever
    reports a certified lower bound relative to whatever cores it is given.
    """

    def estimate(
        self,
        error_model: ErrorModel,
        simulator: ForwardSimulator,
        **kwargs: Any,
    ) -> dict[str, Any]:
        weights = kwargs.get("weights")
        if weights is None:
            raise ValueError("weights must be provided to CorePlantingEstimator")
        p_ref = kwargs.get("p_ref")
        if p_ref is None:
            raise ValueError("p_ref must be provided to CorePlantingEstimator")
        cores = kwargs.get("cores")
        if cores is None:
            raise ValueError("cores must be provided to CorePlantingEstimator (see harvest_cores)")

        cores = [frozenset(c) for c in cores]
        points = core_planting_estimate_many(
            error_model,
            simulator,
            cores,
            weights=weights,
            p_ref=p_ref,
            num_samples=kwargs.get("num_samples", 2000),
            alpha=kwargs.get("alpha", "mass"),
            seed=kwargs.get("seed", 1),
            verbose=kwargs.get("verbose", False),
        )
        return {
            "weight_points": points,
            "cores": sorted(tuple(sorted(c)) for c in cores),
        }

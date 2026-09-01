"""
Subregion (partial-resampling) rare-event splitting estimator.

The methods in this module are the package's adaptation of Mullan, Weippert &
Brown, "Improved Methods for Determining Quantum Error Correcting Code
Performance and Fault Tolerance" (arXiv:2607.27153), applied on top of the
locality-aware splitting baseline (`splitting_local`). Three independently
toggleable improvements are provided; the baselines are untouched so their
behavior can always be reproduced for comparison.

1. The subregion proposal (`SubregionConditionalFailureMCMC`)
-------------------------------------------------------------
Both existing chains (`rare_event.ConditionalFailureMCMC`,
`splitting_local.LocalConditionalFailureMCMC`) are single-toggle kernels. At
low p the conditional failure distribution P_p(E | failure) concentrates on
near-minimal malignant sets, and a single-toggle chain has two structural
problems there:

  - *add* moves pass the Metropolis ratio with probability ~ q_i(p), which is
    tiny, so the chain barely ever grows the set, and
  - moving between distinct malignant basins requires passing through
    intermediate states that do not fail, which the conditioning indicator
    rejects, so the chain can only reach basins connected to its start by
    failing single-toggle paths.

The subregion proposal replaces the single toggle with a partial resample.
Each step draws a region R (every mechanism index included independently with
probability ``region_rate``, *independent of the current state*), keeps the
configuration outside R fixed, and redraws every coordinate inside R
independently from a resample distribution f:

    x'_i ~ Bernoulli(f_i)  for i in R,   x'_i = x_i  for i not in R.

Given R, the proposal density is g(E' | E, R) = prod_{i in R}
f_i^{x'_i} (1 - f_i)^{1 - x'_i}, so the Hastings ratio against the target
pi(E) proportional to prod_i q_i^{x_i} (1 - q_i)^{1 - x_i} (restricted to
failing E) is

    A = min(1, [pi(E')/pi(E)] * [g(E | E', R) / g(E' | E, R)])
      = min(1, prod_{i in R : x'_i != x_i} [odds_q(i) / odds_f(i)]^{x'_i - x_i}),

with odds(i) = p_i / (1 - p_i); coordinates with x'_i = x_i cancel. With the
default f = q (resample at the *current level's* probabilities) the product is
identically 1: the proposal is rejection-free with respect to pi, and the
acceptance decision reduces to the conditioning indicator alone -- one call to
``oracle.fails(E')`` per step, never a q-suppressed ratio test. This is the
paper's central observation: resampling part of the pattern at the current
rate is already a valid pi move, so Metropolis rejections are only needed to
enforce the failure conditioning.

The paper's "core resampling" heuristic sets ``region_rate = 1 / w_min``
(`default_region_rate`), with w_min the minimum failing weight (the onset
weight ceil(d/2), see `pipelines.default_onset_weight`). ``region_rate`` is a
*per-mechanism* inclusion probability applied to all n catalog indices, so the
expected region size is n / w_min; the point of the heuristic is that each of
the ~w_min core errors of the current failing set is hit with probability
1 / w_min, i.e. the proposal replaces about one core error per step (while
also redrawing the fluff around it) -- a basin-to-basin jump in a single move.

A step whose proposal changes nothing (R missed the active set and the
resample added nothing -- common at low p) is counted as an accepted no-op
WITHOUT consulting the oracle: E' = E trivially still fails. The fraction of
such no-ops is tracked separately (`KernelCounters.noop_steps`) so acceptance
diagnostics refer to real moves, and oracle calls are counted for cost
accounting.

2. R-hat-driven adaptive level stopping (``stop_rhat``)
-------------------------------------------------------
The baseline spends a fixed step budget at every level. Following the paper's
practice, ``stop_rhat`` instead runs the level's chains in blocks and stops as
soon as the cross-chain split-R-hat of the log weight-ratio series (first half
of each chain discarded as burn-in, both for the diagnostic and for the final
level estimate) drops to the threshold -- or a hard cap is reached. Easy
levels stop early; hard levels get the full cap and are flagged
(`SubregionLevelDiagnostics.rhat_threshold_met`).

3. Bennett acceptance ratio level estimator (``ratio_estimator="bar"``)
-----------------------------------------------------------------------
The baseline estimates each level ratio Z_{k+1} / Z_k one-sidedly, from level-k
samples only. Bennett (1976) combines the level-k ("forward") and level-(k+1)
("reverse") sample sets, which is the minimum-variance choice among a wide
class of two-sided estimators; the descent already produces samples at every
level, so the reverse set is free except at the final p, where one extra
sampling run is performed. With ell(E) = log w_{k+1}(E) - log w_k(E) and
C = log(Z_{k+1} / Z_k), BAR solves the monotone scalar equation

    sum_{E ~ level k} s(ell(E) - C - M) = sum_{E ~ level k+1} s(-(ell(E) - C - M)),

where s is the logistic function and M = log(n_{k+1} / n_k), by bisection.
The one-sided ("forward") estimate is always computed alongside for
comparison (`SubregionLevelDiagnostics.forward_log_ratio`).
"""

from __future__ import annotations

import dataclasses
import math
import sys
from collections.abc import Sequence
from typing import Any

import numpy as np

from .interfaces import ErrorModel, Estimator, ForwardSimulator
from .rare_event import (
    LevelDiagnostics,
    SplittingResult,
    log_weight_ratio,
    logmeanexp,
    split_rhat,
    summarize_chains_for_level,
)

RATIO_ESTIMATORS = ("forward", "bar")


def default_region_rate(min_core_weight: int) -> float:
    """The paper's core-resampling heuristic: region_rate = 1 / w_min.

    ``min_core_weight`` is the minimum failing fault weight (the onset weight
    ceil(d/2) for a distance-d code; see `pipelines.default_onset_weight`).
    Each of the ~w_min core errors of the current failing set is then included
    in the resampled region with probability 1 / w_min, so the proposal
    replaces about one core error per step on average.
    """
    if min_core_weight < 1:
        raise ValueError(f"min_core_weight must be at least 1; got {min_core_weight}.")
    return 1.0 / float(min_core_weight)


@dataclasses.dataclass
class KernelCounters:
    """Cost/behavior counters for one `SubregionConditionalFailureMCMC` chain."""

    steps: int = 0
    noop_steps: int = 0  # proposal changed nothing: accepted without an oracle call
    proposals: int = 0  # steps whose proposal toggled at least one mechanism
    accepted: int = 0  # proposals accepted (ratio test passed and E' still fails)
    oracle_calls: int = 0

    @property
    def acceptance_rate(self) -> float:
        return self.accepted / max(self.proposals, 1)

    @property
    def noop_fraction(self) -> float:
        return self.noop_steps / max(self.steps, 1)


class SubregionConditionalFailureMCMC:
    """Metropolis-Hastings chain targeting P_p(E | failure) with subregion proposals.

    See the module docstring for the proposal, the exact-cancellation argument
    that makes the default ``resample_probs=None`` (f = q) path rejection-free
    up to the conditioning indicator, and the region-rate heuristic. With an
    explicit ``resample_probs`` (f != q) the per-step Hastings ratio
    prod_{toggled i} [odds_q(i)/odds_f(i)]^{x'_i - x_i} is evaluated *before*
    the oracle is consulted ("ratio first, decode second", as in the
    baselines).
    """

    def __init__(
        self,
        oracle: ForwardSimulator,
        probabilities: np.ndarray,
        region_rate: float,
        resample_probs: np.ndarray | None = None,
        rng: np.random.Generator | None = None,
    ):
        self.oracle = oracle
        self.q = np.asarray(probabilities, dtype=np.float64)
        if np.any(self.q <= 0) or np.any(self.q >= 1):
            raise ValueError("All mechanism probabilities must be in (0, 1).")
        if not (0.0 < region_rate <= 1.0):
            raise ValueError(f"region_rate must be in (0, 1]; got {region_rate}.")
        self.region_rate = float(region_rate)
        if resample_probs is None:
            self.f = self.q
            self._log_odds_gap: np.ndarray | None = None
        else:
            self.f = np.asarray(resample_probs, dtype=np.float64)
            if self.f.shape != self.q.shape:
                raise ValueError("resample_probs must have one entry per mechanism.")
            if np.any(self.f <= 0) or np.any(self.f >= 1):
                raise ValueError("All resample probabilities must be in (0, 1).")
            # log[odds_q(i) / odds_f(i)]; the Hastings log-ratio of a proposal is
            # sum over toggled i of (x'_i - x_i) * _log_odds_gap[i].
            self._log_odds_gap = (np.log(self.q) - np.log1p(-self.q)) - (np.log(self.f) - np.log1p(-self.f))
        self.rng = rng if rng is not None else np.random.default_rng()
        self.counters = KernelCounters()
        self._mask = np.zeros(len(self.q), dtype=bool)
        self.active: set[int] = set()

    def set_state(self, active: set[int]) -> None:
        self.active = set(active)
        self._mask[:] = False
        if self.active:
            self._mask[list(self.active)] = True

    def step(self, active: set[int]) -> tuple[set[int], bool]:
        """One subregion step. Returns (new_active, accepted).

        ``accepted`` is True only for proposals that toggled at least one
        mechanism and were accepted; no-op steps return the unchanged state
        with ``accepted=False`` (they are tracked in ``counters.noop_steps``).
        """
        if active != self.active:
            self.set_state(active)
        accepted = self._step_once()
        return set(self.active), accepted

    def _step_once(self) -> bool:
        n = len(self.q)
        self.counters.steps += 1
        region = self.rng.random(n) < self.region_rate
        proposed = self._mask.copy()
        proposed[region] = self.rng.random(int(np.count_nonzero(region))) < self.f[region]
        toggled = np.flatnonzero(proposed != self._mask)
        if toggled.size == 0:
            self.counters.noop_steps += 1
            return False

        self.counters.proposals += 1
        if self._log_odds_gap is not None:
            signs = np.where(proposed[toggled], 1.0, -1.0)
            log_ratio = float(np.sum(signs * self._log_odds_gap[toggled]))
            if log_ratio < 0 and self.rng.random() >= math.exp(log_ratio):
                return False

        candidate = set(np.flatnonzero(proposed).tolist())
        self.counters.oracle_calls += 1
        if not self.oracle.fails(candidate):
            return False

        self._mask = proposed
        self.active = candidate
        self.counters.accepted += 1
        return True

    def seed_from_monte_carlo(self, max_tries: int = 1_000_000) -> set[int]:
        """Find an initial failing state by direct sampling at the current q."""
        n = len(self.q)
        for _ in range(max_tries):
            draws = self.rng.random(n) < self.q
            active = set(np.flatnonzero(draws).tolist())
            self.counters.oracle_calls += 1
            if self.oracle.fails(active):
                return active
        raise RuntimeError(
            "Could not find an initial failing state. Increase p0, max_tries, or seed manually."
        )

    def sample(
        self,
        initial: set[int],
        steps: int,
        burn_in: int = 0,
        thin: int = 1,
    ) -> tuple[list[set[int]], float]:
        """Run ``steps`` steps, returning post-burn-in thinned states and the acceptance rate.

        The acceptance rate refers to real proposals only (no-op steps are
        excluded from both numerator and denominator; see ``counters``).
        """
        if not self.oracle.fails(initial):
            raise ValueError("Initial state is not a logical failure.")
        self.set_state(initial)
        samples: list[set[int]] = []
        start = self.counters.proposals
        start_acc = self.counters.accepted
        for t in range(steps):
            self._step_once()
            if t >= burn_in and ((t - burn_in) % thin == 0):
                samples.append(set(self.active))
        proposals = self.counters.proposals - start
        return samples, (self.counters.accepted - start_acc) / max(proposals, 1)


@dataclasses.dataclass
class SubregionLevelDiagnostics(LevelDiagnostics):
    """`LevelDiagnostics` plus the subregion estimator's per-level extras."""

    steps_per_chain_used: int = 0
    rhat_threshold_met: bool | None = None  # None when stop_rhat was not used
    oracle_calls: int = 0
    noop_fraction: float = 0.0
    forward_log_ratio: float = 0.0
    bar_log_ratio: float | None = None
    bar_reverse_sample_size: int | None = None


def bennett_log_ratio(
    forward_deltas: Sequence[float],
    reverse_deltas: Sequence[float],
    tol: float = 1e-10,
    max_iter: int = 500,
) -> float:
    """Solve the Bennett acceptance-ratio equation for C = log(Z_next / Z_current).

    ``forward_deltas`` are ell(E) = log w_next(E) - log w_current(E) evaluated on
    samples E ~ pi_current; ``reverse_deltas`` are the same quantity evaluated on
    samples E ~ pi_next. Viewing the pooled samples through logistic regression,
    the probability that a state E came from the reverse ensemble is
    s(ell(E) - C + log(n_R / n_F)), and the maximum-likelihood C matches the
    expected and observed ensemble counts. That is the BAR root h(C) = 0 with

        h(C) = sum_F s(ell - C - M) - sum_R s(-(ell - C - M)),   M = log(n_F / n_R),

    (s = logistic; note M carries n_F over n_R -- the shift is *added* to ell - C
    as log(n_R / n_F)). h is strictly decreasing in C, so bisection on an
    expanding bracket around the one-sided estimates converges unconditionally.
    """
    fwd = np.asarray(forward_deltas, dtype=np.float64)
    rev = np.asarray(reverse_deltas, dtype=np.float64)
    if fwd.size == 0 or rev.size == 0:
        raise ValueError("BAR requires at least one sample on each side.")
    m_shift = math.log(fwd.size / rev.size)

    def h(c: float) -> float:
        # logistic(x) = 1/(1+exp(-x)), computed stably via np.
        x_f = fwd - c - m_shift
        x_r = rev - c - m_shift
        s_f = 0.5 * (1.0 + np.tanh(0.5 * x_f))
        s_r = 0.5 * (1.0 + np.tanh(-0.5 * x_r))
        return float(np.sum(s_f) - np.sum(s_r))

    # Bracket from the two one-sided estimates, expanded until h changes sign.
    c_fwd = logmeanexp(list(fwd))[0]
    c_rev = -logmeanexp(list(-rev))[0]
    lo, hi = min(c_fwd, c_rev), max(c_fwd, c_rev)
    span = max(hi - lo, 1.0)
    for _ in range(200):
        if h(lo) > 0 and h(hi) < 0:
            break
        if h(lo) <= 0:
            lo -= span
        if h(hi) >= 0:
            hi += span
        span *= 2.0
    else:  # pragma: no cover - h is monotone with limits +n_F / -n_R
        raise RuntimeError("Failed to bracket the BAR root.")

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        if hi - lo < tol:
            return mid
        if h(mid) > 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


@dataclasses.dataclass
class _LevelRun:
    """Samples and bookkeeping from running one level's chains at one p."""

    per_chain_samples: list[list[set[int]]]
    per_chain_acceptance: list[float]
    final_states: list[set[int]]
    steps_per_chain_used: int
    rhat_threshold_met: bool | None
    oracle_calls: int
    noop_fraction: float


def _run_level(
    chains: list[SubregionConditionalFailureMCMC],
    initials: list[set[int]],
    monitor_probs: tuple[np.ndarray, np.ndarray],
    *,
    fixed_steps: int | None,
    fixed_burn_in: int,
    thin: int,
    stop_rhat: float | None,
    block_steps: int,
    min_steps_per_chain: int,
    max_steps_per_chain: int,
) -> _LevelRun:
    """Run all chains of one level, with a fixed budget or R-hat-adaptive stopping.

    In adaptive mode chains run in blocks of ``block_steps``; after each block
    (once ``min_steps_per_chain`` is reached) the split-R-hat of the per-chain
    log weight-ratio series -- first half of each chain discarded -- is
    compared against ``stop_rhat``. The *returned samples* in adaptive mode are
    likewise the second half of each chain, so the level estimate uses exactly
    the series the diagnostic certified.
    """
    probs_next, probs_current = monitor_probs
    oracle_calls_start = sum(c.counters.oracle_calls for c in chains)

    if stop_rhat is None:
        assert fixed_steps is not None
        per_chain_samples: list[list[set[int]]] = []
        per_chain_acc: list[float] = []
        for chain, initial in zip(chains, initials):
            samples, acc = chain.sample(initial, steps=fixed_steps, burn_in=fixed_burn_in, thin=thin)
            per_chain_samples.append(samples)
            per_chain_acc.append(acc)
        steps_used = fixed_steps
        threshold_met: bool | None = None
    else:
        for chain, initial in zip(chains, initials):
            if not chain.oracle.fails(initial):
                raise ValueError("Initial state is not a logical failure.")
            chain.set_state(initial)
        collected: list[list[set[int]]] = [[] for _ in chains]
        acc_before = [(c.counters.proposals, c.counters.accepted) for c in chains]
        steps_used = 0
        threshold_met = False
        while steps_used < max_steps_per_chain:
            block = min(block_steps, max_steps_per_chain - steps_used)
            for chain, states in zip(chains, collected):
                for t in range(block):
                    chain._step_once()
                    if (steps_used + t) % thin == 0:
                        states.append(set(chain.active))
            steps_used += block
            if steps_used < min_steps_per_chain:
                continue
            halves = [s[len(s) // 2 :] for s in collected]
            series = [
                [log_weight_ratio(state, probs_next, probs_current) for state in half] for half in halves
            ]
            rhat = split_rhat(series)
            if rhat is not None and rhat <= stop_rhat:
                threshold_met = True
                break
        per_chain_samples = [s[len(s) // 2 :] for s in collected]
        per_chain_acc = []
        for chain, (p0, a0) in zip(chains, acc_before):
            proposals = chain.counters.proposals - p0
            per_chain_acc.append((chain.counters.accepted - a0) / max(proposals, 1))

    steps_total = steps_used * len(chains)
    noops = sum(c.counters.noop_steps for c in chains)
    return _LevelRun(
        per_chain_samples=per_chain_samples,
        per_chain_acceptance=per_chain_acc,
        final_states=[set(c.active) for c in chains],
        steps_per_chain_used=steps_used,
        rhat_threshold_met=threshold_met,
        oracle_calls=sum(c.counters.oracle_calls for c in chains) - oracle_calls_start,
        noop_fraction=noops / max(steps_total, 1),
    )


def subregion_splitting_estimate(
    error_model: ErrorModel,
    simulator: ForwardSimulator,
    p_scales: Sequence[float],
    region_rate: float,
    mc_shots_at_p0: int = 10_000,
    steps_per_chain: int | None = None,
    total_steps_per_level: int | None = None,
    burn_in: int | None = None,
    burn_in_fraction: float | None = 0.1,
    thin: int = 1,
    seed: int = 1,
    resample_probs: np.ndarray | None = None,
    anchor_failure_rate: float | None = None,
    anchor_state: set[int] | None = None,
    seed_states: Sequence[set[int]] | None = None,
    num_chains: int = 1,
    stop_rhat: float | None = None,
    block_steps: int = 2_000,
    min_steps_per_chain: int = 4_000,
    max_steps_per_chain: int = 200_000,
    ratio_estimator: str = "forward",
) -> SplittingResult:
    """Estimate failure rates along a descending p_scales schedule with subregion MCMC.

    Same anchor-then-descend flow, multi-chain seeding, and diagnostics as
    `splitting_local.local_splitting_estimate`, with the subregion kernel and
    two further opt-in improvements (see the module docstring):

    - ``stop_rhat``: adaptive per-level stopping (requires ``num_chains >= 2``).
      Levels run in blocks of ``block_steps`` per chain until the cross-chain
      split-R-hat of the log weight-ratio (second halves of the chains) drops
      to ``stop_rhat``, bounded by ``min_steps_per_chain`` /
      ``max_steps_per_chain``. ``steps_per_chain`` / ``total_steps_per_level``
      must not be given in this mode. Fixed mode (``stop_rhat=None``) uses the
      baseline's budget semantics exactly.
    - ``ratio_estimator``: "forward" (baseline one-sided estimator) or "bar"
      (Bennett acceptance ratio over both adjacent levels' samples; runs one
      extra chain set at the final p to provide the last reverse sample set).

    ``region_rate`` follows the core-resampling heuristic
    `default_region_rate(onset_weight)` unless you have a better estimate of
    the minimum failing weight. ``resample_probs`` overrides the resample
    distribution f (default: the current level's probabilities, the
    rejection-free choice).
    """
    if thin < 1:
        raise ValueError(f"thin must be at least 1; got {thin}.")
    if num_chains < 1:
        raise ValueError(f"num_chains must be at least 1; got {num_chains}.")
    if ratio_estimator not in RATIO_ESTIMATORS:
        raise ValueError(f"ratio_estimator must be one of {RATIO_ESTIMATORS}; got {ratio_estimator!r}.")
    if len(p_scales) < 2:
        raise ValueError("p_scales must contain at least two rates (anchor plus one target).")

    if stop_rhat is not None:
        if num_chains < 2:
            raise ValueError("stop_rhat requires num_chains >= 2 (the diagnostic is cross-chain).")
        if stop_rhat <= 1.0:
            raise ValueError(f"stop_rhat must exceed 1; got {stop_rhat}.")
        if steps_per_chain is not None or total_steps_per_level is not None:
            raise ValueError("With stop_rhat, do not pass steps_per_chain/total_steps_per_level; "
                             "use block_steps/min_steps_per_chain/max_steps_per_chain.")
        if not (0 < block_steps <= max_steps_per_chain):
            raise ValueError(f"block_steps must be in (0, max_steps_per_chain]; got {block_steps}.")
        if min_steps_per_chain > max_steps_per_chain:
            raise ValueError("min_steps_per_chain must not exceed max_steps_per_chain.")
        per_chain_steps: int | None = None
        per_chain_burn_in = 0
    else:
        if steps_per_chain is None and total_steps_per_level is None:
            raise ValueError("Specify either steps_per_chain or total_steps_per_level.")
        if steps_per_chain is not None and total_steps_per_level is not None:
            raise ValueError("Specify only one of steps_per_chain or total_steps_per_level.")
        per_chain_steps = int(steps_per_chain) if steps_per_chain is not None else int(total_steps_per_level or 0) // num_chains
        if per_chain_steps < 1:
            raise ValueError(f"MCMC proposal steps must be at least 1; got {per_chain_steps}.")
        if burn_in is not None and burn_in < 0:
            raise ValueError(f"burn_in must be nonnegative; got {burn_in}.")
        if burn_in is None:
            if burn_in_fraction is None:
                burn_in_fraction = 0.1
            if not (0 <= burn_in_fraction < 1):
                raise ValueError(f"burn_in_fraction must be in [0, 1); got {burn_in_fraction}.")
            per_chain_burn_in = int(per_chain_steps * burn_in_fraction)
        else:
            per_chain_burn_in = burn_in
        if per_chain_steps <= per_chain_burn_in:
            raise ValueError(
                f"MCMC proposal steps must exceed burn-in; got steps={per_chain_steps}, "
                f"burn_in={per_chain_burn_in}. Use a larger step count or smaller burn-in fraction."
            )

    master_rng = np.random.default_rng(seed)
    probs = [np.asarray(error_model.probabilities(p), dtype=np.float64) for p in p_scales]

    # Anchor at p0 (identical policy to the baseline estimators).
    initial_failure: set[int]
    if anchor_failure_rate is not None:
        if anchor_failure_rate <= 0:
            raise ValueError(f"anchor_failure_rate must be positive; got {anchor_failure_rate}.")
        if anchor_state is None:
            raise ValueError("anchor_state (a failing configuration at p0) is required with anchor_failure_rate.")
        if not simulator.fails(set(anchor_state)):
            raise ValueError("anchor_state is not a logical failure under the simulator.")
        p_fail0 = float(anchor_failure_rate)
        initial_failure = set(anchor_state)
    else:
        if anchor_state is not None:
            raise ValueError("anchor_state requires anchor_failure_rate.")
        failures = 0
        found: set[int] | None = None
        n = len(probs[0])
        for _ in range(mc_shots_at_p0):
            active = set(np.flatnonzero(master_rng.random(n) < probs[0]).tolist())
            if simulator.fails(active):
                failures += 1
                if found is None:
                    found = set(active)
        if failures == 0 or found is None:
            raise RuntimeError("No failures observed at p0. Choose a larger p0 or increase mc_shots_at_p0.")
        p_fail0 = failures / mc_shots_at_p0
        initial_failure = found

    validated_seeds: list[set[int]] = []
    if seed_states is not None:
        for idx, s in enumerate(seed_states):
            s_set = set(s)
            if not simulator.fails(s_set):
                raise ValueError(f"seed_states[{idx}] is not a logical failure under the simulator.")
            validated_seeds.append(s_set)

    chain_initials: list[set[int]] = [set(initial_failure)]
    for c in range(1, num_chains):
        if validated_seeds:
            chain_initials.append(set(validated_seeds[(c - 1) % len(validated_seeds)]))
        else:
            chain_initials.append(set(initial_failure))

    num_levels = len(p_scales) - 1
    # BAR needs a reverse sample set at every ratio's upper level, including the
    # final p; the forward path only ever samples at levels 0 .. L-2.
    run_indices = list(range(num_levels + 1)) if ratio_estimator == "bar" else list(range(num_levels))

    runs: list[_LevelRun] = []
    for j in run_indices:
        chains = [
            SubregionConditionalFailureMCMC(
                simulator,
                probs[j],
                region_rate=region_rate,
                resample_probs=resample_probs,
                rng=np.random.default_rng(int(master_rng.integers(2**63))),
            )
            for _ in range(num_chains)
        ]
        for c, chain in enumerate(chains):
            if not simulator.fails(chain_initials[c]):
                chain_initials[c] = chain.seed_from_monte_carlo()
        # R-hat monitor: the log weight-ratio series actually used by the level
        # estimate (toward p_{j+1}); the final BAR-only run monitors the ratio
        # it participates in (toward p_{j-1} -- same series up to sign).
        monitor = (probs[j + 1], probs[j]) if j < num_levels else (probs[j], probs[j - 1])
        run = _run_level(
            chains,
            [chain_initials[c] for c in range(num_chains)],
            monitor,
            fixed_steps=per_chain_steps,
            fixed_burn_in=per_chain_burn_in,
            thin=thin,
            stop_rhat=stop_rhat,
            block_steps=block_steps,
            min_steps_per_chain=min_steps_per_chain,
            max_steps_per_chain=max_steps_per_chain,
        )
        runs.append(run)
        # Warm-start the next level from this level's final states (still failing).
        chain_initials = [set(s) for s in run.final_states]

    log_fail = math.log(p_fail0)
    log_failure_estimates = [log_fail]
    log_ratio_estimates: list[float] = []
    acceptance_rates: list[float] = []
    sample_sizes: list[int] = []
    level_diagnostics: list[LevelDiagnostics] = []

    for k in range(num_levels):
        run = runs[k]
        pooled = [state for chain_samples in run.per_chain_samples for state in chain_samples]
        forward_deltas = [log_weight_ratio(state, probs[k + 1], probs[k]) for state in pooled]
        forward_log_ratio, _ = logmeanexp(forward_deltas)

        bar_log_ratio: float | None = None
        bar_reverse_size: int | None = None
        if ratio_estimator == "bar":
            reverse_pool = [
                state for chain_samples in runs[k + 1].per_chain_samples for state in chain_samples
            ]
            reverse_deltas = [log_weight_ratio(state, probs[k + 1], probs[k]) for state in reverse_pool]
            bar_log_ratio = bennett_log_ratio(forward_deltas, reverse_deltas)
            bar_reverse_size = len(reverse_pool)
            log_ratio = bar_log_ratio
        else:
            log_ratio = forward_log_ratio

        log_fail += log_ratio
        acc = float(np.mean(run.per_chain_acceptance))
        log_ratio_estimates.append(log_ratio)
        log_failure_estimates.append(log_fail)
        acceptance_rates.append(acc)
        sample_sizes.append(len(pooled))

        base = summarize_chains_for_level(
            level=k,
            p_current=p_scales[k],
            p_next=p_scales[k + 1],
            samples_by_chain=run.per_chain_samples,
            acceptance_rates=run.per_chain_acceptance,
            probs_next=probs[k + 1],
            probs_current=probs[k],
            pooled_log_ratio=log_ratio,
        )
        diag = SubregionLevelDiagnostics(
            **dataclasses.asdict(base),
            steps_per_chain_used=run.steps_per_chain_used,
            rhat_threshold_met=run.rhat_threshold_met,
            oracle_calls=run.oracle_calls,
            noop_fraction=run.noop_fraction,
            forward_log_ratio=forward_log_ratio,
            bar_log_ratio=bar_log_ratio,
            bar_reverse_sample_size=bar_reverse_size,
        )
        level_diagnostics.append(diag)

        rhat_lr = "n/a" if diag.rhat_log_weight_ratio is None else f"{diag.rhat_log_weight_ratio:.3f}"
        bar_note = "" if bar_log_ratio is None else f" | bar_log_ratio={bar_log_ratio:.6g}"
        stop_note = "" if run.rhat_threshold_met is None else f" | rhat_met={run.rhat_threshold_met}"
        print(
            f"Level {k} -> {k + 1} | p={p_scales[k]:.6g} -> {p_scales[k + 1]:.6g} | "
            f"log_ratio={log_ratio:.6g} | log_fail={log_fail:.6g} | P_fail={math.exp(log_fail):.6e} | "
            f"acc={acc:.3f} | noop={run.noop_fraction:.3f} | chains={num_chains} | "
            f"steps/chain={run.steps_per_chain_used} | oracle_calls={run.oracle_calls} | "
            f"samples={len(pooled)} | Rhat_log_weight_ratio={rhat_lr}{bar_note}{stop_note}"
        )
        sys.stdout.flush()

    return SplittingResult(
        p_scales=list(p_scales),
        log_failure_estimates=log_failure_estimates,
        failure_estimates=[math.exp(x) for x in log_failure_estimates],
        log_ratio_estimates=log_ratio_estimates,
        acceptance_rates=acceptance_rates,
        sample_sizes=sample_sizes,
        level_diagnostics=level_diagnostics,
    )


class SubregionSplittingEstimator(Estimator):
    """Estimator implementing subregion-proposal rare-event splitting (arXiv:2607.27153)."""

    def estimate(
        self,
        error_model: ErrorModel,
        simulator: ForwardSimulator,
        **kwargs: Any,
    ) -> SplittingResult:
        for key in ("p_scales", "region_rate"):
            if key not in kwargs:
                raise ValueError(f"{key} must be provided to SubregionSplittingEstimator")

        return subregion_splitting_estimate(
            error_model=error_model,
            simulator=simulator,
            p_scales=kwargs["p_scales"],
            region_rate=kwargs["region_rate"],
            mc_shots_at_p0=kwargs.get("mc_shots_at_p0", 10_000),
            steps_per_chain=kwargs.get("steps_per_chain"),
            total_steps_per_level=kwargs.get("total_steps_per_level"),
            burn_in=kwargs.get("burn_in"),
            burn_in_fraction=kwargs.get("burn_in_fraction", 0.1),
            thin=kwargs.get("thin", 1),
            seed=kwargs.get("seed", 1),
            resample_probs=kwargs.get("resample_probs"),
            anchor_failure_rate=kwargs.get("anchor_failure_rate"),
            anchor_state=kwargs.get("anchor_state"),
            seed_states=kwargs.get("seed_states"),
            num_chains=kwargs.get("num_chains", 1),
            stop_rhat=kwargs.get("stop_rhat"),
            block_steps=kwargs.get("block_steps", 2_000),
            min_steps_per_chain=kwargs.get("min_steps_per_chain", 4_000),
            max_steps_per_chain=kwargs.get("max_steps_per_chain", 200_000),
            ratio_estimator=kwargs.get("ratio_estimator", "forward"),
        )

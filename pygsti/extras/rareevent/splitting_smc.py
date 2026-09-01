"""SMC / population-splitting estimator for QEC logical failure rates.

Variant C of the splitting-estimator fixes described in
`benchmarks/archive/splitting_fixes/PLAN.md` of the standalone
error-rate-estimation repository.

Why the baseline collapses at high distance
--------------------------------------------
`rare_event.rare_event_splitting_estimate` runs a *single* long
`ConditionalFailureMCMC` chain per level. At low p the conditional failure
distribution concentrates on many separate near-minimal malignant-set
"basins", each reachable from another only via low-probability intermediate
states (every uniform single-mechanism toggle that leaves a basin is either
rejected by the failure conditioning or carries an acceptance ratio of order
q_i << 1). A single chain therefore freezes inside whichever basin it started
in; more steps do not help because the chain simply is not going anywhere
else. Because the level ratios are then estimated from samples confined to a
handful of basins instead of the full conditional distribution, the
per-level ratio is biased low, and the bias compounds multiplicatively over
the schedule.

The SMC fix
-----------
This module keeps the exact same local move kernel
(`rare_event.ConditionalFailureMCMC`, an unchanged single-toggle Metropolis
step) but replaces the "one long chain" architecture with a *population* of
`num_walkers` walkers (a sequential-Monte-Carlo / subset-simulation scheme):

1. Anchor at p0 with direct Monte Carlo. Every distinct failing state seen
   among the `mc_shots_at_p0` i.i.d. draws is collected. Because these draws
   are i.i.d. samples from the unconditional distribution at p0, the failing
   ones seen are themselves i.i.d. draws from the conditional distribution
   P(. | fail, p0) -- so resampling *uniformly* from the collected set of
   distinct failing states, with replacement, produces a population that is
   (up to duplication) distributed as P(. | fail, p0). This population
   therefore *inherits* whatever multimodality direct Monte Carlo happened to
   discover at p0, instead of asking one chain to discover it by mixing.
2. At each level k -> k+1, the level ratio E_{p_k}[dP_{p_{k+1}}/dP_{p_k} |
   fail] is estimated as an importance-weighted mean over the *current*
   (pre-resample) population -- this is the unbiased estimator, computed
   before the population is disturbed by resampling.
3. The population is then multinomial-resampled according to those
   normalized importance weights. This is the step that preserves diversity:
   walkers sitting in under-weighted basins are pruned, walkers in
   over-weighted basins are duplicated, but *no basin that had any
   population mass is ever forced to be rediscovered from scratch*. Only
   after resampling does each walker take `mcmc_steps_per_walker` local
   single-toggle Metropolis steps at p_{k+1} to diversify away from its
   resampling duplicate and adapt to the new level -- a purely local
   refresh, not a global-mixing requirement.

Because diversity is carried level-to-level by the population rather than
demanded from one chain's global mixing, distinct malignant-set basins found
anywhere along the schedule remain represented (with resampling naturally
reweighting their relative importance), which is exactly what the baseline's
single frozen chain cannot do.

Diagnostics mapping (`LevelDiagnostics` fields; see `smc_splitting_estimate`
docstring for the full mapping)
-----------------------------------------------------------------------------
`LevelDiagnostics` was designed for the baseline's one-chain-per-quantity
picture, so several fields are repurposed here for a population:

- `pooled_log_ratio`: the log level-ratio estimate (logmeanexp of the
  pre-resample importance weights), same meaning as the baseline.
- `per_chain_log_ratios`: *not* one estimate per chain (there is only one
  population); instead it holds the individual pre-resample per-walker
  log-weight-ratio draws that the pooled estimate averages over.
- `per_chain_acceptance_rates`: one acceptance rate per walker from the
  post-resample local-move phase (walkers play the role "chains" play in
  the baseline).
- `per_chain_sample_sizes`: a single-element list `[unique_count]`, the
  number of distinct states immediately after resampling (before the move
  phase blurs it) -- this is the resampling-degeneracy diagnostic.
- `per_chain_mean_weights`: a single-element list `[mean active-set size]`
  of that same post-resample, pre-move population.
- `rhat_active_weight`: a genuine split-Rhat computed from each walker's
  trajectory of active-set sizes during its local-move phase (a real
  per-walker time series), diagnosing whether the short local refresh
  mixes reasonably.
- `rhat_log_weight_ratio`: always `None`. SMC produces exactly one
  importance weight per walker per level, not a time series, so a time-
  mixing diagnostic does not apply to it; the effective-sample-size (ESS,
  printed per level and derivable from `per_chain_log_ratios`) is the
  correct diagnostic for importance-weight degeneracy instead.
"""

from __future__ import annotations

import math
import random
import sys
from collections.abc import Sequence
from typing import Any

import numpy as np

from .interfaces import ErrorModel, Estimator, ForwardSimulator
from .rare_event import (
    ConditionalFailureMCMC,
    LevelDiagnostics,
    SplittingResult,
    log_weight_ratio,
    logmeanexp,
    split_rhat,
)


def _draw_active_set(probs: np.ndarray, rng: np.random.Generator) -> set[int]:
    draws = rng.random(len(probs)) < probs
    return set(np.flatnonzero(draws).tolist())


def smc_splitting_estimate(
    error_model: ErrorModel,
    simulator: ForwardSimulator,
    p_scales: Sequence[float],
    mc_shots_at_p0: int = 20_000,
    num_walkers: int = 256,
    mcmc_steps_per_walker: int = 1_500,
    init_mcmc_steps: int = 200,
    seed: int = 1,
) -> SplittingResult:
    """Estimate failure rates along a descending p schedule via population SMC.

    See the module docstring for the method and the `LevelDiagnostics` field
    mapping.

    Args:
        error_model: Provides independent mechanism probabilities q_i(p).
        simulator: Decides logical failure for an active mechanism set.
        p_scales: Descending physical error rates, p_scales[0] first.
        mc_shots_at_p0: Number of i.i.d. Bernoulli draws at p_scales[0] used
            both to estimate P_fail(p0) and to seed the initial population.
        num_walkers: Population size carried through every level.
        mcmc_steps_per_walker: Local single-toggle Metropolis steps applied
            to each walker after resampling at every level.
        init_mcmc_steps: Local single-toggle Metropolis steps applied to each
            walker of the initial p0 population before the first level.
        seed: Seeds both a `numpy.random.default_rng` (all numpy draws) and a
            `random.Random` (the `ConditionalFailureMCMC` move kernel).
    """
    if len(p_scales) < 2:
        raise ValueError("p_scales must contain at least two physical error rates.")
    if mc_shots_at_p0 < 1:
        raise ValueError(f"mc_shots_at_p0 must be at least 1; got {mc_shots_at_p0}.")
    if num_walkers < 2:
        raise ValueError(f"num_walkers must be at least 2; got {num_walkers}.")
    if mcmc_steps_per_walker < 1:
        raise ValueError(f"mcmc_steps_per_walker must be at least 1; got {mcmc_steps_per_walker}.")
    if init_mcmc_steps < 0:
        raise ValueError(f"init_mcmc_steps must be nonnegative; got {init_mcmc_steps}.")

    rng = np.random.default_rng(seed)
    py_rng = random.Random(seed)

    probs = [np.asarray(error_model.probabilities(p), dtype=np.float64) for p in p_scales]

    # --- Anchor at p0: direct Monte Carlo, collecting every distinct failing state. ---
    probs0 = probs[0]
    failures = 0
    seen: set[frozenset[int]] = set()
    failing_states: list[frozenset[int]] = []
    for _ in range(mc_shots_at_p0):
        active = _draw_active_set(probs0, rng)
        if simulator.fails(active):
            failures += 1
            key = frozenset(active)
            if key not in seen:
                seen.add(key)
                failing_states.append(key)

    p_fail0 = failures / mc_shots_at_p0
    if p_fail0 <= 0 or not failing_states:
        raise RuntimeError("No failures observed at p0. Choose a larger p0 or increase mc_shots_at_p0.")

    log_fail = math.log(p_fail0)
    log_failure_estimates = [log_fail]
    log_ratio_estimates: list[float] = []
    acceptance_rates: list[float] = []
    sample_sizes: list[int] = []
    level_diagnostics: list[LevelDiagnostics] = []

    # Initial population: sample the discovered failing states uniformly with
    # replacement (they are i.i.d. draws from P(. | fail, p0)), then
    # diversify with a shared move kernel at p0.
    init_indices = rng.integers(0, len(failing_states), size=num_walkers)
    population: list[set[int]] = [set(failing_states[i]) for i in init_indices]

    init_chain = ConditionalFailureMCMC(simulator, probs0, rng=py_rng)
    for j in range(num_walkers):
        state = population[j]
        for _ in range(init_mcmc_steps):
            state, _ = init_chain.step(state)
        population[j] = state

    print(
        f"Anchor p0={p_scales[0]:.6g} | shots={mc_shots_at_p0} | failures={failures} | "
        f"p_fail0={p_fail0:.6e} | distinct_failing_states={len(failing_states)} | walkers={num_walkers}"
    )
    sys.stdout.flush()

    for k in range(len(p_scales) - 1):
        probs_k = probs[k]
        probs_next = probs[k + 1]

        lw = np.array([log_weight_ratio(x, probs_next, probs_k) for x in population], dtype=np.float64)
        log_ratio, _ = logmeanexp(lw.tolist())
        log_fail += log_ratio

        m = float(np.max(lw))
        if not math.isfinite(m):
            raise RuntimeError(
                f"All importance weights vanished at level {k} "
                f"(p={p_scales[k]:.6g} -> {p_scales[k + 1]:.6g}); the population has collapsed. "
                "Use more walkers, more MCMC steps per walker, or a finer p schedule."
            )
        w = np.exp(lw - m)
        sum_w = float(np.sum(w))
        ess = float(sum_w * sum_w / float(np.sum(w * w)))
        norm_w = w / sum_w

        resample_idx = rng.choice(num_walkers, size=num_walkers, replace=True, p=norm_w)
        resampled: list[set[int]] = [set(population[i]) for i in resample_idx]
        unique_count = len({frozenset(s) for s in resampled})
        mean_weight = float(np.mean([len(s) for s in resampled]))

        move_chain = ConditionalFailureMCMC(simulator, probs_next, rng=py_rng)
        new_population: list[set[int]] = []
        accept_rates: list[float] = []
        weight_traces: list[list[float]] = []
        for j in range(num_walkers):
            state = resampled[j]
            trace = [float(len(state))]
            accepts = 0
            for _ in range(mcmc_steps_per_walker):
                state, accepted = move_chain.step(state)
                accepts += int(accepted)
                trace.append(float(len(state)))
            new_population.append(state)
            accept_rates.append(accepts / mcmc_steps_per_walker)
            weight_traces.append(trace)
        population = new_population
        mean_acc = float(np.mean(accept_rates))

        log_ratio_estimates.append(log_ratio)
        log_failure_estimates.append(log_fail)
        acceptance_rates.append(mean_acc)
        sample_sizes.append(num_walkers)

        rhat_active_weight = split_rhat(weight_traces)
        diag = LevelDiagnostics(
            level=k,
            p_current=p_scales[k],
            p_next=p_scales[k + 1],
            pooled_log_ratio=log_ratio,
            per_chain_log_ratios=lw.tolist(),
            per_chain_acceptance_rates=accept_rates,
            per_chain_sample_sizes=[unique_count],
            per_chain_mean_weights=[mean_weight],
            rhat_log_weight_ratio=None,
            rhat_active_weight=rhat_active_weight,
        )
        level_diagnostics.append(diag)

        rhat_w = "n/a" if rhat_active_weight is None else f"{rhat_active_weight:.3f}"
        line = (
            f"Level {k} -> {k + 1} | p={p_scales[k]:.6g} -> {p_scales[k + 1]:.6g} | "
            f"log_ratio={log_ratio:.6g} | log_fail={log_fail:.6g} | "
            f"P_fail={math.exp(log_fail):.6e} | acc={mean_acc:.3f} | "
            f"ESS={ess:.1f}/{num_walkers} | unique={unique_count}/{num_walkers} | "
            f"Rhat_active_weight={rhat_w}"
        )
        print(line)
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


class SMCSplittingEstimator(Estimator):
    """Estimator implementing SMC / population splitting (see module docstring)."""

    def estimate(
        self,
        error_model: ErrorModel,
        simulator: ForwardSimulator,
        **kwargs: Any,
    ) -> SplittingResult:
        if "p_scales" not in kwargs:
            raise ValueError("p_scales must be provided to SMCSplittingEstimator")

        return smc_splitting_estimate(
            error_model=error_model,
            simulator=simulator,
            p_scales=kwargs["p_scales"],
            mc_shots_at_p0=kwargs.get("mc_shots_at_p0", 20_000),
            num_walkers=kwargs.get("num_walkers", 256),
            mcmc_steps_per_walker=kwargs.get("mcmc_steps_per_walker", 1_500),
            init_mcmc_steps=kwargs.get("init_mcmc_steps", 200),
            seed=kwargs.get("seed", 1),
        )

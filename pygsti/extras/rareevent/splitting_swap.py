"""
Detector-adjacent swap-move splitting estimator (Variant B).

This is an alternative to ``rare_event.ConditionalFailureMCMC`` /
``rare_event_splitting_estimate`` that fixes a specific failure mode of the
baseline chain at high code distance: freeze-out on near-minimal malignant
sets.

The problem being fixed
------------------------
The baseline kernel proposes a *uniform single-mechanism toggle*: pick a
mechanism index uniformly over all ``n`` mechanisms and flip its membership.
At low physical error rate ``p`` the conditional-on-failure distribution
concentrates on near-minimal malignant sets, where every *remove* move
breaks the failure (rejected by the conditioning indicator) and every *add*
move carries Metropolis acceptance ratio ``odds_i ~= q_i`` -- a number as
small as ``1e-3`` to ``1e-4``. The chain is then stuck on one malignant set
for the overwhelming majority of steps, and it essentially never finds a
*different* same-weight malignant set, because doing so requires two
separate ``q``-suppressed add events (or one add plus one already-rejected
remove) to line up. Missing modes bias the per-level ratio estimate low, and
the bias compounds multiplicatively across levels.

The fix
-------
Add a second move type: a *detector-adjacent swap*. Instead of adding a
mechanism on its own, a swap simultaneously removes one active mechanism
``i`` and adds one *inactive* mechanism ``j`` that shares a detector with
``i`` (so ``j`` is a plausible local repair of the syndrome that removing
``i`` breaks). The Metropolis ratio for a swap is ``odds_j / odds_i``, an
``O(1)`` quantity for mechanisms with comparable probabilities -- no
``q``-suppression at all. Swaps therefore let the chain hop directly between
different same-weight malignant sets that lie in the same detector
neighbourhood, without ever passing through a q-suppressed add/remove
intermediate. Because the proposal is state dependent (it depends on which
mechanisms are locally adjacent to the currently active set), it needs its
own Metropolis-Hastings correction; see ``SwapConditionalFailureMCMC.step``
for the derivation of the correction term.

At each step the chain samples from a *mixture* of two kernels: the
baseline global toggle (with probability ``1 - swap_prob``) and the
detector-adjacent swap (with probability ``swap_prob``). Because each
component of a mixture of reversible kernels is individually
detailed-balance-preserving for the same target distribution, no
cross-component correction is required -- it suffices that each component
kernel, applied on its own, leaves the target invariant.

Detector adjacency
-------------------
``build_detector_adjacency`` maps each mechanism ``i`` to the mechanisms
that share at least one detector with it. Note this module's convention
excludes ``i`` from its own neighbor list (unlike Variant A's ``N(E)``
convention, which includes self-membership) -- neighbours here are only
ever used as *candidates to swap in*, and a mechanism can never usefully
swap with itself.
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
    LevelDiagnostics,
    MechanismCatalog,
    SplittingResult,
    direct_monte_carlo_failure_rate,
    log_weight_ratio,
    logmeanexp,
    split_rhat,
)


def build_detector_adjacency(catalog: MechanismCatalog) -> list[tuple[int, ...]]:
    """Build the detector-sharing adjacency list for a mechanism catalog.

    ``neighbors[i]`` is the sorted tuple of mechanism indices that share at
    least one detector target with mechanism ``i``, EXCLUDING ``i`` itself.
    (This differs from Variant A's ``N(E)`` convention, which folds ``i``
    into its own neighborhood; here neighbors are only ever used as swap-in
    candidates, so self-membership would be meaningless.) The relation is
    symmetric by construction: ``j in neighbors[i]`` iff mechanisms ``i``
    and ``j`` share a detector, which holds iff ``i in neighbors[j]``.
    """
    detector_to_mechanisms: dict[int, list[int]] = {}
    for idx, mech in enumerate(catalog.mechanisms):
        for d in mech.detectors:
            detector_to_mechanisms.setdefault(d, []).append(idx)

    neighbor_sets: list[set[int]] = [set() for _ in catalog.mechanisms]
    for mechs in detector_to_mechanisms.values():
        for a in mechs:
            for b in mechs:
                if a != b:
                    neighbor_sets[a].add(b)

    return [tuple(sorted(s)) for s in neighbor_sets]


class SwapConditionalFailureMCMC:
    """Metropolis chain targeting P_p(DEM mechanism set E | decoder failure).

    Mixture of two self-reversible Metropolis kernels, chosen independently
    at each step:

    - with probability ``1 - swap_prob``, the baseline global uniform single
      toggle (identical math to ``rare_event.ConditionalFailureMCMC.step``);
    - with probability ``swap_prob``, a detector-adjacent swap that removes
      one active mechanism and adds one detector-adjacent inactive one.

    Swap move derivation
    ---------------------
    Let ``E`` be the current active set, ``i`` an element of ``E`` chosen
    uniformly (probability ``1/|E|``), and ``C(i, E) = neighbors[i] \\ E``
    the inactive mechanisms adjacent to ``i``. Propose ``j`` uniformly from
    ``C(i, E)`` (probability ``1/|C(i, E)|``) and set
    ``E' = E - {i} + {j}``.

    The reverse move (from ``E'`` back to ``E``) must pick ``j`` from
    ``E'`` (probability ``1/|E'| = 1/|E|``, since the swap preserves
    cardinality) and then pick ``i`` from ``C(j, E') = neighbors[j] \\ E'``.
    Because adjacency is symmetric and ``j in neighbors[i]`` by
    construction, ``i in neighbors[j]``; and ``i`` is not in ``E'`` (it was
    just removed), so ``i in C(j, E')`` always holds -- the reverse proposal
    can always reach ``E`` from ``E'``. The forward proposal density is
    ``1/(|E| * |C(i, E)|)`` and the reverse is ``1/(|E'| * |C(j, E')|)``;
    since ``|E'| = |E|`` these cancel to a Hastings correction of
    ``|C(i, E)| / |C(j, E')|``. Combined with the unconditioned probability
    ratio ``odds_j / odds_i`` (swap out mechanism ``i``, swap in ``j``), the
    Metropolis-Hastings acceptance probability is

        min(1, (odds_j / odds_i) * |C(i, E)| / |C(j, E')|).

    As with the baseline chain, the ratio test is applied first (cheap),
    and the proposal is rejected if it is not a decoder failure (the
    conditioning indicator), which requires a call to the oracle -- for
    swaps this oracle check is consulted much more often than for toggles,
    since swap ratios are not q-suppressed.
    """

    def __init__(
        self,
        oracle: ForwardSimulator,
        probabilities: np.ndarray,
        neighbors: Sequence[tuple[int, ...]],
        rng: random.Random | None = None,
        swap_prob: float = 0.5,
    ) -> None:
        self.oracle = oracle
        self.p = np.asarray(probabilities, dtype=np.float64)
        if np.any(self.p <= 0) or np.any(self.p >= 1):
            raise ValueError("All mechanism probabilities must be in (0, 1).")
        if len(neighbors) != len(self.p):
            raise ValueError(
                f"neighbors has {len(neighbors)} entries but probabilities has {len(self.p)}."
            )
        if not (0.0 <= swap_prob <= 1.0):
            raise ValueError(f"swap_prob must be in [0, 1]; got {swap_prob}.")
        self.odds = self.p / (1 - self.p)
        self.neighbors = neighbors
        self.swap_prob = swap_prob
        self.rng = rng or random.Random()

        # Separate acceptance bookkeeping per kernel component, for diagnostics.
        self.toggle_attempts = 0
        self.toggle_accepts = 0
        self.swap_attempts = 0
        self.swap_accepts = 0

    @property
    def toggle_acceptance_rate(self) -> float:
        return self.toggle_accepts / self.toggle_attempts if self.toggle_attempts else float("nan")

    @property
    def swap_acceptance_rate(self) -> float:
        return self.swap_accepts / self.swap_attempts if self.swap_attempts else float("nan")

    def seed_from_monte_carlo(self, max_tries: int = 1_000_000) -> set[int]:
        """Find an initial failing state by direct sampling at the current p."""
        n = len(self.p)
        for _ in range(max_tries):
            draws = np.random.random(n) < self.p
            active = set(np.flatnonzero(draws).tolist())
            if self.oracle.fails(active):
                return active
        raise RuntimeError(
            "Could not find an initial failing state. Increase p0, max_tries, or seed manually."
        )

    def _toggle_step(self, active: set[int]) -> tuple[set[int], bool]:
        """Baseline global uniform single-mechanism toggle (see ConditionalFailureMCMC.step)."""
        self.toggle_attempts += 1
        n = len(self.p)
        i = self.rng.randrange(n)
        proposed = set(active)

        if i in active:
            proposed.remove(i)
            ratio = 1.0 / self.odds[i]
        else:
            proposed.add(i)
            ratio = self.odds[i]

        accept_prob = min(1.0, ratio)
        if self.rng.random() >= accept_prob:
            return active, False

        if not self.oracle.fails(proposed):
            return active, False

        self.toggle_accepts += 1
        return proposed, True

    def _swap_step(self, active: set[int]) -> tuple[set[int], bool]:
        """Detector-adjacent swap: remove active i, add inactive detector-adjacent j."""
        self.swap_attempts += 1
        if not active:
            return active, False

        # Uniform choice of i from E. `active` is a plain set; materialize a
        # sorted list so the draw is reproducible given the rng's seed
        # (Python set iteration order is not part of the language spec).
        ordered_active = sorted(active)
        i = self.rng.choice(ordered_active)

        candidates_i = [m for m in self.neighbors[i] if m not in active]
        if not candidates_i:
            return active, False
        j = self.rng.choice(candidates_i)

        proposed = set(active)
        proposed.discard(i)
        proposed.add(j)

        candidates_j = [m for m in self.neighbors[j] if m not in proposed]
        # Invariant: since j in neighbors[i] and adjacency is symmetric,
        # i in neighbors[j]; and i was just removed so i not in proposed.
        # Hence i must appear in candidates_j -- the reverse move can always
        # reach `active` back from `proposed`.
        assert i in candidates_j, "adjacency symmetry invariant violated"

        ratio = (self.odds[j] / self.odds[i]) * (len(candidates_i) / len(candidates_j))
        accept_prob = min(1.0, ratio)
        if self.rng.random() >= accept_prob:
            return active, False

        if not self.oracle.fails(proposed):
            return active, False

        self.swap_accepts += 1
        return proposed, True

    def step(self, active: set[int]) -> tuple[set[int], bool]:
        """One Metropolis step: dispatch to the swap or toggle kernel."""
        if self.rng.random() < self.swap_prob:
            return self._swap_step(active)
        return self._toggle_step(active)

    def sample(
        self,
        initial: set[int],
        steps: int,
        burn_in: int = 0,
        thin: int = 1,
    ) -> tuple[list[set[int]], float]:
        if not self.oracle.fails(initial):
            raise ValueError("Initial state is not a logical failure.")
        active = set(initial)
        samples: list[set[int]] = []
        accepts = 0
        total = 0
        for t in range(steps):
            active, accepted = self.step(active)
            accepts += int(accepted)
            total += 1
            if t >= burn_in and ((t - burn_in) % thin == 0):
                samples.append(set(active))
        return samples, accepts / max(total, 1)


def swap_splitting_estimate(
    error_model: ErrorModel,
    simulator: ForwardSimulator,
    p_scales: Sequence[float],
    catalog: MechanismCatalog,
    mc_shots_at_p0: int,
    steps_per_chain: int | None,
    total_steps_per_level: int | None,
    burn_in: int | None,
    burn_in_fraction: float | None = 0.1,
    thin: int = 1,
    seed: int = 1,
    swap_prob: float = 0.5,
) -> SplittingResult:
    """Estimate failure rates along a descending p_scales schedule using swap moves.

    Same anchor-then-walk flow as ``rare_event.rare_event_splitting_estimate``:
    direct Monte Carlo at ``p_scales[0]``, then for each level a
    ``SwapConditionalFailureMCMC`` chain at ``p_scales[k]`` estimates the
    ratio to ``p_scales[k + 1]`` via ``logmeanexp`` of ``log_weight_ratio``
    over the kept (post-burn-in, thinned) samples.
    """

    rng = random.Random(seed)
    np.random.seed(seed)
    if thin < 1:
        raise ValueError(f"thin must be at least 1; got {thin}.")
    if steps_per_chain is None and total_steps_per_level is None:
        raise ValueError("Specify either steps_per_chain or total_steps_per_level.")
    if steps_per_chain is not None and total_steps_per_level is not None:
        raise ValueError("Specify only one of steps_per_chain or total_steps_per_level.")

    if steps_per_chain is not None:
        per_chain_steps = int(steps_per_chain)
    else:
        assert total_steps_per_level is not None
        per_chain_steps = int(total_steps_per_level)

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
            f"MCMC proposal steps must exceed burn-in; got steps={per_chain_steps}, burn_in={per_chain_burn_in}. "
            "Use a larger step count or smaller burn-in fraction."
        )

    neighbors = build_detector_adjacency(catalog)
    probs = [error_model.probabilities(p) for p in p_scales]

    # Anchor the estimator at p0 where ordinary Monte Carlo is feasible.
    p_fail0, se0, initial_failure = direct_monte_carlo_failure_rate(
        simulator, probs[0], mc_shots_at_p0
    )
    if p_fail0 <= 0 or initial_failure is None:
        raise RuntimeError(
            "No failures observed at p0. Choose a larger p0 or increase mc_shots_at_p0."
        )

    log_fail = math.log(p_fail0)
    log_failure_estimates = [log_fail]
    log_ratio_estimates: list[float] = []
    acceptance_rates: list[float] = []
    sample_sizes: list[int] = []
    level_diagnostics: list[LevelDiagnostics] = []

    active = initial_failure

    for k in range(len(p_scales) - 1):
        chain = SwapConditionalFailureMCMC(simulator, probs[k], neighbors, rng=rng, swap_prob=swap_prob)
        if not simulator.fails(active):
            active = chain.seed_from_monte_carlo()

        samples, acc = chain.sample(
            initial=active,
            steps=per_chain_steps,
            burn_in=per_chain_burn_in,
            thin=thin,
        )

        log_ratio, _ = logmeanexp(
            [log_weight_ratio(state, probs[k + 1], probs[k]) for state in samples]
        )
        log_fail += log_ratio

        log_ratio_estimates.append(log_ratio)
        log_failure_estimates.append(log_fail)
        acceptance_rates.append(acc)
        sample_sizes.append(len(samples))

        per_chain_log_ratio_samples = []
        per_chain_weight_samples = []
        for state in samples:
            per_chain_log_ratio_samples.append(log_weight_ratio(state, probs[k + 1], probs[k]))
            per_chain_weight_samples.append(float(len(state)))

        diag = LevelDiagnostics(
            level=k,
            p_current=p_scales[k],
            p_next=p_scales[k + 1],
            pooled_log_ratio=log_ratio,
            per_chain_log_ratios=[log_ratio],
            per_chain_acceptance_rates=[acc],
            per_chain_sample_sizes=[len(samples)],
            per_chain_mean_weights=[float(np.mean(per_chain_weight_samples))],
            rhat_log_weight_ratio=split_rhat([per_chain_log_ratio_samples]),
            rhat_active_weight=split_rhat([per_chain_weight_samples]),
        )
        level_diagnostics.append(diag)

        rhat_lr = "n/a" if diag.rhat_log_weight_ratio is None else f"{diag.rhat_log_weight_ratio:.3f}"
        rhat_w = "n/a" if diag.rhat_active_weight is None else f"{diag.rhat_active_weight:.3f}"
        line = (
            f"Level {k} -> {k+1} | "
            f"p={p_scales[k]:.6g} -> {p_scales[k+1]:.6g} | "
            f"log_ratio={log_ratio:.6g} | log_fail={log_fail:.6g} | "
            f"P_fail={math.exp(log_fail):.6e} | acc={acc:.3f} "
            f"(toggle={chain.toggle_acceptance_rate:.3f}, swap={chain.swap_acceptance_rate:.3f}) | "
            f"samples={len(samples)} | Rhat_log_weight_ratio={rhat_lr} | Rhat_active_weight={rhat_w}"
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


class SwapSplittingEstimator(Estimator):
    """Rare-event splitting estimator with detector-adjacent swap MCMC moves (Variant B)."""

    def estimate(
        self,
        error_model: ErrorModel,
        simulator: ForwardSimulator,
        **kwargs: Any,
    ) -> SplittingResult:
        if "p_scales" not in kwargs:
            raise ValueError("p_scales must be provided to SwapSplittingEstimator")
        if "catalog" not in kwargs:
            raise ValueError("catalog must be provided to SwapSplittingEstimator")

        return swap_splitting_estimate(
            error_model=error_model,
            simulator=simulator,
            p_scales=kwargs["p_scales"],
            catalog=kwargs["catalog"],
            mc_shots_at_p0=kwargs.get("mc_shots_at_p0", 10000),
            steps_per_chain=kwargs.get("steps_per_chain"),
            total_steps_per_level=kwargs.get("total_steps_per_level"),
            burn_in=kwargs.get("burn_in"),
            burn_in_fraction=kwargs.get("burn_in_fraction", 0.1),
            thin=kwargs.get("thin", 1),
            seed=kwargs.get("seed", 1),
            swap_prob=kwargs.get("swap_prob", 0.5),
        )

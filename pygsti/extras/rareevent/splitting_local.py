"""
Locality-aware rare-event splitting estimator ("Variant A" of the splitting fixes).

Why this exists
----------------
`rare_event.ConditionalFailureMCMC` (the baseline) proposes toggling a single DEM
mechanism chosen *uniformly* over all n mechanisms. At low p, the conditional
failure distribution P_p(E | failure) concentrates on near-minimal malignant
sets. For a chain sitting on such a set E:

  - every *remove* move breaks the failure condition and is rejected by the
    conditioning indicator, and
  - every *add* move is accepted with Metropolis ratio odds_i ~ q_i(p), which
    is tiny at low p.

So almost every proposal is either doomed by the conditioning or heavily
q-suppressed, and -- crucially -- the uniform proposal only touches a
mechanism that is actually *relevant* to the current set (i.e. detector-
adjacent to it, and therefore capable of producing another nearby malignant
set) with probability |E|/n. n grows quickly with code distance, so this
"relevant touch rate" collapses and the chain freezes on one malignant-set
basin, undercounting the true conditional support and biasing P_fail low.

The fix in this module raises that relevant-touch rate from |E|/n to O(1) by
proposing mechanisms from the detector neighborhood of the current active set,
N(E) = union over i in E of neighbors[i] (neighbors[i] itself includes i, so
N(E) always contains E). The proposal is a two-component mixture:

    q(i | E) = beta/n + (1 - beta) * 1[i in N(E)] / |N(E)|

With probability `beta` (`beta_global`, default 0.1) we fall back to the
baseline's uniform-over-all-n proposal; with probability `1 - beta` we pick
uniformly from N(E). The `beta` component alone already guarantees the chain
has the same support/irreducibility properties as the baseline (any move the
baseline can make, this proposal can also make, with probability >= beta/n),
so nothing is lost, while the local component makes most proposals land on a
mechanism that can plausibly keep (or restore) the failure condition.

Because the proposal density depends on the current state, a plain symmetric
Metropolis acceptance is *not* valid here: this is a genuine Metropolis-
Hastings step, with acceptance

    min(1, odds_i^{+-1} * q(i | E') / q(i | E))

(+1 exponent for adding i, -1 for removing i), where E' is the toggled state.
The oracle (`ForwardSimulator.fails`) is only ever consulted after the cheap
Metropolis ratio test has already passed, mirroring the baseline's "ratio
first, decode second" order of operations.

N(E) is maintained incrementally rather than recomputed each step: an integer
`cover_count[m]` tracks how many currently-active mechanisms have m in their
detector-adjacency neighborhood; N(E) = {m : cover_count[m] > 0}, kept as a
dynamic array (`members`) plus a position index (`position`) so that sampling
uniformly from N(E) and adding/removing a mechanism's neighborhood from the
cover are both O(degree), not O(n) or O(|E|).
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
    """Return neighbors[i] = sorted mechanism indices sharing >= 1 detector with i.

    Each mechanism is included in its own neighbor list. Mechanisms that flip no
    detectors (e.g. a pure-observable mechanism) get neighbors == (i,). The
    relation is symmetric by construction (shared-detector membership).
    """
    n = len(catalog.mechanisms)
    detector_to_mechanisms: dict[int, list[int]] = {}
    for idx, mech in enumerate(catalog.mechanisms):
        for d in mech.detectors:
            detector_to_mechanisms.setdefault(d, []).append(idx)

    neighbor_sets: list[set[int]] = [{i} for i in range(n)]
    for idx, mech in enumerate(catalog.mechanisms):
        for d in mech.detectors:
            neighbor_sets[idx].update(detector_to_mechanisms[d])

    return [tuple(sorted(s)) for s in neighbor_sets]


class LocalConditionalFailureMCMC:
    """Metropolis-Hastings chain targeting P_p(E | failure) with locality-aware proposals.

    See the module docstring for the proposal density, acceptance rule, and the
    incremental cover-count bookkeeping used to maintain N(E) = union of
    neighbors[i] for i in the current active set E.
    """

    def __init__(
        self,
        oracle: ForwardSimulator,
        probabilities: np.ndarray,
        neighbors: Sequence[tuple[int, ...]],
        rng: random.Random | None = None,
        beta_global: float = 0.1,
    ):
        self.oracle = oracle
        self.p = np.asarray(probabilities, dtype=np.float64)
        if np.any(self.p <= 0) or np.any(self.p >= 1):
            raise ValueError("All mechanism probabilities must be in (0, 1).")
        if len(neighbors) != len(self.p):
            raise ValueError("neighbors must have one entry per mechanism.")
        if not (0.0 < beta_global <= 1.0):
            raise ValueError(f"beta_global must be in (0, 1]; got {beta_global}.")
        self.odds = self.p / (1 - self.p)
        self.neighbors: list[tuple[int, ...]] = [tuple(nb) for nb in neighbors]
        self.rng = rng or random.Random()
        self.beta_global = float(beta_global)

        n = len(self.p)
        self.active: set[int] = set()
        self.cover_count: np.ndarray = np.zeros(n, dtype=np.int64)
        self.members: list[int] = []
        self.position: dict[int, int] = {}

    # -- cover-count / N(E) bookkeeping -----------------------------------

    def _cover_add(self, m: int) -> None:
        count = int(self.cover_count[m]) + 1
        self.cover_count[m] = count
        if count == 1:
            self.position[m] = len(self.members)
            self.members.append(m)

    def _cover_remove(self, m: int) -> None:
        count = int(self.cover_count[m]) - 1
        self.cover_count[m] = count
        if count == 0:
            pos = self.position.pop(m)
            last = self.members.pop()
            if pos < len(self.members):
                self.members[pos] = last
                self.position[last] = pos

    def _apply_delta(self, i: int, sign: int) -> None:
        if sign > 0:
            for m in self.neighbors[i]:
                self._cover_add(m)
        else:
            for m in self.neighbors[i]:
                self._cover_remove(m)

    def _peek_toggle(self, i: int, sign: int) -> tuple[int, bool]:
        """Read-only preview of |N(E')| and whether i in N(E') after toggling i by `sign`.

        Does not mutate cover_count/members/position. Used to evaluate the reverse
        proposal density q(i | E') for the Metropolis-Hastings ratio without paying
        for a mutate-then-revert round trip on the (common) rejected-proposal path.
        """
        k = len(self.members)
        in_n_after_i = False
        for m in self.neighbors[i]:
            count = int(self.cover_count[m])
            new_count = count + sign
            if count == 0 and new_count > 0:
                k += 1
            elif count > 0 and new_count == 0:
                k -= 1
            if m == i:
                in_n_after_i = new_count > 0
        return k, in_n_after_i

    def set_state(self, active: set[int]) -> None:
        """Rebuild the cover-count / N(E) structure from scratch for `active`."""
        n = len(self.p)
        self.active = set(active)
        self.cover_count = np.zeros(n, dtype=np.int64)
        self.members = []
        self.position = {}
        for i in self.active:
            for m in self.neighbors[i]:
                self._cover_add(m)

    # -- proposal ------------------------------------------------------------

    def _density(self, k: int, in_n: bool) -> float:
        n = len(self.p)
        if k == 0:
            # N(E) is empty (only when E is empty): the local component has no
            # support, so all proposal mass is effectively the global uniform.
            return 1.0 / n
        density = self.beta_global / n
        if in_n:
            density += (1.0 - self.beta_global) / k
        return density

    def _proposal_density(self, i: int) -> float:
        return self._density(len(self.members), i in self.position)

    def _propose_index(self) -> int:
        n = len(self.p)
        k = len(self.members)
        if k == 0 or self.rng.random() < self.beta_global:
            return self.rng.randrange(n)
        return self.members[self.rng.randrange(k)]

    def step(self, active: set[int]) -> tuple[set[int], bool]:
        """One Metropolis-Hastings step using the locality-aware toggle proposal.

        Mirrors `ConditionalFailureMCMC.step`'s signature, but the chain owns
        persistent cover-count state internally: if `active` matches the
        chain's current state (the common case, when callers feed back the
        set returned by the previous call), no rebuild is needed and this is
        O(degree) per step. If `active` differs (e.g. externally reseeded),
        the cover-count structure is rebuilt from scratch first.
        """
        if active != self.active:
            self.set_state(active)
        accepted = self._step_once()
        return set(self.active), accepted

    def _step_once(self) -> bool:
        i = self._propose_index()
        adding = i not in self.active
        sign = 1 if adding else -1

        q_forward = self._proposal_density(i)
        # Read-only preview of q(i | E'): the common case is rejection at the cheap
        # ratio test below, so avoid mutating cover_count/members until a move is
        # actually accepted (both here and by the oracle).
        k_after, in_n_after = self._peek_toggle(i, sign)
        q_backward = self._density(k_after, in_n_after)

        ratio = self.odds[i] if adding else 1.0 / self.odds[i]
        accept_prob = min(1.0, ratio * q_backward / q_forward)

        if self.rng.random() >= accept_prob:
            return False

        self._apply_delta(i, sign)
        if adding:
            self.active.add(i)
        else:
            self.active.discard(i)

        if not self.oracle.fails(self.active):
            if adding:
                self.active.discard(i)
            else:
                self.active.add(i)
            self._apply_delta(i, -sign)
            return False

        return True

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

    def sample(
        self,
        initial: set[int],
        steps: int,
        burn_in: int = 0,
        thin: int = 1,
    ) -> tuple[list[set[int]], float]:
        if not self.oracle.fails(initial):
            raise ValueError("Initial state is not a logical failure.")
        self.set_state(initial)
        active = set(self.active)
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


def local_splitting_estimate(
    error_model: ErrorModel,
    simulator: ForwardSimulator,
    catalog: MechanismCatalog,
    p_scales: Sequence[float],
    mc_shots_at_p0: int,
    steps_per_chain: int | None,
    total_steps_per_level: int | None,
    burn_in: int | None,
    burn_in_fraction: float | None = 0.1,
    thin: int = 1,
    seed: int = 1,
    beta_global: float = 0.1,
    anchor_failure_rate: float | None = None,
    anchor_state: set[int] | None = None,
    seed_states: Sequence[set[int]] | None = None,
    num_chains: int = 1,
) -> SplittingResult:
    """Estimate failure rates along a descending p_scales schedule using locality-aware MCMC.

    Same anchor-then-descend flow as `rare_event.rare_event_splitting_estimate`, but each
    level's conditional expectation is estimated with `LocalConditionalFailureMCMC` instead
    of the baseline's uniform-toggle `ConditionalFailureMCMC`. The detector adjacency used by
    the local proposal is built once from `catalog` before the descent begins.

    By default the anchor P_fail(p0) is measured internally with `mc_shots_at_p0` direct
    per-shot Monte Carlo samples. When p0 itself is deep enough that per-shot sampling is
    impractical, pass `anchor_failure_rate` (an externally measured estimate of P_fail(p0),
    e.g. from vectorized catalog-model MC) together with `anchor_state` (a failing
    configuration at p0 to start the first chain); the internal anchor MC is then skipped
    and `mc_shots_at_p0` is ignored.

    Multi-chain levels (`num_chains` > 1) attack the freeze-out bias directly: each level
    runs `num_chains` independent chains and pools their samples for the level ratio. Chain 0
    starts from the anchor/MC state as before; chains 1..num_chains-1 start from
    `seed_states` (cycled), which should be *light* failing configurations the p-weighted
    chain cannot reach by local moves — e.g. near-onset states harvested by
    `gap_splitting.estimate_f_w_gap_splitting(..., harvest_states=...)`. Failure of a
    configuration is p-independent, so the same seeds are valid at every level. Every seed
    is validated with the simulator up front. Per-chain level ratios and a cross-chain
    split-Rhat are reported in `LevelDiagnostics`; chain disagreement (Rhat >> 1) is the
    signature of basins the single-chain estimator was missing. When `total_steps_per_level`
    is given it is divided evenly across the chains; `steps_per_chain` applies to each chain
    individually. With `num_chains=1` (default) the behavior and RNG stream are identical to
    the single-chain estimator, and `seed_states` is ignored.
    """

    rng = random.Random(seed)
    np.random.seed(seed)
    if thin < 1:
        raise ValueError(f"thin must be at least 1; got {thin}.")
    if num_chains < 1:
        raise ValueError(f"num_chains must be at least 1; got {num_chains}.")
    if steps_per_chain is None and total_steps_per_level is None:
        raise ValueError("Specify either steps_per_chain or total_steps_per_level.")
    if steps_per_chain is not None and total_steps_per_level is not None:
        raise ValueError("Specify only one of steps_per_chain or total_steps_per_level.")

    if steps_per_chain is not None:
        per_chain_steps = int(steps_per_chain)
    else:
        assert total_steps_per_level is not None
        per_chain_steps = int(total_steps_per_level) // num_chains

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

    # Anchor the estimator at p0: either an externally supplied estimate (with a
    # known failing state) or internal direct Monte Carlo where that is feasible.
    initial_failure: set[int] | None
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
        p_fail0, _se0, initial_failure = direct_monte_carlo_failure_rate(simulator, probs[0], mc_shots_at_p0)
        if p_fail0 <= 0 or initial_failure is None:
            raise RuntimeError("No failures observed at p0. Choose a larger p0 or increase mc_shots_at_p0.")

    log_fail = math.log(p_fail0)
    log_failure_estimates = [log_fail]
    log_ratio_estimates: list[float] = []
    acceptance_rates: list[float] = []
    sample_sizes: list[int] = []
    level_diagnostics: list[LevelDiagnostics] = []

    # Chain 0 starts from the anchor state; chains 1..num_chains-1 start from the
    # (validated) externally supplied seed states, cycled. A configuration's failure
    # status is p-independent, so the same initial states are reused at every level.
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

    for k in range(len(p_scales) - 1):
        per_chain_samples: list[list[set[int]]] = []
        per_chain_acc: list[float] = []
        for c in range(num_chains):
            chain = LocalConditionalFailureMCMC(simulator, probs[k], neighbors, rng=rng, beta_global=beta_global)
            if not simulator.fails(chain_initials[c]):
                chain_initials[c] = chain.seed_from_monte_carlo()

            samples_c, acc_c = chain.sample(
                initial=chain_initials[c],
                steps=per_chain_steps,
                burn_in=per_chain_burn_in,
                thin=thin,
            )
            per_chain_samples.append(samples_c)
            per_chain_acc.append(acc_c)

        all_samples = [state for chain_samples in per_chain_samples for state in chain_samples]
        log_ratio, _ = logmeanexp([log_weight_ratio(state, probs[k + 1], probs[k]) for state in all_samples])
        log_fail += log_ratio
        acc = float(np.mean(per_chain_acc))

        log_ratio_estimates.append(log_ratio)
        log_failure_estimates.append(log_fail)
        acceptance_rates.append(acc)
        sample_sizes.append(len(all_samples))

        per_chain_log_ratio_samples = [
            [log_weight_ratio(state, probs[k + 1], probs[k]) for state in chain_samples]
            for chain_samples in per_chain_samples
        ]
        per_chain_weight_samples = [
            [float(len(state)) for state in chain_samples] for chain_samples in per_chain_samples
        ]

        diag = LevelDiagnostics(
            level=k,
            p_current=p_scales[k],
            p_next=p_scales[k + 1],
            pooled_log_ratio=log_ratio,
            per_chain_log_ratios=[logmeanexp(lrs)[0] for lrs in per_chain_log_ratio_samples],
            per_chain_acceptance_rates=list(per_chain_acc),
            per_chain_sample_sizes=[len(chain_samples) for chain_samples in per_chain_samples],
            per_chain_mean_weights=[float(np.mean(ws)) for ws in per_chain_weight_samples],
            rhat_log_weight_ratio=split_rhat(per_chain_log_ratio_samples),
            rhat_active_weight=split_rhat(per_chain_weight_samples),
        )
        level_diagnostics.append(diag)

        rhat_lr = "n/a" if diag.rhat_log_weight_ratio is None else f"{diag.rhat_log_weight_ratio:.3f}"
        rhat_w = "n/a" if diag.rhat_active_weight is None else f"{diag.rhat_active_weight:.3f}"
        line = (
            f"Level {k} -> {k+1} | "
            f"p={p_scales[k]:.6g} -> {p_scales[k+1]:.6g} | "
            f"log_ratio={log_ratio:.6g} | log_fail={log_fail:.6g} | "
            f"P_fail={math.exp(log_fail):.6e} | acc={acc:.3f} | chains={num_chains} | "
            f"samples={len(all_samples)} | Rhat_log_weight_ratio={rhat_lr} | Rhat_active_weight={rhat_w}"
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


class LocalSplittingEstimator(Estimator):
    """Estimator implementing locality-aware rare-event splitting ("Variant A")."""

    def estimate(
        self,
        error_model: ErrorModel,
        simulator: ForwardSimulator,
        **kwargs: Any,
    ) -> SplittingResult:
        if "p_scales" not in kwargs:
            raise ValueError("p_scales must be provided to LocalSplittingEstimator")
        if "catalog" not in kwargs:
            raise ValueError("catalog must be provided to LocalSplittingEstimator")

        return local_splitting_estimate(
            error_model=error_model,
            simulator=simulator,
            catalog=kwargs["catalog"],
            p_scales=kwargs["p_scales"],
            mc_shots_at_p0=kwargs.get("mc_shots_at_p0", 10000),
            steps_per_chain=kwargs.get("steps_per_chain"),
            total_steps_per_level=kwargs.get("total_steps_per_level"),
            burn_in=kwargs.get("burn_in"),
            burn_in_fraction=kwargs.get("burn_in_fraction", 0.1),
            thin=kwargs.get("thin", 1),
            seed=kwargs.get("seed", 1),
            beta_global=kwargs.get("beta_global", 0.1),
            anchor_failure_rate=kwargs.get("anchor_failure_rate"),
            anchor_state=kwargs.get("anchor_state"),
            seed_states=kwargs.get("seed_states"),
            num_chains=kwargs.get("num_chains", 1),
        )

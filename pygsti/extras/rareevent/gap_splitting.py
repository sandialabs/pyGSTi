"""
Fixed-weight gap-splitting estimator for the failure spectrum f(w).

f(w) = P(fail | W = w) is the failure fraction of weight-w fault sets (see the
module docstring of ``failure_spectrum.py``). Plain rejection sampling
(``failure_spectrum.sample_fixed_weight_failure_fraction``) estimates f(w) by
drawing i.i.d. weight-w sets and counting failures; it cannot resolve f(w)
below roughly ``1 / max_trials``. This module reaches much smaller f(w) with a
subset-simulation (adaptive multilevel splitting) scheme built around
PyMatching's *complementary gap*: a per-syndrome score that tells us, without
ever calling the decoder's pass/fail oracle, how close a fault set is to
flipping the decoder's answer.

The gap score
-------------
For a fixed-weight fault set E with true logical class t, decode the syndrome
twice with an extra "gap detector" pinned to each class b in {0, 1}
(``GapOracle.gap``); this returns the MWPM weight ``w_b`` of the best
correction in class b. The signed gap ``G(E) = w_{1-t} - w_t`` is negative
exactly when the decoder would prefer the wrong class (a strong local
predictor of `oracle.fails(E)`, though ties and implementation details mean
the two are not identical -- the true failure event is always resolved with
`oracle.fails`, never with the sign of `G` alone).

Subset simulation
------------------
Starting from the exact conditional distribution P(E | |E| = w), a sequence of
nested level sets {G(E) <= g_1} superset {G(E) <= g_2} superset ... is walked
down (each g_k chosen as an empirical quantile of the current population's G
values) using weight-preserving MCMC to rejuvenate a particle population after
each resampling step, exactly as in classical subset simulation / sequential
Monte Carlo for rare events. The product of the empirical survival fractions
at each level, times the true failure fraction of the final population, is an
unbiased-in-expectation (over the MCMC noise) estimate of f(w) that can reach
far smaller values than direct rejection sampling for the same particle
budget, because each level only has to resolve a probability of order
``quantile`` rather than the full f(w).

Dependencies: this module is decoder-facing (it builds its own PyMatching
`Matching` objects from the DEM's gap-detector construction), consistent with
the rest of the package. The MCMC state is a `MechanismCatalog` index set.
"""

from __future__ import annotations

import dataclasses
import math
import random
from collections.abc import Sequence
from typing import Any, Protocol

import numpy as np
import pymatching
import stim

from .failure_spectrum import tilted_probabilities
from .interfaces import ErrorModel, Estimator, ForwardSimulator
from .rare_event import FailureOracle, MechanismCatalog
from .splitting_swap import build_detector_adjacency
from .weight_points import WeightPoint

# ---------------------------------------------------------------------------
# Complementary-gap matching construction
# ---------------------------------------------------------------------------


def _copy_edge_kwargs(attr: dict[str, Any]) -> dict[str, Any]:
    """Copy PyMatching edge attributes into kwargs for add_edge/add_boundary_edge."""
    kwargs: dict[str, Any] = {
        "fault_ids": set(attr.get("fault_ids", set())),
        "weight": float(attr.get("weight", 1.0)),
    }
    p = attr.get("error_probability", -1.0)
    if p is not None and p >= 0:
        kwargs["error_probability"] = float(p)
    return kwargs


def make_gap_matching_from_vanilla_dem(
    dem: stim.DetectorErrorModel, logical_id: int = 0
) -> tuple[pymatching.Matching, pymatching.Matching, int]:
    """Build a base decoder and a "gap" decoder exposing the complementary-gap score.

    Adapted from the archived exploratory script
    ``benchmarks/diagnostics/complementary_gap.py`` in the standalone
    error-rate-estimation repository (function of the same name), typed and
    folded into the package.

    Returns ``(base_matching, gap_matching, gap_detector)``:

    - ``base_matching``: the ordinary Stim/PyMatching decoder built directly
      from ``dem``.
    - ``gap_matching``: the same graph, except virtual-boundary edges carrying
      ``logical_id`` are replaced by edges to one new ordinary detector node
      (``gap_detector = base_matching.num_nodes``). Decoding ``gap_matching``
      with the new detector's syndrome bit pinned to 0 or 1 forces the
      decoder into the corresponding logical sector, and the returned
      min-weight correction's weight is the cost of the best correction in
      that sector.

    Raises:
        ValueError: if no virtual-boundary edge carries ``logical_id`` (the
            logical boundary was not found), or if any non-boundary
            (detector-detector) edge carries ``logical_id`` (the simple
            boundary-edge gap construction does not apply without further
            graph surgery).
    """
    base_matching = pymatching.Matching.from_detector_error_model(dem)

    gap_detector = base_matching.num_nodes
    gap_matching = pymatching.Matching()

    logical_boundary_edges = 0
    bad_nonboundary_logical_edges: list[tuple[int, int | None, dict[str, Any]]] = []

    for u, v, attr in base_matching.edges():
        fault_ids = set(attr.get("fault_ids", set()))
        kwargs = _copy_edge_kwargs(attr)

        if v is None:
            # Virtual-boundary edge.
            if logical_id in fault_ids:
                gap_matching.add_edge(u, gap_detector, **kwargs)
                logical_boundary_edges += 1
            else:
                gap_matching.add_boundary_edge(u, **kwargs)
        else:
            # Ordinary detector-detector edge.
            if logical_id in fault_ids:
                bad_nonboundary_logical_edges.append((u, v, attr))
            gap_matching.add_edge(u, v, **kwargs)

    gap_matching.ensure_num_fault_ids(base_matching.num_fault_ids)

    if logical_boundary_edges == 0:
        raise ValueError(
            f"No virtual-boundary edges carried logical fault id {logical_id}. "
            "This construction did not find the logical boundary."
        )
    if bad_nonboundary_logical_edges:
        raise ValueError(
            f"Found {len(bad_nonboundary_logical_edges)} non-boundary edges "
            f"carrying logical fault id {logical_id}. The simple boundary-edge "
            "construction is not valid for this DEM without further graph surgery."
        )

    return base_matching, gap_matching, gap_detector


# ---------------------------------------------------------------------------
# GapOracle
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class GapOracle:
    """Computes the signed complementary-gap score for a mechanism set.

    Attributes:
        catalog: The mechanism catalog shared with ``oracle``.
        gap_matching: The gap-detector decoder from `make_gap_matching_from_vanilla_dem`.
        gap_detector: Index of the explicit gap-detector node in ``gap_matching``.
        oracle: The ordinary decode oracle (built from the same DEM's base matching).
        decode_count: Running count of PyMatching `decode` calls made through
            this instance (2 per `gap` call, 1 per `fails` call); reset by
            constructing a fresh instance.
    """

    catalog: MechanismCatalog
    gap_matching: pymatching.Matching
    gap_detector: int
    oracle: FailureOracle
    decode_count: int = 0

    @classmethod
    def from_dem(cls, dem: stim.DetectorErrorModel, catalog: MechanismCatalog) -> GapOracle:
        """Build a GapOracle from a flattened, decomposed DEM and its catalog."""
        if catalog.num_observables != 1:
            raise ValueError(
                "GapOracle requires exactly one logical observable; "
                f"catalog has num_observables={catalog.num_observables}."
            )
        base_matching, gap_matching, gap_detector = make_gap_matching_from_vanilla_dem(dem, logical_id=0)
        oracle = FailureOracle(catalog, base_matching)
        return cls(catalog=catalog, gap_matching=gap_matching, gap_detector=gap_detector, oracle=oracle)

    def gap(self, active: set[int]) -> float:
        """Signed gap G(E) = w_{1-t} - w_t (wrong-class weight minus true-class weight).

        MWPM picks the lighter class, so G < 0 implies `oracle.fails(active)`
        and G > 0 implies not-fails; G == 0 is a tie broken by implementation
        detail. Costs 2 decodes; nothing is cached across calls.
        """
        det, obs = self.oracle.syndrome_and_observable(active)
        true_class = int(obs[0])
        extended = np.zeros(self.gap_matching.num_detectors, dtype=np.uint8)
        extended[: len(det)] = det

        s_true = extended.copy()
        s_true[self.gap_detector] = true_class
        s_wrong = extended.copy()
        s_wrong[self.gap_detector] = 1 - true_class

        _, w_true = self.gap_matching.decode(s_true, return_weight=True)
        _, w_wrong = self.gap_matching.decode(s_wrong, return_weight=True)
        self.decode_count += 2
        return float(w_wrong) - float(w_true)

    def fails(self, active: set[int]) -> bool:
        """The true failure event, resolved with the ordinary decoder (never with sign(G))."""
        self.decode_count += 1
        return self.oracle.fails(active)


class _GapLike(Protocol):
    """Structural type for the two GapOracle methods the MCMC kernel needs."""

    def gap(self, active: set[int]) -> float: ...
    def fails(self, active: set[int]) -> bool: ...


# ---------------------------------------------------------------------------
# Weight-preserving MCMC kernel
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class KernelStats:
    """Acceptance bookkeeping for the two mixture components of the swap kernel."""

    global_attempts: int = 0
    global_accepts: int = 0
    local_attempts: int = 0
    local_accepts: int = 0

    @property
    def global_acceptance_rate(self) -> float:
        return self.global_accepts / self.global_attempts if self.global_attempts else float("nan")

    @property
    def local_acceptance_rate(self) -> float:
        return self.local_accepts / self.local_attempts if self.local_attempts else float("nan")

    def as_dict(self) -> dict[str, float | int]:
        return {
            "global_attempts": self.global_attempts,
            "global_accepts": self.global_accepts,
            "global_acceptance_rate": self.global_acceptance_rate,
            "local_attempts": self.local_attempts,
            "local_accepts": self.local_accepts,
            "local_acceptance_rate": self.local_acceptance_rate,
        }


class WeightPreservingSwapKernel:
    """Metropolis kernel targeting pi_w(E) proportional to prod_{i in E} odds_i on |E| = w,
    optionally restricted to a level set {E : G(E) <= threshold}.

    Mixture of two moves, chosen independently each step:

    - with probability ``1 - local_prob``, a *global swap*: pick ``i``
      uniformly from ``E`` and ``j`` uniformly from the complement restricted
      to mechanisms with nonzero probability. The proposal is symmetric
      (``1 / (w * (n - w))`` both ways, ``n`` = number of nonzero-probability
      mechanisms), so the Metropolis acceptance is ``min(1, odds_j / odds_i)``.
    - with probability ``local_prob``, the *detector-adjacent swap* of
      ``splitting_swap.SwapConditionalFailureMCMC._swap_step``: pick ``i``
      uniformly from ``E`` and ``j`` uniformly from ``neighbors[i] \\ E``;
      Hastings-corrected acceptance
      ``min(1, (odds_j / odds_i) * |C(i, E)| / |C(j, E')|)`` where
      ``C(i, E) = neighbors[i] \\ E``. Rejects if ``C(i, E)`` is empty.

    Both components individually leave pi_w invariant on ``|E| = w`` (each is
    a reversible swap that preserves cardinality), so their mixture does too.
    Conditioning on a level set is layered on top as an extra rejection: the
    (cheap) Metropolis ratio test is applied first, and only if it passes is
    the proposal's gap evaluated and rejected if it exceeds ``threshold``.
    Passing ``threshold=math.inf`` disables the level constraint entirely
    (samples pi_w unconditioned on any set membership); passing
    ``threshold=None`` conditions on `oracle.fails` instead of a gap
    threshold (the "final level" indicator).
    """

    def __init__(
        self,
        gap_oracle: _GapLike | None,
        probabilities: np.ndarray,
        neighbors: Sequence[tuple[int, ...]],
        rng: random.Random,
        local_prob: float = 0.5,
    ) -> None:
        self.gap_oracle = gap_oracle
        p = np.asarray(probabilities, dtype=np.float64)
        if np.any(p < 0) or np.any(p >= 1):
            raise ValueError("All mechanism probabilities must be in [0, 1).")
        if len(neighbors) != len(p):
            raise ValueError(f"neighbors has {len(neighbors)} entries but probabilities has {len(p)}.")
        if not (0.0 <= local_prob <= 1.0):
            raise ValueError(f"local_prob must be in [0, 1]; got {local_prob}.")

        self.odds = np.divide(p, 1.0 - p, out=np.zeros_like(p), where=p > 0)
        self.nonzero: list[int] = np.flatnonzero(p > 0).tolist()
        self.neighbors = neighbors
        self.rng = rng
        self.local_prob = local_prob
        self.stats = KernelStats()

    def _propose_global(self, active: set[int]) -> tuple[set[int], float] | None:
        if not active:
            return None
        complement = [j for j in self.nonzero if j not in active]
        if not complement:
            return None
        i = self.rng.choice(sorted(active))
        j = self.rng.choice(complement)
        proposed = set(active)
        proposed.discard(i)
        proposed.add(j)
        ratio = self.odds[j] / self.odds[i]
        return proposed, min(1.0, ratio)

    def _propose_local(self, active: set[int]) -> tuple[set[int], float] | None:
        if not active:
            return None
        i = self.rng.choice(sorted(active))
        candidates_i = [m for m in self.neighbors[i] if m not in active]
        if not candidates_i:
            return None
        j = self.rng.choice(candidates_i)

        proposed = set(active)
        proposed.discard(i)
        proposed.add(j)

        candidates_j = [m for m in self.neighbors[j] if m not in proposed]
        # i in neighbors[j] (adjacency is symmetric) and i not in proposed
        # (just removed), so i in candidates_j: the reverse move always exists.
        assert i in candidates_j, "adjacency symmetry invariant violated"

        ratio = (self.odds[j] / self.odds[i]) * (len(candidates_i) / len(candidates_j))
        return proposed, min(1.0, ratio)

    def step(
        self,
        active: set[int],
        current_gap: float | None,
        *,
        threshold: float | None,
    ) -> tuple[set[int], float | None, bool]:
        """One mixture step. Returns (new_active, new_gap, accepted).

        ``new_gap`` is the gap of the returned state when it is known
        (``threshold`` finite and the move was accepted), else it is passed
        through unchanged (rejected moves) or ``None`` (unconstrained /
        fails-conditioned moves, where the gap is never computed).
        """
        is_local = self.rng.random() < self.local_prob
        result = self._propose_local(active) if is_local else self._propose_global(active)
        if result is None:
            return active, current_gap, False
        proposed, accept_prob = result

        if is_local:
            self.stats.local_attempts += 1
        else:
            self.stats.global_attempts += 1

        if self.rng.random() >= accept_prob:
            return active, current_gap, False

        new_gap: float | None
        if threshold is None:
            assert self.gap_oracle is not None
            conditioning_ok = self.gap_oracle.fails(proposed)
            new_gap = None
        elif math.isinf(threshold):
            conditioning_ok = True
            new_gap = None
        else:
            assert self.gap_oracle is not None
            new_gap = self.gap_oracle.gap(proposed)
            conditioning_ok = new_gap <= threshold

        if not conditioning_ok:
            return active, current_gap, False

        if is_local:
            self.stats.local_accepts += 1
        else:
            self.stats.global_accepts += 1
        return proposed, new_gap, True


# ---------------------------------------------------------------------------
# Subset-simulation estimator
# ---------------------------------------------------------------------------


def _sample_fixed_weight_particles(
    q_tilted: np.ndarray,
    weight: int,
    num_particles: int,
    rng: np.random.Generator,
    batch_size: int = 4096,
) -> list[set[int]]:
    """Draw i.i.d. sets from P(E | |E| = weight) by tilting + rejection (batched)."""
    n = len(q_tilted)
    particles: list[set[int]] = []
    while len(particles) < num_particles:
        draws = rng.random((batch_size, n)) < q_tilted
        counts = draws.sum(axis=1)
        for row in np.flatnonzero(counts == weight):
            particles.append(set(np.flatnonzero(draws[row]).tolist()))
            if len(particles) >= num_particles:
                break
    return particles


def _next_threshold(gs: Sequence[float], quantile: float, prev_threshold: float) -> float | None:
    """Pick the next (strictly lower) level threshold as an order statistic of ``gs``.

    Returns the smallest value ``g`` in ``gs`` such that at least a
    ``quantile`` fraction of ``gs`` is ``<= g``, nudged down to the largest
    value of ``gs`` strictly below ``prev_threshold`` if that quantile is not
    itself strictly below ``prev_threshold``. Returns None if no value of
    ``gs`` lies strictly below ``prev_threshold`` (the population has
    stalled at the previous threshold and no further progress is possible).
    """
    sorted_gs = sorted(gs)
    k = max(1, math.ceil(quantile * len(sorted_gs)))
    candidate = sorted_gs[k - 1]
    if candidate < prev_threshold:
        return candidate
    below = [g for g in sorted_gs if g < prev_threshold]
    if not below:
        return None
    return max(below)


def estimate_f_w_gap_splitting(
    error_model: ErrorModel,
    oracle: ForwardSimulator,
    catalog: MechanismCatalog,
    dem_or_gap_oracle: stim.DetectorErrorModel | GapOracle,
    weight: int,
    *,
    p_ref: float,
    num_particles: int = 500,
    quantile: float = 0.25,
    mcmc_steps_per_particle: int = 30,
    max_levels: int = 30,
    repeats: int = 3,
    seed: int = 1,
    local_prob: float = 0.5,
    harvest_states: int = 0,
    verbose: bool = False,
) -> WeightPoint:
    """Estimate f(weight) = P(fail | W = weight) by fixed-weight gap-splitting.

    Args:
        error_model: Provides mechanism probabilities q_i(p); `probabilities(p_ref)`
            defines both the sampling distribution and the target pi_w.
        oracle: Resolves the true failure event `oracle.fails(E)` used to
            measure the failure fraction of the population at each level.
        catalog: The mechanism catalog (for detector adjacency).
        dem_or_gap_oracle: Either a flattened, decomposed DEM (a `GapOracle`
            is built from it via `GapOracle.from_dem`) or an already-built
            `GapOracle` to reuse across multiple weights/calls.
        weight: The fixed fault-set weight w.
        p_ref: Reference physical rate defining q_i(p_ref).
        num_particles: Subset-simulation population size.
        quantile: Target survival fraction per level (0 < quantile < 1).
        mcmc_steps_per_particle: Rejuvenation MCMC steps applied to each
            resampled particle after every level.
        max_levels: Hard cap on the number of levels.
        repeats: Independent repeats (different seeds), combined in log space.
        seed: Base seed; repeat r uses seed + r.
        local_prob: Probability of the detector-adjacent swap move (vs. the
            global swap) in the mixture kernel.
        harvest_states: When > 0, retain up to this many distinct *failing*
            weight-w states encountered during the run and return them in
            ``meta["failing_states"]`` (as sorted index lists). These are the
            light, near-onset malignant sets the p-schedule splitting chain
            cannot reach by local moves — pass them as ``seed_states`` to
            `splitting_local.local_splitting_estimate` to run additional
            chains from those basins.
        verbose: Print per-level progress.

    Returns:
        A `WeightPoint` with `method="gap_splitting"`, `kind="f_w"`.
    """
    if weight < 1:
        raise ValueError(f"weight must be at least 1; got {weight}.")
    if not (0.0 < quantile < 1.0):
        raise ValueError(f"quantile must be in (0, 1); got {quantile}.")
    if num_particles < 2:
        raise ValueError(f"num_particles must be at least 2; got {num_particles}.")
    if max_levels < 1:
        raise ValueError(f"max_levels must be at least 1; got {max_levels}.")
    if repeats < 1:
        raise ValueError(f"repeats must be at least 1; got {repeats}.")

    if isinstance(dem_or_gap_oracle, GapOracle):
        gap_oracle = dem_or_gap_oracle
    else:
        gap_oracle = GapOracle.from_dem(dem_or_gap_oracle, catalog)

    q_ref = np.asarray(error_model.probabilities(p_ref), dtype=np.float64)
    neighbors = build_detector_adjacency(catalog)
    decode_count_start = gap_oracle.decode_count

    per_repeat_meta: list[dict[str, Any]] = []
    log_f_hats: list[float] = []
    total_kernel_stats = KernelStats()
    harvested: set[tuple[int, ...]] = set()

    def collect_failures(population: list[set[int]], flags: list[bool]) -> None:
        if harvest_states <= 0:
            return
        for state, flag in zip(population, flags):
            if len(harvested) >= harvest_states:
                return
            if flag:
                harvested.add(tuple(sorted(state)))

    for r in range(repeats):
        rng_np = np.random.default_rng(seed + r)
        rng_py = random.Random(seed + r)
        kernel = WeightPreservingSwapKernel(gap_oracle, q_ref, neighbors, rng_py, local_prob=local_prob)

        q_tilted = tilted_probabilities(q_ref, weight)
        particles = _sample_fixed_weight_particles(q_tilted, weight, num_particles, rng_np)
        # threshold is always a concrete float within this driver's rejuvenation loop
        # (never None/inf), so kernel.step always returns a known float gap here.
        gs: list[float] = [gap_oracle.gap(e) for e in particles]

        level_factors: list[float] = []
        level_thresholds: list[float] = []
        prev_threshold = math.inf
        final_factor = 0.0
        converged = False
        stalled = False
        levels_used = 0

        for level in range(max_levels):
            fail_flags = [oracle.fails(e) for e in particles]
            collect_failures(particles, fail_flags)
            frac_fail = sum(fail_flags) / len(particles)
            if frac_fail >= quantile:
                final_factor = frac_fail
                converged = True
                levels_used = level
                break

            g_target = _next_threshold(gs, quantile, prev_threshold)
            if g_target is None:
                final_factor = frac_fail
                converged = frac_fail > 0
                stalled = True
                levels_used = level
                break

            survivor_idx = [i for i, g in enumerate(gs) if g <= g_target]
            level_factor = len(survivor_idx) / len(particles)
            level_factors.append(level_factor)
            level_thresholds.append(g_target)

            chosen = rng_np.integers(0, len(survivor_idx), size=num_particles)
            new_particles: list[set[int]] = []
            new_gs: list[float] = []
            for c in chosen.tolist():
                idx = survivor_idx[c]
                new_particles.append(particles[idx])
                new_gs.append(gs[idx])

            rejuvenated: list[set[int]] = []
            rejuvenated_gs: list[float] = []
            for state, g0 in zip(new_particles, new_gs):
                cur, cur_g = state, g0
                for _ in range(mcmc_steps_per_particle):
                    cur, new_g, _accepted = kernel.step(cur, cur_g, threshold=g_target)
                    assert new_g is not None
                    cur_g = new_g
                rejuvenated.append(cur)
                rejuvenated_gs.append(cur_g)

            particles = rejuvenated
            gs = rejuvenated_gs
            prev_threshold = g_target
            levels_used = level + 1

            if verbose:
                print(
                    f"  repeat {r} level {level}: g<={g_target:.6g} level_factor={level_factor:.4g} "
                    f"frac_fail={frac_fail:.4g}"
                )
        else:
            fail_flags = [oracle.fails(e) for e in particles]
            collect_failures(particles, fail_flags)
            final_factor = sum(fail_flags) / len(particles)
            converged = final_factor > 0
            levels_used = max_levels

        f_hat_r = float(np.prod(level_factors) * final_factor) if level_factors else float(final_factor)
        if f_hat_r > 0:
            log_f_hats.append(math.log(f_hat_r))

        per_repeat_meta.append(
            {
                "seed": seed + r,
                "f_hat": f_hat_r,
                "level_thresholds": level_thresholds,
                "level_factors": level_factors,
                "num_levels": levels_used,
                "final_factor": final_factor,
                "converged": converged,
                "stalled": stalled,
            }
        )
        if verbose:
            print(f"repeat {r}: f_hat={f_hat_r:.6g} levels={levels_used} converged={converged}")

        total_kernel_stats.global_attempts += kernel.stats.global_attempts
        total_kernel_stats.global_accepts += kernel.stats.global_accepts
        total_kernel_stats.local_attempts += kernel.stats.local_attempts
        total_kernel_stats.local_accepts += kernel.stats.local_accepts

    if log_f_hats:
        estimate = math.exp(float(np.mean(log_f_hats)))
        rel_err = (
            float(np.std(log_f_hats, ddof=1) / math.sqrt(len(log_f_hats))) if len(log_f_hats) > 1 else float("nan")
        )
    else:
        estimate = 0.0
        rel_err = float("nan")

    meta: dict[str, Any] = {
        "p_ref": p_ref,
        "num_particles": num_particles,
        "quantile": quantile,
        "mcmc_steps_per_particle": mcmc_steps_per_particle,
        "max_levels": max_levels,
        "local_prob": local_prob,
        "repeats": per_repeat_meta,
        "num_repeats_with_failure": len(log_f_hats),
        "converged": all(bool(m["converged"]) for m in per_repeat_meta),
        "kernel_acceptance": total_kernel_stats.as_dict(),
        "total_decode_count": gap_oracle.decode_count - decode_count_start,
    }
    if harvest_states > 0:
        meta["failing_states"] = [list(t) for t in sorted(harvested)]

    return WeightPoint(
        method="gap_splitting",
        kind="f_w",
        weight=weight,
        estimate=estimate,
        rel_err=rel_err,
        exact=False,
        lower_bound=False,
        meta=meta,
    )


class GapSplittingEstimator(Estimator):
    """Estimator implementing fixed-weight gap-splitting subset simulation for f(w)."""

    def estimate(
        self,
        error_model: ErrorModel,
        simulator: ForwardSimulator,
        **kwargs: Any,
    ) -> dict[str, Any]:
        if "weights" not in kwargs:
            raise ValueError("weights must be provided to GapSplittingEstimator")
        if "catalog" not in kwargs:
            raise ValueError("catalog must be provided to GapSplittingEstimator")
        if "p_ref" not in kwargs:
            raise ValueError("p_ref must be provided to GapSplittingEstimator")
        if "dem" not in kwargs and "gap_oracle" not in kwargs:
            raise ValueError("dem or gap_oracle must be provided to GapSplittingEstimator")

        catalog: MechanismCatalog = kwargs["catalog"]
        p_ref: float = kwargs["p_ref"]
        weights: Sequence[int] = kwargs["weights"]

        source = kwargs["gap_oracle"] if "gap_oracle" in kwargs else kwargs["dem"]
        gap_oracle = source if isinstance(source, GapOracle) else GapOracle.from_dem(source, catalog)

        seed = kwargs.get("seed", 1)
        weight_points = [
            estimate_f_w_gap_splitting(
                error_model=error_model,
                oracle=simulator,
                catalog=catalog,
                dem_or_gap_oracle=gap_oracle,
                weight=int(w),
                p_ref=p_ref,
                num_particles=kwargs.get("num_particles", 500),
                quantile=kwargs.get("quantile", 0.25),
                mcmc_steps_per_particle=kwargs.get("mcmc_steps_per_particle", 30),
                max_levels=kwargs.get("max_levels", 30),
                repeats=kwargs.get("repeats", 3),
                seed=seed,
                local_prob=kwargs.get("local_prob", 0.5),
                harvest_states=kwargs.get("harvest_states", 0),
                verbose=kwargs.get("verbose", False),
            )
            for w in weights
        ]
        return {"weight_points": weight_points, "p_ref": p_ref, "gap_oracle": gap_oracle}

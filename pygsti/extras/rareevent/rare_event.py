"""
Rare-event splitting estimator for QEC logical failure rates using Stim + PyMatching.

This implementation treats the natural MCMC state as a subset of independent
Stim detector-error-model (DEM) mechanisms. Equivalently, these are decoder
hyperedges carrying detector targets, optional logical-observable targets, and a
Bernoulli probability. This is usually the right abstraction layer for Stim-based
circuit-level Pauli simulations: the decoder only sees the detector syndrome, and
logical failure is determined by comparing the decoder prediction against the true
logical flips induced by the active DEM mechanisms.

Intuition
---------
Let E be a set of independent DEM mechanisms. Each mechanism flips a set of
detectors and optionally flips one or more logical observables. A decoder fails on
E when decoded_observables(syndrome(E)) != true_observables(E).

At physical error parameter p, the probability of a DEM-mechanism set E is

    P_p(E) = prod_{i in E} q_i(p) prod_{i not in E} (1 - q_i(p)).

We estimate a very small failure probability at p_final by:

    P_fail(p_final) = P_fail(p0) * prod_k E_{E~P(.|fail,p_k)}[ P_{p_{k+1}}(E) / P_{p_k}(E) ].

The conditional expectations are estimated by Metropolis MCMC over failing DEM
mechanism sets. Proposed moves add/remove one DEM mechanism and reject any
proposed state that is not a logical failure.

Representation caveat
---------------------
The MCMC target is only as faithful as the DEM representation. This code assumes
that the DEM mechanisms are independent Bernoulli events with correct probabilities.
Do not run the chain on a lossy decoder graph that has discarded logical targets,
hyperedge structure, multiplicities, or correlations. If the noise model includes
hidden source-level state not representable as detector/logical flips -- for
example leakage, non-Pauli dynamics, or postselection constraints -- the state
space must be enlarged accordingly.

Dependencies
------------
    pip install stim pymatching numpy

Example
-------
    python rare-event.py --distance 5 --rounds 10 \
        --p0 0.003 --p-final 1e-4 --levels 8 --mc-shots 10000 \
        --total-steps 50000 --burn-in-fraction 0.1
"""

from __future__ import annotations

import dataclasses
import math
import random
import sys
from collections.abc import Sequence
from typing import Any

import numpy as np
import stim

try:
    import pymatching
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Install PyMatching with: pip install pymatching") from exc


from .interfaces import ErrorModel, Estimator, ForwardSimulator


@dataclasses.dataclass(frozen=True)
class ErrorMechanism:
    """One independent DEM error mechanism.

    Attributes:
        detectors: Detector indices flipped by this mechanism.
        observables: Logical observable indices flipped by this mechanism.
        p_ref: The probability reported by the DEM at the reference physical rate.
        multiplicity: Number of identical DEM mechanisms merged into this entry.
            By default this implementation expands multiplicity=1 mechanisms. If you
            use deduplication, multiplicity can be used for diagnostics, but the MCMC
            state should usually contain expanded independent Bernoulli mechanisms.
    """

    detectors: tuple[int, ...]
    observables: tuple[int, ...]
    p_ref: float
    multiplicity: int = 1

    def __str__(self) -> str:
        targets = []
        for d in self.detectors:
            targets.append(f"D{d}")
        for o in self.observables:
            targets.append(f"L{o}")
        target_str = " ".join(targets)
        return f"error({self.p_ref}) {target_str}"

    def __repr__(self) -> str:
        return self.__str__()


@dataclasses.dataclass
class MechanismCatalog:
    """Compiled mechanism catalog used by the rare-event sampler."""

    mechanisms: list[ErrorMechanism]
    num_detectors: int
    num_observables: int

    def __str__(self) -> str:
        s = f"MechanismCatalog(num_detectors={self.num_detectors}, num_observables={self.num_observables}, mechanisms={len(self.mechanisms)}):\n"
        limit = min(5, len(self.mechanisms))
        for i in range(limit):
            s += f"  [{i}]: {self.mechanisms[i]}\n"
        if len(self.mechanisms) > limit:
            s += f"  ... ({len(self.mechanisms) - limit} more mechanisms)"
        return s.strip()

    def __repr__(self) -> str:
        return self.__str__()

    @staticmethod
    def from_detector_error_model(dem: stim.DetectorErrorModel) -> MechanismCatalog:
        """Build a mechanism catalog from a flattened Stim detector error model.

        The DEM should usually be created with:

            circuit.detector_error_model(decompose_errors=True, flatten_loops=True)

        This parser intentionally keeps each ERROR instruction as a separate
        independent Bernoulli mechanism. It handles separator targets by splitting
        decomposed DEM errors into independent components.
        """

        dem = dem.flattened()
        mechanisms: list[ErrorMechanism] = []
        num_detectors = dem.num_detectors
        num_observables = dem.num_observables

        def add_from_instruction(inst: stim.DemInstruction) -> None:
            if inst.type != "error":
                return
            p = float(inst.args_copy()[0])
            if p <= 0:
                return
            if p >= 1:
                raise ValueError(f"DEM mechanism has probability >= 1: {p}")

            dets: list[int] = []
            obs: list[int] = []

            # Stim uses separator targets for decomposed errors, e.g.
            # error(p) D0 D1 ^ D2 L0
            # Treat each component as a separate independent mechanism with the
            # same probability. This mirrors Stim's decomposition convention.
            def flush_component() -> None:
                nonlocal dets, obs
                if dets or obs:
                    mechanisms.append(
                        ErrorMechanism(
                            detectors=tuple(sorted(set(dets))),
                            observables=tuple(sorted(set(obs))),
                            p_ref=p,
                        )
                    )
                dets = []
                obs = []

            for t in inst.targets_copy():
                if t.is_separator():
                    flush_component()
                elif t.is_relative_detector_id():
                    dets.append(t.val)
                elif t.is_logical_observable_id():
                    obs.append(t.val)
                else:
                    raise ValueError(f"Unsupported DEM target in error instruction: {t!r}")
            flush_component()

        for inst in dem:
            add_from_instruction(inst)
        if not mechanisms:
            raise ValueError("No ERROR mechanisms found in detector error model.")

        return MechanismCatalog(
            mechanisms=mechanisms,
            num_detectors=num_detectors,
            num_observables=num_observables,
        )

    def probabilities_scaled_from_reference(self, p: float, p_ref: float) -> np.ndarray:
        """Scale DEM mechanism probabilities approximately linearly with physical p.

        This is appropriate when each DEM mechanism is effectively first order in a
        common physical error parameter. For production estimates, especially across
        large changes in p or for heterogeneous error models, prefer supplying exact
        q_i(p) values extracted from DEMs generated at each splitting level.
        """

        scale = p / p_ref
        probs = np.array([m.p_ref * scale for m in self.mechanisms], dtype=np.float64)
        if np.any(probs >= 1):
            raise ValueError("Scaled mechanism probability reached >= 1. Use smaller p or exact p_i(p).")
        return probs


@dataclasses.dataclass
class ScaledMechanismErrorModel:
    """An ErrorModel that linearly scales mechanism probabilities from a reference p."""

    catalog: MechanismCatalog
    p_ref: float

    def probabilities(self, p: float) -> np.ndarray:
        return self.catalog.probabilities_scaled_from_reference(p, self.p_ref)

    def __str__(self) -> str:
        return f"ScaledMechanismErrorModel(p_ref={self.p_ref})\nCatalog: {self.catalog}"
        
    def __repr__(self) -> str:
        return self.__str__()

class FailureOracle:
    """Computes whether a given set of DEM mechanisms causes a logical failure."""

    def __init__(self, catalog: MechanismCatalog, matching: pymatching.Matching):
        self.catalog = catalog
        self.matching = matching

    def syndrome_and_observable(self, active: set[int]) -> tuple[np.ndarray, np.ndarray]:
        det = np.zeros(self.catalog.num_detectors, dtype=np.uint8)
        obs = np.zeros(self.catalog.num_observables, dtype=np.uint8)
        for i in active:
            mech = self.catalog.mechanisms[i]
            if mech.detectors:
                det[list(mech.detectors)] ^= 1
            if mech.observables:
                obs[list(mech.observables)] ^= 1
        return det, obs

    def fails(self, active: set[int]) -> bool:
        det, true_obs = self.syndrome_and_observable(active)
        pred = np.asarray(self.matching.decode(det), dtype=np.uint8)
        if pred.shape == ():
            pred = pred.reshape(1)
        return bool(np.any(pred ^ true_obs))


class ConditionalFailureMCMC:
    """Metropolis chain targeting P_p(DEM mechanism set E | decoder failure)."""

    def __init__(
        self,
        oracle: ForwardSimulator,
        probabilities: np.ndarray,
        rng: random.Random | None = None,
    ):
        self.oracle = oracle
        self.p = np.asarray(probabilities, dtype=np.float64)
        if np.any(self.p <= 0) or np.any(self.p >= 1):
            raise ValueError("All mechanism probabilities must be in (0, 1).")
        self.odds = self.p / (1 - self.p)
        self.rng = rng or random.Random()

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

    def step(self, active: set[int]) -> tuple[set[int], bool]:
        """One Metropolis step using a single DEM-mechanism toggle proposal.

        Proposal is symmetric: choose mechanism i uniformly and toggle membership.
        The unconditioned probability ratio is odds_i for add, 1/odds_i for remove.
        The conditioned chain rejects proposals that are not logical failures.
        """
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

        return proposed, True

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


def log_probability_of_state(active: set[int], probs: np.ndarray) -> float:
    """log P_p(E) for independent Bernoulli mechanisms."""
    active_mask = np.zeros(len(probs), dtype=bool)
    if active:
        active_mask[list(active)] = True
    return float(np.sum(np.log(probs[active_mask])) + np.sum(np.log1p(-probs[~active_mask])))


def log_weight_ratio(active: set[int], probs_next: np.ndarray, probs_current: np.ndarray) -> float:
    return log_probability_of_state(active, probs_next) - log_probability_of_state(active, probs_current)


def logmeanexp(values: Sequence[float]) -> tuple[float, float]:
    """Return log(mean(exp(values))) and a naive standard error in log space."""
    x = np.asarray(values, dtype=np.float64)
    m = float(np.max(x))
    y = np.exp(x - m)
    mean = float(np.mean(y))
    if len(y) > 1:
        se = float(np.std(y, ddof=1) / math.sqrt(len(y)))
    else:
        se = float("nan")
    log_mean = m + math.log(mean)
    log_se = se / mean if mean > 0 and math.isfinite(se) else float("nan")
    return log_mean, log_se


def split_rhat(chains: Sequence[Sequence[float]]) -> float | None:
    """Compute the standard split-chain Gelman--Rubin R-hat statistic.

    This is a generic scalar-chain diagnostic. Here it is useful for flagging
    obviously poor mixing in quantities such as log weight ratios or active-set
    weights. It is not a proof of convergence, and it can miss multimodality.

    Returns None if there are fewer than two usable split chains.
    """
    split_chains: list[np.ndarray] = []
    min_len = min((len(c) for c in chains), default=0)
    if min_len < 4:
        return None

    # Use equal-length chains so between- and within-chain variances are comparable.
    n_even = min_len if min_len % 2 == 0 else min_len - 1
    if n_even < 4:
        return None

    for c in chains:
        x = np.asarray(c[:n_even], dtype=np.float64)
        half = n_even // 2
        split_chains.append(x[:half])
        split_chains.append(x[half:])

    m = len(split_chains)
    n = len(split_chains[0])
    if m < 2 or n < 2:
        return None

    means = np.array([np.mean(c) for c in split_chains], dtype=np.float64)
    variances = np.array([np.var(c, ddof=1) for c in split_chains], dtype=np.float64)
    W = float(np.mean(variances))
    if W == 0:
        return 1.0 if float(np.var(means)) == 0 else float("inf")
    B = float(n * np.var(means, ddof=1))
    var_hat = ((n - 1) / n) * W + B / n
    return math.sqrt(max(var_hat / W, 0.0))


def summarize_chains_for_level(
    *,
    level: int,
    p_current: float,
    p_next: float,
    samples_by_chain: Sequence[Sequence[set[int]]],
    acceptance_rates: Sequence[float],
    probs_next: np.ndarray,
    probs_current: np.ndarray,
    pooled_log_ratio: float,
) -> LevelDiagnostics:
    """Compute per-chain ratio estimates and convergence diagnostics."""
    per_chain_log_ratio_samples: list[list[float]] = []
    per_chain_weight_samples: list[list[float]] = []
    per_chain_log_ratios: list[float] = []
    per_chain_mean_weights: list[float] = []
    per_chain_sample_sizes: list[int] = []

    for samples in samples_by_chain:
        log_ratios = [log_weight_ratio(s, probs_next, probs_current) for s in samples]
        weights = [float(len(s)) for s in samples]
        per_chain_log_ratio_samples.append(log_ratios)
        per_chain_weight_samples.append(weights)
        per_chain_sample_sizes.append(len(samples))
        per_chain_mean_weights.append(float(np.mean(weights)) if weights else float("nan"))
        if log_ratios:
            per_chain_log_ratios.append(logmeanexp(log_ratios)[0])
        else:
            per_chain_log_ratios.append(float("nan"))

    return LevelDiagnostics(
        level=level,
        p_current=p_current,
        p_next=p_next,
        pooled_log_ratio=pooled_log_ratio,
        per_chain_log_ratios=per_chain_log_ratios,
        per_chain_acceptance_rates=list(map(float, acceptance_rates)),
        per_chain_sample_sizes=per_chain_sample_sizes,
        per_chain_mean_weights=per_chain_mean_weights,
        rhat_log_weight_ratio=split_rhat(per_chain_log_ratio_samples),
        rhat_active_weight=split_rhat(per_chain_weight_samples),
    )


def direct_monte_carlo_failure_rate(
    oracle: ForwardSimulator,
    probs: np.ndarray,
    shots: int,
) -> tuple[float, float, set[int] | None]:
    """Estimate P_fail(p) directly and return one failing seed if found."""
    failures = 0
    seed = None
    n = len(probs)
    for _ in range(shots):
        active = set(np.flatnonzero(np.random.random(n) < probs).tolist())
        if oracle.fails(active):
            failures += 1
            if seed is None:
                seed = set(active)
    phat = failures / shots
    stderr = math.sqrt(phat * (1 - phat) / shots) if shots > 0 else float("nan")
    return phat, stderr, seed


@dataclasses.dataclass
class LevelDiagnostics:
    """Per-splitting-level diagnostics from one or more MCMC chains."""

    level: int
    p_current: float
    p_next: float
    pooled_log_ratio: float
    per_chain_log_ratios: list[float]
    per_chain_acceptance_rates: list[float]
    per_chain_sample_sizes: list[int]
    per_chain_mean_weights: list[float]
    rhat_log_weight_ratio: float | None
    rhat_active_weight: float | None


@dataclasses.dataclass
class SplittingResult:
    p_scales: list[float]
    log_failure_estimates: list[float]
    failure_estimates: list[float]
    log_ratio_estimates: list[float]
    acceptance_rates: list[float]
    sample_sizes: list[int]
    level_diagnostics: list[LevelDiagnostics]


def rare_event_splitting_estimate(
    error_model: ErrorModel,
    simulator: ForwardSimulator,
    p_scales: Sequence[float],
    mc_shots_at_p0: int,
    steps_per_chain: int | None,
    total_steps_per_level: int | None,
    burn_in: int | None,
    burn_in_fraction: float | None = 0.1,
    thin: int = 1,
    seed: int = 1,
) -> SplittingResult:
    """Estimate failure rates along a descending sequence of p values."""
    
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
        chain = ConditionalFailureMCMC(simulator, probs[k], rng=rng)
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
            per_chain_weight_samples.append(sum(1 for _ in state))
        
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
            f"P_fail={math.exp(log_fail):.6e} | acc={acc:.3f} | "
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

class RareEventSplittingEstimator(Estimator):
    """Estimator implementing the rare-event splitting (Bravyi-Vargo) method."""
    
    def estimate(
        self,
        error_model: ErrorModel,
        simulator: ForwardSimulator,
        **kwargs: Any,
    ) -> SplittingResult:
        if "p_scales" not in kwargs:
            raise ValueError("p_scales must be provided to RareEventSplittingEstimator")
            
        return rare_event_splitting_estimate(
            error_model=error_model,
            simulator=simulator,
            p_scales=kwargs["p_scales"],
            mc_shots_at_p0=kwargs.get("mc_shots_at_p0", 10000),
            steps_per_chain=kwargs.get("steps_per_chain"),
            total_steps_per_level=kwargs.get("total_steps_per_level"),
            burn_in=kwargs.get("burn_in"),
            burn_in_fraction=kwargs.get("burn_in_fraction", 0.1),
            thin=kwargs.get("thin", 1),
            seed=kwargs.get("seed", 1),
        )


def make_surface_code_memory_circuit(distance: int, rounds: int, p: float, basis: str = "X") -> stim.Circuit:
    """Stim built-in rotated surface-code memory circuit with circuit-level depolarizing noise."""
    return stim.Circuit.generated(
        "surface_code:rotated_memory_" + basis.lower(),
        distance=distance,
        rounds=rounds,
        after_clifford_depolarization=p,
        after_reset_flip_probability=p,
        before_measure_flip_probability=p,
        before_round_data_depolarization=p,
    )


def make_repetition_code_memory_circuit(distance: int, rounds: int, p: float) -> stim.Circuit:
    """Stim built-in repetition-code memory circuit with circuit-level noise."""
    return stim.Circuit.generated(
        "repetition_code:memory",
        distance=distance,
        rounds=rounds,
        after_clifford_depolarization=p,
        after_reset_flip_probability=p,
        before_measure_flip_probability=p,
        before_round_data_depolarization=p,
    )


def append_global_dem_event(dem: stim.DetectorErrorModel, probability: float) -> None:
    """Append one DEM mechanism flipping every detector and logical target."""
    if probability <= 0:
        return
    if probability >= 1:
        raise ValueError(f"Global DEM event probability must be in (0, 1); got {probability}.")
    targets = [stim.DemTarget.relative_detector_id(i) for i in range(dem.num_detectors)]
    targets += [stim.DemTarget.logical_observable_id(i) for i in range(dem.num_observables)]
    dem.append("error", probability, targets, tag="global_all_bits")


def build_catalog_decoder_and_dem_text(
    circuit: stim.Circuit,
    global_dem_event_probability: float = 0.0,
) -> tuple[MechanismCatalog, FailureOracle, str]:
    dem = circuit.detector_error_model(decompose_errors=True, flatten_loops=True)
    append_global_dem_event(dem, global_dem_event_probability)
    catalog = MechanismCatalog.from_detector_error_model(dem)
    matching = pymatching.Matching.from_detector_error_model(dem)
    oracle = FailureOracle(catalog, matching)
    return catalog, oracle, str(dem)


def geometric_p_schedule(p0: float, p_final: float, levels: int) -> list[float]:
    if levels < 2:
        raise ValueError("levels must be at least 2")
    return [float(x) for x in np.geomspace(p0, p_final, levels)]

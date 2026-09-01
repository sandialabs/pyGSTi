from __future__ import annotations

import contextlib
import io
import math
import random
import unittest

import numpy as np
import pymatching

from pygsti.extras.rareevent.malignant import MalignantSetEstimator
from pygsti.extras.rareevent.noise import ExactNoiseErrorModel, SI1000NoiseModel
from pygsti.extras.rareevent.rare_event import (
    ErrorMechanism,
    FailureOracle,
    MechanismCatalog,
    direct_monte_carlo_failure_rate,
    make_repetition_code_memory_circuit,
)
from pygsti.extras.rareevent.splitting_swap import (
    SwapConditionalFailureMCMC,
    SwapSplittingEstimator,
    build_detector_adjacency,
)


class ThresholdSimulator:
    """Stub ForwardSimulator: fails iff at least two of mechanisms {0, 1, 2} are active."""

    def fails(self, active: set[int]) -> bool:
        return len(active & {0, 1, 2}) >= 2


def _toy_catalog() -> MechanismCatalog:
    # Detectors chosen so mechanisms form a 6-cycle: 0-1-2-3-4-5-0. This gives
    # a nontrivial adjacency graph (some pairs adjacent, some not) that still
    # covers all six mechanisms.
    detector_pairs = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)]
    probs = [0.3, 0.25, 0.1, 0.05, 0.02, 0.15]
    mechanisms = [
        ErrorMechanism(detectors=d, observables=(), p_ref=p) for d, p in zip(detector_pairs, probs)
    ]
    return MechanismCatalog(mechanisms=mechanisms, num_detectors=6, num_observables=0)


def _exact_conditional_distribution(probs: np.ndarray, simulator: ThresholdSimulator) -> dict[frozenset[int], float]:
    """Enumerate all 2^n subsets and return pi(E) ~ prod_{i in E} odds_i restricted to failures."""
    odds = probs / (1 - probs)
    n = len(probs)
    unnorm: dict[frozenset[int], float] = {}
    for mask in range(1 << n):
        active = frozenset(i for i in range(n) if mask & (1 << i))
        if simulator.fails(set(active)):
            weight = 1.0
            for i in active:
                weight *= odds[i]
            unnorm[active] = weight
    norm = sum(unnorm.values())
    return {state: w / norm for state, w in unnorm.items()}


class TestBuildDetectorAdjacency(unittest.TestCase):
    def test_six_cycle_adjacency(self) -> None:
        catalog = _toy_catalog()
        neighbors = build_detector_adjacency(catalog)
        expected = [
            (1, 5),
            (0, 2),
            (1, 3),
            (2, 4),
            (3, 5),
            (0, 4),
        ]
        self.assertEqual(neighbors, expected)
        # Symmetric by construction: j in neighbors[i] iff i in neighbors[j].
        for i, nbrs in enumerate(neighbors):
            for j in nbrs:
                self.assertIn(i, neighbors[j])
            self.assertNotIn(i, nbrs)


def _batched_standard_error(indicator: np.ndarray, num_batches: int) -> float:
    """Standard error of the mean of a (possibly autocorrelated) 0/1 series via batch means.

    MCMC samples are correlated, so the naive binomial standard error
    ``sqrt(p(1-p)/n)`` understates the true sampling error of the visit
    frequency. Splitting the kept steps into ``num_batches`` contiguous
    batches and treating the batch means as approximately independent
    (valid once batch size well exceeds the chain's integrated
    autocorrelation time) gives a standard error that correctly reflects
    the correlation.
    """
    batch_size = len(indicator) // num_batches
    trimmed = indicator[: batch_size * num_batches].reshape(num_batches, batch_size)
    batch_means = trimmed.mean(axis=1)
    return float(batch_means.std(ddof=1) / math.sqrt(num_batches))


class TestSwapKernelExactness(unittest.TestCase):
    def _run_exactness_check(self, swap_prob: float, steps: int, seed: int) -> None:
        catalog = _toy_catalog()
        neighbors = build_detector_adjacency(catalog)
        probs = np.array([m.p_ref for m in catalog.mechanisms])
        simulator = ThresholdSimulator()
        exact = _exact_conditional_distribution(probs, simulator)

        chain = SwapConditionalFailureMCMC(
            oracle=simulator,
            probabilities=probs,
            neighbors=neighbors,
            rng=random.Random(seed),
            swap_prob=swap_prob,
        )

        initial = {0, 1, 2}
        self.assertTrue(simulator.fails(initial))
        burn_in = steps // 10
        active = set(initial)
        kept_masks: list[int] = []
        for t in range(steps):
            active, _ = chain.step(active)
            self.assertTrue(simulator.fails(active))  # every visited state is a real failure
            if t >= burn_in:
                mask = 0
                for i in active:
                    mask |= 1 << i
                kept_masks.append(mask)

        masks = np.array(kept_masks)
        num_batches = 50
        worst = 0.0
        for state, p_exact in exact.items():
            state_mask = 0
            for i in state:
                state_mask |= 1 << i
            indicator = (masks == state_mask).astype(np.float64)
            p_emp = float(indicator.mean())
            se = _batched_standard_error(indicator, num_batches)
            tol = 5 * se + 1e-4
            diff = abs(p_emp - p_exact)
            worst = max(worst, diff / tol)
            self.assertLess(
                diff, tol, f"swap_prob={swap_prob} state={sorted(state)}: emp={p_emp:.5f} exact={p_exact:.5f}"
            )

    def test_exact_conditional_distribution_toggle_and_swap_mixed(self) -> None:
        self._run_exactness_check(swap_prob=0.5, steps=1_000_000, seed=1234)

    def test_exact_conditional_distribution_swap_dominated(self) -> None:
        self._run_exactness_check(swap_prob=0.9, steps=1_000_000, seed=5678)


class TestSwapSplittingEstimatorIntegration(unittest.TestCase):
    def test_repetition_code_against_references(self) -> None:
        np.random.seed(9)
        p0 = 0.02
        circuit = make_repetition_code_memory_circuit(distance=3, rounds=2, p=0)
        noise = SI1000NoiseModel()
        error_model = ExactNoiseErrorModel(circuit, noise, p_ref=p0)
        dem = noise(circuit, p0).detector_error_model(decompose_errors=True, flatten_loops=True)
        matching = pymatching.Matching.from_detector_error_model(dem)
        oracle = FailureOracle(error_model.catalog, matching)

        estimator = SwapSplittingEstimator()
        with contextlib.redirect_stdout(io.StringIO()):
            result = estimator.estimate(
                error_model=error_model,
                simulator=oracle,
                p_scales=[0.02, 0.008, 0.003],
                catalog=error_model.catalog,
                mc_shots_at_p0=20_000,
                total_steps_per_level=60_000,
                thin=20,
                seed=7,
            )

        self.assertEqual(len(result.failure_estimates), 3)

        # p = 0.02 (the anchor level) should agree with direct Monte Carlo.
        probs0 = error_model.probabilities(p0)
        mc_pfail, mc_stderr, _ = direct_monte_carlo_failure_rate(oracle, probs0, 20_000)
        self.assertLess(
            abs(math.log(result.failure_estimates[0]) - math.log(mc_pfail)),
            math.log(1.5),
        )

        # p = 0.003 (deepest level) should agree with the exact weight-<=3 lower bound.
        with contextlib.redirect_stdout(io.StringIO()):
            res_mal = MalignantSetEstimator().estimate(
                error_model=error_model,
                simulator=oracle,
                p_scales=[0.003],
                max_weight=3,
                num_mechanisms=error_model.num_mechanisms,
            )
        exact_low = res_mal["failure_estimates"][0]
        self.assertLess(
            abs(math.log(result.failure_estimates[-1]) - math.log(exact_low)),
            math.log(2.0),
        )


if __name__ == "__main__":
    unittest.main()

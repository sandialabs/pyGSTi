from __future__ import annotations

import itertools
import math
import unittest

import numpy as np
import pymatching

from pygsti.extras.rareevent.core_planting import (
    CountingOracle,
    _peel_to_minimal,
    core_planting_estimate_f_w,
    harvest_cores,
    log_mixture_density,
    peel_to_minimal_subset,
)
from pygsti.extras.rareevent.noise import ExactNoiseErrorModel, SI1000NoiseModel
from pygsti.extras.rareevent.rare_event import FailureOracle, make_repetition_code_memory_circuit


def _build_pipeline() -> tuple[float, ExactNoiseErrorModel, FailureOracle, np.ndarray, int, np.ndarray]:
    """d=3 repetition-code pipeline, as in tests/test_splitting_local.py::TestExternalAnchor._setup."""
    p0 = 0.02
    circuit = make_repetition_code_memory_circuit(distance=3, rounds=2, p=0)
    noise = SI1000NoiseModel()
    error_model = ExactNoiseErrorModel(circuit, noise, p_ref=p0)
    dem = noise(circuit, p0).detector_error_model(decompose_errors=True, flatten_loops=True)
    matching = pymatching.Matching.from_detector_error_model(dem)
    oracle = FailureOracle(error_model.catalog, matching)
    probs = error_model.probabilities(p0)
    n = error_model.num_mechanisms
    odds = probs / (1.0 - probs)
    return p0, error_model, oracle, probs, n, odds


def _brute_force_weight_data(
    oracle: FailureOracle, odds: np.ndarray, n: int, weights: tuple[int, ...]
) -> tuple[dict[int, float], dict[int, list[tuple[frozenset[int], float]]]]:
    """For each weight, the exact Z_w and the list of (failing set, prod-odds) pairs."""
    z_w: dict[int, float] = {}
    failing: dict[int, list[tuple[frozenset[int], float]]] = {}
    for w in weights:
        total = 0.0
        fails_w: list[tuple[frozenset[int], float]] = []
        for combo in itertools.combinations(range(n), w):
            prod_odds = math.prod(odds[i] for i in combo)
            total += prod_odds
            if oracle.fails(set(combo)):
                fails_w.append((frozenset(combo), prod_odds))
        z_w[w] = total
        failing[w] = fails_w
    return z_w, failing


def _is_minimal(oracle: FailureOracle, s: frozenset[int]) -> bool:
    for r in range(1, len(s)):
        for sub in itertools.combinations(sorted(s), r):
            if oracle.fails(set(sub)):
                return False
    return True


class TestExactRecoveryCompleteCores(unittest.TestCase):
    def test_matches_exact_f_w_both_alpha_modes(self) -> None:
        p_ref, error_model, oracle, probs, n, odds = _build_pipeline()
        z_w, failing_by_weight = _brute_force_weight_data(oracle, odds, n, (1, 2, 3))

        all_failing = [s for w in (1, 2, 3) for s, _ in failing_by_weight[w]]
        cores = [s for s in all_failing if _is_minimal(oracle, s)]
        self.assertGreater(len(cores), 0)

        for w in (2, 3):
            exact_f = sum(v for _, v in failing_by_weight[w]) / z_w[w]
            for alpha_mode in ("mass", "uniform"):
                point = core_planting_estimate_f_w(
                    error_model,
                    oracle,
                    cores,
                    weight=w,
                    p_ref=p_ref,
                    num_samples=4000,
                    alpha=alpha_mode,
                    seed=123,
                )
                se = point.estimate * point.rel_err
                tol = 5 * se + 1e-9
                self.assertLess(
                    abs(point.estimate - exact_f),
                    tol,
                    f"w={w} alpha={alpha_mode}: estimate={point.estimate} exact={exact_f} tol={tol}",
                )


class TestCertifiedLowerBoundPartialCores(unittest.TestCase):
    def test_partial_cores_recover_covered_value_and_stay_below_full(self) -> None:
        p_ref, error_model, oracle, probs, n, odds = _build_pipeline()
        z_w, failing_by_weight = _brute_force_weight_data(oracle, odds, n, (1, 2, 3))

        all_failing = [s for w in (1, 2, 3) for s, _ in failing_by_weight[w]]
        cores = sorted({s for s in all_failing if _is_minimal(oracle, s)}, key=lambda s: sorted(s))
        self.assertGreater(len(cores), 1)

        partial_cores = cores[0::2]  # deterministic half
        self.assertLess(len(partial_cores), len(cores))

        for w in (2, 3):
            exact_full = sum(v for _, v in failing_by_weight[w]) / z_w[w]
            exact_covered = sum(
                v for s, v in failing_by_weight[w] if any(c <= s for c in partial_cores)
            ) / z_w[w]

            point = core_planting_estimate_f_w(
                error_model,
                oracle,
                partial_cores,
                weight=w,
                p_ref=p_ref,
                num_samples=4000,
                alpha="mass",
                seed=321,
            )
            se = point.estimate * point.rel_err
            tol = 5 * se + 1e-9
            self.assertLess(
                abs(point.estimate - exact_covered),
                tol,
                f"w={w}: estimate={point.estimate} exact_covered={exact_covered} tol={tol}",
            )
            self.assertLessEqual(point.estimate, exact_full + tol)


class TestHarvestSanity(unittest.TestCase):
    def test_harvested_cores_fail_are_1_minimal_and_antichain(self) -> None:
        _p_ref, _error_model, oracle, probs, _n, _odds = _build_pipeline()

        cores = harvest_cores(
            oracle,
            probs,
            weights=[3, 4],
            target_failures_per_weight=20,
            max_trials_per_weight=5000,
            seed=7,
        )
        self.assertGreater(len(cores), 0)

        for c in cores:
            self.assertTrue(oracle.fails(set(c)), f"harvested core {c} does not fail")
            for i in c:
                self.assertFalse(
                    oracle.fails(set(c) - {i}),
                    f"core {c} is not 1-minimal: removing {i} still fails",
                )

        for c1 in cores:
            for c2 in cores:
                if c1 != c2:
                    self.assertFalse(c1 <= c2, f"{c1} is a subset of {c2}: not an antichain")


class TestWeightBookkeeping(unittest.TestCase):
    def test_log_mixture_density_matches_direct_sum(self) -> None:
        probs = np.array([0.1, 0.2, 0.05, 0.15, 0.3])
        odds = probs / (1.0 - probs)
        n = len(probs)
        cores = [frozenset({0, 1}), frozenset({2})]
        weight = 3

        def brute_force_zc(core: frozenset[int]) -> float:
            complement = [i for i in range(n) if i not in core]
            remaining = weight - len(core)
            return float(
                sum(math.prod(float(odds[i]) for i in combo) for combo in itertools.combinations(complement, remaining))
            )

        z_c = {c: brute_force_zc(c) for c in cores}
        unnorm = {c: math.prod(odds[i] for i in c) * z_c[c] for c in cores}
        total_unnorm = sum(unnorm.values())
        alpha = {c: unnorm[c] / total_unnorm for c in cores}

        def direct_q(active: frozenset[int]) -> float:
            total = 0.0
            for c in cores:
                if c <= active:
                    total += alpha[c] * math.prod(odds[i] for i in (active - c)) / z_c[c]
            return total

        examples = [
            frozenset({0, 1, 2}),  # contains both cores
            frozenset({0, 1, 3}),  # contains core {0, 1} only
            frozenset({2, 3, 4}),  # contains core {2} only
            frozenset({0, 3, 4}),  # contains neither core
        ]
        for active in examples:
            q_direct = direct_q(active)
            log_q = log_mixture_density(cores, probs, weight, active, alpha="mass")
            if q_direct == 0.0:
                self.assertEqual(log_q, -math.inf, f"active={active}")
            else:
                self.assertAlmostEqual(math.log(q_direct), log_q, places=8, msg=f"active={active}")


class PlantedCoreOracle:
    """Fails iff the active set contains at least one planted core (monotone, so fluff removal is always safe)."""

    def __init__(self, cores: list[set[int]]) -> None:
        self.cores = [frozenset(c) for c in cores]

    def fails(self, active: set[int]) -> bool:
        return any(c <= active for c in self.cores)


class TestPeelToMinimalSubset(unittest.TestCase):
    def test_returns_failing_1_minimal_set(self) -> None:
        oracle = PlantedCoreOracle([{0, 1}, {2, 3, 4}])
        heavy = frozenset({0, 1} | set(range(10, 30)))  # planted core + 20 fluff elements
        rng = np.random.default_rng(11)

        result = peel_to_minimal_subset(oracle, heavy, rng)

        self.assertTrue(oracle.fails(set(result)))
        for i in result:
            self.assertFalse(oracle.fails(set(result) - {i}), f"{result} is not 1-minimal at {i}")
        # For a monotone oracle, a 1-minimal failing subset of `heavy` must be a planted core.
        self.assertIn(result, oracle.cores)

    def test_raises_on_non_failing_input(self) -> None:
        oracle = PlantedCoreOracle([{0, 1}])
        with self.assertRaises(ValueError):
            peel_to_minimal_subset(oracle, {5, 6, 7}, np.random.default_rng(0))

    def test_deterministic_given_seed(self) -> None:
        oracle = PlantedCoreOracle([{0, 1}, {2, 3, 4}, {5, 6}])
        heavy = frozenset({2, 3, 4, 5, 6} | set(range(20, 50)))
        results = [
            peel_to_minimal_subset(oracle, heavy, np.random.default_rng(42), max_rounds=64, max_stall_rounds=4)
            for _ in range(2)
        ]
        self.assertEqual(results[0], results[1])

    def test_fewer_oracle_calls_than_single_element_baseline(self) -> None:
        core = {0, 1, 2}
        heavy = frozenset(core | set(range(3, 123)))  # weight-123 pattern hiding a weight-3 core

        single_oracle = CountingOracle(PlantedCoreOracle([core]))
        single_result = _peel_to_minimal(single_oracle, heavy, np.random.default_rng(5))
        # Baseline cost is deterministic: 1 validation + 123 (first pass) + 3 (no-removal pass).
        self.assertEqual(single_oracle.calls, 127)
        self.assertEqual(single_result, frozenset(core))

        subset_oracle = CountingOracle(PlantedCoreOracle([core]))
        subset_result = peel_to_minimal_subset(subset_oracle, heavy, np.random.default_rng(5))
        self.assertEqual(subset_result, frozenset(core))
        self.assertLessEqual(subset_oracle.calls, single_oracle.calls)
        self.assertLessEqual(
            subset_oracle.calls,
            int(0.7 * single_oracle.calls),
            f"subset peel used {subset_oracle.calls} calls vs baseline {single_oracle.calls}",
        )

    def test_counting_oracle_delegates_results(self) -> None:
        inner = PlantedCoreOracle([{0, 1}])
        counting = CountingOracle(inner)
        self.assertTrue(counting.fails({0, 1, 7}))
        self.assertFalse(counting.fails({0, 7}))
        self.assertEqual(counting.calls, 2)


class TestHarvestPeelMethod(unittest.TestCase):
    def _synthetic(self) -> tuple[PlantedCoreOracle, np.ndarray]:
        oracle = PlantedCoreOracle([{0, 1}, {2, 3, 4}])
        probs = np.full(12, 0.25)
        return oracle, probs

    def test_subset_peel_produces_failing_1_minimal_antichain(self) -> None:
        oracle, probs = self._synthetic()
        cores = harvest_cores(
            oracle,
            probs,
            weights=[4, 5],
            target_failures_per_weight=15,
            max_trials_per_weight=3000,
            seed=13,
            peel_method="subset",
        )
        self.assertGreater(len(cores), 0)
        for c in cores:
            self.assertTrue(oracle.fails(set(c)), f"harvested core {c} does not fail")
            for i in c:
                self.assertFalse(oracle.fails(set(c) - {i}), f"core {c} is not 1-minimal at {i}")
        for c1 in cores:
            for c2 in cores:
                if c1 != c2:
                    self.assertFalse(c1 <= c2, f"{c1} is a subset of {c2}: not an antichain")

    def test_unknown_peel_method_raises(self) -> None:
        oracle, probs = self._synthetic()
        with self.assertRaises(ValueError):
            harvest_cores(
                oracle,
                probs,
                weights=[4],
                target_failures_per_weight=1,
                max_trials_per_weight=10,
                seed=1,
                peel_method="garbage",
            )

    def test_default_matches_explicit_single(self) -> None:
        oracle, probs = self._synthetic()
        default_cores = harvest_cores(
            oracle, probs, weights=[4, 5], target_failures_per_weight=15, max_trials_per_weight=3000, seed=13
        )
        single_cores = harvest_cores(
            oracle,
            probs,
            weights=[4, 5],
            target_failures_per_weight=15,
            max_trials_per_weight=3000,
            seed=13,
            peel_method="single",
        )
        self.assertEqual(default_cores, single_cores)


if __name__ == "__main__":
    unittest.main()

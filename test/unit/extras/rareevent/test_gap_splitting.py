from __future__ import annotations

import itertools
import math
import random
import unittest

import numpy as np
import pymatching

from pygsti.extras.rareevent.failure_spectrum import sample_fixed_weight_failure_fraction
from pygsti.extras.rareevent.gap_splitting import (
    GapOracle,
    GapSplittingEstimator,
    WeightPreservingSwapKernel,
    estimate_f_w_gap_splitting,
)
from pygsti.extras.rareevent.noise import ExactNoiseErrorModel, SI1000NoiseModel
from pygsti.extras.rareevent.rare_event import (
    ErrorMechanism,
    FailureOracle,
    MechanismCatalog,
    make_repetition_code_memory_circuit,
)
from pygsti.extras.rareevent.splitting_swap import build_detector_adjacency


def _build_ring_catalog() -> MechanismCatalog:
    # A 6-mechanism ring where consecutive mechanisms share a detector (copied
    # from tests/test_splitting_local.py::_build_ring_catalog; that helper is
    # private to its module, so it is duplicated here rather than imported).
    p_refs = [0.02, 0.05, 0.08, 0.12, 0.2, 0.3]
    detector_pairs = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)]
    mechanisms = [
        ErrorMechanism(detectors=pair, observables=(), p_ref=p) for pair, p in zip(detector_pairs, p_refs)
    ]
    return MechanismCatalog(mechanisms=mechanisms, num_detectors=6, num_observables=0)


def _setup_repetition_pipeline() -> tuple[float, ExactNoiseErrorModel, FailureOracle, object]:
    # Same pipeline construction as TestExternalAnchor._setup in
    # tests/test_splitting_local.py.
    p0 = 0.02
    circuit = make_repetition_code_memory_circuit(distance=3, rounds=2, p=0)
    noise = SI1000NoiseModel()
    error_model = ExactNoiseErrorModel(circuit, noise, p_ref=p0)
    dem = noise(circuit, p0).detector_error_model(decompose_errors=True, flatten_loops=True)
    matching = pymatching.Matching.from_detector_error_model(dem)
    oracle = FailureOracle(error_model.catalog, matching)
    return p0, error_model, oracle, dem


class TestGapSignConsistency(unittest.TestCase):
    def test_gap_sign_matches_true_failure_on_repetition_code(self) -> None:
        p0, error_model, oracle, dem = _setup_repetition_pipeline()
        gap_oracle = GapOracle.from_dem(dem, error_model.catalog)
        n = error_model.num_mechanisms

        rng = random.Random(2024)
        num_checked = 0
        num_ties = 0
        for _ in range(200):
            w = rng.randint(1, 4)
            active = set(rng.sample(range(n), w))
            g = gap_oracle.gap(active)
            fails = oracle.fails(active)
            if g < 0:
                self.assertTrue(fails, f"G={g} < 0 but oracle did not fail on {active}")
            elif g > 0:
                self.assertFalse(fails, f"G={g} > 0 but oracle failed on {active}")
            else:
                num_ties += 1
            num_checked += 1

        self.assertEqual(num_checked, 200)
        # Ties (G == 0) are allowed but should be rare, not the norm.
        self.assertLess(num_ties, 50)


class TestWeightPreservingKernel(unittest.TestCase):
    def test_visit_frequencies_match_exact_conditional_distribution(self) -> None:
        catalog = _build_ring_catalog()
        neighbors = build_detector_adjacency(catalog)
        probs = np.array([m.p_ref for m in catalog.mechanisms])
        odds = probs / (1 - probs)
        n = len(probs)
        weight = 2

        # Exact pi_w(E) proportional to prod_{i in E} odds_i over all weight-2
        # subsets (no conditioning: threshold=inf disables the level check).
        exact_weight: dict[tuple[int, ...], float] = {}
        for comb in itertools.combinations(range(n), weight):
            w = 1.0
            for i in comb:
                w *= odds[i]
            exact_weight[comb] = w
        total_weight = sum(exact_weight.values())
        exact_pi = {k: v / total_weight for k, v in exact_weight.items()}

        rng = random.Random(1234)
        kernel = WeightPreservingSwapKernel(None, probs, neighbors, rng, local_prob=0.5)

        active = {0, 1}
        current_gap = None
        steps = 200_000
        burn_in = 5_000
        sample_keys: list[tuple[int, ...]] = []
        for t in range(steps):
            active, current_gap, _accepted = kernel.step(active, current_gap, threshold=math.inf)
            if t >= burn_in:
                sample_keys.append(tuple(sorted(active)))

        num_samples = len(sample_keys)
        counts: dict[tuple[int, ...], int] = {}
        for key in sample_keys:
            counts[key] = counts.get(key, 0) + 1

        # Batch-means standard error (consecutive MCMC samples are
        # correlated), same pattern as TestKernelExactness in
        # tests/test_splitting_local.py.
        num_blocks = 40
        block_size = num_samples // num_blocks
        self.assertGreater(block_size, 100)

        for key, p_exact in exact_pi.items():
            indicator = np.array([1.0 if k == key else 0.0 for k in sample_keys])
            block_means = [
                float(indicator[b * block_size : (b + 1) * block_size].mean()) for b in range(num_blocks)
            ]
            p_emp = counts.get(key, 0) / num_samples
            se_block = float(np.std(block_means, ddof=1)) / math.sqrt(num_blocks)
            tol = 5 * se_block + 2e-3
            self.assertLess(
                abs(p_emp - p_exact), tol, f"state {key}: empirical {p_emp} vs exact {p_exact} (tol {tol})"
            )

        # Every weight-2 state should have actually been visited (mode coverage).
        self.assertEqual(set(counts.keys()), set(exact_pi.keys()))


class TestEndToEndAgreement(unittest.TestCase):
    def test_gap_splitting_agrees_with_rejection_sampling(self) -> None:
        p0, error_model, oracle, dem = _setup_repetition_pipeline()
        q_ref = error_model.probabilities(p0)

        weight = 2  # f(2) ~ 0.17: rare enough to be interesting, common enough
        # for plain rejection sampling to measure well in a few thousand trials.
        rng = np.random.default_rng(0)
        trials, failures = sample_fixed_weight_failure_fraction(
            oracle, q_ref, weight, rng, target_failures=100, max_trials=5_000
        )
        self.assertGreater(failures, 0)
        f_ref = failures / trials

        point = estimate_f_w_gap_splitting(
            error_model=error_model,
            oracle=oracle,
            catalog=error_model.catalog,
            dem_or_gap_oracle=dem,
            weight=weight,
            p_ref=p0,
            num_particles=200,
            quantile=0.3,
            mcmc_steps_per_particle=20,
            max_levels=15,
            repeats=3,
            seed=1,
        )

        self.assertEqual(point.method, "gap_splitting")
        self.assertEqual(point.kind, "f_w")
        self.assertEqual(point.weight, weight)
        self.assertGreater(point.estimate, 0.0)
        self.assertLess(abs(math.log(point.estimate) - math.log(f_ref)), math.log(2.5))

    def test_gap_splitting_estimator_wrapper_runs_multiple_weights(self) -> None:
        p0, error_model, oracle, dem = _setup_repetition_pipeline()

        result = GapSplittingEstimator().estimate(
            error_model=error_model,
            simulator=oracle,
            catalog=error_model.catalog,
            dem=dem,
            weights=[2, 3],
            p_ref=p0,
            num_particles=100,
            quantile=0.3,
            mcmc_steps_per_particle=10,
            max_levels=10,
            repeats=2,
            seed=7,
        )

        self.assertIn("weight_points", result)
        points = result["weight_points"]
        self.assertEqual(len(points), 2)
        self.assertEqual([p.weight for p in points], [2, 3])
        for p in points:
            self.assertEqual(p.method, "gap_splitting")
            self.assertGreaterEqual(p.estimate, 0.0)

        # Reusing an already-built GapOracle should work identically to
        # passing the raw DEM.
        gap_oracle = result["gap_oracle"]
        self.assertIsInstance(gap_oracle, GapOracle)
        result2 = GapSplittingEstimator().estimate(
            error_model=error_model,
            simulator=oracle,
            catalog=error_model.catalog,
            gap_oracle=gap_oracle,
            weights=[2],
            p_ref=p0,
            num_particles=50,
            mcmc_steps_per_particle=5,
            max_levels=5,
            repeats=1,
            seed=3,
        )
        self.assertEqual(len(result2["weight_points"]), 1)


class TestHarvestFailingStates(unittest.TestCase):
    def test_harvested_states_are_distinct_failing_and_of_requested_weight(self) -> None:
        p0, error_model, oracle, dem = _setup_repetition_pipeline()

        point = estimate_f_w_gap_splitting(
            error_model=error_model,
            oracle=oracle,
            catalog=error_model.catalog,
            dem_or_gap_oracle=dem,
            weight=3,
            p_ref=p0,
            num_particles=100,
            quantile=0.3,
            mcmc_steps_per_particle=10,
            max_levels=10,
            repeats=1,
            seed=7,
            harvest_states=10,
        )

        states = point.meta["failing_states"]
        self.assertGreater(len(states), 0)
        self.assertLessEqual(len(states), 10)
        self.assertEqual(len({tuple(s) for s in states}), len(states))  # distinct
        for s in states:
            self.assertEqual(len(s), 3)
            self.assertTrue(oracle.fails(set(s)))

    def test_no_harvest_key_when_disabled(self) -> None:
        p0, error_model, oracle, dem = _setup_repetition_pipeline()
        point = estimate_f_w_gap_splitting(
            error_model=error_model,
            oracle=oracle,
            catalog=error_model.catalog,
            dem_or_gap_oracle=dem,
            weight=2,
            p_ref=p0,
            num_particles=50,
            mcmc_steps_per_particle=5,
            max_levels=5,
            repeats=1,
            seed=3,
        )
        self.assertNotIn("failing_states", point.meta)


class TestGapOracleValidation(unittest.TestCase):
    def test_requires_exactly_one_logical_observable(self) -> None:
        mechanisms = [
            ErrorMechanism(detectors=(0,), observables=(0,), p_ref=0.1),
            ErrorMechanism(detectors=(0,), observables=(1,), p_ref=0.1),
        ]
        catalog = MechanismCatalog(mechanisms=mechanisms, num_detectors=1, num_observables=2)

        with self.assertRaises(ValueError):
            # The observable-count check runs before the dem argument is
            # inspected, so a dummy sentinel value suffices here.
            GapOracle.from_dem(dem=None, catalog=catalog)


if __name__ == "__main__":
    unittest.main()

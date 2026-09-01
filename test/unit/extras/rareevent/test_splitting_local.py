from __future__ import annotations

import contextlib
import io
import itertools
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
from pygsti.extras.rareevent.splitting_local import (
    LocalConditionalFailureMCMC,
    LocalSplittingEstimator,
    build_detector_adjacency,
)


class CoreMajorityOracle:
    """E fails iff at least 2 of mechanisms {0, 1, 2} are active."""

    def fails(self, active: set[int]) -> bool:
        return len(active & {0, 1, 2}) >= 2


def _build_ring_catalog() -> MechanismCatalog:
    # A 6-mechanism ring where consecutive mechanisms share a detector, so the
    # detector-adjacency neighborhood of any mechanism is itself plus its two
    # ring neighbors. Heterogeneous probabilities exercise the odds ratio.
    p_refs = [0.02, 0.05, 0.08, 0.12, 0.2, 0.3]
    detector_pairs = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)]
    mechanisms = [
        ErrorMechanism(detectors=pair, observables=(), p_ref=p)
        for pair, p in zip(detector_pairs, p_refs)
    ]
    return MechanismCatalog(mechanisms=mechanisms, num_detectors=6, num_observables=0)


class TestBuildDetectorAdjacency(unittest.TestCase):
    def test_ring_neighbors_include_self_and_ring_neighbors_only(self) -> None:
        catalog = _build_ring_catalog()
        neighbors = build_detector_adjacency(catalog)
        expected = [
            (0, 1, 5),
            (0, 1, 2),
            (1, 2, 3),
            (2, 3, 4),
            (3, 4, 5),
            (0, 4, 5),
        ]
        self.assertEqual(neighbors, expected)

    def test_mechanism_with_no_detectors_has_only_self_as_neighbor(self) -> None:
        mechanisms = [
            ErrorMechanism(detectors=(), observables=(0,), p_ref=0.1),
            ErrorMechanism(detectors=(0,), observables=(), p_ref=0.1),
        ]
        catalog = MechanismCatalog(mechanisms=mechanisms, num_detectors=1, num_observables=1)
        neighbors = build_detector_adjacency(catalog)
        self.assertEqual(neighbors[0], (0,))
        self.assertEqual(neighbors[1], (1,))


class TestKernelExactness(unittest.TestCase):
    def test_visit_frequencies_match_exact_conditional_distribution(self) -> None:
        catalog = _build_ring_catalog()
        neighbors = build_detector_adjacency(catalog)
        probs = np.array([m.p_ref for m in catalog.mechanisms])
        odds = probs / (1 - probs)
        oracle = CoreMajorityOracle()
        rng = random.Random(1234)

        # Exact conditional distribution pi(E) proportional to prod_{i in E} odds_i,
        # restricted to failing sets, by brute-force enumeration of all 2^6 subsets.
        n = len(probs)
        exact_weight: dict[tuple[int, ...], float] = {}
        for bits in itertools.product([0, 1], repeat=n):
            active = {i for i, b in enumerate(bits) if b}
            if oracle.fails(active):
                w = 1.0
                for i in active:
                    w *= odds[i]
                exact_weight[tuple(sorted(active))] = w
        total_weight = sum(exact_weight.values())
        exact_pi = {k: v / total_weight for k, v in exact_weight.items()}

        # Sanity check: the failing family is connected under single toggles
        # that stay within the family (required for any single-toggle MCMC,
        # local or baseline, to be able to visit every failing state).
        failing_states = set(exact_weight.keys())

        def neighbors_in_family(state: tuple[int, ...]) -> set[tuple[int, ...]]:
            s = set(state)
            out = set()
            for i in range(n):
                t = set(s)
                if i in t:
                    t.discard(i)
                else:
                    t.add(i)
                key = tuple(sorted(t))
                if key in failing_states:
                    out.add(key)
            return out

        visited = {next(iter(failing_states))}
        frontier = list(visited)
        while frontier:
            cur = frontier.pop()
            for nxt in neighbors_in_family(cur):
                if nxt not in visited:
                    visited.add(nxt)
                    frontier.append(nxt)
        self.assertEqual(visited, failing_states, "failing family must be connected for this test to be valid")

        chain = LocalConditionalFailureMCMC(
            oracle=oracle,
            probabilities=probs,
            neighbors=neighbors,
            rng=rng,
            beta_global=0.2,
        )

        initial = {0, 1}
        self.assertTrue(oracle.fails(initial))
        steps = 400_000
        samples, acceptance_rate = chain.sample(initial=initial, steps=steps, burn_in=5_000, thin=1)

        self.assertGreater(acceptance_rate, 0.0)

        sample_keys = [tuple(sorted(s)) for s in samples]
        num_samples = len(sample_keys)
        counts: dict[tuple[int, ...], int] = {}
        for key in sample_keys:
            counts[key] = counts.get(key, 0) + 1

        # Consecutive MCMC samples are correlated, so the naive iid binomial
        # standard error understates the true sampling noise (measured
        # inflation factor ~3-4x for this kernel/chain). Use a batch-means
        # standard error instead: split the chain into blocks, treat each
        # block's mean indicator as one (approximately independent) draw.
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

        # Every failing state should have actually been visited (mode coverage).
        self.assertEqual(set(counts.keys()), failing_states)


class TestNeighborhoodBookkeeping(unittest.TestCase):
    def test_incremental_cover_matches_rebuild_from_scratch(self) -> None:
        catalog = _build_ring_catalog()
        neighbors = build_detector_adjacency(catalog)
        probs = np.array([m.p_ref for m in catalog.mechanisms])
        oracle = CoreMajorityOracle()

        chain = LocalConditionalFailureMCMC(
            oracle=oracle,
            probabilities=probs,
            neighbors=neighbors,
            rng=random.Random(7),
            beta_global=0.3,
        )

        active = {0, 1}
        for _ in range(2_000):
            active, _accepted = chain.step(active)

        # Rebuild N(E) from scratch and compare against the chain's incremental state.
        n = len(probs)
        expected_cover = np.zeros(n, dtype=np.int64)
        for i in chain.active:
            for m in chain.neighbors[i]:
                expected_cover[m] += 1
        expected_members = set(np.flatnonzero(expected_cover > 0).tolist())

        np.testing.assert_array_equal(chain.cover_count, expected_cover)
        self.assertEqual(set(chain.members), expected_members)
        self.assertEqual(len(chain.members), len(set(chain.members)))  # no duplicates
        self.assertEqual(set(chain.position.keys()), expected_members)
        for m, pos in chain.position.items():
            self.assertEqual(chain.members[pos], m)


class TestLocalSplittingEstimatorIntegration(unittest.TestCase):
    def test_repetition_code_against_monte_carlo_and_malignant_bound(self) -> None:
        np.random.seed(11)
        p0 = 0.02
        circuit = make_repetition_code_memory_circuit(distance=3, rounds=2, p=0)
        noise = SI1000NoiseModel()
        error_model = ExactNoiseErrorModel(circuit, noise, p_ref=p0)
        dem = noise(circuit, p0).detector_error_model(decompose_errors=True, flatten_loops=True)
        matching = pymatching.Matching.from_detector_error_model(dem)
        oracle = FailureOracle(error_model.catalog, matching)

        p_scales = [0.02, 0.008, 0.003]
        estimator = LocalSplittingEstimator()
        with contextlib.redirect_stdout(io.StringIO()):
            result = estimator.estimate(
                error_model=error_model,
                simulator=oracle,
                catalog=error_model.catalog,
                p_scales=p_scales,
                mc_shots_at_p0=20_000,
                total_steps_per_level=60_000,
                thin=20,
                seed=17,
                beta_global=0.1,
            )

        self.assertEqual(result.p_scales, p_scales)
        self.assertEqual(len(result.failure_estimates), len(p_scales))

        # p0 estimate should agree with direct Monte Carlo at p0.
        np.random.seed(99)
        probs0 = error_model.probabilities(p0)
        mc_pfail, _mc_se, _seed = direct_monte_carlo_failure_rate(oracle, probs0, 20_000)
        self.assertLess(abs(math.log(result.failure_estimates[0]) - math.log(mc_pfail)), math.log(1.5))

        # Final (lowest-p) estimate should agree with the exact weight-<=3 malignant bound.
        with contextlib.redirect_stdout(io.StringIO()):
            res_mal = MalignantSetEstimator().estimate(
                error_model=error_model,
                simulator=oracle,
                p_scales=[p_scales[-1]],
                max_weight=3,
                num_mechanisms=error_model.num_mechanisms,
            )
        exact_low = res_mal["failure_estimates"][0]
        self.assertLess(abs(math.log(result.failure_estimates[-1]) - math.log(exact_low)), math.log(2.0))


class TestExternalAnchor(unittest.TestCase):
    def _setup(self) -> tuple[float, ExactNoiseErrorModel, FailureOracle]:
        p0 = 0.02
        circuit = make_repetition_code_memory_circuit(distance=3, rounds=2, p=0)
        noise = SI1000NoiseModel()
        error_model = ExactNoiseErrorModel(circuit, noise, p_ref=p0)
        dem = noise(circuit, p0).detector_error_model(decompose_errors=True, flatten_loops=True)
        matching = pymatching.Matching.from_detector_error_model(dem)
        oracle = FailureOracle(error_model.catalog, matching)
        return p0, error_model, oracle

    def test_external_anchor_pins_p0_and_descends(self) -> None:
        np.random.seed(23)
        p0, error_model, oracle = self._setup()
        probs0 = error_model.probabilities(p0)
        mc_pfail, _mc_se, failing_state = direct_monte_carlo_failure_rate(oracle, probs0, 20_000)
        assert failing_state is not None

        p_scales = [p0, 0.008]
        with contextlib.redirect_stdout(io.StringIO()):
            result = LocalSplittingEstimator().estimate(
                error_model=error_model,
                simulator=oracle,
                catalog=error_model.catalog,
                p_scales=p_scales,
                total_steps_per_level=30_000,
                thin=20,
                seed=5,
                anchor_failure_rate=mc_pfail,
                anchor_state=failing_state,
            )

        # The supplied anchor is used verbatim, and the descent produces a lower estimate.
        self.assertAlmostEqual(result.failure_estimates[0], mc_pfail)
        self.assertLess(result.failure_estimates[1], result.failure_estimates[0])

    def test_external_anchor_validation(self) -> None:
        np.random.seed(29)
        p0, error_model, oracle = self._setup()
        estimator = LocalSplittingEstimator()

        def run(**anchor_kwargs: object) -> None:
            estimator.estimate(
                error_model=error_model,
                simulator=oracle,
                catalog=error_model.catalog,
                p_scales=[p0, 0.008],
                total_steps_per_level=1_000,
                **anchor_kwargs,
            )

        with self.assertRaises(ValueError):  # anchor rate without a failing state
            run(anchor_failure_rate=1e-3)
        with self.assertRaises(ValueError):  # state without a rate
            run(anchor_state={0})
        with self.assertRaises(ValueError):  # non-failing anchor state
            run(anchor_failure_rate=1e-3, anchor_state=set())


class TestMultiChainSeeds(unittest.TestCase):
    def _setup(self) -> tuple[float, ExactNoiseErrorModel, FailureOracle]:
        p0 = 0.02
        circuit = make_repetition_code_memory_circuit(distance=3, rounds=2, p=0)
        noise = SI1000NoiseModel()
        error_model = ExactNoiseErrorModel(circuit, noise, p_ref=p0)
        dem = noise(circuit, p0).detector_error_model(decompose_errors=True, flatten_loops=True)
        matching = pymatching.Matching.from_detector_error_model(dem)
        oracle = FailureOracle(error_model.catalog, matching)
        return p0, error_model, oracle

    def test_multi_chain_with_seed_states_agrees_with_exact_bound(self) -> None:
        np.random.seed(31)
        p0, error_model, oracle = self._setup()
        probs0 = np.asarray(error_model.probabilities(p0), dtype=np.float64)

        # Harvest three distinct failing states by direct sampling at p0 (in
        # production these come from gap_splitting's harvest_states option).
        rng = np.random.default_rng(4)
        seeds: list[set[int]] = []
        seen: set[tuple[int, ...]] = set()
        while len(seeds) < 3:
            draws = rng.random(len(probs0)) < probs0
            active = set(np.flatnonzero(draws).tolist())
            key = tuple(sorted(active))
            if key not in seen and oracle.fails(active):
                seen.add(key)
                seeds.append(active)

        p_scales = [p0, 0.008, 0.003]
        with contextlib.redirect_stdout(io.StringIO()):
            result = LocalSplittingEstimator().estimate(
                error_model=error_model,
                simulator=oracle,
                catalog=error_model.catalog,
                p_scales=p_scales,
                mc_shots_at_p0=20_000,
                total_steps_per_level=90_000,
                thin=20,
                seed=13,
                num_chains=3,
                seed_states=seeds,
            )

        # Diagnostics carry one entry per chain and a genuine cross-chain Rhat.
        for diag in result.level_diagnostics:
            self.assertEqual(len(diag.per_chain_log_ratios), 3)
            self.assertEqual(len(diag.per_chain_acceptance_rates), 3)
            self.assertEqual(len(diag.per_chain_sample_sizes), 3)
            self.assertIsNotNone(diag.rhat_log_weight_ratio)

        # Final (lowest-p) estimate agrees with the exact weight-<=3 malignant bound.
        with contextlib.redirect_stdout(io.StringIO()):
            res_mal = MalignantSetEstimator().estimate(
                error_model=error_model,
                simulator=oracle,
                p_scales=[p_scales[-1]],
                max_weight=3,
                num_mechanisms=error_model.num_mechanisms,
            )
        exact_low = res_mal["failure_estimates"][0]
        self.assertLess(abs(math.log(result.failure_estimates[-1]) - math.log(exact_low)), math.log(2.0))

    def test_non_failing_seed_state_raises(self) -> None:
        np.random.seed(37)
        p0, error_model, oracle = self._setup()
        with self.assertRaises(ValueError):
            with contextlib.redirect_stdout(io.StringIO()):
                LocalSplittingEstimator().estimate(
                    error_model=error_model,
                    simulator=oracle,
                    catalog=error_model.catalog,
                    p_scales=[p0, 0.008],
                    mc_shots_at_p0=5_000,
                    total_steps_per_level=1_000,
                    num_chains=2,
                    seed_states=[set()],
                )

    def test_invalid_num_chains_raises(self) -> None:
        p0, error_model, oracle = self._setup()
        with self.assertRaises(ValueError):
            LocalSplittingEstimator().estimate(
                error_model=error_model,
                simulator=oracle,
                catalog=error_model.catalog,
                p_scales=[p0, 0.008],
                total_steps_per_level=1_000,
                num_chains=0,
            )


if __name__ == "__main__":
    unittest.main()

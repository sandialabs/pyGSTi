from __future__ import annotations

import contextlib
import io
import itertools
import math
import unittest

import numpy as np
import pymatching

from pygsti.extras.rareevent.core_planting import CountingOracle
from pygsti.extras.rareevent.interfaces import ForwardSimulator
from pygsti.extras.rareevent.malignant import MalignantSetEstimator
from pygsti.extras.rareevent.noise import ExactNoiseErrorModel, SI1000NoiseModel
from pygsti.extras.rareevent.rare_event import (
    FailureOracle,
    direct_monte_carlo_failure_rate,
    make_repetition_code_memory_circuit,
)
from pygsti.extras.rareevent.splitting_subregion import (
    SubregionConditionalFailureMCMC,
    SubregionLevelDiagnostics,
    SubregionSplittingEstimator,
    bennett_log_ratio,
    default_region_rate,
    subregion_splitting_estimate,
)


class CoreMajorityOracle:
    """E fails iff at least 2 of mechanisms {0, 1, 2} are active."""

    def fails(self, active: set[int]) -> bool:
        return len(active & {0, 1, 2}) >= 2


def _exact_conditional(probs: np.ndarray, oracle: ForwardSimulator) -> dict[tuple[int, ...], float]:
    """Brute-force pi(E | failure) proportional to prod_{i in E} odds_i over failing sets."""
    odds = probs / (1 - probs)
    n = len(probs)
    weight: dict[tuple[int, ...], float] = {}
    for bits in itertools.product([0, 1], repeat=n):
        active = {i for i, b in enumerate(bits) if b}
        if oracle.fails(active):
            w = 1.0
            for i in active:
                w *= odds[i]
            weight[tuple(sorted(active))] = w
    total = sum(weight.values())
    return {k: v / total for k, v in weight.items()}


PROBS6 = np.array([0.02, 0.05, 0.08, 0.12, 0.2, 0.3])


class TestDefaultRegionRate(unittest.TestCase):
    def test_heuristic_and_validation(self) -> None:
        self.assertAlmostEqual(default_region_rate(4), 0.25)
        self.assertAlmostEqual(default_region_rate(1), 1.0)
        with self.assertRaises(ValueError):
            default_region_rate(0)


class TestKernelExactness(unittest.TestCase):
    def _run_and_check(self, chain: SubregionConditionalFailureMCMC, oracle: ForwardSimulator) -> None:
        exact_pi = _exact_conditional(PROBS6, oracle)
        initial = {0, 1}
        self.assertTrue(oracle.fails(initial))
        steps = 200_000
        samples, acceptance_rate = chain.sample(initial=initial, steps=steps, burn_in=5_000, thin=1)
        self.assertGreater(acceptance_rate, 0.0)

        sample_keys = [tuple(sorted(s)) for s in samples]
        num_samples = len(sample_keys)
        counts: dict[tuple[int, ...], int] = {}
        for key in sample_keys:
            counts[key] = counts.get(key, 0) + 1

        # Batch-means standard error (consecutive MCMC samples are correlated).
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
        # Mode coverage: every failing state visited.
        self.assertEqual(set(counts.keys()), set(exact_pi.keys()))

    def test_visit_frequencies_match_exact_conditional_rejection_free_path(self) -> None:
        oracle = CoreMajorityOracle()
        chain = SubregionConditionalFailureMCMC(
            oracle=oracle,
            probabilities=PROBS6,
            region_rate=0.4,
            rng=np.random.default_rng(1234),
        )
        self._run_and_check(chain, oracle)

    def test_visit_frequencies_match_exact_conditional_mh_corrected_path(self) -> None:
        oracle = CoreMajorityOracle()
        # f != q exercises the Hastings-corrected branch.
        resample = np.full_like(PROBS6, 0.25)
        chain = SubregionConditionalFailureMCMC(
            oracle=oracle,
            probabilities=PROBS6,
            region_rate=0.4,
            resample_probs=resample,
            rng=np.random.default_rng(4321),
        )
        self._run_and_check(chain, oracle)


class TestKernelMechanics(unittest.TestCase):
    def test_rejection_free_with_default_resample(self) -> None:
        """With f = q, every toggling proposal that still fails is accepted."""
        oracle = CountingOracle(CoreMajorityOracle())
        chain = SubregionConditionalFailureMCMC(
            oracle=oracle,
            probabilities=PROBS6,
            region_rate=0.5,
            rng=np.random.default_rng(7),
        )
        chain.set_state({0, 1})
        for _ in range(20_000):
            chain._step_once()
        c = chain.counters
        # Every real proposal consulted the oracle (no Metropolis pre-rejection)...
        self.assertEqual(c.oracle_calls, c.proposals)
        # ...and acceptance count equals the number of proposals the oracle passed.
        self.assertEqual(c.steps, c.noop_steps + c.proposals)
        self.assertGreater(c.accepted, 0)

    def test_noop_steps_skip_the_oracle(self) -> None:
        oracle = CountingOracle(CoreMajorityOracle())
        # Tiny region rate: most steps miss the active set entirely.
        chain = SubregionConditionalFailureMCMC(
            oracle=oracle,
            probabilities=np.full(6, 0.01),
            region_rate=0.02,
            rng=np.random.default_rng(3),
        )
        chain.set_state({0, 1})
        for _ in range(5_000):
            chain._step_once()
        c = chain.counters
        self.assertGreater(c.noop_steps, 0)
        self.assertEqual(oracle.calls, c.proposals)  # no oracle call on no-ops
        self.assertLess(c.proposals, c.steps)

    def test_validation(self) -> None:
        oracle = CoreMajorityOracle()
        with self.assertRaises(ValueError):
            SubregionConditionalFailureMCMC(oracle, np.array([0.0, 0.5]), region_rate=0.5)
        with self.assertRaises(ValueError):
            SubregionConditionalFailureMCMC(oracle, PROBS6, region_rate=0.0)
        with self.assertRaises(ValueError):
            SubregionConditionalFailureMCMC(oracle, PROBS6, region_rate=1.5)
        with self.assertRaises(ValueError):
            SubregionConditionalFailureMCMC(oracle, PROBS6, region_rate=0.5, resample_probs=np.array([0.5]))
        with self.assertRaises(ValueError):
            chain = SubregionConditionalFailureMCMC(oracle, PROBS6, region_rate=0.5)
            chain.sample(initial=set(), steps=10)  # empty set does not fail


class TestBennettLogRatio(unittest.TestCase):
    def test_matches_exact_ratio_on_enumerable_model(self) -> None:
        """BAR recovers log(Z_next/Z_current) from exact iid conditional samples."""
        oracle = CoreMajorityOracle()
        rng = np.random.default_rng(11)
        probs_current = PROBS6
        probs_next = PROBS6 * 0.5

        def z(probs: np.ndarray) -> float:
            odds = probs / (1 - probs)
            total = 0.0
            for bits in itertools.product([0, 1], repeat=len(probs)):
                active = {i for i, b in enumerate(bits) if b}
                if oracle.fails(active):
                    w = float(np.prod(np.where([b for b in bits], odds, 1.0)))
                    total += w
            # Normalize both Z's by prod(1-p) at their own probs: the ratio of
            # *conditional* normalizers is Z_next/Z_current with these factors.
            return total * float(np.prod(1 - probs))

        exact_log_ratio = math.log(z(probs_next)) - math.log(z(probs_current))

        def sample_conditional(probs: np.ndarray, m: int) -> list[set[int]]:
            out: list[set[int]] = []
            while len(out) < m:
                draws = rng.random(len(probs)) < probs
                active = set(np.flatnonzero(draws).tolist())
                if oracle.fails(active):
                    out.append(active)
            return out

        from pygsti.extras.rareevent.rare_event import log_weight_ratio

        fwd_states = sample_conditional(probs_current, 4_000)
        rev_states = sample_conditional(probs_next, 4_000)
        fwd = [log_weight_ratio(s, probs_next, probs_current) for s in fwd_states]
        rev = [log_weight_ratio(s, probs_next, probs_current) for s in rev_states]

        bar = bennett_log_ratio(fwd, rev)
        self.assertLess(abs(bar - exact_log_ratio), 0.05)

        # Strongly unequal sample sizes must still recover the exact ratio: a
        # sign error in the log(n_R/n_F) shift biases C by 2*log(5) ~ 3.2 here.
        bar_unequal = bennett_log_ratio(fwd, rev[:800])
        self.assertLess(abs(bar_unequal - exact_log_ratio), 0.15)
        bar_unequal_rev = bennett_log_ratio(fwd[:800], rev)
        self.assertLess(abs(bar_unequal_rev - exact_log_ratio), 0.15)

    def test_validation(self) -> None:
        with self.assertRaises(ValueError):
            bennett_log_ratio([], [0.0])
        with self.assertRaises(ValueError):
            bennett_log_ratio([0.0], [])


def _repetition_setup() -> tuple[float, ExactNoiseErrorModel, FailureOracle]:
    p0 = 0.02
    circuit = make_repetition_code_memory_circuit(distance=3, rounds=2, p=0)
    noise = SI1000NoiseModel()
    error_model = ExactNoiseErrorModel(circuit, noise, p_ref=p0)
    dem = noise(circuit, p0).detector_error_model(decompose_errors=True, flatten_loops=True)
    matching = pymatching.Matching.from_detector_error_model(dem)
    oracle = FailureOracle(error_model.catalog, matching)
    return p0, error_model, oracle


class TestSubregionSplittingEstimatorIntegration(unittest.TestCase):
    def test_repetition_code_against_monte_carlo_and_malignant_bound(self) -> None:
        p0, error_model, oracle = _repetition_setup()
        p_scales = [0.02, 0.008, 0.003]
        with contextlib.redirect_stdout(io.StringIO()):
            result = SubregionSplittingEstimator().estimate(
                error_model=error_model,
                simulator=oracle,
                p_scales=p_scales,
                region_rate=default_region_rate(2),
                mc_shots_at_p0=20_000,
                total_steps_per_level=60_000,
                thin=20,
                seed=17,
            )

        self.assertEqual(result.p_scales, p_scales)
        self.assertEqual(len(result.failure_estimates), len(p_scales))
        for diag in result.level_diagnostics:
            self.assertIsInstance(diag, SubregionLevelDiagnostics)

        np.random.seed(99)
        probs0 = error_model.probabilities(p0)
        mc_pfail, _mc_se, _seed = direct_monte_carlo_failure_rate(oracle, probs0, 20_000)
        self.assertLess(abs(math.log(result.failure_estimates[0]) - math.log(mc_pfail)), math.log(1.5))

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

    def test_bar_ratio_estimator_agrees_with_forward(self) -> None:
        _p0, error_model, oracle = _repetition_setup()
        p_scales = [0.02, 0.008, 0.003]
        with contextlib.redirect_stdout(io.StringIO()):
            result = subregion_splitting_estimate(
                error_model=error_model,
                simulator=oracle,
                p_scales=p_scales,
                region_rate=0.5,
                mc_shots_at_p0=20_000,
                total_steps_per_level=60_000,
                thin=20,
                seed=23,
                ratio_estimator="bar",
            )
        for diag in result.level_diagnostics:
            assert isinstance(diag, SubregionLevelDiagnostics)
            self.assertIsNotNone(diag.bar_log_ratio)
            self.assertIsNotNone(diag.bar_reverse_sample_size)
            assert diag.bar_log_ratio is not None and diag.bar_reverse_sample_size is not None
            self.assertGreater(diag.bar_reverse_sample_size, 0)
            # BAR and forward are estimating the same ratio; on this easy model
            # they must agree well within a factor of ~1.5 in log space.
            self.assertLess(abs(diag.bar_log_ratio - diag.forward_log_ratio), math.log(1.5))
            self.assertAlmostEqual(diag.pooled_log_ratio, diag.bar_log_ratio)

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

    def test_rhat_stopping_stops_early_and_respects_cap(self) -> None:
        _p0, error_model, oracle = _repetition_setup()
        p_scales = [0.02, 0.008]

        # Loose threshold: an easy level should stop well before the cap.
        with contextlib.redirect_stdout(io.StringIO()):
            result = subregion_splitting_estimate(
                error_model=error_model,
                simulator=oracle,
                p_scales=p_scales,
                region_rate=0.5,
                mc_shots_at_p0=20_000,
                thin=1,
                seed=29,
                num_chains=2,
                stop_rhat=1.2,
                block_steps=1_000,
                min_steps_per_chain=2_000,
                max_steps_per_chain=50_000,
            )
        diag = result.level_diagnostics[0]
        assert isinstance(diag, SubregionLevelDiagnostics)
        self.assertTrue(diag.rhat_threshold_met)
        self.assertLess(diag.steps_per_chain_used, 50_000)

        # Unreachable threshold: the level must run to the cap and flag it.
        with contextlib.redirect_stdout(io.StringIO()):
            result = subregion_splitting_estimate(
                error_model=error_model,
                simulator=oracle,
                p_scales=p_scales,
                region_rate=0.5,
                mc_shots_at_p0=20_000,
                thin=1,
                seed=31,
                num_chains=2,
                stop_rhat=1.0000001,
                block_steps=1_000,
                min_steps_per_chain=1_000,
                max_steps_per_chain=4_000,
            )
        diag = result.level_diagnostics[0]
        assert isinstance(diag, SubregionLevelDiagnostics)
        self.assertFalse(diag.rhat_threshold_met)
        self.assertEqual(diag.steps_per_chain_used, 4_000)

    def test_validation_errors(self) -> None:
        _p0, error_model, oracle = _repetition_setup()

        def run(**kwargs: object) -> None:
            with contextlib.redirect_stdout(io.StringIO()):
                subregion_splitting_estimate(
                    error_model=error_model,
                    simulator=oracle,
                    p_scales=[0.02, 0.008],
                    region_rate=0.5,
                    mc_shots_at_p0=2_000,
                    **kwargs,  # type: ignore[arg-type]
                )

        with self.assertRaises(ValueError):  # stop_rhat needs multiple chains
            run(stop_rhat=1.05, num_chains=1)
        with self.assertRaises(ValueError):  # stop_rhat conflicts with fixed budgets
            run(stop_rhat=1.05, num_chains=2, total_steps_per_level=1_000)
        with self.assertRaises(ValueError):  # rhat threshold must exceed 1
            run(stop_rhat=0.9, num_chains=2)
        with self.assertRaises(ValueError):  # missing step budget in fixed mode
            run()
        with self.assertRaises(ValueError):  # both budgets given
            run(steps_per_chain=100, total_steps_per_level=100)
        with self.assertRaises(ValueError):  # unknown ratio estimator
            run(total_steps_per_level=1_000, ratio_estimator="sideways")
        with self.assertRaises(ValueError):  # anchor state without rate
            run(total_steps_per_level=1_000, anchor_state={0})


if __name__ == "__main__":
    unittest.main()

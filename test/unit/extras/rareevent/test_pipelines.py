from __future__ import annotations

import contextlib
import io
import math
import unittest

import numpy as np
import pymatching

from pygsti.extras.rareevent.gap_splitting import GapOracle
from pygsti.extras.rareevent.malignant import MalignantSetEstimator
from pygsti.extras.rareevent.noise import ExactNoiseErrorModel, SI1000NoiseModel
from pygsti.extras.rareevent.pipelines import (
    GapSeededSplittingEstimator,
    GapSpectrumEstimator,
    default_onset_weight,
    gap_seeded_splitting_estimate,
    gap_spectrum_estimate,
    measure_gap_weight_points,
)
from pygsti.extras.rareevent.rare_event import (
    FailureOracle,
    direct_monte_carlo_failure_rate,
    make_repetition_code_memory_circuit,
)


def _setup_repetition_pipeline() -> tuple[float, ExactNoiseErrorModel, FailureOracle, object]:
    # Same pipeline construction as tests/test_gap_splitting.py.
    p0 = 0.02
    circuit = make_repetition_code_memory_circuit(distance=3, rounds=2, p=0)
    noise = SI1000NoiseModel()
    error_model = ExactNoiseErrorModel(circuit, noise, p_ref=p0)
    dem = noise(circuit, p0).detector_error_model(decompose_errors=True, flatten_loops=True)
    matching = pymatching.Matching.from_detector_error_model(dem)
    oracle = FailureOracle(error_model.catalog, matching)
    return p0, error_model, oracle, dem


class TestDefaultOnsetWeight(unittest.TestCase):
    def test_values(self) -> None:
        self.assertEqual(default_onset_weight(3), 2)
        self.assertEqual(default_onset_weight(5), 3)
        self.assertEqual(default_onset_weight(11), 6)

    def test_invalid(self) -> None:
        with self.assertRaises(ValueError):
            default_onset_weight(0)


class TestMeasureGapWeightPoints(unittest.TestCase):
    def test_points_and_harvest(self) -> None:
        p0, error_model, oracle, dem = _setup_repetition_pipeline()
        points = measure_gap_weight_points(
            error_model,
            oracle,
            error_model.catalog,
            dem,
            weights=[2, 3],
            p_ref=p0,
            num_particles=60,
            repeats=2,
            mcmc_steps_per_particle=10,
            seed=7,
            harvest_states=8,
        )
        self.assertEqual([pt.weight for pt in points], [2, 3])
        for pt in points:
            self.assertEqual(pt.kind, "f_w")
            self.assertGreater(pt.estimate, 0.0)
            harvested = pt.meta["failing_states"]
            self.assertGreater(len(harvested), 0)
            for state in harvested:
                self.assertEqual(len(state), pt.weight)
                self.assertTrue(oracle.fails(set(state)))

    def test_reuses_prebuilt_gap_oracle(self) -> None:
        p0, error_model, oracle, dem = _setup_repetition_pipeline()
        gap_oracle = GapOracle.from_dem(dem, error_model.catalog)
        points = measure_gap_weight_points(
            error_model,
            oracle,
            error_model.catalog,
            gap_oracle,
            weights=[2],
            p_ref=p0,
            num_particles=40,
            repeats=1,
            mcmc_steps_per_particle=5,
            seed=3,
        )
        self.assertEqual(len(points), 1)
        self.assertGreater(gap_oracle.decode_count, 0)


class TestGapSpectrumEstimate(unittest.TestCase):
    def test_matches_direct_monte_carlo_at_p0(self) -> None:
        p0, error_model, oracle, dem = _setup_repetition_pipeline()
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            result = gap_spectrum_estimate(
                error_model,
                oracle,
                error_model.catalog,
                dem,
                p_scales=[p0, 0.01, 0.005],
                onset_weight=2,
                gap_weight_span=2,
                gap_num_particles=60,
                gap_repeats=2,
                gap_mcmc_steps_per_particle=10,
                target_failures=60,
                max_trials_per_weight=4_000,
                seed=5,
            )
        self.assertGreater(len(result.aux_points), 0)
        self.assertEqual(result.aux_points[0].method, "gap_splitting")
        self.assertTrue(all(f > 0 and math.isfinite(f) for f in result.failure_estimates))
        # The decomposition is exact at p_ref, so the prediction there should
        # be close to direct Monte Carlo.
        np.random.seed(11)
        q0 = np.asarray(error_model.probabilities(p0), dtype=np.float64)
        mc, _se, _state = direct_monte_carlo_failure_rate(oracle, q0, 4_000)
        self.assertLess(abs(math.log(result.failure_estimates[0]) - math.log(mc)), math.log(1.8))

    def test_rejects_explicit_aux_points(self) -> None:
        p0, error_model, oracle, dem = _setup_repetition_pipeline()
        with self.assertRaises(ValueError):
            gap_spectrum_estimate(
                error_model,
                oracle,
                error_model.catalog,
                dem,
                p_scales=[p0],
                onset_weight=2,
                aux_points=[],
            )

    def test_rejects_bad_onset_weight(self) -> None:
        p0, error_model, oracle, dem = _setup_repetition_pipeline()
        with self.assertRaises(ValueError):
            gap_spectrum_estimate(
                error_model, oracle, error_model.catalog, dem, p_scales=[p0], onset_weight=0
            )


class TestGapSeededSplittingEstimate(unittest.TestCase):
    def test_multi_chain_estimate_brackets_malignant_bound(self) -> None:
        p0, error_model, oracle, dem = _setup_repetition_pipeline()
        p_scales = [p0, 0.008, 0.003]
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            result = gap_seeded_splitting_estimate(
                error_model,
                oracle,
                error_model.catalog,
                dem,
                p_scales,
                onset_weight=2,
                num_chains=3,
                harvest_weight_span=1,
                harvest_states_per_weight=8,
                gap_num_particles=60,
                gap_mcmc_steps_per_particle=10,
                mc_shots_at_p0=6_000,
                total_steps_per_level=90_000,
                thin=20,
                seed=2,
            )
        self.assertEqual(len(result.failure_estimates), len(p_scales))
        for diag in result.level_diagnostics:
            self.assertEqual(len(diag.per_chain_log_ratios), 3)
            self.assertEqual(len(diag.per_chain_acceptance_rates), 3)
            self.assertIsNotNone(diag.rhat_log_weight_ratio)
        # Weight-<=3 malignant enumeration is a tight lower bound at the final
        # (lowest) rate for this small catalog.
        bound = MalignantSetEstimator().estimate(
            error_model=error_model,
            simulator=oracle,
            p_scales=[p_scales[-1]],
            max_weight=3,
            num_mechanisms=error_model.num_mechanisms,
        )["failure_estimates"][0]
        self.assertLess(abs(math.log(result.failure_estimates[-1]) - math.log(bound)), math.log(2.0))

    def test_single_chain_skips_harvest(self) -> None:
        p0, error_model, oracle, dem = _setup_repetition_pipeline()
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            result = gap_seeded_splitting_estimate(
                error_model,
                oracle,
                error_model.catalog,
                dem,
                [p0, 0.012],
                onset_weight=2,
                num_chains=1,
                mc_shots_at_p0=4_000,
                total_steps_per_level=20_000,
                thin=10,
                seed=4,
            )
        self.assertEqual(len(result.level_diagnostics[0].per_chain_log_ratios), 1)

    def test_rejects_seed_states_kwarg(self) -> None:
        p0, error_model, oracle, dem = _setup_repetition_pipeline()
        with self.assertRaises(ValueError):
            gap_seeded_splitting_estimate(
                error_model,
                oracle,
                error_model.catalog,
                dem,
                [p0, 0.01],
                onset_weight=2,
                seed_states=[{0}],
            )


class TestEstimatorClasses(unittest.TestCase):
    def test_gap_spectrum_estimator_requires_kwargs(self) -> None:
        p0, error_model, oracle, dem = _setup_repetition_pipeline()
        with self.assertRaises(ValueError):
            GapSpectrumEstimator().estimate(error_model=error_model, simulator=oracle)
        with self.assertRaises(ValueError):
            GapSpectrumEstimator().estimate(
                error_model=error_model,
                simulator=oracle,
                p_scales=[p0],
                catalog=error_model.catalog,
                onset_weight=2,
            )  # missing dem/gap_oracle

    def test_gap_seeded_splitting_estimator_runs(self) -> None:
        p0, error_model, oracle, dem = _setup_repetition_pipeline()
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            result = GapSeededSplittingEstimator().estimate(
                error_model=error_model,
                simulator=oracle,
                catalog=error_model.catalog,
                dem=dem,
                p_scales=[p0, 0.012],
                onset_weight=2,
                num_chains=2,
                harvest_weight_span=0,
                harvest_states_per_weight=4,
                gap_num_particles=40,
                gap_mcmc_steps_per_particle=5,
                mc_shots_at_p0=4_000,
                total_steps_per_level=20_000,
                thin=10,
                seed=9,
            )
        self.assertEqual(len(result.level_diagnostics[0].per_chain_log_ratios), 2)


if __name__ == "__main__":
    unittest.main()

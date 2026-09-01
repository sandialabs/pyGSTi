from __future__ import annotations

import contextlib
import io
import math
import os
import pathlib
import random
import unittest

import numpy as np
import stim

ROOT = pathlib.Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib-cache"))
(ROOT / ".matplotlib-cache").mkdir(exist_ok=True)

from pygsti.extras.rareevent import rare_event  # noqa: E402


class CountingOracle:
    def __init__(self, result: bool = True):
        self.result = result
        self.calls = 0

    def fails(self, active: set[int]) -> bool:
        self.calls += 1
        return self.result


class FixedRng(random.Random):
    def __init__(self, index: int, uniforms: list[float]):
        self.index = index
        self.uniforms = list(uniforms)

    def randrange(self, start: int, stop: int | None = None, step: int = 1) -> int:
        n = stop if stop is not None else start
        assert 0 <= self.index < n
        return self.index

    def random(self) -> float:
        if not self.uniforms:
            raise AssertionError("No fixed uniform values left.")
        return self.uniforms.pop(0)


class RareEventTests(unittest.TestCase):
    def test_log_probability_and_ratio_match_manual_bernoulli_products(self) -> None:
        probs_current = np.array([0.1, 0.2, 0.3])
        probs_next = np.array([0.05, 0.4, 0.15])
        active = {0, 2}

        expected_current = math.log(0.1) + math.log1p(-0.2) + math.log(0.3)
        expected_next = math.log(0.05) + math.log1p(-0.4) + math.log(0.15)

        self.assertAlmostEqual(
            rare_event.log_probability_of_state(active, probs_current),
            expected_current,
        )
        self.assertAlmostEqual(
            rare_event.log_weight_ratio(active, probs_next, probs_current),
            expected_next - expected_current,
        )

    def test_mcmc_rejects_by_metropolis_draw_before_decoding(self) -> None:
        rare_event.MechanismCatalog(
            mechanisms=[rare_event.ErrorMechanism(detectors=(0,), observables=(), p_ref=0.01)],
            num_detectors=1,
            num_observables=1,
        )
        oracle = CountingOracle(result=True)
        rng = FixedRng(index=0, uniforms=[0.5])
        chain = rare_event.ConditionalFailureMCMC(
            oracle=oracle,
            probabilities=np.array([0.01]),
            rng=rng,
        )

        state, accepted = chain.step(set())

        self.assertEqual(state, set())
        self.assertFalse(accepted)
        self.assertEqual(oracle.calls, 0)

    def test_mcmc_decodes_only_after_passing_metropolis_draw(self) -> None:
        rare_event.MechanismCatalog(
            mechanisms=[rare_event.ErrorMechanism(detectors=(0,), observables=(), p_ref=0.01)],
            num_detectors=1,
            num_observables=1,
        )
        oracle = CountingOracle(result=True)
        rng = FixedRng(index=0, uniforms=[0.0])
        chain = rare_event.ConditionalFailureMCMC(
            oracle=oracle,
            probabilities=np.array([0.01]),
            rng=rng,
        )

        state, accepted = chain.step(set())

        self.assertEqual(state, {0})
        self.assertTrue(accepted)
        self.assertEqual(oracle.calls, 1)

    def test_repetition_catalog_includes_global_all_bits_dem_event(self) -> None:
        p0 = 0.01
        dem_probability = 1e-4 * p0
        circuit = rare_event.make_repetition_code_memory_circuit(distance=3, rounds=3, p=p0)
        catalog, _oracle, dem_text = rare_event.build_catalog_decoder_and_dem_text(
            circuit,
            global_dem_event_probability=dem_probability,
        )

        global_event = catalog.mechanisms[-1]

        self.assertIn("global_all_bits", dem_text)
        self.assertAlmostEqual(global_event.p_ref, dem_probability)
        self.assertEqual(global_event.detectors, tuple(range(catalog.num_detectors)))
        self.assertEqual(global_event.observables, tuple(range(catalog.num_observables)))

    def test_catalog_builder_flattens_detector_error_models(self) -> None:
        dem = stim.DetectorErrorModel(
            """
            error(0.125) D0
            repeat 2 {
                error(0.25) D0 D1
                shift_detectors 1
            }
            error(0.5) D0 L0
            """
        )

        catalog = rare_event.MechanismCatalog.from_detector_error_model(dem)

        self.assertEqual(catalog.num_detectors, 3)
        self.assertEqual(catalog.num_observables, 1)
        self.assertEqual(
            [m.detectors for m in catalog.mechanisms],
            [(0,), (0, 1), (1, 2), (2,)],
        )
        self.assertEqual(catalog.mechanisms[-1].observables, (0,))

    def test_repetition_splitting_agrees_with_direct_monte_carlo_at_high_error_rate(self) -> None:
        p0 = 0.08
        p_final = 0.04
        np.random.seed(321)
        random.seed(321)

        circuit = rare_event.make_repetition_code_memory_circuit(distance=3, rounds=3, p=p0)
        catalog, oracle, _dem_text = rare_event.build_catalog_decoder_and_dem_text(
            circuit,
            global_dem_event_probability=1e-4 * p0,
        )

        error_model = rare_event.ScaledMechanismErrorModel(catalog, p0)
        with contextlib.redirect_stdout(io.StringIO()):
            result = rare_event.rare_event_splitting_estimate(
                error_model=error_model,
                simulator=oracle,
                p_scales=[p0, p_final],
                mc_shots_at_p0=5_000,
                steps_per_chain=None,
                total_steps_per_level=8_000,
                burn_in=None,
                burn_in_fraction=0.1,
                thin=10,
                seed=123,
            )

        np.random.seed(999)
        direct, direct_se, _seed = rare_event.direct_monte_carlo_failure_rate(
            oracle=oracle,
            probs=catalog.probabilities_scaled_from_reference(p_final, p0),
            shots=10_000,
        )

        split = result.failure_estimates[-1]
        tolerance = max(0.03, 8 * direct_se)
        self.assertLess(abs(split - direct), tolerance)

    def test_repetition_plot_splitting_against_direct_monte_carlo(self) -> None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        p0 = 0.08
        p_scales = [float(x) for x in np.geomspace(p0, 0.03, 5)]
        np.random.seed(1234)
        random.seed(1234)

        circuit = rare_event.make_repetition_code_memory_circuit(distance=3, rounds=3, p=p0)
        catalog, oracle, _dem_text = rare_event.build_catalog_decoder_and_dem_text(
            circuit,
            global_dem_event_probability=1e-4 * p0,
        )

        error_model = rare_event.ScaledMechanismErrorModel(catalog, p0)
        with contextlib.redirect_stdout(io.StringIO()):
            result = rare_event.rare_event_splitting_estimate(
                error_model=error_model,
                simulator=oracle,
                p_scales=p_scales,
                mc_shots_at_p0=8_000,
                steps_per_chain=None,
                total_steps_per_level=12_000,
                burn_in=None,
                burn_in_fraction=0.1,
                thin=20,
                seed=456,
            )

        direct_estimates = []
        direct_errors = []
        for i, p in enumerate(p_scales):
            np.random.seed(10_000 + i)
            estimate, stderr, _seed = rare_event.direct_monte_carlo_failure_rate(
                oracle=oracle,
                probs=catalog.probabilities_scaled_from_reference(float(p), p0),
                shots=12_000,
            )
            direct_estimates.append(estimate)
            direct_errors.append(stderr)

        pdf_path = ROOT / "tests.pdf"
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        ax.loglog(p_scales, result.failure_estimates, marker="o", label="Rare-event splitting")
        ax.errorbar(
            p_scales,
            direct_estimates,
            yerr=direct_errors,
            marker="s",
            linestyle="--",
            capsize=3,
            label="Direct Monte Carlo",
        )
        ax.set_title("Repetition code logical failure rate")
        ax.set_xlabel("Physical error rate p")
        ax.set_ylabel("Logical failure rate")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(pdf_path)
        plt.close(fig)

        self.assertTrue(pdf_path.exists())
        self.assertGreater(pdf_path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()

import contextlib
import io
import itertools
import math
import unittest

import numpy as np
import pymatching

from pygsti.extras.rareevent.malignant import MalignantSetEstimator
from pygsti.extras.rareevent.noise import ExactNoiseErrorModel, SI1000NoiseModel
from pygsti.extras.rareevent.rare_event import (
    FailureOracle,
    SplittingResult,
    direct_monte_carlo_failure_rate,
    make_repetition_code_memory_circuit,
)
from pygsti.extras.rareevent.splitting_smc import SMCSplittingEstimator, smc_splitting_estimate


class ToyErrorModel:
    """Linear-in-p scaling of a fixed base probability vector (6 mechanisms)."""

    def __init__(self, base: np.ndarray) -> None:
        self.base = base

    def probabilities(self, p: float) -> np.ndarray:
        return self.base * p


class ToySimulator:
    """Fails iff at least two of mechanisms {0, 1, 2} are simultaneously active."""

    def fails(self, active: set[int]) -> bool:
        return len(active & {0, 1, 2}) >= 2


def exact_p_fail(probs: np.ndarray) -> float:
    """Brute-force P_fail by enumerating all 2^n subsets (n small)."""
    n = len(probs)
    total = 0.0
    for bits in itertools.product([0, 1], repeat=n):
        active = {i for i, b in enumerate(bits) if b}
        if len(active & {0, 1, 2}) < 2:
            continue
        prob = 1.0
        for i in range(n):
            prob *= probs[i] if bits[i] else (1.0 - probs[i])
        total += prob
    return total


BASE_PROBS = np.array([0.5, 0.6, 0.4, 0.7, 0.3, 0.55])
P_SCALES = [0.1, 0.03, 0.01, 0.003]


class TestSMCSplittingToyProblem(unittest.TestCase):
    """Validates the SMC weight/resample math end to end against exact enumeration."""

    error_model: ToyErrorModel
    simulator: ToySimulator
    exact: list[float]
    result: SplittingResult

    @classmethod
    def setUpClass(cls) -> None:
        cls.error_model = ToyErrorModel(BASE_PROBS)
        cls.simulator = ToySimulator()
        cls.exact = [exact_p_fail(cls.error_model.probabilities(p)) for p in P_SCALES]

        with contextlib.redirect_stdout(io.StringIO()):
            cls.result = smc_splitting_estimate(
                error_model=cls.error_model,
                simulator=cls.simulator,
                p_scales=P_SCALES,
                mc_shots_at_p0=40_000,
                num_walkers=512,
                mcmc_steps_per_walker=2_000,
                init_mcmc_steps=200,
                seed=7,
            )

    def test_matches_exact_conditional_at_every_level(self) -> None:
        self.assertEqual(len(self.result.failure_estimates), len(P_SCALES))
        for k, (est, exact) in enumerate(zip(self.result.failure_estimates, self.exact)):
            self.assertGreater(est, 0.0)
            self.assertGreater(exact, 0.0)
            log_err = abs(math.log(est) - math.log(exact))
            self.assertLess(log_err, 0.3, f"level {k}: est={est:.6g} exact={exact:.6g} log_err={log_err:.4g}")

    def test_resampling_diagnostics_are_sane(self) -> None:
        num_walkers = 512
        for diag in self.result.level_diagnostics:
            lw = np.asarray(diag.per_chain_log_ratios, dtype=np.float64)
            m = np.max(lw)
            w = np.exp(lw - m)
            ess = float(np.sum(w) ** 2 / np.sum(w * w))
            self.assertGreater(ess, 0.0)
            self.assertLessEqual(ess, num_walkers + 1e-6)

            unique_count = diag.per_chain_sample_sizes[0]
            self.assertGreaterEqual(unique_count, 1)
            self.assertLessEqual(unique_count, num_walkers)


class TestSMCSplittingEstimatorIntegration(unittest.TestCase):
    def test_repetition_code_against_direct_mc_and_malignant_bound(self) -> None:
        p0 = 0.02
        circuit = make_repetition_code_memory_circuit(distance=3, rounds=2, p=0)
        noise = SI1000NoiseModel()
        error_model = ExactNoiseErrorModel(circuit, noise, p_ref=p0)
        dem = noise(circuit, p0).detector_error_model(decompose_errors=True, flatten_loops=True)
        matching = pymatching.Matching.from_detector_error_model(dem)
        oracle = FailureOracle(error_model.catalog, matching)

        estimator = SMCSplittingEstimator()
        with contextlib.redirect_stdout(io.StringIO()):
            result = estimator.estimate(
                error_model=error_model,
                simulator=oracle,
                p_scales=[0.02, 0.008, 0.003],
                num_walkers=128,
                mcmc_steps_per_walker=400,
                mc_shots_at_p0=20_000,
                seed=11,
            )

        self.assertEqual(len(result.failure_estimates), 3)
        self.assertEqual(result.sample_sizes, [128, 128])

        probs0 = error_model.probabilities(p0)
        mc_pfail, _, _ = direct_monte_carlo_failure_rate(oracle, probs0, 20_000)
        self.assertLess(abs(math.log(result.failure_estimates[0]) - math.log(mc_pfail)), math.log(1.5))

        with contextlib.redirect_stdout(io.StringIO()):
            res_mal = MalignantSetEstimator().estimate(
                error_model=error_model,
                simulator=oracle,
                p_scales=[0.003],
                max_weight=3,
                num_mechanisms=error_model.num_mechanisms,
            )
        exact_low = res_mal["failure_estimates"][0]
        self.assertLess(abs(math.log(result.failure_estimates[-1]) - math.log(exact_low)), math.log(2.0))


if __name__ == "__main__":
    unittest.main()

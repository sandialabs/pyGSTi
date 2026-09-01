import contextlib
import io
import itertools
import math
import unittest

import numpy as np
import pymatching
from scipy.stats import binom

from pygsti.extras.rareevent.failure_spectrum import (
    FailureSpectrumEstimator,
    FittedFailureSpectrum,
    failure_spectrum_estimate,
    fit_failure_spectrum,
    logspaced_integer_weights,
    poisson_binomial_pmf,
    sample_fixed_weight_failure_fraction,
    tilted_probabilities,
    transform_spectrum_to_failure_rate,
)
from pygsti.extras.rareevent.malignant import MalignantSetEstimator
from pygsti.extras.rareevent.noise import ExactNoiseErrorModel, SI1000NoiseModel
from pygsti.extras.rareevent.rare_event import (
    FailureOracle,
    direct_monte_carlo_failure_rate,
    make_repetition_code_memory_circuit,
)
from pygsti.extras.rareevent.weight_points import WeightPoint


class TestPoissonBinomialPmf(unittest.TestCase):
    def test_matches_binomial_for_uniform_probabilities(self) -> None:
        n, q = 20, 0.03
        pmf = poisson_binomial_pmf(np.full(n, q))
        expected = binom.pmf(np.arange(n + 1), n, q)
        np.testing.assert_allclose(pmf, expected, rtol=1e-10)

    def test_matches_brute_force_for_heterogeneous_probabilities(self) -> None:
        rng = np.random.default_rng(7)
        q = rng.uniform(0.01, 0.4, size=8)
        pmf = poisson_binomial_pmf(q)
        expected = np.zeros(len(q) + 1)
        for bits in itertools.product([0, 1], repeat=len(q)):
            prob = np.prod(np.where(np.array(bits) == 1, q, 1 - q))
            expected[sum(bits)] += prob
        np.testing.assert_allclose(pmf, expected, rtol=1e-10)

    def test_truncation_keeps_low_weight_mass_exact(self) -> None:
        rng = np.random.default_rng(8)
        q = rng.uniform(0.05, 0.3, size=12)
        full = poisson_binomial_pmf(q)
        truncated = poisson_binomial_pmf(q, max_weight=4)
        np.testing.assert_allclose(truncated, full[:5], rtol=1e-12)


class TestFixedWeightSampling(unittest.TestCase):
    def test_tilted_probabilities_hit_target_mean(self) -> None:
        q = np.array([0.001, 0.002, 0.01, 0.03, 0.05, 0.1, 0.2, 0.0])
        for w in [1, 3, 5]:
            q_t = tilted_probabilities(q, w)
            self.assertAlmostEqual(float(q_t.sum()), w, places=8)
            self.assertEqual(q_t[-1], 0.0)  # zero-probability mechanisms stay off

    def test_conditional_distribution_is_exact(self) -> None:
        # Sample weight-2 sets from 5 heterogeneous mechanisms and compare the
        # empirical set frequencies with the exact conditional distribution.
        q = np.array([0.01, 0.03, 0.05, 0.10, 0.20])
        w = 2

        counts: dict[tuple[int, ...], int] = {}

        class RecordingSimulator:
            def fails(self, active: set[int]) -> bool:
                key = tuple(sorted(active))
                counts[key] = counts.get(key, 0) + 1
                return False

        rng = np.random.default_rng(123)
        num_samples = 40_000
        trials, failures = sample_fixed_weight_failure_fraction(
            RecordingSimulator(),
            q,
            w,
            rng,
            target_failures=1,
            max_trials=num_samples,
        )
        self.assertEqual(trials, num_samples)
        self.assertEqual(failures, 0)
        self.assertEqual(sum(counts.values()), num_samples)
        for key in counts:
            self.assertEqual(len(key), w)

        odds = q / (1 - q)
        exact = {
            pair: odds[pair[0]] * odds[pair[1]]
            for pair in itertools.combinations(range(len(q)), 2)
        }
        norm = sum(exact.values())
        for pair, unnorm in exact.items():
            p_exact = unnorm / norm
            p_emp = counts.get(pair, 0) / num_samples
            tol = 5 * math.sqrt(p_exact * (1 - p_exact) / num_samples) + 1e-4
            self.assertLess(abs(p_emp - p_exact), tol, f"pair {pair}: {p_emp} vs {p_exact}")


class TestAnsatz(unittest.TestCase):
    def test_shape_constraints(self) -> None:
        spec = FittedFailureSpectrum(ansatz="3", a=0.5, w0=4.0, f0=1e-3, gamma1=4.0)
        w = np.arange(0, 500)
        f = spec(w)
        np.testing.assert_array_equal(f[:4], 0.0)  # zero below onset
        self.assertAlmostEqual(float(f[4]), 1e-3, delta=1e-5)  # f(w0) ~= f0
        self.assertAlmostEqual(float(f[-1]), 0.5, places=6)  # asymptote a
        self.assertTrue(np.all(np.diff(f[4:]) >= 0))  # monotone

    def test_transform_matches_binomial_tail_for_step_spectrum(self) -> None:
        # A d=5 repetition code at code capacity fails iff >= 3 of 5 flips:
        # the spectrum is a step function and the transform must reproduce the
        # exact binomial tail.
        step = FittedFailureSpectrum(ansatz="3", a=1.0, w0=3.0, f0=200.0, gamma1=200.0)
        n, p = 5, 0.02
        self.assertAlmostEqual(float(step(np.array([3.0]))[0]), 1.0, places=6)
        got = transform_spectrum_to_failure_rate(step, np.full(n, p))
        expected = float(1.0 - binom.cdf(2, n, p))
        self.assertAlmostEqual(got, expected, delta=1e-12)

    def test_fit_roundtrip_recovers_spectrum(self) -> None:
        true = FittedFailureSpectrum(ansatz="3", a=0.5, w0=4.0, f0=2e-3, gamma1=5.0)
        weights = logspaced_integer_weights(4, 120, 14)
        rng = np.random.default_rng(2024)
        trials = [40_000] * len(weights)
        failures = [int(rng.binomial(t, float(true(w)[0]))) for t, w in zip(trials, weights)]

        fitted, report = fit_failure_spectrum(weights, trials, failures, a=0.5, ansatz="3", w0=4.0)
        self.assertTrue(report["success"])

        # The fitted spectrum should agree with the truth over the sampled
        # range and extrapolate accurately down to the onset weight.
        for w in [4, 6, 10, 30, 60, 120]:
            t, f = float(true(w)[0]), float(fitted(w)[0])
            self.assertLess(abs(math.log(f) - math.log(t)), 0.25, f"w={w}: {f} vs {t}")

        # Predictions through the transform should match the truth closely.
        q = np.full(600, 0.01)
        p_true = transform_spectrum_to_failure_rate(true, q)
        p_fit = transform_spectrum_to_failure_rate(fitted, q)
        self.assertLess(abs(math.log(p_fit) - math.log(p_true)), 0.25)

    def test_fit_with_free_onset_weight(self) -> None:
        true = FittedFailureSpectrum(ansatz="3", a=0.5, w0=4.0, f0=2e-3, gamma1=5.0)
        weights = logspaced_integer_weights(4, 120, 14)
        rng = np.random.default_rng(11)
        trials = [40_000] * len(weights)
        failures = [int(rng.binomial(t, float(true(w)[0]))) for t, w in zip(trials, weights)]

        fitted, report = fit_failure_spectrum(weights, trials, failures, a=0.5, ansatz="3", w0=None)
        self.assertTrue(report["fitted_w0"])
        self.assertLessEqual(fitted.w0, 4.0 + 1e-9)
        p_true = transform_spectrum_to_failure_rate(true, np.full(600, 0.01))
        p_fit = transform_spectrum_to_failure_rate(fitted, np.full(600, 0.01))
        self.assertLess(abs(math.log(p_fit) - math.log(p_true)), 0.5)

    def test_fit_rejects_insufficient_data(self) -> None:
        with self.assertRaises(ValueError):
            fit_failure_spectrum([5], [1000], [10], a=0.5, ansatz="3", w0=4.0)

    def test_fit_escapes_saturated_initialization(self) -> None:
        # Regression test for a d=11 surface-code failure mode: failures are
        # only observable far above the onset weight, so the legacy initial
        # guess (f0 = lowest measured fraction, gamma1 = w0) puts every
        # measured weight on the saturated plateau f ~= a, where the optimizer
        # stalls at its starting point with chi2/point in the hundreds.
        true = FittedFailureSpectrum(ansatz="3", a=0.5, w0=6.0, f0=8e-9, gamma1=5.5)
        weights = [6, 8, 11, 16, 21, 29, 40, 55, 75, 104, 142, 195]
        rng = np.random.default_rng(7)
        trials = [20_000] * len(weights)
        failures = [int(rng.binomial(t, float(true(w)[0]))) for t, w in zip(trials, weights)]
        self.assertEqual(failures[:5], [0] * 5)  # low weights genuinely unobservable

        fitted, report = fit_failure_spectrum(weights, trials, failures, a=0.5, ansatz="3", w0=6.0)
        self.assertLess(report["chi2_per_point"], 5.0)
        for w in [6, 20, 60, 195]:
            t, f = float(true(w)[0]), float(fitted(w)[0])
            self.assertLess(abs(math.log(f) - math.log(t)), 0.5, f"w={w}: {f} vs {t}")


class TestAuxPoints(unittest.TestCase):
    def test_aux_points_recover_unobservable_onset(self) -> None:
        # Same regime as test_fit_escapes_saturated_initialization: counting
        # sees no failures below weight ~16, so a free-onset fit is only
        # bounded by the lowest *observed* failure. Auxiliary f(w) points
        # (as fixed-weight gap-splitting supplies) pin the true onset at 6.
        true = FittedFailureSpectrum(ansatz="3", a=0.5, w0=6.0, f0=8e-9, gamma1=5.5)
        weights = [16, 21, 29, 40, 55, 75, 104, 142, 195]
        rng = np.random.default_rng(21)
        trials = [20_000] * len(weights)
        failures = [int(rng.binomial(t, float(true(w)[0]))) for t, w in zip(trials, weights)]

        aux_w = [6.0, 8.0, 11.0]
        aux_f = [float(true(w)[0]) * math.exp(rng.normal(0.0, 0.1)) for w in aux_w]
        aux_s = [0.1, 0.1, 0.1]

        fitted, report = fit_failure_spectrum(
            weights,
            trials,
            failures,
            a=0.5,
            ansatz="3",
            w0=None,
            aux_weights=aux_w,
            aux_fractions=aux_f,
            aux_sigma_log=aux_s,
        )
        self.assertEqual(report["num_aux_points"], 3)
        self.assertTrue(report["fitted_w0"])
        # The auxiliary point at weight 6 lowers the fitted-onset upper bound
        # from the lowest counted failure (>= 16) to 6.
        self.assertLessEqual(fitted.w0, 6.0 + 1e-9)
        for w in [6, 8, 11, 40, 195]:
            t_val, f_val = float(true(w)[0]), float(fitted(w)[0])
            self.assertLess(abs(math.log(f_val) - math.log(t_val)), 0.5, f"w={w}: {f_val} vs {t_val}")

    def test_aux_argument_validation(self) -> None:
        weights = [10, 20, 30]
        trials = [1000] * 3
        failures = [5, 10, 20]
        with self.assertRaises(ValueError):  # mismatched lengths
            fit_failure_spectrum(
                weights, trials, failures, a=0.5, ansatz="3", w0=4.0,
                aux_weights=[4.0], aux_fractions=[1e-6, 1e-5], aux_sigma_log=[0.1],
            )
        with self.assertRaises(ValueError):  # nonpositive fraction
            fit_failure_spectrum(
                weights, trials, failures, a=0.5, ansatz="3", w0=4.0,
                aux_weights=[4.0], aux_fractions=[0.0], aux_sigma_log=[0.1],
            )
        with self.assertRaises(ValueError):  # nonpositive sigma
            fit_failure_spectrum(
                weights, trials, failures, a=0.5, ansatz="3", w0=4.0,
                aux_weights=[4.0], aux_fractions=[1e-6], aux_sigma_log=[0.0],
            )

    def test_estimate_accepts_and_filters_weight_points(self) -> None:
        np.random.seed(17)
        p0 = 0.02
        circuit = make_repetition_code_memory_circuit(distance=3, rounds=2, p=0)
        noise = SI1000NoiseModel()
        error_model = ExactNoiseErrorModel(circuit, noise, p_ref=p0)
        dem = noise(circuit, p0).detector_error_model(decompose_errors=True, flatten_loops=True)
        matching = pymatching.Matching.from_detector_error_model(dem)
        oracle = FailureOracle(error_model.catalog, matching)

        q0 = np.asarray(error_model.probabilities(p0), dtype=np.float64)
        rng = np.random.default_rng(9)
        trials, fails = sample_fixed_weight_failure_fraction(
            oracle, q0, 2, rng, target_failures=200, max_trials=40_000
        )
        aux = [
            WeightPoint(method="test", kind="f_w", weight=2, estimate=fails / trials, rel_err=0.1),
            WeightPoint(method="test", kind="f_w", weight=3, estimate=0.0, rel_err=0.1),  # skipped
            WeightPoint(method="test", kind="f_w", weight=4, estimate=0.5, rel_err=float("nan")),  # skipped
        ]
        with contextlib.redirect_stdout(io.StringIO()):
            result = failure_spectrum_estimate(
                error_model=error_model,
                simulator=oracle,
                p_scales=[p0, 1e-3],
                w0=2,
                target_failures=100,
                max_trials_per_weight=20_000,
                aux_points=aux,
                seed=3,
            )
        self.assertEqual([pt.weight for pt in result.aux_points], [2])
        self.assertEqual(result.fit_report["num_aux_points"], 1)

        # Predictions remain consistent with direct Monte Carlo at p0.
        mc_pfail, _mc_se, _seed_state = direct_monte_carlo_failure_rate(oracle, q0, 40_000)
        self.assertLess(abs(math.log(result.failure_estimates[0]) - math.log(mc_pfail)), math.log(1.6))

        # Wrong-kind aux points are rejected outright.
        with self.assertRaises(ValueError):
            with contextlib.redirect_stdout(io.StringIO()):
                failure_spectrum_estimate(
                    error_model=error_model,
                    simulator=oracle,
                    p_scales=[p0],
                    w0=2,
                    target_failures=20,
                    max_trials_per_weight=2_000,
                    aux_points=[WeightPoint(method="x", kind="m_v", weight=2, estimate=1.0, rel_err=0.1)],
                    seed=3,
                )


class TestFailureSpectrumEstimatorIntegration(unittest.TestCase):
    def test_repetition_code_against_direct_monte_carlo(self) -> None:
        np.random.seed(5)
        p0 = 0.02
        circuit = make_repetition_code_memory_circuit(distance=3, rounds=2, p=0)
        noise = SI1000NoiseModel()
        error_model = ExactNoiseErrorModel(circuit, noise, p_ref=p0)
        dem = noise(circuit, p0).detector_error_model(decompose_errors=True, flatten_loops=True)
        matching = pymatching.Matching.from_detector_error_model(dem)
        oracle = FailureOracle(error_model.catalog, matching)

        estimator = FailureSpectrumEstimator()
        with contextlib.redirect_stdout(io.StringIO()):
            result = estimator.estimate(
                error_model=error_model,
                simulator=oracle,
                p_scales=[p0, 1e-3],
                w0=2,
                target_failures=200,
                max_trials_per_weight=40_000,
                seed=3,
            )

        self.assertEqual(len(result.failure_estimates), 2)
        self.assertGreater(len(result.samples), 2)
        self.assertEqual(result.p_ref, p0)

        # Prediction at p0 should agree with direct Monte Carlo.
        probs = error_model.probabilities(p0)
        mc_pfail, mc_stderr, _ = direct_monte_carlo_failure_rate(oracle, probs, 40_000)
        pred = result.failure_estimates[0]
        self.assertLess(abs(math.log(pred) - math.log(mc_pfail)), math.log(1.5))

        # Prediction at low p should agree with exact weight-<=3 enumeration
        # (a tight bound at p=1e-3).
        with contextlib.redirect_stdout(io.StringIO()):
            res_mal = MalignantSetEstimator().estimate(
                error_model=error_model,
                simulator=oracle,
                p_scales=[1e-3],
                max_weight=3,
                num_mechanisms=error_model.num_mechanisms,
            )
        exact_low = res_mal["failure_estimates"][0]
        self.assertLess(abs(math.log(result.failure_estimates[1]) - math.log(exact_low)), math.log(2.0))


if __name__ == "__main__":
    unittest.main()

import contextlib
import io
import random
import unittest

import numpy as np
import pymatching
import stim
from scipy.stats import binom

from pygsti.extras.rareevent.malignant import MalignantSetEstimator
from pygsti.extras.rareevent.rare_event import (
    FailureOracle,
    MechanismCatalog,
    RareEventSplittingEstimator,
)


class CodeCapacityErrorModel:
    def __init__(self, catalog: MechanismCatalog, p_ref: float):
        self.catalog = catalog
        self.p_ref = p_ref

    def probabilities(self, p: float) -> np.ndarray:
        # Code capacity scaling: probability is exact
        # For depolarizing noise, p_eff = 2*p/3
        return np.asarray(self.catalog.probabilities_scaled_from_reference(p, self.p_ref))

class TestExactSolvableModels(unittest.TestCase):
    def setUp(self) -> None:
        np.random.seed(42)
        random.seed(42)

    def test_code_capacity_repetition_code(self) -> None:
        """Compare MCMC/Splitting and Malignant Set against exact binomial distribution."""
        d = 5
        p0 = 0.1
        p_final = 0.01

        circuit = stim.Circuit.generated(
            "repetition_code:memory",
            distance=d,
            rounds=1,
            before_round_data_depolarization=p0,
            before_measure_flip_probability=0,
            after_reset_flip_probability=0,
            after_clifford_depolarization=0,
        )
        dem = circuit.detector_error_model(decompose_errors=True, flatten_loops=True)
        catalog = MechanismCatalog.from_detector_error_model(dem)
        matching = pymatching.Matching.from_detector_error_model(dem)
        oracle = FailureOracle(catalog, matching)
        error_model = CodeCapacityErrorModel(catalog, p0)

        # 1. Exact Binomial calculation
        def exact_fail_rate(p: float) -> float:
            p_eff = 2 * p / 3  # Effective flip probability for depolarizing
            return float(1.0 - binom.cdf((d - 1) // 2, d, p_eff))

        # 2. Brute-Force Summation (Malignant Set Estimator)
        malignant_estimator = MalignantSetEstimator()
        
        # Suppress prints
        with contextlib.redirect_stdout(io.StringIO()):
            res_malignant = malignant_estimator.estimate(
                error_model=error_model,
                simulator=oracle,
                p_scales=[p0, p_final],
                max_weight=d,
                num_mechanisms=len(catalog.mechanisms),
            )

        mal_p0 = res_malignant["failure_estimates"][0]
        mal_p_final = res_malignant["failure_estimates"][1]

        self.assertAlmostEqual(mal_p0, exact_fail_rate(p0), places=7)
        self.assertAlmostEqual(mal_p_final, exact_fail_rate(p_final), places=7)

        # 3. Rare-Event Splitting
        splitting_estimator = RareEventSplittingEstimator()
        p_scales = [float(x) for x in np.geomspace(p0, p_final, 3)]
        
        with contextlib.redirect_stdout(io.StringIO()):
            res_splitting = splitting_estimator.estimate(
                error_model=error_model,
                simulator=oracle,
                p_scales=p_scales,
                mc_shots_at_p0=50_000,
                steps_per_chain=10_000,
                burn_in_fraction=0.1,
            )

        split_p_final = res_splitting.failure_estimates[-1]
        
        # Because MCMC is stochastic, we check within a relative tolerance
        # e.g., 20% relative error is well within bounds for 10k steps
        exact_pf = exact_fail_rate(p_final)
        rel_error = abs(split_p_final - exact_pf) / exact_pf
        self.assertLess(rel_error, 0.20)

    def test_brute_force_summation_tiny_code(self) -> None:
        """Iterate over all 2^N configurations for a tiny d=3, 1 round code."""
        d = 3
        p = 0.05
        
        # We use a phenomenological noise model (data + measurement error)
        circuit = stim.Circuit.generated(
            "surface_code:rotated_memory_x",
            distance=d,
            rounds=1,
            before_round_data_depolarization=p,
            before_measure_flip_probability=p,
            after_reset_flip_probability=0,
            after_clifford_depolarization=0,
        )
        
        dem = circuit.detector_error_model(decompose_errors=True, flatten_loops=True)
        catalog = MechanismCatalog.from_detector_error_model(dem)
        matching = pymatching.Matching.from_detector_error_model(dem)
        oracle = FailureOracle(catalog, matching)
        error_model = CodeCapacityErrorModel(catalog, p)
        
        N = len(catalog.mechanisms)
        
        malignant_estimator = MalignantSetEstimator()
        with contextlib.redirect_stdout(io.StringIO()):
            res = malignant_estimator.estimate(
                error_model=error_model,
                simulator=oracle,
                p_scales=[p],
                max_weight=N, # Full 2^N brute-force summation
                num_mechanisms=N,
            )
            
        brute_force_pfail = res["failure_estimates"][0]
        
        # To ensure the brute-force is correct, we can cross-verify against high-shots Monte Carlo
        # (Since d=3, rounds=1, p=0.05 has a fairly high fail rate)
        from pygsti.extras.rareevent.rare_event import direct_monte_carlo_failure_rate
        mc_shots = 200_000
        probs = error_model.probabilities(p)
        mc_pfail, mc_stderr, _ = direct_monte_carlo_failure_rate(oracle, probs, mc_shots)
        
        # MC should match Brute-force within ~4 standard deviations
        self.assertLess(abs(brute_force_pfail - mc_pfail), 4 * mc_stderr)

    def test_independent_chains(self) -> None:
        """Verify that two independent 1D problems combine exactly as expected."""
        # Create a circuit with two independent distance-3 repetition codes
        circuit = stim.Circuit("""
            # Code 1
            R 0 1 2
            X_ERROR(0.1) 0 1 2
            M 0 1 2
            DETECTOR rec[-1] rec[-2]
            DETECTOR rec[-2] rec[-3]
            OBSERVABLE_INCLUDE(0) rec[-1]
            
            # Code 2
            R 3 4 5
            X_ERROR(0.1) 3 4 5
            M 3 4 5
            DETECTOR rec[-1] rec[-2]
            DETECTOR rec[-2] rec[-3]
            OBSERVABLE_INCLUDE(1) rec[-1]
        """)
        
        dem = circuit.detector_error_model(decompose_errors=True, flatten_loops=True)
        catalog = MechanismCatalog.from_detector_error_model(dem)
        matching = pymatching.Matching.from_detector_error_model(dem)
        oracle = FailureOracle(catalog, matching)
        error_model = CodeCapacityErrorModel(catalog, p_ref=0.1)
        
        p = 0.1
        malignant_estimator = MalignantSetEstimator()
        
        with contextlib.redirect_stdout(io.StringIO()):
            res = malignant_estimator.estimate(
                error_model=error_model,
                simulator=oracle,
                p_scales=[p],
                max_weight=6,
                num_mechanisms=len(catalog.mechanisms),
            )
            
        pfail = res["failure_estimates"][0]
        
        # Exact theoretical calculation: P(fail) = 1 - P(both succeed)
        # For each code, the probability of failure is the probability of 2 or more errors
        p_1d = float(1.0 - binom.cdf(1, 3, p))
        expected_pfail = 1.0 - (1.0 - p_1d)**2
        
        self.assertAlmostEqual(pfail, expected_pfail, places=7)

if __name__ == "__main__":
    unittest.main()

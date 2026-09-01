import unittest

from pygsti.extras.rareevent.malignant import MalignantSetEstimator
from pygsti.extras.rareevent.noise import ExactNoiseErrorModel, SI1000NoiseModel
from pygsti.extras.rareevent.rare_event import FailureOracle, make_repetition_code_memory_circuit


class TestMalignantSetEstimator(unittest.TestCase):
    def test_malignant_set_counting_repetition_code(self) -> None:
        # Distance 3 repetition code
        c = make_repetition_code_memory_circuit(distance=3, rounds=3, p=0)
        
        # A simple depolarizing noise model
        noise = SI1000NoiseModel()
        error_model = ExactNoiseErrorModel(c, noise, p_ref=0.01)
        
        import pymatching
        dem = noise(c, 0.01).detector_error_model(decompose_errors=True, flatten_loops=True)
        matching = pymatching.Matching.from_detector_error_model(dem)
        oracle = FailureOracle(error_model.catalog, matching)
        
        estimator = MalignantSetEstimator()
        
        # Distance 3 should have failures starting at weight 2
        res = estimator.estimate(
            error_model=error_model,
            simulator=oracle,
            p_scales=[0.01, 0.001],
            max_weight=2,
            num_mechanisms=error_model.num_mechanisms,
        )
        
        p_fails = res["failure_estimates"]
        self.assertEqual(len(p_fails), 2)
        
        # At p=0.01, failure rate should be roughly C * p^2 * (1-p)^N
        # At p=0.001, failure rate should be roughly C * (p/10)^2 * (1-p/10)^N
        ratio = p_fails[0] / p_fails[1]
        self.assertGreater(ratio, 65)
        self.assertLess(ratio, 120)
        
        # Ensure we found some malignant sets
        self.assertGreater(len(res["malignant_sets"]), 0)
        
        # All malignant sets should have weight 2 (none of weight 1 for d=3)
        for s in res["malignant_sets"]:
            self.assertEqual(len(s), 2)

if __name__ == "__main__":
    unittest.main()

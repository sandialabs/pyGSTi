import unittest

import numpy as np
import stim

from pygsti.extras.rareevent.noise import ExactNoiseErrorModel, SI1000NoiseModel


class TestNoiseModels(unittest.TestCase):
    def test_si1000_noise_model(self) -> None:
        c = stim.Circuit("""
            R 0 1
            TICK
            H 0
            TICK
            CX 0 1
            TICK
            M 0 1
        """)
        model = SI1000NoiseModel()
        noisy = model(c, 0.03)

        expected = stim.Circuit("""
            R 0 1
            X_ERROR(0.03) 0 1
            TICK
            H 0
            DEPOLARIZE1(0.03) 0
            DEPOLARIZE1(0.01) 1
            TICK
            CX 0 1
            DEPOLARIZE2(0.03) 0 1
            TICK
            X_ERROR(0.06) 0 1
            M 0 1
        """)
        
        self.assertEqual(str(noisy), str(expected))

    def test_exact_noise_error_model_with_builtin(self) -> None:
        c = stim.Circuit.generated(
            "surface_code:rotated_memory_x",
            distance=3,
            rounds=3,
            after_clifford_depolarization=0,
            before_round_data_depolarization=0,
            before_measure_flip_probability=0,
            after_reset_flip_probability=0,
        )
        model = SI1000NoiseModel()
        error_model = ExactNoiseErrorModel(c, model, p_ref=0.01)

        probs = error_model.probabilities(0.005)
        self.assertEqual(len(probs), error_model.num_mechanisms)
        self.assertTrue(np.all(probs >= 0))
        self.assertTrue(np.all(probs < 1))
        
        # At p=0, all probabilities should be 0
        probs0 = error_model.probabilities(0.0)
        self.assertTrue(np.all(probs0 == 0.0))

if __name__ == "__main__":
    unittest.main()

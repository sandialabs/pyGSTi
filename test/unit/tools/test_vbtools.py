import numpy as np

from pygsti.tools import vbtools
from ..util import BaseCase


class VBToolsTest(BaseCase):

    def test_logspace_shapes_skip_widths_without_room(self):
        # Width 4 has no depth of at least 2**4, so only width 2 can be sampled.
        widths, depths = vbtools.sample_circuit_shapes_logspace_uniform({2: 5, 4: 3}, 200, min_depth_log2=4,
                                                                        seed=0)
        self.assertEqual(len(widths), 200)
        self.assertTrue(all(w == 2 for w in widths))
        self.assertTrue(all(16 <= d <= 32 for d in depths))

    def test_logspace_shapes_min_equals_max(self):
        widths, depths = vbtools.sample_circuit_shapes_logspace_uniform({3: 3}, 50, min_depth_log2=3, seed=0)
        self.assertEqual(len(depths), 50)
        self.assertTrue(all(w == 3 for w in widths))
        self.assertTrue(all(d == 8 for d in depths))

    def test_logspace_shapes_no_width_has_room_raises(self):
        with self.assertRaisesRegex(ValueError, 'larger than max_depths_log2 at every width'):
            vbtools.sample_circuit_shapes_logspace_uniform({2: 2, 4: 1}, 5, min_depth_log2=3, seed=0)
        with self.assertRaisesRegex(ValueError, 'larger than max_depths_log2 at every width'):
            vbtools.sample_circuit_shapes_logspace_uniform({}, 5, seed=0)

    def test_logspace_shapes_width_weights(self):
        # Weights 4 and 2 (power-of-two depths in [2, 16] and [2, 4]) give width 1 a probability of 2/3.
        widths, depths = vbtools.sample_circuit_shapes_logspace_uniform({1: 4, 2: 2}, 20000, min_depth_log2=1,
                                                                        seed=0)
        widths = np.array(widths)
        depths = np.array(depths)
        self.assertAlmostEqual(np.mean(widths == 1), 2 / 3, delta=0.02)
        self.assertTrue(np.all((depths[widths == 1] >= 2) & (depths[widths == 1] <= 16)))
        self.assertTrue(np.all((depths[widths == 2] >= 2) & (depths[widths == 2] <= 4)))

    def test_logspace_shapes_reproducible_with_seed(self):
        max_depths_log2 = {1: 4, 2: 2, 3: 0}
        first = vbtools.sample_circuit_shapes_logspace_uniform(max_depths_log2, 30, seed=7)
        second = vbtools.sample_circuit_shapes_logspace_uniform(max_depths_log2, 30, seed=7)
        other = vbtools.sample_circuit_shapes_logspace_uniform(max_depths_log2, 30, seed=8)
        self.assertEqual(first, second)
        self.assertNotEqual(first, other)

    def test_quasi_uniform_reproducible_with_seed(self):
        max_values = [16, 100, 10]
        min_values = [1, 1, 0]
        args = (8, 3, max_values, min_values, [True, False, False], [False, False, True])
        first = vbtools.quasi_uniform_feature_vectors(*args, seed=3)
        second = vbtools.quasi_uniform_feature_vectors(*args, seed=3)
        other = vbtools.quasi_uniform_feature_vectors(*args, seed=4)
        self.assertTrue(all(np.array_equal(a, b) for a, b in zip(first, second)))
        self.assertFalse(all(np.array_equal(a, b) for a, b in zip(first, other)))
        for values, lo, hi in zip(first, min_values, max_values):
            self.assertTrue(np.all((values >= lo) & (values <= hi)))

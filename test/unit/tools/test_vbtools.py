import json
import subprocess
import sys

import numpy as np

from pygsti.tools import vbtools
from ..util import BaseCase

# Run in a child interpreter: loads vbtools.py by path (no pygsti import, to keep start-up short),
# calls sample_circuit_shapes_logspace_uniform and prints one JSON line with the result or a ValueError.
CHILD_CODE = """
import ast
import importlib.util
import json
import sys

spec = importlib.util.spec_from_file_location('_vbtools_child', sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
args, kwargs = ast.literal_eval(sys.argv[2])
try:
    widths, depths = module.sample_circuit_shapes_logspace_uniform(*args, **kwargs)
except ValueError as err:
    print(json.dumps({'ValueError': str(err)}))
else:
    print(json.dumps({'widths': [int(w) for w in widths], 'depths': [int(d) for d in depths]}))
"""


def sample_shapes_in_subprocess(args, kwargs, path=vbtools.__file__, timeout=120):
    """
    Runs `sample_circuit_shapes_logspace_uniform(*args, **kwargs)` in a fresh interpreter and returns
    `(widths, depths)`, re-raising a `ValueError` from the call. A subprocess is used because a thread
    guard can't stop a spinning sampling loop, and under pytest an endless loop only failed after minutes.
    """
    try:
        proc = subprocess.run([sys.executable, '-c', CHILD_CODE, path, repr((args, kwargs))],
                              capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        raise AssertionError(f"sample_circuit_shapes_logspace_uniform(*{args!r}, **{kwargs!r}) did not finish "
                             f"within {timeout} s and may loop forever.")
    if proc.returncode != 0:
        raise AssertionError(f"The sampling subprocess failed:\n{proc.stderr}")
    result = json.loads(proc.stdout.strip().splitlines()[-1])
    if 'ValueError' in result:
        raise ValueError(result['ValueError'])
    return result['widths'], result['depths']


class VBToolsTest(BaseCase):

    def test_logspace_shapes_skip_widths_without_room(self):
        # Width 4 has no depth of at least 2**4, so only width 2 can be sampled.
        widths, depths = sample_shapes_in_subprocess(({2: 5, 4: 3}, 200), {'min_depth_log2': 4, 'seed': 0})
        self.assertEqual(len(widths), 200)
        self.assertTrue(all(w == 2 for w in widths))
        self.assertTrue(all(16 <= d <= 32 for d in depths))

    def test_logspace_shapes_min_equals_max(self):
        widths, depths = sample_shapes_in_subprocess(({3: 3}, 50), {'min_depth_log2': 3, 'seed': 0})
        self.assertEqual(len(depths), 50)
        self.assertTrue(all(w == 3 for w in widths))
        self.assertTrue(all(d == 8 for d in depths))

    def test_logspace_shapes_no_width_has_room_raises(self):
        with self.assertRaisesRegex(ValueError, 'larger than max_depths_log2 at every width'):
            sample_shapes_in_subprocess(({2: 2, 4: 1}, 5), {'min_depth_log2': 3, 'seed': 0})
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

    def test_logspace_shapes_seed_none_leaves_global_rng(self):
        np.random.seed(0)
        before = np.random.get_state()
        vbtools.sample_circuit_shapes_logspace_uniform({1: 4, 2: 2}, 30)
        after = np.random.get_state()
        self.assertEqual(before[0], after[0])
        self.assertTrue(np.array_equal(before[1], after[1]))
        self.assertEqual(before[2:], after[2:])

    def test_max_depth_log2_values(self):
        # 0.99**64 > 0.5 >= 0.99**128, so depth 2**7 is the first power-of-two depth whose fidelity is at most 0.5.
        self.assertEqual(vbtools.max_depth_log2(0.01, 0.5), 7)
        self.assertEqual(vbtools.max_depth_log2(0.3, 0.5), 1)
        self.assertEqual(vbtools.max_depth_log2(0.9, 0.5), -1)
        error_per_layer = {1: 0.01, 2: 0.3, 3: 0.9}
        self.assertEqual(vbtools.max_depths_log2([1, 2, 3], error_per_layer, 0.5), {1: 7, 2: 1, 3: 0})
        self.assertEqual(vbtools.max_depths_log2([1, 2, 3], error_per_layer, 0.5, include_all_widths=False),
                         {1: 7, 2: 1})

    def test_depth_helpers_reject_out_of_range(self):
        for error_per_layer, min_fidelity in [(0, 0.5), (1, 0.5), (-0.1, 0.5), (0.1, 0), (0.1, 1), (0.1, 1.5)]:
            with self.subTest(error_per_layer=error_per_layer, min_fidelity=min_fidelity):
                with self.assertRaisesRegex(ValueError, 'must be between 0 and 1'):
                    vbtools.max_depth_log2(error_per_layer, min_fidelity)
        with self.assertRaisesRegex(ValueError, 'must be between 0 and 1'):
            vbtools.max_depths_log2([1], {1: 0.0}, 0.5)

    def test_quasi_uniform_rejects_no_samples(self):
        for num_samples in [0, -1]:
            with self.assertRaisesRegex(ValueError, 'num_samples must be at least 1'):
                vbtools.quasi_uniform_feature_vectors(num_samples, 1, [1], [0], [False], [False], seed=0)
        vectors = vbtools.quasi_uniform_feature_vectors(1, 1, [1], [0], [False], [False], seed=0)
        self.assertEqual(len(vectors), 1)
        self.assertEqual(len(vectors[0]), 1)

    def test_quasi_uniform_logspace_and_integer_paths(self):
        # Feature 0 is uniform in log2 on [0, 10], and 64 Sobol points put exactly half of each coordinate
        # below its midpoint, so exactly 32 values lie below 2**5. A draw linear in value would give about 2.
        for s in range(5):
            with self.subTest(seed=s):
                f0, f1 = vbtools.quasi_uniform_feature_vectors(64, 2, [1024, 10], [1, 0], [True, False],
                                                               [False, True], seed=s)
                self.assertEqual(int(np.sum(f0 < 32)), 32)
                self.assertTrue(np.all((f0 >= 1) & (f0 <= 1024)))
                self.assertTrue(np.issubdtype(f1.dtype, np.integer))
                self.assertTrue(np.all((f1 >= 0) & (f1 <= 10)))

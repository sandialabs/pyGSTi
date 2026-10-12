import math
import multiprocessing as mp
import os
import sys
import unittest
import unittest.mock as mock
import warnings
import numpy as np

import pygsti.tools.mptools as mpt
from ..util import BaseCase


# Top-level helper functions (must be module-level for spawn/pickle compatibility)
def _square(x):
    return x * x


def _add(x, y):
    return x + y


def _kwarg_fn(x, factor=1):
    return x * factor


def _matrix_outer_product(rows, cols):
    return np.array([[float(r) * 1000.0 + float(c) for c in cols] for r in rows], dtype=np.float64)


def _eval_pair(pair):
    a, b = pair
    return {f"sum_{a+b}": a * b, f"diff_{abs(a-b)}": a + b}


def _reduce_dicts(d1, d2):
    res = d1.copy()
    for k, v in d2.items():
        res[k] = res.get(k, 0) + v
    return res


def _get_pid(_=None):
    return os.getpid()


class MPToolsTester(BaseCase):
    def test_starmap_with_kwargs(self):
        # Sequential (1 processor)
        args_list = [(1,), (2,), (3,)]
        kwargs_list = [{'factor': 2}, {'factor': 3}, {'factor': 4}]
        res_seq = mpt.starmap_with_kwargs(_kwarg_fn, 3, 1, args_list, kwargs_list)
        self.assertEqual(res_seq, [2, 6, 12])

        # Parallel (2 processors)
        res_par = mpt.starmap_with_kwargs(_kwarg_fn, 3, 2, args_list, kwargs_list)
        self.assertEqual(res_par, [2, 6, 12])

    def test_determine_num_workers_none_and_integers(self):
        self.assertEqual(mpt.determine_num_workers(None), 1)
        self.assertEqual(mpt.determine_num_workers(4), 4)
        self.assertEqual(mpt.determine_num_workers(16, max_limit=8), 8)
        self.assertEqual(mpt.determine_num_workers(4, max_limit=8), 4)
        with self.assertRaises(ValueError):
            mpt.determine_num_workers(0)
        with self.assertRaises(ValueError):
            mpt.determine_num_workers(-5)

    def test_determine_num_workers_auto(self):
        workers = mpt.determine_num_workers("auto")
        self.assertIsInstance(workers, int)
        self.assertGreaterEqual(workers, 1)

        limited = mpt.determine_num_workers("auto", max_limit=2)
        self.assertLessEqual(limited, 2)

    def test_determine_num_workers_invalid(self):
        with self.assertRaises(ValueError):
            mpt.determine_num_workers("invalid_option")

    @unittest.skipUnless(sys.platform.startswith("linux"), "cgroups only applicable on Linux")
    def test_determine_num_workers_cgroup_v2_clamping(self):
        original_exists = os.path.exists

        def mock_exists(path):
            if path == "/sys/fs/cgroup/cpu.max":
                return True
            return original_exists(path)

        mock_file = mock.mock_open(read_data="400000 100000\n")
        with mock.patch("os.path.exists", side_effect=mock_exists), \
             mock.patch("builtins.open", mock_file), \
             mock.patch("psutil.cpu_count", return_value=64):
            workers = mpt.determine_num_workers("auto")
            self.assertEqual(workers, 4)

    @unittest.skipUnless(sys.platform.startswith("linux"), "cgroups only applicable on Linux")
    def test_determine_num_workers_cgroup_v1_clamping(self):
        original_exists = os.path.exists

        def mock_exists(path):
            if path in ("/sys/fs/cgroup/cpu/cpu.cfs_quota_us", "/sys/fs/cgroup/cpu/cpu.cfs_period_us"):
                return True
            if path == "/sys/fs/cgroup/cpu.max":
                return False
            return original_exists(path)

        def mock_file_reader(path, *args, **kwargs):
            if path == "/sys/fs/cgroup/cpu/cpu.cfs_quota_us":
                return mock.mock_open(read_data="200000\n")()
            elif path == "/sys/fs/cgroup/cpu/cpu.cfs_period_us":
                return mock.mock_open(read_data="100000\n")()
            return open(path, *args, **kwargs)

        with mock.patch("os.path.exists", side_effect=mock_exists), \
             mock.patch("builtins.open", side_effect=mock_file_reader), \
             mock.patch("psutil.cpu_count", return_value=32):
            workers = mpt.determine_num_workers("auto")
            self.assertEqual(workers, 2)

    def test_measure_multiprocessing_ipc_latency(self):
        ipc_s = mpt.measure_multiprocessing_ipc_latency(num_workers=2)
        self.assertIsInstance(ipc_s, float)
        self.assertGreater(ipc_s, 0.0)

        # Subsequent call returns cached value
        ipc_cached = mpt.measure_multiprocessing_ipc_latency(num_workers=2)
        self.assertEqual(ipc_cached, ipc_s)

    def test_compute_2d_worker_grid(self):
        # 1D row cases (M=1)
        self.assertEqual(mpt.compute_2d_worker_grid(1, 1000, 16), (1, 16))
        self.assertEqual(mpt.compute_2d_worker_grid(1, 4, 16), (1, 4))

        # 1D column cases
        self.assertEqual(mpt.compute_2d_worker_grid(1000, 1, 16), (16, 1))

        # 2D balanced cases
        pr, pc = mpt.compute_2d_worker_grid(4, 1000, 16)
        self.assertLessEqual(pr * pc, 16)
        self.assertGreaterEqual(pr, 1)
        self.assertGreaterEqual(pc, 1)

        pr, pc = mpt.compute_2d_worker_grid(16, 1000, 16)
        self.assertEqual(pr * pc, 16)

        # 1 worker
        self.assertEqual(mpt.compute_2d_worker_grid(10, 10, 1), (1, 1))
        self.assertEqual(mpt.compute_2d_worker_grid(0, 0, 16), (1, 1))

    def test_shared_worker_pool_parallel(self):
        with mpt.SharedWorkerPool(num_workers=2) as pool:
            res = pool.map(_square, [1, 2, 3, 4])
            self.assertEqual(res, [1, 4, 9, 16])

            starmap_res = pool.starmap(_add, [(1, 2), (3, 4)])
            self.assertEqual(starmap_res, [3, 7])

    def test_shared_worker_pool_sequential(self):
        with mpt.SharedWorkerPool(num_workers=1) as pool:
            res = pool.map(_square, [1, 2, 3])
            self.assertEqual(res, [1, 4, 9])

            starmap_res = pool.starmap(_add, [(1, 2), (3, 4)])
            self.assertEqual(starmap_res, [3, 7])

    @unittest.skipUnless(sys.platform.startswith("linux"), "fork start method only supported and safe on Linux")
    def test_shared_worker_pool_fork_start_method(self):
        with mpt.SharedWorkerPool(num_workers=2, start_method="fork") as pool:
            res = pool.map(_square, [2, 3, 4])
            self.assertEqual(res, [4, 9, 16])

    def test_shared_worker_pool_spawn_start_method(self):
        with mpt.SharedWorkerPool(num_workers=2, start_method="spawn", prewarm_packages=("math",)) as pool:
            res = pool.map(_square, [2, 3, 4])
            self.assertEqual(res, [4, 9, 16])

    def test_parallel_2d_grid_map_exact_equivalence(self):
        row_items = list(range(8)) 
        col_items = list(range(100))

        expected = _matrix_outer_product(row_items, col_items)

        # Parallel via ephemeral pool
        par_result = mpt.parallel_2d_grid_map(_matrix_outer_product, row_items, col_items, num_workers=4)
        self.assertEqual(par_result.shape, (len(row_items), len(col_items)))
        self.assertArraysAlmostEqual(par_result, expected)

        # Parallel reusing an existing pool
        with mpt.SharedWorkerPool(num_workers=4) as shared_pool:
            reused_result = mpt.parallel_2d_grid_map(
                _matrix_outer_product, row_items, col_items, pool=shared_pool
            )
            self.assertArraysAlmostEqual(reused_result, expected)

        # 1D row case (M=1)
        row_1 = [0]
        expected_1 = _matrix_outer_product(row_1, col_items)
        par_1 = mpt.parallel_2d_grid_map(_matrix_outer_product, row_1, col_items, num_workers=4)
        self.assertArraysAlmostEqual(par_1, expected_1)

    def test_parallel_pair_reduce_exact_equivalence(self):
        pairs = [(i, j) for i in range(12) for j in range(i)]  # 66 pairs

        seq_accum = None
        for p in pairs:
            res = _eval_pair(p)
            seq_accum = res if seq_accum is None else _reduce_dicts(seq_accum, res)

        par_accum = mpt.parallel_pair_reduce(_eval_pair, pairs, _reduce_dicts, num_workers=4)
        self.assertEqual(par_accum, seq_accum)

        with mpt.SharedWorkerPool(num_workers=4) as shared_pool:
            reused_accum = mpt.parallel_pair_reduce(
                _eval_pair, pairs, _reduce_dicts, pool=shared_pool
            )
            self.assertEqual(reused_accum, seq_accum)

    def test_parallel_pair_reduce_lpt_load_balancing(self):
        pairs = [(i, j) for i in range(10) for j in range(i)]  # 45 pairs
        # Skewed pair weights
        weights = [float(i * 10 + j) for i, j in pairs]

        seq_accum = None
        for p in pairs:
            res = _eval_pair(p)
            seq_accum = res if seq_accum is None else _reduce_dicts(seq_accum, res)

        par_accum = mpt.parallel_pair_reduce(
            _eval_pair, pairs, _reduce_dicts, num_workers=4, pair_weights=weights
        )
        self.assertEqual(par_accum, seq_accum)

    def test_parallel_pair_reduce_dynamic(self):
        pairs = [(i, j) for i in range(10) for j in range(i)]  # 45 pairs
        weights = [float(i * 10 + j) for i, j in pairs]

        seq_accum = None
        for p in pairs:
            res = _eval_pair(p)
            seq_accum = res if seq_accum is None else _reduce_dicts(seq_accum, res)

        dyn_accum = mpt.parallel_pair_reduce(
            _eval_pair, pairs, _reduce_dicts, num_workers=4, pair_weights=weights, dynamic=True
        )
        self.assertEqual(dyn_accum, seq_accum)

        with mpt.SharedWorkerPool(num_workers=4) as pool:
            pool_dyn_accum = mpt.parallel_pair_reduce(
                _eval_pair, pairs, _reduce_dicts, pool=pool, pair_weights=weights, dynamic=True
            )
            self.assertEqual(pool_dyn_accum, seq_accum)

    def test_lpt_partition(self):
        items = ["a", "b", "c", "d", "e", "f"]
        weights = [10.0, 8.0, 7.0, 5.0, 3.0, 1.0]

        # Standard 3-bin partition
        bins = mpt.lpt_partition(items, weights, 3)
        self.assertEqual(len(bins), 3)
        all_partitioned = [item for b in bins for item in b]
        self.assertEqual(sorted(all_partitioned), sorted(items))

        # Single bin
        single = mpt.lpt_partition(items, weights, 1)
        self.assertEqual(single, [items])

        # Empty items
        self.assertEqual(mpt.lpt_partition([], [], 3), [])

        # Validation errors
        with self.assertRaises(ValueError):
            mpt.lpt_partition(items, weights, 0)
        with self.assertRaises(ValueError):
            mpt.lpt_partition(items, weights, -2)
        with self.assertRaises(ValueError):
            mpt.lpt_partition(items, [1.0, 2.0], 3)

    def test_shared_worker_pool_process_id_reuse(self):
        with mpt.SharedWorkerPool(num_workers=4) as pool:
            pids_1 = set(pool.map(_get_pid, list(range(16))))
            pids_2 = set(pool.map(_get_pid, list(range(16))))
            self.assertEqual(pids_1, pids_2)
            self.assertEqual(len(pids_1), 4)

    def test_get_available_system_memory(self):
        avail = mpt.get_available_system_memory()
        self.assertIsInstance(avail, int)
        self.assertGreater(avail, 0)

    def test_warn_if_insufficient_memory(self):
        avail = mpt.get_available_system_memory()

        # Sufficient memory -> no warning
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always")
            mpt.warn_if_insufficient_memory(avail // 2, available_bytes=avail, task_name="test_ok")
            self.assertEqual(len(recorded), 0)

        # Insufficient memory -> ResourceWarning without exception
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always")
            mpt.warn_if_insufficient_memory(avail + 10**9, available_bytes=avail, task_name="test_oom")
            self.assertEqual(len(recorded), 1)
            self.assertTrue(issubclass(recorded[-1].category, ResourceWarning))
            self.assertIn("Estimated peak memory footprint", str(recorded[-1].message))

    def test_multiprocessing_pool_compatibility(self):
        row_items = list(range(10))
        col_items = list(range(12))
        expected_mat = _matrix_outer_product(row_items, col_items)
        pairs = [(i, i + 1) for i in range(12)]
        seq_accum = None
        for p in pairs:
            res = _eval_pair(p)
            seq_accum = res if seq_accum is None else _reduce_dicts(seq_accum, res)

        with mp.Pool(processes=2) as mp_pool:
            res_mat = mpt.parallel_2d_grid_map(
                _matrix_outer_product, row_items, col_items, pool=mp_pool
            )
            np.testing.assert_allclose(res_mat, expected_mat)

            res_pair = mpt.parallel_pair_reduce(
                _eval_pair, pairs, _reduce_dicts, pool=mp_pool
            )
            self.assertEqual(res_pair, seq_accum)

    def test_resolve_pool_workers(self):
        with mpt.SharedWorkerPool(num_workers=2) as shared_pool:
            self.assertEqual(mpt.resolve_pool_workers(shared_pool), 2)
        with mp.Pool(processes=2) as mp_pool:
            self.assertEqual(mpt.resolve_pool_workers(mp_pool), 2)
        with self.assertRaises(TypeError):
            mpt.resolve_pool_workers("not_a_pool")

    def test_workers_less_than_one_raises_error(self):
        row_items = list(range(4))
        col_items = list(range(4))
        with self.assertRaises(ValueError):
            mpt.parallel_2d_grid_map(_matrix_outer_product, row_items, col_items, num_workers=0)
        with self.assertRaises(ValueError):
            mpt.parallel_2d_grid_map(_matrix_outer_product, row_items, col_items, num_workers=-2)
        with self.assertRaises(ValueError):
            mpt.parallel_pair_reduce(_eval_pair, [(1, 2)], _reduce_dicts, num_workers=0)
        with self.assertRaises(ValueError):
            mpt.compute_2d_worker_grid(4, 4, num_workers=0)

        # Invalid pool type raises TypeError
        class BadPool:
            num_workers = 0
        with self.assertRaises(TypeError):
            mpt.parallel_2d_grid_map(_matrix_outer_product, row_items, col_items, pool=BadPool())
        with self.assertRaises(TypeError):
            mpt.parallel_pair_reduce(_eval_pair, [(1, 2)], _reduce_dicts, pool=BadPool())

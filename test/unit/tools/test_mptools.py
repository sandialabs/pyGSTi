import contextlib
import math
import multiprocessing as mp
import multiprocessing.pool as _mppool
import os
import shutil
import sys
import tempfile
import time
import unittest
import unittest.mock as mock
import warnings
import numpy as np

import pygsti.tools.mptools as mpt
from pygsti.tools.exceptions import InsufficientMemoryWarning
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


def _slow_get_pid(_=None):
    time.sleep(0.01)
    return os.getpid()


mpt_real_quota = mpt._cgroup_cpu_quota  # the unpatched function, for tests that patch the module attribute


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
        self.assertEqual(mpt.determine_num_workers(16), 16)
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

    # ----- worker-count resolution (affinity, SMT, cgroup quotas) -----
    # These tests substitute a fake `psutil` through `sys.modules` (so they behave identically whether
    # or not the optional psutil is installed), fake the affinity mask, and read cgroup and CPU
    # topology files from temporary directory trees.

    def _fake_psutil(self, logical, physical, affinity=None):
        logical_count, physical_count = logical, physical
        fake = mock.Mock()
        fake.cpu_count.side_effect = lambda logical=True: logical_count if logical else physical_count
        fake.Process.return_value.cpu_affinity.return_value = list(range(affinity)) if affinity else []
        return fake

    def _patched_hardware(self, logical, physical, affinity=None, psutil_available=True, topology_cores=None):
        """
        Context manager faking the machine: SMT layout, affinity mask size, and psutil availability.
        The sysfs topology lookup is disabled (as on platforms without it) unless `topology_cores`, the
        number of physical cores it would report for the affinity mask, is given.
        """
        stack = contextlib.ExitStack()
        mpt._logical_to_physical_cpu_ratio.cache_clear()
        stack.callback(mpt._logical_to_physical_cpu_ratio.cache_clear)
        stack.enter_context(mock.patch.dict(sys.modules, {'psutil': self._fake_psutil(logical, physical)
                                                          if psutil_available else None}))
        allowed = affinity if affinity is not None else logical
        stack.enter_context(mock.patch.object(mpt._os, 'process_cpu_count', create=True, return_value=allowed))
        stack.enter_context(mock.patch.object(mpt, '_cgroup_cpu_quota', return_value=None))
        stack.enter_context(mock.patch.object(mpt, '_allowed_physical_core_count', return_value=topology_cores))
        return stack

    def test_available_cpu_count_respects_smt_affinity_and_cgroup_quota(self):
        # 64 logical CPUs with two-way SMT = 32 physical cores
        with self._patched_hardware(64, 32):
            self.assertEqual(mpt.available_cpu_count(), 32)
        # a Slurm-style allocation of 8 logical CPUs (4 whole cores) on that machine: the affinity mask limits
        # the result (this used to be ignored whenever psutil was installed)
        with self._patched_hardware(64, 32, affinity=8):
            self.assertEqual(mpt.available_cpu_count(), 4)
            self.assertEqual(mpt.determine_num_workers("auto"), 4)
        with self._patched_hardware(64, 32, affinity=2):
            self.assertEqual(mpt.available_cpu_count(), 1)
            self.assertEqual(mpt.determine_num_workers("auto"), 1)
        # a cgroup quota below the affinity limit applies
        with self._patched_hardware(64, 32), mock.patch.object(mpt, '_cgroup_cpu_quota', return_value=4):
            self.assertEqual(mpt.available_cpu_count(), 4)
            self.assertEqual(mpt.determine_num_workers("auto"), 4)
        # ... but never raises the count
        with self._patched_hardware(8, 8), mock.patch.object(mpt, '_cgroup_cpu_quota', return_value=64):
            self.assertEqual(mpt.available_cpu_count(), 8)
        # without psutil there is no SMT information: every logical CPU counts, and the affinity mask still applies
        with self._patched_hardware(16, 8, psutil_available=False):
            self.assertEqual(mpt.available_cpu_count(), 16)
        with self._patched_hardware(16, 8, affinity=4, psutil_available=False):
            self.assertEqual(mpt.available_cpu_count(), 4)

    def test_available_cpu_count_uses_topology_where_available(self):
        # the topology count replaces the machine-wide SMT-ratio estimate (here 4 CPUs on 4 distinct cores)
        with self._patched_hardware(64, 32, affinity=4, topology_cores=4):
            self.assertEqual(mpt.available_cpu_count(), 4)
        with self._patched_hardware(64, 32, affinity=4, psutil_available=False, topology_cores=4):
            self.assertEqual(mpt.available_cpu_count(), 4)
        # the cgroup quota still applies, and never raises the count
        with self._patched_hardware(64, 32, affinity=8, topology_cores=4), \
                mock.patch.object(mpt, '_cgroup_cpu_quota', return_value=2):
            self.assertEqual(mpt.available_cpu_count(), 2)
        with self._patched_hardware(64, 32, affinity=8, topology_cores=4), \
                mock.patch.object(mpt, '_cgroup_cpu_quota', return_value=16):
            self.assertEqual(mpt.available_cpu_count(), 4)

    def test_parse_cpu_list(self):
        self.assertEqual(mpt._parse_cpu_list("0"), {0})
        self.assertEqual(mpt._parse_cpu_list("0-3,8,10-11\n"), {0, 1, 2, 3, 8, 10, 11})
        self.assertEqual(mpt._parse_cpu_list(""), set())
        with self.assertRaises(ValueError):
            mpt._parse_cpu_list("a-b")

    def _write_cpu_topology(self, siblings, name="core_cpus_list"):
        """A fake sysfs CPU tree in which `siblings` lists the SMT siblings (as a CPU list string) of each CPU."""
        return self._write_cgroup_tree({f"cpu{cpu}/topology/{name}": text for cpu, text in siblings.items()})

    def test_physical_core_count_from_topology(self):
        # 4 cores with two-way SMT, the siblings of core c being CPUs c and c + 4 (as on many Intel machines)
        smt = {cpu: f"{cpu % 4},{cpu % 4 + 4}" for cpu in range(8)}
        root = self._write_cpu_topology(smt)
        count = lambda cpus: mpt._physical_core_count(cpus, cpu_root=root)
        self.assertEqual(count(range(8)), 4)         # everything
        self.assertEqual(count([0, 4]), 1)           # both threads of one core
        self.assertEqual(count([0, 1, 4, 5]), 2)     # whole cores
        self.assertEqual(count([0, 1, 2, 3]), 4)     # one thread per core: a logical/SMT-ratio estimate gives 2
        self.assertEqual(count([1, 2, 3, 4]), 4)     # one thread on each core
        self.assertEqual(count([0, 1, 5]), 2)        # a partially covered core still counts
        self.assertIsNone(count([]))

        # adjacent numbering of siblings (as on many AMD machines)
        adjacent = {cpu: f"{cpu - cpu % 2}-{cpu - cpu % 2 + 1}" for cpu in range(8)}
        root = self._write_cpu_topology(adjacent)
        self.assertEqual(count([0, 2, 4, 6]), 4)
        self.assertEqual(count([0, 1, 2, 3]), 2)

        # a hybrid CPU: two SMT cores (CPUs 0-3) and four single-threaded cores (CPUs 4-7)
        hybrid = {0: "0-1", 1: "0-1", 2: "2-3", 3: "2-3", 4: "4", 5: "5", 6: "6", 7: "7"}
        root = self._write_cpu_topology(hybrid)
        self.assertEqual(count(range(8)), 6)
        self.assertEqual(count([0, 1, 4, 5]), 3)

        # kernels before 5.3 only have thread_siblings_list
        root = self._write_cpu_topology(smt, name="thread_siblings_list")
        self.assertEqual(count([0, 1, 4, 5]), 2)

        # topology that is missing for any CPU, or malformed, is reported as unknown
        root = self._write_cpu_topology({0: "0,4", 1: "1,5"})
        self.assertIsNone(count([0, 1, 2]))
        root = self._write_cpu_topology({0: "garbage"})
        self.assertIsNone(count([0]))
        self.assertIsNone(mpt._physical_core_count([0], cpu_root="/nonexistent"))

    def test_allowed_physical_core_count(self):
        mask = {0, 1, 2, 3}
        with mock.patch.object(mpt._sys, 'platform', 'linux'), \
                mock.patch.object(mpt._os, 'sched_getaffinity', create=True, return_value=mask), \
                mock.patch.object(mpt, '_allowed_logical_cpu_count', return_value=4), \
                mock.patch.object(mpt, '_physical_core_count', return_value=3) as count:
            self.assertEqual(mpt._allowed_physical_core_count(), 3)
            self.assertEqual(count.call_args.args[0], mask)  # asked about the CPUs in the affinity mask
            # unknown topology: the caller falls back to the logical CPU count
            count.return_value = None
            self.assertIsNone(mpt._allowed_physical_core_count())
            count.return_value = 3
            # an overridden CPU count (the mask no longer describes the usable CPUs)
            with mock.patch.object(mpt, '_allowed_logical_cpu_count', return_value=2):
                self.assertIsNone(mpt._allowed_physical_core_count())
            # no readable affinity mask
            with mock.patch.object(mpt._os, 'sched_getaffinity', create=True, side_effect=OSError):
                self.assertIsNone(mpt._allowed_physical_core_count())
            # not Linux
            with mock.patch.object(mpt._sys, 'platform', 'win32'):
                self.assertIsNone(mpt._allowed_physical_core_count())

    @unittest.skipUnless(sys.platform.startswith("linux") and os.path.isdir("/sys/devices/system/cpu/cpu0/topology"),
                         "needs the Linux sysfs CPU topology")
    def test_available_cpu_count_on_this_machine_matches_topology(self):
        mpt._logical_to_physical_cpu_ratio.cache_clear()
        self.addCleanup(mpt._logical_to_physical_cpu_ratio.cache_clear)
        cores = mpt._allowed_physical_core_count()
        self.assertIsNotNone(cores)
        self.assertGreaterEqual(cores, 1)
        self.assertLessEqual(cores, len(os.sched_getaffinity(0)))
        with mock.patch.object(mpt, '_cgroup_cpu_quota', return_value=None):
            self.assertEqual(mpt.available_cpu_count(), cores)

    def test_available_cpu_count_affinity_fallbacks(self):
        # falls back from os.process_cpu_count to sched_getaffinity, then psutil's affinity, then os.cpu_count
        with self._patched_hardware(16, 16):
            with mock.patch.object(mpt._os, 'process_cpu_count', create=True, return_value=None), \
                    mock.patch.object(mpt._os, 'sched_getaffinity', create=True, return_value={0, 1, 2}):
                self.assertEqual(mpt._allowed_logical_cpu_count(), 3)
            with mock.patch.object(mpt._os, 'process_cpu_count', create=True, side_effect=OSError), \
                    mock.patch.object(mpt._os, 'sched_getaffinity', create=True, side_effect=OSError), \
                    mock.patch.dict(sys.modules, {'psutil': self._fake_psutil(16, 16, affinity=5)}):
                self.assertEqual(mpt._allowed_logical_cpu_count(), 5)
            with mock.patch.object(mpt._os, 'process_cpu_count', create=True, return_value=None), \
                    mock.patch.object(mpt._os, 'sched_getaffinity', create=True, return_value=set()), \
                    mock.patch.dict(sys.modules, {'psutil': None}), \
                    mock.patch.object(mpt._os, 'cpu_count', return_value=7):
                self.assertEqual(mpt._allowed_logical_cpu_count(), 7)

    def test_determine_num_workers_auto_uses_all_available_cores_by_default(self):
        for cores in (1, 2, 3, 4, 16):
            with self._patched_hardware(cores, cores):
                self.assertEqual(mpt.determine_num_workers("auto"), cores)  # no core is reserved by default

    def test_determine_num_workers_reserve_cores(self):
        with self._patched_hardware(16, 16):
            self.assertEqual(mpt.determine_num_workers("auto", reserve_cores=1), 15)
            self.assertEqual(mpt.determine_num_workers("auto", reserve_cores=4), 12)
            self.assertEqual(mpt.determine_num_workers("auto", reserve_cores=15), 1)
            self.assertEqual(mpt.determine_num_workers("auto", reserve_cores=100), 1)  # always at least one worker
        with self._patched_hardware(2, 2):
            self.assertEqual(mpt.determine_num_workers("auto", reserve_cores=1), 1)
        # applies only to "auto"
        with self._patched_hardware(16, 16):
            self.assertEqual(mpt.determine_num_workers(8, reserve_cores=4), 8)
            self.assertEqual(mpt.determine_num_workers(None, reserve_cores=4), 1)
        for bad in (-1, 1.5, "1", None):
            with self.assertRaises(ValueError):
                mpt.determine_num_workers("auto", reserve_cores=bad)

    def test_determine_num_workers_max_limit_caps_the_final_count(self):
        with self._patched_hardware(16, 16):
            self.assertEqual(mpt.determine_num_workers("auto", max_limit=4), 4)
            self.assertEqual(mpt.determine_num_workers("auto", max_limit=100), 16)
            self.assertEqual(mpt.determine_num_workers("auto", max_limit=1), 1)
            # max_limit caps the count left after reserving cores (it does not deduct them again)
            self.assertEqual(mpt.determine_num_workers("auto", max_limit=4, reserve_cores=1), 4)
            self.assertEqual(mpt.determine_num_workers("auto", max_limit=100, reserve_cores=1), 15)
            self.assertEqual(mpt.determine_num_workers("auto", max_limit=15, reserve_cores=4), 12)

    def _write_cgroup_tree(self, files):
        root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, root, True)
        for rel, text in files.items():
            path = os.path.join(root, rel)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as f:
                f.write(text)
        return root

    def _quota(self, files, proc_cgroup):
        root = self._write_cgroup_tree(files)
        proc = os.path.join(root, "proc_self_cgroup")
        with open(proc, "w") as f:
            f.write(proc_cgroup)
        return mpt._cgroup_cpu_quota(cgroup_root=root, proc_cgroup_file=proc)

    def test_cgroup_cpu_quota_v2(self):
        # a container: the process sits at the root of its cgroup namespace
        self.assertEqual(self._quota({"cpu.max": "400000 100000\n"}, "0::/\n"), 4)
        self.assertIsNone(self._quota({"cpu.max": "max 100000\n"}, "0::/\n"))
        self.assertEqual(self._quota({"cpu.max": "250000 100000\n"}, "0::/\n"), 2)       # fractional quotas round down
        self.assertEqual(self._quota({"cpu.max": "50000 100000\n"}, "0::/\n"), 1)        # but never below one CPU
        # a nested cgroup (bare-metal Slurm/systemd): the quota of the process's own cgroup applies ...
        files = {"cpu.max": "max 100000", "job_1/cpu.max": "max 100000", "job_1/step_0/cpu.max": "200000 100000"}
        self.assertEqual(self._quota(files, "0::/job_1/step_0\n"), 2)
        # ... as does the (smaller) quota of an ancestor, even though the cgroup itself is unlimited
        files = {"cpu.max": "max 100000", "job_1/cpu.max": "300000 100000", "job_1/step_0/cpu.max": "max 100000"}
        self.assertEqual(self._quota(files, "0::/job_1/step_0\n"), 3)
        # the smallest of the quotas along the path wins
        files = {"job_1/cpu.max": "300000 100000", "job_1/step_0/cpu.max": "800000 100000"}
        self.assertEqual(self._quota(files, "0::/job_1/step_0\n"), 3)
        # unreadable / malformed files are ignored, and a missing /proc file means no information
        self.assertIsNone(self._quota({"cpu.max": "garbage"}, "0::/\n"))
        self.assertIsNone(self._quota({}, "0::/\n"))
        self.assertIsNone(mpt._cgroup_cpu_quota(cgroup_root="/nonexistent", proc_cgroup_file="/nonexistent/proc"))

    def test_cgroup_cpu_quota_v1(self):
        proc = "12:cpu,cpuacct:/docker/abc\n11:memory:/docker/abc\n"
        files = {"cpu/cpu.cfs_quota_us": "200000", "cpu/cpu.cfs_period_us": "100000"}
        self.assertEqual(self._quota(files, "4:cpu,cpuacct:/\n"), 2)
        files = {"cpu/docker/abc/cpu.cfs_quota_us": "300000", "cpu/docker/abc/cpu.cfs_period_us": "100000"}
        self.assertEqual(self._quota(files, proc), 3)
        files = {"cpu,cpuacct/docker/abc/cpu.cfs_quota_us": "100000", "cpu,cpuacct/docker/abc/cpu.cfs_period_us": "100000"}
        self.assertEqual(self._quota(files, proc), 1)
        # -1 means unlimited
        files = {"cpu/cpu.cfs_quota_us": "-1", "cpu/cpu.cfs_period_us": "100000"}
        self.assertIsNone(self._quota(files, "4:cpu,cpuacct:/\n"))

    @unittest.skipUnless(sys.platform.startswith("linux"), "cgroups only applicable on Linux")
    def test_available_cpu_count_cgroup_quota_is_independent_of_platform(self):
        # the quota is applied wherever one is found, and a platform without cgroups simply has none
        for platform in ('linux', 'win32', 'darwin'):
            with self.subTest(platform=platform), mock.patch.object(mpt._sys, 'platform', platform):
                with self._patched_hardware(64, 32), mock.patch.object(mpt, '_cgroup_cpu_quota', return_value=4):
                    self.assertEqual(mpt.available_cpu_count(), 4)
                with self._patched_hardware(64, 32):  # no quota
                    self.assertEqual(mpt.available_cpu_count(), 32)
        # without a readable /proc/self/cgroup (as on Windows and macOS) the lookup finds no quota
        # instead of failing
        missing = lambda: mpt_real_quota(proc_cgroup_file="/nonexistent/proc")
        with self._patched_hardware(8, 8), mock.patch.object(mpt, '_cgroup_cpu_quota', side_effect=missing) as quota:
            self.assertEqual(mpt.available_cpu_count(), 8)
            quota.assert_called_once()

    def test_determine_num_workers_uses_cgroup_quota(self):
        with self._patched_hardware(64, 64), mock.patch.object(mpt, '_cgroup_cpu_quota', return_value=2):
            self.assertEqual(mpt.available_cpu_count(), 2)
            self.assertEqual(mpt.determine_num_workers("auto"), 2)
        # on the real system the helper must not fail and respects the real affinity mask
        mpt._logical_to_physical_cpu_ratio.cache_clear()
        self.assertGreaterEqual(mpt.available_cpu_count(), 1)
        self.assertLessEqual(mpt.available_cpu_count(), mpt._allowed_logical_cpu_count())

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

    def test_shared_worker_pool_lazy_start(self):
        with mpt.SharedWorkerPool(num_workers=2, lazy=True) as pool:
            self.assertFalse(pool.is_started)
            self.assertEqual(mpt.resolve_pool_workers(pool), 2)  # known without starting workers
            self.assertEqual(pool.map(_square, [1, 2, 3]), [1, 4, 9])
            self.assertTrue(pool.is_started)
            self.assertEqual(sorted(pool.imap_unordered(_square, [1, 2])), [1, 4])
        self.assertFalse(pool.is_started)

        # An eager pool starts on entry; a pool that never started is a no-op on exit.
        with mpt.SharedWorkerPool(num_workers=2) as pool:
            self.assertTrue(pool.is_started)
        with mpt.SharedWorkerPool(num_workers=2, lazy=True) as pool:
            pass
        self.assertFalse(pool.is_started)

    def test_shared_worker_pool_requires_active_context(self):
        pool = mpt.SharedWorkerPool(num_workers=2)
        for method, args in (('map', (_square, [1])), ('starmap', (_add, [(1, 2)])),
                             ('imap_unordered', (_square, [1]))):
            with self.assertRaises(RuntimeError):
                getattr(pool, method)(*args)
        with pool:
            pool.map(_square, [1])
        with self.assertRaises(RuntimeError):  # already closed
            pool.map(_square, [1])
        with self.assertRaises(RuntimeError):  # cannot be entered while active
            with pool:
                with pool:
                    pass

        # A single-worker pool is always serial and never needs an active context.
        self.assertEqual(mpt.SharedWorkerPool(num_workers=1).map(_square, [2, 3]), [4, 9])

    def test_shared_worker_pool_terminates_on_exception(self):
        with self.assertRaises(ZeroDivisionError):
            with mpt.SharedWorkerPool(num_workers=2) as pool:
                inner = pool._pool
                terminate = mock.patch.object(inner, 'terminate', wraps=inner.terminate).start()
                close = mock.patch.object(inner, 'close', wraps=inner.close).start()
                self.addCleanup(mock.patch.stopall)
                raise ZeroDivisionError
        terminate.assert_called_once()
        close.assert_not_called()

        with mpt.SharedWorkerPool(num_workers=2) as pool:
            inner = pool._pool
            terminate = mock.patch.object(inner, 'terminate', wraps=inner.terminate).start()
            close = mock.patch.object(inner, 'close', wraps=inner.close).start()
        close.assert_called_once()
        terminate.assert_not_called()

    def test_shared_worker_pool_in_daemonic_process_runs_serially(self):
        daemonic = mock.Mock(daemon=True)
        with mock.patch.object(mpt._mp, 'current_process', return_value=daemonic), \
                mpt.SharedWorkerPool(num_workers=2, lazy=True) as pool:
            with warnings.catch_warnings(record=True) as rec:
                warnings.simplefilter("always")
                self.assertEqual(pool.map(_get_pid, [0, 1]), [os.getpid()] * 2)
                pool.map(_square, [1])
            self.assertFalse(pool.is_started)
            self.assertEqual(sum(issubclass(w.category, RuntimeWarning) for w in rec), 1)

    def test_pool_scope(self):
        with mpt.SharedWorkerPool(num_workers=2, lazy=True) as supplied:
            with mpt._pool_scope(4, supplied) as scoped:
                self.assertIs(scoped, supplied)
        for n in (None, 1):
            with mpt._pool_scope(n, None) as scoped:
                self.assertIsNone(scoped)
        with mpt._pool_scope(2, None) as scoped:
            self.assertIsInstance(scoped, mpt.SharedWorkerPool)
            self.assertFalse(scoped.is_started)
            scoped.map(_square, [1])
            self.assertTrue(scoped.is_started)
        self.assertFalse(scoped.is_started)

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

        # Standard 3-bin partition: greedy LPT places items into min-load bin
        bins = mpt.lpt_partition(items, weights, 3)
        self.assertEqual(len(bins), 3)
        self.assertEqual(bins, [["a", "f"], ["b", "e"], ["c", "d"]])

        # Verify load balance and weights
        weight_map = dict(zip(items, weights))
        bin_weights = [sum(weight_map[item] for item in b) for b in bins]
        self.assertEqual(bin_weights, [11.0, 11.0, 12.0])

        all_partitioned = [item for b in bins for item in b]
        self.assertEqual(sorted(all_partitioned), sorted(items))

        # Single bin
        single = mpt.lpt_partition(items, weights, 1)
        self.assertEqual(single, [items])

        # More bins than items (empty bins filtered out)
        more_bins = mpt.lpt_partition(["a", "b"], [2.0, 1.0], 5)
        self.assertEqual(more_bins, [["a"], ["b"]])

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
            # Map across workers; 10ms sleep prevents fast workers from draining all tasks under CPU contention
            pids_1 = set(pool.map(_slow_get_pid, list(range(16)), chunksize=1))
            pids_2 = set(pool.map(_slow_get_pid, list(range(16)), chunksize=1))
            self.assertEqual(pids_1, pids_2)
            self.assertEqual(len(pids_1), 4)
            if isinstance(pool._pool, _mppool.Pool):
                pool_pids = {w.pid for w in pool._pool._pool}
                self.assertEqual(pids_1, pool_pids)

    def test_get_available_system_memory(self):
        # None is a legitimate answer on a platform without psutil or a /proc fallback
        avail = mpt.get_available_system_memory()
        if avail is None:
            self.assertFalse(sys.platform.startswith("linux"))
        else:
            self.assertIsInstance(avail, int)
            self.assertGreater(avail, 0)

    def test_get_available_system_memory_backends(self):
        fake_psutil = mock.Mock()
        fake_psutil.virtual_memory.return_value.available = 12345
        with mock.patch.dict(sys.modules, {'psutil': fake_psutil}):
            self.assertEqual(mpt.get_available_system_memory(), 12345)
        # without psutil: the Linux /proc fallback ...
        with mock.patch.dict(sys.modules, {'psutil': None}):
            avail = mpt.get_available_system_memory()
            if sys.platform.startswith("linux"):
                self.assertIsInstance(avail, int)
                self.assertGreater(avail, 0)
            # ... and no fallback at all on other platforms
            with mock.patch.object(mpt._sys, 'platform', 'darwin'):
                self.assertIsNone(mpt.get_available_system_memory())

    def test_warn_if_insufficient_memory_when_memory_is_unknown(self):
        with mock.patch.object(mpt, 'get_available_system_memory', return_value=None):
            with warnings.catch_warnings(record=True) as recorded:
                warnings.simplefilter("always")
                mpt.warn_if_insufficient_memory(10**9, task_name="test_unknown")
        self.assertEqual(len(recorded), 1)
        self.assertTrue(issubclass(recorded[0].category, InsufficientMemoryWarning))
        self.assertIn("Unable to resolve available system memory", str(recorded[0].message))

    def test_warn_if_insufficient_memory(self):
        avail = mpt.get_available_system_memory()

        # Sufficient memory -> no warning
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always")
            mpt.warn_if_insufficient_memory(avail // 2, available_bytes=avail, task_name="test_ok")
            self.assertEqual(len(recorded), 0)

        # Insufficient memory -> InsufficientMemoryWarning (visible under default filters) without exception
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always")
            mpt.warn_if_insufficient_memory(avail + 10**9, available_bytes=avail, task_name="test_oom")
            self.assertEqual(len(recorded), 1)
            self.assertTrue(issubclass(recorded[-1].category, InsufficientMemoryWarning))
            self.assertFalse(issubclass(recorded[-1].category, ResourceWarning))
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

"""
Functions for working with Python multiprocessing (more than just map)
"""
#***************************************************************************************************
# Copyright 2015, 2019, 2025 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

import contextlib as _contextlib
import functools as _functools
import itertools as _itertools
import math as _math
import multiprocessing as _mp
import multiprocessing.pool as _mppool
import os as _os
import sys as _sys
import time as _time
import warnings as _warnings
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as _np

from pygsti.tools.exceptions import InsufficientMemoryWarning as _InsufficientMemoryWarning


# ===================================================================================================
# Existing Starmap Helper (Preserved for Backward Compatibility)
# ===================================================================================================

# Modified from https://stackoverflow.com/a/53173433
def starmap_with_kwargs(fn, num_runs, num_processors, args_list, kwargs_list):
    # If only one processor, run serial (makes it easier to profile and avoids any Pool overhead)
    if num_processors == 1:
        return [fn(*args_list[i], **kwargs_list[i]) for i in range(num_runs)]

    with _mp.Pool(num_processors) as pool:
        args_for_starmap = zip(
            _itertools.repeat(fn, num_runs),
            args_list,
            kwargs_list)
        return pool.starmap(_apply_args_and_kwargs, args_for_starmap)


def _apply_args_and_kwargs(fn, args, kwargs):
    return fn(*args, **kwargs)


# ===================================================================================================
# Worker Process Pre-Warming & Module Level Helpers
# ===================================================================================================

def _prewarm_worker(packages: Tuple[str, ...]):
    """
    Top-level worker initializer dynamically importing caller-specified packages.
    Eliminates cold-start import penalties on 'spawn' platforms (macOS, Windows, Python 3.14+)
    without hardcoding domain-specific dependencies into mptools.
    """
    import importlib
    for pkg in packages:
        try:
            importlib.import_module(pkg)
        except ImportError:
            pass


def _noop_ipc_ping(x):
    """Trivial task for measuring IPC dispatch latency."""
    return x


# ===================================================================================================
# Worker Count & Hardware / Container Resolution
# ===================================================================================================

@_functools.lru_cache(maxsize=1)
def _logical_to_physical_cpu_ratio() -> float:
    """
    The (hardware-determined, hence cached) ratio of logical CPUs to physical cores, e.g. 2 on a
    machine with two-way SMT / hyper-threading, or 1 if it is unknown (psutil unavailable).
    """
    try:
        import psutil
        logical, physical = psutil.cpu_count(logical=True), psutil.cpu_count(logical=False)
        if logical and physical and logical >= physical:
            return logical / physical
    except Exception:
        pass
    return 1.0


def _allowed_logical_cpu_count() -> int:
    """
    The number of logical CPUs this process is permitted to run on: its CPU affinity mask (as set
    by `taskset`, Slurm, `numactl`, `docker --cpuset-cpus`, ...) where the platform exposes it, or
    else the total number of logical CPUs.
    """
    n = None
    if hasattr(_os, "process_cpu_count"):  # Python >= 3.13, respects the affinity mask
        try:
            n = _os.process_cpu_count()
        except Exception:
            n = None
    if not n and hasattr(_os, "sched_getaffinity"):  # Linux
        try:
            n = len(_os.sched_getaffinity(0))
        except Exception:
            n = None
    if not n:  # Windows (and Linux) via psutil
        try:
            import psutil
            n = len(psutil.Process().cpu_affinity())
        except Exception:
            n = None
    return max(1, int(n or _os.cpu_count() or 1))


def _cgroup_cpu_quota(cgroup_root: str = "/sys/fs/cgroup", proc_cgroup_file: str = "/proc/self/cgroup") -> Optional[int]:
    """
    The whole number of CPUs this process is limited to by Linux cgroup CPU quotas (as set by
    `docker --cpus`, Kubernetes CPU limits, systemd or Slurm), or None if no quota applies or the
    platform has no cgroups.

    The cgroup the process belongs to is read from `proc_cgroup_file`, and the smallest quota
    found on that cgroup and each of its ancestors is used, since a limit on any ancestor applies.
    cgroups v2 (`cpu.max`) and v1 (`cpu.cfs_quota_us` / `cpu.cfs_period_us`) are both supported.
    A quota of less than one CPU is rounded up to one.
    """
    try:
        with open(proc_cgroup_file, "r") as f:
            entries = [line.rstrip("\n").split(":", 2) for line in f if line.count(":") >= 2]
    except OSError:
        return None

    def ancestors(path):  # '/a/b/c' -> ['/a/b/c', '/a/b', '/a', '']
        parts = [p for p in path.split("/") if p]
        return ["/" + "/".join(parts[:i]) if i else "" for i in range(len(parts), -1, -1)]

    def read(path):
        try:
            with open(path, "r") as f:
                return f.read().strip()
        except OSError:
            return None

    limits = []
    for _, controllers, path in entries:
        names = controllers.split(",")
        if controllers == "":  # cgroups v2 (unified hierarchy)
            for anc in ancestors(path):
                text = read(f"{cgroup_root}{anc}/cpu.max")
                if text:
                    parts = text.split()
                    if len(parts) >= 2 and parts[0] != "max":
                        try:
                            quota, period = float(parts[0]), float(parts[1])
                            if quota > 0 and period > 0:
                                limits.append(quota / period)
                        except ValueError:
                            pass
        elif "cpu" in names:  # cgroups v1
            for mount in ("cpu", "cpu,cpuacct", "cpuacct,cpu"):
                for anc in ancestors(path):
                    quota_text = read(f"{cgroup_root}/{mount}{anc}/cpu.cfs_quota_us")
                    period_text = read(f"{cgroup_root}/{mount}{anc}/cpu.cfs_period_us")
                    if quota_text and period_text:
                        try:
                            quota, period = float(quota_text), float(period_text)
                            if quota > 0 and period > 0:  # a quota of -1 means unlimited
                                limits.append(quota / period)
                        except ValueError:
                            pass
    if not limits:
        return None
    return max(1, int(_math.floor(min(limits))))


def available_cpu_count() -> int:
    """
    The number of physical CPU cores that this process can actually use.

    Three separate limits are combined, since each is imposed by different tools:

    1. the hardware: SMT / hyper-threading logical CPUs are not counted separately from the
       physical core they share (assuming a uniform number of logical CPUs per core, as reported by
       `psutil`; without `psutil` all logical CPUs are counted);
    2. the CPU affinity mask: the subset of CPUs the operating system permits this process to run
       on (`taskset`, Slurm allocations, `numactl`, `docker --cpuset-cpus`);
    3. the cgroup CPU quota (Linux): the amount of CPU time the process may use (`docker --cpus`,
       Kubernetes CPU limits).

    Returns
    -------
    int
        The number of usable physical cores (>= 1).
    """
    n = max(1, int(_allowed_logical_cpu_count() / _logical_to_physical_cpu_ratio()))
    if _sys.platform.startswith("linux"):
        quota = _cgroup_cpu_quota()
        if quota is not None:
            n = min(n, quota)
    return n


def determine_num_workers(
    num_workers: Union[int, str, None] = "auto",
    max_limit: Optional[int] = None,
) -> int:
    """
    Resolve the number of worker processes to allocate for a multiprocessing operation.

    Parameters
    ----------
    num_workers : int, str, or None, optional
        - None : Returns 1 (strictly sequential execution without multiprocessing).
        - int : Exact requested worker count, clamped by `max_limit` if provided.
        - "auto" : The number of cores usable by this process, as given by `available_cpu_count`
          (which accounts for SMT / hyper-threading, the CPU affinity mask, and Linux cgroup CPU
          quotas). On systems with >= 4 usable cores, one core is reserved for the main
          orchestration process to prevent core oversubscription and CFS throttling during
          streaming IPC reduction.
    max_limit : int, optional
        Maximum upper bound on workers to return. Applied last, after the reservation of a core
        for the main process.

    Returns
    -------
    int
        An integer >= 1 specifying the number of worker processes to use.
    """
    if num_workers is None:
        return 1

    if isinstance(num_workers, (int, _np.integer)):
        n = int(num_workers)
        if n < 1:
            raise ValueError(f"num_workers must be an integer >= 1, got {num_workers}.")
        if max_limit is not None and max_limit > 0:
            return min(n, int(max_limit))
        return n

    if isinstance(num_workers, str):
        if num_workers.lower() == "auto":
            n = available_cpu_count()
            if n >= 4:
                n -= 1
            if max_limit is not None and max_limit > 0:
                n = min(n, int(max_limit))
            return max(1, n)

    raise ValueError(f"Invalid num_workers specification: {num_workers!r}. Expected int, None, or 'auto'.")


# Cache for measured IPC dispatch latencies: (workers, start_method) -> float (seconds)
_IPC_LATENCY_CACHE: Dict[Tuple[int, Optional[str]], float] = {}


def measure_multiprocessing_ipc_latency(
    num_workers: Union[int, str] = "auto",
    start_method: Optional[str] = None,
) -> float:
    """
    Measure local round-trip inter-process communication dispatch latency (T_IPC)
    in seconds for the active start method ('fork' vs. 'spawn'). Caches results
    for future reference by `mptools` utilities.

    Parameters
    ----------
    num_workers : int or str, optional
        Target worker count (defaults to "auto").
    start_method : str, optional
        Start method to evaluate ('fork', 'spawn', 'forkserver'). Defaults to system default.

    Returns
    -------
    float
        Average round-trip dispatch latency in seconds.
    """
    resolved_workers = determine_num_workers(num_workers)
    active_method = start_method or _mp.get_start_method()
    cache_key = (resolved_workers, active_method)

    if cache_key in _IPC_LATENCY_CACHE:
        return _IPC_LATENCY_CACHE[cache_key]

    test_workers = min(resolved_workers, 4)
    if test_workers < 1:
        raise ValueError(f"Worker count must be at least 1, got {test_workers}.")
    if test_workers == 1:
        # Sequential: negligible latency
        _IPC_LATENCY_CACHE[cache_key] = 0
        return 0

    try:
        ctx = _mp.get_context(active_method)
        with ctx.Pool(processes=test_workers) as pool:
            # Warm up
            pool.map(_noop_ipc_ping, range(test_workers))
            t0 = _time.perf_counter()
            n_iters = 8
            for _ in range(n_iters):
                pool.map(_noop_ipc_ping, range(test_workers))
            t1 = _time.perf_counter()
            ipc_s = (t1 - t0) / n_iters
    except Exception:
        # Fallback conservative nominal latency if pool creation fails in test environment
        ipc_s = 0.00025 if active_method == "spawn" else 0.00010

    _IPC_LATENCY_CACHE[cache_key] = ipc_s
    return ipc_s


# ===================================================================================================
# 2D Worker Grid Computation
# ===================================================================================================

def compute_2d_worker_grid(
    n_rows: int,
    n_cols: int,
    num_workers: int,
) -> Tuple[int, int]:
    """
    Compute an optimal 2D worker process grid (P_rows, P_cols) such that
    P_rows * P_cols <= num_workers, maximizing worker utilization and balancing workload.

    Parameters
    ----------
    n_rows : int
        Number of rows.
    n_cols : int
        Number of columns.
    num_workers : int
        Total worker processes available.

    Returns
    -------
    tuple of (int, int)
        (P_rows, P_cols) process grid factors.
    """
    if num_workers < 1:
        raise ValueError(f"num_workers must be at least 1, got {num_workers}.")
    if num_workers == 1 or (n_rows <= 1 and n_cols <= 1):
        return (1, 1)
    if n_rows <= 1:
        return (1, min(n_cols, num_workers))
    if n_cols <= 1:
        return (min(n_rows, num_workers), 1)

    candidates = []
    max_pr = min(n_rows, num_workers)
    for pr in range(1, max_pr + 1):
        pc = min(n_cols, num_workers // pr)
        if pc < 1:
            continue
        utilized = pr * pc
        candidates.append((utilized, pr, pc))

    if not candidates:
        return (1, 1)

    max_util = max(c[0] for c in candidates)
    best_candidates = [c for c in candidates if c[0] == max_util]

    def _score(c):
        _, pr, pc = c
        # Prefer pr that cleanly divides n_rows, then balance pr and pc
        rem = n_rows % pr
        balance = abs(pr - pc)
        return (rem, balance)

    best_candidates.sort(key=_score)
    return (best_candidates[0][1], best_candidates[0][2])


# ===================================================================================================
# Reusable Worker Pool Context Manager
# ===================================================================================================

class SharedWorkerPool:
    """
    Reusable worker pool context manager with caller-configurable worker pre-warming.

    Eliminates process spin-up and teardown overhead across sequential algorithm stages:
    create the pool once in a `with` block and pass it (via the `pool` argument) to every
    function that should share its worker processes.

    Lifecycle: a pool is *created* by the constructor, becomes *active* on entering its `with`
    block, has its worker processes *started* either on entry (default) or on first use
    (`lazy=True`), and is *closed* on leaving the block. Outside of its `with` block a pool with
    more than one worker cannot be used; calling `map`, `starmap` or `imap_unordered` then raises
    a `RuntimeError` rather than silently running serially. A pool with a single worker never
    starts any processes and always executes serially in the calling process.

    Parameters
    ----------
    num_workers : int, str, or None, optional
        Target worker process count (defaults to "auto").
    start_method : str, optional
        Multiprocessing start method ('fork', 'spawn', 'forkserver').
    prewarm_packages : sequence of str, optional
        List or tuple of module names to eagerly import in each worker process upon pool
        initialization. By default (None), no implicit packages are imported.
    initializer : callable, optional
        Custom worker initializer callable. Overrides `prewarm_packages` if provided.
    initargs : tuple, optional
        Arguments passed to `initializer`.
    lazy : bool, optional (default False)
        If False, worker processes are started on entering the `with` block. If True, they are
        started only when work is first dispatched to the pool, so a multi-stage workflow whose
        stages all decide to run serially never pays the process start-up cost.

    Notes
    -----
    When used from inside a daemonic process (e.g. from within a worker of another pool), which
    is not permitted to start child processes, work dispatched to the pool is executed serially
    in the calling process and a `RuntimeWarning` is issued once per pool.
    """

    def __init__(
        self,
        num_workers: Union[int, str, None] = "auto",
        start_method: Optional[str] = None,
        prewarm_packages: Optional[Sequence[str]] = None,
        initializer: Optional[Callable] = None,
        initargs: tuple = (),
        lazy: bool = False,
    ):
        self.num_workers = determine_num_workers(num_workers)
        self.start_method = start_method
        self.prewarm_packages = prewarm_packages
        self.initializer = initializer
        self.initargs = initargs
        self.lazy = lazy
        self._pool = None
        self._active = False
        self._warned_daemonic = False

    def __enter__(self):
        if self._active:
            raise RuntimeError("SharedWorkerPool is already active; it cannot be entered twice.")
        self._active = True
        if self.num_workers > 1 and not self.lazy:
            self._ensure_started()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._active = False
        pool, self._pool = self._pool, None
        if pool is not None:
            if exc_type is None:
                # Normal exit: let outstanding work finish, then shut the workers down.
                pool.close()
            else:
                # Exceptional exit (including KeyboardInterrupt): stop the workers immediately
                # instead of waiting for tasks whose results will never be used.
                pool.terminate()
            pool.join()

    @property
    def is_started(self) -> bool:
        """Whether worker processes are currently running for this pool."""
        return self._pool is not None

    def _start(self):
        """Create the underlying multiprocessing pool (worker processes)."""
        ctx = _mp.get_context(self.start_method)
        if self.initializer is not None:
            init_fn = self.initializer
            init_args = self.initargs
        elif self.prewarm_packages:
            init_fn = _prewarm_worker
            init_args = (tuple(self.prewarm_packages),)
        else:
            init_fn = None
            init_args = ()
        self._pool = ctx.Pool(processes=self.num_workers, initializer=init_fn, initargs=init_args)

    def _ensure_started(self):
        """
        Return the underlying multiprocessing pool, starting it on first use, or None when
        work should be executed serially in the calling process.
        """
        if self.num_workers == 1:
            return None
        if not self._active:
            raise RuntimeError(
                f"SharedWorkerPool(num_workers={self.num_workers}) must be used inside its 'with' block; "
                "it is either not yet entered or already closed."
            )
        if self._pool is None:
            if _mp.current_process().daemon:
                if not self._warned_daemonic:
                    _warnings.warn("SharedWorkerPool used from within a daemonic process (e.g. a worker of "
                                   "another pool), which cannot start child processes; executing serially.",
                                   RuntimeWarning, stacklevel=3)
                    self._warned_daemonic = True
                return None
            self._start()
        return self._pool

    def map(self, func: Callable, iterable: Sequence, chunksize: Optional[int] = None):
        """Execute func across iterable items (serially if the pool has a single worker)."""
        pool = self._ensure_started()
        if pool is None:
            return [func(x) for x in iterable]
        return pool.map(func, iterable, chunksize=chunksize)

    def starmap(self, func: Callable, iterable: Sequence, chunksize: Optional[int] = None):
        """Execute func(*args) across iterable item tuples (serially if the pool has a single worker)."""
        pool = self._ensure_started()
        if pool is None:
            return [func(*args) for args in iterable]
        return pool.starmap(func, iterable, chunksize=chunksize)

    def imap_unordered(self, func: Callable, iterable: Sequence, chunksize: int = 1):
        """
        Return an iterator over func(x) results in completion order (serially, in input order,
        if the pool has a single worker).
        """
        pool = self._ensure_started()
        if pool is None:
            return (func(x) for x in iterable)
        return pool.imap_unordered(func, iterable, chunksize=chunksize)


@_contextlib.contextmanager
def _pool_scope(num_workers: Union[int, str, None] = None,
                pool: Optional[Union[SharedWorkerPool, _mppool.Pool]] = None,
                prewarm_packages: Optional[Sequence[str]] = None):
    """
    Context manager providing a worker pool shared across the stages of a multi-stage workflow.

    Yields `pool` unchanged if one was supplied by the caller. Otherwise, if `num_workers`
    resolves to more than one worker, yields a *lazy* `SharedWorkerPool` owned by this scope:
    its worker processes are started only when a stage first dispatches work to it (so stages
    that decide to run serially cost nothing) and are reused by every later stage. Yields None
    when execution should be serial.
    """
    if pool is not None:
        yield pool
        return
    workers = determine_num_workers(num_workers) if num_workers is not None else 1
    if workers > 1:
        with SharedWorkerPool(num_workers=workers, prewarm_packages=prewarm_packages, lazy=True) as scoped:
            yield scoped
    else:
        yield None


def resolve_pool_workers(pool: Any) -> int:
    """
    Extract worker count from a SharedWorkerPool or multiprocessing.pool.Pool object.

    Parameters
    ----------
    pool : SharedWorkerPool or multiprocessing.pool.Pool
        Active worker pool to extract worker count from.

    Returns
    -------
    int
        Worker count (>= 1).
    """
    if isinstance(pool, SharedWorkerPool):
        return pool.num_workers
    elif isinstance(pool, _mppool.Pool):
        return getattr(pool, "_processes", 1)
    else:
        raise TypeError(
            f"Expected pool to be a SharedWorkerPool or multiprocessing.pool.Pool instance, got {type(pool)}."
        )


# ===================================================================================================
# Parallel 2D Grid Mapping
# ===================================================================================================

def _grid_task_worker(args):
    """Module-level task runner for 2D grid tasks (must be top-level for spawn/pickle)."""
    r_start, r_end, c_start, c_end, sub_rows, sub_cols, fn = args
    sub_mat = fn(sub_rows, sub_cols)
    return r_start, r_end, c_start, c_end, sub_mat


def parallel_2d_grid_map(
    worker_fn: Callable[[Sequence, Sequence], _np.ndarray],
    row_items: Sequence,
    col_items: Sequence,
    num_workers: Union[int, str, None] = "auto",
    pool: Optional[Union[SharedWorkerPool, _mppool.Pool]] = None,
    grid: Optional[Tuple[int, int]] = None,
    prewarm_packages: Optional[Sequence[str]] = None,
) -> _np.ndarray:
    """
    Execute worker_fn across a 2D partitioned grid of row_items and col_items,
    returning an assembled 2D NumPy array.

    Parameters
    ----------
    worker_fn : callable
        Function `worker_fn(sub_rows, sub_cols) -> np.ndarray` returning a 2D array
        of shape `(len(sub_rows), len(sub_cols))`.
    row_items : sequence
        Row elements.
    col_items : sequence
        Column elements.
    num_workers : int, str, or None, optional
        Target worker count (defaults to "auto").
    pool : Pool or SharedWorkerPool, optional
        Existing worker pool to reuse. If None, an ephemeral pool is instantiated.
    grid : tuple of (int, int), optional
        Explicit (P_rows, P_cols) process grid. Computed automatically if None.
    prewarm_packages : sequence of str, optional
        Optional packages to pre-warm in workers if an ephemeral pool is created.

    Returns
    -------
    numpy.ndarray
        2D array of shape `(len(row_items), len(col_items))`.
    """
    n_rows = len(row_items)
    n_cols = len(col_items)

    if n_rows == 0 or n_cols == 0:
        return _np.empty((n_rows, n_cols), dtype=_np.float64)

    workers = determine_num_workers(num_workers) if pool is None else resolve_pool_workers(pool)

    if workers < 1:
        raise ValueError(f"Worker count must be at least 1, got {workers}.")
    if workers == 1:
        return worker_fn(row_items, col_items)

    pr, pc = grid if grid is not None else compute_2d_worker_grid(n_rows, n_cols, workers)

    if pr <= 1 and pc <= 1:
        return worker_fn(row_items, col_items)

    row_bounds = [int(round(i * n_rows / pr)) for i in range(pr + 1)]
    col_bounds = [int(round(j * n_cols / pc)) for j in range(pc + 1)]

    tasks = []
    for i in range(pr):
        r0, r1 = row_bounds[i], row_bounds[i + 1]
        for j in range(pc):
            c0, c1 = col_bounds[j], col_bounds[j + 1]
            sub_r = row_items[r0:r1]
            sub_c = col_items[c0:c1]
            tasks.append((r0, r1, c0, c1, sub_r, sub_c, worker_fn))

    if pool is not None:
        results = pool.map(_grid_task_worker, tasks)
    else:
        with SharedWorkerPool(num_workers=workers, prewarm_packages=prewarm_packages) as local_pool:
            results = local_pool.map(_grid_task_worker, tasks)

    out = None
    for r0, r1, c0, c1, sub_mat in results:
        if out is None:
            out = _np.empty((n_rows, n_cols), dtype=sub_mat.dtype)
        out[r0:r1, c0:c1] = sub_mat

    return out if out is not None else _np.empty((n_rows, n_cols), dtype=_np.float64)


# ===================================================================================================
# Greedy LPT Bin-Packing Partition
# ===================================================================================================

def lpt_partition(
    items: Sequence[Any],
    weights: Sequence[float],
    num_bins: int,
) -> List[List[Any]]:
    """
    Partition `items` into at most `num_bins` chunks using greedy Longest Processing Time (LPT) bin-packing.

    Parameters
    ----------
    items : sequence
        Elements to partition across bins.
    weights : sequence of float
        Relative computational costs corresponding to each item in `items`.
    num_bins : int
        Number of bins to distribute items across (must be >= 1).

    Returns
    -------
    list of list
        List of non-empty item chunks assigned to each bin.
    """
    if not isinstance(num_bins, (int, _np.integer)) or num_bins < 1:
        raise ValueError(f"num_bins must be an integer >= 1, got {num_bins}.")
    if len(items) != len(weights):
        raise ValueError(f"Length of items ({len(items)}) must match length of weights ({len(weights)}).")
    n_items = len(items)
    if n_items == 0:
        return []
    if num_bins == 1 or n_items == 1:
        return [items.copy() if hasattr(items, "copy") else list(items)]

    sorted_items_with_weights = sorted(
        zip(items, weights), key=lambda x: x[1], reverse=True
    )
    bins: List[List[Any]] = [[] for _ in range(num_bins)]
    bin_weights = [0.0] * num_bins
    for item, weight in sorted_items_with_weights:
        min_bin = min(range(num_bins), key=lambda b: bin_weights[b])
        bins[min_bin].append(item)
        bin_weights[min_bin] += weight
    return [b for b in bins if b]


# ===================================================================================================
# Parallel Pair Reduction
# ===================================================================================================

def _chunk_pair_reduce_worker(args):
    """Module-level task runner for chunked pair reduction."""
    chunk, pair_eval_fn, reduce_fn = args
    accumulator = None
    for pair in chunk:
        val = pair_eval_fn(pair)
        if accumulator is None:
            accumulator = val
        else:
            accumulator = reduce_fn(accumulator, val)
    return accumulator


def parallel_pair_reduce(
    pair_eval_fn: Callable[[Any], Any],
    pair_items: Sequence[Tuple[Any, Any]],
    reduce_fn: Callable[[Any, Any], Any],
    num_workers: Union[int, str, None] = "auto",
    pool: Optional[Union[SharedWorkerPool, _mppool.Pool]] = None,
    prewarm_packages: Optional[Sequence[str]] = None,
    pair_weights: Optional[Sequence[float]] = None,
    dynamic: bool = False,
) -> Any:
    """
    Execute pair_eval_fn over pairs in parallel chunks and aggregate results via reduce_fn.
    When pair_weights is provided, uses greedy Longest Processing Time (LPT) bin-packing
    to balance workloads across workers.

    Parameters
    ----------
    pair_eval_fn : callable
        Function `pair_eval_fn(pair) -> result` evaluating a single pair.
    pair_items : sequence of (item1, item2)
        List or sequence of pair tuples to evaluate.
    reduce_fn : callable
        Binary reduction function `reduce_fn(accum, new_val) -> updated_accum`.
    num_workers : int, str, or None, optional
        Target worker count (defaults to "auto").
    pool : Pool or SharedWorkerPool, optional
        Existing worker pool to reuse.
    prewarm_packages : sequence of str, optional
        Optional packages to pre-warm in workers if an ephemeral pool is created.
    pair_weights : sequence of float, optional
        Estimated relative computational costs per pair for greedy LPT load balancing.
    dynamic : bool, optional (default False)
        If True, dispatches individual tasks dynamically to workers on-demand (longest-first),
        streaming and reducing results as they arrive. If False, pre-partitions tasks into
        static worker chunks.

    Returns
    -------
    result
        Aggregated reduction result across all pairs.
    """
    n_pairs = len(pair_items)
    if n_pairs == 0:
        return None

    workers = determine_num_workers(num_workers) if pool is None else resolve_pool_workers(pool)

    if workers < 1:
        raise ValueError(f"Worker count must be at least 1, got {workers}.")
    if workers == 1 or n_pairs <= 1:
        accum = None
        for p in pair_items:
            res = pair_eval_fn(p)
            accum = res if accum is None else reduce_fn(accum, res)
        return accum

    if dynamic:
        if pair_weights is not None and len(pair_weights) == n_pairs:
            sorted_items = [item for item, _ in sorted(zip(pair_items, pair_weights), key=lambda x: x[1], reverse=True)]
        else:
            sorted_items = pair_items

        def _run_dynamic(active_pool):
            final_accum = None
            for res in active_pool.imap_unordered(pair_eval_fn, sorted_items, chunksize=1):
                if res is not None:
                    final_accum = res if final_accum is None else reduce_fn(final_accum, res)
            return final_accum

        if pool is not None:
            return _run_dynamic(pool)
        else:
            with SharedWorkerPool(num_workers=workers, prewarm_packages=prewarm_packages) as local_pool:
                return _run_dynamic(local_pool)

    if pair_weights is not None and len(pair_weights) == n_pairs:
        chunks = lpt_partition(pair_items, pair_weights, workers)
    else:
        chunk_size = (n_pairs + workers - 1) // workers
        chunks = [pair_items[i : i + chunk_size] for i in range(0, n_pairs, chunk_size) if i < n_pairs]

    tasks = [(chunk, pair_eval_fn, reduce_fn) for chunk in chunks]

    if pool is not None:
        chunk_results = pool.map(_chunk_pair_reduce_worker, tasks)
    else:
        with SharedWorkerPool(num_workers=workers, prewarm_packages=prewarm_packages) as local_pool:
            chunk_results = local_pool.map(_chunk_pair_reduce_worker, tasks)

    final_accum = None
    for cr in chunk_results:
        if cr is not None:
            final_accum = cr if final_accum is None else reduce_fn(final_accum, cr)

    return final_accum


# ===================================================================================================
# Memory Utilities & Headroom Checking
# ===================================================================================================

# Baseline memory footprint of one pool worker process: the unique set size (USS) of a worker that
# has imported numpy, stim and pyGSTi's Cython extensions, measured at ~125 MiB for 'spawn' and
# 'forkserver' workers (Linux, Python 3.13; see review/scripts/phase3_baseline_rss.py). 'fork'
# workers start out sharing their parent's pages (USS ~2 MiB) but un-share them as reference counts
# are touched, so the spawn value is used as a conservative bound for every start method. The main
# process's own footprint is deliberately *not* part of any estimate: it is already resident, and
# hence already excluded from the available memory the estimates are compared against.
WORKER_PROCESS_BASELINE_BYTES = 125 * 1024 * 1024


def get_available_system_memory() -> int:
    """
    Query currently available system memory in bytes.

    Uses psutil if installed, otherwise inspects OS-level indicators
    (/proc/meminfo or sysconf on Linux, sysctl on macOS).

    Returns
    -------
    int or None
        Available memory in bytes. If unable to retrieve this value, None is returned.
    """
    try:
        import psutil
        return int(psutil.virtual_memory().available)
    except Exception:
        pass

    if _sys.platform.startswith("linux"):
        try:
            with open("/proc/meminfo", "r") as f:
                for line in f:
                    if line.startswith("MemAvailable:"):
                        return int(line.split()[1]) * 1024
        except Exception:
            pass
        try:
            return _os.sysconf("SC_PAGE_SIZE") * _os.sysconf("SC_AVPHYS_PAGES")
        except Exception:
            pass

    return None


def warn_if_insufficient_memory(
    estimated_bytes: int,
    available_bytes: Optional[int] = None,
    task_name: str = "",
) -> None:
    """
    Compare estimated peak memory against available system RAM.
    Issues a non-fatal `InsufficientMemoryWarning` if estimated memory exceeds available RAM
    (or if available memory cannot be determined).

    Parameters
    ----------
    estimated_bytes : int
        Estimated peak memory footprint in bytes.
    available_bytes : int, optional
        Available system memory in bytes. Queried dynamically if None.
    task_name : str, optional
        Descriptive name of the operation for the warning message.
    """
    if available_bytes is None:
        available_bytes = get_available_system_memory()

    if available_bytes is None:
        msg = (
            "Unable to resolve available system memory. Cannot determine whether or not operation"
            + " can be successfully performed within memory budget."
        )
        _warnings.warn(msg, _InsufficientMemoryWarning, stacklevel=2)
    elif estimated_bytes > available_bytes:
        msg = (
            f"Estimated peak memory footprint ({estimated_bytes / (1024**3):.2f} GB) for "
            f"{task_name or 'parallel operation'} exceeds currently available system memory "
            f"({available_bytes / (1024**3):.2f} GB). The operation will proceed, but may risk "
            f"swapping or memory exhaustion. Consider reducing num_workers or adjusting truncation."
        )
        _warnings.warn(msg, _InsufficientMemoryWarning, stacklevel=2)

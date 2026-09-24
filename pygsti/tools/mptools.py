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

import itertools as _itertools
import math as _math
import multiprocessing as _mp
import os as _os
import sys as _sys
import time as _time
import warnings as _warnings
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as _np


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

def _worker_prewarm_init():
    """
    Initializer executed once per worker process at pool startup. Eagerly imports
    core computational dependencies to eliminate the cold-start import penalty
    under the 'spawn' start method (macOS, Windows, Python 3.14+).
    """
    try:
        import numpy  # noqa: F401
    except ImportError:
        pass
    try:
        import stim  # noqa: F401
    except ImportError:
        pass
    try:
        import pygsti.tools.fasterrgencalc  # noqa: F401
    except ImportError:
        pass


def _noop_ipc_ping(x):
    """Trivial task for measuring IPC dispatch latency."""
    return x


# ===================================================================================================
# Worker Count & Hardware / Container Resolution
# ===================================================================================================

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
        - "auto" : Automatically infers physical CPU cores (excluding SMT / hyperthreading
          logical threads) via `psutil.cpu_count(logical=False)` with a fallback to
          `os.cpu_count()`. On Linux, inspects cgroups v2 (`/sys/fs/cgroup/cpu.max`) and
          cgroups v1 (`cpu.cfs_quota_us`) to clamp workers to the container quota.
    max_limit : int, optional
        Maximum upper bound on workers to return.

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
            return 1
        if max_limit is not None and max_limit > 0:
            return min(n, int(max_limit))
        return n

    if isinstance(num_workers, str):
        if num_workers.lower() == "auto":
            n_phys = None
            try:
                import psutil
                n_phys = psutil.cpu_count(logical=False)
            except Exception:
                pass

            if n_phys is None or n_phys < 1:
                if hasattr(_os, "sched_getaffinity"):
                    try:
                        n_phys = len(_os.sched_getaffinity(0))
                    except Exception:
                        pass
                if n_phys is None or n_phys < 1:
                    n_phys = _os.cpu_count() or 1

            # On Linux containers, clamp by cgroup quota if enforced
            if _sys.platform.startswith("linux"):
                # cgroups v2
                cgroup_v2 = "/sys/fs/cgroup/cpu.max"
                if _os.path.exists(cgroup_v2):
                    try:
                        with open(cgroup_v2, "r") as f:
                            parts = f.read().strip().split()
                            if len(parts) >= 2 and parts[0] != "max":
                                quota = float(parts[0])
                                period = float(parts[1])
                                if period > 0:
                                    cgroup_cores = _math.floor(quota / period)
                                    if cgroup_cores > 0:
                                        n_phys = min(n_phys, cgroup_cores)
                    except Exception:
                        pass

                # cgroups v1
                quota_v1 = "/sys/fs/cgroup/cpu/cpu.cfs_quota_us"
                period_v1 = "/sys/fs/cgroup/cpu/cpu.cfs_period_us"
                if _os.path.exists(quota_v1) and _os.path.exists(period_v1):
                    try:
                        with open(quota_v1, "r") as f_q, open(period_v1, "r") as f_p:
                            quota = float(f_q.read().strip())
                            period = float(f_p.read().strip())
                            if quota > 0 and period > 0:
                                cgroup_cores = _math.floor(quota / period)
                                if cgroup_cores > 0:
                                    n_phys = min(n_phys, cgroup_cores)
                    except Exception:
                        pass

            if max_limit is not None and max_limit > 0:
                n_phys = min(n_phys, int(max_limit))
            return max(1, int(n_phys))

    raise ValueError(f"Invalid num_workers specification: {num_workers!r}. Expected int, None, or 'auto'.")


# Cache for measured IPC dispatch latencies: (workers, start_method) -> float (seconds)
_IPC_LATENCY_CACHE: Dict[Tuple[int, Optional[str]], float] = {}


def measure_multiprocessing_ipc_latency(
    num_workers: Union[int, str] = "auto",
    start_method: Optional[str] = None,
) -> float:
    """
    Measure local round-trip inter-process communication dispatch latency (T_IPC)
    in seconds for the active start method ('fork' vs. 'spawn').

    Results are cached per process lifetime for instantaneous subsequent lookups (< 1 us).

    Parameters
    ----------
    num_workers : int or str, optional
        Target worker count (defaults to "auto").
    start_method : str, optional
        Start method to evaluate ('fork', 'spawn', 'forkserver'). Defaults to system default.

    Returns
    -------
    float
        Average round-trip dispatch latency in seconds (typically ~0.0001s for fork,
        ~0.00025s for pre-warmed spawn).
    """
    resolved_workers = determine_num_workers(num_workers)
    active_method = start_method or _mp.get_start_method()
    cache_key = (resolved_workers, active_method)

    if cache_key in _IPC_LATENCY_CACHE:
        return _IPC_LATENCY_CACHE[cache_key]

    test_workers = min(resolved_workers, 4)
    if test_workers <= 1:
        # Sequential: negligible latency
        _IPC_LATENCY_CACHE[cache_key] = 1e-6
        return 1e-6

    try:
        ctx = _mp.get_context(active_method)
        with ctx.Pool(processes=test_workers, initializer=_worker_prewarm_init) as pool:
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
        Number of rows (e.g. bitstrings or Pauli strings).
    n_cols : int
        Number of columns (e.g. elementary error generators).
    num_workers : int
        Total worker processes available.

    Returns
    -------
    tuple of (int, int)
        (P_rows, P_cols) process grid factors.
    """
    if num_workers <= 1 or (n_rows <= 1 and n_cols <= 1):
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
    Reusable worker pool context manager with worker pre-warming.

    Eliminates process spin-up and teardown overhead across sequential algorithm stages.
    When wrapping an existing Pool or SharedWorkerPool instance, transparently reuses
    the active processes without spawning duplicates.

    Parameters
    ----------
    num_workers : int, str, or None, optional
        Target worker process count (defaults to "auto").
    start_method : str, optional
        Multiprocessing start method ('fork', 'spawn', 'forkserver').
    prewarm : bool, optional (default True)
        If True, eagerly imports stim, numpy, and pyGSTi Cython extensions into
        workers upon pool initialization.
    """

    def __init__(
        self,
        num_workers: Union[int, str, None] = "auto",
        start_method: Optional[str] = None,
        prewarm: bool = True,
    ):
        self.num_workers = determine_num_workers(num_workers)
        self.start_method = start_method
        self.prewarm = prewarm
        self._pool = None
        self._owns_pool = True

    def __enter__(self):
        if self.num_workers <= 1:
            self._pool = None
            return self

        ctx = _mp.get_context(self.start_method)
        init_fn = _worker_prewarm_init if self.prewarm else None
        self._pool = ctx.Pool(processes=self.num_workers, initializer=init_fn)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._pool is not None and self._owns_pool:
            self._pool.close()
            self._pool.join()
            self._pool = None

    def map(self, func: Callable, iterable: Sequence, chunksize: Optional[int] = None):
        """Execute func across iterable items, falling back to serial execution if pool has 1 worker."""
        if self._pool is None:
            return [func(x) for x in iterable]
        return self._pool.map(func, iterable, chunksize=chunksize)

    def starmap(self, func: Callable, iterable: Sequence, chunksize: Optional[int] = None):
        """Execute func(*args) across iterable item tuples."""
        if self._pool is None:
            return [func(*args) for args in iterable]
        return self._pool.starmap(func, iterable, chunksize=chunksize)


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
    pool: Optional[Any] = None,
    grid: Optional[Tuple[int, int]] = None,
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
        Row elements (e.g. bitstrings or Pauli strings).
    col_items : sequence
        Column elements (e.g. elementary error generators).
    num_workers : int, str, or None, optional
        Target worker count (defaults to "auto").
    pool : Pool or SharedWorkerPool, optional
        Existing worker pool to reuse. If None, an ephemeral pool is instantiated.
    grid : tuple of (int, int), optional
        Explicit (P_rows, P_cols) process grid. Computed automatically if None.

    Returns
    -------
    numpy.ndarray
        2D array of shape `(len(row_items), len(col_items))`.
    """
    n_rows = len(row_items)
    n_cols = len(col_items)

    if n_rows == 0 or n_cols == 0:
        return _np.empty((n_rows, n_cols), dtype=_np.float64)

    workers = determine_num_workers(num_workers) if pool is None else getattr(pool, "num_workers", 1)

    if workers <= 1:
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
        with SharedWorkerPool(num_workers=workers) as local_pool:
            results = local_pool.map(_grid_task_worker, tasks)

    out = None
    for r0, r1, c0, c1, sub_mat in results:
        if out is None:
            out = _np.empty((n_rows, n_cols), dtype=sub_mat.dtype)
        out[r0:r1, c0:c1] = sub_mat

    return out if out is not None else _np.empty((n_rows, n_cols), dtype=_np.float64)


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
    pool: Optional[Any] = None,
) -> Any:
    """
    Execute pair_eval_fn over pairs in parallel chunks and aggregate results via reduce_fn.

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

    Returns
    -------
    result
        Aggregated reduction result across all pairs.
    """
    n_pairs = len(pair_items)
    if n_pairs == 0:
        return None

    workers = determine_num_workers(num_workers) if pool is None else getattr(pool, "num_workers", 1)

    if workers <= 1 or n_pairs <= 1:
        accum = None
        for p in pair_items:
            res = pair_eval_fn(p)
            accum = res if accum is None else reduce_fn(accum, res)
        return accum

    chunk_size = (n_pairs + workers - 1) // workers
    chunks = [pair_items[i : i + chunk_size] for i in range(0, n_pairs, chunk_size) if i < n_pairs]

    tasks = [(chunk, pair_eval_fn, reduce_fn) for chunk in chunks]

    if pool is not None:
        chunk_results = pool.map(_chunk_pair_reduce_worker, tasks)
    else:
        with SharedWorkerPool(num_workers=workers) as local_pool:
            chunk_results = local_pool.map(_chunk_pair_reduce_worker, tasks)

    final_accum = None
    for cr in chunk_results:
        if cr is not None:
            final_accum = cr if final_accum is None else reduce_fn(final_accum, cr)

    return final_accum


# ===================================================================================================
# Memory Utilities & Headroom Checking
# ===================================================================================================

def get_available_system_memory() -> int:
    """
    Query currently available system memory in bytes.

    Uses psutil if installed, otherwise inspects OS-level indicators
    (/proc/meminfo or sysconf on Linux, sysctl on macOS).

    Returns
    -------
    int
        Available memory in bytes.
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

    # Conservative default fallback: 8 GB
    return 8 * 1024 * 1024 * 1024


def warn_if_insufficient_memory(
    estimated_bytes: int,
    available_bytes: Optional[int] = None,
    task_name: str = "",
) -> None:
    """
    Compare estimated peak memory against available system RAM.
    Issues a non-fatal ResourceWarning if estimated memory exceeds available RAM.

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

    if estimated_bytes > available_bytes:
        msg = (
            f"Estimated peak memory footprint ({estimated_bytes / (1024**3):.2f} GB) for "
            f"{task_name or 'parallel operation'} exceeds currently available system memory "
            f"({available_bytes / (1024**3):.2f} GB). The operation will proceed, but may risk "
            f"swapping or memory exhaustion. Consider reducing num_workers or adjusting truncation."
        )
        _warnings.warn(msg, ResourceWarning, stacklevel=2)

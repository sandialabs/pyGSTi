"""
Parallelization configuration, thresholds, memory estimators, multiprocessing workers, and
dynamic calibration utilities for the error generator propagation tools.

The worker classes in this module are top-level so that they are picklable under the `spawn`
start method. The routines they call live in `expansions.py` and `sensitivities.py`; those names
are imported lazily inside the worker `__call__` methods (and inside `_measure_alpha_throughput_us`)
so that this module has no import-time dependency on either of them.
"""
#***************************************************************************************************
# Copyright 2015, 2019, 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************
from __future__ import annotations
import copy as _copy
import functools as _functools
import gc as _gc
import math as _math
import sys as _sys
import time as _time
import warnings as _warnings
from contextlib import contextmanager as _contextmanager
from functools import wraps as _wraps
from types import MappingProxyType as _MappingProxyType
from typing import Any, Final, Iterator, Optional, Sequence, Union, cast as _cast

try:
    import stim
except ImportError:
    msg = "Stim is required for use of the error generator propagation tools module, " \
          "and it does not appear to be installed. If you intend to use this module please update" \
          " your environment."
    _warnings.warn(msg)

import numpy as _np
from pygsti.errorgenpropagation.localstimerrorgen import LocalStimErrorgenLabel as _LSE
from pygsti.circuits import Circuit as _Circuit
from pygsti.models.model import Model as _Model
from pygsti.tools.mptools import (
    determine_num_workers as _determine_num_workers,
    measure_multiprocessing_ipc_latency as _measure_ipc_latency,
    WORKER_PROCESS_BASELINE_BYTES as _WORKER_PROCESS_BASELINE_BYTES,
)

from .errgentypes import _ErrorgenDict, _F

__all__ = [
    'EGPROP_PREWARM_PACKAGES',
    'DEFAULT_PARALLEL_THRESHOLDS',
    'configure_errorgen_propagation_parallelization',
    'optimize_errorgen_propagation_parallelization',
    'measure_lse_entry_size',
    'estimate_magnus_peak_memory',
    'estimate_taylor_peak_memory',
    'estimate_bulk_alpha_peak_memory',
]


@_contextmanager
def _cyclic_gc_paused() -> Iterator[None]:
    """
    Suspend Python's cyclic garbage collector for the duration of a block, restoring its
    previous state afterwards. The drivers below allocate millions of small, acyclic
    containers (labels, term tuples, dicts); every full collection traverses all of them,
    which costs ~20 % of the run time at 100 qubits while never finding anything to free.
    Reference counting is unaffected, so memory use does not change.
    """
    was_enabled = _gc.isenabled()
    _gc.disable()
    try:
        yield
    finally:
        if was_enabled:
            _gc.enable()


def _with_cyclic_gc_paused(fn: _F) -> _F:
    """Decorator form of `_cyclic_gc_paused` for the driver functions."""
    @_wraps(fn)
    def wrapper(*args, **kwargs):
        with _cyclic_gc_paused():
            return fn(*args, **kwargs)
    return _cast(_F, wrapper)


# Standard packages to pre-warm in worker processes for error generator propagation (read-only: pass
# it as `prewarm_packages` when creating a `SharedWorkerPool` for these routines).
EGPROP_PREWARM_PACKAGES: Final[tuple[str, ...]] = ("numpy", "stim", "pygsti.tools.errorgenpropagation.fasterrgencalc")

# Default multi-scale threshold table (sensible hardcoded absolute cutoffs across qubit scales). It is
# read-only (a mapping of read-only mappings); use `configure_errorgen_propagation_parallelization` to
# override individual cutoffs.
DEFAULT_PARALLEL_THRESHOLDS: Final[_MappingProxyType] = _MappingProxyType({
    'small': _MappingProxyType({   # n <= 16 qubits
        'min_km_alpha': 1000,
        'min_magnus_pairs': 4,
        'min_magnus_terms': 100,
        'min_layer_prop_depth': 16,
        'min_terms_commuting_product': 1000,
    }),
    'medium': _MappingProxyType({  # 16 < n <= 64 qubits
        'min_km_alpha': 500,
        'min_magnus_pairs': 2,
        'min_magnus_terms': 50,
        'min_layer_prop_depth': 8,
        'min_terms_commuting_product': 500,
    }),
    'large': _MappingProxyType({   # n > 64 qubits
        'min_km_alpha': 250,
        'min_magnus_pairs': 2,
        'min_magnus_terms': 50,
        'min_layer_prop_depth': 4,
        'min_terms_commuting_product': 250,
    }),
})

# Manual overrides configured via configure_errorgen_propagation_parallelization
_MANUAL_PARALLEL_OVERRIDES: dict[str, Any] = {}

# Names of the parallelized operations accepted by `force_parallel` / `force_serial`.
_PARALLEL_OPERATIONS = frozenset({'alpha', 'magnus', 'layer_prop', 'commuting_product'})


def configure_errorgen_propagation_parallelization(
    min_km_alpha: Optional[int] = None,
    min_magnus_pairs: Optional[int] = None,
    min_terms_per_magnus_pair: Optional[int] = None,
    min_layer_prop_depth: Optional[int] = None,
    min_terms_commuting_product: Optional[int] = None,
    force_parallel: Optional[Union[bool, Sequence[str]]] = None,
    force_serial: Optional[Union[bool, Sequence[str]]] = None,
    reset_to_defaults: bool = False,
) -> dict[str, Any]:
    """
    Manually configure or override the parallelization threshold criteria
    for error generator propagation workflows.

    Parameters
    ----------
    min_km_alpha : int, optional
        Minimum generator-outcome workload product W = K * M required to engage
        2D grid parallelization in `bulk_alpha` and `bulk_alpha_pauli`. Workloads
        with K * M below this cutoff execute via the sequential execution.
    min_magnus_pairs : int, optional
        Minimum number of layer pairs (L * (L - 1) / 2) required to engage chunked
        parallel pairwise reduction in `_second_order_magnus_term` and `magnus_expansion`.
    min_terms_per_magnus_pair : int, optional
        Minimum average error generator terms per layer required to engage parallel
        Magnus expansion.
    min_layer_prop_depth : int, optional
        Minimum circuit layer depth (L) required to engage chunked layer propagation
        in `ErrorGeneratorPropagator.propagate_errorgens`.
    min_terms_commuting_product : int, optional
        Minimum workload product W = n1 * n2 of term counts in the two input dictionaries
        required to engage parallel bilinear slicing in `_commuting_product`.
    force_parallel : bool, str, or sequence of str, optional
        If True, unconditionally enables multiprocessing across all operations regardless
        of workload size. If an operation name or a sequence of operation names (any of
        'alpha', 'magnus', 'layer_prop', 'commuting_product'), forces parallelization only
        for the named operations. False clears any previous setting.
    force_serial : bool, str, or sequence of str, optional
        If True, unconditionally forces single-threaded sequential execution across all
        operations, bypassing multiprocessing. If an operation name or a sequence of operation
        names, forces sequential execution only for the named operations. False clears any
        previous setting. An operation may not be forced both parallel and serial.
    reset_to_defaults : bool, optional (default False)
        If True, clears all dynamic profiling calibrations and manual overrides,
        restoring pyGSTi's default multi-scale threshold table.

    Returns
    -------
    dict
        Dictionary showing the currently active configuration and threshold cutoffs.
    """
    if reset_to_defaults:
        _MANUAL_PARALLEL_OVERRIDES.clear()
        if "_ACTIVE_PARALLEL_THRESHOLDS" in globals():
            _ACTIVE_PARALLEL_THRESHOLDS.pop("dynamic_calibrated", None)
            _ACTIVE_PARALLEL_THRESHOLDS.pop("alpha_a", None)
            _ACTIVE_PARALLEL_THRESHOLDS.pop("alpha_b", None)
            _ACTIVE_PARALLEL_THRESHOLDS.pop("t_ipc_s", None)
        return {"status": "reset_to_defaults",
                "defaults": {tier: dict(cutoffs) for tier, cutoffs in DEFAULT_PARALLEL_THRESHOLDS.items()}}

    # Validate the force settings against the combined (existing + requested) state before
    # mutating anything, so an invalid call leaves the configuration unchanged.
    new_force_parallel = _MANUAL_PARALLEL_OVERRIDES.get("force_parallel", False) if force_parallel is None \
        else _normalize_forced_operations(force_parallel, "force_parallel")
    new_force_serial = _MANUAL_PARALLEL_OVERRIDES.get("force_serial", False) if force_serial is None \
        else _normalize_forced_operations(force_serial, "force_serial")
    par_ops = set(_PARALLEL_OPERATIONS) if new_force_parallel is True else set(new_force_parallel or ())
    ser_ops = set(_PARALLEL_OPERATIONS) if new_force_serial is True else set(new_force_serial or ())
    if par_ops & ser_ops:
        raise ValueError(f"Operations {sorted(par_ops & ser_ops)} cannot be forced both parallel and serial. "
                         "Clear one of the settings (e.g. force_serial=False) first.")

    if min_km_alpha is not None:
        _MANUAL_PARALLEL_OVERRIDES["min_km_alpha"] = int(min_km_alpha)
    if min_magnus_pairs is not None:
        _MANUAL_PARALLEL_OVERRIDES["min_magnus_pairs"] = int(min_magnus_pairs)
    if min_terms_per_magnus_pair is not None:
        _MANUAL_PARALLEL_OVERRIDES["min_terms_per_magnus_pair"] = int(min_terms_per_magnus_pair)
    if min_layer_prop_depth is not None:
        _MANUAL_PARALLEL_OVERRIDES["min_layer_prop_depth"] = int(min_layer_prop_depth)
    if min_terms_commuting_product is not None:
        _MANUAL_PARALLEL_OVERRIDES["min_terms_commuting_product"] = int(min_terms_commuting_product)
    if force_parallel is not None:
        _MANUAL_PARALLEL_OVERRIDES["force_parallel"] = new_force_parallel
    if force_serial is not None:
        _MANUAL_PARALLEL_OVERRIDES["force_serial"] = new_force_serial

    return _copy.deepcopy(_MANUAL_PARALLEL_OVERRIDES)


def _normalize_forced_operations(value: Union[bool, str, Sequence[str]], arg_name: str) -> Union[bool, tuple[str, ...]]:
    """
    Normalize a `force_parallel`/`force_serial` setting to either a bool or a tuple of
    validated operation names (a bare string is treated as a single operation name).
    """
    if isinstance(value, (bool, _np.bool_)):
        return bool(value)
    if isinstance(value, str):
        value = (value,)
    try:
        ops = tuple(value)
    except TypeError:
        raise TypeError(f"{arg_name} must be a bool, an operation name, or a sequence of operation names; "
                        f"got {value!r}.")
    unknown = [op for op in ops if op not in _PARALLEL_OPERATIONS]
    if unknown:
        raise ValueError(f"Unknown operation name(s) {unknown} in {arg_name}; "
                         f"expected any of {sorted(_PARALLEL_OPERATIONS)}.")
    return ops


def _forced_parallelism_mode(operation: str) -> Optional[bool]:
    """
    The manually forced execution mode for `operation`: False if it is forced serial,
    True if it is forced parallel, and None if neither applies. Serial takes precedence.
    """
    force_ser = _MANUAL_PARALLEL_OVERRIDES.get("force_serial", False)
    if force_ser is True or (isinstance(force_ser, tuple) and operation in force_ser):
        return False
    force_par = _MANUAL_PARALLEL_OVERRIDES.get("force_parallel", False)
    if force_par is True or (isinstance(force_par, tuple) and operation in force_par):
        return True
    return None


def _qubit_scale_tier(n_qubits: int) -> str:
    """The row of the default threshold table ('small', 'medium' or 'large') for `n_qubits`."""
    return 'small' if n_qubits <= 16 else ('medium' if n_qubits <= 64 else 'large')


# Memory of one stim.PauliString (Python object plus its SIMD-aligned bit buffers): measured at
# ~240 bytes for up to ~128 qubits, growing by roughly a byte per 4 additional qubits beyond that.
_STIM_PAULISTRING_BASE_BYTES = 240

# Memory of one slot of a Python dict (hash, key and value pointers, plus the index entry), scaled
# for the dict's typical fill factor.
_DICT_SLOT_BYTES = 64


@_functools.lru_cache(maxsize=None)
def measure_lse_entry_size(n_qubits: int) -> int:
    """
    Estimate the memory in bytes of a worst-case (C- or A-type, full-weight) `LocalStimErrorgenLabel`
    dictionary entry with a real rate: the label instance and its attribute dictionary, its two
    stim.PauliStrings and their cached string forms, its cached support mask, the rate, and the
    dictionary slot. Results are cached per qubit count.

    Parameters
    ----------
    n_qubits : int
        Number of qubits in the circuit.

    Returns
    -------
    int
        Memory in bytes consumed by a single dictionary entry holding a LocalStimErrorgenLabel and real rate.
    """
    n = max(1, int(n_qubits))
    lbl = _LSE("C", [stim.PauliString("X" * n), stim.PauliString("Z" * n)])
    mask = lbl.support_mask  # cached on the instance by the commutator routines
    return (
        _sys.getsizeof(lbl)
        + _sys.getsizeof(lbl.__dict__)
        + _sys.getsizeof(lbl.basis_element_labels)
        + 2 * (_STIM_PAULISTRING_BASE_BYTES + n // 4)
        + _sys.getsizeof(lbl._hashable_basis_element_labels)
        + sum(_sys.getsizeof(s) for s in lbl._hashable_basis_element_labels)
        + _sys.getsizeof(lbl._hashable_string_rep)
        + _sys.getsizeof(mask)
        + _sys.getsizeof(1.0)
        + _DICT_SLOT_BYTES
    )


def _string_entry_size(n_qubits: int) -> int:
    """
    Memory in bytes of one entry of a string-keyed rate dictionary (as exchanged with and merged
    from worker processes), keyed by a worst-case (two-Pauli) `_hashable_string_rep`.
    """
    n = max(1, int(n_qubits))
    return _sys.getsizeof('C' + 'X' * (2 * n)) + _sys.getsizeof(1.0) + _DICT_SLOT_BYTES


def _num_distinct_errorgen_labels(n_qubits: int) -> int:
    """
    The number of distinct elementary error generator labels on `n_qubits` qubits: 4^n - 1 each of
    H and S type (one non-identity Pauli), and (4^n - 1)(4^n - 2)/2 each of C and A type (an
    unordered pair of distinct non-identity Paulis).
    """
    m = 4**int(n_qubits) - 1
    return 2 * m + m * (m - 1)


def estimate_magnus_peak_memory(
    n_qubits: int,
    n_layers: int,
    avg_errorgens_per_layer: int,
    num_workers: Union[int, str, None] = "auto",
    magnus_order: int = 2,
) -> int:
    """
    Worst-case estimate of the additional peak memory in bytes (beyond the calling process's
    existing footprint) needed by a Magnus expansion, from summary statistics of its input. This
    assumes every pair of terms in every pair of layers emits the maximal number of new terms,
    capped by the number of distinct labels on `n_qubits` qubits, and so can grossly overestimate
    the memory needed by large, local workloads. `magnus_expansion` itself uses a tighter estimate
    computed from the supports of the actual input terms.

    Parameters
    ----------
    n_qubits : int
        Number of qubits in the circuit.
    n_layers : int
        Number of error generator layers (L).
    avg_errorgens_per_layer : int
        Average number of elementary error generator terms per layer (K).
    num_workers : int, str, or None, optional
        Target worker count (defaults to "auto").
    magnus_order : int, optional (default 2)
        Order of the Magnus expansion (1, 2, or 3).

    Returns
    -------
    int
        Estimated peak memory footprint in bytes.
    """
    w = _determine_num_workers(num_workers)
    if w < 1:
        raise ValueError(f"Worker count must be at least 1, got {w}.")
    if magnus_order not in (1, 2, 3):
        raise ValueError(f"magnus_order must be 1, 2, or 3, got {magnus_order}.")

    master_base = 0  # the calling process's footprint is already excluded from available memory
    worker_base = _WORKER_PROCESS_BASELINE_BYTES if w > 1 else 0
    bytes_per_term = measure_lse_entry_size(n_qubits)
    max_distinct_terms = _num_distinct_errorgen_labels(n_qubits)

    K = avg_errorgens_per_layer
    # In the worst case, each pair of elementary error generators in a layer pair emits at most 4 terms
    max_terms_per_pair = min(4 * (K ** 2), max_distinct_terms) if K > 0 else 0

    if magnus_order == 1 or n_layers <= 1:
        input_bytes = n_layers * K * bytes_per_term
        return int(master_base + input_bytes)

    n_pairs = (n_layers * (n_layers - 1)) // 2
    pairs_per_worker = max(1, n_pairs // max(1, w))
    p1_worker_buffer = min(pairs_per_worker * max_terms_per_pair, max_distinct_terms) * bytes_per_term

    if magnus_order == 2:
        total_bytes = master_base + (w * worker_base) + (w * p1_worker_buffer)
        return int(total_bytes)

    # magnus_order == 3: Two-phase parallel evaluation with held columns and outer commutators
    col_terms = min((n_layers - 1) * max_terms_per_pair, max_distinct_terms)
    held_columns_bytes = n_layers * col_terms * bytes_per_term

    outer_per_worker = max(1, n_layers // max(1, w))
    outer_terms_per_task = min(4 * K * max(1, col_terms), max_distinct_terms)
    p2_worker_buffer = min(outer_per_worker * outer_terms_per_task, max_distinct_terms) * bytes_per_term

    worker_buffer = max(p1_worker_buffer, p2_worker_buffer)
    total_bytes = master_base + held_columns_bytes + (w * worker_base) + (w * worker_buffer)
    return int(total_bytes)


def _num_labels_with_support_at_most(n_qubits: int, max_support: int) -> int:
    """
    The number of distinct elementary error generator labels on `n_qubits` qubits whose (combined)
    support has at most `max_support` qubits.

    Labels are counted by their exact support set S, of size k, outside of which all Paulis are the
    identity (there are C(n, k) such sets):

    - H/S labels: one Pauli that is non-identity on every qubit of S, so 3^k of each type.
    - C/A labels: an unordered pair of distinct non-identity Paulis that together cover S. There are
      15^k ordered pairs of Paulis covering S (any of the 16 pairs of single-qubit Paulis except
      (I, I) on each qubit), of which 3^k have P = Q and 2 * 3^k have one Pauli equal to the identity,
      leaving 15^k - 3 * 3^k ordered pairs of distinct non-identity Paulis. The labels store their two
      Paulis in sorted order, so each unordered pair is one label: (15^k - 3 * 3^k) / 2 pairs, each
      giving a C and an A label, i.e. 15^k - 3 * 3^k C/A labels in total.

    In total each support set holds 2 * 3^k + 15^k - 3 * 3^k = 15^k - 3^k labels. For
    `max_support >= n_qubits` this is `_num_distinct_errorgen_labels(n_qubits)`.
    """
    n = int(n_qubits)
    s = max(0, min(n, int(max_support)))
    return sum(_math.comb(n, k) * (15**k - 3**k) for k in range(1, s + 1))


def _layer_support_statistics(errorgen_layers: list[_ErrorgenDict], n_qubits: int):
    """
    Per-layer support statistics of a list of error generator layers.

    Returns
    -------
    sizes : numpy.ndarray of shape (L,)
        Number of terms in each layer.
    qubit_counts : numpy.ndarray of shape (L, n_qubits)
        `qubit_counts[i, q]` is the number of terms of layer i acting non-trivially on qubit q.
    max_weight : int
        The largest support size (number of qubits acted on) of any term.
    """
    num_layers = len(errorgen_layers)
    nbytes = (n_qubits + 7) // 8
    sizes = _np.array([len(layer) for layer in errorgen_layers], dtype=_np.float64)
    qubit_counts = _np.zeros((num_layers, n_qubits), dtype=_np.float64)
    max_weight = 0
    for i, layer in enumerate(errorgen_layers):
        if not layer:
            continue
        # support masks are cached on the labels (and needed by the commutator routines anyway)
        buf = b''.join(lbl.support_mask.to_bytes(nbytes, 'little') for lbl in layer)
        bits = _np.unpackbits(_np.frombuffer(buf, dtype=_np.uint8).reshape(len(layer), nbytes),
                              axis=1, bitorder='little')[:, :n_qubits]
        qubit_counts[i] = bits.sum(axis=0)
        max_weight = max(max_weight, int(bits.sum(axis=1).max()))
    return sizes, qubit_counts, max_weight


def _estimate_magnus_peak_memory_from_layers(errorgen_layers: list[_ErrorgenDict], n_qubits: int,
                                             num_workers: int = 1, magnus_order: int = 2) -> int:
    """
    Estimate the additional peak memory in bytes needed by `magnus_expansion` for the given layers,
    using the supports of the actual input terms. Compared to the worst-case
    `estimate_magnus_peak_memory`, this

    1. counts only pairs of terms with overlapping support (pairs acting on disjoint qubits commute
       and are skipped by the commutator routines), bounding the number of overlapping pairs
       between layers i and j by sum_q c_i[q] c_j[q], where c_i[q] is the number of terms of
       layer i acting on qubit q;
    2. caps the number of distinct output terms by locality: a commutator's output acts within the
       union of its inputs' supports, so second-order (third-order) terms have support of at most
       2w (3w) qubits, w being the largest input support;
    3. models every intermediate result as a dictionary, whose memory is set by its number of
       distinct keys, min(emitted terms, distinct-label cap), rather than by the number of emitted terms.

    The estimate assumes every overlapping pair emits the maximal number (4) of terms, so it is
    still an upper bound in that respect.
    """
    num_layers = len(errorgen_layers)
    if magnus_order < 2 or num_layers < 2:
        return 0

    entry = measure_lse_entry_size(n_qubits)
    str_entry = _string_entry_size(n_qubits)
    sizes, counts, max_weight = _layer_support_statistics(errorgen_layers, n_qubits)
    cap2 = float(_num_labels_with_support_at_most(n_qubits, 2 * max_weight))

    # 1. bound on overlapping term pairs for every layer pair (i > j)
    pair_overlaps = _np.minimum(counts @ counts.T, _np.outer(sizes, sizes))
    lower = _np.tril(pair_overlaps, k=-1)
    pair_emissions = 4.0 * lower  # terms emitted by each layer-pair commutator
    distinct_first = min(float(sizes.sum()), float(_num_distinct_errorgen_labels(n_qubits)))
    distinct_second = min(float(pair_emissions.sum()), cap2)

    # magnus_expansion holds the first- and second-order dictionaries, their sum, and its truncated copy.
    final_entries = distinct_first + 3.0 * distinct_second

    if magnus_order == 2:
        if num_workers <= 1:
            return int(final_entries * entry)
        # Main process: merged string-keyed dict, then the materialized survivors (accounted above).
        # Each worker holds the inputs and the (label- and string-keyed) result of its largest task.
        largest_task = min(float(pair_emissions.max()), cap2)
        worker = (2.0 * sizes.max() + largest_task) * entry + largest_task * str_entry
        return int(final_entries * entry + distinct_second * str_entry
                   + num_workers * (worker + _WORKER_PROCESS_BASELINE_BYTES))

    # magnus_order == 3
    cap3 = float(_num_labels_with_support_at_most(n_qubits, 3 * max_weight))
    row_terms = pair_emissions.sum(axis=1)  # emitted into row i (pairs (i, k), k < i)
    col_terms = pair_emissions.sum(axis=0)  # emitted into column k (pairs (i, k), i > k)
    prefix_terms = _np.minimum(_np.cumsum(row_terms), cap2)  # size bound of the running row sum R_i
    suffix_terms = _np.minimum(_np.cumsum(col_terms[::-1])[::-1], cap2)  # running column sum C_k

    # Per-qubit bounds on the terms of R_i / C_k acting on qubit q: an output of pair (a, b) acts within
    # supp(a) | supp(b), so outputs of layer pair (j, k) touching q number at most 4 (c_j[q] K_k + K_j c_k[q]).
    cum_sizes = _np.concatenate(([0.0], _np.cumsum(sizes)[:-1]))            # sum_{k<j} K_k
    cum_counts = _np.vstack((_np.zeros(n_qubits), _np.cumsum(counts, axis=0)[:-1]))  # sum_{k<j} c_k
    row_q = 4.0 * (counts * cum_sizes[:, None] + sizes[:, None] * cum_counts)
    prefix_q = _np.minimum(_np.cumsum(row_q, axis=0), prefix_terms[:, None])
    rev_sizes = _np.concatenate((_np.cumsum(sizes[::-1])[::-1][1:], [0.0]))   # sum_{j>k} K_j
    rev_counts = _np.vstack((_np.cumsum(counts[::-1], axis=0)[::-1][1:], _np.zeros(n_qubits)))
    col_q = 4.0 * (counts * rev_sizes[:, None] + sizes[:, None] * rev_counts)
    suffix_q = _np.minimum(_np.cumsum(col_q[::-1], axis=0)[::-1], suffix_terms[:, None])

    # third-order emissions: [A(i), R_i] and [A(k), C_k], restricted to overlapping pairs
    outer_row = 4.0 * _np.minimum(sizes * prefix_terms, (counts * prefix_q).sum(axis=1))
    outer_col = 4.0 * _np.minimum(sizes * suffix_terms, (counts * suffix_q).sum(axis=1))
    distinct_third = min(float(outer_row.sum() + outer_col.sum()), cap3)
    final_entries += 3.0 * distinct_third

    held_columns = float(_np.minimum(col_terms, cap2).sum())
    if num_workers <= 1:
        # held columns, the running sum and its truncated copy, plus the final dictionaries
        return int((final_entries + held_columns + 2.0 * distinct_second) * entry)

    # Parallel: the main process keeps every pairwise result plus the rows and columns as string-keyed
    # dicts, and materializes every prefix R_i and suffix C_k as label-keyed dicts for the outer tasks.
    pairwise = float(_np.minimum(pair_emissions[pair_emissions > 0], cap2).sum())
    rows = float(_np.minimum(row_terms, cap2).sum())
    main = (pairwise + rows + held_columns) * str_entry \
        + float(prefix_terms.sum() + suffix_terms.sum()) * entry + final_entries * entry
    largest_p1 = min(float(pair_emissions.max()), cap2)
    largest_p2 = min(float(max(outer_row.max(), outer_col.max())), cap3)
    largest_acc = float(max(prefix_terms.max(), suffix_terms.max()))
    worker = max((2.0 * sizes.max() + largest_p1) * entry + largest_p1 * str_entry,
                 (sizes.max() + largest_acc + largest_p2) * entry + largest_p2 * str_entry)
    return int(main + num_workers * (worker + _WORKER_PROCESS_BASELINE_BYTES))


def estimate_taylor_peak_memory(
    n_qubits: int,
    num_errorgens: int,
    order: int = 1,
    num_workers: Union[int, str, None] = "auto",
) -> int:
    """
    Worst-case estimate of the additional peak memory in bytes (beyond the calling process's
    existing footprint) needed by an error generator Taylor expansion.

    Parameters
    ----------
    n_qubits : int
        Number of qubits in the circuit / error generator.
    num_errorgens : int
        Number of elementary error generator terms in the generator (K).
    order : int, optional (default 1)
        Order of the Taylor expansion (>= 1).
    num_workers : int, str, or None, optional
        Target worker count (defaults to "auto").

    Returns
    -------
    int
        Estimated peak memory footprint in bytes.
    """
    w = _determine_num_workers(num_workers)
    if w < 1:
        raise ValueError(f"Worker count must be at least 1, got {w}.")
    if order < 1:
        raise ValueError(f"order must be an integer >= 1, got {order}.")

    master_base = 0  # the calling process's footprint is already excluded from available memory
    worker_base = _WORKER_PROCESS_BASELINE_BYTES if w > 1 else 0
    entry_bytes = measure_lse_entry_size(n_qubits)
    max_distinct_terms = _num_distinct_errorgen_labels(n_qubits)

    K = num_errorgens
    if order == 1 or K == 0:
        return int(master_base + K * entry_bytes)

    current_k_terms = K
    total_retained_terms = K
    max_worker_buffer_bytes = 0

    for k in range(2, order + 1):
        prev_terms = current_k_terms
        current_k_terms = min(K * prev_terms * 7, max_distinct_terms)
        total_retained_terms += current_k_terms

        if w > 1:
            slice_terms = max(1, K // w)
            worker_task_terms = min(slice_terms * prev_terms * 7, max_distinct_terms)
            worker_mem = (prev_terms + worker_task_terms) * entry_bytes
            max_worker_buffer_bytes = max(max_worker_buffer_bytes, worker_mem)
        else:
            max_worker_buffer_bytes = max(max_worker_buffer_bytes, current_k_terms * entry_bytes)

    retained_bytes = total_retained_terms * entry_bytes
    total_bytes = master_base + retained_bytes + (w * worker_base) + (w * max_worker_buffer_bytes)
    return int(total_bytes)


def _should_parallelize_magnus(
    num_layers: int,
    avg_terms_per_layer: int,
    num_workers: Union[int, str, None] = "auto",
    n_qubits: Optional[int] = None,
) -> bool:
    """
    Internal helper: determines whether Magnus expansion should be parallelized across layer pairs.
    """
    if n_qubits is None:
        raise ValueError("n_qubits must be specified as an integer >= 1.")
    if not isinstance(n_qubits, (int, _np.integer)) or n_qubits < 1:
        raise ValueError(f"n_qubits must be an integer >= 1, got {n_qubits}.")

    workers = _determine_num_workers(num_workers)
    if workers < 1:
        raise ValueError(f"Worker count must be at least 1, got {workers}.")
    if workers == 1:
        return False
    n_pairs = (num_layers * (num_layers - 1)) // 2
    if n_pairs < 1:
        return False

    forced = _forced_parallelism_mode("magnus")
    if forced is not None:
        return forced

    # Each manual override replaces only its own default-table cutoff.
    cutoffs = DEFAULT_PARALLEL_THRESHOLDS[_qubit_scale_tier(n_qubits)]
    min_pairs = _MANUAL_PARALLEL_OVERRIDES.get("min_magnus_pairs", cutoffs['min_magnus_pairs'])
    min_terms = _MANUAL_PARALLEL_OVERRIDES.get("min_terms_per_magnus_pair", cutoffs['min_magnus_terms'])
    return n_pairs >= min_pairs and avg_terms_per_layer >= min_terms


def _should_parallelize_commuting_product(
    n_qubits: int,
    num_terms_1: int,
    num_terms_2: int,
    num_workers: Union[int, str, None] = "auto",
) -> bool:
    """
    Internal helper: determines whether commuting product should be parallelized via bilinear slicing.
    """
    if n_qubits is None:
        raise ValueError("n_qubits must be specified as an integer >= 1.")
    if not isinstance(n_qubits, (int, _np.integer)) or n_qubits < 1:
        raise ValueError(f"n_qubits must be an integer >= 1, got {n_qubits}.")

    workers = _determine_num_workers(num_workers)
    if workers < 1:
        raise ValueError(f"Worker count must be at least 1, got {workers}.")
    if workers == 1:
        return False

    w = num_terms_1 * num_terms_2
    if w <= 0:
        return False

    forced = _forced_parallelism_mode("commuting_product")
    if forced is not None:
        return forced

    if "min_terms_commuting_product" in _MANUAL_PARALLEL_OVERRIDES:
        return w >= _MANUAL_PARALLEL_OVERRIDES["min_terms_commuting_product"]

    threshold = DEFAULT_PARALLEL_THRESHOLDS[_qubit_scale_tier(n_qubits)]['min_terms_commuting_product']
    return w >= threshold


def _lse_from_string_rep(rep: str, n: int) -> _LSE:
    """Reconstruct a LocalStimErrorgenLabel from its cached string representation."""
    typ = rep[0]
    if typ in ('H', 'S'):
        s = rep[1:]
        return _LSE(typ, (stim.PauliString(s),), pauli_str_reps=(s,))
    else:
        s1 = rep[1 : 1 + n]
        s2 = rep[1 + n :]
        return _LSE(typ, (stim.PauliString(s1), stim.PauliString(s2)), pauli_str_reps=(s1, s2))


def _merge_string_dictionaries(d1: dict[str, float], d2: dict[str, float]) -> dict[str, float]:
    """
    Binary reduction merging string rate dictionaries by summing overlapping rates. Updates and
    returns `d1` in place (avoiding a copy of the whole running total at every step of a reduction),
    so `d1` must not be referenced elsewhere; `d2` is not modified.
    """
    get = d1.get
    for k, v in d2.items():
        d1[k] = get(k, 0.0) + v
    return d1


class _MagnusPairEvaluator:
    """Worker task evaluating commutators for a single pair of layers (picklable under spawn)."""
    def __init__(self, identity, threshold):
        self.identity = identity
        self.threshold = threshold

    def __call__(self, pair):
        from .expansions import _accumulate_layer_pairwise_commutators
        with _cyclic_gc_paused():
            l1, l2 = pair
            target = {}
            _accumulate_layer_pairwise_commutators(
                target, l1, l2, self.identity, addl_weight=0.5, truncation_threshold=self.threshold
            )
            return {k._hashable_string_rep: v for k, v in target.items()}


class _MagnusPhase1PairEvaluator:
    """Worker task evaluating commutators for a single pair of layers in Phase 1 of Magnus expansion."""
    def __init__(self, identity, threshold):
        self.identity = identity
        self.threshold = threshold

    def __call__(self, item):
        from .expansions import _accumulate_layer_pairwise_commutators
        with _cyclic_gc_paused():
            i, k, l1, l2 = item
            target = {}
            _accumulate_layer_pairwise_commutators(
                target, l1, l2, self.identity, truncation_threshold=self.threshold
            )
            return {(i, k): {lbl._hashable_string_rep: rate for lbl, rate in target.items()}}


class _MagnusPhase2Evaluator:
    """Worker task evaluating outer commutators for Phase 2 of joint 2nd/3rd order Magnus expansion."""
    def __init__(self, identity, threshold):
        self.identity = identity
        self.threshold = threshold

    def __call__(self, task):
        from .expansions import _accumulate_layer_pairwise_commutators
        with _cyclic_gc_paused():
            layer, acc, weight = task
            target = {}
            _accumulate_layer_pairwise_commutators(
                target, layer, acc, self.identity, addl_weight=weight, truncation_threshold=self.threshold
            )
            return {lbl._hashable_string_rep: rate for lbl, rate in target.items()}


def _merge_pair_indexed_string_dictionaries(d1: dict, d2: dict) -> dict:
    """
    Binary reduction merging (i, k) layer-pair indexed string rate dictionaries additively. Updates and
    returns `d1` in place (including its per-pair dictionaries), so `d1` must not be referenced
    elsewhere. The per-pair dictionaries of `d2` that are new to `d1` are stored in it by reference, so
    `d2` must not be used (or modified) afterwards either.
    """
    for pair_key, rates in d2.items():
        sub = d1.get(pair_key)
        if sub is None:
            d1[pair_key] = rates
        else:
            get = sub.get
            for lbl, rate in rates.items():
                sub[lbl] = get(lbl, 0.0) + rate
    return d1


def _generate_balanced_magnus_pairs(
    errorgen_layers: list[_ErrorgenDict],
    num_workers: int,
    alpha: float = 0.45,
    include_indices: bool = False,
) -> tuple[list[Any], list[float]]:
    """
    Decompose Magnus layer pairs into load-balanced tasks using Dynamic Worker-Budget
    Capacity Slicing with order-preserving strided partitioning.

    Parameters
    ----------
    errorgen_layers : list of dicts
        List of error generator layer dictionaries.
    num_workers : int
        Active worker process count.
    alpha : float, optional (default 0.45)
        Fraction of ideal worker capacity defining the maximum task budget.
    include_indices : bool, optional (default False)
        If True, tasks include layer indices (i, k, ...) for Phase 1 of 3rd-order expansion.

    Returns
    -------
    tasks : list
        List of task tuples.
    task_weights : list of float
        Estimated computational weights corresponding to each task.
    """
    num_layers = len(errorgen_layers)
    if num_layers <= 1:
        return [], []

    raw_pairs = []
    total_work = 0.0
    for i in range(num_layers):
        for j in range(i):
            w = len(errorgen_layers[i]) * len(errorgen_layers[j])
            raw_pairs.append((i, j, errorgen_layers[i], errorgen_layers[j], w))
            total_work += w

    w_ideal = total_work / max(1, num_workers)
    t_max = alpha * w_ideal

    tasks = []
    task_weights = []

    for i, j, l1, l2, w in raw_pairs:
        if w <= t_max or num_workers <= 2 or t_max <= 0:
            s_ij = 1
        else:
            s_ideal = int(_math.ceil(w / t_max))
            max_terms = max(len(l1), len(l2))
            s_ij = min(s_ideal, max_terms)

        if s_ij <= 1:
            if include_indices:
                tasks.append((i, j, l1, l2))
            else:
                tasks.append((l1, l2))
            task_weights.append(float(w))
        else:
            w_slice = float(w) / s_ij
            if len(l1) >= len(l2):
                # Slice first operand l1; l2 remains intact
                items = list(l1.items())
                for k in range(s_ij):
                    sub = dict(items[k::s_ij])
                    if include_indices:
                        tasks.append((i, j, sub, l2))
                    else:
                        tasks.append((sub, l2))
                    task_weights.append(w_slice)
            else:
                # Slice second operand l2; l1 remains intact (OPERAND ORDER PRESERVED)
                items = list(l2.items())
                for k in range(s_ij):
                    sub = dict(items[k::s_ij])
                    if include_indices:
                        tasks.append((i, j, l1, sub))
                    else:
                        tasks.append((l1, sub))
                    task_weights.append(w_slice)

    return tasks, task_weights


class _CommutingProductSliceWorker:
    """
    Worker task evaluating a bilinear slice of a commuting product (picklable under spawn).

    The partial sum is returned (keyed by `_hashable_string_rep`) with entries of magnitude at most
    `chunk_threshold` dropped, which reduces the data sent back to the main process. The partial sums
    of the slices are later added, so an entry that is small in one slice may belong to a label whose
    total is large (and, conversely, dropped entries may have cancelled part of a larger one). The
    caller therefore passes `chunk_threshold = threshold / n_slices`: if a label's total exceeds the
    threshold then at least one slice's entry exceeds `threshold / n_slices` and is kept, so the label
    is never lost entirely, and the total of the dropped entries of any label is below `threshold`.
    This does not make the merged value exactly equal to the serial one: it can differ by up to about
    `(n_slices - 1) / n_slices * threshold`, which can move a value across the threshold.
    """
    def __init__(self, full_other, identity, chunk_threshold):
        self.full_other = full_other
        self.identity = identity
        self.chunk_threshold = chunk_threshold

    def __call__(self, sub_dict):
        from .expansions import _raw_commuting_product
        with _cyclic_gc_paused():
            part = _raw_commuting_product(sub_dict, self.full_other, self.identity)
            thr = self.chunk_threshold
            if thr > 0:
                return {k._hashable_string_rep: v for k, v in part.items() if abs(v) > thr}
            return {k._hashable_string_rep: v for k, v in part.items()}


# ===================================================================================================
# Parallelization Thresholds, Memory Estimation, and Dynamic Target-System Calibration
# ===================================================================================================

# Active dynamic calibration parameters (populated only when optimize_errorgen_propagation_parallelization is run)
_ACTIVE_PARALLEL_THRESHOLDS: dict[str, Any] = {
    "safety_factor": 10.0,
}


def estimate_bulk_alpha_peak_memory(
    n_qubits: int,
    n_errorgens: int,
    n_outcomes: int,
    num_workers: Union[int, str, None] = "auto",
) -> int:
    """
    Estimate the additional peak memory in bytes (beyond the calling process's existing footprint)
    for bulk_alpha / bulk_alpha_pauli computation.

    Parameters
    ----------
    n_qubits : int
        Number of qubits in the circuit / stabilizer tableau.
    n_errorgens : int
        Number of elementary error generators.
    n_outcomes : int
        Number of bitstrings or Pauli strings requested.
    num_workers : int, str, or None, optional
        Target worker count (defaults to "auto").

    Returns
    -------
    int
        Estimated peak memory footprint in bytes.
    """
    w = _determine_num_workers(num_workers)
    if w < 1:
        raise ValueError(f"Worker count must be at least 1, got {w}.")
    output_matrix_bytes = n_outcomes * n_errorgens * 8
    master_base = 0  # the calling process's footprint is already excluded from available memory
    worker_base_rss = _WORKER_PROCESS_BASELINE_BYTES if w > 1 else 0
    tableau_bytes = (n_qubits ** 2 // 8 + 64) * 8
    sub_matrix_bytes = (n_outcomes * n_errorgens * 8) // max(1, w)
    total_bytes = master_base + output_matrix_bytes + (w * worker_base_rss) + (w * sub_matrix_bytes) + w * tableau_bytes
    return int(total_bytes)


def _should_parallelize_alpha(
    n_qubits: int,
    num_generators: int,
    num_outcomes: int,
    num_workers: Union[int, str, None] = "auto",
    start_method: Optional[str] = None,
) -> bool:
    r"""
    Internal helper: determines whether a sensitivity calculation (bulk_alpha or bulk_alpha_pauli)
    should be executed in parallel or routed to the zero-overhead sequential execution.

    Checks active manual configuration overrides, dynamic profiling calibration (if run),
    or the default multi-scale absolute cutoff table.
    """
    workers = _determine_num_workers(num_workers)
    if workers < 1:
        raise ValueError(f"Worker count must be at least 1, got {workers}.")
    if workers == 1:
        return False

    w = num_generators * num_outcomes
    if w <= 0:
        return False

    forced = _forced_parallelism_mode("alpha")
    if forced is not None:
        return forced

    if "min_km_alpha" in _MANUAL_PARALLEL_OVERRIDES:
        return w >= _MANUAL_PARALLEL_OVERRIDES["min_km_alpha"]

    # Check if dynamic optimizer calibration was run by the user
    if _ACTIVE_PARALLEL_THRESHOLDS.get("dynamic_calibrated", False):
        a = _ACTIVE_PARALLEL_THRESHOLDS.get("alpha_a")
        b = _ACTIVE_PARALLEL_THRESHOLDS.get("alpha_b")
        if a is not None and b is not None:
            beta = _ACTIVE_PARALLEL_THRESHOLDS.get("safety_factor", 10.0)
            tau_alpha_s = _alpha_tau_us(a, b, n_qubits) * 1e-6
            t_ipc_s = _ACTIVE_PARALLEL_THRESHOLDS.get("t_ipc_s")
            if t_ipc_s is None:
                t_ipc_s = _measure_ipc_latency(num_workers=workers, start_method=start_method)
            threshold_km = (beta * t_ipc_s) / max(1e-9, tau_alpha_s)
            return w >= threshold_km
        # If any locally-run parameter is missing, immediately fallback to the simpler table

    # Default fallback: multi-scale cutoff table
    threshold_km = DEFAULT_PARALLEL_THRESHOLDS[_qubit_scale_tier(n_qubits)]['min_km_alpha']
    return w >= threshold_km


# Number of qubits separating the two problem sizes at which the sensitivity throughput is
# measured, to fit its (linear) qubit-count scaling.
_ALPHA_CALIBRATION_QUBIT_SPAN = 16


def _calibration_tableau_and_errorgens(n: int, num_errorgens: int = 400):
    """
    A stabilizer state on exactly `n` qubits (Bell pairs on adjacent qubits, plus a Hadamard on the
    last qubit when `n` is odd) and a reproducible random set of weight-1 and weight-2 elementary error
    generators across the H, S, C and A sectors, for profiling the sensitivity calculation.
    """
    lines = []
    for i in range(0, n - 1, 2):
        lines.append(f"H {i}\nCX {i} {i+1}")
    if n % 2 == 1:
        lines.append(f"H {n - 1}")
    tableau = stim.Tableau.from_circuit(stim.Circuit("\n".join(lines)))
    assert len(tableau) == n

    rng = _np.random.default_rng(1234)
    pauli_letters = ["X", "Y", "Z"]
    types = ["H", "S", "C", "A"]
    gens = []
    for _ in range(num_errorgens):
        eg_type = str(rng.choice(types))
        weight = int(rng.choice([1, 2])) if n > 1 else 1
        targets = [int(x) for x in rng.choice(n, size=weight, replace=False)]

        p1 = ["_"] * n
        for t in targets:
            p1[t] = str(rng.choice(pauli_letters))
        p1_stim = stim.PauliString("".join(p1))

        if eg_type in ("H", "S"):
            gens.append(_LSE(eg_type, [p1_stim]))
        else:
            p2 = ["_"] * n
            for t in targets:
                p2[t] = str(rng.choice(pauli_letters))
            p2_stim = stim.PauliString("".join(p2))
            if p1_stim == p2_stim:
                alt = "Y" if p1[targets[0]] != "Y" else "Z"
                p2[targets[0]] = alt
                p2_stim = stim.PauliString("".join(p2))
            gens.append(_LSE(eg_type, [p1_stim, p2_stim]))
    return tableau, gens


def _measure_alpha_throughput_us(n: int, repeats: int = 3) -> float:
    """
    Measure the single-threaded time in microseconds per (error generator, outcome) cell of the
    sensitivity calculation `bulk_alpha` on `n` qubits. The per-call overhead is cancelled by
    differencing the times of a full and a half-size generator set (best of `repeats` each).
    """
    from .sensitivities import _raw_bulk_alpha
    tableau, all_gens = _calibration_tableau_and_errorgens(n)
    num_outcomes = 4
    bitstrings = ["0" * n] * num_outcomes
    g1 = all_gens[: len(all_gens) // 2]
    g2 = all_gens

    _raw_bulk_alpha(g1[:10], tableau, bitstrings[:1])  # warm-up

    def best_time(gens):
        best = float("inf")
        for _ in range(repeats):
            t0 = _time.perf_counter()
            _raw_bulk_alpha(gens, tableau, bitstrings)
            best = min(best, _time.perf_counter() - t0)
        return best

    t1, t2 = best_time(g1), best_time(g2)
    delta_w = (len(g2) - len(g1)) * num_outcomes
    return max(_MIN_ALPHA_TAU_US, ((t2 - t1) / max(1, delta_w)) * 1e6)


# Smallest sensitivity throughput (microseconds per cell) the model will report or use; also the floor
# applied to measurements.
_MIN_ALPHA_TAU_US = 0.1


def _alpha_tau_us(a: float, b: float, n_qubits: int) -> float:
    """The fitted sensitivity throughput a + b*n (microseconds per cell), floored at `_MIN_ALPHA_TAU_US`
    since the line is a local fit that can extrapolate to non-positive values at other qubit counts."""
    return max(_MIN_ALPHA_TAU_US, a + b * n_qubits)


def _fit_alpha_scaling(n_lo: int, tau_lo_us: float, n_hi: int, tau_hi_us: float) -> tuple[float, float]:
    """
    Fit the linear throughput model tau(n) = a + b*n (microseconds per cell) through measurements
    at two qubit counts. The slope is clamped to be non-negative (measurement noise could otherwise
    make the throughput appear to improve with size), in which case the model is the constant mean.
    """
    b = (tau_hi_us - tau_lo_us) / (n_hi - n_lo)
    if b < 0:
        return 0.5 * (tau_lo_us + tau_hi_us), 0.0
    return tau_lo_us - b * n_lo, b


def optimize_errorgen_propagation_parallelization(
    num_qubits: Optional[int] = None,
    model: Optional[Any] = None,
    circuit: Optional[Any] = None,
    num_workers: Union[int, str, None] = "auto",
    start_method: Optional[str] = None,
    set_as_default: bool = True,
) -> dict[str, Any]:
    r"""
    Empirically calibrate the single-threaded throughput of the sensitivity calculation and the
    IPC dispatch latency on the target machine, updating the active parallelization threshold for
    `bulk_alpha` and `bulk_alpha_pauli`.

    The sensitivity throughput is measured at the target qubit count and at a larger one
    (`_ALPHA_CALIBRATION_QUBIT_SPAN` qubits more), and a linear model
    :math:`\tau_\alpha(n) = a + b n` (microseconds per generator-outcome cell, with `b` clamped to
    be non-negative) is fit through the two measurements. Since the throughput is not exactly linear in
    the qubit count, this is a local fit, most accurate for problems near the calibrated size. The
    calibrated parallelization threshold
    is then :math:`K M \geq \beta T_{IPC} / \tau_\alpha(n)` for K generators and M outcomes,
    where :math:`\beta` is the safety factor stored in `_ACTIVE_PARALLEL_THRESHOLDS` (10 by default).

    Parameters
    ----------
    num_qubits : int, optional
        Qubit count to profile. Inferred from `model` or `circuit` if provided.
        At least one of `num_qubits`, `circuit`, or `model` must be provided.
    model : Model, optional
        Optional model to use for profiling.
    circuit : Circuit, optional
        Optional circuit to use for profiling.
    num_workers : int, str, or None, optional
        Worker count specification (defaults to "auto"). If this resolves to a single worker the
        IPC latency is not recorded (there is nothing to dispatch to), and is instead measured
        when it is first needed.
    start_method : str, optional
        Multiprocessing start method ('fork', 'spawn', 'forkserver').
    set_as_default : bool, optional (default True)
        If True, caches the calibrated parameters in `_ACTIVE_PARALLEL_THRESHOLDS`
        to supersede the built-in reference defaults for subsequent operations.

    Returns
    -------
    dict
        Dictionary containing the measured parameters:
        {'num_qubits': int, 'tau_alpha_us': float, 'alpha_a': float, 'alpha_b': float,
         't_ipc_s': float, 'num_workers': int, 'threshold_KM': int, 'calibration_ms': float}
        where `tau_alpha_us` is the fitted throughput at `num_qubits`, a + b * num_qubits (floored at 0.1 microseconds).

    Note
    ----
    **Scope:** the calibration affects only the parallelize-or-not decision for `bulk_alpha` and
    `bulk_alpha_pauli`. The decisions for the Magnus expansion, commuting products and layer
    propagation always use the built-in threshold table (or your manual configuration).

    **Empirical Nature and Disclaimer:**
    The linear scaling model is fit to measurements at only two problem sizes, and the IPC latency
    is measured with trivial tasks. Actual single-threaded throughput and IPC dispatch overhead
    may vary substantially across different CPU architectures (e.g. x86 AVX-512 vs.
    ARM NEON), operating systems, and memory bus topologies.

    For deterministic, manual control over parallelization cutoffs without running
    runtime profiling, users should use `configure_errorgen_propagation_parallelization()`,
    which allows setting explicit integer thresholds (e.g. `min_km_alpha`, `min_magnus_pairs`)
    or forcing parallel/serial modes directly.
    """
    t0_start = _time.perf_counter()
    workers = _determine_num_workers(num_workers)
    t_ipc_s = _measure_ipc_latency(num_workers=workers, start_method=start_method)

    if model is not None:
        if not isinstance(model, _Model):
            raise TypeError(f"Expected model to be a pygsti Model instance, got {type(model)}.")
        n = model.state_space.num_qubits
    elif circuit is not None:
        if not isinstance(circuit, _Circuit):
            raise TypeError(f"Expected circuit to be a pygsti Circuit instance, got {type(circuit)}.")
        n = circuit.num_lines
    elif num_qubits is not None:
        if not isinstance(num_qubits, (int, _np.integer)):
            raise TypeError(f"Expected num_qubits to be an integer, got {type(num_qubits)}.")
        n = int(num_qubits)
        if n < 1:
            raise ValueError(f"num_qubits must be an integer >= 1, got {n}.")
    else:
        raise ValueError("Must provide at least one of 'num_qubits', 'circuit', or 'model'.")

    n_hi = n + _ALPHA_CALIBRATION_QUBIT_SPAN
    a, b = _fit_alpha_scaling(n, _measure_alpha_throughput_us(n), n_hi, _measure_alpha_throughput_us(n_hi))
    tau_us = _alpha_tau_us(a, b, n)

    beta = _ACTIVE_PARALLEL_THRESHOLDS.get("safety_factor", 10.0)
    threshold_km = int(_math.ceil((beta * t_ipc_s) / (tau_us * 1e-6)))

    calibration_duration_ms = (_time.perf_counter() - t0_start) * 1000

    results = {
        "num_qubits": n,
        "tau_alpha_us": tau_us,
        "alpha_a": a,
        "alpha_b": b,
        "t_ipc_s": t_ipc_s,
        "num_workers": workers,
        "threshold_KM": threshold_km,
        "calibration_ms": calibration_duration_ms,
    }

    if set_as_default:
        _ACTIVE_PARALLEL_THRESHOLDS["alpha_a"] = a
        _ACTIVE_PARALLEL_THRESHOLDS["alpha_b"] = b
        if workers > 1:
            _ACTIVE_PARALLEL_THRESHOLDS["t_ipc_s"] = t_ipc_s
        else:
            # a single worker has no dispatch latency to measure; let it be measured when needed.
            _ACTIVE_PARALLEL_THRESHOLDS.pop("t_ipc_s", None)
        _ACTIVE_PARALLEL_THRESHOLDS["dynamic_calibrated"] = True

    return results


# ===================================================================================================
# Parallel Worker Callables and Public Dispatchers
# ===================================================================================================

class _BulkAlphaWorker:
    """Module-level worker task for parallel bulk_alpha (picklable under spawn)."""
    def __init__(self, tableau):
        if isinstance(tableau, stim.TableauSimulator):
            self.tableau = tableau.current_inverse_tableau() ** -1
        else:
            self.tableau = tableau

    def __call__(self, sub_bitstrings, sub_errorgens):
        from .sensitivities import _raw_bulk_alpha
        with _cyclic_gc_paused():
            return _raw_bulk_alpha(sub_errorgens, self.tableau, list(sub_bitstrings))


class _BulkAlphaPauliWorker:
    """Module-level worker task for parallel bulk_alpha_pauli (picklable under spawn)."""
    def __init__(self, tableau):
        if isinstance(tableau, stim.TableauSimulator):
            self.tableau = tableau.current_inverse_tableau() ** -1
        else:
            self.tableau = tableau

    def __call__(self, sub_paulis, sub_errorgens):
        from .sensitivities import _raw_bulk_alpha_pauli
        with _cyclic_gc_paused():
            return _raw_bulk_alpha_pauli(sub_errorgens, self.tableau, list(sub_paulis))

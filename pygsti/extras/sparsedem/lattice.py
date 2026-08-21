import os
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

import numpy as np
from collections import Counter
from scipy.stats import norm

from .estimation import estimate_dem_and_covariance, fit_specified_dem
from .utils import counts_to_arrays


def marginalize_syndrome_counts(syndrome_counts, bitmask):
    """
    Take dictionary of observed syndromes and marginalize over a subset of bits.

    Parameters:
        syndrome_counts: dict
            Observed n-bit syndrome data.
        bitmask: str
            n-bit string, indicates bits to keep.

    Returns:
        marginalized_syndrome_counts: dict
            Syndrome data marginalized over bitmask.
    """
    marginalized_counts = {}
    indices_to_keep = [i for i, b in enumerate(bitmask) if b == '1']
    for syndrome, count in syndrome_counts.items():
        marginalized_syndrome = ''.join(syndrome[i] for i in indices_to_keep)
        marginalized_counts[marginalized_syndrome] = marginalized_counts.get(marginalized_syndrome, 0) + count
    return marginalized_counts


def check_event_mask(bitmask, syndrome_counts, alpha=0.05):
    """
    Check if there is evidence of DEM events that flip all bits in the bitmask simultaneously.
    Uses a one-sided z-test. Not currently propagating any confidence through the bitmask trie.

    Parameters:
        bitmask: str or int
            Bitmask indicating the bits to keep.
        syndrome_counts: dict
            Observed n-bit syndrome data.
        alpha: float
            Statistical confidence level.

    Returns:
        event_present: bool
            True if statistically significant evidence of event.
    """
    if isinstance(bitmask, int):
        n_bits = len(next(iter(syndrome_counts)))
        bitmask = f"{bitmask:0{n_bits}b}"

    subset_counts = marginalize_syndrome_counts(syndrome_counts, bitmask)
    event_probs, cov_matrix = estimate_dem_and_covariance(subset_counts)
    p_hat = event_probs[-1]
    var = cov_matrix[-1, -1]
    if not np.isfinite(var) or var <= 0:
        return False
    z = p_hat / np.sqrt(var)
    z_threshold = norm.ppf(1 - alpha)
    return z > z_threshold


def bitmask_trie_search(n, check_flip):
    """
    Discover valid flip events using a bitmask trie.
    # TODO: add significance passing logic here. 
    # Will need to make check_flip accept a significance parameter, and will need
    # to keep a dictionary of significances. Look up "Closed hypothesis testing."
    #  
    
    Parameters:
        n: int
            Number of bits in each observation.
        check_flip: Callable[[int], bool]
            Function that returns True if the bits in the mask flip together.

    Returns:
        Set[int]: Valid event bitmasks.
    """
    valid_events = set()

    def dfs(current_mask, bit_index):
        if bit_index == n:
            return
        next_mask = current_mask | (1 << bit_index)
        if check_flip(next_mask):
            valid_events.add(next_mask)
            dfs(next_mask, bit_index + 1)
        dfs(current_mask, bit_index + 1)

    dfs(0, 0)
    return valid_events


#: Largest event-mask weight for which the batch checker will materialize a
#: 2**k marginal distribution.
MAX_BATCH_MASK_WEIGHT = 24

#: Element budget (syndromes x masks x weight) per chunk when gathering
#: marginal indices; bounds peak memory of the batch checker.
_BATCH_ELEMENT_BUDGET = 2 ** 26


def _fwht_rows(a):
    """Fast Walsh-Hadamard transform (Sylvester ordering) along the last axis."""
    a = np.array(a, dtype=float)
    n = a.shape[-1]
    h = 1
    while h < n:
        a = a.reshape(a.shape[:-1] + (n // (2 * h), 2, h))
        top = a[..., 0, :] + a[..., 1, :]
        bot = a[..., 0, :] - a[..., 1, :]
        a = np.concatenate([top[..., None, :], bot[..., None, :]],
                           axis=-2).reshape(a.shape[:-3] + (n,))
        h *= 2
    return a


def check_event_masks_batch(masks, keys, counts, alpha=0.05):
    """
    Vectorized equivalent of `check_event_mask` for many same-weight masks.

    Computes, for each mask, only what the sequential check consumes: the
    estimated probability of the top event of the marginal (the event
    flipping all masked bits together) and its delta-method variance under
    the multinomial sampling distribution. Statistically identical to
    calling `check_event_mask` per mask; NaN/degenerate marginals fail the
    check, matching the sequential behavior.

    Parameters:
        masks: Sequence[int]
            Integer event bitmasks (bit d = detector d), all of the same
            Hamming weight k <= MAX_BATCH_MASK_WEIGHT.
        keys: np.ndarray
            (K, n) uint8 array of distinct observed syndromes, one row per
            bitstring key (column j = string position j), as produced by
            `utils.counts_to_arrays`.
        counts: np.ndarray
            (K,) counts aligned with keys.
        alpha: float
            Statistical confidence level.

    Returns:
        passed: np.ndarray
            (len(masks),) boolean array, True where there is statistically
            significant evidence for the event.
        z_scores: np.ndarray
            (len(masks),) z statistics (NaN where the marginal is
            degenerate).
    """
    masks = list(masks)
    if not masks:
        return np.zeros(0, dtype=bool), np.zeros(0, dtype=float)
    num_syndromes, n = keys.shape
    weights_k = {int(m).bit_count() for m in masks}
    if len(weights_k) != 1:
        raise ValueError("check_event_masks_batch requires same-weight masks.")
    k = weights_k.pop()
    if k == 0 or k > MAX_BATCH_MASK_WEIGHT:
        raise ValueError(f"mask weight {k} outside 1..{MAX_BATCH_MASK_WEIGHT}.")

    # Integer mask bit d corresponds to syndrome-string position n - 1 - d
    # (sparsedem's reversed-bitstring convention, matching check_event_mask).
    columns = np.array([[n - 1 - b for b in range(n) if (m >> b) & 1]
                        for m in masks], dtype=np.int64)
    counts = np.asarray(counts, dtype=float)
    n_runs = counts.sum()
    size = 2 ** k
    signs = 1.0 - 2.0 * (np.bitwise_count(np.arange(size, dtype=np.uint64))
                         & np.uint64(1)).astype(float)
    place = (1 << np.arange(k, dtype=np.int64)).astype(np.int32)

    num_masks = len(masks)
    z_scores = np.full(num_masks, np.nan)
    chunk = max(1, int(_BATCH_ELEMENT_BUDGET // max(num_syndromes * k, 1)))
    for i0 in range(0, num_masks, chunk):
        cols = columns[i0:i0 + chunk]                    # (B, k)
        b_sz = cols.shape[0]
        idx = keys[:, cols].astype(np.int32) @ place     # (K, B)
        probs = np.empty((b_sz, size))
        for b in range(b_sz):
            probs[b] = np.bincount(idx[:, b], weights=counts,
                                   minlength=size)
        probs /= n_runs                                  # (B, 2^k)

        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            pol = _fwht_rows(probs)
            dep = -np.log(pol)
            att_top = 2.0 * (dep @ signs) / size
            p_top = 0.5 - 0.5 * np.exp(att_top)
            # Only the top row of the Jacobian is needed:
            # J = -(exp(att_top)/size) * FWHT(signs / pol)
            jac = -(np.exp(att_top)[:, None] / size) * _fwht_rows(signs / pol)
            j_p = np.einsum("bm,bm->b", jac, probs)
            var = (np.einsum("bm,bm,bm->b", jac, jac, probs) - j_p ** 2) / n_runs
            z_scores[i0:i0 + b_sz] = p_top / np.sqrt(var)

    z_threshold = norm.ppf(1 - alpha)
    with np.errstate(invalid="ignore"):
        passed = np.isfinite(z_scores) & (z_scores > z_threshold)
    return passed, z_scores


# Read-only per-worker state for the process-parallel path; populated by the
# pool initializer (or inherited copy-on-write under the fork start method).
_WORKER_STATE = {}


def _pool_initializer(keys, counts, alpha):
    _WORKER_STATE["keys"] = keys
    _WORKER_STATE["counts"] = counts
    _WORKER_STATE["alpha"] = alpha


def _pool_check(mask_chunk):
    return check_event_masks_batch(mask_chunk, _WORKER_STATE["keys"],
                                   _WORKER_STATE["counts"],
                                   alpha=_WORKER_STATE["alpha"])[0]


def bitmask_frontier_search(n, batch_checker, closure="prefix"):
    """
    Level-synchronous (Apriori-style) equivalent of `bitmask_trie_search`.

    Explores the same set of masks as the recursive trie search -- a mask is
    tested iff its increasing-bit prefix chain all passed -- but evaluates
    each level's candidates as one batch, which enables vectorized and
    process-parallel checking. With closure="prefix" the returned set is
    identical to `bitmask_trie_search`.

    Parameters:
        n: int
            Number of bits in each observation.
        batch_checker: Callable[[list[int]], np.ndarray]
            Maps a list of same-weight masks to a boolean pass array.
        closure: str
            "prefix" (default) reproduces the trie search exactly; "full"
            additionally requires every weight-(w-1) subset of a candidate
            to have passed (stricter Apriori pruning: tests fewer masks and
            may return a subset of the "prefix" result).

    Returns:
        Set[int]: Valid event bitmasks.
    """
    if closure not in ("prefix", "full"):
        raise ValueError(f"Unknown closure '{closure}'.")
    valid_events = set()
    survivors = []
    candidates = [1 << b for b in range(n)]
    while candidates:
        passed = np.asarray(batch_checker(candidates), dtype=bool)
        survivors = [m for m, ok in zip(candidates, passed) if ok]
        valid_events.update(survivors)
        survivor_set = set(survivors)
        candidates = []
        for m in survivors:
            for b in range(m.bit_length(), n):
                cand = m | (1 << b)
                if closure == "full":
                    # Every one-bit-removed subset must have passed.
                    mm, ok = cand, True
                    while mm:
                        low = mm & -mm
                        if (cand ^ low) not in survivor_set:
                            ok = False
                            break
                        mm ^= low
                    if not ok:
                        continue
                candidates.append(cand)
    return valid_events


def lattice_pruning_dem_estimation(syndrome_counts, confidence=0.95,
                                   return_covariance=False, n_jobs=1,
                                   closure="prefix"):
    """
    Estimate a sparse DEM from syndrome counts using the lattice pruning algorithm.

    The subset lattice is explored level-by-level (identical result to the
    recursive trie search) with a vectorized batch hypothesis check; with
    n_jobs != 1 each level's candidates are additionally distributed over a
    process pool. Runs as an ordinary function call -- no MPI or launcher
    required.

    Parameters:
        syndrome_counts: dict
            Mapping bitstrings (e.g., '0011') to counts.
        confidence: float
            Statistical confidence level for event detection.
        return_covariance: bool
            Also return the covariance matrix of the fitted probabilities.
        n_jobs: int
            1 (default) checks each level in-process with vectorized numpy;
            >1 splits levels across that many worker processes; -1 uses all
            CPUs. Worth it only for wide frontiers (hundreds of candidate
            masks) over large count tables.
        closure: str
            Lattice pruning rule, see `bitmask_frontier_search`.

    Returns:
        stim.DetectorErrorModel
    """
    n_bits = len(next(iter(syndrome_counts)))
    alpha = 1 - confidence
    keys, count_values = counts_to_arrays(syndrome_counts)
    count_values = count_values.astype(float)

    if n_jobs == 1:
        def checker(masks):
            return check_event_masks_batch(masks, keys, count_values,
                                           alpha=alpha)[0]
        masks = bitmask_frontier_search(n_bits, checker, closure=closure)
    else:
        max_workers = os.cpu_count() if n_jobs in (-1, None) else int(n_jobs)
        max_workers = max(1, max_workers)
        # Prefer fork so the (potentially large) count arrays are inherited
        # copy-on-write; fall back to the platform default elsewhere.
        methods = multiprocessing.get_all_start_methods()
        ctx = multiprocessing.get_context("fork" if "fork" in methods else None)
        with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx,
                                 initializer=_pool_initializer,
                                 initargs=(keys, count_values, alpha)) as pool:
            def checker(masks):
                if len(masks) < 2 * max_workers:
                    return check_event_masks_batch(masks, keys, count_values,
                                                   alpha=alpha)[0]
                chunks = np.array_split(np.array(masks, dtype=object),
                                        4 * max_workers)
                chunks = [list(c) for c in chunks if len(c)]
                return np.concatenate(list(pool.map(_pool_check, chunks)))
            masks = bitmask_frontier_search(n_bits, checker, closure=closure)

    return fit_specified_dem(syndrome_counts, masks,
                             return_covariance=return_covariance)

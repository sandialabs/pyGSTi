"""
Singleton search/refine methods for learning DEMs. 

When the total probability of events is not too big, then observations comprising
only single events become probable. We can identify these events, coarsely estimate
their probabilities from their observed frequencies, and then refine those estimates
using random polarizations. 
"""

import numpy as np
from collections import Counter
from typing import Tuple

from .utils import (
    bits_to_binary_number,
    binary_number_to_bits,
    rows_to_tuples,
    parity_dot,
    estimate_polarizations,
    counts_from_samples,
)


def find_atoms_by_singletons(
    counts: Counter,
    min_count: int = 5,
    max_atoms: int | None = None,
) -> tuple[list[np.ndarray], float]:
    """
    From sample counts in {0,1}^n, find candidate atoms b_i as heavy hitters.

    Parameters:
        counts: Counter
            Mapping from bitstring keys to counts.
        min_count: int
            Minimum count threshold for candidate atoms.
        max_atoms: int | None
            Maximum number of atoms to return.

    Returns:
        B_list: list[np.ndarray]
            Candidate atoms as 0/1 arrays of length n.
        P0_hat: float
            Estimated P[Y=0].
    """
    if not counts:
        raise ValueError("counts must be non-empty.")
    m = sum(counts.values())
    n = len(next(iter(counts)))
    zero = "0" * n
    zero_count = counts.get(zero, 0)
    P0_hat = zero_count / m

    items = [(vec, c) for vec, c in counts.items() if vec != zero and c >= min_count]
    items.sort(key=lambda x: x[1], reverse=True)
    if max_atoms is not None:
        items = items[:max_atoms]

    B_list = [np.array([int(bit) for bit in vec], dtype=np.uint8) for vec, _ in items]
    return B_list, P0_hat


def initial_p_from_counts(
    counts: Counter,
    B_list: list[np.ndarray],
    m: int,
    P0_hat: float,
) -> np.ndarray:
    """
    p_i_hat ≈ P(Y=b_i) / P(Y=0) = count(b_i)/count(0), first-order in small p.

    Parameters:
        counts: Counter
            Mapping from bitstring keys to counts.
        B_list: list[np.ndarray]
            Candidate atoms as 0/1 arrays of length n.
        m: int
            Total number of samples.
        P0_hat: float
            Estimated P[Y=0].

    Returns:
        p_hats: np.ndarray
            Initial probability estimates aligned with B_list.
    """
    p_hats = []
    denom = max(P0_hat, 1.0 / (m + 1))
    for b in B_list:
        bitstring = "".join(map(str, b.tolist()))
        c = counts.get(bitstring, 0)
        p_hats.append((c / m) / denom)
    return np.array(p_hats, dtype=float)


def refine_p_by_parities(
    counts: Counter,
    B_list: list[np.ndarray],
    p_init: np.ndarray | None = None,
    num_masks: int | None = None,
    ridge: float = 1e-6,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Given candidate atoms B_list (columns b_i), refine p_i by solving:
        log phi(s) = sum_i A[s,i] * w_i,  w_i = log(1-2 p_i),
    where A[s,i] = 1 if s·b_i = 1 (mod 2).
    Uses random masks s and least squares on the real system.
    Returns refined p_hat.

    Parameters:
        counts: Counter
            Mapping from bitstring keys to counts.
        B_list: list[np.ndarray]
            Candidate atoms as 0/1 arrays of length n.
        p_init: np.ndarray | None
            Optional initial estimates (unused but kept for API parity).
        num_masks: int | None
            Number of random masks to use.
        ridge: float
            Ridge regularization term for least squares.
        rng: np.random.Generator | None
            Random generator for mask sampling.

    Returns:
        p_hat: np.ndarray
            Refined probability estimates aligned with B_list.
    """
    _ = p_init
    if rng is None:
        rng = np.random.default_rng()
    if not counts:
        return np.array([], dtype=float)
    n = len(next(iter(counts)))
    K = len(B_list)
    if K == 0:
        return np.array([], dtype=float)

    if num_masks is None:
        num_masks = max(4 * K, K + 16)

    # Random parity masks define the linear system relating atoms to log-phi values.
    total_masks = (1 << n) - 1
    if total_masks <= np.iinfo(np.int64).max:
        if num_masks >= total_masks:
            masks = np.arange(1, total_masks + 1, dtype=np.int64)
        else:
            masks = rng.choice(total_masks, size=num_masks, replace=False).astype(np.int64) + 1
        bit_shifts = np.arange(n - 1, -1, -1, dtype=np.int64)
        S = ((masks[:, None] >> bit_shifts) & 1).astype(np.uint8)
    else:
        # Fallback for very large n: ensure uniqueness with a set of tuples.
        # This is efficient when num_masks << 2^n, which is the typical regime.
        masks_set = set()
        while len(masks_set) < num_masks:
            s = rng.integers(0, 2, size=n, dtype=np.uint8)
            if s.any():
                masks_set.add(tuple(s.tolist()))
        S = np.array(list(masks_set), dtype=np.uint8)
    
    # Reset num_masks to be consistent with S
    num_masks = len(S)

    # A[i, j] indicates whether mask i toggles atom j (s · b_j mod 2).
    B = np.column_stack(B_list).astype(np.uint8)
    A = S @ B % 2
    
    # Estimate phi(s) = E[(-1)^{s·Y}] from counts and map to log domain.
    phi_hat = estimate_polarizations(counts, S)
    eps = 1e-8
    phi_hat = np.clip(phi_hat, eps, 1.0 - eps)
    y = np.log(phi_hat)

    # Solve least squares for w where w_i = log(1 - 2 p_i).
    ATA = A.T @ A + ridge * np.eye(K)
    ATy = A.T @ y
    w = np.linalg.solve(ATA, ATy)

    # Map back from w to p and clip to a valid probability range.
    p_hat = (1.0 - np.exp(w)) / 2.0
    # return S
    return np.clip(p_hat, 0.0, 0.49)


def learn_atoms_and_ps(
    counts: Counter,
    min_count: int = 5,
    max_atoms: int | None = None,
    num_masks: int | None = None,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    """
    End-to-end:
      1) heavy-hitter pass -> candidate atoms B
      2) initial p via singleton ratio
      3) refine p via parity equations

    Parameters:
        counts: Counter
            Mapping from bitstring keys to counts.
        min_count: int
            Minimum count threshold for candidate atoms.
        max_atoms: int | None
            Maximum number of atoms to return.
        num_masks: int | None
            Number of random masks to use.
        seed: int | None
            Seed for random mask generation.

    Returns:
        B_hat: np.ndarray
            (n, K) uint8 matrix of atoms (columns).
        p_init: np.ndarray
            Initial estimates from singleton counts.
        p_refined: np.ndarray
            Refined estimates from parity least squares.
        meta: dict[str, object]
            Metadata containing P0_hat and counts.
    """
    rng = np.random.default_rng(seed)
    if not counts:
        raise ValueError("counts must be non-empty.")
    m = sum(counts.values())
    n = len(next(iter(counts)))

    B_list, P0_hat = find_atoms_by_singletons(
        counts,
        min_count=min_count,
        max_atoms=max_atoms,
    )
    K = len(B_list)
    B_hat = np.column_stack(B_list).astype(np.uint8) if K > 0 else np.zeros((n, 0), dtype=np.uint8)

    p_init = (
        initial_p_from_counts(counts, B_list, m, P0_hat)
        if K > 0
        else np.array([], dtype=float)
    )
    p_refined = (
        refine_p_by_parities(counts, B_list, p_init=p_init, num_masks=num_masks, rng=rng)
        if K > 0
        else p_init
    )

    meta = dict(P0_hat=P0_hat, counts=counts)
    return B_hat, p_init, p_refined, meta


def simulate_Y(
    B: np.ndarray,
    p: np.ndarray,
    m: int,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulate Y = B Z (mod 2) with independent Bernoulli(p).

    Parameters:
        B: np.ndarray
            Binary matrix of shape (n, N).
        p: np.ndarray
            Bernoulli parameters of shape (N,).
        m: int
            Number of samples to draw.
        rng: np.random.Generator | None
            Random generator for sampling.

    Returns:
        Z: np.ndarray
            Latent Bernoulli samples of shape (m, N).
        Y: np.ndarray
            Observed samples of shape (m, n) in {0,1}.
    """
    if rng is None:
        rng = np.random.default_rng()
    n, N = B.shape
    Z = rng.random((m, N)) < p
    Y = (Z @ B.T) % 2
    return Z, Y.astype(np.uint8)


def find_columns_in_B(A: np.ndarray, B: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    For each column of A, return the column index in B if present, else -1.
    Exact match. If B has duplicate columns, returns the first index.

    Parameters:
        A: np.ndarray
            Matrix whose columns are to be matched.
        B: np.ndarray
            Matrix providing candidate columns.

    Returns:
        idx: np.ndarray
            Column indices into B for each column of A (or -1).
        present: np.ndarray
            Boolean mask indicating which columns were found.
    """
    if A.shape[0] != B.shape[0]:
        raise ValueError("A and B must have the same number of rows (same column length).")

    AT = np.ascontiguousarray(A.T)
    BT = np.ascontiguousarray(B.T)

    b_map = {}
    for j in range(BT.shape[0]):
        key = BT[j].tobytes()
        if key not in b_map:
            b_map[key] = j

    idx = np.empty(AT.shape[0], dtype=np.int64)
    for i in range(AT.shape[0]):
        idx[i] = b_map.get(AT[i].tobytes(), -1)

    present = idx != -1
    return idx, present

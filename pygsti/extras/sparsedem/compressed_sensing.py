"""
Compressed sensing style sparse Walsh-Hadamard estimation with randomized low-weight masks.

This module sketches a workflow for recovering sparse attenuations by querying
only low-Hamming-weight parity masks. It never materializes the 2^N Hadamard 
matrix, instead working with a lazy masked operator.
"""

from __future__ import annotations

import itertools
import math
import random
from dataclasses import dataclass
from typing import List, Sequence

import numpy as np


def _polarization_for_mask(syndrome_counts: dict, mask_bits: Sequence[int]) -> float:
    """Compute polarization for a single parity mask represented as bit list."""
    n_runs = sum(syndrome_counts.values())
    pol = 0.0
    for syndrome, count in syndrome_counts.items():
        bits = [int(x) for x in syndrome]
        pol += count * (-1) ** np.dot(bits, mask_bits)
    return pol / n_runs


def _bitlist(mask: int, n_bits: int) -> List[int]:
    return [int(b) for b in f"{mask:0{n_bits}b}"]


def _random_low_weight_masks(n_bits: int, weight: int, num: int, rng: random.Random) -> List[int]:
    """Sample ``num`` distinct masks of a given Hamming weight."""
    if weight > n_bits:
        return []
    masks = set()
    while len(masks) < num:
        positions = rng.sample(range(n_bits), weight)
        mask = sum(1 << (n_bits - 1 - p) for p in positions)
        masks.add(mask)
    return sorted(masks)


def _all_low_weight_masks(n_bits: int, max_weight: int) -> List[int]:
    """Enumerate all masks with Hamming weight <= max_weight."""
    masks = []
    for w in range(1, max_weight + 1):
        for positions in itertools.combinations(range(n_bits), w):
            mask = sum(1 << (n_bits - 1 - p) for p in positions)
            masks.append(mask)
    return masks


@dataclass
class CSConfig:
    """Configuration for randomized low-weight compressive sensing.

    ``candidate_masks`` can be set to an explicit mask list (including
    high-weight events) to avoid enumerating all 2^N outputs.
    """

    max_weight: int = 2
    budget: int = 200
    sample_per_weight: int | None = None
    l1_penalty: float = 1e-3
    positivity: bool = True
    tol: float = 1e-6
    max_iter: int = 1_000
    seed: int | None = None
    candidate_masks: Sequence[int] | None = None


class _MaskedHadamardOperator:
    """Lightweight linear operator for the (1 - H)/2 transform without materializing it."""

    def __init__(self, row_masks: Sequence[int], col_masks: Sequence[int]):
        self.row_masks = np.array(row_masks, dtype=np.uint64)
        self.col_masks = np.array(col_masks, dtype=np.uint64)
        self.shape = (len(row_masks), len(col_masks))

    def matvec(self, x: np.ndarray) -> np.ndarray:
        out = np.zeros(self.shape[0], dtype=float)
        for i, r in enumerate(self.row_masks):
            acc = 0.0
            r_int = int(r)
            for val, c in zip(x, self.col_masks):
                acc += val * ((r_int & int(c)).bit_count() & 1)
            out[i] = acc
        return out

    def rmatvec(self, x: np.ndarray) -> np.ndarray:
        out = np.zeros(self.shape[1], dtype=float)
        for j, c in enumerate(self.col_masks):
            acc = 0.0
            c_int = int(c)
            for val, r in zip(x, self.row_masks):
                acc += val * ((int(r) & c_int).bit_count() & 1)
            out[j] = acc
        return out

    def power_iteration_lipschitz(self, iters: int = 30) -> float:
        if self.shape[1] == 0:
            return 1.0
        v = np.random.default_rng().standard_normal(self.shape[1])
        v /= np.linalg.norm(v) if np.linalg.norm(v) > 0 else 1.0
        for _ in range(iters):
            v = self.rmatvec(self.matvec(v))
            norm = np.linalg.norm(v)
            if norm == 0:
                return 1.0
            v /= norm
        wv = self.rmatvec(self.matvec(v))
        rayleigh = float(np.dot(v, wv))
        return max(rayleigh, 1e-12)


def _ista(W, y: np.ndarray, lam: float, positivity: bool, tol: float, max_iter: int):
    """ISTA solver that works with either dense matrices or lightweight operators."""

    is_dense = isinstance(W, np.ndarray)
    if is_dense:
        spectrum = np.linalg.svd(W, compute_uv=False)
        lipschitz = (spectrum.max() ** 2) if spectrum.size else 1.0
        mv = lambda v: W @ v
        mtv = lambda v: W.T @ v
        n_cols = W.shape[1]
    else:
        lipschitz = W.power_iteration_lipschitz()
        mv = W.matvec
        mtv = W.rmatvec
        n_cols = W.shape[1]

    step = 1.0 / lipschitz

    def soft_thresh(x):
        return np.sign(x) * np.maximum(np.abs(x) - lam * step, 0.0)

    x = np.zeros(n_cols, dtype=float)
    for _ in range(max_iter):
        prev = x.copy()
        grad = mtv(mv(x) - y)
        x = soft_thresh(x - step * grad)
        if positivity:
            x = np.clip(x, 0.0, None)
        delta = np.linalg.norm(x - prev)
        scale = np.linalg.norm(prev)
        if delta <= tol * max(1.0, scale):
            break
    return x


def estimate_sparse_wh(syndrome_counts: dict, config: CSConfig | None = None):
    """
    Estimate sparse attenuations using randomized low-weight polarization measurements.

    This draws a budgeted set of parity masks (favoring low Hamming weight), measures
    their polarizations from data, and solves an L1-regularized masked-Hadamard system
    to recover attenuations. The candidate set defaults to all masks up to ``max_weight``.

    Parameters
    ----------
    syndrome_counts : dict
        Observed bitstrings mapped to counts.
    config : CSConfig, optional
        Controls mask selection, sparsity penalty, and solver tolerances. If
        ``candidate_masks`` is provided in the config, that explicit list is
        used instead of enumerating low-weight masks, allowing high-weight
        events without materializing the full 2^N space.

    Returns
    -------
    event_probabilities : np.ndarray
        Estimated probabilities for the candidate mask set.
    attenuations : np.ndarray
        Corresponding attenuations.
    candidate_masks : list[int]
        The mask ordering for the outputs.
    used_masks : list[int]
        Measurement masks actually queried for polarizations.
    """
    if config is None:
        config = CSConfig()
    rng = random.Random(config.seed)

    n_bits = len(next(iter(syndrome_counts)))
    if config.candidate_masks is not None:
        candidate_masks = [int(m) for m in config.candidate_masks]
    else:
        candidate_masks = _all_low_weight_masks(n_bits, config.max_weight)

    # Build a measurement set under the budget, sampling per weight if requested.
    used_masks: list[int] = []
    remaining_budget = config.budget
    for weight in range(1, config.max_weight + 1):
        if remaining_budget <= 0:
            break
        if config.sample_per_weight is None:
            needed = min(remaining_budget, math.comb(n_bits, weight))
        else:
            needed = min(remaining_budget, config.sample_per_weight)
        sampled = _random_low_weight_masks(n_bits, weight, needed, rng)
        used_masks.extend(sampled)
        remaining_budget -= len(sampled)

    used_masks = sorted(set(used_masks))

    # Measure polarizations and form depolarizations.
    pols = []
    for mask in used_masks:
        pols.append(_polarization_for_mask(syndrome_counts, _bitlist(mask, n_bits)))
    polarizations = np.array(pols, dtype=float)

    # Avoid singularities in log; noisy near-zero pols won't dominate due to sparsity.
    polarizations = np.clip(polarizations, 1e-12, 1.0 - 1e-12)
    depols = -np.log(polarizations)

    W = _MaskedHadamardOperator(row_masks=used_masks, col_masks=candidate_masks)

    attenuations = _ista(
        W=W,
        y=depols,
        lam=config.l1_penalty,
        positivity=config.positivity,
        tol=config.tol,
        max_iter=config.max_iter,
    )

    event_probabilities = 0.5 - 0.5 * np.exp(-attenuations)
    return event_probabilities, attenuations, candidate_masks, used_masks

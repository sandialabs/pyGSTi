"""
Model validation ("consistency") tests for detector error models.

Given a candidate DEM and detector-sample data, the functions in this module
test whether the data are statistically consistent with the model. There is
no uniformly most powerful test for this problem, so we provide a battery of
tests with different power profiles:

  * exact-marginal likelihood (G-) tests on detector subsets, with a workflow
    for building the subsets (random, all-weight-k, detector-graph
    neighborhoods, spacetime structures derived from a stim circuit);
  * moment/spectrum tests comparing observed Walsh polarizations (click
    rates, pairwise and higher correlators) against closed-form predictions;
  * distribution tests on scalar-valued functions of the syndrome (Hamming
    weight, matching weight, complementary gap, ...), calibrated by DEM
    Monte Carlo;
  * decoder-based logical-error-rate consistency tests;
  * stationarity tests of the i.i.d.-shots assumption (these test the data
    collection, not the DEM itself, and are labeled accordingly).

Every test returns a `ValidationResult` carrying a p-value AND an effect-size
diagnostic that points at what disagrees, and suites of results are
aggregated with standard multiple-testing corrections.

Conventions (see `sparsedem.utils`): sample arrays are in stim column order
(column d = detector d); integer bitmasks use bit d for detector d; marginal
outcome index m has bit j set iff the j-th smallest detector in the subset
fired.
"""

import itertools
import itertools as _itertools
import math
import warnings
from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional, Sequence, Union

import numpy as np
import scipy.sparse
import scipy.stats

import stim

from . import io
from .io import dem_from_str, dem_to_dict
from .logical_decoration import (
    build_matcher,
    dem_to_check_matrix,
    _dem_for_decoding,
)

try:
    import pymatching as _pymatching
except ImportError:  # pragma: no cover - exercised only without pymatching
    _pymatching = None

try:
    from tesseract_decoder import tesseract as _tesseract
except ImportError:  # pragma: no cover - exercised only without tesseract
    _tesseract = None

#: Largest detector subset for which exact marginal distributions are
#: computed (2**MAX_MARGINAL_SIZE outcome probabilities are materialized).
MAX_MARGINAL_SIZE = 20


# ---------------------------------------------------------------------------
# Results and multiple-testing aggregation
# ---------------------------------------------------------------------------

@dataclass
class ValidationResult:
    """
    Outcome of a single model-validation test.

    Attributes:
        name: str
            Identifier of the test, encoding what was tested
            (e.g. "marginal_g[3,7,12]" or "polarization_w2[(3,7)]").
        statistic: float
            Value of the test statistic.
        pvalue: float
            p-value under the null hypothesis that the data were generated
            by the candidate DEM (or, for stationarity tests, that shots
            are i.i.d.).
        effect_size: Optional[float]
            Scalar effect size on a documented scale (e.g. total variation
            distance for marginal tests, z-score for moment tests). None if
            the test has no natural scalar effect size.
        effect_description: str
            Human-readable description of the most significant deviation,
            e.g. "pair (3, 7) is 5.2 sigma more correlated than predicted".
        num_shots: int
            Number of shots the test consumed.
        null_model: str
            What the null hypothesis is: "dem" for tests of the candidate
            DEM, "iid" for stationarity tests of the data itself.
        details: dict
            Test-specific diagnostics (residuals, per-cell counts, ...).
    """
    name: str
    statistic: float
    pvalue: float
    effect_size: Optional[float] = None
    effect_description: str = ""
    num_shots: int = 0
    null_model: str = "dem"
    details: dict = field(default_factory=dict)


def adjusted_pvalues(pvalues: np.ndarray, method: str = "fdr_bh") -> np.ndarray:
    """
    Adjust p-values for multiple testing.

    Parameters:
        pvalues: np.ndarray
            Raw p-values.
        method: str
            One of "bonferroni", "holm", "fdr_bh" (Benjamini-Hochberg).

    Returns:
        adjusted: np.ndarray
            Adjusted p-values, clipped to [0, 1], aligned with the input.
    """
    p = np.asarray(pvalues, dtype=float)
    m = p.size
    if m == 0:
        return p.copy()
    if method == "bonferroni":
        return np.clip(p * m, 0.0, 1.0)
    order = np.argsort(p)
    ranked = p[order]
    if method == "holm":
        stepped = np.maximum.accumulate(ranked * (m - np.arange(m)))
        adj = np.clip(stepped, 0.0, 1.0)
    elif method == "fdr_bh":
        stepped = ranked * m / (np.arange(m) + 1)
        adj = np.clip(np.minimum.accumulate(stepped[::-1])[::-1], 0.0, 1.0)
    else:
        raise ValueError(f"Unknown method '{method}'.")
    out = np.empty(m, dtype=float)
    out[order] = adj
    return out


@dataclass
class ValidationSuiteResult:
    """
    A collection of ValidationResults with multiple-testing aggregation.

    Attributes:
        results: list
            The individual ValidationResult objects.
    """
    results: list

    def pvalues(self) -> np.ndarray:
        return np.array([r.pvalue for r in self.results], dtype=float)

    def rejected(self, alpha: float = 0.05, method: str = "fdr_bh") -> list:
        """
        The results whose adjusted p-value falls below alpha.

        Parameters:
            alpha: float
                Significance level applied to adjusted p-values.
            method: str
                Multiple-testing correction, see `adjusted_pvalues`.

        Returns:
            rejected: list
                ValidationResults inconsistent with the model, most
                significant first.
        """
        if not self.results:
            return []
        adj = adjusted_pvalues(self.pvalues(), method)
        idx = np.nonzero(adj < alpha)[0]
        return [self.results[i] for i in idx[np.argsort(adj[idx])]]

    def summary(self, alpha: float = 0.05, method: str = "fdr_bh",
                max_rows: int = 20) -> str:
        """
        Format a human-readable summary table, most significant tests first.
        """
        if not self.results:
            return "No validation tests were run."
        adj = adjusted_pvalues(self.pvalues(), method)
        order = np.argsort(adj)
        n_rej = int(np.sum(adj < alpha))
        lines = [
            f"{len(self.results)} tests, {n_rej} rejected at alpha={alpha} "
            f"({method}); most significant first:",
            f"{'test':<44} {'p':>9} {'p_adj':>9} {'effect':>9}  description",
        ]
        for i in order[:max_rows]:
            r = self.results[i]
            eff = f"{r.effect_size:9.3g}" if r.effect_size is not None else "        -"
            lines.append(
                f"{r.name:<44.44} {r.pvalue:9.2e} {adj[i]:9.2e} {eff}  "
                f"{r.effect_description}"
            )
        if len(self.results) > max_rows:
            lines.append(f"... ({len(self.results) - max_rows} more)")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# DEM Monte Carlo and closed-form model predictions
# ---------------------------------------------------------------------------

def sample_dem(dem: stim.DetectorErrorModel, num_shots: int,
               seed: Optional[int] = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Sample detector and observable data from a DEM with stim.

    Parameters:
        dem: stim.DetectorErrorModel
        num_shots: int
        seed: Optional[int]

    Returns:
        detector_samples: np.ndarray
            (num_shots, num_detectors) uint8 array in stim column order.
        observable_samples: np.ndarray
            (num_shots, num_observables) uint8 array.
    """
    sampler = dem.compile_sampler(seed=seed)
    det, obs, _ = sampler.sample(num_shots)
    return det.astype(np.uint8), obs.astype(np.uint8)


def _fwht(values: np.ndarray) -> np.ndarray:
    """In-place-style fast Walsh-Hadamard transform (Sylvester ordering)."""
    a = np.array(values, dtype=float)
    n = a.size
    h = 1
    while h < n:
        a = a.reshape(n // (2 * h), 2, h)
        top = a[:, 0, :] + a[:, 1, :]
        bot = a[:, 0, :] - a[:, 1, :]
        a = np.concatenate([top[:, None, :], bot[:, None, :]], axis=1).reshape(n)
        h *= 2
    return a


def project_event_probabilities(dem_dict: dict, subset: Sequence[int]) -> dict:
    """
    Project a {bitmask: probability} DEM dict onto a detector subset.

    Events with identical projected masks are combined as independent
    flips: p <- p1 (1 - p2) + p2 (1 - p1). Events projecting to the empty
    mask are dropped.

    Parameters:
        dem_dict: dict
            Mapping from integer detector bitmask to event probability.
        subset: Sequence[int]
            Detector indices; sorted internally. Local bit j of a projected
            mask corresponds to the j-th smallest detector in the subset.

    Returns:
        projected: dict
            Mapping from local integer bitmask to combined flip probability.
    """
    dets = sorted(subset)
    projected: dict = {}
    for mask, prob in dem_dict.items():
        local = 0
        for j, d in enumerate(dets):
            if (mask >> d) & 1:
                local |= 1 << j
        if local == 0:
            continue
        if local in projected:
            q = projected[local]
            projected[local] = q * (1 - prob) + prob * (1 - q)
        else:
            projected[local] = prob
    return projected


def marginal_distribution(dem: Union[stim.DetectorErrorModel, dict],
                          subset: Sequence[int]) -> np.ndarray:
    """
    Exact model marginal distribution on a detector subset.

    Parameters:
        dem: stim.DetectorErrorModel or dict
            Candidate DEM, or a {bitmask: probability} dict from
            `io.dem_to_dict`.
        subset: Sequence[int]
            Detector indices, at most MAX_MARGINAL_SIZE of them.

    Returns:
        probs: np.ndarray
            2**len(subset) outcome probabilities; outcome index m has bit j
            set iff the j-th smallest detector in the subset fired.
    """
    k = len(subset)
    if k == 0 or len(set(subset)) != k:
        raise ValueError("subset must be non-empty with distinct detectors.")
    if k > MAX_MARGINAL_SIZE:
        raise ValueError(
            f"subset of size {k} exceeds MAX_MARGINAL_SIZE={MAX_MARGINAL_SIZE}; "
            "exact marginal distributions scale as 2**k."
        )
    dem_dict = dem if isinstance(dem, dict) else dem_to_dict(dem)
    projected = project_event_probabilities(dem_dict, subset)

    # attenuation(o) = sum_e w_e [|o & e| odd] = (S - (W a)[o]) / 2, where a is
    # the sparse vector of event attenuations w_e = -log1p(-2 p_e), S = sum_e
    # w_e and W is the Walsh-Hadamard matrix. One transform of the sparse
    # coefficient vector therefore replaces one full parity pass per event.
    coefficients = np.zeros(2 ** k, dtype=float)
    for local_mask, prob in projected.items():
        coefficients[local_mask] += -np.log1p(-2 * min(prob, 0.5 - 1e-15))
    attenuations = 0.5 * (coefficients.sum() - _fwht(coefficients))
    probs = _fwht(np.exp(-attenuations)) / (2 ** k)
    return np.clip(probs, 0.0, 1.0)


def marginal_counts(detector_samples: np.ndarray, subset: Sequence[int]) -> np.ndarray:
    """
    Histogram observed samples over the outcomes of a detector subset.

    Parameters:
        detector_samples: np.ndarray
            (num_shots, num_detectors) array in {0,1}, stim column order.
        subset: Sequence[int]
            Detector indices, at most MAX_MARGINAL_SIZE of them.

    Returns:
        counts: np.ndarray
            Length 2**len(subset) counts, aligned with
            `marginal_distribution`'s outcome indexing.
    """
    dets = sorted(subset)
    if len(dets) > MAX_MARGINAL_SIZE:
        raise ValueError(f"subset exceeds MAX_MARGINAL_SIZE={MAX_MARGINAL_SIZE}.")
    sub = np.asarray(detector_samples)[:, dets].astype(np.int64)
    indices = sub @ (1 << np.arange(len(dets), dtype=np.int64))
    return np.bincount(indices, minlength=2 ** len(dets)).astype(np.int64)


def polarization_from_dem(dem: Union[stim.DetectorErrorModel, dict],
                          mask_detectors: Iterable[int]) -> float:
    """
    Closed-form Walsh polarization E[(-1)^(sum of the masked detectors)].

    Equals prod_e (1 - 2 p_e)^(parity of |mask & event_e|) over DEM events.

    Parameters:
        dem: stim.DetectorErrorModel or dict
        mask_detectors: Iterable[int]
            Detector indices in the parity mask.

    Returns:
        polarization: float
    """
    dem_dict = dem if isinstance(dem, dict) else dem_to_dict(dem)
    mask = 0
    for d in mask_detectors:
        mask |= 1 << d
    pol = 1.0
    for event_mask, prob in dem_dict.items():
        if (mask & event_mask).bit_count() & 1:
            pol *= (1 - 2 * prob)
    return pol


# ---------------------------------------------------------------------------
# Goodness-of-fit primitives
# ---------------------------------------------------------------------------

def g_test(observed_counts: np.ndarray, expected_probs: np.ndarray,
           min_expected: float = 5.0) -> tuple[float, int, float]:
    """
    Multinomial likelihood-ratio (G-) test with small-cell pooling.

    Cells whose expected count falls below `min_expected` are pooled into a
    single cell before computing the statistic, so the chi-squared reference
    distribution is trustworthy at moderate sample sizes.

    Parameters:
        observed_counts: np.ndarray
            Nonnegative integer counts per outcome.
        expected_probs: np.ndarray
            Model probabilities per outcome, aligned with observed_counts.
        min_expected: float
            Pooling threshold on expected counts.

    Returns:
        statistic: float
            G = 2 sum obs * log(obs / expected).
        dof: int
            Degrees of freedom after pooling (>= 1).
        pvalue: float
    """
    obs = np.asarray(observed_counts, dtype=float)
    probs = np.asarray(expected_probs, dtype=float)
    total = obs.sum()
    if total <= 0:
        raise ValueError("observed_counts must contain at least one shot.")
    expected = probs * total
    small = expected < min_expected
    keep_obs, keep_exp = obs[~small], expected[~small]
    if small.any():
        keep_obs = np.append(keep_obs, obs[small].sum())
        keep_exp = np.append(keep_exp, expected[small].sum())
    # A pooled/kept cell can still have expected ~ 0 while observed > 0;
    # floor the expectation so impossible-under-the-model outcomes produce a
    # decisive (finite) statistic instead of a NaN.
    keep_exp = np.maximum(keep_exp, 1e-290)
    nonzero = keep_obs > 0
    statistic = 2.0 * float(np.sum(keep_obs[nonzero]
                                   * np.log(keep_obs[nonzero] / keep_exp[nonzero])))
    statistic = max(statistic, 0.0)
    dof = max(int(keep_obs.size) - 1, 1)
    pvalue = float(scipy.stats.chi2.sf(statistic, dof))
    return statistic, dof, pvalue


# ===========================================================================
# Marginal-subset construction workflow and marginal likelihood tests
# ===========================================================================

# ---------------------------------------------------------------------------
# Subset builders
# ---------------------------------------------------------------------------

def _check_subset_size(k: int, num_detectors: Optional[int] = None,
                       what: str = "k") -> None:
    """Validate a requested subset size against hard limits."""
    if k < 1:
        raise ValueError(f"{what} must be at least 1 (got {k}).")
    if k > MAX_MARGINAL_SIZE:
        raise ValueError(
            f"{what}={k} exceeds MAX_MARGINAL_SIZE={MAX_MARGINAL_SIZE}; exact "
            "marginal distributions scale as 2**k, so larger subsets are not "
            "supported."
        )
    if num_detectors is not None and k > num_detectors:
        raise ValueError(
            f"{what}={k} exceeds the number of detectors ({num_detectors})."
        )


def all_weight_k_subsets(num_detectors: int, k: int,
                         max_subsets: Optional[int] = 10000,
                         seed: Optional[int] = None) -> list:
    """
    All C(num_detectors, k) detector subsets of size k, or a random sample.

    If the number of weight-k subsets exceeds `max_subsets`, a uniform random
    sample of `max_subsets` distinct subsets is drawn instead (with a warning;
    there is no silent cap), without materializing the full combinatorial list
    when it is astronomically large.

    Parameters:
        num_detectors: int
            Number of detectors; subsets index into range(num_detectors).
        k: int
            Subset size, at most MAX_MARGINAL_SIZE.
        max_subsets: Optional[int]
            Sampling threshold (default 10000). None means "no limit": always
            enumerate everything (use with care).
        seed: Optional[int]
            Seed for the random sample, if one is needed.

    Returns:
        subsets: list[tuple[int, ...]]
            Sorted tuples of distinct detector indices, deduplicated.
    """
    _check_subset_size(k, num_detectors)
    total = math.comb(num_detectors, k)
    if max_subsets is None or total <= max_subsets:
        return [tuple(c) for c in itertools.combinations(range(num_detectors), k)]

    warnings.warn(
        f"C({num_detectors},{k}) = {total} weight-{k} subsets exceed "
        f"max_subsets={max_subsets}; returning a seeded uniform random sample "
        f"of {max_subsets} distinct subsets instead."
    )
    rng = np.random.default_rng(seed)
    if total <= 10 * max_subsets:
        # Small enough to enumerate: sample exactly without replacement.
        all_subsets = list(itertools.combinations(range(num_detectors), k))
        idx = rng.choice(total, size=max_subsets, replace=False)
        return sorted(all_subsets[i] for i in idx)
    # Astronomically many subsets: rejection-sample distinct tuples (collision
    # probability is negligible when total >= 10 * max_subsets).
    chosen: set = set()
    for _ in range(50):
        for _ in range(max_subsets - len(chosen)):
            pick = rng.choice(num_detectors, size=k, replace=False)
            chosen.add(tuple(sorted(pick.tolist())))
        if len(chosen) >= max_subsets:
            break
    return sorted(chosen)


def random_subsets(num_detectors: int, k: int, num_subsets: int,
                   seed: Optional[int] = None, min_size: int = 2) -> list:
    """
    Random detector subsets with sizes drawn uniformly in [min_size, k].

    This is the "random marginals up to weight k" mode; set min_size=k for
    fixed-size subsets. Duplicates are discarded; if the requested number of
    distinct subsets cannot be found (tiny detector counts), fewer are
    returned with a warning.

    Parameters:
        num_detectors: int
            Number of detectors; subsets index into range(num_detectors).
        k: int
            Maximum subset size, at most MAX_MARGINAL_SIZE.
        num_subsets: int
            Number of distinct subsets requested.
        seed: Optional[int]
            RNG seed.
        min_size: int
            Minimum subset size (default 2).

    Returns:
        subsets: list[tuple[int, ...]]
            Sorted tuples of distinct detector indices, deduplicated.
    """
    _check_subset_size(k, num_detectors)
    _check_subset_size(min_size, num_detectors, what="min_size")
    if min_size > k:
        raise ValueError(f"min_size={min_size} exceeds k={k}.")
    if num_subsets < 0:
        raise ValueError("num_subsets must be nonnegative.")
    rng = np.random.default_rng(seed)
    subsets: list = []
    seen: set = set()
    max_attempts = max(20 * num_subsets, 100)
    attempts = 0
    while len(subsets) < num_subsets and attempts < max_attempts:
        attempts += 1
        size = int(rng.integers(min_size, k + 1))
        pick = tuple(sorted(rng.choice(num_detectors, size=size,
                                       replace=False).tolist()))
        if pick not in seen:
            seen.add(pick)
            subsets.append(pick)
    if len(subsets) < num_subsets:
        warnings.warn(
            f"Only {len(subsets)} distinct subsets of sizes in "
            f"[{min_size}, {k}] on {num_detectors} detectors could be found "
            f"({num_subsets} were requested)."
        )
    return subsets


def _mask_to_indices(mask: int) -> list:
    """Detector indices set in an integer bitmask (bit d = detector d)."""
    indices = []
    d = 0
    while mask:
        if mask & 1:
            indices.append(d)
        mask >>= 1
        d += 1
    return indices


def detector_graph(dem: Union[stim.DetectorErrorModel, dict]) -> dict:
    """
    Adjacency structure of the detector graph of a DEM.

    Two detectors are adjacent iff they co-occur in some DEM error event.
    Every detector in range(num_detectors) appears as a key (isolated
    detectors map to an empty set).

    Parameters:
        dem: stim.DetectorErrorModel or dict
            Candidate DEM, or a {bitmask: probability} dict from
            `io.dem_to_dict` (for a dict, num_detectors is inferred from the
            highest bit present).

    Returns:
        adjacency: dict
            Mapping from detector index to the set of adjacent detectors.
    """
    if isinstance(dem, dict):
        dem_dict = dem
        num_detectors = max((m.bit_length() for m in dem_dict), default=0)
    else:
        dem_dict = io.dem_to_dict(dem)
        num_detectors = dem.num_detectors
    adjacency: dict = {d: set() for d in range(num_detectors)}
    for mask in dem_dict:
        dets = _mask_to_indices(mask)
        for a, b in itertools.combinations(dets, 2):
            adjacency[a].add(b)
            adjacency[b].add(a)
    return adjacency


def _bfs_ball(adjacency: dict, start: int, radius: int) -> dict:
    """Graph distances from `start` out to `radius` (inclusive) via BFS."""
    dist = {start: 0}
    frontier = [start]
    d = 0
    while frontier and d < radius:
        d += 1
        nxt = []
        for u in frontier:
            for v in adjacency[u]:
                if v not in dist:
                    dist[v] = d
                    nxt.append(v)
        frontier = nxt
    return dist


def graph_neighborhood_subsets(dem: Union[stim.DetectorErrorModel, dict],
                               radius: int = 1,
                               centers: Optional[Sequence[int]] = None,
                               max_size: int = MAX_MARGINAL_SIZE) -> list:
    """
    Induced r-neighborhood (BFS ball) subsets of the detector graph.

    For each center, the subset contains all detectors within graph distance
    `radius`. Subsets exceeding `max_size` are truncated to the `max_size`
    graph-closest detectors (ties broken by ascending detector index, so the
    result is deterministic). Identical subsets from different centers are
    deduplicated.

    Parameters:
        dem: stim.DetectorErrorModel or dict
            Candidate DEM (or dict from `io.dem_to_dict`).
        radius: int
            BFS radius (>= 0).
        centers: Optional[Sequence[int]]
            Center detectors (default: every detector).
        max_size: int
            Truncation size, at most MAX_MARGINAL_SIZE.

    Returns:
        subsets: list[tuple[int, ...]]
            Sorted tuples of detector indices, deduplicated.
    """
    if radius < 0:
        raise ValueError("radius must be nonnegative.")
    _check_subset_size(max_size, what="max_size")
    adjacency = detector_graph(dem)
    if centers is None:
        centers = sorted(adjacency)
    subsets: list = []
    seen: set = set()
    truncated = 0
    for c in centers:
        if c not in adjacency:
            raise ValueError(f"center {c} is not a detector of the DEM.")
        dist = _bfs_ball(adjacency, c, radius)
        members = sorted(dist, key=lambda d: (dist[d], d))
        if len(members) > max_size:
            truncated += 1
            members = members[:max_size]
        subset = tuple(sorted(members))
        if subset not in seen:
            seen.add(subset)
            subsets.append(subset)
    if truncated:
        warnings.warn(
            f"{truncated} neighborhood(s) exceeded max_size={max_size} and "
            "were truncated to the graph-closest detectors."
        )
    return subsets


def distant_subsets(dem: Union[stim.DetectorErrorModel, dict], size: int,
                    num_subsets: int, min_distance: int = 3,
                    seed: Optional[int] = None) -> list:
    """
    "Anti-structured" subsets of pairwise-distant detectors.

    Detectors in each subset are pairwise at graph distance >= min_distance in
    the detector graph (detectors in different connected components count as
    infinitely distant). Under the model such detectors are nearly
    independent, so G-tests on these subsets are powered against unmodeled
    long-range correlations. Construction is greedy and randomized; if the
    constraints admit fewer than `num_subsets` distinct subsets a warning is
    issued (and a ValueError is raised if none can be built at all).

    Parameters:
        dem: stim.DetectorErrorModel or dict
            Candidate DEM (or dict from `io.dem_to_dict`).
        size: int
            Subset size, at most MAX_MARGINAL_SIZE.
        num_subsets: int
            Number of distinct subsets requested.
        min_distance: int
            Minimum pairwise graph distance (>= 1).
        seed: Optional[int]
            RNG seed.

    Returns:
        subsets: list[tuple[int, ...]]
            Sorted tuples of detector indices, deduplicated.
    """
    _check_subset_size(size, what="size")
    if min_distance < 1:
        raise ValueError("min_distance must be at least 1.")
    if num_subsets < 0:
        raise ValueError("num_subsets must be nonnegative.")
    adjacency = detector_graph(dem)
    nodes = sorted(adjacency)
    if size > len(nodes):
        raise ValueError(
            f"size={size} exceeds the number of detectors ({len(nodes)})."
        )
    rng = np.random.default_rng(seed)
    subsets: list = []
    seen: set = set()
    max_attempts = max(20 * num_subsets, 50)
    attempts = 0
    while len(subsets) < num_subsets and attempts < max_attempts:
        attempts += 1
        forbidden: set = set()
        chosen: list = []
        for idx in rng.permutation(len(nodes)):
            d = nodes[int(idx)]
            if d in forbidden:
                continue
            chosen.append(d)
            if len(chosen) == size:
                break
            # Forbid everything within distance min_distance - 1 of d, so the
            # next pick is at distance >= min_distance from it.
            forbidden.update(_bfs_ball(adjacency, d, min_distance - 1))
        if len(chosen) == size:
            subset = tuple(sorted(chosen))
            if subset not in seen:
                seen.add(subset)
                subsets.append(subset)
    if num_subsets > 0 and not subsets:
        raise ValueError(
            f"Could not build any subset of {size} detectors pairwise at "
            f"graph distance >= {min_distance}: the constraint appears "
            "unsatisfiable for this DEM. Reduce size or min_distance."
        )
    if len(subsets) < num_subsets:
        warnings.warn(
            f"Only {len(subsets)} distinct subsets of {size} detectors "
            f"pairwise at distance >= {min_distance} were found "
            f"({num_subsets} were requested)."
        )
    return subsets


# ---------------------------------------------------------------------------
# Circuit-based spacetime subset builders
# ---------------------------------------------------------------------------

def _detector_spacetime(circuit: stim.Circuit, time_axis: int = -1) -> tuple:
    """
    Split detector coordinates of a circuit into space and time parts.

    Parameters:
        circuit: stim.Circuit
        time_axis: int
            Index of the time coordinate within each detector's coordinate
            list (default -1: the last coordinate).

    Returns:
        space: dict
            Detector index -> tuple of spatial coordinates (may be empty).
        time: dict
            Detector index -> float time coordinate.
    """
    coords = circuit.get_detector_coordinates()
    space: dict = {}
    time: dict = {}
    skipped = []
    for det, c in coords.items():
        if not c:
            skipped.append(det)
            continue
        axis = time_axis if time_axis >= 0 else len(c) + time_axis
        if axis < 0 or axis >= len(c):
            raise ValueError(
                f"time_axis={time_axis} is out of range for detector {det}, "
                f"which has {len(c)} coordinate(s)."
            )
        time[det] = float(c[axis])
        space[det] = tuple(float(v) for i, v in enumerate(c) if i != axis)
    if skipped:
        warnings.warn(
            f"{len(skipped)} detector(s) have no coordinates (e.g. "
            f"{skipped[:5]}) and are excluded from spacetime subsets. Add "
            "DETECTOR coordinate arguments to the circuit to include them."
        )
    if not time:
        raise ValueError(
            "No detector in the circuit has coordinates; spacetime subset "
            "builders require DETECTOR(...) coordinate annotations."
        )
    return space, time


def _split_oversized(members: list, max_size: int, what: str) -> list:
    """Deterministically split an ordered list into chunks of <= max_size."""
    if len(members) <= max_size:
        return [tuple(sorted(members))]
    n_chunks = math.ceil(len(members) / max_size)
    warnings.warn(
        f"A {what} of {len(members)} detectors exceeds max_size={max_size}; "
        f"splitting it deterministically into {n_chunks} contiguous chunks."
    )
    return [tuple(sorted(int(d) for d in chunk))
            for chunk in np.array_split(np.array(members), n_chunks)]


def time_column_subsets(circuit: stim.Circuit, window: Optional[int] = None,
                        time_axis: int = -1,
                        max_size: int = MAX_MARGINAL_SIZE) -> list:
    """
    Time-like subsets: detectors sharing the same spatial coordinates.

    Detectors are grouped into "columns" by identical spatial coordinates
    (all coordinates except the time axis). With window=None each column
    spanning all rounds is one subset (columns longer than max_size are split
    into consecutive-in-time chunks, with a warning). With an integer window,
    a sliding window over each column's distinct time values yields one
    subset per position, containing the detectors of `window` consecutive
    rounds. These subsets are powered against drift and other time-correlated
    deviations from the model.

    Parameters:
        circuit: stim.Circuit
            Circuit with DETECTOR coordinate annotations. By convention the
            last coordinate is time; override with time_axis.
        window: Optional[int]
            Number of consecutive distinct time values per subset (default
            None: the whole column).
        time_axis: int
            Index of the time coordinate (default -1).
        max_size: int
            Largest subset size, at most MAX_MARGINAL_SIZE.

    Returns:
        subsets: list[tuple[int, ...]]
            Sorted tuples of detector indices, deduplicated.
    """
    _check_subset_size(max_size, what="max_size")
    if window is not None and window < 1:
        raise ValueError("window must be at least 1 (or None).")
    space, time = _detector_spacetime(circuit, time_axis)
    columns: dict = {}
    for det in sorted(time):
        columns.setdefault(space[det], []).append(det)
    subsets: list = []
    seen: set = set()
    for _, dets in sorted(columns.items()):
        dets = sorted(dets, key=lambda d: (time[d], d))
        if window is None:
            candidates = _split_oversized(dets, max_size, "time column")
        else:
            times = sorted({time[d] for d in dets})
            candidates = []
            n_positions = max(len(times) - window + 1, 1)
            for i in range(n_positions):
                lo, hi = times[i], times[min(i + window, len(times)) - 1]
                members = [d for d in dets if lo <= time[d] <= hi]
                candidates.extend(_split_oversized(members, max_size,
                                                   "time window"))
        for subset in candidates:
            if subset and subset not in seen:
                seen.add(subset)
                subsets.append(subset)
    return subsets


def space_slice_subsets(circuit: stim.Circuit, window: int = 1,
                        time_axis: int = -1,
                        max_size: int = MAX_MARGINAL_SIZE) -> list:
    """
    Space-like subsets: all detectors within a window of consecutive rounds.

    Detectors are grouped by time value; each subset collects the detectors
    of `window` consecutive distinct time values (window=1: one round per
    subset). Slices exceeding max_size are split deterministically into
    spatially contiguous chunks (detectors sorted by spatial coordinates),
    with a warning.

    Parameters:
        circuit: stim.Circuit
            Circuit with DETECTOR coordinate annotations. By convention the
            last coordinate is time; override with time_axis.
        window: int
            Number of consecutive distinct time values per slice (default 1).
        time_axis: int
            Index of the time coordinate (default -1).
        max_size: int
            Largest subset size, at most MAX_MARGINAL_SIZE.

    Returns:
        subsets: list[tuple[int, ...]]
            Sorted tuples of detector indices, deduplicated.
    """
    _check_subset_size(max_size, what="max_size")
    if window < 1:
        raise ValueError("window must be at least 1.")
    space, time = _detector_spacetime(circuit, time_axis)
    times = sorted({t for t in time.values()})
    subsets: list = []
    seen: set = set()
    n_positions = max(len(times) - window + 1, 1)
    for i in range(n_positions):
        lo, hi = times[i], times[min(i + window, len(times)) - 1]
        members = [d for d in sorted(time) if lo <= time[d] <= hi]
        # Spatially contiguous ordering for deterministic splitting.
        members.sort(key=lambda d: (space[d], time[d], d))
        for subset in _split_oversized(members, max_size, "space slice"):
            if subset and subset not in seen:
                seen.add(subset)
                subsets.append(subset)
    return subsets


def spacetime_ball_subsets(circuit: stim.Circuit, space_radius: float,
                           time_radius: float,
                           centers: Optional[Sequence[int]] = None,
                           max_size: int = MAX_MARGINAL_SIZE,
                           time_axis: int = -1) -> list:
    """
    Spacetime-ball subsets around center detectors.

    For each center, the subset contains every detector within Euclidean
    spatial distance `space_radius` AND absolute time difference
    `time_radius` of it. Balls exceeding max_size are truncated to the
    closest detectors (by spatial distance, then time difference, then
    detector index), with a warning. Identical subsets are deduplicated.

    Parameters:
        circuit: stim.Circuit
            Circuit with DETECTOR coordinate annotations. By convention the
            last coordinate is time; override with time_axis.
        space_radius: float
            Euclidean radius in the spatial coordinates.
        time_radius: float
            Half-width of the time window.
        centers: Optional[Sequence[int]]
            Center detectors (default: every detector with coordinates).
        max_size: int
            Truncation size, at most MAX_MARGINAL_SIZE.
        time_axis: int
            Index of the time coordinate (default -1).

    Returns:
        subsets: list[tuple[int, ...]]
            Sorted tuples of detector indices, deduplicated.
    """
    _check_subset_size(max_size, what="max_size")
    if space_radius < 0 or time_radius < 0:
        raise ValueError("space_radius and time_radius must be nonnegative.")
    space, time = _detector_spacetime(circuit, time_axis)
    dims = {len(s) for s in space.values()}
    if len(dims) > 1:
        raise ValueError(
            f"Detectors have inconsistent spatial dimensions {sorted(dims)}; "
            "spacetime balls require uniform coordinate lengths."
        )
    dets = np.array(sorted(time), dtype=np.int64)
    spatial = np.array([space[d] for d in dets], dtype=float)
    if spatial.size == 0:
        spatial = spatial.reshape(len(dets), 0)
    tvals = np.array([time[d] for d in dets], dtype=float)
    pos = {int(d): i for i, d in enumerate(dets)}
    if centers is None:
        centers = [int(d) for d in dets]
    tol = 1e-9
    subsets: list = []
    seen: set = set()
    truncated = 0
    for c in centers:
        if c not in pos:
            raise ValueError(
                f"center {c} is not a detector with coordinates."
            )
        i = pos[c]
        sdist = np.linalg.norm(spatial - spatial[i], axis=1)
        tdist = np.abs(tvals - tvals[i])
        mask = (sdist <= space_radius + tol) & (tdist <= time_radius + tol)
        members = dets[mask]
        if members.size > max_size:
            truncated += 1
            order = np.lexsort((members, tdist[mask], sdist[mask]))
            members = members[order[:max_size]]
        subset = tuple(sorted(int(d) for d in members))
        if subset and subset not in seen:
            seen.add(subset)
            subsets.append(subset)
    if truncated:
        warnings.warn(
            f"{truncated} spacetime ball(s) exceeded max_size={max_size} and "
            "were truncated to the closest detectors."
        )
    return subsets


# ---------------------------------------------------------------------------
# Front door: subset-building workflow
# ---------------------------------------------------------------------------

#: What each build_marginal_subsets method requires and dispatches to.
_SUBSET_METHODS = {
    "all_weight_k": ("num_detectors", "all C(n,k) weight-k subsets "
                                      "(sampled above max_subsets)"),
    "random": ("num_detectors", "random subsets with sizes in [min_size, k]"),
    "neighborhood": ("dem", "detector-graph BFS balls around centers"),
    "distant": ("dem", "pairwise graph-distant detector sets"),
    "time": ("circuit", "same-space detector columns across rounds"),
    "space": ("circuit", "all detectors within a window of rounds"),
    "spacetime": ("circuit", "spatial-radius x time-window balls"),
}


def build_marginal_subsets(method: str, num_detectors: Optional[int] = None,
                           dem: Optional[Union[stim.DetectorErrorModel, dict]] = None,
                           circuit: Optional[stim.Circuit] = None,
                           **kwargs) -> list:
    """
    Build detector subsets for marginal likelihood tests (the front door).

    Dispatches to the individual subset builders; supply whichever of
    num_detectors / dem / circuit the chosen method needs (num_detectors is
    inferred from dem or circuit when possible). Every subset is a sorted
    tuple of detector indices of size at most MAX_MARGINAL_SIZE, and the
    returned list is deduplicated.

    The menu (method -> required input, extra keyword arguments):

      * "all_weight_k" -> num_detectors; k, max_subsets=10000, seed.
        Every weight-k subset, or a seeded uniform sample if there are more
        than max_subsets of them. See `all_weight_k_subsets`.
      * "random" -> num_detectors; k, num_subsets, seed, min_size=2.
        Random subsets with sizes uniform in [min_size, k]. See
        `random_subsets`.
      * "neighborhood" -> dem; radius=1, centers, max_size.
        BFS balls in the detector graph (detectors adjacent iff they share a
        DEM event). Powered against locally misestimated event probabilities.
        See `graph_neighborhood_subsets`.
      * "distant" -> dem; size, num_subsets, min_distance=3, seed.
        Pairwise graph-distant detectors, nearly independent under the model;
        powered against unmodeled long-range correlations. See
        `distant_subsets`.
      * "time" -> circuit; window, time_axis=-1, max_size.
        Same-spatial-coordinate detector columns across rounds; powered
        against drift. See `time_column_subsets`.
      * "space" -> circuit; window=1, time_axis=-1, max_size.
        All detectors within a window of consecutive rounds. See
        `space_slice_subsets`.
      * "spacetime" -> circuit; space_radius, time_radius, centers, max_size,
        time_axis=-1. Euclidean-space x time balls around center detectors.
        See `spacetime_ball_subsets`.

    Examples:
        >>> subsets = build_marginal_subsets("all_weight_k",
        ...                                  num_detectors=24, k=2)
        >>> subsets = build_marginal_subsets("neighborhood", dem=my_dem,
        ...                                  radius=2)
        >>> subsets = build_marginal_subsets("time", circuit=my_circuit,
        ...                                  window=3)

    Parameters:
        method: str
            One of "all_weight_k", "random", "neighborhood", "distant",
            "time", "space", "spacetime".
        num_detectors: Optional[int]
            Needed for "all_weight_k" and "random" (inferred from dem or
            circuit if omitted).
        dem: Optional[stim.DetectorErrorModel or dict]
            Needed for "neighborhood" and "distant" (the detector graph comes
            from DEM events).
        circuit: Optional[stim.Circuit]
            Needed for "time", "space", "spacetime" (DEMs carry no
            coordinates; detector spacetime locations come from the
            circuit's DETECTOR annotations).
        **kwargs:
            Forwarded to the underlying builder (see the menu above).

    Returns:
        subsets: list[tuple[int, ...]]
            Deduplicated detector-index subsets.
    """
    if method not in _SUBSET_METHODS:
        raise ValueError(
            f"Unknown method '{method}'. Choose one of: "
            + ", ".join(f"'{m}' ({desc})" for m, (_, desc)
                        in _SUBSET_METHODS.items()) + "."
        )
    if method in ("all_weight_k", "random"):
        if num_detectors is None:
            if dem is not None and not isinstance(dem, dict):
                num_detectors = dem.num_detectors
            elif circuit is not None:
                num_detectors = circuit.num_detectors
            elif isinstance(dem, dict):
                num_detectors = max((m.bit_length() for m in dem), default=0)
            else:
                raise ValueError(
                    f"method='{method}' needs num_detectors (or a dem/circuit "
                    "to infer it from): these builders only choose index "
                    "combinations and need to know the index range."
                )
        if method == "all_weight_k":
            return all_weight_k_subsets(num_detectors, **kwargs)
        return random_subsets(num_detectors, **kwargs)
    if method in ("neighborhood", "distant"):
        if dem is None:
            raise ValueError(
                f"method='{method}' needs a dem: these builders use the "
                "detector graph, whose edges are the detector co-occurrences "
                "in DEM error events."
            )
        if method == "neighborhood":
            return graph_neighborhood_subsets(dem, **kwargs)
        return distant_subsets(dem, **kwargs)
    # Circuit-based spacetime methods.
    if circuit is None:
        raise ValueError(
            f"method='{method}' needs a circuit: DEMs carry no spacetime "
            "structure, so detector coordinates must come from the stim "
            "circuit's DETECTOR annotations "
            "(circuit.get_detector_coordinates())."
        )
    if method == "time":
        return time_column_subsets(circuit, **kwargs)
    if method == "space":
        return space_slice_subsets(circuit, **kwargs)
    return spacetime_ball_subsets(circuit, **kwargs)


# ---------------------------------------------------------------------------
# Marginal likelihood tests
# ---------------------------------------------------------------------------

def _pooled_g_statistics(counts_matrix: np.ndarray,
                         expected: np.ndarray) -> np.ndarray:
    """G statistics for rows of a counts matrix against expected counts."""
    c = np.asarray(counts_matrix, dtype=float)
    e = np.maximum(np.asarray(expected, dtype=float), 1e-290)
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = c * np.log(c / e)
    terms[c == 0] = 0.0
    return np.maximum(2.0 * terms.sum(axis=-1), 0.0)


def marginal_likelihood_test(dem: Union[stim.DetectorErrorModel, dict],
                             detector_samples: np.ndarray,
                             subset: Sequence[int],
                             min_expected: float = 5.0,
                             bootstrap: Optional[int] = None,
                             seed: Optional[int] = None) -> ValidationResult:
    """
    G-test of observed marginal counts against the exact model marginal.

    The exact 2**k model marginal on the subset (`marginal_distribution`) is
    compared with the observed histogram (`marginal_counts`) via the
    likelihood-ratio G statistic with small-cell pooling (`g_test`). By
    default the p-value uses the chi-squared asymptotics; if `bootstrap` is
    an integer, the p-value is instead calibrated by parametric bootstrap:
    `bootstrap` multinomial(N, model_probs) count vectors are drawn (on the
    pooled cells) and the observed G is ranked among the bootstrap G's. Use
    the bootstrap when N is small or the pooled table is dominated by
    low-expectation cells, where the chi-squared reference is unreliable; at
    large N the two paths agree.

    Parameters:
        dem: stim.DetectorErrorModel or dict
            Candidate DEM (or dict from `io.dem_to_dict`).
        detector_samples: np.ndarray
            (num_shots, num_detectors) array in {0,1}, stim column order.
            Must contain at least one shot (ValueError otherwise); a single
            shot is allowed and yields an uninformative p-value near 1.
        subset: Sequence[int]
            Detector indices, at most MAX_MARGINAL_SIZE of them.
        min_expected: float
            Small-cell pooling threshold on expected counts (see `g_test`).
        bootstrap: Optional[int]
            Number of parametric-bootstrap replicates; None (default) uses
            the chi-squared asymptotics.
        seed: Optional[int]
            RNG seed for the bootstrap.

    Returns:
        result: ValidationResult
            name "marginal_g[d1,d2,...]"; statistic G; effect_size the total
            variation distance between the empirical and model marginals;
            effect_description identifying the worst cell; details with
            per-cell counts, expected counts and standardized residuals.
    """
    dets = tuple(sorted(subset))
    probs = marginal_distribution(dem, dets)
    counts = marginal_counts(detector_samples, dets)
    num_shots = int(counts.sum())
    if num_shots <= 0:
        raise ValueError("detector_samples must contain at least one shot.")

    statistic, dof, pvalue = g_test(counts, probs, min_expected=min_expected)
    method = "asymptotic"

    if bootstrap is not None:
        if bootstrap < 1:
            raise ValueError("bootstrap must be a positive integer or None.")
        method = "bootstrap"
        rng = np.random.default_rng(seed)
        # Replicate g_test's pooling so the bootstrap replicates use the same
        # statistic as the observed data.
        expected = probs * num_shots
        small = expected < min_expected
        if small.any():
            pooled_probs = np.append(probs[~small], probs[small].sum())
            pooled_counts = np.append(counts[~small], counts[small].sum())
        else:
            pooled_probs = probs
            pooled_counts = counts
        g_obs = float(_pooled_g_statistics(pooled_counts[None, :],
                                           pooled_probs * num_shots)[0])
        total_p = pooled_probs.sum()
        draw_probs = pooled_probs / total_p if total_p > 0 else \
            np.full(pooled_probs.size, 1.0 / pooled_probs.size)
        replicates = rng.multinomial(num_shots, draw_probs, size=int(bootstrap))
        g_boot = _pooled_g_statistics(replicates, pooled_probs * num_shots)
        pvalue = float((1 + np.sum(g_boot >= g_obs - 1e-9))
                       / (int(bootstrap) + 1))

    # Effect size: total variation distance between empirical and model
    # marginals (on the unpooled cells).
    empirical = counts / num_shots
    tvd = float(0.5 * np.abs(empirical - probs).sum())

    # Standardized (Pearson) residuals and the worst cell.
    expected = probs * num_shots
    with np.errstate(divide="ignore", invalid="ignore"):
        std = np.sqrt(expected * np.maximum(1.0 - probs, 0.0))
        residuals = np.where(std > 0, (counts - expected) / np.where(std > 0, std, 1.0),
                             np.where(counts > 0, np.inf, 0.0))
    worst = int(np.argmax(np.abs(residuals)))
    bits = "".join(str((worst >> j) & 1) for j in range(len(dets)))
    z = residuals[worst]
    effect_description = (
        f"outcome {bits} on detectors {dets}: observed {int(counts[worst])}, "
        f"expected {expected[worst]:.3g} ({z:+.1f} sigma)"
    )

    return ValidationResult(
        name="marginal_g[" + ",".join(map(str, dets)) + "]",
        statistic=float(statistic),
        pvalue=float(pvalue),
        effect_size=tvd,
        effect_description=effect_description,
        num_shots=num_shots,
        null_model="dem",
        details={
            "subset": dets,
            "counts": counts,
            "expected_counts": expected,
            "std_residuals": residuals,
            "dof": dof,
            "method": method,
            "num_bootstrap": int(bootstrap) if bootstrap is not None else 0,
        },
    )


def run_marginal_tests(dem: Union[stim.DetectorErrorModel, dict],
                       detector_samples: np.ndarray,
                       subsets: Sequence[Sequence[int]],
                       min_expected: float = 5.0,
                       bootstrap: Optional[int] = None,
                       seed: Optional[int] = None) -> ValidationSuiteResult:
    """
    Run marginal likelihood tests on many detector subsets.

    The DEM is converted to its dict representation once and reused across
    all subsets. Aggregate significance should be judged with the returned
    suite's multiple-testing tools (`rejected`, `summary`).

    Parameters:
        dem: stim.DetectorErrorModel or dict
            Candidate DEM (or dict from `io.dem_to_dict`).
        detector_samples: np.ndarray
            (num_shots, num_detectors) array in {0,1}, stim column order.
        subsets: Sequence[Sequence[int]]
            Detector subsets, e.g. from `build_marginal_subsets`.
        min_expected: float
            Small-cell pooling threshold, see `marginal_likelihood_test`.
        bootstrap: Optional[int]
            Bootstrap replicates per test (None: chi-squared asymptotics).
        seed: Optional[int]
            Master seed; per-test bootstrap seeds are spawned from it.

    Returns:
        suite: ValidationSuiteResult
            One ValidationResult per subset, in input order.
    """
    dem_dict = dem if isinstance(dem, dict) else io.dem_to_dict(dem)
    subsets = list(subsets)
    child_seeds = np.random.SeedSequence(seed).spawn(len(subsets))
    results = []
    for subset, child in zip(subsets, child_seeds):
        results.append(marginal_likelihood_test(
            dem_dict, detector_samples, subset, min_expected=min_expected,
            bootstrap=bootstrap,
            seed=int(child.generate_state(1)[0]),
        ))
    return ValidationSuiteResult(results)


# ===========================================================================
# Moment/spectrum tests, scalar-distribution engine, stationarity tests
# ===========================================================================

#: Element budget (shots x masks) per batch when computing sample parities;
#: bounds peak memory of the intermediate parity matrix.
_PARITY_BATCH_ELEMENTS = 32_000_000

#: A polarization z-test switches to an exact binomial test when the expected
#: number of odd-parity (or even-parity) shots falls below this threshold.
_MIN_EXPECTED_FOR_NORMAL = 10.0


# ---------------------------------------------------------------------------
# Walsh polarization (moment) tests
# ---------------------------------------------------------------------------

def _dem_as_dict(dem: Union[stim.DetectorErrorModel, dict]) -> dict:
    return dem if isinstance(dem, dict) else dem_to_dict(dem)


def _num_detectors_of(dem: Union[stim.DetectorErrorModel, dict]) -> int:
    if isinstance(dem, stim.DetectorErrorModel):
        return dem.num_detectors
    return max((m.bit_length() for m in dem), default=0)


def _mask_indicator_matrix(masks: Sequence[Sequence[int]],
                           num_columns: int) -> np.ndarray:
    """(num_columns, len(masks)) int8 indicator matrix of the parity masks."""
    mat = np.zeros((num_columns, len(masks)), dtype=np.int8)
    for j, mask in enumerate(masks):
        for d in mask:
            mat[d, j] = 1
    return mat


def predicted_polarizations(dem: Union[stim.DetectorErrorModel, dict],
                            masks: Sequence[Sequence[int]]) -> np.ndarray:
    """
    Closed-form Walsh polarizations for many masks at once.

    Vectorized equivalent of calling `validation.polarization_from_dem` for
    each mask: pol(M) = prod over events e of (1 - 2 p_e)^(parity of |M & e|).

    Parameters:
        dem: stim.DetectorErrorModel or dict
            Candidate DEM, or a {bitmask: probability} dict.
        masks: Sequence[Sequence[int]]
            Parity masks as tuples of detector indices.

    Returns:
        polarizations: np.ndarray
            len(masks) predicted polarizations in [-1, 1].
    """
    dem_dict = _dem_as_dict(dem)
    events = [(m, p) for m, p in dem_dict.items() if m]
    n_masks = len(masks)
    if n_masks == 0:
        return np.zeros(0, dtype=float)
    if not events:
        return np.ones(n_masks, dtype=float)
    width = max(max(m.bit_length() for m, _ in events),
                max((max(mask) + 1 for mask in masks if len(mask)), default=1))
    ebits = np.zeros((len(events), width), dtype=np.int8)
    probs = np.empty(len(events), dtype=float)
    for i, (m, p) in enumerate(events):
        probs[i] = p
        while m:
            low = m & -m
            ebits[i, low.bit_length() - 1] = 1
            m ^= low
    factors = 1.0 - 2.0 * probs
    zero = np.abs(factors) < 1e-300
    logabs = np.where(zero, 0.0, np.log(np.abs(np.where(zero, 1.0, factors))))
    neg = (factors < 0).astype(np.int64)
    zero_i = zero.astype(np.int64)

    mask_mat = _mask_indicator_matrix(masks, width)  # (width, n_masks)
    out = np.empty(n_masks, dtype=float)
    chunk = max(1, int(4_000_000 // max(len(events), 1)))
    for i0 in range(0, n_masks, chunk):
        cols = mask_mat[:, i0:i0 + chunk].astype(np.int32)
        parity = (ebits.astype(np.int32) @ cols) & 1        # (E, chunk)
        parity_f = parity.astype(float)
        logsum = logabs @ parity_f
        sign = np.where((neg @ parity) & 1, -1.0, 1.0)
        pol = sign * np.exp(logsum)
        pol[(zero_i @ parity) > 0] = 0.0
        out[i0:i0 + chunk] = pol
    return np.clip(out, -1.0, 1.0)


def _observed_odd_counts(detector_samples: np.ndarray,
                         masks: Sequence[Sequence[int]]) -> np.ndarray:
    """Number of shots with odd parity, per mask, batched over shots."""
    samples = np.asarray(detector_samples)
    num_shots, num_dets = samples.shape
    mask_mat = _mask_indicator_matrix(masks, num_dets)
    counts = np.zeros(len(masks), dtype=np.int64)
    batch = max(1, int(_PARITY_BATCH_ELEMENTS // max(len(masks), 1)))
    for i0 in range(0, num_shots, batch):
        block = samples[i0:i0 + batch].astype(np.int32)
        counts += ((block @ mask_mat) & 1).sum(axis=0, dtype=np.int64)
    return counts


def polarization_tests(dem: Union[stim.DetectorErrorModel, dict],
                       detector_samples: np.ndarray,
                       masks: Sequence[Sequence[int]],
                       name_prefix: str = "polarization") -> ValidationSuiteResult:
    """
    z-tests of observed vs model-predicted Walsh polarizations.

    For each mask M the model predicts pol0 = E[(-1)^(parity of M)] in
    closed form; the empirical polarization over N shots has variance
    (1 - pol0**2)/N under the null, giving a two-sided z-test per mask.
    When the expected odd- or even-parity count falls below
    `_MIN_EXPECTED_FOR_NORMAL` (in particular when pol0 is at or near +-1
    and the normal approximation breaks down), the p-value is instead an
    exact binomial tail-doubling p = min(1, 2*min(P[X <= k], P[X >= k])).

    Fully vectorized: parities of all masks are computed with one masked
    matrix product per shot batch.

    Parameters:
        dem: stim.DetectorErrorModel or dict
            Candidate DEM (or {bitmask: probability} dict).
        detector_samples: np.ndarray
            (num_shots, num_detectors) array in {0,1}, stim column order.
        masks: Sequence[Sequence[int]]
            Parity masks as tuples of detector indices.
        name_prefix: str
            Prefix for the per-mask test names.

    Returns:
        suite: ValidationSuiteResult
            One ValidationResult per mask; statistic and effect_size are the
            z-score, details record observed/predicted polarizations and
            whether the exact binomial fallback was used.
    """
    samples = np.asarray(detector_samples)
    if samples.ndim != 2:
        raise ValueError("detector_samples must be a 2D (shots, detectors) array.")
    num_shots, num_dets = samples.shape
    masks = [tuple(sorted(set(int(d) for d in mask))) for mask in masks]
    for mask in masks:
        if len(mask) == 0:
            raise ValueError("Parity masks must be non-empty.")
        if mask[-1] >= num_dets:
            raise ValueError(f"Mask {mask} references detector >= {num_dets}.")
    if not masks:
        return ValidationSuiteResult(results=[])
    if num_shots == 0:
        raise ValueError("detector_samples must contain at least one shot.")

    pol0 = predicted_polarizations(dem, masks)
    odd = _observed_odd_counts(samples, masks)
    pol_hat = 1.0 - 2.0 * odd / num_shots
    p0 = np.clip((1.0 - pol0) / 2.0, 0.0, 1.0)

    var = np.maximum((1.0 - pol0 ** 2) / num_shots, 0.0)
    z = (pol_hat - pol0) / np.sqrt(np.maximum(var, 1e-300))
    pvals = 2.0 * scipy.stats.norm.sf(np.abs(z))

    expected_min = num_shots * np.minimum(p0, 1.0 - p0)
    exact = expected_min < _MIN_EXPECTED_FOR_NORMAL
    if exact.any():
        k = odd[exact]
        pe = p0[exact]
        lo = scipy.stats.binom.cdf(k, num_shots, pe)
        hi = scipy.stats.binom.sf(k - 1, num_shots, pe)
        pvals[exact] = np.minimum(1.0, 2.0 * np.minimum(lo, hi))

    results = []
    for j, mask in enumerate(masks):
        mask_str = "(" + ",".join(str(d) for d in mask) + ")"
        results.append(ValidationResult(
            name=f"{name_prefix}[{mask_str}]",
            statistic=float(z[j]),
            pvalue=float(pvals[j]),
            effect_size=float(z[j]),
            effect_description=(
                f"mask {mask_str}: observed polarization {pol_hat[j]:.3f} vs "
                f"predicted {pol0[j]:.3f} ({z[j]:+.1f} sigma)"
            ),
            num_shots=num_shots,
            null_model="dem",
            details={
                "mask": mask,
                "observed_polarization": float(pol_hat[j]),
                "predicted_polarization": float(pol0[j]),
                "odd_count": int(odd[j]),
                "method": "binomial" if exact[j] else "normal",
            },
        ))
    return ValidationSuiteResult(results=results)


# ---------------------------------------------------------------------------
# Mask-collection builders
# ---------------------------------------------------------------------------

def weight1_masks(num_detectors: int) -> list:
    """
    All weight-1 parity masks (individual detector click rates).

    Parameters:
        num_detectors: int

    Returns:
        masks: list
            [(0,), (1,), ...].
    """
    return [(d,) for d in range(num_detectors)]


def weight2_masks(num_detectors: int, max_masks: Optional[int] = None,
                  seed: Optional[int] = None) -> list:
    """
    All weight-2 parity masks (pairwise correlators), optionally subsampled.

    Parameters:
        num_detectors: int
        max_masks: Optional[int]
            If the number of pairs exceeds this, a uniform random subset of
            this size is returned (with a warning). None keeps all pairs;
            ~300 detectors give ~45k pairs, which the vectorized
            `polarization_tests` handles comfortably.
        seed: Optional[int]
            Seed for the subsampling RNG.

    Returns:
        masks: list
            Sorted (i, j) tuples with i < j.
    """
    n = int(num_detectors)
    total = n * (n - 1) // 2
    if max_masks is None or total <= max_masks:
        return [(i, j) for i in range(n) for j in range(i + 1, n)]
    warnings.warn(
        f"weight2_masks: subsampling {max_masks} of {total} detector pairs.",
        stacklevel=2,
    )
    rng = np.random.default_rng(seed)
    if total <= 5_000_000:
        rows, cols = np.triu_indices(n, k=1)
        idx = rng.choice(total, size=max_masks, replace=False)
        return sorted((int(rows[t]), int(cols[t])) for t in idx)
    chosen: set = set()
    while len(chosen) < max_masks:
        i, j = rng.integers(0, n, size=2)
        if i != j:
            chosen.add((int(min(i, j)), int(max(i, j))))
    return sorted(chosen)


def random_masks(num_detectors: int, max_weight: int, num_masks: int,
                 seed: Optional[int] = None) -> list:
    """
    Random parity masks with weights drawn uniformly from 1..max_weight.

    Parameters:
        num_detectors: int
        max_weight: int
            Largest mask weight (clipped to num_detectors).
        num_masks: int
            Number of distinct masks requested (fewer are returned if the
            space is exhausted).
        seed: Optional[int]

    Returns:
        masks: list
            Sorted list of distinct detector-index tuples.
    """
    n = int(num_detectors)
    max_weight = max(1, min(int(max_weight), n))
    rng = np.random.default_rng(seed)
    chosen: set = set()
    for _ in range(50 * num_masks):
        if len(chosen) >= num_masks:
            break
        w = int(rng.integers(1, max_weight + 1))
        chosen.add(tuple(sorted(int(d) for d in rng.choice(n, size=w, replace=False))))
    return sorted(chosen)


def event_aligned_masks(dem: Union[stim.DetectorErrorModel, dict]) -> list:
    """
    The DEM's own event supports as parity masks.

    A mis-estimated event probability shows up first in the polarizations of
    masks with odd overlap with that event -- in particular, an odd-weight
    event's own mask.

    Parameters:
        dem: stim.DetectorErrorModel or dict

    Returns:
        masks: list
            Sorted list of distinct detector-index tuples, one per DEM event
            (empty-support events are skipped).
    """
    dem_dict = _dem_as_dict(dem)
    masks = set()
    for m in dem_dict:
        if m == 0:
            continue
        bits = []
        mm = m
        while mm:
            low = mm & -mm
            bits.append(low.bit_length() - 1)
            mm ^= low
        masks.add(tuple(bits))
    return sorted(masks)


def triple_masks(dem: Union[stim.DetectorErrorModel, dict],
                 num_masks: int = 200, seed: Optional[int] = None) -> list:
    """
    Weight-3 parity masks targeting hyperedge structure.

    Half of the masks (when available) are *connected triples* in the DEM's
    detector co-occurrence graph -- a center detector plus two of its
    neighbors. A weight-3 error mechanism present in the truth but absent
    from a graphlike candidate changes exactly these third-order Walsh
    polarizations, so connected triples are where such model errors are most
    visible. The remainder are uniformly random triples.

    Parameters:
        dem: stim.DetectorErrorModel or dict
        num_masks: int
            Total number of masks requested.
        seed: Optional[int]

    Returns:
        masks: list
            Sorted list of distinct weight-3 detector-index tuples (may be
            shorter than num_masks for small detector counts).
    """
    dem_dict = _dem_as_dict(dem)
    num_dets = _num_detectors_of(dem)
    rng = np.random.default_rng(seed)
    if num_dets < 3:
        return []

    neighbors: dict = {}
    for m in dem_dict:
        bits = []
        mm = m
        while mm:
            low = mm & -mm
            bits.append(low.bit_length() - 1)
            mm ^= low
        for a, b in _itertools.combinations(bits, 2):
            neighbors.setdefault(a, set()).add(b)
            neighbors.setdefault(b, set()).add(a)

    target_connected = num_masks // 2
    centers = [c for c, nb in neighbors.items() if len(nb) >= 2]
    connected: set = set()
    total_connected = sum(len(neighbors[c]) * (len(neighbors[c]) - 1) // 2
                          for c in centers)
    if total_connected <= max(4 * target_connected, 1000):
        for c in centers:
            for a, b in _itertools.combinations(sorted(neighbors[c]), 2):
                connected.add(tuple(sorted((c, a, b))))
        if len(connected) > target_connected:
            idx = rng.choice(len(connected), size=target_connected, replace=False)
            all_conn = sorted(connected)
            connected = {all_conn[i] for i in idx}
    else:
        attempts = 0
        while len(connected) < target_connected and attempts < 50 * target_connected:
            attempts += 1
            c = centers[int(rng.integers(0, len(centers)))]
            nb = sorted(neighbors[c])
            a, b = rng.choice(len(nb), size=2, replace=False)
            connected.add(tuple(sorted((c, nb[int(a)], nb[int(b)]))))

    masks = set(connected)
    max_total = num_dets * (num_dets - 1) * (num_dets - 2) // 6
    attempts = 0
    while len(masks) < min(num_masks, max_total) and attempts < 50 * num_masks:
        attempts += 1
        masks.add(tuple(sorted(int(d) for d in
                               rng.choice(num_dets, size=3, replace=False))))
    return sorted(masks)


def run_polarization_battery(dem: Union[stim.DetectorErrorModel, dict],
                             detector_samples: np.ndarray,
                             collections: Sequence[str] = ("weight1", "weight2",
                                                           "events", "triples"),
                             max_weight2_masks: Optional[int] = None,
                             num_triple_masks: int = 200,
                             seed: Optional[int] = None) -> ValidationSuiteResult:
    """
    Run polarization tests over standard mask collections.

    Parameters:
        dem: stim.DetectorErrorModel or dict
        detector_samples: np.ndarray
            (num_shots, num_detectors) array in {0,1}, stim column order.
        collections: Sequence[str]
            Any of "weight1", "weight2", "events", "triples".
        max_weight2_masks: Optional[int]
            Cap on the weight-2 collection (see `weight2_masks`).
        num_triple_masks: int
            Size of the triple collection (see `triple_masks`).
        seed: Optional[int]
            Seed for the subsampling RNGs.

    Returns:
        suite: ValidationSuiteResult
            Concatenation of the per-collection suites; names are prefixed
            "polarization_w1" / "polarization_w2" / "polarization_event" /
            "polarization_w3".
    """
    num_dets = np.asarray(detector_samples).shape[1]
    results = []
    for coll in collections:
        if coll == "weight1":
            masks, prefix = weight1_masks(num_dets), "polarization_w1"
        elif coll == "weight2":
            masks = weight2_masks(num_dets, max_masks=max_weight2_masks, seed=seed)
            prefix = "polarization_w2"
        elif coll == "events":
            masks = [m for m in event_aligned_masks(dem) if m[-1] < num_dets]
            prefix = "polarization_event"
        elif coll == "triples":
            masks = [m for m in triple_masks(dem, num_masks=num_triple_masks,
                                             seed=seed) if m[-1] < num_dets]
            prefix = "polarization_w3"
        else:
            raise ValueError(f"Unknown mask collection '{coll}'.")
        if masks:
            results.extend(
                polarization_tests(dem, detector_samples, masks,
                                   name_prefix=prefix).results)
    return ValidationSuiteResult(results=results)


# ---------------------------------------------------------------------------
# Scalar-function distribution tests
# ---------------------------------------------------------------------------

def hamming_weight(detector_samples: np.ndarray) -> np.ndarray:
    """
    Per-shot Hamming weight (total number of detector clicks).

    Parameters:
        detector_samples: np.ndarray
            (num_shots, num_detectors) array in {0,1}.

    Returns:
        weights: np.ndarray
            (num_shots,) integer array.
    """
    return np.asarray(detector_samples).sum(axis=1, dtype=np.int64)


def _pooled_two_sample_chi2(obs_vals: np.ndarray, null_vals: np.ndarray,
                            num_bins: Optional[int],
                            discrete_threshold: int) -> tuple[float, float, dict]:
    """2 x B contingency chi-squared with adjacent small-cell pooling."""
    n_obs, n_null = obs_vals.size, null_vals.size
    pooled = np.concatenate([obs_vals, null_vals])
    uniq = np.unique(pooled)
    if uniq.size <= max(num_bins or 0, discrete_threshold):
        # Discrete cells: one per observed value.
        obs_c = np.array([(obs_vals == u).sum() for u in uniq], dtype=float)
        null_c = np.array([(null_vals == u).sum() for u in uniq], dtype=float)
        bin_labels = [f"{u:g}" for u in uniq]
    else:
        nb = num_bins or 20
        edges = np.unique(np.quantile(pooled, np.linspace(0, 1, nb + 1)))
        edges[0], edges[-1] = -np.inf, np.inf
        obs_c = np.histogram(obs_vals, bins=edges)[0].astype(float)
        null_c = np.histogram(null_vals, bins=edges)[0].astype(float)
        bin_labels = [f"[{edges[i]:.4g},{edges[i + 1]:.4g})"
                      for i in range(len(edges) - 1)]

    # Pool adjacent cells so every expected count is at least 5 in the
    # smaller row of the contingency table.
    total = float(n_obs + n_null)
    min_col_total = 5.0 * total / max(min(n_obs, n_null), 1)
    pooled_obs, pooled_null, pooled_labels = [], [], []
    acc_o = acc_n = 0.0
    acc_l: list = []
    for o, nl, lab in zip(obs_c, null_c, bin_labels):
        acc_o += o
        acc_n += nl
        acc_l.append(lab)
        if acc_o + acc_n >= min_col_total:
            pooled_obs.append(acc_o)
            pooled_null.append(acc_n)
            pooled_labels.append("+".join(acc_l))
            acc_o = acc_n = 0.0
            acc_l = []
    if acc_l:
        if pooled_obs:
            pooled_obs[-1] += acc_o
            pooled_null[-1] += acc_n
            pooled_labels[-1] += "+" + "+".join(acc_l)
        else:
            pooled_obs, pooled_null = [acc_o], [acc_n]
            pooled_labels = ["+".join(acc_l)]
    obs_c = np.array(pooled_obs)
    null_c = np.array(pooled_null)

    details = {"bin_labels": pooled_labels,
               "observed_counts": obs_c.tolist(),
               "null_counts": null_c.tolist()}
    if obs_c.size < 2:
        details["note"] = "fewer than 2 usable bins after pooling"
        details["standardized_residuals"] = [0.0] * obs_c.size
        return 0.0, 1.0, details
    col_tot = obs_c + null_c
    exp_obs = col_tot * n_obs / total
    exp_null = col_tot * n_null / total
    stat = float(np.sum((obs_c - exp_obs) ** 2 / exp_obs
                        + (null_c - exp_null) ** 2 / exp_null))
    dof = obs_c.size - 1
    pvalue = float(scipy.stats.chi2.sf(stat, dof))
    with np.errstate(divide="ignore", invalid="ignore"):
        resid = (obs_c - exp_obs) / np.sqrt(
            exp_obs * (1 - n_obs / total) * (1 - col_tot / total))
    details["standardized_residuals"] = np.nan_to_num(resid).tolist()
    details["dof"] = dof
    return stat, pvalue, details


def scalar_distribution_test(dem: stim.DetectorErrorModel,
                             detector_samples: np.ndarray,
                             func: Callable[[np.ndarray], np.ndarray], *,
                             num_null_shots: Optional[int] = None,
                             method: str = "auto",
                             num_bins: Optional[int] = None,
                             seed: Optional[int] = None,
                             name: Optional[str] = None) -> ValidationResult:
    """
    Test whether a scalar function of the syndrome matches its DEM-implied
    distribution.

    The null distribution of func is obtained by Monte Carlo from the DEM
    (`validation.sample_dem`), then observed and null samples are compared
    with a two-sample test.

    Methods:
        "chi2": binned/discrete two-sample chi-squared contingency test with
            adjacent small-cell pooling (expected >= 5 per cell). Discrete
            values become their own cells; continuous data are quantile-
            binned into `num_bins` (default 20) bins.
        "ks": two-sample Kolmogorov-Smirnov (scipy, asymptotic mode). With
            discrete/tied data the KS test is conservative -- its true
            type-I error rate is below nominal -- which is safe but loses
            some power.
        "mean": z-test of the observed mean against the null mean/SD,
            including the Monte Carlo uncertainty of the null mean. Highest
            power against pure location shifts.
        "auto": "chi2" when the pooled support has at most
            max(num_bins, 50) distinct values, else "ks".

    With a single observed shot every method degrades, so S == 1 falls back
    to a two-sided tail probability of the single value within the null
    sample (method "tail" in details).

    Parameters:
        dem: stim.DetectorErrorModel
            Candidate DEM (the null model).
        detector_samples: np.ndarray
            (num_shots, num_detectors) observed array in {0,1}, stim column
            order.
        func: Callable
            Vectorized map from an (S, D) uint8 array to an (S,) array of
            per-shot scalars. May close over the DEM (e.g. decoder-based
            statistics).
        num_null_shots: Optional[int]
            Monte Carlo null sample size; default max(10*S, 20000) capped at
            1e6 (recorded in details).
        method: str
            "auto", "chi2", "ks", or "mean".
        num_bins: Optional[int]
            Bin count for the chi2 method on continuous data.
        seed: Optional[int]
            Seed for the null Monte Carlo.
        name: Optional[str]
            Label for the scalar (defaults to func.__name__).

    Returns:
        result: ValidationResult
            effect_size is the observed-mean shift in units of the null SD;
            details include null/observed summary statistics, the method
            used, and per-bin residuals for the chi2 method.
    """
    samples = np.asarray(detector_samples)
    if samples.ndim != 2:
        raise ValueError("detector_samples must be a 2D (shots, detectors) array.")
    num_shots = samples.shape[0]
    if num_shots == 0:
        raise ValueError("detector_samples must contain at least one shot.")

    if num_null_shots is None:
        num_null_shots = int(min(max(10 * num_shots, 20_000), 1_000_000))
        null_shots_rule = "default: max(10*num_shots, 20000) capped at 1e6"
    else:
        num_null_shots = int(num_null_shots)
        null_shots_rule = "user-specified"

    null_det, _ = sample_dem(dem, num_null_shots, seed=seed)
    obs_vals = np.asarray(func(samples), dtype=float).ravel()
    null_vals = np.asarray(func(null_det), dtype=float).ravel()

    obs_mean = float(np.mean(obs_vals))
    null_mean = float(np.mean(null_vals))
    null_sd = float(np.std(null_vals, ddof=1)) if null_vals.size > 1 else 0.0
    if null_sd > 0:
        effect = (obs_mean - null_mean) / null_sd
    elif np.isclose(obs_mean, null_mean):
        effect = 0.0
    else:
        effect = float(np.sign(obs_mean - null_mean) * np.inf)

    details = {
        "num_null_shots": num_null_shots,
        "num_null_shots_rule": null_shots_rule,
        "null_mean": null_mean, "null_sd": null_sd,
        "null_min": float(np.min(null_vals)), "null_max": float(np.max(null_vals)),
        "observed_mean": obs_mean,
        "observed_sd": float(np.std(obs_vals, ddof=1)) if num_shots > 1 else 0.0,
    }

    if num_shots == 1:
        method_used = "tail"
        x = obs_vals[0]
        p_hi = float(np.mean(null_vals >= x))
        p_lo = float(np.mean(null_vals <= x))
        pvalue = min(1.0, 2.0 * min(p_hi, p_lo))
        statistic = (x - null_mean) / null_sd if null_sd > 0 else 0.0
        details["tail_probabilities"] = {"lower": p_lo, "upper": p_hi}
    else:
        if method == "auto":
            n_unique = np.unique(np.concatenate([obs_vals, null_vals])).size
            method_used = "chi2" if n_unique <= max(num_bins or 0, 50) else "ks"
        elif method in ("chi2", "ks", "mean"):
            method_used = method
        else:
            raise ValueError(f"Unknown method '{method}'.")

        if method_used == "chi2":
            statistic, pvalue, chi2_details = _pooled_two_sample_chi2(
                obs_vals, null_vals, num_bins, discrete_threshold=50)
            details.update(chi2_details)
        elif method_used == "ks":
            # Asymptotic mode keeps large samples fast; with ties this is
            # conservative (see the method notes in the docstring).
            res = scipy.stats.ks_2samp(obs_vals, null_vals, method="asymp")
            statistic, pvalue = float(res.statistic), float(res.pvalue)
        else:  # mean
            se = null_sd * np.sqrt(1.0 / num_shots + 1.0 / num_null_shots)
            if se > 0:
                statistic = (obs_mean - null_mean) / se
                pvalue = float(2.0 * scipy.stats.norm.sf(abs(statistic)))
            elif np.isclose(obs_mean, null_mean):
                statistic, pvalue = 0.0, 1.0
            else:
                statistic = float(np.sign(obs_mean - null_mean) * np.inf)
                pvalue = 0.0
    details["method_used"] = method_used

    label = name or getattr(func, "__name__", "scalar")
    if null_sd > 0:
        direction = "high" if effect >= 0 else "low"
        description = (f"{label} runs {abs(effect):.2f} null-SDs {direction}; "
                       f"obs mean {obs_mean:.4g} vs model {null_mean:.4g}")
    else:
        description = (f"{label} is constant under the model "
                       f"(obs mean {obs_mean:.4g} vs model {null_mean:.4g})")

    return ValidationResult(
        name=f"scalar[{label}]",
        statistic=float(statistic),
        pvalue=float(np.clip(pvalue, 0.0, 1.0)),
        effect_size=float(effect),
        effect_description=description,
        num_shots=num_shots,
        null_model="dem",
        details=details,
    )


def hamming_weight_test(dem: stim.DetectorErrorModel,
                        detector_samples: np.ndarray, **kwargs) -> ValidationResult:
    """
    Distribution test on the per-shot Hamming weight (total click count).

    Convenience wrapper of `scalar_distribution_test` with
    `func=hamming_weight`; keyword arguments are forwarded.

    Parameters:
        dem: stim.DetectorErrorModel
        detector_samples: np.ndarray
            (num_shots, num_detectors) array in {0,1}, stim column order.

    Returns:
        result: ValidationResult
    """
    kwargs.setdefault("name", "hamming_weight")
    return scalar_distribution_test(dem, detector_samples, hamming_weight, **kwargs)


# ---------------------------------------------------------------------------
# Stationarity tests (null: shots are i.i.d.; the DEM is not involved)
# ---------------------------------------------------------------------------

def _block_slices(num_shots: int, num_blocks: int) -> list:
    """Boundaries of near-equal consecutive shot blocks."""
    edges = np.linspace(0, num_shots, num_blocks + 1).astype(int)
    return [(int(edges[b]), int(edges[b + 1])) for b in range(num_blocks)
            if edges[b + 1] > edges[b]]


def click_rate_drift_test(detector_samples: np.ndarray,
                          num_blocks: int = 20) -> ValidationResult:
    """
    Test for drift of the total click rate over consecutive shot blocks.

    Splits the (ordered) shots into `num_blocks` consecutive blocks and
    tests homogeneity of the mean per-shot click count (Hamming weight)
    across blocks; additionally reports a trend z-statistic from a
    regression of block mean on block position (powered against monotone
    drifts).

    The block-mean variance is estimated empirically from the per-shot
    weights rather than assumed binomial: multi-detector DEM events
    correlate clicks within a shot, so a per-(detector, shot) binomial
    model would be anti-conservative on perfectly stationary data.

    Parameters:
        detector_samples: np.ndarray
            (num_shots, num_detectors) array in {0,1}, in shot order.
        num_blocks: int
            Number of consecutive blocks (reduced if there are few shots).

    Returns:
        result: ValidationResult
            null_model="iid"; statistic and pvalue come from the
            homogeneity chi-squared on block means; effect_size is the
            fitted slope of the mean shot weight per 1000 shots; details
            include block means and the trend z and p-value.
    """
    samples = np.asarray(detector_samples)
    num_shots, num_dets = samples.shape
    weights = samples.sum(axis=1, dtype=np.int64)
    blocks = _block_slices(num_shots, max(1, min(num_blocks, num_shots)))
    base = dict(name="stationarity_click_rate", num_shots=num_shots,
                null_model="iid")

    if len(blocks) < 2:
        return ValidationResult(statistic=0.0, pvalue=1.0, effect_size=0.0,
                                effect_description="too few shots to test drift",
                                details={"num_blocks": len(blocks)}, **base)

    sizes = np.array([b - a for a, b in blocks], dtype=float)
    means = np.array([weights[a:b].mean() for a, b in blocks])
    centers = np.array([(a + b - 1) / 2.0 for a, b in blocks])
    grand_mean = float(weights.mean())
    variance = float(np.var(weights, ddof=1))

    if variance <= 0.0:
        return ValidationResult(statistic=0.0, pvalue=1.0, effect_size=0.0,
                                effect_description=(
                                    "per-shot weight is constant; drift "
                                    "untestable"),
                                details={"num_blocks": len(blocks),
                                         "grand_mean": grand_mean}, **base)

    chi2_stat = float(np.sum(sizes * (means - grand_mean) ** 2) / variance)
    dof = len(blocks) - 1
    pvalue = float(scipy.stats.chi2.sf(chi2_stat, dof))

    # Weighted least squares of block mean on block center; under the
    # i.i.d. null Var(mean_b) = variance / n_b, so z = slope * sqrt(Sxx).
    wgt = sizes / variance
    xbar = np.sum(wgt * centers) / np.sum(wgt)
    sxx = np.sum(wgt * (centers - xbar) ** 2)
    slope = float(np.sum(wgt * (centers - xbar) * means) / sxx)
    trend_z = float(slope * np.sqrt(sxx))
    trend_p = float(2.0 * scipy.stats.norm.sf(abs(trend_z)))

    return ValidationResult(
        statistic=chi2_stat,
        pvalue=pvalue,
        effect_size=slope * 1e3,
        effect_description=(
            f"mean shot weight slope {slope * 1e3:+.3g} per 10^3 shots "
            f"({trend_z:+.1f} sigma trend); block means "
            f"{means.min():.4f}..{means.max():.4f}"
        ),
        details={
            "num_blocks": len(blocks), "dof": dof,
            "block_sizes": sizes.tolist(), "block_means": means.tolist(),
            "grand_mean": grand_mean, "weight_variance": variance,
            "trend_slope_per_shot": slope, "trend_z": trend_z,
            "trend_pvalue": trend_p,
        },
        **base,
    )


def polarization_drift_test(detector_samples: np.ndarray,
                            masks: Optional[Sequence[Sequence[int]]] = None,
                            num_blocks: int = 10) -> ValidationResult:
    """
    Test for drift of parity-mask polarizations over consecutive blocks.

    Catches nonstationarity that conserves the total click rate (e.g. rate
    shifting between detectors). For each mask, block odd-parity counts are
    tested for homogeneity against the pooled rate with a binomial
    chi-squared; the statistics are summed over masks.

    Parameters:
        detector_samples: np.ndarray
            (num_shots, num_detectors) array in {0,1}, in shot order.
        masks: Optional[Sequence[Sequence[int]]]
            Parity masks; default is a handful of weight-1 masks on the
            highest-rate detectors plus weight-2 masks on their pairs.
        num_blocks: int
            Number of consecutive blocks.

    Returns:
        result: ValidationResult
            null_model="iid"; statistic is the summed chi-squared,
            effect_size the largest single block z-deviation; details hold
            per-mask chi-squared values and p-values.
    """
    samples = np.asarray(detector_samples)
    num_shots, num_dets = samples.shape
    base = dict(name="stationarity_polarization_drift", num_shots=num_shots,
                null_model="iid")

    if masks is None:
        rates = samples.mean(axis=0) if num_shots else np.zeros(num_dets)
        top = [int(d) for d in np.argsort(rates)[::-1][:4] if rates[d] > 0]
        masks = [(d,) for d in top]
        masks += [tuple(sorted(pair)) for pair in
                  _itertools.combinations(top[:3], 2)]
    masks = [tuple(sorted(set(int(d) for d in m))) for m in masks if len(m)]

    blocks = _block_slices(num_shots, max(1, min(num_blocks, num_shots)))
    if len(blocks) < 2 or not masks:
        return ValidationResult(statistic=0.0, pvalue=1.0, effect_size=0.0,
                                effect_description=(
                                    "too few shots or no usable masks"),
                                details={"num_blocks": len(blocks),
                                         "masks": masks}, **base)

    total_stat, total_dof = 0.0, 0
    per_mask: dict = {}
    worst_z, worst_desc = 0.0, ""
    for mask in masks:
        parity = samples[:, list(mask)].sum(axis=1) & 1
        k_blocks = np.array([parity[a:b].sum() for a, b in blocks], dtype=float)
        n_blocks_arr = np.array([b - a for a, b in blocks], dtype=float)
        q = k_blocks.sum() / n_blocks_arr.sum()
        if q <= 0.0 or q >= 1.0:
            per_mask[mask] = {"skipped": True, "pooled_rate": float(q)}
            continue
        var_b = n_blocks_arr * q * (1.0 - q)
        z_b = (k_blocks - n_blocks_arr * q) / np.sqrt(var_b)
        stat_m = float(np.sum(z_b ** 2))
        dof_m = len(blocks) - 1
        p_m = float(scipy.stats.chi2.sf(stat_m, dof_m))
        per_mask[mask] = {"chi2": stat_m, "dof": dof_m, "pvalue": p_m,
                          "block_rates": (k_blocks / n_blocks_arr).tolist()}
        total_stat += stat_m
        total_dof += dof_m
        b_worst = int(np.argmax(np.abs(z_b)))
        if abs(z_b[b_worst]) > abs(worst_z):
            worst_z = float(z_b[b_worst])
            worst_desc = (
                f"mask {mask}: block {b_worst} parity rate "
                f"{k_blocks[b_worst] / n_blocks_arr[b_worst]:.3f} vs pooled "
                f"{q:.3f} ({worst_z:+.1f} sigma)"
            )

    if total_dof == 0:
        return ValidationResult(statistic=0.0, pvalue=1.0, effect_size=0.0,
                                effect_description=(
                                    "all masks have constant parity; drift "
                                    "untestable"),
                                details={"per_mask": per_mask}, **base)
    pvalue = float(scipy.stats.chi2.sf(total_stat, total_dof))
    return ValidationResult(
        statistic=total_stat,
        pvalue=pvalue,
        effect_size=worst_z,
        effect_description=worst_desc,
        details={"num_blocks": len(blocks), "dof": total_dof,
                 "per_mask": per_mask},
        **base,
    )


def shot_autocorrelation_test(detector_samples: np.ndarray, max_lag: int = 1,
                              num_permutations: int = 200,
                              seed: Optional[int] = None) -> ValidationResult:
    """
    Permutation test for shot-to-shot autocorrelation of the Hamming weight.

    Computes the lag-l autocorrelation of the per-shot click count for
    l = 1..max_lag; the statistic is the largest absolute value. The
    p-value comes from shuffling the shot order (the denominator of the
    autocorrelation is permutation-invariant, so each permutation costs one
    shuffle plus max_lag dot products).

    Parameters:
        detector_samples: np.ndarray
            (num_shots, num_detectors) array in {0,1}, in shot order.
        max_lag: int
            Largest lag tested.
        num_permutations: int
            Number of random shuffles for the null.
        seed: Optional[int]

    Returns:
        result: ValidationResult
            null_model="iid"; statistic is max_l |r_l|, effect_size the
            signed correlation at the most extreme lag; details include the
            per-lag correlations.
    """
    samples = np.asarray(detector_samples)
    num_shots = samples.shape[0]
    weights = samples.sum(axis=1, dtype=np.int64).astype(float)
    base = dict(name="stationarity_autocorrelation", num_shots=num_shots,
                null_model="iid")

    max_lag = max(1, int(max_lag))
    centered = weights - weights.mean()
    denom = float(np.dot(centered, centered))
    if num_shots <= max_lag + 1 or denom <= 0.0:
        return ValidationResult(statistic=0.0, pvalue=1.0, effect_size=0.0,
                                effect_description=(
                                    "too few shots or constant weight; "
                                    "autocorrelation untestable"),
                                details={"max_lag": max_lag}, **base)

    corrs = np.array([float(np.dot(centered[l:], centered[:-l])) / denom
                      for l in range(1, max_lag + 1)])
    observed = float(np.max(np.abs(corrs)))
    best_lag = int(np.argmax(np.abs(corrs))) + 1

    rng = np.random.default_rng(seed)
    exceed = 0
    for _ in range(num_permutations):
        perm = rng.permutation(centered)
        stat = max(abs(float(np.dot(perm[l:], perm[:-l]))) / denom
                   for l in range(1, max_lag + 1))
        if stat >= observed - 1e-12:
            exceed += 1
    pvalue = (1.0 + exceed) / (1.0 + num_permutations)

    return ValidationResult(
        statistic=observed,
        pvalue=pvalue,
        effect_size=float(corrs[best_lag - 1]),
        effect_description=(
            f"lag-{best_lag} autocorrelation of shot weight "
            f"r={corrs[best_lag - 1]:+.3f}"
        ),
        details={"max_lag": max_lag, "correlations": corrs.tolist(),
                 "num_permutations": num_permutations},
        **base,
    )


def run_stationarity_battery(detector_samples: np.ndarray,
                             num_blocks: int = 20,
                             masks: Optional[Sequence[Sequence[int]]] = None,
                             max_lag: int = 1, num_permutations: int = 200,
                             seed: Optional[int] = None) -> ValidationSuiteResult:
    """
    Run the stationarity tests of the i.i.d.-shots assumption.

    Parameters:
        detector_samples: np.ndarray
            (num_shots, num_detectors) array in {0,1}, in shot order.
        num_blocks: int
            Blocks for `click_rate_drift_test` (`polarization_drift_test`
            uses at most 10).
        masks: Optional[Sequence[Sequence[int]]]
            Parity masks for `polarization_drift_test` (default: automatic).
        max_lag: int
            Largest lag for `shot_autocorrelation_test`.
        num_permutations: int
            Permutations for `shot_autocorrelation_test`.
        seed: Optional[int]

    Returns:
        suite: ValidationSuiteResult
            Results of the three stationarity tests, all with
            null_model="iid".
    """
    return ValidationSuiteResult(results=[
        click_rate_drift_test(detector_samples, num_blocks=num_blocks),
        polarization_drift_test(detector_samples, masks=masks,
                                num_blocks=min(10, num_blocks)),
        shot_autocorrelation_test(detector_samples, max_lag=max_lag,
                                  num_permutations=num_permutations, seed=seed),
    ])


# ===========================================================================
# Decoder-based tests: logical error rate, matching weight, gap
# ===========================================================================

# ---------------------------------------------------------------------------
# Shared decoder plumbing
# ---------------------------------------------------------------------------

def _as_detector_samples(detector_samples) -> np.ndarray:
    """Validate and canonicalize a (num_shots, num_detectors) binary array."""
    shots = np.asarray(detector_samples)
    if shots.ndim != 2:
        raise ValueError(
            "detector_samples must be 2D of shape (num_shots, num_detectors)."
        )
    return shots.astype(np.uint8) % 2


def _pad_dem(dem: stim.DetectorErrorModel, num_detectors: int) -> stim.DetectorErrorModel:
    """Pad a DEM with a trailing detector declaration to span num_detectors."""
    if num_detectors < dem.num_detectors:
        raise ValueError(
            f"detector_samples have {num_detectors} detectors but the DEM "
            f"declares {dem.num_detectors}."
        )
    if num_detectors == dem.num_detectors:
        return dem
    return dem + stim.DetectorErrorModel(f"detector D{num_detectors - 1}")


def _require_backend(decoder: str) -> None:
    """Raise a clear ImportError if the requested backend is missing."""
    if decoder == "pymatching":
        if _pymatching is None:
            raise ImportError(
                "pymatching is required for decoder='pymatching'; "
                "pip install pymatching."
            )
    elif decoder == "tesseract":
        if _tesseract is None:
            raise ImportError(
                "tesseract-decoder is required for decoder='tesseract'; "
                "pip install tesseract-decoder (no aarch64-linux wheels exist "
                "on PyPI as of 2026-08; a source build via bazel may be "
                "required)."
            )
    else:
        raise ValueError("decoder must be 'pymatching' or 'tesseract'.")


def decode_logical_predictions(dem: stim.DetectorErrorModel,
                               detector_samples,
                               decoder: str = "pymatching") -> np.ndarray:
    """
    Decode detector samples into predicted logical observable outcomes.

    Parameters:
        dem: stim.DetectorErrorModel
            Decorated DEM (at least one logical observable, i.e. events carry
            L targets).
        detector_samples: np.ndarray
            (num_shots, num_detectors) binary array in stim column order. May
            have more detectors than the DEM touches.
        decoder: str
            'pymatching' (minimum-weight perfect matching; graph-like DEMs)
            or 'tesseract' (most-likely-error; supports hyperedge events).

    Returns:
        predictions: np.ndarray
            (num_shots, dem.num_observables) uint8 array of predicted logical
            outcomes.
    """
    shots = _as_detector_samples(detector_samples)
    if dem.num_observables == 0:
        raise ValueError(
            "The DEM has no logical observables; a decorated DEM (events "
            "carrying L0 targets, see logical_decoration) is required."
        )
    _require_backend(decoder)
    padded = _pad_dem(dem, shots.shape[1])
    if decoder == "pymatching":
        matcher = _pymatching.Matching.from_detector_error_model(padded)
        preds = np.asarray(matcher.decode_batch(shots), dtype=np.uint8) % 2
    else:  # tesseract
        dec = _tesseract.TesseractConfig(dem=padded).compile_decoder()
        preds = np.asarray(dec.decode_batch(shots.astype(bool)), dtype=np.uint8) % 2
    if preds.ndim == 1:
        preds = preds[:, None]
    return preds


def _dem_events_with_logicals(dem: stim.DetectorErrorModel) -> list:
    """
    Extract ((detector_mask, observable_mask), probability) DEM events.

    Unlike `io.dem_to_dict`, the logical observable targets are kept: events
    are keyed by the pair (detector bitmask, observable bitmask). Events with
    identical keys are combined as independent flips
    (p <- p1 (1 - p2) + p2 (1 - p1)); events touching neither detectors nor
    observables are dropped. The result is sorted by key.
    """
    events: dict = {}
    for instruction in dem.flattened():
        if instruction.type != "error":
            continue
        p = instruction.args_copy()[0]
        det_mask = 0
        obs_mask = 0
        for t in instruction.targets_copy():
            if t.is_relative_detector_id():
                det_mask ^= 1 << t.val
            elif t.is_logical_observable_id():
                obs_mask ^= 1 << t.val
        key = (det_mask, obs_mask)
        if key == (0, 0):
            continue
        if key in events:
            q = events[key]
            events[key] = q * (1 - p) + p * (1 - q)
        else:
            events[key] = p
    return sorted(events.items())


# ---------------------------------------------------------------------------
# Logical-error-rate consistency test
# ---------------------------------------------------------------------------

def logical_error_rate_test(
    dem: stim.DetectorErrorModel,
    detector_samples,
    observed_logicals,
    *,
    decoder: str = "pymatching",
    predicted_ler=None,
    ler_estimator=None,
    num_mc_shots: int = 100000,
    seed=None,
) -> ValidationResult:
    """
    Test whether the decoded logical error rate matches the DEM's prediction.

    A decoder is built from the (decorated) candidate DEM and applied to the
    experimental detector data; the observed logical error rate LER_obs is
    the fraction of shots where the decoder's predicted logical outcome(s)
    disagree with the measured ones. LER_obs is then compared against the
    logical error rate the candidate DEM predicts for itself, obtained (in
    priority order) from:

      (a) `predicted_ler`: an explicit rate, treated as exactly known;
      (b) `ler_estimator`: a callable ``estimator(dem) -> float`` or
          ``-> (float, stderr)``. **This is the extension point for plugging
          in external logical-error-rate estimation methods** (analytic
          formulas, splitting/rare-event Monte Carlo, tensor-network
          estimators, other codebases, ...). A returned stderr is folded
          into the z-test's standard error; a bare float is treated as
          exactly known.
      (c) default: DEM Monte Carlo. `num_mc_shots` shots are sampled from
          the candidate DEM itself (`sample_dem`), decoded with the SAME
          decoder, and compared to the sampled observables, giving LER_mc
          with a binomial stderr.

    Statistical test (two-sided; the branch taken is reported in
    ``details['test_method']``):

      * Monte Carlo prediction: two-proportion pooled z-test of
        k_obs/S vs k_mc/num_mc_shots with a continuity correction of
        (1/S + 1/num_mc_shots)/2; when any expected 2x2 cell count under the
        pooled rate falls below 5 the z approximation is unreliable AND the
        MC point estimate is itself too noisy to condition on, so the test
        switches to Fisher's exact test on the 2x2 table (the exact
        conditional test for two binomials, which handles zero counts).
      * Known-constant prediction (cases (a), and (b) without stderr): a
        one-sample z-test against the rate with a 0.5-count continuity
        correction; when the expected number of failures or successes
        (min(S p, S (1-p))) falls below 10 it switches to the exact binomial
        test against the predicted rate.
      * Prediction with stderr (case (b) with stderr): one-sample z-test
        with the stderr added in quadrature to the binomial standard error;
        at small expected counts it falls back to the exact binomial test
        against the point estimate (the stderr is then ignored, noted in
        the details).

    Parameters:
        dem: stim.DetectorErrorModel
            Decorated candidate DEM (events carry L0 logical targets).
        detector_samples: np.ndarray
            (S, num_detectors) binary detector data in stim column order.
        observed_logicals: array-like
            (S,) or (S, num_observables) binary measured logical outcomes.
        decoder: str
            'pymatching' or 'tesseract' (see `decode_logical_predictions`).
        predicted_ler: float, optional
            Externally known predicted logical error rate, in [0, 1].
        ler_estimator: callable, optional
            ``estimator(dem) -> float`` or ``-> (float, stderr)``; see above.
        num_mc_shots: int
            Monte Carlo shots for the default prediction path.
        seed: int, optional
            Seed for the Monte Carlo sampling.

    Returns:
        result: ValidationResult
            ``statistic`` is the signed Wald z-score (observed - predicted in
            standard errors), ``effect_size`` the LER ratio
            observed/predicted, and ``details`` carries the rates, failure
            counts, the difference with a 95% Wald CI, the prediction source
            and the test branch taken.
    """
    shots = _as_detector_samples(detector_samples)
    num_shots = shots.shape[0]
    if num_shots == 0:
        raise ValueError("detector_samples must contain at least one shot.")
    if dem.num_observables == 0:
        raise ValueError(
            "logical_error_rate_test requires a decorated DEM (events carrying "
            "L0 targets, see logical_decoration.assign_logical_flags)."
        )
    observed = np.asarray(observed_logicals, dtype=np.uint8) % 2
    if observed.ndim == 1:
        observed = observed[:, None]
    if observed.ndim != 2 or observed.shape[0] != num_shots:
        raise ValueError(
            "observed_logicals must be (S,) or (S, num_observables) with one "
            "row per shot."
        )
    if observed.shape[1] != dem.num_observables:
        raise ValueError(
            f"observed_logicals has {observed.shape[1]} observables but the "
            f"DEM declares {dem.num_observables}."
        )

    predictions = decode_logical_predictions(dem, shots, decoder=decoder)
    k_obs = int(np.any(predictions != observed, axis=1).sum())
    p_obs = k_obs / num_shots

    # --- Predicted LER, in priority order --------------------------------
    n_pred = None      # finite MC sample size (None => treated as constant)
    k_pred = None
    pred_stderr = 0.0
    if predicted_ler is not None:
        p_pred = float(predicted_ler)
        if not 0.0 <= p_pred <= 1.0:
            raise ValueError("predicted_ler must lie in [0, 1].")
        source = "predicted_ler"
    elif ler_estimator is not None:
        estimate = ler_estimator(dem)
        if isinstance(estimate, (tuple, list)) or (
                isinstance(estimate, np.ndarray) and estimate.ndim > 0):
            p_pred = float(estimate[0])
            pred_stderr = float(estimate[1])
        else:
            p_pred = float(estimate)
        if not 0.0 <= p_pred <= 1.0:
            raise ValueError("ler_estimator returned a rate outside [0, 1].")
        source = "ler_estimator"
    else:
        det_mc, obs_mc = sample_dem(dem, num_mc_shots, seed=seed)
        preds_mc = decode_logical_predictions(dem, det_mc, decoder=decoder)
        if obs_mc.ndim == 1:
            obs_mc = obs_mc[:, None]
        k_pred = int(np.any(preds_mc != obs_mc, axis=1).sum())
        n_pred = num_mc_shots
        p_pred = k_pred / n_pred
        pred_stderr = math.sqrt(p_pred * (1 - p_pred) / n_pred)
        source = "dem_monte_carlo"

    # --- Effect sizes (guarded against zero rates) -----------------------
    diff = p_obs - p_pred
    if p_pred > 0:
        ratio = p_obs / p_pred
    else:
        ratio = 1.0 if k_obs == 0 else float("inf")
    # Wald standard error for the descriptive z and CI. The observed-variance
    # term is floored by the predicted-rate variance so that degenerate
    # observed rates (p_obs in {0, 1}, e.g. at very small S) do not collapse
    # the standard error and overstate the significance.
    var_obs = max(p_obs * (1 - p_obs), p_pred * (1 - p_pred)) / num_shots
    se_wald = math.sqrt(var_obs + pred_stderr ** 2)
    if se_wald > 0:
        z = diff / se_wald
    else:
        z = 0.0 if diff == 0 else math.copysign(float("inf"), diff)
    ci95 = (diff - 1.959963984540054 * se_wald, diff + 1.959963984540054 * se_wald)

    # --- p-value ----------------------------------------------------------
    stderr_ignored = False
    if n_pred is not None:
        # Two-binomial comparison against the Monte Carlo estimate.
        pooled = (k_obs + k_pred) / (num_shots + n_pred)
        expected_cells = (
            num_shots * pooled, num_shots * (1 - pooled),
            n_pred * pooled, n_pred * (1 - pooled),
        )
        if min(expected_cells) < 5.0:
            test_method = "fisher_exact"
            _, pvalue = scipy.stats.fisher_exact(
                [[k_obs, num_shots - k_obs], [k_pred, n_pred - k_pred]],
                alternative="two-sided",
            )
        else:
            test_method = "two_proportion_z"
            se_pooled = math.sqrt(
                pooled * (1 - pooled) * (1.0 / num_shots + 1.0 / n_pred)
            )
            continuity = 0.5 * (1.0 / num_shots + 1.0 / n_pred)
            z_test = max(abs(diff) - continuity, 0.0) / se_pooled
            pvalue = 2.0 * scipy.stats.norm.sf(z_test)
    elif p_pred <= 0.0 or p_pred >= 1.0:
        # Degenerate constant null: the model claims certainty.
        test_method = "degenerate_constant"
        consistent = (k_obs == 0) if p_pred <= 0.0 else (k_obs == num_shots)
        pvalue = 1.0 if consistent else 0.0
    elif min(num_shots * p_pred, num_shots * (1 - p_pred)) < 10.0:
        # Small expected counts: exact binomial test against the rate.
        test_method = "binomial_exact"
        stderr_ignored = pred_stderr > 0
        pvalue = float(scipy.stats.binomtest(k_obs, num_shots, p_pred).pvalue)
    else:
        # One-sample z against a (possibly noisy) constant rate, with a
        # 0.5-count continuity correction.
        test_method = "one_sample_z"
        se_null = math.sqrt(
            p_pred * (1 - p_pred) / num_shots + pred_stderr ** 2
        )
        z_test = max(abs(diff) - 0.5 / num_shots, 0.0) / se_null
        pvalue = 2.0 * scipy.stats.norm.sf(z_test)
    pvalue = float(min(max(pvalue, 0.0), 1.0))

    ratio_str = f"{ratio:.3g}" if np.isfinite(ratio) else "inf"
    description = (
        f"observed LER {p_obs:.3g} vs predicted {p_pred:.3g} "
        f"(ratio {ratio_str}, {z:+.1f} sigma)"
    )
    details = {
        "ler_observed": p_obs,
        "ler_predicted": p_pred,
        "predicted_stderr": pred_stderr,
        "predicted_source": source,
        "observed_failures": k_obs,
        "predicted_failures": k_pred,
        "num_predicted_shots": n_pred,
        "difference": diff,
        "difference_ci95": ci95,
        "z": z,
        "test_method": test_method,
        "stderr_ignored_by_exact_test": stderr_ignored,
        "decoder": decoder,
        "num_observables": dem.num_observables,
    }
    return ValidationResult(
        name=f"logical_error_rate[{decoder}]",
        statistic=float(z),
        pvalue=pvalue,
        effect_size=float(ratio),
        effect_description=description,
        num_shots=num_shots,
        null_model="dem",
        details=details,
    )


# ---------------------------------------------------------------------------
# Scalar functions of the syndrome (for scalar_distribution_test)
# ---------------------------------------------------------------------------

def matching_weight_function(dem: stim.DetectorErrorModel,
                             decoder: str = "pymatching",
                             min_probability: float = 1e-12):
    """
    Per-shot weight of the decoder's best correction, as a syndrome scalar.

    The returned callable maps (S, num_detectors) binary detector samples to
    an (S,) float array of decoder costs: the summed log((1-p)/p) weight of
    the minimum-weight correction (pymatching) or of the most-likely error
    set (tesseract). The all-zero syndrome has weight 0; a syndrome explained
    by a single DEM event has that event's log-weight. Intended for use with
    `scalar_distribution_test`, which calibrates the distribution of this
    scalar by DEM Monte Carlo.

    Parameters:
        dem: stim.DetectorErrorModel
            Candidate DEM; logical decorations (L targets) are ignored.
            The pymatching backend requires the DEM to be graph-like.
        decoder: str
            'pymatching' (batched, fast; use for large shot counts) or
            'tesseract' (per-shot most-likely-error cost via
            `cost_from_errors`; supports hyperedge events but decodes shots
            one at a time in Python, so it is slower).
        min_probability: float
            Probabilities are clipped to [min_probability, 0.5 - min_probability]
            before conversion to weights.

    Returns:
        func: callable
            ``func(detector_samples) -> (S,) float`` array of weights.
    """
    _require_backend(decoder)
    cache: dict = {}

    if decoder == "pymatching":
        def func(detector_samples):
            shots = _as_detector_samples(detector_samples)
            n_det = shots.shape[1]
            matcher = cache.get(n_det)
            if matcher is None:
                matcher, _ = build_matcher(
                    dem, num_detectors=n_det, min_probability=min_probability
                )
                cache[n_det] = matcher
            _, weights = matcher.decode_batch(shots, return_weights=True)
            return np.asarray(weights, dtype=float)
        return func

    def func(detector_samples):
        shots = _as_detector_samples(detector_samples)
        n_det = shots.shape[1]
        dec = cache.get(n_det)
        if dec is None:
            _, probs, masks = dem_to_check_matrix(dem, num_detectors=n_det)
            decode_dem = _dem_for_decoding(masks, probs, n_det, min_probability)
            dec = _tesseract.TesseractConfig(dem=decode_dem).compile_decoder()
            cache[n_det] = dec
        bool_shots = shots.astype(bool)
        weights = np.zeros(shots.shape[0], dtype=float)
        for i in range(shots.shape[0]):
            dec.decode_to_errors(bool_shots[i])
            predicted = list(dec.predicted_errors_buffer)
            if predicted:
                weights[i] = float(dec.cost_from_errors(predicted))
        return weights
    return func


def complementary_gap_function(dem: stim.DetectorErrorModel,
                               decoder: str = "pymatching",
                               sign: bool = False,
                               min_probability: float = 1e-12):
    """
    Per-shot complementary gap between the two logical classes.

    For each syndrome the decoder finds the best correction with the logical
    outcome forced to 0 (weight w0) and forced to 1 (weight w1); the
    complementary gap ``|w1 - w0|`` is the extra weight of the best
    correction in the logical class OPPOSITE to the decoder's choice — a
    continuous decoder-confidence signal (large gap = confident decode, gap
    near 0 = coin toss). A trivial (all-zero) syndrome at low error rates
    has a large gap: flipping the logical there costs an entire logical
    operator's worth of weight. Intended for `scalar_distribution_test`.

    Implementation: the check matrix is augmented with the logical-membership
    row as one extra "detector" row, a single decoder is built over the
    augmented matrix, and each syndrome is decoded twice with the appended
    logical bit forced to 0 and to 1. The pymatching backend requires each
    augmented column to touch at most two rows, i.e. any event that flips a
    logical must flip at most one detector (typical for decorated boundary
    events); otherwise a ValueError suggests the tesseract backend, which
    decodes the augmented hyperedges natively (costs via
    `cost_from_errors`, one shot at a time).

    Parameters:
        dem: stim.DetectorErrorModel
            Decorated DEM with exactly one logical observable (L0).
        decoder: str
            'pymatching' (batched) or 'tesseract' (per-shot, hyperedge-safe).
        sign: bool
            If False (default) return the unsigned gap ``|w1 - w0| >= 0``.
            If True return the signed gap ``w1 - w0``: its magnitude is the
            unsigned gap and its sign encodes the decoder's chosen logical
            class (positive = class 0 is cheaper/chosen, negative = class 1).
        min_probability: float
            Probability clipping for the weight conversion.

    Returns:
        func: callable
            ``func(detector_samples) -> (S,) float`` array of gaps.
    """
    _require_backend(decoder)
    if dem.num_observables != 1:
        raise ValueError(
            f"complementary_gap_function requires exactly one logical "
            f"observable (got {dem.num_observables}); decorate the DEM with "
            "L0 flags first (see logical_decoration)."
        )
    events = _dem_events_with_logicals(dem)
    if not any(obs_mask for (_, obs_mask), _ in events):
        raise ValueError(
            "No DEM event flips the logical observable; the opposite logical "
            "class is unreachable and the complementary gap is undefined."
        )
    cache: dict = {}

    if decoder == "pymatching":
        # Each augmented column touches popcount(det_mask) + (obs_mask != 0)
        # rows; pymatching needs at most two. Validate eagerly.
        bad = [key for key, _ in events
               if key[0].bit_count() + (1 if key[1] else 0) > 2]
        if bad:
            raise ValueError(
                f"DEM is not graph-like after logical augmentation: events "
                f"{bad} (detector_mask, observable_mask) touch more than two "
                "rows of the augmented check matrix. Use decoder='tesseract', "
                "which handles hyperedges."
            )

        def _augmented_matcher(n_det):
            n_events = len(events)
            H = np.zeros((n_det + 1, n_events), dtype=np.uint8)
            probs = np.zeros(n_events, dtype=float)
            for j, ((det_mask, obs_mask), p) in enumerate(events):
                if det_mask >> n_det:
                    raise ValueError(
                        "DEM events touch detectors beyond the sample width."
                    )
                for d in range(n_det):
                    H[d, j] = (det_mask >> d) & 1
                if obs_mask:
                    H[n_det, j] = 1
                probs[j] = p
            p = np.clip(probs, min_probability, 0.5 - min_probability)
            return _pymatching.Matching(
                scipy.sparse.csc_matrix(H),
                weights=np.log((1.0 - p) / p),
            )

        def func(detector_samples):
            shots = _as_detector_samples(detector_samples)
            n_det = shots.shape[1]
            matcher = cache.get(n_det)
            if matcher is None:
                matcher = _augmented_matcher(n_det)
                cache[n_det] = matcher
            logical_col = np.empty((shots.shape[0], 1), dtype=np.uint8)
            logical_col.fill(0)
            _, w0 = matcher.decode_batch(
                np.concatenate([shots, logical_col], axis=1), return_weights=True
            )
            logical_col.fill(1)
            _, w1 = matcher.decode_batch(
                np.concatenate([shots, logical_col], axis=1), return_weights=True
            )
            gap = np.asarray(w1, dtype=float) - np.asarray(w0, dtype=float)
            return gap if sign else np.abs(gap)
        return func

    def _augmented_tesseract(n_det):
        lines = []
        for (det_mask, obs_mask), p in events:
            if det_mask >> n_det:
                raise ValueError(
                    "DEM events touch detectors beyond the sample width."
                )
            p_clipped = min(max(p, min_probability), 0.5 - min_probability)
            targets = [f"D{d}" for d in range(n_det) if (det_mask >> d) & 1]
            if obs_mask:
                targets.append(f"D{n_det}")  # logical row as an extra detector
            lines.append(f"error({p_clipped}) {' '.join(targets)}")
        lines.append(f"detector D{n_det}")
        aug_dem = dem_from_str("\n".join(lines) + "\n")
        return _tesseract.TesseractConfig(dem=aug_dem).compile_decoder()

    def func(detector_samples):
        shots = _as_detector_samples(detector_samples)
        n_det = shots.shape[1]
        dec = cache.get(n_det)
        if dec is None:
            dec = _augmented_tesseract(n_det)
            cache[n_det] = dec
        gap = np.zeros(shots.shape[0], dtype=float)
        augmented = np.zeros(n_det + 1, dtype=bool)
        for i in range(shots.shape[0]):
            augmented[:n_det] = shots[i]
            costs = []
            for forced_bit in (False, True):
                augmented[n_det] = forced_bit
                dec.decode_to_errors(augmented)
                predicted = list(dec.predicted_errors_buffer)
                costs.append(float(dec.cost_from_errors(predicted))
                             if predicted else 0.0)
            gap[i] = costs[1] - costs[0]
        return gap if sign else np.abs(gap)
    return func


# ---------------------------------------------------------------------------
# Convenience wrappers around scalar_distribution_test
# ---------------------------------------------------------------------------

def _scalar_distribution_test():
    """Late-bound import of the scalar-distribution engine."""
    try:
        from .validation import scalar_distribution_test
    except ImportError as exc:  # pragma: no cover - only pre-merge
        raise ImportError(
            "scalar_distribution_test is not available in "
            "sparsedem.validation; the scalar-distribution engine is required "
            "for the matching-weight and complementary-gap tests."
        ) from exc
    return scalar_distribution_test


def matching_weight_test(dem: stim.DetectorErrorModel, detector_samples, *,
                         decoder: str = "pymatching", name=None,
                         **kwargs) -> ValidationResult:
    """
    Distribution test on the per-shot decoder matching weight.

    Convenience wrapper: builds `matching_weight_function(dem, decoder)` and
    runs `scalar_distribution_test` on it, comparing the observed
    distribution of decoder costs against the DEM-Monte-Carlo null. Extra
    keyword arguments (num_null_shots, method, num_bins, seed, ...) are
    forwarded to `scalar_distribution_test`.
    """
    func = matching_weight_function(dem, decoder=decoder)
    if name is None:
        name = f"matching_weight[{decoder}]"
    return _scalar_distribution_test()(
        dem, detector_samples, func, name=name, **kwargs
    )


def complementary_gap_test(dem: stim.DetectorErrorModel, detector_samples, *,
                           decoder: str = "pymatching", sign: bool = False,
                           name=None, **kwargs) -> ValidationResult:
    """
    Distribution test on the per-shot complementary gap.

    Convenience wrapper: builds `complementary_gap_function(dem, decoder,
    sign)` and runs `scalar_distribution_test` on it, comparing the observed
    gap distribution (a decoder-confidence signal) against the
    DEM-Monte-Carlo null. Extra keyword arguments are forwarded to
    `scalar_distribution_test`.
    """
    func = complementary_gap_function(dem, decoder=decoder, sign=sign)
    if name is None:
        name = f"complementary_gap[{decoder}]"
    return _scalar_distribution_test()(
        dem, detector_samples, func, name=name, **kwargs
    )

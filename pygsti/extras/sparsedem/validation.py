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

from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional, Sequence, Union

import numpy as np
import scipy.stats

import stim

from .io import dem_to_dict

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
    outcomes = np.arange(2 ** k, dtype=np.uint64)
    attenuations = np.zeros(2 ** k, dtype=float)
    for local_mask, prob in projected.items():
        parity = (np.bitwise_count(outcomes & np.uint64(local_mask)) & 1).astype(float)
        attenuations += -np.log1p(-2 * min(prob, 0.5 - 1e-15)) * parity
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

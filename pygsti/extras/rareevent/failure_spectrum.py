"""
Failure-spectrum ansatz estimator for QEC logical failure rates.

Implements the "failure spectrum" technique from:

    Beverland, Carroll, Cross, Yoder,
    "Fail fast: techniques to probe rare events in quantum error correction",
    arXiv:2511.15177 (2025).

Method
------
The failure spectrum f(w) is the failure fraction of weight-w fault sets: the
probability that a set of exactly w simultaneously active error mechanisms
causes a logical failure. The logical failure rate at physical rate p is then
an exact decomposition over fault weights,

    P_fail(p) = sum_w  f(w) * Pr[W = w; q(p)],

where W is the number of active mechanisms. In the paper all fault locations
share a uniform rate q (via an "expanded representation" that replicates
columns in proportion to their probabilities), so Pr[W = w] is binomial and
f(w) is the fraction of the (N choose w) weight-w bitstrings that fail.

This implementation generalizes that to the heterogeneous per-mechanism
probabilities q_i(p) of a `MechanismCatalog`:

- Pr[W = w; q(p)] is the Poisson-binomial weight distribution.
- f(w) is defined as P(fail | W = w) under the reference distribution at
  p_ref, and is measured by sampling weight-w mechanism sets from the exact
  conditional distribution P(E | |E| = w). Sampling uses exponential tilting
  plus rejection: tilting each Bernoulli leaves the conditional distribution
  given W = w unchanged, so rejecting on the exact weight is exact.

The decomposition is exact at p_ref. Applying the same f(w) at other p assumes
the conditional distribution of *which* mechanisms are active given the weight
is independent of p, which holds whenever the relative mechanism probabilities
are p-independent (exact for uniform rates and for linear scaling such as
`ScaledMechanismErrorModel`, and a good approximation for first-order noise
models generally). This mirrors the fixed-ratio assumption of the paper's
expanded representation.

The spectrum is only sampled at weights where failures are observable; a
low-parameter ansatz (Eq. 10 of the paper) is fitted to the measurements and
extrapolated down to the onset weight w0 (the minimum failing weight, e.g.
ceil(d/2) for a distance-d code with a min-weight decoder):

    f_ansatz^(2)(w) = a * (1 - exp(-(f0/a) * (w/w0)^w0))
    f_ansatz^(3)(w) = a * (1 - exp(-(f0/a) * (w/w0)^gamma1))
    f_ansatz^(5)(w) = a * (1 - exp(-(f0/a) * (w/w0)^gamma1
                          * ((1 + (w/wc)^c) / (1 + (w0/wc)^c))^((gamma2-gamma1)/c)))

with f(w) = 0 for w < w0, fixed c = 2, and high-weight asymptote
a = 1 - 2^-K for K logical observables (the failure rate of a random
high-weight fault set).
"""

from __future__ import annotations

import dataclasses
import math
import sys
from collections.abc import Sequence
from typing import Any

import numpy as np
from scipy.optimize import brentq, least_squares

from .interfaces import ErrorModel, Estimator, ForwardSimulator
from .weight_points import WeightPoint

ANSATZ_FORMS = ("2", "3", "5")


# ---------------------------------------------------------------------------
# Weight distribution: Poisson-binomial pmf of the number of active mechanisms
# ---------------------------------------------------------------------------


def poisson_binomial_pmf(probs: np.ndarray, max_weight: int | None = None) -> np.ndarray:
    """Exact pmf of W = sum of independent Bernoulli(q_i), truncated at max_weight.

    Returns an array of length min(N, max_weight) + 1 where entry w is
    Pr[W = w]. When truncated, mass above max_weight is dropped (the returned
    array sums to Pr[W <= max_weight]).
    """
    q = np.asarray(probs, dtype=np.float64)
    if np.any(q < 0) or np.any(q >= 1):
        raise ValueError("All mechanism probabilities must be in [0, 1).")
    n = len(q)
    cap = n if max_weight is None else min(int(max_weight), n)
    if cap < 0:
        raise ValueError(f"max_weight must be nonnegative; got {max_weight}.")

    pmf = np.zeros(cap + 1, dtype=np.float64)
    pmf[0] = 1.0
    support = 0
    for qi in q:
        if qi == 0.0:
            continue
        upper = min(support + 1, cap)
        new = pmf * (1.0 - qi)
        new[1 : upper + 1] += qi * pmf[:upper]
        pmf = new
        support = upper
    return pmf


# ---------------------------------------------------------------------------
# Exact conditional fixed-weight sampling via exponential tilting + rejection
# ---------------------------------------------------------------------------


def tilted_probabilities(probs: np.ndarray, weight: int) -> np.ndarray:
    """Exponentially tilt Bernoulli probabilities so the mean weight equals `weight`.

    Tilting maps q_i -> q_i*t / (1 - q_i + q_i*t) for a common t > 0. It does
    not change the conditional distribution of the active set given its total
    weight, so rejection sampling on the exact weight under the tilted
    distribution samples P(E | |E| = weight) under the original distribution.
    """
    q = np.asarray(probs, dtype=np.float64)
    nonzero = int(np.count_nonzero(q))
    if weight < 0 or weight > nonzero:
        raise ValueError(f"weight must be in [0, {nonzero}] (number of nonzero-probability mechanisms); got {weight}.")
    if weight == 0:
        return np.zeros_like(q)
    if weight == nonzero:
        return (q > 0).astype(np.float64)

    def mean_weight(log_t: float) -> float:
        t = math.exp(log_t)
        return float(np.sum(q * t / (1.0 - q + q * t)))

    lo, hi = -60.0, 60.0
    log_t = float(brentq(lambda s: mean_weight(s) - weight, lo, hi, xtol=1e-12))
    t = math.exp(log_t)
    return np.asarray(q * t / (1.0 - q + q * t), dtype=np.float64)


def sample_fixed_weight_failure_fraction(
    simulator: ForwardSimulator,
    probs: np.ndarray,
    weight: int,
    rng: np.random.Generator,
    *,
    target_failures: int,
    max_trials: int,
    batch_size: int = 4096,
) -> tuple[int, int]:
    """Estimate f(weight) = P(fail | W = weight) by conditional sampling.

    Draws weight-`weight` mechanism sets from the exact conditional
    distribution at the given probabilities and evaluates the simulator on
    each. Stops when `target_failures` failures have been observed or
    `max_trials` sets have been evaluated. Returns (trials, failures).
    """
    if target_failures < 1:
        raise ValueError(f"target_failures must be at least 1; got {target_failures}.")
    if max_trials < 1:
        raise ValueError(f"max_trials must be at least 1; got {max_trials}.")
    q_t = tilted_probabilities(probs, weight)
    n = len(q_t)
    trials = 0
    failures = 0
    while trials < max_trials and failures < target_failures:
        draws = rng.random((batch_size, n)) < q_t
        counts = draws.sum(axis=1)
        for row in np.flatnonzero(counts == weight):
            active = set(np.flatnonzero(draws[row]).tolist())
            if simulator.fails(active):
                failures += 1
            trials += 1
            if trials >= max_trials or failures >= target_failures:
                break
    return trials, failures


# ---------------------------------------------------------------------------
# Ansatz forms (Eq. 10 of arXiv:2511.15177)
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class FittedFailureSpectrum:
    """A fitted failure-spectrum ansatz f(w), callable on (arrays of) weights.

    Attributes:
        ansatz: Which form was used: "2", "3", or "5" (parameter count).
        a: High-weight asymptote, a = 1 - 2^-K for K logical observables.
        w0: Onset weight; f(w) = 0 for w < w0.
        f0: Failure fraction scale at the onset weight, f(w0) ~= f0.
        gamma1: Low-weight exponent (equals w0 for the 2-parameter form).
        gamma2: High-weight exponent (5-parameter form only).
        wc: Crossover weight between the two exponents (5-parameter form only).
        c: Crossover smoothness (fixed to 2 in the paper).
    """

    ansatz: str
    a: float
    w0: float
    f0: float
    gamma1: float
    gamma2: float | None = None
    wc: float | None = None
    c: float = 2.0

    def __call__(self, w: np.ndarray | float | int) -> np.ndarray:
        warr = np.atleast_1d(np.asarray(w, dtype=np.float64))
        out = np.zeros_like(warr)
        mask = warr >= self.w0
        if not np.any(mask):
            return out
        wm = warr[mask]
        with np.errstate(over="ignore"):
            arg = self.f0 * (wm / self.w0) ** self.gamma1
            if self.ansatz == "5":
                assert self.gamma2 is not None and self.wc is not None
                crossover = (1.0 + (wm / self.wc) ** self.c) / (1.0 + (self.w0 / self.wc) ** self.c)
                arg = arg * crossover ** ((self.gamma2 - self.gamma1) / self.c)
            out[mask] = self.a * (-np.expm1(-arg / self.a))
        return out


def fit_failure_spectrum(
    weights: Sequence[int],
    trials: Sequence[int],
    failures: Sequence[int],
    *,
    a: float,
    ansatz: str = "3",
    w0: float | None = None,
    aux_weights: Sequence[float] | None = None,
    aux_fractions: Sequence[float] | None = None,
    aux_sigma_log: Sequence[float] | None = None,
) -> tuple[FittedFailureSpectrum, dict[str, Any]]:
    """Fit a failure-spectrum ansatz to sampled failure fractions.

    Only weights with at least one observed failure are used. The fit
    minimizes log-space residuals weighted by the binomial standard error of
    each measured fraction (per the paper's fitting guidelines). If w0 is
    None, the onset weight is included as a free (continuous) fit parameter,
    bounded above by the smallest weight with an observed failure.

    Auxiliary spectrum points (aux_weights / aux_fractions / aux_sigma_log)
    are f(w) measurements that do not come from binomial counting — e.g.
    fixed-weight gap-splitting estimates (``gap_splitting.py``) far below the
    rejection-sampling floor. Each enters the fit as one more log-space
    residual with the supplied log-standard-error, and participates in the
    onset bound exactly like a sampled point: an auxiliary point at a low
    weight lowers the upper bound on a fitted w0.

    Returns the fitted spectrum and a fit report dictionary.
    """
    if ansatz not in ANSATZ_FORMS:
        raise ValueError(f"ansatz must be one of {ANSATZ_FORMS}; got {ansatz!r}.")
    if not (0 < a <= 1):
        raise ValueError(f"Asymptote a must be in (0, 1]; got {a}.")

    w_arr = np.asarray(weights, dtype=np.float64)
    t_arr = np.asarray(trials, dtype=np.float64)
    f_arr = np.asarray(failures, dtype=np.float64)
    usable = f_arr > 0
    wu, tu, fu = w_arr[usable], t_arr[usable], f_arr[usable]

    f_hat = fu / tu
    # Adjusted (Agresti-Coull-style) fraction keeps the standard error nonzero
    # even when all or no trials failed at some weight.
    f_adj = (fu + 1.0) / (tu + 2.0)
    sigma = np.sqrt(f_adj * (1.0 - f_adj) / tu)
    sigma_log = sigma / f_hat

    aux_w = np.asarray(list(aux_weights) if aux_weights is not None else [], dtype=np.float64)
    aux_f = np.asarray(list(aux_fractions) if aux_fractions is not None else [], dtype=np.float64)
    aux_s = np.asarray(list(aux_sigma_log) if aux_sigma_log is not None else [], dtype=np.float64)
    if not (len(aux_w) == len(aux_f) == len(aux_s)):
        raise ValueError(
            f"aux_weights, aux_fractions, aux_sigma_log must have equal lengths; "
            f"got {len(aux_w)}, {len(aux_f)}, {len(aux_s)}."
        )
    if len(aux_f) and (np.any(~np.isfinite(aux_f)) or np.any(aux_f <= 0)):
        raise ValueError("aux_fractions must be finite and positive.")
    if len(aux_s) and (np.any(~np.isfinite(aux_s)) or np.any(aux_s <= 0)):
        raise ValueError("aux_sigma_log must be finite and positive.")
    num_aux = len(aux_w)
    wu = np.concatenate([wu, aux_w])
    f_hat = np.concatenate([f_hat, aux_f])
    sigma_log = np.concatenate([sigma_log, aux_s])
    num_points = len(wu)

    min_fail_weight = float(np.min(wu)) if len(wu) else float("nan")
    fit_w0 = w0 is None
    if not fit_w0:
        assert w0 is not None
        if len(wu) and w0 > min_fail_weight:
            raise ValueError(
                f"Onset weight w0={w0} exceeds the smallest weight with an observed failure ({min_fail_weight})."
            )

    num_free = {"2": 1, "3": 2, "5": 4}[ansatz] + int(fit_w0)
    if num_points < num_free:
        raise ValueError(
            f"Fitting the {ansatz}-parameter ansatz needs at least {num_free} measured points "
            f"(weights with observed failures plus auxiliary points); got {num_points}. "
            "Sample more weights, increase the trial budget, or supply auxiliary points."
        )

    f0_init = float(np.clip(f_hat[np.argmin(wu)], 1e-12, min(a, 0.999)))
    w0_init = min_fail_weight if fit_w0 else float(w0)  # type: ignore[arg-type]
    gamma_init = max(w0_init, 1.0)
    wc_init = float(np.sqrt(np.min(wu) * np.max(wu)))

    # All parameters are fitted in log space to enforce positivity.
    lo = [math.log(1e-15)]
    hi = [0.0]
    if ansatz in ("3", "5"):
        lo.append(math.log(0.05))
        hi.append(math.log(500.0))
    if ansatz == "5":
        lo.append(math.log(0.05))
        hi.append(math.log(500.0))
        lo.append(math.log(0.5))
        hi.append(math.log(10.0 * float(np.max(wu))))
    if fit_w0:
        lo.append(math.log(0.5))
        hi.append(math.log(min_fail_weight))
    lo_arr = np.asarray(lo)
    hi_arr = np.asarray(hi)

    def make_x0(f0_i: float, gamma_i: float) -> np.ndarray:
        x0 = [math.log(f0_i)]
        if ansatz in ("3", "5"):
            x0.append(math.log(gamma_i))
        if ansatz == "5":
            x0.append(math.log(max(gamma_i / 2.0, 0.1)))
            x0.append(math.log(wc_init))
        if fit_w0:
            x0.append(math.log(w0_init))
        return np.asarray(np.clip(np.asarray(x0), lo_arr, hi_arr), dtype=np.float64)

    # A poor initial guess can strand the optimizer on the saturated plateau
    # f ~= a, where the residual gradients vanish (all measured weights map to
    # the asymptote and no parameter moves them off it). Add a second start
    # from a weighted log-log regression of the sub-saturation points, where
    # f(w) ~= f0 * (w/w0)^gamma1, and keep whichever start fits better.
    starts = [make_x0(f0_init, gamma_init)]
    sub = f_hat < 0.5 * a
    if int(np.sum(sub)) >= 2 and len(np.unique(wu[sub])) >= 2:
        log_w = np.log(wu[sub] / w0_init)
        log_f = np.log(f_hat[sub])
        reg_weights = np.sqrt(1.0 / np.square(sigma_log[sub]))
        if ansatz == "2":
            gamma_reg = w0_init
            intercept = float(np.average(log_f - gamma_reg * log_w, weights=np.square(reg_weights)))
        else:
            coeffs = np.polyfit(log_w, log_f, 1, w=reg_weights)
            gamma_reg = float(np.clip(coeffs[0], 0.05, 500.0))
            intercept = float(coeffs[1])
        f0_reg = float(np.clip(math.exp(intercept), 1e-15, min(a, 0.999)))
        starts.append(make_x0(f0_reg, gamma_reg))

    def build(theta: np.ndarray) -> FittedFailureSpectrum:
        vals = [math.exp(v) for v in theta]
        f0 = vals[0]
        idx = 1
        gamma1 = gamma2 = wc = None
        if ansatz in ("3", "5"):
            gamma1 = vals[idx]
            idx += 1
        if ansatz == "5":
            gamma2 = vals[idx]
            wc = vals[idx + 1]
            idx += 2
        w0_val = vals[idx] if fit_w0 else float(w0)  # type: ignore[arg-type]
        if ansatz == "2":
            gamma1 = w0_val
        assert gamma1 is not None
        return FittedFailureSpectrum(ansatz=ansatz, a=a, w0=w0_val, f0=f0, gamma1=gamma1, gamma2=gamma2, wc=wc)

    def residuals(theta: np.ndarray) -> np.ndarray:
        model = np.clip(build(theta)(wu), 1e-300, None)
        return np.asarray((np.log(model) - np.log(f_hat)) / sigma_log, dtype=np.float64)

    result = None
    for x0_arr in starts:
        candidate = least_squares(residuals, x0_arr, bounds=(lo_arr, hi_arr))
        if result is None or candidate.cost < result.cost:
            result = candidate
    assert result is not None
    spectrum = build(np.asarray(result.x))

    report: dict[str, Any] = {
        "success": bool(result.success),
        "message": str(result.message),
        "cost": float(result.cost),
        "num_points": num_points,
        "num_aux_points": num_aux,
        "num_free_parameters": num_free,
        "num_starts": len(starts),
        "fitted_w0": fit_w0,
        "chi2_per_point": float(2.0 * result.cost / max(num_points, 1)),
    }
    return spectrum, report


# ---------------------------------------------------------------------------
# Transform: P_fail(p) = sum_w f(w) Pr[W = w; q(p)]   (Eq. 1 of the paper)
# ---------------------------------------------------------------------------


def transform_spectrum_to_failure_rate(
    spectrum: FittedFailureSpectrum,
    probs: np.ndarray,
    max_weight: int | None = None,
) -> float:
    """Combine a failure spectrum with the Poisson-binomial weight distribution.

    If max_weight is None a cutoff is chosen automatically far into the tail
    of the weight distribution (truncation error is bounded by the tail mass,
    which is negligible at the chosen cutoff).
    """
    q = np.asarray(probs, dtype=np.float64)
    if max_weight is None:
        mu = float(np.sum(q))
        var = float(np.sum(q * (1.0 - q)))
        max_weight = min(len(q), int(math.ceil(mu + 12.0 * math.sqrt(var) + 20.0)))
    pmf = poisson_binomial_pmf(q, max_weight=max_weight)
    w = np.arange(len(pmf), dtype=np.float64)
    return float(np.sum(spectrum(w) * pmf))


# ---------------------------------------------------------------------------
# Estimator
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class SpectrumSample:
    """One measured point of the failure spectrum."""

    weight: int
    trials: int
    failures: int

    @property
    def failure_fraction(self) -> float:
        return self.failures / self.trials if self.trials else float("nan")

    @property
    def stderr(self) -> float:
        if not self.trials:
            return float("nan")
        f = self.failure_fraction
        return math.sqrt(f * (1.0 - f) / self.trials)


@dataclasses.dataclass
class FailureSpectrumResult:
    p_scales: list[float]
    failure_estimates: list[float]
    log_failure_estimates: list[float]
    p_ref: float
    samples: list[SpectrumSample]
    spectrum: FittedFailureSpectrum
    fit_report: dict[str, Any]
    aux_points: list[WeightPoint] = dataclasses.field(default_factory=list)


def logspaced_integer_weights(w_lo: int, w_hi: int, num: int) -> list[int]:
    """Approximately log-uniformly spaced distinct integers in [w_lo, w_hi]."""
    if w_lo < 1 or w_hi < w_lo:
        raise ValueError(f"Need 1 <= w_lo <= w_hi; got w_lo={w_lo}, w_hi={w_hi}.")
    grid = np.geomspace(w_lo, w_hi, num=max(num, 1))
    return sorted({int(round(x)) for x in grid})


def failure_spectrum_estimate(
    error_model: ErrorModel,
    simulator: ForwardSimulator,
    p_scales: Sequence[float],
    *,
    p_ref: float | None = None,
    weights: Sequence[int] | None = None,
    num_weights: int = 12,
    w_lo: int | None = None,
    w_hi: int | None = None,
    target_failures: int = 100,
    max_trials_per_weight: int = 20_000,
    batch_size: int = 4096,
    ansatz: str = "3",
    w0: float | None = None,
    a: float | None = None,
    num_observables: int | None = None,
    transform_max_weight: int | None = None,
    aux_points: Sequence[WeightPoint] | None = None,
    min_aux_sigma_log: float = 0.05,
    seed: int = 1,
    verbose: bool = True,
) -> FailureSpectrumResult:
    """Estimate failure rates at each p in p_scales via the failure-spectrum ansatz.

    Args:
        error_model: Provides mechanism probabilities q_i(p).
        simulator: Decides logical failure for an active mechanism set.
        p_scales: Physical error rates to predict (any order; extrapolation to
            rates far above p_ref stresses the fixed-ratio assumption).
        p_ref: Reference rate defining the sampling distribution; defaults to
            max(p_scales).
        weights: Explicit fault weights at which to sample the spectrum. If
            None, ~num_weights log-spaced integers in [w_lo, w_hi] are used,
            where w_lo defaults to w0 (or 1) and w_hi defaults to a few
            standard deviations above the mean fault weight at p_ref.
        target_failures: Stop sampling a weight once this many failures are
            seen (yields roughly uniform failure counts across weights, per
            the paper's fitting guidelines).
        max_trials_per_weight: Hard cap on evaluated sets per weight.
        ansatz: "2", "3", or "5" — which ansatz form to fit.
        w0: Onset weight (minimum failing weight), e.g. ceil(d/2) for a
            distance-d code under min-weight decoding. If None it is fitted.
        a: High-weight asymptote. Defaults to 1 - 2^-K where K is
            num_observables (taken from error_model.catalog if present).
        transform_max_weight: Truncation of the weight distribution in the
            final transform; None picks a safe automatic cutoff.
        aux_points: Auxiliary f(w) measurements to include in the fit, as
            `WeightPoint`s with kind "f_w" (e.g. from
            `gap_splitting.estimate_f_w_gap_splitting`, which resolves f(w)
            far below this function's rejection-sampling floor). Each point's
            `rel_err` is used as its log-space standard error. Points with a
            nonpositive estimate or non-finite rel_err are skipped (reported
            when verbose). The measurements must refer to the same reference
            conditional distribution P(E | |E| = w) as this fit — exact
            whenever relative mechanism probabilities are p-independent (the
            same fixed-ratio assumption the ansatz transform already makes).
        min_aux_sigma_log: Floor applied to each auxiliary point's log-space
            standard error, so an over-confident auxiliary point cannot
            dominate the fit.
    """
    if not p_scales:
        raise ValueError("p_scales must be a non-empty sequence of physical error rates.")
    if ansatz not in ANSATZ_FORMS:
        raise ValueError(f"ansatz must be one of {ANSATZ_FORMS}; got {ansatz!r}.")

    if p_ref is None:
        p_ref = float(max(p_scales))
    q_ref = np.asarray(error_model.probabilities(p_ref), dtype=np.float64)
    num_nonzero = int(np.count_nonzero(q_ref))
    if num_nonzero == 0:
        raise ValueError("All mechanism probabilities are zero at p_ref.")

    if a is None:
        if num_observables is None:
            catalog = getattr(error_model, "catalog", None)
            num_observables = getattr(catalog, "num_observables", None)
        if num_observables is None:
            raise ValueError(
                "Provide either a (high-weight asymptote) or num_observables so a = 1 - 2^-K can be computed."
            )
        a = 1.0 - 2.0 ** (-int(num_observables))

    if weights is None:
        if w_lo is None:
            w_lo = max(int(math.ceil(w0)), 1) if w0 is not None else 1
        if w_hi is None:
            mu = float(np.sum(q_ref))
            sd = math.sqrt(float(np.sum(q_ref * (1.0 - q_ref))))
            w_hi = int(math.ceil(mu + 4.0 * sd))
        w_hi = min(max(w_hi, w_lo + 1), num_nonzero)
        weights = logspaced_integer_weights(w_lo, w_hi, num_weights)
    else:
        weights = sorted({int(w) for w in weights})
        if weights and (weights[0] < 1 or weights[-1] > num_nonzero):
            raise ValueError(f"weights must lie in [1, {num_nonzero}]; got {weights[0]}..{weights[-1]}.")

    rng = np.random.default_rng(seed)
    samples: list[SpectrumSample] = []
    for w in weights:
        trials, failures = sample_fixed_weight_failure_fraction(
            simulator,
            q_ref,
            w,
            rng,
            target_failures=target_failures,
            max_trials=max_trials_per_weight,
            batch_size=batch_size,
        )
        sample = SpectrumSample(weight=w, trials=trials, failures=failures)
        samples.append(sample)
        if verbose:
            print(
                f"weight {w} | trials={trials} | failures={failures} | "
                f"f_hat={sample.failure_fraction:.6g} | stderr={sample.stderr:.3g}"
            )
            sys.stdout.flush()

    if min_aux_sigma_log <= 0:
        raise ValueError(f"min_aux_sigma_log must be positive; got {min_aux_sigma_log}.")
    aux_used: list[WeightPoint] = []
    for pt in aux_points or []:
        if pt.kind != "f_w":
            raise ValueError(f"aux_points must have kind='f_w'; got {pt.kind!r} (weight {pt.weight}).")
        if pt.estimate <= 0 or not math.isfinite(pt.rel_err):
            if verbose:
                print(
                    f"skipping aux point at weight {pt.weight}: "
                    f"estimate={pt.estimate:.6g}, rel_err={pt.rel_err:.6g}"
                )
                sys.stdout.flush()
            continue
        aux_used.append(pt)
        if verbose:
            print(
                f"aux weight {pt.weight} | f={pt.estimate:.6g} | "
                f"sigma_log={max(pt.rel_err, min_aux_sigma_log):.3g} | method={pt.method}"
            )
            sys.stdout.flush()

    spectrum, fit_report = fit_failure_spectrum(
        [s.weight for s in samples],
        [s.trials for s in samples],
        [s.failures for s in samples],
        a=a,
        ansatz=ansatz,
        w0=w0,
        aux_weights=[float(pt.weight) for pt in aux_used],
        aux_fractions=[float(pt.estimate) for pt in aux_used],
        aux_sigma_log=[max(float(pt.rel_err), min_aux_sigma_log) for pt in aux_used],
    )
    if verbose:
        print(
            f"fitted {ansatz}-parameter ansatz | a={spectrum.a:.6g} | w0={spectrum.w0:.6g} | "
            f"f0={spectrum.f0:.6g} | gamma1={spectrum.gamma1:.6g}"
            + (f" | gamma2={spectrum.gamma2:.6g} | wc={spectrum.wc:.6g}" if spectrum.ansatz == "5" else "")
            + f" | chi2/point={fit_report['chi2_per_point']:.3g}"
        )
        sys.stdout.flush()

    failure_estimates: list[float] = []
    log_failure_estimates: list[float] = []
    for p in p_scales:
        q_p = np.asarray(error_model.probabilities(p), dtype=np.float64)
        p_fail = transform_spectrum_to_failure_rate(spectrum, q_p, max_weight=transform_max_weight)
        failure_estimates.append(p_fail)
        log_failure_estimates.append(math.log(p_fail) if p_fail > 0 else -float("inf"))
        if verbose:
            print(f"p={p:.6g} | P_fail={p_fail:.6e}")
            sys.stdout.flush()

    return FailureSpectrumResult(
        p_scales=[float(p) for p in p_scales],
        failure_estimates=failure_estimates,
        log_failure_estimates=log_failure_estimates,
        p_ref=p_ref,
        samples=samples,
        spectrum=spectrum,
        fit_report=fit_report,
        aux_points=aux_used,
    )


class FailureSpectrumEstimator(Estimator):
    """Estimator implementing the failure-spectrum ansatz (arXiv:2511.15177)."""

    def estimate(
        self,
        error_model: ErrorModel,
        simulator: ForwardSimulator,
        **kwargs: Any,
    ) -> FailureSpectrumResult:
        if "p_scales" not in kwargs:
            raise ValueError("p_scales must be provided to FailureSpectrumEstimator")

        return failure_spectrum_estimate(
            error_model=error_model,
            simulator=simulator,
            p_scales=kwargs["p_scales"],
            p_ref=kwargs.get("p_ref"),
            weights=kwargs.get("weights"),
            num_weights=kwargs.get("num_weights", 12),
            w_lo=kwargs.get("w_lo"),
            w_hi=kwargs.get("w_hi"),
            target_failures=kwargs.get("target_failures", 100),
            max_trials_per_weight=kwargs.get("max_trials_per_weight", 20_000),
            batch_size=kwargs.get("batch_size", 4096),
            ansatz=kwargs.get("ansatz", "3"),
            w0=kwargs.get("w0"),
            a=kwargs.get("a"),
            num_observables=kwargs.get("num_observables"),
            transform_max_weight=kwargs.get("transform_max_weight"),
            aux_points=kwargs.get("aux_points"),
            min_aux_sigma_log=kwargs.get("min_aux_sigma_log", 0.05),
            seed=kwargs.get("seed", 1),
            verbose=kwargs.get("verbose", True),
        )

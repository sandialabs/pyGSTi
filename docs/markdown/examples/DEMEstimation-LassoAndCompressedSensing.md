---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.19.5
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Lasso and Compressed Sensing for DEM Estimation

This is the companion to [`DEMEstimation.ipynb`](DEMEstimation.ipynb), which introduces
detector error models, the data conventions of `pygsti.extras.sparsedem`, and the two
workhorse estimators (dense Hadamard estimation with thresholding, and lattice pruning). Here
we cover the *alternative* sparse-recovery routes:

1. **Non-negative lasso model selection** (`model_selection.lasso_dem_selection`) — convex
   sparse regression on the dense estimate, with automatic penalty selection by BIC;
2. **Compressed sensing** (`compressed_sensing.estimate_sparse_wh`) — an experimental sketch
   that recovers sparse event sets from a *budgeted* set of parity-mask polarizations, never
   materializing the $2^n$ Hadamard transform;
3. **Fitting a known support** (`SparseDEMEstimator.fit_custom_masks`) — when you already know
   *which* events exist and only need unbiased probabilities with error bars.

We reuse the running example from the main notebook — 10 detectors, 20 graph-like events
(10 boundary singles, 10 ring pairs) — with the same seed, so numbers are directly comparable,
and we benchmark every method against lattice pruning at the end.

```{code-cell} ipython3
import time

import numpy as np

from pygsti.extras.sparsedem.core import SparseDEMEstimator
from pygsti.extras.sparsedem import io as sdio
from pygsti.extras.sparsedem import utils as sdutils
from pygsti.extras.sparsedem.model_selection import lasso_dem_selection
from pygsti.extras.sparsedem.compressed_sensing import CSConfig, estimate_sparse_wh
from pygsti.extras.sparsedem.estimation import fit_specified_dem

SEED = 2026
np.set_printoptions(precision=4, suppress=True)

# --- The running example from DEMEstimation.ipynb -------------------------
N_DETECTORS = 10


def build_ground_truth():
    """20 distinct graph-like events: 10 boundary singles + 10 ring pairs."""
    masks, probs = [], []
    for d in range(N_DETECTORS):
        masks.append(1 << d)
        probs.append(0.008 + 0.001 * d)
    for d in range(N_DETECTORS):
        d2 = (d + 1) % N_DETECTORS
        masks.append((1 << d) | (1 << d2))
        probs.append(0.010 + 0.001 * d)
    order = np.argsort(masks)
    return np.array(masks)[order], np.array(probs)[order]


def mask_to_str(mask):
    return " ".join(f"D{d}" for d in range(N_DETECTORS) if (mask >> d) & 1)


def compare_to_truth(name, learned_dict, seconds=None):
    """One-line scoreboard against the ground truth."""
    found = sorted(m for m in learned_dict if m in true_set)
    spurious = sorted(m for m in learned_dict if m not in true_set)
    missed = sorted(true_set - set(learned_dict))
    max_err = (max(abs(learned_dict[m] - true_prob_of[m]) for m in found)
               if found else float('nan'))
    timing = f" in {seconds:.2f} s" if seconds is not None else ""
    print(f"{name}{timing}: {len(found)}/{len(true_set)} true events, "
          f"{len(spurious)} spurious, {len(missed)} missed, "
          f"max |p_hat - p| = {max_err:.4f}")
    return {"found": found, "spurious": spurious, "missed": missed,
            "max_err": max_err}


rng = np.random.default_rng(SEED)
true_masks, true_probs = build_ground_truth()
true_set = set(true_masks.tolist())
true_prob_of = dict(zip(true_masks.tolist(), true_probs.tolist()))

H_true = np.zeros((N_DETECTORS, len(true_masks)), dtype=np.uint8)
for j, mask in enumerate(true_masks):
    for d in range(N_DETECTORS):
        H_true[d, j] = (int(mask) >> d) & 1

N_SHOTS = 4000
occurrences = (rng.random((N_SHOTS, len(true_masks))) < true_probs).astype(np.uint8)
syndromes = ((occurrences @ H_true.T) % 2).astype(np.uint8)
syndrome_counts = dict(sdutils.counts_from_samples(syndromes))

# Lattice pruning as the reference point for every comparison below.
estimator = SparseDEMEstimator(syndrome_counts)
t0 = time.perf_counter()
lattice_dem = estimator.estimate_lattice_pruned(confidence=0.999)
t_lattice = time.perf_counter() - t0
lattice_dict = sdio.dem_to_dict(lattice_dem)
scores = {}
scores["lattice pruning"] = compare_to_truth("lattice pruning", lattice_dict, t_lattice)
```

## 1. Non-negative lasso model selection

The dense Hadamard estimator (main notebook, section 3) returns a point estimate
$\hat p \in \mathbb{R}^{2^n}$ *and* its covariance $\Sigma$. Thresholding tests each entry in
isolation; the lasso instead treats event selection as one joint, convex problem:

$$\min_{p \ge 0}\ \tfrac12\,(p - \hat p)^\top \Sigma^{+} (p - \hat p) \;+\; \lambda\,\mathbf{1}^\top p .$$

The quadratic term is the Gaussian log-likelihood of the dense estimate, *whitened* by the
(eigenvalue-truncated) pseudo-inverse of $\Sigma$ — truncation projects out the near-zero
multinomial-constraint direction. On the positive orthant the $\ell_1$ penalty is linear, so
the optimization problem is a smooth QP solved by projected gradient descent.

The implementation here contains to additional refinements to accelerate the computation and 
reduce bias:

* **Automatic $\lambda$:** the solver sweeps a geometric $\lambda$ path from $\lambda_{\max}$
  (empty model) downward and scores each distinct event by BIC after an unpenalized
  refit on that support (the "relaxed lasso"), picking the support with the best score.
* **Debiasing:** the winning support is refit with `fit_specified_dem`, removing the lasso's
  shrinkage bias and providing a covariance for the sparse model.

The implementation is pure NumPy (no external convex solver). Like other dense methods it
uses $2^n$-dimensional vectors, so it is only viable for small detector counts. This approach
improves on per-entry thresholding that ignores covariance.

```{code-cell} ipython3
t0 = time.perf_counter()
lasso_dem, lasso_masks, lasso_probs, lasso_cov, lasso_info = lasso_dem_selection(
    syndrome_counts, n_lambdas=25, max_iter=3000, return_path=True)
t_lasso = time.perf_counter() - t0

lasso_dict = sdio.dem_to_dict(lasso_dem)
scores["lasso"] = compare_to_truth("lasso (BIC-selected)", lasso_dict, t_lasso)
print(f"lambda path: {len(lasso_info['lambdas'])} points, "
      f"{len(lasso_info['supports'])} distinct supports scored by BIC, "
      f"lambda_best = {lasso_info['lambda_best']:.3g}")

# The same workflow is available on the estimator object as estimate_lasso().
best_k = int(np.argmin(lasso_info['bics']))
print(f"BIC-optimal support size: {len(lasso_info['supports'][best_k])} events")
```

## 2. Compressed sensing with randomized low-weight masks

Both methods above start from the dense $2^n$ estimate. `compressed_sensing.estimate_sparse_wh`
avoids it entirely, which is the point: it is a *sketch* of how estimation can scale when
$2^n$ is out of reach.

The idea: each parity mask $m$ gives one linear measurement of the DEM in the log domain.
Writing $a_e = -\tfrac12\log(1 - 2p_e)$ for the *attenuation* of event $e$, the measured
depolarization of mask $m$ is

$$-\log\,\langle(-1)^{m\cdot s}\rangle \;=\; \sum_{e\,:\,|m \wedge e|\ \mathrm{odd}} 2\,a_e ,$$

i.e. a row of a (0/2)-valued masked-Hadamard operator applied to the sparse attenuation
vector. The routine

1. draws a **budget** of measurement masks, favoring low Hamming weight (low-weight
   polarizations are the best-conditioned: they stay near 1 and their logs are well behaved);
2. measures their polarizations directly from the syndrome counts;
3. solves an $\ell_1$-regularized least-squares problem (ISTA, with optional positivity) over a
   **candidate event set** — by default all events up to `max_weight`, or an explicit
   `candidate_masks` list when you want specific high-weight events without enumerating
   $2^n$ — using a lazy operator, so the full Hadamard matrix is never built.

Assumptions and caveats: the true events must lie in the candidate set; the $\ell_1$ penalty
biases probabilities low (debias by refitting the recovered support, below); and mask/budget
design is an open knob — this module is exploratory, not the recommended default.

```{code-cell} ipython3
config = CSConfig(max_weight=2, budget=200, l1_penalty=3e-4, positivity=True, seed=SEED)
t0 = time.perf_counter()
cs_probs, cs_attenuations, candidate_masks, used_masks = estimate_sparse_wh(
    syndrome_counts, config)
t_cs = time.perf_counter() - t0

print(f"candidate events (weight <= {config.max_weight}): {len(candidate_masks)}; "
      f"measurement masks used: {len(used_masks)} (budget {config.budget})")

# Keep events with non-negligible recovered probability.
cs_dict = {m: p for m, p in zip(candidate_masks, cs_probs) if p > 2e-3}
scores["compressed sensing"] = compare_to_truth("compressed sensing", cs_dict, t_cs)

# The l1 shrinkage biases probabilities low and lets some noise through.
# Debias by refitting the recovered support unpenalized:
cs_refit_dem = fit_specified_dem(syndrome_counts, sorted(cs_dict), atol=1e-4)
cs_refit_dict = sdio.dem_to_dict(cs_refit_dem)
scores["cs + refit"] = compare_to_truth("cs + unpenalized refit", cs_refit_dict)

# A tighter budget sub-samples the measurement masks: recovery degrades but
# the workflow still runs without ever touching all 2^n masks.
config_small = CSConfig(max_weight=2, budget=30, l1_penalty=3e-4, seed=SEED)
cs_probs_s, _, cand_s, used_s = estimate_sparse_wh(syndrome_counts, config_small)
cs_dict_s = {m: p for m, p in zip(cand_s, cs_probs_s) if p > 2e-3}
compare_to_truth(f"compressed sensing ({len(used_s)}-mask budget)", cs_dict_s);
```

## 3. Fitting a known support: `fit_custom_masks`

Often you are not discovering events at all — the device layout or the decoder graph already
tells you which events are possible, and you only want their probabilities. Skipping the
selection step entirely is then both faster and statistically cleaner: `fit_custom_masks`
solves the polarization system restricted to your masks (`fit_specified_dem` under the hood)
and returns unbiased probabilities with a covariance, from which error bars follow.

Here we cheat and pass the exact ground-truth support; in practice the support would come
from a stim circuit's DEM, hardware connectivity, or a previous estimation run.

```{code-cell} ipython3
t0 = time.perf_counter()
fit_dem, fit_masks, fit_probs, fit_cov = estimator.fit_custom_masks(
    sorted(true_set), return_covariance=True)
t_fit = time.perf_counter() - t0

fit_dict = sdio.dem_to_dict(fit_dem)
scores["known support"] = compare_to_truth("known-support refit", fit_dict, t_fit)

fit_se = np.sqrt(np.diag(fit_cov))
print(f"\n{'event':10s} {'p_true':>8s} {'p_fit':>8s} {'std err':>8s} {'pull':>6s}")
n_within_2se = 0
for m, p_hat, se in zip(fit_masks, fit_probs, fit_se):
    pull = (p_hat - true_prob_of[int(m)]) / se
    n_within_2se += abs(pull) < 2
    print(f"{mask_to_str(int(m)):10s} {true_prob_of[int(m)]:8.4f} {p_hat:8.4f} "
          f"{se:8.4f} {pull:6.2f}")
print(f"\n{n_within_2se}/{len(fit_masks)} estimates within 2 standard errors of truth")
```

## 4. Which method when?

The scoreboard below gathers every method on the same 4000 shots of the same DEM.

| method | scales past small $n$? | needs | strengths | prefer when |
|---|---|---|---|---|
| dense + threshold | no ($2^n$) | nothing | simple, exact transform | quick look, $n \lesssim 15$ |
| **lattice pruning** | **yes** (prunes) | nothing | scales with true sparsity, calibrated tests | **default choice** |
| lasso (BIC) | no ($2^n$) | nothing (pure NumPy) | joint selection, handles correlated estimates | small $n$, thresholding is borderline |
| compressed sensing | yes (budgeted) | candidate set | never touches $2^n$; explicit measurement budget | exploratory, very large $n$ with a good candidate list |
| known-support fit | yes | the support | unbiased + error bars, no selection noise | support known from circuit/decoder graph |

Note the pattern in the scoreboard: on a small, well-sampled problem the selection-based
methods (thresholding, lattice pruning, lasso) all find the same support, and their refit
probabilities agree because they share the same final fitting step. Compressed sensing pays
for its scalability with selection noise — and even it recovers most of the support. The
known-support fit is the accuracy ceiling: everything else can only aspire to match it.

```{code-cell} ipython3
# Add the dense-threshold result from the main notebook's workflow for completeness.
t0 = time.perf_counter()
threshold_dem = estimator.threshold(alpha=0.05)
t_thr = time.perf_counter() - t0
scores["dense + threshold"] = compare_to_truth(
    "dense + threshold", sdio.dem_to_dict(threshold_dem), t_thr)

print(f"\n{'method':22s} {'true found':>10s} {'spurious':>9s} {'missed':>7s} {'max |dp|':>9s}")
for name in ["dense + threshold", "lattice pruning", "lasso",
             "compressed sensing", "cs + refit", "known support"]:
    s = scores[name]
    print(f"{name:22s} {len(s['found']):>7d}/20 {len(s['spurious']):>9d} "
          f"{len(s['missed']):>7d} {s['max_err']:>9.4f}")
```

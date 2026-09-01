---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.19.5
kernelspec:
  display_name: rare-event
  language: python
  name: python3
---

# 04 — Failure-spectrum ansatz (IBM "Fail fast")

`FailureSpectrumEstimator` implements the failure-spectrum technique from Beverland, Carroll, Cross & Yoder, *"Fail fast: techniques to probe rare events in quantum error correction"*, [arXiv:2511.15177](https://arxiv.org/abs/2511.15177).

The **failure spectrum** $f(w)$ is the probability that exactly $w$ simultaneously active mechanisms cause a logical failure. The logical error rate decomposes exactly over fault weights:

$$P_{\rm fail}(p) \;=\; \sum_w f(w)\; \Pr[W = w;\, q(p)],$$

where $W$ is the number of active mechanisms (binomial in the paper's uniform-rate setting; here the **Poisson-binomial** distribution of the heterogeneous $q_i(p)$). The key insight: $f(w)$ is *cheap to measure* at weights where failures are common, even when $P_{\rm fail}$ is astronomically small — and it empirically follows a low-parameter ansatz (Eq. 10 of the paper):

$$f^{(3)}_{\rm ansatz}(w) = a\left[1 - \exp\!\left(-\tfrac{f_0}{a}\,(w/w_0)^{\gamma}\right)\right],\qquad f(w < w_0) = 0,\qquad a = 1 - 2^{-K},$$

with onset weight $w_0$ (min failing weight), onset fraction $f_0$, and $K$ logical observables. Fit the ansatz once, and the transform predicts $P_{\rm fail}(p)$ at **every** $p$.

Two generalizations for this package's heterogeneous catalogs: fixed-weight sets are drawn from the exact conditional distribution $P(E \mid |E| = w)$ (exponential tilting + rejection — the clean equivalent of the paper's "expanded representation"), and the transform uses the Poisson-binomial pmf.

```{code-cell} ipython3
import contextlib
import io

import matplotlib.pyplot as plt
import numpy as np
import pymatching

from pygsti.extras.rareevent.noise import ExactNoiseErrorModel, SI1000NoiseModel
from pygsti.extras.rareevent.rare_event import (
    FailureOracle,
    direct_monte_carlo_failure_rate,
    make_repetition_code_memory_circuit,
)
from pygsti.extras.rareevent.failure_spectrum import FailureSpectrumEstimator
from pygsti.extras.rareevent.malignant import MalignantSetEstimator
```

```{code-cell} ipython3
p_ref = 0.02

# Noiseless skeleton circuit; the NoiseModel decorates it as a function of p.
circuit = make_repetition_code_memory_circuit(distance=5, rounds=2, p=0)
noise_model = SI1000NoiseModel()
error_model = ExactNoiseErrorModel(circuit, noise_model, p_ref=p_ref)
catalog = error_model.catalog

# Decoder built from the same DEM, wrapped as a ForwardSimulator.
dem = noise_model(circuit, p_ref).detector_error_model(decompose_errors=True, flatten_loops=True)
matching = pymatching.Matching.from_detector_error_model(dem)
oracle = FailureOracle(catalog, matching)

print(catalog)
```

## Sample the spectrum and fit the ansatz

For this distance-5 code with min-weight decoding the onset weight is $w_0 = \lceil d/2 \rceil = 3$. Sampling stops at each weight once `target_failures` failures are seen, which spreads the failure counts roughly uniformly across weights — the paper's recommended allocation.

```{code-cell} ipython3
p_scales = [float(x) for x in np.geomspace(p_ref, 1e-4, 8)]

estimator = FailureSpectrumEstimator()
result = estimator.estimate(
    error_model=error_model,
    simulator=oracle,
    p_scales=p_scales,
    w0=3,
    ansatz='3',
    target_failures=200,
    max_trials_per_weight=60_000,
    seed=7,
)
```

```{code-cell} ipython3
spec = result.spectrum
w_fine = np.linspace(1.0, 14.0, 400)

fig, ax = plt.subplots(figsize=(6, 4))
ax.errorbar(
    [s.weight for s in result.samples],
    [s.failure_fraction for s in result.samples],
    yerr=[s.stderr for s in result.samples],
    fmt='o', capsize=3, label=r'measured $\hat f(w)$',
)
ax.plot(w_fine, spec(w_fine), '-', label=f'fitted {spec.ansatz}-parameter ansatz')
ax.axhline(spec.a, color='k', ls=':', alpha=0.6, label=f'asymptote a = {spec.a:g}')
ax.axvline(spec.w0, color='gray', ls='--', alpha=0.6, label=f'onset w0 = {spec.w0:g}')
ax.set_yscale('log')
ax.set_xlabel('fault weight $w$'); ax.set_ylabel('failure fraction $f(w)$')
ax.set_title('Failure spectrum, d=5 repetition code (SI1000)')
ax.legend(loc='lower right'); ax.grid(alpha=0.3)
plt.tight_layout()
```

## Predictions at all $p$, validated against the other methods

The fitted spectrum was already transformed through the Poisson-binomial weight distribution at each $p$ in `p_scales` (printed above). Compare against exact malignant-set enumeration at low $p$ and direct Monte Carlo at the reference rate:

```{code-cell} ipython3
with contextlib.redirect_stdout(io.StringIO()):
    res_mal = MalignantSetEstimator().estimate(
        error_model=error_model,
        simulator=oracle,
        p_scales=p_scales,
        max_weight=4,
        num_mechanisms=error_model.num_mechanisms,
    )
np.random.seed(4)
mc, se, _ = direct_monte_carlo_failure_rate(oracle, error_model.probabilities(p_ref), 30_000)

print(f"direct MC at p_ref: {mc:.4e} +- {se:.1e} (spectrum: {result.failure_estimates[0]:.4e})")
print()
for p, f_spec, f_mal in zip(p_scales, result.failure_estimates, res_mal['failure_estimates']):
    print(f"p={p:.4g}: spectrum {f_spec:.4e} | malignant w<=4 {f_mal:.4e} | ratio {f_spec / f_mal:.3f}")
```

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(p_scales, result.failure_estimates, 'o-', label='failure-spectrum ansatz')
ax.plot(p_scales, res_mal['failure_estimates'], 's--', label='malignant sets (w<=4, lower bound)')
ax.errorbar([p_ref], [mc], yerr=[se], fmt='k*', markersize=12, capsize=3, label='direct MC (30k shots)')
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel('physical error rate $p$'); ax.set_ylabel(r'$P_{\rm fail}$')
ax.set_title('Failure-spectrum predictions, d=5 repetition code (SI1000)')
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
```

## Practical notes

- **Onset weight**: pass `w0 = ceil(d/2)` for a min-weight decoder; pass `w0=None` to fit it (suboptimal decoders can have a lower effective onset).
- **Ansatz choice**: `'3'` (free exponent) is a good default; `'2'` fixes the exponent to $w_0$; `'5'` adds a crossover between low- and high-weight exponents for systems where one power law isn't enough. Check `result.fit_report['chi2_per_point']`.
- **Assumption**: applying $f(w)$ measured at `p_ref` across all $p$ assumes the *relative* mechanism probabilities are $p$-independent (exact for linear scaling, first-order accurate for SI1000-style models).
- **Why it scales**: cost is a few thousand decoder calls per sampled weight — independent of how small $P_{\rm fail}$ is. The paper applies this to distance-6/12/18 bivariate bicycle codes at failure rates below $10^{-12}$.

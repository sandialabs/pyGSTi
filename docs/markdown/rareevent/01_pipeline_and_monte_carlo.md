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

# 01 — The estimation pipeline and direct Monte Carlo

This package predicts logical error rates of quantum error-correcting (QEC) circuits at physical error rates $p$ far below what direct sampling can reach. Its design keeps the physical simulation strictly decoupled from the statistical estimation methods via three protocols (`pygsti/extras/rareevent/interfaces.py`):

- **`ErrorModel`** — `probabilities(p)`: independent Bernoulli probabilities $q_i(p)$, one per error mechanism.
- **`ForwardSimulator`** — `fails(active)`: does this set of active mechanisms cause a logical failure after decoding?
- **`Estimator`** — `estimate(error_model, simulator, ...)`: a statistical method built only on the two interfaces above.

The standard `stim`-based pipeline is:

```
noiseless stim.Circuit → NoiseModel decoration → detector error model (DEM)
        → MechanismCatalog (state space)  +  pymatching.Matching (decoder)
        → FailureOracle (ForwardSimulator)
```

This notebook builds that pipeline for a small repetition code and runs the direct Monte Carlo baseline — then shows why Monte Carlo alone is not enough.

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
```

## Build the pipeline

We use a distance-5 repetition-code memory circuit under SI1000 circuit-level noise (2-qubit gates depolarize with probability $p$, measurement flips with $2p$, idling with $p/3$, ...). `ExactNoiseErrorModel` re-decorates the circuit and regenerates the DEM at every requested $p$, so mechanism probabilities are exact rather than linearly scaled. Each independent DEM error becomes one `ErrorMechanism` in the catalog, and **a mechanism's identity is its index**.

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

The mechanism probabilities are heterogeneous — different fault locations get different rates under SI1000 — which is why the estimators in this package work with per-mechanism $q_i(p)$ rather than a single uniform rate.

```{code-cell} ipython3
q = error_model.probabilities(p_ref)
print(f"mechanisms: {len(q)}")
print(f"expected number of active mechanisms at p_ref: {q.sum():.3f}")
print(f"distinct mechanism probabilities at p_ref: {np.unique(np.round(q, 8))}")
```

`explain_mechanism` maps a catalog index back to the physical circuit faults that produce it, via stim's `explain_detector_error_model_errors`:

```{code-cell} ipython3
print(f"mechanism [0]: {catalog.mechanisms[0]}")
print()
print(error_model.explain_mechanism(0)[0])
```

The `FailureOracle` implements `ForwardSimulator`: given a set of active mechanism indices it computes the syndrome, decodes it, and compares the prediction against the true logical flips.

```{code-cell} ipython3
print(f"fails(empty set)   = {oracle.fails(set())}")
print(f"fails({{0}})         = {oracle.fails({0})}")
print(f"fails({{0, 1, 2, 3}}) = {oracle.fails({0, 1, 2, 3})}")
```

## Direct Monte Carlo baseline

Sample mechanism sets from the Bernoulli product distribution and count decoding failures. This is exact and unbiased — but the cost to see even one failure grows like $1/P_{\rm fail}(p)$, so it becomes intractable exactly in the regime we care about.

```{code-cell} ipython3
np.random.seed(0)
shots = 20_000
p_list = [0.04, 0.03, 0.02, 0.012, 0.008]
mc_estimates, mc_errors = [], []
for p in p_list:
    phat, se, _ = direct_monte_carlo_failure_rate(oracle, error_model.probabilities(p), shots)
    mc_estimates.append(phat)
    mc_errors.append(se)
    print(f"p={p:.4g}: P_fail = {phat:.3e} +- {se:.1e}  ({int(round(phat * shots))} failures in {shots} shots)")
```

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(6, 4))
ax.errorbar(p_list, mc_estimates, yerr=mc_errors, fmt='o-', capsize=3, label=f'direct MC ({shots} shots)')
ref = mc_estimates[0] * (np.asarray(p_list) / p_list[0]) ** 3
ax.plot(p_list, ref, 'k--', alpha=0.5, label=r'$\propto p^{3}$ (distance 5)')
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel('physical error rate $p$'); ax.set_ylabel(r'$P_{\rm fail}$')
ax.set_title('Direct Monte Carlo, d=5 repetition code (SI1000)')
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
```

At $p \approx 0.008$ this budget already yields only a handful of failures; at $p = 10^{-3}$ the failure rate is $\sim 10^{-7}$ and direct sampling would need $\gtrsim 10^{8}$ shots per point. The other notebooks demonstrate the estimators that take over from here:

- **02 — rare-event splitting** (Bravyi–Vargo MCMC): moderate distances and rates.
- **03 — malignant set counting**: exact enumeration, tight as $p \to 0$.
- **04 — failure-spectrum ansatz** (IBM "Fail fast", arXiv:2511.15177): fit a low-parameter spectrum and predict all $p$ at once.

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

# 03 — Malignant set counting

`MalignantSetEstimator` brute-force enumerates every mechanism combination up to `max_weight` and sums the exact probability of each *malignant* (failure-causing) configuration:

$$P_{\rm fail}(p) \;\geq\; \sum_{E\;\text{malignant},\;|E| \le w_{\max}} \;\prod_{i \in E} q_i(p) \prod_{i \notin E} \bigl(1 - q_i(p)\bigr).$$

This is a strict lower bound that becomes exact as $p \to 0$, because low-weight configurations dominate. The price is combinatorial: $\binom{N}{w}$ decoder calls at weight $w$.

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

## Enumerate malignant sets

This distance-5 code has onset weight $\lceil d/2 \rceil = 3$: no set of fewer than 3 mechanisms can fool the decoder, which the weight-1 and weight-2 counts confirm.

```{code-cell} ipython3
p_grid = [float(x) for x in np.geomspace(p_ref, 1e-5, 12)]

estimator = MalignantSetEstimator()
res_w3 = estimator.estimate(
    error_model=error_model,
    simulator=oracle,
    p_scales=p_grid,
    max_weight=3,
    num_mechanisms=error_model.num_mechanisms,
)
```

## What *is* a malignant set, physically?

Each malignant set is a tuple of catalog indices. `explain_malignant_set` maps it back to physical circuit faults via stim, so you can see exactly which gate/measurement errors conspire to defeat the decoder:

```{code-cell} ipython3
worst = res_w3['malignant_sets'][0]
print(f"malignant set {worst}:")
for i in worst:
    print(f"  [{i}] {catalog.mechanisms[i]}")
print()
explained = error_model.explain_malignant_set(worst)
print(f"physical faults ({len(explained)} explained errors), first one:")
print(explained[0])
```

## Convergence in `max_weight`

Adding weight-4 configurations tightens the bound at moderate $p$; at low $p$ the weight-3 contribution dominates and the two curves converge — the enumeration is *exact* in the $p \to 0$ limit.

```{code-cell} ipython3
with contextlib.redirect_stdout(io.StringIO()):
    res_w4 = estimator.estimate(
        error_model=error_model,
        simulator=oracle,
        p_scales=p_grid,
        max_weight=4,
        num_mechanisms=error_model.num_mechanisms,
    )

for p, f3, f4 in zip(p_grid, res_w3['failure_estimates'], res_w4['failure_estimates']):
    print(f"p={p:.4g}: w<=3 gives {f3:.4e} | w<=4 gives {f4:.4e} | ratio {f4 / f3:.4f}")
```

```{code-cell} ipython3
np.random.seed(3)
mc_ps = [0.02, 0.012]
mc_est, mc_err = [], []
for p in mc_ps:
    phat, se, _ = direct_monte_carlo_failure_rate(oracle, error_model.probabilities(p), 30_000)
    mc_est.append(phat)
    mc_err.append(se)

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(p_grid, res_w3['failure_estimates'], 's--', label='malignant sets, w<=3')
ax.plot(p_grid, res_w4['failure_estimates'], 'o-', label='malignant sets, w<=4')
ax.errorbar(mc_ps, mc_est, yerr=mc_err, fmt='k*', markersize=12, capsize=3, label='direct MC (30k shots)')
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel('physical error rate $p$'); ax.set_ylabel(r'$P_{\rm fail}$')
ax.set_title('Malignant set counting, d=5 repetition code (SI1000)')
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
```

At $p = 10^{-5}$ the bound is effectively exact ($P_{\rm fail} \sim 10^{-13}$ — completely unreachable by sampling). The limitation is scale: at weight $w$ the enumeration costs $\binom{N}{w}$ decoder calls, so for thousands of mechanisms only very small $w_{\max}$ is feasible. Notebook 04 shows the failure-spectrum ansatz, which reaches the same regime statistically.

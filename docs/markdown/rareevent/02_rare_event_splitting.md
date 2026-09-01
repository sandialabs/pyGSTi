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

# 02 — Rare-event splitting (Bravyi–Vargo)

`RareEventSplittingEstimator` estimates tiny failure rates by *splitting* the problem across a descending schedule of physical rates `p_scales` $= p_0 > p_1 > \dots > p_K$:

$$P_{\rm fail}(p_K) \;=\; P_{\rm fail}(p_0)\; \prod_{k=0}^{K-1}\; \mathbb{E}_{E \sim P(\cdot\,|\,\text{fail},\,p_k)}\!\left[\frac{P_{p_{k+1}}(E)}{P_{p_k}(E)}\right]\!,$$

anchored by direct Monte Carlo at $p_0$ (where failures are common). Each level ratio is estimated with `ConditionalFailureMCMC`, a Metropolis chain over *failing* mechanism sets: propose toggling one mechanism, accept with the Bernoulli odds ratio, and reject any state that is not a logical failure.

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
from pygsti.extras.rareevent.rare_event import RareEventSplittingEstimator
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

## Run the estimator

Seven geometrically spaced levels from $p_0 = 0.02$ down to $p = 10^{-3}$. The per-level log output is captured to keep the notebook compact; everything it prints is also available in `result.level_diagnostics`.

```{code-cell} ipython3
np.random.seed(1)
p_scales = [float(x) for x in np.geomspace(p_ref, 1e-3, 7)]

estimator = RareEventSplittingEstimator()
with contextlib.redirect_stdout(io.StringIO()):
    result = estimator.estimate(
        error_model=error_model,
        simulator=oracle,
        p_scales=p_scales,
        mc_shots_at_p0=30_000,
        steps_per_chain=None,
        total_steps_per_level=15_000,
        burn_in=None,
        burn_in_fraction=0.1,
        thin=5,
        seed=1,
    )

for p, f in zip(result.p_scales, result.failure_estimates):
    print(f"p={p:.5g}: P_fail = {f:.4e}")
```

## Convergence diagnostics

Each level records the Metropolis acceptance rate, the mean number of active mechanisms in the failing states, and split-chain $\hat R$ statistics on the log weight ratios and active-set weights. $\hat R$ close to 1 is necessary (not sufficient!) for convergence — the chain can still miss disconnected failure modes, which is what the multi-seeded generalization in the "Fail fast" paper (arXiv:2511.15177) addresses for qLDPC codes.

```{code-cell} ipython3
for d in result.level_diagnostics:
    rhat_lr = 'n/a' if d.rhat_log_weight_ratio is None else f'{d.rhat_log_weight_ratio:.3f}'
    rhat_w = 'n/a' if d.rhat_active_weight is None else f'{d.rhat_active_weight:.3f}'
    print(
        f'level {d.level}: p {d.p_current:.5g} -> {d.p_next:.5g} | '
        f'acceptance={d.per_chain_acceptance_rates[0]:.3f} | '
        f'mean active weight={d.per_chain_mean_weights[0]:.2f} | '
        f'Rhat(log ratio)={rhat_lr} | Rhat(weight)={rhat_w}'
    )
```

## Cross-check against exact enumeration

At the low end of the schedule the failure rate is dominated by the minimum-weight malignant sets, so exact enumeration up to weight 3 (this is a distance-5 code, onset weight $\lceil d/2 \rceil = 3$) gives a tight independent reference.

```{code-cell} ipython3
with contextlib.redirect_stdout(io.StringIO()):
    res_mal = MalignantSetEstimator().estimate(
        error_model=error_model,
        simulator=oracle,
        p_scales=p_scales,
        max_weight=3,
        num_mechanisms=error_model.num_mechanisms,
    )

ratio = result.failure_estimates[-1] / res_mal['failure_estimates'][-1]
print(f"splitting at p={p_scales[-1]:.4g}:  {result.failure_estimates[-1]:.4e}")
print(f"malignant (w<=3) bound:    {res_mal['failure_estimates'][-1]:.4e}")
print(f"ratio: {ratio:.3f}")
```

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(result.p_scales, result.failure_estimates, 'o-', label='rare-event splitting')
ax.plot(p_scales, res_mal['failure_estimates'], 's--', label='malignant sets (w<=3, lower bound)')
np.random.seed(2)
mc, se, _ = direct_monte_carlo_failure_rate(oracle, error_model.probabilities(p_ref), 30_000)
ax.errorbar([p_ref], [mc], yerr=[se], fmt='k*', markersize=12, capsize=3, label='direct MC anchor')
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel('physical error rate $p$'); ax.set_ylabel(r'$P_{\rm fail}$')
ax.set_title('Rare-event splitting, d=5 repetition code (SI1000)')
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
```

The splitting estimate tracks the exact low-$p$ reference while only ever simulating at rates where failures are reachable by MCMC. Cost scales with the number of levels and the mixing time of the chain — for much larger systems or many inequivalent logical operators, see the failure-spectrum approach in notebook 04.

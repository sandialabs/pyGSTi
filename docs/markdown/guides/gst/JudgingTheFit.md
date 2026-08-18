---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.17.3
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Judging the fit

A GST estimate is a set of gate matrices and a number saying how well those matrices explain your counts. The second half gets skipped a lot, and skipping it is how people end up quoting a process fidelity from an estimate that does not describe their device. This page is about the number: what $2\Delta\log\mathcal{L}$ is, where its degrees of freedom come from, how $N_\sigma$ is built out of the two, and how to pull all of it out of a `ModelEstimateResults` object.

The conceptual framing lives in [key concepts](../../start/KeyConcepts); this page assumes you have read it and goes after the practitioner-level detail. If your fit turns out to be bad, [bad fits](BadFits) covers what to do next.

## What $2\Delta\log\mathcal{L}$ measures

Fix a circuit and an outcome. Your model predicts probability $p$; you observed that outcome $N f$ times out of $N$ repetitions, so $f$ is the observed frequency. The *maximal model* is the imaginary model that predicts $p = f$ for every circuit and outcome at once: it fits the data perfectly, by construction, and no model can do better. $\Delta\log\mathcal{L}$ is the gap between the log-likelihood of your model and the log-likelihood of that maximal model, and pyGSTi reports twice it.

The per-term contribution pyGSTi actually sums (`RawPoissonPicDeltaLogLFunction`) is

$$ N f \log(f/p) - N(f - p), $$

which is the Poisson-picture form of the log-likelihood ratio. Each term is non-negative and vanishes exactly when $p = f$, so $2\Delta\log\mathcal{L} = 0$ means a perfect fit and larger means worse. The units are meaningless on their own. A value of 261 tells you nothing until you know what value to expect.

## Why the likelihood and not chi-squared

$\chi^2$ answers the same question, and pyGSTi implements it too (`RawChi2Function`, summing $N(p-f)^2/p$). The two agree in the limit of many counts and probabilities away from 0 and 1. GST circuits violate the second condition on purpose: the whole point of repeating a germ many times is to amplify a small error until some outcome probability swings to nearly 0 or nearly 1, and that $p$ sits in the denominator of the $\chi^2$ term. pyGSTi has to regularize it, which is what `min_prob_clip_for_weighting` is for. The likelihood ratio needs no such patch, because it is the exact multinomial statistic rather than a quadratic approximation to it.

What actually runs by default is a hybrid, and it is worth knowing about because the naming in the results object reflects it. `GSTObjFnBuilders.create_from(objective='logl')`, the default, optimizes $\chi^2$ on every iteration and then adds one final maximum-likelihood step on the last (longest) circuit list. $\chi^2$ is a least-squares objective, so the optimizer can exploit its structure and get to the right neighborhood quickly; the MLE step is what fixes the answer. The statistic reported for goodness-of-fit is always $2\Delta\log\mathcal{L}$. Pass `objfn_builders=pygsti.protocols.GSTObjFnBuilders.create_from(objective='chi2')` to `GateSetTomography` if you want $\chi^2$ end to end, and the report's column headers change to match.

## Running a fit and reading the number back

```{code-cell} ipython3
import numpy as np
import pygsti
from pygsti.modelpacks import smq1Q_XYI

target_model = smq1Q_XYI.target_model("full TP")
edesign = smq1Q_XYI.create_gst_experiment_design(max_max_length=4)

# Data from a depolarized version of the target: inside the TP model class,
# so this fit should come out clean.
datagen_model = smq1Q_XYI.target_model().depolarize(op_noise=0.01, spam_noise=0.005)
ds = pygsti.data.simulate_data(datagen_model, edesign.all_circuits_needing_data,
                               num_samples=1000, sample_error='binomial', seed=1234)

data = pygsti.protocols.ProtocolData(edesign, ds)
results = pygsti.protocols.GateSetTomography(target_model, verbosity=0).run(data)
estimate = results.estimates['GateSetTomography']
```

`Estimate.misfit_sigma()` is the single call that gives you $N_\sigma$ for the final estimate. It is what the report uses.

```{code-cell} ipython3
print("N_sigma =", estimate.misfit_sigma())
```

## Degrees of freedom

$k$, the number of degrees of freedom, is the number of independent numbers in your data minus the number of parameters your model spent fitting them.

The data side is `DataSet.degrees_of_freedom`. A circuit with two possible outcomes contributes one independent number, not two, because the counts sum to $N$. The default counting method is `'present_outcomes-1'`, meaning per circuit it counts the outcomes that actually appear and subtracts one. For this single-qubit design that comes to exactly one per circuit.

The model side is `Model.num_modeltest_params`, which is the count of *non-gauge* parameters, not `num_params`. Gauge directions do not change any predicted probability (see [gauge freedom](../analysis/GaugeFreedom)), so they cannot be spent fitting data and must not be subtracted. For a one-qubit `full TP` model the gauge group has dimension $d^4 - d^2 = 12$, and you can see that show up as the gap between the two counts below.

```{code-cell} ipython3
final_model = estimate.models['final iteration estimate']
final_circuits = results.circuit_lists['final']

Ns = ds.degrees_of_freedom(final_circuits)      # independent numbers in the data
Np = final_model.num_modeltest_params           # non-gauge model parameters
k = max(Ns - Np, 1)

print("circuits:                    ", len(final_circuits))
print("dataset degrees of freedom:  ", Ns)
print("model parameters (total):    ", final_model.num_params)
print("model parameters (non-gauge):", Np)
print("k = Ns - Np:                 ", k)
```

When pyGSTi cannot work out the gauge dimension for a model it warns (`UnknownGaugeSpaceDimension`) and falls back to `num_params`, which makes $k$ too small and $N_\sigma$ too large. When $N_s \le N_p$ the model has at least as many free parameters as the data has numbers, $k$ is clamped to 1, and you get an `OverparameterizationWarning`; the fit statistic is uninterpretable in that regime because a perfect fit was guaranteed.

## How $N_\sigma$ is computed

If your model class contains the truth and the counts are large, $2\Delta\log\mathcal{L}$ is approximately $\chi^2_k$ distributed. That distribution has mean $k$ and standard deviation $\sqrt{2k}$, which is the whole basis for the conversion:

$$ N_\sigma = \frac{2\Delta\log\mathcal{L} - k}{\sqrt{2k}}. $$

The objective value itself is on the cached objective function. `CachedObjectiveFunction.fn` holds the raw objective value, which for the log-likelihood objective is $\Delta\log\mathcal{L}$, and `chi2k_distributed_fn` holds the version scaled to be $\chi^2_k$ distributed, which is twice that. Always take the `chi2k_distributed_*` quantity when you are going to compare against $k$; it is the one that has been converted for every objective function, so the arithmetic below works unchanged if you switch to $\chi^2$.

```{code-cell} ipython3
objfn_cache = estimate.final_objective_fn_cache()

two_dlogl = objfn_cache.chi2k_distributed_fn
print("raw objective value (deltaLogL):", objfn_cache.fn)
print("2*deltaLogL:                    ", two_dlogl)
print("expected value k:               ", k)
print("std deviation sqrt(2k):         ", np.sqrt(2 * k))
print("N_sigma, recomputed:            ", (two_dlogl - k) / np.sqrt(2 * k))
```

That should reproduce `misfit_sigma()` exactly, because it is the same arithmetic. Note that $N_\sigma$ is gauge-invariant: a gauge transformation leaves every predicted probability alone, so it leaves the likelihood alone. You get the same answer from `estimate.models['stdgaugeopt']` as from the un-gauge-optimized final model. That makes it one of the few numbers in a GST report you can quote without also stating how you gauge-fixed.

## The per-iteration progression

GST fits iteratively, adding longer circuits each round, and the sequence of $N_\sigma$ values across iterations is more informative than the final one alone. A fit that is clean at short circuit depths and degrades as $L$ grows points at gate-level non-Markovianity; a fit that is bad at every depth points at something wrong with SPAM, drift, or the data itself.

`pygsti.tools.two_delta_logl` returns the triple $(2\Delta\log\mathcal{L}, N_\sigma, p)$ when you give it `dof_calc_method='modeltest'`, which selects `num_modeltest_params` for the parameter count. Pair each iteration's model with the circuit list it was fit to.

```{code-cell} ipython3
print(f"{'L':>4} {'circuits':>9} {'2*dlogL':>10} {'k':>6} {'N_sigma':>9} {'p-value':>9}")
for i, (L, clist) in enumerate(zip(edesign.maxlengths, results.circuit_lists['iteration'])):
    mdl = estimate.models[f'iteration {i} estimate']
    val, nsigma, pvalue = pygsti.tools.two_delta_logl(mdl, ds, clist,
                                                      dof_calc_method='modeltest')
    kk = max(ds.degrees_of_freedom(clist) - mdl.num_modeltest_params, 1)
    print(f"{L:>4} {len(clist):>9} {val:>10.1f} {kk:>6} {nsigma:>9.2f} {pvalue:>9.3f}")
```

`pygsti.tools.two_delta_logl_nsigma` is the same computation when you only want $N_\sigma$ back.

## Which circuits are responsible

The aggregate number tells you that something is wrong but not where. `chi2k_distributed_percircuit` gives you the $2\Delta\log\mathcal{L}$ contribution of each circuit separately, indexed to match `objfn_cache.layout.circuits`.

Each entry is a one-degree-of-freedom quantity, and the threshold you should apply to it is *not* the one you would use for a single test. With several hundred circuits you are looking at the largest of several hundred draws, so a per-box cutoff of 4 is useless: $P(\chi^2_1 > 4) \approx 0.046$, which over 285 circuits means about 13 boxes above 4 in a fit that is behaving perfectly. pyGSTi's report colorings use a family-wise threshold instead, chosen so that the *worst* box has a given chance of exceeding it (`LinlogColormap` in `pygsti/report/colormaps.py`). For 285 circuits at the 5% level that threshold is about 14. Expect to see individual values near 10 in a clean fit, and judge a box suspicious only when it clears the family-wise line.

```{code-cell} ipython3
percircuit = objfn_cache.chi2k_distributed_percircuit
circuits = objfn_cache.layout.circuits

worst = np.argsort(percircuit)[::-1][:5]
for i in worst:
    print(f"{percircuit[i]:7.2f}   {circuits[i].str}")
```

For any real diagnosis you want the color box plot in the report rather than this list, because the plot arranges circuits by germ and length so that clusters of bad circuits are visible as clusters. Isolated large values are usually statistical; a whole germ's worth of them is a finding.

## $N_\sigma$ grows with shot count

This is the part that trips up people who read $N_\sigma$ as a pass/fail threshold. $N_\sigma$ measures *statistical significance* of model violation, not its size, and significance grows without bound as you take more data. A model that is wrong by a fixed small amount accumulates $2\Delta\log\mathcal{L}$ roughly linearly in $N$, while $k$ stays fixed, so $N_\sigma$ grows roughly linearly in $N$ too.

The cell below makes that concrete without any fitting. It takes a fixed, slightly wrong model (the ideal target) against data from a device with 0.1% depolarization per gate, at three shot counts.

```{code-cell} ipython3
circs = edesign.all_circuits_needing_data
slightly_off = smq1Q_XYI.target_model().depolarize(op_noise=0.001)

print(f"{'shots':>7} {'2*dlogL':>12} {'N_sigma':>10}")
for N in (100, 1000, 10000):
    ds_N = pygsti.data.simulate_data(slightly_off, circs, num_samples=N,
                                     sample_error='binomial', seed=2024)
    val, nsigma, _ = pygsti.tools.two_delta_logl(target_model, ds_N, circs,
                                                 dof_calc_method='modeltest')
    print(f"{N:>7} {val:>12.1f} {nsigma:>10.2f}")
```

Ten times the data, roughly ten times the $N_\sigma$, for exactly the same physical discrepancy. A well-sampled experiment on a good device will report a large $N_\sigma$, and reporting "$N_\sigma = 250$, so the fit failed" is not a statement about the device. The honest reading is that $N_\sigma$ tells you whether the violation is real and the per-circuit and wildcard-budget views tell you whether it is large enough to matter. pyGSTi's wildcard budget exists precisely to answer the size question: it computes how much slack a circuit's predicted outcome probabilities have to be granted, measured in total variation distance, before the model becomes consistent with the data. That is a statement about magnitude rather than about significance. See [bad fits](BadFits).

The corollary for experiment design: do not compare $N_\sigma$ between two datasets with different shot counts or different numbers of circuits, ever. Compare wildcard budgets, or compare $N_\sigma$ only within a single dataset across iterations.

## Where these numbers live in the report

The report's **Model Violation** tab is the canonical view, and everything above appears there in a form that is easier to read than the API. The per-$L$ progress table has one row per iteration with columns for $2\Delta\log\mathcal{L}$, $k$, their difference, $\sqrt{2k}$, $N_\sigma$, $N_s$, $N_p$, and a star rating. The rating is a coarse binning of $N_\sigma$ alone:

| $N_\sigma$ | Stars |
|---|---|
| $\le 2$ | 5 |
| $\le 20$ | 4 |
| $\le 100$ | 3 |
| $\le 500$ | 2 |
| $> 500$ | 1 |

Because it is a function of $N_\sigma$ only, the rating inherits the shot-count problem described above: a large, careful experiment on a good device can easily rate two or three stars. Treat it as a rough sort key, not as a verdict.

Two things are much easier to get from the report than from the API, and you should go there rather than reconstruct them: the per-sequence color box plot, which colors each circuit by its $2\Delta\log\mathcal{L}$ contribution and lays them out by germ and length, and the histogram of per-circuit contributions against the theoretical $\chi^2$ curve. Red boxes in the first are individually significant at a family-wise confidence level, so even one of them is evidence. See [reports](../analysis/Reports) for generating these.

## Caveats worth carrying

The $\chi^2_k$ reference distribution is asymptotic. It assumes enough counts per circuit that each term is near its Gaussian limit, and it assumes the fit is in the interior of the parameter space. Neither holds perfectly for GST. Circuits where one outcome is nearly certain contribute terms that are not close to Gaussian, which is what the `'tuned'` option to `DataSet.degrees_of_freedom` is an attempt to correct for; that option is still marked as under development in the source, and it is not the default. Constrained parameterizations like `CPTPLND` can land on the boundary of the feasible set, where the standard degrees-of-freedom counting is not exactly right either. In both cases the direction of the error is toward overstating $N_\sigma$.

None of this makes $N_\sigma$ useless. It makes it a strong ordinal signal and a weak cardinal one: trust it to tell you that this fit is much worse than that one, and do not lean hard on the claim that a particular $N_\sigma$ corresponds to a particular p-value.

If your fit is clean, go read the [error metrics](../analysis/Metrics), keeping the gauge caveat in mind. If it is not, [bad fits](BadFits) is the next page, and [model testing](../analysis/ModelTesting) is how you check whether some other model class does better without running a full GST fit against it.

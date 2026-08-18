---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.19.4
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Error bars

pyGSTi gives you two independent routes to uncertainties on a GST estimate. The first is the Hessian-based confidence region: pyGSTi computes the Hessian of the loglikelihood at the estimate, projects out the gauge directions, and hands you a region you can propagate through any function of the model. This is what reports use, and it is fast. The second is bootstrapping: refit the model many times on resampled data and take the spread of the results. Bootstrapping costs one GST fit per resample and makes fewer assumptions about the shape of the likelihood near its maximum, but in practice we have found the two agree closely enough that the Hessian route is usually the better trade.

This page runs both on the same simulated dataset so you can compare them.

## Setup

Simulate a noisy 1-qubit GST experiment. Everything below reuses this one dataset.

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt

import pygsti
from pygsti.modelpacks import smq1Q_XY
from pygsti.report import reportables as rptbl, modelfunction as modelfn
```

```{code-cell} ipython3
target_model = smq1Q_XY.target_model('full TP')

prep_fiducials = smq1Q_XY.prep_fiducials()
meas_fiducials = smq1Q_XY.meas_fiducials()
germs = smq1Q_XY.germs()
max_lengths = [1, 2, 4, 8, 16]

edesign = pygsti.protocols.StandardGSTDesign(
    smq1Q_XY.processor_spec(), prep_fiducials, meas_fiducials, germs, max_lengths)

mdl_datagen = target_model.depolarize(op_noise=0.1, spam_noise=0.001)
ds = pygsti.data.simulate_data(mdl_datagen, edesign, num_samples=1000,
                               sample_error='binomial', seed=1234)
```

Fit two parameterizations in one pass. The confidence-region section below uses the CPTPLND estimate. The bootstrap section uses the `full TP` estimate instead: the bootstrap helpers refit against whatever `target_model` you hand them and then pack the summarized results back into a copy of the estimate you pass in, so the estimate has to carry the same parameterization as `target_model`, which is `full TP` here.

```{code-cell} ipython3
gst_proto = pygsti.protocols.StandardGST(modes=['full TP', 'CPTPLND'], verbosity=1)
results = gst_proto.run(pygsti.protocols.ProtocolData(edesign, ds))
```

## Hessian-based confidence regions

Attach a `ConfidenceRegionFactory` to the estimate, compute and project the Hessian, then take a 95% confidence-region "view".

```{code-cell} ipython3
crfact = results.estimates['CPTPLND'].add_confidence_region_factory('stdgaugeopt', 'final')
crfact.compute_hessian(comm=None, mem_limit=3.0*(1024.0)**3)  # optionally use multiple processors & set memlimit
crfact.project_hessian('intrinsic error')
crf_view = results.estimates['CPTPLND'].confidence_region_factories['stdgaugeopt', 'final'].view(95)
```

If you want a report on disk carrying these error bars, now is the moment to write it (see the [reports guide](Reports)). This page is about the numbers themselves, so we move on.

The `pygsti.report.reportables` module described in the [model-analysis metrics guide](Metrics) is the other half of this. Wrap any function of a model in a `ModelFunction` and pyGSTi will propagate the confidence region through it, giving you an error bar on whatever you asked for.

Start with the simplest case: error bars on a process matrix. The `ModelFunction` only has to return the operation, but it has to return a *copy* of it. `to_dense()` hands back a view of the model's internal storage, and the finite-difference machinery perturbs that same model in place; without the copy the base value moves along with the perturbed one, every derivative comes out exactly zero, and so does every error bar.

```{code-cell} ipython3
final_model = results.estimates['CPTPLND'].models['stdgaugeopt'].copy()
```

```{code-cell} ipython3
def get_op(model, lbl):
    return model[lbl].to_dense().copy()
get_op_modelfn = modelfn.modelfn_factory(get_op)
```

```{code-cell} ipython3
rptbl.evaluate(get_op_modelfn(final_model, ("Gxpi2", 0)), crf_view)
```

```{code-cell} ipython3
rptbl.evaluate(get_op_modelfn(final_model, ("Gypi2", 0)), crf_view)
```

Nothing restricts you to raw matrix elements. The wrapped function can compute any derived quantity, including the other reportables. Note the calling convention: the model you want the quantity evaluated on must be the first argument. This one needs no copy, because it reduces the dense arrays to a scalar before returning.

```{code-cell} ipython3
def ddist(model, ideal_model, lbl, basis):
    return rptbl.half_diamond_norm(model[lbl].to_dense(), ideal_model[lbl].to_dense(), basis)
ddist_modelfn = modelfn.modelfn_factory(ddist)
```

```{code-cell} ipython3
rptbl.evaluate(ddist_modelfn(final_model, target_model, ("Gxpi2", 0), 'pp'), crf_view)
```

```{code-cell} ipython3
rptbl.evaluate(ddist_modelfn(final_model, target_model, ("Gypi2", 0), 'pp'), crf_view)
```

## Bootstrapped error bars

The bootstrap route generates a batch of surrogate datasets, runs GST on each, and reads the error bars off the spread of the resulting models. `pygsti.drivers.create_bootstrap_models` does the generating and fitting; it needs the fiducials, germs and max-lengths spelled out again because it builds its own circuit lists.

Two caveats before you start. The summarizing helpers used below, `_to_mean_model` and `_to_std_model`, are private, so treat them as unstable across versions; `gauge_optimize_models` itself is public API. And the resample count here is ten, which is well below what you would want for error bars you intend to quote. It keeps this page fast, and the scatter in the last figure shows what it costs.

```{code-cell} ipython3
estimated_model = results.estimates['full TP'].models['stdgaugeopt']
num_boot_models = 10
```

### Parametric bootstrap

Parametric bootstrapping resamples from the *fitted* model's probabilities. Pass `'parametric'` and supply `input_model`.

```{code-cell} ipython3
param_boot_models = pygsti.drivers.create_bootstrap_models(
                        num_boot_models, ds, 'parametric', prep_fiducials, meas_fiducials, germs, max_lengths,
                        input_model=estimated_model, target_model=target_model,
                        start_seed=0, return_data=False,
                        verbosity=1)
```

Each bootstrapped model lands in its own gauge, so they have to be brought into a common one before their spread means anything. `gauge_optimize_models` sweeps the SPAM weight used in the gauge optimization and keeps the value that minimizes a product of two averages of the ensemble's per-parameter standard deviation, one over a leading block of the parameter vector and one over the rest. The docstring calls these the average SPAM error and the average gate error with respect to a target; the code measures the spread of the bootstrap ensemble, and it splits the parameter vector at a hardcoded index that will not in general fall on the SPAM/gate boundary.

```{code-cell} ipython3
gauge_opt_pboot_models = pygsti.drivers.gauge_optimize_models(param_boot_models, estimated_model,
                                                              plot=False)  # plot=True raises NotImplementedError
```

```{code-cell} ipython3
print(gauge_opt_pboot_models[0])
```

The results collapse into two `Model` objects: a mean model and a standard-deviation model. The latter is the collection of error bars, one per model parameter. The former is worth comparing against the original estimate: if the mean of the resamples sits far from the estimate relative to the error bars, the bootstrap is telling you something is off. Be careful how hard you lean on that check at this resample count, though, since the mean of ten draws carries roughly a third of an error bar in noise of its own. The vectors printed in this section and the next also carry far more digits than ten resamples justify, so read the leading digit or two and no further.

```{code-cell} ipython3
pboot_mean = pygsti.drivers.bootstrap._to_mean_model(gauge_opt_pboot_models, estimated_model)
pboot_std = pygsti.drivers.bootstrap._to_std_model(gauge_opt_pboot_models, estimated_model)

print("Largest |mean - estimate| over parameters:",
      np.max(np.abs(pboot_mean.to_vector() - estimated_model.to_vector())))
print("Largest bootstrapped error bar:          ",
      np.max(pboot_std.to_vector()), end='\n\n')

print("Parametric bootstrapped error bars, with", num_boot_models, "resamples\n")
print("Error in rho vec:")
print(pboot_std['rho0'].to_vector(), end='\n\n')
print("Error in effect vecs:")
print(pboot_std['Mdefault'].to_vector(), end='\n\n')
print("Error in Gxpi2:")
print(pboot_std['Gxpi2', 0].to_vector(), end='\n\n')
print("Error in Gypi2:")
print(pboot_std['Gypi2', 0].to_vector())
```

### Non-parametric bootstrap

Non-parametric bootstrapping resamples from the observed count frequencies instead of from a fitted model, so there is no `input_model` to pass. Everything downstream is identical.

`_to_std_model` defaults to `ddof=1`, a sample standard deviation. That is not `numpy.std`'s own default, which is `ddof=0`, the population convention.

```{code-cell} ipython3
nonparam_boot_models = pygsti.drivers.create_bootstrap_models(
                          num_boot_models, ds, 'nonparametric', prep_fiducials, meas_fiducials, germs, max_lengths,
                          target_model=target_model, start_seed=0, return_data=False, verbosity=1)
```

```{code-cell} ipython3
gauge_opt_npboot_models = pygsti.drivers.gauge_optimize_models(nonparam_boot_models, estimated_model,
                                                               plot=False)
```

```{code-cell} ipython3
npboot_mean = pygsti.drivers.bootstrap._to_mean_model(gauge_opt_npboot_models, estimated_model)
npboot_std = pygsti.drivers.bootstrap._to_std_model(gauge_opt_npboot_models, estimated_model)

print("Largest |mean - estimate| over parameters:",
      np.max(np.abs(npboot_mean.to_vector() - estimated_model.to_vector())))
print("Largest bootstrapped error bar:          ",
      np.max(npboot_std.to_vector()), end='\n\n')

print("Non-parametric bootstrapped error bars, with", num_boot_models, "resamples\n")
print("Error in rho vec:")
print(npboot_std['rho0'].to_vector(), end='\n\n')
print("Error in effect vecs:")
print(npboot_std['Mdefault'].to_vector(), end='\n\n')
print("Error in Gxpi2:")
print(npboot_std['Gxpi2', 0].to_vector(), end='\n\n')
print("Error in Gypi2:")
print(npboot_std['Gypi2', 0].to_vector())
```

### Comparing the two

Scatter the two standard-deviation vectors against each other. Points on the dashed diagonal are parameters where the two bootstraps agree. With ten resamples the scatter is substantial, and most of it is sampling noise in the error bars themselves rather than disagreement between the methods. Ten resamples is enough to see that the two routes are in the same ballpark and not enough to separate them.

```{code-cell} ipython3
plt.loglog(npboot_std.to_vector(), pboot_std.to_vector(), '.')
plt.loglog(np.logspace(-4, -2, 10), np.logspace(-4, -2, 10), '--')
plt.xlabel('Non-parametric')
plt.ylabel('Parametric')
plt.xlim((1e-4, 1e-2)); plt.ylim((1e-4, 1e-2))
plt.title('Parametric vs. non-parametric bootstrap error bars')
plt.show()
```

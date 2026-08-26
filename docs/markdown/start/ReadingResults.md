---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.19.3
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Reading your results

A GST run ends by handing you two things: a Python object that holds the estimated models, and an HTML report that holds a few hundred numbers computed from them. Neither is self-explanatory the first time you see it. This page walks through both in the order you meet them, and says which number to look at before any of the others.

We need something to read, so start by running a small GST. `max_max_length=4` caps the germ-power length $L$ at 4, not the circuit depth: fiducials are prepended and appended, so the longest circuit here is actually 10 layers. That is far too shallow for a real characterization, but it finishes in a couple of seconds.

```{code-cell} ipython3
import pygsti
from pygsti.modelpacks import smq1Q_XYI

edesign = smq1Q_XYI.create_gst_experiment_design(max_max_length=4)

# Stand-in for real data: simulate from a slightly depolarized version of the target.
noisy_model = smq1Q_XYI.target_model().depolarize(op_noise=0.01, spam_noise=0.001)
ds = pygsti.data.simulate_data(noisy_model, edesign.all_circuits_needing_data,
                               num_samples=1000, sample_error='binomial', seed=1234)

data = pygsti.protocols.ProtocolData(edesign, ds)
results = pygsti.protocols.StandardGST('full TP', verbosity=0).run(data)
type(results)
```

## What `.run()` hands back

Every protocol's `.run()` returns a results object, and for the GST protocols that object is a `ModelEstimateResults`. It is a container, not a number. It holds the data it was fit to, the circuit lists that data came from, and one or more `Estimate` objects, each of which is a separate fit of the same data.

The object describes itself when you print it, which is the fastest way to find your bearings:

```{code-cell} ipython3
print(results)
```

The `.estimates` dictionary is the part you care about. Its keys depend on which protocol you ran, and this catches people out. `StandardGST` names each estimate after the mode that produced it, so the run above gives a single estimate keyed `'full TP'`; had you passed the default `modes=('full TP', 'CPTPLND', 'Target')` you would have three, one per mode. Note that `'Target'` is not a parameterization: it means "use the ideal gates as the estimate", which is there so a report can show your fit beside the thing it was aiming at. The lower-level `GateSetTomography` protocol instead names its single estimate after the protocol itself, so the key is `'GateSetTomography'`. Do not guess: look.

```{code-cell} ipython3
list(results.estimates.keys())
```

## Reaching the estimated model

An `Estimate` holds a dictionary of `Model` objects, not one model. The distinction matters, because most of the entries are not the one you want.

```{code-cell} ipython3
est = results.estimates['full TP']
print("models:      ", list(est.models.keys()))
print("gauge opts:  ", list(est.goparameters.keys()))
```

`'target'` is the ideal model you fit against. `'seed'` is the starting point the optimizer was handed. The `'iteration N estimate'` entries are the fits to progressively longer circuit subsets, and `'final iteration estimate'` is the last of them: the raw maximum-likelihood fit, in whatever gauge the optimizer happened to land in. Every remaining key is a *gauge-optimized* version of that final fit, and the keys of `.goparameters` tell you which ones those are. GST performs one gauge optimization by default and labels it `'stdgaugeopt'`.

**The model you want is almost always the gauge-optimized one.** Read `'final iteration estimate'` only when you specifically want the un-gauge-fixed fit, for instance because you intend to gauge-optimize it yourself against something other than the target.

```{code-cell} ipython3
mdl = est.models['stdgaugeopt']
target = est.models['target']

print("gate labels: ", list(mdl.operations.keys()))
print("prep labels: ", list(mdl.preps.keys()))
print("POVM labels: ", list(mdl.povms.keys()))
```

From there it is an ordinary pyGSTi `Model`, and you can compute whatever you like from it:

```{code-cell} ipython3
gx = ('Gxpi2', 0)
infid = pygsti.tools.entanglement_infidelity(mdl.operations[gx], target.operations[gx], target.basis)
print(f"entanglement infidelity of {gx}: {infid:.5f}")
```

## Building the report

The report is where GST results become readable, and one function makes it. `pygsti.report.construct_standard_report` takes the results object (or a dictionary of them, if you want several runs side by side in one document) and returns a `Report`. The `Report` itself is cheap; the work happens when you ask it to render.

```{code-cell} ipython3
report = pygsti.report.construct_standard_report(results, title="Reading Results Example", verbosity=0)
report.write_html("../../tutorial_files/reading_results_report", connected=True, verbosity=0)
```

Served with these docs: <a href="../../reports/reading_results_report.html">reading_results_report</a>.

`write_html` writes a *directory*, not a file. Open `main.html` inside it in a browser and you have the report. The `connected=True` above keeps that directory down to the single `main.html`, at the cost of needing a network connection to view it; see [report generation](../guides/analysis/Reports) for the trade-off. Pass `auto_open=True` to have it opened for you when rendering finishes, or `single_file=True` to get one self-contained HTML document instead of a directory (convenient for emailing, slow to load for large reports). Rendering time scales with the size of your experiment design, so a real two-qubit run takes minutes rather than the couple of seconds this toy example takes; the `brevity` argument (0 through 4) drops progressively more detail if you want it faster.

Error bars are the one thing the report will not compute for you. Passing `confidence_level=95` asks for them, but the underlying Hessian is expensive and is never computed as a side effect of report generation, so you must call `results.add_hessians()` first. Skip that step and `construct_standard_report` warns you and renders the tables bare. See [error bars](../guides/analysis/ErrorBars) for what the Hessian buys you and what it costs.

## What is in the report

The sidebar groups the report into five stacks of tabs, and they are ordered by how much you should trust them.

**Summary** is the landing tab and holds four figures: a bar plot of model violation for each estimate in the report, a bar plot of model violation per GST iteration, a histogram of per-circuit model violation, and a table comparing every estimated gate to its target under several error metrics.

**Model violation** expands the first three of those. Its overview tab carries a table breaking the fit down by iteration, with $N_\sigma$, degrees of freedom and a crude one-to-five-star rating per row, plus a scatter plot of per-circuit violation against circuit length. The per-sequence detail tab gives the color box plot, which is how you find *which* circuits the model fails to predict: one colored cell per circuit, arranged into blocks by germ (down the rows) and germ power (across the columns), with one pixel per fiducial pair inside each block. Gray cells are consistent with the model and red cells are not.

**Gauge invariant error metrics** holds the quantities that do not depend on how the gauge was fixed: predicted Clifford and direct RB error rates, the spectrum of the Gram matrix, spectral (eigenvalue-based) distances between estimated and target gates, and gauge-robust intrinsic and relational error metrics.

**Gauge dependent error metrics** holds the familiar ones: SPAM error metrics, per-gate infidelity and diamond distance, per-germ metrics, raw process matrices, gate decompositions into rotation axes and angles, and error generators. These are the numbers most people came for, and they are the ones the next section warns you about.

**For reference** records what went in (target model, fiducials, germs, dataset summary) and what pyGSTi and its dependencies were at the time. The **Help** tab at the bottom explains the report's own layout, which is worth a click on your first report.

## The number to look at first

Look at $N_\sigma$ before you look at anything else.

It is the top figure on the Summary tab, and it answers a question that comes logically before every other number in the report: does the model you fit actually describe your data? $N_\sigma$ counts how many standard deviations the observed $2\Delta\log\mathcal{L}$ sits above what $\chi^2$ theory predicts for a model that fits. You can also get it straight from the estimate:

```{code-cell} ipython3
est.misfit_sigma()
```

Small means the fit is consistent with statistical fluctuation and the rest of the report is worth reading. Large means *model violation*: your data contains structure a fixed set of gates cannot express, usually drift, context dependence or crosstalk. Model violation does not invalidate the estimate, but it does degrade it, and the more of it you see the less the fidelities and diamond distances on the gauge-dependent tabs mean. Note that $N_\sigma$ grows with the number of shots, so a well-sampled experiment on a very slightly non-Markovian device will report a large value; read it alongside the per-circuit plots rather than as a pass/fail. The [key concepts](KeyConcepts) page states the statistics behind it.

Only once $N_\sigma$ is small enough to live with should you go to the gate error metrics table at the bottom of the Summary tab. Entanglement infidelity and half diamond distance are the two columns most people quote. They agree for purely stochastic errors and diverge sharply when coherent errors are present, which is itself diagnostic: a diamond distance much larger than the infidelity says your errors are coherent.

## The gauge caveat

Infidelity and diamond distance are gauge-variant, which means they are not determined by your data alone. GST can only estimate a gate set up to gauge, so pyGSTi picks a representative of the gauge orbit (the one closest to your target model, by whatever weighting the gauge optimization used) and reports metrics against that. Change the gauge optimization weights and the reported fidelity of an individual gate moves, even though every predicted circuit probability is identical. The consequence is practical: a number from the gauge-dependent error metrics tabs is a statement about your device *and* about a gauge-fixing choice, so report the gauge optimization alongside it, prefer the gauge-invariant tabs when they answer your question, and treat a surprising fidelity as possible evidence that the gauge optimization went somewhere strange before you treat it as evidence about your hardware. See [gauge optimization](../guides/analysis/GaugeFreedom) for how to control the choice, and [key concepts](KeyConcepts) for why the freedom exists at all.

## Where to go next

The [results object guide](../guides/analysis/Results) covers the parts of `ModelEstimateResults` this page skipped: adding your own gauge optimizations after the fact, combining estimates from separate runs into one results object, and confidence region factories. The [report generation guide](../guides/analysis/Reports) covers multi-estimate and multi-dataset reports, PDF and notebook output, and reusing a `Workspace` across reports. [Error bars](../guides/analysis/ErrorBars) covers the Hessian machinery.

If your $N_\sigma$ came back large, that is the thread to pull on next, and the per-sequence color box plot is where to pull it: the pattern of which germs and which circuit lengths misfit usually names the physical cause.

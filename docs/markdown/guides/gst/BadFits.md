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

# When the fit is bad

A two-qubit GST fit that reports $N_\sigma = 300$ has not failed. It has told you something, and the something is usually that your device does not behave like a fixed set of gates. That is a physics result, not a software error, and the job now is to find out which physics it is.

This page assumes you already know how to read the number: [judging the fit](JudgingTheFit) covers $N_\sigma$, per-circuit goodness-of-fit, and what counts as large. What follows is what to do next, in the order that gets you the answer for the least effort.

## What model violation actually means

GST fits a *model*: a fixed set of superoperators plus a state preparation and a measurement, composed in the order the circuit specifies. That model asserts something strong. Every occurrence of `Gxpi2` in every circuit, at every depth, at every point in the experiment, is the same operation. Nothing in the model can depend on what came before, on the wall-clock time, or on what a neighboring qubit was doing.

Real devices violate that assertion constantly. Calibrations drift over the hours a two-qubit GST experiment takes, so the gate at the end of the run is not the gate at the start. A gate preceded by a long idle behaves differently from the same gate preceded by another gate, because of heating, leakage population, or amplifier settling. Simultaneous operations on spectator qubits shift the frequency of the qubits you are characterizing. Each of these makes the data a *mixture* of what several different models predict, and a mixture of two Markovian predictions is generally not the prediction of any single Markovian model.

Note that $N_\sigma$ measures statistical significance, not size. It is computed as $(2\Delta\log\mathcal{L} - k)/\sqrt{2k}$, where $k$ is the difference between the number of independent parameters in the data and in the model, so it grows roughly linearly in the number of counts per circuit for a fixed amount of model error. A very well-sampled experiment on a very slightly non-Markovian device will report a huge $N_\sigma$. That is why the first question is never "is the violation significant" but "how big is it, and where does it live".

## Work through the diagnostics in order

The order below is cheapest-first, and most bad fits are resolved before you reach the end of it.

Start by ruling out the boring failures. Confirm that the fit converged, that the target model you passed in is the one you meant (a `set_all_parameterizations` call that silently restricted your SPAM will produce violation that has nothing to do with the device), and that the data actually matches the experiment design, with no circuits missing or mislabeled. A surprising fraction of dramatic bad fits are bookkeeping.

Then look at *where* the violation sits, circuit by circuit. This is the single most informative diagnostic and it costs nothing beyond a report you were going to generate anyway. Violation concentrated in a handful of circuits with no shared structure means outliers: a few corrupted measurements, a bad batch, a cosmic ray. Violation that grows with circuit depth means the error is being amplified the way a coherent or slowly-varying error would be. Violation confined to circuits containing one germ, or one gate, points at that operation. Violation that is roughly uniform across all circuits including the shortest ones usually means SPAM.

Third, ask whether the device was stable. If your data is timestamped, run a stability analysis; see [drift characterization](../drift/DriftCharacterization). If you took the same circuits twice, or split one long run into halves, compare the two datasets directly with `pygsti.data.DataComparator`; see [comparing datasets](../drift/ComparingDataSets). Drift is the most common cause of large model violation in a multi-hour two-qubit experiment, and it is the easiest to confirm.

Fourth, ask whether your parameterization is too small. Refit with a strictly more expressive model and see how much of the violation survives. This is discussed at length below, because the answer determines whether you are looking at a modeling problem or a device problem.

Only then reach for the bad-fit machinery. Wildcard budgets quantify a violation you have already characterized; they are not a way of finding out what it is.

## A worked example

The example below manufactures a violation that no gate set can absorb. Two data-generating models differ only in the depolarization rate of `Gxpi2`, and half the counts for every circuit come from each. That is what a slow drift in one gate's fidelity looks like after you aggregate the run: the observed frequency for each circuit is an average of two exponential decays, and a single exponential cannot reproduce it.

```{code-cell} ipython3
import numpy as np
import pygsti
from pygsti.modelpacks import smq1Q_XY

pspec = smq1Q_XY.processor_spec()
target = smq1Q_XY.target_model()
edesign = smq1Q_XY.create_gst_experiment_design(8)
circuits = edesign.all_circuits_needing_data

def drifted(px):
    return pygsti.models.create_crosstalk_free_model(
        pspec, depolarization_strengths={('Gxpi2', 0): px, ('Gypi2', 0): 0.005})

ds_early = pygsti.data.simulate_data(drifted(0.005), circuits, num_samples=2500, seed=1)
ds_late = pygsti.data.simulate_data(drifted(0.20), circuits, num_samples=2500, seed=2)

ds = ds_early.copy_nonstatic()
ds.add_counts_from_dataset(ds_late)   # sums the counts circuit by circuit
ds.done_adding_data()

data = pygsti.protocols.ProtocolData(edesign, ds)
print(len(circuits), "circuits,", int(ds[circuits[0]].total), "counts each")
```

Now fit it with a CPTP-constrained Lindblad model, which is about as expressive as a physically sensible single-qubit gate set gets.

```{code-cell} ipython3
mdl = target.copy()
mdl.set_all_parameterizations("CPTPLND")

proto = pygsti.protocols.GateSetTomography(mdl, gaugeopt_suite=None, verbosity=0)
results = proto.run(data, disable_checkpointing=True)
est = results.estimates['GateSetTomography']

print("N_sigma = %.1f" % est.misfit_sigma())
```

That is not a marginal fit. Before doing anything else, look at how the misfit is distributed over the experiment design's max-length blocks. A two-outcome circuit contributes about 1 to $2\Delta\log\mathcal{L}$ when the model fits, and pyGSTi's bad-fit machinery calls a single circuit inconsistent when its contribution exceeds the $0.025/N_{\rm circuits}$ tail of a one-degree-of-freedom $\chi^2$: a Bonferroni correction, so that seeing any circuit over threshold is itself a 2.5%-level event.

```{code-cell} ipython3
from scipy.stats import chi2

final = est.models['final iteration estimate']
percircuit = pygsti.tools.two_delta_logl_per_circuit(final, ds, circuits)
threshold = chi2.ppf(1 - 0.025 / len(circuits), 1)
index_of = {c: i for i, c in enumerate(circuits)}

print("per-circuit threshold = %.1f" % threshold)
seen = set()
for L, clist in zip(edesign.maxlengths, edesign.circuit_lists):
    new = [index_of[c] for c in clist if c not in seen]
    seen.update(clist)
    v = percircuit[new]
    print("L = %2d: %3d new circuits, median %4.2f, %2d over threshold, worst %6.1f"
          % (L, len(new), np.median(v), (v > threshold).sum(), v.max()))
```

The median stays near 1 in every block, so the typical circuit is fitted fine. What fails is a tail, and that tail is spread across all four max lengths rather than sitting in one place. Roughly a tenth of the design is individually inconsistent, on a threshold chosen so that seeing even one such circuit would be surprising. That pattern (many circuits, all depths, median unaffected) is what a structurally unmodeled error looks like. A handful of corrupted measurements would look completely different: a few enormous values and nothing else.

Now ask which circuits they are.

```{code-cell} ipython3
for i in np.argsort(percircuit)[-4:][::-1]:
    print("%7.1f   %s" % (percircuit[i], circuits[i].str))
```

Long repetitions of `Gxpi2` sit at the top, which is where the drift was injected. Note also that the empty circuit is right behind them even though both data-generating models had identical SPAM. Model violation does not stay local: to compromise between two incompatible descriptions of `Gxpi2`, the optimizer distorts parameters that had nothing wrong with them. Do not read a bad fit as evidence about the specific parameters it appears to implicate until you have checked the rest of the design.

## Wildcard error budgets

Once you know the fit is bad, you still have to report a number. The estimated gates are the best Markovian description of a device that is not Markovian, so their diamond norms and fidelities understate the real error: any figure of merit computed from the estimate inherits the fiction that the model was right. A wildcard budget is pyGSTi's answer to "how much unmodeled error would I have to admit for this fit to be credible".

The mechanics are a slack variable per operation. Each circuit gets an allowance $W$, and the fit is recomputed allowing the model's predicted outcome distribution for that circuit to move by up to $W$ in total variation distance toward the observed frequencies. A `PrimitiveOpsWildcardBudget` builds $W$ additively: one allowance per occurrence of each primitive operation in the circuit, plus a single uniform `'SPAM'` term charged once per circuit. pyGSTi then finds the smallest budget (smallest weighted $L_1$ norm of the per-operation allowances) that brings both the aggregate $2\Delta\log\mathcal{L}$ and every individual circuit's contribution inside their 2.5%-tail thresholds.

Request one by passing `badfit_options` to the protocol. Note that the wildcard search only runs when the fit is worse than `threshold`, which defaults to 2 standard deviations.

```{code-cell} ipython3
proto_wc = pygsti.protocols.GateSetTomography(
    mdl, gaugeopt_suite=None,
    badfit_options={'actions': ['wildcard'], 'wildcard_methods': ('barrier',)},
    verbosity=0)
results_wc = proto_wc.run(data, disable_checkpointing=True)
est_wc = results_wc.estimates['GateSetTomography']

from pygsti.objectivefns.wildcardbudget import WildcardBudget
budget = WildcardBudget.from_nice_serialization(est_wc.parameters['unmodeled_error'])
for label, (description, value) in budget.description.items():
    print("%-12s %-32s %.4f" % (label, description, value))
```

The budget lands on `Gxpi2`, which is where the drift was injected, and gives `Gypi2` nothing. That is the useful output: the budget is a per-operation attribution of unmodeled error, and it says which gate you should not trust the reported error rate for.

Because gate allowances accumulate with depth, the slack granted to a deep circuit is much larger than the per-instance number suggests.

```{code-cell} ipython3
deepest = max(circuits, key=lambda c: c.depth)
print("depth %d, circuit budget = %.4f" % (deepest.depth, budget.circuit_budget(deepest)))
print("N_sigma with wildcard = %.1f"
      % pygsti.tools.two_delta_logl_nsigma(final, ds, circuits, wildcard=budget))
print(est_wc.parameters['unmodeled_active_constraints'])
```

The wildcard-adjusted $N_\sigma$ comes out strongly *negative*, which is worth understanding rather than glossing over. The optimizer has to satisfy the aggregate constraint and every per-circuit constraint at once, and here only per-circuit constraints are active: one deep `Gxpi2` repetition pins the gate allowance and the empty circuit pins the SPAM allowance, and at that budget the aggregate likelihood is far better than it had to be. The `unmodeled_active_constraints` entry names those circuits, which is how you find out what actually set the number.

A few practical points about the budget machinery, all of which will bite you eventually.

The per-operation budget is not unique. Allowance can be shifted between operations that always appear together, so comparing the `Gxpi2` and `Gypi2` numbers across two different fits is not meaningful in the way comparing two diamond distances is. The `'wildcard1d'` action exists to fix this: it fits a single scale factor $\alpha$ multiplying a fixed reference vector (the diamond distance from each estimated operation to its target), which removes the ambiguity at the cost of assuming that noisier gates carry proportionally more unmodeled error. The model is described in [doi:10.1038/s41534-023-00764-y](https://doi.org/10.1038/s41534-023-00764-y). It is skipped for an estimate labeled `'Target'`, since there is no fitted model to take diamond distances from.

The optimization method matters for large problems. The default `wildcard_methods=('neldermead',)` is the safe choice and is slow; `'barrier'` is much faster and generally reliable, and is what you want on two qubits.

You can group operations onto shared budget parameters. Pass `wildcard_primitive_op_labels` as a dictionary from operation label to a 0-based parameter index, mapping every gate to `0` and `'SPAM'` to `1` to get one allowance for all gates and one for readout; [running GST](RunningGST) has a worked version. Two smaller traps in the same area: `wildcard_L1_weights` is documented as an array but is consumed as a dictionary keyed by operation label, and `wildcard_inadmissable_action` defaults to `'print'`, so a wildcard analysis that fails to find an admissible budget will say so in the log and let the rest of the analysis continue rather than raising.

Wildcard is not the only bad-fit action. The `'robust'`, `'Robust'`, `'robust+'` and `'Robust+'` actions rescale the data for poorly-fitting circuits instead, and unlike the wildcard actions (which annotate the existing estimate in place) they add a new estimate named `GateSetTomography.robust` and so on. Rescaling is defensible when you genuinely believe a few circuits are corrupted; it is not defensible as a way of making a drift problem go away. If outliers really are your problem, the TVD-based objective in [robust GST using TVD](../../advanced/specialist/RobustGST-TVD) is a cleaner treatment than post-hoc rescaling.

Reports pick the budget up automatically. If an estimate carries an `unmodeled_error` entry, `pygsti.report.construct_standard_report` adds the unmodeled-error tables and plots.

## Parameterization problem or device problem?

The most consequential fork in this whole process is whether the violation is your model's fault or the device's. A restricted parameterization that cannot express an error the device really has will produce large $N_\sigma$ that says nothing about non-Markovianity. The test is direct: refit with a strictly larger model class.

```{code-cell} ipython3
truth = pygsti.models.create_crosstalk_free_model(
    pspec,
    depolarization_strengths={('Gxpi2', 0): 0.01, ('Gypi2', 0): 0.01},
    lindblad_error_coeffs={('Gxpi2', 0): {('H', 'Z'): 0.02}})
ds2 = pygsti.data.simulate_data(truth, circuits, num_samples=2000, seed=7)
data2 = pygsti.protocols.ProtocolData(edesign, ds2)

for ptype in ("S", "H+S"):
    m = target.copy()
    m.set_all_parameterizations(ptype, prep_type="full TP", povm_type="full TP")
    r = pygsti.protocols.GateSetTomography(m, gaugeopt_suite=None, verbosity=0).run(
        data2, disable_checkpointing=True)
    print("%-4s N_sigma = %6.2f" % (ptype, r.estimates['GateSetTomography'].misfit_sigma()))
```

Here the data was generated with a coherent $Z$ error on `Gxpi2`. A stochastic-only parameterization cannot represent it and reports real violation; adding Hamiltonian parameters makes the violation vanish. Nothing was wrong with the device, and a wildcard budget computed on the `"S"` fit would have been a budget for the analyst's mistake.

The practical rule: enlarge the parameterization until either the fit becomes good or you run out of physically meaningful parameters to add. If `"CPTPLND"` (or `"full TP"`, if you are willing to give up complete positivity) still cannot fit the data, no fixed gate set can, and the violation is telling you about the device. If the fit becomes good, adopt the larger model, and be aware that it costs you circuits: more parameters means more of the design's degrees of freedom go into fitting rather than testing. See [model testing](../analysis/ModelTesting) for comparing candidate parameterizations without refitting from scratch, and [judging the fit](JudgingTheFit) for reading the resulting numbers.

The same logic extends past parameterization to the structure of the model itself. If a gate behaves differently depending on what precedes it, you do not have one gate, you have two, and the fix is to model them as two: give each context its own operation label, then rewrite the circuits with `pygsti.circuits.manipulate_circuits` so every occurrence carries the label matching its context. If the device changes over the run, the honest options are to make the run shorter, interleave the circuit order so drift averages rather than correlates with depth, or move to a time-resolved analysis.

## Where to go next

If you have not yet checked stability, do that before anything else on this page: [drift characterization](../drift/DriftCharacterization) will tell you in minutes whether the answer is "your calibration moved". If you have ruled out drift and the violation tracks which gates precede which, splitting the offending label in two and letting GST estimate both is the next thing to try. If a small number of circuits are doing all the damage, [robust GST using TVD](../../advanced/specialist/RobustGST-TVD) is the right tool.

And if none of that applies, you are in the position GST is actually for. You have a device whose behavior no fixed gate set reproduces, a quantitative bound on how far off it is, and an attribution of that gap to particular operations. The estimated error rates are then lower bounds with a stated uncertainty attached, which is a more useful thing to report than a clean fit you did not interrogate. [Key concepts](../../start/KeyConcepts) sets out why that framing, rather than a pass/fail on $N_\sigma$, is how GST results are meant to be read.

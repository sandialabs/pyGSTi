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

# Adding and retargeting gauge optimizations

GST fits a model to data, but the data cannot prefer one gauge over another. Gauge optimization is a separate post-processing step that picks a representative from the gauge orbit, and pyGSTi records each choice next to the estimate rather than overwriting it. A gauge optimization is therefore never a decision you're stuck with. You can add more of them to results you computed months ago, vary the weights, aim at a different target, and compare the outcomes side by side.

This page covers both knobs. The first is *how* the optimizer scores a gauge choice, set by the arguments you pass to `pygsti.gaugeopt_to_target`. The second is *what it aims at*, which does not have to be the ideal target gates and frequently should not be.

## Setup

Everything below works off a single pre-computed `ModelEstimateResults` object holding one `Estimate` labeled `"full TP"`. The data-generating model has amplitude damping on every gate plus a little SPAM depolarization. Amplitude damping is non-unital, which matters later: non-unital error is exactly the kind that gauge freedom likes to move around.

```{code-cell} ipython3
import numpy as np
import pygsti
from pygsti.modelpacks import smq1Q_XYI
```

```{code-cell} ipython3
def amplitude_damping(gamma):
    """The amplitude damping superoperator, in the normalized Pauli basis."""
    s = np.sqrt(1 - gamma)
    return np.array([[1, 0, 0, 0],
                     [0, s, 0, 0],
                     [0, 0, s, 0],
                     [gamma, 0, 0, 1 - gamma]])

target_model = smq1Q_XYI.target_model()
mdl_datagen = target_model.copy().depolarize(spam_noise=0.001)
for lbl in mdl_datagen.operations:
    mdl_datagen.operations[lbl] = amplitude_damping(0.1) @ target_model.operations[lbl].to_dense()
```

```{code-cell} ipython3
exp_design = smq1Q_XYI.create_gst_experiment_design(max_max_length=4)
ds = pygsti.data.simulate_data(mdl_datagen, exp_design.all_circuits_needing_data, num_samples=1000, seed=1234)
data = pygsti.protocols.ProtocolData(exp_design, ds)

gaugeopt_params = {'item_weights': {'gates': 1, 'spam': 1}}
gst = pygsti.protocols.StandardGST("full TP", gaugeopt_suite={'equal weights': gaugeopt_params})
results = gst.run(data)
results.write("../../../example_files/regaugeopt_example")
```

Nothing below depends on the write and read-back round trip, but it is the realistic case: you come back to results sitting on disk and want a gauge optimization you didn't think to ask for the first time.

```{code-cell} ipython3
my_results = pygsti.io.read_results_from_dir("../../../example_files/regaugeopt_example", name="StandardGST")
estimate = my_results.estimates['full TP']
print(list(estimate.goparameters.keys()))
```

Had you passed `gaugeopt_suite=None` to `StandardGST`, no gauge optimization would have run. The list would not be empty, though: you would see a single entry, `trivial_gauge_opt`, whose `goparameters` value is `None` and whose model in `estimate.models` is the final iteration estimate untouched. `StandardGST` inserts that placeholder so report generation has a name to look up. Starting there is perfectly reasonable: run the fit, then do every gauge optimization by hand.

## Adding a gauge optimization

`Estimate.add_gaugeoptimized` takes a dictionary of arguments for `pygsti.gaugeopt_to_target`. You leave out that function's two positional arguments, the model to optimize and the target, because the method fills them in from `estimate.models['final iteration estimate']` and `estimate.models['target']`. The `label` argument names the new entry in both `estimate.models` and `estimate.goparameters`. Omit it and you get `go0`, `go1`, and so on.

Here is a gauge choice that cares a thousand times more about matching the target gates than the target SPAM operations:

```{code-cell} ipython3
estimate.add_gaugeoptimized({'item_weights': {'gates': 1, 'spam': 0.001}}, label="Spam 1e-3")
mdl_gaugeopt = estimate.models['Spam 1e-3']

print(list(estimate.goparameters.keys()))
print(mdl_gaugeopt.frobeniusdist(estimate.models['target']))
```

### Storing an optimization you ran yourself

Pass a `model` and `add_gaugeoptimized` skips the optimization, storing what you hand it. Use this when the gauge optimization has to happen somewhere else: on a cluster, in an earlier session, or through a code path pyGSTi doesn't own.

```{code-cell} ipython3
mdl_unfixed = estimate.models['final iteration estimate']
mdl_gaugefixed = pygsti.gaugeopt_to_target(mdl_unfixed, estimate.models['target'],
                                           item_weights={'gates': 1, 'spam': 0.001})
estimate.add_gaugeoptimized({'note': "computed externally",
                             'item_weights': {'gates': 1, 'spam': 0.001}},
                            model=mdl_gaugefixed, label="Spam 1e-3 custom")
print(list(estimate.goparameters.keys()))
print(estimate.models['Spam 1e-3 custom'].frobeniusdist(estimate.models['Spam 1e-3']))
```

Watch what the first argument is doing in that call. With `model` supplied, the dictionary is stored verbatim and never checked against `gaugeopt_to_target`'s signature, so any keys whatsoever will be accepted. Put the real parameters in it anyway. It is the only record of how the model you stored was produced.

Read parameters back through `.goparameters`:

```{code-cell} ipython3
import pprint
pp = pprint.PrettyPrinter()
pp.pprint(dict(estimate.goparameters['Spam 1e-3']))
```

The `maxiter` and `return_all` entries you did not set are defaults `add_gaugeoptimized` filled in, and `_gaugeGroupEl` is an output: the gauge group element that was applied.

## Aiming at a non-ideal target

Gauge optimization defaults to the ideal target gates and SPAM operations, which is convenient because you already had to specify them for GST. It is often the wrong choice. Gauge transformations can slosh error between the SPAM operations and the non-unital parts of the gates, so an optimizer told to match unital targets will buy agreement on the gates by moving error onto SPAM. What comes back is an estimate reporting nearly unital gates and visibly broken SPAM: a true statement about the gauge orbit, a useless one about the device. Weighting gates above SPAM makes the trade more lopsided, but lopsided weights are not required to see it, and the comparison below uses equal weights on both sides.

Separating the ideal targets from the gauge-optimization target lets you tell the optimizer what gates you *think* you have, known errors included. The estimate that comes back is usually much easier to interpret.

The mechanism is one dictionary key. `target_model` is the second positional argument of `gaugeopt_to_target`, so setting it in the parameter dictionary overrides the default. Suppose you know your gates suffer roughly 10% amplitude damping but you have no particular expectation about SPAM:

```{code-cell} ipython3
mdl_guess = target_model.copy()
for lbl in mdl_guess.operations:
    mdl_guess.operations[lbl] = amplitude_damping(0.1) @ target_model.operations[lbl].to_dense()

estimate.add_gaugeoptimized({**gaugeopt_params, 'target_model': mdl_guess}, label="guess target")
```

That loop is the same expression that built `mdl_datagen` back in the setup, so `mdl_guess` carries exactly the data-generating gates and differs from the truth only by the 0.001 SPAM depolarization. The repetition is deliberate, and it makes this a flattering guess; the paragraph after the numbers says what that costs the demonstration.

Everything except the target is held fixed against the `"equal weights"` optimization from the setup, so the only difference between the two is what they aim at. Both describe the same estimate and fit the data equally well. They disagree about where the error lives:

```{code-cell} ipython3
mdl_1 = estimate.models['equal weights']
mdl_2 = estimate.models['guess target']
print("ideal-target GO,  distance to ideal   = %.5f" % mdl_1.frobeniusdist(target_model))
print("mdl_guess GO,     distance to ideal   = %.5f" % mdl_2.frobeniusdist(target_model))
print("ideal-target GO,  distance to datagen = %.5f" % mdl_1.frobeniusdist(mdl_datagen))
print("mdl_guess GO,     distance to datagen = %.5f" % mdl_2.frobeniusdist(mdl_datagen))
print("distance between the two gauge choices= %.5f" % mdl_1.frobeniusdist(mdl_2))

print("\nPer-op difference from the data-generating model, ideal-target gauge opt")
print(mdl_1.strdiff(mdl_datagen))

print("\nPer-op difference from the data-generating model, mdl_guess gauge opt")
print(mdl_2.strdiff(mdl_datagen))
```

The ideal-target gauge optimization pushes a few percent of error onto `rho0` and the POVM effects, which is an artifact: the data-generating SPAM operations are almost perfect. Optimizing against `mdl_guess` leaves the SPAM operations close to where they belong and lands several times closer to the truth overall. Neither model is more correct than the other, they are the same physical estimate, but only one of them will lead you to the right conclusion when you read it off a report.

Be honest about what this can and cannot do. The guess here is not merely the right error *type*: it is the data-generating gates, to the last digit. The roughly fivefold improvement is what a perfect guess buys, which is the ceiling rather than a typical result. A guess with the right type and the wrong magnitude does less well, a guess pointed in the wrong direction moves the error somewhere else that is equally arbitrary, and a guess that differs from the ideal targets only in directions the gauge group cannot reach changes nothing at all.

## Setting the target when you run GST

You don't have to wait until after the fit. `GSTGaugeOptSuite` carries a `gaugeopt_target` that applies to every gauge optimization the protocol performs:

```{code-cell} ipython3
gaugeopt_suite = pygsti.protocols.GSTGaugeOptSuite(gaugeopt_suite_names=['stdgaugeopt'],
                                                   gaugeopt_target=mdl_guess)
results2 = pygsti.protocols.StandardGST("full TP", gaugeopt_suite).run(data)
```

The older driver `pygsti.run_long_sequence_gst` gets there differently: its `gauge_opt_params` dictionary becomes the argument dict for a single gauge optimization labeled `go0`, so a `target_model` key in it does the same job.

Parameter dictionaries a protocol builds are the same kind of dictionary you would write by hand, so you can lift one out of a finished estimate and replay it on another:

```{code-cell} ipython3
estimate.add_gaugeoptimized(results2.estimates['full TP'].goparameters['stdgaugeopt'],
                            label="replayed from results2")
mdl_2b = estimate.models['replayed from results2']
print(mdl_2b.frobeniusdist(results2.estimates['full TP'].models['stdgaugeopt']))
```

`"replayed from results2"` is not the same gauge choice as `"guess target"`, even though both aim at `mdl_guess`. The built-in `stdgaugeopt` suite is a three-stage optimization, each stage starting from the output of the last. The first pass weights gates and SPAM equally over the model's default gauge group, which for a "full TP" model is the TP gauge group rather than the full one. The second restricts to the unitary gauge group and nails down the gates at SPAM's expense. The third moves over the TP-SPAM gauge group only (the plain SPAM gauge group, for a `Full` model), weighting SPAM alone and adding a penalty on SPAM operations that stray outside the positive semidefinite cone; that stage is what puts the SPAM scaling back. The dictionaries used earlier on this page were all single-stage.

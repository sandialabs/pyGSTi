---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.17.3
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Gauge-Optimization With Non-Ideal Targets
Typically gauge optimizations are performed with respect to the set of ideal target gates and spam operations.  This is convenient, since you need to specify the ideal targets as points of comparison, but not always the best approach.  Particularly when you expect all or some of the gate estimates to either substantially differ from the ideal operations or differ, even by small amounts, in particular ways from the ideal operations, it can be hugely aid later interpretation to specify a non-ideal `Model` as the target for gauge-optimization.  By separating the "ideal targets" from the "gauge optimization targets", you're able to tell the gauge optimizer what gates you *think* you have, including any known errors.  This can result in a gauge-optimized estimate which is much more sensible and straightforward to interpet.

For example, gauge transformations can slosh error between the SPAM operations and the non-unital parts of gates.  If you know your gates are slightly non-unital you can include this information in the gauge-optimization-target (by specifying a `Model` which is slightly non-unital) and obtain a resulting estimate of low SPAM-error and slightly non-unital gates.  If you just used the ideal (unital) target gates, the gauge-optimizer, which is often setup to care more about matching gate than SPAM ops, could have sloshed all the error into the SPAM ops, resulting in a confusing estimate that indicates perfectly unital gates and horrible SPAM operations.

This example demonstrates how to separately specify the gauge-optimization-target `Model`.  There are two places where you might want to do this: 1) when calling `pygsti.run_long_sequence_gst`, to direct the gauge-optimization it performs, or 2) when calling `estimate.add_gaugeoptimized` to add a gauge-optimized version of an estimate after the main GST algorithms have been run.  

In both cases, a dictionary of gauge-optimization "parameters" (really just a dictionary of arguments for `pygsti.gaugeopt_to_target`) is required, and one simply needs to set the `targetModel` argument of `pygsti.gaugeopt_to_target` by specifying `targetModel` within the parameter dictionary.  We demonstrate this below.

First, we'll setup a standard GST analysis as usual except we'll create a `mdl_guess` model that is meant to be an educated guess at what we expect the estimate to be.  We'll gauge optimize to `mdl_guess` instead of the usual `target_model`:

```{code-cell} ipython3
import pygsti
from pygsti.modelpacks import smq1Q_XYI
```

```{code-cell} ipython3
#Generate some fake data (all usual stuff here)
exp_design = smq1Q_XYI.create_gst_experiment_design(max_max_length=4)
mdl_datagen  = smq1Q_XYI.target_model().depolarize(op_noise=0.1, spam_noise=0.001)
ds = pygsti.data.simulate_data(mdl_datagen, exp_design.all_circuits_needing_data, num_samples=1000, seed=1234)
data = pygsti.protocols.ProtocolData(exp_design, ds)
```

## Create a "guess" model that anticipates a more-depolarized Gx gate

```{code-cell} ipython3
mdl_guess = smq1Q_XYI.target_model()
mdl_guess[('Gxpi2',0)].depolarize(0.1)
```

## Run GST with and without the guess model

```{code-cell} ipython3
# GST with standard "ideal target" gauge optimization
results1 = pygsti.protocols.StandardGST("full TP").run(data)
```

```{code-cell} ipython3
# GST with our guess as the gauge optimization target
gaugeopt_suite = pygsti.protocols.GSTGaugeOptSuite(gaugeopt_suite_names=['stdgaugeopt'],
                                                  gaugeopt_target=mdl_guess)
results2 = pygsti.protocols.StandardGST("full TP", gaugeopt_suite).run(data)
```

## Comparisons
After running both the "ideal-target" and "mdl_guess-target" gauge optimizations, we can compare them with the ideal targets and the data-generating gates themselves.  We see that using `mdl_guess` results in a similar frobenius distance to the ideal targets, a slightly closer estimate to the data-generating model, and reflects our expectation that the `Gx` gate is slightly worse than the other gates.

```{code-cell} ipython3
target_model = smq1Q_XYI.target_model()
mdl_1 = results1.estimates['full TP'].models['stdgaugeopt']
mdl_2 = results2.estimates['full TP'].models['stdgaugeopt']
print("Diff between ideal and ideal-target-gauge-opt = ", mdl_1.frobeniusdist(target_model))
print("Diff between ideal and mdl_guess-gauge-opt = ", mdl_2.frobeniusdist(target_model))
print("Diff between data-gen and ideal-target-gauge-opt = ", mdl_1.frobeniusdist(mdl_datagen))
print("Diff between data-gen and mdl_guess-gauge-opt = ", mdl_2.frobeniusdist(mdl_datagen))
print("Diff between ideal-target-GO and mdl_guess-GO = ", mdl_1.frobeniusdist(mdl_2))

print("\nPer-op difference between ideal and ideal-target-GO")
print(mdl_1.strdiff(target_model))

print("\nPer-op difference between ideal and mdl_guess-GO")
print(mdl_2.strdiff(target_model))
```

## Adding a gauge optimization to existing `Results`
We can also include our `mdl_guess` as the `targetModel` when adding a new gauge-optimized result.  See other examples for more info on using `add_gaugeoptimized`.

```{code-cell} ipython3
results1.estimates['full TP'].add_gaugeoptimized(results2.estimates['full TP'].goparameters['stdgaugeopt'],
                                            label="using mdl_guess")
```

```{code-cell} ipython3
mdl_1b = results1.estimates['full TP'].models['using mdl_guess']
print(mdl_1b.frobeniusdist(mdl_2)) # gs1b is the same as gs2
```


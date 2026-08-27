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

# Running GST

The `pygsti` package provides multiple ways to use its core gate set tomography (GST) algorithms.  This tutorial will show you how to work with pyGSTi's GST protocol objects to perform GST in different ways with a minimal amount of effort.  In order to run the GST protocol there are 3 essential ingredients: 1) an "experiment design" specifying the structure of the GST circuits and how the data should be collected, 2) the outcome counts for the circuits specified by the experiment design, and 3) a desired, or "target", `Model`.  The [GST overview tutorial](../../start/FirstGST), gave an end-to-end example of how to construct a GST experiment design, run GST, and generate a report.  This tutorial focuses on the first and second steps in more detail; related information about circuit construction and report generation can be found in the [GST circuits tutorial](GSTCircuits) and [report generation tutorial](../analysis/Reports).

There are two different `Protocol` objects within pyGSTi for running GST:

- `GateSetTomography` - runs a single model optimization based on a *given* initial model that can have any parameterization you like.  This protocol can be run on any `GateSetTomographyDesign` experiment design, which only needs a target model (to describe what gates occur in the circuits) and a list of circuit lists to specify the circuits used for each iteration of the model optimization.

- `StandardGST` - runs multiple model optimizations based on an `ExplicitOpModel` target model by parameterizing this model in different ways.  The target model is expected to be a part of the experiment design, and only `StandardGSTDesign`-type experiment designs are allowed since the usual germs-and-fiducials structure of the GST circuits is expected.

Overall, the `GateSetTomography` protocol is more flexible than the `StandardGST` protocol, but requires a little more work to get going because its inputs are more complicated.  Both protocols return a `ModelEstimateResults` object when they are run.

```{code-cell} ipython3
import pygsti
import os
```

## Setup
In the [DataSet tutorial](../workflow/DataSets) we simulate the circuits required by a GST experiment design and save the results.  In this tutorial, we'll be analyzing that data.  This illustrates a typical workflow where at some earlier time you setup an experiment (a "GST experiment" in this case) and save the experiment design to disk and at some later time (after the data has been collected) you want to analyze it.  Now *is* that later time, and we start by reading the data we've collected.

```{code-cell} ipython3
data_dir = "../../../tutorial_files/Example_GST_Data"

if not os.path.isdir(data_dir):
    # Not run the DataSet tutorial? Generate the same data here so this page
    # stands on its own (same model pack, max length and seed it uses).
    from pygsti.modelpacks import smq1Q_XYI
    noisy_model = smq1Q_XYI.target_model().depolarize(op_noise=0.1)
    edesign = smq1Q_XYI.create_gst_experiment_design(max_max_length=128)
    ds = pygsti.data.simulate_data(noisy_model, edesign.all_circuits_needing_data,
                                   num_samples=1000, sample_error='binomial', seed=100)
    pygsti.protocols.ProtocolData(edesign, ds).write(data_dir)

data = pygsti.io.read_data_from_dir(data_dir)
```

## `GateSetTomography`
This protocol performs a single model optimization, and so computes a **single GST estimate** given a `DataSet`, a target `Model`, and other parameters.  (The returned `ModelEstimateResults` object may sometimes contain multiple related estimates in certain cases, but in these cases all the estimates are closely related.)  The experiment design provides all of the information about the GST circuits, in this case a *standard*  (*prep_fiducial + germ^power + meas_fiducial*) set, so the only thing needed by the protocol is an initial `Model` to optimize.  Thus, the `GateSetTomography` protocol is essentially just a model optimizer that you give an initial point.  Importantly, this initial point (a `Model`) also specifies the *parameterization*, i.e. the space of parameters that are optimized over.

Minimally, when using `GateSetTomography` you should set the parameterization of the initial model.  This can be viewed as setting the constraints on the optimization.  For instance, when the gates in the model are parameterized as trace-preserving (TP) maps, the optimization will be constrained to trying gate sets with TP gates (because every set of parameters corresponds to a set of TP gates).  In the cell below, we constrain the optimization to TP gate sets by using `.target_model("full TP")`, which returns a version of the target model where all the gates are TP-parameterized, the state preparation has trace = 1, and the POVM effects always add to the identity.  This could also be done by calling `set_all_parameterizations("full TP")` on the fully-parameterized target model returned by `.target_model()`.  See the [tutorial on explicit models](../models/Models) for more information on setting a model's parameterization.

The fits on this page pass `disable_checkpointing=True` to skip the per-iteration checkpoint files GST writes by default; checkpointing, and warmstarting a fit from those files, is covered on the [Parallelism](Parallelism) page.

```{code-cell} ipython3
from pygsti.modelpacks import smq1Q_XYI
target_model_TP = smq1Q_XYI.target_model("full TP")
proto = pygsti.protocols.GateSetTomography(target_model_TP)
results_TP = proto.run(data, disable_checkpointing=True)
```

A summary of what's inside a Results object is obtained by printing it
(for more examples of how to use a Results object, see the [Results tutorial](../analysis/Results)).

```{code-cell} ipython3
print(results_TP)
```

### Gauge optimization parameters
The `gaugeopt_suite` argument specifies a set of gauge optimizations to be performed on the final GST estimate.  It is a dictionary whose keys are gauge-optimization names (these can be whatever you want) and whose values are dictionaries of arguments ultimately to be passed to the `gaugeopt_to_target` function (which provides full documentation).  (For example, by specifying `item_weights` we can set the ratio of the state preparation and measurement (SPAM) weighting to the gate weighting when performing a gauge optimization.)  In lieu of a dictionary of `gaugeopt_to_target` arguments, the elements of `gaugeopt_suite` may also be strings which name a built-in set of gauge optimizations (e.g. `"stdgaugeopt"` is the name of the standard gauge optimization).

If `gaugeopt_suite` is set to a string, this is the same as passing a dictionary with a single key-value pair where both key and value are equal to the string.  Thus, the default `"stdgaugeopt"` is equivalent to specifying the dictionary `{"stdgaugeopt": "stdgaugeopt"}`.

The example below performs a customized gauge-optimization where the gate parameters are weighted 1000 times more relative to the SPAM parameters.  Mathematically this corresponds to a multiplicative factor of 0.001 preceding the sum-of-squared-difference terms corresponding to SPAM elements in the model.   Typically it is good to weight the gates parameters more heavily since GST amplifies gate parameter errors via long operation sequences but cannot amplify SPAM parameter errors.  For more details on the arguments of `gaugeopt_to_target`, see the [low-level algorithms tutorial](../../advanced/extending/LowLevelGST).  For more information, see the [gauge optimization tutorial](../analysis/GaugeFreedom).

The cell below also illustrates how you can create a TP target model by calling `set_all_parameterizations` explicitly instead of using the equivalent and more condensed `.target_model("full TP")`.

```{code-cell} ipython3
target_model_TP2 = smq1Q_XYI.target_model() # a "fully parameterized" (unconstrained) model
target_model_TP2.set_all_parameterizations("full TP") # change parameterization to TP gates

proto = pygsti.protocols.GateSetTomography(
    target_model_TP2, name="GSTwithMyGO",
    gaugeopt_suite={'my_gauge_opt': {'item_weights': {'gates': 1.0, 'spam': 0.001}}}
    )
results_TP2 = proto.run(data, disable_checkpointing=True)
```

```{code-cell} ipython3
print(results_TP2.estimates['GSTwithMyGO'].goparameters.keys())  # names of all the gauge opts that were done
custom_gauge_opt_model = results_TP2.estimates['GSTwithMyGO'].models['my_gauge_opt']
```

### Running GST using a custom set of circuits
So far we've given the `GateSetTomography.run` method a "standard" experiment design containing circuits chosen to amplify all of a standard TP (or CPTP) model's parameters (see the `StandardGSTDesign` used in the [DataSet tutorial](../workflow/DataSets)).  A `GateSetTomography` protocol can also be run on the more general `GateSetTomographyDesign`, which accepts whatever circuit lists or structures you hand it — see the [experiment designs tutorial](../workflow/ExperimentDesigns) for how the two design classes relate.  In this example, we'll just generate a standard set of circuit structures, but with some of the sequences randomly dropped (see the [tutorial on GST circuit reduction](FewerCircuits)).

```{code-cell} ipython3
# Create the same sequences but drop 50% of them randomly for each repeated-germ block.
# and only go out to a max-length of 8
pspec = target_model_TP2.create_processor_spec() # ProcessorSpec based on the target model
orig_design = data.edesign  # the original StandardGSTDesign
custom_maxlengths = [1, 2, 4, 8]  # a subset of orig_design.maxlengths
circuit_structs = pygsti.circuits.create_lsgst_circuit_lists(
    target_model_TP2, orig_design.prep_fiducials, orig_design.meas_fiducials,
    orig_design.germs, custom_maxlengths, keep_fraction=0.5, keep_seed=2020)
reduced_exp_design = pygsti.protocols.GateSetTomographyDesign(pspec, circuit_structs)
reduced_data = pygsti.protocols.ProtocolData(reduced_exp_design, data.dataset)


proto = pygsti.protocols.GateSetTomography(target_model_TP2, name="GSTwithReducedData")
results_reduced = proto.run(reduced_data, disable_checkpointing=True)
```

One `GateSetTomography` argument not exercised here is `badfit_options`, which controls what happens when the optimized model still fails to fit the data — including *wildcard* error budgets, an amount of unmodeled error distributed over the operations to account for the gap.  [When the fit is bad](BadFits) covers these options and when reaching for them is the right move.

## `StandardGST`
The protocol embodies a standard *set* of GST protocols to be run on a set of data.  It essentially runs multiple `GateSetTomography` protocols on the given data which use different parameterizations of an `ExplicitOpModel`  (the `StandardGST` protocol doesn't work with other types of `Model` objects, e.g. *implicit* models, which don't implement `set_all_parameterizations`).  The `modes` argument is a list strings corresponding to the parameterization types that should be run (e.g. `["full TP","CPTPLND"]` will compute a Trace-Preserving estimate *and* a Completely-Positive & Trace-Preserving estimate). The currently available modes are:
 - `"full"`    : unconstrained gates (fully parameterized)
 - `"full TP"` : TP-constrained gates and state preparations (`"TP"` is accepted as a synonym)
 - `"CPTPLND"` : Lindbladian CPTP-constrained gates
 - `"H+S"`     : only Hamiltonian and Pauli-stochastic errors allowed (CPTP)
 - `"S"`       : only Pauli-stochastic errors allowed (CPTP)
 - `"Target"`  : use the target (ideal) gates as the estimate
 - any key in the `models_to_test` argument

The default is `("full TP", "CPTPLND", "Target")`.  `"CPTP"` still runs but raises a deprecation warning; use `"CPTPLND"` instead.  A mode that is neither `"Target"` nor a `models_to_test` key is handed to `set_all_parameterizations`, so other Lindblad parameterization names it accepts will also run, though they are off the path this protocol is set up for.

Gauge optimization is controlled by the `gaugeopt_suite` argument, just as in `GateSetTomography`.  Neither protocol takes a `gaugeopt_target` argument.  To gauge optimize toward something other than the experiment design's (typically ideal) target gates, build a `GSTGaugeOptSuite` with its own `gaugeopt_target=` and pass that in as the suite, which is what the last example on this page does.

On this data the CPTPLND optimization routinely stops at its default iteration cap, which the optimizer treats as convergence — the `WARNING: Treating result as *converged* after maximum iterations (100) were exceeded.` line in the output below is expected, not a sign the fit failed.

```{code-cell} ipython3
:tags: [output_scroll]

results_stdprac = pygsti.protocols.StandardGST(verbosity=4).run(data, disable_checkpointing=True)
```

```{code-cell} ipython3
print("Estimates: ", ", ".join(results_stdprac.estimates.keys()))
print("TP Estimate's gauge optimized models: ", ", ".join(results_stdprac.estimates["full TP"].goparameters.keys()))
```

Next, we'll perform the same analysis but with a **non-default standard suite of gauge optimizations** - this one toggles the SPAM penalty in addition to varying the spam weight (the default suite just varies the spam weight without any SPAM penalty).  See the [gauge optimization tutorial](../analysis/GaugeFreedom) for more details on gauge optmization "suites".

```{code-cell} ipython3
:tags: [output_scroll]

proto = pygsti.protocols.StandardGST(gaugeopt_suite="varySpam", name="StdGST_varySpam", verbosity=4)
results_stdprac_nondefaultgo = proto.run(data, disable_checkpointing=True)
```

```{code-cell} ipython3
print("Estimates: ", ", ".join(results_stdprac_nondefaultgo.estimates.keys()))
print("TP Estimate's gauge optimized models: ", ", ".join(results_stdprac_nondefaultgo.estimates["full TP"].goparameters.keys()))
```

Finally, we'll demonstrate how to specify a fully custom set of gauge optimization parameters and how to use a **separately-specified target model for gauge optimization**.  You can get a more intuitive gauge-optimized `Model` by placing as much expected noise as possible into the gauge-optimization target, as this essentially tells the algorithm "this is what I think the estimated model should look like".  If you just use the perfect or ideal model for this (the default), then the gauge optimizer may make tradeoffs which don't reflect the expected physics (remember, all gauge-equivalent models produce the same observables!).  For example, it may spread error across all your gate operations when you expect just the 2-qubit operations are noisy.

```{code-cell} ipython3
:tags: [output_scroll]

my_goparams = { 'item_weights': {'gates': 1.0, 'spam': 0.001} }
my_gaugeOptTarget = smq1Q_XYI.target_model().depolarize(
    op_noise=0.005, spam_noise=0.01) # a guess at what estimate should be
my_gaugeopt_suite = pygsti.protocols.GSTGaugeOptSuite(gaugeopt_argument_dicts={'myGO': my_goparams},
                                                      gaugeopt_target=my_gaugeOptTarget)

proto = pygsti.protocols.StandardGST(gaugeopt_suite=my_gaugeopt_suite,
                                     name="StdGST_myGO", verbosity=4)
results_stdprac_nondefaultgo = proto.run(data, disable_checkpointing=True)
```

```{code-cell} ipython3
print("Estimates: ", ", ".join(results_stdprac_nondefaultgo.estimates.keys()))
print("TP Estimate's gauge optimized models: ", ", ".join(results_stdprac_nondefaultgo.estimates["full TP"].goparameters.keys()))
```

To finish up, we'll write the results for processing in other tutorials.  We do this by calling `.write` on the results objects, optionally specifying the root directory under which the results should be written.  This is the *same* root directory that the experiment design and data are written to, as subdirectories beneath this directory separate these quantities.

Two remarks are in order:
1. When results are from running a protocol on data that was loaded with the `read_data_from_dir` method (see the beginning of this notebook), then knowledge of this directory is remembered and you don't need to give a directory to `write` (this is the case for all except `results_reduced`, which created a new experiment design containing less experiments).

2. Notice how the `name=` arguments given to protocols above are used as sub-directory names, e.g. under the "../../../tutorial_files/Example_GST_Data/results" parent directory.

```{code-cell} ipython3
results_TP.write()  # uses "../../../tutorial_files/Example_GST_Data" (where data was loaded from)
results_TP2.write() # ditto
results_stdprac.write() # ditto
results_reduced.write("../../../tutorial_files/Example_Reduced_GST_Data") # choose a different dir
```

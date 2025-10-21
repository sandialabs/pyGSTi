---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.17.3
kernelspec:
  display_name: pygsti
  language: python
  name: python3
---

# Different GST Protocols

The `pygsti` package provides multiple ways to use its core Gate Set Tomography (GST) algorithms.  This  tutorial will show you how to work with pyGSTi's GST protocol objects to perform GST in different ways with a minimial amount of effort.  In order to run the GST protocol there are 3 essential ingredients: 1) an "experiment design" specifying the structure of the GST circuits and how the data should be collected, 2) the outcome counts for the circuits specified by the experiment design, and 3) a desired, or "target", `Model`.  The [GST overview tutorial](Overview), gave an end-to-end example of how to construct a GST experiment design, run GST, and generate a report.  This tutorial focuses on the first and second steps in more detail; related information about circuit construction and report generation can be found in the [GST circuits tutorial](CircuitConstruction) and [report generation tutorial](../reporting/ReportGeneration).

There are two different `Protocol` objects within pyGSTi for running GST:

- `GateSetTomography` - runs a single model optimization based on a *given* initial model that can have any parameterization you like.  This protocol can be run on any `GateSetTomographyDesign` experiment design, which only needs a target model (to describe what gates occur in the circuits) and a list of circuit lists to specify the circuits used for each iteration of the model optimization.

- `StandardGST` - runs multiple model optimizations based on an `ExplicitOpModel` target model by parameterizing this model in different ways.  The target model is expected to be a part of the experiment design, and only `StandardGSTDesign`-type experiment designs are allowed since the usual germs-and-fiducials structure of the GST circuits is expected.

Overall, the `GateSetTomography` protocol is more flexible than the `StandardGST` protocol, but requires a little more work to get going because its inputs are more complicated.  Both protocols return a `ModelEstimateResults` object when they are run.

```{code-cell} ipython3
import pygsti
```

## Setup
In the [DataSet tutorial](../objects/DataSet) we simulate the circuits required by a GST experiment design and save the results.  In this tutorial, we'll be analyzing that data.  This illustrates a typical workflow where at some earlier time you setup an experiment (a "GST experiment in this case) and save the experiment design to disk and at some later time (after the data has been collected) you want to analyze it.  Now *is* that later time, and we start by reading the the data we've collected.

```{code-cell} ipython3
data = pygsti.io.read_data_from_dir("../../tutorial_files/Example_GST_Data")
```

## `GateSetTomography`
This protocol performs a single model optimization, and so computes a **single GST estimate** given a `DataSet`, a target `Model`, and other parameters.  (The returned `ModelEstimateResults` object may sometimes contain multiple related estimates in certain cases, but in these cases all the estimates are closely related.)  The experiment design provides all of the information about the GST circuits, in this case a *standard*  (*prep_fiducial + germ^power + meas_fiducial*) set, so the only thing needed by the protocol is an initial `Model` to optimize.  Thus, the `GateSetTomography` protocol is essentially just a model optimizer that you give an initial point.  Importantly, this initial point (a `Model`) also specifies the *parameterization*, i.e. the space of parameters that are optimized over.

Minimally, when using `GateSetTomography` you should set the parameterization of the initial model.  This can be viewed as setting the constraints on the optimization.  For instance, when the gates in the model are parameterized as trace-preserving (TP) maps, the optimization will be constrained to trying gate sets with TP gates (because every set of parameters corresponds to a set of TP gates).  In the cell below, we constrain the optimization to TP gate sets by using `.target_model("full TP")`, which returns a version of the target model where all the gates are TP-parameterized, the state preparation has trace = 1, and the POVM effects always add to the identity.  This could also be done by calling `set_all_parameterizations("TP")` on the fully-parameterized target model returned by `.target_model()`.  See the [tutorial on explicit models](../objects/ExplicitModel) for more information on setting a model's parameterization.

```{code-cell} ipython3
from pygsti.modelpacks import smq1Q_XYI
target_model_TP = smq1Q_XYI.target_model("full TP")
proto = pygsti.protocols.GateSetTomography(target_model_TP)
results_TP = proto.run(data)
```

A summary of what's inside a Results object is obtained by printing it
(for more examples of how to use a Results object, see the [Results tutorial](../objects/Results)).

```{code-cell} ipython3
print(results_TP)
```

### Gauge optimization parameters
The `gaugeopt_suite` argument specifies a set of gauge optimizations to be performed on the final GST estimate.  It is a dictionary whose keys are gauge-optimization names (these can be whatever you want) and whose values are dictionaries of arguments ultimately to be passed to the `gaugeopt_to_target` function (which provides full documentation).  (For example, by specifying `item_weights` we can set the ratio of the state preparation and measurement (SPAM) weighting to the gate weighting when performing a gauge optimization.)  In lieu of a dictionary of `gaugeopt_to_target` arguments, the elements of `gaugeopt_suite` may also be strings which name a built-in set of gauge optimizations (e.g. `"stdgaugeopt"` is the name of the standard gauge optimization).

If `gaugeopt_suite` is set to a string, this is the same as passing a dictionary with a single key-value pair where both key and value are equal to the string.  Thus, the default `"stdgaugeopt"` is equivalent to specifying the dictionary `{"stdgaugeopt": "stdgagueopt"}`.

The example below performs a customized gauge-optimization where the gate parameters are weighted 1000 times more relative to the SPAM parameters.  Mathematically this corresponds to a multiplicative factor of 0.001 preceding the sum-of-squared-difference terms corresponding to SPAM elements in the model.   Typically it is good to weight the gates parameters more heavily since GST amplifies gate parameter errors via long operation sequences but cannot amplify SPAM parameter errors.  For more details on the arguments of `gaugeopt_to_target`, see the previous tutorial on low-level algorithms.  For more infomation, see the [gauge optimization tutorial](../utilities/GaugeOpt).

The cell below also illustrates how you can create a TP target model by calling `set_all_parameterizations` explicitly instead of using the equivalent and more condensed `.target_model("TP")`.

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

### Wildcard parameters

TODO

```{code-cell} ipython3
proto = pygsti.protocols.GateSetTomography(
    target_model_TP, name="GSTwithPerGateWildcard",
    badfit_options={'actions': ['wildcard']}
    )

# Artifically unset threshold so that wildcard runs. YOU WOULD NOT DO THIS IN PRODUCTION RUNS
proto.badfit_options.threshold = None

results_pergate_wildcard = proto.run(data, disable_checkpointing=True)
```

```{code-cell} ipython3
# The wildcard can be retrieved by looking at unmodeled_error in the estimates
results_pergate_wildcard.estimates['GSTwithPerGateWildcard'].parameters['unmodeled_error']
```

Another common form of wildcard is to have one parameter for SPAM and one for all the other gates.

```{code-cell} ipython3
op_label_dict = {k:0 for k in target_model_TP.operations} # Assign all gates to value 0
op_label_dict['SPAM'] = 1 # Assign SPAM to value 1

proto = pygsti.protocols.GateSetTomography(
    target_model_TP, name="GSTwithPerGateWildcard",
    badfit_options={'actions': ['wildcard'], 'wildcard_primitive_op_labels': op_label_dict}
    )

# Artifically unset threshold so that wildcard runs. YOU WOULD NOT DO THIS IN PRODUCTION RUNS
proto.badfit_options.threshold = None

results_globalgate_wildcard = proto.run(data, disable_checkpointing=True)
```

Unfortunately both of these wildcard strategies have the same problem. They are not unique, i.e. it is possible to "slosh" wildcard strength from one parameter to another to get another valid wildcard solution. This makes it difficult to make any quantitative statements about relative wildcard strengths.

In order to avoid this, we have also introduced a 1D wildcard solution. This takes some reference weighting for the model operations and scales a single wildcard parameter ($\alpha$) up until the model fits the data. Since there is only one parameter, this does not have any of the ambiguity of the above wildcard strategies. Currently, the reference weighting used is the diamond distance from the noisy model to the target model, with the intuition that "noisier" operations are more likely to contribute to model violation.

```{code-cell} ipython3
proto = pygsti.protocols.GateSetTomography(
    target_model_TP, name="GSTwithPerGateWildcard",
    badfit_options={'actions': ['wildcard1d'], 'wildcard1d_reference': 'diamond distance'}
    )

# Artifically unset threshold so that wildcard runs. YOU WOULD NOT DO THIS IN PRODUCTION RUNS
proto.badfit_options.threshold = None

results_1d_wildcard = proto.run(data, disable_checkpointing=True)
```

### running GST using a custom set of circuits
So far we've giving the `GateSetTomography.run` method an "standard" experiment design containing circuits chosen to amplify all of a standard TP (or CPTP) model's parameters (see the `StandardGSTExpermentDesign` used in the [DataSet tutorial](../objects/DataSet)).  A `GateSetTomography` protocol can be run on more general experiment designs, namely those that specify the circuits to use as either a list of lists of `Circuit` objects or a list of or single `CircuitStructure` object(s).  A `CircuitStructure` is preferable as it allows the structured plotting of the sequences in report figures.  In this example, we'll just generate a standard set of circuit structures, but with some of the sequences randomly dropped (see the [tutorial on GST circuit reduction](FiducialPairReduction)).

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

## `StandardGST`
The protocol embodies a standard *set* of GST protocols to be run on a set of data.  It essentially runs multiple `GateSetTomography` protocols on the given data which use different parameterizations of an `ExplicitOpModel`  (the `StandardGST` protocol doesn't work with other types of `Model` objects, e.g. *implicit* models, which don't implement `set_all_parameterizations`).  The `modes` argument is a list strings corresponding to the parameterization types that should be run (e.g. `["full TP","CPTPLND"]` will compute a Trace-Preserving estimate *and* a Completely-Positive & Trace-Preserving estimate). The currently available modes are:
 - "full" : unconstrained gates (fully parameterized)                                                                 
 - "TP"   : TP-constrained gates and state preparations
 - "CPTP" : CPTP-constrained gates and TP-constrained state preparations               
 - "H+S"  : Only Hamiltonian and Pauli stochastic errors allowed (CPTP)                                             
 - "S"    : Only Pauli-stochastic errors allowed (CPTP)                                                           
 - "Target" : use the target (ideal) gates as the estimate     

Gauge optimization(s) are controlled by the `gaugeopt_suite` and `gaugeopt_target` arguments, jsut as in `GateSetTomography`.  The `gaugeopt_target` argument may be set to a `Model` that is used as the target for gauge optimization, overriding the (typically ideal) target gates given by within the experiment design.

```{code-cell} ipython3
results_stdprac = pygsti.protocols.StandardGST().run(data)
```

```{code-cell} ipython3
print("Estimates: ", ", ".join(results_stdprac.estimates.keys()))
print("TP Estimate's gauge optimized models: ", ", ".join(results_stdprac.estimates["full TP"].goparameters.keys()))
```

Next, we'll perform the same analysis but with a **non-default standard suite of gauge optimizations** - this one toggles the SPAM penalty in addition to varying the spam weight (the default suite just varies the spam weight without any SPAM penalty).  See the [gauge optimization tutorial](../utilities/GaugeOpt) for more details on gauge optmization "suites".

```{code-cell} ipython3
proto = pygsti.protocols.StandardGST(gaugeopt_suite="varySpam", name="StdGST_varySpam")
results_stdprac_nondefaultgo = proto.run(data)
```

```{code-cell} ipython3
print("Estimates: ", ", ".join(results_stdprac_nondefaultgo.estimates.keys()))
print("TP Estimate's gauge optimized models: ", ", ".join(results_stdprac_nondefaultgo.estimates["full TP"].goparameters.keys()))
```

Finally, we'll demonstrate how to specify a fully custom set of gauge optimization parameters and how to use a **separately-specified target model for gauge optimization**.  You can get a more intuitive gauge-optimized `Model` when by placing as much expected noise as possible into the gauge-optimization target, as this essentially tells the algorithm "this is what I think the estimated model should look like".  If you just use the perfect or ideal model for this (the default), then the gauge optimizer may make tradeoffs which don't reflect the expected physics (remember, all gauge-equivalent models product the same observables!).  For example, it may spread error across all your gate operations when you expect just the 2-qubit operations are noisy.

```{code-cell} ipython3
my_goparams = { 'item_weights': {'gates': 1.0, 'spam': 0.001} }
my_gaugeOptTarget = smq1Q_XYI.target_model().depolarize(
    op_noise=0.005, spam_noise=0.01) # a guess at what estimate should be
my_gaugeopt_suite = pygsti.protocols.GSTGaugeOptSuite(gaugeopt_argument_dicts={'myGO': my_goparams},
                                                      gaugeopt_target=my_gaugeOptTarget)

proto = pygsti.protocols.StandardGST(gaugeopt_suite=my_gaugeopt_suite,
                                     name="StdGST_myGO")
results_stdprac_nondefaultgo = proto.run(data)
```

```{code-cell} ipython3
print("Estimates: ", ", ".join(results_stdprac_nondefaultgo.estimates.keys()))
print("TP Estimate's gauge optimized models: ", ", ".join(results_stdprac_nondefaultgo.estimates["full TP"].goparameters.keys()))
```

To finish up, we'll write the results for processing in other tutorials.  We do this by calling `.write` on the results objects, optionally specifying the root diretory under which the results should be written.  This is the *same* root directory that the experiment design and data are written to, as subdirectories beneath this directory separate these quantities.

Two remarks are in order:
1. When results are from running a protocol on data that was loaded with the `load_data_from_dir` method (see the beginning of this notebook), then knowledge of this directory is remembered and you don't need to give a directory to `write` (this is the case for all except `results_reduced`, which created a new experiment design containing less experiments).

2. Notice how the `name=` arguments given to protocols above are used as sub-directory names, e.g. under the "tutorial_files/Example_GST_Data/results" parent directory.

```{code-cell} ipython3
results_TP.write()  # uses "../../tutorial_files/Example_GST_Data" (where data was loaded from)
results_TP2.write() # ditto
results_stdprac.write() # ditto
results_reduced.write("../../tutorial_files/Example_Reduced_GST_Data") # choose a different dir
```

While it is also possible to **pickle** a results object, this method of serialization is **not recommended** for long-term storage since pickle files are relatively fragile to changes in pyGSTi or other python libraries.

```{code-cell} ipython3
#Not recommended:
# import pickle
# pickle.dump(results_TP, open('../../tutorial_files/exampleResults_TP.pkl',"wb"))
# pickle.dump(results_TP2, open('../../tutorial_files/exampleResults_TP2.pkl',"wb"))
# pickle.dump(results_reduced, open('../../tutorial_files/exampleResults_reduced.pkl',"wb"))
# pickle.dump(results_stdprac, open('../../tutorial_files/exampleResults_stdprac.pkl',"wb"))
```

## Checkpointing/Warmstarting

The `GateSetTomography` and `StandardGST` protocols both support checkpointing to enable resuming GST analysis after an unexpected failure, such as an out-of-memory error, or an unexpected timeout in resource limited compute environments (clusters etc.), or for whatever other reason. Checkpointing is enabled by default, so no additional changes are needed in order to have these generated. 

Each protocol has a corresponding checkpoint object, `GateSetTomographyCheckpoint` and `StandardGSTCheckpoint`, which are saved to disk over the course of an iterative fit in serialized json format. By default checkpoint files associated with a `GateSetTomographyCheckpoint` object are saved to a new directory located in whichever current working directory the protocol is being run from named 'gst_checkpoints'. A new file is written to disk after each iteration with default naming of the form `GateSetTomography_iteration_{i}.json` where i is the index of the completed GST iteration associated with that checkpoint. Similarly, for a `StandardGSTCheckpoint` object the checkpoints are by default saved to a directory named 'standard_gst_checkpoints' with default file names of the form `StandardGST_{mode}_iteration_{i}` where mode corresponds to the current parameterized fit or model test associated with that file (including checkpoint information for all previously completed modes prior to the currently running one) and i is the index of the completed iteration within that current mode.

Below we repeat our first example of the notebook, but this time with checkpointing enabled (as is the default)

```{code-cell} ipython3
from pygsti.modelpacks import smq1Q_XYI
target_model_TP = smq1Q_XYI.target_model("full TP")
proto = pygsti.protocols.GateSetTomography(target_model_TP)
results_TP = proto.run(data, checkpoint_path = '../../tutorial_files/gst_checkpoints/GateSetTomography')
```

Note that in the example above we have specified a value for an additional kwarg called `checkpoint_path`. This allows for overriding the default behavior for the save location and naming of checkpoint files. The expected format is `{path}/{name}` where path is the directory to save the checkpoint files to (with that directory being created is required) and where name is the stem of the checkpoint file names `{name}_iteration_{i}.json`. Inspecting the contents of the directory we just specified, we can see that it is now populated by 8 new checkpoint files.

```{code-cell} ipython3
import os
os.listdir('../../tutorial_files/gst_checkpoints/')
```

Suppose hypothetically that a GST fit had failed at iteration 5 and we wanted to restart from that point without redoing all of the previous iterations from scratch again. We'll call this warmstarting. We can do so by reading in the appropriate serialized checkpoint object using the `read` class method of `GateSetTomographyCheckpoint` and passing that now loaded checkpoint object in for the `checkpoint` kwarg of `run`.

```{code-cell} ipython3
from pygsti.protocols import GateSetTomographyCheckpoint
gst_iter_5_checkpoint = GateSetTomographyCheckpoint.read('../../tutorial_files/gst_checkpoints/GateSetTomography_iteration_5.json')
results_TP_from_iter_5= proto.run(data, checkpoint= gst_iter_5_checkpoint, checkpoint_path = '../../tutorial_files/gst_checkpoints/GateSetTomography')
```

We can see from the output that we indeed started from iteration 6 (note the output log indexes from 1 instead of 0). Moreover we can see that we've indeed produced the same output as before without warmstarting, as we would expect/hope:

```{code-cell} ipython3
all(results_TP.estimates['GateSetTomography'].models['final iteration estimate'].to_vector() == \
results_TP_from_iter_5.estimates['GateSetTomography'].models['final iteration estimate'].to_vector())
```

The checkpoint object itself contains information that could be useful for diagnostics or debugging, including the current list of models associated each iterative fit, the last completed iteration it is associated with, and the list of circuits for the last completed iteration it is associated with.

+++

Checkpointing with the `StandardGST` protocol works similarly:

```{code-cell} ipython3
proto_standard_gst = pygsti.protocols.StandardGST(modes=['full TP', 'CPTPLND', 'Target'], verbosity=3)
results_stdprac = proto_standard_gst.run(data, checkpoint_path = '../../tutorial_files/standard_gst_checkpoints/StandardGST')
```

Except this time we have significantly more files saved, as during the course of the StandardGST protocol we're actually running three subprotocols:

```{code-cell} ipython3
os.listdir('../../tutorial_files/standard_gst_checkpoints/')
```

Note that the StandardGST protocol runs the subprotocols in the order listed in the `modes` argument, and checkpoint objects labeled with a given model label additionally contain the checkpointing information for the final iterations of any preceding modes which have been completed. i.e. the CPTPLND checkpoint objects contain the information required for full TP. Likewise, checkpoints for Target contain the information required for the full TP and CPTPLND modes. As before, imagine that our fitting failed for whatever reason during iteration 5 of CPTPLND, we can warmstart the protocol by loading in the checkpoint object associated with iteration 4 as below:

```{code-cell} ipython3
from pygsti.protocols import StandardGSTCheckpoint
standard_gst_checkpoint = StandardGSTCheckpoint.read('../../tutorial_files/standard_gst_checkpoints/StandardGST_CPTPLND_iteration_4.json')
results_stdprac_warmstart= proto_standard_gst.run(data, checkpoint= standard_gst_checkpoint, checkpoint_path = '../../tutorial_files/standard_gst_checkpoints/StandardGST')
```

Notice that we've indeed skipped past the previously completed full TP mode and jumped straight to the 6th iteration of the CPTPLND fit as expected. 

As for the GateSetTomographyCheckpoint object described above, the `StandardGSTCheckpoint` can often be useful to inspect as a debugging/diagnostic tool. `StandardGSTCheckpoints` are essentially structured as container object that hold a set of child `GateSetTomographyCheckpoint` and `ModelTestCheckpoint` (more on these in the ModelTest tutorial) objects for each of the modes being run (and potentially more types of chile checkpoints in the future as we add additional functionality). These children can be accessed using the `children` attribute of a `StandardGSTCheckpoint` instance which is a dictionary with keys given by the mode names contained therein.

```{code-cell} ipython3
print(standard_gst_checkpoint.children['CPTPLND'])
```

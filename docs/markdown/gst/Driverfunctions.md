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

# Driver functions for running Gate Set Tomography

+++

The `pygsti` package provides multiple levels of abstraction over the core Gate Set Tomography (GST) algorithms.  This  tutorial will show you how to work with pyGSTi's top-level functions for performing GST with a minimial amount of effort.  In order to run the GST protocol there are 3 essential ingredients: 1) data specifing the experimental outcomes, 2) a desired, or "target", `Model`, and 3) lists of `Circuit` objects, specifying the operation sequences to use at each successive step in the GST optimization.  The [GST overview tutorial](Overview), gave an end-to-end example of how to construct GST circuits, run GST, and generate a report.  This tutorial focus on the second step in more detail; more information about circuit construction and report generation can be found in the [GST circuits tutorial](CircuitConstruction) and [report generation tutorial](../reporting/ReportGeneration).

There are several different "driver routines" for running GST, and we'll cover in turn:
- `run_long_sequence_gst` - runs a single instance of GST with "standard" circuit lists.
- `run_long_sequence_gst_base` - runs a single instance of GST with custom circuit lists.
- `run_stdpractice_gst` - runs a multiple instances of GST with "standard" circuits based on an `ExplicitOpModel` model.

Each function returns a single `pygsti.objects.Results` object (see the [Result object tutorial](../objects/Results)), which contains the *single* input `DataSet` and one or more *estimates* (`pygsti.objects.Estimate` objects). 

Note: The abbreviation **LSGST** (lowercase in function names to follow Python naming conventions) stands for *Long Sequence LinearOperator Set Tomography*, and refers to the more powerful and flexible of GST that utilizes long sequences to find model estimates.  LSGST can be compared to *Linear GST*, or **LGST**, which only uses short sequences and as a result provides much less accurate estimates.

```{code-cell} ipython3
import pygsti
```

## Setup
First, we set our desired *target model* to be the standard $X(\pi/2)$, $Y(\pi/2)$ model that we've been using in many of these tutorials, and use the standard fiducial and germ sequences needed to generate the GST operation sequences (see the [modelpack tutorial](../objects/ModelPacks)).  We also specify a list of maximum lengths.  We'll analyze the simulated data generated in the [DataSet tutorial](../objects/DataSet), so **you'll need to run that tutorial if you haven't already**.

```{code-cell} ipython3
from pygsti.modelpacks import smq1Q_XY
target_model = smq1Q_XY.target_model()
prep_fiducials, meas_fiducials = smq1Q_XY.prep_fiducials(), smq1Q_XY.meas_fiducials()
germs = smq1Q_XY.germs()

maxLengths = [1,2,4,8,16]

ds = pygsti.io.load_dataset("../../tutorial_files/Example_Dataset.txt", cache=True)
```

## `run_long_sequence_gst`
This driver function finds what is logically a **single GST estimate** given a `DataSet`, a target `Model`, and other parameters.  We say "logically" because the returned `Results` object may actually contain multiple related estimates in certain cases.  Most important among the other parameters are the fiducial and germ sequences and list of maximum lengths needed to define a *standard*  (*prep_fiducial + germ^power + meas_fiducial*) set of GST circuit lists.

```{code-cell} ipython3
results = pygsti.run_long_sequence_gst(ds, target_model, prep_fiducials, meas_fiducials, germs, maxLengths)
```

A summary of what's inside a Results object is obtained by printing it
(for more examples of how to use a Results object, see the [Results tutorial](../objects/Results)).

```{code-cell} ipython3
print(results)
```

### Beyond the minimum
The above example supplies the minimal amount of information required to run the long-sequence GST algorithm.  `run_long_sequence_gst` can be used in a variety of contexts and accepts additional (optional) arguments that affect the way the algorithm is run.  Here we make several remarks regarding alternate or more advanced usage of `run_long_sequence_gst`.

- For many of the arguments, you can supply either a filename or a python object (e.g. dataset, target model, operation sequence lists), so if you find yourself loading things from files just to pass them in as arguments, you're probabaly working too hard.

- Typically we want to apply certain constraints to a GST optimization.  As mentioned in the model tutorial, the space over which a gate-set estimation is carried out is dictated by the parameterization of the `target_model` argument.  For example, to constrain a GST estimate to be trace-preserving, one should call `set_all_parameterizations("TP")` on the target `Model` before calling `run_long_sequence_gst`.  See the [tutorial on explicit models](../objects/ExplicitModel) for more information.

- the `gauge_opt_params` argument specifies a dictionary of parameters ultimately to be passed to the `gaugeopt_to_target` function (which provides full documentation).  By specifying an `item_weights` argument we can set the ratio of the state preparation and measurement (SPAM) weighting to the gate weighting when performing a gauge optimization.  In the example below, the gate parameters are weighted 1000 times more relative to the SPAM parameters.  Mathematically this corresponds to a multiplicative factor of 0.001 preceding the sum-of-squared-difference terms corresponding to SPAM elements in the model.   Typically it is good to weight the gates parameters more heavily since GST amplifies gate parameter errors via long operation sequences but cannot amplify SPAM parameter errors.  If unsure, 0.001 is a good value to start with.  For more details on the arguments of `gaugeopt_to_target`, see the previous tutorial on low-level algorithms.  For more infomation, see the [gauge optimization tutorial](../utilities/GaugeOpt).

The below call illustrates all three of these.

```{code-cell} ipython3
mdl_target_TP = target_model.copy() #make a copy so we don't change target_model's parameterization, 
                                #  since this could be confusing later...
mdl_target_TP.set_all_parameterizations("full TP") #constrain GST estimate to TP

results_TP = pygsti.run_long_sequence_gst("../../tutorial_files/Example_Dataset.txt", mdl_target_TP,
                                         prep_fiducials, meas_fiducials, germs, maxLengths,
                                        gauge_opt_params={'item_weights': {'gates': 1.0, 'spam': 0.001}})
```

## `run_long_sequence_gst_base`
This function performs the same analysis as `run_long_sequence_gst` except it allows the user to fully specify the list of operation sequences as either a list of lists of `Circuit` objects or a list of or single `CircuitStructure` object(s). A `CircuitStructure` is preferable as it allows the structured plotting of the sequences in report figures.  In this example, we'll just generate a standard set of structures, but with some of the sequences randomly dropped (see the [tutorial on GST circuit reduction](FiducialPairReduction)).  Note that like `run_long_sequence_gst`, `run_long_sequence_gst_base` is able to take filenames as arguments and accepts a `gauge_opt_params` argument for customizing the gauge optimization that is performed.

```{code-cell} ipython3
#Create the same sequences but drop 50% of them randomly for each repeated-germ block.
lsgst_structs = pygsti.circuits.create_lsgst_circuit_lists(target_model, prep_fiducials, meas_fiducials,
                                                       germs, maxLengths, keep_fraction=0.5, keep_seed=2018)
results_reduced = pygsti.run_long_sequence_gst_base(ds, target_model, lsgst_structs)
```

## `run_stdpractice_gst`
This function calls `run_long_sequence_gst` multiple times using typical variations in gauge optimization parameters and `ExplicitOpModel` parameterization (this doesn't work for other types `Model` objects, e.g. *implicit* models, which don't implement `set_all_parameterizations`).  This function provides a clean and simple interface to performing a "usual" set of GST analyses on a set of data.  As such, it takes a single `DataSet`, similar gate-sequence-specifying parameters to `run_long_sequence_gst`, and a new `modes` argument which is a comma-separated list of "canned" GST analysis types (e.g. `"TP,CPTP"` will compute a Trace-Preserving estimate *and* a Completely-Positive & Trace-Preserving estimate). The currently available modes are:
 - "full" : unconstrained gates (fully parameterized)                                                                 
 - "TP"   : TP-constrained gates and state preparations
 - "CPTP" : CPTP-constrained gates and TP-constrained state preparations               
 - "H+S"  : Only Hamiltonian and Pauli stochastic errors allowed (CPTP)                                             
 - "S"    : Only Pauli-stochastic errors allowed (CPTP)                                                           
 - "Target" : use the target (ideal) gates as the estimate       

The gauge optimization(s) `run_stdpractice_gst` performs are controlled by its `gauge_opt_suite` and `gaugeOptTarget` arguments. The former is can be either a string, specifying a standard "suite" of gauge optimizations, or a dictionary of dictionaries similar to the `gauge_opt_params` argument of `run_long_sequence_gst` (see docstring).  The `gaugeOptTarget` argument may be set to a `Model` that is used as the target for gauge optimization, overriding the (typically ideal) target gates given by the `target_modelFilenameOrObj` argument.

```{code-cell} ipython3
results_stdprac = pygsti.run_stdpractice_gst(ds, target_model, prep_fiducials, meas_fiducials, germs, maxLengths,
                                        modes="full TP,CPTP,Target") #uses the default suite of gauge-optimizations
```

```{code-cell} ipython3
print("Estimates: ", ", ".join(results_stdprac.estimates.keys()))
print("TP Estimate's gauge optimized models: ", ", ".join(results_stdprac.estimates["full TP"].goparameters.keys()))
```

Next, we'll perform the same analysis but with a **non-default standard suite of gauge optimizations** - this one toggles the SPAM penalty in addition to varying the spam weight (the default suite just varies the spam weight without any SPAM penalty).  See the [gauge optimization tutorial](../utilities/GaugeOpt) for more details on gauge optmization "suites".

```{code-cell} ipython3
results_stdprac_nondefaultgo = pygsti.run_stdpractice_gst(
    ds, target_model, prep_fiducials, meas_fiducials, germs, maxLengths,
    modes="full TP,CPTP,Target", gaugeopt_suite="varySpam")
```

```{code-cell} ipython3
print("Estimates: ", ", ".join(results_stdprac_nondefaultgo.estimates.keys()))
print("TP Estimate's gauge optimized models: ", ", ".join(results_stdprac_nondefaultgo.estimates["full TP"].goparameters.keys()))
```

Finally, we'll demonstrate how to specify a fully custom set of gauge optimization parameters and how to use a **separately-specified target model for gauge optimization**.  You can get a more intuitive gauge-optimized `Model` when by placing as much expected noise as possible into the gauge-optimization target, as this essentially tells the algorithm "this is what I think the estimated model should look like".  If you just use the perfect or ideal model for this (the default), then the gauge optimizer may make tradeoffs which don't reflect the expected physics (remember, all gauge-equivalent models product the same observables!).  For example, it may spread error across all your gate operations when you expect just the 2-qubit operations are noisy.

```{code-cell} ipython3
my_goparams = { 'item_weights': {'gates': 1.0, 'spam': 0.001} }
my_gaugeOptTarget= smq1Q_XY.target_model('full TP')
my_gaugeOptTarget = my_gaugeOptTarget.depolarize(op_noise=0.005, spam_noise=0.01) # a guess at what estimate should be
results_stdprac_customgo = pygsti.run_stdpractice_gst(
    ds, target_model, prep_fiducials, meas_fiducials, germs, maxLengths,
    modes="full TP,CPTP,Target", gaugeopt_suite={ 'myGO': my_goparams }, gaugeopt_target=my_gaugeOptTarget)
```

```{code-cell} ipython3
print("Estimates: ", ", ".join(results_stdprac_customgo.estimates.keys()))
print("TP Estimate's gauge optimized models: ", ", ".join(results_stdprac_customgo.estimates["full TP"].goparameters.keys()))
```

To finish up, we'll pickle the results, potentially for processing in other tutorials.

```{code-cell} ipython3
import pickle
pickle.dump(results, open('../../tutorial_files/exampleResults.pkl',"wb"))
pickle.dump(results_TP, open('../../tutorial_files/exampleResults_TP.pkl',"wb"))
pickle.dump(results_reduced, open('../../tutorial_files/exampleResults_reduced.pkl',"wb"))
pickle.dump(results_stdprac, open('../../tutorial_files/exampleResults_stdprac.pkl',"wb"))
```

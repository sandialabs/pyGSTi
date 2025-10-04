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

# Gauge Optimization Tutorial

This tutorial is about gauge optimization.  First, we'll explain what gauge optimization is, then we'll demonstrate how pyGSTi can used to perform it.

## What is gauge optimization?
In pyGSTi, *models* of quantum information processors have parameters - that is, they're *parameterized models* which posess some number of knobs, or *parameters*, that you can adjust to tune the model.  For example, a "fully parameterized model" has an independent parameter for every SPAM-vector or process-matrix element (see the [explicit model tutorial](../../objects/ExplicitModel.ipynb) for more details).  Sometimes (more often than you might think) there are combinations of knob movements (directions in parameter-space) which do not affect *any* physically observed quantities, i.e. any circuit outcome probabilities.  These are called *gauge degrees of freedom*, and **gauge optimization is an optimization over a model's gauge degrees of freedom**.  Sometimes the gauge degrees of freedom form a group - the *gauge group* - which makes the space of *gate transformations* (a movement along just the gauge directions of parameter space) particularly straightforward to optimize over.  

In the example of a fully-parameterized model, a gauge transformation $T_M$ is associated with an invertable matrix *M* and transforms the state preparation vectors $|\rho\rangle\rangle$, the POVM effect vectors $\langle\langle E|$, and the process matrices $G$ of a model as follows:

$$ |\rho\rangle\rangle \rightarrow M^{-1} |\rho\rangle\rangle \\
 \langle\langle E| \rightarrow M \langle\langle E| \\
 G \rightarrow M^{-1} G M $$
 
You can check that under this transformation none of the outcome probabilities (which are expressed as bra-kets $\langle\langle E| G_n \cdots G_2 G_1 |\rho\rangle\rangle$) are affected.  Since the model is fully-parameterized each vector and matrix element is a parameter, and the above transformation on elements is equally a transformation of the model's parameter space.

## Why do it?
You may be thinking, if the gauge doesn't matter for any observable quantities why do we car about it at all?  This is a great question, and **in an ideal world we wouldn't care about gauge or gauge optimization**.  The reason we sometimes want to optimize the non-physical gauge degrees of freedom is to try to make a model look like some target model so that we can extract some meaning from *gauge-variant* metrics.  Such metrics (e.g. the process fidelity and diamond distance) are inherently dubious because they vary with gauge choice, but they have become so prolific in the literature that it's worthwhile to interact as best we can with such metrics.  

## How to optimize the gauge in pyGSTi
In pyGSTi there are many ways to tune exactly how gauge optimization is performed.  This can be somewhat of a black art, since there's no *a priori* best way to choose the gauge of a model - it depends on what you're after.  Furthermore, **pyGSTi can only perform gauge optimization when there exists a well-defined gauge group** (see above).  Here's we'll demonstrate some of the basics of how gauge optimization works in pyGSTi.

### Gauge transformations
Before *optimizing* the gauge, lets take a look at how to *change* the gauge.  Throughout this notebook we'll be looking at a fully-parameterized 1-qubit model whose gauge group is the set of $4 \times 4$ invertible matrices.  (The picture is essentially the same for trace-preserving (TP) parameterized models, but becomes more complicated when moving to CPTP constrainted models.)  Lets use one of the target models builtin to pyGSTi (see the [model packs tutorial](../../objects/advanced/ModelPacks.ipynb) for more information):

```{code-cell} ipython3
import numpy as np
import pygsti
from pygsti.modelpacks import smq1Q_XYI as std

mdl = std.target_model()
print(mdl)
```

You can transform this gateset into a different gauge using the `transform` method.  It takes as an argument a gauge-group-element (similar to the matrix $M$ above).  In this case, we'll use a `FullGaugeGroupElement`, which can be initialized from an arbitrary invertible matrix:  

```{code-cell} ipython3
from pygsti.models.gaugegroup import FullGaugeGroupElement

ggEl = FullGaugeGroupElement(np.array([[1,   0,      0,     0],
                                       [0,   1,   -0.1,     0],
                                       [0,   0.1,    1,  -0.1],
                                       [0,   0,    0.1,     1]],'d')) # an arbitrary M matrix
mdl.transform_inplace(ggEl)
print(mdl)
```

### Gauge optimization with `gaugeopt_to_target`
When we optimize the gauge degrees of freedom of a model, this means we (attempt to) find the gauge which maximizes (or minimizes) the value of some necessarily gauge-variant objective function.  While it is possible to supply a custom objective function (using the `gaugeopt_custom` function), this objective function is typically a gauge-variant distance or fidelity between the model being optimized and a fixed *target* model.  When this is the case, we use the `gaugeopt_to_target` function: 

```{code-cell} ipython3
target_mdl = std.target_model()
gauge_optimized_mdl = pygsti.algorithms.gaugeopt_to_target(mdl, target_mdl)
print(gauge_optimized_mdl)
```

Note how the above cell recovered (perfectly) the `target_mdl`, since `mdl` only varied from the target by a gauge transformation.  The above cell optimized the gauge over a fully-parameterized gauge group (meaning the group of *all* invertible matrices).  This is so because all `ExplicitOpModel` objects have a default gauge group that is used by `gaugeopt_to_target` when no `gauge_group` argument is supplied.  In our case, this default group is:

```{code-cell} ipython3
mdl.default_gauge_group
```

There are other argument of `gaugeopt_to_target` which specify details of the objective function, such as what distance metric is used and how to weight the different operations within a model.  Here are a few examples (please see the docstring for more details):

```{code-cell} ipython3
# Optimize the (average) trace distance instead of the frobenius distance (the default)
gauge_optimized_mdl = pygsti.algorithms.gaugeopt_to_target(mdl, target_mdl, gates_metric="tracedist")

# Only optimize over trace-preserving gauge transformations (i.e. those that map TP -> TP operations)
from pygsti.models.gaugegroup import TPGaugeGroup
gauge_optimized_mdl = pygsti.algorithms.gaugeopt_to_target(mdl, target_mdl, gauge_group=TPGaugeGroup(1))

# Weight the gates more heavily than the SPAM within the (default) frobenius distance
gauge_optimized_mdl = pygsti.algorithms.gaugeopt_to_target(mdl, target_mdl, item_weights={'gates': 1.0, 'spam': 0.1})

# Weight the Gx gate more heavily than the SPAM & other gates within the (default) frobenius distance
gauge_optimized_mdl = pygsti.algorithms.gaugeopt_to_target(mdl, target_mdl,
                                                           item_weights={'Gx': 1.0, 'gates': 0.1, 'spam': 0.01},
                                                           verbosity=2)
```

### Gauge optimization in high-level routines
Because gauge optimization is a common post-processing step to Gate Set Tomography (GST), `pygsti.run_long_sequence_gst` and `pygsti.run_stdpractice_gst` have arguments that allow you to specify arguments to underlying calls to `gaugeopt_to_target`.  To demonstrate, we'll create some simulated data to run GST.

```{code-cell} ipython3
#Create (simulate) some data
mdl_datagen = std.target_model().depolarize(op_noise=0.1, spam_noise=0.001)
listOfExperiments = pygsti.circuits.create_lsgst_circuits(
    mdl_datagen, std.prep_fiducials(), std.meas_fiducials(), std.germs(), [1,2,4])
ds = pygsti.data.simulate_data(mdl_datagen, listOfExperiments, num_samples=10000, seed=1234)
```

The `gauge_opt_params` argument of `run_long_sequence_gst` takes a dictionary of arguments to be passed to a `gaugeopt_to_target`:

```{code-cell} ipython3
#Perform GST with custom gauge optmization post-processing parameters
target_model = std.target_model()
results = pygsti.run_long_sequence_gst(
    ds, target_model, std.prep_fiducials(), std.meas_fiducials(), std.germs(), [1,2,4],
    gauge_opt_params={'item_weights': {'gates': 1.0, 'spam': 0.1}}, verbosity=1)
```

The `run_stdpractice_gst` function supports performing multiple gauge transformations and also supports daisy-chaining several calls to `gaugeopt_to_target` (performing multiple calls to `gaugeopt_to_target` whereby the output `Model` of a call serves as the input of the subsequence call).  This is specified via the `gauge_opt_suite` argument, which may be set to a dictionary of whose keys are gauge-optimization-suite labels and whose values are one of:
1. a dictionary of `gaugeopt_to_target` arguments, similar to the `gauge_opt_params` argument of `run_long_sequence_gst`.
2. a list of `gaugeopt_to_target`-argument dictionaries specifying multiple daisy-chained calls of `gaugeopt_to_target`.

In the below call we demonstrate both.  Note that we set `modes="TP"` so that `run_stdpractice_gst` only runs TP-constrained GST:

```{code-cell} ipython3
oneQ_TPGaugeGroup = TPGaugeGroup(1)
target_model = std.target_model()
std_results = pygsti.run_stdpractice_gst(
    ds, target_model, std.prep_fiducials(), std.meas_fiducials(), std.germs(), [1,2,4], modes="full TP", verbosity=1,
    gaugeopt_suite={'myGOpt_highGateWeight': {'item_weights': {'gates': 1.0, 'spam': 0.1}},
                   'myGOpt_daisyChain': [ {'gauge_group': oneQ_TPGaugeGroup,
                                           'item_weights': {'gates': 1.0, 'spam': 0.0}},
                                          {'gauge_group': oneQ_TPGaugeGroup,
                                           'item_weights': {'gates': 0.0, 'spam': 1.0}}
                                        ]
                  } )
```

We can see that the `"TP"` estimate constains the expected gauge-optimized models (see the [Results object tutorial](../../objects/advanced/Results.ipynb) on how to access the elements of a `ModelEstimateResults` object):

```{code-cell} ipython3
print("\n".join(std_results.estimates['full TP'].models.keys()))
```

#### Other tips:
- by default, the target model used is the starting-point model given to `run_long_sequence_gst` or `run_stdpractice_gst`.  You can gauge-optimize the distance to a *different* model by specifying a `target_model` in a `gaugopt_to_target` argument dictionary. You can also do this by specifying the `gaugeOptTarget` argument of `run_stdpractice_gst` (see the docstring).
- you can add gauge optimizations to an existing `ModelEstimateResults` object using the `add_gaugeoptimized` method of one of its contained `Estimate` objects.

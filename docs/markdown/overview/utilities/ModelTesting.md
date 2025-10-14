---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.17.3
kernelspec:
  display_name: gst_checkpointing
  language: python
  name: gst_checkpointing
---

# Model Testing
This tutorial covers different methods of **comparing data to given (fixed) QIP models**.  This is distinct from model-based *tomography*, which finds the best-fitting model for a data set within a space of models set by a `Model` object's parameterization.  You might use this as a tool alongside or separate from GST.  Perhaps you suspect that a given noisy QIP model is compatible with your data - model *testing* is the way to find out. Because there is no optimization involved, model testing requires much less time than GST does, and doens't place any requirements on which circuits are used in performing the test (though some circuits will give a more precise result).

## Setup
First, after some usual imports, we'll create some test data based on a depolarized and rotated version of a standard 1-qubit model consisting of $I$ (the identity), $X(\pi/2)$ and $Y(\pi/2)$ gates.  Here we just use a set of standard GST circuits.

```{code-cell} ipython3
import pygsti
import numpy as np
import scipy
from scipy import stats
from pygsti.modelpacks import smq1Q_XY
```

```{code-cell} ipython3
datagen_model = smq1Q_XY.target_model().depolarize(op_noise=0.05, spam_noise=0.1).rotate((0.05,0,0.03))
exp_design = smq1Q_XY.create_gst_experiment_design(max_max_length=16)
ds = pygsti.data.simulate_data(datagen_model, exp_design.all_circuits_needing_data,
                                            num_samples=1000, sample_error='binomial', seed=100)
data = pygsti.protocols.ProtocolData(exp_design, ds)
```

## Step 1: Construct a test model
After we have some data, the first step is creating a model or models that we want to test.  This just means creating a `Model` object containing the operations (including SPAM) found in the data set.  We'll create several models that are meant to look like guesses (some including more types of noise) of the true underlying model.

```{code-cell} ipython3
target_model = smq1Q_XY.target_model()
test_model1 = target_model.copy()
test_model2 = target_model.depolarize(op_noise=0.07, spam_noise=0.07)
test_model3 = target_model.depolarize(op_noise=0.07, spam_noise=0.07).rotate( (0.02,0.02,0.02) )
```

## Step 2: Test it!
There are three different ways to test a model.  Note that in each case the default behavior (and the only behavior demonstrated here) is to **never gauge-optimize the test `Model`**.  (Whenever gauge-optimized versions of an `Estimate` are useful for comparisons with other estimates, *copies* of the test `Model` are used *without* actually performing any modification of the original `Model`.)

### Method1: the `ModelTest` protocol
The most straightforward way to perform model testing is using the `ModelTest` protocol.  This is invoked similarly to other protocols: you create a protocol object and `.run()` it on a `ProtocolData` object.  This returns a `ModelEstimateResults` object, similarly to the `GateSetTomography` and `StandardGST` protocols.  The "estimateLabel" advanced option names the `Estimate` within the returned results, and can be particularly useful when testing multiple models.

```{code-cell} ipython3
# creates a Results object with a "model1" estimate
results = pygsti.protocols.ModelTest(test_model1, target_model, name='model1').run(data)

# creates a Results object with a "model2" estimate
results2 = pygsti.protocols.ModelTest(test_model2, target_model, name='model2').run(data)

# creates a Results object with a "model3" estimate
results3 = pygsti.protocols.ModelTest(test_model3, target_model, name='model3').run(data)
```

Like any other set of `ModelEstimateResults` objects which share the same underlying `ProtocolData` object, we can collect all of these estimates into a single `ModelEstimateResults` object and easily make a report containing all three.

```{code-cell} ipython3
results.add_estimates(results2)
results.add_estimates(results3)

pygsti.report.construct_standard_report(
    results, title="Model Test Example Report", verbosity=1
).write_html("../tutorial_files/modeltest_report", auto_open=False, verbosity=1)
```

```{code-cell} ipython3
print(results)
```

### Method 2: `add_model_test`
Alternatively, you can add a model-to-test to an existing `ModelEstimateResults` object.  This is convenient when running GST via `GateSetTomography` or `StandardGST` has left you with a `ModelEstimateResults` object and you also want to see how well a hand-picked model fares.  Since the results object already contains a `ProtocolData`, all you need to do is provide a `Model`.  This is accomplished using the `add_model_test` method of a `ModelEstimateResults` object.

```{code-cell} ipython3
#Create some GST results using standard practice GST
gst_results = pygsti.protocols.StandardGST().run(data)

#Add a model to test
gst_results.add_model_test(target_model, test_model3, estimate_key='MyModel3')

#Create a report to see that we've added an estimate labeled "MyModel3"
pygsti.report.construct_standard_report(
    gst_results, title="GST with Model Test Example Report 1", verbosity=1
).write_html("../tutorial_files/gstwithtest_report1", auto_open=False, verbosity=1)
```

### Method 3: `modelToTest` argument
Finally, yet another way to perform model testing alongside GST is by using the `models_to_test` argument of the `StandardGST` protocol.  This essentially combines calls to `StandardGST.run` and `ModelEstimateResults.add_model_test` (demonstrated above) with the added control of being able to specify the ordering of the estimates via the `modes` argument.  Two important remarks are in order:

1. You *must* specify the names (keys of the `models_to_test` argument) of your test models in the comma-delimited string that is the `modes` argument.  Just giving a dictionary of `Model`s as `models_to_test` will not automatically test those models in the returned `ModelEstimateResults` object.

2. You don't actually need to run any GST modes, and can use `StandardGST` in this way to in one call create a single `ModelEstimateResults` object containing multiple model tests, with estimate names that you specify.  Thus, running `StandardGST` can replace running `ModelTest` multiple tiems (with "estimateLabel" advanced options) followed by collecting the estimates using `ModelEstimateResults.add_estimates` demonstrated under "Method 1" above.

```{code-cell} ipython3
proto = pygsti.protocols.StandardGST(modes=["full TP","Test2","Test3","Target"], # You MUST put Test2 and Test3 here...
                                     models_to_test={'Test2': test_model2, 'Test3': test_model3})
gst_results = proto.run(data, disable_checkpointing=True)

pygsti.report.construct_standard_report(
    gst_results, title="GST with Model Test Example Report 2", verbosity=1
).write_html("../tutorial_files/gstwithtest_report2", auto_open=False, verbosity=1)
```

## Checkpointing/Warmstarting

Just like the GST protocols discussed in [GST-Protocols](GST-Protocols.ipynb), `ModelTest` protocols also support checkpointing/warmstarting in the event of unexpected failure or termination using `ModelTestCheckpoint` objects which are serialized and written to disk over the course of running an iterative `ModelTest`. We direct you to the linked tutorial for more on the basic syntax and usage, which essentially identical for `ModelTestCheckpoints` objects.

```{code-cell} ipython3
# creates a Results object with a "model1" estimate
modeltest_proto = pygsti.protocols.ModelTest(test_model1, target_model, name='model1')
results = modeltest_proto.run(data, checkpoint_path = '../tutorial_files/modeltest_checkpoints/ModelTest_model1')
```

We now have a bunch of checkpoint files associated with this `ModelTest` protocol saved in the specified directory:

```{code-cell} ipython3
import os
os.listdir('../tutorial_files/modeltest_checkpoints/')
```

Suppose hypothetically we needed to restart the running of this ModelTest from where it left off following iteration 4. We can do so reading in the saved `ModelTestCheckpoint` object and passing it into the `ModelTest`'s `run` method.

```{code-cell} ipython3
from pygsti.protocols import ModelTestCheckpoint
modeltest_checkpoint = ModelTestCheckpoint.read('../tutorial_files/modeltest_checkpoints/ModelTest_model1_iteration_4.json')
results_warmstarted= modeltest_proto.run(data, checkpoint= modeltest_checkpoint, checkpoint_path= '../tutorial_files/modeltest_checkpoints/ModelTest_model1')
```

As expected, the results are identical in both cases:

```{code-cell} ipython3
print(results.estimates['model1'].parameters['model_test_values'])
print(results.estimates['model1'].parameters['model_test_values'] == \
results_warmstarted.estimates['model1'].parameters['model_test_values'])
```

Thats it!  Now that you know more about model-testing you may want to go back to the [overview of pyGSTi protocols](../03-Protocols.ipynb).

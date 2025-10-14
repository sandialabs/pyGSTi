---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.17.3
kernelspec:
  display_name: api_updates
  language: python
  name: api_updates
---

# Functions for Model Testing
This tutorial covers different methods of **comparing data to given (fixed) QIP models**.  This is distinct from model-based *tomography*, which finds the best-fitting model for a data set within a space of models set by a `Model` object's parameterization.  You might use this as a tool alongside or separate from GST.  Perhaps you suspect that a given noisy QIP model is compatible with your data - model *testing* is the way to find out. Because there is no optimization involved, model testing requires much less time than GST does, and doens't place any requirements on which circuits are used in performing the test (though some circuits will give a more precise result).

## Setup
First, after some usual imports, we'll create some test data based on a depolarized and rotated version of a standard 1-qubit model consisting of $I$ (the identity), $X(\pi/2)$ and $Y(\pi/2)$ gates.

```{code-cell} ipython3
import pygsti
import numpy as np
import scipy
from scipy import stats
from pygsti.modelpacks import smq1Q_XYI
```

```{code-cell} ipython3
datagen_model = smq1Q_XYI.target_model().depolarize(op_noise=0.05, spam_noise=0.1).rotate((0.05,0,0.03))
max_lens = [1,2,4,8]
exp_list = pygsti.circuits.create_lsgst_circuits(
    smq1Q_XYI.target_model(), smq1Q_XYI.prep_fiducials(), smq1Q_XYI.meas_fiducials(),
    smq1Q_XYI.germs(), max_lens)
ds = pygsti.data.simulate_data(datagen_model, exp_list, num_samples=1000,
                                             sample_error='binomial', seed=100)
```

## Step 1: Construct a test model
After we have some data, the first step is creating a model or models that we want to test.  This just means creating a `Model` object containing the operations (including SPAM) found in the data set.  We'll create several models that are meant to look like guesses (some including more types of noise) of the true underlying model.

```{code-cell} ipython3
target_model = smq1Q_XYI.target_model()
test_model1 = target_model.copy()
test_model2 = target_model.depolarize(op_noise=0.07, spam_noise=0.07)
test_model3 = target_model.depolarize(op_noise=0.07, spam_noise=0.07).rotate( (0.02,0.02,0.02) )
```

## Step 2: Test it!
There are three different ways to test a model.  Note that in each case the default behavior (and the only behavior demonstrated here) is to **never gauge-optimize the test `Model`**.  (Whenever gauge-optimized versions of an `Estimate` are useful for comparisons with other estimates, *copies* of the test `Model` are used *without* actually performing any modification of the original `Model`.)

### Method1: `run_model_test`
First, you can do it "from scratch" by calling `run_model_test`, which has a similar signature as `run_long_sequence_gst` and folows its pattern of returning a `Results` object.  The "estimateLabel" advanced option, which names the `Estimate` within the returned `Results` object, can be particularly useful.

```{code-cell} ipython3
# creates a Results object with a "default" estimate
results = pygsti.run_model_test(test_model1, ds, target_model, 
                               smq1Q_XYI.prep_fiducials(), smq1Q_XYI.meas_fiducials(), smq1Q_XYI.germs(),
                               max_lens) 

# creates a Results object with a "default2" estimate
results2 = pygsti.run_model_test(test_model2, ds, target_model, 
                               smq1Q_XYI.prep_fiducials(), smq1Q_XYI.meas_fiducials(), smq1Q_XYI.germs(),
                               max_lens, advanced_options={'estimate_label': 'default2'}) 

# creates a Results object with a "default3" estimate
results3 = pygsti.run_model_test(test_model3, ds, target_model, 
                               smq1Q_XYI.prep_fiducials(), smq1Q_XYI.meas_fiducials(), smq1Q_XYI.germs(),
                               max_lens, advanced_options={'estimate_label': 'default3'})
```

```{code-cell} ipython3
print(results.estimates['ModelTest'])
```

Like any other set of `Results` objects which share the same `DataSet` and operation sequences, we can collect all of these estimates into a single `Results` object and easily make a report containing all three.

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
Alternatively, you can add a model-to-test to an existing `Results` object.  This is convenient when running GST via `run_long_sequence_gst` or `run_stdpractice_gst` has left you with a `Results` object and you also want to see how well a hand-picked model fares.  Since the `Results` object already contains a `DataSet` and list of sequences, all you need to do is provide a `Model`.  This is accomplished using the `add_model_test` method of a `Results` object.

```{code-cell} ipython3
#Create some GST results using run_stdpractice_gst
gst_results = pygsti.run_stdpractice_gst(ds, target_model, 
                                        smq1Q_XYI.prep_fiducials(), smq1Q_XYI.meas_fiducials(), smq1Q_XYI.germs(),
                                        max_lens)

#Add a model to test
gst_results.add_model_test(target_model, test_model3, estimate_key='MyModel3')

#Create a report to see that we've added an estimate labeled "MyModel3"
pygsti.report.construct_standard_report(
    gst_results, title="GST with Model Test Example Report 1", verbosity=1
).write_html("../tutorial_files/gstwithtest_report1", auto_open=False, verbosity=1)
```

### Method 3: `models_to_test` argument
Finally, yet another way to perform model testing alongside GST is by using the `models_to_test` argument of `run_stdpractice_gst`.  This essentially combines calls to `run_stdpractice_gst` and `Results.add_model_test` (demonstrated above) with the added control of being able to specify the ordering of the estimates via the `modes` argument.  To important remarks are in order:

1. You *must* specify the names (keys of the `models_to_test` argument) of your test models in the comma-delimited string that is the `modes` argument.  Just giving a dictionary of `Model`s as `models_to_test` will not automatically test those models in the returned `Results` object.

2. You don't actually need to run any GST modes, and can use `run_stdpractice_gst` in this way to in one call create a single `Results` object containing multiple model tests, with estimate names that you specify.  Thus `run_stdpractice_gst` can replace the multiple `run_model_test` calls (with "estimateLabel" advanced options) followed by collecting the estimates using `Results.add_estimates` demonstrated under "Method 1" above.

```{code-cell} ipython3
gst_results = pygsti.run_stdpractice_gst(ds, target_model, smq1Q_XYI.prep_fiducials(), smq1Q_XYI.meas_fiducials(), smq1Q_XYI.germs(),
                                       max_lens, modes="full TP,Test2,Test3,Target", # You MUST 
                                       models_to_test={'Test2': test_model2, 'Test3': test_model3})

pygsti.report.construct_standard_report(
    gst_results, title="GST with Model Test Example Report 2", verbosity=1
).write_html("../tutorial_files/gstwithtest_report2", auto_open=False, verbosity=1)
```

Thats it!  Now that you know more about model-testing you may want to go back to the [overview of pyGST applications](../02-Using-Essential-Objects.ipynb).

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

# How to add new gauge-optimizations to GST results
This example demonstrates how to take a previously computed `Results` object and add new gauge-optimized version of to one of the estimates.  First, let's "pre-compute" a `ModelEstimateResults` object using `StandardGST`, which contains a single `Estimate` called "TP":

```{code-cell} ipython3
import pygsti
from pygsti.modelpacks import smq1Q_XYI
```

```{code-cell} ipython3
#Generate some fake data and run GST on it.
exp_design = smq1Q_XYI.create_gst_experiment_design(max_max_length=4)
mdl_datagen  = smq1Q_XYI.target_model().depolarize(op_noise=0.1, spam_noise=0.001)
ds = pygsti.data.simulate_data(mdl_datagen, exp_design.all_circuits_needing_data, num_samples=1000, seed=1234)
data = pygsti.protocols.ProtocolData(exp_design, ds)

gst = pygsti.protocols.StandardGST("full TP", gaugeopt_suite={'go0': {'item_weights': {'gates': 1, 'spam': 1}}})
results = gst.run(data) 
results.write("example_files/regaugeopt_example")
```

Next, let's load in the pre-computed results and use the `add_gauge_optimization` method of the `pygsti.objects.Estimate` object to add a new gauge-optimized version of the (gauge un-fixed) model estimate stored in `my_results.estimates['default']`.  The first argument of `add_gauge_optimization` is just a dictionary of arguments to `pygsti.gaugeopt_to_target` **except** that you don't need to specify the `Model` to gauge optimize or the target `Model` (just like the `gaugeOptParams` argument of `run_long_sequence_gst`).  The optional "`label`" argument defines the key name for the gauge-optimized `Model` and the corresponding parameter dictionary within the `Estimate`'s `.models` and `.goparameters` dictionaries, respectively.

```{code-cell} ipython3
my_results = pygsti.io.read_results_from_dir("example_files/regaugeopt_example", name="StandardGST")
```

```{code-cell} ipython3
estimate = my_results.estimates['full TP']
estimate.add_gaugeoptimized( {'item_weights': {'gates': 1, 'spam': 0.001}}, label="Spam 1e-3" )
mdl_gaugeopt = estimate.models['Spam 1e-3']

print(list(estimate.goparameters.keys())) # 'go0' is the default gauge-optimization label
print(mdl_gaugeopt.frobeniusdist(estimate.models['target']))
```

One can also perform the gauge optimization separately and specify it using the `model` argument (this is useful when you want or need to compute the gauge optimization elsewhere):

```{code-cell} ipython3
mdl_unfixed = estimate.models['final iteration estimate']
mdl_gaugefixed = pygsti.gaugeopt_to_target(mdl_unfixed, estimate.models['target'], {'gates': 1, 'spam': 0.001})
estimate.add_gaugeoptimized( {'any': "dictionary", 
                              "doesn't really": "matter",
                              "but could be useful it you put gaugeopt params": 'here'},
                            model=mdl_gaugefixed, label="Spam 1e-3 custom" )
print(list(estimate.goparameters.keys()))
print(estimate.models['Spam 1e-3 custom'].frobeniusdist(estimate.models['Spam 1e-3']))
```

You can look at the gauge optimization parameters using `.goparameters`:

```{code-cell} ipython3
import pprint
pp = pprint.PrettyPrinter()
pp.pprint(dict(estimate.goparameters['Spam 1e-3']))
```

Finally, note that if, in the original creation of `StandardGST`, you set **`gaugeopt_suite=None`** then no gauge optimizations are performed (there would be no "`go0`" elements) and you start with a blank slate to perform whatever gauge optimizations you want on your own.

```{code-cell} ipython3

```

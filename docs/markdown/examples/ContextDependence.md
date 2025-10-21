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

# Context-Dependent GST
This example shows how to introduce new operation labels into a GST analysis so as to test for context dependence.  In particular, we'll look at the 1-qubit X, Y, I model.  Suppose a usual GST analysis cannot fit the model well, and that we think this is due to the fact that a "Gi" gate which immediately follows a "Gx" gate is affected by some residual noise that isn't otherwise present.  In this case, we can model the system as having two different "Gi" gates: "Gi" and "Gi2", and model the "Gi" gate as "Gi2" whenever it follows a "Gx" gate.

```{code-cell} ipython3
from __future__ import print_function
import pygsti
from pygsti.modelpacks.legacy import std1Q_XYI
```

First we'll create a mock data set that exhibits this context dependence.  To do this, we add an additional "Gi2" gate to the data-generating model, generate some data using "Gi2"-containing operation sequences, and finally replace all instances of "Gi2" with "Gi" so that it looks like data that was supposed to have just a single "Gi" gate.

```{code-cell} ipython3
# The usual setup: identify the target model, fiducials, germs, and max-lengths
target_model = std1Q_XYI.target_model()
fiducials = std1Q_XYI.fiducials
germs = std1Q_XYI.germs
maxLengths = [1,2,4,8,16,32]

# Create a Model to generate the data - one that has two identity gates: Gi and Gi2
mdl_datagen = target_model.depolarize(op_noise=0.1, spam_noise=0.001)
mdl_datagen["Gi2"] = mdl_datagen["Gi"].copy()
mdl_datagen["Gi2"].depolarize(0.1) # depolarize Gi2 even further
mdl_datagen["Gi2"].rotate( (0,0,0.1), mdl_datagen.basis) # and rotate it slightly about the Z-axis

# Create a set of operation sequences by constructing the usual set of experiments and using 
# "manipulate_circuits" to replace Gi with Gi2 whenever it follows Gx.  Create a 
# DataSet using the resulting Gi2-containing list of sequences.
listOfExperiments = pygsti.circuits.create_lsgst_circuits(target_model, fiducials, fiducials, germs, maxLengths)
rules = [ (("Gx","Gi") , ("Gx","Gi2")) ] # a single replacement rule: GxGi => GxGi2 
listOfExperiments = pygsti.circuits.manipulate_circuits(listOfExperiments, rules)
ds = pygsti.data.simulate_data(mdl_datagen, listOfExperiments, num_samples=10000,
                                            sample_error="binomial", seed=1234)

# Revert all the Gi2 labels back to Gi, so that the DataSet doesn't contain any Gi2 labels.
rev_rules = [ (("Gi2",) , ("Gi",)) ] # returns all Gi2's to Gi's 
ds = ds.copy_nonstatic()
ds = ds.process_circuits(lambda opstr: pygsti.circuits.manipulate_circuit(opstr,rev_rules))
ds.done_adding_data()
```

Running "standard" GST on this `DataSet` resulst in a bad fit:

```{code-cell} ipython3
target_model.set_all_parameterizations("TP")
results = pygsti.run_long_sequence_gst(ds, target_model, fiducials, fiducials,
                                      germs, maxLengths, verbosity=2)
```

So, since we have a hunch that the reason for the bad fit is that when "Gi" follows "Gx" it looks different, we can fit that data to a model that has two identity gates, call them "Gi" and "Gi2" again, and tell GST to perform the "GxGi => GxGi2" manipulation rule before computing the probability of a gate sequence:

```{code-cell} ipython3
#Create a target model which includes a duplicate Gi called Gi2
mdl_targetB = target_model.copy()
mdl_targetB['Gi2'] = target_model['Gi'].copy() # Gi2 should just be another Gi

#Run GST with:
# 1) replacement rules giving instructions how to process all of the operation sequences
# 2) data set aliases which replace labels in the *processed* strings before querying the DataSet.
rules = [ (("Gx","Gi") , ("Gx","Gi2")) ] # a single replacement rule: GxGi => GxGi2 
resultsB = pygsti.run_long_sequence_gst(ds, mdl_targetB, fiducials, fiducials,
                                       germs, maxLengths, 
                                       advanced_options={"op_label_aliases": {'Gi2': pygsti.circuits.Circuit(['Gi'])},
                                                        "string_manipulation_rules": rules},
                                       verbosity=2)
```

This gives a much better fit.  In earlier versions of pyGSTi, before certain optimizer improvements, the above cell gave a slightly better fit but not one as good as it should have (given that we know the data was generated from exactly the model being used).  This was because the (default) LGST initial model wasn't a good enought starting point, and the optimizer failed to find the global minimum.  This is no longer the case, but we keep the cell below as a demonstration of how to specify a custom guess as the starting point, which could be useful in other scenarios where the default starting point proves insufficient.  To explicitly set the starting model, we do the following:

```{code-cell} ipython3
#Create a guess, which we'll use instead of LGST - here we just
# take a slightly depolarized target.
mdl_start = mdl_targetB.depolarize(op_noise=0.01, spam_noise=0.01)

#Run GST with the replacement rules as before.
resultsC = pygsti.run_long_sequence_gst(ds, mdl_targetB, fiducials, fiducials,
                                       germs, maxLengths, 
                                       advanced_options={"op_label_aliases": {'Gi2': pygsti.circuits.Circuit(['Gi'])},
                                                         "string_manipulation_rules": rules,
                                                         "starting_point": mdl_start},
                                       verbosity=2)
```

Both results from using the model containing context dependence give a much better fit and estimate, as seen from the final `2*Delta(log(L))` numbers.

```{code-cell} ipython3
gsA = pygsti.gaugeopt_to_target(results.estimates['GateSetTomography'].models['final iteration estimate'], mdl_datagen)
gsB = pygsti.gaugeopt_to_target(resultsB.estimates['GateSetTomography'].models['final iteration estimate'], mdl_datagen)
gsC = pygsti.gaugeopt_to_target(resultsC.estimates['GateSetTomography'].models['final iteration estimate'], mdl_datagen)
gsA['Gi2'] = gsA['Gi'] #so gsA is comparable with mdl_datagen
print("Diff between truth and standard GST: ", mdl_datagen.frobeniusdist(gsA))
print("Diff between truth and context-dep GST w/LGST starting pt: ", mdl_datagen.frobeniusdist(gsB))
print("Diff between truth and context-dep GST w/custom starting pt: ", mdl_datagen.frobeniusdist(gsC))
```



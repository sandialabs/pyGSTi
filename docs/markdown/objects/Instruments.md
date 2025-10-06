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

# Instruments and Intermediate Measurements
This tutorial will demonstrate how perform tomography on models which, in addition to normal gates, contain *quantum instruments*.  Quantum instruments are maps that act on a qubit state (density matrix) and produce a qubit state along with a classical outcome.  That is, instruments are maps from $\mathcal{B}(\mathcal{H})$, the space of density matrices, to $\mathcal{B}(\mathcal{H}) \otimes K(n)$, where $K(n)$ is a classical space of $n$ elements.

In pyGSTi, instruments are represented as collections of gates, one for each classical "outcome" of the instrument.  This tutorial will demonstrate how to add instruments to `Model` objects, compute probabilities using such `Model`s, and ultimately perform tomography on them.  We'll start with a few familiar imports:

```{code-cell} ipython3
import pygsti
from pygsti.modelpacks import smq1Q_XYI as std
import numpy as np
```

## Instrument construction
Next, we'll add an instrument to our "standard" model - a 1-qubit model containing $I$, $X(\pi/2)$, and $Y(\pi/2)$ gates.  The ideal instrument will be named `"Iz"` (all instrument names must begin with `"I"`), and consist of perfect projectors onto the 0 and 1 states.  Instead of labelling the associated outcomes "0" and "1", which might me most logical, we'll name them "p0" and "p1" so it's easier to distinguish them from the final POVM outcomes which *are* labelled "0" and "1".

```{code-cell} ipython3
#Make a copy so we don't modify the original
target_model = std.target_model()

#Create and add the ideal instrument
E0 = target_model.effects['0']
E1 = target_model.effects['1']
 # Alternate indexing that uses POVM label explicitly
 # E0 = target_model['Mdefault']['0']  # 'Mdefault' = POVM label, '0' = effect label
 # E1 = target_model['Mdefault']['1']
Gmz_plus = np.dot(E0,E0.T) #note effect vectors are stored as column vectors
Gmz_minus = np.dot(E1,E1.T)
target_model[('Iz',0)] = pygsti.modelmembers.instruments.Instrument({'p0': Gmz_plus, 'p1': Gmz_minus})

#For later use, record the identity POVM vector
povm_ident = target_model.effects['0'] + target_model.effects['1'] 
```

In order to generate some simulated data later on, we'll now create a noisy version of `target_model` by depolarizing the state preparation, gates, and POVM, and also rotating the basis that is measured by the instrument and POVM.

```{code-cell} ipython3
mdl_noisy = target_model.depolarize(op_noise=0.01, spam_noise=0.01)
mdl_noisy.effects.depolarize(0.01)  #because above call only depolarizes the state prep, not the POVMs

# add a rotation error to the POVM
Uerr = pygsti.rotation_gate_mx([0,0.02,0])
E0 = np.dot(mdl_noisy['Mdefault']['0'].T,Uerr).T
E1 = povm_ident - E0
mdl_noisy.povms['Mdefault'] = pygsti.modelmembers.povms.UnconstrainedPOVM({'0': E0, '1': E1},
                                                                          evotype='default')

# Use the same rotated effect vectors to "rotate" the instrument Iz too
E0 = mdl_noisy.effects['0']
E1 = mdl_noisy.effects['1']
Gmz_plus = np.dot(E0,E0.T)
Gmz_minus = np.dot(E1,E1.T)
mdl_noisy[('Iz',0)] = pygsti.modelmembers.instruments.Instrument({'p0': Gmz_plus, 'p1': Gmz_minus})

#print(mdl_noisy) #print the model
```

## Generating probabilities 
Instrument labels (e.g. `"Iz"`) may be included within `Circuit` objects, and `Model` objects are able to compute probabilities for them just like normal (non-instrument) operation sequences.  The difference is that probabilities are labeled by tuples of instrument and POVM outcomes - referred to as **"outcome tuples"** - one for each instrument and one for the final POVM:

```{code-cell} ipython3
dict(target_model.probabilities( pygsti.circuits.Circuit(( ('Gxpi2',0) , ('Iz',0) )) ))
```

```{code-cell} ipython3
dict(target_model.probabilities( pygsti.circuits.Circuit(( ('Iz',0), ('Gxpi2',0) , ('Iz',0) )) ))
```

In fact, pyGSTi *always* labels probabilties using outcome tuples, it's just that in the non-instrument case they're always 1-tuples and by `OutcomeLabelDict` magic can be treated as if they were just strings:

```{code-cell} ipython3
probs = target_model.probabilities( pygsti.circuits.Circuit([('Gxpi2',0)]) )
print("probs = ",dict(probs))
print("probs['0'] = ", probs['0']) #This works...
print("probs[('0',)] = ", probs[('0',)]) # and so does this.
```

## Performing tomography

### Simulated data generation
Now let's perform tomography on a model that includes instruments.  First, we'll generate some data using `mdl_noisy` in exactly the same way as we would for any other model:

```{code-cell} ipython3
germs = std.germs()
germs += [pygsti.circuits.Circuit([('Iz', 0)])]  # add the instrument as a germ.

prep_fiducials = std.prep_fiducials()
meas_fiducials = std.meas_fiducials()
max_lengths = [1] # keep it simple & fast

lsgst_list = pygsti.circuits.create_lsgst_circuits(
    mdl_noisy,prep_fiducials,meas_fiducials,germs,max_lengths)

#print("LinearOperator sequences:")
#print(lsgst_list) #note that this contains LGST strings with "Iz"

#Create the DataSet
ds = pygsti.data.simulate_data(mdl_noisy,lsgst_list,1000,'multinomial',seed=2018)

#Write it to a text file to demonstrate the format:
pygsti.io.write_dataset("../../tutorial_files/intermediate_meas_dataset.txt",ds)
```

Notice the format of [intermediate_meas_dataset.txt](../../tutorial_files/intermediate_meas_dataset.txt), which includes a column for each distinct outcome tuple.  Since not all experiments contain data for all outcome tuples, the `"--"` is used as a placeholder.  Now that the data is generated, we run LGST or LSGST just like we would for any other model:

### LGST

```{code-cell} ipython3
#Run LGST
mdl_lgst = pygsti.run_lgst(ds, prep_fiducials, meas_fiducials, target_model)
#print(mdl_lgst)

#Gauge optimize the result to the true data-generating model (mdl_noisy),
# and compare.  Mismatch is due to finite sample noise.
mdl_lgst_opt = pygsti.gaugeopt_to_target(mdl_lgst,mdl_noisy)
print(mdl_noisy.strdiff(mdl_lgst_opt))
print("Frobdiff after GOpt = ",mdl_noisy.frobeniusdist(mdl_lgst_opt))
```

### Long-sequence GST
Instruments just add parameters to a `Model` like gates, state preparations, and POVMs do.  The total number of parameters in our model is 

$4$ (prep) + $2\times 4$ (2 effects) + $5\times 16$ (3 gates and 2 instrument members) $ = 92$.

```{code-cell} ipython3
target_model.num_params
```

```{code-cell} ipython3
#Run long sequence GST
results = pygsti.run_long_sequence_gst(ds,target_model,prep_fiducials,meas_fiducials,germs,max_lengths)
```

```{code-cell} ipython3
#Compare estimated model (after gauge opt) to data-generating one
mdl_est = results.estimates['GateSetTomography'].models['go0']
mdl_est_opt = pygsti.gaugeopt_to_target(mdl_est,mdl_noisy)
print("Frobdiff after GOpt = ", mdl_noisy.frobeniusdist(mdl_est_opt))
```

The same analysis can be done for a trace-preserving model, whose instruments are constrained to *add* to a perfectly trace-preserving map.  The number of parameters in the model are now:  

$3$ (prep) + $1\times 4$ (effect and complement) + $3\times 12$ (3 gates) + $(2\times 16 - 3)$ (TP instrument) $ = 71$

```{code-cell} ipython3
mdl_targetTP = target_model.copy()
mdl_targetTP.set_all_parameterizations("full TP")
print("POVM type = ",type(mdl_targetTP["Mdefault"])," Np=",mdl_targetTP["Mdefault"].num_params)
print("Instrument type = ",type(mdl_targetTP[("Iz",0)])," Np=",mdl_targetTP[("Iz",0)].num_params)
print("Number of model parameters = ", mdl_targetTP.num_params)
```

```{code-cell} ipython3
resultsTP = pygsti.run_long_sequence_gst(ds,mdl_targetTP,prep_fiducials,meas_fiducials,germs,max_lengths)
```

```{code-cell} ipython3
#Again compare estimated model (after gauge opt) to data-generating one
mdl_est = resultsTP.estimates['GateSetTomography'].models['go0']
mdl_est_opt = pygsti.gaugeopt_to_target(mdl_est,mdl_noisy)
print("Frobdiff after GOpt = ", mdl_noisy.frobeniusdist(mdl_est_opt))
```

**Thats it!**  You've done tomography on a model with intermediate measurments (instruments).

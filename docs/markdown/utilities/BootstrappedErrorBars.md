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

# An example showing how to generate bootstrapped error bars.

```{code-cell} ipython3
import os
import sys
import time
import json
import numpy as np
import pygsti
from pygsti.modelpacks import smq1Q_XY
import matplotlib.pyplot as plt
```

```{code-cell} ipython3
#Get a GST estimate (similar to Tutorial 0)

# 1) get the target Model
target_model = smq1Q_XY.target_model('full TP')

# 2) get the building blocks needed to specify which operation sequences are needed
prep_fiducials = smq1Q_XY.prep_fiducials()[0:4]
meas_fiducials = smq1Q_XY.meas_fiducials()[0:3]
germs = smq1Q_XY.germs()
maxLengths = [1,2,4,8]

# 3) generate "fake" data from a depolarized version of target_model
mdl_datagen = target_model.depolarize(op_noise=0.1, spam_noise=0.001)
listOfExperiments = pygsti.circuits.create_lsgst_circuits(
    target_model, prep_fiducials, meas_fiducials, germs, maxLengths)
ds = pygsti.data.simulate_data(mdl_datagen, listOfExperiments, num_samples=1000,
                                            sample_error="binomial", seed=1234)


results = pygsti.run_stdpractice_gst(ds, target_model, prep_fiducials, meas_fiducials,
                                    germs, maxLengths, modes="full TP")
estimated_model = results.estimates['full TP'].models['stdgaugeopt']
```

## Parametric Bootstrapping
Here we do parametric bootstrapping, as indicated by the 'parametric' argument below.
The output is eventually stored in the "mean" and "std" Models, which hold the mean and standard deviation values of the set of bootstrapped models (after gauge optimization).  It is this latter "standard deviation Model"
which holds the collection of error bars.  Note: due to print setting issues, the outputs that are printed here will not necessarily reflect the true accuracy of the estimates made.

```{code-cell} ipython3
#The number of simulated datasets & models made for bootstrapping purposes.  
# For good statistics, should probably be greater than 10.
numGatesets=10

param_boot_models = pygsti.drivers.create_bootstrap_models(
                        numGatesets, ds, 'parametric', prep_fiducials, meas_fiducials, germs, maxLengths,
                        input_model=estimated_model, target_model=target_model,
                        start_seed=0, return_data=False,
                        verbosity=2)
```

```{code-cell} ipython3
gauge_opt_pboot_models = pygsti.drivers.gauge_optimize_models(param_boot_models, estimated_model,
                                                                 plot=False) #plotting support removed w/matplotlib
```

```{code-cell} ipython3
print(gauge_opt_pboot_models[0])
```

```{code-cell} ipython3
pboot_mean = pygsti.drivers.bootstrap._to_mean_model(gauge_opt_pboot_models, estimated_model)
pboot_std  = pygsti.drivers.bootstrap._to_std_model(gauge_opt_pboot_models, estimated_model)

#Summary of the error bars
print("Parametric bootstrapped error bars, with", numGatesets, "resamples\n")
print("Error in rho vec:") 
print(pboot_std['rho0'].to_vector(), end='\n\n')
print("Error in effect vecs:")
print(pboot_std['Mdefault'].to_vector(), end='\n\n')
print("Error in Gxpi2:")
print(pboot_std['Gxpi2',0].to_vector(), end='\n\n')
print("Error in Gypi2:")
print(pboot_std['Gypi2',0].to_vector())
```

## Non-parametric Bootstrapping
Here we do non-parametric bootstrapping, as indicated by the 'nonparametric' argument below.
The output is again eventually stored in the "mean" and "std" Models, which hold the mean and standard deviation values of the set of bootstrapped models (after gauge optimization).  It is this latter "standard deviation Model"
which holds the collection of error bars.  Note: due to print setting issues, the outputs that are printed here will not necessarily reflect the true accuracy of the estimates made.

(Technical note: ddof = 1 is by default used when computing the standard deviation -- see numpy.std -- meaning that we are computing a standard deviation of the sample, not of the population.)

```{code-cell} ipython3
#The number of simulated datasets & models made for bootstrapping purposes.  
# For good statistics, should probably be greater than 10.
numModels=10

nonparam_boot_models = pygsti.drivers.create_bootstrap_models(
                          numModels, ds, 'nonparametric', prep_fiducials, meas_fiducials, germs, maxLengths,
                          target_model=target_model, start_seed=0, return_data=False, verbosity=2)
```

```{code-cell} ipython3
gauge_opt_npboot_models = pygsti.drivers.gauge_optimize_models(nonparam_boot_models, estimated_model,
                                                                 plot=False) #plotting removed w/matplotlib
```

```{code-cell} ipython3
npboot_mean = pygsti.drivers.bootstrap._to_mean_model(gauge_opt_npboot_models, estimated_model)
npboot_std  = pygsti.drivers.bootstrap._to_std_model(gauge_opt_npboot_models, estimated_model)

#Summary of the error bars
print("Non-parametric bootstrapped error bars, with", numGatesets, "resamples\n")
print("Error in rho vec:")
print(npboot_std['rho0'].to_vector(), end='\n\n')
print("Error in effect vecs:")
print(npboot_std['Mdefault'].to_vector(), end='\n\n')
print("Error in Gxpi2:")
print(npboot_std['Gxpi2',0].to_vector(), end='\n\n')
print("Error in Gypi2:")
print(npboot_std['Gypi2',0].to_vector())
```

```{code-cell} ipython3
plt.loglog(npboot_std.to_vector(),pboot_std.to_vector(),'.')
plt.loglog(np.logspace(-4,-2,10),np.logspace(-4,-2,10),'--')
plt.xlabel('Non-parametric')
plt.ylabel('Parametric')
plt.xlim((1e-4,1e-2)); plt.ylim((1e-4,1e-2))
plt.title('Scatter plot comparing param vs. non-param bootstrapping error bars.')
plt.show()
```

```{code-cell} ipython3

```

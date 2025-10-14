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

# Qutrit GST & Leakage
This notebook demonstrates how to construct the operation sequences and perform the analysis for qutrit GST when the model consists of symmetric $\pi/2$-rotations on each single qubit separately, `X`, `Y` and a 2-qubit Molmer-Sorenson gate which rotates around the `XX` axis by $\pi/2$.

```{code-cell} ipython3
import pygsti
from pygsti.models import qutrit
from pygsti.algorithms.fiducialselection import find_fiducials
from pygsti.algorithms.germselection import find_germs

from numpy import pi, array
import pickle

import numpy as np
```

First, we construct the target model.  This functionality is built into pyGSTi, so we just need to specify the single-qubit and M-S angles.
Note there are alternative approaches for building a qutrit model in pygsti using processor specification objects, but for this particular class of qutrit models in this example notebook there exist helper functions for creating the relevant models.

```{code-cell} ipython3
target_model = qutrit.create_qutrit_model(error_scale=0, x_angle=pi/2, y_angle=pi/2, ms_global=pi/2, ms_local=0, basis="qt")
#change the forward simulator for the purposes of experiment design code
target_model.sim = 'matrix'
```

Now construct the operation sequences needed by GST. Then we construct an empty dataset containing all of the necessary experimental sequences which can serve as a template for the actual experimental results.

```{code-cell} ipython3
fiducialPrep, fiducialMeasure = find_fiducials(target_model, candidate_fid_counts={4: 'all upto'}, algorithm= 'greedy')
germs = find_germs(target_model, randomize=False, candidate_germ_counts={4: 'all upto'}, mode= 'compactEVD', assume_real=True, float_type=np.double)
maxLengths = [1,2,4]
```

```{code-cell} ipython3
print("%d prep fiducials" % len(fiducialPrep))
print("%d meas fiducials" % len(fiducialMeasure))
print("%d germs" % len(germs))
```

```{code-cell} ipython3
#generate data template
expList = pygsti.circuits.create_lsgst_circuits(target_model.operations.keys(), fiducialPrep, fiducialMeasure, germs,  maxLengths)
pygsti.io.write_empty_dataset("example_files/dataTemplate_qutrit_maxL=4.txt", expList, "## Columns = 0bright count, 1bright count, 2bright count")
```

At this point **STOP** and create/fill a dataset file using the template written in the above cell.  Then proceed with the lines below to run GST on the data and create (hopefully useful) reports telling you about your gates.

```{code-cell} ipython3
mdl_datagen = target_model.depolarize(op_noise=0.05, spam_noise = .01)
DS = pygsti.data.simulate_data(mdl_datagen, expList, 1000, sample_error='multinomial', seed=2018)
```

```{code-cell} ipython3
#DS = pygsti.io.load_dataset('PATH_TO_YOUR_DATASET',cache=True) # (cache=True speeds up future loads)
```

```{code-cell} ipython3
#Run qutrit GST... which could take a while on a single CPU.  Please adjust memLimit to machine specs 
# (now 3GB; usually set to slightly less than the total machine memory)
#Setting max_iterations lower than default for the sake of the example running faster. 
target_model.sim = "matrix"
result = pygsti.run_stdpractice_gst(DS, target_model, fiducialPrep, fiducialMeasure, germs, maxLengths,
                                    verbosity=3, comm=None, mem_limit=3*(1024)**3, modes="CPTPLND",
                                    advanced_options= {'max_iterations':50})
```

```{code-cell} ipython3
#Create a report
ws = pygsti.report.construct_standard_report(
    result, "Example Qutrit Report", verbosity=3
).write_html('example_files/sampleQutritReport', auto_open=False, verbosity=3)
```

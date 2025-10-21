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

# 2Q-GST Error Bars

```{attention} Is this redundant with the error bar tutorials in utilities folder?```

```{code-cell} ipython3
import pygsti
import time

#If we were using MPI
# from mpi4py import MPI
# comm = MPI.COMM_WORLD
comm = None

#Load the 2-qubit results (if you don't have this directory, run the 2Q-GST example)
results = pygsti.io.read_results_from_dir("../../example_files/My2QExample", "StandardGST")
```

```{code-cell} ipython3
# error bars in reports require the presence of a fully-initialized
# "confidence region factory" within the relevant Estimate object.
# In most cases "fully-initialized" means that a Hessian has been 
# computed and projected onto the non-gauge space.

# Choose forward simulator and how computation should be split up. Here are a couple examples:
#results.estimates['CPTP'].models['stdgaugeopt'].sim = pygsti.forwardsims.MapForwardSimulator(num_atoms=100)
results.estimates['CPTPLND'].models['stdgaugeopt'].sim = pygsti.forwardsims.MatrixForwardSimulator(param_blk_sizes=(30,30))

# initialize a factory for the 'go0' gauge optimization within the 'default' estimate
crfact = results.estimates['CPTPLND'].add_confidence_region_factory('stdgaugeopt', 'final')
```

```{code-cell} ipython3
:tags: [nbval-skip]

crfact.compute_hessian(comm=comm, mem_limit=3.0*(1024.0)**3) #optionally use multiple processors & set memlimit
crfact.project_hessian('intrinsic error')
```

Note above cell was executed for demonstration purposes, and was **keyboard-interrupted intentionally** since it would have taken forever on a single processor.

```{code-cell} ipython3
:tags: [nbval-skip]

#write results back to disk
results.write()
```

```{code-cell} ipython3

```

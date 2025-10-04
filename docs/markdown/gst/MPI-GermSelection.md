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

# Germ Selection
The code below should be put into a script and run using mpiexec.  It's primary function is to pass a MPI Comm object to `pygsti.algorithms.germselection.build_up_breadth`.

```{code-cell} ipython3
from __future__ import print_function
import time

import pygsti
from pygsti.modelpacks import smq2Q_XYICNOT
from pygsti.algorithms import germselection as germsel
```

```{code-cell} ipython3
:tags: [nbval-skip]

from mpi4py import MPI
comm = MPI.COMM_WORLD
```

```{code-cell} ipython3
def do_greedy_germsel(target_model, forced_germs, candidate_counts,
                      seedStart, outFilename, comm):
    #candidate_counts is a dict of keys = germ lengths, values = # of germs at that length                                                            

    tStart = time.time()

    candidate_germs = []
    for i,(germLength, count) in enumerate(candidate_counts.items()):
        if count == "all upto":
            candidate_germs.extend( pygsti.circuits.list_all_circuits_without_powers_and_cycles(
                    target_model.operations.keys(), max_length=germLength) )
        else:
            candidate_germs.extend( pygsti.circuits.list_random_circuits_onelen(
                    target_model.operations.keys(), germLength, count, seed=seedStart+i))

    available_germs = pygsti.tools.remove_duplicates( forced_germs + candidate_germs )
    print("%d available germs" % len(available_germs))
    germs = germsel.find_germs_breadthfirst(target_model, available_germs,
                     randomization_strength=1e-3, num_copies=3, seed=1234,
                     op_penalty=10.0, score_func='all', tol=1e-6, threshold=1e5,
                     pretest=False, force=forced_germs, verbosity=5, comm=comm, mem_limit=0.5*(1024**3))

    if comm is None or comm.Get_rank() == 0:
        print("Germs (%d) = \n" % len(germs), "\n".join(map(str,germs)))
        print("Total time = %mdl" % (time.time()-tStart))
        pickle.dump(germs,open(outFilename,"wb"))
    return germs
```

```{code-cell} ipython3
#2Q case                                                                                                                                              
target_model = smq2Q_XYICNOT.target_model()
forced_germs = pygsti.circuits.to_circuits([(gl,) for gl in target_model.operations.keys()]) #singletons                                                                                      
candidate_counts = { 3:"all upto", 4:30, 5:20, 6:20, 7:20, 8:20} # germLength:num_candidates                                                          
seedStart = 4
```

```{code-cell} ipython3
---
jupyter:
  outputs_hidden: true
tags: [nbval-skip]
---
do_greedy_germsel(target_model, forced_germs, candidate_counts,
                  seedStart, "germs_EXAMPLE.pkl", comm)
```

Above is **keyboard-interrupted on purpose**, as this output was produced with a single processor and it would have taken a very long time.

```{code-cell} ipython3

```

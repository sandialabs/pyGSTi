---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.19.3
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Clifford RB

This tutorial contains a few details on how to run [Clifford randomized benchmarking](http://journals.aps.org/prl/abstract/10.1103/PhysRevLett.106.180504) that are not covered in the [RB overview tutorial](HowRBWorks). 

## What is Clifford RB? 

By Clifford randomized benchmarking we mean RB of the $n$-qubit Clifford group, as defined by Magesan *et al.* in [*Scalable and Robust Benchmarking of Quantum Processes*](http://journals.aps.org/prl/abstract/10.1103/PhysRevLett.106.180504). This protocol is routinely run on 1 and 2 qubits.

```{code-cell} ipython3
import pygsti
from pygsti.processors import QubitProcessorSpec as QPS
from pygsti.processors import CliffordCompilationRules as CCR
import numpy as np
```

## Creating a Clifford RB experiment design

The only aspects of running Clifford RB with pyGSTi that are not covered in the [RB overview tutorial](HowRBWorks) are some subtleties in generating a Clifford RB experiment design (and what those subtleties mean for interpretting the results). To cover these subtleties, here we go through the inputs used to generate a Clifford RB experiment design in more detail.

### 1. Generic RB inputs

`pspec` (the device to benchmark), `k` (how many circuits to sample at each depth) and `qubits` (which qubits to benchmark) mean the same thing here as in every RB protocol; [how RB works](HowRBWorks) introduces them. What is protocol-specific is `depths`. For Clifford RB on $n$ qubits, the RB depth is the number of (uncompiled) $n$-qubit Clifford gates in the sequence minus two. That convention is chosen so that zero is the minimum RB depth for every RB method in pyGSTi.

Every other argument to the Clifford RB experiment design generation function is optional.

```{code-cell} ipython3
n_qubits = 2
qubit_labels = ['Q0','Q1'] 
gate_names = ['Gxpi2', 'Gxmpi2', 'Gypi2', 'Gympi2', 'Gcphase'] 
availability = {'Gcphase':[('Q0','Q1')]}
pspec = QPS(n_qubits, gate_names, availability=availability, qubit_labels=qubit_labels)

compilations = {'absolute': CCR.create_standard(pspec, 'absolute', ('paulis', '1Qcliffords'), verbosity=0),            
                'paulieq': CCR.create_standard(pspec, 'paulieq', ('1Qcliffords', 'allcnots'), verbosity=0)}

depths = [0,1,2,4,8]
k = 10
qubits = ['Q0','Q1']
```

### 2. The target output
In the standard formulation of Clifford RB, the circuit should always return the all-zeros bit-string if there is no errors. But it can be useful to randomized the "target" bit-string (e.g., then the asymptote in the RB decay is fixed to $1/2^n$ even with biased measurement errors). This randomization is specified via the `randomizeout` argument, and it defaults to `False` (the standard protocol).

```{code-cell} ipython3
randomizeout = True
```

### 3. The Clifford compilation algorithm
To generate a Clifford RB circuit in terms of native gates, it is necessary to decompose each $n$-qubit Clifford gate into the native gates. pyGSTi has a few different Clifford gate compilation algorithms, that can be accessed via the `compilerargs` optional argument. Note: **The Clifford RB error rate is compiler dependent!** So it is not possible to properly interpret the Clifford RB error rate without understanding at least some aspects of the compilation algorithm (e.g., the mean two-qubit gate count in a compiled $n$-qubit Clifford circuit). This is one of the reasons that [Direct RB](DirectRB) is arguably a preferable method to Clifford RB.

None of the Clifford compilation algorithms in pyGSTi are a simple look-up table with some optimized property (e.g., minimized two-qubit gate count or depth). Look-up tables like this are typically used for 1- and 2-qubit Clifford RB experiments, but we instead used a method that scales to many qubits.

There are multiple compilation algorithms in pyGSTi, and the algorithm can be set using the `compilerargs` argument (see the `pygsti.algorithms.compile_clifford` function for some details on the available algorithms, and the `CliffordRBDesign` docstring for how to specify the desired algorithm). The default algorthm is the one that we estimate to be our "best" algorithm in the regime of 1-20ish qubits. This algorithm (and some of the other algorithms) are randomized. So when creating a `CliffordRBDesign` you can also specify the number of randomization, via `citerations`. Increasing this will reduce the average depth and two-qubit gate count of each $n$-qubit Clifford gate, up to a point, making Clifford RB feasable on more qubits. 
But note that time to generate the circuits can increase quickly as `citerations` increases (because a depth $m$ circuit contains $(m+2)$ $n$-qubit Clifford gates to compile).

```{code-cell} ipython3
citerations = 20
```

From here, everything proceeds as in the RB overview tutorial (except for adding in the optional arguments).

```{code-cell} ipython3
# Here we construct an error model with 1% local depolarization on each qubit after each gate.
def simulate_taking_data(data_template_filename):
    """Simulate taking data and filling the results into a template dataset.txt file"""
    noisemodel = pygsti.models.create_crosstalk_free_model(pspec, depolarization_strengths={g:0.01 for g in pspec.gate_names})
    noisemodel.sim = 'map'
    pygsti.io.fill_in_empty_dataset_with_fake_data(data_template_filename, noisemodel, num_samples=1000, seed=1234)
```

```{code-cell} ipython3
design = pygsti.protocols.CliffordRBDesign(pspec, compilations, depths, k, qubit_labels=qubits, 
                                           randomizeout=randomizeout, citerations=citerations)
pygsti.io.write_empty_protocol_data('../../../tutorial_files/test_crb_dir', design, clobber_ok=True)

# -- fill in the dataset file in ../../../tutorial_files/test_crb_dir/data/dataset.txt --
simulate_taking_data('../../../tutorial_files/test_crb_dir/data/dataset.txt') # REPLACE with actual data-taking

data = pygsti.io.read_data_from_dir('../../../tutorial_files/test_crb_dir')

protocol = pygsti.protocols.RB() 
results = protocol.run(data)
ws = pygsti.report.Workspace()
ws.init_notebook_mode(autodisplay=True)
rb_fig = ws.RandomizedBenchmarkingPlot(results)
```

```{code-cell} ipython3
#If the figure doesn't appear in the output above, try uncommenting the contents of this cell and running it.
#rb_fig.figs[0].plotlyfig
```

Clifford RB is also the subroutine behind *interleaved* RB, which estimates the error rate of
one Clifford of interest by comparing a plain CRB decay against one with that Clifford
interleaved. That protocol has its own page: [Interleaved RB](InterleavedRB).

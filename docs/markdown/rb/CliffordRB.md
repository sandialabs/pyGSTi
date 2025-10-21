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

# Clifford RB

+++

This tutorial contains a few details on how to run [Clifford Randomized Benchmarking](http://journals.aps.org/prl/abstract/10.1103/PhysRevLett.106.180504) that are not covered in the [RB overview tutorial](Overview). 


## What is Clifford RB? 

By Clifford randomized benchmarking we mean RB of the $n$-qubit Clifford group, as defined by Magesan *et al.* in [*Scalable and Robust Benchmarking of Quantum Processes*](http://journals.aps.org/prl/abstract/10.1103/PhysRevLett.106.180504). This protocol is routinely run on 1 and 2 qubits.

```{code-cell} ipython3
import pygsti
from pygsti.processors import QubitProcessorSpec as QPS
from pygsti.processors import CliffordCompilationRules as CCR
import numpy as np
```

## Creating a Clifford RB experiment design

The only aspects of running Clifford RB with pyGSTi that are not covered in the [RB overview tutorial](Overview) are some subtleties in generating a Clifford RB experiment design (and what those subtleties mean for interpretting the results). To cover these subtleties, here we go through the inputs used to generate a Clifford RB experiment design in more detail.

### 1. Generic RB inputs

The first inputs to create an RB experiment design are the same as in all RB protocols, and these are covered in the [RB overview tutorial](Overview). They are:

- The device to benchmark (`pspec`).
- The "RB depths" at which we will sample circuits (`depths`). For Clifford RB on $n$ qubits, the RB depth is the number of (uncompiled) $n$-qubit Clifford gates in the sequence minus two. This convention is chosen so that zero is the minimum RB depth for all RB methods in pyGSTi.
- The number of circuits to sample at each length (`k`).
- The qubits to benchmark (`qubits`).

All other arguments to Clifford RB experiment design generation function are optional.

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
pygsti.io.write_empty_protocol_data('../../tutorial_files/test_crb_dir', design, clobber_ok=True)

# -- fill in the dataset file in tutorial_files/test_rb_dir/data/dataset.txt --
simulate_taking_data('../../tutorial_files/test_crb_dir/data/dataset.txt') # REPLACE with actual data-taking

data = pygsti.io.read_data_from_dir('../../tutorial_files/test_crb_dir')

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

## Interleaved Randomized Benchmarking
In this subsection we'll discuss built-in support for performing interleaved randomized benchmarking (IRB). IRB is a method for estimating the error rate of a particular clifford of interest using CRB as a subroutine.

```{code-cell} ipython3
from pygsti.protocols.rb import InterleavedRBDesign, InterleavedRandomizedBenchmarking
from pygsti.circuits import Circuit
from pygsti.baseobjs import Label
```

The creation of an IRB design largely follows that of CRB, with the addition of the specification of an interleaved circuit. That is, the clifford which we want to estimate the individual error rate for.

```{code-cell} ipython3
n_qubits = 1
qubit_labels = ['Q0']
gate_names = ['Gxpi2', 'Gxmpi2', 'Gypi2', 'Gympi2']
pspec = QPS(n_qubits, gate_names, qubit_labels=qubit_labels)
compilations = {'absolute': CCR.create_standard(pspec, 'absolute', ('paulis', '1Qcliffords'), verbosity=0),            
                'paulieq': CCR.create_standard(pspec, 'paulieq', ('1Qcliffords', 'allcnots'), verbosity=0)}
depths = [0,1,2,4,8,16,32]
k = 50
interleaved_circuit = Circuit([Label('Gxpi2', 'Q0')], line_labels=('Q0',))
```

```{code-cell} ipython3
irb_design = InterleavedRBDesign(pspec, compilations, depths, k, interleaved_circuit, qubit_labels)
```

`InterleavedRBDesign` is structured somewhat differently than `CliffordRBDesign`, instead acting as a container class which constructs and stores a pair of CRB experiment designs (one interleaved with the specified `interleaved_circuit`) with settings as specified by the given arguments. `InterleavedRBDesign` is a subclass of the more general `CombinedExperimentDesign`, and like `CombinedExperimentDesign` its child subdesigns can be accessed by indexing into it like a dictionary, as shown below.

```{code-cell} ipython3
print(irb_design.keys())
print(irb_design['crb'])
```

Here we construct an error model with 1% local depolarization on each qubit after each one-qubit gate, except for Gxpi2 which has a 2% depolarization rate.

```{code-cell} ipython3
def simulate_taking_data_irb(data_template_filename):
    """Simulate taking data and filling the results into a template dataset.txt file"""
    depolarization_strengths={g:0.01 for g in pspec.gate_names if g!= 'Gxpi2'}
    depolarization_strengths['Gxpi2'] = .02
    noisemodel = pygsti.models.create_crosstalk_free_model(pspec, depolarization_strengths=depolarization_strengths)
    noisemodel.sim = 'map'
    pygsti.io.fill_in_empty_dataset_with_fake_data(data_template_filename, noisemodel, num_samples=1000, seed=1234)
```

```{code-cell} ipython3
pygsti.io.write_empty_protocol_data('../../tutorial_files/test_irb_dir', irb_design, clobber_ok=True)

# -- fill in the dataset file in tutorial_files/test_rb_dir/data/dataset.txt --
simulate_taking_data_irb('../../tutorial_files/test_irb_dir/data/dataset.txt') # REPLACE with actual data-taking
data_irb = pygsti.io.read_data_from_dir('../../tutorial_files/test_irb_dir')
```

```{code-cell} ipython3
protocol_irb = InterleavedRandomizedBenchmarking()
results_irb = protocol_irb.run(data_irb)
```

Now that we have the results we can index into them to get the estimated IRB numbers and bounds. In this context, 'bounds' really refers to the half-width of the bounds as described in equation 5 of the original IRB paper from Magesan et al. https://arxiv.org/pdf/1203.4550.
The object that is returned by `InterleavedRandomizedBenchmarking` is a so-called `ProtocolResultsDir`, and this object stores both the IRB specific estimates as well as the results objects associated with each of the subexperiments used to perform IRB. This makes extracting the values slightly more cumbersome than usual, but ensures that the relevant results remain grouped together at all times. Below we show how to access the IRB numbers and bounds.

```{code-cell} ipython3
results_irb.for_protocol['InterleavedRandomizedBenchmarking'].irb_numbers
```

```{code-cell} ipython3
results_irb.for_protocol['InterleavedRandomizedBenchmarking'].irb_bounds
```

To access the results objects of the standard and interleaved CRB experiments that we performed we can index into `results_irb` like a dictionary. The relevant keys are 'crb' and 'icrb', respectively.

```{code-cell} ipython3
results_irb['crb'].for_protocol['RandomizedBenchmarking']
```

```{code-cell} ipython3
results_irb['icrb'].for_protocol['RandomizedBenchmarking']
```

From which we can access various information about the fits as well as other useful RB related estimates. E.g. below we extract the rb number and the estimated exponential decay parameters for one of the RB fits performed on the CRB subexperiment|.

```{code-cell} ipython3
print(results_irb['crb'].for_protocol['RandomizedBenchmarking'].fits['full'].estimates)
```

Finally, we'll note that this results object for IRB can be written to and read from disk using the `write` method and the function `pygsti.io.read_results_from_dir`, respectively.

```{code-cell} ipython3
results_irb.write('../../tutorial_files/test_irb_results')
```

```{code-cell} ipython3
irb_results_from_disk = pygsti.io.read_results_from_dir('../../tutorial_files/test_irb_results')
print(irb_results_from_disk['crb'].for_protocol['RandomizedBenchmarking'].fits['full'].estimates)
#As expected these values are the same as above when we accessed them in `results_irb`
```

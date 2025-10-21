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

# Mirror RB

+++

This tutorial contains a few details on how to run *Mirror Randomized Benchmarking* that are not covered in the [RB overview tutorial](Overview).

## What is Mirror RB? 

Like Direct RB, Mirror RB is a streamlined RB method partly inspired by [Clifford RB](CliffordRB). It has the same core purpose as Clifford RB - quantifying average gate performance - but it is feasable on more qubits, and it provides more directly useful information. However, Mirror RB is even more streamlined than Direct RB, making it feasable on 10s or 100s of qubits (it is possible to holistically benchmark around $1/\epsilon$ qubits if the error rate per-gate per-qubit is around $\epsilon$).

Mirror RB can be implemented with non-Clifford gates as well (see the [Universal Gate Set MRB tutorial](MirrorRB-Universal-Gate-Sets)). A depth $m$ ($m\geq 0$) mirror RB circuit consists of:

1. A uniformly random 1-qubit Clifford gate on every qubit. 
2. A "compute" circuit consisting of $m/2$ independently sampled layers of the native Clifford gates in the device, sampled according to a user-specified distribution $\Omega$. Each of these layers is proceeded by a uniformly random Pauli gate on each qubit.
3. A layer of uniformly random Pauli gates.
4. An "uncompute" circuit consisting of the $m/2$ layers from step (2) in the reverse order with each gate replaced with its inverse. Each of these layers is followed by a uniformly random Pauli gate on each qubit, with these Pauli gates sampled *independently* of the Pauli layers in step (2).
5. The inverse of the random 1-qubit Clifford gates in step (1).

This construction means that Mirror RB circuits can be much shorter than Clifford RB circuits, or Direct RB circuits. Yet they still have the core randomization properties of both Clifford and Direct RB.

**More information on Mirror RB will be added to this tutorial in a future release.**

```{code-cell} ipython3
from __future__ import print_function #python 2 & 3 compatibility
import pygsti
from pygsti.processors import QubitProcessorSpec as QPS
from pygsti.processors import CliffordCompilationRules as CCR
```

## Creating a Mirror RB experiment design

Generating a Mirror RB experiment design is very similar to creating a Direct RB design. The only difference is that there is no compilation in a Mirror RB circuit, so there is no compilation algorithm to tweak.

### 1. Generic RB inputs

The first inputs to create a Mirror RB experiment design are the same as in all RB protocols, and these are covered in the [RB overview tutorial](Overview). They are:

- The device to benchmark (`pspec`).
- The "RB depths" at which we will sample circuits (`depths`). For Mirror RB, these depths must be even integers. They correspond to the number of total layers in the "compute" and "uncompute" sub-circuits (but where we don't include the randomized Pauli gates in the layer count). 
- The number of circuits to sample at each length (`k`).
- The qubits to benchmark (`qubits`).

```{code-cell} ipython3
# Mirror RB can be run on many many more qubit than this, but this notebook creates simulated data. As 
# we are using a full density matrix simulator this limits the number of qubits we can use here.
n_qubits = 4
qubit_labels = ['Q'+str(i) for i in range(n_qubits)] 
gate_names = ['Gi', 'Gxpi2', 'Gxpi', 'Gxmpi2', 'Gypi2', 'Gypi', 'Gympi2', 
              'Gzpi2', 'Gzpi', 'Gzmpi2', 'Gcphase'] 
availability = {'Gcphase':[('Q'+str(i),'Q'+str((i+1) % n_qubits)) for i in range(n_qubits)]}
pspec = QPS(n_qubits, gate_names, availability=availability, qubit_labels=qubit_labels)

compilations = {'absolute': CCR.create_standard(pspec, 'absolute', ('paulis', '1Qcliffords'), verbosity=0),            
                'paulieq': CCR.create_standard(pspec, 'paulieq', ('1Qcliffords', 'allcnots'), verbosity=0)}

depths = [0, 2, 4, 8, 16, 32]
k = 40
qubits = qubit_labels
```

All other arguments to the Mirror RB experiment design generation function `MirrorRBDesign` are optional. But, as with Direct RB, to make the most out of Mirror RB it is typically important to at least understand the circuit layer sampling.

### 2. The circuit layer sampler
Exactly as with Direct RB, the Mirror RB circuit layer sampling distribution $\Omega$ is perhaps the most important input to the Mirror RB experiment design. This is because, by construction, the Mirror RB error rate $r$ is $\Omega$-dependent. This $\Omega$-dependence is useful, because by carefully choosing or varying $\Omega$ we can learn a lot about device performance. But it also means that the $\Omega$ has to be carefully chosen! At the very least, **you need to know what sampling distribution you are using in order to interpret the results!**

This might seem like a drawback in comparison to Clifford RB, but note that this $\Omega$-dependence is analogous to the Clifford-compiler dependence of the Clifford RB error rate (with the advantage that it is more easily controlled and understood). And Mirror RB can be run on many, many more qubits!

The sampling distribution is specified via the optional arguements `sampler` and `samplerargs`. Here we use what we call the "edge grab" sampler. 

Because both Direct and Mirror RB have the this sampling-distribution dependence, there is a separate [random circuit sampling tutorial](Samplers) that introduces the different built-in sampling algorithms within pyGSTi (which includes details of the "edge grab" algorithm).

```{code-cell} ipython3
sampler = 'edgegrab'
samplerargs = [0.5]
```

From here, generating the design and collecting data proceeds as in the RB overview tutorial.

```{code-cell} ipython3
qubit_error_rate = 0.002
def simulate_taking_data(data_template_filename):
    """Simulate taking data and filling the results into a template dataset.txt file"""
    error_rates = {}
    for gn in pspec.gate_names:
        n = pspec.gate_num_qubits(gn)
        gate_error_rate = n * qubit_error_rate
        error_rates[gn] = [gate_error_rate/(4**n - 1)] * (4**n - 1)
    noisemodel = pygsti.models.create_crosstalk_free_model(pspec, stochastic_error_probs=error_rates)
    pygsti.io.fill_in_empty_dataset_with_fake_data(data_template_filename, noisemodel, num_samples=1000, seed=1234)
```

```{code-cell} ipython3
design = pygsti.protocols.MirrorRBDesign(pspec, depths, k, qubit_labels=qubits, sampler=sampler, 
                                         clifford_compilations=compilations, samplerargs=samplerargs)

pygsti.io.write_empty_protocol_data('../../tutorial_files/test_mrb_dir', design, clobber_ok=True)

# -- fill in the dataset file in tutorial_files/test_rb_dir/data/dataset.txt --
simulate_taking_data('../../tutorial_files/test_mrb_dir/data/dataset.txt') # REPLACE with actual data-taking

data = pygsti.io.read_data_from_dir('../../tutorial_files/test_mrb_dir')
```

## Running the Mirror RB protocol
As with all RB methods in pyGSTi, to analyze the data we instantiate an `RB` protocol and `.run` it on our data object.  However, there is a slight difference for Mirror RB. Mirror RB doesn't fit simple success/fail format data: instead it fits what we call *Hamming weight adjusted success probabilities* to an exponential decay ($P_m = A + B p^m$ where $P_m$ is the average adjusted success probability at RB length $m$). 

To obtain this data analysis we simply specify the data type when instantiate an `RB` protocol: we set `datatype = adjusted_success_probabilities`.

```{code-cell} ipython3
protocol = pygsti.protocols.RB(datatype = 'adjusted_success_probabilities', defaultfit='A-fixed')
results = protocol.run(data)
ws = pygsti.report.Workspace()
ws.init_notebook_mode(autodisplay=True)
ws.RandomizedBenchmarkingPlot(results)
```

```{code-cell} ipython3
# The error rate we *approximately* expect accord to Mirror RB theory
print(1 - (1 - qubit_error_rate)**(2 * len(qubits)))
```

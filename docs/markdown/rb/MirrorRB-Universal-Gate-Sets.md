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

# MRB with Universal Gate Sets

This tutorial contains a few details on how to run *Mirror randomized benchmarking* with universal gate sets, that are not covered in the [RB overview tutorial](Overview) or the [Clifford MRB tutorial](MirrorRB).

## What is Mirror RB? 

Mirror RB is a streamlined, computationally-efficient RB method. It has the same core purpose as Clifford RB - quantifying average gate performance - but it is feasable on more qubits, and it provides more directly useful information. Unlike oter RB protocols, Mirror RB can be implemented with non-Clifford gates on many qubits. The general structure of MRB circuits with non-Clifford gates is similar to that of MRB circuits with Clifford gates. The structure of a depth $m$ ($m\geq 0$) mirror RB circuit is:
1. A Haar-random 1-qubit gate (or random 1-qubit Clifford gate) on every qubit. 
2. A "compute" circuit consisting of $m/2$ independently sampled layers of gates, sampled according to a user-specified distribution $\Omega$. Each of these layers is a *composite layer* consisting of (1) randomly-sampled native two-qubit gates, followed by (2) Haar-random 1-qubit gates (or random 1-qubit Clifford gates) on each qubit. Variations on this structure are possible, but they are not currently implemented in pyGSTi. 
4. An "uncompute" circuit consisting of the $m/2$ layers from step (2) in the reverse order with each gate replaced with its inverse. 
5. The inverse of the random 1-qubit gates in step (1).
This circuit then undergoes a version of randomized compilation to get the final circuit, in which random Pauli gates are compiled into the composite layers. 

See [Demonstrating scalable randomized benchmarking of universal gate sets](https://arxiv.org/abs/2207.07272) for further details on MRB with universal gate sets.

```{code-cell} ipython3
from __future__ import print_function #python 2 & 3 compatibility
import pygsti
from pygsti.processors import QubitProcessorSpec as QPS
from pygsti.processors import CliffordCompilationRules as CCR
```

## Creating a Mirror RB experiment design

Generating a Mirror RB experiment design for universal gate sets is very similar to creating an experiment design for other RB methods or for Clifford Mirror RB. 

### 1. Generic RB inputs

The first inputs to create a Mirror RB experiment design are the same as in all RB protocols, and these are covered in the [RB overview tutorial](Overview). They are:

- The device to benchmark (`pspec`). Universal gate set MRB in pyGSTi currently requires the `Gzr` and `Xpi2` gates to be in the list of gate names.
- The "RB depths" at which we will sample circuits (`depths`). For Mirror RB, these depths must be even integers. They correspond to the number of total layers in the "compute" and "uncompute" sub-circuits.
- The number of circuits to sample at each length (`k`).
- The qubits to benchmark (`qubits`).

```{code-cell} ipython3
# Mirror RB can be run on many many more qubit than this, but this notebook creates simulated data. As 
# we are using a full density matrix simulator this limits the number of qubits we can use here.
n_qubits = 3
qubit_labels = ['Q'+str(i) for i in range(n_qubits)] 
gate_names = ['Gi', 'Gxpi2', 'Gzr', 'Gcphase'] 
availability = {'Gcphase':[('Q'+str(i),'Q'+str((i+1) % n_qubits)) for i in range(n_qubits)]}
pspec = QPS(n_qubits, gate_names, availability=availability, qubit_labels=qubit_labels)

depths = [0, 2, 4, 8, 16, 32]
k = 40
qubits = qubit_labels
```

Mirror RB is implemented in pyGSTi for specific random circuit structures, specified via the option argument `circuit_type`:
1. `clifford+zxzxz-haar`: Clifford two-qubit gates and Haar-random 1-qubit gates
2. `clifford+zxzxz-clifford`: Clifford two-qubit gates and random Clifford 1-qubit gates (this is *not* a universal gate set)
3. `cz(theta)+zxzxz-haar`: Controlled Z axis rotation gates CZ(theta) and Haar-random 1-qubit gates. This option requires that both CZ(theta) and CZ(-theta) are in the allowed two-qubit gates. 

All single-qubit gates are decomposed into a series of five single qubit gates: Two $X_{\pi/2}$ gates and three $Z(\theta)$ gates. 

### 2. The circuit layer sampler
As with Clifford Mirror RB, the Mirror RB error rate $r$ depends on the circuit layer sampling distribution $\Omega$. This $\Omega$-dependence is useful, because by carefully choosing or varying $\Omega$ we can learn a lot about device performance. But it also means that the $\Omega$ has to be carefully chosen! At the very least, **you need to know what sampling distribution you are using in order to interpret the results!**

Universal gate set MRB in pyGSTi uses the "edge grab" sampler. The frequency of two-qubit gates in the layers can be changed via the optional arguement `samplerargs`.

```{code-cell} ipython3
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
design = pygsti.protocols.MirrorRBDesign(pspec, depths, k, qubit_labels=qubits,
                                         circuit_type='clifford+zxzxz-haar', samplerargs=samplerargs)

pygsti.io.write_empty_protocol_data('../../tutorial_files/test_mrb_dir', design, clobber_ok=True)

# -- fill in the dataset file in tutorial_files/test_rb_dir/data/dataset.txt --MirrorRBDesign
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

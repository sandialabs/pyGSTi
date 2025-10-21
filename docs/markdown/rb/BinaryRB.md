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

# Binary RB

+++

This tutorial contains a few details on how to run *Binary Randomized Benchmarking (BiRB)* that are not covered in the [RB overview tutorial](Overview).

## What is Binary RB? 

Binary RB is a streamlined RB method that draws upon the strengths of [Direct RB](DirectRB), but uses a highly gate-efficient state preparation and measurement method that allows it to run on many, many more qubits. It has the same core purpose as Clifford RB - quantifying average gate performance - but it is feasable on more qubits, and it provides more directly useful information. BiRB is feasable on 10s or 100s of qubits (it is possible to holistically benchmark around $1/\epsilon$ qubits if the error rate per-gate per-qubit is around $\epsilon$).
 
A depth $m$ ($m\geq 0$) Binary RB circuit consists of:

1. A layer of random single-qubit gates that prepare a tensor product state that stabilizes a random (non-Identity) Pauli operator.
2. A "core" circuit consisting of $m$ independently sampled layers of the native Clifford gates in the device, sampled according to a user-specified distribution $\Omega$. 
3. A layer of single-qubit gates that transforms the evolved Pauli into a tensor product of Z and I operators. 

Each circuit has an associated target Pauli that gets measured at the end. The results of computational basis measurements are processed to determine the result of measuring the target Pauli.  

Binary RB circuits are much shorter than Direct or Clifford RB circuits, but they retain the core randomization properties of both Clifford and Direct RB circuits, and they have a simpler structure than Mirror RB circuits.

```{code-cell} ipython3
from __future__ import print_function #python 2 & 3 compatibility
import pygsti
from pygsti.processors import QubitProcessorSpec as QPS
from pygsti.processors import CliffordCompilationRules as CCR
```

## Creating a Binary RB experiment design

Generating a Mirror RB experiment design is very similar to creating a Direct RB design. The only difference is that there is no compilation in a Mirror RB circuit, so there is no compilation algorithm to tweak.

### 1. Generic RB inputs

The first inputs to create a Binary RB experiment design are the same as in all RB protocols, and these are covered in the [RB overview tutorial](Overview). They are:

- The device to benchmark (`pspec`).
- The "RB depths" at which we will sample circuits (`depths`), which must be nonnegative integers. These correspond to the number of randomly sampled core layers in a circuit. 
- The number of circuits to sample at each length (`k`).
- The qubits to benchmark (`qubits`).

```{code-cell} ipython3
# BiRB can be run on many many more qubit than this, but this notebook creates simulated data. As 
# we are using a full density matrix simulator this limits the number of qubits we can use here.
n_qubits = 4
qubit_labels = ['Q'+str(i) for i in range(n_qubits)] 
gate_names = ['Gi', 'Gxpi2', 'Gxpi', 'Gxmpi2', 'Gypi2', 'Gypi', 'Gympi2', 
              'Gzpi2', 'Gzpi', 'Gzmpi2', 'Gcphase'] 
availability = {'Gcphase':[('Q'+str(i),'Q'+str((i+1) % n_qubits)) for i in range(n_qubits)]}
pspec = QPS(n_qubits, gate_names, availability=availability, qubit_labels=qubit_labels)

compilations = {'absolute': CCR.create_standard(pspec, 'absolute', ('paulis', '1Qcliffords'), verbosity=0),            
                'paulieq': CCR.create_standard(pspec, 'paulieq', ('1Qcliffords', 'allcnots'), verbosity=0)}

depths = [0, 2, 4, 8, 16, 32, 64, 128]
k = 40
qubits = qubit_labels
```

All other arguments to the Binary RB experiment design generation function `BinaryRBDesign` are optional. But, as with Direct RB and Mirror RB, it is important to understand the layer sampling.

### 2. The circuit layer sampler
Exactly as with Direct  and Mirror RB, the circuit layer sampling distribution $\Omega$ is perhaps the most important input to the Binary RB experiment design. This is because, by construction, the BiRB error rate $r$ is $\Omega$-dependent. This $\Omega$-dependence is useful, because by carefully choosing or varying $\Omega$ we can learn a lot about device performance. But it also means that the $\Omega$ has to be carefully chosen! At the very least, **you need to know what sampling distribution you are using in order to interpret the results!**

This might seem like a drawback in comparison to Clifford RB, but note that this $\Omega$-dependence is analogous to the Clifford-compiler dependence of the Clifford RB error rate (with the advantage that it is more easily controlled and understood). And Binary RB can be run on many, many more qubits!

The structure of the circuit layers is specificed via the option argument `layertype`, which currently supports two layer structures. 
1. `mixed1q2q`: Each layer consists of a mixture of single- and two-qubit gates. This is analogous to the default in a DirectRBDesign.
2. `alternating1q2q`: Each layer consists of two parts: First, random single-qubit gates on every qubit, then random two-qubit gates on a subset of the qubits.

Further details of the sampling distribution are specified via the optional arguements `sampler` and `samplerargs`.

Because Direct, Mirror, and Binary RB have the this sampling-distribution dependence, there is a separate [random circuit sampling tutorial](Samplers) that introduces the different built-in sampling algorithms within pyGSTi.

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
design = pygsti.protocols.BinaryRBDesign(pspec, compilations['absolute'], depths, k, qubit_labels=qubits, sampler=sampler, 
                                            samplerargs=samplerargs, layer_sampling='mixed1q2q')

pygsti.io.write_empty_protocol_data('../../tutorial_files/test_birb_dir', design, clobber_ok=True)

# -- fill in the dataset file in tutorial_files/test_rb_dir/data/dataset.txt --
simulate_taking_data('../../tutorial_files/test_birb_dir/data/dataset.txt') # REPLACE with actual data-taking

data = pygsti.io.read_data_from_dir('../../tutorial_files/test_birb_dir')
```

## Running the Binary RB protocol
As with all RB methods in pyGSTi, to analyze the data we instantiate an `RB` protocol and `.run` it on our data object.  However, the data analysis for BiRB is different from that of other RB protocols. From the computational basis measurement results, BiRB computes the expected result of a measurement of the target Pauli for the circuit. It then averages these values for all circuits of benchmark depth $m$ to get an average polarization $f_m$. BiRB fits these average polarizations to an exponential decay $f_m = B p^m$. 

To obtain this data analysis we simply specify the data type when we instantiate an `RB` protocol: we set `datatype = energies`.

```{code-cell} ipython3
protocol = pygsti.protocols.RB(datatype = 'energies', defaultfit='A-fixed')
results = protocol.run(data)
ws = pygsti.report.Workspace()
ws.init_notebook_mode(autodisplay=True)
ws.RandomizedBenchmarkingPlot(results)
```

```{code-cell} ipython3
# The error rate we *approximately* expect accord to Mirror RB theory
print(1 - (1 - qubit_error_rate)**(len(qubits)))
```

```{code-cell} ipython3

```

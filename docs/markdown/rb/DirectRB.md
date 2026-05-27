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

# Direct RB

This tutorial contains a few details on how to run [Direct randomized benchmarking](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.123.030503) that are not covered in the [RB overview tutorial](Overview).

## What is Direct RB? 

In essence, Direct RB is a streamlined, generalized version of the popular [Clifford RB](CliffordRB) method. It has the same core purpose - quantifying average gate performance - but it is feasable on more qubits, and it provides more directly useful information.

The basic requirements for running Clifford RB and Direct RB are the same. Both methods can be implemented on a set of $n$ qubits whenever the $n$-qubit Clifford group can be **generated** by the native gates on those $n$ qubits. Clifford RB runs circuits containing $m+1$ uniformly random $n$-qubit Cliffords followed by the unique inversion $n$-qubit Clifford gate (all of which must be compiled into the native gates of the device), where $m \geq 0$. In contrast, Direct RB circuits consist of:

1. A sub-circuit that generates a uniformly random $n$-qubit stabilizer state. 
2. $m$ independently sampled layers of the native gates in the device, with these layers sampled according to a user-specified distribution $\Omega$ over all possible circuit layers. 
3. A sub-circuit that maps the ideal output of the preceeding circuit to a uniformly random computational basis state (or, if preferred, to the all-zeros state).

This construction means that Direct RB circuits can be shorter than Clifford RB circuits - for the same $m$ a Direct RB circuit is typically much shorter, including for the shortest allowed depth $m=0$. This means that Direct RB can be run on more qubits (without just obtaining a useless, entirely decohered output). But Direct RB circuits still contain sufficient randomization (if $\Omega$ is chosen appropriately) to retain the core features of Clifford RB (exponential decays, etc).

For more information on what Direct RB is and why it is useful, see [*Direct randomized benchmarking for multi-qubit devices*](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.123.030503).

```{code-cell} ipython3
from __future__ import print_function #python 2 & 3 compatibility
import pygsti
from pygsti.processors import QubitProcessorSpec as QPS
from pygsti.processors import CliffordCompilationRules as CCR
```

## Creating a Direct RB experiment design

The data analysis in Direct RB is exactly as in Clifford RB, and how to do this analysis is covered in the [RB overview tutorial](Overview). The differences and flexibility in Direct RB are all at the experiment design stage, and so this is what is covered in this tutorial. 

### 1. Generic RB inputs

The first inputs to create a Direct RB experiment design are the same as in all RB protocols, and these are covered in the [RB overview tutorial](Overview). They are:

- The device to benchmark (`pspec`).
- The "RB depths" at which we will sample circuits (`depths`). For Direct RB, these depths are the number of layers in the "core" circuit, outlined in step (2) above. These depths can be any non-negative integers.
- The number of circuits to sample at each length (`k`).
- The qubits to benchmark (`qubits`).

```{code-cell} ipython3
n_qubits = 4
qubit_labels = ['Q0','Q1','Q2','Q3'] 
gate_names = ['Gxpi2', 'Gxmpi2', 'Gypi2', 'Gympi2', 'Gcphase'] 
pspec = QPS(n_qubits, gate_names, qubit_labels=qubit_labels, geometry='ring')

compilations = {'absolute': CCR.create_standard(pspec, 'absolute', ('paulis', '1Qcliffords'), verbosity=0),            
                'paulieq': CCR.create_standard(pspec, 'paulieq', ('1Qcliffords', 'allcnots'), verbosity=0)}

depths = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
k = 10
qubits = ['Q0','Q1','Q2', 'Q3']
```

All other arguments to the Direct RB experiment design generation function `DirectRBDesign` are optional. But to make the most out of Direct RB it is typically important to manually set at least some of them.

### 2. The circuit layer sampler
The Direct RB circuit layer sampling distribution $\Omega$ is perhaps the most important input to the Direct RB experiment design. This is because, by construction, the Direct RB error rate $r$ is $\Omega$-dependent. This is because $r$ quantifies gate performance over circuits that are sampled according to $\Omega$. This $\Omega$-dependence is useful, because by carefully choosing or varying $\Omega$ we can learn a lot about device performance. But it also means that the $\Omega$ has to be carefully chosen! At the very least, **you need to know what sampling distribution you are using in order to interpret the results!**

This might seem like a drawback in comparison to Clifford RB, but note that this $\Omega$-dependence is analogous to the Clifford-compiler dependence of the Clifford RB error rate (with the advantage that it is more easily controlled and understood).

The sampling distribution is specified via the optional arguements `sampler` and `samplerargs`. Here we use what we call the "edge grab" sampler. 

Because Mirror RB has an equivalent sampling-distribution dependence, there is a separate [random circuit sampling tutorial](Samplers) that introduces the different built-in sampling algorithms within pyGSTi (which includes details of the "edge grab" algorithm).

```{code-cell} ipython3
sampler = 'edgegrab'
samplerargs = [0.5]
```

### 3. The target output
By design, a specific Direct RB circuit should always return a particular bit-string if there is no errors, which we call it's target bit-string. This target bit-string can either be randomized (so that it is a uniformly random bit-string), or it can be set to always be the all-zeros bit-string. This is specified via the `randomizeout` argument. We advise randomizing the target output.

```{code-cell} ipython3
randomizeout = True
```

### 4. The stabilizer state compilation algorithm
To generate a Direct RB circuit in terms of native gates, it is necessary for pyGSTi to compile the sub-circuits in steps (1) and (3) that implement a randomly sampled stabilizer state preparation and measurement, respectively. We do this compilation using a randomized algorithm, and the number of randomization is controlled via `citerations`. Increasing this will reduce the average depth of these subcircuits, up to a point, making Direct RB feasable on more qubits. 
But note that time to generate the circuits increases linearly with `citerations` (so we'll leave it at the default value of 20 here). For the experiments presented in [Direct randomized benchmarking for multi-qubit devices](https://arxiv.org/abs/1807.07975) it was increased to 200, and values around this are probably advisable if you want to push the limits of how many qubits you can holistically benchmark with Direct RB for given gate fidelities.

Note that, unlike Clifford RB, there is (approximately) no compiler dependence to the Direct RB error rate. So the value of `citerations` only effects the feasability of Direct RB not its ouput error rate.

```{code-cell} ipython3
citerations = 20
```

From here, everything proceeds as in the RB overview tutorial (except for adding in the optional arguments).

```{code-cell} ipython3
# Here we construct an error model with 0.1% local depolarization on each qubit after each gate.
gate_error_rate = 0.001
def simulate_taking_data(data_template_filename):
    """Simulate taking data and filling the results into a template dataset.txt file"""
    error_rates = {}
    for gn in pspec.gate_names:
        n = pspec.gate_num_qubits(gn)
        error_rates[gn] = [gate_error_rate/(4**n - 1)] * (4**n - 1)
    noisemodel = pygsti.models.create_crosstalk_free_model(pspec, stochastic_error_probs=error_rates)
    pygsti.io.fill_in_empty_dataset_with_fake_data(data_template_filename, noisemodel, num_samples=1000, seed=1234)
```

```{code-cell} ipython3
design = pygsti.protocols.DirectRBDesign(pspec, compilations, depths, k, qubit_labels=qubits, sampler=sampler, 
                                           samplerargs=samplerargs, randomizeout=randomizeout,
                                           citerations=citerations)

pygsti.io.write_empty_protocol_data('../../tutorial_files/test_drb_dir', design, clobber_ok=True)

# -- fill in the dataset file in tutorial_files/test_rb_dir/data/dataset.txt --
simulate_taking_data('../../tutorial_files/test_drb_dir/data/dataset.txt') # REPLACE with actual data-taking

data = pygsti.io.read_data_from_dir('../../tutorial_files/test_drb_dir')

protocol = pygsti.protocols.RB()
results = protocol.run(data)
ws = pygsti.report.Workspace()
ws.init_notebook_mode(autodisplay=True)
ws.RandomizedBenchmarkingPlot(results)
```

```{code-cell} ipython3
# The error rate we approximately expect accord to Direct RB theory
print(1 - (1 - gate_error_rate)**len(qubits))
```

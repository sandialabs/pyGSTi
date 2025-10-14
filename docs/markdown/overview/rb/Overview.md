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

# Randomized Benchmarking

+++

This tutorial is an overview of randomized benchmarking (RB) in pyGSTi. The are multiple flavours of RB, that have different strengths and weaknesses. pyGSTi contains end-to-end methods for:

- [Clifford randomized benchmarking](http://journals.aps.org/prl/abstract/10.1103/PhysRevLett.106.180504), meaning benchmarking of the $n$-qubit Clifford group. This is a popular protocol for benchmarking 1 or 2 qubits. It is used in this notebook to demonstrate the general work-flow for running RB in pyGSTi. More details that are specific to Clifford RB can be found in the [Clifford RB tutorial](RB-CliffordRB.ipynb).
- [Direct randomized benchmarking](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.123.030503) is more streamlined than Clifford RB. As the name suggests, it directly benchmarks a set of native gates, rather than indirectly benchmarking them in the form of the $n$-qubit Clifford group gate set. Direct RB can be used to benchmark more qubits than standard Clifford RB, and only one small change is required to this notebook to run direct RB (this change is called out). Further detail on how to use Direct RB are given in the [Direct RB tutorial](RB-DirectRB.ipynb).
- *Mirror randomized benchmarking* is a new form of RB that is similar to Direct RB, but it is further streamlined so that it can be run on 10s to 100s of qubits. Running this method requires only two small adjustments to this notebook (again, these changes are called out). Further detail on how to use Mirror RB are given in the [Mirror RB tutorial](RB-MirrorRB.ipynb).
- [Simultaneous randomized benchmarking](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.109.240504) involves running RB circuits simultaneously on different subsets of a device, and then perhaps comparing the results to when RB is run while idling all other qubits. It is not a protocol per se. It is better thought of as an add-on to any other RB protocol, for exploring crosstalk and/or benchmarking a large device more efficiently. pyGSTi contains integrated methods for running simultaneous Clifford, Direct or Mirror RB. This is covered in the [multiple RB experiments tutorial](RB-MultiRBExperiments.ipynb).
- Benchmarking multiple device regions, by running multiple RB sub-experiments in one experimental run. This is also covered in the [multiple RB experiments tutorial](RB-MultiRBExperiments.ipynb) and [volumetric benchmarks tutorial](VolumetricBenchmarks.ipynb).

```{code-cell} ipython3
from __future__ import print_function #python 2 & 3 compatibility
import pygsti
from pygsti.processors import QubitProcessorSpec as QPS
from pygsti.processors import CliffordCompilationRules as CCR
```

## Step 1: create an experiment design

First, we specify the device to be benchmarked, so that pyGSTi can create circuits that use only the native gates in the device (including respecting the device's connectivity). We do this using a `QubitProcessorSpec` object (see the [QubitProcessorSpec tutorial](../objects/advanced/QubitProcessorSpec.ipynb) for details). Here we'll demonstrate RB on a device with:
- Five qubits on a ring.
- 1-qubit gates consisting of $\sigma_x$ and $\sigma_y$ rotations by $\pm \pi/2$, and an idle gate
- Controlled-Z gates connecting adjacent qubits on the ring

```{code-cell} ipython3
n_qubits = 5 
qubit_labels = ['Q0','Q1','Q2','Q3','Q4'] 
gate_names = ['Gxpi2', 'Gxmpi2', 'Gypi2', 'Gympi2', 'Gcphase'] 
availability = {'Gcphase':[('Q0','Q1'), ('Q1','Q2'), ('Q2','Q3'), ('Q3','Q4'),('Q4','Q0')]}
pspec = QPS(n_qubits, gate_names, availability=availability,qubit_labels=qubit_labels)

compilations = {'absolute': CCR.create_standard(pspec, 'absolute', ('paulis', '1Qcliffords'), verbosity=0),            
                'paulieq': CCR.create_standard(pspec, 'paulieq', ('1Qcliffords', 'allcnots'), verbosity=0)}
```

All RB methods require a set of "RB depths" and a number of circuits to sample at each length ($k$). For all RB methods in pyGSTi, we use a convention where the smallest RB depth ($m$) allowed is $m=0$. So, in the case of Clifford RB on $n$ qubits, $m$ is the number of (uncompiled) $n$-qubit Clifford gates in the sequence minus two.

We can also specify the qubits to be benchmarked (if this is not specified then it defaults to holistic benchmarking of all the qubits). Here, we'll create an experiment for running 2-qubit Clifford RB on qubits 'Q0' and 'Q1'.

```{code-cell} ipython3
depths = [0,1,2,4,8,16,32,64]
k = 10
qubits = ['Q0','Q1']
# To run direct / mirror RB change CliffordRBDesign -> DirectRBDesign / MirrorRBDesign
exp_design = pygsti.protocols.CliffordRBDesign(pspec, compilations, depths, k, qubit_labels=qubits)
```

## Step 2: collect data as specified by the experiment design
Next, we just follow the instructions in the experiment design to collect data from the quantum processor.  In this example, we'll generate the data using a depolarizing noise model since we don't have a real quantum processor lying around.  The call to `simulate_taking_data` function should be replaced with the user filling out the empty "template" data set file with real data.  Note also that we set `clobber_ok=True`; this is so the tutorial can be run multiple times without having to manually remove the dataset.txt file - we recommend you leave this set to False (the default) when using it in your own scripts.

```{code-cell} ipython3
# Here we construct an error model with 1% local depolarization on each qubit after each gate.
def simulate_taking_data(data_template_filename):
    """Simulate taking data and filling the results into a template dataset.txt file"""
    noisemodel = pygsti.models.create_crosstalk_free_model(pspec, depolarization_strengths={g:0.01 for g in pspec.gate_names})
    pygsti.io.fill_in_empty_dataset_with_fake_data(data_template_filename, noisemodel, num_samples=1000, seed=1234)
```

```{code-cell} ipython3
pygsti.io.write_empty_protocol_data('../tutorial_files/test_rb_dir', exp_design, clobber_ok=True)

# -- fill in the dataset file in tutorial_files/test_rb_dir/data/dataset.txt --
simulate_taking_data('../tutorial_files/test_rb_dir/data/dataset.txt') # REPLACE with actual data-taking

data = pygsti.io.read_data_from_dir('../tutorial_files/test_rb_dir')
```

## Step 3: Run the RB protocol
Now we just instantiate an `RB` protocol and `.run` it on our data object. This involves converting the data to the success/fail format of RB and then fitting it to an exponential decay ($P_m = A + B p^m$ where $P_m$ is the average success probability at RB length $m$). The `.run` method returns a results object that can be used to plot decay curves, and display error rates.

```{code-cell} ipython3
# To run Mirror RB, set datatype = 'adjusted_success_probabilities' in this init.
protocol = pygsti.protocols.RB()
results = protocol.run(data)
ws = pygsti.report.Workspace()
ws.init_notebook_mode(autodisplay=True)
ws.RandomizedBenchmarkingPlot(results)
```

By default, pyGSTi uses an RB error rate ($r$) convention whereby
$$ r = \frac{(4^n - 1)(1 - p)}{4^n}, $$
where $n$ is the number of qubits (here, 2) and $p$ is the estimated decay constant obtained from fitting to $P_m = A + Bp^m$. This approximately corresponds to the mean entanglement infidelity of the benchmarked gate set [(modulo some subtleties)](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.119.130502). A common alternative convention is to define $r$ by 
$$ r = \frac{(2^n - 1)(1 - p)}{2^n}. $$
In this case, $r$ approximately corresponds to the mean average gate infidelity of the benchmarked gate set (modulo the same subtleties). This alternative convention can be obtained by setting the optional argument `rtype = 'AGI'` when initializing an `RB` protocol.

We have the entanglement infidelity convention as the default because it is more convenient when comparing RB error rates obtained from benchmarking different numbers of qubits (as the entanglement fidelity of a tensor product of gates is the product of the constituent fidelities).

```{code-cell} ipython3
# We can also access the estimated error rate directly without plotting the decay
r = results.fits['full'].estimates['r']
rstd = results.fits['full'].stds['r']
rAfix = results.fits['A-fixed'].estimates['r']
rAfixstd = results.fits['A-fixed'].stds['r']
print("r = {0:1.2e} +/- {1:1.2e} (fit with a free asymptote)".format(r, 2*rstd))
print("r = {0:1.2e} +/- {1:1.2e} (fit with the asymptote fixed to 1/2^n)".format(rAfix, 2*rAfixstd))
```

```{code-cell} ipython3

```

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

# Simulating Clifford randomized benchmarking using implicit models

This tutorial demonstrates shows how to simulate Clifford RB sequences using $n$-qubit "implicit" models which build $n$-qubit process matrices from smaller building blocks.  This restricts the noise allowed in the $n$-qubit model; in this tutorial we take $n=3$ and use a `LocalNoiseModel`.

```{code-cell} ipython3
import pygsti
import numpy as np

from pygsti.processors import QubitProcessorSpec
from pygsti.processors import CliffordCompilationRules as CCR
```

## Get some CRB circuits

First, we follow the [Clifford RB](../CliffordRB.ipynb) tutorial to generate a set of sequences.  If you want to perform Direct RB instead, just replace this cell with the contents of the [Direct RB](../DirectRB.ipynb) tutorial up until the point where it creates `circuitlist`:

```{code-cell} ipython3
#Specify the device to be benchmarked - in this case 2 qubits
n_qubits = 2
qubit_labels = list(range(n_qubits)) 
gate_names = ['Gxpi2', 'Gypi2','Gcphase'] 
availability = {'Gcphase':[(i,i+1) for i in range(n_qubits-1)]}
pspec = QubitProcessorSpec(n_qubits, gate_names, availability=availability, 
                                 qubit_labels=qubit_labels)

compilations = {'absolute': CCR.create_standard(pspec, 'absolute', ('paulis', '1Qcliffords'), verbosity=0),            
                'paulieq': CCR.create_standard(pspec, 'paulieq', ('1Qcliffords', 'allcnots'), verbosity=0)}

#Specify RB parameters (k = number of repetitions at each length)
lengths = [0,1,2,4,8]
k = 8
subsetQs = qubit_labels
randomizeout = False # ==> all circuits have the *same* ideal outcome (the all-zeros bitstring)

#Generate clifford RB circuits
exp_design = pygsti.protocols.CliffordRBDesign(pspec, compilations, lengths, k,
                                               qubit_labels=subsetQs, randomizeout=randomizeout)

#Collect all the circuits into one list:
circuitlist = exp_design.all_circuits_needing_data
```

## Create a model to simulate these circuits
Now we need to create a model that can simulate circuits like this. The RB circuits use pyGSTi's "multi-qubit" conventions, which mean:
1. RB circuits use our "multi-qubit" gate naming, so you have gates like `Gxpi2:0` and `Gcphase:0:1`.
2. RB circuits do gates in parallel (this only matters for >1 qubits), so you have layers like `[Gypi2:0Gypi2:1]`

"Implicit" models in pyGSTi (see the [implicit model tutorial](../../objects/ImplicitModel.ipynb)) are designed to efficiently describe multi-qubit processors.  There are numerous ways of constructing implicit models, all of which can simulate the type of circuits described above.  Here we'll demonstrate the simplest type: a "local noise model" (class `LocalNoiseModel`) where the noise on a gate can only act on that gate's target qubits - so, for instance, 1-qubit gates are still given by 1-qubit operators, not $n$-qubit ones.

One of the easiest ways to construct a `LocalNoiseModel` is to use the `create_crosstalk_free_model` function, which takes our `QubitProcessorSpec` and other optional kwargs.

```{code-cell} ipython3
myModel = pygsti.models.create_crosstalk_free_model(pspec, ideal_gate_type='full')
myModel.sim = 'map'
```

Setting `ideal_gate_type="full"` is important, as it lets us assign arbitrary numpy arrays to gates as we'll show below.  If you need to use other gates that aren't built into pyGSTi, you can use the `nonstd_gate_unitaries`
argument of `from_parameterization` (see the docstring).

The `from_parameterization` function creates a model with ideal (perfect) gates.  We'll now create a 1-qubit depolarization superoperator, and a corresponding 2-qubit one (just the tensor product of two 1-qubit ones) to add some simple noise.  

```{code-cell} ipython3
depol1Q = np.array([[1, 0,   0, 0],
                    [0, 0.99, 0, 0],
                    [0, 0, 0.99, 0],
                    [0, 0, 0, 0.99]], 'd') # 1-qubit depolarizing operator
depol2Q = np.kron(depol1Q,depol1Q)
```

As detailed in the [implicit model tutorial](../../objects/ImplicitModel.ipynb), the gate operations of a `LocalNoiseModel` are held in its `.operation_blks['gates']` dictionary.  We'll alter these by assigning new process matrices to each gate.  In this case, it will be just a depolarized version of the original gate.

```{code-cell} ipython3
myModel.operation_blks['gates']["Gxpi2"] = np.dot(depol1Q, myModel.operation_blks['gates']["Gxpi2"])
myModel.operation_blks['gates']["Gypi2"] = np.dot(depol1Q, myModel.operation_blks['gates']["Gypi2"])  
myModel.operation_blks['gates']["Gcphase"] = np.dot(depol2Q, myModel.operation_blks['gates']["Gcphase"])
```

Here's what the gates look like now:

```{code-cell} ipython3
print(myModel.operation_blks['gates']["Gxpi2"])
print(myModel.operation_blks['gates']["Gypi2"])
print(myModel.operation_blks['gates']["Gcphase"])
```

Now that our `Model` object is set to go, generating simulated data is easy:

```{code-cell} ipython3
ds = pygsti.data.simulate_data(myModel, circuitlist, 100, seed=1234)
```

## Running RB on the simulated `DataSet`
To run an RB analysis, we just package up the experiment design and data set into a `ProtocolData` object and give this to a `RB` protocol's `run` method.  This returns a `RandomizedBenchmarkingResults` object that can be used to plot the RB decay curve.  (See the [RB analysis tutorial](../RBAnalysis.ipynb) for more details.)

```{code-cell} ipython3
data = pygsti.protocols.ProtocolData(exp_design, ds)
results = pygsti.protocols.RB().run(data)
```

```{code-cell} ipython3
%matplotlib inline
results.plot()
```

```{code-cell} ipython3

```

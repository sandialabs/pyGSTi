---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.17.3
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Random Circuit Samplers

+++

This tutorial introduces the different random circuit layer samplers in-built into `pyGSTi`.

```{code-cell} ipython3
from __future__ import print_function #python 2 & 3 compatibility
import pygsti
```

The circuit layer samplers in pyGSTi exist primarily for the purpose of running [Direct Randomized Benchmarking](RB-DirectRB.ipynb) and/or [Mirror Randomized Benchmarking](RB-MirroRB.ipynb). Here we'll demonstate them by just creating a more generic random circuit consisting of independently sampled random layers, but the same syntax is used to select these samplers in the `DirectRBDesign` and `MirrorRBDesign` functions.

```{code-cell} ipython3
from pygsti.algorithms.randomcircuit import create_random_circuit
from pygsti.processors import QubitProcessorSpec as QPS
```

The random circuit samplers generate circuits for a specific device, so we define that device via a QubitProcessorSpec.

```{code-cell} ipython3
n_qubits = 4
qubit_labels = ['Q0','Q1','Q2','Q3'] 
gate_names = ['Gxpi2', 'Gxmpi2', 'Gypi2', 'Gympi2', 'Gcphase'] 
availability = {'Gcphase':[('Q0','Q1'), ('Q1','Q2'), ('Q2','Q3'), ('Q3','Q0')]}
pspec = QPS(n_qubits, gate_names, availability=availability, qubit_labels=qubit_labels)
```

### What kind of circuit layer samplers do we need for Direct and Mirror RB?

As these samplers are mostly useful for Direct and Mirror RB, it's first worthwhile to go over what properties we want of the circuit layer samplers in those protocols.

By construction, the Direct and Mirror RB error rates depend on how the circuits are sampled. Essentially, these error rates quantify gate performance over circuits that are sampled according to the used sampling distribution. So we need to pick a sampling distribution that provides useful error rates.

A second consideration is that Direct and Mirror RB are only reliable - i.e., the success probabilities decay exponentially and the error rates are easily interpreted - if the sampling distribution is sufficiently scrambling. To be "sufficiently scrambling" we need a high enough rate of two-qubit gates, in order to spread any errors that occur locally across the full set of qubits being benchmarked. We also need sufficient local randomization to quickly convert coherent to stochastic errors, to stop errors coherently adding across many layers.

```{code-cell} ipython3
depth = 10
```

## 1. The qubit elimination layer sampler

**Inputs.** A probability $p$.

**Algorithm.**
1. Start with an empty layer $L$.
2. Pick a qubit $q_1$ uniformly at random from all the qubits that do not yet have a gate assigned to them in $L$.
2. Pick another qubit $q_2$, uniformly at random, that also does not have a gate assigned to it in $L$ *and* that is connected to $q_1$. If there is no such qubits, skip to (4).
3. If such a qubit has been found, assign a two-qubit gate to this pair of qubits ($q_1$ and $q_2$) with probability $p$.
4. If a two-qubit gate has not been assigned to $q_1$, pick a uniformly random 1-qubit gate to assign to $q_1$.
5. Repeat 2-4 until all qubits have been assigned a gate.

```{code-cell} ipython3
qs = [1,2]
remaining_qubits = [1,2,3]
set(qs).issubset(remaining_qubits)
```

```{code-cell} ipython3
pspec.compute_ops_on_qubits()
```

```{code-cell} ipython3
sampler = 'Qelimination'
samplerargs = [0.5] # Setting p = 0.5 (this is the default).
circuit = create_random_circuit(pspec, depth, sampler=sampler, samplerargs=samplerargs)
print(circuit)
```

The qubit elimination sampler is easy to use: it works for any connectivity, and the same value of $p$ can be used for benchmarking any number of qubits. But it's perhaps not the best sampler, because the relationship between $p$ and the number of two-qubit gates per layer is opaque, except for devices with full connectivity. A sampler that is similar in spirit, but that we've found more useful, is the "edge grab" sampler. 

### 2. The edge-grab sampler

**Inputs.** A float $\gamma$, corresponding to the expected number of two-qubit gates in the layer.

**Algorithm.** 
1. Select a set of candidate edges in the connective graph as follows:
   (1.1) Start with an empty candidate edge list $A$, and a list that initial contains all the edges $B$.
   (1.2) Select an edge from $B$ uniformly at random, and add it to the $A$. Delete all the edges in $B$ that contain a vertex (qubit) in common with the selected edge.
   (1.3) Repeat (1.2) until $B$ contains no edges.
2. Indepedently consider each edge in $A$, and add a two-qubit gate on that edge to the layer with probability $\gamma/|A|$ where $|A|$ is the number of edges in $A$.
3. Independently and uniformly at random, select a one-qubit gate to apply to each qubit that does not have a two-qubit gate assigned to it in the layer.

This algorithm is designed so that the expected number of two-qubit gates in the layer is $\gamma$, and every possible circuit layer is sampled with a non-zero probability (unless $\gamma=0$). But note that this algorithm will fail if $\gamma/|A|$ can ever be greater then $1$. The maximum allowable $\gamma$ depends on the connectivity of the graph. 

```{code-cell} ipython3
sampler = 'edgegrab'
samplerargs = [1] # Setting gamma = 1
circuit = create_random_circuit(pspec, depth, sampler=sampler, samplerargs=samplerargs)
print(circuit)
```

### 3. The compatible two-qubit gates sampler

The compatible two-qubit gates sampler is more involved to use, but it provides a lot of control over the distribution of two-qubit gates in a layer, while keeping the 1-qubit gate distribution simple.

```{code-cell} ipython3
sampler = 'co2Qgates'
```

This sampler requires the user to specify sets of compatible 2-qubit gates, meaning 2-qubit gates that can applied in parallel. We specifying this as a list of lists of `Label` objects (see the [Ciruit tutorial](../objects/Circuit.ipynb) for more on `Label` objects), so let's import the `Label` object:

```{code-cell} ipython3
from pygsti.baseobjs import Label as L
```

In this example, we have 4 qubits in a ring. So we can easily write down all of the possible compatible 2-qubit gate lists over these 4 qubits. There are only 7 of them: a list containing no 2-qubit gates, and 4 lists containing only 1 2-qubit gate, and 2 lists containing 2 2-qubit gates.

```{code-cell} ipython3
C2QGs1 = [] #  A list containing no 2-qubit gates is an acceptable set of compatible 2-qubit gates.
C2QGs2 = [L('Gcphase',('Q0','Q1')),]
C2QGs3 = [L('Gcphase',('Q1','Q2')),] 
C2QGs4 = [L('Gcphase',('Q2','Q3')),] 
C2QGs5 = [L('Gcphase',('Q3','Q0')),] 
C2QGs6 = [L('Gcphase',('Q0','Q1')), L('Gcphase',('Q2','Q3')),] 
C2QGs7 = [L('Gcphase',('Q1','Q2')), L('Gcphase',('Q3','Q0')),] 
```

Note that we often wouldn't want to start by writting down all possible sets of compatible 2-qubit gates - there can be a lot of them - and we only need to specify the compatible sets that we want to use. Continuing the example, we put all of these compatible 2-qubit gate lists into a list **`co2Qgates`**, we also pick a probability distribution over this list **`co2Qgatesprob`**, and we pick a probability **`twoQprob`** between 0 and 1.

```{code-cell} ipython3
co2Qgates = [C2QGs1, C2QGs2, C2QGs3, C2QGs4, C2QGs5, C2QGs6, C2QGs7]
co2Qgatesprob = [0.5, 0.125, 0.125, 0.125, 0.125, 0, 0]
twoQprob = 1
```

The sampler then picks a layer as follows:
1. Sample a list from `co2Qgates` according to the distribution `co2Qgatesprob`.
2. Consider each gate in this list, and add it to the layer with probability `twoQprob`.
3. For every qubit that doesn't yet have a gate assigned to it in the layer, independently and uniformly at random, sample a 1-qubit gate to assign to that qubit, sampled from the "native" 1-qubit gates in the device.

So with the example above there is a 50% probability of no 2-qubit gates in a layer, a 50% chance that there is one 2-qubit gate in the layer, there is no probability of more than one 2-qubit gate in the layer, and each of the 4 possible 2-qubit gates is equally likely to appear in a layer.

Note that there is more than one way to achieve the same sampling here. Instead, we could have set `co2Qgatesprob = [0,0.25,0.25,0.25,0.25,0,0]` and `twoQprob = 0.5`.

To use these sampler parameters, we put them (in this order) into the samplerargs list:

```{code-cell} ipython3
samplerargs = [co2Qgates, co2Qgatesprob, twoQprob]
circuit = create_random_circuit(pspec, depth, sampler=sampler, samplerargs=samplerargs)
print(circuit)
```

This is similar to the sampling used in the Direct RB experiments of [*Direct randomized benchmarking for multi-qubit devices*](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.123.030503).

There is also an option for "nested" sets of compatible two-qubit gates. That is, `co2Qgates` can be a list where some or all of the elements are not lists containing compatable two-qubit gates, but are instead lists of lists of compatible two-qubit gates. 

An element of `co2Qgates` is sampled according to the `co2Qgatesprob` distribution (which defaults to the uniform distribution if not specified). If the chosen element is just a list of `Labels` (i.e., a list of compatible 2-qubit gates), the algorithm proceeds as above. But if the chosen element is a list of lists of `Labels`, the sampler picks one of these sublists uniformly at random; this sublist should be a list of compatible 2-qubit gates.

This may sound complicated, so below we show how to re-write the previous example in this format. 

```{code-cell} ipython3
co2Qgates = [C2QGs1,[C2QGs2,C2QGs3,C2QGs4, C2QGs5]]
co2Qgatesprob = [0.5,0.5] # This doesn't need to be specified, as the uniform dist is the default.
twoQprob = 1 # This also doesn't need to be specifed, as this value is the default.
samplerargs = [co2Qgates,] # We leave the latter two values off this list, because they are the defaults.
```

```{code-cell} ipython3
circuit = create_random_circuit(pspec, depth, sampler=sampler, samplerargs=samplerargs)
print(circuit)
```

```{code-cell} ipython3

```

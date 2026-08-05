---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.19.5
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# QPANN Part 2: Error Propagation and Locality Filters

This is **Part 2** of the 3-part Quantum Physics-Aware Neural Network (QPANN) tutorial series.

In Part 1, we learned how to encode circuits and index error generators. In this tutorial, we will explore:
1. **Clifford Error Propagation:** How we push layer-wise errors forward to the end of a Clifford circuit.
2. **First-Order Sensitivity (Alphas):** How outcome probabilities react linearly to error rates.
3. **The "Snipper" (Locality Filter):** How we constrain the network's input features so that each error generator's sub-network only sees gate features in its local neighborhood.

---
## 1. Clifford Error Propagation

The core mathematical trick of a QPANN is that for **Clifford-only circuits**, any elementary error generator $G_j$ inserted after a layer $l$ can be propagated forward through all subsequent layers to the end of the circuit. Under conjugation by Clifford gates, a Pauli operator is mapped to another Pauli operator (with a possible sign flip).

Thus, an error generator $G_j$ at layer $l$ propagates to some canonical end-of-circuit generator $G_{k}$ with a sign $\pm 1$:
$$G_j \xrightarrow{\text{Clifford}} s \cdot G_k$$

The function `circuit_error_propagation_matrices` computes this mapping for a single circuit, yielding:
* `indices`: Which canonical global index the error propagates to.
* `signs`: The $\pm 1$ sign acquired during propagation.

Let's compute these matrices for a simple circuit.

```{code-cell} ipython3
import numpy as np
import pygsti
from pygsti.circuits import Circuit
from pygsti.processors.processorspec import QubitProcessorSpec as QPS
from pygsti.extras.ml import encoding, errgentools as et

# 1. Build a 2-qubit processor spec
pspec = QPS(num_qubits=2, qubit_labels=[0, 1], gate_names=['Gxpi2', 'Gypi2', 'Gcphase'],
            availability={'Gcphase': [(0, 1)]})

# 2. Define a simple 2-layer circuit
circuit = Circuit('[Gxpi2:0][Gcphase:0:1]@(0,1)')
print(circuit)

# 3. List the error generators we want to propagate
error_generators = [('H', ('XI',)), ('S', ('IX',)), ('S', ('ZI',))]

# 4. Compute the propagation matrices
indices, signs = encoding.circuit_error_propagation_matrices(circuit, error_generators)
print("Indices matrix (shape: depth x num_generators):\n", indices)
print("Signs matrix (shape: depth x num_generators):\n", signs)
```

### Interpreting the Output:
* **Last Layer ($l = 1$, row 1):** The error is inserted after the last gate, so there is no subsequent gate left to propagate through. It always maps back to its original index with sign $+1$.
* **First Layer ($l = 0$, row 0):** The error is inserted after the first gate and must propagate through the `Gcphase:0:1` gate. Since `Gcphase` is diagonal, a $Z$ error on qubit 0 (`S_ZI`) commutes with it and remains unchanged (`signs[0,2] = 1`). However, a coherent $X$ error on qubit 0 (`H_XI`) is transformed by `Gcphase` into a joint `H_XZ` error generator, which is reflected in its updated index value.

---
## 2. First-Order Sensitivities (Alpha Tensors)

Using the mathematical derivation of [Efficient simulation of Clifford circuits with small Markovian errors](https://arxiv.org/abs/2504.15128), the first-order sensitivity of a bitstring outcome probability $P(x)$ to an end-of-circuit error generator $G_k$ is represented by the coefficient $\alpha(G_k, x)$. 

Using the batched entry point `error_generator_tensors` with `alpha_representation='concise'` (the default and most scalable mode), we can compute:
1. **`probabilities`:** The noise-free, ideal outcome probabilities of each circuit.
2. **`alphas`:** The 4D sensitivity tensor of shape `(num_circuits, 2**n, max_depth, num_error_generators)`.

Let's compute these tensors.

```{code-cell} ipython3
# 1. Define a couple of circuits
circuits = [
    Circuit('[Gxpi2:0Gypi2:1]@(0,1)'),
    Circuit('[Gxpi2:0][Gcphase:0:1]@(0,1)')
]

# 2. Compute the tensors
tensors = encoding.error_generator_tensors(circuits, error_generators, pspec,
                                             alpha_representation='concise')

print("tensors.keys():", list(tensors.keys()))
print("Ideal probabilities shape:", tensors['probabilities'].shape, " (num_circuits, 2**n)")
print("Alphas shape:", tensors['alphas'].shape, " (num_circuits, 2**n, max_depth, num_error_generators)")

print("\nCircuit 1 ideal bitstring probabilities:\n", tensors['probabilities'][1])
```

---
## 3. Locality Filters: The Snipper

In a physical quantum device, a coherent or stochastic error affecting a set of qubits is typically caused by gates applied to those same qubits or their nearest neighbors. 

To embed this physics assumption, a QPANN uses a **snipper** (a list of index lists). For each candidate error generator, the snipper lists the specific indices of the circuit-encoding tensor that the generator's sub-network is allowed to see. This drastically reduces the number of network parameters and prevents overfitting.

We construct a snipper using `layer_snipper_from_qubit_graph`, which describes the device's connectivity using a `networkx`/`igraph`/`graph-tool` graph, a pygsti `QubitGraph`/`QubitProcessorSpec`, or a raw graph Laplacian/adjacency matrix (see `pygsti.extras.ml.graphtools`):
* `hops=0`: The sub-network for an error generator acting on qubit $q$ only sees gate features applied directly to qubit $q$.
* `hops=1`: It sees gate features on qubit $q$ AND any of its directly adjacent neighbor qubits in the graph.

Let's build and compare these snippers.

```{code-cell} ipython3
from pygsti.extras.ml import snippers
import networkx as nx

# 1. Describe the device's connectivity (the Gcphase edge) as a networkx graph. Passing `pspec`
# directly (its 2-qubit-gate connectivity would be used) gives an identical result here, since
# `pspec` already declares `Gcphase` on edge (0, 1); a raw adjacency matrix
# (`snippers.undirected_adjacency_matrix_from_edges([(0, 1)], [0, 1])`) still works too.
qubit_graph = nx.Graph()
qubit_graph.add_nodes_from([0, 1])
qubit_graph.add_edge(0, 1)
print("Device connectivity graph edges:", list(qubit_graph.edges()))

# 2. Build the standard multi-hot circuit encoder
encoder = encoding.StandardCircuitEncoder(pspec)

# 3. Build a hops=0 and a hops=1 snipper
snipper_hops0 = snippers.layer_snipper_from_qubit_graph(
    error_generators, encoder, qubit_graph, hops=0
)
snipper_hops1 = snippers.layer_snipper_from_qubit_graph(
    error_generators, encoder, qubit_graph, hops=1
)

print("\nModelled error generators:")
for j, eg in enumerate(error_generators):
    print(f"  Gen {j} -> {eg}")
    print(f"    Feature indices seen (hops=0): {snipper_hops0[j]} (gates: {[encoder.gate_indexing[i] for i in snipper_hops0[j]]})")
    print(f"    Feature indices seen (hops=1): {snipper_hops1[j]} (gates: {[encoder.gate_indexing[i] for i in snipper_hops1[j]]})")
```

---
### What's Next?
We now have all the necessary ingredients:
* **`circuits_tensor`**: Our multi-hot circuit-encoding inputs.
* **`alphas`**: First-order sensitivities.
* **`probabilities`**: Noise-free ideal probabilities.
* **`snipper`**: Our local connectivity filter.

Proceed to **[Part 3: Model Building, Synthetic Data, and Training](QPANN-Training.ipynb)** to assemble these into a QPANN model and train it!

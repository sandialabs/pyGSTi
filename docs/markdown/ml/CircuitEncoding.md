---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.19.5
  notebook_metadata_filter: substitutions
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
substitutions:
  num_qubits: '3'
  num_hops: '1'
---

# QPANN Part 1: Error Generators and Circuit Encoding

This is **Part 1** of the 3-part Quantum Physics-Aware Neural Network (QPANN) tutorial series.

Before we can build a QPANN, we must understand the fundamental representations it uses. In this tutorial, we will explore:
1. **Elementary Error Generators:** How we mathematically represent candidate noise channels.
2. **Enumerating Tracked Generators:** How we choose a subset of "plausible" local errors based on our device's qubit connectivity graph.
3. **Pauli-Correlation (C) and Active (A) Errors:** An advanced look at the other error types in the taxonomy.
4. **Circuit Encoding:** How we convert a `pygsti.circuits.Circuit` into a multi-hot numeric tensor suitable for input to Keras layers.

---
## 1. Pauli and Error Generator Bookkeeping

Under the hood, noise is represented using the elementary-error-generator formalism of the *Taxonomy of Small Errors* (Blume-Kohout et al.). The most common error types are:
* **Hamiltonian (coherent) errors ('H'):** Represented by a single Pauli string (e.g. `H_XI` represents a coherent X-rotation-like error on the first qubit).
* **Stochastic (incoherent) errors ('S'):** Also represented by a single Pauli string (e.g. `S_IX` represents a stochastic Pauli-X flip on the second qubit).

The `errgentools` module (`pygsti.extras.ml.errgentools`) provides essential tools to convert Pauli strings to integers and vice-versa, which acts as our global index bookkeeping. Let's examine some of these functions.

```{code-cell} ipython3
import numpy as np
import pygsti
from pygsti.extras.ml import errgentools as et
```

```{code-cell} ipython3
# Map a Pauli string to its base-4 integer index
# Direct convention: leftmost character corresponds to qubit 0 (direct string position)
n = 2
ps_idx = et.paulistring_to_index('IX', n)
print(f"Pauli string 'IX' maps to integer index: {ps_idx}")
print(f"Index {ps_idx} maps back to Pauli string: {et.index_to_paulistring(ps_idx, n)!r}")

# Retrieve the global index of a modeled error generator.
# Hamiltonian ('H') ranges from [0, 4**n); Stochastic ('S') ranges from [4**n, 2*4**n).
h_idx = et.error_generator_index('H', ('IX',))
s_idx = et.error_generator_index('S', ('IX',))
print(f"Index of H_IX: {h_idx}")
print(f"Index of S_IX: {s_idx}")

# We can easily round-trip these indices back into error generator descriptors
print(f"Index {s_idx} represents error generator: {et.index_to_error_gen(s_idx, n)}")
print(f"Total possible H+S error generators for {n} qubits: {2 * (4**n)}")
```

---
## 2. Enumerating Local, Low-Weight Error Generators

For a system of $n$ qubits, the total number of possible error generators grows exponentially ($O(4^n)$). To keep a QPANN polynomial in size, we only let the network predict rates for a "plausible" subset of candidate error generators $G$. Typically, this subset consists of:
* All weight-1 error generators (single-qubit coherent and stochastic errors).
* Weight-2 error generators that lie on connected edges of our device's qubit connectivity graph.

To do this, we describe our device's connectivity as a graph and call the convenience entry point `up_to_weight_k_error_gens_from_qubit_graph`. This (and `layer_snipper_from_qubit_graph`, used in Part 2) accepts a `networkx` graph, an `igraph`/`graph-tool` graph, pyGSTi's own `QubitGraph` or `QubitProcessorSpec`, or a raw adjacency matrix -- see `pygsti.tools.graphs` for the full list. Let's demonstrate this on a {{num_qubits}}-qubit line graph (0-1-2) with a hop distance of {{num_hops}}.

```{code-cell} ipython3
from pygsti.processors.processorspec import QubitProcessorSpec as QPS
import networkx as nx
import random

num_qubits = 3
num_hops = 1
# 1. Set up a 3-qubit device spec with Gcphase gates available on edges (0,1) and (1,2)
qubit_labels = [0, 1, 2]
availability = {'Gcphase': [(0, 1), (1, 2)]}
pspec = QPS(num_qubits=num_qubits, qubit_labels=qubit_labels,
            gate_names=['Gxpi2', 'Gypi2', 'Gcphase'], availability=availability)

# 2. Describe the 0-1-2 line connectivity as a networkx graph.
qubit_graph = nx.Graph()
qubit_graph.add_nodes_from(qubit_labels)
qubit_graph.add_edges_from(availability['Gcphase'])
# An igraph/graph-tool graph, a pygsti QubitGraph, or `pspec` itself (its 2-qubit-gate
# connectivity would be used) all work identically here. A raw adjacency matrix still works
# too -- e.g. `qubit_graph=np.array([[0,1,0],[1,0,1],[0,1,0]])` below would give the same
# result as the networkx graph we just built.

# 3. Enumerate all weight-1 and graph-local weight-2 generators for 'H' and 'S'
modelled_error_generators = et.up_to_weight_k_error_gens_from_qubit_graph(
    k=2, n=num_qubits, qubit_graph=qubit_graph, num_hops=num_hops, egtypes=['H', 'S']
)

print(f"Total local error generators enumerated (k=2, hops=1): {len(modelled_error_generators)}")
print("Sample generators:")
for eg in random.sample(modelled_error_generators, min(10, len(modelled_error_generators))):
    print(" ", eg)
```

---
## 3. Pauli-Correlation (C) and Active (A) Error Generators (Advanced)

While the QPANN paper primarily demonstrates Hamiltonian ('H') and Stochastic ('S') errors, the underlying pyGSTi codebase on this branch is fully unified and supports all four sectors of the error taxonomy:
* **Pauli-Correlation errors ('C'):** Symmetric errors indexed by an unordered pair of two distinct non-identity Paulis (representing correlated stochastic noise).
* **Active errors ('A'):** Antisymmetric errors indexed by an unordered pair of distinct non-identity Paulis (representing coherent/amplitude-damping-like non-unital crosstalk).

Because Active errors are antisymmetric ($A_{P,Q} = -A_{Q,P}$), errgentools provides a sign canonicalization utility that tracks when a swap is needed. Let's see how these are indexed.

```{code-cell} ipython3
# Canonicalize a pair of Paulis lexicographically: 'ZZ' and 'XY' -> 'XY' < 'ZZ' (True means swapped)
P, Q, was_swapped = et.canonical_pauli_pair('ZZ', 'XY')
print(f"Canonical pair: ({P!r}, {Q!r}), swapped={was_swapped}")

# The canonicalization sign is -1 ONLY when we swap the Paulis of an Active ('A') type generator
print("C canonicalization sign (swapped):", et.error_generator_canonicalization_sign('C', ('ZZ', 'XY')))
print("A canonicalization sign (swapped):", et.error_generator_canonicalization_sign('A', ('ZZ', 'XY')))
print("A canonicalization sign (no-swap):", et.error_generator_canonicalization_sign('A', ('XY', 'ZZ')))

# We can easily include 'C' and 'A' types in our device-restricted enumeration
modelled_all_four = et.up_to_weight_k_error_gens_from_qubit_graph(
    k=2, n=3, qubit_graph=qubit_graph, num_hops=1, egtypes=['H', 'S', 'C', 'A']
)
print(f"Total error generators including C/A (k=2, hops=1): {len(modelled_all_four)}")
```

---
## 4. Encoding Circuits as Tensors

To feed quantum circuits into our neural network, we must convert each circuit layer into a numeric vector. The standard encoder `StandardCircuitEncoder` maps each layer to a multi-hot binary vector of length `encoder.length`. 

The length of this vector corresponds exactly to the total number of available gates (names + qubit placements) on the processor. A layer containing gates $g_1, g_2, ...$ has $1.0$ at their respective indices, and $0.0$ elsewhere.

Let's build an encoder, encode some circuits, and look at the resulting tensor. We'll use `circuits_to_tensor` to batch multiple encoded circuits into a padded 3D array of shape `(num_circuits, max_depth, encoder.length)`.

```{code-cell} ipython3
from pygsti.extras.ml import encoding
from pygsti.circuits import Circuit

# 1. Define some test circuits
circuits = [
    Circuit('[Gxpi2:0Gypi2:1]@(0,1,2)'),                    # depth 1
    Circuit('[Gcphase:0:1Gypi2:2][Gxpi2:1Gypi2:0]@(0,1,2)')  # depth 2
]

# 2. Build the encoder
encoder = encoding.StandardCircuitEncoder(pspec)
print(f"Encoder gate indexing (length {encoder.length}):")
for idx, gate in enumerate(encoder.gate_indexing):
    print(f"  Index {idx} -> {gate}")
print()

print(f"Defined {len(circuits)} test circuits:")
for idx, circuit in enumerate(circuits):
    print(f"  Circuit {idx}:")
    print(circuit)

# 3. Convert circuits list to a batched tensor
circuits_tensor = encoding.circuits_to_tensor(circuits, encoder)
print(f"\nCircuits tensor shape: {circuits_tensor.shape} (num_circuits, max_depth, encoder.length)")
print("Circuit 0 encoded tensor (depth 1, zero-padded at layer 1):\n", circuits_tensor[0])
print("Circuit 1 encoded tensor (depth 2):\n", circuits_tensor[1])
```

---
### What's Next?
Now that we can represent error generators and encode circuits as tensors, proceed to **[Part 2: Error Propagation and Locality Filters](QPANN-ErrorPropagation.ipynb)** to learn how we propagate these errors through the circuit and filter them using device locality graphs.

```{code-cell} ipython3

```

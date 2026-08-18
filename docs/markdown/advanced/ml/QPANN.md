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

# Quantum Physics-Aware Neural Networks (QPANNs) — Overview

This notebook serves as a top-level landing page and overview for pyGSTi's **Quantum Physics-Aware Neural Network (QPANN)** subpackage, located in `pygsti.extras.ml`.

QPANNs are machine learning models designed for *quantum capability learning*—predicting how well a noisy quantum computer will execute a given quantum circuit. It is a direct implementation of the architecture introduced in the NeurIPS 2024 paper:
> D. Hothem, A. Miller, T. Proctor, *"What is my quantum computer good for? Quantum capability learning with physics-aware neural networks"*, NeurIPS 2024. [ArXiv Link](https://arxiv.org/abs/2406.05636)

## Why "Physics-Aware"?
Traditional "black-box" machine learning models for predicting circuit performance ignore the rich physical structure of quantum devices and quantum circuits. They suffer from poor sample efficiency and struggle to generalize to deep or wide circuits.

A QPANN avoids this by splitting the prediction task into two components:
1. **A Trainable Neural Network ($N$):** A vectorized, highly localized multi-layer perceptron (MLP) that maps the local gate context of each circuit layer to the predicted *rates* of a set of local "elementary error generators" (including coherent rotations, stochastic Pauli flips, Pauli-correlation, and active errors).
2. **A Fixed, Non-Trainable Physics Layer ($f$):** A function that propagates these layer-wise error rates through the Clifford circuit structure to compute a first-order approximation of the circuit's final outcome probabilities or fidelity. This propagation is done using efficient Clifford error propagation (which scales polynomially with qubit count).

By embedding the laws of quantum mechanics and error propagation directly into the model's architecture, a QPANN:
* Generalizes spectacularly to circuits much deeper and wider than those in the training set.
* Learns from extremely small datasets (high sample efficiency).
* Yields physically interpretable predictions—the learned weights directly represent the underlying gate-dependent error rates on the physical qubits.

---

## Tutorial Series Structure
To help you get started with building and training QPANNs, we have organized this tutorial series into three subsequent parts:

1. **[Part 1: Error Generators and Circuit Encoding](QPANN-CircuitEncoding.ipynb)**  
   Learn how to mathematically represent candidate noise processes (elementary error generators) and convert quantum circuits into multi-hot tensors suitable for input to neural networks.
2. **[Part 2: Error Propagation and Locality Filters](QPANN-ErrorPropagation.ipynb)**  
   Explore how error generators propagate through Clifford circuits, how we calculate the first-order outcome sensitivity (alpha coefficients) of bitstring probabilities, and how to define "snippers" (locality filters) that restrict network connectivity to local device neighborhoods.
3. **[Part 3: Model Building, Synthetic Data, and Training](QPANN-Training.ipynb)**  
   Bring all the pieces together. Learn how to construct a QPANN model in Keras, build an independent, physically realistic noisy simulator (including Hamiltonian, Stochastic, Pauli-Correlation, and Active error types) to generate ground-truth training data, and train the QPANN to recover the known-but-hidden physical error rates.

---

## Quick Verification
Let's run a quick import check to verify that pyGSTi's ML subpackage and its Keras/TensorFlow dependencies are correctly configured and available in your environment.

```{code-cell} ipython3
import pygsti
from pygsti.extras import ml
import numpy as np
import tensorflow as tf
import keras

print("pyGSTi version:", pygsti.__version__)
print("TensorFlow version:", tf.__version__)
print("Keras version:", keras.__version__)
print("All imports successful! You are ready to proceed to Part 1.")
```

---
### What's Next?
Proceed to **[Part 1: Error Generators and Circuit Encoding](QPANN-CircuitEncoding.ipynb)** to learn about the fundamental representations used by QPANNs.

```{code-cell} ipython3

```

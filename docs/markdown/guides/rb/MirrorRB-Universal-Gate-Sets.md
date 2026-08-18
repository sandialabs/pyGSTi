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

This tutorial contains a few details on how to run *Mirror randomized benchmarking* with universal gate sets, that are not covered in the [RB overview tutorial](HowRBWorks) or the [Clifford MRB tutorial](MirrorRB).

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

`pspec` (the device to benchmark), `k` (how many circuits to sample at each depth) and `qubits` (which qubits to benchmark) mean the same thing here as in every RB protocol; [how RB works](HowRBWorks) introduces them. What is protocol-specific is `depths`. For Mirror RB the depths must be *even* integers: they count the total layers in the "compute" and "uncompute" sub-circuits. One extra constraint applies to `pspec` here — universal-gate-set MRB currently requires `Gzr` and `Gxpi2` to be among the gate names.

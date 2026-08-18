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

# Mirror RB

This tutorial contains a few details on how to run *Mirror randomized benchmarking* that are not covered in the [RB overview tutorial](HowRBWorks).

## What is Mirror RB? 

Like Direct RB, Mirror RB is a streamlined RB method partly inspired by [Clifford RB](CliffordRB). It has the same core purpose as Clifford RB - quantifying average gate performance - but it is feasable on more qubits, and it provides more directly useful information. However, Mirror RB is even more streamlined than Direct RB, making it feasable on 10s or 100s of qubits (it is possible to holistically benchmark around $1/\epsilon$ qubits if the error rate per-gate per-qubit is around $\epsilon$).

Mirror RB can be implemented with non-Clifford gates as well (see the [Universal Gate Set MRB tutorial](MirrorRB-Universal-Gate-Sets)). A depth $m$ ($m\geq 0$) mirror RB circuit consists of:

1. A uniformly random 1-qubit Clifford gate on every qubit. 
2. A "compute" circuit consisting of $m/2$ independently sampled layers of the native Clifford gates in the device, sampled according to a user-specified distribution $\Omega$. Each of these layers is proceeded by a uniformly random Pauli gate on each qubit.
3. A layer of uniformly random Pauli gates.
4. An "uncompute" circuit consisting of the $m/2$ layers from step (2) in the reverse order with each gate replaced with its inverse. Each of these layers is followed by a uniformly random Pauli gate on each qubit, with these Pauli gates sampled *independently* of the Pauli layers in step (2).
5. The inverse of the random 1-qubit Clifford gates in step (1).

This construction means that Mirror RB circuits can be much shorter than Clifford RB circuits, or Direct RB circuits. Yet they still have the core randomization properties of both Clifford and Direct RB.

**More information on Mirror RB will be added to this tutorial in a future release.**

```{code-cell} ipython3
from __future__ import print_function #python 2 & 3 compatibility
import pygsti
from pygsti.processors import QubitProcessorSpec as QPS
from pygsti.processors import CliffordCompilationRules as CCR
```

## Creating a Mirror RB experiment design

Generating a Mirror RB experiment design is very similar to creating a Direct RB design. The only difference is that there is no compilation in a Mirror RB circuit, so there is no compilation algorithm to tweak.

### 1. Generic RB inputs

`pspec` (the device to benchmark), `k` (how many circuits to sample at each depth) and `qubits` (which qubits to benchmark) mean the same thing here as in every RB protocol; [how RB works](HowRBWorks) introduces them. What is protocol-specific is `depths`. For Mirror RB the depths must be *even* integers: they count the total layers in the "compute" and "uncompute" sub-circuits, not counting the randomized Pauli gates.

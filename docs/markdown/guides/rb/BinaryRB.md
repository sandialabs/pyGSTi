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

This tutorial contains a few details on how to run *Binary randomized benchmarking (BiRB)* that are not covered in the [RB overview tutorial](HowRBWorks).

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

`pspec` (the device to benchmark), `k` (how many circuits to sample at each depth) and `qubits` (which qubits to benchmark) mean the same thing here as in every RB protocol; [how RB works](HowRBWorks) introduces them. What is protocol-specific is `depths`. For Binary RB the depths must be non-negative integers, counting the randomly sampled core layers in a circuit.

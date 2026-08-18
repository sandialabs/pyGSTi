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

# Clifford RB

This tutorial contains a few details on how to run [Clifford randomized benchmarking](http://journals.aps.org/prl/abstract/10.1103/PhysRevLett.106.180504) that are not covered in the [RB overview tutorial](HowRBWorks). 

## What is Clifford RB? 

By Clifford randomized benchmarking we mean RB of the $n$-qubit Clifford group, as defined by Magesan *et al.* in [*Scalable and Robust Benchmarking of Quantum Processes*](http://journals.aps.org/prl/abstract/10.1103/PhysRevLett.106.180504). This protocol is routinely run on 1 and 2 qubits.

```{code-cell} ipython3
import pygsti
from pygsti.processors import QubitProcessorSpec as QPS
from pygsti.processors import CliffordCompilationRules as CCR
import numpy as np
```

## Creating a Clifford RB experiment design

The only aspects of running Clifford RB with pyGSTi that are not covered in the [RB overview tutorial](HowRBWorks) are some subtleties in generating a Clifford RB experiment design (and what those subtleties mean for interpretting the results). To cover these subtleties, here we go through the inputs used to generate a Clifford RB experiment design in more detail.

### 1. Generic RB inputs

`pspec` (the device to benchmark), `k` (how many circuits to sample at each depth) and `qubits` (which qubits to benchmark) mean the same thing here as in every RB protocol; [how RB works](HowRBWorks) introduces them. What is protocol-specific is `depths`. For Clifford RB on $n$ qubits, the RB depth is the number of (uncompiled) $n$-qubit Clifford gates in the sequence minus two. That convention is chosen so that zero is the minimum RB depth for every RB method in pyGSTi.

Every other argument to the Clifford RB experiment design generation function is optional.

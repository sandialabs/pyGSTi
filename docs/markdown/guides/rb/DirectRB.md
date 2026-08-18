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

# Direct RB

This tutorial contains a few details on how to run [Direct randomized benchmarking](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.123.030503) that are not covered in the [RB overview tutorial](HowRBWorks).

## What is Direct RB? 

In essence, Direct RB is a streamlined, generalized version of the popular [Clifford RB](CliffordRB) method. It has the same core purpose - quantifying average gate performance - but it is feasable on more qubits, and it provides more directly useful information.

The basic requirements for running Clifford RB and Direct RB are the same. Both methods can be implemented on a set of $n$ qubits whenever the $n$-qubit Clifford group can be **generated** by the native gates on those $n$ qubits. Clifford RB runs circuits containing $m+1$ uniformly random $n$-qubit Cliffords followed by the unique inversion $n$-qubit Clifford gate (all of which must be compiled into the native gates of the device), where $m \geq 0$. In contrast, Direct RB circuits consist of:

1. A sub-circuit that generates a uniformly random $n$-qubit stabilizer state. 
2. $m$ independently sampled layers of the native gates in the device, with these layers sampled according to a user-specified distribution $\Omega$ over all possible circuit layers. 
3. A sub-circuit that maps the ideal output of the preceeding circuit to a uniformly random computational basis state (or, if preferred, to the all-zeros state).

This construction means that Direct RB circuits can be shorter than Clifford RB circuits - for the same $m$ a Direct RB circuit is typically much shorter, including for the shortest allowed depth $m=0$. This means that Direct RB can be run on more qubits (without just obtaining a useless, entirely decohered output). But Direct RB circuits still contain sufficient randomization (if $\Omega$ is chosen appropriately) to retain the core features of Clifford RB (exponential decays, etc).

For more information on what Direct RB is and why it is useful, see [*Direct randomized benchmarking for multi-qubit devices*](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.123.030503).

```{code-cell} ipython3
from __future__ import print_function #python 2 & 3 compatibility
import pygsti
from pygsti.processors import QubitProcessorSpec as QPS
from pygsti.processors import CliffordCompilationRules as CCR
```

## Creating a Direct RB experiment design

The data analysis in Direct RB is exactly as in Clifford RB, and how to do this analysis is covered in the [RB overview tutorial](HowRBWorks). The differences and flexibility in Direct RB are all at the experiment design stage, and so this is what is covered in this tutorial. 

### 1. Generic RB inputs

`pspec` (the device to benchmark), `k` (how many circuits to sample at each depth) and `qubits` (which qubits to benchmark) mean the same thing here as in every RB protocol; [how RB works](HowRBWorks) introduces them. What is protocol-specific is `depths`. For Direct RB, the depths are the number of layers in the "core" circuit outlined in step (2) above, and can be any non-negative integers.

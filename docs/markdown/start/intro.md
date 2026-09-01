---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.19.3
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# pyGSTi

```{note}
You are reading the documentation for **pyGSTi version {{ pygsti_version }}**. Use the version selector in the bottom-right corner of the page to switch between tagged releases and the latest development build.
```

pyGSTi is a Python framework for modeling and characterizing quantum information processors, from single qubits to devices with tens or hundreds of them. It was built to perform gate set tomography, which is where the name comes from, and GST is still its most detailed capability. It now also runs randomized benchmarking, volumetric and mirror-circuit benchmarks, drift detection, idle tomography and robust phase estimation, and it will build noise models, simulate circuits and generate interactive reports along the way.

## Install and check it works

```bash
pip install pygsti
```

See [installation](Install) for conda, optional dependencies and building from source. To track the development branch these docs are built from:

```bash
pip install git+https://github.com/sandialabs/pyGSTi.git@develop
```

Then build an ideal one-qubit model and ask it for a circuit's outcome probabilities:

```python
import pygsti
from pygsti.modelpacks import smq1Q_XYI

model = smq1Q_XYI.target_model()                    # ideal 1-qubit model
circuit = pygsti.circuits.Circuit([('Gxpi2', 0)])   # a single X(pi/2) gate
model.probabilities(circuit)                        # ~ {('0',): 0.5, ('1',): 0.5}
```

## Where to go next

This documentation is arranged in three tiers, and which one you want depends on what you are doing rather than on how much you already know.

- **[Start here](Index)** is a short guided path from a fresh install to a characterization result you can read. It is the whole of what most people need.
- **[Characterization guides](../guides/Index)** is the practitioner layer: one chapter per protocol, plus the workflow, modeling and analysis chapters that all of them draw on.
- **[Advanced topics and internals](../advanced/Index)** is for extending pyGSTi, working with unusual devices, or research use. If you are characterizing a device with one of the protocols above, you do not need any of it.

If you have a specific problem and want to know which few lines of Python solve it, try [troubleshooting](../guides/analysis/Troubleshooting). If that does not cover it, email us at pygsti@sandia.gov or open an issue on [GitHub](https://github.com/sandialabs/pyGSTi).

## Citing pyGSTi

If pyGSTi contributed to work you are publishing, please see [citing pyGSTi](../advanced/Citing) for the references to use.

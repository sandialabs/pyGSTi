# pyGSTi

```{note}
You are reading the documentation for **pyGSTi version {{ pygsti_version }}**. Use the version selector in the bottom-right corner of the page to switch between tagged releases and the latest development build.
```

pyGSTi is a Python framework for modeling and characterizing quantum information processors, from single qubits to devices with tens or hundreds of them. It was built to perform gate set tomography, which is where the name comes from, and GST is still its most detailed capability. It now also runs randomized benchmarking, volumetric and mirror-circuit benchmarks, drift detection, idle tomography and robust phase estimation, and it will build noise models, simulate circuits and generate interactive reports along the way.

## Getting started

[Installing pyGSTi](Install) covers installation, optional dependencies, and building from source. Then choose [Your first GST run](FirstGST) for a fitted gate model or [Your first benchmark](FirstBenchmark) for a benchmarking example.

## Where to go next

This documentation is arranged in six sections, and which one you want depends on what you are doing rather than on how much you already know.

- **[Start here](Index)** is a short guided path from a fresh install to a characterization result you can read.
- **[Characterization guides](../guides/Index)** is the practitioner layer: one chapter per protocol, plus the workflow and analysis chapters that all of them draw on.
- **[Defining models](../guides/models/DeviceModels)** covers explicit and implicit models, their parameterizations, and adding noise to describe your device.
- **[Advanced capabilities](../advanced/Index)** covers specialist protocols, machine-learned error models, and circuits and data exchanged with Cirq or IBM Q.
- **[Core objects](../internals/Index)** is the machinery underneath: operators, conventions, simulators, extension points and the figures that reports are assembled from. Use it when you need to inspect or extend those components.
- **[Reference](../reference/Index)** holds the [FAQ](../reference/Troubleshooting), [citing pyGSTi](../reference/Citing), [migrating from the function-based API](../reference/FromFunctionAPI) and the API reference.

If you have a specific problem and want to know which few lines of Python solve it, try the [FAQ](../reference/Troubleshooting). If that does not cover it, email us at pygsti@sandia.gov or open an issue on [GitHub](https://github.com/sandialabs/pyGSTi).

## Citing pyGSTi

If pyGSTi contributed to work you are publishing, please see [citing pyGSTi](../reference/Citing) for the references to use.

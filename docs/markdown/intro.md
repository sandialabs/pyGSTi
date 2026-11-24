# pyGSTi

```{note}
You are reading the documentation for **pyGSTi version {{ pygsti_version }}**. Use the version selector in the bottom-right corner of the page to switch between tagged releases and the latest development build.
```

pyGSTi is a Python framework for modeling and characterizing quantum information processors, from single qubits to devices with tens or hundreds of them. It was built to perform gate set tomography, which gives the package its name. It now also runs randomized benchmarking, volumetric and mirror-circuit benchmarks, drift detection, idle tomography and robust phase estimation, and it will build noise models, simulate circuits and generate interactive reports along the way.

## Getting started

[Installing pyGSTi](start/Install) covers installation, optional dependencies, and building from source. Then choose [Your first GST run](start/FirstGST) for a fitted gate model or [Your first benchmark](start/FirstBenchmark) for a benchmarking example.

## Where to go next

This documentation is arranged in five sections, and which one you want depends on what you are doing rather than on how much you already know.

- **[Start here](start/Index)** is a short guided path from a fresh install to a characterization result you can read.
- **[Characterization guides](guides/Index)** is the practitioner layer: one chapter per protocol, plus the workflow and analysis chapters that all of them draw on.
- **[Defining models](models/Index)** covers processor specifications, explicit and implicit models, their parameterizations, adding noise to describe your device, and simulating data from a model.
- **[Core objects](core/Index)** is the machinery underneath: operators, conventions, simulators, extension points and the figures that reports are assembled from. Use it when you need to inspect or extend those components.
- **[Reference](reference/Index)** holds the [FAQ](reference/FAQ), [citing pyGSTi](reference/Citing), [migrating from the function-based API](reference/FromFunctionAPI) and the API reference.

For answers to common usage questions, try the [FAQ](reference/FAQ). For other questions, email us at pygsti@sandia.gov or open an issue on [GitHub](https://github.com/sandialabs/pyGSTi).

## Citing pyGSTi

If pyGSTi contributed to work you are publishing, please see [citing pyGSTi](reference/Citing) for the references to use.

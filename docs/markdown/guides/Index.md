# Characterization guides

These guides cover experiment design, protocol options, and interpretation of results. [Start here](../start/Index) provides introductory examples; use the chapters below when you're ready to configure an analysis for your device and data.

## Pick the chapter for your protocol

- **Gate set tomography** estimates gates, state preparations, and measurements self-consistently. Four chapters cover [designing the circuits](gst/GSTDesigns), [running the fit](gst/RunningGST), [judging the fit](gst/JudgingTheFit), and [richer device models](gst/RicherDeviceModels) for mid-circuit measurement, qutrits, leakage, and time dependence.
- [Randomized benchmarking](rb/HowRBWorks) estimates error rates from random circuits. Choose among Clifford, direct, mirror, and binary RB, simultaneous benchmarks, and qudit protocols.
- [Volumetric benchmarking](benchmarks/VolumetricBenchmarks) maps performance across circuit widths and depths. [Mirror circuit fidelity estimation](benchmarks/MirrorFidelityEstimation) estimates fidelity for a supplied circuit.
- [Comparing datasets](drift/ComparingDataSets) tests whether counts for the same circuits are statistically consistent across times, devices, or experimental settings.
- [Drift characterization](drift/DriftCharacterization) detects and characterizes changes in device behavior using time-stamped data.

## Shared workflows and analysis

- [Measurement workflow](workflow/Workflow) introduces the experiment design, data, protocol, and results objects used throughout these guides.
- [Results and reporting](analysis/Results) covers accessing estimates, computing metrics and error bars, and generating reports. It also links to [gauge freedom in practice](analysis/GaugeFreedom).

## Defining your device model

[Defining models](models/DeviceModels) covers explicit and implicit models, parameterizations, and noise construction. Start here to describe your own gate set or customize a model from a model pack.

## Working with experimental data

[Getting your own data in](../start/YourOwnData) shows how to load measured counts. The [IBM Q integration](../advanced/interop/IBMQ) can submit an experiment design and collect its counts through Qiskit.

[Advanced capabilities](../advanced/Index) covers specialist protocols, machine-learned error models, and platform integrations. [Core objects](../internals/Index) covers model components, simulation engines, and extension interfaces. The [FAQ](../reference/Troubleshooting) answers common usage questions.

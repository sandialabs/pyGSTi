# Characterization guides

These guides cover experiment design, protocol options, and interpretation of results. [Start here](../start/Index) provides introductory examples; use the chapters below when you're ready to configure an analysis for your device and data.

## Pick the chapter for your protocol

- **Gate set tomography** estimates gates, state preparations, and measurements self-consistently. Four chapters cover [designing the circuits](gst/GSTDesigns), [running the fit](gst/RunningGST), [judging the fit](gst/JudgingTheFit), and [richer device models](gst/RicherDeviceModels) for mid-circuit measurement, qutrits, leakage, and time dependence.
- [Randomized benchmarking](rb/HowRBWorks) estimates error rates from random circuits. Choose among Clifford, direct, mirror, and binary RB, simultaneous benchmarks, and qudit protocols.
- [Benchmarks beyond RB](benchmarks/VolumetricBenchmarks) covers volumetric benchmarks, which map performance across circuit widths and depths; [mirror circuit fidelity estimation](benchmarks/MirrorFidelityEstimation) for a supplied circuit; and [parity benchmarking](../advanced/specialist/ParityBenchmarking) for a parity-check circuit.
- [Comparing datasets](drift/ComparingDataSets) tests whether counts for the same circuits are statistically consistent across times, devices, or experimental settings.
- [Robust phase estimation](../advanced/specialist/RobustPhaseEstimation) estimates single-qubit rotation angles to high precision.
- [Drift characterization](drift/DriftCharacterization) detects and characterizes changes in device behavior using time-stamped data.

## Shared workflows and analysis

- [Circuits, designs, and data](workflow/Workflow) covers the circuits, experiment designs, and data sets that every protocol takes as input, how to save and load them, and how to run circuits and collect counts through Cirq or IBM Q.
- [Results and reporting](analysis/Results) covers accessing estimates, computing metrics and error bars, and generating reports. It also links to [gauge freedom in practice](analysis/GaugeFreedom).

## Defining your device model

[Defining models](models/DeviceModels) covers processor specifications, explicit and implicit models, noise construction, computing probabilities and simulated data from a model, and machine-learned error models. Start here to describe your own gate set or customize a model from a model pack.

## Beyond these guides

[Core objects](../internals/Index) covers model components, simulation engines, and extension interfaces. The [FAQ](../reference/Troubleshooting) answers common usage questions.

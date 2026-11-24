# Characterization guides

These guides cover experiment design, protocol options, and interpretation of results. [Start here](../start/Index) provides introductory examples; use the chapters below when you're ready to configure an analysis for your device and data.

## Pick the chapter for your protocol

- **Gate set tomography** estimates gates, state preparations, and measurements self-consistently. Four chapters cover [designing the circuits](gst/designs/Index), [running the fit](gst/running/Index), [judging the fit](gst/fit/Index), and [richer device models](gst/richer/Index) for mid-circuit measurement, qutrits, leakage, and time dependence.
- [Randomized benchmarking](rb/Index) estimates error rates from random circuits. Choose among Clifford, direct, mirror, and binary RB, simultaneous benchmarks, and qudit protocols.
- [Benchmarks beyond RB](benchmarks/Index) covers volumetric benchmarks, which map performance across circuit widths and depths; [mirror circuit fidelity estimation](benchmarks/MirrorFidelityEstimation) for a supplied circuit; and [parity benchmarking](benchmarks/ParityBenchmarking) for a parity-check circuit.
- [Comparing datasets](ComparingDataSets) tests whether counts for the same circuits are statistically consistent across times, devices, or experimental settings.
- [Robust phase estimation](RobustPhaseEstimation) estimates single-qubit rotation angles to high precision.
- [Drift characterization](drift/Index) detects and characterizes changes in device behavior using time-stamped data.

## Shared workflows and analysis

- [Circuits, designs, and data](data/Index) covers the circuits, experiment designs, and data sets that every protocol takes as input, how to save and load them, and how to run circuits and collect counts through Cirq or IBM Q.
- [Results and reporting](analysis/Index) covers accessing estimates, computing metrics and error bars, and generating reports. It also links to [gauge freedom in practice](analysis/GaugeOptimization).

## Defining your device model

[Defining models](../models/Index) covers processor specifications, explicit and implicit models, noise construction, computing probabilities and simulated data from a model, and machine-learned error models. Start here to describe your own gate set or customize a model from a model pack.

## Beyond these guides

[Core objects](../core/Index) covers model components, simulation engines, and extension interfaces. The [FAQ](../reference/FAQ) answers common usage questions.

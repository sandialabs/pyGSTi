# Protocols

These tutorials cover the QCVV and benchmarking protocols in `pyGSTi` and how to run them on data. For a high-level tour of every protocol and how to invoke it, start with the [protocols overview](overview/00-Protocols).

- [Gate set tomography (GST)](gst/Overview) — pyGSTi's flagship protocol for full, self-consistent characterization of a gate set.
- [Linearized GST](protocols/LinearizedGST) — scalable estimation of sparse (coherent + Pauli-stochastic) error models on many qubits from shallow circuits, including mid-circuit measurements.
- [Linearized GST II: designing the circuits](protocols/LinearizedGST_2) — constructing maximally informative probe circuits from the error model and the processor's gates, instead of sampling random circuits.
- [Linearized GST III: optimal experiments for a fixed shot budget](protocols/LinearizedGST_3) — the shot-noise statistics and Fisher information of each candidate circuit, a figure of merit for an experiment, and the convex optimization of which circuits to run with how many shots, with the matched generalized-least-squares estimator.
- [Randomized benchmarking (RB)](rb/Overview) — Clifford, Direct, Mirror, and Binary RB for scalable error-rate estimation.
- [Robust phase estimation (RPE)](protocols/RobustPhaseEstimation) — precise estimation of gate rotation angles.
- [Drift characterization](protocols/DriftCharacterization) — detecting time-dependent (non-Markovian) drift across data passes.
- [Volumetric benchmarks](protocols/VolumetricBenchmarks) — summarizing capability over circuit shape (width × depth).
- [Parity benchmarking](protocols/ParityBenchmarking) — parity-based benchmarking of multi-qubit circuits.


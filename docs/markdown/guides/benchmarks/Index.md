# Benchmarks beyond RB

These benchmarks report something other than a decay-curve error rate. Volumetric benchmarks compare performance across circuit widths and depths; mirror circuit fidelity estimation targets one circuit; parity benchmarking scores a parity-check circuit against a reference.

- [Running volumetric benchmarks](VolumetricBenchmarks) builds direct-RB and mirror-circuit experiments, summarizes their data, and plots capability regions.
- [Mirror circuit fidelity estimation](MirrorFidelityEstimation) estimates the process fidelity of a supplied circuit mapped to U3 and controlled-X gates. The example uses Qiskit.
- [Parity benchmarking](ParityBenchmarking) computes the weight-X disturbance between a reference and a test data set, illustrated with a four-qubit parity check. It requires cvxpy.

For protocols that report an error rate from a decay curve, see [Randomized benchmarking](../rb/Index).

# Randomized benchmarking

Randomized benchmarking estimates error rates from the decay of outcomes across random circuits. [Running an RB experiment](RBWorkflow) demonstrates the shared design, data, and analysis workflow using Clifford RB.

Choose a protocol based on what you want to benchmark:

- [Clifford RB](CliffordRB) benchmarks compiled Clifford operations. [Interleaved RB](InterleavedRB) compares reference and interleaved experiments to estimate the error of a particular gate.
- [Direct RB](DirectRB) benchmarks layers sampled from the device's native gate set.
- [Mirror RB](MirrorRB) benchmarks native Clifford layers with a circuit followed by its inverse, reducing the compilation overhead of Clifford RB. [Mirror RB with universal gate sets](MirrorRB-Universal-Gate-Sets) covers the extension beyond Clifford gate sets.
- [Binary RB](BinaryRB) uses gate-efficient state preparation and measurement to benchmark randomly sampled layers on larger sets of qubits.
- [Simultaneous RB](SimultaneousRB) runs benchmarks on multiple subsets of qubits at once.
- [SU(2) qudit RB](SU2QuditRB) benchmarks collective rotations of a spin or qudit.

[Samplers](Samplers) covers the layer distributions used to generate circuits. [Simulating RB data](SimulatingRBData) shows how to obtain artificial counts from explicit and implicit noise models.

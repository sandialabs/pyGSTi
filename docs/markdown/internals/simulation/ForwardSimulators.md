# Forward simulators

A forward simulator computes circuit outcome probabilities from a model. [Choosing a forward simulator](ChoosingForwardSimulators) introduces evolution types and compares simulation engines as the number of qubits grows.

- [CHP interface](StabilizerCHP) demonstrates stabilizer simulation with the CHP evolution type and the pure-Python `chp_sim` backend.
- [Error generator propagation](ErrorGeneratorPropagation) propagates error generators through Clifford circuits.
- [Parallelization for error generator propagation](ErrorGeneratorPropagation-Parallelization) configures worker pools, reuse, and parallelization thresholds.
- [Error generator polynomials](ErrorGeneratorPolynomials) constructs symbolic expressions that can be evaluated at different noise parameters.

To sample experimental counts from model probabilities, see [Simulating data](../../guides/workflow/SimulatingData).

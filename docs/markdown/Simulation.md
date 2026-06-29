# Simulation

These tutorials cover how `pyGSTi` simulates circuits — computing the outcome probabilities a model predicts. Simulation underlies model fitting, but you can also use it standalone for forward simulation. Start with circuit simulation, then see how to pick a forward-simulation method and generate synthetic data.

- [Circuit simulation](simulation/CircuitSimulation) — compute outcome probabilities for a circuit from a model, including stabilizer/CHP and explicit- vs. implicit-model variants.
- [Data simulation](simulation/DataSimulation) — turn model predictions into simulated experimental datasets (finite-shot samples).
- [Error generator propagation](simulation/ErrorGeneratorPropagation) — propagate error generators through the layers of a circuit.
- [Forward simulation types](simulation/ForwardSimulationTypes) — the available forward simulators (matrix, map, term, …) and when to use each.

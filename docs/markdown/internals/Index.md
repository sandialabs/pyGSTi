# Core objects

These chapters explain the model components, conventions, simulation engines, and reporting objects used by pyGSTi's protocols. Use them to customize an analysis, inspect how its objects work, or extend the library.

- [Operators](models/Operators) covers operation, state-preparation, and measurement classes, their parameterizations, and custom implementations. For complete device models, start with [Defining models](../guides/models/DeviceModels).
- [Bases and state spaces](conventions/Bases) explains matrix-basis conventions, subsystem labels, and state-space structure.
- [Forward simulators](simulation/ForwardSimulators) covers computing circuit probabilities, stabilizer simulation, and numerical and symbolic error-generator propagation.
- [Low-level GST](extending/LowLevelGST) provides direct access to fitting and gauge-optimization algorithms for analyses assembled below the protocol interface.
- [Workspace tables and plots](figures/WorkspaceFigures) shows how to create individual report components and add controls that select their inputs.

For analyses organized around an experiment design and a protocol, see the [Characterization guides](../guides/Index).

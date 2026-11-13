# Defining models

A model specifies the operations, state preparations, and measurements used to predict circuit outcomes. Start with [Building a noisy model](BuildingModels) for a short example comparing explicit and implicit models, adding noise, and computing probabilities.

- [Explicit models](../../internals/models/ExplicitModels) builds a model operation by operation. Use this for small gate sets whose layers you want to specify directly.
- [Implicit models](MultiQubitModels) constructs circuit layers from local gates and noise rules. Use this for devices with gates acting on different subsets of qubits.
- [Model noise](ModelNoise) adds depolarizing, stochastic, and Lindblad noise, including noise on state preparation and measurement.

For a standard gate set, [Model packs](../../start/TargetModels) may already provide your target model. [Key concepts](../../start/KeyConcepts) explains the parameterizations and error generators used in these notebooks.

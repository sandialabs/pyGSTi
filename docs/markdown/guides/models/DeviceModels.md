# Defining models

A model specifies the operations, state preparations, and measurements used to predict circuit outcomes. Start with [Building a noisy model](BuildingModels) for a short example comparing explicit and implicit models, adding noise, and computing probabilities.

- [Processor specifications](../workflow/DescribeYourDevice) describes a device's qubits, gate names, connectivity, and compilation rules; most model construction functions take one.
- [Explicit models](../../internals/models/ExplicitModels) builds a model operation by operation. Use this for small gate sets whose layers you want to specify directly.
- [Implicit models](MultiQubitModels) constructs circuit layers from local gates and noise rules. Use this for devices with gates acting on different subsets of qubits.
- [Model noise](ModelNoise) adds depolarizing, stochastic, and Lindblad noise, including noise on state preparation and measurement.
- [Computing probabilities and simulating data](../workflow/SimulatingData) computes outcome probabilities from a model and samples counts from them when you don't yet have experimental data.
- [Quantum physics-aware neural networks](../../advanced/ml/QPANN) trains a model whose predicted error-generator rates are used to estimate circuit outcome probabilities. It requires TensorFlow and Keras, and is experimental.

For a standard gate set, [Model packs](../../start/TargetModels) may already provide your target model. [Key concepts](../../start/KeyConcepts) explains the parameterizations and error generators used in these notebooks.

# Advanced capabilities

These notebooks cover targeted characterization protocols, machine-learned error models, and integration with other quantum software. Choose the area that matches your experiment or analysis:

- [Specialist protocols](specialist/Index) covers robust phase estimation for single-qubit rotation angles and parity benchmarking for a four-qubit parity check.
- [Quantum physics-aware neural networks](ml/QPANN) trains a model whose predicted error-generator rates are used to estimate circuit outcome probabilities. The notebook covers circuit encoding, training, and validation, and requires TensorFlow and Keras.
- [Cirq integration](interop/Cirq) converts circuits between Cirq and pyGSTi and loads experiment results into a pyGSTi dataset.
- [IBM Q](interop/IBMQ) submits experiment designs to an IBM backend through Qiskit and collects counts for protocol analysis. The hardware workflow requires Qiskit and an IBM Q account.

For other experiment platforms, [Getting your own data in](../start/YourOwnData) explains how to load measured counts. To extend pyGSTi's model components or simulation machinery, see [Core objects](../internals/Index).

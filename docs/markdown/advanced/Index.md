# Do I need any of this?

Nothing in this part is required to run a characterization protocol and read the answer. If that is what you came for, [Start here](../start/Index) and the [characterization guides](../guides/Index) are the whole path, and you can leave this part unread.

What lives here is the machinery underneath that path, plus the corners of pyGSTi that only some readers need.

**Building models by hand.** [Operators](models/Operators) is the inventory of operator types and parameterizations — the thing to read when a model pack does not describe your device. [Custom operators](models/CustomOperators) and [custom POVMs](models/CustomPOVMs) cover subclassing when nothing built in fits, and [tying parameters](models/TyingParameters) covers constraining parameters across model members.

**Conventions.** [Bases](conventions/Bases) and [state spaces](conventions/StateSpaces) pin down the representation choices that the rest of the documentation assumes. Read them when a matrix does not look the way you expected.

**Simulation.** [Forward simulators](simulation/ForwardSimulators) explains how pyGSTi turns a model into circuit probabilities and how to pick a simulator that scales to your qubit count. The remaining pages cover the stabilizer/CHP path, generating simulated RB data, and the error-generator propagation machinery.

**Extending pyGSTi.** [Low-level GST](extending/LowLevelGST) drops beneath the `Protocol` layer to the optimization routines themselves, and [the gauge-optimization reference](extending/GaugeOptReference) covers adding new gauge objectives.

**Specialist protocols.** [Time-dependent GST](specialist/TimeDependentGST) and the pages under it cover protocols with a narrow audience: idle tomography, robust phase estimation, qudit and qutrit work, parity benchmarking, and hand-built leakage models.

**Everything else.** Machine-learned error models ([QPANN](ml/QPANN)), the [figure and report internals](figures/WorkspaceFigures), [Cirq interoperability](interop/Cirq), and the [migration table](migration/FromFunctionAPI) for the old function-based API.

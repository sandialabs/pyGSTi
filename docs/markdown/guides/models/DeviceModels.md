---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.19.3
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Modeling a noisy device

A `Model` maps circuits to outcome probabilities. GST fits one to data, model testing scores a fixed one against data, and simulated data is sampled from one. Each of those starts from a *target model*, whose operations are the perfect unitaries the device is supposed to implement. Two choices remain: the kind of model, and the noise to put on it.

## Choosing a model

There are two routes to a target model. For the standard one- and two-qubit gate sets, import a [model pack](../../start/TargetModels) and call its `target_model()`; the pack also carries the fiducials and germs that GST needs, which is why it's the route to prefer whenever one fits your gates. For anything else, describe the device with a [`QubitProcessorSpec`](../workflow/DescribeYourDevice) (qubit labels, gate names, which qubits each gate can act on) and hand it to a construction function such as `create_explicit_model` or `create_crosstalk_free_model`. Either way the result is noise-free until you say otherwise.

An explicit model is a dictionary of layer operations: one $4^N \times 4^N$ operation for each circuit layer it can simulate, and a `KeyError` for any layer it can't, including two of its own gates in parallel. That is fine at one or two qubits and impractical past two or three. An implicit model stores a small set of building blocks (the gates) plus rules for assembling them, and builds each layer operation on demand, so parallel gates are no trouble and the storage grows with the number of gates rather than the number of possible layers. [Implicit models](MultiQubitModels) covers the two implicit kinds: crosstalk-free models, in which a gate's noise stays on its target qubits, and cloud-noise models, in which it may spill onto a neighborhood around them.

Noise goes on a model for three reasons: to [simulate data](../workflow/SimulatingData) that behaves like a real device, to [test](../analysis/ModelTesting) a hypothesis about the device's errors against data you already have, and to give GST a starting point (`GateSetTomography(initial_model=...)`), since the model's parameterization is the space the fit searches. The construction functions take `depolarization_strengths`, `stochastic_error_probs` and `lindblad_error_coeffs` dictionaries keyed by gate or layer label, and explicit models also have `depolarize` and `rotate` methods for the quick version. [Model noise](ModelNoise.md) covers those dictionaries in full: noise on state preparation and measurement, noise off the target qubits, and how the noise is parameterized.

## A short example

The one-qubit model pack with $X(\pi/2)$, $Y(\pi/2)$ and idle gates (`smq1Q_XYI`) gives an explicit model. Its default parameterization is "full", one parameter per matrix or vector element. "Full TP" pins the first row of each gate and the first element of the state preparation, and makes the measurement's effects sum to the identity (so the last effect is no longer free); it is what you'd normally fit with.

```{code-cell} ipython3
import pygsti
from pygsti.modelpacks import smq1Q_XYI

target = smq1Q_XYI.target_model()
print(type(target).__name__, "with", target.num_params, "parameters")
print(smq1Q_XYI.target_model("full TP").num_params, "parameters as full TP")
```

A processor specification gives an implicit one. With no noise arguments every gate is a static unitary and the model has no parameters at all; with a depolarization strength per gate name it has one parameter per gate name, shared across the qubits that gate acts on.

```{code-cell} ipython3
pspec = pygsti.processors.QubitProcessorSpec(2, ['Gxpi2', 'Gypi2', 'Gcnot'], geometry='line')
ideal = pygsti.models.create_crosstalk_free_model(pspec)
noisy = pygsti.models.create_crosstalk_free_model(
    pspec, depolarization_strengths={'Gxpi2': 0.02, 'Gypi2': 0.02, 'Gcnot': 0.05})
print(type(ideal).__name__, "with", ideal.num_params, "and", noisy.num_params, "parameters")
```

Simulating a circuit is a single call on either model.

```{code-cell} ipython3
c = pygsti.circuits.Circuit([('Gxpi2', 0), ('Gcnot', 0, 1)], line_labels=[0, 1])
print(c)
print("ideal:", ideal.probabilities(c))
print("noisy:", noisy.probabilities(c))
```

For an explicit model the quickest noise is uniform depolarization of every gate, which is how most of the simulated data in these docs is made. Two $X(\pi/2)$ gates take $|0\rangle$ to $|1\rangle$ exactly; with depolarization they don't.

```{code-cell} ipython3
depolarized = target.depolarize(op_noise=0.05)
c1 = pygsti.circuits.Circuit([('Gxpi2', 0)] * 2, line_labels=[0])
print("target:     ", target.probabilities(c1))
print("depolarized:", depolarized.probabilities(c1))
```

## Where next

[Implicit models](MultiQubitModels), then [model noise](ModelNoise.md). The Internals part holds the rest: [explicit models](../../internals/models/ExplicitModels) for building an explicit model by hand, member by member or from expression strings, including a two-qubit gate pyGSTi has no name for; and [operators](../../internals/models/Operators) for the operator classes, their parameterizations, and how small operators are composed and embedded into large ones.

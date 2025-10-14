---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.17.3
kernelspec:
  display_name: api_updates
  language: python
  name: api_updates
---

+++ {"colab_type": "text", "id": "DmNaCESptyWj"}

# Cirq Integration

+++

This notebook shows a simple example of how to use pyGSTi with Cirq. It has three sections:

1. Sets up pyGSTi.
2. Shows how pyGSTi circuits can be converted to Cirq circuits.
3. Shows how Cirq circuits can be converted into pyGSTi circuits.
4. Shows how the Cirq circuits can be run and the results loaded back into pyGSTi for analysis.

```{code-cell} ipython3
:colab: {}
:colab_type: code
:id: gw9bJiKmkST9

import cirq
import pygsti
from pygsti.modelpacks import smq1Q_XYI
from pygsti.circuits import Circuit
import numpy as np
import tqdm
```

+++ {"colab_type": "text", "id": "uugvjGQ3vR0z"}

## 1. Generate the GST circuits

+++ {"colab_type": "text", "id": "cWpHwZVtvejH"}

### Make target gate set $\{R_{X}(\pi/2), R_{Y}(\pi/2),I\}$

```{code-cell} ipython3
target_model = smq1Q_XYI.target_model()
```

+++ {"colab_type": "text", "id": "JVfiXBu4vqJV"}

### Preparation and measurement fiducials, germs

```{code-cell} ipython3
:colab: {}
:colab_type: code
:id: EPNxq24fvo6t

preps = smq1Q_XYI.prep_fiducials()
effects = smq1Q_XYI.meas_fiducials()
germs = smq1Q_XYI.germs()
```

+++ {"colab_type": "text", "id": "u9fHRr8Hv933"}

### Construct pyGSTi circuits

```{code-cell} ipython3
:colab: {}
:colab_type: code
:id: Jptyo9F0vx5N

max_lengths = list(np.logspace(0, 10, 11, base=2, dtype=int))
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 34
colab_type: code
id: SuvgxDpKwCul
outputId: 6654eeeb-3870-4b61-af43-0c66cb09169e
---
print(max_lengths)
```

```{code-cell} ipython3
:colab: {}
:colab_type: code
:id: qk-yEEWTwFJM

pygsti_circuits = pygsti.circuits.gstcircuits.create_lsgst_circuits(target_model, preps, effects, germs, max_lengths)
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 34
colab_type: code
id: 9vD8DXOPwHSV
outputId: 06e10aec-f7ab-4b7b-d0c6-242ce225d5a2
---
len(pygsti_circuits)
```

## 2. Convert to runable `cirq.Circuit`'s

+++

### Setup

+++

Now, we need to map the qubit names from pyGSTi (`0`, `1`, etc.) into cirq qubits. There's nothing special about `cirq.GridQubit(8, 3)`; it's just an example.

```{code-cell} ipython3
q0 = cirq.GridQubit(8, 3)
qubit_label_dict = {0: q0}
```

### Testing examples

+++

Do an example conversion.

```{code-cell} ipython3
pygsti_circuit = pygsti_circuits[111]
print('pyGSTi:')
print(pygsti_circuit)
print('Cirq:')
print(pygsti_circuit.convert_to_cirq(qubit_label_dict))
```

Do another example conversion.

```{code-cell} ipython3
pygsti_circuit = pygsti_circuits[90]
print('pyGSTi:')
print(pygsti_circuit)
print('Cirq:')
print(pygsti_circuit.convert_to_cirq(qubit_label_dict))
```

Now, lets try the same thing but specifing a wait duration for the idle operation.

```{code-cell} ipython3
wait_duration = cirq.Duration(nanos=100)
```

```{code-cell} ipython3
pygsti_circuit = pygsti_circuits[111]
print('pyGSTi:')
print(pygsti_circuit)
print('Cirq:')
print(pygsti_circuit.convert_to_cirq(qubit_label_dict, wait_duration))
```

```{code-cell} ipython3
pygsti_circuit = pygsti_circuits[90]
print('pyGSTi:')
print(pygsti_circuit)
print('Cirq:')
print(pygsti_circuit.convert_to_cirq(qubit_label_dict, wait_duration))
```

### The real thing

+++

Now, convert all the circuits.

```{code-cell} ipython3
cirq_circuits = [c.convert_to_cirq(qubit_label_dict, wait_duration) for c in tqdm.tqdm(pygsti_circuits)]
```

```{code-cell} ipython3
cirq_circuits
```

Note that we're missing the measurments and the first circuit is empty (it's should just be an idle). Otherwise, the results look good, and those things should be easy to fix.

+++

## 3. Convert Cirq circuits to pyGSTi circuits
We also have support for converting a cirq circuit to a pyGSTi circuit, which is demonstrated below.
Begin by constructing a cirq circuit directly.

```{code-cell} ipython3
#create to cirq qubit objects
qubit_00 = cirq.GridQubit(0,0)
qubit_01 = cirq.GridQubit(0,1)
#define a series of Moment objects, which fill the same role as circuit layers in pyGSTi.
moment1 = cirq.Moment([cirq.XPowGate(exponent=.5).on(qubit_00), cirq.I(qubit_01)])
moment2 = cirq.Moment([cirq.I(qubit_00), cirq.I(qubit_01)])
#This weird looking gate is the so-called N gate.
moment3 = cirq.Moment([cirq.PhasedXZGate(axis_phase_exponent=0.14758361765043326, 
                                         x_exponent=0.4195693767448338, 
                                         z_exponent=-0.2951672353008665).on(qubit_00),
                    cirq.I(qubit_01)])
moment4 = cirq.Moment([cirq.H(qubit_00), (cirq.T**-1).on(qubit_01)])
moment5 = cirq.Moment([cirq.CNOT.on(qubit_00, qubit_01)])
cirq_circuit_example = cirq.Circuit([moment1, moment2, moment3, moment4, moment5])
print(cirq_circuit_example)
```

To convert this into a pyGSTi circuit we can use the `from_cirq` class method of the Circuit class.

```{code-cell} ipython3
converted_cirq_circuit_default = Circuit.from_cirq(cirq_circuit_example)
print(converted_cirq_circuit_default)
```

Above you can see the result of converting the circuit using the default conversion settings. The classmethod has multiple options for customizing the returned pyGSTi circuit.
1. By default the method constructs a mapping between cirq qubit objects and pygsti qubit labels based on the type of cirq qubit provided. E.g. a GridQubit gets mapped to `Q{row}_{col}` where row and col are the corresponding attribute values for the GridQubit. Something similar is done for NamedQubit and LineQubit objects. This can be overridden by passing in a dictionary for the `qubit_conversion` kwarg.

```{code-cell} ipython3
converted_cirq_circuit_custom_qubit_map = Circuit.from_cirq(cirq_circuit_example, qubit_conversion={qubit_00: 'Qalice', qubit_01: 'Qbob'})
print(converted_cirq_circuit_custom_qubit_map)
```

2. By default cirq included idle gates explicitly on all qubits in a layer without a specified operation applied. In pygsti we typically treat these as implied, and so the default behavior is to strip these extra idles. This can be turned off by setting `remove_implied_idles` to `False`.

```{code-cell} ipython3
converted_cirq_circuit_implied_idles = Circuit.from_cirq(cirq_circuit_example, remove_implied_idles=True)
print(converted_cirq_circuit_implied_idles)
```

3. Layers consisting entirely of idle gates are by default converted to the default pyGSTi global idle convention or Label(()), or to a user specified replacement. This is controlled by the `global_idle_replacement_label` kwarg. The default value is the string 'auto', which will utilize the aforementioned default convention. Users can instead pass in either a string, which is converted to a corresponding Label object, or a circuit Label object directly. Finally, by passing in `None` the global idle replacement is not performed, and the full verbatim translation of that cirq layer is produced.

```{code-cell} ipython3
#auto is the default value, explicitly including here for comparison to alternative options.
converted_cirq_circuit_global_idle = Circuit.from_cirq(cirq_circuit_example, global_idle_replacement_label='auto')
print(converted_cirq_circuit_global_idle)
```

```{code-cell} ipython3
converted_cirq_circuit_global_idle_1 = Circuit.from_cirq(cirq_circuit_example, global_idle_replacement_label='Gbanana')
print(converted_cirq_circuit_global_idle_1)
```

```{code-cell} ipython3
from pygsti.baseobjs import Label
converted_cirq_circuit_global_idle_2 = Circuit.from_cirq(cirq_circuit_example, global_idle_replacement_label=Label('Gbanana', ('Q0_0','Q0_1')))
print(converted_cirq_circuit_global_idle_2)
```

```{code-cell} ipython3
converted_cirq_circuit_global_idle_3 = Circuit.from_cirq(cirq_circuit_example, global_idle_replacement_label= None)
print(converted_cirq_circuit_global_idle_3)
```

4. There is built-in support for converting _most_ Cirq gates into their corresponding built-in pyGSTi gate names (see `cirq_gatenames_standard_conversions` in `pygsti.tools.internalgates` for more on this). There is also a fallback behavior where if not found in the default map, the converter will search among the built-in gate unitaries for one that matches (up to a global phase). If this doesn't work for a particular gate of user interest, of you simply want to override the default mapping as needed, this can be done by passing in a custom dictionary for the `cirq_gate_conversion` kwarg.

```{code-cell} ipython3
custom_gate_map = pygsti.tools.internalgates.cirq_gatenames_standard_conversions()
custom_gate_map[cirq.H] = 'Gdefinitelynoth'
converted_cirq_circuit_custom_gate_map = Circuit.from_cirq(cirq_circuit_example, cirq_gate_conversion=custom_gate_map)
print(converted_cirq_circuit_custom_gate_map)
```

## 4. Run the circuits

+++

Add measurements to the circuits.

```{code-cell} ipython3
for circuit in cirq_circuits:
    circuit.append(cirq.measure(q0, key='result'))
```

Simulate the circuits (or run them on a real quantum computer!)

```{code-cell} ipython3
simulator = cirq.Simulator()
results = [simulator.run(circuit, repetitions=1000) for circuit in tqdm.tqdm(cirq_circuits)]
```

Load everything the results into a pyGSTi dataset.

```{code-cell} ipython3
dataset = pygsti.data.dataset.DataSet()
for pygsti_circuit, trial_result in zip(pygsti_circuits, results):
    dataset.add_cirq_trial_result(pygsti_circuit, trial_result, key='result')
```

Perform GST.

```{code-cell} ipython3
gst_results = pygsti.run_stdpractice_gst(dataset, target_model, preps, effects, germs, max_lengths, modes=["full TP","Target"], verbosity=1)
```

See what if finds.

```{code-cell} ipython3
mdl_estimate = gst_results.estimates['full TP'].models['stdgaugeopt']
print("2DeltaLogL(estimate, data): ", pygsti.tools.two_delta_logl(mdl_estimate, dataset))
print("2DeltaLogL(ideal, data): ", pygsti.tools.two_delta_logl(target_model, dataset))
```

```{code-cell} ipython3

```

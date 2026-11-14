---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.17.3
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Detector Error Models from Error Generators
This tutorial demonstrates constructing a stim detector error model (DEM) from pyGSTi error generators using `pygsti.extras.dem_construction`, and using it for decoding with `pymatching`. This module is under active development, so the API shown here may change.

This tutorial requires the `stim` and `pymatching` python packages.

```{code-cell} ipython3
import numpy as np
import stim
import pygsti
import pymatching

#Error generator propagation
from pygsti.errorgenpropagation.errorpropagator import ErrorGeneratorPropagator

#DEM construction
from pygsti.extras.dem_construction import pygsti_object_builders as obj
from pygsti.extras.dem_construction import dem_tools as dems
```

## Setting up an error model
The function below constructs a family of H+S error models. Note that SPAM error is currently limited to uniform single-qubit error, and the error on mid-circuit measurements is modelled by (1) the `Mdefault` error, then (2) the perfect measurement with reset, then (3) the `rho0` error.

```{code-cell} ipython3
def make_sh_sweep_error_param_dict(p=1, h_scale=0, spam_error=0):
    error_rates_dict = {}
    error_rates_dict['Gh'] = {('H','Z'):np.sqrt(0.0001*p*h_scale),('H','X'):np.sqrt(0.0001*p*h_scale),
                              ('S','Z'):0.0001*p+0.0001*p*(1-h_scale),('S','Y'):0.0001*p,('S','X'):0.0001*p*(1-h_scale)}
    error_rates_dict['Gcnot'] = {('H','IX'): np.sqrt(0.01*p*h_scale),('H', 'ZI'):np.sqrt(0.01*p*h_scale), ('H', 'ZX'):np.sqrt(0.01*p*h_scale),
                                 ('S','IX'): 0.01*p/15+0.01*p*(1-h_scale),('S', 'IY'):0.01*p/15,('S', 'IZ'):0.01*p/15,
                                 ('S', 'XX'):0.01*p/15,('S', 'XY'):0.01*p/15,('S', 'XZ'):0.01*p/15,
                                 ('S', 'YX'):0.01*p/15, ('S', 'YY'):0.01*p/15, ('S', 'YZ'):0.01*p/15,
                                 ('S', 'ZX'):0.01*p/15+0.01*p*(1-h_scale), ('S', 'ZY'):0.01*p/15, ('S', 'ZZ'):0.01*p/15,
                                 ('S','XI'): 0.01*p/15,('S', 'YI'):0.01*p/15,('S', 'ZI'):0.01*p/15+0.01*p*(1-h_scale)}
    error_rates_dict['Mdefault'] = {('S', 'X'): spam_error}
    error_rates_dict['rho0'] = {('S', 'X'): 0}
    return error_rates_dict
```

We also need a helper function for unrolling `REPEAT` blocks in stim circuits.

```{code-cell} ipython3
def unroll_repeat_blocks(circuit, add_2q_idles=False, add_measurement_idles=False):
    new_circuit = stim.Circuit()
    for operation in circuit:
        if operation.name == "REPEAT":
            for _ in range(operation.repeat_count):
                for op in operation.body_copy():
                    if op.name == "REPEAT":
                        break  # Stop if we hit another repeat block
                    new_circuit.append_operation(op.name, op.targets_copy(), op.gate_args_copy())
                    if (op.name=='MR' or op.name=='R') and add_measurement_idles:
                        unmeasured_qubits = set(range(circuit.num_qubits))-set(t.qubit_value for t in op.targets_copy())
                        new_circuit.append_operation('I', unmeasured_qubits, op.gate_args_copy())
                    elif (op.name=='CX' or op.name=='CZ') and add_2q_idles:
                        unused_qubits = set(range(circuit.num_qubits))-set(t.qubit_value for t in op.targets_copy())
                        new_circuit.append_operation('I', unused_qubits, op.gate_args_copy())
        else:
            new_circuit.append_operation(operation.name, operation.targets_copy())
            if (operation.name=='MR' or operation.name=='R') and add_measurement_idles:
                unmeasured_qubits = set(range(circuit.num_qubits))-set(t.qubit_value for t in operation.targets_copy())
                new_circuit.append_operation('I', unmeasured_qubits, operation.gate_args_copy())
            elif (operation.name=='CX' or operation.name=='CZ') and add_2q_idles:
                unused_qubits = set(range(circuit.num_qubits))-set(t.qubit_value for t in operation.targets_copy())
                new_circuit.append_operation('I', unused_qubits, operation.gate_args_copy())
    return new_circuit
```

## Building the circuit and model
For demo purposes we construct a DEM for a single model and code distance, using a stim surface code memory circuit with detectors specified in it. You can replace this with another stim circuit. Note that there are known bugs for circuits where `OBSERVABLE_INCLUDE(n)` for a fixed `n` occurs at multiple points in the circuit.

```{code-cell} ipython3
distance = 5
p = 0.5
hscale = 0.5
rounds = distance

error_rates_dict = make_sh_sweep_error_param_dict(p=p, h_scale=hscale, spam_error=0)

stimc = stim.Circuit.generated("surface_code:rotated_memory_z", distance=distance, rounds=rounds)
stimc = unroll_repeat_blocks(stimc)
```

Next we convert the stim circuit to pyGSTi objects and build the noise model. Make sure `include_observables=True` when doing logical error rate estimation.

```{code-cell} ipython3
pcircuit, qubit_mapping, measurements, detectors = obj.stim_to_pygsti_circuit(
    stimc, range(stimc.num_qubits), include_idles=False, include_observables=True)
pspec = obj.create_processor_spec(pcircuit, pcircuit.line_labels, gates=['Gh','Gcnot'])
n_qubits = len(pcircuit.line_labels)
model = obj.build_model(error_rates_dict, pspec, oneQ_gate_names=['Gh'], twoQ_gate_names=['Gcnot'])
```

## Constructing the DEM step by step
First we represent each detector as a Pauli operator, given by the product of the Paulis of the measurements it references.

```{code-cell} ipython3
dets_as_pauli_strings = [dems.get_detector_as_parity(det, measurements, n_qubits) for det in detectors]
```

Next we set up a `stim.TableauSimulator` for the ideal circuit, which is needed to compute the contributions of error generators to DEM events.

```{code-cell} ipython3
pcircuit_serial = pcircuit.serialize()
stim_c_no_extras = obj.pygsti_c_to_stim(pcircuit_serial, include_measurement=False)
tableau = stim.Tableau.from_circuit(stim_c_no_extras)
sim = stim.TableauSimulator()
sim.set_inverse_tableau(tableau**-1)
```

We then propagate the error generators to the end of the circuit and combine them into a single end-of-circuit error generator using the first-order BCH approximation.

```{code-cell} ipython3
egp = ErrorGeneratorPropagator(model)
eoc_eeg = egp.propagate_errorgens_bch(pcircuit_serial, bch_order=1, include_spam=True)
```

Finally we construct the DEM. `generate_dem_higher_order` returns the DEM as a dictionary mapping detector-flip bitstrings to probabilities, which `format_dem_stim` converts into a stim DEM string.

```{code-cell} ipython3
total_dem = dems.generate_dem_higher_order(dets_as_pauli_strings, eoc_eeg, sim, zassenhaus_order=1)
stim_dem_str = dems.format_dem_stim(total_dem, n_logical=stimc.num_observables)
stim_dem = stim.DetectorErrorModel(stim_dem_str)
```

## Decoding with pymatching
With the DEM in hand we can sample syndromes from it and decode them with `pymatching` to estimate the logical error rate (LER).

```{code-cell} ipython3
n_shots = 100000
dec = pymatching.Matching.from_detector_error_model(stim_dem)

sampler = stim_dem.compile_sampler()
syndrome, log_flips, _ = sampler.sample(shots=n_shots)

predictions = dec.decode_batch(syndrome)
log_errors = (predictions != log_flips)

ler = np.mean(np.any(log_errors, axis=1))
print(f'LER of {ler} for distance={distance}, p={p}')
```

## Building the DEM in one call
`build_dem_end2end` combines the steps above into a single call: detector-to-Pauli conversion, simulator setup, error generator propagation, and DEM construction. It takes the un-serialized circuit returned by `stim_to_pygsti_circuit`, together with its detectors and measurements.

```{code-cell} ipython3
stim_dem_e2e = dems.build_dem_end2end(pcircuit, model, detectors, measurements,
                                      n_logical=stimc.num_observables, zassenhaus_order=1)
```

The result can be decoded exactly as before.

```{code-cell} ipython3
dec_e2e = pymatching.Matching.from_detector_error_model(stim_dem_e2e)
syndrome, log_flips, _ = stim_dem_e2e.compile_sampler().sample(shots=n_shots)
ler_e2e = np.mean(np.any(dec_e2e.decode_batch(syndrome) != log_flips, axis=1))
print(f'End-to-end LER of {ler_e2e} for distance={distance}, p={p}')
```

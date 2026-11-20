---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.19.6
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Detector Error Models from Error Generators
This tutorial demonstrates constructing a Stim detector error model (DEM) from pyGSTi error generators using `pygsti.extras.dem_construction`, and using it for decoding with `pymatching`. This module is under active development, so the API shown here may change.

This tutorial requires the `stim` and `pymatching` python packages.

```{code-cell} ipython3
import numpy as np
import stim
import pygsti
import pymatching

#DEM construction
from pygsti.extras.dem_construction import pygsti_object_builders as obj
from pygsti.extras.dem_construction import dem_tools as dems
```

## Setting up an error model
The function below constructs a family of H+S error models for the gates. Preparation and measurement error are not part of it: they are the `prep_err` and `meas_err` arguments of `build_model` below, a uniform single-qubit S(X) error that applies to every qubit and to each mid-circuit reset and measurement.

```{code-cell} ipython3
def make_sh_sweep_error_param_dict(p=1, h_scale=0):
    error_rates_dict = {}
    error_rates_dict['Gh'] = {('H','Z'):np.sqrt(0.0001*p*h_scale),('H','X'):np.sqrt(0.0001*p*h_scale),
                              ('S','Z'):0.0001*p+0.0001*p*(1-h_scale),('S','Y'):0.0001*p,('S','X'):0.0001*p*(1-h_scale)}
    error_rates_dict['Gcnot'] = {('H','IX'): np.sqrt(0.01*p*h_scale),('H', 'ZI'):np.sqrt(0.01*p*h_scale), ('H', 'ZX'):np.sqrt(0.01*p*h_scale),
                                 ('S','IX'): 0.01*p/15+0.01*p*(1-h_scale),('S', 'IY'):0.01*p/15,('S', 'IZ'):0.01*p/15,
                                 ('S', 'XX'):0.01*p/15,('S', 'XY'):0.01*p/15,('S', 'XZ'):0.01*p/15,
                                 ('S', 'YX'):0.01*p/15, ('S', 'YY'):0.01*p/15, ('S', 'YZ'):0.01*p/15,
                                 ('S', 'ZX'):0.01*p/15+0.01*p*(1-h_scale), ('S', 'ZY'):0.01*p/15, ('S', 'ZZ'):0.01*p/15,
                                 ('S','XI'): 0.01*p/15,('S', 'YI'):0.01*p/15,('S', 'ZI'):0.01*p/15+0.01*p*(1-h_scale)}
    return error_rates_dict
```

## Building the circuit and model
For demo purposes we construct a DEM for a single model and code distance, using a Stim surface code memory circuit with detectors specified in it. You can replace this with another Stim circuit, as long as it has no noise instructions, only Z-basis resets and measurements, and no feedback; anything else raises a `ValueError`. `REPEAT` blocks are handled by the converter.

```{code-cell} ipython3
distance = 5
p = 0.5
hscale = 0.5
rounds = distance
meas_error = 0.005

error_rates_dict = make_sh_sweep_error_param_dict(p=p, h_scale=hscale)

stimc = stim.Circuit.generated("surface_code:rotated_memory_z", distance=distance, rounds=rounds)
```

Next we convert the Stim circuit to pyGSTi objects and build the noise model. The converter returns the circuit with every reset starting a fresh virtual qubit, all measurements deferred to the end and one layer per time step of the circuit (the instructions between two `TICK`s), the detectors and observables as signed Paulis on those virtual qubits, and `virtual_to_physical`, which gives the Stim qubit of each virtual qubit. Passing `virtual_to_physical` to `create_processor_spec` puts the processor spec, and so the model, on the Stim qubits. Here preparation is perfect and each measurement has an S(X) error of rate `meas_error`, a flip probability of about `meas_error`.

```{code-cell} ipython3
pcircuit, detectors, observables, virtual_to_physical = obj.stim_to_pygsti_circuit(stimc)
pspec = obj.create_processor_spec(pcircuit, list(range(stimc.num_qubits)), gates=['Gh','Gcnot'],
                                  virtual_to_physical=virtual_to_physical)
model = obj.build_model(error_rates_dict, pspec, oneQ_gate_names=['Gh'], twoQ_gate_names=['Gcnot'],
                        meas_err=meas_error, prep_err=0)
```

Alternatively the model can be built on the converted circuit's own qubits, by calling `create_processor_spec` on `pcircuit` without `virtual_to_physical` and `end_of_circuit_errorgen` below without it. That is the place to start when the noise differs between rounds, since each round's qubits are then separate qubits of the model.

## Constructing the DEM step by step
First we propagate the error generators to the end of the circuit and combine them into a single end-of-circuit error generator using the first-order BCH approximation. This includes the preparation and measurement error. The function also returns the tableau of the whole ideal circuit.

```{code-cell} ipython3
eoc_eeg, tableau = dems.end_of_circuit_errorgen(pcircuit, model, virtual_to_physical)
```

Next we set up a `stim.TableauSimulator` for the ideal circuit, which is needed to compute the contributions of error generators to DEM events.

```{code-cell} ipython3
sim = stim.TableauSimulator()
sim.do_tableau(tableau, list(range(len(virtual_to_physical))))
```

Finally we construct the DEM. The detectors and observables are signed Paulis, and `generate_dem_higher_order` accepts either ideal sign. It returns the DEM as a dictionary mapping flip bitstrings to probabilities, which `format_dem_stim` converts into a Stim DEM, given the number of detectors and of observables.

```{code-cell} ipython3
paulis = list(detectors) + list(observables)
total_dem = dems.generate_dem_higher_order(paulis, eoc_eeg, sim, zassenhaus_order=1)
stim_dem = dems.format_dem_stim(total_dem, len(detectors), len(observables))
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
`build_dem_end2end` combines the steps above into a single call: conversion, error generator propagation, simulator setup, and DEM construction. It takes the Stim circuit and the model, and returns a `stim.DetectorErrorModel`.

```{code-cell} ipython3
stim_dem_e2e = dems.build_dem_end2end(stimc, model, zassenhaus_order=1)
```

Both routes give the same DEM, which we check here.

```{code-cell} ipython3
assert stim_dem.approx_equals(stim_dem_e2e, atol=1e-9)
```

The result can be decoded exactly as before.

```{code-cell} ipython3
dec_e2e = pymatching.Matching.from_detector_error_model(stim_dem_e2e)
syndrome, log_flips, _ = stim_dem_e2e.compile_sampler().sample(shots=n_shots)
ler_e2e = np.mean(np.any(dec_e2e.decode_batch(syndrome) != log_flips, axis=1))
print(f'End-to-end LER of {ler_e2e} for distance={distance}, p={p}')
```

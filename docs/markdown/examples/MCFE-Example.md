---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.17.3
kernelspec:
  display_name: qiskit2_venv
  language: python
  name: python3
---

# Running MCFE on target circuits

```{code-cell} ipython3
import pygsti
from collections import defaultdict
import numpy as np
import time
```

```{code-cell} ipython3
# Create pyGSTi circs
unmapped_circs = [pygsti.circuits.Circuit([["Gxpi2", "Q0"], ["Gypi2", "Q1"]]),pygsti.circuits.Circuit([["Gypi2", "Q0"], ["Gxpi2", "Q1"]])]
```

## Map circuits to device connectivity and U3-CX gate set

+++

This step will be different depending on what architecture you are using. For this example, we are using an IBM device. You need to end up with pyGSTi circuits in a U3-CX gate set so that circuit mirroring can be performed.

```{code-cell} ipython3
mapped_circs = defaultdict(list)

import qiskit
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager as _pass_manager
from qiskit_ibm_runtime.fake_provider import FakeSherbrooke, FakeAthensV2

fake_backend = FakeAthensV2()

pm = _pass_manager(coupling_map=fake_backend.coupling_map, basis_gates=['u3', 'cx'], optimization_level=0)


for i, circ in enumerate(unmapped_circs):
    # Convert from pyGSTi to Qiskit
    # Comment these lines out and do qiskit_circ = circ if passing in Qiskit
    pygsti_openqasm_circ = circ.convert_to_openqasm(block_between_layers=True, include_delay_on_idle=False)
    # print(pygsti_openqasm_circ)
    qiskit_circ = qiskit.QuantumCircuit.from_qasm_str(pygsti_openqasm_circ)

    # print(qiskit_circ.draw())

    mapped_qiskit_circ = pm.run(qiskit_circ)

    # print(mapped_qiskit_circ.draw())
    pygsti_circ, _ = pygsti.circuits.Circuit.from_qiskit(mapped_qiskit_circ)
    # print(pygsti_circ)

    mapped_circ = pygsti_circ

    metadata = {'width': len(mapped_circ.line_labels), 'depth': mapped_circ.depth, 'dropped_gates': 0, 'id': i}
    mapped_circs[mapped_circ] += [metadata]


unmirrored_design = pygsti.protocols.FreeformDesign(mapped_circs)
```

## Mirror circuit generation

+++

We use Pauli random compiling (`pauli_rc`) here. Central Pauli (`central_pauli`) is also an option.

```{code-cell} ipython3
# Highly recommended to seed all RNG
mcfe_rand_state = np.random.RandomState(20240718)

start = time.time()
mirror_design = pygsti.protocols.mirror_edesign.make_mirror_edesign(
    unmirrored_design,
    account_for_routing=False,
    num_mcs_per_circ=100,
    num_ref_per_qubit_subset=100,
    mirroring_strategy='pauli_rc',
    rand_state=mcfe_rand_state)
print(f'Mirroring time:', time.time() - start)
```

We have created the MCFE experiment design.

+++

## Run the Edesign

+++

This example will run the edesign on a fake IBM backend, but this is not strictly required. This step needs to generate a `ProtocolData(edesign=mirror_edesign, dataset=circuit_counts_data)` where `mirror_edesign` is the variable defined earlier and `circuit_counts_data` is a `DataSet` that contains the outcomes for each circuit.

```{code-cell} ipython3
from pygsti.extras.devices import ExperimentalDevice
from pygsti.extras import devices, ibmq

device = ExperimentalDevice.from_qiskit_backend(fake_backend)
pspec = device.create_processor_spec(['Gc{}'.format(i) for i in range(24)] + ['Gcnot'])

start = time.time()
exp = ibmq.IBMQExperiment(mirror_design, pspec, circuits_per_batch=300, num_shots=1024, seed=20240718, checkpoint_override=True)
print(time.time() - start)
```

```{code-cell} ipython3
from qiskit_aer import AerSimulator

sim_backend = AerSimulator.from_backend(fake_backend)

qiskit_convert_kwargs={}

start = time.time()
exp.transpile(sim_backend, direct_to_qiskit=True, qiskit_convert_kwargs=qiskit_convert_kwargs)
end = time.time()
print(f'Total transpilation time: {end - start}')
```

```{code-cell} ipython3
exp.submit(sim_backend)
```

```{code-cell} ipython3
start = time.time()
exp.batch_results = []
exp.retrieve_results()
end = time.time()
print(end - start)
```

```{code-cell} ipython3
data = exp.data
```

## Compute process fidelity for each circuit

```{code-cell} ipython3
from pygsti.protocols.vbdataframe import VBDataFrame

df = VBDataFrame.from_mirror_experiment(unmirrored_design, data)
```

If you used Central Pauli instead, you can swap `'RC Process Fidelity'` for `'CP Process Fidelity'` in the cell below.

```{code-cell} ipython3
process_fidelities = df.dataframe['RC Process Fidelity']
```

```{code-cell} ipython3
process_fidelities.tolist()
```

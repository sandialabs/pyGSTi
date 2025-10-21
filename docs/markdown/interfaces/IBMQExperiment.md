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

# Running experiments on IBM Q Processors
This tutorial will demonstrate how to run an experiment on IBM Q Processors. To do so you will need [QisKit](https://qiskit.org/) installed and an [IBM Q account](https://quantum-computing.ibm.com/).

```{warning}
There have been major changes to `IBMQExperiment` as of pygsti 0.9.13. This is due to Qiskit 1.0 and subsequent deprecations of V1 backends and `qiskit-ibm-provider`. The `IBMQExperiment` class only supports V2 backends and is based on `qiskit-ibm-runtime`.
```

For details on how to migrate from `qiskit<1` or `qiskit-ibm-provider`, see [this blog post](https://www.ibm.com/quantum/blog/transition-to-1), [this Qiskit 1.0 migration guide](https://docs.quantum.ibm.com/api/migration-guides/qiskit-1.0-features), or [this Qiskit Runtime migration guide](https://docs.quantum.ibm.com/api/migration-guides/qiskit-runtime).</font>

This was last run with QisKit versions:

```{code-cell} ipython3
#qiskit.__version__ = '1.1.1'
#qiskit_ibm_runtime.__version__ = '0.25.0'
```

```{code-cell} ipython3
import pygsti
from pygsti.extras.devices import ExperimentalDevice
from pygsti.extras import ibmq
from pygsti.processors import CliffordCompilationRules as CCR
```

```{code-cell} ipython3
:tags: [nbval-skip]

from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ibm_runtime.fake_provider import FakeSherbrooke
```

## Load your IBM Q access
First, load you IBM Q account, get your `provider` and select a device. To do this, follow IBM Q's instructions.

```{code-cell} ipython3
# If your first time, you may need to initialize your account with your IBMQ API token

# You can also specify instances (i.e. "ibm-q/open/main" is the default instance)
# You can also save/load named accounts for different instances, etc. See save_account docs for more information.

#QiskitRuntimeService.save_account(channel="ibm_quantum", token="<IQP_TOKEN>", overwrite=True, set_as_default=True)
```

```{code-cell} ipython3
:tags: [nbval-skip]

# Once credentials are saved, the service can be loaded each time:
service = QiskitRuntimeService(channel="ibm_quantum")
```

```{code-cell} ipython3
:tags: [nbval-skip]

# You can list all the available backends to ensure your instance is running properly
service.backends()
```

```{code-cell} ipython3
:tags: [nbval-skip]

# Can use a physical device...
#backend = service.backend('ibm_sherbrooke')

# Can also ask for the least busy physical device
backend = service.least_busy()

# ... or can use a simulated fake backend
sim_backend = FakeSherbrooke()
```

```{code-cell} ipython3
:tags: [nbval-skip]

# Let's see which backend is the least busy!
print(backend)
```

## Make a ProcessorSpec for IBM Q's processor.

Next we create a ProcessorSpec for the device you're going to run on. This ProcessorSpec must also contain the details needed for creating the pyGSTi experiment design that you want to run, which you can tweak by varying the optional arguments to the `devices.create_processor_spec()` function.

In `v0.9.12`, the `pygsti.extras.devices` module has been updated. You can still use the existing files in `pygsti.extras.devices` if you are offline, and thus may still want to add your own device files. However, you can now also simply use the IBMQ backend to create an `ExperimentalDevice` which is compatible with ProcessorSpecs and Models.

```{code-cell} ipython3
:tags: [nbval-skip]

# Using the configuration files in pygsti.extras.devices (legacy and may not be up-to-date)
#device = ExperimentalDevice.from_legacy_device('ibmq_bogota')

# Using the active backend to pull current device specification
device = ExperimentalDevice.from_qiskit_backend(backend)
```

```{code-cell} ipython3
pspec = device.create_processor_spec(['Gc{}'.format(i) for i in range(24)] + ['Gcnot'])
```

## Create an ExperimentDesign
Next we create an `ExperimentDesign` that specifies the circuits you want to run on that device. Here we create a very simple mirror circuit benchmarking experiment. We'll use randomized mirror circuits, constructed using a `MirrorRBDesign`.

First we pick the circuit design parameters:

```{code-cell} ipython3
#circuit design parameters
depths = [0, 2, 4, 16]
circuits_per_shape = 20

# dict setting the circuit widths (# qubits) you want to probe 
# and the qubits you want to use at each width
# You can use device.graph.edges() to make sure these are connected components
def get_N_connected_qubits(device, N, starting_qubits = None):
    if starting_qubits is None:
        starting_qubits = []
    qubits = set(starting_qubits)

    for edge in device.graph.edges():
        # Check if connected, and add if so
        if not len(qubits) or edge[0] in qubits or edge[1] in qubits:
            qubits.update(edge)
        
        # Check if we can break
        if len(qubits) >= N:
            break
    
    return list(qubits)[:N]

max_width = 4
selected_qubits = get_N_connected_qubits(device, max_width)
print(f"Selected qubits {selected_qubits} for device {backend.name}")

qubit_lists = {}
for i in range(max_width):
    qubit_lists[i] = [tuple(selected_qubits[:i+1])]

widths = list(qubit_lists.keys())

print('total circuits: {}'.format(circuits_per_shape*len(widths)*len(depths)))
total_circuits = 0
for w in widths:
    total_circuits += len(qubit_lists[w]) * circuits_per_shape * len(depths)
print('full total circuits: {}'.format(total_circuits) )

# We'll use the `edgegrab` sampler, which requires specifying the expected number
# of two-qubit gates per random layer.
twoQmean = {w:w/8 for w in widths}
if 1 in widths: twoQmean[1] = 0 # No two-qubit gates in one-qubit circuits.
```

```{code-cell} ipython3
# In order to do Mirror RB, we need some Clifford compilations. See the RB-MirrorRB tutorial for more details.
compilations = {'absolute': CCR.create_standard(pspec, 'absolute', ('paulis', '1Qcliffords'), verbosity=0)}
```

```{code-cell} ipython3
edesigns_dict = {}
edesign_index = 1
for w in widths:
    for qubits in qubit_lists[w]:
        sub_edesign = pygsti.protocols.MirrorRBDesign(pspec, depths, circuits_per_shape, qubit_labels=qubits,
                                                      clifford_compilations=compilations, sampler='edgegrab', samplerargs=[twoQmean[w],])
        
        edesigns_dict[str(edesign_index)] = sub_edesign
        edesign_index += 1
        
combined_edesign = pygsti.protocols.CombinedExperimentDesign(edesigns_dict)
```

## Running on IBM Q
We're now ready to run on the IBM Q processor. We do this using an `IBMQExperiment` object.

We can enable checkpointing for `IBMQExperiment` objects by providing a path. This is the default and is recommended! We are also overriding old checkpoints here to ensure we have a clean starting point.

```{code-cell} ipython3
:tags: [nbval-skip]

exp = ibmq.IBMQExperiment(combined_edesign, pspec, circuits_per_batch=75, num_shots=1024, seed=20231201,
                          checkpoint_path='test_ibmq', checkpoint_override=True)
```

First we convert pyGSTi circuits into jobs that can be submitted to IBM Q. **This step includes transpiling of the pyGSTi circuits into OpenQASM** (and then into QisKit objects).

This can now be done in parallel (with progress bars) using the `max_workers` kwarg!

```{code-cell} ipython3
:tags: [nbval-skip]

exp.transpile(backend, num_workers=4)
```

```{code-cell} ipython3
:tags: [nbval-skip]

# We can simulate having been interrupted by removing the last few transpiled batches
del exp.qiskit_isa_circuit_batches[3:]

# And now transpilation should only redo the missing batches
# We don't need to reprovide the options as they are saved by the first transpile call
exp.transpile(backend)
```

If the `IBMQExperiment` object is lost and needs to be reloaded (i.e. notebook restarts), it can be loaded from file now.

```{code-cell} ipython3
:tags: [nbval-skip]

exp2 = ibmq.IBMQExperiment.from_dir('test_ibmq')
```

We're now ready to submit this experiment to IBM Q.Note that we can submit using a different backend than what was used to generate the experiment design. In general, it is not a good idea to mix and match backends for physical devices unless they have the exact same connectivity and qubit labeling; however, it **is** often useful for debugging purposes to use the simulator backend rather than a physical device.

```{code-cell} ipython3
:tags: [nbval-skip]

exp2.submit(backend)
```

You can then monitor the jobs. If get an error message, you can query the error using `exp.qjobs[i].error_message()` for batch `i`.

```{code-cell} ipython3
:tags: [nbval-skip]

exp2.monitor()
```

Again, the `IBMQExperiment` can be loaded from file if checkpointing is being used. The Qiskit RuntimeJobs are not serialized; however, they can be retrieved from the IBMQ service from their job ids. In order to do this, pass `regen_jobs=True` and a `service` to the `from_dir()` call.

```{code-cell} ipython3
:tags: [nbval-skip]

exp3 = ibmq.IBMQExperiment.from_dir('test_ibmq', regen_jobs=True, service=service)
```

```{code-cell} ipython3
:tags: [nbval-skip]

exp3.monitor()
```

You can then grab the results, **Once you see that all the jobs are complete** (`.retrieve_results()` will just hang if the jobs have not yet completed).

```{code-cell} ipython3
:tags: [nbval-skip]

exp3.retrieve_results()
```

This `IBMQExperiment` object now contains the results of your experiment. It contains much of the information about exactly what was submitted to IBM Q, and raw results objects that IBM Q returned.

```{code-cell} ipython3
:tags: [nbval-skip]

display(exp3.qjobs)
display(exp3.batch_results)
```

But, most importantly, it contains the data formatted into a pyGSTi `ProtocolData` object, which is the packaged-up data that pyGSTi analysis proctols use.

```{code-cell} ipython3
:tags: [nbval-skip]

data = exp3.data
```

## Analyzing the results
Because `retrieve_results()` has formatted the data into a `ProctocolData` object, we can just hand this to the analysis protocol(s) that are designed for analyzing this type of data. Here we'll analyze this data using a standard RB curve-fitting analysis.

```{code-cell} ipython3
:tags: [nbval-skip]

rb = pygsti.protocols.RandomizedBenchmarking(datatype='adjusted_success_probabilities', defaultfit='A-fixed')
results = {}
for key in data.keys():
    results[key] = rb.run(data[key])
```

```{code-cell} ipython3
:tags: [nbval-skip]

ws = pygsti.report.Workspace()
ws.init_notebook_mode(autodisplay=True)
```

```{code-cell} ipython3
:tags: [nbval-skip]

for i in data.keys(): 
    print(i)
    ws.RandomizedBenchmarkingPlot(results[i])
```



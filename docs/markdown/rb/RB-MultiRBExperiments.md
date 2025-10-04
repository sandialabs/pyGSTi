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

# Running Multiple Randomized Benchmarking Experiments

+++

This tutorial shows how to easily run multiple RB experiments concurrently. This includes running RB on different subsets of a device, as well as running [simultaneous RB]() experiments. Here we'll demonstrate generating an experiment to run 1, 2, 3 and 4 qubit RB in sequence (i.e., separately), as well as running 1-qubit RB in parallel on all the qubits (i.e., simultaneously).

Note that this functionality is not specific to RB: similar code could be used to combine multiple GST experiments, or GST and RB experiments, etc.

```{code-cell} ipython3
import pygsti
from pygsti.processors import QubitProcessorSpec as QPS
from pygsti.processors import CliffordCompilationRules as CCR
```

Let's define the layout and gates of a 4-qubit device that we want to do this experiment on.

```{code-cell} ipython3
n_qubits = 4
qubit_labels = ['Q'+str(i) for i in range(n_qubits)]
gate_names = ['Gc{}'.format(i) for i in range(24)] + ['Gcnot'] 
pspec = QPS(n_qubits, gate_names, qubit_labels=qubit_labels, geometry='ring')

compilations = {'absolute': CCR.create_standard(pspec, 'absolute', ('paulis', '1Qcliffords'), verbosity=0),            
                'paulieq': CCR.create_standard(pspec, 'paulieq', ('1Qcliffords', 'allcnots'), verbosity=0)}
```

Now let's generate the separate 1-4 qubit RB experiments. We'll run Mirror RB, but this works for all types of RB.

```{code-cell} ipython3
# The qubit sets of each size to benchmark, and sampling details.
qubits = {1: ['Q0',], 2:['Q0','Q1'], 3:['Q0','Q1','Q2'], 4:['Q0','Q1','Q2', 'Q3']}
# The depths for the different number of qubits.
depths = {1: [0, 2, 4, 8, 16, 32, 64, 128, 256, 512],
          2: [0, 2, 4, 8, 16, 32, 64, 128],
          3: [0, 2, 4, 8, 16, 32, 64],
          4: [0, 2, 4, 8, 16, 32]}

# This loops through an generates the experiment design for each case.
designs = {}
for n in [1,2,3,4]:
    
    designs[str(n)+'Q-RB'] = pygsti.protocols.MirrorRBDesign(pspec, depths[n], 10, qubit_labels=qubits[n],
                                                             clifford_compilations=compilations,
                                                              sampler='edgegrab', samplerargs=[0.5],
                                                              add_default_protocol=True)
```

Next, we generate the simultaneous 1-qubit RB experiment. We do this by constructing each of the 1-qubit experiment designs, and then combining them together in a `SimultaneousExperimentDesign`.

```{code-cell} ipython3
oneQdesigns = []
for q in qubit_labels:
    oneQdesigns.append(pygsti.protocols.MirrorRBDesign(pspec, depths[1], 10, qubit_labels=[q,], 
                                                       clifford_compilations=compilations,
                                                sampler='edgegrab', samplerargs=[0.],
                                                add_default_protocol=True))
    
sim1Qdesign = pygsti.protocols.SimultaneousExperimentDesign(oneQdesigns)
```

As we want to run this simultanoeus 1-qubit RB experiment alongside the 1-4 qubit RB experiments, we add it to the `designs` dictionary where we're storing all the experiment designs that are to be run together (but not in parallel).

```{code-cell} ipython3
designs['1Q-SRB'] = sim1Qdesign
```

Then we just combine them together in a `CombinedExperimentDesign`. This is then written to a directory in the same way as with all experiment designs. The dataset template contains all the circuits that need to be run. This will include all the circuits from all the sub-designs, including the simultaneous 1-qubit circuits in the necessary parallel form. After data is added to the template, it is then read-in in the same way as always.

```{code-cell} ipython3
qubit_error_rate = 0.002
def simulate_taking_data(data_template_filename):
    """Simulate taking data and filling the results into a template dataset.txt file"""
    error_rates = {}
    for gn in pspec.gate_names:
        n = pspec.gate_num_qubits(gn)
        gate_error_rate = n * qubit_error_rate
        error_rates[gn] = [gate_error_rate/(4**n - 1)] * (4**n - 1)
    noisemodel = pygsti.models.create_crosstalk_free_model(pspec, stochastic_error_probs=error_rates)
    pygsti.io.fill_in_empty_dataset_with_fake_data(data_template_filename, noisemodel, num_samples=1000, seed=1234)
```

```{code-cell} ipython3
combdesign = pygsti.protocols.CombinedExperimentDesign(designs)

pygsti.io.write_empty_protocol_data('../tutorial_files/test_combrb_dir', combdesign, clobber_ok=True)

# -- fill in the dataset file in tutorial_files/test_rb_dir/data/dataset.txt --
simulate_taking_data('../tutorial_files/test_combrb_dir/data/dataset.txt') # REPLACE with actual data-taking

data = pygsti.io.read_data_from_dir('../tutorial_files/test_combrb_dir')
```

We can then run any protocols that we want to on the imported data. Because we set `add_default_protocol=True` when creating the protocols, the imported data contains the `RB` protocols ready to run (with the appropriate optional arguments set for Mirror RB). We can run all these default protocols by creating a `pygsti.protocols.DefaultRunner()` protocol, and running it on the data.

We can also run more protocols (e.g., a test for instability if the data is time-stamped) just by creating the relevant protocols objects, and passing them this data.

```{code-cell} ipython3
protocol = pygsti.protocols.DefaultRunner()
results = protocol.run(data)
```

```{code-cell} ipython3
results['1Q-SRB'].keys()
```

The `results` behaves like a dictionary, for accessing the individual results.

```{code-cell} ipython3
for i in ['1Q-RB', '2Q-RB', '3Q-RB' , '4Q-RB']:
    r = results[i].for_protocol['RB'].fits['A-fixed'].estimates['r']
    print('The RB error rate on {} qubit is {}'.format(i, r))
    
print()
for i in [('Q0',),('Q1',),('Q2',),('Q3',)]:
    r = results['1Q-SRB'][i].for_protocol['RB'].fits['A-fixed'].estimates['r']
    print('When running simultaneously, the RB error rate on {} qubit is {}'.format(i, r))
```

```{code-cell} ipython3
ws = pygsti.report.Workspace()
ws.init_notebook_mode(autodisplay=True)
for i in ['1Q-RB', '2Q-RB', '3Q-RB' , '4Q-RB']:
    ws.RandomizedBenchmarkingPlot(results[i].for_protocol['RB'])
```

**For more information** and examples on running multiple benchmarking protocols on a processor, check out the [volumetric benchmarking tutorial](VolumetricBenchmarks.ipynb).

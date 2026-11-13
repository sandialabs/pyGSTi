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

# Interleaved RB

Clifford RB gives you an average error rate over the whole Clifford group. Interleaved RB (IRB) answers a narrower question: how good is *one particular* gate? It does that by running Clifford RB twice, once normally and once with the gate of interest interleaved between the random Cliffords, and comparing the two decay rates.

The cost is that you run two experiments instead of one, and the payoff is a per-gate number rather than a per-Clifford average. Read [Clifford RB](CliffordRB) first; everything here builds on it, and IRB uses CRB as a subroutine.

```{warning}
IRB's point estimate comes with rigorous bounds, and on realistic data those bounds are wide enough to swallow the estimate. In the run below the point estimate is about 0.019 with a bound half-width near 0.10, so the rigorous interval contains zero and extends to several times the true error rate. The point estimate is still informative, but do not report it without the bound, and do not treat a small IRB number as a tight measurement of a single gate.
```

## Setting up

This page benchmarks a single qubit, which keeps the run short. See [Clifford RB](CliffordRB) for what the processor specification and compilation rules mean.

```{code-cell} ipython3
import pygsti
from pygsti.processors import QubitProcessorSpec as QPS
from pygsti.processors import CliffordCompilationRules as CCR
```

```{code-cell} ipython3
from pygsti.protocols.rb import InterleavedRBDesign, InterleavedRandomizedBenchmarking
from pygsti.circuits import Circuit
from pygsti.baseobjs import Label
```

The creation of an IRB design largely follows that of CRB, with the addition of the specification of an interleaved circuit. That is, the clifford which we want to estimate the individual error rate for.

```{code-cell} ipython3
n_qubits = 1
qubit_labels = ['Q0']
gate_names = ['Gxpi2', 'Gxmpi2', 'Gypi2', 'Gympi2']
pspec = QPS(n_qubits, gate_names, qubit_labels=qubit_labels)
compilations = {'absolute': CCR.create_standard(pspec, 'absolute', ('paulis', '1Qcliffords'), verbosity=0),            
                'paulieq': CCR.create_standard(pspec, 'paulieq', ('1Qcliffords', 'allcnots'), verbosity=0)}
depths = [0,1,2,4,8,16,32]
k = 50
interleaved_circuit = Circuit([Label('Gxpi2', 'Q0')], line_labels=('Q0',))
```

```{code-cell} ipython3
irb_design = InterleavedRBDesign(pspec, compilations, depths, k, interleaved_circuit, qubit_labels)
```

`InterleavedRBDesign` is structured somewhat differently than `CliffordRBDesign`, instead acting as a container class which constructs and stores a pair of CRB experiment designs (one interleaved with the specified `interleaved_circuit`) with settings as specified by the given arguments. `InterleavedRBDesign` is a subclass of the more general `CombinedExperimentDesign`, and like `CombinedExperimentDesign` its child subdesigns can be accessed by indexing into it like a dictionary, as shown below.

```{code-cell} ipython3
print(irb_design.keys())
print(irb_design['crb'])
```

Here we construct an error model with 1% local depolarization on each qubit after each one-qubit gate, except for Gxpi2 which has a 2% depolarization rate.

```{code-cell} ipython3
def simulate_taking_data_irb(data_template_filename):
    """Simulate taking data and filling the results into a template dataset.txt file"""
    depolarization_strengths={g:0.01 for g in pspec.gate_names if g!= 'Gxpi2'}
    depolarization_strengths['Gxpi2'] = .02
    noisemodel = pygsti.models.create_crosstalk_free_model(pspec, depolarization_strengths=depolarization_strengths)
    noisemodel.sim = 'map'
    pygsti.io.fill_in_empty_dataset_with_fake_data(data_template_filename, noisemodel, num_samples=1000, seed=1234)
```

```{code-cell} ipython3
pygsti.io.write_empty_protocol_data('../../../tutorial_files/test_irb_dir', irb_design, clobber_ok=True)

# -- fill in the dataset file in ../../../tutorial_files/test_irb_dir/data/dataset.txt --
simulate_taking_data_irb('../../../tutorial_files/test_irb_dir/data/dataset.txt') # REPLACE with actual data-taking
data_irb = pygsti.io.read_data_from_dir('../../../tutorial_files/test_irb_dir')
```

```{code-cell} ipython3
protocol_irb = InterleavedRandomizedBenchmarking()
results_irb = protocol_irb.run(data_irb)
```

Now that we have the results we can index into them to get the estimated IRB numbers and bounds. In this context, 'bounds' really refers to the half-width of the bounds as described in equation 5 of the original IRB paper from Magesan et al. https://arxiv.org/pdf/1203.4550.
The object that is returned by `InterleavedRandomizedBenchmarking` is a so-called `ProtocolResultsDir`, and this object stores both the IRB specific estimates as well as the results objects associated with each of the subexperiments used to perform IRB. This makes extracting the values slightly more cumbersome than usual, but ensures that the relevant results remain grouped together at all times. Below we show how to access the IRB numbers and bounds.

```{code-cell} ipython3
results_irb.for_protocol['InterleavedRandomizedBenchmarking'].irb_numbers
```

```{code-cell} ipython3
results_irb.for_protocol['InterleavedRandomizedBenchmarking'].irb_bounds
```

To access the results objects of the standard and interleaved CRB experiments that we performed we can index into `results_irb` like a dictionary. The relevant keys are 'crb' and 'icrb', respectively.

```{code-cell} ipython3
results_irb['crb'].for_protocol['RandomizedBenchmarking']
```

```{code-cell} ipython3
results_irb['icrb'].for_protocol['RandomizedBenchmarking']
```

From which we can access various information about the fits as well as other useful RB related estimates. Below we extract the RB number and the fitted decay parameters for one of the RB fits performed on the CRB subexperiment.

The printed dictionary has four keys, from the fit $P_m = a + b p^m$: `a` is the asymptote, `b` the amplitude, `p` the decay parameter, and `r` the RB number derived from `p`. Two fit variants are always available — `'full'` lets $a$ float, while `'A-fixed'` pins it to the value expected from random guessing — and comparing them is a cheap sanity check, since large disagreement between the two usually means the decay is not well resolved.

```{code-cell} ipython3
print(results_irb['crb'].for_protocol['RandomizedBenchmarking'].fits['full'].estimates)
```

Finally, we'll note that this results object for IRB can be written to and read from disk using the `write` method and the function `pygsti.io.read_results_from_dir`, respectively.

```{code-cell} ipython3
results_irb.write('../../../tutorial_files/test_irb_results')
```

```{code-cell} ipython3
irb_results_from_disk = pygsti.io.read_results_from_dir('../../../tutorial_files/test_irb_results')
print(irb_results_from_disk['crb'].for_protocol['RandomizedBenchmarking'].fits['full'].estimates)
#As expected these values are the same as above when we accessed them in `results_irb`
```

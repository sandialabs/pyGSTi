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

# Data Simulation
"Data simulation" in pyGSTi refers generally to the computation of per-circuit quantities that are independent of any experimental data.  Typically the "data" that is simulated for each circuit is the circuit's sampled outcome probability distribution, resulting in simulated experimental data counts.  In this typical case, the "data" in "data simulation" refers specificially to "experimental data".

This is not, however, necessarily the case.  Other types of per-circuit "data" that may be simulated are properties of a circuit, such as its width, depth, or ideal/expected outcome.  PyGSTi provides an extensible framework for computing arbitrary per-circuit quantities.  These can be based, if desired, on the circuit's outcome probabilities, the final state at the end of a circuit, or the process matrix representation of the circuit action.

A data simulator object generates data, which can be outcome counts or other custom quantities, for every circuit within an `ExperimentDesign`.  The `DataSimulator` class has a `run` method that takes an experiment design and produces a `ProtocolData` object (recall that a `ProtocolData` object comprises an experiment design and corresponding - in this case simulated - data set).  This follows the pattern set by the `Protocol` object, whose `run` method takes a `ProtocolData` object and produces a results object.

In this tutorial, we'll briefly show how standard experimental data simulation is performed, and then move on to the more interesting "free-form" data simulators. 

```{code-cell} ipython3
import pygsti
from pygsti.modelpacks import smq1Q_XYI
from pygsti.circuits import Circuit as C
```

## Simulating experimental data
Simulating experimental data using a `Model` object can be done via the `simulate_data` method.  This method computes the outcome distribution for each supplied circuit and samples from this distribution to get outcome counts.  The result is a `DataSet` object.

```{code-cell} ipython3
mdl = smq1Q_XYI.target_model().depolarize(op_noise=0.1)
circuits = pygsti.circuits.to_circuits(['{}@(0)', 'Gxpi2:0', 'Gypi2:0', 'Gxpi2:0^2'])
ds = pygsti.data.simulate_data(mdl, circuits, num_samples=1000, seed=2021)
print(ds)
```

This task can also be accomplished using a `DataCountsSimulator` object.  This requires packaging our circuit list as an experiment design, and results in a `ProtocolData` object that packages together this experiment design and the generated `DataSet`.

```{code-cell} ipython3
edesign = pygsti.protocols.ExperimentDesign(circuits)
dsim = pygsti.protocols.DataCountsSimulator(mdl, num_samples=1000, seed=2021)
data = dsim.run(edesign)
print(data.dataset)
```

## Free form data simulators
Data sets in pyGSTi usually hold circuit outcome counts.  Indeed the so-named `DataSet` object does just this.  Sometimes, however, we want to compute other quantities based on each circuit.  The `FreeformDataset` generalizes pyGSTi's standard `DataSet` and stores arbitrary quantities for a set of circuits.  The `ModelFreeformSimulator` class generalizes the `DataCountsSimulator` used above and provides a base class for customized per-circuit computations based on a model's simulation of a circuit.  When it is run, it produces a `ProtocolData` object containing a `FreeformDataSet` that associates the computed (simulated) quantities with each circuit.

The customized circuit computations of a `ModelFreeformSimulator` are often used alongside the ability of a `FreeformDesign` to associate arbitrary meta-data with each circuit as an experiment design.  Below we demonstrate how a typical workflow might look.  We begin by creating a free-form experiment design that associates some meta-data (in this case the circuit depty) with each of several circuits.

```{code-cell} ipython3
circuits = [C("Gxpi2:0"), C("Gypi2:0"), C("Gxpi2:0^2"), C("Gypi2:0^2")]
circuit_info_dict = {c: {'depth': c.depth} for c in circuits}
ff_edesign = pygsti.protocols.FreeformDesign(circuit_info_dict)
```

Next, we create a custom data simulator to compute the quantities we want.  To do this, we create a class derived from `ModelFreeformSimulator` and implement several simple methods.  `ModelFreeformSimulator.__init__` takes a dictionary of models that are named by the keys of the dictionary.  Our simulator will compare the outputs from a noisy model and a perfect model, which are named `"base"` and `"target"`.

```{code-cell} ipython3
class MyDataSimulator(pygsti.protocols.ModelFreeformSimulator):
    def __init__(self, model, target_model):
        super().__init__({'base': model, 'target': target_model})
    
    def compute_freeform_data(self, circuit): # pass in aux data TODO
        ret = {}  # we return a dict of all the things we compute for this circuit
        
        #Get the raw ingredients we need: probabilities, final_states and/or circuit process matrices
        #You'd usually call just *one* of these - the one giving the hardest ingredient you need -
        # and set flags to True to get the easier ingredients so there's only one forward sim per circuit.
        probs = self.compute_probabilities(circuit)
        final_states = self.compute_final_states(circuit, include_probabilities=False)
        process_matrices = self.compute_process_matrices(circuit, include_final_state=False, include_probabilities=False)
        
        #Compute the things we want using the ingredients
        A = process_matrices['base']
        B = process_matrices['target']
        ret['process fidelity'] = pygsti.tools.entanglement_fidelity(A, B, 'pp')
        
        state = pygsti.tools.ppvec_to_stdmx(final_states['base'])  
        target_state = pygsti.tools.ppvec_to_stdmx(final_states['target'])
        ret['final state fidelity'] = pygsti.tools.fidelity(state, target_state)
        
        p = probs['base']
        q = probs['target']
        ret['TVD'] = 0.5 * sum([abs(p[i] - q[i]) for i in p])
        
        #Return a dict of all the computed quantities
        return ret
```

Within the workhorse function, `compute_freeform_data`, we compute and return a dictionary of values for the given `circuit`.  To aid in this computation, the base class provides convenient ways to extract:
1. the circuit's outcome probabilities, via `compute_probabilities`
2. the final state (right before measurement), via `compute_final_states`
3. the overall action of the circuit as a process matrix (excluding the state preparation and measurement), via `compute_process_matrices`.

Only *one* of these `compute_*` routines should be needed (in the example above we call all three just to demonstrate them).  When multiple value types are desired you should use the `include_*` arguments of the method performing the most difficult computation.  For example, if you want both final states and outcome probabilities, you should call `compute_final_states` with `include_probabilities=True`.  Also note that the `compute_process_matrices` function requires that the models use a `MatrixForwardSimulator` as their circuit simulator, and that this mehod scales poorly with the number of qubits as it stores and multiplies the process matrices of each model's operations.  The `compute_*` functions return dictionaries whose keys are the model names as specified in `__init__`.

In this example, we compute the total variation distance between the "base" and "target" ouctome distributions, as well as the state fidelity between final states and process fidelity between the circuit actions as quantum processes.

Now, we just need to create a data simulator object and run it on our experiment design.

```{code-cell} ipython3
mysim = MyDataSimulator(mdl, smq1Q_XYI.target_model())  #Create the data-simulator object (this just constructs it - it doesn't run yet)
ff_data = mysim.run(ff_edesign)
```

The resulting `ff_data` is a `ProtocolData` object containing a `FreeformDataSet` as its `.dataset`.  This data set contains a dictionary of computed values for each circuit within it:

```{code-cell} ipython3
ff_data.dataset[C('Gxpi2:0')]  # the "freeform dataset" contains an arbitrary dictionary of value for each circuit 
```

A nice feature of a `FreeformDataSet` is that it's data can be converted to a Pandas *dataframe*.  (If the below code doesn't work you might need to install the `pandas` Python package.)

```{code-cell} ipython3
ff_data.dataset.to_dataframe(pivot_value="Value")
```

Furthermore, a `ProtocolData` containing a free-form dataset can also be converted to a dataframe.  When the `ProtocolData` object's experiment design is a `FreeformDesign`, the meta-data of contained thereis is also included in the dataframe.  Thus, in the cell below, the resulting dataframe contains a "depth" column. 

```{code-cell} ipython3
ff_data.to_dataframe(pivot_value="Value")
```

### Data-frame centered workflow
In addition to being able to render `FreeformDataSet` and parent `ProtocolData` objects as dataframes, `FreeformDesign` objects can be loaded from and converted to dataframes.  Furthermore, the `ModelFreeformSimulator` has an `apply` method that effectively runs a data simulator on an experiment-design's dataframe.   This allows an end-to-end use of Pandas dataframes.  This can be particularly useful in large analyses as the dataframes just need to hold the string representations of circuits which are much faster to load and save than pyGSTi `Circuit` objects which require a parsing step.  (The advantage of circuit objects is that they can be manipulated, but that's ofen unnecessary.)

Below we demonstrate this dataframe-centric workflow.  We begin by converting our experiment design to a dataframe, but remark that *any* code generating a dataframe with a `Circuits` (capital "C"!) column would do equally well here.

```{code-cell} ipython3
edesign_df = ff_edesign.to_dataframe(pivot_value='Value')  # convert our experiment design to a dataframe
edesign_df
```

Using the same data simulator defined above, we can simply "apply" this simulator to the above dataframe.  This produces the same dataframe we created above (up to a potential column reordering).

```{code-cell} ipython3
data_df = mysim.apply(edesign_df)
data_df
```

```{code-cell} ipython3

```

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

# Simulating data

Everything on this page starts with a `Model` and ends with numbers.  Which numbers depends on what you ask for: the outcome probabilities of a single circuit, sampled outcome counts for every circuit in an experiment design, or arbitrary per-circuit quantities that you define yourself.  PyGSTi calls the underlying computation *forward simulation*, and all three cases run through it.

"Data simulation" usually means the second of those: compute each circuit's outcome distribution, sample from it, and get simulated experimental counts.  But the "data" attached to a circuit need not be counts.  It can be the circuit's width, its depth, its ideal outcome, its final state, or a fidelity computed against some reference model.  PyGSTi provides an extensible framework for computing arbitrary per-circuit quantities, and the last part of this page shows how to plug into it.

We start small, with one circuit at a time, and work up to whole experiment designs.

```{code-cell} ipython3
import pygsti
from pygsti.modelpacks import smq1Q_XYI
from pygsti.circuits import Circuit as C
```

## Outcome probabilities for one circuit

`Model` objects are statistical models that predict the outcome probabilities of events, and for every current model type an "event" is a circuit.  So computing outcome probabilities takes three steps:

1. create a `Model`
2. create a `Circuit`
3. call `model.probabilities(circuit)`

Steps 1 and 2 are covered elsewhere (see the [circuits guide](Circuits) and the [explicit-op model](../models/Models) and [implicit-op model](../models/MultiQubitModels) guides).  This section is about step 3, and about the `Model` options that change how probabilities get computed.  This is the right approach when you have a large number of circuits that are known and fixed beforehand.

Here is a simple example:

```{code-cell} ipython3
mdl_2q = pygsti.models.create_explicit_model_from_expressions((0,1),
            [(),      ('Gxpi2',0),    ('Gypi2',0),    ('Gxpi2',1),    ('Gypi2',1),    ('Gcnot',0,1)],
            ["I(0,1)","X(pi/2,0)", "Y(pi/2,0)", "X(pi/2,1)", "Y(pi/2,1)", "CNOT(0,1)"])
c = pygsti.circuits.Circuit([('Gxpi2',0),('Gcnot',0,1),('Gypi2',1)], line_labels=[0,1])
print(c)
mdl_2q.probabilities(c)  # compute the outcome probabilities of circuit `c`
```

That builds an `ExplicitOpModel` (best for 1-2 qubits) on 2 qubits, with $X(\pi/2)$ and $Y(\pi/2)$ rotations on each qubit and a CNOT between them.  The model can simulate any circuit *layer* (a.k.a. "time-step" or "clock-cycle") containing any *one* of those gates.  That restriction is what "explicit-op" means: the operation for every simulatable circuit layer has to be handed to the `Model` explicitly.  A layer with two `Gxpi2` gates in parallel is not one of them:

```{code-cell} ipython3
c2 = pygsti.circuits.Circuit([ [('Gxpi2',0), ('Gxpi2',1)], ('Gcnot',0,1) ], line_labels=[0,1])
print(c2)
try:
    mdl_2q.probabilities(c2)
except KeyError as e:
    print("KEY ERROR (can't simulate this layer): " + str(e))
```

An "implicit-operation" model builds layer operations from constituent gates on demand, so it handles `c2` without complaint.  The [implicit-op model guide](../models/MultiQubitModels) has the details.

```{code-cell} ipython3
pspec = pygsti.processors.QubitProcessorSpec(2, ('Gxpi2', 'Gypi2', 'Gcnot'), geometry='line')
implicit_mdl = pygsti.models.create_crosstalk_free_model(pspec)
print(c2)
implicit_mdl.probabilities(c2)
```

## Propagating a state layer by layer

The other way to simulate a circuit is to take a state object (a `State` in pyGSTi) and push it through the circuit one layer at a time.  This is useful when there are few circuits, or just one, and the circuit involves substantial classical logic or needs to be probed at intermediate points in time.  It is slower than calling `probabilities`, because it crosses the boundary between pyGSTi's Python and C routines far more often.

The two cells below redo the two circuits above by state propagation.

```{code-cell} ipython3
# circuit `c` above, using `mdl_2q`: [('Gxpi2',0),('Gcnot',0,1),('Gypi2',1)]
rho = mdl_2q['rho0']
rho = mdl_2q[('Gxpi2',0)].acton(rho)
rho = mdl_2q[('Gcnot',0,1)].acton(rho)
rho = mdl_2q[('Gypi2',1)].acton(rho)
probs = mdl_2q['Mdefault'].acton(rho)
print(probs)
```

For implicit models you have to reach into the model's operation blocks by hand, which is clunky.

```{code-cell} ipython3
# circuit `c2` above, using `implicit_mdl`: [ [('Gxpi2',0), ('Gxpi2',1)], ('Gcnot',0,1) ]
from pygsti.baseobjs import Label as L
rho = implicit_mdl.prep_blks['layers'][L('rho0')]
rho = implicit_mdl.operation_blks['layers'][ L('Gxpi2',0) ].acton(rho)
rho = implicit_mdl.operation_blks['layers'][ L('Gxpi2',1) ].acton(rho)
rho = implicit_mdl.operation_blks['layers'][ L('Gcnot',(0,1)) ].acton(rho)
probs = implicit_mdl.povm_blks['layers']['Mdefault'].acton(rho)
print(probs)
```

```{warning}
Simulation by state propagation is a work in progress in pyGSTi, and users should expect that this interface may change (improve!) in the future.
```

## Forward-simulation types

Several forward-simulation methods are available, and a `Model` holds its active one in the `.sim` attribute (an instance of a `ForwardSimulator` subclass).  The default, selected when a construction function is passed `simulator="auto"`, is `"map"`: repeated matrix-vector products against the state representation, with operations treated as abstract *maps*.  The `"matrix"` method multiplies together dense process matrices for the whole circuit; it can win on small (1-2 qubit) Hilbert spaces where caching dense matrices pays off across very large circuit batches, but it is not the automatic choice.  See the [forward simulation types guide](../../advanced/simulation/ForwardSimulators) for the full list, including the term-based and CHP simulators.

Usually you don't need to think about this.  When you do, `.sim` is both readable and assignable:

```{code-cell} ipython3
print("2Q mdl_2q will simulate probabilities using the '%s' forward-simulation method." % mdl_2q.sim)
mdl_2q.probabilities(c)
```

```{code-cell} ipython3
pspec3 = pygsti.processors.QubitProcessorSpec(3, ('Gxpi2', 'Gypi2', 'Gcnot'), geometry='line')
implicit_3q = pygsti.models.create_crosstalk_free_model(pspec3)
print("3Q implicit_3q will simulate probabilities using the '%s' forward-simulation method." % implicit_3q.sim)
implicit_3q.probabilities(c)
```

Switching is a one-line assignment.  We do it on the two-qubit model rather than the three-qubit one deliberately: `"matrix"` composes a dense process matrix for the entire circuit, so it can pay off at one or two qubits and [should not be pointed at a many-qubit model](../../advanced/simulation/ForwardSimulators).

```{code-cell} ipython3
mdl_2q.sim = 'matrix'
print("2Q mdl_2q will now simulate probabilities using the '%s' forward-simulation method." % mdl_2q.sim)
mdl_2q.probabilities(c)
```

## Simulating experimental data

To go from probabilities to counts, use `pygsti.data.simulate_data`.  It computes the outcome distribution for each supplied circuit, samples from it, and returns a `DataSet`.

```{code-cell} ipython3
noisy_mdl = smq1Q_XYI.target_model().depolarize(op_noise=0.1)
circuits = pygsti.circuits.to_circuits(['{}@(0)', 'Gxpi2:0', 'Gypi2:0', 'Gxpi2:0^2'])
ds = pygsti.data.simulate_data(noisy_mdl, circuits, num_samples=1000, seed=2021)
print(ds)
```

The same job can be done by a `DataCountsSimulator`, which fits pyGSTi's protocol objects.  A data simulator generates data for every circuit in an `ExperimentDesign`: its `run` method takes an experiment design and produces a `ProtocolData` object, which packages that design together with the generated `DataSet`.  This mirrors `Protocol.run`, which takes a `ProtocolData` and produces a results object.

```{code-cell} ipython3
edesign = pygsti.protocols.ExperimentDesign(circuits)
dsim = pygsti.protocols.DataCountsSimulator(noisy_mdl, num_samples=1000, seed=2021)
data = dsim.run(edesign)
print(data.dataset)
```

## Free-form data simulators

Data sets in pyGSTi usually hold circuit outcome counts, and the `DataSet` object does exactly that.  Sometimes you want other per-circuit quantities instead.  `FreeformDataSet` generalizes `DataSet` and stores arbitrary values for a set of circuits.  `ModelFreeformSimulator` generalizes `DataCountsSimulator` and gives you a base class for custom per-circuit computations built on a model's simulation of the circuit.  Running one produces a `ProtocolData` whose `.dataset` is a `FreeformDataSet`.

These custom computations pair naturally with `FreeformDesign`, which associates arbitrary metadata with each circuit.  Start by building a free-form experiment design that records each circuit's depth:

```{code-cell} ipython3
circuits = [C("Gxpi2:0"), C("Gypi2:0"), C("Gxpi2:0^2"), C("Gypi2:0^2")]
circuit_info_dict = {c: {'depth': c.depth} for c in circuits}
ff_edesign = pygsti.protocols.FreeformDesign(circuit_info_dict)
```

Next, derive a class from `ModelFreeformSimulator`.  `ModelFreeformSimulator.__init__` takes a dictionary of models keyed by name.  The simulator below compares a noisy model against a perfect one, named `"base"` and `"target"`.

```{code-cell} ipython3
class MyDataSimulator(pygsti.protocols.ModelFreeformSimulator):
    def __init__(self, model, target_model):
        super().__init__({'base': model, 'target': target_model})

    def compute_freeform_data(self, circuit):
        ret = {}  # we return a dict of all the things we compute for this circuit

        # Get the raw ingredients: probabilities, final states and/or circuit process matrices.
        # You'd usually call just *one* of these - the one giving the hardest ingredient you need -
        # and set flags to True to get the easier ingredients, so there's only one forward sim per circuit.
        probs = self.compute_probabilities(circuit)
        final_states = self.compute_final_states(circuit, include_probabilities=False)
        process_matrices = self.compute_process_matrices(circuit, include_final_state=False, include_probabilities=False)

        # Compute the things we want using the ingredients
        A = process_matrices['base']
        B = process_matrices['target']
        ret['process fidelity'] = pygsti.tools.entanglement_fidelity(A, B, 'pp')

        state = pygsti.tools.ppvec_to_stdmx(final_states['base'].to_dense())
        target_state = pygsti.tools.ppvec_to_stdmx(final_states['target'].to_dense())
        ret['final state fidelity'] = pygsti.tools.fidelity(state, target_state)

        p = probs['base']
        q = probs['target']
        ret['TVD'] = 0.5 * sum([abs(p[i] - q[i]) for i in p])

        # Return a dict of all the computed quantities
        return ret
```

The workhorse is `compute_freeform_data`, which returns a dictionary of values for the given `circuit`.  The base class hands you three ways to get raw ingredients:

1. the circuit's outcome probabilities, via `compute_probabilities`
2. the final state, right before measurement, via `compute_final_states`
3. the overall action of the circuit as a process matrix, excluding state preparation and measurement, via `compute_process_matrices`

Only *one* of these should be needed; the example calls all three just to show them off.  When you want several kinds of value, call the method that does the hardest computation and turn on its `include_*` arguments.  For both final states and outcome probabilities, that means `compute_final_states(circuit, include_probabilities=True)`.  Note that `compute_process_matrices` works by densifying each layer operation and multiplying the results together, so it needs operations that have a dense representation and it scales badly with qubit count.  All three `compute_*` functions return dictionaries keyed by the model names given to `__init__`.

This example computes the total variation distance between the "base" and "target" outcome distributions, the state fidelity between the final states, and the process fidelity between the circuit actions as quantum processes.

Now build the simulator and run it on the experiment design.

```{code-cell} ipython3
mysim = MyDataSimulator(noisy_mdl, smq1Q_XYI.target_model())  # constructs it; doesn't run yet
ff_data = mysim.run(ff_edesign)
```

`ff_data` is a `ProtocolData` whose `.dataset` is a `FreeformDataSet`, holding a dictionary of computed values per circuit:

```{code-cell} ipython3
ff_data.dataset[C('Gxpi2:0')]
```

A `FreeformDataSet` converts to a Pandas dataframe.  (If the cell below fails, you probably need to install the `pandas` package.)

```{code-cell} ipython3
ff_data.dataset.to_dataframe(pivot_value="Value")
```

The parent `ProtocolData` converts too, and when its experiment design is a `FreeformDesign` the design's metadata comes along.  That's why the dataframe below has a "depth" column and the one above doesn't.

```{code-cell} ipython3
ff_data.to_dataframe(pivot_value="Value")
```

### A dataframe-centered workflow

`FreeformDesign` objects can be loaded from and converted to dataframes, and `ModelFreeformSimulator` has an `apply` method that runs a data simulator on an experiment design's dataframe.  Together these let you stay in Pandas end to end.  For large analyses that's worth something: a dataframe only has to hold string representations of circuits, which load and save much faster than pyGSTi `Circuit` objects, since those require a parsing step.  Circuit objects buy you the ability to manipulate circuits, which is often unnecessary.

Start by converting the experiment design to a dataframe.  Any code producing a dataframe with a `Circuits` column (capital "C") would work equally well here.

```{code-cell} ipython3
edesign_df = ff_edesign.to_dataframe(pivot_value='Value')
edesign_df
```

Then apply the simulator defined above to that dataframe.  The result matches the dataframe built earlier, up to column ordering.

```{code-cell} ipython3
data_df = mysim.apply(edesign_df)
data_df
```

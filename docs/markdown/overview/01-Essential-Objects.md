---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.17.3
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Essential Objects
This tutorial covers several object types that are foundational to much of what pyGSTi does: circuits, processor specifications, models, and data sets.  Our objective is to explain what these objects are and how they relate to one another at a high level while providing links to other notebooks that cover details we skip over here.

```{code-cell} ipython3
import pygsti
from pygsti.circuits import Circuit
from pygsti.models import Model
from pygsti.data import DataSet
```

## Circuits
The `Circuit` object encapsulates a quantum circuit as a sequence of *layers*, each of which contains zero or more non-identity *gates*.  A `Circuit` has some number of labeled *lines* and each gate label is assigned to one or more lines. Line labels can be integers or strings.   Gate labels have two parts: a `str`-type name and a tuple of line labels.  A gate name typically begins with 'G' because this is expected when we parse circuits from text files.

For example, `('Gx',0)` is a gate label that means "do the Gx gate on qubit 0", and `('Gcnot',(2,3))` means "do the Gcnot gate on qubits 2 and 3".

A `Circuit` can be created from a list of gate labels:

```{code-cell} ipython3
c = Circuit( [('Gx',0),('Gcnot',0,1),(),('Gy',3)], line_labels=[0,1,2,3])
print(c)
```

If you want multiple gates in a single layer, just put those gate labels in their own nested list:

```{code-cell} ipython3
c = Circuit( [('Gx',0),[('Gcnot',0,1),('Gy',3)],()] , line_labels=[0,1,2,3])
print(c)
```

We distinguish three basic types of circuit layers.  We call layers containing quantum gates *operation layers*.  All the circuits we've seen so far just have operation layers.  It's also possible to have a *preparation layer* at the beginning of a circuit and a *measurement layer* at the end of a circuit.  There can also be a fourth type of layer called an *instrument layer* which we dicuss in a separate [tutorial on Instruments](../objects/Instruments).  Assuming that `'rho'` labels a (n-qubit) state preparation and `'Mz'` labels a (n-qubit) measurement, here's a circuit with all three types of layers:

```{code-cell} ipython3
c = Circuit( ['rho',('Gz',1),[('Gswap',0,1),('Gy',2)],'Mz'] , line_labels=[0,1,2])
print(c)
```

Finally, when dealing with small systems (e.g. 1 or 2 qubits), we typically just use a `str`-type label (without any line-labels) to denote every possible layer.  In this case, all the labels operate on the entire state space so we don't need the notion of 'lines' in a `Circuit`.  When there are no line-labels, a `Circuit` assumes a single default **'\*'-label**, which you can usually just ignore:

```{code-cell} ipython3
c = Circuit( ['Gx','Gy','Gi'] )
print(c)
```

Pretty simple, right?  The `Circuit` object allows you to easily manipulate its labels (similar to a NumPy array) and even perform some basic operations like depth reduction and simple compiling.  For lots more details on how to create, modify, and use circuit objects see the [circuit tutorial](../objects/Circuit).

## Processor Specifications
A processor specification describes the interface that a quantum processor exposes to the outside world.  Actual quantum processors often have a "native" interface associated with them, but can also be viewed as implementing various other derived interfaces.  For example, while a 1-qubit quantum processor may natively implement the $X(\pi/2)$ and $Z(\pi/2)$ gates, it can also implement the set of all 1-qubit Clifford gates.  Both of these interfaces would correspond to a processor specification in pyGSTi.

Currently pyGSTi only supports processor specifications having an integral number of qubits.  The `QubitProcessorSpec` object describes the number of qubits and what gates are available on them. For example,

```{code-cell} ipython3
pspec = pygsti.processors.QubitProcessorSpec(num_qubits=2, gate_names=['Gxpi2', 'Gypi2', 'Gcnot'],
                                             geometry="line")
print("Qubit labels are", pspec.qubit_labels)
print("X(pi/2) gates on qubits: ", pspec.resolved_availability('Gxpi2'))
print("CNOT gates on qubits: ", pspec.resolved_availability('Gcnot'))
```

creates a processor specification for a 2-qubits with $X(\pi/2)$, $Y(\pi/2)$, and CNOT gates.  Setting the geometry to `"line"` causes 1-qubit gates to be available on each qubit and the CNOT between the two qubits (in either control/target direction).  Processor specifications are used to build experiment designs and models, and so defining or importing an appropriate processor specification is often the first step in many analyses.  To learn more about processor specification objects, see the [processor specification tutorial](../objects/ProcessorSpec).


## Models
An instance of the `Model` class represents something that can predict the outcome probabilities of quantum circuits.  We define any such thing to be a "QIP model", or just a "model", as these probabilities define the behavior of some real or virtual QIP.  Because there are so many types of models, the `Model` class in pyGSTi is just a base class and is never instaniated directly.  Classes `ExplicitOpModel` and `ImplicitOpModel` (subclasses of `Model`) define two broad categories of models, both of which sequentially operate on circuit *layers* (the "Op" in the class names is short for "layer operation"). 

### Explicit layer-operation models
An `ExplicitOpModel` is a container object.  Its `.preps`, `.povms`, and `.operations` members are essentially dictionaires of state preparation, measurement, and layer-operation objects, respectively.  How to create these objects and build up explicit models from scratch is a central capability of pyGSTi and a topic of the [explicit-model tutorial](../objects/ExplicitModel).  Presently, we'll create a 2-qubit model using the processor specification above via the `create_explicit_model` function:

```{code-cell} ipython3
mdl = pygsti.models.create_explicit_model(pspec)
```

This creates an `ExplicitOpModel` with a default preparation (prepares all qubits in the zero-state) labeled `'rho0'`, a default measurement labeled `'Mdefault'` in the Z-basis and with 5 layer-operations given by the labels in the 2nd argument (the first argument is akin to a circuit's line labels and the third argument contains special strings that the function understands):

```{code-cell} ipython3
print("Preparations: ", ', '.join(map(str,mdl.preps.keys())))
print("Measurements: ", ', '.join(map(str,mdl.povms.keys())))
print("Layer Ops: ",    ', '.join(map(str,mdl.operations.keys())))
```

We can now use this model to do what models were made to do: compute the outcome probabilities of circuits.

```{code-cell} ipython3
c = Circuit( [('Gxpi2',0),('Gcnot',0,1),('Gypi2',1)] , line_labels=[0,1])
print(c)
mdl.probabilities(c) # Compute the outcome probabilities of circuit `c`
```

An `ExplictOpModel` only "knows" how to operate on circuit layers it explicitly contains in its dictionaries,
so, for example, a circuit layer with two X gates in parallel (layer-label = `[('Gxpi2',0),('Gxpi2',1)]`) cannot be used with our model until we explicitly associate an operation with the layer-label `[('Gxpi2',0),('Gxpi2',1)]`:

```{code-cell} ipython3
import numpy as np

c = Circuit( [[('Gxpi2',0),('Gxpi2',1)],('Gxpi2',1)] , line_labels=[0,1])
print(c)

try: 
    p = mdl.probabilities(c)
except KeyError as e:
    print("!!KeyError: ",str(e))
    
    #Create an operation for two parallel X-gates & rerun (now it works!)
    mdl.operations[ [('Gxpi2',0),('Gxpi2',1)] ] = np.dot(mdl.operations[('Gxpi2',0)].to_dense(),
                                                         mdl.operations[('Gxpi2',1)].to_dense())
    p = mdl.probabilities(c)
    
print("Probability_of_outcome(00) = ", p['00']) # p is like a dictionary of outcomes
```

```{code-cell} ipython3
mdl.probabilities((('Gxpi2',0),('Gcnot',0,1)))
```

### Implicit layer-operation models
In the above example, you saw how it is possible to manually add a layer-operation to an `ExplicitOpModel` based on its other, more primitive layer operations.  This often works fine for a few qubits, but can quickly become tedious as the number of qubits increases (since the number of potential layers that involve a given set of gates grows exponentially with qubit number).  This is where `ImplicitOpModel` objects come into play: these models contain rules for building up arbitrary layer-operations based on more primitive operations.  PyGSTi offers several "built-in" types of implicit models and a rich set of tools for building your own custom ones.  See the [tutorial on implicit models](../objects/ImplicitModel) for details. 

## Data Sets
The `DataSet` object is a container for tabulated outcome counts.  It behaves like a dictionary whose keys are `Circuit` objects and whose values are dictionaries that associate *outcome labels* with (usually) integer counts.  There are two primary ways you go about getting a `DataSet`.  The first is by reading in a simply formatted text file:

```{code-cell} ipython3
dataset_txt = \
"""## Columns = 00 count, 01 count, 10 count, 11 count
{}                    100   0   0   0
Gxpi2:0                55   5  40   0
Gxpi2:0Gypi2:1         20  27  23  30
Gxpi2:0^4              85   3  10   2
Gxpi2:0Gcnot:0:1       45   1   4  50
[Gxpi2:0Gxpi2:1]Gypi2:0 25  32  17  26
"""
with open("../../tutorial_files/Example_Short_Dataset.txt","w") as f:
    f.write(dataset_txt)
ds = pygsti.io.read_dataset("../../tutorial_files/Example_Short_Dataset.txt")
```

The second is by simulating a `Model` and thereby generating "fake data".  This essentially calls `mdl.probabilities(c)` for each circuit in a given list, and samples from the output probability distribution to obtain outcome counts:

```{code-cell} ipython3
circuit_list = pygsti.circuits.to_circuits([ (), 
                                             (('Gxpi2',0),),
                                             (('Gxpi2',0),('Gypi2',1)),
                                             (('Gxpi2',0),)*4,
                                             (('Gxpi2',0),('Gcnot',0,1)),
                                             ((('Gxpi2',0),('Gxpi2',1)),('Gxpi2',0)) ], line_labels=(0,1))
ds_fake = pygsti.data.simulate_data(mdl, circuit_list, num_samples=100,
                                                 sample_error='multinomial', seed=8675309)
```

Outcome counts are accessible by indexing a `DataSet` as if it were a dictionary with `Circuit` keys:

```{code-cell} ipython3
c = Circuit( (('Gxpi2',0),('Gypi2',1)), line_labels=(0,1) )
print(ds[c])                     # index using a Circuit
print(ds[ [('Gxpi2',0),('Gypi2',1)] ]) # or with something that can be converted to a Circuit 
```

Because `DataSet` object can also store *timestamped* data (see the [time-dependent data tutorial](../objects/TimestampedDataSets)), the values or "rows" of a `DataSet` aren't simple dictionary objects.  When you'd like a `dict` of counts use the `.counts` member of a data set row:

```{code-cell} ipython3
row = ds[c]
row['00'] # this is ok
for outlbl, cnt in row.counts.items(): # Note: `row` doesn't have .items(), need ".counts"
    print(outlbl, cnt)
```

Another thing to note is that `DataSet` objects can be made "sparse" by dropping 0-counts:

```{code-cell} ipython3
ds_sparse = ds_fake.drop_zero_counts()

c = Circuit([('Gxpi2',0)], line_labels=(0,1))
print("No 01 or 11 outcomes here: ",ds_fake[c])
for outlbl, cnt in ds_sparse[c].counts.items():
    print("Item: ",outlbl, cnt) # Note: this loop never loops over 01 or 11!
```

In this case, simulated `Datasets` can be initialized to always drop 0-counts also:

```{code-cell} ipython3
ds_sparse2 = pygsti.data.simulate_data(mdl, circuit_list, num_samples=100,
                                       sample_error='multinomial', seed=8675309,
                                       record_zero_counts=False)


for outlbl, cnt in ds_sparse2[c].counts.items():
    print("Item: ",outlbl, cnt) # Note: this loop never loops over 01 or 11!
```

You can manipulate `DataSets` in a variety of ways, including:
- adding and removing rows
- "trucating" a `DataSet` to include only a subset of it's string
- "filtering" a $n$-qubit `DataSet` to a $m < n$-qubit dataset

To find out more about these and other operations, see our [data set tutorial](../objects/DataSet).

## What's next?
You've learned about the three main object types in pyGSTi!  The next step is to learn about how these objects are used within pyGSTi, which is the topic of the next [overview tutorial](02-Using-Essential-Objects).  Alternatively, if you're interested in learning more about the above-described or other objects, here are some links to relevant tutorials:
- [Circuit](../objects/Circuit) - how to build circuits ([GST circuits](../gst/CircuitConstruction) in particular)
- [ExplicitModel](../objects/ExplicitModel) - constructing explicit layer-operation models
- [ImplicitModel](../objects/ImplicitModel) - constructing implicit layer-operation models
- [ModelNoise](../objects/ModelNoise) - describing noise for both explicit and implicit models
- [ParameterBounds](../objects/ParameterBounds) - bounding model parameters
- [DataSet](../objects/DataSet) - constructing data sets ([timestamped data](../objects/TimestampedDataSets) in particular)
- [Basis](../objects/MatrixBases) - defining matrix and vector bases
- [Results](../objects/Results) - the container object for model-based results
- [ProcessorSpec](../objects/ProcessorSpec) - represents a QIP as a collection of models and meta information. 
- [Instrument](../objects/Instruments) - allows for circuits with intermediate measurements
- [Operation Factories](../objects/OperationFactories) - allows continuously parameterized gates

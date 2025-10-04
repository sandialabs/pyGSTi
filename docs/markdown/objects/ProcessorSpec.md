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

# Processor Specifications
This tutorial covers the creation and use of processor specification objects.  These objects are used to define the application programming interface (API) exposed by a quantum information processor (QIP).  Currently, processor specifications in pyGSTi are restricted to QIPs with a given number of *qubits* (`QubitProcessorSpec`), though in the future support may be added for more exotic processor types containing, for example qudits or a mix of qubits and classical bits.  Often a processor specification is created for the "primitive" gates of the device, though it can be useful to create processor specifications for "compiled" gates as well.

```{code-cell} ipython3
import pygsti
```

## Using a `QubitProcessorSpec` to specify a multi-qubit device.
A `QubitProcessorSpec` object specifies the API of a quantum computer in terms of operations on some number of qubits.  It holds three main quantities:

1. The number of qubits in the device, and, optionally, the labels of these qubits.

2. The gate operations allowed on the qubits.  Gates are specified by associating labels with (usually 1- or 2-qubit) unitary matrices.  Common gate labels are recognized by pyGSTi (e.g. `"Gxpi2"` specifies a 1-qubit $X(\pi/2)$ rotation gate). Non-built-in gate names must be accompanied by a corresponding unitary matrix.  Continuously-parameterized gates (e.g. a $z$-rotation by an arbitrary angle) can be specified by supplying a *function* that returns a unitary matrix instead.  

3. The availability of the gates (possibly via the connectivity of the device).  Sets of the allowed target qubit labels for each gate specify where in the processor each gate name can be applied.  Alternatively, special values such as `"all-edges"` can be used in conjunction with a `QubitGraph` object that gives the connectivity of the qubits. 

(Note that currently processor specifications do not specify the preparation and measurement operations available on a processor.  This functionality will be added in the future.)

So let's create a `QubitProcessorSpec`.  First, we'll choose the number of qubits:

```{code-cell} ipython3
n_qubits = 4
```

Next, we pick some names for the qubits.  These are akin to the *line labels* in a `Circuit` object (see the [Circuit tutorial](../Circuit.ipynb)).  Qubits are typically labelled by names beginning with "Q" or integers (if not specified, the qubit labels default to the integers $0, 1, 2, \ldots$).  Here we choose:

```{code-cell} ipython3
qubit_labels = ['Q0','Q1','Q2','Q3']
```

Next, we pick a set of fundamental gates. These can be specified via in-built names,such as 'Gcnot' for a CNOT gate. The full set of in-built names is specified in the dictionary returned by `pygsti.tools.internalgates.standard_gatename_unitaries()`, and note that there is redundency in this set. E.g., 'Gi' is a 1-qubit identity gate but so is 'Gc0' (as one of the 24 1-qubit Cliffords named as 'Gci' for i = 0, 1, 2, ...).  Additionally, any gate name with "idle" in it, e.g., `"Gidle"`, `"Gglobalidle"`, or `"{idle}"`, is recognized as a idle gate on some number of qubits.  That number is determined by the gate's availability, if any is given, and defaults to all the qubits.  Gate names in curly braces are interpreted as "hidden" gates by models, meaning that such gates are omitted when listing the "primitive operations" of the model.

```{code-cell} ipython3
gate_names = ['Gxpi2', # A X rotation by pi/2
              'Gypi2', # A Y rotation by pi/2
              'Gzpi2', # A Z rotation by pi/2
              'Gh', # The Hadamard gate
              'Gcphase']  # The controlled-Z gate.
```

Additionally, we can define gates with user-specified names and actions, via a dictionary with keys that are strings (gate names) and values that are unitary matrices. For example, if you want to call the hadamard gate 'Ghad' we could do this here. The gate names should all start with a 'G', but are otherwise unrestricted. Here we'll leave this dictionary empty.

```{code-cell} ipython3
nonstd_gate_unitaries = {}
```

Specify the "availability" of gates: which qubits they can be applied to. When not specified for a gate, it is assumed that it can be applied to all dimension-appropriate sets of qubits. E.g., a 1-qubit gate will be assumed to be applicable to each qubit; a 2-qubit gate will be assumed to be applicable to all ordered pairs of qubits, etc.

Let's make our device have ring connectivity:

```{code-cell} ipython3
availability = {
    'Gxpi2': [('Q0',), ('Q1',)],
    'Gypi2': [('Q0',), ('Q1',)],
    'Gzpi2': [('Q0',), ('Q1',)],
    'Gh': [('Q0',), ('Q1',)],
    'Gcphase':[('Q0','Q1'),('Q1','Q2'),('Q2','Q3'),('Q3','Q0')]}
```

We then create a `QubitProcessorSpec` by handing it all of this information. This then generates a variety of auxillary information about the device from this input (e.g., optimal compilations for the Pauli operators and CNOT). The defaults here that haven't been specified will be ok for most purposes. But sometimes they will need to be changed to avoid slow QubitProcessorSpec initialization - fixes for these issues will likely be implemented in the future.

```{code-cell} ipython3
pspec = pygsti.processors.QubitProcessorSpec(num_qubits=n_qubits, gate_names=gate_names,
                                 nonstd_gate_unitaries=nonstd_gate_unitaries, 
                                 availability=availability, qubit_labels=qubit_labels)
```

### Geometries and availability
The availability of gates can also be specified via the `geometry` argument.  The geometry is a graph containing connectivity information.  You can set `geometry` to a special builtin name like `"line"` or `"grid"`, or any `pygsti.baseobjs.QubitGraph` object.  The elements of the `availability` argument (a dictionary) can then take the special value `"all-edges"`, which they also default to.  When `"all-edges"` is given as an availability for a 2-qubit gate, the availability is taken to be all the edges of the graph.  Here's an example of 9 qubits on a grid (note that edges of builtin graphs like `"grid"` are *undirected*, so the 2Q gates occur in both directions):
~~~
0-1-2
| | |
3-4-5
| | |
6-7-8
~~~

**TODO: create and link to tutorial on graphs here**

```{code-cell} ipython3
grid_pspec = pygsti.processors.QubitProcessorSpec(9, ['Gxpi','Gypi','Gcnot'], geometry='grid')
grid_pspec.resolved_availability('Gcnot')
```

## Non-standard gates
Gates that are not built into pyGSTi can be specifed via the `nonstd_gate_unitaries` argument.  For example, the 

```{code-cell} ipython3
import numpy as np
U = np.array([[1, 0, 0, 0],
              [0, 1, 0, 0],
              [0, 0, 1j, 0],
              [0, 0, 0, -1j]])
assert(np.allclose(U.T.conjugate(), np.linalg.inv(U))) # U is a unitary matrix

pspec = pygsti.processors.QubitProcessorSpec(num_qubits=1, gate_names=['Gi', 'Gcustom'],
                                 nonstd_gate_unitaries={'Gcustom': U},
                                 availability={'Gi': [(0,)], 'Gcustom': [(0,)]})
```

## Continuously parameterized gates
By passing a unitary-matrix-valued "function" in instead of a unitary matrix a continuously parameterized gate can be specified.  This "function" is actually the instance of a subclass of `pygsti.processors.UnitaryGateFunction` containing a `__call__` method that implements the desired function.  This additional complexity is necessary for later serializing the processor specification, as functions in Python are not easily serialized.  In addition to its `__call__` method, the class must have a `shape` attribute, similar to a NumPy array.

```{code-cell} ipython3
from pygsti.baseobjs import UnitaryGateFunction

class MyContinuouslyParameterizedGateFunction(UnitaryGateFunction):
    shape = (2, 2)
    def __call__(self, theta):                                                                                                                                                                                                                                                                                                                                                                                                                   
        return _np.array([[1., 0.], [0., _np.exp(-1j * float(theta[0]))]])
```

```{code-cell} ipython3
pspec = pygsti.processors.QubitProcessorSpec(num_qubits=1, gate_names=['Gi', 'Gcustom'],
                                 nonstd_gate_unitaries={'Gcustom': MyContinuouslyParameterizedGateFunction()},
                                 availability={'Gi': [(0,)], 'Gcustom': [(0,)]})

#Write this to a file then read it pack
pspec.write("../tutorial_files/test_pspec.json")
pspec_readin = pygsti.processors.QubitProcessorSpec.read("../tutorial_files/test_pspec.json")
```

## Idle gates
Specifying idle gates within processor specifications can be done in various ways, and this can sometimes lead to confusion regarding how these idles are modeled in circuits.  Here are two common ways to specify idle operations in a processor spec:

- A single-qubit idle gate, e.g. `"Gi"`, is specified with availability on each qubit.  This is the simplest approach, as the idle appears just as another 1-qubit gate.  This works well with implicit models, that compose the potentially noisy 1-qubit idles into layer operations just like any other gate.  In explicit models every gate is treated as a separate layer, and so the resulting model will have multiple N-qubit idle layers (one per  qubit, e.g. labelled by `("Gi", 0)`, `("Gi", 1)`, etc.).  This may be desirable in some cases, but more often just a single idle layer is desired when working with explicit models (see next item).

- A N-qubit global idle gate is specified.  This can be done by specifying any gate name, e.g. `"Gwait"`, and setting its `nonstd_gate_unitaries` entry to a $2^N \times 2^N$ identity matrix **or to the integer $N$**.  Setting elements of `nonstd_gate_unitaries` to integers indicates that this gate label has target equal to the $N$-qubit identity matrix, but that this matrix should not actually be built unless absolutely necessary (this is useful when $N$ is large and constructing $2^N \times 2^N$ matrices is impractical).  If an "idle"-containing gate name, e.g. `"Gidle"`, is used, then pyGSTi behaves as though `nonstd_gate_unitaries` was set to $N$ by default and there is no need to provide this information.  The availability for a global idle gate can be specified as, e.g., `{'Gidle': [(0,1,...N)], ...}` or `{'Gidle': [None]}` (which one depends on how you want to refer to the layer in circuits -  as `"Gidle:0:1:..N"` or just `"Gidle"`).  If left unspecified the default availability for a "idle"-containing gate name is `[None]`.  When implicit models are constructed they will often (see model construction function arguments for details) automatically detect a global idle operation in the processor spec and associate it (via "layer rules") with the empty-layer label (the empty tuple `()` or string `"[]"`) that conventionally designates a idle layer in circuits. In the explicit model case, however, this automatic association is not performed, and an empty-layer label operation must be added explicitly via an idle gate name in curly braces (see below).

Various algorithms in pyGSTi iterate over a model's "primitive operations".  Sometimes an idle gate, especially when it's a global idle gate, should not be considered one of these operations, as it is undesirable to have the idle gate reported or processed alongside all of the other gates.  Choosing a gate name in curly braces, e.g., `"{idle}"`, indicates to the `ProcessorSpec` and subsequently created models that this gate is "hidden" and should not be included when listing primitive operations.  

**There is one special case regarding gate names in curly braces:** when an `ExpilcitModel` is built, e.g., using `create_explicit_model`, and the supplied processor specification has a global idle gate with a name in curly braces, then this idle gate will be associated to the empty-layer label in the created model and the curly-braced name will not exist within the model.  This special behavior makes it convenient to construct explicit models with an empty-layer operation (note that all the gate names in a `QubitProcessorSpec` must be simple strings and cannot be empty or non-empty layer labels). 

The cells below illustrate this behavior in some simple cases.  For more information on explicit and implicit models, see the tutorials linked below.

```{code-cell} ipython3
print("Case 1: processor spec with 1-qubit Gi gate\n")
pspec_Gi = pygsti.processors.QubitProcessorSpec(num_qubits=2, gate_names=['Gi'],
                                             availability={'Gi': [(0,), (1,)]})
explicit_model = pygsti.models.modelconstruction.create_explicit_model(pspec_Gi)
print("Explicit model has multiple idle (layer) operations:\n", list(explicit_model.operations.keys()))

print("")

implicit_model = pygsti.models.modelconstruction.create_crosstalk_free_model(pspec_Gi)
print("Implicit model has a single Gi gate, two primitive layers, and an automatically-constructed global idle:")
print("  gates: ", list(implicit_model.operation_blks['gates'].keys()))
print("  layers: ", list(implicit_model.operation_blks['layers'].keys()))
```

```{code-cell} ipython3
print("Case 2: processor spec with N-qubit Gwait gate, specified with integer\n")
pspec_Gwait = pygsti.processors.QubitProcessorSpec(num_qubits=2, gate_names=['Gwait'],
                                             nonstd_gate_unitaries={'Gwait': 2},
                                             availability={'Gwait': [None]})

explicit_model = pygsti.models.modelconstruction.create_explicit_model(pspec_Gwait)
print("Explicit model has single idle (layer) operation:\n", list(explicit_model.operations.keys()))

print("")

implicit_model = pygsti.models.modelconstruction.create_crosstalk_free_model(pspec_Gwait)
print("Implicit model has a single Gwait gate & layer, which is also used to simulate empty circuit layers:")
print("  gates: ", list(implicit_model.operation_blks['gates'].keys()))
print("  layers: ", list(implicit_model.operation_blks['layers'].keys()))
```

```{code-cell} ipython3
print(("Case 3: processor spec with N-qubit Gidle gate; defaults for 'idle'-containing"
       " gates allow us to omit availability and nonstd_gate_unitaries\n"))
pspec_Gidle = pygsti.processors.QubitProcessorSpec(num_qubits=2, gate_names=['Gidle'], availability={})

explicit_model = pygsti.models.modelconstruction.create_explicit_model(pspec_Gidle)
print("Explicit model has single idle (layer) operation:\n", list(explicit_model.operations.keys()))

print("")

implicit_model = pygsti.models.modelconstruction.create_crosstalk_free_model(pspec_Gidle)
print("Implicit model has a single Gidle gate & layer, which is also used to simulate empty circuit layers:")
print("  gates: ", list(implicit_model.operation_blks['gates'].keys()))
print("  layers: ", list(implicit_model.operation_blks['layers'].keys()))
```

```{code-cell} ipython3
print(("Case 4: similar to above except we specify availability == [(0,1)] instead of [(None)] (the default).\n"))
pspec_Gidle2 = pygsti.processors.QubitProcessorSpec(num_qubits=2, gate_names=['Gidle'],
                                                   availability={'Gidle': [(0, 1)]})  # still a global idle, but model op labels are different

explicit_model = pygsti.models.modelconstruction.create_explicit_model(pspec_Gidle2)
print("Explicit model has single idle (layer) operation, but note laybe is Gidle:0:1 not just Gidle:\n", list(explicit_model.operations.keys()))

print("")

implicit_model = pygsti.models.modelconstruction.create_crosstalk_free_model(pspec_Gidle2)
print(("Implicit model has a single Gidle gate & layer, but now layer has explicit state-space labels.\n"
       " Still, this layer is detected as a global idle and used to simulate empty circuit layers:"))
print("  gates: ", list(implicit_model.operation_blks['gates'].keys()))
print("  layers: ", list(implicit_model.operation_blks['layers'].keys()))
```

```{code-cell} ipython3
print(("Case 5: N-qubit idle assocated with a curly-braced 'idle'-containing gate name.\n"))
pspec_curlyidle = pygsti.processors.QubitProcessorSpec(num_qubits=2, gate_names=['{idle}'], availability={})

explicit_model = pygsti.models.modelconstruction.create_explicit_model(pspec_curlyidle)
print("Explicit model shows SPECIAL BEHAVIOR and instead of an '{idle}' label associates the global idle gate with the empty-circuit-layer label:\n", list(explicit_model.operations.keys()))

print("")

implicit_model = pygsti.models.modelconstruction.create_crosstalk_free_model(pspec_curlyidle)
print("Implicit model has a single {idle} gate & layer just like for 'Gidle' above (no special behavior):")
print("  gates: ", list(implicit_model.operation_blks['gates'].keys()))
print("  layers: ", list(implicit_model.operation_blks['layers'].keys()))

print("\nNote, however, that {idle} doesn't show up in the primitive ops of the processor spec or implicit model:")
print(pspec_curlyidle.primitive_op_labels)
print(implicit_model.primitive_op_labels)

print("\nThis isn't true of the explicit model, which reports all of its operations as 'primitive' ones:")
print(explicit_model.primitive_op_labels)
```

## Compilation Rules
<font style="color:red">Note: This is a new and incomplete feature in pyGSTi</font>

A new functionality currently under development allows new `QubitProcessorSpec` objects to be created from old ones by applying a set of "compilation rules".  These rules, for instance, allow us to create a new processor specification that contains $\pi$-rotation gates:

```{code-cell} ipython3
from pygsti.processors.compilationrules import CompilationRules
pspec = pygsti.processors.QubitProcessorSpec(num_qubits=1, gate_names=['Gxpi2', 'Gypi2'],
                                             availability={'Gxpi2': [(0,)], 'Gypi2': [(0,)]})

rules = CompilationRules()
rules.add_compilation_rule('Gxpi', pygsti.circuits.Circuit("Gxpi2:0^2"), (0,))
rules.add_compilation_rule('Gypi', pygsti.circuits.Circuit("Gypi2:0^2"), (0,))

pspec2 = rules.apply_to_processorspec(pspec, action='add')
pspec2.gate_names
```

In the future, these compilation rules will also be able to be applied to circuits to convert circuits intended for the "derived" processor specification into circuits for the original "native" one.

+++

## Next Steps:
`QubitProcessorSpec` objects are primarily used for creating models and experiment designs.  Most of the functions for creating models (see the [explicit model tutorial](ExplicitModel.ipynb) and [implicit model tutorial](ImplicitModel.ipynb)) take as their first argument a processor specification.  Processor specifications are also used to construct randomized benchmarking (RB) experiment designs (see the [Clifford](../algorithms/RB-CliffordRB.ipynb), [Direct](../algorithms/RB-DirectRB.ipynb) and [Mirror](../algorithms/RB-MirrorRB.ipynb) RB tutorials) as well as in gate set tomography experiment designs.

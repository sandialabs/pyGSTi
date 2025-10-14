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

# Explicit Models
This tutorial will show you how to create and use `ExplicitOpModel` objects.  `Model` objects are fundamental to pyGSTi, as each represents a set of quantum gates along with state preparation and measurement (i.e. POVM) operations.  In pyGSTi, a "state space" refers to a Hilbert space of *pure* quantum states (often thought of as length-$d$ vectors, where $d=2^N$ for $N$ qubits). A "density matrix space" refers to a Hilbert space of density matrices, which while often thought of as $d \times d$ matrices can also be represented by length $d^2$ vectors.  Mathematically, these vectors live in Hilbert-Schmidt space, the space of linear operators on the original $d\times d$ density matrix space.  pyGSTi uses the "Liouville" vector-representation for density matrices and POVM effects, since this allows quantum gates to be represented by $d^2 \times d^2$ matrices which act on Hilbert-Schmidt vectors.

`ExplicitOpModel` objects are the simplest type of `Model` objects in pyGSTi.  They have the look and feel of Python dictionaries which hold $d^2\times d^2$ operation matrices, length-$d^2$ state preparation vectors, and sets of length-$d^2$ effect vectors which encode positive operator value measures (POVMs).  State preparation and POVM effect vectors are both generically referred to as "SPAM" (state preparation and measurement) vectors.

```{code-cell} ipython3
import pygsti
```

## Creating models
Before getting to `ExplicitOpModels` in particular, lets explain two quantites that *all* `Model` objects posess: a *basis* and *state space labels*:
- A model's `.state_space` member (a `StateSpace` object) describes the model's state space as the direct sum and tensor product of labelled *factors*.  Typically, this is just a tensor product of one or more 2-dimensional qubit spaces labelled by the integers 0 to $N_{qubits}-1$ or `"Q0"`, `"Q1"`, etc.  We specify a 1-qubit state space using `["Q0"]` below (the "Q" tells pyGSTi it's a 2-dimensional *qubit* space).  If you had two qubits you could use `["Q0","Q1"]` or `[0,1]` to describe the tensor product of two qubit spaces, as pyGSTi assumes integer labels stand for qubit spaces too.  To learn more about the `StateSpace` object, see the [state space  tutorial](advanced/StateSpace.ipynb).
- A model's `.basis` member (a `Basis` object) describes how any dense representations (matrices or vectors) of the the operations in a `Model` should be interpreted.  We'll be using the "Pauli product" basis, which is named `"pp"` in pyGSTi and consists of the tensor products of Pauli matrices (since our example has just a 1-qubit state space the `"pp"` basis is just the 4 Pauli matrices $\{\sigma_0,\sigma_X,\sigma_Y,\sigma_Z\}$).  To learn more about `Basis` objects see the [Basis object tutorial](advanced/MatrixBases.ipynb)).


## Creating explicit models
There are more or less four ways to create `ExpicitOpModel` objects in pyGSTi:

* By creating an empty `ExpicitOpModel` and setting its elements directly.
* By a single call to a `pygsti.models.modelconstruction` function, which automates the above approach.
* By loading from a text-format model file using `pygsti.io.read_model` (see the [File IO tutorial](../other/FileIO.ipynb)).
* By loading one from the `pygsti.modelpacks` module (see the [ModelPacks tutorial](advanced/ModelPacks.ipynb).

+++

### Creating a `ExplicitOpModel` from scratch

Layer operations (often called "gates" in a 1- or even 2-qubit context) and SPAM vectors can be assigned to a `ExplicitOpModel` object as to an ordinary python dictionary.  Internally a `ExpicitOpModel` holds these quantities as `LinearOperator`- and `SPAMVec`- and `POVM`-derived objects (all types of `ModelMember` objects from `pygsti.modelmembers`), but you may assign lists, Numpy arrays, or other types of Python iterables to a `ExplicitOpModel` key and a conversion will be performed automatically.  To keep gates, state preparations, and POVMs separate, the `ExplicitOpModel` object looks at the beginning of the dictionary key being assigned: keys beginning with `rho`, `M`, and `G` are categorized as state preparations, POVMs, and gates, respectively.  To avoid ambiguity, each key *must* begin with one of these three prefixes.

To separately access (set or get) the state preparations, POVMs, and operations contained in a `ExplicitOpModel` use the `preps`, `povms`, and `operations` members respectively.  Each one provides dictionary-like access to the underlying objects.  For example, `myModel.operations['Gx']` accesses the same underlying `LinearOperator` object as `myModel['Gx']`, and similarly for `myModel.preps['rho0']` and `myModel['rho0']`.  The values of operations and state preparation vectors can be read and written in this way.  

A POVM object acts similarly to dictionary of `SPAMVec`-derived effect vectors, but typically requires all such vectors to be initialized at once, that is, you cannot assign individual effect vectors to a `POVM`.  The string-valued keys of a `POVM` label the outcome associated with each effect vector, and are therefore termed *effect labels* or *outcome labels*.  The outcome labels also designate data within a `DataSet` object (see the [DataSet tutorial](DataSet.ipynb)), and thereby associate modeled POVMs with experimental measurements. 



The below cell illustrates how to create a `ExplicitOpModel` from scratch.

```{code-cell} ipython3
from math import sqrt
import numpy as np

import pygsti.modelmembers as mm

#Initialize an empty Model object
#Designate the basis being used for the matrices and vectors below 
# as the "Pauli product" basis of dimension 2 - i.e. the four 2x2 Pauli matrices I,X,Y,Z
model1 = pygsti.models.ExplicitOpModel(['Q0'],'pp')

#Populate the Model object with states, effects, gates,
# all in the *normalized* Pauli basis: { I/sqrt(2), X/sqrt(2), Y/sqrt(2), Z/sqrt(2) }
# where I, X, Y, and Z are the standard Pauli matrices.
model1['rho0'] = [ 1/sqrt(2), 0, 0, 1/sqrt(2) ] # density matrix [[1, 0], [0, 0]] in Pauli basis
model1['Mdefault'] = mm.povms.UnconstrainedPOVM(
    {'0': [ 1/sqrt(2), 0, 0, 1/sqrt(2) ],   # projector onto [[1, 0], [0, 0]] in Pauli basis
     '1': [ 1/sqrt(2), 0, 0, -1/sqrt(2) ] },# projector onto [[0, 0], [0, 1]] in Pauli basis
    evotype='densitymx') # Specify the evolution type when initializing from NumPy arrays.
                         # densitymx is the default

model1['Gi'] = np.identity(4,'d') # 4x4 identity matrix
model1['Gx'] = [[1, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 0,-1],
                  [0, 0, 1, 0]] # pi/2 X-rotation in Pauli basis

model1['Gy'] = [[1, 0, 0, 0],
                  [0, 0, 0, 1],
                  [0, 0, 1, 0],
                  [0,-1, 0, 0]] # pi/2 Y-rotation in Pauli basis
```

### Creating a `ExplicitOpModel` from scratch using `modelconstruction.create_operation` and `modelconstruction.create_spam_vector`
The `modelconstruction.create_operation` and `modelconstruction.create_spam_vector` functions take a human-readable string representation of a gate or SPAM vector, and return a `LinearOperator` or `SPAMVector` object that gets stored in a dictionary-like `ExplicitOpModel` or `POVM` object.  To use these functions, you must specify what state space you're working with, and the basis for that space - so the `.state_space_labels` and `.basis` member of your `Model` object, as described above.

`create_spam_vector` currently only understands strings which are integers (e.g. "1"), for which it creates a vector performing state preparation of (or, equivalently, a state projection onto) the $i^{th}$ state of the Hilbert space, that is, the state corresponding to the $i^{th}$ row and column of the $d\times d$ density matrix.

`create_operation` accepts a wider range of descriptor strings, which take the form of *functionName*(*args*) and include:
- `I(label0, label1, ...)` : the identity on the spaces labeled by `label0`, `label1`, etc.
- `X(theta,Qlabel)`, `Y(theta,Qlabel)`, `Z(theta,Qlabel)` : single qubit X-, Y-, and Z-axis rotations by angle `theta` (in radians) on the qubit labeled by `Qlabel`.  Note that `pi` can be used within an expression for `theta`, e.g. `X(pi/2,Q0)`.
- `CX(theta, Qlabel1, Qlabel2)`, `CY(theta, Qlabel1, Qlabel2)`, `CZ(theta, Qlabel1, Qlabel2)` : two-qubit controlled rotations by angle `theta` (in radians) on qubits `Qlabel1` (the control) and `Qlabel2` (the target).

```{code-cell} ipython3
#Initialize an empty Model object
model2 = pygsti.models.ExplicitOpModel(['Q0'],'pp') # single qubit labelled 'Q0'; Pauli basis
statespace = model2.state_space
basis = model2.basis

from pygsti.models import modelconstruction as mc

#Populate the Model object with states, effects, and gates using 
# build_vector, build_operation, and create_identity_vec.   
model2['rho0'] = mc.create_spam_vector("0", statespace, basis)
model2['Mdefault'] = mm.povms.UnconstrainedPOVM(
    { '0': mc.create_spam_vector("0", statespace, basis),
      '1': mc.create_spam_vector("1", statespace, basis) },
    evotype='densitymx')
model2['Gi'] = mc.create_operation("I(Q0)", statespace, basis)
model2['Gx'] = mc.create_operation("X(pi/2,Q0)", statespace, basis)
model2['Gy'] = mc.create_operation("Y(pi/2,Q0)", statespace, basis)
```

### Create a `ExplicitOpModel` in a single call to `create_explicit_model_from_expressions`
The approach illustrated above using calls to `create_spam_vector` and `create_operation` can be performed in a single call to `create_explicit_model_from_expresssions`.  You will notice that all of the arguments to `create_explicit_model_from_expressions` correspond to those used to construct a model using `create_spam_vector` and `create_operation`; the `create_explicit_model_from_expressions` function is merely a convenience function which allows you to specify everything at once.  These arguments are:
- Arg 1 : the state-space-labels, as described above.
- Args 2 & 3 : list-of-gate-labels, list-of-gate-expressions (labels *must* begin with 'G'; "expressions" being the descriptor strings passed to `create_operation`)
- Args 4 & 5 : list-of-prep-labels, list-of-prep-expressions (labels *must* begin with 'rho'; "expressions" being the descriptor strings passed to `create_spam_vector`)
- Args 6 & 7 : list-of-effect-labels, list-of-effect-expressions (labels can be anything; "expressions" being the descriptor strings passed to `create_spam_vector`).  These effect vectors will comprise a single POVM named `"Mdefault"` by default, but which can be changed via the `povmLabels` argument (see doc string for details).

The optional argument `basis` can be set to any of the known built-in basis *names* (e.g. `"gm"`, `"pp"`, `"qt"`, or `"std"`) to select the basis for the Model as described above.  By default, `"pp"` is used when possible (if the state space corresponds to an integer number of qubits), `"qt"` if the state space has dimension 3, and `"gm"` otherwise.  Optional arguments `gate_type`, `prep_type`, and `povm_type` are used to specify the type of created gate, state, and POVM objects respectively (see below).

```{code-cell} ipython3
model3 = mc.create_explicit_model_from_expressions(['Q0'],
    ['Gi','Gx','Gy'], [ "I(Q0)","X(pi/2,Q0)", "Y(pi/2,Q0)"],
    prep_labels=['rho0'], prep_expressions=["0"], 
    effect_labels=['0','1'], effect_expressions=["0","1"] ) 
```

In this case, the parameters to `create_explicit_model_from_expressions`, specify:

 - The state space has dimension 2 and is interpreted as that of a single qubit labeled "Q0" (label must begin with 'Q' or be an integer if we don't want to create a full `StateSpace` object that contains the dimensions too.)
 
 - there are three gates: Idle, $\pi/2$ x-rotation, $\pi/2$ y-rotation, labeled `Gi`, `Gx`, and `Gy`.
 
 - there is one state prep operation, labeled `rho0`, which prepares the 0-state (the first basis element of the 2D state space)
 
 - there is one POVM (~ measurement), named `Mdefault` with two effect vectors: `'0'` projects onto the 0-state (the first basis element of the 2D state space) and `'1'` projects onto the 1-state.
 
Note that **by default**, there is a single state prep, `"rho0"`, that prepares the 0-state and a single POVM, `"Mdefault"`, which consists of projectors onto each standard basis state that are labelled by their integer indices (so just `'0'` and `'1'` in the case of 1-qubit).  Thus, all but the first four arguments used above just specify the default behavior and can be omitted:

```{code-cell} ipython3
model4 = mc.create_explicit_model_from_expressions( ['Q0'],
    ['Gi','Gx','Gy'], [ "I(Q0)","X(pi/2,Q0)", "Y(pi/2,Q0)"] )
```

The arguments generalize straightforwardly to multiple qubits, though explicit models become impractical (due to the large amount of memory required) with more than two or three qubits.  Here's a 2-qubit example:

```{code-cell} ipython3
model4_2qubit = pygsti.models.create_explicit_model_from_expressions((0,1),
    [(),      ('Gx',0),    ('Gy',0),    ('Gx',1),    ('Gy',1),    ('Gcnot',0,1)],
    ["I(0,1)","X(pi/2,0)", "Y(pi/2,0)", "X(pi/2,1)", "Y(pi/2,1)", "CNOT(0,1)"])
```

## Create a `ExplicitOpModel` in a single call to `create_explicit_model`

While `create_explicit_model_from_expressions` allows the user to define gates based on X, Y, and Z rotations, it is often the case that a set of "standard" gates want to be used. Some of these standard gates are defined by PyGSTi (such as X/Y/Z or $\sqrt{X/Y/Z}$) and can be used directly without specifying their explicit expression. The construction routine for using these standard names is `create_explicit_model`, which takes its information from a `QubitProcessorSpec`.

A `QubitProcessorSpec` is an object that contains information for some experimental device, including:
- Number of qubits
- Names of standard gates
- Qubit labels and topology/availability of gates.
To learn more, see the [processor specification tutorial](ProcessorSpec.ipynb). For our purposes here, it is sufficient to view this as a container for qubit/gate information, and is a common input to most model construction routines.

```{code-cell} ipython3
from pygsti.processors import QubitProcessorSpec

pspec = QubitProcessorSpec(1, ['Gi', 'Gxpi2', 'Gypi2'], qubit_labels=['Q0']) # single qubit with idle, X(pi/2), and Y(pi/2) gates

model5 = mc.create_explicit_model(pspec)
```

```{code-cell} ipython3
# There is one difference with this function - the gate labels are automatically generated
# based on the base gate name and qubit labels.
# To exactly match the other models, we can rename the operations
model5.operations['Gi'] = model5.operations['Gi', 'Q0']
model5.operations['Gx'] = model5.operations['Gxpi2', 'Q0']
model5.operations['Gy'] = model5.operations['Gypi2', 'Q0']
del model5.operations['Gi', 'Q0']
del model5.operations['Gxpi2', 'Q0']
del model5.operations['Gypi2', 'Q0']
```

```{code-cell} ipython3
#All six of the above models are identical.  See this by taking the frobenius differences between them:
assert(model1.frobeniusdist(model2) < 1e-8)
assert(model1.frobeniusdist(model3) < 1e-8)
assert(model1.frobeniusdist(model4) < 1e-8)
assert(model1.frobeniusdist(model5) < 1e-8)
```

## Viewing models
Next, we demonstrate how to print and access information within a `ExplicitOpModel`. We can print the matrix and vector contents of an explicit model by just printing the object:

```{code-cell} ipython3
print("Model 1:\n", model1)
```

```{code-cell} ipython3
#You can also access individual gates like they're numpy arrays:
Gx = model1['Gx'] # a LinearOperator object, but behaves like a numpy array

#By printing a gate, you can see that it's not just a numpy array
print("Gx = ", Gx)

#But can be accessed as one:
print("Array-like printout\n", Gx[:,:],"\n")
print("First row\n", Gx[0,:],"\n")
print("Element [2,3] = ",Gx[2,3], "\n")

Id = np.identity(4,'d')
Id_dot_Gx = np.dot(Id,Gx)
print("Id_dot_Gx\n", Id_dot_Gx, "\n")
```

We can also print the members (state preparations, operations, POVMs) of a model using the `print_modelmembers` method.  This shows the model contents in a more condensed format that includes the type of object each member and a short summary of what it holds.

```{code-cell} ipython3
model1.print_modelmembers()
```

## Basic Operations with Explicit Models

`ExplicitOpModel` objects have a number of methods that support a variety of operations, including:

* Depolarizing or rotating every gate
* Writing the model to a JSON file
* Computing products of operation matrices
* Printing more information about the model

```{code-cell} ipython3
#Add 10% depolarization noise to the gates
depol_model3 = model3.depolarize(op_noise=0.1)

#Add a Y-axis rotation uniformly to all the gates
rot_model3 = model3.rotate(rotate=(0,0.1,0))
```

```{code-cell} ipython3
#Writing a model as a text file
depol_model3.write("../tutorial_files/Example_depolarizedModel.json")
```

```{code-cell} ipython3
print("Probabilities of outcomes of the gate\n sequence GxGx (rho0 and Mdefault assumed)= ",
      depol_model3.probabilities( ("Gx", "Gx")))
print("Probabilities of outcomes of the \"complete\" gate\n sequence rho0+GxGx+Mdefault = ",
      depol_model3.probabilities( ("rho0", "Gx", "Gx", "Mdefault")))
```

It is also possible to manipulate the underlying operations by accessing the forward simulator in the `Model` class. For example, using the `matrix` simulator type, one can compute the product of two gate operations. For more details, see the [ForwardSimulationTypes tutorial](../algorithms/advanced/ForwardSimulationTypes.ipynb).

```{code-cell} ipython3
# Computing the product of operation matrices (only allowed with the matrix simulator type)
print("Product of Gx * Gx = \n",depol_model3.sim.product(("Gx", "Gx")), end='\n\n')
```

## Next steps
Next, you may want to take a look a [implicit models](ImplicitModel.ipynb), which are similar to explicit models but more powerful in ways relevant to multi-qubit modeling.  You may also be interested to check out the [model parameterizations tutorial](ModelParameterization.ipynb) and the [model noise tutoria](ModelNoise.ipynb), which have information relevant to explicit and implicit models.

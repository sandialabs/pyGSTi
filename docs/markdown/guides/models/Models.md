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

# Explicit models

An `ExplicitOpModel` is the simplest kind of `Model` in pyGSTi: a dictionary-like container holding $d^2 \times d^2$ operation matrices, length-$d^2$ state preparation vectors, and sets of length-$d^2$ effect vectors that encode positive operator-valued measures (POVMs). State preparation and POVM effect vectors are collectively called "SPAM" (state preparation and measurement) vectors.

A word on conventions before the mechanics. A "state space" is a Hilbert space of *pure* quantum states, often thought of as length-$d$ vectors with $d = 2^N$ for $N$ qubits. A "density matrix space" is a Hilbert space of density matrices, which you can think of as $d \times d$ matrices or, equivalently, as length-$d^2$ vectors. Those vectors live in Hilbert-Schmidt space, the space of linear operators on the $d \times d$ density matrix space. pyGSTi uses this "Liouville" vector representation for density matrices and POVM effects, which lets quantum gates be represented by $d^2 \times d^2$ matrices acting on Hilbert-Schmidt vectors.

## Setup

Everything on this page uses the following imports.

```{code-cell} ipython3
from math import sqrt

import numpy as np

import pygsti
import pygsti.modelmembers as mm
from pygsti.models import modelconstruction as mc
from pygsti.processors import QubitProcessorSpec
from pygsti.modelpacks import smq2Q_XYICNOT
```

## State spaces and bases

Two quantities belong to *every* `Model`, not just explicit ones.

A model's `.state_space` member (a `StateSpace` object) describes the state space as a direct sum and tensor product of labelled *factors*. Usually this is a tensor product of one or more 2-dimensional qubit spaces labelled by the integers 0 through $N_{qubits}-1$, or by `"Q0"`, `"Q1"`, and so on. Below we specify a 1-qubit state space with `["Q0"]`; the leading "Q" tells pyGSTi the factor is a 2-dimensional *qubit* space. For two qubits use `["Q0","Q1"]` or `[0,1]`, since integer labels also stand for qubit spaces. See the [state space tutorial](../../advanced/conventions/StateSpaces) for more.

A model's `.basis` member (a `Basis` object) says how dense representations (matrices and vectors) of the model's operations should be interpreted. We use the "Pauli product" basis throughout, named `"pp"` in pyGSTi, whose elements are tensor products of Pauli matrices. For a 1-qubit state space that is just $\{\sigma_0,\sigma_X,\sigma_Y,\sigma_Z\}$. See the [Basis object tutorial](../../advanced/conventions/Bases) for more.

## Four ways to build one

There are roughly four routes to an `ExplicitOpModel`:

* Create an empty one and set its elements directly.
* Call a `pygsti.models.modelconstruction` function, which automates the above.
* Load from a text-format model file with `pygsti.io.read_model` (see the [File IO tutorial](../workflow/FilesAndDirectories)).
* Load one from `pygsti.modelpacks` (see the [ModelPacks tutorial](../../start/TargetModels)).

The first three are shown below, building the same 1-qubit model each time.

### From scratch

Layer operations (called "gates" in a 1- or 2-qubit context) and SPAM vectors are assigned to an `ExplicitOpModel` the way you'd assign to an ordinary Python dictionary. Internally the model holds these as `LinearOperator`-, `State`-, and `POVM`-derived objects (all `ModelMember` types from `pygsti.modelmembers`), but you may assign lists, NumPy arrays, or other Python iterables and the conversion happens automatically.

Keys carry type information. The model looks at the start of each key: keys beginning with `rho`, `M`, `G`, and `I` are categorized as state preparations, POVMs, gates, and instruments respectively. Any other key raises a `KeyError`, so there is no ambiguity. (The empty label `Label(())` is the one exception; it is filed under operations.)

The `preps`, `povms`, and `operations` members give you separate dictionary-like access to each category. `myModel.operations['Gx']` reaches the same underlying `LinearOperator` as `myModel['Gx']`, and likewise for `myModel.preps['rho0']` and `myModel['rho0']`. Values can be read and written either way.

A `POVM` behaves like a dictionary of effect vectors, but it typically requires all of them to be initialized at once: you cannot assign individual effect vectors into an existing `POVM`. Its string keys label the outcome associated with each effect vector, and are therefore called *effect labels* or *outcome labels*. Those same labels designate data inside a `DataSet` (see the [DataSet tutorial](../workflow/DataSets)), which is what ties a modeled POVM to an experimental measurement.

```{code-cell} ipython3
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

### From expression strings, member by member

Typing matrices out by hand gets old fast. `mc.create_operation` and `mc.create_spam_vector` take a human-readable string description of a gate or SPAM vector and return the corresponding object, ready to be stored in an `ExplicitOpModel` or `POVM`. Both need to know the state space you're working in and the basis for that space, which is exactly the `.state_space` and `.basis` discussed above.

`create_spam_vector` currently understands only strings that are integers, for instance `"1"`. It creates a vector preparing (or, equivalently, projecting onto) the $i^{th}$ state of the Hilbert space, meaning the state corresponding to the $i^{th}$ row and column of the $d\times d$ density matrix.

`create_operation` accepts a wider range of descriptors, each of the form *functionName*(*args*). The ones you are most likely to want:

- `I(label0, label1, ...)`: the identity on the spaces labeled by `label0`, `label1`, and so on.
- `X(theta,Qlabel)`, `Y(theta,Qlabel)`, `Z(theta,Qlabel)`: single-qubit X-, Y-, and Z-axis rotations by angle `theta` (radians) on the qubit labeled `Qlabel`. You can write `pi` inside an expression for `theta`, as in `X(pi/2,Q0)`.
- `CX(theta, Qlabel1, Qlabel2)`, `CY(theta, Qlabel1, Qlabel2)`, `CZ(theta, Qlabel1, Qlabel2)`: two-qubit controlled rotations by angle `theta` (radians) with `Qlabel1` the control and `Qlabel2` the target.
- `CNOT(Qlabel1, Qlabel2)`, `CPHASE(Qlabel1, Qlabel2)`: the standard controlled-NOT and controlled-phase gates, again with `Qlabel1` the control.

Multiple descriptors can be composed with a colon, so `"X(pi/2,Q0):X(pi/2,Q1)"` is a single operation applying a $\pi/2$ X-rotation to each of two qubits.

That list is not everything the parser handles. `N(theta,sx,sy,sz,Qlabel)` is a single-qubit rotation by `theta` about the axis $(s_x,s_y,s_z)$, and `LX(theta,i1,i2)` is an X-rotation between density-matrix basis states `i1` and `i2`, meant for leakage. `XX(...)` and `ZZ(...)` are two fixed two-qubit unitaries that ignore their arguments and act on the whole (2-qubit) space. `D(...)` is parsed but raises `NotImplementedError`.

```{code-cell} ipython3
#Initialize an empty Model object
model2 = pygsti.models.ExplicitOpModel(['Q0'],'pp') # single qubit labelled 'Q0'; Pauli basis
statespace = model2.state_space
basis = model2.basis

#Populate the Model object with states, effects, and gates
model2['rho0'] = mc.create_spam_vector("0", statespace, basis)
model2['Mdefault'] = mm.povms.UnconstrainedPOVM(
    { '0': mc.create_spam_vector("0", statespace, basis),
      '1': mc.create_spam_vector("1", statespace, basis) },
    evotype='densitymx')
model2['Gi'] = mc.create_operation("I(Q0)", statespace, basis)
model2['Gx'] = mc.create_operation("X(pi/2,Q0)", statespace, basis)
model2['Gy'] = mc.create_operation("Y(pi/2,Q0)", statespace, basis)
```

### From expression strings, all at once

`create_explicit_model_from_expressions` collapses the previous section into one call. Its arguments correspond one-for-one with what you'd pass to `create_spam_vector` and `create_operation`:

- Arg 1: the state space labels, as described above.
- Args 2 & 3: list of gate labels, list of gate expressions. Labels *must* begin with `G`; expressions are the descriptor strings passed to `create_operation`.
- Args 4 & 5: list of prep labels, list of prep expressions. Labels *must* begin with `rho`; expressions are the descriptor strings passed to `create_spam_vector`.
- Args 6 & 7: list of effect labels, list of effect expressions. Effect labels can be anything. These effect vectors form a single POVM named `"Mdefault"` by default, changeable via the `povm_labels` argument (see the docstring).

The optional `basis` argument accepts any built-in basis name (`"gm"`, `"pp"`, `"qt"`, `"std"`). By default it uses `"pp"` when the state space corresponds to an integer number of qubits, `"qt"` when the state space has dimension 3, and `"gm"` otherwise. The `gate_type`, `prep_type`, and `povm_type` arguments control the type of the created gate, state, and POVM objects.

```{code-cell} ipython3
model3 = mc.create_explicit_model_from_expressions(['Q0'],
    ['Gi','Gx','Gy'], [ "I(Q0)","X(pi/2,Q0)", "Y(pi/2,Q0)"],
    prep_labels=['rho0'], prep_expressions=["0"], 
    effect_labels=['0','1'], effect_expressions=["0","1"] ) 
```

Reading those arguments back off: the state space has dimension 2 and is interpreted as a single qubit labeled `"Q0"` (the label must begin with `Q`, or be an integer, if you don't want to build a full `StateSpace` object carrying dimensions too). There are three gates, an idle and $\pi/2$ x- and y-rotations, labeled `Gi`, `Gx`, and `Gy`. There is one state prep, `rho0`, preparing the 0-state (the first basis element of the 2D state space). There is one POVM, `Mdefault`, with two effect vectors: `'0'` projects onto the 0-state and `'1'` onto the 1-state.

That last paragraph describes the **defaults**: a single prep `"rho0"` preparing the 0-state, and a single POVM `"Mdefault"` of projectors onto each standard basis state labelled by integer index. So everything after the first three arguments above is redundant.

```{code-cell} ipython3
model4 = mc.create_explicit_model_from_expressions( ['Q0'],
    ['Gi','Gx','Gy'], [ "I(Q0)","X(pi/2,Q0)", "Y(pi/2,Q0)"] )
```

### From a processor specification

`create_explicit_model_from_expressions` makes you spell out gates as X, Y, and Z rotations. Often you want standard gates by name instead. pyGSTi defines a set of these (X/Y/Z, $\sqrt{X/Y/Z}$, and others) that can be used without writing out an expression, via `create_explicit_model`, which takes its information from a `QubitProcessorSpec`.

A `QubitProcessorSpec` describes an experimental device: number of qubits, names of the standard gates it implements, qubit labels, and the topology or availability of gates. See the [processor specification tutorial](../workflow/DescribeYourDevice) for the full story. Here it's enough to treat it as a container for qubit and gate information, and it's a common input to most model construction routines.

```{code-cell} ipython3
pspec = QubitProcessorSpec(1, ['Gi', 'Gxpi2', 'Gypi2'], qubit_labels=['Q0']) # single qubit with idle, X(pi/2), and Y(pi/2) gates

model5 = mc.create_explicit_model(pspec)
```

One difference: `create_explicit_model` generates gate labels automatically from the base gate name and the qubit labels, so `Gxpi2` on qubit `Q0` becomes the label `('Gxpi2', 'Q0')`. Rename the operations to match the models built above.

```{code-cell} ipython3
model5.operations['Gi'] = model5.operations['Gi', 'Q0']
model5.operations['Gx'] = model5.operations['Gxpi2', 'Q0']
model5.operations['Gy'] = model5.operations['Gypi2', 'Q0']
del model5.operations['Gi', 'Q0']
del model5.operations['Gxpi2', 'Q0']
del model5.operations['Gypi2', 'Q0']
```

All five models are the same. Check by taking Frobenius distances between them.

```{code-cell} ipython3
assert(model1.frobeniusdist(model2) < 1e-8)
assert(model1.frobeniusdist(model3) < 1e-8)
assert(model1.frobeniusdist(model4) < 1e-8)
assert(model1.frobeniusdist(model5) < 1e-8)
```

## Viewing models

Print the object to see the matrix and vector contents of an explicit model.

```{code-cell} ipython3
print("Model 1:\n", model1)
```

```{code-cell} ipython3
#You can access individual gates as LinearOperator objects:
Gx = model1['Gx'] # a LinearOperator object; call .to_dense() to get a numpy array

#By printing a gate, you can see that it's not just a numpy array
print("Gx = ", Gx)

#Use .to_dense() to obtain a plain numpy array for indexing or arithmetic:
print("Array-like printout\n", Gx.to_dense()[:,:],"\n")
print("First row\n", Gx.to_dense()[0,:],"\n")
print("Element [2,3] = ",Gx.to_dense()[2,3], "\n")

Id = np.identity(4,'d')
Id_dot_Gx = np.dot(Id,Gx.to_dense())
print("Id_dot_Gx\n", Id_dot_Gx, "\n")
```

The `print_modelmembers` method gives a more condensed view of the state preparations, operations, and POVMs: the type of each member and a short summary of what it holds.

```{code-cell} ipython3
model1.print_modelmembers()
```

## Basic operations with explicit models

`ExplicitOpModel` objects support depolarizing or rotating every gate, writing themselves to a JSON file, computing products of operation matrices, and computing outcome probabilities.

```{code-cell} ipython3
#Add 10% depolarization noise to the gates
depol_model3 = model3.depolarize(op_noise=0.1)

#Add a Y-axis rotation uniformly to all the gates
rot_model3 = model3.rotate(rotate=(0,0.1,0))
```

```{code-cell} ipython3
#Writing a model as a text file
depol_model3.write("../../../tutorial_files/Example_depolarizedModel.json")
```

```{code-cell} ipython3
print("Probabilities of outcomes of the gate\n sequence GxGx (rho0 and Mdefault assumed)= ",
      depol_model3.probabilities( ("Gx", "Gx")))
print("Probabilities of outcomes of the \"complete\" gate\n sequence rho0+GxGx+Mdefault = ",
      depol_model3.probabilities( ("rho0", "Gx", "Gx", "Mdefault")))
```

You can also reach the underlying operations through the model's forward simulator. With the `matrix` simulator type, for instance, you can compute the product of two gate operations. See the [forward simulators tutorial](../../advanced/simulation/ForwardSimulators) for details.

```{code-cell} ipython3
# Computing the product of operation matrices (only allowed with the matrix simulator type)
print("Product of Gx * Gx = \n",depol_model3.sim.product(("Gx", "Gx")), end='\n\n')
```

## Two-qubit models

Explicit models scale badly. Memory cost grows as $d^4 = 16^N$ per operation, so they become impractical past two or three qubits; for larger devices use [implicit models](MultiQubitModels) instead. Two qubits, though, is a regime where explicit models are still the right tool, and it's the regime where you're most likely to need a gate pyGSTi doesn't know how to name.

The construction generalizes straightforwardly. Here is a 2-qubit model built the same way as `model4` above, with integer qubit labels and a CNOT:

```{code-cell} ipython3
model4_2qubit = mc.create_explicit_model_from_expressions((0,1),
    [(),      ('Gx',0),    ('Gy',0),    ('Gx',1),    ('Gy',1),    ('Gcnot',0,1)],
    ["I(0,1)","X(pi/2,0)", "Y(pi/2,0)", "X(pi/2,1)", "Y(pi/2,1)", "CNOT(0,1)"])
```

The space of possible 2-qubit gates is large enough that sooner or later you'll want one that no expression string covers. The rest of this section builds such a model: standard 1-qubit gates from expressions, plus a custom 2-qubit gate specified by the $4 \times 4$ unitary it performs.

### The single-qubit part

The space of single-qubit gates is small, so assume the 1-qubit gates you want *can* be written as expressions. Build a model containing all of them and nothing else.

```{code-cell} ipython3
target_model = mc.create_explicit_model_from_expressions( 
            [('Q0','Q1')],['Gii','Gix','Giy','Gxi','Gyi'], 
            [ "I(Q0):I(Q1)", "X(pi/2,Q1)", "Y(pi/2,Q1)", "X(pi/2,Q0)", "Y(pi/2,Q0)" ],
            effect_labels=['00','01','10','11'], effect_expressions=["0","1","2","3"])
```

Two arguments here differ from the 1-qubit case. `[('Q0','Q1')]` says to interpret this 4-dimensional space as two qubits `'Q0'` and `'Q1'`, tensored together. The four effect labels `'00'` through `'11'` name the outcomes, and their expressions `"0"` through `"3"` are projections onto the corresponding computational basis elements, 0-based.

A rotation acting on one qubit of a two-qubit space needs no explicit identity on the other, but you can write one if you prefer the symmetry:

```{code-cell} ipython3
mdl_targetB = mc.create_explicit_model_from_expressions( 
            [('Q0','Q1')],['Gii','Gix','Giy','Gxi','Gyi'], 
            [ "I(Q0):I(Q1)", "I(Q0):X(pi/2,Q1)", "I(Q0):Y(pi/2,Q1)", "X(pi/2,Q0):I(Q1)", "Y(pi/2,Q0):I(Q1)" ],
            effect_labels=['00','01','10','11'], effect_expressions=["0","1","2","3"])
assert(abs(target_model.frobeniusdist(mdl_targetB)) < 1e-6)
```

If your 2-qubit gate happens to be one the expression parser already knows, you're done: add it and skip the rest of this section. The parser covers any controlled $X$, $Y$, or $Z$ rotation via `CX`, `CY`, `CZ`, plus the standard `CNOT` and `CPHASE`. Adding `CNOT(Q0,Q1)` here reproduces the `smq2Q_XYICNOT` model pack exactly.

The two use different label conventions, which is worth seeing side by side. Models built from expressions take whatever operation labels you hand them, so the gates above are `Gix`, `Gxi` and so on. Model packs instead name a gate and the qubits it acts on, giving `('Gxpi2', 1)`, `('Gxpi2', 0)` and `('Gcnot', 0, 1)`. The superoperators are the same objects under either naming, so comparing the two means comparing gate by gate through that correspondence rather than calling `frobeniusdist` on the models, which would not find matching keys.

```{code-cell} ipython3
mdl_withCNOT = mc.create_explicit_model_from_expressions( 
            [('Q0','Q1')],['Gii','Gix','Giy','Gxi','Gyi','Gcnot'], 
            [ "I(Q0):I(Q1)", "I(Q0):X(pi/2,Q1)", "I(Q0):Y(pi/2,Q1)", "X(pi/2,Q0):I(Q1)", "Y(pi/2,Q0):I(Q1)", "CNOT(Q0,Q1)" ],
            effect_labels=['00','01','10','11'], effect_expressions=["0","1","2","3"])

#Note this is the same model as one of pyGSTi's standard model packs:
import numpy as np
to_modelpack = {'Gii': (), 'Gix': ('Gxpi2', 1), 'Giy': ('Gypi2', 1),
                'Gxi': ('Gxpi2', 0), 'Gyi': ('Gypi2', 0), 'Gcnot': ('Gcnot', 0, 1)}
mdl_pack = smq2Q_XYICNOT.target_model()
assert all(np.allclose(mdl_withCNOT.operations[ours].to_dense(),
                       mdl_pack.operations[theirs].to_dense())
           for ours, theirs in to_modelpack.items())
```

That equivalence runs backwards too. Since `target_model` holds exactly the 1-qubit gates of the model pack, a third route to it is to load the pack and delete the two-qubit gate.

```{code-cell} ipython3
mdl_targetC = smq2Q_XYICNOT.target_model()
del mdl_targetC.operations[('Gcnot', 0, 1)]
assert all(np.allclose(target_model.operations[ours].to_dense(),
                       mdl_targetC.operations[theirs].to_dense())
           for ours, theirs in to_modelpack.items() if ours != 'Gcnot')
```

### A custom two-qubit gate

Now suppose the parser can't make the gate you want. Write down its unitary and convert. The unitary below rotates the second qubit by $\pi/2$ in the (+) or (-) direction depending on the state of the first.

```{code-cell} ipython3
#Unitary acting on the state-space { |A>, |B>, |C>, |D> } == { |00>, |01>, |10>, |11> }.
myUnitary = 1./np.sqrt(2) * np.array([[1,-1j,0,0],
                                      [-1j,1,0,0],
                                      [0,0,1,1j],
                                      [0,0,1j,1]])

#Convert this unitary into a "superoperator", which acts on the 
# space of vectorized density matrices instead of just the state space.
# These superoperators are what GST calls "gates".
mySuperOp_stdbasis = pygsti.unitary_to_std_process_mx(myUnitary)

#The superoperator is now a complex matrix in the "standard" or "matrix unit"
# basis given by { |A><A|, |A><B|, etc }.  For use in GST, we want a *real*
# matrix in either the Gell-Mann or Pauli-product basis.  Here we choose the
# Pauli-product basis, which is typically more intuitive when working with 2 qubits.
mySuperOp_ppbasis = pygsti.change_basis(mySuperOp_stdbasis, "std", "pp")

#The resulting superoperator is exactly what goes into the Model object,
# which can be set using dictionary syntax.  This names the gate 'Gtq'.
target_model['Gtq'] = mySuperOp_ppbasis
```

That's the whole model.

```{code-cell} ipython3
print(target_model)
```

### Running GST on it

To run 2-qubit GST against a custom model you would ideally generate fiducials and germs specifically for it. In this case the 1-qubit gates match the standard 2Q model packs, and the fiducial sequences for those packs contain only 1-qubit gates, so you can reuse a standard fiducial set such as `smq2Q_XYICNOT`'s. Germs are the harder part. They should be computed, but in practice you can often take the germ set of a standard model and substitute your custom gate for its 2-qubit gate; check whether the resulting set is amplificationally complete before committing to a full germ-selection run. With fiducials and germs in hand, 2-qubit GST proceeds exactly as it does for a built-in 2Q model.

## Next steps

Look at [implicit models](MultiQubitModels), which trade the dictionary-of-matrices interface for something that scales to more qubits. The [operators tutorial](../../advanced/models/Operators#choosing-types-when-you-build-a-model) and the [model noise tutorial](ModelNoise.md) both apply to explicit and implicit models alike.

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

# 2-Qubit GST Model

```{attention} Should this be here or in a new Model or Examples section?```

While pyGSTi is able to support several common types of 2-qubit gates, the space of all possible 2-qubit gates is so large that some users will need to construct their own particular 2-qubit gate "from scratch".  In this example, we look at how to construct a 2-qubit model manually.  We'll use `pygsti.construction.create_explicit_model` to construct the single-qubit gates and we'll create a "non-standard" 2-qubit gate by specifying the unitary operation it performs as a $4\times 4$ matrix. 

To perform GST on such a model, one may need to compute new sets of fiducials and/or germ sequences.  Once these are obtained, 2-qubit GST can be run just as in the example which uses one of pyGSTi's built-in 2Q models.

```{code-cell} ipython3
import pygsti
```

## Step 1: Create a model with only single-qubit gates
Since the space of single-qubit gates is relatively small, we'll assume that the single-qubit gates in our model are able to be specified using `pygsti.construction.create_explicit_model`.  So we'll start by construct a `Model` object containing all but the two-qubit gate(s) using `create_explicit_model`.

```{code-cell} ipython3
target_model = pygsti.models.modelconstruction.create_explicit_model_from_expressions( 
            [('Q0','Q1')],['Gii','Gix','Giy','Gxi','Gyi'], 
            [ "I(Q0):I(Q1)", "X(pi/2,Q1)", "Y(pi/2,Q1)", "X(pi/2,Q0)", "Y(pi/2,Q0)" ],
            effect_labels=['00','01','10','11'], effect_expressions=["0","1","2","3"])
```

There are lots of arguments to this function, so let's review what they mean:
- `[('Q0','Q1')]` = interpret this 4-d space as that of two qubits 'Q0', and 'Q1' (note these labels *must* begin with 'Q'!)
- `"Gix"` = operation label; can be anything that begins with 'G' and is followed by lowercase letters
- `"X(pi/2,Q1)"` = pi/2 single-qubit x-rotation gate on the qubit labeled Q1
- `"rho0"` = prep label; can be anything that begins with "rho"
- `'10'` = designates a POVM effect label whose corresponding vector is given by the `effectExpressions` argument.
- `"2"` = a prep or effect expression indicating a projection/preparation of the 3rd (b/c 0-based) computational basis element

You can also explicity add identity operations, e.g. `"I(Q0)"`, to the rotation gates to get the same model (see `mdl_targetB` below),  and this same syntax can be used for non-entangling 2-qubit gates, e.g. `"X(pi/2,Q0):X(pi/2,Q1)"`.

```{code-cell} ipython3
mdl_targetB = pygsti.models.modelconstruction.create_explicit_model_from_expressions( 
            [('Q0','Q1')],['Gii','Gix','Giy','Gxi','Gyi','Gcnot'], 
            [ "I(Q0):I(Q1)", "I(Q0):X(pi/2,Q1)", "I(Q0):Y(pi/2,Q1)", "X(pi/2,Q0):I(Q1)", "Y(pi/2,Q0):I(Q1)", "CNOT(Q0,Q1)" ],
            effect_labels=['00','01','10','11'], effect_expressions=["0","1","2","3"])
assert(abs(target_model.frobeniusdist(mdl_targetB)) < 1e-6)
```

If our 2-qubit gate happens to be one that *can* be specified using `create_explicit_model` then we can just use it to construct the entire `Model` and be done.  Currently, `create_explicit_model` can create any controlled $X$, $Y$, or $Z$ rotation using `CX`, `CY` and `CZ`, as well as the standard `CNOT` and `CPHASE` gates.  Below we demonstrate creation with the CNOT gate.  The resulting `Model` is then identical to `pygsti.modelpacks.legacy.std2Q_XYCNOT.target_model()`.

```{code-cell} ipython3
mdl_withCNOT = pygsti.models.modelconstruction.create_explicit_model_from_expressions( 
            [('Q0','Q1')],['Gii','Gix','Giy','Gxi','Gyi','Gcnot'], 
            [ "I(Q0):I(Q1)", "I(Q0):X(pi/2,Q1)", "I(Q0):Y(pi/2,Q1)", "X(pi/2,Q0):I(Q1)", "Y(pi/2,Q0):I(Q1)", "CNOT(Q0,Q1)" ],
            effect_labels=['00','01','10','11'], effect_expressions=["0","1","2","3"])

#Note this is the same model as one of pyGSTi's standard models:
from pygsti.modelpacks.legacy import std2Q_XYICNOT
assert(abs(mdl_withCNOT.frobeniusdist(std2Q_XYICNOT.target_model())) < 1e-6)
```

Thus, since our `target_model` just contains the 1-qubit gates of `std2Q_XYCNOT.target_model`, we could also create has all the 1-qubit gates are the same a third way to obtain `target_model` is to just remove `Gcnot` from the standard model:

```{code-cell} ipython3
mdl_targetC = std2Q_XYICNOT.target_model()
del mdl_targetC.operations['Gcnot']
assert(abs(target_model.frobeniusdist(mdl_targetC)) < 1e-6)
```

## Step 2: Create a custom 2-qubit gate
We're assuming that `create_explicit_model` can't make the 2-qubit gate we want, so we'll need to create our own. Below we demonstrate how to create a 2-qubit gate from a given unitary which acts on the 2-qubit, 4-dimensional, state space.

```{code-cell} ipython3
import numpy as np

#Unitary in acting on the state-space { |A>, |B>, |C>, |D> } == { |00>, |01>, |10>, |11> }.
# This unitary rotates the second qubit by pi/2 in either the (+) or (-) direction based on 
# the state of the first qubit.
myUnitary = 1./np.sqrt(2) * np.array([[1,-1j,0,0],
                                      [-1j,1,0,0],
                                      [0,0,1,1j],
                                      [0,0,1j,1]])

#Convert this unitary into a "superoperator", which acts on the 
# space of vectorized density matrices instead of just the state space.
# These superoperators are what GST calls "gates".
mySuperOp_stdbasis = pygsti.unitary_to_std_process_mx(myUnitary)

#After the call to unitary_to_process_mx, the superoperator is a complex matrix
# in the "standard" or "matrix unit" basis given by { |A><A|, |A><B|, etc }.
# For use in GST, we want to work with a *real* matrix in either the 
# Gell-Mann or Pauli-product basis. Here we choose the Pauli-product basis,
# which is typically more intuitive when working with 2 qubits.
mySuperOp_ppbasis = pygsti.change_basis(mySuperOp_stdbasis, "std", "pp")

#The resulting superoperator in the Pauli-product basis is exactly
# what goes into the Model object, which can be set using 
# dictionary syntax.  The line below names our two-qubit gate 'Gtq'
target_model['Gtq'] = mySuperOp_ppbasis
```

## That's it!
We're done creating our 2-qubit model, `target_model`, printed below.  To run 2-qubit GST with this model we ideally would generate fiducials and germs specifically for it.  Actually, since the 1-qubit gates are the same as other standard 2Q models, and the fiducial seqeuences for these standard sets only contain 1-qubit gates, we can just use the fiducial sets from a standard set (e.g. `std2Q_XYCNOT`).  The germs should be computed, though typically you can get away with just using the germ set of a standard model and replacing it's 2-qubit gate with the custom one - so it can be worth checking whether such a set is complete before running a full germ-selection procedure.

```{code-cell} ipython3
print(target_model)
```

```{code-cell} ipython3

```

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

# Continuously parameterized gates
This tutorial demonstrates how gate labels can be given "arguments".  Let's get started by some usual imports:

```{code-cell} ipython3
import numpy as np

import pygsti
from pygsti.baseobjs import Label
from pygsti.circuits import Circuit
from pygsti.modelmembers import operations as op
```

**Arguments** are just tags that get associated with a gate label, and can include continuous parameters such as an angle of rotation.  Arguments are held distinct from the "state space labels" (usually equivalent to "qubit labels") associated with a gate, which typically specify the *target* qubits for a gate, and thereby determing where gate is displayed when drawing a circuit (on which qubit lines).

Here are some ways you can create labels containing arguments.  A common theme is that arguments are indicated by a preceding semicolon (;):

```{code-cell} ipython3
#Different ways of creating a gate label that contains a single argument
l = Label('Ga',args=(1.4,1.2))
l2 = Label(('Ga',';1.4',';1.2')) #Note: in this case the arguments are *strings*, not floats
l3 = Label(('Ga',';',1.4,';',1.2))
```

You can use the more compact preceded-with-semicolon notation when construting `Circuit`s from tuples or strings:

```{code-cell} ipython3
# standard 1Q circuit, just for reference
c = Circuit( ('Gx','Gy') )
print(c)

# 1Q circuit with explicit qubit label
c = Circuit( [('Gx',0),('Gy',0)] )
print(c)

# adding arguments
c = Circuit( [('Gx',0,';1.4'),('Gy',';1.2',0)] )
print(c)

#Or like this:
c = Circuit("Gx;1.4:0*Gy;1.2:0")
print(c)
```

Now that we know how to make circuits containing labels with arguments, let's cover how you connect these labels with gate operations.  A gate label without any arguments corresponds to an "operator" object in pyGSTi; a label with arguments typically corresponds to an object *factory* object.  A factory, as its name implies, creates operator objects "on demand" using a supplied set of arguments which are taken from the label in a circuit.  The main function in an `OpFactory` object is `create_object`, which accepts a tuple of arguments as `args` and is expected to return a gate object.

Here's an example of a simple factory that expects a single argument (see the assert statements), and so would correspond to a continuously-parameterized-gate with a single continuous parameter.  In this case, our factory generates a X-rotation gate whose rotation angle is given by the one and only argument.  We return this as a `StaticArbitraryOp` because we're not worrying about how the gate is parameterized for now (parameters are the things that GST twiddles with, and are distinct from arguments, which are fixed by the circuit).

```{code-cell} ipython3
class XRotationOpFactory(op.OpFactory):
    def __init__(self):
        op.OpFactory.__init__(self, state_space=1, evotype="densitymx")
        
    def create_object(self, args=None, sslbls=None):
        # Note: don't worry about sslbls (unused) -- this argument allow factories to create different operations on different target qubits
        assert(len(args) == 1)
        theta = float(args[0])/2.0  #note we convert to float b/c the args can be strings depending on how the circuit is specified
        b = 2*np.cos(theta)*np.sin(theta)
        c = np.cos(theta)**2 - np.sin(theta)**2
        superop = np.array([[1,   0,   0,   0],
                            [0,   1,   0,   0],
                            [0,   0,   c,  -b],
                            [0,   0,   b,   c]],'d')
        return op.StaticArbitraryOp(superop)
```

Next, we build a model that contains an instance of `XRotationFactory` that will be invoked when a circuit contains a `"Ga"` gate.  So far, only *implicit* models are allowed to contain factories, so we'll create a `LocalNoiseModel`  (see the [implicit model tutorial](../ImplicitModel.ipynb)) for a single qubit with the standard X, and Y gates, and then add our factory:

```{code-cell} ipython3
pspec = pygsti.processors.QubitProcessorSpec(1, ['Gx', 'Gy'])
mdl = pygsti.models.create_crosstalk_free_model(pspec)

Ga_factory = XRotationOpFactory()
mdl.factories['layers'][('Ga',0)] = Ga_factory
```

The resulting model is capable of computing outcome probabilities for circuits containing `Gx`, `Gy`, *or* `Ga;<ANGLE>` on any of the qubits, where ANGLE is a floating point angle in radians that will get passed to the `create_object` function of our `XRotationFactory` instance.  Let's try this out (note that we need to specify the qubit label, 0, because local noise models create gates using multi-qubit conventions):

```{code-cell} ipython3
c1 = pygsti.circuits.Circuit('Gx:0*Ga;3.1:0*Gx:0')
print(c1)
mdl.probabilities(c1)
```

The above is readily extensible to systems with more qubits.  The only nontrivial addition is that our factory, which creates 1-qubit gates, must be "embedded" within a larger collection of qubits to result in a n-qubit-gate factory.  This step is easily accomplished using the builtin `EmbeddedOpFactory` object, which takes a tuple of all the qubits, e.g. `(0,1)` and a tuple of the subset of qubits therein to embed into, e.g. `(0,)`.  This is illustrated below for the 2-qubit case, along with a demonstration of how a more complex 2-qubit circuit can be simulated:

```{code-cell} ipython3
pspec2 = pygsti.processors.QubitProcessorSpec(2, ('Gx','Gy','Gcnot'), geometry='line')
mdl2 = pygsti.models.create_crosstalk_free_model(pspec2)

Ga_factory = XRotationOpFactory()
mdl2.factories['layers'][('Ga',0)] = op.EmbeddedOpFactory((0,1),(0,),Ga_factory)
mdl2.factories['layers'][('Ga',1)] = op.EmbeddedOpFactory((0,1),(1,),Ga_factory)

c2 = pygsti.circuits.Circuit("[Gx:0Ga;1.2:1][Ga;1.4:0][Gcnot:0:1][Gy:0Ga;0.3:1]" )
print(c2)

mdl2.probabilities(c2)
```

```{code-cell} ipython3

```

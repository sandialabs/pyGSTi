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

# Circuit Simulation
This tutorial demonstrates how to compute (simulate) the outcome probabilities of circuits in pyGSTi.  There are currently two basic ways to to this - but constructing and simulating a `Circuit` object, or by constructing and propagating a state.

## Method 1: `Circuit` simulation
This is the primary way circuit simulation is done in pyGSTi.  `Model` objects are statistical models that predict the outcome probabilities of events, and (at least for all current model types) "events" are circuits, described by `Circuit` objects.  Thus, the three steps to simulating a circuit using this approach are:

1. create a `Model`
2. create a `Circuit`
3. call `model.probabilities(circuit)`

Building models and circuits (steps 1 and 2) are largely covered in other tutorials (see the [essential objects tutorial](../01-EssentialObjects.ipynb), [circuits tutorial](../objects/Circuit.ipynb), and [explicit-op model](../objects/ExplicitModel.ipynb) and [implicit-op model](../objects/ImplicitModel.ipynb) tutorials).  This section focuses on step 3 and `Model` options which impact the way in which a model computes probabilities.  This approach to circuit simulation is most convenient when you have a large number of circuits which are known (and fixed) beforehand.

Let's begin with a simple example, essentially the same as the one in the [using-essential-objects tutorial](../02-Using-Essential-Objects.ipynb):

```{code-cell} ipython3
import pygsti
mdl = pygsti.models.create_explicit_model_from_expressions((0,1),
            [(),      ('Gxpi2',0),    ('Gypi2',0),    ('Gxpi2',1),    ('Gypi2',1),    ('Gcnot',0,1)],
            ["I(0,1)","X(pi/2,0)", "Y(pi/2,0)", "X(pi/2,1)", "Y(pi/2,1)", "CNOT(0,1)"]) 
c = pygsti.circuits.Circuit([('Gxpi2',0),('Gcnot',0,1),('Gypi2',1)] , line_labels=[0,1])
print(c)
mdl.probabilities(c) # Compute the outcome probabilities of circuit `c`
```

This example builds an `ExplicitOpModel` (best for 1-2 qubits) on 2 qubits with $X(\pi/2)$ and $Y(\pi/2)$ rotation gates on each qubit and a CNOT gate between them.  This model is able to simulate any circuit *layer* (a.k.a. "time-step" or "clock-cycle") that contains any *one* of these gates (this is what it means to be an explicit-op model: the operation for every simulate-able circuit layer must be explicitly supplied to the `Model`).  For example, this model cannot simulate a circuit layer where two `Gxpi2` gates occur in parallel:

```{code-cell} ipython3
c2 = pygsti.circuits.Circuit([ [('Gxpi2',0), ('Gxpi2',1)],('Gcnot',0,1) ] , line_labels=[0,1])
print(c2)
try:
    mdl.probabilities(c2)
except KeyError as e:
    print("KEY ERROR (can't simulate this layer): " + str(e))
```

As is detailed in the [implicit-op model tutorial](../objects/ImplicitModel.ipynb), an "implicit-operation" model *is* able to implicitly create layer operations from constituent gates, and thus perform the simulation of `c2`:

```{code-cell} ipython3
pspec = pygsti.processors.QubitProcessorSpec(2, ('Gxpi2', 'Gypi2', 'Gcnot'), geometry='line')
implicit_mdl = pygsti.models.create_crosstalk_free_model(pspec)
print(c2)
implicit_mdl.probabilities(c2)
```

## Method 2: state propagation
In this method of circuit simulation, a state object (a `SPAMVec` in pyGSTi) is propagated circuit-layer by circuit-layer.  This method is convenient when a there are few (or just one!) circuit that involves substantial classical logic or needs to be probed at various points in time.  It is slower to simulate circuits in this way, as it requires calls more calls between pyGSTi's Python and C routines than method 1 does.

The two cells below show how to perform the same two circuits above using the state-propagation method.

```{code-cell} ipython3
#Simulating circuit `c` above using `mdl`: [('Gxpi2',0),('Gcnot',0,1),('Gypi2',1)]
rho = mdl['rho0']
rho = mdl[('Gxpi2',0)].acton(rho)
rho = mdl[('Gcnot',0,1)].acton(rho)
rho = mdl[('Gypi2',1)].acton(rho)
probs = mdl['Mdefault'].acton(rho)
print(probs)
```

Note that, especially for implicit models, the interface is a bit clunky.  <font style="color:red">Simulation by state propagation is a work in progress in pyGSTi, and users should expect that this interface may change (improve!) in the future</font>.

```{code-cell} ipython3
#Simulating circuit `c2` above using `implicit_mdl`: [ [('Gxpi2',0), ('Gxpi2',1)], ('Gcnot',0,1) ]
from pygsti.baseobjs import Label as L
rho = implicit_mdl.prep_blks['layers'][L('rho0')]
rho = implicit_mdl.operation_blks['layers'][ L('Gxpi2',0) ].acton(rho)
rho = implicit_mdl.operation_blks['layers'][ L('Gxpi2',1) ].acton(rho)
rho = implicit_mdl.operation_blks['layers'][ L('Gcnot',(0,1)) ].acton(rho)
probs = implicit_mdl.povm_blks['layers']['Mdefault'].acton(rho)
print(probs)
```

## Method 3: hybrid
(an addition planned in future releases of pyGSTi)

+++

## Forward-simulation types

PyGSTi refers to the process of computing circuit-outcome probabilities as *forward simulation*, and there are several methods of forward simulation currently available.  The default method for 1- and 2-qubit models multiplies together dense process matrices, and is named `"matrix"` (because operations are *matrices*). The default method for 3+ qubit models performs sparse matrix-vector products, and is named `"map"` (because operations are abstract *maps*).  A `Model` is constructed for a single type of forward simulation, and it stores this within its `.simtype` member.  For more information on using different types of forward simulation see the [forward simulation types tutorial](algorithms/advanced/ForwardSimulationTypes.ipynb).

Here are some examples showing which method is being used and how to switch between them.  Usually you don't need to worry about the forward-simulation type, but in the future pyGSTi may have more options for specialized purposes.

```{code-cell} ipython3
c3 = pygsti.circuits.Circuit([('Gxpi2',0),('Gcnot',0,1)] , line_labels=[0,1])

explicit_mdl = pygsti.models.create_explicit_model_from_expressions((0,1),
            [(),      ('Gxpi2',0),    ('Gypi2',0),    ('Gxpi2',1),    ('Gypi2',1),    ('Gcnot',0,1)],
            ["I(0,1)","X(pi/2,0)", "Y(pi/2,0)", "X(pi/2,1)", "Y(pi/2,1)", "CNOT(0,1)"]) 

print("2Q explicit_mdl will simulate probabilities using the '%s' forward-simulation method." % explicit_mdl.sim)
explicit_mdl.probabilities(c3)
```

```{code-cell} ipython3
pspec = pygsti.processors.QubitProcessorSpec(3, ('Gxpi2', 'Gypi2', 'Gcnot'), geometry='line')
implicit_mdl = pygsti.models.create_crosstalk_free_model(pspec)
print("3Q implicit_mdl will simulate probabilities using the '%s' forward-simulation method." % implicit_mdl.sim)
implicit_mdl.probabilities(c)
```

```{code-cell} ipython3
implicit_mdl.sim = 'matrix'
print("3Q implicit_mdl will simulate probabilities using the '%s' forward-simulation method." % implicit_mdl.sim)
implicit_mdl.probabilities(c)
```

```{code-cell} ipython3

```

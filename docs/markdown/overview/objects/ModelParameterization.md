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

# Model Parameterization
The fundamental role of Model objects in pyGSTi is to simulate circuits, that is, to map circuits to outcome probability distributions.  This mapping is *parameterized* by some set of real-valued parameters, meaning that the mapping between circuits and outcome distribution depends on the values of a `Model`'s parameters.  Model objects have a `num_params` attribute holding its parameter count, and `to_vector` and `from_vector` methods which get or set a model's vector of parameters.

`ModelMember` objects such as state prepations, operations, and measurements (POVMs) are also parameterized, and similarly possess a `num_params` attribute and `to_vector` and `from_vector` methods.  For models that hold member objects to implement their operations (e.g., both explicit and implicit models), the model's parameterization the result of combining the parameterizations of all its members.

In explicit models, the parameterization is properly viewed as a mapping between the model's parameter space and the space of $d^2 \times d^2$ operation matrices and length-$d^2$ SPAM vectors.  A `Model`'s contents always correspond to a valid set of parameters, which can be obtained by its `to_vector` method, and can always be initialized from a vector of parameters via its `from_vector` method.  The number of parameters (obtained via `num_params`) is independent (and need not equal!) the total number of gate-matrix and SPAM-vector elements comprising the `Model`.  For example, in a "TP-parameterized" model, the first row of each operation matrix is fixed at `[1,0,...0]`, regardless to what the `Model`'s underlying parameters are.  One of pyGSTi's primary capabilities is model optimization: the optimization of a fit function (often the log-likelihood) over the parameter space of an initial `Model` (often times the "target" model).  Thus, specifying a model's parameterization specifies the constraints under which the model is optimized, or equivalently the space of possible circuit-to-outcome-distribution mappings that are searched for a best-fit estimate.  

In the simplest case, each gate and SPAM vector within a `ExplicitOpModel` have independent paramterizations, so that each `pygsti.modelmembers.ModelMember`-derived object has its own separate parameters accessed by its `to_vector` and `from_vector` methods.  The `ExplictOpModel`'s parameter vector, in this case, is just the concatenation of the parameter vectors of its contents, usually in the order: 1) state preparation vectors, 2) measurement vectors, 3) gates.

## Operation types

Operations on quantum states exist within the `pygsti.modelmembers.operations` subpackage.  Most of the classes therein represent a unique combination of a:

a. category of operation that can be represented, and
b. parameterization of that category of operations.

For example, the `FullArbitraryOp` class can represent an arbitrary (Markovian) operation, and "fully" parameterizes the operation by exposing every element of the operation's dense process matrix as a parameter.  The `StaticCliffordOp` class can only represent Clifford operations, and is "static", meaning it exposes no parameters and so cannot be changed in an optimization.  Here are brief descriptions of several of the most commonly used operation types:

- The `FullArbitraryOp` class represents a arbitrary process matrix which has a parameter for every element, and thus optimizations using this gate class allow the operation matrix to be varied completely.
- The `StaticArbitraryOp` class also represents an arbitrary process matrix but has no parameters, and thus is not optimized at all.
- The `FullTPOp` class represents a process matrix whose first row must be `[1,0,...0]`.  This corresponds to a trace-preserving (TP) gate in the Gell-Mann and Pauli-product bases.  Each element in the remaining rows is a separate parameter, similar to a fully parameterized gate.  Optimizations using this gate type are used to constrain the estimated gate to being trace preserving.
- The `LindbladErrorgen` class defines an error generator that takes a particular Lindblad form.  This class is fairly flexible, but is predominantly used to constrain optimizations to the set of infinitesimally-generated CPTP maps.  To produce a gate or layer operation, error generators must be exponentiated using the `ExpErrorgenOp` class.

Similarly, there classes represnting quantum states in `pygsti.modelmembers.states` and those for POVMs and POVM effects in `pygsti.modelmembers.povms`.  Many of these classes run parallel to those for operations.  For example, there exist `FullState` and `TPState` classes, the latter which fixes its first element to $\sqrt{d}$, where $d^2$ is the vector length, as this is the appropriate value for a unit-trace state preparation.

There are other operation types that simply combine or modify other operations.  These types don't correspond to a particular category of operations or parameterization, they simply inherit these from the operations they act upon.  The are:

- The `ComposedOp` class combines zero or more other operations by acting them one after the other.  This has the effect of producing a map whose process matrix would be the product of the process matrices of the factor operations. 
- The `ComposedErrorgen` class combines zero or more error generators by effectively summing them together.
- The `EmbeddedOp` class embeds a lower-dimensional operation (e.g. a 1-qubit gate) into a higer-dimensional space (e.g. a 3-qubit space).
- The `EmbeddedErrorgen` class embeds a lower-dimensional error generator into a higher-dimensional space.
- The `ExpErrorgenOp` class exponentiates an error generator operation, making it into a map on quantum states.
- The `RepeatedOp` class simply repeats a single operation $k$ times.

These operations act as critical building blocks when constructing complex gate and circuit-layer operations, especially on a many-qubit spaces.  Again, there are analogous classes for states, POVMs, etc., within the other sub-packages beneath `pygsti.modelmembers`.


## Specifying operation types when creating models

Many of the model construction functions take arguments dictating the type of modelmember objects to create.  As described above, by changing the type of a gate you select how that gate is represented (e.g. Clifford gates can be represented more efficiently than arbitrary gates) and how it is parameterized.  This in turn dictates how the overall model is paramterized.

For a brief overview of the available options, here is an incomplete list of parameterization arguments and their associated `pygsti.modelmember` class.  Most types start with either `"full"` or `"static"` - these indicate whether the model members have parameters or not, respectively. Parameterizations without a prefix are "full" by default. See the related [ForwardSimulation tutorial](../algorithms/advanced/ForwardSimulationTypes.ipynb) for how each parameterization relates to the allowed types of forward simulation in PyGSTi.

- `gate_type` for `modelmember.operations`:
  - `"static"` $\rightarrow$ `StaticArbitraryOp`
  - `"full"` $\rightarrow$ `FullArbitraryOp`
  - `"static standard"` $\rightarrow$ `StaticStandardOp`
  - `"static clifford"` $\rightarrow$ `StaticCliffordOp`
  - `"static unitary"` $\rightarrow$ `StaticUnitaryOp`
  - `"full unitary"` $\rightarrow$ `FullUnitaryOp`
  - `"full TP"` $\rightarrow$ `FullTPOp`
  - `"CPTP"`, `"H+S"`, etc. $\rightarrow$ `ExpErrorgenOp` + `LindbladErrorgen`


- `prep_type` for `modelmember.states`:
  - `"computational"` $\rightarrow$ `ComputationalBasisState`
  - `"static pure"` $\rightarrow$ `StaticPureState`
  - `"full pure"` $\rightarrow$ `FullPureState`
  - `"static"` $\rightarrow$ `StaticState`
  - `"full"` $\rightarrow$ `FullState`
  - `"full TP"` $\rightarrow$ `TPState`


- `povm_type` for `modelmember.povms`:
  - `"computational"` $\rightarrow$ `ComputationalBasisPOVM`
  - `"static pure"` $\rightarrow$ `UnconstrainedPOVM` + `StaticPureEffect`
  - `"full pure"` $\rightarrow$ `UnconstrainedPOVM` + `FullPureEffect`
  - `"static"` $\rightarrow$ `UnconstrainedPOVM` + `StaticEffect`
  - `"full"` $\rightarrow$ `UnconstrainedPOVM` + `FullEffect`
  - `"full TP"` $\rightarrow$ `TPPOVM`
  
For convenience, the `prep_type` and `povm_type` arguments also accept `"auto"`, which will try to set the parameterization based on the given `gate_type`. An incomplete list of this `gate_type` $\rightarrow$ `prep_type` / `povm_type` mapping is:

- `"auto"`, `"static standard"`, `"static clifford"` $\rightarrow$ `"computational"`
- `"unitary"` $\rightarrow$ `"pure"`
- All others map directly

### Explicit Models
We now illustrate how one may specify the type of paramterization in `create_explicit_model`, and change the object types of all of a `ExplicitOpModel`'s contents using its `set_all_parameterizaions` method.  The `create_explicit_model` function builds (layer) operations that are compositions of the ideal operations and added noise (see the [model noise tutorial](ModelNoise.ipynb)).  By setting `ideal_gate_type` and similar arguments, the object type used for the initial "ideal" part of the operations is decided.

```{code-cell} ipython3
import pygsti
from pygsti.processors import QubitProcessorSpec
from pygsti.models import modelconstruction as mc
```

```{code-cell} ipython3
pspec = QubitProcessorSpec(1, ['Gi', 'Gxpi2', 'Gypi2'])  # simple single qubit processor
model = mc.create_explicit_model(pspec)
model.print_modelmembers()
print("%d parameters" % model.num_params)
```

By default, an explicit model creates static (zero parameter) operations of types `StaticUnitaryOp`.  If we specify an `ideal_gate_type` we can change this:

```{code-cell} ipython3
model = mc.create_explicit_model(pspec, ideal_gate_type="full TP")
model.print_modelmembers()
print("%d parameters" % model.num_params)
```

Switching the parameterizatio to "CPTP" gates changes the gate type accordingly:

```{code-cell} ipython3
model.set_all_parameterizations('CPTP')
model.print_modelmembers()
print("%d parameters" % model.num_params)
```

To alter an *individual* gate or SPAM vector's parameterization, one can simply construct a replacement object of the desired type and assign it to the `Model`.

```{code-cell} ipython3
# Turning ComposedOp into a dense matrix for conversion into a dense FullTPOp
newOp = pygsti.modelmembers.operations.FullTPOp(model[('Gi', 0)].to_dense())
model['Gi'] = newOp
print("model['Gi'] =",model['Gi'])
```

**NOTE:** When a `LinearOperator` or `SPAMVec`-derived object is assigned as an element of an `ExplicitOpModel` (as above), the object *replaces* any existing object with the given key.  However, if any other type of object is assigned to an `ExplicitOpModel` element, an attempt is made to initialize or update the existing existing gate using the assigned data (using its `set_matrix` function internally).  For example:

```{code-cell} ipython3
import numpy as np
numpy_array = np.array( [[1, 0, 0, 0],
                         [0, 0.5, 0, 0],
                         [0, 0, 0.5, 0],
                         [0, 0, 0, 0.5]], 'd')
model['Gi'] = numpy_array # after assignment with a numpy array...
print("model['Gi'] =",model['Gi']) # this is STILL a FullTPOp object

#If you try to assign a gate to something that is either invalid or it doesn't know how
# to deal with, it will raise an exception
invalid_TP_array = np.array( [[2, 1, 3, 0],
                              [0, 0.5, 0, 0],
                              [0, 0, 0.5, 0],
                              [0, 0, 0, 0.5]], 'd')
try:
    model['Gi'] = invalid_TP_array
except ValueError as e:
    print("ERROR!! " + str(e))
```

### Implicit models

The story is similar with implicit models.  Operations are built as compositions of ideal operations and noise, and by specifying the `ideal_gate_type` and similar arguments, you can set what type of ideal operation is created.  Below we show some examples with a `LocalNoiseModel`.  Let's start with the default static operation type:

```{code-cell} ipython3
mdl_locnoise = pygsti.models.create_crosstalk_free_model(pspec)
mdl_locnoise.print_modelmembers()
```

Suppose we'd like to modify the gate operations.  Then we should make a model with `ideal_gate_type="full"`, so the operations are `FullArbitraryOp` objects:

```{code-cell} ipython3
mdl_locnoise = pygsti.models.create_crosstalk_free_model(pspec, ideal_gate_type='full')
mdl_locnoise.print_modelmembers()
```

These can now be modified by matrix assignment, since their parameters allow them to take on any other process matrix.  Let's set the process matrix (more accurately, this is the Pauli-transfer-matrix of the gate) of `"Gxpi"` to include some depolarization:

```{code-cell} ipython3
mdl_locnoise.operation_blks['gates']['Gxpi2'] = np.array([[1,   0,   0,   0],
                                                          [0, 0.9,   0,   0],
                                                          [0,   0,-0.9,   0],
                                                          [0,   0,   0,-0.9]],'d')
```

In `CloudNoiseModel` objects, all of the model's parameterization is inherited from its noise operations, and so there are no `ideal_gate_type` and similar arguments in `create_cloud_crosstalk_model`.  All of the ideal operations are always static (have no parameters) in a cloud noise model.  See the [tutorial on model noise](ModelNoise.ipynb) to see how the types of the noise objects can be set.

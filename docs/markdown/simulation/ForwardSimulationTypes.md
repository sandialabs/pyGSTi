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

# Simulation Types (advanced)
The ability prepare, propagate, and measure quantum states in simulations is a core capability of pyGSTi.  We call the calculation of circuit outcome probabilities *forward simulation*, and this type of calculation lies at the heart of gate set tomography and other characterization protocols which need to compare a model's predictions with actual data.  As you're probably already encountered through other tutorials, `Model` objects have a `.probabilities(circuit)` method that performs forward simulation (computes the outcome probabilities of `circuit`).  

What is not so apparent is that there can be several different types of computational "engines" under the hood of a `Model` that do the heavy lifting within a call to `probs`.  **Understanding the different types of forward simulation engines in pyGSTi is the topic of this tutorial.**

First, let's lay down a bit of background.  A `Model` contains a `.evotype` attribute that describes what type of underlying *state representation* is being stored and propagated by the model - the *evolution type*.  Allowed values are:
- `"densitymx"`: a *mixed* state is propagated as a vectorized density matrix (of length $4^n$ for $n$ qubits) in a Hermitian basis (so its elements are *real*).  If an operation is represented as a dense (real) matrix, its shape is $4^n \times 4^n$.  This is the default evolution type in pyGSTi.
- `"statevec"`: a *pure* state is propagated as a complex state vector (of length $2^n$ for $n$ qubits).  Dense operations representations are $2^n \times 2^n$ unitary matrices in this case.  Because of the lower dimensionality, state-vector propagation is faster than density-matrix propagation, and is therefore preferable when all preparations and POVM effects are pure states and projections, and all operations are unitary.
- `"stabilizer"`: a *stabilizer* state is propagated by keeping track of its stabilizer (and anti-stabilizer) group generators.  This only requires memory that scales with $n$ for $n$ qubits, and so this state representation offers significantly more efficient simulation than the two aforementioned evolution types.  The caveat is preparations must prepare a stabilizer state, POVMs must measure in the computational $z$-basis, and *all operations must be Clifford elements*, mapping Pauli group elements to Pauli group elements.
- `"[densitymx|statevec|stabilizer]_slow"`: a pure-Python "slow" version of each of the above evolution types exists so that building pyGSTi's Cython extensions is optional.  If you're unable to build the Cython extensions to any reason, any of these `_slow` evolution types can still be used.  As their name implies, these evotypes can result in significantly longer run times when compared with their faster Cython-extension versions.
- `"default"`: whatever value is in `pygsti.evotypes.Evotype.default_evotype`.  When pygsti is imported, this is set to `densitymx` if the Cython extensions are built and `densitymx_slow` otherwise.  Users may modify this value to set a different default, which can be convenient since many objects that require an evolution type default to `"default"`.

Every operator (state preparation, POVM, gate/layer operation) within a `Model` also has an evolution type, and it must match the parent `Model`'s `.evotype`.  (You usually don't need to worry about this; `Model` objects are created with a given `evotype` and their contained elements are created to match that type.)  

Related to their `evotype`, `Model` objects also have a `sim` attribute.  This attribute is the forward simulation engine that is used when the model is asked to compute circuit probabilities.  It is an instance of a subclass of `pygsti.forwardsims.ForwardSimulator`.  There are several different types of forward simulators that the `sim` attribute can be set to, but each type is only compatible with certain evolution types (`evotype` value):
- `MapForwardSimulator` or `"map"`: propagate a `"densitymx"`, `"statevec"`, `"stabilizer"` state by repeatedly acting with circuit layer operators, treated as maps from an input to an output state.  When there is a dense layer-operation matrix for each circuit layer, this engine repeatedly performs matrix-vector products (one per layer, between the operation matrix and column-vector state) and finally contracts the result with each row-vector POVM effect to get outcome probabilities.  This is the most straightforward and intuitive of the forward simulation engines, so much so that you may be thinking "what else would you do?".  Keep reading :)
- `MatrixForwardSimulator` or `"matrix"`: propagate a `"densitymx"` or `"statevec"` state by composing a dense operation matrix for the entire circuit and then applying that "circuit map" to the input state.  Each circuit layer *is represented as a dense matrix* when using this forward simulation type, as it multiplies together the matrices of each layer (in reverse order!) to get a single "circuit matrix* and then contracts this matrix between the (column vector) state preparation and each of the (row vector) POVM effects (and norm-squaring the result when the evolution type is `"statevec"`) to get outcome probabilities.  At first glance this seems like a very inefficient way to compute probabilities since it performs matrix-matrix instead of matrix-vector multiplications, and it is true that this method should not be used with many-qubit models.  However, for lower dimensional Hilbert spaces (1-2 qubits in practice) the ability to cache and reuse intermediate results can make this forward simulator faster than the `"map"` type when the outcome probabilities of *many* circuits are needed at once (as is the case in gate set tomography, for instance).  Indeed, it was specifically for 1- and 2-qubit GST that this method was implemented.  Apart from this case, `"map"` should be used (thought using `"matrix"` instead on 1 or 2 qubit systems often achieves similar run times because of other technical implementation factors).
- `TermForwardSimulator`: computes circuit outcome probabilites by evaluating a truncated path integral assembled by Taylor-expanding each circuit operation and keeping terms that are below some Taylor-term order or that have a weight below some threshold.  Each path is evaluated by propagating a pure state under unitary actions, and so this forward simulator works with `"statevec"` and `"stabilizer"` evotypes.
- `CHPForwardSimulator` or `"chp"`: simulates Clifford circuits using Scott Aaronson's CHP program.

Thus, by setting a model's `evotype` and `sim` you can specify how circuit-probability-computation is implemented.  But when and how do you set these values?  The `evotype` of a `Model` is almost always set for good when the object is created.  The simulator *can* be changed assigning a new forward simulator object to the `sim` property of a `Model`, but it's usually set to an appropriate value at object-creation time and doesn't need to be altered.  In most model construction functions (in `pygsti.models.modelconstruction`) there are `evotype` and `simulator` arguments that determine these values.  

Often the `simulator` argument can be left as `"auto"`, which selects the `"matrix"` simulator for $\le 2$ qubits and the `"map"` simulator otherwise.  The default for an `evotype` argument is often `"default"`, which is usually a sensible (and user configurable, as described above).
 
 
Below, we'll demonstrate the use of different evolution and forward-simulation types using a local-noise model on 5 and then 10 qubits.  

## 5 qubits
First, let's generate a random circuit using the `pygsti.algorithms.randomcicuit` module.  Random circuits are primarily used in randomized benchmarking (see the [RB tutorial](../rb/Overview) for more information about running RB); here we just use it as an easy way to create an example circuit without having to write down one by hand.

```{code-cell} ipython3
import pygsti, time
import numpy as np

from pygsti.processors import QubitProcessorSpec

n_qubits = 5
ps = QubitProcessorSpec(num_qubits=n_qubits, gate_names=['Gx','Gy','Gcnot'],
                        availability={'Gcnot': [(i,i+1) for i in range(n_qubits-1)]})

c = pygsti.algorithms.randomcircuit.create_random_circuit(ps, length=20)
print(c)
```

Propagate a **density matrix** (`"densitymx"` evolution type) using the **matrix-matrix multiplying** forward simulator (`"matrix"`):

```{code-cell} ipython3
mdl = pygsti.models.create_crosstalk_free_model(ps, simulator="matrix")
print("Gx gate is a ",type(mdl.operation_blks['gates']['Gx']))
t0 = time.time()
out = mdl.probabilities(c)
print("%d probabilities computed in %.3fs" % (len(out), time.time()-t0))
```

We can also propagate a density matrix using the **matrix-vector multiplying** forward simulator (`"map"`).  This forward simulator is much faster for even several (5) qubits.  This is why pyGSTi automatically selects the `"map"` simulator when the number of qubits is $\ge 2$.

```{code-cell} ipython3
mdl = pygsti.models.create_crosstalk_free_model(ps, simulator="map")
t0 = time.time()
out2 = mdl.probabilities(c)
print("%d probabilities computed in %.3fs" % (len(out2), time.time()-t0))
assert(all([np.isclose(out[k],out2[k]) for k in out])) # check that the probabilites are the same
```

If we don't need to consider mixed states, we can represent the quantum state using a **state vector** (`"statevec"`, selected by the `"static unitary"` parameterization type) and use either the matrix-matrix or matrix-vector product simulation types (the latter is again considerably faster even for just 5 qubits):

```{code-cell} ipython3
# TODO: Unitary evolution is not yet supported for matrix
#mdl = pygsti.models.create_crosstalk_free_model(ps, ideal_gate_type='static unitary',
#                                                evotype='statevec', simulator='matrix')
#t0 = time.time()
#out3 = mdl.probabilities(c)
#print("Mat-mat: %d probabilities in %.3fs" % (len(out3), time.time()-t0))
#assert(all([np.isclose(out[k],out3[k]) for k in out])) # check that the probabilites are the same

mdl = pygsti.models.create_crosstalk_free_model(ps, ideal_gate_type='static unitary',
                                                evotype='statevec', simulator='map')
t0 = time.time()
out4 = mdl.probabilities(c)
print("Mat-vec: %d probabilities in %.3fs" % (len(out4), time.time()-t0))
assert(all([np.isclose(out[k],out4[k]) for k in out])) # check that the probabilites are the same
```

Finally, if all the gates are **Clifford** operations (as they are in this case), we can use the `"clifford"` parameterization to propagate a `"stabilizer"` state.  Only the `"map"` simulation type is compatible with the `"stabilizer"` evolution type (selected automatically).

```{code-cell} ipython3
mdl = pygsti.models.create_crosstalk_free_model(ps, ideal_gate_type='static clifford',
                                                simulator='map', evotype='stabilizer')
print("Gx gate is a ",type(mdl.operation_blks['gates']['Gx']))
t0 = time.time()
out5 = mdl.probabilities(c)
print("%d probabilities in %.3fs" % (len(out5), time.time()-t0))
assert(all([np.isclose(out[k],out5[k]) for k in out])) # check that the probabilites are the same
```

## 10 qubits
Let's create a function to compare the above methods for a given number of qubits.  We'll automatically exclude the `"densitymx"`-`"matrix"` case when the number of qubits is greater than 5 as we know this is getting slow at this point.  At 10 qubits, the stabilizer and state-vector simulations are of comparable runtime (though this is largely due to the fact that *all* the outcomes are always computed - see below).

```{code-cell} ipython3
import pygsti, time

def compare_calc_methods(n_qubits):
    print("---- Comparing times for %d qubits (%d outcomes) ----" % (n_qubits,2**n_qubits))
    t0=time.time()
    ps = QubitProcessorSpec(n_qubits, gate_names=['Gx','Gy','Gcnot'],
                                  availability={'Gcnot': [(i,i+1) for i in range(n_qubits-1)]})
    print("Create processor spec: %.3fs" % (time.time()-t0))

    c = pygsti.algorithms.randomcircuit.create_random_circuit(ps, 20)
    print("Random Circuit:")
    print(c)

    if n_qubits <= 5:
        mdl = pygsti.models.create_crosstalk_free_model(ps, simulator="matrix")
        t0 = time.time()
        mdl.probabilities(c)
        print("densitymx, matrix: %.3fs" % (time.time()-t0))

        #mdl = pygsti.models.create_crosstalk_free_model(ps, simulator="matrix",
        #                                               ideal_gate_type='static unitary')
        #t0 = time.time()
        #mdl.probabilities(c)
        #print("statevec, matrix: %.3fs" % (time.time()-t0))

    if n_qubits <= 12:
        mdl = pygsti.models.create_crosstalk_free_model(ps, simulator="map")
        t0 = time.time()
        mdl.probabilities(c)
        print("densitymx, map: %.3fs" % (time.time()-t0))

    mdl = pygsti.models.create_crosstalk_free_model(ps, simulator="map",
                                                    ideal_gate_type='static unitary')
    t0 = time.time()
    mdl.probabilities(c)
    print("statevec, map: %.3fs" % (time.time()-t0))
    
    mdl = pygsti.models.create_crosstalk_free_model(ps, simulator="map",
                                                    evotype='stabilizer',
                                                    ideal_gate_type='static clifford')
    t0 = time.time()
    out5 = mdl.probabilities(c)
    print("stabilizer, map: %.3fs" % (time.time()-t0))
    
compare_calc_methods(10)
```

## More qubits
Going beyond 10 qubits, run times will start to get long even for the stabilizer-simulation case.  This is because pyGSTi currently *always* computes *all* the outcomes of a circuit, the number of which scales exponentially with the system size (as $2^n$ for $n$ qubits).  Future versions will remedy this technical issue, allowing you to compute *just* the outcome probabilites you want.  Once this update is released, the stabilizer state simulation will clearly be faster than either the density-matrix or state-vector approaches; for now, we can see that it get's marginally faster as the number of qubits rises.

```{warning}
This cell takes several minutes to run!
```{warning}

```{code-cell} ipython3
:tags: [nbval-skip]

compare_calc_methods(12)
```

```{code-cell} ipython3
#compare_calc_methods(16)
```

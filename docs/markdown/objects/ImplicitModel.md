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

# Implicit Models
This tutorial explains how to create and use the implicit-layer-operation models present in pyGSTi.  It *doesn't* show you how to build your own custom model class, which will be the topic of a future tutorial.

"Implicit models", as we'll refer to implicit-layer-operation models from now on, store building blocks needed to construct layer operations but not usually the layer operations themselves.  When simulating a circuit, an implicit model creates on the fly, from its building blocks, an operator for each circuit layer.  It therefore only creates operators for the layers that are actually needed for the circuit simulation.   Simulating a circuit with an implicit model is similar to first building an *explicit* model on the fly that contains just operations for the present circuit layers.  These layers are based on the implicit model's **building blocks** and **layer rules**.

Implicit models are very useful within multi-qubit contexts, where there are so many possible circuit layers one cannot easily create and store separate operators for every possible layer.  It is much more convenient to instead specify a smaller set of building-block-operators and rules for combining them into full $n$-qubit layer operations.

PyGSTi currently contains two types of implicit models, both derived from `ImplicitOpModel` (which is derived from `Model`):
- `LocalNoiseModel` objects are noise models where "noise" (the departure or deviance from perfection) of a given gate is localized to *only* the qubits where that gate is intended to act.  Said another way, the key assumption of a `LocalNoiseModel` is that gates act as the perfect identity everywhere except on their *target qubits* - the qubits they are supposed to act nontrivially upon.  Because errors on non-target qubits can broadly be interpreted as "crosstalk", we can think of a `LocalNoiseModel` as a *crosstalk-free* model.  Indeed, in most cases you'll create a `LocalNoiseModel` using the `create_crosstalk_free_model` function.  For concreteness, some examples of local noise are:
    - a rotation gate over-rotates the qubit it's *supposed* to rotate
    - a controlled-not gate acts imperfectly on its control and target qubits but *perfectly* on all other qubits
  
  
- `CloudNoiseModel` objects allow imperfections in a gate to involve qubits in a *neighborhood* of or *cloud* around  the gate's target qubits. When the neighborhood is shrunk to just the target qubits themselves this reduced to a local noise model.  What exactly constitutes a neighborhood or cloud is up to the user.  The `create_cloudnoise_model_from_hops_and_weights` function defines clouds based on a number of "hops" (edge-traversals) on a graph of qubit connectivity that is supplied by the user.  The `create_cloud_crosstalk_model` model allows more flexibility.


## Inside an implicit model: `.prep_blks`, `.povm_blks`, `.operation_blks`, and `.instrument_blks`

Whereas an `ExplicitModel` contains the dictionaries `.preps`, `.povms`, `.operations`, and `.instruments` (which hold *layer* operators), an `ImplicitModel` contains the dictionaries `.prep_blks`, `.povm_blks`, `.operation_blks`, and `.instrument_blks`.  Each of these dictionaries contains a second level of dictionaries, and it is this second level which hold actual operators (`LinearOperator`-, `State`-, and `POVM`-derived objects) - the **building blocks** of the model.  The keys of the top-level dictionary are *category* names, and the keys of the second-level dictionaries are typically gate names or circuit layer labels.  For example, a `LocalNoiseModel` has two categories within its `.operation_blks`: `"gates"`, and `"layers"`, which we'll see more of below.  

To begin, we'll import pyGSTi and define a function which prints the 1st and 2nd level keys of any `ImplicitModel`:

```{code-cell} ipython3
import pygsti
import numpy as np

def print_implicit_model_blocks(mdl, showSPAM=False):
    if showSPAM:
        print('State prep building blocks (.prep_blks):')
        for blk_lbl,blk in mdl.prep_blks.items():
            print(" " + blk_lbl, ": ", ', '.join(map(str,blk.keys())))
        print()

        print('POVM building blocks (.povm_blks):')
        for blk_lbl,blk in mdl.povm_blks.items():
            print(" "  + blk_lbl, ": ", ', '.join(map(str,blk.keys())))
        print()
    
    print('Operation building blocks (.operation_blks):')
    for blk_lbl,blk in mdl.operation_blks.items():
        print(" " + blk_lbl, ": ", ', '.join(map(str,blk.keys())))
    print()
```

## Local-noise (crosstalk free) implicit models
The `LocalNoiseModel` class represents a model whose gates are only have *local noise* (described above) applied to them.  This makes it trivial to combine gate-operations into layer-operations because within a layer gates act on disjoint sets of qubits and therefore so does the (local) noise.

A `LocalNoiseModel` can be built from the default constructors as well as several class methods, but the easiest method for constructing these models is to use the `create_crosstalk_free_model` function. The change in terminology from "local" noise to a "crosstalk-free" model is not significant, and is mostly to keep consistency for other more complex noise models that are specifically structured to study crosstalk.

This function works almost exactly the same as the `create_explicit_model` function from the [ExplicitModel tutorial](ExplicitModel) - given a `ProcessorSpec` and several other options to add nonstandard gates and gate noise, the corresponding model will be returned.

```{code-cell} ipython3
from pygsti.processors import QubitProcessorSpec
pspec = QubitProcessorSpec(4, ['Gxpi','Gypi','Gcnot'], geometry='line')

mdl_locnoise = pygsti.models.create_crosstalk_free_model(pspec)

print("Type = ",type(mdl_locnoise), "\n")
print_implicit_model_blocks(mdl_locnoise, showSPAM=True)
```

An alternative way of viewing the contents of the model is via its `print_modelmembers` method, which gives the  information printed above along with some additional details.

```{code-cell} ipython3
mdl_locnoise.print_modelmembers()
```

We've created a model on 4 qubits with $X(\pi)$, $Y(\pi)$ and *CNOT* gates.  The default qubit labeling (see the `qubit_labels` argument) is the integers starting at 0, so in this case our qubits are labelled $0$, $1$, $2$, and $3$.  The given qubit connectivity (the `geometry` argument) is `"line"`, so there are *CNOT* gates between each adjacent pair of qubits when arranged as $0-1-2-3$.  

Let's take a look at what's inside the model:
- There is just a single `"layers"` category within `.prep_blks` and `.povm_blks`, each containing just a single operator (a state preparation or POVM) which prepares or measures the entire 4-qubit register.  Currently, the preparation and measurement portions of both a `LocalNoiseModel` and a `CloudNoiseModel` are not divided into components (e.g. 1-qubit factors) and so `.prep_blks["layers"]` and `.povm_blks["layers"]` behave similarly to an `ExplicitOpModel`'s `.preps` and `.povms` dictionaries.  Because there's nothing special going on here, we'll omit printing `.prep_blks` and `.povm_blks` for the rest of this tutorial by leaving the default `showSPAM=False` in future calls to `print_implicit_model_blocks`. 

- There are two categories within `.operation_blks`: `"gates"` and `"layers"`.  The former contains three elements which are just the gate names (`"Gxpi"`, `"Gypi"`, and `"Gcnot"`), which hold the 1- and 2-qubit gate operations.  The `"layers"` category contains holds (4-qubit) *primitive layer operations* which give the action of layers containing just a single gate (called "primitive layers").  From their labels we can see which gate is placed where within the layer.

 *(Aside)* The question may come to mind: *why does the model create these layer operations now?*  Why not just create these on an as-needed basis?  The answer is *for efficiency* - it takes some nontrivial amount of work to "embed" a 1- or 2-qubit process matrix within a larger (e.g. 4-qubit) process matrix, and we perform this work once up front so it doesn't need to be repeated later on.

- Gate operations are *linked* to all of the layer operations containing that gate.  For example, the `Gxpi` element of `.operation_blks["gates"]` is linked to the `Gxpi:0`, `Gxpi:1`, `Gxpi:2`, `Gxpi:3` members of `.operation_blks["layers"]`.   Technically, this means that these layer operations contain a *reference* to (not a *copy* of) the `.operation_blks["gates"]["Gxpi"]` object.  This is visible in the `print_modelmembers` output, which places integer ids, shown in parenthesis, on each object.  When an object is repeated `"--link--^"` is printed to indicate that this element is a referce to a previously seen object.  Functionally, this means that whatever noise or imperfections are present in the `"Gxpi"` gate operation will be manifest in all of the corresponding layer operations, as we'll see below.  This behavior is specified by the `independent_gates` argument, whose default value is `False`.  We'll see what happens when we change this farther down below.


The types of individual operators can be accessed straightforwardly.  For example, let's pring the `"Gxpi"` operator:

```{code-cell} ipython3
print(mdl_locnoise.operation_blks['gates']['Gxpi']) # Static!
```

Notice that is a `StaticStandardOp` object, just as the output from `print_modelmembers` indicates.  The gate operations in `.operation_blks["gates"]` are all *static* operators (they have no adjustable parameters - see the [Operators tutorial](Operators) for an explanation of the different kinds of operators).  This is because the default value of the `ideal_gate_type` argument of `"auto"` is equivalent to attempting a number of static types. See the [model parameterization tutorial](ModelParameterization) for a more complete description of parameterization types.

### Creating a `LocalNoiseModel` with independent gates
As we've just seen, by default `create_crosstalk_free_model` creates a `LocalNoiseModel` that contains just a single gate operation for each gate name (e.g. `"Gxpi"`).  This is convenient when we expect the same gate acting on different qubits will have identical (or very similar) noise properties.  What if, however, we expect that the $X(\pi)$ gate on qubit $0$ has a different type of noise than the $X(\pi)$ gate on qubit $1$?  In this case, we want gates on different qubits to have *independent* noise, so we set `independent_gates=True`.  We'll also set a `ideal_gate_type='full'` to demonstrate how to change the type of the created gate objects.

```{code-cell} ipython3
mdl_locnoise_full_indep = pygsti.models.create_crosstalk_free_model(pspec, ideal_gate_type='full',
                                                                    independent_gates=True)
mdl_locnoise_full_indep.print_modelmembers()
```

We see from the above that now there is a different gate object for each set of target qubits, and that the primitive layer operations each embed a *different* gate operation rather than, e.g. both Gxpi:1 and Gxpi:2 linking to (embedding) the same underlying 1-qubit gate.

### Circuit simulation
Now that we have a models, we'll simulate a circuit with four "primitive $X(\pi)$" layers.  Notice from the outcome probabilities that all for layers have imperfect (depolarized) $X(\pi)$ gates:

```{code-cell} ipython3
c = pygsti.circuits.Circuit( [('Gxpi',0),('Gxpi',1),('Gxpi',2),('Gxpi',3)], num_lines=4)
print(c)
mdl_locnoise.probabilities(c)
```

If we compress the circuit's depth (to 1) we can also simulate this circuit, since a `LocalNoiseModel` knows how to automatically create this single non-primitive (contains 4 $X(\pi)$ gates) layer from its gate and primitive-layer building blocks.  Note that the probabilities are identical to the above case.

```{code-cell} ipython3
c2 = c.parallelize()
print(c2)

mdl_locnoise.probabilities(c2)
```

So far we've use the perfect (noiseless) model.  Let's create a model similar to `mdl_locnoise` but with `ideal_gate_type='full'` so that we can modify the gates after they're built.  Then we'll depolarize the $X(\pi)$ gate and compute the circuit probabilities again.

```{code-cell} ipython3
mdl_locnoise_full = pygsti.models.create_crosstalk_free_model(pspec, ideal_gate_type='full')
mdl_locnoise_full.operation_blks['gates']['Gxpi'] = np.array([[1,   0,   0,   0],
                                                              [0, 0.9,   0,   0],
                                                              [0,   0,-0.9,   0],
                                                              [0,   0,   0,-0.9]],'d')

mdl_locnoise_full.probabilities(c)
```

As expected, depolarizing the `'Gxpi'` gate has the effect of adding noise to *all* 4 of the $X(\pi)$ gates in the circuit, since `independent_gates=False`.  If we follow a similar procedure for the model created above with `independent_gates=True`, then we must choose which gate to depolarize (we choose qubit 0):

```{code-cell} ipython3
mdl_locnoise_full_indep.operation_blks['gates'][('Gxpi',0)] = np.array([[1,   0,   0,   0],
                                                                        [0, 0.9,   0,   0],
                                                                        [0,   0,-0.9,   0],
                                                                        [0,   0,   0,-0.9]],'d')
```

When we simulate the same circuit as above, we find that only the first (on qubit $0$) $X(\pi)$ gate has depolarization error on it now:

```{code-cell} ipython3
print(c)
mdl_locnoise_full_indep.probabilities(c)
```

### Construction by appending
The `create_crosstalk_free_model` demonstrated above allows you to build a wide variety of local noise models fairly easily and quickly - but what if you need something just a little different?  By setting `ensure_composed_gates=True`, all of the output gates (the elements of `.operation_blks['gates']`) will be *composed* gates - i.e. `ComposedOp` objects.  This is nice because composed operations allow you to easily tag on additional operations - to add additional elements to whatever is being composed.  Here's an example of how to create a noiseless crosstalk-free model and then add an arbitrary additional error term to both the `Gx` and `Gy` gates:

```{code-cell} ipython3
mdl = pygsti.models.create_crosstalk_free_model(pspec, ensure_composed_gates=True, independent_gates=False)

additional_error = pygsti.modelmembers.operations.FullTPOp(np.identity(4,'d')) # this could be *any* operator

#ComposedOp objects support .append( operation )
mdl.operation_blks['gates']['Gxpi'].append(additional_error)
mdl.operation_blks['gates']['Gypi'].append(additional_error)

#It's a good idea to query the number of parameters after messing with a model's internals as this causes
# the parameter indices to be rebuild/relinked.
mdl.num_params

mdl.print_modelmembers()
```

This has advantages over simply setting gates to numpy arrays after using `parameterization="full"` (as demonstrated above) because you're adding a custom operation *object*, which can posess a custom *parameterization*.

## Cloud-noise implicit models

`CloudNnoiseModel` objects are designed to represent gates whose imperfections may affect the qubits in a neighborhood, or *cloud*, around a gate's target qubits.  Since this neighborhood is defined by the user, and can be arbitrarily large, cloud noise models have the potential to capture everything from purely local noise (each gate acts imperfectly only on its target qubits) to completely general Markovian noise (each gate acts as a $N$-qubit process matrix on all $N$ qubits).

Importantly, `CloudNoiseModel` objects can capture **crosstalk errors**  - errors which violate either the locality of a gate (if it operates non-trivially on non-target qubits) or its independence from its "environment" (the other gates it appears with in a circuit layer).  

We can create a `CloudNoiseModel` using the `create_cloud_crosstalk_model` function, which is similar to the model creation function for other types of models. In particular, we specify a processor specification, and optionally additional noise broken down into depolarization, stochastic, and Lindblad types.

Let's start with a simple noise-free case, where we initialize a model just from a processor specification:

```{code-cell} ipython3
mdl_cloudnoise_perfect = pygsti.models.create_cloud_crosstalk_model(pspec)

print_implicit_model_blocks(mdl_cloudnoise_perfect, showSPAM=True)
```

We see that a `CloudNoiseModel` has *three operation categories*: `"gates"`, `"layers"`, and `"cloudnoise"`.  The first two serve a similar function as in a `LocalNoiseModel`, and hold the (1- and 2-qubit) gate operations and the (4-qubit) layer operations, respectively.  The `"cloudnoise"` category contains layer operations corresponding to the "cloud-noise" associated with each primitive layer, i.e. each single-gate layer.

This structure is very similar to a local noise model, but there are some important differences in the way the parts of the structure are used.  In particular,

- in a local-noise model, the elements of the `"gates"` and `"layers"` categories hold a composition of ideal and noisy operations.  The type of ideal operators can be set by the `ideal_gate_type` argument to `create_crosstalk_free_model`.

- in a cloud-noise model, `"gates"` and `"layers"` categories *always contain perfect operations*. As such, there are no `ideal_gate_type` and similar arguments to `create_cloud_crosstalk_model`.  Noise is added by composition of these perfect gates and layers with elements in the `"cloudnoise"` category.

In the model we created above, there are no `"cloudnoise"` elements because we didn't specify any noise.  The `create_cloud_crosstalk_model` function constructs a *noisy* `CloudNoiseModel` via the `depolarization_strengths`, `stochastic_error_probs`, and `lindblad_error_coeffs` arguments.  The use of these arguments is covered in more depth in the [model noise tutorial](ModelNoise), but we give a brief example here to illustrate how the internals of a cloud noise model are updated when noise is added.

The following constructs a model that places certain on- and off-target noise on particular $X(\pi)$ and CNOT gates:

```{code-cell} ipython3
import pygsti
from pygsti.processors import QubitProcessorSpec
pspec = QubitProcessorSpec(4, ['Gxpi','Gypi','Gcnot'], geometry='line')
mdl_cloudnoise = pygsti.models.create_cloud_crosstalk_model(pspec,
    lindblad_error_coeffs={
        ('Gxpi',0): { ('H','X'): 0.1, ('S','XY:0,1'): 0.1},
        ('Gcnot',0,1): { ('H','ZZ'): 0.02, ('S','XX:0,1'): 0.02 },
    }
)
```

Notice that the keys of the error dictionary give specific qubits, e.g. `('Gxpi',0)`, but that the basis elements given in the corresponding dictionary of error-generator rates by contain other/additional qubits (e.g. `'XY:0,1'`, which is the 2-qubit Pauli operator that acts as $X$ on qubit 0 and $Y$ on qubit 1).  Furthermore, note that when a basis element does not have any qubit specification then the target qubit(s) of the current gate is/are assumed (e.g. the `'X'` in `('H','X')` for key `('Gxpi',0)` is the $X$ Pauli on qubit 0).

Now let's take a look at the operations within this model:

```{code-cell} ipython3
mdl_cloudnoise.print_modelmembers()
```

We see that now there are items within the `"cloudnoise"` category, labeled in the same manner as the primitive layers within the `"layers"` category.  To simulate a circuit layer, a cloud noise model composes (in this order): 

1. the perfect target gates
2. global "background" noise
3. gate-specific "cloud" noise

Steps 2 and 3 are optional - if there is no background or gate-specific cloud noise then these are omitted.  Step 1 is itself a composition of primitive layers (for all the gates in the layer).  This is straightforward since the gates act on disjoint qubits.  Step 2 utilizes the model's global idle operation if the `implicit_idle_mode='add_global'` or is omitted if `implicit_idle_mode='none'` (the default).  Step 3 combines all of the gates' `"cloudnoise"` elements.

```{note}
(Advanced aside: not demonstrated in this tutorial yet.)
```
Since the clouds of different gates may overlap, this process is non-trivial and can be performed in different ways.  By default, the operation of Step 3 is generated by simply composing the different cloud-noise maps (elements of the `"cloudnoise"` category).  The `errcomp_type` argument can change this behavior so that the Lindbladian *error generators* are composed (summed) instead of the maps.  When `errcomp_type="gates"` (the default) the noise maps for the components are of a gate layer are composed; when `errcomp_type="errorgens"` the error generators of the noise maps are added and used as the error generator for the final operation 

For example, the circuit layer `"[Gxpi:0]"` is the composition of 2 factors (steps 1 and 3 above):

```{code-cell} ipython3
example_circuit_layer = pygsti.circuits.circuitparser.parse_label("[Gxpi:0]")
op = mdl_cloudnoise.circuit_layer_operator(example_circuit_layer)
print(op)
```

When we simulate a circuit containing the $X(\pi)$ gate on qubit $0$ we find that it indeed affects qubit $1$ as well:

```{code-cell} ipython3
c = pygsti.circuits.Circuit("[Gxpi:0]", (0,1,2,3))
print(c)
mdl_cloudnoise.probabilities(c)
```

(compare this with the results from the perfect model below)

```{code-cell} ipython3
print(c)
mdl_cloudnoise_perfect.probabilities(c)
```

### Construction from hops and weights

Sometimes we would like to see if a given set of data can be described by a model that only allows errors that are ***geometrically-local* and *low-weight***.  By geometrically local errors we mean that those that only affect a gate's target and neighboring qubits, where neighbors are defined by a graph of qubit connectivity.  Low weight errors mean that only errors with weight (roughly the number of qubits an error affects simultaneously) less than a maximum value are allowed.

We can build a cloud noise model with these properties using the `create_cloud_crosstalk_model_from_hops_and_weights` function.  It utilizes a processor specification with a `geometry` (a qubit graph) to create errors that are geometrically local and limited to a maximum weight.  The "cloud" for each gate is defined as the set of qubits that can be reached by some number ($k$, say) of edge traversals (or *hops*) from the gate's target qubits along the connectivity graph.  Within a gate's cloud, all errors of the specified types up to a given maximum weight are allowed.  

Let's create a cloud noise model using our processor spec, where there are 4 qubits arranged in a line.

```{code-cell} ipython3
mdl_hopsweights = pygsti.models.modelconstruction.create_cloud_crosstalk_model_from_hops_and_weights(
                        pspec,
                        max_idle_weight=0,
                        max_spam_weight=1,
                        maxhops=1,
                        extra_weight_1_hops=0,
                        extra_gate_weight=0)
```

The call to `create_cloud_crosstalk_model_from_hops_and_weights` resembles `create_crosstalk_free_model` but contains arguments that specify the cloud sizes and maximum error weights used:
- `maxhops` specifies how many hops from a gate's target qubits (along the qubit graph given by the `geometry` argument ,which is `"line"` here) describe which qubits comprise the gate's *cloud*.
- `max_idle_weight` specifies the maximum-weight of error terms in the global idle operation.
- `max_spam_weight` specifies the maximum-weight of error terms in the state preparation and measurement operations.
- `extra_gate_weight` specifies the maximum-weight of error terms in gates' clouds *relative* to the number of target qubits of the gate.  For instance, if `extra_gate_weight=0` then 1-qubit gates can have up to weight-1 error terms in their clouds and 2-qubit gates can have up to weight-2 error terms.  If `extra_gate_weight=1` then this changes to weight-2 errors for 1Q gates and weight-3 errors for 2Q gates.
- `extra_weight_1_hops` specifies an additional number of hops (added to `maxhops`) that applies only to weight-1 error terms.  For example, in a 8-qubit line example, if `maxhops=1`, `extra_gate_weight=0`, and `extra_weight_1_hops=1` then a 2-qubit gate on qubits $4$ and $5$ can have up-to-weight-2 errors on qubits $\{3,4,5,6\}$ and additionally weight-1 errors on qubits $2$ and $7$.
- `errcomp_type` specifes how errors are composed when creating layer operations.  An advanced topic that we don't explore here.

If our processor specification contained a global idle operation then we could set `max_idle_weight` to be greater than zero, and the model would conatin a single (noisy) global idle operation.  This noisy idle operation has the form $\exp{\mathcal{L}}$, where $\mathcal{L}$ is a Lindbladian containing error terms only up to some *maximum weight* (typically 1 or 2).  If we set `implicit_idle_mode='add_global'` then this global idle is also treated as "background idle" that occurs in every circuit layer, even ones having gates.

In our example above, the processor specification's graph specifies four qubits in a line: $0-1-2-3$ and we allow at most 1 hop along the graph (`maxhops=1`).  Thus, the noise cloud of a single-qubit gate on qubit $1$ is the set of qubits $\{0,1,2\}$ and the cloud for a two-qubit gate on qubits $1$ and $2$ is the set $\{0,1,2,3\}$.  This can be seen by printing the cloud noise operation for the $X(\pi)$ gate on qubit $1$ and noticing that it includes gates embedded into the `(0,)`, `(1,)`, and `(2,)` spaces:

```{code-cell} ipython3
print(mdl_hopsweights.operation_blks['cloudnoise'][('Gxpi', 1)])
```

It's also possible, as we've done before, to view the contents of the model using its `print_modelmembers` method.  The output is quite long, as we see below.  This illustrates the convenience of the `create_cloud_crosstalk_model_from_hops_and_weights` function -- it would take a lot of effort creating this model by specifying the coefficients of all these Lindbladian terms via `create_cloud_crosstalk_model`'s `lindblad_error_coeffs` argument!

```{code-cell} ipython3
mdl_hopsweights.print_modelmembers()
```

Note also that when a model is created using `create_cloud_crosstalk_model_from_hops_and_weights` all of the Lindbladian coefficients start with value 0.  A model is created with the *potential* for errors in any of the geometrically-local and low-weight ways specificed, but there aren't actually any errors yet.  Such models are useful primarily as the initial model for a fitting procedure that will tweak the model's parameters (that are initially zero) to fit data.

## Additional resources

Getting a list of the gate names recognized by pyGSTi:

```{code-cell} ipython3
known_gate_names = list(pygsti.tools.internalgates.standard_gatename_unitaries().keys())
print(known_gate_names)
```

## Next steps
To learn more about using implicit models, you may want to check out the [model parameterizations tutorial](ModelParameterization), which covers material especially relevant when optimizing implicit models, and the [model noise tutorial](ModelNoise), which describes how to add noise to implicit (and explicit) models.

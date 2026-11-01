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

# Model Noise
This tutorial covers how various kinds of noise (errors) can be added to a model's operations.  The procedure for specifying the noise is similar for the different types of models, though the types of noise allowed and how this noise is incorporated into the model's structure differ as we point out below.

```{code-cell} ipython3
import pygsti
from pygsti.models import modelconstruction as mc
from pygsti.models import modelnoise as mn
```

## Standard noise types
There are three standard types of noise that can be added to operations in pyGSTi: depolarization, stochastic, and Lindbladian.  The first two types are common in the literature, while the third, "Lindbladian", needs a bit more explanation.  Many types of gate errors can be represented in terms of an *error generator*.  If $G$ is a noisy gate (a CPTP map) and $G_0$ is it's ideal counterpart, then if we write $G = e^{\Lambda}G_0$ then $\Lambda$ is called the gate's *error generator*.  A `LindbladErrorgen` object, exponentiated using a `ExpErrorgenOp` object represent this $e^{\Lambda}$ in pyGSTi.  If we write $\Lambda$ as a sum of terms, $\Lambda = \sum_i \alpha_i F_i + \sum_{i\neq j} \alpha_{ij} F_{ij}$ then, when the $F_i/F_{ij}$ are specific generators for well-known errors (e.g. rotations or stochastic errors), the $\alpha_i/\alpha_{ij}$ can roughly be interpreted as the error *rates* corresponding to the well-known error types.  PyGSTi has three specific generator types (where $P_i$ is a Pauli operator or tensor product of Pauli operators):

- **Hamiltonian**: $H_i : \rho \rightarrow -i[P_i,\rho]$
- **Stochastic**: $S_i : \rho \rightarrow P_i \rho P_i - \rho$
- **Correlated**: $C_{ij} : \rho \rightarrow P_i \rho P_j + P_j \rho P_i - \frac{1}{2}\{\{P_i,P_j\}, \rho\}$
- **Affine/Active**: $A_{ij} : \rho \rightarrow i\left(P_i \rho P_j + P_j \rho P_i + \frac{1}{2}\{[P_i,P_j], \rho\}\right)$

See our paper on [the taxonomy of small errors](https://arxiv.org/abs/2103.01928v1) for a more theoretical foundation of error generators.

Many of the model construction functions take arguments that allow users to add these standard noise types conveniently when a model is created.  Each argument expects a dictionary, where the keys are gate names and the values specify the corresponding noise. The values are different types for each argument:

- `depolarization_strengths`: Values are floats, which corresponding to strengths of a `DepolarizeOp`
- `stochastic_error_probs`: Values are lists of length $4^{N_{qubits} - 1}$, which correspond to coefficients of a stochastic Pauli channel in a `StochasticNoiseOp`. Order of the rates is lexographical, and can be checked by looking at the elements of a `"pp"` Basis object.
- `lindblad_error_coeffs`: Values are a dict where the key has the form `(<type>, <basis_elements>)` and the values are the $\alpha_i$ coefficients in the sum of Lindblad terms, which are then exponentiated to give the final noise. The type includes:
  - `'H'` for Hamiltonian errors
  - `'S'` for Pauli-stochastic errors
  - `'C'` for correlated Pauli-stochastic errors
  - `'A'` for affine/active errors
  
  and strings of `I`, `X`, `Y`, and `Z` can be used to label a Pauli basis element. 

### Crosstalk-free (local noise) models
We'll start with an example of placing noise on a crosstalk-free model.

```{code-cell} ipython3
pspec = pygsti.processors.QubitProcessorSpec(2, ['Gxpi2', 'Gypi2', 'Gcnot'], geometry='line')

mdl_locnoise = pygsti.models.create_crosstalk_free_model(
    pspec,
    depolarization_strengths={'Gxpi2': 0.1},  # Depolarizing noise on X
    stochastic_error_probs={'Gypi2': [0.04, 0.05, 0.02]}, # Stochastic Pauli noise on Y
    lindblad_error_coeffs={'Gcnot': {('H','ZZ'): 0.15} } # Hamiltonian ZZ error on CNOT
)
```

Let's print out the gates to see the corresponding construction
These should all be ComposedOps with the following format:
 - Op 1: The ideal gate, parameterized by the ideal_gate_type kwarg
 - Op 2: The noise, as parameterized by whichever noise specification was provided

```{code-cell} ipython3
for gate_name, gate in mdl_locnoise.operation_blks['gates'].items():
    print(gate_name)
    print(gate)
```

This can also be seen via the models' `print_modelmembers()` method (see the "`operation_blks|gates`" category):

```{code-cell} ipython3
mdl_locnoise.print_modelmembers()
```

#### Overriding gate noise with layer noise
By specifying a *primitive layer label* instead of a gate name as a key, we can modify the noise on a gate when it's applied to a particular qubit.  Note that this *doesn't* require that we set `independent_gates=True` in the construction function, as this argument refers to the ideal gates.  Noise can be separately given to individual primitive layer operation regardless of the values of `independent_gates`.

Here's an example where we override the noise on the `Gxpi2` gate.  Notice how the `operation_blks|gates` category has both a `Gxpi2` and `Gxpi2:0` keys.

```{code-cell} ipython3
mdl_locnoise = pygsti.models.create_crosstalk_free_model(pspec,
    stochastic_error_probs={
        'Gxpi2': [0.04, 0.05, 0.02], # Stochastic Pauli noise on X
        ('Gxpi2', 0): [0.08, 0.1, 0.06] # Override rates for X on one of the qubits
    }
)
mdl_locnoise.print_modelmembers()
```

### Explicit models
Specifying noise in explicit models is similar, but since explicit models only hold circuit *layer* operations, we must always specify a layer.  For example, noise cannot be attributed to just the $X(\pi/2)$ gate, it needs to be attributed to the gate *on* a particular qubit.  Note also that primitive layer labels can be specified as either a tuple, e.g. `("Gxpi2", 0)`, or by a more compact string, e.g. `"Gxpi2:0"`.  Similarly, Lindbladian error rates can be identified by `(type, basis_labels)` tuples or a compact string, e.g., `('H','ZZ')` or `'HZZ'`.

Here is an example of adding noise to an explicit model using both tuple and compact string formats:

```{code-cell} ipython3
pspec = pygsti.processors.QubitProcessorSpec(2, ['Gxpi2', 'Gypi2', 'Gcnot'], geometry='line')

explicit_model = mc.create_explicit_model(pspec,
                                          depolarization_strengths={'Gxpi2:0': 0.1,},
                                          stochastic_error_probs={('Gypi2', 0): [.01]*15},
                                          lindblad_error_coeffs={('Gcnot', 0, 1): {'HZZ': 0.07},
                                                                 'Gcnot:1:0': {('H','IX'): 0.07} },
                                         )
```

Printing the model's members reveals the simpler internal structure of an `ExplicitOpModel` - there's only a single `operations` category which holds layer operations.  These operations are `ComposedOp` operators that compose the ideal gate (`StaticUnitaryOps` in this case, since we didn't specify `ideal_gate_type`) with the specified noise operations.

```{code-cell} ipython3
explicit_model.print_modelmembers()
```

#### Noise on SPAM operators
State preparation and measurement (SPAM) operators always act on all $N$ qubits.  To add noise to the $N$-qubit state preparation, or measurement operations, you can specify `'prep'`, and `'povm'` as keys in the error dictionaries, just like any other layer label.   The below cell placed depolarization noise on both the state preparation and measurement of a model.

```{code-cell} ipython3
mdl_locnoise = pygsti.models.create_crosstalk_free_model(pspec,
                                                         depolarization_strengths={
                                                             'Gxpi2': 0.1,
                                                             'prep': 0.01,
                                                             'povm': 0.01}
                                                        )

explicit_model = mc.create_explicit_model(pspec,
                                          depolarization_strengths={
                                                             'Gxpi2:0': 0.1,
                                                             'Gxpi2:1': 0.1,
                                                             'prep': 0.01,
                                                             'povm': 0.01}
                                                        )

mdl_locnoise.print_modelmembers()
```

The printout above shows that the SPAM operators are now the ideal operators composed with depolarization noise.  It also reveals that the specified depolarization strength is applied as a parallel 1-qubit depolarization to each of the qubits.

### Nonlocal noise (crosstalk)
So far, all the noise we've specified has been directed at the *target* qubits of the relevant operation.  For instance, when a depolarization strength is specified for a 1-qubit gates, it applies the given depolarization to gate's single target qubit.  When depolarization is applied to a 2-qubit gate, 2-qubit depolarization is applied to the target qubits.  When Lindblad error rates are given for a 1-qubit gate, they are indexed by single Pauli elements, e.g. `('H','X')`, whereas for a 2-qubit gate they are indexed by 2-qubit Paulis, e.g. `('H','XX')`.

In a crosstalk-free model, noise can *only* be specified on the target qubits - noise on non-target qubits is simply not allowed.  But for an explicit model, which holds solely $N$-qubit layer operations, noise for a gate (layer) can be applied to *any* of the qubits.  To specify noise that is not on the target qubits of a gate,

- as the values of `depolarization_strengths` or `stochastic_error_probs`, pass a dictionary that maps qubit labels to noise values.  The qubit labels (keys) designate which qubits the noise acts upon.
- add a colon followed by comma-separated qubit labels to the basis labels in a Lindblad error term.

Here's an example of how to set non-local errors on the gates in an explicit model:

```{code-cell} ipython3
#Nonlocal explicit model
explicit_model = mc.create_explicit_model(
    pspec,
    depolarization_strengths={('Gxpi2', 0): {(0,1): 0.2}},  # 2-qubit depolarization
    lindblad_error_coeffs={('Gypi2', 0): {('H','ZZ:0,1'): 0.07,  # Hamiltonian ZZ on qubits 0 and 1
                                          ('S', 'X:1'): 0.04}  # Stochastic X on qubit 1 (not a target qubit)
                          },
    )
```

The errors can be verified, as usual, by printing the model's member structure:

```{code-cell} ipython3
explicit_model.print_modelmembers()
```

### Reduced error generator models

One potentially powerful way to include nonlocal noise with a few lines of code is to include entire sectors of the elementary error generators. For example, one can extend past a crosstalk-free model with only a few parameters by including the H and S sectors on neighboring qubits.

First, let us create an ideal model similar to those above:

```{code-cell} ipython3
ideal_model = mc.create_explicit_model(pspec) # No noise, we will add that manually!
```

Next, we will make lists of all the weight-1 and weight-2 Pauli strings. We will use these to restrict the CA blocks to only weight-1 later.

```{code-cell} ipython3
w1_labels = [lbl for lbl in ideal_model.basis.labels if sum([c != 'I' for c in lbl]) == 1]
w2_labels = [lbl for lbl in ideal_model.basis.labels if sum([c != 'I' for c in lbl]) == 2]
```

Now we can go through each operation and create three "coefficient blocks". Naively, what we want are weight-1 and weight-2 H and S errors (HS2) and only weight-1 C and A (CA1) errors, but we have to organize our blocks slightly differently due to how they are stored internally. The blocks we can make are:

- H-only blocks
- S-only blocks
- SCA blocks

So we instead build our blocks as: H12, SCA1, S2.

Finally, once we have our blocks, we create the actual Lindbladian error generator and append the exponentiated Lindbladian to the ideal operation.

```{code-cell} ipython3
import pygsti.modelmembers.operations as ops
from pygsti.modelmembers.operations.lindbladcoefficients import LindbladCoefficientBlock as LindbladCBlock

HS2_CA1_model = ideal_model.copy()
HS2_CA1_model.operations.flags['auto_embed'] = False

for lbl, op in HS2_CA1_model.operations.items():
    # Lindblad coefficient blocks
    H12_block = LindbladCBlock('ham', ideal_model.basis, w1_labels + w2_labels, param_mode='elements')
    
    S2_block = LindbladCBlock('other_diagonal', ideal_model.basis, w2_labels, param_mode='cholesky')
    
    SCA1_block = LindbladCBlock('other', ideal_model.basis, w1_labels, param_mode='cholesky')
    
    # Build op
    errgen = ops.LindbladErrorgen([H12_block, S2_block, SCA1_block], state_space=ideal_model.state_space)
    HS2_CA1_model.operations[lbl] = ops.ComposedOp([
        op.copy(),
        ops.ExpErrorgenOp(errgen)
    ])
```

### Cloud-crosstalk (cloud noise) models
Cloud-crosstalk models can accept the most general noise operations: they can be given local, per-gate noise as for a local noise model, and non-local noise on specific layers like for an explicit model.  Furthermore, non-local noise can be specified for *gates* (rather than layers) in a meaningful way by using **stencils**, as we demonstrate below.

Let's start with an example using error specifications we've already seen:

```{code-cell} ipython3
cloudmodel = pygsti.models.create_cloud_crosstalk_model(
    pspec,
    stochastic_error_probs={'Gypi2': (0.03, 0.01, 0.03)},  # local noise on a gate
    lindblad_error_coeffs={
        ('Gxpi2',0): { ('H','X'): 0.01, ('S','XY:0,1'): 0.01},  # local and nonlocal noise on a layer 
        ('Gcnot',0,1): { ('H','ZZ'): 0.02, ('S','XX'): 0.02 },  # local noise on a layer
    }
)
```

The effect of these errors is entirely in the `operation_blks|cloudnoise` category.  As described in the tutorial on cloud noise models, *all* of the non-idealities in a `CloudNoiseModel` are contained in this category.

```{code-cell} ipython3
cloudmodel.print_modelmembers()
```

#### Stencils
When there are many qubits, it can be tedious to specify non-local noise in the way shown above since you need to specify the qubits upon which the noise acts differently for each primitive layer operation.  For example, if we have a chain of 10 qubits and want an $X(\pi/2)$ gate to depolarize the qubit to the left and right of the gate's target qubit, we would need to specify nonlocal noise on qubits 1 and 3 for $X(\pi/2)$ on qubit 2, nonlocal noise on qubits 2 and 4 for $X(\pi/2)$ on qubit 3, etc.

*Stencils* have been developed to make this process easier.  `ProcessorSpec` objects can contain a `QubitGraph` giving the geometry of the processor (or, more accurately, of an API of the processor).  This graph can include tags that associate a *direction name* with each of its edges.  For example, the built-in (and default) linear chain (`"line"`) graph has two directions, "left" and "right".  When specifying non-local noise, instead of specifying absolute qubit names, you can specify *relative qubit labels*.  


The use of relative qubit labels - or *stencil labels* as they're called in pyGSTi - causes noise to be placed on qubits relative to a gate's target qubits.  The format of a stencil label is:

`"@<target_qubit_index> + <direction> + <direction> ..."`

where `<target_qubit_index>` is an integer between 0 and the gate's number of target qubits minus 1 (e.g. 0 and 1 for a 2-qubit gate), and `<direction>` is an optional direction name to move along.  Any direction names that are used must be present in the processor specification's graph (its `geometry`).

Here's a simple example that uses stencils on a processor of 4-qubits in a chain:

```{code-cell} ipython3
pspec_4Qchain = pygsti.processors.QubitProcessorSpec(4, ['Gxpi', 'Gcnot'], geometry='line')
cloudmodel_stencils = pygsti.models.create_cloud_crosstalk_model(
    pspec_4Qchain,
    depolarization_strengths = {
        'Gxpi': {('@0+left',): 0.01,   # apply depolarization noise to the qubit to the left of the target
                 ('@0+right',): 0.01}  # apply depolarization noise to the qubit to the right of the target
    },
    stochastic_error_probs = {
        'Gxpi': {('@0',) : (0.03, 0.05, 0.05)}  # local noise, so could omit use of stencil here and
        #'Gxpi': (0.03, 0.05, 0.05)             # <-- use this line instead
    }
)
```

Here we place Pauli stochastic noise on the `Gxpi` gate and depolarizing noise on the qubits to either side.  The stencil has the effect of describing noise for each `('Gxpi',#)` layer based on the contents of the `'Gxpi'` noise specifications in a way that is dependent on the target qubit (`#`).  To be pedantic,

- for `('Gxpi',0)`, `@0` evaluates to `0`, `@0+left` evaluates to NULL and `@0+right` evaluates to `1`.
- for `('Gxpi',1)`, `@0` evaluates to `1`, `@0+left` evaluates to `0` and `@0+right` evaluates to `2`.
- for `('Gxpi',2)`, `@0` evaluates to `2`, `@0+left` evaluates to `1` and `@0+right` evaluates to `3`.
- for `('Gxpi',3)`, `@0` evaluates to `3`, `@0+left` evaluates to `2` and `@0+right` evaluates to NULL.

When a NULL is encountered in stencil evaluation, no noise is added and evaluation continues.  This can be seen by reading through the model member structure (thought it's quite long!):

```{code-cell} ipython3
cloudmodel_stencils.print_modelmembers()
```

Here's another examples of using stencils to construct in a 4-qubit cloud noise model:

```{code-cell} ipython3
cloudmodel_stencils2 = pygsti.models.create_cloud_crosstalk_model(pspec_4Qchain,
    lindblad_error_coeffs={
            'Gxpi': { ('H','X'): 0.01, ('S','X:@0+left'): 0.01, ('S','XX:@0,@0+right'): 0.02},
            'Gcnot': { ('H','ZZ'): 0.02, ('S','XX:@1+right,@0+left'): 0.02 },
        })
```

Basis-element notation can also be abbreviated for convenience (this builds the same model a above):

```{code-cell} ipython3
cloudmodel_stencils2_abbrev = pygsti.models.create_cloud_crosstalk_model(pspec_4Qchain,
    lindblad_error_coeffs={
        'Gxpi': { 'HX': 0.01, 'SX:@0+left': 0.01, ('SXX:@0,@0+right'): 0.02},
        'Gcnot': { 'HZZ': 0.02, 'SXX:@1+right,@0+left': 0.02 },
    })
```

And here's one more final example that combines several types of errors:

```{code-cell} ipython3
cloudmodel_stencils3 = pygsti.models.create_cloud_crosstalk_model(
    pspec_4Qchain,
    depolarization_strengths = {'Gxpi': 0.05},
    lindblad_error_coeffs = {
        'Gxpi': {'HX:@0': 0.1},
        'Gxpi:2': {'HY:2': 0.2},
        'Gcnot': {'HZZ:@0+left,@0': 0.02, 'HZZ:@1,@1+right': 0.02}
    })
```

#### Noise parameterizations
Along with the amount and type of noise added to a model, you can also specify how it is represented as pyGSTi operator objects. This is significant when the model will be tweaked or optimized later on, in which case the paramterization determines how the noise operations are allowed to be tweaked (changed).  Parameterization types are especially important in cloud noise models where this constitutes the entirety of how the model is parameterized.

The parameterizations for the three standard noise types are given by specifying the `depolarization_parameterization`, `stochastic_parameterization`, and `lindblad_parameterization` arguments.  The options for these are as follows:

- `depolarization_parameterization`:
    - `"depolarize"` (default) builds `DepolarizeOp` objects with the strength given in `depolarization_strengths`.  A `DepolarizeOp` object contains a single parameter for the depolarization rate.
    - `"stochastic"` builds `StochasticNoiseOp` objects, which have separate parameters for each Pauli stochastic error rate, all of which are initially equal.  The depolarization strength is thus split evenly among the individual Pauli stochastic channels of a `StochasticNoiseOp`.
    - `"lindblad"` builds exponentiated `LindbladErrorgen` object containing Pauli stochastic terms.  The error generator object is built with `parameterization="D"` (an argument of `LindbladErrorgen.__init__`).  We will refer to this as the *mode* of the `LindbladErrorgen`.  `"D"` (depolarization) mode means it that has a single parameter is squared to give the depolarization rate.

- `stochastic_parameterization`:
    - `"stochastic"` (default) builds `StochasticNoiseOp` objects which have separate parameters for each Pauli stochastic error rate.  Elements of `stochastic_error_probs` are used as coefficients in a linear combination of individual Pauli stochastic channels.
    - `"lindblad"` builds exponentiated `LindbladErrorgen` object containing Pauli stochastic terms.  The error generator object is built in `"CPTP"` mode, which means it that there is one parameter per stochastic rate (equal to the square root of the rate so it is constrained to be positive).
    
- `lindblad_parameterization`:  Lindblad errors are always represented by exponentiated `LindbladErrorgen` objects.  A `LindbladErrorgen` can have several different internal parameterizations, or *modes* as we refer to them here to avoid confusion with the noise or model parameterization.  The mode is by definition the value of the `parameterization` argument supplied to `LindbladErrorgen.__init__`, which influences what types of elementary error generators (e.g. 'H', 'S', 'C' and 'A') are allowed and whether the Pauli stochastic error rates are constrained to be positive. The value of `lindblad_parameterization` can be set to any valid mode value (see documentation for more details).  Usually the default values of `"auto"` is fine. 

The examples below show how the number of parameters in a cloud noise model can vary based on the way in which noise is parameterized.  For these examples, we'll go back to the 2-qubit processor specification for simplicity.

```{code-cell} ipython3
cloudmodel = pygsti.models.create_cloud_crosstalk_model(
    pspec,
    depolarization_strengths={'Gxpi2': 0.1},
    depolarization_parameterization='depolarize',
    stochastic_error_probs={'Gypi2': (0.03, 0.01, 0.03)},  # local noise on a gate
    stochastic_parameterization='stochastic'
)
print(cloudmodel.num_params, "params, 1 depolarization rate, 3 stochastic rates")
```

By setting the `independent_gates` argument to `True` (`False` is the default), the noise is applied to each set of target qubits independently and so have separate parameters.

```{code-cell} ipython3
cloudmodel = pygsti.models.create_cloud_crosstalk_model(
    pspec,
    depolarization_strengths={'Gxpi2': 0.1},
    depolarization_parameterization='depolarize',
    stochastic_error_probs={'Gypi2': (0.03, 0.01, 0.03)},  # local noise on a gate
    stochastic_parameterization='stochastic',
    independent_gates=True
)
print(cloudmodel.num_params, "params, 2 depolarization rates, 6 stochastic rates (because there are 2 qubits)")
```

Changing the depolarization parameterization to `"stochastic"` makes the `Gxpi2` gate take 3 parameters.  Changing the stochastic parameterization to `"lindblad"` doesn't change the number of parameters (still 3) but does change the type of gate object used to model the noise:

```{code-cell} ipython3
cloudmodel = pygsti.models.create_cloud_crosstalk_model(
    pspec,
    depolarization_strengths={'Gxpi2': 0.1},
    depolarization_parameterization='stochastic',
    stochastic_error_probs={'Gypi2': (0.03, 0.01, 0.03)},  # local noise on a gate
    stochastic_parameterization='lindblad'
)

print(cloudmodel.num_params, "params, 3 initially equal from depolarization, 3 stochastic error-generator rates")

print("\nGyp2 cloud noise is:")
print(cloudmodel.operation_blks['cloudnoise'][('Gypi2',0)].embedded_op)
```

Changing the depolarization parameterization to `"lindblad"` makes the `Gxpi2` gate go back to taking only 1 parameter since its noise is represented by a `LindbladErrorgen` with mode `"D"`.

```{code-cell} ipython3
cloudmodel = pygsti.models.create_cloud_crosstalk_model(
    pspec,
    depolarization_strengths={'Gxpi2': 0.1},
    depolarization_parameterization='lindblad',
    stochastic_error_probs={'Gypi2': (0.03, 0.01, 0.03)},  # local noise on a gate
    stochastic_parameterization='lindblad'
)

print(cloudmodel.num_params, "params, 1 from lindblad D-mode error-generator, 3 stochastic error-generator rates")
```

By changing the mode of Lindblad error coefficients we can:
1. dictate what types of errors are allowed in the error generator (and could be possibly added later on)
2. alter whether the Pauli stochastic error rates are constrained to be postive or not

```{code-cell} ipython3
# H+S => H and S type elementary error generators are allowed; S-type coefficients must be positive
cloudmodel = pygsti.models.create_cloud_crosstalk_model(
    pspec,
    lindblad_error_coeffs={
        'Gcnot': { ('H','ZZ'): 0.02, ('S','XX'): 0.02 }
    },
    lindblad_parameterization='H+S'
)
print(cloudmodel.num_params, "params")
```

```{code-cell} ipython3
# H+s => H and S type elementary error generators are allowed; S-type coefficients can be postive or negative (no constraint)
cloudmodel = pygsti.models.create_cloud_crosstalk_model(
    pspec,
    lindblad_error_coeffs={
        'Gcnot': { ('H','ZZ'): 0.02, ('S','XX'): 0.02 }
    },
    lindblad_parameterization='H+s'
)
print(cloudmodel.num_params, "params")
```

```{code-cell} ipython3
# H => Only H-type elementary error generators are allowed
cloudmodel = pygsti.models.create_cloud_crosstalk_model(
    pspec,
    lindblad_error_coeffs={
        'Gcnot': { ('H','ZZ'): 0.02 }
    },
    lindblad_parameterization='H'
)
print(cloudmodel.num_params, "params")
```

```{code-cell} ipython3
# CPTP => All elementary error-generator types are allowed; Pauli (diagonal) S-type coefficients must be positive.
# Note that for CPTP types, S-type coefficients are indexed by 2 Paulis instead of 1.
cloudmodel = pygsti.models.create_cloud_crosstalk_model(
    pspec,
    lindblad_error_coeffs={
        'Gcnot': { ('H','ZZ'): 0.02, 
                   ('S', 'XX', 'XX'): 0.05, ('S', 'XY', 'XY'): 0.05, ('S', 'YX', 'YX'): 0.05,
                   ('S', 'XX', 'XY'): 0.01 + 0.01j, ('S', 'XY', 'XX'): 0.01 - 0.01j,
                   ('S', 'XX', 'YX'): 0.01 + 0.0j, ('S', 'YX', 'XX'): 0.01 - 0.0j,
                   ('S', 'XY', 'YX'): 0.01 - 0.005j, ('S', 'YX', 'XY'): 0.01 + 0.005j}
    },
    lindblad_parameterization='CPTP'
)
print(cloudmodel.num_params, "params because 1 H-type and a Hermitian 3x3 block of S-type coefficients")
```

## Weight-limited CPTPLND post-gate errors

`LindbladErrorgen.from_weight_limited` constructs a CPTP Lindblad error generator whose selected elementary-error support remains fixed when its parameters change. The stochastic directions index a Kossakowski matrix $K$, with $K_{ii}=S_i$ and $K_{ij}=C_{ij}-iA_{ij}$ for an oriented pair $(i,j)$. Its sparse Cholesky factor makes $K$ positive semidefinite for every parameter vector.

Direction weight counts the subsystems on which the operator is nonidentity. A C/A pair's weight counts the **union** of its two supports. Thus `C(XII, IXX)` has weight three, even though each direction has weight at most two. Each C/A endpoint must also be an allowed S direction. Limits are specified independently by sector; an omitted limit leaves that sector unrestricted.

For example, create independent post-gate errors with one-body Hamiltonian terms and S/C/A terms of weight at most two:

```{code-cell} ipython3
import numpy as np
from pygsti.baseobjs import QubitSpace
from pygsti.modelmembers.operations import (
    LindbladErrorgen, ExpErrorgenOp, StaticUnitaryOp, ComposedOp
)

space = QubitSpace(3)
errorgen = LindbladErrorgen.from_weight_limited(
    space, max_weights=dict(H=1, S=2, C=2, A=2), initial_stochastic_rate=1e-4
)

# Use a positive stochastic diagonal with smaller factor perturbations for a
# well-conditioned example. S/C/A factor parameters are not physical rates.
rng = np.random.default_rng(1234)
errorgen.from_vector(errorgen.to_vector() + rng.normal(scale=.001, size=errorgen.num_params))

ideal_gate = StaticUnitaryOp(
    np.kron(np.array([[0., 1.], [1., 0.]]), np.eye(4)), state_space=space
)
noisy_gate = ComposedOp([ideal_gate, ExpErrorgenOp(errorgen)])
# ComposedOp lists factors in application order: exp(errorgen) @ ideal_gate.
print(errorgen.num_params, "parameters")  # 657: 9 H + 648 sparse S/C/A

# SPAM has independent noise channels with exactly the same weight limits.
from pygsti.models import ExplicitOpModel
from pygsti.modelmembers.states import ComputationalBasisState, ComposedState
from pygsti.modelmembers.povms import ComputationalBasisPOVM, ComposedPOVM


def weight_limited_noise(max_weights, seed):
    generator = LindbladErrorgen.from_weight_limited(
        space, max_weights, initial_stochastic_rate=1e-4
    )
    vector = generator.to_vector()
    generator.from_vector(vector + np.random.default_rng(seed).normal(scale=.001, size=len(vector)))
    return ExpErrorgenOp(generator)


limits = dict(H=1, S=2, C=2, A=2)
weight_model = ExplicitOpModel(space, basis='pp')
weight_model.preps['rho0'] = ComposedState(
    ComputationalBasisState([0, 0, 0], state_space=space), weight_limited_noise(limits, 21)
)
weight_model.povms['Mdefault'] = ComposedPOVM(
    weight_limited_noise(limits, 22), ComputationalBasisPOVM(3, state_space=space), mx_basis='pp'
)
weight_model.operations['Gx'] = noisy_gate
assert weight_model.num_params == 3 * 657
print(weight_model.probabilities(('Gx', 'Gx')))

sca_block = errorgen.coefficient_blocks[-1]
K = sca_block.kossakowski_matrix()  # Optional dense inspection, in basis_element_labels order.
print(K.shape)                     # (36, 36)
assert np.linalg.eigvalsh(K)[0] >= -1e-12
```

Preparation noise acts after the reference state; measurement noise acts before the reference POVM, so its adjoint acts on effects. The gate and both SPAM channels have independent parameters in the same hypothesis class. Construct a separate error generator for each independently parameterized gate. Model copying and nice serialization retain the selected support and the factor parameterization.

The weight-two case admits a perfect elimination order. All weight-one directions form a clique. For each two-subsystem support, its weight-two directions and the weight-one directions on those two subsystems form another clique. Eliminate all weight-two directions first, then the weight-one directions. Every eliminated vertex's remaining neighbors form a clique, so factorization adds no fill. On three qubits, this leaves 648 real S/C/A parameters rather than the 1296 of a full Hermitian matrix on the same 36 stochastic directions.

By default, the constructor chooses a fill-free chart for a chordal graph with matching C/A masks, or C-only real factors. For S+C errors, pass `errorgen_types=('S', 'C')`; for S-only errors, pass `('S',)`. Nonchordal patterns, independent C/A limits, and A-only off-diagonals automatically use completed Cholesky. Set `param_mode='cholesky'` to require the fill-free chart, or `param_mode='cholesky completion'` to request completion explicitly.

The default operator basis is the canonical PP/GM tensor product for the state space. Qudits are supported; supply a compatible matrix basis such as `mx_basis='gm'` for a qutrit. The fill-free chart starts at zero and includes singular PSD matrices. As with dense CPTPLND, its quadratic S/C/A derivatives vanish at zero factor parameters. `sample_errorgen_rates` remains available to draw physical initial rates, while this constructor supplies the parameterization that preserves zeros during optimization.

### Higher weights and independent C/A limits

Completion makes factor entries at forbidden matrix components **dependent** on the free entries. Write $L=DT$, with positive diagonal $D$ and unit-lower-triangular $T$. For $i>j$, a forbidden complex entry is enforced by

$$T_{ij}=-\sum_{a<j}T_{ia}\overline{T_{ja}}.$$

If only the real or imaginary component is forbidden, apply the corresponding part of this equation. The resulting $K=DTT^\dagger D$ preserves the requested zeros. Symbolic fill entries do not add free coefficients or relax the weight limits, and completion does not divide by small pivots.

Completion uses positive pivots: diagonal parameters are $\log D_{ii}$ and free off-diagonal parameters are the allowed components of $T_{ij}=L_{ij}/L_{ii}$. These row-normalized coordinates make diagonal congruences a shift of the log pivots with unchanged off-diagonal coordinates. The map covers the positive-definite interior; singular PSD matrices and exact zero noise are outside this chart. Numerical conditioning can still deteriorate near the boundary.

`completion_normalization='column'` selects the alternative $L=TD$ chart. It uses pivot-dividing completion and can be better conditioned when approaching a rank-deficient matrix with nonzero diagonal entries. Row normalization is better suited to shrinking entire matrix rows and columns. Neither chart has uniformly good conditioning at every PSD boundary; both describe the same positive-definite coefficient class.

Switch charts by constructing a generator with the desired normalization and transferring its elementary coefficients with `set_coefficients`. Raw parameter vectors have chart-specific meanings.

The default completion initialization is a diagonal Kossakowski matrix with S rates given by `DEFAULT_COMPLETION_STOCHASTIC_RATE` in `pygsti.modelmembers.operations.lindbladerrorgen`, zero C/A, and zero H. Supply `initial_stochastic_rate` to choose a different positive rate; an explicit zero raises `ValueError`. For example, use different C and A limits on three qubits:

```{code-cell} ipython3
from pygsti.baseobjs import QubitSpace
from pygsti.modelmembers.operations import LindbladErrorgen, ExpErrorgenOp

interior_errorgen = LindbladErrorgen.from_weight_limited(
    QubitSpace(3), dict(H=1, S=2, C=2, A=3),
    initial_stochastic_rate=1e-4
)
interior_block = interior_errorgen.coefficient_blocks[-1]
assert interior_block.num_params == 972
assert interior_errorgen.num_params == 981  # 9 H + 972 S/C/A
error_channel = ExpErrorgenOp(interior_errorgen)

interior_limits = dict(H=1, S=2, C=2, A=3)
interior_model = ExplicitOpModel(space, basis='pp')
interior_model.preps['rho0'] = ComposedState(
    ComputationalBasisState([0, 0, 0], state_space=space), weight_limited_noise(interior_limits, 31)
)
interior_model.povms['Mdefault'] = ComposedPOVM(
    weight_limited_noise(interior_limits, 32), ComputationalBasisPOVM(3, state_space=space), mx_basis='pp'
)
interior_model.operations['Gx'] = ComposedOp([ideal_gate.copy(), error_channel])
assert interior_model.num_params == 3 * 981
print(interior_model.probabilities(('Gx', 'Gx')))
```

Common S/C/A weight three on four or more qubits also selects completion automatically. Dense density-matrix simulation still constructs global superoperator data; coefficient sparsity does not remove that state-space cost. The full four-qubit weight-three coefficient block has 174 directions and 14562 real parameters.

Completed blocks provide analytic derivatives, Hessians, Torch reconstruction, and model persistence. Their parameter map is nonpolynomial, so polynomial term simulation raises `NotImplementedError`; use matrix simulation or Torch. The resolved elimination ordering is serialized to preserve the meaning of the parameters across installations.

## Correlated dephasing Kossakowski graphs

`LindbladErrorgen.from_correlated_dephasing` selects commuting diagonal directions and constrains their real Kossakowski matrix to a specified graph. By default it selects the one-body diagonal PP/GM directions in subsystem order: $Z_i$ for each qubit, or all $d-1$ diagonal generators for a dimension-$d$ qudit. The generator is

$$\mathcal L(\rho)=-\tfrac12\sum_{ab}K_{ab}[F_a,[F_b,\rho]],\qquad K\succeq0.$$

The constructor uses S and C, with no H or A. These are real Gaussian Markovian phase-noise covariances, not a colored-noise spectrum or the entire quantum pure-dephasing family. The graph encodes covariance zeros; it does not encode zeros in an inverse covariance or follow automatically from device layout.

For example, use a four-cycle with positive and negative correlations:

```{code-cell} ipython3
from pygsti.baseobjs import QubitSpace
from pygsti.modelmembers.operations import LindbladErrorgen, ExpErrorgenOp
import numpy as np

phase_directions = ('ZIII', 'IZII', 'IIZI', 'IIIZ')
phase_edges = [(phase_directions[i], phase_directions[(i + 1) % 4]) for i in range(4)]
phase_covariance = np.diag([.028, .033, .025, .030])
for (i, j), value in zip(((0, 1), (1, 2), (2, 3), (0, 3)), (.007, -.006, .005, .004)):
    phase_covariance[i, j] = phase_covariance[j, i] = value
phase_errorgen = LindbladErrorgen.from_correlated_dephasing(
    QubitSpace(4), phase_edges, initial_covariance=phase_covariance
)
assert phase_errorgen.num_params == 8  # Four diagonal entries and four edges.
phase_channel = ExpErrorgenOp(phase_errorgen)

# This example also includes noisy SPAM in the same eight-parameter family.
from pygsti.baseobjs import Basis
from pygsti.modelmembers.states import StaticState
from pygsti.modelmembers.povms import StaticPOVMEffect, UnconstrainedPOVM
from pygsti.tools.basistools import stdmx_to_vec

phase_space = QubitSpace(4)
phase_prep = ExpErrorgenOp(LindbladErrorgen.from_correlated_dephasing(
    phase_space, phase_edges, initial_covariance=.3 * phase_covariance
))
phase_meas = ExpErrorgenOp(LindbladErrorgen.from_correlated_dephasing(
    phase_space, phase_edges, initial_covariance=.2 * phase_covariance
))
plus = StaticState(stdmx_to_vec(np.ones((16, 16)) / 16, 'pp'), basis='pp', state_space=phase_space)
xx = Basis.cast('PP', 256)['XXII']
parity_povm = UnconstrainedPOVM([
    (label, StaticPOVMEffect(stdmx_to_vec((np.eye(16) + sign * xx) / 2, 'pp'),
                            basis='pp', state_space=phase_space))
    for label, sign in (('+', 1), ('-', -1))
])
phase_model = ExplicitOpModel(phase_space, basis='pp')
phase_model.preps['rho0'] = ComposedState(plus, phase_prep)
phase_model.povms['Mdefault'] = ComposedPOVM(phase_meas, parity_povm, mx_basis='pp')
phase_model.operations['Gi'] = phase_channel
assert phase_model.num_params == 24
print(phase_model.probabilities(('Gi',)))
```

The cycle is nonchordal and automatically uses completion. Pass `correlation_edges=()` for independent noise or omit it for a dense covariance. An explicit `basis_element_labels` selects other diagonal directions, including higher-body strings such as `('ZI', 'IZ', 'ZZ')`. With supplied covariance, its matrix order is the selected label order. Forbidden entries must be zero, and invalid covariances are rejected rather than projected.

All parameter updates preserve the graph and positivity. Without `initial_covariance`, initialization and chart keywords behave as in `from_weight_limited`: zero for fill-free, positive diagonal for completion, and configurable `initial_stochastic_rate` and `completion_normalization`. Completion requires a numerically positive-definite covariance after diagonal scaling; singular inputs are rejected. Tiny overall rates and strongly unequal diagonal scales alone do not cause that numerical-rank rejection. Dense/chordal fill-free graphs can represent collective rank-one covariance exactly.

For unnormalized $Z_i$, each matrix unit $|x\rangle\langle y|$ decays at rate $(z(x)-z(y))^TK(z(x)-z(y))/2$. Thus individual Ramsey means depend only on $K_{ii}$; pair correlations are needed to identify off-diagonal entries. A graph with $n$ vertices and $|E|$ edges has $n+|E|$ real parameters per channel, versus $n(n+1)/2$ for a dense matrix on the same directions. These are different hypotheses even though all directions have support weight one. The [correlated-dephasing experiment](../examples/CorrelatedDephasing.md) jointly fits independent gate, preparation, and pre-measurement channels in each of these families, using composed Lindblad SPAM as above.

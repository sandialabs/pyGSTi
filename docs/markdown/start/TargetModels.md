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

# Model packs

PyGSTi comes shipped with a number of "model packs", each of which contains a small (few-qubit) noise model and related information.  This related information is typically *derived* from the base noise model, but it's computation is nontrivial or inconvenient.  The convenience of having meta-data packaged together with the base model is the sole reason for model packs in pyGSTi. 

Model-packs look like modules and you import them from `pygsti.modelpacks` in the usual way.

Here's an example ("smq" stands for "standard multi-qubit"):

```{code-cell} ipython3
from pygsti.modelpacks import smq1Q_XYI
```

## Target model
Each model pack defines a 1- or 2-qubit model and number of related quantites.  Sometimes you'll just want to use the `Model`, and importing a model pack is just a convenient way to create a commonly used model for 1 or 2 qubits (the `smq1Q_XYI` module is for the 1-qubit model containing *Idle*, $X(\pi/2)$ and $Y(\pi/2)$ gates).  A model pack's model always contains *perfect* (unitary) operations, and is called the *target model* because often times this is the model you wish described your system.  You can get a copy of it by using the `target_model` function:

```{code-cell} ipython3
mdl = smq1Q_XYI.target_model()
print(mdl)
```

Now let's review a few things about this target model:

1. **It's a *copy*.**  If you modify it, it won't change what's stored in the model pack.  This means that you don't need to add a `.copy()` (e.g. `mdl = smq1Q_XYI.target_model().copy()`).


2. **It's *fully parameterized*.**  By default, `target_model()` returns a fully-parameterized `Model`, meaning that each of its operations contain an independent parameter for each one of their elements.  If you want a different parameterization, such as a TP-constrained model, you can specify this as an argument:

```{code-cell} ipython3
mdl_TP = smq1Q_XYI.target_model("full TP")
```

3. **It has gate names that are tuples of the form (name, *qubits*).**  The gate names (keys of the models `.operations` dictionary) use pyGSTi's multi-qubit labeling convention, e.g. `("Gx",0)`, `("Gx",1)`, or `("Gcnot",0,1)`.  Note that the label for an idle is just an empty tuple, indicating an empty circuit layer.

## General additional quantities
For convenience model packs contain `description` and `gates` members giving a simple text description of the pack's target model and its gates:

```{code-cell} ipython3
smq1Q_XYI.description
```

```{code-cell} ipython3
smq1Q_XYI.gates
```

## Quantities for running GST
In addition to a target `Model`, a GST-type model pack (most of them are this type) contains a number of `Circuit` list generating functions used for running gate set tomography (GST).  All of these functions (like `target_model`) take a `qubit_labels` argument that can specify a non-default set of qubit labels to use.  The circuit-list functions include:
- preparation fiducials: `prep_fiducials`
- measurement (effect) fiducials: `meas_fiducials`
- germ sequences: `germs`
 - this function has an additional `lite` argument that, when True (the default) gives a shorter list of germ circuits that amplify all the errors in the target model to *first order*.  This is usually all that is needed to achieve the high-accuracy typically desired from GST results, and so we recommend starting with this list of germs since it's shorter.  When `lite=False` a longer list of germ circuits is returned that amplify all the errors in the target model to *higher orders*.  Although typically unnecessary, this "paranoid" set of germs can be particularly helpful when you expect and don't care about some departures (errors) from the target model.
- fiducial pair reductions (see the [circuit reduction tutorial](../guides/gst/FewerCircuits) for more details):
 - `global_fid_pairs` is not a function, but just a list of 2-tuples giving the indices (within `prep_fiducials` and `meas_fiducials`) of the fiducial circuits to keep when implementing global fiducial pair reduction.
 - `pergerm_fidpair_dict` and `pergerm_fidpair_dict_lite` are dictionaries of lists-of-2-tuples giving the indices of the fiducial circuits to keep on a per-germ basis (dict keys are germ circuits) when implementing per-germ fiducial pair reduction.
 
Here are some examples:

```{code-cell} ipython3
smq1Q_XYI.prep_fiducials()
```

```{code-cell} ipython3
smq1Q_XYI.pergerm_fidpair_dict_lite()
```

## Quantities for running RB
Standard Clifford-based randomized benchmarking (RB) requires knowing how to "compile" the elements of the Clifford group from your native gate set.  Most model packs also contain a `clifford_compilation` function that returns a dictionary describing this compilation, which can in turn be used when running Clifford RB (see the [Clifford RB tutorial](../guides/rb/CliffordRB) for more info).

```{code-cell} ipython3
smq1Q_XYI.clifford_compilation()
```

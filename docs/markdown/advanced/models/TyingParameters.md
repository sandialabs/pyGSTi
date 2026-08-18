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

# Parameter labels, bounds, and tying

Every parameter of a pyGSTi model carries a label. Those labels are what let you reach into a model and say something about one specific number: bound it to an interval, or tie it to another parameter so the two always move together. This page covers all three, in that order. The techniques apply to any model type, so the choice of model below is arbitrary; they're most useful when you're building your own objects (see the [custom operator tutorial](CustomOperators)) whose parameters have restrictions.

The examples below use a 1-qubit `H+s` model, whose parameterization is small enough to print in full.

```{code-cell} ipython3
import pygsti
import numpy as np
from pygsti.modelpacks import smq1Q_XY as std

mdl = std.target_model("H+s")
```

## Getting parameter labels

A `Model`'s parameters have corresponding labels, which you can get at in a few ways. Individual operators have labeled parameters too. An `OpModel` (an `ExplicitOpModel` or `ImplicitOpModel`, say) sets default parameter labels from the labels of the operators it contains, but the model's parameters can then vary independently of them.

```{code-cell} ipython3
# print the raw labels, straight up
mdl.parameter_labels
```

Raw labels are usually tuples of `(op_label, description)`. You can overwrite any of them:

```{code-cell} ipython3
# model parameters can be set to arbitrary user-defined values
mdl.set_parameter_label(index=0, label="My favorite parameter")
```

```{code-cell} ipython3
# Model parameters in a nice format for printing
mdl.parameter_labels_pretty
```

The "pretty" form flattens each tuple into a single string, joining the pieces with `": "`. Either form works where a parameter label is expected, but don't mix them inside one call. `collect_parameters`, for example, resolves its entire list against `parameter_labels` and only falls back to `parameter_labels_pretty` wholesale, so a list mixing raw tuples with pretty strings raises `KeyError`. The same goes for mixing integer indices with labels of either kind.

```{code-cell} ipython3
# For a single operator: you can get its "local" parameter labels
# (in general different from the model's parameter labels)
mdl.operations[('Gxpi2',0)].parameter_labels
```

```{code-cell} ipython3
# The parameters of all the operators, with mappings to non-default model parameters
mdl.print_parameters_by_op()
```

## Bounding a parameter's values

Optimizers respect per-parameter bounds, so a bound is the way to keep a fit from wandering into values you know are unphysical or uninteresting. Suppose you want to hold the coherent Z error on the $X(\pi/2)$ gate between 0 and 0.2. Bounds that survive a rebuild live on model members rather than on the model itself (you can set them on the model, but see the caveat at the end of the next section), so first find the member that owns the parameter.

```{code-cell} ipython3
# Here's the X(pi/2) gate:
print(mdl.operations[('Gxpi2', 0)])
```

```{code-cell} ipython3
# this is the error generator whose parameters we want to bound
eg = mdl.operations[('Gxpi2', 0)].factorops[1].errorgen
for i, lbl in enumerate(eg.parameter_labels):
    print(i, lbl)
```

The parameter we want has index 2. Currently the bounds are `None`, which means there aren't any:

```{code-cell} ipython3
print(eg.parameter_bounds)
```

Set the `parameter_bounds` attribute of a model member to a 2D NumPy array of shape `(num_params, 2)`, whose rows are `(min, max)` for each parameter. Use `numpy.inf` and `-numpy.inf` where you don't want one or both bounds.

```{code-cell} ipython3
bounds = np.empty((eg.num_params, 2), 'd')
bounds[:, 0] = -np.inf  # initial lower bounds
bounds[:, 1] = np.inf   # initial upper bounds
bounds[2, :] = (0, 0.2) # bounds for "Z Hamiltonian error coefficient" parameter
eg.parameter_bounds = bounds
```

Setting bounds on a member marks the containing model as needing a rebuild, but you don't have to trigger that rebuild yourself. Reading `mdl.parameter_bounds` cleans up the model's parameter vector first, so the member's bounds are already in place the first time you look. The model's bounds array has one row per model parameter, and only the row belonging to our error coefficient is finite:

```{code-cell} ipython3
model_bounds = mdl.parameter_bounds
bounded = np.flatnonzero(np.isfinite(model_bounds).any(axis=1))
for i in bounded:
    print(i, mdl.parameter_labels_pretty[i], model_bounds[i])
```

From here on, an optimization of `mdl` will keep that parameter inside the interval.

## Tying parameters together

`collect_parameters` replaces several parameters with a single one, so that everything that used to read the originals now reads the same number. This is how you impose "these errors are equal" without writing a new operator class.

The next cell prints a warning about model-level parameter bounds being overwritten. It fires on any rebuild where the model is holding a bounds array, and the model is holding one here only because reading `mdl.parameter_bounds` above cached the member's bounds onto it. Nothing is lost in that case, since the rebuild re-reads the same bounds off the member. The warning is worth heeding when you set bounds on the model itself; the last paragraph of this section says why.

```{code-cell} ipython3
mdl.collect_parameters([ (('Gxpi2',0), 'X Hamiltonian error coefficient'),
                         (('Gypi2',0), 'Y Hamiltonian error coefficient')],
                       new_param_label='Over-rotation')
```

```{code-cell} ipython3
# Using "pretty" labels works too:
mdl.collect_parameters(['Gxpi2:0: Y stochastic coefficient',
                        'Gxpi2:0: Z stochastic coefficient' ],
                       new_param_label='Gxpi2 off-axis stochastic')
```

```{code-cell} ipython3
# You can also use integer indices, and parameter labels can be tuples too.
mdl.collect_parameters([3,4,5], new_param_label=("rho0", "common stochastic coefficient"))
```

Each call shrinks the parameter vector:

```{code-cell} ipython3
# There are now fewer parameters
mdl.parameter_labels_pretty
```

```{code-cell} ipython3
# And you can see how they're wired up for each op:
mdl.print_parameters_by_op()
```

One caveat worth taking seriously: after a `collect_parameters` call, treat the model's parameter vector (the result of `to_vector()`) as having an entirely new format. Untouched parameters do not keep their old indices, so any index you cached beforehand is stale. The bounds set above survive, because they were set on the member and get re-propagated on every rebuild. Bounds set directly at the model level with `Model.set_parameter_bounds` do not survive; a rebuild overwrites them with whatever the members say.

## Un-tying parameters

The reverse operation promotes each use of a shared parameter back to an independent one. The new parameters get their labels from the operations that use them.

```{code-cell} ipython3
mdl.uncollect_parameters('Gxpi2 off-axis stochastic')
```

```{code-cell} ipython3
mdl.print_parameters_by_op()
```

This is not an undo. `uncollect_parameters` splits apart every occurrence of the parameter, including ones that were shared before you ever called `collect_parameters`, and the resulting indices differ from the ones you started with.

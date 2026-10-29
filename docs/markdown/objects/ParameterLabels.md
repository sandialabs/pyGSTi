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

# Parameter Labels
This tutorials show how model parameters are labelled, and how this can be used to create more complex parameterizations for a model.

```{code-cell} ipython3
import pygsti
import numpy as np
from pygsti.modelpacks import smq1Q_XY as std
from pygsti.baseobjs import Label

mdl1 = std.target_model("H+s")  # choose a H+s model because it has a simple parameterization
```

## Getting parameter labels
A `Model`'s parameters have corresponding labels, which can be accessed in a variety of ways.  Individual operators also have labeled parameters.  An `OpModel` (e.g. an `ExplicitModel` or `ImplicitModel`) sets default parameter labels based on the parameter labels of its contained operators, but the model's parameters can vary independently.

```{code-cell} ipython3
# print the raw labels, straight up
mdl1.parameter_labels
```

```{code-cell} ipython3
# model parameters can be set to arbitrary user-defined values
mdl1.set_parameter_label(index=0, label="My favorite parameter")
```

```{code-cell} ipython3
# Model parameters in a nice format for printing
mdl1.parameter_labels_pretty
```

```{code-cell} ipython3
# For a single operator: you can get it's "local" parameter labels (in general different from the model's parameter labels)
mdl1.operations[('Gxpi2',0)].parameter_labels
```

```{code-cell} ipython3
# The parameters of all the operators, with mappings to non-default model parameters 
mdl1.print_parameters_by_op()
```

## Collecting parameters
You can combined multiple parameters into one using the `collect_parameters` method.  This effectively ties the values for all the original parameters together.

```{code-cell} ipython3
mdl1.collect_parameters([ (('Gxpi2',0), 'X Hamiltonian error coefficient'),
                          (('Gypi2',0), 'Y Hamiltonian error coefficient')],
                        new_param_label='Over-rotation')
```

```{code-cell} ipython3
# Using "pretty" labels works too:
mdl1.collect_parameters(['Gxpi2:0: Y stochastic coefficient',
                         'Gxpi2:0: Z stochastic coefficient' ],
                        new_param_label='Gxpi2 off-axis stochastic')
```

```{code-cell} ipython3
# You can also use integer indices, and parameter labels can be tuples too.
mdl1.collect_parameters([3,4,5], new_param_label=("rho0", "common stochastic coefficient"))
```

```{code-cell} ipython3
# There are now fewer parameters
mdl1.parameter_labels_pretty
```

```{code-cell} ipython3
# And you can see how they're wired up for each op:
mdl1.print_parameters_by_op()
```

## Un-collecting parameters
You can also reverse the above process and "un-collect" a parameter so that one parameter gets replaced my multiple independent ones.

```{code-cell} ipython3
mdl1.uncollect_parameters('Gxpi2 off-axis stochastic')
```

```{code-cell} ipython3
mdl1.print_parameters_by_op()
```

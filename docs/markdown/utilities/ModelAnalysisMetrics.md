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

# Model analysis functions
One reason QIP models are useful is that they provide a compact description of a device that can be compared to other such descriptions.  For example, the process fidelity between an estimated process matrix and the ideal one is a common metric for assessing how "good" a QIP operatio is.  PyGSTi has a number of functions for comparing process matrices (often contained within models, e.g. in the `.operations` dictionary of an `ExplicitOpModel`) or for comparing two `Model` objects.  Many of these functions live in the `pygsti.report.reportables` module.  First, let's create two models to compare (for more information on building models see [this tutorial](../objects/ExplicitModel)).

```{code-cell} ipython3
import pygsti
from pygsti.report import reportables as rptbl
from pygsti.modelpacks import smq1Q_XYI # "stock" information about the X(pi/2), Y(pi/2), Idle 1-qubit system.

#Get two models to compare.
ideal_model = smq1Q_XYI.target_model()
noisy_model = ideal_model.depolarize(op_noise=0.01, spam_noise=0.01)
print(list(ideal_model.operations.keys()))
```

In these models the "layer operations" are just 1-qubit operators and so can represented by $4\times 4$ matrices.  Indeed, the elements of each model's `.operations` dictionary can be treated as a Numpy array with shape `(4,4)`.  This makes it easy to demonstrate a few of the common metrics found in `pygsti.report.reportables`.  Many of these functions need to know the basis of the matrix or vector arguments being supplied; for more information on bases, see [this tutorial on the Basis object](../objects/MatrixBases).

```{code-cell} ipython3
basis = pygsti.baseobjs.Basis.cast("pp",4) # 1-qubit Pauli basis (2x2 matrices)
print("State Fidelity = ", float(rptbl.vec_fidelity(ideal_model['rho0'],noisy_model['rho0'],basis)))
print("Process Fidelity = ", float(rptbl.entanglement_fidelity(ideal_model[('Gxpi2',0)], noisy_model[('Gxpi2',0)], basis)))
print("Diamond Distance = ", rptbl.half_diamond_norm(ideal_model[('Gxpi2',0)], noisy_model[('Gxpi2',0)], basis))
print("Frobenius Distance = ", rptbl.frobenius_diff(ideal_model[('Gxpi2',0)], noisy_model[('Gxpi2',0)], basis))
```

While the above metrics are the usually the most familiar, all of them suffer because they are not *gauge invariant*.  Gauge Invariance is the property that the metric is unchanged when either of the models being compared is replaced with another model that predicts exactly the same circuit outcomes as the replaced model (but need not have exactly the same operation matrices).  For more information on gauge invariance, see XXXCITEXXX.  The practical takeaway is that it's much easier to compare two models using gauge invariant metrics (using gauge variant metrics like those above one needs to perform an extra *gauge optimization* step which in many cases is difficult or even ill-defined; for more information on gauge optimization in pyGSTi see [this tutorial](GaugeOpt)).  Some gauge invariant metrics are:

```{code-cell} ipython3
print("Eigenvalue Process Fidelity = ", float(rptbl.eigenvalue_entanglement_infidelity(ideal_model[('Gxpi2',0)], noisy_model[('Gxpi2',0)], basis)))
print("Diamond Distance = ", rptbl.eigenvalue_diamondnorm(ideal_model[('Gxpi2',0)], noisy_model[('Gxpi2',0)], basis))
```



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

# Metrics
PyGSTi contains implementation of common ways to compare quantum processes and models.  You may just want to `import pygsti` just for this functionality, as many of the functions below act on standard NumPy arrays.  Here are some of the most common functions (this tutorial is <font style="color:red">under construction, and we plan to expand it in future releases.  We apologize for it's current brevity</font>.

Let's begin by getting some gate (process) matrices for several simple 1-qubit operations.  Note that `Gx`, `Gy` and `Gi` below are superoperator matrices in the Pauli basis - they're $4 \times 4$ *real* matrices.  We do this for a model pack (see the [model packs tutorial](../objects/ModelPacks)) and a version of this model with slightly rotated gates.

```{code-cell} ipython3
import pygsti.tools as tls
import pygsti.report.reportables as rptbls
from pygsti.modelpacks import smq1Q_XYI as std
import numpy as np

mdl = std.target_model()
Gx = mdl[('Gxpi2',0)].to_dense()
Gy = mdl[('Gypi2',0)].to_dense()
Gi = mdl[()].to_dense()

mdl_overrot = mdl.rotate( (0.1,0,0) )
Gx_overrot = mdl_overrot[('Gxpi2',0)].to_dense()
Gy_overrot = mdl_overrot[('Gypi2',0)].to_dense()
Gi_overrot = mdl_overrot[()].to_dense()

tls.print_mx(Gx_overrot)
```

## Process matrix comparisons
### Fidelities

```{code-cell} ipython3
rptbls.entanglement_infidelity(Gx, Gx_overrot, 'pp')
```

```{code-cell} ipython3
rptbls.avg_gate_infidelity(Gx, Gx_overrot, 'pp')
```

```{code-cell} ipython3
rptbls.eigenvalue_entanglement_infidelity(Gx, Gx_overrot, 'pp')
```

```{code-cell} ipython3
rptbls.eigenvalue_avg_gate_infidelity(Gx, Gx_overrot, 'pp')
```

### Diamond distance

```{code-cell} ipython3
rptbls.half_diamond_norm(Gx, Gx_overrot, 'pp')
```

```{code-cell} ipython3
rptbls.eigenvalue_diamondnorm(Gx, Gx_overrot, 'pp')
```

### Unitarity

```{code-cell} ipython3
tls.unitarity(Gx_overrot)
```

### Jamiolkowski trace distance

```{code-cell} ipython3
rptbls.jtrace_diff(Gx, Gx_overrot, 'pp')
```

## State comparisons

### State fidelity

```{code-cell} ipython3
rhoA = tls.ppvec_to_stdmx(mdl['rho0'].to_dense())
rhoB = np.array( [ [0.9,   0],
                   [ 0,  0.1]], complex)
tls.fidelity(rhoA, rhoB)
```

```{code-cell} ipython3

```

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

# Metrics for comparing models and processes

A model is a compact description of a device, which means you can compare it to another such description and get a number out. The process fidelity between an estimated process matrix and its ideal counterpart is the usual example: it is how most people answer "how good is this gate?". PyGSTi implements the common metrics for comparing quantum processes, states and whole models. Most of them live in `pygsti.report.reportables` and `pygsti.tools.optools`, and most act on plain NumPy arrays, so you can import pyGSTi for this functionality alone without building an experiment design or running a fit.

```{warning}
This page is under construction. It covers the metrics people reach for most often, not the full contents of `pygsti.report.reportables`.
```

## Setup

Build three models: the ideal 1-qubit $X(\pi/2)$, $Y(\pi/2)$, idle model from a model pack, a depolarized version of it, and a version with a coherent over-rotation on every gate. Two noise types are worth carrying through the page because several metrics tell them apart. For more on building models see [Models](../models/Models); for the model packs themselves see [target models](../../start/TargetModels).

```{code-cell} ipython3
import numpy as np
import pygsti
import pygsti.tools as tls
from pygsti.tools import optools as opt
from pygsti.report import reportables as rptbl
from pygsti.modelpacks import smq1Q_XYI

ideal_model = smq1Q_XYI.target_model()
noisy_model = ideal_model.depolarize(op_noise=0.01, spam_noise=0.01)
overrot_model = ideal_model.rotate((0.1, 0, 0))

print(list(ideal_model.operations.keys()))
```

The layer operations here act on a single qubit, so each is represented by a $4 \times 4$ *real* superoperator matrix in the Pauli basis. The entries of a model's `.operations` dictionary are `LinearOperator` objects, not arrays; call `.to_dense()` to get a NumPy array, and `.to_dense(on_space='minimal')` for state preparations and effects. Pull out the gates you will compare:

```{code-cell} ipython3
Gx = ideal_model[('Gxpi2', 0)].to_dense()
Gx_depol = noisy_model[('Gxpi2', 0)].to_dense()
Gx_overrot = overrot_model[('Gxpi2', 0)].to_dense()

tls.print_mx(Gx_overrot)
```

Nearly every metric function needs to know which basis its matrix and vector arguments are written in. You can pass the string `'pp'`, or an explicit `Basis` object; see [Bases](../../advanced/conventions/Bases) for what the choice means.

```{code-cell} ipython3
basis = pygsti.baseobjs.Basis.cast("pp", 4)  # 1-qubit Pauli basis (2x2 matrices)
```

## Fidelity and infidelity

Entanglement fidelity (also called process fidelity) and average gate fidelity are the two conventions in circulation. They differ by a dimension-dependent factor, so say which one you mean when you report a number. `pygsti.tools.optools` has both the fidelities and their infidelity counterparts; `pygsti.report.reportables` carries only the infidelities, so reach for `optools` when you want a fidelity.

```{code-cell} ipython3
print("Process fidelity (depolarized) =", opt.entanglement_fidelity(Gx, Gx_depol, basis))
print("Entanglement infidelity (over-rotated) =", rptbl.entanglement_infidelity(Gx, Gx_overrot, basis))
print("Average gate infidelity (over-rotated) =", rptbl.avg_gate_infidelity(Gx, Gx_overrot, basis))
```

## Diamond distance

The diamond norm of the difference between two processes bounds how distinguishable they are under any experiment, including ones that use entangled ancillas. `half_diamond_norm` returns half that norm, which is the convention used throughout pyGSTi's reports. It solves a semidefinite program, so it is much slower than the fidelities above.

```{code-cell} ipython3
print("Diamond distance (depolarized) =", rptbl.half_diamond_norm(Gx, Gx_depol, basis))
print("Diamond distance (over-rotated) =", rptbl.half_diamond_norm(Gx, Gx_overrot, basis))
```

## Frobenius and Jamiolkowski trace distance

Two cheaper distances between superoperators. The Frobenius difference is exactly `numpy.linalg.norm(a - b)`, with no normalization by dimension and no operational meaning; it is fast, which makes it useful as a convergence diagnostic and little else. The Jamiolkowski trace distance is the trace distance between the two Choi states, and it does carry an operational reading. A Choi state is what comes out when you send half of a maximally entangled pair through the process, so this number is the single-use distinguishability under that one particular input. It is therefore a lower bound on the diamond distance, which optimizes over all inputs, and not a bound in the other direction.

```{code-cell} ipython3
print("Frobenius difference =", rptbl.frobenius_diff(Gx, Gx_overrot, basis))
print("Jamiolkowski trace distance =", rptbl.jtrace_diff(Gx, Gx_overrot, basis))
```

For this over-rotation the Jamiolkowski distance matches the diamond distance printed above to seven significant figures, because the maximally entangled input is already an optimal distinguishing input here. Don't generalize from that: the gap can be large, and closing it is exactly what the semidefinite program is for.

## Unitarity

Unitarity measures how far a process is from being reversible, independent of what unitary it is trying to implement. It is a property of one process, not a comparison of two, and it separates coherent errors from stochastic ones: an over-rotation stays unitary, depolarization does not. Pass the basis explicitly, since `unitarity` defaults to `'gm'`.

```{code-cell} ipython3
print("Unitarity of the over-rotated gate =", tls.unitarity(Gx_overrot, basis))
print("Unitarity of the depolarized gate =", tls.unitarity(Gx_depol, basis))
```

## Gauge invariance and eigenvalue metrics

Everything above is *gauge variant*, and that is a real problem. Gauge invariance is the property that a metric doesn't change when either model is replaced by another model predicting exactly the same outcomes for every circuit, even though its operation matrices differ. Since experiments only ever constrain circuit outcomes, a gauge-variant metric is partly reporting a bookkeeping choice rather than a property of the device.

The usual workaround is gauge optimization: search the gauge orbit for the representative closest to your target, then compute the metric there. That works, but it is an extra step, and it is difficult or ill-defined often enough to be worth avoiding. See [gauge freedom](GaugeFreedom) for what pyGSTi does and how to control it.

The alternative is a metric that never needed the fix. Gauge transformations act by similarity, so the eigenvalues of a superoperator are gauge invariant, and any metric built only from spectra inherits that. The cost is that these metrics are blind to any error leaving the spectrum alone: a gate whose eigenvalues are right but whose eigenvectors are rotated scores perfectly.

```{code-cell} ipython3
print("Eigenvalue entanglement infidelity =", rptbl.eigenvalue_entanglement_infidelity(Gx, Gx_overrot, basis))
print("Eigenvalue average gate infidelity =", rptbl.eigenvalue_avg_gate_infidelity(Gx, Gx_overrot, basis))
print("Eigenvalue diamond distance =", rptbl.eigenvalue_diamondnorm(Gx, Gx_overrot, basis))
```

Compare those to the gauge-variant values above. The two infidelities agree to numerical precision here, but not because the models happen to be in the same gauge. The over-rotation is about the same axis as the gate itself, so the two superoperators share an eigenbasis and the entire error lands in the spectrum. Move the over-rotation to a different axis and the agreement falls apart, with no gauge transformation applied to anything:

```{code-cell} ipython3
Gx_overrot_y = ideal_model.rotate((0, 0.1, 0))[('Gxpi2', 0)].to_dense()
print("Entanglement infidelity            =", rptbl.entanglement_infidelity(Gx, Gx_overrot_y, basis))
print("Eigenvalue entanglement infidelity =", rptbl.eigenvalue_entanglement_infidelity(Gx, Gx_overrot_y, basis))
```

The ordinary infidelity is the same size as before, while the eigenvalue version reports a number roughly three orders of magnitude smaller. That is the blindness to eigenvector errors described above, and it is the general case, not the exception.

The eigenvalue diamond distance does not agree with the diamond distance computed from the full superoperators either: for the $X$ over-rotation it comes out larger. These quantities are computed from the spectra alone, not derived as bounds on their gauge-variant namesakes, so don't read them as bounds in either direction.

## State comparisons

State preparations and POVM effects get compared the same way. `rptbl.vec_fidelity` takes two superkets in a given basis:

```{code-cell} ipython3
rho_ideal = ideal_model['rho0'].to_dense(on_space='minimal')
rho_noisy = noisy_model['rho0'].to_dense(on_space='minimal')
print("State fidelity =", rptbl.vec_fidelity(rho_ideal, rho_noisy, basis))
```

If you have density matrices instead of superkets, use `pygsti.tools.fidelity` directly. `ppvec_to_stdmx` converts a Pauli-basis superket into a standard density matrix, which is the bridge between the two representations:

```{code-cell} ipython3
rhoA = tls.ppvec_to_stdmx(rho_ideal)
rhoB = np.array([[0.9, 0],
                 [0, 0.1]], complex)
print("Fidelity with a nearly-mixed state =", tls.fidelity(rhoA, rhoB))
```

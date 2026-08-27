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

# Leakage

Data from an experiment design built for a two-level system can be used to fit a *three*-level model, which is how you detect and quantify leakage out of the computational subspace. This page shows how, and how to generate a report whose gate error metrics respect the distinguished role of the first two levels.

## Why a third level

A qubit is usually the lowest two levels of something larger, such as a transmon's anharmonic ladder. When a control pulse has spectral weight on a neighboring transition, population leaves the computational subspace. That is leakage.

Leakage hides information rather than destroying it. Population that leaves can come back, carrying a phase that depends on how long it was gone. A two-level model has nowhere to put that population, hence nowhere to put the memory, and the data ends up matching no single Markovian two-level gate set. Fit three levels and the memory becomes explicit: the leaked amplitude has somewhere to live, and the leaky gate is once again a fixed CPTP map applied identically at every occurrence. What looks non-Markovian in two levels is ordinary Markovian error in three.

Note that this design's fiducials and germs were chosen to amplify two-level parameters, with no guarantee that they amplify every three-level one.

```{code-cell} ipython3
from pygsti.modelpacks import smq1Q_XYI as mp
from pygsti.leakage import leaky_qubit_model_from_pspec, construct_leakage_report
from pygsti.data import simulate_data
from pygsti.protocols import StandardGST, ProtocolData
import numpy as np
import scipy.linalg as la
```

## A coherent leakage error

`with_leaky_gate` draws a real unit vector $v$ whose first component is zero, forms the rank-one generator $H = s\,vv^{T}$, and composes $U = \exp(iH)$ after the ideal gate. The zero first component is what makes this a leakage error rather than a generic qutrit error: it confines $v$ to levels 1 and 2, so the ground state is untouched and amplitude moves only between the excited computational level and the leakage level. That is the picture for a weakly anharmonic transmon, whose 0-1 drive is detuned from 1-2 by only the anharmonicity. At `strength=0.125` about 0.4% of the population in level 1 leaks per application.

On three levels the error is a unitary: exactly CPTP, exactly Markovian, the same at every occurrence. Restricted to the computational subspace it is not even trace preserving.

```{code-cell} ipython3
def with_leaky_gate(m, gate_label, strength):
    rng = np.random.default_rng(0)
    v = np.concatenate([[0.0], rng.standard_normal(size=(2,))])
    v /= la.norm(v)
    H = v.reshape((-1, 1)) @ v.reshape((1, -1))
    H *= strength
    U = la.expm(1j*H)
    m_copy = m.copy()
    G_ideal = m_copy.operations[gate_label]
    from pygsti.modelmembers.operations import ComposedOp, StaticUnitaryOp
    m_copy.operations[gate_label] = ComposedOp([G_ideal, StaticUnitaryOp(U, basis=m.basis)])
    return m_copy, v
```

## The target model

`leaky_qubit_model_from_pspec` returns a qutrit *lift* of the qubit model, not a leaky one; the name is a misnomer. Each 2-by-2 gate unitary $u$ is promoted to the 3-by-3 unitary with $u$ in the upper-left block and a 1 in the remaining diagonal entry, so the ideal gates act trivially on the third level. The preparation is $|0\rangle\langle 0|$, and `Mdefault` keeps exactly two effects: $|0\rangle\langle 0|$ for outcome "0", and $|1\rangle\langle 1| + |2\rangle\langle 2|$ for outcome "1". The leakage level is not separately observable; it reads out as a 1. That is why the two-level design's data can be used unchanged.

The default `mx_basis='l2p1'` ("leakage, 2 plus 1") is a Hermitian qutrit basis sorted by that split: four elements span the computational subspace's operators (the qubit identity and Paulis, padded with a zero row and column), one is the projector onto the leakage level, and four span the coherences between them. Hermiticity is required because these models hold real parameter vectors; the `C[...]`/`L[...]` labels matter as much, since pyGSTi reads them to decide that a basis designates a *proper* computational subspace, which switches on the subspace-restricted gauge optimization and report metrics below.

```{code-cell} ipython3
ed = mp.create_gst_experiment_design(max_max_length=8)
tm3 = leaky_qubit_model_from_pspec(mp.processor_spec(), mx_basis='l2p1')
dgm3, leaking_state = with_leaky_gate(tm3, ('Gxpi2', 0), strength=0.125)
```

## Simulating and fitting

Short circuits carry less information, so `num_samples` is large: $10^5$ shots buy back some of the precision the missing depth would have supplied. That count also outruns pyGSTi's default likelihood regularization, which replaces the log-likelihood with a quadratic below `min_prob_clip` and softens the zero-frequency terms inside `radius`. Both default to $10^{-4}$, against documented advice to keep `radius` below the smallest expected frequency, which here is $10^{-5}$. Lowering both keeps the regularization out of the region where the fit is decided, at the cost of an objective that is sharper near zero probability and harder on the optimizer.

```{code-cell} ipython3
num_samples = 100_000
if num_samples > 10_000:
    from pygsti.objectivefns import objectivefns
    objectivefns.DEFAULT_MIN_PROB_CLIP = objectivefns.DEFAULT_RADIUS = 1e-12
ds = simulate_data(dgm3, ed.all_circuits_needing_data, num_samples=num_samples, seed=1997)
```

The bad-fit machinery acts only when the misfit exceeds `threshold` standard deviations, so setting it to 0.0 makes the `wildcard1d` analysis run for essentially any fit. A wildcard budget is the slack you would have to allow the model's predicted probabilities to make them consistent with the data. The one-dimensional version fits a single scale, spread across gates in proportion to each gate's diamond distance from its target, and the report prints it beside the other error metrics.

```{code-cell} ipython3
:tags: [output_scroll]

gst = StandardGST(
    modes=('CPTPLND',), target_model=tm3, verbosity=4,
    badfit_options={'actions': ['wildcard1d'], 'threshold': 0.0}
)
pd = ProtocolData(ed, ds)
res = gst.run(pd)
```

## Reporting

Ordinary gauge optimization minimizes the distance between estimate and target over the gauge group, comparing full 9-by-9 superoperators. That is the wrong objective here: the target's action on the leakage level is a convention we chose, not anything the device was built to do. `construct_leakage_report` compares only the computational block, then finishes with a step restricted to the block-diagonal subgroup that does not mix the computational subspace with the leakage level.

Each estimate in `updated_res` gains a model keyed `LAGO` beside the usual `stdgaugeopt` one, and the two disagree usefully. The error was injected into `Gxpi2` alone; the leakage-aware gauge attributes it there far more sharply than standard gauge optimization does, which pushes more of it onto `Gypi2`. How large the gap looks depends on which fidelity measure you compare them with, but its direction does not. The report's gate tables also gain leakage columns and a switch between subspace-restricted and full-space metrics.

```{code-cell} ipython3
report_dir = '../../../example_files/leakage-report-automagic'
report_object, updated_res = construct_leakage_report(res, title='easy leakage analysis!')
report_object.write_html(report_dir, connected=True)
```

Served with these docs: <a href="../../../reports/leakage-report-automagic.html">leakage-report-automagic</a>.

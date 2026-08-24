---
jupyter:
  jupytext:
    default_lexer: ipython3
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.19.5
  kernelspec:
    display_name: cgst
    language: python
    name: python3
---

# Character Gate Set Tomography (cGST)

Character gate set tomography combines the *amplifying germs* of gate set tomography (GST) with the *representation-theoretic filtering* of character randomized benchmarking. This tutorial builds a cGST experiment design for the one-qubit $\{S, \sqrt{Y}\}$ gate set, simulates data from a noise model whose error parameters are known exactly, runs the cGST analysis, and compares the fitted parameters to the truth.

## The idea

A cGST **germ** is a short circuit whose ideal implementation generates a small finite cyclic group: the phase gate $S$ generates $\mathbb{Z}_4$, $\sqrt{Y}$ generates $\mathbb{Z}_4$, and the compound "triangle" germ $\triangle = \sqrt{Y}\cdot S$ generates $\mathbb{Z}_3$ (it is a third root of the identity). Instead of only repeating a germ deterministically (as GST does), cGST inserts uniformly random powers of the germ and, in post-processing, weights each circuit's outcome by the conjugated character $\chi^*_j(n)$ of one irrep $j$ of that group, evaluated at the circuit's total germ power $n$. Averaging then cancels every other irrep's contribution, so each character-weighted signal

$$ z_j(k) \;=\; \big\langle\, \chi^*_j(n)\, \hat{p}(n) \,\big\rangle_{\text{circuits at depth } k} $$

decays as a single exponential: the multi-exponential fitting problem of ordinary GST decomposes into a collection of effective $T_1$ experiments (trivial irrep, $j=0$) and Ramsey experiments (complex irreps):

$$ z_0(k) = (B - C)\,\lambda_1^k + C, \qquad z_1(k) = A\,\lambda_2^k\, e^{i(\theta k + \varphi)} . $$

The decay magnitudes are the noisy germ's eigenvalue magnitudes (stochastic error rates) and the phase-winding rate $\theta$ is its coherent (over-rotation) angle error, read off directly from a linear fit to the unwrapped phase.

Three sampling modes are implemented by `CharacterGSTGermDesign`:

* **`'reduced'`** *(used below)*: a fixed number $k_0$ of random germ powers act as "synthetic SPAM" (an approximate projector onto the target irrep), followed by $k$ deterministic germ repetitions. The fitted per-$k$ decay is the noisy germ's own eigenvalue.
* **`'full'`**: depth $k$ means $k$ i.i.d. random powers. The fitted decay is then an eigenvalue of the group *Fourier operator* $\Pi_j = \frac{1}{N}\sum_m \chi_j^*(m)\Lambda^m$, which relates to the germ eigenvalue deviation $y$ through $f(y) = \frac{1-y^N}{N(1-y)}$. Note $\arg f(e^{i\theta}) \approx \frac{N-1}{2}\theta$: full mode *amplifies* phase errors by $(N{-}1)/2$, and `pygsti.algorithms.cgstfit.invert_projector_eigenvalue` numerically inverts $f$ to recover the bare eigenvalue.
* **`'exact'`**: for a cyclic germ the circuit depends only on the total power, so the $k_0$ Monte-Carlo rounds can be replaced by deterministic quadrature over all possible totals, weighted by their exact probabilities: the synthetic projector is evaluated with zero sampling error.

The experiments built by `create_1q_szy_cgst_design` (following the cGST manuscript's Experiments 1–5 and 9–11):

| name | germ | group | irrep | estimates |
|---|---|---|---|---|
| `s_t1` | $S$ | $\mathbb{Z}_4$ | trivial | $\lambda_1$, active error $a$ |
| `s_ramsey` | $S$ | $\mathbb{Z}_4$ | complex | $\lambda_2$, over-rotation $\theta$ |
| `y_t1` | $\sqrt{Y}$ | $\mathbb{Z}_4$ | trivial | $r_1$ (axis-parallel decay) |
| `y_ramsey` | $\sqrt{Y}$ | $\mathbb{Z}_4$ | complex | $r_2$, over-rotation $\alpha$ |
| `tri_t1` | $\sqrt{Y} S$ | $\mathbb{Z}_3$ | trivial | $\lambda_1^\triangle$, active combination |
| `tri_ramsey` | $\sqrt{Y} S$ | $\mathbb{Z}_3$ | complex | $\lambda_2^\triangle$, $\omega$ (yields angle-between-axes $\beta$) |
| `idle_s/y/tri` | germ · $G_i$ | n/a | complex | idle coherent components $\theta_{I,z}$, $\theta_{I,y}$, $\theta_{I,x}$ |

```python
import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt

import pygsti
from pygsti.processors import QubitProcessorSpec
from pygsti.models.modelconstruction import create_explicit_model
from pygsti.protocols import ProtocolData
from pygsti.protocols.cgst import (CharacterGST, CharacterDecay, create_1q_szy_cgst_design,
                                   extract_szy_error_parameters, true_germ_eigenvalues)
from pygsti.tools.optools import unitary_to_pauligate
```

## A noise model with exactly known error parameters

We construct the noisy $S$ and $\sqrt{Y}$ gates *directly in the manuscript's standard-gauge channel forms*, so that every parameter cGST is supposed to estimate has a known injected value: over-rotations $\theta, \alpha$, the relational angle-between-axes $\beta$, stochastic decays $\lambda_1, \lambda_2$ (for $S$) and the $E_{\rm sto}$ block (for $\sqrt{Y}$) with correlated rates $c_{xy}, c_{xz}, c_{yz}$, and active (amplitude-damping-type) errors $a, a_y, a_{\rm rel}$. The idle gets known coherent rotation errors about all three axes.

```python
sx = np.array([[0, 1], [1, 0]], complex)
sy = np.array([[0, -1j], [1j, 0]], complex)
sz = np.array([[1, 0], [0, -1]], complex)

def rot(axis, angle):
    """PTM of exp(-i*angle/2 * axis)."""
    return unitary_to_pauligate(expm(-0.5j * angle * axis))

def standard_gauge_model(theta=0., alpha=0., beta=0., lam1=1., lam2=1., a=0.,
                         r1=0., r2=0., cxy=0., cxz=0., cyz=0., ay=0., arel=0.,
                         idle_angles=(0., 0., 0.)):
    E_S = np.array([[1, 0, 0, 0],
                    [0, lam2 * np.cos(theta), -lam2 * np.sin(theta), 0],
                    [0, lam2 * np.sin(theta),  lam2 * np.cos(theta), 0],
                    [-a, 0, 0, lam1]])
    E_sto = np.array([[1, 0, 0, 0],
                      [arel, 1 - r2, cxy, cxz],
                      [arel, cxy, 1 - r2, cyz],
                      [ay, cxz, cyz, 1 - r1]])
    Lam_S = E_S @ rot(sz, np.pi / 2)
    Lam_Y = rot(sx, beta) @ E_sto @ rot(sy, np.pi / 2 + alpha) @ rot(sx, -beta)
    thx, thy, thz = idle_angles
    Lam_I = rot(sx, thx) @ rot(sy, thy) @ rot(sz, thz)

    pspec = QubitProcessorSpec(1, ['Gzpi2', 'Gypi2', 'Gi'], qubit_labels=['Q0'])
    mdl = create_explicit_model(pspec, ideal_gate_type='full')
    mdl.operations[('Gzpi2', 'Q0')] = Lam_S
    mdl.operations[('Gypi2', 'Q0')] = Lam_Y
    mdl.operations[('Gi', 'Q0')] = Lam_I
    return mdl

# error magnitudes at the scale used in the manuscript's visual appendix
injected = dict(theta=0.010, alpha=0.008, beta=0.006,
                lam1=1 - 0.020, lam2=1 - 0.015, a=-0.004,
                r1=0.018, r2=0.012, cxy=0.004, cxz=0.003, cyz=0.002,
                ay=-0.003, arel=-0.002, idle_angles=(0.002, 0.003, 0.004))
true_model = standard_gauge_model(**injected)
```

(Any pyGSTi model works here, e.g. one built with `create_explicit_model(pspec, lindblad_error_coeffs={'Gzpi2:Q0': {('H','Z'): 0.005, ('S','X'): 0.004, ...}})`. We use the standard-gauge construction because it makes the truth table below exact; for a generic model the gauge-invariant germ eigenvalues computed next play the role of the truth.)

## Numeric ground truth

cGST's decay fits estimate the noisy germs' eigenvalue deviations, which are gauge-invariant quantities we can extract from the true model by eigendecomposition, with no first-order approximations:

```python
depths = [0, 1, 2, 4, 8, 16, 24, 32, 48, 64, 96]
edesign = create_1q_szy_cgst_design(depths, circuits_per_depth=24, mode='reduced',
                                    num_projection_rounds=4, include_idle=True, seed=2026)

truth = {}
for name in edesign.keys():
    d = edesign[name]
    devs = true_germ_eigenvalues(true_model, d.germ, d.group_order)
    truth[name] = devs[d.irrep_index]
    print(f"{name:10s}  |y| = {abs(truth[name]):.5f}   arg y = {np.angle(truth[name]):+.5f}")
```

## The experiment design

Each sub-experiment holds, per depth, `circuits_per_depth` circuits of the form *prep fiducial · germ$^n$ · measurement fiducial*, where $n$ contains the random contribution. The realized random exponents are stored in the design (serialized alongside the circuits), and the character weights are recomputed from them:

```python
sub = edesign['s_ramsey']
print("germ:", sub.germ.str, "  group order:", sub.group_order, "  irrep:", sub.irrep_index)
print("prep fiducial:", sub.prep_fiducial.str, "  meas fiducial:", sub.meas_fiducial.str)
print("\nexample circuit (depth 4):")
print(sub.circuit_lists[3][0])
print("\ncharacter weights at depth 4:", np.round(sub.character_weights()[3][:6], 3), "...")
```

Two variance-reduction refinements are built into the sampling (both leave expectations unchanged): the random powers are *stratified* over the residue classes of the total power, so character weights of constant backgrounds cancel exactly at each depth, and in `'reduced'` mode the projection-round draws are *reused across depths* (common random numbers), so their contribution is a depth-independent constant absorbed by the fit amplitude rather than per-depth noise.

## Simulate and run

```python
ds = pygsti.data.simulate_data(true_model, edesign.all_circuits_needing_data,
                               num_samples=1000, sample_error='multinomial', seed=2026)
data = ProtocolData(edesign, ds)

protocol = CharacterGST(bootstrap_samples=200, gateset_inversion='szy', seed=7)
results = protocol.run(data)
top = results.for_protocol['CharacterGST']
```

Each sub-experiment's character-weighted signal and fit:

```python
names = list(edesign.keys())
fig, axes = plt.subplots(3, 3, figsize=(13, 9))
for name, ax in zip(names, axes.ravel()):
    decay = results[name].for_protocol['CharacterDecay']
    decay.plot(ax=ax, title=name)
fig.tight_layout()
```

## Fitted vs. true error parameters

The germ-level decays should match the numeric eigenvalue truth tightly; the derived standard-gauge parameters are first-order formulas, so they carry $O(\epsilon^2)$ truncation on top of the statistical error.

```python
import pandas as pd

params = top.error_parameters
true_params = {
    'theta': injected['theta'], 'alpha': injected['alpha'], 'beta': injected['beta'],
    'lambda1': injected['lam1'], 'lambda2': injected['lam2'], 'a': injected['a'],
    'r1': 1 - abs(true_germ_eigenvalues(true_model, edesign['y_t1'].germ, 4)[0]),
    'r2': 1 - abs(true_germ_eigenvalues(true_model, edesign['y_ramsey'].germ, 4)[1]),
    'lambda1_triangle': abs(truth['tri_t1']), 'lambda2_triangle': abs(truth['tri_ramsey']),
    'omega_deviation': np.angle(truth['tri_ramsey']),
    'c_sum': injected['cxy'] + injected['cxz'] + injected['cyz'],
    'active_combo': -injected['a'] + injected['ay'] + 2 * injected['arel'],
    'theta_idle_x': injected['idle_angles'][0],
    'theta_idle_y': injected['idle_angles'][1],
    'theta_idle_z': injected['idle_angles'][2],
}
rows = [{'parameter': key, 'fitted': params[key], 'true': true_params[key],
         'error': params[key] - true_params[key], 'rel error': (params[key] - true_params[key])/true_params[key]}
        for key in true_params if key in params]
pd.DataFrame(rows).style.format({'fitted': '{:+.5f}', 'true': '{:+.5f}', 'error': '{:+.2e}', 'rel error': '{:+.2e}'})
```

The germ-level summary (decay magnitude/phase with bootstrap uncertainties) is also available as a dataframe via `top.to_dataframe()`.

```python
summary = top.to_dataframe()
summary[summary['type'] == 'decay']
```

## The triangle equations, adjudicated numerically

The manuscript's first-order "triangle equations" relate the triangle germ's decays to the relational errors; its own draft notes flag a suspicious sign and coefficient. The implementation in `extract_szy_error_parameters` uses relations adjudicated numerically against the standard-gauge channel forms (locked in by `test/unit/protocols/test_cgst.py`):

* $\omega - \frac{2\pi}{3} = \frac{\theta + \alpha + 2\beta}{\sqrt{3}}$  *(as in the manuscript)*;
* $\lambda_1^\triangle - \lambda_2^\triangle = c_{xy} + c_{xz} + c_{yz}$: coefficient $1$, not the manuscript's $2/3$ (the splitting is $+\tfrac{2}{3}c_{\rm sum}$ on the trivial branch and $-\tfrac{1}{3}c_{\rm sum}$ on the complex branch);
* $3\,(1-\lambda_1^\triangle)(2C^\triangle - 1) = -a + a_y + 2a_{\rm rel}$: a different weighting than the manuscript's $(a + a_y + a_{\rm rel})$; the sign flip on $a$ comes from $\sqrt{Y}$ rotating $S$'s active shift ($\hat z \to \hat x$) before it projects onto the triangle axis $(\hat x + \hat y + \hat z)/\sqrt{3}$;
* for the $S$ germ itself, $a = (1-\lambda_1)(1 - 2\tilde C)$, with the opposite sign to the manuscript's formula, given its own channel convention $E_S[3,0] = -a$.

A quick demonstration of the splitting coefficient (sweeping only the correlated errors, with a little depolarization to keep the model completely positive):

```python
for c in (0.002, 0.004, 0.008):
    mdl_c = standard_gauge_model(cxy=c, cxz=c, cyz=c, lam1=0.99, lam2=0.99, r1=0.01, r2=0.01)
    tri = edesign['tri_t1']
    devs = true_germ_eigenvalues(mdl_c, tri.germ, tri.group_order)
    split = abs(devs[0]) - abs(devs[1])
    print(f"c_sum = {3*c:.4f}   lambda1_tri - lambda2_tri = {split:+.6f}   ratio = {split/(3*c):.4f}")
```

## Idle characterization

The idle-interleaved germs $(S\, G_i)$, $(\sqrt{Y}\, G_i)$, $(\sqrt{Y} S\, G_i)$ pick up the idle's coherent error along each germ's rotation axis. Differencing their fitted phases against the bare germs' phases isolates the idle contributions ($z$, $y$, and, through the triangle axis, $x$), exactly as interleaved RB isolates a single gate's error:

```python
for comp in 'xyz':
    fitted = params['theta_idle_' + comp]
    true_val = true_params['theta_idle_' + comp]
    print(f"theta_idle_{comp}: fitted {fitted:+.5f}   true {true_val:+.5f}")
```

## Full-random-sampling mode and the $(N-1)/2$ phase gain

Rebuilding the $S$ Ramsey experiment in `'full'` mode: the raw fitted phase is $\approx \tfrac{3}{2}\theta$ for the $\mathbb{Z}_4$ group, and inverting the projector eigenvalue map recovers the same bare eigenvalue as reduced mode.

```python
edesign_full = create_1q_szy_cgst_design([1, 2, 4, 8, 12, 16], circuits_per_depth=64,
                                         mode='full', include_idle=False, seed=99)
ds_full = pygsti.data.simulate_data(true_model, edesign_full.all_circuits_needing_data,
                                    num_samples=1000, sample_error='multinomial', seed=42)
data_full = ProtocolData(edesign_full, ds_full)

raw = CharacterDecay(invert_full_mode=False).run(data_full['s_ramsey'])
inverted = CharacterDecay(invert_full_mode=True).run(data_full['s_ramsey'])
print(f"raw fitted phase      = {raw.germ_eigenvalue_phase:+.5f}  (~1.5 x theta = {1.5*injected['theta']:+.5f})")
print(f"inverted eigenvalue   = {inverted.germ_eigenvalue_magnitude:.5f} @ {inverted.germ_eigenvalue_phase:+.5f}")
print(f"reduced-mode estimate = {params['lambda2']:.5f} @ {params['theta']:+.5f}")
print(f"truth                 = {abs(truth['s_ramsey']):.5f} @ {np.angle(truth['s_ramsey']):+.5f}")
```

## Notes and caveats

* **First-order validity.** The extraction formulas for $\beta$, `c_sum` and `active_combo` are first order in the error rates; with errors at the $10^{-2}$ scale, expect $O(10^{-4})$ truncation on top of statistical error. The germ eigenvalues themselves ($\lambda$'s, $\theta$, $\alpha$, $\omega$) are not truncated: they are exact spectral properties, fit directly.
* **Sampling variance.** Character weighting is a signed/complex average, so its statistical error at fixed shots exceeds a plain probability estimate's. The `'exact'` quadrature mode removes the character-sampling component entirely for cyclic germs, at the cost of a fixed circuits-per-depth; the stratification and common-random-number refinements do most of that work for the Monte-Carlo modes.
* **Reduced-mode bias.** The $k_0$-round synthetic projector leaks into unwanted irreps at $O(r^{k_0})$ for per-germ infidelity $r$; with $k_0 = 4$ and $r \sim 10^{-2}$ this is negligible against shot noise.
* **Scaling.** Everything here generalizes to any germ whose ideal implementation generates a *cyclic* group, and the character utilities (`pygsti.tools.chartools`) support arbitrary finite abelian groups (products of cyclics) toward the multi-qubit constructions of the manuscript; non-abelian groups and multiplicity handling are future work.

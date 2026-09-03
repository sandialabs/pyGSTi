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

# Character Phase Estimation (CPE)

Robust phase estimation (RPE) estimates one eigenphase difference of a repeated gate with Heisenberg-like scaling, tolerating any additive error in the measured probabilities below a fixed threshold (each generation's estimate of $2^k\phi$ must err by less than $\pi/3$). It requires, however, a state preparation and measurement that confine the signal to a single coherence $|E_a\rangle\langle E_b|$ of the gate. Any support on other eigenstate pairs injects additional frequencies into the signal, and if that support is large the per-generation window selection locks onto the wrong frequency, so the experimenter no longer knows which eigenvalue difference they estimated.

Character phase estimation replaces that requirement with the character-projector machinery of cGST (see the `CharacterGST` tutorial). A fixed number $k_0$ of uniformly random germ powers, weighted in post-processing by the conjugated character $\chi_j^*(n)$ of the total power $n$, act as a synthetic state preparation supported on a single irrep: the filtered complex signal

$$ z_j(k) = \big\langle \chi_j^*(n)\,\hat P_{\rm success} \big\rangle $$

contains only irrep $j$'s coherence, up to a residual bias $O(r^{k_0/2})$ in the gate's infidelity $r$. The per-depth raw angles $\arg\!\big(z_j(k)\, z_j(0)^*\big) = k\,\delta_j \pmod{2\pi}$ (the depth-0 reference cancels the depth-independent SPAM and projector-tilt phase) then feed the standard RPE unwinding. Because the character weight is evaluated at the total germ power, the ideal phase cancels and $\delta_j$ is directly the deviation of irrep $j$'s eigenphase from ideal. One generic preparation serves every irrep at once: the same circuits, reweighted with a different character, estimate a different eigenvalue difference.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import pygsti
from pygsti.protocols import ProtocolData
from pygsti.protocols.cgst import true_germ_eigenvalues
from pygsti.protocols.cpe import (CharacterPhaseEstimation, create_2q_diagonal_cpe_design,
                                  extract_izz_phase_deviations, izz_coherence_irreps,
                                  make_2q_diagonal_model)
```

## A non-degenerate diagonal two-qubit gate

The demonstration gate is $G_d = \exp\!\big(-\tfrac{i\pi}{13}(IZ + 3\,ZI + 4\,ZZ)\big)$: generator angles $\theta_P = 2\pi k_P/13$ with $(k_{IZ}, k_{ZI}, k_{ZZ}) = (1,3,4)$. Its four computational-basis eigenphases, in units of $\pi/13$, are $(-8, 2, 6, 0)$, which form a perfect difference set mod 13: all twelve pairwise (signed) differences are distinct, so every one of the twelve coherences of the superoperator occupies its own $\mathbb{Z}_{13}$ irrep. (Order 13 is the smallest with this property, the Sidon-set bound $4\cdot 3+1$, which is why the demonstration gate is a thirteenth root of the identity rather than a lower-order rotation.)

Three coherences are enough to determine the three generator angles, and all three are accessible from product states:

| irrep $j$ | coherence | eigenphase deviation |
|---|---|---|
| 5  | $\vert 01\rangle\langle 00\vert$ | $\delta_5 = d_{IZ} + d_{ZZ}$ |
| 7  | $\vert 10\rangle\langle 00\vert$ | $\delta_7 = d_{ZI} + d_{ZZ}$ |
| 10 | $\vert 11\rangle\langle 10\vert$ | $\delta_{10} = d_{IZ} - d_{ZZ}$ |

We inject known generator-angle deviations plus uniform depolarization, so the ground truth is exact (everything commutes with the ideal gate):

```python
ks, order = (1, 3, 4), 13
deviations = (0.012, -0.007, 0.004)   # (d_IZ, d_ZI, d_ZZ), radians
depol = 0.01

noisy_model = make_2q_diagonal_model(ks=ks, order=order, deviations=deviations, depol=depol)

irreps = izz_coherence_irreps(ks, order)
germ_evals = true_germ_eigenvalues(noisy_model, 
                                   pygsti.circuits.Circuit([('Gd','Q0','Q1')], line_labels=('Q0','Q1')),
                                   order)
print("all twelve coherence irreps:", sorted(j for j in germ_evals if j != 0))
for j in irreps:
    print(f"targeted irrep {j:2d}:  |y| = {abs(germ_evals[j]):.5f}   "
          f"delta = {np.angle(germ_evals[j]):+.6f} rad")
```

## The experiment design

The design uses cGST's `'exact'` sampling mode: for a cyclic germ the $k_0$ Monte-Carlo projection rounds are replaced by deterministic quadrature over the total random power, so the synthetic projector carries no character-sampling noise. With $k_0 = 2$ that is $k_0 \cdot 12 + 1 = 25$ circuits per depth, on the RPE schedule $k = 0, 1, 2, 4, \dots, 64$ (depth 0 is the phase reference).

The preparation is deliberately generic: $|{+}{+}\rangle$ overlaps all twelve coherences with equal amplitude $1/16$. This is the multi-frequency worst case for naive RPE, and the situation the character filter resolves in post-processing.

```python
edesign = create_2q_diagonal_cpe_design(depths=(0, 1, 2, 4, 8, 16, 32, 64),
                                        num_projection_rounds=2, ks=ks, order=order)
print(f"{edesign.circuits_per_depth} circuits/depth, "
      f"{len(edesign.all_circuits_needing_data)} circuits total")
```

## Simulate

```python
ds = pygsti.data.simulate_data(noisy_model, edesign.all_circuits_needing_data,
                               num_samples=3000, sample_error='multinomial', seed=2026)
data = ProtocolData(edesign, ds)
```

## Naive RPE breaks on this data

The unfiltered survival probability $p(n)$ over one group period carries all twelve ideal frequencies at once. Its 13-point DFT shows twelve equal-magnitude peaks: there is no dominant coherence for RPE's window selection to lock onto, and interpreting $2p(k)-1$ as a single coherence's $\cos(k\,\delta_5)$ is wrong by $O(1)$:

```python
# p(n) for n = 0..12 comes free from the depth-0 circuits (total power = quadrature index j)
depth0 = edesign.depths.index(0)
period = np.array([ds[c].counts.get((out,), 0.0) / ds[c].total
                   for c, out in zip(edesign.circuit_lists[depth0],
                                     edesign.idealout_lists[depth0])][:13])
dft = np.abs(np.fft.fft(period - period.mean()))
print("DFT magnitudes at frequencies 1..12:", np.round(dft[1:], 3))

true_d5 = deviations[0] + deviations[2]
for k in (4, 8):
    idx = edesign.depths.index(k)
    p_k = next(ds[c].counts.get((out,), 0.0) / ds[c].total
               for c, draws, out in zip(edesign.circuit_lists[idx], edesign.exponent_lists[idx],
                                        edesign.idealout_lists[idx]) if draws == [0])
    naive_angle = np.arccos(np.clip(2 * p_k - 1, -1, 1))
    print(f"depth {k:2d}: naive angle {naive_angle:.3f} rad vs true k*delta_5 = {k*true_d5:.3f} rad")
```

## CPE on the same data

Character filtering pulls three clean single-frequency signals out of the identical dataset, and the RPE window selection converges on each.

One feature of the exact-quadrature design is worth knowing before looking at the plots. Depth $k$ uses the circuits with total germ powers $k, \dots, k + k_0(N-1)$, so adjacent depths reuse most of the same physical circuits (depths 0 and 1 share 24 of 25), and duplicated circuits merge in the dataset, accumulating shots. The shot noise in $z_j(k)$ and the reference $z_j(0)$ is therefore mostly *common* at low depth and cancels in the raw angle $\arg[z_j(k)z_j(0)^*]$. As a result the generation estimates start out already accurate (about 1 mrad here) and improve only modestly with depth, rather than showing the wide-to-narrow funnel of textbook RPE. The per-generation bootstrap error bars below make the actual contraction visible.

```python
proto = CharacterPhaseEstimation(bootstrap_samples=200, seed=7)
results = proto.run(data)

truth = {j: np.angle(germ_evals[j]) for j in irreps}
rows = [{'irrep': j, 'estimate (rad)': est, 'bootstrap sigma': err, 'truth (rad)': truth[j],
         'pull (sigma)': (est - truth[j]) / err}
        for j, (est, err) in results.phases_by_irrep().items()]
pd.DataFrame(rows).round(6)
```

```python
results.plot(true_phases=truth);
```

## Recovering the generator angles

```python
extracted = extract_izz_phase_deviations(results.phases_by_irrep(), ks=ks, order=order)
for name, true_val in zip(('d_iz', 'd_zi', 'd_zz'), deviations):
    print(f"{name}: {extracted[name]:+.5f} +- {extracted[name + '_stderr']:.5f}"
          f"   (injected {true_val:+.5f})")
```

## The manuscript figure

Panel (a): the twelve-peak DFT of the unfiltered signal (why naive RPE is ambiguous). Panel (b): per-irrep RPE convergence of the character-filtered estimates toward the injected deviations, inside the $\pm\pi/2k$ RPE envelope.

```python
import os

colors = {irreps[0]: '#0072B2', irreps[1]: '#D55E00', irreps[2]: '#009E73'}
fig, (ax_a, ax_b) = plt.subplots(2, 1, figsize=(3.4, 4.4))

ax_a.bar(np.arange(1, 13), dft[1:], width=0.55, color='#6b7280')
ax_a.set_xlabel('frequency index $j$ (units of $2\\pi/13$)', fontsize=8)
ax_a.set_ylabel('$|{\\rm DFT}[p(n)]|$', fontsize=8)
ax_a.set_xticks(range(1, 13))
ax_a.text(0.02, 0.9, '(a)', transform=ax_a.transAxes, fontsize=9, fontweight='bold')

gen_depths = [k for k in results.depths if k > 0]
for j in irreps:
    ests = np.array(results.generation_estimates[str(j)])
    ests = (ests + np.pi) % (2 * np.pi) - np.pi
    yerr = 1e3 * np.array(results.generation_stderrs[str(j)])
    ax_b.errorbar(gen_depths, 1e3 * ests, yerr=yerr, fmt='o-', ms=3.5, lw=1.2,
                  capsize=2, color=colors[j], label=f'irrep {j}')
    ax_b.axhline(1e3 * truth[j], color=colors[j], ls=':', lw=1, alpha=0.7)
ax_b.set_xscale('log', base=2)
ax_b.set_ylim(-12, 22)
ax_b.set_xlabel('germ depth $k$', fontsize=8)
ax_b.set_ylabel('estimated deviation (mrad)', fontsize=8)
ax_b.legend(fontsize=7, loc='lower right', frameon=False)
ax_b.text(0.02, 0.9, '(b)', transform=ax_b.transAxes, fontsize=9, fontweight='bold')

for ax in (ax_a, ax_b):
    ax.tick_params(labelsize=7)
    for spine in ('top', 'right'):
        ax.spines[spine].set_visible(False)
fig.tight_layout()

figdir = '../../../../new/figs'
if os.path.isdir(figdir):
    fig.savefig(os.path.join(figdir, 'CPE_Demo.pdf'), bbox_inches='tight')
    print('saved', os.path.join(figdir, 'CPE_Demo.pdf'))
```

## Notes and caveats

* **Which coherences can CPE target?** Only irreps carrying a *single* coherence yield single-frequency filtered signals. For the demonstration gate every nontrivial irrep qualifies (the difference-set property); for a degenerate germ (e.g. a bare CZ, whose $-1$ irrep carries many coherences) the filtered signal inside a multiplicity block is still multi-frequency, the same limitation as in cGST, addressed there by commuting-gate-set constructions.
* **Shot cost.** The generic $|{+}{+}\rangle$ prep spreads amplitude over sixteen directions, so each coherence's signal is $|z(0)| \approx 1/16$: character filtering trades preparation accuracy for shots. A physically motivated prep with larger overlap on the targeted coherences reduces the cost; the filter guarantee is unchanged.
* **Depth schedule.** The raw angles come from $\arg z_j(k)$ directly; do not fit them with `pygsti.algorithms.cgstfit.fit_complex_decay`, whose `numpy.unwrap` phase seeding assumes closely spaced depths and aliases on geometric schedules.

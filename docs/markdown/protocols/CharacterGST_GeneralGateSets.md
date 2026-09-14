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

# Character GST for arbitrary finite-order gate sets

The `CharacterGST` tutorial builds a hand-designed one-qubit $\{S, \sqrt{Y}\}$ experiment and inverts it with closed-form formulas. This tutorial does the same job for *any* gate set, automatically.

## The idea

Every cGST observable is an eigenvalue of a *noisy germ* on one irrep of the finite cyclic group its *ideal* germ generates: a decay rate $|\lambda|$ (stochastic error per germ repetition), a phase $\arg\lambda$ (coherent error per repetition), and, on the trivial irrep, an asymptote (active, amplitude-damping-type error). Those eigenvalues are gauge invariant, and to first order in the error rates each is a linear function of the gates' elementary error generator coefficients. A germ set is *good enough* exactly when GST's twirled-Jacobian criterion says so — i.e. when it amplifies every non-gauge direction — which is a question pyGSTi's germ selection already answers. So the whole pipeline can be automated:

1. **germ selection** restricted to germs whose ideal superoperator has finite order (`cgstdesign.finite_order_candidate_germs`, `find_cgst_germs`) — only those generate a cyclic group to filter characters on;
2. **irrep enumeration** from the ideal germ's eigenvalues (`chartools.germ_irrep_multiplicities`, `cgstdesign.germ_irreps`);
3. **fiducial choice** maximizing the ideal decaying signal on each (germ, irrep) block (`cgstdesign.select_cgst_fiducials`);
4. **first-order inversion** of the fitted decays into error generator coefficients (`cgstinversion`), followed by
5. **gauge fixing** into the cGST standard gauge (`cgstgauge`).

Steps 1–3 are `cgstdesign.create_cgst_design`; steps 4–5 are `CharacterGST(gateset_inversion='linear')`.

```python
import time
import warnings

import numpy as np
import pandas as pd

import pygsti
from pygsti.algorithms import cgstdesign, cgstgauge
from pygsti.models.modelconstruction import create_explicit_model
from pygsti.processors import QubitProcessorSpec
from pygsti.protocols import ProtocolData
from pygsti.protocols.cgst import CharacterGST, true_germ_eigenvalues
from pygsti.tools import chartools

notebook_start = time.time()
pd.set_option('display.width', 200)


def short(circuit_str, qubit='Q0'):
    """Drop qubit labels from a circuit string, for compact tables."""
    return circuit_str.replace(':%s' % qubit, '').replace('@(%s)' % qubit, '') or '{}'
```

## Germs, groups and irreps of the XY gate set

The gate set here is the standard $\{X_{\pi/2}, Y_{\pi/2}\}$ one, rather than the $\{S, \sqrt{Y}\}$ set of the `CharacterGST` tutorial. `find_cgst_germs` runs pyGSTi's ordinary greedy germ search over the finite-order candidates only, so the result is amplificationally complete *and* usable by cGST:

```python
pspec = QubitProcessorSpec(1, ['Gxpi2', 'Gypi2'], qubit_labels=['Q0'])
target = create_explicit_model(pspec, ideal_gate_type='full TP', ideal_spam_type='full TP',
                               simulator='matrix')

with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    germs = cgstdesign.find_cgst_germs(target, max_length=7, seed=0)

rows = []
for germ in germs:
    order, multiplicities = cgstdesign.germ_irreps(target, germ)
    rows.append({'germ': short(germ.str), 'length': len(germ), 'group order': order,
                 'irrep multiplicities': multiplicities,
                 'designed irreps': cgstdesign.nonconjugate_irreps(order, multiplicities)})
pd.DataFrame(rows)
```

Five germs, of orders $(4, 4, 3, 3, 3)$: the two generators and three order-3 compound germs (rotations by $2\pi/3$ about body diagonals of the Bloch cube). The multiplicity dictionary is the ideal germ superoperator's eigenvalue spectrum sorted into characters $\chi_j(1) = e^{2\pi i j / N}$: a $\pi/2$ rotation has trivial multiplicity 2 (the identity direction plus its rotation axis) and multiplicity 1 on each of the conjugate irreps $j = 1, N-1$. Only one member of each conjugate pair is worth measuring — their decays are complex conjugates — hence the "designed irreps" column, which is `nonconjugate_irreps`.

Every block of this germ set is therefore a *scalar* decay: a two-dimensional trivial block (identity plus one decaying direction) or a multiplicity-one nontrivial irrep. That is what the analysis can fit; the *degenerate* case — a $\pi$ rotation's two-dimensional nontrivial block, say — shows up below, once an idle enters the gate set.

Candidates are deduplicated by ideal superoperator *and* gate content before the search. pyGSTi's own `find_germs` deduplicates by superoperator alone, which is harmless here but fatal for a gate set with an idle: `Gzpi2 Gi` is then discarded as a duplicate of `Gzpi2` although only the former amplifies the idle's errors, and no amplificationally complete candidate set survives (see the idle section below).

## The experiment design

`create_cgst_design` emits one `CharacterGSTGermDesign` child per (germ, irrep, fiducial pair) and records the layout in `germ_table`:

```python
depths = [0, 1, 2, 4, 8, 16, 32, 64, 128]

with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    design = cgstdesign.create_cgst_design(target, depths, circuits_per_depth=12, germs=germs,
                                           num_projection_rounds=3, seed=0)

table = pd.DataFrame(design.germ_table)
for col in ('germ', 'prep_fiducial', 'meas_fiducial'):
    table[col] = [short(s) for s in table[col]]
print(len(design.keys()), 'children,', len(design.all_circuits_needing_data), 'circuits')
table
```

Child names are `<germ layers>_irrep<j>_pair<p>` — filesystem- and JSON-safe on every platform (no `:`), because children are serialized as directories; for multi-qubit germs the gate names carry their qubit labels. Fiducials are words in the model's own gates (chosen up to length 3) that maximize the ideal decaying amplitude of that block, so `{}` means "the native prep/measurement already probes this block".

The sampling mode defaults to `'exact'`, which evaluates the synthetic irrep projector by deterministic quadrature over the germ powers with zero sampling error; that is not just a variance matter — the linear inversion below differentiates the fitted decays by finite differences and needs a deterministic, ripple-free response (`'reduced'` mode's Monte-Carlo projector leaves an $O(\text{error})$ ripple that degrades it, and the protocol warns; `'full'` mode is rejected outright). Blocks the analysis cannot use — nontrivial irreps of multiplicity $> 1$, trivial blocks of multiplicity $\ne 2$ — are still *designed* by default (the $d \times d$ fiducial grid needed to resolve a $d$-dimensional block is emitted) and then skipped, with a warning, by the inversion; pass `include_degenerate_blocks=False` to emit only children the analysis supports.

## A second gate set: X and T

Nothing above is specific to $\pi/2$ rotations. The $\{X_{\pi/2}, T\}$ gate set works the same way, with $T$ generating $\mathbb{Z}_8$:

```python
t_pspec = QubitProcessorSpec(1, ['Gxpi2', 'Gt'], qubit_labels=['Q0'])
t_target = create_explicit_model(t_pspec, ideal_gate_type='full TP', ideal_spam_type='full TP',
                                 simulator='matrix')

with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    t_design = cgstdesign.create_cgst_design(t_target, [0, 1, 2, 4, 8, 16, 32, 64],
                                             circuits_per_depth=12, mode='exact',
                                             num_projection_rounds=3, seed=0)

t_table = pd.DataFrame(t_design.germ_table)
for col in ('germ', 'prep_fiducial', 'meas_fiducial'):
    t_table[col] = [short(s) for s in t_table[col]]
print(len(t_design.keys()), 'children,', len(t_design.all_circuits_needing_data), 'circuits')
t_table
```

The germ search again returns five germs — now of orders $(4, 8, 3, 3, 3)$ — and this time no germ has a degenerate block, so all ten children are analyzable. `chartools.germ_group_order` is what decides the order, and it is also what rejects candidates: an infinite-order germ (a rotation by an irrational fraction of $\pi$) never enters the candidate list.

```python
for gate in ('Gxpi2', 'Gt'):
    superop = t_target.operations[(gate, 'Q0')].to_dense()
    print(gate, ' order', chartools.germ_group_order(superop),
          ' multiplicities', chartools.germ_irrep_multiplicities(
              superop, chartools.germ_group_order(superop)))
```

## A gate set with an idle

The $\{S, \sqrt{Y}\}$ set of the `CharacterGST` tutorial comes with an idle, $G_I$. A bare idle can never be a cGST germ — its ideal product is the identity, of group order 1, with nothing to filter on — so `finite_order_candidate_germs` excludes it (and `create_cgst_design` skips it with a warning if it is passed explicitly). The idle's twelve error-generator coefficients are amplified instead by finite-order germs that *contain* it, which is exactly what the search returns once the candidate list is deduplicated correctly:

```python
i_pspec = QubitProcessorSpec(1, ['Gzpi2', 'Gypi2', 'Gi'], qubit_labels=['Q0'])
i_target = create_explicit_model(i_pspec, ideal_gate_type='full TP', ideal_spam_type='full TP',
                                 simulator='matrix')

with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    i_germs = cgstdesign.find_cgst_germs(i_target, max_length=7, seed=0)
    i_design = cgstdesign.create_cgst_design(i_target, depths, circuits_per_depth=12, germs=i_germs,
                                             num_projection_rounds=3, seed=0)
    i_lean = cgstdesign.create_cgst_design(i_target, depths, circuits_per_depth=12, germs=i_germs,
                                           num_projection_rounds=3, seed=0,
                                           include_degenerate_blocks=False)

i_table = pd.DataFrame(i_design.germ_table)
for col in ('germ', 'prep_fiducial', 'meas_fiducial'):
    i_table[col] = [short(s) for s in i_table[col]]
print(len(i_germs), 'germs;', len(i_design.keys()), 'children, or',
      len(i_lean.keys()), 'without the degenerate blocks')
i_table.groupby(['germ', 'group_order', 'irrep_index', 'multiplicity']).size().rename('children').reset_index()
```

Three of the ten germs are $\pi$ rotations ($N = 2$) whose nontrivial block is two-dimensional: the *degenerate* case. `select_cgst_fiducials` emits a $2 \times 2$ grid of pairs (`pair0`…`pair3`) with a well-conditioned projected overlap matrix — the data needed to fit a $2 \times 2$ matrix decay is collected — but the analysis only fits scalar decays, so those twelve children would be skipped by the inversion; `include_degenerate_blocks=False` leaves them out. One caveat follows from this: pyGSTi's amplificational-completeness test counts the amplification those blocks would provide, so a germ set that is complete in GST's sense can still leave a direction unmeasured by cGST's scalar analysis (for this set it is the idle's $H(X)$, whose only idle-containing carriers are the $\pi$-rotation germs' degenerate blocks). The inversion reports that honestly through its rank.

## A truth model

Back to the XY gate set. The truth is a generic Lindblad model: Hamiltonian, stochastic *and* a couple of correlation/active terms at the $10^{-3}$ level. The C and A sectors need the unconstrained `'GLND'` parameterization.

```python
truth_rates = {
    'Gxpi2:Q0': {('H', 'X'): 0.0015, ('H', 'Z'): 0.0006,
                 ('S', 'X'): 0.0012, ('S', 'Y'): 0.0012, ('S', 'Z'): 0.0015,
                 ('C', 'X', 'Y'): 0.0004, ('A', 'Y', 'Z'): 0.0005},
    'Gypi2:Q0': {('H', 'Y'): 0.0012, ('H', 'X'): 0.0009,
                 ('S', 'X'): 0.0015, ('S', 'Y'): 0.0008, ('S', 'Z'): 0.0013,
                 ('A', 'X', 'Y'): 0.0004},
}
truth = create_explicit_model(pspec, lindblad_error_coeffs=truth_rates,
                              lindblad_parameterization='GLND', simulator='matrix')

ds = pygsti.data.simulate_data(truth, design.all_circuits_needing_data, num_samples=4000,
                               sample_error='multinomial', seed=2026)
data = ProtocolData(design, ds)
```

## The gauge-invariant check first

Before any inversion or gauge fixing, the fitted decays can be checked against the truth directly: `true_germ_eigenvalues` diagonalizes the true noisy germ and reports each irrep's eigenvalue deviation from ideal, with no approximation.

```python
t0 = time.time()
decays = CharacterGST(bootstrap_samples=0).run(data)
print('decay fits: %.1f s' % (time.time() - t0))

rows = []
for row in design.germ_table:
    child = design[row['name']]
    fit = decays[row['name']].for_protocol['CharacterDecay']
    exact = true_germ_eigenvalues(truth, child.germ, child.group_order)[child.irrep_index]
    rows.append({'child': row['name'], 'fitted |lambda|': fit.germ_eigenvalue_magnitude,
                 'true |lambda|': abs(exact), 'fitted arg': fit.germ_eigenvalue_phase,
                 'true arg': np.angle(exact)})
eigen = pd.DataFrame(rows)
eigen['d|lambda|'] = eigen['fitted |lambda|'] - eigen['true |lambda|']
eigen['d arg'] = eigen['fitted arg'] - eigen['true arg']
eigen.style.format({c: '{:+.5f}' for c in eigen.columns if c != 'child'})
```

Every row agrees with the exact eigenvalue at the shot-noise level. This check needs no gauge and no target model, which is why it is the right first thing to look at.

## The linear inversion

`gateset_inversion='linear'` assembles those decays into observables, solves the first-order linear system for the gates' elementary error generator coefficients, and reports the result in the standard gauge.

```python
t0 = time.time()
protocol = CharacterGST(gateset_inversion='linear', target_model=target,
                        reference_gate='Gxpi2', bootstrap_samples=50, seed=7)
results = protocol.run(data)
top = results.for_protocol['CharacterGST']
print('linear inversion: %.0f s' % (time.time() - t0))

info = top.inversion_info
print('observables %d, parameters %d, rank %d, unamplified %d, residual %.2e'
      % (info['num_observables'], info['num_params'], info['rank'],
         info['num_unamplified'], info['residual_norm']))
print('skipped children:', [short(n) for n in info['skipped_children']])
print('singular values:', np.array2string(np.array(info['singular_values']), precision=2))
```

Two gates $\times$ 12 elementary error generators = 24 parameters; 20 observables (two per child, none skipped); rank 13, with a five-order-of-magnitude gap in the singular values on either side of the cutoff, so the rank is unambiguous. That rank is the whole story of the design: the gauge group of a one-qubit two-gate model contributes 11 directions that no experiment can see (the twelfth TP gauge direction, the uniform Bloch-ball scaling, commutes with every ideal gate and so does not move the error generators at first order), leaving $24 - 11 = 13$ non-gauge directions — and the design amplifies all of them. A rank below 13 would mean the germ set, the irrep choice or the depth range was missing something.

## Comparing to the truth — in the same gauge

The coefficients typed into `truth_rates` are **not** what to compare against: they are written in whatever gauge `create_explicit_model` happened to produce, and cGST cannot see gauge. The comparison must be made after putting the truth into the *same* standard gauge the protocol reports in.

```python
truth_standard = cgstgauge.errorgen_coefficients_in_gauge(
    cgstgauge.fix_standard_gauge(truth, target, 'Gxpi2'), target)

rows = []
for gate_str, per_gate in top.errorgen_estimates.items():
    gate = next(g for g in truth_standard if str(g) == gate_str)
    for errgen_str, entry in per_gate.items():
        label = next(l for l in truth_standard[gate] if str(l) == errgen_str)
        typed = truth_rates[gate_str].get(
            tuple([label.errorgen_type] + [str(b).split(':')[0]
                                           for b in label.basis_element_labels]), 0.0)
        rows.append({'gate': gate_str, 'errorgen': errgen_str, 'estimate': entry['value'],
                     'stderr': entry['stderr'], 'truth (std gauge)': truth_standard[gate][label],
                     'as typed': typed})
comparison = pd.DataFrame(rows)
comparison['deviation'] = comparison['estimate'] - comparison['truth (std gauge)']
comparison['z'] = comparison['deviation'] / comparison['stderr'].clip(lower=1e-12)
comparison.style.format({'estimate': '{:+.2e}', 'stderr': '{:.1e}', 'truth (std gauge)': '{:+.2e}',
                         'as typed': '{:+.2e}', 'deviation': '{:+.1e}', 'z': '{:+.1f}'})
```

```python
print('max |deviation| = %.1e   max |z| = %.1f'
      % (comparison['deviation'].abs().max(), comparison['z'].abs().max()))
```

The `as typed` column is a different gauge, and it shows: $X_{\pi/2}$'s typed $H(Z) = 6 \times 10^{-4}$ and $C(X,Y) = 4 \times 10^{-4}$ are *identically zero* in the standard gauge. That is stage 1 at work — the reference gate's error generator is forced to commute with the ideal $X_{\pi/2}$, which leaves it only $H(X)$, $S(X)$, $S(Y) = S(Z)$ and $A(Y,Z)$ — and it is why comparing estimates against typed-in coefficients would look like a catastrophic failure of the protocol rather than a change of coordinates.

The estimates track the standard-gauge column, with the Hamiltonian sector by far the best determined and the correlation/active sector the noisiest. Do not read the $z$ column as a calibrated significance, though: entries of $|z| \sim 4$–$5$ occur here, and repeating the run with fresh shot noise shows why — the deviations average to zero (this is not first-order bias, which is an order of magnitude smaller at these rates; see the caveats) but their spread is two to three times the reported error bar. The error bars come from a residual bootstrap of each nine-point decay fit, which underestimates the spread of the fitted decay rates, propagated as if the observables were uncorrelated; treat them as optimistic by a factor of a few.

The estimated gate set itself is available as a model, and everything is tabulated by `to_dataframe`:

```python
print(top.estimated_model.operations[('Gxpi2', 'Q0')].to_dense().round(5))
top.to_dataframe().query("type == 'error generator'").head()
```

## What the standard gauge is

Reporting error generators requires a gauge convention. cGST uses the *standard gauge*, built in two stages by `cgstgauge`:

* **Stage 1 (exact).** The *reference gate* is brought to the gauge in which its error channel commutes with its ideal implementation, by matching eigenspaces of the noisy and ideal superoperators (`commuting_gauge_transform`). Here that is `reference_gate='Gxpi2'`; for the $\{S, \sqrt{Y}\}$ set it is $S$.
* **Stage 2 (least squares).** What remains is the group of TP gauge transformations commuting with the ideal reference gate (`tp_commutant_basis`; four-dimensional for a one-qubit $\pi/2$ rotation). It is spent minimizing the summed squared Frobenius error of the *other* gates. For $\{S, \sqrt{Y}\}$ this fixes the equatorial scale $\xi$, the axis scale $\zeta$ and the axis shift $\eta$, and puts the relational coherent error along $X$.

One direction is fixed by neither stage: gauge transformations commuting with *every* ideal gate — for an irreducible one-qubit gate set, the uniform Bloch-ball scaling $\mathrm{diag}(0,1,1,1)$ — do not move the gate errors at first order, and the Frobenius objective actually decreases monotonically as that scale shrinks. `standard_gauge_transform` splits those directions off and pins them by minimizing the distance of the model's SPAM from the target's (`fix_spam_gauge=True`, the default), which leaves them essentially untouched for cGST's reconstructions (whose SPAM is ideal) while making the gauge fixing well posed.

## Notes and caveats

* **First order only.** The inversion linearizes the observables about the ideal gate set, so the estimates carry an $O(\text{rate}^2)$ bias — but the relevant rate is the error *per germ repetition*, which for a length-7 germ is several times the per-gate rate. That bias is what the nonzero residual norm above reports. Re-running this notebook with `sample_error='none'` isolates it: the Hamiltonian, stochastic and correlation coefficients then come back to $1$–$2 \times 10^{-5}$, and the active sector to $\sim 3 \times 10^{-4}$ (see the next bullet) — i.e. at these rates, below the statistical error of a few thousand shots per circuit for everything but the active terms.
* **Unconstrained coefficients.** The least-squares solve knows nothing about complete positivity, so `S` coefficients can and do come out slightly negative when their true value is at or below the noise level. That is expected, not a failure.
* **Active (A-type) coefficients are the least accurate.** Their only first-order signature is the trivial-block observable $(1-\lambda)(B - C)$, a *product* of two first-order quantities, so its inversion carries an $O(\text{rate}^2)$ bias with a much larger prefactor than the eigenvalue observables: about $14\,\text{rate}^2$ on the $\{S, \sqrt{Y}\}$ model (verified to scale as $\text{rate}^2$ by halving every rate, and unchanged when `design_matrix_step` is raised to $10^{-3}$, so it is truncation, not finite-difference error). At $10^{-3}$ rates that is $\sim 10^{-5}$ — small, but the Hamiltonian sector does two orders of magnitude better.
* **Degenerate blocks.** Nontrivial irreps of multiplicity $> 1$ (and trivial blocks of multiplicity $\ne 2$, which includes most multi-qubit germs) are designed — the fiducial grid is emitted — but not analyzed; `include_degenerate_blocks=False` drops them. Matrix-valued decays are future work, and until then a germ set certified amplificationally complete by pyGSTi can still leave a direction unmeasured (see the idle example above).
* **The Jacobian is the expensive step.** Building the design matrix means re-fitting every decay curve $2 \times 24$ times: about 15 s for this 10-child one-qubit design (most of the 'linear inversion' time above); propagating the uncertainties through the linear solve and the gauge fit costs a couple of seconds more. It depends only on the design and the target model, so `CharacterGST` caches it in a class-level dictionary shared by all instances (and never serialized), keyed by the design's germs, fiducials, depths, sampling mode *and realized random germ powers* and by the target model's gates and SPAM; re-running the protocol on the same design, e.g. to change `bootstrap_samples`, is cheap.
* **Design the design for the Jacobian.** A trivial-block decay curve bends away from a straight line only by $\sim((1-\lambda) k_{\max})^2$, so the finite differences need `design_matrix_step * max(depths)` to be at least $\sim 10^{-2}$: use depths out to $\sim 128$. Keep the default `mode='exact'`: its quadrature over germ powers removes character-sampling noise entirely, which both sharpens the Jacobian and makes the whole pipeline deterministic. A `'reduced'`-mode design still works (the same realized germ powers enter the data and the Jacobian, so the inversion is self-consistent) but its Monte-Carlo projector leaves an $O(\text{error})$ ripple in every decay curve: the protocol warns, the estimates come out roughly 20–40$\times$ less accurate than in exact mode, and a spurious extra singular value of order 1 makes the reported rank unreliable. `'full'`-mode designs are rejected: the Fourier-operator eigenvalue inversion they require is not first-order linearizable.

```python
print('notebook run time: %.0f s' % (time.time() - notebook_start))
```

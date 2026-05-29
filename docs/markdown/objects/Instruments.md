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

# Instruments and Intermediate Measurements

This tutorial demonstrates how to model, simulate, and perform tomography on quantum *instruments*: maps that act on a qubit state (density matrix) and produce a qubit state *together with* a classical outcome.  Formally, an instrument is a map from $\mathcal{B}(\mathcal{H})$, the space of density matrices, to $\mathcal{B}(\mathcal{H}) \otimes K(n)$, where $K(n)$ is a classical space of $n$ elements.  Instruments are the natural model for **mid-circuit measurements** (MCMs): operations that read out a qubit partway through a circuit and leave behind a (possibly disturbed) post-measurement state.

In pyGSTi, an instrument is represented as a collection of operations -- one *member* for each classical outcome.  Writing the instrument as $\mathcal{I} = \{\mathcal{I}_k\}_k$, member $\mathcal{I}_k$ is a completely-positive, trace-*non-increasing* (CPTR) map, and the members jointly form a trace-preserving (CPTP) map $\sum_k \mathcal{I}_k$.  The probability of recording outcome $k$ on state $\rho$ is $\mathrm{tr}\,\mathcal{I}_k(\rho)$, and the (subnormalized) post-measurement state is $\mathcal{I}_k(\rho)$.

This tutorial pays particular attention to *how that collection of members is parameterized*, because the parameterization determines whether a fitted instrument is physically valid.  A recent characterization of mid-circuit measurement on a transmon qubit ([arXiv:2602.03938](https://arxiv.org/abs/2602.03938)) shows that an interpretable MCM model decomposes into physically meaningful error mechanisms -- amplitude damping, readout (assignment) error, and imperfect collapse.  We will build a data-generating instrument out of exactly these effects, then recover it with gate set tomography (GST) and show why a *completely-positive* parameterization matters.

We start with a few familiar imports:

```{code-cell} ipython3
import pygsti
from pygsti.modelpacks import smq1Q_XYI as std
import numpy as np
from pprint import pprint

from pygsti.modelmembers.instruments import (
    Instrument, TPInstrument, kraus_polar_instrument, convert)
from pygsti.modelmembers.operations import FullArbitraryOp
```

## Constructing an ideal instrument

We will add an instrument to our "standard" 1-qubit model -- which contains $I$, $X(\pi/2)$, and $Y(\pi/2)$ gates -- representing an ideal $Z$-basis measurement.  The instrument is named `"Iz"` (all instrument names must begin with `"I"`), and its members are the perfect projectors onto the 0 and 1 states.  Rather than labelling the outcomes `"0"` and `"1"`, we name them `"p0"` and `"p1"` so they are easy to distinguish from the *final* POVM outcomes, which are labelled `"0"` and `"1"`.

```{code-cell} ipython3
mdl_ideal = std.target_model()

# Build the ideal projective Z-measurement instrument from the POVM effects.
E0 = mdl_ideal.effects['0']
E1 = mdl_ideal.effects['1']
# Alternate indexing that names the POVM explicitly:
#   E0 = mdl_ideal['Mdefault']['0']   # 'Mdefault' = POVM label, '0' = effect label
Gmz_plus  = E0 @ E0.T   # note: effect vectors are stored as column vectors
Gmz_minus = E1 @ E1.T
mdl_ideal[('Iz', 0)] = Instrument({'p0': Gmz_plus, 'p1': Gmz_minus})
```

## How instruments are parameterized

Before we add noise, it is worth understanding the three ways pyGSTi can parameterize an instrument's members.  They differ in *which constraints they enforce on the fit*, and therefore in their parameter counts:

- **`Instrument`** stores each member as an independent, unconstrained dense superoperator.  Nothing ties the members together or keeps them physical -- during a fit the members can drift to maps that are neither completely positive nor jointly trace-preserving.
- **`TPInstrument`** constrains the members to *sum* to a trace-preserving map.  This is the right number of degrees of freedom for a physical instrument, but it constrains only the *sum*: an individual member can still be non-completely-positive (it can have negative Choi eigenvalues), which is unphysical.
- **Kraus-polar instruments** keep the whole instrument trace-preserving and build each member from a Kraus/polar decomposition (described below). This is a strategy that can be used to produce a family of Lindblad-like parameterizations -- `"CPTPLND"`, `"GLND"`, `"H+S"`, `"H+s"`, ... -- introduced for modeling physically-meaningful instruments. It is the focus of this tutorial. Whether each member is additionally *completely positive* is a choice you make through the parameterization; trace preservation is guaranteed either way.

The `convert` function moves a plain `Instrument` between parameterizations.  String parameterization types such as `"CPTPLND"` (the completely-positive, trace-preserving Lindbladian parameterization) and `"GLND"` (the general, trace-preserving-but-not-necessarily-CP Lindbladian) route to the Kraus-polar construction:

```{code-cell} ipython3
basis = mdl_ideal.basis
plain = Instrument({'p0': Gmz_plus, 'p1': Gmz_minus})

for to_type in ['static', 'H+s', 'full TP', 'full', 'CPTPLND', 'GLND']:
    conv = convert(plain, to_type, basis)
    print(f"{to_type:8s} -> {type(conv).__name__:14s} num_params = {conv.num_params}")
```

`"full TP"` yields a `TPInstrument`; the remaining types yield an `Instrument` built from the Kraus-polar construction.  All of these keep the instrument trace-preserving.  They differ in whether the *individual members* are also completely positive: `"CPTPLND"` and `"H+S"` constrain the members to be CP, while `"GLND"` and `"H+s"` are trace-preserving but allow non-CP members (`"GLND"` is the general Lindbladian; `"H+s"` is the non-CP-constrained cousin of the CP-constrained `"H+S"`).

### What the Kraus-polar representation does

For each member -- a CPTR map -- pyGSTi computes a Kraus decomposition and polar-decomposes each Kraus operator $K = u\,p^{1/2}$ into a unitary part $u$ and a positive-semidefinite part $p^{1/2}$.  The member is then represented as a composition of

1. a **root-conjugation** operator $\rho \mapsto p^{1/2}\,\rho\,p^{1/2}$ (a `RootConjOperator`), built from the positive-semidefinite part, and
2. a (parameterized) **post-conjugation** channel $\rho \mapsto u\,\rho\,u^{\dagger}$, ideally a unitary.

These two factors play different roles and are constrained differently:

- The positive-semidefinite parts must stay positive, and across the whole instrument they must sum to the identity -- i.e. they are the effects of a single POVM.  pyGSTi collects them into one shared `ComposedPOVM` whose error map is *always* promoted to a CP-constrained parameterization: `convert` uses the least-expressive CP parameterization that subsumes your request (`"GLND"` → `"CPTPLND"`, `"H+s"` → `"H+S"`).  This is what guarantees the root-conjugation factors are genuinely positive and that the instrument is trace-preserving.
- The post-conjugation channels take whatever parameterization you requested.  If that parameterization is itself CP-constrained (`"CPTPLND"`, `"H+S"`), each member is completely positive.  If it is only trace-preserving (`"GLND"`, `"H+s"`), the members remain jointly trace-preserving but may individually fail to be CP.

So complete positivity of the members is something you *opt into* by choosing a CP-constrained post-conjugation parameterization -- it is not an automatic consequence of the Kraus-polar structure, which by itself guarantees only trace preservation.

You can build such an instrument directly with `kraus_polar_instrument`, passing the members as dense superoperators:

```{code-cell} ipython3
kp = kraus_polar_instrument({'p0': Gmz_plus, 'p1': Gmz_minus}, basis)
print(type(kp).__name__, "num_params =", kp.num_params)
for outcome, member in kp.items():
    print(f"  member {outcome!s:6s}: {type(member).__name__}")
```

```{admonition} Instrument names and outcome tuples
:class: note
Instrument labels must begin with `"I"`.  A circuit that contains an instrument produces probabilities indexed by **outcome tuples** -- one entry per instrument plus one for the final POVM -- as we will see below.
```

## A physically-motivated noisy instrument

To generate realistic data we model a noisy $Z$ measurement out of the three error mechanisms identified in [arXiv:2602.03938](https://arxiv.org/abs/2602.03938):

- **Amplitude damping** ($\gamma$): $T_1$ relaxation $\lvert 1\rangle \to \lvert 0\rangle$ during the readout pulse.
- **Imperfect collapse** ($\theta$): a *weak* measurement that does not fully dephase the qubit, leaving residual coherence.  $\theta = 0$ is a perfect projective measurement.
- **Readout (assignment) error** ($\varepsilon$): the recorded classical label is occasionally flipped.

Each mechanism is a small, physically meaningful piece of Kraus algebra.  We compose amplitude damping with a weak measurement, then mix the recorded labels to add assignment error.  Because we build each member from explicit Kraus operators (via `FullArbitraryOp.from_kraus_operators`) and the assignment-error mix is a convex combination of completely-positive maps, the resulting instrument is guaranteed to be CP per-member and jointly trace-preserving:

```{code-cell} ipython3
def noisy_z_instrument(gamma, theta, eps):
    """A noisy Z-measurement instrument built from physical error mechanisms.

    gamma : amplitude damping (T1 relaxation 1->0 during readout)
    theta : weak-measurement angle (imperfect collapse; 0 = projective)
    eps   : readout assignment error (classical label flip)
    """
    # Amplitude-damping Kraus operators.
    A0 = np.array([[1, 0], [0, np.sqrt(1 - gamma)]], dtype=complex)
    A1 = np.array([[0, np.sqrt(gamma)], [0, 0]], dtype=complex)
    # Weak Z-measurement Kraus operators (theta=0 -> projective).
    M0 = np.array([[np.cos(theta), 0], [0, np.sin(theta)]], dtype=complex)
    M1 = np.array([[np.sin(theta), 0], [0, np.cos(theta)]], dtype=complex)

    # Each "true" outcome is weak-measurement-after-damping (Kraus rank 2).
    S0 = FullArbitraryOp.from_kraus_operators([M0 @ A0, M0 @ A1], 'pp').to_dense()
    S1 = FullArbitraryOp.from_kraus_operators([M1 @ A0, M1 @ A1], 'pp').to_dense()

    # Classical assignment error mixes the recorded labels.
    R0 = (1 - eps) * S0 + eps * S1
    R1 = eps * S0 + (1 - eps) * S1
    return Instrument({'p0': R0, 'p1': R1})
```

We build a noisy data-generating model: depolarize the gates and SPAM a little, and install a noisy `"Iz"`:

```{code-cell} ipython3
mdl_noisy = std.target_model().depolarize(op_noise=0.01, spam_noise=0.01)
mdl_noisy[('Iz', 0)] = noisy_z_instrument(gamma=0.05, theta=np.deg2rad(10), eps=0.02)
```

Each member of this instrument really is completely positive -- its superoperator has no negative Choi eigenvalues:

```{code-cell} ipython3
for outcome, member in mdl_noisy.instruments[('Iz', 0)].items():
    neg = pygsti.tools.sum_of_negative_choi_eigenvalues_gate(member.to_dense(), 'pp')
    print(f"  member {outcome!s:6s}: sum of negative Choi eigenvalues = {neg:.2e}")
```

## Generating probabilities

Instrument labels (e.g. `"Iz"`) may be included within `Circuit` objects, and `Model` objects compute probabilities for them just like ordinary operation sequences.  The difference is that probabilities are labeled by **outcome tuples** -- one entry for each instrument and one for the final POVM:

```{code-cell} ipython3
c = pygsti.circuits.Circuit((('Gxpi2', 0), ('Iz', 0)))
pprint(dict(mdl_noisy.probabilities(c)))
```

```{code-cell} ipython3
c = pygsti.circuits.Circuit((('Iz', 0), ('Gxpi2', 0), ('Iz', 0)))
pprint(dict(mdl_noisy.probabilities(c)))
```

In fact, pyGSTi *always* labels probabilities using outcome tuples; in the non-instrument case they are simply 1-tuples that, by `OutcomeLabelDict` magic, can be treated as if they were strings:

```{code-cell} ipython3
probs = mdl_ideal.probabilities(pygsti.circuits.Circuit([('Gxpi2', 0)]))
print("probs       = ", dict(probs))
print("probs['0']  = ", probs['0'])      # this works...
print("probs[('0',)] = ", probs[('0',)])  # and so does this.
```

## Performing tomography

Now we perform tomography on a model that includes an instrument.  We build an experiment design from our standard modelpack, adding the bare instrument as a germ so that its parameters are well-constrained:

```{code-cell} ipython3
germs = std.germs() + [pygsti.circuits.Circuit([('Iz', 0)])]
edesign = std.create_gst_experiment_design(max_max_length=4, germs=germs)
print("number of circuits:", len(edesign.all_circuits_needing_data))
```

### Simulated data generation

We generate data from `mdl_noisy` exactly as we would for any other model, and write it to disk so you can see how datasets look when they contain measurement data:

```{code-cell} ipython3
ds = pygsti.data.simulate_data(
    mdl_noisy, edesign.all_circuits_needing_data, 2000, 'multinomial', seed=2018)
pygsti.io.write_dataset("../../tutorial_files/intermediate_meas_dataset.txt", ds)
```

Notice the format of [intermediate_meas_dataset.txt](../../tutorial_files/intermediate_meas_dataset.txt): it includes a column for each distinct outcome tuple.  Since not every circuit contains data for every outcome tuple, `"--"` is used as a placeholder.

### Running GST under two parameterizations

We fit the data with `StandardGST` under two parameterizations: `"full TP"` (a `TPInstrument`, which constrains only the *sum* of the instrument members) and `"CPTPLND"` (the Kraus-polar representation, which makes *every* member completely positive).  Both use the ideal model as the target.

```{code-cell} ipython3
from pygsti.protocols import StandardGST, ProtocolData

gst = StandardGST(modes=('full TP', 'CPTPLND'), target_model=mdl_ideal, verbosity=2)
results = gst.run(ProtocolData(edesign, ds))
```

Both fits recover the data-generating model about equally well.  We compare each gauge-optimized estimate to the (ideal) target with the Frobenius distance:

```{code-cell} ipython3
for mode in ('full TP', 'CPTPLND'):
    mdl_go = results.estimates[mode].models['stdgaugeopt']
    print(f"{mode:8s}: Frobenius distance to target = {mdl_ideal.frobeniusdist(mdl_go):.4f}")
```

## Why a completely-positive parameterization matters

The two fits agree on the physics, but they differ in a way that matters for interpretation.  A `TPInstrument` only constrains the members to *sum* to a trace-preserving map -- so when fit to finite, noisy data, its individual members can drift slightly outside the set of completely-positive maps.  Such members have **negative Choi eigenvalues**, which makes them physically meaningless: they are not valid quantum operations on their own.  The Kraus-polar `"CPTPLND"` instrument cannot do this, because each member is completely positive by construction.

We can see the difference directly by summing the negative Choi eigenvalues of each fitted instrument member (a value of zero means the member is completely positive):

```{code-cell} ipython3
for mode in ('full TP', 'CPTPLND'):
    mdl_fit = results.estimates[mode].models['final iteration estimate']
    inst = mdl_fit.instruments[('Iz', 0)]
    print(f"{mode}  ({type(inst).__name__}, {inst.num_params} params):")
    for outcome, member in inst.items():
        viol = max(0.0, pygsti.tools.sum_of_negative_choi_eigenvalues_gate(member.to_dense(), 'pp'))
        print(f"    member {outcome!s:6s}: CP violation = {viol:.2e}")
```

The `"full TP"` fit produces small but nonzero CP violations, while the `"CPTPLND"` fit has none.  When a downstream analysis needs to treat each instrument member as a bona-fide quantum operation -- for example, to decompose it into the physically meaningful error mechanisms of [arXiv:2602.03938](https://arxiv.org/abs/2602.03938) -- the completely-positive parameterization is the one to use.

```{admonition} Reporting
:class: tip
GST results that include instruments can be rendered into an interactive HTML report just like any other GST result, e.g. with
`pygsti.report.construct_standard_report(results, title='MCM GST').write_html(...)`.
```

**That's it!**  You have built a physically-motivated instrument, simulated mid-circuit-measurement data, and performed tomography under two parameterizations -- seeing why a completely-positive representation gives instrument estimates you can interpret.
```

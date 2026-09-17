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

# Instruments and Intermediate Measurements

This tutorial demonstrates how to model, simulate, and perform tomography on quantum *instruments*: maps that act on a qudit state (density matrix) and produce a qudit state *together with* a classical outcome.  Formally, an instrument is a map from $\mathcal{B}(\mathcal{H})$, the space of density matrices, to $\mathcal{B}(\mathcal{H}) \otimes K(n)$, where $K(n)$ is a classical space of $n$ elements.  Instruments are the natural model for **mid-circuit measurements** (MCMs): operations that read out a qudit partway through a circuit and leave behind a (possibly disturbed) post-measurement state.

In pyGSTi, an instrument is represented as a collection of operations -- one *member* for each classical outcome.  Writing the instrument as $\mathcal{I} = \{\mathcal{I}_k\}_k$, member $\mathcal{I}_k$ is a completely-positive, trace-*non-increasing* (CPTR) map, and the members jointly form a trace-preserving (CPTP) map $\sum_k \mathcal{I}_k$.  The probability of recording outcome $k$ on state $\rho$ is $\mathrm{tr}\,\mathcal{I}_k(\rho)$, and the (subnormalized) post-measurement state is $\mathcal{I}_k(\rho)$.

This tutorial pays particular attention to *how that collection of members is parameterized*, because the parameterization determines whether a fitted instrument is physically valid.  A recent characterization of mid-circuit measurement on a transmon qubit ([arXiv:2602.03938](https://arxiv.org/abs/2602.03938)) shows that an interpretable MCM model decomposes into physically meaningful error mechanisms -- amplitude damping, readout (assignment) error, and imperfect collapse.  We will build a data-generating instrument out of these effects (although in a slightly different way than in the paper), then recover it with GST and show why a *completely-positive* parameterization matters.

We start with a few familiar imports:

```{code-cell} ipython3
import pygsti
from pygsti.modelpacks import smq1Q_XYI as std
import numpy as np
from pprint import pprint

from pygsti.modelmembers.instruments import Instrument, TPInstrument, convert
from pygsti.modelmembers.operations import FullArbitraryOp
```

## Constructing an ideal instrument

We will add an instrument to our "standard" 1-qubit model -- which contains $I$, $X(\pi/2)$, and $Y(\pi/2)$ gates -- representing an ideal $Z$-basis measurement.  The instrument is named `"Iz"` (all instrument names must begin with `"I"`), and its members are the perfect projectors onto the 0 and 1 states.  Rather than labelling the outcomes `"0"` and `"1"`, we name them `"p0"` and `"p1"` so they are easy to distinguish from the *final* POVM outcomes, which are labelled `"0"` and `"1"`.

```{code-cell} ipython3
mdl_ideal = std.target_model()

# Build the ideal projective Z-measurement instrument from the POVM effects.
E0 = mdl_ideal.effects['0'].to_dense()
E1 = mdl_ideal.effects['1'].to_dense()
# Alternate indexing that names the POVM explicitly:
#   E0 = mdl_ideal['Mdefault']['0'].to_dense()   # 'Mdefault' = POVM label, '0' = effect label
Gmz_plus  = np.outer(E0, E0)
Gmz_minus = np.outer(E1, E1)
mdl_ideal[('Iz', 0)] = Instrument({'p0': Gmz_plus, 'p1': Gmz_minus})
```

## How instruments are parameterized

Before we add noise, it is worth understanding how pyGSTi represents an instrument's members and what keeps a fitted instrument physical.  `Instrument` itself is just a container; the constraints that make a fit physical come from *how its members are parameterized*, and the options below differ in *which* constraints they enforce -- and therefore in their parameter counts:

- **`Instrument`** is a lightweight wrapper around pyGSTi LinearOperator objects. Whether the instrument is physical depends entirely on how the parameters of those objects relate to each other. You can ensure a physically meaningful instrument by calling `Instrument.from_effects` or `Instrument.from_cptr_superops` with appropriate arguments (the defaults suffice). The downside of physical instruments constructed in this way is that they use a higher-dimensional representation in order for pyGSTi to do model fitting.
- **`TPInstrument`** constrains the members to *sum* to a trace-preserving map.  This is the right number of degrees of freedom for a physical instrument, but it constrains only the *sum*: an individual member can still be non-completely-positive (it can have negative Choi eigenvalues), which is unphysical.
- **CP-constrained, Lindblad-parameterized instruments** are not a separate class -- they are `Instrument` objects whose members carry the effect-then-gate structure of the next section (exactly what `Instrument.from_effects` and `Instrument.from_cptr_superops` build).  They keep the whole instrument trace-preserving *and* make each member individually completely positive, and support a family of Lindblad-like parameterizations (`"CPTPLND"`, `"GLND"`, `"H+S"`, `"H+s"`, ...).  Whether each member is additionally *completely positive* is a choice you make through the parameterization; trace preservation is guaranteed either way.

The `convert` function moves a plain `Instrument` between parameterizations.  String parameterization types such as `"CPTPLND"` (the completely-positive, trace-preserving Lindbladian parameterization) and `"GLND"` (the general, trace-preserving-but-not-necessarily-CP Lindbladian) route to the effect-then-gate construction described below:

```{code-cell} ipython3
basis = mdl_ideal.basis
plain = Instrument({'p0': Gmz_plus, 'p1': Gmz_minus})

for to_type in ['static', 'H+s', 'full TP', 'full', 'CPTPLND', 'GLND']:
    conv = convert(plain, to_type, basis)
    print(f"{to_type:8s} -> {type(conv).__name__:14s} num_params = {conv.num_params}")
```

`"full TP"` yields a `TPInstrument`; the remaining types yield an `Instrument` built from the effect-then-gate construction.  As the printed type names show, a CP-constrained instrument is still an `Instrument` -- the physical constraints live in how its members are parameterized, not in a distinct container class.  All of these keep the instrument trace-preserving.  They differ in whether the *individual members* are also completely positive: `"CPTPLND"` and `"H+S"` constrain the members to be CP, while `"GLND"` and `"H+s"` are trace-preserving but allow non-CP members (`"GLND"` is the general Lindbladian; `"H+s"` is the non-CP-constrained cousin of the CP-constrained `"H+S"`).

### The effect-then-CPTP-gate representation

Every completely-positive instrument member factors as a **measurement effect followed by a post-measurement CPTP gate**:

$$\mathcal{I}_k(\rho) = \mathcal{G}_k\!\left( E_k^{1/2}\,\rho\,E_k^{1/2} \right), \qquad E_k = \mathcal{I}_k^{\dagger}(I).$$

This is a "soft" measurement of the POVM effect $E_k$ -- which sets the outcome probability $\mathrm{tr}\,\mathcal{I}_k(\rho)$ -- followed by the trace-preserving back-action $\mathcal{G}_k$ on the surviving state.  The effect $E_k$ is just the Heisenberg-dual of the member applied to the identity.

pyGSTi represents each member as a single `ComposedOp([RootConjOperator(E_k), G_k])`, and the two physical guarantees come apart cleanly:

- **Trace preservation** of the *whole instrument* is exactly the statement that the effects $\{E_k\}$ form one POVM, $\sum_k E_k = I$.  pyGSTi gathers them into a single shared `ComposedPOVM` whose error map is *always* promoted to a CP-constrained parameterization (`convert` uses the least-expressive CP parameterization that subsumes your request: `"GLND"` → `"CPTPLND"`, `"H+s"` → `"H+S"`).  This keeps every $E_k$ positive and their sum equal to the identity.
- **Complete positivity** of an *individual member* is the statement that its post-measurement gate $\mathcal{G}_k$ is CP (the `RootConjOperator` is always CP).  So you *opt into* per-member CP by choosing a CP-constrained gate parameterization (`"CPTPLND"`, `"H+S"`); a trace-preserving-only choice (`"GLND"`, `"H+s"`, `"full TP"`) keeps the instrument TP but allows non-CP members.

An $n$-outcome instrument therefore needs only **one POVM ($n$ effects) and $n$ gates**, regardless of any member's Kraus rank.  Two constructors build instruments in this form directly: `Instrument.from_effects`, when you have the measurement effects $\{E_k\}$ (and, optionally, explicit post-measurement gates), and `Instrument.from_cptr_superops`, when you have arbitrary dense CPTR member superoperators and want pyGSTi to recover the $(E_k, \mathcal{G}_k)$ decomposition for you:

```{code-cell} ipython3
# From the measurement effects (post-measurement gates default to the identity):
fe = Instrument.from_effects({'p0': E0, 'p1': E1}, basis)
print("from_effects      :", type(fe).__name__, "num_params =", fe.num_params)

# From arbitrary dense CPTR superoperators (here the ideal projectors):
cptr = Instrument.from_cptr_superops(
    {'p0': Gmz_plus, 'p1': Gmz_minus},
    basis,
    gate_parameterization='CPTPLND'  # 'GLND' / 'H+s' / 'full TP' relax the per-member CP constraint
)
print("from_cptr_superops:", type(cptr).__name__, "num_params =", cptr.num_params)
for outcome, member in cptr.items():
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

We fit the data with `StandardGST` under two parameterizations: `"full TP"` (a `TPInstrument`, which constrains only the *sum* of the instrument members) and `"CPTPLND"` (the effect-then-CPTP-gate representation, which makes *every* member completely positive).  Both use the ideal model as the target.

```{code-cell} ipython3
from pygsti.protocols import StandardGST, ProtocolData

gst = StandardGST(modes=('full TP', 'CPTPLND'), target_model=mdl_ideal, verbosity=2)
results = gst.run(ProtocolData(edesign, ds))
```

Both fits recover the data-generating model fairly well. We compare each gauge-optimized estimate to the (ideal) target with the Frobenius distance:

```{code-cell} ipython3
for mode in ('full TP', 'CPTPLND'):
    mdl_go = results.estimates[mode].models['stdgaugeopt']
    print(f"{mode:8s}: Frobenius distance to target = {mdl_ideal.frobeniusdist(mdl_go):.4f}")
```

## Why a completely-positive parameterization matters

A `TPInstrument` only constrains the members to sum to a trace-preserving map -- so when fit to finite, noisy data, its individual members can drift slightly outside the set of completely-positive maps.  Such members have **negative Choi eigenvalues**, which makes them physically meaningless: they are not valid quantum operations on their own.  The `"CPTPLND"` instrument cannot do this, because each member is completely positive by construction.

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

The `"full TP"` fit produces small but nonzero CP violations, while the `"CPTPLND"` fit has none.  Here too, the reported type confirms the framing from the start of the tutorial: the CP-constrained estimate is an ordinary `Instrument` (the `"full TP"` estimate is a `TPInstrument`), and its members are completely positive because of how they are parameterized -- the higher-dimensional representation noted earlier -- not because of the container type.  When a downstream analysis needs to treat each instrument member as a bona-fide quantum operation -- for example, to decompose it into the physically meaningful error mechanisms of [arXiv:2602.03938](https://arxiv.org/abs/2602.03938) -- the completely-positive parameterization is the one to use.

```{admonition} Reporting
:class: tip
GST results that include instruments can be rendered into an interactive HTML report just like any other GST result, e.g. with
`pygsti.report.construct_standard_report(results, title='MCM GST').write_html(...)`.
```

**That's it!**  You have built a physically-motivated instrument, simulated mid-circuit-measurement data, and performed tomography under two parameterizations -- seeing why a completely-positive representation gives instrument estimates you can interpret.
```

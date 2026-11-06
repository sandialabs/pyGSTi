---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.19.4
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# A qubit with incoherent leakage

Represent the system on three levels, with allowed states

$$\rho=\begin{pmatrix}R&0\\0&p\end{pmatrix},\qquad R\succeq0,\quad p\geq0,\quad\operatorname{Tr}R+p=1.$$

The computational block $R$ is an unnormalized qubit density matrix. Its trace can change through leakage and return. Qubit coherence is allowed; only coherence between the computational block and level 2 is excluded. We will construct CPTPLND gates that preserve this structure for every parameter vector, including after exponentiation and composition, fit the resulting model to GST data, and generate a leakage report.

The workflow follows [Leakage (Automatic)](Leakage-automagic): an experiment design for the qubit model pack `smq1Q_XYI`, a three-level target from `leaky_qubit_model_from_pspec`, and `construct_leakage_report`. The difference is the hypothesis class. That notebook fits a general qutrit CPTPLND model; this one fits a sparse model of incoherent leakage, and compares the two fits.

## The restricted generator

A general Lindblad generator on this state algebra can be expressed as

$$\begin{aligned}
\dot R&=-i[H_q,R]+\mathcal D_{K_q}(R)-\tfrac12\{\Gamma,R\}+pT,\\
\dot p&=\operatorname{Tr}(\Gamma R)-p\operatorname{Tr}T,
\end{aligned}$$

where $K_q\succeq0$ is a $3\times3$ qubit Kossakowski matrix, and $\Gamma,T\succeq0$ are $2\times2$ matrices. $\mathcal D_{K_q}$ is the trace-preserving qubit dissipator. Factorizations $\Gamma=\sum_a v_av_a^\dagger$ and $T=\sum_b w_bw_b^\dagger$ correspond to leakage jumps $|2\rangle\langle v_a|$ and return jumps $|w_b\rangle\langle2|$. Thus this generator has a qutrit Lindblad extension and its exponential is CPTP.

The loss rate $\operatorname{Tr}(\Gamma R)$ can depend on qubit coherence. Likewise, off-diagonal entries of $T$ allow return into a qubit superposition. Neither effect creates coherence with level 2. Taking $\Gamma$ and $T$ diagonal would impose an additional restriction.

There are 20 physical generator degrees of freedom in the interior: 3 Hamiltonian, 9 in $K_q$, and 4 each in $\Gamma$ and $T$. This is a statement about generators; a single exponential does not cover every block-preserving CPTP channel.

## Expressing it with sparse S/C/A coefficients

The existing Hermitian-basis interface gives a convenient, redundant construction. Divide selected Gell-Mann directions into

- computational directions $X_{01},Y_{01},Z_1$;
- transfer directions $X_{02},X_{12},Y_{02},Y_{12}$.

Allow S on every selected direction and C/A only within each group. The Kossakowski graph is two disjoint cliques, so ordinary fill-free sparse Cholesky applies, including PSD boundary points and zero noise. Hamiltonian directions are confined to the computational group.

With $Z=\operatorname{diag}(1,1,-1)$, computational directions are even under $F\mapsto ZFZ$ and transfer directions are odd. The coefficient mask forbids C/A pair terms between even and odd directions. Each resulting Lindblad term preserves the block-diagonal state space. Merely removing Hamiltonian leakage terms would not impose this condition on a general dissipator.

The omitted $Z_2$ direction is scalar on each block: its stochastic dissipator vanishes on allowed states, and its C/A pair terms with computational directions reduce to Hamiltonian terms already included. Keeping it would add more redundant parameters.

The error generators act on the state space of the model pack's three-level target, in its `l2p1` matrix basis.

```{code-cell} ipython3
import itertools
import numpy as np

from pygsti.modelpacks import smq1Q_XYI as mp
from pygsti.leakage import leaky_qubit_model_from_pspec
from pygsti.baseobjs import Basis
from pygsti.baseobjs.errorgenlabel import LocalElementaryErrorgenLabel as LEEL
from pygsti.modelmembers.operations import LindbladErrorgen
from pygsti.modelmembers.operations.lindbladcoefficients import LindbladCoefficientBlock

target_model = leaky_qubit_model_from_pspec(mp.processor_spec(), mx_basis='l2p1')
target_model.sim = 'matrix'  # Analytic derivatives; the default here is 'map'.
leaky_space = target_model.state_space
leaky_basis = target_model.basis  # Matrix coordinates; includes all qutrit coherences.
leaky_directions = Basis.cast('GM', 9)  # Hermitian, identity-first Lindblad directions.
computational = ('X_{0,1}', 'Y_{0,1}', 'Z_{1}')
transfer = ('X_{0,2}', 'X_{1,2}', 'Y_{0,2}', 'Y_{1,2}')


def incoherent_leakage_errorgen(initial_rate=0.):
    """Example factory: arbitrary parameters preserve the 2+1 block structure."""
    if not np.isfinite(initial_rate) or initial_rate < 0:
        raise ValueError('initial_rate must be finite and nonnegative')
    ham = LindbladCoefficientBlock(
        'ham', leaky_directions, computational, param_mode='elements'
    )
    labels = [LEEL('S', (b,)) for b in computational + transfer]
    for group in (computational, transfer):
        for pair in itertools.combinations(group, 2):
            labels.extend(LEEL(sector, pair) for sector in ('C', 'A'))
    sca = LindbladCoefficientBlock(
        'other_sparse', leaky_directions, param_mode='cholesky',
        error_generator_labels=labels
    )
    sca.block_data[:] = [initial_rate if label.errorgen_type == 'S' else 0.
                         for label in labels]
    return LindbladErrorgen(
        [ham, sca], mx_basis=leaky_basis, state_space=leaky_space
    )


assert incoherent_leakage_errorgen(1e-4).num_params == 28  # 3 H + 3**2 + 4**2 S/C/A.
```

The transfer block has 16 parameters, but only 8 combinations act on allowed states: those specifying $\Gamma$ and $T$. The other 8 change the evolution of excluded qubit–leakage coherences. This redundancy is separate from ordinary GST gauge freedom. The example expresses the full restricted generator family, but does not provide a minimal 20-parameter fitting chart.

## States, gates, and measurements

Every allowed state can be written as $(BB^\dagger)\oplus|b|^2$, divided by its trace. We keep the target's ideal members as fixed references and attach an independent block-preserving CPTP Lindblad error channel to each: after the preparation $|0\rangle\langle0|$, before the measurement, and after each of the three lifted qubit gates. All five error channels have the same 28-parameter hypothesis class. Updates keep the state and POVM physical and block diagonal.

The target's measurement has two outcomes and reads level 2 as `1`, as many qubit readouts do. Leakage is then visible only indirectly, because leaked population does not respond to later rotations. A resolved leakage outcome would make it directly observable without changing the preservation condition.

```{code-cell} ipython3
from pygsti.modelmembers.operations import ExpErrorgenOp, StaticArbitraryOp, ComposedOp
from pygsti.modelmembers.states import StaticState, ComposedState
from pygsti.modelmembers.povms import StaticPOVMEffect, UnconstrainedPOVM, ComposedPOVM


def with_block_preserving_noise(target, new_errorgen):
    """Copy `target`, composing each member with an error channel from `new_errorgen()`."""
    model = target.copy()
    reference_state = StaticState(
        target.preps['rho0'].to_dense(), basis=leaky_basis, state_space=leaky_space
    )
    model.preps['rho0'] = ComposedState(reference_state, ExpErrorgenOp(new_errorgen()))
    reference_povm = UnconstrainedPOVM([
        (label, StaticPOVMEffect(effect.to_dense(), basis=leaky_basis, state_space=leaky_space))
        for label, effect in target.povms['Mdefault'].items()
    ])
    model.povms['Mdefault'] = ComposedPOVM(
        ExpErrorgenOp(new_errorgen()), reference_povm, mx_basis=leaky_basis
    )
    for label, op in target.operations.items():
        ideal = StaticArbitraryOp(op.to_dense(), basis=leaky_basis, state_space=leaky_space)
        model.operations[label] = ComposedOp([ideal, ExpErrorgenOp(new_errorgen())])
    return model


leaky_model = with_block_preserving_noise(
    target_model, lambda: incoherent_leakage_errorgen(1e-4)
)
print(list(leaky_model.operations.keys()))
assert leaky_model.num_params == 140  # Independent 28-parameter gate and SPAM errors.
print(leaky_model.probabilities((('Gxpi2', 0), ('Gypi2', 0), ('Gxpi2', 0))))
```

## Checking the hypothesis class

In `l2p1` coordinates, allowed states have support on `C[I]`, `C[X]`, `C[Y]`, `C[Z]`, and `L[I]`. The remaining four coordinates are excluded coherences. The following checks vary all noise parameters, check the maps on a basis of the entire allowed operator algebra, and propagate a gate sequence. Preservation is guaranteed by the construction above; these numerical checks also catch basis or composition mistakes.

```{code-cell} ipython3
from pygsti.models import ExplicitOpModel
from pygsti.tools.basistools import stdmx_to_vec, vec_to_stdmx
from pygsti.tools.jamiolkowski import jamiolkowski_iso

allowed = [leaky_basis.labels.index(b) for b in ('C[I]', 'C[X]', 'C[Y]', 'C[Z]', 'L[I]')]
excluded = [i for i in range(9) if i not in allowed]
trace_vector = stdmx_to_vec(np.eye(3), leaky_basis).ravel()
check_model = leaky_model.copy()
rng = np.random.default_rng(25)


def check_channel(channel):
    np.testing.assert_allclose(channel[np.ix_(excluded, allowed)], 0., atol=1e-12)
    np.testing.assert_allclose(trace_vector @ channel, trace_vector, atol=1e-12)
    choi = jamiolkowski_iso(channel, op_mx_basis=leaky_basis, choi_mx_basis='gm')
    assert np.linalg.eigvalsh(choi)[0] >= -1e-12


def check_state(vector):
    rho = vec_to_stdmx(vector, leaky_basis)
    np.testing.assert_allclose(rho[:2, 2], 0., atol=1e-12)
    assert np.linalg.eigvalsh(rho)[0] >= -1e-12
    np.testing.assert_allclose(np.trace(rho), 1., atol=1e-12)


for _ in range(3):
    check_model.from_vector(rng.normal(scale=.03, size=check_model.num_params))
    # Check both SPAM channels as well as the post-gate error channels.
    channels = [check_model.preps['rho0'].error_map, check_model.povms['Mdefault'].error_map]
    channels += [op.factorops[1] for op in check_model.operations.values()]
    for noise in channels:
        check_channel(noise.to_dense())
    vector = check_model.preps['rho0'].to_dense().ravel()
    check_state(vector)
    for label in [('Gxpi2', 0), ('Gypi2', 0), (), ('Gxpi2', 0), ('Gypi2', 0)]:
        channel = check_model.operations[label].to_dense()
        check_channel(channel)
        vector = channel @ vector
        check_state(vector)
    effect_sum = np.zeros((3, 3), complex)
    for effect in check_model.povms['Mdefault'].values():
        effect_matrix = vec_to_stdmx(effect.to_dense(), leaky_basis)
        effect_sum += effect_matrix
        np.testing.assert_allclose(effect_matrix[:2, 2], 0., atol=1e-12)
        assert np.linalg.eigvalsh(effect_matrix)[0] >= -1e-12
    np.testing.assert_allclose(effect_sum, np.eye(3), atol=1e-12)

restored_model = ExplicitOpModel.from_nice_serialization(check_model.to_nice_serialization())
restored_model.from_vector(rng.normal(scale=.02, size=check_model.num_params))
for op in restored_model.operations.values():
    np.testing.assert_allclose(op.to_dense()[np.ix_(excluded, allowed)], 0., atol=1e-12)
```

The retained five coordinates support an exact reduced simulation, since the discarded coordinates are never populated. This example deliberately keeps the full qutrit representation and uses existing model members; it does not implement a five-dimensional simulator or remove the redundant transfer parameters.

## Simulating data

The data-generating model lies in the same hypothesis class. Its parameters are drawn at random: Hamiltonian coefficients of order $10^{-2}$ radians, and sparse Cholesky factors of order $3\times10^{-2}$. Each squared factor contributes about $10^{-3}$, and contributions add over the four transfer directions, so a gate leaks or returns population with probability of order $10^{-2}$.

Each gate's leakage profile is printed below: the largest and smallest probability that one application moves a computational state to level 2, followed by the corresponding seepage (return) profile. These quantities are unchanged by gauge transformations that are unitary on each block, which is the freedom the leakage report fixes last.

```{code-cell} ipython3
from pygsti.data import simulate_data
from pygsti.protocols import ProtocolData
from pygsti.leakage.metrics import gate_leakage_profile, gate_seepage_profile

truth_rng = np.random.default_rng(2026)


def random_incoherent_leakage_errorgen():
    errorgen = incoherent_leakage_errorgen()
    errorgen.from_vector(np.concatenate([
        truth_rng.normal(scale=.01, size=3),   # Hamiltonian coefficients.
        truth_rng.normal(scale=.03, size=25),  # Sparse Cholesky factor.
    ]))
    return errorgen


true_model = with_block_preserving_noise(target_model, random_incoherent_leakage_errorgen)


def print_leakage(model):
    for label, op in model.operations.items():
        leak = gate_leakage_profile(op.to_dense(), leaky_basis)[0]
        seep = gate_seepage_profile(op.to_dense(), leaky_basis)[0]
        print(f'{str(label):12s} leakage {leak.max():.2e} to {leak.min():.2e}, '
              f'seepage {seep.max():.2e}')


print_leakage(true_model)

edesign = mp.create_gst_experiment_design(max_max_length=8)
dataset = simulate_data(true_model, edesign.all_circuits_needing_data,
                        num_samples=1000, seed=1997)
data = ProtocolData(edesign, dataset)
print(len(edesign.all_circuits_needing_data), 'circuits')
```

## Fitting the restricted model

`GateSetTomography` fits a user-supplied model when it is given as the initial model, with the model pack's target for comparison. The seed uses stochastic rates of $10^{-4}$ rather than zero: at zero rates the Cholesky factors vanish, and with them 25 of the 28 derivative columns of each error channel, so the optimizer could not move them.

The protocol's own gauge optimization is switched off. The leakage report below performs a leakage-aware gauge optimization that, for models built from `ComposedState`, `ComposedOp`, and `ComposedPOVM`, preserves the parameterization, and hence the block structure.

```{code-cell} ipython3
from pygsti.protocols import GateSetTomography, GSTInitialModel

seed_model = with_block_preserving_noise(
    target_model, lambda: incoherent_leakage_errorgen(1e-4)
)
gst = GateSetTomography(
    GSTInitialModel(seed_model, target_model=target_model),
    gaugeopt_suite=None, optimizer={'maxiter': 300},
    name='incoherent leakage', verbosity=1,
)
results = gst.run(data, disable_checkpointing=True)
fit = results.estimates['incoherent leakage'].models['final iteration estimate']
print_leakage(fit)
```

The largest leakage probabilities come out at the right order of magnitude. Seepage and the smallest leakage probabilities are much less certain. Level 2 is never prepared and reads as `1`, so return can only be seen acting on population that noise has already leaked, and this design constrains it weakly. A resolved leakage outcome, longer circuits, or more shots would all help.

## A general qutrit fit for comparison

For comparison, fit the same data with the general qutrit CPTPLND model used in [Leakage (Automatic)](Leakage-automagic). It contains the restricted model, but has 72 rather than 28 parameters per error channel.

Expect warnings that this fit treats its result as converged after reaching the maximum number of iterations. The optimizer is making slow progress along directions the data barely constrain. When this example was written, letting it run to convergence took about three times as long and lowered $2\Delta\log L$ by less than one, which does not affect the comparison below.

```{code-cell} ipython3
from pygsti.protocols import StandardGST
from pygsti.tools import two_delta_logl

general_results = StandardGST(
    modes=('CPTPLND',), target_model=target_model, gaugeopt_suite=None, verbosity=1,
).run(data, disable_checkpointing=True)
general_fit = general_results.estimates['CPTPLND'].models['final iteration estimate']


def excluded_coherence(model):
    """Largest map from an allowed state coordinate to an excluded coherence."""
    return max(np.abs(op.to_dense()[np.ix_(excluded, allowed)]).max()
               for op in model.operations.values())


print(f'{"model":20s} {"params":>6s} {"2ΔlogL":>8s} {"excluded":>9s}')
for name, model in [('data-generating', true_model), ('incoherent leakage', fit),
                    ('general CPTPLND', general_fit)]:
    print(f'{name:20s} {model.num_params:6d} {two_delta_logl(model, dataset):8.1f} '
          f'{excluded_coherence(model):9.2e}')

assert two_delta_logl(fit, dataset) < two_delta_logl(true_model, dataset)
assert two_delta_logl(true_model, dataset) - two_delta_logl(general_fit, dataset) < 220
assert excluded_coherence(fit) < 1e-12 < 1e-3 < excluded_coherence(general_fit)
```

The restricted fit reaches a lower $2\Delta\log L$ than the data-generating model, as a converged maximum-likelihood fit within the true hypothesis class should. The general model lowers it further, but by far less than its 220 additional nominal parameters. Many of those parameters are unidentifiable from this design, so this is a sanity check rather than a formal likelihood-ratio test. The last column shows what the extra freedom buys: the general fit maps allowed states into qubit–leakage coherences that this design never prepares or measures, while the restricted fit cannot.

## The leakage report

`construct_leakage_report` adds a leakage-aware gauge-optimized model, `LAGO`, to each estimate and builds a standard report. The report detects the leakage basis and shows subspace metrics by default; its "Metrics" switch shows full-space metrics instead. Merging the general fit into `results` puts both estimates in one report.

```{code-cell} ipython3
from pygsti.leakage import construct_leakage_report

results.add_estimates(general_results)
report, gauge_optimized = construct_leakage_report(results, title='Incoherent leakage')
lago = gauge_optimized.estimates['incoherent leakage'].models['LAGO']
assert lago.num_params == fit.num_params
assert excluded_coherence(lago) < 1e-12
report.write_html('../../example_files/incoherent-leakage-report')
```

Open the report [here](../../example_files/incoherent-leakage-report/main.html).

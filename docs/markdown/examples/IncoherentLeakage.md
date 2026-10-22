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

# A qubit with incoherent leakage

Represent the system on three levels, with allowed states

$$\rho=\begin{pmatrix}R&0\\0&p\end{pmatrix},\qquad R\succeq0,\quad p\geq0,\quad\operatorname{Tr}R+p=1.$$

The computational block $R$ is an unnormalized qubit density matrix. Its trace can change through leakage and return. Qubit coherence is allowed; only coherence between the computational block and level 2 is excluded. We will construct CPTPLND gates that preserve this structure for every parameter vector, including after exponentiation and composition.

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

With $Z=\operatorname{diag}(1,1,-1)$, computational directions are even under $F\mapsto ZFZ$ and transfer directions are odd. The coefficient mask forbids correlations between even and odd directions. Each resulting Lindblad term preserves the block-diagonal state space. Merely removing Hamiltonian leakage terms would not impose this condition on a general dissipator.

The omitted $Z_2$ direction is scalar on each block: its stochastic dissipator vanishes on allowed states, and its correlations with computational directions reduce to Hamiltonian terms already included. Keeping it would add more redundant parameters.

```{code-cell} ipython3
import itertools
import numpy as np

from pygsti.baseobjs import Basis
from pygsti.baseobjs.statespace import QuditSpace
from pygsti.baseobjs.errorgenlabel import LocalElementaryErrorgenLabel as LEEL
from pygsti.modelmembers.operations import (
    LindbladErrorgen, ExpErrorgenOp, StaticUnitaryOp, ComposedOp
)
from pygsti.modelmembers.operations.lindbladcoefficients import LindbladCoefficientBlock

leaky_space = QuditSpace(('T0',), 3)
leaky_basis = Basis.cast('l2p1', 9)  # Matrix coordinates; includes all qutrit coherences.
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


leaky_errorgen = incoherent_leakage_errorgen(1e-4)
assert leaky_errorgen.num_params == 28  # 3 H + 3**2 + 4**2 S/C/A.
```

The transfer block has 16 parameters, but only 8 combinations act on allowed states: those specifying $\Gamma$ and $T$. The other 8 change the evolution of excluded qubit–leakage coherences. This redundancy is separate from ordinary GST gauge freedom. The example expresses the full restricted generator family, but does not provide a minimal 20-parameter fitting chart.

## States, gates, and measurements

Every allowed state can be written as $(BB^\dagger)\oplus|b|^2$, divided by its trace. Preparation uses an independent block-preserving CPTP Lindblad error channel after a fixed reference state; measurement uses another before a fixed reference POVM. Both error channels have the same 28-parameter hypothesis class as the post-gate errors. Their parameters are independent, and updates keep the state and POVM physical and block diagonal.

```{code-cell} ipython3
from pygsti.models import ExplicitOpModel
from pygsti.modelmembers.states import StaticState, ComposedState
from pygsti.modelmembers.povms import StaticPOVMEffect, UnconstrainedPOVM, ComposedPOVM
from pygsti.tools.basistools import stdmx_to_vec, vec_to_stdmx

leaky_model = ExplicitOpModel(leaky_space, basis=leaky_basis)
reference_state = StaticState(
    stdmx_to_vec(np.diag([1., 0., 0.]), leaky_basis),
    basis=leaky_basis, state_space=leaky_space
)
leaky_model.preps['rho0'] = ComposedState(
    reference_state, ExpErrorgenOp(incoherent_leakage_errorgen(1e-4))
)

reference_povm = UnconstrainedPOVM([
    (label, StaticPOVMEffect(
        stdmx_to_vec(np.diag(np.eye(3)[i]), leaky_basis),
        basis=leaky_basis, state_space=leaky_space
    )) for i, label in enumerate(('0', '1', 'L'))
])
leaky_model.povms['Mdefault'] = ComposedPOVM(
    ExpErrorgenOp(incoherent_leakage_errorgen(1e-4)), reference_povm,
    mx_basis=leaky_basis
)

for name, pauli in (('Gx', np.array([[0., 1.], [1., 0.]])),
                    ('Gy', np.array([[0., -1j], [1j, 0.]]))):
    u3 = np.eye(3, dtype=complex)
    u3[:2, :2] = (np.eye(2) - 1j * pauli) / np.sqrt(2)
    ideal = StaticUnitaryOp(u3, basis=leaky_basis, state_space=leaky_space)
    leaky_model.operations[name] = ComposedOp([
        ideal, ExpErrorgenOp(incoherent_leakage_errorgen(1e-4))
    ])

assert leaky_model.num_params == 112  # Independent 28-parameter gate and SPAM errors.
print(leaky_model.probabilities(('Gx', 'Gy', 'Gx')))
```

The resolved leakage outcome `L` is convenient here. Effects may instead be combined to describe a two-outcome detector. This changes observability, not the preservation condition.

## Checking the hypothesis class

In `l2p1` coordinates, allowed states have support on `C[I]`, `C[X]`, `C[Y]`, `C[Z]`, and `L[I]`. The remaining four coordinates are excluded coherences. The following checks vary all noise parameters, check the maps on a basis of the entire allowed operator algebra, and propagate a gate sequence. Preservation is guaranteed by the construction above; these numerical checks also catch basis or composition mistakes.

```{code-cell} ipython3
from pygsti.tools.jamiolkowski import jamiolkowski_iso

allowed = [leaky_basis.labels.index(b) for b in ('C[I]', 'C[X]', 'C[Y]', 'C[Z]', 'L[I]')]
excluded = [i for i in range(9) if i not in allowed]
trace_vector = stdmx_to_vec(np.eye(3), leaky_basis).ravel()
rng = np.random.default_rng(25)

for _ in range(3):
    leaky_model.from_vector(rng.normal(scale=.03, size=leaky_model.num_params))
    # Check both SPAM channels as well as the post-gate error channels.
    channels = [leaky_model.preps['rho0'].error_map,
                leaky_model.povms['Mdefault'].error_map]
    channels += [op.factorops[1] for op in leaky_model.operations.values()]
    for noise in channels:
        channel = noise.to_dense()
        np.testing.assert_allclose(channel[np.ix_(excluded, allowed)], 0., atol=1e-12)
        np.testing.assert_allclose(trace_vector @ channel, trace_vector, atol=1e-12)
        choi = jamiolkowski_iso(channel, op_mx_basis=leaky_basis, choi_mx_basis='gm')
        assert np.linalg.eigvalsh(choi)[0] >= -1e-12
    vector = leaky_model.preps['rho0'].to_dense().ravel()
    rho = vec_to_stdmx(vector, leaky_basis)
    assert np.linalg.eigvalsh(rho)[0] >= -1e-12
    np.testing.assert_allclose(np.trace(rho), 1., atol=1e-12)
    for name in ('Gx', 'Gy', 'Gx', 'Gy'):
        channel = leaky_model.operations[name].to_dense()
        np.testing.assert_allclose(channel[np.ix_(excluded, allowed)], 0., atol=1e-12)
        np.testing.assert_allclose(trace_vector @ channel, trace_vector, atol=1e-12)
        choi = jamiolkowski_iso(channel, op_mx_basis=leaky_basis, choi_mx_basis='gm')
        assert np.linalg.eigvalsh(choi)[0] >= -1e-12
        vector = channel @ vector
        rho = vec_to_stdmx(vector, leaky_basis)
        np.testing.assert_allclose(rho[:2, 2], 0., atol=1e-12)
        assert np.linalg.eigvalsh(rho)[0] >= -1e-12
        np.testing.assert_allclose(np.trace(rho), 1., atol=1e-12)
    effect_sum = np.zeros((3, 3), complex)
    for effect in leaky_model.povms['Mdefault'].values():
        effect_matrix = vec_to_stdmx(effect.to_dense(), leaky_basis)
        effect_sum += effect_matrix
        np.testing.assert_allclose(effect_matrix[:2, 2], 0., atol=1e-12)
        assert np.linalg.eigvalsh(effect_matrix)[0] >= -1e-12
    np.testing.assert_allclose(effect_sum, np.eye(3), atol=1e-12)

restored_leaky_model = ExplicitOpModel.from_nice_serialization(leaky_model.to_nice_serialization())
restored_leaky_model.from_vector(rng.normal(scale=.02, size=leaky_model.num_params))
for op in restored_leaky_model.operations.values():
    np.testing.assert_allclose(op.to_dense()[np.ix_(excluded, allowed)], 0., atol=1e-12)
```

The retained five coordinates support an exact reduced simulation, since the discarded coordinates are never populated. This example deliberately keeps the full qutrit representation and uses existing model members; it does not implement a five-dimensional simulator or remove the redundant transfer parameters.

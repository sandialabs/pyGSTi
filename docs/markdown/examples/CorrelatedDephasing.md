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

# Correlated dephasing with noisy Lindblad SPAM

Consider commuting diagonal Hermitian noise operators $F_a$ and a real Kossakowski matrix $K \succeq 0$, interpreted as phase-noise covariance per unit time:

$$\mathcal L(\rho)=-\tfrac12\sum_{ab}K_{ab}[F_a,[F_b,\rho]].$$

This describes Gaussian Markovian random phases with independent time increments. Populations in the common eigenbasis are fixed. If $f(x)$ is the vector of diagonal eigenvalues on level $x$, then

$$e^{t\mathcal L}(|x\rangle\langle y|)=\exp[-t r_{xy}]|x\rangle\langle y|,\qquad
r_{xy}=\tfrac12(f(x)-f(y))^TK(f(x)-f(y)).$$

For unnormalized Pauli directions $F_i=Z_i$, this fixes all factors of two in the formulas below. Equivalently, random phases $\phi\sim\mathcal N(0,tK)$ in $U_\phi=\exp(-i\sum_i\phi_iZ_i)$ give the same channel. Other operator normalizations rescale $K$.

Spatially correlated Markovian master equations have a substantial literature; see [Jeske and Cole](https://arxiv.org/abs/1301.1381). The graph below is an explicit model hypothesis, not a claim that geometry alone makes all other Kossakowski entries zero.

## A prescribed phase-noise covariance graph

`LindbladErrorgen.from_correlated_dephasing` defaults to local diagonal PP/GM directions in subsystem order. Set `correlation_edges=()` for independent phase noise; omit it for a dense phase-noise covariance matrix; or give allowed pairs of direction labels. All graph constraints remain in place as the parameters change. The API's `initial_covariance` argument specifies this real Kossakowski matrix. Its off-diagonal entries are C-type correlation coefficients; A-type active pair terms are excluded by the real-matrix restriction.

```{code-cell} ipython3
import numpy as np
from pygsti.baseobjs import QubitSpace
from pygsti.modelmembers.operations import LindbladErrorgen, ExpErrorgenOp

directions = ('ZIII', 'IZII', 'IIZI', 'IIIZ')
edges = [(directions[i], directions[(i + 1) % 4]) for i in range(4)]
phase_covariance = np.diag([.028, .033, .025, .030])
for (i, j), value in zip(((0, 1), (1, 2), (2, 3), (0, 3)), (.007, -.006, .005, .004)):
    phase_covariance[i, j] = phase_covariance[j, i] = value
errorgen = LindbladErrorgen.from_correlated_dephasing(
    QubitSpace(4), edges, initial_covariance=phase_covariance
)
assert errorgen.num_params == 8
channel = ExpErrorgenOp(errorgen)

# Independent SPAM channels use the same graph and parameterization as the idle.
from pygsti.models import ExplicitOpModel
from pygsti.modelmembers.states import StaticState, ComposedState
from pygsti.modelmembers.povms import StaticPOVMEffect, UnconstrainedPOVM, ComposedPOVM
from pygsti.tools.basistools import stdmx_to_vec
from pygsti.baseobjs import Basis

space = QubitSpace(4)
prep_channel = ExpErrorgenOp(LindbladErrorgen.from_correlated_dephasing(
    space, edges, initial_covariance=.3 * phase_covariance
))
meas_channel = ExpErrorgenOp(LindbladErrorgen.from_correlated_dephasing(
    space, edges, initial_covariance=.2 * phase_covariance
))
reference_state = StaticState(stdmx_to_vec(np.ones((16, 16)) / 16, 'pp'),
                              basis='pp', state_space=space)
xxii = Basis.cast('PP', 256)['XXII']
reference_povm = UnconstrainedPOVM([
    (label, StaticPOVMEffect(stdmx_to_vec((np.eye(16) + sign * xxii) / 2, 'pp'),
                            basis='pp', state_space=space))
    for label, sign in (('+', 1), ('-', -1))
])
model = ExplicitOpModel(space, basis='pp')
model.preps['rho0'] = ComposedState(reference_state, prep_channel)
model.povms['Mdefault'] = ComposedPOVM(meas_channel, reference_povm, mx_basis='pp')
model.operations['Gi'] = channel
assert model.num_params == 24  # Independent 8-parameter gate, prep, and measurement maps.
assert model.probabilities(())[('+',)] < 1  # Noisy SPAM even with no idle.
```

All four directions have support weight one, and every C-type pair term has union weight two. A weight cutoff alone would allow nonzero phase-noise covariance on the two absent edges. The cycle graph instead has 8 Kossakowski parameters per channel, compared with 4 for independent noise and 10 for a dense matrix. Its induced cycle has no fill-free Cholesky chart, so the constructor uses the contraction chart, which includes zero and rank-deficient limits. Dense/chordal fill-free support also includes zero and collective rank-one limits.

For qudits the default includes each subsystem's $d-1$ diagonal Gell-Mann directions. Explicit `basis_element_labels` can select higher-body diagonal noise, for example `('ZI', 'IZ', 'ZZ')` on two qubits. A stochastic `S(ZZ)` term is different from a C-type correlation pair term `C(ZI, IZ)`. The former describes stochastic joint Pauli errors while contributing to the diagonal of $K$; the latter is an off-diagonal phase-noise covariance in this model. The local-$Z_i$ experiment below does not attempt to infer arbitrary higher-body noise.

## Jointly fitting SPAM and gate errors

Preparation is $\rho=e^{\mathcal L_p}(\rho_0)$ with $\rho_0=|+\rangle\langle+|^{\otimes4}$. Measurement probabilities are $\operatorname{Tr}[E_0e^{\mathcal L_m}e^{t\mathcal L_g}e^{\mathcal L_p}(\rho_0)]$. Thus measurement noise acts before the reference POVM, or through its adjoint on effects. All three generators have independent parameters in the same candidate hypothesis class. Their directions are the same commuting $Z_i$, so define

$$B(t)=tK_g+K_p+K_m,\qquad q_\pm(t)=2(B_{ii}(t)+B_{jj}(t)\pm2B_{ij}(t)).$$

Then

$$\begin{aligned}
\langle X_i\rangle&=e^{-2B_{ii}(t)},\\
\langle X_iX_j\rangle&=\tfrac12(e^{-q_+(t)}+e^{-q_-(t)}),\\
\langle Y_iY_j\rangle&=\tfrac12(e^{-q_-(t)}-e^{-q_+(t)}).
\end{aligned}$$

Individual Ramsey means reveal only diagonal entries. XX alone loses the sign of the off-diagonal phase-noise covariance; YY retains it and has linear sensitivity when that covariance is zero. Multiple durations distinguish the gate matrix from the SPAM intercept. Zero-duration data directly constrain that intercept. Two-qubit cross-correlation spectroscopy with other control/readout schemes is discussed by [Szańkowski et al.](https://arxiv.org/abs/1507.03897), and experimentally demonstrated by [von Lüpke et al.](https://arxiv.org/abs/1912.04982).

There is an exact ambiguity: replacing $K_p\mapsto K_p+D$ and $K_m\mapsto K_m-D$ preserves every probability whenever both matrices remain physical. We fit both channels freely, without fixing either to its generating value. Only $K_g$ and $K_p+K_m$ are compared to truth; the separate fitted SPAM matrices are representatives of this ambiguity.

At the tested interior point, the dense model's 30 coefficient directions give probability-Jacobian rank 20: 10 for gate noise and 10 for the SPAM sum. After projecting away all SPAM nuisance directions, gate sensitivity still has rank 10. Singles alone give gate rank 4. This is identifiability within the specified real dephasing family and fixed $Z_i$ frame.

One unknown pre-measurement channel is shared across the reference product-Pauli POVMs. This is an explicit hypothesis about measurement errors. There is no separate pulse-gate model: fixed reference state/POVMs define the ideal actions, and their errors are fitted through the stated Lindblad channels. Same-family dephasing SPAM does not allow arbitrary axis errors or arbitrary general CPTP SPAM.

## Reproducible model comparison

The companion script uses four qubits, five durations $(0,1,2,4,8)$, and 80 independently sampled binomial settings, each with 50,000 shots. Every candidate fits the same scenario's counts, jointly optimizing gate and both SPAM channels with analytic derivatives and binomial likelihood-deviance residuals. The 96 held-out settings use product-Pauli parities on three or four sites at durations $(0.5,1.5,3,6)$, with the fitted noisy SPAM included.

Two scenarios are generated: cycle-graph matrices on all three channels, and cycle plus a uniform common component on all three. The common rates are $0.006$ for the gate, $0.0015$ for preparation, and $0.0025$ for measurement. Candidate families are independent, cycle graph, dense, and cycle plus a common mode with fixed uniform spatial profile. Their per-channel parameter counts are 4, 8, 10, and 9; joint counts are 12, 24, 30, and 27. The last family uses $K_{\rm graph}+a^2\mathbf1\mathbf1^T$ separately on each channel; it remains an example-only chart, rather than a persistent nine-parameter model member.

```{code-cell} ipython3
from pathlib import Path
import runpy
import pygsti
import pandas as pd

study_file = Path(pygsti.__file__).resolve().parent.parent / 'docs/markdown/examples/correlated_dephasing_study.py'
study = runpy.run_path(str(study_file))
study_results = study['run_study']()
rows = []
for scenario, result in study_results['scenarios'].items():
    for family, fit in result['fits'].items():
        rows.append(dict(truth=scenario, model=family, parameters=fit['parameters'],
                         deviance_minus_truth=fit['deviance_minus_truth'],
                         heldout_mean_tvd=fit['heldout_mean_tvd'], converged=fit['converged']))
comparison = pd.DataFrame(rows)
comparison
```

`deviance_minus_truth` compares each fit to the generating model on that same dataset. Negative values reflect fitting the sampled fluctuations; a larger training improvement from a more flexible model does not establish better prediction. The script also reports held-out prediction errors, gate-matrix and SPAM-sum errors, optimizer status/evaluation counts, and the noiseless baseline. The two fixed datasets illustrate distinguishability and misspecification; they do not establish a general model-selection guarantee or finite-shot confidence interval.

The script checks the noisy-SPAM formulas against explicit pyGSTi models with composed states and POVMs, including CPTP, effect completeness, and persistence checks. It verifies analytic joint derivatives by finite differences and recovers the generating gate matrix and total SPAM matrix from expected counts. The previous ideal-SPAM experiments and their numerical results are replaced by this study.

## Two limitations worth checking explicitly

First, hard-zeroing entries of a valid Kossakowski matrix can destroy positive semidefiniteness. The script starts with $\mathbf1\mathbf1^T+0.1I$ on three directions and deletes its off-diagonal entry at $(0, 2)$ and its transpose. The resulting matrix has a negative eigenvalue despite every remaining pair satisfying its two-by-two PSD inequality. Sparse positivity is a constraint on the whole matrix.

Second, this constructor uses a real Kossakowski matrix. A more general quantum pure-dephasing generator can have a complex PSD Kossakowski matrix on diagonal Hermitian directions. Its A-type active pair terms, represented by imaginary off-diagonal Kossakowski components, generate coherence phases that need not be expressible as a diagonal Hamiltonian. A Hamiltonian produces phases $h_y-h_x$, whose sum around any cycle of basis states is zero. The companion script constructs a valid two-qubit S/C/A generator with zero population transfer and a nonzero phase-cycle sum. Those phase terms require the existing lower-level complex coefficient blocks; the real-matrix factory does not infer them.

A nonzero phase cycle is not, by itself, evidence of a quantum bath: a classical Poisson process applying diagonal unitaries can also produce one. The script includes that counterexample. The distinction here is between the stated Gaussian phase-noise covariance model with a diagonal Hamiltonian and more general dephasing processes.

A constant Markovian generator is also a modeling assumption. For colored or non-Gaussian phase noise, or noise under noncommuting control during the interval, the time dependence and effective operator basis need separate investigation. Here the fitted time dependence and selected commuting directions are fixed by the stated hypothesis.

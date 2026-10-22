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

# Correlated dephasing with covariance graphs

Consider commuting diagonal Hermitian noise operators $F_a$ and a real covariance $K\succeq0$:

$$\mathcal L(\rho)=-\tfrac12\sum_{ab}K_{ab}[F_a,[F_b,\rho]].$$

This describes Gaussian Markovian random phases with independent time increments. Populations in the common eigenbasis are fixed. If $f(x)$ is the vector of diagonal eigenvalues on level $x$, then

$$e^{t\mathcal L}(|x\rangle\langle y|)=\exp[-t r_{xy}]|x\rangle\langle y|,\qquad
r_{xy}=\tfrac12(f(x)-f(y))^TK(f(x)-f(y)).$$

For unnormalized Pauli directions $F_i=Z_i$, this fixes all factors of two in the formulas below. Equivalently, random phases $\phi\sim\mathcal N(0,tK)$ in $U_\phi=\exp(-i\sum_i\phi_iZ_i)$ give the same channel. Other operator normalizations rescale $K$.

Spatially correlated Markovian master equations have a substantial literature; see [Jeske and Cole](https://arxiv.org/abs/1301.1381). The graph below is an explicit model hypothesis, not a claim that geometry alone makes all other covariances zero.

## A prescribed covariance graph

`LindbladErrorgen.from_correlated_dephasing` defaults to local diagonal PP/GM directions in subsystem order. Set `correlation_edges=()` for independent noise; omit it for a dense covariance; or give allowed pairs of direction labels. All graph constraints remain in place as the parameters change.

```{code-cell} ipython3
import numpy as np
from pygsti.baseobjs import QubitSpace
from pygsti.modelmembers.operations import LindbladErrorgen, ExpErrorgenOp

directions = ('ZIII', 'IZII', 'IIZI', 'IIIZ')
edges = [(directions[i], directions[(i + 1) % 4]) for i in range(4)]
covariance = np.diag([.028, .033, .025, .030])
for (i, j), value in zip(((0, 1), (1, 2), (2, 3), (0, 3)), (.007, -.006, .005, .004)):
    covariance[i, j] = covariance[j, i] = value
errorgen = LindbladErrorgen.from_correlated_dephasing(
    QubitSpace(4), edges, initial_covariance=covariance
)
assert errorgen.num_params == 8
channel = ExpErrorgenOp(errorgen)
```

All four directions have support weight one, and every C term has union weight two. A weight cutoff alone would allow correlations on the two absent edges. The cycle graph instead has 8 covariance parameters, compared with 4 for independent noise and 10 for a dense covariance. Its induced cycle requires Cholesky completion. Completion covers the numerically positive-definite interior; dense/chordal fill-free support also includes zero and collective rank-one limits.

For qudits the default includes each subsystem's $d-1$ diagonal Gell-Mann directions. Explicit `basis_element_labels` can select higher-body diagonal noise, for example `('ZI', 'IZ', 'ZZ')` on two qubits. A stochastic `S(ZZ)` term is different from a covariance `C(ZI,IZ)` term. The local-$Z_i$ experiment below does not attempt to infer arbitrary higher-body noise.

## Which measurements identify the correlations?

Prepare $|+\rangle^{\otimes n}$, let dephasing act for duration $t$, and measure single-qubit or pair parities using local readout. For a pair $i,j$, define

$$r_\pm=2(K_{ii}+K_{jj}\pm2K_{ij}).$$

Then

$$\begin{aligned}
\langle X_i\rangle&=e^{-2tK_{ii}},\\
\langle X_iX_j\rangle&=\tfrac12(e^{-tr_+}+e^{-tr_-}),\\
\langle Y_iY_j\rangle&=\tfrac12(e^{-tr_-}-e^{-tr_+}).
\end{aligned}$$

Thus the individual Ramsey means reveal only the diagonal entries. XX alone loses the sign of the pair covariance. The YY signal retains its sign and has linear sensitivity at zero correlation. Product preparations suffice, but this design does require pair-parity statistics; marginal single-qubit means discard the relevant information. Two-qubit cross-correlation spectroscopy with different control/readout schemes is discussed by [Szańkowski et al.](https://arxiv.org/abs/1507.03897), and experimentally demonstrated by [von Lüpke et al.](https://arxiv.org/abs/1912.04982).

In this ideal calibrated setting,

$$e^{-tr_+}=\langle XX\rangle-\langle YY\rangle,\qquad
e^{-tr_-}=\langle XX\rangle+\langle YY\rangle,\qquad
K_{ij}=(r_+-r_-)/8.$$

Directly taking logs of noisy contrasts can be unstable or produce a non-PSD estimate. We instead fit binomial parity probabilities while retaining the PSD covariance parameterization. This is known-SPAM noise identification, not a full self-consistent GST fit with unknown preparations, measurements, and controls.

## Reproducible model comparison

The companion script uses four qubits, four durations, and 64 independently sampled binomial settings, each with 50,000 shots. All candidate models in a scenario fit the same counts, using analytic coefficient/probability derivatives and binomial likelihood-deviance residuals. The held-out predictions cover 96 time/coherence combinations involving three or four sites, absent from training.

Two scenarios are generated: the cycle covariance above, and that covariance plus $0.006\,\mathbf1\mathbf1^T$. The latter represents a shared phase source. Candidate families are independent (4 parameters), cycle graph (8), dense (10), and cycle plus a common mode of known uniform spatial profile (9). The last family uses a PSD sum $K_{\rm graph}+a^2\mathbf1\mathbf1^T$ in the example fit; it is not a new persistent nine-parameter model-member class. A freely fitted spatial profile would introduce additional parameters and identifiability questions.

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

`deviance_minus_truth` compares each fit to the generating model on that same dataset. Negative values reflect fitting the sampled fluctuations; a larger training improvement from a more flexible model does not establish better prediction. The script also reports held-out prediction errors, covariance errors, optimizer status/evaluation counts, and the noiseless baseline. The two fixed datasets illustrate distinguishability and misspecification; they do not establish a general model-selection guarantee or finite-shot confidence interval.

The script checks the parity formulas against complete pyGSTi channels, verifies analytic derivatives by finite differences, and recovers the generating graph covariance from expected counts. With individual means the covariance design has rank 4; with all the product parities it has rank 10. Correlation measurements on absent edges are particularly important for testing the graph hypothesis against a common source.

## Two limitations worth checking explicitly

First, hard-zeroing entries of a valid covariance can destroy positive semidefiniteness. The script starts with $\mathbf1\mathbf1^T+0.1I$ on three directions and deletes its $(0,2)$ correlation. The resulting matrix has a negative eigenvalue despite every remaining pair satisfying its two-by-two PSD inequality. Sparse positivity is a constraint on the whole covariance.

Second, this constructor uses a real covariance. A more general quantum pure-dephasing generator can have a complex PSD Kossakowski matrix on diagonal Hermitian directions. Its imaginary correlations generate coherence phases that need not be expressible as a diagonal Hamiltonian. A Hamiltonian produces phases $h_y-h_x$, whose sum around any cycle of basis states is zero. The companion script constructs a valid two-qubit S/C/A generator with zero population transfer and a nonzero phase-cycle sum. Those phase terms require the existing lower-level complex coefficient blocks; the real-covariance factory does not infer them.

A nonzero phase cycle is not, by itself, evidence of a quantum bath: a classical Poisson process applying diagonal unitaries can also produce one. The script includes that counterexample. The distinction here is between the stated Gaussian covariance-plus-Hamiltonian model and more general dephasing processes.

A constant Markovian generator is also a modeling assumption. For colored or non-Gaussian phase noise, or noise under noncommuting control during the interval, the time dependence and effective operator basis need separate investigation. Here the fitted time dependence and selected commuting directions are fixed by the stated hypothesis.

---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.19.5
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Linearized Gate Set Tomography III: Optimal Experiments for a Fixed Shot Budget

**Linearized GST** ([Miller et al., arXiv:2605.11158](https://arxiv.org/abs/2605.11158)) learns the rates of a sparse error model -- elementary error generators attached to the gates, the state preparation and measurement, and the mid-circuit measurements (MCMs) of a processor -- from shallow Clifford circuits, by solving the linear system $\overrightarrow{\Delta\langle Q\rangle} = D\vec\epsilon$ that relates the error rates to the shifts of Pauli expectation values.  The [first tutorial](LinearizedGST) estimated such a model from random circuits and thinned them out with `select_circuits`; the [second](LinearizedGST_2) constructed circuits that probe every parameter on purpose, and ended with the observation that at a fixed *total* number of shots it pays to give different circuits different numbers of shots.

This tutorial takes that last idea to its conclusion.  The question is the one an experimentalist actually faces: **I can afford $S$ shots in total.  Which circuits should I run, with how many shots each, to learn the error model as precisely as possible?**  Answering it properly requires three things that the first two tutorials did not have:

1. the *actual* shot-noise statistics of a circuit's data -- the earlier tutorials' predictions used the worst-case variance for every observable whose ideal value is $\pm1$ (precisely the ones that carry the information about stochastic errors) and ignored their correlations with the others, and the second tutorial flagged this as "a natural but unresolved extension";
2. a *figure of merit* for an experiment that follows from those statistics and says what "as precisely as possible" means for a vector of 57 (or 93) rates of different kinds; and
3. an *estimator* that actually attains the precision the design promises -- the pseudo-inverse / non-negative least squares fit of the earlier tutorials weights every observable of every circuit equally, which is wrong as soon as shot counts differ.

With these in place, the design problem is a convex optimization problem over the shot counts of a large pool of candidate circuits -- the "approximate design" problem of classical optimal experimental design (Kiefer & Wolfowitz 1960; Fedorov, *Theory of Optimal Experiments*, 1972; Pukelsheim, *Optimal Design of Experiments*, 1993), applied to quantum tomography by Kosut, Walmsley and Rabitz ([quant-ph/0411093](https://arxiv.org/abs/quant-ph/0411093)) and, closest to what we do here, to Pauli-noise learning by Hockings, Doherty and Harper's optimized ACES experiments ([PRX Quantum 6, 010334 (2025)](https://journals.aps.org/prxquantum/abstract/10.1103/PRXQuantum.6.010334)).  pyGSTi's own germ selection for standard GST is an instance of the same idea (it scores germ sets by the sum of the inverse eigenvalues of a Jacobian Gram matrix).  Section 3 defines the figure of merit and Section 9 summarizes it in words; the sections in between build the design, validate its predictions by simulation, test how much it depends on what we assume about the device, show how to target it at an application, and run everything for the fully general H+S+C+A error model with mid-circuit measurements and SPAM errors.

```{code-cell} ipython3
import os
# Experiment design is thousands of small dense linear-algebra operations, for which multi-threaded BLAS is
# counterproductive (in some environments catastrophically so); limit the thread count before numpy is imported.
for var in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(var, '4')

import itertools
import time
import warnings
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import stim

import pygsti
from pygsti.baseobjs import Label
from pygsti.circuits import Circuit
from pygsti.processors import QubitProcessorSpec
from pygsti.algorithms import linearizedgst as lgst
from pygsti.errorgenpropagation import mcmgadget as mg

# Models containing instruments must be simulated with the matrix forward simulator, which
# warns that random (non-periodic) circuits make its evaluation tree inefficient. Harmless here.
warnings.filterwarnings('ignore', message='Created an evaluation tree that is inefficient')
pd.set_option('display.width', 160)
```

## 1. Setup: the processor, the error model and the candidate circuits

The processor and the H+S error model are exactly those of the first two tutorials (same random seed, same rates): three qubits with $X_{\pi/2}$, $Y_{\pi/2}$, $Z_{\pi/2}$ gates, CZ gates between neighbours, a mid-circuit measurement `Iz` on the middle qubit described by the 13 first-order MCM-gauge-invariant (FOMGI) quantities of [Wysocki et al.](https://arxiv.org/abs/2602.03938), coherent $Z$ crosstalk from every single-qubit gate onto its neighbours, and bit-flip SPAM errors.  See the first tutorial for the meaning of every term.

```{code-cell} ipython3
num_qubits = 3
edges = [(0, 1), (1, 2)]
pspec = QubitProcessorSpec(num_qubits, ['Gxpi2', 'Gypi2', 'Gcphase', 'Gzpi2'], availability={'Gcphase': edges})
mcm_qubit = 1

rng = np.random.default_rng(2026)

def coherent():   # Hamiltonian rates in [-0.01, 0.01]
    return rng.uniform(-0.01, 0.01)

def stochastic(): # Pauli-stochastic rates in [0, 0.002]
    return rng.uniform(0, 0.002)

def neighbours(q):
    return [q2 for (a, b) in edges for q2 in ((b,) if a == q else (a,) if b == q else ())]

error_model = {}
for q in pspec.qubit_labels:
    error_model[('Gxpi2', q)] = {('H', 'X'): coherent(), ('S', 'X'): stochastic(),
                                 **{('H', 'Z:%d' % q2): coherent() for q2 in neighbours(q)}}
    error_model[('Gypi2', q)] = {('H', 'Y'): coherent(), ('S', 'Y'): stochastic(),
                                 **{('H', 'Z:%d' % q2): coherent() for q2 in neighbours(q)}}
    error_model[('Gzpi2', q)] = {('H', 'Z'): coherent(), ('S', 'Z'): stochastic(),
                                 **{('H', 'Z:%d' % q2): coherent() for q2 in neighbours(q)}}
for (q1, q2) in edges:
    error_model[('Gcphase', q1, q2)] = {('H', 'ZZ'): coherent(), ('S', 'ZZ'): stochastic(),
                                        ('H', 'ZI'): coherent(), ('H', 'IZ'): coherent()}

fomgi_true = {'s_read': 0.01, 's_meas': 0.001, 's_prep': 0.002,
              'r_meas_y': 0.005, 'r_prep_x_ind': 0.01, 'w0': -0.005}
error_model[('Iz', mcm_qubit)] = mg.fomgi_ansatz(rates=fomgi_true)

error_model['prep'] = {('S', 'X:%d' % q): stochastic() for q in pspec.qubit_labels}
error_model['povm'] = {('S', 'X:%d' % q): 2 * stochastic() for q in pspec.qubit_labels}

params, true_rates = lgst.build_model_parameter_indexing(error_model, qubit_labels=pspec.qubit_labels)
true_rates = np.array(true_rates)
types = np.array(params.errorgen_types())
P = len(params)
print(P, "parameters:", {t: int((types == t).sum()) for t in 'HS'})
```

### The candidate pool

An optimal design can only be as good as the pool of circuits it chooses from.  We assemble a large and varied pool from the two families of the earlier tutorials, plus one that anticipates Section 7:

* **300 random circuits** of depth 6 with 0-2 mid-circuit measurements (the baseline of the first tutorial) -- the circuits with two MCMs have 15 observables each and are the most information-dense per shot;
* the **structured probe family** of the second tutorial: for every parameter, every combination of Pauli-eigenstate preparations and measurement bases on the qubits it involves, around a single application of its gate;
* **parity-check circuits**: the middle qubit is used as an ancilla to measure the $ZZ$ parity of the outer qubits (two CZ gates between $Y_{\pi/2}$ rotations, followed by the mid-circuit measurement), once or twice, with the data qubits prepared and measured in various bases -- a miniature syndrome-extraction circuit, which is both a natural probe of the MCM in context and the "application" of Section 7.

```{code-cell} ipython3
def sample_circuit(rng, depth, num_mcms):
    layers = []
    for _ in range(depth):
        active = [q for q in pspec.qubit_labels if rng.random() < 0.75]
        components = []
        if rng.random() < 0.5:
            edge = edges[rng.integers(len(edges))]
            if all(q in active for q in edge):
                components.append(Label('Gcphase', edge))
                active = [q for q in active if q not in edge]
        components += [Label(str(rng.choice(['Gxpi2', 'Gzpi2', 'Gypi2'])), q) for q in active]
        if components:
            layers.append(components[0] if len(components) == 1 else Label(components))
    for _ in range(num_mcms):
        layers.insert(rng.integers(0, len(layers) + 1), Label('Iz', mcm_qubit))
    return Circuit(layers, line_labels=pspec.qubit_labels)

random_circuits = [sample_circuit(rng, 6, rng.integers(0, 3)) for _ in range(300)]

# single-qubit gate sequences that prepare the six Pauli eigenstates from |0>, and that rotate X, Y or Z onto Z
PREPS = {'+Z': (), '-Z': ('Gxpi2', 'Gxpi2'), '+X': ('Gypi2',), '-X': ('Gxpi2', 'Gxpi2', 'Gypi2'),
         '-Y': ('Gxpi2',), '+Y': ('Gypi2', 'Gzpi2')}
MEAS = {'Z': (), 'X': ('Gypi2',), 'Y': ('Gxpi2',)}

def parallel_layers(sequences):
    layers = []
    for k in range(max((len(s) for s in sequences.values()), default=0)):
        components = [Label(s[k], q) for q, s in sequences.items() if k < len(s)]
        layers.append(components[0] if len(components) == 1 else Label(components))
    return layers

def sandwich(prep, middle, meas):
    '''prep: {qubit: '+X', ...}; middle: list of layers; meas: {qubit: 'Y', ...}.'''
    layers = parallel_layers({q: PREPS[a] for q, a in prep.items()}) + list(middle) \
        + parallel_layers({q: MEAS[a] for q, a in meas.items()})
    return Circuit(layers, line_labels=pspec.qubit_labels)

def error_support(p):
    qubits = set()
    for pauli in p.errorgen.basis_element_labels:
        s = str(pauli).lstrip('+-')[:params.num_qubits]
        qubits |= {params.qubit_labels[i] for i, ch in enumerate(s) if ch not in '_I'}
    return qubits

def gate_label(key):
    return Label(key[0], tuple(key[1:]))

def probe_family(p, keys):
    '''(target layer, qubits to prepare/measure) pairs for parameter p, as in the second tutorial.'''
    if not p.is_spam:
        target = gate_label(p.key)
        return [(target, sorted(error_support(p) | set(target.qubits)))]
    q = next(iter(error_support(p)))
    family = [(None, [q])]
    for key in keys:
        if isinstance(key, str):
            continue
        target = gate_label(key)
        if q in target.qubits or any((q, q2) in edges or (q2, q) in edges for q2 in target.qubits):
            family.append((target, sorted({q} | set(target.qubits))))
    return family

def structured_circuits(params):
    seen = {}
    for p in params:
        for target, support in probe_family(p, params.keys):
            for prep in itertools.product(PREPS, repeat=len(support)):
                for meas in itertools.product(MEAS, repeat=len(support)):
                    c = sandwich(dict(zip(support, prep)), [] if target is None else [target], dict(zip(support, meas)))
                    if c.depth > 0 and c.str not in seen:
                        seen[c.str] = c
    return list(seen.values())

structured = structured_circuits(params)

def parity_round():
    return [Label('Gypi2', mcm_qubit), Label('Gcphase', (0, 1)), Label('Gcphase', (1, 2)),
            Label('Gypi2', mcm_qubit), Label('Iz', mcm_qubit)]

parity_circuits = [sandwich(dict(zip((0, 2), prep)), parity_round(), dict(zip((0, 2), meas)))
                   for prep in itertools.product(PREPS, repeat=2) for meas in itertools.product(MEAS, repeat=2)]
parity_circuits += [sandwich(dict(zip((0, 2), prep)), parity_round() * 2, {}) for prep in itertools.product(PREPS, repeat=2)]

print("random: %d, structured: %d, parity-check: %d circuits" % (len(random_circuits), len(structured), len(parity_circuits)))
```

The structured family is large because it enumerates all sign choices of the preparations.  Many of its members have design matrices that coincide up to the signs of rows and ideal expectation values and the order of rows; to first order such circuits carry the same information about the rates, so we keep only the shallowest representative of each class.  (This is a convenience, not a necessity: the optimization below handles pools of this size.)

```{code-cell} ipython3
def design_key(d):
    rows = sorted(tuple(np.round(np.abs(r), 6)) + (abs(float(e)),)
                  for r, e in zip(d.design_matrix, d.ideal_expectations))
    return (d.num_mcms,) + tuple(rows)

structured_designs = lgst.create_design_matrix_list(structured, params, return_info=True)
representatives = {}
for d in sorted(structured_designs, key=lambda d: d.circuit.depth):
    representatives.setdefault(design_key(d), d)
structured_designs = list(representatives.values())

random_designs = lgst.create_design_matrix_list(random_circuits, params, return_info=True)
parity_designs = lgst.create_design_matrix_list(parity_circuits, params, return_info=True)

pool_designs = structured_designs + random_designs + parity_designs
pool_circuits = [d.circuit for d in pool_designs]
origin = np.array(['structured'] * len(structured_designs) + ['random'] * len(random_designs) + ['parity'] * len(parity_designs))
depths = np.array([c.depth for c in pool_circuits])
num_mcms = np.array([d.num_mcms for d in pool_designs])

print("candidate pool: %d circuits (%d structured after removing equivalent ones, %d random, %d parity-check)"
      % (len(pool_designs), len(structured_designs), len(random_designs), len(parity_designs)))
print("unidentifiable parameter combinations in the pool:", lgst.unidentifiable_directions(pool_designs, params))
```

A few helpers for printing circuits:

```{code-cell} ipython3
STIM_GATE_NAMES = {'Gxpi2': 'SQRT_X', 'Gypi2': 'SQRT_Y', 'Gcphase': 'CZ', 'Gcnot': 'CX', 'Iz': 'M', 'Gzpi2': 'SQRT_Z'}

def to_stim_circuit(circuit):
    sc = stim.Circuit()
    for i in range(circuit.depth):
        layer = circuit.layer_label(i)
        for component in (layer.components if hasattr(layer, 'components') else [layer]):
            sc.append(STIM_GATE_NAMES[component.name], component.qubits)
        sc.append('TICK')
    return sc

def describe(circuit):
    return ' | '.join(str(l) for l in circuit.layertup)

def by_sector(error_bars, sectors='HS'):
    return {**{'%s mean' % t: error_bars[types == t].mean() for t in sectors},
            **{'%s max' % t: error_bars[types == t].max() for t in sectors}}
```

## 2. What a circuit's data actually tell us

Every circuit is run $N_c$ times and produces a bitstring per shot (data qubits and MCM records).  Linearized GST uses the sample means $\hat m_a$ of the Z-type observables $Q_a$ of weight $\le 2$ -- functions of one and the same multinomial sample, so their covariance is fixed by the outcome distribution:

$$\mathrm{Cov}(\hat m_a, \hat m_b) = \frac{1}{N_c}\Big(\langle Q_a Q_b\rangle - \langle Q_a\rangle\langle Q_b\rangle\Big).$$

The product $Q_aQ_b$ of two Z-type Paulis is another Z-type Pauli, so *both* ingredients are available from the same machinery that builds the design matrix: ideal expectation values from the stabilizer tableau and first-order shifts from design rows -- for the observables themselves (the rows of $D_c$) and for their pairwise products.  `create_design_moments` computes these second-moment rows for every circuit, and `CircuitMoments.covariance` evaluates the per-shot covariance $\Sigma_c$ at any set of *nominal* rates.  (This is the same bookkeeping as Eq. (21)-(23) of the ACES paper, where the covariance of simultaneously estimated circuit eigenvalues is $\Lambda_{a+a'} - \Lambda_a\Lambda_{a'}$.)

```{code-cell} ipython3
t0 = time.time()
moments = lgst.create_design_moments(pool_designs, params)
print("first and second moments of %d circuits computed in %.0f s" % (len(moments), time.time() - t0))
```

Let us look at the example circuit of the earlier tutorials -- a random circuit with one MCM -- and compare three covariances of its ten observables: the worst case used by `predicted_error_bars` so far (variance $1$ for every observable; correlations only between observables with ideal value 0), the first-order nominal covariance evaluated at the true rates, and the exact covariance computed from the exact outcome distribution of the noisy model.

```{code-cell} ipython3
example = next(d for d in random_designs if d.num_mcms == 1 and 2 <= mg.mcm_infos(d.circuit)[0].layer_index <= 4)
example_moments = moments[next(i for i, d in enumerate(pool_designs) if d is example)]
print(to_stim_circuit(example.circuit).diagram('timeline-text'))

noisy_model = lgst.rate_to_model(pspec, error_model)
probs = noisy_model.probabilities(example.circuit)
paulis = list(example.paulis)
k = len(paulis)
m_exact = lgst.compute_pauli_expectations(probs, paulis)
M_exact = np.eye(k)
for a in range(k):
    for b in range(a + 1, k):
        M_exact[a, b] = M_exact[b, a] = lgst.compute_pauli_expectations(probs, [paulis[a] * paulis[b]])[0]
Sigma_exact = M_exact - np.outer(m_exact, m_exact)
Sigma_nominal = example_moments.covariance(true_rates, min_variance=0.0)
Sigma_worst = lgst._ideal_observable_covariance(example)

pd.DataFrame({'ideal <Q>': example.ideal_expectations, 'variance: worst case': np.diag(Sigma_worst),
              'variance: nominal, 1st order': np.diag(Sigma_nominal), 'variance: exact': np.diag(Sigma_exact)},
             index=[str(p) for p in paulis]).round(4)
```

```{code-cell} ipython3
print("largest deviation from the exact covariance:  nominal %.1e,  worst case %.2f"
      % (np.abs(Sigma_nominal - Sigma_exact).max(), np.abs(Sigma_worst - Sigma_exact).max()))
print("eigenvalues of the nominal covariance:", np.round(np.linalg.eigvalsh(Sigma_nominal), 4))

fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), constrained_layout=True)
for ax, (S, title) in zip(axes, [(Sigma_worst, 'worst case (tutorials I-II)'), (Sigma_nominal, 'nominal model, first order')]):
    im = ax.imshow(S, cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_xticks(range(k)); ax.set_yticks(range(k))
    ax.set_xticklabels([str(p) for p in paulis], rotation=90); ax.set_yticklabels([str(p) for p in paulis])
    ax.set_title(title)
fig.suptitle('per-shot covariance of the observable estimates of the example circuit')
fig.colorbar(im, ax=axes, shrink=0.8)
plt.show()
```

Two things are wrong with the worst case, and both matter for the design.  The observables with ideal value $\pm1$ -- the ones that carry the information about the *stochastic* rates -- do not have unit variance: an observable that should always read $+1$ fluctuates only because errors flip it, so its variance is $1 - \langle Q\rangle^2 \approx 2\,|D_Q\,\vec\epsilon|$, of the order of a few percent here.  Treating it as 1 overestimates its variance by a factor of 14-30 and, in a design, undervalues the circuits that measure such observables by the same factor.  Second, the first-order nominal covariance is accurate to about $10^{-3}$, with the correct correlations between all observables.

Its eigenvalues range from 3 down to $3\times 10^{-3}$, and in some circuits down to exactly zero: a direction of zero variance is a combination of observables that no error generator of the ansatz can move, and it carries no information either.  Since the covariance has to be inverted, we impose a floor on its eigenvalues (`min_variance`, default $10^{-3}$).  Its meaning is operational: it is the smallest variance -- equivalently, half of it is the smallest flip probability -- that we allow the design to *count on*.  Section 6 checks that the design is insensitive to its value.

## 3. Fisher information and the figure of merit

With $\Sigma_c$ in hand, each circuit's data have a well-defined information content.  The observed shifts $y_c = \hat m_c - m_c^{\rm ideal}$ have mean $D_c\vec\epsilon$ and covariance $\Sigma_c/N_c$, so the Fisher information they carry about the rates is

$$N_c\,F_c, \qquad F_c = D_c^{\mathsf T}\,\Sigma_c^{-1}\,D_c,$$

and the information of a whole experiment is additive, $\mathcal I(\vec N) = \sum_c N_c F_c$.  Its inverse is the Cramér-Rao bound on the covariance of *any* unbiased estimate of the rates, and it is attained by the generalized least-squares (GLS) estimator of Section 5.  `circuit_information` computes $F_c$ for every candidate (in a factored form, so that memory stays linear in the number of observables), given the nominal rates at which to evaluate the statistics.  We use a deliberately crude nominal model -- "every stochastic rate is $10^{-3}$, every coherent rate is $0$" -- which is the kind of prior knowledge one has about a device before characterizing it.  The true rates are kept aside for validation.

```{code-cell} ipython3
budget = 3_000_000     # total number of shots, as in the second tutorial
nominal = lgst.nominal_rates_by_type(params, {'H': 0.0, 'S': 1e-3})

info_nominal = lgst.circuit_information(moments, nominal)        # for designing (what we assume about the device)
info_true = lgst.circuit_information(moments, true_rates)        # for validating (what the device actually does)
```

`predicted_covariance` turns an allocation of shots into the covariance of the estimate -- for the GLS estimator (the inverse information) or, with `estimator='ols'`, for the plain pseudo-inverse estimator of the earlier tutorials under the same statistics.  Here are the predicted error bars of the baseline design, 300 random circuits with 10,000 shots each, in three accountings:

```{code-cell} ipython3
is_random = origin == 'random'
random_index = np.nonzero(is_random)[0]
shots_random = np.zeros(len(pool_designs))
shots_random[random_index] = budget / len(random_index)
random_moments = [moments[i] for i in random_index]

eb_worst = lgst.predicted_error_bars(random_designs, shots=budget / len(random_index))
eb_ols = lgst.predicted_error_bars_nominal(random_moments, shots_random[random_index], nominal, estimator='ols')
eb_gls = lgst.predicted_error_bars_nominal(info_nominal, shots_random)

pd.DataFrame([{'statistics, estimator': name, **by_sector(eb)}
              for name, eb in [('worst case, pseudo-inverse (tutorials I-II)', eb_worst),
                               ('nominal model, pseudo-inverse', eb_ols),
                               ('nominal model, GLS', eb_gls)]]).set_index('statistics, estimator').style.format('{:.2e}')
```

The Hamiltonian error bars barely move (they come from observables with ideal value 0, whose variance really is 1), but the stochastic ones were overestimated by a factor of four, and the GLS estimator -- which weights every observable by the inverse of its variance and knows about the correlations -- gains another 10-25% on top.

### The figure of merit

An experiment design is a vector of shot counts $\vec N$ over the candidate pool with $\sum_c N_c = S$.  We score it by

$$\boxed{\;\Phi_W(\vec N) \;=\; \mathrm{tr}\!\Big[W\,\mathcal I(\vec N)^{-1}\Big] \;=\; \mathbb E\Big[(\hat{\vec\epsilon}-\vec\epsilon)^{\mathsf T}\,W\,(\hat{\vec\epsilon}-\vec\epsilon)\Big],\qquad \mathcal I(\vec N)=\sum_c N_c\,D_c^{\mathsf T}\Sigma_c^{-1}D_c\;}$$

-- the expected squared error of the rate estimate, measured in the metric $W$, under the nominal model's shot-noise statistics.  This is the **A-criterion** of optimal design (A for "average variance"), and the design we construct minimizes it.  Its ingredients deserve spelling out:

* **What it measures.**  With $W = I$, $\Phi_I$ is the sum of the variances of all rates, and $\sqrt{\Phi_I/P}$ is the expected root-mean-square error of a rate.  Up to normalization this is exactly the ACES figure of merit of Hockings et al. ("expected normalized RMS error", their Eq. 28 $\approx\sqrt{\mathrm{tr}\,\Sigma}$): they, too, evaluate the GLS covariance of their estimator under a nominal noise model and minimize its trace over the shot weights of a candidate set of circuits.  Comparing a coherent rate with a stochastic one in the same units is legitimate here because every elementary error-generator rate in this parametrization has the same meaning -- the first-order shift of an expectation value per use of the operation (sensitivity $\pm 2$ per occurrence).
* **What $W$ is for.**  $W$ says what the rates are *for*.  A diagonal $W$ of inverse precision targets turns $\Phi_W$ into "how far am I from the precision I need", summed over parameters; a projector onto the FOMGI columns designs an experiment for the MCM alone; and the *prediction weight* $W = \overline{D_a^{\mathsf T}D_a}$ of Section 7 makes $\Phi_W$ the mean squared error of the model's predictions for a set of application circuits -- a choice that is invariant under reparametrization of the rates and automatically counts each parameter as often as the application exercises it.
* **What it does not contain.**  $\Phi_W$ is shot noise only.  The systematic error of the linearization is controlled by the depth of the circuits in the pool, not by the allocation; we check it separately in Section 5.
* **Alternatives.**  `optimal_allocation` also offers the D-criterion ($-\log\det\mathcal I$, the volume of the confidence ellipsoid, invariant under any linear reparametrization) and the E-criterion (the variance of the worst-determined parameter combination).  Both are useful cross-checks; neither reports in units an experiment can be planned in, which is why the A-criterion with an explicit $W$ is our figure of merit.

Minimizing $\Phi_W$ over $\vec N \ge 0$, $\sum_c N_c = S$ is a convex problem (the inverse of a positive-definite matrix is matrix-convex), so its optimum is global and *certifiable*: by the Kiefer-Wolfowitz equivalence theorem, no allocation can beat a candidate design $\vec N$ by more than $\max_c \mathrm{tr}(\mathcal I^{-1} W \mathcal I^{-1} F_c)\,S - \Phi_W(\vec N)$.  `optimal_allocation` solves it with a multiplicative-weight algorithm (Titterington 1976; monotone convergence proved by Yu, *Ann. Statist.* 38, 1593 (2010)) that needs no external solver and stops when this certified gap is below a relative tolerance; `method='cvxpy'` solves the same problem as a semidefinite program and agrees to the digits shown.  Candidates the optimum does not want simply receive zero shots, so *which circuits* and *how many shots* are answered together.

## 4. The optimal experiment

```{code-cell} ipython3
t0 = time.time()
shots_continuous, result_continuous = lgst.optimal_allocation(info_nominal, budget, criterion='A')
print("solved in %.0f s: %d of %d candidates used, certified within %.1e of the optimum over the pool"
      % (time.time() - t0, result_continuous.support_size, len(pool_designs), result_continuous.gap))
print("Phi = %.3e  ->  expected RMS error of a rate %.2e" % (result_continuous.objective, np.sqrt(result_continuous.objective / P)))
print("shots per used circuit: min %d, median %d, max %d" % (shots_continuous[shots_continuous > 0].min(),
      np.median(shots_continuous[shots_continuous > 0]), shots_continuous.max()))
```

The continuous optimum spreads the budget very unevenly, and keeps a tail of circuits with only a handful of shots -- an artefact of the continuous relaxation (the support of an A-optimal design can be as large as $P(P+1)/2$) that is pointless experimentally.  `min_shots` drops candidates below a floor and re-solves; the price is invisible:

```{code-cell} ipython3
shots_opt, result = lgst.optimal_allocation(info_nominal, budget, criterion='A', min_shots=5000)
shots_opt = lgst.round_shots(shots_opt, budget).astype(float)        # integers that still sum to the budget
used = shots_opt > 0
phi_opt = lgst.design_criterion(info_nominal, shots_opt)
print("%d circuits with at least 5000 shots; Phi = %.3e (%.2f%% above the unconstrained optimum); certified gap %.1e"
      % (used.sum(), phi_opt, 100 * (phi_opt / result_continuous.objective - 1), result.gap))
```

### Comparison with the earlier designs

How much better is this than what the first two tutorials did?  We compare, at the same budget of $3\times 10^6$ shots and with the same nominal statistics and the GLS estimator throughout (so that only the *design* differs):

* **(a)** the 300 random circuits with equal shots (tutorial I);
* **(b)** 100 structured circuits chosen greedily by the A-criterion on $D^{\mathsf T}D$ with equal shots (tutorial II's main design);
* **(c)** tutorial II's closing idea -- a convex allocation over the whole pool, but with the worst-case information $D^{\mathsf T}D$ (every observable independent with unit variance);
* **(d)** the allocation of this section.

```{code-cell} ipython3
selected_100 = lgst.select_circuits(structured_designs, num_circuits=100)
shots_b = np.zeros(len(pool_designs))
shots_b[selected_100] = budget / 100

info_unit = lgst.DesignInformation([d.design_matrix for d in pool_designs])   # F_c = D_c^T D_c, as in tutorial II
shots_c, result_c = lgst.optimal_allocation(info_unit, budget, criterion='A', min_shots=5000)
shots_c = lgst.round_shots(shots_c, budget).astype(float)

def summarize(name, shots, info=info_nominal):
    used = shots > 0
    cov = lgst.predicted_covariance(info, shots)
    eb = np.sqrt(np.diag(cov))
    return {'design': name, 'circuits': int(used.sum()),
            'shots per circuit (min / median / max)': '%d / %d / %d' % (shots[used].min(), np.median(shots[used]), shots[used].max()),
            'shots on circuits with 0 / 1 / 2 MCMs (%)': ' / '.join('%.0f' % (100 * shots[used & (num_mcms == m)].sum() / budget) for m in range(3)),
            'mean depth (shot-weighted)': float(np.sum(shots * depths) / budget),
            'Phi': float(np.trace(cov)), 'RMS error sqrt(Phi/P)': float(np.sqrt(np.trace(cov) / P)), **by_sector(eb)}

designs = {'(a) 300 random circuits, equal shots': shots_random,
           '(b) 100 structured circuits selected by DᵀD, equal shots': shots_b,
           '(c) allocation with the DᵀD information': shots_c,
           '(d) allocation with the nominal-model information': shots_opt}
comparison = pd.DataFrame([summarize(name, shots) for name, shots in designs.items()]).set_index('design').T
comparison.map(lambda v: ('%.2e' % v if v < 0.1 else '%.2f' % v) if isinstance(v, float) else v)
```

Each step helps, and the steps are of similar size: concentrating equal shots on 100 well-chosen circuits (b), letting the shot counts vary (c), and finally using the true information content of the circuits (d).  Altogether the expected RMS error of a rate halves relative to the random baseline -- the same precision for a quarter of the shots -- and the worst-determined Hamiltonian rate improves threefold.  Notice *where* the gain of (d) over (c) comes from: the stochastic error bars actually grow back to the level of the random design.  Observables with ideal value $\pm1$ are so quiet that the stochastic rates are cheap in any design, and their variances contribute almost nothing to $\Phi_I$; it is the coherent rates, which must be extracted from observables with unit variance, that eat the budget.  (d) knows this and moves shots from stochastic-probing to coherent-probing circuits, whereas (c), believing every observable equally noisy, over-invests in the former.  If that trade-off is not the one wanted, $W$ is the place to say so (Section 7).

What does the optimal experiment look like?

```{code-cell} ipython3
print("design (d) by origin of the circuits:")
for o in ('structured', 'random', 'parity'):
    sel = used & (origin == o)
    print("   %-11s %3d circuits, %5.1f%% of the shots, mean depth %.1f" % (o, sel.sum(), 100 * shots_opt[sel].sum() / budget, depths[sel].mean() if sel.any() else 0))

from matplotlib.patches import Patch
order = np.argsort(-shots_opt)
fig, ax = plt.subplots(figsize=(11, 3.5))
colors = {'structured': 'tab:blue', 'random': 'tab:orange', 'parity': 'tab:green'}
ax.bar(range(used.sum()), shots_opt[order][:used.sum()], color=[colors[o] for o in origin[order][:used.sum()]], width=1.0)
ax.legend(handles=[Patch(color=c, label=o) for o, c in colors.items()])
ax.set_xlabel('circuit (sorted by allocation)'); ax.set_ylabel('shots'); ax.set_title('the optimal allocation')
plt.tight_layout(); plt.show()

print("\nthe five most-used circuits:")
for i in order[:5]:
    print("%7d shots   %-10s depth %d, %d MCM:  %s" % (shots_opt[i], origin[i], depths[i], num_mcms[i], describe(pool_circuits[i])))
```

The design is a hybrid, as the second tutorial anticipated: random circuits with one or two mid-circuit measurements -- many observables, many parameters per shot -- take the bulk of the budget in large allocations, while structured probes fill in the parameters they leave poorly determined with smaller ones.  The parity-check circuits are used only sparingly: they are superb at a few parameters but, with a single MCM and few gates, not information-dense.

### Two practical variations

*Costs.*  A mid-circuit measurement takes far longer than a gate, so a fixed amount of *time* buys fewer shots of MCM-heavy circuits.  `circuit_information` accepts a per-shot `cost`, and the budget is then spent in those units:

```{code-cell} ipython3
cost = 1.0 + 4.0 * num_mcms          # a shot of a circuit with m MCMs costs 1 + 4m units of time
info_cost = lgst.circuit_information(moments, nominal, cost=cost)
shots_time, _ = lgst.optimal_allocation(info_cost, budget, criterion='A', min_shots=5000)
print("fraction of shots on MCM circuits: %.0f%% when shots are the budget, %.0f%% when time is"
      % (100 * shots_opt[num_mcms > 0].sum() / shots_opt.sum(), 100 * shots_time[num_mcms > 0].sum() / shots_time.sum()))
```

*Other criteria.*  The D- and E-optimal allocations for the same pool (`criterion='D'`, `'E'`) are computed the same way; Section 6 uses the D-criterion as a check that the A-optimal design is not fragile.

## 5. The matched estimator, and does the design deliver?

The covariance the design optimizes is that of **generalized least squares**: every circuit's observed shifts $y_c$ are weighted by $N_c\,\Sigma_c^{-1}$ and the weighted problem

$$\min_{\vec\epsilon}\ \sum_c N_c\,(y_c - D_c\vec\epsilon)^{\mathsf T}\Sigma_c^{-1}(y_c - D_c\vec\epsilon),\qquad s_P \ge 0,$$

is solved with the stochastic rates constrained to be non-negative.  Because $\Sigma_c$ depends on the rates, `estimate_error_rates_gls` iterates: the covariance is evaluated at the nominal rates first, then at the current estimate (feasible GLS, as in the ACES paper's weighted least squares).  Its error bars are the square roots of the diagonal of the inverse information at the final estimate (or a bootstrap).  Unlike `estimate_error_rates`, it is correct for unequal shot counts -- and, as the table above showed, better even for equal ones.

To check that the predictions hold, we simulate the full experiment many times: exact outcome probabilities of every circuit from the true noisy model, multinomial sampling with the design's shot counts, estimation, and the error against the truth.  We do this for the random baseline with both estimators and for designs (b)-(d) with GLS.

```{code-cell} ipython3
needed = sorted(set(np.nonzero(sum(designs.values()) > 0)[0]))
t0 = time.time()
exact_probs = {i: noisy_model.probabilities(pool_circuits[i]) for i in needed}
print("exact outcome probabilities of %d circuits in %.0f s" % (len(needed), time.time() - t0))

mc_rng = np.random.default_rng(1)

def monte_carlo(shots, estimator='gls', trials=40):
    '''Estimation errors (trials x P) of repeated simulated experiments with the given shot allocation.'''
    used = np.nonzero(shots > 0)[0]
    shots_used = lgst.round_shots(shots[used], budget).astype(float)
    sub_moments = [moments[i] for i in used]
    sub_designs = [pool_designs[i] for i in used]
    probs = [exact_probs[i] for i in used]
    errors = np.zeros((trials, P))
    for t in range(trials):
        freqs = lgst.resample_frequencies(probs, shots_used, mc_rng)
        if estimator == 'ols':
            est, _ = lgst.estimate_error_rates(sub_designs, [freqs, None], None, params)
        else:
            est, _ = lgst.estimate_error_rates_gls(sub_moments, [freqs, None], params, shots_used,
                                                   nominal_rates=nominal, error_bars=None)
        errors[t] = est - true_rates
    return errors

def predicted_at_truth(shots, estimator='gls'):
    '''The design's predicted error bars, evaluated with the statistics of the true rates.'''
    used = shots > 0
    if estimator == 'gls':
        return np.sqrt(np.diag(lgst.predicted_covariance(info_true, shots)))
    return np.sqrt(np.diag(lgst.predicted_covariance([moments[i] for i in np.nonzero(used)[0]], shots[used], true_rates, 'ols')))

t0 = time.time()
mc_errors, rows = {}, []
for name, shots, estimator in [('(a) random, pseudo-inverse / NNLS', shots_random, 'ols'), ('(a) random, GLS', shots_random, 'gls'),
                               ('(b) tutorial II circuits, GLS', shots_b, 'gls'), ('(c) DᵀD allocation, GLS', shots_c, 'gls'),
                               ('(d) optimal allocation, GLS', shots_opt, 'gls')]:
    mc_errors[name] = monte_carlo(shots, estimator)
    realized = np.sqrt((mc_errors[name] ** 2).mean(axis=0))
    predicted = predicted_at_truth(shots, estimator)
    rows.append({'design, estimator': name, 'RMS error: realized': np.sqrt((realized ** 2).mean()), 'RMS error: predicted': np.sqrt((predicted ** 2).mean()),
                 'H mean: realized': realized[types == 'H'].mean(), 'H mean: predicted': predicted[types == 'H'].mean(),
                 'S mean: realized': realized[types == 'S'].mean(), 'S mean: predicted': predicted[types == 'S'].mean(),
                 'worst parameter: realized': realized.max(), 'worst parameter: predicted': predicted.max()})
print("%d simulated experiments in %.0f s" % (5 * 40, time.time() - t0))
pd.DataFrame(rows).set_index('design, estimator').style.format('{:.2e}')
```

The realized RMS errors agree with the predictions to within a few percent overall and for the Hamiltonian rates, in every design.  The stochastic rates come out 10-30% worse than predicted: their statistical errors are so small that the linearization bias (a few $10^{-5}$, see below) is no longer negligible against them, and the estimator evaluates their variances at its own estimate rather than at the truth.  The worst-determined parameter, conversely, is slightly *better* than predicted, because the non-negativity constraint helps stochastic rates near zero.  Design (d) delivers the promised factor of two in RMS error over the random baseline, and three and a half in the worst case.

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(5.5, 5))
for name, color in [('(a) random, pseudo-inverse / NNLS', 'tab:gray'), ('(d) optimal allocation, GLS', 'tab:red')]:
    realized = np.sqrt((mc_errors[name] ** 2).mean(axis=0))
    predicted = predicted_at_truth(shots_random if name.startswith('(a)') else shots_opt, 'ols' if name.startswith('(a)') else 'gls')
    ax.loglog(predicted[types == 'H'], realized[types == 'H'], 'o', color=color, ms=4, label=name + ' (H)')
    ax.loglog(predicted[types == 'S'], realized[types == 'S'], 's', color=color, ms=4, mfc='none', label=name + ' (S)')
lims = [3e-5, 5e-3]
ax.plot(lims, lims, 'k-', lw=0.8); ax.set_xlim(lims); ax.set_ylim(lims)
ax.set_xlabel('predicted error bar'); ax.set_ylabel('realized RMS error (40 experiments)'); ax.legend(fontsize=8); ax.grid(alpha=0.3)
plt.tight_layout(); plt.show()
```

Finally a single experiment with design (d), analysed as one would analyse real data -- the estimate, its error bars from the inverse information, and (for the simulation only) the systematic error of the linearization from a fit to the exact probabilities:

```{code-cell} ipython3
used_index = np.nonzero(shots_opt > 0)[0]
design_moments = [moments[i] for i in used_index]
design_shots = shots_opt[used_index]
dataset = lgst.resample_frequencies([exact_probs[i] for i in used_index], design_shots, seed=2026)

rates_data, error_bars = lgst.estimate_error_rates_gls(design_moments, [dataset, None], params, design_shots, nominal_rates=nominal)
rates_exact, _ = lgst.estimate_error_rates_gls(design_moments, [[exact_probs[i] for i in used_index], None], params, design_shots,
                                               nominal_rates=nominal, error_bars=None)
for typ, name in [('H', 'Hamiltonian'), ('S', 'stochastic')]:
    mask = types == typ
    print("%-12s linearization error (no shot noise) %.1e | data: mean |estimate - true| %.2e, mean error bar %.2e"
          % (name, np.abs(rates_exact - true_rates)[mask].mean(), np.abs(rates_data - true_rates)[mask].mean(), error_bars[mask].mean()))
z = (rates_data - true_rates) / error_bars
print("fraction of estimates within 1 / 2 error bars of the truth: %.2f / %.2f" % ((np.abs(z) < 1).mean(), (np.abs(z) < 2).mean()))

fig, ax = plt.subplots(figsize=(5.5, 5))
for typ, color, label in [('H', 'tab:blue', 'Hamiltonian'), ('S', 'tab:orange', 'stochastic')]:
    mask = types == typ
    ax.errorbar(true_rates[mask], rates_data[mask], yerr=error_bars[mask], fmt='o', ms=4, color=color, label=label, alpha=0.8)
lim = 1.15 * np.abs(true_rates).max()
ax.plot([-lim, lim], [-lim, lim], 'k-', lw=0.8)
ax.set_xlabel('true rate'); ax.set_ylabel('estimated rate'); ax.set_title('design (d), one experiment of %d shots' % budget)
ax.legend(); ax.grid(alpha=0.3); plt.tight_layout(); plt.show()

mcm_columns = params.indices_for_key(('Iz', mcm_qubit))
fomgi_names = list(mg.fomgi_names_for_ansatz(error_model[('Iz', mcm_qubit)].keys()).values())
pd.DataFrame({'true': true_rates[mcm_columns], 'estimate': rates_data[mcm_columns], 'error bar': error_bars[mcm_columns]},
             index=fomgi_names).round(4)
```

## 6. How much does the design depend on what we assumed?

The information matrices were evaluated at a nominal model, so the design is only *locally* optimal -- optimal for a device with those rates.  This is unavoidable (the statistics of a $\pm1$ observable depend on how often it is flipped) and is shared by every nonlinear design problem; Kosut et al. suggest iterating between estimation and design, and the ACES paper reports that designs optimized at half or twice the true error rates need at most 5-13% more samples.  Here we design at several nominal models and evaluate every design under the *true* statistics:

```{code-cell} ipython3
rows = []
variants = [('true rates', true_rates, 1e-3),
            ('generic: S = 1e-3, H = 0 (used above)', nominal, 1e-3),
            ('generic / 2', nominal / 2, 1e-3),
            ('generic x 2', nominal * 2, 1e-3),
            ('zero rates (ideal statistics, floor only)', None, 1e-3),
            ('generic, variance floor 1e-2', nominal, 1e-2)]
for name, rates, floor in variants:
    shots_v, _ = lgst.optimal_allocation(lgst.circuit_information(moments, rates, min_variance=floor), budget, criterion='A', min_shots=5000)
    rows.append({'nominal model used for the design': name, 'circuits': int((shots_v > 0).sum()),
                 'Phi under the true statistics': lgst.design_criterion(info_true, shots_v)})
rows.append({'nominal model used for the design': '(c) DᵀD information, no statistics', 'circuits': int((shots_c > 0).sum()),
             'Phi under the true statistics': lgst.design_criterion(info_true, shots_c)})
shots_D, _ = lgst.optimal_allocation(info_nominal, budget, criterion='D', min_shots=5000)
rows.append({'nominal model used for the design': 'generic, but the D-criterion', 'circuits': int((shots_D > 0).sum()),
             'Phi under the true statistics': lgst.design_criterion(info_true, shots_D)})
robustness = pd.DataFrame(rows).set_index('nominal model used for the design')
robustness['relative to the design at the true rates'] = robustness['Phi under the true statistics'] / robustness['Phi under the true statistics'].iloc[0]
robustness.style.format({'Phi under the true statistics': '{:.3e}', 'relative to the design at the true rates': '{:.3f}'})
```

Designing with the crude generic model instead of the truth costs half a percent; being wrong by a factor of two either way costs about one percent, and so does a tenfold larger variance floor.  What *does* cost is ignoring the statistics altogether: the ideal-model design (all $\pm1$ observables at the floor) is 14% worse and the $D^{\mathsf T}D$ design of tutorial II 60% worse.  The D-optimal design is 38% worse *in A-value*: the two criteria genuinely prefer different experiments (D is indifferent to the units in which the parameters are measured, A is not), which is why the criterion is a choice about what the experiment is for, not an interchangeable technicality.  In practice one would design from whatever is known about the device, run, re-design from the estimate, and run again; the first round is already nearly optimal.

## 7. Designing for an application

So far every rate counted equally.  But an error model is usually learned *for* something -- to predict, mitigate, or decode the circuits the processor will actually run -- and then the natural figure of merit is the error of those *predictions*.  For application circuits with design matrices $D_a$, the first-order prediction error of a fitted model is $D_a(\hat{\vec\epsilon}-\vec\epsilon)$ per observable, so its mean square is $(\hat{\vec\epsilon}-\vec\epsilon)^{\mathsf T}\,W\,(\hat{\vec\epsilon}-\vec\epsilon)$ with $W = \overline{D_a^{\mathsf T}D_a}$ averaged over all observables of all application circuits -- a weight for the A-criterion (this is the I- or V-criterion of the design literature).  `prediction_weight` builds it, adding a small ridge (by default $10^{-3}$ of the mean diagonal) so that parameters the application never exercises remain *weakly* wanted -- without it the design would, quite logically, not bother to identify them at all.

Take as the application two rounds of the parity check of Section 1 with the data qubits in random states and bases -- the kind of circuit a small error-correction experiment is made of:

```{code-cell} ipython3
app_rng = np.random.default_rng(99)
application = [sandwich(dict(zip((0, 2), app_rng.choice(list(PREPS), 2))), parity_round() * 2,
                        dict(zip((0, 2), app_rng.choice(list(MEAS), 2)))) for _ in range(40)]
application_designs = lgst.create_design_matrix_list(application, params, return_info=True)
D_app = np.vstack([d.design_matrix for d in application_designs])
W_app = lgst.prediction_weight(application_designs)
print("application: %d circuits of depth %d-%d, exercising %d of the %d parameters"
      % (len(application), min(c.depth for c in application), max(c.depth for c in application),
         int((np.abs(D_app) > 1e-9).any(axis=0).sum()), P))

shots_app, result_app = lgst.optimal_allocation(info_nominal, budget, criterion='A', weight=W_app, min_shots=5000)
shots_app = lgst.round_shots(shots_app, budget).astype(float)
has_cz = np.array([any(l.name == 'Gcphase' for layer in c.layertup for l in (layer.components if hasattr(layer, 'components') else [layer]))
                   for c in pool_circuits])
rows = []
for name, shots in [('(d) general purpose', shots_opt), ('(e) targeted at the parity check', shots_app)]:
    rows.append({'design': name, 'circuits': int((shots > 0).sum()),
                 'shots on MCM circuits (%)': 100 * shots[num_mcms > 0].sum() / budget, 'shots on CZ circuits (%)': 100 * shots[has_cz].sum() / budget,
                 'shots on parity-check circuits (%)': 100 * shots[origin == 'parity'].sum() / budget,
                 'predicted RMS prediction error (application)': np.sqrt(lgst.design_criterion(info_nominal, shots, weight=W_app)),
                 'predicted RMS error of a rate (all 57)': np.sqrt(lgst.design_criterion(info_nominal, shots) / P)})
pd.DataFrame(rows).set_index('design').T.map(lambda v: ('%.2e' % v if v < 0.1 else '%.0f' % v) if isinstance(v, float) else v)
```

The targeted design moves the budget onto circuits with mid-circuit measurements and CZ gates (and now uses the parity-check circuits themselves), predicts the application's observables a third better, and in exchange determines the parameters the application does not care about less well -- its RMS error over all 57 rates is almost twice that of the general-purpose design, and would be several times larger without the ridge.  That is the trade-off the weight expresses, and the simulation confirms it:

```{code-cell} ipython3
exact_probs.update({i: noisy_model.probabilities(pool_circuits[i]) for i in np.nonzero(shots_app > 0)[0] if i not in exact_probs})
errors_app = monte_carlo(shots_app)
for name, errors, shots in [('(d) general purpose', mc_errors['(d) optimal allocation, GLS'], shots_opt), ('(e) targeted', errors_app, shots_app)]:
    cov = lgst.predicted_covariance(info_true, shots)
    print("%-22s realized RMS prediction error on the application %.2e  (predicted %.2e)"
          % (name, np.sqrt(np.mean((errors @ D_app.T) ** 2)), np.sqrt(np.mean(np.einsum('ri,ij,rj->r', D_app, cov, D_app)))))
```

## 8. The general case: H+S+C+A errors, mid-circuit measurements and SPAM

Nothing above was specific to Hamiltonian and Pauli-stochastic errors.  The design machinery only ever sees design matrices, ideal expectation values and nominal rates, and all of it is produced by the same error-generator propagation whatever sectors the ansatz contains.  We therefore repeat the whole procedure for the physical H+S+C+A model of the earlier tutorials -- relaxation on every single-qubit gate, a tilted dephasing axis on the $X_{\pi/2}$ gates, common-mode dephasing of the CZ pairs, and an MCM that relaxes before and after the record is formed and has an asymmetric readout error -- fitted with all 28 FOMGI quantities of the MCM:

```{code-cell} ipython3
def relaxation(Pp, Q, gamma):
    '''Amplitude damping (decay probability gamma) as elementary error generators on the Pauli pair (P, Q).'''
    return {('S', Pp): gamma / 4, ('S', Q): gamma / 4, ('A', Pp, Q): -gamma / 4}

def tilted_dephasing(Pp, Q, rate, theta):
    c, s = np.cos(theta), np.sin(theta)
    return {('S', Pp): float(rate * c * c), ('S', Q): float(rate * s * s), ('C', Pp, Q): float(rate * s * c)}

def add_terms(errors, terms):
    for key, rate in terms.items():
        errors[key] = float(errors.get(key, 0.0) + rate)

true_model_hsca = {key: dict(errors) for key, errors in error_model.items()}
for q in pspec.qubit_labels:
    add_terms(true_model_hsca[('Gxpi2', q)], relaxation('X', 'Y', gamma=0.004))
    add_terms(true_model_hsca[('Gxpi2', q)], tilted_dephasing('X', 'Y', rate=0.002, theta=0.3))
    add_terms(true_model_hsca[('Gypi2', q)], relaxation('X', 'Y', gamma=0.004))
for (q1, q2) in edges:
    add_terms(true_model_hsca[('Gcphase', q1, q2)], {('S', 'ZI'): 0.001, ('S', 'IZ'): 0.001, ('C', 'ZI', 'IZ'): 0.001})
true_gadget = dict(error_model[('Iz', mcm_qubit)])
add_terms(true_gadget, relaxation('XX', 'YX', gamma=0.004))
add_terms(true_gadget, relaxation('XI', 'YI', gamma=0.004))
add_terms(true_gadget, relaxation('IX', 'IY', gamma=0.002))
true_model_hsca[('Iz', mcm_qubit)] = true_gadget

ansatz_hsca = {key: dict(errors) for key, errors in true_model_hsca.items()}
ansatz_hsca[('Iz', mcm_qubit)] = mg.fomgi_ansatz(sectors='all')
params_hsca, _ = lgst.build_model_parameter_indexing(ansatz_hsca, qubit_labels=pspec.qubit_labels)
types_hsca = np.array(params_hsca.errorgen_types())
P_hsca = len(params_hsca)

fomgi_true_hsca = mg.fomgi_quantities(true_gadget, sectors='all')
truth_in_ansatz = {key: dict(errors) for key, errors in true_model_hsca.items()}
truth_in_ansatz[('Iz', mcm_qubit)] = mg.fomgi_ansatz(sectors='all', rates=fomgi_true_hsca)
_, true_rates_hsca = lgst.build_model_parameter_indexing(truth_in_ansatz, qubit_labels=pspec.qubit_labels)
true_rates_hsca = np.array(true_rates_hsca)
print(P_hsca, "parameters:", {t: int((types_hsca == t).sum()) for t in 'HSCA'})
```

The candidate circuits are the same (they depend on the gates, not on the ansatz); their moments and information have to be recomputed for the new parameters.  The nominal model is again generic -- every stochastic rate $10^{-3}$, everything else $0$ -- and the design is validated against the truth as before:

```{code-cell} ipython3
t0 = time.time()
moments_hsca = lgst.create_design_moments(pool_circuits, params_hsca)
nominal_hsca = lgst.nominal_rates_by_type(params_hsca, {'S': 1e-3})
info_nominal_hsca = lgst.circuit_information(moments_hsca, nominal_hsca)
info_true_hsca = lgst.circuit_information(moments_hsca, true_rates_hsca)
shots_hsca, result_hsca = lgst.optimal_allocation(info_nominal_hsca, budget, criterion='A', min_shots=5000)
shots_hsca = lgst.round_shots(shots_hsca, budget).astype(float)
print("moments, information and design in %.0f s: %d circuits, certified gap %.1e" % (time.time() - t0, (shots_hsca > 0).sum(), result_hsca.gap))

shots_random_hsca = np.zeros(len(pool_designs))
shots_random_hsca[random_index] = budget / len(random_index)
rows = []
for name, shots in [('300 random circuits, equal shots', shots_random_hsca), ('optimal allocation', shots_hsca)]:
    eb = np.sqrt(np.diag(lgst.predicted_covariance(info_nominal_hsca, shots)))
    rows.append({'design': name, 'circuits': int((shots > 0).sum()), 'RMS error': np.sqrt((eb ** 2).mean()),
                 **{'%s mean' % t: eb[types_hsca == t].mean() for t in 'HSCA'}, **{'%s max' % t: eb[types_hsca == t].max() for t in 'HSCA'}})
pd.DataFrame(rows).set_index('design').style.format({c: '{:.2e}' for c in rows[0] if c not in ('design', 'circuits')})
```

```{code-cell} ipython3
noisy_model_hsca = lgst.rate_to_model(pspec, true_model_hsca)
used_hsca = np.nonzero(shots_hsca > 0)[0]
design_moments_hsca = [moments_hsca[i] for i in used_hsca]
design_shots_hsca = shots_hsca[used_hsca]
exact_hsca = [noisy_model_hsca.probabilities(pool_circuits[i]) for i in used_hsca]

rates_exact_hsca, _ = lgst.estimate_error_rates_gls(design_moments_hsca, [exact_hsca, None], params_hsca, design_shots_hsca,
                                                    nominal_rates=nominal_hsca, error_bars=None)
errors_hsca = []
for t in range(20):
    freqs = lgst.resample_frequencies(exact_hsca, design_shots_hsca, mc_rng)
    est, _ = lgst.estimate_error_rates_gls(design_moments_hsca, [freqs, None], params_hsca, design_shots_hsca,
                                           nominal_rates=nominal_hsca, error_bars=None)
    errors_hsca.append(est - true_rates_hsca)
realized = np.sqrt((np.array(errors_hsca) ** 2).mean(axis=0))
predicted = np.sqrt(np.diag(lgst.predicted_covariance(info_true_hsca, shots_hsca)))
pd.DataFrame([{'sector': t, 'mean |true rate|': np.abs(true_rates_hsca[types_hsca == t]).mean(),
               'linearization error': np.abs(rates_exact_hsca - true_rates_hsca)[types_hsca == t].mean(),
               'RMS error: realized': realized[types_hsca == t].mean(), 'RMS error: predicted': predicted[types_hsca == t].mean(),
               'worst parameter: realized': realized[types_hsca == t].max(), 'worst parameter: predicted': predicted[types_hsca == t].max()}
              for t in 'HSCA']).set_index('sector').style.format('{:.2e}')
```

The predictions hold in all four sectors -- to within 10% on average (the worst-case parameters of the correlation and active sectors come out 15-30% worse than predicted, with twenty simulated experiments and the smallest statistical errors of all).  One detail is instructive: the active (relaxation) rates come out *slightly less* precise than with 300 random circuits.  The unweighted A-criterion minimizes the *sum* of variances, and the active rates, with the smallest variances of all, contribute least to it -- so the optimizer trades a little of their precision for a lot of the Hamiltonian and correlation rates'.  If the active rates were the point of the experiment, a weight (Section 7) would say so.  A single experiment, with the MCM's 28 FOMGI quantities:

```{code-cell} ipython3
dataset_hsca = lgst.resample_frequencies(exact_hsca, design_shots_hsca, seed=2027)
rates_hsca, error_bars_hsca = lgst.estimate_error_rates_gls(design_moments_hsca, [dataset_hsca, None], params_hsca, design_shots_hsca,
                                                            nominal_rates=nominal_hsca)

fig, ax = plt.subplots(figsize=(5.5, 5))
for typ, color, label in [('H', 'tab:blue', 'Hamiltonian'), ('S', 'tab:orange', 'stochastic'), ('C', 'tab:green', 'correlation'), ('A', 'tab:red', 'active')]:
    mask = types_hsca == typ
    ax.errorbar(true_rates_hsca[mask], rates_hsca[mask], yerr=error_bars_hsca[mask], fmt='o', ms=4, color=color, label=label, alpha=0.8)
lim = 1.15 * np.abs(true_rates_hsca).max()
ax.plot([-lim, lim], [-lim, lim], 'k-', lw=0.8)
ax.set_xlabel('true rate'); ax.set_ylabel('estimated rate'); ax.set_title('H+S+C+A, optimal design, one experiment')
ax.legend(); ax.grid(alpha=0.3); plt.tight_layout(); plt.show()

mcm_columns_hsca = params_hsca.indices_for_key(('Iz', mcm_qubit))
fomgi_hsca = lgst.mcm_fomgi_estimates(params_hsca, rates_hsca)[('Iz', mcm_qubit)]
table = pd.DataFrame({'sector': [mg.FOMGI_SECTORS[name] for name in fomgi_hsca], 'true': [fomgi_true_hsca[name] for name in fomgi_hsca],
                      'estimate': list(fomgi_hsca.values()), 'error bar': error_bars_hsca[mcm_columns_hsca]}, index=list(fomgi_hsca.keys()))
highlight = lambda row: ['background-color: #fff3cd' if row['sector'] in ('C', 'A') else '' for _ in row]
table.style.format(precision=4).apply(highlight, axis=1)
```

## 9. Summary

**The figure of merit.**  An experiment -- a set of circuits with shot counts $N_c$ summing to the budget -- is scored by the expected squared error of the error-rate estimate it will produce,

$$\Phi_W(\vec N) = \mathrm{tr}\big[W\,\mathcal I(\vec N)^{-1}\big],\qquad \mathcal I(\vec N) = \sum_c N_c\,D_c^{\mathsf T}\Sigma_c^{-1}D_c,$$

where $D_c$ is the circuit's design matrix, $\Sigma_c$ the per-shot covariance of its observables under a nominal error model (first and second moments, both first order in the rates, with a floor on the eigenvalues), and $W$ the metric in which errors are measured: the identity (expected RMS error of a rate, the ACES figure of merit) by default, or the application's own metric $\overline{D_a^{\mathsf T}D_a}$ when the model is learned to predict specific circuits.  $\mathcal I^{-1}$ is the Cramér-Rao bound; the generalized least-squares estimator attains it; and minimizing $\Phi_W$ over the shot counts of a candidate pool is a convex problem with a certificate of optimality.

**The recipe.**

1. Build a large, varied pool of shallow candidate circuits (random circuits with MCMs, structured probes, application circuits); check that it identifies every parameter (`unidentifiable_directions`).
2. Compute design matrices and second moments once (`create_design_matrix_list`, `create_design_moments`); the pool is reusable for any ansatz on the same gates.
3. Choose a nominal model -- typical error magnitudes are enough -- and compute the per-shot information of every candidate (`circuit_information`; use `cost` if shots of different circuits cost different amounts of time).
4. Choose $W$ (and, if the point of the experiment is a few parameters or an application, say so through it) and solve `optimal_allocation` with a `min_shots` floor; round with `round_shots`.
5. Run; fit with `estimate_error_rates_gls`; if the estimate differs materially from the nominal model, re-design from it for the next run.

**What the figure of merit does not cover.**  The linearization bias (keep the pool shallow; check with two depths), errors outside the ansatz (keep the design diverse -- `min_shots` and `max_shots` are the knobs), and drift (re-design between runs).

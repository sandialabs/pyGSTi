---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.19.6
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Crosstalk-free GST on many qubits, one lane at a time

This notebook walks through fitting a **crosstalk-free** model to **simultaneous GST** data
with `pygsti.extras.crosstalkfreegst`, entirely in simulation:

1. build a simultaneous-GST experiment design on a line of qubits;
2. simulate data from a noisy crosstalk-free "truth";
3. fit the *general* crosstalk-free model to the data;
4. gauge-optimize the estimate over the per-qubit product gauge group;
5. tabulate per-gate and per-qubit errors, and compare them with the truth;
6. repeat with mid-circuit measurements (single-qubit instruments).

**Why this scales.** In a crosstalk-free model every gate acts only on its target qubits, and
state preparation and measurement are products of single-qubit operations. Split a circuit
into *lanes* -- groups of qubits linked by gates. Then the circuit's outcome distribution is
the product of its lanes' distributions, and the log-likelihood is *exactly* a sum of lane
terms that use the counts marginalized onto each lane:

$$\sum_x N(x)\log p(x) = \sum_{\text{lanes}} \sum_{x_\ell} N_\ell(x_\ell)\log p_\ell(x_\ell).$$

The lane terms share parameters (the gates on qubit 3 appear in lanes $(3)$, $(2,3)$ and
$(3,4)$), so this is still one joint fit. However, every probability, derivative and gauge
transformation involves at most a two-qubit object, and the cost grows linearly, not
exponentially, with the number of qubits.

We use 4 qubits so the notebook runs in a few minutes. Nothing below changes at 30+ qubits
except run time; the last section shows the data path to use there.

```{code-cell} ipython3
import time
import warnings
import numpy as np

import pygsti
from pygsti.processors import QubitProcessorSpec
from pygsti.models import create_crosstalk_free_model
from pygsti.modelpacks import smq1Q_XYI, smq2Q_XYICNOT
from pygsti.protocols import make_simultaneous_gst_design

import pygsti.extras.crosstalkfreegst as xf

warnings.filterwarnings('ignore')
np.set_printoptions(precision=4, linewidth=120)
```

## 1. Device and experiment design

A line of `n` qubits with $X_{\pi/2}$, $Y_{\pi/2}$ and idle gates on every qubit, and a CNOT
and a two-qubit idle `Gii` on every edge. `Gii` pads the shorter two-qubit circuits when
designs are stitched together.

```{code-cell} ipython3
n = 4
max_length = 4   # longest germ power: the general model needs at least 4 (shorter leaves 2Q gates poorly determined)

qubits = tuple(range(n))
edges = [(q, q + 1) for q in range(n - 1)]
singles = [(q,) for q in qubits]
pspec = QubitProcessorSpec(n, gate_names=['Gi', 'Gxpi2', 'Gypi2', 'Gcnot', 'Gii'],
                           nonstd_gate_unitaries={'Gii': np.identity(4)},
                           availability={'Gi': singles, 'Gxpi2': singles, 'Gypi2': singles,
                                         'Gcnot': edges, 'Gii': edges},
                           qubit_labels=qubits)
```

A `SimultaneousGSTDesign` stitches a 1Q and a 2Q GST design into circuits on all `n` qubits. It
runs the 2Q design on a set of non-overlapping edges (a "patch" of an edge coloring) and the
1Q design on the remaining qubits.

```{code-cell} ipython3
oneq_design = smq1Q_XYI.create_gst_experiment_design(max_max_length=max_length, qubit_labels=(0,))
twoq_design = smq2Q_XYICNOT.create_gst_experiment_design(max_max_length=max_length, qubit_labels=(0, 1))
design = make_simultaneous_gst_design(pspec, oneq_design, twoq_design, seed=1)

circuits = design.all_circuits_needing_data
print(f"{len(circuits)} circuits on {n} qubits; edge coloring: {design.color_patches}")
```

Each circuit is split into lanes independently. Nothing assumes a fixed patch structure, so
datasets that mix different lane structures work unchanged. Note the circuit's line order:
stitched circuits carry patch-specific orders, and outcome bit strings follow each circuit's
own `line_labels`.

```{code-cell} ipython3
c = circuits[-1]
print(c.str, '\n')
for support, lane_circuit in xf.split_circuit_into_lanes(c, qubits):
    print(support, lane_circuit.str)
```

## 2. A noisy crosstalk-free "truth", and simulated data

The truth has random Hamiltonian and stochastic errors on every gate and on every qubit's
state preparation and measurement, independently per qubit and per edge.

For the truth we use the `LaneModels` *factory* form. It builds a model on a lane's qubits
whenever one is needed, from the processor spec restricted to that lane, and ties all lane
models' parameters into one global vector by keys such as `('gate', Label('Gxpi2', 3))`. The
factory form never builds an `n`-qubit model, which matters beyond about 12 qubits, where
pyGSTi's `n`-qubit model needs a dense $4^n$ state vector. (For small `n`,
`xf.LaneModels.from_model(model)` restricts an ordinary `n`-qubit model instead.)

```{code-cell} ipython3
errors = {
    'Gi':    {('H', 'Z'): 0, ('S', 'X'): 0, ('S', 'Z'): 0},
    'Gxpi2': {('H', 'X'): 0, ('H', 'Z'): 0, ('S', 'X'): 0, ('S', 'Z'): 0},
    'Gypi2': {('H', 'Y'): 0, ('H', 'Z'): 0, ('S', 'Y'): 0, ('S', 'Z'): 0},
    'Gcnot': {('H', 'ZZ'): 0, ('H', 'IX'): 0, ('H', 'ZI'): 0, ('S', 'XX'): 0, ('S', 'ZZ'): 0, ('S', 'IX'): 0},
    'Gii':   {('H', 'ZI'): 0, ('H', 'IZ'): 0, ('H', 'ZZ'): 0, ('S', 'ZI'): 0, ('S', 'IZ'): 0},
    'prep':  {('H', 'X'): 0, ('S', 'X'): 0},
    'povm':  {('H', 'X'): 0, ('S', 'X'): 0},
}

def noisy_lane_model(lane_pspec):
    return create_crosstalk_free_model(lane_pspec, lindblad_error_coeffs=errors,
                                       lindblad_parameterization='H+s', independent_gates=True)

truth = xf.LaneModels(pspec, noisy_lane_model)
for support in singles + edges:     # build every lane model, so every parameter exists
    truth[support]

# Random error rates: Hamiltonian coefficients in [-0.02, 0.02], stochastic rates in [0, 0.002].
labels = np.empty(truth.num_params, dtype=object)
for _, (lane_model, global_indices) in truth.items():
    labels[global_indices] = [lbl[1] for lbl in lane_model.parameter_labels]
stochastic = np.array(['stochastic' in lbl for lbl in labels])
rng = np.random.default_rng(2026)
truth.set_vector(np.where(stochastic, rng.uniform(0, 2e-3, truth.num_params),
                          rng.uniform(-0.02, 0.02, truth.num_params)))
print(f"truth: {truth.num_params} parameters")
```

`simulate_lanewise_data` samples each shot lane by lane and assembles full bit strings. The
result has the same distribution as `pygsti.data.simulate_data`, but needs only lane-sized
simulations. Its output is an ordinary `DataSet`, just like real data would be.

```{code-cell} ipython3
t0 = time.time()
dataset = xf.simulate_lanewise_data(truth, circuits, num_samples=1000, seed=7)
print(f"simulated {len(dataset)} circuits x 1000 shots in {time.time() - t0:.1f}s")
print(dataset[c])
```

## 3. Fit the general crosstalk-free model

`create_general_crosstalk_free_model` builds the most general crosstalk-free model. Each gate
placement is an arbitrary trace-preserving map on its target qubits, and each qubit has
arbitrary trace-preserving errors after an ideal $|0\rangle$ preparation and before an ideal
computational-basis measurement. This is the model class whose structure-preserving gauge
group is the tensor product of single-qubit gauge groups. (Its SPAM error maps carry some
unidentifiable parameters; these are flat directions of the likelihood, and harmless.)

`run_crosstalk_free_gst` mirrors pyGSTi's iterative GST: a $\chi^2$ fit for each germ-power
circuit list, seeded by the previous one, then a final log-likelihood fit. Every objective is
evaluated lane by lane, and a Levenberg-Marquardt optimizer works on the sparse normal
equations that the lane terms assemble into.

The lane terms are independent given the parameters, so `processes` spreads them over worker
processes. Each worker builds its lanes' objective functions once per fit stage, and then
exchanges only parameter-sized arrays with the optimizer. Workers take a few seconds to start,
so this pays off for fits longer than about a minute. The speedup is limited by the number of
lanes and by the largest lane: here there are 7 lanes, 3 of them much heavier two-qubit
lanes.

```{code-cell} ipython3
import os
estimate = xf.LaneModels(pspec, xf.create_general_crosstalk_free_model)

t0 = time.time()
results = xf.run_crosstalk_free_gst(dataset, estimate, design.circuit_lists,
                                    processes=min(4, os.cpu_count()), verbosity=1)
print(f"\nfit {estimate.num_params} parameters in {time.time() - t0:.0f}s")
```

**Goodness of fit.** The objective is pyGSTi's $\Delta\log L$ (not $2\Delta\log L$), computed
from lane-marginal counts. A model that is right up to statistical noise gives
$\Delta\log L \approx k/2$. Here $k$ is the number of independent lane outcome frequencies
(one fewer than the number of outcomes per lane circuit) minus the number of identifiable
parameters. Even the crude count below, which ignores the parameters, shows we are in that
regime, as expected: the truth is inside the model class.

```{code-cell} ipython3
final = results.iterations[-1]
k = final.num_elements - final.num_lane_circuits
print(f"Delta(logL) = {results.final_fn:.0f};  k/2 (ignoring parameters) = {k / 2:.0f}")
print("per lane:", {s: round(v) for s, v in results.lane_fns.items()})
```

## 4. Gauge optimization over the per-qubit product gauge group

A product gauge $S = \bigotimes_q g_q$ maps a crosstalk-free model to another crosstalk-free
model:
- a gate on qubits $Q$ becomes $S_Q^{-1} G S_Q$;
- qubit $q$'s state becomes $g_q^{-1}\rho_q$;
- qubit $q$'s effects become $e\,g_q$.

The distance to a target is therefore a sum of terms that each involve only one or two
qubits' gauges, and `gauge_optimize` minimizes it with the same sparse machinery as the fit.
It first fits each qubit on its own single-qubit gates and SPAM, then fits all qubits jointly
with the two-qubit gates included. The target is the ideal model.

```{code-cell} ipython3
target = xf.LaneModels(pspec, xf.create_general_crosstalk_free_model)
gauge = xf.gauge_optimize(estimate, target)
print(f"squared distance to target: {gauge.fn_before:.6f} -> {gauge.fn_after:.6f}")
print("qubit 0's gauge g_0:\n", gauge.gauges[0])
```

The fit starts at the target, and pure-gauge directions don't change the likelihood. So the
estimate is usually already close to the target's gauge, and the change above is small.
Gauge optimization matters more when the starting point or the data put the estimate in a
different gauge. The gauge transformation leaves every predicted probability unchanged (the
package's tests check this to machine precision).

## 5. Per-gate and per-qubit errors

`error_summary` reports, for every gate, the entanglement and average gate infidelity (and
half the diamond distance, if `cvxpy` is installed). For every qubit it reports the
preparation infidelity and the readout error rates $P(1|0)$ and $P(0|1)$. These numbers are
gauge-dependent, hence the gauge optimization above.

```{code-cell} ipython3
rows = xf.error_summary(estimate, target)
print(xf.format_error_summary(rows))
```

In the table above, readout rates can come out slightly negative. The general model is
trace-preserving but not constrained to be completely positive, so statistical noise can push
an estimate outside the physical set. The gauge can too: a gauge transformation need not keep
each gate or measurement physical, so even a physical model can have slightly unphysical
members in the gauge closest to the target.

Since this is a simulation, we can compare with the truth. The two have to be compared in the
same gauge, and the truth is in its own gauge, not the one closest to the target. So we move
the estimate to the gauge closest to the truth, and then summarize both against the target.
What remains is statistical noise, which shrinks with longer germ powers (`max_length`) and
more shots.

```{code-cell} ipython3
xf.gauge_optimize(estimate, truth)   # (in place: from here on the estimate is in the truth's gauge)
est_rows = xf.error_summary(estimate, target, diamond=False)
true_rows = {(r['kind'], r['qubits'], r['name']): r for r in xf.error_summary(truth, target, diamond=False)}
print(f"{'gate':<8}{'qubits':<9}{'estimate':>10}{'truth':>10}   (entanglement infidelity)")
for r in est_rows:
    if r['kind'] == 'gate':
        tr = true_rows[r['kind'], r['qubits'], r['name']]
        print(f"{r['name']:<8}{str(r['qubits']):<9}{r['entanglement_infidelity']:10.2e}{tr['entanglement_infidelity']:10.2e}")
```

## 6. Working with the estimate

Each lane model is an ordinary pyGSTi model on one or two qubits, holding the estimate
restricted to that qubit or edge.

```{code-cell} ipython3
edge_model, _ = estimate[(1, 2)]
print(type(edge_model).__name__, edge_model.state_space)
print(edge_model.operation_blks['gates'][pygsti.baseobjs.Label('Gcnot', (1, 2))].to_dense().round(3)[:4])
```

Parameters are also available by member key:

```{code-cell} ipython3
params = estimate.parameters()
print(len(params), "members, e.g.", list(params)[:3])
```

For small `n` you can load the estimate into an ordinary `n`-qubit pyGSTi model. Giving it a
`CrosstalkFreeForwardSimulator` makes all of pyGSTi's simulation fast: `probabilities`,
`simulate_data`, objective functions, even the standard `GateSetTomography` protocol. The
simulator splits each circuit into lanes, simulates each lane with a small model, and
multiplies the lanes' marginal probabilities (derivatives follow by the product rule). On
200 simultaneous-GST circuits on 10 qubits it computed probabilities in about 1 s, against
about 280 s for pyGSTi's map simulator, with identical results. (pyGSTi still builds the
`n`-qubit model and enumerates all $2^n$ outcomes per circuit, which limits this route to
roughly a dozen qubits.)

```{code-cell} ipython3
full_estimate = estimate.to_model(xf.create_general_crosstalk_free_model(pspec))
full_estimate.sim = xf.CrosstalkFreeForwardSimulator()
print(full_estimate.probabilities(pygsti.circuits.Circuit([('Gxpi2', 0), ('Gcnot', 0, 1)], line_labels=qubits)))
```

## 7. Mid-circuit measurements

Single-qubit instruments (mid-circuit measurements) fit into the same scheme. An instrument on
qubit $q$ acts only on $q$, and its outcome belongs to $q$'s lane. So the joint distribution of
all mid-circuit and final outcomes is still a product over lanes, and the likelihood still
factorizes exactly. A product gauge maps each instrument element as
$E_k \mapsto g_q^{-1} E_k g_q$, so the gauge structure carries over too.

Declare the instrument names in the processor spec, so circuits can be split and outcomes
assigned to lanes:

```{code-cell} ipython3
pspec_mcm = QubitProcessorSpec(n, gate_names=['Gi', 'Gxpi2', 'Gypi2', 'Gcnot', 'Gii'],
                               nonstd_gate_unitaries={'Gii': np.identity(4)},
                               availability={'Gi': singles, 'Gxpi2': singles, 'Gypi2': singles,
                                             'Gcnot': edges, 'Gii': edges},
                               instrument_names=('Iz',), qubit_labels=qubits)
```

**Designs.** `create_simple_mcm_oneq_design` adds a Z-basis mid-circuit measurement `Iz`
to the standard 1Q design, as one more germ. `create_simple_mcm_twoq_design` adds the 1Q
design's measurement circuits, on each qubit in turn, to the standard 2Q design. Every qubit's
measurement has to be exercised somewhere. A qubit that every patch of the edge coloring covers,
like each interior qubit of a line, never runs 1Q component circuits, so with measurements in the
1Q design only, its instrument would be completely undetermined (checked from the rank of the
fit's Gauss-Newton matrix). Putting measurements in the 2Q design is one fix; adding circuits that
run 1Q measurement circuits on those qubits is another. With both component designs containing
measurements, every gauge-invariant parameter is determined.

```{code-cell} ipython3
mcm_design = make_simultaneous_gst_design(pspec_mcm, xf.create_simple_mcm_oneq_design(max_length),
                                          xf.create_simple_mcm_twoq_design(max_length), seed=1)
mcm_circuits = mcm_design.all_circuits_needing_data
c_mcm = next(c for c in mcm_circuits if c.str.count('Iz') >= 2)
print(f"{len(mcm_circuits)} circuits, {sum('Iz' in c.str for c in mcm_circuits)} with mid-circuit measurements\n")
print(c_mcm.str, '\n')
for support, lane_circuit in xf.split_circuit_into_lanes(c_mcm, qubits):
    print(support, lane_circuit.str)
```

**Models.** `add_single_qubit_instruments` adds an independent instrument per qubit to a
crosstalk-free model (pyGSTi's `create_crosstalk_free_model` doesn't build instruments yet).
By default its parameterization matches the gates': pyGSTi's effect-then-gate CPTP
parameterization (`'CPTPLND'`) for Lindblad models like the truth below, and full TP for the
general model. `create_general_crosstalk_free_model(..., instruments=...)` does the latter.

```{code-cell} ipython3
def noisy_lane_model_with_mcm(lane_pspec):
    return xf.add_single_qubit_instruments(noisy_lane_model(lane_pspec), {'Iz': 'Z'})

truth_mcm = xf.LaneModels(pspec_mcm, noisy_lane_model_with_mcm)
for support in singles + edges:
    truth_mcm[support]
labels = np.empty(truth_mcm.num_params, dtype=object)
for _, (lane_model, global_indices) in truth_mcm.items():
    labels[global_indices] = [str(lbl[1]) for lbl in lane_model.parameter_labels]
stochastic = np.array(['stochastic' in lbl for lbl in labels])
hamiltonian = np.array(['Hamiltonian' in lbl for lbl in labels])
rng = np.random.default_rng(2027)
truth_mcm.set_vector(truth_mcm.current_vector() + np.where(
    stochastic, rng.uniform(0, 2e-3, truth_mcm.num_params),
    np.where(hamiltonian, rng.uniform(-0.02, 0.02, truth_mcm.num_params), rng.uniform(-0.01, 0.01, truth_mcm.num_params))))
print(f"truth: {truth_mcm.num_params} parameters, of which "
      f"{sum(len(i) for k, i in truth_mcm.keys.items() if k[0] == 'instrument')} are the instruments'")
```

**Data.** Outcome labels follow pyGSTi's convention: one outcome per mid-circuit measurement,
in circuit order, then the final bits.

```{code-cell} ipython3
mcm_dataset = xf.simulate_lanewise_data(truth_mcm, mcm_circuits, num_samples=1000, seed=8)
print(dict(list(mcm_dataset[c_mcm].counts.items())[:4]))
```

**Fit, gauge optimization and errors** work exactly as before:

```{code-cell} ipython3
def general_with_mcm(lane_pspec):
    return xf.create_general_crosstalk_free_model(lane_pspec, instruments={'Iz': 'Z'})

estimate_mcm = xf.LaneModels(pspec_mcm, general_with_mcm)
t0 = time.time()
results_mcm = xf.run_crosstalk_free_gst(mcm_dataset, estimate_mcm, mcm_design.circuit_lists,
                                        processes=min(4, os.cpu_count()), verbosity=1)
print(f"\nfit {estimate_mcm.num_params} parameters in {time.time() - t0:.0f}s")

target_mcm = xf.LaneModels(pspec_mcm, general_with_mcm)
gauge_mcm = xf.gauge_optimize(estimate_mcm, target_mcm)
print(f"squared distance to target: {gauge_mcm.fn_before:.6f} -> {gauge_mcm.fn_after:.6f}\n")
rows_mcm = xf.error_summary(estimate_mcm, target_mcm, diamond=False)
print(xf.format_error_summary(rows_mcm))
```

The instrument rows report pyGSTi's instrument infidelity, and the measurement's error rates
$P(k|j) = \mathrm{Tr}\,E_k(|j\rangle\langle j|)$ on ideal inputs. In the gauge closest to
the target, the error rates above are all negative. Much of that is the gauge, as described in
section 5: the true model has negative rates in that gauge too.

Below we compare with the truth in the truth's gauge. The truth's error rates are a few
$\times 10^{-4}$. That is below what this small demo (germ powers up to 4, 1000 shots)
resolves, so the estimates scatter around them, sometimes below zero: the general model is
trace-preserving but not constrained to be physical. Longer germ powers and more shots tighten
them.

```{code-cell} ipython3
xf.gauge_optimize(estimate_mcm, truth_mcm)   # (in place, as in section 5)
true_mcm = {(r['kind'], r['qubits'], r['name']): r for r in xf.error_summary(truth_mcm, target_mcm, diamond=False)}
print(f"{'qubit':<7}{'infidelity: est.':>18}{'truth':>10}{'P(1|0): est.':>15}{'truth':>10}{'P(0|1): est.':>15}{'truth':>10}")
for r in xf.error_summary(estimate_mcm, target_mcm, diamond=False):
    if r['kind'] == 'instrument':
        t = true_mcm[r['kind'], r['qubits'], r['name']]
        print(f"{r['qubits'][0]:<7}{r['instrument_infidelity']:18.2e}{t['instrument_infidelity']:10.2e}"
              f"{r['mcm_p1_given_0']:15.2e}{t['mcm_p1_given_0']:10.2e}{r['mcm_p0_given_1']:15.2e}{t['mcm_p0_given_1']:10.2e}")
```

**Caveat.** Mid-circuit measurements are a common source of crosstalk, e.g. measurement-induced
dephasing of neighbouring qubits. A crosstalk-free model can't represent that, so in real data
it would show up as model violation (an excess $\Delta\log L$), not as error on the measured qubit.

## 8. Going to tens of qubits

Two things change at scale.

1. **Models:** use the factory form of `LaneModels`, as above, never an `n`-qubit model.
2. **Data:** a joint `DataSet` stores every distinct outcome string. At 32 qubits and 1000
   shots almost every shot is distinct, giving millions of labels and several GB. The fit
   only needs lane marginals, so stream counts into a `LaneDecomposition` instead and pass
   that in place of the dataset.

```python
decomposition = xf.LaneDecomposition(pspec.qubit_labels)
for circuit, counts in my_count_source:          # counts: {'0110...': n, ...} over circuit.line_labels
    decomposition.add(circuit, counts)
results = xf.run_crosstalk_free_gst(decomposition, estimate, design.circuit_lists)
```

`xf.simulate_lanewise_marginals(...)` produces such a decomposition directly. With it, a
32-qubit line fit (an `H+s` model, 821 parameters, 110,568 lane circuits) took about 7
minutes and 3.2 GB of peak memory on the 7 GB development machine. The general model has
more parameters per lane (about 600 per 2-qubit lane, against about 40 here) and fits more
slowly: about a minute for this notebook's 4 qubits with 4 processes, growing linearly with
the number of qubits.

**Caveat for full-model comparisons.** Fitting an ordinary `n`-qubit `LocalNoiseModel` with
parameterized measurement errors to stitched circuits currently gives wrong results. pyGSTi
handles circuits whose line order differs from the model's through a static, cached
marginalized POVM. This package's lane models are unaffected, because lane circuits are
always in the lane model's own order, and so is an `n`-qubit model that uses the
`CrosstalkFreeForwardSimulator` of section 6. For a comparison with one of pyGSTi's general
simulators, reorder the circuits' lines to model order first, and permute the outcome bits to
match.

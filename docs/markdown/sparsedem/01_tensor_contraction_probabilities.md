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

# 01 — Exact detector-outcome probabilities by tensor-network contraction

A detector error model (DEM) is a list of independent error events. Event $i$ fires with probability $p_i$ and flips the detectors in a set $T_i$ (and possibly some logical observables). Writing $x_i \in \{0, 1\}$ for "event $i$ fired", the probability of observing the detector outcome $s \in \{0,1\}^n$ is

$$
P(s) \;=\; \sum_{x \in \{0,1\}^m} \;\prod_{i=1}^{m} (1-p_i)^{1-x_i} p_i^{x_i} \;\prod_{d=1}^{n} \Big[\, \textstyle\bigoplus_{i : d \in T_i} x_i \;=\; s_d \Big],
$$

a sum over $2^m$ error configurations. `pygsti.extras.sparsedem.estimation.compute_outcome_distribution_from_dem` evaluates the whole distribution at once with a Walsh–Hadamard product formula; that costs $O(n\,2^n)$ time (and, as implemented, a dense $2^n \times 2^n$ Hadamard matrix), so it stops being practical at roughly $n \approx 12$–$14$ detectors.

`pygsti.extras.sparsedem.tensor_contraction` instead evaluates $P(s)$ for *one* outcome (or a small marginal) exactly, by writing the sum above as a tensor network and contracting it. The cost is governed by how "wide" the DEM's detector–event hypergraph is, not by $n$, so chain-like DEMs (repetition codes) with dozens of detectors take milliseconds.

```{code-cell} ipython3
import time
import warnings

import numpy as np
import stim

from pygsti.extras.sparsedem import tensor_contraction as tc
from pygsti.extras.sparsedem.estimation import compute_outcome_distribution_from_dem
from pygsti.extras.sparsedem.io import dem_from_str
from pygsti.extras.sparsedem.utils import counts_from_samples

warnings.filterwarnings("ignore", message=".*kahypar.*")  # cotengra's optional-dependency notice
print("quimb backend available:", tc.quimb_available())
```

## The tensor network

Three kinds of tensors, all with binary indices, turn the sum into a network:

1. **Probability vector** — for each event $i$, the vector $\begin{pmatrix} 1-p_i & p_i \end{pmatrix}$ on an index `e{i}` carrying $x_i$.
2. **Copy tensor** — $x_i$ is needed by every detector in $T_i$. The all-equal (delta) tensor $\delta_{x_i, a_{i,d_1}, a_{i,d_2},\dots}$, which is 1 iff all its indices agree, fans `e{i}` out to one auxiliary index `e{i}_D{d}` per detector $d \in T_i$. (By default the module fuses 1 and 2 into a single *weighted* copy tensor whose two nonzero entries are $1-p_i$ and $p_i$.)
3. **Parity (XOR) tensor** — for each detector $d$, a tensor on the auxiliary indices of all events touching $d$, equal to 1 iff their parity is $s_d$.

Summing over every index of this network is exactly the sum over $x$: the copy tensors force each event bit to be seen consistently by all of its detectors, the parity tensors implement the indicator brackets, and the probability vectors supply the Bernoulli weights. Hence

$$
P(s) = \text{contraction of the network built from } s .
$$

Two refinements make this practical. If a detector's parity tensor is given an extra, *open* output index instead of being fixed to $s_d$, the contraction returns the marginal distribution over that detector; and if a detector is dropped from the network altogether (which equals closing its parity tensor with an all-ones tensor) it is marginalised. A parity tensor over $k$ inputs has $2^k$ entries, so tensors are split into chains of rank at most `max_rank` (default 8) with carry indices — a detector touched by 30 events never materialises a $2^{30}$ array.

Let's build the network for the second example DEM of the prototype notebook, keeping the textbook (unfused) structure so all three tensor types are visible.

```{code-cell} ipython3
dem = dem_from_str("""
error(0.01) D0
error(0.02) D0 D1
error(0.01) D1
error(0.02) D0 D2
error(0.02) D2
""")

# outcome bits in detector-index order: detector 0 fired, detectors 1 and 2 did not
tn = tc.dem_to_tensor_network(dem, detector_bits=[1, 0, 0], fuse_events=False)
print(tn)
for t in tn.tensors:
    print(f"{t.tag:>3}  inds={t.inds}")
    print(np.array2string(t.data, prefix="       "))
```

The parity tensor `D0` acts on the three events that touch detector 0 and is 1 exactly on the odd-parity entries because we asked for $s_0 = 1$; `D1` and `D2` select even parity.

## Verification against the dense distribution

A note on bit ordering, because it is the easiest thing to get wrong in this package: the functions here take outcomes as `bits[d]` = outcome of detector `d` (stim's sampling order). The dense array from `compute_outcome_distribution_from_dem` is indexed by the integer mask with bit `d` = detector `d`, and sparsedem's counts dictionaries are keyed by the *reversed* bitstring. `tc.mask_to_detector_bits`, `tc.bitstring_to_detector_bits` and `tc.detector_bits_to_bitstring` convert between them.

```{code-cell} ipython3
dense = compute_outcome_distribution_from_dem(dem)
n = dem.num_detectors
rows = []
for mask in range(2 ** n):
    bits = tc.mask_to_detector_bits(mask, n)
    p = tc.detector_outcome_probability(dem, bits)
    rows.append((tc.detector_bits_to_bitstring(bits), p, dense[mask]))
    assert abs(p - dense[mask]) < 1e-14

print("key    P(tensor network)   P(dense)")
for key, p, q in rows:
    print(f"{key}    {p:.12f}      {q:.12f}")
print("sum over outcomes:", sum(r[1] for r in rows))
```

The `bitstring=` keyword accepts the sparsedem key directly, and both contraction backends agree to machine precision:

```{code-cell} ipython3
p_np = tc.detector_outcome_probability(dem, bitstring="001", backend="numpy")
p_auto = tc.detector_outcome_probability(dem, bitstring="001")  # quimb if installed, else numpy
print(p_np, p_auto, abs(p_np - p_auto))
```

## Marginals and conditionals

Leaving parity tensors open gives joint marginals over any subset of detectors, without enumerating $2^n$ outcomes. The result has one axis per requested detector, indexed by its bit.

```{code-cell} ipython3
axes = dense.reshape((2,) * n).transpose(2, 1, 0)  # axis d <-> detector d

m02 = tc.marginal_distribution(dem, [0, 2])         # P(D0, D2)
print("P(D0, D2) =\n", m02)
print("matches dense:", np.allclose(m02, axes.sum(axis=1)))

joint = tc.marginal_distribution(dem, [2], condition={1: 1})  # P(D2, D1 = 1)
print("P(D2 | D1 = 1) =", joint / joint.sum())
print("matches dense:", np.allclose(joint / joint.sum(), axes[:, 1, :].sum(axis=0) / axes[:, 1, :].sum()))
```

Logical observables (`L` targets) are marginalised by default; pass `observable_bits=` to condition on them, or use the label `"L0"` in `detectors`/`condition` to treat an observable like a detector.

## A circuit DEM too large for the dense method

A distance-7 repetition-code memory with 7 rounds has 48 detectors — $2^{48}$ outcomes, hopeless for the dense method — but its DEM is a narrow ladder, so the network contracts in milliseconds. (The `auto` backend uses quimb when installed; most of the wall time below is cotengra's path search, which is done once per network structure.)

```{code-cell} ipython3
circuit = stim.Circuit.generated("repetition_code:memory", distance=7, rounds=7,
                                 after_clifford_depolarization=0.01)
dem_rep = circuit.detector_error_model()
n_events = sum(1 for inst in dem_rep.flattened() if inst.type == "error")
print(f"{dem_rep.num_detectors} detectors, {n_events} events, 2^n = {2 ** dem_rep.num_detectors:.1e}")

sampler = dem_rep.compile_sampler(seed=1)
det, obs, _ = sampler.sample(shots=2000)
det = det.astype(np.uint8)

tn_rep = tc.dem_to_tensor_network(dem_rep, det[0])
plan = tn_rep.plan(backend="numpy")
print(tn_rep)
print("largest intermediate tensor: 2^%d entries" % plan.width)

t0 = time.perf_counter()
p_quiet = tc.detector_outcome_probability(dem_rep, np.zeros(dem_rep.num_detectors, dtype=np.uint8))
p_shot = tc.detector_outcome_probability(dem_rep, det[0])
dt = time.perf_counter() - t0
print(f"P(no detector fires) = {p_quiet:.6f}")
print(f"P(first sampled syndrome, weight {det[0].sum()}) = {p_shot:.3e}")
print(f"two exact probabilities in {dt * 1e3:.0f} ms")
```

Marginals are just as cheap. Here is the joint distribution of two neighbouring detectors in the same round, compared with the sample frequencies:

```{code-cell} ipython3
pair = [12, 13]
m = tc.marginal_distribution(dem_rep, pair)
emp = np.zeros((2, 2))
for a, b in det[:, pair]:
    emp[a, b] += 1
emp /= len(det)
print("exact P(D12, D13):\n", np.round(m, 4))
print("empirical (2000 shots):\n", np.round(emp, 4))
```

## Log-likelihood model comparison

Because every outcome uses the same network structure (only the parity tensors' data change), `outcome_probabilities` / `log_likelihood` plan the contraction once and evaluate all distinct syndromes in vectorised batches. That turns the exact log-likelihood $\sum_s N_s \log P(s)$ into a routine quantity.

As a demonstration, let the *true* noise contain a correlated hyperedge that the circuit-derived DEM lacks. Sampling from the true model, the exact log-likelihood should prefer the true DEM over the one missing the hyperedge, and both should beat a mis-calibrated model.

```{code-cell} ipython3
dem_true = dem_rep.flattened()
dem_true.append("error", [0.03], [stim.target_relative_detector_id(d) for d in (10, 17, 24)])

det_true, _, _ = dem_true.compile_sampler(seed=2).sample(shots=3000)
counts = counts_from_samples(det_true.astype(np.uint8))
print(f"{len(counts)} distinct syndromes among {sum(counts.values())} shots")

dem_scaled = stim.DetectorErrorModel()
for inst in dem_true:
    if inst.type == "error":
        dem_scaled.append("error", [min(0.5, 1.5 * inst.args_copy()[0])], inst.targets_copy())

models = {"true DEM (with hyperedge)": dem_true,
          "circuit DEM (hyperedge missing)": dem_rep,
          "true DEM, all p x 1.5": dem_scaled}
t0 = time.perf_counter()
lls = {name: tc.log_likelihood(m, counts, backend="numpy") for name, m in models.items()}
print(f"three exact log-likelihoods in {time.perf_counter() - t0:.1f} s\n")
best = max(lls.values())
for name, ll in lls.items():
    print(f"{name:35s} log L = {ll:12.2f}   delta = {ll - best:9.2f}")
```

A difference of several tens of nats on 3000 shots is decisive evidence against the model with the missing hyperedge — exactly the kind of comparison a model-selection or goodness-of-fit routine can build on, with no sampling noise in the likelihood itself.

## Looking at the contraction

With `quimb`/`cotengra` installed, the contraction tree can be inspected and plotted. The "rubber band" picture draws the hypergraph of the network (nodes are tensors, edges are indices) and nests a loop around every intermediate tensor in the order it is formed; for the repetition code the loops sweep along the ladder.

```{code-cell} ipython3
small = stim.Circuit.generated("repetition_code:memory", distance=5, rounds=3,
                               after_clifford_depolarization=0.01).detector_error_model()
if tc.quimb_available():
    tree = tc.contraction_tree(small)
    print(f"contraction width {tree.contraction_width():.0f}, cost {tree.contraction_cost():.2e} flops")
    fig, ax = tc.plot_contraction_tree(tree, kind="rubberband")
    ax.set_title("repetition code d=5, 3 rounds: contraction tree (rubber bands)")
else:
    print("quimb not installed - skipping the contraction-tree plot "
          "(numpy backend width: %d)" % tc.dem_to_tensor_network(small, [0] * small.num_detectors).plan("numpy").width)
```

## Complexity: width, not detector count

The cost of contracting the network is exponential in its *contraction width* — the base-2 logarithm of the largest intermediate tensor along the path, a treewidth-like property of the DEM's detector–event hypergraph — and only linear in the number of tensors. The table below reports the width found by the (cheap, greedy) numpy path finder as codes grow:

```{code-cell} ipython3
def width_of(dem):
    tn = tc.dem_to_tensor_network(dem, np.zeros(dem.num_detectors, dtype=np.uint8))
    return tn.num_tensors, tn.plan(backend="numpy").width

print("repetition code, rounds = distance")
for d in (3, 5, 7, 9, 11):
    dem_d = stim.Circuit.generated("repetition_code:memory", distance=d, rounds=d,
                                   after_clifford_depolarization=0.01).detector_error_model()
    n_t, w = width_of(dem_d)
    print(f"  d={d:2d}: {dem_d.num_detectors:3d} detectors, {n_t:4d} tensors, width {w}")

print("rotated surface code, distance 3")
for r in (1, 2, 3):
    dem_s = stim.Circuit.generated("surface_code:rotated_memory_x", distance=3, rounds=r,
                                   after_clifford_depolarization=0.01).detector_error_model()
    n_t, w = width_of(dem_s)
    print(f"  rounds={r}: {dem_s.num_detectors:3d} detectors, {n_t:4d} tensors, width {w}")
```

A repetition-code DEM is a $(d-1) \times \text{rounds}$ ladder, so the width scales with $\min(d, \text{rounds})$ — about $2d$ here — while the cost grows only linearly with the number of rounds: any number of rounds is affordable, and distances up to $d \approx 11$–$13$ stay within memory. Surface-code DEMs are three-dimensional ($d \times d$ in space, plus time) and their width grows like $d^2$: $d=3$ is comfortable, while $d=5$ already needs a much better path optimiser than greedy search (cotengra's hyper-optimisers with `kahypar`, plus slicing) and quickly becomes infeasible beyond that. The greedy width is an upper bound; `optimize="auto-hq"` on the quimb backend finds narrower paths (12, 17 and 21 instead of 15, 21 and 29 for the $d = 7, 9, 11$ repetition codes above) at the price of a longer search.

**When to use which.**

- `compute_outcome_distribution_from_dem`: you need the *entire* distribution and $n \lesssim 12$–$14$.
- `tensor_contraction`: you need probabilities of specific outcomes, exact log-likelihoods of observed syndromes, or low-order marginals/conditionals — for any $n$, provided the DEM's width is moderate (repetition codes, small surface codes, and generally sparse, locally connected DEMs).

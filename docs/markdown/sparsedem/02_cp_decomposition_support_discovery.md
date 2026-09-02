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

# 02 — Support discovery from CP decompositions of cumulant tensors

A detector error model (DEM) with $R$ independent events says that the vector of detector indicators $Y \in \{0,1\}^n$ is

$$
Y = B Z \pmod 2, \qquad Z_j \sim \mathrm{Bernoulli}(p_j) \text{ independent},
$$

where $B$ is the binary $n \times R$ *event matrix*: column $b_j$ is the signature of event $j$, the set of detectors it flips. Learning a DEM means finding the columns of $B$ (the **support**) and the probabilities $p_j$. This notebook shows how the support falls out of a symmetric tensor decomposition of the *joint cumulants* of the detector bits (`pygsti.extras.sparsedem.cp_decomposition`), why the third-order tensor is needed (the covariance used by the pairwise "$p_{ij}$" method is not enough), and how the resulting pipeline compares with lattice pruning.

## Derivation: cumulant tensors are (leading-order) CP decompositions

Let $\kappa(Y_{d_1}, \dots, Y_{d_k})$ be the joint cumulant of $k$ detector indicators and collect them in the order-$k$ tensor $K_k[d_1, \dots, d_k]$ (repeated indices allowed). For a *single* event, $Y = b_j Z_j$ exactly, so by multilinearity

$$
\kappa(b_{j d_1} Z_j, \dots, b_{j d_k} Z_j) = \Big(\prod_i b_{j d_i}\Big)\, \kappa_k(Z_j), \qquad \kappa_k(Z_j) = p_j + O(p_j^2).
$$

For several events, $Y_d = \bigoplus_j b_{jd} Z_j = \sum_j b_{jd} Z_j - 2\sum_{j<l} b_{jd} b_{ld} Z_j Z_l + \dots$, i.e. the mod-2 sum differs from the *real* sum $BZ$ only by products of two or more independent rare Bernoullis, which are $O(p^2)$. Cumulants are additive over independent summands, so for the real sum $K_k(BZ) = \sum_j \kappa_k(Z_j)\, b_j^{\otimes k}$ exactly, and the mod-2 correction perturbs every entry by $O(p^2)$. Hence, to leading order,

$$
K_k \;\approx\; \sum_{j=1}^{R} p_j \; b_j \otimes b_j \otimes \cdots \otimes b_j \quad (k \text{ copies}),
$$

a **symmetric CP decomposition** of rank $R$ whose factors are the binary event signatures and whose weights are the event probabilities. The statement includes entries with repeated indices: for binary $Y_a$, $(Y_a - \mu_a)^2 = (1 - 2\mu_a) Y_a + \mu_a^2$, so $\kappa(Y_a, Y_a, Y_b) = (1-2\mu_a)\,\mathrm{Cov}(Y_a, Y_b) \approx \sum_{j \ni a, b} p_j$. This is why weight-1 and weight-2 events show up in the order-3 tensor (on its diagonal and "edges"), not just hyperedges.

- $k = 2$ is the covariance matrix used by the $p_{ij}$ method. A symmetric matrix factorization is **not unique**.
- $k = 3$: Kruskal's theorem gives uniqueness of the CP decomposition (up to permutation) when the $k$-ranks of the three identical factor matrices satisfy $3\,k_B \ge 2R + 2$, which holds for generic, sufficiently distinct binary signatures. The recovered factors then *are* the event signatures, hyperedges included.

Everything above is leading order in $p$. The CP fit is used only for **support discovery**; probabilities are refit with `estimation.fit_specified_dem`, which uses the exact product formula.

```{code-cell} ipython3
import time

import matplotlib.pyplot as plt
import numpy as np
import stim

from pygsti.extras.sparsedem import cp_decomposition as cpd
from pygsti.extras.sparsedem.io import dem_from_str, dem_to_dict, dem_to_matrix
from pygsti.extras.sparsedem.lattice import lattice_pruning_dem_estimation
from pygsti.extras.sparsedem.utils import counts_from_samples

np.set_printoptions(precision=5, suppress=True, linewidth=120)


def sample_counts(dem, shots, seed):
    samples = dem.compile_sampler(seed=seed).sample(shots)[0].astype(int)
    return counts_from_samples(samples)  # sparsedem bitstring convention (reversed rows)
```

## Hyperedge versus triangle: why $k=3$

A weight-3 hyperedge $\{D_0 D_1 D_2\}$ with probability $p$ and the triangle of pairs $\{D_0D_1\}, \{D_1D_2\}, \{D_0D_2\}$, each with probability $p$, have the same covariance matrix to leading order.

```{code-cell} ipython3
p = 0.01
hyper = dem_from_str(f"error({p}) D0 D1 D2")
triangle = dem_from_str(f"error({p}) D0 D1\nerror({p}) D1 D2\nerror({p}) D0 D2")

for name, dem in [("hyperedge", hyper), ("triangle", triangle)]:
    cov = cpd.exact_cumulant_tensor_from_dem(dem, order=2)
    k3 = cpd.exact_cumulant_tensor_from_dem(dem, order=3)
    print(f"{name}: covariance matrix\n{cov}")
    print(f"{name}: kappa3(Y0, Y1, Y2) = {k3[0, 1, 2]:+.6f}   (p = {p}, p^2 = {p**2})\n")
```

The off-diagonal covariances differ only at $O(p^2)$ (the diagonals differ because the triangle fires each detector twice as often, but the diagonal of a rank-$R$ symmetric factorization is also matched by a suitably reweighted triangle). The order-3 entry $\kappa_3(Y_0, Y_1, Y_2)$ is $\approx p$ for the hyperedge but $-6p^2 + O(p^3)$ for the triangle: two pair events can never flip an odd number of the three detectors. Running the sampled pipeline at order 3 separates the two supports:

```{code-cell} ipython3
for name, dem in [("hyperedge", hyper), ("triangle", triangle)]:
    counts = sample_counts(dem, 100_000, seed=1)
    est, info = cpd.cp_dem_estimation(counts, order=3, return_info=True)
    labels = [" ".join(f"D{d}" for d in range(3) if mask >> d & 1) for mask in info["masks"]]
    print(f"{name}: recovered supports {labels}")
    print(est)
```

## Leading-order structure and sample cumulants

A small DEM with singles, pairs and a weight-3 hyperedge. We compare the exact population cumulant tensors (from the $2^n$ outcome distribution) with the leading-order model $\sum_j p_j b_j^{\otimes k}$, then with sample cumulants from stim shots.

```{code-cell} ipython3
dem_str = """
error(0.01) D0 D1 D2
error(0.012) D1 D3
error(0.008) D2 D3
error(0.015) D0
error(0.01) D3 D4
error(0.02) D4
"""
dem = dem_from_str(dem_str)
for scale in (1.0, 0.25):
    scaled = dem_from_str("\n".join(
        f"error({float(l.split('(')[1].split(')')[0]) * scale}){l.split(')')[1]}"
        for l in dem_str.strip().splitlines()))
    for k in (2, 3):
        exact = cpd.exact_cumulant_tensor_from_dem(scaled, k)
        lead = cpd.leading_order_cumulant_tensor(scaled, k)
        rel = np.linalg.norm(exact - lead) / np.linalg.norm(lead)
        print(f"p scaled by {scale:<5} order {k}: relative deviation from sum_j p_j b_j^k = {rel:.4f}")
```

The relative deviation is $O(p)$: it shrinks by about the same factor as the probabilities.

```{code-cell} ipython3
shots = 200_000
counts = sample_counts(dem, shots, seed=3)
ct = cpd.cumulant_tensors(counts, order=3)  # orders 2 and 3, with standard errors
exact3 = cpd.exact_cumulant_tensor_from_dem(dem, 3)
z = (ct.tensors[3] - exact3) / np.maximum(ct.stderrs[3], 1 / shots)
print(f"{shots} shots, {ct.num_detectors} detectors: order-3 tensor has {ct.tensors[3].size} entries, "
      f"max |z| vs exact = {np.abs(z).max():.2f}, fraction within 2 SE = {np.mean(np.abs(z) < 2):.2f}")
```

A slice of the sample order-3 tensor, and the CP factor matrix recovered from it next to the true event matrix $B$. The slice $K_3[\,\cdot, \cdot, D_3]$ lights up on the events containing $D_3$: $\{D_1 D_3\}$, $\{D_2 D_3\}$, $\{D_3 D_4\}$ (and the diagonal entry from all of them).

```{code-cell} ipython3
est, info = cpd.cp_dem_estimation(counts, order=3, return_info=True)
B_true, p_true = dem_to_matrix(dem)
B_true = B_true[::-1, :]  # dem_to_matrix rows are MSB-first; flip to row d = detector d
B_rec = cpd.masks_to_factors(info["masks"], info["detectors"])  # rounded, pruned CP factors

fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), gridspec_kw={"width_ratios": [1.1, 1, 1]})
d = 3
im = axes[0].imshow(ct.tensors[3][:, :, d], cmap="Blues", vmin=0)
axes[0].set_title(f"sample $K_3[\\,\\cdot,\\cdot, D_{d}]$ ({shots} shots)")
axes[0].set_xlabel("detector"); axes[0].set_ylabel("detector")
fig.colorbar(im, ax=axes[0], fraction=0.046, label="cumulant")
for ax, M, title in [(axes[1], B_true, "true event matrix $B$"),
                     (axes[2], B_rec, "recovered signatures (CP)")]:
    ax.imshow(M, cmap="Greys", vmin=0, vmax=1, aspect="auto")
    ax.set_title(title); ax.set_xlabel("event"); ax.set_ylabel("detector")
    ax.set_xticks(range(M.shape[1])); ax.set_yticks(range(M.shape[0]))
    ax.set_yticklabels([f"D{i}" for i in range(M.shape[0])])
fig.tight_layout()

truth = dem_to_dict(dem)
print("recovered support == true support:", sorted(info["masks"]) == sorted(truth))
print("CP rank used:", info["rank"], "| structured-init supports:", len(info["init_supports"]))
print(est)
```

The recovered columns are the true signatures up to permutation (the plots order events differently). The CP weights themselves are only leading-order probabilities; the printed DEM comes from the exact refit on the recovered masks.

## The pipeline on a circuit DEM, compared with lattice pruning

`cp_dem_estimation` runs: sample cumulants of orders 2 and 3 with standard errors → structured initialization (nonnegative least squares on the supports suggested by significant entries) → whitened symmetric CP fit with shared factors for both orders (L-BFGS-B, weights $\ge 0$, factors in $[0,1]$) → greedy rank growth with a likelihood-ratio stopping rule → rounding, deduplication, NNLS refit and z-pruning → `fit_specified_dem`, followed by a z-test on the refit probabilities (delta-method covariance) that removes spurious low-weight events the leading-order model can produce from $O(p^2)$ structure, and a final refit.

```{code-cell} ipython3
circuit = stim.Circuit.generated("repetition_code:memory", distance=3, rounds=3,
                                 after_clifford_depolarization=0.02,
                                 before_measure_flip_probability=0.01)
circuit_dem = circuit.detector_error_model(decompose_errors=False).flattened()
truth = dem_to_dict(circuit_dem)
weights = sorted(set(bin(m).count("1") for m in truth))
print(f"{circuit_dem.num_detectors} detectors, {len(truth)} events, event weights present: {weights}")

counts = sample_counts(circuit_dem, 300_000, seed=5)
t0 = time.time()
cp_dem, info = cpd.cp_dem_estimation(counts, order=3, return_info=True)
t_cp = time.time() - t0
t0 = time.time()
lat_dem = lattice_pruning_dem_estimation(counts)
t_lat = time.time() - t0

def summarize(name, est, seconds):
    rec = dem_to_dict(est)
    hit = set(rec) & set(truth)
    rel = [abs(rec[m] - truth[m]) / truth[m] for m in hit]
    print(f"{name:>16}: {len(hit)}/{len(truth)} true events found, {len(set(rec) - set(truth))} spurious, "
          f"median |dp|/p = {np.median(rel):.3f}, max = {np.max(rel):.3f}, {seconds:.1f} s")

summarize("CP (order 3)", cp_dem, t_cp)
summarize("lattice pruning", lat_dem, t_lat)
print("rank path (rank, reduced chi^2):", [(s["rank"], round(s["reduced_chi2"], 1)) for s in info["cp_info"]["path"]])
```

This particular circuit DEM has no hyperedges (a repetition code with depolarizing noise only produces weight-1 and weight-2 detector events), so both methods are being asked for the same graph-like support. Note the reduced $\chi^2$ of the whitened residual is far above 1 even at the correct support: at $3 \times 10^5$ shots the $O(p^2)$ model error, not the sampling noise, dominates the residual. The rank-growth and pruning rules therefore inflate the noise scale by the reduced $\chi^2$ rather than trusting the nominal standard errors.

## Limitations

- **Leading-order approximation.** $K_k = \sum_j p_j b_j^{\otimes k}$ only up to $O(p^2)$ corrections. The CP weights are biased (e.g. weight-1 events see $\kappa_3 = \mu(1-\mu)(1-2\mu)$), and with enough shots the model error exceeds the statistical error, so absolute goodness-of-fit is not meaningful. Use the CP factors for the support and refit probabilities exactly.
- **Memory and time.** The dense order-3 tensor has $m^3$ entries and each CP gradient costs $O(R\, m^3)$; the dense path is practical for $m \lesssim 200$–$300$ detectors. Use `CPConfig(detectors=...)` for subsets. The `screen=True` mode fits only the order-3 entries whose detector pairs all have a significant covariance, which removes noise-only entries from the fit (the dense tensor is still formed).
- **Identifiability.** The continuous relaxation is unique only while $R$ is below the generic symmetric rank ($\approx \binom{m+2}{3}/m$); the binary constraint, the order-2 coupling and the structured initialization extend recovery beyond that (the circuit example above has $R = 21 > 15$), but very dense DEMs on few detectors are the unfavourable case.
- **Rank selection and noise.** Rank growth stops by a likelihood-ratio test with an inflated noise scale; weak events whose entries are not significant (roughly $\sqrt{N p} \lesssim 4$–$5$) are neither proposed by the structured initialization nor survive the z-pruning, and would need more shots.

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

# Detector Error Model Estimation with `pygsti.extras.sparsedem`

A **detector error model (DEM)** is a compact stochastic description of the errors seen in a
quantum error correction experiment. It is a list of independent *events*: each event fires
with some probability $p$ and, when it fires, flips a fixed subset of *detectors* (parity
checks) — and possibly a logical observable. The data structure used throughout this package
is stim's `stim.DetectorErrorModel`, e.g.

```
error(0.01) D0 D1
error(0.02) D3 L0
```

**DEM estimation** is the inverse problem: given *only* the observed detector data (many shots
of detector bitstrings, with no circuit-level noise model), infer which events exist and with
what probabilities. The methods in `pygsti.extras.sparsedem` exploit the fact that physical
DEMs are *sparse*: out of the $2^n - 1$ possible events on $n$ detectors, only a modest number
(typically low-weight ones) actually occur with non-negligible probability. The estimation
algorithms are described in [arXiv:2504.14643](https://arxiv.org/abs/2504.14643).

This notebook is the main tutorial. A companion notebook,
[`DEMEstimation-LassoAndCompressedSensing.ipynb`](DEMEstimation-LassoAndCompressedSensing.ipynb),
covers the alternative sparse-recovery methods (lasso model selection, compressed sensing,
and fitting a known support).

| module | contents |
|---|---|
| `core` | `SparseDEMEstimator` — the high-level entry point wrapping all estimators |
| `io` | converters between stim DEMs, dicts, matrices, and probability vectors |
| `utils` | bit conventions, syndrome-count wrangling, polarizations |
| `estimation` | dense Hadamard-transform estimation, covariances, thresholding, support refits |
| `lattice` | the lattice-pruning (subset-trie) sparse estimator — the recommended default |
| `model_selection` | non-negative lasso event selection (companion notebook) |
| `compressed_sensing` | randomized low-weight polarization sketch (companion notebook) |
| `logical_decoration` | decorate a learned DEM with logical-flip flags from decoded shots |

+++

## 1. Data representation and conventions

The estimators consume **syndrome counts**: a dictionary mapping detector bitstrings to the
number of shots on which that bitstring was observed. Two conventions matter everywhere in
this package:

* **Reversed bitstrings.** Stim samples detectors in increasing index order, but sparsedem
  keys syndromes by the *reversed* string, so that the string reads as a binary number whose
  bit $d$ (counting from the least-significant end) is detector $d$. The stim sample row
  `[1, 1, 0, 0, 1]` (detectors 0, 1, 4 fired) becomes the key `'10011'`.
* **Integer bitmasks.** Events are labeled by the integer whose binary representation marks the
  flipped detectors: the event `D0 D1 D4` has mask $2^0 + 2^1 + 2^4 = 19$. Converters in
  `sparsedem.io` move between stim DEMs, `{mask: probability}` dicts, check matrices, and
  dense $2^n$ probability vectors.

`utils.counts_from_samples` applies the reversal for you when starting from a stim-style
sample array.

```{code-cell} ipython3
import time
import warnings

import numpy as np
import stim

from pygsti.extras.sparsedem.core import SparseDEMEstimator
from pygsti.extras.sparsedem import io as sdio
from pygsti.extras.sparsedem import utils as sdutils
from pygsti.extras.sparsedem.logical_decoration import (
    assign_logical_flags,
    dem_to_check_matrix,
    solve_gf2_robust,
)

SEED = 2026
np.set_printoptions(precision=4, suppress=True)
```

```{code-cell} ipython3
# A small DEM to exercise the converters.
demo_dem = sdio.dem_from_str("""
error(0.01) D0
error(0.02) D0 D2
error(0.03) D1 D2
""")

print("as a stim DEM:")
print(demo_dem)
print("as {mask: probability}:", sdio.dem_to_dict(demo_dem))

# Sample it and build syndrome counts. Note the reversed keys: the stim data
# [0, 1, 1] (D1 and D2 fired) becomes the key '110', which reads as mask 6.
samples = np.array(demo_dem.compile_sampler(seed=SEED).sample(8)[0], dtype=int)
print("\nstim sample rows (columns are D0, D1, D2):")
print(samples)
print("syndrome counts:", dict(sdutils.counts_from_samples(samples)))
```

## 2. The running example

One example threads through this notebook: **10 detectors, 20 events**, all *graph-like*
(each event flips at most two detectors, so a matching decoder applies later):

* 10 **boundary events** `D0`, ..., `D9` — single-detector events, probabilities 0.008–0.017;
* 10 **pair events** forming a ring `D0 D1`, `D1 D2`, ..., `D8 D9`, `D9 D0` — probabilities
  0.010–0.019.

Picture a ring of 10 nodes (one per detector) with a spoke from every node to a common
boundary node; each event is one edge. We also decorate the ground truth with a **logical
observable**: the flags are the edges crossing the cut between detectors $\{0,\dots,4\}$ and
$\{5,\dots,9\}\cup\{\text{boundary}\}$ — the five boundary events `D0`–`D4` plus the two ring
edges `D4 D5` and `D9 D0`. Cut-based flags are how a real code's logical observable behaves:
any two error sets with the same syndrome differ by a cycle, and a cycle crosses a cut an even
number of times, so homologically equivalent corrections predict the same logical outcome.

Each shot is simulated by firing every event independently, XORing the detector patterns of
the fired events into a syndrome, and XORing their flags into the true logical outcome $y$.

```{code-cell} ipython3
N_DETECTORS = 10


def build_ground_truth():
    """20 distinct graph-like events with probabilities and logical flags."""
    masks, probs, flags = [], [], []
    for d in range(N_DETECTORS):                      # boundary events
        masks.append(1 << d)
        probs.append(0.008 + 0.001 * d)
        flags.append(1 if d < 5 else 0)               # cut: {D0..D4} side
    for d in range(N_DETECTORS):                      # ring events
        d2 = (d + 1) % N_DETECTORS
        masks.append((1 << d) | (1 << d2))
        probs.append(0.010 + 0.001 * d)
        flags.append(1 if d in (4, 9) else 0)         # ring edges crossing the cut
    order = np.argsort(masks)
    return (np.array(masks)[order], np.array(probs)[order],
            np.array(flags, dtype=np.uint8)[order])


def simulate(masks, probs, flags, n_shots, rng):
    """Fire events independently; XOR patterns into syndromes, flags into y."""

    H = np.zeros((N_DETECTORS, len(masks)), dtype=np.uint8)
    for j, mask in enumerate(masks):
        for d in range(N_DETECTORS):
            H[d, j] = (int(mask) >> d) & 1
    occurrences = (rng.random((n_shots, len(masks))) < probs).astype(np.uint8)
    syndromes = ((occurrences @ H.T) % 2).astype(np.uint8)
    y = ((occurrences @ flags.astype(np.int64)) % 2).astype(np.uint8)
    return syndromes, y


def mask_to_str(mask):
    return " ".join(f"D{d}" for d in range(N_DETECTORS) if (mask >> d) & 1)


rng = np.random.default_rng(SEED)
true_masks, true_probs, true_flags = build_ground_truth()
true_prob_of = dict(zip(true_masks.tolist(), true_probs.tolist()))
true_flag_of = dict(zip(true_masks.tolist(), true_flags.tolist()))

N_SHOTS = 4000
syndromes, y = simulate(true_masks, true_probs, true_flags, N_SHOTS, rng)
syndrome_counts = dict(sdutils.counts_from_samples(syndromes))

print(f"{len(true_masks)} events on {N_DETECTORS} detectors, {N_SHOTS} shots")
print(f"mean detections per shot:   {syndromes.sum(axis=1).mean():.3f}")
print(f"observed logical-flip rate: {y.mean():.4f}")
print(f"distinct syndromes seen:    {len(syndrome_counts)} of {2**N_DETECTORS} possible")
```

## 3. Dense estimation and thresholding

The syndrome distribution of a DEM factorizes in the polarization (Walsh–Hadamard) domain:
for any parity mask $m$, the polarization $\langle(-1)^{m\cdot s}\rangle$ is a product over
events, each contributing a factor $(1-2p)$ if it anticommutes with $m$. Taking logs turns
this into a linear system, so a Hadamard transform of the log-polarizations recovers *all*
$2^n$ event probabilities at once. `estimate_dense_covariance` additionally propagates the
multinomial sampling covariance through the transform, and `threshold` applies a
Bonferroni-corrected one-sided $z$-test to zero out estimates consistent with zero.

This is exact and simple, but it manipulates $2^n$-dimensional vectors (and $2^n \times 2^n$
covariances)This is practical up to approximately $n = 20$. This exponential scaling is what motivates the
sparse methods below.

```{code-cell} ipython3
estimator = SparseDEMEstimator(syndrome_counts)

t0 = time.perf_counter()
dense_dem, dense_cov = estimator.estimate_dense_covariance()
t_dense = time.perf_counter() - t0
dense_probs = estimator.get_dense_probabilities()

# The raw dense estimate is noisy: sampling fluctuations give small nonzero
# values to masks that correspond to no real event.
print(f"dense estimate + covariance in {t_dense:.2f} s")
print(f"entries of the 2^{N_DETECTORS} vector above 1e-3: "
      f"{int((dense_probs > 1e-3).sum())} (true event count: {len(true_masks)})")

thresholded_dem = estimator.threshold(alpha=0.05)
thr = sdio.dem_to_dict(thresholded_dem)
true_set = set(true_masks.tolist())
print(f"after z-test thresholding: {len(thr)} events "
      f"({len(set(thr) & true_set)} true, {len(set(thr) - true_set)} spurious, "
      f"{len(true_set - set(thr))} missed)")
```

## 4. Lattice pruning — the recommended sparse estimator

`estimate_lattice_pruned` avoids the $2^n$ blow-up by searching the *subset lattice* of
detectors with statistical pruning (the algorithm of arXiv:2504.14643). For a candidate
detector subset $S$:

1. **Marginalize** the syndrome counts onto the detectors in $S$.
2. On that small marginal problem, estimate the probability of the event that flips *all* of
   $S$ together, along with its standard error.
3. Keep $S$ only if a one-sided $z$-test says the probability is significantly positive.

The search walks a trie over bitmasks depth-first, only extending prefixes that pass the test
— so subsets containing no correlated flipping are pruned along with their entire subtree, and
the work scales with the number of real events rather than with $2^n$. The `confidence`
parameter sets the per-test confidence level: higher values admit fewer spurious events but
need more shots to certify weak ones. Finally the surviving masks are refit jointly
(`fit_specified_dem`) for unbiased probabilities and a covariance.

```{code-cell} ipython3
t0 = time.perf_counter()
learned_dem, learned_masks, learned_probs, learned_cov = estimator.estimate_lattice_pruned(
    confidence=0.999, return_covariance=True)
t_lattice = time.perf_counter() - t0

learned = sdio.dem_to_dict(learned_dem)
found = [m for m in true_masks if m in learned]
missed = [m for m in true_masks if m not in learned]
spurious = [m for m in learned if m not in true_set]

print(f"lattice pruning in {t_lattice:.2f} s: {len(learned)} events "
      f"({len(found)}/{len(true_masks)} true, {len(spurious)} spurious, {len(missed)} missed)")
print(f"\n{'event':10s} {'p_true':>8s} {'p_learned':>10s} {'std err':>8s}")
mask_index = {int(m): i for i, m in enumerate(learned_masks)}
for m in sorted(learned):
    se = np.sqrt(learned_cov[mask_index[m], mask_index[m]])
    p_true = true_prob_of.get(m, float('nan'))
    print(f"{mask_to_str(m):10s} {p_true:8.4f} {learned[m]:10.4f} {se:8.4f}")
print(f"\nmax |p_learned - p_true| over found events: "
      f"{max(abs(learned[m] - true_prob_of[m]) for m in found):.4f}")
```

### Shot count 

The $z$-score of an event with probability $p$ after $N$ shots scales like $\sqrt{pN}$, so an
event needs roughly $N \gtrsim z^2/p$ shots to clear the detection threshold. At
`confidence=0.999` the threshold is $z \approx 3.1$, so our weakest events
($p \approx 0.008$–$0.012$) need a few thousand shots. Rerunning on only the first 1000 shots
of the same data shows the weakest events dropping out — the estimator does not invent them,
it simply (and correctly) reports no significant evidence.

```{code-cell} ipython3
counts_1000 = dict(sdutils.counts_from_samples(syndromes[:1000]))
dem_1000 = SparseDEMEstimator(counts_1000).estimate_lattice_pruned(confidence=0.999)
learned_1000 = sdio.dem_to_dict(dem_1000)

for n_shots, d in ((1000, learned_1000), (N_SHOTS, learned)):
    f = set(d) & true_set
    print(f"{n_shots:5d} shots: {len(f)}/{len(true_set)} true events found, "
          f"{len(set(d) - true_set)} spurious")
miss_1000 = sorted(true_set - set(learned_1000))
print("missed at 1000 shots:",
      ", ".join(f"{mask_to_str(m)} (p={true_prob_of[m]:.3f})" for m in miss_1000))
```

## 5. Decorating the learned DEM with logical flags

The learned DEM says which detector patterns occur, but nothing about the **logical** degree
of freedom, because syndrome data alone cannot see it. If each shot also comes with an
observed logical outcome $y \in \{0, 1\}$ (as in a memory experiment where the final logical
measurement is compared to the ideal one), `logical_decoration.assign_logical_flags` can
recover, for every learned event, a binary flag saying whether that event flips the logical
observable:

1. Build a decoder from the *learned* DEM (pymatching by default).
2. Decode every shot: the decoder's correction is a set of learned events, i.e. a binary
   indicator row $b$ over the events. Stack the shots into a matrix $B$.
3. Solve $Y = B L \pmod 2$ for the flag vector $L$.

In the low-logical-error regime the system is massively overdetermined and *almost*
consistent: a row can only be inconsistent when the decoder's correction differs from the
truth by a logical operator (an actual decoder logical error) — mere homological differences
cancel, as discussed in section 2. The solver (`solve_gf2_robust`) therefore does GF(2)
Gaussian elimination, checks the fraction of violated rows, and falls back to a RANSAC-style
loop over row orderings if the naive elimination may have pivoted on a corrupted row. Flags
the data cannot determine (events the decoder never selects, or null-space directions) are
reported explicitly rather than silently guessed.

```{code-cell} ipython3
decorated_dem, flags, residual, diag = assign_logical_flags(
    learned_dem, syndromes, y, num_detectors=N_DETECTORS, seed=SEED)

print(f"residual (inconsistent-row) fraction: {residual:.5f}")
print(f"solve method: {diag['method']}  |  rank(B) = {diag['rank']} of "
      f"{diag['n_events']} events  |  converged = {diag['converged']}")
print(f"undetermined flags: {diag['undetermined_masks'].tolist() or 'none'}")

flag_of = dict(zip(diag['masks'].tolist(), flags.tolist()))
n_ok = sum(flag_of[m] == true_flag_of[m] for m in found)
print(f"\n{'event':10s} {'flag_true':>9s} {'flag_recovered':>14s}")
for m in sorted(learned):
    truth = true_flag_of.get(m)
    marker = '' if truth is None or flag_of[m] == truth else '   <-- MISMATCH'
    shown = '--' if truth is None else truth
    print(f"{mask_to_str(m):10s} {str(shown):>9s} {flag_of[m]:>14d}{marker}")
print(f"\nflags correct on correctly-learned events: {n_ok}/{len(found)}")
print("\ndecorated DEM (first 6 instructions):")
print("\n".join(str(decorated_dem).splitlines()[:6]))
```

### Reading the diagnostics, and what "robust" means

The diagnostics dict is the audit trail of the solve:

* `initial_residual_fraction` / `method` — if plain Gaussian elimination already explains every
  shot (`residual 0`, `method 'gaussian_elimination'`), RANSAC never engages. A nonzero
  residual triggers up to `max_ransac_iterations` re-solves from random row orderings, keeping
  the flag vector that violates the fewest shots.
* `rank`, `undetermined_indices`, `null_space` — if $\mathrm{rank}(B)$ is below the number of
  events, some flags are not determined by the data (e.g. an event so weak the decoder never
  used it). They are reported (and returned as 0) rather than guessed; `undetermined_masks`
  gives them as event bitmasks.
* `converged` — whether the final residual is below `residual_threshold`. Set the threshold to
  the decoder logical-error rate you expect; a residual far above it means the learned DEM or
  the decoder disagrees with the data more than decoder noise can explain.

On our clean example the residual is exactly zero, so the cell below manufactures a corrupted
system to show the robust path: a random consistent system with four flipped outcomes, two of
them planted in the rows naive elimination pivots on. RANSAC recovers the true flags and the
residual equals exactly the fraction of corrupted rows.

```{code-cell} ipython3
rng_demo = np.random.default_rng(7)
n_ev = 12
L_true_demo = rng_demo.integers(0, 2, size=n_ev).astype(np.uint8)
B_demo = (rng_demo.random((600, n_ev)) < 0.3).astype(np.uint8)
B_demo[:n_ev] = np.eye(n_ev, dtype=np.uint8)          # ensure full rank
Y_demo = ((B_demo @ L_true_demo) % 2).astype(np.uint8)
Y_demo[[0, 1, 300, 450]] ^= 1                          # 4 corrupted shots (2 are pivots)

L_hat, res_demo, d_demo = solve_gf2_robust(
    B_demo, Y_demo, residual_threshold=0.01, seed=SEED)
print(f"method: {d_demo['method']}  (initial residual "
      f"{d_demo['initial_residual_fraction']:.4f}, {d_demo['ransac_iterations']} "
      f"RANSAC iterations)")
print(f"final residual: {res_demo:.4f}  (= 4/600 corrupted rows)")
print(f"flags recovered exactly: {bool(np.array_equal(L_hat, L_true_demo))}")
```

## 6. Decoder backends: pymatching vs tesseract

`assign_logical_flags` (and `build_decoder`) accept a `decoder=` argument:

* **`'pymatching'`** (default) — minimum-weight perfect matching. Fast and batched, but it
  requires a *graph-like* DEM: every event may flip at most two detectors. Events with three or
  more detectors are dropped from decoding (with a warning; pass `on_nongraphlike='raise'` to
  forbid this). Dropped events are left undecorated and show up in
  `diagnostics['dropped_masks']` and `diagnostics['undetermined_masks']`.
* **`'tesseract'`** — [Google's Tesseract decoder](https://github.com/quantumlib/tesseract-decoder),
  a most-likely-error decoder that handles **hyperedges** natively, so nothing is dropped. It
  decodes shot-by-shot through the Python API, so expect it to be roughly an order of
  magnitude slower than pymatching's batched C++ decoding (timed below) — still fast in
  absolute terms at these problem sizes.

> **Install note:** `pip install tesseract-decoder` works where PyPI wheels exist (x86-64
> Linux, arm64 macOS). On aarch64 Linux there is no wheel or sdist and the package must be
> built from source — see the *Tesseract decoder* section of this workspace's `CLAUDE.md`
> (`/workspaces/dems/CLAUDE.md`) for the working bazel recipe. All tesseract cells below are
> skipped gracefully when the package is absent.

```{code-cell} ipython3
try:
    import tesseract_decoder  # noqa: F401
    TESSERACT_AVAILABLE = True
    print("tesseract-decoder is available - tesseract cells will run.")
except ImportError:
    TESSERACT_AVAILABLE = False
    print("tesseract-decoder is NOT installed - tesseract cells will be skipped.")
```

```{code-cell} ipython3
# A DEM with a 3-detector hyperedge event that carries the logical flag.
hyper_dem = sdio.dem_from_str("""
error(0.005) D0
error(0.005) D1
error(0.005) D2
error(0.02) D0 D1 D2
""")
H_h, probs_h, masks_h = dem_to_check_matrix(hyper_dem)
flags_h = np.array([0, 0, 0, 1], dtype=np.uint8)          # flag on D0 D1 D2

rng_h = np.random.default_rng(11)
occ_h = (rng_h.random((3000, 4)) < probs_h).astype(np.uint8)
synd_h = ((occ_h @ H_h.T) % 2).astype(np.uint8)
y_h = ((occ_h @ flags_h) % 2).astype(np.uint8)

# pymatching: the hyperedge is dropped, its flag cannot be recovered.
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    _, flags_pm, res_pm, diag_pm = assign_logical_flags(
        hyper_dem, synd_h, y_h, decoder="pymatching", seed=SEED,
        max_ransac_iterations=20)
print("pymatching:")
for w in caught:
    print(f"  warning: {str(w.message).splitlines()[0]}")
print(f"  dropped events: {[mask_to_str(m) for m in diag_pm['dropped_masks']]}, "
      f"residual {res_pm:.4f}")

if TESSERACT_AVAILABLE:
    _, flags_ts, res_ts, diag_ts = assign_logical_flags(
        hyper_dem, synd_h, y_h, decoder="tesseract", seed=SEED)
    print("tesseract:")
    print(f"  dropped events: {diag_ts['dropped_masks'].tolist()}, residual {res_ts:.4f}")
    print(f"  recovered flags {flags_ts.tolist()} vs truth {flags_h.tolist()} -> "
          f"{'match' if np.array_equal(flags_ts, flags_h) else 'MISMATCH'} "
          f"(hyperedge flag recovered)")
else:
    print("tesseract: skipped (tesseract-decoder not installed)")
```

```{code-cell} ipython3
# Backend timing on the running example (decode 4000 shots + GF(2) solve).
t0 = time.perf_counter()
_, flags_pm2, res_pm2, _ = assign_logical_flags(
    learned_dem, syndromes, y, num_detectors=N_DETECTORS,
    decoder="pymatching", seed=SEED)
t_pm = time.perf_counter() - t0
print(f"pymatching: {t_pm:.3f} s, residual {res_pm2:.5f}, "
      f"flags correct: {sum(flags_pm2[list(diag['masks']).index(m)] == true_flag_of[m] for m in found)}/{len(found)}")

if TESSERACT_AVAILABLE:
    t0 = time.perf_counter()
    _, flags_ts2, res_ts2, diag_ts2 = assign_logical_flags(
        learned_dem, syndromes, y, num_detectors=N_DETECTORS,
        decoder="tesseract", seed=SEED)
    t_ts = time.perf_counter() - t0
    print(f"tesseract:  {t_ts:.3f} s, residual {res_ts2:.5f}, "
          f"flags correct: {sum(flags_ts2[list(diag_ts2['masks']).index(m)] == true_flag_of[m] for m in found)}/{len(found)}"
          f"  ({t_ts / t_pm:.0f}x slower than pymatching)")
else:
    print("tesseract:  skipped (tesseract-decoder not installed)")
```

## 7. Practical guidance

* **Shot counts.** An event of probability $p$ needs roughly $N \gtrsim z^2/p$ shots to be
  detected at $z$-score threshold $z$ (about $10/p$ at `confidence=0.999`). Estimate your
  weakest event of interest and budget shots accordingly; section 4 showed exactly this
  failure mode at 1000 shots.
* **Confidence trades false positives for sensitivity.** Lower `confidence` finds weaker events
  but admits spurious ones; spurious weight-3+ events are also what forces the pymatching path
  to drop things later, so with matching decoding in mind prefer higher confidence.
* **The GF(2) solve assumes rare logical errors.** Every row of $Y = BL$ is one shot; rows are
  violated only by decoder *logical* errors (or a badly wrong learned DEM). If your operating
  point has a high logical error rate, the residual grows and flag recovery degrades — RANSAC
  tolerates a small bad fraction, not 30%.
* **Residual 0 does not require a perfect decoder.** Corrections that differ from the true
  error by a *cycle* (a homologically trivial deformation) predict the same logical outcome,
  so they cost nothing. Even shots whose true event was never learned can be consistent, as
  long as the substitute correction crosses the logical cut the same way.
* **Undetermined flags are information.** `undetermined_masks` lists events whose flag the data
  simply cannot fix — usually events the decoder never selected (too weak, or shadowed by
  cheaper explanations). Collect more shots, or accept that those events are irrelevant to the
  decoder's logical predictions at this noise level.
* **Check `diagnostics` before trusting `flags`.** `converged=True`, full rank, and an empty
  `undetermined_masks` is the clean bill of health; anything else tells you which events to be
  suspicious of.

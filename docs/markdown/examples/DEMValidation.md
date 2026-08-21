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

# Validating Detector Error Models with `pygsti.extras.sparsedem.validation`

The [DEM estimation tutorial](DEMEstimation.ipynb) showed how to *learn* a detector error
model (DEM) from detector data. This notebook is about the complementary question: **given a
candidate DEM and detector data, are the data statistically consistent with the model?** The
candidate might be a learned DEM, a circuit-level noise model's prediction
(`circuit.detector_error_model()`), or a hand-built ansatz — validation does not care where it
came from.

There is a fundamental obstruction to answering this with a single test: the alternative
hypothesis ("the data came from *some other* distribution") is astronomically large, and **no
uniformly most powerful test exists**. A test that is exquisitely sensitive to a doubled
error rate on one stabilizer can be completely blind to an unmodeled long-range correlation,
and vice versa. `sparsedem.validation` therefore provides a *battery* of tests with different
power profiles:

* **exact-marginal likelihood ($G$-) tests** on detector subsets, with a workflow for building
  the subsets (random, all-weight-$k$, detector-graph neighborhoods, spacetime structures from
  a stim circuit);
* **moment/spectrum tests** comparing observed Walsh polarizations (click rates, pairwise and
  higher-order correlators) against closed-form model predictions;
* **distribution tests on scalar functions of the syndrome** (Hamming weight, decoder matching
  weight, complementary gap, or anything you write yourself), calibrated by DEM Monte Carlo;
* a **decoder-based logical-error-rate consistency test**;
* **stationarity tests** of the i.i.d.-shots assumption (these test the *data collection*, not
  the DEM, and are labeled accordingly).

Two design principles run through everything:

1. **Every test returns a p-value *and* an effect-size diagnostic.** A p-value alone tells you
   "something is off" but not *what*; each `ValidationResult` carries an `effect_description`
   that points at the most significant deviation — which detectors, which outcome, how many
   sigma, and in which direction. At the end of a validation run you should know not just
   *whether* to reject the model but *what to fix*.
2. **Suites are aggregated with multiple-testing control.** A battery of 700 tests at
   $\alpha = 0.05$ would produce ~35 false alarms on perfect data. `ValidationSuiteResult`
   applies Benjamini–Hochberg FDR control (or Holm / Bonferroni) so that "0 rejections" is a
   meaningful clean bill of health.

```{code-cell} ipython3
import numpy as np
import stim

from pygsti.extras.sparsedem import validation as val
from pygsti.extras.sparsedem.io import dem_to_dict
from pygsti.extras.sparsedem.validation import (
    ValidationSuiteResult,
    # marginal workflow
    build_marginal_subsets, run_marginal_tests, marginal_likelihood_test,
    detector_graph,
    # moment / polarization tests
    run_polarization_battery, polarization_tests, predicted_polarizations,
    # scalar-function engine
    hamming_weight_test, scalar_distribution_test,
    matching_weight_test, complementary_gap_test,
    # decoder-based logical error rate
    logical_error_rate_test,
    # stationarity
    run_stationarity_battery, shot_autocorrelation_test,
    # sampling
    sample_dem,
)

SEED = 2026
np.set_printoptions(precision=4, suppress=True)
```

## 1. The running example: a distance-3 surface code

Our candidate DEM comes from a real circuit: three rounds of a rotated distance-3 surface-code
memory experiment, with uniform depolarizing/flip noise. Stim converts the circuit into its
exact detector error model, which is the *truth* in this notebook — we will sample data from
it, and later perturb either the candidate or the truth to manufacture model violations.

One representation detail worth knowing: stim's `decompose_errors=True` writes some error
mechanisms as suggested decompositions (`D0 D1 ^ D2`). The sparsedem event representation
(`io.dem_to_dict`) merges the components of each mechanism back into a single event whose
detector set is the XOR of the components. The induced distribution over detector data is
identical, but the merged events can flip three or four detectors — the DEM is *not*
graph-like in this representation, which will matter for the decoder-based tests in section 5.

```{code-cell} ipython3
circuit = stim.Circuit.generated(
    "surface_code:rotated_memory_z", distance=3, rounds=3,
    after_clifford_depolarization=0.008,
    before_round_data_depolarization=0.008,
    before_measure_flip_probability=0.008,
    after_reset_flip_probability=0.008,
)
dem = circuit.detector_error_model(decompose_errors=True)
events = dem_to_dict(dem)
weights = np.bincount([bin(m).count("1") for m in events])

N_SHOTS = 20_000
det, obs = sample_dem(dem, N_SHOTS, seed=SEED)

print(f"{dem.num_detectors} detectors, {dem.num_observables} logical observable(s)")
print(f"{len(events)} distinct error events; count by weight 1..4: {weights[1:]}")
print(f"{N_SHOTS} shots; mean clicks per shot {det.sum(axis=1).mean():.3f}; "
      f"logical-flip rate {obs.mean():.4f}")
```

### 1.1 Anatomy of a result

Every test returns a `ValidationResult`. The fields to read, in order: `pvalue` (is the data
consistent?), `effect_size` (how big is the deviation, on a documented scale — total variation
distance for marginal tests, a z-score for moment tests), and `effect_description` (*where* is
the worst disagreement). `details` holds test-specific diagnostics (per-cell counts,
residuals, which statistical branch was taken).

Here is a single marginal likelihood test on three detectors. It computes the exact
$2^3$-outcome model marginal, histograms the observed shots over the same outcomes, and
compares them with a likelihood-ratio $G$-test (small expected cells are pooled so the
chi-squared reference is trustworthy).

```{code-cell} ipython3
r = marginal_likelihood_test(dem, det, subset=(0, 4, 8))
print(f"name:               {r.name}")
print(f"pvalue:             {r.pvalue:.3f}")
print(f"effect_size (TVD):  {r.effect_size:.4f}")
print(f"effect_description: {r.effect_description}")
print(f"null_model:         {r.null_model!r}   (\"dem\" = tests the model, "
      f"\"iid\" = tests the data)")
```

### 1.2 A first battery run: on-model data passes

Batteries are collected in a `ValidationSuiteResult`, whose `summary()` prints a table sorted
by adjusted p-value and whose `rejected(alpha, method)` returns the tests that survive
multiple-testing correction (`"fdr_bh"` by default; `"holm"` and `"bonferroni"` are available
for family-wise control). We bundle a default battery — radius-1 neighborhood marginals, the
full polarization battery (click rates, all pairs, event-aligned masks, connected triples),
and the Hamming-weight distribution test — into a helper we will reuse on every scenario.

On data sampled from the candidate itself, the battery should reject (essentially) nothing
after FDR control. This is the calibration check: a battery that fires on its own model is
useless.

```{code-cell} ipython3
NBHD_SUBSETS = build_marginal_subsets("neighborhood", dem=dem, radius=1)


def full_battery(candidate, samples, seed=0):
    """Neighborhood marginals + polarization battery + Hamming weight."""
    results = list(run_marginal_tests(candidate, samples, NBHD_SUBSETS).results)
    results += run_polarization_battery(candidate, samples, seed=seed).results
    results.append(hamming_weight_test(candidate, samples, seed=seed))
    return ValidationSuiteResult(results)


suite = full_battery(dem, det)
print(f"{len(suite.results)} tests, {len(suite.rejected(alpha=0.05))} rejected "
      f"at alpha=0.05 after FDR (Benjamini-Hochberg)")
print()
print(suite.summary(max_rows=4))
```

Note what the summary shows even on clean data: the *smallest raw p-values* in a 700-test
battery are naturally in the $10^{-3}$ range — that is exactly what uniform p-values do — and
the adjusted `p_adj` column correctly declines to reject any of them. Never read raw p-values
off a battery without correction.

+++

## 2. Building marginal subsets: seven ways to slice the detectors

The marginal likelihood test is exact and assumption-free on any detector subset of size
$k \le 20$, but its cost (and its dilution of power) scales as $2^k$ — and *which* subsets you
test determines *which* violations you can see. `build_marginal_subsets` is the front door to
seven subset-construction methods, each with a different power profile.

**Structure-free methods** need only the number of detectors:

* `all_weight_k` — every $\binom{n}{k}$ subset of size $k$ (or a seeded uniform sample above
  `max_subsets`). Exhaustive coverage at low order.
* `random` — random subsets with sizes in `[min_size, k]`. Cheap broad-spectrum screening.

```{code-cell} ipython3
pairs = build_marginal_subsets("all_weight_k", num_detectors=dem.num_detectors, k=2)
rand = build_marginal_subsets("random", num_detectors=dem.num_detectors,
                              k=4, num_subsets=30, seed=SEED)
print(f"all_weight_k(k=2): {len(pairs)} subsets, e.g. {pairs[:3]}")
print(f"random(k<=4):      {len(rand)} subsets, e.g. {rand[:3]}")
```

**Detector-graph methods** use the candidate DEM itself. The *detector graph* connects two
detectors iff they co-occur in some DEM event — it is the geometry the model claims the noise
has.

* `neighborhood` — BFS balls of radius $r$ around each detector. A locally misestimated event
  probability distorts the joint distribution of exactly these detectors, so neighborhood
  subsets are powered against *local* model errors, and a firing neighborhood names the
  region.
* `distant` — sets of detectors pairwise *far apart* in the graph. Under the model these
  detectors are nearly independent, so their joint marginal is almost a product distribution —
  a *sharp null*. Any unmodeled long-range correlation shows up as a violent deviation from
  independence. (Anti-structured subsets like these are the mirror image of neighborhoods:
  quiet where the model has structure, loud where it claims there is none.)

```{code-cell} ipython3
adjacency = detector_graph(dem)
degrees = np.array([len(adjacency[d]) for d in sorted(adjacency)])
print(f"detector graph: degree min/median/max = "
      f"{degrees.min()}/{int(np.median(degrees))}/{degrees.max()}")
print(f"neighborhood(radius=1): {len(NBHD_SUBSETS)} subsets, sizes "
      f"{sorted(set(len(s) for s in NBHD_SUBSETS))}")

DISTANT_SUBSETS = build_marginal_subsets("distant", dem=dem, size=4,
                                         num_subsets=4, min_distance=3, seed=7)
print(f"distant(size=4, min_distance=3): {DISTANT_SUBSETS}")
```

The warning above is informative, not a failure: this 24-detector graph is so dense (merged
weight-3/4 events add many edges) that only one 4-element set of pairwise-distance-$\ge 3$
detectors exists — the four corners of spacetime, detectors from the first and last rounds on
opposite sides of the patch. On larger patches and more rounds the builder finds many.

**Circuit-based spacetime methods** use the detector *coordinates* from the stim circuit —
DEMs carry no geometry, so these methods require the circuit:

* `time` — "columns": all detectors sharing the same spatial coordinates, across rounds
  (optionally a sliding `window` of rounds). One column is one stabilizer watched over time,
  so these subsets are powered against *time-correlated* deviations: drift, round-dependent
  miscalibration, anything that treats early and late rounds differently.
* `space` — "slices": all detectors within a window of consecutive rounds. One slice is a
  snapshot of the whole patch, powered against *spatially* structured deviations (a bad corner
  of the chip, crosstalk between neighbors) while averaging over time.
* `spacetime` — Euclidean balls in space $\times$ windows in time, the local-in-both option.

```{code-cell} ipython3
coords = circuit.get_detector_coordinates()
print("detector coordinates (x, y, t):",
      {d: coords[d] for d in [0, 1, 8, 20]})

TIME_SUBSETS = build_marginal_subsets("time", circuit=circuit)
SPACE_SUBSETS = build_marginal_subsets("space", circuit=circuit)
ST_SUBSETS = build_marginal_subsets("spacetime", circuit=circuit,
                                    space_radius=2, time_radius=1)
print(f"time columns:    {len(TIME_SUBSETS)} subsets, sizes "
      f"{sorted(set(len(s) for s in TIME_SUBSETS))}  e.g. {TIME_SUBSETS[0]}")
print(f"space slices:    {len(SPACE_SUBSETS)} subsets, sizes "
      f"{sorted(set(len(s) for s in SPACE_SUBSETS))}  e.g. {SPACE_SUBSETS[0]}")
print(f"spacetime balls: {len(ST_SUBSETS)} subsets, sizes "
      f"{sorted(set(len(s) for s in ST_SUBSETS))}")
```

(The two column sizes reflect the code: in a Z-memory experiment the Z-type stabilizers
produce detectors in every round including the data-qubit readout, while the X-type
stabilizers only produce detectors where consecutive rounds can be compared.)

Running a suite over any subset collection is one call. By default p-values use chi-squared
asymptotics; passing `bootstrap=B` calibrates each test by parametric bootstrap instead —
worth the extra cost when shots are few or the marginal is dominated by low-expectation cells,
where the asymptotic reference is unreliable. At our sample size the two agree, which is
itself a useful sanity check:

```{code-cell} ipython3
suite_st = run_marginal_tests(dem, det, TIME_SUBSETS + SPACE_SUBSETS)
print(f"time+space marginals on on-model data: "
      f"{len(suite_st.rejected())} of {len(suite_st.results)} rejected")

r_asym = marginal_likelihood_test(dem, det, SPACE_SUBSETS[0])
r_boot = marginal_likelihood_test(dem, det, SPACE_SUBSETS[0],
                                  bootstrap=2000, seed=5)
print(f"asymptotic p = {r_asym.pvalue:.3f}   bootstrap p = {r_boot.pvalue:.3f} "
      f"(same subset, same statistic)")
```

## 3. A gallery of model violations

Calibration is only half the story — the battery must also *fire*, and fire informatively,
when the model is wrong. We now manufacture four qualitatively different violations and watch
which tests catch each one. The point to take away is the *pattern*: each violation lights up
the tests whose power profile overlaps it and leaves the others quiet, and the
`effect_description`s of the firing tests name the culprit.

```{code-cell} ipython3
def scale_dem(d, factor):
    """A copy of a DEM with every error probability multiplied by `factor`."""
    out = stim.DetectorErrorModel()
    for inst in d.flattened():
        if inst.type == "error":
            p = min(inst.args_copy()[0] * factor, 0.45)
            out.append("error", p, inst.targets_copy())
        else:
            out.append(inst)
    return out
```

### 3a. Globally inflated probabilities: everything fires

The crudest violation: the candidate claims every error is $1.5\times$ more likely than it
really is (the data are our on-model `det`; the *candidate* is wrong). Every detector's click
rate, every correlator, and every marginal is off, so essentially the whole battery rejects.
The most *efficient* detector of a global rate error, though, is the Hamming-weight test with
`method="mean"`: the per-shot click count aggregates the deficit across all 24 detectors into
one number, so its z-score is enormous.

```{code-cell} ipython3
candidate_inflated = scale_dem(dem, 1.5)
suite_a = full_battery(candidate_inflated, det)
print(suite_a.summary(max_rows=6))

r_mean = hamming_weight_test(candidate_inflated, det, method="mean", seed=1)
print(f"\nhamming weight, method='mean': z = {r_mean.statistic:+.1f}, "
      f"{r_mean.effect_description}")
```

Reading the table: the effect descriptions all point the same way (observed counts *below*
expected, polarizations *above* predicted — fewer clicks than the model claims), which is the
signature of a global overestimate rather than a localized problem.

### 3b. One event's probability doubled: local tests fire and name the location

Now a surgical violation: the *truth* gains a second, independent copy of its strongest
weight-2 mechanism, so that event's effective rate doubles ($p \oplus p = 2p(1-p)$), while the
candidate keeps the original rate. In this circuit the strongest weight-2 events connect the
same stabilizer in consecutive rounds — a measurement-error mechanism — so this simulates one
stabilizer's readout being twice as noisy as modeled.

```{code-cell} ipython3
p_ev, m_ev = max((p, m) for m, p in events.items() if bin(m).count("1") == 2)
d_i, d_j = [d for d in range(dem.num_detectors) if (m_ev >> d) & 1]
print(f"doubling event D{d_i} D{d_j} (p = {p_ev:.4f}); coordinates "
      f"{coords[d_i]} and {coords[d_j]} - same stabilizer, consecutive rounds")

truth_b = dem + stim.DetectorErrorModel(f"error({p_ev}) D{d_i} D{d_j}")
det_b, _ = sample_dem(truth_b, N_SHOTS, seed=SEED + 1)
```

The polarization battery localizes the problem. A parity mask $M$ only feels an event $E$
when $|M \cap E|$ is **odd**, so the firing masks are precisely those overlapping the doubled
event in one detector — the click rates of $D_{9}$ and $D_{17}$ themselves and the pair masks
touching exactly one of them. This triangulates the event. It also means the doubled pair's
*own* mask $(9, 17)$ is blind (overlap 2, even): parity correlators and joint marginals see
different projections of the distribution, which is why the battery contains both.

```{code-cell} ipython3
suite_b_pol = run_polarization_battery(dem, det_b,
                                       collections=("weight1", "events"), seed=0)
rej = suite_b_pol.rejected(alpha=0.05)
touching = sum(1 for r in rej
               if d_i in r.details["mask"] or d_j in r.details["mask"])
print(f"polarization tests: {len(rej)} of {len(suite_b_pol.results)} rejected; "
      f"{touching} of the {len(rej)} rejected masks touch D{d_i} or D{d_j}")
print()
print(suite_b_pol.summary(max_rows=4))

own = [r for r in suite_b_pol.results if r.details["mask"] == (d_i, d_j)]
print(f"\nthe doubled pair's own mask ({d_i},{d_j}) is parity-blind to it: "
      f"p = {own[0].pvalue:.3f}")
```

The neighborhood marginals see the *joint* distribution, so they do catch the pair
correlation directly — the worst cell of the most significant subset is literally "both
$D_{9}$ and $D_{17}$ fired" (the outcome bit string lists the subset's detectors in ascending
order, so `01000100` on `(7, 9, 10, 11, 14, 17, 18, 19)` marks exactly $D_9$ and $D_{17}$).
And the distant subsets, which contain neither detector, stay quiet: the violation is local,
and tests that look elsewhere should not fire.

```{code-cell} ipython3
suite_b_marg = run_marginal_tests(dem, det_b, NBHD_SUBSETS)
rej = suite_b_marg.rejected(alpha=0.05)
touching = sum(1 for r in rej if d_i in r.details["subset"] or d_j in r.details["subset"])
print(f"neighborhood marginals: {len(rej)} of {len(suite_b_marg.results)} rejected; "
      f"{touching} of the {len(rej)} rejected subsets contain D{d_i} or D{d_j}")
print("most significant:", rej[0].name)
print("   ", rej[0].effect_description)

suite_b_dist = run_marginal_tests(dem, det_b, DISTANT_SUBSETS)
print(f"\ndistant subsets (no overlap with D{d_i}/D{d_j}): p-values "
      f"{[f'{r.pvalue:.2f}' for r in suite_b_dist.results]} - quiet, as they should be")
```

### 3c. A hyperedge missing from a graph-like candidate: only third-order tests fire

Graph-like DEMs (every event flips at most two detectors) are popular because matching
decoders require them — but real noise contains weight-3+ *hyperedges*. How wrong can a
graph-like approximation be while looking perfect to low-order tests? Completely invisible,
it turns out, up to second order.

The construction (three detectors, minimal by design so the algebra is transparent): the
truth has independent singles at rate $p$ plus a triple event $D_0 D_1 D_2$ at rate $q$. The
Walsh polarization of mask $M$ is $\prod_E (1-2p_E)^{[|M \cap E| \text{ odd}]}$, so in log
space the spectrum is linear in the events — and a graph-like candidate with singles $s$
satisfying $(1-2s) = (1-2p)/(1-2q)$ plus *pair* events at rate $q$ on all three pairs
reproduces **every weight-1 and weight-2 polarization exactly**. The two models differ only
in the third-order correlator, by a factor $(1-2q)^4$: a triple event flips its own parity
mask (overlap 3, odd), while pairs cannot (overlap 2, even). This is exactly the blind spot
the battery's `triples` collection — weight-3 masks on *connected triples* of the candidate's
detector graph — exists to cover.

```{code-cell} ipython3
p, q = 0.03, 0.01
truth_c = stim.DetectorErrorModel(f"""
    error({p}) D0
    error({p}) D1
    error({p}) D2
    error({q}) D0 D1 D2
""")
s = 0.5 * (1 - (1 - 2 * p) / (1 - 2 * q))
candidate_c = stim.DetectorErrorModel(f"""
    error({s}) D0
    error({s}) D1
    error({s}) D2
    error({q}) D0 D1
    error({q}) D0 D2
    error({q}) D1 D2
""")

masks = [(0,), (1,), (2,), (0, 1), (0, 2), (1, 2), (0, 1, 2)]
table = np.vstack([predicted_polarizations(truth_c, masks),
                   predicted_polarizations(candidate_c, masks)])
print("mask:               ", "  ".join(f"{str(m):>9s}" for m in masks))
print("truth polarization: ", "  ".join(f"{v:9.4f}" for v in table[0]))
print("cand. polarization: ", "  ".join(f"{v:9.4f}" for v in table[1]))
```

```{code-cell} ipython3
det_c, _ = sample_dem(truth_c, 50_000, seed=SEED + 2)
suite_c = run_polarization_battery(candidate_c, det_c,
                                   collections=("weight1", "weight2", "triples"),
                                   seed=0)
print(suite_c.summary(max_rows=4))

r_joint = marginal_likelihood_test(candidate_c, det_c, (0, 1, 2))
print(f"\nfull joint marginal on (0,1,2): p = {r_joint.pvalue:.2e}")
print("   ", r_joint.effect_description)
```

The weight-1 and weight-2 tests are quiet *by construction* — the graph-like candidate was
built to fake them — while the triple mask rejects at overwhelming significance, and the full
$2^3$-cell joint marginal names the smoking-gun outcome: all three detectors firing together
far more often than any pairwise model can produce. If you fit graph-like DEMs, third-order
tests are not optional.

### 3d. An unmodeled long-range correlation: the sharp null earns its keep

Finally, the violation that motivated `distant` subsets: the truth gains a weight-2 event
connecting two detectors that are *far apart* in the candidate's detector graph — think
leakage, crosstalk between distant control lines, or a cosmic-ray-like event. We take the two
extreme corners of the distant subset found in section 2 (first round vs. last round,
opposite corners of the patch).

```{code-cell} ipython3
d_a, d_b = DISTANT_SUBSETS[0][0], DISTANT_SUBSETS[0][-1]

# graph distance between them under the candidate
frontier, dist = {d_a}, 0
seen = {d_a}
while d_b not in seen:
    dist += 1
    frontier = {v for u in frontier for v in adjacency[u]} - seen
    seen |= frontier
print(f"adding unmodeled pair event D{d_a} D{d_b} (p = 0.01): coordinates "
      f"{coords[d_a]} vs {coords[d_b]}, graph distance {dist}")

truth_d = dem + stim.DetectorErrorModel(f"error(0.01) D{d_a} D{d_b}")
det_d, _ = sample_dem(truth_d, N_SHOTS, seed=SEED + 3)
```

```{code-cell} ipython3
suite_d_pol = run_polarization_battery(dem, det_d, collections=("weight2",), seed=0)
rej = suite_d_pol.rejected(alpha=0.05)
one_of = sum(1 for r in rej
             if len(set(r.details["mask"]) & {d_a, d_b}) == 1)
print(f"weight-2 polarizations: {len(rej)} of {len(suite_d_pol.results)} rejected; "
      f"{one_of} of the rejected pairs contain exactly one of D{d_a}, D{d_b}")
print("(the pair (%d,%d) itself is parity-blind again - overlap 2 is even)" % (d_a, d_b))
print()

suite_d_dist = run_marginal_tests(dem, det_d, DISTANT_SUBSETS)
r = suite_d_dist.results[0]
print(f"distant marginal {r.name}: p = {r.pvalue:.2e}")
print("   ", r.effect_description)
```

The distant subset delivers the cleanest verdict in the whole gallery. Under the candidate,
these four detectors are nearly independent, so the expected count for "$D_0$ and $D_{23}$
fire together" is tiny — the observed excess is a ~30-sigma cell that *names both endpoints
of the unmodeled correlation*. This is the sharp-null principle: the more confidently the
model predicts "nothing to see here", the more powerful the test of that prediction.

+++

## 4. Scalar functions of the syndrome: a plug-in test engine

A candidate DEM implies a distribution not just for marginals and correlators but for **any
scalar function of the syndrome**. `scalar_distribution_test` turns this into a generic test
engine: it Monte-Carlo samples the null distribution of your function from the DEM
(`num_null_shots` shots, defaulting to $\max(10 S, 20000)$), then compares observed vs. null
samples with a two-sample test:

* `method="chi2"` — binned/discrete contingency test with small-cell pooling. Sensitive to
  *shape* changes anywhere in the distribution.
* `method="ks"` — two-sample Kolmogorov–Smirnov. Nonparametric; with discrete/tied data it is
  *conservative* (true type-I error below nominal), which is safe but costs power.
* `method="mean"` — z-test on the mean, accounting for the Monte Carlo uncertainty of the
  null. The most powerful choice against pure location shifts, blind to everything else.
* `method="auto"` (default) — chi2 when the pooled support is small (discrete data like click
  counts), else KS.

The three methods rank exactly as advertised on a *mild* global inflation (a 5%-off
candidate — far subtler than scenario 3a):

```{code-cell} ipython3
candidate_mild = scale_dem(dem, 1.05)
for method in ("chi2", "ks", "mean"):
    r = hamming_weight_test(candidate_mild, det, method=method, seed=7)
    print(f"hamming weight vs 5%-inflated candidate, method={method:>4s}: "
          f"p = {r.pvalue:.2e}")
```

### 4.1 Decoder-based scalars: matching weight and complementary gap

Two powerful built-in scalars come from running a *decoder*:

* `matching_weight_function(dem)` — the log-likelihood weight of the decoder's best
  correction for each shot. Its distribution reflects how "expensive" syndromes are to
  explain under the model.
* `complementary_gap_function(dem)` — decode each shot twice, with the logical outcome forced
  each way; the gap $|w_1 - w_0|$ is the weight the *losing* logical class would have needed.
  This is a continuous decoder-confidence signal (large gap = confident decode, gap near zero
  = coin toss), and its distribution is a sensitive fingerprint of the whole DEM.

Both are wrapped as one-call tests. The pymatching backend requires a graph-like DEM (and,
for the gap, that logical-flipping events touch at most one detector) — our merged surface
DEM is not, so here we switch to a repetition-code memory circuit, which is. For hyperedge
DEMs, both functions accept `decoder="tesseract"` (below).

```{code-cell} ipython3
rep_circuit = stim.Circuit.generated(
    "repetition_code:memory", rounds=5, distance=5,
    before_round_data_depolarization=0.03, before_measure_flip_probability=0.01)
rep_dem = rep_circuit.detector_error_model(decompose_errors=True)
rep_det, rep_obs = sample_dem(rep_dem, N_SHOTS, seed=SEED)

# data from a hotter device (all physical rates 1.5x) tested against rep_dem
rep_hot = stim.Circuit.generated(
    "repetition_code:memory", rounds=5, distance=5,
    before_round_data_depolarization=0.045, before_measure_flip_probability=0.015)
rep_det_hot, _ = sample_dem(rep_hot.detector_error_model(decompose_errors=True),
                            N_SHOTS, seed=SEED + 4)

for label, data in (("on-model data", rep_det), ("hotter-device data", rep_det_hot)):
    r_mw = matching_weight_test(rep_dem, data, seed=101)
    r_gap = complementary_gap_test(rep_dem, data, seed=102)
    print(f"{label}:")
    print(f"  matching weight:   p = {r_mw.pvalue:.2e}  ({r_mw.effect_description})")
    print(f"  complementary gap: p = {r_gap.pvalue:.2e}  ({r_gap.effect_description})")
```

The directions are diagnostic: on hotter data the matching weight runs *high* (syndromes cost
more to explain than the model expects) and the gap runs *low* (the decoder is less confident
than the model promises). A gap distribution that disagrees with the model also warns you
that gap-based real-time confidence estimates would be miscalibrated on this device.

### 4.2 The tesseract backend (hyperedge-capable)

[Google's Tesseract decoder](https://github.com/quantumlib/tesseract-decoder) is a
most-likely-error decoder that handles hyperedges natively, so `decoder="tesseract"` works on
non-graph-like DEMs where pymatching refuses. It decodes shot-by-shot through its Python API,
so keep shot counts modest.

> **Install note:** `pip install tesseract-decoder` works where PyPI wheels exist (x86-64
> Linux, arm64 macOS); elsewhere it must be built from source. The cells below are skipped
> gracefully when the package is absent.

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
if TESSERACT_AVAILABLE:
    r_mw = matching_weight_test(rep_dem, rep_det[:200], decoder="tesseract",
                                num_null_shots=1000, seed=103)
    r_gap = complementary_gap_test(rep_dem, rep_det[:200], decoder="tesseract",
                                   num_null_shots=1000, seed=104)
    print(f"tesseract matching weight:   p = {r_mw.pvalue:.3f}")
    print(f"tesseract complementary gap: p = {r_gap.pvalue:.3f}")
else:
    print("skipped (tesseract-decoder not installed)")
```

### 4.3 Writing your own scalar

Any vectorized map from an `(S, num_detectors)` array to `(S,)` per-shot scalars plugs into
the same engine — this is the cheapest way to encode domain knowledge as a test. Suppose we
suspect measurement errors: their signature is the *same* stabilizer clicking in *two
consecutive rounds*. Counting such time-like double-clicks per shot gives a targeted
statistic, and (satisfyingly) it fires on the scenario-3b data, whose injected violation was
exactly a doubled measurement-error mechanism:

```{code-cell} ipython3
tpairs = [(d1, d2) for d1 in coords for d2 in coords
          if d1 < d2 and coords[d1][:-1] == coords[d2][:-1]
          and abs(coords[d1][-1] - coords[d2][-1]) == 1]
i1 = np.array([a for a, _ in tpairs])
i2 = np.array([b for _, b in tpairs])


def repeated_clicks(samples):
    """Per shot: number of stabilizers that clicked in two consecutive rounds."""
    s = np.asarray(samples)
    return (s[:, i1] & s[:, i2]).sum(axis=1)


r_ok = scalar_distribution_test(dem, det, repeated_clicks, seed=30,
                                name="repeated_clicks")
r_bad = scalar_distribution_test(dem, det_b, repeated_clicks, seed=30,
                                 name="repeated_clicks")
print(f"on-model data:    p = {r_ok.pvalue:.3f}   (method: "
      f"{r_ok.details['method_used']})")
print(f"scenario-3b data: p = {r_bad.pvalue:.2e}  "
      f"(doubled measurement error shows up in double-click counts)")
```

## 5. Logical-error-rate consistency

For error correction, the bottom line of a DEM is the logical error rate (LER) it implies.
`logical_error_rate_test` builds a decoder from the (decorated) candidate, decodes the
experimental shots, and compares the observed decoder failure rate against the rate the
candidate predicts for itself. The prediction comes from, in priority order:

1. `predicted_ler` — an externally known rate, treated as exact;
2. `ler_estimator` — a callable `estimator(dem) -> rate` or `-> (rate, stderr)`. **This is
   the extension point** for plugging in your own LER estimation machinery: an analytic
   formula, splitting/rare-event Monte Carlo for low-LER regimes where naive sampling is
   hopeless, tensor-network estimators, or another codebase entirely. A returned stderr is
   folded into the test's standard error.
3. default — plain DEM Monte Carlo with the same decoder (`num_mc_shots` shots).

Our surface-code DEM is decorated (circuit DEMs carry `L0` targets), so this works out of the
box. On-model data passes; the $1.5\times$-inflated candidate from scenario 3a fails
decisively — and the *direction* of the effect says the candidate is too pessimistic about
itself (observed LER about half of predicted):

```{code-cell} ipython3
r_ler = logical_error_rate_test(dem, det, obs, num_mc_shots=100_000, seed=11)
print(f"on-model:  p = {r_ler.pvalue:.3f}   {r_ler.effect_description}")
print(f"           (prediction source: {r_ler.details['predicted_source']}, "
      f"test branch: {r_ler.details['test_method']})")

r_ler_bad = logical_error_rate_test(candidate_inflated, det, obs,
                                    num_mc_shots=100_000, seed=12)
print(f"inflated:  p = {r_ler_bad.pvalue:.2e}   {r_ler_bad.effect_description}")
```

The `ler_estimator` hook in action — here a toy callable standing in for your favorite
estimator. It receives the candidate DEM itself, so it can be as model-aware as you like:

```{code-cell} ipython3
calls = []


def my_ler_estimator(model):
    """Stand-in for an external LER estimator (analytic, splitting MC, ...)."""
    calls.append(model)          # it receives the candidate DEM
    return (0.040, 0.002)        # (rate, standard error); a bare float also works


r_hook = logical_error_rate_test(dem, det, obs, ler_estimator=my_ler_estimator)
print(f"hook: p = {r_hook.pvalue:.3f}, source = "
      f"{r_hook.details['predicted_source']!r}, estimator called "
      f"{len(calls)} time(s) with the DEM")
print(f"      predicted {r_hook.details['ler_predicted']} +/- "
      f"{r_hook.details['predicted_stderr']} vs observed "
      f"{r_hook.details['ler_observed']:.4f}")
```

At very small expected failure counts the test automatically switches to exact
(binomial/Fisher) branches — the branch taken is always recorded in
`details['test_method']`, so borderline results can be audited.

+++

## 6. Stationarity: testing the data, not the model

Everything so far assumed shots are i.i.d. draws — but real experiments drift, and a DEM
(which has no notion of time) *cannot* fit drifting data. The stationarity tests check the
i.i.d. assumption itself: they carry `null_model="iid"`, and a rejection means **fix the
experiment or model the drift — no static DEM will fit**, so run them before burning time
re-fitting event probabilities.

* `click_rate_drift_test` — homogeneity of the mean click count over consecutive shot blocks,
  plus a trend statistic for monotone drift;
* `polarization_drift_test` — drift of parity-mask rates, which catches nonstationarity that
  *conserves* the total click rate (noise migrating between detectors);
* `shot_autocorrelation_test` — permutation test for shot-to-shot correlation of the click
  count (bursts, oscillations, duplicated data).

On-model (i.i.d. by construction) data passes. Then we manufacture two pathologies: a device
that heats up halfway through the run, and a dataset in which every shot was accidentally
duplicated.

```{code-cell} ipython3
suite_iid = run_stationarity_battery(det, seed=0)
print("i.i.d. data:  p-values",
      {r.name.replace("stationarity_", ""): round(r.pvalue, 3)
       for r in suite_iid.results})

det_hot, _ = sample_dem(scale_dem(dem, 1.4), 10_000, seed=SEED + 5)
det_drift = np.vstack([det[:10_000], det_hot])   # cool half, then hot half
suite_drift = run_stationarity_battery(det_drift, seed=0)
print("\ndrifting data (error rates jump 1.4x mid-run):")
for r in suite_drift.results:
    print(f"  {r.name:<36s} p = {r.pvalue:.2e}  {r.effect_description}")

det_dup = np.repeat(det[:10_000], 2, axis=0)     # every shot recorded twice
r_dup = shot_autocorrelation_test(det_dup, seed=0)
print(f"\nduplicated shots: {r_dup.name} p = {r_dup.pvalue:.2e}  "
      f"{r_dup.effect_description}")
```

The effect descriptions are directional: the drift case reports a *positive* slope of the
mean shot weight ("per $10^3$ shots"), i.e. the device got worse over time, and the
duplicated data shows the lag-1 autocorrelation of $r \approx 0.5$ that exact pairwise
duplication predicts. (The regime jump also induces a small positive autocorrelation — any
block structure does — but the drift tests are the ones that identify it as drift. Both
autocorrelation p-values sit at $1/(B+1)$, the resolution floor of the default
$B = 200$-permutation test; raise `num_permutations` if you need smaller p-values from it.)

+++

## 7. Practical guidance

* **Run batteries, report corrected results.** Picking the single most alarming p-value out
  of hundreds of tests is multiple-testing malpractice; `rejected(alpha, method="fdr_bh")`
  and `summary()` exist so the correction is never an afterthought. Use `"holm"` or
  `"bonferroni"` when you need family-wise (any-false-alarm) control rather than FDR.
* **Mind the $2^k$ marginal cost.** Exact marginals materialize $2^k$ cells; the hard cap is
  $k \le 20$, but power per shot also dilutes across cells — many small subsets usually beat
  one huge one. Neighborhoods, slices and balls exceeding the cap are truncated or split
  automatically (with warnings).
* **Bootstrap when the asymptotics are doubtful.** Small shot counts or marginals dominated
  by near-zero-probability cells make the chi-squared reference unreliable; `bootstrap=B`
  recalibrates the same $G$ statistic by parametric bootstrap. At large $N$ the two agree
  (section 2), so the asymptotic default is fine for big datasets.
* **At large $N$, effect sizes matter more than p-values.** With enough shots, *every* model
  is rejected — no 24-detector DEM is exactly right. A $10^{-8}$ p-value attached to a total
  variation distance of $10^{-3}$ is a statistically-certain but practically-tiny defect;
  decide on the effect-size scale (TVD, sigma, LER ratio) whether it matters for your
  application.
* **Read `rejected()` as a repair manual.** The pattern of firing tests is the diagnosis:
  everything firing with one-directional effects = global rate misestimate (3a); a cluster of
  weight-1/2 masks and neighborhoods around a few detectors = a local mechanism to re-fit
  (3b); triples firing while pairs stay quiet = missing hyperedge (3c); distant subsets
  firing = unmodeled long-range correlation (3d); `null_model="iid"` tests firing = fix the
  experiment, not the DEM (section 6). Each rejected result's `effect_description` names the
  detectors, outcome or direction to act on.
* **Remember what a pass means.** Passing the battery does not prove the model — only that
  none of these projections of the data disagree with it. Add tests where *your* application
  is sensitive: the scalar-function engine (section 4) makes that a few lines of numpy.

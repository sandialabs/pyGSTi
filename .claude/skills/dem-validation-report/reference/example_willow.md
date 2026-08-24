# Worked example: Google Willow d=5, Z memory, 10 rounds

A complete run of the pipeline and the report, with the numbers it produced.
Use it as a sanity check for the skill, and as a model for how to read the
output.

**Data.** `google_105Q_surface_code_d3_d5_d7` (Zenodo 13273331), patch
`d5_at_q4_7`, Z basis, `r10`. 50,000 shots, 240 detectors (24 stabilizer sites
× 11 detector rounds), mean click rate 0.0672.

## Pipeline

| stage | result |
|---|---|
| `bitmask_trie_search`, confidence 1 − 1e-7 | 3157 events, weights {1:240, 2:1411, 3:1320, 4:186} |
| single joint refit | 552 events fitted negative — over-complete |
| iterative backward elimination, 10 iterations | **2305** events, weights {1:126, 2:939, 3:1065, 4:175} |
| | probabilities in [1.05e-4, 8.27e-2], total mass 7.61/shot |
| `assign_logical_flags`, `decoder="pymatching"` | rank 1065/1065, 41 bit-flips, residual 0.2033 → 0.1245, 121 flagged |
| `assign_logical_flags`, `decoder="tesseract"` | rank **2304/2305**, 124 bit-flips, residual 0.2033 → **0.0830**, 364 flagged, 318 s |

Consistency: total mass 7.61 × mean weight 2.1 = 16.1 expected clicks/shot
against 16.13 observed.

Decoration cross-check: of the learned events whose detector mask also appears
in the circuit-level si1000 DEM, **915/915 got the same logical flag** under
the pymatching decoration and **1595/1595** under the tesseract one (w1
114/114, w2 801/801, w3 508/508, w4 172/172). The solver never sees the
circuit DEM, so this is a free and quite strong check — and the 680 extra
checks are exactly the weight-3/4 events pymatching left undetermined.

The decoder used for the *decoration* decides which columns the GF(2) solve
sees, so this is not a small difference: pymatching gave it 1065 columns of
2305 and left 1240 flags undetermined.

## Report

```bash
python scripts/run_report.py \
  --detectors  $D/detection_events.b8 \
  --observables $D/obs_flips_actual.b8 \
  --circuit    $D/circuit_noisy_si1000.stim \
  --learned-dem   $OUT/learned_dem_refit.dem \
  --decorated-dem $OUT/decorated_dem.dem \
  --baseline-from-circuit --baseline-label si1000 \
  --events $OUT/learned_events_refit.npz \
  --title "Willow d5 (q4_7), Z memory, 10 rounds" \
  --output $OUT/report.html
```

About 2.5 minutes end to end, 0.94 MB of HTML, 11 figures. The learned DEM was
rejected on **267 of 5186** tests; the si1000 circuit DEM on **6597 of 6597**.

| family | tests | rejected (learned) | median effect | rejected (si1000) | median effect |
|---|---|---|---|---|---|
| polarization w1 | 240 | 0 | z = −0.45 | 240 | z = −50.7 |
| polarization w2 | 2000 | 0 | z = −0.59 | 2000 | z = −69.0 |
| polarization, event masks | 2305 | 0 | z = 0.004 | 3717 | z = −73.0 |
| polarization, triples | 300 | 2 | z = −0.19 | 300 | z = −77.9 |
| marginal, distant (size 4) | 4 | 0 | TVD 0.0008 | 4 | TVD 0.083 |
| marginal, random k ≤ 4 | 50 | 2 | TVD 0.0022 | 50 | TVD 0.105 |
| marginal, time columns | 24 | 8 | TVD 0.013 | 24 | TVD 0.249 |
| marginal, space slices | 20 | 15 | TVD 0.028 | 20 | TVD 0.216 |
| marginal, spacetime balls | 240 | 237 | TVD 0.085 | 240 | TVD 0.293 |
| scalar: Hamming weight | | reject, 15.98 vs 16.13 obs (0.02 null-SD) | | reject, 6.65 vs 16.13 | |
| scalar: repeated clicks | | reject, 2.957 vs 3.095 obs (0.07 null-SD) | | reject, 1.776 vs 3.095 | |

The distant family only produced 4 subsets: `min_distance = 6` is nearly
unsatisfiable on a graph with 1240 hyperedges, and the report says so in its
setup note. (An earlier run that forced `min_distance = 5` got 25 subsets, also
zero rejections, and 269/5207 overall — the two-test difference is the BH
threshold shifting with the number of tests, not a different conclusion.)

## What the numbers say

**Right:** all first- and second-order structure. Zero rejections across 4545
weight-1 / weight-2 / event-mask polarization tests and the distant marginals.
Global scalars off by 0.02–0.07 null-SDs — rejected only because 50k shots
gives that much power.

**Wrong:** high-order joint structure inside local spacetime neighbourhoods.
237/240 balls (18–20 detectors) reject at TVD ≈ 0.085. The 18–20-body syndrome
distribution in a ball is not reproduced even though every pair inside it is.
Space slices (whole rounds) reject 15/20; the thinner time columns 8/24.

**The reference model is far worse on everything**, which is the point of
running it: it predicts 6.65 clicks/shot against 16.13 observed (the device is
~2.4× noisier than the simulation) and its polarization z-scores run to −78.

**Decoder — the interesting failure.** This is the section worth running twice.
Rerunning with `--decoders pymatching tesseract` scores all six rows on the
same 50,000 shots (the tesseract decoration above; ~15 min):

| decoder | model | observed | predicted | ratio | z |
|---|---|---|---|---|---|
| pymatching | si1000, as given | 0.04442 | 0.00114 | 38.96 | +46.9 |
| pymatching | si1000, merged, w ≤ 2 | 0.04598 | 0.000235 | 195.66 | +48.8 |
| pymatching | learned, as given | 0.14084 | 0.19208 | 0.73 | −27.4 |
| pymatching | learned, w ≤ 2 | 0.14084 | 0.00571 | 24.65 | +86.6 |
| tesseract | si1000, as given | **0.02544** | 0.00045 | 56.53 | +34.7 |
| tesseract | learned, as given | **0.08298** | 0.04640 | 1.79 | +18.9 |

Three readings, in order of how much they cost to learn:

1. **The matcher was inflating the learned model's LER by 41%.** Identical
   model, identical shots: 0.14084 under pymatching, 0.08298 under tesseract.
   The matcher was silently dropping 1240 events carrying 2.441 probability,
   about a third of the model.
2. **The observed-vs-predicted ratio only means something when both sides see
   the same model.** Under pymatching the prediction came from the whole model
   and the decoding from a third less of it, giving 0.73 — the model appearing
   to over-predict its own failures. With tesseract both sides use everything
   and the ratio is 1.79 (+18.9σ). Improved, not agreement; that 1.79 is the
   honest residual.
3. **si1000 is still the better decoding prior, and still not a description of
   the device.** It decodes at 0.02544 against the learned model's 0.08298 —
   the absolute gap shrank from 0.096 to 0.058 but the ratio stayed near 3.3× —
   while mispredicting its own LER by 56.53×.

Ignore the `w ≤ 2` rows except as a measure of the restriction's cost: they
restrict the model before *predicting* as well as before decoding, which
deletes the events responsible for most failures and makes the prediction
absurdly optimistic (hence 24.65 and 195.66). An earlier version of this skill
predicted that way by default; on a synthetic control whose model was exactly
right it gave 15.4× where predicting from the full model gives 0.95×.

The earlier pymatching-only write-up concluded that dropping hyperedges "is not
the cause", on the grounds that the same restriction costs si1000 only 0.0016.
That argument only ever bounded the effect on a *sparse* reference model, and
measured directly the drop was worth 0.058 of LER on the learned model. What
survives is the weaker claim, and it is still the headline: **matching every
low-order statistic of the data does not make a good decoding prior.** The
learned DEM is the far better description of the syndrome distribution and
still decodes ~3× worse. It is not spurious long-range edges (1 of 939 weight-2
events spans more than 3 lattice units / 2 rounds); it is the edge *weights* —
1.73 total probability on mid-range edges (dx ≤ 3, dt ≤ 2) against si1000's
0.55, and 2.35 on local edges against 1.70. Some of that extra mid-range mass
is real device noise si1000 omits, but it also creates cheap detour paths.
If you only ever look at rejection counts you will miss all of this.

**Stationarity.** All 3 tests reject: click rate drifts +0.0148 per 1000 shots
(block means 15.19 → 16.83), a weight-1 polarization drifts 8.2σ between
blocks, lag-1 autocorrelation of shot weight +0.010. The shots are not i.i.d.,
which puts a floor under how well *any* static DEM can fit this data and
partially explains the residual high-order misfit.

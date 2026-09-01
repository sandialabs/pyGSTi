---
name: dem-validation-report
description: Validate a learned detector error model against experimental shot data and render a self-contained HTML report with figures. Use when someone has a DEM (learned with pygsti.extras.sparsedem, or any stim DetectorErrorModel) plus detector shot data and wants to know how well it fits, which statistics it misses, and how it compares to a reference model or a decoder.
---

# DEM validation report

Runs the `pygsti.extras.sparsedem.validation` battery against detector shot
data and writes one self-contained HTML file (figures inlined as base64 PNGs,
no external assets): numbered narrative sections, test tables, effect-size
distributions, a decoder comparison, and interpretive commentary written by a
subagent that is handed every number the run produced.

This skill validates and reports. It does **not** learn the DEM — see
`reference/pipeline.md` for the learn → refit → decorate workflow that produces
its inputs, including the failure modes that bite in practice.

**The report is produced in two passes.** The first computes everything and
writes `report.html`, `report_brief.md` (every number, laid out for an
analyst) and `report_state.pkl`. The second hands the brief to a Fable
subagent, which returns commentary, and re-renders from the state pickle in
under a second. Do not skip the second pass: a report of raw numbers with no
reading of them is half the deliverable. See **Commentary** below.

## When to use

- "How good is this learned DEM?" / "Does my DEM fit the data?"
- Comparing a learned DEM against a circuit-level reference DEM.
- Asking *which* statistics a DEM gets wrong (weight-1? pairs? local
  spacetime neighbourhoods? the logical error rate?).

## Inputs

Required:

- **detector shot data** — `(num_shots, num_detectors)` uint8 array, or a stim
  `.b8` file.
- **learned DEM** — a `stim.DetectorErrorModel`, or a `.dem` file. The
  candidate need not actually be *learned*: a ground-truth DEM in a control
  study or a circuit-level model being audited works the same way. Set
  `candidate_label` / `--candidate-label` so the report names it correctly.

Strongly recommended, each unlocking a section:

- **circuit** (`.stim`) — gives detector coordinates, which enable the
  spacetime marginal families and the repeated-clicks scalar. Without it the
  report falls back to detector-graph neighbourhoods.
- **decorated DEM** — the learned DEM with `L0` observable targets attached.
  Enables the decoder / logical-error-rate section.
- **observable flips** — the measured logical outcomes, for the observed LER.
- **baseline DEM** — a reference model (e.g. the circuit's own
  `detector_error_model(decompose_errors=True)`). Every test is run against it
  too, and every figure gets a second series. This is what makes the numbers
  interpretable: "TVD 0.085" means little until you see the reference at 0.29.
- **event stderr** — the `stderr` array from the sparsedem fit, for error bars
  on the probability figure.

## Running it

### Pass 1 — compute

CLI, for files on disk:

```bash
python scripts/run_report.py \
  --detectors DATA/detection_events.b8 \
  --observables DATA/obs_flips_actual.b8 \
  --circuit DATA/circuit_noisy_si1000.stim \
  --learned-dem OUT/learned_dem_refit.dem \
  --decorated-dem OUT/decorated_dem.dem \
  --baseline-from-circuit --baseline-label si1000 \
  --events OUT/learned_events_refit.npz \
  --decoders pymatching tesseract \
  --title "Willow d5 (q4_7), Z memory, 10 rounds" \
  --output OUT/report.html
```

This writes `OUT/report.html`, `OUT/report_brief.md` and
`OUT/report_state.pkl`.

Python, when you need a custom coordinate convention, extra statistics, a
description of the pipeline that produced the model, or an artifacts table:

```python
from pygsti.extras.sparsedem.report import ReportInputs, generate_report

result = generate_report(ReportInputs(
    detector_samples=det,            # (shots, detectors) uint8
    observable_flips=obs,            # (shots,) uint8
    learned_dem=learned,
    decorated_dem=decorated,
    baseline_dem=baseline, baseline_label="si1000",
    circuit=circuit, coordinate_mode="google",
    decoders=("pymatching", "tesseract"),
    title="...", subtitle="...",
    pipeline_html="<p>How the model was learned...</p>",
    artifacts=[("decorated_dem.dem", "the decorated model"), ...],
    output_path=Path("report.html"),
))
print(result.summary)                # counts + headline numbers
```

`ReportResult` carries `.html_path`, `.brief_path`, `.state_path`,
`.summary`, `.results` (the raw `ValidationResult` objects per candidate),
`.decoder_table`, `.figures` (base64 PNG strings) and `.state`.

`pipeline_html` is what makes the report read like a write-up rather than a
dump: the engine cannot know how the model was produced, so the opening
section is empty unless you supply it. Give it the search parameters, the
refit history, and the decoration diagnostics. Its heading defaults to "How
the model was built"; set `pipeline_title` when that is wrong — on a synthetic
study where the data was sampled from the model, it is "How the data was
made".

### Pass 2 — commentary

Read the brief yourself first (it is short), then hand it to a Fable subagent.
Use the Agent tool with `model: "fable"`:

> Read `OUT/report_brief.md`. It contains every number produced by a detector
> error model validation run. Write the interpretive commentary for the
> report. Return a single JSON object with keys `summary`, `sections`
> (`model`, `validation`, `polarization`, `marginals`, `scalars`, `decoder`,
> `stationarity` — omit any you have nothing to say about), `reading` (a list
> of conclusion bullets, each leading with a bold claim) and `caveats`.
> Values are Markdown. Quote the real numbers; never invent one. Write it to
> `OUT/commentary.json` and reply with a one-line summary of what you
> concluded.
>
> Context you need beyond the brief: `<what the data is, how the model was
> produced, what question the reader is asking, anything already known that
> the brief does not say>`. Background on the tests: `SKILL_DIR/SKILL.md` and
> `SKILL_DIR/reference/example_willow.md` (a worked example with its numbers
> and the reading of them).

That last paragraph is the part that matters. The brief has every number but
none of the provenance — what device, what the previous run found, what the
reader is trying to decide. Supply it, and point the subagent at
`reference/example_willow.md` so it has a model of the register and the depth
expected. Then:

```bash
python scripts/annotate_report.py \
  --state OUT/report_state.pkl \
  --commentary OUT/commentary.json \
  --output OUT/report.html
```

Sub-second, nothing recomputed. Iterate on the commentary as often as you
like. You can also pass `--commentary` to `run_report.py`, or `commentary=`
to `ReportInputs`, when you already have it.

Check the result before handing it over: the commentary is the one part of the
report that can be confidently wrong. Verify every number it quotes appears in
the brief, and that the claims match the tables.

### Runtime

On a 50k-shot × 240-detector problem with a 2300-event DEM and a reference
model: about 2.5 minutes with pymatching alone, dominated by the polarization
battery and the two 200k-shot null samples. It grows quickly with detector
count and with the ball radius. `skip` accepts `marginals`, `polarization`,
`scalars`, `decoder`, `stationarity`; drop `null_shots` too while iterating on
captions.

**Adding `tesseract` costs ~5 ms per shot per row** — it decodes one shot at a
time. On 50k shots that is ~4 minutes for each of the two tesseract rows, plus
its Monte Carlo (`--tesseract-mc-shots`, default 20,000, another ~100 s per
row). Budget ~15 minutes for the two-decoder version of the run above, and run
it in the background.

**The decoder scalars choose their backend per model.** A fully graph-like
model runs batched pymatching over every shot (seconds). A model with
hyperedges falls back to tesseract on `decoder_scalar_shots` observed shots
(strided across the run) plus as many null shots — at ~4–5 ms per decode the
matching-weight test is ~80 s per model at the default 10,000-shot budget.
Disable with `--skip decoder_scalars` (or `decoder_scalar_tests=()`) while
iterating.

**The complementary gap is opt-in, and the cost asymmetry is not obvious in
advance.** Decoding a shot normally is milliseconds; decoding it with the
*losing* logical class forced makes tesseract search for a correction a whole
logical operator away — measured ~0.5 s per decode on a 3,700-event d=5
model, a 100× penalty, which is hours at the default budget. Use it when the
augmented graph is matchable (pymatching batches it in seconds — typical for
circuit-level models whose logical flips ride on weight-1 boundary events),
or via tesseract with a budget in the hundreds of shots. The synthetic
control has run it once at full budget: p = 0.945 on the true model (0.01
null-SDs), so the test is calibrated — it is only slow.

## What it tests

| section | family | what it catches |
|---|---|---|
| Marginals (G-test, effect size = TVD) | `spacetime_ball`, `time_column`, `space_slice`, `random`, `distant` | joint structure on a detector subset; balls are the high-order test, `distant` is the "no spurious long-range correlation" control |
| Polarization (z-test in the Walsh domain) | `weight1`, `weight2`, `events`, `triples` | first/second-order rates and the model's own event masks |
| Scalars (Monte-Carlo null) | Hamming weight, repeated clicks | global shape of the click distribution |
| Decoder scalars (Monte-Carlo null) | matching weight, complementary gap | the *distribution* of decoder cost and decoder confidence, per shot — decoding-prior quality that no moment or marginal test sees; the gap needs a decorated model |
| Decoder | observed vs predicted LER, per decoding graph, per decoder | whether the DEM is a good *decoding prior*, and whether its logical decoration is self-consistent |
| Stationarity | drift, block polarization, lag-1 autocorrelation | tests the **data**, not the model — a floor on how well any static DEM can do |

Rejections are Benjamini–Hochberg FDR at `alpha` (default 0.05), computed
separately per candidate.

## Interpreting the output

- **Compare families, not raw counts.** With 50k shots almost any real
  discrepancy is significant, so the story is in *which* families reject and
  at what effect size. A DEM with zero weight-1/weight-2 rejections but 237/240
  ball rejections is failing at high order only.
- **A rejected scalar with a tiny effect is not a failure.** The report prints
  each scalar's gap in units of the null SD; 0.02 null-SD is a
  power-of-the-test rejection, not a modelling error.
- **The decoder section answers a separate question.** A DEM can match every
  low-order statistic and still be a worse matching graph than a crude
  circuit-level model — extra mid-range edge mass gives the matcher cheap
  detour paths. The report's controlled comparison (baseline as-decomposed,
  baseline merged + restricted to weight ≤ 2, learned restricted to weight ≤ 2)
  separates "hyperedges got dropped" from "the weights are wrong". Read it
  before blaming the graph-like restriction.
- **`--decoders pymatching tesseract` settles that argument by measurement.**
  The controlled comparison above only *bounds* the cost of dropping
  hyperedges, and only on a sparse reference model. Tesseract decodes the same
  models using the hyperedges, so a model's matching row minus its tesseract
  row is the cost of the restriction, directly. It is slow (see Runtime) and
  gets no weight ≤ 2 row — restricting the model would defeat the point.
- **Every decoder row reports observed *and* predicted LER.** Observed is what
  that graph's matcher achieves on the data; predicted is what the same graph
  gets on Monte Carlo shots sampled from itself. For a learned DEM this is the
  end-to-end check on the logical decoration — the L0 flags decide what counts
  as a failure on both sides, so a bad assignment moves the observed rate and
  leaves the prediction where it was.
- **Compare the *as given* row, not the weight ≤ 2 one.** pymatching silently
  discards any error it cannot make an edge of, so the two rows usually decode
  identically; the difference is which model the prediction was drawn from.
  Restricting first deletes the events responsible for most of the failures, so
  the prediction comes out optimistic. On a synthetic control whose model was
  exactly right, the restricted row rejected at 15.4× (+11.4σ) while the
  as-given row passed at 0.95× (−0.6σ). The candidate's as-given row is the
  one fed to the validation battery and the KPI band — under the
  hyperedge-native decoder when one ran (a matcher's LER on a hyperedge model
  tests the restriction as much as the model), else under the matcher. The
  report states which below its decoder table.
- **A passing LER row is weak evidence; a failing one is strong.** It is one
  scalar test against thousands of polarization tests, so it has little power:
  in the synthetic study a 2%-of-mass misspecification lit up 2507 tests but
  left the LER row at +2.1σ, below the BH threshold.
- **The decoder scalars sit between the moment tests and the LER row.** The
  matching-weight distribution asks whether syndromes cost the decoder what
  the model says they should; the complementary gap asks whether the decoder
  is as confident as the model predicts. Both are whole distributions rather
  than one rate, so they have far more power than the LER row while still
  testing decoding-prior quality. A model that passes every polarization test
  but fails these is mis-shaping exactly the quantity the decoder optimizes.
- **Check stationarity first when the fit is unexplainably imperfect.** Drift
  across the run bounds what a static DEM can achieve.
- **Read the commentary as a draft, not an output.** It is written by a
  subagent from the brief alone. It is the most useful part of the report and
  the only part that can be assertively wrong, so check its numbers against
  the tables before you circulate it.

## Gotchas

- **Detector coordinates are not standardised.** Google's Willow circuits
  annotate each detector with the concatenated `(x, y, t)` of every compared
  measurement, so `DETECTOR` coordinate lists have length 3, 6, 9 or 15 and
  `build_marginal_subsets("spacetime", ...)` raises `inconsistent spatial
  dimensions`. `coordinate_mode="google"` canonicalises to
  `(c[-3], c[-2], c[2])`; `"stim"` takes `(c[0], c[1], c[-1])`; `"auto"` picks
  `stim` for uniform-length lists and `google` for ragged ones; `"none"`
  disables coordinates. Pass `coordinate_fn` for anything else, and always
  check the logged "N sites × M rounds" line — if it says 240 sites × 1 round,
  the convention is wrong.
- **`min_distance` for the distant family is often unsatisfiable.** A learned
  DEM with many weight-3/4 events has a small graph diameter regardless of the
  code distance. The report walks the requested value down until it works and
  logs what it used; the report footnotes the actual value.
- **The baseline DEM gets XOR-merged.** `dem_to_dict` merges stim's decomposed
  components by detector mask, so a decomposed circuit DEM yields fewer,
  higher-weight events than its line count suggests. That is the right
  convention for comparison but makes the baseline event count look small.
- **pymatching needs a graph-like DEM.** Weight-3/4 events are dropped for the
  decoder section; the report states how many and how much probability mass.
  It does this *silently* — handed a DEM with hyperedges it builds a smaller
  graph than the DEM describes and never says so, which is why the report
  counts them for you (`matcher_ignored`).
- **`tesseract-decoder` has no aarch64 Linux wheel and no sdist on PyPI.** On
  ARM it has to be built from source (CMake + FetchContent; the extension and
  the `_tesseract_py_util` package both need to be on `sys.path`). Rows for a
  missing decoder are skipped with a log line rather than failing the run.
- **Commentary is Markdown, and only a little of it.** Paragraphs, `-` bullets,
  `|` tables, `**bold**`, `*italic*`, `` `code` ``. Headings, links and nested
  lists are not converted. Raw HTML passes through if you need more.
- Everything is inlined, so a report with ~12 figures is roughly 2 MB. Fine to
  email, awkward to commit.

## Files

- `pygsti/extras/sparsedem/report.py` — the library: `ReportInputs`,
  `generate_report`, `load_commentary`, and the individual figure functions
  (import as `pygsti.extras.sparsedem.report`).
- `scripts/run_report.py` — CLI driver.
- `scripts/annotate_report.py` — re-renders a report from its state pickle
  with the analyst's commentary; sub-second, nothing recomputed. Also loads
  state pickles from before the move into pygsti (module alias for
  `dem_report`).
- `reference/pipeline.md` — learn → refit → decorate, the prerequisite
  workflow, with its failure modes.
- `reference/example_willow.md` — a full worked example (Google Willow d=5)
  with the numbers it produced, useful as a sanity check.

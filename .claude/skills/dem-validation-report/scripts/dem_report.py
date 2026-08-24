"""
Validate a learned detector error model against shot data and render a
self-contained HTML report.

This is the reusable engine behind the ``dem-validation-report`` skill. It
wraps ``pygsti.extras.sparsedem.validation`` with

  * detector-coordinate canonicalization, so the circuit-derived spacetime
    subset builders work on real experiment circuits;
  * a test battery sized for hundreds of detectors (subsampled families
    instead of exhaustive weight-k families);
  * an optional side-by-side reference model (typically the circuit-level
    DEM), run through the identical battery;
  * a controlled decoder comparison that separates "the decoder lost the
    hyperedges" from "the learned graph is a worse decoding graph";
  * twelve figures and an HTML report with captions generated from the
    numbers actually measured.

Everything is optional except the learned DEM and the detector samples; each
section is skipped when its inputs are absent.

Typical use::

    from dem_report import ReportInputs, generate_report

    result = generate_report(ReportInputs(
        detector_samples=det,
        learned_dem=learned_dem,
        observable_flips=obs,
        decorated_dem=decorated_dem,
        baseline_dem=circuit.detector_error_model(decompose_errors=True),
        circuit=circuit,
        title="Willow d5, Z memory, 10 rounds",
        output_path="report.html",
    ))
    print(result.summary["learned"]["num_rejected"])

See SKILL.md for the surrounding workflow and reference/pipeline.md for how
to produce the learned and decorated DEMs in the first place.
"""

from __future__ import annotations

import base64
import collections
import dataclasses
import io as _io
import pickle
import textwrap
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Sequence

import matplotlib
matplotlib.use("Agg")  # must precede pyplot; reports are rendered headless
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
import stim  # noqa: E402

from pygsti.extras.sparsedem import io as sdio  # noqa: E402
from pygsti.extras.sparsedem.validation import (
    ValidationSuiteResult,
    build_marginal_subsets,
    complementary_gap_function,
    hamming_weight_test,
    logical_error_rate_test,
    matching_weight_function,
    run_marginal_tests,
    run_polarization_battery,
    run_stationarity_battery,
    scalar_distribution_test,
)

try:  # optional, only needed for the decoder-comparison section
    import pymatching as _pymatching
except ImportError:  # pragma: no cover
    _pymatching = None

try:  # optional; hyperedge-native decoder, no aarch64 wheel on PyPI
    from tesseract_decoder import tesseract as _tesseract
except ImportError:  # pragma: no cover
    _tesseract = None


def _decoder_available(name: str) -> bool:
    """Whether a decoder backend can actually be used in this environment."""
    return {"pymatching": _pymatching, "tesseract": _tesseract}.get(name) \
        is not None


# ===========================================================================
# Configuration
# ===========================================================================

@dataclass
class ReportInputs:
    """
    Everything the report generator needs. Only `detector_samples` and
    `learned_dem` are required.

    Attributes:
        detector_samples: np.ndarray
            (num_shots, num_detectors) uint8 array in stim column order.
        learned_dem: stim.DetectorErrorModel
            The model under test, without logical decoration.
        observable_flips: Optional[np.ndarray]
            num_shots binary observable outcomes. Needed for the decoder
            section; everything else works without it.
        decorated_dem: Optional[stim.DetectorErrorModel]
            `learned_dem` with L0 targets, from `assign_logical_flags`.
            Needed for the decoder section.
        baseline_dem: Optional[stim.DetectorErrorModel]
            Reference model run through the identical battery, e.g.
            `circuit.detector_error_model(decompose_errors=True)`. Strongly
            recommended: absolute p-values are meaningless at 10^4+ shots, so
            the interesting quantity is almost always the comparison.
        baseline_label: str
            Short name for the reference model, used throughout the report.
        candidate_label: str
            Short name for the model under test. Defaults to "learned"; set it
            when the candidate is not a learned model (a ground-truth DEM in a
            control study, a circuit-level model being audited, ...) so the
            report does not call it something it is not.
        circuit: Optional[stim.Circuit]
            Source circuit. Only used for its DETECTOR coordinates, which
            unlock the spacetime subset families. Without it the report falls
            back to detector-graph subsets.
        coordinate_mode: str
            "auto" (default), "stim" (coords are already uniform, last axis
            is time), "google" (concatenated (x,y,t) triples per compared
            measurement), or "none" (ignore coordinates).
        coordinate_fn: Optional[Callable]
            Overrides coordinate_mode entirely: maps a raw coordinate list to
            an (x, y, t) triple.
        title, subtitle: str
            Report heading.
        findings_html: Optional[str]
            Interpretation to place in the summary box, as raw HTML. The
            generated captions stick to what was measured; conclusions are
            yours. Superseded by `commentary["summary"]` when both are given.
        commentary: Optional[dict]
            Interpretive prose, in Markdown, woven through the report. See
            `COMMENTARY_SCHEMA`. Produced by handing `brief_path` to an
            analyst (the skill uses a Fable subagent) and fed back in either
            here or through `annotate_report.py`, which re-renders from
            `state_path` without recomputing anything.
        pipeline_html: Optional[str]
            Raw HTML describing how the candidate model was produced --
            search, refit, decoration. The report cannot know this; supply it
            and it becomes the opening section, as in the worked example.
        pipeline_title: str
            Heading for that section. The default suits a learned model;
            override it when nothing was learned ("How the data was made").
        artifacts: Sequence[tuple]
            (filename, description) pairs listed in a closing table.
        decoders: Sequence[str]
            Decoders used for the decoder section: any of "pymatching"
            (minimum-weight matching; graph-like only) and "tesseract"
            (most-likely-error; decodes hyperedges natively). Give both to get
            matching and hypergraph rows side by side, which is the only way
            to separate "the matcher threw the hyperedges away" from "the
            model is wrong".
        num_mc_shots: Optional[int]
            Monte Carlo shots for the predicted LER, per matching row.
            Defaults to min(8 x num_shots, 400_000).
        tesseract_num_mc_shots: int
            The same, for tesseract rows, which cost ~5 ms/shot rather than
            ~1 us/shot and so need their own, much smaller, budget.
        decoder_scalar_tests: Sequence[str]
            Decoder-derived scalar distribution tests, run per model and
            added to the scalar family: "matching_weight" (per-shot weight of
            the decoder's best correction) and "complementary_gap" (extra
            weight of the best correction in the opposite logical class -- a
            decoder-confidence signal; needs a decorated model with exactly
            one observable). Both compare the observed distribution against
            the model's own Monte Carlo, so they test the model *as a
            decoding prior*, which the moment and marginal tests do not.
            Default is matching weight only: the gap must decode every shot
            with the losing logical class forced, and on a hyperedge model
            that sends tesseract hunting for a logical-operator-weight error
            chain at ~0.5 s per decode (measured on a 3,700-event d=5
            model) -- hours at the default budget. Opt in to
            "complementary_gap" when the augmented graph is matchable
            (batched pymatching, cheap) or with a small
            `decoder_scalar_shots`. Empty tuple to disable; also skippable
            via skip name "decoder_scalars".
        decoder_scalar_shots: int
            Shot budget (observed subsample and Monte Carlo null alike) for
            decoder scalars whose backend decodes one shot at a time. A
            graph-like model uses batched pymatching on every shot with the
            usual `null_shots`; a model with hyperedges falls back to
            tesseract at ~5-10 ms/shot, where this budget applies. The
            observed subsample is strided across the run so drift does not
            bias it.
        output_path: str or Path
            Where to write the HTML.
        brief_path: Optional[str or Path]
            Where to write the Markdown commentary brief -- every number in
            the report, laid out for an analyst to interpret. Defaults to
            `<output_path stem>_brief.md`; pass False to suppress.
        state_path: Optional[str or Path]
            Where to pickle the render state, so `annotate_report.py` can
            re-render with commentary without rerunning the battery. Defaults
            to `<output_path stem>_state.pkl`; pass False to suppress.
        results_path: Optional[str or Path]
            If set, pickle the ValidationResult objects here. Arrays in
            `.details` larger than 4096 entries are replaced by a placeholder
            first: a 20-detector marginal test keeps two 2**20-entry
            distributions, and an unslimmed battery pickles to several GB.
            `ReportResult.results` (in memory) is not slimmed.
        seed: int
        null_shots: int
            Shots sampled from each model for the scalar-distribution
            comparisons and per-detector rates.
        space_radius, time_radius: float
            Spacetime ball geometry.
        num_random_subsets, random_k: int
            Random low-weight marginal subsets.
        num_distant_subsets, distant_size, distant_min_distance: int
            Graph-distant marginal subsets. `distant_min_distance` is reduced
            automatically if it is unsatisfiable on the learned graph.
        max_weight2_masks, num_triple_masks: int
            Polarization battery sizing.
        event_stderr: Optional[dict]
            {event bitmask: standard error} from the fit's covariance, e.g.
            `sqrt(diag(cov))` keyed by `dem_masks`. Adds a significance panel
            to the event-probability figure.
        alpha: float
            Family-wise significance level (Benjamini-Hochberg).
        skip: Sequence[str]
            Section names to omit: any of "marginals", "polarization",
            "scalars", "decoder_scalars", "decoder", "stationarity".
    """

    detector_samples: np.ndarray
    learned_dem: stim.DetectorErrorModel
    observable_flips: Optional[np.ndarray] = None
    decorated_dem: Optional[stim.DetectorErrorModel] = None
    baseline_dem: Optional[stim.DetectorErrorModel] = None
    baseline_label: str = "baseline"
    candidate_label: str = "learned"
    circuit: Optional[stim.Circuit] = None
    coordinate_mode: str = "auto"
    coordinate_fn: Optional[Callable[[Sequence[float]], Sequence[float]]] = None
    title: str = "Learned detector error model"
    subtitle: str = ""
    findings_html: Optional[str] = None
    commentary: Optional[dict] = None
    pipeline_html: Optional[str] = None
    pipeline_title: str = "How the model was built"
    artifacts: Sequence[tuple] = ()
    decoders: Sequence[str] = ("pymatching",)
    num_mc_shots: Optional[int] = None
    tesseract_num_mc_shots: int = 20_000
    decoder_scalar_tests: Sequence[str] = ("matching_weight",)
    decoder_scalar_shots: int = 10_000
    output_path: Path = Path("dem_report.html")
    results_path: Optional[Path] = None
    brief_path: Optional[Path] = None
    state_path: Optional[Path] = None
    seed: int = 0
    null_shots: int = 200_000
    space_radius: float = 2.0
    time_radius: float = 1.0
    num_random_subsets: int = 50
    random_k: int = 4
    num_distant_subsets: int = 25
    distant_size: int = 4
    distant_min_distance: int = 6
    max_weight2_masks: int = 2000
    num_triple_masks: int = 300
    event_stderr: Optional[dict] = None
    alpha: float = 0.05
    skip: Sequence[str] = ()


@dataclass
class ReportResult:
    """What `generate_report` returns."""
    html_path: Path
    summary: dict = field(default_factory=dict)
    results: dict = field(default_factory=dict)
    decoder_table: list = field(default_factory=list)
    figures: dict = field(default_factory=dict)
    brief_path: Optional[Path] = None
    state_path: Optional[Path] = None
    state: object = None


# ===========================================================================
# Detector coordinates
# ===========================================================================

def canonical_coordinates(circuit: stim.Circuit, mode: str = "auto",
                          coordinate_fn: Optional[Callable] = None) -> dict:
    """
    Reduce a circuit's DETECTOR annotations to one (x, y, t) triple each.

    The spacetime subset builders need a uniform spatial dimension, which
    real experiment circuits often violate. Two conventions are recognized:

      * "stim": every detector has the same number of coordinates and the
        last one is time. Used verbatim (first two spatial axes are kept).
      * "google": coordinate lists are concatenated (x, y, t) triples, one
        per measurement the detector compares, so lengths are 3, 6, 9, 15,
        ... The *last* triple is the stabilizer's own site and the *first*
        triple's time is the detector's round, giving
        (x_last, y_last, t_first).

    Parameters:
        circuit: stim.Circuit
        mode: str
            "auto", "stim", "google" or "none".
        coordinate_fn: Optional[Callable]
            Overrides `mode`: raw coordinate list -> (x, y, t).

    Returns:
        coords: dict
            Detector index -> (x, y, t). Empty if mode == "none" or the
            circuit carries no coordinates.
    """
    raw = circuit.get_detector_coordinates()
    if mode == "none" or not raw:
        return {}
    if coordinate_fn is not None:
        return {d: tuple(float(v) for v in coordinate_fn(c))
                for d, c in raw.items() if c}

    lengths = {len(c) for c in raw.values() if c}
    if not lengths:
        return {}
    if mode == "auto":
        # Uniform lengths are taken at face value; ragged ones can only be the
        # concatenated-triple convention.
        mode = "stim" if len(lengths) == 1 else "google"
    if mode == "stim":
        if len(lengths) != 1:
            raise ValueError(
                f"coordinate_mode='stim' needs uniform coordinate lengths, "
                f"found {sorted(lengths)}. Use 'google' or pass "
                "coordinate_fn=..."
            )
        n = lengths.pop()
        if n < 2:
            raise ValueError("stim-mode coordinates need at least (x, t).")
        return {d: (float(c[0]), float(c[1]) if n >= 3 else 0.0, float(c[-1]))
                for d, c in raw.items() if c}
    if mode == "google":
        bad = [n for n in lengths if n % 3]
        if bad:
            raise ValueError(
                f"coordinate_mode='google' expects concatenated (x, y, t) "
                f"triples, found coordinate length(s) {sorted(bad)} that are "
                "not multiples of 3. Pass coordinate_fn=... instead."
            )
        return {d: (float(c[-3]), float(c[-2]), float(c[2]))
                for d, c in raw.items() if c}
    raise ValueError(f"unknown coordinate_mode {mode!r}")


def coordinate_circuit(coords: dict, num_detectors: int) -> stim.Circuit:
    """
    A bare DETECTOR-only circuit carrying `coords`, for the subset builders.

    The sparsedem spacetime builders take a circuit and read
    `get_detector_coordinates()`; handing them a synthetic circuit is how you
    feed them canonicalized coordinates without touching the real one.
    """
    return stim.Circuit("\n".join(
        f"DETECTOR({coords[d][0]},{coords[d][1]},{coords[d][2]})"
        if d in coords else "DETECTOR(0,0,0)"
        for d in range(num_detectors)))


# ===========================================================================
# DEM helpers
# ===========================================================================

def dem_with_observables(dem: stim.DetectorErrorModel) -> dict:
    """
    {detector_mask: (probability, observable_parity)} with decomposed
    components XOR-merged, matching `sparsedem.io.dem_to_dict`'s convention.

    Independent duplicate masks are combined as independent flips. Where two
    instructions share a detector mask but disagree on the observable, the
    first occurrence wins (in a circuit-derived DEM they never disagree).
    """
    merged: dict = {}
    for inst in dem.flattened():
        if inst.type != "error":
            continue
        det_mask = obs_parity = 0
        for t in inst.targets_copy():
            if t.is_relative_detector_id():
                det_mask ^= 1 << t.val
            elif t.is_logical_observable_id():
                obs_parity ^= 1
        if not det_mask:
            continue
        p = inst.args_copy()[0]
        q, o = merged.get(det_mask, (0.0, obs_parity))
        merged[det_mask] = (q * (1 - p) + p * (1 - q), o)
    return merged


def flat_dem(events: dict, num_detectors: int) -> stim.DetectorErrorModel:
    """
    A flat, undecorated stim DEM from {mask: probability}, spanning
    `num_detectors` detectors.

    The merged-dict form is what the decoder-scalar functions want: it is the
    same syndrome distribution as the source DEM, but with no separators for
    `build_matcher` to trip over and no decomposition ambiguity.
    """
    lines = []
    for mask, p in sorted(events.items()):
        targets = " ".join(f"D{i}" for i in range(mask.bit_length())
                           if mask >> i & 1)
        p = min(max(float(p), 1e-15), 0.5 - 1e-15)
        lines.append(f"error({p}) {targets}")
    lines.append(f"detector D{num_detectors - 1}")
    return stim.DetectorErrorModel("\n".join(lines))


def graphlike_dem(events: dict) -> tuple:
    """
    Restrict {mask: (prob, obs)} to weight <= 2 and build a stim DEM.

    Returns:
        dem: stim.DetectorErrorModel
        dropped_count: int
        dropped_mass: float
    """
    lines, dropped, mass = [], 0, 0.0
    for mask, (p, o) in events.items():
        if bin(mask).count("1") > 2:
            dropped += 1
            mass += p
            continue
        targets = " ".join(f"D{i}" for i in range(mask.bit_length())
                           if mask >> i & 1)
        p = min(max(float(p), 1e-15), 0.5 - 1e-15)
        lines.append(f"error({p}) {targets}" + (" L0" if o else ""))
    return stim.DetectorErrorModel("\n".join(lines)), dropped, mass


def sample_detectors(dem: stim.DetectorErrorModel, num_shots: int,
                     num_detectors: int, seed: int) -> np.ndarray:
    """Detector samples from a DEM, right-padded to `num_detectors` columns."""
    d, _, _ = dem.compile_sampler(seed=seed).sample(shots=num_shots)
    d = np.asarray(d, dtype=np.uint8)
    if d.shape[1] < num_detectors:
        d = np.hstack([d, np.zeros((num_shots, num_detectors - d.shape[1]),
                                   np.uint8)])
    return d[:, :num_detectors]


# ===========================================================================
# Subsets
# ===========================================================================

def build_subsets(cfg: ReportInputs, coords: dict, learned_dict: dict,
                  num_detectors: int, log=print) -> tuple:
    """
    Build the marginal-test subset families.

    Spacetime families need coordinates; without them the report falls back
    to detector-graph neighbourhoods. `distant_min_distance` is walked down
    until it is satisfiable, because a learned DEM with many hyperedges has a
    much smaller graph diameter than the code distance suggests.
    """
    subsets: dict = {}
    distant_min_distance = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # oversize splits/truncations expected
        if coords:
            cc = coordinate_circuit(coords, num_detectors)
            subsets["spacetime_ball"] = build_marginal_subsets(
                "spacetime", circuit=cc, space_radius=cfg.space_radius,
                time_radius=cfg.time_radius)
            subsets["time_column"] = build_marginal_subsets("time", circuit=cc)
            subsets["space_slice"] = build_marginal_subsets("space", circuit=cc)
        else:
            log("no detector coordinates: using detector-graph neighbourhoods "
                "in place of the spacetime families")
            subsets["neighborhood"] = build_marginal_subsets(
                "neighborhood", dem=learned_dict, radius=1)
        subsets["random"] = build_marginal_subsets(
            "random", num_detectors=num_detectors, k=cfg.random_k,
            num_subsets=cfg.num_random_subsets, seed=cfg.seed)

        for min_distance in range(cfg.distant_min_distance, 1, -1):
            try:
                subsets["distant"] = build_marginal_subsets(
                    "distant", dem=learned_dict, size=cfg.distant_size,
                    num_subsets=cfg.num_distant_subsets,
                    min_distance=min_distance, seed=cfg.seed)
            except ValueError:
                continue
            if min_distance != cfg.distant_min_distance:
                log(f"distant subsets: min_distance {cfg.distant_min_distance} "
                    f"is unsatisfiable on this graph, used {min_distance}")
            distant_min_distance = min_distance
            break
        else:
            log("distant subsets: no satisfiable min_distance, family skipped")

    for name, subs in subsets.items():
        sizes = sorted({len(s) for s in subs})
        log(f"subsets[{name}]: {len(subs)} subsets, sizes "
            f"{sizes[0]}..{sizes[-1]}")
    return subsets, distant_min_distance


# ===========================================================================
# The battery
# ===========================================================================

def repeated_clicks_statistic(coords: dict):
    """
    Per-shot count of stabilizers that clicked in two consecutive rounds.

    A targeted scalar: it is sensitive to temporal correlation structure that
    the pairwise polarization tests can miss because it aggregates over every
    site at once. Returns None when there are no coordinates to define it.
    """
    pairs = [(a, b) for a in coords for b in coords
             if a < b and coords[a][:2] == coords[b][:2]
             and abs(coords[a][2] - coords[b][2]) == 1]
    if not pairs:
        return None
    i1 = np.array([a for a, _ in pairs])
    i2 = np.array([b for _, b in pairs])

    def statistic(samples):
        s = np.asarray(samples)
        return (s[:, i1] & s[:, i2]).sum(axis=1)

    return statistic


def run_battery(cfg: ReportInputs, label: str, dem: stim.DetectorErrorModel,
                dem_dict: dict, subsets: dict, extra_scalars: dict,
                log=print) -> list:
    """Run every enabled test family against one candidate DEM."""
    det = cfg.detector_samples
    results = []

    if "marginals" not in cfg.skip:
        for name, subs in subsets.items():
            suite = run_marginal_tests(dem_dict, det, subs)
            for r in suite.results:
                r.name = f"{label}:{name}:{r.name}"
            results.extend(suite.results)
            log(f"{label}: marginals[{name}] done ({len(subs)} tests)")

    if "polarization" not in cfg.skip:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # weight-2 subsampling warning
            pol = run_polarization_battery(
                dem_dict, det,
                collections=("weight1", "weight2", "events", "triples"),
                max_weight2_masks=cfg.max_weight2_masks,
                num_triple_masks=cfg.num_triple_masks, seed=cfg.seed)
        for r in pol.results:
            r.name = f"{label}:{r.name}"
        results.extend(pol.results)
        log(f"{label}: polarization battery done ({len(pol.results)} tests)")

    if "scalars" not in cfg.skip:
        r = hamming_weight_test(dem, det, seed=cfg.seed,
                                num_null_shots=cfg.null_shots)
        r.name = f"{label}:{r.name}"
        results.append(r)
        for name, fn in extra_scalars.items():
            r = scalar_distribution_test(dem, det, fn, seed=cfg.seed,
                                         num_null_shots=cfg.null_shots,
                                         name=name)
            r.name = f"{label}:{r.name}"
            results.append(r)
        log(f"{label}: scalar tests done")

    return results


def decoder_scalar_battery(cfg: ReportInputs, label: str, dem_dict: dict,
                           decorated: Optional[stim.DetectorErrorModel],
                           num_detectors: int, log=print) -> tuple:
    """
    The decoder-derived scalar tests for one model: matching weight and
    complementary gap.

    Both are `scalar_distribution_test`s on a per-shot decoder statistic, so
    they probe the model *as a decoding prior* -- weight distributions and
    decoder confidence -- which no moment or marginal test sees. Backend per
    model: batched pymatching over every shot when the statistic is
    graph-like, else tesseract on `cfg.decoder_scalar_shots` observed shots
    (strided across the run) and as many null shots. Skipped, with a log
    line, when neither backend can represent the statistic.

    Returns:
        results: list of ValidationResult (named scalar[...], so they join
            the scalar family and the BH battery)
        fig_data: {test_key: (observed_values, null_values, backend)} for
            `figure_decoder_scalars`, reusing the arrays the tests computed
            rather than decoding twice.
    """
    det = np.asarray(cfg.detector_samples, dtype=np.uint8) % 2
    results, fig_data = [], {}
    graphlike = all(bin(m).count("1") <= 2 for m in dem_dict)
    tests = tuple(cfg.decoder_scalar_tests)

    def strided(samples, n):
        step = max(1, samples.shape[0] // n)
        return samples[::step][:n]

    def run(test_name, make_func, dem_for_null, backend):
        slow = backend != "pymatching"
        obs_shots = strided(det, cfg.decoder_scalar_shots) if slow else det
        null_shots = cfg.decoder_scalar_shots if slow else cfg.null_shots
        if slow:
            log(f"{label}: scalar[{test_name}] via {backend}, "
                f"{len(obs_shots):,} observed + {null_shots:,} null shots "
                f"decoded one at a time")
        # scalar_distribution_test calls the statistic exactly twice --
        # observed then null -- so a recording wrapper hands the figure the
        # same arrays the test used, at no extra decoding cost.
        inner, calls = make_func(), []

        def recording(samples):
            values = inner(samples)
            calls.append(np.asarray(values, dtype=float).ravel())
            return values

        r = scalar_distribution_test(
            dem_for_null, obs_shots, recording,
            num_null_shots=null_shots, seed=cfg.seed,
            name=f"{test_name}[{backend}]")
        r.name = f"{label}:{r.name}"
        results.append(r)
        if len(calls) == 2:
            fig_data[test_name] = (calls[0], calls[1], backend)
        log(f"{label}: scalar[{test_name}[{backend}]] done "
            f"(p = {r.pvalue:.3g}, {r.effect_description})")

    def run_with_fallback(test_name, backends, make_func, dem_for_null):
        # pymatching can refuse a statistic at construction (non-graph-like)
        # or at decode time (no perfect matching on the augmented graph);
        # either way the tesseract backend is the fallback, and a scalar
        # test failing must not kill the report.
        for backend in backends:
            try:
                run(test_name, lambda: make_func(backend), dem_for_null,
                    backend)
                return
            except Exception as exc:
                # pymatching's not-graph-like message enumerates every bad
                # event; keep the log readable.
                msg = str(exc)
                log(f"{label}: scalar[{test_name}] via {backend} failed: "
                    f"{msg[:160]}{'...' if len(msg) > 160 else ''}")
        if not backends:
            log(f"{label}: scalar[{test_name}] skipped: no capable decoder "
                f"backend installed")

    if "matching_weight" in tests:
        model = flat_dem(dem_dict, num_detectors)
        backends = (["pymatching"] if graphlike
                    and _decoder_available("pymatching") else [])
        if _decoder_available("tesseract"):
            backends.append("tesseract")
        run_with_fallback(
            "matching_weight", backends,
            lambda b: matching_weight_function(model, decoder=b), model)

    if "complementary_gap" in tests:
        if decorated is None or decorated.num_observables != 1:
            log(f"{label}: scalar[complementary_gap] skipped: needs a "
                f"decorated model with exactly one observable")
        else:
            backends = [b for b in ("pymatching", "tesseract")
                        if _decoder_available(b)]
            run_with_fallback(
                "complementary_gap", backends,
                lambda b: complementary_gap_function(decorated, decoder=b),
                decorated)

    return results, fig_data


def matcher_ignored(dem: stim.DetectorErrorModel) -> tuple:
    """
    (count, probability mass) of errors pymatching cannot turn into edges.

    pymatching builds one edge per component of a decomposed error and
    *silently* drops any component touching more than two detectors. A DEM
    handed to it as given therefore decodes with a smaller graph than it
    describes, with no warning -- worth reporting explicitly.
    """
    n, mass = 0, 0.0
    for inst in dem.flattened():
        if inst.type != "error":
            continue
        component, bad = 0, False
        for t in inst.targets_copy():
            if t.is_separator():
                bad |= component > 2
                component = 0
            elif t.is_relative_detector_id():
                component += 1
        if bad or component > 2:
            n += 1
            mass += inst.args_copy()[0]
    return n, mass


def decoder_comparison(cfg: ReportInputs, learned_events: dict,
                       log=print) -> tuple:
    """
    Controlled decoder comparison, observed against predicted.

    With `decoders=("pymatching",)` four matching graphs are scored:

      1. the reference DEM as given (stim-decomposed, if it is),
      2. the reference DEM merged and restricted to weight <= 2,
      3. the decorated candidate DEM as given,
      4. the decorated candidate DEM restricted to weight <= 2.

    Rows 1 and 2 isolate the cost of discarding hyperedges from the cost of
    the candidate being a worse decoding graph -- without row 2 a large gap in
    row 4 is uninterpretable.

    Adding "tesseract" adds two more rows, the reference and the candidate as
    given, decoded by a most-likely-error decoder that uses the hyperedges
    instead of discarding them. That turns the argument above from an
    inference into a measurement: the candidate's matching row and its
    tesseract row differ by exactly the events the matcher could not use.
    Tesseract costs about 5 ms/shot against pymatching's microsecond, so its
    Monte Carlo budget is separate (`tesseract_num_mc_shots`) and there is no
    weight <= 2 row for it -- restricting the model would defeat the point.

    Each row gets *two* rates, from `logical_error_rate_test`:

      observed  -- what that graph's matcher achieves on the experimental
                   data, i.e. how often its correction disagrees with the
                   measured observables;
      predicted -- what the same graph predicts for itself: Monte Carlo shots
                   sampled from it, decoded by the same matcher, scored
                   against the observables it sampled alongside them.

    Observed alone confounds a wrong model with a hard experiment. The pair
    separates them, and it is the cheapest end-to-end check on a *learned*
    model's logical decoration: the L0 flags decide which corrections count as
    failures on both sides, so mis-assigned flags move the observed rate and
    leave the self-consistent prediction where it was.

    The as-given / restricted split matters more than it looks, which is why
    row 3 exists and is the row fed to the validation battery. pymatching
    drops hyperedges either way (see `matcher_ignored`), so rows 3 and 4
    usually *decode* identically; what differs is where the prediction comes
    from. Restricting the model before predicting throws away the very events
    that cause most of the failures, so the prediction comes out optimistic
    and the test rejects a model that is exactly right. Predicting from the
    full model is the honest comparison.

    Returns:
        table: list of row dicts with keys label, note, decoder, kind
            ("candidate" or "baseline"), observed, predicted,
            predicted_stderr, ratio, z, pvalue
        ler_result: ValidationResult for the candidate-as-given row that
            joins the validation battery: the hyperedge-native decoder's when
            one ran (it tests the model rather than the graph-like
            restriction of it), else the matcher's. None if nothing scored.
        drop_info: (n_dropped, dropped_mass) for the weight <= 2 restriction
    """
    if cfg.observable_flips is None or cfg.decorated_dem is None:
        log("no observables or decorated DEM: decoder section skipped")
        return [], None, (0, 0.0)

    decoders = [d for d in cfg.decoders if _decoder_available(d)]
    for d in cfg.decoders:
        if d not in decoders:
            log(f"decoder '{d}' is not installed: those rows are skipped")
    if not decoders:
        log("no decoder backend installed: decoder section skipped")
        return [], None, (0, 0.0)

    det, obs = cfg.detector_samples, cfg.observable_flips
    cl = cfg.candidate_label
    default_mc = cfg.num_mc_shots or min(len(det) * 8, 400_000)
    table = []

    def score(dem, name, kind, decoder, note="", restricted=False):
        """LER test of one decoding graph; appends a row, returns the result."""
        if dem.num_observables == 0:
            log(f"decoder '{name}' skipped: no logical observables")
            return None
        mc = cfg.tesseract_num_mc_shots if decoder == "tesseract" else default_mc
        try:
            r = logical_error_rate_test(dem, det, obs, decoder=decoder,
                                        num_mc_shots=mc, seed=cfg.seed)
        except Exception as exc:  # pragma: no cover - decoder-specific
            log(f"decoder '{name}' failed: {exc}")
            return None
        d = r.details
        table.append({"label": name, "note": note, "kind": kind,
                      "decoder": decoder, "restricted": restricted,
                      "observed": d["ler_observed"],
                      "predicted": d["ler_predicted"],
                      "predicted_stderr": d["predicted_stderr"],
                      "ratio": float(r.effect_size), "z": float(d["z"]),
                      "pvalue": float(r.pvalue)})
        log(f"  [{decoder}] {name}: observed {d['ler_observed']:.4f}, "
            f"predicted {d['ler_predicted']:.4f} ({mc:,} MC shots), "
            f"ratio {r.effect_size:.2f}, {d['z']:+.1f} sigma")
        return r

    def ignored_note(dem):
        n, m = matcher_ignored(dem)
        return (f"{n} errors ({m:.3f} mass) are not edges and are silently "
                f"ignored by the matcher" if n else "fully graph-like")

    candidate_rows = {}  # decoder -> its candidate-as-given result
    n_drop, mass = graphlike_dem(learned_events)[1:]
    for decoder in decoders:
        if decoder == "pymatching":
            if cfg.baseline_dem is not None:
                score(cfg.baseline_dem, f"{cfg.baseline_label}, as given",
                      "baseline", decoder,
                      f"reference model; {ignored_note(cfg.baseline_dem)}")
                base_gl, n_drop_b, mass_b = graphlike_dem(
                    dem_with_observables(cfg.baseline_dem))
                score(base_gl, f"{cfg.baseline_label}, merged, weight ≤ 2",
                      "baseline", decoder,
                      f"same restriction as the {cl} model "
                      f"({n_drop_b} events, {mass_b:.3f} mass dropped)",
                      restricted=True)
            r = score(cfg.decorated_dem, f"{cl}, decorated, as given",
                      "candidate", decoder,
                      f"predicted from the whole model; "
                      f"{ignored_note(cfg.decorated_dem)}")
            learned_gl = graphlike_dem(learned_events)[0]
            score(learned_gl, f"{cl}, decorated, weight ≤ 2", "candidate",
                  decoder,
                  f"{n_drop} hyperedge events ({mass:.3f} mass) dropped before "
                  f"predicting as well as before decoding", restricted=True)
        else:  # tesseract and any future hyperedge-native backend
            log(f"[{decoder}] decoding {len(det):,} shots one at a time; "
                f"this is the slow part of the report")
            if cfg.baseline_dem is not None:
                score(cfg.baseline_dem, f"{cfg.baseline_label}, as given",
                      "baseline", decoder, "reference model, hyperedges used")
            r = score(cfg.decorated_dem, f"{cl}, decorated, as given",
                      "candidate", decoder,
                      f"every event used, including the {n_drop} hyperedges "
                      f"({mass:.3f} mass) a matcher discards")
        if r is not None:
            candidate_rows[decoder] = r
    # The row that joins the validation battery and headlines the report:
    # prefer a decoder that used the whole model over one that silently
    # restricted it. A matcher's LER on a hyperedge model tests the
    # restriction as much as the model.
    ler_result = None
    for decoder in [d for d in decoders if d != "pymatching"] + ["pymatching"]:
        if decoder in candidate_rows:
            ler_result = candidate_rows[decoder]
            break
    log(f"decoder comparison: {len(table)} rows over "
        f"{', '.join(decoders)}")
    if ler_result is not None:
        ler_result.name = f"learned:{ler_result.name}"
    return table, ler_result, (n_drop, mass)


# ===========================================================================
# Figures
# ===========================================================================

def _fmt_rate(x: float) -> str:
    """Error rates span decades; keep the small ones readable."""
    return f"{x:.4f}" if x >= 1e-3 else f"{x:.2e}"


C_LEARNED, C_BASE, C_DATA = "#2a6fb5", "#c8642a", "#999999"
_STYLE = {"figure.dpi": 130, "font.size": 9, "axes.grid": True,
          "grid.alpha": 0.25, "axes.spines.top": False,
          "axes.spines.right": False, "figure.facecolor": "white"}


def _encode(fig) -> str:
    buf = _io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _weight_stats(dem_dict: dict) -> tuple:
    count, mass = collections.Counter(), collections.defaultdict(float)
    for mask, p in dem_dict.items():
        w = bin(mask).count("1")
        count[w] += 1
        mass[w] += p
    return count, mass


def figure_weights(learned_dict, baseline_dict, baseline_label,
                   cand_label="learned"):
    c_l, m_l = _weight_stats(learned_dict)
    weights = sorted(set(c_l) | set(_weight_stats(baseline_dict)[0]
                                    if baseline_dict else set()))
    x = np.arange(len(weights))
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.0))
    if baseline_dict:
        c_b, m_b = _weight_stats(baseline_dict)
        axes[0].bar(x - 0.19, [c_l.get(w, 0) for w in weights], 0.34,
                    color=C_LEARNED, label=f"{cand_label} ({len(learned_dict)})")
        axes[0].bar(x + 0.19, [c_b.get(w, 0) for w in weights], 0.34,
                    color=C_BASE, label=f"{baseline_label} ({len(baseline_dict)})")
        axes[1].bar(x - 0.19, [m_l.get(w, 0) for w in weights], 0.34,
                    color=C_LEARNED, label=cand_label)
        axes[1].bar(x + 0.19, [m_b.get(w, 0) for w in weights], 0.34,
                    color=C_BASE, label=baseline_label)
        title = (f"Error mass by weight ({sum(m_l.values()):.2f} vs "
                 f"{sum(m_b.values()):.2f} per shot)")
    else:
        axes[0].bar(x, [c_l.get(w, 0) for w in weights], 0.5, color=C_LEARNED,
                    label=f"{cand_label} ({len(learned_dict)})")
        axes[1].bar(x, [m_l.get(w, 0) for w in weights], 0.5, color=C_LEARNED,
                    label=cand_label)
        title = f"Error mass by weight ({sum(m_l.values()):.2f} per shot)"
    for ax, ylab, t in ((axes[0], "number of events", "Event count by weight"),
                        (axes[1], "probability per shot", title)):
        ax.set_xticks(x, [f"weight {w}" for w in weights])
        ax.set_ylabel(ylab)
        ax.set_title(t)
        ax.legend(fontsize=7.5)
    return _encode(fig)


def figure_probabilities(masks, probs, stderr, cand_label="learned"):
    weight = np.array([bin(int(m)).count("1") for m in masks])
    palette = plt.cm.Blues(np.linspace(0.95, 0.4, max(weight.max(), 1)))
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.0))
    lo = max(probs[probs > 0].min() if (probs > 0).any() else 1e-5, 1e-8)
    bins = np.logspace(np.log10(lo), np.log10(max(probs.max(), lo * 10)), 45)
    for w in sorted(set(weight)):
        axes[0].hist(probs[weight == w], bins=bins, histtype="step", lw=1.6,
                     color=palette[w - 1],
                     label=f"weight {w} (n={int((weight == w).sum())})")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("fitted event probability")
    axes[0].set_ylabel("events")
    axes[0].set_title(f"Event probabilities ({cand_label})")
    axes[0].legend(fontsize=7.5)

    if stderr is not None:
        snr = probs / np.where(stderr > 0, stderr, np.inf)
        axes[1].scatter(probs, snr, s=3, alpha=0.35,
                        c=[palette[w - 1] for w in weight])
        axes[1].axhline(3, color="#b03a3a", lw=1, ls="--", label="3 sigma")
        axes[1].set_xscale("log")
        axes[1].set_yscale("log")
        axes[1].set_xlabel("fitted event probability")
        axes[1].set_ylabel("probability / standard error")
        axes[1].set_title(f"Significance ({int((snr < 3).sum())} of {len(snr)} "
                          "below 3 sigma)")
        axes[1].legend(fontsize=7.5)
    else:
        axes[1].axis("off")
        axes[1].text(0.5, 0.5, "no standard errors supplied", ha="center",
                     va="center", fontsize=9, color="#777")
    return _encode(fig)


_POL_TITLES = {"polarization_w1": "weight-1 masks",
               "polarization_w2": "weight-2 masks",
               "polarization_event": "event-aligned masks",
               "polarization_w3": "connected triples"}


def figure_polarization(fams, baseline_label, cand_label="learned"):
    present = [f for f in _POL_TITLES if fams["learned"].get(f)]
    if not present:
        return None
    fig, axes = plt.subplots(1, len(present), figsize=(2.9 * len(present), 2.7),
                             squeeze=False)
    for ax, f in zip(axes[0], present):
        z = np.array([r.effect_size for r in fams["learned"][f]])
        ax.hist(z, bins=40, color=C_LEARNED, alpha=0.85, label=cand_label)
        xs = np.linspace(-5, 5, 200)
        ax.plot(xs, len(z) * (xs[1] - xs[0]) * np.exp(-xs ** 2 / 2)
                / np.sqrt(2 * np.pi), color="k", lw=1, ls="--", label="N(0,1)")
        ax.set_xlim(-6, 6)
        ax.set_title(f"{_POL_TITLES[f]} ({len(z)})", fontsize=8)
        ax.set_xlabel("z (observed vs predicted)")
        if fams.get("baseline", {}).get(f):
            zb = np.array([r.effect_size for r in fams["baseline"][f]])
            ax.text(0.03, 0.95,
                    f"{baseline_label} median z = {np.median(zb):.0f}",
                    transform=ax.transAxes, fontsize=7, color=C_BASE, va="top")
    axes[0][0].set_ylabel("tests")
    axes[0][0].legend(fontsize=7)
    return _encode(fig)


def figure_marginal_tvd(fams, marginal_families, baseline_label,
                        cand_label="learned"):
    present = [f for f in marginal_families if fams["learned"].get(f)]
    if not present:
        return None
    fig, ax = plt.subplots(figsize=(1.6 * len(present) + 2.4, 3.2))
    x = np.arange(len(present))
    series = [("learned", cand_label, C_LEARNED, -0.18)]
    if fams.get("baseline"):
        series.append(("baseline", baseline_label, C_BASE, 0.18))
    for key, lab, col, off in series:
        data = [np.array([r.effect_size for r in fams[key][f]]) for f in present]
        bp = ax.boxplot(data, positions=x + off, widths=0.3, showfliers=False,
                        patch_artist=True, medianprops=dict(color="k", lw=1.1))
        for patch in bp["boxes"]:
            patch.set_facecolor(col)
            patch.set_alpha(0.85)
        ax.plot([], [], color=col, lw=6, label=lab)
    ax.set_xticks(x, [f.replace("_", "\n") for f in present], fontsize=8)
    ax.set_yscale("log")
    ax.set_ylabel("total variation distance")
    ax.set_title("Marginal G-tests: model-vs-data TVD by subset family")
    ax.legend(fontsize=8)
    return _encode(fig)


def figure_pvalue_ecdf(fams, families, cand_label="learned"):
    present = [f for f in families if fams["learned"].get(f)]
    if not present:
        return None
    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    cols = plt.cm.viridis(np.linspace(0.05, 0.85, len(present)))
    for f, col in zip(present, cols):
        p = np.sort(np.array([r.pvalue for r in fams["learned"][f]]))
        ax.step(np.concatenate([[0], p, [1]]),
                np.concatenate([[0], np.arange(1, len(p) + 1) / len(p), [1]]),
                where="post", lw=1.4, color=col, label=f.replace("_", " "))
    ax.plot([0, 1], [0, 1], "k--", lw=0.9, label="uniform (model correct)")
    ax.set_xlabel("p-value")
    ax.set_ylabel("empirical CDF")
    ax.set_title(f"{cand_label}: p-value calibration by family")
    ax.legend(fontsize=6.8, loc="lower right")
    return _encode(fig)


def figure_scalars(det, null_learned, null_baseline, statistics, baseline_label,
                   cand_label="learned"):
    fig, axes = plt.subplots(1, len(statistics), figsize=(4.3 * len(statistics), 3.0),
                             squeeze=False)
    for ax, (title, fn) in zip(axes[0], statistics.items()):
        o = fn(det)
        l = fn(null_learned)
        hi = int(max(np.percentile(o, 99.9), np.percentile(l, 99.9))) + 2
        bins = np.arange(-0.5, hi + 0.5)
        ax.hist(o, bins=bins, density=True, color=C_DATA, alpha=0.45,
                label=f"data (mean {o.mean():.2f})")
        ax.hist(l, bins=bins, density=True, histtype="step", lw=1.7,
                color=C_LEARNED, label=f"{cand_label} ({l.mean():.2f})")
        if null_baseline is not None:
            b = fn(null_baseline)
            ax.hist(b, bins=bins, density=True, histtype="step", lw=1.7,
                    color=C_BASE, label=f"{baseline_label} ({b.mean():.2f})")
        ax.set_title(title)
        ax.set_xlabel("statistic")
        ax.legend(fontsize=7.5)
    axes[0][0].set_ylabel("density")
    return _encode(fig)


def figure_decoder_scalars(data, baseline_label, cand_label="learned"):
    """One panel per (model, decoder-scalar): data against the model's MC."""
    panels = [(key, test) + vals
              for key in ("learned", "baseline") if key in data
              for test, vals in data[key].items()]
    if not panels:
        return None
    fig, axes = plt.subplots(1, len(panels),
                             figsize=(3.5 * len(panels), 2.9), squeeze=False)
    for ax, (key, test, obs, null, backend) in zip(axes[0], panels):
        col = C_LEARNED if key == "learned" else C_BASE
        label = cand_label if key == "learned" else baseline_label
        lo = float(min(obs.min(), null.min()))
        hi = float(max(np.percentile(obs, 99.5), np.percentile(null, 99.5)))
        if hi <= lo:
            hi = lo + 1.0
        bins = np.linspace(lo, hi, 40)
        ax.hist(obs, bins=bins, density=True, color=C_DATA, alpha=0.45,
                label=f"data (mean {obs.mean():.2f})")
        ax.hist(null, bins=bins, density=True, histtype="step", lw=1.7,
                color=col, label=f"model MC ({null.mean():.2f})")
        ax.set_title(f"{test.replace('_', ' ')} — {label} [{backend}]",
                     fontsize=8)
        ax.set_xlabel("statistic")
        ax.legend(fontsize=6.8)
    axes[0][0].set_ylabel("density")
    return _encode(fig)


def figure_click_rates(det, null_learned, null_baseline, baseline_label,
                       cand_label="learned"):
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.1))
    r_obs, r_l = det.mean(0), null_learned.mean(0)
    axes[0].plot(r_obs, lw=3.0, color=C_DATA, alpha=0.9,
                 solid_capstyle="round", label="data")
    axes[0].plot(r_l, lw=1.0, color=C_LEARNED, ls="--", label=cand_label)
    axes[1].scatter(r_obs, r_l, s=8, color=C_LEARNED, label=cand_label)
    top = max(r_obs.max(), r_l.max())
    if null_baseline is not None:
        r_b = null_baseline.mean(0)
        axes[0].plot(r_b, lw=0.9, color=C_BASE, alpha=0.9, label=baseline_label)
        axes[1].scatter(r_obs, r_b, s=8, color=C_BASE, label=baseline_label)
        top = max(top, r_b.max())
    axes[0].set_xlabel("detector index")
    axes[0].set_ylabel("click rate")
    axes[0].set_title("Per-detector click rate")
    axes[0].legend(fontsize=7.5)
    lim = [0, top * 1.08]
    axes[1].plot(lim, lim, "k--", lw=0.9)
    axes[1].set_xlim(lim)
    axes[1].set_ylim(lim)
    axes[1].set_xlabel("observed click rate")
    axes[1].set_ylabel("model click rate")
    axes[1].set_title("Model vs data, detector by detector")
    axes[1].legend(fontsize=7.5)
    return _encode(fig)


def figure_ball_structure(fams, coords, baseline_label, cand_label="learned"):
    balls = fams["learned"].get("spacetime_ball")
    if not balls:
        return None
    tvd = np.array([r.effect_size for r in balls])
    size = np.array([len(r.details["subset"]) for r in balls])
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.0))
    axes[0].scatter(size, tvd, s=12, color=C_LEARNED, label=cand_label)
    if fams.get("baseline", {}).get("spacetime_ball"):
        tb = np.array([r.effect_size
                       for r in fams["baseline"]["spacetime_ball"]])
        axes[0].scatter(size, tb, s=12, color=C_BASE, alpha=0.7,
                        label=baseline_label)
    axes[0].set_xlabel("detectors in the ball")
    axes[0].set_ylabel("TVD")
    axes[0].set_title("Misfit vs the order of the marginal")
    axes[0].legend(fontsize=7.5)

    if coords:
        rounds = np.array([int(np.median([coords[d][2]
                                          for d in r.details["subset"]]))
                           for r in balls])
        uniq = sorted(set(rounds))
        axes[1].boxplot([tvd[rounds == r] for r in uniq], positions=uniq,
                        widths=0.6, showfliers=False, patch_artist=True,
                        boxprops=dict(facecolor=C_LEARNED, alpha=0.85),
                        medianprops=dict(color="k"))
        axes[1].set_xlabel("ball center round")
        axes[1].set_ylabel(f"TVD ({cand_label})")
        axes[1].set_title("Misfit across the experiment")
    else:
        axes[1].axis("off")
    return _encode(fig)


def figure_edges_and_decoder(learned_dict, baseline_dict, coords,
                             decoder_table, baseline_label,
                             cand_label="learned"):
    have_edges = bool(coords)
    n_panels = int(have_edges) + int(bool(decoder_table))
    if not n_panels:
        return None
    width = 4.4 * n_panels + 0.7 * max(len(decoder_table) - 2, 0)
    fig, axes = plt.subplots(1, n_panels, figsize=(width, 3.3), squeeze=False)
    panel = 0

    if have_edges:
        cats = ["local\ndx<=1.5, dt<=1", "mid\ndx<=3, dt<=2", "long range"]

        def edge_mass(dem_dict):
            out = dict.fromkeys(cats, 0.0)
            for mask, p in dem_dict.items():
                if bin(mask).count("1") != 2:
                    continue
                ds = [i for i in range(mask.bit_length()) if mask >> i & 1]
                if ds[0] not in coords or ds[1] not in coords:
                    continue
                a, b = coords[ds[0]], coords[ds[1]]
                s = float(np.hypot(a[0] - b[0], a[1] - b[1]))
                t = abs(a[2] - b[2])
                key = cats[0] if (s <= 1.5 and t <= 1) else \
                    cats[1] if (s <= 3 and t <= 2) else cats[2]
                out[key] += p
            return out

        ax = axes[0][panel]
        panel += 1
        x = np.arange(3)
        el = edge_mass(learned_dict)
        if baseline_dict:
            eb = edge_mass(baseline_dict)
            ax.bar(x - 0.17, [el[c] for c in cats], 0.32, color=C_LEARNED,
                   label=cand_label)
            ax.bar(x + 0.17, [eb[c] for c in cats], 0.32, color=C_BASE,
                   label=baseline_label)
        else:
            ax.bar(x, [el[c] for c in cats], 0.45, color=C_LEARNED,
                   label=cand_label)
        ax.set_xticks(x, cats, fontsize=7.5)
        ax.set_ylabel("total probability per shot")
        ax.set_title("Weight-2 error mass by edge range")
        ax.legend(fontsize=7.5)

    if decoder_table:
        ax = axes[0][panel]
        multi = len({row.get("decoder", "pymatching")
                     for row in decoder_table}) > 1
        labels = [textwrap.fill(
            (row["label"] + f" [{row['decoder']}]") if multi else row["label"],
            17) for row in decoder_table]
        observed = [row["observed"] for row in decoder_table]
        predicted = [row["predicted"] for row in decoder_table]
        cols = [C_LEARNED if row["kind"] == "candidate" else C_BASE
                for row in decoder_table]
        x = np.arange(len(decoder_table))
        ax.bar(x - 0.18, observed, 0.34, color=cols)
        ax.bar(x + 0.18, predicted, 0.34, color=cols, alpha=0.4, hatch="//",
               edgecolor="white")
        top = max(observed + predicted) or 1.0
        for i, (o, p) in enumerate(zip(observed, predicted)):
            ax.text(i - 0.18, o + top * 0.03, _fmt_rate(o), ha="center",
                    fontsize=7)
            ax.text(i + 0.18, p + top * 0.03, _fmt_rate(p), ha="center",
                    fontsize=7)
        # A dividing line between decoder blocks: the eye should not compare a
        # matching row with a hypergraph row without noticing it is doing so.
        if multi:
            for i in range(1, len(decoder_table)):
                if (decoder_table[i]["decoder"]
                        != decoder_table[i - 1]["decoder"]):
                    ax.axvline(i - 0.5, color="#999", lw=0.9, ls=":")
        ax.set_xticks(x, labels, fontsize=6.5)
        ax.set_ylim(0, top * 1.2)
        handles = [
            Patch(facecolor="#666666", label="observed, on the data"),
            Patch(facecolor="#666666", alpha=0.4, hatch="//",
                  edgecolor="white", label="predicted, model Monte Carlo"),
            Patch(facecolor=C_LEARNED, label=cand_label)]
        if any(row["kind"] == "baseline" for row in decoder_table):
            handles.append(Patch(facecolor=C_BASE, label=baseline_label))
        ax.legend(handles=handles, fontsize=7)
        ax.set_ylabel("logical error rate")
        ax.set_title("Observed vs predicted logical error rate")
    return _encode(fig)


def figure_stationarity(det, stationarity_results, num_blocks=50):
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 2.9))
    weights = det.sum(axis=1)
    blocks = np.array_split(weights, num_blocks)
    means = np.array([b.mean() for b in blocks])
    sems = np.array([b.std() / np.sqrt(len(b)) for b in blocks])
    centers = (np.arange(num_blocks) + 0.5) * len(det) / num_blocks
    axes[0].errorbar(centers, means, yerr=sems, fmt="o", ms=3, lw=0.8,
                     color="#444")
    slope, intercept = np.polyfit(centers, means, 1)
    axes[0].plot(centers, slope * centers + intercept, color="#b03a3a", lw=1.4,
                 label=f"slope {slope * 1000:+.4f} / 1000 shots")
    axes[0].axhline(weights.mean(), color=C_LEARNED, ls="--", lw=1,
                    label="run mean")
    axes[0].set_xlabel("shot index")
    axes[0].set_ylabel("mean clicks per shot")
    axes[0].set_title("Detector click rate over the run")
    axes[0].legend(fontsize=7.5)

    names = [r.name.replace("stationarity_", "") for r in stationarity_results]
    pv = [max(r.pvalue, 1e-300) for r in stationarity_results]
    axes[1].barh(names, -np.log10(pv), color="#b03a3a", height=0.5)
    axes[1].axvline(-np.log10(0.05), color="k", ls="--", lw=1,
                    label="alpha = 0.05")
    axes[1].set_xlabel("-log10 p")
    axes[1].set_title("Stationarity battery (null: i.i.d. shots)")
    axes[1].legend(fontsize=7.5)
    return _encode(fig)


def figure_rejections(fams, rejected_names, families, baseline_label,
                      cand_label="learned"):
    present = [f for f in families if fams["learned"].get(f)]
    if not present:
        return None
    fig, ax = plt.subplots(figsize=(7.6, 0.34 * len(present) + 1.4))
    y = np.arange(len(present))

    def frac(key, f):
        tests = fams[key][f]
        if not tests:
            return 0.0
        return sum(r.name in rejected_names[key] for r in tests) / len(tests)

    ax.barh(y + 0.19, [frac("learned", f) for f in present], 0.34,
            color=C_LEARNED, label=cand_label)
    if fams.get("baseline"):
        ax.barh(y - 0.19, [frac("baseline", f) for f in present], 0.34,
                color=C_BASE, label=baseline_label)
    ax.set_yticks(y, [f"{f.replace('_', ' ')}  (n={len(fams['learned'][f])})"
                      for f in present], fontsize=8)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("fraction of tests rejected (FDR-BH)")
    ax.set_title("Validation outcome by test family")
    ax.legend(fontsize=8, loc="lower right")
    ax.invert_yaxis()
    return _encode(fig)


# ===========================================================================
# HTML
# ===========================================================================

_CSS = """
 body { font-family: -apple-system, "Segoe UI", Roboto, Helvetica, sans-serif;
        max-width: 1000px; margin: 2.5rem auto; padding: 0 1.4rem;
        color: #1c1c1c; line-height: 1.55; }
 h1 { font-size: 1.6rem; margin-bottom: 0.2rem; }
 h2 { margin-top: 2.4rem; border-bottom: 2px solid #e3e3e3;
      padding-bottom: .3rem; }
 h3 { margin-top: 1.6rem; font-size: 1.02rem; color: #333; }
 .sub { color: #666; margin-top: 0; }
 figure { margin: 1.4rem 0; text-align: center; }
 img { max-width: 100%; border: 1px solid #e6e6e6; border-radius: 4px; }
 figcaption { font-size: .82rem; color: #666; margin-top: .45rem;
              text-align: left; }
 table { border-collapse: collapse; width: 100%; font-size: .87rem;
         margin: 1rem 0; }
 th, td { border-bottom: 1px solid #e6e6e6; padding: .38rem .6rem;
          text-align: left; }
 th { background: #f6f7f9; }
 td.num { text-align: right; font-variant-numeric: tabular-nums; }
 .good { color: #1a7a3c; font-weight: 600; }
 .bad { color: #b03a3a; font-weight: 600; }
 .box { background: #f6f8fb; border-left: 4px solid #2a6fb5;
        padding: .8rem 1rem; margin: 1.2rem 0; }
 .note { margin: .9rem 0 1.4rem; }
 .note p:first-child { margin-top: .3rem; }
 .note table { font-size: .85rem; }
 code { background: #f2f2f4; padding: .1rem .3rem; border-radius: 3px;
        font-size: .87em; }
 .kpi { display: flex; gap: 1rem; flex-wrap: wrap; margin: 1.2rem 0; }
 .kpi div { flex: 1 1 150px; background: #f6f8fb; border-radius: 6px;
            padding: .7rem .9rem; }
 .kpi b { display: block; font-size: 1.3rem; color: #2a6fb5; }
 .kpi span { font-size: .78rem; color: #666; }
"""


def _fig_html(data, caption):
    if not data:
        return ""
    return (f'<figure><img src="data:image/png;base64,{data}" />'
            f"<figcaption>{caption}</figcaption></figure>\n")


def _family_table(fams, rejected_names, families, baseline_label, unit,
                  cand_label="learned"):
    rows = ["<table><tr><th>family</th><th class='num'>tests</th>"
            f"<th class='num'>rejected ({cand_label})</th>"
            f"<th class='num'>median effect ({cand_label})</th>"]
    if fams.get("baseline"):
        rows[0] += (f"<th class='num'>rejected ({baseline_label})</th>"
                    f"<th class='num'>median effect ({baseline_label})</th>")
    rows[0] += "</tr>"
    for f in families:
        tests = fams["learned"].get(f)
        if not tests:
            continue
        n_rej = sum(r.name in rejected_names["learned"] for r in tests)
        med = np.median([r.effect_size for r in tests])
        cls = "good" if n_rej == 0 else ("bad" if n_rej == len(tests) else "")
        row = (f"<tr><td>{f.replace('_', ' ')}</td>"
               f"<td class='num'>{len(tests)}</td>"
               f"<td class='num {cls}'>{n_rej}</td>"
               f"<td class='num'>{med:.4g}</td>")
        if fams.get("baseline"):
            btests = fams["baseline"].get(f, [])
            if btests:
                bn = sum(r.name in rejected_names["baseline"] for r in btests)
                bcls = "good" if bn == 0 else ("bad" if bn == len(btests) else "")
                # The families can differ in size (event-aligned masks follow
                # each model's own event list); say so rather than let a
                # rejected count exceed the tests column.
                brej = (f"{bn} of {len(btests)}" if len(btests) != len(tests)
                        else f"{bn}")
                row += (f"<td class='num {bcls}'>{brej}</td>"
                        f"<td class='num'>"
                        f"{np.median([r.effect_size for r in btests]):.4g}</td>")
            else:
                row += "<td class='num'>&mdash;</td><td class='num'>&mdash;</td>"
        rows.append(row + "</tr>")
    rows.append(f"</table><p style='font-size:.8rem;color:#666'>"
                f"Effect sizes are {unit}.</p>")
    return "\n".join(rows)


# ===========================================================================
# Top level
# ===========================================================================

MARGINAL_FAMILIES = ["distant", "random", "neighborhood", "time_column",
                     "space_slice", "spacetime_ball"]
POLARIZATION_FAMILIES = list(_POL_TITLES)
ALL_FAMILIES = (POLARIZATION_FAMILIES + MARGINAL_FAMILIES
                + ["scalar", "logical_error_rate"])


_MAX_PICKLED_ARRAY = 4096


def _slim_result(result, limit: int = _MAX_PICKLED_ARRAY):
    """
    A copy of `result` with oversized arrays in `.details` replaced by a
    description of what was there.

    A marginal test on a 20-detector ball keeps two 2**20-entry distributions,
    so pickling a full battery unslimmed runs to several GB. The scalar fields
    every caller actually wants -- name, pvalue, effect_size,
    effect_description -- are untouched.
    """
    details = {}
    for key, value in getattr(result, "details", {}).items():
        if isinstance(value, np.ndarray) and value.size > limit:
            details[key] = f"<dropped ndarray shape={value.shape} " \
                           f"dtype={value.dtype}>"
        else:
            details[key] = value
    return dataclasses.replace(result, details=details)


# ===========================================================================
# Render state, commentary, and the analyst's brief
# ===========================================================================

def _strip_arrays(cfg: ReportInputs) -> ReportInputs:
    """A picklable `cfg` with the shot data, DEMs and circuit removed."""
    return dataclasses.replace(
        cfg, detector_samples=np.zeros((0, 0), np.uint8),
        learned_dem=stim.DetectorErrorModel(), observable_flips=None,
        decorated_dem=None, baseline_dem=None, circuit=None,
        coordinate_fn=None, event_stderr=None)


def _default_sidecar(value, output: Path, suffix: str) -> Optional[Path]:
    """`None` -> a path beside the report; `False` -> do not write it."""
    if value is False:
        return None
    if value is None:
        return output.with_name(output.stem + suffix)
    return Path(value)


@dataclass
class RenderState:
    """
    Everything `_render_html` needs and nothing it does not.

    Pickled to `state_path` so `annotate_report.py` can re-render the report
    with an analyst's commentary without rerunning a battery that takes
    minutes. The config it carries has had its bulk arrays stripped.
    """
    cfg: ReportInputs
    summary: dict
    fams: dict
    rejected_names: dict
    figures: dict
    decoder_table: list
    drop_info: tuple
    stationarity: list
    coords: dict
    distant_min_distance: Optional[int]
    n_shots: int
    n_det: int
    model_stats: dict = field(default_factory=dict)


COMMENTARY_SCHEMA = {
    "summary": "Markdown. The headline box at the top of the report: what "
               "the study found, in 3-6 sentences.",
    "sections": {
        "model": "Markdown on the candidate model itself -- its event "
                 "inventory, error mass, and what those imply.",
        "validation": "Markdown on the overall rejection pattern: which "
                      "families fail, which pass, and what that says.",
        "polarization": "Markdown on the moment tests.",
        "marginals": "Markdown on the G-tests and the TVD structure.",
        "scalars": "Markdown on the scalar-distribution tests.",
        "decoder": "Markdown on the decoder section.",
        "stationarity": "Markdown on the stationarity of the data.",
    },
    "reading": "List of Markdown strings, each a bullet in a closing "
               "'Reading of the results' section. Lead each with a bold "
               "claim, then the evidence.",
    "caveats": "Optional list of Markdown strings: what the numbers do not "
               "support, and what would be needed to settle it.",
}

_COMMENTARY_SLOTS = ("model", "validation", "polarization", "marginals",
                     "scalars", "decoder", "stationarity")


def _md_inline(text: str) -> str:
    """`code`, **bold**, *italic*, and nothing else. Deliberately small."""
    import re
    out = []
    for i, chunk in enumerate(re.split(r"(`[^`]*`)", text)):
        if i % 2:  # inside backticks: no further markup, escape the HTML
            body = chunk[1:-1].replace("&", "&amp;").replace("<", "&lt;")
            out.append(f"<code>{body}</code>")
            continue
        chunk = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", chunk)
        chunk = re.sub(r"(?<![\*\w])\*([^*]+?)\*(?!\w)", r"<em>\1</em>", chunk)
        out.append(chunk)
    return "".join(out)


def md_to_html(text: str) -> str:
    """
    Paragraphs, `-`/`*` bullet lists, `|` tables, and inline markup.

    Enough Markdown for interpretive prose, with no dependency to install and
    no HTML sanitising surprises. Raw HTML in the input passes through, which
    is intentional: an analyst may want a <span class="bad">.
    """
    if not text:
        return ""
    html, buf, bullets, rows = [], [], [], []

    def flush():
        if buf:
            html.append(f"<p>{_md_inline(' '.join(buf))}</p>")
            buf.clear()
        if bullets:
            items = "".join(f"<li>{_md_inline(b)}</li>" for b in bullets)
            html.append(f"<ul>{items}</ul>")
            bullets.clear()
        if rows:
            head, *body = [r for r in rows
                           if not set(r.replace("|", "").strip()) <= set("-: ")]
            cells = lambda r, tag: "".join(  # noqa: E731
                f"<{tag}>{_md_inline(c.strip())}</{tag}>"
                for c in r.strip().strip("|").split("|"))
            trs = "".join(f"<tr>{cells(r, 'td')}</tr>" for r in body)
            html.append(f"<table><tr>{cells(head, 'th')}</tr>{trs}</table>")
            rows.clear()

    for line in str(text).splitlines():
        stripped = line.strip()
        if not stripped:
            flush()
        elif stripped.startswith("|"):
            if buf or bullets:
                flush()
            rows.append(stripped)
        elif stripped[:2] in ("- ", "* "):
            if buf or rows:
                flush()
            bullets.append(stripped[2:])
        elif bullets and line[:1] in " \t":
            bullets[-1] += " " + stripped   # continuation of a bullet
        else:
            if bullets or rows:
                flush()
            buf.append(stripped)
    flush()
    return "\n".join(html)


def _commentary_block(commentary, slot) -> str:
    """The analyst's note for one report section, or nothing."""
    if not commentary:
        return ""
    body = md_to_html((commentary.get("sections") or {}).get(slot, ""))
    return f'<div class="note">{body}</div>\n' if body else ""


def commentary_brief(state: RenderState) -> str:
    """
    Every number in the report, laid out for an analyst to interpret.

    The report's own captions describe what was measured. Interpretation --
    which failures matter, what they imply about the model, what to do next --
    needs a reader who can hold all the numbers at once, and this is what that
    reader is handed. Written as Markdown so it can go straight into a prompt.
    """
    cfg, s = state.cfg, state.summary
    cl, bl = cfg.candidate_label, cfg.baseline_label
    # A reference model may exist even when its battery was skipped.
    has_base = "baseline" in s
    ms = state.model_stats
    L = [f"# Analyst brief: {cfg.title}", ""]
    if cfg.subtitle:
        L += [cfg.subtitle, ""]
    L += [
        "Everything the validation run measured. Your job is to interpret it:",
        f"the model under test is called **{cl}**"
        + (f", the reference model **{bl}**." if has_base
           else " and there is no reference model."),
        "",
        "## Setup",
        "",
        f"- {state.n_shots:,} shots x {state.n_det} detectors; mean click rate "
        f"{ms.get('observed_click_rate', float('nan')):.4f}, mean "
        f"{ms.get('observed_mean_clicks', float('nan')):.2f} clicks/shot.",
        f"- Rejections are Benjamini-Hochberg FDR at alpha = {cfg.alpha}, "
        f"computed separately per model.",
        f"- Detector coordinates: {'available' if state.coords else 'none'}"
        + (f" ({len({(v[0], v[1]) for v in state.coords.values()})} sites x "
           f"{len({v[2] for v in state.coords.values()})} rounds), so the "
           f"spacetime families are real spacetime neighbourhoods "
           f"(space radius {cfg.space_radius}, time radius {cfg.time_radius})."
           if state.coords else
           "; marginal subsets fall back to detector-graph neighbourhoods."),
        f"- Graph-distant family: min_distance = {state.distant_min_distance}"
        + (f" (requested {cfg.distant_min_distance}; the learned graph admits "
           f"no more subsets that far apart)"
           if state.distant_min_distance != cfg.distant_min_distance else "")
        + ".",
        f"- Scalar tests use {cfg.null_shots:,} Monte Carlo shots per model.",
        "",
        "## Headline counts",
        "",
        f"- {cl}: **{s['learned']['num_rejected']} of "
        f"{s['learned']['num_tests']}** tests rejected.",
    ]
    if has_base:
        L.append(f"- {bl}: **{s['baseline']['num_rejected']} of "
                 f"{s['baseline']['num_tests']}** tests rejected.")
    L += ["", "## Per-family results", "",
          f"| family | tests | rejected ({cl}) | median effect ({cl}) | "
          f"max effect ({cl}) |"
          + (f" rejected ({bl}) | median effect ({bl}) |" if has_base else ""),
          "|---|---|---|---|---|" + ("---|---|" if has_base else "")]
    for f in ALL_FAMILIES:
        tests = state.fams["learned"].get(f)
        if not tests:
            continue
        eff = np.array([r.effect_size for r in tests], dtype=float)
        n_rej = sum(r.name in state.rejected_names["learned"] for r in tests)
        row = (f"| {f.replace('_', ' ')} | {len(tests)} | {n_rej} | "
               f"{np.median(eff):.4g} | {np.max(eff):.4g} |")
        if has_base:
            bt = state.fams["baseline"].get(f, [])
            if bt:
                beff = np.array([r.effect_size for r in bt], dtype=float)
                bn = sum(r.name in state.rejected_names["baseline"] for r in bt)
                brej = f"{bn}/{len(bt)}" if len(bt) != len(tests) else f"{bn}"
                row += f" {brej} | {np.median(beff):.4g} |"
            else:
                row += " - | - |"
        L.append(row)
    L += ["",
          "Effect sizes: polarization families are z-scores of observed vs "
          "predicted polarization (correct model => N(0,1)); marginal "
          "families are total variation distances; the logical error rate "
          "family is a ratio observed/predicted.", ""]

    L += ["## The model", ""]
    if ms.get("weight_counts"):
        L += [f"| detector weight | events ({cl}) | mass ({cl}) |"
              + (f" events ({bl}) | mass ({bl}) |" if has_base else ""),
              "|---|---|---|" + ("---|---|" if has_base else "")]
        weights = sorted(set(ms["weight_counts"])
                         | set(ms.get("baseline_weight_counts") or {}))
        for w in weights:
            row = (f"| {w} | {ms['weight_counts'].get(w, 0)} | "
                   f"{ms['weight_mass'].get(w, 0.0):.3f} |")
            if has_base:
                row += (f" {(ms.get('baseline_weight_counts') or {}).get(w, 0)}"
                        f" | {(ms.get('baseline_weight_mass') or {}).get(w, 0.0):.3f} |")
            L.append(row)
        L.append("")
    L += [f"- Total error mass per shot (sum of event probabilities = "
          f"expected error events): {s['total_error_mass']:.3f} ({cl})"
          + (f" vs {ms.get('baseline_total_mass', float('nan')):.3f} ({bl})."
             if has_base else "."),
          f"- Mean clicks per shot: {ms.get('observed_mean_clicks', float('nan')):.3f} "
          f"observed, {ms.get('candidate_mean_clicks', float('nan')):.3f} "
          f"under {cl}"
          + (f", {ms.get('baseline_mean_clicks', float('nan')):.3f} under {bl}."
             if has_base else "."),
          f"- Worst per-detector click-rate discrepancy: "
          f"{ms.get('max_click_rate_gap', float('nan')):.4f} ({cl})"
          + (f", {ms.get('baseline_max_click_rate_gap', float('nan')):.4f} ({bl})."
             if has_base else "."), ""]

    scalars = [r for r in state.fams["learned"].get("scalar", [])]
    if scalars:
        L += ["### Scalar tests", ""]
        for r in scalars:
            L.append(f"- `{r.name.split(':', 2)[-1]}`: p = {r.pvalue:.3g}, "
                     f"{r.effect_description}")
        L.append("")

    if state.decoder_table:
        n_drop, mass = state.drop_info
        L += ["## Decoder", "",
              f"{n_drop} of the {cl} model's events are not graph-like "
              f"({mass:.3f} total probability). A matching decoder silently "
              f"discards them; a hypergraph decoder does not.", "",
              "| decoder | graph | observed LER | predicted LER | ratio | z | "
              "p | note |", "|---|---|---|---|---|---|---|---|"]
        for r in state.decoder_table:
            L.append(f"| {r.get('decoder', 'pymatching')} | {r['label']} | "
                     f"{r['observed']:.5g} | {r['predicted']:.5g} | "
                     f"{r['ratio']:.2f} | {r['z']:+.1f} | {r['pvalue']:.3g} | "
                     f"{r['note']} |")
        L.append("")
        if any(r.get("decoder", "pymatching") == "pymatching"
               for r in state.decoder_table):
            L += ["Read the *as given* rows, not the *weight <= 2* rows: "
                  "pymatching drops hyperedges either way, so the two usually "
                  "decode identically, and restricting the model before "
                  "predicting deletes the events responsible for most of the "
                  "failures, making the prediction optimistic and the ratio "
                  "meaningless.", ""]
        hl = _headline_ler_row(state.decoder_table)
        if hl is not None:
            L += [f"The `logical error rate` row in the per-family table "
                  f"above is the {hl.get('decoder', 'pymatching')} "
                  f"*{cl}, as given* row of this table.", ""]

    if state.stationarity:
        L += ["## Stationarity of the data (tests the data, not the model)", "",
              "| test | p | effect |", "|---|---|---|"]
        for r in state.stationarity:
            L.append(f"| {r.name} | {r.pvalue:.3g} | {r.effect_description} |")
        L.append("")

    L += [
        "## How to read this", "",
        "- **Compare families, not raw counts.** With this many shots almost "
        "any real discrepancy is significant. The story is *which* families "
        "reject and at what effect size.",
        "- **A rejected scalar with a tiny effect is not a modelling error**, "
        "it is the power of the test; the effect descriptions give the gap in "
        "units of the null standard deviation.",
        "- **A passing logical-error-rate row is weak evidence, a failing one "
        "is strong.** It is one scalar test against thousands of moment "
        "tests, so it has little power.",
        "- **Non-stationary data puts a floor under any static DEM.** If the "
        "stationarity battery rejects, some residual misfit is the data, not "
        "the model.",
        "- **A model can match every low-order statistic and still be a bad "
        "decoding prior**, and vice versa. Statistical fit and decoding "
        "quality are different objectives; say which one you are talking "
        "about.",
        "",
        "## What to return", "",
        "A single JSON object, no prose around it, with these keys "
        "(all values Markdown; `sections` keys are optional, omit any you "
        "have nothing to say about):", "",
        "```json",
        '{"summary": "...", "sections": {'
        + ", ".join(f'"{k}": "..."' for k in _COMMENTARY_SLOTS)
        + '}, "reading": ["...", "..."], "caveats": ["..."]}',
        "```", "",
        "`summary` is the box at the top of the report: what was found, in "
        "3-6 sentences, with the numbers in it. Each `sections` entry is a "
        "short note (1-3 paragraphs) placed under that section's heading, "
        "above its figures -- it should say what the reader is about to see "
        "and what it means, not restate the table. `reading` is the closing "
        "list of conclusions, each bullet leading with a bold claim followed "
        "by the evidence for it. Quote real numbers throughout; never invent "
        "one that is not above. The text is rendered as HTML with raw markup "
        "passed through, so write comparisons in words or backticks "
        "(`p < 0.05`) rather than with a bare < sign.",
    ]
    return "\n".join(L)


def generate_report(cfg: ReportInputs, log=print) -> ReportResult:
    """
    Run the battery and write the HTML report. See `ReportInputs`.

    Returns:
        result: ReportResult
            `.html_path`, `.summary` (per-candidate counts and headline
            numbers), `.results` (raw ValidationResults) and `.decoder_table`.
    """
    plt.rcParams.update(_STYLE)
    det = np.asarray(cfg.detector_samples, dtype=np.uint8) % 2
    if det.ndim != 2 or det.shape[0] == 0:
        raise ValueError("detector_samples must be a non-empty 2D array.")
    n_det = det.shape[1]
    n_shots = det.shape[0]
    log(f"{n_shots} shots x {n_det} detectors, mean click rate {det.mean():.4f}")

    coords = {}
    if cfg.circuit is not None and cfg.coordinate_mode != "none":
        coords = canonical_coordinates(cfg.circuit, cfg.coordinate_mode,
                                       cfg.coordinate_fn)
        if coords:
            log(f"coordinates: {len({(v[0], v[1]) for v in coords.values()})} "
                f"sites x {len({v[2] for v in coords.values()})} rounds")

    learned_dict = sdio.dem_to_dict(cfg.learned_dem)
    baseline_dict = (sdio.dem_to_dict(cfg.baseline_dem)
                     if cfg.baseline_dem is not None else None)
    log(f"learned DEM: {len(learned_dict)} events"
        + (f"; {cfg.baseline_label}: {len(baseline_dict)} merged events"
           if baseline_dict else ""))

    subsets, distant_min_distance = build_subsets(cfg, coords, learned_dict,
                                                  n_det, log=log)

    extra_scalars = {}
    rc = repeated_clicks_statistic(coords)
    if rc is not None:
        extra_scalars["repeated_clicks"] = rc

    candidates = [("learned", cfg.learned_dem, learned_dict)]
    if baseline_dict is not None:
        candidates.append(("baseline", cfg.baseline_dem, baseline_dict))

    per_candidate = {}
    for key, dem, dem_dict in candidates:
        per_candidate[key] = run_battery(cfg, key, dem, dem_dict, subsets,
                                         extra_scalars, log=log)

    decoder_scalar_data = {}
    if "decoder_scalars" not in cfg.skip and cfg.decoder_scalar_tests:
        for key, dem, dem_dict in candidates:
            # The gap test needs L0 flags: the candidate's live on the
            # decorated DEM, a circuit-derived baseline carries its own.
            decorated = cfg.decorated_dem if key == "learned" else dem
            res, fig_data = decoder_scalar_battery(cfg, key, dem_dict,
                                                   decorated, n_det, log=log)
            per_candidate[key].extend(res)
            decoder_scalar_data[key] = fig_data

    decoder_table, ler_result, drop_info = ([], None, (0, 0.0))
    if "decoder" not in cfg.skip and cfg.decorated_dem is not None:
        decoder_table, ler_result, drop_info = decoder_comparison(
            cfg, dem_with_observables(cfg.decorated_dem), log=log)
        if ler_result is not None:
            per_candidate["learned"].append(ler_result)

    stationarity = None
    if "stationarity" not in cfg.skip:
        stationarity = run_stationarity_battery(det, seed=cfg.seed)
        log("stationarity battery done")

    fams = {key: collections.defaultdict(list) for key in per_candidate}
    for key, results in per_candidate.items():
        for r in results:
            fams[key][r.name.split(":")[1].split("[")[0]].append(r)
    rejected_names = {
        key: {r.name for r in
              ValidationSuiteResult(results).rejected(alpha=cfg.alpha)}
        for key, results in per_candidate.items()}

    # --- null samples for the distribution figures ---------------------------
    log(f"sampling {cfg.null_shots} null shots per model ...")
    null_learned = sample_detectors(cfg.learned_dem, cfg.null_shots, n_det,
                                    cfg.seed)
    null_baseline = (sample_detectors(cfg.baseline_dem, cfg.null_shots, n_det,
                                      cfg.seed)
                     if cfg.baseline_dem is not None else None)

    # --- figures -------------------------------------------------------------
    cl = cfg.candidate_label
    figures = {}
    figures["rejections"] = figure_rejections(fams, rejected_names,
                                              ALL_FAMILIES, cfg.baseline_label,
                                              cl)
    figures["weights"] = figure_weights(learned_dict, baseline_dict,
                                        cfg.baseline_label, cl)
    event_masks = list(learned_dict)
    event_probs = np.array([learned_dict[m] for m in event_masks], dtype=float)
    event_stderr = None
    if cfg.event_stderr:
        event_stderr = np.array([cfg.event_stderr.get(int(m), 0.0)
                                 for m in event_masks], dtype=float)
    figures["probs"] = figure_probabilities(event_masks, event_probs,
                                            event_stderr, cl)
    figures["polz"] = figure_polarization(fams, cfg.baseline_label, cl)
    figures["tvd"] = figure_marginal_tvd(fams, MARGINAL_FAMILIES,
                                         cfg.baseline_label, cl)
    figures["balls"] = figure_ball_structure(fams, coords, cfg.baseline_label,
                                             cl)
    figures["pvals"] = figure_pvalue_ecdf(fams, MARGINAL_FAMILIES
                                          + ["polarization_w2"], cl)
    statistics = {"Hamming weight (clicks per shot)": lambda s: s.sum(axis=1)}
    if rc is not None:
        statistics["Repeated clicks (same site, adjacent rounds)"] = rc
    figures["scalars"] = figure_scalars(det, null_learned, null_baseline,
                                        statistics, cfg.baseline_label, cl)
    figures["decoder_scalars"] = figure_decoder_scalars(
        decoder_scalar_data, cfg.baseline_label, cl)
    figures["clickrates"] = figure_click_rates(det, null_learned, null_baseline,
                                               cfg.baseline_label, cl)
    figures["edges_decoder"] = figure_edges_and_decoder(
        learned_dict, baseline_dict, coords, decoder_table,
        cfg.baseline_label, cl)
    if stationarity is not None:
        figures["stationarity"] = figure_stationarity(det, stationarity.results)
    log(f"{sum(1 for v in figures.values() if v)} figures rendered")

    # --- summary numbers ------------------------------------------------------
    summary = {}
    for key, results in per_candidate.items():
        summary[key] = {"num_tests": len(results),
                        "num_rejected": len(rejected_names[key])}
    summary["num_events"] = len(learned_dict)
    summary["total_error_mass"] = float(sum(learned_dict.values()))
    summary["decoder_table"] = decoder_table
    if ler_result is not None:
        summary["ler_observed"] = ler_result.details["ler_observed"]
        summary["ler_predicted"] = ler_result.details["ler_predicted"]
        summary["ler_ratio"] = float(ler_result.effect_size)
        summary["ler_z"] = float(ler_result.details["z"])
        summary["ler_pvalue"] = float(ler_result.pvalue)
        summary["ler_decoder"] = ler_result.details["decoder"]

    # --- what the analyst needs that the figures do not carry ---------------
    cand_counts, cand_mass = _weight_stats(learned_dict)
    model_stats = {
        "weight_counts": dict(cand_counts), "weight_mass": dict(cand_mass),
        "observed_click_rate": float(det.mean()),
        "observed_mean_clicks": float(det.sum(axis=1).mean()),
        "candidate_mean_clicks": float(null_learned.sum(axis=1).mean()),
        "max_click_rate_gap": float(np.abs(det.mean(0)
                                           - null_learned.mean(0)).max()),
    }
    if baseline_dict is not None:
        b_counts, b_mass = _weight_stats(baseline_dict)
        model_stats.update(
            baseline_weight_counts=dict(b_counts),
            baseline_weight_mass=dict(b_mass),
            baseline_total_mass=float(sum(baseline_dict.values())),
            baseline_mean_clicks=float(null_baseline.sum(axis=1).mean()),
            baseline_max_click_rate_gap=float(
                np.abs(det.mean(0) - null_baseline.mean(0)).max()))

    state = RenderState(
        cfg=_strip_arrays(cfg), summary=summary,
        fams={k: {f: [_slim_result(r) for r in v] for f, v in d.items()}
              for k, d in fams.items()},
        rejected_names=rejected_names, figures=figures,
        decoder_table=decoder_table, drop_info=drop_info,
        stationarity=[_slim_result(r) for r in
                      (stationarity.results if stationarity else [])],
        coords=coords, distant_min_distance=distant_min_distance,
        n_shots=n_shots, n_det=n_det, model_stats=model_stats)

    out = Path(cfg.output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(_render_html(state))
    log(f"wrote {out} ({out.stat().st_size / 1e6:.2f} MB)")

    brief_path = _default_sidecar(cfg.brief_path, out, "_brief.md")
    if brief_path is not None:
        brief_path.write_text(commentary_brief(state))
        log(f"wrote {brief_path} (hand this to the commentary analyst)")
    state_path = _default_sidecar(cfg.state_path, out, "_state.pkl")
    if state_path is not None:
        with open(state_path, "wb") as f:
            pickle.dump(state, f)
        log(f"wrote {state_path} "
            f"({state_path.stat().st_size / 1e6:.2f} MB; re-render with "
            f"annotate_report.py, no recomputation)")

    if cfg.results_path is not None:
        slim = {k: [_slim_result(r) for r in v]
                for k, v in per_candidate.items()}
        with open(cfg.results_path, "wb") as f:
            pickle.dump({"per_candidate": slim,
                         "stationarity": [_slim_result(r) for r in
                                          (stationarity.results
                                           if stationarity else [])],
                         "summary": summary}, f)
        log(f"wrote {cfg.results_path} "
            f"({Path(cfg.results_path).stat().st_size / 1e6:.1f} MB)")

    return ReportResult(html_path=out, summary=summary,
                        results=per_candidate, decoder_table=decoder_table,
                        figures=figures, brief_path=brief_path,
                        state_path=state_path, state=state)



# ===========================================================================
# The report itself
# ===========================================================================

class _Sections:
    """Numbered <h2> headings, so sections can be present or absent."""

    def __init__(self):
        self.n = 0

    def __call__(self, title: str) -> str:
        self.n += 1
        return f"<h2>{self.n}. {title}</h2>"


def _headline_ler_row(decoder_table) -> Optional[dict]:
    """
    The candidate-as-given row the report headlines: the hyperedge-native
    decoder's when one ran, else the matcher's. Mirrors the preference in
    `decoder_comparison`, but works from the table so that re-renders of old
    states agree with the KPI band.
    """
    rows = [r for r in decoder_table if r["kind"] == "candidate"
            and not r.get("restricted", "weight" in r["label"])]
    for r in rows:
        if r.get("decoder", "pymatching") != "pymatching":
            return r
    return rows[0] if rows else None


def _battery_ler_row(fams, decoder_table) -> Optional[dict]:
    """
    The decoder-table row whose result actually joined the validation
    battery, read off the battery itself (the test name carries the decoder).
    Falls back to `_headline_ler_row` when the battery has no LER row.
    """
    fam = fams.get("learned", {}).get("logical_error_rate") or []
    if fam:
        dec = fam[0].name.split("[")[-1].rstrip("]")
        for r in decoder_table:
            if (r["kind"] == "candidate"
                    and r.get("decoder", "pymatching") == dec
                    and not r.get("restricted", "weight" in r["label"])):
                return r
    return _headline_ler_row(decoder_table)


def _decoder_table_html(decoder_table, alpha, cand_label) -> str:
    """The observed-vs-predicted table, with a decoder column when it earns one."""
    multi = len({r.get("decoder", "pymatching") for r in decoder_table}) > 1
    head = ("<tr>" + ("<th>decoder</th>" if multi else "")
            + "<th>model handed to the decoder</th>"
            "<th class='num'>observed LER</th>"
            "<th class='num'>predicted LER</th><th class='num'>ratio</th>"
            "<th class='num'>z</th></tr>")
    rows = []
    for r in decoder_table:
        se = r.get("predicted_stderr") or 0.0
        pred = _fmt_rate(r["predicted"]) + (
            f" <small>&plusmn; {_fmt_rate(se)}</small>" if se > 0 else "")
        rows.append(
            "<tr>"
            + (f"<td>{r.get('decoder', 'pymatching')}</td>" if multi else "")
            + f"<td>{r['label']}"
            + (f" &mdash; {r['note']}" if r["note"] else "")
            + "</td>"
            + f"<td class='num'><b>{_fmt_rate(r['observed'])}</b></td>"
            + f"<td class='num'>{pred}</td>"
            + f"<td class='num{' bad' if r['pvalue'] < alpha else ' good'}'>"
              f"{r['ratio']:.2f}&times;</td>"
            + f"<td class='num'>{r['z']:+.1f}</td></tr>")
    return (f"<table>{head}\n" + "\n".join(rows) + "</table>"
            "<p style='font-size:.8rem;color:#666'>The ratio is "
            "observed&nbsp;/&nbsp;predicted, coloured by the test verdict at "
            f"&alpha;&nbsp;=&nbsp;{alpha} (green: consistent, red: rejected); "
            "z is the gap in standard errors; &plusmn; is the Monte Carlo "
            "standard error of the prediction.</p>")


def _render_html(state: RenderState) -> str:
    """
    Assemble the report: numbered narrative sections, each opening with the
    analyst's note (when there is one) and then the evidence.

    The generated prose sticks to what was measured. Everything interpretive
    comes from `cfg.commentary`; without it the report is still complete, just
    silent about what the numbers mean.
    """
    cfg = state.cfg
    summary, fams, figures = state.summary, state.fams, state.figures
    rejected_names, decoder_table = state.rejected_names, state.decoder_table
    n_shots, n_det = state.n_shots, state.n_det
    has_base = "baseline" in summary
    bl, cl = cfg.baseline_label, cfg.candidate_label
    com = cfg.commentary or {}
    note = lambda slot: _commentary_block(com, slot)  # noqa: E731
    h2 = _Sections()

    n_distant = len(fams["learned"].get("distant", ()))
    if "marginals" in cfg.skip:
        distant_note = "Marginal tests were skipped."
    elif state.distant_min_distance is None:
        distant_note = ("No satisfiable min_distance: the graph-distant family "
                        "was skipped.")
    else:
        reduced = (state.distant_min_distance != cfg.distant_min_distance)
        distant_note = (
            f"The graph-distant family used min_distance = "
            f"{state.distant_min_distance}"
            + (f" (reduced from {cfg.distant_min_distance}: the {cl} graph "
               f"has no subsets that far apart)" if reduced else ""))
        if n_distant < cfg.num_distant_subsets:
            distant_note += (
                f", yielding {n_distant} of the {cfg.num_distant_subsets} "
                f"subsets requested.")
        else:
            distant_note += "."

    parts = [f"""<!doctype html><html lang="en"><head><meta charset="utf-8" />
<title>{cfg.title}</title><style>{_CSS}</style></head><body>
<h1>{cfg.title}</h1>
<p class="sub">{cfg.subtitle or ''}</p>
<div class="kpi">
 <div><b>{summary['num_events']}</b><span>events in the {cl} model</span></div>
 <div><b>{summary['total_error_mass']:.2f}</b><span>error mass per shot
   (expected error events)</span></div>
 <div><b>{summary['learned']['num_rejected']} / {summary['learned']['num_tests']}</b>
   <span>validation tests rejected ({cl})</span></div>"""]
    if has_base:
        parts.append(
            f" <div><b>{summary['baseline']['num_rejected']} / "
            f"{summary['baseline']['num_tests']}</b>"
            f"<span>validation tests rejected ({bl})</span></div>")
    kpi_ler = _battery_ler_row(fams, decoder_table)
    if kpi_ler is not None:
        dec = kpi_ler.get("decoder", "pymatching")
        parts.append(
            f" <div><b>{_fmt_rate(kpi_ler['observed'])}</b>"
            f"<span>observed logical error rate ({cl} model, {dec})</span></div>"
            f" <div><b>{_fmt_rate(kpi_ler['predicted'])}</b>"
            f"<span>logical error rate the {cl} model predicts for "
            f"itself ({dec})</span></div>")
    parts.append("</div>")

    headline = md_to_html(com.get("summary", "")) or cfg.findings_html
    if headline:
        # "Summary." belongs inside the first paragraph, not floating above it.
        lead = "<b>Summary.</b> "
        headline = (headline.replace("<p>", "<p>" + lead, 1)
                    if headline.startswith("<p>") else lead + headline)
        parts.append(f'<div class="box">{headline}</div>')

    parts.append(f"""
<h2>Setup</h2>
<p>{n_shots:,} shots &times; {n_det} detectors, mean click rate
{state.model_stats.get('observed_click_rate', float('nan')):.4f}. Rejections
are Benjamini&ndash;Hochberg FDR at &alpha;&nbsp;=&nbsp;{cfg.alpha}, computed
separately per model.
{'A reference model (<b>' + bl + '</b>) was run through the identical battery: at this many shots absolute p-values saturate, so the comparison carries the information.' if has_base else 'No reference model was supplied; absolute p-values at this many shots are dominated by statistical power, so read the effect sizes rather than the rejection counts.'}
{'Marginal subsets come from the circuit&rsquo;s detector coordinates (spacetime balls of spatial radius ' + str(cfg.space_radius) + ' and time radius ' + str(cfg.time_radius) + ', whole-round space slices, per-site time columns) plus random low-weight and graph-distant subsets.' if state.coords else 'No usable detector coordinates: marginal subsets are detector-graph neighbourhoods plus random and graph-distant subsets.'}
{distant_note}
Exhaustive weight-k families are deliberately not run: at {n_det} detectors
there are far too many.</p>
""")

    if cfg.pipeline_html:
        # getattr: state pickles written before this field existed.
        parts.append(h2(getattr(cfg, "pipeline_title",
                                "How the model was built")))
        parts.append(cfg.pipeline_html)

    # --- the model -----------------------------------------------------------
    parts.append(h2(f"The {cl} model"))
    parts.append(note("model"))
    parts.append(_fig_html(
        figures.get("weights"),
        "Event count and error mass by detector weight. Hyperedges "
        "(weight&nbsp;&ge;&nbsp;3) matter twice over: they are what a "
        "matching decoder cannot use, and they are where a lattice search's "
        "pruning threshold bites hardest."))
    parts.append(_fig_html(
        figures.get("probs"),
        "Distribution of the fitted event probabilities by weight."))
    parts.append(_fig_html(
        figures.get("clickrates"),
        "Per-detector click rates, model against data. A DEM fitted to this "
        "data lands on the diagonal by construction, so departures mean the "
        "fit did not converge or the event set is over-complete; for a model "
        "that was not fitted to it, any departure is a real discrepancy."))

    # --- validation ----------------------------------------------------------
    parts.append(h2("Validation"))
    parts.append(note("validation"))
    parts.append(_fig_html(
        figures.get("rejections"),
        "Fraction of tests rejected per family. Families with zero rejections "
        "are the ones the model reproduces to sampling accuracy."))

    parts.append("<h3>Moment (Walsh polarization) tests</h3>")
    parts.append(note("polarization"))
    parts.append(_family_table(fams, rejected_names, POLARIZATION_FAMILIES, bl,
                               "z-scores of observed vs predicted polarization",
                               cl))
    parts.append(_fig_html(
        figures.get("polz"),
        "Polarization z-scores against a standard normal. A correct model "
        "puts these on N(0,1); systematic offsets mean the model has the "
        "wrong error rates on those masks."))

    parts.append("<h3>Marginal likelihood (G) tests</h3>")
    parts.append(note("marginals"))
    parts.append(_family_table(fams, rejected_names, MARGINAL_FAMILIES, bl,
                               "total variation distances", cl))
    parts.append(_fig_html(
        figures.get("tvd"),
        "Total variation distance between model and empirical marginals on "
        "each subset family, log scale."))
    parts.append(_fig_html(
        figures.get("balls"),
        "How the misfit depends on the order of the marginal (left) and on "
        "where in the experiment it is measured (right). A model that is "
        "right pairwise but wrong jointly shows a rising left panel and a "
        "flat right one."))
    parts.append(_fig_html(
        figures.get("pvals"),
        "p-value calibration. A correct model traces the diagonal."))

    if figures.get("scalars") or fams["learned"].get("scalar"):
        parts.append("<h3>Scalar statistics</h3>")
        parts.append(note("scalars"))
        scalar_rows = []
        for key, lbl in [("learned", cl)] + ([("baseline", bl)]
                                             if has_base else []):
            for r in fams.get(key, {}).get("scalar", []):
                rej = r.name in rejected_names.get(key, set())
                scalar_rows.append(
                    f"<tr><td>{lbl}</td>"
                    f"<td><code>{r.name.split(':', 1)[-1]}</code></td>"
                    f"<td class='num{' bad' if rej else ' good'}'>"
                    f"{r.pvalue:.3g}</td>"
                    f"<td>{r.effect_description}</td></tr>")
        if scalar_rows:
            parts.append(
                "<table><tr><th>model</th><th>test</th>"
                "<th class='num'>p</th><th>effect</th></tr>"
                + "\n".join(scalar_rows) + "</table>")
        parts.append(_fig_html(
            figures.get("scalars"),
            f"Scalar summary distributions against {cfg.null_shots:,} shots "
            "sampled from each model. These catch aggregate misfit that "
            "per-mask tests average away."))
        parts.append(_fig_html(
            figures.get("decoder_scalars"),
            "Decoder-derived scalars against each model's own Monte Carlo. "
            "The matching weight is the cost of the decoder's best "
            "correction for each shot; the complementary gap is the extra "
            "cost of the best correction in the opposite logical class "
            "(decoder confidence). These test the model as a decoding "
            "prior: it can match every moment and marginal above and still "
            "put the wrong distribution here."))

    # --- decoder -------------------------------------------------------------
    if decoder_table:
        n_drop, mass = state.drop_info
        decoders = {r.get("decoder", "pymatching") for r in decoder_table}
        parts.append(h2("Decoder consistency"))
        parts.append(note("decoder"))
        parts.append(f"""
<p>A DEM can match every low-order statistic of the data and still be a poor
decoding prior. This section separates the two. {n_drop} of the {cl} model's
events are not graph-like ({mass:.3f} total probability) and are dropped for
matching{'; the same restriction is applied to the reference model so the comparison is controlled &mdash; if the reference barely degrades but the ' + cl + ' model does, its graph itself is the problem, not the missing hyperedges' if has_base else ''}.</p>
<p>Each graph is scored twice: the logical error rate its decoder achieves on
the {n_shots:,} experimental shots, and the rate the same graph predicts for
itself &mdash; Monte Carlo shots drawn from it, decoded the same way.
Observed alone cannot distinguish a wrong model from a hard experiment; the
pair can. For a learned model it is also the end-to-end check on the logical
decoration, since the L0 flags decide what counts as a failure on both sides:
a mis-assigned flag moves the observed rate and leaves the prediction where it
was.</p>""")
        if "pymatching" in decoders:
            parts.append("""
<p>Read the <em>as given</em> and <em>weight&nbsp;&le;&nbsp;2</em> matching
rows as a pair. pymatching silently discards any error it cannot make an edge
of, so the two usually decode identically &mdash; what differs is what the
prediction was computed from. Restricting the model first throws away the
events responsible for most of the failures, so its prediction is optimistic
and the ratio blows up even for a model that is exactly right. The
<em>as given</em> rows are the honest comparison.</p>""")
        if "tesseract" in decoders:
            parts.append(f"""
<p>The tesseract rows decode the same models with a most-likely-error
decoder, which uses the {n_drop} hyperedges instead of discarding them.
Comparing a model's matching row with its tesseract row measures directly what
the graph-like restriction costs, rather than inferring it from the reference
model.</p>""")
        parts.append(_decoder_table_html(decoder_table, cfg.alpha, cl))
        hl = _battery_ler_row(fams, decoder_table)
        if hl is not None:
            parts.append(
                f"<p>The <b>{hl.get('decoder', 'pymatching')}</b> "
                f"<em>{cl}, as given</em> row is the one that enters the "
                f"validation battery and the headline numbers above"
                + (": a decoder that uses every event tests the model, "
                   "not its graph-like restriction.</p>"
                   if hl.get("decoder", "pymatching") != "pymatching"
                   else ".</p>"))
        parts.append(_fig_html(
            figures.get("edges_decoder"),
            "Left: weight-2 error mass by edge range. Excess mass on "
            "mid-range edges gives a matcher cheap detour paths and cuts the "
            "effective code distance, even when every individual edge "
            "probability is correct. Right: observed against self-predicted "
            "logical error rate for each decoding graph."))
    elif figures.get("edges_decoder"):
        parts.append(h2("Edge structure"))
        parts.append(_fig_html(figures["edges_decoder"],
                               "Weight-2 error mass by edge range."))

    # --- stationarity --------------------------------------------------------
    if state.stationarity:
        srows = "\n".join(
            f"<tr><td>{r.name}</td><td class='num'>{r.pvalue:.3g}</td>"
            f"<td>{r.effect_description}</td></tr>"
            for r in state.stationarity)
        n_rej = len(ValidationSuiteResult(state.stationarity)
                    .rejected(alpha=cfg.alpha))
        parts.append(h2("Stationarity of the data"))
        parts.append(note("stationarity"))
        parts.append(f"""
<p>These test the data, not the model: the null is that shots are i.i.d.
{n_rej} of {len(state.stationarity)} reject. Non-stationary data puts a
floor under how well <em>any</em> single static DEM can fit, so a residual
misfit above should be read against this.</p>
<table><tr><th>test</th><th class="num">p</th><th>effect</th></tr>
{srows}</table>
{_fig_html(figures.get('stationarity'), 'Mean clicks per shot across the run, with a linear drift fit.')}
""")

    # --- conclusions ---------------------------------------------------------
    reading = com.get("reading") or []
    caveats = com.get("caveats") or []
    if reading or caveats:
        parts.append(h2("Reading of the results"))
        if reading:
            parts.append("<ul>" + "".join(f"<li>{_md_inline(b)}</li>"
                                          for b in reading) + "</ul>")
        if caveats:
            parts.append('<div class="box"><b>Caveats.</b><ul>'
                         + "".join(f"<li>{_md_inline(c)}</li>" for c in caveats)
                         + "</ul></div>")

    if cfg.artifacts:
        rows = "\n".join(f"<tr><td><code>{name}</code></td><td>{what}</td></tr>"
                         for name, what in cfg.artifacts)
        parts.append("<h2>Artifacts</h2>"
                     f"<table><tr><th>file</th><th>contents</th></tr>{rows}"
                     "</table>")

    parts.append("</body></html>")
    return "\n".join(p for p in parts if p)

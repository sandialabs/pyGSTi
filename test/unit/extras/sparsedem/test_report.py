"""
Tests for pygsti.extras.sparsedem.report, the validation-report engine.

Two layers:

  * unit tests for the pure helpers (coordinate canonicalization, DEM
    conversions, the matcher-ignored census, the tiny Markdown renderer,
    commentary parsing);
  * one end-to-end run of `generate_report` on data sampled from the very
    circuit the candidate model was derived from, shared module-wide. On
    self-sampled data every rejection is a false positive, so the run doubles
    as a calibration check of the whole battery, and its artifacts back the
    HTML/brief/state round-trip tests.
"""

import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import stim

from pygsti.extras.sparsedem.report import (
    COMMENTARY_SCHEMA,
    ReportInputs,
    _default_sidecar,
    _fmt_rate,
    _md_inline,
    _render_html,
    _slim_result,
    _weight_stats,
    canonical_coordinates,
    coordinate_circuit,
    dem_with_observables,
    flat_dem,
    generate_report,
    graphlike_dem,
    load_commentary,
    matcher_ignored,
    md_to_html,
    repeated_clicks_statistic,
    sample_detectors,
)

pymatching = pytest.importorskip("pymatching")


# ===========================================================================
# Coordinates
# ===========================================================================

def test_canonical_coordinates_stim_mode():
    circuit = stim.Circuit.generated(
        "repetition_code:memory", distance=3, rounds=3,
        after_clifford_depolarization=0.01)
    coords = canonical_coordinates(circuit, mode="stim")
    assert set(coords) == set(range(circuit.num_detectors))
    for triple in coords.values():
        assert len(triple) == 3
    # 2-entry stim coordinates are (x, t): y is filled with 0.
    raw = circuit.get_detector_coordinates()
    d = next(iter(raw))
    assert coords[d] == (raw[d][0], 0.0, raw[d][-1])


def test_canonical_coordinates_google_mode():
    circuit = stim.Circuit("DETECTOR(1,2,0) rec[-1]\n"
                           "DETECTOR(0,0,0,3,4,1) rec[-1]")
    coords = canonical_coordinates(circuit, mode="google")
    # Last triple's site, first triple's time.
    assert coords[0] == (1.0, 2.0, 0.0)
    assert coords[1] == (3.0, 4.0, 0.0)
    # Ragged lengths force google mode under "auto" too.
    assert canonical_coordinates(circuit, mode="auto") == coords


def test_canonical_coordinates_errors_and_none():
    ragged = stim.Circuit("DETECTOR(1,2) rec[-1]\nDETECTOR(1,2,3) rec[-1]")
    with pytest.raises(ValueError, match="uniform"):
        canonical_coordinates(ragged, mode="stim")
    not_triples = stim.Circuit("DETECTOR(1,2,3,4) rec[-1]")
    with pytest.raises(ValueError, match="triples"):
        canonical_coordinates(not_triples, mode="google")
    with pytest.raises(ValueError, match="unknown"):
        canonical_coordinates(not_triples, mode="martian")
    assert canonical_coordinates(not_triples, mode="none") == {}
    assert canonical_coordinates(stim.Circuit(), mode="auto") == {}


def test_canonical_coordinates_fn_override():
    circuit = stim.Circuit("DETECTOR(7,8,9,10) rec[-1]")
    coords = canonical_coordinates(circuit, mode="google",
                                   coordinate_fn=lambda c: (c[0], c[1], c[3]))
    assert coords == {0: (7.0, 8.0, 10.0)}


def test_coordinate_circuit_round_trip():
    coords = {0: (1.0, 2.0, 3.0), 2: (4.0, 5.0, 6.0)}
    circuit = coordinate_circuit(coords, num_detectors=3)
    raw = circuit.get_detector_coordinates()
    assert raw[0] == [1.0, 2.0, 3.0]
    assert raw[1] == [0.0, 0.0, 0.0]  # missing detectors get the origin
    assert raw[2] == [4.0, 5.0, 6.0]


# ===========================================================================
# DEM helpers
# ===========================================================================

def test_dem_with_observables_merges_and_xors():
    dem = stim.DetectorErrorModel("""
        error(0.1) D0 D1 L0
        error(0.2) D0 D1 L0
        error(0.25) D0 ^ D1 D2
    """)
    merged = dem_with_observables(dem)
    # Independent duplicates combine as independent flips.
    assert merged[0b011][0] == pytest.approx(0.1 * 0.8 + 0.2 * 0.9)
    assert merged[0b011][1] == 1
    # A decomposed error is one event spanning both components.
    assert merged[0b111] == (pytest.approx(0.25), 0)


def test_flat_dem_spans_and_clamps():
    dem = flat_dem({0b101: 0.1, 0b010: 2.0}, num_detectors=5)
    assert dem.num_detectors == 5
    probs = {tuple(sorted(t.val for t in inst.targets_copy())):
             inst.args_copy()[0]
             for inst in dem.flattened() if inst.type == "error"}
    assert probs[(0, 2)] == pytest.approx(0.1)
    assert probs[(1,)] == pytest.approx(0.5, abs=1e-10)  # clamped below 0.5


def test_graphlike_dem_drops_hyperedges():
    events = {0b011: (0.1, 1), 0b111: (0.2, 0), 0b1: (0.05, 0)}
    dem, dropped, mass = graphlike_dem(events)
    assert dropped == 1
    assert mass == pytest.approx(0.2)
    text = str(dem)
    assert "L0" in text            # the observable survives on the w2 event
    assert "D0 D1 D2" not in text


def test_sample_detectors_pads():
    dem = stim.DetectorErrorModel("error(0.5) D0")
    det = sample_detectors(dem, num_shots=100, num_detectors=4, seed=0)
    assert det.shape == (100, 4)
    assert det[:, 1:].sum() == 0
    assert 0 < det[:, 0].sum() < 100


def test_matcher_ignored():
    graphlike = stim.DetectorErrorModel("error(0.1) D0 D1\nerror(0.1) D2")
    assert matcher_ignored(graphlike) == (0, 0.0)
    hyper = stim.DetectorErrorModel("error(0.125) D0 D1 D2\n"
                                    "error(0.1) D0 D1 ^ D2 D3")
    n, mass = matcher_ignored(hyper)
    assert n == 1                  # the decomposed one is two fine edges
    assert mass == pytest.approx(0.125)


def test_repeated_clicks_statistic():
    coords = {0: (0.0, 0.0, 0.0), 1: (0.0, 0.0, 1.0), 2: (1.0, 0.0, 0.0)}
    stat = repeated_clicks_statistic(coords)
    samples = np.array([[1, 1, 1],    # detectors 0,1 same site, dt=1: counts
                        [1, 0, 1],
                        [0, 0, 0]], dtype=np.uint8)
    assert list(stat(samples)) == [1, 0, 0]
    # No same-site adjacent-round pair -> no statistic.
    assert repeated_clicks_statistic({0: (0.0, 0.0, 0.0),
                                      1: (1.0, 0.0, 1.0)}) is None


def test_weight_stats_and_fmt_rate():
    count, mass = _weight_stats({0b1: 0.1, 0b11: 0.2, 0b101: 0.3})
    assert count == {1: 1, 2: 2}
    assert mass[2] == pytest.approx(0.5)
    assert _fmt_rate(0.1234) == "0.1234"
    assert "e-" in _fmt_rate(1e-5)


# ===========================================================================
# Markdown and commentary
# ===========================================================================

def test_md_inline():
    html = _md_inline("a **bold** *word* and `x < y & z`")
    assert "<b>bold</b>" in html
    assert "<em>word</em>" in html
    assert "<code>x &lt; y &amp; z</code>" in html


def test_md_to_html_blocks():
    html = md_to_html("Para one.\n\n- first\n- second\n\n"
                      "| a | b |\n|---|---|\n| 1 | 2 |")
    assert "<p>Para one.</p>" in html
    assert "<li>first</li>" in html and "<li>second</li>" in html
    assert "<th>a</th>" in html and "<td>2</td>" in html
    assert md_to_html("") == ""


def test_load_commentary(tmp_path):
    path = tmp_path / "commentary.json"
    path.write_text('{"summary": "fine", "reading": ["x"]}')
    assert load_commentary(path)["summary"] == "fine"
    # A ```json fence is tolerated.
    path.write_text('```json\n{"summary": "fenced"}\n```')
    assert load_commentary(path)["summary"] == "fenced"
    # Unknown keys warn, non-objects raise.
    path.write_text('{"summary": "s", "conclusion": "nope"}')
    with pytest.warns(UserWarning, match="conclusion"):
        load_commentary(path)
    path.write_text('["not", "an", "object"]')
    with pytest.raises(ValueError, match="expected a JSON object"):
        load_commentary(path)


def test_commentary_schema_slots():
    # The schema handed to the analyst and the render slots must agree.
    from pygsti.extras.sparsedem.report import _COMMENTARY_SLOTS
    assert set(COMMENTARY_SCHEMA["sections"]) == set(_COMMENTARY_SLOTS)


def test_default_sidecar(tmp_path):
    out = tmp_path / "report.html"
    assert _default_sidecar(None, out, "_brief.md") == \
        tmp_path / "report_brief.md"
    assert _default_sidecar(False, out, "_brief.md") is None
    assert _default_sidecar(tmp_path / "b.md", out, "_brief.md") == \
        tmp_path / "b.md"


# ===========================================================================
# End to end, on self-sampled data
# ===========================================================================

NUM_SHOTS = 4000
SEED = 7


@pytest.fixture(scope="module")
def report_run(tmp_path_factory):
    """
    One full generate_report run: repetition-code circuit, data sampled from
    that circuit, and the circuit's own DEM as both candidate and baseline.
    The model is correct by construction, so rejections calibrate the suite.
    """
    out_dir = tmp_path_factory.mktemp("report")
    circuit = stim.Circuit.generated(
        "repetition_code:memory", distance=3, rounds=3,
        after_clifford_depolarization=0.02,
        before_measure_flip_probability=0.01,
        after_reset_flip_probability=0.01)
    det, obs = circuit.compile_detector_sampler(seed=SEED).sample(
        NUM_SHOTS, separate_observables=True)
    dem = circuit.detector_error_model(decompose_errors=True)
    cfg = ReportInputs(
        detector_samples=np.asarray(det, dtype=np.uint8),
        observable_flips=np.asarray(obs, dtype=np.uint8)[:, 0],
        learned_dem=dem,
        decorated_dem=dem,
        baseline_dem=dem,
        baseline_label="circuit",
        candidate_label="ground truth",
        circuit=circuit,
        title="Self-sampled repetition code d=3",
        subtitle="calibration run: every rejection is a false positive",
        decoders=("pymatching",),
        num_mc_shots=4 * NUM_SHOTS,
        null_shots=4 * NUM_SHOTS,
        output_path=out_dir / "report.html",
        results_path=out_dir / "results.pkl",
        seed=SEED,
    )
    logs = []
    result = generate_report(cfg, log=logs.append)
    return cfg, result, logs


def test_report_files_written(report_run):
    cfg, result, _ = report_run
    assert result.html_path.exists()
    assert result.brief_path.exists()
    assert result.state_path.exists()
    assert cfg.results_path.exists()
    html = result.html_path.read_text()
    for heading in ("The ground truth model", "Validation",
                    "Decoder consistency", "Stationarity of the data"):
        assert heading in html
    assert html.count("data:image/png;base64,") >= 8


def test_report_summary_structure(report_run):
    _, result, _ = report_run
    s = result.summary
    for key in ("learned", "baseline"):
        assert s[key]["num_tests"] > 0
    assert s["num_events"] > 0
    assert s["total_error_mass"] > 0
    # The decoder section ran and produced the headline LER numbers.
    assert len(result.decoder_table) == 4  # 2 models x {as given, w<=2}
    assert 0 <= s["ler_observed"] <= 1
    assert s["ler_decoder"] == "pymatching"


def test_calibration_on_self_sampled_data(report_run):
    """A correct model must not be rejected: FDR-BH at alpha keeps the
    false-rejection count near zero under the global null."""
    _, result, _ = report_run
    for key in ("learned", "baseline"):
        s = result.summary[key]
        assert s["num_rejected"] <= max(2, int(0.01 * s["num_tests"])), \
            f"{key}: {s['num_rejected']} of {s['num_tests']} rejected on " \
            f"self-sampled data"


def test_brief_contains_the_numbers(report_run):
    _, result, _ = report_run
    brief = result.brief_path.read_text()
    assert "## Headline counts" in brief
    assert "## Per-family results" in brief
    assert "## Decoder" in brief
    assert "## What to return" in brief
    s = result.summary["learned"]
    assert f"**{s['num_rejected']} of {s['num_tests']}**" in brief


def test_state_round_trip(report_run):
    """The pickled state re-renders to exactly the HTML that was written."""
    _, result, _ = report_run
    with open(result.state_path, "rb") as f:
        state = pickle.load(f)
    assert _render_html(state) == result.html_path.read_text()


def test_annotate_round_trip(report_run):
    """Adding commentary re-renders every slot without recomputation."""
    _, result, _ = report_run
    with open(result.state_path, "rb") as f:
        state = pickle.load(f)
    state.cfg.commentary = {
        "summary": "All quiet on the **null** front.",
        "sections": {k: f"Note about {k}."
                     for k in COMMENTARY_SCHEMA["sections"]},
        "reading": ["**Calibrated.** Nothing rejected."],
        "caveats": ["Only a repetition code."],
    }
    html = _render_html(state)
    assert "All quiet on the <b>null</b> front." in html
    assert "Note about marginals." in html
    assert "<b>Calibrated.</b> Nothing rejected." in html
    assert "Only a repetition code." in html
    assert len(html) > len(result.html_path.read_text()) - 1000


def test_results_pickle_is_slim(report_run):
    cfg, result, _ = report_run
    with open(cfg.results_path, "rb") as f:
        payload = pickle.load(f)
    assert set(payload) == {"per_candidate", "stationarity", "summary"}
    for results in payload["per_candidate"].values():
        for r in results:
            for value in r.details.values():
                assert not (isinstance(value, np.ndarray)
                            and value.size > 4096)


def test_slim_result_drops_only_oversized(report_run):
    _, result, _ = report_run
    r = result.results["learned"][0]
    r.details["huge"] = np.zeros(10_000)
    slim = _slim_result(r)
    assert isinstance(slim.details["huge"], str)
    assert slim.name == r.name and slim.pvalue == r.pvalue
    del r.details["huge"]


def test_minimal_inputs(tmp_path):
    """No circuit, no observables, no baseline: the report still renders,
    falling back to detector-graph subsets."""
    dem = stim.DetectorErrorModel("""
        error(0.05) D0 D1
        error(0.04) D1 D2
        error(0.03) D2 D3
        error(0.02) D0
        error(0.02) D3
    """)
    det, _, _ = dem.compile_sampler(seed=3).sample(shots=800)
    logs = []
    result = generate_report(ReportInputs(
        detector_samples=np.asarray(det, dtype=np.uint8),
        learned_dem=dem,
        null_shots=2000,
        skip=("decoder", "decoder_scalars", "stationarity"),
        output_path=tmp_path / "minimal.html",
        brief_path=False,
        state_path=False,
        seed=3,
    ), log=logs.append)
    assert result.html_path.exists()
    assert result.brief_path is None and result.state_path is None
    assert "baseline" not in result.summary
    assert any("no detector coordinates" in line for line in logs)
    assert "No usable detector coordinates" in result.html_path.read_text()


def test_empty_samples_rejected():
    with pytest.raises(ValueError, match="non-empty"):
        generate_report(ReportInputs(
            detector_samples=np.zeros((0, 4), np.uint8),
            learned_dem=stim.DetectorErrorModel("error(0.1) D0")))


# ===========================================================================
# The skill's CLI wrappers
# ===========================================================================

_SCRIPTS = Path(__file__).resolve().parents[4] \
    / ".claude" / "skills" / "dem-validation-report" / "scripts"


@pytest.mark.skipif(not _SCRIPTS.exists(),
                    reason="skill scripts not present in this checkout")
def test_cli_wrappers(tmp_path):
    """run_report.py and annotate_report.py against files on disk."""
    circuit = stim.Circuit.generated(
        "repetition_code:memory", distance=3, rounds=2,
        after_clifford_depolarization=0.02)
    det = circuit.compile_detector_sampler(seed=1).sample(500)
    np.save(tmp_path / "det.npy", np.asarray(det, dtype=np.uint8))
    dem = circuit.detector_error_model(decompose_errors=True)
    dem.to_file(tmp_path / "learned.dem")
    (tmp_path / "circuit.stim").write_text(str(circuit))
    run = subprocess.run(
        [sys.executable, str(_SCRIPTS / "run_report.py"),
         "--detectors", str(tmp_path / "det.npy"),
         "--circuit", str(tmp_path / "circuit.stim"),
         "--learned-dem", str(tmp_path / "learned.dem"),
         "--skip", "decoder", "decoder_scalars", "stationarity",
         "--null-shots", "2000",
         "--output", str(tmp_path / "report.html")],
        capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    assert (tmp_path / "report.html").exists()
    assert (tmp_path / "report_state.pkl").exists()

    (tmp_path / "commentary.json").write_text(
        '```json\n{"summary": "CLI round trip."}\n```')
    annotate = subprocess.run(
        [sys.executable, str(_SCRIPTS / "annotate_report.py"),
         "--state", str(tmp_path / "report_state.pkl"),
         "--commentary", str(tmp_path / "commentary.json"),
         "--output", str(tmp_path / "annotated.html")],
        capture_output=True, text=True)
    assert annotate.returncode == 0, annotate.stderr
    assert "CLI round trip." in (tmp_path / "annotated.html").read_text()


@pytest.mark.slow
def test_generate_report_is_deterministic(report_run, tmp_path):
    """Same inputs and seed produce an identical analyst brief."""
    cfg, result, _ = report_run
    import dataclasses
    cfg2 = dataclasses.replace(cfg, output_path=tmp_path / "again.html",
                               results_path=None)
    result2 = generate_report(cfg2, log=lambda *_: None)
    assert result2.brief_path.read_text() == result.brief_path.read_text()

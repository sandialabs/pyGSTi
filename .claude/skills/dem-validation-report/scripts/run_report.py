#!/usr/bin/env python
"""
Command-line driver for the DEM validation report.

Enough for the common case (files on disk, stim b8 shot data). For anything
that needs custom detector coordinates, a custom scalar statistic, or a
findings section, import `dem_report` and call `generate_report` directly --
see SKILL.md.

Example::

    python run_report.py \
        --detectors data/detection_events.b8 \
        --observables data/obs_flips_actual.b8 \
        --circuit data/circuit_noisy_si1000.stim \
        --learned-dem out/learned_dem.dem \
        --decorated-dem out/decorated_dem.dem \
        --baseline-from-circuit --baseline-label si1000 \
        --events out/learned_events.npz \
        --title "Willow d5, Z memory, 10 rounds" \
        --output out/report.html
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import stim

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dem_report import ReportInputs, generate_report  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--detectors", required=True,
                   help="detector shot file (.b8) or .npy array")
    p.add_argument("--observables", help="observable shot file (.b8) or .npy")
    p.add_argument("--circuit", help="stim circuit, for detector coordinates")
    p.add_argument("--learned-dem", required=True)
    p.add_argument("--decorated-dem", help="learned DEM with L0 targets")
    p.add_argument("--baseline-dem", help="reference DEM file")
    p.add_argument("--baseline-from-circuit", action="store_true",
                   help="use circuit.detector_error_model(decompose_errors=True)")
    p.add_argument("--baseline-label", default="baseline")
    p.add_argument("--candidate-label", default="learned",
                   help="what to call the model under test in the report")
    p.add_argument("--events", help="npz with masks_hex/probs/stderr from the fit")
    p.add_argument("--num-detectors", type=int,
                   help="required if --detectors is .b8 and no --circuit given")
    p.add_argument("--coordinate-mode", default="auto",
                   choices=["auto", "stim", "google", "none"])
    p.add_argument("--title", default="Learned detector error model")
    p.add_argument("--subtitle", default="")
    p.add_argument("--output", default="dem_report.html")
    p.add_argument("--results", help="also pickle the raw ValidationResults here")
    p.add_argument("--brief", help="where to write the analyst brief "
                                   "(default: alongside --output)")
    p.add_argument("--no-brief", action="store_true")
    p.add_argument("--state", help="where to pickle the render state for "
                                   "annotate_report.py (default: alongside "
                                   "--output)")
    p.add_argument("--no-state", action="store_true")
    p.add_argument("--commentary", help="JSON commentary to render straight "
                                        "in, skipping annotate_report.py")
    p.add_argument("--decoders", nargs="+", default=["pymatching"],
                   choices=["pymatching", "tesseract"],
                   help="decoders for the LER section; give both to compare "
                        "matching against a hyperedge-native decoder")
    p.add_argument("--mc-shots", type=int,
                   help="Monte Carlo shots for the predicted LER "
                        "(default: min(8 x shots, 400000))")
    p.add_argument("--tesseract-mc-shots", type=int, default=20_000,
                   help="the same for tesseract rows, which cost ~5 ms/shot")
    p.add_argument("--decoder-scalar-tests", nargs="*",
                   default=["matching_weight"],
                   choices=["matching_weight", "complementary_gap"],
                   help="decoder-derived scalar distribution tests; pass "
                        "nothing after the flag to disable them. "
                        "complementary_gap is opt-in: on a hyperedge model "
                        "it costs ~0.5 s per decoded shot (see SKILL.md)")
    p.add_argument("--decoder-scalar-shots", type=int, default=10_000,
                   help="observed/null shot budget for decoder scalars when "
                        "the backend decodes one shot at a time (hyperedge "
                        "models fall back to tesseract)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--null-shots", type=int, default=200_000)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--skip", nargs="*", default=[],
                   choices=["marginals", "polarization", "scalars",
                            "decoder_scalars", "decoder", "stationarity"])
    args = p.parse_args(argv)

    circuit = stim.Circuit.from_file(args.circuit) if args.circuit else None

    def read_shots(path, num_detectors=None, num_observables=None):
        path = Path(path)
        if path.suffix == ".npy":
            return np.asarray(np.load(path), dtype=np.uint8)
        return np.asarray(stim.read_shot_data_file(
            path=str(path), format="b8", num_detectors=num_detectors or 0,
            num_observables=num_observables or 0), dtype=np.uint8)

    n_det = args.num_detectors or (circuit.num_detectors if circuit else None)
    if n_det is None and Path(args.detectors).suffix != ".npy":
        p.error("--num-detectors or --circuit is required for .b8 detector data")
    det = read_shots(args.detectors, num_detectors=n_det)

    obs = None
    if args.observables:
        obs = read_shots(args.observables, num_observables=1)
        obs = obs[:, 0] if obs.ndim == 2 else obs

    learned = stim.DetectorErrorModel.from_file(args.learned_dem)
    decorated = (stim.DetectorErrorModel.from_file(args.decorated_dem)
                 if args.decorated_dem else None)
    baseline = None
    if args.baseline_dem:
        baseline = stim.DetectorErrorModel.from_file(args.baseline_dem)
    elif args.baseline_from_circuit:
        if circuit is None:
            p.error("--baseline-from-circuit needs --circuit")
        baseline = circuit.detector_error_model(decompose_errors=True)

    event_stderr = None
    if args.events:
        ev = np.load(args.events)
        if "stderr" in ev:
            event_stderr = {int(m, 16): float(s)
                            for m, s in zip(ev["masks_hex"], ev["stderr"])}

    commentary = None
    if args.commentary:
        from annotate_report import load_commentary
        commentary = load_commentary(Path(args.commentary))

    result = generate_report(ReportInputs(
        detector_samples=det,
        observable_flips=obs,
        learned_dem=learned,
        decorated_dem=decorated,
        baseline_dem=baseline,
        baseline_label=args.baseline_label,
        candidate_label=args.candidate_label,
        circuit=circuit,
        coordinate_mode=args.coordinate_mode,
        event_stderr=event_stderr,
        title=args.title,
        subtitle=args.subtitle,
        commentary=commentary,
        decoders=tuple(args.decoders),
        num_mc_shots=args.mc_shots,
        tesseract_num_mc_shots=args.tesseract_mc_shots,
        decoder_scalar_tests=tuple(args.decoder_scalar_tests),
        decoder_scalar_shots=args.decoder_scalar_shots,
        output_path=Path(args.output),
        results_path=Path(args.results) if args.results else None,
        brief_path=False if args.no_brief else args.brief,
        state_path=False if args.no_state else args.state,
        seed=args.seed,
        null_shots=args.null_shots,
        alpha=args.alpha,
        skip=tuple(args.skip),
    ))
    print(f"\n{result.html_path}")
    if result.brief_path:
        print(f"  brief: {result.brief_path}")
    if result.state_path:
        print(f"  state: {result.state_path}")
    for key in ("learned", "baseline"):
        if key in result.summary:
            s = result.summary[key]
            print(f"  {key}: {s['num_rejected']} / {s['num_tests']} rejected")
    if "ler_observed" in result.summary:
        print(f"  LER: observed {result.summary['ler_observed']:.4f}, "
              f"predicted {result.summary['ler_predicted']:.4f} "
              f"({result.summary['ler_ratio']:.2f}x, "
              f"{result.summary['ler_z']:+.1f} sigma)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

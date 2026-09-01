"""Command-line interface for pygsti.extras.rareevent.

Run as ``python -m pygsti.extras.rareevent.cli``.

Thin wrapper over the Python API (which is the primary interface — every
capability here is reachable programmatically; see the package docstring).
The default estimators are the recommended gap-enhanced pipelines from
`pipelines.py`:

- ``--estimator splitting`` (default): locality-aware rare-event splitting
  with extra chains seeded from gap-harvested light failing states
  (`gap_seeded_splitting_estimate`).
- ``--estimator spectrum``: failure-spectrum ansatz with gap-splitting f(w)
  auxiliary points at the onset weights (`gap_spectrum_estimate`).
- ``--estimator splitting-uniform``: the legacy uniform-toggle
  Bravyi–Vargo baseline (`RareEventSplittingEstimator`), kept for
  benchmarking.

Pass ``--no-gap`` to run splitting/spectrum without the gap enhancement
(single-chain local splitting / plain spectrum fit). If the complementary-gap
construction does not apply to the DEM (it requires one logical observable
carried only by boundary edges), the CLI falls back to ``--no-gap`` behavior
with a warning.
"""

import argparse
import math
import sys

import numpy as np
import stim

from .rare_event import (
    RareEventSplittingEstimator,
    ScaledMechanismErrorModel,
    SplittingResult,
    build_catalog_decoder_and_dem_text,
    geometric_p_schedule,
    make_repetition_code_memory_circuit,
    make_surface_code_memory_circuit,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--estimator",
        choices=["splitting", "spectrum", "splitting-uniform"],
        default="splitting",
        help=(
            "Estimation method. 'splitting' = gap-seeded locality-aware rare-event splitting "
            "(recommended default); 'spectrum' = failure-spectrum ansatz with gap-splitting "
            "auxiliary points (arXiv:2511.15177); 'splitting-uniform' = legacy uniform-toggle "
            "baseline, kept for benchmarking."
        ),
    )
    parser.add_argument(
        "--no-gap",
        action="store_true",
        help="Disable the gap enhancement: single-chain local splitting / plain spectrum fit.",
    )
    parser.add_argument("--code", choices=["surface", "repetition"], default="surface")
    parser.add_argument("--distance", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=None)
    parser.add_argument("--basis", choices=["X", "Z", "x", "z"], default="X")
    parser.add_argument("--p0", type=float, default=3e-3)
    parser.add_argument("--p-final", type=float, default=1e-4)
    parser.add_argument("--levels", type=int, default=8)
    parser.add_argument("--mc-shots", type=int, default=20_000)
    parser.add_argument(
        "--onset-weight",
        type=int,
        default=None,
        help="Minimum failing fault weight w0. Defaults to ceil(distance/2).",
    )
    parser.add_argument(
        "--num-chains",
        type=int,
        default=4,
        help="Chains per splitting level (chains beyond the first start from gap-harvested seeds).",
    )
    parser.add_argument(
        "--steps-per-chain",
        type=int,
        default=None,
        help="MCMC proposal steps per chain per splitting level.",
    )
    parser.add_argument(
        "--total-steps",
        type=int,
        default=50_000,
        help="MCMC proposal steps per splitting level (divided across chains). Ignored if --steps-per-chain is set.",
    )
    parser.add_argument("--burn-in", type=int, default=None, help="Burn-in proposal steps. Overrides --burn-in-fraction if provided.")
    parser.add_argument("--burn-in-fraction", type=float, default=0.1, help="Fraction of MCMC proposal steps to discard as burn-in when --burn-in is not set.")
    parser.add_argument("--thin", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--global-dem-event-scale",
        type=float,
        default=0.0,
        help="Append one DEM event flipping every detector and logical target with probability scale*p0. Use 1e-4 for probability 1e-4*p.",
    )
    parser.add_argument(
        "--noise-model",
        choices=["scaled", "exact-si1000"],
        default="scaled",
        help="Noise model to use. 'scaled' uses linear scaling from p0. 'exact-si1000' uses exact decoration.",
    )
    parser.add_argument(
        "--spectrum-ansatz",
        choices=["2", "3", "5"],
        default="3",
        help="Failure-spectrum ansatz form (parameter count). Only used with --estimator spectrum.",
    )
    parser.add_argument(
        "--spectrum-target-failures",
        type=int,
        default=100,
        help="Failures to collect per sampled weight before moving on.",
    )
    parser.add_argument(
        "--spectrum-max-trials",
        type=int,
        default=20_000,
        help="Maximum fixed-weight fault sets evaluated per sampled weight.",
    )
    parser.add_argument(
        "--spectrum-num-weights",
        type=int,
        default=12,
        help="Number of log-spaced weights at which to sample the failure spectrum.",
    )
    args = parser.parse_args()

    total_steps_was_set = any(a == "--total-steps" or a.startswith("--total-steps=") for a in sys.argv[1:])
    burn_in_fraction_was_set = any(a == "--burn-in-fraction" or a.startswith("--burn-in-fraction=") for a in sys.argv[1:])
    if args.steps_per_chain is not None and total_steps_was_set:
        parser.error("Specify only one of --steps-per-chain or --total-steps.")
    if args.burn_in is not None and burn_in_fraction_was_set:
        parser.error("Specify only one of --burn-in or --burn-in-fraction.")
    if args.global_dem_event_scale < 0:
        parser.error("--global-dem-event-scale must be nonnegative.")
    if args.num_chains < 1:
        parser.error("--num-chains must be at least 1.")

    rounds = args.rounds if args.rounds is not None else 2 * args.distance

    from .interfaces import ErrorModel, ForwardSimulator
    from .rare_event import MechanismCatalog

    error_model: ErrorModel
    oracle: ForwardSimulator
    catalog: MechanismCatalog
    dem: stim.DetectorErrorModel

    if args.noise_model == "scaled":
        # Circuit decorated at p0; mechanism probabilities scale linearly with p.
        if args.code == "surface":
            circuit = make_surface_code_memory_circuit(
                distance=args.distance,
                rounds=rounds,
                p=args.p0,
                basis=args.basis.upper(),
            )
        else:
            circuit = make_repetition_code_memory_circuit(
                distance=args.distance,
                rounds=rounds,
                p=args.p0,
            )
        global_event_probability = args.global_dem_event_scale * args.p0
        catalog, oracle, dem_text = build_catalog_decoder_and_dem_text(
            circuit,
            global_dem_event_probability=global_event_probability,
        )
        dem = stim.DetectorErrorModel(dem_text)
        error_model = ScaledMechanismErrorModel(catalog, args.p0)
    else:
        # exact-si1000: noiseless skeleton re-decorated at every p for exact q_i(p).
        import pymatching

        from .noise import ExactNoiseErrorModel, SI1000NoiseModel
        from .rare_event import FailureOracle

        if args.code == "surface":
            circuit = make_surface_code_memory_circuit(
                distance=args.distance,
                rounds=rounds,
                p=0,  # Noiseless
                basis=args.basis.upper(),
            )
        else:
            circuit = make_repetition_code_memory_circuit(
                distance=args.distance,
                rounds=rounds,
                p=0,  # Noiseless
            )
        global_event_probability = args.global_dem_event_scale * args.p0
        noise_model = SI1000NoiseModel()
        error_model = ExactNoiseErrorModel(circuit, noise_model, p_ref=args.p0, global_dem_event_probability=global_event_probability)

        c_ref = noise_model(circuit, args.p0)
        dem = c_ref.detector_error_model(decompose_errors=True, flatten_loops=True)
        matching = pymatching.Matching.from_detector_error_model(dem)
        oracle = FailureOracle(error_model.catalog, matching)
        catalog = error_model.catalog

    line = (
        f"catalog: code={args.code} | mechanisms={len(catalog.mechanisms)} | "
        f"detectors={catalog.num_detectors} | observables={catalog.num_observables} | "
        f"global_dem_event_probability={global_event_probability:.6g}"
    )
    print(line)

    p_scales = geometric_p_schedule(args.p0, args.p_final, args.levels)
    onset_weight = args.onset_weight if args.onset_weight is not None else math.ceil(args.distance / 2)

    # Resolve the gap enhancement: build the GapOracle unless disabled, falling
    # back gracefully when the construction does not apply to this DEM.
    gap_oracle = None
    if not args.no_gap and args.estimator in ("splitting", "spectrum"):
        from .gap_splitting import GapOracle

        try:
            gap_oracle = GapOracle.from_dem(dem, catalog)
        except ValueError as exc:
            print(f"gap enhancement unavailable ({exc}); falling back to --no-gap behavior.")

    if args.estimator == "spectrum":
        from .failure_spectrum import failure_spectrum_estimate
        from .pipelines import gap_spectrum_estimate

        if gap_oracle is not None:
            spectrum_result = gap_spectrum_estimate(
                error_model,
                oracle,
                catalog,
                gap_oracle,
                p_scales,
                onset_weight=onset_weight,
                p_ref=args.p0,
                ansatz=args.spectrum_ansatz,
                target_failures=args.spectrum_target_failures,
                max_trials_per_weight=args.spectrum_max_trials,
                num_weights=args.spectrum_num_weights,
                num_observables=catalog.num_observables,
                seed=args.seed,
            )
        else:
            spectrum_result = failure_spectrum_estimate(
                error_model=error_model,
                simulator=oracle,
                p_scales=p_scales,
                p_ref=args.p0,
                ansatz=args.spectrum_ansatz,
                w0=float(onset_weight),
                target_failures=args.spectrum_target_failures,
                max_trials_per_weight=args.spectrum_max_trials,
                num_weights=args.spectrum_num_weights,
                num_observables=catalog.num_observables,
                seed=args.seed,
            )
        print("summary")
        for p, lf, f in zip(
            spectrum_result.p_scales,
            spectrum_result.log_failure_estimates,
            spectrum_result.failure_estimates,
        ):
            print(f"p={p:.8g} | log_P_fail={lf:.8g} | P_fail={f:.8e}")
        return

    result: SplittingResult
    if args.estimator == "splitting-uniform":
        result = RareEventSplittingEstimator().estimate(
            error_model=error_model,
            simulator=oracle,
            p_scales=p_scales,
            mc_shots_at_p0=args.mc_shots,
            steps_per_chain=args.steps_per_chain,
            total_steps_per_level=None if args.steps_per_chain is not None else args.total_steps,
            burn_in=args.burn_in,
            burn_in_fraction=None if args.burn_in is not None else args.burn_in_fraction,
            thin=args.thin,
            seed=args.seed,
        )
    elif gap_oracle is not None:
        from .pipelines import gap_seeded_splitting_estimate

        result = gap_seeded_splitting_estimate(
            error_model,
            oracle,
            catalog,
            gap_oracle,
            p_scales,
            onset_weight=onset_weight,
            num_chains=args.num_chains,
            mc_shots_at_p0=args.mc_shots,
            steps_per_chain=args.steps_per_chain,
            total_steps_per_level=None if args.steps_per_chain is not None else args.total_steps,
            burn_in=args.burn_in,
            burn_in_fraction=None if args.burn_in is not None else args.burn_in_fraction,
            thin=args.thin,
            seed=args.seed,
        )
    else:
        from .splitting_local import local_splitting_estimate

        result = local_splitting_estimate(
            error_model=error_model,
            simulator=oracle,
            catalog=catalog,
            p_scales=p_scales,
            mc_shots_at_p0=args.mc_shots,
            steps_per_chain=args.steps_per_chain,
            total_steps_per_level=None if args.steps_per_chain is not None else args.total_steps,
            burn_in=args.burn_in,
            burn_in_fraction=None if args.burn_in is not None else args.burn_in_fraction,
            thin=args.thin,
            seed=args.seed,
        )

    print("summary")
    for p, lf, f in zip(result.p_scales, result.log_failure_estimates, result.failure_estimates):
        print(f"p={p:.8g} | log_P_fail={lf:.8g} | P_fail={f:.8e}")

    if result.level_diagnostics:
        print("level diagnostics")
        for d in result.level_diagnostics:
            rhat_lr = "n/a" if d.rhat_log_weight_ratio is None else f"{d.rhat_log_weight_ratio:.4f}"
            rhat_w = "n/a" if d.rhat_active_weight is None else f"{d.rhat_active_weight:.4f}"
            line = (
                f"level {d.level} | p={d.p_current:.8g}->{d.p_next:.8g} | "
                f"pooled_log_ratio={d.pooled_log_ratio:.8g} | "
                f"chain_log_ratio_min={np.nanmin(d.per_chain_log_ratios):.8g} | "
                f"chain_log_ratio_max={np.nanmax(d.per_chain_log_ratios):.8g} | "
                f"mean_acceptance={np.mean(d.per_chain_acceptance_rates):.4f} | "
                f"mean_active_weight={np.nanmean(d.per_chain_mean_weights):.4f} | "
                f"Rhat_log_weight_ratio={rhat_lr} | Rhat_active_weight={rhat_w}"
            )
            print(line)


if __name__ == "__main__":
    main()

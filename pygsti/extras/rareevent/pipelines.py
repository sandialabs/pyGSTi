"""
Recommended end-to-end estimation pipelines (gap-splitting-enhanced defaults).

The July 2026 benchmark campaign (see ``benchmarks/REPORT.md`` in the
standalone error-rate-estimation repository and the methods pages under
``docs/markdown/rareevent/`` here) established that both headline estimators
improve
markedly when combined with fixed-weight gap-splitting
(`gap_splitting.estimate_f_w_gap_splitting`):

- **Failure spectrum + gap auxiliary points** (`gap_spectrum_estimate`):
  rejection sampling cannot resolve the failure fraction f(w) below roughly
  ``1 / max_trials_per_weight``, so at high distance the ansatz fit
  extrapolates blindly across the onset weights that dominate P_fail at low
  p. Gap-splitting measures those f(w) directly (down to ~1e-10 in minutes)
  and this pipeline feeds them into the fit as auxiliary data points with
  explicit log-space errors, pinning the onset.

- **Splitting + gap-harvested seed chains** (`gap_seeded_splitting_estimate`):
  the conditional-failure MCMC of `splitting_local` explores the failure
  region by local moves from an anchor state and can miss light (near-onset)
  malignant basins, biasing the level ratios. Gap-splitting *visits* those
  light failing states as a side effect (``harvest_states``); this pipeline
  runs extra chains per level started from them and pools the samples,
  reporting a cross-chain split-R̂ that flags residual multi-basin structure.

These two pipelines are the package's default recommendation for the
splitting and failure-spectrum approaches; the un-enhanced building blocks
(`splitting_local.local_splitting_estimate`,
`failure_spectrum.failure_spectrum_estimate`, and the legacy uniform-toggle
`rare_event.RareEventSplittingEstimator`) remain available for benchmarking
and for DEMs where the gap construction does not apply.

Applicability: the complementary-gap construction requires a matchable DEM
with exactly one logical observable carried only by virtual-boundary edges
(`gap_splitting.make_gap_matching_from_vanilla_dem`). This holds for standard
memory experiments on repetition and rotated surface codes. When it does not
hold, `GapOracle.from_dem` raises ``ValueError`` — fall back to the
un-enhanced estimators.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import stim

from .failure_spectrum import FailureSpectrumResult, failure_spectrum_estimate
from .gap_splitting import GapOracle, estimate_f_w_gap_splitting
from .interfaces import ErrorModel, Estimator, ForwardSimulator
from .rare_event import MechanismCatalog, SplittingResult
from .splitting_local import local_splitting_estimate
from .splitting_subregion import default_region_rate, subregion_splitting_estimate
from .weight_points import WeightPoint


def _resolve_gap_oracle(
    dem_or_gap_oracle: stim.DetectorErrorModel | GapOracle, catalog: MechanismCatalog
) -> GapOracle:
    if isinstance(dem_or_gap_oracle, GapOracle):
        return dem_or_gap_oracle
    return GapOracle.from_dem(dem_or_gap_oracle, catalog)


def measure_gap_weight_points(
    error_model: ErrorModel,
    simulator: ForwardSimulator,
    catalog: MechanismCatalog,
    dem_or_gap_oracle: stim.DetectorErrorModel | GapOracle,
    weights: Sequence[int],
    *,
    p_ref: float,
    num_particles: int = 400,
    quantile: float = 0.25,
    mcmc_steps_per_particle: int = 30,
    repeats: int = 3,
    seed: int = 1,
    local_prob: float = 0.5,
    harvest_states: int = 0,
    verbose: bool = False,
) -> list[WeightPoint]:
    """Measure f(w) at each weight in ``weights`` with fixed-weight gap-splitting.

    Thin batching wrapper around `gap_splitting.estimate_f_w_gap_splitting`
    that builds the `GapOracle` once and reuses it for every weight. Weight w
    is run with seed ``seed + w`` so points are independent and reproducible
    per weight. See that function for parameter semantics; when
    ``harvest_states > 0``, each returned point carries up to that many
    distinct failing weight-w states in ``meta["failing_states"]``.
    """
    gap_oracle = _resolve_gap_oracle(dem_or_gap_oracle, catalog)
    points: list[WeightPoint] = []
    for w in weights:
        point = estimate_f_w_gap_splitting(
            error_model=error_model,
            oracle=simulator,
            catalog=catalog,
            dem_or_gap_oracle=gap_oracle,
            weight=int(w),
            p_ref=p_ref,
            num_particles=num_particles,
            quantile=quantile,
            mcmc_steps_per_particle=mcmc_steps_per_particle,
            repeats=repeats,
            seed=seed + int(w),
            local_prob=local_prob,
            harvest_states=harvest_states,
            verbose=verbose,
        )
        points.append(point)
        if verbose:
            print(f"gap f({w}) = {point.estimate:.6g} (rel_err {point.rel_err:.3g})")
    return points


def gap_spectrum_estimate(
    error_model: ErrorModel,
    simulator: ForwardSimulator,
    catalog: MechanismCatalog,
    dem_or_gap_oracle: stim.DetectorErrorModel | GapOracle,
    p_scales: Sequence[float],
    *,
    onset_weight: int,
    p_ref: float | None = None,
    gap_weight_span: int = 4,
    gap_num_particles: int = 400,
    gap_quantile: float = 0.25,
    gap_mcmc_steps_per_particle: int = 30,
    gap_repeats: int = 3,
    seed: int = 1,
    verbose: bool = True,
    **spectrum_kwargs: Any,
) -> FailureSpectrumResult:
    """Failure-spectrum estimate with gap-splitting f(w) auxiliary points (recommended default).

    Runs `gap_splitting.estimate_f_w_gap_splitting` at the onset weights
    ``onset_weight .. onset_weight + gap_weight_span``, then calls
    `failure_spectrum.failure_spectrum_estimate` with those measurements as
    ``aux_points`` (log-space residuals with the points' relative errors) and
    ``w0 = onset_weight``. The auxiliary points pin the low-weight end of the
    ansatz that counting alone cannot observe at high distance.

    Args:
        error_model: Provides mechanism probabilities q_i(p).
        simulator: Resolves the true failure event for an active mechanism set.
        catalog: Mechanism catalog shared by ``simulator`` and the gap oracle.
        dem_or_gap_oracle: Flattened, decomposed DEM (a `GapOracle` is built
            from it) or a prebuilt `GapOracle` to reuse.
        p_scales: Physical error rates to predict.
        onset_weight: Minimum failing weight w0 — ``ceil(d/2)`` for a
            distance-d code under min-weight decoding.
        p_ref: Reference rate defining the sampling distribution for both the
            gap points and the spectrum counting; defaults to max(p_scales).
        gap_weight_span: Auxiliary points are measured at
            ``onset_weight .. onset_weight + gap_weight_span``.
        gap_num_particles / gap_quantile / gap_mcmc_steps_per_particle /
            gap_repeats: Subset-simulation controls, per
            `gap_splitting.estimate_f_w_gap_splitting`.
        seed: Base seed for both stages.
        **spectrum_kwargs: Forwarded to
            `failure_spectrum.failure_spectrum_estimate` (e.g.
            ``target_failures``, ``max_trials_per_weight``, ``ansatz``).

    Returns:
        `FailureSpectrumResult`; the gap measurements used by the fit are in
        ``result.aux_points``.
    """
    if onset_weight < 1:
        raise ValueError(f"onset_weight must be at least 1; got {onset_weight}.")
    if gap_weight_span < 0:
        raise ValueError(f"gap_weight_span must be nonnegative; got {gap_weight_span}.")
    if "aux_points" in spectrum_kwargs:
        raise ValueError("aux_points is produced by this pipeline; pass gap_* parameters instead.")
    if p_ref is None:
        p_ref = float(max(p_scales))

    gap_oracle = _resolve_gap_oracle(dem_or_gap_oracle, catalog)
    aux_points = measure_gap_weight_points(
        error_model,
        simulator,
        catalog,
        gap_oracle,
        weights=range(onset_weight, onset_weight + gap_weight_span + 1),
        p_ref=p_ref,
        num_particles=gap_num_particles,
        quantile=gap_quantile,
        mcmc_steps_per_particle=gap_mcmc_steps_per_particle,
        repeats=gap_repeats,
        seed=seed,
        verbose=verbose,
    )

    spectrum_kwargs.setdefault("w0", float(onset_weight))
    return failure_spectrum_estimate(
        error_model=error_model,
        simulator=simulator,
        p_scales=p_scales,
        p_ref=p_ref,
        aux_points=aux_points,
        seed=seed,
        verbose=verbose,
        **spectrum_kwargs,
    )


def gap_seeded_splitting_estimate(
    error_model: ErrorModel,
    simulator: ForwardSimulator,
    catalog: MechanismCatalog,
    dem_or_gap_oracle: stim.DetectorErrorModel | GapOracle,
    p_scales: Sequence[float],
    *,
    onset_weight: int,
    num_chains: int = 4,
    harvest_weight_span: int = 2,
    harvest_states_per_weight: int = 24,
    harvest_p_ref: float | None = None,
    gap_num_particles: int = 400,
    gap_quantile: float = 0.25,
    gap_mcmc_steps_per_particle: int = 30,
    gap_repeats: int = 1,
    seed: int = 1,
    verbose: bool = True,
    **splitting_kwargs: Any,
) -> SplittingResult:
    """Multi-chain rare-event splitting seeded from gap-harvested light failing states
    (recommended default).

    Stage 1 runs fixed-weight gap-splitting at
    ``onset_weight .. onset_weight + harvest_weight_span`` purely to *harvest*
    distinct failing states of those weights — the near-onset malignant sets a
    p-weighted local chain cannot reach. Stage 2 calls
    `splitting_local.local_splitting_estimate` with ``num_chains`` chains per
    level: chain 0 starts from the anchor as usual, chains 1..num_chains-1
    start from the harvested seeds (cycled). A configuration's failure status
    is p-independent, so the same seeds are valid at every level. Cross-chain
    split-R̂ per level is reported in the result's ``level_diagnostics``.

    Args:
        error_model / simulator / catalog / dem_or_gap_oracle / p_scales: As
            in `gap_spectrum_estimate`; ``p_scales`` is the descending
            splitting schedule starting at the anchor rate p0.
        onset_weight: Minimum failing weight (``ceil(d/2)``); seeds are
            harvested starting there because those are the states local moves
            miss.
        num_chains: Chains per level (1 reproduces the single-chain
            estimator; seeds are then unused).
        harvest_weight_span: Harvest at ``onset_weight .. onset_weight +
            harvest_weight_span``.
        harvest_states_per_weight: Maximum distinct failing states kept per
            harvested weight.
        harvest_p_ref: Reference rate for the harvest's sampling distribution
            (failure status of the harvested states does not depend on it);
            defaults to ``p_scales[0]``.
        gap_num_particles / gap_quantile / gap_mcmc_steps_per_particle /
            gap_repeats: Subset-simulation controls for the harvest runs. One
            repeat suffices for harvesting (the f(w) value is discarded).
        seed: Base seed for both stages.
        **splitting_kwargs: Forwarded to
            `splitting_local.local_splitting_estimate` — most importantly the
            anchor policy: either ``mc_shots_at_p0`` for an internal anchor,
            or ``anchor_failure_rate`` + ``anchor_state`` for an external
            (e.g. catalog-MC) anchor. Also ``total_steps_per_level`` /
            ``steps_per_chain``, ``thin``, ``burn_in_fraction``, ...

    Returns:
        `SplittingResult` with per-level multi-chain `LevelDiagnostics`.
    """
    if onset_weight < 1:
        raise ValueError(f"onset_weight must be at least 1; got {onset_weight}.")
    if harvest_weight_span < 0:
        raise ValueError(f"harvest_weight_span must be nonnegative; got {harvest_weight_span}.")
    if harvest_states_per_weight < 1:
        raise ValueError(f"harvest_states_per_weight must be at least 1; got {harvest_states_per_weight}.")
    if num_chains < 1:
        raise ValueError(f"num_chains must be at least 1; got {num_chains}.")
    for key in ("seed_states", "num_chains"):
        if key in splitting_kwargs:
            raise ValueError(f"{key} is produced by this pipeline; pass harvest_*/num_chains parameters instead.")
    if not p_scales:
        raise ValueError("p_scales must be a non-empty descending schedule.")
    if harvest_p_ref is None:
        harvest_p_ref = float(p_scales[0])

    seed_states: list[set[int]] = []
    if num_chains > 1:
        gap_oracle = _resolve_gap_oracle(dem_or_gap_oracle, catalog)
        harvest_points = measure_gap_weight_points(
            error_model,
            simulator,
            catalog,
            gap_oracle,
            weights=range(onset_weight, onset_weight + harvest_weight_span + 1),
            p_ref=harvest_p_ref,
            num_particles=gap_num_particles,
            quantile=gap_quantile,
            mcmc_steps_per_particle=gap_mcmc_steps_per_particle,
            repeats=gap_repeats,
            seed=seed,
            harvest_states=harvest_states_per_weight,
            verbose=verbose,
        )
        for point in harvest_points:
            for state in point.meta.get("failing_states", []):
                seed_states.append(set(state))
        if verbose:
            print(f"harvested {len(seed_states)} failing seed states at weights "
                  f"{onset_weight}..{onset_weight + harvest_weight_span}")
        if not seed_states:
            raise RuntimeError(
                "Gap harvest produced no failing states; increase gap_num_particles / "
                f"harvest_weight_span, or check that onset_weight={onset_weight} is reachable."
            )

    return local_splitting_estimate(
        error_model=error_model,
        simulator=simulator,
        catalog=catalog,
        p_scales=list(p_scales),
        mc_shots_at_p0=splitting_kwargs.pop("mc_shots_at_p0", 10_000),
        steps_per_chain=splitting_kwargs.pop("steps_per_chain", None),
        total_steps_per_level=splitting_kwargs.pop("total_steps_per_level", None),
        burn_in=splitting_kwargs.pop("burn_in", None),
        burn_in_fraction=splitting_kwargs.pop("burn_in_fraction", 0.1),
        thin=splitting_kwargs.pop("thin", 1),
        seed=seed,
        beta_global=splitting_kwargs.pop("beta_global", 0.1),
        anchor_failure_rate=splitting_kwargs.pop("anchor_failure_rate", None),
        anchor_state=splitting_kwargs.pop("anchor_state", None),
        seed_states=seed_states if seed_states else None,
        num_chains=num_chains,
        **splitting_kwargs,
    )


def _harvest_gap_seed_states(
    error_model: ErrorModel,
    simulator: ForwardSimulator,
    catalog: MechanismCatalog,
    dem_or_gap_oracle: stim.DetectorErrorModel | GapOracle,
    *,
    onset_weight: int,
    harvest_weight_span: int,
    harvest_states_per_weight: int,
    harvest_p_ref: float,
    gap_num_particles: int,
    gap_quantile: float,
    gap_mcmc_steps_per_particle: int,
    gap_repeats: int,
    seed: int,
    verbose: bool,
) -> list[set[int]]:
    """Harvest light failing seed states with fixed-weight gap-splitting.

    Same harvest stage as `gap_seeded_splitting_estimate`; raises RuntimeError
    when nothing was harvested.
    """
    gap_oracle = _resolve_gap_oracle(dem_or_gap_oracle, catalog)
    harvest_points = measure_gap_weight_points(
        error_model,
        simulator,
        catalog,
        gap_oracle,
        weights=range(onset_weight, onset_weight + harvest_weight_span + 1),
        p_ref=harvest_p_ref,
        num_particles=gap_num_particles,
        quantile=gap_quantile,
        mcmc_steps_per_particle=gap_mcmc_steps_per_particle,
        repeats=gap_repeats,
        seed=seed,
        harvest_states=harvest_states_per_weight,
        verbose=verbose,
    )
    seed_states: list[set[int]] = []
    for point in harvest_points:
        for state in point.meta.get("failing_states", []):
            seed_states.append(set(state))
    if verbose:
        print(f"harvested {len(seed_states)} failing seed states at weights "
              f"{onset_weight}..{onset_weight + harvest_weight_span}")
    if not seed_states:
        raise RuntimeError(
            "Gap harvest produced no failing states; increase gap_num_particles / "
            f"harvest_weight_span, or check that onset_weight={onset_weight} is reachable."
        )
    return seed_states


def gap_seeded_subregion_estimate(
    error_model: ErrorModel,
    simulator: ForwardSimulator,
    catalog: MechanismCatalog,
    dem_or_gap_oracle: stim.DetectorErrorModel | GapOracle,
    p_scales: Sequence[float],
    *,
    onset_weight: int,
    region_rate: float | None = None,
    num_chains: int = 4,
    stop_rhat: float | None = None,
    ratio_estimator: str = "forward",
    harvest_weight_span: int = 2,
    harvest_states_per_weight: int = 24,
    harvest_p_ref: float | None = None,
    gap_num_particles: int = 400,
    gap_quantile: float = 0.25,
    gap_mcmc_steps_per_particle: int = 30,
    gap_repeats: int = 1,
    seed: int = 1,
    verbose: bool = True,
    **splitting_kwargs: Any,
) -> SplittingResult:
    """Gap-seeded splitting with the subregion kernel (arXiv:2607.27153 stack).

    Identical two-stage flow to `gap_seeded_splitting_estimate` (harvest light
    failing states with fixed-weight gap-splitting, then a multi-chain
    descent), but stage 2 uses
    `splitting_subregion.subregion_splitting_estimate`: the rejection-free
    partial-resampling kernel, with optional R-hat-driven adaptive level
    stopping (``stop_rhat``) and the Bennett-acceptance-ratio level estimator
    (``ratio_estimator="bar"``). ``region_rate`` defaults to the paper's
    core-resampling heuristic ``default_region_rate(onset_weight)``.

    The locality-aware baseline (`gap_seeded_splitting_estimate`) is kept
    unchanged for comparison; see ``docs/markdown/rareevent/method_evolution.md``
    for the lineage.
    """
    if onset_weight < 1:
        raise ValueError(f"onset_weight must be at least 1; got {onset_weight}.")
    if harvest_weight_span < 0:
        raise ValueError(f"harvest_weight_span must be nonnegative; got {harvest_weight_span}.")
    if harvest_states_per_weight < 1:
        raise ValueError(f"harvest_states_per_weight must be at least 1; got {harvest_states_per_weight}.")
    if num_chains < 1:
        raise ValueError(f"num_chains must be at least 1; got {num_chains}.")
    for key in ("seed_states", "num_chains"):
        if key in splitting_kwargs:
            raise ValueError(f"{key} is produced by this pipeline; pass harvest_*/num_chains parameters instead.")
    if not p_scales:
        raise ValueError("p_scales must be a non-empty descending schedule.")
    if harvest_p_ref is None:
        harvest_p_ref = float(p_scales[0])
    if region_rate is None:
        region_rate = default_region_rate(onset_weight)

    seed_states: list[set[int]] = []
    if num_chains > 1:
        seed_states = _harvest_gap_seed_states(
            error_model,
            simulator,
            catalog,
            dem_or_gap_oracle,
            onset_weight=onset_weight,
            harvest_weight_span=harvest_weight_span,
            harvest_states_per_weight=harvest_states_per_weight,
            harvest_p_ref=harvest_p_ref,
            gap_num_particles=gap_num_particles,
            gap_quantile=gap_quantile,
            gap_mcmc_steps_per_particle=gap_mcmc_steps_per_particle,
            gap_repeats=gap_repeats,
            seed=seed,
            verbose=verbose,
        )

    return subregion_splitting_estimate(
        error_model=error_model,
        simulator=simulator,
        p_scales=list(p_scales),
        region_rate=region_rate,
        seed=seed,
        seed_states=seed_states if seed_states else None,
        num_chains=num_chains,
        stop_rhat=stop_rhat,
        ratio_estimator=ratio_estimator,
        **splitting_kwargs,
    )


class GapSpectrumEstimator(Estimator):
    """Recommended default spectrum estimator: failure-spectrum fit with gap-splitting
    auxiliary f(w) points at the onset weights (see `gap_spectrum_estimate`)."""

    def estimate(
        self,
        error_model: ErrorModel,
        simulator: ForwardSimulator,
        **kwargs: Any,
    ) -> FailureSpectrumResult:
        for key in ("p_scales", "catalog", "onset_weight"):
            if key not in kwargs:
                raise ValueError(f"{key} must be provided to GapSpectrumEstimator")
        if "dem" not in kwargs and "gap_oracle" not in kwargs:
            raise ValueError("dem or gap_oracle must be provided to GapSpectrumEstimator")
        catalog: MechanismCatalog = kwargs.pop("catalog")
        source = kwargs.pop("gap_oracle") if "gap_oracle" in kwargs else kwargs.pop("dem")
        return gap_spectrum_estimate(
            error_model,
            simulator,
            catalog,
            source,
            kwargs.pop("p_scales"),
            onset_weight=int(kwargs.pop("onset_weight")),
            **kwargs,
        )


class GapSeededSplittingEstimator(Estimator):
    """Recommended default splitting estimator: locality-aware splitting with extra
    chains seeded from gap-harvested light failing states (see
    `gap_seeded_splitting_estimate`)."""

    def estimate(
        self,
        error_model: ErrorModel,
        simulator: ForwardSimulator,
        **kwargs: Any,
    ) -> SplittingResult:
        for key in ("p_scales", "catalog", "onset_weight"):
            if key not in kwargs:
                raise ValueError(f"{key} must be provided to GapSeededSplittingEstimator")
        if "dem" not in kwargs and "gap_oracle" not in kwargs:
            raise ValueError("dem or gap_oracle must be provided to GapSeededSplittingEstimator")
        catalog: MechanismCatalog = kwargs.pop("catalog")
        source = kwargs.pop("gap_oracle") if "gap_oracle" in kwargs else kwargs.pop("dem")
        return gap_seeded_splitting_estimate(
            error_model,
            simulator,
            catalog,
            source,
            kwargs.pop("p_scales"),
            onset_weight=int(kwargs.pop("onset_weight")),
            **kwargs,
        )


class GapSeededSubregionEstimator(Estimator):
    """Subregion-kernel splitting with gap-harvested seed chains, optional R-hat
    stopping and BAR level ratios (see `gap_seeded_subregion_estimate`)."""

    def estimate(
        self,
        error_model: ErrorModel,
        simulator: ForwardSimulator,
        **kwargs: Any,
    ) -> SplittingResult:
        for key in ("p_scales", "catalog", "onset_weight"):
            if key not in kwargs:
                raise ValueError(f"{key} must be provided to GapSeededSubregionEstimator")
        if "dem" not in kwargs and "gap_oracle" not in kwargs:
            raise ValueError("dem or gap_oracle must be provided to GapSeededSubregionEstimator")
        catalog: MechanismCatalog = kwargs.pop("catalog")
        source = kwargs.pop("gap_oracle") if "gap_oracle" in kwargs else kwargs.pop("dem")
        return gap_seeded_subregion_estimate(
            error_model,
            simulator,
            catalog,
            source,
            kwargs.pop("p_scales"),
            onset_weight=int(kwargs.pop("onset_weight")),
            **kwargs,
        )


def default_onset_weight(distance: int) -> int:
    """Minimum failing fault weight for a distance-``distance`` code under
    min-weight decoding: ``ceil(distance / 2)``."""
    if distance < 1:
        raise ValueError(f"distance must be at least 1; got {distance}.")
    return math.ceil(distance / 2)

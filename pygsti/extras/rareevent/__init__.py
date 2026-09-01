"""
Rare-event estimation of low logical error rates in high-distance QEC codes.

Predicts logical failure rates at physical error rates p where ordinary
Monte Carlo is intractable (validated down to P_fail ~ 1e-12 at distance 11).
Estimation methods are strictly decoupled from circuit simulation and
decoding: everything plugs into the three Protocols in `interfaces`
(`ErrorModel`, `ForwardSimulator`, `Estimator`), with `stim` + `pymatching`
as the reference implementations behind `MechanismCatalog` / `FailureOracle`.

The recommended default pipelines are `gap_seeded_splitting_estimate` and
`gap_spectrum_estimate` (see `pipelines`). The mathematics of every method is
documented under ``docs/markdown/rareevent/`` in the pyGSTi repository.

This package originated as the standalone ``error-rate-estimation``
repository, which remains the home of its benchmark suite and campaign
results (``benchmarks/REPORT.md`` there).
"""

from . import rare_event
from .core_planting import CountingOracle, harvest_cores, peel_to_minimal_subset
from .failure_spectrum import FailureSpectrumEstimator, FittedFailureSpectrum
from .gap_splitting import GapOracle, GapSplittingEstimator
from .interfaces import ErrorModel, Estimator, ForwardSimulator
from .malignant import MalignantSetEstimator
from .noise import ExactNoiseErrorModel, NoiseModel, SI1000NoiseModel
from .pipelines import (
    GapSeededSplittingEstimator,
    GapSeededSubregionEstimator,
    GapSpectrumEstimator,
    default_onset_weight,
    gap_seeded_splitting_estimate,
    gap_seeded_subregion_estimate,
    gap_spectrum_estimate,
    measure_gap_weight_points,
)
from .rare_event import FailureOracle, MechanismCatalog, RareEventSplittingEstimator
from .splitting_local import LocalSplittingEstimator
from .splitting_subregion import SubregionSplittingEstimator, default_region_rate, subregion_splitting_estimate
from .weight_points import WeightPoint

__all__ = [
    "rare_event",
    # Core protocols
    "ErrorModel",
    "Estimator",
    "ForwardSimulator",
    # Pipeline building blocks
    "NoiseModel",
    "SI1000NoiseModel",
    "ExactNoiseErrorModel",
    "MechanismCatalog",
    "FailureOracle",
    "GapOracle",
    "WeightPoint",
    # Recommended default estimators (gap-enhanced)
    "GapSeededSplittingEstimator",
    "GapSeededSubregionEstimator",
    "GapSpectrumEstimator",
    "gap_seeded_splitting_estimate",
    "gap_seeded_subregion_estimate",
    "gap_spectrum_estimate",
    "measure_gap_weight_points",
    "default_onset_weight",
    # Individual estimators (building blocks / benchmarking)
    "LocalSplittingEstimator",
    "SubregionSplittingEstimator",
    "subregion_splitting_estimate",
    "default_region_rate",
    "FailureSpectrumEstimator",
    "FittedFailureSpectrum",
    "GapSplittingEstimator",
    "MalignantSetEstimator",
    "RareEventSplittingEstimator",
    # Core harvesting utilities
    "CountingOracle",
    "harvest_cores",
    "peel_to_minimal_subset",
]

"""
pyGSTi Error Generator Propagation Package

This package collects the tools used to propagate elementary error generators through
Clifford circuits and to convert the propagated error generators into probabilities and
Pauli expectation values. The public API is re-exported from the following submodules:

- `errgentypes`: shared type aliases and `TypeVar` definitions.
- `algebra`: closed-form commutator, composition, anticommutator and Pauli conjugation rules.
- `parallelutils`: parallelization configuration, thresholds, memory estimators and workers.
- `sensitivities`: stabilizer support/amplitude tableau math and `phi`/`alpha`/`alpha_pauli`.
- `expansions`: BCH, Magnus, Zassenhaus and Taylor expansions of error generator layers.
- `corrections`: ideal stabilizer baselines, tableau fidelities and probability/Pauli
  expectation corrections and approximations.
- `numerical`: dense-matrix numerical reference implementations used for validation.
- `polynomials`: symbolic polynomial error generator propagation routines.
- `fasterrgencalc`: optional Cython accelerations (`None` if the extension is unavailable).

Underscore-prefixed helpers and shared mutable dictionaries are also re-exported explicitly
so that internal callers and tests can reach them through this package.
"""
#***************************************************************************************************
# Copyright 2025 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

from . import errgentypes
from . import algebra
from . import parallelutils
from . import sensitivities
from . import expansions
from . import corrections
from . import numerical
from . import polynomials

try:
    from . import fasterrgencalc
except ImportError:
    fasterrgencalc = None

# Public API.
from .errgentypes import *
from .algebra import *
from .parallelutils import *
from .sensitivities import *
from .expansions import *
from .corrections import *
from .numerical import *
from .polynomials import *

# Internal helpers and shared mutable state used by other pyGSTi modules and tests.
from .errgentypes import _ErrorgenTerms, _ErrorgenDict, _Rate, _SignedPauli, _PairHandler, _F

from .algebra import (
    _H, _S, _C, _A, _prod, _com, _acom, _index,
    _commutator_HH, _commutator_HS, _commutator_SH, _commutator_HC, _commutator_CH, _commutator_HA,
    _commutator_AH, _commutator_SS, _commutator_SC, _commutator_CS, _commutator_SA, _commutator_AS,
    _commutator_CC, _commutator_CA, _commutator_AC, _commutator_AA,
    _composition_HH, _composition_HS, _composition_SH, _composition_SS, _composition_HC, _composition_HA,
    _composition_SC, _composition_SA, _composition_CH, _composition_CS, _composition_AH, _composition_AS,
    _composition_CC, _composition_CA, _composition_AC, _composition_AA,
    _conjugation_H, _conjugation_S, _conjugation_C, _conjugation_A,
    _anticommutator_HH, _anticommutator_HS, _anticommutator_SH, _anticommutator_SS, _anticommutator_HC,
    _anticommutator_CH, _anticommutator_HA, _anticommutator_AH, _anticommutator_SC, _anticommutator_CS,
    _anticommutator_SA, _anticommutator_AS, _anticommutator_CC, _anticommutator_CA, _anticommutator_AC,
    _anticommutator_AA,
    _COMMUTATOR_HANDLERS, _COMPOSITION_HANDLERS, _ANTICOMMUTATOR_HANDLERS, _CONJUGATION_HANDLERS,
)

from .parallelutils import (
    _cyclic_gc_paused, _with_cyclic_gc_paused,
    _MANUAL_PARALLEL_OVERRIDES, _ACTIVE_PARALLEL_THRESHOLDS, _PARALLEL_OPERATIONS,
    _normalize_forced_operations, _forced_parallelism_mode, _qubit_scale_tier,
    _STIM_PAULISTRING_BASE_BYTES, _DICT_SLOT_BYTES,
    _string_entry_size, _num_distinct_errorgen_labels, _num_labels_with_support_at_most,
    _layer_support_statistics, _estimate_magnus_peak_memory_from_layers,
    _should_parallelize_magnus, _should_parallelize_commuting_product, _should_parallelize_alpha,
    _lse_from_string_rep, _merge_string_dictionaries, _merge_pair_indexed_string_dictionaries,
    _generate_balanced_magnus_pairs,
    _MagnusPairEvaluator, _MagnusPhase1PairEvaluator, _MagnusPhase2Evaluator,
    _CommutingProductSliceWorker, _BulkAlphaWorker, _BulkAlphaPauliWorker,
    _ALPHA_CALIBRATION_QUBIT_SPAN, _calibration_tableau_and_errorgens, _measure_alpha_throughput_us,
    _measure_ipc_latency,
    _MIN_ALPHA_TAU_US, _alpha_tau_us, _fit_alpha_scaling,
)

from .sensitivities import _real_if_close, _bitstring_to_int, _raw_bulk_alpha, _raw_bulk_alpha_pauli

from .expansions import (
    _second_order_magnus_term, _second_and_third_order_magnus_terms,
    _accumulate_layer_pairwise_commutators, _truncated, _accumulate_scaled,
    _commuting_product, _raw_commuting_product,
)

from .numerical import _matrix_commutator

from .polynomials import (
    _truncate_lse_support,
    _second_order_magnus_term_symbolic_polynomial,
    _error_generator_layer_pairwise_commutator_symbolic_polynomial,
    _combined_taylor_expansion_polynomial,
)

__all__ = [
    # Submodules
    'errgentypes', 'algebra', 'parallelutils', 'sensitivities', 'expansions', 'corrections',
    'numerical', 'polynomials', 'fasterrgencalc',
    # errgentypes
    'PauliPhaseUpdater', 'PauliPhaseZerosUpdater', 'AmplitudeOfStateType', 'BulkAmplitudeOfStateType',
    'BulkPhiType', 'BulkAlphaType', 'BulkAlphaPauliType',
    # algebra
    'errgen_coeff_label_to_stim_pauli_strs', 'error_generator_commutator', 'error_generator_composition',
    'error_generator_anticommutator', 'pauli_conjugation_composition', 'com', 'acom', 'pauli_product',
    'errorgen_pauli_action', 'iterative_error_generator_composition',
    # parallelutils
    'EGPROP_PREWARM_PACKAGES', 'DEFAULT_PARALLEL_THRESHOLDS', 'configure_errorgen_propagation_parallelization',
    'optimize_errorgen_propagation_parallelization', 'measure_lse_entry_size', 'estimate_magnus_peak_memory',
    'estimate_taylor_peak_memory', 'estimate_bulk_alpha_peak_memory',
    # sensitivities
    'random_support', 'in_stabilizer_support', 'compute_phase_reference', 'PAULI_PHASES_0', 'PAULI_PHASES_1',
    'PAULI_PHASES_0_DUAL', 'PAULI_PHASES_1_DUAL', 'PAULI_FLIPS', 'slow_pauli_phase_update',
    'slow_pauli_phase_update_all_zeros', 'pauli_phase_update', 'pauli_phase_update_all_zeros',
    'slow_amplitude_of_state', 'slow_bulk_amplitude_of_state', 'amplitude_of_state',
    'bulk_amplitude_of_state', 'phi', 'slow_bulk_phi', 'bulk_phi', 'alpha', 'slow_bulk_alpha', 'bulk_alpha',
    'alpha_pauli', 'slow_bulk_alpha_pauli', 'bulk_alpha_pauli',
    # expansions
    'bch_approximation', 'magnus_expansion', 'zassenhaus_formula', 'error_generator_taylor_expansion',
    # corrections
    'tableau_fidelities', 'tableau_fidelity', 'bitstring_to_tableau', 'stabilizer_probabilities',
    'stabilizer_probability', 'stabilizer_pauli_expectation', 'stabilizer_probability_correction',
    'stabilizer_pauli_expectation_correction', 'approximate_stabilizer_probability',
    'approximate_stabilizer_pauli_expectation', 'approximate_stabilizer_probabilities',
    'approximate_stabilizer_pauli_expectations',
    # numerical
    'errorgen_layer_to_matrix', 'error_generator_commutator_numerical', 'error_generator_composition_numerical',
    'error_generator_anticommutator_numerical', 'pauli_conjugation_composition_numerical', 'bch_numerical',
    'pairwise_bch_numerical', 'magnus_numerical', 'errorgen_pauli_action_numerical', 'zassenhaus_formula_numerical',
    'iterative_error_generator_composition_numerical', 'phi_numerical', 'alpha_numerical', 'alpha_pauli_numerical',
    'stabilizer_pauli_expectation_correction_numerical', 'approximate_stabilizer_pauli_expectation_numerical',
    'error_generator_taylor_expansion_numerical',
    # polynomials
    'error_generator_to_polynomial_variable_maps', 'error_generator_to_polynomial_variable_maps_by_gate',
    'errorgen_gate_contributors', 'construct_polynomial_parameter_vector_from_propagator',
    'magnus_symbolic_polynomial', 'error_generator_taylor_expansion_symbolic_polynomial',
    'stabilizer_probability_correction_symbolic_polynomial',
    'bulk_stabilizer_probability_correction_symbolic_polynomial',
    'stabilizer_pauli_expectation_correction_symbolic_polynomial',
    'bulk_stabilizer_pauli_expectation_correction_symbolic_polynomial',
]

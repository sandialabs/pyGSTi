# ***************************************************************************************************
# Copyright 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************
"""
Propagated error-generator tomography.

This module implements the design matrix relating per-gate error-generator rates to first-order
corrections of ideal Pauli expectation values, from

    Scalable linearized gate set tomography, by Miller, Ostrove, Hines, Siekierski, Young,
    Blume-Kohout and Proctor. arXiv:2605.11158.

Notation
--------
A *rate coordinate* is a `(gate label, LocalElementaryErrorgenLabel)` pair, one per error-generator
rate a model assigns to one of its primitive operators. `error_generator_rates` enumerates them; the
enumeration order is the canonical coordinate ordering used by the design matrix built here. Rows of
the design matrix are classified `H` or `S` according to whether the circuit-and-observable pair they
belong to is first-order sensitive to Hamiltonian or to stochastic error-generator rates.
"""

import warnings as _warnings

import numpy as _np
try:
    import stim
except ImportError:
    msg = "Stim is required for use of the propagated error tomography module, " \
          "and it does not appear to be installed. If you intend to use this module please update" \
          " your environment."
    _warnings.warn(msg)

from pygsti.algorithms import randomcircuit as _rc
from pygsti.errorgenpropagation.errorpropagator import ErrorGeneratorPropagator as _ErrorGeneratorPropagator
from pygsti.protocols import protocol as _proto
from pygsti.tools import errgenpolytools as _errgenpolytools
from pygsti.tools import errgenproptools as _errgenproptools

__all__ = [
    'error_generator_rates',
    'PropagatedErrorTomographyDesign',
    'design_matrix_rank_diagnostics',
    'sample_full_rank_design',
]


def error_generator_rates(model):
    """
    The error-generator rates a model assigns, keyed by (gate label, elementary errorgen label).

    Returns
    -------
    dict
        Keys are `(pygsti.baseobjs.Label, LocalElementaryErrorgenLabel)` pairs; values are the
        corresponding rates as floats. Insertion order is the canonical coordinate ordering used
        throughout this module, so `tuple(error_generator_rates(model))` is a stable coordinate
        list for a given model.
    """
    labels = list(model.primitive_prep_labels) + list(model.primitive_op_labels) \
        + list(model.primitive_povm_labels)
    rates = {}
    for lbl in labels:
        op = model.circuit_layer_operator(lbl)
        if not hasattr(op, 'errorgen_coefficients'):
            continue
        for eglbl, rate in op.errorgen_coefficients(label_type='local').items():
            rates[(lbl, eglbl)] = rate
    return rates


def _z_type_observables(qubit_labels, max_weight):
    """
    All Z-type Pauli observables of weight 1 through `max_weight`, as `stim.PauliString`.

    `stim.PauliString` index `i` corresponds to `qubit_labels[i]`.
    """
    return list(stim.PauliString.iter_all(len(qubit_labels), min_weight=1, max_weight=max_weight,
                                          allowed_paulis='Z'))


def _hamiltonian_column_mask(rate_labels):
    """Boolean mask, True where a rate coordinate is a Hamiltonian rate."""
    return _np.array([eglbl.errorgen_type == 'H' for _, eglbl in rate_labels], dtype=bool)


def _split_rows(ideal_expectations):
    """
    Boolean (H-row, S-row) masks from the ideal expectation values.

    Every ideal expectation is exactly 0 (Hamiltonian-sensitive) or +1/-1 (stochastic-sensitive).

    Raises
    ------
    ValueError
        If some ideal expectation is not exactly 0, +1 or -1.
    """
    ideal_expectations = _np.asarray(ideal_expectations)
    h_rows = ideal_expectations == 0
    s_rows = _np.abs(ideal_expectations) == 1
    if not _np.all(h_rows | s_rows):
        bad = ideal_expectations[~(h_rows | s_rows)]
        raise ValueError(f"Ideal expectation values must be exactly 0, +1 or -1; found {bad}.")
    return h_rows, s_rows


def _circuit_design_block(model, propagator, circuit, observables, rate_index):
    """
    Sensitivities and ideal expectations for one circuit.

    Gate contributors are computed only for the polynomial variables appearing in `polys`;
    a `(gate, errorgen)` pair missing from `rate_index` is only an error if some observable's
    sensitivity actually depends on it, since an unused coordinate is not a problem.

    Returns
    -------
    sensitivities : numpy.ndarray, shape (len(observables), len(rate_index))
    ideal : numpy.ndarray, shape (len(observables),)
    """
    tmap = propagator.errorgen_transform_map(circuit, include_spam=True)
    tmaps = propagator.errorgen_transform_maps(circuit, include_spam=True)
    var_map, var_to_errorgen = _errgenpolytools.error_generator_to_polynomial_variable_maps(
        tmap, return_reverse=True)
    magnus = _errgenpolytools.magnus_symbolic_polynomial(tmaps, var_map, magnus_order=1)
    tableau = circuit.convert_to_stim_tableau()
    polys = _errgenpolytools.bulk_stabilizer_pauli_expectation_correction_symbolic_polynomial(
        magnus, var_map, tableau, observables, order=1)

    var_columns = {}
    sensitivities = _np.zeros((len(observables), len(rate_index)))
    for obs_idx, poly in enumerate(polys):
        for var_inds, coeff in poly.coeffs.items():
            if len(var_inds) == 0:
                assert abs(coeff) < 1e-12, f"unexpected nonzero constant term {coeff}"
                continue
            if len(var_inds) > 1:
                raise ValueError(
                    f"Order-1 polynomial has a degree-{len(var_inds)} term; the order-1 "
                    "assumption does not hold.")
            var_idx = var_inds[0]
            if var_idx not in var_columns:
                errorgen, layer_idx = var_to_errorgen[var_idx]
                contributors = _errgenpolytools.errorgen_gate_contributors(
                    model, errorgen, circuit, layer_idx, include_spam=True)
                local_eglbl = errorgen.to_local_eel()
                columns = []
                for gate in contributors:
                    key = (gate, local_eglbl)
                    if key not in rate_index:
                        raise ValueError(f"No rate coordinate for (gate, errorgen) pair {key}.")
                    columns.append(rate_index[key])
                var_columns[var_idx] = columns
            for col in var_columns[var_idx]:
                sensitivities[obs_idx, col] += coeff.real

    ideal = _np.array([_errgenproptools.stabilizer_pauli_expectation(tableau, obs)
                       for obs in observables])
    return sensitivities, ideal


def _design_matrix(model, circuits, observables, rate_labels):
    """
    Stack `_circuit_design_block` over `circuits`.

    Row `i * len(observables) + j` of the returned arrays is circuit `circuits[i]`, observable
    `observables[j]` (circuit-major order).

    Returns
    -------
    design_matrix : numpy.ndarray, shape (len(circuits) * len(observables), len(rate_labels))
    ideal : numpy.ndarray, shape (len(circuits) * len(observables),)

    Raises
    ------
    ValueError
        If `circuits` is empty.
    """
    if len(circuits) == 0:
        raise ValueError("Cannot build a design matrix from an empty circuit list.")
    rate_index = {label: i for i, label in enumerate(rate_labels)}
    propagator = _ErrorGeneratorPropagator(model.copy())
    blocks, ideals = [], []
    for circuit in circuits:
        sensitivities, ideal = _circuit_design_block(model, propagator, circuit, observables,
                                                     rate_index)
        blocks.append(sensitivities)
        ideals.append(ideal)
    design_matrix = _np.vstack(blocks)
    ideal_values = _np.concatenate(ideals)
    return design_matrix, ideal_values


class PropagatedErrorTomographyDesign(_proto.ExperimentDesign):
    """
    A flat set of shallow random circuits, plus the provenance of how they were sampled. The
    ansatz and observable set a rate estimate is built from are not design ingredients, per
    `ExperimentDesign`'s separation of "what data to take" from "how to interpret it".

    Parameters
    ----------
    pspec : QubitProcessorSpec
        The processor spec the circuits are sampled for.

    depth : int
        The depth of every sampled circuit.

    num_circuits : int
        The number of circuits to sample.

    qubit_labels : list, optional
        The qubits to sample over. If `None`, `tuple(pspec.qubit_labels)`.

    sampler : str or function, optional
        Passed to `create_random_circuit`.

    samplerargs : list, optional
        Passed to `create_random_circuit`. If `None`, `[0.25]`, which is the two-qubit gate
        density `'edgegrab'` requires and the default the randomized-benchmarking designs use.

    seed : int, optional
        Seeds a `numpy.random.RandomState` shared across every sampled circuit, so a given seed
        reproduces the whole circuit set. Must be `None` or an `int`.

    descriptor : str, optional
        A string describing the design.
    """

    @classmethod
    def from_circuits(cls, circuits, qubit_labels=None, descriptor=None):
        """
        Build a design from circuits sampled elsewhere. Provenance -- `depth`, `sampler`,
        `samplerargs` and `seed` -- is genuinely unknown for externally supplied circuits, so
        each is set to `None` rather than guessed.

        Parameters
        ----------
        circuits : list of Circuit
            The circuits making up the design.

        qubit_labels : list, optional
            The qubits the design applies to.

        descriptor : str, optional
            A string describing the design.

        Returns
        -------
        PropagatedErrorTomographyDesign
        """
        self = cls.__new__(cls)
        self._init_foundation(circuits, qubit_labels, None, len(circuits), None, None, None,
                              descriptor)
        return self

    def __init__(self, pspec, depth, num_circuits, qubit_labels=None, *,
                 sampler='edgegrab', samplerargs=None, seed=None, descriptor=None):
        if seed is not None and not isinstance(seed, int):
            raise ValueError(f"seed must be None or an int; got {seed!r}.")
        if samplerargs is None:
            samplerargs = [0.25, ]
        if qubit_labels is None:
            qubit_labels = tuple(pspec.qubit_labels)
        rand_state = _np.random.RandomState(seed)
        circuits = [_rc.create_random_circuit(pspec, depth, qubit_labels=qubit_labels,
                                              sampler=sampler, samplerargs=samplerargs,
                                              rand_state=rand_state)
                    for _ in range(num_circuits)]
        self._init_foundation(circuits, qubit_labels, depth, num_circuits, sampler, samplerargs,
                              seed, descriptor)

    def _init_foundation(self, circuits, qubit_labels, depth, num_circuits, sampler, samplerargs,
                         seed, descriptor):
        """Shared constructor body for `__init__` and `from_circuits`."""
        super().__init__(circuits, qubit_labels)
        self.depth = depth
        self.num_circuits = num_circuits
        if sampler is None or isinstance(sampler, str):
            self.sampler = sampler
        else:
            self.sampler = 'function'
        self.samplerargs = samplerargs
        self.seed = seed
        self.descriptor = descriptor


def _block_rank_diagnostics(block, rank_tol):
    """
    Rank, deficit, conditioning and singular values of one design-matrix block.

    Returns
    -------
    dict
        Keys `num_rows`, `num_rates`, `rank`, `deficit`, `condition_number`, `singular_values`.
    """
    num_rows, num_rates = block.shape
    singular_values = _np.linalg.svd(block, compute_uv=False)
    sigma_max = singular_values[0] if singular_values.size else 0.0
    tol = rank_tol if rank_tol is not None else max(block.shape) * _np.finfo(block.dtype).eps * sigma_max
    nonzero = singular_values[singular_values > tol]
    rank = nonzero.size
    condition_number = float(sigma_max / nonzero[-1]) if rank > 0 else float('inf')
    return {
        'num_rows': num_rows,
        'num_rates': num_rates,
        'rank': rank,
        'deficit': num_rates - rank,
        'condition_number': condition_number,
        'singular_values': singular_values,
    }


def design_matrix_rank_diagnostics(circuits, model, *, observables=None, max_weight=2,
                                   rank_tol=None):
    """
    Rank and conditioning of the H and S design-matrix blocks for a given ansatz.

    Takes a bare circuit iterable, not a design, since `sample_full_rank_design` calls this
    against a growing list; pass `edesign.all_circuits_needing_data` otherwise.

    Parameters
    ----------
    circuits : iterable of Circuit
        Each circuit's `line_labels` must equal `tuple(model.state_space.qubit_labels)`.

    model : Model
        The ansatz. Its `state_space.qubit_labels` is the authority on qubit ordering, since the
        design matrix's columns come from the model.

    observables : list of stim.PauliString, optional
        If `None`, all Z-type Pauli observables of weight 1 through `max_weight`.

    max_weight : int, optional
        Used only when `observables` is `None`.

    rank_tol : float, optional
        Absolute singular-value threshold below which a singular value is treated as zero. If
        `None`, numpy's default, `max(block.shape) * eps * sigma_max`, computed per block.

    Returns
    -------
    dict
        Keys `num_circuits`, `num_observables`, `num_rates`, `rank_tol`, `hamiltonian` and
        `stochastic`, the last two being `_block_rank_diagnostics` dicts.

    Raises
    ------
    ValueError
        If any circuit's `line_labels` disagree with the model's qubit ordering.
    """
    circuits = list(circuits)
    qubit_labels = tuple(model.state_space.qubit_labels)
    for circuit in circuits:
        if circuit.line_labels != qubit_labels:
            raise ValueError(
                f"Circuit line labels {circuit.line_labels} disagree with the model's qubit "
                f"ordering {qubit_labels}; a stim.PauliString observable would be attributed to "
                "the wrong qubits.")
    if observables is None:
        observables = _z_type_observables(qubit_labels, max_weight)
    rate_labels = list(error_generator_rates(model))
    design_matrix, ideal = _design_matrix(model, circuits, observables, rate_labels)

    h_rows, s_rows = _split_rows(ideal)
    is_h_col = _hamiltonian_column_mask(rate_labels)
    h_block = design_matrix[h_rows][:, is_h_col]
    s_block = design_matrix[s_rows][:, ~is_h_col]
    return {
        'num_circuits': len(circuits),
        'num_observables': len(observables),
        'num_rates': len(rate_labels),
        'rank_tol': rank_tol,
        'hamiltonian': _block_rank_diagnostics(h_block, rank_tol),
        'stochastic': _block_rank_diagnostics(s_block, rank_tol),
    }


def sample_full_rank_design(pspec, model, depth, qubit_labels=None, *, observables=None,
                            max_weight=2, sampler='edgegrab', samplerargs=None,
                            batch_size=25, max_circuits=None, rank_tol=None, seed=None):
    """
    Sample circuits in batches until the design matrix attains full column rank.

    Circuits are sampled `batch_size` at a time; `design_matrix_rank_diagnostics` is recomputed
    against the accumulated circuit list after each batch, until both the H and S blocks are
    full column rank.

    Termination, absent `max_circuits`, is guaranteed by saturation detection: if the three most
    recent consecutive batches produced no increase in either block's rank, the rank is treated
    as saturated -- a symptom of a gauge freedom in the ansatz, since real gauge-free ansatzes
    keep gaining rank as circuits accumulate. That is a different failure than running out of
    `max_circuits` while the rank is still climbing, and the two raise distinguishable messages.

    Parameters
    ----------
    pspec : QubitProcessorSpec
        The processor spec the circuits are sampled for.

    model : Model
        The ansatz guiding the rank check. Not stored on the returned design.

    depth : int
        The depth of every sampled circuit.

    qubit_labels : list, optional
        The qubits to sample over. If `None`, `tuple(pspec.qubit_labels)`.

    observables : list of stim.PauliString, optional
        If `None`, all Z-type Pauli observables of weight 1 through `max_weight`. Not stored on
        the returned design.

    max_weight : int, optional
        Used only when `observables` is `None`.

    sampler : str or function, optional
        Passed to `create_random_circuit`.

    samplerargs : list, optional
        Passed to `create_random_circuit`. If `None`, `[0.25]`.

    batch_size : int, optional
        Number of circuits sampled per batch.

    max_circuits : int, optional
        If `None`, no circuit-count limit; termination relies on saturation detection instead.

    rank_tol : float, optional
        Passed to `design_matrix_rank_diagnostics`.

    seed : int, optional
        Seeds a `numpy.random.RandomState` shared across every sampled circuit. Must be `None`
        or an `int`.

    Returns
    -------
    design : PropagatedErrorTomographyDesign
        Carries full provenance (`depth`, `sampler`, `samplerargs`, `seed`) but no trace of
        `model` or `observables`.

    diagnostics : dict
        The final `design_matrix_rank_diagnostics` dict, plus `rank_history` (a tuple of
        `(num_circuits, hamiltonian_rank, stochastic_rank)` after each batch), `full_rank` and
        `saturated`.

    Raises
    ------
    ValueError
        If the rank saturates below full column rank, or if `max_circuits` is reached while the
        rank is still increasing. The two messages are distinguishable.
    """
    if seed is not None and not isinstance(seed, int):
        raise ValueError(f"seed must be None or an int; got {seed!r}.")
    if samplerargs is None:
        samplerargs = [0.25, ]
    if qubit_labels is None:
        qubit_labels = tuple(pspec.qubit_labels)

    rand_state = _np.random.RandomState(seed)
    circuits = []
    rank_history = []
    diagnostics = None
    while max_circuits is None or len(circuits) < max_circuits:
        n_new = batch_size if max_circuits is None else min(batch_size, max_circuits - len(circuits))
        circuits.extend(_rc.create_random_circuit(pspec, depth, qubit_labels=qubit_labels,
                                                  sampler=sampler, samplerargs=samplerargs,
                                                  rand_state=rand_state)
                        for _ in range(n_new))
        diagnostics = design_matrix_rank_diagnostics(circuits, model, observables=observables,
                                                     max_weight=max_weight, rank_tol=rank_tol)
        h_rank = diagnostics['hamiltonian']['rank']
        s_rank = diagnostics['stochastic']['rank']
        rank_history.append((len(circuits), h_rank, s_rank))

        if diagnostics['hamiltonian']['deficit'] == 0 and diagnostics['stochastic']['deficit'] == 0:
            design = PropagatedErrorTomographyDesign.__new__(PropagatedErrorTomographyDesign)
            design._init_foundation(circuits, qubit_labels, depth, len(circuits), sampler,
                                    samplerargs, seed, None)
            diagnostics['rank_history'] = tuple(rank_history)
            diagnostics['full_rank'] = True
            diagnostics['saturated'] = False
            return design, diagnostics

        if len(rank_history) >= 4 and all(rank_history[-1][1:] == rank_history[-1 - k][1:]
                                          for k in range(1, 4)):
            raise ValueError(
                f"Rank saturated at {len(circuits)} circuits, with an H-block deficit of "
                f"{diagnostics['hamiltonian']['deficit']} and an S-block deficit of "
                f"{diagnostics['stochastic']['deficit']}: the rank stopped increasing over the "
                "last three batches, which indicates a gauge freedom in the ansatz rather than "
                "too few circuits. Under every in-built pyGSTi sampler, an ansatz whose gates "
                "all carry the same spectator-error parameters saturates with an H-block "
                "deficit of n(n-2), because those samplers emit layers acting on every qubit; "
                "the remedy is a different sampler or a different ansatz, not more circuits.")

    raise ValueError(
        f"Reached max_circuits={max_circuits} circuits with the rank still increasing "
        f"(H-block deficit {diagnostics['hamiltonian']['deficit']}, S-block deficit "
        f"{diagnostics['stochastic']['deficit']}); try a larger max_circuits.")

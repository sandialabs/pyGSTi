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

from pygsti.errorgenpropagation.errorpropagator import ErrorGeneratorPropagator as _ErrorGeneratorPropagator
from pygsti.tools import errgenpolytools as _errgenpolytools
from pygsti.tools import errgenproptools as _errgenproptools

__all__ = [
    'error_generator_rates',
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
    for var_idx, (errorgen, layer_idx) in var_to_errorgen.items():
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
            for col in var_columns[var_inds[0]]:
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
    """
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

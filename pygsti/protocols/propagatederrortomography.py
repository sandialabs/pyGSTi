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
import pandas as _pd
from scipy.optimize import nnls as _nnls
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
from pygsti.tools import matrixtools as _mt

__all__ = [
    'error_generator_rates',
    'PropagatedErrorTomographyDesign',
    'design_matrix_rank_diagnostics',
    'sample_full_rank_design',
    'PropagatedErrorTomography',
    'PropagatedErrorTomographyResults',
]

# `numpy.linalg.pinv`'s own default relative cutoff, passed explicitly wherever the rank it
# implies is also recorded, so that an estimate and its covariance provably share one subspace.
_PINV_RCOND = 1e-15


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


def _observable_signs(observables, qubit_labels):
    """
    The sign each observable assigns to each measurement outcome.

    Parameters
    ----------
    observables : list of stim.PauliString
        The Z-type observables.

    qubit_labels : tuple
        `tuple(model.state_space.qubit_labels)`. Fixes the bit-to-qubit correspondence: bit `i`
        of an outcome string is the measurement result of `qubit_labels[i]`.

    Returns
    -------
    signs : numpy.ndarray, shape (len(observables), 2**len(qubit_labels))
        Entry `(j, b)` is observable `j`'s sign on outcome `b`: its own sign times `(-1)` raised
        to the parity of its support bits in `b`.

    outcomes : list of str
        The outcome bitstrings, in the column order of `signs`.
    """
    num_qubits = len(qubit_labels)
    outcomes = [format(b, f'0{num_qubits}b') for b in range(2 ** num_qubits)]
    bits = _np.array([[int(bit) for bit in outcome] for outcome in outcomes])
    support = _np.array([[obs[i] != 0 for i in range(num_qubits)] for obs in observables])
    obs_signs = _np.array([obs.sign.real for obs in observables])
    parity = (bits @ support.T) % 2  # (num_outcomes, num_observables)
    return obs_signs[:, None] * (1 - 2 * parity).T, outcomes


def _observed_expectations(dataset, circuits, observables, qubit_labels):
    """
    Raw, per-circuit `<Q>` estimates for each observable, with the outcome-count data behind
    them.

    These are the plain empirical means: for observable `Q` and circuit with outcome fractions
    `f`, `<Q> = sum_b sign(b) * f(b)`, where `sign(b)` is `(-1)` raised to the parity of `Q`'s
    support bits in outcome `b`.

    Parameters
    ----------
    dataset : DataSet
        Supplies per-circuit outcome counts, via `dataset[circuit].counts` and
        `dataset[circuit].total`.

    circuits : list of Circuit
        The circuits to read data for, in order.

    observables : list of stim.PauliString
        The Z-type observables to estimate `<Q>` for.

    qubit_labels : tuple
        `tuple(model.state_space.qubit_labels)`. Fixes the bit-to-qubit correspondence: bit `i`
        of an outcome string is the measurement result of `qubit_labels[i]`.

    Returns
    -------
    observed : numpy.ndarray, shape (len(circuits), len(observables))
        Raw (unsmoothed) `<Q>` estimates.

    counts : numpy.ndarray, shape (len(circuits), 2**len(qubit_labels))
        Per-outcome counts. Column `b` is the outcome whose bitstring is `b` written in binary
        with `len(qubit_labels)` digits, e.g. column 0 is the all-zeros outcome.

    totals : numpy.ndarray, shape (len(circuits),)
        Per-circuit shot totals.
    """
    signs, outcomes = _observable_signs(observables, qubit_labels)
    counts = _np.array([[dataset[circuit].counts.get((outcome,), 0.0) for outcome in outcomes]
                        for circuit in circuits])
    totals = _np.array([dataset[circuit].total for circuit in circuits])
    observed = (counts @ signs.T) / totals[:, None]
    return observed, counts, totals


def _multinomial_covariance(signs, circuit_counts, total):
    """
    The smoothed multinomial covariance of one circuit's `<Q>` estimates.

    Applies add-one (Laplace) smoothing to every outcome's estimated probability before forming
    first and second moments, which keeps the covariance finite even for a circuit with a
    deterministic outcome. The smoothing exists only to keep these weights finite; it is scoped
    to this covariance estimate alone.

    Parameters
    ----------
    signs : numpy.ndarray, shape (num_observables, num_outcomes)
        Row `j` is observable `j`'s sign for each outcome, as built by `_observed_expectations`.

    circuit_counts : numpy.ndarray, shape (num_outcomes,)
        Per-outcome counts for one circuit.

    total : float
        That circuit's shot total.

    Returns
    -------
    numpy.ndarray, shape (num_observables, num_observables)
    """
    u = circuit_counts / (total + 2)
    m = signs @ u
    m2 = (signs * u) @ signs.T
    return (m2 - _np.outer(m, m)) / total


def _whitener(covariance, tol):
    """
    The inverse square root `C^{-1/2}` of a PSD covariance matrix.

    Eigenvalues below `tol * lambda_max` are clipped up to that floor before inverting, since
    `covariance` is typically singular (rather than merely ill-conditioned) whenever the number
    of observables exceeds the number of outcomes with nonzero probability.

    Parameters
    ----------
    covariance : numpy.ndarray, shape (n, n)
        A symmetric positive-semidefinite matrix.

    tol : float
        The relative eigenvalue-clipping tolerance.

    Returns
    -------
    numpy.ndarray, shape (n, n)
    """
    evecs, evals, _ = _mt.eigendecomposition(covariance, assume_hermitian=True)
    clipped = _np.maximum(evals, tol * evals.max())
    return evecs @ _np.diag(clipped ** -0.5) @ evecs.T


def _solve_block(block, rhs, rank_tol, ill_posed_action, non_negative):
    """
    Solve one whitened design-matrix block, by ordinary or non-negative least squares.

    The two branches differ in whether rank deficiency can be worked around by truncation: an
    ordinary least-squares problem has a well-defined minimum-norm solution on any subspace, so
    `ill_posed_action` can select a truncated solve; a non-negative least-squares problem does
    not decompose that way, so the stochastic block is always solved on the full matrix and
    `ill_posed_action` only controls whether/how the rank deficiency is reported.

    Parameters
    ----------
    block : numpy.ndarray, shape (num_rows, num_columns)
        The whitened design-matrix block.

    rhs : numpy.ndarray, shape (num_rows,)
        The whitened right-hand side.

    rank_tol : float, optional
        Passed to `_block_rank_diagnostics`.

    ill_posed_action : {'error', 'warn', 'truncate-loud', 'truncate-quiet'}
        What to do when `block` is column-rank-deficient.

    non_negative : bool
        `False` solves an ordinary least-squares problem (the Hamiltonian block); `True` solves
        via `scipy.optimize.nnls` (the stochastic block).

    Returns
    -------
    solution : numpy.ndarray, shape (num_columns,)

    block_diagnostics : dict
        `_block_rank_diagnostics(block, rank_tol)`, plus `'chi2'`, `'dof'`, and either
        `'num_truncated'` and `'solve_rank'` (non-negative is `False`) or `'num_active'`
        (non-negative is `True`). `'solve_rank'` is the dimension of the subspace the returned
        solution lives in, which under `'warn'` is set by `numpy.linalg.pinv`'s tolerance rather
        than by `rank_tol`.

    Raises
    ------
    ValueError
        If `block` is column-rank-deficient and `ill_posed_action == 'error'`.
    """
    num_rows, num_columns = block.shape
    diagnostics = _block_rank_diagnostics(block, rank_tol)
    rank, deficit = diagnostics['rank'], diagnostics['deficit']

    if non_negative:
        if deficit > 0:
            message = (f"Stochastic block has rank {rank} of {num_columns} columns (deficit "
                       f"{deficit}, condition number {diagnostics['condition_number']:.3g}); "
                       "the non-negative solve proceeds on the full matrix regardless.")
            if ill_posed_action == 'error':
                raise ValueError(message)
            elif ill_posed_action in ('warn', 'truncate-loud'):
                _warnings.warn(message)
        solution, _ = _nnls(block, rhs)
        diagnostics['num_active'] = int(_np.sum(solution > 0))
        diagnostics['dof'] = num_rows - diagnostics['num_active']
    else:
        if deficit == 0:
            solution, _, _, _ = _np.linalg.lstsq(block, rhs, rcond=None)
            diagnostics['num_truncated'] = 0
            diagnostics['solve_rank'] = rank
        else:
            message = (f"Hamiltonian block has rank {rank} of {num_columns} columns "
                       f"(deficit {deficit}, condition number "
                       f"{diagnostics['condition_number']:.3g}).")
            if ill_posed_action == 'error':
                raise ValueError(message)
            elif ill_posed_action == 'warn':
                singular_values = diagnostics['singular_values']
                solution = _np.linalg.pinv(block, rcond=_PINV_RCOND) @ rhs
                diagnostics['num_truncated'] = 0
                diagnostics['solve_rank'] = int(_np.sum(
                    singular_values > _PINV_RCOND * singular_values[0]))
                _warnings.warn(message + " Solved with numpy.linalg.pinv at its default "
                               "tolerance; the estimate is not trustworthy.")
            else:
                u, s, vt = _np.linalg.svd(block, full_matrices=False)
                solution = vt[:rank].T @ ((u[:, :rank].T @ rhs) / s[:rank])
                diagnostics['num_truncated'] = num_columns - rank
                diagnostics['solve_rank'] = rank
                if ill_posed_action == 'truncate-loud':
                    _warnings.warn(message + f" Solved the rank-{rank} truncated problem.")
        diagnostics['dof'] = num_rows - rank

    diagnostics['chi2'] = float(_np.sum((block @ solution - rhs) ** 2))
    return solution, diagnostics


class PropagatedErrorTomography(_proto.Protocol):
    """
    Estimate error-generator rates from propagated-error-tomography data, by first-order
    linearized least squares.

    For each Z-type observable and each circuit, `_design_matrix` gives a first-order Taylor
    model of `<Q>` in the model's error-generator rates; this protocol estimates those rates by
    inverting that model against the deviation of the observed `<Q>` from its ideal (noiseless)
    value. Rows are classified `H` or `S` by `_split_rows` and solved as two independent linear
    problems: the Hamiltonian block by (optionally truncated) least squares, the stochastic
    block by non-negative least squares (`scipy.optimize.nnls`), since stochastic rates are
    non-negative by construction. The stochastic block is always solved on its full matrix: the
    non-negativity constraint frequently makes the solution unique on its own, so
    `ill_posed_action` there only controls how a rank deficiency is reported, never whether the
    solved system is truncated.

    Parameters
    ----------
    model : Model
        The ansatz relating circuits to predicted `<Q>` values. Stored as `self.model`, not
        copied (the internal error-generator propagator copies it as needed). Its
        `state_space.qubit_labels` is the authority on qubit ordering for both observables and
        outcome bitstrings.

    k : int, optional
        The propagation order. Only `k == 1` is implemented.

    max_weight : int, optional
        Used only when `observables` is `None`, in which case the observable set is every
        Z-type Pauli of weight `1` through `max_weight`.

    observables : list of stim.PauliString, optional
        The observables to use. If `None`, determined from `max_weight`.

    weighting : {'multinomial', 'none'}, optional
        `'multinomial'` whitens each circuit's H-rows and S-rows separately by the inverse
        square root of a smoothed multinomial covariance estimate. `'none'` solves the
        unweighted system.

    ill_posed_action : {'error', 'warn', 'truncate-loud', 'truncate-quiet'}, optional
        What to do if the Hamiltonian or stochastic design-matrix block is column-rank-deficient
        at `rank_tol`. `'error'` raises `ValueError`. `'warn'` solves via `numpy.linalg.pinv` at
        its default tolerance and warns. `'truncate-loud'` solves the numerically-truncated
        problem and warns. `'truncate-quiet'` does the same without warning. The stochastic
        block is always solved on its full matrix; there, `'error'` still raises and `'warn'` /
        `'truncate-loud'` still warn, each describing the rank deficiency in the message.

    rank_tol : float, optional
        The absolute singular-value threshold below which a design-matrix block is treated as
        rank-deficient. If `None`, `max(num_rows, num_columns) * eps * sigma_max`, computed
        separately for each block.

    name : str, optional
        The name of this protocol.
    """

    def __init__(self, model, k=1, *, max_weight=2, observables=None, weighting='multinomial',
                 ill_posed_action='truncate-loud', rank_tol=None, name=None):
        if k == 1:
            pass
        elif k in (2, 3):
            raise NotImplementedError(
                f"Order-{k} propagated error tomography is not yet implemented; it is follow-up "
                "work beyond the order-1 protocol implemented here.")
        else:
            raise ValueError(f"k must be 1, 2 or 3; got {k!r}.")
        if weighting not in ('multinomial', 'none'):
            raise ValueError(f"weighting must be 'multinomial' or 'none'; got {weighting!r}.")
        if ill_posed_action not in ('error', 'warn', 'truncate-loud', 'truncate-quiet'):
            raise ValueError(
                "ill_posed_action must be one of 'error', 'warn', 'truncate-loud', "
                f"'truncate-quiet'; got {ill_posed_action!r}.")

        super().__init__(name)
        self.auxfile_types['model'] = 'serialized-object'
        self.model = model
        self.k = k
        self.max_weight = max_weight
        self.observables = observables
        self.weighting = weighting
        self.ill_posed_action = ill_posed_action
        self.rank_tol = rank_tol

    def run(self, data, memlimit=None, comm=None):
        """
        Run this protocol on `data`.

        Parameters
        ----------
        data : ProtocolData
            Supplies `data.dataset` and `data.edesign.all_circuits_needing_data`. The edesign
            need not be a `PropagatedErrorTomographyDesign`; any edesign whose circuits are
            consistent with `self.model`'s qubit ordering is acceptable.

        memlimit : int, optional
            Unused (present for `Protocol` interface compatibility).

        comm : mpi4py.MPI.Comm, optional
            Unused (present for `Protocol` interface compatibility).

        Returns
        -------
        PropagatedErrorTomographyResults
        """
        circuits = list(data.edesign.all_circuits_needing_data)
        qubit_labels = tuple(self.model.state_space.qubit_labels)
        observables = (self.observables if self.observables is not None
                       else _z_type_observables(qubit_labels, self.max_weight))
        rate_labels = list(error_generator_rates(self.model))
        is_h_col = _hamiltonian_column_mask(rate_labels)
        num_circuits, num_observables = len(circuits), len(observables)

        design_matrix, ideal = _design_matrix(self.model, circuits, observables, rate_labels)
        observed, counts, totals = _observed_expectations(data.dataset, circuits, observables,
                                                          qubit_labels)
        delta = (observed - ideal.reshape(num_circuits, num_observables)).ravel()
        h_rows, s_rows = _split_rows(ideal)

        if self.weighting == 'none':
            dt_h, rhs_h = design_matrix[h_rows][:, is_h_col], delta[h_rows]
            dt_s, rhs_s = design_matrix[s_rows][:, ~is_h_col], delta[s_rows]
        else:
            signs, _ = _observable_signs(observables, qubit_labels)
            h_blocks, h_rhs, s_blocks, s_rhs = [], [], [], []
            for i in range(num_circuits):
                lo, hi = i * num_observables, (i + 1) * num_observables
                cov = _multinomial_covariance(signs, counts[i], totals[i])
                h_mask, s_mask = h_rows[lo:hi], s_rows[lo:hi]
                if _np.any(h_mask):
                    w = _whitener(cov[_np.ix_(h_mask, h_mask)], 1e-12)
                    h_blocks.append(w @ design_matrix[lo:hi][h_mask][:, is_h_col])
                    h_rhs.append(w @ delta[lo:hi][h_mask])
                if _np.any(s_mask):
                    w = _whitener(cov[_np.ix_(s_mask, s_mask)], 1e-12)
                    s_blocks.append(w @ design_matrix[lo:hi][s_mask][:, ~is_h_col])
                    s_rhs.append(w @ delta[lo:hi][s_mask])
            if not h_blocks:
                raise ValueError(
                    "No circuit in this design contributes a Hamiltonian-sensitive row; the "
                    "Hamiltonian block cannot be estimated from these circuits and observables.")
            if not s_blocks:
                raise ValueError(
                    "No circuit in this design contributes a stochastic-sensitive row; the "
                    "stochastic block cannot be estimated from these circuits and observables.")
            dt_h, rhs_h = _np.vstack(h_blocks), _np.concatenate(h_rhs)
            dt_s, rhs_s = _np.vstack(s_blocks), _np.concatenate(s_rhs)

        solution_h, diag_h = _solve_block(dt_h, rhs_h, self.rank_tol, self.ill_posed_action, False)
        solution_s, diag_s = _solve_block(dt_s, rhs_s, self.rank_tol, self.ill_posed_action, True)

        # Build the Hamiltonian covariance on `solve_rank`, the subspace `_solve_block` actually
        # solved on, so the reported 1-sigma values describe the reported estimate. Forming it
        # from the SVD rather than as `pinv` of the Gram matrix keeps that subspace exact: `pinv`
        # would apply its own relative tolerance to the *squared* singular values, a different
        # cutoff from the one the solve used.
        rank_h = diag_h['solve_rank']
        _, s_h, vt_h = _np.linalg.svd(dt_h, full_matrices=False)
        v_trunc = vt_h[:rank_h] / s_h[:rank_h, None]
        uncertainties_h = _np.linalg.norm(v_trunc, axis=0)
        active = solution_s > 0
        uncertainties_s = _np.zeros(solution_s.shape[0])
        if _np.any(active):
            cov_s = _np.linalg.pinv(dt_s[:, active].T @ dt_s[:, active])
            uncertainties_s[active] = _np.sqrt(_np.clip(_np.diag(cov_s), 0, None))

        rates = _np.zeros(len(rate_labels))
        rates[is_h_col] = solution_h
        rates[~is_h_col] = solution_s
        uncertainties = _np.zeros(len(rate_labels))
        uncertainties[is_h_col] = uncertainties_h
        uncertainties[~is_h_col] = uncertainties_s

        diagnostics = {
            'num_circuits': num_circuits,
            'num_observables': num_observables,
            'num_rates': len(rate_labels),
            'rank_tol': self.rank_tol,
            'weighting': self.weighting,
            'hamiltonian': diag_h,
            'stochastic': diag_s,
        }
        return PropagatedErrorTomographyResults(data, self, rates, uncertainties,
                                                tuple(rate_labels), diagnostics)


class PropagatedErrorTomographyResults(_proto.ProtocolResults):
    """
    The rates, uncertainties and fit diagnostics from running `PropagatedErrorTomography`.

    Parameters
    ----------
    data : ProtocolData
        The data these results were computed from.

    protocol_instance : PropagatedErrorTomography
        The protocol that produced these results.

    rates : numpy.ndarray, shape (kappa,)
        The estimated error-generator rates, ordered by `rate_labels`. `kappa ==
        len(error_generator_rates(protocol_instance.model))`; every coordinate is present,
        including ones the data does not constrain.

    uncertainties : numpy.ndarray, shape (kappa,)
        1-sigma uncertainties from linear propagation through the whitened blocks. A stochastic
        coordinate pinned at exactly zero by the non-negativity constraint is reported as
        `0.0`: its true sampling distribution is one-sided (non-Gaussian) at the boundary, so
        this value is a lower bound on the uncertainty, not an estimate of it.

    rate_labels : tuple of (Label, LocalElementaryErrorgenLabel)
        The coordinate list `rates` and `uncertainties` are ordered by.

    diagnostics : dict
        Keys `'num_circuits'`, `'num_observables'`, `'num_rates'`, `'rank_tol'`, `'weighting'`,
        and `'hamiltonian'` / `'stochastic'`. The latter two are `_block_rank_diagnostics`
        dictionaries computed on the whitened block, plus `'chi2'` (the whitened residual sum of
        squares), `'dof'`, and either `'num_truncated'` (hamiltonian) or `'num_active'`
        (stochastic).
    """

    def __init__(self, data, protocol_instance, rates, uncertainties, rate_labels, diagnostics):
        super().__init__(data, protocol_instance)
        self.rates = rates
        self.uncertainties = uncertainties
        self.rate_labels = tuple(rate_labels)
        self.diagnostics = diagnostics
        self.auxfile_types['rates'] = 'numpy-array'
        self.auxfile_types['uncertainties'] = 'numpy-array'
        self.auxfile_types['rate_labels'] = 'pickle'
        self.auxfile_types['diagnostics'] = 'pickle'

    def to_model(self):
        """
        This protocol's ansatz, with error-generator coefficients set from `rates`.

        Coordinates that `rate_labels` does not cover are left at the ansatz's own value.

        Returns
        -------
        Model
        """
        model = self.protocol.model.copy()
        by_gate = {}
        for (gate, eglbl), rate in zip(self.rate_labels, self.rates):
            by_gate.setdefault(gate, {})[eglbl] = rate
        for gate, coeffs in by_gate.items():
            model.circuit_layer_operator(gate).set_errorgen_coefficients(coeffs)
        return model

    def rates_dataframe(self):
        """
        A `pandas.DataFrame` rate-table export, with one row per rate coordinate.

        Returns
        -------
        pandas.DataFrame
            Columns `gate`, `errorgen`, `type`, `rate`, `uncertainty`, in `rate_labels` order.
        """
        return _pd.DataFrame({
            'gate': [str(gate) for gate, _ in self.rate_labels],
            'errorgen': [str(eglbl) for _, eglbl in self.rate_labels],
            'type': [eglbl.errorgen_type for _, eglbl in self.rate_labels],
            'rate': self.rates,
            'uncertainty': self.uncertainties,
        })

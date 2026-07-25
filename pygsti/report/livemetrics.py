"""
Gauge-invariant metric extraction from GST checkpoints.

This module provides small, pure (side-effect-free) helper functions for
extracting *gauge-invariant* diagnostic quantities from the checkpoints that
:class:`~pygsti.protocols.gst.GateSetTomography` (and its subclasses/callers,
e.g. :class:`~pygsti.protocols.gst.StandardGST`) write to disk during a run.

The motivating use case is a "live" monitoring tool that watches a
currently-running (or already-completed) GST fit's checkpoint directory and
reports on fit progress *before* gauge optimization has been performed on the
final result. Because gauge transformations act on gate (superoperator)
matrices by similarity transform, most familiar reportable quantities (e.g.
process fidelity to a target, SPAM tables, diamond-norm-to-target) are
*meaningless* (or actively misleading) prior to gauge optimization: their
values depend on an essentially arbitrary choice of gauge picked up in the
course of the fit. Two quantities are safe to report before gauge
optimization has happened:

    1. Goodness-of-fit statistics (e.g. chi2/2*deltaLogL, degrees of freedom,
       p-value). These measure how well the *model family* explains the
       *data*, and are computed purely from probabilities, which are
       gauge-invariant by construction (physically-equivalent models -
       related by a gauge transformation - predict identical probabilities).

    2. Gate eigenvalues. Gauge transformations act on a gate's dense
       superoperator matrix G as a similarity transform, G -> M G M^{-1},
       which by definition preserves eigenvalues.

This module intentionally does **not** attempt to recompute goodness-of-fit
statistics from scratch (e.g. by re-running a forward simulation over the
circuit list) - the objective-function evaluations already performed during
ordinary GST fitting are captured cheaply into the checkpoint itself (see
`GateSetTomography.run`'s handling of `per_iter_gof`, and
`pygsti.protocols.gst._compute_iteration_gof_summary`), since forward
simulation is normally the dominant cost of a GST fit and should not be
duplicated just to service a monitoring tool.
"""
#***************************************************************************************************
# Copyright 2015, 2019, 2025 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

import json as _json

import numpy as _np
import scipy.stats as _stats

from pygsti.tools import matrixtools as _mt


def load_checkpoint_state(path):
    """
    Load the raw (JSON-deserialized, but not yet reconstructed-into-objects)
    state dictionary for a `GateSetTomographyCheckpoint` written to disk.

    This is deliberately *not* the same as calling
    `GateSetTomographyCheckpoint.read(path)`, which would fully deserialize
    every model in `mdl_list` into a `Model` object - an expensive operation
    that scales with the (monotonically growing) number of completed
    iterations. Callers that only need the latest model (e.g. for computing
    gauge-invariant eigenvalues) should use `latest_model_from_state` below,
    which only deserializes the single model that is actually needed.

    Parameters
    ----------
    path : str or Path
        Path to a `<checkpoint_prefix>_iteration_<N>.json` file written by
        `GateSetTomography.run`.

    Returns
    -------
    dict
        The raw JSON content of the checkpoint file.
    """
    with open(str(path), 'r') as f:
        return _json.load(f)


def unwrap_gst_checkpoint_state(state, mode=None):
    """
    Normalize a raw checkpoint state to the (bare) `GateSetTomographyCheckpoint`
    state dictionary it contains, transparently handling both of the shapes a
    checkpoint file on disk can take:

    - A bare `GateSetTomographyCheckpoint` (written by a standalone
      `GateSetTomography.run(...)` call). Returned unchanged.

    - A `StandardGSTCheckpoint` (written when running `StandardGST`, which
      fits several modes, e.g. 'full TP', 'CPTPLND', as child protocols).
      Because `ProtocolCheckpoint.write` delegates to a checkpoint's parent
      when one is set, *every* per-mode iteration checkpoint file written in
      the course of a `StandardGST` run actually contains a serialization of
      the *entire* `StandardGSTCheckpoint` (all modes), with each mode's own
      `GateSetTomographyCheckpoint` (or `ModelTestCheckpoint`, for modes like
      "Target" that don't run an iterative fit) nested under
      `state['children'][mode]`. This function extracts the requested mode's
      inner state so callers don't need to know about this wrinkle.

    Parameters
    ----------
    state : dict
        A raw checkpoint state dictionary, e.g. as returned by
        `load_checkpoint_state`.

    mode : str, optional
        Which mode's checkpoint to extract, for a `StandardGSTCheckpoint`
        state. If `state` has exactly one mode, this can be omitted. Ignored
        (and unnecessary) for a bare `GateSetTomographyCheckpoint` state.

    Returns
    -------
    dict
        The (bare) `GateSetTomographyCheckpoint`-shaped state dictionary.

    Raises
    ------
    ValueError
        If `state` is a `StandardGSTCheckpoint` with more than one mode and
        `mode` was not specified or doesn't name a GST (as opposed to
        model-test) mode.
    """
    if 'children' not in state:
        # Already a bare GateSetTomographyCheckpoint state.
        return state

    modes = state.get('modes', [])
    child_types = state.get('child_types', {})
    gst_modes = [m for m in modes if child_types.get(m, None) == 'gatesettomography']

    if mode is None:
        if len(gst_modes) == 1:
            mode = gst_modes[0]
        elif len(gst_modes) == 0:
            raise ValueError(
                "This StandardGSTCheckpoint state has no GateSetTomography "
                "(as opposed to model-test) modes to report on.")
        else:
            raise ValueError(
                "This StandardGSTCheckpoint state has multiple GateSetTomography "
                "modes (%s); specify which one via the `mode` argument." % gst_modes)
    elif child_types.get(mode, None) != 'gatesettomography':
        raise ValueError(
            "Mode '%s' is not a GateSetTomography mode in this checkpoint "
            "(it has no iterative fit history to report on)." % mode)

    return state['children'][mode]


def goodness_of_fit_history(state, mode=None):
    """
    Extract the per-iteration goodness-of-fit history from a checkpoint state.

    Parameters
    ----------
    state : dict
        A raw checkpoint state dictionary, e.g. as returned by
        `load_checkpoint_state`, or `GateSetTomographyCheckpoint._to_nice_serialization()`.
        May also be a `StandardGSTCheckpoint`-shaped state (see
        `unwrap_gst_checkpoint_state`), in which case `mode` selects which
        mode's history to report on.

    mode : str, optional
        Passed through to `unwrap_gst_checkpoint_state`.

    Returns
    -------
    list of dict
        One entry per completed iteration (in order), each a dictionary with
        keys: 'iteration' (0-based index), 'chi2k_distributed_qty',
        'n_data_params', 'n_model_params', 'degrees_of_freedom', 'pvalue',
        and 'objfn_description'. Any statistic that could not be computed
        (either because the underlying value wasn't available at fit time -
        see `_compute_iteration_gof_summary` - or because this checkpoint
        predates the introduction of this field) is reported as `None` rather
        than raising.

        Returns an empty list if `state` has no 'per_iter_gof' entries (e.g.
        it was written by a version of pyGSTi that predates this feature).
    """
    state = unwrap_gst_checkpoint_state(state, mode)
    raw_entries = state.get('per_iter_gof', []) or []
    history = []
    for i, entry in enumerate(raw_entries):
        entry = entry or {}
        chi2k_qty = entry.get('chi2k_distributed_qty', None)
        n_data_params = entry.get('n_data_params', None)
        n_model_params = entry.get('n_model_params', None)

        dof = None
        pvalue = None
        if n_data_params is not None and n_model_params is not None:
            dof = n_data_params - n_model_params
        if chi2k_qty is not None and dof is not None and dof > 0:
            try:
                pvalue = 1.0 - _stats.chi2.cdf(chi2k_qty, dof)
            except Exception:
                pvalue = None

        history.append({
            'iteration': i,
            'chi2k_distributed_qty': chi2k_qty,
            'n_data_params': n_data_params,
            'n_model_params': n_model_params,
            'degrees_of_freedom': dof,
            'pvalue': pvalue,
            'objfn_description': entry.get('objfn_description', None),
        })
    return history


def latest_model_from_state(state, mode=None):
    """
    Deserialize *only* the most-recently-completed model from a checkpoint
    state, without paying the cost of deserializing every earlier iteration's
    model in `mdl_list`.

    Parameters
    ----------
    state : dict
        A raw checkpoint state dictionary, e.g. as returned by
        `load_checkpoint_state`. May also be a `StandardGSTCheckpoint`-shaped
        state (see `unwrap_gst_checkpoint_state`), in which case `mode`
        selects which mode's model to report on.

    mode : str, optional
        Passed through to `unwrap_gst_checkpoint_state`.

    Returns
    -------
    Model or None
        The last model in `state['mdl_list']`, or `None` if `mdl_list` is
        empty (e.g. a checkpoint written before any iteration completed).
    """
    # Local import to avoid a heavyweight/circular import at module load time.
    from pygsti.models import Model as _Model

    state = unwrap_gst_checkpoint_state(state, mode)
    mdl_list = state.get('mdl_list', [])
    if not mdl_list:
        return None
    return _Model.from_nice_serialization(mdl_list[-1])


def _iter_dense_operations(model):
    """
    Yield (label, dense_superoperator_matrix) pairs for the operations of
    `model`, handling both `ExplicitOpModel`-style (`model.operations`) and
    implicit-model-style (`model.operation_blks['gates']`) layouts.
    """
    op_dict = None
    if hasattr(model, 'operations') and len(model.operations) > 0:
        op_dict = model.operations
    elif hasattr(model, 'operation_blks') and 'gates' in model.operation_blks:
        op_dict = model.operation_blks['gates']

    if op_dict is None:
        return

    for lbl, op in op_dict.items():
        try:
            dense = op.to_dense("HilbertSchmidt")
        except TypeError:
            # Some operator classes' to_dense() doesn't accept a basis argument.
            dense = op.to_dense()
        yield lbl, dense


def gate_eigenvalues(model, op_labels=None):
    """
    Compute the eigenvalue spectrum of each requested operation's dense
    superoperator matrix.

    Gate eigenvalues are gauge-invariant: a gauge transformation acts on a
    gate's dense superoperator matrix G via a similarity transform
    (G -> M G M^{-1}), which preserves eigenvalues. This makes eigenvalue
    spectra safe to report even for a model that has not (yet) been gauge
    optimized.

    Parameters
    ----------
    model : Model
        The model (typically the most-recently-completed iteration's model,
        as returned by `latest_model_from_state`) whose operations' eigenvalues
        should be computed.

    op_labels : iterable of Label, optional
        If given, only compute eigenvalues for these operation labels.
        If None (the default), compute eigenvalues for every operation in
        `model`.

    Returns
    -------
    dict
        Maps operation label -> 1D `numpy.ndarray` of (generally complex)
        eigenvalues of that operation's dense superoperator matrix. Labels
        whose eigenvalues could not be computed are omitted (rather than
        raising), so that a single malformed/unusual operator doesn't prevent
        reporting on the rest of the model.
    """
    wanted = set(op_labels) if op_labels is not None else None
    evals = {}
    for lbl, dense in _iter_dense_operations(model):
        if wanted is not None and lbl not in wanted:
            continue
        try:
            evals[lbl] = _mt.eigenvalues(_np.asarray(dense))
        except Exception:
            continue
    return evals


def extract_live_metrics(state, mode=None, op_labels=None):
    """
    Convenience "do everything" extraction: given a raw checkpoint state,
    return both the full goodness-of-fit history and the gate eigenvalues of
    the latest completed model.

    Parameters
    ----------
    state : dict
        A raw checkpoint state dictionary, e.g. as returned by
        `load_checkpoint_state`. May also be a `StandardGSTCheckpoint`-shaped
        state (see `unwrap_gst_checkpoint_state`), in which case `mode`
        selects which mode's data to report on.

    mode : str, optional
        Passed through to `unwrap_gst_checkpoint_state`.

    op_labels : iterable of Label, optional
        Passed through to `gate_eigenvalues`.

    Returns
    -------
    dict
        A dictionary with keys:

        - 'last_completed_iter' : int, the 0-based index of the most recently
          completed iteration (matches `state['last_completed_iter']`).
        - 'gof_history' : list of dict, see `goodness_of_fit_history`.
        - 'eigenvalues' : dict, see `gate_eigenvalues`, computed for the
          latest completed model only (empty dict if no model is available).
    """
    state = unwrap_gst_checkpoint_state(state, mode)
    mdl = latest_model_from_state(state)
    evals = gate_eigenvalues(mdl, op_labels) if mdl is not None else {}
    return {
        'last_completed_iter': state.get('last_completed_iter', -1),
        'gof_history': goodness_of_fit_history(state),
        'eigenvalues': evals,
    }

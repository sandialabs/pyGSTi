#***************************************************************************************************
# Copyright 2015, 2019, 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

"""
Gauge optimization of crosstalk-free models over the per-qubit product gauge group.

A gauge transformation ``S = (x)_q g_q``, with each ``g_q`` a trace-preserving single-qubit
gauge, maps a crosstalk-free model to another crosstalk-free model:

- a gate ``G`` on qubits ``Q`` becomes ``S_Q^-1 G S_Q`` with ``S_Q = (x)_{q in Q} g_q``;
- qubit ``q``'s prepared state ``rho_q`` becomes ``g_q^-1 rho_q``;
- qubit ``q``'s POVM effects ``e`` (as row vectors) become ``e g_q``;
- each element ``E_k`` of a single-qubit instrument on qubit ``q`` becomes ``g_q^-1 E_k g_q``.

(pyGSTi's convention: ``G -> S^-1 G S``.)  So the distance of a gauge-transformed model to a
target is a sum over members, each depending only on the gauges of its own qubits.  That is the
same sparse structure as the fit itself, and :func:`gauge_optimize` minimizes it with the same
Levenberg-Marquardt routine, never forming anything larger than the widest gate's superoperator.

Each qubit must be prepared in ``|0>`` followed by (at most) one error map, and measured after
(at most) one error map; that is how the prepared state and the effects are read off below.
"""
from collections import OrderedDict as _OrderedDict
from dataclasses import dataclass
from typing import Dict

import numpy as _np
import scipy.sparse as _sps

from pygsti.baseobjs import QubitSpace as _QubitSpace
from pygsti.modelmembers import instruments as _inst
from pygsti.modelmembers import operations as _op

from .fit import levenberg_marquardt
from .lanemodels import LaneModels, _decompose_prep, member_objects

_ZERO = _np.array([1, 0, 0, 1]) / _np.sqrt(2)   # |0><0| in the normalized Pauli-product basis
_ONE = _np.array([1, 0, 0, -1]) / _np.sqrt(2)   # |1><1|
_NP = 12                                        # parameters of a single-qubit TP gauge


def tp_gauge(theta: _np.ndarray) -> _np.ndarray:
    """The single-qubit TP gauge matrix (Pauli-product basis) for 12 parameters; zero is the identity."""
    S = _np.identity(4)
    S[1:, :] += _np.reshape(theta, (3, 4))
    return S


def _unit(i):
    """The derivative of :func:`tp_gauge` with respect to its `i`-th parameter."""
    E = _np.zeros((4, 4))
    E[1 + i // 4, i % 4] = 1.0
    return E


def _kron(mats):
    out = mats[0]
    for m in mats[1:]:
        out = _np.kron(out, m)
    return out


@dataclass
class _Item:
    kind: str           # 'gate', 'prep' or 'povm'
    key: tuple
    qubits: tuple
    estimate: _np.ndarray
    target: _np.ndarray
    weight: float       # multiplies the item's squared distance, so its residuals carry sqrt(weight)


def _check_spam(lane_model, members, one_of_each=False):
    """
    Raise unless every qubit of `lane_model` is prepared in ``|0>`` and has at most one error map per
    state preparation and per POVM, which is the form this module and the error summary assume.
    With `one_of_each`, the model must also have a single state preparation and a single POVM.
    """
    preps, povms = lane_model.prep_blks['layers'], lane_model.povm_blks['layers']
    for lbl, prep in preps.items():
        if any(_decompose_prep(prep, lbl)[0]):
            raise NotImplementedError(f"State preparation {lbl} is not |0> on every qubit; only |0> is supported.")
    for key in members:
        if key[0] in ('prep', 'povm') and key[3] > 0:
            raise NotImplementedError(f"Qubit {key[2]} has more than one error map in {key[1]}; "
                                      "only one per qubit is supported.")
    if one_of_each and (len(preps) > 1 or len(povms) > 1):
        raise NotImplementedError("Models with several state preparations or POVMs are not supported.")


def _check_transformable(lane_models):
    """Raise unless every member built so far can be set from a dense matrix (and see :func:`_check_spam`)."""
    for _, (lane_model, _) in lane_models.items():
        members = member_objects(lane_model)
        _check_spam(lane_model, members)
        for key, (_, member) in members.items():
            parts = [op.embedded_op for op in member.values()] if key[0] == 'instrument' else [member]
            for part in parts:
                if not isinstance(part, _op.DenseOperator):
                    raise NotImplementedError(
                        f"Member {key} is a {type(part).__name__}, which cannot be set from a dense matrix.  Gauge "
                        "transformations need densely parameterized members (e.g. the general crosstalk-free "
                        "model's); other parameterizations are not closed under them.")


def _effective(kind, member):
    """
    A member's identifiable dense quantity: a gate's matrix, a prepared state, POVM effects
    (rows), or an instrument's elements (stacked, in the instrument's outcome order).
    """
    if kind == 'instrument':
        return _np.array([op.embedded_op.to_dense() for op in member.values()])
    E = member.to_dense()
    if kind == 'gate':
        return E
    if kind == 'prep':
        return E @ _ZERO
    return _np.array([E.T @ _ZERO, E.T @ _ONE])


def _collect_items(lane_models: LaneModels, target: LaneModels, gate_weight: float, spam_weight: float,
                   instrument_weight: float):
    items = _OrderedDict()
    for support, (lane_model, _) in lane_models.items():
        target_model = target[support][0]
        target_members = member_objects(target_model)
        _check_spam(target_model, target_members)
        for key, (qubits, member) in member_objects(lane_model).items():
            if key in items:
                continue
            kind = key[0]
            if qubits is None:
                raise NotImplementedError(f"Gate {key[1]} is shared by every placement, so no product gauge "
                                          "keeps it a single gate; use independent_gates=True.")
            if key not in target_members:
                raise ValueError(f"The target has no member {key} on {support}.")
            weight = {'gate': gate_weight, 'instrument': instrument_weight}.get(kind, spam_weight)
            items[key] = _Item(kind, key, tuple(qubits), _effective(kind, member),
                               _effective(kind, target_members[key][1]), weight)
    return list(items.values())


def _residual_and_jacobian(item: _Item, gauges: Dict, jac: bool):
    """The item's weighted residual vector and, optionally, its derivative w.r.t. its qubits' gauge parameters."""
    gs = [gauges[q] for q in item.qubits]
    w = _np.sqrt(item.weight)
    if item.kind == 'gate':
        S = _kron(gs)
        Si = _np.linalg.inv(S)
        M = Si @ item.estimate @ S
        r = (M - item.target).ravel()
        if not jac:
            return w * r, None
        cols = []
        for j in range(len(gs)):
            for i in range(_NP):
                dS = _kron([_unit(i) if k == j else gs[k] for k in range(len(gs))])
                cols.append((Si @ (item.estimate @ dS - dS @ M)).ravel())   # d(S^-1 G S)
        return w * r, w * _np.array(cols).T
    g = gs[0]
    if item.kind == 'instrument':                                               # E_k -> g^-1 E_k g
        gi = _np.linalg.inv(g)
        Ms = [gi @ E @ g for E in item.estimate]
        r = _np.concatenate([(M - T).ravel() for M, T in zip(Ms, item.target)])
        if not jac:
            return w * r, None
        J = _np.array([_np.concatenate([(gi @ (E @ _unit(i) - _unit(i) @ M)).ravel()
                                        for E, M in zip(item.estimate, Ms)]) for i in range(_NP)]).T
        return w * r, w * J
    if item.kind == 'prep':
        gi = _np.linalg.inv(g)
        rho = gi @ item.estimate
        r = rho - item.target
        if not jac:
            return w * r, None
        J = _np.array([-(gi @ _unit(i) @ rho) for i in range(_NP)]).T            # d(g^-1 rho)
        return w * r, w * J
    r = (item.estimate @ g - item.target).ravel()                               # effects e -> e g
    if not jac:
        return w * r, None
    J = _np.array([(item.estimate @ _unit(i)).ravel() for i in range(_NP)]).T
    return w * r, w * J


class GaugeObjective(object):
    """
    The weighted sum of squared Frobenius distances of the gauge-transformed model's members to
    their targets, as a function of every qubit's 12 TP-gauge parameters; see :func:`gauge_optimize`.

    Exposes the interface :func:`pygsti.extras.crosstalkfreegst.fit.levenberg_marquardt` minimizes.
    """

    def __init__(self, items, qubit_labels):
        self.items = items
        self.qubit_labels = tuple(qubit_labels)
        self.qindex = {q: i for i, q in enumerate(self.qubit_labels)}
        self.num_params = _NP * len(self.qubit_labels)
        self._param_indices = [_np.concatenate([_np.arange(_NP * self.qindex[q], _NP * (self.qindex[q] + 1))
                                                for q in item.qubits]) for item in items]

    def gauges(self, x) -> Dict:
        return {q: tp_gauge(x[_NP * i:_NP * (i + 1)]) for q, i in self.qindex.items()}

    def fn(self, x) -> float:
        gauges = self.gauges(x)
        return sum(float(_np.dot(r, r)) for r in (_residual_and_jacobian(it, gauges, False)[0] for it in self.items))

    def fn_grad_gn(self, x):
        gauges = self.gauges(x)
        f = 0.0
        jtr = _np.zeros(self.num_params)
        rows, cols, vals = [], [], []
        for item, idx in zip(self.items, self._param_indices):
            r, J = _residual_and_jacobian(item, gauges, True)
            f += float(_np.dot(r, r))
            jtr[idx] += J.T @ r
            rr, cc = _np.meshgrid(idx, idx, indexing='ij')
            rows.append(rr.ravel())
            cols.append(cc.ravel())
            vals.append((J.T @ J).ravel())
        jtj = _sps.coo_matrix((_np.concatenate(vals), (_np.concatenate(rows), _np.concatenate(cols))),
                              shape=(self.num_params, self.num_params)).tocsc()
        return f, jtr, jtj


@dataclass
class GaugeOptResult:
    """
    The output of :func:`gauge_optimize`.

    Attributes
    ----------
    gauges : dict
        Qubit label -> the 4x4 TP gauge matrix ``g_q`` (Pauli-product basis) that was applied.

    x : numpy.ndarray
        The gauge-optimized global parameter vector of the lane models.

    fn_before, fn_after : float
        Weighted sum of squared Frobenius distances to the target before and after.
    """
    gauges: Dict
    x: _np.ndarray
    fn_before: float
    fn_after: float


def apply_product_gauge(lane_models: LaneModels, gauges: Dict) -> _np.ndarray:
    """
    Transform every member of `lane_models` by the product gauge ``(x)_q gauges[q]``, in place.

    Requires densely parameterized members (e.g. those of
    :func:`pygsti.extras.crosstalkfreegst.models.create_general_crosstalk_free_model`); other parameterizations are not
    closed under general gauge transformations.  That is checked before anything is changed.

    Parameters
    ----------
    lane_models : LaneModels

    gauges : dict
        Qubit label -> 4x4 gauge matrix in the Pauli-product basis, for every qubit of the lane
        models built so far.  For a trace-preserving model the matrices must be trace preserving
        (first row ``1, 0, 0, 0``), as :func:`tp_gauge`'s are.

    Returns
    -------
    numpy.ndarray
        The new global parameter vector (also set on `lane_models`).
    """
    _check_transformable(lane_models)
    missing = sorted({q for support, _ in lane_models.items() for q in support} - set(gauges))
    if missing:
        raise ValueError(f"No gauge was given for qubits {missing}.")
    x = lane_models.current_vector()
    done = set()
    for support, (lane_model, _) in lane_models.items():
        for key, (qubits, member) in member_objects(lane_model).items():
            if key in done:
                continue
            if key[0] == 'instrument':
                x[lane_models.keys[key]] = _transformed_instrument_params(member, gauges[qubits[0]], qubits[0],
                                                                          lane_model.evotype)
                done.add(key)
                continue
            E = member.to_dense()
            if key[0] == 'gate':
                S = _kron([gauges[q] for q in qubits])
                member.set_dense(_np.linalg.inv(S) @ E @ S)
            elif key[0] == 'prep':    # rho = E|0>  ->  g^-1 E |0>
                member.set_dense(_np.linalg.inv(gauges[qubits[0]]) @ E)
            else:                     # effects e = E^T z  ->  (E g)^T z
                member.set_dense(E @ gauges[qubits[0]])
            x[lane_models.keys[key]] = member.to_vector()
            done.add(key)
    lane_models.set_vector(x)
    return x


def _transformed_instrument_params(inst, g, qubit, evotype):
    """The parameters of single-qubit instrument `inst` (embedded members) after the gauge ``E_k -> g^-1 E_k g``."""
    gi = _np.linalg.inv(g)
    elements = {k: gi @ op.embedded_op.to_dense() @ g for k, op in inst.items()}
    members = [op.embedded_op for op in inst.values()]
    if all(isinstance(m, _inst.TPInstrumentOp) for m in members):
        # A TPInstrument built from dense elements has the same parameter layout as the embedded one.
        return _inst.TPInstrument(elements, evotype=evotype, state_space=_QubitSpace((qubit,))).to_vector()
    for k, op in inst.items():  # fully parameterized elements (see _check_transformable)
        op.embedded_op.set_dense(elements[k])
    return inst.to_vector()


def gauge_optimize(lane_models: LaneModels, target: LaneModels, gate_weight: float = 1.0, spam_weight: float = 1.0,
                   instrument_weight: float = 1.0, max_iterations: int = 100, verbosity: int = 0) -> GaugeOptResult:
    """
    Gauge-optimize `lane_models` to `target` over the per-qubit product of TP gauge groups.

    Minimizes the weighted sum of squared Frobenius distances between each member of the
    gauge-transformed model and the target's: gate matrices (weight `gate_weight`), each
    qubit's prepared state and POVM effects (weight `spam_weight`), and single-qubit instrument
    elements (weight `instrument_weight`).  The weights multiply the squared distances, as
    the ``item_weights`` of pyGSTi's ``gaugeopt_to_target`` do.  It runs in two stages:
    every qubit on its own using only its single-qubit gates, instruments and SPAM (these
    decouple), then all qubits jointly with the multi-qubit gates included.  The result is
    applied to `lane_models` in place.

    The estimate's members must be densely parameterized (see :func:`apply_product_gauge`);
    that is checked before the optimization starts.

    Parameters
    ----------
    lane_models : LaneModels
        The estimate.  Every support whose members should take part must already be built
        (as it is after :func:`pygsti.extras.crosstalkfreegst.run_crosstalk_free_gst`).

    target : LaneModels
        The target, e.g. ``LaneModels(pspec, create_general_crosstalk_free_model)`` at its
        initial (ideal) parameters.  Built on demand for the estimate's supports.  It may have
        any parameterization: only its members' dense matrices are used.

    gate_weight, spam_weight, instrument_weight : float, optional

    max_iterations : int, optional
        Per stage.

    verbosity : int, optional

    Returns
    -------
    GaugeOptResult
    """
    _check_transformable(lane_models)
    items = _collect_items(lane_models, target, gate_weight, spam_weight, instrument_weight)
    if not items:
        raise ValueError("No lane models have been built, so there is nothing to gauge-optimize.")
    qubits = sorted({q for it in items for q in it.qubits}, key=list(lane_models.qubit_labels).index)
    full = GaugeObjective(items, qubits)
    fn_before = full.fn(_np.zeros(full.num_params))

    local = GaugeObjective([it for it in items if len(it.qubits) == 1], qubits)
    x = levenberg_marquardt(local, _np.zeros(full.num_params), max_iterations=max_iterations,
                            rel_ftol=1e-10, verbosity=verbosity).x
    result = levenberg_marquardt(full, x, max_iterations=max_iterations, rel_ftol=1e-10, verbosity=verbosity)
    gauges = full.gauges(result.x)
    x_model = apply_product_gauge(lane_models, gauges)
    return GaugeOptResult(gauges, x_model, fn_before, result.fn)

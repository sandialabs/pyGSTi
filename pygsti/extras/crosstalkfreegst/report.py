#***************************************************************************************************
# Copyright 2015, 2019, 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

"""
Per-qubit and per-edge error summaries of a (gauge-optimized) crosstalk-free estimate.
"""
import warnings as _warnings
from typing import Dict, List

import numpy as _np

import pygsti.tools as _tools
from pygsti.tools import sdptools as _sdptools
from pygsti.tools.exceptions import NumericalDomainWarning as _NumericalDomainWarning

from .gaugeopt import _ONE, _ZERO, _check_spam
from .lanemodels import LaneModels, member_objects


def _half_diamond_distance(a, b):
    """``0.5 * ||a - b||_diamond`` (pyGSTi's ``diamonddist`` is the full norm), or NaN without cvxpy."""
    if not _sdptools.CVXPY_ENABLED:
        return float('nan')
    return 0.5 * float(_tools.diamonddist(a, b, 'pp'))


def error_summary(lane_models: LaneModels, target: LaneModels, diamond: bool = True) -> List[Dict]:
    """
    One row per gate and per qubit's SPAM, comparing `lane_models` with `target`.

    The numbers are gauge-dependent, so gauge-optimize first (:func:`pygsti.extras.crosstalkfreegst.gauge_optimize`).

    Gate rows hold the entanglement infidelity, average gate infidelity and (if `diamond`
    and cvxpy is installed) half the diamond distance to the target gate.  SPAM rows hold, for
    each qubit, the preparation infidelity ``1 - <0|rho|0>`` and the readout error rates
    ``P(1|0)`` and ``P(0|1)`` of its POVM on ideal ``|0>``/``|1>`` inputs.  Instrument rows
    (mid-circuit measurements) hold pyGSTi's instrument infidelity and, for instruments with
    outcomes ``'0'`` and ``'1'``, the error rates ``P(1|0)`` and ``P(0|1)`` of the measurement
    on ideal inputs, ``P(k|j) = Tr E_k(|j><j|)``.

    The lane models must have one state preparation, of ``|0>`` on every qubit, and one POVM, each
    with at most one error map per qubit.

    Returns
    -------
    list of dict
        Keys: ``'kind'`` (``'gate'``, ``'instrument'`` or ``'spam'``), ``'qubits'``, ``'name'``,
        and the metrics.
    """
    gates, spam, instruments = {}, {}, {}
    done = set()
    for support, (lane_model, _) in lane_models.items():
        target_members = member_objects(target[support][0])
        members = member_objects(lane_model)
        _check_spam(lane_model, members, one_of_each=True)
        for key, (qubits, member) in members.items():
            if key in done:  # members are shared between lane models
                continue
            done.add(key)
            if key[0] == 'instrument':
                instruments[key] = _instrument_row(key[1], qubits, member, target_members[key][1])
                continue
            E, T = member.to_dense(), target_members[key][1].to_dense()
            if key[0] == 'gate':
                gates[key] = {'kind': 'gate', 'qubits': tuple(qubits), 'name': key[1].name,
                              'entanglement_infidelity': float(_tools.entanglement_infidelity(E, T, 'pp')),
                              'average_gate_infidelity': float(_tools.average_gate_infidelity(E, T, 'pp')),
                              'half_diamond_distance': _half_diamond_distance(E, T) if diamond else float('nan')}
            elif key[0] in ('prep', 'povm'):
                row = spam.setdefault(qubits[0], {'kind': 'spam', 'qubits': tuple(qubits), 'name': 'SPAM'})
                if key[0] == 'prep':
                    row['prep_infidelity'] = float(1 - (E @ _ZERO) @ _ZERO)
                else:  # effect e_i = E^T z_i, so P(i|j) = (E^T z_i) . z_j
                    row['readout_p1_given_0'] = float((E.T @ _ONE) @ _ZERO)
                    row['readout_p0_given_1'] = float((E.T @ _ZERO) @ _ONE)
    order = list(lane_models.qubit_labels).index
    rows = sorted(gates.values(), key=lambda r: (len(r['qubits']), [order(q) for q in r['qubits']], r['name']))
    insts = sorted(instruments.values(), key=lambda r: ([order(q) for q in r['qubits']], r['name']))
    return rows + insts + [spam[q] for q in sorted(spam, key=order)]


def _instrument_row(label, qubits, inst, target):
    est = {k: op.embedded_op.to_dense() for k, op in inst.items()}
    tgt = {k: op.embedded_op.to_dense() for k, op in target.items()}
    with _warnings.catch_warnings():  # instrument elements are not trace preserving, which
        _warnings.simplefilter('ignore', _NumericalDomainWarning)  # instrument_infidelity expects
        infidelity = float(_tools.instrument_infidelity(est, tgt, 'pp'))
    row = {'kind': 'instrument', 'qubits': tuple(qubits), 'name': label.name, 'instrument_infidelity': infidelity}
    if set(est) == {'0', '1'}:  # Tr(A) = sqrt(2) * A[0] in the normalized Pauli-product basis
        row['mcm_p1_given_0'] = float(_np.sqrt(2) * (est['1'] @ _ZERO)[0])
        row['mcm_p0_given_1'] = float(_np.sqrt(2) * (est['0'] @ _ONE)[0])
    return row


def format_error_summary(rows: List[Dict]) -> str:
    """
    A plain-text table of :func:`error_summary`'s rows.

    The diamond-distance column is left out if no diamond distance was computed.
    """
    gates = [r for r in rows if r['kind'] == 'gate']
    diamond = not all(_np.isnan(r['half_diamond_distance']) for r in gates)
    lines = [f"{'gate':<10}{'qubits':<10}{'ent. infid.':>13}{'avg. infid.':>13}"
             + (f"{'1/2 diamond':>13}" if diamond else '')]
    for r in gates:
        lines.append(f"{r['name']:<10}{str(r['qubits']):<10}{r['entanglement_infidelity']:>13.2e}"
                     f"{r['average_gate_infidelity']:>13.2e}"
                     + (f"{r['half_diamond_distance']:>13.2e}" if diamond else ''))
    insts = [r for r in rows if r['kind'] == 'instrument']
    if insts:
        lines += ['', f"{'instrument':<12}{'qubit':<8}{'infidelity':>13}{'P(1|0)':>13}{'P(0|1)':>13}"]
        for r in insts:
            values = [r.get(k, float('nan')) for k in ('instrument_infidelity', 'mcm_p1_given_0', 'mcm_p0_given_1')]
            lines.append(f"{r['name']:<12}{str(r['qubits'][0]):<8}" + ''.join(f"{v:>13.2e}" for v in values))
    lines += ['', f"{'qubit':<10}{'prep infid.':>13}{'P(1|0)':>13}{'P(0|1)':>13}"]
    for r in rows:
        if r['kind'] == 'spam':
            values = [r.get(k, float('nan')) for k in ('prep_infidelity', 'readout_p1_given_0', 'readout_p0_given_1')]
            lines.append(f"{str(r['qubits'][0]):<10}" + ''.join(f"{v:>13.2e}" for v in values))
    return '\n'.join(lines)

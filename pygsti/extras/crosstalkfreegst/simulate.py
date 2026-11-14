#***************************************************************************************************
# Copyright 2015, 2019, 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

"""
Simulating data from a crosstalk-free model one lane at a time.
"""
import warnings as _warnings
from collections import Counter as _Counter
from collections import defaultdict as _defaultdict
from typing import Sequence

import numpy as _np

from pygsti.data import DataSet as _DataSet
from pygsti.tools.exceptions import ProbabilityClippingWarning as _ProbabilityClippingWarning

from .lanemodels import LaneModels
from .lanes import LaneDecomposition, split_circuit_into_lanes


def _lane_probabilities(lane_models, splits):
    """
    ``(support, lane circuit) -> (outcome labels, probabilities)`` for every lane in `splits`.

    Outcome labels are the lane model's: ``(bits,)``, or ``(instrument outcomes..., bits)`` for
    lane circuits containing instruments.  Negative probabilities are set to zero and the rest
    rescaled to sum to one; that only removes rounding error unless the model is unphysical, in
    which case there is a warning, as from ``pygsti.data.simulate_data``.
    """
    by_support = _defaultdict(dict)
    for lanes in splits:
        for support, lane_circuit in lanes:
            by_support[support][lane_circuit] = None
    out = {}
    for support, lane_circuits in by_support.items():
        lane_model, _ = lane_models[support]
        for lane_circuit, p in lane_model.sim.bulk_probs(list(lane_circuits)).items():
            labels = list(p.keys())
            probs = _np.array([p[ol] for ol in labels])
            if probs.min() < -1e-10:
                _warnings.warn(f"Clipping probs < 0 to 0 (the smallest is {probs.min():.2g}, for {lane_circuit.str})",
                               _ProbabilityClippingWarning)
            probs = _np.clip(probs, 0, None)
            out[support, lane_circuit] = (labels, probs / probs.sum())
    return out


def lane_outcome_distributions(lane_models: LaneModels, splits):
    """
    The exact outcome distribution of every lane of every circuit, without sampling.

    If the model is unphysical and predicts negative probabilities, they are set to zero and
    the distribution rescaled, with a warning.

    Parameters
    ----------
    lane_models : LaneModels
        Lane models set to the parameters to simulate.

    splits : list
        One list of ``(support, lane circuit)`` per circuit, each as returned by
        :func:`~pygsti.extras.crosstalkfreegst.split_circuit_into_lanes`.  The lane circuits
        must not contain instruments.

    Returns
    -------
    list of numpy.ndarray
        One array of length ``2**k`` per lane, in the order of `splits` (flattened), indexed by
        the lane's bits read as a binary number (the lane's first qubit is the most significant
        bit).  The circuit's outcome distribution is the product of its lanes'.
    """
    probs = _lane_probabilities(lane_models, splits)
    out = []
    for lanes in splits:
        for support, lane_circuit in lanes:
            labels, p = probs[support, lane_circuit]
            if any(len(ol) != 1 for ol in labels):
                raise NotImplementedError("Lane circuits with instruments are not supported.")
            array = _np.zeros(2**len(support))
            array[[int(ol[0], 2) for ol in labels]] = p
            out.append(array)
    return out


def _instrument_qubits(circuit, instrument_names):
    return LaneDecomposition(circuit.line_labels, instrument_names=instrument_names)._instrument_qubits(circuit)


def simulate_lanewise_data(model, circuits: Sequence, num_samples: int, seed=None,
                           qubit_labels: Sequence = None, simulator='auto') -> _DataSet:
    """
    Sample data for `circuits` from a crosstalk-free `model`, without many-qubit simulation.

    Each shot's outcome on each lane is drawn independently from that lane's distribution,
    which is exact for a crosstalk-free model, and the lane outcomes are assembled into
    full outcome labels following pyGSTi's conventions: final bits ordered by each circuit's
    ``line_labels``, preceded by one outcome per instrument (mid-circuit measurement) in
    circuit order.  The result is distributed identically to
    ``pygsti.data.simulate_data(model, circuits, num_samples)``, but costs lane-sized
    simulations only.

    Parameters
    ----------
    model : LocalNoiseModel or LaneModels
        A crosstalk-free model (see :func:`pygsti.extras.crosstalkfreegst.check_factorizable`),
        or lane models set to the parameters to simulate.  Instrument names are taken from
        its processor spec's ``instrument_names``.

    circuits : sequence of Circuit

    num_samples : int
        Shots per circuit.

    seed : int or numpy.random.Generator, optional

    qubit_labels : sequence, optional
        Canonical qubit order; defaults to the processor spec's.

    simulator : str or ForwardSimulator, optional
        Forward simulator for the lane models, when `model` is a many-qubit model.

    Returns
    -------
    DataSet
    """
    rng = _np.random.default_rng(seed)
    lane_models = model if isinstance(model, LaneModels) else LaneModels.from_model(model, simulator)
    qubit_labels = tuple(lane_models.qubit_labels if qubit_labels is None else qubit_labels)
    instrument_names = lane_models.processor_spec.instrument_names

    circuits = list({c: None for c in circuits})
    splits = [split_circuit_into_lanes(c, qubit_labels) for c in circuits]
    probs = _lane_probabilities(lane_models, splits)

    ds = _DataSet(collision_action='aggregate')
    for circuit, lanes in zip(circuits, splits):
        lpos = {q: i for i, q in enumerate(circuit.line_labels)}
        inst_qubits = _instrument_qubits(circuit, instrument_names)
        bits = _np.zeros((num_samples, circuit.num_lines), dtype=_np.int8)
        inst_outcomes = _np.empty((num_samples, len(inst_qubits)), dtype=object)
        for support, lane_circuit in lanes:
            labels, p = probs[support, lane_circuit]
            draws = rng.choice(len(labels), size=num_samples, p=p)
            lane_bits = _np.array([[int(b) for b in ol[-1]] for ol in labels], dtype=_np.int8)
            bits[:, [lpos[q] for q in support]] = lane_bits[draws]
            slots = [j for j, q in enumerate(inst_qubits) if q in support]
            if len(labels[0]) - 1 != len(slots):
                raise ValueError(f"Lane {support} of {circuit.str} has {len(labels[0]) - 1} mid-circuit outcomes "
                                 f"but {len(slots)} instruments named in the processor spec's instrument_names "
                                 f"{tuple(instrument_names)}; declare every instrument there.")
            if slots:
                lane_inst = _np.array([ol[:-1] for ol in labels], dtype=object)
                inst_outcomes[:, slots] = lane_inst[draws]
        if inst_qubits:
            tally = _Counter(tuple(io) + (''.join(map(str, b)),) for io, b in zip(inst_outcomes, bits))
            ds.add_count_dict(circuit, dict(tally), record_zero_counts=False)
        else:
            rows, counts = _np.unique(bits, axis=0, return_counts=True)
            ds.add_count_dict(circuit, {(''.join(map(str, row)),): int(n) for row, n in zip(rows, counts)},
                              record_zero_counts=False)
    ds.done_adding_data()
    return ds


def simulate_lanewise_marginals(model, circuits: Sequence, num_samples: int, seed=None,
                                qubit_labels: Sequence = None, simulator='auto') -> LaneDecomposition:
    """
    Like :func:`simulate_lanewise_data`, but return only each circuit's lane-marginal counts.

    Each lane's counts are drawn from a multinomial over that lane's outcomes.  For a
    crosstalk-free model the lanes of a circuit are independent, so these have exactly the
    distribution of the marginals of :func:`simulate_lanewise_data`'s joint counts -- without
    building joint outcome strings, which at tens of qubits are nearly all distinct.

    Returns
    -------
    LaneDecomposition
        Ready to pass to :func:`pygsti.extras.crosstalkfreegst.run_crosstalk_free_gst` in place
        of a dataset.
    """
    rng = _np.random.default_rng(seed)
    lane_models = model if isinstance(model, LaneModels) else LaneModels.from_model(model, simulator)
    qubit_labels = tuple(lane_models.qubit_labels if qubit_labels is None else qubit_labels)
    circuits = list({c: None for c in circuits})
    splits = [split_circuit_into_lanes(c, qubit_labels) for c in circuits]
    probs = _lane_probabilities(lane_models, splits)

    decomposition = LaneDecomposition(qubit_labels, instrument_names=lane_models.processor_spec.instrument_names)
    for circuit, lanes in zip(circuits, splits):
        lane_counts = {}
        for support, lane_circuit in lanes:
            labels, p = probs[support, lane_circuit]
            counts = rng.multinomial(num_samples, p)
            if all(len(ol) == 1 for ol in labels):  # no instruments: index by the bits as a binary number
                array = _np.zeros(2**len(support))
                array[[int(ol[0], 2) for ol in labels]] = counts
                lane_counts[support] = array
            else:
                lane_counts[support] = {ol: float(n) for ol, n in zip(labels, counts) if n > 0}
        decomposition.add_lane_counts(circuit, lane_counts)
    return decomposition

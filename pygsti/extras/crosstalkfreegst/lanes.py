#***************************************************************************************************
# Copyright 2015, 2019, 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

"""
Splitting tensor-product circuits into lanes, and marginalizing their data onto those lanes.

A *lane* of a circuit is a connected component of the graph whose vertices are qubits and
whose edges join any two qubits acted on by a common gate.  Under a crosstalk-free model with
product state preparation and product measurement, the outcome distribution of a circuit is
the product of its lanes' outcome distributions, so

    sum_x N(x) log p(x) = sum_lanes sum_{x_lane} N_lane(x_lane) log p_lane(x_lane)

where ``N_lane`` are the counts marginalized onto the lane's qubits.  That identity is exact,
and it is what lets a many-qubit crosstalk-free model be fit using only lane-sized
simulations.  Lane circuits on the same qubits that are identical -- which happens a lot in
simultaneous GST, where every qubit or qubit pairs runs a circuit from a 2Q or 1Q GST
design -- have their counts pooled.
"""
from collections import defaultdict as _defaultdict
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as _np

from pygsti.baseobjs.label import Label as _Label
from pygsti.circuits.circuit import Circuit as _Circuit
from pygsti.data import DataSet as _DataSet

Support = Tuple  # a tuple of qubit labels, ordered as in the model's qubit order


def split_circuit_into_lanes(circuit: _Circuit, qubit_labels: Sequence) -> List[Tuple[Support, _Circuit]]:
    """
    Split `circuit` into lane circuits, one per lane of `qubit_labels`.

    Each lane circuit keeps only the gates acting within its lane, and is labelled by the
    lane's qubits in `qubit_labels` order -- a canonical order, so that identical lanes of
    circuits whose line labels are ordered differently compare equal.  Every line of
    `circuit` belongs to exactly one lane; lines without gates become singleton lanes with
    an empty circuit, since they are still prepared and measured.

    A layer in which a lane has no gates is dropped from that lane's circuit: with implicit
    idles treated as perfect identities (``implicit_idle_mode='none'``), which is the only
    mode supported here, it acts trivially on the lane.  A completely *empty* layer is
    rejected, because a model may map it to a noisy global idle that is not a perfect identity.
    Simultaneous GST designs make every idle explicit, so neither case occurs for them.

    Returns
    -------
    list of (support, Circuit)
        Ordered by the position of each support's first qubit in `qubit_labels`.
    """
    qpos = {q: i for i, q in enumerate(qubit_labels)}
    lines = circuit.line_labels
    if any(q not in qpos for q in lines):
        raise ValueError(f"Circuit line labels {lines} are not all among {tuple(qubit_labels)}")
    if circuit._subcircuits_to_expand():
        circuit = circuit.expand_subcircuits()

    # Union-find over the circuit's lines; every gate merges the qubits it acts on.
    parent = {q: q for q in lines}

    def find(q):
        while parent[q] != q:
            parent[q] = parent[parent[q]]
            q = parent[q]
        return q

    layers = []
    for layer in circuit.layertup:
        comps = layer.components
        if len(comps) == 0:
            raise ValueError(f"Circuit {circuit.str} contains an empty layer.  Empty layers may "
                             "denote a noisy global idle, which does not factorize into lanes; "
                             "make idles explicit.")
        for comp in comps:
            sslbls = comp.sslbls
            if sslbls is None:
                raise ValueError(f"Gate {comp} in circuit {circuit.str} has no target qubits; "
                                 "all gates must name the qubits they act on.")
            for q in sslbls:
                if q not in parent:
                    raise ValueError(f"Gate {comp} acts on qubit {q}, which is not a line of {circuit.str}")
            r0 = find(sslbls[0])
            for q in sslbls[1:]:
                parent[find(q)] = r0
        layers.append(comps)

    groups = _defaultdict(list)
    for q in lines:
        groups[find(q)].append(q)
    supports = sorted((tuple(sorted(g, key=qpos.__getitem__)) for g in groups.values()),
                      key=lambda s: qpos[s[0]])

    # One pass over the gates, each going to the lane of its first qubit (a gate lies within one lane).
    lane_of = {find(support[0]): i for i, support in enumerate(supports)}
    lane_layers = [[] for _ in supports]
    for comps in layers:
        by_lane = _defaultdict(list)
        for comp in comps:
            by_lane[lane_of[find(comp.sslbls[0])]].append(comp)
        for i, lane_comps in by_lane.items():
            lane_layers[i].append(lane_comps[0] if len(lane_comps) == 1 else _Label(lane_comps))
    return [(support, _Circuit(lane_layers[i], line_labels=support, editable=False))
            for i, support in enumerate(supports)]


def _outcome_bits(outcome_labels, num_lines: int) -> _np.ndarray:
    """A ``(len(outcome_labels), num_lines)`` array of the bits of bit-string outcome labels."""
    strings = [ol[0] if (isinstance(ol, tuple) and len(ol) == 1) else ol for ol in outcome_labels]
    if not all(isinstance(s, str) and len(s) == num_lines for s in strings):
        raise ValueError(f"Outcome labels must be {num_lines}-bit strings, one bit per circuit line; "
                         "only single, terminal, computational-basis measurements are supported.")
    bits = _np.frombuffer(''.join(strings).encode('ascii'), dtype=_np.uint8).reshape(len(strings), num_lines) - 48
    if _np.any(bits > 1):
        raise ValueError("Outcome labels must consist of the characters '0' and '1'.")
    return bits


class LaneDecomposition(object):
    """
    The lanes of a set of circuits, with each circuit's counts marginalized onto its lanes.

    Marginal counts are computed once per circuit; :meth:`lane_datasets` then pools them
    over any subset of the circuits, which is how the per-germ-power circuit lists of a GST
    design get their own (exactly marginalized) lane data.

    Only marginal counts are kept, so many-qubit data can be streamed in with :meth:`add`
    (or :meth:`add_lane_counts`) without ever holding a joint dataset, whose distinct outcome
    strings can number in the millions at tens of qubits.  A circuit's counts are added once:
    adding the same circuit again is an error, so total its counts first.  The counts of a lane
    of ``k`` qubits (without instruments) are held as an array of length ``2**k``, so lanes are
    expected to be at most a few qubits wide.

    Circuits may contain single-qubit instruments (mid-circuit measurements).  pyGSTi labels
    their outcomes ``(i_1, ..., i_m, bits)``: one outcome per instrument, in circuit order
    (layer by layer, and within a layer in the order of the circuit's stored layer label),
    then the final bit string.  Each instrument outcome belongs to the lane of the qubit it
    measured, so a lane's marginal outcome is ``(its instruments' outcomes..., its bits)`` --
    exactly how the lane model labels the lane circuit's outcomes.

    Parameters
    ----------
    qubit_labels : sequence
        All the qubits, in a canonical order that lane supports and lane outcome bits follow.

    dataset : DataSet, optional
        Counts for (at least) `circuits`.  Final bit strings refer to each circuit's lines in
        its ``line_labels`` order, which is pyGSTi's convention.

    circuits : sequence of Circuit, optional
        The circuits of `dataset` to add (default: all of them).

    instrument_names : sequence of str, optional
        Names of the (single-qubit) instruments that circuits may contain, e.g. ``('Iz',)``;
        typically the processor spec's ``instrument_names``.
    """

    def __init__(self, qubit_labels: Sequence, dataset: _DataSet = None, circuits: Sequence[_Circuit] = None,
                 instrument_names: Sequence[str] = ()):
        self.qubit_labels = tuple(qubit_labels)
        self.instrument_names = frozenset(instrument_names)
        #: circuit -> list of (support, lane circuit, marginal counts).  Marginal counts are an
        #: array indexed by the lane's bits read as a binary number, or -- for lanes containing
        #: instruments -- a dict of lane outcome label -> count.
        self.marginals: Dict[_Circuit, List[Tuple[Support, _Circuit, object]]] = {}
        if dataset is not None:
            for circuit in (dataset.keys() if circuits is None else circuits):
                if circuit not in self.marginals:
                    self.add(circuit, dataset[circuit].counts)

    def _instrument_qubits(self, circuit: _Circuit) -> List:
        """The qubit of each instrument in `circuit`, in pyGSTi's outcome-label order."""
        if not self.instrument_names:
            return []
        if circuit._subcircuits_to_expand():
            circuit = circuit.expand_subcircuits()
        return [comp.sslbls[0] for layer in circuit.layertup for comp in layer.components
                if comp.name in self.instrument_names]

    def _check_not_added(self, circuit: _Circuit) -> None:
        if circuit in self.marginals:
            raise ValueError(f"Counts for circuit {circuit.str} have already been added.  A circuit's counts "
                             "are added once; total the counts of repeated runs before adding them.")

    def add(self, circuit: _Circuit, counts: Mapping) -> None:
        """
        Add `circuit`'s counts, a mapping from outcome labels to counts.  Outcome labels are
        bit strings over the circuit's lines (in ``line_labels`` order), preceded by one
        outcome per instrument if the circuit contains instruments.
        """
        self._check_not_added(circuit)
        lanes = split_circuit_into_lanes(circuit, self.qubit_labels)
        inst_qubits = self._instrument_qubits(circuit)
        labels = list(counts.keys())
        vals = _np.fromiter(counts.values(), dtype=float, count=len(counts))
        if inst_qubits:
            for ol in labels:
                if not isinstance(ol, tuple) or len(ol) != len(inst_qubits) + 1:
                    raise ValueError(f"Outcome {ol!r} of {circuit.str} should have one entry for each of its "
                                     f"{len(inst_qubits)} instruments, then the final bit string.")
            finals = [ol[-1] for ol in labels]
        else:
            finals = labels
        circuit_bits = _outcome_bits(finals, circuit.num_lines).astype(_np.int64)
        lpos = {q: i for i, q in enumerate(circuit.line_labels)}
        entries = []
        for support, lane_circuit in lanes:
            k = len(support)
            positions = [lpos[q] for q in support]
            inst_slots = [j for j, q in enumerate(inst_qubits) if q in support]
            if not inst_slots:
                codes = circuit_bits[:, positions] @ (1 << _np.arange(k - 1, -1, -1))
                entries.append((support, lane_circuit, _np.bincount(codes, weights=vals, minlength=2**k)))
                continue
            marginal = _defaultdict(float)
            for ol, bits, n in zip(labels, circuit_bits, vals):
                marginal[tuple(ol[j] for j in inst_slots) + (''.join(str(bits[p]) for p in positions),)] += n
            entries.append((support, lane_circuit, dict(marginal)))
        self.marginals[circuit] = entries

    def add_lane_counts(self, circuit: _Circuit, lane_counts: Mapping[Support, object]) -> None:
        """
        Add `circuit`'s counts already marginalized onto its lanes: `lane_counts` maps each
        support of ``split_circuit_into_lanes(circuit, qubit_labels)`` to either a length-``2**k``
        array, indexed by the lane's bits (in support order) read as a binary number, or -- as
        lanes containing instruments require -- a dict of lane outcome label -> count.
        """
        self._check_not_added(circuit)
        lanes = split_circuit_into_lanes(circuit, self.qubit_labels)
        if set(lane_counts) != {s for s, _ in lanes}:
            raise ValueError(f"Lane counts given for {sorted(lane_counts)}, but the lanes of {circuit.str} "
                             f"are {[s for s, _ in lanes]}")
        self.marginals[circuit] = [(s, lc, dict(lane_counts[s]) if isinstance(lane_counts[s], Mapping)
                                    else _np.asarray(lane_counts[s], dtype=float)) for s, lc in lanes]

    def lane_datasets(self, circuits: Sequence[_Circuit] = None) -> Dict[Support, Tuple[_DataSet, List[_Circuit]]]:
        """
        Pool the marginal counts of `circuits` (default: all) into one dataset per support.

        Returns
        -------
        dict
            Maps each support to ``(dataset, lane_circuits)``, where `dataset` holds the pooled
            counts of the distinct `lane_circuits` on that support.  Outcome labels are bit
            strings over the support's qubits, preceded by instrument outcomes where present.
        """
        if circuits is None:
            circuits = self.marginals.keys()
        pooled = _defaultdict(dict)
        for circuit in _unique(circuits):
            for support, lane_circuit, counts in self.marginals[circuit]:
                lane_counts = pooled[support]
                if lane_circuit not in lane_counts:
                    lane_counts[lane_circuit] = counts.copy()
                elif isinstance(counts, dict):
                    total = lane_counts[lane_circuit]
                    for ol, n in counts.items():
                        total[ol] = total.get(ol, 0.0) + n
                else:
                    lane_counts[lane_circuit] = lane_counts[lane_circuit] + counts

        result = {}
        for support, lane_counts in pooled.items():
            k = len(support)
            bit_labels = [(format(i, f'0{k}b'),) for i in range(2**k)]
            ds = _DataSet()
            for lane_circuit, counts in lane_counts.items():
                ds.add_count_dict(lane_circuit, counts if isinstance(counts, dict) else dict(zip(bit_labels, counts)))
            ds.done_adding_data()
            result[support] = (ds, list(lane_counts.keys()))
        return result


def _unique(items):
    seen = set()
    for x in items:
        if x not in seen:
            seen.add(x)
            yield x

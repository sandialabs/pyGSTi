import networkx as _nx

from typing import Sequence, Dict, Tuple, Optional, Set, Union
from pygsti.circuits import Circuit as Circuit
from pygsti.baseobjs.label import Label, LabelTup, LabelTupTup


def _map_label_state_space_labels(lbl, sslbl_map):
    """
    Map the state-space labels inside a Label, including LabelTupTup members.

    This is used after applying a layer_mapper value, so we do not rely on
    Circuit.map_state_space_labels_inplace to correctly descend into newly
    created compound/parallel labels.
    """

    if lbl == Label(()):
        return lbl

    if isinstance(lbl, LabelTupTup):
        return Label(tuple(
            _map_label_state_space_labels(member, sslbl_map)
            for member in lbl
        ))

    if isinstance(lbl, LabelTup):
        return lbl.map_state_space_labels(sslbl_map)

    if isinstance(lbl, tuple):
        return tuple(
            _map_label_state_space_labels(member, sslbl_map)
            for member in lbl
        )

    return lbl.map_state_space_labels(sslbl_map)


def compute_qubit_lane_mappings_for_circuit(circuit: Circuit) -> tuple[dict[int, int],
                                                                       dict[int, set[int]]]:
    """
    Partition `circuit`'s lines ("qubits") into lanes: maximal groups of lines that
    are connected to each other via multi-line gates, and so cannot be simulated
    independently of one another.  Lines that never appear together in a multi-line
    gate end up in different lanes.

    Parameters:
    ------------
    circuit: Circuit - the circuit to compute qubit to lanes mapping for

    Returns
    --------
    A tuple `(qubit_to_lane, lane_to_qubits)`:
      - `qubit_to_lane` maps each of `circuit`'s line labels to the index of the
        lane it belongs to.
      - `lane_to_qubits` maps each lane index to the set of line labels in
        that lane.
    """
    # Build a graph whose nodes are this circuit's lines, with an edge between
    # any two lines that ever appear together in the same (multi-line) gate.
    # A lane is then just a connected component of this graph -- every line
    # starts out in its own singleton component (so that lines untouched by
    # any multi-line gate still end up in their own lane).
    graph = _nx.Graph()
    graph.add_nodes_from(circuit.line_labels)
    for layer_ind in range(circuit.num_layers):
        for op in circuit.layer(layer_ind):
            qubits_used = op.qubits
            graph.add_edges_from(
                (qubits_used[0], other) for other in qubits_used[1:]
            )

    lane_to_qubits = {
        lane_num: component
        for lane_num, component in enumerate(_nx.connected_components(graph))
    }
    qubit_to_lane = {
        qubit: lane_num
        for lane_num, qubits in lane_to_qubits.items()
        for qubit in qubits
    }

    return qubit_to_lane, lane_to_qubits


def split_into_tensor_lanes(circuit: Circuit,
                            qubit_to_lanes: Optional[dict[int, int]] = None,
                            lane_to_qubits: Optional[dict[int, tuple[int, ...]]] = None,
                            cache_lanes_in_circuit: bool = False,
                            idle_gate_name: Union[str, Label] = "Gi") -> list[Circuit]:
    """
    Split a circuit into multiple subcircuits ("lanes") which do not talk across lanes.

    Note: this is distinct from the notion of a "subcircuit" elsewhere in `Circuit`
    (a repeated block of layers represented by a :class:`CircuitLabel`) -- here,
    a lane's "subcircuit" is a full-depth `Circuit` restricted to a subset of lines.

    Returns a list of `Circuit` objects, one per lane, indexed to match
    `lane_to_qubits`.  Each lane's circuit is built by deleting all of the other
    lanes' lines from a copy of `circuit` (see :meth:`Circuit.delete_lines`),
    which also validates that no gate actually straddles two different lanes.
    """

    if qubit_to_lanes is None or lane_to_qubits is None:
        qubit_to_lanes, lane_to_qubits = compute_qubit_lane_mappings_for_circuit(circuit)

    # The lanes cache is only trustworthy for *static* (read-only) circuits:
    if circuit._static and "lanes" in circuit.saved_auxinfo:
        # Check if the lane info matches and I can just return that set up.
        if len(lane_to_qubits) == len(circuit.saved_auxinfo["lanes"]):
            # We may have this already in cache.
            lane_circuits = [None] * len(lane_to_qubits)
            for i, key in lane_to_qubits.items():
                if tuple(sorted(key)) in circuit.saved_auxinfo["lanes"]:
                    lane_circuits[i] = circuit.saved_auxinfo["lanes"][tuple(sorted(key))]
                else:
                    raise ValueError(f"lbl cache miss: {key} in circuit {circuit}")
            return lane_circuits

    lane_circuits: list[Circuit] = [None] * len(lane_to_qubits)
    for lane, lane_qubits in lane_to_qubits.items():
        other_qubits = [q for q in circuit.line_labels if q not in lane_qubits]
        lane_circuit = circuit.copy(editable=True)
        # delete_lines gives us straddler-detection for free: if a gate spans
        # this lane and another one, the qubit-to-lane partition computed above
        # was wrong, and we want that surfaced as an error rather than silently
        # dropping/mangling the offending gate.
        lane_circuit.delete_lines(other_qubits, delete_straddlers=False)
        # Materialize this lane's own implicit idles into explicit
        # `idle_gate_name` gates, so each lane's layer explicitly lists every
        # line in that lane (matching the historical behavior of this function).
        lane_circuit.insert_implicit_idles_inplace(idle_gate_name=idle_gate_name)
        lane_circuit.done_editing()
        lane_circuits[lane] = lane_circuit

    if cache_lanes_in_circuit:
        circuit = circuit._cache_tensor_lanes(lane_circuits, lane_to_qubits)

    return lane_circuits


def batch_tensor(
    circuits: Sequence[Circuit],
    layer_mappers: Dict[int, Dict],
    global_line_order: Optional[Tuple[Union[int, str], ...]] = None,
    target_lines: Optional[Sequence[Tuple[Union[int, str], ...]]] = None
) -> Circuit:
    """
    `circuits`: Sequence of `Circuit` the circuits you want to tensor together.
    `layer_mappers`: dictionary of lane size to a dictionary of `Label` to `Label`.
        Need to ensure that you have something that maps from () -> Idle gate.
        Else you may get spurious "COMPOUND" gates.
    'target_lines' : Optional sequence of state space labels used by each circuit.
    `global_line_order`: The final order of the state space labels (used if one wants to make it
        not just arange(total_lines))

    Tensor together a sequence of Circuits padding smaller circuits to the length of the largest circuit
    with noisy idles.
    """
    if len(circuits) < 2:
        raise ValueError("batch_tensor requires at least two circuits to tensor together.")
    if __debug__:
        for num_lines in layer_mappers:
            if Label(()) in layer_mappers[num_lines]:
                assert layer_mappers[num_lines][Label(())] != Label(())

    if target_lines is None:
        target_lines = []
        total_lines = 0
        max_cir_len = 0
        for c in circuits:
            # We are just going to build the target_lines as arange(num_lines)
            target_lines.append(tuple(range(total_lines, total_lines + c.num_lines)))
            total_lines += c.num_lines
            max_cir_len = max(max_cir_len, len(c))
    else:
        total_lines = sum([c.num_lines for c in circuits])
        max_cir_len = max([len(c) for c in circuits])

    s: Set[int] = set()
    for c, t in zip(circuits, target_lines):
        assert not s.intersection(t)
        assert len(t) == c.num_lines
        s.update(t)

    if global_line_order is None:
        global_line_order = tuple(sorted(list(s)))

    c = circuits[0].copy(editable=True)
    c._append_idling_layers_inplace(max_cir_len - len(c))

    local_num_lines = c.num_lines
    local_labels = list(c._labels)
    sslbl_map = {
        k: v
        for k, v in zip(c.line_labels, target_lines[0])
    }

    # First remap the circuit line labels. We will overwrite the operation labels
    # immediately afterward, so we do not care how this maps the old local labels.
    c.map_state_space_labels_inplace(sslbl_map)

    # Now rebuild operation labels from the saved local labels, explicitly mapping
    # any LabelTupTup members.
    # Note: Circuit._labels stores layers as raw Python lists internally (e.g.
    # [] for an empty/idle layer, [Label(...)] for a single-gate layer).
    # Normalise them to Label objects before the mapper lookup.
    def _to_label(ell):
        if isinstance(ell, list):
            return Label(tuple(ell))
        return ell

    c._static = False
    c._labels = [
        _map_label_state_space_labels(
            layer_mappers[local_num_lines][_to_label(ell)],
            sslbl_map,
        )
        for ell in local_labels
    ]

    c.done_editing()
    for i, c2 in enumerate(circuits[1:]):
        c2 = c2.copy(editable=True)
        c2._append_idling_layers_inplace(max_cir_len - len(c2))

        local_num_lines = c2.num_lines
        local_labels = list(c2._labels)
        sslbl_map = {
            k: v
            for k, v in zip(c2.line_labels, target_lines[i + 1])
        }

        # Remap line labels first.
        c2.map_state_space_labels_inplace(sslbl_map)

        # Then rebuild labels from the original local labels, explicitly remapping
        # LabelTupTup members.
        c2._static = False
        c2._labels = [
            _map_label_state_space_labels(
                layer_mappers[local_num_lines][_to_label(ell)],
                sslbl_map,
            )
            for ell in local_labels
        ]

        c2.done_editing()
        c = c.tensor_circuit(c2)

    c = c.reorder_lines(global_line_order)
    c.saved_auxinfo["lanes"] = {}
    split_into_tensor_lanes(c, cache_lanes_in_circuit=True)
    return c

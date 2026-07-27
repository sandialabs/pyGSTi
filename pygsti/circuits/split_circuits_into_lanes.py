import networkx as _nx

from typing import Sequence, Dict, Tuple, Optional, Set, Union
from pygsti.circuits import Circuit as Circuit
from pygsti.baseobjs.label import Label


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


def _lookup_cached_lanes(circuit: Circuit,
                         lane_to_qubits: dict[int, tuple[int, ...]]) -> Optional[list[Circuit]]:
    """
    Return the cached lane circuits for `circuit`, if a matching cache is available.

    The lanes cache is only trustworthy for *static* (read-only) circuits, and only
    when its size matches the (freshly computed) `lane_to_qubits` partition -- otherwise
    `None` is returned so the caller falls back to (re)computing the lanes.
    """
    if not (circuit._static and "lanes" in circuit.saved_auxinfo):
        return None
    cached_lanes = circuit.saved_auxinfo["lanes"]
    if len(lane_to_qubits) != len(cached_lanes):
        return None

    # We may have this already in cache.
    lane_circuits = [None] * len(lane_to_qubits)
    for i, key in lane_to_qubits.items():
        sorted_key = tuple(sorted(key))
        if sorted_key not in cached_lanes:
            raise ValueError(f"lbl cache miss: {key} in circuit {circuit}")
        lane_circuits[i] = cached_lanes[sorted_key]
    return lane_circuits


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

    cached_lane_circuits = _lookup_cached_lanes(circuit, lane_to_qubits)
    if cached_lane_circuits is not None:
        return cached_lane_circuits

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


def _prepare_target_circuit(source: Circuit, target_line_labels: Tuple[Union[int, str], ...],
                            max_len: int, layer_mappers: Dict[int, Dict]) -> Circuit:
    """
    Pad `source` to `max_len` layers, apply the appropriate `layer_mappers` entry to
    each (local) layer label, and relabel its lines to `target_line_labels`.

    Returns a brand-new `Circuit` (built via the public `Circuit` constructor) with
    `target_line_labels` as its lines -- `source` itself is left untouched.
    """
    padded = source.copy(editable=True)
    padded._append_idling_layers_inplace(max_len - len(padded))
    padded.done_editing()

    sslbl_map = dict(zip(padded.line_labels, target_line_labels))
    mapper = layer_mappers[padded.num_lines]
    new_layers = [
        mapper[layer_lbl].map_state_space_labels(sslbl_map)
        for layer_lbl in padded.layertup
    ]
    return Circuit(new_layers, line_labels=target_line_labels, editable=False)


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

    c = _prepare_target_circuit(circuits[0], target_lines[0], max_cir_len, layer_mappers)
    for i, c2 in enumerate(circuits[1:]):
        c2 = _prepare_target_circuit(c2, target_lines[i + 1], max_cir_len, layer_mappers)
        c = c.tensor_circuit(c2)

    c = c.reorder_lines(global_line_order)
    c.saved_auxinfo["lanes"] = {}
    split_into_tensor_lanes(c, cache_lanes_in_circuit=True)
    return c

import numpy as _np

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

def compute_qubit_to_lane_and_lane_to_qubits_mappings_for_circuit(circuit: Circuit) -> tuple[dict[int, int],
                                                                                             dict[int, tuple[int]]]:
    """
    Parameters:
    ------------
    circuit: Circuit - the circuit to compute qubit to lanes mapping for

    Returns
    --------
    Dictionary mapping qubit number to lane number in the circuit.
    """

    qubits_to_potentially_entangled_others = {i: set((i,)) for i in circuit.line_labels}
    num_layers = circuit.num_layers
    for layer_ind in range(num_layers):
        layer = circuit.layer(layer_ind)
        for op in layer:
            qubits_used = op.qubits
            for qb in qubits_used:
                qubits_to_potentially_entangled_others[qb].update(set(qubits_used))

    lanes = {}
    lan_num = 0
    visited: dict[int, int] = {}

    def reachable_nodes(starting_point: int,
                        graph_qubits_to_neighbors: dict[int, set[int]],
                        visited: dict[int, set[int]]):
        """
        Find which nodes are reachable from this starting point.
        """
        if starting_point in visited:
            return visited[starting_point]
        else:
            assert starting_point in graph_qubits_to_neighbors
            visited[starting_point] = graph_qubits_to_neighbors[starting_point]
            output = set(visited[starting_point])
            for child in graph_qubits_to_neighbors[starting_point]:
                if child != starting_point:
                    output.update(output, reachable_nodes(child, graph_qubits_to_neighbors, visited))
            visited[starting_point] = output
            return output

    available_starting_points = list(sorted(qubits_to_potentially_entangled_others.keys()))
    while available_starting_points:
        sp = available_starting_points[0]
        nodes = reachable_nodes(sp, qubits_to_potentially_entangled_others, visited)
        for node in nodes:
            available_starting_points.remove(node)
        lanes[lan_num] = nodes
        lan_num += 1

    def compute_qubits_to_lanes(lanes_to_qubits: dict[int, set[int]]) -> dict[int, int]:
        """
        Determine a mapping from qubit to the lane it is in for this specific circuit.
        """
        out = {}
        for key, val in lanes_to_qubits.items():
            for qb in val:
                out[qb] = key
        return out

    return compute_qubits_to_lanes(lanes), lanes


def compute_subcircuits(circuit: Circuit,
                        qubit_to_lanes: Optional[dict[int, int]] = None,
                        lane_to_qubits: Optional[dict[int, tuple[int, ...]]] = None,
                        cache_lanes_in_circuit: bool = False,
                        idle_gate_name: Union[str, Label]="Gi") -> list[list[LabelTupTup]]:
    """
    Split a circuit into multiple subcircuits which do not talk across lanes.

    Returns a list of those subcircuits in the format of a list of layers.
    """

    if qubit_to_lanes is None or lane_to_qubits is None:
        qubit_to_lanes, lane_to_qubits = compute_qubit_to_lane_and_lane_to_qubits_mappings_for_circuit(circuit)

    # The lanes cache is only trustworthy for *static* (read-only) circuits:
    if circuit._static and "lanes" in circuit.saved_auxinfo:
        # Check if the lane info matches and I can just return that set up.
        if len(lane_to_qubits) == len(circuit.saved_auxinfo["lanes"]):
            # We may have this already in cache.

            lanes_to_gates = [[] for _ in range(len(lane_to_qubits))]
            for i, key in lane_to_qubits.items():
                if tuple(sorted(key)) in circuit.saved_auxinfo["lanes"]:
                    lanes_to_gates[i] = circuit.saved_auxinfo["lanes"][tuple(sorted(key))]

                else:
                    raise ValueError(f"lbl cache miss: {key} in circuit {circuit}")
            return lanes_to_gates

    lanes_to_gates = [[] for _ in range(len(lane_to_qubits))]

    num_layers = circuit.num_layers
    for layer_ind in range(num_layers):
        layer = circuit.layer_with_idles(layer_ind, idle_gate_name)
        lane_groups = [[] for _ in range(len(lane_to_qubits))]
        for op in layer:
            qubits_used = op.qubits
            lane = qubit_to_lanes[qubits_used[0]]
            lane_groups[lane].append(op)

        for lane, group in enumerate(lane_groups):
            # Sort the operations in each lane group by their first qubit for deterministic output
            sorted_group = sorted(group, key=lambda x: x.qubits[0])
            lanes_to_gates[lane].append(Label(tuple(sorted_group))) # Go through the label factory.

    if cache_lanes_in_circuit:
        circuit = circuit._cache_tensor_lanes(lanes_to_gates, lane_to_qubits)

    return lanes_to_gates


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
    compute_subcircuits(c, cache_lanes_in_circuit=True)
    return c

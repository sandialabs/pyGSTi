import numpy as _np

from typing import Sequence, Dict, Tuple, Optional, Set
from pygsti.circuits import Circuit as Circuit
from pygsti.baseobjs.label import Label, LabelTupTup


def compute_qubit_to_lane_and_lane_to_qubits_mappings_for_circuit(circuit: Circuit) -> tuple[dict[int, int],
                                                                                        dict[int, tuple[int]]]:
    """
    Parameters:
    ------------
    circuit: Circuit - the circuit to compute qubit to lanes mapping for

    num_qubits: int - The total number of qubits expected in the circuit.

    Returns
    --------
    Dictionary mapping qubit number to lane number in the circuit.
    """

    qubits_to_potentially_entangled_others = {i: set((i,)) for i in range(circuit.num_lines)}
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


def compute_subcircuits(circuit: Circuit, qubits_to_lanes: dict[int, int]) -> list[list[LabelTupTup]]:
    """
    Split a circuit into multiple subcircuits which do not talk across lanes.
    """

    if "lanes" in circuit.saved_auxinfo:
        # Check if the lane info matches and I can just return that set up.
        lane_to_qubits: dict[int, tuple[int, ...]]= {}
        for qu, val in qubits_to_lanes.items():
            if val in lane_to_qubits:
                lane_to_qubits[val] = (*lane_to_qubits[val], qu)
            else:
                lane_to_qubits[val] = (qu,)

        if len(lane_to_qubits) == len(circuit.saved_auxinfo["lanes"]):
            # We may have this already in cache.

            lanes_to_gates = [[] for _ in range(len(lane_to_qubits))]
            for i, key in lane_to_qubits.items():
                if sorted(key) in circuit.saved_auxinfo["lanes"]:
                    lanes_to_gates[i] = circuit.saved_auxinfo["lanes"][sorted(key)].layertup

                else:
                    raise ValueError(f"lbl cache miss: {key} in circuit {circuit}")
            return lanes_to_gates

    lanes_to_gates = [[] for _ in range(_np.unique(list(qubits_to_lanes.values())).shape[0])]

    num_layers = circuit.num_layers
    for layer_ind in range(num_layers):
        layer = circuit.layer_with_idles(layer_ind)
        group = []
        group_lane = None
        sorted_layer = sorted(layer, key=lambda x: x.qubits[0])

        for op in sorted_layer:
            # We need this to be sorted by the qubit number so we do not get that a lane was split Q1 Q3 Q2 in the layer where Q1 and Q2 are in the same lane.
            qubits_used = op.qubits # This will be a list of qubits used.
            # I am assuming that the qubits are indexed numerically and not by strings.
            lane = qubits_to_lanes[qubits_used[0]]

            if group_lane is None:
                group_lane = lane
                group.append(op)
            elif group_lane == lane:
                group.append(op)
            else:
                lanes_to_gates[group_lane].append(LabelTupTup(tuple(group)))
                group_lane = lane
                group = [op]

        if len(group) > 0:
            # We have a left over group.
            lanes_to_gates[group_lane].append(LabelTupTup(tuple(group)))

    if num_layers == 0:
        return lanes_to_gates

    return lanes_to_gates


@staticmethod
def batch_tensor(
    circuits : Sequence[Circuit],
    layer_mappers: Dict[int, Dict],
    global_line_order: Optional[Tuple[int,...]] = None,
    target_lines : Optional[Sequence[Tuple[int,...]]] = None
    ) -> Circuit:
    """
    """
    assert len(circuits) > 0

    if target_lines is None:
        target_lines = []
        total_lines = 0
        max_cir_len = 0
        for c in circuits:
            target_lines.append(tuple(range(total_lines, total_lines + c.num_lines)))
            total_lines += c.num_lines
            max_cir_len = max(max_cir_len, len(c))
    else:
        total_lines = sum([c.num_lines for c in circuits])
        max_cir_len = max(*[len(c) for c in circuits])

    s : Set[int] = set()
    for c, t in zip(circuits, target_lines):
        assert not s.intersection(t)
        assert len(t) == c.num_lines
        s.update(t)

    if global_line_order is None:
        global_line_order = tuple(sorted(list(s)))
    
    c = circuits[0].copy(editable=True)
    c._append_idling_layers_inplace(max_cir_len - len(c))
    c.done_editing()
    # ^ That changes the format of c._labels. We need to edit c while in this format,
    #   so the next line sets c._static = False. (We repeat this pattern in the loop below.)
    c._static = False
    c._labels = [layer_mappers[c.num_lines][ell] for ell in c._labels]
    c.map_state_space_labels_inplace({k:v for k,v in zip(c.line_labels, target_lines[0])})
    c.done_editing()
    for i, c2 in enumerate(circuits[1:]):
        c2 = c2.copy(editable=True)
        c2._append_idling_layers_inplace(max_cir_len - len(c2))
        c2.done_editing()
        c2._static = False
        c2._labels = [layer_mappers[c2.num_lines][ell] for ell in c2._labels]
        c2.map_state_space_labels_inplace({k:v for k,v in zip(c2.line_labels, target_lines[i+1])})
        c2.done_editing()
        c = c.tensor_circuit(c2)

    c = c.reorder_lines(global_line_order)
    return c

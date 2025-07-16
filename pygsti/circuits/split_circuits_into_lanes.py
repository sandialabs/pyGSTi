import numpy as _np

from pygsti.circuits import Circuit as _Circuit
from pygsti.baseobjs.label import Label, LabelTupTup

def compute_qubit_to_lane_and_lane_to_qubits_mappings_for_circuit(circuit: _Circuit) -> tuple[dict[int, int],
                                                                                        dict[int, tuple[int]]]:
    """
    Parameters:
    ------------
    circuit: _Circuit - the circuit to compute qubit to lanes mapping for

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


def compute_subcircuits(circuit: _Circuit, qubits_to_lanes: dict[int, int]) -> list[list[LabelTupTup]]:
    """
    Split a circuit into multiple subcircuits which do not talk across lanes.
    """

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
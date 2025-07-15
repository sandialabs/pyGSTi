from pygsti.circuits.circuit import Circuit as _Circuit
from pygsti.baseobjs.label import Label
from pygsti.circuits.split_circuits_into_lanes import compute_subcircuits, compute_qubit_to_lane_and_lane_to_qubits_mappings_for_circuit
import numpy as np


def build_circuit(num_qubits: int, depth_L: int, allowed_gates: set[str]):
    my_circuit = []
    for lnum in range(depth_L):
        layer = []
        for qnum in range(num_qubits):
            gate = str(np.random.choice(allowed_gates))
            layer.append((gate, qnum))
        my_circuit.append(layer)
    return _Circuit(my_circuit)


def build_circuit_with_multiple_qubit_gates_with_designated_lanes(num_qubits: int, depth_L: int, lane_end_points: list[int], gates_to_qubits_used: dict[str, int]):

    assert lane_end_points[-1] <= num_qubits # if < then we have a lane from there to num_qubits.
    assert lane_end_points[0] > 0
    assert np.all(np.diff(lane_end_points) > 0) # then it is sorted in increasing order.

    if lane_end_points[-1] < num_qubits:
        lane_end_points.append(num_qubits)

    my_circuit = []
    n_qs_to_gates_avail = {}
    for key, val in gates_to_qubits_used.items():
        if val in n_qs_to_gates_avail:
            n_qs_to_gates_avail[val].append(key)
        else:
            n_qs_to_gates_avail[val] = [key]

    for lnum in range(depth_L):
        layer = []
        start_point = 0

        for lane_ep in lane_end_points:
            num_used: int = 0
            while num_used < (lane_ep - start_point):
                navail = (lane_ep - start_point) - num_used
                nchosen = 0
                if navail >= max(n_qs_to_gates_avail):
                    # we can use any gate
                    nchosen = np.random.randint(1, max(n_qs_to_gates_avail) + 1)
                else:
                    # we need to first choose how many to use.
                    nchosen = np.random.randint(1, navail + 1)
                gate = str(np.random.choice(n_qs_to_gates_avail[nchosen]))
                tmp = list(np.random.permutation(nchosen) + num_used + start_point) # Increase to offset.
                perm_of_qubits_used = [int(tmp[ind]) for ind in range(len(tmp))]
                if gate == "Gcustom":
                    layer.append(Label(gate, *perm_of_qubits_used, args=(np.random.random(4)*4*np.pi)))
                else:
                    layer.append((gate, *perm_of_qubits_used))
                num_used += nchosen

            if num_used > (lane_ep - start_point) + 1:
                print(num_used, f"lane ({start_point}, {lane_ep})")
                raise AssertionError("lane barrier is broken")
            
            start_point = lane_ep
        my_circuit.append(layer)
    return _Circuit(my_circuit, line_labels=[i for i in range(num_qubits)])


def test_subcircuits_splits_can_create_empty_sub_circuit():


    original = _Circuit([], line_labels=[0])

    qubits_to_lanes = {0: 0}

    attempt = compute_subcircuits(original, qubits_to_lanes)

    assert original == _Circuit(attempt, line_labels=[0])


def test_find_qubit_to_lane_splitting():

    gates_to_num_used = {"X": 1, "Y": 1, "Z": 1, "CNOT": 2, "CZ": 2}

    depth = 10
    num_qubits = 6

    lane_eps = [1, 2, 4, 5]
    # So expected lane dist is (0, ), (1), (2,3), (4,), (5,)

    circuit = build_circuit_with_multiple_qubit_gates_with_designated_lanes(num_qubits, depth, lane_eps, gates_to_num_used)

    qubit_to_lane, lane_to_qubits = compute_qubit_to_lane_and_lane_to_qubits_mappings_for_circuit(circuit)


    assert len(qubit_to_lane) == num_qubits

    assert len(lane_to_qubits) <= num_qubits

    circuit = _Circuit([[]])

test_subcircuits_splits_can_create_empty_sub_circuit()
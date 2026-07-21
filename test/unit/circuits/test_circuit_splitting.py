from pygsti.circuits.circuit import Circuit as _Circuit
from pygsti.baseobjs.label import Label
from pygsti.circuits.split_circuits_into_lanes import (
    compute_subcircuits,
    compute_qubit_to_lane_and_lane_to_qubits_mappings_for_circuit,
    batch_tensor
)
import numpy as np
import pytest
from ..util import BaseCase


def build_circuit(num_qubits: int, depth_L: int, allowed_gates: set[str]) -> _Circuit:
    """
    Build a random circuit of depth L which operates on num_qubits and has the allowed
    single qubit gates specified in allowed gates.
    """
    my_circuit = []
    for _ in range(depth_L):
        layer = []
        for qnum in range(num_qubits):
            gate = str(np.random.choice(allowed_gates))
            layer.append((gate, qnum))
        my_circuit.append(layer)
    return _Circuit(my_circuit, line_labels=[i for i in range(num_qubits)])


def build_circuit_with_multiple_qubit_gates_with_designated_lanes(
                                num_qubits: int,
                                depth_L: int,
                                lane_end_points: list[int],
                                gates_to_qubits_used: dict[str, int]) -> _Circuit:
    """
    Builds a circuit with a known lane structure.
    Any two + qubit lanes can be split into smaller lanes if none of the gates
    chosen for that lane actually operate on two or more qubits.
    """

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

    for _ in range(depth_L):
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


class CircuitSplittingTester(BaseCase):
    def test_subcircuits_splits_can_create_empty_sub_circuit(self):
        original = _Circuit([], line_labels=[0])

        qubits_to_lanes = {0: 0}
        lane_to_qubits = {0: (0,)}

        attempt = compute_subcircuits(original, qubits_to_lanes, lane_to_qubits)
        self.assertEqual(original, _Circuit(attempt[0], line_labels=[0]))

    def test_subcircuits_split_can_be_cached(self):
        gates_to_num_used = {"X": 1, "Y": 1, "Z": 1, "CNOT": 2, "CZ": 2}

        depth = 10
        num_qubits = 6

        lane_eps = [1, 2, 4, 5]
        # So expected lane dist is (0, ), (1), (2,3), (4,), (5,)

        # This is a random circuit so the lanes may not be perfect.
        circuit = build_circuit_with_multiple_qubit_gates_with_designated_lanes(num_qubits, depth, lane_eps, gates_to_num_used)
        qubit_to_lane, lane_to_qubits = compute_qubit_to_lane_and_lane_to_qubits_mappings_for_circuit(circuit)

        # The lanes cache is populated lazily, so nothing is cached yet.
        self.assertNotIn("lanes", circuit.saved_auxinfo)

        sub_cirs = compute_subcircuits(circuit, qubit_to_lane, lane_to_qubits, cache_lanes_in_circuit=True)
        self.assertIn("lanes", circuit.saved_auxinfo)
        self.assertEqual(len(circuit.saved_auxinfo["lanes"].keys()), len(sub_cirs))

    def test_subcircuits_split_cache_miss(self):
        c = _Circuit([('Gx', 0), ('Gy', 1)], line_labels=[0, 1])
        qubit_to_lane, lane_to_qubits = compute_qubit_to_lane_and_lane_to_qubits_mappings_for_circuit(c)

        # Cache the lanes in the circuit.
        sub_cirs = compute_subcircuits(c, qubit_to_lane, lane_to_qubits, cache_lanes_in_circuit=True)
        self.assertIn("lanes", c.saved_auxinfo)

        # Create a copy and map state space labels, which will result in cleared lanes cache on done_editing.
        c_mapped = c.copy(editable=True)
        c_mapped.map_state_space_labels_inplace({0: 10, 1: 11})
        c_mapped.done_editing()

        # The cache should be empty (invalidated) after done_editing
        self.assertEqual(c_mapped.saved_auxinfo["lanes"], {})

        # Now, manually define lane mapping for the mapped circuit.
        qubit_to_lane_mapped = {10: 0, 11: 1}
        lane_to_qubits_mapped = {0: (10,), 1: (11,)}

        # Manually inject a stale key to verify that "lbl cache miss" check works when a mismatching key is present.
        c_mapped.saved_auxinfo["lanes"] = {(0,): None, (1,): None}
        with self.assertRaisesRegex(ValueError, "lbl cache miss"):
            compute_subcircuits(c_mapped, qubit_to_lane_mapped, lane_to_qubits_mapped)

    def test_find_qubit_to_lane_splitting(self):
        gates_to_num_used = {"X": 1, "Y": 1, "Z": 1, "CNOT": 2, "CZ": 2}

        depth = 10
        num_qubits = 6

        lane_eps = [1, 2, 4, 5]
        # So expected lane dist is (0, ), (1), (2,3), (4,), (5,)
        minimum_num_lanes = 5

        # This is a random circuit so the lanes may not be perfect.
        circuit = build_circuit_with_multiple_qubit_gates_with_designated_lanes(num_qubits, depth, lane_eps, gates_to_num_used)

        qubit_to_lane, lane_to_qubits = compute_qubit_to_lane_and_lane_to_qubits_mappings_for_circuit(circuit)

        self.assertEqual(len(qubit_to_lane), num_qubits)
        self.assertGreaterEqual(len(lane_to_qubits), minimum_num_lanes)
        self.assertLessEqual(len(lane_to_qubits), num_qubits)

        for qubit in qubit_to_lane:
            self.assertIn(qubit_to_lane[qubit], lane_to_qubits)

        for lane in lane_to_qubits:
            for qu in lane_to_qubits[lane]:
                self.assertIn(qu, qubit_to_lane)
                self.assertEqual(lane, qubit_to_lane[qu])

    def test_batch_tensor_single_circuit(self):
        c1 = _Circuit("Gx:0", line_labels=(0,))
        idle_label = Label(())
        labels_in_circuits = [Label('Gx', (0,)), idle_label]
        map_d = {l: l for l in labels_in_circuits}
        map_d[idle_label] = Label("Gi", 0)
        layer_mappers = {1: map_d}
        with self.assertRaises(ValueError):
            batch_tensor([c1], layer_mappers, target_lines=((0,),))

    def test_batch_tensor_diff_lengths(self):
        c1 = _Circuit("Gx:0", line_labels=(0,))
        c2 = _Circuit("Gy:0Gz:0", line_labels=(0,))

        idle_label = Label(()) # empty label is the idle
        labels_in_circuits = [Label('Gx', (0,)), Label('Gy', (0,)), Label("Gz", 0), idle_label]
        map_d = {l: l for l in labels_in_circuits}
        map_d[idle_label] = Label("Gi", 0)
        layer_mappers = {1: map_d, 2: map_d}

        # Call batch_tensor
        tensored_c = batch_tensor([c1, c2], layer_mappers)

        expected_c = c1.tensor_circuit(c2.map_state_space_labels({0:1}))

        # manually construct the expected circuit
        self.assertEqual(tensored_c, expected_c) # now equal due to explicit idle padding
        self.assertEqual(tensored_c[0], expected_c[0])
        self.assertEqual(tensored_c[1], expected_c[1])

    def test_batch_tensor_reorder(self):
        c1 = _Circuit("Gx:0", line_labels=(0,))
        c2 = _Circuit("Gy:0", line_labels=(0,))
        idle_label = Label(()) # empty label is the idle
        labels_in_circuits = [Label('Gx', (0,)), Label('Gy', (0,)), idle_label]
        map_d = {l: l for l in labels_in_circuits}
        map_d[idle_label] = Label("Gi", 0)
        layer_mappers = {1: map_d, 2: map_d}

        # Call batch_tensor
        tensored_c = batch_tensor([c1, c2], layer_mappers, global_line_order=('Q1', 'Q0'), target_lines=(('Q0',), ('Q1',)))
        expected_c = _Circuit([Label([('Gx', 'Q0'), ('Gy', 'Q1')])], line_labels=['Q1', 'Q0'])

        self.assertEqual(tensored_c, expected_c)
        self.assertEqual(tensored_c.line_labels, ("Q1", "Q0"))
        # We store them still in the canonically ordered form.
        # However, we will print them in the order specified by line labels.
        self.assertEqual(tensored_c[0][1], Label(("Gy", "Q1")))
        self.assertEqual(tensored_c[0][0], Label(("Gx", "Q0")))

    def test_batch_tensor_reorder_with_multiqubit_gate(self):
        # Three circuits, one of which contains a 2-qubit gate, tensored together with
        # non-contiguous/non-default target_lines and a global_line_order that is neither
        # the default sorted order nor a simple reversal.
        c1 = _Circuit([Label('Gcnot', (0, 1))], line_labels=(0, 1))
        c2 = _Circuit("Gy:0", line_labels=(0,))
        c3 = _Circuit("Gz:0", line_labels=(0,))

        circuits = [c1, c2, c3]

        idle_label = Label(())  # empty label is the idle
        labels_in_circuits = [
            Label('Gcnot', (0, 1)), Label('Gy', (0,)), Label('Gz', (0,)), idle_label
        ]
        map_d = {l: l for l in labels_in_circuits}
        map_d[idle_label] = Label("Gi", 0)
        layer_mappers = {1: map_d, 2: map_d}

        # c1 uses two target lines, c2 and c3 each use one.
        target_lines = (('Q0', 'Q2'), ('Q1',), ('Q3',))
        # Neither sorted(('Q0','Q1','Q2','Q3')) nor its reverse.
        global_line_order = ('Q3', 'Q0', 'Q1', 'Q2')

        tensored_c = batch_tensor(
            circuits, layer_mappers,
            global_line_order=global_line_order,
            target_lines=target_lines
        )

        self.assertEqual(tensored_c.line_labels, global_line_order)

        expected_c = _Circuit(
            [Label([('Gcnot', 'Q0', 'Q2'), ('Gy', 'Q1'), ('Gz', 'Q3')])],
            line_labels=['Q3', 'Q0', 'Q1', 'Q2']
        )
        self.assertEqual(tensored_c, expected_c)

        # Confirm the 2-qubit gate still acts on its correct (remapped) target lines,
        # not on some other pair introduced by the reordering.
        layer = tensored_c[0]
        cnot_ops = [op for op in layer if op.name == 'Gcnot']
        self.assertEqual(len(cnot_ops), 1)
        self.assertEqual(set(cnot_ops[0].qubits), {'Q0', 'Q2'})

    def test_batch_tensor_string_labels(self):
        c1 = _Circuit("Gx:0", line_labels=(0,))
        c2 = _Circuit("Gy:0", line_labels=(0,))
        idle_label = Label(()) # empty label is the idle
        labels_in_circuits = [Label('Gx', (0,)), Label('Gy', (0,)), idle_label]
        map_d = {l: l for l in labels_in_circuits}
        map_d[idle_label] = Label("Gi", 0)
        layer_mappers = {1: map_d, 2: map_d}

        # Call batch_tensor
        tensored_c = batch_tensor([c1, c2], layer_mappers, target_lines=(('Q0',), ('Q1',)))
        expected_c = _Circuit([Label([('Gx', 'Q0'), ('Gy', 'Q1')])], line_labels=['Q0', 'Q1'])

        self.assertEqual(tensored_c, expected_c)

    def test_batch_tensor_5_circuits_with_2q_gate(self):
        c1 = _Circuit("Gx:0", line_labels=(0,))
        c2 = _Circuit("Gy:0", line_labels=(0,))
        c3 = _Circuit([Label("Gcnot", (0, 1))], line_labels=(0, 1))
        c4 = _Circuit("Gz:0", line_labels=(0,))
        c5 = _Circuit("Gh:0", line_labels=(0,))

        circuits = [c1, c2, c3, c4, c5]

        idle_label = Label(())  # empty label is the idle
        labels_in_circuits = [
            Label('Gx', (0,)), Label('Gy', (0,)), Label('Gz', (0,)), Label('Gh', (0,)),
            idle_label
        ]
        map_d = {l: l for l in labels_in_circuits}
        map_d[idle_label] = Label("Gi", 0)
        map_2d = {k: v for k,v in map_d.items()}
        map_2d[Label('Gcnot', (0, 1))] = Label('Gcnot', (0, 1))
        
        layer_mappers = {1: map_d, 2: map_2d}

        tensored_c = batch_tensor(circuits, layer_mappers)

        expected_c = _Circuit([
            Label([
                ('Gx', 0),
                ('Gy', 1),
                ('Gcnot', 2, 3),
                ('Gz', 4),
                ('Gh', 5)
            ])
        ])

        self.assertEqual(tensored_c, expected_c)

    def test_subcircuits_interleaved_lanes_depth(self):
        # Create a circuit where Lane 0 contains non-contiguous qubits {0, 2}
        # and Lane 1 contains qubit {1}.
        c_interleaved = _Circuit([
            [Label('Gcnot', (0, 2))],
            [Label('Gx', 0), Label('Gy', 1), Label('Gz', 2)]
        ], line_labels=[0, 1, 2])

        qubit_to_lane, lane_to_qubits = compute_qubit_to_lane_and_lane_to_qubits_mappings_for_circuit(c_interleaved)

        # Confirm the lane structure is interleaved.
        # Lane 0 should be {0, 2} and Lane 1 should be {1}.
        self.assertEqual(lane_to_qubits[0], {0, 2})
        self.assertEqual(lane_to_qubits[1], {1})

        # Compute the subcircuits.
        sub_cirs = compute_subcircuits(c_interleaved, qubit_to_lane, lane_to_qubits, cache_lanes_in_circuit=False)

        # Expected: Each subcircuit has exactly 2 layers (since the input circuit has 2 layers).
        # The lane-keyed grouping fix ensures that Lane 0's subcircuit second layer is not
        # split into two layers, preserving the correct depth of 2.
        self.assertEqual(len(sub_cirs[0]), 2, f"Lane 0 subcircuit depth inflated to {len(sub_cirs[0])}!")
        self.assertEqual(len(sub_cirs[1]), 2)

    def test_subcircuits_contiguous_lanes_depth(self):
        # Control case: Create a circuit where Lane 0 contains contiguous qubits {0, 1}
        # and Lane 1 contains qubit {2}.
        c_contiguous = _Circuit([
            [Label('Gcnot', (0, 1))],
            [Label('Gx', 0), Label('Gy', 1), Label('Gz', 2)]
        ], line_labels=[0, 1, 2])

        qubit_to_lane, lane_to_qubits = compute_qubit_to_lane_and_lane_to_qubits_mappings_for_circuit(c_contiguous)

        # Confirm the lane structure is contiguous.
        # Lane 0 should be {0, 1} and Lane 1 should be {2}.
        self.assertEqual(lane_to_qubits[0], {0, 1})
        self.assertEqual(lane_to_qubits[1], {2})

        # Compute the subcircuits.
        sub_cirs = compute_subcircuits(c_contiguous, qubit_to_lane, lane_to_qubits, cache_lanes_in_circuit=False)

        # For contiguous lanes, sorting by qubit index works perfectly.
        # So each subcircuit should correctly have exactly 2 layers (depth 2).
        self.assertEqual(len(sub_cirs[0]), 2)
        self.assertEqual(len(sub_cirs[1]), 2)

    def test_subcircuits_multi_layer_interleaved_lanes(self):
        # Additional multi-layer interleaved regression test to confirm the one-layer-per-layer-per-lane invariant
        # holds across arbitrary depths.
        # Here we have 4 qubits: Lane 0 = {0, 3}, Lane 1 = {1}, Lane 2 = {2} (interleaved)
        # Deep multi-layer circuit:
        c_deep = _Circuit([
            [Label('Gcnot', (0, 3))],
            [Label('Gx', 0), Label('Gy', 1), Label('Gz', 2), Label('Gh', 3)],
            [Label('Gy', 1), Label('Gz', 2)],
            [Label('Gx', 0), Label('Gh', 3)],
            [Label('Gcnot', (0, 3))]
        ], line_labels=[0, 1, 2, 3])

        qubit_to_lane, lane_to_qubits = compute_qubit_to_lane_and_lane_to_qubits_mappings_for_circuit(c_deep)

        self.assertEqual(lane_to_qubits[0], {0, 3})
        self.assertEqual(lane_to_qubits[1], {1})
        self.assertEqual(lane_to_qubits[2], {2})

        sub_cirs = compute_subcircuits(c_deep, qubit_to_lane, lane_to_qubits, cache_lanes_in_circuit=False)

        # The subcircuit for every single lane must contain exactly 5 layers (equal to input circuit layers),
        # demonstrating that our fix perfectly preserves the one-layer-per-layer invariant under interleaving.
        self.assertEqual(len(sub_cirs[0]), 5)
        self.assertEqual(len(sub_cirs[1]), 5)
        self.assertEqual(len(sub_cirs[2]), 5)

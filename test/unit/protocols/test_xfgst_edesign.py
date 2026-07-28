#***************************************************************************************************
# Copyright 2015, 2019, 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

import numpy as np

from pygsti.circuits.circuit import Circuit
from pygsti.baseobjs.label import Label, LabelTup
from pygsti.protocols.xfgst_edesign import (
    assign_the_designs_with_mapping, assert_circuit_lists_match_color_patches, build_patch_infos,
)
from ..util import BaseCase


class _StubDesign:
    """Minimal stand-in for a GateSetTomographyDesign for stitcher unit tests.

    ``assign_the_designs_with_mapping`` only touches ``.circuit_lists`` (and, via
    the layer-mapper override, not ``.qubit_labels``), so a tiny stub is enough to
    exercise the length-pairing logic without building full GST designs.
    """
    def __init__(self, circuit_lists, qubit_labels):
        self.circuit_lists = circuit_lists
        self.qubit_labels = qubit_labels


def _make_1q_circuits(n):
    """n distinct single-qubit circuits on line (0,)."""
    return [Circuit([Label('Gx', 0)] * (i + 1), line_labels=(0,)) for i in range(n)]


def _make_2q_circuits(n):
    """n distinct two-qubit circuits on lines (0, 1)."""
    return [Circuit([Label('Gcnot', (0, 1))] * (i + 1), line_labels=(0, 1)) for i in range(n)]


def _layer_mappers(oneq_lists, twoq_lists):
    """Build explicit-idle layer mappers for the stub designs.

    Mirrors the mapping strategy used by the integration test: the empty
    (implicit-idle) layer maps to an explicit idle, a 1Q idle maps to itself, and a
    local 2Q idle / any lone 1Q label inside a 2Q lane maps to a parallel pair of
    1Q labels so that no implicit idle survives.
    """
    oneq_idle = Label('Gi', 0)
    twoq_idle = Label(('Gii', 0, 1))
    parallel_idle = Label((Label('Gi', 0), Label('Gi', 1)))

    mapper_1q = {Label(()): oneq_idle, oneq_idle: oneq_idle}
    for cl in oneq_lists:
        for c in cl:
            for ell in c._labels:
                mapper_1q[ell] = oneq_idle if ell == Label(()) else ell

    mapper_2q = {Label(()): parallel_idle, twoq_idle: parallel_idle}
    for cl in twoq_lists:
        for c in cl:
            for ell in c._labels:
                if ell == Label(()) or ell == twoq_idle:
                    mapper_2q[ell] = parallel_idle
                elif isinstance(ell, LabelTup) and ell.num_qubits == 1:
                    tgt = ell.qubits[0]
                    tmp = [None, None]
                    tmp[tgt] = ell
                    tmp[1 - tgt] = Label('Gi', 1 - tgt)
                    mapper_2q[ell] = Label(tuple(tmp))
                else:
                    mapper_2q[ell] = ell

    return {1: mapper_1q, 2: mapper_2q}


class AssignDesignsLengthPairingTester(BaseCase):
    """Cover the pairing of 1Q and 2Q designs of differing per-L lengths."""

    def _run(self, oneq_len, twoq_len, seed=0):
        oneq_lists = [_make_1q_circuits(oneq_len)]
        twoq_lists = [_make_2q_circuits(twoq_len)]
        oneq = _StubDesign(oneq_lists, (0,))
        twoq = _StubDesign(twoq_lists, (0, 1))
        mappers = _layer_mappers(oneq_lists, twoq_lists)

        # A single color patch: one 2Q edge (0, 1) plus one unused 1Q qubit (2),
        # so both the edge slot and the unused-qubit slot are exercised.
        color_patches = {0: [(0, 1)]}
        vertices = [0, 1, 2]

        circuit_lists = assign_the_designs_with_mapping(
            oneq, twoq, vertices, color_patches,
            randgen=np.random.default_rng(seed),
            _layer_mappers_override=mappers,
        )
        # assign_the_designs_with_mapping no longer verifies its own output;
        # that's now stitcher-agnostic and lives in
        # assert_circuit_lists_match_color_patches (normally invoked by
        # CrosstalkFreeExperimentDesign.__init__). Run it explicitly here
        # since this test calls the stitcher directly.
        assert_circuit_lists_match_color_patches(circuit_lists, vertices, color_patches)
        return circuit_lists

    def test_oneq_longer_than_twoq(self):
        # A longer 1Q design is a legitimate input --
        # neither design is required to be the longer one.
        oneq_len, twoq_len = 7, 3
        circuit_lists = self._run(oneq_len, twoq_len)

        self.assertEqual(len(circuit_lists), 1)  # one germ-power group
        # max(len(oneq), len(twoq)) tensored circuits are produced.
        self.assertEqual(len(circuit_lists[0]), max(oneq_len, twoq_len))
        # Every generated circuit spans all three qubits.
        for c in circuit_lists[0]:
            self.assertEqual(set(c.line_labels), {0, 1, 2})

    def test_twoq_longer_than_oneq(self):
        oneq_len, twoq_len = 3, 7
        circuit_lists = self._run(oneq_len, twoq_len)
        self.assertEqual(len(circuit_lists[0]), max(oneq_len, twoq_len))

    def test_equal_lengths(self):
        oneq_len, twoq_len = 5, 5
        circuit_lists = self._run(oneq_len, twoq_len)
        self.assertEqual(len(circuit_lists[0]), 5)

    def test_shorter_twoq_list_recycled_up_to_longer_oneq(self):
        oneq_len, twoq_len = 6, 2
        circuit_lists = self._run(oneq_len, twoq_len)
        generated = circuit_lists[0]
        self.assertEqual(len(generated), oneq_len)

        cnot_depth_counts = {}
        for c in generated:
            n_cnot_layers = sum(
                1 for i in range(c.num_layers)
                if any(op.name == 'Gcnot' for op in c.layer(i))
            )
            cnot_depth_counts[n_cnot_layers] = cnot_depth_counts.get(n_cnot_layers, 0) + 1

        self.assertNotIn(0, cnot_depth_counts)
        self.assertGreaterEqual(cnot_depth_counts.get(1, 0), 1)
        self.assertGreaterEqual(cnot_depth_counts.get(2, 0), 1)
        self.assertEqual(sum(cnot_depth_counts.values()), oneq_len)

    def test_shorter_oneq_list_recycled_up_to_longer_twoq(self):
        oneq_len, twoq_len = 2, 6
        circuit_lists = self._run(oneq_len, twoq_len)
        generated = circuit_lists[0]
        self.assertEqual(len(generated), twoq_len)

        gx_depth_counts = {}
        for c in generated:
            n_gx_layers = sum(
                1 for i in range(c.num_layers)
                if any(op.name == 'Gx' and 2 in op.qubits for op in c.layer(i))
            )
            gx_depth_counts[n_gx_layers] = gx_depth_counts.get(n_gx_layers, 0) + 1

        self.assertNotIn(0, gx_depth_counts)
        self.assertGreaterEqual(gx_depth_counts.get(1, 0), 1)
        self.assertGreaterEqual(gx_depth_counts.get(2, 0), 1)
        self.assertEqual(sum(gx_depth_counts.values()), twoq_len)

    def test_shorter_circuit_depths_padded_with_explicit_idles(self):
        oneq_len, twoq_len = 4, 4
        circuit_lists = self._run(oneq_len, twoq_len)
        generated = circuit_lists[0]

        depths = {c.num_layers for c in generated}
        self.assertGreater(max(depths), 1)

        for c in generated:
            for i in range(c.num_layers):
                # No implicit idles anywhere: filling idles must not change any layer.
                self.assertEqual(set(c.layer(i)), set(c.layer_with_idles(i)))
            # Every layer covers all lines (shallower lanes were padded with explicit
            # idles), so the circuit is rectangular: each layer touches every qubit.
            for i in range(c.num_layers):
                covered = {q for op in c.layer_with_idles(i) for q in op.qubits}
                self.assertEqual(covered, set(c.line_labels))


class MultiplePatchesSameShapeTester(BaseCase):
    """
    Two color patches sharing the same shape (here, both have 1 edge and 1 unused
    qubit) are grouped together and share a randomly-generated "template" circuit;
    all but the first ("representative") patch in the group must have that template
    relabeled (via `Circuit.map_state_space_labels`) onto their own lines.
    """

    def _run(self, oneq_len=4, twoq_len=4, seed=0):
        oneq_lists = [_make_1q_circuits(oneq_len)]
        twoq_lists = [_make_2q_circuits(twoq_len)]
        oneq = _StubDesign(oneq_lists, (0,))
        twoq = _StubDesign(twoq_lists, (0, 1))
        mappers = _layer_mappers(oneq_lists, twoq_lists)

        # A 3-qubit line with two color patches, both of shape (1 edge, 1 unused
        # qubit): patch 0 is edge (0, 1) with qubit 2 left over; patch 1 is edge
        # (1, 2) with qubit 0 left over. Both patches land in the same `groups`
        # bucket, which is exactly the scenario the bug above hits.
        color_patches = {0: [(0, 1)], 1: [(1, 2)]}
        vertices = [0, 1, 2]

        circuit_lists = assign_the_designs_with_mapping(
            oneq, twoq, vertices, color_patches,
            randgen=np.random.default_rng(seed),
            _layer_mappers_override=mappers,
        )
        # assign_the_designs_with_mapping no longer verifies its own output
        # (see assert_circuit_lists_match_color_patches); this is exactly the
        # scenario (two same-shape patches) that the "duplicated representative
        # patch" bug hit, so exercise the check explicitly here.
        assert_circuit_lists_match_color_patches(circuit_lists, vertices, color_patches)
        generated = circuit_lists[0]
        self.assertEqual(len(generated), max(oneq_len, twoq_len) * 2)
        half = len(generated) // 2
        # Output is patch-major: patch 0's circuits first, then patch 1's.
        return generated[:half], generated[half:]

    @staticmethod
    def _cnot_edges(circuit):
        edges = set()
        for i in range(circuit.num_layers):
            for op in circuit.layer(i):
                if op.name == 'Gcnot':
                    edges.add(tuple(op.qubits))
        return edges

    @staticmethod
    def _gx_qubits(circuit):
        qubits = set()
        for i in range(circuit.num_layers):
            for op in circuit.layer(i):
                if op.name == 'Gx':
                    qubits.update(op.qubits)
        return qubits

    def test_patches_are_relabeled_not_duplicated(self):
        patch0_circuits, patch1_circuits = self._run()
        self.assertNotEqual(patch0_circuits, patch1_circuits)

        # Every patch-0 circuit's Gcnot must be on edge (0, 1) only, and every
        # patch-1 circuit's Gcnot must be on edge (1, 2) only -- never the other
        # patch's edge.
        for c in patch0_circuits:
            self.assertEqual(self._cnot_edges(c), {(0, 1)})
        for c in patch1_circuits:
            self.assertEqual(self._cnot_edges(c), {(1, 2)})

        # Patch 0's leftover 1Q circuit belongs to qubit 2; patch 1's belongs to
        # qubit 0.
        for c in patch0_circuits:
            self.assertEqual(self._gx_qubits(c), {2})
        for c in patch1_circuits:
            self.assertEqual(self._gx_qubits(c), {0})


class EnsureContainmentTester(BaseCase):
    """
    With ``ensure_containment=True``, the circuit list for a higher germ-power
    index must contain every circuit generated for any lower germ-power index,
    patch-wise: each patch's own chunk at germ power ``L`` must start with the
    exact same circuits, in the same order, as that patch's chunk at every
    germ power ``L' < L``.
    """

    def _run(self, oneq_lens, twoq_lens, color_patches, vertices, seed=0):
        oneq_lists = [_make_1q_circuits(n) for n in oneq_lens]
        twoq_lists = [_make_2q_circuits(n) for n in twoq_lens]
        oneq = _StubDesign(oneq_lists, (0,))
        twoq = _StubDesign(twoq_lists, (0, 1))
        mappers = _layer_mappers(oneq_lists, twoq_lists)

        circuit_lists = assign_the_designs_with_mapping(
            oneq, twoq, vertices, color_patches,
            randgen=np.random.default_rng(seed),
            ensure_containment=True,
            _layer_mappers_override=mappers,
        )
        assert_circuit_lists_match_color_patches(circuit_lists, vertices, color_patches)
        return circuit_lists

    def _assert_patchwise_containment(self, circuit_lists, vertices, color_patches):
        patch_infos, _ = build_patch_infos(vertices, color_patches)
        num_patches = len(patch_infos)

        chunk_sizes = []
        for germ_power_list in circuit_lists:
            self.assertEqual(len(germ_power_list) % num_patches, 0)
            chunk_sizes.append(len(germ_power_list) // num_patches)

        # Check every pair of germ powers, not just consecutive ones, so
        # this directly verifies "the higher germ power contains the lower
        # germ power" rather than relying on that following by induction.
        for i in range(len(circuit_lists)):
            for j in range(i + 1, len(circuit_lists)):
                lower_list, higher_list = circuit_lists[i], circuit_lists[j]
                lower_chunk_size, higher_chunk_size = chunk_sizes[i], chunk_sizes[j]
                self.assertGreaterEqual(higher_chunk_size, lower_chunk_size)

                for patch_idx in range(num_patches):
                    lower_start = patch_idx * lower_chunk_size
                    lower_chunk = lower_list[lower_start:lower_start + lower_chunk_size]

                    higher_start = patch_idx * higher_chunk_size
                    higher_chunk = higher_list[higher_start:higher_start + higher_chunk_size]

                    self.assertEqual(
                        higher_chunk[:lower_chunk_size], lower_chunk,
                        f"germ-power {j}'s patch {patch_idx} chunk does not "
                        f"contain germ-power {i}'s patch {patch_idx} chunk "
                        "as a prefix."
                    )

    def test_single_patch_containment_across_germ_powers(self):
        vertices = [0, 1, 2]
        color_patches = {0: [(0, 1)]}
        circuit_lists = self._run(
            oneq_lens=[3, 5], twoq_lens=[3, 5],
            color_patches=color_patches, vertices=vertices,
        )
        self.assertEqual(len(circuit_lists), 2)
        self._assert_patchwise_containment(circuit_lists, vertices, color_patches)

    def test_multiple_same_shape_patches_containment_across_germ_powers(self):
        # Two same-shape patches land in the same `groups` bucket and share a
        # representative template circuit; containment must still hold
        # independently for each patch's own chunk.
        vertices = [0, 1, 2]
        color_patches = {0: [(0, 1)], 1: [(1, 2)]}
        circuit_lists = self._run(
            oneq_lens=[3, 6], twoq_lens=[3, 6],
            color_patches=color_patches, vertices=vertices,
        )
        self.assertEqual(len(circuit_lists), 2)
        self._assert_patchwise_containment(circuit_lists, vertices, color_patches)

    def test_three_germ_powers_containment_holds_for_all_pairs(self):
        vertices = [0, 1, 2]
        color_patches = {0: [(0, 1)]}
        circuit_lists = self._run(
            oneq_lens=[2, 4, 7], twoq_lens=[2, 4, 7],
            color_patches=color_patches, vertices=vertices,
        )
        self.assertEqual(len(circuit_lists), 3)
        self._assert_patchwise_containment(circuit_lists, vertices, color_patches)


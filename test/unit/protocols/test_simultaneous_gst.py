#***************************************************************************************************
# Copyright 2015, 2019, 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

import copy as _copy
import functools
import hashlib
import json
import pathlib
import warnings

import numpy as np

import pygsti
from pygsti.circuits.circuit import Circuit
from pygsti.baseobjs.label import Label
from pygsti.modelpacks import smq1Q_XYI, smq2Q_XYICNOT
from pygsti.processors import QubitProcessorSpec
from pygsti.protocols.simultaneous_gst import (
    SimultaneousGSTDesign, _normalize_coloring, _resolve_stitcher_name, _stitcher_name,
    assert_circuit_lists_match_color_patches,
    assert_mapped_circuit_matches_patch, assert_no_implicit_idles,
    assign_the_designs_with_mapping, build_layer_mappers,
    build_patch_infos, make_line_mapper, make_simultaneous_gst_design,
)
from pygsti.protocols.gst import GateSetTomographyDesign
from pygsti.protocols.protocol import CombinedExperimentDesign
from pygsti.tools.graphcoloring import check_valid_edge_coloring
from ..util import BaseCase, with_temp_path


class _StubDesign:
    """Minimal stand-in for a GateSetTomographyDesign for stitcher unit tests.

    ``assign_the_designs_with_mapping`` only touches ``.circuit_lists`` and
    ``.qubit_labels`` -- both directly and via ``build_layer_mappers``, which it
    calls to construct the layer mappers -- so a tiny stub is enough to exercise
    the length-pairing logic without building full GST designs.
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


class AssignDesignsLengthPairingTester(BaseCase):
    """Cover the pairing of 1Q and 2Q designs of differing per-L lengths."""

    def _run(self, oneq_len, twoq_len, seed=0):
        oneq_lists = [_make_1q_circuits(oneq_len)]
        twoq_lists = [_make_2q_circuits(twoq_len)]
        oneq = _StubDesign(oneq_lists, (0,))
        twoq = _StubDesign(twoq_lists, (0, 1))

        # A single color patch: one 2Q edge (0, 1) plus one unused 1Q qubit (2),
        # so both the edge slot and the unused-qubit slot are exercised.
        color_patches = {0: [(0, 1)]}
        vertices = [0, 1, 2]

        circuit_lists = assign_the_designs_with_mapping(
            oneq, twoq, vertices, color_patches,
            randgen=np.random.default_rng(seed),
        )
        # assign_the_designs_with_mapping no longer verifies its own output;
        # that's now stitcher-agnostic and lives in
        # assert_circuit_lists_match_color_patches (normally invoked by
        # SimultaneousGSTDesign.__init__). Run it explicitly here
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


class DefaultOutputIsStableTester(BaseCase):
    """
    Pin the default stitcher's output for fixed seeds.

    ``share_same_shape_schedules`` was added as an option specifically so the
    default behaviour would not move. These hashes were captured *before* that
    option existed; if a refactor perturbs the order in which the random stream
    is consumed, they change and this test fails loudly rather than silently
    handing users a different (still valid-looking) experiment design.

    Regenerating these is legitimate when the randomization is intentionally
    changed -- but it should be a deliberate, reviewed act.
    """

    EXPECTED = {
        'one_patch':
            '122cb38a0e3bc5ea97c6723634c117ea651523605947eede2ee30f9c825cb145',
        'two_same_shape_patches':
            '9007108ae3377766df08406bb2ca04b77cdc00204d288e7427abd04622661f5c',
        'mixed_shape_patches':
            'efd013af0b64e8fb4cbd0fd9fb86b20774b6d4a92e4372d53e0279a2fe259e0e',
    }

    CASES = {
        'one_patch': ([0, 1, 2, 3, 4, 5], {0: [(0, 1)]}),
        'two_same_shape_patches':
            ([0, 1, 2, 3, 4, 5], {0: [(0, 1), (2, 3)], 1: [(1, 2), (3, 4)]}),
        'mixed_shape_patches':
            ([0, 1, 2, 3, 4, 5], {0: [(0, 1), (2, 3)], 1: [(1, 2)]}),
    }

    def test_default_output_matches_recorded_hashes(self):
        oneq = _StubDesign([_make_1q_circuits(n) for n in (3, 5)], (0,))
        twoq = _StubDesign([_make_2q_circuits(n) for n in (3, 5)], (0, 1))

        for name, (vertices, color_patches) in self.CASES.items():
            with self.subTest(case=name):
                circuit_lists = assign_the_designs_with_mapping(
                    oneq, twoq, vertices, color_patches,
                    randgen=np.random.default_rng(12345),
                )
                blob = json.dumps([[c.str for c in L] for L in circuit_lists])
                digest = hashlib.sha256(blob.encode()).hexdigest()
                self.assertEqual(digest, self.EXPECTED[name])


class MultiplePatchesSameShapeTester(BaseCase):
    """
    Two color patches sharing the same shape (here, both have 1 edge and 1 unused
    qubit) are grouped together and share a randomly-generated "template" circuit;
    all but the first ("representative") patch in the group must have that template
    relabeled (via `Circuit.map_state_space_labels`) onto their own lines.

    This is the ``share_same_shape_schedules=True`` contract, i.e. the default.
    See ``IndependentSchedulesTester`` for the opposite setting.
    """

    def _run(self, oneq_len=4, twoq_len=4, seed=0):
        oneq_lists = [_make_1q_circuits(oneq_len)]
        twoq_lists = [_make_2q_circuits(twoq_len)]
        oneq = _StubDesign(oneq_lists, (0,))
        twoq = _StubDesign(twoq_lists, (0, 1))

        # A 3-qubit line with two color patches, both of shape (1 edge, 1 unused
        # qubit): patch 0 is edge (0, 1) with qubit 2 left over; patch 1 is edge
        # (1, 2) with qubit 0 left over. Both patches land in the same `groups`
        # bucket, which is exactly the scenario the bug above hits.
        color_patches = {0: [(0, 1)], 1: [(1, 2)]}
        vertices = [0, 1, 2]

        circuit_lists = assign_the_designs_with_mapping(
            oneq, twoq, vertices, color_patches,
            randgen=np.random.default_rng(seed),
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

    def test_patch1_is_exactly_patch0_relabeled(self):
        # The sharing contract in full: patch 1 is not merely "similar" to patch
        # 0, it is patch 0 pushed through the patch-0 -> patch-1 line mapper,
        # circuit for circuit and slot for slot. This is what
        # share_same_shape_schedules=False switches off.
        vertices = [0, 1, 2]
        patch_infos = build_patch_infos(vertices, {0: [(0, 1)], 1: [(1, 2)]})
        mapper = make_line_mapper(
            patch_infos[0]["tensored_lines"],
            patch_infos[1]["tensored_lines"],
        )

        patch0_circuits, patch1_circuits = self._run()
        self.assertEqual(len(patch0_circuits), len(patch1_circuits))
        for c0, c1 in zip(patch0_circuits, patch1_circuits):
            self.assertEqual(c0.map_state_space_labels(mapper), c1)


class IndependentSchedulesTester(BaseCase):
    """
    With ``share_same_shape_schedules=False`` every patch is its own scheduling
    group, so same-shape patches draw independently instead of one being a
    relabelling of the other.

    Everything else -- patch-major ordering, correct stitching onto each patch's
    own qubits, nesting, no duplicates, seed reproducibility -- must still hold.
    """

    # Two same-shape patches (2 edges + 2 unused qubits each) on 6 qubits, which
    # under the default would collapse into a single scheduling group.
    VERTICES = [0, 1, 2, 3, 4, 5]
    COLOR_PATCHES = {0: [(0, 1), (2, 3)], 1: [(1, 2), (3, 4)]}

    def _run(self, share, seed=0, oneq_lens=(3, 6), twoq_lens=(3, 6)):
        oneq = _StubDesign([_make_1q_circuits(n) for n in oneq_lens], (0,))
        twoq = _StubDesign([_make_2q_circuits(n) for n in twoq_lens], (0, 1))
        return assign_the_designs_with_mapping(
            oneq, twoq, self.VERTICES, self.COLOR_PATCHES,
            randgen=np.random.default_rng(seed),
            share_same_shape_schedules=share,
        )

    def _patch_chunks(self, germ_power_list):
        half = len(germ_power_list) // 2
        return germ_power_list[:half], germ_power_list[half:]

    def test_shared_schedules_make_patch1_a_relabeling_of_patch0(self):
        # Control for the test below: under the default, the two patches *are*
        # related by the line mapper.
        patch_infos = build_patch_infos(self.VERTICES, self.COLOR_PATCHES)
        mapper = make_line_mapper(
            patch_infos[0]["tensored_lines"],
            patch_infos[1]["tensored_lines"],
        )
        patch0, patch1 = self._patch_chunks(self._run(share=True)[0])
        for c0, c1 in zip(patch0, patch1):
            self.assertEqual(c0.map_state_space_labels(mapper), c1)

    def test_independent_schedules_break_the_relabeling(self):
        patch_infos = build_patch_infos(self.VERTICES, self.COLOR_PATCHES)
        mapper = make_line_mapper(
            patch_infos[0]["tensored_lines"],
            patch_infos[1]["tensored_lines"],
        )
        patch0, patch1 = self._patch_chunks(self._run(share=False)[0])
        self.assertTrue(
            any(c0.map_state_space_labels(mapper) != c1
                for c0, c1 in zip(patch0, patch1)),
            "share_same_shape_schedules=False still produced patch 1 as an "
            "exact relabelling of patch 0, i.e. the schedules were shared."
        )

    def test_independent_schedules_still_satisfy_the_output_contract(self):
        circuit_lists = self._run(share=False)
        assert_circuit_lists_match_color_patches(
            circuit_lists, self.VERTICES, self.COLOR_PATCHES
        )
        for L, germ_power_list in enumerate(circuit_lists):
            self.assertEqual(
                len(germ_power_list), len(set(germ_power_list)),
                f"germ-power {L} contains duplicated circuits."
            )
        # Still nested.
        for earlier, later in zip(circuit_lists, circuit_lists[1:]):
            self.assertTrue(set(earlier).issubset(set(later)))

    def test_independent_schedules_are_seed_reproducible(self):
        self.assertEqual(self._run(share=False, seed=7),
                         self._run(share=False, seed=7))

    def test_both_settings_produce_the_same_number_of_circuits(self):
        # Sharing is a randomization choice, not a size choice.
        shared = self._run(share=True)
        independent = self._run(share=False)
        self.assertEqual([len(L) for L in shared], [len(L) for L in independent])

    def test_default_is_to_share(self):
        oneq = _StubDesign([_make_1q_circuits(3)], (0,))
        twoq = _StubDesign([_make_2q_circuits(3)], (0, 1))
        default = assign_the_designs_with_mapping(
            oneq, twoq, self.VERTICES, self.COLOR_PATCHES,
            randgen=np.random.default_rng(0),
        )
        explicit = assign_the_designs_with_mapping(
            oneq, twoq, self.VERTICES, self.COLOR_PATCHES,
            randgen=np.random.default_rng(0),
            share_same_shape_schedules=True,
        )
        self.assertEqual(default, explicit)


class NestingTester(BaseCase):
    """
    The stitcher's output is always nested: the circuit list for a higher
    germ-power index must contain every circuit generated for any lower
    germ-power index.

    Nesting is applied patch-wise so the germ-power-major-then-patch-major
    ordering survives: each patch's own chunk at germ power ``L`` must start
    with the exact same circuits, in the same order, as that patch's chunk at
    every germ power ``L' < L``. (Concatenating whole germ-power lists instead
    would interleave patches and break that contract.)
    """

    def _run(self, oneq_lens, twoq_lens, color_patches, vertices, seed=0):
        oneq_lists = [_make_1q_circuits(n) for n in oneq_lens]
        twoq_lists = [_make_2q_circuits(n) for n in twoq_lens]
        oneq = _StubDesign(oneq_lists, (0,))
        twoq = _StubDesign(twoq_lists, (0, 1))

        circuit_lists = assign_the_designs_with_mapping(
            oneq, twoq, vertices, color_patches,
            randgen=np.random.default_rng(seed),
        )
        assert_circuit_lists_match_color_patches(circuit_lists, vertices, color_patches)
        return circuit_lists

    def _assert_no_duplicates(self, circuit_lists):
        # Nesting must not double-count: each germ power's list is a set of
        # distinct circuits, not the earlier lists pasted on twice.
        for L, germ_power_list in enumerate(circuit_lists):
            self.assertEqual(
                len(germ_power_list), len(set(germ_power_list)),
                f"germ-power {L} contains duplicated circuits "
                f"({len(germ_power_list)} entries, "
                f"{len(set(germ_power_list))} distinct)."
            )

    def _assert_patchwise_containment(self, circuit_lists, vertices, color_patches):
        patch_infos = build_patch_infos(vertices, color_patches)
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
        self._assert_no_duplicates(circuit_lists)

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
        self._assert_no_duplicates(circuit_lists)

    def test_three_germ_powers_containment_holds_for_all_pairs(self):
        vertices = [0, 1, 2]
        color_patches = {0: [(0, 1)]}
        circuit_lists = self._run(
            oneq_lens=[2, 4, 7], twoq_lens=[2, 4, 7],
            color_patches=color_patches, vertices=vertices,
        )
        self.assertEqual(len(circuit_lists), 3)
        self._assert_patchwise_containment(circuit_lists, vertices, color_patches)
        self._assert_no_duplicates(circuit_lists)

    def test_multiple_patches_and_germ_powers_stay_patch_major(self):
        # Regression: renesting by concatenating whole germ-power lists yields
        # [P0 P1][P0 P1], so patch 0's chunk would contain patch 1's circuits
        # (and their Gcnot on patch 1's edge). Renesting patch-wise keeps
        # [all of P0][all of P1].
        vertices = [0, 1, 2]
        color_patches = {0: [(0, 1)], 1: [(1, 2)]}
        circuit_lists = self._run(
            oneq_lens=[3, 6, 10], twoq_lens=[3, 6, 10],
            color_patches=color_patches, vertices=vertices,
        )
        self._assert_patchwise_containment(circuit_lists, vertices, color_patches)
        self._assert_no_duplicates(circuit_lists)

        for germ_power_list in circuit_lists:
            half = len(germ_power_list) // 2
            for c in germ_power_list[:half]:
                self.assertEqual(MultiplePatchesSameShapeTester._cnot_edges(c), {(0, 1)})
            for c in germ_power_list[half:]:
                self.assertEqual(MultiplePatchesSameShapeTester._cnot_edges(c), {(1, 2)})


def _line_pspec(n_qubits=3):
    """An `n_qubits`-qubit line processor spec usable with the SGST design.

    "Gii" (the primitive two-qubit idle) is required on every 2Q edge because
    ``build_layer_mappers`` maps a 2Q lane's implicit-idle layer onto
    ``Label(('Gii',) + twoq_qubit_labels)``. It is not a pyGSTi standard gate
    name, hence the explicit 4x4 identity unitary.
    """
    qubits = tuple(range(n_qubits))
    line_edges = [(q, q + 1) for q in range(n_qubits - 1)]
    oneq_locations = [(q,) for q in qubits]
    availability = {
        'Gi': oneq_locations,
        'Gxpi2': oneq_locations,
        'Gypi2': oneq_locations,
        'Gcnot': line_edges,
        'Gii': line_edges,
    }
    pspec = QubitProcessorSpec(
        n_qubits, gate_names=['Gi', 'Gxpi2', 'Gypi2', 'Gcnot', 'Gii'],
        nonstd_gate_unitaries={'Gii': np.eye(4)},
        availability=availability, qubit_labels=qubits,
    )
    return pspec, qubits, line_edges


def _make_designs(max_max_length=1):
    """Small 1Q and 2Q GST designs; depth kept minimal to keep the test fast."""
    oneq = smq1Q_XYI.create_gst_experiment_design(
        max_max_length=max_max_length, qubit_labels=(0,))
    twoq = smq2Q_XYICNOT.create_gst_experiment_design(
        max_max_length=max_max_length, qubit_labels=(0, 1))
    return oneq, twoq


class _SGSTFixture:
    """A 3-qubit line device and the simultaneous-GST design built on it.

    Class-scoped, so a tester that might mutate the design must copy it first (see
    ``UnsupportedOperationsTester._fresh_design``). Mixed in ahead of ``BaseCase`` so
    that subclasses adding their own setup can chain via ``super().setUpClass()``.
    """

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.pspec, cls.qubits, cls.line_edges = _line_pspec(3)
        cls.oneq, cls.twoq = _make_designs()
        cls.design = make_simultaneous_gst_design(cls.pspec, cls.oneq, cls.twoq, seed=0)


class MakeSimultaneousGSTDesignTester(_SGSTFixture, BaseCase):
    """
    Cover ``make_simultaneous_gst_design``, the public convenience entry point.

    It derives the graph (vertices, edges, neighbors, max degree) from the
    processor spec, computes an edge coloring with the 'auto' algorithm, and
    forwards everything to ``SimultaneousGSTDesign`` with the second of two
    streams spawned off ``seed``. These tests check that each of those derived
    values lands on the returned design correctly.
    """

    def test_returns_crosstalk_free_design_with_inputs_passed_through(self):
        self.assertIsInstance(self.design, SimultaneousGSTDesign)
        self.assertIs(self.design.processor_spec, self.pspec)
        self.assertIs(self.design.oneq_gstdesign, self.oneq)
        self.assertIs(self.design.twoq_gstdesign, self.twoq)

    def test_graph_derived_from_processor_spec(self):
        self.assertEqual(self.design.vertices, self.qubits)
        self.assertEqual(sorted(self.design.edges), sorted(self.line_edges))
        # qubit_labels on the resulting GST design are the full n-qubit register,
        # not the 1Q/2Q sub-design labels.
        self.assertEqual(self.design.qubit_labels, self.qubits)

    def test_defaults_forwarded_to_experiment_design(self):
        # make_simultaneous_gst_design passes neither circuit_stitcher nor nested, so both
        # must land on their SimultaneousGSTDesign defaults. The default stitcher always
        # renests its output, so nested defaults to True.
        self.assertIs(self.design.circuit_stitcher, assign_the_designs_with_mapping)
        self.assertTrue(self.design.nested)
        self.assertEqual(list(self.design.stitcher_kwargs), ['randgen', 'verbosity'])
        # verbosity defaults to 0, i.e. no progress bar from the stitcher.
        self.assertEqual(self.design.stitcher_kwargs['verbosity'], 0)

    def test_edge_coloring_is_a_valid_proper_coloring(self):
        color_patches = self.design.color_patches
        self.assertIsInstance(color_patches, dict)

        # Every edge is coloured exactly once...
        coloured = [tuple(e) for edges in color_patches.values() for e in edges]
        self.assertEqual(sorted(coloured), sorted(self.line_edges))
        self.assertEqual(len(coloured), len(set(coloured)))

        # ...and each colour class is a matching, i.e. its edges are pairwise disjoint.
        for color, edges in color_patches.items():
            qubits_in_patch = [q for e in edges for q in e]
            self.assertEqual(
                len(qubits_in_patch), len(set(qubits_in_patch)),
                msg=f"colour {color} is not a matching: {edges}")

    def test_circuit_lists_match_the_color_patches(self):
        self.assertEqual(len(self.design.circuit_lists), len(self.oneq.circuit_lists))
        self.assertGreater(len(self.design.all_circuits_needing_data), 0)
        # Stitcher-agnostic structural check: no implicit idles, and every
        # circuit sits on its own patch's qubits/edges.
        assert_circuit_lists_match_color_patches(
            self.design.circuit_lists, self.design.vertices, self.design.color_patches)

    def test_stitcher_gets_the_second_spawned_seed(self):
        # make_simultaneous_gst_design splits `seed` into two independent streams
        # with SeedSequence.spawn(2) and hands the second to the design;
        # constructing the design directly with that stream and the same coloring
        # must reproduce it exactly.
        #
        # The design is rebuilt here rather than reusing cls.design on purpose:
        # coverage contexts attribute setUpClass code to whichever test happened
        # to trigger class setup, so a test that only *reads* cls.design is not
        # recorded as covering the spawn expression and is therefore never
        # selected by the diff-mutation tooling to guard it.
        design = make_simultaneous_gst_design(self.pspec, self.oneq, self.twoq, seed=0)
        _, stitcher_seed = np.random.SeedSequence(0).spawn(2)
        direct = SimultaneousGSTDesign(
            self.pspec, self.oneq, self.twoq, design.color_patches, seed=stitcher_seed)
        self.assertEqual(direct.circuit_lists, design.circuit_lists)

    def test_stitcher_stream_is_not_the_coloring_stream(self):
        # The point of spawning is that the two consumers never draw the same
        # sequence, so seeding the stitcher with the *coloring's* stream must not
        # reproduce the design. Guards against a collapse back to a single shared
        # seed (or to an off-by-one offset, which recycles one call's stitcher
        # stream as the next call's coloring stream).
        coloring_seed, _ = np.random.SeedSequence(0).spawn(2)
        direct = SimultaneousGSTDesign(
            self.pspec, self.oneq, self.twoq, self.design.color_patches, seed=coloring_seed)
        self.assertNotEqual(direct.circuit_lists, self.design.circuit_lists)

    def test_different_seeds_give_different_circuit_assignments(self):
        # Guards the seed actually reaching the stitcher's randgen: on a line the
        # coloring is seed-independent, so any difference comes from the seed.
        other = make_simultaneous_gst_design(self.pspec, self.oneq, self.twoq, seed=7)
        self.assertEqual(other.color_patches, self.design.color_patches)
        self.assertNotEqual(other.circuit_lists, self.design.circuit_lists)


class HelperRejectsMalformedInputTester(_SGSTFixture, BaseCase):
    """
    Cover the *detection power* of the verification helpers.

    Every other test in this file hands the helpers well-formed data, which
    only ever proves they accept what they should. That leaves their whole
    reason for existing -- catching a ``circuit_stitcher`` that silently
    produces a bad stitching -- unverified: weakening the checks (skipping
    them, narrowing their conditions, or iterating fewer patches) is invisible
    when nothing malformed is ever passed in.

    These tests feed the helpers deliberately malformed input and require them
    to raise, and check that ``debug_check`` really does wire
    ``assert_circuit_lists_match_color_patches`` into
    ``SimultaneousGSTDesign.__init__``.
    """

    # -- debug_check wiring in SimultaneousGSTDesign.__init__ ------

    @staticmethod
    def _malformed_stitcher(oneq_gstdesign, twoq_gstdesign, vertices,
                            color_patches, **kwargs):
        """A stitcher returning output that cannot be a valid stitching.

        One circuit cannot split evenly into the two color patches below. The
        inputs are ignored so the (slow) real stitching is never run.
        """
        return [[Circuit([Label('Gcnot', (0, 1))], line_labels=(0, 1, 2))]]

    def _build_with_malformed_stitcher(self, **kwargs):
        return SimultaneousGSTDesign(
            self.pspec, self.oneq, self.twoq, {0: [(0, 1)], 1: [(1, 2)]},
            circuit_stitcher=self._malformed_stitcher, **kwargs)

    def test_malformed_stitcher_output_is_rejected_when_debug_check_true(self):
        with self.assertRaises(AssertionError) as ctx:
            self._build_with_malformed_stitcher(debug_check=True)
        self.assertIn('split evenly', str(ctx.exception))

    def test_debug_check_defaults_to_true(self):
        # The self-check is the documented safety net for swapped-in stitchers,
        # so it must be on unless explicitly disabled.
        with self.assertRaises(AssertionError):
            self._build_with_malformed_stitcher()

    def test_malformed_stitcher_output_is_accepted_when_debug_check_false(self):
        design = self._build_with_malformed_stitcher(debug_check=False)
        # The malformed lists must land unchanged: debug_check switches the
        # verification off, it does not repair anything.
        self.assertEqual(
            [list(cl) for cl in design.circuit_lists],
            [[Circuit([Label('Gcnot', (0, 1))], line_labels=(0, 1, 2))]])

    # -- detection power of the helpers themselves -------------------------

    def test_all_patches_are_verified_not_just_the_first(self):
        """Corrupting a *later* patch's chunk must still be caught.

        The patch chunks are checked in a loop, so an off-by-one in the slice
        bounds can leave every patch after the first silently unverified.
        """
        color_patches = self.design.color_patches
        num_patches = len(color_patches)
        self.assertGreaterEqual(
            num_patches, 2, msg="need >1 patch for this test to mean anything")

        # Overwrite the second chunk with a copy of the first, so the circuits
        # sit on patch 0's edge while occupying patch 1's chunk. Total length is
        # unchanged, so the even-split assertion cannot fire first and mask this.
        corrupted = []
        for circuit_list in self.design.circuit_lists:
            chunk = len(circuit_list) // num_patches
            corrupted.append(list(circuit_list[:chunk]) + list(circuit_list[:chunk])
                             + list(circuit_list[2 * chunk:]))
        self.assertEqual([len(cl) for cl in corrupted],
                         [len(cl) for cl in self.design.circuit_lists])

        with self.assertRaises(AssertionError) as ctx:
            assert_circuit_lists_match_color_patches(
                corrupted, self.design.vertices, color_patches)
        self.assertIn('Patch 1', str(ctx.exception))

    def test_multiqubit_gate_outside_patch_edges_is_rejected(self):
        """A 2Q gate on an edge the patch does not own must be caught."""
        patch_infos = build_patch_infos([0, 1, 2], {0: [(0, 1)]})
        info = patch_infos[0]
        expected_labels = {q for line in info['tensored_lines'] for q in line}

        # Gcnot sits on (1, 2); the patch owns only (0, 1). The line labels are
        # deliberately correct so the earlier line-label assertion passes and
        # the edge check is what actually fires.
        bad_circuit = Circuit([Label('Gcnot', (1, 2))],
                              line_labels=tuple(sorted(expected_labels)))
        self.assertEqual(set(bad_circuit.line_labels), expected_labels)

        with self.assertRaises(AssertionError) as ctx:
            assert_mapped_circuit_matches_patch(bad_circuit, info)
        self.assertIn("not one of this patch's own edges", str(ctx.exception))

    def test_wellformed_input_is_accepted(self):
        """Control: the helper accepts the un-corrupted design.

        Without this, a helper that raised unconditionally would make every
        test above pass for the wrong reason.
        """
        assert_circuit_lists_match_color_patches(
            self.design.circuit_lists, self.design.vertices,
            self.design.color_patches)


class EdgeNormalizationTester(_SGSTFixture, BaseCase):
    """
    Cover ``_normalize_coloring`` and the invariant it establishes: an edge in
    ``color_patches`` is always a tuple.

    Nothing downstream *requires* tuples -- ``patch_lines`` casts each edge before use --
    so a coloring of lists produces byte-identical circuits and passes every internal
    check. The invariant exists for callers: it makes two designs that were built
    differently compare equal, and it keeps edges hashable. Neither is caught by any
    circuit-level assertion, so both are tested directly.
    """

    #: The same coloring spelled two ways. JSON round trips produce the list form,
    #: and a caller building a coloring by hand easily might too.
    LIST_COLORING = {0: [[0, 1]], 1: [[1, 2]]}
    TUPLE_COLORING = {0: [(0, 1)], 1: [(1, 2)]}

    def _design(self, coloring):
        return SimultaneousGSTDesign(self.pspec, self.oneq, self.twoq, coloring, seed=0)

    def test_list_built_design_matches_tuple_built_design(self):
        """The two spellings must be indistinguishable, coloring included.

        The circuits already matched before normalization; ``color_patches`` did not, so
        a design's equality depended on how its coloring happened to be spelled.
        """
        from_lists, from_tuples = self._design(self.LIST_COLORING), self._design(self.TUPLE_COLORING)
        self.assertEqual(from_lists.color_patches, self.TUPLE_COLORING)
        self.assertEqual(from_lists.color_patches, from_tuples.color_patches)
        self.assertEqual([list(cl) for cl in from_lists.circuit_lists],
                         [list(cl) for cl in from_tuples.circuit_lists])
        # Lists are unhashable, so this raised TypeError for a list-built design.
        self.assertEqual(set(from_lists.color_patches[0]), {(0, 1)})

    def test_orientation_is_preserved_not_canonicalized(self):
        """Guards against 'improving' the helper into ``order``/``canonical_edges``.

        Orientation picks which lane of the 2Q design a qubit lands in, so rewriting
        (1, 0) to (0, 1) would silently change the experiment while leaving every
        structural check happy.
        """
        self.assertEqual(self._design({0: [(1, 0)]}).color_patches, {0: [(1, 0)]})

    def test_does_not_mutate_the_callers_coloring(self):
        coloring = {0: [[0, 1]], 1: [[1, 2]]}
        self._design(coloring)
        self.assertEqual(coloring, {0: [[0, 1]], 1: [[1, 2]]})

    def test_normalize_coloring(self):
        """The helper alone: tuple-ness, and everything it must leave alone."""
        for name, coloring, expected in (
            ('already tuples', self.TUPLE_COLORING, self.TUPLE_COLORING),
            ('patch keys and edge order kept', {2: [[3, 4], [0, 1]], 0: [[1, 2]]},
             {2: [(3, 4), (0, 1)], 0: [(1, 2)]}),
            ('empty patch survives', {0: [], 1: [[0, 1]]}, {0: [], 1: [(0, 1)]}),
        ):
            with self.subTest(name):
                normalized = _normalize_coloring(coloring)
                self.assertEqual(normalized, expected)
                self.assertEqual(list(normalized), list(expected))  # key order


class MapQubitLabelsTester(_SGSTFixture, BaseCase):
    """
    Cover ``SimultaneousGSTDesign.map_qubit_labels``.

    The inherited version returns a plain ``GateSetTomographyDesign``, silently dropping
    the edge coloring and the 1Q/2Q sub-designs -- a downgrade rather than a crash, so
    nothing catches it but a test that looks.

    Two behaviours are easy to "fix" into being wrong, so each gets its own test: the
    sub-designs must *not* be relabelled (they live on abstract lane labels, not device
    qubits), and the circuits must be relabelled rather than re-stitched (which would
    redraw the stitcher's random schedules and return a different experiment).
    """

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        #: Shifts the device off the lane labels (0, 1, 2) entirely, so a mapper
        #: wrongly applied to the sub-designs shows up as changed lane labels.
        cls.mapper = {q: 'Q%d' % q for q in cls.design.vertices}
        cls.mapped = cls.design.map_qubit_labels(cls.mapper)

    # -- the returned object -------------------------------------------------

    def test_returns_a_simultaneous_gst_design(self):
        # The point of the override: the inherited one returns a GateSetTomographyDesign.
        self.assertIsInstance(self.mapped, SimultaneousGSTDesign)

    def test_processor_spec_and_qubit_labels_are_mapped(self):
        self.assertEqual(self.mapped.processor_spec.qubit_labels, ('Q0', 'Q1', 'Q2'))
        self.assertEqual(self.mapped.vertices, ('Q0', 'Q1', 'Q2'))
        self.assertEqual(self.mapped.qubit_labels, ('Q0', 'Q1', 'Q2'))

    def test_color_patches_are_mapped(self):
        # Otherwise the design describes patches over qubits that no longer exist.
        self.assertEqual(
            self.mapped.color_patches,
            {patch: [tuple('Q%d' % q for q in edge) for edge in edge_set]
             for patch, edge_set in self.design.color_patches.items()})

    def test_mapped_design_passes_its_own_validator(self):
        # The relabelled circuits must still sit on the relabelled patches.
        assert_circuit_lists_match_color_patches(
            self.mapped.circuit_lists, self.mapped.vertices,
            self.mapped.color_patches)

    def test_nested_flag_and_list_shape_are_preserved(self):
        # Relabelling is a bijection, so a shape change here means something was rebuilt.
        self.assertEqual(self.mapped.nested, self.design.nested)
        self.assertEqual([len(cl) for cl in self.mapped.circuit_lists],
                         [len(cl) for cl in self.design.circuit_lists])
        self.assertEqual(len(self.mapped.all_circuits_needing_data),
                         len(self.design.all_circuits_needing_data))

    def test_derived_graph_state_is_recomputed_from_the_mapped_pspec(self):
        # edges/neighbors/deg are re-derived, not mapped: same graph, new names.
        self.assertEqual(sorted(self.mapped.edges), [('Q0', 'Q1'), ('Q1', 'Q2')])
        self.assertEqual(self.mapped.deg, self.design.deg)
        self.assertEqual({v: sorted(ns) for v, ns in self.mapped.neighbors.items()},
                         {'Q0': ['Q1'], 'Q1': ['Q0', 'Q2'], 'Q2': ['Q1']})

    # -- circuits are relabelled, not rebuilt --------------------------------

    def test_circuits_are_relabelled_one_for_one(self):
        expected = [[c.map_state_space_labels(self.mapper) for c in circuit_list]
                    for circuit_list in self.design.circuit_lists]
        self.assertEqual([list(cl) for cl in self.mapped.circuit_lists], expected)

    def test_mapping_back_recovers_the_original_circuits(self):
        # The sharpest statement that no circuit *content* changed: a re-stitched design
        # would draw fresh schedules and differ despite carrying the right line labels.
        inverse = {new: old for old, new in self.mapper.items()}
        roundtripped = self.mapped.map_qubit_labels(inverse)
        self.assertEqual([list(cl) for cl in roundtripped.circuit_lists],
                         [list(cl) for cl in self.design.circuit_lists])

    def test_the_circuit_stitcher_is_never_called(self):
        # Direct version of the above: the tempting "just call __init__" is wrong.
        def _explode(*args, **kwargs):
            raise AssertionError("circuit_stitcher must not be re-run when relabelling")

        original = self.design.circuit_stitcher
        self.design.circuit_stitcher = _explode
        try:
            mapped = self.design.map_qubit_labels(self.mapper)
        finally:
            self.design.circuit_stitcher = original
        self.assertEqual([list(cl) for cl in mapped.circuit_lists],
                         [list(cl) for cl in self.mapped.circuit_lists])

    # -- the sub-designs are deliberately left alone -------------------------

    def test_sub_designs_are_carried_over_unmapped(self):
        # These live on abstract lane labels ((0,) and (0, 1)), not device qubits, so a
        # device mapper would corrupt them -- or KeyError. Pass through untouched.
        self.assertIs(self.mapped.oneq_gstdesign, self.design.oneq_gstdesign)
        self.assertIs(self.mapped.twoq_gstdesign, self.design.twoq_gstdesign)
        self.assertEqual(self.mapped.oneq_gstdesign.qubit_labels, (0,))
        self.assertEqual(self.mapped.twoq_gstdesign.qubit_labels, (0, 1))

    # -- mapper forms, immutability, and the self-check ----------------------

    def test_callable_mapper_matches_dict_mapper(self):
        # `mapper` is documented as "dict or function"; the two spellings must agree.
        via_callable = self.design.map_qubit_labels(lambda q: 'Q%d' % q)
        self.assertEqual([list(cl) for cl in via_callable.circuit_lists],
                         [list(cl) for cl in self.mapped.circuit_lists])
        self.assertEqual(via_callable.color_patches, self.mapped.color_patches)

    def test_a_permutation_of_the_existing_labels_is_handled(self):
        # Mapping onto the *same* label set is where an in-place implementation would
        # corrupt the design mid-map, and where edges must be re-canonicalized.
        reversal = {q: 2 - q for q in self.design.vertices}
        mapped = self.design.map_qubit_labels(reversal)
        self.assertEqual(mapped.vertices, (2, 1, 0))
        self.assertEqual(sorted(mapped.edges), [(0, 1), (1, 2)])
        assert_circuit_lists_match_color_patches(
            mapped.circuit_lists, mapped.vertices, mapped.color_patches)

    def test_original_design_is_not_mutated(self):
        # setUpClass shares `mapped`, so a mutating implementation corrupts every test.
        self.assertEqual(self.design.vertices, (0, 1, 2))
        self.assertEqual(self.design.qubit_labels, (0, 1, 2))
        self.assertEqual(self.design.processor_spec.qubit_labels, (0, 1, 2))
        self.assertEqual(self.design.color_patches, {0: [(0, 1)], 1: [(1, 2)]})
        self.assertEqual(self.design.circuit_lists[-1][0].line_labels, (0, 1, 2))

    def test_debug_check_rejects_a_design_that_was_already_malformed(self):
        # A design built with debug_check=False cannot launder itself clean by relabelling.
        malformed = SimultaneousGSTDesign(
            self.pspec, self.oneq, self.twoq, {0: [(0, 1)], 1: [(1, 2)]},
            circuit_stitcher=(lambda *a, **kw: [[Circuit([Label('Gcnot', (0, 1))],
                                                         line_labels=(0, 1, 2))]]),
            debug_check=False)

        with self.assertRaises(AssertionError) as ctx:
            malformed.map_qubit_labels(self.mapper)
        self.assertIn('split evenly', str(ctx.exception))

        # ...and debug_check=False switches that verification off here too.
        relabelled = malformed.map_qubit_labels(self.mapper, debug_check=False)
        self.assertEqual([list(cl) for cl in relabelled.circuit_lists],
                         [[Circuit([Label('Gcnot', ('Q0', 'Q1'))],
                                   line_labels=('Q0', 'Q1', 'Q2'))]])


class _CallableStitcher:
    """A stitcher supplied as a callable object rather than a function.

    Instances carry ``__module__`` (inherited from the class) but no
    ``__qualname__``, which is the case :func:`_stitcher_name` has to reject.
    """
    def __call__(self, *args, **kwargs):
        return _STUB_CIRCUIT_LISTS


class StitcherNameTester(BaseCase):
    """
    Cover ``_stitcher_name`` / ``_resolve_stitcher_name``, the write/read pair that
    stands in for the un-serializable ``circuit_stitcher``.

    A stitcher is an arbitrary caller-supplied callable, so it is recorded by qualified
    name and imported back. Not every callable has one: ``functools.partial`` objects and
    callable class instances have a ``__module__`` but no ``__qualname__``. Formatting the
    two together would yield a bogus name like ``'functools.None'`` -- harmless, since it
    fails to import into the same "None plus a warning" state as no name at all, but
    misleading in meta.json, so the helper reports None instead.
    """

    def test_plain_function_round_trips(self):
        name = _stitcher_name(assign_the_designs_with_mapping)
        self.assertEqual(name, 'pygsti.protocols.simultaneous_gst.assign_the_designs_with_mapping')
        with self.assertNoWarns():
            self.assertIs(_resolve_stitcher_name(name), assign_the_designs_with_mapping)

    def test_lambda_is_named_but_not_importable(self):
        # A lambda does have a qualname, so it is recorded -- it just cannot be
        # imported back. The two failure modes are distinct and both end at None.
        name = _stitcher_name(lambda *args, **kwargs: None)
        self.assertIsNotNone(name)
        self.assertIn('<lambda>', name)
        with self.assertWarns(Warning):
            self.assertIsNone(_resolve_stitcher_name(name))

    def test_missing_name_resolves_to_none_with_a_warning(self):
        # What a partial or callable instance produces on the way back in.
        with self.assertWarns(Warning):
            self.assertIsNone(_resolve_stitcher_name(None))

    def test_unimportable_module_resolves_to_none_with_a_warning(self):
        with self.assertWarns(Warning):
            self.assertIsNone(_resolve_stitcher_name('no_such_module_xyz.some_stitcher'))

    def test_importable_module_missing_the_member_resolves_to_none(self):
        # The module resolves but the attribute does not -- an AttributeError rather
        # than an ImportError, and the only case that reaches that arm of the except
        # clause. It is what a design written by an older/newer pyGSTi looks like if the
        # stitcher it names has since been renamed or removed.
        with self.assertWarns(Warning):
            self.assertIsNone(_resolve_stitcher_name(
                'pygsti.protocols.simultaneous_gst.no_such_stitcher'))

    def test_method_of_a_class_is_named_but_not_restorable(self):
        # rpartition splits at the last dot, so a method's class ends up inside the
        # module name and the import fails. Recorded for provenance, not restored.
        name = _stitcher_name(_CallableStitcher.__call__)
        self.assertIn('_CallableStitcher.__call__', name)
        with self.assertWarns(Warning):
            self.assertIsNone(_resolve_stitcher_name(name))

    def test_bare_name_with_no_module_resolves_to_none_with_a_warning(self):
        # rpartition('.') yields an empty module name, which import_module rejects with
        # ValueError rather than ImportError -- a distinct path through the except clause.
        with self.assertWarns(Warning):
            self.assertIsNone(_resolve_stitcher_name('stitcher_with_no_module'))


class SerializationTester(_SGSTFixture, BaseCase):
    """
    Cover ``SimultaneousGSTDesign.write`` / ``from_dir``.

    Writing used to fail outright: ``write_meta_based_dir`` puts every attribute not
    named in ``auxfile_types`` straight into 'meta.json', gated by ``_check_jsonable``,
    which rejects live objects *and* non-string dict keys -- and this design adds two
    int-keyed dicts, two sub-designs, a function and a numpy Generator. ``design.write()``
    raised ValueError, taking ``pygsti.io.write_empty_protocol_data`` down with it.

    Members that are *not* stored get reconstructed on load, so these tests check the
    reconstruction as much as the storage.
    """

    def _roundtrip(self, root_path, design=None, name='d'):
        design = self.design if design is None else design
        root = pathlib.Path(root_path) / name
        design.write(root)
        return root, SimultaneousGSTDesign.from_dir(root)

    # -- the round trip ------------------------------------------------------

    @with_temp_path
    def test_write_then_from_dir_preserves_the_design(self, root_path):
        _, loaded = self._roundtrip(root_path)
        self.assertIsInstance(loaded, SimultaneousGSTDesign)
        self.assertEqual([list(cl) for cl in loaded.circuit_lists],
                         [list(cl) for cl in self.design.circuit_lists])
        self.assertEqual(set(loaded.all_circuits_needing_data),
                         set(self.design.all_circuits_needing_data))
        self.assertEqual(loaded.nested, self.design.nested)
        self.assertEqual(loaded.qubit_labels, self.design.qubit_labels)
        self.assertEqual(loaded.processor_spec.qubit_labels,
                         self.design.processor_spec.qubit_labels)

    @with_temp_path
    def test_color_patches_survive_with_int_keys_and_tuple_edges(self, root_path):
        # Why color_patches needs 'fancykeydict:json': its keys are ints, which a JSON
        # object cannot hold, and its edges are tuples, which JSON turns into lists. The
        # isinstance checks matter -- equality alone would pass on two list-valued dicts.
        _, loaded = self._roundtrip(root_path)
        self.assertEqual(loaded.color_patches, self.design.color_patches)
        for patch, edge_set in loaded.color_patches.items():
            self.assertIsInstance(patch, int)
            for edge in edge_set:
                self.assertIsInstance(edge, tuple)

    @with_temp_path
    def test_graph_members_are_recomputed_from_the_processor_spec(self, root_path):
        # Stored as 'none' and rebuilt, so they must come back identical -- including
        # `vertices` being a tuple, which JSON would have decayed to a list.
        _, loaded = self._roundtrip(root_path)
        self.assertEqual(loaded.vertices, self.design.vertices)
        self.assertIsInstance(loaded.vertices, tuple)
        self.assertEqual(loaded.edges, self.design.edges)
        self.assertEqual(loaded.neighbors, self.design.neighbors)
        self.assertEqual(loaded.deg, self.design.deg)

    @with_temp_path
    def test_sub_designs_are_written_and_restored(self, root_path):
        root, loaded = self._roundtrip(root_path)
        # Beside 'edesign', not as TreeNode children: their circuits are never executed,
        # so ProtocolData must not try to carve a dataset out for them.
        self.assertTrue((root / 'sgst_oneq_gstdesign' / 'edesign').is_dir())
        self.assertTrue((root / 'sgst_twoq_gstdesign' / 'edesign').is_dir())
        self.assertEqual(loaded._vals, {})

        self.assertEqual(loaded.oneq_gstdesign.qubit_labels, (0,))
        self.assertEqual(loaded.twoq_gstdesign.qubit_labels, (0, 1))
        for loaded_sub, original_sub in ((loaded.oneq_gstdesign, self.oneq),
                                         (loaded.twoq_gstdesign, self.twoq)):
            self.assertEqual(set(loaded_sub.all_circuits_needing_data),
                             set(original_sub.all_circuits_needing_data))

    @with_temp_path
    def test_loaded_design_passes_its_own_validator(self, root_path):
        # Ties the restored coloring to the restored circuits: every member above can
        # look individually plausible and still not describe the others.
        _, loaded = self._roundtrip(root_path)
        assert_circuit_lists_match_color_patches(
            loaded.circuit_lists, loaded.vertices, loaded.color_patches)

    @with_temp_path
    def test_mapped_design_is_also_serializable(self, root_path):
        # map_qubit_labels calls GateSetTomographyDesign.__init__, which *resets*
        # auxfile_types -- so it must re-declare them or its result is unserializable.
        mapped = self.design.map_qubit_labels({q: 'Q%d' % q for q in self.design.vertices})
        _, loaded = self._roundtrip(root_path, design=mapped, name='mapped')
        self.assertEqual(loaded.color_patches, mapped.color_patches)
        self.assertEqual(loaded.vertices, ('Q0', 'Q1', 'Q2'))

    # -- the workflow that was broken ----------------------------------------

    @with_temp_path
    def test_write_empty_protocol_data_works(self, root_path):
        # The bug report: ProtocolData.write calls edesign.write, so the standard way of
        # emitting a design plus a blank dataset template used to raise.
        root = pathlib.Path(root_path) / 'wepd'
        pygsti.io.write_empty_protocol_data(root, self.design, clobber_ok=True)
        self.assertTrue((root / 'data' / 'dataset.txt').exists())

        data = pygsti.io.read_data_from_dir(root)
        self.assertIsInstance(data.edesign, SimultaneousGSTDesign)
        self.assertEqual(set(data.edesign.all_circuits_needing_data),
                         set(self.design.all_circuits_needing_data))

    # -- the circuit stitcher ------------------------------------------------

    def test_stitcher_name_is_recorded(self):
        self.assertEqual(self.design.circuit_stitcher_name,
                         'pygsti.protocols.simultaneous_gst.assign_the_designs_with_mapping')

    @with_temp_path
    def test_importable_stitcher_is_restored(self, root_path):
        _, loaded = self._roundtrip(root_path)
        self.assertIs(loaded.circuit_stitcher, assign_the_designs_with_mapping)

    @with_temp_path
    def test_unimportable_stitcher_loads_as_none_with_a_warning(self, root_path):
        # A lambda has no importable name. Not fatal -- the stitcher is only called from
        # __init__ -- but the design forgets how it was built, hence the warning.
        design = SimultaneousGSTDesign(
            self.pspec, self.oneq, self.twoq, {0: [(0, 1)], 1: [(1, 2)]},
            circuit_stitcher=(lambda *args, **kwargs: _STUB_CIRCUIT_LISTS),
            debug_check=False)

        root = pathlib.Path(root_path) / 'lam'
        design.write(root)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            loaded = SimultaneousGSTDesign.from_dir(root)

        self.assertIsNone(loaded.circuit_stitcher)
        self.assertTrue(any('circuit_stitcher' in str(w.message) for w in caught),
                        msg='expected a warning naming circuit_stitcher, got %s'
                            % [str(w.message) for w in caught])
        # The circuits themselves are unaffected by the stitcher being unrestorable.
        self.assertEqual([list(cl) for cl in loaded.circuit_lists],
                         [list(cl) for cl in design.circuit_lists])

    @with_temp_path
    def test_rng_is_deliberately_not_preserved(self, root_path):
        # A loaded design is a record of a generated experiment, not a recipe:
        # stitcher_kwargs holds a numpy Generator, unserializable and not stable across
        # versions. Pinned so the lossiness stays a decision, not an accident.
        _, loaded = self._roundtrip(root_path)
        self.assertEqual(loaded.stitcher_kwargs, {})
        self.assertIn('randgen', self.design.stitcher_kwargs)


class UnsupportedOperationsTester(_SGSTFixture, BaseCase):
    """
    Cover the operations ``SimultaneousGSTDesign`` refuses, and the escape hatch.

    ``circuit_lists`` is ordered germ-power-major then patch-major, with every patch
    contributing an equal contiguous chunk per germ power. The inherited truncation and
    merge methods filter or concatenate circuits without regard to that structure, so they
    produce a design that fails ``assert_circuit_lists_match_color_patches`` -- and return
    a downgraded ``CircuitListsDesign`` while doing it. Both failures are silent, which is
    why every entry point gets a test.
    """

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.some_circuits = list(cls.design.all_circuits_needing_data)[:5]

    def _fresh_design(self):
        """A private copy of the shared design, for tests that might mutate it.

        The in-place hooks below are exactly what would corrupt ``setUpClass``'s shared
        design if a refusal ever stopped working, silently poisoning a later test's
        baseline. Copying is cheaper than re-stitching.
        """
        return _copy.deepcopy(self.design)

    def _assert_refused(self, fn, named):
        """Assert `fn` refuses, and that the message names `named` specifically.

        The name matters: with a public override removed the call still reaches the
        ``_truncate_to_*_inplace`` backstop and still raises, just with that hook's
        wording. Asserting only "it raised" -- or only on the shared tail of the
        message -- would pass either way.
        """
        with self.assertRaises(NotImplementedError) as ctx:
            fn()
        message = str(ctx.exception)
        self.assertIn('%s is not supported' % named, message)
        # Every refusal must also point at the way forward, or it is just a dead end.
        self.assertIn('as_circuit_lists_design', message)

    def test_public_methods_are_refused(self):
        """Each must name *itself*, not fall through to a hook's generic wording."""
        for method, args in (('truncate_to_circuits', (self.some_circuits,)),
                             ('truncate_to_available_data', (_tiny_dataset(),)),
                             ('truncate_to_design', (self.design,)),
                             ('truncate_to_lists', ([0],)),
                             ('merge_with', (self.design,))):
            with self.subTest(method):
                self._assert_refused(
                    lambda m=method, a=args: getattr(self._fresh_design(), m)(*a), named=method)

    def test_inplace_hooks_are_refused(self):
        # The public methods above are conveniences; these hooks are what a parent
        # design reaches, so they must refuse on their own account.
        for hook, args, named in (
                ('_truncate_to_circuits_inplace', (self.some_circuits,),
                 'Truncating to a subset of circuits'),
                ('_truncate_to_design_inplace', (self.design,), 'Truncating to another design'),
                ('_truncate_to_available_data_inplace', (_tiny_dataset(),),
                 'Truncating to available data')):
            with self.subTest(hook):
                self._assert_refused(
                    lambda h=hook, a=args: getattr(self._fresh_design(), h)(*a), named=named)

    def test_truncation_from_a_parent_design_is_refused(self):
        # ExperimentDesign._truncate_to_available_data_inplace loops over its children, so
        # a nested SimultaneousGSTDesign is reachable without any direct call. Overriding
        # only the public methods would leave this path producing a broken child.
        parent = CombinedExperimentDesign({'sgst': self._fresh_design()})
        with self.assertRaises(NotImplementedError):
            parent.truncate_to_available_data(_tiny_dataset())

    def test_inplace_hooks_refuse_before_mutating_anything(self):
        """Each in-place hook must refuse *immediately*, not on the way out.

        ``CircuitListsDesign``'s versions overwrite ``self.circuit_lists`` and only then
        delegate upwards, so a hook that relied on the ``_truncate_to_circuits_inplace``
        backstop instead of refusing on its own account would leave a half-truncated
        design behind -- here, all 1814 circuits replaced by none. Each hook gets its own
        copy so that one leaking cannot hide behind another.
        """
        expected = [list(cl) for cl in self.design.circuit_lists]
        for hook, args in (('_truncate_to_circuits_inplace', (self.some_circuits,)),
                           ('_truncate_to_design_inplace', (self.design,)),
                           ('_truncate_to_available_data_inplace', (_tiny_dataset(),))):
            with self.subTest(hook=hook):
                design = self._fresh_design()
                with self.assertRaises(NotImplementedError):
                    getattr(design, hook)(*args)
                self.assertEqual([list(cl) for cl in design.circuit_lists], expected)

    # -- the escape hatch ----------------------------------------------------

    def test_as_circuit_lists_design_returns_a_plain_gst_design(self):
        plain = self.design.as_circuit_lists_design()
        self.assertIsInstance(plain, GateSetTomographyDesign)
        self.assertNotIsInstance(plain, SimultaneousGSTDesign)
        self.assertEqual([list(cl) for cl in plain.circuit_lists],
                         [list(cl) for cl in self.design.circuit_lists])
        self.assertEqual(plain.qubit_labels, self.design.qubit_labels)
        self.assertEqual(plain.nested, self.design.nested)
        # The pspec is what lets write_empty_protocol_data still work on the result.
        self.assertIs(plain.processor_spec, self.design.processor_spec)

    def test_as_circuit_lists_design_can_be_truncated(self):
        # The whole point of the hatch: it does what this class refuses to.
        truncated = self.design.as_circuit_lists_design().truncate_to_circuits(
            self.some_circuits)
        self.assertEqual(set(truncated.all_circuits_needing_data), set(self.some_circuits))

    def test_as_circuit_lists_design_does_not_alias_the_original(self):
        # It hands out its own lists, so truncating the copy cannot reach the original.
        plain = self.design.as_circuit_lists_design()
        plain.circuit_lists[0].pop()
        self.assertEqual([len(cl) for cl in self.design.circuit_lists],
                         [len(cl) + (1 if i == 0 else 0)
                          for i, cl in enumerate(plain.circuit_lists)])


#: Fixed output for a stub stitcher: two circuits so they split evenly over two patches.
_STUB_CIRCUIT_LISTS = [[Circuit([Label('Gcnot', (0, 1))], line_labels=(0, 1, 2)),
                        Circuit([Label('Gcnot', (1, 2))], line_labels=(0, 1, 2))]]


def _tiny_dataset():
    """A DataSet holding a couple of 3-qubit circuits, for truncation tests."""
    dataset = pygsti.data.DataSet(outcome_labels=['000', '111'])
    for circuit in (Circuit([Label('Gxpi2', 0)], line_labels=(0, 1, 2)),
                    Circuit([Label('Gxpi2', 1)], line_labels=(0, 1, 2))):
        dataset.add_count_dict(circuit, {'000': 10, '111': 10})
    dataset.done_adding_data()
    return dataset


def _tee_pspec(edges):
    """
    A 4-qubit "T": qubit 1 is a degree-3 hub joined to 0, 2 and 3.

    ``edges`` selects how that one physical graph is *written* in availability;
    the hardware is the same whichever way each tuple is oriented.

    See :func:`_line_pspec` for why "Gii" must be available on every 2Q edge.
    """
    qubits = (0, 1, 2, 3)
    oneq_locations = [(q,) for q in qubits]
    availability = {
        'Gi': oneq_locations,
        'Gxpi2': oneq_locations,
        'Gypi2': oneq_locations,
        'Gcnot': list(edges),
        'Gii': list(edges),
    }
    return QubitProcessorSpec(
        4, gate_names=['Gi', 'Gxpi2', 'Gypi2', 'Gcnot', 'Gii'],
        nonstd_gate_unitaries={'Gii': np.eye(4)},
        availability=availability, qubit_labels=qubits,
    )


class DirectedAvailabilityTeeTester(BaseCase):
    """
    A degree-3 hub whose availability is written one-directional.

    ``find_neighbors`` used to walk only ``e[0] -> e[1]``, so naming each edge
    once understated the max degree -- 2 instead of 3 here -- and the coloring
    packed adjacent edges into one color. That patch reached ``batch_tensor``
    as the five lines ``[(0, 1), (1, 2), (3,)]`` on a four-qubit device and
    tripped a bare, message-less assert far below its cause.

    Hand-written availability is what armed it: ``geometry='line'`` and friends
    emit both orientations themselves, making the old map accidentally
    symmetric, but spelling out a T or heavy-hex by hand does not -- and one
    entry per edge is natural for CNOT, where control/target is real.

    ``test_graphcoloring.TeeOrientationInvarianceTester`` covers the graph-level
    facts on plain edge lists. What needs a pspec, and is what actually
    regressed, is that the design is constructible at all.
    """

    #: Each edge written once, pointing away from the hub where possible.
    ONE_DIRECTIONAL = [(0, 1), (1, 2), (1, 3)]
    #: The same graph with both orientations of every edge.
    TWO_DIRECTIONAL = ONE_DIRECTIONAL + [(1, 0), (2, 1), (3, 1)]

    def _build(self, edges):
        oneq, twoq = _make_designs()
        return make_simultaneous_gst_design(_tee_pspec(edges), oneq, twoq, seed=0)

    def test_design_construction_survives_a_one_directional_tee(self):
        # The call that used to die inside `batch_tensor` on
        # `assert not s.intersection(t)`. `check_valid_edge_coloring` is the
        # coloring package's own notion of proper: no two edges in a patch may
        # share a qubit, i.e. every patch is physically runnable.
        design = self._build(self.ONE_DIRECTIONAL)
        self.assertTrue(
            check_valid_edge_coloring(design.color_patches, ret_false_on_error=True))

    def test_design_construction_works_for_a_two_directional_tee(self):
        # This spelling used to be the one that worked, back when the doubled
        # edges were what made the adjacency map accidentally symmetric. It is
        # no longer independent cover: `canonical_edges` now collapses both
        # spellings to the same one-directional list before `find_neighbors`
        # sees them, so this fails too if `find_neighbors` regresses. Kept
        # because the doubled list is what a `geometry=`-built pspec produces,
        # and it must keep round-tripping to the same design.
        design = self._build(self.TWO_DIRECTIONAL)
        self.assertTrue(
            check_valid_edge_coloring(design.color_patches, ret_false_on_error=True))

    def test_both_spellings_build_the_same_design(self):
        # Orientation is notation, not physics, so the two availability
        # spellings must yield the same patches and the same circuits.
        one = self._build(self.ONE_DIRECTIONAL)
        two = self._build(self.TWO_DIRECTIONAL)
        self.assertEqual(one.color_patches, two.color_patches)
        self.assertEqual([[c.str for c in L] for L in one.circuit_lists],
                         [[c.str for c in L] for L in two.circuit_lists])

    def test_design_edges_are_canonical_whichever_spelling_is_used(self):
        # `SimultaneousGSTDesign.__init__` re-derives the edges from the pspec
        # rather than reusing the ones the coloring was built from, so it has to
        # canonicalize them the same way `make_simultaneous_gst_design` does.
        one = self._build(self.ONE_DIRECTIONAL)
        two = self._build(self.TWO_DIRECTIONAL)
        self.assertEqual(sorted(one.edges), sorted(two.edges))
        self.assertEqual(sorted(one.edges), [(0, 1), (1, 2), (1, 3)])


class MakeLineMapperValidationTester(BaseCase):
    """
    Cover ``make_line_mapper``'s four rejection paths.

    Each guards a different way the source/target tensor lines can fail to
    describe a well-defined relabeling, and each is reachable independently.
    """

    def test_line_count_mismatch_is_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            make_line_mapper([(0, 1)], [(2, 3), (4,)])
        self.assertIn('different lengths', str(ctx.exception))

    def test_line_arity_mismatch_is_rejected(self):
        # Same number of lines, but a 2Q source line paired with a 1Q target.
        with self.assertRaises(ValueError) as ctx:
            make_line_mapper([(0, 1)], [(2,)])
        self.assertIn('arity mismatch', str(ctx.exception))

    def test_inconsistent_mapping_for_one_label_is_rejected(self):
        # Source label 0 appears twice, mapped to 2 the first time and 4 the
        # second -- there is no single consistent image for it.
        with self.assertRaises(ValueError) as ctx:
            make_line_mapper([(0, 1), (0,)], [(2, 3), (4,)])
        self.assertIn('Inconsistent mapping', str(ctx.exception))

    def test_non_injective_mapping_is_rejected(self):
        # Two distinct source labels collapse onto the same target label. This
        # is consistent per-label (so it clears the check above) but not
        # one-to-one, so it would silently merge two qubits.
        with self.assertRaises(ValueError) as ctx:
            make_line_mapper([(0,), (1,)], [(2,), (2,)])
        self.assertIn('one-to-one', str(ctx.exception))

    def test_valid_lines_produce_the_documented_mapping(self):
        # The docstring's own example, so the rejection tests above are
        # anchored against a known-good input.
        self.assertEqual(
            make_line_mapper([(0, 1), (4,), (5,)], [(2, 3), (0,), (1,)]),
            {0: 2, 1: 3, 4: 0, 5: 1})


class AssignDesignsDefaultRandgenTester(BaseCase):
    """``assign_the_designs_with_mapping`` defaults ``randgen`` to ``default_rng(0)``."""

    @staticmethod
    def _run(**kwargs):
        oneq_lists = [_make_1q_circuits(3)]
        twoq_lists = [_make_2q_circuits(5)]
        oneq = _StubDesign(oneq_lists, (0,))
        twoq = _StubDesign(twoq_lists, (0, 1))
        return assign_the_designs_with_mapping(
            oneq, twoq, [0, 1, 2], {0: [(0, 1)]}, **kwargs)

    def test_omitted_randgen_matches_explicit_default_rng_zero(self):
        omitted = self._run()
        explicit = self._run(randgen=np.random.default_rng(0))
        self.assertEqual(omitted, explicit)

    def test_omitted_randgen_still_produces_valid_output(self):
        circuit_lists = self._run()
        assert_circuit_lists_match_color_patches(circuit_lists, [0, 1, 2], {0: [(0, 1)]})

    def test_default_differs_from_a_different_seed(self):
        # Confirms the default is a real generator being consumed, not an
        # unused placeholder that would make every seed equivalent.
        self.assertNotEqual(self._run(), self._run(randgen=np.random.default_rng(12345)))

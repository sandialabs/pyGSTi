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
from pygsti.baseobjs.label import Label
from pygsti.modelpacks import smq1Q_XYI, smq2Q_XYICNOT
from pygsti.processors import QubitProcessorSpec
from pygsti.protocols.xfgst_edesign import (
    CrosstalkFreeExperimentDesign, assert_circuit_lists_match_color_patches,
    assert_mapped_circuit_matches_patch, assert_no_implicit_idles,
    assign_the_designs_with_mapping, build_layer_mappers,
    build_patch_infos, find_neighbors, make_line_mapper, make_xfgst_design,
)
from ..util import BaseCase


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

        circuit_lists = assign_the_designs_with_mapping(
            oneq, twoq, vertices, color_patches,
            randgen=np.random.default_rng(seed),
            ensure_containment=True,
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



def _line_pspec(n_qubits=3):
    """An `n_qubits`-qubit line processor spec usable with the XFGST design.

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


class MakeXfgstDesignTester(BaseCase):
    """
    Cover ``make_xfgst_design``, the public convenience entry point.

    It derives the graph (vertices, edges, neighbors, max degree) from the
    processor spec, computes an edge coloring with the 'auto' algorithm, and
    forwards everything to ``CrosstalkFreeExperimentDesign`` with ``seed + 1``.
    These tests check that each of those derived values lands on the returned
    design correctly.
    """

    @classmethod
    def setUpClass(cls):
        cls.pspec, cls.qubits, cls.line_edges = _line_pspec(3)
        cls.oneq, cls.twoq = _make_designs()
        cls.design = make_xfgst_design(cls.pspec, cls.oneq, cls.twoq, seed=0)

    def test_returns_crosstalk_free_design_with_inputs_passed_through(self):
        self.assertIsInstance(self.design, CrosstalkFreeExperimentDesign)
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
        # make_xfgst_design passes neither circuit_stitcher nor nested, so both
        # must land on their CrosstalkFreeExperimentDesign defaults.
        self.assertIs(self.design.circuit_stitcher, assign_the_designs_with_mapping)
        self.assertFalse(self.design.nested)
        self.assertFalse(self.design.stitcher_kwargs['ensure_containment'])
        self.assertEqual(self.design.aux_info, {})

    def test_edge_coloring_is_a_valid_proper_coloring(self):
        color_patches = self.design.color_patches
        self.assertIsInstance(color_patches, dict)

        # Every edge is coloured exactly once...
        coloured = [tuple(e) for edges in color_patches.values() for e in edges]
        self.assertEqual(sorted(coloured), sorted(self.line_edges))
        self.assertEqual(len(coloured), len(set(coloured)))

        # ...and each colour class is a matching, i.e. its edges are pairwise
        # disjoint. This is the property the whole crosstalk-free construction
        # rests on: two 2Q GST circuits in one patch must not share a qubit.
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

    def test_seed_is_forwarded_as_seed_plus_one(self):
        # make_xfgst_design builds the design with seed=(seed + 1); constructing
        # the design directly with that seed and the same coloring must reproduce
        # it exactly.
        #
        # The design is rebuilt here rather than reusing cls.design on purpose:
        # coverage contexts attribute setUpClass code to whichever test happened
        # to trigger class setup, so a test that only *reads* cls.design is not
        # recorded as covering the `seed + 1` expression and is therefore never
        # selected by the diff-mutation tooling to guard it.
        design = make_xfgst_design(self.pspec, self.oneq, self.twoq, seed=0)
        direct = CrosstalkFreeExperimentDesign(
            self.pspec, self.oneq, self.twoq, design.color_patches, seed=1)
        self.assertEqual(direct.circuit_lists, design.circuit_lists)

    def test_different_seeds_give_different_circuit_assignments(self):
        # Guards the seed actually reaching the stitcher's randgen: on a line the
        # coloring is seed-independent, so any difference comes from the seed.
        other = make_xfgst_design(self.pspec, self.oneq, self.twoq, seed=7)
        self.assertEqual(other.color_patches, self.design.color_patches)
        self.assertNotEqual(other.circuit_lists, self.design.circuit_lists)


class HelperRejectsMalformedInputTester(BaseCase):
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
    ``CrosstalkFreeExperimentDesign.__init__``.
    """

    @classmethod
    def setUpClass(cls):
        cls.pspec, cls.qubits, cls.line_edges = _line_pspec(3)
        cls.oneq, cls.twoq = _make_designs()
        cls.design = make_xfgst_design(cls.pspec, cls.oneq, cls.twoq, seed=0)

    # -- debug_check wiring in CrosstalkFreeExperimentDesign.__init__ ------

    @staticmethod
    def _malformed_stitcher(oneq_gstdesign, twoq_gstdesign, vertices,
                            color_patches, **kwargs):
        """A stitcher returning output that cannot be a valid stitching.

        One circuit cannot split evenly into the two color patches below. The
        inputs are ignored so the (slow) real stitching is never run.
        """
        return [[Circuit([Label('Gcnot', (0, 1))], line_labels=(0, 1, 2))]]

    def _build_with_malformed_stitcher(self, **kwargs):
        return CrosstalkFreeExperimentDesign(
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
        patch_infos, _ = build_patch_infos([0, 1, 2], {0: [(0, 1)]})
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


class FindNeighborsTester(BaseCase):
    """``find_neighbors`` adjacency construction."""

    def test_undirected_edge_list_is_recorded_once_per_edge(self):
        # `edges` holds one copy of each undirected edge -- deliberately not
        # both (u, v) and (v, u) -- so that downstream edge-coloring colors each
        # edge once rather than once per direction. find_neighbors records each
        # edge in the direction it was given.
        vertices = (0, 1, 2)
        edges = [(0, 1), (1, 2)]
        self.assertEqual(find_neighbors(vertices, edges),
                         {0: [1], 1: [2], 2: []})

    def test_isolated_vertices_get_empty_neighbor_lists(self):
        # Every vertex must appear as a key even with no incident edges, since
        # callers index `neighbors[v]` for all v when computing the max degree.
        self.assertEqual(find_neighbors((0, 1, 2), [(0, 1)]),
                         {0: [1], 1: [], 2: []})

    def test_multiple_neighbors_accumulate_in_edge_order(self):
        self.assertEqual(find_neighbors((0, 1, 2, 3), [(1, 0), (1, 2), (1, 3)]),
                         {0: [], 1: [0, 2, 3], 2: [], 3: []})


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


class LayerMappersOverrideTester(BaseCase):
    """
    Cover the ``_layer_mappers_override`` branch of
    ``assign_the_designs_with_mapping``.

    The contract an override has to satisfy is that no implicit idle survives
    into the stitched circuits -- ``batch_tensor`` relies on every layer being
    fully populated. ``assert_no_implicit_idles`` is exactly that check, so the
    tests below assert it over every circuit the override produced.

    The stub designs deliberately make the 2Q design the *shorter* one (2
    circuits, max depth 2) against a longer 1Q design (7 circuits, max depth 7),
    so the 2Q lane gets depth-padded and the mappers' 2Q-idle entry -- the entry
    these tests vary -- is actually reached.
    """

    VERTICES = [0, 1, 2]
    COLOR_PATCHES = {0: [(0, 1)]}

    @staticmethod
    def _designs():
        return (_StubDesign([_make_1q_circuits(7)], (0,)),
                _StubDesign([_make_2q_circuits(2)], (0, 1)))

    def _run(self, override=None):
        oneq, twoq = self._designs()
        kwargs = {} if override is None else {'_layer_mappers_override': override}
        return assign_the_designs_with_mapping(
            oneq, twoq, self.VERTICES, self.COLOR_PATCHES,
            randgen=np.random.default_rng(0), **kwargs)

    @staticmethod
    def _labels(circuit_lists):
        return {lbl
                for germ_power_list in circuit_lists
                for circuit in germ_power_list
                for i in range(circuit.num_layers)
                for lbl in circuit.layer(i)}

    def _assert_no_implicit_idles(self, circuit_lists):
        for germ_power_list in circuit_lists:
            for circuit in germ_power_list:
                assert_no_implicit_idles(circuit)

    def test_fixture_actually_exercises_the_twoq_idle(self):
        # Guards the tests below: if the 2Q lane ever stopped being padded, the
        # override entry they vary would go unused and they'd pass vacuously.
        self.assertIn(Label(('Gii', 0, 1)), self._labels(self._run()))

    def test_override_output_has_no_implicit_idles(self):
        oneq, twoq = self._designs()
        self._assert_no_implicit_idles(self._run(build_layer_mappers(oneq, twoq)))

    def test_override_equal_to_built_mappers_reproduces_default_path(self):
        # Passing exactly what build_layer_mappers would have built must be
        # indistinguishable from not passing an override at all -- i.e. the
        # branch selects between two spellings of the same thing.
        oneq, twoq = self._designs()
        self.assertEqual(self._run(build_layer_mappers(oneq, twoq)), self._run())

    def test_custom_override_is_actually_used(self):
        # Swap the 2Q idle for a parallel pair of explicit 1Q idles. That is a
        # legal alternative mapping (it introduces no implicit idle) and it must
        # change the output, otherwise the override is being ignored.
        oneq, twoq = self._designs()
        parallel_idle = Label((Label('Gi', 0), Label('Gi', 1)))
        override = build_layer_mappers(oneq, twoq)
        override[2] = dict(override[2])
        override[2][Label(())] = parallel_idle
        override[2][Label(('Gii', 0, 1))] = parallel_idle

        circuit_lists = self._run(override)
        self.assertNotEqual(circuit_lists, self._run())

        labels = self._labels(circuit_lists)
        self.assertNotIn(Label(('Gii', 0, 1)), labels)
        self.assertIn(Label('Gi', 0), labels)
        self.assertIn(Label('Gi', 1), labels)
        self._assert_no_implicit_idles(circuit_lists)

    def test_override_mapping_the_implicit_idle_to_itself_is_rejected(self):
        # An override that sends the empty layer to itself would leave the
        # implicit idle in place. build_layer_mappers asserts Label(()) never
        # appears among a mapper's *values*, but an override bypasses that, so
        # the guarantee has to hold downstream: batch_tensor re-checks the
        # invariant and fails the stitch rather than emitting a bad circuit.
        oneq, twoq = self._designs()
        override = build_layer_mappers(oneq, twoq)
        override[2] = dict(override[2])
        override[2][Label(())] = Label(())

        with self.assertRaises(AssertionError):
            self._run(override)

    def test_override_omitting_the_twoq_idle_key_is_rejected(self):
        # The 2Q lane needs depth padding here, so an override missing the
        # implicit-idle entry cannot supply an explicit idle for the pad layers.
        oneq, twoq = self._designs()
        override = build_layer_mappers(oneq, twoq)
        override[2] = {k: v for k, v in override[2].items() if k != Label(())}

        with self.assertRaises((AssertionError, KeyError)):
            self._run(override)

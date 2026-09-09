#***************************************************************************************************
# Copyright 2015, 2019, 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

import copy as _copy
import hashlib
import json
import pathlib

import numpy as np

import pygsti
from pygsti.circuits.circuit import Circuit
from pygsti.baseobjs.label import Label
from pygsti.modelpacks import smq1Q_XYI, smq2Q_XYICNOT
from pygsti.processors import QubitProcessorSpec
from pygsti.protocols.simultaneous_gst import (
    SimultaneousGSTDesign, _normalize_coloring, _recordable_seed, _seed_from_record,
    assert_circuit_lists_match_color_patches,
    assign_the_designs_with_mapping,
    make_simultaneous_gst_design,
)
from pygsti.protocols._stitchers import (
    CallableStitcher, CircuitStitcher, RandomizedPatchStitcher,
    assert_mapped_circuit_matches_patch, build_patch_infos, make_line_mapper,
)
from pygsti.protocols.gst import GateSetTomographyDesign
from pygsti.protocols.protocol import CombinedExperimentDesign
from pygsti.tools.graphcoloring import check_valid_edge_coloring
from ..util import BaseCase, with_temp_path


class _StubDesign:
    """Minimal stand-in for a GateSetTomographyDesign for stitcher unit tests."""
    def __init__(self, circuit_lists, qubit_labels):
        self.circuit_lists = circuit_lists
        self.qubit_labels = qubit_labels


def _make_1q_circuits(n):
    """n distinct single-qubit circuits on line (0,)."""
    return [Circuit([Label('Gx', 0)] * (i + 1), line_labels=(0,)) for i in range(n)]


def _make_2q_circuits(n):
    """n distinct two-qubit circuits on lines (0, 1)."""
    return [Circuit([Label('Gcnot', (0, 1))] * (i + 1), line_labels=(0, 1)) for i in range(n)]


def _stitch(oneq_lens, twoq_lens, vertices=(0, 1, 2), color_patches=None, seed=0, check=True, **kwargs):
    if color_patches is None:
        color_patches = {0: [(0, 1)]}
    if isinstance(oneq_lens, int):
        oneq_lens = [oneq_lens]
    if isinstance(twoq_lens, int):
        twoq_lens = [twoq_lens]
    oneq = _StubDesign([_make_1q_circuits(n) for n in oneq_lens], (0,))
    twoq = _StubDesign([_make_2q_circuits(n) for n in twoq_lens], (0, 1))
    if 'randgen' not in kwargs and seed is not None:
        kwargs['randgen'] = np.random.default_rng(seed)
    circuit_lists = assign_the_designs_with_mapping(
        oneq, twoq, vertices, color_patches, **kwargs
    )
    if check:
        assert_circuit_lists_match_color_patches(circuit_lists, vertices, color_patches)
    return circuit_lists


def _cnot_edges(circuit):
    edges = set()
    for i in range(circuit.num_layers):
        for op in circuit.layer(i):
            if op.name == 'Gcnot':
                edges.add(tuple(op.qubits))
    return edges


def _gx_qubits(circuit):
    qubits = set()
    for i in range(circuit.num_layers):
        for op in circuit.layer(i):
            if op.name == 'Gx':
                qubits.update(op.qubits)
    return qubits


def _line_pspec(n_qubits=3):
    """An n_qubits-qubit line processor spec with standard gates and explicit two-qubit idles Gii."""
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
    """Small 1Q and 2Q GST designs with minimal depth to keep tests fast."""
    oneq = smq1Q_XYI.create_gst_experiment_design(
        max_max_length=max_max_length, qubit_labels=(0,))
    twoq = smq2Q_XYICNOT.create_gst_experiment_design(
        max_max_length=max_max_length, qubit_labels=(0, 1))
    return oneq, twoq


_STUB_CIRCUIT_LISTS = [[Circuit([Label('Gcnot', (0, 1))], line_labels=(0, 1, 2)),
                        Circuit([Label('Gcnot', (1, 2))], line_labels=(0, 1, 2))]]


def _dataset_over(circuits):
    """A DataSet with an entry for each of `circuits`, for truncate_to_available_data.

    The counts are arbitrary: truncation only asks which circuits the dataset has.
    """
    dataset = pygsti.data.DataSet(outcome_labels=['000', '111'])
    for circuit in circuits:
        dataset.add_count_dict(circuit, {'000': 10, '111': 10})
    dataset.done_adding_data()
    return dataset


def _tee_pspec(edges):
    """A 4-qubit T processor spec with qubit 1 as a degree-3 hub."""
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


class AssignDesignsLengthPairingTester(BaseCase):
    """Cover the pairing of 1Q and 2Q designs of differing per-L lengths."""

    def _run(self, oneq_len, twoq_len, seed=0):
        return _stitch(oneq_len, twoq_len, seed=seed)

    def test_oneq_twoq_lengths(self):
        # A longer 1Q design is a legitimate input --
        # neither design is required to be the longer one.
        for oneq_len, twoq_len in ((7, 3), (3, 7), (5, 5)):
            with self.subTest(oneq_len=oneq_len, twoq_len=twoq_len):
                circuit_lists = self._run(oneq_len, twoq_len)
                self.assertEqual(len(circuit_lists), 1)  # one germ-power group
                # max(len(oneq), len(twoq)) tensored circuits are produced.
                self.assertEqual(len(circuit_lists[0]), max(oneq_len, twoq_len))
                # Every generated circuit spans all three qubits.
                for c in circuit_lists[0]:
                    self.assertEqual(set(c.line_labels), {0, 1, 2})

    def test_shorter_list_recycled_up_to_longer(self):
        cases = [
            (6, 2, lambda op: op.name == 'Gcnot', 6),
            (2, 6, lambda op: op.name == 'Gx' and 2 in op.qubits, 6),
        ]
        for oneq_len, twoq_len, pred, expected in cases:
            with self.subTest(oneq_len=oneq_len, twoq_len=twoq_len):
                generated = self._run(oneq_len, twoq_len)[0]
                self.assertEqual(len(generated), expected)
                counts = {}
                for c in generated:
                    n_layers = sum(1 for i in range(c.num_layers) if any(pred(op) for op in c.layer(i)))
                    counts[n_layers] = counts.get(n_layers, 0) + 1
                self.assertNotIn(0, counts)
                self.assertGreaterEqual(counts.get(1, 0), 1)
                self.assertGreaterEqual(counts.get(2, 0), 1)
                self.assertEqual(sum(counts.values()), expected)

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
    """Pin the default stitcher's output for fixed seeds."""

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
        for name, (vertices, color_patches) in self.CASES.items():
            with self.subTest(case=name):
                circuit_lists = _stitch((3, 5), (3, 5), vertices, color_patches, seed=12345, check=False)
                blob = json.dumps([[c.str for c in L] for L in circuit_lists])
                digest = hashlib.sha256(blob.encode()).hexdigest()
                self.assertEqual(digest, self.EXPECTED[name])


class MultiplePatchesSameShapeTester(BaseCase):
    """Two color patches sharing the same shape are grouped together and share a randomly-generated template circuit."""

    def _run(self, oneq_len=4, twoq_len=4, seed=0):
        color_patches = {0: [(0, 1)], 1: [(1, 2)]}
        vertices = [0, 1, 2]
        circuit_lists = _stitch(oneq_len, twoq_len, vertices, color_patches, seed=seed)
        generated = circuit_lists[0]
        self.assertEqual(len(generated), max(oneq_len, twoq_len) * 2)
        half = len(generated) // 2
        return generated[:half], generated[half:]

    def test_patches_are_relabeled_not_duplicated(self):
        patch0_circuits, patch1_circuits = self._run()
        self.assertNotEqual(patch0_circuits, patch1_circuits)

        # Every patch-0 circuit's Gcnot must be on edge (0, 1) only, and every
        # patch-1 circuit's Gcnot must be on edge (1, 2) only -- never the other
        # patch's edge.
        for c in patch0_circuits:
            self.assertEqual(_cnot_edges(c), {(0, 1)})
        for c in patch1_circuits:
            self.assertEqual(_cnot_edges(c), {(1, 2)})

        # Patch 0's leftover 1Q circuit belongs to qubit 2; patch 1's belongs to
        # qubit 0.
        for c in patch0_circuits:
            self.assertEqual(_gx_qubits(c), {2})
        for c in patch1_circuits:
            self.assertEqual(_gx_qubits(c), {0})

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
    """With share_same_shape_schedules=False same-shape patches draw schedules independently."""

    VERTICES = [0, 1, 2, 3, 4, 5]
    COLOR_PATCHES = {0: [(0, 1), (2, 3)], 1: [(1, 2), (3, 4)]}

    def _run(self, share, seed=0, oneq_lens=(3, 6), twoq_lens=(3, 6)):
        return _stitch(oneq_lens, twoq_lens, self.VERTICES, self.COLOR_PATCHES, seed=seed, share_same_shape_schedules=share)

    def _patch_chunks(self, germ_power_list):
        half = len(germ_power_list) // 2
        return germ_power_list[:half], germ_power_list[half:]

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

    def test_independent_schedules_metadata_and_plumbing(self):
        # seed reproducibility
        self.assertEqual(self._run(share=False, seed=7), self._run(share=False, seed=7))
        # both settings produce the same number of circuits
        shared = self._run(share=True)
        independent = self._run(share=False)
        self.assertEqual([len(L) for L in shared], [len(L) for L in independent])
        # default is to share
        default = _stitch(3, 3, self.VERTICES, self.COLOR_PATCHES, seed=0)
        explicit = _stitch(3, 3, self.VERTICES, self.COLOR_PATCHES, seed=0, share_same_shape_schedules=True)
        self.assertEqual(default, explicit)


class NestingTester(BaseCase):
    """The stitcher's output is always nested patch-wise so the germ-power-major-then-patch-major ordering survives."""

    def _run(self, oneq_lens, twoq_lens, color_patches, vertices, seed=0):
        return _stitch(oneq_lens, twoq_lens, vertices, color_patches, seed=seed)

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
        cases = [
            ("single_patch", [0, 1, 2], {0: [(0, 1)]}, [3, 5], 2),
            ("multiple_same_shape_patches", [0, 1, 2], {0: [(0, 1)], 1: [(1, 2)]}, [3, 6], 2),
            ("three_germ_powers", [0, 1, 2], {0: [(0, 1)]}, [2, 4, 7], 3),
        ]
        for name, vertices, color_patches, lens, expected_len in cases:
            with self.subTest(name):
                circuit_lists = self._run(
                    oneq_lens=lens, twoq_lens=lens,
                    color_patches=color_patches, vertices=vertices,
                )
                self.assertEqual(len(circuit_lists), expected_len)
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
                self.assertEqual(_cnot_edges(c), {(0, 1)})
            for c in germ_power_list[half:]:
                self.assertEqual(_cnot_edges(c), {(1, 2)})

    def test_one_component_may_add_no_circuits_at_a_germ_power(self):
        # A valid nested StandardGSTDesign can repeat a circuit list when no
        # selected germ changes between adjacent maximum lengths.  The other
        # component's new circuits must still be stitchable against a sample
        # from the repeated component's cumulative pool.
        circuit_lists = self._run(
            oneq_lens=[3, 5], twoq_lens=[4, 4],
            color_patches={0: [(0, 1)]}, vertices=[0, 1, 2],
        )
        self.assertEqual([len(circuit_list) for circuit_list in circuit_lists], [4, 6])
        self._assert_patchwise_containment(
            circuit_lists, [0, 1, 2], {0: [(0, 1)]}
        )
        self._assert_no_duplicates(circuit_lists)

    def test_either_component_may_add_no_circuits_at_a_germ_power(self):
        circuit_lists = self._run(
            oneq_lens=[3, 3], twoq_lens=[4, 6],
            color_patches={0: [(0, 1)]}, vertices=[0, 1, 2],
        )
        self.assertEqual([len(circuit_list) for circuit_list in circuit_lists], [4, 6])
        self._assert_patchwise_containment(
            circuit_lists, [0, 1, 2], {0: [(0, 1)]}
        )
        self._assert_no_duplicates(circuit_lists)


class _SGSTFixture:
    """A 3-qubit line device and the simultaneous-GST design built on it.

    Class-scoped, so a tester that might mutate the design must copy it first (see
    ``TruncationTester._fresh_design``). Mixed in ahead of ``BaseCase`` so
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
        self.assertIsInstance(self.design.circuit_stitcher, RandomizedPatchStitcher)
        self.assertTrue(self.design.circuit_stitcher.share_same_shape_schedules)
        self.assertTrue(self.design.nested)

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

    #: The malformed stitcher's output: a 2Q gate on (0, 2), which is not an edge of
    #: either color patch below. The explicit Gi keeps the implicit-idle check from
    #: firing first, so the patch-membership check is what actually rejects it.
    _OFF_PATCH_LISTS = [[Circuit([[Label('Gcnot', (0, 2)), Label('I', 1)]],
                                 line_labels=(0, 1, 2))]]

    @classmethod
    def _malformed_stitcher(cls, oneq_gstdesign, twoq_gstdesign, vertices,
                            color_patches, **kwargs):
        """A stitcher returning output that cannot be a valid stitching.

        The inputs are ignored so the (slow) real stitching is never run.
        """
        return [[c.copy() for c in cl] for cl in cls._OFF_PATCH_LISTS]

    def _build_with_malformed_stitcher(self, **kwargs):
        return SimultaneousGSTDesign(
            self.pspec, self.oneq, self.twoq, {0: [(0, 1)], 1: [(1, 2)]},
            circuit_stitcher=self._malformed_stitcher, **kwargs)

    def test_malformed_stitcher_output_is_rejected_when_debug_check_true(self):
        with self.assertRaises(AssertionError) as ctx:
            self._build_with_malformed_stitcher(debug_check=True)
        self.assertIn('not an edge of any color patch', str(ctx.exception))

    def test_debug_check_defaults_to_true(self):
        # The self-check is the documented safety net for swapped-in stitchers,
        # so it must be on unless explicitly disabled.
        with self.assertRaises(AssertionError):
            self._build_with_malformed_stitcher()

    def test_malformed_stitcher_output_is_accepted_when_debug_check_false(self):
        design = self._build_with_malformed_stitcher(debug_check=False)
        # The malformed lists must land unchanged: debug_check switches the
        # verification off, it does not repair anything.
        self.assertEqual([list(cl) for cl in design.circuit_lists], self._OFF_PATCH_LISTS)

    # -- detection power of the helpers themselves -------------------------

    #: Two patches on a 3-qubit line, so a circuit can be off-patch or span both.
    _TWO_PATCHES = {0: [(0, 1)], 1: [(1, 2)]}

    def _assert_rejects(self, layers, fragment, color_patches=None, line_labels=(0, 1, 2)):
        """A one-circuit design built from `layers` must be rejected, naming `fragment`."""
        circuit = Circuit(layers, line_labels=line_labels)
        with self.assertRaises(AssertionError) as ctx:
            assert_circuit_lists_match_color_patches(
                [[circuit]], (0, 1, 2), color_patches or self._TWO_PATCHES)
        self.assertIn(fragment, str(ctx.exception))

    def test_multiqubit_gate_on_a_non_patch_edge_is_rejected(self):
        # (0, 2) is not an edge of either patch. Circuits carry their patch in their
        # content, so a gate on no patch's edge means the circuit belongs nowhere.
        self._assert_rejects([[Label('Gcnot', (0, 2)), Label('I', 1)]],
                             'not an edge of any color patch')

    def test_a_circuit_spanning_two_patches_is_rejected(self):
        # Gcnot on (0, 1) is patch 0's, on (1, 2) is patch 1's. A simultaneous-GST
        # circuit runs one patch at a time; both in one circuit is not a stitching.
        self._assert_rejects([[Label('Gcnot', (0, 1)), Label('I', 2)],
                              [Label('Gcnot', (1, 2)), Label('I', 0)]],
                             'spans two color patches')

    def test_a_circuit_missing_a_vertex_is_rejected(self):
        # Every patch's tensored lines cover all the vertices, so a circuit that
        # does not name qubit 2 is not a full-register circuit. This one has no
        # implicit idle -- both its lines are busy -- so the line-label check is
        # what has to catch it.
        circuit = Circuit([Label('Gcnot', (0, 1))], line_labels=(0, 1))
        with self.assertRaises(AssertionError) as ctx:
            assert_circuit_lists_match_color_patches(
                [[circuit]], (0, 1, 2), self._TWO_PATCHES)
        self.assertEqual(ctx.exception.args[0], ({0, 1}, {0, 1, 2}))

    def test_implicit_idles_are_rejected(self):
        # Same circuit content as the accepted case below but without the explicit
        # Gi, so the line labels are right and only the implicit idle is wrong.
        self._assert_rejects([Label('Gcnot', (0, 1))], 'Implicit idle gate')

    def test_a_reversed_orientation_edge_is_accepted(self):
        # The coloring records (1, 2); a Gcnot on (2, 1) is the same edge with the
        # control and target swapped, which is a legitimate circuit on that patch.
        circuit = Circuit([[Label('Gcnot', (2, 1)), Label('I', 0)]], line_labels=(0, 1, 2))
        assert_circuit_lists_match_color_patches([[circuit]], (0, 1, 2), self._TWO_PATCHES)

    def test_a_circuit_with_no_multiqubit_gates_is_accepted(self):
        # It is consistent with every patch, so there is nothing to reject.
        circuit = Circuit([[Label('Gxpi2', 0), Label('I', 1), Label('I', 2)]],
                          line_labels=(0, 1, 2))
        assert_circuit_lists_match_color_patches([[circuit]], (0, 1, 2), self._TWO_PATCHES)

    def test_an_edge_in_two_patches_is_rejected(self):
        # Not a proper coloring: the circuit's patch would not be determined by its
        # content, so the validator refuses the coloring rather than guess.
        circuit = Circuit([[Label('Gcnot', (0, 1)), Label('I', 2)]], line_labels=(0, 1, 2))
        with self.assertRaises(AssertionError) as ctx:
            assert_circuit_lists_match_color_patches(
                [[circuit]], (0, 1, 2), {0: [(0, 1)], 1: [(0, 1)]})
        self.assertIn('appears in both patch', str(ctx.exception))

    def test_every_circuit_is_verified_not_just_the_first(self):
        """A bad circuit anywhere in the lists must be caught.

        The old validator sliced each germ-power list into equal patch-major chunks
        and checked each chunk against its patch, so an off-by-one in the slice
        bounds could leave later patches unverified. Content-based validation has no
        slices, but it still walks a list, so a bad circuit in a later position -- or
        in a later list -- must still raise.
        """
        good = Circuit([[Label('Gcnot', (0, 1)), Label('I', 2)]], line_labels=(0, 1, 2))
        bad = Circuit([[Label('Gcnot', (0, 2)), Label('I', 1)]], line_labels=(0, 1, 2))
        for name, lists in (('last in the first list', [[good, bad]]),
                            ('alone in a later list', [[good], [good, bad]])):
            with self.subTest(name):
                with self.assertRaises(AssertionError):
                    assert_circuit_lists_match_color_patches(
                        lists, (0, 1, 2), self._TWO_PATCHES)

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

        # Check key order is preserved
        coloring = {2: [[3, 4], [0, 1]], 0: [[1, 2]]}
        normalized = _normalize_coloring(coloring)
        self.assertEqual(list(normalized), [2, 0])

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

    def test_processor_spec_and_qubit_labels_are_mapped(self):
        self.assertIsInstance(self.mapped, SimultaneousGSTDesign)
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
        self.design.circuit_stitcher = CallableStitcher(_explode)
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
        # A design built with debug_check=False cannot launder itself clean by
        # relabelling. Gcnot on (0, 2) belongs to neither patch, and the relabelled
        # (Q0, Q2) belongs to neither relabelled patch either -- so this also checks
        # the validator sees the *mapped* content, not the original.
        bad = Circuit([[Label('Gcnot', (0, 2)), Label('I', 1)]], line_labels=(0, 1, 2))
        malformed = SimultaneousGSTDesign(
            self.pspec, self.oneq, self.twoq, {0: [(0, 1)], 1: [(1, 2)]},
            circuit_stitcher=(lambda *a, **kw: [[bad.copy()]]), debug_check=False)

        with self.assertRaises(AssertionError) as ctx:
            malformed.map_qubit_labels(self.mapper)
        message = str(ctx.exception)
        self.assertIn('not an edge of any color patch', message)
        self.assertIn("'Q0', 'Q2'", message)

        # ...and debug_check=False switches that verification off here too.
        relabelled = malformed.map_qubit_labels(self.mapper, debug_check=False)
        self.assertEqual([list(cl) for cl in relabelled.circuit_lists],
                         [[Circuit([[Label('Gcnot', ('Q0', 'Q2')), Label('I', 'Q1')]],
                                   line_labels=('Q0', 'Q1', 'Q2'))]])


def _stub_stitcher(oneq_gstdesign, twoq_gstdesign, vertices, color_patches, **kwargs):
    """A module-level (hence importable) stitcher function, for the CallableStitcher tests."""
    return [[c.copy() for c in cl] for cl in _STUB_CIRCUIT_LISTS]


class _CountingStitcher(CircuitStitcher):
    """A stitcher subclass with configuration, to check that options round-trip."""

    def __init__(self, tag='default'):
        super().__init__()
        self.tag = tag

    def _stitch(self, oneq_gstdesign, twoq_gstdesign, vertices, color_patches, randgen,
                verbosity):
        return [[c.copy() for c in cl] for cl in _STUB_CIRCUIT_LISTS]

    def _to_nice_serialization(self):
        state = super()._to_nice_serialization()
        state['tag'] = self.tag
        return state

    @classmethod
    def _from_nice_serialization(cls, state):
        return cls(state['tag'])


class _StatelessStitcher(CircuitStitcher):
    """A stitcher subclass with no configuration, hence no serialization code."""

    def _stitch(self, oneq_gstdesign, twoq_gstdesign, vertices, color_patches, randgen,
                verbosity):
        return [[c.copy() for c in cl] for cl in _STUB_CIRCUIT_LISTS]


class CircuitStitcherTester(BaseCase):
    """The stitcher interface itself: casting, validation, and serialization.

    The stitching *algorithm* is covered by the many tests above that call
    `assign_the_designs_with_mapping` directly. What matters here is the wrapper.
    """

    VERTICES = ('Q0', 'Q1', 'Q2')
    COLOR_PATCHES = {0: [('Q0', 'Q1')]}

    def _stitch_with(self, stitcher, **kwargs):
        return stitcher.stitch(None, None, self.VERTICES, self.COLOR_PATCHES,
                               debug_check=False, **kwargs)

    # -- cast ---------------------------------------------------------------- #

    def test_a_stitcher_casts_to_itself(self):
        stitcher = RandomizedPatchStitcher()
        self.assertIs(CircuitStitcher.cast(stitcher), stitcher)

    def test_a_callable_is_wrapped(self):
        stitcher = CircuitStitcher.cast(_stub_stitcher)
        self.assertIsInstance(stitcher, CallableStitcher)
        self.assertEqual(len(self._stitch_with(stitcher)), len(_STUB_CIRCUIT_LISTS))

    def test_options_passed_alongside_a_built_stitcher_are_rejected(self):
        """Silently dropping them is how the old **stitcher_kwargs swallowed typos."""
        with self.assertRaises(TypeError) as ctx:
            CircuitStitcher.cast(RandomizedPatchStitcher(), share_same_shape_schedules=False)
        self.assertIn('share_same_shape_schedules', str(ctx.exception))

    def test_casting_a_non_callable_says_what_is_accepted(self):
        with self.assertRaises(TypeError):
            CircuitStitcher.cast('randomized')

    # -- the injected rng ----------------------------------------------------- #

    def test_the_seed_reaches_stitch_as_a_generator(self):
        seen = {}

        class _Recording(CircuitStitcher):
            def _stitch(self, oneq, twoq, vertices, patches, randgen, verbosity):
                seen['randgen'] = randgen
                seen['draw'] = randgen.random()
                return []

        self._stitch_with(_Recording(), seed=1234)
        self.assertIsInstance(seen['randgen'], np.random.Generator)
        self.assertEqual(seen['draw'], np.random.default_rng(1234).random())

    def test_an_unimplemented_stitch_says_what_to_implement(self):
        with self.assertRaises(NotImplementedError) as ctx:
            self._stitch_with(CircuitStitcher())
        self.assertIn('_stitch', str(ctx.exception))

    # -- validation ------------------------------------------------------------ #

    def test_returning_something_other_than_a_list_is_rejected(self):
        class _Bad(CircuitStitcher):
            def _stitch(self, oneq, twoq, vertices, patches, randgen, verbosity):
                return 'not circuit lists'

        with self.assertRaises(TypeError) as ctx:
            self._stitch_with(_Bad())
        self.assertIn('_Bad', str(ctx.exception))

    def test_returning_bare_circuits_instead_of_lists_is_rejected(self):
        """One level of nesting missing is an easy mistake and a confusing one later."""
        class _Flat(CircuitStitcher):
            def _stitch(self, oneq, twoq, vertices, patches, randgen, verbosity):
                return [Circuit([Label('Gi', 'Q0')], line_labels=('Q0',))]

        with self.assertRaises(TypeError) as ctx:
            self._stitch_with(_Flat())
        self.assertIn('germ-power index 0', str(ctx.exception))

    def test_returning_a_non_circuit_is_rejected(self):
        class _Strings(CircuitStitcher):
            def _stitch(self, oneq, twoq, vertices, patches, randgen, verbosity):
                return [['Gx']]

        with self.assertRaises(TypeError):
            self._stitch_with(_Strings())

    def test_the_coloring_check_runs_for_direct_calls_too(self):
        """It used to live in SimultaneousGSTDesign.__init__, so it did not."""
        off_patch = [[Circuit([[Label('Gcnot', ('Q0', 'Q2')), Label('I', 'Q1')]],
                              line_labels=self.VERTICES)]]

        class _OffPatch(CircuitStitcher):
            def _stitch(self, oneq, twoq, vertices, patches, randgen, verbosity):
                return [[c.copy() for c in cl] for cl in off_patch]

        with self.assertRaises(AssertionError) as ctx:
            _OffPatch().stitch(None, None, self.VERTICES, self.COLOR_PATCHES,
                               debug_check=True)
        self.assertIn('not an edge of any color patch', str(ctx.exception))

    # -- serialization ---------------------------------------------------------- #

    def test_the_default_stitcher_round_trips_with_its_options(self):
        stitcher = RandomizedPatchStitcher(share_same_shape_schedules=False)
        restored = CircuitStitcher.from_nice_serialization(stitcher.to_nice_serialization())
        self.assertIsInstance(restored, RandomizedPatchStitcher)
        self.assertFalse(restored.share_same_shape_schedules)

    def test_a_third_party_subclass_round_trips_without_being_registered(self):
        restored = CircuitStitcher.from_nice_serialization(
            _CountingStitcher('mine').to_nice_serialization())
        self.assertIsInstance(restored, _CountingStitcher)
        self.assertEqual(restored.tag, 'mine')

    def test_a_stateless_subclass_needs_no_serialization_code(self):
        restored = CircuitStitcher.from_nice_serialization(
            _StatelessStitcher().to_nice_serialization())
        self.assertIsInstance(restored, _StatelessStitcher)

    def test_a_module_level_function_backed_callable_stitcher_round_trips(self):
        stitcher = CallableStitcher(_stub_stitcher, extra=7)
        restored = CircuitStitcher.from_nice_serialization(stitcher.to_nice_serialization())
        self.assertIsInstance(restored, CallableStitcher)
        self.assertIs(restored.func, _stub_stitcher)
        self.assertEqual(restored.kwargs, {'extra': 7})

    def test_a_lambda_backed_callable_stitcher_does_not_round_trip(self):
        """Documented limitation of the escape hatch, pinned so it stays documented."""
        state = CallableStitcher(lambda *a, **kw: []).to_nice_serialization()
        with self.assertRaises(ValueError) as ctx:
            CircuitStitcher.from_nice_serialization(state)
        self.assertIn('lambda', str(ctx.exception))


class SeedRecordTester(BaseCase):
    """`stitch_seed` is what lets a reloaded design re-stitch, so it has to be data."""

    def test_an_int_is_recorded_as_itself(self):
        self.assertEqual(_recordable_seed(7), 7)
        self.assertEqual(_seed_from_record(7), 7)

    def test_none_stays_none(self):
        self.assertIsNone(_recordable_seed(None))
        self.assertIsNone(_seed_from_record(None))

    def test_a_seed_sequence_round_trips_to_the_same_stream(self):
        original = np.random.SeedSequence(12345).spawn(2)[1]
        record = _recordable_seed(original)
        self.assertIsInstance(record, dict)
        json.dumps(record)  # must survive meta.json
        restored = _seed_from_record(record)
        self.assertEqual(np.random.default_rng(restored).random(),
                         np.random.default_rng(original).random())

    def test_a_live_generator_cannot_be_recorded_and_says_so(self):
        with self.assertWarns(Warning) as ctx:
            self.assertIsNone(_recordable_seed(np.random.default_rng(0)))
        self.assertIn('re-stitch', str(ctx.warning))


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
        # The seed is preserved as data, so the loaded design can re-stitch.
        self.assertEqual(loaded.stitch_seed, self.design.stitch_seed)

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

    @with_temp_path
    def test_the_stitcher_is_restored_as_an_object(self, root_path):
        _, loaded = self._roundtrip(root_path)
        self.assertIsInstance(loaded.circuit_stitcher, RandomizedPatchStitcher)
        self.assertEqual(loaded.circuit_stitcher.share_same_shape_schedules,
                         self.design.circuit_stitcher.share_same_shape_schedules)

    @with_temp_path
    def test_a_stitcher_defined_outside_pygsti_is_restored_too(self, root_path):
        """The reason the stitcher is serialized rather than named: a subclass
        anywhere round-trips, where an import path only works for a module-level def."""
        design = SimultaneousGSTDesign(
            self.pspec, self.oneq, self.twoq, {0: [(0, 1)], 1: [(1, 2)]},
            circuit_stitcher=_CountingStitcher('from-a-test-module'), debug_check=False)
        root = pathlib.Path(root_path) / 'custom'
        design.write(root)
        loaded = SimultaneousGSTDesign.from_dir(root)
        self.assertIsInstance(loaded.circuit_stitcher, _CountingStitcher)
        self.assertEqual(loaded.circuit_stitcher.tag, 'from-a-test-module')

    @with_temp_path
    def test_a_loaded_design_can_restitch_its_own_circuits(self, root_path):
        """What the callable stitcher could not do: `stitcher_kwargs` held a live
        Generator, so a reloaded design reproduced its circuits but could not rebuild
        them."""
        _, loaded = self._roundtrip(root_path)
        restitched = loaded.restitch()
        self.assertEqual([list(cl) for cl in restitched.circuit_lists],
                         [list(cl) for cl in self.design.circuit_lists])


class TruncationTester(_SGSTFixture, BaseCase):
    """
    Cover dropping circuits from a ``SimultaneousGSTDesign``.

    A circuit carries its patch in its content -- which multi-qubit gates act on which
    edges -- so removing circuits leaves the survivors valid. What the inherited
    machinery gets wrong is ``nested``: ``CircuitListsDesign._truncate_to_circuits_inplace``
    clears it unconditionally, because filtering circuit lists need not preserve
    containment in general. Here it does, since every germ-power list is filtered by one
    common keep-set. Losing ``nested=True`` is silent and would make iterative GST
    re-derive the full circuit set by unioning the lists, so each route gets a test.
    """

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        # Take the keep-set from the *first* germ-power list so that a nested design
        # keeps circuits at every length; slicing all_circuits_needing_data (the last
        # list) would be biased toward the deepest circuits.
        cls.some_circuits = list(cls.design.circuit_lists[0])[:5]

    def _fresh_design(self):
        """A private copy of the class-scoped design, for tests that mutate in place."""
        return _copy.deepcopy(self.design)

    def _assert_is_a_valid_truncation(self, truncated, expected_circuits):
        """The result is a SimultaneousGSTDesign that still satisfies every invariant."""
        self.assertIsInstance(truncated, SimultaneousGSTDesign)
        self.assertEqual(set(truncated.all_circuits_needing_data), set(expected_circuits))
        self.assertTrue(truncated.nested)
        # Still a well-formed stitching: content-based validation does not care that
        # the equal-chunk patch-major layout is gone.
        assert_circuit_lists_match_color_patches(
            truncated.circuit_lists, truncated.vertices, truncated.color_patches)
        # Nesting is a claim about the lists, so check it rather than trusting the flag.
        for lower, higher in zip(truncated.circuit_lists, truncated.circuit_lists[1:]):
            self.assertTrue(set(lower).issubset(set(higher)))
        # The simultaneous-GST structure survives. Truncation deepcopies, so the
        # sub-designs are copies rather than the same objects -- compare their content.
        self.assertEqual(truncated.color_patches, self.design.color_patches)
        for attr in ('oneq_gstdesign', 'twoq_gstdesign'):
            self.assertEqual(
                list(getattr(truncated, attr).all_circuits_needing_data),
                list(getattr(self.design, attr).all_circuits_needing_data))
        self.assertEqual(truncated.vertices, self.design.vertices)
        self.assertEqual(truncated.processor_spec.qubit_labels,
                         self.design.processor_spec.qubit_labels)

    def test_truncate_to_circuits_keeps_the_class_and_the_nesting(self):
        truncated = self.design.truncate_to_circuits(self.some_circuits)
        self._assert_is_a_valid_truncation(truncated, self.some_circuits)

    def test_truncate_to_circuits_does_not_mutate_the_original(self):
        before = [list(cl) for cl in self.design.circuit_lists]
        self.design.truncate_to_circuits(self.some_circuits)
        self.assertEqual([list(cl) for cl in self.design.circuit_lists], before)
        self.assertTrue(self.design.nested)

    def test_truncate_to_available_data_keeps_the_class_and_the_nesting(self):
        dataset = _dataset_over(self.some_circuits)
        truncated = self.design.truncate_to_available_data(dataset)
        self._assert_is_a_valid_truncation(truncated, self.some_circuits)

    def test_truncate_to_design_keeps_the_class_and_the_nesting(self):
        other = self.design.truncate_to_circuits(self.some_circuits)
        truncated = self.design.truncate_to_design(other)
        self._assert_is_a_valid_truncation(truncated, self.some_circuits)

    def test_truncate_to_design_against_an_unnested_design_drops_the_flag(self):
        # Truncating list L against *other*'s list L is a different keep-set per list,
        # so containment survives only if the other design is nested too. Claiming
        # nested=True here would be a lie about the lists.
        other = self.design.as_circuit_lists_design().truncate_to_circuits(self.some_circuits)
        self.assertFalse(other.nested)
        self.assertFalse(self.design.truncate_to_design(other).nested)

    def test_truncate_to_lists_keeps_the_class(self):
        # CircuitListsDesign.truncate_to_lists builds a plain CircuitListsDesign, which
        # would silently discard the pspec, the coloring and the sub-designs.
        truncated = self.design.truncate_to_lists([0])
        self.assertIsInstance(truncated, SimultaneousGSTDesign)
        self.assertEqual(len(truncated.circuit_lists), 1)
        self.assertEqual(list(truncated.circuit_lists[0]), list(self.design.circuit_lists[0]))
        self.assertEqual(set(truncated.all_circuits_needing_data),
                         set(self.design.circuit_lists[0]))
        self.assertTrue(truncated.nested)
        self.assertEqual(truncated.color_patches, self.design.color_patches)

    def test_truncation_from_a_parent_design_works(self):
        # ExperimentDesign._truncate_to_available_data_inplace loops over its children,
        # so a nested SimultaneousGSTDesign is reached without any direct call.
        parent = CombinedExperimentDesign({'sgst': self._fresh_design()})
        truncated = parent.truncate_to_available_data(_dataset_over(self.some_circuits))
        self._assert_is_a_valid_truncation(truncated['sgst'], self.some_circuits)

    def test_the_result_is_still_truncatable(self):
        once = self.design.truncate_to_circuits(self.some_circuits)
        twice = once.truncate_to_circuits(self.some_circuits[:2])
        self._assert_is_a_valid_truncation(twice, self.some_circuits[:2])

    @with_temp_path
    def test_a_truncated_design_round_trips_through_disk(self, root_path):
        # Truncation swaps the stitcher's plain lists for CircuitLists, so the objects
        # being written are no longer the kind CircuitListsDesign.__init__ inspected when
        # it chose auxfile_types['circuit_lists'] = 'list:text-circuit-list'. That is
        # survivable -- CircuitList.cast on a plain list produces one with no aliases or
        # weights, so text is a lossless encoding of it -- but it is survivable by
        # coincidence, and truncation is now a normal thing to do to this class.
        truncated = self.design.truncate_to_circuits(self.some_circuits)
        root = pathlib.Path(root_path) / 'truncated'
        truncated.write(root)
        loaded = SimultaneousGSTDesign.from_dir(root)
        self.assertEqual([list(cl) for cl in loaded.circuit_lists],
                         [list(cl) for cl in truncated.circuit_lists])
        self.assertEqual(set(loaded.all_circuits_needing_data), set(self.some_circuits))
        self.assertEqual(loaded.nested, truncated.nested)
        self.assertEqual(loaded.color_patches, truncated.color_patches)

    # -- what is still refused ----------------------------------------------

    def test_merge_with_is_still_refused(self):
        # Unlike truncation, merging has no rule for combining two edge colorings.
        with self.assertRaises(NotImplementedError) as ctx:
            self.design.merge_with(self.design)
        message = str(ctx.exception)
        self.assertIn('merge_with is not supported', message)
        self.assertIn('as_circuit_lists_design', message)

    # -- the plain-design hatch ----------------------------------------------

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

    def test_as_circuit_lists_design_supports_merging(self):
        # The remaining reason to reach for the hatch.
        merged = self.design.as_circuit_lists_design().merge_with(
            self.design.as_circuit_lists_design())
        self.assertEqual(set(merged.all_circuits_needing_data),
                         set(self.design.all_circuits_needing_data))

    def test_as_circuit_lists_design_does_not_alias_the_original(self):
        # It hands out its own lists, so truncating the copy cannot reach the original.
        plain = self.design.as_circuit_lists_design()
        plain.circuit_lists[0].pop()
        self.assertEqual([len(cl) for cl in self.design.circuit_lists],
                         [len(cl) + (1 if i == 0 else 0)
                          for i, cl in enumerate(plain.circuit_lists)])


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
    """Cover ``make_line_mapper``'s four rejection paths and valid example."""

    def test_rejections(self):
        cases = [
            ([(0, 1)], [(2, 3), (4,)], 'different lengths'),
            ([(0, 1)], [(2,)], 'arity mismatch'),
            ([(0, 1), (0,)], [(2, 3), (4,)], 'Inconsistent mapping'),
            ([(0,), (1,)], [(2,), (2,)], 'one-to-one'),
        ]
        for src, tgt, msg in cases:
            with self.subTest(msg=msg):
                with self.assertRaises(ValueError) as ctx:
                    make_line_mapper(src, tgt)
                self.assertIn(msg, str(ctx.exception))

    def test_valid_lines_produce_the_documented_mapping(self):
        self.assertEqual(
            make_line_mapper([(0, 1), (4,), (5,)], [(2, 3), (0,), (1,)]),
            {0: 2, 1: 3, 4: 0, 5: 1})


class AssignDesignsDefaultRandgenTester(BaseCase):
    """``assign_the_designs_with_mapping`` defaults ``randgen`` to ``default_rng(0)``."""

    def test_default_randgen_behavior(self):
        omitted = _stitch(3, 5, seed=None)  # no randgen/seed passed
        explicit_zero = _stitch(3, 5, seed=0)
        self.assertEqual(omitted, explicit_zero)

        # Confirms the default is a real generator being consumed
        explicit_other = _stitch(3, 5, seed=12345)
        self.assertNotEqual(omitted, explicit_other)


class ReduceByDoptTester(_SGSTFixture, BaseCase):
    """``SimultaneousGSTDesign.reduce_by_dopt``: the reason truncation was wanted.

    The kernel and the ranking are tested in test/unit/tools/test_blockdopt.py. What
    matters here is that reducing a *stitched* design gives back a well-formed
    SimultaneousGSTDesign, since that is exactly what the class used to refuse.

    The design is cut down before ranking: greedy selection costs one QR per remaining
    candidate per pick, so ranking the full 1814-circuit design against this 330-parameter
    model takes minutes.
    """

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.small = cls.design.truncate_to_circuits(list(cls.design.circuit_lists[0])[:30])
        model = pygsti.models.create_crosstalk_free_model(
            cls.pspec, ideal_gate_type='H+S', ideal_spam_type='H+S')
        # Not the target model: at the target, every cholesky-mode stochastic column of
        # the Jacobian is exactly zero. See perturb_errorgen_rates.
        cls.model = pygsti.tools.perturb_errorgen_rates(model, 1e-3, seed=0)

    def test_reducing_gives_back_a_well_formed_simultaneous_design(self):
        reduced = self.small.reduce_by_dopt(self.model, 8)
        self.assertIsInstance(reduced, SimultaneousGSTDesign)
        self.assertEqual(len(reduced.all_circuits_needing_data), 8)
        self.assertTrue(reduced.nested)
        assert_circuit_lists_match_color_patches(
            reduced.circuit_lists, reduced.vertices, reduced.color_patches)
        self.assertEqual(reduced.color_patches, self.small.color_patches)
        self.assertTrue(set(reduced.all_circuits_needing_data)
                        <= set(self.small.all_circuits_needing_data))

    def test_the_method_delegates_to_the_tools_function(self):
        self.assertEqual(
            set(self.small.reduce_by_dopt(self.model, 6).all_circuits_needing_data),
            set(pygsti.tools.reduce_design_by_dopt(
                self.small, self.model, 6).all_circuits_needing_data))

    def test_the_score_curve_comes_back_on_the_design(self):
        """`return_scores=True` used to return a tuple; the design now carries it."""
        reduced = self.small.reduce_by_dopt(self.model, 6)
        self.assertIsInstance(reduced, SimultaneousGSTDesign)
        scores = reduced.selection.scores
        self.assertEqual(len(scores), 6)
        self.assertTrue(np.all(np.diff(scores) >= -1e-9))

    def test_the_reduced_design_records_what_reduced_it(self):
        from pygsti.tools.edesign import BlockDoptReducer
        reduced = self.small.reduce_by_dopt(self.model, 6, ridge=2.0)
        self.assertIsInstance(reduced.selection.reducer, BlockDoptReducer)
        self.assertEqual(reduced.selection.reducer.ridge, 2.0)
        self.assertEqual(reduced.selection.metadata['num_candidates'],
                         len(self.small.all_circuits_needing_data))

    def test_an_unreduced_design_records_no_selection(self):
        self.assertIsNone(self.small.selection)

    def test_reduce_with_takes_any_reducer(self):
        """The point of the interface: a rule we did not write, on a stitched design."""
        shallowest = sorted(self.small.all_circuits_needing_data, key=len)[:7]
        reduced = self.small.reduce_with(lambda design, n: shallowest, 7)
        self.assertIsInstance(reduced, SimultaneousGSTDesign)
        self.assertEqual(set(reduced.all_circuits_needing_data), set(shallowest))
        assert_circuit_lists_match_color_patches(
            reduced.circuit_lists, reduced.vertices, reduced.color_patches)

    def test_reduce_by_dopt_is_reduce_with_a_dopt_reducer(self):
        from pygsti.tools.edesign import BlockDoptReducer
        self.assertEqual(
            set(self.small.reduce_by_dopt(self.model, 6).all_circuits_needing_data),
            set(self.small.reduce_with(BlockDoptReducer(self.model),
                                       6).all_circuits_needing_data))

    def test_a_target_model_warns_here_too(self):
        target = pygsti.models.create_crosstalk_free_model(
            self.pspec, ideal_gate_type='H+S', ideal_spam_type='H+S')
        with self.assertWarns(UserWarning):
            self.small.reduce_by_dopt(target, 4)

    def test_relabelling_a_reduced_design_keeps_the_record(self):
        """map_qubit_labels bypasses __init__, so `selection` has to be carried by hand."""
        reduced = self.small.reduce_by_dopt(self.model, 6)
        mapper = {q: 'Q%s' % q for q in reduced.qubit_labels}
        self.assertIs(reduced.map_qubit_labels(mapper).selection, reduced.selection)

    def test_the_reduction_beats_taking_the_first_n_circuits(self):
        """Otherwise there is no point to any of this.

        Judged by log-volume under the independent slogdet scorer, not by the
        selector's own numbers.
        """
        candidates = list(self.small.all_circuits_needing_data)
        jac, block_size = pygsti.tools.jacobian_dict_to_array(
            self.model.sim.bulk_dprobs(candidates))
        keys = list(self.model.sim.bulk_dprobs(candidates))

        chosen = set(self.small.reduce_by_dopt(self.model, 8).all_circuits_needing_data)
        greedy = pygsti.tools.greedy_path_log_volumes(
            jac.T, block_size, [i for i, c in enumerate(keys) if c in chosen])[-1]
        first_n = pygsti.tools.greedy_path_log_volumes(
            jac.T, block_size, range(8))[-1]
        self.assertGreater(greedy, first_n)

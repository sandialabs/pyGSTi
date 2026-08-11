#***************************************************************************************************
# Copyright 2015, 2019, 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

import numpy as np
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union, cast, Mapping
import tqdm as _tqdm

from pygsti.protocols.gst import GateSetTomographyDesign
from pygsti.processors import QubitProcessorSpec
from pygsti.circuits.circuit import Circuit
from pygsti.circuits.split_circuits_into_lanes import batch_tensor
from pygsti.baseobjs.label import Label, LabelTup

from pygsti.tools.graphcoloring import switchboard_find_edge_coloring

# Type aliases for the graph / stitching data structures used throughout.
Vertex = Union[int, str]
Edge = Tuple[Vertex, Vertex]
LayerMappers = Dict[int, Dict[Label, Label]]
CircuitStitcher = Callable[..., List[List[Circuit]]]
SeedLike = Union[int, np.random.SeedSequence, np.random.Generator]

# This module is star-imported into ``pygsti.protocols``, so ``__all__`` is kept
# to the documented public surface: the design class, its convenience
# constructor, the default circuit stitcher (documented as pluggable, so callers
# need to be able to name it), and the stitcher-agnostic output validator (which
# anyone writing their own stitcher is expected to run). The remaining helpers
# -- ``find_neighbors``, ``patch_lines``, ``make_line_mapper``,
# ``build_patch_infos``, ``random_index_schedule``, and friends -- are
# deliberately left out: they have generic names that would collide in the
# shared ``pygsti.protocols`` namespace, and they remain importable from this
# module directly.
__all__ = [
    'SimultaneousGSTDesign',
    'make_simultaneous_gst_design',
    'assign_the_designs_with_mapping',
    'assert_circuit_lists_match_color_patches',
]


def find_neighbors(vertices: Sequence[Vertex], edges: Sequence[Edge]) -> Dict[Vertex, List[Vertex]]:
    """
    Scan `edges` to build a dict mapping each vertex to its list of neighbors.
    """
    neighbors = {v: [] for v in vertices}
    for e in edges:
        neighbors[e[0]].append(e[1])
    return neighbors


def build_layer_mappers(oneq_gstdesign: GateSetTomographyDesign, twoq_gstdesign: GateSetTomographyDesign) -> LayerMappers:
    """
    Build the ``layer_mappers`` used by ``batch_tensor`` when stitching 1Q and 2Q
    GST circuits together.

    The returned dict maps a lane size (1 or 2) to a per-label mapper that embeds
    the 1Q/2Q labels into the full tensored state space without implicit idles.

    Any implicit idles present in these designs are mapped to Labels with
    names `Gi` and `Gii` for 1 and 2 qubits, respectively.

    Parameters
    ----------
    oneq_gstdesign : GateSetTomographyDesign
        Design containing the 1Q GST circuits.
    twoq_gstdesign : GateSetTomographyDesign
        Design containing the 2Q GST circuits.

    Returns
    -------
    dict
        ``{1: mapper_1q, 2: mapper_2q}``.
    """
    twoq_idle_label = Label(('Gii',) + twoq_gstdesign.qubit_labels)
    oneq_idle_label = Label(('Gi',)  + oneq_gstdesign.qubit_labels)
    mapper_2q: dict[Label, Label] = {twoq_idle_label: twoq_idle_label}
    mapper_1q: dict[Label, Label] = {oneq_idle_label: oneq_idle_label}
    empty_label = Label(())
    for cl in twoq_gstdesign.circuit_lists:
        for c in cl:
            mapper_2q.update({k:k for k in c._labels})
            mapper_2q[empty_label] = twoq_idle_label
    for cl in oneq_gstdesign.circuit_lists:
        for c in cl:
            mapper_1q.update({k:k for k in c._labels})
            mapper_1q[empty_label] = oneq_idle_label
    assert empty_label not in mapper_2q.values()
    assert empty_label not in mapper_1q.values()

    # Check for any labels in `mapper_2q` that imply a single-qubit target.
    # For any such label, add an explicit single-qubit idle on the non-target
    # qubit, and wrap the whole thing as a LabelTupTup.
    m2q = mapper_2q.copy()
    for k2 in mapper_2q:
        if k2.num_qubits == 1:
            assert isinstance(k2, LabelTup)
            # So we are assuming k2 = Label("Gsingle", x) where x = 0,1.
            tgt = k2[1]
            assert tgt in [0,1]
            tmp = [None, None]
            tmp[tgt] = k2
            tmp[1-tgt] = Label("Gi", 1-tgt)
            m2q[k2] = Label(tuple(tmp))

    mapper_2q = m2q # Reset here.
    # layer mappers handles how big each lane is not the length of a circuit.
    return {1: mapper_1q, 2: mapper_2q}


def make_simultaneous_gst_design(
        nq_pspec: QubitProcessorSpec,
        oneq_gstdesign: GateSetTomographyDesign,
        twoq_gstdesign: GateSetTomographyDesign,
        seed: int = 0,
        verbosity: int = 0
    ) -> "SimultaneousGSTDesign":
    """
    Build a :class:`SimultaneousGSTDesign` for `nq_pspec` without having to supply an
    edge coloring yourself.

    There are two independent random choices downstream of `seed`: which edges land in
    which patch (the coloring), and which circuit from each design lands in which lane
    slot (the stitcher). `seed` is expanded into one independent stream for each via
    ``SeedSequence.spawn``, so the two never draw from the same sequence and neither
    depends on how much randomness the other consumed.

    Note that the coloring stream is frequently unused: "auto" detects the canonical
    topologies produced by ``ProcessorSpec(geometry=...)`` (line/ring/grid/torus) and
    coloring those is a deterministic closed form using the optimal number of colors.
    The seed reaches the coloring only on the randomized bipartite path; the generic
    (deg+1)-color fallback is deterministic too.

    ``verbosity`` is forwarded to the design's circuit stitcher; anything greater
    than 0 displays a progress bar over germ powers while stitching.
    """
    vertices = cast(List[Vertex] , list(nq_pspec.qubit_labels))
    edges = nq_pspec.compute_2Q_connectivity().edges()
    edges = list(set(edges))
    neighbors = find_neighbors(vertices, edges)
    deg = max(len(neighbors[v]) for v in vertices)
    coloring_seed, stitcher_seed = np.random.SeedSequence(seed).spawn(2)
    coloring_seed = np.random.default_rng(coloring_seed)
    edge_coloring = switchboard_find_edge_coloring(
        "auto", deg, vertices, edges, neighbors, seed=coloring_seed
    )
    out = SimultaneousGSTDesign(
        nq_pspec, oneq_gstdesign, twoq_gstdesign, edge_coloring, seed=stitcher_seed,
        verbosity=verbosity
    )
    return out


class SimultaneousGSTDesign(GateSetTomographyDesign):
    """
    A *simultaneous GST* experiment design by combines 1Q and 2Q GST designs
    based on a specified edge coloring. It assumes that the GST designs share
    the same germ powers (Ls) and utilizes a specified circuit stitcher to
    generate the final circuit lists.

    Attributes:
    processor_spec: Specification of the processor, including qubit labels and connectivity.
    oneq_gstdesign: The design for one-qubit GST circuits.
    twoq_gstdesign: The design for two-qubit GST circuits.
    edge_coloring (dict): A dictionary mapping color patches to their corresponding edge sets.
    circuit_stitcher (callable): A function to stitch circuits together (default: assign_the_designs_with_mapping).
    seed (optional): Anything ``np.random.default_rng`` accepts -- an int, a SeedSequence,
        or an already-built Generator -- used to seed the randgen handed to the stitcher.
    nested (bool): Whether ``circuit_stitcher``'s output is nested, i.e. whether
        ``circuit_lists[L+1]`` contains every circuit in ``circuit_lists[L]``. The
        default stitcher always produces nested lists, hence the default of True;
        set this to False only when supplying a stitcher that does not.
    verbosity (int): Forwarded to ``circuit_stitcher``. With the default stitcher,
        anything greater than 0 displays a progress bar over germ powers.
        Defaults to 0 (silent).
    **stitcher_kwargs: Extra keyword arguments forwarded verbatim to ``circuit_stitcher``.

    circuit_lists (list): The generated list of stitched circuits.
    """
    def __init__(self, processor_spec: QubitProcessorSpec,
                 oneq_gstdesign: GateSetTomographyDesign,
                 twoq_gstdesign: GateSetTomographyDesign,
                 edge_coloring: Mapping[int, Sequence[Edge]],
                 circuit_stitcher: Optional[CircuitStitcher] = None,
                 seed: Optional[SeedLike] = None,
                 nested: bool = True,
                 debug_check: bool = True,
                 verbosity: int = 0,
                 **stitcher_kwargs: Any):
        """
        Assume that the GST designs have the same Ls.

        The default ``circuit_stitcher`` is ``assign_the_designs_with_mapping``,
        which expects the (oneq_circuitlists, twoq_circuitlists, vertices,
        color_patches, ...) calling convention used below.

        Any ``circuit_stitcher`` is invoked as::

            circuit_stitcher(oneq_gstdesign, twoq_gstdesign, vertices,
                             color_patches, randgen=..., verbosity=...,
                             **stitcher_kwargs)

        A stitcher that does not want ``verbosity`` should absorb it in its own
        ``**kwargs``, as the default stitcher does for the options it ignores.

        Extra keyword arguments to ``__init__`` are collected into ``**stitcher_kwargs``
        and forwarded verbatim, so alternative stitchers can accept their own options
        without a signature change here. Callers may also override ``randgen``
        this way, or pass the default stitcher's
        ``share_same_shape_schedules=False`` to stop same-shape patches from
        sharing one randomized schedule.

        ``nested`` is a *declaration* about ``circuit_stitcher``'s output, not a
        request: the default stitcher always denests its inputs and renests its
        output, so its lists satisfy ``circuit_lists[L] <= circuit_lists[L+1]``
        unconditionally. Pass ``nested=False`` only for a stitcher that does not
        guarantee that; it is forwarded to ``CircuitListsDesign``, which uses it to
        take ``circuit_lists[-1]`` as the full set of circuits needing data instead
        of unioning every list.

        Idle gates are guaranteed to be explicit: ``build_layer_mappers`` maps the
        empty (implicit-idle) layer label ``Label(())`` onto an explicit idle gate
        (asserting ``Label(())`` never survives into a mapper's values), and
        ``batch_tensor`` re-checks that invariant. When ``debug_check`` is True
        (the default), this constructor itself verifies the resulting
        ``circuit_lists`` via :func:`assert_circuit_lists_match_color_patches`
        -- checking that every generated circuit has no implicit idle gates and
        is correctly stitched onto its own patch's qubits/edges. This runs
        regardless of which ``circuit_stitcher`` was used (unlike the previous
        approach of relying on ``assign_the_designs_with_mapping``'s own
        ``debug_check`` parameter, which a swapped-in stitcher could silently
        skip).
        """
        if circuit_stitcher is None:
            circuit_stitcher = assign_the_designs_with_mapping
        randgen = np.random.default_rng(seed)
        self.processor_spec = processor_spec
        self.oneq_gstdesign = oneq_gstdesign
        self.twoq_gstdesign = twoq_gstdesign
        self.vertices = self.processor_spec.qubit_labels
        self.edges = self.processor_spec.compute_2Q_connectivity().edges()
        self.neighbors = find_neighbors(self.vertices, self.edges)
        self.deg = max([len(self.neighbors[v]) for v in self.vertices])
        self.color_patches = edge_coloring
        self.circuit_stitcher = circuit_stitcher

        # Base kwargs common to the built-in calling convention; caller-supplied
        # stitcher_kwargs take precedence so any option can be overridden.
        kwargs = dict(randgen=randgen, verbosity=verbosity)
        kwargs.update(stitcher_kwargs)
        self.stitcher_kwargs = kwargs

        self.circuit_lists = circuit_stitcher(
            self.oneq_gstdesign, self.twoq_gstdesign, self.vertices, self.color_patches, **kwargs,
        )


        if debug_check:
            # Stitcher-agnostic verification of circuit_lists: runs no matter
            # which circuit_stitcher produced it.
            assert_circuit_lists_match_color_patches(
                self.circuit_lists, self.vertices, self.color_patches
            )

        super().__init__(processor_spec, self.circuit_lists,qubit_labels=self.vertices, nested=nested)


def patch_lines(edge_set: Sequence[Edge],
                vertices: Sequence[Vertex]) -> Tuple[List[Edge], List[Vertex], List[Union[Edge, Tuple[Vertex]]]]:
    """
    Return the ordered tensor lines for a patch:
      - first the 2Q edge lines
      - then the 1Q unused-qubit lines
    """
    edge_set = sorted([tuple(edge) for edge in edge_set])
    used_qubits    = {q for edge in edge_set for q in edge}
    unused_qubits  = [q for q in vertices if q not in used_qubits]
    tensored_lines = list(edge_set) + [(q,) for q in unused_qubits]
    return edge_set, unused_qubits, tensored_lines


def make_line_mapper(source_lines: Sequence[Edge],
                     target_lines: Sequence[Edge]) -> Dict[Vertex, Vertex]:
    """
    Construct a state-space-label mapper from source tensor lines to target
    tensor lines.

    Example:
        source_lines = [(0, 1), (4,), (5,)]
        target_lines = [(2, 3), (0,), (1,)]

        returns:
            {0: 2, 1: 3, 4: 0, 5: 1}
    """
    if len(source_lines) != len(target_lines):
        raise ValueError("Source and target line lists have different lengths.")

    mapper = {}

    for src_line, dst_line in zip(source_lines, target_lines):
        if len(src_line) != len(dst_line):
            raise ValueError(
                f"Line arity mismatch: source {src_line}, target {dst_line}"
            )

        for src_label, dst_label in zip(src_line, dst_line):
            if src_label in mapper and mapper[src_label] != dst_label:
                raise ValueError(
                    f"Inconsistent mapping for {src_label}: "
                    f"{mapper[src_label]} versus {dst_label}"
                )

            mapper[src_label] = dst_label

    if len(set(mapper.values())) != len(mapper):
        raise ValueError("Mapper is not one-to-one.")

    return mapper


def build_patch_infos(vertices: Sequence[Vertex],
                      color_patches: Dict[int, List[Edge]]
                      ) -> List[Dict[str, Any]]:
    """
    Describe each color patch's geometry, in ``color_patches`` order.

    Parameters
    ----------
    vertices : list
        All qubits on the device.

    color_patches : dict[int, list[tuple]]
        Mapping from patch identifier to the list of edges in that patch.

    Returns
    -------
    list[dict]
        One info dict per patch, carrying its edges, its unused (1Q) qubits,
        the tensored line ordering, and the counts of each. Ordering follows
        ``color_patches``, which is what fixes the patch-major output order.
    """
    vertices = list(vertices)

    patch_infos = []

    for patch, edge_set in color_patches.items():
        edge_set, unused_qubits, tensored_lines = patch_lines(edge_set, vertices)

        info = {
            "patch": patch,
            "edge_set": edge_set,
            "unused_qubits": unused_qubits,
            "tensored_lines": tensored_lines,
            "num_edges": len(edge_set),
            "num_unused_qubits": len(unused_qubits),
        }

        patch_infos.append(info)

    return patch_infos


def group_patches_for_scheduling(patch_infos: List[Dict[str, Any]],
                                 share_same_shape_schedules: bool = True
                                 ) -> List[List[Dict[str, Any]]]:
    """
    Partition patches into *scheduling groups*: sets of patches that share a
    single random circuit-index schedule.

    One schedule is drawn per group, and one template circuit is tensored per
    group per simultaneous circuit; every non-representative member of the
    group obtains its circuits by relabelling that template onto its own lines.
    So the grouping alone decides whether two patches get identical circuit
    content (up to qubit relabelling) or independent draws.

    Parameters
    ----------
    patch_infos : list[dict]
        Patch infos from :func:`build_patch_infos`, in ``color_patches`` order.

    share_same_shape_schedules : bool, optional
        If True (the default), patches with the same *shape* -- the same number
        of 2Q edge slots and the same number of unused 1Q qubit slots -- are
        placed in one group and therefore share a schedule. If False, every
        patch becomes its own singleton group and draws independently.

    Returns
    -------
    list[list[dict]]
        The groups, each a non-empty list of patch infos whose first element is
        the group's representative. Groups are ordered by the first appearance
        of a member in ``patch_infos``, so the draw order (and hence the output
        for a given seed) is deterministic.

    Notes
    -----
    A singleton group is exactly the degenerate case of a shared one: its
    representative is the patch itself and there are no other members to
    relabel onto. That is why ``share_same_shape_schedules=False`` needs no
    separate code path downstream.
    """
    if not share_same_shape_schedules:
        return [[info] for info in patch_infos]

    groups: Dict[Tuple[int, int], List[Dict[str, Any]]] = defaultdict(list)
    for info in patch_infos:
        groups[(info["num_edges"], info["num_unused_qubits"])].append(info)

    # dict preserves insertion order, i.e. first appearance of each shape.
    return list(groups.values())


def random_index_schedule(n: int, num_circs_at_germ_power: int, randgen: np.random.Generator) -> np.ndarray:
    """
    A length-``num_circs_at_germ_power`` index schedule into a CircuitList of
    size ``n``.

    This is what fills a single slot (one edge, or one unused qubit) across all
    ``num_circs_at_germ_power`` simultaneous circuits generated for one germ
    power: entry ``j`` says which circuit of the design goes into that slot in
    the ``j``-th simultaneous circuit.

    The ``n`` real indices ``0..n-1`` are always included, so every circuit of
    the design is used at least once. If ``n < num_circs_at_germ_power`` the
    design is bootstrapped up to ``num_circs_at_germ_power``: the remaining
    ``num_circs_at_germ_power - n`` entries are drawn uniformly with replacement
    from ``0..n-1``. The whole schedule is then shuffled, so the design's own
    (arbitrary) circuit order carries no meaning.

    When ``n == num_circs_at_germ_power`` no bootstrapping happens and the
    result is a plain random permutation of ``0..n-1``.

    Parameters
    ----------
    n : int
        Size of the CircuitList being scheduled into ``num_circs_at_germ_power`` slots.

    num_circs_at_germ_power : int
        Desired length of the returned schedule.

    randgen : numpy.random.Generator
        Random number generator used to draw the bootstrap indices (when
        ``n < num_circs_at_germ_power``) and to shuffle the result.

    Returns
    -------
    numpy.ndarray
        A length-``num_circs_at_germ_power`` array of indices into ``0..n-1``.
    """
    if n == num_circs_at_germ_power:
        base = np.arange(num_circs_at_germ_power)
    else:
        base = np.concatenate((
            np.arange(n),
            randgen.integers(0, n, size=num_circs_at_germ_power - n),
        ))
    return randgen.permutation(base)

#region Invariant Helpers

def assert_no_implicit_idles(circuit: Circuit) -> None:
    """
    Assert that every idle gate in ``circuit`` is explicit.

    For every layer, checks that the layer's contents are unchanged by
    filling in idles (``layer(i) == layer_with_idles(i)``), i.e. that the
    circuit contains no implicit idle gates.

    Parameters
    ----------
    circuit : Circuit
        The circuit to check.

    Returns
    -------
    None

    Raises
    ------
    AssertionError
        If any layer of ``circuit`` contains an implicit idle gate.
    """
    for i in range(circuit.num_layers):
        l0 = set(circuit.layer(i))
        l1 = set(circuit.layer_with_idles(i))
        assert l0 == l1, (
            f"Implicit idle gate(s) detected in layer {i}: "
            f"layer()={l0} != layer_with_idles()={l1}"
        )


def assert_mapped_circuit_matches_patch(mapped_circuit: Circuit, info: Dict[str, Any]) -> None:
    """
    Assert that ``mapped_circuit`` was correctly remapped onto its own patch.

    Checks two things:

    1. ``mapped_circuit``'s line labels are exactly the qubits/edges assigned
       to this patch (``info["tensored_lines"]``) -- no more, no fewer.
    2. Every multi-qubit gate in ``mapped_circuit`` lands on one of this
       patch's own edges (``info["edge_set"]``, in either orientation) --
       not on some other patch's edge (e.g. the representative patch's edge,
       which would indicate the mapper was applied incorrectly or never
       applied at all).

    Parameters
    ----------
    mapped_circuit : Circuit
        The circuit produced by mapping a representative patch's template
        circuit onto ``info``'s own qubits/edges.

    info : dict
        One patch's entry from ``build_patch_infos``, containing at least
        ``"patch"``, ``"tensored_lines"``, and ``"edge_set"``.

    Returns
    -------
    None

    Raises
    ------
    AssertionError
        If either check fails.
    """
    expected_labels = {
        q
        for line in info["tensored_lines"]
        for q in line
    }

    actual_labels = set(mapped_circuit.line_labels)

    assert actual_labels == expected_labels, (
        actual_labels,
        expected_labels
    )

    # Also verify *where* the multi-qubit gates actually landed,
    allowed_edges = {tuple(e) for e in info["edge_set"]}
    allowed_edges |= {tuple(reversed(e)) for e in allowed_edges}
    for i in range(mapped_circuit.num_layers):
        for op in mapped_circuit.layer(i):
            if len(op.qubits) > 1:
                assert tuple(op.qubits) in allowed_edges, (
                    f"Patch {info['patch']!r}: found multi-qubit "
                    f"gate {op} on {op.qubits}, which is not one "
                    f"of this patch's own edges {info['edge_set']} "
                    "(mapper likely applied incorrectly, or the "
                    "circuit was never remapped from the "
                    "representative patch)."
                )


def assert_circuit_lists_match_color_patches(
    circuit_lists: List[List[Circuit]],
    vertices: Sequence[Vertex],
    color_patches: Dict[int, List[Edge]],
) -> None:
    """
    Assert that ``circuit_lists`` is a well-formed, patch-major stitching of
    ``color_patches`` onto ``vertices``.

    This is stitcher-agnostic: it validates the *output* of whatever
    ``circuit_stitcher`` produced ``circuit_lists``, not just the built-in
    ``assign_the_designs_with_mapping``, so it can (and is, by
    ``SimultaneousGSTDesign.__init__``) be run regardless of which
    stitcher was actually used.

    For every germ-power entry ``circuit_lists[L]``, this re-derives each
    patch's own tensored lines/edges from ``vertices``/``color_patches`` (the
    same way ``build_patch_infos`` does) and checks that:

    1. ``circuit_lists[L]`` splits evenly into ``len(color_patches)``
       contiguous, equal-size, patch-major chunks -- i.e. the output honors
       the germ-power-major-then-patch-major ordering contract documented on
       ``assign_the_designs_with_mapping`` (this is required of *any*
       ``circuit_stitcher``, not just the built-in one). Note this must hold
       for nested output too, which is why the built-in stitcher renests
       patch-wise rather than concatenating whole germ-power lists.
    2. Every circuit has no implicit idle gates
       (see :func:`assert_no_implicit_idles`).
    3. Every circuit in a given patch's chunk is correctly stitched onto
       that patch's own qubits/edges
       (see :func:`assert_mapped_circuit_matches_patch`).

    Checks 2 and 3 inspect the *denested* content: each distinct circuit is
    checked once per patch, and repeats are skipped. Both are pure functions of
    the circuit and its patch, so re-checking a circuit yields no new
    information -- but ``circuit_lists`` is normally nested, meaning germ power
    ``L``'s chunk repeats germ powers ``0..L-1``, so a naive pass would re-walk
    every layer of germ power 0's circuits once per germ power.

    Check 1 is a length check and stays per-germ-power: it is O(1) per list, and
    it is precisely the nested chunk structure that it exists to verify.

    Parameters
    ----------
    circuit_lists : list[list[Circuit]]
        The stitched circuit lists to check, e.g. ``self.circuit_lists`` on a
        ``SimultaneousGSTDesign``.

    vertices : list[Vertex]
        Vertices/qubits in the connectivity graph.

    color_patches : dict[int, list[tuple]]
        Mapping from patch/color identifier to the list of disjoint 2Q edges
        in that patch, as passed to ``SimultaneousGSTDesign``.

    Returns
    -------
    None

    Raises
    ------
    AssertionError
        If any of the checks above fail.
    """
    patch_infos = build_patch_infos(vertices, color_patches)
    num_patches = len(patch_infos)

    # Circuits already checked, per patch. Nested circuit_lists repeat every
    # earlier germ power's circuits, so without this the per-circuit checks
    # below would redo the same work O(num_germ_powers) times.
    checked: List[set] = [set() for _ in patch_infos]

    for circuit_list in circuit_lists:
        assert len(circuit_list) % num_patches == 0, (
            f"Expected {len(circuit_list)} circuits to split evenly into "
            f"{num_patches} patch-major chunks (one per color patch)."
        )
        chunk_size = len(circuit_list) // num_patches

        for patch_idx, info in enumerate(patch_infos):
            start = patch_idx * chunk_size
            already_checked = checked[patch_idx]
            for circuit in circuit_list[start:start + chunk_size]:
                if circuit in already_checked:
                    continue
                assert_no_implicit_idles(circuit)
                assert_mapped_circuit_matches_patch(circuit, info)
                already_checked.add(circuit)
#endregion

def build_group_schedules(
    num_edges: int,
    num_unused_qubits: int,
    num_circs_at_germ_power: int,
    twoq_len: int,
    oneq_len: int,
    randgen: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build the random circuit-index schedules for one patch-shape group at one
    germ power.

    A patch of this shape has ``num_edges`` 2Q slots and ``num_unused_qubits``
    1Q slots. This draws one independent schedule per slot, each of length
    ``num_circs_at_germ_power``, so that ``twoq_slot_schedules[s, j]`` names the
    2Q circuit occupying edge slot ``s`` of the ``j``-th simultaneous circuit
    (and likewise for the 1Q slots). Whichever design is shorter at this germ
    power is bootstrapped up to ``num_circs_at_germ_power``; see
    :func:`random_index_schedule`.

    Independent draws per slot are the point: they are what make different edges
    of the same patch run different 2Q circuits simultaneously.

    Parameters
    ----------
    num_edges : int
        Number of 2Q edge slots in this patch shape.

    num_unused_qubits : int
        Number of unused (1Q) qubit slots in this patch shape.

    num_circs_at_germ_power : int
        Number of simultaneous circuits to generate for this germ power.

    twoq_len : int
        Number of available 2Q circuits at this germ power.

    oneq_len : int
        Number of available 1Q circuits at this germ power.

    randgen : numpy.random.Generator
        Random number generator used for the index schedules.

    Returns
    -------
    twoq_slot_schedules : numpy.ndarray, shape (num_edges, num_circs_at_germ_power)
        Which 2Q circuit fills each edge slot, for each simultaneous circuit.

    oneq_slot_schedules : numpy.ndarray, shape (num_unused_qubits, num_circs_at_germ_power)
        Which 1Q circuit fills each unused-qubit slot, for each simultaneous
        circuit.
    """
    twoq_slot_schedules = np.empty((num_edges, num_circs_at_germ_power), dtype=np.int64)
    for edge_slot in range(num_edges):
        twoq_slot_schedules[edge_slot, :] = random_index_schedule(twoq_len, num_circs_at_germ_power, randgen)

    oneq_slot_schedules = np.empty((num_unused_qubits, num_circs_at_germ_power), dtype=np.int64)
    for qubit_slot in range(num_unused_qubits):
        oneq_slot_schedules[qubit_slot, :] = random_index_schedule(oneq_len, num_circs_at_germ_power, randgen)

    return twoq_slot_schedules, oneq_slot_schedules


def build_patch_mappers(infos: List[Dict[str, Any]]) -> Dict[int, Optional[Dict[Vertex, Vertex]]]:
    """
    Build the line mappers for one scheduling group (see
    :func:`group_patches_for_scheduling`).

    Patches sharing a group are stitched only once: a template circuit is
    tensored for the group's *representative* patch (``infos[0]``), and every
    other patch obtains its circuit by relabelling that template onto its own
    lines rather than re-tensoring from scratch. For a singleton group this
    returns just ``{patch: None}``, i.e. no relabelling at all.

    The mappers depend only on the patch geometry, not on the germ power or on
    any random draw, so they can be built once and reused for every germ power.

    Parameters
    ----------
    infos : list[dict]
        One scheduling group's patch infos, as produced by
        :func:`group_patches_for_scheduling`. ``infos[0]`` is the
        representative.

    Returns
    -------
    dict[int, Optional[dict]]
        Mapping from patch identifier to the line mapper carrying the
        representative's template circuit onto that patch. ``None`` for the
        representative itself, since it needs no relabelling.
    """
    representative = infos[0]
    representative_lines = representative["tensored_lines"]

    mappers: Dict[int, Optional[Dict[Vertex, Vertex]]] = {}
    for info in infos:
        if info is representative:
            mappers[info["patch"]] = None
        else:
            mappers[info["patch"]] = make_line_mapper(
                representative_lines,
                info["tensored_lines"]
            )

    return mappers


def flatten_patch_major(
    patch_buffers: Dict[int, List[Circuit]],
    patch_order: List[int],
) -> List[Circuit]:
    """
    Flatten one germ power's per-patch circuit buffers into a single list,
    patch-major.

    "Patch-major" means all of the first patch's circuits, then all of the
    second patch's, and so on, following ``patch_order``. Combined with the
    outer per-germ-power list, this gives the germ-power-major-then-patch-major
    ordering that :func:`assert_circuit_lists_match_color_patches` checks.

    Parameters
    ----------
    patch_buffers : dict[int, list[Circuit]]
        This germ power's circuits, keyed by patch identifier.

    patch_order : list[int]
        Patch identifiers, in the desired output order.

    Returns
    -------
    list[Circuit]
        This germ power's circuits, ordered patch-major.
    """
    output_circuits: List[Circuit] = []
    for patch in patch_order:
        output_circuits.extend(patch_buffers[patch])
    return output_circuits


def _denest_a_circuitlist(circuitlist: list[list[Circuit]]) -> list[list[Circuit]]:
    r"""
        Assume circuitlist is a list of lists of circuits which are ordered by germ power.
        Remove any circuits which were duplicated in a previous inner list.

        x[a[z], b[z], c[z], ...] -> x[a[z], b[z] \ a[z], c[z] \ (b[z] | a[z]), ...]
    """
    cop = [[] for _ in range(len(circuitlist))]
    if not circuitlist:
        return cop

    cop[0] = list(circuitlist[0])
    seen = set(cop[0])

    for i in range(1, len(circuitlist)):
        for circ in circuitlist[i]:
            if circ not in seen:
                cop[i].append(circ)
                seen.add(circ)
    return cop


def _nest_a_circuitlist(circuitlist: list[list[Circuit]], num_patches: int = 1) -> list[list[Circuit]]:
    r"""
        Undo :func:`_denest_a_circuitlist`, i.e. accumulate each germ power's list
        onto all the earlier ones::

            x[a[z], b[z] \ a[z], c[z] \ (b[z] | a[z]), ...] -> x[a[z], b[z], c[z], ...]

        where b[z] contains a[z], c[z] contains b[z] and so on.

        Accumulation is done *per patch* so that the germ-power-major-then-patch-major
        ordering survives. Each inner list is assumed to consist of ``num_patches``
        contiguous, equal-size, patch-major chunks; chunk ``p`` of the output at germ
        power ``i`` is the concatenation of chunk ``p`` of the inputs at germ powers
        ``0..i``. Naively concatenating whole germ-power lists instead would interleave
        patches (``[P0 P1][P0 P1]``) and break that contract.

        With ``num_patches == 1`` this is exactly the plain accumulation above.
    """
    cop = [[] for _ in range(len(circuitlist))]
    if not circuitlist:
        return cop

    chunk_sizes = []
    for i, lst in enumerate(circuitlist):
        if len(lst) % num_patches != 0:
            raise ValueError(
                f"Germ power {i} has {len(lst)} circuits, which does not split evenly "
                f"into {num_patches} patch-major chunks. Every patch must contribute "
                "the same number of circuits at each germ power."
            )
        chunk_sizes.append(len(lst) // num_patches)

    for i in range(len(circuitlist)):
        accumulated: List[Circuit] = []
        for patch_idx in range(num_patches):
            for j in range(i + 1):  # Since i < len(circuitlist) this is fine.
                start = patch_idx * chunk_sizes[j]
                accumulated.extend(circuitlist[j][start:start + chunk_sizes[j]])
        cop[i] = accumulated
    return cop


def assign_the_designs_with_mapping(
    oneq_gstdesign: GateSetTomographyDesign,
    twoq_gstdesign: GateSetTomographyDesign,
    vertices: Sequence[Vertex],
    color_patches: Dict[int, List[Edge]],
    randgen: Optional[np.random.Generator] = None,
    share_same_shape_schedules: bool = True,
    verbosity: int = 0,
    **kwargs: Any,
) -> List[List[Circuit]]:
    """
    Given a 1Q GST design, a 2Q GST design, and an edge-colored graph of the topology of the
    processor, construct a simultaneous GST design which runs the 2Q design on every edge
    of the processor's topology. This helper function produces a list of lists of simultaneous circuits
    which have a gate prescribed for every qubit at every layer.

    -------- Intro --------
    The input color patches indicate which sets of edges can run a 2Q design simultaneously since
    they do not share a vertex and thus do not share a qubit. Therefore, for a given color patch,
    we can run a 2Q design on the qubits specified by the vertices of our processor's topology graph.
    On the other vertices we will run a 1Q design so as to not have the other qubits be exclusively idle.
    GST designs contain CircuitLists which are sorted by germ power see `gst.py` for more details
    Importantly for this context, the order in which the circuits are executed for a particular GST design is arbitrary.

    ----- Simultaneous GST circuit ordering -----
    Duplicate the 2Q design for each edge in the color patch
    For each germ power in the CircuitList choose a random permutation of the circuits at that germ power to be executed
    on that particular pair of qubits.

    Duplicate the 1Q design for each qubit not specified by an edge in the color patch
    For each germ power in the CircuitList choose a random permutation of the circuits at that germ power to be executed
    on that particular qubit.    

    ---- Example ----

    Imagine A and B are your only circuits for 1Q GST at germ power 1 and C, D are the two qubit options for 2Q GST at germ power 1.
    Then, for a 5 qubit line processor  0-1-2-3-4, and a coloring of [(0,1), (2,3)] we could have as one possible CircuitList for the full 5Q line topology:

    0 -- D --   | 0 -- C --
    1 -- D --   | 1 -- C --
    2 -- C --   | 2 -- D --
    3 -- C --   | 3 -- D --
    4 -- A --   | 4 -- B --

    or

    0 -- C --   | 0 -- D --
    1 -- C --   | 1 -- D --
    2 -- C --   | 2 -- D --
    3 -- C --   | 3 -- D --
    4 -- B --   | 4 -- A --

    A different way to view this would be each simultaneous circuit has a slot for each subcircuit to use. In the first case,
    the first simultaneous circuit we chose (D,C,A) for slot 0 and (C,D,B) for slot 1.

    ----- Output ordering -----
    The returned lists are indexed by germ power, and within a germ power the
    circuits are grouped by patch: all of the first patch's circuits, then all of
    the second patch's, and so on, following the input order of ``color_patches``.
    We call this germ-power-major, then patch-major. Every patch contributes the
    same number of circuits at a given germ power, so each germ power's list splits
    into equal contiguous per-patch chunks.

    ----- Nesting -----
    The input designs' CircuitLists are assumed to be nested (GST's usual
    convention: germ power L+1's list contains germ power L's). They are denested
    on the way in, so each germ power is stitched from only its own new circuits,
    and the result is renested on the way out. Renesting is done patch-wise, so the
    germ-power-major-then-patch-major ordering above is preserved: patch p's chunk
    at germ power L is patch p's circuits from germ powers 0..L, in order. Nesting
    is therefore about set containment, not about the order of the flat list.

    ----- Randomization across patches -----
    By default (``share_same_shape_schedules=True``) patches with the same shape --
    the same number of 2Q edge slots and unused 1Q qubit slots -- share a single
    random schedule: one is stitched and the rest are relabellings of it onto their
    own qubits. So on a 5Q line with patches ``[(0,1),(2,3)]`` and ``[(1,2),(3,4)]``,
    patch 1's circuits are patch 0's circuits shifted over by one qubit, slot for
    slot. Randomness is drawn once per shape, not once per patch, which makes the
    spectator context each subcircuit sees correlated across same-shape patches; the
    payoff is that tensoring cost scales with the number of distinct shapes rather
    than the number of patches.

    Set ``share_same_shape_schedules=False`` for independent draws per patch, at a
    tensoring cost of roughly ``num_patches / num_shapes`` times the default. Note
    the two settings consume the random stream at different rates, so their outputs
    are unrelated at a fixed seed and should not be diffed against each other.

    ---------- Notes -------------
    - If either the 2Q GST or the 1Q GST has a germ power which contains more circuits than that of the other GST design then for
    either the edges (in the case 1Q has more circuits) or (the unused qubits in the case 2Q has more circuits) will be bootstrapped to the number of circuits
    specified by the other design for that particular germ power. That is, the shorter design's circuits are resampled with replacement
    (after each is used once) to fill the extra slots. This could be different for different germ powers.

    - A full simultaneous circuit will always will pad swallower subcircuits with noisy idle gates to the length of the longest subcircuit.
    
    - This function does not deduplicate color patches. For example, if both
    ``[(0, 1), (2, 3)]`` and ``[(1, 0), (3, 2)]`` are supplied, both designs are
    generated, even though they differ only by edge orientation.

    - This function does not verify its own output (e.g. that no implicit idle
    gates remain, or that circuits landed on the correct patch). That
    verification is stitcher-agnostic and lives in
    :func:`assert_circuit_lists_match_color_patches`, which
    ``SimultaneousGSTDesign.__init__`` runs (by default) against
    whatever this or any other ``circuit_stitcher`` returns.

    Parameters
    ----------
    oneq_gstdesign : GateSetTomographyDesign
        The 1Q GST experiment design.

    twoq_gstdesign : GateSetTomographyDesign
        The 2Q GST experiment design. Must have the same number of germ-power
        groups as ``oneq_gstdesign``.

    vertices : list[int]
        Vertices/qubits in the connectivity graph.

    color_patches : dict[int, list[tuple[int, int]]]
        Mapping from patch/color identifier to the list of disjoint 2Q edges in that patch.
        Each edge is represented as a pair of qubit labels.

    randgen : numpy.random.Generator, optional
        Random number generator used to randomize circuit assignments across edge
        and qubit slots. If None, uses ``np.random.default_rng(0)``.

    share_same_shape_schedules : bool, optional
        Whether patches with the same shape share one random schedule (and hence
        receive identical circuit content up to qubit relabelling). Defaults to
        True. See "Randomization across patches" above for the tradeoff.

    verbosity : int, optional
        If greater than 0, display a progress bar over germ powers while
        stitching. Defaults to 0 (silent), so that library calls and test
        suites produce no output.

    **kwargs
        Ignored. Accepted so this stitcher matches the generic
        ``circuit_stitcher(oneq, twoq, vertices, color_patches, **kwargs)``
        calling convention used by ``SimultaneousGSTDesign``, allowing
        it to be swapped with other stitchers that take extra options.

    Returns
    -------
    list[list]
        ``circuit_lists[L]`` contains the generated simultaneous GST circuits for
        germ-power index ``L``. Within each germ-power group, circuits are ordered
        patch-major according to the input order of ``color_patches``. The lists are
        nested: ``circuit_lists[L+1]`` contains every circuit in ``circuit_lists[L]``.

    Raises
    ------
    NotImplementedError
        If ``oneq_gstdesign`` and ``twoq_gstdesign`` do not have the same number
        of germ-power groups. The two designs are stitched germ power by germ
        power, so pairing designs with differing numbers of germ powers is not
        supported; truncate the longer design (or rebuild both with the same
        ``max_lengths``) before calling.
    """
    if randgen is None:
        randgen = np.random.default_rng(0)

    oneq_gstdesign_circuitlists = oneq_gstdesign.circuit_lists
    twoq_gstdesign_circuitlists = twoq_gstdesign.circuit_lists
    layer_mappers = build_layer_mappers(oneq_gstdesign, twoq_gstdesign)
    # Denest the Circuit lists. We will renest them at the end.
    oneq_gstdesign_circuitlists = _denest_a_circuitlist(oneq_gstdesign_circuitlists)
    twoq_gstdesign_circuitlists = _denest_a_circuitlist(twoq_gstdesign_circuitlists)

    if len(oneq_gstdesign_circuitlists) != len(twoq_gstdesign_circuitlists):
        raise NotImplementedError(
            "The 1Q and 2Q designs must have the same number of germ powers, but got "
            f"{len(oneq_gstdesign_circuitlists)} (1Q) versus "
            f"{len(twoq_gstdesign_circuitlists)} (2Q). The designs are stitched germ "
            "power by germ power, so pairing designs of differing lengths is not "
            "supported; truncate the longer design (or rebuild both with the same "
            "max_lengths) before calling."
        )

    vertices = list(vertices)

    patch_infos = build_patch_infos(vertices, color_patches)

    # Preserve user/color_patches ordering in the final output.
    patch_order = [info["patch"] for info in patch_infos]

    # Which patches share a schedule (and hence a tensored template circuit).
    schedule_groups = group_patches_for_scheduling(
        patch_infos, share_same_shape_schedules
    )

    # Line mappers depend only on patch geometry, not on the germ power or on any
    # random draw, so build them once here rather than inside the germ-power loop.
    # Parallel to schedule_groups; empty-but-for-the-representative when patches
    # do not share schedules.
    group_mappers = [build_patch_mappers(infos) for infos in schedule_groups]

    circuit_lists: List[List[Circuit]] = [[] for _ in twoq_gstdesign_circuitlists]

    for L, (oneq_circuits, twoq_circuits) in _tqdm.tqdm(
        enumerate(zip(oneq_gstdesign_circuitlists, twoq_gstdesign_circuitlists)),
        total=len(twoq_gstdesign_circuitlists),
        disable=(verbosity <= 0), desc="Building Simultaneous Circuits"
    ):
        oneq_len = len(oneq_circuits)
        twoq_len = len(twoq_circuits)

        num_circs_at_germ_power = max(oneq_len, twoq_len)

        # Each generated simultaneous circuit at this germ power draws one 2Q circuit
        # per edge slot and one 1Q circuit per unused-qubit slot. We produce
        # ``num_circs_at_germ_power`` simultaneous circuits so that every circuit in the *longer* of
        # the two CircuitLists is used exactly once; the shorter CircuitList is
        # bootstrapped up to ``num_circs_at_germ_power`` by resampling from itself with replacement.
        # Circuit *depths* are equalized separately: batch_tensor pads each shorter
        # sub-circuit with explicit idle layers (via the ``Label(()) ->
        # explicit-idle`` entry in ``layer_mappers``). This works regardless of which
        # CircuitList is longer.

        # Temporary per-patch storage so output ordering remains patch-major.
        patch_buffers = {
            info["patch"]: []
            for info in patch_infos
        }

        for infos, mappers in zip(schedule_groups, group_mappers):
            representative = infos[0]
            representative_lines = representative["tensored_lines"]

            twoq_slot_schedules, oneq_slot_schedules = build_group_schedules(
                representative["num_edges"], representative["num_unused_qubits"],
                num_circs_at_germ_power, twoq_len, oneq_len, randgen
            )

            for j in range(num_circs_at_germ_power):
                circs_to_tensor = [twoq_circuits[idx] for idx in twoq_slot_schedules[:, j]]
                circs_to_tensor += [oneq_circuits[idx] for idx in oneq_slot_schedules[:, j]]

                template_circuit = batch_tensor(
                    circs_to_tensor,
                    layer_mappers,
                    None,
                    representative_lines
                )

                patch_buffers[representative["patch"]].append(
                    template_circuit.copy()
                )

                for info in infos[1:]:
                    mapper = mappers[info["patch"]]

                    mapped_circuit = template_circuit.map_state_space_labels(mapper)
                    patch_buffers[info["patch"]].append(mapped_circuit)

        # Preserve patch-major output ordering.
        circuit_lists[L] = flatten_patch_major(patch_buffers, patch_order)

    # Renest patch-wise, so germ-power-major-then-patch-major ordering survives.
    return _nest_a_circuitlist(circuit_lists, num_patches=len(patch_order))





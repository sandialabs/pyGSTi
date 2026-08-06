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
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union, cast
import tqdm as _tqdm

from pygsti.protocols.gst import GateSetTomographyDesign
from pygsti.processors import QubitProcessorSpec
from pygsti.circuits.circuit import Circuit
from pygsti.circuits.split_circuits_into_lanes import batch_tensor
from pygsti.baseobjs.label import Label, LabelTup

from pygsti.tools.graphcoloring import switchboard_find_edge_coloring

# Type aliases for the graph / stitching data structures used throughout.
Vertex = Union[int, str]
Edge = Tuple[Vertex, ...]
LayerMappers = Dict[int, Dict[Label, Label]]
CircuitStitcher = Callable[..., List[List[Circuit]]]

__all__ = ['CrosstalkFreeExperimentDesign', 'make_xfgst_design']


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
    oneq_idle_label = Label(('Gi',) + oneq_gstdesign.qubit_labels)
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


def make_xfgst_design(
        nq_pspec: QubitProcessorSpec,
        oneq_gstdesign: GateSetTomographyDesign,
        twoq_gstdesign: GateSetTomographyDesign,
        seed: int = 0
    ) -> "CrosstalkFreeExperimentDesign":
    vertices = cast(List[Vertex] , list(nq_pspec.qubit_labels))
    edges = nq_pspec.compute_2Q_connectivity().edges()
    edges = list(set(edges))
    neighbors = find_neighbors(vertices, edges)
    deg = max(len(neighbors[v]) for v in vertices)
    edge_coloring = switchboard_find_edge_coloring("auto", deg, vertices, edges, neighbors, seed=seed)
    # ^ "auto" detects canonical topologies (line/ring/grid/torus, as produced by
    #   ProcessorSpec(geometry=...)) and uses an optimal closed-form coloring for
    #   them, falling back to a generic (deg+1)-color algorithm otherwise.
    return CrosstalkFreeExperimentDesign(nq_pspec, oneq_gstdesign, twoq_gstdesign, edge_coloring, seed=(seed+1))


class CrosstalkFreeExperimentDesign(GateSetTomographyDesign):
    """
    This class initializes a crosstalk-free GST experiment design by combining 
    1Q and 2Q GST designs based on a specified edge coloring. It assumes that 
    the GST designs share the same germ powers (Ls) and utilizes a specified 
    circuit stitcher to generate the final circuit lists.

    Attributes:
    processor_spec: Specification of the processor, including qubit labels and connectivity.
    oneq_gstdesign: The design for one-qubit GST circuits.
    twoq_gstdesign: The design for two-qubit GST circuits.
    edge_coloring (dict): A dictionary mapping color patches to their corresponding edge sets.
    circuit_stitcher (callable): A function to stitch circuits together (default: assign_the_designs_with_mapping).
    seed (int, optional): Seed for random number generation.
    **stitcher_kwargs: Extra keyword arguments forwarded verbatim to ``circuit_stitcher``.

    circuit_lists (list): The generated list of stitched circuits.
    aux_info (dict): Auxiliary information mapping circuits to their corresponding edges and vertices.
    """
    def __init__(self, processor_spec: QubitProcessorSpec,
                 oneq_gstdesign: GateSetTomographyDesign,
                 twoq_gstdesign: GateSetTomographyDesign,
                 edge_coloring: Dict[int, List[Edge]],
                 circuit_stitcher: Optional[CircuitStitcher] = None,
                 seed: Optional[int] = None,
                 nested: bool = False,
                 debug_check: bool = True,
                 **stitcher_kwargs: Any):
        """
        Assume that the GST designs have the same Ls.

        The default ``circuit_stitcher`` is ``assign_the_designs_with_mapping``,
        which expects the (oneq_circuitlists, twoq_circuitlists, vertices,
        color_patches, ...) calling convention used below.

        Any ``circuit_stitcher`` is invoked as::

            circuit_stitcher(oneq_gstdesign, twoq_gstdesign, vertices,
                             color_patches, randgen=..., ensure_containment=nested,
                             **stitcher_kwargs)

        Extra keyword arguments to ``__init__`` are collected into ``**stitcher_kwargs``
        and forwarded verbatim, so alternative stitchers can accept their own options
        without a signature change here. Callers may also override
        ``randgen``/``ensure_containment`` this way.

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
        kwargs = dict(randgen=randgen, ensure_containment=nested)
        kwargs.update(stitcher_kwargs)
        self.stitcher_kwargs = kwargs

        self.circuit_lists = circuit_stitcher(
            self.oneq_gstdesign, self.twoq_gstdesign, self.vertices, self.color_patches, **kwargs,
        )
        # The default stitcher (assign_the_designs_with_mapping) does not produce
        # aux_info; keep the attribute for API compatibility.
        self.aux_info = {}

        if debug_check:
            # Stitcher-agnostic verification of circuit_lists: runs no matter
            # which circuit_stitcher produced it.
            assert_circuit_lists_match_color_patches(
                self.circuit_lists, self.vertices, self.color_patches
            )

        super().__init__(processor_spec, self.circuit_lists,qubit_labels=self.vertices, nested=nested)


def patch_lines(edge_set: Sequence[Edge],
                vertices: Sequence[Vertex]) -> Tuple[List[Edge], List[Vertex], List[Edge]]:
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
                      ) -> Tuple[List[Dict[str, Any]], Dict[Tuple[int, int], List[Dict[str, Any]]]]:
    vertices = list(vertices)

    patch_infos = []
    groups = defaultdict(list)

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

        key = (len(edge_set), len(unused_qubits))
        groups[key].append(info)
        patch_infos.append(info)

    return patch_infos, groups


def random_index_schedule(n: int, max_len: int, randgen: np.random.Generator) -> np.ndarray:
    """
    A length-``max_len`` index schedule into a CircuitList of size ``n``.

    The ``n`` real indices ``0..n-1`` are always included (each circuit used
    at least once); if ``n < max_len`` the remaining ``max_len - n`` slots are
    filled with uniformly random indices drawn (with repetition) from
    ``0..n-1``, then the whole schedule is shuffled.

    Parameters
    ----------
    n : int
        Size of the CircuitList being scheduled into ``max_len`` slots.

    max_len : int
        Desired length of the returned schedule.

    randgen : numpy.random.Generator
        Random number generator used to draw the extra indices (when
        ``n < max_len``) and to shuffle the result.

    Returns
    -------
    numpy.ndarray
        A length-``max_len`` array of indices into ``0..n-1``.
    """
    if n == max_len:
        base = np.arange(max_len)
    else:
        base = np.concatenate((
            np.arange(n),
            randgen.integers(0, n, size=max_len - n),
        ))
    return randgen.permutation(base)


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
    ``CrosstalkFreeExperimentDesign.__init__``) be run regardless of which
    stitcher was actually used.

    For every germ-power entry ``circuit_lists[L]``, this re-derives each
    patch's own tensored lines/edges from ``vertices``/``color_patches`` (the
    same way ``build_patch_infos`` does) and checks that:

    1. ``circuit_lists[L]`` splits evenly into ``len(color_patches)``
       contiguous, equal-size, patch-major chunks -- i.e. the output honors
       the patch-major-ordering contract documented on
       ``assign_the_designs_with_mapping`` (this is required of *any*
       ``circuit_stitcher``, not just the built-in one).
    2. Every circuit has no implicit idle gates
       (see :func:`assert_no_implicit_idles`).
    3. Every circuit in a given patch's chunk is correctly stitched onto
       that patch's own qubits/edges
       (see :func:`assert_mapped_circuit_matches_patch`).

    Parameters
    ----------
    circuit_lists : list[list[Circuit]]
        The stitched circuit lists to check, e.g. ``self.circuit_lists`` on a
        ``CrosstalkFreeExperimentDesign``.

    vertices : list[Vertex]
        Vertices/qubits in the connectivity graph.

    color_patches : dict[int, list[tuple]]
        Mapping from patch/color identifier to the list of disjoint 2Q edges
        in that patch, as passed to ``CrosstalkFreeExperimentDesign``.

    Returns
    -------
    None

    Raises
    ------
    AssertionError
        If any of the checks above fail.
    """
    patch_infos, _ = build_patch_infos(vertices, color_patches)
    num_patches = len(patch_infos)

    for circuit_list in circuit_lists:
        assert len(circuit_list) % num_patches == 0, (
            f"Expected {len(circuit_list)} circuits to split evenly into "
            f"{num_patches} patch-major chunks (one per color patch)."
        )
        chunk_size = len(circuit_list) // num_patches

        for patch_idx, info in enumerate(patch_infos):
            start = patch_idx * chunk_size
            for circuit in circuit_list[start:start + chunk_size]:
                assert_no_implicit_idles(circuit)
                assert_mapped_circuit_matches_patch(circuit, info)


def build_group_schedules_and_mappers(
    infos: List[Dict[str, Any]],
    representative: Dict[str, Any],
    representative_lines: Sequence[Edge],
    num_edges: int,
    num_unused_qubits: int,
    max_len: int,
    twoq_len: int,
    oneq_len: int,
    randgen: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, Dict[int, Optional[Dict[Vertex, Vertex]]]]:
    """
    Build the random circuit-index schedules and line mappers for one
    patch-shape group (i.e. all patches sharing the same number of 2Q edges
    and unused qubits).

    Every patch in ``infos`` is stitched from a template circuit built once
    for ``representative``; the other patches obtain their circuit by
    applying a line mapper to that template rather than re-tensoring from
    scratch.

    Parameters
    ----------
    infos : list[dict]
        Patch infos (as produced by :func:`build_patch_infos`) sharing this
        group's shape. ``infos[0]`` must be ``representative``.

    representative : dict
        The patch info used as the template patch. Must be ``infos[0]``.

    representative_lines : list
        ``representative["tensored_lines"]``.

    num_edges : int
        Number of 2Q edges in this patch shape.

    num_unused_qubits : int
        Number of unused (1Q) qubits in this patch shape.

    max_len : int
        Number of tensored circuits to generate for this germ power.

    twoq_len : int
        Number of available 2Q circuits at this germ power.

    oneq_len : int
        Number of available 1Q circuits at this germ power.

    randgen : numpy.random.Generator
        Random number generator used for the index schedules.

    Returns
    -------
    edge_perms : numpy.ndarray, shape (num_edges, max_len)
        Random index schedule selecting which 2Q circuit fills each edge
        slot, for each of the ``max_len`` tensored circuits to build.

    oneq_perms : numpy.ndarray, shape (num_unused_qubits, max_len)
        Random index schedule selecting which 1Q circuit fills each
        unused-qubit slot, for each of the ``max_len`` tensored circuits to
        build.

    mappers : dict[int, Optional[dict]]
        Mapping from patch identifier to the line mapper that maps a
        template circuit built for ``representative`` onto that patch.
        ``None`` for ``representative`` itself, since it needs no mapping.
    """
    edge_perms = np.empty((num_edges, max_len), dtype=np.int64)
    for edge_slot in range(num_edges):
        edge_perms[edge_slot, :] = random_index_schedule(twoq_len, max_len, randgen)

    oneq_perms = np.empty((num_unused_qubits, max_len), dtype=np.int64)
    for qubit_slot in range(num_unused_qubits):
        oneq_perms[qubit_slot, :] = random_index_schedule(oneq_len, max_len, randgen)

    mappers: Dict[int, Optional[Dict[Vertex, Vertex]]] = {}
    for info in infos:
        if info is representative:
            mappers[info["patch"]] = None
        else:
            mappers[info["patch"]] = make_line_mapper(
                representative_lines,
                info["tensored_lines"]
            )

    return edge_perms, oneq_perms, mappers


def finalize_patch_buffers(
    patch_buffers: Dict[int, List[Circuit]],
    patch_order: List[int],
    previous_patch_buffers: Dict[int, List[Circuit]],
    ensure_containment: bool,
) -> Tuple[List[Circuit], Dict[int, List[Circuit]]]:
    """
    Flatten one germ-power's per-patch circuit buffers into patch-major
    output, optionally enforcing containment with the previous germ power.

    Parameters
    ----------
    patch_buffers : dict[int, list[Circuit]]
        This germ-power's newly generated circuits, keyed by patch
        identifier.

    patch_order : list[int]
        Patch identifiers, in the desired output order.

    previous_patch_buffers : dict[int, list[Circuit]]
        The previous germ-power's per-patch circuits (after any earlier
        containment merging). Ignored unless ``ensure_containment`` is True.

    ensure_containment : bool
        If True, prepend ``previous_patch_buffers[patch]`` onto
        ``patch_buffers[patch]`` for each patch before flattening, so this
        germ power's output contains every circuit from the previous germ
        power (patch-wise).

    Returns
    -------
    output_circuits : list[Circuit]
        This germ power's circuits, ordered patch-major according to
        ``patch_order``.

    next_previous_patch_buffers : dict[int, list[Circuit]]
        The per-patch buffers to pass back in as ``previous_patch_buffers``
        for the *next* germ power. Equal to the (possibly merged)
        ``patch_buffers`` if ``ensure_containment`` is True; otherwise
        returned unchanged.
    """
    if ensure_containment:
        patch_buffers = {
            patch: previous_patch_buffers[patch] + patch_buffers[patch]
            for patch in patch_order
        }

    output_circuits: List[Circuit] = []
    for patch in patch_order:
        output_circuits.extend(patch_buffers[patch])

    if ensure_containment:
        previous_patch_buffers = {
            patch: list(patch_buffers[patch])
            for patch in patch_order
        }

    return output_circuits, previous_patch_buffers


def assign_the_designs_with_mapping(
    oneq_gstdesign: GateSetTomographyDesign,
    twoq_gstdesign: GateSetTomographyDesign,
    vertices: Sequence[Vertex],
    color_patches: Dict[int, List[Edge]],
    randgen: Optional[np.random.Generator] = None,
    ensure_containment: bool = False,
    _layer_mappers_override: Optional[LayerMappers] = None,
    **kwargs: Any,
) -> List[List[Circuit]]:
    """
    Construct crosstalk-free GST circuit lists for each color patch.

    For each germ-power index, this function combines 2Q GST circuits on the edges
    of each color patch with 1Q GST circuits on the vertices not used by that patch.
    Each color patch should contain mutually disjoint edges, so that the resulting
    tensored circuits do not place simultaneous 2Q operations on overlapping qubits.

    Here, a color patch is one color class from an edge coloring of the 2Q
    connectivity graph. For example, for a five-qubit line,

        0 -- 1 -- 2 -- 3 -- 4

    one valid color patch is ``[(0, 1), (2, 3)]``. For that patch, this function
    uses 2Q GST designs on edges ``(0, 1)`` and ``(2, 3)``, and a 1Q GST design on
    the unused qubit ``4``. Another valid patch is ``[(1, 2), (3, 4)]``, with qubit
    ``0`` receiving a 1Q GST design.

    Patches with the same number of 2Q edges and unused qubits share randomized
    role-based schedules. A representative tensored circuit is constructed once
    for each such group and then mapped onto equivalent patches.

    For each germ power the two CircuitLists need not have the same length.
    ``max(len(oneq), len(twoq))`` tensored circuits are produced. Every circuit in
    the longer CircuitList is used exactly once; the shorter CircuitList is expanded
    to the same length by randomly drawing (with repetition) additional circuits from
    itself (each of its circuits still appears at least once). Either CircuitList may
    be the longer one. Note this list-level expansion is distinct from circuit-*depth*
    equalization: individual sub-circuits that are shallower than their tensor peers
    are padded with explicit idle layers by ``batch_tensor``.

    This function does not deduplicate color patches. For example, if both
    ``[(0, 1), (2, 3)]`` and ``[(1, 0), (3, 2)]`` are supplied, both designs are
    generated, even though they differ only by edge orientation.

    This function does not verify its own output (e.g. that no implicit idle
    gates remain, or that circuits landed on the correct patch). That
    verification is stitcher-agnostic and lives in
    :func:`assert_circuit_lists_match_color_patches`, which
    ``CrosstalkFreeExperimentDesign.__init__`` runs (by default) against
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

    ensure_containment: bool, optional
        If True, ensure that circuitlists[L+1] contains the exact circuits
        from circuitlists[L]. Containment is enforced patch-wise, so the
    output remains patch-major. Default is False. 

    _layer_mappers_override : LayerMappers, optional
        If provided, use these layer mappers instead of building them from the
        two designs via ``build_layer_mappers``. Primarily for testing.

    **kwargs
        Ignored. Accepted so this stitcher matches the generic
        ``circuit_stitcher(oneq, twoq, vertices, color_patches, **kwargs)``
        calling convention used by ``CrosstalkFreeExperimentDesign``, allowing
        it to be swapped with other stitchers that take extra options.

    Returns
    -------
    list[list]
        ``circuit_lists[L]`` contains the generated crosstalk-free GST circuits for
        germ-power index ``L``. Within each germ-power group, circuits are ordered
        patch-major according to the input order of ``color_patches``.

    Raises
    ------
    AssertionError
        If ``oneq_gstdesign`` and ``twoq_gstdesign`` do not have the same number
        of germ-power groups.
    """
    if randgen is None:
        randgen = np.random.default_rng(0)

    oneq_gstdesign_circuitlists = oneq_gstdesign.circuit_lists
    twoq_gstdesign_circuitlists = twoq_gstdesign.circuit_lists
    if _layer_mappers_override is not None:
        layer_mappers = _layer_mappers_override
    else:
        layer_mappers = build_layer_mappers(oneq_gstdesign, twoq_gstdesign)

    assert len(oneq_gstdesign_circuitlists) == len(twoq_gstdesign_circuitlists), \
        "Not implemented."

    vertices = list(vertices)

    patch_infos, groups = build_patch_infos(vertices, color_patches)

    # Preserve user/color_patches ordering in the final output.
    patch_order = [info["patch"] for info in patch_infos]

    previous_patch_buffers = {
        patch: []
        for patch in patch_order
    }

    circuit_lists: List[List[Circuit]] = [[] for _ in twoq_gstdesign_circuitlists]

    for L, (oneq_circuits, twoq_circuits) in _tqdm.tqdm(
        enumerate(zip(oneq_gstdesign_circuitlists, twoq_gstdesign_circuitlists)),
        total=len(twoq_gstdesign_circuitlists)
    ):
        oneq_len = len(oneq_circuits)
        twoq_len = len(twoq_circuits)

        max_len = max(oneq_len, twoq_len)

        # Each generated tensored circuit at this germ power draws one 2Q circuit
        # per edge slot and one 1Q circuit per unused-qubit slot. We produce
        # ``max_len`` tensored circuits so that every circuit in the *longer* of the
        # two CircuitLists is used exactly once; the shorter CircuitList is expanded
        # up to ``max_len`` by randomly drawing (with repetition) additional circuits
        # from itself. Circuit *depths* are equalized separately: batch_tensor pads
        # each shorter sub-circuit with explicit idle layers (via the
        # ``Label(()) -> explicit-idle`` entry in ``layer_mappers``). This works
        # regardless of which CircuitList is longer.

        # Temporary per-patch storage so output ordering remains patch-major.
        patch_buffers = {
            info["patch"]: []
            for info in patch_infos
        }

        for group_key, infos in groups.items():
            num_edges, num_unused_qubits = group_key

            representative = infos[0]
            representative_lines = representative["tensored_lines"]

            edge_perms, oneq_perms, mappers = build_group_schedules_and_mappers(
                infos, representative, representative_lines,
                num_edges, num_unused_qubits, max_len,
                twoq_len, oneq_len, randgen
            )

            for j in range(max_len):
                circs_to_tensor = [twoq_circuits[idx] for idx in edge_perms[:, j]]
                circs_to_tensor += [oneq_circuits[idx] for idx in oneq_perms[:, j]]

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
        circuit_lists[L], previous_patch_buffers = finalize_patch_buffers(
            patch_buffers, patch_order, previous_patch_buffers, ensure_containment
        )

    return circuit_lists

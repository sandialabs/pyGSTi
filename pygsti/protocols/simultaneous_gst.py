#***************************************************************************************************
# Copyright 2015, 2019, 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

import copy as _copy
import importlib as _importlib
import pathlib as _pathlib
import warnings as _warnings

import numpy as np
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union, cast, Mapping
import tqdm as _tqdm

from pygsti import io as _io
from pygsti.protocols.gst import GateSetTomographyDesign
from pygsti.processors import QubitProcessorSpec
from pygsti.circuits.circuit import Circuit
from pygsti.circuits.split_circuits_into_lanes import batch_tensor
from pygsti.baseobjs.label import Label, LabelTup

from pygsti.tools.graphcoloring import (
    canonical_edges, find_neighbors, switchboard_find_edge_coloring,
)

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
# are deliberately left out.
__all__ = [
    'SimultaneousGSTDesign',
    'make_simultaneous_gst_design',
    'assign_the_designs_with_mapping',
    'assert_circuit_lists_match_color_patches',
]


def build_layer_mappers(oneq_gstdesign: GateSetTomographyDesign, twoq_gstdesign: GateSetTomographyDesign) -> LayerMappers:
    """Build the layer_mappers used by batch_tensor when stitching, mapping empty layers to explicit idles."""
    twoq_idle_label = Label(('Gii',) + twoq_gstdesign.qubit_labels)
    oneq_idle_label = Label(('Gi',)  + oneq_gstdesign.qubit_labels)
    mapper_2q: dict[Label, Label] = {twoq_idle_label: twoq_idle_label}
    mapper_1q: dict[Label, Label] = {oneq_idle_label: oneq_idle_label}
    for cl in twoq_gstdesign.circuit_lists:
        for c in cl:
            mapper_2q.update({k: k for k in c._labels})
    for cl in oneq_gstdesign.circuit_lists:
        for c in cl:
            mapper_1q.update({k: k for k in c._labels})
    empty_label = Label(())
    mapper_2q[empty_label] = twoq_idle_label
    mapper_1q[empty_label] = oneq_idle_label
    assert empty_label not in mapper_2q.values()
    assert empty_label not in mapper_1q.values()

    # Check for any labels in `mapper_2q` that imply a single-qubit target.
    # For any such label, add an explicit single-qubit idle on the non-target
    # qubit, and wrap the whole thing as a LabelTupTup.
    for k2 in list(mapper_2q.keys()):
        if k2.num_qubits == 1:
            assert isinstance(k2, LabelTup)
            tgt = k2[1]
            assert tgt in [0,1]
            tmp = [None, None]
            tmp[tgt] = k2
            tmp[1-tgt] = Label("Gi", 1-tgt)
            mapper_2q[k2] = Label(tuple(tmp))

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
    edges = canonical_edges(nq_pspec.compute_2Q_connectivity().edges())
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


def _normalize_coloring(coloring: Mapping[int, Sequence[Edge]]) -> Dict[int, List[Edge]]:
    """
    An edge coloring with every edge as a tuple.
    Do not canonicalize orientation here: (0, 1) and (1, 0) select different lanes of the 2Q design.
    """
    return {patch: [tuple(edge) for edge in edge_set] for patch, edge_set in coloring.items()}


def _stitcher_name(circuit_stitcher: CircuitStitcher) -> Optional[str]:
    """Return the fully-qualified "module.qualname" of a circuit stitcher, or None."""
    qualname = getattr(circuit_stitcher, '__qualname__', None)
    if qualname is None:
        return None
    module_name = getattr(circuit_stitcher, '__module__', None)
    if module_name is None:
        return None
    return module_name + '.' + qualname


def _resolve_stitcher_name(name: Optional[str]) -> Optional[CircuitStitcher]:
    """Import the circuit stitcher recorded by :func:`_stitcher_name`."""
    if name is None:
        _warnings.warn("Loaded a SimultaneousGSTDesign with no recorded circuit_stitcher; "
                       "setting it to None. The design's circuits are unaffected.")
        return None

    try:
        from pygsti.io.metadir import _class_for_name as _resolve_name
        resolved = _resolve_name(name)
    except (ImportError, AttributeError, ValueError) as e:
        _warnings.warn("Could not restore the circuit_stitcher %r of a loaded "
                       "SimultaneousGSTDesign (%s); setting it to None. This is expected for a "
                       "lambda or a locally-defined function. The design's circuits are "
                       "unaffected." % (name, e))
        return None
    return cast(CircuitStitcher, resolved)


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
        Each edge is stored as a tuple regardless of how it was supplied, so that
        ``color_patches`` compares equal across designs however they were built (see
        :func:`_normalize_coloring`). Edge *orientation* is preserved as given.
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
        regardless of which ``circuit_stitcher`` is used.
        """
        if circuit_stitcher is None:
            circuit_stitcher = assign_the_designs_with_mapping
        randgen = np.random.default_rng(seed)
        self.processor_spec = processor_spec
        self.oneq_gstdesign = oneq_gstdesign
        self.twoq_gstdesign = twoq_gstdesign
        self.vertices = self.processor_spec.qubit_labels
        self.edges = canonical_edges(self.processor_spec.compute_2Q_connectivity().edges())
        self.neighbors = find_neighbors(self.vertices, self.edges)
        self.deg = max([len(self.neighbors[v]) for v in self.vertices])
        self.color_patches = _normalize_coloring(edge_coloring)
        self.circuit_stitcher = circuit_stitcher
        self.circuit_stitcher_name = _stitcher_name(circuit_stitcher)

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
        self._register_auxfile_types()

    # region Serialization

    #: Sub-directories (siblings of 'edesign') that :meth:`write` puts the 1Q/2Q
    #: sub-designs in. Written by hand rather than registered as ``TreeNode`` children
    #: because they are not sub-*experiments*: their circuits live on abstract lane
    #: labels and are never executed, so ``ProtocolData`` must not carve a dataset out.
    _SUBDESIGN_DIRS = {'oneq_gstdesign': 'sgst_oneq_gstdesign',
                       'twoq_gstdesign': 'sgst_twoq_gstdesign'}

    def _register_auxfile_types(self) -> None:
        """
        Declare how this design's own attributes are serialized.

        Anything *not* named in ``auxfile_types`` is written straight into 'meta.json'
        as JSON, which fails outright for the members below -- they are int-keyed dicts
        or live Python objects. See ``pygsti.io.metadir._check_jsonable``. ``'none'``
        means "do not write, do not load"; :meth:`from_dir` reconstructs those.
        """
        # Keyed by int, so plain 'json' would be rejected. 'fancykeydict' stores the
        # keys alongside the per-value file metadata instead of as JSON object keys.
        self.auxfile_types['color_patches'] = 'fancykeydict:json'

        # Pure functions of processor_spec, so recomputed on load rather than stored --
        # which also sidesteps JSON turning every tuple into a list.
        for member in ('vertices', 'edges', 'neighbors', 'deg'):
            self.auxfile_types[member] = 'none'

        for member in self._SUBDESIGN_DIRS:  # written/read by hand; see _SUBDESIGN_DIRS
            self.auxfile_types[member] = 'none'

        # The stitcher is recorded by name instead; its kwargs hold a live Generator.
        self.auxfile_types['circuit_stitcher'] = 'none'
        self.auxfile_types['stitcher_kwargs'] = 'none'

    def write(self, dirname=None, parent=None) -> None:
        """
        Write this experiment design to a directory.

        Extends ``ExperimentDesign.write`` by also writing the 1Q and 2Q sub-designs into
        sub-directories of `dirname` that sit alongside 'edesign'.

        Parameters
        ----------
        dirname : str
            The *root* directory to write into.  This directory will have an 'edesign'
            subdirectory, which will be created if needed and overwritten if present.
            If None, then the path this object was loaded from is used.

        parent : ExperimentDesign, optional
            The parent experiment design, when a parent is writing this design as a
            sub-experiment-design.  Otherwise leave as None.

        Returns
        -------
        None
        """
        super().write(dirname=dirname, parent=parent)

        # super().write() resolves a None dirname against _loaded_from, so read it back.
        root = _pathlib.Path(self._loaded_from)
        for member, subdir in self._SUBDESIGN_DIRS.items():
            design = getattr(self, member, None)
            if design is None:
                continue
            design.write(root / subdir)

    @classmethod
    def from_dir(cls, dirname: str, parent=None, name=None, quick_load=False) -> "SimultaneousGSTDesign":
        """
        Initialize a new SimultaneousGSTDesign from `dirname`.

        Reconstructs the members that :meth:`_register_auxfile_types` marks ``'none'``:
        the graph members are recomputed from the processor spec as ``__init__`` does,
        the sub-designs are read back from their sub-directories, and the stitcher is
        re-imported from its recorded name.

        The design is restored as a *record of an already-generated experiment* -- the
        stitcher's random state is not preserved (``stitcher_kwargs`` comes back empty),
        so a loaded design reproduces its circuits but cannot re-stitch them.

        Parameters
        ----------
        dirname : str
            The *root* directory name (under which there is a 'edesign' subdirectory).

        parent : ExperimentDesign, optional
            The parent design object, if there is one.

        name : str, optional
            The sub-name of the design object being loaded.

        quick_load : bool, optional
            Setting this to True skips the loading of the potentially long circuit lists.

        Returns
        -------
        SimultaneousGSTDesign
        """
        ret = super().from_dir(dirname, parent=parent, name=name, quick_load=quick_load)
        root = _pathlib.Path(dirname)

        ret.color_patches = _normalize_coloring(ret.color_patches)  # JSON has no tuples

        # Recompute the graph members, mirroring __init__.
        ret.vertices = ret.processor_spec.qubit_labels
        ret.edges = canonical_edges(ret.processor_spec.compute_2Q_connectivity().edges())
        ret.neighbors = find_neighbors(ret.vertices, ret.edges)
        ret.deg = max(len(ret.neighbors[v]) for v in ret.vertices)

        for member, subdir in cls._SUBDESIGN_DIRS.items():
            subdesign_dir = root / subdir
            if (subdesign_dir / 'edesign' / 'meta.json').exists():
                subdesign_cls = _io.metadir._cls_from_meta_json(subdesign_dir / 'edesign')
                setattr(ret, member, subdesign_cls.from_dir(subdesign_dir, quick_load=quick_load))
            else:
                setattr(ret, member, None)

        ret.circuit_stitcher = _resolve_stitcher_name(getattr(ret, 'circuit_stitcher_name', None))
        ret.stitcher_kwargs = {}
        return ret

    # endregion

    def map_qubit_labels(self, mapper, debug_check: bool = True) -> "SimultaneousGSTDesign":
        """
        Creates a new experiment design whose circuits' qubit labels are updated according to a given mapping.

        This overrides ``GateSetTomographyDesign.map_qubit_labels``, which returns a plain
        ``GateSetTomographyDesign`` and would therefore silently discard the edge coloring
        and the 1Q/2Q sub-designs.

        The mapper relabels the *device's* qubits, so it is applied to the processor spec,
        the vertices, the edges of every color patch, and the stitched circuits -- but
        deliberately **not** to ``oneq_gstdesign``/``twoq_gstdesign``, which live on their
        own abstract lane labels (e.g. ``(0,)`` and ``(0, 1)``) and are carried over
        unchanged.

        The circuits are relabelled rather than re-stitched: re-running the stitcher would
        redraw its random schedules and return different circuit content, whereas a
        relabelling is the same experiment on renamed qubits.

        Parameters
        ----------
        mapper : dict or function
            A dictionary whose keys are the existing self.qubit_labels values
            and whose value are the new labels, or a function which takes a
            single (existing qubit-label) argument and returns a new qubit-label.

        debug_check : bool, optional
            If True (the default), verify the relabelled ``circuit_lists`` against the
            relabelled coloring via :func:`assert_circuit_lists_match_color_patches`, the
            same check ``__init__`` runs.

        Returns
        -------
        SimultaneousGSTDesign
        """
        def mapper_func(label): return mapper[label] if isinstance(mapper, dict) else mapper(label)

        mapped_processor_spec = self.processor_spec.map_qubit_labels(mapper)
        mapped_vertices = tuple(mapper_func(v) for v in self.vertices)
        mapped_color_patches = {
            patch: [tuple(mapper_func(q) for q in edge) for edge in edge_set]
            for patch, edge_set in self.color_patches.items()
        }
        mapped_circuit_lists = [[c.map_state_space_labels(mapper) for c in circuit_list]
                                for circuit_list in self.circuit_lists]

        if debug_check:
            assert_circuit_lists_match_color_patches(
                mapped_circuit_lists, mapped_vertices, mapped_color_patches
            )

        # Bypass __init__: it would re-run the stitcher and redraw its random schedules,
        # which is precisely what relabelling exists to avoid.
        mapped = self.__class__.__new__(self.__class__)
        mapped.oneq_gstdesign = self.oneq_gstdesign
        mapped.twoq_gstdesign = self.twoq_gstdesign
        mapped.vertices = mapped_vertices
        mapped.edges = canonical_edges(mapped_processor_spec.compute_2Q_connectivity().edges())
        mapped.neighbors = find_neighbors(mapped.vertices, mapped.edges)
        mapped.deg = max(len(mapped.neighbors[v]) for v in mapped.vertices)
        mapped.color_patches = mapped_color_patches
        mapped.circuit_stitcher = self.circuit_stitcher
        mapped.circuit_stitcher_name = self.circuit_stitcher_name
        mapped.stitcher_kwargs = self.stitcher_kwargs
        mapped.circuit_lists = mapped_circuit_lists

        # Sets processor_spec, qubit_labels, all_circuits_needing_data, auxfile_types, etc.
        GateSetTomographyDesign.__init__(
            mapped, mapped_processor_spec, mapped_circuit_lists,
            qubit_labels=mapped.vertices, nested=self.nested
        )
        mapped._register_auxfile_types()  # ...which resets auxfile_types, so re-declare ours
        return mapped

    def as_circuit_lists_design(self) -> GateSetTomographyDesign:
        """
        A plain :class:`GateSetTomographyDesign` holding this design's circuits.

        It carries the same circuit lists, processor spec, qubit labels and nesting, but
        none of the simultaneous-GST structure -- no edge coloring, no sub-designs. Useful
        when downstream code should not see, or accidentally rely on, the simultaneous
        structure, and as the way to get an object that supports :meth:`merge_with`.

        Truncation does not need this: it is supported on the design itself and returns a
        ``SimultaneousGSTDesign``.

        Returns
        -------
        GateSetTomographyDesign
        """
        return GateSetTomographyDesign(
            self.processor_spec, [list(cl) for cl in self.circuit_lists],
            qubit_labels=self.qubit_labels, nested=self.nested
        )

    # region Truncation

    # Dropping circuits is safe here: a circuit carries its patch in its content -- which
    # multi-qubit gates it applies to which edges -- so the surviving circuits still
    # validate against the coloring (see assert_circuit_lists_match_color_patches). Only
    # the stitcher's equal-chunk patch-major *positional* layout is lost, and nothing
    # reads it off a built design.
    #
    # The two in-place overrides below exist for one reason: CircuitListsDesign._truncate_
    # to_circuits_inplace sets nested=False, because in general filtering circuit lists
    # need not preserve containment. It does preserve it here. truncate_to_circuits,
    # truncate_to_available_data and truncate_to_lists all filter every germ-power list by
    # one common keep-set, and (list_L & keep) <= (list_{L+1} & keep) whenever
    # list_L <= list_{L+1}. (truncate_to_design is the exception -- see that override.)
    #
    # Losing nested=True matters at the next *rebuild*, not immediately: these hooks mutate
    # in place, but CircuitListsDesign.__init__ reads nested to decide whether to take
    # circuit_lists[-1] as all_circuits_needing_data or to union every list instead. So a
    # design that forgot the flag would come back from map_qubit_labels, merge_with or
    # from_dir having recomputed its circuit set the slow way. StandardGSTDesign.truncate_
    # to_design restores the flag by hand for the same reason (see gst.py).

    def _truncate_to_circuits_inplace(self, circuits_to_keep):
        was_nested = self.nested
        super()._truncate_to_circuits_inplace(circuits_to_keep)
        self.nested = was_nested

    def _truncate_to_design_inplace(self, other_design):
        # This one truncates list L against *other_design*'s list L, a different keep-set
        # per list, so nesting survives only if the other design is nested too.
        was_nested = self.nested and getattr(other_design, 'nested', False)
        super()._truncate_to_design_inplace(other_design)
        self.nested = was_nested

    def truncate_to_lists(self, list_indices_to_keep):
        """
        A new design keeping only some of the germ-power circuit lists.

        Overridden because ``CircuitListsDesign.truncate_to_lists`` builds a plain
        ``CircuitListsDesign``, which would silently discard the processor spec, the edge
        coloring and the sub-designs. A subsequence of a nested chain is still nested, so
        ``nested`` is preserved.

        Parameters
        ----------
        list_indices_to_keep : iterable
            The (integer) indices into ``circuit_lists`` to keep.

        Returns
        -------
        SimultaneousGSTDesign
        """
        base = _copy.deepcopy(self)
        kept = [base.circuit_lists[i] for i in list_indices_to_keep]
        base.circuit_lists = kept
        # Re-truncating the kept lists to their own union is a no-op on them, but it is
        # what recomputes all_circuits_needing_data.
        base._truncate_to_circuits_inplace({c for lst in kept for c in lst})
        return base

    # endregion

    def merge_with(self, other_edesign, remove_duplicates=True):
        """
        Not supported for a SimultaneousGSTDesign; see :meth:`as_circuit_lists_design`.

        Unlike truncation, merging has no obvious right answer: the result would have to
        carry a single edge coloring, and there is no general rule for combining the
        colorings of two stitched designs (they may disagree, or partition different
        qubits). Use ``.as_circuit_lists_design()`` on both to get plain designs holding
        the same circuits, and merge those.
        """
        raise NotImplementedError(
            "merge_with is not supported for a SimultaneousGSTDesign: there is no general "
            "rule for combining two designs' edge colorings. Use .as_circuit_lists_design() "
            "to get a plain GateSetTomographyDesign holding the same circuits, which "
            "supports merging."
        )


def patch_lines(edge_set: Sequence[Edge],
                vertices: Sequence[Vertex]) -> Tuple[List[Edge], List[Vertex], List[Union[Edge, Tuple[Vertex]]]]:
    """Return the ordered tensor lines for a patch: first 2Q edge lines, then 1Q unused-qubit lines."""
    edge_set = sorted([tuple(edge) for edge in edge_set])
    used_qubits    = {q for edge in edge_set for q in edge}
    unused_qubits  = [q for q in vertices if q not in used_qubits]
    tensored_lines = list(edge_set) + [(q,) for q in unused_qubits]
    return edge_set, unused_qubits, tensored_lines


def make_line_mapper(source_lines: Sequence[Edge],
                     target_lines: Sequence[Edge]) -> Dict[Vertex, Vertex]:
    """
    Construct a state-space-label mapper from source tensor lines to target tensor lines.
    Example: [(0, 1), (4,)] to [(2, 3), (0,)] returns {0: 2, 1: 3, 4: 0}.
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
    """Describe each color patch's geometry in color_patches order, fixing patch-major output order."""
    vertices = list(vertices)

    patch_infos = []

    for patch, edge_set in color_patches.items():
        edge_set, unused_qubits, tensored_lines = patch_lines(edge_set, vertices)

        info = {
            "patch": patch, "edge_set": edge_set, "unused_qubits": unused_qubits,
            "tensored_lines": tensored_lines, "num_edges": len(edge_set),
            "num_unused_qubits": len(unused_qubits),
        }

        patch_infos.append(info)

    return patch_infos


def group_patches_for_scheduling(patch_infos: List[Dict[str, Any]],
                                 share_same_shape_schedules: bool = True
                                 ) -> List[List[Dict[str, Any]]]:
    """
    Partition patches into scheduling groups based on their shape (2Q edge slots and 1Q unused qubit slots).
    A singleton group is exactly the degenerate case of a shared one (with no other members to relabel onto).
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
    Build a length-num_circs_at_germ_power index schedule into a CircuitList of size n.
    Samples without replacement if n is larger than the requested schedule; draws
    bootstrap indices uniformly with replacement from 0..n-1 if n is smaller; then
    shuffles the result.
    """
    if n == 0 and num_circs_at_germ_power:
        raise ValueError("Cannot schedule circuits from an empty component-design pool")
    if n == num_circs_at_germ_power:
        base = np.arange(num_circs_at_germ_power)
    elif n > num_circs_at_germ_power:
        base = randgen.permutation(n)[:num_circs_at_germ_power]
    else:
        base = np.concatenate((
            np.arange(n),
            randgen.integers(0, n, size=num_circs_at_germ_power - n),
        ))
    return randgen.permutation(base)

#region Invariant Helpers

def assert_no_implicit_idles(circuit: Circuit) -> None:
    """Assert that every idle gate in `circuit` is explicit (no implicit idle gates)."""
    for i in range(circuit.num_layers):
        l0 = set(circuit.layer(i))
        l1 = set(circuit.layer_with_idles(i))
        assert l0 == l1, (
            f"Implicit idle gate(s) detected in layer {i}: "
            f"layer()={l0} != layer_with_idles()={l1}"
        )


def assert_mapped_circuit_matches_patch(mapped_circuit: Circuit, info: Dict[str, Any]) -> None:
    """Assert that `mapped_circuit`'s line labels match the patch, and its multi-qubit gates land on the patch's own edges."""
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


def build_edge_to_patch_index(patch_infos: List[Dict[str, Any]]) -> Dict[Edge, int]:
    """
    Map each patch edge, in both orientations, to its index in ``patch_infos``.

    Raises ``AssertionError`` if an edge belongs to two patches. A proper edge
    coloring never does that (see
    :func:`pygsti.tools.graphcoloring.check_valid_edge_coloring`), and if it did,
    a circuit's patch could not be recovered from the gates it contains.
    """
    lookup: Dict[Edge, int] = {}
    for patch_idx, info in enumerate(patch_infos):
        for edge in info["edge_set"]:
            for oriented in (tuple(edge), tuple(reversed(tuple(edge)))):
                prior = lookup.setdefault(oriented, patch_idx)
                assert prior == patch_idx, (
                    f"Edge {oriented} appears in both patch "
                    f"{patch_infos[prior]['patch']!r} and patch {info['patch']!r}. "
                    "The color patches must be a proper edge coloring: with an edge in "
                    "two patches, a circuit's patch is not determined by its content."
                )
    return lookup


def identify_circuit_patch(circuit: Circuit,
                           patch_infos: List[Dict[str, Any]],
                           edge_to_patch: Dict[Edge, int]) -> Optional[int]:
    """
    Which patch a circuit belongs to, judged by the edges its multi-qubit gates act on.

    Returns the index into ``patch_infos``, or None if the circuit has no
    multi-qubit gates -- in which case it is consistent with *every* patch, since
    each patch's tensored lines cover all the vertices.

    Raises ``AssertionError`` if a multi-qubit gate acts on something that is not
    an edge of any patch, or if the circuit's multi-qubit gates straddle two
    patches.
    """
    patch_idx = None
    for i in range(circuit.num_layers):
        for op in circuit.layer(i):
            if len(op.qubits) <= 1:
                continue
            found = edge_to_patch.get(tuple(op.qubits))
            assert found is not None, (
                f"Multi-qubit gate {op} acts on {tuple(op.qubits)}, which is not an "
                f"edge of any color patch (patches: "
                f"{[info['edge_set'] for info in patch_infos]})."
            )
            assert patch_idx is None or patch_idx == found, (
                f"Circuit spans two color patches: an earlier multi-qubit gate put it "
                f"in patch {patch_infos[patch_idx]['patch']!r}, but {op} on "
                f"{tuple(op.qubits)} belongs to patch {patch_infos[found]['patch']!r}. "
                "A simultaneous-GST circuit runs one patch at a time."
            )
            patch_idx = found
    return patch_idx


def assert_circuit_lists_match_color_patches(
    circuit_lists: List[List[Circuit]],
    vertices: Sequence[Vertex],
    color_patches: Dict[int, List[Edge]],
) -> None:
    """
    Assert that every circuit in ``circuit_lists`` is a well-formed stitching of
    one of ``color_patches`` onto ``vertices``.

    This is stitcher-agnostic: it validates the *output* of whatever
    ``circuit_stitcher`` produced ``circuit_lists``, not just the built-in
    ``assign_the_designs_with_mapping``, so it can (and is, by
    ``SimultaneousGSTDesign.__init__``) be run regardless of which stitcher was
    actually used.

    A circuit's patch is read off its **content**: the edges its multi-qubit
    gates act on. For each distinct circuit this checks that

    1. it has no implicit idle gates (see :func:`assert_no_implicit_idles`);
    2. every multi-qubit gate acts on an edge of some patch, in either
       orientation, and all of them belong to the *same* patch (see
       :func:`identify_circuit_patch`); and
    3. it is correctly stitched onto that patch's qubits -- in particular its
       line labels cover all the vertices (see
       :func:`assert_mapped_circuit_matches_patch`).

    A circuit with no multi-qubit gates is valid for any patch and is checked
    against the first one; the only patch-dependent part of check 3 is the set
    of allowed edges, and it has none.

    Each distinct circuit is checked once. All three checks are pure functions of
    the circuit, and ``circuit_lists`` is normally nested -- germ power ``L``
    repeats germ powers ``0..L-1`` -- so without deduplication a naive pass would
    re-walk every layer of germ power 0's circuits once per germ power.

    Note what is *not* checked. The built-in stitcher emits each germ-power list
    germ-power-major then patch-major, with every patch contributing an equal
    contiguous chunk, and ``_nest_a_circuitlist`` relies on that while building
    the lists. But that layout is a property of the stitcher's output, not a
    requirement of the design: a ``SimultaneousGSTDesign`` may have had circuits
    dropped from it (see :meth:`SimultaneousGSTDesign.truncate_to_circuits`),
    which leaves every circuit valid while destroying the equal-chunk positional
    structure.
    Tests that care about the stitcher's ordering should assert it on
    ``assign_the_designs_with_mapping``'s output directly.

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
    assert patch_infos, "color_patches is empty; there is no patch for any circuit to belong to."
    edge_to_patch = build_edge_to_patch_index(patch_infos)

    checked: set = set()
    for circuit_list in circuit_lists:
        for circuit in circuit_list:
            if circuit in checked:
                continue
            assert_no_implicit_idles(circuit)
            patch_idx = identify_circuit_patch(circuit, patch_infos, edge_to_patch)
            if patch_idx is None:  # no multi-qubit gates: valid for any patch
                patch_idx = 0
            assert_mapped_circuit_matches_patch(circuit, patch_infos[patch_idx])
            checked.add(circuit)
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
    Build the random circuit-index schedules for one patch-shape group at one germ power.
    Independent draws per slot allow different edges to run different 2Q circuits simultaneously.
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
    Build the line mappers for one scheduling group.
    Mappers are geometry-dependent and built once to be reused across all germ powers.
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
    """Flatten one germ power's per-patch circuit buffers into a single list, patch-major."""
    output_circuits: List[Circuit] = []
    for patch in patch_order:
        output_circuits.extend(patch_buffers[patch])
    return output_circuits


def _denest_a_circuitlist(circuitlist: list[list[Circuit]]) -> list[list[Circuit]]:
    """Remove any circuits which were duplicated in a previous inner list."""
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
    """Undo _denest_a_circuitlist patch-wise to preserve germ-power-major-then-patch-major ordering."""
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

    nested_oneq_circuitlists = oneq_gstdesign.circuit_lists
    nested_twoq_circuitlists = twoq_gstdesign.circuit_lists
    layer_mappers = build_layer_mappers(oneq_gstdesign, twoq_gstdesign)
    # Denest the Circuit lists. We will renest them at the end.
    oneq_gstdesign_circuitlists = _denest_a_circuitlist(nested_oneq_circuitlists)
    twoq_gstdesign_circuitlists = _denest_a_circuitlist(nested_twoq_circuitlists)

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
        new_oneq_len = len(oneq_circuits)
        new_twoq_len = len(twoq_circuits)

        num_circs_at_germ_power = max(new_oneq_len, new_twoq_len)

        # StandardGSTDesign may repeat a nested list when no selected germ adds a
        # circuit at an adjacent maximum length.  If the other component does add
        # circuits, pair those additions with a sample from the unchanged
        # component's cumulative pool.  The number of new simultaneous circuits
        # remains governed by the denested additions, so older component circuits
        # are not spuriously counted as new work.
        if num_circs_at_germ_power:
            if new_oneq_len == 0:
                oneq_circuits = nested_oneq_circuitlists[L]
            if new_twoq_len == 0:
                twoq_circuits = nested_twoq_circuitlists[L]

        oneq_len = len(oneq_circuits)
        twoq_len = len(twoq_circuits)

        # We produce max(oneq, twoq) simultaneous circuits to use every circuit of the longer design.
        # The shorter design is bootstrapped, and batch_tensor pads shorter sub-circuits with explicit idles.

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

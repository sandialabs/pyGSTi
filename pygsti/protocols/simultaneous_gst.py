#***************************************************************************************************
# Copyright 2015, 2019, 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

import copy as _copy
import pathlib as _pathlib
import warnings as _warnings

import numpy as np
from numpy.typing import DTypeLike
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union, cast

from pygsti import io as _io
from pygsti.protocols.gst import GateSetTomographyDesign
from pygsti.models.model import Model
from pygsti.processors import QubitProcessorSpec

from pygsti.protocols._stitchers import (
    CallableStitcher, CircuitStitcher, Edge, RandomizedPatchStitcher, Vertex,
    _validate_stitched_circuits,
)
from pygsti.tools.edesigntools import BlockDoptReducer as _BlockDoptReducer
from pygsti.tools.graphs import (
    canonical_edges, find_neighbors, max_degree,
)
from pygsti.tools.graphs.coloring import switchboard_find_edge_coloring

SeedLike = Union[int, np.random.SeedSequence, np.random.Generator]

# Public names exported by ``pygsti.protocols``.
__all__ = [
    'SimultaneousGSTDesign',
    'make_simultaneous_gst_design',
    'CircuitStitcher',
    'RandomizedPatchStitcher',
    'CallableStitcher',
]


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
    deg = max_degree(neighbors)
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


def _recordable_seed(seed: Optional[SeedLike]) -> Optional[Union[int, Dict[str, Any]]]:
    """`seed` reduced to something JSON can hold, so a design can re-stitch after a reload.

    An int is itself; a ``SeedSequence`` becomes its entropy and spawn key, which is
    enough to rebuild it. A live ``Generator`` has consumed state that cannot be
    recorded, so it becomes None and the design loses the ability to re-stitch -- pass an
    int if that matters.
    """
    if seed is None or isinstance(seed, (int, np.integer)):
        return None if seed is None else int(seed)
    if isinstance(seed, np.random.SeedSequence):
        return {'entropy': seed.entropy, 'spawn_key': list(seed.spawn_key)}
    _warnings.warn("A SimultaneousGSTDesign was seeded with a live Generator, whose state "
                   "cannot be recorded. The design's circuits are unaffected, but it will "
                   "not be able to re-stitch them -- not even in memory, no reload needed "
                   "to see it. Pass an int or a SeedSequence for a design that can.")
    return None


def _seed_from_record(record: Optional[Union[int, Mapping[str, Any]]]) -> Optional[SeedLike]:
    """Invert :func:`_recordable_seed`."""
    if record is None or isinstance(record, (int, np.integer)):
        return record
    return np.random.SeedSequence(entropy=record['entropy'],
                                  spawn_key=tuple(record['spawn_key']))


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
    circuit_stitcher (CircuitStitcher): The rule that combines the two designs
        (default: a :class:`RandomizedPatchStitcher`). Wrap a function explicitly in
        :class:`CallableStitcher`, whose documentation describes its persistence limits.
    seed (optional): Anything ``np.random.default_rng`` accepts -- an int, a SeedSequence,
        or an already-built Generator -- used to seed the randgen handed to the stitcher.
        Recorded as ``stitch_seed``, so an int or a SeedSequence lets a reloaded design
        re-stitch its circuits. Left as the default ``None``, a fresh ``SeedSequence`` is
        generated and recorded automatically -- the resulting circuits are exactly as
        random as an explicit ``None`` always was, but the design can still restitch. An
        already-built ``Generator`` is the one case that still cannot be recorded.
    nested (bool): Whether ``circuit_stitcher``'s output is nested, i.e. whether
        ``circuit_lists[L+1]`` contains every circuit in ``circuit_lists[L]``. The
        default stitcher always produces nested lists, hence the default of True;
        set this to False only when supplying a stitcher that does not.
    verbosity (int): Forwarded to ``circuit_stitcher``. With the default stitcher,
        anything greater than 0 displays a progress bar over germ powers.
        Defaults to 0 (silent).

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
                 verbosity: int = 0):
        """
        Assume that the GST designs have the same Ls.

        The default ``circuit_stitcher`` is a :class:`RandomizedPatchStitcher`.

        A stitcher's *options* are its own constructor arguments, not keyword arguments
        here::

            SimultaneousGSTDesign(..., circuit_stitcher=RandomizedPatchStitcher(
                share_same_shape_schedules=False))

        which is why this signature has no ``**stitcher_kwargs``: a catch-all silently
        accepts a misspelled option, whereas a stitcher constructor rejects one.

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
        (the default), the resulting ``circuit_lists`` are verified via
        :func:`_validate_stitched_circuits` -- checking that every
        generated circuit has no implicit idle gates and is correctly stitched onto its
        own patch's qubits/edges. That check lives in :meth:`CircuitStitcher.stitch`, so
        it runs for any stitcher and for direct stitcher calls too.
        """
        self.processor_spec = processor_spec
        self.oneq_gstdesign = oneq_gstdesign
        self.twoq_gstdesign = twoq_gstdesign
        self.vertices = self.processor_spec.qubit_labels
        self.edges = canonical_edges(self.processor_spec.compute_2Q_connectivity().edges())
        self.neighbors = find_neighbors(self.vertices, self.edges)
        self.deg = max([len(self.neighbors[v]) for v in self.vertices])
        self.color_patches = _normalize_coloring(edge_coloring)
        if circuit_stitcher is None:
            circuit_stitcher = RandomizedPatchStitcher()
        if not isinstance(circuit_stitcher, CircuitStitcher):
            raise TypeError("circuit_stitcher must be a CircuitStitcher instance. "
                            "Instantiate a stitcher class, or wrap a function in CallableStitcher.")
        self.circuit_stitcher = circuit_stitcher
        if seed is None:
            seed = np.random.SeedSequence()
        self.stitch_seed = _recordable_seed(seed)

        self.circuit_lists = self.circuit_stitcher.stitch(
            self.oneq_gstdesign, self.twoq_gstdesign, self.vertices, self.color_patches,
            seed=seed, verbosity=verbosity, debug_check=debug_check,
        )

        super().__init__(processor_spec, self.circuit_lists,qubit_labels=self.vertices, nested=nested)
        self._register_auxfile_types()

    def restitch(self, verbosity: int = 0, debug_check: bool = True) -> "SimultaneousGSTDesign":
        """Rebuild this design's circuits from its recorded stitcher and seed.

        Reproduces the original circuits when the stitcher uses only the supplied RNG
        and its implementation and configuration are unchanged. An omitted `seed`
        generates and records one automatically; a live `Generator` cannot be recorded.
        A loaded :class:`CallableStitcher` must also have a recoverable recipe. Saved
        circuit data remains usable when the recipe is unavailable, but cannot be
        regenerated by this method.

        Returns
        -------
        SimultaneousGSTDesign

        Raises
        ------
        ValueError
            If this design has no stitcher to run (loaded from a directory written before
            the stitcher became a serializable object), or no recorded seed to restitch
            from (built with a live Generator, or loaded from a directory written before
            seeds were recorded), or the callable recipe could not be restored.
        """
        if self.circuit_stitcher is None:
            raise ValueError("This SimultaneousGSTDesign has no circuit_stitcher to run.")
        if self.stitch_seed is None:
            raise ValueError(
                "This SimultaneousGSTDesign has no recorded seed to restitch from -- it "
                "was either built with a live Generator (whose state can't be recorded) "
                "or loaded from a design saved before seeds were recorded. Rebuild it "
                "with an int or a SeedSequence, or let seed default, to make it "
                "restitchable.")
        return SimultaneousGSTDesign(
            self.processor_spec, self.oneq_gstdesign, self.twoq_gstdesign,
            self.color_patches, circuit_stitcher=self.circuit_stitcher,
            seed=_seed_from_record(self.stitch_seed), nested=self.nested,
            debug_check=debug_check, verbosity=verbosity)

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

        # Store the stitcher recipe, or an unavailable-recipe marker for callables
        # whose function or options cannot be restored.
        # `stitch_seed` is a plain int or dict and needs no declaration; together they are
        # what lets a reloaded design re-stitch (see :meth:`restitch`).
        self.auxfile_types['circuit_stitcher'] = 'serialized-object'

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
        and the sub-designs are read back from their sub-directories. A stitcher subclass
        defined outside pyGSTi is restored if its class remains importable and its
        serialization methods preserve its configuration.

        With an executable stitcher and ``stitch_seed`` restored, a loaded design can
        regenerate its own circuits; see :meth:`restitch`. If a :class:`CallableStitcher`
        recipe is unavailable, ordinary loading still restores the saved circuit data.
        ``quick_load=True`` skips circuit data, including the component designs' lists;
        an unavailable recipe provides no alternative source of circuits.

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
            relabelled coloring via :func:`_validate_stitched_circuits`, the
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
            _validate_stitched_circuits(
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
        mapped.stitch_seed = self.stitch_seed
        mapped.circuit_lists = mapped_circuit_lists

        # Sets processor_spec, qubit_labels, all_circuits_needing_data, etc. It also
        # rebuilds auxfile_types from scratch and sets `selection` to None.
        GateSetTomographyDesign.__init__(
            mapped, mapped_processor_spec, mapped_circuit_lists,
            qubit_labels=mapped.vertices, nested=self.nested
        )
        # Re-declare this class's auxfile types, and restore the reduction record:
        # relabelling renames qubits, it does not re-select circuits.
        mapped._register_auxfile_types()
        mapped.selection = self.selection
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
    # validate against the coloring (see _validate_stitched_circuits). Only
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

    def reduce_by_dopt(self, model: Model, num_circuits: int, *, ridge: float = 1.0,
                       dtype: DTypeLike = np.float64,
                       warn_on_target_model: bool = True) -> "SimultaneousGSTDesign":
        """
        A copy of this design keeping the circuits selected by the D-optimal objective.

        Pass a model at a plausible noisy point, not a target model --
        :meth:`pygsti.tools.edesigntools.BlockDoptReducer.from_target_model` produces one,
        and :func:`pygsti.tools.edesigntools.perturb_errorgen_rates` explains why it is
        needed. Passing a target model here warns.

        Parameters
        ----------
        model : Model
            Defines the parameter sensitivities used in selection. The objective uses
            unweighted probability derivatives; it is not multinomial Fisher information.

        num_circuits : int
            The budget.

        ridge, dtype, warn_on_target_model
            As for :class:`~pygsti.tools.edesigntools.BlockDoptReducer`.

        Returns
        -------
        SimultaneousGSTDesign
            Its `selection` attribute holds the reducer and its score curve; read
            ``selection.scores`` to inspect the objective as the budget grows.
        """
        reducer = _BlockDoptReducer(model, ridge=ridge, dtype=dtype,
                                    warn_on_target_model=warn_on_target_model)
        return self.reduce_with(reducer, num_circuits)

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

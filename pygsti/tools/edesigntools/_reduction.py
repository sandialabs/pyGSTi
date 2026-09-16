"""
Pluggable rules for cutting an experiment design down to a budget
"""
#***************************************************************************************************
# Copyright 2015, 2019, 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Iterable, Mapping, Optional, Sequence, Union

import numpy as _np

from pygsti.baseobjs.nicelyserializable import NicelySerializable as _NicelySerializable

if TYPE_CHECKING:
    from pygsti.circuits.circuit import Circuit
    from pygsti.protocols.protocol import ExperimentDesign

__all__ = ['CircuitSelection', 'DesignReducer', 'CallableReducer']

#: What :meth:`DesignReducer.cast` accepts in place of a reducer: called as
#: ``f(design, num_circuits)``, returning a :class:`CircuitSelection` or the circuits to keep.
ReducerFunction = Callable[['ExperimentDesign', Optional[int]],
                           Union['CircuitSelection', Sequence['Circuit']]]


class CircuitSelection(_NicelySerializable):
    """The circuits a :class:`DesignReducer` chose, and what it can say about the choice.

    A reducer returns one of these rather than a bare list so that a score curve is part
    of the contract rather than an optional extra, and so that a reducer with
    diagnostics that fit no shared schema has somewhere to put them.

    Parameters
    ----------
    circuits : sequence of Circuit
        The circuits to keep.  Best-first where the reducer has a notion of "best": for a
        greedy reducer this is selection order, which makes `circuits[:k]` that reducer's
        own answer for a smaller budget without re-running it.  A reducer with no
        meaningful order returns them in any order and should say so in `metadata`.

    scores : array-like, optional
        One float per kept circuit.  The meaning belongs to the reducer -- see
        `score_name` -- but for a greedy reducer reporting a cumulative objective it is
        nondecreasing, and the point at which it flattens is the point past which the
        budget is buying little.  Read it before trusting a budget.

    score_name : str, optional
        What `scores` measures, short enough to label an axis with.

    metadata : dict, optional
        Free-form, reducer-specific.  :meth:`DesignReducer.select` adds
        `'num_candidates'` if the reducer did not.

    Attributes
    ----------
    reducer : DesignReducer or None
        Stamped by :meth:`DesignReducer.select`, so a selection carries the policy that
        produced it.  None if the selection was built by hand.
    """

    def __init__(self, circuits: Iterable[Circuit], scores: Optional[Sequence[float]] = None,
                 score_name: Optional[str] = None, metadata: Optional[Mapping[str, Any]] = None) -> None:
        super().__init__()
        self.circuits: tuple[Circuit, ...] = tuple(circuits)
        self.scores: Optional[_np.ndarray] = None if scores is None else _np.asarray(scores, dtype=_np.float64)
        self.score_name: Optional[str] = score_name
        self.metadata: dict[str, Any] = dict(metadata) if metadata else {}
        self.reducer: Optional[DesignReducer] = None

    def __len__(self) -> int:
        return len(self.circuits)

    def _to_nice_serialization(self) -> dict[str, Any]:
        state = super()._to_nice_serialization()
        state.update({
            'circuits': [c.str for c in self.circuits],
            'scores': None if self.scores is None else self.scores.tolist(),
            'score_name': self.score_name,
            'metadata': self.metadata,
            'reducer': None if self.reducer is None else self.reducer.to_nice_serialization(),
        })
        return state

    @classmethod
    def _from_nice_serialization(cls, state: dict[str, Any]) -> CircuitSelection:
        from pygsti.io.readers import convert_strings_to_circuits as _to_circuits
        ret = cls(_to_circuits(state['circuits']), state['scores'],
                  state['score_name'], state['metadata'])
        if state.get('reducer') is not None:
            ret.reducer = DesignReducer.from_nice_serialization(state['reducer'])
        return ret


class DesignReducer(_NicelySerializable):
    """A rule for choosing which of an experiment design's circuits are worth keeping.

    Subclasses implement :meth:`_select` and nothing else.  Callers use :meth:`select`
    (to see the reducer's diagnostics) or :meth:`reduce` (to get a smaller design), both
    of which validate what `_select` returned before it can do any damage.

    A subclass defined anywhere -- in a user's own package, in a notebook module -- works
    with no registration step, and serializes and reloads correctly, because
    :class:`~pygsti.baseobjs.nicelyserializable.NicelySerializable` records the defining
    module and class name and re-imports on load.

    Reducers that need a model, a random seed or a weighting take it at construction: a
    reducer is a fully configured policy, so that :meth:`select` has the same signature
    whatever the reducer needs to do its job.
    """

    # -- the one method a subclass writes ----------------------------------- #

    def _select(self, design: ExperimentDesign, num_circuits: Optional[int]) -> CircuitSelection:
        """Choose circuits from `design`; return a :class:`CircuitSelection`.

        `num_circuits` arrives already validated and clamped to the number of candidates,
        or as None meaning "use your own stopping rule".  A reducer with no such rule
        should raise `ValueError` on None.

        Returning fewer than `num_circuits` circuits is allowed -- a reducer may run out
        of anything worth adding -- but returning more is an error, as is returning a
        circuit that is not in the design.
        """
        raise NotImplementedError("DesignReducer subclasses must implement _select.")

    # -- what callers use --------------------------------------------------- #

    def select(self, design: ExperimentDesign, num_circuits: Optional[int] = None) -> CircuitSelection:
        """The circuits this reducer would keep, with its diagnostics.

        Parameters
        ----------
        design : ExperimentDesign
            Anything with `all_circuits_needing_data`.  Passed to `_select` whole, not as
            a circuit list, so that a reducer can use the design's structure -- germ-power
            lists, or a simultaneous design's color patches -- to spend its budget.

        num_circuits : int, optional
            The budget, clamped to the number of candidates.  None asks the reducer to
            choose for itself; for a greedy reducer that usually means "rank everything",
            which is the natural way to *pick* a budget from the score curve.

        Returns
        -------
        CircuitSelection
        """
        candidates = list(design.all_circuits_needing_data)
        if num_circuits is not None:
            num_circuits = int(num_circuits)
            if num_circuits < 0:
                raise ValueError(f"num_circuits must be nonnegative, got {num_circuits}.")
            num_circuits = min(num_circuits, len(candidates))

        selection = self._select(design, num_circuits)
        self._validate(selection, candidates, num_circuits)
        selection.reducer = self
        selection.metadata.setdefault('num_candidates', len(candidates))
        return selection

    def reduce(self, design: ExperimentDesign, num_circuits: Optional[int] = None) -> ExperimentDesign:
        """A copy of `design` keeping only the circuits this reducer selects.

        Works on any design with `all_circuits_needing_data` and `truncate_to_circuits`,
        which is every :class:`~pygsti.protocols.ExperimentDesign`.  Designs that are
        fitted to a model also expose this as `design.reduce_with(reducer, n)`.

        Parameters
        ----------
        design : ExperimentDesign
            Not modified.

        num_circuits : int, optional
            As for :meth:`select`.

        Returns
        -------
        ExperimentDesign
            Whatever `design.truncate_to_circuits` returns, so the class is preserved.
            On a design that records its provenance -- any
            :class:`~pygsti.protocols.GateSetTomographyDesign` -- the result's `selection`
            attribute holds the :class:`CircuitSelection` that produced it.
        """
        selection = self.select(design, num_circuits)
        reduced = design.truncate_to_circuits(selection.circuits)
        # Only where the class declared the member: it is written out as a
        # 'serialized-object' auxfile, and setting it on a design that has not registered
        # it would leave a live object for `write` to choke on.
        if hasattr(reduced, 'selection'):
            reduced.selection = selection
        return reduced

    # -- serialization ------------------------------------------------------ #
    #
    # A reducer that holds no configuration needs no serialization code at all: the
    # default below reconstructs it from the module and class name that
    # `NicelySerializable._to_nice_serialization` already records.  A reducer with
    # constructor arguments overrides both halves in the usual way -- see
    # `BlockDoptReducer` for the pattern.

    @classmethod
    def _from_nice_serialization(cls, state: dict[str, Any]) -> DesignReducer:
        return cls()

    @classmethod
    def from_nice_serialization(cls, state: dict[str, Any]) -> DesignReducer:
        """Rebuild the reducer described by `state`, whatever subclass it is.

        Parameters
        ----------
        state : dict
            From a prior :meth:`to_nice_serialization`.

        Returns
        -------
        DesignReducer
        """
        # NicelySerializable's dispatcher reads "the base class supplies
        # _from_nice_serialization" as "the subclass forgot to", and raises
        # NotImplementedError rather than using the default above.  Resolve the concrete
        # class here so that `DesignReducer.from_nice_serialization(...)` works on a
        # subclass that did not need to write any serialization code.
        if cls is DesignReducer:
            target = cls._state_class(state)
            if target is not DesignReducer:
                return target.from_nice_serialization(state)
        return super().from_nice_serialization(state)

    @classmethod
    def cast(cls, obj: Union[DesignReducer, ReducerFunction]) -> DesignReducer:
        """`obj` as a :class:`DesignReducer`: itself, or a callable wrapped in one.

        Parameters
        ----------
        obj : DesignReducer or callable
            A callable is invoked as `obj(design, num_circuits)`; see
            :class:`CallableReducer` for what it may return and for the serialization
            caveat that comes with it.

        Returns
        -------
        DesignReducer
        """
        if isinstance(obj, DesignReducer):
            return obj
        if callable(obj):
            return CallableReducer(obj)
        raise TypeError("A design reducer must be a DesignReducer or a callable "
                        f"(design, num_circuits) -> circuits; got {type(obj).__name__}.")

    # -- validation --------------------------------------------------------- #

    def _validate(self, selection: CircuitSelection, candidates: Sequence[Circuit],
                  num_circuits: Optional[int]) -> None:
        """Check `_select`'s output, or raise explaining what the subclass got wrong."""
        me = type(self).__name__
        if not isinstance(selection, CircuitSelection):
            raise TypeError(
                f"{me}._select must return a CircuitSelection, not "
                f"{type(selection).__name__}. Wrap the circuits: "
                "`return CircuitSelection(chosen)`.")

        chosen = selection.circuits
        if num_circuits is not None and len(chosen) > num_circuits:
            raise ValueError(f"{me}._select returned {len(chosen)} circuits for a budget "
                             f"of {num_circuits}.")

        seen = set()
        duplicates = set()
        for circuit in chosen:
            (duplicates if circuit in seen else seen).add(circuit)
        if duplicates:
            raise ValueError(
                f"{me}._select returned {len(duplicates)} circuit(s) more than once, e.g. "
                f"{_examples(duplicates)}. Selecting a circuit twice does not buy anything "
                "twice, and silently under-fills the budget.")

        foreign = seen - set(candidates)
        if foreign:
            raise ValueError(
                f"{me}._select returned {len(foreign)} circuit(s) that are not in the "
                f"design, e.g. {_examples(foreign)}.\n"
                "If you indexed a Jacobian to get here, map back through "
                "`list(jac_dict)`, not through the circuit list you passed in: "
                "`bulk_dprobs` deduplicates and reorders its input, so row block i "
                "belongs to `list(jac_dict)[i]`.")

        if selection.scores is not None and len(selection.scores) != len(chosen):
            raise ValueError(f"{me}._select returned {len(selection.scores)} scores for "
                             f"{len(chosen)} circuits; they must correspond one-to-one.")

        if len(chosen) == 0 and num_circuits != 0 and candidates:
            raise ValueError(
                f"{me}._select returned no circuits from {len(candidates)} candidates. An "
                "empty design does not fail until something tries to fit it, so it is "
                "rejected here instead.")


def _examples(circuits: Iterable[Circuit], limit: int = 3) -> str:
    """A short, stable sample of `circuits` for an error message."""
    shown = sorted(str(c) for c in circuits)[:limit]
    more = len(circuits) - len(shown)
    return ", ".join(shown) + (f", ... (+{more} more)" if more > 0 else "")


class CallableReducer(DesignReducer):
    """Adapts a plain `f(design, num_circuits)` to the :class:`DesignReducer` interface.

    The low-ceremony option, for a one-off reduction in a notebook or a test.  `f` may
    return a :class:`CircuitSelection` or a bare sequence of circuits.

    Serialization records `f` by module and qualified name, so -- unlike a real
    `DesignReducer` subclass -- it does not round-trip for a lambda, a closure, or a
    function defined interactively.  If a reduced design needs to carry a durable record
    of how it was reduced, subclass :class:`DesignReducer` instead.

    Parameters
    ----------
    func : callable
        Invoked as `func(design, num_circuits)`.
    """

    def __init__(self, func: ReducerFunction) -> None:
        super().__init__()
        self.func = func

    def _select(self, design: ExperimentDesign, num_circuits: Optional[int]) -> CircuitSelection:
        result = self.func(design, num_circuits)
        if isinstance(result, CircuitSelection):
            return result
        return CircuitSelection(result)

    def _to_nice_serialization(self) -> dict[str, Any]:
        state = super()._to_nice_serialization()
        module = getattr(self.func, '__module__', None)
        qualname = getattr(self.func, '__qualname__', None)
        state['func'] = None if (module is None or qualname is None) else f"{module}.{qualname}"
        return state

    @classmethod
    def _from_nice_serialization(cls, state: dict[str, Any]) -> CallableReducer:
        name = state.get('func')
        if name is None:
            raise ValueError("Cannot restore a CallableReducer: its function had no "
                             "module and qualified name to record. Subclass DesignReducer "
                             "for a reducer that reloads.")
        try:
            from pygsti.io.metadir import _class_for_name as _resolve_name
            func = _resolve_name(name)
        except Exception as e:
            # Raised as a ValueError rather than passed through: `_class_for_name` on a
            # lambda's qualname fails with whatever import error the dotted path happens
            # to produce, which says nothing about the actual problem.
            raise ValueError(
                f"Could not restore the CallableReducer function {name!r} ({e}). This is "
                "expected for a lambda, a closure, or a function defined inside another "
                "function -- none of them can be imported by name. Subclass DesignReducer "
                "if a reduced design needs to carry a durable record of how it was "
                "reduced.") from e
        return cls(func)

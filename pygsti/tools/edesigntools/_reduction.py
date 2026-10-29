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

import copy as _copy
from typing import TYPE_CHECKING, Any, Callable, Iterable, Mapping, Optional, Sequence, Union

import numpy as _np

from pygsti.baseobjs.nicelyserializable import NicelySerializable as _NicelySerializable

if TYPE_CHECKING:
    from pygsti.circuits.circuit import Circuit
    from pygsti.protocols.protocol import ExperimentDesign

__all__ = ['CircuitSelection', 'DesignReducer', 'CallableReducer']

#: What :class:`CallableReducer` wraps: called as ``f(design, num_circuits)``,
#: returning a :class:`CircuitSelection` or the circuits to keep.
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
        A one-dimensional array with one float per kept circuit. The reducer defines
        their meaning through `score_name`. For a greedy reducer reporting a
        nondecreasing cumulative objective, a flattening score curve indicates that
        additional circuits contribute little to that objective.

    score_name : str, optional
        What `scores` measures, short enough to label an axis with.

    metadata : dict, optional
        Free-form, reducer-specific.  :meth:`DesignReducer.select` adds
        `'num_candidates'` if the reducer did not.

    qubit_labels : tuple, optional
        The qubit labels of the design the circuits come from.  :meth:`DesignReducer.select`
        sets this, so a reducer need not; pass it only for a selection built by hand and
        attached to a design directly.  A design holds a selection in the labels it was made
        in, and relates them to its own by position; see
        :attr:`~pygsti.protocols.GateSetTomographyDesign.selection`.

    Attributes
    ----------
    reducer : DesignReducer or None
        A deep copy of the reducer, stamped by :meth:`DesignReducer.select` so later
        changes to the original reducer's configuration do not alter this record.
        None if the selection was built by hand. See :class:`CallableReducer` for the
        limitation on functions that depend on mutable state outside the reducer.
    """

    def __init__(self, circuits: Iterable[Circuit], scores: Optional[Sequence[float]] = None,
                 score_name: Optional[str] = None, metadata: Optional[Mapping[str, Any]] = None,
                 qubit_labels: Optional[Sequence[Any]] = None) -> None:
        super().__init__()
        circuits = tuple(circuits)
        seen = set()
        duplicates = set()
        for circuit in circuits:
            (duplicates if circuit in seen else seen).add(circuit)
        if duplicates:
            raise ValueError(
                f"CircuitSelection got {len(duplicates)} circuit(s) more than once, e.g. "
                f"{_examples(duplicates)}. Selecting a circuit twice does not buy anything "
                "twice, and silently under-fills the budget.")
        if scores is not None:
            scores = _np.array(scores, dtype=_np.float64)
            if scores.ndim != 1:
                raise ValueError(f"CircuitSelection got scores with shape {scores.shape}; scores "
                                 "must be one-dimensional with one scalar per circuit.")
            if len(scores) != len(circuits):
                raise ValueError(f"CircuitSelection got {len(scores)} scores for {len(circuits)} "
                                 "circuits; they must correspond one-to-one.")
            scores.setflags(write=False)
        # Read-only: a design that holds this selection checked these against its own
        # circuits when it took it, and has no way to check again.
        self._circuits: tuple[Circuit, ...] = circuits
        self._scores: Optional[_np.ndarray] = scores
        self._qubit_labels: Optional[tuple[Any, ...]] = None if qubit_labels is None else tuple(qubit_labels)
        self.score_name: Optional[str] = score_name
        self.metadata: dict[str, Any] = dict(metadata) if metadata else {}
        self.reducer: Optional[DesignReducer] = None

    @property
    def circuits(self) -> tuple[Circuit, ...]:
        """The circuits kept, best-first where the reducer has a notion of "best"."""
        return self._circuits

    @property
    def scores(self) -> Optional[_np.ndarray]:
        """One score per circuit, or None; a read-only array."""
        return self._scores

    @property
    def qubit_labels(self) -> Optional[tuple[Any, ...]]:
        """The qubit labels of the design the circuits were selected from."""
        return self._qubit_labels

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
            'qubit_labels': None if self.qubit_labels is None else list(self.qubit_labels),
        })
        return state

    @classmethod
    def _from_nice_serialization(cls, state: dict[str, Any]) -> CircuitSelection:
        from pygsti.io.readers import convert_strings_to_circuits as _to_circuits
        ret = cls(_to_circuits(state['circuits']), state['scores'],
                  state['score_name'], state['metadata'], state.get('qubit_labels'))
        if state.get('reducer') is not None:
            ret.reducer = DesignReducer.from_nice_serialization(state['reducer'])
        return ret


class DesignReducer(_NicelySerializable):
    """A rule for choosing which of an experiment design's circuits are worth keeping.

    Subclasses implement :meth:`_select` to choose circuits. Callers use :meth:`select`
    for diagnostics or :meth:`reduce` for a smaller design. Both validate the selection.

    A reducer is a fully configured policy: models, seeds and weightings are constructor
    arguments, so :meth:`select` has one signature for every reducer. That configuration
    must live on the instance and support :func:`copy.deepcopy`, because each selection
    records a deep copy of its reducer.

    Serialization needs no registration, but the subclass must be importable when
    loading. A subclass with configuration implements `_to_nice_serialization` and
    `_from_nice_serialization`, as :class:`BlockDoptReducer` does; the default loader
    calls the constructor with no arguments.
    """

    # -- the selection rule ------------------------------------------------- #

    def _select(self, design: ExperimentDesign, num_circuits: Optional[int]) -> CircuitSelection:
        """Choose circuits from `design`; return a :class:`CircuitSelection`.

        `num_circuits` is already clamped to the number of candidates, or None for "use
        your own stopping rule" (raise `ValueError` if there is none). Returning fewer
        circuits than the budget is allowed; more, duplicates, or foreign circuits are not.
        """
        raise NotImplementedError("DesignReducer subclasses must implement _select.")

    # -- what callers use --------------------------------------------------- #

    def select(self, design: ExperimentDesign, num_circuits: Optional[int] = None) -> CircuitSelection:
        """The circuits this reducer would keep, with its diagnostics.

        Parameters
        ----------
        design : ExperimentDesign
            Passed to `_select` whole, so a reducer can use the design's structure.

        num_circuits : int, optional
            The budget, clamped to the number of candidates.  None asks the reducer to
            choose for itself.

        Returns
        -------
        CircuitSelection
            Includes a deep copy of this reducer taken after `_select` returns.
        """
        candidates = list(design.all_circuits_needing_data)
        if num_circuits is not None:
            num_circuits = int(num_circuits)
            if num_circuits < 0:
                raise ValueError(f"num_circuits must be nonnegative, got {num_circuits}.")
            num_circuits = min(num_circuits, len(candidates))

        selection = self._select(design, num_circuits)
        self._validate(selection, candidates, num_circuits)
        labels = design.qubit_labels
        selection._qubit_labels = None if isinstance(labels, str) else tuple(labels)
        selection.reducer = _copy.deepcopy(self)
        selection.metadata.setdefault('num_candidates', len(candidates))
        return selection

    def reduce(self, design: ExperimentDesign, num_circuits: Optional[int] = None) -> ExperimentDesign:
        """A copy of `design` keeping only the circuits this reducer selects.

        Designs with child experiments raise `NotImplementedError`, because truncating the
        root would leave the children inconsistent; :meth:`select` still works on them.

        Parameters
        ----------
        design : ExperimentDesign
            Not modified.

        num_circuits : int, optional
            As for :meth:`select`.

        Returns
        -------
        ExperimentDesign
            Of the same class as `design`. If that class declares a `selection`
            member, as :class:`~pygsti.protocols.GateSetTomographyDesign` does, it holds
            the :class:`CircuitSelection`.
        """
        selection = self.select(design, num_circuits)
        return self._apply_selection(design, selection)

    def _apply_selection(self, design: ExperimentDesign, selection: CircuitSelection) -> ExperimentDesign:
        """Apply a validated result of :meth:`select`, preserving its provenance."""
        if design.keys():
            raise NotImplementedError(
                "DesignReducer.reduce does not support designs with child experiments; "
                "root-only truncation would leave their circuit requirements inconsistent. "
                "Use select to inspect the root circuit selection without applying it.")
        reduced = design.truncate_to_circuits(selection.circuits)
        # Only where the class declares the member, whose setter checks the selection and
        # registers its auxfile. Setting a plain attribute on any other design would
        # leave a live object for `write` to choke on.
        if hasattr(reduced, 'selection'):
            reduced.selection = selection
        return reduced

    # -- serialization ------------------------------------------------------ #

    @classmethod
    def _from_nice_serialization(cls, state: dict[str, Any]) -> DesignReducer:
        return cls()

    @classmethod
    def from_nice_serialization(cls, state: dict[str, Any]) -> DesignReducer:
        """Rebuild the reducer described by `state`, whatever subclass it is."""
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

        # Duplicates and the shape of `scores` are checked by CircuitSelection itself.
        foreign = set(chosen) - set(candidates)
        if foreign:
            raise ValueError(
                f"{me}._select returned {len(foreign)} circuit(s) that are not in the "
                f"design, e.g. {_examples(foreign)}.\n"
                "If you indexed a Jacobian to get here, map back through "
                "`list(jac_dict)`, not through the circuit list you passed in: "
                "`bulk_dprobs` deduplicates and reorders its input, so row block i "
                "belongs to `list(jac_dict)[i]`.")

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

    Serialization records `f` by module and qualified name, so reloading fails for
    lambdas, closures, and functions defined interactively. A function's closure and
    global state are not deep-copied into the selection's record. For a durable record,
    subclass :class:`DesignReducer`.

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

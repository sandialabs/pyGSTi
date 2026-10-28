"""
Shared type aliases and `TypeVar` definitions for the error generator propagation subpackage.

This is a leaf module: it is imported by the other submodules of `pygsti.tools.errorgenpropagation`
but imports none of them. The optional `stim` dependency is guarded so that importing this module
never fails when `stim` is not installed. Because top-level type aliases are evaluated eagerly at
runtime (`from __future__ import annotations` does not defer them), the runtime fallback binds
`stim` to a namespace whose attributes are `Any`; the static-analysis branch sees the real `stim`
types. `LocalStimErrorgenLabel` is referenced via string forward references for the same reason.
"""
#***************************************************************************************************
# Copyright 2015, 2019, 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************
from types import SimpleNamespace as _SimpleNamespace
from typing import Any, Callable, Iterable, TypeVar, Union, TYPE_CHECKING

if TYPE_CHECKING:
    import stim
    import numpy as _np
else:
    try:
        import stim
    except ImportError:
        # Placeholder so that the aliases below can be built without stim; every stim name is Any.
        stim = _SimpleNamespace(PauliString=Any, Tableau=Any, TableauSimulator=Any)

__all__ = [
    'PauliPhaseUpdater',
    'PauliPhaseZerosUpdater',
    'AmplitudeOfStateType',
    'BulkAmplitudeOfStateType',
    'BulkPhiType',
    'BulkAlphaType',
    'BulkAlphaPauliType',
    '_ErrorgenTerms',
    '_ErrorgenDict',
    '_Rate',
    '_SignedPauli',
    '_PairHandler',
    '_F',
]

# A list of (error generator label, rate) pairs, as produced by the commutator and
# composition routines. Rates may be complex prior to aggregation.
_ErrorgenTerms = list[tuple['LocalStimErrorgenLabel', complex]]

# A dictionary of error generator rates keyed by label, as taken and returned by the drivers
# (BCH, Magnus, Zassenhaus, Taylor) for one layer or one order.
_ErrorgenDict = dict['LocalStimErrorgenLabel', float]

# Rate TypeVar: per-order accumulators hold complex rates until the final merge takes the real part.
_Rate = TypeVar('_Rate', float, complex)

# A signed Pauli w P: `(w, P)` with P an unsigned stim.PauliString and w the phase, or `(w, P, s)`
# with s = bel_str(P) already rendered.
_SignedPauli = Union[tuple[complex, stim.PauliString], tuple[complex, stim.PauliString, str]]

# The signature shared by the `_commutator_XY` and `_composition_XY` handlers:
# (errorgen_1, errorgen_2, weight, identity string) -> terms.
_PairHandler = Callable[['LocalStimErrorgenLabel', 'LocalStimErrorgenLabel', complex, str], _ErrorgenTerms]

_F = TypeVar('_F', bound=Callable)

PauliPhaseUpdater = Callable[[str, str, Union[bool, None]], tuple[complex, str]]
PauliPhaseZerosUpdater = Callable[[str, Union[bool, None]], tuple[complex, str]]
AmplitudeOfStateType = Callable[[Union[stim.Tableau, stim.TableauSimulator], str, bool], complex]
BulkAmplitudeOfStateType = Callable[[Union[stim.Tableau, stim.TableauSimulator], list[Union[str, stim.PauliString]], bool], '_np.ndarray']
BulkPhiType = Callable[[Union[stim.Tableau, stim.TableauSimulator], str, list[Union[str, stim.PauliString]], list[Union[str, stim.PauliString]]], '_np.ndarray']
BulkAlphaType = Callable[[Iterable['LocalStimErrorgenLabel'], Union[stim.Tableau, stim.TableauSimulator], list[str]], '_np.ndarray']
BulkAlphaPauliType = Callable[[Iterable['LocalStimErrorgenLabel'], Union[stim.Tableau, stim.TableauSimulator], list[stim.PauliString]], '_np.ndarray']

"""
Deprecated backwards-compatibility shim for `pygsti.tools.errorgenpropagation.fasterrgencalc`.

The optional Cython accelerations now live in the `pygsti.tools.errorgenpropagation.fasterrgencalc`
submodule. This module forwards every attribute lookup to that submodule (PEP 562 `__getattr__`) and
emits a single `pyGSTiDeprecationWarning` per Python session on first attribute access. Importing this
module alone does not emit a warning. If the Cython extension is not built, attribute access raises
`ImportError`. It will be removed in a future release; import from
`pygsti.tools.errorgenpropagation.fasterrgencalc` instead.
"""
#***************************************************************************************************
# Copyright 2015, 2019, 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

import warnings
import pygsti.tools.errorgenpropagation as _pkg
from pygsti.tools.exceptions import pyGSTiDeprecationWarning

# None if the optional Cython extension has not been built.
_target = _pkg.fasterrgencalc

_WARNED = False


def _warn_once():
    """Emit the deprecation warning the first time it is called in a session."""
    global _WARNED
    if not _WARNED:
        _WARNED = True
        warnings.warn(
            "pygsti.tools.fasterrgencalc is deprecated and will be removed in a future release. "
            "It has moved to pygsti.tools.errorgenpropagation.fasterrgencalc.",
            pyGSTiDeprecationWarning, stacklevel=3)


def __getattr__(name):
    _warn_once()
    if _target is None:
        raise ImportError("The optional Cython extension pygsti.tools.errorgenpropagation.fasterrgencalc "
                          "is not built, so `pygsti.tools.fasterrgencalc` is unavailable.")
    return getattr(_target, name)


def __dir__():
    names = set(globals())
    if _target is not None:
        names |= set(dir(_target))
    return sorted(names)

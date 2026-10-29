"""
Deprecated backwards-compatibility shim for `pygsti.tools.errorgenpropagation.polynomials`.

The contents of this module now live in the `pygsti.tools.errorgenpropagation.polynomials`
submodule (with `pygsti.tools.errorgenpropagation` as a fallback for any other name). This module
forwards every attribute lookup to those locations (PEP 562 `__getattr__`) and emits a single
`pyGSTiDeprecationWarning` per Python session on first attribute access. Importing this module alone
does not emit a warning. It will be removed in a future release; import from
`pygsti.tools.errorgenpropagation` (or `pygsti.tools.errorgenpropagation.polynomials`) instead.
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
import pygsti.tools.errorgenpropagation.polynomials as _target
from pygsti.tools.exceptions import pyGSTiDeprecationWarning

_WARNED = False


def _warn_once():
    """Emit the deprecation warning the first time it is called in a session."""
    global _WARNED
    if not _WARNED:
        _WARNED = True
        warnings.warn(
            "pygsti.tools.errgenpolytools is deprecated and will be removed in a future release. "
            "Import from pygsti.tools.errorgenpropagation (or pygsti.tools.errorgenpropagation.polynomials) "
            "instead.",
            pyGSTiDeprecationWarning, stacklevel=3)


def __getattr__(name):
    _warn_once()
    if hasattr(_target, name):
        return getattr(_target, name)
    return getattr(_pkg, name)


def __dir__():
    return sorted(set(globals()) | set(dir(_target)) | set(dir(_pkg)))

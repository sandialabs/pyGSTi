"""
Explanations for imports that fail because glibc's static TLS reserve is exhausted.
"""
#***************************************************************************************************
# Copyright 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************
#
# Some compiled libraries keep their thread-local variables in "static" TLS (the initial-exec
# model). glibc serves libraries that are loaded after startup, which includes every Python
# extension module, from a small fixed reserve of it, and fails the load once the reserve is used
# up. torch, qiskit, qiskit-aer, and the OpenMP runtimes they bundle all draw on that reserve, so
# loading enough of them into one process fails with "cannot allocate memory in static TLS block",
# whichever of them happens to be loaded last. The dynamic loader reads GLIBC_TUNABLES only at
# process startup, so the reserve can't be enlarged from inside Python; all we can do is explain.

import importlib as _importlib
import os as _os
import re as _re
import warnings as _warnings

from pygsti.tools.exceptions import StaticTLSImportError, StaticTLSWarning

_GLIBC_MESSAGE = 'static TLS block'
_TUNABLE = 'glibc.rtld.optional_static_tls'
_RECOMMENDED_BYTES = 4096


def is_static_tls_error(err):
    """True if `err` is the dynamic loader's "cannot allocate memory in static TLS block" error."""
    return isinstance(err, ImportError) and _GLIBC_MESSAGE in str(err)


def _recommended_setting():
    """The GLIBC_TUNABLES value to recommend, given the one this process started with."""
    current = _os.environ.get('GLIBC_TUNABLES', '')
    match = _re.search(_re.escape(_TUNABLE) + r'=(\d+)', current)
    nbytes = _RECOMMENDED_BYTES
    if match is not None and int(match.group(1)) >= nbytes:
        nbytes = 2 * int(match.group(1))
    setting = f'{_TUNABLE}={nbytes}'
    others = [t for t in current.split(':') if t and not t.startswith(_TUNABLE + '=')]
    return ':'.join(others + [setting]), current


def static_tls_explanation(package, err):
    """
    A message explaining why importing `package` failed with `err`, and how to fix it.

    Parameters
    ----------
    package : str
        The name of the package whose import failed, e.g. 'torch'.

    err : ImportError
        The original error, for which `is_static_tls_error(err)` is True.

    Returns
    -------
    str
    """
    value, current = _recommended_setting()
    started_with = (f"\n\nThis process was started with GLIBC_TUNABLES={current}, which is not enough."
                    if current else "")
    return (
        f"{package} is installed but could not be imported, because this process has run out of "
        f"glibc's reserve of static thread-local storage (TLS). Libraries loaded after Python "
        f"starts share that small, fixed reserve, and torch, qiskit, qiskit-aer, and the OpenMP "
        f"runtimes they bundle all draw on it. Loading enough of them into one process fails, "
        f"whichever one comes last. This is a limit of glibc on Linux rather than a bug in "
        f"{package} or pyGSTi.{started_with}\n\n"
        f"To fix it, restart Python with this environment variable set:\n\n"
        f"    GLIBC_TUNABLES={value}\n\n"
        f"It must be set before the Python process starts; setting it from inside Python (e.g. "
        f"with os.environ) has no effect. For a Jupyter kernel, set it in the environment the "
        f"notebook server is launched from, or in the \"env\" entry of the kernel spec.\n\n"
        f"Original error: {err}"
    )


def import_explaining_static_tls(name):
    """
    Import and return the module `name`, explaining a static TLS failure if one occurs.

    Raises
    ------
    StaticTLSImportError
        If the import fails because the static TLS reserve is exhausted. Other import errors
        propagate unchanged.
    """
    try:
        return _importlib.import_module(name)
    except ImportError as err:
        if is_static_tls_error(err):
            package = name.split('.')[0]
            raise StaticTLSImportError(static_tls_explanation(package, err), name=package) from err
        raise


def import_torch():
    """Import and return torch, explaining a static TLS failure if one occurs."""
    return import_explaining_static_tls('torch')


def warn_if_static_tls(err, package, stacklevel=2):
    """
    Emit a StaticTLSWarning explaining `err` if it is a static TLS failure.

    For call sites that probe for an optional dependency and carry on without it, where raising
    would be wrong but silently treating an installed package as missing would mislead.
    """
    if is_static_tls_error(err):
        _warnings.warn(static_tls_explanation(package, err), StaticTLSWarning, stacklevel=stacklevel + 1)

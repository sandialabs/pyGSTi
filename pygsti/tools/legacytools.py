"""
Functions related deprecating other functions
"""
#***************************************************************************************************
# Copyright 2015, 2019, 2025 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

import functools as _functools
import os as _os
import sys as _sys
import types as _types
import warnings as _warnings
from pygsti.tools.exceptions import pyGSTiDeprecationWarning as _pyGSTiDeprecationWarning

# The pygsti package directory; the trailing separator keeps a sibling such as `pygsti_scripts/`
# from matching.
_PYGSTI_DIR = _os.path.dirname(_os.path.dirname(__file__)) + _os.sep
# Tests set this to False to run the Python 3.10/3.11 path on newer Pythons.
_HAS_SKIP_FILE_PREFIXES = _sys.version_info >= (3, 12)

#: Deprecation message shared by `TermForwardSimulator` and the term-based machinery that exists
#: solely to support it (see https://github.com/sandialabs/pyGSTi/issues/916).  For use with
#: :func:`deprecate` / :func:`warn_deprecated`, which replace ``{name}`` with the deprecated
#: object's name.  (Defined here, rather than near the term code, to avoid import cycles.)
#: Remove together with the term-based machinery in pyGSTi 0.12.
TERM_SIM_DEPRECATION_MESSAGE = (
    "{name} is deprecated as of pyGSTi 0.11 and will be removed in pyGSTi 0.12, together with "
    "TermForwardSimulator and the term-based (path-integral) simulation machinery that supports it. "
    "See https://github.com/sandialabs/pyGSTi/issues/916.")


def warn_deprecated(name, replacement=None, message=None):
    """
    Formats and issues a deprecation warning message.

    The warning is attributed to the first caller outside pyGSTi, so it shows
    under Python's default filters when that caller is the user's script or
    notebook, however deep inside pyGSTi the call was made.

    Parameters
    ----------
    name : str
        The name of the function (or other object) that is now deprecated.

    replacement : str, optional
        the name of the function that should replace it.  Ignored if
        `message` is given.

    message : str, optional
        A custom message that replaces the default message entirely.  Any
        occurrence of the literal substring ``{name}`` is replaced by `name`
        (plain substitution, not ``str.format``, so other braces are safe).

    Returns
    -------
    None
    """
    if message is not None:
        message = message.replace('{name}', name)
    else:
        message = 'The function {} is deprecated, and may not be present in future versions of pygsti.'.format(name)
        if replacement is not None:
            message += '\n    '
            message += 'Please use {} instead.'.format(replacement)
    if _HAS_SKIP_FILE_PREFIXES:
        _warnings.warn(message, _pyGSTiDeprecationWarning, skip_file_prefixes=(_PYGSTI_DIR,))
    else:  # Python 3.10, 3.11: count the frames up to the first one outside pygsti
        frame, stacklevel = _sys._getframe(), 1
        while frame is not None and frame.f_code.co_filename.startswith(_PYGSTI_DIR):
            frame, stacklevel = frame.f_back, stacklevel + 1
        del frame  # break the frame/locals reference cycle
        _warnings.warn(message, _pyGSTiDeprecationWarning, stacklevel=stacklevel)


def deprecate(replacement=None, message=None, suppress_nested=False):
    """
    Decorator for deprecating a function.

    The returned wrapper preserves the wrapped function's metadata
    (``__name__``, ``__qualname__``, ``__doc__``, etc.) via
    :func:`functools.wraps`, so docstrings (including any
    ``.. deprecated::`` notices) remain visible to ``help()`` and Sphinx.

    The decorator can also be applied to methods, and underneath
    ``@property`` / ``@classmethod`` / ``@staticmethod``, e.g.::

        @property
        @deprecate(message="{name} is deprecated.")
        def foo(self): ...

    Parameters
    ----------
    replacement : str, optional
        the name of the function that should replace it.  Ignored if
        `message` is given.

    message : str, optional
        A custom warning message that replaces the default one.  Any literal
        ``{name}`` substring is replaced by the wrapped function's
        ``__qualname__`` (e.g. ``"ComposedOp.taylor_order_terms"``), with
        any ``<locals>`` prefix removed.

    suppress_nested : bool, optional
        If True, `pyGSTiDeprecationWarning`s raised *while executing* the
        deprecated function are silenced, so that a call issues exactly one
        warning (attributed to the first caller outside pyGSTi, as for
        :func:`warn_deprecated`) even when the deprecated function
        itself calls other deprecated functions.  Useful for families of
        deprecated methods that call one another.

    Returns
    -------
    function
    """
    if callable(replacement) and message is None and not suppress_nested:
        # Used as a bare decorator (`@deprecate` instead of `@deprecate()`), so `replacement` is
        # actually the function being deprecated.
        return deprecate()(replacement)

    def decorator(fn):
        if message is not None:
            # e.g. "ComposedOp.taylor_order_terms"; drop any "f.<locals>." prefix
            name = fn.__qualname__.rsplit('<locals>.', 1)[-1]
        else:
            name = fn.__name__  # legacy behavior

        @_functools.wraps(fn)
        def _inner(*args, **kwargs):
            warn_deprecated(name, replacement, message=message)
            if not suppress_nested:
                return fn(*args, **kwargs)
            with _warnings.catch_warnings():
                _warnings.simplefilter("ignore", _pyGSTiDeprecationWarning)
                return fn(*args, **kwargs)
        return _inner
    return decorator


def deprecate_imports(module_name, replacement_map, warning_msg):
    """
    Utility to deprecate imports from a module.

    This works by swapping the underlying module in the import
    mechanisms with a `ModuleType` object that overrides attribute
    lookup to check against the replacement map.

    Note that this will slow down module attribute lookup
    substantially. If you need to deprecate multiple names, DO NOT
    call this method more than once on a given module! Instead, use
    the replacement map to batch multiple deprecations into one
    call. When using this method, plan to remove the deprecated paths
    altogether sooner rather than later.

    Parameters
    ----------
    module_name : str
        The fully-qualified name of the module whose names have been deprecated.

    replacement_map : {name: function}
        A map of each deprecated name to a factory which will be
        called with no arguments when importing the name.

    warning_msg : str
        A message to be displayed as a warning when importing a
        deprecated name. Optionally, this may include the format
        string `name`, which will be formatted with the deprecated
        name.

    Returns
    -------
    None
    """
    module = _sys.modules[module_name]

    class ModuleLookupWrapper(_types.ModuleType):
        def __getattribute__(self, name):
            if name in replacement_map:
                _warnings.warn(warning_msg.format(name=name))
                return replacement_map[name]()
            else:
                return module.__getattribute__(name)

    _sys.modules[module_name] = ModuleLookupWrapper(module_name)

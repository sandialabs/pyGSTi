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
import sys as _sys
import types as _types
import warnings as _warnings
from pygsti.tools.exceptions import pyGSTiDeprecationWarning as _pyGSTiDeprecationWarning


def warn_deprecated(name, replacement=None, message=None, stacklevel=2):
    """
    Formats and issues a deprecation warning message.

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

    stacklevel : int, optional
        Passed to :func:`warnings.warn`.  The default, 2, attributes the
        warning to the code that called `warn_deprecated`.

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
    _warnings.warn(message, _pyGSTiDeprecationWarning, stacklevel=stacklevel)


def deprecate(replacement=None, message=None):
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

    Returns
    -------
    function
    """
    def decorator(fn):
        if message is not None:
            # e.g. "ComposedOp.taylor_order_terms"; drop any "f.<locals>." prefix
            name = fn.__qualname__.rsplit('<locals>.', 1)[-1]
        else:
            name = fn.__name__  # legacy behavior

        @_functools.wraps(fn)
        def _inner(*args, **kwargs):
            # stacklevel=3 attributes the warning to the caller of the deprecated function
            # (1 = warn_deprecated, 2 = _inner, 3 = caller).
            warn_deprecated(name, replacement, message=message, stacklevel=3)
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

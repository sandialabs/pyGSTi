"""
Functions related deprecating other functions
"""
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

import sys as _sys
import types as _types
import warnings as _warnings


def warn_deprecated(name, replacement=None):
    """
    Formats and prints a deprecation warning message.

    Parameters
    ----------
    name : str
        The name of the function that is now deprecated.

    replacement : str, optional
        the name of the function that should replace it.

    Returns
    -------
    None
    """
    message = 'The function {} is deprecated, and may not be present in future versions of pygsti.'.format(name)
    if replacement is not None:
        message += '\n    '
        message += 'Please use {} instead.'.format(replacement)
    _warnings.warn(message)


def deprecate(replacement=None):
    """
    Decorator for deprecating a function.

    Parameters
    ----------
    replacement : str, optional
        the name of the function that should replace it.

    Returns
    -------
    function
    """
    def decorator(fn):
        def _inner(*args, **kwargs):
            warn_deprecated(fn.__name__, replacement)
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

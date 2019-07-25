""" Functions related deprecating other functions """
from __future__ import division, print_function, absolute_import, unicode_literals
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

import warnings as _warnings

from ..baseobjs import parameterized


def warn_deprecated(name, replacement=None):
    """
    Formats and prints a deprecation warning message.

    Parameters
    ----------
    name : str
        The name of the function that is now deprecated.

    replacement : str, optional
        the name of the function that should replace it.
    """
    message = 'The function {} is deprecated, and may not be present in future versions of pygsti.'.format(name)
    if replacement is not None:
        message += '\n    '
        message += 'Please use {} instead.'.format(replacement)
    _warnings.warn(message)


@parameterized
def deprecated_fn(fn, replacement=None):
    """
    Decorator for deprecating a function.

    Parameters
    ----------
    fn : function
        The function that is now deprecated.

    replacement : str, optional
        the name of the function that should replace it.
    """
    def _inner(*args, **kwargs):
        warn_deprecated(fn.__name__, replacement)
        return fn(*args, **kwargs)
    return _inner

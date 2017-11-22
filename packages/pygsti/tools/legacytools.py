""" Functions related deprecating other functions """
from __future__ import division, print_function, absolute_import, unicode_literals
#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************

import warnings  as _warnings

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

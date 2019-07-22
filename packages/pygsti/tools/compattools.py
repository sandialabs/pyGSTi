""" Functions for Python2 / Python3 compatibility """
from __future__ import division, print_function, absolute_import, unicode_literals
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

import numbers as _numbers
from contextlib import contextmanager as _contextmanager
import uuid as _uuid

#Define basestring in python3 so unicode
# strings can be tested for in python2 using
# python2's built-in basestring type.
# When removing __future__ imports, remove
# this and change basestring => str below.
try: basestring
except NameError: basestring = str


def isint(x):
    """ Return whether `x` has an integer type """
    return isinstance(x, _numbers.Integral)


def isstr(x):
    """ Return whether `x` has a string type """
    return isinstance(x, basestring)


def _numpy14einsumfix():
    """ str(.) on first arg of einsum skirts a bug in Numpy 14.0 """
    import numpy as _np
    if _np.__version__ == '1.14.0':
        def fixed_einsum(s, *args, **kwargs):
            return _np.orig_einsum(str(s), *args, **kwargs)
        _np.orig_einsum = _np.einsum
        _np.einsum = fixed_einsum

#Worse way to do this
#import sys as _sys
#
#if _sys.version_info > (3, 0): # Python3?
#    longT = int      # define long and unicode
#    unicodeT = str   #  types to mimic Python2
#else:
#    longT = long
#    unicodeT = unicode
#
#def isint(x):
#    return isinstance(x,(int,longT))
#
#def isstr(x):
#    return isinstance(x,(str,unicodeT))


@_contextmanager
def patched_UUID():
    """Monkeypatch the uuid module with a fake SafeUUID

    This is a workaround for unpickling objects pickled in later python
    versions.

    TODO: objects should be serialized correctly and this should be deprecated.
    """
    if 'SafeUUID' not in dir(_uuid):
        class dummy_SafeUUID(object):
            def __new__(self, *args):
                return _uuid.UUID.__new__(_uuid.UUID, *args)
        _uuid.SafeUUID = dummy_SafeUUID

        yield  # context block

        _uuid.__delattr__('SafeUUID')
    else:
        yield

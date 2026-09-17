"""
Tools for general compatibility.
"""
#***************************************************************************************************
# Copyright 2015, 2019, 2025 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

import numbers as _numbers

import numpy as _np
from numpy.lib import NumpyVersion as _NumpyVersion

# The ``copy`` keyword of ``numpy.reshape`` was added in NumPy 2.1; direct
# assignment to ``ndarray.shape`` was deprecated in NumPy 2.5.
_NUMPY_HAS_RESHAPE_COPY = _NumpyVersion(_np.__version__) >= '2.1.0'


def reshape_no_copy(a, shape):
    """
    Return a view of `a` with the given `shape` without copying its data.

    Forward-compatible replacement for the idiom ``a.shape = shape``, which was
    deprecated in NumPy 2.5. Like that idiom, this raises (rather than copying)
    if the requested shape is incompatible with `a`'s memory layout, preserving
    the no-copy guarantee that callers rely on.

    Parameters
    ----------
    a : numpy.ndarray
        Array to reshape.

    shape : int or tuple or list of int
        The new shape. May be a single int, or a tuple/list of ints that may
        contain a single ``-1`` to infer one dimension.

    Returns
    -------
    numpy.ndarray
        A view of `a` (sharing its memory) with the requested shape.
    """
    if _NUMPY_HAS_RESHAPE_COPY:            # numpy >= 2.1
        return _np.reshape(a, shape, copy=False)
    view = a.view()                        # numpy < 2.1: copy= kwarg unavailable
    view.shape = shape                     # in-place; raises if a copy is needed; not deprecated < 2.5
    return view


def isint(x):
    """
    Return whether `x` has an integer type.

    `numbers.Integral` is the ABC to which most integral types are
    registered, including `int` and all `numpy.int` variants. This
    function should be used in place of `isinstance(x, int)` or
    similar.

    Parameters
    ----------
    x : object
        Object to check.

    Returns
    -------
    bool
    """
    return isinstance(x, _numbers.Integral)


def _numpy14einsumfix():
    """ str(.) on first arg of einsum skirts a bug in Numpy 14.0 """
    import numpy as _np
    if _np.__version__ == '1.14.0':
        def fixed_einsum(s, *args, **kwargs):
            return _np.orig_einsum(str(s), *args, **kwargs)
        _np.orig_einsum = _np.einsum
        _np.einsum = fixed_einsum

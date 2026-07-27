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

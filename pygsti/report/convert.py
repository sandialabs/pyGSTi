"""
This module defines type-differentiation for low level formatting types.

Its main function, convert, takes any item x, a specs dictionary, and a format (ie 'html')
and returns a formatted version of x using the format
"""
import functools

import numpy as _np

from pygsti.report.reportableqty import ReportableQty as _ReportableQty
# ***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
# ***************************************************************************************************
from . import html
from . import latex
from . import python
from ..modelmembers import operations as _op
from ..modelmembers import povms as _povm
from ..modelmembers import states as _state


def functions_in(module):
    """
    Create a dictionary of the functions in a module

    Parameters
    ----------
    module : Module
        module to run on.

    Returns
    -------
    dict
    """
    return {name: f for name, f in module.__dict__.items() if callable(f)}


convert_dict = {
    'html': functions_in(html),
    'latex': functions_in(latex),
    'python': functions_in(python)}


def calc_dim(x):
    """
    Calculate the dimension of some matrix-like type

    Parameters
    ----------
    x : matrix-like
        The object to get the dimension of.

    Returns
    -------
    int
    """
    d = 0
    for l in x.shape:
        if l > 1: d += 1
    return d


def item_type(x):
    """
    Differentiate an item's type

    Parameters
    ----------
    x : anything
        Value to convert.

    Returns
    -------
    string
        name of low-level formatter to use (i.e. value or matrix)
    """
    if isinstance(x, _ReportableQty):
        return 'reportable'
    if isinstance(x, (_np.ndarray, _op.LinearOperator, _state.State, _povm.POVMEffect)):
        d = calc_dim(x)
        if d == 0: return 'value'
        if d == 1: return 'vector'
        if d == 2: return 'matrix'
        raise ValueError("I don't know how to render a rank %d numpy array as html" % d)
    elif type(x) in (float, int, complex, _np.float64, _np.int64):
        return 'value'
    elif type(x) in (list, tuple):
        return 'list'
    elif isinstance(x, str):
        return 'escaped'
    else:
        return 'raw'


def convert(x, specs, fmt):
    """
    Convert any item to a format

    Parameters
    ----------
    x : object
        The object to convert.

    specs : dictionary
        Dictionary of user-specified and default parameters to formatting

    fmt : str
        The format to convert to, e.g. `"html"` or `"latex"`.

    Returns
    -------
    str
        `x` rendered as `fmt`.
    """

    #Squeeze arrays before formatting
    if isinstance(x, (_np.ndarray, _op.LinearOperator, _state.State, _povm.POVMEffect)):
        x = _np.squeeze(x)

    t = item_type(x)
    if t == 'raw':
        print('WARNING: {} not explicitly converted to {}'.format(x, fmt))
        return str(x)
    if t == 'reportable':
        return x.render_with(lambda a, specz: convert(a, specz, fmt))
    if t == 'list':
        return convert_dict[fmt][t]([convert(xi, specs, fmt) for xi in x], specs)
    return convert_dict[fmt][t](x, specs)


def converter(fmt):
    """
    Create a converter function for some specific format

    Parameters
    ----------
    fmt : str
        The format to convert to, e.g. `"html"` or `"latex"`.

    Returns
    -------
    function
    """
    return functools.partial(convert, fmt=fmt)

from __future__ import division, print_function, absolute_import, unicode_literals
#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************
from . import html
from . import latex

import numpy as _np
import cmath
import functools
from .. import objects as _objs
from ..tools import compattools as _compat
from .reportables import ReportableQty as _ReportableQty

def functions_in(module):
    return { name : f for name, f in module.__dict__.items() if callable(f)}

formatDict = {
        'html'  : functions_in(html),
        'latex' : functions_in(latex)}

def calc_dim(x):
    d = 0
    for l in x.shape:
        if l > 1: d += 1
    return d

def item_type(x):
    """

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
    if isinstance(x,_np.ndarray) or \
       isinstance(x,_objs.Gate) or \
       isinstance(x,_objs.SPAMVec):
        d = calc_dim(x)
        x = _np.squeeze(x)
        if d == 0: return 'value' 
        if d == 1: return 'vector' 
        if d == 2: return 'matrix' 
        raise ValueError("I don't know how to render a rank %d numpy array as html" % d)
    elif type(x) in (float,int,complex,_np.float64,_np.int64):
        return 'value'
    elif type(x) in (list,tuple):
        return 'list'
    elif _compat.isstr(x):
        return 'escaped'
    else:
        return 'raw'

def convert(x, specs, fmt):
    t = item_type(x)
    if t == 'raw':
        print('WARNING: {} not explicitly converted to {}'.format(x, fmt))
    if t == 'reportable':
        return x.render_with(lambda a : convert(a, specs, fmt))
    if t == 'list':
        return formatDict[fmt][t]([convert(xi, specs, fmt) for xi in x], specs)
    return formatDict[fmt][t](x, specs)

def sub_convert(fmt):
    return functools.partial(convert, fmt=fmt)


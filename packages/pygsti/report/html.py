from __future__ import division, print_function, absolute_import, unicode_literals
#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************
"""
Routines for converting python objects to HTML.  Parallel rountines as
LatexUtil has for latex conversion.
"""

import numpy as _np
import cmath
from .. import objects as _objs
from ..tools import compattools as _compat
from .latex import vector as latex_vector
from .latex import matrix as latex_matrix
from .reportables import ReportableQty as _ReportableQty

def list(l, specs):
    """
    Convert a list to html.

    Parameters
    ----------
    l : list
        list to convert into HTML. sub-items pre formatted

    specs : dictionary
        Dictionary of user-specified and default parameters to formatting

    Returns
    -------
    string
        html string for l.
    """

    return "<br>".join(l)

def vector(v, specs):
    """
    Convert a 1D numpy array to html.

    Parameters
    ----------
    v : numpy array
        1D array to convert into HTML.

    specs : dictionary
        Dictionary of user-specified and default parameters to formatting

    Returns
    -------
    string
        html string for v.
    """
    return latex_vector(v, specs)

def matrix(m, specs):
    """
    Convert a 2D numpy array to html.

    Parameters
    ----------
    m : numpy array
        2D array to convert into HTML.

    specs : dictionary
        Dictionary of user-specified and default parameters to formatting

    Returns
    -------
    string
        html string for m.
    """
    return latex_matrix(m, specs)

def value(el, specs):
    """
    Convert a floating point or complex value to html.

    Parameters
    ----------
    el : float or complex
        Value to convert into HTML.

    specs : dictionary
        Dictionary of user-specified and default parameters to formatting

    Returns
    -------
    string
        html string for el.
    """

    # ROUND = digits to round values to
    TOL = 1e-9  #tolerance for printing zero values

    precision      = specs['precision']
    sciprecision   = specs['sciprecision']
    polarprecision = specs['polarprecision']
    complexAsPolar = specs['complexAsPolar']

    def render(x):
        if abs(x) < 5*10**(-(precision+1)):
            s = "%.*e" % (sciprecision,x)
        elif abs(x) < 1:
            s = "%.*f" % (precision, x)
        elif abs(x) <= 10**precision:
            s = "%.*f" % (precision-int(_np.log10(abs(x))),x)  #round to get precision+1 digits when x is > 1
        else:
            s = "%.*e" % (sciprecision,x)

        #Fix scientific notition
        p = s.split('e')
        if len(p) == 2:
            ex = str(int(p[1])) #exponent without extras (e.g. +04 => 4)
            s = p[0] + "&times;10<sup>" + ex + "</sup>"

        #Strip superfluous endings
        if "." in s:
            while s.endswith("0"): s = s[:-1]
            if s.endswith("."): s = s[:-1]
        return s


    if _compat.isstr(el):
        return el
    if type(el) in (int,_np.int64):
        return "%d" % el
    if el is None or _np.isnan(el): return "--"

    try:
        if abs(el.real) > TOL:
            if abs(el.imag) > TOL:
                if complexAsPolar:
                    r,phi = cmath.polar(el)
                    ex = ("i%.1f" % phi) if phi >= 0 else ("-i%.1f" % -phi)
                    s = "%se<sup>%s</sup>" % (render(r),ex)
                else:
                    s = "%s%s%si" % (render(el.real),'+' if el.imag > 0 else '-', render(abs(el.imag)))
            else:
                s = "%s" % render(el.real)
        else:
            if abs(el.imag) > TOL:
                s = "%si" % render(el.imag)
            else:
                s = "0"
    except:
        s = str(el)

    return s


def escaped(txt, specs):
    """
    Escape txt so it is html safe.

    Parameters
    ----------
    txt : string
        value to escape
        
    specs : dictionary
        Dictionary of user-specified and default parameters to formatting

    Returns
    -------
    string
    """
    return txt

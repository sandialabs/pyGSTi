from __future__ import division, print_function, absolute_import, unicode_literals
#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************
"""
Routines for converting python objects to Powerpoint compatible values.  Parallel rountines as
LatexUtil has for latex conversion.
"""
import numpy as _np
import cmath
from .. import objects as _objs

#Define basestring in python3 so unicode
# strings can be tested for in python2 using
# python2's built-in basestring type.
# When removing __future__ imports, remove
# this and change basestring => str below.
try:  basestring
except NameError: basestring = str


def ppt(x, brackets=False, precision=6, polarprecision=3, sciprecision=0):
    """
    Convert a numpy array, number, or string to ppt.

    Parameters
    ----------
    x : anything
        Value to convert into powerpoint-friendly value.

    brackets : bool, optional
        Whether to include brackets in the output for array-type variables.

    precision : int, optional
        Rounding for normal numbers

    polarprecision : int, optional
        Rounding for polar numbers

    sciprecision : int, optional
        Precision with which to el when precision
        requires use of scientific notation.


    Returns
    -------
    string
        ppt string for x.
    """
    if isinstance(x,_np.ndarray) or \
       isinstance(x,_objs.Gate) or \
       isinstance(x,_objs.SPAMVec):
        d = 0
        for l in x.shape:
            if l > 1: d += 1
        x = _np.squeeze(x)
        if d == 0: return ppt_value(x, precision=precision, polarprecision=polarprecision, sciprecision=sciprecision)
        if d == 1: return ppt_vector(x, brackets=brackets, precision=precision, polarprecision=polarprecision, sciprecision=sciprecision)
        if d == 2: return ppt_matrix(x, brackets=brackets, precision=precision, polarprecision=polarprecision, sciprecision=sciprecision)
        raise ValueError("I don't know how to render a rank %d numpy array as ppt" % d)
    elif type(x) in (float,int,complex,_np.float64,_np.int64):
        return ppt_value(x, precision=precision, polarprecision=polarprecision, sciprecision=sciprecision)
    elif type(x) in (list,tuple):
        return ppt_list(x, precision=precision, polarprecision=polarprecision, sciprecision=sciprecision)
    elif isinstance(x,basestring):
        return ppt_escaped(x)
    else:
        print("Warning: %s not specifically converted to ppt" % str(type(x)))
        return str(x)


def ppt_list(l, brackets=False, precision=6, polarprecision=3, sciprecision=0):
    """
    Convert a list to powerpoint.

    Parameters
    ----------
    l : list
        list to convert into powerpoint.

    brackets : bool, optional
        Whether to include brackets in the output powerpoint string.

    precision : int, optional
        Precision with which to round el.

    polarprecision : int, optional
        Precision with which to round polars.

    sciprecision : int, optional
        Precision with which to el when precision
        requires use of scientific notation.

    Returns
    -------
    string
        powerpoint string for l.
    """

    lines = [ ]
    for el in l:
        lines.append( ppt(el, brackets, precision=precision,
                          polarprecision=polarprecision,
                          sciprecision=sciprecision) )
    return ",".join(lines)


def ppt_vector(v, brackets=False, precision=6, polarprecision=3, sciprecision=0):
    """
    Convert a 1D numpy array to powerpoint.

    Parameters
    ----------
    v : numpy array
        1D array to convert into powerpoint.

    brackets : bool, optional
        Whether to include brackets in the output powerpoint.

    precision : int, optional
        Rounding for normal numbers

    polarprecision : int, optional
        Rounding for polar numbers

    sciprecision : int, optional
        Precision with which to el when precision
        requires use of scientific notation.


    Returns
    -------
    string
        powerpoint string for v.
    """

    lines = [ ]
    for el in v:
        lines.append( ppt_value(el, precision=precision,
                                polarprecision=polarprecision,
                                sciprecision=sciprecision) )
    #Unsupported: if brackets:
    return  "\n".join(lines)


def ppt_matrix(m, fontsize=None, brackets=False, precision=6, polarprecision=3, sciprecision=0):
    """
    Convert a 2D numpy array to powerpoint.

    Parameters
    ----------
    m : numpy array
        2D array to convert into POWERPOINT.

    fontsize : int, optional
        If not None, the fontsize.

    brackets : bool, optional
        Whether to include brackets in the output powerpoint.

    precision : int, optional
        Rounding for normal numbers

    polarprecision : int, optional
        Rounding for polar numbers

    sciprecision : int, optional
        Precision with which to el when precision
        requires use of scientific notation.


    Returns
    -------
    string
        powerpoint string for m.
    """
    lines = [ ]; prefix = ""
    if fontsize is not None:
        prefix += "" #unsupported currently

    for r in range(m.shape[0]):
        lines.append( "  ".join(
                [ppt_value(el, precision=precision,
                           polarprecision=polarprecision,
                           sciprecision=sciprecision) for el in m[r,:] ] ))

    # Unsupported: if brackets:
    return "\n".join(lines)


def ppt_value(el, precision=6, polarprecision=3,
              sciprecision=0, complexAsPolar=True):
    """
    Convert a floating point or complex value to powerpoint.

    Parameters
    ----------
    el : float or complex
        Value to convert into powerpoint.

    precision : int, optional
        rounding for normal numbers

    polarprecision : int, optional
        rounding for polar numbers

    sciprecision : int, optional
        Precision with which to el when precision
        requires use of scientific notation.

    complexAsPolar : bool, optional
        Whether to output complex values in polar form.  If False, usual
        a+ib form is used.


    Returns
    -------
    string
        powerpoint string for el.
    """

    # ROUND = digits to round values to
    TOL = 1e-9  #tolerance for printing zero values

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
            s = p[0] + "x10^" + ex

        #Strip superfluous endings
        if "." in s:
            while s.endswith("0"): s = s[:-1]
            if s.endswith("."): s = s[:-1]
        return s

    if isinstance(el,basestring):
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
                    s = "%se^{%s}" % (render(r),ex)
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
        #try:
        #    if abs(el) > TOL: #thows exception if el is not a number
        #        s = "%s" % render(el.real)
        #    else:
        #        s = "0"
        #except:
        s = str(el)

    return s


def ppt_escaped(txt):
    """
    Escape txt so it is powerpoint safe.

    Parameters
    ----------
    txt : string
        value to escape

    Returns
    -------
    string
    """
    return txt

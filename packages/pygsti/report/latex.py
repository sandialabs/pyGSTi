from __future__ import division, print_function, absolute_import, unicode_literals
#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************
"""
Routines for converting python objects to latex.  Parallel rountines as
HtmlUtil has for HTML conversion.
"""

import numpy as _np
import cmath
from .. import objects as _objs
from ..tools import compattools as _compat

def list(l, specs):
    """
    Convert a python list to latex tabular column.

    Parameters
    ----------
    l : list
        list to convert into latex. sub-items pre formatted

    specs : dictionary
        Dictionary of user-specified and default parameters to formatting

    Returns
    -------
    string
        latex string for l.
    """
    return "\\begin{tabular}{c}\n" + \
                " \\\\ \n".join(l) + "\n \end{tabular}\n"


def vector(v, specs):
    """
    Convert a 1D numpy array to latex.

    Parameters
    ----------
    v : numpy array
        1D array to convert into latex.

    specs : dictionary
        Dictionary of user-specified and default parameters to formatting

    Returns
    -------
    string
        latex string for v.
    """
    lines = [ ]
    for el in v:
        lines.append( value(el, specs) )
    if specs['brackets']:
        return "$ \\begin{pmatrix}\n" + \
                " \\\\ \n".join(lines) + "\n \end{pmatrix} $\n"
    else:
        return "$ \\begin{pmatrix}\n" + \
                " \\\\ \n".join(lines) + "\n \end{pmatrix} $\n"


def matrix(m, specs):
    """
    Convert a 2D numpy array to latex.

    Parameters
    ----------
    m : numpy array
        2D array to convert into latex.

    specs : dictionary
        Dictionary of user-specified and default parameters to formatting

    Returns
    -------
    string
        latex string for m.
    """
    lines    = [ ]
    prefix   = ""
    fontsize = specs['fontsize']

    if fontsize is not None:
        prefix += "\\fontsize{%f}{%f}\selectfont " % (fontsize, fontsize*1.2)

    for r in range(m.shape[0]):
        lines.append( " & ".join(
                [value(el, specs) for el in m[r,:] ] ) )

    if specs['brackets']:
        return prefix + "$ \\begin{pmatrix}\n"  + \
        " \\\\ \n".join(lines) + "\n \end{pmatrix} $\n"
    else:
        return prefix + "$ \\begin{pmatrix}\n"  + \
        " \\\\ \n".join(lines) + "\n \end{pmatrix} $\n"

def value(el, specs):
    """
    Convert a floating point or complex value to latex.

    Parameters
    ----------
    el : float or complex
        Value to convert into latex.

    specs : dictionary
        Dictionary of user-specified and default parameters to formatting

    Returns
    -------
    string
        latex string for el.
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
            s = "%.*f" % (precision,x)
        elif abs(x) <= 10**precision:
            s = "%.*f" % (precision-int(_np.log10(abs(x))),x)  #round to get precision+1 digits when x is > 1
        else:
            s = "%.*e" % (sciprecision,x)

        #Fix scientific notition
        p = s.split('e')
        if len(p) == 2:
            ex = str(int(p[1])) #exponent without extras (e.g. +04 => 4)
            s = p[0] + "\\times 10^{" + ex + "}"

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
                    ex = ("i%.*f" % (polarprecision, phi)) if phi >= 0 \
                        else ("-i%.*f" % (polarprecision, -phi))
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
        s = str(el)

    return s


def escaped(txt, specs):
    """
    Escape txt so it is latex safe.

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
    ret = txt.replace("_","\_")
    return ret

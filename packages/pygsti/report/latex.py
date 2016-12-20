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

#Define basestring in python3 so unicode
# strings can be tested for in python2 using
# python2's built-in basestring type.
# When removing __future__ imports, remove
# this and change basestring => str below.
try:  basestring
except NameError: basestring = str


def latex(x, brackets=False, precision=6,
          polarprecision=3, sciprecision=0):
    """
    Convert a numpy array, number, or string to latex.

    Parameters
    ----------
    x : anything
        Value to convert into latex.

    brackets : bool, optional
        Whether to include brackets in the output for array-type variables.

    precision : int, optional
        Rounding for normal numbers

    polarprecision : int, optional
        Rounding for polars

    sciprecision : int, optional
        Precision with which to el when precision
        requires use of scientific notation.


    Returns
    -------
    string
        latex string for x.
    """
    if isinstance(x,_np.ndarray) or \
       isinstance(x,_objs.Gate) or \
       isinstance(x,_objs.SPAMVec):
        d = 0
        for l in x.shape:
            if l > 1: d += 1
        x = _np.squeeze(x)
        if d == 0: return latex_value(x, precision=precision, polarprecision=polarprecision, sciprecision=sciprecision)
        if d == 1: return latex_vector(x, brackets=brackets, precision=precision, polarprecision=polarprecision, sciprecision=sciprecision)
        if d == 2: return latex_matrix(x, brackets=brackets, precision=precision, polarprecision=polarprecision, sciprecision=sciprecision)
        raise ValueError("I don't know how to render a rank %d numpy array as latex" % d)
    elif type(x) in (list,tuple):
        return latex_list(x, precision=precision, polarprecision=polarprecision, sciprecision=sciprecision)
    elif type(x) in (float,int,complex,_np.float64,_np.int64):
        return latex_value(x, precision=precision, polarprecision=polarprecision, sciprecision=sciprecision)
    elif isinstance(x,basestring):
        return latex_escaped(x)
    else:
        print("Warning: %s not specifically converted to latex" % str(type(x)))
        return str(x)


def latex_list(l, brackets=False, precision=6, polarprecision=3, sciprecision=0):
    """
    Convert a python list to latex tabular column.

    Parameters
    ----------
    l : list
        list to convert into latex.

    brackets : bool, optional
        Whether to include brackets in the output for array-type variables.

    precision : int, optional
        Rounding for normal numbers

    polarprecision : int, optional
        Rounding for polars

    sciprecision : int, optional
        Precision with which to el when precision
        requires use of scientific notation.


    Returns
    -------
    string
        latex string for l.
    """
    lines = [ ]
    for el in l:
        lines.append( latex(el, brackets, precision, polarprecision, sciprecision) )
    return "\\begin{tabular}{c}\n" + \
                " \\\\ \n".join(lines) + "\n \end{tabular}\n"


def latex_vector(v, brackets=False, precision=6,
                 polarprecision=3, sciprecision=0):
    """
    Convert a 1D numpy array to latex.

    Parameters
    ----------
    v : numpy array
        1D array to convert into latex.

    brackets : bool, optional
        Whether to include brackets in the output latex.

    precision : int, optional
        Rounding for normal numbers

    polarprecision : int, optional
        Rounding for polars

    sciprecision : int, optional
        Precision with which to el when precision
        requires use of scientific notation.


    Returns
    -------
    string
        latex string for v.
    """
    lines = [ ]
    for el in v:
        lines.append( latex_value(el, precision=precision,
                                  polarprecision=polarprecision,
                                  sciprecision=sciprecision) )
    if brackets:
        return "$ \\left(\\!\\!\\begin{array}{c}\n" + \
                " \\\\ \n".join(lines) + "\n \end{array}\\!\\!\\right) $\n"
    else:
        return "$ \\begin{array}{c}\n" + \
                " \\\\ \n".join(lines) + "\n \end{array} $\n"


def latex_matrix(m, fontsize=None, brackets=False, precision=6,
                 polarprecision=3, sciprecision=0):
    """
    Convert a 2D numpy array to latex.

    Parameters
    ----------
    m : numpy array
        2D array to convert into latex.

    fontsize : int, optional
        If not None, the fontsize.

    brackets : bool, optional
        Whether to include brackets in the output latex.

    precision : int, optional
        Rounding for normal numbers

    polarprecision : int, optional
        Rounding for polars

    sciprecision : int, optional
        Precision with which to el when precision
        requires use of scientific notation.


    Returns
    -------
    string
        latex string for m.
    """
    lines = [ ]; prefix = ""
    if fontsize is not None:
        prefix += "\\fontsize{%f}{%f}\selectfont " % (fontsize, fontsize*1.2)

    for r in range(m.shape[0]):
        lines.append( " & ".join(
                [latex_value(el, precision=precision,
                             polarprecision=polarprecision, 
                             sciprecision=sciprecision) for el in m[r,:] ] ) )

    if brackets:
        return prefix + "$ \\left(\\!\\!\\begin{array}{%s}\n" % ("c" * m.shape[1]) + \
        " \\\\ \n".join(lines) + "\n \end{array}\\!\\!\\right) $\n"
    else:
        return prefix + "$ \\begin{array}{%s}\n" % ("c" * m.shape[1]) + \
        " \\\\ \n".join(lines) + "\n \end{array} $\n"



def latex_value(el, precision=6, polarprecision=3,
                sciprecision=0, complexAsPolar=True):
    """
    Convert a floating point or complex value to latex.

    Parameters
    ----------
    el : float or complex
        Value to convert into latex.

    precision : int, optional
        Rounding for normal numbers

    polarprecision : int, optional
        Rounding for polars

    sciprecision : int, optional
        Precision with which to el when precision
        requires use of scientific notation.

    complexAsPolar : bool, optional
        Whether to output complex values in polar form.  If False, usual
        a+ib form is used.


    Returns
    -------
    string
        latex string for el.
    """
    # ROUND = digits to round values to
    TOL = 1e-9  #tolerance for printing zero values

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
            s = p[0] + "\\e{" + ex + "}"

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
        #try:
        #    if abs(el) > TOL: #throws error if el is not a number
        #        s = "%s" % render(el.real)
        #    else:
        #        s = "0"
        #except:
        s = str(el)

    return s


def latex_escaped(txt):
    """
    Escape txt so it is latex safe.

    Parameters
    ----------
    txt : string
        value to escape

    Returns
    -------
    string
    """
    ret = txt.replace("_","\_")
    return ret

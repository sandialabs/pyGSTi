"""
Routines for converting python objects to latex.

Parallel rountines as html.py has for HTML conversion.
"""
#***************************************************************************************************
# Copyright 2015, 2019, 2025 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

import cmath

import numpy as _np

'''
table() and cell() functions are used by table.py in table creation
everything else is used in creating formatters in formatters.py
'''


def table(custom_headings, col_headings_formatted, rows, spec):
    """
    Create a LaTeX table

    Parameters
    ----------
    custom_headings : None, dict
        optional dictionary of custom table headings

    col_headings_formatted : list
        formatted column headings

    rows : list of lists of cell-strings
        Data in the table, pre-formatted

    spec : dict
        options for the formatter

    Returns
    -------
    dict : contains key 'latex', which corresponds to a latex string representing the table
    """
    longtables = spec['longtables']
    table = "longtable" if longtables else "tabular"
    if custom_headings is not None \
            and "latex" in custom_headings:
        latex = custom_headings['latex']
    else:
        latex = "\\begin{%s}[l]{%s}\n\\hline\n" % \
            (table, "|c" * len(col_headings_formatted) + "|")
        latex += ("%s \\\\ \\hline\n"
                  % (" & ".join(col_headings_formatted)))

    for formatted_rowData in rows:

        if len(formatted_rowData) > 0:
            formatted_rowData_latex = [
                (formatted_cell['latex'] if isinstance(formatted_cell, dict)
                 else formatted_cell) for formatted_cell in formatted_rowData]
            latex += " & ".join(formatted_rowData_latex)

            #MULTI-ROW support for *data* (non-col-header) rows of table.  Currently
            # unused (unneeded) - see multirow formatter that is commented out in formatters.py
            #multirows = [ ("multirow" in el) for el in formatted_rowData_latex ]
            #if any(multirows):
            #    latex += " \\\\ "
            #    last = True; lineStart = None; col = 1
            #    for multi,data in zip(multirows,formatted_rowData_latex):
            #        if last == True and multi == False:
            #            lineStart = col #line start
            #        elif last == False and multi == True:
            #            latex += "\cline{%d-%d} " % (lineStart,col) #line end
            #        last=multi
            #        res = _re.search("multicolumn{([0-9])}",data)
            #        if res: col += int(res.group(1))
            #        else:   col += 1
            #    if last == False: #need to end last line
            #        latex += "\cline{%d-%d} "%(lineStart,col-1)
            #    latex += "\n"
            #else:
            latex += " \\\\ \\hline\n"

    latex += "\\end{%s}\n" % table
    return {'latex': latex}


def cell(data, label, spec):
    """
    Format the cell of a latex table

    Parameters
    ----------
    data : string
        string representation of cell content

    label : string
        optional cell label, used for tooltips

    spec : dict
        options for the formatters

    Returns
    -------
    string
    """
    addMathMode = bool(("^" in data or "_" in data) and "$" not in data)
    return "$" + data + "$" if addMathMode else data


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
        " \\\\ \n".join(l) + "\n \\end{tabular}\n"


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
    lines = []
    for el in v:
        lines.append(value(el, specs, mathmode=True))
    if specs['brackets']:
        return "\\ensuremath{ \\begin{pmatrix}\n" + \
            " \\\\ \n".join(lines) + "\n \\end{pmatrix} }\n"
    else:
        return "\\ensuremath{ \\begin{pmatrix}\n" + \
            " \\\\ \n".join(lines) + "\n \\end{pmatrix} }\n"


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
    lines = []
    prefix = ""
    fontsize = specs['fontsize']

    if fontsize is not None:
        prefix += "\\fontsize{%f}{%f}\\selectfont " % (fontsize, fontsize * 1.2)

    for r in range(m.shape[0]):
        lines.append(" & ".join(
            [value(el, specs, mathmode=True) for el in m[r, :]]))

    if specs['brackets']:
        return prefix + "\\ensuremath{ \\begin{pmatrix}\n" + \
            " \\\\ \n".join(lines) + "\n \\end{pmatrix} }\n"
    else:
        return prefix + "\\ensuremath{ \\begin{pmatrix}\n" + \
            " \\\\ \n".join(lines) + "\n \\end{pmatrix} }\n"


def value(el, specs, mathmode=False):
    """
    Convert a floating point or complex value to latex.

    Parameters
    ----------
    el : float or complex
        Value to convert into latex.

    specs : dictionary
        Dictionary of user-specified and default parameters to formatting

    mathmode : bool, optional
        Whether this routine should assume that math-mode is already enabled and
        output is free to contain math-mode instructions.  When False, whenever
        math-mode instructions are needed the output is wrapped in an
        'ensuremath' block.

    Returns
    -------
    string
        latex string for el.
    """
    # ROUND = digits to round values to
    TOL = 1e-9  # tolerance for printing zero values

    precision = specs['precision']
    sciprecision = specs['sciprecision']
    polarprecision = specs['polarprecision']
    complexAsPolar = specs['complex_as_polar']

    def render(x):
        """Render a single float (can be real or imag part)"""
        if abs(x) < 5 * 10**(-(precision + 1)):
            s = "%.*e" % (sciprecision, x)
        elif abs(x) < 1:
            s = "%.*f" % (precision, x)
        elif abs(x) <= 10**precision:
            s = "%.*f" % (precision - int(_np.log10(abs(x))), x)  # round to get precision+1 digits when x is > 1
        else:
            s = "%.*e" % (sciprecision, x)

        #Fix scientific notition
        p = s.split('e')
        if len(p) == 2:
            ex = str(int(p[1]))  # exponent without extras (e.g. +04 => 4)
            if mathmode:
                s = p[0] + "\\times 10^{" + ex + "}"
            else:  # ensure math mode so \times and ^ will work
                s = "\\ensuremath{ " + p[0] + "\\times 10^{" + ex + "} }"

        #Strip superfluous endings
        if "." in s:
            while s.endswith("0"): s = s[:-1]
            if s.endswith("."): s = s[:-1]
        return s

    if isinstance(el, str):
        return el
    if isinstance(el, (int, _np.int64)):
        return "%d" % el
    try:
        if el is None or _np.isnan(el): return "--"
    except TypeError: pass  # if isnan fails b/c a non-numeric type, just proceed

    try:
        if abs(el.real) > TOL:
            if abs(el.imag) > TOL:
                if complexAsPolar:
                    r, phi = cmath.polar(el)
                    ex = ("i%.*f" % (polarprecision, phi / _np.pi)) if phi >= 0 \
                        else ("-i%.*f" % (polarprecision, -phi / _np.pi))
                    if mathmode:
                        s = "%se^{%s\\pi}" % (render(r), ex)
                    else:  # ensure math mode so ^ will work
                        s = "\\ensuremath{ %se^{%s\\pi} }" % (render(r), ex)

                else:
                    s = "%s%s%si" % (render(el.real), '+' if el.imag > 0 else '-', render(abs(el.imag)))
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
    ret = txt.replace("_", r"\_")
    return ret

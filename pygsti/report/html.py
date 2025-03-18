"""
Routines for converting python objects to HTML.

Parallel rountines as latex.py has for latex conversion.
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
    Create an HTML table

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
    dict : contains keys 'html' and 'js', which correspond to a html and js strings representing the table
    """
    tableclass = spec['tableclass']
    tableID = spec['table_id']
    html = ""
    js = ""

    if custom_headings is not None \
            and "html" in custom_headings:
        html += custom_headings['html'] % {'tableclass': tableclass,
                                           'tableid': tableID}
    else:
        html += "<table"
        if tableclass: html += ' class="%s"' % tableclass
        if tableID: html += ' id="%s"' % tableID
        html += "><thead><tr><th> %s </th></tr>" % \
            (" </th><th> ".join(col_headings_formatted))
        html += "</thead><tbody>"
    for formatted_rowData in rows:
        if len(formatted_rowData) > 0:
            html += "<tr>"
            for formatted_cell in formatted_rowData:
                if isinstance(formatted_cell, dict):
                    #cell contains javascript along with html
                    js += formatted_cell['js'] + '\n'
                    formatted_cell = formatted_cell['html']

                if formatted_cell is None:
                    pass  # don't add anything -- not even td tags (this
                    # allows signals *not* to include a cell)
                elif formatted_cell.startswith("<td"):
                    html += formatted_cell  # assume format includes td tags
                else: html += "<td>" + str(formatted_cell) + "</td>"
            html += "</tr>"

    html += "</tbody></table>"

    return {'html': html, 'js': js}


def cell(data, label, spec):
    """
    Format the cell of an HTML table

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
    if isinstance(data, dict) or label is None:
        return data
    return '<span title="{}">{}</span>'.format(label, data)


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
    lines = []
    for el in v:
        lines.append(value(el, specs, mathmode=True))
    if specs['brackets']:
        return "$ \\begin{pmatrix}\n" + \
            " \\\\ \n".join(lines) + "\n \\end{pmatrix} $\n"
    else:
        return "$ \\begin{pmatrix}\n" + \
            " \\\\ \n".join(lines) + "\n \\end{pmatrix} $\n"


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
    lines = []
    prefix = ""
    fontsize = specs['fontsize']

    if fontsize is not None:
        prefix += "\\fontsize{%f}{%f}\\selectfont " % (fontsize, fontsize * 1.2)

    for r in range(m.shape[0]):
        lines.append(" & ".join(
            [value(el, specs, mathmode=True) for el in m[r, :]]))

    if specs['brackets']:
        return prefix + "$ \\begin{pmatrix}\n" + \
            " \\\\ \n".join(lines) + "\n \\end{pmatrix} $\n"
    else:
        return prefix + "$ \\begin{pmatrix}\n" + \
            " \\\\ \n".join(lines) + "\n \\end{pmatrix} $\n"


def value(el, specs, mathmode=False):
    """
    Convert a floating point or complex value to html.

    Parameters
    ----------
    el : float or complex
        Value to convert into HTML.

    specs : dictionary
        Dictionary of user-specified and default parameters to formatting

    mathmode : bool, optional
        Whether this routine should generate HTML for use within a math-rendered
        HTML element (rather than a normal plain-HTML one).  So when set to True
        output is essentially the same as latex format.

    Returns
    -------
    string
        html string for el.
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
            if mathmode:  # don't use <sup> in math mode
                s = p[0] + "\\times 10^{" + ex + "}"
            else:
                s = p[0] + "&times;10<sup>" + ex + "</sup>"

        #Strip superfluous endings
        if "." in s:
            while s.endswith("0"): s = s[:-1]
            if s.endswith("."): s = s[:-1]
        return s

    if isinstance(el, str):
        return el
    if type(el) in (int, _np.int64):
        return "%d" % el
    if el is None or _np.isnan(el): return "--"

    try:
        if abs(el.real) > TOL:
            if abs(el.imag) > TOL:
                if complexAsPolar:
                    r, phi = cmath.polar(el)
                    ex = ("i%.*f" % (polarprecision, phi / _np.pi)) if phi >= 0 \
                        else ("-i%.*f" % (polarprecision, -phi / _np.pi))
                    if mathmode:  # don't use <sup> in math mode
                        s = "%se^{%s\\pi}" % (render(r), ex)
                    else:
                        s = "%se<sup>%s &pi;</sup>" % (render(r), ex)
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

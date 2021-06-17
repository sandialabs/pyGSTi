"""
Functions for generating report tables in different formats
"""

#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************


from pygsti.report.convert import converter
html = converter('html')  # Retrieve low-level formatters
latex = converter('latex')

from pygsti.report.formatter import Formatter as _Formatter
from pygsti.report.reportableqty import ReportableQty as _ReportableQty

import numpy as _np
import numbers as _numbers

##############################################################################
#                          Formatting functions                              #
##############################################################################

'''
For documentation on creating additional formatters, see formatter.py

If the Formatter class does not offer enough functionality,
  any function with the signature (item, specs -> string) can be used as a formatter
'''

#An example of a formatting function
# 'specs' is a dictionary of rendering options


def _no_format(x, specs):
    return x


# This dictionary is intentionally exported to other modules.
# Even though it can change at runtime, it never does and should not
format_dict = dict()

# 'rho' (state prep) formatting
# Replace rho with &rho;
# Numbers following 'rho' -> subscripts
format_dict['Rho'] = {
    'html': _Formatter(regexreplace=('rho([0-9]+)$', '&rho;<sub>%s</sub>')),
    'latex': _Formatter(regexreplace=('rho([0-9]+)$', '$\\rho_{%s}$')),
    'python': _no_format}

# 'E' (POVM) effect formatting
format_dict['Effect'] = {
    # If label == 'Ec', return E sub C
    # Otherwise, match regex and replace with subscript
    'html': _Formatter(stringreturn=('Ec', 'E<sub>C</sub>'),
                       regexreplace=('E([0-9]+)$', 'E<sub>%s</sub>')),
    'latex': _Formatter(stringreturn=('Ec', '$E_C$'),
                        regexreplace=('E([0-9]+)$', '$E_{%s}$')),
    'python': _no_format}

NormalHTML = _Formatter(html,
                        ebstring='%s <span class="errorbar">&plusmn; %s</span>',
                        nmebstring='%s <span class="nmerrorbar">&plusmn; %s</span>')
NormalLatex = _Formatter(latex,
                         ebstring='$ \\begin{array}{c} %s \\\\ \pm %s \\end{array} $')  # nmebstring will match

# Normal replacements
format_dict['Normal'] = {
    'html': NormalHTML,
    'latex': NormalLatex,  # nmebstring will match
    'python': _no_format}

# 'normal' formatting but round to 2 decimal places regardless of what is passed in to table.render()
format_dict['Rounded'] = {
    'html': NormalHTML.variant(defaults={'precision': 2, 'sciprecision': 0}),
    'latex': NormalLatex.variant(defaults={'precision': 2, 'sciprecision': 0}),
    'python': _no_format}

# 'small' formating - make text smaller
format_dict['Small'] = {
    'html': NormalHTML,
    'latex': NormalLatex.variant(formatstring='\\small%s'),
    'python': _no_format}

# 'small' formating - make text smaller
format_dict['Verbatim'] = {
    'html': NormalHTML,
    'latex': NormalLatex.variant(formatstring='\\spverb!%s!'),
    'python': _no_format}


def _pi_python(x, specs):
    if isinstance(x, _numbers.Number):
        return x * _np.pi
    else:
        return x


PiPython = _Formatter(_pi_python)

# Pi formatters
format_dict['Pi'] = {
    'html': NormalHTML.variant(formatstring='%s&pi;',
                               ebstring='%s&pi; <span class="errorbar">&plusmn; %s</span>&pi;',
                               nmebstring='%s&pi; <span class="nmerrorbar">&plusmn; %s</span>&pi;'),
    'latex': NormalLatex.variant(formatstring='%s$\\pi$',
                                 ebstring='$ \\begin{array}{c}(%s \\\\ \\pm %s)\\pi \\end{array} $'),
    'python': PiPython}

# BracketFormatters
format_dict['Brackets'] = {
    'html': NormalHTML.variant(defaults={'brackets': True}),
    'latex': NormalLatex.variant(defaults={'brackets': True}),
    'python': _no_format}

##################################################################################
# 'conversion' formatting: catch all for find/replacing specially formatted text #
##################################################################################

convert_html = NormalHTML.variant(stringreplacers=[
    ('\\', '&#92'),
    ('|', ' '),
    ('<STAR>', '&#9733;')])

pre_convert_latex = _Formatter(stringreplacers=[
    ("\\", "\\textbackslash"),
    ('%', '\\%'),
    ('#', '\\#'),
    ("half-width", "$\\nicefrac{1}{2}$-width"),
    ("1/2", "$\\nicefrac{1}{2}$"),
    ("Diamond", "$\\Diamond$"),
    ("Check", "\\checkmark"),
    ('|', '\\\\'),
    ('<STAR>', '\\bigstar')])


def special_convert_latex(x, specs):
    """
    Special conversion rules for latex

    Parameters
    ----------
    x : object
        Object to convert

    specs : dictionary
        dictionary of formatting options.

    Returns
    -------
    str
    """
    x = pre_convert_latex(str(x), specs)
    if '\\bigstar' in x:
        x = '${}$'.format(x)
    if "\\\\" in x:
        return '\\begin{tabular}{c}' + x + '\\end{tabular}'
    else:
        return x


convert_latex = NormalLatex.variant(custom=special_convert_latex)

mathtext_htmlorlatex = _Formatter(
    stringreplacers=[('log', '\\log'),
                     ('Re', '\\mathrm{Re}'),
                     ('Im', '\\mathrm{Im}'),
                     ('*', '\\cdot ')],
    formatstring='$%s$')

format_dict['Conversion'] = {
    'html': convert_html,
    'latex': convert_latex,
    'python': _no_format}

format_dict['MathText'] = {
    'html': mathtext_htmlorlatex,
    'latex': mathtext_htmlorlatex,
    'python': _no_format}


format_dict['Vec'] = {
    'html': NormalHTML,
    'latex': _Formatter(latex, ebstring='%s $\pm$ %s'),
    'python': _no_format}

format_dict['Circuit'] = {
    'html': _Formatter(lambda s, specs: '.'.join(map(str, s)) if s is not None else ''),
    'latex': _Formatter(lambda s, specs: '' if s is None else ('$%s$' % '\\cdot'.join([('\\mbox{%s}' % str(gl))
                                                                                       for gl in s]))),
    'python': _no_format}

'''
Figure formatters no longer use Formatter objects, because figure formatters are more specialized.
Notice that they still have the function signature (item, specs -> string)
'''


def html_figure(fig, specs):
    """
    Render a html-format figure

    Parameters
    ----------
    fig : ReportableQty
        A reportable quantity holding a `WorkspacePlot` as its value.

    specs : dictionary
        dictionary of formatting options.

    Returns
    -------
    str
    """
    #Create figure inline with 'js' set to only handlers (no further plot init)
    fig.value.set_render_options(switched_item_mode="inline",
                                 resizable="handlers only",
                                 click_to_display=specs['click_to_display'],
                                 link_to=specs['link_to'],
                                 autosize=specs['autosize'],
                                 output_dir=specs['output_dir'])
    render_out = fig.value.render("html")
    return render_out  # a dictionary with 'html' and 'js' keys


def latex_figure(fig, specs):
    """
    Render a latex-format figure

    Parameters
    ----------
    fig : ReportableQty
        A reportable quantity holding a `WorkspacePlot` as its value.

    specs : dictionary
        dictionary of formatting options.

    Returns
    -------
    str
    """
    assert('output_dir' in specs and specs['output_dir']), \
        "Cannot render a figure-containing table as 'latex' without a valid 'output_dir' render option"
    fig.value.set_render_options(output_dir=specs['output_dir'],
                                 render_includes=specs['render_includes'])
    render_out = fig.value.render('latex')
    render_out['latex'] = "\\vcenteredhbox{%s}" % render_out['latex']  # wrap std latex output
    return render_out


def python_figure(fig, specs):
    """
    Render a python-format figure

    Parameters
    ----------
    fig : ReportableQty
        A reportable quantity holding a `WorkspacePlot` as its value.

    specs : dictionary
        dictionary of formatting options.

    Returns
    -------
    ReportableQty
    """
    fig.value.set_render_options(switched_item_mode="inline")
    render_out = fig.value.render('python')  # a dict w/keys == plotIDs
    plotDivID = list(render_out['python'].keys())[0]  # just take info for the first figure (assume only one figure)

    if specs['output_dir'] is not None:  # setting output_dir signals that fig should also be rendered
        fig.value.set_render_options(switched_item_mode="separate files",
                                     output_dir=specs['output_dir'])
        fig.value.render('python')  # to a separate python file

    return _ReportableQty(render_out['python'][plotDivID]['value'],
                          render_out['python'][plotDivID].get('errorbar', None))


format_dict['Figure'] = {
    'html': html_figure,
    'latex': latex_figure,
    'python': python_figure}

# Bold formatting
format_dict['Bold'] = {
    'html': _Formatter(html, formatstring='<b>%s</b>'),
    'latex': _Formatter(latex, formatstring='\\textbf{%s}'),
    'python': _no_format}

#Special formatting for Hamiltonian and Stochastic model types
format_dict['GatesetType'] = {
    'html': _Formatter(),
    'latex': _Formatter(stringreplacers=[('H', '$\\mathcal{H}$'), ('S', '$\\mathcal{S}$')]),
    'python': _no_format}


'''
# 'pre' formatting, where the user gives the data in separate formats
def _pre_fmt_template(formatname):
    return lambda label : label[formatname]

format_dict['Pre'] = {
    'html'   : _pre_fmt_template('html'),
    'latex'  : _pre_fmt_template('latex')}

#Multi-row and multi-column formatting (with "Conversion" type inner formatting)
format_dict['MultiRow'] = {
    'html'  : TupleFormatter(convert_html, formatstring='<td rowspan="{l1}">{l0}</td>'),
    'latex' : TupleFormatter(convert_latex, formatstring='\\multirow{{{l1}}}{{*}}{{{l0}}}')}

def _empty_str(l): return ""
def _return_None(l): return None #signals no <td></td> in HTML

format_dict['SpannedRow'] = {
    'html'  : _return_None,
    'latex' : _empty_str}

def _repeatno_format(label_tuple):
    label, reps = label_tuple
    return ["%s" % label]*reps

format_dict['MultiCol'] = {
    'html'  : TupleFormatter(convert_html, formatstring='<td colspan="{l1}">{l0}</td>'),
    'latex' : TupleFormatter(convert_latex, formatstring='\\multicolumn{{{l1}}}{{c|}}{{{l0}}}')}
'''

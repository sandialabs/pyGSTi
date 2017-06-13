from __future__ import division, print_function, absolute_import, unicode_literals

#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************

""" Functions for generating report tables in different formats """

from .convert import converter
html  = converter('html')  # Retrieve low-level formatters
latex = converter('latex')

from ..objects.formatter import Formatter

import cgi     as _cgi
import numpy   as _np
import numbers as _numbers
import re      as _re
import os      as _os

from plotly.offline import plot as _plot

from .formatset import FormatSet

def no_format(label):
    return label

##############################################################################
#                          Formatting functions                              #
##############################################################################

# 'rho' (state prep) formatting
# Replace rho with &rho;
# Numbers following 'rho' -> subscripts
FormatSet.formatDict['Rho'] = {
    'html'  : Formatter(stringreplacers=[('rho', '&rho;')],
                         regexreplace=('.*?([0-9]+)$', '<sub>%s</sub>')),
    'latex' : Formatter(stringreplacers=[('rho', '\\rho')],
                         regexreplace=('.*?([0-9]+)$', '_{%s}'), formatstring='${}$')}

# 'E' (POVM) effect formatting
FormatSet.formatDict['Effect'] = {
    # If label == 'remainder', return E sub C
    # Otherwise, match regex and replace with subscript
    'html'  : Formatter(stringreturn=('remainder', 'E<sub>C</sub>'),
                         regexreplace=('.*?([0-9]+)$', '<sub>%s</sub>')),
    'latex' : Formatter(stringreturn=('remainder', '$E_C$'),
                         regexreplace=('.*?([0-9]+)$', '_{%s}'), formatstring='$%s$')}

# Normal replacements
FormatSet.formatDict['Normal'] = {
    'html'  : Formatter(html),
    'latex' : Formatter(latex)}

# 'normal' formatting but round to 2 decimal places regardless of what is passed in to table.render()
FormatSet.formatDict['Rounded'] = {
    'html'  : Formatter(html,  defaults={'precision' : 2, 'sciprecision': 0}),
    'latex' : Formatter(latex, defaults={'precision' : 2, 'sciprecision': 0})}

# Similar to the above two formatdicts,
# but recieves precision during table.render(), which is sent as kwarg to html, for example
FormatSet.formatDict['Precision'] = {
    'html'  : Formatter(html),
    'latex' : Formatter(latex)}

# 'small' formating - make text smaller
FormatSet.formatDict['Small'] = {
    'html'  : Formatter(html),
    'latex' : Formatter(latex, formatstring='\\small%s')}

# 'small' formating - make text smaller
FormatSet.formatDict['Verbatim'] = {
    'html'  : Formatter(html),
    'latex' : Formatter(formatstring='\\spverb!%s!')}

#############################################
# Helper functions for formatting pi-labels #
#############################################

def _pi_template(b):
    # Pi Formatting shares a common predicate and first branch condition
    def formatter(label, specs):
        if str(label) == '--' or str(label) == '':
            return str(label)
        else:
            return b(label, specs)
    return formatter

# Pi formatters
FormatSet.formatDict['Pi'] = {
    'html'  : _pi_template(Formatter(html,  formatstring='%s&pi;')),
    'latex' : _pi_template(Formatter(latex, formatstring='%s$\\pi$'))}

# BracketFormatters
FormatSet.formatDict['Brackets'] = {
    'html'  : Formatter(html,  defaults={'brackets' : True}),
    'latex' : Formatter(latex, defaults={'brackets' : True})}

##################################################################################
# 'conversion' formatting: catch all for find/replacing specially formatted text #
##################################################################################

convert_html = Formatter(stringreplacers=[
    ('\\', '&#92'),
    ('|', ' '),
    ('<STAR>', '&#9733;')])

pre_convert_latex = Formatter(stringreplacers=[
    ("\\", "\\textbackslash"),
    ('%','\\%'),
    ('#','\\#'),
    ("half-width", "$\\nicefrac{1}{2}$-width"),
    ("1/2", "$\\nicefrac{1}{2}$"),
    ("Diamond","$\\Diamond$"),
    ("Check","\\checkmark"),
    ('|', '\\\\'),
    ('<STAR>', '\\bigstar')])

def special_convert_latex(x, specs):
    x = pre_convert_latex(str(x), {})
    if '\\bigstar' in x:
        x = '${}$'.format(x)
    if "\\\\" in x:
        return '\\begin{tabular}{c}' + x + '\\end{tabular}'
    else:
        return x

convert_latex = Formatter(special_convert_latex)

FormatSet.formatDict['Conversion'] = {
    'html'  : Formatter(convert_html),
    'latex' : Formatter(convert_latex)}

FormatSet.formatDict['EBConversion'] = {
    'html'  : Formatter(convert_html, formatstring='<span class="errorbar">{}</span>'),
    'latex' : Formatter(convert_latex)}

FormatSet.formatDict['NMEBConversion'] = {
    'html'  : Formatter(convert_html, formatstring='<span class="nmerrorbar">{}</span>'),
    'latex' : Formatter(convert_latex)}

EB_html   = Formatter(html, ebstring='%s <span class="errorbar">+/- %s</span>')
NMEB_html = Formatter(html, ebstring='%s <span class="nmerrorbar">+/- %s</span>')
EB_latex  = Formatter(latex, ebstring='$ \\begin{array}{c} %s \\\\ \pm %s \\end{array} $')

FormatSet.formatDict['ErrorBars'] = {
    'html'  : EB_html,
    'latex' : EB_latex}
FormatSet.formatDict['NMErrorBars'] = {
    'html'  : NMEB_html,
    'latex' : EB_latex}

VEB_latex = Formatter(latex, ebstring='%s $\pm$ %s')

FormatSet.formatDict['VecErrorBars'] = {
    'html'  : EB_html,
    'latex' : VEB_latex}
FormatSet.formatDict['NMVecErrorBars'] = {
    'html'  : NMEB_html,
    'latex' : VEB_latex}

PiEB_latex = Formatter(latex, ebstring='$ \\begin{array}{c}(%s \\\\ \\pm %s)\\pi \\end{array} $', formatstring='%s$\\pi$')

# 'errorbars with pi' formatting: display (scalar_value +/- error bar) * pi
FormatSet.formatDict['PiErrorBars'] = {
    'html'  : Formatter(html, ebstring='(%s <span class="errorbar">+/- %s</span>)&pi'),
    'latex' : PiEB_latex}

FormatSet.formatDict['NMPiErrorBars'] = {
    'html'  : Formatter(html, ebstring='(%s <span class="nmerrorbar">+/- %s</span>)&pi'),
    'latex' : PiEB_latex}

FormatSet.formatDict['GateString'] = {
    'html'  : Formatter(lambda s : '.'.join(s) if s is not None else ''),
    'latex' : Formatter(lambda s : ''          if s is None else ('$%s$' % '\\cdot'.join([ ('\\mbox{%s}' % gl) for gl in s])))}

def html_figure(fig, specs):
    render_out = fig.render("html",
                            resizable="handlers only" if specs['resizable'] else False,
                            autosize=specs['autosize'])
    return render_out #a dictionary with 'html' and 'js' keys

def latex_figure(figInfo, specs):
    extension    = '.pdf' 
    formatstring = "\\vcenteredhbox{\\includegraphics[width=%.2fin,height=%.2fin,keepaspectratio]{%s/%s}}"
    fig, name, W, H = figInfo
    scratchDir = specs['scratchDir']
    if len(scratchDir) > 0: #empty scratchDir signals not to output figure
        fig.save_to(_os.path.join(scratchDir, name + self.extension))
    return formatstring % (W, H, scratchDir,
                           name + self.extension)

FormatSet.formatDict['Figure'] = {
    'html'  : Formatter(html_figure),
    'latex' : Formatter(latex_figure)}

# Bold formatting
FormatSet.formatDict['Bold'] = {
    'html'  : Formatter(html, formatstring='<b>%s</b>'),
    'latex' : Formatter(latex, formatstring='\\textbf{%s}')}

#Special formatting for Hamiltonian and Stochastic gateset types
FormatSet.formatDict['GatesetType'] = {
    'html'  : no_format,
    'latex' : Formatter(stringreplacers=[('H','$\\mathcal{H}$'),('S','$\\mathcal{S}$')])}

'''
# 'pre' formatting, where the user gives the data in separate formats
def _pre_fmt_template(formatname):
    return lambda label : label[formatname]

FormatSet.formatDict['Pre'] = {
    'html'   : _pre_fmt_template('html'),
    'latex'  : _pre_fmt_template('latex')}

#Multi-row and multi-column formatting (with "Conversion" type inner formatting)
FormatSet.formatDict['MultiRow'] = {
    'html'  : TupleFormatter(convert_html, formatstring='<td rowspan="{l1}">{l0}</td>'),
    'latex' : TupleFormatter(convert_latex, formatstring='\\multirow{{{l1}}}{{*}}{{{l0}}}')}

def _empty_str(l): return ""
def _return_None(l): return None #signals no <td></td> in HTML

FormatSet.formatDict['SpannedRow'] = {
    'html'  : _return_None,
    'latex' : _empty_str}

def _repeatno_format(label_tuple): 
    label, reps = label_tuple
    return ["%s" % label]*reps

FormatSet.formatDict['MultiCol'] = {
    'html'  : TupleFormatter(convert_html, formatstring='<td colspan="{l1}">{l0}</td>'),
    'latex' : TupleFormatter(convert_latex, formatstring='\\multicolumn{{{l1}}}{{c|}}{{{l0}}}')}
'''

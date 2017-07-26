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

from .formatter import Formatter

import cgi     as _cgi
import numpy   as _np
import numbers as _numbers
import re      as _re
import os      as _os

from plotly.offline import plot as _plot

##############################################################################
#                          Formatting functions                              #
##############################################################################

'''
For documentation on creating additional formatters, see formatter.py

If the Formatter class does not offer enough functionality, 
  any function with the signature (item, specs -> string) can be used as a formatter
'''

# This dictionary is intentionally exported to other modules. 
# Even though it can change at runtime, it never does and should not
formatDict = dict()

# 'rho' (state prep) formatting
# Replace rho with &rho;
# Numbers following 'rho' -> subscripts
formatDict['Rho'] = {
    'html'  : Formatter(stringreplacers=[('rho', '&rho;')],
                         regexreplace=('.*?([0-9]+)$', '<sub>%s</sub>')),
    'latex' : Formatter(stringreplacers=[('rho', '\\rho')],
                         regexreplace=('.*?([0-9]+)$', '_{%s}'), formatstring='${}$')}

# 'E' (POVM) effect formatting
formatDict['Effect'] = {
    # If label == 'remainder', return E sub C
    # Otherwise, match regex and replace with subscript
    'html'  : Formatter(stringreturn=('remainder', 'E<sub>C</sub>'),
                         regexreplace=('.*?([0-9]+)$', '<sub>%s</sub>')),
    'latex' : Formatter(stringreturn=('remainder', '$E_C$'),
                         regexreplace=('.*?([0-9]+)$', '_{%s}'), formatstring='$%s$')}

# Normal replacements
formatDict['Normal'] = {
    'html'  : Formatter(html, 
                        ebstring='%s <span class="errorbar">+/- %s</span>', 
                        nmebstring='%s <span class="nmerrorbar">+/- %s</span>'),
    'latex' : Formatter(latex,
                        ebstring='$ \\begin{array}{c} %s \\\\ \pm %s \\end{array} $')} #nmebstring will match

# 'normal' formatting but round to 2 decimal places regardless of what is passed in to table.render()
formatDict['Rounded'] = {
    'html'  : Formatter(html,  defaults={'precision' : 2, 'sciprecision': 0}),
    'latex' : Formatter(latex, defaults={'precision' : 2, 'sciprecision': 0})}

# 'small' formating - make text smaller
formatDict['Small'] = {
    'html'  : Formatter(html),
    'latex' : Formatter(latex, formatstring='\\small%s')}

# 'small' formating - make text smaller
formatDict['Verbatim'] = {
    'html'  : Formatter(html),
    'latex' : Formatter(formatstring='\\spverb!%s!')}

# Pi formatters
formatDict['Pi'] = {
    'html'  : Formatter(html,  formatstring='%s&pi;'),
    'latex' : Formatter(latex, formatstring='%s$\\pi$')}

# BracketFormatters
formatDict['Brackets'] = {
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
    x = pre_convert_latex(str(x), specs)
    if '\\bigstar' in x:
        x = '${}$'.format(x)
    if "\\\\" in x:
        return '\\begin{tabular}{c}' + x + '\\end{tabular}'
    else:
        return x

convert_latex = Formatter(special_convert_latex)

formatDict['Conversion'] = {
    'html'  : Formatter(convert_html),
    'latex' : Formatter(convert_latex)}

formatDict['EBConversion'] = {
    'html'  : Formatter(convert_html, formatstring='<span class="errorbar">%s</span>'),
    'latex' : Formatter(convert_latex)}

formatDict['NMEBConversion'] = {
    'html'  : Formatter(convert_html, formatstring='<span class="nmerrorbar">%s</span>'),
    'latex' : Formatter(convert_latex)}

EB_html   = Formatter(html, ebstring='%s <span class="errorbar">+/- %s</span>')
NMEB_html = Formatter(html, nmebstring='%s <span class="nmerrorbar">+/- %s</span>')
EB_latex  = Formatter(latex, ebstring='$ \\begin{array}{c} %s \\\\ \pm %s \\end{array} $')

VEB_latex = Formatter(latex, ebstring='%s $\pm$ %s')

formatDict['VecErrorBars'] = {
    'html'  : EB_html,
    'latex' : VEB_latex}
formatDict['NMVecErrorBars'] = {
    'html'  : NMEB_html,
    'latex' : VEB_latex}

PiEB_latex = Formatter(latex, ebstring='$ \\begin{array}{c}(%s \\\\ \\pm %s)\\pi \\end{array} $', formatstring='%s$\\pi$')

# 'errorbars with pi' formatting: display (scalar_value +/- error bar) * pi
formatDict['PiErrorBars'] = {
    'html'  : Formatter(html, ebstring='(%s <span class="errorbar">+/- %s</span>)&pi'),
    'latex' : PiEB_latex}

formatDict['NMPiErrorBars'] = {
    'html'  : Formatter(html, ebstring='(%s <span class="nmerrorbar">+/- %s</span>)&pi'),
    'latex' : PiEB_latex}

formatDict['GateString'] = {
    'html'  : Formatter(lambda s,specs : '.'.join(s) if s is not None else ''),
    'latex' : Formatter(lambda s,specs : ''          if s is None else ('$%s$' % '\\cdot'.join([ ('\\mbox{%s}' % gl) for gl in s])))}

'''
Figure formatters no longer use Formatter objects, because figure formatters are more specialized.
Notice that they still have the function signature (item, specs -> string)
'''

def html_figure(fig, specs):
    fig.set_render_options(click_to_display=specs['click_to_display'])
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

formatDict['Figure'] = {
    'html'  : html_figure,
    'latex' : latex_figure}

# Bold formatting
formatDict['Bold'] = {
    'html'  : Formatter(html, formatstring='<b>%s</b>'),
    'latex' : Formatter(latex, formatstring='\\textbf{%s}')}

#Special formatting for Hamiltonian and Stochastic gateset types
formatDict['GatesetType'] = {
    'html'  : Formatter(),
    'latex' : Formatter(stringreplacers=[('H','$\\mathcal{H}$'),('S','$\\mathcal{S}$')])}

'''
# 'pre' formatting, where the user gives the data in separate formats
def _pre_fmt_template(formatname):
    return lambda label : label[formatname]

formatDict['Pre'] = {
    'html'   : _pre_fmt_template('html'),
    'latex'  : _pre_fmt_template('latex')}

#Multi-row and multi-column formatting (with "Conversion" type inner formatting)
formatDict['MultiRow'] = {
    'html'  : TupleFormatter(convert_html, formatstring='<td rowspan="{l1}">{l0}</td>'),
    'latex' : TupleFormatter(convert_latex, formatstring='\\multirow{{{l1}}}{{*}}{{{l0}}}')}

def _empty_str(l): return ""
def _return_None(l): return None #signals no <td></td> in HTML

formatDict['SpannedRow'] = {
    'html'  : _return_None,
    'latex' : _empty_str}

def _repeatno_format(label_tuple): 
    label, reps = label_tuple
    return ["%s" % label]*reps

formatDict['MultiCol'] = {
    'html'  : TupleFormatter(convert_html, formatstring='<td colspan="{l1}">{l0}</td>'),
    'latex' : TupleFormatter(convert_latex, formatstring='\\multicolumn{{{l1}}}{{c|}}{{{l0}}}')}
'''

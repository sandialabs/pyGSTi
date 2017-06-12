from __future__ import division, print_function, absolute_import, unicode_literals

#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************

""" Functions for generating report tables in different formats """

from .html  import html,  html_value
from .latex import latex, latex_value

from ..objects.formatters import *

import cgi     as _cgi
import numpy   as _np
import numbers as _numbers
import re      as _re
import os      as _os

from plotly.offline import plot as _plot

class FormatSet():
    '''
    Attributes
    ---------
    formatDict: Static dictionary containing small formatter dictionaries
                Ex: { 'Rho' :  { 'html' : ... , 'text' : ... }, ... }
                (created below)

    Methods
    -------

    __init__: (specs): Specs is a dictionary of the form { 'setting'(kwarg) : value }
                       Ex: { 'precision' : 6, 'polarprecision' : 3, 'sciprecision' : 0 }
                       given to Formatters that need them

    formatList : Given a list of items and formatters and a target format, returns formatted items
    '''
    formatDict = {}

    def __init__(self, specs):
        self.specs = specs

    def formatList(self, items, formatterNames, fmt):
        assert(len(items) == len(formatterNames))
        formatted_items = []

        for item, formatterName in zip(items, formatterNames):
            if formatterName is not None:
                formatter = FormatSet.formatDict[formatterName]
                formatted_item = formatter[fmt](item, self.specs)
                if formatted_item is None:
                    raise ValueError("Formatter " + str(type(formatter[fmt]))
                                     + " returned None for item = " + str(item))
                if isinstance(formatted_item, list): #formatters can create multiple table cells by returning *lists* 
                    formatted_items.extend(formatted_item)
                else:
                    formatted_items.append(formatted_item)
            else:
                if item.get_value() is not None:
                    formatted_items.append(str(item))
                else:
                    raise ValueError("Unformatted None in formatList")

        return formatted_items

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
                         regexreplace=('.*?([0-9]+)$', '_{%s}'), formatstring='$%s$')}

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
    'html'  : Formatter(html_value,  defaults={'precision' : 2, 'sciprecision': 0}),
    'latex' : Formatter(latex_value, defaults={'precision' : 2, 'sciprecision': 0})}

# Similar to the above two formatdicts,
# but recieves precision during table.render(), which is sent as kwarg to html_value, for example
FormatSet.formatDict['Precision'] = {
    'html'  : Formatter(html_value),
    'latex' : Formatter(latex_value)}

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
    def formatter(label):
        if str(label) == '--' or str(label) == '':
            return str(label)
        else:
            return b(label)

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

# These two formatters are more complex, justifying individual functions:

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
    ("Check","\\checkmark")])

def special_convert_latex(x):
    x = pre_convert_latex(str(x), {})
    if "<STAR>" in x: #assume <STAR> never has $ around it already
        x = "$" + x.replace("<STAR>","\\bigstar") + "$"
    if "|" in x:
        return '\\begin{tabular}{c}' + '\\\\'.join(x.split("|")) + '\\end{tabular}'
    else:
        return x

convert_latex = Formatter(special_convert_latex)

FormatSet.formatDict['Conversion'] = {
    'html'  : Formatter(convert_html),
    'latex' : Formatter(convert_latex)}

FormatSet.formatDict['EBConversion'] = {
    'html'  : Formatter(Formatter(convert_html, formatstring='<span class="errorbar">{}</span>')),
    'latex' : Formatter(convert_latex)}

FormatSet.formatDict['NMEBConversion'] = {
    'html'  : Formatter(Formatter(convert_html, formatstring='<span class="nmerrorbar">{}</span>')),
    'latex' : Formatter(convert_latex)}


_EB_html  = EBFormatter(Formatter(html),
                         '%s <span class="errorbar">+/- %s</span>')
_EB_html2  = EBFormatter(Formatter(html),
                         '%s <span class="nmerrorbar">+/- %s</span>')
_EB_latex = EBFormatter(Formatter(latex_value),
                       '$ \\begin{array}{c} %s \\\\ \pm %s \\end{array} $')

FormatSet.formatDict['ErrorBars'] = {
    'html'  : _EB_html,
    'latex' : _EB_latex}
FormatSet.formatDict['NMErrorBars'] = {
    'html'  : _EB_html2,
    'latex' : _EB_latex}

_VEB_latex = EBFormatter(Formatter(latex), '%s $\pm$ %s')

FormatSet.formatDict['VecErrorBars'] = {
    'html'  : _EB_html,
    'latex' : _VEB_latex}
FormatSet.formatDict['NMVecErrorBars'] = {
    'html'  : _EB_html2,
    'latex' : _VEB_latex}


_PiEB_latex = PiEBFormatter(Formatter(latex),
                           '$ \\begin{array}{c}(%s \\\\ \\pm %s)\\pi \\end{array} $',
                           '%s$\\pi$')
def _pi_eb_template(f):
    return EBFormatter(Formatter(html), '(%s <span class="errorbar">+/- %s</span>)&pi')
def _pi_eb_template2(f):
    return EBFormatter(Formatter(html), '(%s <span class="nmerrorbar">+/- %s</span>)&pi')

# 'errorbars with pi' formatting: display (scalar_value +/- error bar) * pi
FormatSet.formatDict['PiErrorBars'] = {
    'html'  : _pi_eb_template(html),
    'latex' : _PiEB_latex}

FormatSet.formatDict['NMPiErrorBars'] = {
    'html'  : _pi_eb_template2(html),
    'latex' : _PiEB_latex}

FormatSet.formatDict['GateString'] = {
    'html'  : Formatter(lambda s : '.'.join(s) if s is not None else ''),
    'latex' : Formatter(lambda s : ''          if s is None else ('$%s$' % '\\cdot'.join([ ('\\mbox{%s}' % gl) for gl in s])))}

'''
# 'pre' formatting, where the user gives the data in separate formats
def _pre_fmt_template(formatname):
    return lambda label : label[formatname]

FormatSet.formatDict['Pre'] = {
    'html'   : _pre_fmt_template('html'),
    'latex'  : _pre_fmt_template('latex')}
'''

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


'''
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


#Special formatting for Hamiltonian and Stochastic gateset types
FormatSet.formatDict['GatesetType'] = {
    'html'  : no_format,
    'latex' : Formatter(stringreplacers=[('H','$\\mathcal{H}$'),('S','$\\mathcal{S}$')])}

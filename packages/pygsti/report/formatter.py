from __future__ import division, print_function, absolute_import, unicode_literals

#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************

""" Functions for generating report tables in different formats """

from .html  import html,  html_value
from .latex import latex, latex_value
from .ppt   import ppt,   ppt_value

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
                       given to ParameterizedFormatters that need them

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
                give_specs(formatter[fmt], self.specs) # Parameters aren't sent until render call
                # Format the item once the formatter has been completely built
                formatted_item = formatter[fmt](item)
                if formatted_item is None:
                    raise ValueError("Formatter " + str(type(formatter[fmt]))
                                     + " returned None for item = " + str(item))
                if isinstance(formatted_item, list): #formatters can create multiple table cells by returning *lists* 
                    formatted_items.extend( formatted_item )
                else:
                    formatted_items.append( formatted_item )
            else:
                if item is None:
                    raise ValueError("Unformatted None in formatList")
                formatted_items.append( item )

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
                         regexreplace=('.*?([0-9]+)$', '_{%s}'), formatstring='$%s$'),
    'text'  : no_format,
    'ppt'   : no_format}

# 'E' (POVM) effect formatting
FormatSet.formatDict['Effect'] = {
    # If label == 'remainder', return E sub C
    # Otherwise, match regex and replace with subscript
    'html'  : Formatter(stringreturn=('remainder', 'E<sub>C</sub>'),
                         regexreplace=('.*?([0-9]+)$', '<sub>%s</sub>')),
    'latex' : Formatter(stringreturn=('remainder', '$E_C$'),
                         regexreplace=('.*?([0-9]+)$', '_{%s}'), formatstring='$%s$'),
    'text'  : no_format,
    'ppt'   : no_format}

# Normal replacements
FormatSet.formatDict['Normal'] = {
    'html'  : PrecisionFormatter(html),
    'latex' : PrecisionFormatter(latex),
    'text'  : no_format,
    'ppt'   : PrecisionFormatter(ppt) }

# 'normal' formatting but round to 2 decimal places regardless of what is passed in to table.render()
FormatSet.formatDict['Rounded'] = {
    'html'  : ParameterizedFormatter(html_value,  ['polarprecision'], {'precision' : 2, 'sciprecision': 0}),
    'latex' : ParameterizedFormatter(latex_value, ['polarprecision'], {'precision' : 2, 'sciprecision': 0}),
    'text'  : no_format,
    'ppt'   : ParameterizedFormatter(ppt_value,   ['polarprecision'], {'precision' : 2, 'sciprecision': 0})}

# Similar to the above two formatdicts,
# but recieves precision during table.render(), which is sent as kwarg to html_value, for example
FormatSet.formatDict['Precision'] = {
    'html'  : PrecisionFormatter(html_value),
    'latex' : PrecisionFormatter(latex_value),
    'text'  : no_format,
    'ppt'   : PrecisionFormatter(ppt_value)}

# 'small' formating - make text smaller
FormatSet.formatDict['Small'] = {
    'html'  : PrecisionFormatter(html),
    'latex' : PrecisionFormatter(latex, formatstring='\\small%s'),
    'text'  : no_format,
    'ppt'   : PrecisionFormatter(ppt)}

# 'small' formating - make text smaller
FormatSet.formatDict['Verbatim'] = {
    'html'  : PrecisionFormatter(html),
    'latex' : Formatter(formatstring='\\spverb!%s!'),
    'text'  : no_format,
    'ppt'   : PrecisionFormatter(ppt)}

#############################################
# Helper functions for formatting pi-labels #
#############################################

def _pi_template(b):
    # Pi Formatting shares a common predicate and first branch condition
    return BranchingFormatter(lambda label : str(label) == '--' or str(label) == '',
                              no_format, b)

# Requires an additional predicate
def _pi_text(label):
    if label == '--' or label == '' or not isinstance(label, _numbers.Number):
        return label
    else:
        return label * _np.pi

# Pi formatters
FormatSet.formatDict['Pi'] = {
    'html'  : _pi_template(PrecisionFormatter(html,  formatstring='%s&pi;')),
    'latex' : _pi_template(PrecisionFormatter(latex, formatstring='%s$\\pi$')),
    'text'  : _pi_text,
    'ppt'   : _pi_template(PrecisionFormatter(ppt,   formatstring='%spi'))}

# Bracket Formatters
FormatSet.formatDict['Brackets'] = {
    'html'  : PrecisionFormatter(html,  defaults={'brackets' : True}),
    'latex' : PrecisionFormatter(latex, defaults={'brackets' : True}),
    'text'  : no_format,
    'ppt'   : PrecisionFormatter(ppt,   defaults={'brackets' : True})}

##################################################################################
# 'conversion' formatting: catch all for find/replacing specially formatted text #
##################################################################################

# These two formatters are more complex, justifying individual functions:

def _fmtCnv_html(x):
    x = x.replace("\\", "&#92"); #backslash
    x = x.replace("|"," ") #remove pipes=>newlines, since html wraps table text automatically
    x = x.replace("<STAR>","REPLACEWITHSTARCODE") #b/c cgi.escape would mangle <STAR> marker
    #x = _cgi.escape(x).encode("ascii","xmlcharrefreplace")
    x = x.replace("REPLACEWITHSTARCODE", '&#9733;') #replace new marker with HTML code
    return str(x)

def _fmtCnv_html_eb(x):
    return '<span class="errorbar">' + _fmtCnv_html(x) + '</span>'
def _fmtCnv_html_nmeb(x):
    return '<span class="nmerrorbar">' + _fmtCnv_html(x) + '</span>'

def _fmtCnv_latex(x):
    x = x.replace("\\", "\\textbackslash")
    x = x.replace('%','\\%')
    x = x.replace('#','\\#')
    x = x.replace("half-width", "$\\nicefrac{1}{2}$-width")
    x = x.replace("1/2", "$\\nicefrac{1}{2}$")
    x = x.replace("Diamond","$\\Diamond$")
    x = x.replace("Check","\\checkmark")
    if "<STAR>" in x: #assume <STAR> never has $ around it already
        x = "$" + x.replace("<STAR>","\\bigstar") + "$"
    if "|" in x:
        return '\\begin{tabular}{c}' + '\\\\'.join(x.split("|")) + '\\end{tabular}'
    else:
        return x

FormatSet.formatDict['Conversion'] = {
    'html'  : _fmtCnv_html,
    'latex' : _fmtCnv_latex,
    'text'  : Formatter(stringreplacers=[('<STAR>', '*'), ('|', ' ')]),
    'ppt'   : Formatter(stringreplacers=[('<STAR>', '*'), ('|', '\n')])}

FormatSet.formatDict['EBConversion'] = {
    'html'  : _fmtCnv_html_eb,
    'latex' : _fmtCnv_latex,
    'text'  : Formatter(stringreplacers=[('<STAR>', '*'), ('|', ' ')]),
    'ppt'   : Formatter(stringreplacers=[('<STAR>', '*'), ('|', '\n')])}

FormatSet.formatDict['NMEBConversion'] = {
    'html'  : _fmtCnv_html_nmeb,
    'latex' : _fmtCnv_latex,
    'text'  : Formatter(stringreplacers=[('<STAR>', '*'), ('|', ' ')]),
    'ppt'   : Formatter(stringreplacers=[('<STAR>', '*'), ('|', '\n')])}

_eb_exists = lambda t : t[1] is not None

class _EBFormatter(object):
    def __init__(self, f, formatstringA='%s +/- %s', formatstringB='%s'):
        self.f = f
        if hasattr(f, 'specs'):
            self.specs = f.specs
        self.formatstringA = formatstringA
        self.formatstringB = formatstringB

    def __call__(self, t):
        if hasattr(self.f, 'specs'):
            give_specs(self.f, self.specs)
        if _eb_exists(t):
            return self.formatstringA % (self.f(t[0]), self.f(t[1]))
        else:
            return self.formatstringB % self.f(t[0])

_EB_html  = _EBFormatter(PrecisionFormatter(html),
                         '%s <span class="errorbar">+/- %s</span>')
_EB_html2  = _EBFormatter(PrecisionFormatter(html),
                         '%s <span class="nmerrorbar">+/- %s</span>')
_EB_latex = _EBFormatter(PrecisionFormatter(latex_value),
                       '$ \\begin{array}{c} %s \\\\ \pm %s \\end{array} $')
_EB_text  = lambda t : {'value' : t[0], 'errbar' : t[1]}
_EB_ppt   = _EBFormatter(PrecisionFormatter(ppt))

FormatSet.formatDict['ErrorBars'] = {
    'html'  : _EB_html,
    'latex' : _EB_latex,
    'text'  : _EB_text,
    'ppt'   : _EB_ppt }
FormatSet.formatDict['NMErrorBars'] = {
    'html'  : _EB_html2,
    'latex' : _EB_latex,
    'text'  : _EB_text,
    'ppt'   : _EB_ppt }

_VEB_latex = _EBFormatter(PrecisionFormatter(latex), '%s $\pm$ %s')

FormatSet.formatDict['VecErrorBars'] = {
    'html'  : _EB_html,
    'latex' : _VEB_latex,
    'text'  : _EB_text,
    'ppt'   : _EB_ppt}
FormatSet.formatDict['NMVecErrorBars'] = {
    'html'  : _EB_html2,
    'latex' : _VEB_latex,
    'text'  : _EB_text,
    'ppt'   : _EB_ppt}

class _PiEBFormatter(_EBFormatter):
    def __call__(self, t):
        if str(t[0]) == '--' or str(t[0]) == '':  return t[0]
        else:
            return super(_PiEBFormatter, self).__call__(t)

_PiEB_latex = _PiEBFormatter(PrecisionFormatter(latex),
                           '$ \\begin{array}{c}(%s \\\\ \\pm %s)\\pi \\end{array} $',
                           '%s$\\pi$')
def _pi_eb_template(f):
    return _EBFormatter(PrecisionFormatter(html), '(%s <span class="errorbar">+/- %s</span>)&pi')
def _pi_eb_template2(f):
    return _EBFormatter(PrecisionFormatter(html), '(%s <span class="nmerrorbar">+/- %s</span>)&pi')

# 'errorbars with pi' formatting: display (scalar_value +/- error bar) * pi
FormatSet.formatDict['PiErrorBars'] = {
    'html'  : _pi_eb_template(html),
    'latex' : _PiEB_latex,
    'text'  : _EB_text,
    'ppt'   : _pi_eb_template(ppt)}

FormatSet.formatDict['NMPiErrorBars'] = {
    'html'  : _pi_eb_template2(html),
    'latex' : _PiEB_latex,
    'text'  : _EB_text,
    'ppt'   : _pi_eb_template(ppt)}


FormatSet.formatDict['GateString'] = {
    'html'  : lambda s : '.'.join(s) if s is not None else '',
    'latex' : lambda s : ''          if s is None else ('$%s$' % '\\cdot'.join([ ('\\mbox{%s}' % gl) for gl in s])),
    'text'  : lambda s : tuple(s)    if s is not None else '',
    'ppt'   : lambda s : '.'.join(s) if s is not None else ''}

# 'pre' formatting, where the user gives the data in separate formats
def _pre_fmt_template(formatname):
    return lambda label : label[formatname]

FormatSet.formatDict['Pre'] = {
    'html'   : _pre_fmt_template('html'),
    'latex'  : _pre_fmt_template('latex'),
    'text'   : _pre_fmt_template('text'),
    'ppt'    : _pre_fmt_template('ppt')}


FormatSet.formatDict['Figure'] = {
    'html'  : HTMLFigureFormatter(),
    'latex' : FigureFormatter(formatstring="\\vcenteredhbox{\\includegraphics[width=%.2fin,height=%.2fin,keepaspectratio]{%s/%s}}",
                               extension='.pdf'),
    'text'  : lambda figInfo : figInfo[0],
    'ppt'   : lambda figInfo : 'Figure formatting not implemented for ppt'} # Not Implemented

# Bold formatting
FormatSet.formatDict['Bold'] = {
    'html'  : PrecisionFormatter(html, formatstring='<b>%s</b>'),
    'latex' : PrecisionFormatter(latex, formatstring='\\textbf{%s}'),
    'text'  : Formatter(formatstring='**%s**'),
    'ppt'   : PrecisionFormatter(ppt)} # No bold in ppt?


#Multi-row and multi-column formatting (with "Conversion" type inner formatting)
FormatSet.formatDict['MultiRow'] = {
    'html'  : TupleFormatter(_fmtCnv_html, formatstring='<td rowspan="{l1}">{l0}</td>'),
    'latex' : TupleFormatter(_fmtCnv_latex, formatstring='\\multirow{{{l1}}}{{*}}{{{l0}}}'),
    'text'  : TupleFormatter(Formatter(stringreplacers=[('<STAR>', '*'), ('|', ' ')])),
    'ppt'   : TupleFormatter(Formatter(stringreplacers=[('<STAR>', '*'), ('|', '\n')]))}


def _empty_str(l): return ""
def _return_None(l): return None #signals no <td></td> in HTML

FormatSet.formatDict['SpannedRow'] = {
    'html'  : _return_None,
    'latex' : _empty_str,
    'text'  : Formatter(stringreplacers=[('<STAR>', '*'), ('|', ' ')]),
    'ppt'   : Formatter(stringreplacers=[('<STAR>', '*'), ('|', '\n')])}

def _repeatno_format(label_tuple): 
    label, reps = label_tuple
    return ["%s" % label]*reps

FormatSet.formatDict['MultiCol'] = {
    'html'  : TupleFormatter(_fmtCnv_html, formatstring='<td colspan="{l1}">{l0}</td>'),
    'latex' : TupleFormatter(_fmtCnv_latex, formatstring='\\multicolumn{{{l1}}}{{c|}}{{{l0}}}'),
    'text'  : _repeatno_format,
    'ppt'   : _repeatno_format}


#Special formatting for Hamiltonian and Stochastic gateset types
FormatSet.formatDict['GatesetType'] = {
    'html'  : no_format,
    'latex' : Formatter(stringreplacers=[('H','$\\mathcal{H}$'),('S','$\\mathcal{S}$')]),
    'text'  : no_format,
    'ppt'   : no_format}

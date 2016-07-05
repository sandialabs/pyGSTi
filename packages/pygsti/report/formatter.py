from __future__ import division, print_function, absolute_import, unicode_literals
#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************
""" Functions for generating report tables in different formats """

from .latex import latex, latex_value
from .html  import html,  html_value
from .ppt   import ppt,   ppt_value

import cgi          as _cgi
import numpy        as _np
import numbers      as _numbers
import re           as _re
import os           as _os

class Formatter():
    '''
    Class for formatting strings to html, latex, powerpoint, or text

    Parameters
    --------
    stringreplacers : tuples of the form (pattern, replacement)
                   (replacement is a normal string)
                 Ex : [('rho', '&rho;')]
    regexreplace  : A tuple of the form (regex,   replacement)
                   (replacement is formattable string,
                      gets formatted with grouped result of regex matching on label)
                 Ex : ('.*?([0-9]+)$', '_{%s}')

    formatstring : Outer formatting for after both replacements have been made
 
    Returns
    --------
    template :
    Formatting function
    '''

    def __init__(stringreplacers=None, regexreplace=None, formatstring='%s', stringreturn=None):
        self.stringreplacers = stringreplacers
        self.regexreplace = regexreplace
        self.formatstring = formatstring

    def __call__():
        '''
        Formatting function template

        Parameters
        --------
        label : the label to be formatted!
        Returns
        --------
        Formatted label
        '''
        if self.stringreturn is not None:
            return self.formatstring % label.replace(stringreturn[0], stringreturn[1])

        if self.stringreplacers is not None:
            for stringreplace in self.stringreplacers:
                label = label.replace(stringreplace[0], stringreplace[1])
        if self.regexreplace is not None:
             result = _re.match(self.regexreplace[0], label)
             if result is not None:
                 grouped = result.group(1)
                 label   = label[0:-len(grouped)] + (self.regexreplace[1] % grouped)
        return self.formatstring % label

# A traditional function, so that pickling is possible
def no_format(label):
    return label

##############################################################################
#Formatting functions
##############################################################################
# 'rho' (state prep) formatting
Rho = { 'html' : Formatter(stringreplacers=[('rho', '&rho;')], 
                                 regexreplace=('.*?([0-9]+)$', '<sub>%s</sub>')), 
        'latex': Formatter(stringreplacers=[('rho', '\\rho')], 
                                 regexreplace=('.*?([0-9]+)$', '_{%s}'), formatstring='$%s$'), 
        'text'   : no_format,
        'ppt'  : no_format }

# 'E' (POVM) effect formatting
Effect = { 
      'html'   : Formatter(stringreturn=('remainder', 'E<sub>C</sub>'), 
                                regexreplace=('.*?([0-9]+)$', '<sub>%s</sub>')), # Regexreplace potentially doesn't run
      'latex'  : Formatter(stringreturn=('remainder', '$E_C$'),
                                regexreplace=('.*?([0-9]+)$', '_{%s}')), 
      'text'   : no_format, 
      'ppt'    : no_format}

# Normal replacements
Normal = { 
        'html'   : html, 
        'latex'  : latex, 
        'text'   : no_format, 
        'ppt'    : ppt }

# 'normal' formatting but round to 2 decimal places
Rounded = { 
         'html'  : lambda x : html_value(x, ROUND=2), 
         'latex' : lambda x : latex_value(x, ROUND=2), 
         'text'  : no_format, 
         'ppt'   : lambda x : ppt_value(x, ROUND=2) }

# 'small' formating - make text smaller
Small = { 
        'html'   : html, 
        'latex'  : lambda x : '\\small' + latex(x), 
        'text'   : no_format, 
        'ppt'    : ppt}

emptyOrDash = lambda x : return x == '--' or x = ''

Pi = { 'html'   : lambda x : x if emptyOrDash(x) else html(x) + '&pi;', 
       'latex'  : lambda x : x if emptyOrDash(x) else latex(x) + '$\\pi$', 
       'text'   : lambda x : x if emptyOrDash(x) or not isinstance(x, _numbers.Number else x * _np.pi,
       'ppt'    : lambda x : x if emptyOrDash(x) else ppt(x) + 'pi' }

Brackets = { 
        'html'  : lambda x : html(x, brackets=True), 
        'latex' : lambda x : latex(x, brackets=True), 
        'text'  : no_format, 
        'ppt'   : lambda x : ppt(x, brackets=True)}


# ####################################################################### #
# These formatters are more complex, I'll keep them how they are for now. #
# ####################################################################### #

# 'conversion' formatting: catch all for find/replacing specially formatted text
def _fmtCnv_html(x):
    x = x.replace("|"," ") #remove pipes=>newlines, since html wraps table text automatically
    x = x.replace("<STAR>","REPLACEWITHSTARCODE") #b/c cgi.escape would mangle <STAR> marker
    x = _cgi.escape(x).encode("ascii","xmlcharrefreplace")
    x = x.replace("REPLACEWITHSTARCODE","&#9733;") #replace new marker with HTML code
    return x
def _fmtCnv_latex(x):
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

Conversion = { 
           'html'  : _fmtCnv_html, 
           'latex' : _fmtCnv_latex, 
           'text'  : Formatter(stringreplacers=[('<STAR>', '*'), ('|', ' ')]), 
           'ppt'   : Formatter(stringreplacers=[('<STAR>', '*'), ('|', '\n')])}

# Essentially takes two formatters and decides which to use, based on if the error bar exists
def eb_template(a, b):
    def template(t):
        if t[1] is not None:
            # A corresponds to when the error bar is present
            return a(t)
        else:
            return b(t)

ErrorBars = { 'html'  : eb_template(lambda t : '%s +/- %s' % (html(t[0], html(t[1])
                                    lambda t : html(t[0])), 
              'latex' : eb_template(lambda t : '$ \\begin{array}{c} %s \\\\ \pm %s \\end{array} $' % (latex_value(t[0]), latex_value(t[1])),
                                    lambda t : latex_value(t[0])) 
              'text'  : lambda t : { 'value' : t[0], 'errbar' : t[1] }, 
              'ppt'   : eb_template(lambda t : '%s +/- %s' % (ppt(t[0]), ppt(t[1])),
                                    lambda t : ppt(t[0]))}

VecErrorBars = { 'html'  : eb_template(lambda t : '%s +/- %s' % (html(t[0]), html(t[1])),
                                       lambda t : html(t[0]), 
                 'latex' : eb_template(lambda t : '%s $\pm$ %s' % (latex(t[0]), latex(t[1])),
                                       lambda t : latex(t[0])), 
                 'text'  : lambda t : return { 'value' : t[0], 'errbar' : t[1] }, 
                 'ppt'   : eb_template(lambda t : '%s +/- %s' % (ppt(t[0]), ppt(t[1])),
                                       lambda t : ppt(t[0])}


# 'errorbars with pi' formatting: display (scalar_value +/- error bar) * pi
PiErrorBars = { 'html'  : eb_template(lambda t : '(%s +/- %s)&pi;' % (html(t[0]), html(t[1])),
                                      lambda t : Pi['html'](t[0])), 
                'latex' : eb_template(lambda t : '$ \\begin{array}{c}(%s \\\\ ]pm %s)\\pi \\end{array} $' % (latex(t[0]), latex(t[1])),
                                      lambda t : Pi['latex'](t[0])), 
                'text'  : lambda t : {'value' : t[0], 'errbar' : t[1]}, 
                'ppt'   : eb_template(lambda t : '(%s +/- %s)pi' % (ppt(t[0]), ppt(t[1])),
                                      lambda t : ppt(t[0]))}
GateString = {
         'html'  : lambda s : '.'.join(s) if s is not None else '', 
         'latex' : lambda s : '' if s is None else ('$%s$' % '\\cdot'.join([ ('\\mbox{%s}' % gl) for gl in s])), 
         'text'  : lambda s : tuple(s) if s is not None else '', 
         'ppt'   : lambda s : '.'.join(s) if s is not None else ''}
# 'pre' formatting, where the user gives the data in separate formats
Pre = { 'html'   : lambda x : x['html'], 
        'latex'  : lambda x : x['latex'], 
        'text'   : lambda x : x['text'], 
        'ppt'    : lambda x : x['ppt'] }

# Still a bit hacky, but no global required
def build_figure_formatters(scratchDir):
    # Specifically, this function remains unevaluated until table rendering - which calls formatList (below)
    #   in formatList, the scratchDir is given, and this function returns the appropriate dictionary of formatters

    # Figure formatting, where a GST figure is displayed in a table cell
    def _fmtFig_html(figInfo):
        fig, name, W, H = figInfo
        fig.save_to(_os.path.join(scratchDir, name + ".png"))
        return "<img width='%.2f' height='%.2f' src='%s/%s'>" \
            % (W, H, scratchDir,name + ".png")
    def _fmtFig_latex(figInfo):
        fig, name, W, H = figInfo
        fig.save_to(_os.path.join(scratchDir, name + ".pdf"))
        return "\\vcenteredhbox{\\includegraphics[width=%.2fin,height=%.2fin" \
            % (W,H) + ",keepaspectratio]{%s/%s}}" % (scratchDir, name + ".pdf")
    
    Fig = { 'html' : _fmtFig_html, 
            'latex': _fmtFig_latex, 
            'text' : lambda figinfo : return figInfo[0], 
            'ppt'  : lambda figinfo : 'Not implemented' }
    return Fig

Figure = build_figure_formatters 

# Bold formatting
Bold = { 'html'  : lambda x : '<b>%s</b>' % html(x),
         'latex' : lambda x : '\\textbf{%s}' % latex(x), 
         'text'  : Formatter(formatstring='**%s**'), 
         'ppt'   : lambda x : ppt(x)}



def formatList(items, formatters, fmt, scratchDir=None):
    assert(len(items) == len(formatters))
    formatted_items = []
    for item, formatter in zip(items, formatters):
        if formatter is not None:
            # If the formatter requires additional information to do its job, give it.
            if callable(formatter):
                formatter = formatter(scratchDir)
            formatted_items.append( formatter[fmt](item) )
        else:
            formatted_items.append( item )
    return formatted_items

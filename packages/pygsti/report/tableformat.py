from __future__ import division, print_function, absolute_import, unicode_literals
#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************
""" Functions for generating report tables in different formats """

from .latex import latex, latex_value
from .html  import html,  html_value
from .pu    import ppt,   ppt_value

import cgi          as _cgi
import numpy        as _np
import re           as _re
import os           as _os

# A factory function for building formatting functions
def build_formatter(regexreplace=None, formatstring='%s', stringreturn=None, stringreplacers=None):
    '''
    Factory function for building formatters!

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
    def template(label):
        '''
        Formatting function template

        Parameters
        --------
        label : the label to be formatted!
        Returns
        --------
        Formatted label
        '''
        # Potential early exit:
        if stringreturn is not None:
            if label == stringreturn[0]: return stringreturn[1]

        if stringreplacers is not None:
            for stringreplace in stringreplacers:
                label = label.replace(stringreplace[0], stringreplace[1])
        if regexreplace is not None:
             result = _re.match(regexreplace[0], label)
             if result is not None:
                 grouped = result.group(1)
                 label = label[0:-len(grouped)] + (regexreplace[1] % grouped)
        return formatstring % label
    return template

no_format = lambda label : label # Do nothing! :)

##############################################################################
#Formatting functions
##############################################################################
# 'rho' (state prep) formatting
Rho = { 'html' : build_formatter(stringreplacers=[('rho', '&rho;')], 
                                 regexreplace=('.*?([0-9]+)$', '<sub>%s</sub>')), 
        'latex': build_formatter(stringreplacers=[('rho', '\\rho')], 
                                 regexreplace=('.*?([0-9]+)$', '_{%s}'), formatstring='$%s$'), 
        'py'   : no_format,
        'ppt'  : no_format }

# 'E' (POVM) effect formatting
Effect = { 'html'   : build_formatter(stringreturn=('remainder', 'E<sub>C</sub>'), 
                                regexreplace=('.*?([0-9]+)$', '<sub>%s</sub>')), # Regexreplace potentially doesn't run
      'latex'  : build_formatter(stringreturn=('remainder', '$E_C$'),
                                regexreplace=('.*?([0-9]+)$', '_{%s}')), 
      'py'     : no_format, 
      'ppt'    : no_format}

# Normal replacements
Normal = { 
        'html'   : html, 
        'latex'  : latex, 
        'py'     : no_format, 
        'ppt'    : ppt }

# 'normal' formatting but round to 2 decimal places
Rounded = { 
         'html'  : lambda x : html_value(x, ROUND=2), 
         'latex' : lambda x : latex_value(x, ROUND=2), 
         'py'    : no_format, 
         'ppt'   : lambda x : ppt_value(x, ROUND=2) }

# 'small' formating - make text smaller
Small = { 
        'html'   : html, 
        'latex'  : lambda x : '\\small' + latex(x), 
        'py'     : no_format, 
        'ppt'    : ppt}

# 'pi' formatting: add pi symbol/text after given quantity
def _fmtPi_py(x):
    if x == "" or x == "--": return ""
    else:
        try: return x * _np.pi #but sometimes can't take product b/c x isn't a number
        except: return x

Pi = { 'html'   : lambda x : x if x == "--" or x == "" else html(x) + '&pi;', 
       'latex'  : lambda x : x if x == "--" or x == "" else latex(x) + '$\\pi$', 
       'py'     : _fmtPi_py,
       'ppt'    : lambda x : x if x == "--" or x == "" else ppt(x) + 'pi' }

Brackets = { 
        'html'  : lambda x : html(x, brackets=True), 
        'latex' : lambda x : latex(x, brackets=True), 
        'py'    : no_format, 
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
           'py'    : build_formatter(stringreplacers=[('<STAR>', '*'), ('|', ' ')]), 
           'ppt'   : build_formatter(stringreplacers=[('<STAR>', '*'), ('|', '\n')])}

# 'errorbars' formatting: display a scalar value +/- error bar
def _fmtEB_html(t):
    if t[1] is not None:
        return "%s +/- %s" % (html(t[0]), html(t[1]))
    else: return html(t[0])
def _fmtEB_latex(t):
    if t[1] is not None:
        return "$ \\begin{array}{c} %s \\\\ \pm %s \\end{array} $" % (latex_value(t[0]), latex_value(t[1]))
    else: return latex_value(t[0])
def _fmtEB_py(t):
    return { 'value': t[0], 'errbar': t[1] }
def _fmtEB_ppt(t):
    if t[1] is not None:
        return "%s +/- %s" % (ppt(t[0]), ppt(t[1]))
    else: return ppt(t[0])

ErrorBars = { 'html': _fmtEB_html, 'latex': _fmtEB_latex, 'py': _fmtEB_py, 'ppt': _fmtEB_ppt }


# 'vector errorbars' formatting: display a vector value +/- error bar
def _fmtEBvec_html(t):
    if t[1] is not None:
        return "%s +/- %s" % (html(t[0]), html(t[1]))
    else: return html(t[0])
def _fmtEBvec_latex(t):
    if t[1] is not None:
        return "%s $\pm$ %s" % (latex(t[0]), latex(t[1]))
    else: return latex(t[0])
def _fmtEBvec_py(t): return { 'value': t[0], 'errbar': t[1] }
def _fmtEBvec_ppt(t):
    if t[1] is not None:
        return "%s +/- %s" % (ppt(t[0]), ppt(t[1]))
    else: return ppt(t[0])
VecErrorBars = { 'html': _fmtEBvec_html, 'latex': _fmtEBvec_latex, 'py': _fmtEBvec_py, 'ppt': _fmtEBvec_ppt }


# 'errorbars with pi' formatting: display (scalar_value +/- error bar) * pi
def _fmtEBPi_html(t):
    if t[1] is not None:
        return "(%s +/- %s)&pi;" % (html(t[0]), html(t[1]))
    else: return _fmtPi_html(t[0])
def _fmtEBPi_latex(t):
    if t[1] is not None:
        return "$ \\begin{array}{c}(%s \\\\ \pm %s)\\pi \\end{array} $" % (latex(t[0]), latex(t[1]))
    else: return _fmtPi_latex(t[0])
def _fmtEBPi_py(t): return { 'value': t[0], 'errbar': t[1] }
def _fmtEBPi_ppt(t):
    if t[1] is not None:
        return "(%s +/- %s)pi" % (ppt(t[0]), ppt(t[1]))
    else: return ppt(t[0])
PiErrorBars = { 'html': _fmtEBPi_html, 'latex': _fmtEBPi_latex, 'py': _fmtEBPi_py, 'ppt': _fmtEBPi_ppt }


# 'gatestring' formatting: display a gate string
def _fmtGStr_latex(s):
    if s is None:
        return ""
    else:
        boxed = [ ("\\mbox{%s}" % gl) for gl in s ]
        return "$" + '\\cdot'.join(boxed) + "$"

GateString = {
         'html'  : lambda s : '.'.join(s) if s is not None else '', 
         'latex' : _fmtGStr_latex, 
         'py'    : lambda s : tuple(s) if s is not None else '', 
         'ppt'   : lambda s : '.'.join(s) if s is not None else ''}
# 'pre' formatting, where the user gives the data in separate formats
Pre = { 'html'   : lambda x : x['html'], 
        'latex'  : lambda x : x['latex'], 
        'py'     : lambda x : x['py'], 
        'ppt'    : lambda x : x['ppt'] }

# Still a bit hacky, but no global required
def build_figure_formatters(scratchDir):

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
    def _fmtFig_py(figInfo):
        fig, name, W, H = figInfo
        return fig
    def _fmtFig_ppt(figInfo):
        return "Not Impl."
    
    Fig = { 'html': _fmtFig_html, 'latex': _fmtFig_latex, 'py': _fmtFig_py, 'ppt': _fmtFig_ppt }
    return Fig

Figure = build_figure_formatters 

# Bold formatting
Bold = { 'html'  : lambda x : '<b>%s</b>' % html(x),
         'latex' : lambda x : '\\textbf{%s}' % latex(x), 
         'py'    : build_formatter(formatstring='**%s**'), 
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

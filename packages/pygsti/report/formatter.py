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

import cgi     as _cgi
import numpy   as _np
import numbers as _numbers
import re      as _re
import os      as _os

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
 
    custom : tuple of a function and additional key word arguments

    Returns
    --------
    template :
    Formatting function
    '''

    def __init__(self, stringreplacers=None, regexreplace=None, formatstring='%s', stringreturn=None, custom=None):
        self.stringreplacers = stringreplacers
        self.regexreplace    = regexreplace
        self.formatstring    = formatstring
        self.stringreturn    = stringreturn
        self.custom          = custom

    def __call__(self, label):
        '''
        Formatting function template

        Parameters
        --------
        label : the label to be formatted!
        Returns
        --------
        Formatted label
        '''
        if self.custom is not None:
            return self.custom[0](label, **self.custom[1])

        if self.stringreturn is not None:
            return self.formatstring % label.replace(self.stringreturn[0], self.stringreturn[1])

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
        'text' : no_format,
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
         'html'  : Formatter(custom=(html_value, {'ROUND' : 2})), 
         'latex' : Formatter(custom=(latex_value, {'ROUND' : 2})), 
         'text'  : no_format, 
         'ppt'   : Formatter(custom=(ppt_value, {'ROUND' : 2}))}

# 'small' formating - make text smaller
Small = { 
        'html'   : html, 
        'latex'  : Formatter(formatstring='\\small%s', custom=(latex, {})), 
        'text'   : no_format, 
        'ppt'    : ppt }

def emptyOrDash (x): 
    return x == '--' or x == ''

def _pi_html(x):
    return x if emptyOrDash(x) else html(x) + '&pi;'

def _pi_latex(x):
    return x if emptyOrDash(x) else latex(x) + '$\\pi$'

def _pi_text(x):
    return x if emptyOrDash(x) or not isinstance(x, _numbers.Number) else x * _np.pi

def _pi_ppt(x):
    return x if emptyOrDash(x) else ppt(x) + 'pi'

Pi = { 'html'   : _pi_html, 
       'latex'  : _pi_latex, 
       'text'   : _pi_text,
       'ppt'    : _pi_ppt}

Brackets = { 
        'html'  : Formatter(custom=(html, {'brackets' : True})), 
        'latex' : Formatter(custom=(latex, {'brackets' : True})), 
        'text'  : no_format, 
        'ppt'   : Formatter(custom=(ppt, {'brackets' : True}))}


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

class ErrorBarFormatter():
    # Essentially takes two formatters and decides which to use, based on if the error bar exists
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def __call__(self, t):
        if t[1] is not None:
            # A corresponds to when the error bar is present
            return self.a(t)
        else:
            return self.b(t)

def _text_error_bar(t):
    return {'value' : t[0], 'errbar' : t[1]}

def _latex_error_bar(t):
    return '$ \\begin{array}{c} %s \\\\ \pm %s \\end{array} $' % (latex_value(t[0]), latex_value(t[1]))

def _plus_or_minus(t, f=no_format):
    return '%s +/- %s' % (f(t[0]), f(t[1]))

def _first_tuple_elem(t, f=no_format):
    return f(t[0])

_html_error_bar = ErrorBarFormatter(Formatter(custom=(_plus_or_minus, {'f' : html})),
                                          Formatter(custom=(_first_tuple_elem, {'f' : html})))

ErrorBars = { 'html'  : _html_error_bar, 
              'latex' : ErrorBarFormatter(_latex_error_bar,
                                          Formatter(custom=(_first_tuple_elem, {'f' : latex_value}))), 
              'text'  : _text_error_bar, 
              'ppt'   : ErrorBarFormatter(Formatter(custom=(_plus_or_minus, {'f' : ppt})),
                                          Formatter(custom=(_first_tuple_elem, {'f' : ppt})))}
def _latex_vec_error_bar(t):
    return '%s $\pm$ %s' % (latex(t[0]), latex(t[1]))

VecErrorBars = {'html': _html_error_bar,
                'latex': ErrorBarFormatter(_latex_vec_error_bar,
                                           Formatter(custom=(_first_tuple_elem,
                                                             {'f': latex}))),
                'text': _text_error_bar,
                'ppt': ErrorBarFormatter(Formatter(custom=(_plus_or_minus,
                                                           {'f': ppt})),
                                         Formatter(custom=(_first_tuple_elem,
                                                           {'f': ppt})))}

def _latex_pi_error_bar(t):
    return ('$ \\begin{array}{c}(%s \\\\ ]pm %s)\\pi \\end{array} $'
            % (latex(t[0]), latex(t[1])))

# 'errorbars with pi' formatting: display (scalar_value +/- error bar) * pi
PiErrorBars = {'html': ErrorBarFormatter(Formatter(formatstring='(%s)&pi;',
                                                   custom=(_plus_or_minus,
                                                           {'f': html})),
                                         Formatter(custom=(_first_tuple_elem,
                                                           {'f': Pi['html']}))),
               'latex': ErrorBarFormatter(_latex_pi_error_bar,
                                          Formatter(custom=(_first_tuple_elem,
                                                            {'f': latex}))),
               'text': _text_error_bar,
               'ppt': ErrorBarFormatter(Formatter(formatstring='(%s)&pi;',
                                                  custom=(_plus_or_minus,
                                                          {'f': ppt})),
                                        Formatter(custom=(_first_tuple_elem,
                                                          {'f': ppt})))}

def _html_gatestring(s):
    return '.'.join(s) if s is not None else ''

def _latex_gatestring(s):
    return '' if s is None else ('$%s$' % '\\cdot'.join([ ('\\mbox{%s}' % gl) for gl in s]))

def _text_gatestring(s):
    return tuple(s) if s is not None else ''

def _ppt_gatestring(s):
    return '.'.join(s) if s is not None else ''

GateString = {
         'html'  : _html_gatestring, 
         'latex' : _latex_gatestring, 
         'text'  : _text_gatestring, 
         'ppt'   : _ppt_gatestring}

def _pre_html(x):
    return x['html']
def _pre_latex(x):
    return x['latex']
def _pre_text(x):
    return x['text']
def _pre_ppt(x):
    return x['ppt']

# 'pre' formatting, where the user gives the data in separate formats
Pre = { 'html'   : _pre_html,
        'latex'  : _pre_latex, 
        'text'   : _pre_text, 
        'ppt'    : _pre_ppt}

class FigureFormatter():
    def __init__(self, extension=None, formatstring='%s%s%s%s', custom=None):
        self.extension = extension
        self.custom = custom
        self.formatstring = formatstring
        self.scratchDir = None
        

    def __call__(self, figInfo):
        fig, name, W, H = figInfo
        if self.extension is not None:
            fig.save_to(_os.path.join(self.scratchDir, name + self.extension))
            if self.custom is not None:
                return (self.formatstring
                        % self.custom[0](W, H, self.scratchDir,
                                         name + self.extension,
                                         **self.custom[1]))
            else:
                return self.formatstring % (W, H, self.scratchDir,
                                            name + self.extension)
        elif self.custom is not None:
            return self.custom[0](figInfo, **self.custom[1])
        else:
            return 'Figure generation for this Formatter is not implemented.'

Figure = {
         'html'  : FigureFormatter(formatstring="<img width='%.2f' height='%.2f' src='%s/%s'>", extension='.png'),
         'latex' : FigureFormatter(formatstring="\\vcenteredhbox{\\includegraphics[width=%.2fin,height=%.2fin,keepaspectratio]{%s/%s}}", extension='.pdf'),
         'text'  : FigureFormatter(custom=(_first_tuple_elem, {})),
         'ppt'   : FigureFormatter()
         }

# Bold formatting
Bold = { 'html'  : Formatter(formatstring='<b>%s</b>', custom=(html, {})),
         'latex' : Formatter(formatstring='\\textbf{%s}', custom=(latex, {})), 
         'text'  : Formatter(formatstring='**%s**'), 
         'ppt'   : Formatter(custom=(ppt, {}))}



def formatList(items, formatters, fmt, scratchDir=None):
    assert(len(items) == len(formatters))
    formatted_items = []
    for item, formatter in zip(items, formatters):
        if formatter is not None:
            # If the formatter requires a scratch directory to do its job, give
            # it.
            if hasattr(formatter, 'scratchDir'):
                formatter.scratchDir = scratchDir
            formatted_items.append( formatter[fmt](item) )
        else:
            formatted_items.append( item )
    return formatted_items

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

from inspect import getargspec as _getargspec


import cgi     as _cgi
import numpy   as _np
import numbers as _numbers
import re      as _re
import os      as _os

class FormatSet():
    formatDict = {} # Static dictionary containing small formatter dictionaries
                    # Ex: { 'Rho' :  { 'html' : ... , 'text' : ... }, ... } (created below)

    def __init__(self, scratchDir, precision):
        self.scratchDir = scratchDir
        self.precision  = precision

    def formatList(self, items, formatters, fmt):
        assert(len(items) == len(formatters))
        formatted_items = []
        for item, formatter in zip(items, formatters):
            if formatter is not None:
                formatter = FormatSet.formatDict[formatter]
                # If the formatter requires a scratch directory  to do its job, give it.
                if hasattr(formatter[fmt], 'scratchDir'):
                    formatter[fmt].scratchDir = self.scratchDir 
                # Likewise with precision
                if hasattr(formatter[fmt], 'precision'):
                    formatter[fmt].precision = self.precision

                formatted_item = formatter[fmt](item)
                if formatted_item is None:
                    raise ValueError("Formatter " + str(type(formatter[fmt]))
                                     + " returned None for item = " + str(item))
                formatted_items.append( formatter[fmt](item) )
            else:
                if item is None:
                    raise ValueError("Unformatted None in formatList")
                formatted_items.append( item )

        return formatted_items

class Formatter():
    '''
    Class for formatting strings to html, latex, powerpoint, or text

    Parameters
    ----------
    stringreplacers : tuples of the form (pattern, replacement)
                   (replacement is a normal string)
                 Ex : [('rho', '&rho;')]
    regexreplace  : A tuple of the form (regex,   replacement)
                   (replacement is formattable string,
                      gets formatted with grouped result of regex matching on label)
                 Ex : ('.*?([0-9]+)$', '_{%s}')

    formatstring : Outer formatting for after both replacements have been made
 
    custom : tuple of a function and additional keyword arguments

    Returns
    -------
    None

    '''

    def __init__(self, stringreplacers=None, regexreplace=None, 
                       formatstring='%s', stringreturn=None, custom=None):
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
        # Return the formatted string of custom formatter
        if self.custom is not None:
            # If keyword args are supplied
            if not callable(self.custom):
                return self.formatstring % self.custom[0](label, **self.custom[1]) 
            # Otherwise..
            else:
                return self.formatstring % self.custom(label)

        # Exit early if string matches stringreturn
        if self.stringreturn is not None and self.stringreturn[0] == label:
            return self.stringreturn[1]
            #Changed by EGN: no need to format string here, but do need to
            # check for equality above

        # Below is the standard formatter case:
        # Replace all occurances of certain substrings
        if self.stringreplacers is not None:            
            for stringreplace in self.stringreplacers:
                label = label.replace(stringreplace[0], stringreplace[1])
        # And then replace all occurances of certain regexes
        if self.regexreplace is not None:
            result = _re.match(self.regexreplace[0], label)
            if result is not None:
                grouped = result.group(1)
                label   = label[0:-len(grouped)] + (self.regexreplace[1] % grouped)
        # Additional formatting, ex $%s$ or <i>%s</i>
        return self.formatstring % label

# A traditional function, so that pickling is possible
def no_format(label):
    return label

# Takes two formatters (a and b), and determines which to use based on a predicate (p)
# (Used in building formatter templates)
class BranchingFormatter():
    def __init__(self, p, a, b):
        self.p = p
        self.a = a
        self.b = b

    def __call__(self, t):
        if self.p(t):
            return self.a(t)
        else:
            return self.b(t)

def has_argname(argname, function):
    return argname in _getargspec(function).args

# Gives precision arguments to formatters
class PrecisionFormatter():
    def __init__(self, custom):
        self.custom    = custom
        self.precision = None

    def __call__(self, label):
        if self.precision is None:
            raise ValueError('Precision was not supplied to PrecisionFormatter')
        
        if not callable(self.custom): # If some keyword arguments were supplied already
            if has_argname('ROUND', self.custom[0]):
                self.custom[1]['ROUND'] = self.precision 
        else:
            if has_argname('ROUND', self.custom):
                self.custom = (self.custom, {'ROUND' : self.precision})
        
        return self.custom[0](label, **self.custom[1])
    

# Formatter class that requires a scratchDirectory from an instance of FormatSet for saving figures to
class FigureFormatter():
    def __init__(self, extension=None, formatstring='%s%s%s%s', custom=None):
        self.extension    = extension
        self.custom       = custom
        self.formatstring = formatstring
        self.scratchDir   = None # Needs to be set to be callable

    def __call__(self, figInfo):
        fig, name, W, H = figInfo
        if self.extension is not None:
            if self.scratchDir is None:
                raise ValueError("Must supply scratch " +
                                 "directory to FigureFormatter")

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
    'html'  : html, 
    'latex' : latex, 
    'text'  : no_format, 
    'ppt'   : ppt }

# 'normal' formatting but round to 2 decimal places
FormatSet.formatDict['Rounded'] = { 
    'html'  : Formatter(custom=(html_value,  {'ROUND' : 2})), # return custom(label, ROUND=2) (Since formatstring is just '%s')
    'latex' : Formatter(custom=(latex_value, {'ROUND' : 2})), 
    'text'  : no_format, 
    'ppt'   : Formatter(custom=(ppt_value,   {'ROUND' : 2}))}

# Similar to the above two formatdicts, but recieves precision during table.render(), which is sent as kwarg to html_value, for example
FormatSet.formatDict['Precision'] = {
    'html'  : PrecisionFormatter(html_value),
    'latex' : PrecisionFormatter(latex_value),
    'text'  : no_format,
    'ppt'   : PrecisionFormatter(ppt_value)}

# 'small' formating - make text smaller
FormatSet.formatDict['Small'] = { 
    'html'  : html, 
    'latex' : Formatter(formatstring='\\small%s', custom=latex), 
    'text'  : no_format, 
    'ppt'   : ppt }

#############################################
# Helper functions for formatting pi-labels #
#############################################

# Predicate for pi_fmt_template
def empty_or_dash(label):
    return str(label) == '--' or str(label) == ''

def pi_fmt_template(b):
    return BranchingFormatter(empty_or_dash, no_format, b) # Pi Formatting shares a common predicate and first branch condition

# Requires an additional predicate
def _pi_text(label):
    if label == '--' or label == '' or not isinstance(label, _numbers.Number):
        return label 
    else:
        return label * _np.pi

# Pi formatters
FormatSet.formatDict['Pi'] = { 
    'html'  : pi_fmt_template(Formatter(custom=html,  formatstring='%s&pi;')), 
    'latex' : pi_fmt_template(Formatter(custom=latex, formatstring='%s$\\pi$')), 
    'text'  : _pi_text,
    'ppt'   : pi_fmt_template(Formatter(custom=ppt,   formatstring='%spi'))}

# Bracket Formatters
FormatSet.formatDict['Brackets'] = { 
    'html'  : Formatter(custom=(html,  {'brackets' : True})), 
    'latex' : Formatter(custom=(latex, {'brackets' : True})), 
    'text'  : no_format, 
    'ppt'   : Formatter(custom=(ppt,   {'brackets' : True}))}

##################################################################################
# 'conversion' formatting: catch all for find/replacing specially formatted text #
##################################################################################

# These two formatters are more complex, justifying individual functions:      

def _fmtCnv_html(x):
    x = x.replace("|"," ") #remove pipes=>newlines, since html wraps table text automatically
    x = x.replace("<STAR>","REPLACEWITHSTARCODE") #b/c cgi.escape would mangle <STAR> marker
    x = _cgi.escape(x).encode("ascii","xmlcharrefreplace") #pylint: disable=deprecated-method
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

FormatSet.formatDict['Conversion'] = { 
    'html'  : _fmtCnv_html, 
    'latex' : _fmtCnv_latex, 
    'text'  : Formatter(stringreplacers=[('<STAR>', '*'), ('|', ' ')]),  
    'ppt'   : Formatter(stringreplacers=[('<STAR>', '*'), ('|', '\n')])}

# Predicate for eb_template
def eb_exists(t):
    return t[1] is not None

# Takes two formatters and decides which to use, based on if the second tuple element (error bar) exists
def eb_template(a, b):
    return BranchingFormatter(eb_exists, a, b)

# Some helper functions for error bar formatters

# Used when the errorbar exists
def _plus_or_minus(t, f=no_format):
    return '%s +/- %s' % (f(t[0]), f(t[1]))

# Used otherwise
def _first_tuple_elem(t, f=no_format):
    return f(t[0])

# Pre-builds formatters that use the above two helper functions, relying on a single formatter f
def eb_fmt_template(f=no_format):
    # If EB exists, return _plus_or_minus of label formatted with f
    # Otherwise, return label[0] formatted with f
    return eb_template(Formatter(custom=(_plus_or_minus,    {'f' : f})), 
                       Formatter(custom=(_first_tuple_elem, {'f' : f}))) 

# These are the same for both ErrorBars and VecErrorBars
# (If eb exists, show plus/minus formatted with html, otherwise show first tuple elem formatted with html)
_html_error_bar = eb_fmt_template(html)
# (If eb exists, show plus/minus formatted with ppt, otherwise show first tuple elem formatted with ppt)
_ppt_error_bar =  eb_fmt_template(ppt)

# See _latex_vec_error_bar (Essentially, this formatter is simpler as a function)
def _latex_error_bar(t):
    return ('$ \\begin{array}{c} %s \\\\ \pm %s \\end{array} $' % 
                          (latex_value(t[0]), latex_value(t[1])))

def _text_error_bar(t):
    return {'value' : t[0], 'errbar' : t[1]}

FormatSet.formatDict['ErrorBars'] = { 
    'html'  : _html_error_bar, 
    'latex' : eb_template(_latex_error_bar,
                         Formatter(custom=(_first_tuple_elem, 
                                          {'f' : latex_value}))), 
    'text'  : _text_error_bar, 
    'ppt'   : _ppt_error_bar}

# _latex_vec_error_bar is better understood as a function, since its class equivalent would be:
# Formatter(formatstring='%s $\pm$ %s', custom=lambda label : tuple(map(latex, label))) 
#  (With a full function instead of a lambda)
def _latex_vec_error_bar(t):
    return '%s $\pm$ %s' % (latex(t[0]), latex(t[1]))

FormatSet.formatDict['VecErrorBars'] = { 
    'html'  : _html_error_bar, 
    'latex' : eb_template(_latex_vec_error_bar,
                          Formatter(custom=(_first_tuple_elem, {'f' : latex}))), 
    'text'  : _text_error_bar, 
    'ppt'   : _ppt_error_bar}


# See _latex_vec_error_bar (This formatter is simpler as a function)
def _latex_pi_error_bar(t):
    if str(t[0]) == '--' or str(t[0]) == '':  return t[0]
    if eb_exists(t):
        return ('$ \\begin{array}{c}(%s \\\\ \\pm %s)\\pi \\end{array} $'
                % (latex(t[0]), latex(t[1])))
    else:
        return '%s$\\pi$' % latex(t[0])


# See eb_fmt_template. The only addition is the formatstring '(%s)&pi;'
def pi_eb_fmt_template(f):
    return eb_template(Formatter(custom=(_plus_or_minus,    {'f' : f}), 
                                 formatstring ='(%s)&pi;'),
                       Formatter(custom=(_first_tuple_elem, {'f' : f}))) 

# 'errorbars with pi' formatting: display (scalar_value +/- error bar) * pi
FormatSet.formatDict['PiErrorBars'] = { 
    'html'  : pi_eb_fmt_template(html), 
    'latex' : _latex_pi_error_bar,
    'text'  : _text_error_bar, 
    'ppt'   : pi_eb_fmt_template(ppt)}

# These could be written with BranchingFormatter, but the only thing in common is (mostly) the predicate: 'if s is not None'
# (If written as classes they would require helper formatters, written as functions, leading to more than four functions)

def _html_gatestring(s):
    return '.'.join(s) if s is not None else ''

def _latex_gatestring(s):
    return '' if s is None else ('$%s$' % '\\cdot'.join([ ('\\mbox{%s}' % gl) for gl in s]))

def _text_gatestring(s):
    return tuple(s) if s is not None else ''

def _ppt_gatestring(s):
    return '.'.join(s) if s is not None else ''

FormatSet.formatDict['GateString'] = {
    'html'  : _html_gatestring, 
    'latex' : _latex_gatestring, 
    'text'  : _text_gatestring, 
    'ppt'   : _ppt_gatestring}

# 'pre' formatting, where the user gives the data in separate formats
def _pre_format(label, formatname=''):
    return label[formatname]

# Factory function
def _pre_fmt_template(formatname):
    return Formatter(custom=(_pre_format, {'formatname' : formatname}))

FormatSet.formatDict['Pre'] = { 
    'html'   : _pre_fmt_template('html'), # As opposed to: Formatter(custom=(_pre_format, {'formatname' : html}))
    'latex'  : _pre_fmt_template('latex'),#   or def _pre_html(label): return label['html']
    'text'   : _pre_fmt_template('text'), 
    'ppt'    : _pre_fmt_template('ppt')}


FormatSet.formatDict['Figure'] = {
    'html'  : FigureFormatter(formatstring="<img width='%.2f' height='%.2f' src='%s/%s'>", 
                              extension='.png'),
    'latex' : FigureFormatter(formatstring="\\vcenteredhbox{\\includegraphics[width=%.2fin,height=%.2fin,keepaspectratio]{%s/%s}}", 
                              extension='.pdf'),
    'text'  : FigureFormatter(custom=_first_tuple_elem),
    'ppt'   : FigureFormatter()} # Not Implemented

# Bold formatting
FormatSet.formatDict['Bold'] = { 
    'html'  : Formatter(formatstring='<b>%s</b>', custom=html),
    'latex' : Formatter(formatstring='\\textbf{%s}', custom=latex), 
    'text'  : Formatter(formatstring='**%s**'), 
    'ppt'   : ppt} # No bold in ppt?



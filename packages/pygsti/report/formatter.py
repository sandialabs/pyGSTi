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

def _give_specs(formatter, specs):
    # If the formatter requires a setting to do its job, give the setting
    if hasattr(formatter, 'specs'):
        print('Giving specs to %s' % formatterName)
        for spec in formatter.specs:
            if spec not in specs or specs[spec] is None:
                # This should make the ValueError thrown by
                #   ParameterizedFormatter redundant
                # This also means that even though specs will be set after
                # the first call to table.render(),
                # they will need to be provided again in subsequent calls
                raise ValueError(
                        ('The spec %s was not supplied to ' % spec) +
                        ('FormatSet, but is needed by %s' % formatterName))
            formatter.specs[spec] = specs[spec]
            print('The spec %s was given to the formatter %s' %
                                         (spec, formatterName))

class FormatSet():
    formatDict = {} # Static dictionary containing small formatter dictionaries
                    # Ex: { 'Rho' :  { 'html' : ... , 'text' : ... }, ... }
                    # (created below)

    def __init__(self, specs):
        # Specs is a dictionary of the form { 'setting'(kwarg) : value }
        # Ex: { 'precision' : 6, 'polarprecision' : 3 }
        # -> given to ParameterizedFormatters that need them
        self.specs = specs

    def formatList(self, items, formatterNames, fmt):
        assert(len(items) == len(formatterNames))
        formatted_items = []

        for item, formatterName in zip(items, formatterNames):
            if formatterName is not None:
                formatter = FormatSet.formatDict[formatterName]
                _give_specs(formatter[fmt], self.specs)
                # Format the item once the formatter has been completely built
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

class Formatter(object):
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


# Helper function to ParameterizedFormatter
def has_argname(argname, function):
    return argname in _getargspec(function).args

# Gives arguments to formatters
class ParameterizedFormatter(object):
    def __init__(self, custom, neededSpecs, defaults={}, formatstring='%s'):
        self.custom       = custom
        self.specs        = { neededSpec : None for neededSpec in neededSpecs }
        self.defaults     = defaults
        self.formatstring = formatstring

    def __call__(self, label):
        self.defaults.update(self.specs)
        # Supply arguments to the custom formatter (if it needs them)
        for argname in self.defaults:
            if not callable(self.custom): # If some keyword arguments were supplied already
                if has_argname(argname, self.custom[0]):             # 'if it needs them'
                    # update the argument in custom's existing keyword dictionary
                    self.custom[1][argname] = self.defaults[argname]
            else:
                if has_argname(argname, self.custom): # If custom is a lone callable (not a tuple)
                # Create keyword dictionary for custom, modifiying it to be a tuple
                #   (function, kwargs)
                    self.custom = (self.custom, {argname : self.defaults[argname]})
        return self.formatstring % self.custom[0](label, **self.custom[1])

# Gives precision arguments to formatters
class PrecisionFormatter(ParameterizedFormatter):
    def __init__(self, custom, defaults={}, formatstring='%s'):
        super(PrecisionFormatter, self).__init__(custom, ['precision', 'polarprecision'], 
                                                 defaults, formatstring)

# Formatter class that requires a scratchDirectory from an instance of FormatSet for saving figures to
class FigureFormatter(ParameterizedFormatter):
    def __init__(self, extension=None, formatstring='%s%s%s%s', custom=None):
        super(FigureFormatter, self).__init__(custom, ['scratchDir'])
        self.extension    = extension
        self.formatstring = formatstring

    # Override call method of Parameterized formatter
    def __call__(self, figInfo):
        fig, name, W, H = figInfo
        if self.extension is not None:

            fig.save_to(_os.path.join(self.specs['scratchDir'], name + self.extension))
            if self.custom is not None:
                return (self.formatstring
                        % self.custom[0](W, H, self.specs['scratchDir'],
                                         name + self.extension,
                                         **self.custom[1]))
            else:
                return self.formatstring % (W, H, self.specs['scratchDir'],
                                            name + self.extension)

        elif self.custom is not None:
            return self.custom[0](figInfo, **self.custom[1])
        else:
            return 'Figure generation for this Formatter is not implemented.'

# Takes two formatters (a and b), and determines which to use based on a predicate
# (Used in building formatter templates)
class BranchingFormatter(object):
    def __init__(self, predicate, a, b):
        self.predicate = predicate
        self.a = a
        self.b = b

        # So that a branching formatter can hold parameterized formatters
        self.specs = {}
        if hasattr(a, 'specs'):
            self.specs.update(a.specs)
        if hasattr(b, 'specs'):
            self.specs.update(b.specs)

    def __call__(self, label):
        if self.predicate(label):
            _give_specs(self.a, self.specs)
            return self.a(label)
        else:
            _give_specs(self.a, self.specs)
            return self.b(label)

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

#DEPRECATED?
# 'normal' formatting but round to 2 decimal places
FormatSet.formatDict['Rounded'] = {
    'html'  : ParameterizedFormatter(html_value,  ['polarprecision'], {'precision' : 2}),
    'latex' : ParameterizedFormatter(latex_value, ['polarprecision'], {'precision' : 2}),
    'text'  : no_format,
    'ppt'   : ParameterizedFormatter(ppt_value,   ['polarprecision'], {'precision' : 2})}

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

#############################################
# Helper functions for formatting pi-labels #
#############################################

# Predicate for pi_fmt_template
def empty_or_dash(label):
    return str(label) == '--' or str(label) == ''

def pi_fmt_template(b):
    # Pi Formatting shares a common predicate and first branch condition
    return BranchingFormatter(empty_or_dash, no_format, b)

# Requires an additional predicate
def _pi_text(label):
    if label == '--' or label == '' or not isinstance(label, _numbers.Number):
        return label
    else:
        return label * _np.pi

# Pi formatters
FormatSet.formatDict['Pi'] = {
    'html'  : pi_fmt_template(PrecisionFormatter(html,  formatstring='%s&pi;')),
    'latex' : pi_fmt_template(PrecisionFormatter(latex, formatstring='%s$\\pi$')),
    'text'  : _pi_text,
    'ppt'   : pi_fmt_template(PrecisionFormatter(ppt,   formatstring='%spi'))}

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

_eb_exists = lambda t : t[1] is not None

class EBFormatter():
    def __init__(self, f, formatstringA='%s +/- %s', formatstringB='%s'):
        self.f = f
        if hasattr(f, 'specs'):
            self.specs = f.specs
        self.formatstringA = formatstringA
        self.formatstringB = formatstringB

    def __call__(t):
        if hasattr(self.f, 'specs'):
            _give_specs(self.f, self.specs)
        if _eb_exists(t):
            return self.formatstringA % (f(t[0]), f(t[1]))
        else:
            return self.formatstringB % f(t[0])

EB_html  = EBFormatter(PrecisionFormatter(html))
EB_latex = EBFormatter(PrecisionFormatter(latex_value), 
                       '$ \\begin{array}{c} %s \\\\ \pm %s \\end{array} $')
EB_text  = lambda t : {'value' : t[0], 'errbar' : t[1]}
EB_ppt   = EBFormatter(PrecisionFormatter(ppt))

FormatSet.formatDict['ErrorBars'] = { 
    'html'  : EB_html, 
    'latex' : EB_latex, 
    'text'  : EB_text, 
    'ppt'   : EB_ppt }

VEB_latex = EBFormatter(PrecisionFormatter(latex), '%s $\pm$ %s')

FormatSet.formatDict['VecErrorBars'] = {
    'html'  : EB_html,
    'latex' : VEB_latex,
    'text'  : EB_text,
    'ppt'   : EB_ppt}

class PiEBFormatter(EBFormatter):
    def __call__(self, t):
        if str(t[0]) == '--' or str(t[0]) == '':  return t[0]
        else:
            return super(PiEBFormatter, self).__call__(t)

PiEB_latex = PiEBFormatter(PrecisionFormatter(latex), 
                           '$ \\begin{array}{c}(%s \\\\ \\pm %s)\\pi \\end{array} $',
                           '%s$\\pi$')
def pi_eb_template(f):
    return EBFormatter(PrecisionFormatter(html), '(%s +/- %s)&pi')

# 'errorbars with pi' formatting: display (scalar_value +/- error bar) * pi
FormatSet.formatDict['PiErrorBars'] = {
    'html'  : pi_eb_template(html),
    'latex' : PiEB_latex,
    'text'  : EB_text,
    'ppt'   : pi_eb_template(ppt)}

# These could be written with BranchingFormatter,
# but the only thing in common is (mostly) the predicate: 'if s is not None'

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
    'html'   : _pre_fmt_template('html'),
    'latex'  : _pre_fmt_template('latex'),
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
    'html'  : PrecisionFormatter(html, formatstring='<b>%s</b>'),
    'latex' : PrecisionFormatter(latex, formatstring='\\textbf{%s}'),
    'text'  : Formatter(formatstring='**%s**'),
    'ppt'   : PrecisionFormatter(ppt)} # No bold in ppt?

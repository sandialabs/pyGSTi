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
    '''
    Pass parameters down to a formatter

    Parameters
    --------
    formatter : callable, takes arguments

    specs : dictionary of argnames : values

    Returns
    ------
    None

    Raises
    ------
    ValueError : If a needed spec is not supplied.
    '''
    # If the formatter requires a setting to do its job, give the setting
    if hasattr(formatter, 'specs'):
        for spec in formatter.specs:
            if spec not in specs or specs[spec] is None:
                '''
                This should make the ValueError thrown by
                  _ParameterizedFormatter redundant
                This also means that even though specs will be set after
                the first call to table.render(),
                they will need to be provided again in subsequent calls
                '''
                raise ValueError(
                        ('The spec %s was not supplied to ' % spec) +
                        ('FormatSet, but is needed by an active formatter'))
            formatter.specs[spec] = specs[spec]

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
                       given to _ParameterizedFormatters that need them

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
                _give_specs(formatter[fmt], self.specs) # Parameters aren't sent until render call
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

class _Formatter(object):
    '''
    Callable class that can replace a formatter function.

    Only defines __init__ and __call__ methods
    '''

    def __init__(self, stringreplacers=None, regexreplace=None,
                       formatstring='%s', stringreturn=None):
        '''
        Parameters
        ----------
        stringreplacers : tuples of the form (pattern, replacement) (optional)
                       (replacement is a normal string)
                     Ex : [('rho', '&rho;')]
        regexreplace  : A tuple of the form (regex,   replacement) (optional)
                       (replacement is formattable string,
                          gets formatted with grouped result of regex matching on label)
                     Ex : ('.*?([0-9]+)$', '_{%s}')

        formatstring : string (optional) Outer formatting for after both replacements have been made

        stringreturn : tuple (string, string) Replaces first string with second and
                         returns early if the first string exists,
                         otherwise does nothing
        '''
        self.stringreplacers = stringreplacers
        self.regexreplace    = regexreplace
        self.formatstring    = formatstring
        self.stringreturn    = stringreturn

    def __call__(self, label):
        '''
        Formatting function template

        Parameters
        --------
        label : string, the label to be formatted!

        Returns
        --------
        formatted label : string
        '''
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


class _TupleFormatter(object):
    '''
    Callable class that can replace a formatter function, similar to
    _Formatter, but expects a tuple as input instead of a single string.

    Only defines __init__ and __call__ methods
    '''

    def __init__(self, label_formatter=None, formatstring='{l0}'):
        '''
        Parameters
        ----------
        label_formatter : callable or None
            Another formatter that is used to format the "label",
            defined to be the first element of the tuple this 
            formatter is called with.

        formatstring : string (optional)
            Outer formatting for after label_formatter has been applied.
        '''
        self.formatstring    = formatstring
        self.label_formatter = label_formatter

    def __call__(self, label_tuple):
        '''
        Formatting function template

        Parameters
        --------
        label_tuple : tuple
            The label, followed by other paramters, to be formatted.

        Returns
        --------
        formatted label : string
        '''
        label = label_tuple[0] #process first element of tuple as _Formatter

        if self.label_formatter is not None:
            label = self.label_formatter(label)

        # Formatting according to format string
        format_dict = { 'l0': label }
        format_dict.update( { 'l%d' % i: label_tuple[i] 
                              for i in range(1,len(label_tuple)) })
        return self.formatstring.format(**format_dict)

def _no_format(label):
    return label

# Helper function to _ParameterizedFormatter
def _has_argname(argname, function):
    return argname in _getargspec(function).args

# Gives arguments to formatters
class _ParameterizedFormatter(object):
    '''
    Class that will pass down specs (arguments) to functions that need them

    For example, a precision-parameterized latex formatter without the help of the _PrecisionFormatter might look like this:
    formatter = _ParameterizedFormatter(latex, ['precision', 'polarprecision', 'sciprecision'])
    Which, when used with a FormatSet, would have arguments to table.render() passed down to the latex() function
    '''
    def __init__(self, custom, neededSpecs, defaults={}, formatstring='%s'):
        self.custom       = custom
        self.specs        = { neededSpec : None for neededSpec in neededSpecs }
        self.defaults     = defaults
        self.formatstring = formatstring

    def __call__(self, label):
        # If the formatter is being called, we know that the needed specs have successfully been supplied by FormatSet
        self.defaults.update(self.specs)
        # Supply arguments to the custom formatter (if it needs them)
        for argname in self.defaults:
            if not callable(self.custom): # If some keyword arguments were supplied already
                if _has_argname(argname, self.custom[0]):             # 'if it needs them'
                    # update the argument in custom's existing keyword dictionary
                    self.custom[1][argname] = self.defaults[argname]
            else:
                if _has_argname(argname, self.custom): # If custom is a lone callable (not a tuple)
                # Create keyword dictionary for custom, modifiying it to be a tuple
                #   (function, kwargs)
                    self.custom = (self.custom, {argname : self.defaults[argname]})
        return self.formatstring % self.custom[0](label, **self.custom[1])

# Gives precision arguments to formatters
class _PrecisionFormatter(_ParameterizedFormatter):
    '''Helper class for Precision Formatting
    Takes a custom function and a dictionary of keyword arguments:
    So, something like _PrecisionFormatter(latex) would pass precision arguments to
      the latex formatter function during table.render() calls
    '''
    def __init__(self, custom, defaults={}, formatstring='%s'):
        super(_PrecisionFormatter, self).__init__(custom, ['precision', 'polarprecision','sciprecision'],
                                                 defaults, formatstring)


# Formatter class that requires a scratchDirectory from an instance of FormatSet for saving figures to
class _FigureFormatter(_ParameterizedFormatter):
    '''
    Helper class that utilizes a scratchDir variable to render figures
    '''
    def __init__(self, extension='.png', formatstring='%s%s%s%s'):
        '''
        Parameters
        ---------
        extension : string, optional. extension of the figure's image
        formatstring : string, optional. Normally formatted with W, H, scratchDir, filename
        '''
        super(_FigureFormatter, self).__init__(_no_format, ['scratchDir'])
        self.extension    = extension
        self.formatstring = formatstring

    # Override call method of Parameterized formatter
    def __call__(self, figInfo):
        fig, name, W, H = figInfo
        scratchDir = self.specs['scratchDir']
        if len(scratchDir) > 0: #empty scratchDir signals not to output figure
            fig.save_to(_os.path.join(scratchDir, name + self.extension))
        return self.formatstring % (W, H, scratchDir,
                                    name + self.extension)

# Takes two formatters (a and b), and determines which to use based on a predicate
# (Used in building formatter templates)
class _BranchingFormatter(object):
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
    'html'  : _Formatter(stringreplacers=[('rho', '&rho;')],
                         regexreplace=('.*?([0-9]+)$', '<sub>%s</sub>')),
    'latex' : _Formatter(stringreplacers=[('rho', '\\rho')],
                         regexreplace=('.*?([0-9]+)$', '_{%s}'), formatstring='$%s$'),
    'text'  : _no_format,
    'ppt'   : _no_format}

# 'E' (POVM) effect formatting
FormatSet.formatDict['Effect'] = {
    # If label == 'remainder', return E sub C
    # Otherwise, match regex and replace with subscript
    'html'  : _Formatter(stringreturn=('remainder', 'E<sub>C</sub>'),
                         regexreplace=('.*?([0-9]+)$', '<sub>%s</sub>')),
    'latex' : _Formatter(stringreturn=('remainder', '$E_C$'),
                         regexreplace=('.*?([0-9]+)$', '_{%s}'), formatstring='$%s$'),
    'text'  : _no_format,
    'ppt'   : _no_format}

# Normal replacements
FormatSet.formatDict['Normal'] = {
    'html'  : _PrecisionFormatter(html),
    'latex' : _PrecisionFormatter(latex),
    'text'  : _no_format,
    'ppt'   : _PrecisionFormatter(ppt) }

# 'normal' formatting but round to 2 decimal places regardless of what is passed in to table.render()
FormatSet.formatDict['Rounded'] = {
    'html'  : _ParameterizedFormatter(html_value,  ['polarprecision'], {'precision' : 2, 'sciprecision': 0}),
    'latex' : _ParameterizedFormatter(latex_value, ['polarprecision'], {'precision' : 2, 'sciprecision': 0}),
    'text'  : _no_format,
    'ppt'   : _ParameterizedFormatter(ppt_value,   ['polarprecision'], {'precision' : 2, 'sciprecision': 0})}

# Similar to the above two formatdicts,
# but recieves precision during table.render(), which is sent as kwarg to html_value, for example
FormatSet.formatDict['Precision'] = {
    'html'  : _PrecisionFormatter(html_value),
    'latex' : _PrecisionFormatter(latex_value),
    'text'  : _no_format,
    'ppt'   : _PrecisionFormatter(ppt_value)}

# 'small' formating - make text smaller
FormatSet.formatDict['Small'] = {
    'html'  : _PrecisionFormatter(html),
    'latex' : _PrecisionFormatter(latex, formatstring='\\small%s'),
    'text'  : _no_format,
    'ppt'   : _PrecisionFormatter(ppt)}

#############################################
# Helper functions for formatting pi-labels #
#############################################

def _pi_template(b):
    # Pi Formatting shares a common predicate and first branch condition
    return _BranchingFormatter(lambda label : str(label) == '--' or str(label) == '',
                              _no_format, b)

# Requires an additional predicate
def _pi_text(label):
    if label == '--' or label == '' or not isinstance(label, _numbers.Number):
        return label
    else:
        return label * _np.pi

# Pi formatters
FormatSet.formatDict['Pi'] = {
    'html'  : _pi_template(_PrecisionFormatter(html,  formatstring='%s&pi;')),
    'latex' : _pi_template(_PrecisionFormatter(latex, formatstring='%s$\\pi$')),
    'text'  : _pi_text,
    'ppt'   : _pi_template(_PrecisionFormatter(ppt,   formatstring='%spi'))}

# Bracket Formatters
FormatSet.formatDict['Brackets'] = {
    'html'  : _PrecisionFormatter(html,  defaults={'brackets' : True}),
    'latex' : _PrecisionFormatter(latex, defaults={'brackets' : True}),
    'text'  : _no_format,
    'ppt'   : _PrecisionFormatter(ppt,   defaults={'brackets' : True})}

##################################################################################
# 'conversion' formatting: catch all for find/replacing specially formatted text #
##################################################################################

# These two formatters are more complex, justifying individual functions:

def _fmtCnv_html(x):
    x = x.replace("\\", "&#92"); #backslash
    x = x.replace("|"," ") #remove pipes=>newlines, since html wraps table text automatically
    x = x.replace("<STAR>","REPLACEWITHSTARCODE") #b/c cgi.escape would mangle <STAR> marker
    x = _cgi.escape(x).encode("ascii","xmlcharrefreplace")
    x = x.replace(b"REPLACEWITHSTARCODE", b'&#9733;') #replace new marker with HTML code
    return x

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
    'text'  : _Formatter(stringreplacers=[('<STAR>', '*'), ('|', ' ')]),
    'ppt'   : _Formatter(stringreplacers=[('<STAR>', '*'), ('|', '\n')])}

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
            _give_specs(self.f, self.specs)
        if _eb_exists(t):
            return self.formatstringA % (self.f(t[0]), self.f(t[1]))
        else:
            return self.formatstringB % self.f(t[0])

_EB_html  = _EBFormatter(_PrecisionFormatter(html))
_EB_latex = _EBFormatter(_PrecisionFormatter(latex_value),
                       '$ \\begin{array}{c} %s \\\\ \pm %s \\end{array} $')
_EB_text  = lambda t : {'value' : t[0], 'errbar' : t[1]}
_EB_ppt   = _EBFormatter(_PrecisionFormatter(ppt))

FormatSet.formatDict['ErrorBars'] = {
    'html'  : _EB_html,
    'latex' : _EB_latex,
    'text'  : _EB_text,
    'ppt'   : _EB_ppt }

_VEB_latex = _EBFormatter(_PrecisionFormatter(latex), '%s $\pm$ %s')

FormatSet.formatDict['VecErrorBars'] = {
    'html'  : _EB_html,
    'latex' : _VEB_latex,
    'text'  : _EB_text,
    'ppt'   : _EB_ppt}

class _PiEBFormatter(_EBFormatter):
    def __call__(self, t):
        if str(t[0]) == '--' or str(t[0]) == '':  return t[0]
        else:
            return super(_PiEBFormatter, self).__call__(t)

_PiEB_latex = _PiEBFormatter(_PrecisionFormatter(latex),
                           '$ \\begin{array}{c}(%s \\\\ \\pm %s)\\pi \\end{array} $',
                           '%s$\\pi$')
def _pi_eb_template(f):
    return _EBFormatter(_PrecisionFormatter(html), '(%s +/- %s)&pi')

# 'errorbars with pi' formatting: display (scalar_value +/- error bar) * pi
FormatSet.formatDict['PiErrorBars'] = {
    'html'  : _pi_eb_template(html),
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
    'html'  : _FigureFormatter(formatstring="<img width='%.2f' height='%.2f' src='%s/%s'>",
                               extension='.png'),
    'latex' : _FigureFormatter(formatstring="\\vcenteredhbox{\\includegraphics[width=%.2fin,height=%.2fin,keepaspectratio]{%s/%s}}",
                               extension='.pdf'),
    'text'  : lambda figInfo : figInfo[0],
    'ppt'   : lambda figInfo : 'Figure formatting not implemented for ppt'} # Not Implemented

# Bold formatting
FormatSet.formatDict['Bold'] = {
    'html'  : _PrecisionFormatter(html, formatstring='<b>%s</b>'),
    'latex' : _PrecisionFormatter(latex, formatstring='\\textbf{%s}'),
    'text'  : _Formatter(formatstring='**%s**'),
    'ppt'   : _PrecisionFormatter(ppt)} # No bold in ppt?


#Multi-row and multi-column formatting (with "Conversion" type inner formatting)
FormatSet.formatDict['MultiRow'] = {
    'html'  : _TupleFormatter(_fmtCnv_html),
    'latex' : _TupleFormatter(_fmtCnv_latex, formatstring='\\multirow{{{l1}}}{{*}}{{{l0}}}'),
    'text'  : _TupleFormatter(_Formatter(stringreplacers=[('<STAR>', '*'), ('|', ' ')])),
    'ppt'   : _TupleFormatter(_Formatter(stringreplacers=[('<STAR>', '*'), ('|', '\n')]))}


def _empty_str(l): return ""
FormatSet.formatDict['SpannedRow'] = {
    'html'  : _fmtCnv_html,
    'latex' : _empty_str,
    'text'  : _Formatter(stringreplacers=[('<STAR>', '*'), ('|', ' ')]),
    'ppt'   : _Formatter(stringreplacers=[('<STAR>', '*'), ('|', '\n')])}

def _repeat_no_format(label_tuple): 
    label, reps = label_tuple
    return ["%s" % label]*reps

FormatSet.formatDict['MultiCol'] = {
    'html'  : _repeat_no_format,
    'latex' : _TupleFormatter(_fmtCnv_latex, formatstring='\\multicolumn{{{l1}}}{{c|}}{{{l0}}}'),
    'text'  : _repeat_no_format,
    'ppt'   : _repeat_no_format}


#Special formatting for Hamiltonian and Stochastic gateset types
FormatSet.formatDict['GatesetType'] = {
    'html'  : _no_format,
    'latex' : _Formatter(stringreplacers=[('H','$\\mathcal{H}$'),('S','$\\mathcal{S}$')]),
    'text'  : _no_format,
    'ppt'   : _no_format}

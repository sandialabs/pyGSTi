"""
Defines the Formatter class
"""
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

import re as _re
from copy import deepcopy

from ..objects.reportableqty import ReportableQty as _ReportableQty


class Formatter(object):
    """
    Class defining the formatting rules for an object

    Once created, is used like a function with the signature: item, specs -> string
    See __call__ method for details

    __call__ could be renamed to render() for compatibility with table.render(), row.render(), etc..
    However, using __call__ allows for the user to drop in custom functions in place of Formatter objects,
    which is useful (i.e. in creating figure formatters)

    Parameters
    ----------
    custom : function, optional
        A custom-formatting function that has signature `custom(item, specs)` and returns
        `item` formatted as a string.

    stringreplacers : tuple, optional
        A tuple of tuples of the form (pattern, replacement) where replacement is a normal
        string.  Ex : [('rho', '&rho;')]

    regexreplace  : tuple, optional
        A tuple of the form (regex,   replacement) where replacement is formattable string,
        and gets formatted with grouped result of regex matching on item) Ex : ('.*?([0-9]+)$', '_{%s}')

    formatstring : str, optional
        Outer formatting for after both replacements have been made

    ebstring : str, optional
        Format string used if the item being formatted has attached error bars.

    nmebstring : str, optional
        Alternate format string to use for non-Markovian error bars.

    stringreturn : tuple
        A `(string, string)` tuple that creates a formatting rules where the the second string
        is used if a label is equal to the first.

    defaults : dictionary (string, any)
        overriden values to the dictionary passed in during formatted.
        ie for rounded formatters, which override the precision key to be set to two
    """

    def __init__(self,
                 custom=None,
                 stringreplacers=None,
                 regexreplace=None,
                 formatstring='%s',
                 ebstring='%s +/- %s',
                 nmebstring=None,
                 stringreturn=None,
                 defaults=None):
        '''
        Create a Formatter object by supplying formatting rules to be applied

        Parameters
        ----------
        stringreplacers : tuples of the form (pattern, replacement) (optional)
                       (replacement is a normal string)
                     Ex : [('rho', '&rho;')]
        regexreplace  : A tuple of the form (regex,   replacement) (optional)
                       (replacement is formattable string,
                          gets formatted with grouped result of regex matching on item)
                     Ex : ('.*?([0-9]+)$', '_{%s}')

        formatstring : string (optional) Outer formatting for after both replacements have been made

        ebstring : string (optional) formatstring used if the item being formatted has attached error bars

        stringreturn : tuple (string, string)
            return the second string if the label is equal to the first

        defaults : dictionary (string, any)
            overriden values to the dictionary passed in during formatted.
            ie for rounded formatters, which override the precision key to be set to two
        '''
        self.custom = custom
        self.stringreplacers = stringreplacers
        self.stringreturn = stringreturn
        self.regexreplace = regexreplace
        self.formatstring = formatstring
        self.ebstring = ebstring
        if nmebstring is None:
            nmebstring = ebstring
        self.nmebstring = nmebstring

        if defaults is None:
            self.defaults = dict()
        else:
            self.defaults = defaults

    def __call__(self, item, specs):
        '''
        Formatting function template

        Parameters
        --------
        item : string, the item to be formatted!

        specs : dictionary
            dictionary of options to be sent to the formatter and custom functions

        Returns
        --------
        formatted item : string
        '''
        specs = deepcopy(specs)  # Modifying other dictionaries would be rude
        specs.update(self.defaults)

        if isinstance(item, _ReportableQty):
            # If values are replaced with dashes or empty, leave them be
            s = str(item.value)
            if s == '--' or s == '':
                return s
            return item.render_with(self, specs, self.ebstring, self.nmebstring)
        # item is not ReportableQty, and custom is defined
        # avoids calling custom twice on ReportableQty objects
        elif self.custom is not None:
            item = self.custom(item, specs)

        item = str(item)
        # Avoids replacing commonly used string names with formatting
        if self.stringreturn is not None and item == self.stringreturn[0]:
            return self.stringreturn[1]

        # Below is the standard formatter case:
        # Replace all occurrences of certain substrings
        if self.stringreplacers is not None:
            for stringreplace in self.stringreplacers:
                item = item.replace(stringreplace[0], stringreplace[1])
        # And then replace all occurrences of certain regexes
        if self.regexreplace is not None:
            result = _re.search(self.regexreplace[0], item)
            if result is not None:  # Note: specific to 1-group regexps currently...
                s, e = result.span()  # same as result.start(), result.end()
                item = item[0:s] + (self.regexreplace[1] % result.group(1)) + item[e:]
        formatstring = specs['formatstring'] if 'formatstring' in specs else self.formatstring
        # Additional formatting, ex $%s$ or <i>%s</i>
        return formatstring % item

    def variant(self, **kwargs):
        """
        Create a Formatter object from an existing formatter object, tweaking it slightly.

        Parameters
        ----------
        kwargs : various
            Arguments to Formatter.__init__().

        Returns
        -------
        Formatter
        """
        ret = deepcopy(self)
        for k, v in kwargs.items():
            if k not in ret.__dict__:
                raise ValueError('Invalid argument to Formatter.variant: {}={}\n{}'.format(
                    k, v, 'Valid arguments are: {}'.format(list(ret.__dict__.keys()))
                ))
            if k == 'ebstring' and ('nmebstring' not in kwargs) \
               and ret.ebstring == ret.nmebstring:
                ret.__dict__[k] = v
                ret.__dict__['nmebstring'] = v
            elif k == 'defaults':
                ret.__dict__['defaults'].update(v)
            else:
                ret.__dict__[k] = v
        return ret

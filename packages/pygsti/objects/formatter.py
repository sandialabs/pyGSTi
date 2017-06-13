#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************

import re as _re
from copy import deepcopy

class Formatter(object):
    '''
    Class defining the formatting rules for an object
    '''

    def __init__(self, 
            custom=None, 
            stringreplacers=None, 
            regexreplace=None,
            formatstring='%s',
            ebstring='%s +/- %s',
            stringreturn=None,
            defaults=None):
        '''
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

        stringreturn : tuple (string, string) Replaces first string with second and
                         returns early if the first string exists,
                         otherwise does nothing
        '''
        self.custom          = custom
        self.stringreplacers = stringreplacers
        self.regexreplace    = regexreplace
        self.formatstring    = formatstring
        self.ebstring        = ebstring
        self.stringreturn    = stringreturn
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

        Returns
        --------
        formatted item : string
        '''
        if hasattr(item, 'has_eb'): # Check isinstance(item, ReportableQTY without importing ReportableQty)
            # If values are replaced with dashes or empty, leave them be
            s = str(item.get_value())
            if s == '--' or s == '':
                return s
            # Format with ebstring if error bars present
            if item.has_eb():
                return self.ebstring % (self(item.get_value(), specs), self(item.get_err_bar(), specs))
            # Otherwise use value only
            else:
                return self(item.get_value(), specs)

        if len(self.defaults) > 0:
            specs = deepcopy(specs) # Modifying other dictionaries would be rude
            specs.update(self.defaults)

        if self.custom is not None:
            item = self.custom(item, specs)

        item = str(item)
        # Exit early if string matches stringreturn
        if self.stringreturn is not None and self.stringreturn[0] == item:
            return self.stringreturn[1]

        # Below is the standard formatter case:
        # Replace all occurances of certain substrings
        if self.stringreplacers is not None:
            for stringreplace in self.stringreplacers:
                item = item.replace(stringreplace[0], stringreplace[1])
        # And then replace all occurances of certain regexes
        if self.regexreplace is not None:
            result = _re.match(self.regexreplace[0], item)
            if result is not None:
                grouped = result.group(1)
                item   = item[0:-len(grouped)] + (self.regexreplace[1] % grouped)
        # Additional formatting, ex ${}$ or <i>{}</i>
        return self.formatstring % item

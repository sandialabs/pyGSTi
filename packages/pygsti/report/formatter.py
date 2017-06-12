import re as _re

class Formatter(object):
    '''
    Class defining the formatting rules for an object
    '''

    def __init__(self, 
            custom=None, 
            stringreplacers=None, 
            regexreplace=None,
            formatstring='{}',
            stringreturn=None):
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
        self.stringreturn    = stringreturn

    def render(self, item, specs):
        '''
        Formatting function template

        Parameters
        --------
        item : string, the item to be formatted!

        Returns
        --------
        formatted item : string
        '''
        if custom is not None:
            item = custom(item, specs)
        item = str(item)
        # Exit early if string matches stringreturn
        if self.stringreturn is not None and self.stringreturn[0] == item:
            return self.stringreturn[1]
            #Changed by EGN: no need to format string here, but do need to
            # check for equality above

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
        return self.formatstring.format(item)


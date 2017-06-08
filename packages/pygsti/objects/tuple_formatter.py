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

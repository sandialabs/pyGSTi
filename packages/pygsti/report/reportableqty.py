class ReportableQty(object):
    """
    Encapsulates a computed quantity and possibly its error bars,
    primarily for use in reports.
    """

    def __init__(self, value, errbar=None, nonMarkovianEBs=False):
        """
        Initialize a new ReportableQty object, which
        is essentially a container for a value and error bars.

        Parameters
        ----------
        value : anything
           The value to store

        errbar : anything
           The error bar(s) to store

        nonMarkovianEBs : bool
            boolean indicating if non markovian error bars should be used
        """
        self.value   = value
        self.errbar  = errbar

        self.nonMarkovianEBs = nonMarkovianEBs

    def __str__(self):
        def f(x,y): 
            return (str(x) + " +/- " + str(y)) if y else str(x)
        return self.render_with(f)

    def __repr__(self):
        return 'ReportableQty({})'.format(str(self))

    def __getattr__(self, attr):
        return getattr(self.value, attr)

    @staticmethod
    def from_val(value, nonMarkovianEBs=False):
        '''
        Convert Table values into ReportableQtys or leave them be if they are well-formed types
        Well-formed types include:
            strings
            figures
            ReportableQtys
        A tuple will be converted to a ReportableQty 
          holding the first field as a value and second field as an error bar
        Anything else will be converted to a ReportableQty with no error bars
        '''
        if isinstance(value, tuple):
            assert len(value) == 2, 'Tuple does not have eb field ' + \
                                    'or has too many fields: len = {}'.format(
                                            len(value))
            return ReportableQty(value[0], value[1], nonMarkovianEBs=nonMarkovianEBs)
        else:
            return ReportableQty(value, nonMarkovianEBs=nonMarkovianEBs)

    def has_eb(self):
        return self.errbar is not None

    def get_value(self):
        """
        Returns the quantity's value
        """
        return self.value

    def get_err_bar(self):
        """
        Returns the quantity's error bar(s)
        """
        return self.errbar

    def get_value_and_err_bar(self):
        """
        Returns the quantity's value and error bar(s)
        """
        return self.value, self.errbar

    def render_with(self, f, specs=None, ebstring='%s +/- %s', nmebstring=None):
        if nmebstring is None:
            nmebstring = ebstring
        if specs is None:
            specs = dict()
        if self.errbar is not None:
            specs['formatstring'] = '%s' # Don't recursively apply format strings to inside error bars
            if self.nonMarkovianEBs:
                rendered = nmebstring % (f(self.value,  specs), 
                                         f(self.errbar, specs))
            else:
                rendered = ebstring % (f(self.value,  specs), 
                                       f(self.errbar, specs))
        else:
            rendered = f(self.value, specs)
        return rendered

class ReportableQty(object):
    """
    Encapsulates a computed quantity and possibly its error bars,
    primarily for use in reports.
    """

    def __init__(self, value, errbar=None):
        """
        Initialize a new ReportableQty object, which
        is essentially a container for a value and error bars.

        Parameters
        ----------
        value : anything
           The value to store

        errbar : anything
           The error bar(s) to store
        """
        self.value   = value
        self.errbar  = errbar

    def __str__(self):
        return self.render_with(str)

    def __getattr__(self, attr):
        return getattr(self.value, attr)

    @staticmethod
    def from_val(value):
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
            assert len(value) == 2, 'Tuple does not have eb field'
            return ReportableQty(value[0], value[1])
        else:
            return ReportableQty(value)

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

    def render_with(self, f, ebstring='%s +/- %s'):
        if self.errbar is not None:
            rendered = ebstring % (f(self.value), f(self.errbar))
        else: 
            rendered = f(self.value)
        return rendered

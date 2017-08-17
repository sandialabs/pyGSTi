from copy import deepcopy as _deepcopy
import pickle as _pickle
import numpy as _np

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

    def __add__(self,x):
        if self.has_eb():
            return ReportableQty(self.value + x, self.errbar, self.nonMarkovianEBs)
        else:
            return ReportableQty(self.value + x)

    def __mul__(self,x):
        if self.has_eb():
            return ReportableQty(self.value * x, self.errbar * x, self.nonMarkovianEBs)
        else:
            return ReportableQty(self.value * x)

    def __truediv__(self,x):
        if self.has_eb():
            return ReportableQty(self.value / x, self.errbar / x, self.nonMarkovianEBs)
        else:
            return ReportableQty(self.value / x)

    def __getstate__(self):
        state_dict = self.__dict__.copy()
        return state_dict 

    def __setstate__(self, d):
        self.__dict__.update(d)

    def __copy__(self):
        return ReportableQty(self.value, self.errbar)

    def __deepcopy__(self, memo):
        return ReportableQty(_deepcopy(self.value, memo), _deepcopy(self.errbar, memo))

    #def __getattr__(self, attr):
        #print(self.value)
        #return getattr(self.value, attr)

    def log(self):
        """ Returns a ReportableQty that is the logarithm of this one."""
        # log(1 + x) ~ x
        # x + dx        
        # log(x + dx) = log(x(1 + dx/x)) = log x + log(1+dx/x) = log x + dx/x
        if self.has_eb():
            return ReportableQty( _np.log(self.value), _np.log(self.value + self.errbar) - _np.log(self.value),
                                  self.nonMarkovianEBs)
        else:
            return ReportableQty( _np.log(self.value) )

    def real(self):
        """ Returns a ReportableQty that is the real part of this one."""
        if self.has_eb():
            return ReportableQty( _np.real(self.value), _np.real(self.errbar), self.nonMarkovianEBs)
        else:
            return ReportableQty( _np.real(self.value) )
        
    def imag(self):
        """ Returns a ReportableQty that is the imaginary part of this one."""
        if self.has_eb():
            return ReportableQty( _np.real(self.value), _np.real(self.errbar), self.nonMarkovianEBs)
        else:
            return ReportableQty( _np.real(self.value) )

    def reshape(self, *args):
        """ Returns a ReportableQty whose underlying values are reshaped."""
        if self.has_eb():
            return ReportableQty( self.value.reshape(*args), self.errbar.reshape(*args), self.nonMarkovianEBs)
        else:
            return ReportableQty( self.value.reshape(*args) )

    @property
    def size(self):
        """ Returns the size of this ReportableQty's value. """
        return self.value.size
        
        

        

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
        if isinstance(value, ReportableQty):
            return value
        if isinstance(value, tuple):
            assert len(value) == 2, 'Tuple does not have eb field ' + \
                                    'or has too many fields: len = {}'.format(
                                            len(value))
            return ReportableQty(value[0], value[1], nonMarkovianEBs=nonMarkovianEBs)
        else:
            return ReportableQty(value, nonMarkovianEBs=nonMarkovianEBs)

    def has_eb(self):
        return self.errbar is not None

    def scale(self, factor):
        """
        Scale the value and error bar (if present) by `factor`.
        """
        self.value *= factor
        if self.has_eb(): self.errbar *= factor

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

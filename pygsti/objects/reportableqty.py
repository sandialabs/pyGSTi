""" The ReportableQty class """
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

from copy import deepcopy as _deepcopy
import numpy as _np

from .label import Label as _Label


def minimum(qty1, qty2):
    """ Returns a ReportableQty that is the minimum of `qty1` and `qty2`."""
    if qty1.value <= qty2.value:
        return qty1
    else:
        return qty2


def maximum(qty1, qty2):
    """ Returns a ReportableQty that is the maximum of `qty1` and `qty2`."""
    if qty1.value >= qty2.value:
        return qty1
    else:
        return qty2


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
        self.value = value
        self.errbar = errbar

        self.nonMarkovianEBs = nonMarkovianEBs

    def __str__(self):
        def f(x, y): return (str(x) + " +/- " + str(y)) if y else str(x)
        return self.render_with(f)

    def __repr__(self):
        return 'ReportableQty({})'.format(str(self))

    def __add__(self, x):
        if self.has_eb():
            return ReportableQty(self.value + x, self.errbar, self.nonMarkovianEBs)
        else:
            return ReportableQty(self.value + x)

    def __mul__(self, x):
        if self.has_eb():
            return ReportableQty(self.value * x, self.errbar * x, self.nonMarkovianEBs)
        else:
            return ReportableQty(self.value * x)

    def __truediv__(self, x):
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
        v = self.value
        if _np.any(_np.isreal(v)) and _np.any(v < 0):
            v = v.astype(complex)  # so logarithm can be complex

        if self.has_eb():
            return ReportableQty(_np.log(v), _np.log(v + self.errbar) - _np.log(v),
                                 self.nonMarkovianEBs)
        else:
            return ReportableQty(_np.log(v))

    def real(self):
        """ Returns a ReportableQty that is the real part of this one."""
        if self.has_eb():
            return ReportableQty(_np.real(self.value), _np.real(self.errbar), self.nonMarkovianEBs)
        else:
            return ReportableQty(_np.real(self.value))

    def imag(self):
        """ Returns a ReportableQty that is the imaginary part of this one."""
        if self.has_eb():
            return ReportableQty(_np.imag(self.value), _np.imag(self.errbar), self.nonMarkovianEBs)
        else:
            return ReportableQty(_np.imag(self.value))

    def absdiff(self, constant_value, separate_re_im=False):
        """
        Returns a ReportableQty that is the (element-wise in the vector case)
        difference between `constant_value` and this one given by:

        `abs(self - constant_value)`.
        """
        if separate_re_im:
            re_v = _np.fabs(_np.real(self.value) - _np.real(constant_value))
            im_v = _np.fabs(_np.imag(self.value) - _np.imag(constant_value))
            if self.has_eb():
                return (ReportableQty(re_v, _np.fabs(_np.real(self.errbar)), self.nonMarkovianEBs),
                        ReportableQty(im_v, _np.fabs(_np.imag(self.errbar)), self.nonMarkovianEBs))
            else:
                return ReportableQty(re_v), ReportableQty(im_v)

        else:
            v = _np.absolute(self.value - constant_value)
            if self.has_eb():
                return ReportableQty(v, _np.absolute(self.errbar), self.nonMarkovianEBs)
            else:
                return ReportableQty(v)

    def infidelity_diff(self, constant_value):
        """
        Returns a ReportableQty that is the (element-wise in the vector case)
        difference between `constant_value` and this one given by:

        `1.0 - Re(conjugate(constant_value) * self )`
        """
        # let diff(x) = 1.0 - Re(const.C * x) = 1.0 - (const.re * x.re + const.im * x.im)
        # so d(diff)/dx.re = -const.re, d(diff)/dx.im = -const.im
        # diff(x + dx) = diff(x) + d(diff)/dx * dx
        # diff(x + dx) - diff(x) =  - (const.re * dx.re + const.im * dx.im)
        v = 1.0 - _np.real(_np.conjugate(constant_value) * self.value)
        if self.has_eb():
            eb = abs(_np.real(constant_value) * _np.real(self.errbar)
                     + _np.imag(constant_value) * _np.real(self.errbar))
            return ReportableQty(v, eb, self.nonMarkovianEBs)
        else:
            return ReportableQty(v)

    def mod(self, x):
        """
        Returns a ReportableQty that holds `this_qty mod x`, that is,
        the value and error bar (if present are modulus-divided by `x`).
        """
        v = self.value % x
        if self.has_eb():
            eb = self.errbar % x
            return ReportableQty(v, eb, self.nonMarkovianEBs)
        else:
            return ReportableQty(v)

    def hermitian_to_real(self):
        """
        Returns a ReportableQty that holds the real matrix
        whose upper/lower triangle contains the real/imaginary parts
        of the corresponding off-diagonal matrix elements of the
        *Hermitian* matrix stored in this ReportableQty.

        This is used for display purposes.  If this object doesn't
        contain a Hermitian matrix, `ValueError` is raised.
        """
        if _np.linalg.norm(self.value - _np.conjugate(self.value).T) > 1e-8:
            raise ValueError("Contained value must be Hermitian!")

        def _convert(A):
            ret = _np.empty(A.shape, 'd')
            for i in range(A.shape[0]):
                ret[i, i] = A[i, i].real
                for j in range(i + 1, A.shape[1]):
                    ret[i, j] = A[i, j].real
                    ret[j, i] = A[i, j].imag
            return ret

        v = _convert(self.value)
        if self.has_eb():
            eb = _convert(self.errbar)
            return ReportableQty(v, eb, self.nonMarkovianEBs)
        else:
            return ReportableQty(v)

    def reshape(self, *args):
        """ Returns a ReportableQty whose underlying values are reshaped."""
        if self.has_eb():
            return ReportableQty(self.value.reshape(*args), self.errbar.reshape(*args), self.nonMarkovianEBs)
        else:
            return ReportableQty(self.value.reshape(*args))

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
        if isinstance(value, _Label):  # distinguish b/c Label is also a *tuple*
            return ReportableQty(value, nonMarkovianEBs=nonMarkovianEBs)
        if isinstance(value, tuple):
            assert len(value) == 2, 'Tuple does not have eb field ' + \
                                    'or has too many fields: len = {}'.format(
                len(value))
            return ReportableQty(value[0], value[1], nonMarkovianEBs=nonMarkovianEBs)
        else:
            return ReportableQty(value, nonMarkovianEBs=nonMarkovianEBs)

    def has_eb(self):
        """
        Return whether this quantity is storing an error bar (bool).
        """
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
        """
        Render this `ReportableQty` using the function `f`.

        Parameters
        ----------
        f : function
           The `formatter` function which separately converts the stored value
           and error bar (if present) to string quantities that are then
           formatted using `ebstring`, `nmebstring` or just `"%s"` (if there's
           no error bar).  This function must have the signature `f(val, specs)`
           where `val` is either the value or error bar and `specs` is a
           dictionary given by the next argument.

        specs : dict, optional
            Additional parameters to pass to the formatter function `f`.

        ebstring, nmebstring : str, optional
            The formatting strings used to format the values returned from `f`
            for normal and non-Markovian error bars, respectively.  If
            `nmebstring` is None then `ebstring` is used for both types of
            error bars.

        Returns
        -------
        str
        """
        if nmebstring is None:
            nmebstring = ebstring
        if specs is None:
            specs = dict()
        if self.errbar is not None:
            specs['formatstring'] = '%s'  # Don't recursively apply format strings to inside error bars
            if self.nonMarkovianEBs:
                rendered = nmebstring % (f(self.value, specs),
                                         f(self.errbar, specs))
            else:
                rendered = ebstring % (f(self.value, specs),
                                       f(self.errbar, specs))
        else:
            rendered = f(self.value, specs)
        return rendered

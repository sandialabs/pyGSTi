"""
The ReportableQty class
"""
#***************************************************************************************************
# Copyright 2015, 2019, 2025 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

from copy import deepcopy as _deepcopy

import numpy as _np

from pygsti.baseobjs.label import Label as _Label


def minimum(qty1, qty2):
    """
    Returns a ReportableQty that is the minimum of `qty1` and `qty2`.

    Parameters
    ----------
    qty1 : ReportableQty
        First quantity.

    qty2 : ReportableQty
        Second quantity.

    Returns
    -------
    ReportableQty
    """
    if qty1.value <= qty2.value:
        return qty1
    else:
        return qty2


def maximum(qty1, qty2):
    """
    Returns a ReportableQty that is the maximum of `qty1` and `qty2`.

    Parameters
    ----------
    qty1 : ReportableQty
        First quantity.

    qty2 : ReportableQty
        Second quantity.

    Returns
    -------
    ReportableQty
    """
    if qty1.value >= qty2.value:
        return qty1
    else:
        return qty2


class ReportableQty(object):
    """
    A computed quantity and possibly its error bars, primarily for use in reports.

    Parameters
    ----------
    value : object
        The value, usually a float or numpy array.

    errbar : object, optional
        The (symmetric) error bar on `value`.  If `value` is an
        array, `errbar` has the same shape.  `None` is used to
        signify "no error bars".

    non_markovian_ebs : bool, optional
        Whether these error bars are "non-markovian"-type
        error bars (it can be useful to keep track of this
        for formatting).

    Attributes
    ----------
    size : int
        Returns the size of this ReportableQty's value.
    """

    def __init__(self, value, errbar=None, non_markovian_ebs=False):
        """
        Initialize a new ReportableQty object, which
        is essentially a container for a value and error bars.

        Parameters
        ----------
        value : anything
           The value to store

        errbar : anything
           The error bar(s) to store

        non_markovian_ebs : bool
            boolean indicating if non markovian error bars should be used
        """
        self._value = value
        self._errorbar = errbar

        self.nonMarkovianEBs = non_markovian_ebs

    def __str__(self):
        def f(val, specs): return str(val)
        return self.render_with(f)

    def __repr__(self):
        return 'ReportableQty({})'.format(str(self))

    def __add__(self, x):
        if self.has_errorbar:
            return ReportableQty(self.value + x, self.errorbar, self.nonMarkovianEBs)
        else:
            return ReportableQty(self.value + x)

    def __mul__(self, x):
        if self.has_errorbar:
            return ReportableQty(self.value * x, self.errorbar * x, self.nonMarkovianEBs)
        else:
            return ReportableQty(self.value * x)

    def __truediv__(self, x):
        if self.has_errorbar:
            return ReportableQty(self.value / x, self.errorbar / x, self.nonMarkovianEBs)
        else:
            return ReportableQty(self.value / x)

    def __getstate__(self):
        state_dict = self.__dict__.copy()
        return state_dict

    def __setstate__(self, d):
        self.__dict__.update(d)

    def __copy__(self):
        return ReportableQty(self.value, self.errorbar)

    def __deepcopy__(self, memo):
        return ReportableQty(_deepcopy(self.value, memo), _deepcopy(self.errorbar, memo))

    def log(self):
        """
        Returns a ReportableQty that is the logarithm of this one.

        Returns
        -------
        ReportableQty
        """
        # log(1 + x) ~ x
        # x + dx
        # log(x + dx) = log(x(1 + dx/x)) = log x + log(1+dx/x) = log x + dx/x
        v = self.value
        if _np.any(_np.isreal(v)) and _np.any(v < 0):
            v = v.astype(complex)  # so logarithm can be complex

        if self.has_errorbar:
            return ReportableQty(_np.log(v), _np.log(v + self.errorbar) - _np.log(v),
                                 self.nonMarkovianEBs)
        else:
            return ReportableQty(_np.log(v))

    def real(self):
        """
        Returns a ReportableQty that is the real part of this one.

        Returns
        -------
        ReportableQty
        """
        if self.has_errorbar:
            return ReportableQty(_np.real(self.value), _np.real(self.errorbar), self.nonMarkovianEBs)
        else:
            return ReportableQty(_np.real(self.value))

    def imag(self):
        """
        Returns a ReportableQty that is the imaginary part of this one.

        Returns
        -------
        ReportableQty
        """
        if self.has_errorbar:
            return ReportableQty(_np.imag(self.value), _np.imag(self.errorbar), self.nonMarkovianEBs)
        else:
            return ReportableQty(_np.imag(self.value))

    def absdiff(self, constant_value, separate_re_im=False):
        """
        Create a ReportableQty that is the difference between `constant_value` and this one.

        The returned quantity's value is given by (element-wise in the vector case):

        `abs(self - constant_value)`.

        Parameters
        ----------
        constant_value : float or numpy.ndarray
            The constant value to use.

        separate_re_im : bool, optional
            When `True`, two separate real- and imaginary-part
            :class:`ReportableQty` objects are returned (applicable
            to complex-valued quantities).

        Returns
        -------
        ReportableQty or tuple
            The output `ReportableQty`(s).  If `separate_re_im=True` then
            a 2-tuple of (real-part, imaginary-part) quantities is returned.
            Otherwise a single quantity is returned.
        """
        if separate_re_im:
            re_v = _np.fabs(_np.real(self.value) - _np.real(constant_value))
            im_v = _np.fabs(_np.imag(self.value) - _np.imag(constant_value))
            if self.has_errorbar:
                return (ReportableQty(re_v, _np.fabs(_np.real(self.errorbar)), self.nonMarkovianEBs),
                        ReportableQty(im_v, _np.fabs(_np.imag(self.errorbar)), self.nonMarkovianEBs))
            else:
                return ReportableQty(re_v), ReportableQty(im_v)

        else:
            v = _np.absolute(self.value - constant_value)
            if self.has_errorbar:
                return ReportableQty(v, _np.absolute(self.errorbar), self.nonMarkovianEBs)
            else:
                return ReportableQty(v)

    def infidelity_diff(self, constant_value):
        """
        Creates a ReportableQty that is the difference between `constant_value` and this one.

        The returned quantity's value is given by (element-wise in the vector case):

        `1.0 - Re(conjugate(constant_value) * self )`

        Parameters
        ----------
        constant_value : float or numpy.ndarray
            The constant value to use.

        Returns
        -------
        ReportableQty
        """
        # let diff(x) = 1.0 - Re(const.C * x) = 1.0 - (const.re * x.re + const.im * x.im)
        # so d(diff)/dx.re = -const.re, d(diff)/dx.im = -const.im
        # diff(x + dx) = diff(x) + d(diff)/dx * dx
        # diff(x + dx) - diff(x) =  - (const.re * dx.re + const.im * dx.im)
        v = 1.0 - _np.real(_np.conjugate(constant_value) * self.value)
        if self.has_errorbar:
            eb = abs(_np.real(constant_value) * _np.real(self.errorbar)
                     + _np.imag(constant_value) * _np.real(self.errorbar))
            return ReportableQty(v, eb, self.nonMarkovianEBs)
        else:
            return ReportableQty(v)

    def mod(self, x):
        """
        Creates a ReportableQty that holds `this_qty mod x`.

        That is, the value and error bar (if present) are modulus-divided by `x`.

        Parameters
        ----------
        x : int
            Value to modulus-divide by.

        Returns
        -------
        ReportableQty
        """
        v = self.value % x
        if self.has_errorbar:
            eb = self.errorbar % x
            return ReportableQty(v, eb, self.nonMarkovianEBs)
        else:
            return ReportableQty(v)

    def hermitian_to_real(self):
        """
        Creates a ReportableQty that holds a real "version" of a Hermitian matrix.

        Specifically, the returned quantity's value is the real matrix
        whose upper/lower triangle contains the real/imaginary parts
        of the corresponding off-diagonal matrix elements of the
        *Hermitian* matrix stored in this ReportableQty.

        This is used for display purposes.  If this object doesn't
        contain a Hermitian matrix, `ValueError` is raised.

        Returns
        -------
        ReportableQty
        """
        if _np.linalg.norm(self.value - _np.conjugate(self.value).T) > 1e-8:
            raise ValueError("Contained value must be Hermitian!")

        def _convert(a):
            ret = _np.empty(a.shape, 'd')
            for i in range(a.shape[0]):
                ret[i, i] = a[i, i].real
                for j in range(i + 1, a.shape[1]):
                    ret[i, j] = a[i, j].real
                    ret[j, i] = a[i, j].imag
            return ret

        v = _convert(self.value)
        if self.has_errorbar:
            eb = _convert(self.errorbar)
            return ReportableQty(v, eb, self.nonMarkovianEBs)
        else:
            return ReportableQty(v)

    def reshape(self, *args):
        """
        Returns a ReportableQty whose underlying values are reshaped.

        Returns
        -------
        ReportableQty
        """
        if self.has_errorbar:
            return ReportableQty(self.value.reshape(*args), self.errorbar.reshape(*args), self.nonMarkovianEBs)
        else:
            return ReportableQty(self.value.reshape(*args))

    @property
    def size(self):
        """
        Returns the size of this ReportableQty's value.

        Returns
        -------
        int
        """
        return self.value.size

    @staticmethod
    def from_val(value, non_markovian_ebs=False):
        """
        Convert Table values into ReportableQtys or leave them be if they are well-formed types.

        Well-formed types include:
        * strings
        * figures
        * :class:`ReportableQty`

        A tuple will be converted to a :class:`ReportableQty`
        holding the first field as a value and second field as an error bar.
        Anything else will be converted to a ReportableQty with no error bars.

        Parameters
        ----------
        value : object
            The value to convert.

        non_markovian_ebs : bool, optional
            Whether the error bars are of the "non-markovian"-type.

        Returns
        -------
        ReportableQty
        """
        if isinstance(value, ReportableQty):
            return value
        if isinstance(value, _Label):  # distinguish b/c Label is also a *tuple*
            return ReportableQty(value, non_markovian_ebs=non_markovian_ebs)
        if isinstance(value, tuple):
            assert len(value) == 2, 'Tuple does not have eb field ' + \
                                    'or has too many fields: len = {}'.format(
                len(value))
            return ReportableQty(value[0], value[1], non_markovian_ebs=non_markovian_ebs)
        else:
            return ReportableQty(value, non_markovian_ebs=non_markovian_ebs)

    @property
    def has_errorbar(self):
        """
        Return whether this quantity is storing an error bar (bool).

        Returns
        -------
        bool
        """
        return self.errorbar is not None

    def scale_inplace(self, factor):
        """
        Scale the value and error bar (if present) by `factor`.

        Parameters
        ----------
        factor : float
            The scaling factor.

        Returns
        -------
        None
        """
        self._value *= factor
        if self.has_errorbar:
            self._errorbar *= factor

    @property
    def value(self):
        """
        Returns the quantity's value

        Returns
        -------
        object
            Usually a float or numpy array.
        """
        return self._value

    @property
    def errorbar(self):
        """
        Returns the quantity's error bar(s)

        Returns
        -------
        object
            Usually a float or numpy array.
        """
        return self._errorbar

    @property
    def value_and_errorbar(self):
        """
        Returns the quantity's value and error bar(s)

        Returns
        -------
        value : object
            This object's value (usually a float or numpy array).

        error_bar : object
            This object's value (usually a float or numpy array).
        """
        # XXX isn't this redundant?
        return self.value, self.errorbar

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

        ebstring : str, optional
            format string that describes how to display the value and error bar
            after they are rendered as string (`ebstring` should have two `%s` in it).

        nmebstring : str, optional
            format string, similar to `ebstring`, for displaying non-Markovian error
            bars (if None then `ebstring` is used).

        Returns
        -------
        str
        """
        if nmebstring is None:
            nmebstring = ebstring
        if specs is None:
            specs = dict()
        if self.has_errorbar:
            specs['formatstring'] = '%s'  # Don't recursively apply format strings to inside error bars
            if self.nonMarkovianEBs:
                rendered = nmebstring % (f(self.value, specs),
                                         f(self.errorbar, specs))
            else:
                rendered = ebstring % (f(self.value, specs),
                                       f(self.errorbar, specs))
        else:
            rendered = f(self.value, specs)
        return rendered

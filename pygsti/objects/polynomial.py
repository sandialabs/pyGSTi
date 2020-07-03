"""
Defines the Polynomial class
"""
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

import numpy as _np
import platform as _platform
import collections as _collections
from . import replib

assert(_platform.architecture()[0].endswith("bit"))  # e.g. "64bit"
PLATFORM_BITS = int(_platform.architecture()[0].strip("bit"))


def _vinds_to_int(vinds, vindices_per_int, max_num_vars):
    """ Convert tuple index of ints to single int given max_numvars """
    ints_in_key = int(_np.ceil(len(vinds) / vindices_per_int))
    ret_tup = []
    for k in range(ints_in_key):
        ret = 0; m = 1
        for i in vinds[k * vindices_per_int:(k + 1) * vindices_per_int]:  # last tuple index is most significant
            assert(i < max_num_vars), "Variable index exceed maximum!"
            ret += (i + 1) * m
            m *= max_num_vars + 1
        assert(ret >= 0), "vinds = %s -> %d!!" % (str(vinds), ret)
        ret_tup.append(ret)
    return tuple(ret_tup)


class FASTPolynomial(object):
    """
    A polynomial that behaves like a Python dict of coefficients.

    Variables are represented by integer indices, e.g. "2" means "x_2".
    Keys are tuples of variable indices and values are numerical
    coefficients (floating point or complex numbers).  To specify a variable
    to some power, its index is repeated in the key-tuple.

    E.g. x_0^2 + 3*x_1 + 4 is stored as {(0,0): 1.0, (1,): 3.0, (): 4.0}

    Parameters
    ----------
    coeffs : dict
        A dictionary of coefficients.  Keys are tuples of integers that
        specify the polynomial term the coefficient value multiplies
        (see above). If None, the zero polynomial (no terms) is created.

    max_num_vars : int
        The maximum number of independent variables this polynomial can
        hold.  Placing a limit on the number of variables allows more
        compact storage and efficient evaluation of the polynomial.

    Attributes
    ----------
    coeffs : dict
        A dictionary whose keys are tuples of variable indices (e.g.
        `x0^2*x1` would translate to `(0,0,1)`) and values are the
        coefficients.

    max_num_vars : int
        The maximum number of independent variables this polynomial can
        hold.

    vindices_per_int : int
        The number of variable indices that can be compactly fit
        into a single int when there are at most `max_num_vars` variables.
    """

    @classmethod
    def _vindices_per_int(cls, max_num_vars):
        """
        The number of variable indices that fit into a single int when there are at most `max_num_vars` variables.

        This quantity is needed to directly construct Polynomial representations
        and is thus useful internally for forward simulators.

        Parameters
        ----------
        max_num_vars : int
            The maximum number of independent variables.

        Returns
        -------
        int
        """
        # (max_num_vars+1) ** vindices_per_int <= 2**PLATFORM_BITS, so:
        # vindices_per_int * log2(max_num_vars+1) <= PLATFORM_BITS
        # vindices_per_int = int(_np.floor(PLATFORM_BITS / _np.log2(max_num_vars + 1)))
        return int(_np.floor(PLATFORM_BITS / _np.log2(max_num_vars + 1)))

    @classmethod
    def from_rep(cls, rep):
        """
        Creates a Polynomial from a "representation" (essentially a lite-version) of a Polynomial.

        Note: usually we only need to convert from full-featured Python objects
        to the lighter-weight "representation" objects.  Polynomials are an
        exception, since as the results of probability computations they need
        to be converted back from "representation-form" to "full-form".

        Parameters
        ----------
        rep : PolynomialRep
            A polynomial representation.

        Returns
        -------
        Polynomial
        """
        self = cls.__new__(cls)
        self._rep = rep
        return self

    @classmethod
    def product(cls, list_of_polys):
        """
        Take the product of multiple polynomials.

        Parameters
        ----------
        list_of_polys : list
            List of polynomials to take the product of.

        Returns
        -------
        Polynomial
        """
        rep = list_of_polys[0]._rep
        for p in list_of_polys[1:]:
            rep = rep.mult(p._rep)
        return FASTPolynomial.from_rep(rep)

    def __init__(self, coeffs=None, max_num_vars=100):
        """
        Initializes a new Polynomial object (a subclass of dict).

        Internally (as a dict) a Polynomial represents variables by integer
        indices, e.g. "2" means "x_2".  Keys are tuples of variable indices and
        values are numerical coefficients (floating point or complex numbers).
        A variable to a power > 1 has its index repeated in the key-tuple.

        E.g. x_0^2 + 3*x_1 + 4 is stored as `{(0,0): 1.0, (1,): 3.0, (): 4.0}`

        Parameters
        ----------
        coeffs : dict
            A dictionary of coefficients.  Keys are tuples of integers that
            specify the polynomial term the coefficient value multiplies
            (see above). If None, the zero polynomial (no terms) is created.

        max_num_vars : int, optional
            The maximum number of variables the represenatation is allowed to
            have (x_0 to x_(`max_num_vars-1`)).  This sets the maximum allowed
            variable index within this polynomial.
        """
        vindices_per_int = FASTPolynomial._vindices_per_int(max_num_vars)

        int_coeffs = {_vinds_to_int(k, vindices_per_int, max_num_vars): v for k, v in coeffs.items()}
        self._rep = replib.PolynomialRep(int_coeffs, max_num_vars, vindices_per_int)

    @property
    def coeffs(self):
        """
        A dictionary of this polynoial's coefficients.

        Keys are tuples of integers that specify the polynomial term the
        coefficient value multiplies (see above). If None, the zero polynomial
        (no terms) is created.

        Returns
        -------
        dict
        """
        max_num_vars = self._rep.max_num_vars

        def int_to_vinds(indx_tup):
            ret = []
            for indx in indx_tup:
                while indx != 0:
                    nxt = indx // (max_num_vars + 1)
                    i = indx - nxt * (max_num_vars + 1)
                    ret.append(i - 1)
                    indx = nxt
            #assert(len(ret) <= max_order) #TODO: is this needed anymore?
            return tuple(sorted(ret))

        return {int_to_vinds(k): val for k, val in self._rep.int_coeffs.items()}

    @property
    def max_num_vars(self):  # so we can convert back to python Polys
        """
        The maximum number of independent variables this polynomial can hold.

        Powers of variables are not "independent", e.g. the polynomial x0^2 + 2*x0 + 3
        has a single indepdent variable.

        Returns
        -------
        int
        """
        return self._rep.max_num_vars

    @property
    def vindices_per_int(self):
        """
        The number of this polynoial's variable indices that can be compactly fit into a single int.

        Returns
        -------
        int
        """
        return self._rep.vindices_per_int

    def deriv(self, wrt_param):
        """
        Take the derivative of this Polynomial with respect to a single variable.

        The result is another Polynomial.

        E.g. deriv(x_2^3 + 3*x_1, wrt_param=2) = 3x^2

        Parameters
        ----------
        wrt_param : int
            The variable index to differentiate with respect to.
            E.g. "4" means "differentiate w.r.t. x_4".

        Returns
        -------
        Polynomial
        """
        dcoeffs = {}
        for ivar, coeff in self.coeffs.items():
            cnt = float(ivar.count(wrt_param))
            if cnt > 0:
                l = list(ivar)
                del l[l.index(wrt_param)]
                dcoeffs[tuple(l)] = cnt * coeff

        return FASTPolynomial(dcoeffs, self.max_num_vars)

    @property
    def degree(self):
        """
        The largest sum-of-exponents for any term (monomial) within this polynomial.

        E.g. for x_2^3 + x_1^2*x_0^2 has degree 4.

        Returns
        -------
        int
        """
        return max((len(k) for k in self.coeffs), default=0)

    def evaluate(self, variable_values):
        """
        Evaluate this polynomial for a given set of variable values.

        Parameters
        ----------
        variable_values : array-like
            An object that can be indexed so that `variable_values[i]` gives the
            numerical value for i-th variable (x_i).

        Returns
        -------
        float or complex
            Depending on the types of the coefficients and `variable_values`.
        """
        #FUTURE: make this function smarter (Russian peasant)
        ret = 0
        for ivar, coeff in self.coeffs.items():
            ret += coeff * _np.product([variable_values[i] for i in ivar])
        return ret

    def compact(self, complex_coeff_tape=True):
        """
        Generate a compact form of this polynomial designed for fast evaluation.

        The resulting "tapes" can be evaluated using
        :function:`opcalc.bulk_eval_compact_polynomials`.

        Parameters
        ----------
        complex_coeff_tape : bool, optional
            Whether the `ctape` returned array is forced to be of complex type.
            If False, the real part of all coefficients is taken (even if they're
            complex).

        Returns
        -------
        vtape, ctape : numpy.ndarray
            These two 1D arrays specify an efficient means for evaluating this
            polynomial.
        """
        if complex_coeff_tape:
            return self._rep.compact_complex()
        else:
            return self._rep.compact_real()

    def copy(self):
        """
        Returns a copy of this polynomial.

        Returns
        -------
        Polynomial
        """
        return FASTPolynomial.from_rep(self._rep.copy())

    def map_indices(self, mapfn):
        """
        Performs a bulk find & replace on this polynomial's variable indices.

        This is useful when the variable indices have external significance
        (like being the indices of a gate's parameters) and one want to convert
        to another set of indices (like a parent model's parameters).

        Parameters
        ----------
        mapfn : function
            A function that takes as input an "old" variable-index-tuple
            (a key of this Polynomial) and returns the updated "new"
            variable-index-tuple.

        Returns
        -------
        Polynomial
        """
        return FASTPolynomial({mapfn(k): v for k, v in self.coeffs.items()}, self.max_num_vars)

    def map_indices_inplace(self, mapfn):
        """
        Performs an in-place find & replace on this polynomial's variable indices.

        This is useful when the variable indices have external significance
        (like being the indices of a gate's parameters) and one want to convert
        to another set of indices (like a parent model's parameters).

        Parameters
        ----------
        mapfn : function
            A function that takes as input an "old" variable-index-tuple
            (a key of this Polynomial) and returns the updated "new"
            variable-index-tuple.

        Returns
        -------
        None
        """
        new_coeffs = {mapfn(k): v for k, v in self.coeffs.items()}
        new_int_coeffs = {_vinds_to_int(k, self._rep.vindices_per_int, self._rep.max_num_vars): v
                          for k, v in new_coeffs.items()}
        self._rep.reinit(new_int_coeffs)

    def mapvec_indices(self, mapvec):
        """
        Performs a bulk find & replace on this polynomial's variable indices.

        This function is similar to :method:`map_indices` but uses a *vector*
        to describe *individual* index updates instead of a function for
        increased performance.

        Parameters
        ----------
        mapvec : numpy.ndarray
            An array whose i-th element gives the updated "new" index for
            the i-th variable.  Note that this vector maps *individual*
            variable indices old->new, whereas `mapfn` in :method:`map_indices`
            maps between *tuples* of indices.

        Returns
        -------
        Polynomial
        """
        ret = self.copy()
        ret._rep.mapvec_indices_inplace(mapvec)
        return ret

    def mapvec_indices_inplace(self, mapvec):
        """
        Performs an in-place bulk find & replace on this polynomial's variable indices.

        This function is similar to :method:`map_indices_inplace` but uses a *vector*
        to describe *individual* index updates instead of a function for increased
        performance.

        Parameters
        ----------
        mapvec : numpy.ndarray
            An array whose i-th element gives the updated "new" index for
            the i-th variable.  Note that this vector maps *individual*
            variable indices old->new, whereas `mapfn` in
            :method:`map_indices_inplace` maps between *tuples* of indices.

        Returns
        -------
        Polynomial
        """
        self._rep.mapvec_indices_inplace(mapvec)

    def mult(self, x):
        """
        Multiplies this polynomial by another polynomial `x`.

        Parameters
        ----------
        x : Polynomial
            The polynomial to multiply by.

        Returns
        -------
        Polynomial
            The polynomial representing self * x.
        """
        return FASTPolynomial.from_rep(self._rep.mult(x._rep))

    def scale(self, x):
        """
        Scale this polynomial by `x` (multiply all coefficients by `x`).

        Parameters
        ----------
        x : float or complex
            The value to scale by.

        Returns
        -------
        None
        """
        # assume a scalar that can multiply values
        self._rep.scale(x)

    def scalar_mult(self, x):
        """
        Multiplies this polynomial by a scalar `x`.

        Parameters
        ----------
        x : float or complex
            The value to multiply by.

        Returns
        -------
        Polynomial
        """
        newpoly = self.copy()
        newpoly.scale(x)
        return newpoly

    def __str__(self):
        def fmt(x):
            if abs(_np.imag(x)) > 1e-6:
                if abs(_np.real(x)) > 1e-6: return "(%.3f+%.3fj)" % (x.real, x.imag)
                else: return "(%.3fj)" % x.imag
            else: return "%.3f" % x.real

        termstrs = []
        coeffs = self.coeffs
        sorted_keys = sorted(list(coeffs.keys()))
        for k in sorted_keys:
            varstr = ""; last_i = None; n = 1
            for i in sorted(k):
                if i == last_i: n += 1
                elif last_i is not None:
                    varstr += "x%d%s" % (last_i, ("^%d" % n) if n > 1 else "")
                    n = 1
                last_i = i
            if last_i is not None:
                varstr += "x%d%s" % (last_i, ("^%d" % n) if n > 1 else "")
            #print("DB: k = ",k, " varstr = ",varstr)
            if abs(coeffs[k]) > 1e-4:
                termstrs.append("%s%s" % (fmt(coeffs[k]), varstr))
        if len(termstrs) > 0:
            return " + ".join(termstrs)
        else: return "0"

    def __repr__(self):
        return "Poly[ " + str(self) + " ]"

    def __add__(self, x):
        newpoly = self.copy()
        if isinstance(x, FASTPolynomial):
            newpoly._rep.add_inplace(x._rep)
        else:  # assume a scalar that can be added to values
            newpoly._rep.add_scalar_to_all_coeffs_inplace(x)
        return newpoly

    def __iadd__(self, x):
        """ Does self += x more efficiently """
        if isinstance(x, FASTPolynomial):
            self._rep.add_inplace(x._rep)
        else:  # assume a scalar that can be added to values
            self._rep.add_scalar_to_all_coeffs_inplace(x)
        return self

    def __mul__(self, x):
        if isinstance(x, FASTPolynomial):
            return self.mult(x)
        else:  # assume a scalar that can multiply values
            return self.scalar_mult(x)

    def __rmul__(self, x):
        return self.__mul__(x)

    #Punt for now??
    #def __imul__(self, x):
    #    if isinstance(x, Polynomial):
    #        newcoeffs = {}
    #        for k1, v1 in self.items():
    #            for k2, v2 in x.items():
    #                k = tuple(sorted(k1 + k2))
    #                if k in newcoeffs: newcoeffs[k] += v1 * v2
    #                else: newcoeffs[k] = v1 * v2
    #        self.clear()
    #        self.update(newcoeffs)
    #    else:
    #        self.scale(x)
    #    return self

    def __pow__(self, n):
        ret = FASTPolynomial({(): 1.0}, self.max_num_vars)  # max_order updated by mults below
        cur = self
        for i in range(int(_np.floor(_np.log2(n))) + 1):
            rem = n % 2  # gets least significant bit (i-th) of n
            if rem == 1: ret *= cur  # add current power of x (2^i) if needed
            cur = cur * cur  # current power *= 2
            n //= 2  # shift bits of n right
        return ret

    def __copy__(self):
        return self.copy()

    def to_rep(self):  # , max_num_vars=None not needed anymore -- given at __init__ time
        """
        Construct a representation of this polynomial.

        "Representations" are lightweight versions of objects used to improve
        the efficiency of intensely computational tasks.  Note that Polynomial
        representations must have the same `max_order` and `max_num_vars` in
        order to interact with each other (add, multiply, etc.).

        Returns
        -------
        PolynomialRep
        """
        return self._rep


Polynomial = FASTPolynomial


class SLOWPolynomial(dict):  # REMOVE THIS CLASS (just for reference)
    """
    Encapsulates a polynomial as a subclass of the standard Python dict.

    Variables are represented by integer indices, e.g. "2" means "x_2".
    Keys are tuples of variable indices and values are numerical
    coefficients (floating point or complex numbers).  To specify a variable
    to some power, its index is repeated in the key-tuple.

    E.g. x_0^2 + 3*x_1 + 4 is stored as {(0,0): 1.0, (1,): 3.0, (): 4.0}

    Parameters
    ----------
    coeffs : dict
        A dictionary of coefficients.  Keys are tuples of integers that
        specify the polynomial term the coefficient value multiplies
        (see above). If None, the zero polynomial (no terms) is created.

    max_num_vars : int
        The maximum number of independent variables this polynomial can
        hold.  Placing a limit on the number of variables allows more
        compact storage and efficient evaluation of the polynomial.
    """

    @classmethod
    def _vindices_per_int(cls, max_num_vars):
        """
        The number of variable indices that fit into a single int when there are at most `max_num_vars` variables.

        This quantity is needed to directly construct Polynomial representations
        and is thus useful internally for forward simulators.

        Parameters
        ----------
        max_num_vars : int
            The maximum number of independent variables.

        Returns
        -------
        int
        """
        return int(_np.floor(PLATFORM_BITS / _np.log2(max_num_vars + 1)))

    @classmethod
    def from_rep(cls, rep):
        """
        Creates a Polynomial from a "representation" (essentially a lite-version) of a Polynomial.

        Note: usually we only need to convert from full-featured Python objects
        to the lighter-weight "representation" objects.  Polynomials are an
        exception, since as the results of probability computations they need
        to be converted back from "representation-form" to "full-form".

        Parameters
        ----------
        rep : PolynomialRep
            A polynomial representation.

        Returns
        -------
        Polynomial
        """
        max_num_vars = rep.max_num_vars  # one of the few/only cases where a rep
        # max_order = rep.max_order        # needs to expose some python properties

        def int_to_vinds(indx_tup):
            ret = []
            for indx in indx_tup:
                while indx != 0:
                    nxt = indx // (max_num_vars + 1)
                    i = indx - nxt * (max_num_vars + 1)
                    ret.append(i - 1)
                    indx = nxt
            #assert(len(ret) <= max_order) #TODO: is this needed anymore?
            return tuple(sorted(ret))

        tup_coeff_dict = {int_to_vinds(k): val for k, val in rep.coeffs.items()}
        ret = cls(tup_coeff_dict)
        ret.fastpoly = FASTPolynomial.from_rep(rep)
        ret._check_fast_polynomial()
        return ret

    def __init__(self, coeffs=None, max_num_vars=100):
        """
        Initializes a new Polynomial object (a subclass of dict).

        Internally (as a dict) a Polynomial represents variables by integer
        indices, e.g. "2" means "x_2".  Keys are tuples of variable indices and
        values are numerical coefficients (floating point or complex numbers).
        A variable to a power > 1 has its index repeated in the key-tuple.

        E.g. x_0^2 + 3*x_1 + 4 is stored as `{(0,0): 1.0, (1,): 3.0, (): 4.0}`

        Parameters
        ----------
        coeffs : dict
            A dictionary of coefficients.  Keys are tuples of integers that
            specify the polynomial term the coefficient value multiplies
            (see above). If None, the zero polynomial (no terms) is created.

        max_num_vars : int
            The maximum number of independent variables this polynomial can
            hold.  Placing a limit on the number of variables allows more
            compact storage and efficient evaluation of the polynomial.
        """
        super(Polynomial, self).__init__()
        if coeffs is not None:
            self.update(coeffs)
        self.max_num_vars = max_num_vars
        self.fastpoly = FASTPolynomial(coeffs, max_num_vars)
        self._check_fast_polynomial()

    def _check_fast_polynomial(self, raise_err=True):
        """
        Check that included FASTPolynomial has remained in-sync with this one.

        This is purely for debugging, to ensure that the FASTPolynomial
        class implements its operations correctly.

        Parameters
        ----------
        raise_err : bool, optional
            Whether to raise an AssertionError if the check fails.

        Returns
        -------
        bool
            Whether or not the check has succeeded (True if the
            fast and slow implementations are in sync).
        """
        if set(self.fastpoly.coeffs.keys()) != set(self.keys()):
            print("FAST", self.fastpoly.coeffs, " != SLOW", dict(self))
            if raise_err: assert(False), "STOP"
            return False
        for k in self.fastpoly.coeffs.keys():
            if not _np.isclose(self.fastpoly.coeffs[k], self[k]):
                print("FAST", self.fastpoly.coeffs, " != SLOW", dict(self))
                if raise_err: assert(False), "STOP"
                return False
        if self.max_num_vars != self.fastpoly.max_num_vars:
            print("#Var mismatch: FAST", self.fastpoly.max_num_vars, " != SLOW", self.max_num_vars)
            if raise_err: assert(False), "STOP"
            return False

        return True

    def deriv(self, wrt_param):
        """
        Take the derivative of this Polynomial with respect to a single variable.

        The result is another Polynomial.

        E.g. deriv(x_2^3 + 3*x_1, wrt_param=2) = 3x^2

        Parameters
        ----------
        wrt_param : int
            The variable index to differentiate with respect to.
            E.g. "4" means "differentiate w.r.t. x_4".

        Returns
        -------
        Polynomial
        """
        dcoeffs = {}
        for ivar, coeff in self.items():
            cnt = float(ivar.count(wrt_param))
            if cnt > 0:
                l = list(ivar)
                del l[l.index(wrt_param)]
                dcoeffs[tuple(l)] = cnt * coeff

        ret = Polynomial(dcoeffs, self.max_num_vars)
        ret.fastpoly = self.fastpoly.deriv(wrt_param)
        ret._check_fast_polynomial()
        return ret

    def degree(self):
        """
        The largest sum-of-exponents for any term (monomial) within this polynomial.

        E.g. for x_2^3 + x_1^2*x_0^2 has degree 4.

        Returns
        -------
        int
        """
        ret = max((len(k) for k in self), default=0)
        assert(self.fastpoly.degree == ret)
        self._check_fast_polynomial()
        return ret

    def evaluate(self, variable_values):
        """
        Evaluate this polynomial for a given set of variable values.

        Parameters
        ----------
        variable_values : array-like
            An object that can be indexed so that `variable_values[i]` gives the
            numerical value for i-th variable (x_i).

        Returns
        -------
        float or complex
            Depending on the types of the coefficients and `variable_values`.
        """
        #FUTURE: make this function smarter (Russian peasant)
        ret = 0
        for ivar, coeff in self.items():
            ret += coeff * _np.product([variable_values[i] for i in ivar])
        assert(_np.isclose(ret, self.fastpoly.evaluate(variable_values)))
        self._check_fast_polynomial()
        return ret

    def compact(self, complex_coeff_tape=True):
        """
        Generate a compact form of this polynomial designed for fast evaluation.

        The resulting "tapes" can be evaluated using
        :function:`opcalc.bulk_eval_compact_polynomials`.

        Parameters
        ----------
        complex_coeff_tape : bool, optional
            Whether the `ctape` returned array is forced to be of complex type.
            If False, the real part of all coefficients is taken (even if they're
            complex).

        Returns
        -------
        vtape, ctape : numpy.ndarray
            These two 1D arrays specify an efficient means for evaluating this
            polynomial.
        """
        #if force_complex:
        #    iscomplex = True
        #else:
        #    iscomplex = any([abs(_np.imag(x)) > 1e-12 for x in self.values()])
        iscomplex = complex_coeff_tape

        nTerms = len(self)
        nVarIndices = sum(map(len, self.keys()))
        vtape = _np.empty(1 + nTerms + nVarIndices, _np.int64)  # "variable" tape
        ctape = _np.empty(nTerms, complex if iscomplex else 'd')  # "coefficient tape"

        i = 0
        vtape[i] = nTerms; i += 1
        for iTerm, k in enumerate(sorted(self.keys())):
            l = len(k)
            ctape[iTerm] = self[k] if iscomplex else _np.real(self[k])
            vtape[i] = l; i += 1
            vtape[i:i + l] = k; i += l
        assert(i == len(vtape)), "Logic Error!"
        fast_vtape, fast_ctape = self.fastpoly.compact(iscomplex)
        assert(_np.allclose(fast_vtape, vtape) and _np.allclose(fast_ctape, ctape))
        self._check_fast_polynomial()
        return vtape, ctape

    def copy(self):
        """
        Returns a copy of this polynomial.

        Returns
        -------
        Polynomial
        """
        fast_cpy = self.fastpoly.copy()
        ret = Polynomial(self, self.max_num_vars)
        ret.fastpoly = fast_cpy
        ret._check_fast_polynomial()
        return ret

    def map_indices(self, mapfn):
        """
        Performs a bulk find & replace on this polynomial's variable indices.

        This is useful when the variable indices have external significance
        (like being the indices of a gate's parameters) and one want to convert
        to another set of indices (like a parent model's parameters).

        Parameters
        ----------
        mapfn : function
            A function that takes as input an "old" variable-index-tuple
            (a key of this Polynomial) and returns the updated "new"
            variable-index-tuple.

        Returns
        -------
        Polynomial
        """
        ret = Polynomial({mapfn(k): v for k, v in self.items()}, self.max_num_vars)
        ret.fastpoly = self.fastpoly.map_indices(mapfn)
        self._check_fast_polynomial()
        ret._check_fast_polynomial()
        return ret

    def map_indices_inplace(self, mapfn):
        """
        Performs an in-place find & replace on this polynomial's variable indices.

        This is useful when the variable indices have external significance
        (like being the indices of a gate's parameters) and one want to convert
        to another set of indices (like a parent model's parameters).

        Parameters
        ----------
        mapfn : function
            A function that takes as input an "old" variable-index-tuple
            (a key of this Polynomial) and returns the updated "new"
            variable-index-tuple.

        Returns
        -------
        None
        """
        self._check_fast_polynomial()
        new_items = {mapfn(k): v for k, v in self.items()}
        self.clear()
        self.update(new_items)
        self.fastpoly.map_indices_inplace(mapfn)
        self._check_fast_polynomial()

    def mult(self, x):
        """
        Multiplies this polynomial by another polynomial `x`.

        Parameters
        ----------
        x : Polynomial
            The polynomial to multiply by.

        Returns
        -------
        Polynomial
            The polynomial representing self * x.
        """
        newpoly = Polynomial({}, self.max_num_vars)
        for k1, v1 in self.items():
            for k2, v2 in x.items():
                k = tuple(sorted(k1 + k2))
                if k in newpoly: newpoly[k] += v1 * v2
                else: newpoly[k] = v1 * v2

        newpoly.fastpoly = self.fastpoly.mult(x.fastpoly)
        self._check_fast_polynomial()
        newpoly._check_fast_polynomial()
        return newpoly

    def scale(self, x):
        """
        Scale this polynomial by `x` (multiply all coefficients by `x`).

        Parameters
        ----------
        x : float or complex
            The value to scale by.

        Returns
        -------
        None
        """
        # assume a scalar that can multiply values
        for k in tuple(self.keys()):  # I think the tuple() might speed things up (why?)
            self[k] *= x
        self.fastpoly.scale(x)
        self._check_fast_polynomial()

    def scalar_mult(self, x):
        """
        Multiplies this polynomial by a scalar `x`.

        Parameters
        ----------
        x : float or complex
            The value to multiply by.

        Returns
        -------
        Polynomial
        """
        newpoly = self.copy()
        newpoly.scale(x)
        self._check_fast_polynomial()
        newpoly._check_fast_polynomial()
        return newpoly

    def __str__(self):
        def fmt(x):
            if abs(_np.imag(x)) > 1e-6:
                if abs(_np.real(x)) > 1e-6: return "(%.3f+%.3fj)" % (x.real, x.imag)
                else: return "(%.3fj)" % x.imag
            else: return "%.3f" % x.real

        termstrs = []
        sorted_keys = sorted(list(self.keys()))
        for k in sorted_keys:
            varstr = ""; last_i = None; n = 1
            for i in sorted(k):
                if i == last_i: n += 1
                elif last_i is not None:
                    varstr += "x%d%s" % (last_i, ("^%d" % n) if n > 1 else "")
                    n = 1
                last_i = i
            if last_i is not None:
                varstr += "x%d%s" % (last_i, ("^%d" % n) if n > 1 else "")
            #print("DB: k = ",k, " varstr = ",varstr)
            if abs(self[k]) > 1e-4:
                termstrs.append("%s%s" % (fmt(self[k]), varstr))

        self._check_fast_polynomial()
        if len(termstrs) > 0:
            return " + ".join(termstrs)
        else: return "0"

    def __repr__(self):
        return "Poly[ " + str(self) + " ]"

    def __add__(self, x):
        newpoly = self.copy()
        if isinstance(x, Polynomial):
            for k, v in x.items():
                if k in newpoly: newpoly[k] += v
                else: newpoly[k] = v
            newpoly.fastpoly = self.fastpoly + x.fastpoly
        else:  # assume a scalar that can be added to values
            for k in newpoly:
                newpoly[k] += x
            newpoly.fastpoly = self.fastpoly + x
        self._check_fast_polynomial()
        newpoly._check_fast_polynomial()
        return newpoly

    def __iadd__(self, x):
        """ Does self += x more efficiently """
        if isinstance(x, Polynomial):
            for k, v in x.items():
                try:
                    self[k] += v
                except KeyError:
                    self[k] = v
            self.fastpoly += x.fastpoly
        else:  # assume a scalar that can be added to values
            for k in self:
                self[k] += x
            self.fastpoly += x
        self._check_fast_polynomial()
        return self

    def __mul__(self, x):
        #if isinstance(x, Polynomial):
        #    newpoly = Polynomial()
        #    for k1,v1 in self.items():
        #        for k2,v2 in x.items():
        #            k = tuple(sorted(k1+k2))
        #            if k in newpoly: newpoly[k] += v1*v2
        #            else: newpoly[k] = v1*v2
        #else:
        #    # assume a scalar that can multiply values
        #    newpoly = self.copy()
        #    for k in newpoly:
        #        newpoly[k] *= x
        #return newpoly
        if isinstance(x, Polynomial):
            ret = self.mult(x)
        else:  # assume a scalar that can multiply values
            ret = self.scalar_mult(x)
        self._check_fast_polynomial()
        ret._check_fast_polynomial()
        return ret

    def __rmul__(self, x):
        return self.__mul__(x)

    def __imul__(self, x):
        self._check_fast_polynomial()
        if isinstance(x, Polynomial):
            x._check_fast_polynomial()
            newcoeffs = {}
            for k1, v1 in self.items():
                for k2, v2 in x.items():
                    k = tuple(sorted(k1 + k2))
                    if k in newcoeffs: newcoeffs[k] += v1 * v2
                    else: newcoeffs[k] = v1 * v2
            self.clear()
            self.update(newcoeffs)
            self.fastpoly *= x.fastpoly
            self._check_fast_polynomial()
        else:
            self.scale(x)
        self._check_fast_polynomial()
        return self

    def __pow__(self, n):
        ret = Polynomial({(): 1.0}, self.max_num_vars)  # max_order updated by mults below
        cur = self
        for i in range(int(_np.floor(_np.log2(n))) + 1):
            rem = n % 2  # gets least significant bit (i-th) of n
            if rem == 1: ret *= cur  # add current power of x (2^i) if needed
            cur = cur * cur  # current power *= 2
            n //= 2  # shift bits of n right
        ret.fastpoly = self.fastpoly ** n
        ret._check_fast_polynomial()
        self._check_fast_polynomial()
        return ret

    def __copy__(self):
        ret = self.copy()
        ret._check_fast_polynomial()
        self._check_fast_polynomial()
        return ret

    def to_rep(self):
        """
        Construct a representation of this polynomial.

        "Representations" are lightweight versions of objects used to improve
        the efficiency of intensely computational tasks.  Note that Polynomial
        representations must have the same `max_order` and `max_num_vars` in
        order to interact with each other (add, multiply, etc.).

        Parameters
        ----------
        max_num_vars : int, optional
            The maximum number of variables the represenatation is allowed to
            have (x_0 to x_(`max_num_vars-1`)).  This sets the maximum allowed
            variable index within the representation.

        Returns
        -------
        PolynomialRep
        """
        # Set max_num_vars (determines based on coeffs if necessary)
        max_num_vars = self.max_num_vars
        default_max_vars = 0 if len(self) == 0 else \
            max([(max(k) + 1 if k else 0) for k in self.keys()])
        if max_num_vars is None:
            max_num_vars = default_max_vars
        else:
            assert(default_max_vars <= max_num_vars)

        vindices_per_int = Polynomial._vindices_per_int(max_num_vars)

        def vinds_to_int(vinds):
            """ Convert tuple index of ints to single int given max_numvars """
            ints_in_key = int(_np.ceil(len(vinds) / vindices_per_int))
            ret_tup = []
            for k in range(ints_in_key):
                ret = 0; m = 1
                for i in vinds[k * vindices_per_int:(k + 1) * vindices_per_int]:  # last tuple index is most significant
                    assert(i < max_num_vars), "Variable index exceed maximum!"
                    ret += (i + 1) * m
                    m *= max_num_vars + 1
                assert(ret >= 0), "vinds = %s -> %d!!" % (str(vinds), ret)
                ret_tup.append(ret)
            return tuple(ret_tup)

        int_coeffs = {vinds_to_int(k): v for k, v in self.items()}

        # (max_num_vars+1) ** vindices_per_int <= 2**PLATFORM_BITS, so:
        # vindices_per_int * log2(max_num_vars+1) <= PLATFORM_BITS
        vindices_per_int = int(_np.floor(PLATFORM_BITS / _np.log2(max_num_vars + 1)))
        self._check_fast_polynomial()

        return replib.PolynomialRep(int_coeffs, max_num_vars, vindices_per_int)


def bulk_load_compact_polynomials(vtape, ctape, keep_compact=False, max_num_vars=100):
    """
    Create a list of Polynomial objects from a "tape" of their compact versions.

    Parameters
    ----------
    vtape : numpy.ndarray
        A 1D array of variable indices that, together with `ctape`, specify an
        efficient means for evaluating a set of polynoials.

    ctape : numpy.ndarray
        A 1D array of coefficients that, together with `vtape`, specify an
       efficient means for evaluating a set of polynoials.

    keep_compact : bool, optional
        If True the returned list has elements which are (vtape,ctape) tuples
        for each individual polynomial.  If False, then the elements are
        :class:`Polynomial` objects.

    max_num_vars : int, optional
        The maximum number of variables the created polynomials
        are allowed to have.

    Returns
    -------
    list
        A list of Polynomial objects.
    """
    result = []
    c = 0; i = 0

    if keep_compact:
        while i < vtape.size:
            i2 = i  # increment i2 instead of i for this poly
            nTerms = vtape[i2]; i2 += 1
            for m in range(nTerms):
                nVars = vtape[i2]  # number of variable indices in this term
                i2 += nVars + 1
            result.append((vtape[i:i2], ctape[c:c + nTerms]))
            i = i2; c += nTerms
    else:
        while i < vtape.size:
            poly_coeffs = {}
            nTerms = vtape[i]; i += 1
            #print("POLY w/%d terms (i=%d)" % (nTerms,i))
            for m in range(nTerms):
                nVars = vtape[i]; i += 1  # number of variable indices in this term
                a = ctape[c]; c += 1
                #print("  TERM%d: %d vars, coeff=%s" % (m,nVars,str(a)))
                poly_coeffs[tuple(vtape[i:i + nVars])] = a; i += nVars
            result.append(Polynomial(poly_coeffs, max_num_vars))
    return result

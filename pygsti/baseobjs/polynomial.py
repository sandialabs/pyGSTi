"""
Defines the Polynomial class
"""
#***************************************************************************************************
# Copyright 2015, 2019, 2025 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

import platform as _platform

import numpy as _np

from pygsti.evotypes.basereps import PolynomialRep as _PolynomialRep

from typing import Optional, Sequence, Tuple, Union, Callable

assert(_platform.architecture()[0].endswith("bit"))  # e.g. "64bit"
PLATFORM_BITS = int(_platform.architecture()[0].strip("bit"))


def _vinds_to_int(vinds, vindices_per_int, max_num_vars):
    """ 
    Convert tuple index of ints to single int given max_numvars 
    
    """
    vinds = sorted(vinds)  # <-- canonicalize for commutative variables
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


class Polynomial(object):
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

    @staticmethod
    def _vindices_per_int(max_num_vars):
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

    @staticmethod
    def product(list_of_polys):
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
        return Polynomial.from_rep(rep)
    
    @staticmethod
    def sum(list_of_polys):
        """
        Take the sum of multiple polynomials.

        Parameters
        ----------
        list_of_polys : list
            List of polynomials to take the product of.

        Returns
        -------
        Polynomial
        """
        if not list_of_polys:
            return Polynomial({})
        max_num_vars = list_of_polys[0].max_num_vars
        assert all([max_num_vars == poly.max_num_vars for poly in list_of_polys])
        vindices_per_int = Polynomial._vindices_per_int(max_num_vars) 
        
        #initalize an empty PolynomialRep and accumulate into this.
        #TODO: There is a bug/incompatibility between the cython and python
        #versions of the PolynomalRep code. The python version accepts None for the int_coeff_dict
        #but cython doesn't. Making a new Polynomial is a workaround.
        newpoly = Polynomial({}, max_num_vars)
        rep = newpoly._rep
        for p in list_of_polys:
            rep.add_inplace(p._rep)
        return newpoly

    def __init__(self, coeffs=None, max_num_vars=100):
        """
        Initializes a new Polynomial object.

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
            The maximum number of variables the representation is allowed to
            have (x_0 to x_(`max_num_vars-1`)).  This sets the maximum allowed
            variable index within this polynomial.
        """
        vindices_per_int = Polynomial._vindices_per_int(max_num_vars)

        int_coeffs = {}
        for k, v in coeffs.items():
            ik = _vinds_to_int(k, vindices_per_int, max_num_vars)  # now sorts internally
            int_coeffs[ik] = int_coeffs.get(ik, 0) + v

        self._rep = _PolynomialRep(int_coeffs, max_num_vars, vindices_per_int)

    @classmethod
    def from_variable_and_coefficient_lists(cls, variables: Sequence[Tuple[int, ...]],
                                            coefficients: Sequence[complex],
                                            max_num_vars: int = 100) -> "Polynomial":
        """
        Construct a Polynomial from parallel lists of monomials and coefficients.

        Unlike the dict-based constructor, `variables` may list the same monomial
        more than once; repeated monomials have their coefficients summed.

        Parameters
        ----------
        variables : sequence of tuple of int
            Monomial terms, each a tuple of variable indices (e.g. `(0, 0, 1)` for
            `x0^2 x1`).  May contain repeats.

        coefficients : sequence of complex
            The coefficient multiplying each monomial in `variables`; must be the
            same length as `variables`.

        max_num_vars : int, optional
            The maximum number of variables the representation may have
            (x_0 to x_(`max_num_vars-1`)).

        Returns
        -------
        Polynomial
        """
        assert len(variables) == len(coefficients), "Iterable of variables and coefficients must have the same length."

        coeffs = {}
        for k, v in zip(variables, coefficients):
            coeffs[k] = coeffs.get(k, 0) + v
        return cls(coeffs, max_num_vars)

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
        has a single independent variable.

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

        return Polynomial(dcoeffs, self.max_num_vars)

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
            ret += coeff * _np.prod([variable_values[i] for i in ivar])
        return ret

    def compact(self, complex_coeff_tape=True):
        """
        Generate a compact form of this polynomial designed for fast evaluation.

        The resulting "tapes" can be evaluated using
        :func:`opcalc.bulk_eval_compact_polynomials`.

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
        return Polynomial.from_rep(self._rep.copy())

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
        return Polynomial({mapfn(k): v for k, v in self.coeffs.items()}, self.max_num_vars)

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
        vindices_per_int = self._rep.vindices_per_int
        max_num_vars = self._rep.max_num_vars
        new_int_coeffs = {}
        for k, v in self.coeffs.items():
            ik = _vinds_to_int(mapfn(k), vindices_per_int, max_num_vars)
            new_int_coeffs[ik] = new_int_coeffs.get(ik, 0) + v
        self._rep.reinit(new_int_coeffs)

    def mapvec_indices(self, mapvec):
        """
        Performs a bulk find & replace on this polynomial's variable indices.

        This function is similar to :meth:`map_indices` but uses a *vector*
        to describe *individual* index updates instead of a function for
        increased performance.

        Parameters
        ----------
        mapvec : numpy.ndarray
            An array whose i-th element gives the updated "new" index for
            the i-th variable.  Note that this vector maps *individual*
            variable indices old->new, whereas `mapfn` in :meth:`map_indices`
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

        This function is similar to :meth:`map_indices_inplace` but uses a *vector*
        to describe *individual* index updates instead of a function for increased
        performance.

        Parameters
        ----------
        mapvec : numpy.ndarray
            An array whose i-th element gives the updated "new" index for
            the i-th variable.  Note that this vector maps *individual*
            variable indices old->new, whereas `mapfn` in
            :meth:`map_indices_inplace` maps between *tuples* of indices.

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
        return Polynomial.from_rep(self._rep.mult(x._rep))

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

    def _format_var_label(self, i, var_labels: Optional[Union[dict, Sequence, Callable]]=None):
        """
        Format the label for variable index `i`.

        Parameters
        ----------
        i : int
            Variable index.

        var_labels : None, dict, sequence, or Callable
            Controls how variable indices are converted to strings:

            - None: use the default label ``x{i}``
            - dict: use ``var_labels[i]`` when present, otherwise ``x{i}``
            - sequence: use ``var_labels[i]`` when possible, otherwise ``x{i}``
            - Callable: use ``var_labels(i)``

        Returns
        -------
        str
        """
        if var_labels is None:
            return f"x{i}"
        try:
            if Callable(var_labels):
                return var_labels(i)
            elif isinstance(var_labels, dict):
                return var_labels.get(i, f"x{i}")
            else: #sequence
                return var_labels[i]
        except (IndexError, KeyError, TypeError):
            return f"x{i}"

    def to_string(self, var_labels: Optional[Union[dict, Sequence, Callable]]=None):
        """
        Construct a string representation of this polynomial.

        Parameters
        ----------
        var_labels : None, dict, sequence, or Callable, optional
            Controls how variable indices are converted to strings:

            - None: use default labels like ``x0``, ``x1``, ...
            - dict: map variable index to label
            - sequence: use the i-th element as the label for variable ``i``
            - Callable: a function taking an integer index and returning a label

        Returns
        -------
        str
        """
        def fmt(x):
            if abs(_np.imag(x)) > 1e-6:
                if abs(_np.real(x)) > 1e-6:
                    return f"({x.real:.3f}+{x.imag:.3f}j)"
                else:
                    return f"({x.imag:.3f}j)"
            else:
                return f"{x.real:.3f}"

        termstrs = []
        coeffs = self.coeffs
        sorted_keys = sorted(coeffs.keys())

        for k in sorted_keys:
            varstr = ""
            last_i = None
            n = 1

            for i in sorted(k):
                if i == last_i:
                    n += 1
                elif last_i is not None:
                    label = self._format_var_label(last_i, var_labels)
                    power = f"^{n}" if n > 1 else ""
                    varstr += f"{label}{power}"
                    n = 1
                last_i = i

            if last_i is not None:
                label = self._format_var_label(last_i, var_labels)
                power = f"^{n}" if n > 1 else ""
                varstr += f"{label}{power}"

            if abs(coeffs[k]) > 1e-4:
                termstrs.append(f"{fmt(coeffs[k])}{varstr}")

        return " + ".join(termstrs) if termstrs else "0"

    def __str__(self):
        return self.to_string()

    def __repr__(self):
        return "Poly[ " + self.__str__() + " ]"

    def __add__(self, x):
        newpoly = self.copy()
        if isinstance(x, Polynomial):
            newpoly._rep.add_inplace(x._rep)
        else:  # assume a scalar that can be added to values
            newpoly._rep.add_scalar_to_all_coeffs_inplace(x)
        return newpoly

    def __iadd__(self, x):
        """ Does self += x more efficiently """
        if isinstance(x, Polynomial):
            self._rep.add_inplace(x._rep)
        else:  # assume a scalar that can be added to values
            self._rep.add_scalar_to_all_coeffs_inplace(x)
        return self

    def __mul__(self, x):
        if isinstance(x, Polynomial):
            return self.mult(x)
        else:  # assume a scalar that can multiply values
            return self.scalar_mult(x)

    def __rmul__(self, x):
        return self.__mul__(x)

    def __pow__(self, n):
        ret = Polynomial({(): 1.0}, self.max_num_vars)  # max_order updated by mults below
        cur = self
        for i in range(int(_np.floor(_np.log2(n))) + 1):
            rem = n % 2  # gets least significant bit (i-th) of n
            if rem == 1: ret *= cur  # add current power of x (2^i) if needed
            cur = cur * cur  # current power *= 2
            n //= 2  # shift bits of n right
        return ret

    def __copy__(self):
        return self.copy()
    
    def __eq__(self, other):
        #TODO: Write a more efficient "rep-level" equality test. (will require updates in c-land).
        if not isinstance(other, Polynomial):
            return False
        return self.coeffs == other.coeffs
    
    def __hash__(self) -> int:
        raise RuntimeError('Polynomial objects are mutable. As such, they cannot be hashed.')

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


FASTPolynomial = Polynomial
# ^ That alias is deprecated and should be removed.


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


def compact_polynomial_list(list_of_polys):
    """
    Create a single vtape,ctape pair from a list of normal Polynomials

    Parameters
    ----------
    list_of_polys : list
        A list of :class:`Polynomial` objects.

    Returns
    -------
    vtape: numpy.ndarray
        A "tape" of the variable indices.

    ctape: numpy.ndarray
        A "tape" of the polynomial coefficients.
    """
    tapes = [p.compact() for p in list_of_polys]
    vtape = _np.concatenate([t[0] for t in tapes])
    ctape = _np.concatenate([t[1] for t in tapes])
    return vtape, ctape

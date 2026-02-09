"""
Base classes for representations.
"""
#***************************************************************************************************
# Copyright 2015, 2019, 2025 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

import math as _math

import numpy as _np


class POVMRep:
    """ The base class for all POVM representation classes """
    pass


try:
    from .basereps_cython import OpRep, StateRep, EffectRep, TermRep, PolynomialRep
except ImportError:
    # If cython is unavailable, just make a pure-python base class to fill in.
    class OpRep:
        """ The base class for all operation representation classes """
        pass

    class StateRep:
        """ The base class for all state representation classes """
        pass

    class EffectRep:
        """ The base class for all POVM effect representation classes """
        pass

    class TermRep:
        """ The base class for rank-1 term representation classes """
        pass

    class PolynomialRep(dict):
        """
        Representation class for a polynomial.

        This is similar to a full Polynomial
        dictionary, but lacks some functionality and is optimized for computation
        speed.  In particular, the keys of this dict are not tuples of variable
        indices (as in Polynomial) but simple integers encoded from such tuples.
        To perform this mapping, one must specify a maximum order and number of
        variables.
        """

        def __init__(self, int_coeff_dict, max_num_vars, vindices_per_int):
            """
            Create a new PolynomialRep object.

            Parameters
            ----------
            int_coeff_dict : dict
                A dictionary of coefficients whose keys are already-encoded
                integers corresponding to variable-index-tuples (i.e poly
                terms).

            max_num_vars : int
                The maximum number of variables allowed.  For example, if
                set to 2, then only "x0" and "x1" are allowed to appear
                in terms.
            """

            self.max_num_vars = max_num_vars
            self.vindices_per_int = vindices_per_int

            super(PolynomialRep, self).__init__()
            if int_coeff_dict is not None:
                self.update(int_coeff_dict)

        def reinit(self, int_coeff_dict):
            """
            Reinitialize this polynomial using new coefficients.

            Parameters
            ----------
            int_coeff_dict : dict
                The new coefficient dictionary to use in place of the current one. This
                dictionaries keys are "integerized" (by self._vinds_to_int) tuples of
                variable indices, and values are the polynomial coefficients themselves.
                Note that an "intergerized" tuple of variable indices is actually a *tuple*
                of integers, even through it acts as a single (extra long) integer.

            Returns
            -------
            None
            """
            self.clear()
            self.update(int_coeff_dict)

        def mapvec_indices_inplace(self, mapfn_as_vector):
            """
            Apply a given mapping vector (function) to all of the variable indices of this polynomial.

            This operation is performed in-place, updating the contents of this polynomial object.

            Parameters
            ----------
            mapfn_as_vector : numpy.ndarray
                An array of integers such that `mapfn_as_vector[old_variable_index] = new_variable_index`.
                An array is used instead of a function for perfomance.

            Returns
            -------
            None
            """
            new_items = {}
            for k, v in self.items():
                new_vinds = tuple((mapfn_as_vector[j] for j in self._int_to_vinds(k)))
                new_items[self._vinds_to_int(new_vinds)] = v
            self.clear()
            self.update(new_items)

        def copy(self):
            """
            Make a copy of this polynomial representation.

            Returns
            -------
            PolynomialRep
            """
            return PolynomialRep(self, self.max_num_vars, self.vindices_per_int)  # construct expects "int" keys

        def abs(self):
            """
            Return a polynomial whose coefficients are the absolute values of this PolynomialRep's coefficients.

            Returns
            -------
            PolynomialRep
            """
            result = {k: abs(v) for k, v in self.items()}
            return PolynomialRep(result, self.max_num_vars, self.vindices_per_int)

        @property
        def int_coeffs(self):  # so we can convert back to python Polys
            """ The coefficient dictionary (with encoded integer keys) """
            return dict(self)  # for compatibility w/C case which can't derive from dict...

        def _vinds_to_int(self, vinds):
            """ Maps tuple of variable indices to encoded int """
            ints_in_key = int(_np.ceil(len(vinds) / self.vindices_per_int))

            ret_tup = []
            for k in range(ints_in_key):
                ret = 0; m = 1
                # last tuple index is most significant
                for i in vinds[k * self.vindices_per_int:(k + 1) * self.vindices_per_int]:
                    assert(i < self.max_num_vars), "Variable index exceed maximum!"
                    ret += (i + 1) * m
                    m *= self.max_num_vars + 1
                assert(ret >= 0), "vinds = %s -> %d!!" % (str(vinds), ret)
                ret_tup.append(ret)
            return tuple(ret_tup)

        def _int_to_vinds(self, indx_tup):
            """ Maps encoded "int" to tuple of variable indices """
            ret = []
            #DB: cnt = 0; orig = indx
            for indx in indx_tup:
                while indx != 0:
                    nxt = indx // (self.max_num_vars + 1)
                    i = indx - nxt * (self.max_num_vars + 1)
                    ret.append(i - 1)
                    indx = nxt
                    #DB: cnt += 1
                    #DB: if cnt > 50:
                    #DB:    print("VINDS iter %d - indx=%d (orig=%d, nv=%d)" % (cnt,indx,orig,self.max_num_vars))
            return tuple(sorted(ret))

        def compact_complex(self):
            """
            Returns a compact representation of this polynomial as a
            `(variable_tape, coefficient_tape)` 2-tuple of 1D nupy arrays.
            The coefficient tape is *always* a complex array, even if
            none of the polynomial's coefficients are complex.

            Such compact representations are useful for storage and later
            evaluation, but not suited to polynomial manipulation.

            Returns
            -------
            vtape : numpy.ndarray
                A 1D array of integers (variable indices).
            ctape : numpy.ndarray
                A 1D array of *complex* coefficients.
            """
            nTerms = len(self)
            vinds = {i: self._int_to_vinds(i) for i in self.keys()}
            nVarIndices = sum(map(len, vinds.values()))
            vtape = _np.empty(1 + nTerms + nVarIndices, _np.int64)  # "variable" tape
            ctape = _np.empty(nTerms, complex)  # "coefficient tape"

            i = 0
            vtape[i] = nTerms; i += 1
            for iTerm, k in enumerate(sorted(self.keys())):
                v = vinds[k]  # so don't need to compute self._int_to_vinds(k)
                l = len(v)
                ctape[iTerm] = self[k]
                vtape[i] = l; i += 1
                vtape[i:i + l] = v; i += l
            assert(i == len(vtape)), "Logic Error!"
            return vtape, ctape

        def compact_real(self):
            """
            Returns a real representation of this polynomial as a
            `(variable_tape, coefficient_tape)` 2-tuple of 1D nupy arrays.
            The coefficient tape is *always* a complex array, even if
            none of the polynomial's coefficients are complex.

            Such compact representations are useful for storage and later
            evaluation, but not suited to polynomial manipulation.

            Returns
            -------
            vtape : numpy.ndarray
                A 1D array of integers (variable indices).
            ctape : numpy.ndarray
                A 1D array of *real* coefficients.
            """
            nTerms = len(self)
            vinds = {i: self._int_to_vinds(i) for i in self.keys()}
            nVarIndices = sum(map(len, vinds.values()))
            vtape = _np.empty(1 + nTerms + nVarIndices, _np.int64)  # "variable" tape
            ctape = _np.empty(nTerms, complex)  # "coefficient tape"

            i = 0
            vtape[i] = nTerms; i += 1
            for iTerm, k in enumerate(sorted(self.keys())):
                v = vinds[k]  # so don't need to compute self._int_to_vinds(k)
                l = len(v)
                ctape[iTerm] = self[k]
                vtape[i] = l; i += 1
                vtape[i:i + l] = v; i += l
            assert(i == len(vtape)), "Logic Error!"
            return vtape, ctape

        def mult(self, x):
            """
            Returns `self * x` where `x` is another polynomial representation.

            Parameters
            ----------
            x : PolynomialRep

            Returns
            -------
            PolynomialRep
            """
            assert(self.max_num_vars == x.max_num_vars)
            newpoly = PolynomialRep(None, self.max_num_vars, self.vindices_per_int)
            for k1, v1 in self.items():
                for k2, v2 in x.items():
                    inds = sorted(self._int_to_vinds(k1) + x._int_to_vinds(k2))
                    k = newpoly._vinds_to_int(inds)
                    if k in newpoly: newpoly[k] += v1 * v2
                    else: newpoly[k] = v1 * v2
            assert(newpoly.degree <= self.degree + x.degree)
            return newpoly

        def scale(self, x):
            """
            Performs `self = self * x` where `x` is a scalar.

            Parameters
            ----------
            x : float or complex

            Returns
            -------
            None
            """
            # assume a scalar that can multiply values
            for k in self:
                self[k] *= x

        def add_inplace(self, other):
            """
            Adds `other` into this PolynomialRep.

            Parameters
            ----------
            other : PolynomialRep

            Returns
            -------
            PolynomialRep
            """
            for k, v in other.items():
                try:
                    self[k] += v
                except KeyError:
                    self[k] = v
            return self

        def add_scalar_to_all_coeffs_inplace(self, x):
            """
            Adds `x` to all of the coefficients in this PolynomialRep.

            Parameters
            ----------
            x : float or complex

            Returns
            -------
            PolynomialRep
            """
            for k in self:
                self[k] += x
            return self

        def __str__(self):
            def fmt(x):
                if abs(_np.imag(x)) > 1e-6:
                    if abs(_np.real(x)) > 1e-6: return "(%.3f+%.3fj)" % (x.real, x.imag)
                    else: return "(%.3fj)" % x.imag
                else: return "%.3f" % x.real

            termstrs = []
            sorted_keys = sorted(list(self.keys()))
            for k in sorted_keys:
                vinds = self._int_to_vinds(k)
                varstr = ""; last_i = None; n = 0
                for i in sorted(vinds):
                    if i == last_i: n += 1
                    elif last_i is not None:
                        varstr += "x%d%s" % (last_i, ("^%d" % n) if n > 1 else "")
                    last_i = i
                if last_i is not None:
                    varstr += "x%d%s" % (last_i, ("^%d" % n) if n > 1 else "")
                #print("DB: vinds = ",vinds, " varstr = ",varstr)
                if abs(self[k]) > 1e-4:
                    termstrs.append("%s%s" % (fmt(self[k]), varstr))
            if len(termstrs) > 0:
                return " + ".join(termstrs)
            else: return "0"

        def __repr__(self):
            return "PolynomialRep[ " + str(self) + " ]"

        @property
        def degree(self):
            """ Used for debugging in slowreplib routines only"""
            return max([len(self._int_to_vinds(k)) for k in self.keys()])


# Other classes
LARGE = 1000000000
# a large number such that LARGE is
# a very high term weight which won't help (at all) a
# path get included in the selected set of paths.

SMALL = 1e-5
# a number which is used in place of zero within the
# product of term magnitudes to keep a running path
# magnitude from being zero (and losing memory of terms).


class StockTermRep(TermRep):
    """
    A basic term representation that just holds other representation types (polys, states, effects, and gates).

    This "stock" class is in many cases entirely sufficient of an evotype, and is used by
    default when an evotype doesn't define its own term-representation types so that evotypes
    don't need to define term-rep types unless they're doing something that is non-standard.

    TODO: rest of StockTermRep docstring
    """
    # just a container for other reps (polys, states, effects, and gates)

    @classmethod
    def composed(cls, terms_to_compose, magnitude):
        logmag = _math.log10(magnitude) if magnitude > 0 else -LARGE
        first = terms_to_compose[0]
        coeffrep = first.coeff
        pre_ops = first.pre_ops[:]
        post_ops = first.post_ops[:]
        for t in terms_to_compose[1:]:
            coeffrep = coeffrep.mult(t.coeff)
            pre_ops += t.pre_ops
            post_ops += t.post_ops
        return StockTermRep(coeffrep, magnitude, logmag, first.pre_state, first.post_state,
                            first.pre_effect, first.post_effect, pre_ops, post_ops)

    def __init__(self, coeff, mag, logmag, pre_state, post_state,
                 pre_effect, post_effect, pre_ops, post_ops):
        self.coeff = coeff
        self.magnitude = mag
        self.logmagnitude = logmag
        self.pre_state = pre_state
        self.post_state = post_state
        self.pre_effect = pre_effect
        self.post_effect = post_effect
        self.pre_ops = pre_ops
        self.post_ops = post_ops

    def set_magnitude(self, mag):
        self.magnitude = mag
        self.logmagnitude = _math.log10(mag) if mag > 0 else -LARGE

    def set_magnitude_only(self, mag):
        self.magnitude = mag

    def mapvec_indices_inplace(self, mapvec):
        self.coeff.mapvec_indices_inplace(mapvec)

    def scalar_mult(self, x):
        coeff = self.coeff.copy()
        coeff.scale(x)
        return StockTermRep(coeff, self.magnitude, self.logmagnitude,
                            self.pre_state, self.post_state, self.pre_effect, self.post_effect,
                            self.pre_ops, self.post_ops)

    def copy(self):
        return StockTermRep(self.coeff.copy(), self.magnitude, self.logmagnitude,
                            self.pre_state, self.post_state, self.pre_effect, self.post_effect,
                            self.pre_ops, self.post_ops)


StockTermDirectRep = StockTermRep

import numpy as np

import platform as _platform
assert(_platform.architecture()[0].endswith("bit"))  # e.g. "64bit"
PLATFORM_BITS = int(_platform.architecture()[0].strip("bit"))

from pygsti.baseobjs.opcalc import compact_deriv
from pygsti.baseobjs import polynomial as poly
from pygsti.evotypes.basereps import PolynomialRep as _PolynomialRep
from ..util import BaseCase


# TODO: use this class to faciliate unit tests. Right now it's just
# sitting here.
class SLOWPolynomial(dict):
    """
    This reference class provides an alternative implementation of poly.Polynomial.

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
        return int(np.floor(PLATFORM_BITS / np.log2(max_num_vars + 1)))

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
        ret.fastpoly = poly.Polynomial.from_rep(rep)
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
        super(SLOWPolynomial, self).__init__()
        if coeffs is not None:
            self.update(coeffs)
        self.max_num_vars = max_num_vars
        self.fastpoly = poly.Polynomial(coeffs, max_num_vars)
        self._check_fast_polynomial()

    def _check_fast_polynomial(self, raise_err=True):
        """
        Check that included poly.Polynomial has remained in-sync with this one.

        This is purely for debugging, to ensure that the poly.Polynomial
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
            if not np.isclose(self.fastpoly.coeffs[k], self[k]):
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

        ret = SLOWPolynomial(dcoeffs, self.max_num_vars)
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
            ret += coeff * np.prod([variable_values[i] for i in ivar])
        assert(np.isclose(ret, self.fastpoly.evaluate(variable_values)))
        self._check_fast_polynomial()
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
        #if force_complex:
        #    iscomplex = True
        #else:
        #    iscomplex = any([abs(np.imag(x)) > 1e-12 for x in self.values()])
        iscomplex = complex_coeff_tape

        nTerms = len(self)
        nVarIndices = sum(map(len, self.keys()))
        vtape = np.empty(1 + nTerms + nVarIndices, np.int64)  # "variable" tape
        ctape = np.empty(nTerms, complex if iscomplex else 'd')  # "coefficient tape"

        i = 0
        vtape[i] = nTerms; i += 1
        for iTerm, k in enumerate(sorted(self.keys())):
            l = len(k)
            ctape[iTerm] = self[k] if iscomplex else np.real(self[k])
            vtape[i] = l; i += 1
            vtape[i:i + l] = k; i += l
        assert(i == len(vtape)), "Logic Error!"
        fast_vtape, fast_ctape = self.fastpoly.compact(iscomplex)
        assert(np.allclose(fast_vtape, vtape) and np.allclose(fast_ctape, ctape))
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
        ret = SLOWPolynomial(self, self.max_num_vars)
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
        ret = SLOWPolynomial({mapfn(k): v for k, v in self.items()}, self.max_num_vars)
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
        newpoly = SLOWPolynomial({}, self.max_num_vars)
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
            if abs(np.imag(x)) > 1e-6:
                if abs(np.real(x)) > 1e-6: return "(%.3f+%.3fj)" % (x.real, x.imag)
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
        if isinstance(x, SLOWPolynomial):
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
        if isinstance(x, SLOWPolynomial):
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
        #if isinstance(x, SLOWPolynomial):
        #    newpoly = SLOWPolynomial()
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
        if isinstance(x, SLOWPolynomial):
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
        if isinstance(x, SLOWPolynomial):
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
        ret = SLOWPolynomial({(): 1.0}, self.max_num_vars)  # max_order updated by mults below
        cur = self
        for i in range(int(np.floor(np.log2(n))) + 1):
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

        vindices_per_int = SLOWPolynomial._vindices_per_int(max_num_vars)

        def vinds_to_int(vinds):
            """ Convert tuple index of ints to single int given max_numvars """
            ints_in_key = int(np.ceil(len(vinds) / vindices_per_int))
            ret_tup = []
            for k in range(ints_in_key):
                ret = 0; m = 1
                for i in vinds[k * vindices_per_int:(k + 1) * vindices_per_int]:  # last tuple index=most significant
                    assert(i < max_num_vars), "Variable index exceed maximum!"
                    ret += (i + 1) * m
                    m *= max_num_vars + 1
                assert(ret >= 0), "vinds = %s -> %d!!" % (str(vinds), ret)
                ret_tup.append(ret)
            return tuple(ret_tup)

        int_coeffs = {vinds_to_int(k): v for k, v in self.items()}

        # (max_num_vars+1) ** vindices_per_int <= 2**PLATFORM_BITS, so:
        # vindices_per_int * log2(max_num_vars+1) <= PLATFORM_BITS
        vindices_per_int = int(np.floor(PLATFORM_BITS / np.log2(max_num_vars + 1)))
        self._check_fast_polynomial()

        return _PolynomialRep(int_coeffs, max_num_vars, vindices_per_int)


class CompactPolynomialTester(BaseCase):
    def test_compact_polys(self):
        # TODO break apart
        p = poly.Polynomial({(): 1.0, (1, 2): 2.0, (1, 1, 2): 3.0})
        v, c = p.compact()
        self.assertArraysAlmostEqual(v, np.array([3, 0, 2, 1, 2, 3, 1, 1, 2]))
        self.assertArraysAlmostEqual(c, np.array([1.0, 2.0, 3.0]))
        # 3x1^2 x2 + 2 x1x2 + 3

        q = poly.Polynomial({(): 4.0, (1, 1): 5.0, (2, 2, 3): 6.0})
        v2, c2 = q.compact()
        self.assertArraysAlmostEqual(v2, np.array([3, 0, 2, 1, 1, 3, 2, 2, 3]))
        self.assertArraysAlmostEqual(c2, np.array([4.0, 5.0, 6.0]))
        # 6x2^2 x3 + 5 x1^2 + 4

        v = np.concatenate((v, v2)).astype(np.int64)
        c = np.concatenate((c, c2))
        c = np.ascontiguousarray(c, complex)

        vout, cout = compact_deriv(v, c, np.array([1, 2, 3], dtype=np.int64))
        compact_polys = poly.bulk_load_compact_polynomials(vout, cout, keep_compact=True)

        def assertCompactPolysEqual(vctups1, vctups2):
            for (v1, c1), (v2, c2) in zip(vctups1, vctups2):
                self.assertArraysAlmostEqual(v1, v2)  # integer arrays
                self.assertArraysAlmostEqual(c1, c2)  # complex arrays

        assertCompactPolysEqual(compact_polys,
                                ((np.array([2, 1, 2, 2, 1, 2]), np.array([2. + 0.j, 6. + 0.j])),
                                 (np.array([2, 1, 1, 2, 1, 1]), np.array([2. + 0.j, 3. + 0.j])),
                                    (np.array([0]), np.array([], dtype=np.complex128)),
                                    (np.array([1, 1, 1]), np.array([10. + 0.j])),
                                    (np.array([1, 2, 2, 3]), np.array([12. + 0.j])),
                                    (np.array([1, 2, 2, 2]), np.array([6. + 0.j]))))

        polys = poly.bulk_load_compact_polynomials(vout, cout)
        self.assertEqual(str(polys[0]), "6.000x1x2 + 2.000x2")
        self.assertEqual(str(polys[1]), "2.000x1 + 3.000x1^2")
        self.assertEqual(str(polys[2]), "0")
        self.assertEqual(str(polys[3]), "10.000x1")
        self.assertEqual(str(polys[4]), "12.000x2x3")
        self.assertEqual(str(polys[5]), "6.000x2^2")

        self.assertEqual(list(vout), [2, 1, 2, 2, 1, 2, 2, 1, 1, 2, 1, 1, 0, 1, 1, 1, 1, 2, 2, 3, 1, 2, 2, 2])
        self.assertEqual(list(cout), [ 2.+0.j,  6.+0.j,  2.+0.j,  3.+0.j, 10.+0.j, 12.+0.j,  6.+0.j])

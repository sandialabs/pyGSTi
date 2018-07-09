""" Defines the Polynomial class """
from __future__ import division, print_function, absolute_import, unicode_literals
#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************

import numpy as _np
import collections as _collections
try:
    from . import fastreplib as replib
except ImportError:
    from . import replib


class Polynomial(dict):
    """ 
    Encapsulates a polynomial as a subclass of the standard Python dict.

    Variables are represented by integer indices, e.g. "2" means "x_2".
    Keys are tuples of variable indices and values are numerical 
    coefficients (floating point or complex numbers).  To specify a variable
    to some power, its index is repeated in the key-tuple.

    E.g. x_0^2 + 3*x_1 + 4 is stored as {(0,0): 1.0, (1,): 3.0, (): 4.0}
    """

    @classmethod
    def fromrep(cls, rep):
        """
        Creates a Polynomial from a "representation" (essentially a
        lite-version) of a Polynomial.

        Note: usually we only need to convert from full-featured Python objects
        to the lighter-weight "representation" objects.  Polynomials are an
        exception, since as the results of probability computations they need 
        to be converted back from "representation-form" to "full-form".

        Parameters
        ----------
        rep : PolyRep
            A polynomial representation.

        Returns
        -------
        Polynomial
        """
        max_num_vars = rep.max_num_vars  # one of the few/only cases where a rep
        max_order = rep.max_order        # needs to expose some python properties

        def int_to_vinds(indx):
            ret = []
            while indx != 0:
                nxt = indx // (max_num_vars+1)
                i = indx - nxt*(max_num_vars+1)
                ret.append(i-1)
                indx = nxt
            assert(len(ret) <= max_order)
            return tuple(sorted(ret))
        
        tup_coeff_dict = { int_to_vinds(k): val for k,val in rep.coeffs.items() }
        return cls(tup_coeff_dict)

    
    def __init__(self, coeffs=None):
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
        """
        super(Polynomial,self).__init__()
        if coeffs is not None:
            self.update(coeffs)
            
    def deriv(self, wrtParam):
        """
        Take the derivative of this Polynomial with respect to a single
        variable/parameter.  The result is another Polynomial.

        E.g. deriv(x_2^3 + 3*x_1, wrtParam=2) = 3x^2

        Parameters
        ----------
        wrtParam : int
            The variable index to differentiate with respect to. 
            E.g. "4" means "differentiate w.r.t. x_4".

        Returns
        -------
        Polynomial
        """
        dcoeffs = {}
        for ivar, coeff in self.items():
            cnt = float(ivar.count(wrtParam))
            if cnt > 0:
                l = list(ivar)
                del l[l.index(wrtParam)]
                dcoeffs[ tuple(l) ] = cnt * coeff

        return Polynomial(dcoeffs)

    def get_degree(self):
        """
        Return the largest sum-of-exponents for any term (monomial) within this
        polynomial. E.g. for x_2^3 + x_1^2*x_0^2 has degree 4.
        """
        return 0 if len(self)==0 else max([len(k) for k in self.keys()])

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
        for ivar,coeff in self.items():
            ret += coeff * _np.product( [variable_values[i] for i in ivar] )
        return ret

    def compact(self, force_complex=False):
        """
        Generate a compact form of this polynomial designed for fast evaluation.

        The resulting "tapes" can be evaluated using 
        :function:`bulk_eval_compact_polys`.

        Parameters
        ----------
        force_complex : bool, optional
            Whether the `ctape` returned array is forced to be of complex type,
            even if all of the polynomial coefficients are real.

        Returns
        -------
        vtape, ctape : numpy.ndarray
            These two 1D arrays specify an efficient means for evaluating this
            polynomial.
        """
        if force_complex:
            iscomplex = True
        else:
            iscomplex = any([ abs(_np.imag(x)) > 1e-12 for x in self.values() ])
            
        nTerms = len(self)
        nVarIndices = sum(map(len,self.keys()))
        vtape = _np.empty(1 + nTerms + nVarIndices, _np.int64) # "variable" tape
        ctape = _np.empty(nTerms, complex if iscomplex else 'd') # "coefficient tape"

        i = 0
        vtape[i] = nTerms; i+=1
        for iTerm,k in enumerate(sorted(self.keys())):
            l = len(k)
            ctape[iTerm] = self[k] if iscomplex else _np.real(self[k])
            vtape[i] = l; i += 1
            vtape[i:i+l] = k; i += l
        assert(i == len(vtape)), "Logic Error!"
        return vtape, ctape

    def copy(self):
        """
        Returns a copy of this polynomial.
        """
        return Polynomial(self)

    def map_indices(self, mapfn):
        """
        Performs a bulk find & replace on this polynomial's variable indices.

        This is useful when the variable indices have external significance
        (like being the indices of a gate's parameters) and one want to convert
        to another set of indices (like a parent gate set's parameters).

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
        new_items = { mapfn(k): v for k,v in self.items() }
        self.clear()
        self.update(new_items)

    def mult(self,x):
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
        newpoly = Polynomial()
        for k1,v1 in self.items():
            for k2,v2 in x.items():
                k = tuple(sorted(k1+k2))
                if k in newpoly: newpoly[k] += v1*v2
                else: newpoly[k] = v1*v2
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
        for k in tuple(self.keys()): # I think the tuple() might speed things up (why?)
            self[k] *= x

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
        sorted_keys = sorted(list(self.keys()))
        for k in sorted_keys:
            varstr = ""; last_i = None; n=1
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
                termstrs.append( "%s%s" % (fmt(self[k]), varstr) )
        if len(termstrs) > 0:
            return " + ".join(termstrs)
        else: return "0"

    def __repr__(self):
        return "Poly[ " + str(self) + " ]"

    def __add__(self,x):
        newpoly = self.copy()
        if isinstance(x, Polynomial):
            for k,v in x.items():
                if k in newpoly: newpoly[k] += v
                else: newpoly[k] = v
        else: # assume a scalar that can be added to values
            for k in newpoly:
                newpoly[k] += x
        return newpoly

    def __iadd__(self,x):
        """ Does self += x more efficiently """
        if isinstance(x, Polynomial):
            for k,v in x.items():
                try:
                    self[k] += v
                except KeyError:
                    self[k] = v
        else: # assume a scalar that can be added to values
            for k in self:
                self[k] += x
        return self

    def __mul__(self,x):
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
            return self.mult(x)
        else: # assume a scalar that can multiply values
            return self.scalar_mult(x)

    def __rmul__(self, x):
        return self.__mul__(x)

    def __pow__(self,n):
        ret = Polynomial({(): 1.0}) # max_order updated by mults below
        cur = self
        for i in range(int(np.floor(np.log2(n)))+1):
            rem = n % 2 #gets least significant bit (i-th) of n
            if rem == 1: ret *= cur # add current power of x (2^i) if needed  
            cur = cur*cur # current power *= 2
            n //= 2 # shift bits of n right 
        return ret

    def __copy__(self):
        return self.copy()

    def torep(self, max_order=None, max_num_vars=None):
        """
        Construct a representation of this polynomial.

        "Representations" are lightweight versions of objects used to improve
        the efficiency of intensely computational tasks.  Note that Polynomial
        representations must have the same `max_order` and `max_num_vars` in
        order to interact with each other (add, multiply, etc.).
        
        Parameters
        ----------
        max_order : int, optional
            The maximum order (degree) terms are allowed to have.  If None,
            then it is taken as the current degree of this polynomial.
        
        max_num_vars : int, optional
            The maximum number of variables the represenatation is allowed to
            have (x_0 to x_(`max_num_vars-1`)).  This sets the maximum allowed
            variable index within the representation.

        Returns
        -------
        PolyRep
        """
        # Set max_order (determines based on coeffs if necessary)
        default_max_order = self.get_degree()
        if max_order is None:
            max_order = default_max_order
        else:
            assert(default_max_order <= max_order)

        # Set max_num_vars (determines based on coeffs if necessary)
        default_max_vars = 0 if len(self) == 0 else \
                           max([ (max(k)+1 if k else 0) for k in self.keys()])
        if max_num_vars is None:
            max_num_vars = default_max_vars
        else:
            assert(default_max_vars <= max_num_vars)

        #new.max_order = max_order            
        #new.max_num_vars = max_num_vars
        def vinds_to_int(vinds):
            """ Convert tuple index of ints to single int given max_order,max_numvars """
            assert(len(vinds) <= max_order), "max_order is too low!"
            ret = 0; m = 1
            for i in vinds: # last tuple index is most significant                                                                                                          
                assert(i < max_num_vars), "Variable index exceed maximum!"
                ret += (i+1)*m
                m *= max_num_vars+1
            return ret
        
        int_coeffs = { vinds_to_int(k): v for k,v in self.items() }
        return replib.PolyRep(int_coeffs, max_order, max_num_vars)
    

#OLD: TODO REMOVE
#class SLOWPolynomial(object):
#    """ Encapsulates a polynomial """
#    def __init__(self, coeffs=None,is_complex=True):
#        """ TODO: docstring - coeffs is a dict of coefficients w/keys == tuples
#             of integer variable indices.  E.g. (1,1) means "variable1 squared"
#        """
#        if coeffs is None:
#            self.coeffs = _np.zeros(0,complex if is_complex else 'd')
#            self.inds = []
#        else:
#            self.inds = sorted(list(coeffs.keys()))
#            self.coeffs = _np.array([coeffs[k] for k in self.inds],complex if is_complex else 'd')
#            
#    def deriv(self, wrtParam):
#        dcoeffs = {}
#        for ivar,coeff in zip(self.inds,self.coeffs):
#            cnt = float(ivar.count(wrtParam))
#            if cnt > 0:
#                l = list(ivar)
#                del l[ivar.index(wrtParam)]
#                dcoeffs[ tuple(l) ] = cnt * coeff
#
#        return Polynomial(dcoeffs) # returns another polynomial
#
#    def evaluate(self, variable_values):
#        """ TODO: docstring -- and make this function smarter (Russian peasant) """
#        ret = 0
#        for ivar,coeff in zip(self.inds,self.coeffs):
#            ret += coeff * _np.product( [variable_values[i] for i in ivar] )
#        return ret
#
#    def compact(self):
#        """ TODO docstring Returns compact representation of (vtape, ctape) 1D nupy arrays """
#        iscomplex = bool(_np.linalg.norm(_np.imag(self.coeffs)) > 1e-12)
#        nTerms = len(self.inds)
#        nVarIndices = sum(map(len,self.inds))
#        vtape = _np.empty(1 + nTerms + nVarIndices, _np.int64) # "variable" tape
#        ctape = _np.empty(nTerms, complex if iscomplex else 'd') # "coefficient tape"
#
#        i = 0
#        vtape[i] = nTerms; i+=1
#        for iTerm,k in enumerate(self.inds):
#            l = len(k)
#            ctape[iTerm] = self.coeffs[iTerm] if iscomplex else _np.real(self.coeffs[iTerm])
#            vtape[i] = l; i += 1
#            vtape[i:i+l] = k; i += l
#        assert(i == len(vtape)), "Logic Error!"
#        return vtape, ctape
#
#    def copy(self):
#        cpy = Polynomial()
#        cpy.coeffs = self.coeffs.copy()
#        cpy.inds = self.inds[:]
#        return cpy
#
#    def map_indices(self, mapfn):
#        """ TODO: docstring - mapfn should map old->new variable-index-tuples """
#        new_coeff_dict = { mapfn(k): c for k,c in zip(self.inds,self.coeffs) }
#        self.inds = sorted(list(new_coeff_dict.keys()))
#        self.coeffs = _np.array([new_coeff_dict[k] for k in self.inds], self.coeffs.dtype)
#
#    def mult_poly(self,x):
#        """ Does self * x where x is a polynomial """
#        coeff_dict = {}
#        for k1,v1 in zip(self.inds,self.coeffs):
#            for k2,v2 in zip(x.inds,x.coeffs):
#                k = tuple(sorted(k1+k2))
#                if k in coeff_dict: coeff_dict[k] += v1*v2
#                else: coeff_dict[k] = v1*v2
#        return Polynomial(coeff_dict)
#
#    def mult_scalar(self, x):
#        # assume a scalar that can multiply values
#        newpoly = self.copy()
#        newpoly.coeffs *= x
#        return newpoly
#
#    #def multin_scalar(self, x):
#    #    """ self *= scalar """
#    #    # assume a scalar that can multiply values
#    #    self.coeffs *= x
#
#    def __str__(self):
#        def fmt(x):
#            if abs(_np.imag(x)) > 1e-6:
#                if abs(_np.real(x)) > 1e-6: return "(%.3f+%.3fj)" % (x.real, x.imag)
#                else: return "(%.3fj)" % x.imag
#            else: return "%.3f" % x.real
#            
#        termstrs = []
#        for ik,k in enumerate(self.inds):
#            varstr = ""; last_i = None; n=0
#            for i in sorted(k):
#                if i == last_i: n += 1
#                elif last_i is not None:
#                    varstr += "x%d%s" % (last_i, ("^%d" % n) if n > 1 else "")
#                last_i = i
#            if last_i is not None:
#                varstr += "x%d%s" % (last_i, ("^%d" % n) if n > 1 else "")
#            #print("DB: k = ",k, " varstr = ",varstr)
#            if abs(self.coeffs[ik]) > 1e-4:
#                termstrs.append( "%s%s" % (fmt(self.coeffs[ik]), varstr) )
#        return " + ".join(termstrs)
#
#    def __repr__(self):
#        return "Poly[ " + str(self) + " ]"
#
#    def __add__(self,x):
#        newinds = []
#        newcoeffs = [] # a list for now
#        
#        i1 = i2 = 0 # pointers to self & x, respectively
#        it1 = iter(self.inds)
#        it2 = iter(x.inds)
#
#        try:
#            vinds1 = next(it1)
#            vinds2 = next(it2)
#            while True:
#                if vinds1 == vinds2:
#                    s = self.coeffs[i1] + x.coeffs[i2]
#                    if abs(s) > 1e-12:
#                        newinds.append(vinds1)
#                        newcoeffs.append(s)
#                    i1 += 1; i2 += 1
#                    vinds1 = next(it1)
#                    vinds2 = next(it2)
#                elif vinds1 < vinds2:
#                    newinds.append(vinds1)
#                    newcoeffs.append(self.coeffs[i1])
#                    i1 += 1; vinds1 = next(it1)
#                else:
#                    newinds.append(vinds2)
#                    newcoeffs.append(x.coeffs[i2])
#                    i2 += 1; vinds2 = next(it2)
#        except StopIteration: pass
#        
#        if i1 < len(self.inds):
#            newinds.extend(self.inds[i1:])
#            newcoeffs.extend(self.coeffs[i1:])
#        if i2 < len(x.inds):
#            newinds.extend(x.inds[i2:])
#            newcoeffs.extend(x.coeffs[i2:])
#
#        newpoly = Polynomial()
#        newpoly.inds = newinds
#        newpoly.coeffs = _np.array(newcoeffs, self.coeffs.dtype)
#        return newpoly
#                
#
#    def __mul__(self,x):
#        if isinstance(x, Polynomial):
#            return self.mult_poly(x)
#        else: # assume a scalar that can multiply values
#            return self.mult_scalar(x)
#
#    def __rmul__(self, x):
#        return self.__mul__(x)
#
#    def __pow__(self,n):
#        ret = Polynomial({(): 1.0}) 
#        cur = self
#        for i in range(int(np.floor(np.log2(n)))+1):
#            rem = n % 2 #gets least significant bit (i-th) of n
#            if rem == 1: ret *= cur # add current power of x (2^i) if needed  
#            cur = cur*cur # current power *= 2
#            n //= 2 # shift bits of n right 
#        return ret
#
#    def __copy__(self):
#        return self.copy()


def bulk_eval_compact_polys(vtape, ctape, paramvec, dest_shape):
    """
    Evaluate many compact polynomial forms at a given set of variable values.

    Parameters
    ----------
    vtape, ctape : numpy.ndarray
        Specifies "variable" and "coefficient" 1D numpy arrays to evaluate.
        These "tapes" can be generated by concatenating the tapes of individual
        complact-polynomial tuples returned by :method:`Polynomial.compact`.

    paramvec : array-like
        An object that can be indexed so that `paramvec[i]` gives the
        numerical value to substitute for i-th polynomial variable (x_i).

    dest_shape : tuple
        The shape of the final array of evaluated polynomials.  The resulting
        1D array of evaluated polynomials is reshaped accordingly.
    
    Returns
    -------
    numpy.ndarray
        An array of the same type as the coefficient tape, with shape given
        by `dest_shape`.
    """
    result = _np.empty(dest_shape,ctape.dtype) # auto-determine type?
    res = result.flat # for 1D access
    
    c = 0; i = 0; r = 0
    while i < vtape.size:
        poly_val = 0
        nTerms = vtape[i]; i+=1
        #print("POLY w/%d terms (i=%d)" % (nTerms,i))
        for m in range(nTerms):
            nVars = vtape[i]; i+=1 # number of variable indices in this term
            a = ctape[c]; c+=1
            #print("  TERM%d: %d vars, coeff=%s" % (m,nVars,str(a)))
            for k in range(nVars):
                a *= paramvec[ vtape[i] ]; i+=1
            poly_val += a
            #print("  -> added %s to poly_val = %s" % (str(a),str(poly_val))," i=%d, vsize=%d" % (i,vtape.size))
        res[r] = poly_val; r+=1
    assert(c == ctape.size),"Coeff Tape length error: %d != %d !" % (c,ctape.size)
    assert(r == result.size),"Result/Tape size mismatch: only %d result entries filled!" % r
    return result


def bulk_load_compact_polys(vtape, ctape, keep_compact=False):
    """
    Create a list of Polynomial objects from a "tape" of their compact versions.

    Parameters
    ----------
    vtape, ctape : numpy.ndarray
        Specifies "variable" and "coefficient" 1D numpy arrays to load.
        These "tapes" can be generated by concatenating the tapes of individual
        complact-polynomial tuples returned by :method:`Polynomial.compact`.

    keep_compact : bool, optional
        If True the returned list has elements which are (vtape,ctape) tuples
        for each individual polynomial.  If False, then the elements are
        :class:`Polynomial` objects.

    Returns
    -------
    list
    """
    result = []
    c = 0; i = 0
    
    if keep_compact:
        while i < vtape.size:
            i2 = i # increment i2 instead of i for this poly
            nTerms = vtape[i2]; i2+=1
            for m in range(nTerms):
                nVars = vtape[i2] # number of variable indices in this term
                i2 += nVars + 1
            result.append( (vtape[i:i2], ctape[c:c+nTerms]) )
            i = i2; c += nTerms
    else:
        while i < vtape.size:
            poly_coeffs = {}
            nTerms = vtape[i]; i+=1
            #print("POLY w/%d terms (i=%d)" % (nTerms,i))
            for m in range(nTerms):
                nVars = vtape[i]; i+=1 # number of variable indices in this term
                a = ctape[c]; c+=1
                #print("  TERM%d: %d vars, coeff=%s" % (m,nVars,str(a)))
                poly_coeffs[ tuple(vtape[i:i+nVars]) ] = a; i += nVars
            result.append( Polynomial(poly_coeffs) )
    return result


def compact_deriv(vtape, ctape, wrtParams):
    """
    Take the derivative of one or more compact Polynomials with respect
    to one or more variables/parameters.

    Parameters
    ----------
    vtape, ctape : numpy.ndarray
        Specifies "variable" and "coefficient" 1D numpy arrays to differentiate.
        These "tapes" can be generated by concatenating the tapes of individual
        complact-polynomial tuples returned by :method:`Polynomial.compact`.

    wrtParams : list
        The variable indices to differentiate with respect to.  They 
        must be sorted in ascending order. E.g. "[0,3]" means separatey
        differentiate w.r.t x_0 and x_3 (concatenated first by wrtParam 
        then by poly).

    Returns
    -------
    vtape, ctape : numpy.ndarray
    """
    result_vtape = []
    result_ctape = []
    wrt = sorted(wrtParams)
    assert(wrt == list(wrtParams)), "`wrtParams` (%s) must be in ascending order!" % wrtParams
    #print("TAPE SIZE = ",vtape.size)
    
    c = 0; i = 0    
    while i < vtape.size:
        j = i # increment j instead of i for this poly
        nTerms = vtape[j]; j+=1
        dctapes = [ list() for x in range(len(wrt)) ]
        dvtapes = [ list() for x in range(len(wrt)) ]
        dnterms = [ 0 ]*len(wrt)
        #print("POLY w/%d terms (i=%d)" % (nTerms,i))
        for m in range(nTerms):
            coeff = ctape[c]; c += 1
            nVars = vtape[j]; j += 1 # number of variable indices in this term

            #print("  TERM%d: %d vars, coeff=%s" % (m,nVars,str(coeff)))
            cur_iWrt = 0;
            j0 = j # the vtape index where the current term starts

            #Loop to get counts of each variable index that is also in `wrt`.
            # Once we've passed an element of `wrt` process it, since there can't
            # see it any more (the var indices are sorted).
            while j < j0+nVars: #loop over variable indices for this term
                # can't be while True above in case nVars == 0 (then vtape[j] isn't valid)
                
                #find an iVar that is also in wrt.
                # - increment the cur_iWrt or j as needed                
                while cur_iWrt < len(wrt) and vtape[j] > wrt[cur_iWrt]: #condition to increment cur_iWrt
                    cur_iWrt += 1 # so wrt[cur_iWrt] >= vtape[j]
                if cur_iWrt == len(wrt): break  # no more possible iVars we're interested in;
                                                # we're done with all wrt elements
                # - at this point we know wrt[cur_iWrt] is valid and wrt[cur_iWrt] >= tape[j]
                while j < j0+nVars and vtape[j] < wrt[cur_iWrt]:
                    j += 1 # so vtape[j] >= wrt[cur_iWrt]
                if j == j0+nVars: break  # no more iVars - we're done

                #print(" check j=%d, val=%d, wrt=%d, cur_iWrt=%d" % (j,vtape[j],wrt[cur_iWrt],cur_iWrt))
                if vtape[j] == wrt[cur_iWrt]:
                    #Yay! a value we're looking for is present in the vtape.
                    # Figure out how many there are (easy since vtape is sorted
                    # and we'll always stop on the first one)
                    cnt = 0
                    while j < j0+nVars and vtape[j] == wrt[cur_iWrt]:
                        cnt += 1; j += 1
                    #Process cur_iWrt: add a term to tape for cur_iWrt
                    dvars = list(vtape[j0:j-1]) + list(vtape[j:j0+nVars]) # removes last wrt[cur_iWrt] var
                    dctapes[cur_iWrt].append(coeff*cnt)
                    dvtapes[cur_iWrt].extend( [nVars-1] + dvars )
                    dnterms[cur_iWrt] += 1
                    #print(" wrt=%d found cnt=%d: adding deriv term coeff=%f vars=%s" % (wrt[cur_iWrt], cnt, coeff*cnt, [nVars-1] + dvars))

                    cur_iWrt += 1 # processed this wrt param - move to next one

            #Now term has been processed, adding derivative terms to the dctapes and dvtapes "tape-lists"
            # We continue processing terms, adding to these tape lists, until all the terms of the
            # current poly are processed.  Then we can concatenate the tapes for each wrtParams element.
            j = j0 + nVars # move to next term; j may not have been incremented if we exited b/c of cur_iWrt reaching end
            
        #Now all terms are processed - concatenate tapes for wrtParams and add to resulting tape.
        for nTerms,dvtape,dctape in zip(dnterms, dvtapes, dctapes):
            result_vtape.extend( [nTerms] + dvtape )
            result_ctape.extend( dctape )
        i = j # update location in vtape after processing poly - actually could just use i instead of j it seems??

    return _np.array(result_vtape,_np.int64), _np.array(result_ctape,complex)

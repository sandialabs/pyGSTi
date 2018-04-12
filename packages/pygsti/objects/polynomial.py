""" Defines the Polynomial class """
from __future__ import division, print_function, absolute_import, unicode_literals
#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************

import numpy as _np

class Polynomial(dict):
    """ Encapsulates a polynomial """
    def __init__(self, coeffs=None):
        """ TODO: docstring - coeffs is a dict of coefficients w/keys == tuples
             of integer variable indices.  E.g. (1,1) means "variable1 squared"
        """
        super(Polynomial,self).__init__()
        if coeffs is not None:
            self.update(coeffs)

    def deriv(self, wrtParam):
        dcoeffs = {}
        for ivar, coeff in self.items():
            cnt = float(ivar.count(wrtParam))
            if cnt > 0:
                l = list(ivar)
                del l[l.index(wrtParam)]
                dcoeffs[ tuple(l) ] = cnt * coeff

        return Polynomial(dcoeffs) # returns another polynomial

    def evaluate(self, variable_values):
        """ TODO: docstring -- and make this function smarter (Russian peasant) """
        ret = 0
        for ivar,coeff in self.items():
            ret += coeff * _np.product( [variable_values[i] for i in ivar] )
        return ret

    def compact(self):
        """ TODO docstring Returns compact representation of (vtape, ctape) 1D nupy arrays """
        iscomplex = any([ abs(_np.imag(x)) > 1e-12 for x in self.values() ])
        nTerms = len(self)
        nVarIndices = sum(map(len,self.keys()))
        vtape = _np.empty(1 + nTerms + nVarIndices, 'i') # "variable" tape
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
        return Polynomial(self)

    def map_indices(self, mapfn):
        """ TODO: docstring - mapfn should map old->new variable-index-tuples """
        new_items = { mapfn(k): v for k,v in self.items() }
        self.clear()
        self.update(new_items)

    def addin(self,x):
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

    def __str__(self):
        def fmt(x):
            if abs(_np.imag(x)) > 1e-6:
                if abs(_np.real(x)) > 1e-6: return "(%.3f+%.3fj)" % (x.real, x.imag)
                else: return "(%.3fj)" % x.imag
            else: return "%.3f" % x.real
            
        termstrs = []
        sorted_keys = sorted(list(self.keys()))
        for k in sorted_keys:
            varstr = ""; last_i = None; n=0
            for i in sorted(k):
                if i == last_i: n += 1
                elif last_i is not None:
                    varstr += "x%d%s" % (last_i, ("^%d" % n) if n > 1 else "")
                last_i = i
            if last_i is not None:
                varstr += "x%d%s" % (last_i, ("^%d" % n) if n > 1 else "")
            #print("DB: k = ",k, " varstr = ",varstr)
            if abs(self[k]) > 1e-4:
                termstrs.append( "%s%s" % (fmt(self[k]), varstr) )
        return " + ".join(termstrs)

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
                

    def __mul__(self,x):
        if isinstance(x, Polynomial):
            newpoly = Polynomial()
            for k1,v1 in self.items():
                for k2,v2 in x.items():
                    k = tuple(sorted(k1+k2))
                    if k in newpoly: newpoly[k] += v1*v2
                    else: newpoly[k] = v1*v2 
        else: # assume a scalar that can multiply values
            newpoly = self.copy()
            for k in newpoly:
                newpoly[k] *= x
        return newpoly

    def __rmul__(self, x):
        return self.__mul__(x)

    def __pow__(self,n):
        ret = Polynomial({(): 1.0}) 
        cur = self
        for i in range(int(np.floor(np.log2(n)))+1):
            rem = n % 2 #gets least significant bit (i-th) of n
            if rem == 1: ret *= cur # add current power of x (2^i) if needed  
            cur = cur*cur # current power *= 2
            n //= 2 # shift bits of n right 
        return ret

    def __copy__(self):
        return self.copy()


def bulk_eval_compact_polys(compact_poly_tapes, paramvec, dest_shape):
    vtape, ctape = compact_poly_tapes
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
        

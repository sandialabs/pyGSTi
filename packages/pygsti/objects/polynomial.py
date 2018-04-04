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
    def __init__(self, nVariables, coeffs=None):
        """ TODO: docstring - coeffs is a dict of coefficients w/keys == tuples
             of integer variable indices.  E.g. (1,1) means "variable1 squared"
        """
        super(Polynomial,self).__init__()
        self.num_variables = nVariables # "named" by integers 0 to nVariables-1
        if coeffs is not None:
            self.update(coeffs)

    def deriv(self):
        pass # TODO - returns another polynomial

    def evaluate(self, variable_values):
        pass # TODO

    def copy(self):
        return Polynomial(self.num_variables, self)

    def __str__(self):
        def fmt(x):
            if _np.imag(x) > 1e-6:
                if _np.real(x) > 1e-6: return "(%.2f+%.2fj)" % (x.real, x.imag)
                else: return "(%.2fj)" % x.imag
            else: return "%.2f" % x.real
            
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
            assert(self.num_variables == x.num_variables), \
                "Polynomials must have same number of variables!"
            for k,v in x.items():
                if k in newpoly: newpoly[k] += v
                else: newpoly[k] = v
        else: # assume a scalar that can be added to values
            for k in newpoly:
                newpoly[k] += x
        return newpoly
                

    def __mul__(self,x):
        if isinstance(x, Polynomial):
            assert(self.num_variables == x.num_variables), \
                "Polynomials must have same number of variables!"
            newpoly = Polynomial(self.num_variables)
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
        ret = Polynomial(self.num_variables, {(): 1.0}) 
        cur = self
        for i in range(int(np.floor(np.log2(n)))+1):
            rem = n % 2 #gets least significant bit (i-th) of n
            if rem == 1: ret *= cur # add current power of x (2^i) if needed  
            cur = cur*cur # current power *= 2
            n //= 2 # shift bits of n right 
        return ret

    def __copy__(self):
        return self.copy()

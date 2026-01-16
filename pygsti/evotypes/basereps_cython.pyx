"""
Base classes for Cython representations.
"""
#***************************************************************************************************
# Copyright 2015, 2019, 2025 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

#I'm not sure why this doesn't seem to work in the .pxd file, but does here:
from cython.operator cimport dereference as deref, preincrement as inc

import numpy as _np


cdef class OpRep:
    pass

cdef class StateRep:
    pass

cdef class EffectRep:
    pass

cdef class TermRep:
    pass


# Other classes
cdef class PolynomialRep:
    #Use normal init here so can bypass to create from an already alloc'd c_polynomial
    def __init__(self, int_coeff_dict, INT max_num_vars, INT vindices_per_int):
        cdef unordered_map[PolynomialVarsIndex, complex] coeffs
        cdef PolynomialVarsIndex indx
        for i_tup,c in int_coeff_dict.items():
            indx = PolynomialVarsIndex(len(i_tup))
            for ii,i in enumerate(i_tup):
                indx._parts[ii] = i
            coeffs[indx] = <double complex>c
        self.c_polynomial = new PolynomialCRep(coeffs, max_num_vars, vindices_per_int)

    def __reduce__(self):
        return (PolynomialRep, (self.int_coeffs, self.max_num_vars, self.vindices_per_int))

    def __pygsti_reduce__(self):
        return self.__reduce__()

    def __dealloc__(self):
        del self.c_polynomial

    def reinit(self, int_coeff_dict):
        #Very similar to init, but same max_num_vars

        #Store these before deleting self.c_polynomial!
        cdef INT max_num_vars = self.c_polynomial._max_num_vars
        cdef INT vindices_per_int = self.c_polynomial._vindices_per_int
        del self.c_polynomial

        cdef unordered_map[PolynomialVarsIndex, complex] coeffs
        cdef PolynomialVarsIndex indx
        for i_tup,c in int_coeff_dict.items():
            indx = PolynomialVarsIndex(len(i_tup))
            for ii,i in enumerate(i_tup):
                indx._parts[ii] = i
            coeffs[indx] = <double complex>c
        self.c_polynomial = new PolynomialCRep(coeffs, max_num_vars, vindices_per_int)

    def mapvec_indices_inplace(self, _np.ndarray[_np.int64_t, ndim=1, mode='c'] mapfn_as_vector):
        cdef INT* mapfv = <INT*> mapfn_as_vector.data
        cdef INT indx, nxt, i, m, k, new_i, new_indx;
        cdef INT divisor = self.c_polynomial._max_num_vars + 1

        cdef PolynomialVarsIndex new_PolynomialVarsIndex
        cdef unordered_map[PolynomialVarsIndex, complex] new_coeffs
        cdef vector[INT].iterator vit
        cdef unordered_map[PolynomialVarsIndex, complex].iterator it = self.c_polynomial._coeffs.begin()

        while(it != self.c_polynomial._coeffs.end()): # for each coefficient
            i_vec = deref(it).first._parts  # the vector[INT] beneath this PolynomialVarsIndex
            new_PolynomialVarsIndex = PolynomialVarsIndex(i_vec.size())

            #map i_vec -> new_PolynomialVarsIndex
            vit = i_vec.begin(); k = 0
            while(vit != i_vec.end()):
                indx = deref(vit)
                new_indx = 0; m=1
                while indx != 0:
                    nxt = indx // divisor
                    i = indx - nxt * divisor
                    indx = nxt

                    # i-1 is variable index (the thing we map)
                    new_i = mapfv[i-1]+1
                    new_indx += new_i * m
                    m *= divisor

                new_PolynomialVarsIndex._parts[k] = new_indx
                inc(vit); k += 1
            new_coeffs[new_PolynomialVarsIndex] = deref(it).second
            inc(it)

        self.c_polynomial._coeffs.swap(new_coeffs)

    def copy(self):
        cdef PolynomialCRep* c_polynomial = new PolynomialCRep(self.c_polynomial._coeffs, self.c_polynomial._max_num_vars, self.c_polynomial._vindices_per_int)
        return PolynomialRep_from_allocd_PolynomialCRep(c_polynomial)

    @property
    def max_num_vars(self): # so we can convert back to python Polynomials
        return self.c_polynomial._max_num_vars

    @property
    def vindices_per_int(self):
        return self.c_polynomial._vindices_per_int

    @property
    def int_coeffs(self): # just convert coeffs keys (PolynomialVarsIndex objs) to tuples of Python ints
        ret = {}
        cdef vector[INT].iterator vit
        cdef unordered_map[PolynomialVarsIndex, complex].iterator it = self.c_polynomial._coeffs.begin()
        while(it != self.c_polynomial._coeffs.end()):
            i_tup = []
            i_vec = deref(it).first._parts
            vit = i_vec.begin()
            while(vit != i_vec.end()):
                i_tup.append( deref(vit) )
                inc(vit)
            ret[tuple(i_tup)] = deref(it).second
            inc(it)
        return ret

    #Get coeffs with tuples of variable indices, not just "ints" - not currently needed
    @property
    def coeffs(self):
        cdef INT indx, nxt, i;
        cdef INT divisor = self.c_polynomial._max_num_vars + 1
        ret = {}
        cdef vector[INT].iterator vit
        cdef unordered_map[PolynomialVarsIndex, complex].iterator it = self.c_polynomial._coeffs.begin()
        while(it != self.c_polynomial._coeffs.end()):
            i_tup = []
            i_vec = deref(it).first._parts

            # inline: int_to_vinds(indx)
            vit = i_vec.begin()
            while(vit != i_vec.end()):
                indx = deref(vit)
                while indx != 0:
                    nxt = indx // divisor
                    i = indx - nxt * divisor
                    i_tup.append(i-1)
                    indx = nxt
                inc(vit)

            ret[tuple(i_tup)] = deref(it).second
            inc(it)

        return ret

    def compact_complex(self):
        cdef INT i,l, iTerm, nVarIndices=0;
        cdef PolynomialVarsIndex k;
        cdef vector[INT] vs;
        cdef vector[INT].iterator vit
        cdef unordered_map[PolynomialVarsIndex, complex].iterator it = self.c_polynomial._coeffs.begin()
        cdef vector[ pair[PolynomialVarsIndex, vector[INT]] ] vinds;
        cdef INT nTerms = self.c_polynomial._coeffs.size()

        while(it != self.c_polynomial._coeffs.end()):
            vs = self.c_polynomial.int_to_vinds( deref(it).first )
            nVarIndices += vs.size()
            vinds.push_back( pair[PolynomialVarsIndex, vector[INT]](deref(it).first, vs) )
            inc(it)

        vtape = _np.empty(1 + nTerms + nVarIndices, _np.int64) # "variable" tape
        ctape = _np.empty(nTerms, _np.complex128) # "coefficient tape"

        i = 0
        vtape[i] = nTerms; i+=1
        stdsort(vinds.begin(), vinds.end(), &compare_pair) # sorts in place
        for iTerm in range(vinds.size()):
            k = vinds[iTerm].first
            v = vinds[iTerm].second
            l = v.size()
            ctape[iTerm] = self.c_polynomial._coeffs[k]
            vtape[i] = l; i += 1
            vtape[i:i+l] = v; i += l

        return vtape, ctape

    def compact_real(self):
        cdef INT i,l, iTerm, nVarIndices=0;
        cdef PolynomialVarsIndex k;
        cdef vector[INT] v;
        cdef vector[INT].iterator vit
        cdef unordered_map[PolynomialVarsIndex, complex].iterator it = self.c_polynomial._coeffs.begin()
        cdef vector[ pair[PolynomialVarsIndex, vector[INT]] ] vinds;
        cdef INT nTerms = self.c_polynomial._coeffs.size()

        while(it != self.c_polynomial._coeffs.end()):
            vs = self.c_polynomial.int_to_vinds( deref(it).first )
            nVarIndices += vs.size()
            vinds.push_back( pair[PolynomialVarsIndex, vector[INT]](deref(it).first, vs) )
            inc(it)

        vtape = _np.empty(1 + nTerms + nVarIndices, _np.int64) # "variable" tape
        ctape = _np.empty(nTerms, _np.float64) # "coefficient tape"

        i = 0
        vtape[i] = nTerms; i+=1
        stdsort(vinds.begin(), vinds.end(), &compare_pair) # sorts in place
        for iTerm in range(vinds.size()):
            k = vinds[iTerm].first
            v = vinds[iTerm].second
            l = v.size()
            ctape[iTerm] = self.c_polynomial._coeffs[k].real
            vtape[i] = l; i += 1
            vtape[i:i+l] = v; i += l

        return vtape, ctape

    def mult(self, PolynomialRep other):
        cdef PolynomialCRep result = self.c_polynomial.mult(deref(other.c_polynomial))
        cdef PolynomialCRep* persistent = new PolynomialCRep(result._coeffs, result._max_num_vars, result._vindices_per_int)
        return PolynomialRep_from_allocd_PolynomialCRep(persistent)
        #print "MULT ", self.coeffs, " * ", other.coeffs, " =", ret.coeffs  #DEBUG!!! HERE
        #return ret

    def scale(self, x):
        self.c_polynomial.scale(x)

    def add_inplace(self, PolynomialRep other):
        self.c_polynomial.add_inplace(deref(other.c_polynomial))

    def add_scalar_to_all_coeffs_inplace(self, x):
        self.c_polynomial.add_scalar_to_all_coeffs_inplace(x)

#cdef class XXXRankOnePolynomialTermWithMagnitude:
#    cdef public object term_ptr
#    cdef public double magnitude
#    cdef public double logmagnitude
#
#    @classmethod
#    def composed(cls, terms, double magnitude):
#        """
#        Compose a sequence of terms.
#
#        Composition is done with *time* ordered left-to-right. Thus composition
#        order is NOT the same as usual matrix order.
#        E.g. if there are three terms:
#        `terms[0]` = T0: rho -> A*rho*A
#        `terms[1]` = T1: rho -> B*rho*B
#        `terms[2]` = T2: rho -> C*rho*C
#        Then the resulting term T = T0*T1*T2 : rho -> CBA*rho*ABC, so
#        that term[0] is applied *first* not last to a state.
#
#        Parameters
#        ----------
#        terms : list
#            A list of terms to compose.
#
#        magnitude : float, optional
#            The magnitude of the composed term (fine to leave as None
#            if you don't care about keeping track of magnitudes).
#
#        Returns
#        -------
#        RankOneTerm
#        """
#        return cls(terms[0].term_ptr.compose([t.term_ptr for t in terms[1:]]), magnitude)
#
#    def __cinit__(self, rankOneTerm, double magnitude):
#        """
#        TODO: docstring
#        """
#        self.term_ptr = rankOneTerm
#        self.magnitude = magnitude
#        self.logmagnitude = log10(magnitude) if magnitude > 0 else -LARGE
#
#    def copy(self):
#        """
#        Copy this term.
#
#        Returns
#        -------
#        RankOneTerm
#        """
#        return RankOnePolynomialTermWithMagnitude(self.term_ptr.copy(), self.magnitude)
#
#    def embed(self, stateSpaceLabels, targetLabels):
#        return RankOnePolynomialTermWithMagnitude(self.term_ptr.embed(stateSpaceLabels, targetLabels), self.magnitude)
#
#    def scalar_mult(self, x):
#        return RankOnePolynomialTermWithMagnitude(self.term_ptr * x, self.magnitude * x)
#
#    def __mul__(self, x):
#        """ Multiply by scalar """
#        return RankOnePolynomialTermWithMagnitude(self.term_ptr * x, self.magnitude * x)
#
#    def __rmul__(self, x):
#        return self.__mul__(x)
#
#    def torep(self):
#        """
#        Construct a representation of this term.
#
#        "Representations" are lightweight versions of objects used to improve
#        the efficiency of intensely computational tasks, used primarily
#        internally within pyGSTi.
#
#        Parameters
#        ----------
#        max_num_vars : int
#            The maximum number of variables for the coefficient polynomial's
#            represenatation.
#
#        typ : { "prep", "effect", "gate" }
#            What type of representation is needed (these correspond to
#            different types of representation objects).  Given the type of
#            operations stored within a term, only one of "gate" and
#            "prep"/"effect" is appropriate.
#
#        Returns
#        -------
#        SVTermRep or SBTermRep
#        """
#        #assert(magnitude <= 1.0), "Individual term magnitudes should be <= 1.0 so that '*_above_mag' routines work!"
#        return self.term_ptr.torep(self.magnitude, self.logmagnitude)
#
#    def mapvec_indices_inplace(self, mapvec):
#        self.term_ptr.mapvec_indices_inplace(mapvec)


# Helper functions
cdef PolynomialRep_from_allocd_PolynomialCRep(PolynomialCRep* crep):
    cdef PolynomialRep ret = PolynomialRep.__new__(PolynomialRep) # doesn't call __init__
    ret.c_polynomial = crep
    return ret

cdef bool compare_pair(const pair[PolynomialVarsIndex, vector[INT]]& a, const pair[PolynomialVarsIndex, vector[INT]]& b):
    return a.first < b.first

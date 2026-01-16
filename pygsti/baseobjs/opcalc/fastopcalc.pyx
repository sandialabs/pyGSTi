# encoding: utf-8
# cython: profile=False
# cython: linetrace=False
# filename: fastcalc.pyx

#***************************************************************************************************
# Copyright 2015, 2019, 2025 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************


import numpy as np
from libc.stdlib cimport malloc, free
from libcpp.algorithm cimport sort as stdsort
from libcpp.unordered_map cimport unordered_map
from libcpp.vector cimport vector
from libcpp.string cimport string
from cython.operator cimport dereference as deref, preincrement as inc
from ...tools import symplectic
cimport numpy as np
cimport cython

#Use 64-bit integers
ctypedef long long INT
ctypedef unsigned long long UINT

def test_map(s):
    cdef string st = s.encode('UTF-8')
    cdef unordered_map[int, complex] my_map
    cdef unordered_map[int, complex] my_map2
    cdef unordered_map[int, complex].iterator it
    cdef vector[string] v
    v = vector[string](3)
    v[0] = st

    my_map[1]=3.0+2.0j
    my_map[2]=6.2
    my_map2 = my_map
    my_map2[2]=10.0
    my_map2[3]=20.0

    print(my_map[1], my_map[2])
    print(my_map2[1], my_map2[2], my_map2[3])
    print("HELLO!!!")

    #try to update map
    it = my_map.begin()
    while it != my_map.end():
        deref(it).second = 12.0+12.0j
        inc(it)

    #Print map
    it = my_map.begin()
    while it != my_map.end():
        print(deref(it).first)
        print(deref(it).second)
        inc(it)
#    for x in my_map:
#        print x.first
#        print my_map[x]

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def bulk_eval_compact_polynomials_real(np.ndarray[np.int64_t, ndim=1, mode="c"] vtape,
                                 np.ndarray[double, ndim=1, mode="c"] ctape,
                                 np.ndarray[double, ndim=1, mode="c"] paramvec,
                                 dest_shape):
    cdef INT dest_size = np.prod(dest_shape)
    cdef np.ndarray[np.float64_t, ndim=1, mode="c"] res = np.empty(dest_size, np.float64)

    cdef INT c = 0
    cdef INT i = 0
    cdef INT r = 0
    cdef INT vtape_sz = vtape.size
    cdef INT nTerms
    cdef INT m
    cdef INT k
    cdef INT nVars
    cdef double a;
    cdef double poly_val;

    while i < vtape_sz:
        poly_val = 0.0
        nTerms = vtape[i]; i+=1
        #print "POLY w/%d terms (i=%d)" % (nTerms,i)
        for m in range(nTerms):
            nVars = vtape[i]; i+=1 # number of variable indices in this term
            a = ctape[c]; c+=1
            #print "  TERM%d: %d vars, coeff=%s" % (m,nVars,str(a))
            for k in range(nVars):
                a *= paramvec[ vtape[i] ]; i+=1
            poly_val += a
            #print "  -> added %s to poly_val = %s" % (str(a),str(poly_val))," i=%d, vsize=%d" % (i,vtape.size)
        res[r] = poly_val; r+=1
    # = dest_shape # reshape w/out possibility of copying
    return res.reshape(dest_shape)


#Same as above, just takes a complex ctape
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def bulk_eval_compact_polynomials_complex(np.ndarray[np.int64_t, ndim=1, mode="c"] vtape,
                                    np.ndarray[np.complex128_t, ndim=1, mode="c"] ctape,
                                    np.ndarray[double, ndim=1, mode="c"] paramvec,
                                    dest_shape):
    cdef INT k
    cdef INT dest_size = 1  # np.prod(dest_shape) #SLOW!
    for k in range(len(dest_shape)):
        dest_size *= dest_shape[k]
    cdef np.ndarray[np.complex128_t, ndim=1, mode="c"] res = np.empty(dest_size, np.complex128)

    cdef INT c = 0
    cdef INT i = 0
    cdef INT r = 0
    cdef INT vtape_sz = vtape.size
    cdef INT nTerms
    cdef INT m
    #cdef INT k
    cdef INT nVars
    cdef double complex a;
    cdef double complex poly_val;

    while i < vtape_sz:
        poly_val = 0.0
        nTerms = vtape[i]; i+=1
        #print "POLY w/%d terms (i=%d)" % (nTerms,i)
        for m in range(nTerms):
            nVars = vtape[i]; i+=1 # number of variable indices in this term
            a = ctape[c]; c+=1
            #print "  TERM%d: %d vars, coeff=%s" % (m,nVars,str(a))
            for k in range(nVars):
                a *= paramvec[ vtape[i] ]; i+=1
            poly_val += a
            #print "  -> added %s to poly_val = %s" % (str(a),str(poly_val))," i=%d, vsize=%d" % (i,vtape.size)
        res[r] = poly_val; r+=1
    # = dest_shape # reshape w/out possibility of copying
    return res.reshape(dest_shape)


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def bulk_eval_compact_polynomials_derivs_real(np.ndarray[np.int64_t, ndim=1, mode="c"] vtape,
                                              np.ndarray[double, ndim=1, mode="c"] ctape,
                                              np.ndarray[np.int64_t, ndim=1, mode="c"] wrtParams,
                                              np.ndarray[double, ndim=1, mode="c"] paramvec,
                                              dest_shape):
    #Note: assumes wrtParams is SORTED but doesn't assert it like Python version does

    cdef INT c, i, iPoly
    cdef INT j,k, m, nTerms, nVars, cur_iWrt, j0, j1, cur_wrt, cnt, off
    cdef INT vtape_sz = vtape.size
    cdef INT wrt_sz = wrtParams.size
    cdef double coeff
    cdef double a

    assert(len(dest_shape) == 2)
    assert(len(wrtParams) == dest_shape[1])
    cdef np.ndarray[np.float64_t, ndim=2, mode="c"] res = np.zeros(dest_shape, np.float64)  # indices [iPoly, iParam]

    c = 0; i = 0; iPoly = 0
    while i < vtape_sz:
        j = i # increment j instead of i for this poly
        nTerms = vtape[j]; j+=1
        #print "POLY w/%d terms (i=%d)" % (nTerms,i)

        for m in range(nTerms):
            coeff = ctape[c]; c += 1
            nVars = vtape[j]; j += 1 # number of variable indices in this term

            #print "  TERM%d: %d vars, coeff=%s" % (m,nVars,str(coeff))
            cur_iWrt = 0
            j0 = j # the vtape index where the current term starts
            j1 = j+nVars # the ending index

            #Loop to get counts of each variable index that is also in `wrt`.
            # Once we've passed an element of `wrt` process it, since there can't
            # see it any more (the var indices are sorted).
            while j < j1: #loop over variable indices for this term
                # can't be while True above in case nVars == 0 (then vtape[j] isn't valid)

                #find an iVar that is also in wrt.
                # - increment the cur_iWrt or j as needed
                while cur_iWrt < wrt_sz and vtape[j] > wrtParams[cur_iWrt]: #condition to increment cur_iWrt
                    cur_iWrt += 1 # so wrtParams[cur_iWrt] >= vtape[j]
                if cur_iWrt == wrt_sz: break  # no more possible iVars we're interested in;
                                                # we're done with all wrt elements
                # - at this point we know wrt[cur_iWrt] is valid and wrt[cur_iWrt] >= tape[j]
                cur_wrt = wrtParams[cur_iWrt]
                while j < j1 and vtape[j] < cur_wrt:
                    j += 1 # so vtape[j] >= wrt[cur_iWrt]
                if j == j1: break  # no more iVars - we're done

                #print " check j=%d, val=%d, wrt=%d, cur_iWrt=%d" % (j,vtape[j],cur_wrt,cur_iWrt)
                if vtape[j] == cur_wrt:
                    #Yay! a value we're looking for is present in the vtape.
                    # Figure out how many there are (easy since vtape is sorted
                    # and we'll always stop on the first one)
                    cnt = 0
                    while j < j1 and vtape[j] == cur_wrt:
                        cnt += 1; j += 1
                    #Process cur_iWrt: add a term to evaluated poly for derivative w.r.t. wrtParams[cur_iWrt]
                    a = coeff*cnt
                    for k in range(j0,j1):
                        if k == j-1: continue # remove this index
                        a *= paramvec[ vtape[k] ]
                    res[iPoly, cur_iWrt] += a
                    cur_iWrt += 1 # processed this wrt param - move to next one

            j = j1 # move to next term; j may not have been incremented if we exited b/c of cur_iWrt reaching end

        i = j # update location in vtape after processing poly - actually could just use i instead of j it seems??
        iPoly += 1

    return res


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def bulk_eval_compact_polynomials_derivs_complex(np.ndarray[np.int64_t, ndim=1, mode="c"] vtape,
                                                np.ndarray[np.complex128_t, ndim=1, mode="c"] ctape,
                                                np.ndarray[np.int64_t, ndim=1, mode="c"] wrtParams,
                                                np.ndarray[double, ndim=1, mode="c"] paramvec,
                                                dest_shape):
    #Note: assumes wrtParams is SORTED but doesn't assert it like Python version does

    cdef INT c, i, iPoly
    cdef INT j,k, m, nTerms, nVars, cur_iWrt, j0, j1, cur_wrt, cnt, off
    cdef INT vtape_sz = vtape.size
    cdef INT wrt_sz = wrtParams.size
    cdef double complex coeff
    cdef double complex a

    assert(len(dest_shape) == 2)
    assert(len(wrtParams) == dest_shape[1])
    cdef np.ndarray[np.complex128_t, ndim=2, mode="c"] res = np.zeros(dest_shape, np.complex128)  # indices [iPoly, iParam]

    c = 0; i = 0; iPoly = 0
    while i < vtape_sz:
        j = i # increment j instead of i for this poly
        nTerms = vtape[j]; j+=1
        #print "POLY w/%d terms (i=%d)" % (nTerms,i)

        for m in range(nTerms):
            coeff = ctape[c]; c += 1
            nVars = vtape[j]; j += 1 # number of variable indices in this term

            #print "  TERM%d: %d vars, coeff=%s" % (m,nVars,str(coeff))
            cur_iWrt = 0
            j0 = j # the vtape index where the current term starts
            j1 = j+nVars # the ending index

            #Loop to get counts of each variable index that is also in `wrt`.
            # Once we've passed an element of `wrt` process it, since there can't
            # see it any more (the var indices are sorted).
            while j < j1: #loop over variable indices for this term
                # can't be while True above in case nVars == 0 (then vtape[j] isn't valid)

                #find an iVar that is also in wrt.
                # - increment the cur_iWrt or j as needed
                while cur_iWrt < wrt_sz and vtape[j] > wrtParams[cur_iWrt]: #condition to increment cur_iWrt
                    cur_iWrt += 1 # so wrtParams[cur_iWrt] >= vtape[j]
                if cur_iWrt == wrt_sz: break  # no more possible iVars we're interested in;
                                                # we're done with all wrt elements
                # - at this point we know wrt[cur_iWrt] is valid and wrt[cur_iWrt] >= tape[j]
                cur_wrt = wrtParams[cur_iWrt]
                while j < j1 and vtape[j] < cur_wrt:
                    j += 1 # so vtape[j] >= wrt[cur_iWrt]
                if j == j1: break  # no more iVars - we're done

                #print " check j=%d, val=%d, wrt=%d, cur_iWrt=%d" % (j,vtape[j],cur_wrt,cur_iWrt)
                if vtape[j] == cur_wrt:
                    #Yay! a value we're looking for is present in the vtape.
                    # Figure out how many there are (easy since vtape is sorted
                    # and we'll always stop on the first one)
                    cnt = 0
                    while j < j1 and vtape[j] == cur_wrt:
                        cnt += 1; j += 1
                    #Process cur_iWrt: add a term to evaluated poly for derivative w.r.t. wrtParams[cur_iWrt]
                    a = coeff*cnt
                    for k in range(j0,j1):
                        if k == j-1: continue # remove this index
                        a *= paramvec[ vtape[k] ]
                    res[iPoly, cur_iWrt] += a
                    cur_iWrt += 1 # processed this wrt param - move to next one

            j = j1 # move to next term; j may not have been incremented if we exited b/c of cur_iWrt reaching end

        i = j # update location in vtape after processing poly - actually could just use i instead of j it seems??
        iPoly += 1

    return res



# sum(abs(bulk_eval_compact_polynomials_complex(.))), made into its own function because numpy sum(abs())
#  is slow and this is done often in get_total_term_magnitude calls.
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def abs_sum_bulk_eval_compact_polynomials_complex(np.ndarray[np.int64_t, ndim=1, mode="c"] vtape,
                                                  np.ndarray[np.complex128_t, ndim=1, mode="c"] ctape,
                                                  np.ndarray[double, ndim=1, mode="c"] paramvec,
                                                  INT dest_size):
    cdef INT c = 0
    cdef INT i = 0
    cdef INT vtape_sz = vtape.size
    cdef INT nTerms
    cdef INT m
    cdef INT k
    cdef INT nVars
    cdef double complex a;
    cdef double complex poly_val;
    cdef double ret = 0.0

    while i < vtape_sz:
        poly_val = 0.0
        nTerms = vtape[i]; i+=1
        #print "POLY w/%d terms (i=%d)" % (nTerms,i)
        for m in range(nTerms):
            nVars = vtape[i]; i+=1 # number of variable indices in this term
            a = ctape[c]; c+=1
            #print "  TERM%d: %d vars, coeff=%s" % (m,nVars,str(a))
            for k in range(nVars):
                a *= paramvec[ vtape[i] ]; i+=1
            poly_val += a
            #print "  -> added %s to poly_val = %s" % (str(a),str(poly_val))," i=%d, vsize=%d" % (i,vtape.size)
        ret = ret + abs(poly_val)
    return ret


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def compact_deriv(np.ndarray[np.int64_t, ndim=1, mode="c"] vtape,
                  np.ndarray[np.complex128_t, ndim=1, mode="c"] ctape,
                  np.ndarray[np.int64_t, ndim=1, mode="c"] wrtParams):

    #Note: assumes wrtParams is SORTED but doesn't assert it like Python version does

    cdef INT c = 0
    cdef INT i = 0
    cdef INT j,k, m, nTerms, nVars, cur_iWrt, j0, j1, cur_wrt, cnt, off
    cdef INT vtape_sz = vtape.size
    cdef INT wrt_sz = wrtParams.size
    cdef double complex coeff

    #Figure out buffer sizes for dctapes & dvtapes
    cdef INT max_nTerms = 0
    cdef INT max_vsz = 0
    cdef INT nPolys = 0
    while i < vtape_sz:
        j = i # increment j instead of i for this poly
        nTerms = vtape[j]; j+=1
        if nTerms > max_nTerms:
            max_nTerms = nTerms
        for m in range(nTerms):
            nVars = vtape[j] # number of variable indices in this term
            j += nVars + 1
        if (j-i) > max_vsz:
            max_vsz = j-i # size of variable tape for this poly
        i = j; c += nTerms; nPolys += 1

    #print "MAX vtape-sz-per-poly = %d, MAX nTerms = %d, WRT size = %d" % (max_vsz, max_nTerms, wrt_sz)

    #Allocate space
    cdef INT vstride = max_vsz+1 # +1 for nTerms insertion
    cdef double complex* dctapes = <double complex *>malloc(wrt_sz * max_nTerms * sizeof(double complex))
    cdef INT*            dvtapes = <INT *>malloc(wrt_sz * vstride * sizeof(INT))
    cdef INT*            dnterms = <INT *>malloc(wrt_sz * sizeof(INT))
    cdef INT*            cptr = <INT *>malloc(wrt_sz * sizeof(INT))
    cdef INT*            vptr = <INT *>malloc(wrt_sz * sizeof(INT))
    #cdef np.ndarray dctapes = np.empty( (wrt_sz, max_nTerms), np.complex128)
    #cdef np.ndarray dvtapes = np.empty( (wrt_sz, max_vsz+1), np.int64) # +1 for nTerms insertion
    #cdef np.ndarray dnterms = np.zeros( wrt_sz, np.int64 )
    #cdef np.ndarray cptr = np.zeros( wrt_sz, np.int64 ) # how much of each dctapes row is used
    #cdef np.ndarray vptr = np.zeros( wrt_sz, np.int64 ) # how much of each dvtapes row is used

    cdef INT res_vptr = 0, res_cptr = 0
    cdef np.ndarray[np.int64_t, ndim=1, mode="c"] result_vtape = np.empty( nPolys*wrt_sz*vstride, np.int64 )
    cdef np.ndarray[np.complex128_t, ndim=1, mode="c"] result_ctape = np.empty( nPolys*wrt_sz*max_nTerms, np.complex128 )
    #print "TAPE SIZE = %d" % vtape_sz
    #print "RESULT SIZE = %d" % result_vtape.size

    c = 0; i = 0
    while i < vtape_sz:
        j = i # increment j instead of i for this poly
        nTerms = vtape[j]; j+=1
        #print "POLY w/%d terms (i=%d)" % (nTerms,i)

        # reset/clear dctapes, dvtapes, dnterms for this poly
        for k in range(wrt_sz):
            cptr[k] = 0
            vptr[k] = 1 # leave room to insert nTerms at end
            dnterms[k] = 0

        for m in range(nTerms):
            coeff = ctape[c]; c += 1
            nVars = vtape[j]; j += 1 # number of variable indices in this term

            #print "  TERM%d: %d vars, coeff=%s" % (m,nVars,str(coeff))
            cur_iWrt = 0
            j0 = j # the vtape index where the current term starts
            j1 = j+nVars # the ending index

            #Loop to get counts of each variable index that is also in `wrt`.
            # Once we've passed an element of `wrt` process it, since there can't
            # see it any more (the var indices are sorted).
            while j < j1: #loop over variable indices for this term
                # can't be while True above in case nVars == 0 (then vtape[j] isn't valid)

                #find an iVar that is also in wrt.
                # - increment the cur_iWrt or j as needed
                while cur_iWrt < wrt_sz and vtape[j] > wrtParams[cur_iWrt]: #condition to increment cur_iWrt
                    cur_iWrt += 1 # so wrtParams[cur_iWrt] >= vtape[j]
                if cur_iWrt == wrt_sz: break  # no more possible iVars we're interested in;
                                                # we're done with all wrt elements
                # - at this point we know wrt[cur_iWrt] is valid and wrt[cur_iWrt] >= tape[j]
                cur_wrt = wrtParams[cur_iWrt]
                while j < j1 and vtape[j] < cur_wrt:
                    j += 1 # so vtape[j] >= wrt[cur_iWrt]
                if j == j1: break  # no more iVars - we're done

                #print " check j=%d, val=%d, wrt=%d, cur_iWrt=%d" % (j,vtape[j],cur_wrt,cur_iWrt)
                if vtape[j] == cur_wrt:
                    #Yay! a value we're looking for is present in the vtape.
                    # Figure out how many there are (easy since vtape is sorted
                    # and we'll always stop on the first one)
                    cnt = 0
                    while j < j1 and vtape[j] == cur_wrt:
                        cnt += 1; j += 1
                    #Process cur_iWrt: add a term to tape for cur_iWrt
                    off = cur_iWrt*vstride
                    dctapes[ cur_iWrt*max_nTerms + cptr[cur_iWrt] ] = coeff*cnt;  cptr[cur_iWrt] += 1
                    dvtapes[ off + vptr[cur_iWrt] ] = nVars-1;                    vptr[cur_iWrt] += 1
                    off += vptr[cur_iWrt] # now off points to next available slot in dvtape
                    for k in range(j0,j1):
                        if k == j-1: continue # remove this index
                        dvtapes[ off ] = vtape[k]; off += 1
                        #print " k=%d -> var %d" % (k,vtape[k])
                    vptr[cur_iWrt] += (nVars-1) # accounts for all the off += 1 calls above.
                    dnterms[cur_iWrt] += 1
                    #print " wrt=%d found cnt=%d: adding deriv term coeff=%s nvars=%d" % (cur_wrt, cnt, str(coeff*cnt), nVars-1)
                    cur_iWrt += 1 # processed this wrt param - move to next one

            #Now term has been processed, adding derivative terms to the dctapes and dvtapes "tape-lists"
            # We continue processing terms, adding to these tape lists, until all the terms of the
            # current poly are processed.  Then we can concatenate the tapes for each wrtParams element.
            j = j1 # move to next term; j may not have been incremented if we exited b/c of cur_iWrt reaching end

        #Now all terms are processed - concatenate tapes for wrtParams and add to resulting tape.
        for k in range(wrt_sz):
            off = k*vstride
            dvtapes[off] = dnterms[k] # insert nTerms into space reserverd at beginning of each dvtape
            for l in range(vptr[k]):
                result_vtape[res_vptr] = dvtapes[off]; off += 1; res_vptr += 1

            off = k*max_nTerms
            for l in range(cptr[k]):
                result_ctape[res_cptr] = dctapes[off]; off += 1; res_cptr += 1

            #Use numpy, but still slower than above C-able code
            #result_vtape[res_vptr:res_vptr+vptr[k]] = dvtapes[k,0:vptr[k]]; res_vptr += vptr[k]
            #result_ctape[res_cptr:res_cptr+cptr[k]] = dctapes[k,0:cptr[k]]; res_cptr += cptr[k]

            #result_vtape = np.concatenate( (result_vtape, dvtapes[k,0:vptr[k]]) ) # SLOW!
            #result_ctape = np.concatenate( (result_ctape, dctapes[k,0:cptr[k]]) ) # SLOW!
        i = j # update location in vtape after processing poly - actually could just use i instead of j it seems??

    free(dctapes)
    free(dvtapes)
    free(dnterms)
    free(cptr)
    free(vptr)

    return result_vtape[0:res_vptr], result_ctape[0:res_cptr]



def prs_as_polynomials(circuit, rho_terms, gate_terms, E_terms, E_indices_py, int numEs, int max_order,
                 int stabilizer_evo):
    #NOTE: circuit and gate_terms use *integers* as operation labels, not Label objects, to speed
    # lookups and avoid weird string conversion stuff with Cython

    #print("DB: pr_as_poly for ",str(tuple(map(str,circuit))), " max_order=",self.max_order)


    #cdef double complex *pLeft = <double complex*>malloc(len(Es) * sizeof(double complex))
    #cdef double complex *pRight = <double complex*>malloc(len(Es) * sizeof(double complex))
    cdef int N = len(circuit)
    cdef int* p = <int*>malloc((N+2) * sizeof(int))
    cdef int i,j,k,order,nTerms
    cdef int max_poly_order=-1, max_poly_vars=-1
    cdef int gn

    #extract raw data from gate_terms dictionary-of-lists for faster lookup
    #gate_term_prefactors = _np.empty( (nOperations,max_order+1,dim,dim)
    cdef unordered_map[int, vector[vector[unordered_map[int, complex]]]] gate_term_coeffs
    cdef vector[vector[unordered_map[int, complex]]] rho_term_coeffs
    cdef vector[vector[unordered_map[int, complex]]] E_term_coeffs
    cdef vector[vector[int]] E_indices
    cdef vector[int] Einds

    for gl in gate_terms.keys():
        gn = gl
        gate_term_coeffs[gn] = extract_term_coeffs(gate_terms[gl], max_order,
                                                   max_poly_vars, max_poly_order)
    rho_term_coeffs = extract_term_coeffs(rho_terms, max_order,
                                          max_poly_vars, max_poly_order)
    E_term_coeffs = extract_term_coeffs(E_terms, max_order,
                                        max_poly_vars, max_poly_order)

    k = len(E_indices_py)
    E_indices = vector[vector[int]](k)
    for ii,inds in enumerate(E_indices_py):
        k = len(inds)
        E_indices[ii] = vector[int](k)
        for jj,indx in enumerate(inds):
            k = indx; E_indices[ii][jj] = k

    #OLD
    #    for order in range(max_order+1):
    #        nTerms = len(gate_terms[gl][order])
    #        gate_term_coeffs[gn][order] = vector[unordered_map[int, complex]](nTerms)
    #        for i,term in enumerate(gate_terms[gl][order]):
    #            #gate_term_coeffs[gn][order][i] = fastpoly_to_unorderedmap(term.coeff)
    #            polymap = unordered_map[int, complex]()
    #            poly = term.coeff
    #            if max_poly_order == -1: max_poly_order = poly.max_order
    #            else: assert(max_poly_order == poly.max_order)
    #            if max_poly_vars == -1: max_poly_vars = poly.max_num_vars
    #            else: assert(max_poly_vars == poly.max_num_vars)
    #            for k,v in poly.items(): polymap[k] = v
    #            gate_term_coeffs[gn][order][i] = polymap

    #        gate_term_prefactors[igl][order] = vector( GateObj(term.pre_ops[0]).acton_fn for term in gate_terms[gl][order] ) # assume all terms collapsed?

    assert(max_order <= 2) # only support this partitioning below (so far)

    cdef vector[ unordered_map[int, complex] ] prps = vector[ unordered_map[int, complex] ](numEs)
    #prps_chk = [None]*numEs
    for order in range(max_order+1):
        #print("DB: pr_as_poly order=",order)

        #for p in partition_into(order, N):
        for i in range(N+2): p[i] = 0 # clear p
        factor_lists = [None]*(N+2)
        coeff_lists = vector[vector[unordered_map[int, complex]]](N+2)

        if order == 0:
            #inner loop(p)
            #factor_lists = [ gate_terms[glbl][pi] for glbl,pi in zip(circuit,p) ]
            factor_lists[0] = rho_terms[p[0]]
            coeff_lists[0] = rho_term_coeffs[p[0]]
            for k in range(N):
                gn = circuit[k]
                factor_lists[k+1] = gate_terms[circuit[k]][p[k+1]]
                coeff_lists[k+1] = gate_term_coeffs[gn][p[k+1]]
                if len(factor_lists[k+1]) == 0: continue
            factor_lists[N+1] = E_terms[p[N+1]]
            coeff_lists[N+1] = E_term_coeffs[p[N+1]]
            Einds = E_indices[p[N+1]]

            #print("Part0 ",p)
            pr_as_poly_innerloop(factor_lists,coeff_lists,Einds,max_poly_vars,
                                 max_poly_order, stabilizer_evo, &prps) #, prps_chk)


        elif order == 1:
            for i in range(N+2):
                p[i] = 1
                #inner loop(p)
                factor_lists[0] = rho_terms[p[0]]
                coeff_lists[0] = rho_term_coeffs[p[0]]
                for k in range(N):
                    gn = circuit[k]
                    factor_lists[k+1] = gate_terms[circuit[k]][p[k+1]]
                    coeff_lists[k+1] = gate_term_coeffs[gn][p[k+1]]
                    if len(factor_lists[k+1]) == 0: continue
                factor_lists[N+1] = E_terms[p[N+1]]
                coeff_lists[N+1] = E_term_coeffs[p[N+1]]
                Einds = E_indices[p[N+1]]

                #print("Part1 ",p)
                pr_as_poly_innerloop(factor_lists,coeff_lists,Einds,
                                     max_poly_vars, max_poly_order,
                                     stabilizer_evo, &prps) #, prps_chk)
                p[i] = 0

        elif order == 2:
            for i in range(N+2):
                p[i] = 2
                #inner loop(p)
                factor_lists[0] = rho_terms[p[0]]
                coeff_lists[0] = rho_term_coeffs[p[0]]
                for k in range(N):
                    gn = circuit[k]
                    factor_lists[k+1] = gate_terms[circuit[k]][p[k+1]]
                    coeff_lists[k+1] = gate_term_coeffs[gn][p[k+1]]
                    if len(factor_lists[k+1]) == 0: continue
                factor_lists[N+1] = E_terms[p[N+1]]
                coeff_lists[N+1] = E_term_coeffs[p[N+1]]
                Einds = E_indices[p[N+1]]

                #print("Part2a ",p)
                pr_as_poly_innerloop(factor_lists, coeff_lists,Einds,
                                     max_poly_vars, max_poly_order,
                                     stabilizer_evo, &prps) #, prps_chk)
                p[i] = 0

            for i in range(N+2):
                p[i] = 1
                for j in range(i+1,N+2):
                    p[j] = 1
                    #inner loop(p)
                    factor_lists[0] = rho_terms[p[0]]
                    coeff_lists[0] = rho_term_coeffs[p[0]]
                    for k in range(N):
                        gn = circuit[k]
                        factor_lists[k+1] = gate_terms[circuit[k]][p[k+1]]
                        coeff_lists[k+1] = gate_term_coeffs[gn][p[k+1]]
                        if len(factor_lists[k+1]) == 0: continue
                    factor_lists[N+1] = E_terms[p[N+1]]
                    coeff_lists[N+1] = E_term_coeffs[p[N+1]]
                    Einds = E_indices[p[N+1]]

                    #print("Part2b ",p)
                    pr_as_poly_innerloop(factor_lists, coeff_lists, Einds,
                                         max_poly_vars, max_poly_order,
                                         stabilizer_evo, &prps) #, prps_chk)
                    p[j] = 0
                p[i] = 0
        else:
            assert(False) # order > 2 not implemented yet...

    return prps



cdef pr_as_poly_innerloop(factor_lists, factor_coeff_lists, vector[int]& Einds,
                          int max_poly_vars, int max_poly_order, int stabilizer_evo,
                          vector[ unordered_map[int, complex] ]* prps): #, prps_chk):
    #print("DB partition = ","listlens = ",[len(fl) for fl in factor_lists])

    cdef int i,j
    cdef int fastmode = 1 # HARDCODED - but it has been checked that non-fast-mode agrees w/fastmode
    cdef unordered_map[int, complex].iterator it1, it2, itk
    cdef unordered_map[int, complex] result, coeff, coeff2, curCoeff
    cdef double complex scale, val, newval, pLeft, pRight, p
    cdef vector[vector[unordered_map[int, complex]]] reduced_coeff_lists

    cdef int incd

    cdef int nFactorLists = len(factor_lists) # may need to recompute this after fast-mode
    cdef int* factorListLens = <int*>malloc(nFactorLists * sizeof(int))
    cdef int last_index = nFactorLists-1

    for i in range(nFactorLists):
        factorListLens[i] = len(factor_lists[i])
        if factorListLens[i] == 0: return # nothing to loop over!

    cdef int* b = <int*>malloc(nFactorLists * sizeof(int))
    for i in range(nFactorLists): b[i] = 0

    #DEBUG
    #if debug > 0:
    #    print "nLists = ", nFactorLists
    #    for i in range(nFactorLists):
    #        print factorListLens[i]
    assert(nFactorLists > 0), "Number of factor lists must be > 0!"

    if fastmode: # filter factor_lists to matrix-compose all length-1 lists

        leftSaved = [None]*(nFactorLists-1)  # saved[i] is state after i-th
        rightSaved = [None]*(nFactorLists-1) # factor has been applied
        coeffSaved = [None]*(nFactorLists-1)
        incd = 0

        #for factors in _itertools.product(*factor_lists):
        #for incd,fi in incd_product(*[range(len(l)) for l in factor_lists]):
        while(True):
            #if debug > 0: # DEBUG
            #    debug += 1
            #    print "DEBUG iter", debug, " b="
            #    for i in range(nFactorLists): print b[i]
            # In this loop, b holds "current" indices into factor_lists

            if incd == 0: # need to re-evaluate rho vector
                factor = factor_lists[0][b[0]]
                rhoVecL = factor.pre_ops[0].to_dense()
                for j in range(1,len(factor.pre_ops)):
                    rhoVecL = factor.pre_ops[j].acton(rhoVecL)
                leftSaved[0] = rhoVecL

                rhoVecR = factor.post_ops[0].to_dense()
                for j in range(1,len(factor.post_ops)):
                    rhoVecR = factor.post_ops[j].acton(rhoVecR)
                rightSaved[0] = rhoVecR

                coeff = factor_coeff_lists[0][b[0]]
                coeffSaved[0] = coeff
                incd += 1
            else:
                rhoVecL = leftSaved[incd-1]
                rhoVecR = rightSaved[incd-1]
                coeff = coeffSaved[incd-1]

            # propagate left and right states, saving as we go
            for i in range(incd,last_index):
                factor = factor_lists[i][b[i]]
                for j in range(len(factor.pre_ops)):
                    rhoVecL = factor.pre_ops[j].acton(rhoVecL)
                leftSaved[i] = rhoVecL

                for j in range(len(factor.post_ops)):
                    rhoVecR = factor.post_ops[j].acton(rhoVecR)
                rightSaved[i] = rhoVecR

                coeff = mult_polynomials(coeff, factor_coeff_lists[i][b[i]],
                                   max_poly_vars, max_poly_order)
                coeffSaved[i] = coeff

            # for the last index, no need to save, and need to construct
            # and apply effect vector
            if stabilizer_evo == 0:
                factor = factor_lists[last_index][b[last_index]] # the last factor (an Evec)
                EVec = factor.post_ops[0].to_dense() # TODO USE scratch here
                for j in range(1,len(factor.post_ops)): # evaluate effect term to arrive at final EVec
                    EVec = factor.post_ops[j].acton(EVec)
                pLeft = np.vdot(EVec,rhoVecL) # complex amplitudes, *not* real probabilities

                EVec = factor.pre_ops[0].to_dense() # TODO USE scratch here
                for j in range(1,len(factor.pre_ops)): # evaluate effect term to arrive at final EVec
                    EVec = factor.pre_ops[j].acton(EVec)
                pRight = np.conjugate(np.vdot(EVec,rhoVecR)) # complex amplitudes, *not* real probabilities
            else: # CLIFFORD - can't propagate effects, but can act w/adjoint of post_ops in reverse order...
                factor = factor_lists[last_index][b[last_index]] # the last factor (an Evec)
                EVec = factor.post_ops[0]
                for j in range(len(factor.post_ops)-1,0,-1): # (reversed)
                    rhoVecL = factor.post_ops[j].adjoint_acton(rhoVecL)
                #OLD: p = stabilizer_measurement_prob(rhoVecL, EVec.outcomes)
                #OLD: pLeft = np.sqrt(p) # sqrt b/c pLeft is just *amplitude*
                pLeft = rhoVecL.extract_amplitude(EVec.outcomes)

                EVec = factor.pre_ops[0]
                for j in range(len(factor.pre_ops)-1,0,-1): # (reversed)
                    rhoVecR = factor.pre_ops[j].adjoint_acton(rhoVecR)
                #OLD: p = stabilizer_measurement_prob(rhoVecR, EVec.outcomes)
                #OLD: pRight = np.sqrt(p) # sqrt b/c pRight is just *amplitude*
                pRight = np.conjugate(rhoVecR.extract_amplitude(EVec.outcomes))

            result = mult_polynomials(coeff, factor_coeff_lists[last_index][b[last_index]],
                               max_poly_vars, max_poly_order)
            scale_poly(result, (pLeft * pRight) )
            final_factor_indx = b[last_index]
            Ei = Einds[final_factor_indx] #final "factor" index == E-vector index
            add_polynomials_inplace(deref(prps)[Ei], result)

            #assert(debug < 100) #DEBUG

            #increment b ~ itertools.product & update vec_index_noop = _np.dot(self.multipliers, b)
            for i in range(nFactorLists-1,-1,-1):
                if b[i]+1 < factorListLens[i]:
                    b[i] += 1; incd = i
                    break
                else:
                    b[i] = 0
            else:
                break # can't increment anything - break while(True) loop


    else: # "slow" mode
        #for factors in _itertools.product(*factor_lists):
        while(True):
            # In this loop, b holds "current" indices into factor_lists

    #        print "Inner loop", b

            #OLD - now that spams are factors to, nFactorLists should always be >= 2
            ##coeff = _functools.reduce(lambda x,y: x.mult_poly(y), [f.coeff for f in factors])
            #if nFactorLists == 0:
            #    #coeff = _FastPolynomial({(): 1.0}, max_poly_vars, max_poly_order)
            #    coeff = unordered_map[int,complex](); coeff[0] = 1.0
            #else:
            coeff = factor_coeff_lists[0][b[0]] # an unordered_map (copies to new "coeff" variable)

            # CHECK POLY MATH
            #print "\n----- PRE MULT ---------"
            #coeff_check = factor_lists[0][b[0]].coeff
            #checkpolys(coeff, coeff_check)

            for i in range(1,nFactorLists):
                coeff = mult_polynomials(coeff, factor_coeff_lists[i][b[i]],
                                   max_poly_vars, max_poly_order)

                #CHECK POLY MATH
                #print "\n----- MULT ---------"
                #coeff_check = coeff_check.mult_poly(factor_lists[i][b[i]].coeff) # DEBUG
                #checkpolys(coeff, coeff_check)


            #pLeft  = self.unitary_sim_pre(rhoLeft,Es, factors, comm, memLimit, pLeft)
            #pRight = self.unitary_sim_post(rhoRight,Es, factors, comm, memLimit, pRight) \
            #         if not self.unitary_evolution else 1
            #NOTE: no unitary_evolution == 1 support yet...

            #pLeft / "pre" sim
            factor = factor_lists[0][b[0]] # 0th-factor = rhoVec
            rhoVec = factor.pre_ops[0].to_dense()
            for j in range(1,len(factor.pre_ops)):
                rhoVec = factor.pre_ops[j].acton(rhoVec)
            for i in range(1,last_index):
                factor = factor_lists[i][b[i]]
                for j in range(len(factor.pre_ops)):
                    rhoVec = factor.pre_ops[j].acton(rhoVec)
            factor = factor_lists[last_index][b[last_index]] # the last factor (an Evec)

            if stabilizer_evo == 0:
                EVec = factor.post_ops[0].to_dense() # TODO USE scratch here
                for j in range(1,len(factor.post_ops)): # evaluate effect term to arrive at final EVec
                    EVec = factor.post_ops[j].acton(EVec)
                pLeft = np.vdot(EVec,rhoVec) # complex amplitudes, *not* real probabilities
            else: # CLIFFORD - can't propagate effects, but can act w/adjoint of post_ops in reverse order...
                EVec = factor.post_ops[0]
                for j in range(len(factor.post_ops)-1,0,-1): # (reversed)
                    rhoVec = factor.post_ops[j].adjoint_acton(rhoVec)
                #OLD: p = stabilizer_measurement_prob(rhoVec, EVec.outcomes)
                #OLD: pLeft = np.sqrt(p) # sqrt b/c pLeft is just *amplitude*
                pLeft = rhoVec.extract_amplitude(EVec.outcomes)


            #pRight / "post" sim
            factor = factor_lists[0][b[0]] # 0th-factor = rhoVec
            rhoVec = factor.post_ops[0].to_dense()
            for j in range(1,len(factor.post_ops)):
                rhoVec = factor.post_ops[j].acton(rhoVec)
            for i in range(1,last_index):
                factor = factor_lists[i][b[i]]
                for j in range(len(factor.post_ops)):
                    rhoVec = factor.post_ops[j].acton(rhoVec)
            factor = factor_lists[last_index][b[last_index]] # the last factor (an Evec)

            if stabilizer_evo == 0:
                EVec = factor.pre_ops[0].to_dense() # TODO USE scratch here
                for j in range(1,len(factor.pre_ops)): # evaluate effect term to arrive at final EVec
                    EVec = factor.pre_ops[j].acton(EVec)
                pRight = np.conjugate(np.vdot(EVec,rhoVec)) # complex amplitudes, *not* real probabilities
            else: # CLIFFORD - can't propagate effects, but can act w/adjoint of post_ops in reverse order...
                EVec = factor.pre_ops[0]
                for j in range(len(factor.pre_ops)-1,0,-1): # (reversed)
                    rhoVec = factor.pre_ops[j].adjoint_acton(rhoVec)
                #OLD: p = stabilizer_measurement_prob(rhoVec, EVec.outcomes)
                #OLD: pRight = np.sqrt(p) # sqrt b/c pRight is just *amplitude*
                pRight = np.conjugate(rhoVec.extract_amplitude(EVec.outcomes))

            #Add result to appropriate poly
            result = coeff  # use a reference?
            scale_poly(result, (pLeft * pRight) )
            final_factor_indx = b[last_index]
            Ei = Einds[final_factor_indx] #final "factor" index == E-vector index

            #CHECK POLY MATH
            #res = coeff_check.mult_scalar( (pLeft * pRight) ) #DEBUG
            #print "\n----- Post SCALE by ",(pLeft * pRight),"---------"
            #print "pLeft, pRight = ",pLeft,pRight
            #checkpolys(result, res)
            #if prps_chk[Ei] is None:  prps_chk[Ei] = res
            #else:                    prps_chk[Ei] += res

            add_polynomials_inplace(deref(prps)[Ei], result)

            #CHECK POLY MATH
            #print "\n---------- PRPS Check ----------",Ei
            #checkpolys(deref(prps)[Ei], prps_chk[Ei])

            #print("DB: pr_as_poly     factor coeff=",coeff," pLeft=",pLeft," pRight=",pRight, "res=",res,str(type(res)))

            #increment b ~ itertools.product & update vec_index_noop = _np.dot(self.multipliers, b)
            for i in range(nFactorLists-1,-1,-1):
                if b[i]+1 < factorListLens[i]:
                    b[i] += 1
                    break
                else:
                    b[i] = 0
            else:
                break # can't increment anything - break while(True) loop

        #print("DB: pr_as_poly   partition=",p,"(cnt ",db_part_cnt," with ",db_nfactors," factors (cnt=",db_factor_cnt,")")

cdef unordered_map[int, complex] mult_polynomials(unordered_map[int, complex]& poly1,
                                            unordered_map[int, complex]& poly2,
                                            int max_poly_vars, int max_poly_order):
    cdef unordered_map[int, complex].iterator it1, it2
    cdef unordered_map[int, complex].iterator itk
    cdef unordered_map[int, complex] result = unordered_map[int,complex]()
    cdef double complex val

    it1 = poly1.begin()
    while it1 != poly1.end():
        it2 = poly2.begin()
        while it2 != poly2.end():
            k = mult_vinds_ints(deref(it1).first, deref(it2).first, max_poly_vars, max_poly_order) #key to add
            itk = result.find(k)
            val = deref(it1).second * deref(it2).second
            if itk != result.end():
                deref(itk).second = deref(itk).second + val
            else: result[k] = val
            inc(it2)
        inc(it1)
    return result


cdef void add_polynomials_inplace(unordered_map[int, complex]& poly1,
                            unordered_map[int, complex]& poly2):
    """ poly1 += poly2 """
    cdef unordered_map[int, complex].iterator it2, itk
    cdef double complex val, newval

    it2 = poly2.begin()
    while it2 != poly2.end():
        k = deref(it2).first # key
        val = deref(it2).second # value
        itk = poly1.find(k)
        if itk != poly1.end():
            newval = deref(itk).second + val
            if abs(newval) > 1e-12:
                deref(itk).second = newval # note: += doens't work here (complex Cython?)
            else: poly1.erase(itk)
        elif abs(val) > 1e-12:
            poly1[k] = val
        inc(it2)
    return


cdef void scale_poly(unordered_map[int, complex]& poly,
                     double complex scale):
    """" poly *= scale """
    cdef unordered_map[int, complex].iterator it
    it = poly.begin()
    while it != poly.end():
        deref(it).second = deref(it).second * scale # note: *= doesn't work here (complex Cython?)
        inc(it)
    return


cdef vinds_to_int(vector[int] vinds, int max_num_vars, int max_order):
    cdef int ret = 0
    cdef int i,m = 1
    for i in vinds: # last tuple index is most significant
        ret += (i+1)*m
        m *= max_num_vars+1
    return ret

cdef int_to_vinds(int indx, int max_num_vars, int max_order):
    cdef vector[int] ret
    cdef int nxt, i
    while indx != 0:
        nxt = indx // (max_num_vars+1)
        i = indx - nxt*(max_num_vars+1)
        ret.push_back(i-1)
        indx = nxt
    stdsort(ret.begin(),ret.end())
    return ret

cdef mult_vinds_ints(int i1, int i2, int max_num_vars, int max_order):
    cdef vector[int] vinds1 = int_to_vinds(i1, max_num_vars, max_order)
    cdef vector[int] vinds2 = int_to_vinds(i2, max_num_vars, max_order)
    vinds1.insert( vinds1.end(), vinds2.begin(), vinds2.end() )
    stdsort(vinds1.begin(),vinds1.end())
    return vinds_to_int(vinds1, max_num_vars, max_order)

def check_polynomials(unordered_map[int,complex] coeff, coeff_check):
    cdef int mismatch = 0
    cdef unordered_map[int,complex].iterator it = coeff.begin()
    while it != coeff.end():
        k = deref(it).first # key
        if k in coeff_check and abs(coeff_check[k]-deref(it).second) < 1e-6:
            inc(it)
        else:
            mismatch = 1; break

    print("MISMATCH = ", mismatch)
    print("coeff=")
    it = coeff.begin()
    while it != coeff.end():
        print(deref(it)); inc(it)
    print("coeff_check=",coeff_check)
    #    assert(0),"Mismatch!"

cdef vector[vector[unordered_map[int, complex] ]] extract_term_coeffs(python_terms, int max_order, int& max_poly_vars, int& max_poly_order):
    ret = vector[vector[unordered_map[int, complex] ]](max_order+1)
    for order in range(max_order+1):
        nTerms = len(python_terms[order])
        ret[order] = vector[unordered_map[int, complex]](nTerms)
        for i,term in enumerate(python_terms[order]):
            #ret[order][i] = fastpoly_to_unorderedmap(term.coeff)
            polymap = unordered_map[int, complex]()
            poly = term.coeff
            if max_poly_order == -1: (&max_poly_order)[0] = poly.max_order #reference assignment workaround (known Cython bug)
            else: assert(max_poly_order == poly.max_order)
            if max_poly_vars == -1: (&max_poly_vars)[0] = poly.max_num_vars
            else: assert(max_poly_vars == poly.max_num_vars)
            for k,v in poly.items(): polymap[k] = v
            ret[order][i] = polymap
    return ret

cdef double stabilizer_measurement_prob(state_sp_tuple, moutcomes):
    #Note: an abridged version from what is in ForwardSimulator... (no qubit_filter or return_state)
    #TODO: make this routine faster - port pauli_z_measurement to C?
    cdef float p = 1.0
    state_s, state_p = state_sp_tuple
    for i,outcm in enumerate(moutcomes):
        p0,p1,ss0,ss1,sp0,sp1 = symplectic.pauli_z_measurement(state_s, state_p, i)

        if outcm == 0:
            p *= p0; state_s, state_p = ss0, sp0
        else:
            p *= p1; state_s, state_p = ss1, sp1
    return p

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def float_product(np.ndarray[double, ndim=1] ar):
    cdef double ret = 1.0
    cdef INT N = ar.shape[0]
    cdef INT i
    for i in range(N):
        ret = ret * ar[i]
    return ret

def dot(np.ndarray[double, ndim=1] f, np.ndarray[double, ndim=1] g):
    cdef long N = f.shape[0]
    cdef float ret = 0.0
    cdef int i
    for i in range(N):
        ret += f[i]*g[i]
    return ret

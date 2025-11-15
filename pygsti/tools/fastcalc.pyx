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
from functools import lru_cache
cimport numpy as np
cimport cython

ctypedef long INT

def dot(np.ndarray[double, ndim=1] f, np.ndarray[double, ndim=1] g):
    cdef long N = f.shape[0]
    cdef float ret = 0.0
    cdef INT i
    for i in range(N):
        ret += f[i]*g[i]
    return ret

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def embedded_fast_acton_sparse(embedded_gate_acton_fn,
                               np.ndarray[double, ndim=1] output_state,
                               np.ndarray[double, ndim=1] state,
                               np.ndarray[np.int64_t, ndim=1] noop_incrementers,
                               np.ndarray[np.int64_t, ndim=1] numBasisEls_noop_blankaction,
                               np.ndarray[np.int64_t, ndim=1] baseinds):

    cdef long i
    cdef INT k
    cdef INT vec_index_noop = 0
    cdef INT nParts = numBasisEls_noop_blankaction.shape[0]
    cdef INT nActionIndices = baseinds.shape[0]
    #cdef np.ndarray b = np.zeros(nParts, dtype=np.int64_t)
    #cdef np.ndarray op_b = np.zeros(nAction, dtype=np.int64_t)
    #cdef np.ndarray[long, ndim=1] baseinds = np.empty(nActionIndices, dtype=np.int64_t) #for FASTER
    cdef INT b[100]

    #These need to be numpy arrays for python interaction
    cdef np.ndarray[double, ndim=1, mode="c"] slc1 = np.empty(nActionIndices, dtype='d')
    cdef np.ndarray[double, ndim=1, mode="c"] slc2 = np.empty(nActionIndices, dtype='d')

    # nActionIndices = np.prod(numBasisEls_action)
    #for i in range(nAction):
    #    nActionIndices *= numBasisEls_action[i]

    if nParts > 100: assert(0) # need to increase size of static arrays above
    for i in range(nParts): b[i] = 0

    #vec_index_noop = 0 assigned above
    while(True):

        #Act with embedded gate on appropriate sub-space of state
        #output_state[ inds ] += embedded_gate_acton_fn( state[inds] ) #Fancy indexing...
        #output_state[inds] += state[inds]
        for k in range(nActionIndices):
            slc1[k] = state[ vec_index_noop+baseinds[k] ]
        slc2 = embedded_gate_acton_fn( slc1 )
        for k in range(nActionIndices):
            output_state[ vec_index_noop+baseinds[k] ] += slc2[k] #state[ inds[k] ]

        #increment b ~ itertools.product & update vec_index_noop = _np.dot(self.multipliers, b)
        for i in range(nParts-1,-1,-1):
            if b[i]+1 < numBasisEls_noop_blankaction[i]:
                b[i] += 1; vec_index_noop += noop_incrementers[i]
                break
            else:
                b[i] = 0
        else:
            break # can't increment anything - break while(True) loop

    return output_state

# i = 0 1 2 3
# -----------
# N = 2 3 1 2
#     0 0 0 0
#     0 0 0 1 + m[3]
#     0 1 0 0 + m[1] - 1*m[3]
#     0 1 0 1 + m[3]
#     0 2 0 0 + m[1] - 1*m[3]
#     0 2 0 1 + m[3]
#     1 0 0 0 + m[0] - 2*m[1] - 1*m[3]
#     1 0 0 1 + m[3]
#     1 1 0 0 + m[1] - 1*m[3]
#     1 1 0 1 + m[3]
#     1 2 0 0 + m[1] - 1*m[3]
#     1 2 0 1 + m[3]
#END

#SPECIAL CASE 1: embedded gate is Lindblad gate with no unitary postfactor -
# so just pass the args to custom_expm_multiply_simple_core
#@cython.profile(True)
#@cython.linetrace(True)
#@cython.binding(True)
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def embedded_fast_acton_sparse_spc1(
        np.ndarray[double, ndim=1, mode="c"] Adata not None,
        np.ndarray[np.int64_t, ndim=1, mode="c"] Aindptr not None,
        np.ndarray[np.int64_t, ndim=1, mode="c"] Aindices not None,
        double mu, INT m_star, INT s, double tol, double eta,
                               np.ndarray[double, ndim=1] output_state,
                               np.ndarray[double, ndim=1] state,
                               np.ndarray[np.int64_t, ndim=1] noop_incrementers,
                               np.ndarray[np.int64_t, ndim=1] numBasisEls_noop_blankaction,
                               np.ndarray[np.int64_t, ndim=1] baseinds):

    #                          INT offset,
    #                          np.ndarray[np.int64_t, ndim=1] numBasisEls_action,
    #                          np.ndarray[np.int64_t, ndim=1] actionInds,

    cdef long i
    cdef INT k
    cdef INT vec_index_noop = 0
    cdef long nParts = numBasisEls_noop_blankaction.shape[0]
    #cdef INT nAction = numBasisEls_action.shape[0]
    cdef long nActionIndices = baseinds.shape[0]
    cdef INT b[100]
    #cdef INT op_b[100]
    cdef long Annz = Adata.shape[0]

    #Note: malloc just as fast as stack alloc
    #cdef np.ndarray[double, ndim=1, mode="c"] slc1 = np.empty(nActionIndices, dtype='d')
    #cdef np.ndarray[double, ndim=1, mode="c"] slc2 = np.empty(nActionIndices, dtype='d')
    #cdef np.ndarray[double, ndim=1, mode="c"] scratch = np.empty(nActionIndices, dtype='d')
    cdef double *slc1 = <double *>malloc(nActionIndices * sizeof(double))
    cdef double *slc2 = <double *>malloc(nActionIndices * sizeof(double))
    cdef double *scratch = <double *>malloc(nActionIndices * sizeof(double))

    if not slc1 or not slc2 or not scratch: # or not inds:
        raise MemoryError()

    if nParts > 100:
        raise ValueError("Need to increase size of static arrays!")
    for i in range(nParts): b[i] = 0
    #for i in range(nActionIndices): inds[i] = baseinds[i]


    #vec_index_noop = 0 assigned above
    while(True):

        if Annz > 0:
            #Act with embedded gate on appropriate sub-space of state
            for k in range(nActionIndices):
                slc1[k] = state[ vec_index_noop+baseinds[k] ]# inds[k] ]

            #SPECIAL ACTON for output_state[ inds ] += acton( state[inds] )
            # replaces:  slc2 = embedded_gate_acton_fn( slc1 )
            custom_expm_multiply_simple_core_c(&Adata[0], <INT*>&Aindptr[0],
                                               <INT*>&Aindices[0], &slc1[0], nActionIndices,
                                               mu, m_star, s, tol, eta,
                                               &slc2[0], &scratch[0])

            for k in range(nActionIndices):
                output_state[ vec_index_noop+baseinds[k] ] += slc2[k] #state[ inds[k] ]

        else: #act as identity
            for k in range(nActionIndices):
                output_state[vec_index_noop+baseinds[k]] += state[vec_index_noop+baseinds[k]]

        #increment b ~ itertools.product & update vec_index_noop = _np.dot(self.multipliers, b)
        for i in range(nParts-1,-1,-1):
            if b[i]+1 < numBasisEls_noop_blankaction[i]:
                b[i] += 1; vec_index_noop += noop_incrementers[i]
                break
            else:
                b[i] = 0
        else:
            break # can't increment anything - break while(True) loop

    free(slc1)
    free(slc2)
    free(scratch)
    return output_state


#Special case #2: embedded gate has a dense matrix representation
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def embedded_fast_acton_sparse_spc2(np.ndarray[double, ndim=2, mode="c"] densemx not None,
                                    np.ndarray[double, ndim=1] output_state,
                                    np.ndarray[double, ndim=1] state,
                                    np.ndarray[np.int64_t, ndim=1] noop_incrementers,
                                    np.ndarray[np.int64_t, ndim=1] numBasisEls_noop_blankaction,
                                    np.ndarray[np.int64_t, ndim=1] baseinds):

    cdef long i
    cdef long j
    cdef double cum
    cdef INT k
    cdef INT vec_index_noop = 0
    cdef long nParts = numBasisEls_noop_blankaction.shape[0]
    cdef long nActionIndices = baseinds.shape[0]
    cdef INT b[100]

    #Note: malloc just as fast as stack alloc
    #cdef double *slc1 = <double *>malloc(nActionIndices * sizeof(double))
    #cdef double *slc2 = <double *>malloc(nActionIndices * sizeof(double))
    #if not slc1 or not slc2:
    #    raise MemoryError()

    if nParts > 100:
        raise ValueError("Need to increase size of static arrays!")
    for i in range(nParts): b[i] = 0
    #for i in range(nActionIndices): inds[i] = baseinds[i]


    #vec_index_noop = 0 assigned above
    while(True):

        #Act with embedded gate on appropriate sub-space of state
        #for k in range(nActionIndices):
        #    slc1[k] = state[ vec_index_noop+baseinds[k] ]

        #SPECIAL ACTON for output_state[ inds ] += acton( state[inds] )
        # replaces:  slc2 = embedded_gate_acton_fn( slc1 )
        # Dense matrix multiplication: w_i = sum_j M_ij * v_j
        for i in range(nActionIndices):
            cum = densemx[i,0] * state[ vec_index_noop+baseinds[0] ]
            for j in range(1,nActionIndices):
                cum += densemx[i,j] * state[ vec_index_noop+baseinds[j] ]
            output_state[ vec_index_noop+baseinds[i] ] += cum

        #for k in range(nActionIndices):
        #    output_state[ vec_index_noop+baseinds[k] ] += slc2[k] #state[ inds[k] ]

        #increment b ~ itertools.product & update vec_index_noop = _np.dot(self.multipliers, b)
        for i in range(nParts-1,-1,-1):
            if b[i]+1 < numBasisEls_noop_blankaction[i]:
                b[i] += 1; vec_index_noop += noop_incrementers[i]
                break
            else:
                b[i] = 0
        else:
            break # can't increment anything - break while(True) loop

    #free(slc1)
    #free(slc2)
    return output_state



@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def embedded_fast_acton_sparse_complex(embedded_gate_acton_fn,
                                       np.ndarray[np.complex128_t, ndim=1] output_state,
                                       np.ndarray[np.complex128_t, ndim=1] state,
                                       np.ndarray[np.int64_t, ndim=1] noop_incrementers,
                                       np.ndarray[np.int64_t, ndim=1] numBasisEls_noop_blankaction,
                                       np.ndarray[np.int64_t, ndim=1] baseinds):

    cdef long i
    cdef INT k
    cdef INT vec_index_noop = 0
    cdef INT nParts = numBasisEls_noop_blankaction.shape[0]
    cdef INT nActionIndices = baseinds.shape[0]
    #cdef np.ndarray b = np.zeros(nParts, dtype=np.int64_t)
    #cdef np.ndarray op_b = np.zeros(nAction, dtype=np.int64_t)
    #cdef np.ndarray[long, ndim=1] baseinds = np.empty(nActionIndices, dtype=np.int64_t) #for FASTER
    cdef INT b[100]

    #These need to be numpy arrays for python interaction
    cdef np.ndarray[np.complex128_t, ndim=1, mode="c"] slc1 = np.empty(nActionIndices, dtype=np.complex128)
    cdef np.ndarray[np.complex128_t, ndim=1, mode="c"] slc2 = np.empty(nActionIndices, dtype=np.complex128)

    # nActionIndices = np.prod(numBasisEls_action)
    #for i in range(nAction):
    #    nActionIndices *= numBasisEls_action[i]

    if nParts > 100: assert(0) # need to increase size of static arrays above
    for i in range(nParts): b[i] = 0

    #vec_index_noop = 0 assigned above
    while(True):

        #Act with embedded gate on appropriate sub-space of state
        #output_state[ inds ] += embedded_gate_acton_fn( state[inds] ) #Fancy indexing...
        #output_state[inds] += state[inds]
        for k in range(nActionIndices):
            slc1[k] = state[ vec_index_noop+baseinds[k] ]
        slc2 = embedded_gate_acton_fn( slc1 )
        for k in range(nActionIndices):
            output_state[ vec_index_noop+baseinds[k] ] = output_state[ vec_index_noop+baseinds[k] ] + slc2[k] #state[ inds[k] ]
              # Note: in-place addition doesn't compile correctly with complex type

        #increment b ~ itertools.product & update vec_index_noop = _np.dot(self.multipliers, b)
        for i in range(nParts-1,-1,-1):
            if b[i]+1 < numBasisEls_noop_blankaction[i]:
                b[i] += 1; vec_index_noop += noop_incrementers[i]
                break
            else:
                b[i] = 0
        else:
            break # can't increment anything - break while(True) loop

    return output_state

# i = 0 1 2 3
# -----------
# N = 2 3 1 2
#     0 0 0 0
#     0 0 0 1 + m[3]
#     0 1 0 0 + m[1] - 1*m[3]
#     0 1 0 1 + m[3]
#     0 2 0 0 + m[1] - 1*m[3]
#     0 2 0 1 + m[3]
#     1 0 0 0 + m[0] - 2*m[1] - 1*m[3]
#     1 0 0 1 + m[3]
#     1 1 0 0 + m[1] - 1*m[3]
#     1 1 0 1 + m[3]
#     1 2 0 0 + m[1] - 1*m[3]
#     1 2 0 1 + m[3]
#END

#SPECIAL CASE 1: embedded gate is Lindblad gate with no unitary postfactor -
# so just pass the args to custom_expm_multiply_simple_core
#@cython.profile(True)
#@cython.linetrace(True)
#@cython.binding(True)
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def embedded_fast_acton_sparse_spc1_complex(
        np.ndarray[np.complex128_t, ndim=1, mode="c"] Adata not None,
        np.ndarray[np.int64_t, ndim=1, mode="c"] Aindptr not None,
        np.ndarray[np.int64_t, ndim=1, mode="c"] Aindices not None,
        double mu, INT m_star, INT s, double tol, double eta,
                               np.ndarray[np.complex128_t, ndim=1] output_state,
                               np.ndarray[np.complex128_t, ndim=1] state,
                               np.ndarray[np.int64_t, ndim=1] noop_incrementers,
                               np.ndarray[np.int64_t, ndim=1] numBasisEls_noop_blankaction,
                               np.ndarray[np.int64_t, ndim=1] baseinds):

    #                          INT offset,
    #                          np.ndarray[np.int64_t, ndim=1] numBasisEls_action,
    #                          np.ndarray[np.int64_t, ndim=1] actionInds,

    cdef long i
    cdef INT k
    cdef INT vec_index_noop = 0
    cdef long nParts = numBasisEls_noop_blankaction.shape[0]
    #cdef INT nAction = numBasisEls_action.shape[0]
    cdef long nActionIndices = baseinds.shape[0]
    cdef INT b[100]
    #cdef INT op_b[100]
    cdef long Annz = Adata.shape[0]

    #Note: malloc just as fast as stack alloc
    #cdef np.ndarray[double, ndim=1, mode="c"] slc1 = np.empty(nActionIndices, dtype='d')
    #cdef np.ndarray[double, ndim=1, mode="c"] slc2 = np.empty(nActionIndices, dtype='d')
    #cdef np.ndarray[double, ndim=1, mode="c"] scratch = np.empty(nActionIndices, dtype='d')
    cdef double complex *slc1 = <double complex *>malloc(nActionIndices * sizeof(double complex))
    cdef double complex *slc2 = <double complex *>malloc(nActionIndices * sizeof(double complex))
    cdef double complex *scratch = <double complex *>malloc(nActionIndices * sizeof(double complex))

    if not slc1 or not slc2 or not scratch: # or not inds:
        raise MemoryError()

    if nParts > 100:
        raise ValueError("Need to increase size of static arrays!")
    for i in range(nParts): b[i] = 0
    #for i in range(nActionIndices): inds[i] = baseinds[i]


    #vec_index_noop = 0 assigned above
    while(True):

        if Annz > 0:
            #Act with embedded gate on appropriate sub-space of state
            for k in range(nActionIndices):
                slc1[k] = state[ vec_index_noop+baseinds[k] ]# inds[k] ]

            #SPECIAL ACTON for output_state[ inds ] += acton( state[inds] )
            # replaces:  slc2 = embedded_gate_acton_fn( slc1 )
            custom_expm_multiply_simple_core_c_complex(
                &Adata[0], <INT*>&Aindptr[0],
                <INT*>&Aindices[0], &slc1[0], nActionIndices,
                mu, m_star, s, tol, eta,
                &slc2[0], &scratch[0])

            for k in range(nActionIndices):
                output_state[ vec_index_noop+baseinds[k] ] = output_state[ vec_index_noop+baseinds[k] ] + slc2[k] #state[ inds[k] ]
                  # Note: in-place addition doesn't compile correctly with complex type

        else: #act as identity
            for k in range(nActionIndices):
                output_state[vec_index_noop+baseinds[k]] = output_state[vec_index_noop+baseinds[k]] + state[vec_index_noop+baseinds[k]]
                  # Note: in-place addition doesn't compile correctly with complex type

        #increment b ~ itertools.product & update vec_index_noop = _np.dot(self.multipliers, b)
        for i in range(nParts-1,-1,-1):
            if b[i]+1 < numBasisEls_noop_blankaction[i]:
                b[i] += 1; vec_index_noop += noop_incrementers[i]
                break
            else:
                b[i] = 0
        else:
            break # can't increment anything - break while(True) loop

    free(slc1)
    free(slc2)
    free(scratch)
    return output_state


#Special case #2: embedded gate has a dense matrix representation
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def embedded_fast_acton_sparse_spc2_complex(np.ndarray[np.complex128_t, ndim=2, mode="c"] densemx not None,
                                            np.ndarray[np.complex128_t, ndim=1] output_state,
                                            np.ndarray[np.complex128_t, ndim=1] state,
                                            np.ndarray[np.int64_t, ndim=1] noop_incrementers,
                                            np.ndarray[np.int64_t, ndim=1] numBasisEls_noop_blankaction,
                                            np.ndarray[np.int64_t, ndim=1] baseinds):

    cdef long i
    cdef long j
    cdef double complex cum
    cdef INT k
    cdef INT vec_index_noop = 0
    cdef long nParts = numBasisEls_noop_blankaction.shape[0]
    cdef long nActionIndices = baseinds.shape[0]
    cdef INT b[100]

    if nParts > 100:
        raise ValueError("Need to increase size of static arrays!")
    for i in range(nParts): b[i] = 0
    #for i in range(nActionIndices): inds[i] = baseinds[i]


    #vec_index_noop = 0 assigned above
    while(True):

        #Act with embedded gate on appropriate sub-space of state
        #for k in range(nActionIndices):
        #    slc1[k] = state[ vec_index_noop+baseinds[k] ]

        #SPECIAL ACTON for output_state[ inds ] += acton( state[inds] )
        # replaces:  slc2 = embedded_gate_acton_fn( slc1 )
        # Dense matrix multiplication: w_i = sum_j M_ij * v_j
        for i in range(nActionIndices):
            cum = densemx[i,0] * state[ vec_index_noop+baseinds[0] ]
            for j in range(1,nActionIndices):
                cum += densemx[i,j] * state[ vec_index_noop+baseinds[j] ]
            output_state[ vec_index_noop+baseinds[i] ] = output_state[ vec_index_noop+baseinds[i] ] + cum
              # Note: in-place addition doesn't compile correctly with complex type

        #for k in range(nActionIndices):
        #    output_state[ vec_index_noop+baseinds[k] ] += slc2[k] #state[ inds[k] ]

        #increment b ~ itertools.product & update vec_index_noop = _np.dot(self.multipliers, b)
        for i in range(nParts-1,-1,-1):
            if b[i]+1 < numBasisEls_noop_blankaction[i]:
                b[i] += 1; vec_index_noop += noop_incrementers[i]
                break
            else:
                b[i] = 0
        else:
            break # can't increment anything - break while(True) loop

    #free(slc1)
    #free(slc2)
    return output_state


# Embed densemx and add result to outputmx
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def fast_add_embeded(np.ndarray[double, ndim=2, mode="c"] densemx not None,
                     np.ndarray[double, ndim=2, mode="c"] outputmx not None,
                     np.ndarray[np.int64_t, ndim=1] noop_incrementers,
                     np.ndarray[np.int64_t, ndim=1] numBasisEls_noop_blankaction,
                     np.ndarray[np.int64_t, ndim=1] baseinds):

    cdef long i
    cdef long j
    cdef double cum
    cdef INT k
    cdef INT vec_index_noop = 0
    cdef long nParts = numBasisEls_noop_blankaction.shape[0]
    cdef long nActionIndices = baseinds.shape[0]

    cdef INT b[100]
    if nParts > 100:
        raise ValueError("Need to increase size of static arrays!")

    for i in range(nParts): b[i] = 0

    #vec_index_noop = 0 assigned above
    while(True):
        #Act with embedded gate on appropriate sub-space of state
        for i in range(nActionIndices):
            for j in range(nActionIndices):
                outputmx[vec_index_noop + baseinds[i], vec_index_noop + baseinds[j]] += densemx[i, j]

        #increment b ~ itertools.product & update vec_index_noop = _np.dot(self.multipliers, b)
        for i in range(nParts - 1, -1, -1):
            if b[i] + 1 < numBasisEls_noop_blankaction[i]:
                b[i] += 1; vec_index_noop += noop_incrementers[i]
                break
            else:
                b[i] = 0
        else:
            break  # can't increment anything - break while(True) loop

    return


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def medium_kron(np.ndarray[double, ndim=1, mode="c"] outvec not None,
              np.ndarray[double, ndim=2, mode="c"] fastArray not None,
              np.ndarray[np.int64_t, ndim=1, mode="c"] fastArraySizes not None):

    cdef INT mi[100] # multi-index holding
    cdef INT multipliers[100]
    cdef double preprods[101] # +1 from other static dims
    cdef INT nFactors = fastArray.shape[0]
    cdef INT i
    cdef INT k
    cdef INT p

    if nFactors > 100:
        assert(False) # need to increase static dimensions above

    #set indices to zero
    k=0
    for i in range(nFactors): mi[i] = 0

    # preprods[i] = prod_{k<i}( fastArray[k,m[k]] ) i.e. the product of the first i-1 factors
    # this means that preprods[nFactors] == prod, the final product to assign to outvec
    preprods[0] = 1.0
    for i in range(nFactors):
        preprods[i+1] = preprods[i] * fastArray[i,0] # 0 b/c m[i] == 0

    #multipliers[i] gives multiplicative factor for i-th element of mi
    # when computing the total index 'k'
    multipliers[nFactors-1] = 1
    for i in range(nFactors-2,-1,-1):
        multipliers[i] = multipliers[i+1]*fastArraySizes[i+1]

    #loop over indices (incrementing mi & updating k and preprods as we go)
    while True:
        outvec[k] = preprods[nFactors]

        #increment mi as a multindex
        for i in range(nFactors-1,-1,-1):
            if mi[i]+1 < fastArraySizes[i]:
                mi[i] += 1; k += multipliers[i]
                preprods[i+1] = preprods[i]*fastArray[i,mi[i]] #ok even if i+1 == nFactors
                for p in range(i+1,nFactors): #all other factors have index=0
                    preprods[p+1] = preprods[p]*fastArray[p,0]
                break
            else: #can't increment, so set index back to zero
                k -= mi[i]*multipliers[i]; mi[i] = 0
        else:
            break # can't increment anything - break while(True) loop


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def fast_kron(np.ndarray[double, ndim=1, mode="c"] outvec not None,
              np.ndarray[double, ndim=2, mode="c"] fastArray not None,
              np.ndarray[np.int64_t, ndim=1, mode="c"] fastArraySizes not None):

    cdef INT nFactors = fastArray.shape[0]
    cdef INT N = outvec.shape[0]
    cdef INT i
    cdef INT j
    cdef INT k
    cdef INT sz
    cdef INT off
    cdef INT endoff
    cdef double mult

    #Put last factor at end of outvec
    k = nFactors-1  #last factor
    off = N-fastArraySizes[k] #offset into outvec
    for i in range(fastArraySizes[k]):
        outvec[off+i] = fastArray[k,i]
    sz = fastArraySizes[k]

    #Repeatedly scale&copy last "sz" elements of outputvec forward
    # (as many times as there are elements in the current factor array)
    # - but multiply *in-place* the last "sz" elements.
    for k in range(nFactors-2,-1,-1): #for all but the last factor
        off = N-sz*fastArraySizes[k]
        endoff = N-sz

        #For all but the final element of fastArray[k,:],
        # mult&copy final sz elements of outvec into position
        for j in range(fastArraySizes[k]-1):
            mult = fastArray[k,j]
            for i in range(sz):
                outvec[off+i] = mult*outvec[endoff+i]
            off += sz

        #Last element: in-place mult
        #assert(off == endoff)
        mult = fastArray[k, fastArraySizes[k]-1]
        for i in range(sz):
            outvec[endoff+i] *= mult
        sz *= fastArraySizes[k]

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def fast_kron_complex(np.ndarray[np.complex128_t, ndim=1, mode="c"] outvec not None,
                      np.ndarray[np.complex128_t, ndim=2, mode="c"] fastArray not None,
                      np.ndarray[np.int64_t, ndim=1, mode="c"] fastArraySizes not None):

    cdef INT nFactors = fastArray.shape[0]
    cdef INT N = outvec.shape[0]
    cdef INT i
    cdef INT j
    cdef INT k
    cdef INT sz
    cdef INT off
    cdef INT endoff
    cdef double complex mult

    #Put last factor at end of outvec
    k = nFactors-1  #last factor
    off = N-fastArraySizes[k] #offset into outvec
    for i in range(fastArraySizes[k]):
        outvec[off+i] = fastArray[k,i]
    sz = fastArraySizes[k]

    #Repeatedly scale&copy last "sz" elements of outputvec forward
    # (as many times as there are elements in the current factor array)
    # - but multiply *in-place* the last "sz" elements.
    for k in range(nFactors-2,-1,-1): #for all but the last factor
        off = N-sz*fastArraySizes[k]
        endoff = N-sz

        #For all but the final element of fastArray[k,:],
        # mult&copy final sz elements of outvec into position
        for j in range(fastArraySizes[k]-1):
            mult = fastArray[k,j]
            for i in range(sz):
                outvec[off+i] = mult*outvec[endoff+i]
            off += sz

        #Last element: in-place mult
        #assert(off == endoff)
        mult = fastArray[k, fastArraySizes[k]-1]
        for i in range(sz):
            outvec[endoff+i] = outvec[endoff+i] * mult
              # Note: in-place multiplication doesn't compile correctly with complex type
        sz *= fastArraySizes[k]

    #assert(sz == N)


@cython.cdivision(True) # turn off divide-by-zero checking
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def custom_expm_multiply_simple_core(np.ndarray[double, ndim=1, mode="c"] Adata,
                                     np.ndarray[np.int64_t, ndim=1, mode="c"] Aindptr,
                                     np.ndarray[np.int64_t, ndim=1, mode="c"] Aindices,
                                     np.ndarray[double, ndim=1, mode="c"] B,
                                     double mu, INT m_star, INT s, double tol, double eta):

    cdef INT N = B.shape[0] #Aindptr.shape[0]-1
    if s == 0: return B #short circuit

    cdef np.ndarray[double, ndim=1, mode="c"] F = np.empty(N,'d')
    cdef np.ndarray[double, ndim=1, mode="c"] scratch = np.empty(N,'d')

    custom_expm_multiply_simple_core_c(&Adata[0], <INT*>&Aindptr[0],
                                       <INT*>&Aindices[0], &B[0], N,
                                       mu, m_star, s, tol, eta,
                                       &F[0], &scratch[0])
    return F

@cython.cdivision(True) # turn off divide-by-zero checking
cdef custom_expm_multiply_simple_core_c(double* Adata, INT* Aindptr,
                                        INT* Aindices, double* B,
                                        INT N, double mu, INT m_star,
                                        INT s, double tol, double eta,
                                        double* F, double* scratch):

    cdef INT i
    cdef INT j
    cdef INT r
    cdef INT k

    cdef double a
    cdef double c1
    cdef double c2
    cdef double coeff
    cdef double normF

    #F = B
    for i in range(N): F[i] = B[i]

    for i in range(s):
        if m_star > 0: #added by EGN
            #c1 = vec_inf_norm(B) #_exact_inf_norm(B)
            c1 = 0.0
            for k in range(N):
                a = abs(B[k])
                if a > c1: c1 = a

        for j in range(m_star):
            coeff = 1.0 / (s*(j+1)) # t == 1.0

            #B = coeff * A.dot(B)
            # inline csr_matvec: implements result = coeff * A * B
            for r in range(N):
                scratch[r] = 0
                for k in range(Aindptr[r],Aindptr[r+1]):
                    scratch[r] += Adata[k] * B[ Aindices[k]]


            ##if False: #j % 3 == 0: #every == 3 #TODO: work on this
            c2 = 0.0
            normF = 0.0
            for k in range(N):
                B[k] = coeff * scratch[k] #finishes B = coeff * A.dot(B)
                F[k] += B[k] #F += B

                a = abs(B[k])
                if a > c2: c2 = a #c2 = vec_inf_norm(B) #_exact_inf_norm(B)
                a = abs(F[k])
                if a > normF: normF = a #normF = vec_inf_norm(F) #_exact_inf_norm(F)

            #print("Iter %d,%d of %d,%d: %g+%g=%g < %g?" % (i,j,s,m_star,c1,c2,c1+c2,tol*normF))
            if c1 + c2 <= tol * normF:
                #print(" --> YES - break early at %d of %d" % (i+1,s))
                break
            c1 = c2

        #F *= eta
        #B = F
        for k in range(N):
            F[k] *= eta
            B[k] = F[k]

    #return F # updates passed-in memory, so don't need this


@cython.cdivision(True) # turn off divide-by-zero checking
cdef custom_expm_multiply_simple_core_c_complex(double complex* Adata, INT* Aindptr,
                                                INT* Aindices, double complex* B,
                                                INT N, double mu, INT m_star,
                                                INT s, double tol, double eta,
                                                double complex* F, double complex* scratch):

    cdef INT i
    cdef INT j
    cdef INT r
    cdef INT k

    cdef double a
    cdef double c1
    cdef double c2
    cdef double coeff
    cdef double normF

    #F = B
    for i in range(N): F[i] = B[i]

    for i in range(s):
        if m_star > 0: #added by EGN
            #c1 = vec_inf_norm(B) #_exact_inf_norm(B)
            c1 = 0.0
            for k in range(N):
                a = abs(B[k])
                if a > c1: c1 = a

        for j in range(m_star):
            coeff = 1.0 / (s*(j+1)) # t == 1.0

            #B = coeff * A.dot(B)
            # inline csr_matvec: implements result = coeff * A * B
            for r in range(N):
                scratch[r] = 0
                for k in range(Aindptr[r],Aindptr[r+1]):
                    scratch[r] += Adata[k] * B[ Aindices[k]]


            ##if False: #j % 3 == 0: #every == 3 #TODO: work on this
            c2 = 0.0
            normF = 0.0
            for k in range(N):
                B[k] = coeff * scratch[k] #finishes B = coeff * A.dot(B)
                F[k] += B[k] #F += B

                a = abs(B[k])
                if a > c2: c2 = a #c2 = vec_inf_norm(B) #_exact_inf_norm(B)
                a = abs(F[k])
                if a > normF: normF = a #normF = vec_inf_norm(F) #_exact_inf_norm(F)

            #print("Iter %d,%d of %d,%d: %g+%g=%g < %g?" % (i,j,s,m_star,c1,c2,c1+c2,tol*normF))
            if c1 + c2 <= tol * normF:
                #print(" --> YES - break early at %d of %d" % (i+1,s))
                break
            c1 = c2

        #F *= eta
        #B = F
        for k in range(N):
            F[k] *= eta
            B[k] = F[k]

    #return F # updates passed-in memory, so don't need this



# Implements B = A - lmb*I; returns used length of Bindices/Bdata
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def csr_subtract_identity(np.ndarray[double, ndim=1] Adata,
                          np.ndarray[np.int64_t, ndim=1] Aindptr,
                          np.ndarray[np.int64_t, ndim=1] Aindices,
                          np.ndarray[double, ndim=1] Bdata,
                          np.ndarray[np.int64_t, ndim=1] Bindptr,
                          np.ndarray[np.int64_t, ndim=1] Bindices,
                          double lmb, INT n):

    cdef INT nxt = 0
    cdef INT iRow = 0
    cdef INT i = 0
    cdef INT bFound = 0
    Bindptr[0] = 0

    for iRow in range(n):
        bFound = 0
        for i in range(Aindptr[iRow],Aindptr[iRow+1]):
            Bindices[nxt] = Aindices[i]
            if Aindices[i] == iRow:
                Bdata[nxt] = Adata[i] + lmb
                bFound = 1
            else:
                Bdata[nxt] = Adata[i]
            nxt += 1
        if not bFound: #insert new diagonal element
            Bindices[nxt] = iRow
            Bdata[nxt] = lmb
            nxt += 1
        Bindptr[iRow+1] = nxt

    return nxt

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def fast_fas_helper_1d(np.ndarray[double, ndim=1] a,
                       np.ndarray[double, ndim=1] rhs,
                       np.ndarray[np.int64_t, mode="c", ndim=1] inds0):
    cdef INT nDims = 1
    cdef INT b[1]
    cdef INT a_strides[1]
    cdef INT rhs_strides[1]
    cdef INT rhs_dims[1]
    cdef INT rhs_indx = 0
    cdef INT a_indx = 0

    for i in range(nDims):
        b[i] = 0
        a_strides[i] = a.strides[i] // a.itemsize
        rhs_strides[i] = rhs.strides[i] // rhs.itemsize
        rhs_dims[i] = rhs.shape[i]

    a_indx += inds0[0] * a_strides[0]

    cdef double* a_ptr = <double*>a.data
    cdef double* rhs_ptr = <double*>rhs.data

    if rhs.flags['C_CONTIGUOUS']:
        while(True):
            a_ptr[a_indx] = rhs_ptr[rhs_indx]
            rhs_indx += 1 # always increments by 1 (1D and contiguous)
    
            #increment b ~ itertools.product        
            if b[0]+1 < rhs_dims[0]: # "i = 0" loop
                a_indx += (inds0[b[0]+1] - inds0[b[0]]) * a_strides[0]
                b[0] += 1
            else:
                break # can't increment anything - break while(True) loop
        
            #For general nDims (but we unroll for speed)
            #for i in range(nDims-1,-1,-1):
            #    if b[i]+1 < rhs_dims[i]:
            #        a_indx += (indsPerDim[i][b[i]+1] - indsPerDim[i][b[i]]) * a_strides[i]
            #        b[i] += 1
            #        break
            #    else:
            #        a_indx += (indsPerDim[i][0]-indsPerDim[i][b[i]]) * a_strides[i]
            #        b[i] = 0
            #else:
            #    break # can't increment anything - break while(True) loop
            
    else: # rhs is *not* c-contiguous, so need to use its rhs_strides
        while(True):
            a_ptr[a_indx] = rhs_ptr[rhs_indx]
            rhs_indx += rhs_strides[0] # always increments by strides[0] (only 1D)

            #increment b ~ itertools.product        
            if b[0]+1 < rhs_dims[0]: # "i = 0" loop
                a_indx += (inds0[b[0]+1] - inds0[b[0]]) * a_strides[0]
                b[0] += 1
            else:
                break # can't increment anything - break while(True) loop


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def fast_fas_helper_2d(np.ndarray[double, ndim=2] a,
                       np.ndarray[double, ndim=2] rhs,
                       np.ndarray[np.int64_t, mode="c", ndim=1] inds0,
                       np.ndarray[np.int64_t, mode="c", ndim=1] inds1):

    cdef INT nDims = 2
    cdef INT b[2]
    cdef INT a_strides[2]
    cdef INT rhs_strides[2]
    cdef INT rhs_dims[2]
    cdef INT rhs_indx = 0
    cdef INT a_indx = 0

    for i in range(nDims):
        b[i] = 0
        a_strides[i] = a.strides[i] // a.itemsize
        rhs_strides[i] = rhs.strides[i] // rhs.itemsize
        rhs_dims[i] = rhs.shape[i]

    a_indx += inds0[0] * a_strides[0]
    a_indx += inds1[0] * a_strides[1]

    cdef double* a_ptr = <double*>a.data
    cdef double* rhs_ptr = <double*>rhs.data

    if rhs.flags['C_CONTIGUOUS']:
        while(True):
            a_ptr[a_indx] = rhs_ptr[rhs_indx]
            rhs_indx += 1 # always increments by 1

            #increment b ~ itertools.product
            if b[1]+1 < rhs_dims[1]: # "i = 1" loop
                a_indx += (inds1[b[1]+1]-inds1[b[1]]) * a_strides[1]
                b[1] += 1
            else:
                a_indx += (inds1[0]-inds1[b[1]]) * a_strides[1]
                b[1] = 0
                if b[0]+1 < rhs_dims[0]: # "i = 0" loop
                    a_indx += (inds0[b[0]+1]-inds0[b[0]]) * a_strides[0]
                    b[0] += 1
                else:
                    break # can't increment anything - break while(True) loop

    else: # rhs is *not* c-contiguous, so need to use its strides
        while(True):
            a_ptr[a_indx] = rhs_ptr[rhs_indx]

            #increment b ~ itertools.product
            if b[1]+1 < rhs_dims[1]: # "i = 1" loop
                a_indx += (inds1[b[1]+1]-inds1[b[1]]) * a_strides[1]
                rhs_indx += rhs_strides[1]
                b[1] += 1
            else:
                a_indx += (inds1[0]-inds1[b[1]]) * a_strides[1]
                rhs_indx -= b[1]*rhs_strides[1]
                b[1] = 0
                if b[0]+1 < rhs_dims[0]: # "i = 0" loop
                    a_indx += (inds0[b[0]+1]-inds0[b[0]]) * a_strides[0]
                    rhs_indx += rhs_strides[0]
                    b[0] += 1
                else:
                    break # can't increment anything - break while(True) loop



@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def fast_fas_helper_3d(np.ndarray[double, ndim=3] a,
                       np.ndarray[double, ndim=3] rhs,
                       np.ndarray[np.int64_t, mode="c", ndim=1] inds0,
                       np.ndarray[np.int64_t, mode="c", ndim=1] inds1,
                       np.ndarray[np.int64_t, mode="c", ndim=1] inds2):

    cdef INT nDims = 3
    cdef INT b[3]
    cdef INT a_strides[3]
    cdef INT rhs_strides[3]
    cdef INT rhs_dims[3]
    cdef INT rhs_indx = 0
    cdef INT a_indx = 0

    for i in range(nDims):
        b[i] = 0
        a_strides[i] = a.strides[i] // a.itemsize
        rhs_strides[i] = rhs.strides[i] // rhs.itemsize
        rhs_dims[i] = rhs.shape[i]

    #for i in range(nDims):
    #    a_indx += indsPerDim[i][0] * a_strides[i]
    a_indx += inds0[0] * a_strides[0]
    a_indx += inds1[0] * a_strides[1]
    a_indx += inds2[0] * a_strides[2]

    cdef double* a_ptr = <double*>a.data
    cdef double* rhs_ptr = <double*>rhs.data

    if rhs.flags['C_CONTIGUOUS']:
        while(True):
            a_ptr[a_indx] = rhs_ptr[rhs_indx]
            rhs_indx += 1 # always increments by 1

            #increment b ~ itertools.product
            if b[2]+1 < rhs_dims[2]: # "i = 2" loop
                a_indx += (inds2[b[2]+1]-inds2[b[2]]) * a_strides[2]
                b[2] += 1
            else:
                a_indx += (inds2[0]-inds2[b[2]]) * a_strides[2]
                b[2] = 0
                if b[1]+1 < rhs_dims[1]: # "i = 1" loop
                    a_indx += (inds1[b[1]+1]-inds1[b[1]]) * a_strides[1]
                    b[1] += 1
                else:
                    a_indx += (inds1[0]-inds1[b[1]]) * a_strides[1]
                    b[1] = 0
                    if b[0]+1 < rhs_dims[0]: # "i = 0" loop
                        a_indx += (inds0[b[0]+1]-inds0[b[0]]) * a_strides[0]
                        b[0] += 1
                    else:
                        break # can't increment anything - break while(True) loop

    else: # rhs is *not* c-contiguous, so need to use its strides
        while(True):
            a_ptr[a_indx] = rhs_ptr[rhs_indx]

            #increment b ~ itertools.product
            if b[2]+1 < rhs_dims[2]: # "i = 2" loop
                a_indx += (inds2[b[2]+1]-inds2[b[2]]) * a_strides[2]
                rhs_indx += rhs_strides[2]
                b[2] += 1
            else:
                a_indx += (inds2[0]-inds2[b[2]]) * a_strides[2]
                rhs_indx -= b[2]*rhs_strides[2]
                b[2] = 0
                if b[1]+1 < rhs_dims[1]: # "i = 1" loop
                    a_indx += (inds1[b[1]+1]-inds1[b[1]]) * a_strides[1]
                    rhs_indx += rhs_strides[1]
                    b[1] += 1
                else:
                    a_indx += (inds1[0]-inds1[b[1]]) * a_strides[1]
                    rhs_indx -= b[1]*rhs_strides[1]
                    b[1] = 0
                    if b[0]+1 < rhs_dims[0]: # "i = 0" loop
                        a_indx += (inds0[b[0]+1]-inds0[b[0]]) * a_strides[0]
                        rhs_indx += rhs_strides[0]
                        b[0] += 1
                    else:
                        break # can't increment anything - break while(True) loop

                    
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def fast_csr_sum_flat(np.ndarray[np.complex128_t, ndim=1, mode="c"] data,
                      np.ndarray[np.complex128_t, ndim=1, mode="c"] coeffs,
                      np.ndarray[np.int64_t, ndim=1, mode="c"] flat_dest_index_array,
                      np.ndarray[np.complex128_t, ndim=1, mode="c"] flat_csr_mx_data,
                      np.ndarray[np.int64_t, ndim=1, mode="c"] mx_nnz_indptr):
    cdef int Nmxs = mx_nnz_indptr.size - 1  # the number of CSR matrices
    cdef int iMx
    cdef int i
    cdef double complex coeff
    
    for iMx in range(Nmxs):
        coeff = coeffs[iMx]
        for i in range(mx_nnz_indptr[iMx], mx_nnz_indptr[iMx+1]):
            data[flat_dest_index_array[i]] = data[flat_dest_index_array[i]] + coeff * flat_csr_mx_data[i]


@cython.cdivision(True) # turn off divide-by-zero checking
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def faster_update_rows(np.ndarray[double, ndim=2, mode='c'] a,
                     np.ndarray[double, ndim=1, mode='c'] b,
                     int icol, int ipivot_local,
                     np.ndarray[double, ndim=1, mode='c'] pivot_row,
                     double pivot_b):
    cdef int i
    cdef int m = a.shape[0]
    cdef int n = a.shape[1]
    cdef double* a_ptr = <double*>a.data
    cdef double* b_ptr = <double*>b.data
    cdef double* pivot_row_ptr = <double*>pivot_row.data
    cdef double* row
    cdef double pivot_val = pivot_row_ptr[icol]
    cdef double alpha
    
    if 0 <= ipivot_local and ipivot_local < m:  #split loop into 2 parts
        for i in range(ipivot_local):
            row = &a_ptr[i * n]
            alpha = row[icol] / pivot_val
            for j in range(icol):
                row[j] = row[j] - alpha * pivot_row_ptr[j]
            for j in range(icol+1, n):
                row[j] = row[j] - alpha * pivot_row_ptr[j]
            row[icol] = 0.0  # this sometimes isn't exactly true because of finite precision error
            b_ptr[i] = b_ptr[i] - alpha * pivot_b

        for i in range(ipivot_local + 1, m):
            row = &a_ptr[i * n]
            alpha = row[icol] / pivot_val
            for j in range(icol):
                row[j] = row[j] - alpha * pivot_row_ptr[j]
            for j in range(icol+1, n):
                row[j] = row[j] - alpha * pivot_row_ptr[j]
            row[icol] = 0.0  # this sometimes isn't exactly true because of finite precision error
            b_ptr[i] = b_ptr[i] - alpha * pivot_b

    else:  # we don't own pivot - just use a single loop
        for i in range(m):
            row = &a_ptr[i * n]
            alpha = row[icol] / pivot_val
            for j in range(icol):
                row[j] = row[j] - alpha * pivot_row_ptr[j]
            for j in range(icol+1, n):
                row[j] = row[j] - alpha * pivot_row_ptr[j]
            row[icol] = 0.0  # this sometimes isn't exactly true because of finite precision error
            b_ptr[i] = b_ptr[i] - alpha * pivot_b

    #a[:, icol] = 0.0  # this sometimes isn't exactly true because of finite precision error

    #alpha = a[:, icol] / pivot_row[icol]  # (k,)
    #a[:, :] -= _np.kron(alpha, pivot_row)
    #a[:, icol] = 0  # this sometimes isn't exactly true because of finite precision error
    #b[:] -= alpha * pivot_b

@cython.cdivision(True) # turn off divide-by-zero checking
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def fast_update_rows(np.ndarray[double, ndim=2] a,
                     np.ndarray[double, ndim=1] b,
                     int icol, int ipivot_local,
                     np.ndarray[double, ndim=1] pivot_row,
                     double pivot_b):
    cdef int i
    cdef int m = a.shape[0]
    cdef double alpha
    cdef double pivot_val = pivot_row[icol]

    if 0 <= ipivot_local and ipivot_local < m:  #split loop into 2 parts
        for i in range(ipivot_local):
            alpha = a[i, icol] / pivot_val
            a[i, :] -=  alpha * pivot_row
            b[i] = b[i] - alpha * pivot_b

        for i in range(ipivot_local + 1, m):
            alpha = a[i, icol] / pivot_val
            a[i, :] -=  alpha * pivot_row
            b[i] = b[i] - alpha * pivot_b

        a[0:ipivot_local, icol] = 0.0  # this sometimes isn't exactly true because of finite precision error
        a[ipivot_local + 1:, icol] = 0.0

    else:  # we don't own pivot - just use a single loop
        for i in range(m):
            alpha = a[i, icol] / pivot_val
            a[i, :] -=  alpha * pivot_row
            b[i] = b[i] - alpha * pivot_b

        a[:, icol] = 0.0  # this sometimes isn't exactly true because of finite precision error


@cython.cdivision(True) # turn off divide-by-zero checking
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def restricted_abs_argmax(np.ndarray[double, ndim=1] ar, np.ndarray[np.int64_t, ndim=1] restrict_to):

    cdef int i
    cdef int indx
    cdef int n = restrict_to.shape[0]
    cdef int max_indx
    cdef double val
    cdef double max_val
    cdef double* ar_ptr = <double*>ar.data
    cdef INT* to_ptr = <INT*>restrict_to.data
    cdef INT ar_stride = ar.strides[0] // ar.itemsize
    cdef INT to_stride = restrict_to.strides[0] // restrict_to.itemsize

    #indx = restrict_to[0]
    #max_val = abs(ar[indx])
    #max_indx = indx
    #
    #for i in range(1, n):
    #    indx = restrict_to[i]
    #    val = abs(ar[indx])
    #    if val > max_val:
    #        max_val = val
    #        max_indx = indx

    indx = to_ptr[0]
    max_val = abs(ar_ptr[indx * ar_stride])
    max_indx = indx
    
    for i in range(1, n):
        indx = to_ptr[i * to_stride]
        val = abs(ar_ptr[indx * ar_stride])
        if val > max_val:
            max_val = val
            max_indx = indx

    return max_val, max_indx

#@cython.cdivision(True) # turn off divide-by-zero checking (keeping off for Python behavior of div/mods)
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def fast_compose_cliffords(np.ndarray[np.int64_t, ndim=2] s1, np.ndarray[np.int64_t, ndim=1] p1,
                           np.ndarray[np.int64_t, ndim=2] s2, np.ndarray[np.int64_t, ndim=1] p2):
    cdef INT i
    cdef INT j
    cdef INT k
    cdef INT N = s1.shape[0] // 2 # Number of qubits

    # Temporary space of C^T U C terms
    cdef np.ndarray[np.int64_t, ndim=2, mode="c"] inner = np.zeros([2*N, 2*N], dtype=np.int64)

    # Outputs
    cdef np.ndarray[np.int64_t, ndim=2, mode="c"] s = np.zeros([2*N, 2*N], dtype=np.int64)
    cdef np.ndarray[np.int64_t, ndim=1, mode="c"] p = np.zeros([2*N], dtype=np.int64)

    # If C' = [[C^00, C^01], [C^10, C^11]] and U = [[0, 0], [I, 0]]
    # then C'^T U C' = [[C^10^T C00, C^10^T C^01], [C^11^T C^00, C^11^T C^01]]
    for i in range(N):
        for j in range(N):
            for k in range(N):
                inner[i, j]     += s2[k+N, i] * s2[k, j]     # C^10^T C00
                inner[i, j+N]   += s2[k+N, i] * s2[k, j+N]   # C^10^T C^01
                inner[i+N, j]   += s2[k+N, i+N] * s2[k, j]   # C^11^T C^00
                inner[i+N, j+N] += s2[k+N, i+N] * s2[k, j+N] # C^11^T C^01
    
    # 2P_upps(C'^T U C') + P_diag(C'^T U C') in-place (now equivalent to matrix in Python version)
    for i in range(2*N):
        for j in range(0, i):
            inner[i, j] = 0
        for j in range(i+1, 2*N):
            inner[i, j] = 2*inner[i,j]

    # Eqn 8 from Hostens and De Moor PRA 71, 042315 (2005)
    for i in range(2*N):
        p[i] += p1[i] # h
        for j in range(2*N):
            p[i] += s1[j, i] * p2[j] # C^T h'
            p[i] -= s1[j, i] * inner[j, j] # - C^T Vdiag(C'^T U C') (OK because diagonal only)
            for k in range(2*N):
                s[i, j] += s2[i, k] * s1[k, j] # C'' = C' C
                p[i] += s1[j, i] * inner[j, k] * s1[k, i] # Vdiag(C^T [2P_upps(C'^T U C') + P_diag(C'^T U C')] C)

    # Mod d/2d
    for i in range(2*N):
        p[i] = p[i] % 4
        for j in range(2*N):
            s[i, j] = s[i, j] % 2

    return s, p


#Faster generation of upper triangular indices specialized to first
#superdiagonal and up.
@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
@cython.cdivision(True)
@lru_cache(maxsize=16)
def fast_triu_indices(int n):
    if n < 1:
        raise ValueError('n must be greater than 0')
        
    cdef int size = (n**2-n)/2
    cdef int curr_idx = 0
    cdef int j, i
    
    cdef np.ndarray[np.int64_t, ndim=1, mode="c"] row_indices_np = np.empty(size, dtype=np.int64)
    cdef np.ndarray[np.int64_t, ndim=1, mode="c"] col_indices_np = np.empty(size, dtype=np.int64)
    
    cdef np.int64_t[::1] row_indices = row_indices_np
    cdef np.int64_t[::1] col_indices = col_indices_np

    for j in range(n-1):
        for i in range(n-j-1, 0, -1):
            row_indices[curr_idx] = j
            curr_idx += 1

    curr_idx = 0
    for j in range(1, n):
        for i in range(j, n):
            col_indices[curr_idx] = i
            curr_idx += 1

    return row_indices_np, col_indices_np
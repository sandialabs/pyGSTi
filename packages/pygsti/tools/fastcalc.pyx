# encoding: utf-8
# cython: profile=False
# cython: linetrace=False
# filename: fastcalc.pyx

import numpy as np
from libc.stdlib cimport malloc, free
cimport numpy as np
cimport cython

def dot(np.ndarray[double, ndim=1] f, np.ndarray[double, ndim=1] g):
    cdef long N = f.shape[0]
    cdef float ret = 0.0
    cdef int i
    for i in range(N):
        ret += f[i]*g[i]
    return ret

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def embedded_fast_acton_sparse(embedded_gate_acton_fn,
                               np.ndarray[double, ndim=1] output_state,
                               np.ndarray[double, ndim=1] state,
                               np.ndarray[int, ndim=1] noop_incrementers,
                               np.ndarray[int, ndim=1] numBasisEls_noop_blankaction,
                               np.ndarray[int, ndim=1] baseinds):

    cdef long i
    cdef int k
    cdef int vec_index_noop = 0
    cdef int nParts = numBasisEls_noop_blankaction.shape[0]
    cdef int nActionIndices = baseinds.shape[0]
    #cdef np.ndarray b = np.zeros(nParts, dtype=int)
    #cdef np.ndarray gate_b = np.zeros(nAction, dtype=int)
    #cdef np.ndarray[long, ndim=1] baseinds = np.empty(nActionIndices, dtype=int) #for FASTER
    cdef int b[100]

    #These need to be numpy arrays for python interaction
    cdef np.ndarray[double, ndim=1, mode="c"] slc1 = np.empty(nActionIndices, dtype='d')
    cdef np.ndarray[double, ndim=1, mode="c"] slc2 = np.empty(nActionIndices, dtype='d')

    # nActionIndices = np.product(numBasisEls_action)
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
        np.ndarray[int, ndim=1, mode="c"] Aindptr not None,
        np.ndarray[int, ndim=1, mode="c"] Aindices not None,
        double mu, int m_star, int s, double tol, double eta,
                               np.ndarray[double, ndim=1] output_state,
                               np.ndarray[double, ndim=1] state,
                               np.ndarray[int, ndim=1] noop_incrementers,
                               np.ndarray[int, ndim=1] numBasisEls_noop_blankaction,
                               np.ndarray[int, ndim=1] baseinds):

    #                          int offset,
    #                          np.ndarray[int, ndim=1] numBasisEls_action,
    #                          np.ndarray[int, ndim=1] actionInds,

    cdef long i
    cdef int k
    cdef int vec_index_noop = 0
    cdef long nParts = numBasisEls_noop_blankaction.shape[0]
    #cdef int nAction = numBasisEls_action.shape[0]
    cdef long nActionIndices = baseinds.shape[0]
    cdef int b[100]
    #cdef int gate_b[100]
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
            custom_expm_multiply_simple_core_c(&Adata[0], &Aindptr[0],
                                               &Aindices[0], &slc1[0], nActionIndices,
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
                                    np.ndarray[int, ndim=1] noop_incrementers,
                                    np.ndarray[int, ndim=1] numBasisEls_noop_blankaction,
                                    np.ndarray[int, ndim=1] baseinds):

    cdef long i
    cdef long j
    cdef double cum
    cdef int k
    cdef int vec_index_noop = 0
    cdef long nParts = numBasisEls_noop_blankaction.shape[0]
    cdef long nActionIndices = baseinds.shape[0]
    cdef int b[100]

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
def medium_kron(np.ndarray[double, ndim=1, mode="c"] outvec not None,
              np.ndarray[double, ndim=2, mode="c"] fastArray not None,
              np.ndarray[int, ndim=1, mode="c"] fastArraySizes not None):
    
    cdef int mi[100] # multi-index holding
    cdef int multipliers[100]
    cdef double preprods[101] # +1 from other static dims
    cdef int nFactors = fastArray.shape[0]
    cdef int i
    cdef int k
    cdef int p
    
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
              np.ndarray[int, ndim=1, mode="c"] fastArraySizes not None):
    
    cdef int nFactors = fastArray.shape[0]
    cdef int N = outvec.shape[0]
    cdef int i
    cdef int j
    cdef int k
    cdef int sz
    cdef int off
    cdef int endoff
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
        
    #assert(sz == N)


#Manually inline to avoid overhead of argument passing
#@cython.boundscheck(False) # turn off bounds-checking for entire function
#@cython.wraparound(False)  # turn off negative index wrapping for entire function
#cdef vec_inf_norm(np.ndarray[double, ndim=1] v):
#    cdef int i
#    cdef int N = v.shape[0]
#    cdef double mx = 0.0
#    cdef double a
#    for i in range(N):
#        a = abs(v[i])
#        if a > mx: mx = a
#    return mx
    

            
@cython.cdivision(True) # turn off divide-by-zero checking
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def custom_expm_multiply_simple_core(np.ndarray[double, ndim=1, mode="c"] Adata,
                                       np.ndarray[int, ndim=1, mode="c"] Aindptr,
                                       np.ndarray[int, ndim=1, mode="c"] Aindices,
                                       np.ndarray[double, ndim=1, mode="c"] B,
                                       double mu, int m_star, int s, double tol, double eta):

    cdef int N = B.shape[0] #Aindptr.shape[0]-1
    if s == 0: return B #short circuit

    cdef np.ndarray[double, ndim=1, mode="c"] F = np.empty(N,'d')
    cdef np.ndarray[double, ndim=1, mode="c"] scratch = np.empty(N,'d')

    custom_expm_multiply_simple_core_c(&Adata[0], &Aindptr[0],
                                       &Aindices[0], &B[0], N,
                                       mu, m_star, s, tol, eta,
                                       &F[0], &scratch[0])
    return F




@cython.cdivision(True) # turn off divide-by-zero checking
cdef custom_expm_multiply_simple_core_c(double* Adata, int* Aindptr,
                                        int* Aindices, double* B,
                                        int N, double mu, int m_star,
                                        int s, double tol, double eta,
                                        double* F, double* scratch):

    cdef int i
    cdef int j
    cdef int r
    cdef int k

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
                B[k] = scratch[k] #finishes B = coeff * A.dot(B) 
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
                          np.ndarray[int, ndim=1] Aindptr,
                          np.ndarray[int, ndim=1] Aindices,
                          np.ndarray[double, ndim=1] Bdata,
                          np.ndarray[int, ndim=1] Bindptr,
                          np.ndarray[int, ndim=1] Bindices,
                          double lmb, int n):

    cdef int nxt = 0
    cdef int iRow = 0
    cdef int i = 0
    cdef int bFound = 0
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

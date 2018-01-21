import numpy as np
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
                               long offset,
                               np.ndarray[int, ndim=1] multipliers,
                               np.ndarray[int, ndim=1] numBasisEls_noop_blankaction,
                               np.ndarray[int, ndim=1] numBasisEls_action,
                               np.ndarray[int, ndim=1] actionInds,
                               np.ndarray[int, ndim=1] inds):

    cdef int i
    cdef int k
    cdef int vec_index_noop = 0
    cdef int vec_index = 0
    cdef int nParts = numBasisEls_noop_blankaction.shape[0]
    cdef int nAction = numBasisEls_action.shape[0]
    cdef int nActionIndices = inds.shape[0]
    #cdef np.ndarray b = np.zeros(nParts, dtype=int)
    #cdef np.ndarray gate_b = np.zeros(nAction, dtype=int)
    #cdef np.ndarray[long, ndim=1] baseinds = np.empty(nActionIndices, dtype=int) #for FASTER
    cdef int b[100]
    cdef int gate_b[100]

    cdef np.ndarray[double, ndim=1] slc1 = np.empty(nActionIndices, dtype='d')
    cdef np.ndarray[double, ndim=1] slc2 = np.empty(nActionIndices, dtype='d')

    # nActionIndices = np.product(numBasisEls_action)
    #for i in range(nAction):
    #    nActionIndices *= numBasisEls_action[i]

    if nParts > 100: assert(0) # need to increase size of static arrays above
    for i in range(nParts): b[i] = 0

    #FASTER, but not much
    ##Loop to fill baseinds array
    #for i in range(nAction): gate_b[i] = 0 # ~ gate_b.fill(0)
    #vec_index = 0
    #for k in range(nActionIndices):
    #    baseinds[k] = offset+vec_index
    #
    #    #increment gate_b
    #    for i in range(nAction-1,-1,-1):
    #        if gate_b[i]+1 < numBasisEls_action[i]:
    #            gate_b[i] += 1; vec_index += multipliers[actionInds[i]]
    #            break
    #        else:
    #            vec_index -= gate_b[i]*multipliers[actionInds[i]]
    #            gate_b[i] = 0
    #    #Unnecessary b/c for loop will terminate on its own:
    #    #else:
    #    #    break # can't increment anything - break while(True) loop
    #    
    ##Now inds is filled with "action digits", and just need to
    ## add "noop digits" in sequence below

    
    #vec_index_noop = 0 assigned above
    while(True):

        #Loop to fill inds array
        for i in range(nAction): gate_b[i] = 0 # ~ gate_b.fill(0)
        vec_index = vec_index_noop
        for k in range(nActionIndices):
            inds[k] = offset+vec_index
        
            #increment gate_b
            for i in range(nAction-1,-1,-1):
                if gate_b[i]+1 < numBasisEls_action[i]:
                    gate_b[i] += 1; vec_index += multipliers[actionInds[i]]
                    break
                else:
                    vec_index -= gate_b[i]*multipliers[actionInds[i]]
                    gate_b[i] = 0
            #Unnecessary b/c for loop will terminate on its own:
            #else:
            #    break # can't increment anything - break while(True) loop

        #FASTER, but not much
        #for k in range(nActionIndices):
        #    inds[k] = baseinds[k]+vec_index_noop

        #Act with embedded gate on appropriate sub-space of state
        #output_state[ inds ] += embedded_gate_acton_fn( state[inds] ) #Fancy indexing...
        #output_state[inds] += state[inds]
        for k in range(nActionIndices):
            slc1[k] = state[ inds[k] ]
        slc2 = embedded_gate_acton_fn( slc1 )
        for k in range(nActionIndices):
            output_state[ inds[k] ] += slc2[k] #state[ inds[k] ]
        
        #increment b ~ itertools.product & update vec_index_noop = _np.dot(self.multipliers, b)
        for i in range(nParts-1,-1,-1):
            if b[i]+1 < numBasisEls_noop_blankaction[i]:
                b[i] += 1; vec_index_noop += multipliers[i]
                break
            else:
                vec_index_noop -= b[i]*multipliers[i]; b[i] = 0
        else:
            break # can't increment anything - break while(True) loop

    return output_state



#SPECIAL CASE 1: embedded gate is Lindblad gate with no unitary postfactor -
# so just pass the args to custom_expm_multiply_simple_core
#@cython.boundscheck(False) # turn off bounds-checking for entire function
#@cython.wraparound(False)  # turn off negative index wrapping for entire function
def embedded_fast_acton_sparse_spc1(
        np.ndarray[double, ndim=1, mode="c"] Adata not None,
        np.ndarray[int, ndim=1, mode="c"] Aindptr not None,
        np.ndarray[int, ndim=1, mode="c"] Aindices not None,
        double mu, int m_star, int s, double tol, double eta,
                               np.ndarray[double, ndim=1] output_state,
                               np.ndarray[double, ndim=1] state,
                               long offset,
                               np.ndarray[int, ndim=1] multipliers,
                               np.ndarray[int, ndim=1] numBasisEls_noop_blankaction,
                               np.ndarray[int, ndim=1] numBasisEls_action,
                               np.ndarray[int, ndim=1] actionInds,
                               np.ndarray[int, ndim=1] inds):

    cdef int i
    cdef int k
    cdef int vec_index_noop = 0
    cdef int vec_index = 0
    cdef int nParts = numBasisEls_noop_blankaction.shape[0]
    cdef int nAction = numBasisEls_action.shape[0]
    cdef int nActionIndices = inds.shape[0]
    cdef int b[100]
    cdef int gate_b[100]
    cdef int Annz = Adata.shape[0]

    cdef np.ndarray[double, ndim=1, mode="c"] slc1 = np.empty(nActionIndices, dtype='d')
    cdef np.ndarray[double, ndim=1, mode="c"] slc2 = np.empty(nActionIndices, dtype='d')
    cdef np.ndarray[double, ndim=1, mode="c"] scratch = np.empty(nActionIndices, dtype='d')

    if nParts > 100: assert(0) # need to increase size of static arrays above
    for i in range(nParts): b[i] = 0

    #vec_index_noop = 0 assigned above
    while(True):

        #Loop to fill inds array
        for i in range(nAction): gate_b[i] = 0 # ~ gate_b.fill(0)
        vec_index = vec_index_noop
        for k in range(nActionIndices):
            inds[k] = offset+vec_index
        
            #increment gate_b
            for i in range(nAction-1,-1,-1):
                if gate_b[i]+1 < numBasisEls_action[i]:
                    gate_b[i] += 1; vec_index += multipliers[actionInds[i]]
                    break
                else:
                    vec_index -= gate_b[i]*multipliers[actionInds[i]]
                    gate_b[i] = 0

        #Act with embedded gate on appropriate sub-space of state
        for k in range(nActionIndices):
            slc1[k] = state[ inds[k] ]

        #SPECIAL ACTON for output_state[ inds ] += acton( state[inds] )
        # replaces:  slc2 = embedded_gate_acton_fn( slc1 )
        if Annz > 0:
            custom_expm_multiply_simple_core_c(&Adata[0], &Aindptr[0],
                                               &Aindices[0], &slc1[0], nActionIndices,
                                               mu, m_star, s, tol, eta,
                                               &slc2[0], &scratch[0])
        else: #act as identity
            for k in range(nActionIndices):
                slc2[k] = slc1[k]
            

        for k in range(nActionIndices):
            output_state[ inds[k] ] += slc2[k] #state[ inds[k] ]
        
        #increment b ~ itertools.product & update vec_index_noop = _np.dot(self.multipliers, b)
        for i in range(nParts-1,-1,-1):
            if b[i]+1 < numBasisEls_noop_blankaction[i]:
                b[i] += 1; vec_index_noop += multipliers[i]
                break
            else:
                vec_index_noop -= b[i]*multipliers[i]; b[i] = 0
        else:
            break # can't increment anything - break while(True) loop

    return output_state


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
            for k in range(N): scratch[k] = 0
            for r in range(N):
                for k in range(Aindptr[r],Aindptr[r+1]):
                    scratch[r] += Adata[k] * B[ Aindices[k]]

                    
            for k in range(N):
                B[k] = scratch[k] #finishes B = coeff * A.dot(B) 
                F[k] += B[k] #F += B

            ##if False: #j % 3 == 0: #every == 3 #TODO: work on this
            #c2 = vec_inf_norm(B) #_exact_inf_norm(B)
            c2 = 0.0
            for k in range(N):
                a = abs(B[k])
                if a > c2: c2 = a

            #normF = vec_inf_norm(F) #_exact_inf_norm(F)
            normF = 0.0
            for k in range(N):
                a = abs(F[k])
                if a > normF: normF = a
                
            if c1 + c2 <= tol * normF: 
                break
            c1 = c2

        #F *= eta
        #B = F
        for k in range(N):
            F[k] *= eta
            B[k] = F[k]

    #return F # updates passed-in memory, so don't need this

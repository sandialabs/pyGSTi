# encoding: utf-8
# cython: profile=False
# cython: linetrace=False
# filename: fastactonlib.pyx

import sys
import numpy as np
from libc.stdlib cimport malloc, free
from libcpp cimport bool
from libcpp.vector cimport vector
#from libcpp.unordered_map cimport unordered_map
from cython.operator cimport dereference as deref, preincrement as inc
cimport numpy as np
cimport cython

from ..tools import mpitools as _mpit
from ..tools import slicetools as _slct


cdef extern from "fastreps.h" namespace "CReps":    
    cdef cppclass DMStateCRep:
        DMStateCRep() except +
        DMStateCRep(int) except +
        DMStateCRep(double*,int,bool) except +
        int _dim
        double* _dataptr

    cdef cppclass DMEffectCRep:
        DMEffectCRep() except +
        DMEffectCRep(int) except +
        double probability(DMStateCRep* state)
        int _dim

    cdef cppclass DMEffectCRep_Dense(DMEffectCRep):
        DMEffectCRep_Dense() except +
        DMEffectCRep_Dense(double*,int) except +
        double probability(DMStateCRep* state)
        int _dim
        double* _dataptr

    cdef cppclass DMEffectCRep_TensorProd(DMEffectCRep):
        DMEffectCRep_TensorProd() except +
        DMEffectCRep_TensorProd(double*, int*, int, int, int) except +
        double probability(DMStateCRep* state)
        int _dim

    cdef cppclass DMGateCRep:
        DMGateCRep(int) except +
        DMStateCRep* acton(DMStateCRep*, DMStateCRep*);
        int _dim

    cdef cppclass DMGateCRep_Dense(DMGateCRep):
        DMGateCRep_Dense(double*,int) except +
        DMStateCRep* acton(DMStateCRep*, DMStateCRep*)
        double* _dataptr
        int _dim

    cdef cppclass DMGateCRep_Embedded(DMGateCRep):
        DMGateCRep_Embedded(DMGateCRep*, int*, int*, int*, int*, int, int, int, int, int) except +
        DMStateCRep* acton(DMStateCRep*, DMStateCRep*)

    cdef cppclass DMGateCRep_Composed(DMGateCRep):
        DMGateCRep_Composed(vector[DMGateCRep*], int) except +
        DMStateCRep* acton(DMStateCRep*, DMStateCRep*)

    cdef cppclass DMGateCRep_Lindblad(DMGateCRep):
        DMGateCRep_Lindblad(double* A_data, int* A_indices, int* A_indptr, int nnz,
			    double mu, double eta, int m_star, int s, int dim,
			    double* unitarypost_data, int* unitarypost_indices,
                            int* unitarypost_indptr, int unitarypost_nnz) except +
        DMStateCRep* acton(DMStateCRep*, DMStateCRep*)


        
    # Stabilizer classes
    cdef cppclass SBStateCRep:
        SBStateCRep(int*, int*, double complex*, int, int) except +
        SBStateCRep(int, int) except +
        SBStateCRep(double*,int,bool) except +
        int _n
        int _namps

    cdef cppclass SBEffectCRep:
        SBEffectCRep(int*, int) except +
        double probability(SBStateCRep* state)
        double complex amplitude(SBStateCRep* state)
        int _n

    cdef cppclass SBGateCRep:
        SBGateCRep(int) except +
        SBStateCRep* acton(SBStateCRep*, SBStateCRep*);
        int _n

    cdef cppclass SBGateCRep_Embedded(SBGateCRep):
        SBGateCRep_Embedded(SBGateCRep*, int, int*, int) except +
        SBStateCRep* acton(SBStateCRep*, SBStateCRep*)

    cdef cppclass SBGateCRep_Composed(SBGateCRep):
        SBGateCRep_Composed(vector[SBGateCRep*], int) except +
        SBStateCRep* acton(SBStateCRep*, SBStateCRep*)

    cdef cppclass SBGateCRep_Clifford(SBGateCRep):
        SBGateCRep_Clifford(int*, int*, double complex*, int*, int*, double complex*, int) except +
        SBStateCRep* acton(SBStateCRep*, SBStateCRep*)
    

ctypedef DMGateCRep* DMGateCRep_ptr
ctypedef DMStateCRep* DMStateCRep_ptr
ctypedef DMEffectCRep* DMEffectCRep_ptr
ctypedef SBGateCRep* SBGateCRep_ptr
        
#cdef class StateRep:
#    pass

cdef class DMStateRep: #(StateRep):
    cdef DMStateCRep* c_state
    cdef np.ndarray data_ref
    #cdef double [:] data_view # alt way to hold a reference

    def __cinit__(self, np.ndarray[double, ndim=1, mode='c'] data):
        #print("PYX state constructed w/dim ",data.shape[0])
        #cdef np.ndarray[double, ndim=1, mode='c'] np_cbuf = np.ascontiguousarray(data, dtype='d') # would allow non-contig arrays
        #cdef double [:] view = data;  self.data_view = view # ALT: holds reference...
        self.data_ref = data # holds reference to data so it doesn't get garbage collected - or could copy=true
        #self.c_state = new DMStateCRep(<double*>np_cbuf.data,<int>np_cbuf.shape[0],<bool>0)
        self.c_state = new DMStateCRep(<double*>data.data,<int>data.shape[0],<bool>0)

    def __dealloc__(self):
        del self.c_state

    def __str__(self):
        return str([self.c_state._dataptr[i] for i in range(self.c_state._dim)])


cdef class DMEffectRep:
    cdef DMEffectCRep* c_effect

    def __cinit__(self):
        pass # no init; could set self.c_effect = NULL? could assert(False)?
    def __dealloc__(self):
        del self.c_effect # check for NULL?

    def probability(self, DMStateRep state not None):
        #unnecessary (just put in signature): cdef DMStateRep st = <DMStateRep?>state
        return self.c_effect.probability(state.c_state)

cdef class DMEffectRep_Dense(DMEffectRep):
    cdef np.ndarray data_ref

    def __cinit__(self, np.ndarray[double, ndim=1, mode='c'] data):
        self.data_ref = data # holds reference to data
        self.c_effect = new DMEffectCRep_Dense(<double*>data.data,
                                               <int>data.shape[0])

cdef class DMEffectRep_TensorProd(DMEffectRep):
    cdef np.ndarray data_ref1
    cdef np.ndarray data_ref2

    def __cinit__(self, np.ndarray[double, ndim=2, mode='c'] kron_array,
                  np.ndarray[int, ndim=1, mode='c'] factor_dims, int nfactors, int max_factor_dim, int dim):
        # cdef int dim = np.product(factor_dims) -- just send as argument
        self.data_ref1 = kron_array
        self.data_ref2 = factor_dims
        self.c_effect = new DMEffectCRep_TensorProd(<double*>kron_array.data,
                                                    <int*>factor_dims.data,
                                                    nfactors, max_factor_dim, dim)


cdef class DMGateRep:
    cdef DMGateCRep* c_gate

    def __cinit__(self):
        pass # self.c_gate = NULL ?
    
    def __dealloc__(self):
        del self.c_gate

    def acton(self, DMStateRep state not None):
        cdef DMStateRep out_state = DMStateRep(np.empty(self.c_gate._dim, dtype='d'))
        #print("PYX acton called w/dim ", self.c_gate._dim, out_state.c_state._dim)
        # assert(state.c_state._dataptr != out_state.c_state._dataptr) # DEBUG
        self.c_gate.acton(state.c_state, out_state.c_state)
        return out_state

    #FUTURE: adjoint acton
        
cdef class DMGateRep_Dense(DMGateRep):
    cdef np.ndarray data_ref

    def __cinit__(self, np.ndarray[double, ndim=2, mode='c'] data):
        self.data_ref = data
        #print("PYX dense gate constructed w/dim ",data.shape[0])
        self.c_gate = new DMGateCRep_Dense(<double*>data.data,
                                           <int>data.shape[0])
    def __str__(self):
        s = ""
        cdef DMGateCRep_Dense* my_cgate = <DMGateCRep_Dense*>self.c_gate # b/c we know it's a _Dense gate...
        cdef int i,j,k 
        for i in range(my_cgate._dim):
            k = i*my_cgate._dim
            for j in range(my_cgate._dim):
                s += str(my_cgate._dataptr[k+j]) + " "
            s += "\n"
        return s


cdef class DMGateRep_Embedded(DMGateRep):
    cdef np.ndarray data_ref1
    cdef np.ndarray data_ref2
    cdef np.ndarray data_ref3
    cdef np.ndarray data_ref4
    cdef DMGateRep embedded

    def __cinit__(self, DMGateRep embedded_gate,
                  np.ndarray[int, ndim=1, mode='c'] noop_incrementers,
		  np.ndarray[int, ndim=1, mode='c'] numBasisEls_noop_blankaction,
                  np.ndarray[int, ndim=1, mode='c'] baseinds,
                  np.ndarray[int, ndim=1, mode='c'] blocksizes,
		  int nActive, int nComponents, int iActiveBlock, int nBlocks, int dim):
        self.data_ref1 = noop_incrementers
        self.data_ref2 = numBasisEls_noop_blankaction
        self.data_ref3 = baseinds
        self.data_ref4 = blocksizes
        self.embedded = embedded_gate # needed to prevent garbage collection?
        self.c_gate = new DMGateCRep_Embedded(embedded_gate.c_gate, # may need to have a base class w/ DMGateCRep* cgate member(?) - and maybe put acton there?
                                              <int*>noop_incrementers.data, <int*>numBasisEls_noop_blankaction.data,
                                              <int*>baseinds.data, <int*>blocksizes.data,
                                              nActive, nComponents, iActiveBlock, nBlocks, dim)


cdef class DMGateRep_Composed(DMGateRep):
    cdef object list_of_factors # list of DMGateRep objs?

    def __cinit__(self, factor_gates, int dim):
        self.list_of_factors = factor_gates
        cdef int i
        cdef int nfactors = len(factor_gates)
        cdef vector[DMGateCRep*] gate_creps = vector[DMGateCRep_ptr](nfactors)
        for i in range(nfactors):
            gate_creps[i] = (<DMGateRep?>factor_gates[i]).c_gate
        self.c_gate = new DMGateCRep_Composed(gate_creps, dim)


cdef class DMGateRep_Lindblad(DMGateRep):
    cdef np.ndarray data_ref1
    cdef np.ndarray data_ref2
    cdef np.ndarray data_ref3
    cdef np.ndarray data_ref4
    cdef np.ndarray data_ref5
    cdef np.ndarray data_ref6

    def __cinit__(self, np.ndarray[double, ndim=1, mode='c'] A_data,
                  np.ndarray[int, ndim=1, mode='c'] A_indices,
                  np.ndarray[int, ndim=1, mode='c'] A_indptr,
                  double mu, double eta, int m_star, int s,
                  np.ndarray[double, ndim=1, mode='c'] unitarypost_data,
                  np.ndarray[int, ndim=1, mode='c'] unitarypost_indices,
                  np.ndarray[int, ndim=1, mode='c'] unitarypost_indptr):
        self.data_ref1 = A_data
        self.data_ref2 = A_indices
        self.data_ref3 = A_indptr
        self.data_ref4 = unitarypost_data
        self.data_ref5 = unitarypost_indices
        self.data_ref6 = unitarypost_indptr
        cdef int nnz = A_data.shape[0]
        cdef int dim = A_indptr.shape[0]-1
        cdef int upost_nnz = unitarypost_data.shape[0]
        self.c_gate = new DMGateCRep_Lindblad(<double*>A_data.data, <int*>A_indices.data,
                                              <int*>A_indptr.data, nnz, mu, eta, m_star, s, dim,
                                              <double*>unitarypost_data.data,
                                              <int*>unitarypost_indices.data,
                                              <int*>unitarypost_indptr.data, upost_nnz)

        
# Stabilizer evolution-type class wrappers
cdef class SBStateRep: #(StateRep):
    cdef SBStateCRep* c_state
    cdef np.ndarray data_ref1
    cdef np.ndarray data_ref2
    cdef np.ndarray data_ref3

    def __cinit__(self, np.ndarray[int, ndim=2, mode='c'] smatrix,
                  np.ndarray[int, ndim=2, mode='c'] pvectors,
                  np.ndarray[np.complex128_t, ndim=1, mode='c'] amps):
        self.data_ref1 = smatrix
        self.data_ref2 = pvectors
        self.data_ref3 = amps
        cdef int namps = amps.shape[0]
        cdef int n = smatrix.shape[0] // 2
        self.c_state = new SBStateCRep(<int*>smatrix.data,<int*>pvectors.data,
                                       <double complex*>amps.data, namps, n)

    def __dealloc__(self):
        del self.c_state

    def __str__(self):
        return "STABSTATE **"


cdef class SBEffectRep:
    cdef SBEffectCRep* c_effect
    cdef np.ndarray data_ref

    def __cinit__(self, np.ndarray[int, ndim=1, mode='c'] zvals):
        self.data_ref = zvals
        self.c_effect = new SBEffectCRep(<int*>zvals.data,
                                         <int>zvals.shape[0])

    def __dealloc__(self):
        del self.c_effect # check for NULL?

    def probability(self, SBStateRep state not None):
        #unnecessary (just put in signature): cdef SBStateRep st = <SBStateRep?>state
        return self.c_effect.probability(state.c_state)


cdef class SBGateRep:
    cdef SBGateCRep* c_gate

    def __cinit__(self):
        pass # self.c_gate = NULL ?
    
    def __dealloc__(self):
        del self.c_gate

    def acton(self, SBStateRep state not None):
        cdef int n = self.c_gate._n
        cdef int namps = state.c_state._namps
        cdef SBStateRep out_state = SBStateRep(np.empty((2*n,2*n), dtype='i'),
                                               np.empty((namps,2*n), dtype='i'),
                                               np.empty(namps, dtype=np.complex128))
        self.c_gate.acton(state.c_state, out_state.c_state)
        return out_state

    #FUTURE: adjoint acton
        

cdef class SBGateRep_Embedded(SBGateRep):
    cdef np.ndarray data_ref
    cdef SBGateRep embedded

    def __cinit__(self, SBGateRep embedded_gate, int n, 
                  np.ndarray[int, ndim=1, mode='c'] qubits):
        self.data_ref = qubits
        self.embedded = embedded_gate # needed to prevent garbage collection?
        self.c_gate = new SBGateCRep_Embedded(embedded_gate.c_gate, n,
                                              <int*>qubits.data, <int>qubits.shape[0])


cdef class SBGateRep_Composed(SBGateRep):
    cdef object list_of_factors # list of SBGateRep objs?

    def __cinit__(self, factor_gates, int n):
        self.list_of_factors = factor_gates
        cdef int i
        cdef int nfactors = len(factor_gates)
        cdef vector[SBGateCRep*] gate_creps = vector[SBGateCRep_ptr](nfactors)
        for i in range(nfactors):
            gate_creps[i] = (<SBGateRep?>factor_gates[i]).c_gate
        self.c_gate = new SBGateCRep_Composed(gate_creps, n)



cdef class SBGateRep_Clifford(SBGateRep):
    cdef np.ndarray data_ref1
    cdef np.ndarray data_ref2
    cdef np.ndarray data_ref3
    cdef np.ndarray data_ref4
    cdef np.ndarray data_ref5
    cdef np.ndarray data_ref6

    def __cinit__(self, np.ndarray[int, ndim=2, mode='c'] smatrix,
                  np.ndarray[int, ndim=1, mode='c'] svector,
                  np.ndarray[int, ndim=2, mode='c'] smatrix_inv,
                  np.ndarray[int, ndim=1, mode='c'] svector_inv,
                  np.ndarray[np.complex128_t, ndim=2, mode='c'] unitary):
        
        self.data_ref1 = smatrix
        self.data_ref2 = svector
        self.data_ref3 = unitary
        self.data_ref4 = smatrix_inv
        self.data_ref5 = svector_inv
        self.data_ref6 = np.conjugate(np.transpose(unitary))
        cdef int n = smatrix.shape[0] // 2
        self.c_gate = new SBGateCRep_Clifford(<int*>smatrix.data, <int*>svector.data, <double complex*>unitary.data,
                                              <int*>smatrix_inv.data, <int*>svector_inv.data,
                                              <double complex*>self.data_ref6.data, n)


## END CLASSES -- BEGIN CALC METHODS


def propagate_staterep(staterep, gatereps):
    ret = staterep
    for gaterep in gatereps:
        ret = gaterep.acton(ret)
    return ret


cdef vector[vector[int]] convert_evaltree(evalTree, gate_lookup):
    # c_evalTree :
    # an array of int-arrays; each int-array is [i,iStart,iCache,<remainder gate indices>]
    cdef vector[int] intarray
    cdef vector[vector[int]] c_evalTree = vector[vector[int]](len(evalTree))
    for kk,ii in enumerate(evalTree.get_evaluation_order()):
        iStart,remainder,iCache = evalTree[ii]
        if iStart is None: iStart = -1 # so always an int
        if iCache is None: iCache = -1 # so always an int
        intarray = vector[int](3 + len(remainder))
        intarray[0] = ii
        intarray[1] = iStart
        intarray[2] = iCache
        for jj,gl in enumerate(remainder,start=3):
            intarray[jj] = gate_lookup[gl]
        c_evalTree[kk] = intarray
        
    return c_evalTree

cdef vector[DMStateCRep*] create_rhocache(int cacheSize, int state_dim):
    cdef int i
    cdef vector[DMStateCRep*] rho_cache = vector[DMStateCRep_ptr](cacheSize)
    for i in range(cacheSize): # fill cache with empty but alloc'd states
        rho_cache[i] = new DMStateCRep(state_dim)
    return rho_cache

cdef void free_rhocache(vector[DMStateCRep*] rho_cache):
    cdef unsigned int i
    for i in range(rho_cache.size()): # fill cache with empty but alloc'd states
        del rho_cache[i]


cdef vector[DMGateCRep*] convert_gatereps(gatereps):
    # c_gatereps : an array of DMGateCReps
    cdef vector[DMGateCRep*] c_gatereps = vector[DMGateCRep_ptr](len(gatereps))
    for ii,grep in gatereps.items(): # (ii = python variable)
        c_gatereps[ii] = (<DMGateRep?>grep).c_gate
    return c_gatereps

cdef DMStateCRep* convert_rhorep(rhorep):
    # extract c-reps from rhorep and ereps => c_rho and c_ereps
    return (<DMStateRep?>rhorep).c_state

cdef vector[DMEffectCRep*] convert_ereps(ereps):
    cdef vector[DMEffectCRep*] c_ereps = vector[DMEffectCRep_ptr](len(ereps))
    for i in range(len(ereps)):
        c_ereps[i] = (<DMEffectRep>ereps[i]).c_effect
    return c_ereps


def DM_compute_pr_cache(calc, rholabel, elabels, evalTree, comm, scratch=None): # TODO remove scratch


    rhoVec,EVecs = calc._rhoEs_from_labels(rholabel, elabels)
    pCache = np.empty((len(evalTree),len(EVecs)),'d')

    #Get (extension-type) representation objects
    rhorep = rhoVec.torep('prep')
    ereps = [ E.torep('effect') for E in EVecs]  # could cache these? then have torep keep a non-dense rep that can be quickly kron'd for a tensorprod spamvec
    gate_lookup = { lbl:i for i,lbl in enumerate(calc.gates.keys()) } # gate labels -> ints for faster lookup
    gatereps = { i:calc.gates[lbl].torep() for lbl,i in gate_lookup.items() }
    
    # convert to C-mode:  evaltree, gate_lookup, gatereps
    cdef c_evalTree = convert_evaltree(evalTree, gate_lookup)
    cdef DMStateCRep *c_rho = convert_rhorep(rhorep)
    cdef vector[DMGateCRep*] c_gatereps = convert_gatereps(gatereps)
    cdef vector[DMEffectCRep*] c_ereps = convert_ereps(ereps)

    # create rho_cache = vector of DMStateCReps
    cdef vector[DMStateCRep*] rho_cache = create_rhocache(evalTree.cache_size(), c_rho._dim)

    #OLD cdef double[:,:] ret_view = ret
    dm_compute_pr_cache(pCache, c_evalTree, c_gatereps, c_rho, c_ereps, &rho_cache, comm)

    free_rhocache(rho_cache)  #delete cache entries
    return pCache
        


cdef dm_compute_pr_cache(double[:,:] ret,
                         vector[vector[int]] c_evalTree,
                         vector[DMGateCRep*] c_gatereps,
                         DMStateCRep *c_rho, vector[DMEffectCRep*] c_ereps,
                         vector[DMStateCRep*]* prho_cache, comm): # any way to transmit comm?
    #Note: we need to take in rho_cache as a pointer b/c we may alter the values its
    # elements point to (instead of copying the states) - we just guarantee that in the end
    # all of the cache entries are filled with allocated (by 'new') states that the caller
    # can deallocate at will.
    cdef int k,l, i, istart, icache
    cdef double p
    cdef int dim = c_rho._dim
    cdef DMStateCRep *init_state
    cdef DMStateCRep *prop1
    cdef DMStateCRep *tprop
    cdef DMStateCRep *final_state
    cdef DMStateCRep *prop2 = new DMStateCRep(dim)
    cdef DMStateCRep *shelved = new DMStateCRep(dim)

    #print "BEGIN"

    #Invariants required for proper memory management:
    # - upon loop entry, prop2 is allocated and prop1 is not (it doesn't "own" any memory)
    # - all rho_cache entries have been allocated via "new"
    for k in range(c_evalTree.size()):
        intarray = c_evalTree[k]
        i = intarray[0]
        istart = intarray[1]
        icache = intarray[2]

        if istart == -1:  init_state = c_rho
        else:             init_state = deref(prho_cache)[istart]
        
        #DEBUG
        #print "LOOP i=",i," istart=",istart," icache=",icache," remcnt=",(intarray.size()-3)
        #print [ init_state._dataptr[t] for t in range(4) ]
    
        #Propagate state rep
        # prop2 should already be alloc'd; need to "allocate" prop1 - either take from cache or from "shelf"
        #OLD: prop1 = new DMStateCRep(init_state._dataptr, dim, <bool>1) 
        prop1 = shelved if icache == -1 else deref(prho_cache)[icache]
        for j in range(dim): prop1._dataptr[j] = init_state._dataptr[j] # copy init_state -> prop1   FUTURE: a method of DMStateCRep?
        #print " prop1:";  print [ prop1._dataptr[t] for t in range(4) ]
        for l in range(3,intarray.size()): #during loop, both prop1 & prop2 are alloc'd        
            c_gatereps[intarray[l]].acton(prop1,prop2)
            #print " post-act prop2:"; print [ prop2._dataptr[t] for t in range(4) ]
            tprop = prop1; prop1 = prop2; prop2 = tprop # swap prop1 <-> prop2
        final_state = prop1 # output = prop1 (after swap from loop above)
        # Note: prop2 is the other alloc'd state and this maintains invariant
        #print " final:"; print [ final_state._dataptr[t] for t in range(4) ]
        
        for j in range(c_ereps.size()):
            p = c_ereps[j].probability(final_state) #outcome probability
            #print("processing ",i,j,p)
            ret[i,j] = p

        if icache != -1:
            deref(prho_cache)[icache] = final_state # store this state in the cache
        else: # our 2nd state was pulled from the shelf before; return it
            shelved = final_state
            final_state = NULL


    #delete our temp states
    del prop2
    del shelved


    
def DM_compute_dpr_cache(calc, rholabel, elabels, evalTree, wrtSlice, comm, scratch=None):

    cdef double eps = 1e-7 #hardcoded?

    #Compute finite difference derivatives, one parameter at a time.
    param_indices = range(calc.Np) if (wrtSlice is None) else _slct.indices(wrtSlice)
    nDerivCols = len(param_indices) # *all*, not just locally computed ones

    rhoVec,EVecs = calc._rhoEs_from_labels(rholabel, elabels)
    pCache = np.empty((len(evalTree),len(elabels)),'d')
    dpr_cache  = np.zeros((len(evalTree), len(elabels), nDerivCols),'d')

    #Get (extension-type) representation objects
    rhorep = calc.preps[rholabel].torep('prep')
    ereps = [ calc.effects[el].torep('effect') for el in elabels]
    gate_lookup = { lbl:i for i,lbl in enumerate(calc.gates.keys()) } # gate labels -> ints for faster lookup
    gatereps = { i:calc.gates[lbl].torep() for lbl,i in gate_lookup.items() }
    
    # convert to C-mode:  evaltree, gate_lookup, gatereps
    cdef c_evalTree = convert_evaltree(evalTree, gate_lookup)
    cdef DMStateCRep *c_rho = convert_rhorep(rhorep)
    cdef vector[DMGateCRep*] c_gatereps = convert_gatereps(gatereps)
    cdef vector[DMEffectCRep*] c_ereps = convert_ereps(ereps)

    # create rho_cache = vector of DMStateCReps
    cdef vector[DMStateCRep*] rho_cache = create_rhocache(evalTree.cache_size(), c_rho._dim)

    dm_compute_pr_cache(pCache, c_evalTree, c_gatereps, c_rho, c_ereps, &rho_cache, comm)
    pCache_delta = pCache.copy() # for taking finite differences

    all_slices, my_slice, owners, subComm = \
            _mpit.distribute_slice(slice(0,len(param_indices)), comm)

    my_param_indices = param_indices[my_slice]
    st = my_slice.start #beginning of where my_param_indices results
                        # get placed into dpr_cache
    
    #Get a map from global parameter indices to the desired
    # final index within dpr_cache
    iParamToFinal = { i: st+ii for ii,i in enumerate(my_param_indices) }

    orig_vec = calc.to_vector().copy()
    for i in range(calc.Np):
        #print("dprobs cache %d of %d" % (i,self.Np))
        if i in iParamToFinal:
            iFinal = iParamToFinal[i]
            vec = orig_vec.copy(); vec[i] += eps
            calc.from_vector(vec)

            #rebuild reps (not evaltree or gate_lookup)
            rhorep = calc.preps[rholabel].torep('prep')
            ereps = [ calc.effects[el].torep('effect') for el in elabels]
            gatereps = { i:calc.gates[lbl].torep() for lbl,i in gate_lookup.items() }
            c_rho = convert_rhorep(rhorep)
            c_ereps = convert_ereps(ereps)
            c_gatereps = convert_gatereps(gatereps)

            dm_compute_pr_cache(pCache_delta, c_evalTree, c_gatereps, c_rho, c_ereps, &rho_cache, comm)
            dpr_cache[:,:,iFinal] = (pCache_delta - pCache)/eps

    calc.from_vector(orig_vec)
    free_rhocache(rho_cache)
    
    #Now each processor has filled the relavant parts of dpr_cache,
    # so gather together:
    _mpit.gather_slices(all_slices, owners, dpr_cache,[], axes=2, comm=comm)
    
    # DEBUG LINE USED FOR MONITORION N-QUBIT GST TESTS
    #print("DEBUG TIME: dpr_cache(Np=%d, dim=%d, cachesize=%d, treesize=%d, napplies=%d) in %gs" % 
    #      (self.Np, self.dim, cacheSize, len(evalTree), evalTree.get_num_applies(), _time.time()-tStart)) #DEBUG

    return dpr_cache


    

#OLD LOOP - maybe good for python version?
#    #comm is currently ignored
#    #TODO: if evalTree is split, distribute among processors
#    for i,iStart,remainder,iCache in ievalTree:
#        if iStart is None:  init_state = rhorep
#        else:               init_state = rho_cache[iStart] #[:,None]
#
#        
#        #Propagate state rep
#        #print("init ",i,str(init_state))
#        #print("applying")
#        #for r in remainder:
#        #    print("mx:")
#        #    print(gatereps[r])
#        final_state = propagate_staterep(init_state, [gatereps[r] for r in remainder] )
#        #print("final ",i,str(final_state))
#        if iCache is not None: rho_cache[iCache] = final_state # [:,0] #store this state in the cache
#
#        for j,erep in enumerate(ereps):
#            p = erep.probability(final_state) #outcome probability
#            #print("processing ",i,j,p)
#            ret[i,j] = p



        #if self.evotype == "statevec":
        #    for j,E in enumerate(EVecs):
        #        ret[i,j] = _np.abs(_np.vdot(E.todense(Escratch),final_state))**2
        #elif self.evotype == "densitymx":
        #    for j,E in enumerate(EVecs):
        #        ret[i,j] = _np.vdot(E.todense(Escratch),final_state)
        #        #OLD (slower): _np.dot(_np.conjugate(E.todense(Escratch)).T,final_state)
        #        # FUTURE: optionally pre-compute todense() results for speed if mem is available?
        #else: # evotype == "stabilizer" case
        #    #TODO: compute using tree-like fanout, only fanning when necessary. -- at least when there are O(d=2^nQ) effects
        #    for j,E in enumerate(EVecs):
        #        ret[i,j] = rho.measurement_probability(E.outcomes)

    #print("DEBUG TIME: pr_cache(dim=%d, cachesize=%d) in %gs" % (self.dim, cacheSize,_time.time()-tStart)) #DEBUG
    #return ret


#cdef struct Gate_crep:
#    int x
#    char * y
#    #put everything potentially necessary here
#ctypedef Gate_crep Gate_crep_t


#def fast_compute_pr_cache(rholabel, elabels, evalTree, comm, scratch=None)::
#    #needs to construct gate creps, etc...
#    
#    #calls propagate_state:
#    
#cdef propagate_state(Gate_crep_t* gate_creps, int* gids,
#                     State_crep* state_crep):
#    for gateid in gids:
#    	actonlib[gateid](gate_creps[gateid],state_crep) # act in-place / don't require copy?
	


## You can also typedef pointers too
#
#ctypedef int * int_ptr
#
#
#ctypedef int (* no_arg_c_func)(BaseThing)
#cdef no_arg_c_func funcs[1000]
#
#ctypedef void (*cfptr)(int)
#
## then we use the function pointer:
#cdef cfptr myfunctionptr = &myfunc



def dot(np.ndarray[double, ndim=1] f, np.ndarray[double, ndim=1] g):
    cdef long N = f.shape[0]
    cdef float ret = 0.0
    cdef int i
    for i in range(N):
        ret += f[i]*g[i]
    return ret


# encoding: utf-8
# cython: profile=False
# cython: linetrace=False
# filename: fastactonlib.pyx

import sys, time
import numpy as np
from libc.stdlib cimport malloc, free
from libcpp cimport bool
from libcpp.vector cimport vector
from libcpp.unordered_map cimport unordered_map
from cython.operator cimport dereference as deref, preincrement as inc
cimport numpy as np
cimport cython

import itertools as _itertools
from ..tools import mpitools as _mpit
from ..tools import slicetools as _slct

#Use 64-bit integers
ctypedef long long INT
ctypedef unsigned long long UINT

cdef extern from "fastreps.h" namespace "CReps":

    # Density Matrix (DM) propagation classes
    cdef cppclass DMStateCRep:
        DMStateCRep() except +
        DMStateCRep(INT) except +
        DMStateCRep(double*,INT,bool) except +
        void copy_from(DMStateCRep*)
        INT _dim
        double* _dataptr

    cdef cppclass DMEffectCRep:
        DMEffectCRep() except +
        DMEffectCRep(INT) except +
        double probability(DMStateCRep* state)
        INT _dim

    cdef cppclass DMEffectCRep_Dense(DMEffectCRep):
        DMEffectCRep_Dense() except +
        DMEffectCRep_Dense(double*,INT) except +
        double probability(DMStateCRep* state)
        INT _dim
        double* _dataptr

    cdef cppclass DMEffectCRep_TensorProd(DMEffectCRep):
        DMEffectCRep_TensorProd() except +
        DMEffectCRep_TensorProd(double*, INT*, INT, INT, INT) except +
        double probability(DMStateCRep* state)
        INT _dim

    cdef cppclass DMEffectCRep_Computational(DMEffectCRep):
        DMEffectCRep_Computational() except +
        DMEffectCRep_Computational(INT, INT, double, INT) except +
        double probability(DMStateCRep* state)
        INT _dim

    cdef cppclass DMGateCRep:
        DMGateCRep(INT) except +
        DMStateCRep* acton(DMStateCRep*, DMStateCRep*)
        DMStateCRep* adjoint_acton(DMStateCRep*, DMStateCRep*)
        INT _dim

    cdef cppclass DMGateCRep_Dense(DMGateCRep):
        DMGateCRep_Dense(double*,INT) except +
        DMStateCRep* acton(DMStateCRep*, DMStateCRep*)
        DMStateCRep* adjoint_acton(DMStateCRep*, DMStateCRep*)
        double* _dataptr
        INT _dim

    cdef cppclass DMGateCRep_Embedded(DMGateCRep):
        DMGateCRep_Embedded(DMGateCRep*, INT*, INT*, INT*, INT*, INT, INT, INT, INT, INT) except +
        DMStateCRep* acton(DMStateCRep*, DMStateCRep*)
        DMStateCRep* adjoint_acton(DMStateCRep*, DMStateCRep*)

    cdef cppclass DMGateCRep_Composed(DMGateCRep):
        DMGateCRep_Composed(vector[DMGateCRep*]) except +
        DMStateCRep* acton(DMStateCRep*, DMStateCRep*)
        DMStateCRep* adjoint_acton(DMStateCRep*, DMStateCRep*)

    cdef cppclass DMGateCRep_Lindblad(DMGateCRep):
        DMGateCRep_Lindblad(double* A_data, INT* A_indices, INT* A_indptr, INT nnz,
			    double mu, double eta, INT m_star, INT s, INT dim,
			    double* unitarypost_data, INT* unitarypost_indices,
                            INT* unitarypost_indptr, INT unitarypost_nnz) except +
        DMStateCRep* acton(DMStateCRep*, DMStateCRep*)
        DMStateCRep* adjoint_acton(DMStateCRep*, DMStateCRep*)

    # State vector (SV) propagation classes
    cdef cppclass SVStateCRep:
        SVStateCRep() except +
        SVStateCRep(INT) except +
        SVStateCRep(double complex*,INT,bool) except +
        void copy_from(SVStateCRep*)
        INT _dim
        double complex* _dataptr

    cdef cppclass SVEffectCRep:
        SVEffectCRep() except +
        SVEffectCRep(INT) except +
        double probability(SVStateCRep* state)
        double complex amplitude(SVStateCRep* state)
        INT _dim

    cdef cppclass SVEffectCRep_Dense(SVEffectCRep):
        SVEffectCRep_Dense() except +
        SVEffectCRep_Dense(double complex*,INT) except +
        double probability(SVStateCRep* state)
        double complex amplitude(SVStateCRep* state)
        INT _dim
        double complex* _dataptr

    cdef cppclass SVEffectCRep_TensorProd(SVEffectCRep):
        SVEffectCRep_TensorProd() except +
        SVEffectCRep_TensorProd(double complex*, INT*, INT, INT, INT) except +
        double probability(SVStateCRep* state)
        double complex amplitude(SVStateCRep* state)
        INT _dim

    cdef cppclass SVGateCRep:
        SVGateCRep(INT) except +
        SVStateCRep* acton(SVStateCRep*, SVStateCRep*)
        SVStateCRep* adjoint_acton(SVStateCRep*, SVStateCRep*)
        INT _dim

    cdef cppclass SVGateCRep_Dense(SVGateCRep):
        SVGateCRep_Dense(double complex*,INT) except +
        SVStateCRep* acton(SVStateCRep*, SVStateCRep*)
        SVStateCRep* adjoint_acton(SVStateCRep*, SVStateCRep*)
        double complex* _dataptr
        INT _dim

    cdef cppclass SVGateCRep_Embedded(SVGateCRep):
        SVGateCRep_Embedded(SVGateCRep*, INT*, INT*, INT*, INT*, INT, INT, INT, INT, INT) except +
        SVStateCRep* acton(SVStateCRep*, SVStateCRep*)
        SVStateCRep* adjoint_acton(SVStateCRep*, SVStateCRep*)

    cdef cppclass SVGateCRep_Composed(SVGateCRep):
        SVGateCRep_Composed(vector[SVGateCRep*]) except +
        SVStateCRep* acton(SVStateCRep*, SVStateCRep*)
        SVStateCRep* adjoint_acton(SVStateCRep*, SVStateCRep*)

        
    # Stabilizer state (SB) propagation classes
    cdef cppclass SBStateCRep:
        SBStateCRep(INT*, INT*, double complex*, INT, INT) except +
        SBStateCRep(INT, INT) except +
        SBStateCRep(double*,INT,bool) except +
        void copy_from(SBStateCRep*)
        INT _n
        INT _namps
        # for DEBUG
        INT* _smatrix
        INT* _pvectors
        INT _zblock_start
        double complex* _amps


    cdef cppclass SBEffectCRep:
        SBEffectCRep(INT*, INT) except +
        double probability(SBStateCRep* state)
        double complex amplitude(SBStateCRep* state)
        INT _n

    cdef cppclass SBGateCRep:
        SBGateCRep(INT) except +
        SBStateCRep* acton(SBStateCRep*, SBStateCRep*)
        SBStateCRep* adjoint_acton(SBStateCRep*, SBStateCRep*)
        INT _n

    cdef cppclass SBGateCRep_Embedded(SBGateCRep):
        SBGateCRep_Embedded(SBGateCRep*, INT, INT*, INT) except +
        SBStateCRep* acton(SBStateCRep*, SBStateCRep*)
        SBStateCRep* adjoint_acton(SBStateCRep*, SBStateCRep*)

    cdef cppclass SBGateCRep_Composed(SBGateCRep):
        SBGateCRep_Composed(vector[SBGateCRep*]) except +
        SBStateCRep* acton(SBStateCRep*, SBStateCRep*)
        SBStateCRep* adjoint_acton(SBStateCRep*, SBStateCRep*)

    cdef cppclass SBGateCRep_Clifford(SBGateCRep):
        SBGateCRep_Clifford(INT*, INT*, double complex*, INT*, INT*, double complex*, INT) except +
        SBStateCRep* acton(SBStateCRep*, SBStateCRep*)
        SBStateCRep* adjoint_acton(SBStateCRep*, SBStateCRep*)
        #for DEBUG:
        INT *_smatrix
        INT *_svector
        INT *_smatrix_inv
        INT *_svector_inv
        double complex *_unitary
        double complex *_unitary_adj


    #Other classes
    cdef cppclass PolyCRep:
        PolyCRep() except +        
        PolyCRep(unordered_map[INT, complex], INT, INT) except +
        PolyCRep mult(PolyCRep&)
        void add_inplace(PolyCRep&)
        void scale(double complex scale)
        unordered_map[INT, complex] _coeffs
        INT _max_order
        INT _max_num_vars
    

    cdef cppclass SVTermCRep:    
        SVTermCRep(PolyCRep*, SVStateCRep*, SVStateCRep*, vector[SVGateCRep*], vector[SVGateCRep*]) except +
        SVTermCRep(PolyCRep*, SVEffectCRep*, SVEffectCRep*, vector[SVGateCRep*], vector[SVGateCRep*]) except +
        SVTermCRep(PolyCRep*, vector[SVGateCRep*], vector[SVGateCRep*]) except +
        PolyCRep* _coeff
        SVStateCRep* _pre_state
        SVEffectCRep* _pre_effect
        vector[SVGateCRep*] _pre_ops
        SVStateCRep* _post_state
        SVEffectCRep* _post_effect
        vector[SVGateCRep*] _post_ops

    cdef cppclass SBTermCRep:    
        SBTermCRep(PolyCRep*, SBStateCRep*, SBStateCRep*, vector[SBGateCRep*], vector[SBGateCRep*]) except +
        SBTermCRep(PolyCRep*, SBEffectCRep*, SBEffectCRep*, vector[SBGateCRep*], vector[SBGateCRep*]) except +
        SBTermCRep(PolyCRep*, vector[SBGateCRep*], vector[SBGateCRep*]) except +
        PolyCRep* _coeff
        SBStateCRep* _pre_state
        SBEffectCRep* _pre_effect
        vector[SBGateCRep*] _pre_ops
        SBStateCRep* _post_state
        SBEffectCRep* _post_effect
        vector[SBGateCRep*] _post_ops

        
    

ctypedef DMGateCRep* DMGateCRep_ptr
ctypedef DMStateCRep* DMStateCRep_ptr
ctypedef DMEffectCRep* DMEffectCRep_ptr
ctypedef SVGateCRep* SVGateCRep_ptr
ctypedef SVStateCRep* SVStateCRep_ptr
ctypedef SVEffectCRep* SVEffectCRep_ptr
ctypedef SVTermCRep* SVTermCRep_ptr
ctypedef SBGateCRep* SBGateCRep_ptr
ctypedef SBStateCRep* SBStateCRep_ptr
ctypedef SBEffectCRep* SBEffectCRep_ptr
ctypedef SBTermCRep* SBTermCRep_ptr
ctypedef PolyCRep* PolyCRep_ptr
ctypedef vector[SVTermCRep_ptr]* vector_SVTermCRep_ptr_ptr
ctypedef vector[SBTermCRep_ptr]* vector_SBTermCRep_ptr_ptr

#Create a function pointer type for term-based calc inner loop
ctypedef void (*sv_innerloopfn_ptr)(vector[vector_SVTermCRep_ptr_ptr],
                                    vector[INT]*, vector[PolyCRep*]*, INT)
ctypedef void (*sb_innerloopfn_ptr)(vector[vector_SBTermCRep_ptr_ptr],
                                    vector[INT]*, vector[PolyCRep*]*, INT)

        
#cdef class StateRep:
#    pass

# Density matrix (DM) propagation wrapper classes
cdef class DMStateRep: #(StateRep):
    cdef DMStateCRep* c_state
    cdef np.ndarray data_ref
    #cdef double [:] data_view # alt way to hold a reference

    def __cinit__(self, np.ndarray[double, ndim=1, mode='c'] data):
        #print("PYX state constructed w/dim ",data.shape[0])
        #cdef np.ndarray[double, ndim=1, mode='c'] np_cbuf = np.ascontiguousarray(data, dtype='d') # would allow non-contig arrays
        #cdef double [:] view = data;  self.data_view = view # ALT: holds reference...
        self.data_ref = data # holds reference to data so it doesn't get garbage collected - or could copy=true
        #self.c_state = new DMStateCRep(<double*>np_cbuf.data,<INT>np_cbuf.shape[0],<bool>0)
        self.c_state = new DMStateCRep(<double*>data.data,<INT>data.shape[0],<bool>0)

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
                                               <INT>data.shape[0])

cdef class DMEffectRep_TensorProd(DMEffectRep):
    cdef np.ndarray data_ref1
    cdef np.ndarray data_ref2

    def __cinit__(self, np.ndarray[double, ndim=2, mode='c'] kron_array,
                  np.ndarray[np.int64_t, ndim=1, mode='c'] factor_dims, INT nfactors, INT max_factor_dim, INT dim):
        # cdef INT dim = np.product(factor_dims) -- just send as argument
        self.data_ref1 = kron_array
        self.data_ref2 = factor_dims
        self.c_effect = new DMEffectCRep_TensorProd(<double*>kron_array.data,
                                                    <INT*>factor_dims.data,
                                                    nfactors, max_factor_dim, dim)


cdef class DMEffectRep_Computational(DMEffectRep):

    def __cinit__(self, np.ndarray[np.int64_t, ndim=1, mode='c'] zvals, INT dim):
        # cdef INT dim = 4**zvals.shape[0] -- just send as argument
        cdef INT nfactors = zvals.shape[0]
        cdef double abs_elval = 1/(np.sqrt(2)**nfactors)
        cdef INT base = 1
        cdef INT zvals_int = 0
        for i in range(nfactors):
            zvals_int += base * zvals[i]
            base = base << 1 # *= 2
        self.c_effect = new DMEffectCRep_Computational(nfactors, zvals_int, abs_elval, dim)


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

    def adjoint_acton(self, DMStateRep state not None):
        cdef DMStateRep out_state = DMStateRep(np.empty(self.c_gate._dim, dtype='d'))
        #print("PYX acton called w/dim ", self.c_gate._dim, out_state.c_state._dim)
        # assert(state.c_state._dataptr != out_state.c_state._dataptr) # DEBUG
        self.c_gate.adjoint_acton(state.c_state, out_state.c_state)
        return out_state

        
cdef class DMGateRep_Dense(DMGateRep):
    cdef np.ndarray data_ref

    def __cinit__(self, np.ndarray[double, ndim=2, mode='c'] data):
        self.data_ref = data
        #print("PYX dense gate constructed w/dim ",data.shape[0])
        self.c_gate = new DMGateCRep_Dense(<double*>data.data,
                                           <INT>data.shape[0])
    def __str__(self):
        s = ""
        cdef DMGateCRep_Dense* my_cgate = <DMGateCRep_Dense*>self.c_gate # b/c we know it's a _Dense gate...
        cdef INT i,j,k 
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
                  np.ndarray[np.int64_t, ndim=1, mode='c'] numBasisEls,
                  np.ndarray[np.int64_t, ndim=1, mode='c'] actionInds,                  
                  np.ndarray[np.int64_t, ndim=1, mode='c'] blocksizes,
		  INT embedded_dim, INT nComponentsInActiveBlock,
                  INT iActiveBlock, INT nBlocks, INT dim):

#OLD: TODO REMOVE
#                 np.ndarray[np.int64_t, ndim=1, mode='c'] noop_incrementers,
#		  np.ndarray[np.int64_t, ndim=1, mode='c'] numBasisEls_noop_blankaction,
#                 np.ndarray[np.int64_t, ndim=1, mode='c'] baseinds,


        cdef INT i, j

        # numBasisEls_noop_blankaction is just numBasisEls with actionInds == 1
        cdef np.ndarray[np.int64_t, ndim=1, mode='c'] numBasisEls_noop_blankaction = numBasisEls.copy()
        for i in actionInds:
            numBasisEls_noop_blankaction[i] = 1 # for indexing the identity space

        # multipliers to go from per-label indices to tensor-product-block index
        # e.g. if map(len,basisInds) == [1,4,4] then multipliers == [ 16 4 1 ]
        cdef np.ndarray tmp = np.empty(nComponentsInActiveBlock,np.int64)
        tmp[0] = 1
        for i in range(1,nComponentsInActiveBlock):
            tmp[i] = numBasisEls[nComponentsInActiveBlock-1-i]
        multipliers = np.array( np.flipud( np.cumprod(tmp) ), np.int64)
        
        # noop_incrementers[i] specifies how much the overall vector index
        #  is incremented when the i-th "component" digit is advanced
        cdef INT dec = 0
        cdef np.ndarray[np.int64_t, ndim=1, mode='c'] noop_incrementers = np.empty(nComponentsInActiveBlock,np.int64)
        for i in range(nComponentsInActiveBlock-1,-1,-1):
            noop_incrementers[i] = multipliers[i] - dec
            dec += (numBasisEls_noop_blankaction[i]-1)*multipliers[i]

        cdef INT vec_index
        cdef INT offset = 0 #number of basis elements preceding our block's elements
        for i in range(iActiveBlock):
            offset += blocksizes[i]
                
        # self.baseinds specifies the contribution from the "active
        #  component" digits to the overall vector index.
        cdef np.ndarray[np.int64_t, ndim=1, mode='c'] baseinds = np.empty(embedded_dim,np.int64)
        basisInds_action = [ list(range(numBasisEls[i])) for i in actionInds ]
        for ii,gate_b in enumerate(_itertools.product(*basisInds_action)):
            vec_index = offset
            for j,bInd in zip(actionInds,gate_b):
                vec_index += multipliers[j]*bInd
            baseinds[ii] = vec_index
        
        self.data_ref1 = noop_incrementers
        self.data_ref2 = numBasisEls_noop_blankaction
        self.data_ref3 = baseinds
        self.data_ref4 = blocksizes
        self.embedded = embedded_gate # needed to prevent garbage collection?
        self.c_gate = new DMGateCRep_Embedded(embedded_gate.c_gate,
                                              <INT*>noop_incrementers.data, <INT*>numBasisEls_noop_blankaction.data,
                                              <INT*>baseinds.data, <INT*>blocksizes.data,
                                              embedded_dim, nComponentsInActiveBlock,
                                              iActiveBlock, nBlocks, dim)


cdef class DMGateRep_Composed(DMGateRep):
    cdef object list_of_factors # list of DMGateRep objs?

    def __cinit__(self, factor_gate_reps):
        self.list_of_factors = factor_gate_reps
        cdef INT i
        cdef INT nfactors = len(factor_gate_reps)
        cdef vector[DMGateCRep*] gate_creps = vector[DMGateCRep_ptr](nfactors)
        for i in range(nfactors):
            gate_creps[i] = (<DMGateRep?>factor_gate_reps[i]).c_gate
        self.c_gate = new DMGateCRep_Composed(gate_creps)


cdef class DMGateRep_Lindblad(DMGateRep):
    cdef np.ndarray data_ref1
    cdef np.ndarray data_ref2
    cdef np.ndarray data_ref3
    cdef np.ndarray data_ref4
    cdef np.ndarray data_ref5
    cdef np.ndarray data_ref6

    def __cinit__(self, np.ndarray[double, ndim=1, mode='c'] A_data,
                  np.ndarray[np.int64_t, ndim=1, mode='c'] A_indices,
                  np.ndarray[np.int64_t, ndim=1, mode='c'] A_indptr,
                  double mu, double eta, INT m_star, INT s,
                  np.ndarray[double, ndim=1, mode='c'] unitarypost_data,
                  np.ndarray[np.int64_t, ndim=1, mode='c'] unitarypost_indices,
                  np.ndarray[np.int64_t, ndim=1, mode='c'] unitarypost_indptr):
        self.data_ref1 = A_data
        self.data_ref2 = A_indices
        self.data_ref3 = A_indptr
        self.data_ref4 = unitarypost_data
        self.data_ref5 = unitarypost_indices
        self.data_ref6 = unitarypost_indptr
        cdef INT nnz = A_data.shape[0]
        cdef INT dim = A_indptr.shape[0]-1
        cdef INT upost_nnz = unitarypost_data.shape[0]
        self.c_gate = new DMGateCRep_Lindblad(<double*>A_data.data, <INT*>A_indices.data,
                                              <INT*>A_indptr.data, nnz, mu, eta, m_star, s, dim,
                                              <double*>unitarypost_data.data,
                                              <INT*>unitarypost_indices.data,
                                              <INT*>unitarypost_indptr.data, upost_nnz)

# State vector (SV) propagation wrapper classes
cdef class SVStateRep: #(StateRep):
    cdef SVStateCRep* c_state
    cdef np.ndarray data_ref

    def __cinit__(self, np.ndarray[np.complex128_t, ndim=1, mode='c'] data):
        self.data_ref = data # holds reference to data so it doesn't get garbage collected - or could copy=true
        self.c_state = new SVStateCRep(<double complex*>data.data,<INT>data.shape[0],<bool>0)

    def __dealloc__(self):
        del self.c_state

    def __str__(self):
        return str([self.c_state._dataptr[i] for i in range(self.c_state._dim)])


cdef class SVEffectRep:
    cdef SVEffectCRep* c_effect

    def __cinit__(self):
        pass # no init; could set self.c_effect = NULL? could assert(False)?
    def __dealloc__(self):
        del self.c_effect # check for NULL?

    def probability(self, SVStateRep state not None):
        #unnecessary (just put in signature): cdef SVStateRep st = <SVStateRep?>state
        return self.c_effect.probability(state.c_state)

cdef class SVEffectRep_Dense(SVEffectRep):
    cdef np.ndarray data_ref

    def __cinit__(self, np.ndarray[np.complex128_t, ndim=1, mode='c'] data):
        self.data_ref = data # holds reference to data
        self.c_effect = new SVEffectCRep_Dense(<double complex*>data.data,
                                               <INT>data.shape[0])

cdef class SVEffectRep_TensorProd(SVEffectRep):
    cdef np.ndarray data_ref1
    cdef np.ndarray data_ref2

    def __cinit__(self, np.ndarray[np.complex128_t, ndim=2, mode='c'] kron_array,
                  np.ndarray[np.int64_t, ndim=1, mode='c'] factor_dims, INT nfactors, INT max_factor_dim, INT dim):
        # cdef INT dim = np.product(factor_dims) -- just send as argument
        self.data_ref1 = kron_array
        self.data_ref2 = factor_dims
        self.c_effect = new SVEffectCRep_TensorProd(<double complex*>kron_array.data,
                                                    <INT*>factor_dims.data,
                                                    nfactors, max_factor_dim, dim)


cdef class SVGateRep:
    cdef SVGateCRep* c_gate

    def __cinit__(self):
        pass # self.c_gate = NULL ?
    
    def __dealloc__(self):
        del self.c_gate

    def acton(self, SVStateRep state not None):
        cdef SVStateRep out_state = SVStateRep(np.empty(self.c_gate._dim, dtype=np.complex128))
        #print("PYX acton called w/dim ", self.c_gate._dim, out_state.c_state._dim)
        # assert(state.c_state._dataptr != out_state.c_state._dataptr) # DEBUG
        self.c_gate.acton(state.c_state, out_state.c_state)
        return out_state

    #FUTURE: adjoint acton
        
cdef class SVGateRep_Dense(SVGateRep):
    cdef np.ndarray data_ref

    def __cinit__(self, np.ndarray[np.complex128_t, ndim=2, mode='c'] data):
        self.data_ref = data
        #print("PYX dense gate constructed w/dim ",data.shape[0])
        self.c_gate = new SVGateCRep_Dense(<double complex*>data.data,
                                           <INT>data.shape[0])
    def __str__(self):
        s = ""
        cdef SVGateCRep_Dense* my_cgate = <SVGateCRep_Dense*>self.c_gate # b/c we know it's a _Dense gate...
        cdef INT i,j,k 
        for i in range(my_cgate._dim):
            k = i*my_cgate._dim
            for j in range(my_cgate._dim):
                s += str(my_cgate._dataptr[k+j]) + " "
            s += "\n"
        return s


cdef class SVGateRep_Embedded(SVGateRep):
    cdef np.ndarray data_ref1
    cdef np.ndarray data_ref2
    cdef np.ndarray data_ref3
    cdef np.ndarray data_ref4
    cdef SVGateRep embedded


    def __cinit__(self, SVGateRep embedded_gate,
                  np.ndarray[np.int64_t, ndim=1, mode='c'] numBasisEls,
                  np.ndarray[np.int64_t, ndim=1, mode='c'] actionInds,                  
                  np.ndarray[np.int64_t, ndim=1, mode='c'] blocksizes,
		  INT embedded_dim, INT nComponentsInActiveBlock,
                  INT iActiveBlock, INT nBlocks, INT dim):

        cdef INT i, j

        # numBasisEls_noop_blankaction is just numBasisEls with actionInds == 1
        cdef np.ndarray[np.int64_t, ndim=1, mode='c'] numBasisEls_noop_blankaction = numBasisEls.copy()
        for i in actionInds:
            numBasisEls_noop_blankaction[i] = 1 # for indexing the identity space

        # multipliers to go from per-label indices to tensor-product-block index
        # e.g. if map(len,basisInds) == [1,4,4] then multipliers == [ 16 4 1 ]
        cdef np.ndarray tmp = np.empty(nComponentsInActiveBlock,np.int64)
        tmp[0] = 1
        for i in range(1,nComponentsInActiveBlock):
            tmp[i] = numBasisEls[nComponentsInActiveBlock-1-i]
        multipliers = np.array( np.flipud( np.cumprod(tmp) ), np.int64)
        
        # noop_incrementers[i] specifies how much the overall vector index
        #  is incremented when the i-th "component" digit is advanced
        cdef INT dec = 0
        cdef np.ndarray[np.int64_t, ndim=1, mode='c'] noop_incrementers = np.empty(nComponentsInActiveBlock,np.int64)
        for i in range(nComponentsInActiveBlock-1,-1,-1):
            noop_incrementers[i] = multipliers[i] - dec
            dec += (numBasisEls_noop_blankaction[i]-1)*multipliers[i]

        cdef INT vec_index
        cdef INT offset = 0 #number of basis elements preceding our block's elements
        for i in range(iActiveBlock):
            offset += blocksizes[i]
                
        # self.baseinds specifies the contribution from the "active
        #  component" digits to the overall vector index.
        cdef np.ndarray[np.int64_t, ndim=1, mode='c'] baseinds = np.empty(embedded_dim,np.int64)
        basisInds_action = [ list(range(numBasisEls[i])) for i in actionInds ]
        for ii,gate_b in enumerate(_itertools.product(*basisInds_action)):
            vec_index = offset
            for j,bInd in zip(actionInds,gate_b):
                vec_index += multipliers[j]*bInd
            baseinds[ii] = vec_index
        
        self.data_ref1 = noop_incrementers
        self.data_ref2 = numBasisEls_noop_blankaction
        self.data_ref3 = baseinds
        self.data_ref4 = blocksizes
        self.embedded = embedded_gate # needed to prevent garbage collection?
        self.c_gate = new SVGateCRep_Embedded(embedded_gate.c_gate,
                                              <INT*>noop_incrementers.data, <INT*>numBasisEls_noop_blankaction.data,
                                              <INT*>baseinds.data, <INT*>blocksizes.data,
                                              embedded_dim, nComponentsInActiveBlock,
                                              iActiveBlock, nBlocks, dim)


cdef class SVGateRep_Composed(SVGateRep):
    cdef object list_of_factors # list of SVGateRep objs?

    def __cinit__(self, factor_gate_reps):
        self.list_of_factors = factor_gate_reps
        cdef INT i
        cdef INT nfactors = len(factor_gate_reps)
        cdef vector[SVGateCRep*] gate_creps = vector[SVGateCRep_ptr](nfactors)
        for i in range(nfactors):
            gate_creps[i] = (<SVGateRep?>factor_gate_reps[i]).c_gate
        self.c_gate = new SVGateCRep_Composed(gate_creps)

        
# Stabilizer state (SB) propagation wrapper classes
cdef class SBStateRep: #(StateRep):
    cdef SBStateCRep* c_state
    cdef np.ndarray data_ref1
    cdef np.ndarray data_ref2
    cdef np.ndarray data_ref3

    def __cinit__(self, np.ndarray[np.int64_t, ndim=2, mode='c'] smatrix,
                  np.ndarray[np.int64_t, ndim=2, mode='c'] pvectors,
                  np.ndarray[np.complex128_t, ndim=1, mode='c'] amps):
        self.data_ref1 = smatrix
        self.data_ref2 = pvectors
        self.data_ref3 = amps
        cdef INT namps = amps.shape[0]
        cdef INT n = smatrix.shape[0] // 2
        self.c_state = new SBStateCRep(<INT*>smatrix.data,<INT*>pvectors.data,
                                       <double complex*>amps.data, namps, n)

    def __dealloc__(self):
        del self.c_state

    def __str__(self):
        #DEBUG
        cdef INT n = self.c_state._n
        cdef INT namps = self.c_state._namps
        s = "SBStateRep\n"
        s +=" smx = " + str([ self.c_state._smatrix[ii] for ii in range(2*n*2*n) ])
        s +=" pvecs = " + str([ self.c_state._pvectors[ii] for ii in range(2*n) ])
        s +=" amps = " + str([ self.c_state._amps[ii] for ii in range(namps) ])
        s +=" zstart = " + str(self.c_state._zblock_start)
        return s


cdef class SBEffectRep:
    cdef SBEffectCRep* c_effect
    cdef np.ndarray data_ref

    def __cinit__(self, np.ndarray[np.int64_t, ndim=1, mode='c'] zvals):
        self.data_ref = zvals
        self.c_effect = new SBEffectCRep(<INT*>zvals.data,
                                         <INT>zvals.shape[0])

    def __dealloc__(self):
        del self.c_effect # check for NULL?

    def probability(self, SBStateRep state not None):
        #unnecessary (just put in signature): cdef SBStateRep st = <SBStateRep?>state
        return self.c_effect.probability(state.c_state)

    def amplitude(self, SBStateRep state not None):
        return self.c_effect.amplitude(state.c_state)



cdef class SBGateRep:
    cdef SBGateCRep* c_gate

    def __cinit__(self):
        pass # self.c_gate = NULL ?
    
    def __dealloc__(self):
        del self.c_gate

    def acton(self, SBStateRep state not None):
        cdef INT n = self.c_gate._n
        cdef INT namps = state.c_state._namps
        cdef SBStateRep out_state = SBStateRep(np.empty((2*n,2*n), dtype=np.int64),
                                               np.empty((namps,2*n), dtype=np.int64),
                                               np.empty(namps, dtype=np.complex128))
        self.c_gate.acton(state.c_state, out_state.c_state)
        return out_state

    def adjoint_acton(self, SBStateRep state not None):
        cdef INT n = self.c_gate._n
        cdef INT namps = state.c_state._namps
        cdef SBStateRep out_state = SBStateRep(np.empty((2*n,2*n), dtype=np.int64),
                                               np.empty((namps,2*n), dtype=np.int64),
                                               np.empty(namps, dtype=np.complex128))
        self.c_gate.adjoint_acton(state.c_state, out_state.c_state)
        return out_state


cdef class SBGateRep_Embedded(SBGateRep):
    cdef np.ndarray data_ref
    cdef SBGateRep embedded

    def __cinit__(self, SBGateRep embedded_gate, INT n, 
                  np.ndarray[np.int64_t, ndim=1, mode='c'] qubits):
        self.data_ref = qubits
        self.embedded = embedded_gate # needed to prevent garbage collection?
        self.c_gate = new SBGateCRep_Embedded(embedded_gate.c_gate, n,
                                              <INT*>qubits.data, <INT>qubits.shape[0])


cdef class SBGateRep_Composed(SBGateRep):
    cdef object list_of_factors # list of SBGateRep objs?

    def __cinit__(self, factor_gate_reps):
        self.list_of_factors = factor_gate_reps
        cdef INT i
        cdef INT nfactors = len(factor_gate_reps)
        cdef vector[SBGateCRep*] gate_creps = vector[SBGateCRep_ptr](nfactors)
        for i in range(nfactors):
            gate_creps[i] = (<SBGateRep?>factor_gate_reps[i]).c_gate
        self.c_gate = new SBGateCRep_Composed(gate_creps)



cdef class SBGateRep_Clifford(SBGateRep):
    cdef np.ndarray data_ref1
    cdef np.ndarray data_ref2
    cdef np.ndarray data_ref3
    cdef np.ndarray data_ref4
    cdef np.ndarray data_ref5
    cdef np.ndarray data_ref6

    def __cinit__(self, np.ndarray[np.int64_t, ndim=2, mode='c'] smatrix,
                  np.ndarray[np.int64_t, ndim=1, mode='c'] svector,
                  np.ndarray[np.int64_t, ndim=2, mode='c'] smatrix_inv,
                  np.ndarray[np.int64_t, ndim=1, mode='c'] svector_inv,
                  np.ndarray[np.complex128_t, ndim=2, mode='c'] unitary):
        
        self.data_ref1 = smatrix
        self.data_ref2 = svector
        self.data_ref3 = unitary
        self.data_ref4 = smatrix_inv
        self.data_ref5 = svector_inv
        self.data_ref6 = np.ascontiguousarray(np.conjugate(np.transpose(unitary)))
           # the "ascontiguousarray" is crucial, since we just use the .data below
        cdef INT n = smatrix.shape[0] // 2
        self.c_gate = new SBGateCRep_Clifford(<INT*>smatrix.data, <INT*>svector.data, <double complex*>unitary.data,
                                              <INT*>smatrix_inv.data, <INT*>svector_inv.data,
                                              <double complex*>self.data_ref6.data, n)

# Other classes
cdef class PolyRep:
    cdef PolyCRep* c_poly
    
    #Use normal init here so can bypass to create from an already alloc'd c_poly
    def __init__(self, coeff_dict, INT max_order, INT max_num_vars):
        cdef unordered_map[INT, complex] coeffs
        for i,c in coeff_dict.items():
            coeffs[<INT>i] = <double complex>c
        self.c_poly = new PolyCRep(coeffs, max_order, max_num_vars)
    
    def __dealloc__(self):
        del self.c_poly

    @property
    def max_order(self): # so we can convert back to python Polys
        return self.c_poly._max_order
    
    @property
    def max_num_vars(self): # so we can convert back to python Polys
        return self.c_poly._max_num_vars

    @property
    def coeffs(self): # so we can convert back to python Polys
        return self.c_poly._coeffs


    
cdef class SVTermRep:
    cdef SVTermCRep* c_term

    #Hold references to other reps so we don't GC them
    cdef PolyRep coeff_ref
    cdef SVStateRep state_ref1
    cdef SVStateRep state_ref2
    cdef SVEffectRep effect_ref1
    cdef SVEffectRep effect_ref2
    cdef object list_of_preops_ref
    cdef object list_of_postops_ref

    def __cinit__(self, PolyRep coeff, SVStateRep pre_state, SVStateRep post_state,
                  SVEffectRep pre_effect, SVEffectRep post_effect, pre_ops, post_ops):
        self.coeff_ref = coeff
        self.list_of_preops_ref = pre_ops
        self.list_of_postops_ref = post_ops

        cdef INT i
        cdef INT npre = len(pre_ops)
        cdef INT npost = len(post_ops)
        cdef vector[SVGateCRep*] c_pre_ops = vector[SVGateCRep_ptr](npre)
        cdef vector[SVGateCRep*] c_post_ops = vector[SVGateCRep_ptr](<INT>len(post_ops))
        for i in range(npre):
            c_pre_ops[i] = (<SVGateRep?>pre_ops[i]).c_gate
        for i in range(npost):
            c_post_ops[i] = (<SVGateRep?>post_ops[i]).c_gate
        
        if pre_state is not None or post_state is not None:
            assert(pre_state is not None and post_state is not None)
            self.state_ref1 = pre_state
            self.state_ref2 = post_state
            self.c_term = new SVTermCRep(coeff.c_poly, pre_state.c_state, post_state.c_state,
                                         c_pre_ops, c_post_ops);
        elif pre_effect is not None or post_effect is not None:
            assert(pre_effect is not None and post_effect is not None)
            self.effect_ref1 = pre_effect
            self.effect_ref2 = post_effect
            self.c_term = new SVTermCRep(coeff.c_poly, pre_effect.c_effect, post_effect.c_effect,
                                         c_pre_ops, c_post_ops);
        else:
            self.c_term = new SVTermCRep(coeff.c_poly, c_pre_ops, c_post_ops);

    def __dealloc__(self):
        del self.c_term


cdef class SBTermRep:
    cdef SBTermCRep* c_term

    #Hold references to other reps so we don't GC them
    cdef PolyRep coeff_ref
    cdef SBStateRep state_ref1
    cdef SBStateRep state_ref2
    cdef SBEffectRep effect_ref1
    cdef SBEffectRep effect_ref2
    cdef object list_of_preops_ref
    cdef object list_of_postops_ref

    def __cinit__(self, PolyRep coeff, SBStateRep pre_state, SBStateRep post_state,
                  SBEffectRep pre_effect, SBEffectRep post_effect, pre_ops, post_ops):
        self.coeff_ref = coeff
        self.list_of_preops_ref = pre_ops
        self.list_of_postops_ref = post_ops

        cdef INT i
        cdef INT npre = len(pre_ops)
        cdef INT npost = len(post_ops)
        cdef vector[SBGateCRep*] c_pre_ops = vector[SBGateCRep_ptr](npre)
        cdef vector[SBGateCRep*] c_post_ops = vector[SBGateCRep_ptr](<INT>len(post_ops))
        for i in range(npre):
            c_pre_ops[i] = (<SBGateRep?>pre_ops[i]).c_gate
        for i in range(npost):
            c_post_ops[i] = (<SBGateRep?>post_ops[i]).c_gate
        
        if pre_state is not None or post_state is not None:
            assert(pre_state is not None and post_state is not None)
            self.state_ref1 = pre_state
            self.state_ref2 = post_state
            self.c_term = new SBTermCRep(coeff.c_poly, pre_state.c_state, post_state.c_state,
                                         c_pre_ops, c_post_ops);
        elif pre_effect is not None or post_effect is not None:
            assert(pre_effect is not None and post_effect is not None)
            self.effect_ref1 = pre_effect
            self.effect_ref2 = post_effect
            self.c_term = new SBTermCRep(coeff.c_poly, pre_effect.c_effect, post_effect.c_effect,
                                         c_pre_ops, c_post_ops);
        else:
            self.c_term = new SBTermCRep(coeff.c_poly, c_pre_ops, c_post_ops);

    def __dealloc__(self):
        del self.c_term


## END CLASSES -- BEGIN CALC METHODS


def propagate_staterep(staterep, gatereps):
    # FUTURE: could use inner C-reps to do propagation
    # instead of using extension type wrappers as this does now
    ret = staterep
    for gaterep in gatereps:
        ret = gaterep.acton(ret)
        # DEBUG print("post-action rhorep = ",str(ret))
    return ret


cdef vector[vector[INT]] convert_evaltree(evalTree, gate_lookup):
    # c_evalTree :
    # an array of INT-arrays; each INT-array is [i,iStart,iCache,<remainder gate indices>]
    cdef vector[INT] intarray
    cdef vector[vector[INT]] c_evalTree = vector[vector[INT]](len(evalTree))
    for kk,ii in enumerate(evalTree.get_evaluation_order()):
        iStart,remainder,iCache = evalTree[ii]
        if iStart is None: iStart = -1 # so always an int
        if iCache is None: iCache = -1 # so always an int
        intarray = vector[INT](3 + len(remainder))
        intarray[0] = ii
        intarray[1] = iStart
        intarray[2] = iCache
        for jj,gl in enumerate(remainder,start=3):
            intarray[jj] = gate_lookup[gl]
        c_evalTree[kk] = intarray
        
    return c_evalTree

cdef vector[DMStateCRep*] create_rhocache(INT cacheSize, INT state_dim):
    cdef INT i
    cdef vector[DMStateCRep*] rho_cache = vector[DMStateCRep_ptr](cacheSize)
    for i in range(cacheSize): # fill cache with empty but alloc'd states
        rho_cache[i] = new DMStateCRep(state_dim)
    return rho_cache

cdef void free_rhocache(vector[DMStateCRep*] rho_cache):
    cdef UINT i
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


def DM_compute_pr_cache(calc, rholabel, elabels, evalTree, comm):

    rhoVec,EVecs = calc._rhoEs_from_labels(rholabel, elabels)
    pCache = np.empty((len(evalTree),len(EVecs)),'d')

    #Get (extension-type) representation objects
    rhorep = rhoVec.torep('prep')
    ereps = [ E.torep('effect') for E in EVecs]  # could cache these? then have torep keep a non-dense rep that can be quickly kron'd for a tensorprod spamvec
    gate_lookup = { lbl:i for i,lbl in enumerate(evalTree.gateLabels) } # gate labels -> ints for faster lookup
    gatereps = { i:calc._getgate(lbl).torep() for lbl,i in gate_lookup.items() }
    
    # convert to C-mode:  evaltree, gate_lookup, gatereps
    cdef c_evalTree = convert_evaltree(evalTree, gate_lookup)
    cdef DMStateCRep *c_rho = convert_rhorep(rhorep)
    cdef vector[DMGateCRep*] c_gatereps = convert_gatereps(gatereps)
    cdef vector[DMEffectCRep*] c_ereps = convert_ereps(ereps)

    # create rho_cache = vector of DMStateCReps
    #print "DB: creating rho_cache of size %d * %g GB => %g GB" % \
    #   (evalTree.cache_size(), 8.0 * c_rho._dim / 1024.0**3, evalTree.cache_size() * 8.0 * c_rho._dim / 1024.0**3)
    cdef vector[DMStateCRep*] rho_cache = create_rhocache(evalTree.cache_size(), c_rho._dim)

    #OLD cdef double[:,:] ret_view = ret
    dm_compute_pr_cache(pCache, c_evalTree, c_gatereps, c_rho, c_ereps, &rho_cache, comm)

    free_rhocache(rho_cache)  #delete cache entries
    return pCache
        


cdef dm_compute_pr_cache(double[:,:] ret,
                         vector[vector[INT]] c_evalTree,
                         vector[DMGateCRep*] c_gatereps,
                         DMStateCRep *c_rho, vector[DMEffectCRep*] c_ereps,
                         vector[DMStateCRep*]* prho_cache, comm): # any way to transmit comm?
    #Note: we need to take in rho_cache as a pointer b/c we may alter the values its
    # elements point to (instead of copying the states) - we just guarantee that in the end
    # all of the cache entries are filled with allocated (by 'new') states that the caller
    # can deallocate at will.
    cdef INT k,l,i,istart, icache
    cdef double p
    cdef INT dim = c_rho._dim
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
    for k in range(<INT>c_evalTree.size()):
        #t0 = time.time() # DEBUG
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
        #OLD for j in range(dim): prop1._dataptr[j] = init_state._dataptr[j]   NOW: a method of DMStateCRep?
        prop1.copy_from(init_state) # copy init_state -> prop1 
        #print " prop1:";  print [ prop1._dataptr[t] for t in range(4) ]
        #t1 = time.time() # DEBUG
        for l in range(3,<INT>intarray.size()): #during loop, both prop1 & prop2 are alloc'd        
            #print "begin acton %d: %.2fs since last, %.2fs elapsed" % (l-2,time.time()-t1,time.time()-t0) # DEBUG
            #t1 = time.time() #DEBUG
            c_gatereps[intarray[l]].acton(prop1,prop2)
            #print " post-act prop2:"; print [ prop2._dataptr[t] for t in range(4) ]
            tprop = prop1; prop1 = prop2; prop2 = tprop # swap prop1 <-> prop2
        final_state = prop1 # output = prop1 (after swap from loop above)
        # Note: prop2 is the other alloc'd state and this maintains invariant
        #print " final:"; print [ final_state._dataptr[t] for t in range(4) ]
        
        #print "begin prob comps: %.2fs since last, %.2fs elapsed" % (time.time()-t1, time.time()-t0) # DEBUG
        for j in range(<INT>c_ereps.size()):
            p = c_ereps[j].probability(final_state) #outcome probability
            #print("processing ",i,j,p)
            ret[i,j] = p

        if icache != -1:
            deref(prho_cache)[icache] = final_state # store this state in the cache
        else: # our 2nd state was pulled from the shelf before; return it
            shelved = final_state
            final_state = NULL
        #print "%d of %d (i=%d,istart=%d,remlen=%d): %.1fs" % (k, c_evalTree.size(), i, istart,
        #                                                      intarray.size()-3, time.time()-t0)

    #delete our temp states
    del prop2
    del shelved


    
def DM_compute_dpr_cache(calc, rholabel, elabels, evalTree, wrtSlice, comm, scratch=None):
    # can remove unused 'scratch' arg once we move hpr_cache to replibs

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
    gate_lookup = { lbl:i for i,lbl in enumerate(evalTree.gateLabels) } # gate labels -> ints for faster lookup
    gatereps = { i:calc._getgate(lbl).torep() for lbl,i in gate_lookup.items() }
    
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
            gatereps = { i:calc._getgate(lbl).torep() for lbl,i in gate_lookup.items() }
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


# Helper functions
cdef PolyRep_from_allocd_PolyCRep(PolyCRep* crep):
    cdef PolyRep ret = PolyRep.__new__(PolyRep) # doesn't call __init__
    ret.c_poly = crep
    return ret

cdef vector[vector[SVTermCRep_ptr]] sv_extract_cterms(python_termrep_lists, INT max_order):
    cdef vector[vector[SVTermCRep_ptr]] ret = vector[vector[SVTermCRep_ptr]](max_order+1)
    cdef vector[SVTermCRep*] vec_of_terms
    for order,termreps in enumerate(python_termrep_lists): # maxorder+1 lists
        vec_of_terms = vector[SVTermCRep_ptr](len(termreps))
        for i,termrep in enumerate(termreps):
            vec_of_terms[i] = (<SVTermRep?>termrep).c_term
        ret[order] = vec_of_terms
    return ret


def SV_prs_as_polys(calc, rholabel, elabels, gatestring, comm=None, memLimit=None, fastmode=True):

    # Create gatelable -> int mapping to be used throughout
    distinct_gateLabels = sorted(set(gatestring))
    glmap = { gl: i for i,gl in enumerate(distinct_gateLabels) }

    # Convert gatestring to a vector of ints
    cdef INT i
    cdef vector[INT] cgatestring
    for gl in gatestring:
        cgatestring.push_back(<INT>glmap[gl])

    cdef INT mpv = calc.Np # max_poly_vars
    cdef INT mpo = calc.max_order*2 #max_poly_order
    cdef INT order;
    cdef INT numEs = len(elabels)

    # Construct dict of gate term reps, then *convert* to c-reps, as this
    #  keeps alive the non-c-reps which keep the c-reps from being deallocated...
    gate_term_reps = { glmap[glbl]: [ [t.torep(mpo,mpv,"gate") for t in calc._getgate(glbl).get_order_terms(order)]
                                      for order in range(calc.max_order+1) ]
                       for glbl in distinct_gateLabels }

    #Similar with rho_terms and E_terms
    rho_term_reps = [ [t.torep(mpo,mpv,"prep") for t in calc.preps[rholabel].get_order_terms(order)]
                      for order in range(calc.max_order+1) ]

    E_term_reps = []
    E_indices = []
    for order in range(calc.max_order+1):
        cur_term_reps = [] # the term reps for *all* the effect vectors
        cur_indices = [] # the Evec-index corresponding to each term rep
        for i,elbl in enumerate(elabels):
            term_reps = [t.torep(mpo,mpv,"effect") for t in calc.effects[elbl].get_order_terms(order) ]
            cur_term_reps.extend( term_reps )
            cur_indices.extend( [i]*len(term_reps) )
        E_term_reps.append( cur_term_reps )
        E_indices.append( cur_indices )


    #convert to c-reps
    cdef INT gi
    cdef vector[vector[SVTermCRep_ptr]] rho_term_creps = sv_extract_cterms(rho_term_reps,calc.max_order)
    cdef vector[vector[SVTermCRep_ptr]] E_term_creps = sv_extract_cterms(E_term_reps,calc.max_order)
    cdef unordered_map[INT, vector[vector[SVTermCRep_ptr]]] gate_term_creps
    for gi,termrep_lists in gate_term_reps.items():
        gate_term_creps[gi] = sv_extract_cterms(termrep_lists,calc.max_order)

    E_cindices = vector[vector[INT]](<INT>len(E_indices))
    for ii,inds in enumerate(E_indices):
        E_cindices[ii] = vector[INT](<INT>len(inds))
        for jj,indx in enumerate(inds):
            E_cindices[ii][jj] = <INT>indx

    #Note: term calculator "dim" is the full density matrix dim
    stateDim = int(round(np.sqrt(calc.dim)))
            
    #Call C-only function (which operates with C-representations only)
    cdef vector[PolyCRep*] polys = sv_prs_as_polys(
        cgatestring, rho_term_creps, gate_term_creps, E_term_creps,
        E_cindices, numEs, calc.max_order, mpo, mpv, stateDim, <bool>fastmode)

    return [ PolyRep_from_allocd_PolyCRep(polys[i]) for i in range(<INT>polys.size()) ]


cdef vector[PolyCRep*] sv_prs_as_polys(
    vector[INT]& gatestring, vector[vector[SVTermCRep_ptr]] rho_term_reps,
    unordered_map[INT, vector[vector[SVTermCRep_ptr]]] gate_term_reps,
    vector[vector[SVTermCRep_ptr]] E_term_reps, vector[vector[INT]] E_term_indices,
    INT numEs, INT max_order, INT max_poly_order, INT max_poly_vars, INT dim, bool fastmode):

    #NOTE: gatestring and gate_terms use *integers* as gate labels, not Label objects, to speed
    # lookups and avoid weird string conversion stuff with Cython
    
    cdef INT N = len(gatestring)
    cdef INT* p = <INT*>malloc((N+2) * sizeof(INT))
    cdef INT i,j,k,order,nTerms
    cdef INT gn

    cdef sv_innerloopfn_ptr innerloop_fn;
    if fastmode:
        innerloop_fn = sv_pr_as_poly_innerloop_savepartials
    else:
        innerloop_fn = sv_pr_as_poly_innerloop

    #extract raw data from gate_terms dictionary-of-lists for faster lookup
    #gate_term_prefactors = _np.empty( (nGates,max_order+1,dim,dim)
    #cdef unordered_map[INT, vector[vector[unordered_map[INT, complex]]]] gate_term_coeffs
    #cdef vector[vector[unordered_map[INT, complex]]] rho_term_coeffs
    #cdef vector[vector[unordered_map[INT, complex]]] E_term_coeffs
    #cdef vector[vector[INT]] E_indices
                         
    cdef vector[INT]* Einds
    cdef vector[vector_SVTermCRep_ptr_ptr] factor_lists

    assert(max_order <= 2) # only support this partitioning below (so far)

    cdef vector[PolyCRep_ptr] prps = vector[PolyCRep_ptr](numEs)
    for i in range(numEs):
        prps[i] = new PolyCRep(unordered_map[INT,complex](),max_poly_order, max_poly_vars)
        # create empty polys - maybe overload constructor for this?
        # these PolyCReps are alloc'd here and returned - it is the job of the caller to
        #  free them (or assign them to new PolyRep wrapper objs)

    for order in range(max_order+1):
        #print("DB: pr_as_poly order=",order)
        
        #for p in partition_into(order, N):
        for i in range(N+2): p[i] = 0 # clear p
        factor_lists = vector[vector_SVTermCRep_ptr_ptr](N+2)
        
        if order == 0:
            #inner loop(p)
            #factor_lists = [ gate_terms[glbl][pi] for glbl,pi in zip(gatestring,p) ]
            factor_lists[0] = &rho_term_reps[p[0]]
            for k in range(N):
                gn = gatestring[k]
                factor_lists[k+1] = &gate_term_reps[gatestring[k]][p[k+1]]
                #if factor_lists[k+1].size() == 0: continue # WHAT???
            factor_lists[N+1] = &E_term_reps[p[N+1]]
            Einds = &E_term_indices[p[N+1]]
        
            #print("Part0 ",p)
            innerloop_fn(factor_lists,Einds,&prps,dim) #, prps_chk)


        elif order == 1:
            for i in range(N+2):
                p[i] = 1
                #inner loop(p)
                factor_lists[0] = &rho_term_reps[p[0]]
                for k in range(N):
                    gn = gatestring[k]
                    factor_lists[k+1] = &gate_term_reps[gn][p[k+1]]
                    #if len(factor_lists[k+1]) == 0: continue #WHAT???
                factor_lists[N+1] = &E_term_reps[p[N+1]]
                Einds = &E_term_indices[p[N+1]]

                #print "DB: Order1 "
                innerloop_fn(factor_lists,Einds,&prps,dim) #, prps_chk)
                p[i] = 0
            
        elif order == 2:
            for i in range(N+2):
                p[i] = 2
                #inner loop(p)
                factor_lists[0] = &rho_term_reps[p[0]]
                for k in range(N):
                    gn = gatestring[k]
                    factor_lists[k+1] = &gate_term_reps[gatestring[k]][p[k+1]]
                    #if len(factor_lists[k+1]) == 0: continue # WHAT???
                factor_lists[N+1] = &E_term_reps[p[N+1]]
                Einds = &E_term_indices[p[N+1]]

                innerloop_fn(factor_lists,Einds,&prps,dim) #, prps_chk)
                p[i] = 0

            for i in range(N+2):
                p[i] = 1
                for j in range(i+1,N+2):
                    p[j] = 1
                    #inner loop(p)
                    factor_lists[0] = &rho_term_reps[p[0]]
                    for k in range(N):
                        gn = gatestring[k]
                        factor_lists[k+1] = &gate_term_reps[gatestring[k]][p[k+1]]
                        #if len(factor_lists[k+1]) == 0: continue #WHAT???
                    factor_lists[N+1] = &E_term_reps[p[N+1]]
                    Einds = &E_term_indices[p[N+1]]

                    innerloop_fn(factor_lists,Einds,&prps,dim) #, prps_chk)
                    p[j] = 0
                p[i] = 0
        else:
            assert(False) # order > 2 not implemented yet...

    free(p)
    return prps

                

cdef void sv_pr_as_poly_innerloop(vector[vector_SVTermCRep_ptr_ptr] factor_lists, vector[INT]* Einds,
                                  vector[PolyCRep*]* prps, INT dim): #, prps_chk):
    #print("DB partition = ","listlens = ",[len(fl) for fl in factor_lists])

    cdef INT i,j,Ei
    cdef double complex scale, val, newval, pLeft, pRight, p

    cdef SVTermCRep* factor

    cdef INT nFactorLists = factor_lists.size() # may need to recompute this after fast-mode
    cdef INT* factorListLens = <INT*>malloc(nFactorLists * sizeof(INT))
    cdef INT last_index = nFactorLists-1

    for i in range(nFactorLists):
        factorListLens[i] = factor_lists[i].size()
        if factorListLens[i] == 0:
            free(factorListLens)
            return # nothing to loop over! - (exit before we allocate more)

    cdef PolyCRep coeff
    cdef PolyCRep result    

    cdef SVStateCRep *prop1 = new SVStateCRep(dim)
    cdef SVStateCRep *prop2 = new SVStateCRep(dim)
    cdef SVStateCRep *tprop
    cdef SVEffectCRep* EVec

    cdef INT* b = <INT*>malloc(nFactorLists * sizeof(INT))
    for i in range(nFactorLists): b[i] = 0

    assert(nFactorLists > 0), "Number of factor lists must be > 0!"
    
    #for factors in _itertools.product(*factor_lists):
    while(True):
        # In this loop, b holds "current" indices into factor_lists
        factor = deref(factor_lists[0])[b[0]] # the last factor (an Evec)
        coeff = deref(factor._coeff) # an unordered_map (copies to new "coeff" variable)
            
        for i in range(1,nFactorLists):
            coeff = coeff.mult( deref(deref(factor_lists[i])[b[i]]._coeff) )
            
        #pLeft / "pre" sim
        factor = deref(factor_lists[0])[b[0]] # 0th-factor = rhoVec
        prop1.copy_from(factor._pre_state)
        for j in range(<INT>factor._pre_ops.size()):
            factor._pre_ops[j].acton(prop1,prop2)
            tprop = prop1; prop1 = prop2; prop2 = tprop
        for i in range(1,last_index):
            factor = deref(factor_lists[i])[b[i]]
            for j in range(<INT>factor._pre_ops.size()):
                factor._pre_ops[j].acton(prop1,prop2)
                tprop = prop1; prop1 = prop2; prop2 = tprop # final state in prop1
        factor = deref(factor_lists[last_index])[b[last_index]] # the last factor (an Evec)

        # can't propagate effects, so act w/adjoint of post_ops in reverse order...
        EVec = factor._post_effect
        for j in range(<INT>factor._post_ops.size()-1,-1,-1): # (reversed)
            rhoVec = factor._post_ops[j].adjoint_acton(prop1,prop2)
            tprop = prop1; prop1 = prop2; prop2 = tprop # final state in prop1
        pLeft = EVec.amplitude(prop1)
                        
        #pRight / "post" sim
        factor = deref(factor_lists[0])[b[0]] # 0th-factor = rhoVec
        prop1.copy_from(factor._post_state)
        for j in range(<INT>factor._post_ops.size()):
            factor._post_ops[j].acton(prop1,prop2)
            tprop = prop1; prop1 = prop2; prop2 = tprop # final state in prop1
        for i in range(1,last_index):
            factor = deref(factor_lists[i])[b[i]]
            for j in range(<INT>factor._post_ops.size()):
                factor._post_ops[j].acton(prop1,prop2)
                tprop = prop1; prop1 = prop2; prop2 = tprop # final state in prop1
        factor = deref(factor_lists[last_index])[b[last_index]] # the last factor (an Evec)
        
        EVec = factor._pre_effect
        for j in range(<INT>factor._pre_ops.size()-1,-1,-1): # (reversed)
            factor._pre_ops[j].adjoint_acton(prop1,prop2)
            tprop = prop1; prop1 = prop2; prop2 = tprop # final state in prop1
        pRight = EVec.amplitude(prop1).conjugate()


        #Add result to appropriate poly
        result = coeff  # use a reference?
        result.scale(pLeft * pRight)
        final_factor_indx = b[last_index]
        Ei = deref(Einds)[final_factor_indx] #final "factor" index == E-vector index
        deref(prps)[Ei].add_inplace(result)
        
        #increment b ~ itertools.product & update vec_index_noop = _np.dot(self.multipliers, b)
        for i in range(nFactorLists-1,-1,-1):
            if b[i]+1 < factorListLens[i]:
                b[i] += 1
                break
            else:
                b[i] = 0
        else:
            break # can't increment anything - break while(True) loop

    #Clenaup: free allocated memory
    del prop1
    del prop2
    free(factorListLens)
    free(b)
    return


cdef void sv_pr_as_poly_innerloop_savepartials(vector[vector_SVTermCRep_ptr_ptr] factor_lists,
                                               vector[INT]* Einds, vector[PolyCRep*]* prps, INT dim): #, prps_chk):
    #print("DB partition = ","listlens = ",[len(fl) for fl in factor_lists])

    cdef INT i,j,Ei
    cdef double complex scale, val, newval, pLeft, pRight, p

    cdef INT incd
    cdef SVTermCRep* factor

    cdef INT nFactorLists = factor_lists.size() # may need to recompute this after fast-mode
    cdef INT* factorListLens = <INT*>malloc(nFactorLists * sizeof(INT))
    cdef INT last_index = nFactorLists-1

    for i in range(nFactorLists):
        factorListLens[i] = factor_lists[i].size()
        if factorListLens[i] == 0:
            free(factorListLens)
            return # nothing to loop over! (exit before we allocate anything else)
        
    cdef PolyCRep coeff
    cdef PolyCRep result    

    #fast mode
    cdef vector[SVStateCRep*] leftSaved = vector[SVStateCRep_ptr](nFactorLists-1)  # saved[i] is state after i-th
    cdef vector[SVStateCRep*] rightSaved = vector[SVStateCRep_ptr](nFactorLists-1) # factor has been applied
    cdef vector[PolyCRep] coeffSaved = vector[PolyCRep](nFactorLists-1)
    cdef SVStateCRep *shelved = new SVStateCRep(dim)
    cdef SVStateCRep *prop2 = new SVStateCRep(dim) # prop2 is always a temporary allocated state not owned by anything else
    cdef SVStateCRep *prop1
    cdef SVStateCRep *tprop
    cdef SVEffectCRep* EVec
    
    cdef INT* b = <INT*>malloc(nFactorLists * sizeof(INT))
    for i in range(nFactorLists): b[i] = 0
    assert(nFactorLists > 0), "Number of factor lists must be > 0!"
    
    incd = 0

    #Fill saved arrays with allocated states
    for i in range(nFactorLists-1):
        leftSaved[i] = new SVStateCRep(dim)
        rightSaved[i] = new SVStateCRep(dim)

    #for factors in _itertools.product(*factor_lists):
    #for incd,fi in incd_product(*[range(len(l)) for l in factor_lists]):
    while(True):
        # In this loop, b holds "current" indices into factor_lists
        #print "DB: iter-product BEGIN"

        if incd == 0: # need to re-evaluate rho vector
            #print "DB: re-eval at incd=0"
            factor = deref(factor_lists[0])[b[0]]

            #print "DB: re-eval left"
            prop1 = leftSaved[0] # the final destination (prop2 is already alloc'd)
            prop1.copy_from(factor._pre_state)
            for j in range(<INT>factor._pre_ops.size()):
                #print "DB: re-eval left item"
                factor._pre_ops[j].acton(prop1,prop2)
                tprop = prop1; prop1 = prop2; prop2 = tprop # swap prop1 <-> prop2
            rhoVecL = prop1
            leftSaved[0] = prop1 # final state -> saved
            # (prop2 == the other allocated state)

            #print "DB: re-eval right"
            prop1 = rightSaved[0] # the final destination (prop2 is already alloc'd)
            prop1.copy_from(factor._post_state)
            for j in range(<INT>factor._post_ops.size()):
                #print "DB: re-eval right item"
                factor._post_ops[j].acton(prop1,prop2)
                tprop = prop1; prop1 = prop2; prop2 = tprop # swap prop1 <-> prop2
            rhoVecR = prop1
            rightSaved[0] = prop1 # final state -> saved
            # (prop2 == the other allocated state)

            #print "DB: re-eval coeff"
            coeff = deref(factor._coeff)
            coeffSaved[0] = coeff
            incd += 1
        else:
            #print "DB: init from incd"
            rhoVecL = leftSaved[incd-1]
            rhoVecR = rightSaved[incd-1]
            coeff = coeffSaved[incd-1]

        # propagate left and right states, saving as we go
        for i in range(incd,last_index):
            #print "DB: propagate left begin"
            factor = deref(factor_lists[i])[b[i]]
            prop1 = leftSaved[i] # destination
            prop1.copy_from(rhoVecL) #starting state
            for j in range(<INT>factor._pre_ops.size()):
                #print "DB: propagate left item"
                factor._pre_ops[j].acton(prop1,prop2)
                tprop = prop1; prop1 = prop2; prop2 = tprop
            rhoVecL = prop1
            leftSaved[i] = prop1
            # (prop2 == the other allocated state)

            #print "DB: propagate right begin"
            prop1 = rightSaved[i] # destination
            prop1.copy_from(rhoVecR) #starting state
            for j in range(<INT>factor._post_ops.size()):
                #print "DB: propagate right item"
                factor._post_ops[j].acton(prop1,prop2)
                tprop = prop1; prop1 = prop2; prop2 = tprop
            rhoVecR = prop1
            rightSaved[i] = prop1
            # (prop2 == the other allocated state)

            #print "DB: propagate coeff mult"
            coeff = coeff.mult(deref(factor._coeff)) # copy a PolyCRep
            coeffSaved[i] = coeff

        # for the last index, no need to save, and need to construct
        # and apply effect vector
        prop1 = shelved # so now prop1 (and prop2) are alloc'd states

        #print "DB: left ampl"
        factor = deref(factor_lists[last_index])[b[last_index]] # the last factor (an Evec)
        EVec = factor._post_effect
        prop1.copy_from(rhoVecL) # initial state (prop2 already alloc'd)
        for j in range(<INT>factor._post_ops.size()-1,-1,-1): # (reversed)
            factor._post_ops[j].adjoint_acton(prop1,prop2)
            tprop = prop1; prop1 = prop2; prop2 = tprop
        pLeft = EVec.amplitude(prop1) # output in prop1, so this is final amplitude

        #print "DB: right ampl"
        EVec = factor._pre_effect
        prop1.copy_from(rhoVecR)
        for j in range(<INT>factor._pre_ops.size()-1,-1,-1): # (reversed)
            factor._pre_ops[j].adjoint_acton(prop1,prop2)
            tprop = prop1; prop1 = prop2; prop2 = tprop
        pRight = EVec.amplitude(prop1).conjugate()

        shelved = prop1 # return prop1 to the "shelf" since we'll use prop1 for other things next

        #print "DB: final block"
        #print "DB running coeff = ",dict(coeff._coeffs)
        #print "DB factor coeff = ",dict(factor._coeff._coeffs)
        result = coeff.mult(deref(factor._coeff))
        #print "DB result = ",dict(result._coeffs)
        result.scale(pLeft * pRight)
        final_factor_indx = b[last_index]
        Ei = deref(Einds)[final_factor_indx] #final "factor" index == E-vector index
        deref(prps)[Ei].add_inplace(result)
        #print "DB prps[",INT(Ei),"] = ",dict(deref(prps)[Ei]._coeffs)

        #assert(debug < 100) #DEBUG
        #print "DB: end product loop"
        
        #increment b ~ itertools.product & update vec_index_noop = _np.dot(self.multipliers, b)
        for i in range(nFactorLists-1,-1,-1):
            if b[i]+1 < factorListLens[i]:
                b[i] += 1; incd = i
                break
            else:
                b[i] = 0
        else:
            break # can't increment anything - break while(True) loop

    #Cleanup: free allocated memory
    for i in range(nFactorLists-1):
        del leftSaved[i]
        del rightSaved[i]
    del prop2
    del shelved
    free(factorListLens)
    free(b)
    return


# Stabilizer-evolution version of poly term calcs -----------------------

cdef vector[vector[SBTermCRep_ptr]] sb_extract_cterms(python_termrep_lists, INT max_order):
    cdef vector[vector[SBTermCRep_ptr]] ret = vector[vector[SBTermCRep_ptr]](max_order+1)
    cdef vector[SBTermCRep*] vec_of_terms
    for order,termreps in enumerate(python_termrep_lists): # maxorder+1 lists
        vec_of_terms = vector[SBTermCRep_ptr](len(termreps))
        for i,termrep in enumerate(termreps):
            vec_of_terms[i] = (<SBTermRep?>termrep).c_term
        ret[order] = vec_of_terms
    return ret


def SB_prs_as_polys(calc, rholabel, elabels, gatestring, comm=None, memLimit=None, fastmode=True):

    # Create gatelable -> int mapping to be used throughout
    distinct_gateLabels = sorted(set(gatestring))
    glmap = { gl: i for i,gl in enumerate(distinct_gateLabels) }

    # Convert gatestring to a vector of ints
    cdef INT i
    cdef vector[INT] cgatestring
    for gl in gatestring:
        cgatestring.push_back(<INT>glmap[gl])

    cdef INT mpv = calc.Np # max_poly_vars
    cdef INT mpo = calc.max_order*2 #max_poly_order
    cdef INT order;
    cdef INT numEs = len(elabels)
    
    # Construct dict of gate term reps, then *convert* to c-reps, as this
    #  keeps alive the non-c-reps which keep the c-reps from being deallocated...
    gate_term_reps = { glmap[glbl]: [ [t.torep(mpo,mpv,"gate") for t in calc._getgate(glbl).get_order_terms(order)]
                                      for order in range(calc.max_order+1) ]
                       for glbl in distinct_gateLabels }

    #Similar with rho_terms and E_terms
    rho_term_reps = [ [t.torep(mpo,mpv,"prep") for t in calc.preps[rholabel].get_order_terms(order)]
                      for order in range(calc.max_order+1) ]

    E_term_reps = []
    E_indices = []
    for order in range(calc.max_order+1):
        cur_term_reps = [] # the term reps for *all* the effect vectors
        cur_indices = [] # the Evec-index corresponding to each term rep
        for i,elbl in enumerate(elabels):
            term_reps = [t.torep(mpo,mpv,"effect") for t in calc.effects[elbl].get_order_terms(order) ]
            cur_term_reps.extend( term_reps )
            cur_indices.extend( [i]*len(term_reps) )
        E_term_reps.append( cur_term_reps )
        E_indices.append( cur_indices )


    #convert to c-reps
    cdef INT gi
    cdef vector[vector[SBTermCRep_ptr]] rho_term_creps = sb_extract_cterms(rho_term_reps,calc.max_order)
    cdef vector[vector[SBTermCRep_ptr]] E_term_creps = sb_extract_cterms(E_term_reps,calc.max_order)
    cdef unordered_map[INT, vector[vector[SBTermCRep_ptr]]] gate_term_creps
    for gi,termrep_lists in gate_term_reps.items():
        gate_term_creps[gi] = sb_extract_cterms(termrep_lists,calc.max_order)

    E_cindices = vector[vector[INT]](<INT>len(E_indices))
    for ii,inds in enumerate(E_indices):
        E_cindices[ii] = vector[INT](<INT>len(inds))
        for jj,indx in enumerate(inds):
            E_cindices[ii][jj] = <INT>indx

    # Assume when we calculate terms, that "dimension" of GateSet is
    # a full vectorized-density-matrix dimension, so nqubits is:
    cdef INT nqubits = <INT>(np.log2(calc.dim)//2)

    #Call C-only function (which operates with C-representations only)
    cdef vector[PolyCRep*] polys = sb_prs_as_polys(
        cgatestring, rho_term_creps, gate_term_creps, E_term_creps,
        E_cindices, numEs, calc.max_order, mpo, mpv, nqubits, <bool>fastmode)

    return [ PolyRep_from_allocd_PolyCRep(polys[i]) for i in range(<INT>polys.size()) ]


cdef vector[PolyCRep*] sb_prs_as_polys(
    vector[INT]& gatestring, vector[vector[SBTermCRep_ptr]] rho_term_reps,
    unordered_map[INT, vector[vector[SBTermCRep_ptr]]] gate_term_reps,
    vector[vector[SBTermCRep_ptr]] E_term_reps, vector[vector[INT]] E_term_indices,
    INT numEs, INT max_order, INT max_poly_order, INT max_poly_vars, INT nqubits, bool fastmode):

    #NOTE: gatestring and gate_terms use *integers* as gate labels, not Label objects, to speed
    # lookups and avoid weird string conversion stuff with Cython
    
    cdef INT N = len(gatestring)
    cdef INT* p = <INT*>malloc((N+2) * sizeof(INT))
    cdef INT i,j,k,order,nTerms
    cdef INT gn

    cdef sb_innerloopfn_ptr innerloop_fn;
    if fastmode:
        innerloop_fn = sb_pr_as_poly_innerloop_savepartials
    else:
        innerloop_fn = sb_pr_as_poly_innerloop

    #extract raw data from gate_terms dictionary-of-lists for faster lookup
    #gate_term_prefactors = _np.empty( (nGates,max_order+1,dim,dim)
    #cdef unordered_map[INT, vector[vector[unordered_map[INT, complex]]]] gate_term_coeffs
    #cdef vector[vector[unordered_map[INT, complex]]] rho_term_coeffs
    #cdef vector[vector[unordered_map[INT, complex]]] E_term_coeffs
    #cdef vector[vector[INT]] E_indices
                         
    cdef vector[INT]* Einds
    cdef vector[vector_SBTermCRep_ptr_ptr] factor_lists

    assert(max_order <= 2) # only support this partitioning below (so far)

    cdef vector[PolyCRep_ptr] prps = vector[PolyCRep_ptr](numEs)
    for i in range(numEs):
        prps[i] = new PolyCRep(unordered_map[INT,complex](),max_poly_order, max_poly_vars)
        # create empty polys - maybe overload constructor for this?
        # these PolyCReps are alloc'd here and returned - it is the job of the caller to
        #  free them (or assign them to new PolyRep wrapper objs)

    for order in range(max_order+1):
        #print "DB CYTHON: pr_as_poly order=",INT(order)
        
        #for p in partition_into(order, N):
        for i in range(N+2): p[i] = 0 # clear p
        factor_lists = vector[vector_SBTermCRep_ptr_ptr](N+2)
        
        if order == 0:
            #inner loop(p)
            #factor_lists = [ gate_terms[glbl][pi] for glbl,pi in zip(gatestring,p) ]
            factor_lists[0] = &rho_term_reps[p[0]]
            for k in range(N):
                gn = gatestring[k]
                factor_lists[k+1] = &gate_term_reps[gatestring[k]][p[k+1]]
                #if factor_lists[k+1].size() == 0: continue # WHAT???
            factor_lists[N+1] = &E_term_reps[p[N+1]]
            Einds = &E_term_indices[p[N+1]]
        
            #print "DB CYTHON: Order0"
            innerloop_fn(factor_lists,Einds,&prps,nqubits) #, prps_chk)


        elif order == 1:
            for i in range(N+2):
                p[i] = 1
                #inner loop(p)
                factor_lists[0] = &rho_term_reps[p[0]]
                for k in range(N):
                    gn = gatestring[k]
                    factor_lists[k+1] = &gate_term_reps[gn][p[k+1]]
                    #if len(factor_lists[k+1]) == 0: continue #WHAT???
                factor_lists[N+1] = &E_term_reps[p[N+1]]
                Einds = &E_term_indices[p[N+1]]

                #print "DB CYTHON: Order1 "
                innerloop_fn(factor_lists,Einds,&prps,nqubits) #, prps_chk)
                p[i] = 0
            
        elif order == 2:
            for i in range(N+2):
                p[i] = 2
                #inner loop(p)
                factor_lists[0] = &rho_term_reps[p[0]]
                for k in range(N):
                    gn = gatestring[k]
                    factor_lists[k+1] = &gate_term_reps[gatestring[k]][p[k+1]]
                    #if len(factor_lists[k+1]) == 0: continue # WHAT???
                factor_lists[N+1] = &E_term_reps[p[N+1]]
                Einds = &E_term_indices[p[N+1]]

                innerloop_fn(factor_lists,Einds,&prps,nqubits) #, prps_chk)
                p[i] = 0

            for i in range(N+2):
                p[i] = 1
                for j in range(i+1,N+2):
                    p[j] = 1
                    #inner loop(p)
                    factor_lists[0] = &rho_term_reps[p[0]]
                    for k in range(N):
                        gn = gatestring[k]
                        factor_lists[k+1] = &gate_term_reps[gatestring[k]][p[k+1]]
                        #if len(factor_lists[k+1]) == 0: continue #WHAT???
                    factor_lists[N+1] = &E_term_reps[p[N+1]]
                    Einds = &E_term_indices[p[N+1]]

                    innerloop_fn(factor_lists,Einds,&prps,nqubits) #, prps_chk)
                    p[j] = 0
                p[i] = 0
        else:
            assert(False) # order > 2 not implemented yet...

    free(p)
    return prps

                

cdef void sb_pr_as_poly_innerloop(vector[vector_SBTermCRep_ptr_ptr] factor_lists, vector[INT]* Einds,
                                  vector[PolyCRep*]* prps, INT n): #, prps_chk):
    #print("DB partition = ","listlens = ",[len(fl) for fl in factor_lists])

    cdef INT i,j,Ei
    cdef double complex scale, val, newval, pLeft, pRight, p

    cdef INT incd
    cdef SBTermCRep* factor

    cdef INT nFactorLists = factor_lists.size() # may need to recompute this after fast-mode
    cdef INT* factorListLens = <INT*>malloc(nFactorLists * sizeof(INT))
    cdef INT last_index = nFactorLists-1

    for i in range(nFactorLists):
        factorListLens[i] = factor_lists[i].size()
        if factorListLens[i] == 0:
            free(factorListLens)
            return # nothing to loop over! - (exit before we allocate more)

    cdef PolyCRep coeff
    cdef PolyCRep result    

    cdef INT namps = 1 # HARDCODED namps for SB states - in future this may be just the *initial* number
    cdef SBStateCRep *prop1 = new SBStateCRep(namps, n)
    cdef SBStateCRep *prop2 = new SBStateCRep(namps, n)
    cdef SBStateCRep *tprop
    cdef SBEffectCRep* EVec

    cdef INT* b = <INT*>malloc(nFactorLists * sizeof(INT))
    for i in range(nFactorLists): b[i] = 0

    assert(nFactorLists > 0), "Number of factor lists must be > 0!"
    
    #for factors in _itertools.product(*factor_lists):
    while(True):
        # In this loop, b holds "current" indices into factor_lists
        factor = deref(factor_lists[0])[b[0]] # the last factor (an Evec)
        coeff = deref(factor._coeff) # an unordered_map (copies to new "coeff" variable)
            
        for i in range(1,nFactorLists):
            coeff = coeff.mult( deref(deref(factor_lists[i])[b[i]]._coeff) )
            
        #pLeft / "pre" sim
        factor = deref(factor_lists[0])[b[0]] # 0th-factor = rhoVec
        prop1.copy_from(factor._pre_state)
        for j in range(<INT>factor._pre_ops.size()):
            factor._pre_ops[j].acton(prop1,prop2)
            tprop = prop1; prop1 = prop2; prop2 = tprop
        for i in range(1,last_index):
            factor = deref(factor_lists[i])[b[i]]
            for j in range(<INT>factor._pre_ops.size()):
                factor._pre_ops[j].acton(prop1,prop2)
                tprop = prop1; prop1 = prop2; prop2 = tprop # final state in prop1
        factor = deref(factor_lists[last_index])[b[last_index]] # the last factor (an Evec)

        # can't propagate effects, so act w/adjoint of post_ops in reverse order...
        EVec = factor._post_effect
        for j in range(<INT>factor._post_ops.size()-1,-1,-1): # (reversed)
            rhoVec = factor._post_ops[j].adjoint_acton(prop1,prop2)
            tprop = prop1; prop1 = prop2; prop2 = tprop # final state in prop1
        pLeft = EVec.amplitude(prop1)
                        
        #pRight / "post" sim
        factor = deref(factor_lists[0])[b[0]] # 0th-factor = rhoVec
        prop1.copy_from(factor._post_state)
        for j in range(<INT>factor._post_ops.size()):
            factor._post_ops[j].acton(prop1,prop2)
            tprop = prop1; prop1 = prop2; prop2 = tprop # final state in prop1
        for i in range(1,last_index):
            factor = deref(factor_lists[i])[b[i]]
            for j in range(<INT>factor._post_ops.size()):
                factor._post_ops[j].acton(prop1,prop2)
                tprop = prop1; prop1 = prop2; prop2 = tprop # final state in prop1
        factor = deref(factor_lists[last_index])[b[last_index]] # the last factor (an Evec)
        
        EVec = factor._pre_effect
        for j in range(<INT>factor._pre_ops.size()-1,-1,-1): # (reversed)
            factor._pre_ops[j].adjoint_acton(prop1,prop2)
            tprop = prop1; prop1 = prop2; prop2 = tprop # final state in prop1
        pRight = EVec.amplitude(prop1).conjugate()


        #Add result to appropriate poly
        result = coeff  # use a reference?
        result.scale(pLeft * pRight)
        final_factor_indx = b[last_index]
        Ei = deref(Einds)[final_factor_indx] #final "factor" index == E-vector index
        deref(prps)[Ei].add_inplace(result)
        
        #increment b ~ itertools.product & update vec_index_noop = _np.dot(self.multipliers, b)
        for i in range(nFactorLists-1,-1,-1):
            if b[i]+1 < factorListLens[i]:
                b[i] += 1
                break
            else:
                b[i] = 0
        else:
            break # can't increment anything - break while(True) loop

    #Clenaup: free allocated memory
    del prop1
    del prop2
    free(factorListLens)
    free(b)
    return


cdef void sb_pr_as_poly_innerloop_savepartials(vector[vector_SBTermCRep_ptr_ptr] factor_lists,
                                               vector[INT]* Einds, vector[PolyCRep*]* prps, INT n): #, prps_chk):
    #print("DB partition = ","listlens = ",[len(fl) for fl in factor_lists])

    cdef INT i,j,Ei
    cdef double complex scale, val, newval, pLeft, pRight, p

    cdef INT incd
    cdef SBTermCRep* factor

    cdef INT nFactorLists = factor_lists.size() # may need to recompute this after fast-mode
    cdef INT* factorListLens = <INT*>malloc(nFactorLists * sizeof(INT))
    cdef INT last_index = nFactorLists-1

    for i in range(nFactorLists):
        factorListLens[i] = factor_lists[i].size()
        if factorListLens[i] == 0:
            free(factorListLens)
            return # nothing to loop over! (exit before we allocate anything else)
        
    cdef PolyCRep coeff
    cdef PolyCRep result    

    cdef INT namps = 1 # HARDCODED namps for SB states - in future this may be just the *initial* number
    cdef vector[SBStateCRep*] leftSaved = vector[SBStateCRep_ptr](nFactorLists-1)  # saved[i] is state after i-th
    cdef vector[SBStateCRep*] rightSaved = vector[SBStateCRep_ptr](nFactorLists-1) # factor has been applied
    cdef vector[PolyCRep] coeffSaved = vector[PolyCRep](nFactorLists-1)
    cdef SBStateCRep *shelved = new SBStateCRep(namps, n)
    cdef SBStateCRep *prop2 = new SBStateCRep(namps, n) # prop2 is always a temporary allocated state not owned by anything else
    cdef SBStateCRep *prop1
    cdef SBStateCRep *tprop
    cdef SBEffectCRep* EVec
    
    cdef INT* b = <INT*>malloc(nFactorLists * sizeof(INT))
    for i in range(nFactorLists): b[i] = 0
    assert(nFactorLists > 0), "Number of factor lists must be > 0!"
    
    incd = 0

    #Fill saved arrays with allocated states
    for i in range(nFactorLists-1):
        leftSaved[i] = new SBStateCRep(namps, n)
        rightSaved[i] = new SBStateCRep(namps, n)

    #for factors in _itertools.product(*factor_lists):
    #for incd,fi in incd_product(*[range(len(l)) for l in factor_lists]):
    while(True):
        # In this loop, b holds "current" indices into factor_lists
        #print "DB: iter-product BEGIN"

        if incd == 0: # need to re-evaluate rho vector
            #print "DB: re-eval at incd=0"
            factor = deref(factor_lists[0])[b[0]]

            #print "DB: re-eval left"
            prop1 = leftSaved[0] # the final destination (prop2 is already alloc'd)
            prop1.copy_from(factor._pre_state)
            for j in range(<INT>factor._pre_ops.size()):
                #print "DB: re-eval left item"
                factor._pre_ops[j].acton(prop1,prop2)
                tprop = prop1; prop1 = prop2; prop2 = tprop # swap prop1 <-> prop2
            rhoVecL = prop1
            leftSaved[0] = prop1 # final state -> saved
            # (prop2 == the other allocated state)

            #print "DB: re-eval right"
            prop1 = rightSaved[0] # the final destination (prop2 is already alloc'd)
            prop1.copy_from(factor._post_state)
            for j in range(<INT>factor._post_ops.size()):
                #print "DB: re-eval right item"
                factor._post_ops[j].acton(prop1,prop2)
                tprop = prop1; prop1 = prop2; prop2 = tprop # swap prop1 <-> prop2
            rhoVecR = prop1
            rightSaved[0] = prop1 # final state -> saved
            # (prop2 == the other allocated state)

            #print "DB: re-eval coeff"
            coeff = deref(factor._coeff)
            coeffSaved[0] = coeff
            incd += 1
        else:
            #print "DB: init from incd"
            rhoVecL = leftSaved[incd-1]
            rhoVecR = rightSaved[incd-1]
            coeff = coeffSaved[incd-1]

        # propagate left and right states, saving as we go
        for i in range(incd,last_index):
            #print "DB: propagate left begin"
            factor = deref(factor_lists[i])[b[i]]
            prop1 = leftSaved[i] # destination
            prop1.copy_from(rhoVecL) #starting state
            for j in range(<INT>factor._pre_ops.size()):
                #print "DB: propagate left item"
                factor._pre_ops[j].acton(prop1,prop2)
                tprop = prop1; prop1 = prop2; prop2 = tprop
            rhoVecL = prop1
            leftSaved[i] = prop1
            # (prop2 == the other allocated state)

            #print "DB: propagate right begin"
            prop1 = rightSaved[i] # destination
            prop1.copy_from(rhoVecR) #starting state
            for j in range(<INT>factor._post_ops.size()):
                #print "DB: propagate right item"
                factor._post_ops[j].acton(prop1,prop2)
                tprop = prop1; prop1 = prop2; prop2 = tprop
            rhoVecR = prop1
            rightSaved[i] = prop1
            # (prop2 == the other allocated state)

            #print "DB: propagate coeff mult"
            coeff = coeff.mult(deref(factor._coeff)) # copy a PolyCRep
            coeffSaved[i] = coeff

        # for the last index, no need to save, and need to construct
        # and apply effect vector
        prop1 = shelved # so now prop1 (and prop2) are alloc'd states

        #print "DB: left ampl"
        factor = deref(factor_lists[last_index])[b[last_index]] # the last factor (an Evec)
        EVec = factor._post_effect
        prop1.copy_from(rhoVecL) # initial state (prop2 already alloc'd)
        for j in range(<INT>factor._post_ops.size()-1,-1,-1): # (reversed)
            factor._post_ops[j].adjoint_acton(prop1,prop2)
            tprop = prop1; prop1 = prop2; prop2 = tprop
        pLeft = EVec.amplitude(prop1) # output in prop1, so this is final amplitude

        #print "DB: right ampl"
        EVec = factor._pre_effect
        prop1.copy_from(rhoVecR)
        pRight = EVec.amplitude(prop1)
        #DEBUG print "  - begin: ",complex(pRight)
        for j in range(<INT>factor._pre_ops.size()-1,-1,-1): # (reversed)
            #DEBUG print " - state = ", [ prop1._smatrix[ii] for ii in range(2*2)]
            #DEBUG print "         = ", [ prop1._pvectors[ii] for ii in range(2)]
            #DEBUG print "         = ", [ prop1._amps[ii] for ii in range(1)]
            factor._pre_ops[j].adjoint_acton(prop1,prop2)
            #DEBUG print " - action with ", [ (<SBGateCRep_Clifford*>factor._pre_ops[j])._smatrix_inv[ii] for ii in range(2*2)]
            #DEBUG print " - action with ", [ (<SBGateCRep_Clifford*>factor._pre_ops[j])._svector_inv[ii] for ii in range(2)]
            #DEBUG print " - action with ", [ (<SBGateCRep_Clifford*>factor._pre_ops[j])._unitary_adj[ii] for ii in range(2*2)]
            tprop = prop1; prop1 = prop2; prop2 = tprop
            pRight = EVec.amplitude(prop1)
            #DEBUG print "  - prop ",INT(j)," = ",complex(pRight)
            #DEBUG print " - post state = ", [ prop1._smatrix[ii] for ii in range(2*2)]
            #DEBUG print "              = ", [ prop1._pvectors[ii] for ii in range(2)]
            #DEBUG print "              = ", [ prop1._amps[ii] for ii in range(1)]

        pRight = EVec.amplitude(prop1).conjugate()

        shelved = prop1 # return prop1 to the "shelf" since we'll use prop1 for other things next

        #DO THIS IN SB VARIANT
        #else: # CLIFFORD - can't propagate effects, but can act w/adjoint of post_ops in reverse order...
        #    factor = factor_lists[last_index][b[last_index]] # the last factor (an Evec)
        #    EVec = factor.post_ops[0]
        #    for j in range(len(factor.post_ops)-1,0,-1): # (reversed)
        #        rhoVecL = factor.post_ops[j].adjoint_acton(rhoVecL)
        #    #OLD: p = stabilizer_measurement_prob(rhoVecL, EVec.outcomes)
        #    #OLD: pLeft = np.sqrt(p) # sqrt b/c pLeft is just *amplitude*
        #    pLeft = rhoVecL.extract_amplitude(EVec.outcomes)
        #
        #    EVec = factor.pre_ops[0]
        #    for j in range(len(factor.pre_ops)-1,0,-1): # (reversed)
        #        rhoVecR = factor.pre_ops[j].adjoint_acton(rhoVecR)
        #    #OLD: p = stabilizer_measurement_prob(rhoVecR, EVec.outcomes)
        #    #OLD: pRight = np.sqrt(p) # sqrt b/c pRight is just *amplitude*
        #    pRight = np.conjugate(rhoVecR.extract_amplitude(EVec.outcomes))

        #print "DB: final block: pLeft=",complex(pLeft)," pRight=",complex(pRight)
        #print "DB running coeff = ",dict(coeff._coeffs)
        #print "DB factor coeff = ",dict(factor._coeff._coeffs)
        result = coeff.mult(deref(factor._coeff))
        #print "DB result = ",dict(result._coeffs)
        result.scale(pLeft * pRight)
        final_factor_indx = b[last_index]
        Ei = deref(Einds)[final_factor_indx] #final "factor" index == E-vector index
        deref(prps)[Ei].add_inplace(result)
        #print "DB prps[",INT(Ei),"] = ",dict(deref(prps)[Ei]._coeffs)

        #assert(debug < 100) #DEBUG
        #print "DB: end product loop"
        
        #increment b ~ itertools.product & update vec_index_noop = _np.dot(self.multipliers, b)
        for i in range(nFactorLists-1,-1,-1):
            if b[i]+1 < factorListLens[i]:
                b[i] += 1; incd = i
                break
            else:
                b[i] = 0
        else:
            break # can't increment anything - break while(True) loop

    #Cleanup: free allocated memory
    for i in range(nFactorLists-1):
        del leftSaved[i]
        del rightSaved[i]
    del prop2
    del shelved
    free(factorListLens)
    free(b)
    return


    
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
#    INT x
#    char * y
#    #put everything potentially necessary here
#ctypedef Gate_crep Gate_crep_t


#def fast_compute_pr_cache(rholabel, elabels, evalTree, comm, scratch=None)::
#    #needs to construct gate creps, etc...
#    
#    #calls propagate_state:
#    
#cdef propagate_state(Gate_crep_t* gate_creps, INT* gids,
#                     State_crep* state_crep):
#    for gateid in gids:
#    	actonlib[gateid](gate_creps[gateid],state_crep) # act in-place / don't require copy?
	


## You can also typedef pointers too
#
#ctypedef INT * int_ptr
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
    cdef INT N = f.shape[0]
    cdef float ret = 0.0
    cdef INT i
    for i in range(N):
        ret += f[i]*g[i]
    return ret


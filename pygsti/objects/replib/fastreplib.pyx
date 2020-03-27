# encoding: utf-8
# cython: profile=False
# cython: linetrace=False

#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************


import sys
import time as pytime
import numpy as np
from libc.stdlib cimport malloc, free
from libc.math cimport log10, sqrt, log
from libc cimport time
from libcpp cimport bool
from libcpp.vector cimport vector
from libcpp.pair cimport pair
from libcpp.algorithm cimport sort as stdsort
from libcpp.unordered_map cimport unordered_map
from cython.operator cimport dereference as deref, preincrement as inc
cimport numpy as np
cimport cython

import itertools as _itertools
from ...tools import mpitools as _mpit
from ...tools import slicetools as _slct
from ...tools import optools as _gt
from ...tools.matrixtools import _fas
from ..opcalc import fastopcalc as _fastopcalc
from scipy.sparse.linalg import LinearOperator

cdef double LARGE = 1000000000
# a large number such that LARGE is
# a very high term weight which won't help (at all) a
# path get included in the selected set of paths.

cdef double SMALL = 1e-5
# a number which is used in place of zero within the
# product of term magnitudes to keep a running path
# magnitude from being zero (and losing memory of terms).


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
        INT _nfactors
        INT _max_factor_dim

    cdef cppclass DMEffectCRep_Computational(DMEffectCRep):
        DMEffectCRep_Computational() except +
        DMEffectCRep_Computational(INT, INT, double, INT) except +
        double probability(DMStateCRep* state)
        INT _dim

    cdef cppclass DMEffectCRep_Errgen(DMEffectCRep):
        DMEffectCRep_Errgen() except +
        DMEffectCRep_Errgen(DMOpCRep*, DMEffectCRep*, INT, INT) except +
        double probability(DMStateCRep* state)
        INT _dim
        INT _errgen_id

    cdef cppclass DMOpCRep:
        DMOpCRep(INT) except +
        DMStateCRep* acton(DMStateCRep*, DMStateCRep*)
        DMStateCRep* adjoint_acton(DMStateCRep*, DMStateCRep*)
        INT _dim

    cdef cppclass DMOpCRep_Dense(DMOpCRep):
        DMOpCRep_Dense(double*,INT) except +
        DMStateCRep* acton(DMStateCRep*, DMStateCRep*)
        DMStateCRep* adjoint_acton(DMStateCRep*, DMStateCRep*)
        double* _dataptr
        INT _dim

    cdef cppclass DMOpCRep_Embedded(DMOpCRep):
        DMOpCRep_Embedded(DMOpCRep*, INT*, INT*, INT*, INT*, INT, INT, INT, INT, INT) except +
        DMStateCRep* acton(DMStateCRep*, DMStateCRep*)
        DMStateCRep* adjoint_acton(DMStateCRep*, DMStateCRep*)
        INT _nComponents
        INT _embeddedDim
        INT _iActiveBlock
        INT _nBlocks

    cdef cppclass DMOpCRep_Composed(DMOpCRep):
        DMOpCRep_Composed(vector[DMOpCRep*], INT) except +
        void reinit_factor_op_creps(vector[DMOpCRep*])
        DMStateCRep* acton(DMStateCRep*, DMStateCRep*)
        DMStateCRep* adjoint_acton(DMStateCRep*, DMStateCRep*)

    cdef cppclass DMOpCRep_Sum(DMOpCRep):
        DMOpCRep_Sum(vector[DMOpCRep*], INT) except +
        DMStateCRep* acton(DMStateCRep*, DMStateCRep*)
        DMStateCRep* adjoint_acton(DMStateCRep*, DMStateCRep*)

    cdef cppclass DMOpCRep_Exponentiated(DMOpCRep):
        DMOpCRep_Exponentiated(DMOpCRep*, INT, INT) except +
        DMStateCRep* acton(DMStateCRep*, DMStateCRep*)
        DMStateCRep* adjoint_acton(DMStateCRep*, DMStateCRep*)

    cdef cppclass DMOpCRep_Lindblad(DMOpCRep):
        DMOpCRep_Lindblad(DMOpCRep* errgen_rep,
			    double mu, double eta, INT m_star, INT s, INT dim,
			    double* unitarypost_data, INT* unitarypost_indices,
                            INT* unitarypost_indptr, INT unitarypost_nnz) except +
        DMStateCRep* acton(DMStateCRep*, DMStateCRep*)
        DMStateCRep* adjoint_acton(DMStateCRep*, DMStateCRep*)
        double _mu
        double _eta
        INT _m_star
        INT _s

    cdef cppclass DMOpCRep_Sparse(DMOpCRep):
        DMOpCRep_Sparse(double* A_data, INT* A_indices, INT* A_indptr,
                          INT nnz, INT dim) except +
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
        INT _nfactors
        INT _max_factor_dim

    cdef cppclass SVEffectCRep_Computational(SVEffectCRep):
        SVEffectCRep_Computational() except +
        SVEffectCRep_Computational(INT, INT, INT) except +
        double probability(SVStateCRep* state)
        double complex amplitude(SVStateCRep* state)
        INT _dim

    cdef cppclass SVOpCRep:
        SVOpCRep(INT) except +
        SVStateCRep* acton(SVStateCRep*, SVStateCRep*)
        SVStateCRep* adjoint_acton(SVStateCRep*, SVStateCRep*)
        INT _dim

    cdef cppclass SVOpCRep_Dense(SVOpCRep):
        SVOpCRep_Dense(double complex*,INT) except +
        SVStateCRep* acton(SVStateCRep*, SVStateCRep*)
        SVStateCRep* adjoint_acton(SVStateCRep*, SVStateCRep*)
        double complex* _dataptr
        INT _dim

    cdef cppclass SVOpCRep_Embedded(SVOpCRep):
        SVOpCRep_Embedded(SVOpCRep*, INT*, INT*, INT*, INT*, INT, INT, INT, INT, INT) except +
        SVStateCRep* acton(SVStateCRep*, SVStateCRep*)
        SVStateCRep* adjoint_acton(SVStateCRep*, SVStateCRep*)
        INT _nComponents
        INT _embeddedDim
        INT _iActiveBlock
        INT _nBlocks

    cdef cppclass SVOpCRep_Composed(SVOpCRep):
        SVOpCRep_Composed(vector[SVOpCRep*], INT) except +
        void reinit_factor_op_creps(vector[SVOpCRep*])
        SVStateCRep* acton(SVStateCRep*, SVStateCRep*)
        SVStateCRep* adjoint_acton(SVStateCRep*, SVStateCRep*)

    cdef cppclass SVOpCRep_Sum(SVOpCRep):
        SVOpCRep_Sum(vector[SVOpCRep*], INT) except +
        SVStateCRep* acton(SVStateCRep*, SVStateCRep*)
        SVStateCRep* adjoint_acton(SVStateCRep*, SVStateCRep*)

    cdef cppclass SVOpCRep_Exponentiated(SVOpCRep):
        SVOpCRep_Exponentiated(SVOpCRep*, INT, INT) except +
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

    cdef cppclass SBOpCRep:
        SBOpCRep(INT) except +
        SBStateCRep* acton(SBStateCRep*, SBStateCRep*)
        SBStateCRep* adjoint_acton(SBStateCRep*, SBStateCRep*)
        INT _n

    cdef cppclass SBOpCRep_Embedded(SBOpCRep):
        SBOpCRep_Embedded(SBOpCRep*, INT, INT*, INT) except +
        SBStateCRep* acton(SBStateCRep*, SBStateCRep*)
        SBStateCRep* adjoint_acton(SBStateCRep*, SBStateCRep*)

    cdef cppclass SBOpCRep_Composed(SBOpCRep):
        SBOpCRep_Composed(vector[SBOpCRep*], INT) except +
        SBStateCRep* acton(SBStateCRep*, SBStateCRep*)
        SBStateCRep* adjoint_acton(SBStateCRep*, SBStateCRep*)

    cdef cppclass SBOpCRep_Sum(SBOpCRep):
        SBOpCRep_Sum(vector[SBOpCRep*], INT) except +
        SBStateCRep* acton(SBStateCRep*, SBStateCRep*)
        SBStateCRep* adjoint_acton(SBStateCRep*, SBStateCRep*)

    cdef cppclass SBOpCRep_Exponentiated(SBOpCRep):
        SBOpCRep_Exponentiated(SBOpCRep*, INT, INT) except +
        SBStateCRep* acton(SBStateCRep*, SBStateCRep*)
        SBStateCRep* adjoint_acton(SBStateCRep*, SBStateCRep*)

    cdef cppclass SBOpCRep_Clifford(SBOpCRep):
        SBOpCRep_Clifford(INT*, INT*, double complex*, INT*, INT*, double complex*, INT) except +
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
    cdef cppclass PolyVarsIndex:
        PolyVarsIndex() except +
        PolyVarsIndex(INT) except +
        bool operator<(PolyVarsIndex i)
        vector[INT] _parts

    cdef cppclass PolyCRep:
        PolyCRep() except +
        PolyCRep(unordered_map[PolyVarsIndex, complex], INT, INT) except +
        PolyCRep abs()
        PolyCRep mult(PolyCRep&)
        PolyCRep abs_mult(PolyCRep&)
        void add_inplace(PolyCRep&)
        void add_abs_inplace(PolyCRep&)
        void add_scalar_to_all_coeffs_inplace(double complex offset)
        void scale(double complex scale)
        vector[INT] int_to_vinds(PolyVarsIndex indx_tup)
        unordered_map[PolyVarsIndex, complex] _coeffs
        INT _max_num_vars
        INT _vindices_per_int

    cdef cppclass SVTermCRep:
        SVTermCRep(PolyCRep*, double, double, SVStateCRep*, SVStateCRep*, vector[SVOpCRep*], vector[SVOpCRep*]) except +
        SVTermCRep(PolyCRep*, double, double, SVEffectCRep*, SVEffectCRep*, vector[SVOpCRep*], vector[SVOpCRep*]) except +
        SVTermCRep(PolyCRep*, double, double, vector[SVOpCRep*], vector[SVOpCRep*]) except +
        PolyCRep* _coeff
        double _magnitude
        double _logmagnitude
        SVStateCRep* _pre_state
        SVEffectCRep* _pre_effect
        vector[SVOpCRep*] _pre_ops
        SVStateCRep* _post_state
        SVEffectCRep* _post_effect
        vector[SVOpCRep*] _post_ops

    cdef cppclass SVTermDirectCRep:
        SVTermDirectCRep(double complex, double, double, SVStateCRep*, SVStateCRep*, vector[SVOpCRep*], vector[SVOpCRep*]) except +
        SVTermDirectCRep(double complex, double, double, SVEffectCRep*, SVEffectCRep*, vector[SVOpCRep*], vector[SVOpCRep*]) except +
        SVTermDirectCRep(double complex, double, double, vector[SVOpCRep*], vector[SVOpCRep*]) except +
        double complex _coeff
        double _magnitude
        double _logmagnitude
        SVStateCRep* _pre_state
        SVEffectCRep* _pre_effect
        vector[SVOpCRep*] _pre_ops
        SVStateCRep* _post_state
        SVEffectCRep* _post_effect
        vector[SVOpCRep*] _post_ops

    cdef cppclass SBTermCRep:
        SBTermCRep(PolyCRep*, double, double, SBStateCRep*, SBStateCRep*, vector[SBOpCRep*], vector[SBOpCRep*]) except +
        SBTermCRep(PolyCRep*, double, double, SBEffectCRep*, SBEffectCRep*, vector[SBOpCRep*], vector[SBOpCRep*]) except +
        SBTermCRep(PolyCRep*, double, double, vector[SBOpCRep*], vector[SBOpCRep*]) except +
        PolyCRep* _coeff
        double _magnitude
        double _logmagnitude
        SBStateCRep* _pre_state
        SBEffectCRep* _pre_effect
        vector[SBOpCRep*] _pre_ops
        SBStateCRep* _post_state
        SBEffectCRep* _post_effect
        vector[SBOpCRep*] _post_ops



ctypedef double complex DCOMPLEX
ctypedef DMOpCRep* DMGateCRep_ptr
ctypedef DMStateCRep* DMStateCRep_ptr
ctypedef DMEffectCRep* DMEffectCRep_ptr
ctypedef SVOpCRep* SVGateCRep_ptr
ctypedef SVStateCRep* SVStateCRep_ptr
ctypedef SVEffectCRep* SVEffectCRep_ptr
ctypedef SVTermCRep* SVTermCRep_ptr
ctypedef SVTermDirectCRep* SVTermDirectCRep_ptr
ctypedef SBOpCRep* SBGateCRep_ptr
ctypedef SBStateCRep* SBStateCRep_ptr
ctypedef SBEffectCRep* SBEffectCRep_ptr
ctypedef SBTermCRep* SBTermCRep_ptr
ctypedef PolyCRep* PolyCRep_ptr
ctypedef vector[SVTermCRep_ptr]* vector_SVTermCRep_ptr_ptr
ctypedef vector[SBTermCRep_ptr]* vector_SBTermCRep_ptr_ptr
ctypedef vector[SVTermDirectCRep_ptr]* vector_SVTermDirectCRep_ptr_ptr
ctypedef vector[INT]* vector_INT_ptr

#Create a function pointer type for term-based calc inner loop
ctypedef void (*sv_innerloopfn_ptr)(vector[vector_SVTermCRep_ptr_ptr],
                                    vector[INT]*, vector[PolyCRep*]*, INT)
ctypedef INT (*sv_innerloopfn_direct_ptr)(vector[vector_SVTermDirectCRep_ptr_ptr],
                                           vector[INT]*, vector[DCOMPLEX]*, INT, vector[double]*, double)
ctypedef void (*sb_innerloopfn_ptr)(vector[vector_SBTermCRep_ptr_ptr],
                                    vector[INT]*, vector[PolyCRep*]*, INT)
ctypedef void (*sv_addpathfn_ptr)(vector[PolyCRep*]*, vector[INT]&, INT, vector[vector_SVTermCRep_ptr_ptr]&,
                                  SVStateCRep**, SVStateCRep**, vector[INT]*,
                                  vector[SVStateCRep*]*, vector[SVStateCRep*]*, vector[PolyCRep]*)

ctypedef double (*TD_obj_fn)(double, double, double, double, double, double, double)


#cdef class StateRep:
#    pass

# Density matrix (DM) propagation wrapper classes
cdef class DMStateRep: #(StateRep):
    cdef DMStateCRep* c_state
    cdef public np.ndarray base
    #cdef double [:] data_view # alt way to hold a reference

    def __cinit__(self, np.ndarray[double, ndim=1, mode='c'] data, int reducefix=0):
        #print("PYX state constructed w/dim ",data.shape[0])
        #cdef np.ndarray[double, ndim=1, mode='c'] np_cbuf = np.ascontiguousarray(data, dtype='d') # would allow non-contig arrays
        #cdef double [:] view = data;  self.data_view = view # ALT: holds reference...
        if reducefix == 0:
            self.base = data # holds reference to data so it doesn't get garbage collected - or could copy=true
        else:
            # because serialization of numpy array flags is borked (around Numpy v1.16), we need to copy data
            # (so self.base *owns* it's data) and manually convey the writeable flag.
            self.base = np.require(data.copy(), requirements=['OWNDATA', 'C_CONTIGUOUS'])
            self.base.flags.writeable = True if reducefix == 1 else False
        #self.c_state = new DMStateCRep(<double*>np_cbuf.data,<INT>np_cbuf.shape[0],<bool>0)
        self.c_state = new DMStateCRep(<double*>self.base.data,<INT>self.base.shape[0],<bool>0)

    def __reduce__(self):
        reducefix = 1 if self.base.flags.writeable else 2
        return (DMStateRep, (self.base, reducefix))

    def todense(self):
        return self.base

    @property
    def dim(self):
        return self.c_state._dim

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

    def __reduce__(self):
        return (DMEffectRep, ())

    @property
    def dim(self):
        return self.c_effect._dim

    def probability(self, DMStateRep state not None):
        #unnecessary (just put in signature): cdef DMStateRep st = <DMStateRep?>state
        return self.c_effect.probability(state.c_state)


cdef class DMEffectRepDense(DMEffectRep):
    cdef public np.ndarray base

    def __cinit__(self, np.ndarray[double, ndim=1, mode='c'] data, int reducefix=0):
        if reducefix == 0:
            self.base = data  # holds reference to data
        else:
            # because serialization of numpy array flags is borked (around Numpy v1.16), we need to copy data
            # (so self.base *owns* it's data) and manually convey the writeable flag.
            self.base = np.require(data.copy(), requirements=['OWNDATA', 'C_CONTIGUOUS'])
            self.base.flags.writeable = True if reducefix == 1 else False
        self.c_effect = new DMEffectCRep_Dense(<double*>self.base.data,
                                               <INT>self.base.shape[0])

    def __str__(self):
        return str([ (<DMEffectCRep_Dense*>self.c_effect)._dataptr[i] for i in range(self.c_effect._dim)])

    def __reduce__(self):
        reducefix = 1 if self.base.flags.writeable else 2
        return (DMEffectRepDense, (self.base, reducefix))


cdef class DMEffectRepTensorProd(DMEffectRep):
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

    def __reduce__(self):
        return (DMEffectRepTensorProd,
                (self.data_ref1, self.data_ref2,
                 (<DMEffectCRep_TensorProd*>self.c_effect)._nfactors,
                 (<DMEffectCRep_TensorProd*>self.c_effect)._max_factor_dim,
                 self.c_effect._dim
                ))


cdef class DMEffectRepComputational(DMEffectRep):
    cdef public np.ndarray zvals

    def __cinit__(self, np.ndarray[np.int64_t, ndim=1, mode='c'] zvals, INT dim):
        # cdef INT dim = 4**zvals.shape[0] -- just send as argument
        cdef INT nfactors = zvals.shape[0]
        cdef double abs_elval = 1/(np.sqrt(2)**nfactors)
        cdef INT base = 1
        cdef INT zvals_int = 0
        for i in range(nfactors):
            zvals_int += base * zvals[i]
            base = base << 1 # *= 2
        self.zvals = zvals
        self.c_effect = new DMEffectCRep_Computational(nfactors, zvals_int, abs_elval, dim)

    def __reduce__(self):
        return (DMEffectRepComputational, (self.zvals, self.c_effect._dim))


cdef class DMEffectRepErrgen(DMEffectRep):  #TODO!! Need to make SV version
    cdef public DMOpRep errgen_rep
    cdef public DMEffectRep effect_rep

    def __cinit__(self, DMOpRep errgen_oprep not None, DMEffectRep effect_rep not None, errgen_id):
        cdef INT dim = effect_rep.c_effect._dim
        self.errgen_rep = errgen_oprep
        self.effect_rep = effect_rep
        self.c_effect = new DMEffectCRep_Errgen(errgen_oprep.c_gate,
                                                effect_rep.c_effect,
                                                <INT>errgen_id, dim)

    def __reduce__(self):
        return (DMEffectRepErrgen, (self.errgen_rep, self.effect_rep,
                                     (<DMEffectCRep_Errgen*>self.c_effect)._errgen_id))


cdef class DMOpRep:
    cdef DMOpCRep* c_gate

    def __cinit__(self):
        pass # self.c_gate = NULL ?

    def __reduce__(self):
        return (DMOpRep, ())

    def __dealloc__(self):
        del self.c_gate

    @property
    def dim(self):
        return self.c_gate._dim

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

    def aslinearoperator(self):
        def mv(v):
            if v.ndim == 2 and v.shape[1] == 1: v = v[:,0]
            in_state = DMStateRep(np.ascontiguousarray(v,'d'))
            return self.acton(in_state).todense()
        def rmv(v):
            if v.ndim == 2 and v.shape[1] == 1: v = v[:,0]
            in_state = DMStateRep(np.ascontiguousarray(v,'d'))
            return self.adjoint_acton(in_state).todense()
        dim = self.c_gate._dim
        return LinearOperator((dim,dim), matvec=mv, rmatvec=rmv) # transpose, adjoint, dot, matmat?



cdef class DMOpRepDense(DMOpRep):
    cdef public np.ndarray base

    def __cinit__(self, np.ndarray[double, ndim=2, mode='c'] data, int reducefix=0):
        if reducefix == 0:
            self.base = data  # the usual case - just take data ptr
        else:
            # because serialization of numpy array flags is borked (around Numpy v1.16), we need to copy data
            # (so self.base *owns* it's data) and manually convey the writeable flag.
            self.base = np.require(data.copy(), requirements=['OWNDATA', 'C_CONTIGUOUS'])
            self.base.flags.writeable = True if reducefix == 1 else False

        #print("PYX dense gate constructed w/dim ",data.shape[0])
        self.c_gate = new DMOpCRep_Dense(<double*>self.base.data,
                                           <INT>self.base.shape[0])

    def __reduce__(self):
        reducefix = 1 if self.base.flags.writeable else 2
        return (DMOpRepDense, (self.base, reducefix))

    def __str__(self):
        s = ""
        cdef DMOpCRep_Dense* my_cgate = <DMOpCRep_Dense*>self.c_gate # b/c we know it's a _Dense gate...
        cdef INT i,j,k
        for i in range(my_cgate._dim):
            k = i*my_cgate._dim
            for j in range(my_cgate._dim):
                s += str(my_cgate._dataptr[k+j]) + " "
            s += "\n"
        return s


cdef class DMOpRepEmbedded(DMOpRep):
    cdef np.ndarray data_ref1
    cdef np.ndarray data_ref2
    cdef np.ndarray data_ref3
    cdef np.ndarray data_ref4
    cdef np.ndarray data_ref5
    cdef np.ndarray data_ref6
    cdef public DMOpRep embedded

    def __cinit__(self, DMOpRep embedded_op,
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
            tmp[i] = numBasisEls[nComponentsInActiveBlock-i]
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
        for ii,op_b in enumerate(_itertools.product(*basisInds_action)):
            vec_index = offset
            for j,bInd in zip(actionInds,op_b):
                vec_index += multipliers[j]*bInd
            baseinds[ii] = vec_index

        self.data_ref1 = noop_incrementers
        self.data_ref2 = numBasisEls_noop_blankaction
        self.data_ref3 = baseinds
        self.data_ref4 = blocksizes
        self.data_ref5 = numBasisEls
        self.data_ref6 = actionInds
        self.embedded = embedded_op # needed to prevent garbage collection?
        self.c_gate = new DMOpCRep_Embedded(embedded_op.c_gate,
                                              <INT*>noop_incrementers.data, <INT*>numBasisEls_noop_blankaction.data,
                                              <INT*>baseinds.data, <INT*>blocksizes.data,
                                              embedded_dim, nComponentsInActiveBlock,
                                              iActiveBlock, nBlocks, dim)

    def __reduce__(self):
        return (DMOpRepEmbedded, (self.embedded,
                                   self.data_ref5, self.data_ref6, self.data_ref4,
                                   (<DMOpCRep_Embedded*>self.c_gate)._embeddedDim,
                                   (<DMOpCRep_Embedded*>self.c_gate)._nComponents,
                                   (<DMOpCRep_Embedded*>self.c_gate)._iActiveBlock,
                                   (<DMOpCRep_Embedded*>self.c_gate)._nBlocks,
                                   self.c_gate._dim))


cdef class DMOpRepComposed(DMOpRep):
    cdef public object factor_reps # list of DMOpRep objs?

    def __cinit__(self, factor_op_reps, INT dim):
        self.factor_reps = factor_op_reps
        cdef INT i
        cdef INT nfactors = len(factor_op_reps)
        cdef vector[DMOpCRep*] gate_creps = vector[DMGateCRep_ptr](nfactors)
        for i in range(nfactors):
            gate_creps[i] = (<DMOpRep?>factor_op_reps[i]).c_gate
        self.c_gate = new DMOpCRep_Composed(gate_creps, dim)

    def __reduce__(self):
        return (DMOpRepComposed, (self.factor_reps, self.c_gate._dim))

    def reinit_factor_op_reps(self, new_factor_op_reps):
        cdef INT i
        cdef INT nfactors = len(new_factor_op_reps)
        cdef vector[DMOpCRep*] creps = vector[DMGateCRep_ptr](nfactors)
        for i in range(nfactors):
            creps[i] = (<DMOpRep?>new_factor_op_reps[i]).c_gate
        (<DMOpCRep_Composed*>self.c_gate).reinit_factor_op_creps(creps)


cdef class DMOpRepSum(DMOpRep):
    cdef public object factor_reps # list of DMOpRep objs?

    def __cinit__(self, factor_reps, INT dim):
        self.factor_reps = factor_reps
        cdef INT i
        cdef INT nfactors = len(factor_reps)
        cdef vector[DMOpCRep*] factor_creps = vector[DMGateCRep_ptr](nfactors)
        for i in range(nfactors):
            factor_creps[i] = (<DMOpRep?>factor_reps[i]).c_gate
        self.c_gate = new DMOpCRep_Sum(factor_creps, dim)

    def __reduce__(self):
        return (DMOpRepSum, (self.factor_reps, self.c_gate._dim))


cdef class DMOpRepExponentiated(DMOpRep):
    cdef public DMOpRep exponentiated_op
    cdef public INT power

    def __cinit__(self, DMOpRep exponentiated_op_rep, INT power, INT dim):
        self.exponentiated_op = exponentiated_op_rep
        self.power = power
        self.c_gate = new DMOpCRep_Exponentiated(exponentiated_op_rep.c_gate, power, dim)

    def __reduce__(self):
        return (DMOpRepExponentiated, (self.exponentiated_op, self.power, self.c_gate._dim))


cdef class DMOpRepLindblad(DMOpRep):
    cdef public object errgen_rep
    cdef np.ndarray data_ref2
    cdef np.ndarray data_ref3
    cdef np.ndarray data_ref4

    def __cinit__(self, errgen_rep,
                  double mu, double eta, INT m_star, INT s,
                  np.ndarray[double, ndim=1, mode='c'] unitarypost_data,
                  np.ndarray[np.int64_t, ndim=1, mode='c'] unitarypost_indices,
                  np.ndarray[np.int64_t, ndim=1, mode='c'] unitarypost_indptr):
        self.errgen_rep = errgen_rep
        self.data_ref2 = unitarypost_data
        self.data_ref3 = unitarypost_indices
        self.data_ref4 = unitarypost_indptr
        cdef INT dim = errgen_rep.dim
        cdef INT upost_nnz = unitarypost_data.shape[0]
        self.c_gate = new DMOpCRep_Lindblad((<DMOpRep?>errgen_rep).c_gate,
                                              mu, eta, m_star, s, dim,
                                              <double*>unitarypost_data.data,
                                              <INT*>unitarypost_indices.data,
                                              <INT*>unitarypost_indptr.data, upost_nnz)

    def set_exp_params(self, double mu, double eta, INT m_star, INT s):
        (<DMOpCRep_Lindblad*>self.c_gate)._mu = mu
        (<DMOpCRep_Lindblad*>self.c_gate)._eta = eta
        (<DMOpCRep_Lindblad*>self.c_gate)._m_star = m_star
        (<DMOpCRep_Lindblad*>self.c_gate)._s = s

    def get_exp_params(self):
        return ( (<DMOpCRep_Lindblad*>self.c_gate)._mu,
                 (<DMOpCRep_Lindblad*>self.c_gate)._eta,
                 (<DMOpCRep_Lindblad*>self.c_gate)._m_star,
                 (<DMOpCRep_Lindblad*>self.c_gate)._s)

    def __reduce__(self):
        return (DMOpRepLindblad, (self.errgen_rep,
                                   (<DMOpCRep_Lindblad*>self.c_gate)._mu,
                                   (<DMOpCRep_Lindblad*>self.c_gate)._eta,
                                   (<DMOpCRep_Lindblad*>self.c_gate)._m_star,
                                   (<DMOpCRep_Lindblad*>self.c_gate)._s,
                                   self.data_ref2, self.data_ref3, self.data_ref4))


cdef class DMOpRepSparse(DMOpRep):
    cdef public np.ndarray data
    cdef public np.ndarray indices
    cdef public np.ndarray indptr

    def __cinit__(self, np.ndarray[double, ndim=1, mode='c'] A_data,
                  np.ndarray[np.int64_t, ndim=1, mode='c'] A_indices,
                  np.ndarray[np.int64_t, ndim=1, mode='c'] A_indptr):
        self.data = A_data
        self.indices = A_indices
        self.indptr = A_indptr
        cdef INT nnz = A_data.shape[0]
        cdef INT dim = A_indptr.shape[0]-1
        self.c_gate = new DMOpCRep_Sparse(<double*>A_data.data, <INT*>A_indices.data,
                                             <INT*>A_indptr.data, nnz, dim);

    def __reduce__(self):
        return (DMOpRepSparse, (self.data, self.indices, self.indptr))


# State vector (SV) propagation wrapper classes
cdef class SVStateRep: #(StateRep):
    cdef SVStateCRep* c_state
    cdef public np.ndarray base

    def __cinit__(self, np.ndarray[np.complex128_t, ndim=1, mode='c'] data, int reducefix=0):
        if reducefix == 0:
            self.base = data # holds reference to data so it doesn't get garbage collected - or could copy=true
        else:
            # because serialization of numpy array flags is borked (around Numpy v1.16), we need to copy data
            # (so self.base *owns* it's data) and manually convey the writeable flag.
            self.base = np.require(data.copy(), requirements=['OWNDATA', 'C_CONTIGUOUS'])
            self.base.flags.writeable = True if reducefix == 1 else False
        self.c_state = new SVStateCRep(<double complex*>self.base.data,<INT>self.base.shape[0],<bool>0)

    def __reduce__(self):
        reducefix = 1 if self.base.flags.writeable else 2
        return (SVStateRep, (self.base, reducefix))

    @property
    def dim(self):
        return self.c_state._dim

    def todense(self):
        return self.base

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

    def __reduce__(self):
        return (SVEffectRep, ())

    def probability(self, SVStateRep state not None):
        #unnecessary (just put in signature): cdef SVStateRep st = <SVStateRep?>state
        return self.c_effect.probability(state.c_state)

    def amplitude(self, SVStateRep state not None):
        return self.c_effect.amplitude(state.c_state)

    @property
    def dim(self):
        return self.c_effect._dim


cdef class SVEffectRepDense(SVEffectRep):
    cdef public np.ndarray base

    def __cinit__(self, np.ndarray[np.complex128_t, ndim=1, mode='c'] data, int reducefix=0):
        if reducefix == 0:
            self.base = data # holds reference to data
        else:
            # because serialization of numpy array flags is borked (around Numpy v1.16), we need to copy data
            # (so self.base *owns* it's data) and manually convey the writeable flag.
            self.base = np.require(data.copy(), requirements=['OWNDATA', 'C_CONTIGUOUS'])
            self.base.flags.writeable = True if reducefix == 1 else False
        self.c_effect = new SVEffectCRep_Dense(<double complex*>self.base.data,
                                               <INT>self.base.shape[0])

    def __reduce__(self):
        reducefix = 1 if self.base.flags.writeable else 2
        return (SVEffectRepDense, (self.base, reducefix))


cdef class SVEffectRepTensorProd(SVEffectRep):
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

    def __reduce__(self):
        return (SVEffectRepTensorProd, (self.data_ref1, self.data_ref2,
                                         (<SVEffectCRep_TensorProd*>self.c_effect)._nfactors,
                                         (<SVEffectCRep_TensorProd*>self.c_effect)._max_factor_dim,
                                         self.c_effect._dim))


cdef class SVEffectRepComputational(SVEffectRep):
    cdef public np.ndarray zvals;

    def __cinit__(self, np.ndarray[np.int64_t, ndim=1, mode='c'] zvals, INT dim):
        # cdef INT dim = 2**zvals.shape[0] -- just send as argument
        cdef INT nfactors = zvals.shape[0]
        cdef double abs_elval = 1/(np.sqrt(2)**nfactors)
        cdef INT base = 1
        cdef INT zvals_int = 0
        for i in range(nfactors):
            zvals_int += base * zvals[i]
            base = base << 1 # *= 2
        self.zvals = zvals
        self.c_effect = new SVEffectCRep_Computational(nfactors, zvals_int, dim)

    def __reduce__(self):
        return (SVEffectRepComputational, (self.zvals, self.c_effect._dim))


cdef class SVOpRep:
    cdef SVOpCRep* c_gate

    def __cinit__(self):
        pass # self.c_gate = NULL ?

    def __reduce__(self):
        return (SVOpRep, ())

    def __dealloc__(self):
        del self.c_gate

    def acton(self, SVStateRep state not None):
        cdef SVStateRep out_state = SVStateRep(np.empty(self.c_gate._dim, dtype=np.complex128))
        #print("PYX acton called w/dim ", self.c_gate._dim, out_state.c_state._dim)
        # assert(state.c_state._dataptr != out_state.c_state._dataptr) # DEBUG
        self.c_gate.acton(state.c_state, out_state.c_state)
        return out_state

    #FUTURE: adjoint acton

    @property
    def dim(self):
        return self.c_gate._dim


cdef class SVOpRepDense(SVOpRep):
    cdef public np.ndarray base

    def __cinit__(self, np.ndarray[np.complex128_t, ndim=2, mode='c'] data, int reducefix=0):
        if reducefix == 0:
            self.base = data  # the usual case - just take data ptr
        else:
            # because serialization of numpy array flags is borked (around Numpy v1.16), we need to copy data
            # (so self.base *owns* it's data) and manually convey the writeable flag.
            self.base = np.require(data.copy(), requirements=['OWNDATA', 'C_CONTIGUOUS'])
            self.base.flags.writeable = True if reducefix == 1 else False
        #print("PYX dense gate constructed w/dim ",data.shape[0])
        self.c_gate = new SVOpCRep_Dense(<double complex*>self.base.data,
                                           <INT>self.base.shape[0])

    def __reduce__(self):
        reducefix = 1 if self.base.flags.writeable else 2
        return (SVOpRepDense, (self.base, reducefix))

    def __str__(self):
        s = ""
        cdef SVOpCRep_Dense* my_cgate = <SVOpCRep_Dense*>self.c_gate # b/c we know it's a _Dense gate...
        cdef INT i,j,k
        for i in range(my_cgate._dim):
            k = i*my_cgate._dim
            for j in range(my_cgate._dim):
                s += str(my_cgate._dataptr[k+j]) + " "
            s += "\n"
        return s


cdef class SVOpRepEmbedded(SVOpRep):
    cdef np.ndarray data_ref1
    cdef np.ndarray data_ref2
    cdef np.ndarray data_ref3
    cdef np.ndarray data_ref4
    cdef np.ndarray data_ref5
    cdef np.ndarray data_ref6
    cdef public SVOpRep embedded

    def __cinit__(self, SVOpRep embedded_op,
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
            tmp[i] = numBasisEls[nComponentsInActiveBlock-i]
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
        for ii,op_b in enumerate(_itertools.product(*basisInds_action)):
            vec_index = offset
            for j,bInd in zip(actionInds,op_b):
                vec_index += multipliers[j]*bInd
            baseinds[ii] = vec_index

        self.data_ref1 = noop_incrementers
        self.data_ref2 = numBasisEls_noop_blankaction
        self.data_ref3 = baseinds
        self.data_ref4 = blocksizes
        self.data_ref5 = numBasisEls
        self.data_ref6 = actionInds
        self.embedded = embedded_op # needed to prevent garbage collection?
        self.c_gate = new SVOpCRep_Embedded(embedded_op.c_gate,
                                              <INT*>noop_incrementers.data, <INT*>numBasisEls_noop_blankaction.data,
                                              <INT*>baseinds.data, <INT*>blocksizes.data,
                                              embedded_dim, nComponentsInActiveBlock,
                                              iActiveBlock, nBlocks, dim)

    def __reduce__(self):
        return (SVOpRepEmbedded, (self.embedded,
                                   self.data_ref5, self.data_ref6, self.data_ref4,
                                   (<SVOpCRep_Embedded*>self.c_gate)._embeddedDim,
                                   (<SVOpCRep_Embedded*>self.c_gate)._nComponents,
                                   (<SVOpCRep_Embedded*>self.c_gate)._iActiveBlock,
                                   (<SVOpCRep_Embedded*>self.c_gate)._nBlocks,
                                   self.c_gate._dim))


cdef class SVOpRepComposed(SVOpRep):
    cdef public object factor_reps # list of SVOpRep objs?

    def __cinit__(self, factor_op_reps, INT dim):
        self.factor_reps = factor_op_reps
        cdef INT i
        cdef INT nfactors = len(factor_op_reps)
        cdef vector[SVOpCRep*] gate_creps = vector[SVGateCRep_ptr](nfactors)
        for i in range(nfactors):
            gate_creps[i] = (<SVOpRep?>factor_op_reps[i]).c_gate
        self.c_gate = new SVOpCRep_Composed(gate_creps, dim)

    def reinit_factor_op_reps(self, new_factor_op_reps):
        cdef INT i
        cdef INT nfactors = len(new_factor_op_reps)
        cdef vector[SVOpCRep*] creps = vector[SVGateCRep_ptr](nfactors)
        for i in range(nfactors):
            creps[i] = (<SVOpRep?>new_factor_op_reps[i]).c_gate
        (<SVOpCRep_Composed*>self.c_gate).reinit_factor_op_creps(creps)

    def __reduce__(self):
        return (SVOpRepComposed, (self.factor_reps, self.c_gate._dim))


cdef class SVOpRepSum(SVOpRep):
    cdef public object factor_reps # list of SVOpRep objs?

    def __cinit__(self, factor_reps, INT dim):
        self.factor_reps = factor_reps
        cdef INT i
        cdef INT nfactors = len(factor_reps)
        cdef vector[SVOpCRep*] factor_creps = vector[SVGateCRep_ptr](nfactors)
        for i in range(nfactors):
            factor_creps[i] = (<SVOpRep?>factor_reps[i]).c_gate
        self.c_gate = new SVOpCRep_Sum(factor_creps, dim)

    def __reduce__(self):
        return (SVOpRepSum, (self.factor_reps, self.c_gate._dim))


cdef class SVOpRepExponentiated(SVOpRep):
    cdef public SVOpRep exponentiated_op
    cdef public INT power

    def __cinit__(self, SVOpRep exponentiated_op_rep, INT power, INT dim):
        self.exponentiated_op = exponentiated_op_rep
        self.power = power
        self.c_gate = new SVOpCRep_Exponentiated(exponentiated_op_rep.c_gate, power, dim)

    def __reduce__(self):
        return (SVOpRepExponentiated, (self.exponentiated_op, self.power, self.c_gate._dim))


# Stabilizer state (SB) propagation wrapper classes
cdef class SBStateRep: #(StateRep):
    cdef SBStateCRep* c_state
    cdef public np.ndarray smatrix
    cdef public np.ndarray pvectors
    cdef public np.ndarray amps

    def __cinit__(self, np.ndarray[np.int64_t, ndim=2, mode='c'] smatrix,
                  np.ndarray[np.int64_t, ndim=2, mode='c'] pvectors,
                  np.ndarray[np.complex128_t, ndim=1, mode='c'] amps):
        self.smatrix = smatrix
        self.pvectors = pvectors
        self.amps = amps
        cdef INT namps = amps.shape[0]
        cdef INT n = smatrix.shape[0] // 2
        self.c_state = new SBStateCRep(<INT*>smatrix.data,<INT*>pvectors.data,
                                       <double complex*>amps.data, namps, n)

    def __reduce__(self):
        return (SBStateRep, (self.smatrix, self.pvectors, self.amps))

    @property
    def nqubits(self):
        return self.c_state._n

    @property
    def dim(self):
        return 2**(self.c_state._n) # assume "unitary evolution"-type mode

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
    cdef public np.ndarray zvals

    def __cinit__(self, np.ndarray[np.int64_t, ndim=1, mode='c'] zvals):
        self.zvals = zvals
        self.c_effect = new SBEffectCRep(<INT*>zvals.data,
                                         <INT>zvals.shape[0])

    def __reduce__(self):
        return (SBEffectRep, (self.zvals,))

    def __dealloc__(self):
        del self.c_effect # check for NULL?

    @property
    def nqubits(self):
        return self.c_effect._n

    @property
    def dim(self):
        return 2**(self.c_effect._n)  # assume "unitary evolution"-type mode

    def probability(self, SBStateRep state not None):
        #unnecessary (just put in signature): cdef SBStateRep st = <SBStateRep?>state
        return self.c_effect.probability(state.c_state)

    def amplitude(self, SBStateRep state not None):
        return self.c_effect.amplitude(state.c_state)



cdef class SBOpRep:
    cdef SBOpCRep* c_gate

    def __cinit__(self):
        pass # self.c_gate = NULL ?

    def __reduce__(self):
        return (SBOpRep, ())

    def __dealloc__(self):
        del self.c_gate

    @property
    def nqubits(self):
        return self.c_gate._n

    @property
    def dim(self):
        return 2**(self.c_gate._n)  # assume "unitary evolution"-type mode

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


cdef class SBOpRepEmbedded(SBOpRep):
    cdef public np.ndarray qubits
    cdef public SBOpRep embedded

    def __cinit__(self, SBOpRep embedded_op, INT n,
                  np.ndarray[np.int64_t, ndim=1, mode='c'] qubits):
        self.qubits = qubits
        self.embedded = embedded_op # needed to prevent garbage collection?
        self.c_gate = new SBOpCRep_Embedded(embedded_op.c_gate, n,
                                              <INT*>qubits.data, <INT>qubits.shape[0])

    def __reduce__(self):
        return (SBOpRepEmbedded, (self.embedded, self.c_gate._n, self.qubits))


cdef class SBOpRepComposed(SBOpRep):
    cdef public object factor_reps # list of SBOpRep objs?

    def __cinit__(self, factor_op_reps, INT n):
        self.factor_reps = factor_op_reps
        cdef INT i
        cdef INT nfactors = len(factor_op_reps)
        cdef vector[SBOpCRep*] gate_creps = vector[SBGateCRep_ptr](nfactors)
        for i in range(nfactors):
            gate_creps[i] = (<SBOpRep?>factor_op_reps[i]).c_gate
        self.c_gate = new SBOpCRep_Composed(gate_creps, n)

    def __reduce__(self):
        return (SBOpRepComposed, (self.factor_reps, self.c_gate._n))


cdef class SBOpRepSum(SBOpRep):
    cdef public object factor_reps # list of SBOpRep objs?

    def __cinit__(self, factor_reps, INT n):
        self.factor_reps = factor_reps
        cdef INT i
        cdef INT nfactors = len(factor_reps)
        cdef vector[SBOpCRep*] factor_creps = vector[SBGateCRep_ptr](nfactors)
        for i in range(nfactors):
            factor_creps[i] = (<SBOpRep?>factor_reps[i]).c_gate
        self.c_gate = new SBOpCRep_Sum(factor_creps, n)

    def __reduce__(self):
        return (SBOpRepSum, (self.factor_reps, self.c_gate._n))


cdef class SBOpRepExponentiated(SBOpRep):
    cdef public SBOpRep exponentiated_op
    cdef public INT power

    def __cinit__(self, SBOpRep exponentiated_op_rep, INT power, INT n):
        self.exponentiated_op = exponentiated_op_rep
        self.power = power
        self.c_gate = new SBOpCRep_Exponentiated(exponentiated_op_rep.c_gate, power, n)

    def __reduce__(self):
        return (SBOpRepExponentiated, (self.exponentiated_op, self.power, self.c_gate._n))


cdef class SBOpRepClifford(SBOpRep):
    cdef public np.ndarray smatrix
    cdef public np.ndarray svector
    cdef public np.ndarray unitary
    cdef public np.ndarray smatrix_inv
    cdef public np.ndarray svector_inv
    cdef public np.ndarray unitary_dagger

    def __cinit__(self, np.ndarray[np.int64_t, ndim=2, mode='c'] smatrix,
                  np.ndarray[np.int64_t, ndim=1, mode='c'] svector,
                  np.ndarray[np.int64_t, ndim=2, mode='c'] smatrix_inv,
                  np.ndarray[np.int64_t, ndim=1, mode='c'] svector_inv,
                  np.ndarray[np.complex128_t, ndim=2, mode='c'] unitary):

        self.smatrix = smatrix
        self.svector = svector
        self.unitary = unitary
        self.smatrix_inv = smatrix_inv
        self.svector_inv = svector_inv
        self.unitary_dagger = np.ascontiguousarray(np.conjugate(np.transpose(unitary)))
           # the "ascontiguousarray" is crucial, since we just use the .data below
        cdef INT n = smatrix.shape[0] // 2
        self.c_gate = new SBOpCRep_Clifford(<INT*>smatrix.data, <INT*>svector.data, <double complex*>unitary.data,
                                              <INT*>smatrix_inv.data, <INT*>svector_inv.data,
                                              <double complex*>self.unitary_dagger.data, n)

    def __reduce__(self):
        return (SBOpRepClifford, (self.smatrix, self.svector, self.smatrix_inv, self.svector_inv, self.unitary))


# Other classes
cdef class PolyRep:
    cdef PolyCRep* c_poly

    #Use normal init here so can bypass to create from an already alloc'd c_poly
    def __init__(self, int_coeff_dict, INT max_num_vars, INT vindices_per_int):
        cdef unordered_map[PolyVarsIndex, complex] coeffs
        cdef PolyVarsIndex indx
        for i_tup,c in int_coeff_dict.items():
            indx = PolyVarsIndex(len(i_tup))
            for ii,i in enumerate(i_tup):
                indx._parts[ii] = i
            coeffs[indx] = <double complex>c
        self.c_poly = new PolyCRep(coeffs, max_num_vars, vindices_per_int)

    def __reduce__(self):
        return (PolyRep, (self.int_coeffs, self.max_num_vars, self.vindices_per_int))

    def __dealloc__(self):
        del self.c_poly

    #TODO: REMOVE this function when things are working again
    def reinit(self, int_coeff_dict):
        #Very similar to init, but same max_num_vars

        #Store these before deleting self.c_poly!
        cdef INT max_num_vars = self.c_poly._max_num_vars
        cdef INT vindices_per_int = self.c_poly._vindices_per_int
        del self.c_poly

        cdef unordered_map[PolyVarsIndex, complex] coeffs
        cdef PolyVarsIndex indx
        for i_tup,c in int_coeff_dict.items():
            indx = PolyVarsIndex(len(i_tup))
            for ii,i in enumerate(i_tup):
                indx._parts[ii] = i
            coeffs[indx] = <double complex>c
        self.c_poly = new PolyCRep(coeffs, max_num_vars, vindices_per_int)

    def mapvec_indices_inplace(self, np.ndarray[np.int64_t, ndim=1, mode='c'] mapfn_as_vector):
        cdef INT* mapfv = <INT*> mapfn_as_vector.data
        cdef INT indx, nxt, i, m, k, new_i, new_indx;
        cdef INT divisor = self.c_poly._max_num_vars + 1

        #REMOVE print "MAPPING divisor=", divisor
        cdef PolyVarsIndex new_PolyVarsIndex
        cdef unordered_map[PolyVarsIndex, complex] new_coeffs
        cdef vector[INT].iterator vit
        cdef unordered_map[PolyVarsIndex, complex].iterator it = self.c_poly._coeffs.begin()

        while(it != self.c_poly._coeffs.end()): # for each coefficient
            i_vec = deref(it).first._parts  # the vector[INT] beneath this PolyVarsIndex
            new_PolyVarsIndex = PolyVarsIndex(i_vec.size())
            #REMOVE print " ivec =", i_vec

            #map i_vec -> new_PolyVarsIndex
            vit = i_vec.begin(); k = 0
            while(vit != i_vec.end()):
                indx = deref(vit)
                new_indx = 0; m=1
                #REMOVE print "begin indx ",indx
                while indx != 0:
                    nxt = indx / divisor
                    i = indx - nxt * divisor
                    indx = nxt
                    #REMOVE print indx, " -> nxt=", nxt, " i=",i

                    # i-1 is variable index (the thing we map)
                    new_i = mapfv[i-1]+1
                    new_indx += new_i * m
                    #REMOVE print "   new_i=",new_i," m=",m," => new_indx=",new_indx
                    m *= divisor
                #REMOVE print "Setting new ivec[",k,"] = ",new_indx
                new_PolyVarsIndex._parts[k] = new_indx
                inc(vit); k += 1
            #REMOVE print " new_ivec =", new_PolyVarsIndex._parts
            new_coeffs[new_PolyVarsIndex] = deref(it).second
            inc(it)

        self.c_poly._coeffs.swap(new_coeffs)

    def copy(self):
        cdef PolyCRep* c_poly = new PolyCRep(self.c_poly._coeffs, self.c_poly._max_num_vars, self.c_poly._vindices_per_int)
        return PolyRep_from_allocd_PolyCRep(c_poly)

    @property
    def max_num_vars(self): # so we can convert back to python Polys
        return self.c_poly._max_num_vars

    @property
    def vindices_per_int(self):
        return self.c_poly._vindices_per_int

    @property
    def int_coeffs(self): # just convert coeffs keys (PolyVarsIndex objs) to tuples of Python ints
        ret = {}
        cdef vector[INT].iterator vit
        cdef unordered_map[PolyVarsIndex, complex].iterator it = self.c_poly._coeffs.begin()
        while(it != self.c_poly._coeffs.end()):
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
        cdef INT divisor = self.c_poly._max_num_vars + 1
        ret = {}
        cdef vector[INT].iterator vit
        cdef unordered_map[PolyVarsIndex, complex].iterator it = self.c_poly._coeffs.begin()
        while(it != self.c_poly._coeffs.end()):
            i_tup = []
            i_vec = deref(it).first._parts

            # inline: int_to_vinds(indx)
            vit = i_vec.begin()
            while(vit != i_vec.end()):
                indx = deref(vit)
                while indx != 0:
                    nxt = indx / divisor
                    i = indx - nxt * divisor
                    i_tup.append(i-1)
                    indx = nxt
                inc(vit)

            ret[tuple(i_tup)] = deref(it).second
            inc(it)

        return ret

    def compact_complex(self):
        cdef INT i,l, iTerm, nVarIndices=0;
        cdef PolyVarsIndex k;
        cdef vector[INT] vs;
        cdef vector[INT].iterator vit
        cdef unordered_map[PolyVarsIndex, complex].iterator it = self.c_poly._coeffs.begin()
        cdef vector[ pair[PolyVarsIndex, vector[INT]] ] vinds;
        cdef INT nTerms = self.c_poly._coeffs.size()

        while(it != self.c_poly._coeffs.end()):
            vs = self.c_poly.int_to_vinds( deref(it).first )
            #REMOVE print "-> ", (deref(it).first)._parts, vs
            nVarIndices += vs.size()
            vinds.push_back( pair[PolyVarsIndex, vector[INT]](deref(it).first, vs) )
            inc(it)

        #REMOVE print "COMP: ",nTerms, nVarIndices
        vtape = np.empty(1 + nTerms + nVarIndices, np.int64) # "variable" tape
        ctape = np.empty(nTerms, np.complex128) # "coefficient tape"

        i = 0
        vtape[i] = nTerms; i+=1
        stdsort(vinds.begin(), vinds.end(), &compare_pair) # sorts in place
        for iTerm in range(vinds.size()):
            k = vinds[iTerm].first
            v = vinds[iTerm].second
            l = v.size()
            ctape[iTerm] = self.c_poly._coeffs[k]
            vtape[i] = l; i += 1
            vtape[i:i+l] = v; i += l

        return vtape, ctape

    def compact_real(self):
        cdef INT i,l, iTerm, nVarIndices=0;
        cdef PolyVarsIndex k;
        cdef vector[INT] v;
        cdef vector[INT].iterator vit
        cdef unordered_map[PolyVarsIndex, complex].iterator it = self.c_poly._coeffs.begin()
        cdef vector[ pair[PolyVarsIndex, vector[INT]] ] vinds;
        cdef INT nTerms = self.c_poly._coeffs.size()

        while(it != self.c_poly._coeffs.end()):
            vs = self.c_poly.int_to_vinds( deref(it).first )
            nVarIndices += vs.size()
            vinds.push_back( pair[PolyVarsIndex, vector[INT]](deref(it).first, vs) )
            inc(it)

        vtape = np.empty(1 + nTerms + nVarIndices, np.int64) # "variable" tape
        ctape = np.empty(nTerms, np.float64) # "coefficient tape"

        i = 0
        vtape[i] = nTerms; i+=1
        stdsort(vinds.begin(), vinds.end(), &compare_pair) # sorts in place
        for iTerm in range(vinds.size()):
            k = vinds[iTerm].first
            v = vinds[iTerm].second
            l = v.size()
            ctape[iTerm] = self.c_poly._coeffs[k].real
            vtape[i] = l; i += 1
            vtape[i:i+l] = v; i += l

        return vtape, ctape

    def mult(self, PolyRep other):
        cdef PolyCRep result = self.c_poly.mult(deref(other.c_poly))
        cdef PolyCRep* persistent = new PolyCRep(result._coeffs, result._max_num_vars, result._vindices_per_int)
        return PolyRep_from_allocd_PolyCRep(persistent)
        #print "MULT ", self.coeffs, " * ", other.coeffs, " =", ret.coeffs  #DEBUG!!! HERE
        #return ret

    def scale(self, x):
        self.c_poly.scale(x)

    def add_inplace(self, PolyRep other):
        self.c_poly.add_inplace(deref(other.c_poly))

    def add_scalar_to_all_coeffs_inplace(self, x):
        self.c_poly.add_scalar_to_all_coeffs_inplace(x)

#cdef class XXXRankOnePolyTermWithMagnitude:
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
#        return RankOnePolyTermWithMagnitude(self.term_ptr.copy(), self.magnitude)
#
#    def embed(self, stateSpaceLabels, targetLabels):
#        return RankOnePolyTermWithMagnitude(self.term_ptr.embed(stateSpaceLabels, targetLabels), self.magnitude)
#
#    def scalar_mult(self, x):
#        return RankOnePolyTermWithMagnitude(self.term_ptr * x, self.magnitude * x)
#
#    def __mul__(self, x):
#        """ Multiply by scalar """
#        return RankOnePolyTermWithMagnitude(self.term_ptr * x, self.magnitude * x)
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



cdef bool compare_pair(const pair[PolyVarsIndex, vector[INT]]& a, const pair[PolyVarsIndex, vector[INT]]& b):
    return a.first < b.first

cdef class SVTermRep:
    cdef SVTermCRep* c_term

    #Hold references to other reps so we don't GC them
    cdef public PolyRep coeff
    cdef public SVStateRep pre_state
    cdef public SVStateRep post_state
    cdef public SVEffectRep pre_effect
    cdef public SVEffectRep post_effect
    cdef public object pre_ops
    cdef public object post_ops
    cdef public object compact_coeff


    @classmethod
    def composed(cls, terms_to_compose, double magnitude):
        cdef double logmag = log10(magnitude) if magnitude > 0 else -LARGE
        cdef SVTermRep first = terms_to_compose[0]
        cdef PolyRep coeffrep = first.coeff
        pre_ops = first.pre_ops[:]
        post_ops = first.post_ops[:]
        for t in terms_to_compose[1:]:
            coeffrep = coeffrep.mult(t.coeff)
            pre_ops += t.pre_ops
            post_ops += t.post_ops
        return SVTermRep(coeffrep, magnitude, logmag, first.pre_state, first.post_state,
                         first.pre_effect, first.post_effect, pre_ops, post_ops)

    def __cinit__(self, PolyRep coeff, double mag, double logmag,
                  SVStateRep pre_state, SVStateRep post_state,
                  SVEffectRep pre_effect, SVEffectRep post_effect, pre_ops, post_ops):
        self.coeff = coeff
        self.compact_coeff = coeff.compact_complex()
        self.pre_ops = pre_ops
        self.post_ops = post_ops

        cdef INT i
        cdef INT npre = len(pre_ops)
        cdef INT npost = len(post_ops)
        cdef vector[SVOpCRep*] c_pre_ops = vector[SVGateCRep_ptr](npre)
        cdef vector[SVOpCRep*] c_post_ops = vector[SVGateCRep_ptr](<INT>len(post_ops))
        for i in range(npre):
            c_pre_ops[i] = (<SVOpRep?>pre_ops[i]).c_gate
        for i in range(npost):
            c_post_ops[i] = (<SVOpRep?>post_ops[i]).c_gate

        if pre_state is not None or post_state is not None:
            assert(pre_state is not None and post_state is not None)
            self.pre_state = pre_state
            self.post_state = post_state
            self.pre_effect = self.post_effect = None
            self.c_term = new SVTermCRep(coeff.c_poly, mag, logmag, pre_state.c_state, post_state.c_state,
                                         c_pre_ops, c_post_ops);
        elif pre_effect is not None or post_effect is not None:
            assert(pre_effect is not None and post_effect is not None)
            self.pre_effect = pre_effect
            self.post_effect = post_effect
            self.pre_state = self.post_state = None
            self.c_term = new SVTermCRep(coeff.c_poly, mag, logmag, pre_effect.c_effect, post_effect.c_effect,
                                         c_pre_ops, c_post_ops);
        else:
            self.pre_state = self.post_state = None
            self.pre_effect = self.post_effect = None
            self.c_term = new SVTermCRep(coeff.c_poly, mag, logmag, c_pre_ops, c_post_ops);

    def __dealloc__(self):
        del self.c_term

    def __reduce__(self):
        return (SVTermRep, (self.coeff, self.magnitude, self.logmagnitude,
                self.pre_state, self.post_state, self.pre_effect, self.post_effect,
                self.pre_ops, self.post_ops))

    def set_magnitude(self, double mag):
        self.c_term._magnitude = mag
        self.c_term._logmagnitude = log10(mag) if mag > 0 else -LARGE

    def set_magnitude_only(self, double mag):  # TODO: better name?
        self.c_term._magnitude = mag

    def mapvec_indices_inplace(self, mapvec):
        self.coeff.mapvec_indices_inplace(mapvec)
        self.compact_coeff = self.coeff.compact_complex()  #b/c indices have been updated!

    @property
    def magnitude(self):
        return self.c_term._magnitude

    @property
    def logmagnitude(self):
        return self.c_term._logmagnitude

    def scalar_mult(self, x):
        coeff = self.coeff.copy()
        coeff.scale(x)
        return SVTermRep(coeff, self.magnitude * x, self.logmagnitude + log10(x),
                         self.pre_state, self.post_state, self.pre_effect, self.post_effect,
                         self.pre_ops, self.post_ops)

    def copy(self):
        return SVTermRep(self.coeff.copy(), self.magnitude, self.logmagnitude,
                         self.pre_state, self.post_state, self.pre_effect, self.post_effect,
                         self.pre_ops, self.post_ops)

    #Not needed - and this implementation is quite right as it will need to change
    # the ordering of the pre/post ops also.
    #def conjugate(self):
    #    return SVTermRep(self.coeff.copy(), self.magnitude, self.logmagnitude,
    #                     self.post_state, self.pre_state, self.post_effect, self.pre_effect,
    #                     self.post_ops, self.pre_ops)


#Note: to use direct term reps (numerical coeffs) we'll need to update
# what the members are called and add methods as was done for SVTermRep.
cdef class SVTermDirectRep:
    cdef SVTermDirectCRep* c_term

    #Hold references to other reps so we don't GC them
    cdef SVStateRep state_ref1
    cdef SVStateRep state_ref2
    cdef SVEffectRep effect_ref1
    cdef SVEffectRep effect_ref2
    cdef object list_of_preops_ref
    cdef object list_of_postops_ref

    def __cinit__(self, double complex coeff, double mag, double logmag,
                  SVStateRep pre_state, SVStateRep post_state,
                  SVEffectRep pre_effect, SVEffectRep post_effect, pre_ops, post_ops):
        self.list_of_preops_ref = pre_ops
        self.list_of_postops_ref = post_ops

        cdef INT i
        cdef INT npre = len(pre_ops)
        cdef INT npost = len(post_ops)
        cdef vector[SVOpCRep*] c_pre_ops = vector[SVGateCRep_ptr](npre)
        cdef vector[SVOpCRep*] c_post_ops = vector[SVGateCRep_ptr](<INT>len(post_ops))
        for i in range(npre):
            c_pre_ops[i] = (<SVOpRep?>pre_ops[i]).c_gate
        for i in range(npost):
            c_post_ops[i] = (<SVOpRep?>post_ops[i]).c_gate

        if pre_state is not None or post_state is not None:
            assert(pre_state is not None and post_state is not None)
            self.state_ref1 = pre_state
            self.state_ref2 = post_state
            self.c_term = new SVTermDirectCRep(coeff, mag, logmag, pre_state.c_state, post_state.c_state,
                                               c_pre_ops, c_post_ops);
        elif pre_effect is not None or post_effect is not None:
            assert(pre_effect is not None and post_effect is not None)
            self.effect_ref1 = pre_effect
            self.effect_ref2 = post_effect
            self.c_term = new SVTermDirectCRep(coeff, mag, logmag, pre_effect.c_effect, post_effect.c_effect,
                                               c_pre_ops, c_post_ops);
        else:
            self.c_term = new SVTermDirectCRep(coeff, mag, logmag, c_pre_ops, c_post_ops);

    def __dealloc__(self):
        del self.c_term

    def set_coeff(self, coeff):
        self.c_term._coeff = coeff

    def set_magnitude(self, double mag, double logmag):
        self.c_term._magnitude = mag
        self.c_term._logmagnitude = logmag

    @property
    def magnitude(self):
        return self.c_term._magnitude

    @property
    def logmagnitude(self):
        return self.c_term._logmagnitude



cdef class SBTermRep:
    cdef SBTermCRep* c_term

    #Hold references to other reps so we don't GC them
    cdef public PolyRep coeff
    cdef public SBStateRep pre_state
    cdef public SBStateRep post_state
    cdef public SBEffectRep pre_effect
    cdef public SBEffectRep post_effect
    cdef public object pre_ops
    cdef public object post_ops

    @classmethod
    def composed(cls, terms_to_compose, double magnitude):
        cdef double logmag = log10(magnitude) if magnitude > 0 else -LARGE
        cdef SBTermRep first = terms_to_compose[0]
        cdef PolyRep coeffrep = first.coeff
        pre_ops = first.pre_ops[:]
        post_ops = first.post_ops[:]
        for t in terms_to_compose[1:]:
            coeffrep = coeffrep.mult(t.coeff)
            pre_ops += t.pre_ops
            post_ops += t.post_ops
        return SBTermRep(coeffrep, magnitude, logmag, first.pre_state, first.post_state,
                         first.pre_effect, first.post_effect, pre_ops, post_ops)

    def __cinit__(self, PolyRep coeff, double mag, double logmag,
                  SBStateRep pre_state, SBStateRep post_state,
                  SBEffectRep pre_effect, SBEffectRep post_effect, pre_ops, post_ops):
        self.coeff = coeff
        self.pre_ops = pre_ops
        self.post_ops = post_ops

        cdef INT i
        cdef INT npre = len(pre_ops)
        cdef INT npost = len(post_ops)
        cdef vector[SBOpCRep*] c_pre_ops = vector[SBGateCRep_ptr](npre)
        cdef vector[SBOpCRep*] c_post_ops = vector[SBGateCRep_ptr](<INT>len(post_ops))
        for i in range(npre):
            c_pre_ops[i] = (<SBOpRep?>pre_ops[i]).c_gate
        for i in range(npost):
            c_post_ops[i] = (<SBOpRep?>post_ops[i]).c_gate

        if pre_state is not None or post_state is not None:
            assert(pre_state is not None and post_state is not None)
            self.pre_state = pre_state
            self.post_state = post_state
            self.pre_effect = self.post_effect = None
            self.c_term = new SBTermCRep(coeff.c_poly, mag, logmag,
                                         pre_state.c_state, post_state.c_state,
                                         c_pre_ops, c_post_ops);
        elif pre_effect is not None or post_effect is not None:
            assert(pre_effect is not None and post_effect is not None)
            self.pre_effect = pre_effect
            self.post_effect = post_effect
            self.pre_state = self.post_state = None
            self.c_term = new SBTermCRep(coeff.c_poly, mag, logmag,
                                         pre_effect.c_effect, post_effect.c_effect,
                                         c_pre_ops, c_post_ops);
        else:
            self.pre_state = self.post_state = None
            self.pre_effect = self.post_effect = None
            self.c_term = new SBTermCRep(coeff.c_poly, mag, logmag, c_pre_ops, c_post_ops);

    def __dealloc__(self):
        del self.c_term

    def __reduce__(self):
        return (SBTermRep, (self.coeff, self.magnitude, self.logmagnitude,
                self.pre_state, self.post_state, self.pre_effect, self.post_effect,
                self.pre_ops, self.post_ops))

    def set_magnitude(self, double mag):
        self.c_term._magnitude = mag
        self.c_term._logmagnitude = log10(mag) if mag > 0 else -LARGE

    def mapvec_indices_inplace(self, mapvec):
        self.coeff.mapvec_indices_inplace(mapvec)

    @property
    def magnitude(self):
        return self.c_term._magnitude

    @property
    def logmagnitude(self):
        return self.c_term._logmagnitude

    def scalar_mult(self, x):
        coeff = self.coeff.copy()
        coeff.scale(x)
        return SBTermRep(coeff, self.magnitude * x, self.logmagnitude + log10(x),
                         self.pre_state, self.post_state, self.pre_effect, self.post_effect,
                         self.pre_ops, self.post_ops)

    def copy(self):
        return SBTermRep(self.coeff.copy(), self.magnitude, self.logmagnitude,
                         self.pre_state, self.post_state, self.pre_effect, self.post_effect,
                         self.pre_ops, self.post_ops)

    #Not needed - and this implementation is quite right as it will need to change
    # the ordering of the pre/post ops also.
    #def conjugate(self):
    #    return SBTermRep(self.coeff.copy(), self.magnitude, self.logmagnitude,
    #                     self.post_state, self.pre_state, self.post_effect, self.pre_effect,
    #                     self.post_ops, self.pre_ops)





cdef class RepCacheEl:
    cdef vector[SVTermCRep_ptr] reps
    cdef vector[INT] foat_indices
    cdef vector[INT] E_indices
    cdef public object pyterm_references

    def __cinit__(self):
        self.reps = vector[SVTermCRep_ptr](0)
        self.foat_indices = vector[INT](0)
        self.E_indices = vector[INT](0)
        self.pyterm_references = []


cdef class CircuitSetupCacheEl:
    cdef vector[INT] cgatestring
    cdef vector[SVTermCRep_ptr] rho_term_reps
    cdef unordered_map[INT, vector[SVTermCRep_ptr] ] op_term_reps
    cdef vector[SVTermCRep_ptr] E_term_reps
    cdef vector[INT] rho_foat_indices
    cdef unordered_map[INT, vector[INT] ] op_foat_indices
    cdef vector[INT] E_foat_indices
    cdef vector[INT] E_indices
    cdef object pyterm_references

    def __cinit__(self):
        self.cgatestring = vector[INT](0)
        self.rho_term_reps = vector[SVTermCRep_ptr](0)
        self.op_term_reps = unordered_map[INT, vector[SVTermCRep_ptr] ]()
        self.E_term_reps = vector[SVTermCRep_ptr](0)
        self.rho_foat_indices = vector[INT](0)
        self.op_foat_indices = unordered_map[INT, vector[INT] ]()
        self.E_foat_indices = vector[INT](0)
        self.E_indices = vector[INT](0)
        self.pyterm_references = []


## END CLASSES -- BEGIN CALC METHODS


def propagate_staterep(staterep, operationreps):
    # FUTURE: could use inner C-reps to do propagation
    # instead of using extension type wrappers as this does now
    ret = staterep
    for oprep in operationreps:
        ret = oprep.acton(ret)
        # DEBUG print("post-action rhorep = ",str(ret))
    return ret


cdef vector[vector[INT]] convert_mapevaltree(evalTree, operation_lookup, rho_lookup):
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
        if iStart == -1:  # then first element of remainder is a rholabel
            intarray[3] = rho_lookup[remainder[0]]
            for jj,gl in enumerate(remainder[1:],start=4):
                intarray[jj] = operation_lookup[gl]
        else:
            for jj,gl in enumerate(remainder,start=3):
                intarray[jj] = operation_lookup[gl]
        c_evalTree[kk] = intarray

    return c_evalTree

cdef vector[vector[INT]] convert_dict_of_intlists(d):
    # d is an dict of lists of integers, whose keys are integer
    # indices from 0 to len(d).  We can convert this
    # to a vector of vector[INT] elements.
    cdef INT i, j;
    cdef vector[vector[INT]] ret = vector[vector[INT]](len(d))
    for i, intlist in d.items():
        ret[i] = vector[INT](len(intlist))
        for j in range(len(intlist)):
            ret[i][j] = intlist[j]
    return ret

cdef vector[vector[INT]] convert_and_wrap_dict_of_intlists(d, wrapper):
    # d is an dict of lists of integers, whose keys are integer
    # indices from 0 to len(d).  We can convert this
    # to a vector of vector[INT] elements.
    cdef INT i, j;
    cdef vector[vector[INT]] ret = vector[vector[INT]](len(d))
    for i, intlist in d.items():
        ret[i] = vector[INT](len(intlist))
        for j in range(len(intlist)):
            ret[i][j] = wrapper[intlist[j]]
    return ret


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


cdef vector[DMOpCRep*] convert_gatereps(operationreps):
    # c_gatereps : an array of DMGateCReps
    cdef vector[DMOpCRep*] c_gatereps = vector[DMGateCRep_ptr](len(operationreps))
    for ii,grep in operationreps.items(): # (ii = python variable)
        c_gatereps[ii] = (<DMOpRep?>grep).c_gate
    return c_gatereps

cdef DMStateCRep* convert_rhorep(rhorep):
    # extract c-reps from rhorep and ereps => c_rho and c_ereps
    return (<DMStateRep?>rhorep).c_state

cdef vector[DMStateCRep*] convert_rhoreps(rhoreps):
    cdef vector[DMStateCRep*] c_rhoreps = vector[DMStateCRep_ptr](len(rhoreps))
    for ii,rrep in rhoreps.items(): # (ii = python variable)
        c_rhoreps[ii] = (<DMStateRep?>rrep).c_state
    return c_rhoreps

cdef vector[DMEffectCRep*] convert_ereps(ereps):
    cdef vector[DMEffectCRep*] c_ereps = vector[DMEffectCRep_ptr](len(ereps))
    for i in range(len(ereps)):
        c_ereps[i] = (<DMEffectRep>ereps[i]).c_effect
    return c_ereps


def DM_mapfill_probs_block(calc, np.ndarray[double, mode="c", ndim=1] mxToFill,
                           dest_indices, evalTree, comm):

    dest_indices = _slct.as_array(dest_indices)  # make sure this is an array and not a slice
    #dest_indices = np.ascontiguousarray(dest_indices) #unneeded

    #Get (extension-type) representation objects
    rho_lookup = { lbl:i for i,lbl in enumerate(evalTree.rholabels) } # rho labels -> ints for faster lookup
    rhoreps = { i: calc._rho_from_label(rholbl)._rep for rholbl,i in rho_lookup.items() }
    operation_lookup = { lbl:i for i,lbl in enumerate(evalTree.opLabels) } # operation labels -> ints for faster lookup
    operationreps = { i:calc.sos.get_operation(lbl)._rep for lbl,i in operation_lookup.items() }
    ereps = [E._rep for E in calc._es_from_labels(evalTree.elabels)]  # cache these in future

    # convert to C-mode:  evaltree, operation_lookup, operationreps
    cdef c_evalTree = convert_mapevaltree(evalTree, operation_lookup, rho_lookup)
    cdef vector[DMStateCRep*] c_rhos = convert_rhoreps(rhoreps)
    cdef vector[DMEffectCRep*] c_ereps = convert_ereps(ereps)
    cdef vector[DMOpCRep*] c_gatereps = convert_gatereps(operationreps)

    # create rho_cache = vector of DMStateCReps
    #print "DB: creating rho_cache of size %d * %g GB => %g GB" % \
    #   (evalTree.cache_size(), 8.0 * calc.dim / 1024.0**3, evalTree.cache_size() * 8.0 * calc.dim / 1024.0**3)
    cdef vector[DMStateCRep*] rho_cache = create_rhocache(evalTree.cache_size(), calc.dim)

    cdef vector[vector[INT]] elabel_indices_per_circuit = convert_dict_of_intlists(evalTree.eLbl_indices_per_circuit)
    cdef vector[vector[INT]] final_indices_per_circuit = convert_and_wrap_dict_of_intlists(
        evalTree.final_indices_per_circuit, dest_indices)

    dm_mapfill_probs(mxToFill, c_evalTree, c_gatereps, c_rhos, c_ereps, &rho_cache,
                     elabel_indices_per_circuit, final_indices_per_circuit, calc.dim, comm)

    free_rhocache(rho_cache)  #delete cache entries


cdef dm_mapfill_probs(double[:] mxToFill,
                      vector[vector[INT]] c_evalTree,
                      vector[DMOpCRep*] c_gatereps,
                      vector[DMStateCRep*] c_rhoreps, vector[DMEffectCRep*] c_ereps,
                      vector[DMStateCRep*]* prho_cache,
                      vector[vector[INT]] elabel_indices_per_circuit,
                      vector[vector[INT]] final_indices_per_circuit,
                      INT dim, comm): # any way to transmit comm?

    #Note: we need to take in rho_cache as a pointer b/c we may alter the values its
    # elements point to (instead of copying the states) - we just guarantee that in the end
    # all of the cache entries are filled with allocated (by 'new') states that the caller
    # can deallocate at will.
    cdef INT k,l,i,istart, icache, iFirstOp
    cdef double p
    cdef DMStateCRep *init_state
    cdef DMStateCRep *prop1
    cdef DMStateCRep *tprop
    cdef DMStateCRep *final_state
    cdef DMStateCRep *prop2 = new DMStateCRep(dim)
    cdef DMStateCRep *shelved = new DMStateCRep(dim)
    cdef vector[INT] final_indices
    cdef vector[INT] elabel_indices

    #Invariants required for proper memory management:
    # - upon loop entry, prop2 is allocated and prop1 is not (it doesn't "own" any memory)
    # - all rho_cache entries have been allocated via "new"
    for k in range(<INT>c_evalTree.size()):
        #t0 = pytime.time() # DEBUG
        intarray = c_evalTree[k]
        i = intarray[0]
        istart = intarray[1]
        icache = intarray[2]

        if istart == -1:
            init_state = c_rhoreps[intarray[3]]
            iFirstOp = 4
        else:
            init_state = deref(prho_cache)[istart]
            iFirstOp = 3

        #DEBUG
        #print "LOOP i=",i," istart=",istart," icache=",icache," remcnt=",(intarray.size()-3)
        #print [ init_state._dataptr[t] for t in range(4) ]

        #Propagate state rep
        # prop2 should already be alloc'd; need to "allocate" prop1 - either take from cache or from "shelf"
        prop1 = shelved if icache == -1 else deref(prho_cache)[icache]
        prop1.copy_from(init_state) # copy init_state -> prop1
        #print " prop1:";  print [ prop1._dataptr[t] for t in range(4) ]
        #t1 = pytime.time() # DEBUG
        for l in range(iFirstOp,<INT>intarray.size()): #during loop, both prop1 & prop2 are alloc'd
            #print "begin acton %d: %.2fs since last, %.2fs elapsed" % (l-2,pytime.time()-t1,pytime.time()-t0) # DEBUG
            #t1 = pytime.time() #DEBUG
            c_gatereps[intarray[l]].acton(prop1,prop2)
            #print " post-act prop2:"; print [ prop2._dataptr[t] for t in range(4) ]
            tprop = prop1; prop1 = prop2; prop2 = tprop # swap prop1 <-> prop2
        final_state = prop1 # output = prop1 (after swap from loop above)
        # Note: prop2 is the other alloc'd state and this maintains invariant
        #print " final:"; print [ final_state._dataptr[t] for t in range(4) ]

        #print "begin prob comps: %.2fs since last, %.2fs elapsed" % (pytime.time()-t1, pytime.time()-t0) # DEBUG
        final_indices = final_indices_per_circuit[i]
        elabel_indices = elabel_indices_per_circuit[i]
        for j in range(<INT>elabel_indices.size()):
            mxToFill[ final_indices[j] ] = c_ereps[elabel_indices[j]].probability(final_state) #outcome probability

        if icache != -1:
            deref(prho_cache)[icache] = final_state # store this state in the cache
        else: # our 2nd state was pulled from the shelf before; return it
            shelved = final_state
            final_state = NULL
        #print "%d of %d (i=%d,istart=%d,remlen=%d): %.1fs" % (k, c_evalTree.size(), i, istart,
        #                                                      intarray.size()-3, pytime.time()-t0)

    #delete our temp states
    del prop2
    del shelved


def DM_mapfill_dprobs_block(calc,
                            np.ndarray[double, mode="c", ndim=2] mxToFill,
                            dest_indices,
                            dest_param_indices,
                            evalTree, param_indices, comm):

    cdef double eps = 1e-7 #hardcoded?

    if param_indices is None:
        param_indices = list(range(calc.Np))
    if dest_param_indices is None:
        dest_param_indices = list(range(_slct.length(param_indices)))

    param_indices = _slct.as_array(param_indices)
    dest_param_indices = _slct.as_array(dest_param_indices)

    dest_indices = _slct.as_array(dest_indices)  # make sure this is an array and not a slice
    dest_indices = np.ascontiguousarray(dest_indices)

    #Get (extension-type) representation objects
    # NOTE: these calc._X_from_label(lbl) functions cache the returned operation
    # inside calc.sos's (the layer lizard's) opcache.  This speeds up future calls, but
    # more importantly causes calc.from_vector to be aware of these operations and to
    # re-initialize them with updated parameter vectors as is necessary for the finite difference loop.
    rho_lookup = { lbl:i for i,lbl in enumerate(evalTree.rholabels) } # rho labels -> ints for faster lookup
    rhoreps = { i: calc._rho_from_label(rholbl)._rep for rholbl,i in rho_lookup.items() }
    operation_lookup = { lbl:i for i,lbl in enumerate(evalTree.opLabels) } # operation labels -> ints for faster lookup
    operationreps = { i:calc._op_from_label(lbl)._rep for lbl,i in operation_lookup.items() }
    ereps = [E._rep for E in calc._es_from_labels(evalTree.elabels)]  # cache these in future

    # convert to C-mode:  evaltree, operation_lookup, operationreps
    cdef c_evalTree = convert_mapevaltree(evalTree, operation_lookup, rho_lookup)
    cdef vector[DMStateCRep*] c_rhos = convert_rhoreps(rhoreps)
    cdef vector[DMEffectCRep*] c_ereps = convert_ereps(ereps)
    cdef vector[DMOpCRep*] c_gatereps = convert_gatereps(operationreps)

    # create rho_cache = vector of DMStateCReps
    #print "DB: creating rho_cache of size %d * %g GB => %g GB" % \
    #   (evalTree.cache_size(), 8.0 * calc.dim / 1024.0**3, evalTree.cache_size() * 8.0 * calc.dim / 1024.0**3)
    cdef vector[DMStateCRep*] rho_cache = create_rhocache(evalTree.cache_size(), calc.dim)

    cdef vector[vector[INT]] elabel_indices_per_circuit = convert_dict_of_intlists(evalTree.eLbl_indices_per_circuit)
    cdef vector[vector[INT]] final_indices_per_circuit = convert_dict_of_intlists(evalTree.final_indices_per_circuit)

    all_slices, my_slice, owners, subComm = \
        _mpit.distribute_slice(slice(0, len(param_indices)), comm)

    my_param_indices = param_indices[my_slice]
    st = my_slice.start  # beginning of where my_param_indices results get placed into dpr_cache

    #Get a map from global parameter indices to the desired
    # final index within mxToFill (fpoffset = final parameter offset)
    iParamToFinal = {i: dest_param_indices[st + ii] for ii, i in enumerate(my_param_indices)}

    nEls = evalTree.num_final_elements()
    probs = np.empty(nEls, 'd') #must be contiguous!
    probs2 = np.empty(nEls, 'd') #must be contiguous!
    dm_mapfill_probs(probs, c_evalTree, c_gatereps, c_rhos, c_ereps, &rho_cache,
                     elabel_indices_per_circuit, final_indices_per_circuit, calc.dim, subComm)

    orig_vec = calc.to_vector().copy()
    for i in range(calc.Np):
        #print("dprobs cache %d of %d" % (i,self.Np))
        if i in iParamToFinal:
            iFinal = iParamToFinal[i]
            vec = orig_vec.copy(); vec[i] += eps
            calc.from_vector(vec, close=True)
            dm_mapfill_probs(probs2, c_evalTree, c_gatereps, c_rhos, c_ereps, &rho_cache,
                             elabel_indices_per_circuit, final_indices_per_circuit, calc.dim, subComm)
            _fas(mxToFill, [dest_indices, iFinal], (probs2 - probs) / eps)
    calc.from_vector(orig_vec, close=True)

    #Now each processor has filled the relavant parts of mxToFill, so gather together:
    _mpit.gather_slices(all_slices, owners, mxToFill, [], axes=1, comm=comm)

    free_rhocache(rho_cache)  #delete cache entries


cdef double TDchi2_obj_fn(double p, double f, double Ni, double N, double omitted_p, double minProbClipForWeighting, double extra):
    cdef double cp, v, omitted_cp
    cp = p if p > minProbClipForWeighting else minProbClipForWeighting
    cp = cp if cp < 1 - minProbClipForWeighting else 1 - minProbClipForWeighting
    v = (p - f) * sqrt(N / cp)

    if omitted_p != 0.0:
        # if this is the *last* outcome at this time then account for any omitted probability
        if omitted_p < minProbClipForWeighting:        omitted_cp = minProbClipForWeighting
        elif omitted_p > 1 - minProbClipForWeighting:  omitted_cp = 1 - minProbClipForWeighting
        else:                                          omitted_cp = omitted_p
        v = sqrt(v*v + N * omitted_p*omitted_p / omitted_cp)
    return v  # sqrt(the objective function term)  (the qty stored in cache)

def DM_mapfill_TDchi2_terms(calc, mxToFill, dest_indices, num_outcomes, evalTree, dataset_rows,
                            minProbClipForWeighting, probClipInterval, comm):
    DM_mapfill_TDterms(calc, "chi2", mxToFill, dest_indices, num_outcomes, evalTree,
                       dataset_rows, comm, minProbClipForWeighting, 0.0)


cdef double TDloglpp_obj_fn(double p, double f, double Ni, double N, double omitted_p, double min_p, double a):
    cdef double freq_term, S, S2, v, tmp
    cdef double pos_p = max(p, min_p)

    if Ni != 0.0:
        freq_term = Ni * (log(f) - 1.0)
    else:
        freq_term = 0.0

    S = -Ni / min_p + N
    S2 = 0.5 * Ni / (min_p*min_p)
    v = freq_term + -Ni * log(pos_p) + N * pos_p  # dims K x M (K = nSpamLabels, M = nCircuits)

    # remove small negative elements due to roundoff error (above expression *cannot* really be negative)
    v = max(v, 0)

    # quadratic extrapolation of logl at min_p for probabilities < min_p
    if p < min_p:
        tmp = (p - min_p)
        v = v + S * tmp + S2 * tmp * tmp

    if Ni == 0.0:
        if p >= a:
            v = N * p
        else:
            v = N * ((-1.0 / (3 * a*a)) * p*p*p + p*p / a + a / 3.0)
    # special handling for f == 0 terms
    # using quadratic rounding of function with minimum: max(0,(a-p)^2)/(2a) + p

    if omitted_p != 0.0:
        # if this is the *last* outcome at this time then account for any omitted probability
        v += N * omitted_p if omitted_p >= a else \
            N * ((-1.0 / (3 * a*a)) * omitted_p*omitted_p*omitted_p + omitted_p*omitted_p / a + a / 3.0)

    return v  # objective function term (the qty stored in cache)


def DM_mapfill_TDloglpp_terms(calc, mxToFill, dest_indices, num_outcomes, evalTree, dataset_rows,
                              minProbClip, radius, probClipInterval, comm):
    DM_mapfill_TDterms(calc, "logl", mxToFill, dest_indices, num_outcomes, evalTree,
                       dataset_rows, comm, minProbClip, radius)


def DM_mapfill_TDterms(calc, objective, mxToFill, dest_indices, num_outcomes,
                       evalTree, dataset_rows, comm, double fnarg1, double fnarg2):

    cdef INT i, j, k, l, n, kinit, nTotOutcomes, N, Ni
    cdef double cur_probtotal, t, t0
    cdef TD_obj_fn objfn
    if objective == "chi2":
        objfn = TDchi2_obj_fn
    else:
        objfn = TDloglpp_obj_fn

    mxToFill[dest_indices] = 0.0  # reset destination (we sum into it)
    dest_indices = _slct.as_array(dest_indices)  # make sure this is an array and not a slice

    cdef INT cacheSize = evalTree.cache_size()
    #cdef np.ndarray ret = np.zeros((len(evalTree), len(elabels)), 'd')  # zeros so we can just add contributions below
    #rhoVec, EVecs = calc._rho_es_from_labels(rholabel, elabels)
    EVecs = calc._es_from_labels(evalTree.elabels)

    elabels_as_outcomes = [(_gt.e_label_to_outcome(e),) for e in evalTree.elabels]
    outcome_to_elabel_index = {outcome: i for i, outcome in enumerate(elabels_as_outcomes)}
    dataset_rows = {i: row for i,row in enumerate(dataset_rows) } # change to dict for indexing speed - maybe pass this in? FUTURE
    num_outcomes = {i: N for i,N in enumerate(num_outcomes) } # change to dict for indexing speed

    #comm is currently ignored
    #TODO: if evalTree is split, distribute among processors
    for i in evalTree.get_evaluation_order():
        iStart, remainder, iCache = evalTree[i]
        #--
        rholabel = remainder[0]; remainder = remainder[1:]
        rhoVec = calc._rho_from_label(rholabel) #Cache?

        datarow = dataset_rows[i]
        nTotOutcomes = num_outcomes[i]
        N = 0; nOutcomes = 0

        elbl_indices = evalTree.eLbl_indices_per_circuit[i]
        final_indices = [dest_indices[j] for j in evalTree.final_indices_per_circuit[i]]
        elbl_to_final_index = {elbl_index: final_index for elbl_index, final_index in zip(elbl_indices, final_indices)}

        n = len(datarow.reps) # == len(datarow.time)
        kinit = 0
        while kinit < n:
            #Process all outcomes of this datarow occuring at a single time, t0
            t0 = datarow.time[kinit]

            #Compute N, nOutcomes for t0
            N = 0; k = kinit
            while k < n and datarow.time[k] == t0:
                N += datarow.reps[k]
                k += 1
            nOutcomes = k - kinit

            #Compute each outcome's contribution
            cur_probtotal = 0.0
            for l in range(kinit,k):
                t = t0
                rhoVec.set_time(t)
                rho = rhoVec._rep
                t += rholabel.time

                Ni = datarow.reps[l]
                outcome = datarow.outcomes[l]

                for gl in remainder:
                    op = calc.sos.get_operation(gl)
                    op.set_time(t); t += gl.time  # time in gate label == gate duration?
                    rho = op._rep.acton(rho)

                j = outcome_to_elabel_index[outcome]
                E = EVecs[j]; E.set_time(t)
                p = E._rep.probability(rho)  # outcome probability
                f = float(Ni) / float(N)
                cur_probtotal += p

                omitted_p = 1.0 - cur_probtotal if (l == k-1 and nOutcomes < nTotOutcomes) else 0.0
                # and cur_probtotal < 1.0?

                mxToFill[elbl_to_final_index[j]] += objfn(p, f, Ni, N, omitted_p, fnarg1, fnarg2)
            kinit = k


def DM_mapfill_TDdchi2_terms(calc, mxToFill, dest_indices, dest_param_indices, num_outcomes,
                             evalTree, dataset_rows, minProbClipForWeighting, probClipInterval, wrtSlice, comm):

    def fillfn(mxToFill, dest_indices, n_outcomes, evTree, dataset_rows, fillComm):
        DM_mapfill_TDchi2_terms(calc, mxToFill, dest_indices, n_outcomes, evTree, dataset_rows,
                                minProbClipForWeighting, probClipInterval, fillComm)

    DM_mapfill_timedep_dterms(calc, mxToFill, dest_indices, dest_param_indices, num_outcomes,
                              evalTree, dataset_rows, fillfn, wrtSlice, comm)


def DM_mapfill_TDdloglpp_terms(calc, mxToFill, dest_indices, dest_param_indices, num_outcomes,
                               evalTree, dataset_rows, minProbClip, radius, probClipInterval, wrtSlice, comm):

    def fillfn(mxToFill, dest_indices, n_outcomes, evTree, dataset_rows, fillComm):
        DM_mapfill_TDloglpp_terms(calc, mxToFill, dest_indices, n_outcomes, evTree, dataset_rows,
                                  minProbClip, radius, probClipInterval, fillComm)

    DM_mapfill_timedep_dterms(calc, mxToFill, dest_indices, dest_param_indices, num_outcomes,
                              evalTree, dataset_rows, fillfn, wrtSlice, comm)


def DM_mapfill_timedep_dterms(calc, mxToFill, dest_indices, dest_param_indices, num_outcomes,
                              evalTree, dataset_rows, fillfn, wrtSlice, comm):

    cdef INT i, ii, iFinal
    cdef double eps = 1e-7  # hardcoded?

    #Compute finite difference derivatives, one parameter at a time.
    param_indices = range(calc.Np) if (wrtSlice is None) else _slct.indices(wrtSlice)
    #cdef INT nDerivCols = len(param_indices)  # *all*, not just locally computed ones

    #rhoVec, EVecs = calc._rho_es_from_labels(rholabel, elabels)
    #cdef np.ndarray cache = np.empty((len(evalTree), len(elabels)), 'd')
    #cdef np.ndarray dcache = np.zeros((len(evalTree), len(elabels), nDerivCols), 'd')

    cdef INT cacheSize = evalTree.cache_size()
    cdef INT nEls = evalTree.num_final_elements()
    cdef np.ndarray vals = np.empty(nEls, 'd')
    cdef np.ndarray vals2 = np.empty(nEls, 'd')
    #assert(cacheSize == 0)

    fillfn(vals, slice(0, nEls), num_outcomes, evalTree, dataset_rows, comm)

    all_slices, my_slice, owners, subComm = \
        _mpit.distribute_slice(slice(0, len(param_indices)), comm)

    my_param_indices = param_indices[my_slice]
    cdef INT st = my_slice.start  # beginning of where my_param_indices results
    # get placed into dpr_cache

    #Get a map from global parameter indices to the desired
    # final index within dpr_cache
    iParamToFinal = {i: st + ii for ii, i in enumerate(my_param_indices)}

    orig_vec = calc.to_vector().copy()
    for i in range(calc.Np):
        #print("dprobs cache %d of %d" % (i,calc.Np))
        if i in iParamToFinal:
            iFinal = iParamToFinal[i]
            vec = orig_vec.copy(); vec[i] += eps
            calc.from_vector(vec, close=True)
            fillfn(vals2, slice(0, nEls), num_outcomes, evalTree, dataset_rows, subComm)
            _fas(mxToFill, [dest_indices, iFinal], (vals2 - vals) / eps)
    calc.from_vector(orig_vec, close=True)

    #Now each processor has filled the relavant parts of dpr_cache,
    # so gather together:
    _mpit.gather_slices(all_slices, owners, mxToFill, [], axes=1, comm=comm)

    #REMOVE
    # DEBUG LINE USED FOR MONITORION N-QUBIT GST TESTS
    #print("DEBUG TIME: dpr_cache(Np=%d, dim=%d, cachesize=%d, treesize=%d, napplies=%d) in %gs" %
    #      (calc.Np, calc.dim, cacheSize, len(evalTree), evalTree.get_num_applies(), _time.time()-tStart)) #DEBUG


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


def SV_prs_as_polys(calc, rholabel, elabels, circuit, comm=None, memLimit=None, fastmode=True):

    # Create gatelable -> int mapping to be used throughout
    distinct_gateLabels = sorted(set(circuit))
    glmap = { gl: i for i,gl in enumerate(distinct_gateLabels) }

    # Convert circuit to a vector of ints
    cdef INT i
    cdef vector[INT] cgatestring
    for gl in circuit:
        cgatestring.push_back(<INT>glmap[gl])

    cdef INT mpv = calc.Np # max_poly_vars
    #cdef INT mpo = calc.max_order*2 #max_poly_order
    cdef INT vpi = calc.poly_vindices_per_int
    cdef INT order;
    cdef INT numEs = len(elabels)

    # Construct dict of gate term reps, then *convert* to c-reps, as this
    #  keeps alive the non-c-reps which keep the c-reps from being deallocated...
    op_term_reps = { glmap[glbl]: [ [t.torep() for t in calc.sos.get_operation(glbl).get_taylor_order_terms(order, mpv)]
                                      for order in range(calc.max_order+1) ]
                       for glbl in distinct_gateLabels }

    #Similar with rho_terms and E_terms
    rho_term_reps = [ [t.torep() for t in calc.sos.get_prep(rholabel).get_taylor_order_terms(order, mpv)]
                      for order in range(calc.max_order+1) ]

    E_term_reps = []
    E_indices = []
    for order in range(calc.max_order+1):
        cur_term_reps = [] # the term reps for *all* the effect vectors
        cur_indices = [] # the Evec-index corresponding to each term rep
        for i,elbl in enumerate(elabels):
            term_reps = [t.torep() for t in calc.sos.get_effect(elbl).get_taylor_order_terms(order, mpv) ]
            cur_term_reps.extend( term_reps )
            cur_indices.extend( [i]*len(term_reps) )
        E_term_reps.append( cur_term_reps )
        E_indices.append( cur_indices )

    #convert to c-reps
    cdef INT gi
    cdef vector[vector[SVTermCRep_ptr]] rho_term_creps = sv_extract_cterms(rho_term_reps,calc.max_order)
    cdef vector[vector[SVTermCRep_ptr]] E_term_creps = sv_extract_cterms(E_term_reps,calc.max_order)
    cdef unordered_map[INT, vector[vector[SVTermCRep_ptr]]] gate_term_creps
    for gi,termrep_lists in op_term_reps.items():
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
        E_cindices, numEs, calc.max_order, mpv, vpi, stateDim, <bool>fastmode)

    return [ PolyRep_from_allocd_PolyCRep(polys[i]) for i in range(<INT>polys.size()) ]


cdef vector[PolyCRep*] sv_prs_as_polys(
    vector[INT]& circuit, vector[vector[SVTermCRep_ptr]] rho_term_reps,
    unordered_map[INT, vector[vector[SVTermCRep_ptr]]] op_term_reps,
    vector[vector[SVTermCRep_ptr]] E_term_reps, vector[vector[INT]] E_term_indices,
    INT numEs, INT max_order, INT max_poly_vars, INT vindices_per_int, INT dim, bool fastmode):

    #NOTE: circuit and gate_terms use *integers* as operation labels, not Label objects, to speed
    # lookups and avoid weird string conversion stuff with Cython

    cdef INT N = len(circuit)
    cdef INT* p = <INT*>malloc((N+2) * sizeof(INT))
    cdef INT i,j,k,order,nTerms
    cdef INT gn

    cdef sv_innerloopfn_ptr innerloop_fn;
    if fastmode:
        innerloop_fn = sv_pr_as_poly_innerloop_savepartials
    else:
        innerloop_fn = sv_pr_as_poly_innerloop

    #extract raw data from gate_terms dictionary-of-lists for faster lookup
    #gate_term_prefactors = np.empty( (nOperations,max_order+1,dim,dim)
    #cdef unordered_map[INT, vector[vector[unordered_map[INT, complex]]]] gate_term_coeffs
    #cdef vector[vector[unordered_map[INT, complex]]] rho_term_coeffs
    #cdef vector[vector[unordered_map[INT, complex]]] E_term_coeffs
    #cdef vector[vector[INT]] E_indices

    cdef vector[INT]* Einds
    cdef vector[vector_SVTermCRep_ptr_ptr] factor_lists

    assert(max_order <= 2) # only support this partitioning below (so far)

    cdef vector[PolyCRep_ptr] prps = vector[PolyCRep_ptr](numEs)
    for i in range(numEs):
        prps[i] = new PolyCRep(unordered_map[PolyVarsIndex,complex](), max_poly_vars, vindices_per_int)
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
            #factor_lists = [ gate_terms[glbl][pi] for glbl,pi in zip(circuit,p) ]
            factor_lists[0] = &rho_term_reps[p[0]]
            for k in range(N):
                gn = circuit[k]
                factor_lists[k+1] = &op_term_reps[circuit[k]][p[k+1]]
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
                    gn = circuit[k]
                    factor_lists[k+1] = &op_term_reps[gn][p[k+1]]
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
                    gn = circuit[k]
                    factor_lists[k+1] = &op_term_reps[circuit[k]][p[k+1]]
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
                        gn = circuit[k]
                        factor_lists[k+1] = &op_term_reps[circuit[k]][p[k+1]]
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

	# can't propagate effects, so effect's post_ops are constructed to act on *state*
        EVec = factor._post_effect
        for j in range(<INT>factor._post_ops.size()):
            rhoVec = factor._post_ops[j].acton(prop1,prop2)
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
        for j in range(<INT>factor._pre_ops.size()):
            factor._pre_ops[j].acton(prop1,prop2)
            tprop = prop1; prop1 = prop2; prop2 = tprop # final state in prop1
        pRight = EVec.amplitude(prop1).conjugate()


        #Add result to appropriate poly
        result = coeff  # use a reference?
        result.scale(pLeft * pRight)
        final_factor_indx = b[last_index]
        Ei = deref(Einds)[final_factor_indx] #final "factor" index == E-vector index
        deref(prps)[Ei].add_inplace(result)

        #increment b ~ itertools.product & update vec_index_noop = np.dot(self.multipliers, b)
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
            #print "DB: init from incd " #,incd,last_index,nFactorLists,dim
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
        for j in range(<INT>factor._post_ops.size()):
            factor._post_ops[j].acton(prop1,prop2)
            tprop = prop1; prop1 = prop2; prop2 = tprop
        pLeft = EVec.amplitude(prop1) # output in prop1, so this is final amplitude

        #print "DB: right ampl"
        EVec = factor._pre_effect
        prop1.copy_from(rhoVecR)
        for j in range(<INT>factor._pre_ops.size()):
            factor._pre_ops[j].acton(prop1,prop2)
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

        #increment b ~ itertools.product & update vec_index_noop = np.dot(self.multipliers, b)
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


# State-vector pruned-poly-term calcs -------------------------
def SV_create_circuitsetup_cacheel(calc, rholabel, elabels, circuit, repcache, opcache, min_term_mag, mpv):

    cdef INT i, j
    cdef vector[INT] cgatestring

    cdef RepCacheEl repcel;
    cdef vector[SVTermCRep_ptr] treps;
    cdef SVTermRep rep;
    cdef unordered_map[INT, vector[SVTermCRep_ptr] ] op_term_reps = unordered_map[INT, vector[SVTermCRep_ptr] ]();
    cdef unordered_map[INT, vector[INT] ] op_foat_indices = unordered_map[INT, vector[INT] ]();
    cdef vector[SVTermCRep_ptr] rho_term_reps;
    cdef vector[INT] rho_foat_indices;
    cdef vector[SVTermCRep_ptr] E_term_reps = vector[SVTermCRep_ptr](0);
    cdef vector[INT] E_foat_indices = vector[INT](0);
    cdef vector[INT] E_indices = vector[INT](0);
    cdef SVTermCRep_ptr cterm;
    cdef CircuitSetupCacheEl cscel = CircuitSetupCacheEl()

    # Create gatelable -> int mapping to be used throughout
    distinct_gateLabels = sorted(set(circuit))
    glmap = { gl: i for i,gl in enumerate(distinct_gateLabels) }

    # Convert circuit to a vector of ints
    for gl in circuit:
        cgatestring.push_back(<INT>glmap[gl])

    # Construct dict of gate term reps, then *convert* to c-reps, as this
    #  keeps alive the non-c-reps which keep the c-reps from being deallocated...
    for glbl in distinct_gateLabels:
        if glbl in repcache:
            repcel = <RepCacheEl>repcache[glbl]
            op_term_reps[ glmap[glbl] ] = repcel.reps
            op_foat_indices[ glmap[glbl] ] = repcel.foat_indices
        else:
            repcel = RepCacheEl()
            if glbl in opcache:
                op = opcache[glbl]
                #db_made_op = False
            else:
                op = calc.sos.get_operation(glbl)
                opcache[glbl] = op
                #db_made_op = True

            hmterms, foat_indices = op.get_highmagnitude_terms(
                min_term_mag, max_taylor_order=calc.max_order, max_poly_vars=mpv)

            #TODO REMOVE
            #if glbl in check_opcache:
            #    if np.linalg.norm( check_opcache[glbl].to_vector() - op.to_vector() ) > 1e-6:
            #        print("HERE!!!")
            #        raise ValueError("HERE!!!")
            #else:
            #    check_opcache[glbl] = op


            #DEBUG CHECK TERM MAGNITUDES make sense
            #chk_tot_mag = sum([t.magnitude for t in hmterms])
            #chk_tot_mag2 = op.get_total_term_magnitude()
            #if chk_tot_mag > chk_tot_mag2+1e-5: # give a tolerance here
            #    print "Warning: highmag terms for ",str(glbl),": ",len(hmterms)," have total mag = ",chk_tot_mag," but max should be ",chk_tot_mag2,"!!"
            #else:
            #    print "Highmag terms recomputed (OK) - made op = ", db_made_op

            for t in hmterms:
                rep = (<SVTermRep>t.torep())
                repcel.pyterm_references.append(rep)
                repcel.reps.push_back( rep.c_term )

            for i in foat_indices:
                repcel.foat_indices.push_back(<INT>i)

            op_term_reps[ glmap[glbl] ] = repcel.reps
            op_foat_indices[ glmap[glbl] ] = repcel.foat_indices
            repcache[glbl] = repcel

    #Similar with rho_terms and E_terms
    if rholabel in repcache:
        repcel = repcache[rholabel]
        rho_term_reps = repcel.reps
        rho_foat_indices = repcel.foat_indices
    else:
        repcel = RepCacheEl()
        if rholabel not in opcache:
            opcache[rholabel] = calc.sos.get_prep(rholabel)
        rhoOp = opcache[rholabel]
        hmterms, foat_indices = rhoOp.get_highmagnitude_terms(
            min_term_mag, max_taylor_order=calc.max_order,
            max_poly_vars=mpv)

        for t in hmterms:
            rep = (<SVTermRep>t.torep())
            repcel.pyterm_references.append(rep)
            repcel.reps.push_back( rep.c_term )

        for i in foat_indices:
            repcel.foat_indices.push_back(<INT>i)

        rho_term_reps = repcel.reps
        rho_foat_indices = repcel.foat_indices
        repcache[rholabel] = repcel

    elabels = tuple(elabels) # so hashable
    if elabels in repcache:
        repcel = <RepCacheEl>repcache[elabels]
        E_term_reps = repcel.reps
        E_indices = repcel.E_indices
        E_foat_indices = repcel.foat_indices
    else:
        repcel = RepCacheEl()
        E_term_indices_and_reps = []
        for i,elbl in enumerate(elabels):
            if elbl not in opcache:
                opcache[elbl] = calc.sos.get_effect(elbl)
            hmterms, foat_indices = opcache[elbl].get_highmagnitude_terms(
                min_term_mag, max_taylor_order=calc.max_order, max_poly_vars=mpv)
            E_term_indices_and_reps.extend(
                [ (i,t,t.magnitude,1 if (j in foat_indices) else 0) for j,t in enumerate(hmterms) ] )

        #Sort all terms by magnitude
        E_term_indices_and_reps.sort(key=lambda x: x[2], reverse=True)
        for j,(i,t,_,is_foat) in enumerate(E_term_indices_and_reps):
            rep = (<SVTermRep>t.torep())
            repcel.pyterm_references.append(rep)
            repcel.reps.push_back( rep.c_term )
            repcel.E_indices.push_back(<INT>i)
            if(is_foat): repcel.foat_indices.push_back(<INT>j)

        E_term_reps = repcel.reps
        E_indices = repcel.E_indices
        E_foat_indices = repcel.foat_indices
        repcache[elabels] = repcel

    cscel.cgatestring = cgatestring
    cscel.rho_term_reps = rho_term_reps
    cscel.op_term_reps = op_term_reps
    cscel.E_term_reps = E_term_reps
    cscel.rho_foat_indices = rho_foat_indices
    cscel.op_foat_indices = op_foat_indices
    cscel.E_foat_indices = E_foat_indices
    cscel.E_indices = E_indices
    return cscel

def SV_refresh_magnitudes_in_repcache(repcache, paramvec):
    cdef RepCacheEl repcel
    cdef SVTermRep termrep
    cdef np.ndarray coeff_array

    for repcel in repcache.values():
        #repcel = <RepCacheEl?>repcel
        for termrep in repcel.pyterm_references:
            coeff_array = _fastopcalc.bulk_eval_compact_polys_complex(termrep.compact_coeff[0],termrep.compact_coeff[1],paramvec,(1,))
            termrep.set_magnitude_only(abs(coeff_array[0]))


def SV_find_best_pathmagnitude_threshold(calc, rholabel, elabels, circuit, repcache, opcache, circuitsetup_cache, comm=None, memLimit=None,
                                         pathmagnitude_gap=0.0, min_term_mag=0.01, max_paths=500, threshold_guess=0.0):

    cdef INT i
    cdef INT numEs = len(elabels)
    cdef INT mpv = calc.Np # max_poly_vars
    cdef INT vpi = calc.poly_vindices_per_int
    cdef CircuitSetupCacheEl cscel;

    bHit = (circuit in circuitsetup_cache)
    if circuit in circuitsetup_cache:
        cscel = <CircuitSetupCacheEl?>circuitsetup_cache[circuit]
    else:
        cscel = <CircuitSetupCacheEl?>SV_create_circuitsetup_cacheel(calc, rholabel, elabels, circuit, repcache, opcache, min_term_mag, mpv)
        circuitsetup_cache[circuit] = cscel

    cdef vector[double] target_sum_of_pathmags = vector[double](numEs)
    cdef vector[double] achieved_sum_of_pathmags = vector[double](numEs)
    cdef vector[INT] npaths = vector[INT](numEs)

    #Get MAX-SOPM for circuit outcomes and thereby the target SOPM (via MAX - gap)
    cdef double max_partial_sopm = (opcache[rholabel] if rholabel in opcache else calc.sos.get_prep(rholabel)).get_total_term_magnitude()
    for glbl in circuit:
        op = opcache[glbl] if glbl in opcache else calc.sos.get_operation(glbl)
        max_partial_sopm *= op.get_total_term_magnitude()
    for i,elbl in enumerate(elabels):
        target_sum_of_pathmags[i] = max_partial_sopm * (opcache[elbl] if elbl in opcache else calc.sos.get_effect(elbl)).get_total_term_magnitude() - pathmagnitude_gap  # absolute gap
        #target_sum_of_pathmags[i] = max_partial_sopm * calc.sos.get_effect(elbl).get_total_term_magnitude() * (1.0 - pathmagnitude_gap)  # relative gap

    cdef double threshold = sv_find_best_pathmagnitude_threshold(
        cscel.cgatestring, cscel.rho_term_reps, cscel.op_term_reps, cscel.E_term_reps,
        cscel.rho_foat_indices, cscel.op_foat_indices, cscel.E_foat_indices, cscel.E_indices,
        numEs, pathmagnitude_gap, min_term_mag, max_paths, threshold_guess, target_sum_of_pathmags,
        achieved_sum_of_pathmags, npaths)

    cdef INT total_npaths = 0
    cdef double total_target_sopm = 0.0
    cdef double total_achieved_sopm = 0.0
    for i in range(numEs):
        total_npaths += npaths[i]
        total_target_sopm += target_sum_of_pathmags[i]
        total_achieved_sopm += achieved_sum_of_pathmags[i]

    return total_npaths, threshold, total_target_sopm, total_achieved_sopm


cdef double sv_find_best_pathmagnitude_threshold(
    vector[INT]& circuit, vector[SVTermCRep_ptr] rho_term_reps, unordered_map[INT, vector[SVTermCRep_ptr]] op_term_reps, vector[SVTermCRep_ptr] E_term_reps,
    vector[INT] rho_foat_indices, unordered_map[INT,vector[INT]] op_foat_indices, vector[INT] E_foat_indices, vector[INT] E_indices,
    INT numEs, double pathmagnitude_gap, double min_term_mag, INT max_paths, double threshold_guess,
    vector[double]& target_sum_of_pathmags, vector[double]& achieved_sum_of_pathmags, vector[INT]& npaths):

    #NOTE: circuit and gate_terms use *integers* as operation labels, not Label objects, to speed
    # lookups and avoid weird string conversion stuff with Cython

    cdef INT N = circuit.size()
    cdef INT nFactorLists = N+2
    #cdef INT n = N+2 # number of factor lists
    #cdef INT* p = <INT*>malloc((N+2) * sizeof(INT))
    cdef INT i #,j,k #,order,nTerms
    #cdef INT gn

    cdef INT t0 = time.clock()
    #cdef INT t, nPaths; #for below

    cdef vector[vector_SVTermCRep_ptr_ptr] factor_lists = vector[vector_SVTermCRep_ptr_ptr](nFactorLists)
    cdef vector[vector_INT_ptr] foat_indices_per_op = vector[vector_INT_ptr](nFactorLists)
    cdef vector[INT] nops = vector[INT](nFactorLists)
    cdef vector[INT] b = vector[INT](nFactorLists)

    factor_lists[0] = &rho_term_reps
    foat_indices_per_op[0] = &rho_foat_indices
    for i in range(N):
        factor_lists[i+1] = &op_term_reps[circuit[i]]
        foat_indices_per_op[i+1] = &op_foat_indices[circuit[i]]
    factor_lists[N+1] = &E_term_reps
    foat_indices_per_op[N+1] = &E_foat_indices

    cdef double threshold = pathmagnitude_threshold(factor_lists, E_indices, numEs, target_sum_of_pathmags, foat_indices_per_op,
                                              threshold_guess, pathmagnitude_gap / (3.0*max_paths), max_paths,
                                              achieved_sum_of_pathmags, npaths)  # 3.0 is heuristic

    #DEBUG TODO REMOVE - CHECK that counting paths using this threshold gives the same results
    cdef INT NO_LIMIT = 1000000000
    cdef vector[double] check_mags = vector[double](numEs)
    cdef vector[INT] check_npaths = vector[INT](numEs)
    for i in range(numEs):
        check_mags[i] = 0.0; check_npaths[i] = 0
    count_paths_upto_threshold(factor_lists, threshold, numEs,
                               foat_indices_per_op, E_indices, NO_LIMIT,
                               check_mags, check_npaths)
    for i in range(numEs):
        assert(abs(achieved_sum_of_pathmags[i] - check_mags[i]) < 1e-8)
        assert(npaths[i] == check_npaths[i])
    #print("Threshold = ",threshold)
    #print("Mags = ",achieved_sum_of_pathmags)
    ##print("Check Mags = ",check_mags)
    #print("npaths = ",npaths)
    ##print("Check npaths = ",check_npaths)
    ##print("Target sopm = ",target_sum_of_pathmags)  # max - gap

    return threshold




def SV_compute_pruned_path_polys_given_threshold(
        threshold, calc, rholabel, elabels, circuit, repcache, opcache, circuitsetup_cache,
        comm=None, memLimit=None, fastmode=1):

    cdef INT i
    cdef INT numEs = len(elabels)
    cdef INT mpv = calc.Np # max_poly_vars
    cdef INT vpi = calc.poly_vindices_per_int
    cdef INT stateDim = int(round(np.sqrt(calc.dim)))
    cdef double min_term_mag = calc.min_term_mag
    cdef CircuitSetupCacheEl cscel;

    bHit = (circuit in circuitsetup_cache)
    if circuit in circuitsetup_cache:
        cscel = <CircuitSetupCacheEl?>circuitsetup_cache[circuit]
    else:
        cscel = <CircuitSetupCacheEl?>SV_create_circuitsetup_cacheel(calc, rholabel, elabels, circuit, repcache, opcache, min_term_mag, mpv)
        circuitsetup_cache[circuit] = cscel

    cdef vector[PolyCRep*] polys = sv_compute_pruned_polys_given_threshold(
        <double>threshold, cscel.cgatestring, cscel.rho_term_reps, cscel.op_term_reps, cscel.E_term_reps,
        cscel.rho_foat_indices, cscel.op_foat_indices, cscel.E_foat_indices, cscel.E_indices,
        numEs, stateDim, <INT>fastmode,  mpv, vpi)

    return [ PolyRep_from_allocd_PolyCRep(polys[i]) for i in range(<INT>polys.size()) ]


cdef vector[PolyCRep*] sv_compute_pruned_polys_given_threshold(
    double threshold, vector[INT]& circuit,
    vector[SVTermCRep_ptr] rho_term_reps, unordered_map[INT, vector[SVTermCRep_ptr]] op_term_reps, vector[SVTermCRep_ptr] E_term_reps,
    vector[INT] rho_foat_indices, unordered_map[INT,vector[INT]] op_foat_indices, vector[INT] E_foat_indices, vector[INT] E_indices,
    INT numEs, INT dim, INT fastmode, INT max_poly_vars, INT vindices_per_int):

    cdef INT N = circuit.size()
    cdef INT nFactorLists = N+2
    cdef INT i

    cdef vector[vector_SVTermCRep_ptr_ptr] factor_lists = vector[vector_SVTermCRep_ptr_ptr](nFactorLists)
    cdef vector[vector_INT_ptr] foat_indices_per_op = vector[vector_INT_ptr](nFactorLists)
    cdef vector[INT] nops = vector[INT](nFactorLists)
    cdef vector[INT] b = vector[INT](nFactorLists)

    factor_lists[0] = &rho_term_reps
    foat_indices_per_op[0] = &rho_foat_indices
    for i in range(N):
        factor_lists[i+1] = &op_term_reps[circuit[i]]
        foat_indices_per_op[i+1] = &op_foat_indices[circuit[i]]
    factor_lists[N+1] = &E_term_reps
    foat_indices_per_op[N+1] = &E_foat_indices

    cdef vector[PolyCRep_ptr] prps = vector[PolyCRep_ptr](numEs)
    for i in range(numEs):
        prps[i] = new PolyCRep(unordered_map[PolyVarsIndex,complex](), max_poly_vars, vindices_per_int)
        # create empty polys - maybe overload constructor for this?
        # these PolyCReps are alloc'd here and returned - it is the job of the caller to
        #  free them (or assign them to new PolyRep wrapper objs)

    cdef double log_thres = log10(threshold)
    cdef double current_mag = 1.0
    cdef double current_logmag = 0.0
    for i in range(nFactorLists):
        nops[i] = factor_lists[i].size()
        b[i] = 0

    ## fn_visitpath(b, current_mag, 0) # visit root (all 0s) path
    cdef sv_addpathfn_ptr addpath_fn;
    cdef vector[SVStateCRep*] leftSaved = vector[SVStateCRep_ptr](nFactorLists-1)  # saved[i] is state after i-th
    cdef vector[SVStateCRep*] rightSaved = vector[SVStateCRep_ptr](nFactorLists-1) # factor has been applied
    cdef vector[PolyCRep] coeffSaved = vector[PolyCRep](nFactorLists-1)

    #Fill saved arrays with allocated states
    if fastmode == 1: # fastmode
        #fast mode
        addpath_fn = add_path_savepartials
        for i in range(nFactorLists-1):
            leftSaved[i] = new SVStateCRep(dim)
            rightSaved[i] = new SVStateCRep(dim)

    elif fastmode == 2: #achieved-SOPM mode
        addpath_fn = add_path_achievedsopm
        for i in range(nFactorLists-1):
            leftSaved[i] = NULL
            rightSaved[i] = NULL

    else:
        addpath_fn = add_path
        for i in range(nFactorLists-1):
            leftSaved[i] = NULL
            rightSaved[i] = NULL

    cdef SVStateCRep *prop1 = new SVStateCRep(dim)
    cdef SVStateCRep *prop2 = new SVStateCRep(dim)
    addpath_fn(&prps, b, 0, factor_lists, &prop1, &prop2, &E_indices, &leftSaved, &rightSaved, &coeffSaved)
    ## -------------------------------

    add_paths(addpath_fn, b, factor_lists, foat_indices_per_op, numEs, nops, E_indices, 0, log_thres,
              current_mag, current_logmag, 0, &prps, &prop1, &prop2, &leftSaved, &rightSaved, &coeffSaved)

    del prop1
    del prop2

    return prps


cdef void add_path(vector[PolyCRep*]* prps, vector[INT]& b, INT incd, vector[vector_SVTermCRep_ptr_ptr]& factor_lists,
                   SVStateCRep **pprop1, SVStateCRep **pprop2, vector[INT]* Einds,
                   vector[SVStateCRep*]* pleftSaved, vector[SVStateCRep*]* prightSaved, vector[PolyCRep]* pcoeffSaved):

    cdef PolyCRep coeff
    cdef PolyCRep result
    cdef double complex pLeft, pRight

    cdef INT i,j, Ei
    cdef SVTermCRep* factor
    cdef SVStateCRep *prop1 = deref(pprop1)
    cdef SVStateCRep *prop2 = deref(pprop2)
    cdef SVStateCRep *tprop
    cdef SVEffectCRep* EVec
    cdef SVStateCRep *rhoVec
    cdef INT nFactorLists = b.size()
    cdef INT last_index = nFactorLists-1
    # ** Assume prop1 and prop2 begin as allocated **

    # In this loop, b holds "current" indices into factor_lists
    factor = deref(factor_lists[0])[b[0]]
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

	# can't propagate effects, so effect's post_ops are constructed to act on *state*
    EVec = factor._post_effect
    for j in range(<INT>factor._pre_ops.size()):
        rhoVec = factor._pre_ops[j].acton(prop1,prop2)
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
    for j in range(<INT>factor._post_ops.size()):
        factor._post_ops[j].acton(prop1,prop2)
        tprop = prop1; prop1 = prop2; prop2 = tprop # final state in prop1
    pRight = EVec.amplitude(prop1).conjugate()


    #Add result to appropriate poly
    result = coeff  # use a reference?
    result.scale(pLeft * pRight)
    Ei = deref(Einds)[ b[last_index] ] #final "factor" index == E-vector index
    #print("Ei = ",Ei," size = ",deref(prps).size())
    #print("result = ")
    #for x in result._coeffs:
    #    print x.first._parts, x.second
    #print("prps = ")
    #for x in deref(prps)[Ei]._coeffs:
    #    print x.first._parts, x.second
    deref(prps)[Ei].add_inplace(result)

    #Update the slots held by prop1 and prop2, which still have allocated states (though really no need?)
    pprop1[0] = prop1
    pprop2[0] = prop2


cdef void add_path_achievedsopm(vector[PolyCRep*]* prps, vector[INT]& b, INT incd, vector[vector_SVTermCRep_ptr_ptr]& factor_lists,
                                SVStateCRep **pprop1, SVStateCRep **pprop2, vector[INT]* Einds,
                                vector[SVStateCRep*]* pleftSaved, vector[SVStateCRep*]* prightSaved, vector[PolyCRep]* pcoeffSaved):

    cdef PolyCRep coeff
    cdef PolyCRep result

    cdef INT i,j, Ei
    cdef SVTermCRep* factor
    cdef INT nFactorLists = b.size()
    cdef INT last_index = nFactorLists-1

    # In this loop, b holds "current" indices into factor_lists
    factor = deref(factor_lists[0])[b[0]]
    coeff = deref(factor._coeff).abs() # an unordered_map (copies to new "coeff" variable)

    for i in range(1,nFactorLists):
        coeff = coeff.abs_mult( deref(deref(factor_lists[i])[b[i]]._coeff) )

    #Add result to appropriate poly
    result = coeff  # use a reference?
    Ei = deref(Einds)[ b[last_index] ] #final "factor" index == E-vector index
    deref(prps)[Ei].add_abs_inplace(result)


cdef void add_path_savepartials(vector[PolyCRep*]* prps, vector[INT]& b, INT incd, vector[vector_SVTermCRep_ptr_ptr]& factor_lists,
                                SVStateCRep** pprop1, SVStateCRep** pprop2, vector[INT]* Einds,
                                vector[SVStateCRep*]* pleftSaved, vector[SVStateCRep*]* prightSaved, vector[PolyCRep]* pcoeffSaved):

    cdef PolyCRep coeff
    cdef PolyCRep result
    cdef double complex pLeft, pRight

    cdef INT i,j, Ei
    cdef SVTermCRep* factor
    cdef SVStateCRep *prop1 = deref(pprop1)
    cdef SVStateCRep *prop2 = deref(pprop2)
    cdef SVStateCRep *tprop
    cdef SVStateCRep *shelved = prop1
    cdef SVEffectCRep* EVec
    cdef SVStateCRep *rhoVec
    cdef INT nFactorLists = b.size()
    cdef INT last_index = nFactorLists-1
    # ** Assume shelved and prop2 begin as allocated **

    if incd == 0: # need to re-evaluate rho vector
        #print "DB: re-eval at incd=0"
        factor = deref(factor_lists[0])[b[0]]

        #print "DB: re-eval left"
        prop1 = deref(pleftSaved)[0] # the final destination (prop2 is already alloc'd)
        prop1.copy_from(factor._pre_state)
        for j in range(<INT>factor._pre_ops.size()):
            #print "DB: re-eval left item"
            factor._pre_ops[j].acton(prop1,prop2)
            tprop = prop1; prop1 = prop2; prop2 = tprop # swap prop1 <-> prop2
        rhoVecL = prop1
        deref(pleftSaved)[0] = prop1 # final state -> saved
        # (prop2 == the other allocated state)

        #print "DB: re-eval right"
        prop1 = deref(prightSaved)[0] # the final destination (prop2 is already alloc'd)
        prop1.copy_from(factor._post_state)
        for j in range(<INT>factor._post_ops.size()):
            #print "DB: re-eval right item"
            factor._post_ops[j].acton(prop1,prop2)
            tprop = prop1; prop1 = prop2; prop2 = tprop # swap prop1 <-> prop2
        rhoVecR = prop1
        deref(prightSaved)[0] = prop1 # final state -> saved
        # (prop2 == the other allocated state)

        #print "DB: re-eval coeff"
        coeff = deref(factor._coeff)
        deref(pcoeffSaved)[0] = coeff
        incd += 1
    else:
        #print "DB: init from incd"
        rhoVecL = deref(pleftSaved)[incd-1]
        rhoVecR = deref(prightSaved)[incd-1]
        coeff = deref(pcoeffSaved)[incd-1]

    # propagate left and right states, saving as we go
    for i in range(incd,last_index):
        #print "DB: propagate left begin"
        factor = deref(factor_lists[i])[b[i]]
        prop1 = deref(pleftSaved)[i] # destination
        prop1.copy_from(rhoVecL) #starting state
        for j in range(<INT>factor._pre_ops.size()):
            #print "DB: propagate left item"
            factor._pre_ops[j].acton(prop1,prop2)
            tprop = prop1; prop1 = prop2; prop2 = tprop
        rhoVecL = prop1
        deref(pleftSaved)[i] = prop1
        # (prop2 == the other allocated state)

        #print "DB: propagate right begin"
        prop1 = deref(prightSaved)[i] # destination
        prop1.copy_from(rhoVecR) #starting state
        for j in range(<INT>factor._post_ops.size()):
            #print "DB: propagate right item"
            factor._post_ops[j].acton(prop1,prop2)
            tprop = prop1; prop1 = prop2; prop2 = tprop
        rhoVecR = prop1
        deref(prightSaved)[i] = prop1
        # (prop2 == the other allocated state)

        #print "DB: propagate coeff mult"
        coeff = coeff.mult(deref(factor._coeff)) # copy a PolyCRep
        deref(pcoeffSaved)[i] = coeff

    # for the last index, no need to save, and need to construct
    # and apply effect vector
    prop1 = shelved # so now prop1 (and prop2) are alloc'd states

    #print "DB: left ampl"
    factor = deref(factor_lists[last_index])[b[last_index]] # the last factor (an Evec)
    EVec = factor._post_effect
    prop1.copy_from(rhoVecL) # initial state (prop2 already alloc'd)
    for j in range(<INT>factor._pre_ops.size()):
        factor._pre_ops[j].acton(prop1,prop2)
        tprop = prop1; prop1 = prop2; prop2 = tprop
    pLeft = EVec.amplitude(prop1) # output in prop1, so this is final amplitude

    #print "DB: right ampl"
    EVec = factor._pre_effect
    prop1.copy_from(rhoVecR)
    for j in range(<INT>factor._post_ops.size()):
        factor._post_ops[j].acton(prop1,prop2)
        tprop = prop1; prop1 = prop2; prop2 = tprop
    pRight = EVec.amplitude(prop1).conjugate()

    shelved = prop1 # return prop1 to the "shelf" since we'll use prop1 for other things next

    #print "DB: final block"
    #print "DB running coeff = ",dict(coeff._coeffs)
    #print "DB factor coeff = ",dict(factor._coeff._coeffs)
    result = coeff.mult(deref(factor._coeff))
    #print "DB result = ",dict(result._coeffs)
    result.scale(pLeft * pRight)
    Ei = deref(Einds)[b[last_index]] #final "factor" index == E-vector index
    deref(prps)[Ei].add_inplace(result)

    #Update the slots held by prop1 and prop2, which still have allocated states
    pprop1[0] = prop1 # b/c can't deref(pprop1) = prop1 isn't allowed (?)
    pprop2[0] = prop2 # b/c can't deref(pprop2) = prop1 isn't allowed (?)
    #print "DB prps[",INT(Ei),"] = ",dict(deref(prps)[Ei]._coeffs)


cdef void add_paths(sv_addpathfn_ptr addpath_fn, vector[INT]& b, vector[vector_SVTermCRep_ptr_ptr] oprep_lists,
                    vector[vector_INT_ptr] foat_indices_per_op, INT num_elabels,
                    vector[INT]& nops, vector[INT]& E_indices, INT incd, double log_thres,
                    double current_mag, double current_logmag, INT order,
                    vector[PolyCRep*]* prps, SVStateCRep **pprop1, SVStateCRep **pprop2,
                    vector[SVStateCRep*]* pleftSaved, vector[SVStateCRep*]* prightSaved, vector[PolyCRep]* pcoeffSaved):
    """ first_order means only one b[i] is incremented, e.g. b == [0 1 0] or [4 0 0] """
    cdef INT i, j, k, orig_bi, orig_bn
    cdef INT n = b.size()
    cdef INT sub_order
    cdef double mag, mag2

    for i in range(n-1, incd-1, -1):
        if b[i]+1 == nops[i]: continue
        b[i] += 1

        if order == 0: # then incd doesn't matter b/c can inc anything to become 1st order
            sub_order = 1 if (i != n-1 or b[i] >= num_elabels) else 0
        elif order == 1:
            # we started with a first order term where incd was incremented, and now
            # we're incrementing something else
            sub_order = 1 if i == incd else 2 # signifies anything over 1st order where >1 column has be inc'd
        else:
            sub_order = order

        logmag = current_logmag + (deref(oprep_lists[i])[b[i]]._logmagnitude - deref(oprep_lists[i])[b[i]-1]._logmagnitude)
        if logmag >= log_thres:
            if deref(oprep_lists[i])[b[i]-1]._magnitude == 0:
                mag = 0
            else:
                mag = current_mag * (deref(oprep_lists[i])[b[i]]._magnitude / deref(oprep_lists[i])[b[i]-1]._magnitude)

            ## fn_visitpath(b, mag, i) ##
            addpath_fn(prps, b, i, oprep_lists, pprop1, pprop2, &E_indices, pleftSaved, prightSaved, pcoeffSaved)
            ## --------------------------

            #add any allowed paths beneath this one
            add_paths(addpath_fn, b, oprep_lists, foat_indices_per_op, num_elabels, nops, E_indices,
                      i, log_thres, mag, logmag, sub_order, prps, pprop1, pprop2,
                      pleftSaved, prightSaved, pcoeffSaved)

        elif sub_order <= 1:
            #We've rejected term-index b[i] (in column i) because it's too small - the only reason
            # to accept b[i] or term indices higher than it is to include "foat" terms, so we now
            # iterate through any remaining foat indices for this column (we've accepted all lower
            # values of b[i], or we wouldn't be here).  Note that we just need to visit the path,
            # we don't need to traverse down, since we know the path magnitude is already too low.
            orig_bi = b[i]
            for j in deref(foat_indices_per_op[i]):
                if j >= orig_bi:
                    b[i] = j
                    mag = 0 if deref(oprep_lists[i])[orig_bi-1]._magnitude == 0 else \
                        current_mag * (deref(oprep_lists[i])[b[i]]._magnitude / deref(oprep_lists[i])[orig_bi-1]._magnitude)

                    ## fn_visitpath(b, mag, i) ##
                    addpath_fn(prps, b, i, oprep_lists, pprop1, pprop2, &E_indices, pleftSaved, prightSaved, pcoeffSaved)
                    ## --------------------------

                    if i != n-1:
                        # if we're not incrementing (from a zero-order term) the final index, then we
                        # need to to increment it until we hit num_elabels (*all* zero-th order paths)
                        orig_bn = b[n-1]
                        for k in range(1,num_elabels):
                            b[n-1] = k
                            mag2 = mag * (deref(oprep_lists[n-1])[b[n-1]]._magnitude / deref(oprep_lists[i])[orig_bn]._magnitude)

                            ## fn_visitpath(b, mag2, n-1) ##
                            addpath_fn(prps, b, n-1, oprep_lists, pprop1, pprop2, &E_indices, pleftSaved, prightSaved, pcoeffSaved)
                            ## --------------------------
                        b[n-1] = orig_bn
            b[i] = orig_bi
        b[i] -= 1 # so we don't have to copy b


#HACK - need a way to add up magnitudes based on *current* coeffs (evaluated polys of terms at *current* paramvec) while
# using the locked in magnitudes to determine how many paths to actually include.  Currently, this is done by only
# "refreshing" the .magnitudes of the terms and leaving the .logmagnitudes (which are used to determing which paths to
# include) at their "locked in" values.  Thus, it is not always true that log10(term.magnitude) == term.logmagnitude.  This
# seems non-intuitive and could be problematic if it isn't at least clarified in the FUTURE.
cdef bool count_paths(vector[INT]& b, vector[vector_SVTermCRep_ptr_ptr]& oprep_lists,
                      vector[vector_INT_ptr]& foat_indices_per_op, INT num_elabels,
                      vector[INT]& nops, vector[INT]& E_indices, vector[double]& pathmags, vector[INT]& nPaths,
                      INT incd, double log_thres, double current_mag, double current_logmag, INT order, INT max_npaths,
                      INT current_nzeros):
    """ first_order means only one b[i] is incremented, e.g. b == [0 1 0] or [4 0 0] """
    cdef INT i, j, k, orig_bi, orig_bn
    cdef INT n = b.size()
    cdef INT sub_order
    cdef double mag, mag2
    cdef double numerator, denom

    for i in range(n-1, incd-1, -1):
        if b[i]+1 == nops[i]: continue
        b[i] += 1

        if order == 0: # then incd doesn't matter b/c can inc anything to become 1st order
            sub_order = 1 if (i != n-1 or b[i] >= num_elabels) else 0
        elif order == 1:
            # we started with a first order term where incd was incremented, and now
            # we're incrementing something else
            sub_order = 1 if i == incd else 2 # signifies anything over 1st order where >1 column has be inc'd
        else:
            sub_order = order

        logmag = current_logmag + (deref(oprep_lists[i])[b[i]]._logmagnitude - deref(oprep_lists[i])[b[i]-1]._logmagnitude)
        if logmag >= log_thres:
            numerator = deref(oprep_lists[i])[b[i]]._magnitude
            denom = deref(oprep_lists[i])[b[i]-1]._magnitude
            nzeros = current_nzeros
            if denom == 0:
                denom = SMALL; nzeros -= 1
            if numerator == 0:
                numerator = SMALL; nzeros += 1
            mag = current_mag * (numerator / denom)

            ## fn_visitpath(b, mag, i) ##
            if nzeros == 0:
                pathmags[E_indices[b[n-1]]] += mag
            nPaths[E_indices[b[n-1]]] += 1
            if nPaths[E_indices[b[n-1]]] == max_npaths: return True
            #print("Adding ",b)
            ## --------------------------

            #add any allowed paths beneath this one
            if count_paths(b, oprep_lists, foat_indices_per_op, num_elabels, nops,
                           E_indices, pathmags, nPaths, i, log_thres, mag, logmag, sub_order, max_npaths, nzeros):
                return True

        elif sub_order <= 1:
            #We've rejected term-index b[i] (in column i) because it's too small - the only reason
            # to accept b[i] or term indices higher than it is to include "foat" terms, so we now
            # iterate through any remaining foat indices for this column (we've accepted all lower
            # values of b[i], or we wouldn't be here).  Note that we just need to visit the path,
            # we don't need to traverse down, since we know the path magnitude is already too low.
            orig_bi = b[i]
            for j in deref(foat_indices_per_op[i]):
                if j >= orig_bi:
                    b[i] = j
                    nzeros = current_nzeros
                    numerator = deref(oprep_lists[i])[b[i]]._magnitude
                    denom = deref(oprep_lists[i])[orig_bi-1]._magnitude
                    if denom == 0: denom = SMALL
                    #if numerator == 0: nzeros += 1  # not needed b/c we just leave numerator = 0
                    mag = current_mag * (numerator / denom)  # OK if mag == 0 as it's not passed to any recursive calls

                    ## fn_visitpath(b, mag, i) ##
                    if nzeros == 0:
                        pathmags[E_indices[b[n-1]]] += mag
                    nPaths[E_indices[b[n-1]]] += 1
                    #if E_indices[b[n-1]] == 0:  # TODO REMOVE
                    #    print nPaths[E_indices[b[n-1]]], mag, pathmags[E_indices[b[n-1]]], b, current_mag, deref(oprep_lists[i])[b[i]]._magnitude, deref(oprep_lists[i])[orig_bi-1]._magnitude, incd, i, "*2"
                    if nPaths[E_indices[b[n-1]]] == max_npaths: return True
                    #print("FOAT Adding ",b)
                    ## --------------------------

                    if i != n-1:
                        # if we're not incrementing (from a zero-order term) the final index, then we
                        # need to to increment it until we hit num_elabels (*all* zero-th order paths)
                        orig_bn = b[n-1]
                        for k in range(1,num_elabels):
                            b[n-1] = k
                            numerator = deref(oprep_lists[n-1])[b[n-1]]._magnitude
                            denom = deref(oprep_lists[i])[orig_bn]._magnitude
                            if denom == 0: denom = SMALL

                            mag2 = mag * (numerator / denom)

                            ## fn_visitpath(b, mag2, n-1) ##
                            if nzeros == 0:  # if numerator was zero above, mag2 will be zero, so we still won't add anyting (good)
                                pathmags[E_indices[b[n-1]]] += mag2
                            nPaths[E_indices[b[n-1]]] += 1
                            #if E_indices[b[n-1]] == 0:  # TODO REMOVE
                            #    print nPaths[E_indices[b[n-1]]], mag2, pathmags[E_indices[b[n-1]]], b, mag, incd, i, " *3"
                            if nPaths[E_indices[b[n-1]]] == max_npaths: return True
                            #print("FOAT Adding ",b)
                            ## --------------------------
                        b[n-1] = orig_bn
            b[i] = orig_bi
        b[i] -= 1 # so we don't have to copy b
    return False


cdef void count_paths_upto_threshold(vector[vector_SVTermCRep_ptr_ptr] oprep_lists, double pathmag_threshold, INT num_elabels,
                                     vector[vector_INT_ptr] foat_indices_per_op, vector[INT]& E_indices, INT max_npaths,
                                     vector[double]& pathmags, vector[INT]& nPaths):
    """ TODO: docstring """
    cdef INT i
    cdef INT n = oprep_lists.size()
    cdef vector[INT] nops = vector[INT](n)
    cdef vector[INT] b = vector[INT](n)
    cdef double log_thres = log10(pathmag_threshold)
    cdef double current_mag = 1.0
    cdef double current_logmag = 0.0

    for i in range(n):
        nops[i] = oprep_lists[i].size()
        b[i] = 0

    ## fn_visitpath(b, current_mag, 0) # visit root (all 0s) path
    pathmags[E_indices[0]] += current_mag
    nPaths[E_indices[0]] += 1
    #print("Adding ",b)
    ## -------------------------------
    count_paths(b, oprep_lists, foat_indices_per_op, num_elabels, nops, E_indices, pathmags, nPaths,
                0, log_thres, current_mag, current_logmag, 0, max_npaths, 0)
    return


cdef double pathmagnitude_threshold(vector[vector_SVTermCRep_ptr_ptr] oprep_lists, vector[INT]& E_indices,
                                    INT nEffects, vector[double] target_sum_of_pathmags,
                                    vector[vector_INT_ptr] foat_indices_per_op,
                                    double initial_threshold, double min_threshold, INT max_npaths,
                                    vector[double]& mags, vector[INT]& nPaths):
    """
    TODO: docstring - note: target_sum_of_pathmags is a *vector* that holds a separate value for each E-index
    """
    cdef INT nIters = 0
    cdef double threshold = initial_threshold if (initial_threshold >= 0) else 0.1 # default value
    #target_mag = target_sum_of_pathmags
    cdef double threshold_upper_bound = 1.0
    cdef double threshold_lower_bound = -1.0
    cdef INT i, j
    cdef INT try_larger_threshold

    while nIters < 100: # TODO: allow setting max_nIters as an arg?
        for i in range(nEffects):
            mags[i] = 0.0; nPaths[i] = 0
        count_paths_upto_threshold(oprep_lists, threshold, nEffects,
                                   foat_indices_per_op, E_indices, max_npaths,
                                   mags, nPaths)

        try_larger_threshold = 1 # True
        for i in range(nEffects):
            #if(mags[i] > target_sum_of_pathmags[i]): #DEBUG TODO REMOVE
            #    print "MAGS TOO LARGE!!! mags=",mags[i]," target_sum=",target_sum_of_pathmags[i]

            if(mags[i] < target_sum_of_pathmags[i]):
                try_larger_threshold = 0 # False

                #Check that max_npaths has not been reached - if so, *still* try a larger threshold
                for j in range(nEffects):
                    if nPaths[j] >= max_npaths:
                        try_larger_threshold = 1 # True
                        break
                break

        if try_larger_threshold:
            threshold_lower_bound = threshold
            if threshold_upper_bound >= 0: # ~(is not None)
                threshold = (threshold_upper_bound + threshold_lower_bound)/2
            else: threshold *= 2
        else: # try smaller threshold
            threshold_upper_bound = threshold
            if threshold_lower_bound >= 0: # ~(is not None)
                threshold = (threshold_upper_bound + threshold_lower_bound)/2
            else: threshold /= 2

        #print("  Interval: threshold in [%s,%s]: %s %s" % (str(threshold_upper_bound),str(threshold_lower_bound),mag,nPaths))
        if threshold_upper_bound >= 0 and threshold_lower_bound >= 0 and \
           (threshold_upper_bound - threshold_lower_bound)/threshold_upper_bound < 1e-3:
            #print("Converged after %d iters!" % nIters)
            break
        if threshold_upper_bound < min_threshold: # could also just set min_threshold to be the lower bound initially?
            threshold_upper_bound = threshold_lower_bound = min_threshold
            #print("Hit min threshold after %d iters!" % nIters)
            break

        nIters += 1

    #Run path traversal once more to count final number of paths
    for i in range(nEffects):
        mags[i] = 0.0; nPaths[i] = 0
    count_paths_upto_threshold(oprep_lists, threshold_lower_bound, nEffects,
                               foat_indices_per_op, E_indices, 1000000000, mags, nPaths) # sets mags and nPaths
    # 1000000000 == NO_LIMIT; we want to test that the threshold above limits the number of
    # paths to (approximately) max_npaths -- it's ok if the count is slightly higher since additional paths
    # may be needed to ensure all equal-weight paths are considered together (needed for the resulting prob to be *real*).

    return threshold_lower_bound

def SV_circuit_achieved_and_max_sopm(calc, rholabel, elabels, circuit, repcache, opcache, threshold, min_term_mag):
    """ TODO: docstring """

    #Same beginning as SV_prs_as_pruned_polys -- should consolidate this setup code elsewhere

    #opcache = {} # DEBUG - test if this is responsible for warnings

    #t0 = pytime.time()
    #if debug is not None:
    #    debug['tstartup'] += pytime.time()-t0
    #    t0 = pytime.time()

    cdef INT i, j
    cdef INT numEs = len(elabels)
    cdef INT mpv = calc.Np # max_poly_vars
    cdef CircuitSetupCacheEl cscel;
    circuitsetup_cache = {} # for now...

    if circuit in circuitsetup_cache:
        cscel = <CircuitSetupCacheEl?>circuitsetup_cache[circuit]
    else:
        cscel = <CircuitSetupCacheEl?>SV_create_circuitsetup_cacheel(calc, rholabel, elabels, circuit, repcache, opcache, min_term_mag, mpv)
        circuitsetup_cache[circuit] = cscel

    #Get MAX-SOPM for circuit outcomes and thereby the target SOPM (via MAX - gap)
    cdef double max_partial_sopm = (opcache[rholabel] if rholabel in opcache else calc.sos.get_prep(rholabel)).get_total_term_magnitude()
    cdef vector[double] max_sum_of_pathmags = vector[double](numEs)
    for glbl in circuit:
        op = opcache[glbl] if glbl in opcache else calc.sos.get_operation(glbl)
        max_partial_sopm *= op.get_total_term_magnitude()
    for i,elbl in enumerate(elabels):
        max_sum_of_pathmags[i] = max_partial_sopm * (opcache[elbl] if elbl in opcache else calc.sos.get_effect(elbl)).get_total_term_magnitude()

    #Note: term calculator "dim" is the full density matrix dim
    stateDim = int(round(np.sqrt(calc.dim)))

    #------ From sv_prs_pruned ---- build up factor_lists and foat_indices_per_op
    cdef INT N = cscel.cgatestring.size()
    cdef INT nFactorLists = N+2
    cdef vector[vector_SVTermCRep_ptr_ptr] factor_lists = vector[vector_SVTermCRep_ptr_ptr](nFactorLists)
    cdef vector[vector_INT_ptr] foat_indices_per_op = vector[vector_INT_ptr](nFactorLists)

    factor_lists[0] = &cscel.rho_term_reps
    foat_indices_per_op[0] = &cscel.rho_foat_indices
    for i in range(N):
        factor_lists[i+1] = &cscel.op_term_reps[cscel.cgatestring[i]]
        foat_indices_per_op[i+1] = &cscel.op_foat_indices[cscel.cgatestring[i]]
    factor_lists[N+1] = &cscel.E_term_reps
    foat_indices_per_op[N+1] = &cscel.E_foat_indices
    # --------------------------------------------

    # Specific path magnitude summing (and we count paths, even though this isn't needed)
    cdef INT NO_LIMIT = 1000000000
    cdef vector[double] mags = vector[double](numEs)
    cdef vector[INT] npaths = vector[INT](numEs)

    for i in range(numEs):
        mags[i] = 0.0; npaths[i] = 0

    count_paths_upto_threshold(factor_lists, threshold, numEs,
                               foat_indices_per_op, cscel.E_indices, NO_LIMIT,
                               mags, npaths)

    ##DEBUG TODO REMOVE
    #print("Getting GAP for: ", circuit)
    #print("Threshold = ",threshold)
    #print("Mags = ",mags)
    #print("npaths = ",npaths)
    #print("MAX sopm = ",max_sum_of_pathmags)

    achieved_sopm = np.empty(numEs,'d')
    max_sopm = np.empty(numEs,'d')
    for i in range(numEs):
        achieved_sopm[i] = mags[i]
        max_sopm[i] = max_sum_of_pathmags[i]

    return achieved_sopm, max_sopm




# State-vector direct-term calcs -------------------------

#cdef vector[vector[SVTermDirectCRep_ptr]] sv_extract_cterms_direct(python_termrep_lists, INT max_order):
#    cdef vector[vector[SVTermDirectCRep_ptr]] ret = vector[vector[SVTermDirectCRep_ptr]](max_order+1)
#    cdef vector[SVTermDirectCRep*] vec_of_terms
#    for order,termreps in enumerate(python_termrep_lists): # maxorder+1 lists
#        vec_of_terms = vector[SVTermDirectCRep_ptr](len(termreps))
#        for i,termrep in enumerate(termreps):
#            vec_of_terms[i] = (<SVTermDirectRep?>termrep).c_term
#        ret[order] = vec_of_terms
#    return ret

#def SV_prs_directly(calc, rholabel, elabels, circuit, repcache, comm=None, memLimit=None, fastmode=True, wtTol=0.0, resetTermWeights=True, debug=None):
#
#    # Create gatelable -> int mapping to be used throughout
#    distinct_gateLabels = sorted(set(circuit))
#    glmap = { gl: i for i,gl in enumerate(distinct_gateLabels) }
#    t0 = pytime.time()
#
#    # Convert circuit to a vector of ints
#    cdef INT i, j
#    cdef vector[INT] cgatestring
#    for gl in circuit:
#        cgatestring.push_back(<INT>glmap[gl])
#
#    #TODO: maybe compute these weights elsewhere and pass in?
#    cdef double circuitWeight
#    cdef double remaingingWeightTol = <double?>wtTol
#    cdef vector[double] remainingWeight = vector[double](<INT>len(elabels))
#    if 'circuitWeights' not in repcache:
#        repcache['circuitWeights'] = {}
#    if resetTermWeights or circuit not in repcache['circuitWeights']:
#        circuitWeight = calc.sos.get_prep(rholabel).get_total_term_weight()
#        for gl in circuit:
#            circuitWeight *= calc.sos.get_operation(gl).get_total_term_weight()
#        for i,elbl in enumerate(elabels):
#            remainingWeight[i] = circuitWeight * calc.sos.get_effect(elbl).get_total_term_weight()
#        repcache['circuitWeights'][circuit] = [ remainingWeight[i] for i in range(remainingWeight.size()) ]
#    else:
#        for i,wt in enumerate(repcache['circuitWeights'][circuit]):
#            assert(wt > 1.0)
#            remainingWeight[i] = wt
#
#    #if resetTermWeights:
#    #    print "Remaining weights: "
#    #    for i in range(remainingWeight.size()):
#    #        print remainingWeight[i]
#
#    cdef double order_base = 0.1 # default for now - TODO: make this a calc param like max_order?
#    cdef INT order
#    cdef INT numEs = len(elabels)
#
#    cdef RepCacheEl repcel;
#    cdef vector[SVTermDirectCRep_ptr] treps;
#    cdef DCOMPLEX* coeffs;
#    cdef vector[SVTermDirectCRep*] reps_at_order;
#    cdef np.ndarray coeffs_array;
#    cdef SVTermDirectRep rep;
#
#    # Construct dict of gate term reps, then *convert* to c-reps, as this
#    #  keeps alive the non-c-reps which keep the c-reps from being deallocated...
#    cdef unordered_map[INT, vector[vector[SVTermDirectCRep_ptr]] ] op_term_reps = unordered_map[INT, vector[vector[SVTermDirectCRep_ptr]] ](); # OLD = {}
#    for glbl in distinct_gateLabels:
#        if glbl in repcache:
#            repcel = <RepCacheEl?>repcache[glbl]
#            op_term_reps[ glmap[glbl] ] = repcel.reps
#            for order in range(calc.max_order+1):
#                treps = repcel.reps[order]
#                coeffs_array = calc.sos.get_operation(glbl).get_direct_order_coeffs(order,order_base)
#                coeffs = <DCOMPLEX*?>(coeffs_array.data)
#                for i in range(treps.size()):
#                    treps[i]._coeff = coeffs[i]
#                    if resetTermWeights: treps[i]._magnitude = abs(coeffs[i])
#            #for order,treps in enumerate(op_term_reps[ glmap[glbl] ]):
#            #    for coeff,trep in zip(calc.sos.get_operation(glbl).get_direct_order_coeffs(order,order_base), treps):
#            #        trep.set_coeff(coeff)
#        else:
#            repcel = RepCacheEl(calc.max_order)
#            for order in range(calc.max_order+1):
#                reps_at_order = vector[SVTermDirectCRep_ptr](0)
#                for t in calc.sos.get_operation(glbl).get_direct_order_terms(order,order_base):
#                    rep = (<SVTermDirectRep?>t.torep(None,None,"gate"))
#                    repcel.pyterm_references.append(rep)
#                    reps_at_order.push_back( rep.c_term )
#                repcel.reps[order] = reps_at_order
#            #OLD
#            #reps = [ [t.torep(None,None,"gate") for t in calc.sos.get_operation(glbl).get_direct_order_terms(order,order_base)]
#            #                                for order in range(calc.max_order+1) ]
#            op_term_reps[ glmap[glbl] ] = repcel.reps
#            repcache[glbl] = repcel
#
#    #OLD
#    #op_term_reps = { glmap[glbl]: [ [t.torep(None,None,"gate") for t in calc.sos.get_operation(glbl).get_direct_order_terms(order,order_base)]
#    #                                  for order in range(calc.max_order+1) ]
#    #                   for glbl in distinct_gateLabels }
#
#    #Similar with rho_terms and E_terms
#    cdef vector[vector[SVTermDirectCRep_ptr]] rho_term_reps;
#    if rholabel in repcache:
#        repcel = repcache[rholabel]
#        rho_term_reps = repcel.reps
#        for order in range(calc.max_order+1):
#            treps = rho_term_reps[order]
#            coeffs_array = calc.sos.get_prep(rholabel).get_direct_order_coeffs(order,order_base)
#            coeffs = <DCOMPLEX*?>(coeffs_array.data)
#            for i in range(treps.size()):
#                treps[i]._coeff = coeffs[i]
#                if resetTermWeights: treps[i]._magnitude = abs(coeffs[i])
#
#        #for order,treps in enumerate(rho_term_reps):
#        #    for coeff,trep in zip(calc.sos.get_prep(rholabel).get_direct_order_coeffs(order,order_base), treps):
#        #        trep.set_coeff(coeff)
#    else:
#        repcel = RepCacheEl(calc.max_order)
#        for order in range(calc.max_order+1):
#            reps_at_order = vector[SVTermDirectCRep_ptr](0)
#            for t in calc.sos.get_prep(rholabel).get_direct_order_terms(order,order_base):
#                rep = (<SVTermDirectRep?>t.torep(None,None,"prep"))
#                repcel.pyterm_references.append(rep)
#                reps_at_order.push_back( rep.c_term )
#            repcel.reps[order] = reps_at_order
#        rho_term_reps = repcel.reps
#        repcache[rholabel] = repcel
#
#        #OLD
#        #rho_term_reps = [ [t.torep(None,None,"prep") for t in calc.sos.get_prep(rholabel).get_direct_order_terms(order,order_base)]
#        #              for order in range(calc.max_order+1) ]
#        #repcache[rholabel] = rho_term_reps
#
#    #E_term_reps = []
#    cdef vector[vector[SVTermDirectCRep_ptr]] E_term_reps = vector[vector[SVTermDirectCRep_ptr]](0);
#    cdef SVTermDirectCRep_ptr cterm;
#    E_indices = [] # TODO: upgrade to C-type?
#    if all([ elbl in repcache for elbl in elabels]):
#        for order in range(calc.max_order+1):
#            reps_at_order = vector[SVTermDirectCRep_ptr](0) # the term reps for *all* the effect vectors
#            cur_indices = [] # the Evec-index corresponding to each term rep
#            for j,elbl in enumerate(elabels):
#                repcel = <RepCacheEl?>repcache[elbl]
#                #term_reps = [t.torep(None,None,"effect") for t in calc.sos.get_effect(elbl).get_direct_order_terms(order,order_base) ]
#
#                treps = repcel.reps[order]
#                coeffs_array = calc.sos.get_effect(elbl).get_direct_order_coeffs(order,order_base)
#                coeffs = <DCOMPLEX*?>(coeffs_array.data)
#                for i in range(treps.size()):
#                    treps[i]._coeff = coeffs[i]
#                    if resetTermWeights: treps[i]._magnitude = abs(coeffs[i])
#                    reps_at_order.push_back(treps[i])
#                cur_indices.extend( [j]*reps_at_order.size() )
#
#                #OLD
#                #term_reps = repcache[elbl][order]
#                #for coeff,trep in zip(calc.sos.get_effect(elbl).get_direct_order_coeffs(order,order_base), term_reps):
#                #    trep.set_coeff(coeff)
#                #cur_term_reps.extend( term_reps )
#                # cur_indices.extend( [j]*len(term_reps) )
#
#            E_term_reps.push_back(reps_at_order)
#            E_indices.append( cur_indices )
#            # E_term_reps.append( cur_term_reps )
#
#    else:
#        for elbl in elabels:
#            if elbl not in repcache: repcache[elbl] = RepCacheEl(calc.max_order) #[None]*(calc.max_order+1) # make sure there's room
#        for order in range(calc.max_order+1):
#            reps_at_order = vector[SVTermDirectCRep_ptr](0) # the term reps for *all* the effect vectors
#            cur_indices = [] # the Evec-index corresponding to each term rep
#            for j,elbl in enumerate(elabels):
#                repcel = <RepCacheEl?>repcache[elbl]
#                treps = vector[SVTermDirectCRep_ptr](0) # the term reps for *all* the effect vectors
#                for t in calc.sos.get_effect(elbl).get_direct_order_terms(order,order_base):
#                    rep = (<SVTermDirectRep?>t.torep(None,None,"effect"))
#                    repcel.pyterm_references.append(rep)
#                    treps.push_back( rep.c_term )
#                    reps_at_order.push_back( rep.c_term )
#                repcel.reps[order] = treps
#                cur_indices.extend( [j]*treps.size() )
#                #term_reps = [t.torep(None,None,"effect") for t in calc.sos.get_effect(elbl).get_direct_order_terms(order,order_base) ]
#                #repcache[elbl][order] = term_reps
#                #cur_term_reps.extend( term_reps )
#                #cur_indices.extend( [j]*len(term_reps) )
#            E_term_reps.push_back(reps_at_order)
#            E_indices.append( cur_indices )
#            #E_term_reps.append( cur_term_reps )
#
#    #convert to c-reps
#    cdef INT gi
#    #cdef vector[vector[SVTermDirectCRep_ptr]] rho_term_creps = rho_term_reps # already c-reps...
#    #cdef vector[vector[SVTermDirectCRep_ptr]] E_term_creps = E_term_reps # already c-reps...
#    #cdef unordered_map[INT, vector[vector[SVTermDirectCRep_ptr]]] gate_term_creps = op_term_reps # already c-reps...
#    #cdef vector[vector[SVTermDirectCRep_ptr]] rho_term_creps = sv_extract_cterms_direct(rho_term_reps,calc.max_order)
#    #cdef vector[vector[SVTermDirectCRep_ptr]] E_term_creps = sv_extract_cterms_direct(E_term_reps,calc.max_order)
#    #for gi,termrep_lists in op_term_reps.items():
#    #    gate_term_creps[gi] = sv_extract_cterms_direct(termrep_lists,calc.max_order)
#
#    E_cindices = vector[vector[INT]](<INT>len(E_indices))
#    for ii,inds in enumerate(E_indices):
#        E_cindices[ii] = vector[INT](<INT>len(inds))
#        for jj,indx in enumerate(inds):
#            E_cindices[ii][jj] = <INT>indx
#
#    #Note: term calculator "dim" is the full density matrix dim
#    stateDim = int(round(np.sqrt(calc.dim)))
#    if debug is not None:
#        debug['tstartup'] += pytime.time()-t0
#        t0 = pytime.time()
#
#    #Call C-only function (which operates with C-representations only)
#    cdef vector[float] debugvec = vector[float](10)
#    debugvec[0] = 0.0
#    cdef vector[DCOMPLEX] prs = sv_prs_directly(
#        cgatestring, rho_term_reps, op_term_reps, E_term_reps,
#        #cgatestring, rho_term_creps, gate_term_creps, E_term_creps,
#        E_cindices, numEs, calc.max_order, stateDim, <bool>fastmode, &remainingWeight, remaingingWeightTol, debugvec)
#
#    debug['total'] += debugvec[0]
#    debug['t1'] += debugvec[1]
#    debug['t2'] += debugvec[2]
#    debug['t3'] += debugvec[3]
#    debug['n1'] += debugvec[4]
#    debug['n2'] += debugvec[5]
#    debug['n3'] += debugvec[6]
#    debug['t4'] += debugvec[7]
#    debug['n4'] += debugvec[8]
#    #if not all([ abs(prs[i].imag) < 1e-4 for i in range(<INT>prs.size()) ]):
#    #    print("ERROR: prs = ",[ prs[i] for i in range(<INT>prs.size()) ])
#    #assert(all([ abs(prs[i].imag) < 1e-6 for i in range(<INT>prs.size()) ]))
#    return [ prs[i].real for i in range(<INT>prs.size()) ] # TODO: make this into a numpy array? - maybe pass array to fill to sv_prs_directy above?
#
#
#cdef vector[DCOMPLEX] sv_prs_directly(
#    vector[INT]& circuit, vector[vector[SVTermDirectCRep_ptr]] rho_term_reps,
#    unordered_map[INT, vector[vector[SVTermDirectCRep_ptr]]] op_term_reps,
#    vector[vector[SVTermDirectCRep_ptr]] E_term_reps, vector[vector[INT]] E_term_indices,
#    INT numEs, INT max_order, INT dim, bool fastmode, vector[double]* remainingWeight, double remTol, vector[float]& debugvec):
#
#    #NOTE: circuit and gate_terms use *integers* as operation labels, not Label objects, to speed
#    # lookups and avoid weird string conversion stuff with Cython
#
#    cdef INT N = len(circuit)
#    cdef INT* p = <INT*>malloc((N+2) * sizeof(INT))
#    cdef INT i,j,k,order,nTerms
#    cdef INT gn
#
#    cdef INT t0 = time.clock()
#    cdef INT t, n, nPaths; #for below
#
#    cdef sv_innerloopfn_direct_ptr innerloop_fn;
#    if fastmode:
#        innerloop_fn = sv_pr_directly_innerloop_savepartials
#    else:
#        innerloop_fn = sv_pr_directly_innerloop
#
#    #extract raw data from gate_terms dictionary-of-lists for faster lookup
#    #gate_term_prefactors = np.empty( (nOperations,max_order+1,dim,dim)
#    #cdef unordered_map[INT, vector[vector[unordered_map[INT, complex]]]] gate_term_coeffs
#    #cdef vector[vector[unordered_map[INT, complex]]] rho_term_coeffs
#    #cdef vector[vector[unordered_map[INT, complex]]] E_term_coeffs
#    #cdef vector[vector[INT]] E_indices
#
#    cdef vector[INT]* Einds
#    cdef vector[vector_SVTermDirectCRep_ptr_ptr] factor_lists
#
#    assert(max_order <= 2) # only support this partitioning below (so far)
#
#    cdef vector[DCOMPLEX] prs = vector[DCOMPLEX](numEs)
#
#    for order in range(max_order+1):
#        #print("DB: pr_as_poly order=",order)
#
#        #for p in partition_into(order, N):
#        for i in range(N+2): p[i] = 0 # clear p
#        factor_lists = vector[vector_SVTermDirectCRep_ptr_ptr](N+2)
#
#        if order == 0:
#            #inner loop(p)
#            #factor_lists = [ gate_terms[glbl][pi] for glbl,pi in zip(circuit,p) ]
#            t = time.clock()
#            factor_lists[0] = &rho_term_reps[p[0]]
#            for k in range(N):
#                gn = circuit[k]
#                factor_lists[k+1] = &op_term_reps[circuit[k]][p[k+1]]
#                #if factor_lists[k+1].size() == 0: continue # WHAT???
#            factor_lists[N+1] = &E_term_reps[p[N+1]]
#            Einds = &E_term_indices[p[N+1]]
#
#            #print("Part0 ",p)
#            nPaths = innerloop_fn(factor_lists,Einds,&prs,dim,remainingWeight,0.0) #remTol) # force 0-order
#            debugvec[1] += float(time.clock() - t)/time.CLOCKS_PER_SEC
#            debugvec[4] += nPaths
#
#        elif order == 1:
#            t = time.clock(); n=0
#            for i in range(N+2):
#                p[i] = 1
#                #inner loop(p)
#                factor_lists[0] = &rho_term_reps[p[0]]
#                for k in range(N):
#                    gn = circuit[k]
#                    factor_lists[k+1] = &op_term_reps[gn][p[k+1]]
#                    #if len(factor_lists[k+1]) == 0: continue #WHAT???
#                factor_lists[N+1] = &E_term_reps[p[N+1]]
#                Einds = &E_term_indices[p[N+1]]
#
#                #print "DB: Order1 "
#                nPaths = innerloop_fn(factor_lists,Einds,&prs,dim,remainingWeight,0.0) #remTol) # force 1st-order
#                p[i] = 0
#                n += nPaths
#            debugvec[2] += float(time.clock() - t)/time.CLOCKS_PER_SEC
#            debugvec[5] += n
#
#        elif order == 2:
#            t = time.clock(); n=0
#            for i in range(N+2):
#                p[i] = 2
#                #inner loop(p)
#                factor_lists[0] = &rho_term_reps[p[0]]
#                for k in range(N):
#                    gn = circuit[k]
#                    factor_lists[k+1] = &op_term_reps[circuit[k]][p[k+1]]
#                    #if len(factor_lists[k+1]) == 0: continue # WHAT???
#                factor_lists[N+1] = &E_term_reps[p[N+1]]
#                Einds = &E_term_indices[p[N+1]]
#
#                nPaths = innerloop_fn(factor_lists,Einds,&prs,dim,remainingWeight,remTol)
#                p[i] = 0
#                n += nPaths
#
#            debugvec[3] += float(time.clock() - t)/time.CLOCKS_PER_SEC
#            debugvec[6] += n
#            t = time.clock(); n=0
#
#            for i in range(N+2):
#                p[i] = 1
#                for j in range(i+1,N+2):
#                    p[j] = 1
#                    #inner loop(p)
#                    factor_lists[0] = &rho_term_reps[p[0]]
#                    for k in range(N):
#                        gn = circuit[k]
#                        factor_lists[k+1] = &op_term_reps[circuit[k]][p[k+1]]
#                        #if len(factor_lists[k+1]) == 0: continue #WHAT???
#                    factor_lists[N+1] = &E_term_reps[p[N+1]]
#                    Einds = &E_term_indices[p[N+1]]
#
#                    nPaths = innerloop_fn(factor_lists,Einds,&prs,dim,remainingWeight,remTol)
#                    p[j] = 0
#                    n += nPaths
#                p[i] = 0
#            debugvec[7] += float(time.clock() - t)/time.CLOCKS_PER_SEC
#            debugvec[8] += n
#
#        else:
#            assert(False) # order > 2 not implemented yet...
#
#    free(p)
#
#    debugvec[0] += float(time.clock() - t0)/time.CLOCKS_PER_SEC
#    return prs
#
#
#
#cdef INT sv_pr_directly_innerloop(vector[vector_SVTermDirectCRep_ptr_ptr] factor_lists, vector[INT]* Einds,
#                                   vector[DCOMPLEX]* prs, INT dim, vector[double]* remainingWeight, double remainingWeightTol):
#    #print("DB partition = ","listlens = ",[len(fl) for fl in factor_lists])
#
#    cdef INT i,j,Ei
#    cdef double complex scale, val, newval, pLeft, pRight, p
#    cdef double wt, cwt
#    cdef int nPaths = 0
#
#    cdef SVTermDirectCRep* factor
#
#    cdef INT nFactorLists = factor_lists.size() # may need to recompute this after fast-mode
#    cdef INT* factorListLens = <INT*>malloc(nFactorLists * sizeof(INT))
#    cdef INT last_index = nFactorLists-1
#
#    for i in range(nFactorLists):
#        factorListLens[i] = factor_lists[i].size()
#        if factorListLens[i] == 0:
#            free(factorListLens)
#            return 0 # nothing to loop over! - (exit before we allocate more)
#
#    cdef double complex coeff   # THESE are only real changes from "as_poly"
#    cdef double complex result  # version of this function (where they are PolyCRep type)
#
#    cdef SVStateCRep *prop1 = new SVStateCRep(dim)
#    cdef SVStateCRep *prop2 = new SVStateCRep(dim)
#    cdef SVStateCRep *tprop
#    cdef SVEffectCRep* EVec
#
#    cdef INT* b = <INT*>malloc(nFactorLists * sizeof(INT))
#    for i in range(nFactorLists): b[i] = 0
#
#    assert(nFactorLists > 0), "Number of factor lists must be > 0!"
#
#    #for factors in _itertools.product(*factor_lists):
#    while(True):
#        final_factor_indx = b[last_index]
#        Ei = deref(Einds)[final_factor_indx] #final "factor" index == E-vector index
#        wt = deref(remainingWeight)[Ei]
#        if remainingWeightTol == 0.0 or wt > remainingWeightTol: #if we need this "path"
#            # In this loop, b holds "current" indices into factor_lists
#            factor = deref(factor_lists[0])[b[0]] # the last factor (an Evec)
#            coeff = factor._coeff
#            cwt = factor._magnitude
#
#            for i in range(1,nFactorLists):
#                coeff *= deref(factor_lists[i])[b[i]]._coeff
#                cwt *= deref(factor_lists[i])[b[i]]._magnitude
#
#            #pLeft / "pre" sim
#            factor = deref(factor_lists[0])[b[0]] # 0th-factor = rhoVec
#            prop1.copy_from(factor._pre_state)
#            for j in range(<INT>factor._pre_ops.size()):
#                factor._pre_ops[j].acton(prop1,prop2)
#                tprop = prop1; prop1 = prop2; prop2 = tprop
#            for i in range(1,last_index):
#                factor = deref(factor_lists[i])[b[i]]
#                for j in range(<INT>factor._pre_ops.size()):
#                    factor._pre_ops[j].acton(prop1,prop2)
#                    tprop = prop1; prop1 = prop2; prop2 = tprop # final state in prop1
#            factor = deref(factor_lists[last_index])[b[last_index]] # the last factor (an Evec)
#
#        	# can't propagate effects, so effect's post_ops are constructed to act on *state*
#            EVec = factor._post_effect
#            for j in range(<INT>factor._post_ops.size()):
#                rhoVec = factor._post_ops[j].acton(prop1,prop2)
#                tprop = prop1; prop1 = prop2; prop2 = tprop # final state in prop1
#            pLeft = EVec.amplitude(prop1)
#
#            #pRight / "post" sim
#            factor = deref(factor_lists[0])[b[0]] # 0th-factor = rhoVec
#            prop1.copy_from(factor._post_state)
#            for j in range(<INT>factor._post_ops.size()):
#                factor._post_ops[j].acton(prop1,prop2)
#                tprop = prop1; prop1 = prop2; prop2 = tprop # final state in prop1
#            for i in range(1,last_index):
#                factor = deref(factor_lists[i])[b[i]]
#                for j in range(<INT>factor._post_ops.size()):
#                    factor._post_ops[j].acton(prop1,prop2)
#                    tprop = prop1; prop1 = prop2; prop2 = tprop # final state in prop1
#            factor = deref(factor_lists[last_index])[b[last_index]] # the last factor (an Evec)
#
#            EVec = factor._pre_effect
#            for j in range(<INT>factor._pre_ops.size()):
#                factor._pre_ops[j].acton(prop1,prop2)
#                tprop = prop1; prop1 = prop2; prop2 = tprop # final state in prop1
#            pRight = EVec.amplitude(prop1).conjugate()
#
#            #Add result to appropriate poly
#            result = coeff * pLeft * pRight
#            deref(prs)[Ei] = deref(prs)[Ei] + result #TODO - see why += doesn't work here
#            deref(remainingWeight)[Ei] = wt - cwt # "weight" of this path
#            nPaths += 1 # just for debuggins
#
#        #increment b ~ itertools.product & update vec_index_noop = np.dot(self.multipliers, b)
#        for i in range(nFactorLists-1,-1,-1):
#            if b[i]+1 < factorListLens[i]:
#                b[i] += 1
#                break
#            else:
#                b[i] = 0
#        else:
#            break # can't increment anything - break while(True) loop
#
#    #Clenaup: free allocated memory
#    del prop1
#    del prop2
#    free(factorListLens)
#    free(b)
#    return nPaths
#
#
#cdef INT sv_pr_directly_innerloop_savepartials(vector[vector_SVTermDirectCRep_ptr_ptr] factor_lists,
#                                                vector[INT]* Einds, vector[DCOMPLEX]* prs, INT dim,
#                                                vector[double]* remainingWeight, double remainingWeightTol):
#    #print("DB partition = ","listlens = ",[len(fl) for fl in factor_lists])
#
#    cdef INT i,j,Ei
#    cdef double complex scale, val, newval, pLeft, pRight, p
#
#    cdef INT incd
#    cdef SVTermDirectCRep* factor
#
#    cdef INT nFactorLists = factor_lists.size() # may need to recompute this after fast-mode
#    cdef INT* factorListLens = <INT*>malloc(nFactorLists * sizeof(INT))
#    cdef INT last_index = nFactorLists-1
#
#    for i in range(nFactorLists):
#        factorListLens[i] = factor_lists[i].size()
#        if factorListLens[i] == 0:
#            free(factorListLens)
#            return 0 # nothing to loop over! (exit before we allocate anything else)
#
#    cdef double complex coeff
#    cdef double complex result
#
#    #fast mode
#    cdef vector[SVStateCRep*] leftSaved = vector[SVStateCRep_ptr](nFactorLists-1)  # saved[i] is state after i-th
#    cdef vector[SVStateCRep*] rightSaved = vector[SVStateCRep_ptr](nFactorLists-1) # factor has been applied
#    cdef vector[DCOMPLEX] coeffSaved = vector[DCOMPLEX](nFactorLists-1)
#    cdef SVStateCRep *shelved = new SVStateCRep(dim)
#    cdef SVStateCRep *prop2 = new SVStateCRep(dim) # prop2 is always a temporary allocated state not owned by anything else
#    cdef SVStateCRep *prop1
#    cdef SVStateCRep *tprop
#    cdef SVEffectCRep* EVec
#
#    cdef INT* b = <INT*>malloc(nFactorLists * sizeof(INT))
#    for i in range(nFactorLists): b[i] = 0
#    assert(nFactorLists > 0), "Number of factor lists must be > 0!"
#
#    incd = 0
#
#    #Fill saved arrays with allocated states
#    for i in range(nFactorLists-1):
#        leftSaved[i] = new SVStateCRep(dim)
#        rightSaved[i] = new SVStateCRep(dim)
#
#    #for factors in _itertools.product(*factor_lists):
#    #for incd,fi in incd_product(*[range(len(l)) for l in factor_lists]):
#    while(True):
#        # In this loop, b holds "current" indices into factor_lists
#        #print "DB: iter-product BEGIN"
#
#        if incd == 0: # need to re-evaluate rho vector
#            #print "DB: re-eval at incd=0"
#            factor = deref(factor_lists[0])[b[0]]
#
#            #print "DB: re-eval left"
#            prop1 = leftSaved[0] # the final destination (prop2 is already alloc'd)
#            prop1.copy_from(factor._pre_state)
#            for j in range(<INT>factor._pre_ops.size()):
#                #print "DB: re-eval left item"
#                factor._pre_ops[j].acton(prop1,prop2)
#                tprop = prop1; prop1 = prop2; prop2 = tprop # swap prop1 <-> prop2
#            rhoVecL = prop1
#            leftSaved[0] = prop1 # final state -> saved
#            # (prop2 == the other allocated state)
#
#            #print "DB: re-eval right"
#            prop1 = rightSaved[0] # the final destination (prop2 is already alloc'd)
#            prop1.copy_from(factor._post_state)
#            for j in range(<INT>factor._post_ops.size()):
#                #print "DB: re-eval right item"
#                factor._post_ops[j].acton(prop1,prop2)
#                tprop = prop1; prop1 = prop2; prop2 = tprop # swap prop1 <-> prop2
#            rhoVecR = prop1
#            rightSaved[0] = prop1 # final state -> saved
#            # (prop2 == the other allocated state)
#
#            #print "DB: re-eval coeff"
#            coeff = factor._coeff
#            coeffSaved[0] = coeff
#            incd += 1
#        else:
#            #print "DB: init from incd"
#            rhoVecL = leftSaved[incd-1]
#            rhoVecR = rightSaved[incd-1]
#            coeff = coeffSaved[incd-1]
#
#        # propagate left and right states, saving as we go
#        for i in range(incd,last_index):
#            #print "DB: propagate left begin"
#            factor = deref(factor_lists[i])[b[i]]
#            prop1 = leftSaved[i] # destination
#            prop1.copy_from(rhoVecL) #starting state
#            for j in range(<INT>factor._pre_ops.size()):
#                #print "DB: propagate left item"
#                factor._pre_ops[j].acton(prop1,prop2)
#                tprop = prop1; prop1 = prop2; prop2 = tprop
#            rhoVecL = prop1
#            leftSaved[i] = prop1
#            # (prop2 == the other allocated state)
#
#            #print "DB: propagate right begin"
#            prop1 = rightSaved[i] # destination
#            prop1.copy_from(rhoVecR) #starting state
#            for j in range(<INT>factor._post_ops.size()):
#                #print "DB: propagate right item"
#                factor._post_ops[j].acton(prop1,prop2)
#                tprop = prop1; prop1 = prop2; prop2 = tprop
#            rhoVecR = prop1
#            rightSaved[i] = prop1
#            # (prop2 == the other allocated state)
#
#            #print "DB: propagate coeff mult"
#            coeff *= factor._coeff
#            coeffSaved[i] = coeff
#
#        # for the last index, no need to save, and need to construct
#        # and apply effect vector
#        prop1 = shelved # so now prop1 (and prop2) are alloc'd states
#
#        #print "DB: left ampl"
#        factor = deref(factor_lists[last_index])[b[last_index]] # the last factor (an Evec)
#        EVec = factor._post_effect
#        prop1.copy_from(rhoVecL) # initial state (prop2 already alloc'd)
#        for j in range(<INT>factor._post_ops.size()):
#            factor._post_ops[j].acton(prop1,prop2)
#            tprop = prop1; prop1 = prop2; prop2 = tprop
#        pLeft = EVec.amplitude(prop1) # output in prop1, so this is final amplitude
#
#        #print "DB: right ampl"
#        EVec = factor._pre_effect
#        prop1.copy_from(rhoVecR)
#        for j in range(<INT>factor._pre_ops.size()):
#            factor._pre_ops[j].acton(prop1,prop2)
#            tprop = prop1; prop1 = prop2; prop2 = tprop
#        pRight = EVec.amplitude(prop1).conjugate()
#
#        shelved = prop1 # return prop1 to the "shelf" since we'll use prop1 for other things next
#
#        #print "DB: final block"
#        #print "DB running coeff = ",dict(coeff._coeffs)
#        #print "DB factor coeff = ",dict(factor._coeff._coeffs)
#        result = coeff * factor._coeff
#        #print "DB result = ",dict(result._coeffs)
#        result *= pLeft * pRight
#        final_factor_indx = b[last_index]
#        Ei = deref(Einds)[final_factor_indx] #final "factor" index == E-vector index
#        deref(prs)[Ei] += result
#        #print "DB prs[",INT(Ei),"] = ",dict(deref(prs)[Ei]._coeffs)
#
#        #assert(debug < 100) #DEBUG
#        #print "DB: end product loop"
#
#        #increment b ~ itertools.product & update vec_index_noop = np.dot(self.multipliers, b)
#        for i in range(nFactorLists-1,-1,-1):
#            if b[i]+1 < factorListLens[i]:
#                b[i] += 1; incd = i
#                break
#            else:
#                b[i] = 0
#        else:
#            break # can't increment anything - break while(True) loop
#
#    #Cleanup: free allocated memory
#    for i in range(nFactorLists-1):
#        del leftSaved[i]
#        del rightSaved[i]
#    del prop2
#    del shelved
#    free(factorListLens)
#    free(b)
#    return 0 #TODO: fix nPaths


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


def SB_prs_as_polys(calc, rholabel, elabels, circuit, comm=None, memLimit=None, fastmode=True):

    # Create gatelable -> int mapping to be used throughout
    distinct_gateLabels = sorted(set(circuit))
    glmap = { gl: i for i,gl in enumerate(distinct_gateLabels) }

    # Convert circuit to a vector of ints
    cdef INT i
    cdef vector[INT] cgatestring
    for gl in circuit:
        cgatestring.push_back(<INT>glmap[gl])

    cdef INT mpv = calc.Np # max_poly_vars
    #cdef INT mpo = calc.max_order*2 #max_poly_order
    cdef INT vpi = calc.poly_vindices_per_int
    cdef INT order;
    cdef INT numEs = len(elabels)

    # Construct dict of gate term reps, then *convert* to c-reps, as this
    #  keeps alive the non-c-reps which keep the c-reps from being deallocated...
    op_term_reps = { glmap[glbl]: [ [t.torep() for t in calc.sos.get_operation(glbl).get_taylor_order_terms(order, mpv)]
                                      for order in range(calc.max_order+1) ]
                       for glbl in distinct_gateLabels }

    #Similar with rho_terms and E_terms
    rho_term_reps = [ [t.torep() for t in calc.sos.get_prep(rholabel).get_taylor_order_terms(order, mpv)]
                      for order in range(calc.max_order+1) ]

    E_term_reps = []
    E_indices = []
    for order in range(calc.max_order+1):
        cur_term_reps = [] # the term reps for *all* the effect vectors
        cur_indices = [] # the Evec-index corresponding to each term rep
        for i,elbl in enumerate(elabels):
            term_reps = [t.torep() for t in calc.sos.get_effect(elbl).get_taylor_order_terms(order, mpv) ]
            cur_term_reps.extend( term_reps )
            cur_indices.extend( [i]*len(term_reps) )
        E_term_reps.append( cur_term_reps )
        E_indices.append( cur_indices )


    #convert to c-reps
    cdef INT gi
    cdef vector[vector[SBTermCRep_ptr]] rho_term_creps = sb_extract_cterms(rho_term_reps,calc.max_order)
    cdef vector[vector[SBTermCRep_ptr]] E_term_creps = sb_extract_cterms(E_term_reps,calc.max_order)
    cdef unordered_map[INT, vector[vector[SBTermCRep_ptr]]] gate_term_creps
    for gi,termrep_lists in op_term_reps.items():
        gate_term_creps[gi] = sb_extract_cterms(termrep_lists,calc.max_order)

    E_cindices = vector[vector[INT]](<INT>len(E_indices))
    for ii,inds in enumerate(E_indices):
        E_cindices[ii] = vector[INT](<INT>len(inds))
        for jj,indx in enumerate(inds):
            E_cindices[ii][jj] = <INT>indx

    # Assume when we calculate terms, that "dimension" of Model is
    # a full vectorized-density-matrix dimension, so nqubits is:
    cdef INT nqubits = <INT>(np.log2(calc.dim)//2)

    #Call C-only function (which operates with C-representations only)
    cdef vector[PolyCRep*] polys = sb_prs_as_polys(
        cgatestring, rho_term_creps, gate_term_creps, E_term_creps,
        E_cindices, numEs, calc.max_order, mpv, vpi, nqubits, <bool>fastmode)

    return [ PolyRep_from_allocd_PolyCRep(polys[i]) for i in range(<INT>polys.size()) ]


cdef vector[PolyCRep*] sb_prs_as_polys(
    vector[INT]& circuit, vector[vector[SBTermCRep_ptr]] rho_term_reps,
    unordered_map[INT, vector[vector[SBTermCRep_ptr]]] op_term_reps,
    vector[vector[SBTermCRep_ptr]] E_term_reps, vector[vector[INT]] E_term_indices,
    INT numEs, INT max_order, INT max_poly_vars, INT vindices_per_int, INT nqubits, bool fastmode):

    #NOTE: circuit and gate_terms use *integers* as operation labels, not Label objects, to speed
    # lookups and avoid weird string conversion stuff with Cython

    cdef INT N = len(circuit)
    cdef INT* p = <INT*>malloc((N+2) * sizeof(INT))
    cdef INT i,j,k,order,nTerms
    cdef INT gn

    cdef sb_innerloopfn_ptr innerloop_fn;
    if fastmode:
        innerloop_fn = sb_pr_as_poly_innerloop_savepartials
    else:
        innerloop_fn = sb_pr_as_poly_innerloop

    #extract raw data from gate_terms dictionary-of-lists for faster lookup
    #gate_term_prefactors = np.empty( (nOperations,max_order+1,dim,dim)
    #cdef unordered_map[INT, vector[vector[unordered_map[INT, complex]]]] gate_term_coeffs
    #cdef vector[vector[unordered_map[INT, complex]]] rho_term_coeffs
    #cdef vector[vector[unordered_map[INT, complex]]] E_term_coeffs
    #cdef vector[vector[INT]] E_indices

    cdef vector[INT]* Einds
    cdef vector[vector_SBTermCRep_ptr_ptr] factor_lists

    assert(max_order <= 2) # only support this partitioning below (so far)

    cdef vector[PolyCRep_ptr] prps = vector[PolyCRep_ptr](numEs)
    for i in range(numEs):
        prps[i] = new PolyCRep(unordered_map[PolyVarsIndex,complex](), max_poly_vars, vindices_per_int)
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
            #factor_lists = [ gate_terms[glbl][pi] for glbl,pi in zip(circuit,p) ]
            factor_lists[0] = &rho_term_reps[p[0]]
            for k in range(N):
                gn = circuit[k]
                factor_lists[k+1] = &op_term_reps[circuit[k]][p[k+1]]
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
                    gn = circuit[k]
                    factor_lists[k+1] = &op_term_reps[gn][p[k+1]]
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
                    gn = circuit[k]
                    factor_lists[k+1] = &op_term_reps[circuit[k]][p[k+1]]
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
                        gn = circuit[k]
                        factor_lists[k+1] = &op_term_reps[circuit[k]][p[k+1]]
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

        # can't propagate effects, so effect's post_ops are constructed to act on *state*
        EVec = factor._post_effect
        for j in range(<INT>factor._pre_ops.size()):
            rhoVec = factor._pre_ops[j].acton(prop1,prop2)
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
        for j in range(<INT>factor._post_ops.size()):
            factor._post_ops[j].acton(prop1,prop2)
            tprop = prop1; prop1 = prop2; prop2 = tprop # final state in prop1
        pRight = EVec.amplitude(prop1).conjugate()


        #Add result to appropriate poly
        result = coeff  # use a reference?
        result.scale(pLeft * pRight)
        final_factor_indx = b[last_index]
        Ei = deref(Einds)[final_factor_indx] #final "factor" index == E-vector index
        deref(prps)[Ei].add_inplace(result)

        #increment b ~ itertools.product & update vec_index_noop = np.dot(self.multipliers, b)
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
        for j in range(<INT>factor._pre_ops.size()):
            factor._pre_ops[j].acton(prop1,prop2)
            tprop = prop1; prop1 = prop2; prop2 = tprop
        pLeft = EVec.amplitude(prop1) # output in prop1, so this is final amplitude

        #print "DB: right ampl"
        EVec = factor._pre_effect
        prop1.copy_from(rhoVecR)
        pRight = EVec.amplitude(prop1)
        #DEBUG print "  - begin: ",complex(pRight)
        for j in range(<INT>factor._post_ops.size()):
            #DEBUG print " - state = ", [ prop1._smatrix[ii] for ii in range(2*2)]
            #DEBUG print "         = ", [ prop1._pvectors[ii] for ii in range(2)]
            #DEBUG print "         = ", [ prop1._amps[ii] for ii in range(1)]
            factor._post_ops[j].acton(prop1,prop2)
            #DEBUG print " - action with ", [ (<SBOpCRep_Clifford*>factor._pre_ops[j])._smatrix_inv[ii] for ii in range(2*2)]
            #DEBUG print " - action with ", [ (<SBOpCRep_Clifford*>factor._pre_ops[j])._svector_inv[ii] for ii in range(2)]
            #DEBUG print " - action with ", [ (<SBOpCRep_Clifford*>factor._pre_ops[j])._unitary_adj[ii] for ii in range(2*2)]
            tprop = prop1; prop1 = prop2; prop2 = tprop
            pRight = EVec.amplitude(prop1)
            #DEBUG print "  - prop ",INT(j)," = ",complex(pRight)
            #DEBUG print " - post state = ", [ prop1._smatrix[ii] for ii in range(2*2)]
            #DEBUG print "              = ", [ prop1._pvectors[ii] for ii in range(2)]
            #DEBUG print "              = ", [ prop1._amps[ii] for ii in range(1)]

        pRight = EVec.amplitude(prop1).conjugate()

        shelved = prop1 # return prop1 to the "shelf" since we'll use prop1 for other things next

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

        #increment b ~ itertools.product & update vec_index_noop = np.dot(self.multipliers, b)
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

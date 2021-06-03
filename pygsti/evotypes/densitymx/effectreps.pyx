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
import numpy as _np
from ...models.statespace import StateSpace as _StateSpace
from ...tools import matrixtools as _mt


cdef class EffectRep(_basereps_cython.EffectRep):
    def __cinit__(self):
        self.c_effect = NULL
        self.state_space = None

    def __dealloc__(self):
        if self.c_effect != NULL:
            del self.c_effect

    def __reduce__(self):
        return (EffectRep, ())

    @property
    def dim(self):
        return self.state_space.dim

    def probability(self, StateRep state not None):
        #unnecessary (just put in signature): cdef StateRep st = <StateRep?>state
        return self.c_effect.probability(state.c_state)


cdef class EffectRepConjugatedState(EffectRep):
    cdef public StateRep state_rep

    def __cinit__(self, StateRep state_rep):
        self.state_rep = state_rep
        self.c_effect = new EffectCRep_Dense(<double*>self.state_rep.data.data,
                                               <INT>self.state_rep.data.shape[0])
        self.state_space = state_rep.state_space

    def __str__(self):
        return str([ (<EffectCRep_Dense*>self.c_effect)._dataptr[i] for i in range(self.c_effect._dim)])

    def __reduce__(self):
        return (EffectRepConjugatedState, (self.state_rep,))

    def to_dense(self):
        return self.state_rep.to_dense()


cdef class EffectRepComputational(EffectRep):
    cdef public _np.ndarray zvals

    def __cinit__(self, _np.ndarray[_np.int64_t, ndim=1, mode='c'] zvals, basis, state_space):

        state_space = _StateSpace.cast(state_space)
        assert(state_space.num_qubits == len(zvals))
        assert(basis.name == 'pp'), "Only Pauli-product computational effect vectors are currently supported"

        cdef INT nfactors = zvals.shape[0]
        cdef double abs_elval = 1/(_np.sqrt(2)**nfactors)
        cdef INT base = 1
        cdef INT zvals_int = 0
        for i in range(nfactors):
            zvals_int += base * zvals[i]
            base = base << 1 # *= 2
        self.zvals = zvals
        self.c_effect = new EffectCRep_Computational(nfactors, zvals_int, abs_elval, state_space.dim)
        self.state_space = state_space

    def __reduce__(self):
        return (EffectRepComputational, (self.zvals, self.state_space))

    def to_dense(self, outvec=None):
        return _mt.zvals_int64_to_dense((<EffectCRep_Computational*>self.c_effect)._zvals_int,
                                        self.zvals.shape[0], outvec, False,
                                        (<EffectCRep_Computational*>self.c_effect)._abs_elval)


cdef class EffectRepTensorProduct(EffectRep):
    cdef public object povm_factors
    cdef public object effect_labels
    cdef public _np.ndarray kron_array
    cdef public _np.ndarray factor_dims

    def __init__(self, povm_factors, effect_labels, state_space):
        #Arrays for speeding up kron product in effect reps
        cdef INT max_factor_dim = max(fct.state_space.dim for fct in povm_factors)
        cdef _np.ndarray[double, ndim=2, mode='c'] kron_array = \
            _np.ascontiguousarray(_np.empty((len(povm_factors), max_factor_dim), 'd'))
        cdef _np.ndarray[_np.int64_t, ndim=1, mode='c'] factor_dims = \
            _np.ascontiguousarray(_np.array([fct.state_space.dim for fct in povm_factors], _np.int64))

        cdef INT dim = _np.product(factor_dims)
        cdef INT nfactors = len(povm_factors)
        self.povm_factors = povm_factors
        self.effect_labels = effect_labels
        self.kron_array = kron_array
        self.factor_dims = factor_dims
        self.c_effect = new EffectCRep_TensorProd(<double*>kron_array.data,
                                                  <INT*>factor_dims.data,
                                                  nfactors, max_factor_dim, dim)
        self.state_space = _StateSpace.cast(state_space)
        assert(self.state_space.dim == dim)
        self.factor_effects_have_changed()  # computes self.kron_array

    def __reduce__(self):
        return (EffectRepTensorProduct, (self.povm_factors, self.effect_labels, self.state_space))

    def _fill_fast_kron(self):
        """ Fills in self._fast_kron_array based on current self.factors """
        for i, (factor_dim, Elbl) in enumerate(zip(self.factor_dims, self.effect_labels)):
            self.kron_array[i][0:factor_dim] = self.povm_factors[i][Elbl].to_dense()

    def factor_effects_have_changed(self):
        self._fill_fast_kron()  # updates effect reps

    def to_dense(self, outvec=None):  # taken from slow version - CONSOLIDATE?

        if outvec is None:
            outvec = _np.zeros(self.state_space.dim, 'd')

        N = self.state_space.dim
        nfactors = len(self.povm_factors)
        #Put last factor at end of outvec
        k = nfactors - 1  # last factor
        off = N - self.factor_dims[k]  # offset into outvec
        for i in range(self.factor_dims[k]):
            outvec[off + i] = self.kron_array[k, i]
        sz = self.factor_dims[k]

        #Repeatedly scale&copy last "sz" elements of outputvec forward
        # (as many times as there are elements in the current factor array)
        # - but multiply *in-place* the last "sz" elements.
        for k in range(nfactors - 2, -1, -1):  # for all but the last factor
            off = N - sz * self.factor_dims[k]
            endoff = N - sz

            #For all but the final element of self.kron_array[k,:],
            # mult&copy final sz elements of outvec into position
            for j in range(self.factor_dims[k] - 1):
                mult = self.kron_array[k, j]
                for i in range(sz):
                    outvec[off + i] = mult * outvec[endoff + i]
                off += sz

            #Last element: in-place mult
            #assert(off == endoff)
            mult = self.kron_array[k, self.factor_dims[k] - 1]
            for i in range(sz):
                outvec[endoff + i] *= mult
            sz *= self.factor_dims[k]

        return outvec


cdef class EffectRepComposed(EffectRep):
    cdef public OpRep op_rep
    cdef public EffectRep effect_rep
    cdef public object op_id

    def __cinit__(self, OpRep op_rep not None, EffectRep effect_rep not None, op_id, state_space):
        cdef INT dim = effect_rep.c_effect._dim
        self.op_id = op_id
        self.op_rep = op_rep
        self.effect_rep = effect_rep
        self.c_effect = new EffectCRep_Composed(op_rep.c_rep,
                                                effect_rep.c_effect,
                                                <INT>op_id, dim)
        self.state_space = _StateSpace.cast(state_space)
        assert(self.state_space.dim == dim)

    def __reduce__(self):
        return (EffectRepComposed, (self.op_rep, self.effect_rep, self.op_id, self.state_space))

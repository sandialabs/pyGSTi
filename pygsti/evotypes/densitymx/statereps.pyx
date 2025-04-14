# encoding: utf-8
# cython: profile=False
# cython: linetrace=False

#***************************************************************************************************
# Copyright 2015, 2019, 2025 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

import numpy as _np
import functools as _functools
from ...baseobjs.statespace import StateSpace as _StateSpace
from ...tools import basistools as _bt
from ...tools import optools as _ot
from ...tools import fastcalc as _fastcalc

from .opreps cimport OpRep


cdef class StateRep(_basereps_cython.StateRep):
    def __cinit__(self):
        self.data = None
        self.c_state = NULL
        self.state_space = None

    def _cinit_base(self, _np.ndarray[double, ndim=1] data, state_space):
        # Not __init__ or __cinit__ - to be called from derived class __cinit__ fns to initialize base
        self.data = _np.require(data.copy(), requirements=['OWNDATA', 'C_CONTIGUOUS'])
        self.c_state = new StateCRep(<double*>self.data.data, <INT>self.data.shape[0], <bool>0)
        self.state_space = _StateSpace.cast(state_space)
        assert(len(self.data) == self.state_space.dim)

    def __reduce__(self):
        return (StateRep, (), (self.data.flags.writeable,))

    def __pygsti_reduce__(self):
        return self.__reduce__()

    def __setstate__(self, state):
        writeable, = state
        self.data.flags.writeable = writeable

    def copy_from(self, other):
        self.data[:] = other.data[:]

    def actionable_staterep(self):
        # return a state rep that can be acted on by op reps or mapped to
        # a probability/amplitude by POVM effect reps.
        return self  # for most classes, the rep itself is actionable

    def to_dense(self, on_space):
        if on_space not in ('minimal', 'HilbertSchmidt'):
            raise ValueError("'densitymx' evotype cannot produce Hilbert-space ops!")
        return self.to_dense_superket()

    def to_dense_superket(self):
        return self.data

    @property
    def dim(self):
        return self.c_state._dim

    def __dealloc__(self):
        del self.c_state

    def __str__(self):
        return str([self.c_state._dataptr[i] for i in range(self.c_state._dim)])


cdef class StateRepDense(StateRep):
    def __cinit__(self, _np.ndarray[double, ndim=1] data, state_space, basis):
        #Ignore basis for now?
        self._cinit_base(data, state_space)

    def __reduce__(self):
        return (StateRepDense, (self.data, self.state_space, None), (self.data.flags.writeable,))

    @property
    def base(self):
        return self.data

    def base_has_changed(self):
        pass


cdef class StateRepPure(StateRep):
    cdef public _np.ndarray base
    cdef public object basis

    def __cinit__(self, purevec, basis, state_space):
        assert(purevec.dtype == _np.dtype(complex))
        self.base = _np.require(purevec.copy(), requirements=['OWNDATA', 'C_CONTIGUOUS'])
        self.basis = basis
        dmVec_std = _ot.state_to_dmvec(self.base)
        self._cinit_base(_bt.change_basis(dmVec_std, 'std', self.basis), state_space)

    def base_has_changed(self):
        dmVec_std = _ot.state_to_dmvec(self.base)
        self.data[:] = _bt.change_basis(dmVec_std, 'std', self.basis)

    def __reduce__(self):
        return (StateRepPure, (self.base, self.basis, self.state_space), (self.base.flags.writeable,))


cdef class StateRepComputational(StateRep):
    cdef public object zvals
    cdef public object basis

    def __cinit__(self, zvals, basis, state_space):

        #Convert zvals to dense vec:
        assert(basis.name == 'pp' or basis.name.split('*') == ['pp'] * (basis.name.count('*') + 1)), \
            "Only Pauli-product-basis computational states are supported so far"
        self.basis = basis
        factor_dim = 4
        v0 = 1.0 / _np.sqrt(2) * _np.array((1, 0, 0, 1), 'd')  # '0' qubit state as Pauli dmvec
        v1 = 1.0 / _np.sqrt(2) * _np.array((1, 0, 0, -1), 'd')  # '1' qubit state as Pauli dmvec
        v = (v0, v1)

        if _fastcalc is None:  # do it the slow way using numpy
            vec = _functools.reduce(_np.kron, [v[i] for i in zvals])
        else:
            typ = 'd'
            fast_kron_array = _np.ascontiguousarray(
                _np.empty((len(zvals), factor_dim), 'd'))
            fast_kron_factordims = _np.ascontiguousarray(_np.array([factor_dim] * len(zvals), _np.int64))
            for i, zi in enumerate(zvals):
                fast_kron_array[i, :] = v[zi]
            vec = _np.ascontiguousarray(_np.empty(factor_dim**len(zvals), typ))
            _fastcalc.fast_kron(vec, fast_kron_array, fast_kron_factordims)

        self.zvals = zvals
        self._cinit_base(vec, state_space)

    def __reduce__(self):
        return (StateRepComputational, (self.zvals, self.basis, self.state_space), (self.data.flags.writeable,))


cdef class StateRepComposed(StateRep):
    cdef public StateRep state_rep
    cdef public OpRep op_rep

    def __cinit__(self, StateRep state_rep, OpRep op_rep, state_space):
        self.state_rep = state_rep
        self.op_rep = op_rep
        self._cinit_base(state_rep.to_dense_superket(), state_space)
        self.reps_have_changed()

    def reps_have_changed(self):
        rep = self.op_rep.acton(self.state_rep)
        self.data[:] = rep.data[:]

    def __reduce__(self):
        return (StateRepComposed, (self.state_rep, self.op_rep, self.state_space), (self.data.flags.writeable,))


cdef class StateRepTensorProduct(StateRep):
    cdef public object factor_reps

    def __cinit__(self, factor_state_reps, state_space):
        self.factor_reps = factor_state_reps
        dim = _np.prod([fct.dim for fct in self.factor_reps])
        self._cinit_base(_np.zeros(dim, 'd'), state_space)
        self.reps_have_changed()

    def reps_have_changed(self):
        if len(self.factor_reps) == 0:
            vec = _np.empty(0, 'd')
        else:
            vec = self.factor_reps[0].to_dense_superket()
            for i in range(1, len(self.factor_reps)):
                vec = _np.kron(vec, self.factor_reps[i].to_dense_superket())
        self.data[:] = vec

    def __reduce__(self):
        return (StateRepTensorProduct, (self.factor_reps, self.state_space), (self.data.flags.writeable,))

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

import numpy as _np
import functools as _functools
from ...models.statespace import StateSpace as _StateSpace
from ...tools import fastcalc as _fastcalc

from .opreps cimport OpRep


cdef class StateRep(_basereps_cython.StateRep):

    def __cinit__(self):
        self.data = None
        self.c_state = NULL
        self.state_space = None

    def _cinit_base(self, _np.ndarray[_np.complex128_t, ndim=1, mode='c'] data, state_space):
        self.data = _np.require(data.copy(), requirements=['OWNDATA', 'C_CONTIGUOUS'])
        self.c_state = new StateCRep(<double complex*>self.data.data,<INT>self.data.shape[0],<bool>0)
        self.state_space = _StateSpace.cast(state_space)
        assert(len(self.data) == self.state_space.udim)

    def __reduce__(self):
        return (StateRep, (self.data, self.state_space), (self.data.flags.writeable,))

    def __setstate__(self, state):
        writeable, = state
        self.data.flags.writeable = writeable

    def copy_from(self, other):
        self.data[:] = other.data[:]

    def to_dense(self):
        return self.data

    @property
    def dim(self):
        return self.state_space.udim

    def __dealloc__(self):
        del self.c_state

    def __str__(self):
        return str([self.c_state._dataptr[i] for i in range(self.c_state._dim)])


cdef class StateRepDensePure(StateRep):
    cdef public object basis

    def __cinit__(self, purevec, basis, state_space):
        self.basis = basis
        self._cinit_base(purevec, state_space)

    @property
    def base(self):
        return self.data

    def base_has_changed(self):
        pass

    def __reduce__(self):
        return (StateRepDensePure, (self.data, self.basis, self.state_space), (self.data.flags.writeable,))


cdef class StateRepComputational(StateRep):
    cdef public object zvals
    cdef public object basis

    def __cinit__(self, zvals, basis, state_space):

        #Convert zvals to dense vec:
        factor_dim = 2
        v0 = _np.array((1, 0), complex)  # '0' qubit state as complex state vec
        v1 = _np.array((0, 1), complex)  # '1' qubit state as complex state vec
        v = (v0, v1)

        if _fastcalc is None:  # do it the slow way using numpy
            vec = _functools.reduce(_np.kron, [v[i] for i in zvals])
        else:
            typ = complex
            fast_kron_array = _np.ascontiguousarray(
                _np.empty((len(zvals), factor_dim), 'd'))
            fast_kron_factordims = _np.ascontiguousarray(_np.array([factor_dim] * len(zvals), _np.int64))
            for i, zi in enumerate(zvals):
                fast_kron_array[i, :] = v[zi]
            vec = _np.ascontiguousarray(_np.empty(factor_dim**len(zvals), typ))
            _fastcalc.fast_kron_complex(vec, fast_kron_array, fast_kron_factordims)

        self.zvals = zvals
        self.basis = basis
        self._cinit_base(vec, state_space)

    def __reduce__(self):
        return (StateRepComputational, (self.zvals, self.basis, self.state_space), (self.data.flags.writeable,))


cdef class StateRepComposed(StateRep):
    cdef public StateRep state_rep
    cdef public OpRep op_rep

    def __init__(self, state_rep, op_rep, state_space):
        self.state_rep = state_rep
        self.op_rep = op_rep
        self._cinit_base(state_rep.to_dense(), self.state_space)
        self.reps_have_changed()

    def reps_have_changed(self):
        rep = self.op_rep.acton(self.state_rep)
        self.data[:] = rep.data[:]

    def __reduce__(self):
        return (StateRepComposed, (self.state_rep, self.op_rep, self.state_space), (self.data.flags.writeable,))


cdef class StateRepTensorProduct(StateRep):
    cdef public object factor_reps

    def __init__(self, factor_state_reps, state_space):
        self.factor_reps = factor_state_reps
        dim = _np.product([fct.dim for fct in self.factor_reps])
        self._cinit_base(_np.zeros(dim, complex), state_space)
        self.reps_have_changed()

    def reps_have_changed(self):
        if len(self.factor_reps) == 0:
            vec = _np.empty(0, complex)
        else:
            vec = self.factor_reps[0].to_dense()
            for i in range(1, len(self.factors_reps)):
                vec = _np.kron(vec, self.factor_reps[i].to_dense())
        self.data[:] = vec

    def __reduce__(self):
        return (StateRepTensorProduct, (self.factor_state_reps, self.state_space), (self.data.flags.writeable,))

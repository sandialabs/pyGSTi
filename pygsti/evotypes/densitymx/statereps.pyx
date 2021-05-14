"""
Base classes for Cython representations.
"""
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
from ...tools import basistools as _bt
from ...tools import optools as _ot
from ...tools import fastcalc as _fastcalc

cdef class StateRep(_basereps_cython.StateRep):
    def __cinit__(self, _np.ndarray[double, ndim=1, mode='c'] data):
        self.base = _np.require(data.copy(), requirements=['OWNDATA', 'C_CONTIGUOUS'])
        self.c_state = new StateCRep(<double*>self.base.data,<INT>self.base.shape[0],<bool>0)

    def __reduce__(self):
        return (StateRep, (), (self.base,))

    def __setstate__(self, state):
        self.base, writeable = state
        self.base.flags.writeable = writeable

    def copy_from(self, other):
        self.base[:] = other.base[:]

    def to_dense(self):
        return self.base

    @property
    def dim(self):
        return self.c_state._dim

    def __dealloc__(self):
        del self.c_state

    def __str__(self):
        return str([self.c_state._dataptr[i] for i in range(self.c_state._dim)])


cdef class StateRepDense(StateRep):
    def base_has_changed(self):
        pass


cdef class StateRepPure(StateRep):
    def __init__(self, purevec, basis):
        assert(purevec.dtype == _np.dtype(complex))
        self.purebase = _np.require(purevec.copy(), requirements=['OWNDATA', 'C_CONTIGUOUS'])
        self.basis = basis
        dmVec_std = _ot.state_to_dmvec(self.purebase)
        super(StateRepPure, self).__init__(_bt.change_basis(dmVec_std, 'std', self.basis))

    def purebase_has_changed(self):
        dmVec_std = _ot.state_to_dmvec(self.purebase)
        self.base[:] = _bt.change_basis(dmVec_std, 'std', self.basis)


cdef class StateRepComputational(StateRep):
    def __init__(self, zvals):

        #Convert zvals to dense vec:
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

        super(StateRepComputational, self).__init__(vec)


cdef class StateRepComposed(StateRep):
    def __init__(self, state_rep, op_rep):
        self.state_rep = state_rep
        self.op_rep = op_rep
        super(StateRepComposed, self).__init__(state_rep.to_dense())
        self.reps_have_changed()

    def reps_have_changed(self):
        rep = self.op_rep.acton(self.state_rep)
        self.base[:] = rep.base[:]


cdef class StateRepTensorProduct(StateRep):
    def __init__(self, factor_state_reps):
        self.factor_reps = factor_state_reps
        dim = _np.product([fct.dim for fct in self.factor_reps])
        super(StateRepTensorProduct, self).__init__(_np.zeros(dim, 'd'))
        self.reps_have_changed()

    def reps_have_changed(self):
        if len(self.factor_reps) == 0:
            vec = _np.empty(0, 'd')
        else:
            vec = self.factor_reps[0].to_dense()
            for i in range(1, len(self.factors_reps)):
                vec = _np.kron(vec, self.factor_reps[i].to_dense())
        self.base[:] = vec

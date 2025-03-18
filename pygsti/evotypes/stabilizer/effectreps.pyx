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

import sys
import numpy as _np
from ...baseobjs.statespace import StateSpace as _StateSpace
from ...tools import matrixtools as _mt


cdef class EffectRep(_basereps_cython.EffectRep):

    def __cinit__(self):
        self.c_effect = NULL
        self.state_space = None

    def _cinit_base(self, state_space):
        self.state_space = _StateSpace.cast(state_space)

    def __reduce__(self):
        return (EffectRep, ())

    def __pygsti_reduce__(self):
        return self.__reduce__()

    def __dealloc__(self):
        del self.c_effect # check for NULL?

    @property
    def nqubits(self):
        return self.state_space.num_qubits

    def probability(self, StateRep state not None):
        #unnecessary (just put in signature): cdef StateRep st = <StateRep?>state
        return self.c_effect.probability(state.c_state)

    def amplitude(self, StateRep state not None):
        return self.c_effect.amplitude(state.c_state)


#cdef class EffectRepConjugatedState(EffectRep):
#    pass  # TODO - this should be possible


cdef class EffectRepComputational(EffectRep):
    cdef public _np.ndarray zvals
    cdef public object basis

    def __init__(self, _np.ndarray[_np.int64_t, ndim=1, mode='c'] zvals, basis, state_space):
        self.zvals = zvals
        self.c_effect = new EffectCRep_Computational(<INT*>zvals.data,
                                                     <INT>zvals.shape[0])
        self.basis = basis
        self._cinit_base(state_space)

    def __reduce__(self):
        return (EffectRepComputational, (self.zvals, self.basis, self.state_space))

    def to_dense(self, on_space, outvec=None):
        return _mt.zvals_to_dense(self.zvals, superket=(on_space not in ('minimal', 'Hilbert')))


cdef class EffectRepComposed(EffectRep):
    cdef public OpRep op_rep
    cdef public EffectRep effect_rep
    cdef public object op_id

    def __cinit__(self, OpRep op_rep not None, EffectRep effect_rep not None, op_id, state_space):
        cdef INT n = effect_rep.c_effect._n
        self.op_id = op_id
        self.op_rep = op_rep
        self.effect_rep = effect_rep
        self.c_effect = new EffectCRep_Composed(op_rep.c_rep,
                                                effect_rep.c_effect,
                                                <INT>op_id, n)
        self.state_space = _StateSpace.cast(state_space)
        assert(self.state_space.num_qubits == n)

    def __reduce__(self):
        return (EffectRepComposed, (self.op_rep, self.effect_rep, self.op_id, self.state_space))

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


cdef class EffectRep(_basereps_cython.EffectRep):

    def __cinit__(self, _np.ndarray[_np.int64_t, ndim=1, mode='c'] zvals):
        self.zvals = zvals
        self.c_effect = new EffectCRep(<INT*>zvals.data,
                                       <INT>zvals.shape[0])

    def __reduce__(self):
        return (EffectRep, (self.zvals,))

    def __dealloc__(self):
        del self.c_effect # check for NULL?

    @property
    def nqubits(self):
        return self.c_effect._n

    @property
    def dim(self):
        return 2**(self.c_effect._n)  # assume "unitary evolution"-type mode

    def probability(self, StateRep state not None):
        #unnecessary (just put in signature): cdef StateRep st = <StateRep?>state
        return self.c_effect.probability(state.c_state)

    def amplitude(self, StateRep state not None):
        return self.c_effect.amplitude(state.c_state)


#cdef class EffectRepConjugatedState(EffectRep):
#    pass  # TODO - this should be possible


cdef class EffectRepComputational(EffectRep):

    def __cinit__(self, _np.ndarray[_np.int64_t, ndim=1, mode='c'] zvals, INT dim):
        self.dim = dim
        self.zvals = zvals
        self.c_effect = new EffectCRep(<INT*>zvals.data,
                                       <INT>zvals.shape[0])

    def __reduce__(self):
        return (EffectRepComputational, (self.zvals, self.dim))

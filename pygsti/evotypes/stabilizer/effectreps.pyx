

import sys
import time as pytime
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


cdef class EffectRepConjugatedState(EffectRep):
    pass  # TODO - this should be possible


cdef class EffectRepComputational(EffectRep):

    def __cinit__(self, _np.ndarray[_np.int64_t, ndim=1, mode='c'] zvals, INT dim):
        self.dim = dim
        self.zvals = zvals
        self.c_effect = new EffectCRep(<INT*>zvals.data,
                                       <INT>zvals.shape[0])

    def __reduce__(self):
        return (EffectRepComputational, (self.zvals, self.dim))

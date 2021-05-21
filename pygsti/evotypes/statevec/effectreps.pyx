

import sys
import time as pytime
import numpy as _np
import itertools as _itertools

from ...tools import mpitools as _mpit
from ...tools import slicetools as _slct
from ...tools import optools as _gt
from ...tools.matrixtools import _fas
from ...objects.opcalc import fastopcalc as _fastopcalc
from scipy.sparse.linalg import LinearOperator


cdef double LARGE = 1000000000
# a large number such that LARGE is
# a very high term weight which won't help (at all) a
# path get included in the selected set of paths.

cdef double SMALL = 1e-5
# a number which is used in place of zero within the
# product of term magnitudes to keep a running path
# magnitude from being zero (and losing memory of terms).


cdef class EffectRep(_basereps_cython.EffectRep):
    def __cinit__(self):
        self.c_effect = NULL

    def __dealloc__(self):
        if self.c_effect != NULL:
            del self.c_effect

    def __reduce__(self):
        return (EffectRep, ())

    @property
    def dim(self):
        return self.c_effect._dim

    def probability(self, StateRep state not None):
        #unnecessary (just put in signature): cdef StateRep st = <StateRep?>state
        return self.c_effect.probability(state.c_state)

    def amplitude(self, StateRep state not None):
        return self.c_effect.amplitude(state.c_state)


cdef class EffectRepConjugatedState(EffectRep):
    cdef public StateRep state_rep

    def __cinit__(self, StateRep state_rep):
        self.state_rep = state_rep
        self.c_effect = new EffectCRep_Dense(<double complex*>self.state_rep.base.data,
                                             <INT>self.state_rep.base.shape[0])

    def __str__(self):
        return str([ (<EffectCRep_Dense*>self.c_effect)._dataptr[i] for i in range(self.c_effect._dim)])

    def __reduce__(self):
        return (EffectRepConjugatedState, (self.state_rep,))

    def to_dense(self):
        return self.state_rep.to_dense()


cdef class EffectRepComputational(EffectRep):
    cdef public _np.ndarray zvals

    def __cinit__(self, _np.ndarray[_np.int64_t, ndim=1, mode='c'] zvals, INT dim):
        # cdef INT dim = 4**zvals.shape[0] -- just send as argument
        cdef INT nfactors = zvals.shape[0]
        cdef double abs_elval = 1/(_np.sqrt(2)**nfactors)
        cdef INT base = 1
        cdef INT zvals_int = 0
        for i in range(nfactors):
            zvals_int += base * zvals[i]
            base = base << 1 # *= 2
        self.zvals = zvals
        self.c_effect = new EffectCRep_Computational(nfactors, zvals_int, dim)

    def __reduce__(self):
        return (EffectRepComputational, (self.zvals, self.c_effect._dim))

    #Add party & to_dense from slow version?


cdef class EffectRepTensorProduct(EffectRep):
    cdef public object povm_factors
    cdef public object effect_labels
    cdef public _np.ndarray kron_array
    cdef public _np.ndarray factor_dims

    def __init__(self, povm_factors, effect_labels):
        #Arrays for speeding up kron product in effect reps
        cdef INT max_factor_dim = max(fct.dim for fct in povm_factors)
        cdef _np.ndarray[double, ndim=2, mode='c'] kron_array = \
            _np.ascontiguousarray(_np.empty((len(povm_factors), max_factor_dim), 'd'))
        cdef _np.ndarray[_np.int64_t, ndim=1, mode='c'] factor_dims = \
            _np.ascontiguousarray(_np.array([fct.dim for fct in povm_factors], _np.int64))

        cdef INT dim = _np.product(factor_dims)
        cdef INT nfactors = len(self.povm_factors)
        self.povm_factors = povm_factors
        self.effect_labels = effect_labels
        self.kron_array = kron_array
        self.factor_dims = factor_dims
        self.c_effect = new EffectCRep_TensorProd(<double complex*>kron_array.data,
                                                  <INT*>factor_dims.data,
                                                  nfactors, max_factor_dim, dim)
        self.factor_effects_have_changed()  # computes self.kron_array

    def __reduce__(self):
        return (EffectRepTensorProduct, (self.povm_factors, self.effect_labels))

    def _fill_fast_kron(self):
        """ Fills in self._fast_kron_array based on current self.factors """
        for i, (factor_dim, Elbl) in enumerate(zip(self.factor_dims, self.effect_labels)):
                self.kron_array[i][0:factor_dim] = self.povm_factors[i][Elbl].to_dense()

    def factor_effects_have_changed(self):
        self._fill_fast_kron()  # updates effect reps

    #TODO: Take to_dense from slow version?

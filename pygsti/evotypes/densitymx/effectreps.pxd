#***************************************************************************************************
# Copyright 2015, 2019, 2025 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

from libc.stdlib cimport malloc, free
from libc.math cimport log10, sqrt, log
from libc cimport time
from libcpp cimport bool
from libcpp.vector cimport vector
from libcpp.pair cimport pair
from libcpp.algorithm cimport sort as stdsort
from libcpp.unordered_map cimport unordered_map
from cython.operator cimport dereference as deref, preincrement as inc
cimport numpy as _np
cimport cython

from .. cimport basereps_cython as _basereps_cython
from .statereps cimport StateRep, StateCRep
from .opreps cimport OpRep, OpCRep


#Use 64-bit integers
ctypedef long long INT
ctypedef unsigned long long UINT

cdef extern from "statecreps.h" namespace "CReps_densitymx":
    cdef cppclass StateCRep:
        pass

cdef extern from "opcreps.h" namespace "CReps_densitymx":
    cdef cppclass OpCRep:
        pass
    
cdef extern from "effectcreps.h" namespace "CReps_densitymx":
    cdef cppclass EffectCRep:
        EffectCRep() except +
        EffectCRep(INT) except +
        double probability(StateCRep* state)
        double probability_using_cache(StateCRep* state, StateCRep* precomp_state, INT& precomp_id)
        INT _dim

    cdef cppclass EffectCRep_Dense(EffectCRep):
        EffectCRep_Dense() except +
        EffectCRep_Dense(double*,INT) except +
        double probability(StateCRep* state)
        double probability_using_cache(StateCRep* state, StateCRep* precomp_state, INT& precomp_id)
        INT _dim
        double* _dataptr

    cdef cppclass EffectCRep_TensorProd(EffectCRep):
        EffectCRep_TensorProd() except +
        EffectCRep_TensorProd(double*, INT*, INT, INT, INT) except +
        double probability(StateCRep* state)
        double probability_using_cache(StateCRep* state, StateCRep* precomp_state, INT& precomp_id)
        INT _dim
        INT _nfactors
        INT _max_factor_dim

    cdef cppclass EffectCRep_Computational(EffectCRep):
        EffectCRep_Computational() except +
        EffectCRep_Computational(INT, INT, double, INT) except +
        double probability(StateCRep* state)
        double probability_using_cache(StateCRep* state, StateCRep* precomp_state, INT& precomp_id)
        INT _dim
        INT _zvals_int
        double _abs_elval

    cdef cppclass EffectCRep_Composed(EffectCRep):
        EffectCRep_Composed() except +
        EffectCRep_Composed(OpCRep*, EffectCRep*, INT, INT) except +
        double probability(StateCRep* state)
        double probability_using_cache(StateCRep* state, StateCRep* precomp_state, INT& precomp_id)
        INT _dim
        INT op_id

cdef class EffectRep(_basereps_cython.EffectRep):
    cdef EffectCRep* c_effect
    cdef public object state_space

ctypedef EffectCRep* EffectCRep_ptr
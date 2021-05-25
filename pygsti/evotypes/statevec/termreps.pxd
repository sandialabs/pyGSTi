#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
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
from ..basereps_cython cimport PolynomialRep, PolynomialCRep
from .statereps cimport StateRep, StateCRep
from .opreps cimport OpRep, OpCRep
from .effectreps cimport EffectRep, EffectCRep


#Use 64-bit integers
ctypedef long long INT
ctypedef unsigned long long UINT

cdef double LARGE = 1000000000
# a large number such that LARGE is
# a very high term weight which won't help (at all) a
# path get included in the selected set of paths.

cdef double SMALL = 1e-5
# a number which is used in place of zero within the
# product of term magnitudes to keep a running path
# magnitude from being zero (and losing memory of terms).


cdef extern from "termcreps.h" namespace "CReps":
    cdef cppclass TermCRep:
        TermCRep(PolynomialCRep*, double, double, StateCRep*, StateCRep*, vector[OpCRep*], vector[OpCRep*]) except +
        TermCRep(PolynomialCRep*, double, double, EffectCRep*, EffectCRep*, vector[OpCRep*], vector[OpCRep*]) except +
        TermCRep(PolynomialCRep*, double, double, vector[OpCRep*], vector[OpCRep*]) except +
        PolynomialCRep* _coeff
        double _magnitude
        double _logmagnitude
        StateCRep* _pre_state
        EffectCRep* _pre_effect
        vector[OpCRep*] _pre_ops
        StateCRep* _post_state
        EffectCRep* _post_effect
        vector[OpCRep*] _post_ops

    cdef cppclass TermDirectCRep:
        TermDirectCRep(double complex, double, double, StateCRep*, StateCRep*, vector[OpCRep*], vector[OpCRep*]) except +
        TermDirectCRep(double complex, double, double, EffectCRep*, EffectCRep*, vector[OpCRep*], vector[OpCRep*]) except +
        TermDirectCRep(double complex, double, double, vector[OpCRep*], vector[OpCRep*]) except +
        double complex _coeff
        double _magnitude
        double _logmagnitude
        StateCRep* _pre_state
        EffectCRep* _pre_effect
        vector[OpCRep*] _pre_ops
        StateCRep* _post_state
        EffectCRep* _post_effect
        vector[OpCRep*] _post_ops


cdef class TermRep(_basereps_cython.TermRep):
    cdef TermCRep* c_term

    #Hold references to other reps so we don't GC them
    cdef public _basereps_cython.PolynomialRep coeff
    cdef public StateRep pre_state
    cdef public StateRep post_state
    cdef public EffectRep pre_effect
    cdef public EffectRep post_effect
    cdef public object pre_ops
    cdef public object post_ops
    cdef public object compact_coeff


cdef class TermDirectRep(_basereps_cython.TermRep):
    cdef TermDirectCRep* c_term

    #Hold references to other reps so we don't GC them
    cdef StateRep state_ref1
    cdef StateRep state_ref2
    cdef EffectRep effect_ref1
    cdef EffectRep effect_ref2
    cdef object list_of_preops_ref
    cdef object list_of_postops_ref


ctypedef StateCRep* StateCRep_ptr
ctypedef OpCRep* OpCRep_ptr
ctypedef EffectCRep* EffectCRep_ptr
ctypedef TermCRep* TermCRep_ptr

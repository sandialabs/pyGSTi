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
from .statereps cimport StateRep, StateCRep, StateRepDensePure


#Use 64-bit integers
ctypedef long long INT
ctypedef unsigned long long UINT


cdef extern from "opcreps.h" namespace "CReps_statevec":
    cdef cppclass OpCRep:
        OpCRep(INT) except +
        StateCRep* acton(StateCRep*, StateCRep*)
        StateCRep* adjoint_acton(StateCRep*, StateCRep*)
        StateCRep* acton_random(StateCRep*, StateCRep*, object)
        StateCRep* adjoint_acton_random(StateCRep*, StateCRep*, object)
        INT _dim

    cdef cppclass OpCRep_DenseUnitary(OpCRep):
        OpCRep_DenseUnitary(double complex*,INT) except +
        StateCRep* acton(StateCRep*, StateCRep*)
        StateCRep* adjoint_acton(StateCRep*, StateCRep*)
        double complex* _dataptr
        INT _dim

    cdef cppclass OpCRep_Embedded(OpCRep):
        OpCRep_Embedded(OpCRep*, INT*, INT*, INT*, INT*, INT, INT, INT, INT, INT) except +
        StateCRep* acton(StateCRep*, StateCRep*)
        StateCRep* adjoint_acton(StateCRep*, StateCRep*)
        StateCRep* acton_random(StateCRep*, StateCRep*, object)
        StateCRep* adjoint_acton_random(StateCRep*, StateCRep*, object)
        INT _nComponents
        INT _embeddedDim
        INT _iActiveBlock
        INT _nBlocks

    cdef cppclass OpCRep_Composed(OpCRep):
        OpCRep_Composed(vector[OpCRep*], INT) except +
        void reinit_factor_op_creps(vector[OpCRep*])
        StateCRep* acton(StateCRep*, StateCRep*)
        StateCRep* adjoint_acton(StateCRep*, StateCRep*)
        StateCRep* acton_random(StateCRep*, StateCRep*, object)
        StateCRep* adjoint_acton_random(StateCRep*, StateCRep*, object)

    cdef cppclass OpCRep_Sum(OpCRep):
        OpCRep_Sum(vector[OpCRep*], INT) except +
        StateCRep* acton(StateCRep*, StateCRep*)
        StateCRep* adjoint_acton(StateCRep*, StateCRep*)
        StateCRep* acton_random(StateCRep*, StateCRep*, object)
        StateCRep* adjoint_acton_random(StateCRep*, StateCRep*, object)

    cdef cppclass OpCRep_Repeated(OpCRep):
        OpCRep_Repeated(OpCRep*, INT, INT) except +
        StateCRep* acton(StateCRep*, StateCRep*)
        StateCRep* adjoint_acton(StateCRep*, StateCRep*)
        StateCRep* acton_random(StateCRep*, StateCRep*, object)
        StateCRep* adjoint_acton_random(StateCRep*, StateCRep*, object)

    cdef cppclass OpCRep_RandomUnitary(OpCRep):
        OpCRep_RandomUnitary(object, vector[OpCRep*], object, INT) except +
        StateCRep* acton_random(StateCRep*, StateCRep*, object)
        StateCRep* adjoint_acton_random(StateCRep*, StateCRep*, object)


cdef class OpRep(_basereps_cython.OpRep):
    cdef OpCRep* c_rep
    cdef public object state_space

ctypedef OpCRep* OpCRep_ptr

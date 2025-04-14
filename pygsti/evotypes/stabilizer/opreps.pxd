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


#Use 64-bit integers
ctypedef long long INT
ctypedef unsigned long long UINT


cdef extern from "opcreps.h" namespace "CReps_stabilizer":
    cdef cppclass OpCRep:
        OpCRep(INT) except +
        StateCRep* acton(StateCRep*, StateCRep*)
        StateCRep* adjoint_acton(StateCRep*, StateCRep*)
        INT _n

    cdef cppclass OpCRep_Embedded(OpCRep):
        OpCRep_Embedded(OpCRep*, INT, INT*, INT) except +
        StateCRep* acton(StateCRep*, StateCRep*)
        StateCRep* adjoint_acton(StateCRep*, StateCRep*)

    cdef cppclass OpCRep_Composed(OpCRep):
        OpCRep_Composed(vector[OpCRep*], INT) except +
        void reinit_factor_op_creps(vector[OpCRep*])
        StateCRep* acton(StateCRep*, StateCRep*)
        StateCRep* adjoint_acton(StateCRep*, StateCRep*)

    cdef cppclass OpCRep_Sum(OpCRep):
        OpCRep_Sum(vector[OpCRep*], INT) except +
        StateCRep* acton(StateCRep*, StateCRep*)
        StateCRep* adjoint_acton(StateCRep*, StateCRep*)

    cdef cppclass OpCRep_Repeated(OpCRep):
        OpCRep_Repeated(OpCRep*, INT, INT) except +
        StateCRep* acton(StateCRep*, StateCRep*)
        StateCRep* adjoint_acton(StateCRep*, StateCRep*)

    cdef cppclass OpCRep_Clifford(OpCRep):
        OpCRep_Clifford(INT*, INT*, double complex*, INT*, INT*, double complex*, INT) except +
        StateCRep* acton(StateCRep*, StateCRep*)
        StateCRep* adjoint_acton(StateCRep*, StateCRep*)
        #for DEBUG:
        INT *_smatrix
        INT *_svector
        INT *_smatrix_inv
        INT *_svector_inv
        double complex *_unitary
        double complex *_unitary_adj


cdef class OpRep(_basereps_cython.OpRep):
    cdef OpCRep* c_rep
    cdef public object state_space

ctypedef OpCRep* OpCRep_ptr

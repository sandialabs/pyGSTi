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

from libcpp cimport bool
from libcpp.vector cimport vector
from libcpp.pair cimport pair
from libcpp.algorithm cimport sort as stdsort
from libcpp.unordered_map cimport unordered_map
cimport numpy as _np
cimport cython


#Use 64-bit integers
ctypedef long long INT
ctypedef unsigned long long UINT


cdef class OpRep:
    pass

cdef class StateRep:
    pass

cdef class EffectRep:
    pass

cdef class TermRep:
    pass


cdef extern from "basecreps.h" namespace "CReps":

    cdef cppclass PolynomialVarsIndex:
        PolynomialVarsIndex() except +
        PolynomialVarsIndex(INT) except +
        bool operator<(PolynomialVarsIndex i)
        vector[INT] _parts

    cdef cppclass PolynomialCRep:
        PolynomialCRep() except +
        PolynomialCRep(unordered_map[PolynomialVarsIndex, complex], INT, INT) except +
        PolynomialCRep abs()
        PolynomialCRep mult(PolynomialCRep&)
        PolynomialCRep abs_mult(PolynomialCRep&)
        void add_inplace(PolynomialCRep&)
        void add_abs_inplace(PolynomialCRep&)
        void add_scalar_to_all_coeffs_inplace(double complex offset)
        void scale(double complex scale)
        vector[INT] int_to_vinds(PolynomialVarsIndex indx_tup)
        unordered_map[PolynomialVarsIndex, complex] _coeffs
        INT _max_num_vars
        INT _vindices_per_int


cdef class PolynomialRep:
    cdef PolynomialCRep* c_polynomial

cdef PolynomialRep_from_allocd_PolynomialCRep(PolynomialCRep* crep)
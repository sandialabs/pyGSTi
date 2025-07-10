#***************************************************************************************************
# Copyright 2015, 2019, 2025 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

cimport numpy as _np
from libcpp cimport bool
from .. cimport basereps_cython as _basereps_cython

#Use 64-bit integers
ctypedef long long INT
ctypedef unsigned long long UINT

cdef extern from "statecreps.h" namespace "CReps_stabilizer":

    cdef cppclass StateCRep:
        StateCRep(INT*, INT*, double complex*, INT, INT) except +
        StateCRep(INT, INT) except +
        StateCRep(double*,INT,bool) except +
        void copy_from(StateCRep*)
        INT _n
        INT _namps
        # for DEBUG
        INT* _smatrix
        INT* _pvectors
        INT _zblock_start
        double complex* _amps


cdef class StateRep(_basereps_cython.StateRep):
    cdef StateCRep* c_state
    cdef public _np.ndarray smatrix
    cdef public _np.ndarray pvectors
    cdef public _np.ndarray amps
    cdef public object state_space

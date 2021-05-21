
cimport numpy as _np
from libcpp cimport bool
from .. cimport basereps_cython as _basereps_cython

#Use 64-bit integers
ctypedef long long INT
ctypedef unsigned long long UINT

cdef extern from "statecreps.h" namespace "CReps":

    cdef cppclass StateCRep:
        StateCRep() except +
        StateCRep(INT) except +
        StateCRep(double complex*,INT,bool) except +
        void copy_from(StateCRep*)
        INT _dim
        double complex* _dataptr


cdef class StateRep(_basereps_cython.StateRep):
    cdef StateCRep* c_state
    cdef public _np.ndarray base
    #cdef double [:] data_view # alt way to hold a reference

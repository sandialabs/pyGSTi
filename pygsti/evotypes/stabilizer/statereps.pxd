
cimport numpy as _np
from libcpp cimport bool
from .. cimport basereps_cython as _basereps_cython

#Use 64-bit integers
ctypedef long long INT
ctypedef unsigned long long UINT

cdef extern from "statecreps.h" namespace "CReps":

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

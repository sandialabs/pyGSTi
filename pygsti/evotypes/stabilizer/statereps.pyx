# encoding: utf-8
# cython: profile=False
# cython: linetrace=False

#***************************************************************************************************
# Copyright 2015, 2019, 2025 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

import numpy as _np
import itertools as _itertools
from ...baseobjs.statespace import StateSpace as _StateSpace

from .opreps cimport OpRep


cdef class StateRep(_basereps_cython.StateRep):

    def __cinit__(self):
        self.smatrix = self.pvectors = self.amps = None
        self.c_state = NULL
        self.state_space = None

    def _cinit_base(self, _np.ndarray[_np.int64_t, ndim=2, mode='c'] smatrix,
                    _np.ndarray[_np.int64_t, ndim=2, mode='c'] pvectors,
                    _np.ndarray[_np.complex128_t, ndim=1, mode='c'] amps,
                    state_space):
        self.smatrix = smatrix
        self.pvectors = pvectors
        self.amps = amps
        cdef INT namps = amps.shape[0]
        self.state_space = _StateSpace.cast(state_space)
        assert(smatrix.shape[0] // 2 == self.state_space.num_qubits)
        self.c_state = new StateCRep(<INT*>smatrix.data,<INT*>pvectors.data,
                                     <double complex*>amps.data, namps,
                                     self.state_space.num_qubits)

    def __reduce__(self):
        return (StateRep, ())

    def __pygsti_reduce__(self):
        return self.__reduce__()

    @property
    def nqubits(self):
        return self.state_space.num_qubits

    #@property
    #def dim(self):
    #    return 2**(self.c_state._n) # assume "unitary evolution"-type mode

    def actionable_staterep(self):
        # return a state rep that can be acted on by op reps or mapped to
        # a probability/amplitude by POVM effect reps.
        return self  # for most classes, the rep itself is actionable

    def __dealloc__(self):
        del self.c_state

    def __str__(self):
        #DEBUG
        cdef INT n = self.c_state._n
        cdef INT namps = self.c_state._namps
        s = "StateRep\n"
        s +=" smx = " + str([ self.c_state._smatrix[ii] for ii in range(2*n*2*n) ])
        s +=" pvecs = " + str([ self.c_state._pvectors[ii] for ii in range(2*n) ])
        s +=" amps = " + str([ self.c_state._amps[ii] for ii in range(namps) ])
        s +=" zstart = " + str(self.c_state._zblock_start)
        return s


cdef class StateRepComputational(StateRep):
    cdef public object zvals
    cdef public object basis
    
    def __cinit__(self, zvals, basis, state_space):

        nqubits = len(zvals)
        state_s = _np.fliplr(_np.identity(2 * nqubits, _np.int64))  # flip b/c stab cols are *first*
        state_ps = _np.zeros(2 * nqubits, _np.int64)
        for i, z in enumerate(zvals):
            state_ps[i] = state_ps[i + nqubits] = 2 if <bool>z else 0
            # TODO: check this is right -- (how/need to update the destabilizers?)

        s = state_s.copy()  # needed?
        ps = state_ps.reshape(1, 2 * nqubits)
        a = _np.ones(1, complex)  # all == 1.0 by default

        self.zvals = zvals
        self.basis = basis
        self._cinit_base(s, ps, a, state_space)

    #TODO: copy methods from StabilizerFrame or StateCRep - or maybe do this for base StateRep class? ----------------------------

    def __reduce__(self):
        return (StateRepComputational, (self.zvals, self.basis, self.state_space))


cdef class StateRepComposed(StateRep):
    cdef public StateRep state_rep
    cdef public OpRep op_rep

    def __cinit__(self, state_rep, op_rep, state_space):
        self.state_rep = state_rep
        self.op_rep = op_rep
        self._cinit_base(state_rep.smatrix, state_rep.pvectors, state_rep.amps, state_space)
        self.reps_have_changed()

    def reps_have_changed(self):
        pass  # don't do anything here - all work in actionalble_staterep

    def actionable_staterep(self):
        state_rep = self.state_rep.actionable_staterep()
        rep = self.op_rep.acton(state_rep)
        #self.smatrix[:, :] = rep.smatrix[:, :]
        #self.pvectors[:, :] = rep.pvectors[:, :]
        #self.amps[:] = rep.amps[:]
        return rep

    def __reduce__(self):
        return (StateRepComposed, (self.state_rep, self.op_rep, self.state_space))


cdef class StateRepTensorProduct(StateRep):
    def __cinit__(self, factor_state_reps, state_space):
        self.factor_reps = factor_state_reps
        n = sum([sf.nqubits for sf in self.factor_reps])  # total number of qubits
        np = int(_np.prod([len(sf.pvectors) for sf in self.factor_reps]))
        self._cinit_base(_np.zeros((2 * n, 2 * n), _np.int64),
                         _np.zeros((np, 2 * n), _np.int64),
                         _np.ones(np, complex),
                         state_space)
        self.reps_have_changed()

    def reps_have_changed(self):
        # Similar to symplectic_kronecker and stabilizer.sframe_kronecker for each factor
        sframe_factors = self.factor_reps
        n = sum([sf.nqubits for sf in sframe_factors])  # total number of qubits

        # (common) state matrix
        sout = self.smatrix  #_np.zeros((2 * n, 2 * n), _np.int64)
        k = 0  # current qubit index
        for sf in sframe_factors:
            nq = sf.nqubits
            sout[k:k + nq, k:k + nq] = sf.smatrix[0:nq, 0:nq]
            sout[k:k + nq, n + k:n + k + nq] = sf.smatrix[0:nq, nq:2 * nq]
            sout[n + k:n + k + nq, k:k + nq] = sf.smatrix[nq:2 * nq, 0:nq]
            sout[n + k:n + k + nq, n + k:n + k + nq] = sf.smatrix[nq:2 * nq, nq:2 * nq]
            k += nq
    
        # phase vectors and amplitudes
        self.amps[:] = 1.0 + 0j  # reset all amplitudes to 1.0
        inds = [range(len(sf.pvectors)) for sf in sframe_factors]
        for pi, ii in enumerate(_itertools.product(*inds)):
            for i, sf in zip(ii, sframe_factors):
                nq = sf.nqubits
                self.pvectors[pi][k:k + nq] = sf.pvectors[i][0:nq]
                self.pvectors[pi][n + k:n + k + nq] = sf.pvectors[i][nq:2 * nq]
                self.amps[pi] *= sf.amps[i]

    def __reduce__(self):
        return (StateRepTensorProduct, (self.factor_reps, self.state_space))

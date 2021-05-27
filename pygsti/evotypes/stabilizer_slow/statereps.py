"""
State representations for "stabilizer_slow" evolution type.
"""
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************


import numpy as _np
import functools as _functools

from .. import basereps as _basereps
from ...objects import stabilizer as _stabilizer
from ...models.statespace import StateSpace as _StateSpace


class StateRep(_basereps.StateRep):
    def __init__(self, smatrix, pvectors, amps, state_space):
        self.state_space = _StateSpace.cast(state_space)
        self.sframe = _stabilizer.StabilizerFrame(smatrix, pvectors, amps)
        # just rely on StabilizerFrame class to do all the heavy lifting...
        assert(self.sframe.n == self.state_space.num_qubits)

    @property
    def smatrix(self):
        return self.sframe.s

    @property
    def pvectors(self):
        return self.sframe.ps

    @property
    def amps(self):
        return self.sframe.a

    @property
    def nqubits(self):
        return self.sframe.n

    #@property
    #def dim(self):
    #    return 2**self.nqubits  # assume "unitary evolution"-type mode

    def copy(self):
        cpy = StateRep(_np.zeros((0, 0), _np.int64), None, None, self.state_space)  # makes a dummy cpy.sframe
        cpy.sframe = self.sframe.copy()  # a legit copy *with* qubit filers copied too
        return cpy

    def __str__(self):
        return "StateRep:\n" + str(self.sframe)


class StateRepComputational(StateRep):

    def __init__(self, zvals, basis, state_space):

        nqubits = len(zvals)
        state_s = _np.fliplr(_np.identity(2 * nqubits, int))  # flip b/c stab cols are *first*
        state_ps = _np.zeros(2 * nqubits, int)
        for i, z in enumerate(zvals):
            state_ps[i] = state_ps[i + nqubits] = 2 if bool(z) else 0
            # TODO: check this is right -- (how/need to update the destabilizers?)

        ps = state_ps.reshape(1, 2 * nqubits)
        a = _np.ones(1, complex)  # all == 1.0 by default

        self.basis = basis
        super(StateRepComputational, self).__init__(state_s, ps, a, state_space)

    #TODO: copy methods from StabilizerFrame or StateCRep - or maybe do this for base StateRep class? ----------------------------


class StateRepComposed(StateRep):
    def __init__(self, state_rep, op_rep, state_space):
        self.state_rep = state_rep
        self.op_rep = op_rep
        super(StateRepComposed, self).__init__(state_rep.smatrix, state_rep.pvectors, state_rep.amps, state_space)
        self.reps_have_changed()

    def reps_have_changed(self):
        rep = self.op_rep.acton(self.state_rep)
        self.smatrix[:, :] = rep.smatrix[:, :]
        self.pvectors[:, :] = rep.pvectors[:, :]
        self.amps[:] = rep.amps[:]


class StateRepTensorProduct(StateRep):
    def __init__(self, factor_state_reps, state_space):
        self.factor_reps = factor_state_reps
        n = sum([sf.nqubits for sf in self.factor_reps])  # total number of qubits
        np = int(_np.product([len(sf.pvectors) for sf in self.factor_reps]))

        super(StateRepTensorProduct, self).__init__(_np.zeros((2 * n, 2 * n), int),
                                                    _np.zeros((np, 2 * n), int),
                                                    _np.ones(np, complex),
                                                    state_space)
        self.reps_have_changed()

    def reps_have_changed(self):
        # Similar to symplectic_kronecker and stabilizer.sframe_kronecker for each factor
        sframe_factors = [state_rep.sframe for state_rep in self.factor_reps]  # StabilizerFrame for each factor
        new_rep = _stabilizer.sframe_kronecker(sframe_factors).to_rep()
        self.smatrix[:, :] = new_rep.smatrix[:, :]
        self.pvectors[:, :] = new_rep.pvectors[:, :]
        self.amps[:, :] = new_rep.amps[:, :]

"""
State representation classes for the `statevec_slow` evolution type.
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
from ...models.statespace import StateSpace as _StateSpace

try:
    from ...tools import fastcalc as _fastcalc
except ImportError:
    _fastcalc = None


class StateRep(_basereps.StateRep):
    def __init__(self, data, state_space):
        assert(data.dtype == _np.dtype(complex))
        self.base = _np.require(data.copy(), requirements=['OWNDATA', 'C_CONTIGUOUS'])
        self.state_space = _StateSpace.cast(state_space)
        assert(len(self.base) == self.state_space.udim)

    def __reduce__(self):
        return (StateRep, (self.base,), (self.base.flags.writeable,))

    def __setstate__(self, state):
        writeable, = state
        self.base.flags.writeable = writeable

    def copy_from(self, other):
        self.base[:] = other.base

    def to_dense(self):
        return self.base

    @property
    def dim(self):
        return self.state_space.udim

    def __str__(self):
        return str(self.base)


class StateRepPure(StateRep):
    def __init__(self, purevec, basis, state_space):
        self.basis = basis
        super(StateRepPure, self).__init__(purevec, state_space)

    def purebase_has_changed(self):
        pass


class StateRepComputational(StateRep):
    def __init__(self, zvals, state_space):

        #Convert zvals to dense vec:
        factor_dim = 2
        v0 = _np.array((1, 0), complex)  # '0' qubit state as complex state vec
        v1 = _np.array((0, 1), complex)  # '1' qubit state as complex state vec
        v = (v0, v1)

        if _fastcalc is None:  # do it the slow way using numpy
            vec = _functools.reduce(_np.kron, [v[i] for i in zvals])
        else:
            typ = complex
            fast_kron_array = _np.ascontiguousarray(
                _np.empty((len(self._zvals), factor_dim), typ))
            fast_kron_factordims = _np.ascontiguousarray(_np.array([factor_dim] * len(zvals), _np.int64))
            for i, zi in enumerate(self._zvals):
                fast_kron_array[i, :] = v[zi]
            vec = _np.ascontiguousarray(_np.empty(factor_dim**len(zvals), typ))
            _fastcalc.fast_kron_complex(vec, fast_kron_array, fast_kron_factordims)

        self.zvals = zvals
        super(StateRepComputational, self).__init__(vec, state_space)


class StateRepComposed(StateRep):
    def __init__(self, state_rep, op_rep, state_space):
        self.state_rep = state_rep
        self.op_rep = op_rep
        super(StateRepComposed, self).__init__(state_rep.to_dense(), state_space)
        self.reps_have_changed()

    def reps_have_changed(self):
        rep = self.op_rep.acton(self.state_rep)
        self.base[:] = rep.base[:]


class StateRepTensorProduct(StateRep):
    def __init__(self, factor_state_reps, state_space):
        self.factor_reps = factor_state_reps
        dim = _np.product([fct.dim for fct in self.factor_reps])
        super(StateRepTensorProduct, self).__init__(_np.zeros(dim, complex), state_space)
        self.reps_have_changed()

    def reps_have_changed(self):
        if len(self.factor_reps) == 0:
            vec = _np.empty(0, complex)
        else:
            vec = self.factor_reps[0].to_dense()
            for i in range(1, len(self.factors_reps)):
                vec = _np.kron(vec, self.factor_reps[i].to_dense())
        self.base[:] = vec

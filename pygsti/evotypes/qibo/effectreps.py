"""
POVM effect representation classes for the `qibo` evolution type.
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

from .. import basereps as _basereps
from pygsti.baseobjs.statespace import StateSpace as _StateSpace

from . import _get_densitymx_mode, _get_nshots


class EffectRep(_basereps.EffectRep):
    def __init__(self, state_space):
        self.state_space = _StateSpace.cast(state_space)

    @property
    def nqubits(self):
        return self.state_space.num_qubits


class EffectRepComputational(EffectRep):
    def __init__(self, zvals, basis, state_space):
        self.zvals = zvals
        self.basis = basis
        super(EffectRepComputational, self).__init__(state_space)


class EffectRepConjugatedState(EffectRep):

    def __init__(self, state_rep):
        self.state_rep = state_rep
        super(EffectRepConjugatedState, self).__init__(state_rep.state_space)

    def probability(self, state):
        # compute <s2|s1>
        assert(_get_densitymx_mode() is True), "Can only use EffectRepConjugatedState when densitymx_mode == True!"

        initial_state = state.qibo_state
        effect_state = self.state_rep.qibo_state
        assert(effect_state.ndim == 2)  # density matrices

        qibo_circuit = state.qibo_circuit
        results = qibo_circuit(initial_state)
        return _np.real_if_close(_np.dot(effect_state.flatten().conjugate(), results.state().flatten()))

    def to_dense(self, on_space):
        return self.state_rep.to_dense(on_space)

    @property
    def basis(self):
        # (all qibo effect reps need to have a .basis property)
        return self.state_rep.basis

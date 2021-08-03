"""
POVM effect representation classes for the `stabilizer_slow` evolution type.
"""
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

from .. import basereps as _basereps
from pygsti.baseobjs.statespace import StateSpace as _StateSpace
from ...tools import matrixtools as _mt


class EffectRep(_basereps.EffectRep):
    def __init__(self, state_space):
        self.state_space = _StateSpace.cast(state_space)

    @property
    def nqubits(self):
        return self.state_space.num_qubits

    #@property
    #def dim(self):
    #    return 2**self.nqubits  # assume "unitary evolution"-type mode

    def probability(self, state):
        return state.sframe.measurement_probability(self.zvals, check=True)  # use check for now?

    def amplitude(self, state):
        return state.sframe.extract_amplitude(self.zvals)

    def to_dense(self, on_space):
        return _mt.zvals_to_dense(self.zvals, superket=bool(on_space not in ('minimal', 'Hilbert')))


#class EffectRepConjugatedState(EffectRep):
#    pass  # TODO - this should be possible


class EffectRepComputational(EffectRep):

    def __init__(self, zvals, basis, state_space):
        self.zvals = zvals
        self.basis = basis
        assert(self.state_space.num_qubits == len(self.zvals))
        super(EffectRepComputational, self).__init__(state_space)

    #@property
    #def outcomes(self):
    #    """
    #    The 0/1 outcomes identifying this effect within its StabilizerZPOVM
    #
    #    Returns
    #    -------
    #    numpy.ndarray
    #    """
    #    return self.zvals

    def __str__(self):
        nQubits = len(self.zvals)
        s = "Stabilizer effect vector for %d qubits with outcome %s" % (nQubits, str(self.zvals))
        return s

    def to_dense(self, on_space, outvec=None):
        return _mt.zvals_to_dense(self.zvals, superket=bool(on_space not in ('minimal', 'Hilbert')))


class EffectRepComposed(EffectRep):
    def __init__(self, op_rep, effect_rep, op_id, state_space):
        self.op_rep = op_rep
        self.effect_rep = effect_rep
        self.op_id = op_id

        state_space = _StateSpace.cast(state_space)
        assert(state_space.is_compatible_with(effect_rep.state_space))

        super(EffectRepComposed, self).__init__(state_space)

    #def __reduce__(self):
    #    return (EffectRepComposed, (self.op_rep, self.effect_rep, self.op_id, self.state_space))

    def probability(self, state):
        state = self.op_rep.acton(state)  # *not* acton_adjoint
        return self.effect_rep.probability(state)

    def amplitude(self, state):  # allow scratch to be passed in?
        state = self.op_rep.acton(state)  # *not* acton_adjoint
        return self.effect_rep.amplitude(state)

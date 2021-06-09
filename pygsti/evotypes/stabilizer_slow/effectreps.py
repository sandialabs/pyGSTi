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

import numpy as _np
import functools as _functools

from .. import basereps as _basereps
from ...models.statespace import StateSpace as _StateSpace
from ...tools import matrixtools as _mt


class EffectRep(_basereps.EffectRep):
    def __init__(self, zvals, state_space):
        self.zvals = zvals
        self.state_space = _StateSpace.cast(state_space)
        assert(self.state_space.num_qubits == len(self.zvals))

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

#OLD REMOVE:
#    def to_dense(self, on_space):
#        """
#        Return this SPAM vector as a (dense) numpy array.
#
#        The memory in `scratch` maybe used when it is not-None.
#
#        Parameters
#        ----------
#        scratch : numpy.ndarray, optional
#            scratch space available for use.
#
#        Returns
#        -------
#        numpy.ndarray
#        """
#        if on_space not in ('minimal', 'Hilbert'):
#            raise ValueError('stabilizer evotype cannot (yet) generate dense Hilbert-Schmidt effect vectors')
#        v = (_np.array([1, 0], 'd'), _np.array([0, 1], 'd'))  # (v0,v1) - eigenstates of sigma_z
#        statevec = _functools.reduce(_np.kron, [v[i] for i in self.zvals])
#        statevec.shape = (statevec.size, 1)
#        return statevec


#class EffectRepConjugatedState(EffectRep):
#    pass  # TODO - this should be possible


class EffectRepComputational(EffectRep):

    def __init__(self, zvals, state_space):
        super(EffectRepComputational, self).__init__(zvals, state_space)

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

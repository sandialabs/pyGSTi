"""
The StaticPOVMEffect class and supporting functionality.
"""
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

from pygsti.modelmembers.povms.conjugatedeffect import ConjugatedStatePOVMEffect as _ConjugatedStatePOVMEffect
from pygsti.modelmembers.states.staticstate import StaticState as _StaticState


class StaticPOVMEffect(_ConjugatedStatePOVMEffect):
    """
    A POVM effect vector that is completely fixed, or "static" (i.e. that posesses no parameters).

    Parameters
    ----------
    vec : array_like or POVMEffect
        a 1D numpy array representing the POVM effect.  The
        shape of this array sets the dimension of the POVM effect.

    evotype : Evotype or str, optional
        The evolution type.  The special value `"default"` is equivalent
        to specifying the value of `pygsti.evotypes.Evotype.default_evotype`.

    state_space : StateSpace, optional
        The state space for this operation.  If `None` a default state space
        with the appropriate number of qubits is used.
    """

    def __init__(self, vec, basis=None, evotype="default", state_space=None):
        _ConjugatedStatePOVMEffect.__init__(self, _StaticState(vec, basis, evotype, state_space))

    def set_dense(self, vec):
        """
        Set the dense-vector value of this POVM effect vector.

        Attempts to modify this POVM effect vector's parameters so that the raw
        POVM effect vector becomes `vec`.  Will raise ValueError if this operation
        is not possible.

        Parameters
        ----------
        vec : array_like or POVMEffect
            A numpy array representing a POVM effect vector, or a POVMEffect object.

        Returns
        -------
        None
        """
        self.state.set_dense(vec)
        self.dirty = True

"""
The StaticPOVMPureEffect class and supporting functionality.
"""
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

from .conjugatedeffect import ConjugatedStatePOVMEffect as _ConjugatedStatePOVMEffect
from ..states.staticpurestate import StaticPureState as _StaticPureState


class StaticPOVMPureEffect(_ConjugatedStatePOVMEffect):
    """
    TODO: docstring
    A state vector that is completely fixed, or "static" (i.e. that posesses no parameters).

    Parameters
    ----------
    vec : array_like or POVMEffect
        a 1D numpy array representing the state.  The
        shape of this array sets the dimension of the state.

    basis : Basis or {'pp','gm','std'}, optional
        The basis used to construct the Hilbert-Schmidt space representation
        of this state as a super-bra.

    evotype : Evotype or str, optional
        The evolution type.  The special value `"default"` is equivalent
        to specifying the value of `pygsti.evotypes.Evotype.default_evotype`.

    state_space : StateSpace, optional
        The state space for this operation.  If `None` a default state space
        with the appropriate number of qubits is used.
    """

    def __init__(self, vec, basis='pp', evotype="default", state_space=None):
        _ConjugatedStatePOVMEffect.__init__(self, _StaticPureState(vec, basis, evotype, state_space))

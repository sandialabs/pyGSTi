"""
The StaticState class and supporting functionality.
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
from pygsti.modelmembers.states.densestate import DenseState as _DenseState


class StaticState(_DenseState):
    """
    A state vector that is completely fixed, or "static" (i.e. that posesses no parameters).

    Parameters
    ----------
    vec : array_like or State
        a 1D numpy array representing the state.  The
        shape of this array sets the dimension of the state.

    basis : Basis or str
        The basis that `vec` is in.

    evotype : Evotype or str, optional
        The evolution type.  The special value `"default"` is equivalent
        to specifying the value of `pygsti.evotypes.Evotype.default_evotype`.

    state_space : StateSpace, optional
        The state space for this operation.  If `None` a default state space
        with the appropriate number of qubits is used.
    """

    def __init__(self, vec, basis=None, evotype="default", state_space=None):
        _DenseState.__init__(self, vec, basis, evotype, state_space)

    def _is_similar(self, other, rtol, atol):
        """ Returns True if `other` model member (which it guaranteed to be the same type as self) has
            the same local structure, i.e., not considering parameter values or submembers """
        # static objects must also test their values in is_similar, since these aren't parameters.
        return (super()._is_similar(other, rtol, atol)
                and _np.allclose(self.to_dense(), other.to_dense(), rtol=rtol, atol=atol))

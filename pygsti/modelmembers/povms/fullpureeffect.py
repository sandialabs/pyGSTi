"""
The FullPOVMPureEffect class and supporting functionality.
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
from ..states.fullpurestate import FullPureState as _FullPureState


class FullPOVMPureEffect(_ConjugatedStatePOVMEffect):
    """
    TODO: docstring
    A "fully parameterized" effect vector where each element is an independent parameter.

    Parameters
    ----------
    vec : array_like or POVMEffect
        a 1D numpy array representing the POVM effect.  The
        shape of this array sets the dimension of the POVM effect.

    evotype : Evotype or str, optional
        The evolution type.  The special value `"default"` is equivalent
        to specifying the value of `pygsti.evotypes.Evotype.default_evotype`.
    """

    def __init__(self, vec, evotype="default"):
        _ConjugatedStatePOVMEffect.__init__(self, _FullPureState(vec, evotype))

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

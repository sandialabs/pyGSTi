"""
Defines the UnconstrainedPOVM class
"""
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

from pygsti.modelmembers.povms.basepovm import _BasePOVM
from pygsti.modelmembers.povms.effect import POVMEffect as _POVMEffect
from pygsti.baseobjs.statespace import StateSpace as _StateSpace


class UnconstrainedPOVM(_BasePOVM):
    """
    A POVM that just holds a set of effect vectors, parameterized individually however you want.

    Parameters
    ----------
    effects : dict of POVMEffects or array-like
        A dict (or list of key,value pairs) of the effect vectors.

    evotype : Evotype or str, optional
        The evolution type.  If `None`, the evotype is inferred
        from the first effect vector.  If `len(effects) == 0` in this case,
        an error is raised.

    state_space : StateSpace, optional
        The state space for this POVM.  If `None`, the space is inferred
        from the first effect vector.  If `len(effects) == 0` in this case,
        an error is raised.
    """

    def __init__(self, effects, evotype=None, state_space=None, called_from_reduce=False):
        super(UnconstrainedPOVM, self).__init__(effects, evotype, state_space, preserve_sum=False,
                                                called_from_reduce=called_from_reduce)

    def __reduce__(self):
        """ Needed for OrderedDict-derived classes (to set dict items) """
        assert(self.complement_label is None)
        effects = [(lbl, effect.copy()) for lbl, effect in self.items()]
        return (UnconstrainedPOVM, (effects, self.evotype, self.state_space, True),
                {'_gpindices': self._gpindices, '_submember_rpindices': self._submember_rpindices})

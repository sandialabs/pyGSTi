"""
Defines the TPPOVM class
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


class TPPOVM(_BasePOVM):
    """
    A POVM whose sum-of-effects is constrained to what, by definition, we call the "identity".

    Parameters
    ----------
    effects : dict of POVMEffects or array-like
        A dict (or list of key,value pairs) of the effect vectors.  The
        final effect vector will be stripped of any existing
        parameterization and turned into a ComplementPOVMEffect which has
        no additional parameters and is always equal to
        `identity - sum(other_effects`, where `identity` is the sum of
        `effects` when this __init__ call is made.

    evotype : Evotype or str, optional
        The evolution type.  If `None`, the evotype is inferred
        from the first effect vector.  If `len(effects) == 0` in this case,
        an error is raised.

    state_space : StateSpace, optional
        The state space for this POVM.  If `None`, the space is inferred
        from the first effect vector.  If `len(effects) == 0` in this case,
        an error is raised.
    """

    def __init__(self, effects, evotype=None, state_space=None):
        super(TPPOVM, self).__init__(effects, evotype, state_space, preserve_sum=True)

    def __reduce__(self):
        """ Needed for OrderedDict-derived classes (to set dict items) """
        assert(self.complement_label is not None)
        effects = [(lbl, effect.copy()) for lbl, effect in self.items()
                   if lbl != self.complement_label]

        #add complement effect as a std numpy array - it will get
        # re-created correctly by __init__ w/preserve_sum == True
        effects.append((self.complement_label,
                        self[self.complement_label].to_dense().reshape((-1, 1))))

        return (TPPOVM, (effects, self.evotype, self.state_space), {'_gpindices': self._gpindices})

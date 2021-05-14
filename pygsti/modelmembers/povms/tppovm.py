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

from .basepovm import _BasePOVM


class TPPOVM(_BasePOVM):
    """
    A POVM whose sum-of-effects is constrained to what, by definition, we call the "identity".

    Parameters
    ----------
    effects : dict of SPAMVecs or array-like
        A dict (or list of key,value pairs) of the effect vectors.  The
        final effect vector will be stripped of any existing
        parameterization and turned into a ComplementSPAMVec which has
        no additional parameters and is always equal to
        `identity - sum(other_effects`, where `identity` is the sum of
        `effects` when this __init__ call is made.
    """

    def __init__(self, effects):
        """
        Creates a new POVM object.

        Parameters
        ----------
        effects : dict of SPAMVecs or array-like
            A dict (or list of key,value pairs) of the effect vectors.  The
            final effect vector will be stripped of any existing
            parameterization and turned into a ComplementSPAMVec which has
            no additional parameters and is always equal to
            `identity - sum(other_effects`, where `identity` is the sum of
            `effects` when this __init__ call is made.
        """
        super(TPPOVM, self).__init__(effects, preserve_sum=True)

    def __reduce__(self):
        """ Needed for OrderedDict-derived classes (to set dict items) """
        assert(self.complement_label is not None)
        effects = [(lbl, effect.copy()) for lbl, effect in self.items()
                   if lbl != self.complement_label]

        #add complement effect as a std numpy array - it will get
        # re-created correctly by __init__ w/preserve_sum == True
        effects.append((self.complement_label,
                        self[self.complement_label].to_dense().reshape((-1, 1))))

        return (TPPOVM, (effects,), {'_gpindices': self._gpindices})

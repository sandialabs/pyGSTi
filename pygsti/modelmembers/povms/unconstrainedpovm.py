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

    def __init__(self, effects, evotype=None, state_space=None):
        super(UnconstrainedPOVM, self).__init__(effects, evotype, state_space, preserve_sum=False)

    def __reduce__(self):
        """ Needed for OrderedDict-derived classes (to set dict items) """
        assert(self.complement_label is None)
        effects = [(lbl, effect.copy()) for lbl, effect in self.items()]
        return (UnconstrainedPOVM, (effects, self.evotype, self.state_space), {'_gpindices': self._gpindices})

    def to_memoized_dict(self, mmg_memo):
        """Create a serializable dict with references to other objects in the memo.

        Parameters
        ----------
        mmg_memo: dict
            Memo dict from a ModelMemberGraph, i.e. keys are object ids and values
            are ModelMemberGraphNodes (which contain the serialize_id). This is NOT
            the same as other memos in ModelMember (e.g. copy, allocate_gpindices, etc.).

        Returns
        -------
        mm_dict: dict
            A dict representation of this ModelMember ready for serialization
            This must have at least the following fields:
                module, class, submembers, params, state_space, evotype
            Additional fields may be added by derived classes.
        """
        mm_dict = super().to_memoized_dict(mmg_memo)

        mm_dict['effects'] = [(lbl, effect.to_memoized_dict({})) for lbl, effect in self.items()]  # TEMPORARY!!!!!!

        return mm_dict

    @classmethod
    def from_memoized_dict(cls, mm_dict, serial_memo):
        """Deserialize a ModelMember object and relink submembers from a memo.

        Parameters
        ----------
        mm_dict: dict
            A dict representation of this ModelMember ready for deserialization
            This must have at least the following fields:
                module, class, submembers, state_space, evotype

        serial_memo: dict
            Keys are serialize_ids and values are ModelMembers. This is NOT the same as
            other memos in ModelMember, (e.g. copy(), allocate_gpindices(), etc.).
            This is similar but not the same as mmg_memo in to_memoized_dict(),
            as we do not need to build a ModelMemberGraph for deserialization.

        Returns
        -------
        ModelMember
            An initialized object
        """
        cls._check_memoized_dict(mm_dict, serial_memo)
        state_space = _StateSpace.from_nice_serialization(mm_dict['state_space'])
        effects = {lbl: _POVMEffect._state_class(effect).from_memoized_dict(effect, serial_memo)
                   for lbl, effect in mm_dict['effects']}
        return cls(effects, mm_dict['evotype'], state_space)

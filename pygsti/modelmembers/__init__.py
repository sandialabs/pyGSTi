"""
A sub-package containing the objects that are held within :class:`OpModel` models.
"""
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

import importlib as _importlib

from .modelmember import ModelMember, ModelChild


# This is available as a class method in every ModelMember-derived class
# This is purely a convenience function switching on the type
def from_memoized_dict(mm_dict, serial_memo):
    """Deserialize a ModelMember object and relink submembers from a memo.

    Parameters
    ----------
    mm_dict: dict
        A dict representation of this ModelMember ready for deserialization
        This must have at least the following fields:
            module, class, submembers, params, state_space, evotype

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
    # Any additional refactoring could add code here to reroute to new class locations
    # For now, this just needs to call from_memoized_dict on the correct class
    module = _importlib.import_module(mm_dict['module'])
    cls = getattr(module, mm_dict['class'])
    assert isinstance(cls, ModelMember), "Can only deserialize ModelMembers"

    return cls.from_memoized_dict(mm_dict, serial_memo)

"""
Defines the UnitaryGateFunction class and supporting functionality.
"""
#***************************************************************************************************
# Copyright 2015, 2019, 2025 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

from pygsti.baseobjs.nicelyserializable import NicelySerializable as _NicelySerializable


class UnitaryGateFunction(_NicelySerializable):
    """
    A convenient base class for building serializable "functions" for unitary gate matrices.

    Subclasses that don't need to initialize any attributes other than `shape` only need to
    impliement the `__call__` method and declare their shape as either a class or instance variable.
    """
    def __call__(self, arg):
        raise NotImplementedError("Derived classes should implement this!")

    def _to_nice_serialization(self):
        state = super()._to_nice_serialization()

        if not hasattr(self, 'shape'):
            raise ValueError('%s class need to have a .shape attribute!' % self.__class__.__name__)
        state['shape'] = self.shape
        return state

    @classmethod
    def _from_nice_serialization(cls, state):
        ret = cls()  # assumes no __init__ args
        ret.shape = tuple(state['shape'])
        return ret

    def __init__(self):
        super().__init__()

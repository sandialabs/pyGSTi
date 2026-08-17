"""
The StaticStandardOp class and supporting functionality.
"""
#***************************************************************************************************
# Copyright 2015, 2019, 2025 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************
import numpy as _np
import warnings as _warnings

from pygsti.modelmembers.operations.staticarbitraryop import StaticArbitraryOp as _StaticArbitraryOp
from pygsti.modelmembers.operations.linearop import LinearOperator as _LinearOperator
from pygsti.modelmembers.errorgencontainer import NoErrorGeneratorInterface as _NoErrorGeneratorInterface
from pygsti.modelmembers import term as _term
from pygsti.evotypes import Evotype as _Evotype
from pygsti.baseobjs import statespace as _statespace
from pygsti.baseobjs.basis import Basis as _Basis
from pygsti.baseobjs.polynomial import Polynomial as _Polynomial
from pygsti.tools import internalgates as _itgs
from pygsti import SpaceT
from pygsti.modelmembers.torchable import StaticTorchable as _StaticTorchable
from pygsti.tools.exceptions import pyGSTiDeprecationWarning as _pyGSTiDeprecationWarning
from pygsti.tools import optools as _ot


class StaticStandardOp(_StaticArbitraryOp):
    """
    [Deprecated] A static operation built from a standard gate name.

    `StaticStandardOp` is deprecated and will be removed in a future release.
    Use :class:`~pygsti.modelmembers.operations.StaticArbitraryOp` or its classmethod
    :meth:`~pygsti.modelmembers.operations.StaticArbitraryOp.from_standard_gate_name`
    instead.

    Parameters
    ----------
    name : str
        Standard gate name (as defined in `pygsti.tools.internalgates`).

    basis : Basis or {'pp','gm','std'}, optional
        The basis used to construct the Hilbert-Schmidt space representation
        of this operation as a super-operator.

    evotype : Evotype or str, optional
        The evolution type. The special value `"default"` is equivalent
        to specifying `pygsti.evotypes.Evotype.default_evotype`.

    state_space : StateSpace, optional
        The state space for this operation. If `None`, a default state space
        with the appropriate number of qubits is inferred from the standard gate.
    """

    def __init__(self, name, basis='pp', evotype="default", state_space=None):
        _warnings.warn(
            "StaticStandardOp is deprecated and will be removed in a future release. "
            "Please use StaticArbitraryOp.from_standard_gate_name(...) or StaticArbitraryOp(...) instead.",
            _pyGSTiDeprecationWarning,
            stacklevel=2,
        )

        std_unitaries = _itgs.standard_gatename_unitaries()
        if name not in std_unitaries:
            raise ValueError(f"'{name}' does not name a standard operation")

        U = std_unitaries[name]
        state_space = (
            _statespace.default_space_for_udim(U.shape[0])
            if (state_space is None)
            else _statespace.StateSpace.cast(state_space)
        )
        basis = _Basis.cast(basis, state_space.dim) if (basis is not None) else None
        superop = _ot.unitary_to_superop(U, basis)

        super().__init__(superop, basis=basis, evotype=evotype, state_space=state_space)
        self.name = name

    @classmethod
    def _from_memoized_dict(cls, mm_dict, serial_memo):
        """
        Deserialize legacy StaticStandardOp serialization dictionaries into StaticArbitraryOp instances.
        """
        _warnings.warn(
            "Deserializing legacy StaticStandardOp. StaticStandardOp is deprecated and will be "
            "removed in a future release; converting to StaticArbitraryOp.",
            _pyGSTiDeprecationWarning,
            stacklevel=2,
        )

        basis = (
            _Basis.from_nice_serialization(mm_dict['basis'])
            if (mm_dict.get('basis') is not None)
            else None
        )
        state_space = _statespace.StateSpace.from_nice_serialization(mm_dict['state_space'])

        return _StaticArbitraryOp.from_standard_gate_name(
            mm_dict['name'],
            basis=basis,
            evotype=mm_dict.get('evotype', 'default'),
            state_space=state_space,
        )
    
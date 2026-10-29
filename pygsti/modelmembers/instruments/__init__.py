"""
Sub-package holding model instrument objects.
"""
#***************************************************************************************************
# Copyright 2015, 2019, 2025 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************
from __future__ import annotations

from .instrument import Instrument
from .tpinstrument import TPInstrument
from .tpinstrumentop import TPInstrumentOp

from typing import Optional, TYPE_CHECKING
from pygsti.baseobjs.basis import BasisLike
from pygsti.modelmembers import operations as _op

if TYPE_CHECKING:
    InstrumentLike = Instrument | TPInstrument


def instrument_type_from_op_type(op_type: str | list[str]) -> str:
    """Decode an op type into an appropriate instrument type.

    Parameters:
    -----------
    op_type: str or list of str
        Operation parameterization type (or list of preferences)

    Returns
    -------
    instr_type_preferences: tuple of str
        POVM parameterization types
    """
    op_type_preferences = _op.verbose_type_from_op_type(op_type)
    return op_type_preferences[0]


def convert(
        instrument: InstrumentLike,
        to_type: str,
        basis: BasisLike,
        ideal_instrument: Optional[InstrumentLike]=None,  # pylint: disable=unused-argument
        flatten_structure: bool=False
    ) -> InstrumentLike:
    """
    Convert `instrument` to a new type of parameterization.
    This potentially creates a new object.

    Parameters
    ----------
    instrument : InstrumentLike
        Instrument to convert.

    to_type : str
        The type of parameterizaton to convert to. Supported types
        are "full", "full TP", "static", and all Lindblad types.

    basis : BasisLike
        The basis for Hilbert--Schmidt superoperators in instrument.values().

    ideal_instrument : Instrument, optional
        Currently ignored. This argument is retained only to provide polymorphism
        across pyGSTi's ModelMember `convert` functions.

    flatten_structure : bool, optional
        Only relevant when to_type is in {"full", "static"}. In these
        cases `flatten_structure` is forwarded to :func:`_op.convert` 
        when converting `instrument`'s constituent CPTR maps.

    Returns
    -------
    Instrument

    Raises
    ------
    ValueError for invalid conversions.
    """
    if not isinstance(to_type, str):
        if len(to_type) > 1:
            raise ValueError(f"Expected to_type to be a string, but got {to_type}")
        to_type = to_type[0]
        assert isinstance(to_type, str)

    if to_type == "full TP":
        if isinstance(instrument, TPInstrument):
            return instrument
        inst_arrays = dict()
        for k, v in instrument.items():
            if hasattr(v, 'to_dense'):
                inst_arrays[k] = v.to_dense('HilbertSchmidt')
            else:
                inst_arrays[k] = v
        members = list(inst_arrays.items())
        return TPInstrument(members, instrument.evotype, instrument.state_space)

    if to_type in ("full", "static"):
        members = []
        for k, g in instrument.items():
            g_conv  = _op.convert(g, to_type, basis, None, flatten_structure)
            members.append((k, g_conv))
        return Instrument(members, instrument.evotype, instrument.state_space)

    # Else, route Lindblad-type conversions (e.g. "CPTPLND", "GLND", "H+S", "H+s")
    # through the effect-then-CPTP-gate construction: a single CP-constrained POVM
    # for the effects {E_k} plus one parameterized post-measurement gate per outcome.
    # The POVM error map is always promoted to the minimal CP parameterization that
    # subsumes `to_type`, which is what keeps the whole instrument trace-preserving.
    op_arrays   = {k: v.to_dense('HilbertSchmidt') for (k, v) in instrument.items()}
    povm_errmap = _op.LindbladParameterization.minimal_cp_paramtype(to_type)
    out = Instrument.from_cptr_superops(
        op_arrays, basis, gate_parameterization=to_type, povm_errormap=povm_errmap
    )
    return out

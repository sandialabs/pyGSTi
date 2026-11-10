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
from .unitaryfactors import UnitaryApproximationReport, infer_unitary_factors

from typing import Optional, TYPE_CHECKING
from pygsti.baseobjs.basis import Basis as _Basis, BasisLike
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
        are "full", "full TP", "static", and all Lindblad types.  A Lindblad
        conversion reproduces the instrument's members or raises.  An instrument
        already in effect-then-gate form (as built by :meth:`Instrument.from_effects`
        and :meth:`Instrument.from_cptr_superops`, or by an earlier Lindblad
        conversion) keeps its effects and its gates' static unitaries, and its
        error maps are converted.  Any other instrument goes through
        :meth:`Instrument.from_cptr_superops`, so every member must have a single
        Kraus operator, as ideal instruments do; for a noisy instrument, build the
        replacement with :func:`infer_unitary_factors` and
        :meth:`Instrument.from_effects`.

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

    # Else, a Lindblad-type conversion (e.g. "CPTPLND", "GLND", "H+S", "H+s") produces
    # the effect-then-unitary form: a single CP-constrained POVM for the effects {E_k}
    # plus one parameterized post-measurement gate per outcome, around a static unitary.
    # The POVM error map is always promoted to the minimal CP parameterization that
    # subsumes `to_type`, which is what keeps the whole instrument trace-preserving.
    # The conversion is exact.  An instrument already in that form is converted part by
    # part; any other is factored from its dense members, which raises unless every
    # member has a single Kraus operator (as ideal instruments do).
    from pygsti.modelmembers.instruments._construction import (
        _effect_then_gate_parts, _converted_effect_then_gate_members
    )
    povm_errmap = _op.LindbladParameterization.minimal_cp_paramtype(to_type)
    parts = _effect_then_gate_parts(instrument) if isinstance(instrument, Instrument) else None
    if parts is not None:
        basis = _Basis.cast(basis, instrument.state_space.dim)
        return Instrument(_converted_effect_then_gate_members(parts, to_type, povm_errmap, basis))
    op_arrays   = {k: v.to_dense('HilbertSchmidt') for (k, v) in instrument.items()}
    try:
        out = Instrument.from_cptr_superops(
            op_arrays, basis, gate_parameterization=to_type, povm_errormap=povm_errmap
        )
    except ValueError as e:
        raise ValueError(
            f"Cannot convert this instrument to {to_type!r}; see the error above. In a model, first replace each "
            "instrument by its effect-preserving unitary approximation:\n"
            "    from pygsti.modelmembers.instruments import Instrument, infer_unitary_factors\n"
            "    for lbl, inst in list(mdl.instruments.items()):\n"
            "        pairs, reports = infer_unitary_factors(inst, mdl.basis)\n"
            "        mdl.instruments[lbl] = Instrument.from_effects(pairs, mdl.basis)\n"
            f"    mdl.set_all_parameterizations({to_type!r})") from e
    return out

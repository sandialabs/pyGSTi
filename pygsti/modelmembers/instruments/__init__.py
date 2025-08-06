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

from .instrument import Instrument
from .tpinstrument import TPInstrument
from .tpinstrumentop import TPInstrumentOp

from pygsti.tools import optools as _ot

# Avoid circular import
import pygsti.modelmembers as _mm


def instrument_type_from_op_type(op_type):
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
    op_type_preferences = _mm.operations.verbose_type_from_op_type(op_type)

    # Limited set (only matching what is in convert)
    instr_conversion = {
        'auto': 'full',
        'static unitary': 'static unitary',
        'static clifford': 'static clifford',
        'static': 'static',
        'full': 'full',
        'full TP': 'full TP',
        'full CPTP': 'full CPTP',
        'full unitary': 'full unitary',
    }

    instr_type_preferences = []
    for typ in op_type_preferences:
        instr_type = None
        if _ot.is_valid_lindblad_paramtype(typ):
            # Lindblad types are passed through as TP only (matching current convert logic)
            instr_type = "full TP"
        else:
            instr_type = instr_conversion.get(typ, None)

        if instr_type is None:
            continue

        if instr_type not in instr_type_preferences:
            instr_type_preferences.append(instr_type)

    if len(instr_type_preferences) == 0:
        raise ValueError(
            'Could not convert any op types from {}.\n'.format(op_type_preferences)
            + '\tKnown op_types: Lindblad types or {}\n'.format(sorted(list(instr_conversion.keys())))
            + '\tValid instrument_types: Lindblad types or {}'.format(sorted(list(set(instr_conversion.values()))))
        )

    return instr_type_preferences


def convert(instrument, to_type, basis, ideal_instrument=None, flatten_structure=False):
    """
    TODO: update docstring
    Convert intrument to a new type of parameterization.

    This potentially creates a new object.
    Raises ValueError for invalid conversions.

    Parameters
    ----------
    instrument : Instrument
        Instrument to convert

    to_type : {"full","TP","static","static unitary"}
        The type of parameterizaton to convert to.  See
        :meth:`Model.set_all_parameterizations` for more details.

    basis : {'std', 'gm', 'pp', 'qt'} or Basis object
        The basis for `povm`.  Allowed values are Matrix-unit (std),
        Gell-Mann (gm), Pauli-product (pp), and Qutrit (qt)
        (or a custom basis object).

    ideal_instrument : Instrument, optional
        The ideal version of `instrument`, potentially used when
        converting to an error-generator type.

    flatten_structure : bool, optional
        When `False`, the sub-members of composed and embedded operations
        are separately converted, leaving the original instrument's structure
        unchanged.  When `True`, composed and embedded operations are "flattened"
        into a single instrument of the requested `to_type`.

    Returns
    -------
    Instrument
        The converted instrument, usually a distinct
        object from the object passed as input.
    """
    to_types = to_type if isinstance(to_type, (tuple, list)) else (to_type,)  # HACK to support multiple to_type values
    destination_types = {'full TP': TPInstrument}
    NoneType = type(None)

    for to_type in to_types:
        try:
            if isinstance(instrument, destination_types.get(to_type, NoneType)):
                return instrument

            if to_type == "full TP":
                return TPInstrument(list(instrument.items()), instrument.evotype, instrument.state_space)
            elif to_type in ("full", "static", "static unitary"):
                from ..operations import convert as _op_convert
                ideal_items = dict(ideal_instrument.items()) if (ideal_instrument is not None) else {}
                members = [(k, _op_convert(g, to_type, basis, ideal_items.get(k, None), flatten_structure))
                           for k, g in instrument.items()]
                return Instrument(members, instrument.evotype, instrument.state_space)
            else:
                raise ValueError("Cannot convert an instrument to type %s" % to_type)
        except:
            pass  # try next to_type

    raise ValueError("Could not convert instrument to to type(s): %s" % str(to_types))

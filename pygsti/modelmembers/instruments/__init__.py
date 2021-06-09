"""
Sub-package holding model instrument objects.
"""
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

from .instrument import Instrument
from .tpinstrument import TPInstrument
from .tpinstrumentop import TPInstrumentOp


def convert(instrument, to_type, basis, extra=None):
    """
    Convert intrument to a new type of parameterization.

    This potentially creates a new object.
    Raises ValueError for invalid conversions.

    Parameters
    ----------
    instrument : Instrument
        Instrument to convert

    to_type : {"full","TP","static","static unitary"}
        The type of parameterizaton to convert to.  See
        :method:`Model.set_all_parameterizations` for more details.

    basis : {'std', 'gm', 'pp', 'qt'} or Basis object
        The basis for `povm`.  Allowed values are Matrix-unit (std),
        Gell-Mann (gm), Pauli-product (pp), and Qutrit (qt)
        (or a custom basis object).

    extra : object, optional
        Additional information for conversion.

    Returns
    -------
    Instrument
        The converted instrument, usually a distinct
        object from the object passed as input.
    """

    if to_type == "TP":
        if isinstance(instrument, TPInstrument):
            return instrument
        else:
            return TPInstrument(list(instrument.items()), instrument.evotype, instrument.state_space)
    elif to_type in ("full", "static", "static unitary"):
        from ..operations import convert as _op_convert
        gate_list = [(k, _op_convert(g, to_type, basis)) for k, g in instrument.items()]
        return Instrument(gate_list, instrument.evotype, instrument.state_space)
    else:
        raise ValueError("Cannot convert an instrument to type %s" % to_type)

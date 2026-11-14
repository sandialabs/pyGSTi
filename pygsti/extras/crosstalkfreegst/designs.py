#***************************************************************************************************
# Copyright 2015, 2019, 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

"""
Simple experiment designs with mid-circuit measurements.

This code is not part of crosstalk-free GST itself: nothing in the fitting, simulation or
analysis code uses it.  It is here because pyGSTi has no ready-made GST designs containing
mid-circuit measurements, and the tests and the tutorial of this package need some -- in
particular a two-qubit one, for simultaneous GST designs.

Like the model packs they are built from, these designs write an idle as an empty circuit
layer.  They are meant as the one- and two-qubit component designs of
:func:`pygsti.protocols.make_simultaneous_gst_design`, which turns those layers into explicit
idle gates (``Gi`` on one qubit, ``Gii`` on two).  The crosstalk-free GST code needs explicit
idles -- it rejects circuits with empty layers -- so to use a design's circuits directly, replace
their empty layers first.
"""
from pygsti.baseobjs import Label as _Label
from pygsti.circuits import Circuit as _Circuit
from pygsti.modelpacks import smq1Q_XYI as _smq1Q_XYI
from pygsti.modelpacks import smq2Q_XYICNOT as _smq2Q_XYICNOT
from pygsti.processors import QubitProcessorSpec as _QubitProcessorSpec
from pygsti.protocols import GateSetTomographyDesign as _GateSetTomographyDesign
from pygsti.protocols import StandardGSTDesign as _StandardGSTDesign


def create_simple_mcm_oneq_design(max_max_length: int, instrument_name: str = 'Iz', qubit_label=0):
    """
    A simple single-qubit GST design for the X/2, Y/2, idle gate set plus a mid-circuit measurement.
    The idle is an empty layer (see the module docstring).

    It is the standard design built from the ``smq1Q_XYI`` model pack's fiducials and (lite)
    germs, with the instrument by itself as one more germ -- the recipe of pyGSTi's instruments
    tutorial.  This design is for demonstrations: it determines every gauge-invariant parameter
    of a single qubit's gates, SPAM and instrument (checked numerically, from the rank of the
    Jacobian of the outcome probabilities), but it was not optimized in any way.

    In a :class:`~pygsti.protocols.SimultaneousGSTDesign`, every qubit's instrument must be
    exercised somewhere.  A qubit that every patch of the edge coloring covers (e.g. each
    interior qubit of a line) never runs one-qubit component circuits, so pair this design with
    :func:`create_simple_mcm_twoq_design`, which puts mid-circuit measurements in the two-qubit
    design, or add circuits that run one-qubit measurement circuits on those qubits.

    Parameters
    ----------
    max_max_length : int
        The longest germ power; the design uses powers of 2 up to it.

    instrument_name : str, optional
        The instrument's name.  Circuits use ``Label(instrument_name, qubit_label)``.

    qubit_label : int or str, optional

    Returns
    -------
    StandardGSTDesign
    """
    q = (qubit_label,)
    pspec = _QubitProcessorSpec(1, list(_smq1Q_XYI.processor_spec(q).gate_names),
                                instrument_names=(instrument_name,), qubit_labels=q)
    mcm_germ = _Circuit([_Label(instrument_name, qubit_label)], line_labels=q)
    max_lengths = [2**i for i in range(max_max_length.bit_length()) if 2**i <= max_max_length]
    return _StandardGSTDesign(pspec, _smq1Q_XYI.prep_fiducials(q), _smq1Q_XYI.meas_fiducials(q),
                              list(_smq1Q_XYI.germs(q, lite=True)) + [mcm_germ], max_lengths, qubit_labels=q)


def create_simple_mcm_twoq_design(max_max_length: int, instrument_name: str = 'Iz', modelpack=None):
    """
    A simple two-qubit GST design for a two-qubit model pack's gate set plus mid-circuit measurements.

    It is the model pack's standard design, plus -- on each of the two qubits, with the other
    idling -- the circuits of :func:`create_simple_mcm_oneq_design` that contain the
    instrument.  Like that design, it is meant for demonstrations and is not optimized, and its
    idles are empty layers (see the module docstring).  On qubits ``(0, 1)``.

    Parameters
    ----------
    max_max_length : int
        The longest germ power.

    instrument_name : str, optional

    modelpack : module, optional
        A two-qubit pyGSTi model pack, e.g. ``pygsti.modelpacks.smq2Q_XYCPHASE`` for a CZ gate set.
        Defaults to ``smq2Q_XYICNOT``.

    Returns
    -------
    GateSetTomographyDesign
    """
    modelpack = _smq2Q_XYICNOT if modelpack is None else modelpack
    base = modelpack.create_gst_experiment_design(max_max_length=max_max_length, qubit_labels=(0, 1))
    oneq = create_simple_mcm_oneq_design(max_max_length, instrument_name, qubit_label=0)
    if len(oneq.circuit_lists) != len(base.circuit_lists):
        raise ValueError("The one- and two-qubit designs have different numbers of circuit lists.")
    lists = []
    for base_list, oneq_list in zip(base.circuit_lists, oneq.circuit_lists):
        mcm = [c for c in oneq_list
               if any(comp.name == instrument_name for layer in c.layertup for comp in layer.components)]
        added = [_Circuit(c.map_state_space_labels({0: q}).layertup, line_labels=(0, 1)) for q in (0, 1) for c in mcm]
        lists.append(list(base_list) + added)
    base_pspec = modelpack.processor_spec((0, 1))
    pspec = _QubitProcessorSpec(2, list(base_pspec.gate_names),
                                availability={g: base_pspec.resolved_availability(g) for g in base_pspec.gate_names},
                                instrument_names=(instrument_name,), qubit_labels=(0, 1))
    return _GateSetTomographyDesign(pspec, lists, qubit_labels=(0, 1), nested=True)

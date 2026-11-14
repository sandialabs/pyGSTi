#***************************************************************************************************
# Copyright 2015, 2019, 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

"""
Constructing the general crosstalk-free model, and adding single-qubit instruments.
"""
import numpy as _np

from pygsti.baseobjs import Label as _Label
from pygsti.baseobjs import QubitSpace as _QubitSpace
from pygsti.models import create_crosstalk_free_model as _create_crosstalk_free_model
from pygsti.modelmembers import instruments as _inst
from pygsti.modelmembers import operations as _op
from pygsti.modelmembers import povms as _povm
from pygsti.modelmembers import states as _state

from .lanemodels import _decompose_povm, _decompose_prep


def create_general_crosstalk_free_model(processor_spec, evotype='default', simulator='auto', instruments=None):
    """
    The most general crosstalk-free model on `processor_spec`'s qubits.

    Every gate placement is an arbitrary trace-preserving map on its target qubits (a
    ``FullTPOp``, initialized to the ideal gate), and every qubit has its own arbitrary
    trace-preserving error map after an ideal computational-basis preparation (``|0>``, for
    the usual ``'rho0'``) and before an ideal computational-basis measurement.  Each
    preparation and measurement that `processor_spec` declares gets its own error maps.
    This is the model class whose gauge group, among
    gauges that preserve the crosstalk-free structure, is a tensor product of single-qubit
    gauges.  ``pygsti.models.create_crosstalk_free_model`` cannot express arbitrary per-qubit
    SPAM by itself, hence this function.

    The SPAM error maps over-parameterize: of a qubit's 12 prep-error parameters only the 3
    of the prepared state are identifiable, and of its 12 measurement-error parameters only
    the 4 of its two-outcome POVM.  The rest are flat directions of the likelihood.

    It also works as a lane-model factory: ``LaneModels(pspec, create_general_crosstalk_free_model)``.
    Built directly on many qubits it allocates a dense ``4**n`` state vector, so use the
    factory form beyond ~12 qubits.

    Parameters
    ----------
    processor_spec : QubitProcessorSpec

    evotype : str or Evotype, optional

    simulator : str or ForwardSimulator, optional

    instruments : dict, optional
        Single-qubit instruments (mid-circuit measurements) to add, as for
        :func:`add_single_qubit_instruments`; they are parameterized as full TP instruments,
        consistent with the gates.  For example ``{'Iz': 'Z'}`` adds an ideal Z-basis
        measurement ``Label('Iz', q)`` on every qubit ``q``.

    Returns
    -------
    LocalNoiseModel
    """
    model = _create_crosstalk_free_model(processor_spec, ideal_gate_type='full TP', independent_gates=True,
                                         evotype=evotype, simulator=simulator)
    qubits = tuple(processor_spec.qubit_labels)
    state_space, evotype = model.state_space, model.evotype

    def per_qubit_errors():
        def identity_map(q):
            return _op.FullTPOp(_np.identity(4), 'pp', evotype, _QubitSpace((q,)))
        return _op.ComposedOp([_op.EmbeddedOp(state_space, (q,), identity_map(q)) for q in qubits])

    for lbl, prep in list(model.prep_blks['layers'].items()):
        zvals, _ = _decompose_prep(prep, f"prep {lbl}")  # an error unless it is a computational basis state
        model.prep_blks['layers'][lbl] = _state.ComposedState(
            _state.ComputationalBasisState(zvals, 'pp', evotype, state_space), per_qubit_errors())
    for lbl, povm in list(model.povm_blks['layers'].items()):
        _decompose_povm(povm, f"POVM {lbl}")  # an error unless it is the computational-basis POVM
        model.povm_blks['layers'][lbl] = _povm.ComposedPOVM(
            per_qubit_errors(), _povm.ComputationalBasisPOVM(len(qubits), evotype, state_space=state_space), 'pp')
    if instruments:
        add_single_qubit_instruments(model, instruments, parameterization='full TP')
    return model


def ideal_z_measurement():
    """
    The superoperators (normalized Pauli-product basis) of an ideal single-qubit Z-basis
    measurement that leaves the qubit in the measured state: ``{'0': P0, '1': P1}`` with
    ``P_k(rho) = |k><k| rho |k><k|``.
    """
    r = 1 / _np.sqrt(2)
    zero, one = _np.array([r, 0, 0, r]), _np.array([r, 0, 0, -r])
    return {'0': _np.outer(zero, zero), '1': _np.outer(one, one)}


def add_single_qubit_instruments(model, instruments, qubits=None, parameterization='auto'):
    """
    Add independent single-qubit instruments (mid-circuit measurements) to a crosstalk-free model.

    For each instrument name and qubit ``q``, the model gets an instrument labeled
    ``Label(name, q)`` in ``instrument_blks['layers']``, whose members are its single-qubit
    elements embedded on ``q``.  Each qubit's instrument has its own parameters, so it is a
    separate member (``('instrument', Label(name, q))``) of the crosstalk-free model, and a
    product of per-qubit gauges keeps it single-qubit.  Circuits use qubit-qualified labels,
    e.g. ``('Iz', 0)``; outcome labels follow pyGSTi's convention (one string per instrument
    in the circuit, then the final bit string).

    pyGSTi's ``create_crosstalk_free_model`` does not build instruments (and processor specs
    cannot yet declare where instruments are available), hence this function.  The names must
    be declared in the processor spec's `instrument_names`, which is how the lane-splitting and
    data-handling code tell instruments from gates; an undeclared name is an error.

    Parameters
    ----------
    model : LocalNoiseModel
        Modified in place (and returned).

    instruments : dict
        Instrument name -> specification: ``'Z'`` for :func:`ideal_z_measurement`, a dict of
        outcome label -> 4x4 superoperator (normalized Pauli-product basis), or a single-qubit
        pyGSTi ``Instrument``.

    qubits : sequence or dict, optional
        The qubits to add each instrument to: a sequence for every instrument, or a dict
        mapping instrument names to sequences.  Defaults to all of the model's qubits.

    parameterization : str, optional
        Any instrument type ``pygsti.modelmembers.instruments.convert`` accepts, e.g.
        ``'full TP'`` or ``'CPTPLND'`` (pyGSTi's effect-then-gate CPTP parameterization).
        ``'auto'`` picks ``'full TP'`` when the model's gates are densely parameterized and
        ``'CPTPLND'`` otherwise.

    Returns
    -------
    LocalNoiseModel
    """
    state_space, evotype = model.state_space, model.evotype
    undeclared = sorted(set(instruments) - set(model.processor_spec.instrument_names))
    if undeclared:
        raise ValueError(f"Instruments {undeclared} are not among the processor spec's instrument_names "
                         f"{tuple(model.processor_spec.instrument_names)}; declare them when creating the spec.")
    if parameterization == 'auto':
        dense = all(isinstance(g, _op.DenseOperator) for g in model.operation_blks['gates'].values())
        parameterization = 'full TP' if dense else 'CPTPLND'
    for name, spec in instruments.items():
        if isinstance(spec, str):
            if spec != 'Z':
                raise ValueError(f"Unknown instrument specification {spec!r}; use 'Z', a dict of superoperators, "
                                 "or an Instrument")
            spec = ideal_z_measurement()
        where = qubits.get(name, state_space.qubit_labels) if isinstance(qubits, dict) else qubits
        for q in (state_space.qubit_labels if where is None else where):
            single = _inst.Instrument({k: _np.asarray(v.to_dense() if hasattr(v, 'to_dense') else v, float)
                                       for k, v in spec.items()}, evotype=evotype, state_space=_QubitSpace((q,)))
            single = _inst.convert(single, parameterization, 'pp')
            model.instrument_blks['layers'][_Label(name, q)] = _inst.Instrument(
                {k: _op.EmbeddedOp(state_space, (q,), member) for k, member in single.items()})
    model._clean_paramvec()
    model._reinit_opcaches()
    return model

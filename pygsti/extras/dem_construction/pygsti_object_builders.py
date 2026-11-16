""" Stim-to-pyGSTi circuit conversion and pyGSTi model builders for DEM construction """
#***************************************************************************************************
# Copyright 2015, 2019, 2025 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

import stim
import pygsti

from pygsti.baseobjs import Label, QubitSpace, BuiltinBasis
from pygsti.models import LocalNoiseModel
from pygsti.modelmembers.operations import ComposedOp, LindbladErrorgen, ExpErrorgenOp, StaticCliffordOp, EmbeddedErrorgen, ComposedErrorgen
from pygsti.modelmembers.states import ComposedState, ComputationalBasisState
from pygsti.modelmembers.povms import ComposedPOVM, ComputationalBasisPOVM
from pygsti.tools.internalgates import standard_gatenames_stim_conversions

try:  # the compiled stabilizer evotype needs pyGSTi's Cython extensions; fall back to the pure-Python one
    from pygsti.evotypes import stabilizer as _stabilizer_ext  # noqa: F401
    DEFAULT_EVOTYPE = 'stabilizer'
except ImportError:
    DEFAULT_EVOTYPE = 'stabilizer_slow'


def create_processor_spec(pcircuit, qubit_labels, gates=['Gcnot','Gh'], virtual_to_physical=None):
    """
    Creates a processor spec that from a pyGSTi syndrome extraction circuit, containing the 
    required gates (CNOTs between qubits that are coupled in those circuits). Will then be
    used to define the noise model we simulate.

    If `virtual_to_physical` (as returned by `stim_to_pygsti_circuit`) is given, each
    connection is mapped through it and duplicates are dropped, so that the spec is on the
    stim qubits, and `qubit_labels` should be those.
    """
    ###Note to future self: watch out for the possibility of wanting other gates###
    pyg_cstr = pcircuit.str
    # Find all the CNOT gates used in the circuit, using brute force string comprehension
    cleaned_pcstr = pyg_cstr.replace(']', '').replace('[', '').split('@')[0]
    connections = [(int(s.split(':')[1]), int(s.split(':')[2])) for s in cleaned_pcstr.split('G') if len(s.split(':')) > 2]
    if virtual_to_physical is not None:
        connections = list(dict.fromkeys((virtual_to_physical[a], virtual_to_physical[b]) for a, b in connections))
    availability={'Gcnot':connections, 'Gcphase':connections}
    pspec = pygsti.processors.QubitProcessorSpec(len(qubit_labels), gates, qubit_labels=qubit_labels,
                                             availability=availability)
    
    return pspec


def local_x_errorgen(num_qubits, rate, evotype=DEFAULT_EVOTYPE):
    state_space = QubitSpace(1)
    basis = BuiltinBasis('PP', state_space)
    errdict = {('S', bl): rate for bl in basis.labels[1:] if 'Y' not in bl and 'Z' not in bl}
    errgen = LindbladErrorgen.from_elementary_errorgens(
        errdict, "D", basis, mx_basis='pp',
        truncate=False, evotype=evotype, state_space=state_space)
    embedded_errorgens = []
    for i in range(num_qubits):
        embedded_errorgens.append(EmbeddedErrorgen(QubitSpace(num_qubits), [i], errgen))
    composed_errorgen = ComposedErrorgen(embedded_errorgens, evotype=evotype, state_space=QubitSpace(num_qubits))    
        
    return composed_errorgen
    

def build_model(error_rates, pspec, oneQ_gate_names, twoQ_gate_names, meas_err=0, prep_err=0,
                evotype=DEFAULT_EVOTYPE):
    """
    Builds a model with Lindblad gate noise (`error_rates`, per gate name), an S(X) preparation
    error `prep_err` on every qubit, and an S(X) measurement error `meas_err`. Its `evotype` is
    `stabilizer`, or `stabilizer_slow` when pyGSTi's extensions aren't built.

    The measurement error is a one-qubit POVM on purpose: pyGSTi builds every effect of an
    n-qubit POVM when a model is made, which is infeasible beyond about 14 qubits.
    `build_dem_end2end` reads it as the same error on every qubit.
    """
    gate_dictionary = dict()

    
    def create_gate_object(erates, num_qubits, gate_name):
        state_space = QubitSpace(num_qubits)
        lindblad_errorgen = LindbladErrorgen.from_elementary_errorgens(erates,
                                                                       state_space=state_space,
                                                                       evotype=evotype)
        perfect_gate = StaticCliffordOp(pspec.gate_unitaries[gate_name], evotype=evotype)
        exp_errorgen = ExpErrorgenOp(lindblad_errorgen)
        gate_mm = ComposedOp([perfect_gate, exp_errorgen])
        return gate_mm
    
    for i, gate_name in enumerate(oneQ_gate_names):         
        for j, q in enumerate(pspec.qubit_labels):
            label = pygsti.baseobjs.label.Label(gate_name, q)
            gate_dictionary[label] = create_gate_object(error_rates[gate_name], 1, gate_name)

    for i, gate_name in enumerate(twoQ_gate_names):
        for j, qs in enumerate(pspec.availability[gate_name]):
            label = pygsti.baseobjs.label.Label(gate_name, qs)
            gate_dictionary[label] = create_gate_object(error_rates[gate_name], 2, gate_name)

    nqs = pspec.num_qubits

    povm_errorgen = LindbladErrorgen.from_elementary_errorgens({('S', 'X'): meas_err}, state_space=QubitSpace(1), evotype=evotype)

    povm_layers = {Label('Mdefault'): ComposedPOVM(ExpErrorgenOp(povm_errorgen), ComputationalBasisPOVM(1, evotype=evotype, state_space=QubitSpace(1)))}
    
    prep_errorgen = local_x_errorgen(nqs, prep_err, evotype) 

    prep_layers = {Label('rho0'): ComposedState(ComputationalBasisState(zvals = ['0']*nqs, evotype=evotype), ExpErrorgenOp(prep_errorgen))}

    model = LocalNoiseModel(pspec, gate_dictionary, prep_layers, povm_layers, evotype=evotype)
    return model


def _pygsti_gate(stim_name, table):
    """
    Returns `(pyGSTi gate name, qubit count)` for the unitary one- or two-qubit stim gate
    `stim_name`, matching its tableau against `table` (the result of
    `standard_gatenames_stim_conversions()`); the first match in table order wins.
    Returns None for any other gate.
    """
    data = stim.gate_data(stim_name)
    # `.tableau` raises for gates such as SPP, so check the gate's shape first.
    if not (data.is_unitary and (data.is_single_qubit_gate or data.is_two_qubit_gate)):
        return None
    tableau = data.tableau
    for pyg_name, pyg_tableau in table.items():
        if pyg_tableau == tableau:
            return pyg_name, (1 if data.is_single_qubit_gate else 2)
    return None


def stim_to_pygsti_circuit(circuit, include_idles=False, include_meas_idles=False):
    """
    Converts a non-adaptive `stim.Circuit` into a pyGSTi circuit in which every reset starts a
    fresh virtual qubit and all measurements are deferred to the end.

    Parameters
    ----------
    circuit : stim.Circuit
        The circuit. Noise, non-Z-basis resets and measurements, `MPP`, feedback and gates
        without a pyGSTi equivalent raise a `ValueError`, as does reusing a measured qubit
        without resetting it.

    include_idles : bool
        Add an idle gate on every live qubit not touched by each gate layer.

    include_meas_idles : bool
        Add a layer of idle gates on every live qubit not touched by each `M`, `R` or `MR`
        instruction.

    Returns
    -------
    pyg_circuit : pygsti.circuits.Circuit
        Circuit on virtual qubits `0 .. n_virtual-1`, without measurements.
    detectors : list of stim.PauliString
        Stim detector `D<i>` as a signed Pauli on the virtual qubits.
    observables : list of stim.PauliString
        Stim observable `L<j>` as a signed Pauli on the virtual qubits.
    virtual_to_physical : list of int
        The stim qubit each virtual qubit stands for.
    """
    table = standard_gatenames_stim_conversions()
    idle_name = _pygsti_gate('I', table)[0]
    n_phys = circuit.num_qubits
    virt = list(range(n_phys))
    virtual_to_physical = list(range(n_phys))
    FRESH, ACTIVE, MEASURED = 'fresh', 'active', 'measured'
    state = [FRESH] * n_phys

    layers = []
    measurements = []  # (virtual qubit, inverted)
    detector_idx = []
    observable_idx = {}

    def idles(skip):
        return [(idle_name, virt[q]) for q in range(n_phys) if state[q] != MEASURED and q not in skip]

    def reuse_error(inst, q):
        return ValueError(f"Qubit {q} is used after being measured without a reset in "
                          f"instruction '{inst}'; add an `R` (or `MR`) on it first.")

    for inst in circuit.flattened():
        name = inst.name
        targets = inst.targets_copy()
        if name in ('TICK', 'QUBIT_COORDS', 'SHIFT_COORDS'):
            # Note for future development: TICK information is relevant for time-dependent error
            continue

        if name in ('DETECTOR', 'OBSERVABLE_INCLUDE'):
            if not all(t.is_measurement_record_target for t in targets):
                raise ValueError(f"Unsupported instruction '{inst}': only measurement record "
                                 "targets are supported.")
            idx = [len(measurements) + t.value for t in targets]
            if any(i < 0 for i in idx):
                raise ValueError(f"Unsupported instruction '{inst}': refers to a measurement "
                                 "before the start of the circuit.")
            if name == 'DETECTOR':
                detector_idx.append(idx)
            else:
                observable_idx.setdefault(int(inst.gate_args_copy()[0]), []).extend(idx)
            continue

        if name in ('M', 'R', 'MR'):
            if any(a != 0 for a in inst.gate_args_copy()):
                raise ValueError(f"Noise in instruction '{inst}': measurement flip probabilities "
                                 "are not supported; remove noise with `stim.Circuit.without_noise()`.")
            if not all(t.is_qubit_target for t in targets):
                raise ValueError(f"Unsupported instruction '{inst}': non-qubit target.")
            for t in targets:
                q = t.value
                if name != 'R':
                    if state[q] == MEASURED:
                        raise reuse_error(inst, q)
                    measurements.append((virt[q], t.is_inverted_result_target))
                    state[q] = MEASURED
                if name != 'M' and state[q] != FRESH:
                    virt[q] = len(virtual_to_physical)
                    virtual_to_physical.append(q)
                    state[q] = FRESH
            if include_meas_idles and targets:
                touched = {t.value for t in targets}
                layer = idles(touched)
                if layer:
                    layers.append(layer)
            continue

        # X- and Y-basis resets and measurements are refused rather than rewritten as Z-basis
        # ones plus H gates: those H gates would pick up the model's H-gate noise, which the
        # stim circuit never asked for. Whether to add them is the user's visible choice, made
        # by rewriting the stim circuit.
        match = _pygsti_gate(name, table)
        if match is None:
            data = stim.gate_data(name)
            if data.is_noisy_gate and not data.produces_measurements:
                raise ValueError(f"Noise in instruction '{inst}': stim noise channels are not "
                                 "supported; remove noise with `stim.Circuit.without_noise()`.")
            raise ValueError(f"Unsupported instruction '{inst}': no matching pyGSTi gate or "
                             "not a Z-basis reset or measurement.")
        if not all(t.is_qubit_target and not t.is_inverted_result_target for t in targets):
            raise ValueError(f"Unsupported instruction '{inst}': classically controlled gates "
                             "(`rec[...]` or `sweep[...]` targets) are not supported.")
        pyg_name, arity = match

        layer, used = [], set()

        def flush():
            if include_idles:
                layer.extend(idles(used))
            layers.append(list(layer))

        for i in range(0, len(targets), arity):
            qs = [t.value for t in targets[i:i + arity]]
            if any(state[q] == MEASURED for q in qs):
                raise reuse_error(inst, next(q for q in qs if state[q] == MEASURED))
            if used & set(qs):
                flush()
                layer, used = [], set()
            used |= set(qs)
            layer.append((pyg_name,) + tuple(virt[q] for q in qs))
            for q in qs:
                state[q] = ACTIVE
        if layer:
            flush()

    n_virtual = len(virtual_to_physical)
    pyg_circuit = pygsti.circuits.Circuit(layers, line_labels=tuple(range(n_virtual)))

    def to_pauli(indices):
        pauli = stim.PauliString(n_virtual)
        for i in indices:
            virtual, inverted = measurements[i]
            pauli *= _z_on(virtual, n_virtual)
            if inverted:
                pauli *= -1
        return pauli

    detectors = [to_pauli(idx) for idx in detector_idx]
    observables = [to_pauli(observable_idx.get(j, [])) for j in range(circuit.num_observables)]
    return pyg_circuit, detectors, observables, virtual_to_physical


def _z_on(qubit, n_qubits):
    pauli = stim.PauliString(n_qubits)
    pauli[qubit] = 'Z'
    return pauli

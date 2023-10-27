""" Functions for interfacing pyGSTi with external devices, including IBM Q and Rigetti """
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

import numpy as _np

from . import ibmq_algiers      # New system
from . import ibmq_athens
from . import ibmq_auckland     # New system
from . import ibmq_belem
from . import ibmq_bogota
from . import ibmq_brisbane     # New system
from . import ibmq_burlington
from . import ibmq_cairo        # New system
from . import ibmq_cambridge
from . import ibmq_casablanca
from . import ibmq_essex
from . import ibmq_guadalupe
from . import ibmq_hanoi        # New system
from . import ibmq_kolkata      # New system
from . import ibmq_lagos         # New system
from . import ibmq_lima
from . import ibmq_london
from . import ibmq_manhattan
from . import ibmq_melbourne
from . import ibmq_montreal
from . import ibmq_mumbai       # New system
from . import ibmq_nairobi      # New system
from . import ibmq_nazca        # New system
from . import ibmq_ourense
from . import ibmq_perth        # New system
from . import ibmq_quito
from . import ibmq_rome
from . import ibmq_rueschlikon
from . import ibmq_santiago
from . import ibmq_sherbrooke   # New system
from . import ibmq_sydney
from . import ibmq_tenerife
from . import ibmq_toronto
from . import ibmq_vigo
from . import ibmq_yorktown
from . import rigetti_agave
from . import rigetti_aspen4
from . import rigetti_aspen6
from . import rigetti_aspen7
from pygsti.processors import QubitProcessorSpec as _QubitProcessorSpec
from pygsti.processors import CliffordCompilationRules as _CliffordCompilationRules
from pygsti.models import oplessmodel as _oplessmodel, modelconstruction as _mconst
from pygsti.modelmembers.povms import povm as _povm
from pygsti.tools import rbtools as _anl
from pygsti.tools.legacytools import deprecate as _deprecated_fn
from pygsti.baseobjs.qubitgraph import QubitGraph as _QubitGraph

@_deprecated_fn('basic_device_information')
def get_device_specs(devname):
    return basic_device_information(devname)


def basic_device_information(devname):
    return _get_dev_specs(devname)


def _get_dev_specs(devname):
    if devname == 'ibm_algiers' or devname == 'ibmq_algiers': dev = ibmq_algiers
    elif devname == 'ibmq_athens': dev = ibmq_athens
    elif devname == 'ibm_auckland' or devname == 'ibmq_auckland': dev = ibmq_auckland
    elif devname == 'ibmq_belem': dev = ibmq_belem
    elif devname == 'ibmq_bogota': dev = ibmq_bogota
    elif devname == 'ibm_brisbane' or devname == 'ibmq_brisbane': dev = ibmq_brisbane
    elif devname == 'ibmq_burlington': dev = ibmq_burlington
    elif devname == 'ibm_cairo' or devname == 'ibmq_cairo': dev = ibmq_cairo
    elif devname == 'ibmq_cambridge': dev = ibmq_cambridge
    elif devname == 'ibmq_casablanca': dev = ibmq_casablanca
    elif devname == 'ibmq_essex': dev = ibmq_essex
    elif devname == 'ibmq_guadalupe': dev = ibmq_guadalupe
    elif devname == 'ibm_hanoi' or devname == 'ibmq_hanoi': dev = ibmq_hanoi
    elif devname == 'ibm_kolkata' or devname == 'ibmq_kolkata': dev = ibmq_kolkata
    elif devname == 'ibm_lagos' or devname == 'ibmq_lagos': dev = ibmq_lagos
    elif devname == 'ibmq_lima': dev = ibmq_lima
    elif devname == 'ibmq_london': dev = ibmq_london
    elif devname == 'ibmq_manhattan': dev = ibmq_manhattan
    elif devname == 'ibmq_melbourne' or devname == 'ibmq_16_melbourne': dev = ibmq_melbourne
    elif devname == 'ibmq_montreal': dev = ibmq_montreal
    elif devname == 'ibm_mumbai' or devname == 'ibmq_mumbai': dev = ibmq_mumbai
    elif devname == 'ibm_nairobi' or devname == 'ibmq_nairobi': dev = ibmq_nairobi
    elif devname == 'ibm_nazco' or devname == 'ibmq_nazco': dev = ibmq_nazca
    elif devname == 'ibmq_ourense': dev = ibmq_ourense
    elif devname == 'ibm_perth' or devname == 'ibmq_perth': dev = ibmq_perth
    elif devname == 'ibmq_quito': dev = ibmq_quito
    elif devname == 'ibmq_rome': dev = ibmq_rome
    elif devname == 'ibmq_rueschlikon': dev = ibmq_rueschlikon
    elif devname == 'ibmq_santiago': dev = ibmq_santiago
    elif devname == 'ibm_sherbrooke' or devname == 'ibmq_sherbrooke': dev = ibmq_sherbrooke
    elif devname == 'ibmq_sydney': dev = ibmq_sydney
    elif devname == 'ibmq_tenerife': dev = ibmq_tenerife
    elif devname == 'ibmq_toronto': dev = ibmq_toronto
    elif devname == 'ibmq_vigo': dev = ibmq_vigo
    elif devname == 'ibmq_yorktown' or devname == 'ibmqx2': dev = ibmq_yorktown
    elif devname == 'rigetti_agave': dev = rigetti_agave
    elif devname == 'rigetti_aspen4': dev = rigetti_aspen4
    elif devname == 'rigetti_aspen6': dev = rigetti_aspen6
    elif devname == 'rigetti_aspen7': dev = rigetti_aspen7
    else:
        raise ValueError("This device name is not known!")

    return dev


def edgelist(device):

    specs = _get_dev_specs(device)

    return specs.edgelist


def create_clifford_processor_spec(device, one_qubit_gates, qubitsubset=None, removeedges=(),
                                   clifford_compilation_type='absolute', what_to_compile=('1Qcliffords',),
                                   verbosity=0):
    """
    TODO: docstring

    Parameters
    ----------
    device
    one_qubit_gates
    qubitsubset
    removeedges
    clifford_compilation_type
    what_to_compile
    verbosity

    Returns
    -------
    QubitProcessorSpec
    """
    native_pspec = create_processor_spec(device, one_qubit_gates, qubitsubset, removeedges)
    clifford_compilation = _CliffordCompilationRules.create_standard(
        native_pspec, clifford_compilation_type, what_to_compile, verbosity)
    clifford_pspec = clifford_compilation.apply_to_processorspec(native_pspec)
    return clifford_pspec


def create_processor_spec(device, one_qubit_gates, qubitsubset=None, removeedges=()):
    """
    todo

    clifford compilation type & what_to_compile = {'paulieq': ('1Qcliffords',),
                                           'absolute': ('paulis', '1Qcliffords')}
    """
    dev = _get_dev_specs(device)

    if qubitsubset is not None:
        qubits = qubitsubset
        assert(set(qubitsubset).issubset(set(dev.qubits)))
    else:
        qubits = dev.qubits.copy()

    total_qubits = len(qubits)
    two_qubit_gate = dev.two_qubit_gate
    gate_names = [two_qubit_gate] + one_qubit_gates

    edgelist = dev.edgelist.copy()

    if qubitsubset is not None:
        subset_edgelist = []
        for edge in edgelist:
            if edge[0] in qubits and edge[1] in qubits:
                subset_edgelist.append(edge)

        edgelist = subset_edgelist

    for edge in removeedges: del edgelist[edgelist.index(edge)]

    # Replaced availability with a QubitGraph due to a bug(?) in how an availability is propogated
    # into a QubitProcessorSpec's QubitGraph (whereas the `geometry` input is directly stored as 
    # the provided QubitGraph).
    #availability = {two_qubit_gate: edgelist}

    qubit_graph = _QubitGraph(qubits, initial_edges=edgelist)

    return _QubitProcessorSpec(total_qubits, gate_names, geometry=qubit_graph, qubit_labels=qubits)


def create_error_rates_model(caldata, device, one_qubit_gates, one_qubit_gates_to_native={}, calformat=None,
                             model_type='TwirledLayers', idle_name=None):
    """
    calformat: 'ibmq-v2018', 'ibmq-v2019', 'rigetti', 'native'.
    """

    specs = _get_dev_specs(device)
    two_qubit_gate = specs.two_qubit_gate
    if 'Gc0' in one_qubit_gates:
        assert('Gi' not in one_qubit_gates), "Cannot ascertain idle gate name!"
        idle_name = 'Gc0'
    elif 'Gi' in one_qubit_gates:
        assert('Gc0' not in one_qubit_gates), "Cannot ascertain idle gate name!"
        idle_name = 'Gi'
    else:
        if model_type == 'dict':
            pass
        else:
            raise ValueError("Must specify the idle gate!")

    assert(not ((calformat is None) and (device is None))), "Must specify `calformat` or `device`"
    if calformat is None:
        calformat = specs.spec_format

    def average_gate_infidelity_to_entanglement_infidelity(agi, numqubits):

        dep = _anl.r_to_p(agi, 2**numqubits, 'AGI')
        ent_inf = _anl.p_to_r(dep, 2**numqubits, 'EI')

        return ent_inf

    error_rates = {}
    error_rates['gates'] = {}
    error_rates['readout'] = {}

    if calformat == 'ibmq-v2018':

        assert(one_qubit_gates_to_native == {}), \
            "There is only a single one-qubit gate error rate for this calibration data format!"
        # This goes through the multi-qubit gates and records their error rates
        for dct in caldata['multiQubitGates']:

            # Converts to our gate name convention.
            gatename = (two_qubit_gate, 'Q' + str(dct['qubits'][0]), 'Q' + str(dct['qubits'][1]))
            # Assumes that the error rate is an average gate infidelity (as stated in qiskit docs).
            agi = dct['gateError']['value']
            # Maps the AGI to an entanglement infidelity.
            error_rates['gates'][gatename] = average_gate_infidelity_to_entanglement_infidelity(agi, 2)

        # This goes through the 1-qubit gates and readouts and stores their error rates.
        for dct in caldata['qubits']:

            q = dct['name']
            agi = dct['gateError']['value']
            error_rates['gates'][q] = average_gate_infidelity_to_entanglement_infidelity(agi, 1)

            # This assumes that this error rate is the rate of bit-flips.
            error_rates['readout'][q] = dct['readoutError']['value']

        # Because the one-qubit gates are all set to the same error rate, we have an alias dict that maps each one-qubit
        # gate on each qubit to that qubits label (the error rates key in error_rates['gates'])
        alias_dict = {}
        for q in specs.qubits:
            alias_dict.update({(oneQgate, q): q for oneQgate in one_qubit_gates})

    elif calformat == 'ibmq-v2019':

        # These'll be the keys in the error model, with the pyGSTi gate names aliased to these keys. If unspecified,
        # we set the error rate of a gate to the 'u3' gate error rate.
        oneQgatekeys = []
        for oneQgate in one_qubit_gates:
            # TIM UPDATED THIS BECAUSE THE ASSERT FAILS WITH THE LATEST IBM Q SPEC FORMAT. NOT SURE IF THIS TRY/EXCEPT
            # DID ANYTHING IMPORTANT.
            #try:
            nativekey = one_qubit_gates_to_native[oneQgate]
            #except:
            #    one_qubit_gates_to_native[oneQgate] = 'u3'
            #    nativekey = 'u3'
            #assert(nativekey in ('id', 'u1', 'u2', 'u3')
            #       ), "{} is not a gate specified in the IBM Q calibration data".format(nativekey)
            if nativekey not in oneQgatekeys:
                oneQgatekeys.append(nativekey)

        alias_dict = {}
        for q in specs.qubits:
            alias_dict.update({(oneQgate, q): (one_qubit_gates_to_native[oneQgate], q)
                               for oneQgate in one_qubit_gates})

        # Loop through all the gates, and record the error rates that we use in our error model.
        for gatecal in caldata['gates']:

            if gatecal['gate'] == 'cx':

                # The qubits the gate is on, in the IBM Q notation
                qubits = gatecal['qubits']
                # Converts to our gate name convention.
                gatename = (two_qubit_gate, 'Q' + str(qubits[0]), 'Q' + str(qubits[1]))
                # Assumes that the error rate is an average gate infidelity (as stated in qiskit docs).
                agi = gatecal['parameters'][0]['value']
                # Maps the AGI to an entanglement infidelity.
                error_rates['gates'][gatename] = average_gate_infidelity_to_entanglement_infidelity(agi, 2)

            if gatecal['gate'] in oneQgatekeys:

                # The qubits the gate is on, in the IBM Q notation
                qubits = gatecal['qubits']
                # Converts to pyGSTi-like gate name convention, but using the IBM Q name.
                gatename = (gatecal['gate'], 'Q' + str(qubits[0]))
                # Assumes that the error rate is an average gate infidelity (as stated in qiskit docs).
                agi = gatecal['parameters'][0]['value']
                # Maps the AGI to an entanglement infidelity.
                error_rates['gates'][gatename] = average_gate_infidelity_to_entanglement_infidelity(agi, 1)

        # Record the readout error rates. Because we don't do any rescaling, this assumes that this error
        # rate is the rate of bit-flips.
        for q, qcal in enumerate(caldata['qubits']):

            for qcaldatum in qcal:
                if qcaldatum['name'] == 'readout_error':
                    error_rates['readout']['Q' + str(q)] = qcaldatum['value']

    elif calformat == 'rigetti':

        # This goes through the multi-qubit gates and records their error rates
        for qs, gatedata in caldata['2Q'].items():

            # The qubits the qubit is on.
            qslist = qs.split('-')
            # Converts to our gate name convention. Do both orderings of the qubits as symmetric and we
            # are not necessarily consistent with Rigetti's ordering in the cal dict.
            gatename1 = (two_qubit_gate, 'Q' + qslist[0], 'Q' + qslist[1])
            gatename2 = (two_qubit_gate, 'Q' + qslist[1], 'Q' + qslist[0])

            # We use the controlled-Z fidelity if available, and the Bell state fidelity otherwise.
            # Here we are assuming that this is an average gate fidelity (as stated in the pyQuil docs)
            if gatedata['fCZ'] is not None:
                agi = 1 - gatedata['fCZ']
            else:
                agi = 1 - gatedata['fBellState']
            # We map the infidelity to 0 if it is less than 0 (sometimes this occurs with Rigetti
            # calibration data).
            agi = max([0, agi])
            # Maps the AGI to an entanglement infidelity.
            error_rates['gates'][gatename1] = average_gate_infidelity_to_entanglement_infidelity(agi, 2)
            error_rates['gates'][gatename2] = average_gate_infidelity_to_entanglement_infidelity(agi, 2)

        for q, qdata in caldata['1Q'].items():

            qlabel = 'Q' + q
            # We are assuming that this is an average gate fidelity (as stated in the pyQuil docs).
            agi = 1 - qdata['f1QRB']
            # We map the infidelity to 0 if it is less than 0 (sometimes this occurs with Rigetti
            # calibration data).
            agi = max([0, agi])
            # Maps the AGI to an entanglement infidelity. Use the qlabel, ..... TODO
            error_rates['gates'][qlabel] = average_gate_infidelity_to_entanglement_infidelity(agi, 1)
            # Record the readout error rates. Because we don't do any rescaling (except forcing to be
            # non-negative) this assumes that this error rate is the rate of bit-flips.
            error_rates['readout'][qlabel] = 1 - min([1, qdata['fRO']])

        # Because the one-qubit gates are all set to the same error rate, we have an alias dict that maps each one-qubit
        # gate on each qubit to that qubits label (the error rates key in error_rates['gates'])
        alias_dict = {}
        for q in specs.qubits:
            alias_dict.update({(oneQgate, q): q for oneQgate in one_qubit_gates})

    elif calformat == 'native':
        error_rates = caldata['error_rates'].copy()
        alias_dict = caldata['alias_dict'].copy()

    else:
        raise ValueError("Calibration data format not understood!")

    nQubits = len(specs.qubits)
    if model_type == 'dict':
        model = {'error_rates': error_rates, 'alias_dict': alias_dict}

    elif model_type == 'TwirledLayers':
        model = _oplessmodel.TwirledLayersModel(error_rates, nQubits, state_space_labels=specs.qubits,
                                                alias_dict=alias_dict, idle_name=idle_name)
    elif model_type == 'TwirledGates':
        model = _oplessmodel.TwirledGatesModel(error_rates, nQubits, state_space_labels=specs.qubits,
                                               alias_dict=alias_dict, idle_name=idle_name)
    elif model_type == 'AnyErrorCausesFailure':
        model = _oplessmodel.AnyErrorCausesFailureModel(error_rates, nQubits, state_space_labels=specs.qubits,
                                                        alias_dict=alias_dict, idle_name=idle_name)
    elif model_type == 'AnyErrorCausesRandomOutput':
        model = _oplessmodel.AnyErrorCausesRandomOutputModel(error_rates, nQubits, state_space_labels=specs.qubits,
                                                             alias_dict=alias_dict, idle_name=idle_name)
    else:
        raise ValueError("Model type not understood!")

    return model


def create_local_depolarizing_model(caldata, device, one_qubit_gates, one_qubit_gates_to_native={},
                                    calformat=None, qubits=None):
    """
    todo

    Note: this model is *** NOT *** suitable for optimization: it is not aware that it is a local depolarization
    with non-independent error rates model.
    """

    def _get_local_depolarization_channel(rate, num_qubits):

        if num_qubits == 1:

            channel = _np.identity(4, float)
            channel[1, 1] = _anl.r_to_p(rate, 2, 'EI')
            channel[2, 2] = _anl.r_to_p(rate, 2, 'EI')
            channel[3, 3] = _anl.r_to_p(rate, 2, 'EI')

            return channel

        if num_qubits == 2:

            perQrate = 1 - _np.sqrt(1 - rate)
            channel = _np.identity(4, float)
            channel[1, 1] = _anl.r_to_p(perQrate, 2, 'EI')
            channel[2, 2] = _anl.r_to_p(perQrate, 2, 'EI')
            channel[3, 3] = _anl.r_to_p(perQrate, 2, 'EI')

            return _np.kron(channel, channel)

    def _get_local_povm(rate):

        # Increase the error rate of X,Y,Z, as rate correpsonds to bit-flip rate.
        deprate = 3 * rate / 2
        p = _anl.r_to_p(deprate, 2, 'EI')
        povm = _povm.UnconstrainedPOVM({'0': [1 / _np.sqrt(2), 0, 0, p / _np.sqrt(2)],
                                        '1': [1 / _np.sqrt(2), 0, 0, -p / _np.sqrt(2)]
                                        })
        return povm

    tempdict = create_error_rates_model(caldata, device, one_qubit_gates,
                                        one_qubit_gates_to_native=one_qubit_gates_to_native,
                                        calformat=calformat, model_type='dict')

    error_rates = tempdict['error_rates']
    alias_dict = tempdict['alias_dict']
    devspecs = basic_device_information(device)

    if qubits is None:
        qubits = devspecs.qubits
        edgelist = devspecs.edgelist
    else:
        edgelist = [edge for edge in devspecs.edgelist if set(edge).issubset(set(qubits))]

    #print(qubits)
    #print(edgelist)

    model = _mconst.create_localnoise_model(n_qubits=len(qubits),
                                            qubit_labels=qubits,
                                            gate_names=[devspecs.two_qubit_gate] + one_qubit_gates,
                                            availability={devspecs.two_qubit_gate: edgelist},
                                            parameterization='full', independent_gates=True)

    for lbl in model.operation_blks['gates'].keys():

        gatestr = str(lbl)

        if len(lbl.qubits) == 1:
            errormap = _get_local_depolarization_channel(error_rates['gates'][alias_dict.get(gatestr, gatestr)], 1)
            model.operation_blks['gates'][lbl] = _np.dot(errormap, model.operation_blks['gates'][lbl])

        if len(lbl.qubits) == 2:
            errormap = _get_local_depolarization_channel(error_rates['gates'][alias_dict.get(gatestr, gatestr)], 2)
            model.operation_blks['gates'][lbl] = _np.dot(errormap, model.operation_blks['gates'][lbl])

    povms = [_get_local_povm(error_rates['readout'][q]) for q in model.qubit_labels]
    model.povm_blks['layers']['Mdefault'] = _povm.TensorProdPOVM(povms)

    return model

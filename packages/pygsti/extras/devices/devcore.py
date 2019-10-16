""" ... """
from __future__ import division, print_function, absolute_import, unicode_literals
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

from ..rb import analysis as _anl
from ..rb import errorratesmodel as _erm
from ...objects import processorspec as _pspec
from ...objects import povm as _povm
from ...construction import modelconstruction as _mconst

from . import ibmq_melbourne
from . import ibmq_ourense
from . import ibmq_rueschlikon
from . import ibmq_tenerife
from . import ibmq_vigo
from . import ibmq_yorktown
from . import rigetti_agave
from . import rigetti_aspen4
from . import rigetti_aspen6

import numpy as _np


def _get_dev_specs(devname):

    if devname == 'ibmq_melbourne': dev = ibmq_melbourne
    elif devname == 'ibmq_ourense': dev = ibmq_ourense
    elif devname == 'ibmq_rueschlikon': dev = ibmq_rueschlikon
    elif devname == 'ibmq_tenerife': dev = ibmq_tenerife
    elif devname == 'ibmq_vigo': dev = ibmq_vigo
    elif devname == 'ibmq_yorktown': dev = ibmq_yorktown
    elif devname == 'rigetti_agave': dev = rigetti_agave
    elif devname == 'rigetti_aspen4': dev = rigetti_aspen4
    elif devname == 'rigetti_aspen6': dev = rigetti_aspen6
    else:
        raise ValueError("This device name is not known!")

    return dev


def create_processor_spec(device, oneQgates, qubitsubset=None, removeedges=[],
                          construct_clifford_compilations={'paulieq': ('1Qcliffords',),
                                                           'absolute': ('paulis', '1Qcliffords')},
                          verbosity=0):
    """
    todo

    """
    dev = _get_dev_specs(device)

    if qubitsubset is not None:
        qubits = qubitsubset
        assert(set(qubitsubset).issubset(set(dev.qubits)))
    else:
        qubits = dev.qubits.copy()

    total_qubits = len(qubits)
    twoQgate = dev.twoQgate
    gate_names = [twoQgate] + oneQgates

    edgelist = dev.edgelist.copy()

    if qubitsubset is not None:
        subset_edgelist = []
        for edge in edgelist:
            if edge[0] in qubits and edge[1] in qubits:
                subset_edgelist.append(edge)

        edgelist = subset_edgelist

    for edge in removeedges: del edgelist[edgelist.index(edge)]

    availability = {twoQgate: edgelist}
    pspec = _pspec.ProcessorSpec(total_qubits, gate_names, availability=availability,
                                 construct_clifford_compilations=construct_clifford_compilations,
                                 verbosity=verbosity, qubit_labels=qubits)

    return pspec


def create_error_rates_model(caldata, calformat=None, device=None, model_type='GlobalDep'):
    """
    calformat: 'ibmq-v2018', 'ibmq-v2019', 'rigetti', 'native'.
    """
    assert(not ((calformat is None) and (device is None))), "Must specify `calformat` or `device`"
    if calformat is None:
        dev = _get_dev_specs(device)
        calformat = dev.spec_format

    error_rates = {}
    error_rates['gates'] = {}
    error_rates['readout'] = {}

    if calformat == 'ibmq-v2018':

        # This goes through the multi-qubit gates.
        for dct in caldata['multiQubitGates']:

            q1 = 'Q' + str(dct['qubits'][0])
            q2 = 'Q' + str(dct['qubits'][1])

            dep = _anl.r_to_p(dct['gateError']['value'], 4, 'AGI')
            errate = _anl.p_to_r(dep, 4, 'EI')

            error_rates['gates'][frozenset((q1, q2))] = errate

        # This stores the error rates of the 1-qubit gates and measurements.
        for dct in caldata['qubits']:

            q = dct['name']

            dep = _anl.r_to_p(dct['gateError']['value'], 2, 'AGI')
            errate = _anl.p_to_r(dep, 2, 'EI')

            error_rates['gates'][q] = errate
            error_rates['readout'][q] = dct['readoutError']['value']

    elif calformat == 'ibmq-v2019':

        for gatecal in caldata['gates']:

            if gatecal['gate'] == 'cx':

                qubits = gatecal['qubits']

                dep = _anl.r_to_p(gatecal['parameters'][0]['value'], 4, 'AGI')
                errate = _anl.p_to_r(dep, 4, 'EI')

                error_rates['gates'][frozenset(('Q' + str(qubits[0]), 'Q' + str(qubits[1])))] = errate

            if gatecal['gate'] == 'u3':

                qubits = gatecal['qubits']
                dep = _anl.r_to_p(gatecal['parameters'][0]['value'], 2, 'AGI')
                errate = _anl.p_to_r(dep, 2, 'EI')
                error_rates['gates']['Q' + str(qubits[0])] = errate

        for q, qcal in enumerate(caldata['qubits']):

            for qcaldatum in qcal:
                if qcaldatum['name'] == 'readout_error':
                    error_rates['readout']['Q' + str(q)] = qcaldatum['value']

    elif calformat == 'rigetti':

        # Fidelities reported by Rigetti can be > 1. Any such fidelity is mapped to 1.
        for qs in caldata['2Q'].keys():

            qslist = qs.split('-')

            # We use the CZ fidelity if available, and the Bell state fidelity otherwise.
            if caldata['2Q'][qs]['fCZ'] is not None: f = caldata['2Q'][qs]['fCZ']
            else: f = caldata['2Q'][qs]['fBellState']

            dep = _anl.r_to_p(1 - min([1, f]), 4, 'AGI')
            errate = _anl.p_to_r(dep, 4, 'EI')
            error_rates['gates'][frozenset(('Q' + qslist[0], 'Q' + qslist[1]))] = errate

        for q in caldata['1Q'].keys():

            dep = _anl.r_to_p(1 - min([1, caldata['1Q'][q]['f1QRB']]), 2, 'AGI')
            error_rate = _anl.p_to_r(dep, 2, 'EI')
            error_rates['gates']['Q' + q] = error_rate
            error_rates['readout']['Q' + q] = 1 - min([1, caldata['1Q'][q]['fRO']])

    elif calformat == 'native':

        error_rates = caldata

    else:
        raise ValueError("Calibration data format not understood!")

    model = _erm.ErrorRatesModel(error_rates, model_type)

    return model


def create_local_depolarizing_model(pspec, caldata, calformat):
    """
    todo
    """

    def get_local_depolarization_channel(rate, numQs):

        if numQs == 1:

            channel = _np.identity(4, float)
            channel[1, 1] = _anl.r_to_p(rate, 2, 'EI')
            channel[2, 2] = _anl.r_to_p(rate, 2, 'EI')
            channel[3, 3] = _anl.r_to_p(rate, 2, 'EI')

            return channel

        if numQs == 2:

            perQrate = 1 - _np.sqrt(1 - rate)
            channel = _np.identity(4, float)
            channel[1, 1] = _anl.r_to_p(perQrate, 2, 'EI')
            channel[2, 2] = _anl.r_to_p(perQrate, 2, 'EI')
            channel[3, 3] = _anl.r_to_p(perQrate, 2, 'EI')

            return _np.kron(channel, channel)

    def get_local_povm(rate):

        # Increase the error rate of X,Y,Z, as rate correpsonds to bit-flip rate.
        deprate = 3 * rate / 2
        p = _anl.r_to_p(deprate, 2, 'EI')
        povm = _povm.UnconstrainedPOVM({'0': [1 / _np.sqrt(2), 0, 0, p / _np.sqrt(2)],
                                        '1': [1 / _np.sqrt(2), 0, 0, -p / _np.sqrt(2)]
                                        })
        return povm

    erm = create_error_rates_model(caldata, calformat)
    model = _mconst.build_localnoise_model(nQubits=pspec.number_of_qubits,
                                           qubit_labels=pspec.qubit_labels,
                                           gate_names=pspec.root_gate_names,
                                           availability=pspec.availability,
                                           parameterization='full', independent_gates=True)

    for lbl in model.operation_blks['gates'].keys():

        if len(lbl.qubits) == 1:
            errormap = get_local_depolarization_channel(erm.error_rates['gates'][lbl.qubits[0]], 1)
            model.operation_blks['gates'][lbl] = _np.dot(errormap, model.operation_blks['gates'][lbl])

        if len(lbl.qubits) == 2:
            errormap = get_local_depolarization_channel(erm.error_rates['gates'][frozenset(lbl.qubits)], 2)
            model.operation_blks['gates'][lbl] = _np.dot(errormap, model.operation_blks['gates'][lbl])

    povms = [get_local_povm(erm.error_rates['readout'][q]) for q in model.qubit_labels]
    model.povm_blks['layers']['Mdefault'] = _povm.TensorProdPOVM(povms)

    return model

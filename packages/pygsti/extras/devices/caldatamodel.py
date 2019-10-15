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

import numpy as _np


def create_error_rates_model(caldata, calformat):
    """
    calformat: 'ibmq-v2018', 'ibmq-v2019', 'rigetti', 'native'.
    """
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

    model = _erm.ErrorRatesModel(error_rates, "Local-OneTwoReadout-NonUniform")

    return model
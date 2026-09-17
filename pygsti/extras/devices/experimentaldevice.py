""" Functions for interfacing pyGSTi with external devices, including IBM Q and Rigetti """
#***************************************************************************************************
# Copyright 2015, 2019, 2025 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

import numpy as _np
from importlib import import_module

from pygsti.processors import QubitProcessorSpec as _QubitProcessorSpec
from pygsti.processors import CliffordCompilationRules as _CliffordCompilationRules
from pygsti.models import oplessmodel as _oplessmodel, modelconstruction as _mconst
from pygsti.modelmembers.povms import povm as _povm
from pygsti.tools import rbtools as _anl
from pygsti.tools.legacytools import deprecate as _deprecated_fn
from pygsti.baseobjs.qubitgraph import QubitGraph as _QubitGraph

class ExperimentalDevice(object):
    """Specification of an experimental device.
    """
    def __init__(self, qubits, graph, gate_mapping=None):
        """Initialize an IBMQ device from qubits and connectivity info.

        Parameters
        ----------
        qubits: list
            Qubit labels

        graph: QubitGraph
            QubitGraph depicting device connectivity.

        gate_mapping: dict, optional
            Mapping between pyGSTi gate names (keys) and IBM native gates (values).
            If None, simply use {'Gcnot': 'cx'} to recover legacy behavior.
        """
        self.qubits = qubits
        self.graph = graph
        self.gate_mapping = gate_mapping if gate_mapping is not None else {'Gcnot': 'cx'}
    
    @classmethod
    def from_qiskit_backend(cls, backend, gate_mapping=None):
        """Construct a ExperimentalDevice from Qiskit provider backend information.

        Provider backends can be obtained via:
            IBMQ.load_account()
            provider = IBMQ.get_provider() # with potential optional kwargs
            backend = provider.get_backend(<device name>)

        Parameters
        ----------
        backend: IBMQBackend
            Backend obtained from IBMQ
        
        gate_mapping: dict, optional
            Mapping between pyGSTi gate names (keys) and IBM native gates (values).
            If None, simply use {'Gcnot': 'cx'} to recover legacy behavior.
        
        Returns
        -------
        Initialized ExperimentalDevice
        """
        # Get qubits
        num_qubits = backend.num_qubits
        qubits = [f'Q{i}' for i in range(num_qubits)]

        # Get qubit connectivity
        edges = [[qubits[edge[0]], qubits[edge[1]]] for edge in backend.coupling_map]
        graph = _QubitGraph(qubits, initial_edges=edges)

        return cls(qubits, graph, gate_mapping)
    
    @classmethod
    def from_legacy_device(cls, devname):
        """Create a ExperimentalDevice from a legacy pyGSTi pygsti.extras.devices module.

        Parameters
        ----------
        devname: str
            Name of the pygsti.extras.devices module to use
        
        Returns
        -------
        Initialized ExperimentalDevice
        """
        try:
            dev = import_module(f'pygsti.extras.devices.{devname}')
        except ImportError:
            raise RuntimeError(f"Failed to import device {devname}. Use an existing device from pygsti.extras.devices" \
                               + " or use an up-to-date IBMQ backend object instead.")

        return cls(dev.qubits, _QubitGraph(dev.qubits, initial_edges=dev.edgelist))

    def create_processor_spec(self, gate_names=None, qubit_subset=None, subset_only=False, remove_edges=None):
        """Create a QubitProcessorSpec from user-specified gates and device connectivity.

        Parameters
        ----------
        gate_names: list of str
            List of one-qubit and two-qubit gate names. If None, use the keys of self.gate_mapping.
        
        qubit_subset: list
            A subset of qubits to include in the processor spec. If None, use self.qubits.
        
        subset_only: bool
            Whether or not to include all the device qubits in the processor spec (False, default)
            or just qubit_subset (True).
        
        remove_edges: list
            A list of edges to drop from the connectivity graph.
        
        Returns
        -------
        The created QubitProcessorSpec
        """
        if gate_names is None:
            gate_names = list(self.gate_mapping.keys())
        if qubit_subset is None:
            qubit_subset = self.qubits
        assert set(qubit_subset).issubset(set(self.qubits)), "Qubit subset must actually be a subset"
        if remove_edges is None:
            remove_edges = []

        # Get subgraph
        graph = self.graph.subgraph(qubit_subset)
        for edge in remove_edges:
            graph.remove_edge(edge)

        # Decide whether to include all qubits or not
        qubits = qubit_subset if subset_only else self.qubits

        return _QubitProcessorSpec(len(qubits), gate_names, geometry=graph, qubit_labels=qubits)

    def create_error_rates_model(self, caldata=None, calformat='ibmq-v2019',
                                 model_type='TwirledLayers', idle_name=None):
        """Create an error rates model (OplessModel) from calibration data.

        Parameters
        ----------
        caldata: dict
            Calibration data. Currently, this can be retrieved via
            `backend.properties().to_dict()`.

        calformat: One of ['ibmq-v2018', 'ibmq-v2019', 'rigetti', 'native']
            Calibration data format, defaults to ibmq-v2019. TODO: It seems this has
            been changed, what version are we actually on?

        model_type: One of ['TwirledLayers', 'TwirledGates', 'AnyErrorCausesFailure', 'AnyErrorCausesRandomOutput']
            Type of OplessModel to create
        
        idle_name: str
            Name for the idle gate
        
        Returns
        -------
        OplessModel
        """

        def average_gate_infidelity_to_entanglement_infidelity(agi, numqubits):

            dep = _anl.r_to_p(agi, 2**numqubits, 'AGI')
            ent_inf = _anl.p_to_r(dep, 2**numqubits, 'EI')

            return ent_inf

        error_rates = {}
        error_rates['gates'] = {}
        error_rates['readout'] = {}

        one_qubit_gates = [v for k,v in self.gate_mapping.items() if k != 'cx']
        two_qubit_gate = self.gate_mapping['cx']

        if calformat == 'ibmq-v2018':

            assert(len(one_qubit_gates) == 1), \
                "There is only a single one-qubit gate error rate for this calibration data format!"
            # This goes through the multi-qubit gates and records their error rates
            for dct in caldata['multiQubitGates']:

                # Converts to our gate name convention.
                gatename = (self.gate_mapping['cx'], 'Q' + str(dct['qubits'][0]), 'Q' + str(dct['qubits'][1]))
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
            for q in self.qubits:
                alias_dict.update({(oneQgate, q): q for oneQgate in one_qubit_gates})

        elif calformat == 'ibmq-v2019':

            # These'll be the keys in the error model, with the pyGSTi gate names aliased to these keys. If unspecified,
            # we set the error rate of a gate to the 'u3' gate error rate.
            oneQgatekeys = []
            for oneQgate in one_qubit_gates:
                # TIM UPDATED THIS BECAUSE THE ASSERT FAILS WITH THE LATEST IBM Q SPEC FORMAT. NOT SURE IF THIS TRY/EXCEPT
                # DID ANYTHING IMPORTANT.
                #try:
                nativekey = self.gate_mapping[oneQgate]
                #except:
                #    one_qubit_gates_to_native[oneQgate] = 'u3'
                #    nativekey = 'u3'
                #assert(nativekey in ('id', 'u1', 'u2', 'u3')
                #       ), "{} is not a gate specified in the IBM Q calibration data".format(nativekey)
                if nativekey not in oneQgatekeys:
                    oneQgatekeys.append(nativekey)

            alias_dict = {}
            for q in self.qubits:
                alias_dict.update({(oneQgate, q): (self.gate_mapping[oneQgate], q)
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
            for q in self.qubits:
                alias_dict.update({(oneQgate, q): q for oneQgate in one_qubit_gates})

        elif calformat == 'native':
            error_rates = caldata['error_rates'].copy()
            alias_dict = caldata['alias_dict'].copy()

        else:
            raise ValueError("Calibration data format not understood!")

        nQubits = len(self.qubits)
        if model_type == 'dict':
            model = {'error_rates': error_rates, 'alias_dict': alias_dict}

        elif model_type == 'TwirledLayers':
            model = _oplessmodel.TwirledLayersModel(error_rates, nQubits, state_space_labels=self.qubits,
                                                    alias_dict=alias_dict, idle_name=idle_name)
        elif model_type == 'TwirledGates':
            model = _oplessmodel.TwirledGatesModel(error_rates, nQubits, state_space_labels=self.qubits,
                                                alias_dict=alias_dict, idle_name=idle_name)
        elif model_type == 'AnyErrorCausesFailure':
            model = _oplessmodel.AnyErrorCausesFailureModel(error_rates, nQubits, state_space_labels=self.qubits,
                                                            alias_dict=alias_dict, idle_name=idle_name)
        elif model_type == 'AnyErrorCausesRandomOutput':
            model = _oplessmodel.AnyErrorCausesRandomOutputModel(error_rates, nQubits, state_space_labels=self.qubits,
                                                                alias_dict=alias_dict, idle_name=idle_name)
        else:
            raise ValueError("Model type not understood!")

        return model

    def create_local_depolarizing_model(self, caldata=None, calformat='ibmq-v2019', qubits=None):
        """
        Create a LocalNoiseModel with depolarizing noise based on calibration data.

        Note: this model is *** NOT *** suitable for optimization: it is not aware that it is a local depolarization
        with non-independent error rates model.

        Parameters
        ----------
        caldata: dict
            Calibration data. Currently, this can be retrieved via
            `backend.properties().to_dict()`.

        calformat: One of ['ibmq-v2018', 'ibmq-v2019', 'rigetti', 'native']
            Calibration data format, defaults to ibmq-v2019. TODO: It seems this has
            been changed, what version are we actually on?

        qubits: list
            Qubit labels to include in the model
        
        Returns
        -------
        OplessModel
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

        tempdict = self.create_error_rates_model(caldata, calformat=calformat, model_type='dict')

        error_rates = tempdict['error_rates']
        alias_dict = tempdict['alias_dict']

        if qubits is None:
            qubits = self.qubits

        pspec = self.create_processor_spec(qubit_subset=qubits)
        model = _mconst.create_crosstalk_free_model(pspec, parameterization='full', independent_gates=True)

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

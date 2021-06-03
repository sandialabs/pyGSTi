""" Functions for sending experiments to IBMQ devices and converting the results to pyGSTi objects """
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

from ... import objects as _obj
from ...protocols import ProtocolData as _ProtocolData
import numpy as _np
import time as _time

try: import qiskit as _qiskit
except: _qiskit = None

# Most recent version of QisKit that this has been tested on:
# qiskit.__qiskit_version_ =  {
#   'qiskit-terra': '0.16.4',
#   'qiskit-aer': '0.7.5',
#   'qiskit-ignis': '0.5.2',
#   'qiskit-ibmq-provider': '0.11.1',
#   'qiskit-aqua': '0.8.2',
#   'qiskit': '0.23.6'
#}


def reverse_dict_key_bits(counts_dict):
    new_dict = {}
    for key in counts_dict.keys():
        new_dict[key[::-1]] = counts_dict[key]
    return new_dict


# NOTE: This is probably duplicative of some other code in pyGSTi
def partial_trace(ordered_target_indices, input_dict):
    output_dict = {}
    for bitstring in input_dict.keys():
        new_string = ''
        for index in ordered_target_indices:
            new_string += bitstring[index]
        try:
            output_dict[new_string] += input_dict[bitstring]
        except:
            output_dict[new_string] = input_dict[bitstring]
    return output_dict


def q_list_to_ordered_target_indices(q_list, num_qubits):
    if q_list is None:
        return list(range(num_qubits))
    else:
        output = []
        for q in q_list:
            assert q[0] == 'Q'
            output.append(int(q[1:]))
        return output


class IBMQExperiment(dict):

    def __init__(self, edesign, pspec, remove_duplicates=True, randomized_order=True, circuits_per_batch=900,
                 num_shots=1024):
        """

        The ProcessorSpecs qubit ordering *must* correspond to that of the IBM device. I.e., pspecs qubits
        should be labelled Q0 through Qn-1 and the labelling of the qubits should agree with IBM's labelling.

        """

        self['edesign'] = edesign
        self['pspec'] = pspec
        self['remove_duplicates'] = remove_duplicates
        self['randomized_order'] = randomized_order
        self['circuits_per_batch'] = circuits_per_batch
        self['num_shots'] = num_shots

        circuits = edesign.all_circuits_needing_data.copy()
        if randomized_order:
            if remove_duplicates:
                circuits = list(set(circuits))
            _np.random.shuffle(circuits)
        else:
            assert(not remove_duplicates), "Can only remove duplicates if randomizing order!"

        num_batches = int(_np.ceil(len(circuits) / circuits_per_batch))

        self['pygsti_circuits'] = [[] for i in range(num_batches)]
        self['pygsti_openqasm_circuits'] = [[] for i in range(num_batches)]
        self['qiskit_QuantumCircuits'] = [[] for i in range(num_batches)]
        self['qiskit_QuantumCircuits_as_openqasm'] = [[] for i in range(num_batches)]
        self['submit_time_calibration_data'] = []
        self['qobj'] = []

        batch_idx = 0
        for circ_idx, circ in enumerate(circuits):
            self['pygsti_circuits'][batch_idx].append(circ)
            if len(self['pygsti_circuits'][batch_idx]) == circuits_per_batch:
                batch_idx += 1

        #create Qiskit quantum circuit for each circuit defined in experiment list
        total_num = 0

        #start = _time.time()
        for batch_idx, circuit_batch in enumerate(self['pygsti_circuits']):
            print("Constructing job for circuit batch {} of {}".format(batch_idx + 1, num_batches))
            #openqasm_circuit_ids = []
            for circ_idx, circ in enumerate(circuit_batch):
                pygsti_openqasm_circ = circ.convert_to_openqasm(num_qubits =pspec.number_of_qubits, standard_gates_version='x-sx-rz')
                qiskit_qc = _qiskit.QuantumCircuit.from_qasm_str(pygsti_openqasm_circ)

                self['pygsti_openqasm_circuits'][batch_idx].append(pygsti_openqasm_circ)
                self['qiskit_QuantumCircuits'][batch_idx].append(qiskit_qc)
                self['qiskit_QuantumCircuits_as_openqasm'][batch_idx].append(qiskit_qc.qasm())

                #print(batch_idx, circ_idx, len(submitted_openqasm_circuits), total_num)
                total_num += 1

            self['qobj'].append(_qiskit.compiler.assemble(self['qiskit_QuantumCircuits'][batch_idx], shots=num_shots))

    def submit(self, ibmq_backend, wait_time=1, wait_steps=10):

        #start = _time.time()
        total_waits = 0
        self['qjob'] = []
        self['job_ids'] = []

        for batch_idx, qobj in enumerate(self['qobj']):

            print("Submitting batch {}".format(batch_idx))
            submit_status = False
            batch_waits = 0
            while not submit_status:
                try:
                    backend_properties = ibmq_backend.properties()
                    self['submit_time_calibration_data'].append(backend_properties.to_dict())
                    self['qjob'].append(ibmq_backend.run(qobj))
                    status = self['qjob'][-1].status()
                    initializing = True
                    initializing_steps = 0
                    while initializing:
                        if status.name == 'INITIALIZING' or status.name == 'VALIDATING':
                            #print(status)
                            status = self['qjob'][-1].status()
                            print('  - {} (query {})'.format(status, initializing_steps))
                            _time.sleep(1)
                            initializing_steps += 1
                        else:
                            initializing = False
                    #print("   -Done intializing.  Job status is {}".format(status.name))
                    #print(status)
                    try:
                        job_id = self['qjob'][-1].job_id()
                        print('  - Job ID is {}'.format(job_id))
                        self['job_ids'].append(job_id)
                    except:
                        print('  - Failed to get job_id.')
                        self['job_ids'].append(None)
                    try:
                        print('  - Queue position is {}'.format(self['qjob'][-1].queue_position()))
                    except:
                        print('  - Failed to get queue position'.format(batch_idx))
                    submit_status = True
                except Exception as ex:
                    template = "An exception of type {0} occurred. Arguments:\n{1!r}"
                    message = template.format(type(ex).__name__, ex.args)
                    print(message)
                    try:
                        print('Machine status is {}.'.format(ibmq_backend.status().status_msg))
                    except Exception as ex1:
                        print('Failed to get machine status!')
                        template = "An exception of type {0} occurred. Arguments:\n{1!r}"
                        message = template.format(type(ex).__name__, ex1.args)
                        print(message)
                    total_waits += 1
                    batch_waits += 1
                    print("This batch has failed {0} times and there have been {1} total failures".format(batch_waits, total_waits))
                    print('Waiting ', end='')
                    for step in range(wait_steps):
                        print('{} '.format(step), end='')
                        _time.sleep(wait_time)
                    print()

    def monitor(self):

        for counter, qjob in enumerate(self['qjob']):
            status = qjob.status()
            print("Batch {}: {}".format(counter, status))
            if status.name == 'QUEUED:':
                print('  - Queue position is {}'.format(qjob.queue_position()))

    def get_results(self):

        self['batch_result_object'] = []
        #get results from backend jobs and add to dict
        ds = _obj.DataSet()
        for exp_idx, qjob in enumerate(self['qjob']):
            print("Querying IBMQ for results objects for batch {}...".format(exp_idx))
            batch_result = qjob.result()
            self['batch_result_object'].append(batch_result)
            #exp_dict['batch_data'] = []
            for i, circ in enumerate(self['pygsti_circuits'][exp_idx]):
                ordered_target_indices = [self['pspec'].qubit_labels.index(q) for q in circ.line_labels]
                counts_data = partial_trace(ordered_target_indices, reverse_dict_key_bits(batch_result.get_counts(i)))
                #exp_dict['batch_data'].append(counts_data)
                ds.add_count_dict(circ, counts_data)

        self['data'] = _ProtocolData(self['edesign'], ds)

        #return ds #_obj.ProtocolData()

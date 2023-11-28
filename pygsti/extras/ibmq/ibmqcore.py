""" Functions for sending experiments to IBMQ devices and converting the results to pyGSTi objects """
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

import numpy as _np
import pathlib as _pathlib
import time as _time
import warnings as _warnings

try: import qiskit as _qiskit
except: _qiskit = None

from ... import data as _data, io as _io
from ...protocols import ProtocolData as _ProtocolData, HasProcessorSpec as _HasPSpec
from ...protocols.protocol import _TreeNode

# Most recent version of QisKit that this has been tested on:
#qiskit.__qiskit_version__ = {
#    'qiskit-terra': '0.25.3',
#    'qiskit': '0.44.3',
#    'qiskit-aer': None,
#    'qiskit-ignis': None,
#    'qiskit-ibmq-provider': '0.20.2',
#    'qiskit-nature': None,
#    'qiskit-finance': None,
#    'qiskit-optimization': None,
#    'qiskit-machine-learning': None
#}
#qiskit_ibm_provider.__version__ = '0.7.2'


class IBMQExperiment(_TreeNode, _HasPSpec):

    def __init__(self, edesign, pspec, remove_duplicates=True, randomized_order=True, circuits_per_batch=75,
                 num_shots=1024, seed=None):
        """
        A object that converts pyGSTi ExperimentDesigns into jobs to be submitted to IBM Q, submits these
        jobs to IBM Q and receives the results.

        Parameters
        ----------
        edesign: ExperimentDesign
            The ExperimentDesign to be run on IBM Q. This can be a combined experiment design (e.g., a GST
            design combined with an RB design).

        pspec: QubitProcessorSpec
            A QubitProcessorSpec that represents the IBM Q device being used. This can be created using the
            extras.devices.create_processor_spec(). The ProcessorSpecs qubit ordering *must* correspond
            to that of the IBM device (which will be the case if you create it using that function).
            I.e., pspecs qubits should be labelled Q0 through Qn-1 and the labelling of the qubits
            should agree with IBM's labelling.

        remove_duplicates: bool, optional
            If true, each distinct circuit in `edesign` is run only once. If false, if a circuit is
            repeated multiple times in `edesign` it is run multiple times.

        randomized_order: bool, optional
            Whether or not to randomize the order of the circuits in `edesign` before turning them
            into jobs to be submitted to IBM Q.

        circuits_per_batch: int, optional
            The circuits in edesign are divded into batches, each containing at most this many
            circuits. The default value of 75 is (or was) the maximal value allowed on the public
            IBM Q devices.

        num_shots: int, optional
            The number of samples from / repeats of each circuit.
        
        seed: int, optional
            Seed for RNG during order randomization of circuits.

        checkpoint_dirname: str, optional
            Name of checkpoint directory. If None, no checkpointing is used.

        Returns
        -------
        IBMQExperiment
            An object containing jobs to be submitted to IBM Q, which can then be submitted
            using the methods .submit() and whose results can be grabbed from IBM Q using
            the method .retrieve_results(). This object has dictionary-like access for all of
            the objects it contains (e.g., ['qobj'] is a list of the objects to be submitted to
            IBM Q).

        """
        _TreeNode.__init__(self, edesign._dirs)
        _HasPSpec.__init__(self, pspec)

        self.edesign = edesign
        self.processor_spec = pspec # Must be called processor_spec for _HasPSpec class
        self.remove_duplicates = remove_duplicates
        self.randomized_order = randomized_order
        self.circuits_per_batch = circuits_per_batch
        self.num_shots = num_shots
        self.seed = seed
        # Populated with transpiling to IBMQ with .transpile()
        self.pygsti_circuit_batches = []
        self.qasm_circuit_batches = []
        self.qiskit_circuit_batches = []
        # Populated when submitting to IBM Q with .submit()
        self.qjobs = []
        self.job_ids = []
        self.submit_time_calibration_data = []
        # Populated when grabbing results from IBM Q with .retrieve_results()
        self.batch_result_objects = []
        self.data = None

        # If not in this list, will be automatically dumped to meta.json
        # 'none' means it will not be read in, 'reset' means it will come back in as None
        # Several of these could be stored in the meta.json but are kept external for easy chkpts
        self.auxfile_types['edesign'] = 'none'
        self.auxfile_types['data'] = 'reset'
        # self.processor_spec is handled by _HasPSpec base class
        self.auxfile_types['pygsti_circuit_batches'] = 'list:text-circuit-list'
        self.auxfile_types['qasm_circuit_batches'] = 'list:json'
        self.auxfile_types['qiskit_circuit_batches'] = 'none'
        self.auxfile_types['qjobs'] = 'none'
        self.auxfile_types['job_ids'] = 'json'
        self.auxfile_types['submit_time_calibration_data'] = 'list:json'
        self.auxfile_types['batch_result_objects'] = 'none'

    def transpile(self, dirname=None):
        """Transpile pyGSTi circuits into Qiskit circuits for submission to IBMQ.

        Parameters
        ----------
        dirname: str, optional
            If provided, the root directory (i.e. same as passed to write()
            and from_dir()) to use for checkpointing
        """
        num_batches = int(_np.ceil(len(circuits) / self.circuits_per_batch))
        
        if self.pygsti_circuit_batches is None:
            self.pygsti_circuit_batches = []
            rand_state = _np.random.RandomState(self.seed)

            circuits = self.edesign.all_circuits_needing_data.copy()
            if self.randomized_order:
                if self.remove_duplicates:
                    circuits = list(set(circuits))
                rand_state.shuffle(circuits)
            else:
                assert(not self.remove_duplicates), "Can only remove duplicates if randomizing order!"

            for batch_idx in range(num_batches):
                start = batch_idx*self.circuits_per_batch
                end = min(len(circuits), (batch_idx+1)*self.circuits_per_batch)
                self.pygsti_circuit_batches[batch_idx] = circuits[start:end]
            
            if dirname is not None:
                exp_dir = _pathlib.Path(dirname) / 'ibmqexperiment'
                exp_dir.mkdir(parents=True, exist_ok=True)

                # Checkpoint by writing this member as .write() would
                _io.metadir._write_auxfile_member(exp_dir, 'pygsti_circuit_batches',
                                                  self.auxfile_types['pygsti_circuit_batches'],
                                                  self.pygsti_circuit_batches)

        #create Qiskit quantum circuit for each circuit defined in experiment list
        self.qasm_circuit_batches = [] if self.qasm_circuit_batches is None else self.qasm_circuit_batches
        self.qiskit_circuit_batches = [] if self.qiskit_circuit_batches is None else self.qiskit_circuit_batches

        if len(self.qiskit_circuit_batches):
            print(f'Already completed transpilation of {len(self.qiskit_circuit_batches)}/{num_batches} circuit batches')

        for batch_idx in range(len(self.qiskit_circuit_batches, num_batches)):
            print(f"Transpiling circuit batch {batch_idx+1}/{num_batches}")
            batch = []
            batch_strs = []
            for circ in self.pygsti_circuit_batches[batch_idx]:
                pygsti_openqasm_circ = circ.convert_to_openqasm(num_qubits=self.pspec.num_qubits,
                                                                standard_gates_version='x-sx-rz')
                batch_strs.append(pygsti_openqasm_circ)

                qiskit_qc = _qiskit.QuantumCircuit.from_qasm_str(pygsti_openqasm_circ)
                batch.append(qiskit_qc)
            
            self.qasm_circuit_batches.append(batch_strs)
            self.qiskit_circuit_batches.append(batch)
            
            if dirname is not None:
                exp_dir = _pathlib.Path(dirname) / 'ibmqexperiment'
            
                # Checkpoint by writing this member as .write() would
                # Here, we go to a lower level than pygsti_circuit_batches and write each list entry as they are completed
                subtypes = self.auxfile_types['qasm_circuit_batches'].split(':')
                assert subtypes[0] == 'list', "Checkpointing of qasm_circuit_batches only possible when auxfile_type is list:<subtype>"
                subtype = ':'.join(subtypes[1:])
                _io.metadir._write_auxfile_member(exp_dir, f'qasm_circuit_batches{batch_idx}', subtype, batch_strs)

    def submit(self, ibmq_backend, start=None, stop=None, ignore_job_limit=True,
               wait_time=1, wait_steps=10, dirname=None):
        """
        Submits the jobs to IBM Q, that implements the experiment specified by the ExperimentDesign
        used to create this object.

        Parameters
        ----------
        ibmq_backend: qiskit.providers.ibmq.ibmqbackend.IBMQBackend
            The IBM Q backend to submit the jobs to. Should be the backend corresponding to the
            processor that this experiment has been designed for.

        start: int, optional
            Batch index to start submission (inclusive). Defaults to None,
            which will start submission on the first unsubmitted job.
            Jobs can be resubmitted by manually specifying this,
            i.e. start=0 will start resubmitting jobs from the beginning.

        stop: int, optional
            Batch index to stop submission (exclusive). Defaults to None,
            which will submit as many jobs as possible given the backend's
            maximum job limit.

        ignore_job_limit: bool, optional
            If True, then stop is set to submit all remaining jobs. This is set
            as True to maintain backwards compatibility. Note that is more jobs
            are needed than the max limit, this will enter a wait loop until all
            jobs have been successfully submitted.

        wait_time: int
            Number of seconds for each waiting step.

        wait_steps: int
            Number of steps to take before retrying job submission.

        Returns
        -------
        None
        """
        assert len(self.qiskit_circuit_batches) == len(self.pygsti_circuit_batches), \
            "Transpilation missing! Either run .transpile() first, or if loading from file, " + \
            "use the regen_qiskit_circs=True option in from_dir()."
        
        #Get the backend version
        backend_version = ibmq_backend.version
        
        total_waits = 0
        self.qjobs = [] if self.qjobs is None else self.qjobs
        self.job_ids = [] if self.job_ids is None else self.job_ids

        # Set start and stop to submit the next unsubmitted jobs if not specified
        if start is None:
            start = len(self.qjobs)

        stop = len(self.qiskit_circuit_batches) if stop is None else min(stop, len(self.qiskit_circuit_batches))
        if not ignore_job_limit:
            job_limit = ibmq_backend.job_limit()
            allowed_jobs = job_limit.maximum_jobs - job_limit.active_jobs
            if start + allowed_jobs < stop:
                print(f'Given job limit and active jobs, only {allowed_jobs} can be submitted')

            stop = min(start + allowed_jobs, stop)
        
        for batch_idx, batch in self.qiskit_circuit_batches:
            if batch_idx < start or batch_idx >= stop:
                continue

            print(f"Submitting batch {batch_idx + 1}")
            submit_status = False
            batch_waits = 0
            while not submit_status:
                try:
                    #If submitting to a real device, get calibration data
                    if not ibmq_backend.simulator:
                        backend_properties = ibmq_backend.properties()
                        self.submit_time_calibration_data.append(backend_properties.to_dict())
                    
                    if backend_version == 1:
                        # If using qiskit-ibmq-provider API, assemble into Qobj first
                        qobj = _qiskit.compiler.assemble(batch, shots=self.num_shots)
                        self.qjobs.append(ibmq_backend.run(qobj))
                    else:
                        # Newer qiskit-ibm-provider can take list of Qiskit circuits directly
                        self.qjobs.append(ibmq_backend.run(batch, shots = self.num_shots))
                        
                    status = self.qjobs[-1].status()
                    initializing = True
                    initializing_steps = 0
                    while initializing:
                        if status.name == 'INITIALIZING' or status.name == 'VALIDATING':
                            status = self.qjobs[-1].status()
                            print(f'  - {status} (query {initializing_steps})')
                            _time.sleep(wait_time)
                            initializing_steps += 1
                        else:
                            initializing = False

                    try:
                        job_id = self.qjobs.job_id()
                        print(f'  - Job ID is {job_id}')
                        self.job_ids.append(job_id)
                    except:
                        print('  - Failed to get job_id.')
                        self.job_ids.append(None)
                    
                    try:
                        print(f'  - Queue position is {self.qjobs[-1].queue_info().position)}')
                    except:
                        print(f'  - Failed to get queue position for batch {batch_idx + 1}')
                    submit_status = True

                    # Checkpoint calibration and job id data
                    if dirname is not None:
                        p = _pathlib.Path(dirname)
                        
                        exp_dir = p / 'ibmqexperiment'
                        # Write as .write() would for checkpointing
                        _io.metadir._write_auxfile_member(exp_dir, 'job_ids', 'json', self.job_ids)
                        _io.metadir._write_auxfile_member(exp_dir, 'submit_time_calibration_data', 'json',
                                                          self.submit_time_calibration_data)

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
                    print("This batch has failed {0} times and there have been {1} total failures".format(
                        batch_waits, total_waits))
                    print('Waiting ', end='')
                    for step in range(wait_steps):
                        print('{} '.format(step), end='')
                        _time.sleep(wait_time)
                    print()

    def monitor(self):
        """
        Queries IBM Q for the status of the jobs.
        """
        assert len(self.qjobs) == len(self.job_ids), \
            "Mismatch between jobs and job ids! If loading from file, use the regen_jobs=True option in from_dir()."
        
        for counter, qjob in enumerate(self.qjobs):
            status = qjob.status()
            print(f"Batch {counter + 1}: {status}")
            if status.name == 'QUEUED':
                print(f'  - Queue position is {qjob.queue_info().position}')

        # Print unsubmitted for any entries in qobj but not qjob
        for counter in range(len(self.qjobs), len(self.qiskit_circuit_batches)):
            print(f"Batch {counter + 1}: NOT SUBMITTED")

    def retrieve_results(self):
        """
        Gets the results of the completed jobs from IBM Q, and processes
        them into a pyGSTi DataProtocol object (stored as the key 'data'),
        which can then be used in pyGSTi data analysis routines (e.g., if this
        was a GST experiment, it can input into a GST protocol object that will
        analyze the data).
        """
        assert len(self.qjobs) == len(self.job_ids), \
            "Mismatch between jobs and job ids! If loading from file, use the regen_jobs=True option in from_dir()."
        
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

        #get results from backend jobs and add to dict
        self.batch_result_objects = []
        ds = _data.DataSet()
        for exp_idx, qjob in enumerate(self.qjobs):
            print(f"Querying IBMQ for results objects for batch {exp_idx + 1}...")
            batch_result = qjob.result()
            self.batch_result_objects.append(batch_result)

            # TODO: Checkpoint

            for i, circ in enumerate(self.pygsti_circuit_batches[exp_idx]):
                ordered_target_indices = [self.processor_spec.qubit_labels.index(q) for q in circ.line_labels]
                counts_data = partial_trace(ordered_target_indices, reverse_dict_key_bits(batch_result.get_counts(i)))
                ds.add_count_dict(circ, counts_data)

        self.data = _ProtocolData(self['edesign'], ds)

        # TODO: Checkpoint

    def write(self, dirname=None):
        """
        Writes to disk, storing both the pyGSTi DataProtocol object in pyGSTi's standard
        format and saving all of the IBM Q submission information stored in this object,
        written into the subdirectory 'ibmqexperiment'.

        Parameters
        ----------
        dirname : str
            The *root* directory to write into.  This directory will have
            an 'edesign' subdirectory, which will be created if needed and
            overwritten if present.  If None, then the path this object
            was loaded from is used (if this object wasn't loaded from disk,
            an error is raised).

        """
        if dirname is None:
            dirname = self.edesign._loaded_from
            if dirname is None: raise ValueError("`dirname` must be given because there's no default directory")
        
        dirname = _pathlib.Path(dirname)

        self.edesign.write(dirname)
        
        if self.data is not None:
            self.data.write(dirname, edesign_already_written=True)

        exp_dir = dirname / 'ibmqexperiment'
        exp_dir.mkdir(parents=True, exist_ok=True)
        _io.metadir.write_obj_to_meta_based_dir(self, exp_dir, 'auxfile_types')

        # Handle nonstandard serializations
        # TODO: batch_result_objs?

    @classmethod
    def from_dir(cls, dirname, regen_qiskit_circs=False,
                 regen_runtime_jobs=False, provider=None):
        """
        Initialize a new IBMQExperiment object from `dirname`.

        Parameters
        ----------
        dirname : str
            The directory name.
        
        regen_qiskit_circs: bool, optional
            Whether to recreate the Qiskit circuits from the transpiled
            OpenQASM strings. Defaults to False. You should set this to True
            if you would like to call submit().
        
        regen_runtime_jobs: bool, optional
            Whether to recreate the RuntimeJobs from IBMQ based on the job ides.
            Defaults to False. You should set this to True if you would like to
            call monitor() or retrieve_results().

        provider: IBMProvider
            Provider used to retrieve RuntimeJobs from IBMQ based on job_ids
            (if lazy_qiskit_load is False)

        Returns
        -------
        IBMQExperiment
        """
        p = _pathlib.Path(dirname)
        edesign = _io.read_edesign_from_dir(dirname)
        
        exp_dir = p / 'ibmqexperiment'
        attributes_from_meta = _io.load_meta_based_dir(exp_dir)
        
        ret = cls(edesign, None)
        ret.__dict__.update(attributes_from_meta)

        # Handle nonstandard serialization
        try:
            data = _ProtocolData.from_dir(p, preloaded_edesign=edesign)
            ret.data = data
        except:
            pass

        ret.qiskit_circuit_batches = []
        ret.qjobs = []
        if regen_qiskit_circs:
            # Regenerate Qiskit circuits
            for batch_strs in attributes_from_meta['qasm_circuit_batches']:
                batch = [_qiskit.QuantumCircuit.from_qasm_str(bs) for bs in batch_strs]
                ret.qiskit_circuit_batches.append(batch)
        
        if regen_runtime_jobs:
            # Regenerate RuntimeJobs (if have provider)
            if provider is None:
                _warnings.warn("No provider specified, cannot retrieve IBM jobs")
            else:
                ret._retrieve_jobs(provider)
        
        # Handle nonstandard serializations
        # TODO: batch_result_objs?
        
        return ret

    def _retrieve_jobs(self, provider):
        """Retrieves RuntimeJobs from IBMQ based on job_ids.

        Parameters
        ----------
        provider: IBMProvider
            Provider used to retrieve RuntimeJobs from IBMQ based on job_ids
        """
        for i, jid in enumerate(self.job_ids):
            print(f"Loading job {i+1}/{len(self.job_ids)}...")
            self.qjobs.append(provider.backend.retrieve_job(jid))

""" Functions for sending experiments to IBMQ devices and converting the results to pyGSTi objects """
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

from datetime import datetime as _datetime
from functools import partial as _partial
import json as _json
import numpy as _np
import os as _os
from pathos import multiprocessing as _mp
import pathlib as _pathlib
import pickle as _pickle
import time as _time
import tqdm as _tqdm
import warnings as _warnings

# Try to load Qiskit
try:
    import qiskit as _qiskit
    from qiskit.providers import JobStatus as _JobStatus
except:
    _qiskit = None

# Try to load IBM Runtime
try: 
    from qiskit_ibm_runtime import SamplerV2 as _Sampler
    from qiskit_ibm_runtime import Session as _Session
    from qiskit_ibm_runtime import RuntimeJobV2 as _RuntimeJobV2
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager as _pass_manager
except: _Sampler = None

# Most recent version of QisKit that this has been tested on:
#qiskit.__version__ = '1.1.1'
#qiskit_ibm_runtime.__version__ = '0.25.0'
# Note that qiskit<1.0 is going EOL in August 2024,
# and v1 backends are also being deprecated (we now support only v2)
# Also qiskit-ibm-provider is ALSO being deprecated,
# so I'm only supporting runtime here

try:
    from bson import json_util as _json_util
except ImportError:
    _json_util = None

from pygsti import data as _data, io as _io
from pygsti.protocols import ProtocolData as _ProtocolData, HasProcessorSpec as _HasPSpec
from pygsti.protocols.protocol import _TreeNode
from pygsti.io import metadir as _metadir


# Needs to be defined first for multiprocessing reasons
def _transpile_batch(circs, pass_manager, qasm_convert_kwargs):
    batch = []
    for circ in circs:
        # TODO: Replace this with direct to qiskit
        pygsti_openqasm_circ = circ.convert_to_openqasm(**qasm_convert_kwargs)
        qiskit_qc = _qiskit.QuantumCircuit.from_qasm_str(pygsti_openqasm_circ)
        batch.append(qiskit_qc)
    
    # Run pass manager on batch
    return pass_manager.run(batch)


class IBMQExperiment(_TreeNode, _HasPSpec):
    """
    A object that converts pyGSTi ExperimentDesigns into jobs to be submitted to IBM Q, submits these
    jobs to IBM Q and receives the results.
    """

    @classmethod
    def from_dir(cls, dirname, regen_jobs=False, service=None, new_checkpoint_path=None):
        """
        Initialize a new IBMQExperiment object from `dirname`.

        Parameters
        ----------
        dirname : str
            The directory name.
        
        regen_jobs: bool, optional
            Whether to recreate the RuntimeJobs from IBMQ based on the job ides.
            Defaults to False. You should set this to True if you would like to
            call monitor() or retrieve_results().
        
        service: QiskitRuntimeService
            Service used to retrieve RuntimeJobs from IBMQ based on job_ids
            (if regen_jobs is True).
        
        new_checkpoint_path: str, optional
            A string for the path to use for writing intermediate checkpoint
            files to disk. If None, this defaults to using the same checkpoint
            as the serialized IBMQExperiment object. If provided, this will be
            the new checkpoint path used moving forward. Note that this can be
            the desired {dirname} for an eventual `write({dirname})` call, i.e. the
            serialized IBMQExperiment checkpoint after a successful `retrieve_results()`
            is equivalent to the serialized IBMQExperiment after `write()`.

        Returns
        -------
        IBMQExperiment
        """
        p = _pathlib.Path(dirname)
        edesign = _io.read_edesign_from_dir(dirname)
        
        try:
            exp_dir = p / 'ibmqexperiment'
            attributes_from_meta = _io.load_meta_based_dir(exp_dir)

            # Don't override checkpoint during this construction
            ret = cls(edesign, None, disable_checkpointing=True)
            ret.__dict__.update(attributes_from_meta)
            ret.edesign = edesign
        except KeyError:
            _warnings.warn("Failed to load ibmqexperiment, falling back to old serialization format logic")

            # Don't override checkpoint during this construction
            ret = cls(edesign, None, disable_checkpointing=True)
            with open(p / 'ibmqexperiment' / 'meta.json', 'r') as f:
                from_json = _json.load(f)
            ret.__dict__.update(from_json)

            # Old keys to new class members
            key_attr_map = {
                'pspec': ('processor_spec', None),
                'pygsti_circuits': ('pygsti_circuit_batches', []),
                'pygsti_openqasm_circuits': ('qasm_circuit_batches', []),
                'submit_time_calibration_data': ('submit_time_calibration_data', []),
                'batch_result_object': ('batch_results', [])
            }

            for key, (attr, def_val) in key_attr_map.items():
                with open(p / f'ibmqexperiment' / '{key}.pkl', 'rb') as f:
                    try:
                        setattr(ret, attr, _pickle.load(f))
                    except:
                        _warnings.warn(f"Couldn't unpickle {key}, so setting {attr} to {def_val}.")
                        setattr(ret, attr, def_val)

        # Handle nonstandard serialization
        try:
            data = _ProtocolData.from_dir(p, preloaded_edesign=edesign)
            ret.data = data
        except:
            pass
        
        if ret.qiskit_isa_circuit_batches is None:
            ret.qiskit_isa_circuit_batches = []
        
        # Regenerate Qiskit RuntimeJobs
        ret.qjobs = []
        if regen_jobs:
            assert _Sampler is not None, "Could not import qiskit-ibm-runtime, needed for regen_jobs=True"
            assert service is not None, "No service specified, cannot retrieve IBM jobs"
            ret._retrieve_jobs(service=service)
        
        # Update checkpoint path if requested
        if new_checkpoint_path is not None:
            ret.checkpoint_path = new_checkpoint_path
            if not ret.disable_checkpointing:
                ret.write(ret.checkpoint_path)
        
        return ret

    def __init__(self, edesign, pspec, remove_duplicates=True, randomized_order=True, circuits_per_batch=75,
                 num_shots=1024, seed=None, checkpoint_path=None, disable_checkpointing=False, checkpoint_override=False):
        _TreeNode.__init__(self, None, None)

        self.auxfile_types = {}
        _HasPSpec.__init__(self, pspec)

        self.edesign = edesign
        self.remove_duplicates = remove_duplicates
        self.randomized_order = randomized_order
        self.circuits_per_batch = circuits_per_batch
        self.num_shots = num_shots
        self.seed = seed
        self.checkpoint_path = str(checkpoint_path) if checkpoint_path is not None else 'ibmqexperiment_checkpoint'
        self.disable_checkpointing = disable_checkpointing
        # Populated with transpiling to IBMQ with .transpile()
        self.pygsti_circuit_batches = []
        self.qiskit_isa_circuit_batches = []
        # Populated when submitting to IBM Q with .submit()
        self.qjobs = []
        self.job_ids = []
        self.submit_time_calibration_data = []
        # Populated when grabbing results from IBM Q with .retrieve_results()
        self.batch_results = []
        self.data = None

        # If not in this list, will be automatically dumped to meta.json
        # 'none' means it will not be read in, 'reset' means it will come back in as None
        # Several of these could be stored in the meta.json but are kept external for easy chkpts
        # DEV NOTE: If any of these change, make sure to update the checkpointing code appropriately
        self.auxfile_types['edesign'] = 'none'
        self.auxfile_types['data'] = 'reset'
        # self.processor_spec is handled by _HasPSpec base class
        self.auxfile_types['pygsti_circuit_batches'] = 'list:text-circuit-list'
        self.auxfile_types['qiskit_isa_circuit_batches'] = 'list:qpy'
        self.auxfile_types['qjobs'] = 'none'
        self.auxfile_types['job_ids'] = 'json'
        self.auxfile_types['batch_results'] = 'none' # TODO: Fix this
        if _json_util is not None:
            self.auxfile_types['submit_time_calibration_data'] = 'list:json'
        else:
            # Fall back to pickles if we do not have bson to deal with datetime.datetime
            self.auxfile_types['submit_time_calibration_data'] = 'pickle'

        if not self.disable_checkpointing:
            chkpath = _pathlib.Path(self.checkpoint_path)
            if chkpath.exists() and not checkpoint_override:
                raise RuntimeError(f"Checkpoint {self.checkpoint_path} already exists. Either "
                    + "specify a different checkpoint_path, set checkpoint_override=True to clobber the current checkpoint,"
                    + " or turn checkpointing off via disable_checkpointing=True (not recommended)."
                    )
            self.write(chkpath)
        
    def monitor(self):
        """
        Queries IBM Q for the status of the jobs.
        """
        assert _qiskit is not None, "Could not import qiskit, needed for monitor()"
        assert len(self.qjobs) == len(self.job_ids), \
            "Mismatch between jobs and job ids! If loading from file, use the regen_jobs=True option in from_dir()."
        
        for counter, qjob in enumerate(self.qjobs):
            status = qjob.status()
            print(f"Batch {counter + 1}: {status}")
            if status in [_JobStatus.QUEUED, 'QUEUED']:
                try:
                    print(f'  - Queue position is {qjob.queue_position(True)}')
                except Exception:
                    print('  - Unable to retrieve queue position')
                    if isinstance(self.qjobs[-1], _RuntimeJobV2):
                            print('    (because queue position not available in RuntimeJobV2)')
                            try:
                                metrics = qjob.metrics()
                                start_time = _datetime.fromisoformat(metrics["estimated_start_time"])
                                local_time = start_time.astimezone()
                                print(f'  - Estimated start time: {local_time.strftime("%Y-%m-%d %H:%M:%S")} (local timezone)')
                            except Exception:
                                print(f'  - Unable to retrieve estimated start time')
            elif status in [_JobStatus.ERROR, 'ERROR']:
                try:
                    print(f'  - Error logs: {qjob.logs()}')
                except Exception:
                    print(f'  - Unable to access error logs')

        # Print unsubmitted for any entries in qobj but not qjob
        for counter in range(len(self.qjobs), len(self.qiskit_isa_circuit_batches)):
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
        
        def to_labeled_counts(input_dict, ordered_target_indices, num_qubits_in_pspec): 
            """
            Implements handling for mid-circuit measurement outcomes.  
            """
            outcome_labels = []
            counts_data = []
            for bitstring, count in input_dict.items():
                new_label = []
                term_string = ''
                term_bits = bitstring[:num_qubits_in_pspec][::-1]
                mid_bits = bitstring[num_qubits_in_pspec:][::-1]
                for index in ordered_target_indices:
                    term_string += term_bits[index]
                for bit in mid_bits:
                    new_label.append('p'+bit)
                new_label.append(term_string)
                outcome_labels.append(tuple(new_label))
                counts_data.append(count)
            return outcome_labels, counts_data

        if len(self.batch_results):
            print(f'Already retrieved results of {len(self.batch_results)}/{len(self.qiskit_isa_circuit_batches)} circuit batches')

        #get results from backend jobs and add to dict
        ds = _data.DataSet()
        for exp_idx in range(len(self.batch_results), len(self.qjobs)):
            qjob = self.qjobs[exp_idx]
            print(f"Querying IBMQ for results objects for batch {exp_idx + 1}...")
            batch_result = qjob.result()
            self.batch_results.append(batch_result)

            if not self.disable_checkpointing:
                self._write_checkpoint()

            num_qubits_in_pspec = self.processor_spec.num_qubits
            for i, circ in enumerate(self.pygsti_circuit_batches[exp_idx]):
                ordered_target_indices = [self.processor_spec.qubit_labels.index(q) for q in circ.line_labels] 
                labeled_counts = to_labeled_counts(batch_result[i].data.cr.get_counts(), ordered_target_indices, num_qubits_in_pspec)
                outcome_labels = labeled_counts[0]
                counts_data = labeled_counts[1]
                ds.add_count_list(circ, outcome_labels, counts_data)


        self.data = _ProtocolData(self.edesign, ds)

        if not self.disable_checkpointing:
            self.data.write(self.checkpoint_path, edesign_already_written=True)

    def submit(self, ibmq_backend, start=None, stop=None, ignore_job_limit=True, wait_time=5, max_attempts=10):
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
        assert _qiskit is not None, "Could not import qiskit, needed for submit()"
        assert _Sampler is not None, "Could not import qiskit-ibm-runtime, needed for submit()"

        assert len(self.qiskit_isa_circuit_batches) == len(self.pygsti_circuit_batches), \
            "Transpilation missing! Either run .transpile() first, or if loading from file, " + \
            "use the regen_qiskit_circs=True option in from_dir()."
        
        #Get the backend version
        backend_version = ibmq_backend.version
        assert backend_version >= 2, "IBMQExperiment no longer supports v1 backends due to their deprecation by IBM"
        
        total_waits = 0
        self.qjobs = [] if self.qjobs is None else self.qjobs
        self.job_ids = [] if self.job_ids is None else self.job_ids

        # Set start and stop to submit the next unsubmitted jobs if not specified
        if start is None:
            start = len(self.qjobs)

        stop = len(self.qiskit_isa_circuit_batches) if stop is None else min(stop, len(self.qiskit_isa_circuit_batches))
        if not ignore_job_limit:
            job_limit = ibmq_backend.job_limit()
            allowed_jobs = job_limit.maximum_jobs - job_limit.active_jobs
            if start + allowed_jobs < stop:
                print(f'Given job limit and active jobs, only {allowed_jobs} can be submitted')

            stop = min(start + allowed_jobs, stop)
        
        ibmq_session = _Session(backend = ibmq_backend)
        sampler = _Sampler(session=ibmq_session)
        
        for batch_idx, batch in enumerate(self.qiskit_isa_circuit_batches):
            if batch_idx < start or batch_idx >= stop:
                continue

            print(f"Submitting batch {batch_idx + 1}")
            submit_status = False
            batch_waits = 0
            while not submit_status and batch_waits < max_attempts:
                try:
                    #If submitting to a real device, get calibration data
                    try:
                        backend_properties = ibmq_backend.properties()
                        self.submit_time_calibration_data.append(backend_properties.to_dict())
                    except AttributeError:
                        # We can't get the properties
                        # Likely this is a fake backend/simulator, append empty submit data
                        self.submit_time_calibration_data.append({})

                    # Submit job
                    self.qjobs.append(sampler.run(batch, shots = self.num_shots))
                        
                    status = self.qjobs[-1].status()
                    initializing = True
                    initializing_steps = 0
                    while initializing and initializing_steps < max_attempts:
                        if status in [_JobStatus.INITIALIZING, "INITIALIZING", _JobStatus.VALIDATING, "VALIDATING"]:
                            status = self.qjobs[-1].status()
                            print(f'  - {status} (query {initializing_steps})')
                            _time.sleep(wait_time)
                            initializing_steps += 1
                        else:
                            initializing = False

                    try:
                        job_id = self.qjobs[-1].job_id()
                        print(f'  - Job ID is {job_id}')
                        self.job_ids.append(job_id)
                    except Exception:
                        
                        print('  - Failed to get job_id.')
                        self.job_ids.append(None)
                    
                    try:
                        print(f'  - Queue position is {self.qjobs[-1].queue_position()}')
                    except Exception:
                        print(f'  - Failed to get queue position for batch {batch_idx + 1}')
                        if isinstance(self.qjobs[-1], _RuntimeJobV2):
                            print('    (because queue position not available in RuntimeJobV2)')
                            try:
                                metrics = self.qjobs[-1].metrics()
                                start_time = _datetime.fromisoformat(metrics["estimated_start_time"])
                                print(f'  - Estimated start time: {start_time.astimezone()} (local timezone)')
                            except Exception:
                                print(f'  - Unable to retrieve estimated start time')
                            
                    submit_status = True

                except Exception as ex:
                    template = "  An exception of type {0} occurred. Arguments:\n{1!r}"
                    message = template.format(type(ex).__name__, ex.args)
                    print(message)
                    try:
                        print('  Machine status is {}.'.format(ibmq_backend.status().status_msg))
                    except Exception as ex1:
                        print('  Failed to get machine status!')
                        template = "  An exception of type {0} occurred. Arguments:\n{1!r}"
                        message = template.format(type(ex).__name__, ex1.args)
                        print(message)
                    total_waits += 1
                    batch_waits += 1
                    print(f"This batch has failed {batch_waits} times and there have been {total_waits} total failures")
                    print('Waiting', end='')
                    _time.sleep(wait_time)
                finally:
                    # Checkpoint calibration and job id data
                    if not self.disable_checkpointing:
                        chkpt_path = _pathlib.Path(self.checkpoint_path) / "ibmqexperiment"
                        with open(chkpt_path / 'meta.json', 'r') as f:
                            metadata = _json.load(f)

                        _metadir._write_auxfile_member(chkpt_path, 'job_ids', self.auxfile_types['job_ids'], self.job_ids)
                        
                        if self.auxfile_types['submit_time_calibration_data'] == 'list:json':
                            # We only need to write the last calibration data
                            filenm = f"submit_time_calibration_data{len(self.submit_time_calibration_data)-1}"
                            _metadir._write_auxfile_member(chkpt_path, filenm, 'json', self.submit_time_calibration_data[-1])
                            metadata['submit_time_calibration_data'].append(None)
                        else:
                            # We are pickling the whole thing, no option to do incremental
                            _metadir._write_auxfile_member(chkpt_path, 'submit_time_calibration_data', 'pickle', self.submit_time_calibration_data)
                        
                        with open(chkpt_path / 'meta.json', 'w') as f:
                            _json.dump(metadata, f, indent=4)
            
            if submit_status is False:
                raise RuntimeError("Ran out of max attempts and job was still not submitted successfully")

    def transpile(self, ibmq_backend, qiskit_pass_kwargs=None, qasm_convert_kwargs=None, num_workers=1):
        """Transpile pyGSTi circuits into Qiskit circuits for submission to IBMQ.

        Parameters
        ----------
        ibmq_backend:
            IBM backend to use during Qiskit transpilation
        
        opt_level: int, optional
            Optimization level for Qiskit `generate_preset_pass_manager`.
        
        qiskit_pass_kwargs: dict, optional
            Additional kwargs to pass in to `generate_preset_pass_manager`.
            If not defined, the default is {'seed_transpiler': self.seed, 'optimization_level': 0,
            'basis_gates': ibmq_backend.operation_names}
            Note that "optimization_level" is a required argument to the pass manager.
        
        qasm_convert_kwargs: dict, optional
            Additional kwargs to pass in to `Circuit.convert_to_openqasm`.
            If not defined, the default is {'num_qubits': self.processor_spec.num_qubits,
            'standard_gates_version': 'x-sx-rz'}
        
        num_workers: int, optional
            Number of workers to use for parallel (by batch) transpilation
        """
        circuits = self.edesign.all_circuits_needing_data.copy()
        num_batches = int(_np.ceil(len(circuits) / self.circuits_per_batch))

        if qiskit_pass_kwargs is None:
            qiskit_pass_kwargs = {}
        qiskit_pass_kwargs['seed_transpiler'] = qiskit_pass_kwargs.get('seed_transpiler', self.seed)
        qiskit_pass_kwargs['optimization_level'] = qiskit_pass_kwargs.get('optimization_level', 0)
        qiskit_pass_kwargs['basis_gates'] = qiskit_pass_kwargs.get('basis_gates', ibmq_backend.operation_names)
        
        if qasm_convert_kwargs is None:
            qasm_convert_kwargs = {}
        qasm_convert_kwargs['num_qubits'] = qasm_convert_kwargs.get('num_qubits', self.processor_spec.num_qubits)
        qasm_convert_kwargs['standard_gates_version'] = qasm_convert_kwargs.get('standard_gates_version', 'x-sx-rz')

        if not len(self.pygsti_circuit_batches):
            rand_state = _np.random.RandomState(self.seed) # TODO: Should this be a different seed as transpiler?

            if self.randomized_order:
                if self.remove_duplicates:
                    circuits = list(set(circuits))
                rand_state.shuffle(circuits)
            else:
                assert(not self.remove_duplicates), "Can only remove duplicates if randomizing order!"

            for batch_idx in range(num_batches):
                start = batch_idx*self.circuits_per_batch
                end = min(len(circuits), (batch_idx+1)*self.circuits_per_batch)
                self.pygsti_circuit_batches.append(circuits[start:end])
            
            if not self.disable_checkpointing:
                chkpt_path = _pathlib.Path(self.checkpoint_path) / "ibmqexperiment"
                with open(chkpt_path / 'meta.json', 'r') as f:
                    metadata = _json.load(f)
                
                pcbdata = _metadir._write_auxfile_member(chkpt_path, 'pygsti_circuit_batches', self.auxfile_types['pygsti_circuit_batches'], self.pygsti_circuit_batches)
                if 'pygsti_circuit_batches' in metadata:
                    metadata['pygsti_circuit_batches'] = pcbdata
                
                with open(chkpt_path / 'meta.json', 'w') as f:
                    _json.dump(metadata, f)

        if len(self.qiskit_isa_circuit_batches):
            print(f'Already completed transpilation of {len(self.qiskit_isa_circuit_batches)}/{num_batches} circuit batches')
            if len(self.qiskit_isa_circuit_batches) == num_batches:
                return

        pm = _pass_manager(backend=ibmq_backend, **qiskit_pass_kwargs)

        # Set up parallel tasks
        tasks = [self.pygsti_circuit_batches[i] for i in range(len(self.qiskit_isa_circuit_batches), num_batches)]

        # We want to use transpile_batch and it's the same pm/convert kwargs, so create a new function with partially applied kwargs
        # This function now only takes circs as an argument (which are our task elements above)
        task_fn = _partial(_transpile_batch, pass_manager=pm, qasm_convert_kwargs=qasm_convert_kwargs)

        # Run in parallel (p.imap) with progress bars (tqdm)
        #with _mp.Pool(num_workers) as p:
        #    isa_circuits = list(_tqdm.tqdm(p.imap(task_fn, tasks), total=len(tasks)))
        for task in _tqdm.tqdm(tasks):
            self.qiskit_isa_circuit_batches.append(task_fn(task))

            # Save single batch
            chkpt_path = _pathlib.Path(self.checkpoint_path) / "ibmqexperiment"
            with open(chkpt_path / 'meta.json', 'r') as f:
                metadata = _json.load(f)

            filenm = f"qiskit_isa_circuit_batches{len(self.qiskit_isa_circuit_batches)-1}"
            _metadir._write_auxfile_member(chkpt_path, filenm, 'qpy', self.qiskit_isa_circuit_batches[-1])    
            if 'qiskit_isa_circuit_batches' in metadata:
                metadata['qiskit_isa_circuit_batches'].append(None)
            
            with open(chkpt_path / 'meta.json', 'w') as f:
                _json.dump(metadata, f)

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
            dirname = self.checkpoint_path
            if dirname is None:
                raise ValueError("`dirname` must be given because there's no checkpoint or default edesign directory")
        
        dirname = _pathlib.Path(dirname)

        self.edesign.write(dirname)
        
        if self.data is not None:
            self.data.write(dirname, edesign_already_written=True)

        self._write_checkpoint(dirname)
    
    def _write_checkpoint(self, dirname=None):
        """Write only the ibmqexperiment part of .write().
        
        Parameters
        ----------
        dirname : str
            The *root* directory to write into.  This directory will have
            an 'edesign' subdirectory, which will be created if needed and
            overwritten if present.  If None, then the path this object
            was loaded from is used (if this object wasn't loaded from disk,
            an error is raised).
        """
        dirname = dirname if dirname is not None else self.checkpoint_path
        exp_dir = _pathlib.Path(dirname) / 'ibmqexperiment'
        exp_dir.mkdir(parents=True, exist_ok=True)
        _io.metadir.write_obj_to_meta_based_dir(self, exp_dir, 'auxfile_types')

    def _retrieve_jobs(self, service):
        """Retrieves RuntimeJobs from IBMQ based on job_ids.

        Parameters
        ----------
        provider: IBMProvider
            Provider used to retrieve RuntimeJobs from IBMQ based on job_ids
        """
        for i, jid in enumerate(self.job_ids):
            print(f"Loading job {i+1}/{len(self.job_ids)}...")
            self.qjobs.append(service.job(jid))



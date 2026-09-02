from ...util import BaseCase

import pytest
import pygsti
from pygsti.extras.devices.experimentaldevice import ExperimentalDevice
from pygsti.extras import ibmq
from pygsti.extras.ibmq import ibmqexperiment
from pygsti.processors import CliffordCompilationRules as CCR
from pygsti.protocols import MirrorRBDesign as RMCDesign
from pygsti.protocols import PeriodicMirrorCircuitDesign as PMCDesign
from pygsti.protocols import FreeformDesign
from pygsti.protocols import ByDepthSummaryStatistics
from pygsti.modelpacks import smq1Q_XY
from pygsti.protocols import StandardGSTDesign
from pygsti.tools.exceptions import pyGSTiDeprecationWarning
import numpy as np
from unittest import mock


try:
    from qiskit.providers.fake_provider import GenericBackendV2
except:
    GenericBackendV2 = None


try:
    from qiskit_ibm_runtime import QiskitRuntimeService
except:
    QiskitRuntimeService = None


try:
    from qiskit_ibm_runtime import Batch as _Batch
    from qiskit_ibm_runtime import Session as _Session
except ImportError:
    _Batch = None
    _Session = None


class IBMQExperimentTester(BaseCase):

    @classmethod
    def setup_class(cls):
        if GenericBackendV2 is None:
            pytest.skip('Qiskit is required for this operation, and does not appear to be installed.')

        cls.backend = GenericBackendV2(num_qubits=4, noise_info=False) # noise_info=False guarantees ideal simulation, which is needed at least for test_e2e_mirror_rb
        cls.device = ExperimentalDevice.from_qiskit_backend(cls.backend)
        cls.pspec = cls.device.create_processor_spec(['Gc{}'.format(i) for i in range(24)] + ['Gcphase'])

        compilations = {'absolute': CCR.create_standard(cls.pspec, 'absolute', ('paulis', '1Qcliffords'), verbosity=0)}

        mirror_design = RMCDesign(cls.pspec, [0, 2, 4], 10, qubit_labels=('Q0', 'Q1', 'Q2'),
                                  clifford_compilations=compilations, sampler='edgegrab', samplerargs=[3/8,])
        cls.edesign = pygsti.protocols.CombinedExperimentDesign([mirror_design])

    def test_init(self):
        exp1 = ibmq.IBMQExperiment(self.edesign, self.pspec, circuits_per_batch=5, num_shots=1024, seed=20231201,
                                   disable_checkpointing=True)

        chkpt = 'test_ibmq_init_checkpoint'
        exp2 = ibmq.IBMQExperiment(self.edesign, self.pspec, circuits_per_batch=5, num_shots=1024, seed=20231201,
                                   checkpoint_path=chkpt, checkpoint_override=True)

        assert exp2.pygsti_circuit_batches == exp1.pygsti_circuit_batches

        exp3 = ibmq.IBMQExperiment.from_dir(chkpt)
        assert exp3.pygsti_circuit_batches == exp1.pygsti_circuit_batches

    def test_transpile(self):
        if QiskitRuntimeService is None:
            pytest.skip('Qiskit Runtime is required for this operation, and does not appear to be installed.')
        chkpt = 'test_ibmq_transpile_checkpoint'
        exp1 = ibmq.IBMQExperiment(self.edesign, self.pspec, circuits_per_batch=5, num_shots=1024, seed=20231201,
                                   checkpoint_path=chkpt, checkpoint_override=True)
        exp1.transpile(self.backend)

        # Test checkpoint load
        exp2 = ibmq.IBMQExperiment.from_dir(chkpt, regen_jobs=True, service=QiskitRuntimeService(channel='local'))
        assert exp2.qiskit_isa_circuit_batches == exp1.qiskit_isa_circuit_batches

        # Test restart
        del exp2.qiskit_isa_circuit_batches[2:]
        exp2.transpile(self.backend)
        assert exp2.qiskit_isa_circuit_batches == exp1.qiskit_isa_circuit_batches

    def test_submit(self):
        chkpt = 'test_ibmq_submit_checkpoint'
        exp1 = ibmq.IBMQExperiment(self.edesign, self.pspec, circuits_per_batch=5, num_shots=1024, seed=20231201,
                                   checkpoint_path=chkpt, checkpoint_override=True)
        exp1.transpile(self.backend)

        # Submit first 3 jobs
        exp1.submit(self.backend, stop=3, max_attempts=1)
        assert len(exp1.qjobs) == 3

        # Submit rest of jobs
        exp1.submit(self.backend, max_attempts=1)
        assert len(exp1.qjobs) == len(exp1.qiskit_isa_circuit_batches)

    def test_submit_default_uses_batch(self):
        """Default submit() call constructs Batch (not Session) and closes it."""
        chkpt = 'test_ibmq_submit_default_uses_batch'
        exp = ibmq.IBMQExperiment(self.edesign, self.pspec, circuits_per_batch=5, num_shots=1024, seed=20231201,
                                  checkpoint_path=chkpt, checkpoint_override=True)
        exp.transpile(self.backend)

        # Track calls to _Batch and _Session constructors and close() calls.
        batch_instances = []
        session_instances = []
        original_batch = ibmqexperiment._Batch
        original_session = ibmqexperiment._Session

        def mock_batch_constructor(*args, **kwargs):
            instance = original_batch(*args, **kwargs)
            batch_instances.append(instance)
            # Wrap close() to track calls
            original_close = instance.close
            close_calls = []

            def close_wrapper():
                close_calls.append(True)
                original_close()
            instance.close = close_wrapper
            instance._close_calls = close_calls
            return instance

        def mock_session_constructor(*args, **kwargs):
            instance = original_session(*args, **kwargs)
            session_instances.append(instance)
            return instance

        with mock.patch.object(ibmqexperiment, '_Batch', side_effect=mock_batch_constructor), \
             mock.patch.object(ibmqexperiment, '_Session', side_effect=mock_session_constructor):
            exp.submit(self.backend, stop=1, max_attempts=1)
            # Batch should be constructed exactly once
            assert len(batch_instances) == 1
            # Session should never be constructed
            assert len(session_instances) == 0
            # close() should be called exactly once on the Batch
            assert len(batch_instances[0]._close_calls) == 1

    def test_submit_caller_supplied_runtime_mode_not_closed(self):
        """Caller-supplied ibmq_runtime_mode is used and not closed by submit()."""
        chkpt = 'test_ibmq_submit_caller_mode_not_closed'
        exp = ibmq.IBMQExperiment(self.edesign, self.pspec, circuits_per_batch=5, num_shots=1024, seed=20231201,
                                  checkpoint_path=chkpt, checkpoint_override=True)
        exp.transpile(self.backend)

        # Create a real Batch instance that we will supply to submit()
        caller_batch = _Batch(backend=self.backend)

        with mock.patch.object(ibmqexperiment, '_Batch', wraps=ibmqexperiment._Batch) as mock_batch_cls, \
             mock.patch.object(ibmqexperiment, '_Session', wraps=ibmqexperiment._Session) as mock_session_cls, \
             mock.patch.object(caller_batch, 'close') as mock_instance_close:
            exp.submit(self.backend, ibmq_runtime_mode=caller_batch, stop=1, max_attempts=1)
            # No new Batch/Session should be constructed (caller supplied theirs)
            mock_batch_cls.assert_not_called()
            mock_session_cls.assert_not_called()
            # The supplied batch instance's close() should NOT be called
            mock_instance_close.assert_not_called()

    def test_submit_deprecated_ibmq_session_still_works(self):
        """Deprecated ibmq_session parameter still works and emits deprecation warning."""
        chkpt = 'test_ibmq_submit_deprecated_session'
        exp = ibmq.IBMQExperiment(self.edesign, self.pspec, circuits_per_batch=5, num_shots=1024, seed=20231201,
                                  checkpoint_path=chkpt, checkpoint_override=True)
        exp.transpile(self.backend)

        # Create a real Session instance to pass as deprecated ibmq_session
        caller_session = _Session(backend=self.backend)

        # Must catch the deprecation warning and verify job submission succeeded
        with pytest.warns(pyGSTiDeprecationWarning):
            with mock.patch.object(caller_session, 'close') as mock_session_close:
                exp.submit(self.backend, ibmq_session=caller_session, stop=1, max_attempts=1)
                # Jobs should be submitted (qjobs list grows)
                assert len(exp.qjobs) > 0
                # Caller-supplied session should NOT be closed
                mock_session_close.assert_not_called()

    def test_submit_conflicting_session_and_runtime_mode_raises(self):
        """Passing both ibmq_session and ibmq_runtime_mode raises ValueError."""
        chkpt = 'test_ibmq_submit_conflicting_params'
        exp = ibmq.IBMQExperiment(self.edesign, self.pspec, circuits_per_batch=5, num_shots=1024, seed=20231201,
                                  checkpoint_path=chkpt, checkpoint_override=True)
        exp.transpile(self.backend)

        # Use non-None placeholders (ValueError is raised before these are used)
        with pytest.raises(ValueError, match="Cannot specify both"):
            exp.submit(self.backend, ibmq_session=object(), ibmq_runtime_mode=object())

    def test_retrieve_results_checkpointing_mode_data_default(self):
        """Default retrieve_results() calls data.write(), not full write()."""
        chkpt = 'test_ibmq_retrieve_checkpointing_data_default'
        exp = ibmq.IBMQExperiment(self.edesign, self.pspec, circuits_per_batch=5, num_shots=1024, seed=20231201,
                                  checkpoint_path=chkpt, checkpoint_override=True)
        exp.transpile(self.backend)
        exp.submit(self.backend, stop=1, max_attempts=1)

        # Mock the write method at the IBMQExperiment level and track data.write calls
        with mock.patch.object(exp, 'write') as mock_write, \
             mock.patch('pygsti.protocols.ProtocolData.write') as mock_pdata_write:
            exp.retrieve_results()  # Use default checkpointing_mode="data"
            # Full write() should NOT be called
            mock_write.assert_not_called()
            # data.write() should be called exactly once
            mock_pdata_write.assert_called_once()

    def test_retrieve_results_checkpointing_mode_full(self):
        """retrieve_results(checkpointing_mode='full') calls full write()."""
        chkpt = 'test_ibmq_retrieve_checkpointing_full'
        exp = ibmq.IBMQExperiment(self.edesign, self.pspec, circuits_per_batch=5, num_shots=1024, seed=20231201,
                                  checkpoint_path=chkpt, checkpoint_override=True)
        exp.transpile(self.backend)
        exp.submit(self.backend, stop=1, max_attempts=1)

        # Mock the write method at the IBMQExperiment level and track data.write calls
        with mock.patch.object(exp, 'write') as mock_write, \
             mock.patch('pygsti.protocols.ProtocolData.write') as mock_pdata_write:
            exp.retrieve_results(checkpointing_mode="full")
            # Full write() should be called exactly once
            mock_write.assert_called_once()
            # data.write() should NOT be called (write() handles it internally)
            mock_pdata_write.assert_not_called()

    def test_retrieve_results_checkpointing_mode_none(self):
        """retrieve_results(checkpointing_mode='none') calls neither write() nor data.write()."""
        chkpt = 'test_ibmq_retrieve_checkpointing_none'
        exp = ibmq.IBMQExperiment(self.edesign, self.pspec, circuits_per_batch=5, num_shots=1024, seed=20231201,
                                  checkpoint_path=chkpt, checkpoint_override=True)
        exp.transpile(self.backend)
        exp.submit(self.backend, stop=1, max_attempts=1)

        # Mock the write method at the IBMQExperiment level and track data.write calls
        with mock.patch.object(exp, 'write') as mock_write, \
             mock.patch('pygsti.protocols.ProtocolData.write') as mock_pdata_write:
            exp.retrieve_results(checkpointing_mode="none")
            # Neither write() nor data.write() should be called
            mock_write.assert_not_called()
            mock_pdata_write.assert_not_called()

    def test_retrieve_results_checkpointing_mode_invalid(self):
        """retrieve_results() with invalid checkpointing_mode raises ValueError."""
        chkpt = 'test_ibmq_retrieve_checkpointing_invalid'
        exp = ibmq.IBMQExperiment(self.edesign, self.pspec, circuits_per_batch=5, num_shots=1024, seed=20231201,
                                  checkpoint_path=chkpt, checkpoint_override=True)
        exp.transpile(self.backend)
        exp.submit(self.backend, stop=1, max_attempts=1)

        # Invalid checkpointing_mode should raise ValueError listing the valid options
        with pytest.raises(ValueError, match="Invalid checkpointing_mode.*'data'.*'full'.*'none'"):
            exp.retrieve_results(checkpointing_mode="bogus")

    #integration tests with end-to-end workflows.
    def test_e2e_mirror_rb(self):
        # Have to do int(i) because variable is of wrong type. Well, maybe.
        edges = [(int(i), int(j)) for (i,j) in list(self.backend.coupling_map.get_edges())]
        qubit_labels = [i for i in range(self.backend.num_qubits)]
        num_qubits = self.backend.num_qubits
        two_qubit_gate = 'Gcphase'
        gate_names = ['Gc{}'.format(i) for i in range(24)] + [two_qubit_gate,]
        availability = {two_qubit_gate: edges}
        pspec = pygsti.processors.QubitProcessorSpec(num_qubits, gate_names, availability=availability,
                                                    qubit_labels=qubit_labels)
        clifford_compilations = {'absolute': pygsti.processors.CliffordCompilationRules.create_standard(pspec, verbosity=0)}

        #mirror rb design parameters
        qubit_labels = [i for i in range(self.backend.num_qubits)]
        widths = [1, 2, 3, 4]
        depths = [0, 10]
        qubits = {w: tuple(qubit_labels[0:w]) for w in widths}
        circuits_per_shape = 5
        xi = {w:1/4 for w in widths}
        if 1 in widths: xi[1] = 0 # No two-qubit gates in one-qubit circuits.

        #build mirror RB design
        edesigns = {}
        for w in widths:
            key = str(w)+ '-' 'random'
            edesigns[key] = RMCDesign(pspec, depths, circuits_per_shape, clifford_compilations=clifford_compilations,
                                    qubit_labels=qubits[w], sampler='edgegrab', samplerargs=[xi[w],])

        for w in widths:
            key = str(w)+ '-' 'periodic'
            # xi has a different meaning in the PMC design --> twice what it is in RMC design
            edesigns[key] = PMCDesign(pspec, depths, circuits_per_shape, clifford_compilations=clifford_compilations,
                                    qubit_labels=qubits[w], sampler='edgegrab', samplerargs=[xi[w]/2,])

        combined_edesign = pygsti.protocols.CombinedExperimentDesign(edesigns)

        exp = ibmq.IBMQExperiment(combined_edesign, pspec, checkpoint_override=True)
        exp.transpile(self.backend)
        exp.submit(self.backend)
        exp.monitor()
        exp.retrieve_results()

        data = exp.data

        # import ipdb

        # ipdb.set_trace()

        # The summary statistics to calculate for each circuit.
        statistics = ['polarization', 'success_probabilities', 'success_counts', 'total_counts', 'two_q_gate_count']
        stats_generator = pygsti.protocols.SimpleRunner(ByDepthSummaryStatistics(statistics_to_compute=statistics))

        # Computes the summary statistics for each circuit
        results = stats_generator.run(data)

        # Turns the results into a data frame.
        df = results.to_dataframe('ValueName', drop_columns=['ProtocolName','ProtocolType'])

        # Here's a simple test that everything worked correctly (it's a noise-free simulation)
        assert(all(1. == df['success_probabilities']))

    #End-to-end integration test for MCM GST.
    def test_e2e_MCM_gst(self):
        ql = ('Q0', )
        target_model = smq1Q_XY.target_model(qubit_labels=ql)
        prep_fiducials = smq1Q_XY.prep_fiducials(qubit_labels=ql)
        meas_fiducials = smq1Q_XY.meas_fiducials(qubit_labels=ql)
        germs = smq1Q_XY.germs(qubit_labels=ql)

        Q0 = np.array([[0.5,0,0,0.5],[0,0,0,0],[0,0,0,0],[0.5,0,0,0.5]])
        Q1 = np.array([[0.5,0,0,-0.5],[0,0,0,0],[0,0,0,0],[-0.5,0,0,0.5]])
        target_model['Iz', ql[0]] = pygsti.modelmembers.instruments.TPInstrument({'p0':Q0,'p1':Q1})
        germs += [pygsti.circuits.Circuit([('Iz', ql[0])])]

        edesign = StandardGSTDesign(target_model, prep_fiducials, meas_fiducials, germs, [1])
        exp = ibmq.IBMQExperiment(edesign, self.pspec, checkpoint_override=True)
        exp.transpile(self.backend)
        exp.submit(self.backend)
        exp.monitor()
        exp.retrieve_results()

    def test_e2e_openqasm_w_mcms(self):
        backend = GenericBackendV2(num_qubits=9, noise_info=False)
        device = ExperimentalDevice.from_qiskit_backend(backend)
        pspec = device.create_processor_spec(['Gc{}'.format(i) for i in range(24)] + ['Gcnot'])

        circ = pygsti.circuits.Circuit('Iz:Q1Iz:Q3Gxpi:Q1Gxpi:Q3Gxpi:Q3@(Q1,Q3)')
        edesign = FreeformDesign({circ: {}})

        exp = ibmq.IBMQExperiment(edesign, pspec, disable_checkpointing=True)
        exp.transpile(ibmq_backend=backend)

        exp.submit(ibmq_backend=backend)
        exp.retrieve_results()

        self.assertEqual(exp.data.dataset[circ].counts, {('p0', 'p0', '10',): 1024})

    def test_e2e_openqasm_no_mcms(self):
        backend = GenericBackendV2(num_qubits=9, noise_info=False)
        device = ExperimentalDevice.from_qiskit_backend(backend)
        pspec = device.create_processor_spec(['Gc{}'.format(i) for i in range(24)] + ['Gcnot'])

        circ = pygsti.circuits.Circuit('Gxpi:Q1Gxpi:Q3Gxpi:Q3@(Q1,Q3)')
        edesign = FreeformDesign({circ: {}})

        exp = ibmq.IBMQExperiment(edesign, pspec, disable_checkpointing=True)
        exp.transpile(ibmq_backend=backend)

        exp.submit(ibmq_backend=backend)
        exp.retrieve_results()

        self.assertEqual(exp.data.dataset[circ].counts, {('10',): 1024})

    def test_e2e_qiskit_all_w_mcms(self):
        backend = GenericBackendV2(num_qubits=9, noise_info=False)
        device = ExperimentalDevice.from_qiskit_backend(backend)
        pspec = device.create_processor_spec(['Gc{}'.format(i) for i in range(24)] + ['Gcnot'])

        circ = pygsti.circuits.Circuit('Iz:Q1Iz:Q3Gxpi:Q1Gxpi:Q3Gxpi:Q3@(Q1,Q3)')
        edesign = FreeformDesign({circ: {}})

        exp = ibmq.IBMQExperiment(edesign, pspec, disable_checkpointing=True)

        qiskit_convert_kwargs = {'qubits_to_measure': 'all'}
        exp.transpile(ibmq_backend=backend, direct_to_qiskit=True, qiskit_convert_kwargs=qiskit_convert_kwargs)

        exp.submit(ibmq_backend=backend)
        exp.retrieve_results()

        self.assertEqual(exp.data.dataset[circ].counts, {('p0', 'p0', '10',): 1024})

    def test_e2e_qiskit_all_no_mcms(self):
        backend = GenericBackendV2(num_qubits=9, noise_info=False)
        device = ExperimentalDevice.from_qiskit_backend(backend)
        pspec = device.create_processor_spec(['Gc{}'.format(i) for i in range(24)] + ['Gcnot'])

        circ = pygsti.circuits.Circuit('Gxpi:Q1Gxpi:Q3Gxpi:Q3@(Q1,Q3)')
        edesign = FreeformDesign({circ: {}})

        exp = ibmq.IBMQExperiment(edesign, pspec, disable_checkpointing=True)

        qiskit_convert_kwargs = {'qubits_to_measure': 'all'}
        exp.transpile(ibmq_backend=backend, direct_to_qiskit=True, qiskit_convert_kwargs=qiskit_convert_kwargs)

        exp.submit(ibmq_backend=backend)
        exp.retrieve_results()

        self.assertEqual(exp.data.dataset[circ].counts, {('10',): 1024})

    def test_e2e_qiskit_active_w_mcms(self):
        backend = GenericBackendV2(num_qubits=9, noise_info=False)
        device = ExperimentalDevice.from_qiskit_backend(backend)
        pspec = device.create_processor_spec(['Gc{}'.format(i) for i in range(24)] + ['Gcnot'])

        circ = pygsti.circuits.Circuit('Iz:Q1Iz:Q3Gxpi:Q1Gxpi:Q3Gxpi:Q3@(Q1,Q3)')
        edesign = FreeformDesign({circ: {}})

        exp = ibmq.IBMQExperiment(edesign, pspec, disable_checkpointing=True)

        qiskit_convert_kwargs = {'qubits_to_measure': 'active'}
        exp.transpile(ibmq_backend=backend, direct_to_qiskit=True, qiskit_convert_kwargs=qiskit_convert_kwargs)

        exp.submit(ibmq_backend=backend)
        exp.retrieve_results()

        self.assertEqual(exp.data.dataset[circ].counts, {('p0', 'p0', '10',): 1024})

    def test_e2e_qiskit_active_no_mcms(self):
        backend = GenericBackendV2(num_qubits=9, noise_info=False)
        device = ExperimentalDevice.from_qiskit_backend(backend)
        pspec = device.create_processor_spec(['Gc{}'.format(i) for i in range(24)] + ['Gcnot'])

        circ = pygsti.circuits.Circuit('Gxpi:Q1Gxpi:Q3Gxpi:Q3@(Q1,Q3)')
        edesign = FreeformDesign({circ: {}})

        exp = ibmq.IBMQExperiment(edesign, pspec, disable_checkpointing=True)

        qiskit_convert_kwargs = {'qubits_to_measure': 'active'}
        exp.transpile(ibmq_backend=backend, direct_to_qiskit=True, qiskit_convert_kwargs=qiskit_convert_kwargs)

        exp.submit(ibmq_backend=backend)
        exp.retrieve_results()

        self.assertEqual(exp.data.dataset[circ].counts, {('10',): 1024})

    def test_circuits_per_batch_default_is_3000(self):
        """circuits_per_batch defaults to 3000 when unspecified."""
        exp = ibmq.IBMQExperiment(self.edesign, self.pspec, disable_checkpointing=True)
        self.assertEqual(exp.circuits_per_batch, 3000)

    def test_batch_exceeding_max_executions_raises(self):
        """Batch whose execution count exceeds MAX_EXECUTIONS_PER_JOB raises ValueError."""
        chkpt = 'test_ibmq_batch_max_executions'
        exp = ibmq.IBMQExperiment(self.edesign, self.pspec, circuits_per_batch=5, num_shots=1024,
                                  seed=20231201, checkpoint_path=chkpt, checkpoint_override=True)

        # Patch MAX_EXECUTIONS_PER_JOB to a low value to trigger the check without needing
        # to construct an actual huge batch
        with mock.patch.object(ibmqexperiment, 'MAX_EXECUTIONS_PER_JOB', 1000):
            with pytest.raises(ValueError, match="Batch 0 would submit.*executions.*exceeding"):
                exp.transpile(self.backend)

    def test_circuit_exceeding_max_two_qubit_gates_raises(self):
        """Circuit whose two-qubit gate count exceeds MAX_TWO_QUBIT_GATES_PER_CIRCUIT raises ValueError."""
        chkpt = 'test_ibmq_circuit_max_two_qubit_gates'
        exp = ibmq.IBMQExperiment(self.edesign, self.pspec, circuits_per_batch=5, num_shots=1024,
                                  seed=20231201, checkpoint_path=chkpt, checkpoint_override=True)

        # Patch num_nonlocal_gates() to return a large count without needing a real huge circuit
        with mock.patch('qiskit.circuit.QuantumCircuit.num_nonlocal_gates') as mock_nonlocal:
            mock_nonlocal.return_value = ibmqexperiment.MAX_TWO_QUBIT_GATES_PER_CIRCUIT + 1
            with pytest.raises(ValueError, match="Circuit has.*two-qubit gates.*exceeding"):
                exp.transpile(self.backend)

    def test_circuit_exceeding_max_rz_gates_raises(self):
        """Circuit whose RZ gate count exceeds MAX_RZ_GATES_PER_CIRCUIT raises ValueError."""
        chkpt = 'test_ibmq_circuit_max_rz_gates'
        exp = ibmq.IBMQExperiment(self.edesign, self.pspec, circuits_per_batch=5, num_shots=1024,
                                  seed=20231201, checkpoint_path=chkpt, checkpoint_override=True)

        # Patch count_ops() to return a large RZ count without needing a real huge circuit
        with mock.patch('qiskit.circuit.QuantumCircuit.count_ops') as mock_count:
            mock_count.return_value = {'rz': ibmqexperiment.MAX_RZ_GATES_PER_CIRCUIT + 1}
            with pytest.raises(ValueError, match="Circuit has.*RZ gates.*exceeding"):
                exp.transpile(self.backend)

    def test_circuit_exceeding_max_sx_gates_raises(self):
        """Circuit whose SX gate count exceeds MAX_SX_GATES_PER_CIRCUIT raises ValueError."""
        chkpt = 'test_ibmq_circuit_max_sx_gates'
        exp = ibmq.IBMQExperiment(self.edesign, self.pspec, circuits_per_batch=5, num_shots=1024,
                                  seed=20231201, checkpoint_path=chkpt, checkpoint_override=True)

        # Patch count_ops() to return a large SX count without needing a real huge circuit
        with mock.patch('qiskit.circuit.QuantumCircuit.count_ops') as mock_count:
            mock_count.return_value = {'sx': ibmqexperiment.MAX_SX_GATES_PER_CIRCUIT + 1}
            with pytest.raises(ValueError, match="Circuit has.*SX gates.*exceeding"):
                exp.transpile(self.backend)

    def test_ignore_batch_limit_checks_bypasses_all_validation(self):
        """ignore_batch_limit_checks=True bypasses all four limit checks."""
        chkpt = 'test_ibmq_ignore_batch_limit_checks'
        exp = ibmq.IBMQExperiment(self.edesign, self.pspec, circuits_per_batch=5, num_shots=1024,
                                  seed=20231201, checkpoint_path=chkpt, checkpoint_override=True)

        # Patch all limits to trigger validation, but pass ignore_batch_limit_checks=True
        with mock.patch.object(ibmqexperiment, 'MAX_EXECUTIONS_PER_JOB', 1000), \
             mock.patch('qiskit.circuit.QuantumCircuit.num_nonlocal_gates') as mock_nonlocal, \
             mock.patch('qiskit.circuit.QuantumCircuit.count_ops') as mock_count:
            # Set up mocks to return huge counts that would trigger all checks
            mock_nonlocal.return_value = ibmqexperiment.MAX_TWO_QUBIT_GATES_PER_CIRCUIT + 1
            mock_count.return_value = {
                'rz': ibmqexperiment.MAX_RZ_GATES_PER_CIRCUIT + 1,
                'sx': ibmqexperiment.MAX_SX_GATES_PER_CIRCUIT + 1,
            }
            # Transpile should succeed when ignore_batch_limit_checks=True
            exp.transpile(self.backend, ignore_batch_limit_checks=True)
            # Verify we actually transpiled something
            self.assertGreater(len(exp.qiskit_isa_circuit_batches), 0)

    def test_auto_batch_size_formula_case_1(self):
        """Auto batch size formula correctness with num_shots=512 and fixed duration."""
        chkpt = 'test_ibmq_auto_batch_size_case_1'
        exp = ibmq.IBMQExperiment(self.edesign, self.pspec, circuits_per_batch="auto",
                                  num_shots=512, seed=20231201,
                                  checkpoint_path=chkpt, checkpoint_override=True)

        # Mock estimate_duration to return a known value (e.g., 0.001 seconds = 1ms per circuit)
        # The backend doesn't have configuration(), so the fallback will be used
        fixed_duration = 0.001
        with mock.patch('qiskit.circuit.QuantumCircuit.estimate_duration') as mock_est_dur:
            mock_est_dur.return_value = fixed_duration
            exp.transpile(self.backend)

            # Manually compute expected value using the formula
            per_circuit_time = fixed_duration + ibmqexperiment.DEFAULT_REP_DELAY_SECONDS
            raw_n = int(ibmqexperiment.TARGET_JOB_DURATION_SECONDS
                        // (512 * per_circuit_time + ibmqexperiment.CIRCUIT_OVERHEAD_SECONDS))
            capped_n = ibmqexperiment.MAX_EXECUTIONS_PER_JOB // 512
            expected = max(1, min(raw_n, capped_n))

            # Verify that circuits_per_batch was resolved to the expected value
            self.assertEqual(exp.circuits_per_batch, expected)

    def test_auto_batch_size_formula_case_2(self):
        """Auto batch size formula correctness with num_shots=1024 and different duration."""
        chkpt = 'test_ibmq_auto_batch_size_case_2'
        exp = ibmq.IBMQExperiment(self.edesign, self.pspec, circuits_per_batch="auto",
                                  num_shots=1024, seed=20231201,
                                  checkpoint_path=chkpt, checkpoint_override=True)

        # Mock estimate_duration to return a different value (0.002 seconds = 2ms per circuit)
        fixed_duration = 0.002
        with mock.patch('qiskit.circuit.QuantumCircuit.estimate_duration') as mock_est_dur:
            mock_est_dur.return_value = fixed_duration
            exp.transpile(self.backend)

            # Manually compute expected value
            per_circuit_time = fixed_duration + ibmqexperiment.DEFAULT_REP_DELAY_SECONDS
            raw_n = int(ibmqexperiment.TARGET_JOB_DURATION_SECONDS
                        // (1024 * per_circuit_time + ibmqexperiment.CIRCUIT_OVERHEAD_SECONDS))
            capped_n = ibmqexperiment.MAX_EXECUTIONS_PER_JOB // 1024
            expected = max(1, min(raw_n, capped_n))

            self.assertEqual(exp.circuits_per_batch, expected)

    def test_auto_batch_size_clipped_by_executions_cap(self):
        """Auto batch size is clipped to MAX_EXECUTIONS_PER_JOB // num_shots when the
        duration-based N would otherwise exceed it."""
        chkpt = 'test_ibmq_auto_batch_size_clipped'
        num_shots = 100_000
        exp = ibmq.IBMQExperiment(self.edesign, self.pspec, circuits_per_batch="auto",
                                  num_shots=num_shots, seed=20231201,
                                  checkpoint_path=chkpt, checkpoint_override=True)

        # DEFAULT_REP_DELAY_SECONDS (250us) alone is too large for the duration-based N to
        # ever exceed the executions cap at any num_shots, so this scenario also mocks a much
        # smaller backend-reported rep_delay to actually drive the duration-based N above the cap.
        fixed_duration = 1e-9
        rep_delay = 1e-8
        mock_config = mock.MagicMock()
        mock_config.default_rep_delay = rep_delay
        with mock.patch('qiskit.circuit.QuantumCircuit.estimate_duration') as mock_est_dur, \
             mock.patch.object(self.backend, 'configuration', return_value=mock_config, create=True):
            mock_est_dur.return_value = fixed_duration
            exp.transpile(self.backend)

        per_circuit_time = fixed_duration + rep_delay
        raw_n = int(ibmqexperiment.TARGET_JOB_DURATION_SECONDS
                    // (num_shots * per_circuit_time + ibmqexperiment.CIRCUIT_OVERHEAD_SECONDS))
        capped_n = ibmqexperiment.MAX_EXECUTIONS_PER_JOB // num_shots
        # Sanity check that this scenario actually exercises the clip, not just that the
        # result happens to satisfy <= the cap.
        self.assertGreater(raw_n, capped_n)
        self.assertEqual(exp.circuits_per_batch, capped_n)

    def test_auto_batch_size_rep_delay_fallback(self):
        """Auto batch size uses DEFAULT_REP_DELAY_SECONDS when backend has no configuration()."""
        chkpt = 'test_ibmq_auto_batch_size_rep_delay_fallback'
        exp = ibmq.IBMQExperiment(self.edesign, self.pspec, circuits_per_batch="auto",
                                  num_shots=512, seed=20231201,
                                  checkpoint_path=chkpt, checkpoint_override=True)

        fixed_duration = 0.001
        with mock.patch('qiskit.circuit.QuantumCircuit.estimate_duration') as mock_est_dur:
            mock_est_dur.return_value = fixed_duration
            exp.transpile(self.backend)

            # Verify that the fallback rep_delay was used
            per_circuit_time = fixed_duration + ibmqexperiment.DEFAULT_REP_DELAY_SECONDS
            raw_n = int(ibmqexperiment.TARGET_JOB_DURATION_SECONDS
                        // (512 * per_circuit_time + ibmqexperiment.CIRCUIT_OVERHEAD_SECONDS))
            capped_n = ibmqexperiment.MAX_EXECUTIONS_PER_JOB // 512
            expected = max(1, min(raw_n, capped_n))

            self.assertEqual(exp.circuits_per_batch, expected)

    def test_auto_batch_size_floor_of_one(self):
        """Auto batch size is at least 1 even with pathologically long circuit times."""
        chkpt = 'test_ibmq_auto_batch_size_floor'
        exp = ibmq.IBMQExperiment(self.edesign, self.pspec, circuits_per_batch="auto",
                                  num_shots=1024, seed=20231201,
                                  checkpoint_path=chkpt, checkpoint_override=True)

        # Mock estimate_duration to return a huge value
        huge_duration = 1000.0  # 1000 seconds per circuit
        with mock.patch('qiskit.circuit.QuantumCircuit.estimate_duration') as mock_est_dur:
            mock_est_dur.return_value = huge_duration
            exp.transpile(self.backend)

            # Even with huge duration, circuits_per_batch should be at least 1
            self.assertGreaterEqual(exp.circuits_per_batch, 1)

    def test_auto_batch_size_provenance_tracking(self):
        """_auto_batch_size attribute correctly tracks whether 'auto' was requested."""
        chkpt_auto = 'test_ibmq_auto_batch_size_provenance_auto'
        chkpt_fixed = 'test_ibmq_auto_batch_size_provenance_fixed'

        # Create with "auto"
        exp_auto = ibmq.IBMQExperiment(self.edesign, self.pspec, circuits_per_batch="auto",
                                       num_shots=512, seed=20231201,
                                       checkpoint_path=chkpt_auto, checkpoint_override=True)
        self.assertTrue(exp_auto._auto_batch_size)

        # Transpile should update circuits_per_batch to int, but _auto_batch_size stays True
        with mock.patch('qiskit.circuit.QuantumCircuit.estimate_duration') as mock_est_dur:
            mock_est_dur.return_value = 0.001
            exp_auto.transpile(self.backend)

        # After transpile, circuits_per_batch should be an int
        self.assertIsInstance(exp_auto.circuits_per_batch, int)
        # But _auto_batch_size should still be True
        self.assertTrue(exp_auto._auto_batch_size)

        # Create with fixed batch size
        exp_fixed = ibmq.IBMQExperiment(self.edesign, self.pspec, circuits_per_batch=5,
                                        num_shots=512, seed=20231201,
                                        checkpoint_path=chkpt_fixed, checkpoint_override=True)
        self.assertFalse(exp_fixed._auto_batch_size)

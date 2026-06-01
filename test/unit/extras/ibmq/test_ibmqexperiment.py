from ...util import BaseCase

import pygsti
from pygsti.extras.devices.experimentaldevice import ExperimentalDevice
from pygsti.extras import ibmq
from pygsti.processors import CliffordCompilationRules as CCR
from pygsti.protocols import MirrorRBDesign as RMCDesign
from pygsti.protocols import PeriodicMirrorCircuitDesign as PMCDesign
from pygsti.protocols import ByDepthSummaryStatistics
from pygsti.modelpacks import smq1Q_XY
from pygsti.protocols import StandardGSTDesign
import numpy as np

try:
    from qiskit.providers.fake_provider import GenericBackendV2
except:
    GenericBackendV2 = None

try:
    from qiskit_ibm_runtime import QiskitRuntimeService
except:
    QiskitRuntimeService = None

import pytest

class IBMQExperimentTester():
    @classmethod
    def setup_class(cls):
        if GenericBackendV2 is None:
            pytest.skip('Qiskit is required for this operation, and does not appear to be installed.')
        elif QiskitRuntimeService is None:
            pytest.skip('Qiskit Runtime is required for this operation, and does not appear to be installed.')
            
        cls.backend = GenericBackendV2(num_qubits=4)
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
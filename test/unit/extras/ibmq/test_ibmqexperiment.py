import pytest
import shutil

try: import qiskit as _qiskit
except: _qiskit = None

import pygsti
from pygsti.extras.devices.experimentaldevice import ExperimentalDevice
from pygsti.extras import ibmq
from pygsti.processors import CliffordCompilationRules as CCR

class IBMQExperimentTester():
    @classmethod
    def setup_class(cls):
        cls.device = ExperimentalDevice.from_legacy_device('ibmq_bogota')
        cls.pspec = cls.device.create_processor_spec(['Gc{}'.format(i) for i in range(24)] + ['Gcnot'])

        compilations = {'absolute': CCR.create_standard(cls.pspec, 'absolute', ('paulis', '1Qcliffords'), verbosity=0)}

        mirror_design = pygsti.protocols.MirrorRBDesign(cls.pspec, [0, 2, 4], 10, qubit_labels=('Q0', 'Q1', 'Q2'),
                                                        clifford_compilations=compilations, sampler='edgegrab', samplerargs=[3/8,])
        cls.edesign = pygsti.protocols.CombinedExperimentDesign([mirror_design])

    
    def test_init(self):
        exp1 = ibmq.IBMQExperiment(self.edesign, self.pspec, circuits_per_batch=5, num_shots=1024, seed=20231201,
                                   disable_checkpointing=True)

        shutil.rmtree('ibmq_init_checkpoint', ignore_errors=True)
        exp2 = ibmq.IBMQExperiment(self.edesign, self.pspec, circuits_per_batch=5, num_shots=1024, seed=20231201,
                                   checkpoint_path='ibmq_init_checkpoint')
        
        assert exp2.pygsti_circuit_batches == exp1.pygsti_circuit_batches

        exp3 = ibmq.IBMQExperiment.from_dir('ibmq_init_checkpoint')
        assert exp3.pygsti_circuit_batches == exp1.pygsti_circuit_batches





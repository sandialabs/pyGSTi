import pytest
import shutil

try: import qiskit
except: qiskit = None

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

        chkpt = 'test_ibmq_init_checkpoint'
        shutil.rmtree(chkpt, ignore_errors=True)
        exp2 = ibmq.IBMQExperiment(self.edesign, self.pspec, circuits_per_batch=5, num_shots=1024, seed=20231201,
                                   checkpoint_path=chkpt)
        
        assert exp2.pygsti_circuit_batches == exp1.pygsti_circuit_batches

        exp3 = ibmq.IBMQExperiment.from_dir(chkpt)
        assert exp3.pygsti_circuit_batches == exp1.pygsti_circuit_batches
    
    def test_transpile(self):
        chkpt = 'test_ibmq_transpile_checkpoint'
        shutil.rmtree(chkpt, ignore_errors=True)
        exp1 = ibmq.IBMQExperiment(self.edesign, self.pspec, circuits_per_batch=5, num_shots=1024, seed=20231201,
                                   checkpoint_path=chkpt)
        exp1.transpile()

        # Test checkpoint load
        exp2 = ibmq.IBMQExperiment.from_dir(chkpt, regen_qiskit_circs=True)
        assert exp2.qiskit_circuit_batches == exp1.qiskit_circuit_batches

        # Test restart
        del exp2.qiskit_circuit_batches[2:]
        del exp2.qasm_circuit_batches[2:]
        exp2.transpile()
        assert exp2.qiskit_circuit_batches == exp1.qiskit_circuit_batches

    @pytest.mark.parametrize('backend_version', ['v1', 'v2'])
    def test_submit(self, backend_version):
        chkpt = 'test_ibmq_submit_checkpoint'
        shutil.rmtree(chkpt, ignore_errors=True)
        exp1 = ibmq.IBMQExperiment(self.edesign, self.pspec, circuits_per_batch=5, num_shots=1024, seed=20231201,
                                   checkpoint_path=chkpt)
        exp1.transpile()

        if backend_version == 'v1':
            from qiskit.providers.fake_provider import FakeBogota, FakeProvider
            backend = FakeBogota()
            provider = FakeProvider()
        elif backend_version == 'v2':
            from qiskit.providers.fake_provider import FakeBogotaV2, FakeProviderForBackendV2
            backend = FakeBogotaV2()
            provider = FakeProviderForBackendV2()
        else:
            raise RuntimeError("Unknown backend version for testing")
        
        exp1.submit(backend, stop=3, max_attempts=1)
        # Submit first 3 jobs
        assert len(exp1.qjobs) == 3

        # Load from checkpoint
        exp2 = ibmq.IBMQExperiment.from_dir(chkpt, regen_qiskit_circs=True, regen_runtime_jobs=True, provider=provider)
        exp2.submit(max_attempts=1)
        assert len(exp2.qjobs) == len(exp2.qasm_circuit_batches)







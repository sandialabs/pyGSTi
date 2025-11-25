from ..util import BaseCase

import numpy as np

from pygsti.circuits import Circuit as C
from pygsti.baseobjs import Label as L
from pygsti.protocols import FreeformDesign
from pygsti.protocols.mirror_edesign import *
from pygsti.models import modelconstruction as mc
from pygsti.data import simulate_data
from pygsti.processors import QubitProcessorSpec

class TestMirrorEDesign(BaseCase):
    def test_circuit_count_and_target_outcome_rc(self):
        line_labels = ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']

        I_args = [0,0,0]
        X_args = [np.pi,0,np.pi]
        Y_args = [np.pi,np.pi/2,np.pi/2]
        Z_args = [0,0,np.pi]

        layers = [
            [L('Gu3', ['Q1'], args=Z_args), L('Gu3', ['Q2'], args=Y_args), L('Gu3', ['Q3'], args=X_args), L('Gu3', ['Q4'], args=I_args), L('Gu3', ['Q5'], args=I_args)],
            [L('Gcnot', ['Q1', 'Q2'], args=None), L('Gcnot', ['Q4', 'Q3'], args=None)],
            [L('Gu3', ['Q1'], args=Y_args), L('Gu3', ['Q2'], args=Z_args), L('Gu3', ['Q3'], args=X_args), L('Gu3', ['Q4'], args=I_args), L('Gu3', ['Q5'], args=X_args)],
            [L('Gcnot', ['Q2', 'Q3'], args=None), L('Gcnot', ['Q4', 'Q5'], args=None)],
            [L('Gcphase', ['Q1', 'Q2'], args=None), L('Gcphase', ['Q3', 'Q4'], args=None)],
            [L('Gcnot', ['Q3', 'Q2'], args=None), L('Gcnot', ['Q4', 'Q5'], args=None)],
            [L('Gu3', ['Q1'], args=X_args), L('Gu3', ['Q2'], args=Y_args), L('Gu3', ['Q3'], args=I_args), L('Gu3', ['Q4'], args=I_args), L('Gu3', ['Q5'], args=Y_args)],
            [L('Gu3', ['Q1'], args=X_args), L('Gu3', ['Q2'], args=Y_args), L('Gu3', ['Q3'], args=I_args), L('Gu3', ['Q4'], args=Y_args), L('Gu3', ['Q5'], args=Y_args)],
            [L('Gu3', ['Q1'], args=Y_args), L('Gu3', ['Q2'], args=X_args), L('Gu3', ['Q3'], args=Z_args), L('Gu3', ['Q4'], args=Y_args), L('Gu3', ['Q5'], args=I_args)],
            [L('Gcphase', ['Q1', 'Q2'], args=None), L('Gcphase', ['Q4', 'Q5'], args=None)]
                    ]
                
        circ = C(layers, line_labels=line_labels)
        circ_dict = {circ: [{'id': 0, 'width': circ.width, 'depth': circ.depth}]}
        test_edesign = FreeformDesign(circ_dict)

        num_mcs_per_circ = 2
        num_ref_per_qubit_subset = 2

        mirror_edesign = make_mirror_edesign(test_edesign=test_edesign, account_for_routing=False,
                                             ref_edesign=None,
                                             num_mcs_per_circ=num_mcs_per_circ,
                                             num_ref_per_qubit_subset=num_ref_per_qubit_subset,
                                             mirroring_strategy='pauli_rc',
                                             gate_set='u3_cx_cz'
                                             )
        
        self.assertEqual(2*num_mcs_per_circ + num_ref_per_qubit_subset, len(mirror_edesign.all_circuits_needing_data))

        flat_test_auxlist = [aux for auxlist in test_edesign.aux_info.values() for aux in auxlist]
        flat_mirror_auxlist = [aux['base_aux'] for auxlist in mirror_edesign['rr'].aux_info.values() for aux in auxlist]
        self.assertTrue(all(flat_mirror_auxlist.count(test_aux) == num_mcs_per_circ for test_aux in flat_test_auxlist))

        flat_mirror_auxlist = [aux['base_aux'] for auxlist in mirror_edesign['br'].aux_info.values() for aux in auxlist]
        self.assertTrue(all(flat_mirror_auxlist.count(test_aux) == num_mcs_per_circ for test_aux in flat_test_auxlist))
        
        pspec = QubitProcessorSpec(5, ['Gu3', 'Gcphase', 'Gcnot', 'Gi'],
                                   availability={'Gcnot':'all-permutations', 'Gcphase': 'all-permutations'},
                                   qubit_labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
        
        ideal_model = mc.create_crosstalk_free_model(pspec)

        for edkey, ed in mirror_edesign.items():
            for circ, auxlist in ed.aux_info.items():
                target_bs = auxlist[0]['idealout']
                result = ideal_model.probabilities(circ)
                self.assertAlmostEqual(result[(target_bs,)], 1.0)


    def test_circuit_count_and_target_outcome_cp(self):
        line_labels = ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']

        I_args = [0,0,0]
        X_args = [np.pi,0,np.pi]
        Y_args = [np.pi,np.pi/2,np.pi/2]
        Z_args = [0,0,np.pi]

        layers = [
            [L('Gu3', ['Q1'], args=Z_args), L('Gu3', ['Q2'], args=Y_args), L('Gu3', ['Q3'], args=X_args), L('Gu3', ['Q4'], args=I_args), L('Gu3', ['Q5'], args=I_args)],
            [L('Gcnot', ['Q1', 'Q2'], args=None), L('Gcnot', ['Q4', 'Q3'], args=None)],
            [L('Gu3', ['Q1'], args=Y_args), L('Gu3', ['Q2'], args=Z_args), L('Gu3', ['Q3'], args=X_args), L('Gu3', ['Q4'], args=I_args), L('Gu3', ['Q5'], args=X_args)],
            [L('Gcnot', ['Q2', 'Q3'], args=None), L('Gcnot', ['Q4', 'Q5'], args=None)],
            [L('Gcphase', ['Q1', 'Q2'], args=None), L('Gcphase', ['Q3', 'Q4'], args=None)],
            [L('Gcnot', ['Q3', 'Q2'], args=None), L('Gcnot', ['Q4', 'Q5'], args=None)],
            [L('Gu3', ['Q1'], args=X_args), L('Gu3', ['Q2'], args=Y_args), L('Gu3', ['Q3'], args=I_args), L('Gu3', ['Q4'], args=I_args), L('Gu3', ['Q5'], args=Y_args)],
            [L('Gu3', ['Q1'], args=X_args), L('Gu3', ['Q2'], args=Y_args), L('Gu3', ['Q3'], args=I_args), L('Gu3', ['Q4'], args=Y_args), L('Gu3', ['Q5'], args=Y_args)],
            [L('Gu3', ['Q1'], args=Y_args), L('Gu3', ['Q2'], args=X_args), L('Gu3', ['Q3'], args=Z_args), L('Gu3', ['Q4'], args=Y_args), L('Gu3', ['Q5'], args=I_args)],
            [L('Gcphase', ['Q1', 'Q2'], args=None), L('Gcphase', ['Q4', 'Q5'], args=None)]
                    ]
                
        circ = C(layers, line_labels=line_labels)
        circ_dict = {circ: [{'id': 0, 'width': circ.width, 'depth': circ.depth}]}
        test_edesign = FreeformDesign(circ_dict)

        num_mcs_per_circ = 4
        num_ref_per_qubit_subset = 10

        mirror_edesign = make_mirror_edesign(test_edesign=test_edesign, account_for_routing=False,
                                             ref_edesign=None,
                                             num_mcs_per_circ=num_mcs_per_circ,
                                             num_ref_per_qubit_subset=num_ref_per_qubit_subset,
                                             mirroring_strategy='central_pauli',
                                             gate_set='u3_cx_cz'
                                             )
        
        self.assertEqual(num_mcs_per_circ + num_ref_per_qubit_subset, len(mirror_edesign.all_circuits_needing_data))

        flat_test_auxlist = [aux for auxlist in test_edesign.aux_info.values() for aux in auxlist]
        flat_mirror_auxlist = [aux['base_aux'] for auxlist in mirror_edesign['cp'].aux_info.values() for aux in auxlist]
        self.assertTrue(all(flat_mirror_auxlist.count(test_aux) == num_mcs_per_circ for test_aux in flat_test_auxlist))
        
        pspec = QubitProcessorSpec(5, ['Gu3', 'Gcphase', 'Gcnot', 'Gi'],
                                   availability={'Gcnot':'all-permutations', 'Gcphase': 'all-permutations'},
                                   qubit_labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
        
        ideal_model = mc.create_crosstalk_free_model(pspec)


        for edkey, ed in mirror_edesign.items():
            for circ, auxlist in ed.aux_info.items():
                target_bs = auxlist[0]['idealout']
                result = ideal_model.probabilities(circ)
                self.assertAlmostEqual(result[(target_bs,)], 1.0)


class TestScalableBenchmarks(BaseCase):
    def test_noise_mirror_benchmark(self):
        try:
            import qiskit
        except:
            self.skipTest('Qiskit is required for this operation, and does not appear to be installed.')

        try:
            import qiskit_ibm_runtime
        except:
            self.skipTest('Qiskit Runtime is required for this operation, and does not appear to be installed.')

        backend = qiskit_ibm_runtime.fake_provider.FakeFez()

        num_mcs_per_circ = 5
        num_ref_per_qubit_subset = 7
        mirroring_kwargs_dict = {'num_mcs_per_circ': num_mcs_per_circ,
                                 'num_ref_per_qubit_subset': num_ref_per_qubit_subset}

        qk_circ = qiskit.circuit.library.QFT(6)
        qk_circ = qiskit.transpile(qk_circ, backend=backend)

        test_edesign, mirror_edesign = qiskit_circuits_to_mirror_edesign([qk_circ],
                                                                         mirroring_kwargs_dict=mirroring_kwargs_dict)
        
        self.assertEqual(2*num_mcs_per_circ + num_ref_per_qubit_subset, len(mirror_edesign.all_circuits_needing_data))
        
        flat_test_auxlist = [aux for auxlist in test_edesign.aux_info.values() for aux in auxlist]
        flat_mirror_auxlist = [aux['base_aux'] for auxlist in mirror_edesign['br'].aux_info.values() for aux in auxlist]
        self.assertTrue(all(flat_mirror_auxlist.count(test_aux) == num_mcs_per_circ for test_aux in flat_test_auxlist))

        flat_mirror_auxlist = [aux['base_aux'] for auxlist in mirror_edesign['rr'].aux_info.values() for aux in auxlist]
        self.assertTrue(all(flat_mirror_auxlist.count(test_aux) == num_mcs_per_circ for test_aux in flat_test_auxlist))

        test_circ = test_edesign.all_circuits_needing_data[0]

        pspec = QubitProcessorSpec(test_circ.width, ['Gu3','Gxpi2','Gxpi','Gzr','Gi','Gcphase'],
                                   availability={'Gcnot':'all-permutations','Gcphase':'all-permutations'},
                                   qubit_labels=test_circ.line_labels)
        
        ideal_model = mc.create_crosstalk_free_model(pspec)

        for edkey, ed in mirror_edesign.items():
            for circ, auxlist in ed.aux_info.items():
                target_bs = auxlist[0]['idealout']
                result = ideal_model.probabilities(circ)

                ######
                # qk_circ = circ.convert_to_qiskit(qubit_conversion={q:i for i,q in enumerate(circ.line_labels)},
                #                                  block_between_layers=True)
                
                # statevec = qiskit.quantum_info.Statevector.from_instruction(qk_circ)
                # probs = statevec.probabilities_dict()
                # probs = {k[::-1]: v for k,v in probs.items()} # Qiskit endianness

                self.assertAlmostEqual(result[(target_bs,)], 1.0)



    def test_fullstack_mirror_benchmark(self):
        try:
            import qiskit
        except:
            self.skipTest('Qiskit is required for this operation, and does not appear to be installed.')

        try:
            import qiskit_ibm_runtime
        except:
            self.skipTest('Qiskit Runtime is required for this operation, and does not appear to be installed.')

        backend = qiskit_ibm_runtime.fake_provider.FakeFez()

        num_mcs_per_circ = 3
        num_ref_per_qubit_subset = 4

        mirroring_kwargs_dict = {'num_mcs_per_circ': num_mcs_per_circ,
                                 'num_ref_per_qubit_subset': num_ref_per_qubit_subset}
        transpiler_kwargs_dict = {'optimization_level': 2}

        qk_circ = qiskit.circuit.library.QFT(6)

        test_edesign, mirror_edesign = qiskit_circuits_to_fullstack_mirror_edesign([qk_circ],
                                                                                   qk_backend=backend,
                                                                                   transpiler_kwargs_dict=transpiler_kwargs_dict,
                                                                         mirroring_kwargs_dict=mirroring_kwargs_dict)
        
        self.assertEqual(2*num_mcs_per_circ + num_ref_per_qubit_subset, len(mirror_edesign.all_circuits_needing_data))
        
        flat_test_auxlist = [aux for auxlist in test_edesign.aux_info.values() for aux in auxlist]
        flat_mirror_auxlist = [aux['base_aux'] for auxlist in mirror_edesign['br'].aux_info.values() for aux in auxlist]
        self.assertTrue(all(flat_mirror_auxlist.count(test_aux) == num_mcs_per_circ for test_aux in flat_test_auxlist))

        flat_mirror_auxlist = [aux['base_aux'] for auxlist in mirror_edesign['rr'].aux_info.values() for aux in auxlist]
        self.assertTrue(all(flat_mirror_auxlist.count(test_aux) == num_mcs_per_circ for test_aux in flat_test_auxlist))

        test_circ = test_edesign.all_circuits_needing_data[0]

        pspec = QubitProcessorSpec(test_circ.width, ['Gu3','Gxpi2','Gxpi','Gzr','Gi','Gcphase'],
                                   availability={'Gcnot':'all-permutations','Gcphase':'all-permutations'},
                                   qubit_labels=test_circ.line_labels)
        
        ideal_model = mc.create_crosstalk_free_model(pspec)

        for edkey, ed in mirror_edesign.items():
            for circ, auxlist in ed.aux_info.items():
                target_bs = auxlist[0]['idealout']
                result = ideal_model.probabilities(circ)

                ######
                # qk_circ = circ.convert_to_qiskit(qubit_conversion={q:i for i,q in enumerate(circ.line_labels)},
                #                                  block_between_layers=True)
                
                # statevec = qiskit.quantum_info.Statevector.from_instruction(qk_circ)
                # probs = statevec.probabilities_dict()
                # probs = {k[::-1]: v for k,v in probs.items()} # Qiskit endianness

                self.assertAlmostEqual(result[(target_bs,)], 1.0)

    def test_subcircuit_mirror_benchmark(self):
        try:
            import qiskit
        except:
            self.skipTest('Qiskit is required for this operation, and does not appear to be installed.')

        try:
            import qiskit_ibm_runtime
        except:
            self.skipTest('Qiskit Runtime is required for this operation, and does not appear to be installed.')

        from pygsti.baseobjs.unitarygatefunction import UnitaryGateFunction

        class Gdelay(UnitaryGateFunction):
            shape = (2, 2)
            def __call__(self, dt):
                return np.eye(2)
            

        backend = qiskit_ibm_runtime.fake_provider.FakeFez()

        width_depths = {2: [2,4,6],
                        4: [4,8,10]}
        
        num_samples_per_width_depth = 3

        subcirc_kwargs_dict = {'num_samples_per_width_depth': num_samples_per_width_depth}

        num_mcs_per_circ = 3
        num_ref_per_qubit_subset = 4

        mirroring_kwargs_dict = {'num_mcs_per_circ': num_mcs_per_circ,
                                 'num_ref_per_qubit_subset': num_ref_per_qubit_subset}
        
        


        qk_circ = qiskit.circuit.library.QFT(6)
        qk_circ = qiskit.transpile(qk_circ, backend=backend)

        test_edesign, mirror_edesign = qiskit_circuits_to_subcircuit_mirror_edesign(
                                                        [qk_circ],
                                                        aggregate_subcircs=True,
                                                        width_depth_dict=width_depths,
                                                        coupling_map=backend.coupling_map,
                                                        instruction_durations=backend.instruction_durations,
                                                        subcirc_kwargs_dict=subcirc_kwargs_dict,
                                                        mirroring_kwargs_dict=mirroring_kwargs_dict
                                                        )
        
        
        flat_test_auxlist = [aux for auxlist in test_edesign.aux_info.values() for aux in auxlist]
        flat_mirror_auxlist = [aux['base_aux'] for auxlist in mirror_edesign['br'].aux_info.values() for aux in auxlist]
        self.assertTrue(all(flat_mirror_auxlist.count(test_aux) == num_mcs_per_circ for test_aux in flat_test_auxlist))

        flat_mirror_auxlist = [aux['base_aux'] for auxlist in mirror_edesign['rr'].aux_info.values() for aux in auxlist]
        self.assertTrue(all(flat_mirror_auxlist.count(test_aux) == num_mcs_per_circ for test_aux in flat_test_auxlist))

        test_circ = test_edesign.all_circuits_needing_data[0]

        pspec = QubitProcessorSpec(test_circ.width, ['Gu3','Gxpi2','Gxpi','Gzr','Gi','Gcphase','Gdelay'],
                                   nonstd_gate_unitaries={'Gdelay': Gdelay()},
                                   availability={'Gcnot':'all-permutations','Gcphase':'all-permutations'},
                                   qubit_labels=test_circ.line_labels)
        
        ideal_model = mc.create_crosstalk_free_model(pspec)

        for edkey, ed in mirror_edesign.items():
            for circ, auxlist in ed.aux_info.items():

                pspec = QubitProcessorSpec(circ.width, ['Gu3','Gxpi2','Gxpi','Gzr','Gi','Gcphase','Gdelay'],
                                   nonstd_gate_unitaries={'Gdelay': Gdelay()},
                                   availability={'Gcnot':'all-permutations','Gcphase':'all-permutations'},
                                   qubit_labels=circ.line_labels)    
                
                ideal_model = mc.create_crosstalk_free_model(pspec)

                target_bs = auxlist[0]['idealout']
                result = ideal_model.probabilities(circ)

                ######
                # qk_circ = circ.convert_to_qiskit(qubit_conversion={q:i for i,q in enumerate(circ.line_labels)},
                #                                  block_between_layers=True)
                
                # statevec = qiskit.quantum_info.Statevector.from_instruction(qk_circ)
                # probs = statevec.probabilities_dict()
                # probs = {k[::-1]: v for k,v in probs.items()} # Qiskit endianness

                self.assertAlmostEqual(result[(target_bs,)], 1.0)

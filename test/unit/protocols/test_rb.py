from ..util import BaseCase

import pygsti
from pygsti.protocols import rb as _rb


class TestCliffordRBDesign(BaseCase):

    def setUp(self):
        self.num_qubits = 4
        self.qubit_labels = ['Q'+str(i) for i in range(self.num_qubits)]
        
        gate_names = ['Gxpi2', 'Gxmpi2', 'Gypi2', 'Gympi2', 'Gcphase']
        availability = {'Gcphase':[('Q'+str(i),'Q'+str((i+1) % self.num_qubits)) for i in range(self.num_qubits)]}

        self.pspec = pygsti.baseobjs.ProcessorSpec(self.num_qubits, gate_names, availability=availability,
                                                   qubit_labels=self.qubit_labels, construct_models=('clifford',))

        # TODO: Test a lot of these, currently just the default from the tutorial
        self.depths = [0, 1, 2]#, 4, 8]
        self.circuits_per_depth = 5
        self.qubits = ['Q0', 'Q1']
        self.citerations = 20
        self.randomizeout = True
        self.interleaved_circuit = None
        self.compiler_args = ()
        self.seed = 2021
        self.verbosity = 1

    def test_design_construction(self):
        num_mp_procs = 4
        
        serial_design = _rb.CliffordRBDesign(
            self.pspec, self.depths, self.circuits_per_depth, qubit_labels=self.qubits,
            randomizeout=self.randomizeout, interleaved_circuit=self.interleaved_circuit,
            citerations=self.citerations, compilerargs=self.compiler_args, seed=self.seed,
            verbosity=self.verbosity, num_processes=1)
        
        # Test parallel circuit generation works and is seeded properly
        mp_design = _rb.CliffordRBDesign(
            self.pspec, self.depths, self.circuits_per_depth, qubit_labels=self.qubits,
            randomizeout=self.randomizeout, interleaved_circuit=self.interleaved_circuit,
            citerations=self.citerations, compilerargs=self.compiler_args, seed=self.seed,
            verbosity=self.verbosity, num_processes=num_mp_procs)

        # for sd_circ, md_circ in zip(serial_design.all_circuits_needing_data, mp_design.all_circuits_needing_data):
        #     if str(sd_circ) != str(md_circ):
        #         print('Mismatch found!')
        #         print('  Serial circuit:   ' + str(sd_circ))
        #         print('  Parallel circuit: ' + str(md_circ))

        self.assertTrue(all([str(sd) == str(md) for sd, md in zip(serial_design.all_circuits_needing_data,
                                                        mp_design.all_circuits_needing_data)]))


class TestDirectRBDesign(BaseCase):

    def setUp(self):
        self.num_qubits = 4
        self.qubit_labels = ['Q'+str(i) for i in range(self.num_qubits)]
        
        gate_names = ['Gxpi2', 'Gxmpi2', 'Gypi2', 'Gympi2', 'Gcphase']
        availability = {'Gcphase':[('Q'+str(i),'Q'+str((i+1) % self.num_qubits)) for i in range(self.num_qubits)]}

        self.pspec = pygsti.baseobjs.ProcessorSpec(self.num_qubits, gate_names, availability=availability,
                                                   qubit_labels=self.qubit_labels, construct_models=('clifford',))

        # TODO: Test a lot of these, currently just the default from the tutorial
        self.depths = [0, 1, 2]#, 4, 8]
        self.circuits_per_depth = 5
        self.qubits = ['Q0', 'Q1']
        self.randomizeout = True
        self.sampler = 'edgegrab'
        self.samplerargs = [0.5]
        self.citerations = 20
        self.compiler_args = ()
        self.seed = 2021
        self.verbosity = 1

    def test_design_construction(self):
        num_mp_procs = 4
        
        serial_design = _rb.DirectRBDesign(self.pspec, self.depths, self.circuits_per_depth,
            qubit_labels=self.qubits, sampler=self.sampler, samplerargs=self.samplerargs,
            addlocal=False, lsargs=(), randomizeout=self.randomizeout, cliffordtwirl=True,
            conditionaltwirl=True, citerations=self.citerations, compilerargs=self.compiler_args,
            partitioned=False, seed=self.seed, verbosity=self.verbosity, num_processes=1)
        
        # Test parallel circuit generation works and is seeded properly
        mp_design = _rb.DirectRBDesign(self.pspec, self.depths, self.circuits_per_depth,
            qubit_labels=self.qubits, sampler=self.sampler, samplerargs=self.samplerargs,
            addlocal=False, lsargs=(), randomizeout=self.randomizeout, cliffordtwirl=True,
            conditionaltwirl=True, citerations=self.citerations, compilerargs=self.compiler_args,
            partitioned=False, seed=self.seed, verbosity=self.verbosity, num_processes=num_mp_procs)
        
        # for sd_circ, md_circ in zip(serial_design.all_circuits_needing_data, mp_design.all_circuits_needing_data):
        #     if str(sd_circ) != str(md_circ): print('MISMATCH!')
        #     print('  Serial circuit:   ' + str(sd_circ))
        #     print('  Parallel circuit: ' + str(md_circ))
        #     print()
            
        self.assertTrue(all([str(sd) == str(md) for sd, md in zip(serial_design.all_circuits_needing_data,
                                                        mp_design.all_circuits_needing_data)]))

class TestMirrorRBDesign(BaseCase):

    def setUp(self):
        self.num_qubits = 4
        self.qubit_labels = ['Q'+str(i) for i in range(self.num_qubits)]
        
        gate_names = ['Gi', 'Gxpi2', 'Gxpi', 'Gxmpi2', 'Gypi2', 'Gypi', 'Gympi2', 'Gzpi2', 'Gzpi', 'Gzmpi2', 'Gcphase'] 
        availability = {'Gcphase':[('Q'+str(i),'Q'+str((i+1) % self.num_qubits)) for i in range(self.num_qubits)]}

        self.pspec = pygsti.baseobjs.ProcessorSpec(self.num_qubits, gate_names, availability=availability,
                                                   construct_clifford_compilations={'absolute': ('paulis', '1Qcliffords')},  # SS: I think this is for speed, don't need paulieq for MirrorRB?
                                                   qubit_labels=self.qubit_labels, construct_models=('clifford',))

        # TODO: Test a lot of these, currently just the default from the tutorial
        self.depths = [0, 2, 4]#, 8, 16]
        self.circuits_per_depth = 5
        self.qubits = ['Q0', 'Q1']
        self.sampler = 'edgegrab'
        self.samplerargs = [0.5]
        self.seed = 2021
        self.verbosity = 1

    def test_design_construction(self):
        num_mp_procs = 4
        
        serial_design = _rb.MirrorRBDesign(self.pspec, self.depths, self.circuits_per_depth,
            qubit_labels=self.qubits, sampler=self.sampler, samplerargs=self.samplerargs,
            localclifford=True, paulirandomize=True, seed=self.seed, verbosity=self.verbosity,
            num_processes=1)
        
        # Test parallel circuit generation works and is seeded properly
        mp_design = _rb.MirrorRBDesign(self.pspec, self.depths, self.circuits_per_depth,
            qubit_labels=self.qubits, sampler=self.sampler, samplerargs=self.samplerargs,
            localclifford=True, paulirandomize=True, seed=self.seed, verbosity=self.verbosity,
            num_processes=num_mp_procs)

        # for sd_circ, md_circ in zip(serial_design.all_circuits_needing_data, mp_design.all_circuits_needing_data):
        #     if str(sd_circ) != str(md_circ): print('MISMATCH!')
        #     print('  Serial circuit:   ' + str(sd_circ))
        #     print('  Parallel circuit: ' + str(md_circ))
        #     print()
            
        self.assertTrue(all([str(sd) == str(md) for sd, md in zip(serial_design.all_circuits_needing_data,
                                                        mp_design.all_circuits_needing_data)]))

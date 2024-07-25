from ..util import BaseCase

import numpy as _np

import pygsti
from pygsti.protocols import rb as _rb
from pygsti.processors import CliffordCompilationRules as CCR
from pygsti.processors import QubitProcessorSpec as QPS
from pygsti.circuits import Circuit
from pygsti.baseobjs import Label

class TestCliffordRBDesign(BaseCase):

    def setUp(self):
        self.num_qubits = 2
        self.qubit_labels = ['Q'+str(i) for i in range(self.num_qubits)]
        
        gate_names = ['Gi', 'Gxpi2', 'Gxmpi2', 'Gypi2', 'Gympi2', 'Gcphase']
        availability = {'Gcphase':[('Q'+str(i),'Q'+str((i+1) % self.num_qubits)) for i in range(self.num_qubits)]}

        self.pspec = pygsti.processors.QubitProcessorSpec(self.num_qubits, gate_names, availability=availability,
                                                          qubit_labels=self.qubit_labels)
        self.compilations = {
            'absolute': CCR.create_standard(self.pspec, 'absolute', ('paulis', '1Qcliffords'), verbosity=0),
            'paulieq': CCR.create_standard(self.pspec, 'paulieq', ('1Qcliffords', 'allcnots'), verbosity=0)
        }

        gate_names_1Q = gate_names[:-1]
        self.qubit_labels1Q = ['Q0']
        self.pspec1Q = pygsti.processors.QubitProcessorSpec(1, gate_names_1Q, qubit_labels=self.qubit_labels1Q)
        self.compilations1Q = {
            'absolute': CCR.create_standard(self.pspec1Q, 'absolute', ('paulis', '1Qcliffords'), verbosity=0),
            'paulieq': CCR.create_standard(self.pspec1Q, 'paulieq', ('1Qcliffords', 'allcnots'), verbosity=0)
        }

        # TODO: Test a lot of these, currently just the default from the tutorial
        # Probably as pytest mark parameterize for randomizeout, compilerargs?
        self.depths = [0, 2]#, 4, 8]
        self.circuits_per_depth = 5
        self.qubits = ['Q0', 'Q1']
        self.citerations = 20
        self.randomizeout = True
        self.interleaved_circuit = None
        self.compiler_args = ()
        self.seed = 2021
        self.verbosity = 0

    def test_parallel_design_construction(self):
        num_mp_procs = 4
        crb_design = _rb.CliffordRBDesign(
            self.pspec, self.compilations, self.depths, self.circuits_per_depth, qubit_labels=self.qubits,
            randomizeout=self.randomizeout, interleaved_circuit=self.interleaved_circuit,
            citerations=self.citerations, compilerargs=self.compiler_args, seed=self.seed,
            verbosity=self.verbosity, num_processes=1)


        # Test parallel circuit generation works and is seeded properly
        mp_design = _rb.CliffordRBDesign(
            self.pspec, self.compilations, self.depths, self.circuits_per_depth, qubit_labels=self.qubits,
            randomizeout=self.randomizeout, interleaved_circuit=self.interleaved_circuit,
            citerations=self.citerations, compilerargs=self.compiler_args, seed=self.seed,
            verbosity=self.verbosity, num_processes=num_mp_procs)


        self.assertTrue(all([str(sd) == str(md) for sd, md in zip(crb_design.all_circuits_needing_data,
                                                        mp_design.all_circuits_needing_data)]))

        tmodel = pygsti.models.create_crosstalk_free_model(self.pspec)

        [[self.assertAlmostEqual(c.simulate(tmodel)[bs],1.) for c, bs in zip(cl, bsl)] for cl, bsl in zip(mp_design.circuit_lists, mp_design.idealout_lists)]

    def test_deterministic_compilation(self):        
        # TODO: Figure out good test for this. Full circuit is a synthetic idle, we need to somehow check the non-inverted
        # Clifford is the same as the random case?
        abs_design = _rb.CliffordRBDesign(
            self.pspec1Q, self.compilations1Q, self.depths, self.circuits_per_depth, qubit_labels=self.qubit_labels1Q,
            randomizeout=self.randomizeout, interleaved_circuit=self.interleaved_circuit,
            citerations=self.citerations, compilerargs=self.compiler_args, seed=self.seed,
            verbosity=self.verbosity, exact_compilation_key='absolute')
        
        peq_design = _rb.CliffordRBDesign(
            self.pspec1Q, self.compilations1Q, self.depths, self.circuits_per_depth, qubit_labels=self.qubit_labels1Q,
            randomizeout=self.randomizeout, interleaved_circuit=self.interleaved_circuit,
            citerations=self.citerations, compilerargs=self.compiler_args, seed=self.seed,
            verbosity=self.verbosity, exact_compilation_key='paulieq')

        # Testing a non-standard (but unrealistic) compilation
        rule_dict = {f'C{i}': (_np.eye(2), pygsti.circuits.Circuit([], (0,))) for i in range(24)}
        compilations = self.compilations1Q.copy()
        compilations["idle"] = pygsti.processors.CompilationRules(rule_dict)
        idle_design = _rb.CliffordRBDesign(
            self.pspec1Q, compilations, self.depths, self.circuits_per_depth, qubit_labels=self.qubit_labels1Q,
            randomizeout=False, interleaved_circuit=self.interleaved_circuit,
            citerations=self.citerations, compilerargs=self.compiler_args, seed=self.seed,
            verbosity=self.verbosity, exact_compilation_key='idle')

        # All circuits should be the empty circuit (since we've turned off randomizeout)
        for clist in idle_design.circuit_lists:
            self.assertTrue(set(clist) == set([pygsti.circuits.Circuit([], self.qubit_labels1Q)]))

        # Also a handy place to test native gate counts since it should be 0
        avg_gate_counts = idle_design.average_native_gates_per_clifford()
        for v in avg_gate_counts.values():
            self.assertTrue(v == 0)

    def test_serialization(self):

        crb_design = _rb.CliffordRBDesign(
            self.pspec, self.compilations, self.depths, self.circuits_per_depth, qubit_labels=self.qubits,
            randomizeout=self.randomizeout, interleaved_circuit=Circuit([Label('Gxpi2', 'Q0')], line_labels=('Q0','Q1')),
            citerations=self.citerations, compilerargs=self.compiler_args, seed=self.seed,
            verbosity=self.verbosity, num_processes=1)

        crb_design.write('../../test_packages/temp_test_files/test_CliffordRBDesign_serialization')
        #then read this back in
        crb_design_read = _rb.CliffordRBDesign.from_dir('../../test_packages/temp_test_files/test_CliffordRBDesign_serialization')

        self.assertEqual(crb_design.all_circuits_needing_data, crb_design_read.all_circuits_needing_data)
        self.assertEqual(crb_design.interleaved_circuit, crb_design_read.interleaved_circuit)

class TestInterleavedRBDesign(BaseCase):

    def setUp(self):
        self.num_qubits = 2
        self.qubit_labels = ['Q'+str(i) for i in range(self.num_qubits)]
        
        gate_names = ['Gi', 'Gxpi2', 'Gxmpi2', 'Gypi2', 'Gympi2', 'Gcphase']
        availability = {'Gcphase':[('Q'+str(i),'Q'+str((i+1) % self.num_qubits)) for i in range(self.num_qubits)]}

        self.pspec = pygsti.processors.QubitProcessorSpec(self.num_qubits, gate_names, availability=availability,
                                                          qubit_labels=self.qubit_labels)
        self.compilations = {
            'absolute': CCR.create_standard(self.pspec, 'absolute', ('paulis', '1Qcliffords'), verbosity=0),
            'paulieq': CCR.create_standard(self.pspec, 'paulieq', ('1Qcliffords', 'allcnots'), verbosity=0)
        }

        # TODO: Test a lot of these, currently just the default from the tutorial
        # Probably as pytest mark parameterize for randomizeout, compilerargs?
        self.depths = [0, 2]
        self.circuits_per_depth = 5
        self.qubits = ['Q0', 'Q1']
        self.citerations = 20
        self.randomizeout = False
        self.interleaved_circuit = Circuit([Label('Gxpi2', 'Q0')], line_labels=('Q0','Q1'))
        self.compiler_args = ()
        self.seed = 2021
        self.verbosity = 0

        self.irb_design = _rb.InterleavedRBDesign(
            self.pspec, self.compilations, self.depths, self.circuits_per_depth, self.interleaved_circuit, qubit_labels=self.qubits,
            randomizeout=self.randomizeout,
            citerations=self.citerations, compilerargs=self.compiler_args, seed=self.seed,
            verbosity=self.verbosity, num_processes=1)

    def test_combined_design_access(self):
        assert(isinstance(self.irb_design['crb'], _rb.CliffordRBDesign))
        assert(isinstance(self.irb_design['icrb'], _rb.CliffordRBDesign))
        
        self.assertEqual(set(self.irb_design.all_circuits_needing_data), 
                         set(self.irb_design['crb'].all_circuits_needing_data)|  set(self.irb_design['icrb'].all_circuits_needing_data))
    
        self.assertEqual(self.irb_design['icrb'].interleaved_circuit, self.interleaved_circuit)

    def test_serialization(self):

        self.irb_design.write('../../test_packages/temp_test_files/test_InterleavedRBDesign_serialization')
        #then read this back in
        irb_design_read = _rb.InterleavedRBDesign.from_dir('../../test_packages/temp_test_files/test_InterleavedRBDesign_serialization')

        self.assertEqual(self.irb_design.all_circuits_needing_data, irb_design_read.all_circuits_needing_data)
        self.assertEqual(self.irb_design['crb'].all_circuits_needing_data, irb_design_read['crb'].all_circuits_needing_data)
        self.assertEqual(self.irb_design['icrb'].all_circuits_needing_data, irb_design_read['icrb'].all_circuits_needing_data)
        self.assertEqual(self.irb_design.interleaved_circuit, irb_design_read.interleaved_circuit)

class TestDirectRBDesign(BaseCase):

    def setUp(self):
        self.num_qubits = 2
        self.qubit_labels = ['Q'+str(i) for i in range(self.num_qubits)]
        
        gate_names = ['Gxpi2', 'Gxmpi2', 'Gypi2', 'Gympi2', 'Gcphase']
        availability = {'Gcphase':[('Q'+str(i),'Q'+str((i+1) % self.num_qubits)) for i in range(self.num_qubits)]}

        self.pspec = pygsti.processors.QubitProcessorSpec(self.num_qubits, gate_names, availability=availability,
                                                          qubit_labels=self.qubit_labels, geometry='line')
        self.compilations = {
            'absolute': CCR.create_standard(self.pspec, 'absolute', ('paulis', '1Qcliffords'), verbosity=0),
            'paulieq': CCR.create_standard(self.pspec, 'paulieq', ('1Qcliffords', 'allcnots'), verbosity=0)
        }


        # TODO: Test a lot of these, currently just the default from the tutorial
        self.depths = [0, 2]#, 4, 8]
        self.circuits_per_depth = 5
        self.qubits = ['Q0', 'Q1']
        self.randomizeout = True
        self.sampler = 'edgegrab'
        self.samplerargs = [0.5]
        self.citerations = 20
        self.compiler_args = ()
        self.seed = 2021
        self.verbosity = 0

    def test_design_construction(self):
        num_mp_procs = 4
        
        serial_design = _rb.DirectRBDesign(self.pspec, self.compilations, self.depths, self.circuits_per_depth,
            qubit_labels=self.qubits, sampler=self.sampler, samplerargs=self.samplerargs,
            addlocal=False, lsargs=(), randomizeout=self.randomizeout, cliffordtwirl=True,
            conditionaltwirl=True, citerations=self.citerations, compilerargs=self.compiler_args,
            partitioned=False, seed=self.seed, verbosity=self.verbosity, num_processes=1)
        
        # Test parallel circuit generation works and is seeded properly
        mp_design = _rb.DirectRBDesign(self.pspec, self.compilations, self.depths, self.circuits_per_depth,
            qubit_labels=self.qubits, sampler=self.sampler, samplerargs=self.samplerargs,
            addlocal=False, lsargs=(), randomizeout=self.randomizeout, cliffordtwirl=True,
            conditionaltwirl=True, citerations=self.citerations, compilerargs=self.compiler_args,
            partitioned=False, seed=self.seed, verbosity=self.verbosity, num_processes=num_mp_procs)

        
        tmodel = pygsti.models.create_crosstalk_free_model(self.pspec)

        [[self.assertAlmostEqual(c.simulate(tmodel)[bs],1.) for c, bs in zip(cl, bsl)] for cl, bsl in zip(mp_design.circuit_lists, mp_design.idealout_lists)]

        #Print more debugging info since this test can fail randomly but we can't reproduce this.
        unequal_circuits = []
        for i, (sd, md) in enumerate(zip(serial_design.all_circuits_needing_data,
                                         mp_design.all_circuits_needing_data)):
            if str(sd) != str(md):
                unequal_circuits.append((i, sd, md))
        if len(unequal_circuits) > 0:
            print("%d unequal circuits!!" % len(unequal_circuits))
            print("Seed = ",self.seed, " depths=", self.depths, " circuits_per_depth=", self.circuits_per_depth)
            for i, sd, md in unequal_circuits:
                print("Index: ", i)
                print("Serial design: ", sd.str)
                print("Parall design: ", md.str)
                print()

        self.assertTrue(all([str(sd) == str(md) for sd, md in zip(serial_design.all_circuits_needing_data,
                                                                  mp_design.all_circuits_needing_data)]))
        
    def test_serialization(self):

        drb_design = _rb.DirectRBDesign(self.pspec, self.compilations, self.depths, self.circuits_per_depth,
            qubit_labels=self.qubits, sampler=self.sampler, samplerargs=self.samplerargs,
            addlocal=False, lsargs=(), randomizeout=self.randomizeout, cliffordtwirl=True,
            conditionaltwirl=True, citerations=self.citerations, compilerargs=self.compiler_args,
            partitioned=False, seed=self.seed, verbosity=self.verbosity, num_processes=1)

        drb_design.write('../../test_packages/temp_test_files/test_DirectRBDesign_serialization')
        #then read this back in
        drb_design_read = _rb.DirectRBDesign.from_dir('../../test_packages/temp_test_files/test_DirectRBDesign_serialization')

        self.assertEqual(drb_design.all_circuits_needing_data, drb_design_read.all_circuits_needing_data)

class TestMirrorRBDesign(BaseCase):

    def setUp(self):
        self.num_qubits = 2
        self.qubit_labels = ['Q'+str(i) for i in range(self.num_qubits)]

        gate_names = ['Gi', 'Gxpi2', 'Gxpi', 'Gxmpi2', 'Gypi2', 'Gypi', 'Gympi2', 'Gzpi2', 'Gzpi', 'Gzmpi2', 'Gcphase'] 
        availability = {'Gcphase':[('Q'+str(i),'Q'+str((i+1) % self.num_qubits)) for i in range(self.num_qubits)]}

        self.pspec = pygsti.processors.QubitProcessorSpec(self.num_qubits, gate_names, availability=availability,
                                                          qubit_labels=self.qubit_labels, geometry='line')
        self.clifford_compilations = {
            'absolute': CCR.create_standard(self.pspec, 'absolute', ('paulis', '1Qcliffords'), verbosity=0)
            # SS: I think this is for speed, don't need paulieq for MirrorRB?
        }

        # TODO: Test a lot of these, currently just the default from the tutorial
        self.depths = [0, 2]
        self.circuits_per_depth = 5
        self.qubits = ['Q0', 'Q1']
        self.circuit_type = 'clifford'
        self.sampler = 'edgegrab'
        self.samplerargs = [0.5]
        self.seed = 2021
        self.verbosity = 0

    def test_design_construction(self):
        num_mp_procs = 4
        
        serial_design = _rb.MirrorRBDesign(self.pspec, self.depths, self.circuits_per_depth,
            qubit_labels=self.qubits, circuit_type=self.circuit_type, clifford_compilations=self.clifford_compilations,
            sampler=self.sampler, samplerargs=self.samplerargs,
            localclifford=True, paulirandomize=True, seed=self.seed, verbosity=self.verbosity,
            num_processes=1)
        
        # Test parallel circuit generation works and is seeded properly
        mp_design = _rb.MirrorRBDesign(self.pspec, self.depths, self.circuits_per_depth,
            qubit_labels=self.qubits, circuit_type=self.circuit_type, clifford_compilations=self.clifford_compilations,
            sampler=self.sampler, samplerargs=self.samplerargs,
            localclifford=True, paulirandomize=True, seed=self.seed, verbosity=self.verbosity,
            num_processes=num_mp_procs)
            
        self.assertTrue(all([str(sd) == str(md) for sd, md in zip(serial_design.all_circuits_needing_data,
                                                        mp_design.all_circuits_needing_data)]))


    def test_clifford_design_construction(self):

        n = 2
        qs = ['Q'+str(i) for i in range(n)]
        ring = [('Q'+str(i),'Q'+str(i+1)) for i in range(n-1)]

        gateset1 = ['Gcphase'] + ['Gc'+str(i) for i in range(24)]
        pspec1 = QPS(n, gateset1, availability={'Gcphase':ring}, qubit_labels=qs)
        tmodel1 = pygsti.models.create_crosstalk_free_model(pspec1)

        depths = [0, 2, 8]
        q_set = ('Q0', 'Q1')

        clifford_compilations = {'absolute': CCR.create_standard(pspec1, 'absolute', ('paulis', '1Qcliffords'), verbosity=0)}

        design1 = _rb.MirrorRBDesign(pspec1, depths, 3, qubit_labels=q_set, circuit_type='clifford',
                                        clifford_compilations=clifford_compilations, sampler='edgegrab', samplerargs=(0.25,),
                                        localclifford=True, paulirandomize=True, descriptor='A mirror RB experiment',
                                        add_default_protocol=False, seed=None, num_processes=1, verbosity=0)

        [[self.assertAlmostEqual(c.simulate(tmodel1)[bs],1.) for c, bs in zip(cl, bsl)] for cl, bsl in zip(design1.circuit_lists, design1.idealout_lists)]

    def test_nonclifford_design_type1_construction(self):

        n = 2
        qs = ['Q'+str(i) for i in range(n)]
        ring = [('Q'+str(i),'Q'+str(i+1)) for i in range(n-1)]

        gateset2 = ['Gcphase'] + ['Gxpi2', 'Gzr']
        pspec2 = QPS(n, gateset2, availability={'Gcphase':ring}, qubit_labels=qs)
        tmodel2 = pygsti.models.create_crosstalk_free_model(pspec2)

        depths = [0, 2, 8]
        q_set = ('Q0', 'Q1')


        design2 = _rb.MirrorRBDesign(pspec2, depths, 3, qubit_labels=q_set, circuit_type='clifford+zxzxz-haar',
                                       clifford_compilations=None, sampler='edgegrab', samplerargs=(0.25,),
                                       localclifford=True, paulirandomize=True, descriptor='A mirror RB experiment',
                                       add_default_protocol=False, seed=None, num_processes=1, verbosity=0)


        [[self.assertAlmostEqual(c.simulate(tmodel2)[bs],1.) for c, bs in zip(cl, bsl)] for cl, bsl in zip(design2.circuit_lists, design2.idealout_lists)]
 
    def test_nonclifford_design_type2_construction(self):

        n = 2
        qs = ['Q'+str(i) for i in range(n)]
        ring = [('Q'+str(i),'Q'+str(i+1)) for i in range(n-1)]

        gateset3 = ['Gczr'] + ['Gxpi2', 'Gzr']
        pspec3 = QPS(n, gateset3, availability={'Gczr':ring}, qubit_labels=qs)
        tmodel3 = pygsti.models.create_crosstalk_free_model(pspec3)

        depths = [0, 2, 8]
        q_set = ('Q0', 'Q1')

        
        design3 = _rb.MirrorRBDesign(pspec3, depths, 3, qubit_labels=q_set, circuit_type='cz(theta)+zxzxz-haar',
                                       clifford_compilations=None, sampler='edgegrab', samplerargs=(0.25,),
                                       localclifford=True, paulirandomize=True, descriptor='A mirror RB experiment',
                                       add_default_protocol=False, seed=None, num_processes=1, verbosity=0)


        [[self.assertAlmostEqual(c.simulate(tmodel3)[bs],1.) for c, bs in zip(cl, bsl)] for cl, bsl in zip(design3.circuit_lists, design3.idealout_lists)]


    def test_serialization(self):

        mrb_design = _rb.MirrorRBDesign(self.pspec, self.depths, self.circuits_per_depth,
            qubit_labels=self.qubits, circuit_type=self.circuit_type, clifford_compilations=self.clifford_compilations,
            sampler=self.sampler, samplerargs=self.samplerargs,
            localclifford=True, paulirandomize=True, seed=self.seed, verbosity=self.verbosity,
            num_processes=1)

        mrb_design.write('../../test_packages/temp_test_files/test_MirrorRBDesign_serialization')
        #then read this back in
        mrb_design_read = _rb.MirrorRBDesign.from_dir('../../test_packages/temp_test_files/test_MirrorRBDesign_serialization')

        self.assertEqual(mrb_design.all_circuits_needing_data, mrb_design_read.all_circuits_needing_data)

class TestBiRBDesign(BaseCase):

    def setUp(self):
        self.num_qubits = 2
        self.qubit_labels = ['Q'+str(i) for i in range(self.num_qubits)]

        gate_names = ['Gi', 'Gxpi2', 'Gxpi', 'Gxmpi2', 'Gypi2', 'Gypi', 'Gympi2', 'Gzpi2', 'Gzpi', 'Gzmpi2', 'Gcphase'] 
        availability = {'Gcphase':[('Q'+str(i),'Q'+str((i+1) % self.num_qubits)) for i in range(self.num_qubits)]}

        self.pspec = pygsti.processors.QubitProcessorSpec(self.num_qubits, gate_names, availability=availability,
                                                          qubit_labels=self.qubit_labels)
        self.clifford_compilations = CCR.create_standard(self.pspec, 'absolute', ('paulis', '1Qcliffords'), verbosity=0)

        # TODO: Test a lot of these, currently just the default from the tutorial
        self.depths = [0, 2, 4]
        self.circuits_per_depth = 5
        self.qubits = ['Q0', 'Q1']
        self.circuit_type = 'clifford'
        self.sampler = 'edgegrab'
        self.samplerargs = [0.5]
        self.seed = 2021
        self.verbosity = 0

    def test_birb_design_construction_mixed1q2q(self):

        design = pygsti.protocols.BinaryRBDesign(self.pspec, self.clifford_compilations, self.depths, 
                                                 self.circuits_per_depth, qubit_labels=self.qubits, layer_sampling='mixed1q2q',
                                                 sampler=self.sampler, samplerargs=self.samplerargs, 
                                                 seed=self.seed, verbosity=0)
        
    def test_birb_design_construction_alternating1q2q(self):

        design = pygsti.protocols.BinaryRBDesign(self.pspec, self.clifford_compilations, self.depths, 
                                                 self.circuits_per_depth, qubit_labels=self.qubits, layer_sampling='alternating1q2q',
                                                 sampler=self.sampler, samplerargs=self.samplerargs, 
                                                 seed=self.seed, verbosity=0)
        
    def test_serialization(self):
        birb_design = pygsti.protocols.BinaryRBDesign(self.pspec, self.clifford_compilations, self.depths, 
                                                 self.circuits_per_depth, qubit_labels=self.qubits, layer_sampling='mixed1q2q',
                                                 sampler=self.sampler, samplerargs=self.samplerargs, 
                                                 seed=self.seed, verbosity=0)
        
        birb_design.write('../../test_packages/temp_test_files/test_BinaryRBDesign_serialization')
        #then read this back in
        birb_design_read = _rb.BinaryRBDesign.from_dir('../../test_packages/temp_test_files/test_BinaryRBDesign_serialization')

        self.assertEqual(birb_design.all_circuits_needing_data, birb_design_read.all_circuits_needing_data)
        
class TestBiRBProtocol(BaseCase):
    def setUp(self):
        self.num_qubits = 2
        self.qubit_labels = ['Q'+str(i) for i in range(self.num_qubits)]

        gate_names = ['Gi', 'Gxpi2', 'Gxpi', 'Gxmpi2', 'Gypi2', 'Gypi', 'Gympi2', 'Gzpi2', 'Gzpi', 'Gzmpi2', 'Gcphase'] 
        availability = {'Gcphase':[('Q'+str(i),'Q'+str((i+1) % self.num_qubits)) for i in range(self.num_qubits)]}

        self.pspec = pygsti.processors.QubitProcessorSpec(self.num_qubits, gate_names, availability=availability,
                                                          qubit_labels=self.qubit_labels)
        self.clifford_compilations = CCR.create_standard(self.pspec, 'absolute', ('paulis', '1Qcliffords'), verbosity=0)

        # TODO: Test a lot of these, currently just the default from the tutorial
        self.depths = [0, 2, 4]
        self.circuits_per_depth = 5
        self.qubits = ['Q0', 'Q1']
        self.circuit_type = 'clifford'
        self.sampler = 'edgegrab'
        self.samplerargs = [0.5]
        self.seed = 2021
        self.verbosity = 0

        self.design = pygsti.protocols.BinaryRBDesign(self.pspec, self.clifford_compilations, self.depths, 
                                                      self.circuits_per_depth, qubit_labels=self.qubits, layer_sampling='mixed1q2q',
                                                      sampler=self.sampler, samplerargs=self.samplerargs, 
                                                      seed=self.seed, verbosity=0)
        
        self.target_model =  pygsti.models.create_crosstalk_free_model(self.pspec)
        self.noisy_model =  pygsti.models.create_crosstalk_free_model(self.pspec, depolarization_strengths={name: .01 for name in gate_names})

        self.ds = pygsti.data.datasetconstruction.simulate_data(self.target_model, self.design.all_circuits_needing_data, 
                                                                num_samples = 100, seed= self.seed)
        self.ds_noisy = pygsti.data.datasetconstruction.simulate_data(self.noisy_model, self.design.all_circuits_needing_data, 
                                                                num_samples = 100, seed=self.seed)
        
        self.data = pygsti.protocols.ProtocolData(self.design, self.ds)
        self.data_noisy = pygsti.protocols.ProtocolData(self.design, self.ds_noisy)
        
    def test_birb_protocol_ideal(self):
        proto = pygsti.protocols.rb.RandomizedBenchmarking(datatype='energies', defaultfit='A-fixed', rtype='EI',
                 seed=(0.8, 0.95), bootstrap_samples=200, depths='all', name=None)
        
        result = proto.run(self.data)
        self.assertTrue(abs(result.fits['A-fixed'].estimates['r'])<=3e-5)
        
    def test_birb_protocol_noisy(self):
        proto = pygsti.protocols.rb.RandomizedBenchmarking(datatype='energies', defaultfit='A-fixed', rtype='EI',
                 seed=(0.8, 0.95), bootstrap_samples=200, depths='all', name=None)
        
        result = proto.run(self.data_noisy)


class TestCliffordRBProtocol(BaseCase):
    def setUp(self):
        self.num_qubits = 2
        self.qubit_labels = ['Q'+str(i) for i in range(self.num_qubits)]
        
        gate_names = ['Gxpi2', 'Gxmpi2', 'Gypi2', 'Gympi2', 'Gcphase']
        availability = {'Gcphase':[('Q'+str(i),'Q'+str((i+1) % self.num_qubits)) for i in range(self.num_qubits)]}

        self.pspec = pygsti.processors.QubitProcessorSpec(self.num_qubits, gate_names, availability=availability,
                                                          qubit_labels=self.qubit_labels)
        self.compilations = {
            'absolute': CCR.create_standard(self.pspec, 'absolute', ('paulis', '1Qcliffords'), verbosity=0),
            'paulieq': CCR.create_standard(self.pspec, 'paulieq', ('1Qcliffords', 'allcnots'), verbosity=0)
        }

        # TODO: Test a lot of these, currently just the default from the tutorial
        self.depths = [0, 2, 8]
        self.circuits_per_depth = 5
        self.qubits = ['Q0', 'Q1']
        self.citerations = 20
        self.randomizeout = True
        self.interleaved_circuit = None
        self.compiler_args = ()
        self.seed = 2021
        self.verbosity = 0

        self.design = _rb.CliffordRBDesign(self.pspec, self.compilations, self.depths, self.circuits_per_depth, qubit_labels=self.qubits,
                                           randomizeout=self.randomizeout, interleaved_circuit=self.interleaved_circuit,
                                           citerations=self.citerations, compilerargs=self.compiler_args, seed=self.seed,
                                           verbosity=self.verbosity, num_processes=1)
        
        self.target_model =  pygsti.models.create_crosstalk_free_model(self.pspec)
        self.noisy_model =  pygsti.models.create_crosstalk_free_model(self.pspec, depolarization_strengths={name: .01 for name in gate_names})

        self.ds = pygsti.data.datasetconstruction.simulate_data(self.target_model, self.design.all_circuits_needing_data, 
                                                                num_samples = 100, seed=self.seed)
        self.ds_noisy = pygsti.data.datasetconstruction.simulate_data(self.noisy_model, self.design.all_circuits_needing_data, 
                                                                num_samples = 100, seed=self.seed)
        
        self.data = pygsti.protocols.ProtocolData(self.design, self.ds)
        self.data_noisy = pygsti.protocols.ProtocolData(self.design, self.ds_noisy)
        
    def test_cliffordrb_protocol_ideal(self):
        proto = pygsti.protocols.rb.RandomizedBenchmarking(datatype='success_probabilities', defaultfit='A-fixed', rtype='EI',
                 seed=(0.8, 0.95), bootstrap_samples=200, depths='all', name=None)
        
        result = proto.run(self.data)

        self.assertTrue(abs(result.fits['A-fixed'].estimates['r'])<=3e-5)

        #also test writing and reading the results from disk.
        result.write('../../test_packages/temp_test_files/test_RandomizedBenchmarking_results')
        result_read = pygsti.io.read_results_from_dir('../../test_packages/temp_test_files/test_RandomizedBenchmarking_results')
        
    def test_cliffordrb_protocol_noisy(self):
        proto = pygsti.protocols.rb.RandomizedBenchmarking(datatype='success_probabilities', defaultfit='A-fixed', rtype='EI',
                 seed=(0.8, 0.95), bootstrap_samples=200, depths='all', name=None)
        
        result = proto.run(self.data_noisy)

class TestDirectRBProtocol(BaseCase):
    def setUp(self):
        self.num_qubits = 2
        self.qubit_labels = ['Q'+str(i) for i in range(self.num_qubits)]
        
        gate_names = ['Gxpi2', 'Gxmpi2', 'Gypi2', 'Gympi2', 'Gcphase']
        availability = {'Gcphase':[('Q'+str(i),'Q'+str((i+1) % self.num_qubits)) for i in range(self.num_qubits)]}

        self.pspec = pygsti.processors.QubitProcessorSpec(self.num_qubits, gate_names, availability=availability,
                                                          qubit_labels=self.qubit_labels, geometry='line')
        self.compilations = {
            'absolute': CCR.create_standard(self.pspec, 'absolute', ('paulis', '1Qcliffords'), verbosity=0),
            'paulieq': CCR.create_standard(self.pspec, 'paulieq', ('1Qcliffords', 'allcnots'), verbosity=0)
        }


        # TODO: Test a lot of these, currently just the default from the tutorial
        self.depths = [0, 2, 8]
        self.circuits_per_depth = 5
        self.qubits = ['Q0', 'Q1']
        self.randomizeout = True
        self.sampler = 'edgegrab'
        self.samplerargs = [0.5]
        self.citerations = 20
        self.compiler_args = ()
        self.seed = 2021
        self.verbosity = 0

        self.design =_rb.DirectRBDesign(self.pspec, self.compilations, self.depths, self.circuits_per_depth,
                                        qubit_labels=self.qubits, sampler=self.sampler, samplerargs=self.samplerargs,
                                        addlocal=False, lsargs=(), randomizeout=self.randomizeout, cliffordtwirl=True,
                                        conditionaltwirl=True, citerations=self.citerations, compilerargs=self.compiler_args,
                                        partitioned=False, seed=self.seed, verbosity=self.verbosity, num_processes=1)
                                    
        
        self.target_model =  pygsti.models.create_crosstalk_free_model(self.pspec)
        self.noisy_model =  pygsti.models.create_crosstalk_free_model(self.pspec, depolarization_strengths={name: .01 for name in gate_names})

        self.ds = pygsti.data.datasetconstruction.simulate_data(self.target_model, self.design.all_circuits_needing_data, 
                                                                num_samples = 100, seed=self.seed)
        self.ds_noisy = pygsti.data.datasetconstruction.simulate_data(self.noisy_model, self.design.all_circuits_needing_data, 
                                                                num_samples = 100, seed=self.seed)
        
        self.data = pygsti.protocols.ProtocolData(self.design, self.ds)
        self.data_noisy = pygsti.protocols.ProtocolData(self.design, self.ds_noisy)
        
    def test_directrb_protocol_ideal(self):
        proto = pygsti.protocols.rb.RandomizedBenchmarking(datatype='success_probabilities', defaultfit='A-fixed', rtype='EI',
                 seed=(0.8, 0.95), bootstrap_samples=200, depths='all', name=None)
        
        result = proto.run(self.data)
        self.assertTrue(abs(result.fits['A-fixed'].estimates['r'])<=3e-5)
        
    def test_directrb_protocol_noisy(self):
        proto = pygsti.protocols.rb.RandomizedBenchmarking(datatype='success_probabilities', defaultfit='A-fixed', rtype='EI',
                 seed=(0.8, 0.95), bootstrap_samples=200, depths='all', name=None)
        
        result = proto.run(self.data_noisy)

class TestMirrorRBProtocol(BaseCase):
    def setUp(self):
        self.num_qubits = 2
        self.qubit_labels = ['Q'+str(i) for i in range(self.num_qubits)]

        gate_names = ['Gi', 'Gxpi2', 'Gxpi', 'Gxmpi2', 'Gypi2', 'Gypi', 'Gympi2', 'Gzpi2', 'Gzpi', 'Gzmpi2', 'Gcphase'] 
        availability = {'Gcphase':[('Q'+str(i),'Q'+str((i+1) % self.num_qubits)) for i in range(self.num_qubits)]}

        self.pspec = pygsti.processors.QubitProcessorSpec(self.num_qubits, gate_names, availability=availability,
                                                          qubit_labels=self.qubit_labels, geometry='line')
        self.clifford_compilations = {
            'absolute': CCR.create_standard(self.pspec, 'absolute', ('paulis', '1Qcliffords'), verbosity=0)
            # SS: I think this is for speed, don't need paulieq for MirrorRB?
        }

        # TODO: Test a lot of these, currently just the default from the tutorial
        self.depths = [0, 2, 8]
        self.circuits_per_depth = 5
        self.qubits = ['Q0', 'Q1']
        self.circuit_type = 'clifford'
        self.sampler = 'edgegrab'
        self.samplerargs = [0.5]
        self.seed = 2021
        self.verbosity = 0

        self.design =_rb.MirrorRBDesign(self.pspec, self.depths, self.circuits_per_depth,
                                        qubit_labels=self.qubits, circuit_type=self.circuit_type, clifford_compilations=self.clifford_compilations,
                                        sampler=self.sampler, samplerargs=self.samplerargs,
                                        localclifford=True, paulirandomize=True, seed=self.seed, verbosity=self.verbosity,
                                        num_processes=1)
        
        self.target_model =  pygsti.models.create_crosstalk_free_model(self.pspec)
        self.noisy_model =  pygsti.models.create_crosstalk_free_model(self.pspec, depolarization_strengths={name: .01 for name in gate_names})

        self.ds = pygsti.data.datasetconstruction.simulate_data(self.target_model, self.design.all_circuits_needing_data, 
                                                                num_samples = 100, seed=self.seed)
        self.ds_noisy = pygsti.data.datasetconstruction.simulate_data(self.noisy_model, self.design.all_circuits_needing_data, 
                                                                num_samples = 100, seed=self.seed)
        
        self.data = pygsti.protocols.ProtocolData(self.design, self.ds)
        self.data_noisy = pygsti.protocols.ProtocolData(self.design, self.ds_noisy)
        
    def test_mirrorrb_protocol_ideal(self):
        proto = pygsti.protocols.rb.RandomizedBenchmarking(datatype='adjusted_success_probabilities', defaultfit='A-fixed', rtype='EI',
                 seed=(0.8, 0.95), bootstrap_samples=200, depths='all', name=None)
        
        result = proto.run(self.data)
        self.assertTrue(abs(result.fits['A-fixed'].estimates['r'])<=3e-5)
        
    def test_mirrorrb_protocol_noisy(self):
        proto = pygsti.protocols.rb.RandomizedBenchmarking(datatype='adjusted_success_probabilities', defaultfit='A-fixed', rtype='EI',
                 seed=(0.8, 0.95), bootstrap_samples=200, depths='all', name=None)
        
        result = proto.run(self.data_noisy)


class TestInterleavedRBProtocol(BaseCase):
    def setUp(self):
        n_qubits = 1
        qubit_labels = ['Q0']
        gate_names = ['Gxpi2', 'Gxmpi2', 'Gypi2', 'Gympi2']
        pspec = QPS(n_qubits, gate_names, qubit_labels=qubit_labels)
        compilations = {'absolute': CCR.create_standard(pspec, 'absolute', ('paulis', '1Qcliffords'), verbosity=0),            
                        'paulieq': CCR.create_standard(pspec, 'paulieq', ('1Qcliffords', 'allcnots'), verbosity=0)}
        interleaved_circuit = Circuit([Label('Gxpi2', 'Q0')], line_labels=('Q0',))

        # TODO: Test a lot of these, currently just the default from the tutorial
        depths = [0, 1, 2, 4, 8, 16, 32]
        circuits_per_depth = 30
        citerations = 20
        randomizeout = False
        compiler_args = ()
        seed = 1234
        verbosity = 0

        self.design = _rb.InterleavedRBDesign(pspec, compilations, depths, circuits_per_depth, interleaved_circuit, qubit_labels,
                                           randomizeout=randomizeout, citerations=citerations, compilerargs=compiler_args, seed=seed,
                                           verbosity=verbosity, num_processes=1)
        
        self.target_model =  pygsti.models.create_crosstalk_free_model(pspec)
        self.target_model.sim = 'map'
        depolarization_strengths={g:0.01 for g in pspec.gate_names if g!= 'Gxpi2'}
        depolarization_strengths['Gxpi2'] = .02
        self.noisy_model =  pygsti.models.create_crosstalk_free_model(pspec, depolarization_strengths=depolarization_strengths)
        self.noisy_model.sim = 'map'
        self.ds = pygsti.data.datasetconstruction.simulate_data(self.target_model, self.design.all_circuits_needing_data, 
                                                                num_samples = 1000, seed=seed)
        self.ds_noisy = pygsti.data.datasetconstruction.simulate_data(self.noisy_model, self.design.all_circuits_needing_data, 
                                                                num_samples = 1000, seed=seed)
        
        self.data = pygsti.protocols.ProtocolData(self.design, self.ds)
        self.data_noisy = pygsti.protocols.ProtocolData(self.design, self.ds_noisy)
        
    def test_interleavedrb_protocol_ideal(self):
        #running with all default settings
        proto = _rb.InterleavedRandomizedBenchmarking()
        
        result = proto.run(self.data)
        estimated_irb_num = result.for_protocol['InterleavedRandomizedBenchmarking'].irb_numbers['full'] 
        self.assertTrue(abs(estimated_irb_num) <= 1e-5)

        #also test writing and reading the results from disk.
        result.write('../../test_packages/temp_test_files/test_InterleavedRandomizedBenchmarking_results')
        result_read = pygsti.io.read_results_from_dir('../../test_packages/temp_test_files/test_InterleavedRandomizedBenchmarking_results')
        
        
    def test_interleavedrb_protocol_noisy(self):
        #running with all default settings
        proto = _rb.InterleavedRandomizedBenchmarking()
        
        result = proto.run(self.data_noisy)
        estimated_irb_num = result.for_protocol['InterleavedRandomizedBenchmarking'].irb_numbers['full'] 
        print(result.for_protocol['InterleavedRandomizedBenchmarking'].irb_numbers)

        self.assertTrue(abs(estimated_irb_num-.02) <= 5e-3)

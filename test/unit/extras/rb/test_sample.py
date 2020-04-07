from ...util import BaseCase

from pygsti.objects import ProcessorSpec, Label
#from pygsti.extras.rb import sample


class RBSampleTester(BaseCase):
    @classmethod
    def setUpClass(cls):
        super(RBSampleTester, cls).setUpClass()
        glist = ['Gxpi2', 'Gypi2', 'Gcnot']  # 'Gi',
        cls.pspec_1 = ProcessorSpec(4, glist, verbosity=0, qubit_labels=['Q0', 'Q1', 'Q2', 'Q3'])

        # # XXX this takes nearly a minute to construct on my machine....
        # glist = ['Gxpi', 'Gypi', 'Gzpi', 'Gh', 'Gp', 'Gcphase']  # 'Gi',
        # availability = {'Gcphase': [(0, 1), (1, 2)]}
        # cls.pspec_2 = ProcessorSpec(3, glist, availability=availability, verbosity=0)

        # XXX is this an OK test fixture? see above.
        glist = ['Gxpi2', 'Gypi2', 'Gcphase']  # 'Gi',
        availability = {'Gcphase': [(0, 1), (1, 2)]}
        cls.pspec_2 = ProcessorSpec(3, glist, availability=availability, verbosity=0)

        glist = ['Gxpi2', 'Gxmpi2', 'Gypi2', 'Gympi2', 'Gcnot']  # 'Gi',
        cls.pspec_inv = ProcessorSpec(4, glist, verbosity=0, qubit_labels=['Q0', 'Q1', 'Q2', 'Q3'])

    def test_clifford_rb_experiment(self):
        self.skipTest("RB analysis is known to be broken.  Skip tests until it gets fixed.")
        lengths = [0, 2, 5]
        circuits_per_length = 2
        subsetQs = ['Q1', 'Q2', 'Q3']
        out = sample.clifford_rb_experiment(
            self.pspec_1, lengths, circuits_per_length,
            subsetQs=subsetQs, randomizeout=False, citerations=2,
            compilerargs=[], descriptor='A Clifford RB experiment',
            verbosity=0
        )
        for key in list(out['idealout'].keys()):
            self.assertEqual(out['idealout'][key], (0, 0, 0))

        self.assertEqual(len(out['circuits']), circuits_per_length * len(lengths))

        out = sample.clifford_rb_experiment(
            self.pspec_2, lengths, circuits_per_length,
            subsetQs=None, randomizeout=False, citerations=1,
            compilerargs=[], descriptor='A Clifford RB experiment',
            verbosity=0
        )
        # TODO assert correctness

    def test_circuit_layer_by_pairing_qubits(self):
        self.skipTest("RB analysis is known to be broken.  Skip tests until it gets fixed.")
        n = self.pspec_1.number_of_qubits
        layer = sample.circuit_layer_by_pairing_qubits(
            self.pspec_1, twoQprob=0.0, oneQgatenames='all',
            twoQgatenames='all', modelname='clifford'
        )
        self.assertEqual(len(layer), n)

        layer = sample.circuit_layer_by_pairing_qubits(
            self.pspec_1, twoQprob=1.0, oneQgatenames='all',
            twoQgatenames='all', modelname='clifford'
        )
        self.assertEqual(len(layer), n // 2)

        layer = sample.circuit_layer_by_pairing_qubits(
            self.pspec_1, twoQprob=0.0, oneQgatenames=['Gx'],
            twoQgatenames='all', modelname='clifford'
        )
        self.assertEqual(layer[0].name, 'Gx')

        layer = sample.circuit_layer_by_pairing_qubits(
            self.pspec_1, twoQprob=0.0, oneQgatenames=['Gxpi'],
            twoQgatenames='all', modelname='target'
        )
        self.assertEqual(layer[0].name, 'Gxpi')

    def test_circuit_layer_by_Qelimination(self):
        self.skipTest("RB analysis is known to be broken.  Skip tests until it gets fixed.")
        n = self.pspec_2.number_of_qubits
        layer = sample.circuit_layer_by_Qelimination(
            self.pspec_2, twoQprob=0.0, oneQgates='all',
            twoQgates='all', modelname='clifford'
        )
        self.assertEqual(len(layer), self.pspec_2.number_of_qubits)
        layer = sample.circuit_layer_by_Qelimination(
            self.pspec_2, twoQprob=1.0, oneQgates='all',
            twoQgates='all', modelname='clifford'
        )
        self.assertEqual(len(layer), (n % 2) + n // 2)

    def test_circuit_layer_by_co2Qgates(self):
        self.skipTest("RB analysis is known to be broken.  Skip tests until it gets fixed.")
        n = self.pspec_1.number_of_qubits
        C01 = Label('Gcnot', ('Q0', 'Q1'))
        C23 = Label('Gcnot', ('Q2', 'Q3'))
        co2Qgates = [[], [C01, C23]]

        layer = sample.circuit_layer_by_co2Qgates(
            self.pspec_1, None, co2Qgates, co2Qgatesprob='uniform',
            twoQprob=1.0, oneQgatenames='all', modelname='clifford'
        )
        self.assertTrue(len(layer) == n or len(layer) == n // 2)

        layer = sample.circuit_layer_by_co2Qgates(
            self.pspec_1, None, co2Qgates, co2Qgatesprob=[0., 1.],
            twoQprob=1.0, oneQgatenames='all', modelname='clifford'
        )
        self.assertEqual(len(layer), n // 2)

        layer = sample.circuit_layer_by_co2Qgates(
            self.pspec_1, None, co2Qgates, co2Qgatesprob=[1., 0.],
            twoQprob=1.0, oneQgatenames=['Gx'], modelname='clifford'
        )
        self.assertEqual(len(layer), n)
        self.assertEqual(layer[0].name, 'Gx')

        co2Qgates = [[], [C23]]
        layer = sample.circuit_layer_by_co2Qgates(
            self.pspec_1, ['Q2', 'Q3'], co2Qgates,
            co2Qgatesprob=[0.25, 0.75], twoQprob=0.5,
            oneQgatenames='all', modelname='clifford'
        )
        # TODO assert correctness

        co2Qgates = [[C01]]
        layer = sample.circuit_layer_by_co2Qgates(
            self.pspec_1, None, co2Qgates, co2Qgatesprob=[1.],
            twoQprob=1.0, oneQgatenames='all', modelname='clifford'
        )
        self.assertEqual(layer[0].name, 'Gcnot')
        self.assertEqual(len(layer), 3)

        # Tests the nested co2Qgates option.
        co2Qgates = [[], [[C01, C23], [C01]]]
        layer = sample.circuit_layer_by_co2Qgates(
            self.pspec_1, None, co2Qgates, co2Qgatesprob='uniform',
            twoQprob=1.0, oneQgatenames='all', modelname='clifford'
        )
        # TODO assert correctness

    def test_circuit_layer_of_oneQgates(self):
        self.skipTest("RB analysis is known to be broken.  Skip tests until it gets fixed.")
        layer = sample.circuit_layer_of_oneQgates(
            self.pspec_1, oneQgatenames='all', pdist='uniform',
            modelname='clifford'
        )
        self.assertEqual(len(layer), self.pspec_1.number_of_qubits)
        layer = sample.circuit_layer_of_oneQgates(
            self.pspec_1, subsetQs=['Q1', 'Q2'],
            oneQgatenames=['Gx', 'Gy'], pdist=[1., 0.],
            modelname='clifford'
        )
        self.assertEqual(len(layer), 2)
        self.assertEqual(layer[0].name, 'Gx')
        layer = sample.circuit_layer_of_oneQgates(
            self.pspec_1, subsetQs=['Q2'], oneQgatenames=['Gx'],
            pdist=[3.], modelname='clifford'
        )
        self.assertEqual(layer[0], Label('Gx', 'Q2'))
        self.assertEqual(len(layer), 1)
        layer = sample.circuit_layer_of_oneQgates(
            self.pspec_1, oneQgatenames=['Gx'], pdist='uniform',
            modelname='clifford'
        )

    def test_random_circuit(self):
        self.skipTest("RB analysis is known to be broken.  Skip tests until it gets fixed.")
        C01 = Label('Gcnot', ('Q0', 'Q1'))
        C23 = Label('Gcnot', ('Q2', 'Q3'))
        co2Qgates = [[], [[C01, C23], [C01, ]]]

        circuit = sample.random_circuit(self.pspec_1, length=100, sampler='Qelimination')
        self.assertEqual(circuit.depth(), 100)

        circuit = sample.random_circuit(self.pspec_2, length=100, sampler='Qelimination',
                                        samplerargs=[0.1, ], addlocal=True)
        self.assertEqual(circuit.depth(), 201)
        self.assertLessEqual(len(circuit.get_layer(0)), self.pspec_2.number_of_qubits)

        circuit = sample.random_circuit(self.pspec_1, length=100, sampler='pairingQs')
        # TODO assert correctness

        circuit = sample.random_circuit(
            self.pspec_1, length=10, sampler='pairingQs',
            samplerargs=[0.1, ['Gx', ]]
        )
        # TODO assert correctness

        circuit = sample.random_circuit(self.pspec_1, length=100, sampler='co2Qgates', samplerargs=[co2Qgates])
        # TODO assert correctness

        circuit = sample.random_circuit(
            self.pspec_1, length=100, sampler='co2Qgates',
            samplerargs=[co2Qgates, [0.1, 0.2], 0.1], addlocal=True,
            lsargs=[['Gx', ]]
        )
        self.assertEqual(circuit.depth(), 201)

        circuit = sample.random_circuit(self.pspec_1, length=5, sampler='local')
        self.assertEqual(circuit.depth(), 5)

        circuit = sample.random_circuit(self.pspec_1, length=5, sampler='local', samplerargs=[['Gx']])
        self.assertEqual(circuit[0, 'Q0'].name, 'Gx')

    def test_direct_rb_experiment(self):
        self.skipTest("RB analysis is known to be broken.  Skip tests until it gets fixed.")
        lengths = [0, 2, 5]
        circuits_per_length = 2

        # Test DRB experiment with all defaults.
        exp = sample.direct_rb_experiment(self.pspec_2, lengths, circuits_per_length, verbosity=0)
        # TODO assert correctness

        exp = sample.direct_rb_experiment(
            self.pspec_2, lengths, circuits_per_length, subsetQs=[0, 1],
            sampler='pairingQs', cliffordtwirl=False,
            conditionaltwirl=False, citerations=2, partitioned=True,
            verbosity=0
        )
        # TODO assert correctness

        exp = sample.direct_rb_experiment(
            self.pspec_2, lengths, circuits_per_length, subsetQs=[0, 1],
            sampler='co2Qgates',
            samplerargs=[[[], [Label('Gcphase', (0, 1)), ]], [0., 1.]],
            cliffordtwirl=False, conditionaltwirl=False,
            citerations=2, partitioned=True, verbosity=0
        )
        # TODO assert correctness

        exp = sample.direct_rb_experiment(
            self.pspec_2, lengths, circuits_per_length, subsetQs=[0, 1],
            sampler='local', cliffordtwirl=False,
            conditionaltwirl=False, citerations=2, partitioned=True,
            verbosity=0
        )
        # TODO assert correctness

    def test_mirror_rb_experiment(self):
        self.skipTest("RB analysis is known to be broken.  Skip tests until it gets fixed.")
        lengths = [0, 4, 8]
        circuits_per_length = 10

        exp = sample.mirror_rb_experiment(
            self.pspec_inv, lengths, circuits_per_length,
            subsetQs=['Q1', 'Q2', 'Q3'], sampler='Qelimination',
            samplerargs=[], localclifford=True, paulirandomize=True
        )
        # TODO assert correctness

        exp = sample.mirror_rb_experiment(
            self.pspec_inv, lengths, circuits_per_length,
            subsetQs=['Q1', 'Q2', 'Q3'], sampler='Qelimination',
            samplerargs=[], localclifford=True, paulirandomize=False
        )
        # TODO assert correctness

        exp = sample.mirror_rb_experiment(
            self.pspec_inv, lengths, circuits_per_length,
            subsetQs=['Q1', 'Q2', 'Q3'], sampler='Qelimination',
            samplerargs=[], localclifford=False, paulirandomize=False
        )
        # TODO assert correctness

        exp = sample.mirror_rb_experiment(
            self.pspec_inv, lengths, circuits_per_length,
            subsetQs=['Q1', 'Q2', 'Q3'], sampler='Qelimination',
            samplerargs=[], localclifford=False, paulirandomize=True
        )
        # TODO assert correctness

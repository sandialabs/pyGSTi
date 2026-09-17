import numpy as np

from pygsti.algorithms import compilers
from pygsti.baseobjs import Label
from pygsti.processors import QubitProcessorSpec
from pygsti.processors import CliffordCompilationRules
from pygsti.circuits import Circuit
from pygsti.tools import symplectic
from ..util import BaseCase, Namespace

## Immutable test fixture data
fixture_1Q = Namespace(
    n=1,
    # arbitrary symplectic representation of a 1-qubit Clifford
    # (generated with `symplectic.random_clifford(1)`)
    clifford_sym=np.array([[1, 0],
                           [0, 1]], dtype=np.int8),
    clifford_phase=np.array([0, 2])
)
fixture_1Q.pspec = QubitProcessorSpec(num_qubits=1, gate_names=['Gcnot', 'Gh', 'Gp', 'Gxpi', 'Gypi', 'Gzpi'], geometry='line')
fixture_1Q.clifford_abs = CliffordCompilationRules.create_standard(fixture_1Q.pspec, compile_type="absolute",
                                                                   what_to_compile=("1Qcliffords","paulis"), verbosity=1)
fixture_1Q.clifford_peq = CliffordCompilationRules.create_standard(fixture_1Q.pspec, compile_type="paulieq",
                                                                   what_to_compile=("1Qcliffords","allcnots"), verbosity=1)
                                             
fixture_2Q = Namespace(
    n=2,
    qubit_labels=['Q0', 'Q1'],
    availability={'Gcnot': [('Q0', 'Q1')]},
    gate_names=['Gh', 'Gp', 'Gxpi', 'Gpdag', 'Gcnot'],
    # generated as before:
    clifford_sym=np.array([[0, 1, 1, 1],
                           [1, 0, 1, 1],
                           [1, 0, 1, 0],
                           [0, 1, 0, 1]]),
    clifford_phase=np.array([2, 0, 1, 3])
)
fixture_2Q.pspec = QubitProcessorSpec(fixture_2Q.n, gate_names=fixture_2Q.gate_names,
                                      availability=fixture_2Q.availability,
                                      qubit_labels=fixture_2Q.qubit_labels, geometry='line')
fixture_2Q.clifford_abs = CliffordCompilationRules.create_standard(fixture_2Q.pspec, compile_type="absolute",
                                             what_to_compile=("1Qcliffords","paulis"), verbosity=1)
fixture_2Q.clifford_peq = CliffordCompilationRules.create_standard(fixture_2Q.pspec, compile_type="paulieq",
                                             what_to_compile=("1Qcliffords","allcnots"), verbosity=1)


# Totally arbitrary CNOT circuit
fixture_2Q.cnot_circuit = Circuit(layer_labels=[
    Label('CNOT', ('Q1', 'Q0')),
    Label('CNOT', ('Q1', 'Q0')),
    Label('CNOT', ('Q0', 'Q1')),
    Label('CNOT', ('Q1', 'Q0'))
], line_labels=fixture_2Q.qubit_labels)
fixture_2Q.cnot_circuit_sym, fixture_2Q.cnot_circuit_phase = \
    symplectic.symplectic_rep_of_clifford_circuit(fixture_2Q.cnot_circuit)


fixture_3Q = Namespace(
    n=3,
    qubit_labels=['Q0', 'Q1', 'Q2'],
    availability={'Gcnot': [('Q0', 'Q1'), ('Q1', 'Q2')]},
    gate_names=['Gh', 'Gp', 'Gxpi', 'Gpdag', 'Gcnot'],
    # generated as before:
    clifford_sym=np.array([[0, 0, 1, 1, 0, 1],
                           [0, 0, 0, 0, 1, 0],
                           [1, 0, 0, 0, 1, 1],
                           [1, 0, 1, 1, 1, 1],
                           [0, 1, 0, 1, 1, 0],
                           [0, 0, 1, 0, 0, 1]]),
    clifford_phase=np.array([2, 2, 3, 1, 1, 2])
)
fixture_3Q.pspec = QubitProcessorSpec(fixture_3Q.n, gate_names=fixture_3Q.gate_names,
                                      availability=fixture_3Q.availability,
                                      qubit_labels=fixture_3Q.qubit_labels, geometry='line')
fixture_3Q.clifford_abs = CliffordCompilationRules.create_standard(fixture_3Q.pspec, compile_type="absolute",
                                                                   what_to_compile=("1Qcliffords","paulis"), verbosity=1)
fixture_3Q.clifford_peq = CliffordCompilationRules.create_standard(fixture_3Q.pspec, compile_type="paulieq",
                                                                   what_to_compile=("1Qcliffords","allcnots"), verbosity=1)


class CompilersTester(BaseCase):
    def test_compile_symplectic_using_GGE_core(self):
        compiled = compilers._compile_symplectic_using_gge_core(fixture_2Q.clifford_sym)
        # TODO assert intermediate correctness
        sym_out, _ = symplectic.symplectic_rep_of_clifford_circuit(compiled)
        self.assertArraysEqual(fixture_2Q.clifford_sym, sym_out)


class LabelAwareCliffordCompilationTester(BaseCase):
    def test_template_search_uses_argumented_gate_labels(self):
        def unitary_factory(args):
            return np.identity(4, 'd')

        unitary_factory.shape = (4, 4)

        def srep_factory(label):
            return symplectic.symplectic_rep_of_clifford_layer(Label('P', label.args[0]), n=2)

        pspec = QubitProcessorSpec(
            2, ['Gargp'], nonstd_gate_unitaries={'Gargp': unitary_factory},
            availability={'Gargp': [('Q0', 'Q1')]}, qubit_labels=('Q0', 'Q1'),
            nonstd_gate_symplecticreps={'Gargp': srep_factory},
            gate_arg_label_indices={'Gargp': (0,)})
        compilation_rules = CliffordCompilationRules(pspec, compile_type="absolute")

        label0 = Label('Gargp', (0, 1), args=(0,))
        label1 = Label('Gargp', (0, 1), args=(1,))
        target_srep = symplectic.symplectic_rep_of_clifford_layer(Label('P', 1), n=2)
        template = compilation_rules.add_clifford_compilation_template(
            'Gtarget', nqubits=2, unitary=None, srep=target_srep,
            available_gatelabels=(label0, label1), available_sreps={'Gargp': srep_factory},
            verbosity=0, max_iterations=1)

        self.assertEqual(template, [label1])

        real_circuit = compilation_rules._create_local_compilation_of(
            Label('Gtarget', ('Q0', 'Q1')), srep=target_srep, verbosity=0)
        self.assertEqual(real_circuit.layer_label(0), Label('Gargp', ('Q0', 'Q1'), args=('Q1',)))


class CompileSymplecticTester(BaseCase):
    def setUp(self):
        super(CompileSymplecticTester, self).setUp()
        self.options = dict(
            iterations=3,
            algorithms=['BGGE', 'ROGGE']
        )

    def test_compile_symplectic(self):
        compiled = compilers.compile_symplectic(
            fixture_2Q.clifford_sym,
            **self.options
        )
        # TODO assert intermediate correctness
        sym_out, _ = symplectic.symplectic_rep_of_clifford_circuit(
            compiled, pspec=self.options.get('pspec', None)
        )
        self.assertArraysEqual(fixture_2Q.clifford_sym, sym_out)


class CompileSymplecticPspecTester(CompileSymplecticTester):
    def setUp(self):
        super(CompileSymplecticPspecTester, self).setUp()
        self.options.update(pspec=fixture_2Q.pspec,
                            absolute_compilation=fixture_2Q.clifford_abs,
                            paulieq_compilation=fixture_2Q.clifford_peq)


class CompileSymplecticSubsetTester(CompileSymplecticTester):
    def setUp(self):
        super(CompileSymplecticSubsetTester, self).setUp()
        self.options.update(
            pspec=fixture_3Q.pspec,
            absolute_compilation=fixture_3Q.clifford_abs,
            paulieq_compilation=fixture_3Q.clifford_peq,
            qubit_labels=['Q1', 'Q2'],
            iterations=2,
            algorithms=['BGGE', 'ROGGE', 'iAGvGE']
        )


class CompileCliffordBase(object):
    def setUp(self):
        super(CompileCliffordBase, self).setUp()
        self.options = dict(
            iterations=2,
            algorithm='ROGGE'
        )

    def test_compile_clifford(self):
        compiled = compilers.compile_clifford(
            self.fixture.clifford_sym,
            self.fixture.clifford_phase,
            **self.options
        )
        # TODO assert intermediate correctness
        sym_out, phase_out = symplectic.symplectic_rep_of_clifford_circuit(
            compiled, pspec=self.options.get('pspec', None)
        )
        self.assertArraysEqual(self.fixture.clifford_sym, sym_out)
        self.assertArraysEqual(self.fixture.clifford_phase, phase_out)


class CompileClifford1QTester(CompileCliffordBase, BaseCase):
    fixture = fixture_1Q


class CompileClifford1QPspecTester(CompileClifford1QTester):
    def setUp(self):
        super(CompileClifford1QPspecTester, self).setUp()
        self.options.update(
            pspec=fixture_2Q.pspec,
            absolute_compilation=fixture_2Q.clifford_abs,
            paulieq_compilation=fixture_2Q.clifford_peq,
            qubit_labels=['Q1'],
            prefixpaulis=False,
            paulirandomize=True
        )


class CompileClifford2QTester(CompileClifford1QTester):
    fixture = fixture_2Q


class CompileClifford2QPspecTester(CompileClifford2QTester):
    def setUp(self):
        super(CompileClifford2QPspecTester, self).setUp()
        self.options.update(pspec=fixture_2Q.pspec,
                            absolute_compilation=fixture_2Q.clifford_abs,
                            paulieq_compilation=fixture_2Q.clifford_peq)


class CompileCliffordSubsetTester(CompileClifford2QTester):
    def setUp(self):
        super(CompileCliffordSubsetTester, self).setUp()
        self.options.update(
            pspec=fixture_3Q.pspec,
            absolute_compilation=fixture_3Q.clifford_abs,
            paulieq_compilation=fixture_3Q.clifford_peq,
            qubit_labels=['Q1', 'Q2'],
            prefixpaulis=True,
            paulirandomize=True
        )


class CompileCliffordSubsetBGGETester(CompileCliffordSubsetTester):
    def setUp(self):
        super(CompileCliffordSubsetBGGETester, self).setUp()
        self.options.update(
            algorithm='BGGE'
        )


class CompileCliffordSubsetiAGvGETester(CompileCliffordSubsetTester):
    def setUp(self):
        super(CompileCliffordSubsetiAGvGETester, self).setUp()
        self.options.update(
            algorithm='iAGvGE',
            paulirandomize=False
        )


class CompileCNOTCircuitBase(object):
    def test_compile_cnot_circuit(self):
        compiled = compilers.compile_cnot_circuit(
            fixture_2Q.cnot_circuit_sym,
            fixture_2Q.pspec,
            fixture_2Q.clifford_abs,
            algorithm=self.algorithm,
            aargs=self.aargs
        )
        # TODO assert correctness

    def test_compile_cnot_circuit_subset(self):
        compiled = compilers.compile_cnot_circuit(
            fixture_2Q.cnot_circuit_sym,
            fixture_3Q.pspec,
            fixture_3Q.clifford_abs,
            qubit_labels=fixture_2Q.qubit_labels,
            algorithm=self.algorithm,
            aargs=self.aargs
        )
        # TODO assert correctness


class CompileCNOTCircuitCOiCAGETester(CompileCNOTCircuitBase, BaseCase):
    algorithm = 'COiCAGE'
    aargs = []


class CompileCNOTCircuitOiCAGETester(CompileCNOTCircuitBase, BaseCase):
    algorithm = 'OiCAGE'
    aargs = [['Q0', 'Q1'], ]


class CompileCNOTCircuitCOCAGETester(CompileCNOTCircuitBase, BaseCase):
    algorithm = 'COCAGE'
    aargs = []


class CompileCNOTCircuitROCAGETester(CompileCNOTCircuitBase, BaseCase):
    algorithm = 'ROCAGE'
    aargs = []


class CompileCNOTCircuitBGETester(CompileCNOTCircuitBase, BaseCase):
    # Regression coverage for the basic-Gaussian-elimination CNOT compiler. It previously crashed
    # because `_compile_cnot_circuit_using_bge_algorithm` built its Circuit with the obsolete
    # `gatesring` keyword instead of `layer_labels`, so the 'BGE' path had no working coverage.
    algorithm = 'BGE'
    aargs = []

    def test_compile_cnot_circuit_bge_is_correct(self):
        compiled = compilers.compile_cnot_circuit(
            fixture_2Q.cnot_circuit_sym,
            fixture_2Q.pspec,
            fixture_2Q.clifford_abs,
            algorithm='BGE',
            check=True
        )
        sym_out, _ = symplectic.symplectic_rep_of_clifford_circuit(compiled, pspec=fixture_2Q.pspec)
        self.assertArraysEqual(fixture_2Q.cnot_circuit_sym, sym_out)


class CompileStabilizerBase(object):
    def setUp(self):
        super(CompileStabilizerBase, self).setUp()
        self.options = {}

    def test_compile_stabilizer_state(self):
        compiled = compilers.compile_stabilizer_state(
            self.fixture.clifford_sym,
            self.fixture.clifford_phase,
            **self.options
        )
        sym_0, phase_0 = symplectic.prep_stabilizer_state(self.fixture.n)
        sym_compiled, phase_compiled = symplectic.symplectic_rep_of_clifford_circuit(
            compiled, pspec=self.options.get('pspec', None)
        )
        sym_out, phase_out = symplectic.apply_clifford_to_stabilizer_state(
            sym_compiled, phase_compiled, sym_0, phase_0
        )
        sym_target, phase_target = symplectic.apply_clifford_to_stabilizer_state(
            self.fixture.clifford_sym, self.fixture.clifford_phase, sym_0, phase_0
        )
        for i in range(self.fixture.n):
            self.assertArraysAlmostEqual(
                symplectic.pauli_z_measurement(sym_target, phase_target, i)[0],
                symplectic.pauli_z_measurement(sym_out, phase_out, i)[0]
            )

    def test_compile_stabilizer_measurement(self):
        compiled = compilers.compile_stabilizer_measurement(
            self.fixture.clifford_sym,
            self.fixture.clifford_phase,
            **self.options
        )
        sym_compiled, phase_compiled = symplectic.symplectic_rep_of_clifford_circuit(
            compiled, pspec=self.options.get('pspec', None)
        )
        sym_state, phase_state = symplectic.prep_stabilizer_state(self.fixture.n)
        sym_state, phase_state = symplectic.apply_clifford_to_stabilizer_state(
            self.fixture.clifford_sym, self.fixture.clifford_phase, sym_state, phase_state
        )
        sym_out, phase_out = symplectic.apply_clifford_to_stabilizer_state(
            sym_compiled, phase_compiled, sym_state, phase_state
        )

        # This asserts that a particular stabilizer propagation yields the expected result -
        #  the all-0 state.  This test preparation, acting-on, and measurment of stabilizer states.
        for i in range(self.fixture.n):
            self.assertAlmostEqual(
                symplectic.pauli_z_measurement(sym_out, phase_out, i)[1],
                0.
            )


class CompileStabilizerCOCAGE1QTester(CompileStabilizerBase, BaseCase):
    def setUp(self):
        super(CompileStabilizerCOCAGE1QTester, self).setUp()
        self.fixture = fixture_1Q
        self.options.update(
            pspec=self.fixture.pspec,
            absolute_compilation=self.fixture.clifford_abs,
            paulieq_compilation=self.fixture.clifford_peq,
            algorithm='COCAGE',
            paulirandomize=False
        )


class CompileStabilizerCOCAGE2QTester(CompileStabilizerCOCAGE1QTester):
    def setUp(self):
        super(CompileStabilizerCOCAGE2QTester, self).setUp()
        self.fixture = fixture_2Q
        self.options.update(
            pspec=self.fixture.pspec,
            absolute_compilation=self.fixture.clifford_abs,
            paulieq_compilation=self.fixture.clifford_peq,
        )


class CompileStabilizerROCAGE1QTester(CompileStabilizerCOCAGE1QTester):
    def setUp(self):
        super(CompileStabilizerROCAGE1QTester, self).setUp()
        self.options.update(
            algorithm='ROCAGE',
            paulirandomize=True
        )


class CompileStabilizerROCAGE2QTester(CompileStabilizerROCAGE1QTester):
    def setUp(self):
        super(CompileStabilizerROCAGE2QTester, self).setUp()
        self.fixture = fixture_2Q
        self.options.update(
            pspec=self.fixture.pspec,
            absolute_compilation=self.fixture.clifford_abs,
            paulieq_compilation=self.fixture.clifford_peq,
        )


class CompileStabilizer1QSubsetTester(CompileStabilizerCOCAGE1QTester):
    def setUp(self):
        super(CompileStabilizer1QSubsetTester, self).setUp()
        self.options.update(
            pspec=fixture_3Q.pspec,
            absolute_compilation=fixture_3Q.clifford_abs,
            paulieq_compilation=fixture_3Q.clifford_peq,
            qubit_labels=['Q1', ],
            algorithm='COiCAGE',
            paulirandomize=False
        )


class CompileStabilizer2QSubsetTester(CompileStabilizer1QSubsetTester):
    def setUp(self):
        super(CompileStabilizer2QSubsetTester, self).setUp()
        self.fixture = fixture_2Q
        self.options.update(
            qubit_labels=['Q1', 'Q2']
        )


class CompileConditionalSymplecticRingTester(BaseCase):
    # Regression coverage for compile_conditional_symplectic(..., check=True) on a restricted
    # (ring) geometry. The check path previously crashed with "Cannot edit a read-only circuit!"
    # because it copied the precircuit without editable=True.
    def setUp(self):
        super(CompileConditionalSymplecticRingTester, self).setUp()
        self.qubit_labels = ['Q0', 'Q1', 'Q2', 'Q3']
        self.pspec = QubitProcessorSpec(
            4, gate_names=['Gxpi2', 'Gxmpi2', 'Gypi2', 'Gympi2', 'Gcphase'],
            qubit_labels=self.qubit_labels, geometry='ring')

    def test_compile_conditional_symplectic_check_on_ring(self):
        rand_state = np.random.RandomState(2021)
        n = 4
        for _ in range(20):
            s, _p = symplectic.random_clifford(n, rand_state=rand_state)
            # check=True exercises the editable-copy path; it must not raise.
            circuit, precircuit = compilers.compile_conditional_symplectic(
                s, self.pspec, qubit_labels=self.qubit_labels, check=True, rand_state=rand_state)
            # The conditional-symplectic guarantee: prefixing precircuit reproduces the RHS of `s`.
            checkcircuit = precircuit.copy(editable=True)
            checkcircuit.append_circuit_inplace(circuit)
            scheck, _pcheck = symplectic.symplectic_rep_of_clifford_circuit(checkcircuit)
            self.assertArraysEqual(scheck[:, n:2 * n], s[:, n:2 * n])

import numpy as np
from pygsti.baseobjs.label import Label

from pygsti.circuits.circuit import Circuit
from pygsti.modelmembers.operations import ComposedOp, EmbeddedOp
from pygsti.models.localnoisemodel import LocalNoiseModel
from pygsti.models.modelconstruction import create_crosstalk_free_model
from pygsti.processors.processorspec import QubitProcessorSpec
from pygsti.modelmembers.operations import StaticArbitraryOp, ExpErrorgenOp, LindbladErrorgen
from ..util import BaseCase


class LocalNoiseModelInstanceTester(BaseCase):

    def setUp(self):
        nQubits = 2
        self.pspec_2Q = QubitProcessorSpec(nQubits, ('Gx', 'Gy', 'Gcnot'), geometry="line",
                                           qubit_labels=['qb{}'.format(i) for i in range(nQubits)])
        nQubits = 4
        self.pspec_4Q = QubitProcessorSpec(nQubits, ('Gx', 'Gy', 'Gcnot'), geometry="line",
                                           qubit_labels=['qb{}'.format(i) for i in range(nQubits)])
    
    def test_indep_localnoise(self):
        mdl_local = create_crosstalk_free_model(self.pspec_2Q,
                                                ideal_gate_type='H+S', ideal_spam_type='tensor product H+S', independent_gates=True,
                                                ensure_composed_gates=False)

        assert(set(mdl_local.operation_blks['gates'].keys()) == set(
            [('Gx', 'qb0'), ('Gx', 'qb1'), ('Gy', 'qb0'), ('Gy', 'qb1'), ('Gcnot', 'qb0', 'qb1'), ('Gcnot', 'qb1', 'qb0')]))
        assert(set(mdl_local.operation_blks['layers'].keys()) == set(
            [('Gx', 'qb0'), ('Gx', 'qb1'), ('Gy', 'qb0'), ('Gy', 'qb1'), ('Gcnot', 'qb0', 'qb1'), ('Gcnot', 'qb1', 'qb0')]))
        test_circuit = ([('Gx', 'qb0'), ('Gy', 'qb1')], ('Gcnot', 'qb0', 'qb1'), [('Gx', 'qb1'), ('Gy', 'qb0')])
        self.assertAlmostEqual(sum(mdl_local.probabilities(test_circuit).values()), 1.0)
        self.assertEqual(mdl_local.num_params, 108)

    def test_dep_localnoise(self):
        mdl_local = create_crosstalk_free_model(self.pspec_2Q,
                                                ideal_gate_type='H+S', ideal_spam_type='exp(H+S)', independent_gates=False,
                                                ensure_composed_gates=False)

        assert(set(mdl_local.operation_blks['gates'].keys()) == set(["Gx", "Gy", "Gcnot"]))
        assert(set(mdl_local.operation_blks['layers'].keys()) == set(
            [('Gx', 'qb0'), ('Gx', 'qb1'), ('Gy', 'qb0'), ('Gy', 'qb1'), ('Gcnot', 'qb0', 'qb1'), ('Gcnot', 'qb1', 'qb0')]))
        test_circuit = ([('Gx', 'qb0'), ('Gy', 'qb1')], ('Gcnot', 'qb0', 'qb1'), [('Gx', 'qb1'), ('Gy', 'qb0')])
        self.assertAlmostEqual(sum(mdl_local.probabilities(test_circuit).values()), 1.0)
        self.assertEqual(mdl_local.num_params, 66)

    def test_localnoise_1Q_global_idle(self):
        nQubits = 2
        noisy_idle = StaticArbitraryOp(np.array([[1, 0, 0, 0],
                                                 [0, 0.9, 0, 0],
                                                 [0, 0, 0.9, 0],
                                                 [0, 0, 0, 0.9]], 'd'))
        pspec_2Q = QubitProcessorSpec(nQubits, ('Gx', 'Gy', 'Gcnot', 'Gidle'), geometry="line",
                                      availability={'Gidle': [('qb0',), ('qb1',)]},
                                      qubit_labels=['qb{}'.format(i) for i in range(nQubits)])

        mdl_local = create_crosstalk_free_model(pspec_2Q, {'Gidle': noisy_idle},
                                                ideal_gate_type='static', independent_gates=False,
                                                ensure_composed_gates=False, implicit_idle_mode='add_global')

        assert(set(mdl_local.operation_blks['gates'].keys()) == set(["Gx", "Gy", "Gcnot", "Gidle"]))
        assert(set(mdl_local.operation_blks['layers'].keys()) == set(
            [('Gx', 'qb0'), ('Gx', 'qb1'), ('Gy', 'qb0'), ('Gy', 'qb1'), ('Gcnot', 'qb0', 'qb1'), ('Gcnot', 'qb1', 'qb0'), ('Gidle', 'qb0'), ('Gidle', 'qb1'), '{auto_global_idle}']))
        test_circuit = (('Gx', 'qb0'), ('Gcnot', 'qb0', 'qb1'), [], [('Gx', 'qb1'), ('Gy', 'qb0')])
        self.assertAlmostEqual(sum(mdl_local.probabilities(test_circuit).values()), 1.0)
        self.assertAlmostEqual(mdl_local.probabilities(test_circuit)['00'], 0.3576168)
        self.assertEqual(mdl_local.num_params, 0)

        op = mdl_local.circuit_layer_operator(Label('Gx', 'qb1'))
        ref_op = ComposedOp([
            ComposedOp([
                EmbeddedOp(mdl_local.state_space, ('qb0',), mdl_local.operation_blks['gates']['Gidle']),
                EmbeddedOp(mdl_local.state_space, ('qb1',), mdl_local.operation_blks['gates']['Gidle']),
            ]), # Global idle
            EmbeddedOp(mdl_local.state_space, ('qb1',), mdl_local.operation_blks['gates']['Gx']) # Gx operation
        ])
        ref_op2 = ComposedOp([
            mdl_local.operation_blks['layers']['{auto_global_idle}'], # Global idle
            EmbeddedOp(mdl_local.state_space, ('qb1',), mdl_local.operation_blks['gates']['Gx']) # Gx operation
        ])
        self.assertEqual(str(op), str(ref_op))
        self.assertEqual(str(op), str(ref_op2))

    def test_localnoise_NQ_global_idle(self):
        nQubits = 2
        noisy_idle = 0.9 * np.identity(4**nQubits, 'd')
        noisy_idle[0, 0] = 1.0
        noisy_idle = ExpErrorgenOp(LindbladErrorgen.from_operation_matrix(noisy_idle, "H+S"))

        pspec_2Q = QubitProcessorSpec(nQubits, ('Gx', 'Gy', 'Gcnot', 'Gidle'), geometry="line",
                                      qubit_labels=['qb{}'.format(i) for i in range(nQubits)])

        mdl_local = create_crosstalk_free_model(pspec_2Q, {'Gidle': noisy_idle},
                                                ideal_gate_type='H+S', ideal_spam_type="exp(H+S)",
                                                independent_gates=False, ensure_composed_gates=False,
                                                implicit_idle_mode='add_global')

        self.assertEqual(set(mdl_local.operation_blks['gates'].keys()), set(["Gx", "Gy", "Gcnot", "Gidle"]))
        assert(set(mdl_local.operation_blks['layers'].keys()) == set(
            [('Gx', 'qb0'), ('Gx', 'qb1'), ('Gy', 'qb0'), ('Gy', 'qb1'), ('Gcnot', 'qb0', 'qb1'), ('Gcnot', 'qb1', 'qb0'), 'Gidle']))
        test_circuit = (('Gx', 'qb0'), ('Gcnot', 'qb0', 'qb1'), [], [('Gx', 'qb1'), ('Gy', 'qb0')])
        self.assertAlmostEqual(sum(mdl_local.probabilities(test_circuit).values()), 1.0)
        self.assertAlmostEqual(mdl_local.probabilities(test_circuit)['00'], 0.414025)
        self.assertEqual(mdl_local.num_params, 96)  # was 144 when used old H+S+A param mode (now removed)

        op = mdl_local.circuit_layer_operator(Label('Gx', 'qb1'))
        ref_op = ComposedOp([
            mdl_local.operation_blks['layers'][Label('Gidle')], # Global idle
            EmbeddedOp(mdl_local.state_space, ('qb1',), mdl_local.operation_blks['gates']['Gx']) # Gx operation
        ])
        self.assertEqual(str(op), str(ref_op))

    def test_marginalized_povm(self):
        mdl_local = create_crosstalk_free_model(self.pspec_4Q,
                                                ideal_gate_type='H+S', independent_gates=True,
                                                ensure_composed_gates=False)

        c = Circuit( [('Gx','qb0'),('Gx','qb1'),('Gx','qb2'),('Gx','qb3')], num_lines=4)
        prob = mdl_local.probabilities(c)
        self.assertEqual(len(prob), 16) # Full 4 qubit space

        c2 = Circuit( [('Gx','qb0'),('Gx','qb1')], num_lines=2)
        prob2 = mdl_local.probabilities(c2)
        self.assertEqual(len(prob2), 4) # Full 2 qubit space

        c3 = Circuit( [('Gx','qb0'),('Gx','qb1')], editable=True)
        c3.insert_idling_lines_inplace(None, ['qb2', 'qb3'])
        c3.done_editing()
        prob3 = mdl_local.probabilities(c3)
        self.assertEqual(len(prob3), 16) # Full 4 qubit space




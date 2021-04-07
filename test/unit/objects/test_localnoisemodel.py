import numpy as np

from ..util import BaseCase

from pygsti.objects.circuit import Circuit
from pygsti.objects.label import Label
from pygsti.objects.localnoisemodel import LocalNoiseModel
from pygsti.objects.operation import ComposedDenseOp, EmbeddedDenseOp, StaticDenseOp


class LocalNoiseModelInstanceTester(BaseCase):
    def test_indep_localnoise(self):
        nQubits = 2
        mdl_local = LocalNoiseModel.from_parameterization(
            nQubits, ('Gx', 'Gy', 'Gcnot'), geometry="line",
            qubit_labels=['qb{}'.format(i) for i in range(nQubits)],
            parameterization='H+S', independent_gates=True,
            ensure_composed_gates=False, global_idle=None)

        assert(set(mdl_local.operation_blks['gates'].keys()) == set(
            [('Gx', 'qb0'), ('Gx', 'qb1'), ('Gy', 'qb0'), ('Gy', 'qb1'), ('Gcnot', 'qb0', 'qb1'), ('Gcnot', 'qb1', 'qb0')]))
        assert(set(mdl_local.operation_blks['layers'].keys()) == set(
            [('Gx', 'qb0'), ('Gx', 'qb1'), ('Gy', 'qb0'), ('Gy', 'qb1'), ('Gcnot', 'qb0', 'qb1'), ('Gcnot', 'qb1', 'qb0')]))
        test_circuit = ([('Gx', 'qb0'), ('Gy', 'qb1')], ('Gcnot', 'qb0', 'qb1'), [('Gx', 'qb1'), ('Gy', 'qb0')])
        self.assertAlmostEqual(sum(mdl_local.probabilities(test_circuit).values()), 1.0)
        self.assertEqual(mdl_local.num_params, 108)

    def test_dep_localnoise(self):
        nQubits = 2
        mdl_local = LocalNoiseModel.from_parameterization(
            nQubits, ('Gx', 'Gy', 'Gcnot'), geometry="line",
            qubit_labels=['qb{}'.format(i) for i in range(nQubits)],
            parameterization='H+S', independent_gates=False,
            ensure_composed_gates=False, global_idle=None)

        assert(set(mdl_local.operation_blks['gates'].keys()) == set(["Gx", "Gy", "Gcnot"]))
        assert(set(mdl_local.operation_blks['layers'].keys()) == set(
            [('Gx', 'qb0'), ('Gx', 'qb1'), ('Gy', 'qb0'), ('Gy', 'qb1'), ('Gcnot', 'qb0', 'qb1'), ('Gcnot', 'qb1', 'qb0')]))
        test_circuit = ([('Gx', 'qb0'), ('Gy', 'qb1')], ('Gcnot', 'qb0', 'qb1'), [('Gx', 'qb1'), ('Gy', 'qb0')])
        self.assertAlmostEqual(sum(mdl_local.probabilities(test_circuit).values()), 1.0)
        self.assertEqual(mdl_local.num_params, 66)

    def test_localnoise_1Q_global_idle(self):
        nQubits = 2
        noisy_idle = np.array([[1, 0, 0, 0],
                               [0, 0.9, 0, 0],
                               [0, 0, 0.9, 0],
                               [0, 0, 0, 0.9]], 'd')

        mdl_local = LocalNoiseModel.from_parameterization(
            nQubits, ('Gx', 'Gy', 'Gcnot'), geometry="line",
            qubit_labels=['qb{}'.format(i) for i in range(nQubits)],
            parameterization='static', independent_gates=False,
            ensure_composed_gates=False, global_idle=noisy_idle)

        assert(set(mdl_local.operation_blks['gates'].keys()) == set(["Gx", "Gy", "Gcnot", "1QGlobalIdle"]))
        assert(set(mdl_local.operation_blks['layers'].keys()) == set(
            [('Gx', 'qb0'), ('Gx', 'qb1'), ('Gy', 'qb0'), ('Gy', 'qb1'), ('Gcnot', 'qb0', 'qb1'), ('Gcnot', 'qb1', 'qb0'), 'globalIdle']))
        test_circuit = (('Gx', 'qb0'), ('Gcnot', 'qb0', 'qb1'), [], [('Gx', 'qb1'), ('Gy', 'qb0')])
        self.assertAlmostEqual(sum(mdl_local.probabilities(test_circuit).values()), 1.0)
        self.assertAlmostEqual(mdl_local.probabilities(test_circuit)['00'], 0.3576168)
        self.assertEqual(mdl_local.num_params, 0)

        op = mdl_local.circuit_layer_operator(Label('Gx', 'qb1'))
        ref_op = ComposedDenseOp([
            ComposedDenseOp([
                EmbeddedDenseOp(mdl_local.state_space_labels, ('qb0',), mdl_local.operation_blks['gates']['1QGlobalIdle']),
                EmbeddedDenseOp(mdl_local.state_space_labels, ('qb1',), mdl_local.operation_blks['gates']['1QGlobalIdle']),
            ]), # Global idle
            EmbeddedDenseOp(mdl_local.state_space_labels, ('qb1',), mdl_local.operation_blks['gates']['Gx']) # Gx operation
        ])
        ref_op2 = ComposedDenseOp([
            mdl_local.operation_blks['layers']['globalIdle'], # Global idle
            EmbeddedDenseOp(mdl_local.state_space_labels, ('qb1',), mdl_local.operation_blks['gates']['Gx']) # Gx operation
        ])
        self.assertEqual(str(op), str(ref_op))
        self.assertEqual(str(op), str(ref_op2))

    def test_localnoise_NQ_global_idle(self):
        nQubits = 2
        noisy_idle = 0.9 * np.identity(4**nQubits, 'd')
        noisy_idle[0, 0] = 1.0

        mdl_local = LocalNoiseModel.from_parameterization(
            nQubits, ('Gx', 'Gy', 'Gcnot'), geometry="line",
            qubit_labels=['qb{}'.format(i) for i in range(nQubits)],
            parameterization='H+S+A', independent_gates=False,
            ensure_composed_gates=False, global_idle=noisy_idle)

        self.assertEqual(set(mdl_local.operation_blks['gates'].keys()), set(["Gx", "Gy", "Gcnot"]))
        assert(set(mdl_local.operation_blks['layers'].keys()) == set(
            [('Gx', 'qb0'), ('Gx', 'qb1'), ('Gy', 'qb0'), ('Gy', 'qb1'), ('Gcnot', 'qb0', 'qb1'), ('Gcnot', 'qb1', 'qb0'), 'globalIdle']))
        test_circuit = (('Gx', 'qb0'), ('Gcnot', 'qb0', 'qb1'), [], [('Gx', 'qb1'), ('Gy', 'qb0')])
        self.assertAlmostEqual(sum(mdl_local.probabilities(test_circuit).values()), 1.0)
        self.assertAlmostEqual(mdl_local.probabilities(test_circuit)['00'], 0.414025)
        self.assertEqual(mdl_local.num_params, 144)

        op = mdl_local.circuit_layer_operator(Label('Gx', 'qb1'))
        ref_op = ComposedDenseOp([
            mdl_local.operation_blks['layers'][Label('globalIdle')], # Global idle
            EmbeddedDenseOp(mdl_local.state_space_labels, ('qb1',), mdl_local.operation_blks['gates']['Gx']) # Gx operation
        ])
        self.assertEqual(str(op), str(ref_op))
    
    def test_localnoise_1Q_padded_idle(self):
        nQubits = 2
        noisy_idle = np.array([[1, 0, 0, 0],
                               [0, 0.9, 0, 0],
                               [0, 0, 0.9, 0],
                               [0, 0, 0, 0.9]], 'd')

        mdl_local = LocalNoiseModel.from_parameterization(
            nQubits, ('Gx', 'Gy', 'Gcnot'), geometry="line",
            qubit_labels=['qb{}'.format(i) for i in range(nQubits)],
            parameterization='static', independent_gates=False,
            ensure_composed_gates=False, padded_idle=noisy_idle)

        self.assertEqual(set(mdl_local.operation_blks['gates'].keys()), set(["Gx", "Gy", "Gcnot", "1QPadIdle"]))
        self.assertEqual(set(mdl_local.operation_blks['layers'].keys()),
            set([('Gx', 'qb0'), ('Gx', 'qb1'), ('Gy', 'qb0'), ('Gy', 'qb1'), ('Gcnot', 'qb0', 'qb1'), ('Gcnot', 'qb1', 'qb0')]))
        self.assertEqual(set(mdl_local.factories['layers'].keys()), set(['padIdle']))
        
        test_circuit = (('Gx', 'qb0'), ('Gcnot', 'qb0', 'qb1'), [], [('Gx', 'qb1'), ('Gy', 'qb0')])
        self.assertAlmostEqual(sum(mdl_local.probabilities(test_circuit).values()), 1.0)
        # Note that this probability is different than the global idle, since noise is not applied to active gates
        self.assertAlmostEqual(mdl_local.probabilities(test_circuit)['00'], 0.432249999)
        self.assertEqual(mdl_local.num_params, 0)
    
        test_circuit = ([('padIdle', 'qb0'), ('Gx', 'qb0')], [('padIdle', 'qb0'), ('padIdle', 'qb1'), ('Gcnot', 'qb0', 'qb1')],
            [('padIdle', 'qb0'), ('padIdle', 'qb1')], [('padIdle', 'qb0'), ('padIdle', 'qb1'), ('Gx', 'qb1'), ('Gy', 'qb0')])
        self.assertAlmostEqual(sum(mdl_local.probabilities(test_circuit).values()), 1.0)
        # This now should be the same circuit as one from test_localnoise_1Q_global_idle
        # Since I've added the padded idle before non-idle gates as well (i.e. all gates have it prepended)
        self.assertAlmostEqual(mdl_local.probabilities(test_circuit)['00'], 0.3576168)
        self.assertEqual(mdl_local.num_params, 0)

        op = mdl_local.circuit_layer_operator(Label('Gx', 'qb1'))
        # Order matters here for test, we start with base operation and then idles are added on (probably in order)
        ref_op = ComposedDenseOp([
            EmbeddedDenseOp(mdl_local.state_space_labels, ('qb1',), mdl_local.operation_blks['gates']['Gx']), # Gx operation
            EmbeddedDenseOp(mdl_local.state_space_labels, ('qb0',), mdl_local.operation_blks['gates']['1QPadIdle']), # Idle on qb0
        ])
        self.assertEqual(str(op), str(ref_op))

    def test_marginalized_povm(self):
        nQubits = 4
        mdl_local = LocalNoiseModel.from_parameterization(
            nQubits, ('Gx', 'Gy', 'Gcnot'), geometry="line",
            qubit_labels=['qb{}'.format(i) for i in range(nQubits)],
            parameterization='H+S', independent_gates=True,
            ensure_composed_gates=False, global_idle=None)

        c = Circuit( [('Gx','qb0'),('Gx','qb1'),('Gx','qb2'),('Gx','qb3')], num_lines=4)
        prob = mdl_local.probabilities(c)
        self.assertEqual(len(prob), 16) # Full 4 qubit space

        c2 = Circuit( [('Gx','qb0'),('Gx','qb1')], num_lines=2)
        prob2 = mdl_local.probabilities(c2)
        self.assertEqual(len(prob2), 4) # Full 2 qubit space

        c3 = Circuit( [('Gx','qb0'),('Gx','qb1')])
        c3.insert_idling_lines_inplace(None, ['qb2', 'qb3'])
        prob3 = mdl_local.probabilities(c3)
        self.assertEqual(len(prob3), 16) # Full 4 qubit space




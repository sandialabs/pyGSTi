from unittest import mock

import networkx
import numpy as np

from pygsti.baseobjs import Label as L
from pygsti.processors import QubitProcessorSpec as QPS
from ..util import BaseCase

from pygsti.algorithms import randomcircuit as _rc

class RandomCircuitTest(BaseCase):

    def test_random_circuit(self):

        n_qubits = 4
        qubit_labels = ['Q0','Q1','Q2','Q3']
        gate_names = ['Gxpi2', 'Gxmpi2', 'Gypi2', 'Gympi2', 'Gcphase']
        availability = {'Gcphase':[('Q0','Q1'), ('Q1','Q2'), ('Q2','Q3'), ('Q3','Q0')]}
        pspec = QPS(n_qubits, gate_names, availability=availability, qubit_labels=qubit_labels)
        depth = 10
        circuit = _rc.create_random_circuit(pspec, depth, sampler='Qelimination', samplerargs=[0.5])
        circuit = _rc.create_random_circuit(pspec, depth, sampler='edgegrab', samplerargs=[0.25])


        C2QGs1 = [] #  A list containing no 2-qubit gates is an acceptable set of compatible 2-qubit gates.
        C2QGs2 = [L('Gcphase',('Q0','Q1')),]
        C2QGs3 = [L('Gcphase',('Q1','Q2')),]
        C2QGs4 = [L('Gcphase',('Q2','Q3')),]
        C2QGs5 = [L('Gcphase',('Q3','Q0')),]
        C2QGs6 = [L('Gcphase',('Q0','Q1')), L('Gcphase',('Q2','Q3')),]
        C2QGs7 = [L('Gcphase',('Q1','Q2')), L('Gcphase',('Q3','Q0')),]

        co2Qgates = [C2QGs1, C2QGs2, C2QGs3, C2QGs4, C2QGs5, C2QGs6, C2QGs7]
        co2Qgatesprob = [0.5, 0.125, 0.125, 0.125, 0.125, 0, 0]
        twoQprob = 1

        samplerargs = [co2Qgates, co2Qgatesprob, twoQprob]
        circuit = _rc.create_random_circuit(pspec, depth, sampler='co2Qgates', samplerargs=samplerargs)

        co2Qgates = [C2QGs1,[C2QGs2,C2QGs3,C2QGs4, C2QGs5]]
        co2Qgatesprob = [0.5,0.5]
        twoQprob = 1
        samplerargs = [co2Qgates,]
        circuit = _rc.create_random_circuit(pspec, depth, sampler='co2Qgates', samplerargs=samplerargs)


class LayerSamplerTester(BaseCase):

    def test_edgegrab(self):

        n = 4
        qs = ['Q'+str(i) for i in range(n)]
        ring = [('Q'+str(i),'Q'+str(i+1)) for i in range(n-1)]

        gateset1 =  ['Gcphase'] + ['Gc'+str(i) for i in range(24)]
        gateset2 =  ['Gcphase'] + ['Gxpi2', 'Gzr']
        gateset3 =  ['Gczr'] + ['Gxpi2', 'Gzr']

        pspec1 = QPS(n, gateset1, availability={'Gcphase':ring}, qubit_labels=qs)
        pspec2 = QPS(n, gateset2, availability={'Gcphase':ring}, qubit_labels=qs)
        pspec3 = QPS(n, gateset3, availability={'Gczr':ring}, qubit_labels=qs)

        q_set = ('Q0', 'Q1', 'Q2')

        l1 = _rc.sample_circuit_layer_by_edgegrab(pspec1, qubit_labels=q_set, two_q_gate_density=0.25, one_q_gate_names=None,
                        gate_args_lists={'Gczr':[('-0.1',),('+0.1',)]})

        l2 = _rc.sample_circuit_layer_by_edgegrab(pspec2, qubit_labels=q_set,  two_q_gate_density=0.25, one_q_gate_names=[],
         gate_args_lists=None)

        l3 = _rc.sample_circuit_layer_by_edgegrab(pspec3, qubit_labels=q_set,  two_q_gate_density=0.25, one_q_gate_names=[],
                gate_args_lists=None)

        l4 = _rc.sample_circuit_layer_by_edgegrab(pspec3, qubit_labels=q_set,  two_q_gate_density=0.25, one_q_gate_names=['Gxpi2',],
                gate_args_lists={'Gczr':[('-0.1',),('+0.1',)]})

    def test_qelimination(self):

        n = 4
        qs = ['Q'+str(i) for i in range(n)]
        ring = [('Q'+str(i),'Q'+str(i+1)) for i in range(n-1)]

        gateset1 =  ['Gcphase'] + ['Gc'+str(i) for i in range(24)]
        gateset2 =  ['Gcphase'] + ['Gxpi2', 'Gzr']
        gateset3 =  ['Gczr'] + ['Gxpi2', 'Gzr']

        pspec1 = QPS(n, gateset1, availability={'Gcphase':ring}, qubit_labels=qs)
        pspec2 = QPS(n, gateset2, availability={'Gcphase':ring}, qubit_labels=qs)
        pspec3 = QPS(n, gateset3, availability={'Gczr':ring}, qubit_labels=qs)

        q_set = ('Q0', 'Q1', 'Q2')

        l1 = _rc.sample_circuit_layer_by_q_elimination(pspec1, qubit_labels=q_set, two_q_prob=0.5)

        l2 = _rc.sample_circuit_layer_by_q_elimination(pspec2, qubit_labels=q_set, two_q_prob=0.5)

        l3 = _rc.sample_circuit_layer_by_q_elimination(pspec3, qubit_labels=q_set, two_q_prob=0.5)

        l4 = _rc.sample_circuit_layer_by_q_elimination(pspec3, qubit_labels=q_set, two_q_prob=0.5)


QS = ['Q0', 'Q1', 'Q2', 'Q3']
INFEASIBLE_RE = r'xi of at most 2/3 always fits'


def two_qubit_gates_by_layer(circuit):
    """The two-qubit gate labels of each layer of `circuit`, as a list of lists."""
    return [[g for g in layer.components if len(g.sslbls) == 2] for layer in circuit]


class FixedFeatureValueCircuitTest(BaseCase):

    def setUp(self):
        self.line = QPS(4, ['Gi', 'Gxpi2', 'Gypi2', 'Gcphase'], geometry='line', qubit_labels=QS)
        self.oneway_pairs = [('Q0', 'Q1'), ('Q1', 'Q2'), ('Q2', 'Q3')]
        self.oneway = QPS(4, ['Gi', 'Gxpi2', 'Gypi2', 'Gcnot'], availability={'Gcnot': self.oneway_pairs},
                          qubit_labels=QS)
        self.f = _rc.create_random_circuit_with_fixed_feature_values

    def test_full_density_fits_on_even_line(self):
        for pspec in (self.line, self.oneway):
            for seed in range(20):
                c = self.f(pspec, 4, 10, 1.0, rand_state=np.random.RandomState(seed))
                layers = two_qubit_gates_by_layer(c)
                self.assertEqual(len(layers), 10)
                for gates in layers:
                    self.assertEqual(len(gates), 2)
                    self.assertEqual({frozenset(g.sslbls) for g in gates},
                                     {frozenset(['Q0', 'Q1']), frozenset(['Q2', 'Q3'])})
                    if pspec is self.oneway:
                        for g in gates:
                            self.assertIn(tuple(g.sslbls), self.oneway_pairs)

    def test_transition_zone_succeeds(self):
        for xi in (0.8, 0.9, 1.0):
            target = xi * 4 * 20 / 2
            for seed in range(20):
                c = self.f(self.line, 4, 20, xi, rand_state=np.random.RandomState(seed))
                n = sum(len(gates) for gates in two_qubit_gates_by_layer(c))
                self.assertIn(n, (int(np.floor(target)), int(np.ceil(target))))

    def test_infeasible_density_raises(self):
        for seed in range(5):
            with self.assertRaisesRegex(ValueError, INFEASIBLE_RE):
                self.f(self.line, 3, 4, 1.0, qubit_labels=['Q0', 'Q1', 'Q2'],
                       rand_state=np.random.RandomState(seed))

    def test_rounding_past_the_limit(self):
        outcomes = set()
        for seed in range(20):
            try:
                c = self.f(self.line, 3, 1, 0.9, qubit_labels=['Q0', 'Q1', 'Q2'],
                           rand_state=np.random.RandomState(seed))
            except ValueError as err:
                self.assertRegex(str(err), INFEASIBLE_RE)
                outcomes.add('raised')
            else:
                self.assertEqual(sum(len(gates) for gates in two_qubit_gates_by_layer(c)), 1)
                outcomes.add('returned')
        self.assertEqual(outcomes, {'raised', 'returned'})

    def test_limit_checks_the_drawn_count(self):
        # The greedy layer can fall short (picking only Q1-Q2) and the target 2.25 exceeds the
        # limit 2, so a draw of 2 must upgrade and return; checking the target would raise.
        pspec = QPS(5, ['Gi', 'Gxpi2', 'Gypi2', 'Gcnot'], availability={'Gcnot': self.oneway_pairs},
                    qubit_labels=QS + ['Q4'])
        outcomes = set()
        with mock.patch.object(_rc, '_sample_maximum_matching', wraps=_rc._sample_maximum_matching) as spy:
            for seed in range(20):
                spy.reset_mock()
                try:
                    c = self.f(pspec, 5, 1, 0.9, rand_state=np.random.RandomState(seed))
                except ValueError as err:
                    self.assertRegex(str(err), r'xi of at most 4/5 always fits')
                    outcomes.add('raised')
                else:
                    self.assertEqual(sum(len(gates) for gates in two_qubit_gates_by_layer(c)), 2)
                    outcomes.add('returned after upgrade' if spy.called else 'returned')
        self.assertEqual(outcomes, {'raised', 'returned', 'returned after upgrade'})

    def test_max_feasible_density_on_odd_line(self):
        for seed in range(10):
            c = self.f(self.line, 3, 4, 2 / 3, qubit_labels=['Q0', 'Q1', 'Q2'],
                       rand_state=np.random.RandomState(seed))
            layers = two_qubit_gates_by_layer(c)
            self.assertEqual(len(layers), 4)
            for gates in layers:
                self.assertEqual(len(gates), 1)

    def test_low_density_uses_every_edge(self):
        pairs = set()
        for seed in range(20):
            c = self.f(self.line, 4, 10, 0.5, rand_state=np.random.RandomState(seed))
            for gates in two_qubit_gates_by_layer(c):
                pairs.update(frozenset(g.sslbls) for g in gates)
        self.assertEqual(pairs, {frozenset(['Q0', 'Q1']), frozenset(['Q1', 'Q2']), frozenset(['Q2', 'Q3'])})

    def test_reproducible_with_seed(self):
        for seed in range(5):
            c1 = self.f(self.line, 4, 10, 1.0, rand_state=np.random.RandomState(seed))
            c2 = self.f(self.line, 4, 10, 1.0, rand_state=np.random.RandomState(seed))
            self.assertEqual(c1, c2)

    def test_independent_of_matching_order(self):
        orig = networkx.max_weight_matching

        def reversed_and_flipped(*args, **kwargs):
            return [(v, u) for (u, v) in reversed(list(orig(*args, **kwargs)))]

        for depth, xi in ((10, 1.0), (20, 0.9)):
            for seed in range(5):
                expected = self.f(self.line, 4, depth, xi, rand_state=np.random.RandomState(seed))
                with mock.patch('networkx.max_weight_matching', side_effect=reversed_and_flipped):
                    actual = self.f(self.line, 4, depth, xi, rand_state=np.random.RandomState(seed))
                self.assertEqual(actual, expected)

    def test_maximum_matching_helper_is_random(self):
        q3 = ['Q0', 'Q1', 'Q2']
        edges = [e for e in self.line.compute_2Q_connectivity().edges() if all(q in q3 for q in e)]
        self.assertEqual(edges, [('Q0', 'Q1'), ('Q1', 'Q0'), ('Q1', 'Q2'), ('Q2', 'Q1')])
        graph = networkx.Graph()
        graph.add_nodes_from(q3)
        graph.add_edges_from(edges)
        seen = set()
        for seed in range(10):
            matching = _rc._sample_maximum_matching(graph, edges, np.random.RandomState(seed))
            self.assertEqual(len(matching), 1)
            self.assertIn(matching[0], edges)
            seen.add(matching[0])
        self.assertEqual({frozenset(e) for e in seen}, {frozenset(['Q0', 'Q1']), frozenset(['Q1', 'Q2'])})
        self.assertTrue(any((v, u) in seen for (u, v) in seen))

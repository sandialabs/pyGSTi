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


class EdgegrabOneQubitGateTest(BaseCase):

    def setUp(self):
        ring = [('Q0', 'Q1'), ('Q1', 'Q2'), ('Q2', 'Q3')]
        # Gypi2 is available on Q0 and Q1 only.
        self.het = QPS(4, ['Gcphase', 'Gxpi2', 'Gypi2', 'Gzpi2'],
                       availability={'Gcphase': ring, 'Gypi2': [('Q0',), ('Q1',)]}, qubit_labels=QS)
        self.rot = QPS(4, ['Gczr', 'Gxpi2', 'Gzr'], availability={'Gczr': ring}, qubit_labels=QS)
        self.eg = _rc.sample_circuit_layer_by_edgegrab

    def test_distribution_uses_only_available_gates(self):
        seen = {'Q0': set(), 'Q1': set()}
        for seed in range(20):
            layer = self.eg(self.het, two_q_gate_density=0.0, one_q_gate_names=['Gxpi2', 'Gypi2'],
                            one_q_gate_distribution=[0.5, 0.5], rand_state=np.random.RandomState(seed))
            self.assertEqual(sorted(lbl.qubits[0] for lbl in layer), QS)
            for lbl in layer:
                self.assertEqual(len(lbl.qubits), 1)
                q = lbl.qubits[0]
                if q in ('Q2', 'Q3'):
                    self.assertEqual(lbl.name, 'Gxpi2')
                else:
                    seen[q].add(lbl.name)
        self.assertEqual(seen, {'Q0': {'Gxpi2', 'Gypi2'}, 'Q1': {'Gxpi2', 'Gypi2'}})

    def test_distribution_renormalized_over_available_gates(self):
        names = []
        for seed in range(500):
            layer = self.eg(self.het, two_q_gate_density=0.0, one_q_gate_names=['Gxpi2', 'Gypi2', 'Gzpi2'],
                            one_q_gate_distribution=[0.1, 0.6, 0.3], rand_state=np.random.RandomState(seed))
            names.extend(lbl.name for lbl in layer if lbl.qubits[0] in ('Q2', 'Q3'))
        self.assertEqual(len(names), 1000)
        self.assertAlmostEqual(names.count('Gzpi2') / len(names), 0.75, delta=0.06)

    def test_raises_when_qubit_has_no_usable_gate(self):
        cases = [(['Gypi2'], [1.0]), (['Gxpi2', 'Gypi2'], [0.0, 1.0]), (['Gypi2'], None)]
        for names, dist in cases:
            with self.assertRaisesRegex(ValueError, 'Cannot sample a one-qubit gate for qubit Q2'):
                self.eg(self.het, two_q_gate_density=0.0, one_q_gate_names=names,
                        one_q_gate_distribution=dist, rand_state=np.random.RandomState(0))

    def test_one_qubit_gate_args(self):
        allowed = [('0.1',), ('0.2',)]
        for dist in (None, [0.5, 0.5]):
            args_seen = []
            for seed in range(10):
                layer = self.eg(self.rot, two_q_gate_density=0.0, one_q_gate_names=['Gxpi2', 'Gzr'],
                                gate_args_lists={'Gzr': allowed}, one_q_gate_distribution=dist,
                                rand_state=np.random.RandomState(seed))
                args_seen.extend(tuple(lbl.args) for lbl in layer if lbl.name == 'Gzr')
            self.assertGreater(len(args_seen), 0)
            for args in args_seen:
                self.assertIn(args, allowed)
            self.assertEqual(set(args_seen), set(allowed))

    def test_samplerargs_positional_order(self):
        czr_args = [('0.1',), ('-0.1',)]
        num_czr = 0
        for seed in range(5):
            c = _rc.create_random_circuit(self.rot, 3, sampler='edgegrab',
                                          samplerargs=[0.5, ['Gxpi2'], {'Gczr': czr_args}],
                                          rand_state=np.random.RandomState(seed))
            for layer in c:
                for lbl in layer.components:
                    if len(lbl.qubits) == 1:
                        self.assertEqual(lbl.name, 'Gxpi2')
                    else:
                        self.assertEqual(lbl.name, 'Gczr')
                        self.assertIn(tuple(lbl.args), czr_args)
                        num_czr += 1
        self.assertGreater(num_czr, 0)

        for seed in range(5):
            c = _rc.create_random_circuit(self.rot, 3, sampler='edgegrab',
                                          samplerargs=[0.5, ['Gxpi2', 'Gzr'],
                                                       {'Gczr': [('0.1',)], 'Gzr': [('0.3',)]}, [0.0, 1.0]],
                                          rand_state=np.random.RandomState(seed))
            for layer in c:
                for lbl in layer.components:
                    if len(lbl.qubits) == 1:
                        self.assertEqual(lbl.name, 'Gzr')
                        self.assertEqual(tuple(lbl.args), ('0.3',))

    def test_rejects_bad_distribution(self):
        # Gxpi is a standard gate name, but self.het doesn't have it.
        with self.assertRaisesRegex(ValueError, "the processor spec has no gate named 'Gxpi'"):
            self.eg(self.het, two_q_gate_density=0.0, one_q_gate_names=['Gxpi2', 'Gxpi'],
                    one_q_gate_distribution=[0.5, 0.5], rand_state=np.random.RandomState(0))
        with self.assertRaisesRegex(ValueError, "'Gcphase' is not a one-qubit gate"):
            self.eg(self.het, two_q_gate_density=0.0, one_q_gate_names=['Gxpi2', 'Gcphase'],
                    one_q_gate_distribution=[0.5, 0.5], rand_state=np.random.RandomState(0))
        for dist in ([0.5, 0.6], [1.5, -0.5]):
            with self.assertRaisesRegex(ValueError, 'must be non-negative and sum to 1'):
                self.eg(self.het, two_q_gate_density=0.0, one_q_gate_names=['Gxpi2', 'Gzpi2'],
                        one_q_gate_distribution=dist, rand_state=np.random.RandomState(0))

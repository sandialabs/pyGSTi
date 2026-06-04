import unittest

import numpy as np
import scipy
import itertools
import pygsti.tools.symplectic as st

from pygsti.processors import QubitProcessorSpec
from pygsti.models import modelconstruction as mc
from pygsti.circuits import Circuit
from pygsti.baseobjs.qubitgraph import QubitGraph
from pygsti.baseobjs.label import Label
from ..util import BaseCase, with_temp_path


def save_and_load(obj, pth):
    obj.write(pth + ".json")
    return QubitProcessorSpec.read(pth + '.json')


class ProcessorSpecTester(BaseCase):
    @unittest.skip("REMOVEME")
    def test_construct_with_nonstd_gate_unitary_factory(self):
        nQubits = 2

        def fn(args):
            if args is None: args = (0,)
            a, = args
            sigmaZ = np.array([[1, 0], [0, -1]], 'd')
            return scipy.linalg.expm(1j * float(a) * sigmaZ)

        ps = QubitProcessorSpec(nQubits, ('Gx', 'Gy', 'Gcnot', 'Ga'), nonstd_gate_unitaries={'Ga': fn})
        mdl = mc.create_crosstalk_free_model(ps)

        c = Circuit("Gx:1Ga;0.3:1Gx:1@(0,1)")
        p = mdl.probabilities(c)

        self.assertAlmostEqual(p['00'], 0.08733219254516078)
        self.assertAlmostEqual(p['01'], 0.9126678074548386)

        c2 = Circuit("Gx:1Ga;0.78539816:1Gx:1@(0,1)")  # a clifford: 0.78539816 = pi/4
        p2 = mdl.probabilities(c2)
        self.assertAlmostEqual(p2['00'], 0.5)
        self.assertAlmostEqual(p2['01'], 0.5)

    @with_temp_path
    def test_with_spam(self, pth):
        pspec_defaults = QubitProcessorSpec(4, ['Gxpi2', 'Gypi2'], geometry='line')

        pspec_names = QubitProcessorSpec(4, ['Gxpi2', 'Gypi2'], geometry='line',
                                         prep_names=("rho1", "rho_1100"), povm_names=("Mz",))

        prep_vec = np.zeros(2**4, complex)
        prep_vec[4] = 1.0
        EA = np.zeros(2**4, complex)
        EA[14] = 1.0
        EB = np.zeros(2**4, complex)
        EB[15] = 1.0

        pspec_vecs = QubitProcessorSpec(4, ['Gxpi2', 'Gypi2'], geometry='line',
                                        prep_names=("rhoA", "rhoC"), povm_names=("Ma", "Mc"),
                                        nonstd_preps={'rhoA': "rho0", 'rhoC': prep_vec},
                                        nonstd_povms={'Ma': {'0': "0000", '1': EA},
                                                      'Mc': {'OutA': "0000", 'OutB': [EA, EB]}})

        pspec_defaults = save_and_load(pspec_defaults, pth)
        pspec_names = save_and_load(pspec_names, pth)
        pspec_vecs = save_and_load(pspec_vecs, pth)

    def test_resolved_availability_contradiction(self):
        nQubits = 1
        qubit_labels = [0]
        
        gate_names = ['Ga', 'Gb']
        
        # Define two distinct dummy unitaries for the gates
        Ua = np.array([[1, 0], [0, 1]], 'd')
        Ub = np.array([[0, 1], [1, 0]], 'd')
        
        nonstd_gate_unitaries = {'Ga': Ua, 'Gb': Ub}
        
        # Both gates are available on the same qubit
        availability = {'Ga': [(0,)], 'Gb': [(0,)]}
        
        ps = QubitProcessorSpec(nQubits, gate_names, nonstd_gate_unitaries=nonstd_gate_unitaries, 
                                availability=availability, qubit_labels=qubit_labels)
        
        ga_available = ps.is_available(('Ga', 0))
        gb_available = ps.is_available(('Gb', 0))
        
        self.assertTrue(ga_available and gb_available)

    def test_compute_2Q_connectivity(self):
        qubit_labels = ['q0', 'q1', 'q2']
        gate_names = ['Gcnot']
        availability = {'Gcnot': [('q0', 'q1'), ('q1', 'q2')]}
        
        ps = QubitProcessorSpec(3, gate_names, availability=availability, qubit_labels=qubit_labels)
        
        # Manually create the expected graph
        expected_graph = QubitGraph(qubit_labels)
        expected_graph.add_edge('q0', 'q1')
        expected_graph.add_edge('q1', 'q2')
        
        # The function compute_2Q_connectivity returns a QubitGraph with symmetric edges, so we need to add the reverse edges to our expected graph
        expected_graph.add_edge('q1', 'q0')
        expected_graph.add_edge('q2', 'q1')

        computed_graph = ps.compute_2Q_connectivity()

        self.assertEqual(set(computed_graph.node_names), set(expected_graph.node_names))
        self.assertEqual(set(computed_graph.edges()), set(expected_graph.edges()))

    def test_gate_num_qubits(self):
        ps = QubitProcessorSpec(2, gate_names=['Gx', 'Gcnot'], geometry='line')
        self.assertEqual(ps.gate_num_qubits('Gx'), 1)
        self.assertEqual(ps.gate_num_qubits('Gcnot'), 2)

    def test_rename_gate_inplace(self):
        ps = QubitProcessorSpec(1, gate_names=['Gx', 'Gy'], availability={'Gx': [(0,)], 'Gy': [(0,)]})
        ps.rename_gate_inplace('Gx', 'MyGx')
        self.assertNotIn('Gx', ps.gate_names)
        self.assertIn('MyGx', ps.gate_names)
        self.assertNotIn('Gx', ps.gate_unitaries)
        self.assertIn('MyGx', ps.gate_unitaries)
        self.assertNotIn('Gx', ps.availability)
        self.assertIn('MyGx', ps.availability)

    def test_resolved_availability_modes(self):
        ps = QubitProcessorSpec(3, gate_names=['Gcnot'], availability={'Gcnot': [(0, 1)]}, geometry='line')
        self.assertEqual(ps.resolved_availability('Gcnot', 'tuple'), [(0, 1)])

        avail_fn = ps.resolved_availability('Gcnot', 'function')
        self.assertTrue(avail_fn((0, 1)))
        self.assertFalse(avail_fn((1, 0)))
        self.assertFalse(avail_fn((0, 2)))

    def test_availability_specifiers(self):
        qubit_labels = [0, 1, 2]
        # Test "all-permutations"
        ps_perm = QubitProcessorSpec(3, gate_names=['Gcnot'], availability={'Gcnot': 'all-permutations'}, qubit_labels=qubit_labels)
        self.assertEqual(set(ps_perm.resolved_availability('Gcnot', 'tuple')), set(itertools.permutations(qubit_labels, 2)))

        # Test "all-combinations"
        ps_comb = QubitProcessorSpec(3, gate_names=['Gcnot'], availability={'Gcnot': 'all-combinations'}, qubit_labels=qubit_labels)
        self.assertEqual(set(ps_comb.resolved_availability('Gcnot', 'tuple')), set(itertools.combinations(qubit_labels, 2)))

        # Test "all-edges"
        ps_edges = QubitProcessorSpec(3, gate_names=['Gcnot'], geometry='line', qubit_labels=qubit_labels)
        self.assertEqual(set(ps_edges.resolved_availability('Gcnot', 'tuple')), { (0, 1), (1, 0), (1, 2), (2, 1)})

    def test_available_gatenames(self):
        qubit_labels = [0, 1, 2]
        gate_names = ['Gx', 'Gy', 'Gcnot']
        availability = {'Gx': [(0,)], 'Gy': [(1,)], 'Gcnot': [(0, 1)]}
        ps = QubitProcessorSpec(3, gate_names=gate_names, availability=availability, qubit_labels=qubit_labels)

        self.assertEqual(set(ps.available_gatenames((0,))), {'Gx'})
        self.assertEqual(set(ps.available_gatenames((1,))), {'Gy'})
        self.assertEqual(set(ps.available_gatenames((0, 1))), {'Gx', 'Gy', 'Gcnot'})
        self.assertEqual(set(ps.available_gatenames((2,))), set())

    def test_available_gatelabels(self):
        qubit_labels = [0, 1, 2]
        gate_names = ['Gx', 'Gcnot']
        availability = {'Gx': [(0,), (1,)], 'Gcnot': 'all-permutations'}
        ps = QubitProcessorSpec(3, gate_names=gate_names, availability=availability, qubit_labels=qubit_labels)

        self.assertEqual(set(ps.available_gatelabels('Gx', (0, 1))), {Label('Gx', (0,)), Label('Gx', (1,))})
        self.assertEqual(set(ps.available_gatelabels('Gx', (0, 2))), {Label('Gx', (0,))})
        self.assertEqual(set(ps.available_gatelabels('Gcnot', (0, 1, 2))), set(map(lambda t: Label('Gcnot', t), itertools.permutations([0, 1, 2], 2))))

    def test_compute_ops_on_qudits(self):
        qubit_labels = [0, 1]
        gate_names = ['Gx', 'Gcnot']
        availability = {'Gx': [(0,)], 'Gcnot': [(0, 1)]}
        ps = QubitProcessorSpec(2, gate_names=gate_names, availability=availability, qubit_labels=qubit_labels)

        ops_on_qudits = ps.compute_ops_on_qudits()
        self.assertEqual(ops_on_qudits, {(0,): [Label('Gx', (0,))], (0, 1): [Label('Gcnot', (0, 1))]})

    def test_subset(self):
        qubit_labels = [0, 1, 2]
        gate_names = ['Gx', 'Gy', 'Gcnot']
        availability = {'Gx': [(0,), (1,)], 'Gy': [(1,), (2,)], 'Gcnot': [(0, 1), (1, 2)]}
        ps = QubitProcessorSpec(3, gate_names=gate_names, availability=availability, qubit_labels=qubit_labels)

        subset_ps = ps.subset(gate_names_to_include=['Gx', 'Gcnot'], qubit_labels_to_keep=[0, 1])

        self.assertEqual(subset_ps.gate_names, ('Gx', 'Gcnot'))
        self.assertEqual(subset_ps.qubit_labels, (0, 1))
        self.assertEqual(subset_ps.availability, {'Gx': ((0,), (1,)), 'Gcnot': ((0, 1),)})

    def test_map_qudit_labels(self):
        qubit_labels = [0, 1]
        gate_names = ['Gx', 'Gcnot']
        availability = {'Gx': [(0,)], 'Gcnot': [(0, 1)]}
        ps = QubitProcessorSpec(2, gate_names=gate_names, availability=availability, qubit_labels=qubit_labels)

        mapped_ps = ps.map_qudit_labels({0: 'a', 1: 'b'})

        self.assertEqual(mapped_ps.qubit_labels, ('a', 'b'))
        self.assertEqual(mapped_ps.availability, {'Gx': (('a',),), 'Gcnot': (('a', 'b'),)})

    def test_compute_clifford_symplectic_reps(self):
        # Create a non-Clifford gate unitary
        non_clifford_U = np.array([[1, 0], [0, np.exp(1j * np.pi / 8)]], 'D')

        ps = QubitProcessorSpec(1, gate_names=['Gh', 'Gp', 'Gnc'], 
                                nonstd_gate_unitaries={'Gnc': non_clifford_U})
        
        srep_dict = ps.compute_clifford_symplectic_reps()
        
        internal_srep_dict = st.compute_internal_gate_symplectic_representations()
        
        self.assertIn('Gh', srep_dict)
        self.assertIn('Gp', srep_dict)
        self.assertNotIn('Gnc', srep_dict)
        
        expected_Gh_s, expected_Gh_p = internal_srep_dict['H']
        expected_Gp_s, expected_Gp_p = internal_srep_dict['P']
        
        actual_Gh_s, actual_Gh_p = srep_dict['Gh']
        actual_Gp_s, actual_Gp_p = srep_dict['Gp']
        
        self.assertArraysEqual(actual_Gh_s, expected_Gh_s)
        self.assertArraysEqual(actual_Gh_p, expected_Gh_p)
        self.assertArraysEqual(actual_Gp_s, expected_Gp_s)
        self.assertArraysEqual(actual_Gp_p, expected_Gp_p)

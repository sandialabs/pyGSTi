import unittest

import numpy as np
import scipy

from pygsti.processors import QubitProcessorSpec
from pygsti.models import modelconstruction as mc
from pygsti.circuits import Circuit
from pygsti.baseobjs.qubitgraph import QubitGraph
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


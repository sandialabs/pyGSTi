import unittest

import numpy as np
import scipy

from pygsti.baseobjs.label import Label
from pygsti.processors import QubitProcessorSpec
from pygsti.models import modelconstruction as mc
from pygsti.circuits import Circuit
from pygsti.tools import symplectic
from ..util import BaseCase, with_temp_path


def save_and_load(obj, pth):
    obj.write(pth + ".json")
    return QubitProcessorSpec.read(pth + '.json')


class ProcessorSpecTester(BaseCase):
    def test_argumented_symplectic_rep_factory_maps_label_args(self):
        def unitary_factory(args):
            return np.identity(4, 'd')

        unitary_factory.shape = (4, 4)

        def srep_factory(label):
            if not label.args:
                raise ValueError("Gargp labels must specify the active qubit as their first argument.")
            return symplectic.symplectic_rep_of_clifford_layer(Label('P', label.args[0]), q_labels=label.sslbls)

        pspec = QubitProcessorSpec(
            2, ['Gargp'], nonstd_gate_unitaries={'Gargp': unitary_factory},
            availability={'Gargp': [(0, 1)]}, geometry='line',
            nonstd_gate_symplecticreps={'Gargp': srep_factory},
            gate_arg_label_indices={'Gargp': (0,)})

        label = Label('Gargp', (0, 1), args=(1,))
        expected = symplectic.symplectic_rep_of_clifford_layer(Label('P', 1), q_labels=(0, 1))
        actual = pspec.clifford_symplectic_rep_of(label)
        self.assertArraysAlmostEqual(actual[0], expected[0])
        self.assertArraysAlmostEqual(actual[1], expected[1])

        mapped_label = pspec.map_gate_label_state_space(label, {0: 'Q0', 1: 'Q1'})
        self.assertEqual(mapped_label, Label('Gargp', ('Q0', 'Q1'), args=('Q1',)))

        mapped_pspec = pspec.map_qubit_labels({0: 'Q0', 1: 'Q1'})
        mapped_expected = symplectic.symplectic_rep_of_clifford_layer(
            Label('P', 'Q1'), q_labels=('Q0', 'Q1'))
        mapped_actual = mapped_pspec.clifford_symplectic_rep_of(mapped_label)
        self.assertArraysAlmostEqual(mapped_actual[0], mapped_expected[0])
        self.assertArraysAlmostEqual(mapped_actual[1], mapped_expected[1])

    @with_temp_path
    def test_label_specific_symplectic_rep_serialization(self, pth):
        label = Label('Gargp', (0, 1), args=(1,))
        srep = symplectic.symplectic_rep_of_clifford_layer(Label('P', 1), q_labels=(0, 1))
        pspec = QubitProcessorSpec(
            2, ['Gargp'], nonstd_gate_unitaries={'Gargp': np.identity(4, 'd')},
            availability={'Gargp': [(0, 1)]}, geometry='line',
            nonstd_gate_symplecticreps={label: srep})

        loaded_pspec = save_and_load(pspec, pth)
        loaded_srep = loaded_pspec.clifford_symplectic_rep_of(label)
        self.assertArraysAlmostEqual(loaded_srep[0], srep[0])
        self.assertArraysAlmostEqual(loaded_srep[1], srep[1])

    @with_temp_path
    def test_arity_only_nonstd_gate(self, pth):
        srep = (np.identity(6, dtype=int), np.zeros(6, dtype=int))
        pspec = QubitProcessorSpec(
            3, ['Gglobal'], availability={'Gglobal': [(0, 1, 2)]}, geometry='line',
            nonstd_gate_num_qubits={'Gglobal': 3},
            nonstd_gate_symplecticreps={'Gglobal': srep})

        self.assertEqual(pspec.gate_num_qubits('Gglobal'), 3)
        self.assertNotIn('Gglobal', pspec.gate_unitaries)
        self.assertEqual(pspec.compute_ops_on_qubits()[(0, 1, 2)], [Label('Gglobal', (0, 1, 2))])

        actual_srep = pspec.clifford_symplectic_rep_of(Label('Gglobal', (0, 1, 2)))
        self.assertArraysAlmostEqual(actual_srep[0], srep[0])
        self.assertArraysAlmostEqual(actual_srep[1], srep[1])

        subset_pspec = pspec.subset(['Gglobal'], [0, 1, 2])
        self.assertEqual(subset_pspec.gate_num_qubits('Gglobal'), 3)
        self.assertNotIn('Gglobal', subset_pspec.gate_unitaries)

        mapped_pspec = pspec.map_qubit_labels({0: 'Q0', 1: 'Q1', 2: 'Q2'})
        self.assertEqual(mapped_pspec.gate_num_qubits('Gglobal'), 3)
        self.assertEqual(mapped_pspec.compute_ops_on_qubits()[('Q0', 'Q1', 'Q2')],
                         [Label('Gglobal', ('Q0', 'Q1', 'Q2'))])

        loaded_pspec = save_and_load(pspec, pth)
        self.assertEqual(loaded_pspec.gate_num_qubits('Gglobal'), 3)
        self.assertNotIn('Gglobal', loaded_pspec.gate_unitaries)
        loaded_srep = loaded_pspec.clifford_symplectic_rep_of(Label('Gglobal', (0, 1, 2)))
        self.assertArraysAlmostEqual(loaded_srep[0], srep[0])
        self.assertArraysAlmostEqual(loaded_srep[1], srep[1])

    def test_arity_only_gate_requires_explicit_symplectic_rep(self):
        pspec = QubitProcessorSpec(
            3, ['Gglobal'], availability={'Gglobal': [(0, 1, 2)]}, geometry='line',
            nonstd_gate_num_qubits={'Gglobal': 3})

        with self.assertRaisesRegex(ValueError, "No unitary is available for arity-only gate"):
            pspec.clifford_symplectic_rep_of(Label('Gglobal', (0, 1, 2)))

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

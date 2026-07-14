import numpy as np

from pygsti.baseobjs.label import Label
from pygsti.processors import QubitProcessorSpec
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

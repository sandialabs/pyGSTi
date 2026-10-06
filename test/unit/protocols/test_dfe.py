from pygsti.baseobjs import Label
from pygsti.circuits import Circuit
from pygsti.processors import CliffordCompilationRules as CCR
from pygsti.processors import QubitProcessorSpec as QPS
from pygsti.protocols import dfe as _dfe
from ..util import BaseCase


QUBITS = ('Q0', 'Q1', 'Q2')


def active_qubits(circuit):
    """Line labels that some gate acts on, ignoring the empty layers that dressing may insert."""
    idle = circuit.idling_lines(idle_layer_labels=[Label(())])
    return set(circuit.line_labels) - set(idle)


class DFEDesignTest(BaseCase):

    @classmethod
    def setUpClass(cls):
        gate_names = ['Gi', 'Gxpi2', 'Gxmpi2', 'Gypi2', 'Gympi2', 'Gcphase']
        availability = {'Gcphase': [('Q0', 'Q1'), ('Q1', 'Q2')]}
        cls.pspec = QPS(3, gate_names, availability=availability, qubit_labels=QUBITS, geometry='line')
        cls.compilations = CCR.create_standard(cls.pspec, 'absolute', ('paulis', '1Qcliffords'), verbosity=0)
        # Clifford targets active on (Q0, Q1) and on (Q1, Q2) respectively.
        cls.c_a = Circuit([[('Gxpi2', 'Q0'), ('Gypi2', 'Q1')], [('Gcphase', 'Q0', 'Q1')]], line_labels=QUBITS)
        cls.c_b = Circuit([[('Gypi2', 'Q1'), ('Gxpi2', 'Q2')], [('Gcphase', 'Q1', 'Q2')]], line_labels=QUBITS)
        cls.num_samples = 5

    def make_design(self, circuits=None, **kwargs):
        circuits = [self.c_a, self.c_b] if circuits is None else circuits
        return _dfe.DFEDesign(self.pspec, circuits, self.compilations, self.num_samples, **kwargs)

    def assert_pinned(self, design, idx, qubit_idx):
        q = QUBITS[qubit_idx]
        for circ, meas in zip(design.circuit_lists[idx], design.measurements[idx]):
            self.assertNotIn(q, active_qubits(circ))
            self.assertEqual(meas[qubit_idx], 'I')

    def test_per_circuit_fixed_identity_locations(self):
        design = self.make_design(seed=1, fixed_identity_locations=[[False, False, True], [True, False, False]])
        self.assert_pinned(design, 0, 2)
        self.assert_pinned(design, 1, 0)

        design = self.make_design(seed=1, fixed_identity_locations=[None, [True, False, False]])
        self.assert_pinned(design, 1, 0)

    def test_idle_shortcut_matches_explicit_masks(self):
        d_idle = self.make_design(seed=7, fixed_identity_locations='idle')
        d_explicit = self.make_design(seed=7, fixed_identity_locations=[[False, False, True], [True, False, False]])
        self.assertEqual(d_idle.circuit_lists, d_explicit.circuit_lists)
        self.assertEqual(d_idle.measurements, d_explicit.measurements)
        self.assertEqual(d_idle.signs, d_explicit.signs)

    def test_default_matches_none_list(self):
        d_default = self.make_design(seed=3)
        d_none = self.make_design(seed=3, fixed_identity_locations=[None, None])
        self.assertEqual(d_default.circuit_lists, d_none.circuit_lists)

    def test_rejects_bad_fixed_identity_locations(self):
        with self.assertRaises(ValueError):
            self.make_design(seed=0, fixed_identity_locations=[[False, False, True]])
        with self.assertRaisesRegex(ValueError, "'idle'"):
            self.make_design(seed=0, fixed_identity_locations='active')
        with self.assertRaises(ValueError):
            self.make_design(circuits=[self.c_a, self.c_b, self.c_a], seed=0,
                             fixed_identity_locations=[False, False, True])

    def test_create_same_measurement_reference_not_implemented(self):
        with self.assertRaises(NotImplementedError):
            self.make_design(seed=0, create_same_measurement_reference=True)

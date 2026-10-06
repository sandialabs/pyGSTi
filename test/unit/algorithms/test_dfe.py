import numpy as np

import pygsti
from pygsti.algorithms import dfe as _dfe
from pygsti.algorithms import randomcircuit as _rc
from pygsti.baseobjs import Label
from pygsti.circuits import Circuit
from pygsti.processors import CliffordCompilationRules as CCR
from pygsti.processors import QubitProcessorSpec as QPS
from pygsti.tools import symplectic as _symp
from ..util import BaseCase


QUBITS = ('Q0', 'Q1', 'Q2')


def active_qubits(circuit):
    """Line labels that some gate acts on, ignoring the empty layers that dressing may insert."""
    idle = circuit.idling_lines(idle_layer_labels=[Label(())])
    return set(circuit.line_labels) - set(idle)


class DFEDressingTest(BaseCase):

    @classmethod
    def setUpClass(cls):
        gate_names = ['Gi', 'Gxpi2', 'Gxmpi2', 'Gypi2', 'Gympi2', 'Gcphase']
        availability = {'Gcphase': [('Q0', 'Q1'), ('Q1', 'Q2')]}
        cls.pspec = QPS(3, gate_names, availability=availability, qubit_labels=QUBITS, geometry='line')
        cls.compilations = CCR.create_standard(cls.pspec, 'absolute', ('paulis', '1Qcliffords'), verbosity=0)
        # A Clifford target that acts on Q0 and Q1 only, with one two-qubit gate.
        cls.target = Circuit([[('Gxpi2', 'Q0'), ('Gypi2', 'Q1')], [('Gcphase', 'Q0', 'Q1')], [('Gympi2', 'Q0')]],
                             line_labels=QUBITS)

    def test_stabilizer_to_all_zs_leaves_identity_qubits_idle(self):
        for stabilizer in (['I', 'X', 'Z'], ['Y', 'I', 'I'], ['I', 'I', 'X']):
            for seed in range(10):
                _, _, circ = _rc._stabilizer_to_all_zs(stabilizer, QUBITS, self.compilations, seed=seed,
                                                       randomize_for_identity=False)
                active = active_qubits(circ)
                for q, p in zip(QUBITS, stabilizer):
                    if p == 'I':
                        self.assertNotIn(q, active)
                    elif p in ('X', 'Y'):
                        self.assertIn(q, active)

    def test_stabilizer_to_all_zs_default_draw_unchanged(self):
        reps = _symp.compute_internal_gate_symplectic_representations()
        for seed in range(20):
            k = np.random.default_rng(seed).choice(np.arange(24))
            s_exp, p_exp = _symp.symplectic_kronecker([reps['C' + str(k)], reps['H'], reps['C0']])
            s, p, _ = _rc._stabilizer_to_all_zs(['I', 'X', 'Z'], QUBITS, self.compilations, seed=seed)
            self.assertTrue(np.array_equal(s, s_exp))
            self.assertTrue(np.array_equal(p, p_exp))

    def test_fixed_identity_qubit_stays_idle(self):
        self.assertGreaterEqual(self.target.two_q_gate_count(), 1)
        for seed in range(20):
            circ, meas, _ = _dfe.sample_dfe_circuit(self.pspec, self.target, self.compilations, seed=seed,
                                                    fixed_identity_locations=[False, False, True])
            self.assertNotIn('Q2', active_qubits(circ))
            self.assertEqual(meas[2], 'I')
            self.assertEqual(active_qubits(circ), active_qubits(self.target))
            self.assertEqual(circ.two_q_gate_count(), self.target.two_q_gate_count())

    def test_default_path_identity_measured_qubit_stays_idle(self):
        num_identity = 0
        for seed in range(40):
            circ, meas, _ = _dfe.sample_dfe_circuit(self.pspec, self.target, self.compilations, seed=seed)
            if meas[2] == 'I':
                num_identity += 1
                self.assertNotIn('Q2', active_qubits(circ))
        self.assertGreaterEqual(num_identity, 3)

    def test_sign_matches_noiseless_simulation(self):
        model = pygsti.models.create_crosstalk_free_model(self.pspec)
        for locs in (None, [False, False, True]):
            for seed in range(5):
                circ, meas, sign = _dfe.sample_dfe_circuit(self.pspec, self.target, self.compilations, seed=seed,
                                                           fixed_identity_locations=locs)
                probs = model.probabilities(circ)
                expval = 0.0
                for outcome, prob in probs.items():
                    bits = outcome[0]
                    parity = sum(int(bits[i]) for i, m in enumerate(meas) if m == 'Z') % 2
                    expval += prob * (-1) ** parity
                self.assertAlmostEqual(expval, sign, delta=1e-6)

    def test_rejects_bad_fixed_identity_locations(self):
        for locs in ([False, True], [False, False, True, False], [True, True, True]):
            with self.assertRaises(ValueError):
                _dfe.sample_dfe_circuit(self.pspec, self.target, self.compilations, seed=0,
                                        fixed_identity_locations=locs)

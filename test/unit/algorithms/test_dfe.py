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


def make_pspec_and_compilations():
    """A 3-qubit line processor with CZ on (Q0, Q1) and (Q1, Q2), and its absolute Clifford compilations."""
    gate_names = ['Gi', 'Gxpi2', 'Gxmpi2', 'Gypi2', 'Gympi2', 'Gcphase']
    availability = {'Gcphase': [('Q0', 'Q1'), ('Q1', 'Q2')]}
    pspec = QPS(3, gate_names, availability=availability, qubit_labels=QUBITS, geometry='line')
    compilations = CCR.create_standard(pspec, 'absolute', ('paulis', '1Qcliffords'), verbosity=0)
    return pspec, compilations


def _final_layers(dressed, target):
    """The layers after `target` inside `dressed`."""
    target_layers = target.layertup
    tup = dressed.layertup
    start = next(i for i in range(len(tup) - len(target_layers) + 1)
                 if tup[i:i + len(target_layers)] == target_layers)
    return tup[start + len(target_layers):]


class DFEDressingTest(BaseCase):

    @classmethod
    def setUpClass(cls):
        cls.pspec, cls.compilations = make_pspec_and_compilations()
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

    def test_sign_matches_noiseless_simulation(self):
        model = pygsti.models.create_crosstalk_free_model(self.pspec)
        for seed in range(10):
            circ, meas, sign = _dfe.sample_dfe_circuit(self.pspec, self.target, self.compilations, seed=seed)
            probs = model.probabilities(circ)
            expval = 0.0
            for outcome, prob in probs.items():
                bits = outcome[0]
                parity = sum(int(bits[i]) for i, m in enumerate(meas) if m == 'Z') % 2
                expval += prob * (-1) ** parity
            self.assertAlmostEqual(expval, sign, delta=1e-6)

    def test_seeded_output_and_reference_keeps_draws(self):
        golden = {
            0: ('[Gypi2:Q0Gympi2:Q1Gxmpi2:Q2][Gxmpi2:Q0Gxmpi2:Q2]Gxmpi2:Q0[Gxpi2:Q0Gypi2:Q1]Gcphase:Q0:Q1Gympi2:Q0[]'
                '@(Q0,Q1,Q2)', ['Z', 'I', 'I'], 1,
                '[Gxmpi2:Q1Gxmpi2:Q2][Gxmpi2:Q1Gxmpi2:Q2][]@(Q0,Q1,Q2)',
                '[Gxmpi2:Q1Gxmpi2:Q2][Gxmpi2:Q1Gxmpi2:Q2]@(Q0,Q1,Q2)'),
            1: ('[Gympi2:Q0Gxpi2:Q1Gxpi2:Q2][Gympi2:Q1Gympi2:Q2][Gxpi2:Q0Gypi2:Q1]Gcphase:Q0:Q1Gympi2:Q0'
                '[Gypi2:Q0Gypi2:Q1Gypi2:Q2][Gxpi2:Q0Gxmpi2:Q1Gxpi2:Q2]Gxmpi2:Q1@(Q0,Q1,Q2)',
                ['Z', 'Z', 'Z'], 1,
                '[Gxpi2:Q0Gympi2:Q1Gxmpi2:Q2][Gympi2:Q0Gympi2:Q2][Gypi2:Q0Gypi2:Q1Gypi2:Q2]'
                '[Gxpi2:Q0Gxmpi2:Q1Gxpi2:Q2]Gxmpi2:Q1@(Q0,Q1,Q2)',
                '[Gxmpi2:Q0Gxmpi2:Q1][Gxmpi2:Q0Gxmpi2:Q1]@(Q0,Q1,Q2)'),
            2: ('Gxmpi2:Q1Gympi2:Q1[Gxpi2:Q0Gypi2:Q1]Gcphase:Q0:Q1Gympi2:Q0[Gypi2:Q0Gypi2:Q1][Gxmpi2:Q0Gxpi2:Q1]'
                'Gxmpi2:Q0@(Q0,Q1,Q2)', ['Z', 'Z', 'Z'], -1,
                '[Gypi2:Q0Gxmpi2:Q1Gxmpi2:Q2][Gxmpi2:Q0Gympi2:Q1Gxmpi2:Q2]Gxmpi2:Q0[Gypi2:Q0Gypi2:Q1]'
                '[Gxmpi2:Q0Gxpi2:Q1]Gxmpi2:Q0@(Q0,Q1,Q2)',
                'Gxmpi2:Q2Gxmpi2:Q2@(Q0,Q1,Q2)'),
        }
        for seed, (circ_str, meas, sign, paired_str, readout_str) in golden.items():
            plain = _dfe.sample_dfe_circuit(self.pspec, self.target, self.compilations, seed=seed)
            self.assertEqual((plain[0].str, plain[1], plain[2]), (circ_str, meas, sign))
            for kind, ref_str in (('paired', paired_str), ('readout', readout_str)):
                with_ref = _dfe.sample_dfe_circuit(self.pspec, self.target, self.compilations, seed=seed,
                                                   reference=kind)
                self.assertEqual(len(with_ref), 4)
                self.assertEqual((with_ref[0].str, with_ref[1], with_ref[2]), (circ_str, meas, sign))
                self.assertEqual(with_ref[3].str, ref_str)

    def test_reference_circuit_form(self):
        model = pygsti.models.create_crosstalk_free_model(self.pspec)
        for kind in ('paired', 'readout'):
            for seed in range(10):
                dressed, meas, sign, ref = _dfe.sample_dfe_circuit(
                    self.pspec, self.target, self.compilations, seed=seed, reference=kind)
                self.assertEqual(ref.line_labels, dressed.line_labels)
                self.assertEqual(ref.two_q_gate_count(), 0)

                expval = 0.0
                probs = model.probabilities(ref)
                for outcome, prob in probs.items():
                    parity = sum(int(outcome[0][i]) for i, m in enumerate(meas) if m == 'Z') % 2
                    expval += prob * (-1) ** parity
                self.assertAlmostEqual(expval, sign, delta=1e-6)

                if kind == 'paired':
                    final = _final_layers(dressed, self.target)
                    self.assertEqual(ref.layertup[len(ref.layertup) - len(final):], final)
                else:
                    self.assertAlmostEqual(max(probs.values()), 1.0, delta=1e-9)

    def test_birb_circuits_unchanged(self):
        golden = {
            0: ('[Gypi2:Q0Gxpi2:Q2][Gxmpi2:Q0Gympi2:Q2]Gxmpi2:Q0[Gcphase:Q0:Q1Gympi2:Q2][Gcphase:Q0:Q1Gi:Q2]'
                '[Gympi2:Q0Gypi2:Q2][Gympi2:Q0Gxpi2:Q2]Gxpi2:Q0@(Q0,Q1,Q2)', ['I', 'Z', 'Z'], -1),
            1: ('[Gxpi2:Q0Gxpi2:Q1Gympi2:Q2][Gympi2:Q0Gympi2:Q1][Gympi2:Q0Gi:Q1Gympi2:Q2][Gcphase:Q0:Q1Gxpi2:Q2]'
                '[Gypi2:Q0Gypi2:Q1Gypi2:Q2][Gxmpi2:Q0Gxmpi2:Q1Gxpi2:Q2][Gxmpi2:Q0Gxmpi2:Q1]@(Q0,Q1,Q2)',
                ['Z', 'Z', 'Z'], 1),
            2: ('[Gxmpi2:Q0Gympi2:Q1Gxmpi2:Q2][Gympi2:Q0Gympi2:Q1Gxmpi2:Q2]Gxpi2:Q1[Gxpi2:Q0Gcphase:Q1:Q2]'
                '[Gi:Q0Gcphase:Q1:Q2]Gympi2:Q1Gympi2:Q1Gxmpi2:Q1@(Q0,Q1,Q2)', ['Z', 'I', 'Z'], -1),
        }
        for seed, (circ_str, meas, sign) in golden.items():
            circ, m, s = _rc.create_binary_rb_circuit(self.pspec, self.compilations, 2, seed=seed)
            self.assertEqual((circ.str, list(m), s), (circ_str, meas, sign))

    def test_eigenstate_bits_uniform(self):
        for support in ([True, True, True], [True, True, True, False], [False, True]):
            n_ok = 2 ** (len(support) - 1)
            for sign in (1, -1):
                counts = {}
                for k in range(4000):
                    bits = _dfe._sample_eigenstate_bits(support, sign, seed=k)
                    parity = sum(int(b) for b, s in zip(bits, support) if s) % 2
                    self.assertEqual((-1) ** parity, sign)
                    key = tuple(int(b) for b in bits)
                    counts[key] = counts.get(key, 0) + 1
                self.assertEqual(len(counts), n_ok)
                for count in counts.values():
                    self.assertAlmostEqual(count / 4000, 1 / n_ok, delta=0.03)

    def test_identity_qubit_gets_random_bit_and_no_final_gate(self):
        model = pygsti.models.create_crosstalk_free_model(self.pspec)
        ones = {'dressed': 0, 'reference': 0}
        num_identity = 0
        for seed in range(80):
            dressed, meas, _, ref = _dfe.sample_dfe_circuit(
                self.pspec, self.target, self.compilations, seed=seed, reference='paired')
            if meas[2] != 'I':
                continue
            num_identity += 1
            self.assertNotIn('Q2', active_qubits(Circuit(_final_layers(dressed, self.target),
                                                         line_labels=QUBITS)))
            for kind, circ in (('dressed', dressed), ('reference', ref)):
                p1 = sum(prob for outcome, prob in model.probabilities(circ).items() if outcome[0][2] == '1')
                self.assertTrue(abs(p1) < 1e-9 or abs(p1 - 1) < 1e-9)
                ones[kind] += round(p1)
        self.assertGreaterEqual(num_identity, 5)
        for kind, count in ones.items():
            self.assertTrue(0 < count < num_identity, kind)

    def test_fixed_feature_targets_keep_width(self):
        gate_names = ['Gi', 'Gxpi2', 'Gxmpi2', 'Gypi2', 'Gympi2', 'Gcphase']
        labels4 = ('Q0', 'Q1', 'Q2', 'Q3')
        pspec4 = QPS(4, gate_names, qubit_labels=labels4, geometry='line')
        compilations4 = CCR.create_standard(pspec4, 'absolute', ('paulis', '1Qcliffords'), verbosity=0)
        rs = np.random.RandomState(3)
        for w, d, xi in ((2, 4, 0.5), (3, 6, 0.6)):
            target = _rc.create_random_circuit_with_fixed_feature_values(pspec4, w, d, xi, rand_state=rs)
            self.assertEqual(active_qubits(target), set(target.line_labels))
            for seed in range(30):
                for kwargs in ({}, {'reference': 'paired'}, {'reference': 'readout'}):
                    out = _dfe.sample_dfe_circuit(pspec4, target, compilations4, seed=seed, **kwargs)
                    self.assertEqual(active_qubits(out[0]), active_qubits(target))

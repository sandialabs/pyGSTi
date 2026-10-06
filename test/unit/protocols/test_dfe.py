import math

import pygsti
from pygsti.baseobjs import Label
from pygsti.circuits import Circuit
from pygsti.processors import CliffordCompilationRules as CCR
from pygsti.processors import QubitProcessorSpec as QPS
from pygsti.protocols import dfe as _dfe
from pygsti.protocols.vb import BenchmarkingDesign
from ..util import BaseCase, with_temp_path


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


class DFEDesignTest(BaseCase):

    @classmethod
    def setUpClass(cls):
        cls.pspec, cls.compilations = make_pspec_and_compilations()
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


class DFEProtocolTest(BaseCase):

    @classmethod
    def setUpClass(cls):
        cls.pspec, cls.compilations = make_pspec_and_compilations()
        cls.c_a = Circuit([[('Gxpi2', 'Q0'), ('Gypi2', 'Q1')], [('Gcphase', 'Q0', 'Q1')]], line_labels=QUBITS)
        cls.c_c = Circuit([[('Gcphase', 'Q0', 'Q1')], [('Gxpi2', 'Q1')], [('Gcphase', 'Q1', 'Q2')]],
                          line_labels=QUBITS)
        cls.ideal = pygsti.models.create_crosstalk_free_model(cls.pspec)
        # Dressing compiles only to single-qubit gates, so this error touches only the targets' CZs.
        cls.noisy = pygsti.models.create_crosstalk_free_model(
            cls.pspec, stochastic_error_probs={'Gcphase': [0.1 / 15] * 15})

    def make_design(self, num_samples, circuits=None, **kwargs):
        circuits = [self.c_a, self.c_c] if circuits is None else circuits
        return _dfe.DFEDesign(self.pspec, circuits, self.compilations, num_samples, **kwargs)

    def make_data(self, design, model, shots, seed):
        ds = pygsti.data.simulate_data(model, design.all_circuits_needing_data, shots, seed=seed)
        return pygsti.protocols.ProtocolData(design, ds)

    def test_default_protocol_only_when_requested(self):
        self.assertEqual(self.make_design(2, seed=0).default_protocols, {})
        design = self.make_design(2, seed=0, add_default_protocol=True)
        self.assertEqual(list(design.default_protocols.keys()), ['DFE'])
        self.assertIsInstance(design.default_protocols['DFE'], _dfe.DFE)

    def test_noiseless_fidelity_is_one(self):
        design = self.make_design(20, seed=0, add_default_protocol=True)
        data = self.make_data(design, self.ideal, 100, seed=1)
        results = pygsti.protocols.run_default_protocols(data).for_protocol['DFE']
        self.assertIsInstance(results, _dfe.DFEResults)
        self.assertEqual(len(results.fidelities), 2)
        self.assertEqual(len(results.standard_errors), 2)
        for f, se in zip(results.fidelities, results.standard_errors):
            self.assertAlmostEqual(f, 1.0, delta=1e-9)
            self.assertAlmostEqual(se, 0.0, delta=1e-9)

    def test_injected_pauli_error(self):
        design = self.make_design(200, seed=0)
        data = self.make_data(design, self.noisy, 500, seed=100)
        results = _dfe.DFE().run(data)
        self.assertEqual(results.name, 'DFE')

        expected = [pygsti.tools.entanglement_fidelity(self.noisy.sim.product(c), self.ideal.sim.product(c), 'pp')
                    for c in (self.c_a, self.c_c)]
        self.assertAlmostEqual(expected[0], 0.9, delta=1e-9)  # 1 - eps for a single CZ
        for f, se, f_exp in zip(results.fidelities, results.standard_errors, expected):
            self.assertGreater(se, 0)
            self.assertLess(se, 0.005)
            self.assertLessEqual(abs(f - f_exp), 4 * se)

    def test_narrower_target_uses_its_own_width(self):
        # Without line_labels this CZ spans only (Q0, Q1), narrower than the design's 3 qubits.
        # Every 2-qubit Pauli fidelity equals 1 - 16/15 * 0.1 under its error, so exact data
        # gives exactly F = 0.9 if, and only if, the conversion uses the target's own n = 2.
        c_2q = Circuit([[('Gxpi2', 'Q0'), ('Gypi2', 'Q1')], [('Gcphase', 'Q0', 'Q1')]])
        self.assertEqual(c_2q.line_labels, ('Q0', 'Q1'))
        design = self.make_design(10, circuits=[self.c_a, c_2q], seed=0)
        ds = pygsti.data.simulate_data(self.noisy, design.all_circuits_needing_data, 100, sample_error='none')
        results = _dfe.DFE().run(pygsti.protocols.ProtocolData(design, ds))
        self.assertAlmostEqual(results.fidelities[1], 0.9, delta=1e-7)

    def test_rejects_pinned_designs(self):
        with self.assertRaises(ValueError):
            self.make_design(2, seed=0, fixed_identity_locations='idle', add_default_protocol=True)

        for fil in ('idle', [[False, False, True], None]):
            design = self.make_design(2, seed=0, fixed_identity_locations=fil, add_default_protocol=False)
            self.assertTrue(design.has_fixed_identity_locations)
            with self.assertRaises(ValueError):
                _dfe.DFE().run(pygsti.protocols.ProtocolData(design))

        self.assertFalse(self.make_design(2, seed=0).has_fixed_identity_locations)
        design = self.make_design(2, seed=0, fixed_identity_locations=[None, None])
        self.assertFalse(design.has_fixed_identity_locations)

    def test_single_sample_standard_error_is_nan(self):
        design = self.make_design(1, seed=0)
        data = self.make_data(design, self.ideal, 100, seed=1)
        results = _dfe.DFE().run(data)
        for f, se in zip(results.fidelities, results.standard_errors):
            self.assertAlmostEqual(f, 1.0, delta=1e-9)
            self.assertTrue(math.isnan(se))

    def test_rejects_non_dfe_design(self):
        design = BenchmarkingDesign([0], [[self.c_a]], [[('000',)]])
        with self.assertRaises(TypeError):
            _dfe.DFE().run(pygsti.protocols.ProtocolData(design))

    @with_temp_path
    def test_round_trip(self, root):
        design = self.make_design(5, seed=0, add_default_protocol=True)
        data = self.make_data(design, self.noisy, 100, seed=1)
        results_dir = pygsti.protocols.run_default_protocols(data)
        results = results_dir.for_protocol['DFE']
        results_dir.write(root)

        loaded_design = pygsti.io.read_data_from_dir(root).edesign
        self.assertIsInstance(loaded_design, _dfe.DFEDesign)
        self.assertEqual(list(loaded_design.default_protocols.keys()), ['DFE'])
        self.assertFalse(loaded_design.has_fixed_identity_locations)

        loaded = pygsti.io.read_results_from_dir(root, name='DFE')
        self.assertIsInstance(loaded, _dfe.DFEResults)
        self.assertEqual(loaded.fidelities, results.fidelities)
        self.assertEqual(loaded.standard_errors, results.standard_errors)

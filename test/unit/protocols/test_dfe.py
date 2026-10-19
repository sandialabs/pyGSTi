import copy
import math
import os
import warnings

import numpy as np

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

    @with_temp_path
    def test_seed_recorded(self, root):
        d = self.make_design(seed=None)
        self.assertIsInstance(d.seed, int)
        d_again = self.make_design(seed=d.seed)
        self.assertEqual(d.circuit_lists, d_again.circuit_lists)
        self.assertEqual(d.measurements, d_again.measurements)
        self.assertEqual(d.signs, d_again.signs)

        self.assertEqual(self.make_design(seed=3).seed, 3)

        np.random.seed(5)
        seed_1 = self.make_design().seed
        np.random.seed(5)
        seed_2 = self.make_design().seed
        self.assertEqual(seed_1, seed_2)

        self.make_design(seed=7).write(root)
        self.assertEqual(pygsti.io.read_edesign_from_dir(root).seed, 7)

        np_seeded = self.make_design(seed=np.int64(7))
        self.assertEqual(np_seeded.circuit_lists, self.make_design(seed=7).circuit_lists)
        np_root = os.path.join(root, 'np_seed')
        np_seeded.write(np_root)
        self.assertEqual(pygsti.io.read_edesign_from_dir(np_root).seed, 7)


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

    def test_dataframe_warns_on_mixed_line_labels(self):
        c_2q = Circuit([[('Gxpi2', 'Q0'), ('Gypi2', 'Q1')], [('Gcphase', 'Q0', 'Q1')]])
        self.assertEqual(c_2q.line_labels, ('Q0', 'Q1'))
        for circuits in ([c_2q, self.c_a], [self.c_a, c_2q]):
            design = self.make_design(2, circuits=circuits, seed=0)
            results = _dfe.DFE().run(self.make_data(design, self.ideal, 10, seed=1))
            with self.assertWarnsRegex(UserWarning, "differ from the design's qubit_labels"):
                df = results.to_dataframe()
            self.assertEqual(len(df), 2)

        design = self.make_design(2, seed=0)
        results = _dfe.DFE().run(self.make_data(design, self.ideal, 10, seed=1))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            results.to_dataframe()
        self.assertFalse([w for w in caught if "differ from the design's qubit_labels" in str(w.message)])

    def test_rejects_pinned_designs(self):
        with self.assertRaises(ValueError):
            self.make_design(2, seed=0, fixed_identity_locations='idle', add_default_protocol=True)

        for fil in ('idle', [[False, False, True], None]):
            design = self.make_design(2, seed=0, fixed_identity_locations=fil, add_default_protocol=False)
            self.assertTrue(design.has_fixed_identity_locations)
            data = self.make_data(design, self.ideal, 10, seed=1)
            with self.assertRaisesRegex(ValueError, 'pins identity locations'):
                _dfe.DFE().run(data)

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

    def test_map_qubit_labels(self):
        mapper = {'Q0': 'Q5', 'Q1': 'Q6', 'Q2': 'Q7'}
        inverse = {v: k for k, v in mapper.items()}
        design = self.make_design(5, seed=0, add_default_protocol=True)
        mapped = design.map_qubit_labels(mapper)
        self.assertIsInstance(mapped, _dfe.DFEDesign)
        self.assertEqual(mapped.qubit_labels, ('Q5', 'Q6', 'Q7'))
        self.assertEqual(list(mapped.depths), list(design.depths))
        for cl, mapped_cl in zip(design.circuit_lists, mapped.circuit_lists):
            self.assertEqual(list(mapped_cl), [c.map_state_space_labels(mapper) for c in cl])
        self.assertEqual(mapped.measurements, design.measurements)
        self.assertEqual(mapped.signs, design.signs)
        self.assertEqual(mapped.seed, 0)
        self.assertFalse(mapped.has_fixed_identity_locations)
        self.assertEqual(mapped.default_protocols, {})

        back = mapped.map_qubit_labels(inverse)
        self.assertEqual([list(cl) for cl in back.circuit_lists], [list(cl) for cl in design.circuit_lists])
        data = self.make_data(design, self.noisy, 100, seed=1)
        expected = _dfe.DFE().run(data)
        results = _dfe.DFE().run(pygsti.protocols.ProtocolData(back, data.dataset))
        self.assertEqual(results.fidelities, expected.fidelities)
        self.assertEqual(results.standard_errors, expected.standard_errors)

        pinned = self.make_design(2, seed=0, fixed_identity_locations='idle')
        self.assertTrue(pinned.map_qubit_labels(mapper).has_fixed_identity_locations)

        self.assertEqual(list(design.truncate_to_lists([1]).map_qubit_labels(mapper).depths), [1])

        # A design saved before `seed` and `has_fixed_identity_locations` existed lacks both.
        old = copy.deepcopy(design)
        del old.seed
        del old.has_fixed_identity_locations
        mapped_old = old.map_qubit_labels(mapper)
        self.assertIsNone(mapped_old.seed)
        self.assertFalse(mapped_old.has_fixed_identity_locations)

    @with_temp_path
    def test_design_saved_without_new_attributes_loads_and_runs(self, root):
        # A design saved before `has_fixed_identity_locations` and `seed` existed has neither.
        design = self.make_design(3, seed=0)
        del design.has_fixed_identity_locations
        del design.seed
        self.make_data(design, self.ideal, 10, seed=1).write(root)

        loaded = pygsti.io.read_data_from_dir(root)
        self.assertFalse(hasattr(loaded.edesign, 'has_fixed_identity_locations'))
        for f in _dfe.DFE().run(loaded).fidelities:
            self.assertAlmostEqual(f, 1.0, delta=1e-9)

    def test_rejects_partial_data(self):
        design = self.make_design(5, seed=0)
        ds = self.make_data(design, self.noisy, 100, seed=1).dataset
        circuits = list(dict.fromkeys(design.all_circuits_needing_data))

        with self.assertRaisesRegex(ValueError, r'\(1 missing, 0 with zero counts\)') as cm:
            _dfe.DFE().run(pygsti.protocols.ProtocolData(design, ds.truncate(circuits[1:])))
        self.assertIn('truncate_to_available_data', str(cm.exception))

        ds_zero = pygsti.data.DataSet(outcome_labels=ds.outcome_labels)
        ds_zero.add_count_dict(circuits[0], {})
        for c in circuits[1:]:
            ds_zero.add_count_dict(c, ds[c].counts)
        ds_zero.done_adding_data()
        with self.assertRaisesRegex(ValueError, r'\(0 missing, 1 with zero counts\)'):
            _dfe.DFE().run(pygsti.protocols.ProtocolData(design, ds_zero))

        with self.assertRaisesRegex(ValueError, 'has no dataset'):
            _dfe.DFE().run(pygsti.protocols.ProtocolData(design))

    def test_rejects_misaligned_design(self):
        design = self.make_design(5, seed=0)
        ds = self.make_data(design, self.noisy, 100, seed=1).dataset

        short_measurements = copy.deepcopy(design)
        short_measurements.measurements = short_measurements.measurements[:-1]
        with self.assertRaisesRegex(ValueError, "don't line up with its circuit lists"):
            _dfe.DFE().run(pygsti.protocols.ProtocolData(short_measurements, ds))

        short_signs = copy.deepcopy(design)
        short_signs.signs[0] = short_signs.signs[0][:-1]
        with self.assertRaisesRegex(ValueError, "don't line up with its circuit lists"):
            _dfe.DFE().run(pygsti.protocols.ProtocolData(short_signs, ds))

    def test_truncation_emptying_a_target_is_never_misaligned(self):
        design = self.make_design(5, circuits=[self.c_a, self.c_c, self.c_a], seed=0)
        ds_full = self.make_data(design, self.noisy, 100, seed=1).dataset
        expected = _dfe.DFE().run(pygsti.protocols.ProtocolData(design, ds_full)).fidelities

        kept = list(dict.fromkeys(design.circuit_lists[0] + design.circuit_lists[2]))
        ds_partial = ds_full.truncate(kept)
        truncated = design.truncate_to_available_data(ds_partial)

        # Truncating away every dressed circuit of target 1 should leave it empty (NaN), but
        # BenchmarkingDesign's truncation currently misaligns the paired measurements and signs,
        # which DFE refuses. Either outcome is accepted, so this test survives a fix of that bug;
        # what it rules out is the silently wrong [F0, nan, nan] that the run used to return.
        try:
            fidelities = _dfe.DFE().run(pygsti.protocols.ProtocolData(truncated, ds_partial)).fidelities
        except ValueError as err:
            self.assertRegex(str(err), "don't line up with its circuit lists")
        else:
            self.assertTrue(math.isnan(fidelities[1]))
            self.assertAlmostEqual(fidelities[0], expected[0], delta=1e-12)
            self.assertAlmostEqual(fidelities[2], expected[2], delta=1e-12)

    def test_standard_error_matches_replica_spread(self):
        # Each replica redraws the design as well as the data: the standard error covers Pauli
        # sampling as well as shot noise, and a fixed design would measure shot noise only.
        # The fixture must stay 3-qubit, so that most dressed circuits are distinct (about 44 of
        # 50). A 2-qubit target would expose the known underestimate of the standard error when
        # dressed circuits repeat; whoever fixes that should add a 2-qubit case here.
        fidelities, standard_errors = [], []
        for r in range(30):
            design = self.make_design(50, seed=r)
            data = self.make_data(design, self.noisy, 200, seed=1000 + r)
            results = _dfe.DFE().run(data)
            fidelities.append(results.fidelities)
            standard_errors.append(results.standard_errors)
        fidelities = np.array(fidelities)
        standard_errors = np.array(standard_errors)
        for t in range(2):
            ratio = standard_errors[:, t].mean() / fidelities[:, t].std(ddof=1)
            self.assertGreaterEqual(ratio, 0.7, msg='target %d' % t)
            self.assertLessEqual(ratio, 1.4, msg='target %d' % t)

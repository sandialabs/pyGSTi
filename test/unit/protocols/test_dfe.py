import copy
import math
import os
import warnings

import numpy as np

import pygsti
from pygsti.algorithms import dfe as _alg_dfe
from pygsti.circuits import Circuit
from pygsti.protocols import dfe as _dfe
from pygsti.protocols import vb as _vb
from pygsti.protocols.vb import BenchmarkingDesign
from ..algorithms.test_dfe import QUBITS, make_pspec_and_compilations
from ..util import BaseCase, with_temp_path


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

    def test_reference_design_pairs_and_keeps_dressed_circuits(self):
        plain = self.make_design(seed=3)
        self.assertIsNone(plain.reference)
        self.assertIsNone(plain.reference_circuits)
        self.assertNotIn('reference_circuits', plain.paired_with_circuit_attrs)
        dressed = [c for cl in plain.circuit_lists for c in cl]
        self.assertEqual(list(plain.all_circuits_needing_data), dressed)

        for kind in ('paired', 'readout'):
            design = self.make_design(seed=3, reference=kind)
            self.assertEqual(design.reference, kind)
            self.assertEqual(design.circuit_lists, plain.circuit_lists)
            self.assertEqual(design.measurements, plain.measurements)
            self.assertEqual(design.signs, plain.signs)

            self.assertEqual(len(design.reference_circuits), 2)
            for cl, refs in zip(design.circuit_lists, design.reference_circuits):
                self.assertEqual(len(refs), len(cl))
            references = [c for refs in design.reference_circuits for c in refs]
            self.assertEqual(list(design.all_circuits_needing_data), dressed + references)
            self.assertIn('reference_circuits', design.paired_with_circuit_attrs)

    def test_rejects_unknown_reference(self):
        for bad in ('both', True):
            with self.assertRaises(ValueError):
                self.make_design(seed=3, reference=bad)
            with self.assertRaises(ValueError):
                _alg_dfe.sample_dfe_circuit(self.pspec, self.c_a, self.compilations, reference=bad, seed=0)

    @with_temp_path
    def test_references_survive_round_trip_map_and_truncation(self, root):
        for kind in ('paired', 'readout'):
            self.check_references_survive(os.path.join(root, kind), kind)

    def check_references_survive(self, root, kind):
        design = self.make_design(seed=3, reference=kind)
        design.write(root)
        loaded = pygsti.io.read_edesign_from_dir(root)
        self.assertEqual(loaded.reference, kind)
        self.assertEqual([list(r) for r in loaded.reference_circuits],
                         [list(r) for r in design.reference_circuits])

        mapper = {'Q0': 'Q5', 'Q1': 'Q6', 'Q2': 'Q7'}
        mapped = design.map_qubit_labels(mapper)
        self.assertEqual(mapped.reference, kind)
        self.assertEqual([list(r) for r in mapped.reference_circuits],
                         [[c.map_state_space_labels(mapper) for c in refs] for refs in design.reference_circuits])
        back = mapped.map_qubit_labels({v: k for k, v in mapper.items()})
        self.assertEqual([list(r) for r in back.reference_circuits],
                         [list(r) for r in design.reference_circuits])

        pairs = {(d, r) for cl, refs in zip(design.circuit_lists, design.reference_circuits)
                 for d, r in zip(cl, refs)}
        keep = [cl[0] for cl in design.circuit_lists] + [c for refs in design.reference_circuits for c in refs]
        truncated = design.truncate_to_circuits(keep)
        for cl, refs in zip(truncated.circuit_lists, truncated.reference_circuits):
            self.assertGreaterEqual(len(cl), 1)
            self.assertEqual(len(cl), len(refs))
            for d, r in zip(cl, refs):
                self.assertIn((d, r), pairs)

        only_1 = design.truncate_to_lists([1])
        self.assertEqual([list(r) for r in only_1.reference_circuits], [list(design.reference_circuits[1])])

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
        # Asymmetric readout error on each qubit: P(1|0) = 0.96% and P(0|1) = 4.88%.
        cls.readout = pygsti.models.create_crosstalk_free_model(
            cls.pspec, stochastic_error_probs={'povm': [0.01, 0, 0]},
            lindblad_error_coeffs={'povm': {('S', 'X'): 0.01, ('S', 'Y'): 0.01, ('A', 'X', 'Y'): -0.01}})
        cls.noisy_readout = pygsti.models.create_crosstalk_free_model(
            cls.pspec, stochastic_error_probs={'Gcphase': [0.1 / 15] * 15, 'povm': [0.02, 0, 0]})

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
            self.assertEqual(len(df), 4)

        design = self.make_design(2, seed=0)
        results = _dfe.DFE().run(self.make_data(design, self.ideal, 10, seed=1))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            results.to_dataframe()
        self.assertFalse([w for w in caught if "differ from the design's qubit_labels" in str(w.message)])

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
        self.assertEqual(mapped.default_protocols, {})

        back = mapped.map_qubit_labels(inverse)
        self.assertEqual([list(cl) for cl in back.circuit_lists], [list(cl) for cl in design.circuit_lists])
        data = self.make_data(design, self.noisy, 100, seed=1)
        expected = _dfe.DFE().run(data)
        results = _dfe.DFE().run(pygsti.protocols.ProtocolData(back, data.dataset))
        self.assertEqual(results.fidelities, expected.fidelities)
        self.assertEqual(results.standard_errors, expected.standard_errors)

        self.assertEqual(list(design.truncate_to_lists([1]).map_qubit_labels(mapper).depths), [1])

        # A design saved before `seed` existed lacks it.
        old = copy.deepcopy(design)
        del old.seed
        del old.reference
        del old.reference_circuits
        mapped_old = old.map_qubit_labels(mapper)
        self.assertIsNone(mapped_old.seed)
        self.assertIsNone(mapped_old.reference)
        self.assertIsNone(mapped_old.reference_circuits)

    @with_temp_path
    def test_design_saved_without_new_attributes_loads_and_runs(self, root):
        # A design saved before `seed` existed lacks it.
        design = self.make_design(3, seed=0)
        del design.seed
        del design.reference
        del design.reference_circuits
        self.make_data(design, self.ideal, 10, seed=1).write(root)

        loaded = pygsti.io.read_data_from_dir(root)
        self.assertFalse(hasattr(loaded.edesign, 'seed'))
        self.assertFalse(hasattr(loaded.edesign, 'reference'))
        results = _dfe.DFE().run(loaded)
        for f in results.fidelities:
            self.assertAlmostEqual(f, 1.0, delta=1e-9)
        self.assertIsNone(results.reference_corrected_fidelities)

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

    def test_reference_checks_cover_references(self):
        for kind in ('paired', 'readout'):
            self.check_reference_checks(kind)

    def check_reference_checks(self, kind):
        design = self.make_design(5, seed=0, reference=kind)
        ds = self.make_data(design, self.noisy, 100, seed=1).dataset
        circuits = list(dict.fromkeys(design.all_circuits_needing_data))
        ref = design.reference_circuits[0][0]
        self.assertNotIn(ref, design.circuit_lists[0])

        without = [c for c in circuits if c != ref]
        with self.assertRaisesRegex(ValueError, r'\(1 missing, 0 with zero counts\)'):
            _dfe.DFE().run(pygsti.protocols.ProtocolData(design, ds.truncate(without)))

        ds_zero = pygsti.data.DataSet(outcome_labels=ds.outcome_labels)
        for c in circuits:
            ds_zero.add_count_dict(c, {} if c == ref else ds[c].counts)
        ds_zero.done_adding_data()
        with self.assertRaisesRegex(ValueError, r'\(0 missing, 1 with zero counts\)'):
            _dfe.DFE().run(pygsti.protocols.ProtocolData(design, ds_zero))

        short = copy.deepcopy(design)
        short.reference_circuits[0] = short.reference_circuits[0][:-1]
        with self.assertRaisesRegex(ValueError, "don't line up with its circuit lists"):
            _dfe.DFE().run(pygsti.protocols.ProtocolData(short, ds))

    def test_reference_corrected_noiseless_and_readout_only(self):
        for kind in ('paired', 'readout'):
            self.check_corrected_noiseless_and_readout_only(kind)

    def check_corrected_noiseless_and_readout_only(self, kind):
        design = self.make_design(20, seed=0, reference=kind)
        results = _dfe.DFE().run(self.make_data(design, self.ideal, 100, seed=1))
        for f, se in zip(results.reference_corrected_fidelities, results.reference_corrected_standard_errors):
            self.assertAlmostEqual(f, 1.0, delta=1e-9)
            self.assertAlmostEqual(se, 0.0, delta=1e-9)

        design = self.make_design(100, seed=0, reference=kind)
        data = self.make_data(design, self.readout, 1000, seed=1)
        results = _dfe.DFE().run(data)
        for t in range(2):
            fc, sec = results.reference_corrected_fidelities[t], results.reference_corrected_standard_errors[t]
            self.assertLessEqual(abs(fc - 1), 4 * sec)
            self.assertGreater(sec, 0)
            self.assertLess(results.fidelities[t], 1 - 4 * results.standard_errors[t])

        plain = self.make_design(100, seed=0)
        expected = _dfe.DFE().run(pygsti.protocols.ProtocolData(plain, data.dataset))
        self.assertEqual(results.fidelities, expected.fidelities)
        self.assertEqual(results.standard_errors, expected.standard_errors)

    def test_reference_corrected_matches_true_fidelity(self):
        expected = [pygsti.tools.entanglement_fidelity(self.noisy.sim.product(c), self.ideal.sim.product(c), 'pp')
                    for c in (self.c_a, self.c_c)]
        for kind in ('paired', 'readout'):
            design = self.make_design(200, seed=0, reference=kind)
            results = _dfe.DFE().run(self.make_data(design, self.noisy_readout, 500, seed=100))
            for fc, sec, f_true in zip(results.reference_corrected_fidelities,
                                       results.reference_corrected_standard_errors, expected):
                self.assertGreater(sec, 0)
                self.assertLess(sec, 0.005)
                self.assertLessEqual(abs(fc - f_true), 4 * sec)

    def test_nonpositive_reference_mean_gives_nan_and_warns(self):
        design = self.make_design(5, seed=0, reference='paired')
        self.assertFalse(set(design.reference_circuits[0]) & set(design.reference_circuits[1]))
        ds = self.make_data(design, self.ideal, 100, seed=1).dataset
        bad = {}
        for ref, meas, sign in zip(design.reference_circuits[0], design.measurements[0], design.signs[0]):
            n = len(meas)
            if sign == 1:
                out = ''.join('1' if k == meas.index('Z') else '0' for k in range(n))
            else:
                out = '0' * n
            bad[ref] = out
        ds_bad = pygsti.data.DataSet(outcome_labels=ds.outcome_labels)
        for c in dict.fromkeys(design.all_circuits_needing_data):
            ds_bad.add_count_dict(c, {(bad[c],): 100} if c in bad else ds[c].counts)
        ds_bad.done_adding_data()

        with self.assertWarnsRegex(UserWarning, 'non-positive mean reference energy'):
            results = _dfe.DFE().run(pygsti.protocols.ProtocolData(design, ds_bad))
        self.assertTrue(math.isnan(results.reference_corrected_fidelities[0]))
        self.assertTrue(math.isnan(results.reference_corrected_standard_errors[0]))
        self.assertAlmostEqual(results.reference_corrected_fidelities[1], 1.0, delta=1e-9)
        expected = _dfe.DFE().run(pygsti.protocols.ProtocolData(design, ds))
        self.assertEqual(results.fidelities, expected.fidelities)
        self.assertEqual(results.standard_errors, expected.standard_errors)

    def test_reference_standard_error_matches_replica_spread(self):
        fidelities, standard_errors = [], []
        for r in range(30):
            design = self.make_design(50, seed=r, reference='paired')
            data = self.make_data(design, self.noisy_readout, 200, seed=1000 + r)
            results = _dfe.DFE().run(data)
            fidelities.append(results.reference_corrected_fidelities)
            standard_errors.append(results.reference_corrected_standard_errors)
        fidelities = np.array(fidelities)
        standard_errors = np.array(standard_errors)
        # With 30 replicas the spread's own relative error is about 13%, and the ratio varies
        # across platforms for the same seeds, so the bounds are wide enough to be robust while
        # still catching a factor-of-two error in the estimate.
        for t in range(2):
            ratio = standard_errors[:, t].mean() / fidelities[:, t].std(ddof=1)
            self.assertGreaterEqual(ratio, 0.5, msg='target %d' % t)
            self.assertLessEqual(ratio, 1.5, msg='target %d' % t)

    def test_reference_standard_error_uses_the_pairs(self):
        # Each reference reads exactly like its dressed circuit, so every pair's energies agree.
        design = self.make_design(10, seed=3, reference='paired')
        dressed = [c for cl in design.circuit_lists for c in cl]
        references = [c for refs in design.reference_circuits for c in refs]
        self.assertEqual(len(set(dressed)), len(dressed))
        self.assertEqual(len(set(references)), len(references))
        self.assertFalse(set(references) & set(dressed))
        ds = self.make_data(design, self.noisy, 100, seed=1).dataset
        ds_same = pygsti.data.DataSet(outcome_labels=ds.outcome_labels)
        for c in dict.fromkeys(dressed):
            ds_same.add_count_dict(c, ds[c].counts)
        for cl, refs in zip(design.circuit_lists, design.reference_circuits):
            for d, r in zip(cl, refs):
                ds_same.add_count_dict(r, ds[d].counts)
        ds_same.done_adding_data()

        results = _dfe.DFE().run(pygsti.protocols.ProtocolData(design, ds_same))
        for t in range(2):
            self.assertGreater(results.standard_errors[t], 0)
            self.assertAlmostEqual(results.reference_corrected_fidelities[t], 1.0, delta=1e-9)
            self.assertAlmostEqual(results.reference_corrected_standard_errors[t], 0.0, delta=1e-9)

    @with_temp_path
    def test_reference_results_round_trip_and_map(self, root):
        design = self.make_design(5, seed=0, reference='paired', add_default_protocol=True)
        data = self.make_data(design, self.noisy_readout, 100, seed=1)
        results_dir = pygsti.protocols.run_default_protocols(data)
        results = results_dir.for_protocol['DFE']
        results_dir.write(root)
        loaded = pygsti.io.read_results_from_dir(root, name='DFE')
        self.assertEqual(loaded.reference_corrected_fidelities, results.reference_corrected_fidelities)
        self.assertEqual(loaded.reference_corrected_standard_errors, results.reference_corrected_standard_errors)
        self.assertFalse(any(math.isnan(x) for x in loaded.reference_corrected_fidelities
                             + loaded.reference_corrected_standard_errors))

        mapper = {'Q0': 'Q5', 'Q1': 'Q6', 'Q2': 'Q7'}
        back = design.map_qubit_labels(mapper).map_qubit_labels({v: k for k, v in mapper.items()})
        again = _dfe.DFE().run(pygsti.protocols.ProtocolData(back, data.dataset))
        self.assertEqual(again.reference_corrected_fidelities, results.reference_corrected_fidelities)
        self.assertEqual(again.reference_corrected_standard_errors, results.reference_corrected_standard_errors)

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
        fidelities, standard_errors = [], []
        for r in range(30):
            design = self.make_design(50, seed=r)
            data = self.make_data(design, self.noisy, 200, seed=1000 + r)
            results = _dfe.DFE().run(data)
            fidelities.append(results.fidelities)
            standard_errors.append(results.standard_errors)
        fidelities = np.array(fidelities)
        standard_errors = np.array(standard_errors)
        # Wide bounds for the same reason as the reference-corrected test above.
        for t in range(2):
            ratio = standard_errors[:, t].mean() / fidelities[:, t].std(ddof=1)
            self.assertGreaterEqual(ratio, 0.5, msg='target %d' % t)
            self.assertLessEqual(ratio, 1.5, msg='target %d' % t)

    def test_standard_error_counts_repeated_circuits(self):
        # A 2-qubit target's dressed circuits repeat, and the DataSet pools their shots. With
        # 100 replicas the spread's own error is near 7%, and a formula that ignores repeats
        # reads about 0.5 here, so the band is narrower than the 30-replica tests'. The designs
        # stay fixed, so the spread is shot noise only: the CZ error is a uniform Pauli channel
        # and the readout symmetric, so no energy varies with the draw.
        c_2q = Circuit([[('Gxpi2', 'Q0'), ('Gypi2', 'Q1')], [('Gcphase', 'Q0', 'Q1')]],
                       line_labels=('Q0', 'Q1'))
        plain = self.make_design(200, circuits=[c_2q], seed=0)
        paired = self.make_design(200, circuits=[c_2q], seed=0, reference='paired')
        readout = self.make_design(200, circuits=[c_2q], seed=0, reference='readout')
        corrected_circuits = list(paired.all_circuits_needing_data) + list(readout.reference_circuits[0])
        vanilla, paired_est, readout_est = ([], []), ([], []), ([], [])
        for r in range(100):
            ds = pygsti.data.simulate_data(self.noisy, plain.all_circuits_needing_data, 500, seed=1000 + r)
            res = _dfe.DFE().run(pygsti.protocols.ProtocolData(plain, ds))
            vanilla[0].append(res.fidelities[0])
            vanilla[1].append(res.standard_errors[0])
            ds = pygsti.data.simulate_data(self.noisy_readout, corrected_circuits, 500, seed=2000 + r)
            for design, est in ((paired, paired_est), (readout, readout_est)):
                res = _dfe.DFE().run(pygsti.protocols.ProtocolData(design, ds))
                est[0].append(res.reference_corrected_fidelities[0])
                est[1].append(res.reference_corrected_standard_errors[0])
        for f, se in (vanilla, paired_est, readout_est):
            ratio = np.mean(se) / np.std(f, ddof=1)
            self.assertGreaterEqual(ratio, 0.7)
            self.assertLessEqual(ratio, 1.4)

    def test_repeat_covariance_matches_pairwise_covariances(self):
        A = Circuit([[('Gxpi2', 'Q0')]], line_labels=('Q0', 'Q1'))
        B = Circuit([[('Gypi2', 'Q1')]], line_labels=('Q0', 'Q1'))
        ds = pygsti.data.DataSet(outcome_labels=['00', '01', '10', '11'])
        a_counts = {'00': 40, '01': 25, '10': 20, '11': 15}
        ds.add_count_dict(A, a_counts)
        ds.add_count_dict(B, {'00': 70, '11': 30})
        ds.done_adding_data()
        ZZ, ZI = ['Z', 'Z'], ['Z', 'I']
        entries = [[(A, 1.0, ZZ, 1)], [(A, 1.0, ZI, -1)], [(B, 1.0, ZZ, 1), (A, -0.8, ZZ, 1)]]
        a_terms = [(1.0, ZZ, 1), (1.0, ZI, -1), (-0.8, ZZ, 1)]
        M = sum(a_counts.values())

        def y(i, o):
            coef, meas, sign = a_terms[i]
            return coef * _vb._outcome_energy(o, meas, sign)

        def mean(f):
            return sum(n * f(o) for o, n in a_counts.items()) / M

        expected = 0.0
        for i in range(3):
            for j in range(3):
                if i != j:
                    cov = mean(lambda o: y(i, o) * y(j, o)) - mean(lambda o: y(i, o)) * mean(lambda o: y(j, o))
                    expected += cov / (M - 1)
        self.assertAlmostEqual(_dfe._repeat_covariance(ds, entries), expected, delta=1e-12)

    def test_reference_standard_error_counts_repeated_references(self):
        c_2q = Circuit([[('Gxpi2', 'Q0'), ('Gypi2', 'Q1')], [('Gcphase', 'Q0', 'Q1')]],
                       line_labels=('Q0', 'Q1'))
        design = self.make_design(20, circuits=[c_2q], seed=0, reference='readout')
        dressed, refs = design.circuit_lists[0], design.reference_circuits[0]
        meas, signs = design.measurements[0], design.signs[0]
        self.assertLess(len(set(refs)), len(refs))
        ds = self.make_data(design, self.noisy_readout, 100, seed=1).dataset
        results = _dfe.DFE().run(pygsti.protocols.ProtocolData(design, ds))

        e = np.array([_vb._avg_energy(ds[c], m, s) for c, m, s in zip(dressed, meas, signs)])
        r = np.array([_vb._avg_energy(ds[c], m, s) for c, m, s in zip(refs, meas, signs)])
        p = e.mean() / r.mean()
        K = len(e)
        entries = [[(dressed[i], 1.0), (refs[i], -p)] for i in range(K)]

        def cov(row, i, j):
            counts, M = ds[row].counts, ds[row].total

            def mean(f):
                return sum(n * f(o[0]) for o, n in counts.items()) / M

            def y(k, o):
                return _vb._outcome_energy(o, meas[k], signs[k])

            return (mean(lambda o: y(i, o) * y(j, o)) - mean(lambda o: y(i, o)) * mean(lambda o: y(j, o))) / (M - 1)

        repeats = 0.0
        for i in range(K):
            for j in range(K):
                for ci, ai in entries[i]:
                    for cj, aj in entries[j]:
                        if i != j and ci == cj:
                            repeats += ai * aj * cov(ci, i, j)
        d = e - p * r
        var = (np.sum((d - d.mean())**2) + repeats) / (K - 1)
        expected = (1 - 4.0**-2) * np.sqrt(var / K) / r.mean()
        self.assertAlmostEqual(results.reference_corrected_standard_errors[0], expected, delta=1e-6 * expected)

    def test_amplitude_damping_is_unbiased(self):
        cz = Circuit([[('Gcphase', 'Q0', 'Q1')]], line_labels=QUBITS)
        r = 0.4
        for sgn in (-1.0, 1.0):
            coeffs = {('S', 'XI'): r / 4, ('S', 'YI'): r / 4, ('A', 'XI', 'YI'): sgn * r / 4,
                      ('S', 'IX'): r / 4, ('S', 'IY'): r / 4, ('A', 'IX', 'IY'): sgn * r / 4}
            noisy = pygsti.models.create_crosstalk_free_model(self.pspec, lindblad_error_coeffs={'Gcphase': coeffs})
            ideal = self.ideal.copy()
            noisy.sim = 'matrix'
            ideal.sim = 'matrix'
            f_true = pygsti.tools.entanglement_fidelity(noisy.sim.product(cz), ideal.sim.product(cz), 'pp')
            design = self.make_design(600, circuits=[cz], seed=0)
            results = _dfe.DFE().run(self.make_data(design, noisy, 1000, seed=1))
            f, se = results.fidelities[0], results.standard_errors[0]
            self.assertLessEqual(abs(f - f_true), 4 * se)
            self.assertLess(se, 0.015)

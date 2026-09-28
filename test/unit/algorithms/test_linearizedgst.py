import importlib.util

import numpy as np
import pytest
import stim

from ..util import BaseCase

from pygsti.baseobjs import Label
from pygsti.circuits import Circuit
from pygsti.processors import QubitProcessorSpec
from pygsti.algorithms import linearizedgst as lgst
from pygsti.errorgenpropagation import mcmgadget as mg
from pygsti.errorgenpropagation.localstimerrorgen import LocalStimErrorgenLabel as _LSE


def _random_circuits(rng, num_circuits, depth, line_labels=(0, 1), mcm_qubit=None, mcm_every=1):
    """Random 2-qubit Clifford circuits over Gxpi2/Gypi2 and Gcphase(0,1), optionally with an 'Iz' MCM inserted."""
    q0, q1 = line_labels
    circuits = []
    for i in range(num_circuits):
        layers = []
        for d in range(depth):
            if rng.random() < 0.4:
                layers.append(Label('Gcphase', (q0, q1)))
            else:
                comps = [Label(str(rng.choice(['Gxpi2', 'Gypi2'])), q) for q in line_labels if rng.random() < 0.8]
                if len(comps) == 1:
                    layers.append(comps[0])
                elif len(comps) > 1:
                    layers.append(Label(comps))
        if mcm_qubit is not None and i % mcm_every == 0:
            pos = rng.integers(0, len(layers) + 1)
            layers.insert(pos, Label('Iz', mcm_qubit))
        circuits.append(Circuit(layers, line_labels=line_labels))
    return circuits


def _random_rates(rng, keys, h_scale=0.01, s_scale=0.002):
    return {k: (rng.uniform(-h_scale, h_scale) if k[0] == 'H' else rng.uniform(0, s_scale)) for k in keys}


def _gate_ansatz(rng, q0=0, q1=1):
    return {
        ('Gxpi2', q0): _random_rates(rng, [('H', 'X'), ('S', 'X'), ('H', 'Z:%s' % str(q1))]),
        ('Gxpi2', q1): _random_rates(rng, [('H', 'X'), ('S', 'X'), ('H', 'Z:%s' % str(q0))]),
        ('Gypi2', q0): _random_rates(rng, [('H', 'Y'), ('S', 'Y')]),
        ('Gypi2', q1): _random_rates(rng, [('H', 'Y'), ('S', 'Y')]),
        ('Gcphase', q0, q1): _random_rates(rng, [('H', 'ZZ'), ('S', 'ZZ'), ('H', 'IZ')]),
    }


def _simulate(pspec, ansatz, circuits, params, rates=None, **kwargs):
    """Exact (noisy, ideal) outcome probabilities of `circuits` under the ansatz with the given (or stored) rates."""
    mdl = lgst.rate_to_model(pspec, ansatz, rate_ests=rates, **kwargs)
    ideal = lgst.rate_to_model(pspec, ansatz, rate_ests=np.zeros(len(params)), **kwargs)
    return [dict(mdl.probabilities(c)) for c in circuits], [dict(ideal.probabilities(c)) for c in circuits]


class ParameterIndexingTester(BaseCase):

    def test_gate_spam_and_mcm_keys(self):
        ansatz = {('Gxpi2', 0): {('H', 'X'): 0.01, ('S', 'Z:1'): 0.002},
                  ('Gcphase', 0, 1): {('H', 'ZZ'): 0.003, ('C', 'XI', 'YI'): 0.0, ('A', 'ZX:0,1', 'ZY:0,1'): 0.0},
                  ('Iz', 1): {('S', 'IX'): 0.004, ('H', 'ZY'): 0.005, ('S', 'XX:0,v'): 0.006},
                  'prep': {('S', 'X:0'): 0.007, ('S', 'XX'): 0.0},
                  'povm': {('S', 'X:1'): 0.008}}
        params, rates = lgst.build_model_parameter_indexing(ansatz, num_qubits=2)
        self.assertEqual(rates, [0.01, 0.002, 0.003, 0.0, 0.0, 0.004, 0.005, 0.006, 0.007, 0.0, 0.008])
        self.assertEqual(params.qubit_labels, (0, 1))
        self.assertEqual(params.num_qubits, 2)
        self.assertEqual(params.keys, [('Gxpi2', 0), ('Gcphase', 0, 1), ('Iz', 1), 'prep', 'povm'])
        # canonical (n+1)-qubit Pauli strings: data qubits then the virtual-qubit slot
        self.assertEqual(params[0].errorgen, _LSE('H', [stim.PauliString('XII')]))
        self.assertEqual(params[1].errorgen, _LSE('S', [stim.PauliString('IZI')]))
        self.assertEqual(params[2].errorgen, _LSE('H', [stim.PauliString('ZZI')]))
        self.assertEqual(params[3].errorgen, _LSE('C', [stim.PauliString('XII'), stim.PauliString('YII')]))
        self.assertEqual(params[4].errorgen, _LSE('A', [stim.PauliString('ZXI'), stim.PauliString('ZYI')]))
        self.assertEqual(params[5].errorgen, _LSE('S', [stim.PauliString('IIX')]))
        self.assertEqual(params[6].errorgen, _LSE('H', [stim.PauliString('IZY')]))
        self.assertEqual(params[7].errorgen, _LSE('S', [stim.PauliString('XIX')]))
        self.assertEqual(params[8].errorgen, _LSE('S', [stim.PauliString('XII')]))
        self.assertEqual(params[9].errorgen, _LSE('S', [stim.PauliString('XXI')]))
        self.assertEqual(params[10].errorgen, _LSE('S', [stim.PauliString('IXI')]))
        self.assertTrue(params[5].is_mcm and params[6].is_mcm and params[7].is_mcm)
        self.assertFalse(params[0].is_mcm or params[8].is_mcm)
        self.assertTrue(params[8].is_spam and params[10].is_spam and not params[5].is_spam)
        self.assertEqual(params[6].errorgen_type, 'H')
        self.assertEqual(params[6][0], params[6].errorgen)  # tuple access still works
        self.assertEqual(params[6][1], ('Iz', 1))
        self.assertEqual(params.indices_for_key(('Iz', 1)), [5, 6, 7])

    def test_string_qubit_labels_and_label_keys(self):
        ansatz = {(Label('Gxpi2', 'Q1'),): {('H', 'X'): 0.01, ('H', 'Z:Q0'): 0.02}, ('Iz', 'Q0'): {('S', 'IX'): 0.03}}
        params, rates = lgst.build_model_parameter_indexing(ansatz, qubit_labels=['Q0', 'Q1'])
        self.assertEqual(params.keys, [('Gxpi2', 'Q1'), ('Iz', 'Q0')])
        self.assertEqual(params[0].errorgen, _LSE('H', [stim.PauliString('IXI')]))
        self.assertEqual(params[1].errorgen, _LSE('H', [stim.PauliString('ZII')]))
        self.assertEqual(params[2].errorgen, _LSE('S', [stim.PauliString('IIX')]))

    def test_invalid_specifications(self):
        with self.assertRaises(ValueError):  # wrong length plain string
            lgst.build_model_parameter_indexing({('Gxpi2', 0): {('H', 'XX'): 0.01}}, 2)
        with self.assertRaises(ValueError):  # virtual qubit on a gate
            lgst.build_model_parameter_indexing({('Gxpi2', 0): {('H', 'XZ:0,v'): 0.01}}, 2)
        with self.assertRaises(ValueError):  # unknown qubit
            lgst.build_model_parameter_indexing({('Gxpi2', 0): {('H', 'X:5'): 0.01}}, 2)
        with self.assertRaises(ValueError):  # multi-qubit MCM key
            lgst.build_model_parameter_indexing({('Iz', 0, 1): {('S', 'IIX'): 0.01}}, 2)
        with self.assertRaises(ValueError):  # SPAM keys need explicit or full-width support
            lgst.build_model_parameter_indexing({'prep': {('S', 'X'): 0.01}}, 2)
        with self.assertRaises(ValueError):  # bad Pauli character
            lgst.build_model_parameter_indexing({('Gxpi2', 0): {('H', 'Q'): 0.01}}, 2)

    def test_instantiation_on_expanded_system(self):
        params, _ = lgst.build_model_parameter_indexing({('Gxpi2', 0): {('H', 'X'): 0.01},
                                                         ('Iz', 1): {('H', 'ZY'): 0.02}}, 2)
        self.assertEqual(lgst.instantiate_parameter_errorgen(params[0], 2, 0), _LSE('H', [stim.PauliString('XI')]))
        self.assertEqual(lgst.instantiate_parameter_errorgen(params[0], 2, 3), _LSE('H', [stim.PauliString('XIIII')]))
        self.assertEqual(lgst.instantiate_parameter_errorgen(params[1], 2, 3, virtual_index=1),
                         _LSE('H', [stim.PauliString('IZIYI')]))
        with self.assertRaises(ValueError):
            lgst.instantiate_parameter_errorgen(params[1], 2, 3)

    def test_estimated_rates_dict_round_trip_and_unmodeled_keys(self):
        ansatz = {('Gxpi2', 0): {('H', 'X'): 0.01, ('S', 'Z:1'): 0.002}, ('Iz', 1): {('S', 'IX'): 0.004, ('S', 'XX:0,v'): 0.006},
                  'prep': {('S', 'X:0'): 0.007}}
        params, rates = lgst.build_model_parameter_indexing(ansatz, 2)
        rd = lgst.estimated_rates_dict(params, rates)
        self.assertEqual(rd[('Iz', 1)], {('S', 'X:v'): 0.004, ('S', 'XX:0,v'): 0.006})
        self.assertEqual(rd[('Gxpi2', 0)], {('H', 'X:0'): 0.01, ('S', 'Z:1'): 0.002})
        params2, rates2 = lgst.build_model_parameter_indexing(rd, 2)
        self.assertEqual([p.errorgen for p in params2], [p.errorgen for p in params])
        self.assertEqual(rates2, rates)
        circuits = [Circuit([Label('Gxpi2', 0), Label('Gypi2', 1), Label('Iz', (0, 1))], line_labels=(0, 1))]
        self.assertEqual(lgst.unmodeled_gate_keys(circuits, params), {('Gypi2', 1), ('Iz', 0)})

    def test_estimated_rates_dict_uses_union_support_for_C_and_A(self):
        # the two Paulis of a C/A term are written on the union of their supports (required by the model constructors)
        ansatz = {('Gcphase', 0, 1): {('C', 'ZI', 'IZ'): 0.001, ('A', 'X:0', 'Y:1'): 0.002, ('A', 'ZX', 'ZY'): 0.003},
                  ('Gxpi2', 0): {('C', 'X', 'Y'): 0.004},
                  ('Iz', 1): {('A', 'XI', 'YI'): 0.005, ('C', 'IX', 'XZ'): 0.006}}
        params, rates = lgst.build_model_parameter_indexing(ansatz, 2)
        rd = lgst.estimated_rates_dict(params, rates)
        self.assertEqual(rd[('Gcphase', 0, 1)], {('C', 'ZI:0,1', 'IZ:0,1'): 0.001, ('A', 'XI:0,1', 'IY:0,1'): 0.002,
                                                 ('A', 'ZX:0,1', 'ZY:0,1'): 0.003})
        self.assertEqual(rd[('Gxpi2', 0)], {('C', 'X:0', 'Y:0'): 0.004})
        self.assertEqual(rd[('Iz', 1)], {('A', 'X:1', 'Y:1'): 0.005, ('C', 'IX:1,v', 'XZ:1,v'): 0.006})
        params2, rates2 = lgst.build_model_parameter_indexing(rd, 2)
        self.assertEqual([p.errorgen for p in params2], [p.errorgen for p in params])
        self.assertEqual(rates2, rates)


class DesignMatrixTester(BaseCase):
    """Gates-only linearized GST (with crosstalk and SPAM errors), checked against exact simulation."""

    def setUp(self):
        self.rng = np.random.default_rng(2024)
        self.pspec = QubitProcessorSpec(2, ['Gxpi2', 'Gypi2', 'Gcphase'], geometry='line')
        self.ansatz = _gate_ansatz(self.rng)
        self.ansatz['prep'] = {('S', 'X:0'): 0.002, ('S', 'X:1'): 0.001}
        self.ansatz['povm'] = {('S', 'X:0'): 0.003, ('S', 'X:1'): 0.0015}
        self.params, rates = lgst.build_model_parameter_indexing(self.ansatz, 2)
        self.true_rates = np.array(rates)
        self.circuits = _random_circuits(self.rng, 40, 5)

    def test_design_matrix_is_first_order_accurate(self):
        designs = lgst.create_design_matrix_list(self.circuits, self.params, return_info=True)
        self.assertEqual(len(designs), len(self.circuits))
        self.assertIsInstance(designs[0], lgst.CircuitDesign)
        self.assertEqual(designs[0].shape, (3, len(self.params)))   # Z0, Z1, Z0Z1
        self.assertEqual(designs[0].num_mcms, 0)
        D = np.vstack([d.design_matrix for d in designs])
        self.assertEqual(np.linalg.matrix_rank(D), len(self.params))

        obs, ideal = _simulate(self.pspec, self.ansatz, self.circuits, self.params)
        for d, i in zip(designs, ideal):  # ideal expectations from the stabilizer tableau match the ideal model
            self.assertArraysAlmostEqual(lgst.compute_pauli_expectations(i, d.paulis), d.ideal_expectations)
        shifts, _ = lgst.observed_expectation_shifts(designs, [obs, ideal])
        shifts2, _ = lgst.observed_expectation_shifts(designs, [obs, None])
        self.assertArraysAlmostEqual(shifts, shifts2)
        resid1 = np.abs(shifts - D @ self.true_rates).max()
        self.assertGreater(np.abs(shifts).max(), 0.01)
        self.assertLess(resid1, 0.1 * np.abs(shifts).max())

        # the residual is second order in the error rates: halving all rates reduces it ~4x
        obs_half, _ = _simulate(self.pspec, self.ansatz, self.circuits, self.params, rates=self.true_rates / 2)
        shifts_half, _ = lgst.observed_expectation_shifts(designs, [obs_half, None])
        resid2 = np.abs(shifts_half - D @ (self.true_rates / 2)).max()
        self.assertGreater(resid1 / resid2, 3.0)
        self.assertLess(resid1 / resid2, 5.5)

    def test_estimate_error_rates(self):
        designs = lgst.create_design_matrix_list(self.circuits, self.params, return_info=True)
        obs, ideal = _simulate(self.pspec, self.ansatz, self.circuits, self.params)
        rates, errs = lgst.estimate_error_rates(designs, [obs, None], None, self.params)
        self.assertIsNone(errs)
        self.assertLess(np.abs(rates - self.true_rates).max(), 5e-4)
        types = np.array(self.params.errorgen_types())
        self.assertTrue(np.all(rates[types == 'S'] >= 0))
        # pseudo-inverse for the stochastic sector gives (here) essentially the same answer
        rates_inv, _ = lgst.estimate_error_rates(designs, [obs, None], None, self.params, stochastic_solver='inversion')
        self.assertLess(np.abs(rates_inv - self.true_rates).max(), 5e-4)
        # bootstrap error bars
        rates_b, errs_b = lgst.estimate_error_rates(designs, [obs, None], None, self.params,
                                                    error_bar_params=[5, 2000], error_bars='bootstrap', seed=1)
        self.assertArraysAlmostEqual(rates_b, rates)
        self.assertEqual(errs_b.shape, rates.shape)
        self.assertTrue(np.all(errs_b >= 0) and np.any(errs_b > 0))
        with self.assertRaises(ValueError):
            lgst.estimate_error_rates(designs, [obs, None], None, self.params, error_bars='bootstrap')

    def test_legacy_array_interface(self):
        paulis = lgst.default_pauli_measurements(2, 2)
        self.assertEqual([str(p) for p in paulis], ['+Z_', '+_Z', '+ZZ'])
        mats = lgst.create_design_matrix_list(self.circuits, self.params, usePaulis=True, pauli_measurements=paulis)
        self.assertIsInstance(mats[0], np.ndarray)
        stacked = lgst.create_design_matrix(self.circuits, self.params, pauli_measurements=paulis)
        self.assertArraysAlmostEqual(stacked, np.vstack(mats))
        obs, ideal = _simulate(self.pspec, self.ansatz, self.circuits, self.params)
        rates, _ = lgst.estimate_error_rates(mats, [obs, ideal], paulis, self.params)
        self.assertLess(np.abs(rates - self.true_rates).max(), 5e-4)
        with self.assertRaises(ValueError):  # bare arrays need observables
            lgst.estimate_error_rates(mats, [obs, ideal], None, self.params)
        with self.assertRaises(TypeError):  # old (circuit, model, params) call signature
            lgst.create_circuit_design_matrix(self.circuits[0], lgst.rate_to_model(self.pspec, self.ansatz), self.params)
        # permutation matrix / ideal state helpers
        pmat, eoc = lgst.build_permutation_matrix(self.circuits[0], self.params)
        self.assertEqual(pmat.shape, (len(eoc), len(self.params)))
        self.assertIsInstance(lgst.generate_ideal_state(self.circuits[0]), stim.Tableau)

    def test_probability_design_matrix(self):
        circ = self.circuits[3]
        design = lgst.create_circuit_design_matrix(circ, self.params, usePaulis=False, return_info=True)
        self.assertEqual(design.shape, (4, len(self.params)))
        self.assertIsNone(design.paulis)
        obs, ideal = _simulate(self.pspec, self.ansatz, [circ], self.params)
        p_obs = np.array([obs[0].get((lgst.int_to_bin(i, 2),), 0.0) for i in range(4)])
        p_ideal = np.array([ideal[0].get((lgst.int_to_bin(i, 2),), 0.0) for i in range(4)])
        self.assertArraysAlmostEqual(p_ideal, design.ideal_expectations)
        self.assertLess(np.abs((p_obs - p_ideal) - design.design_matrix @ self.true_rates).max(), 5e-4)

    def test_unidentifiable_directions(self):
        # crosstalk onto qubit 1 from gates on qubit 0 vs. from gates on qubit 2 cannot be told apart when
        # every layer contains a gate on both qubits 0 and 2
        ansatz = {('Gxpi2', 0): {('H', 'X'): 0.01, ('H', 'Z:1'): 0.005}, ('Gxpi2', 2): {('H', 'Z:1'): 0.004},
                  ('Gxpi2', 1): {('H', 'X'): 0.01}, ('Gypi2', 1): {('H', 'Y'): 0.01}}
        params, _ = lgst.build_model_parameter_indexing(ansatz, 3)
        full_layers = [Label([Label('Gxpi2', 0), Label(g, 1), Label('Gxpi2', 2)]) for g in ('Gxpi2', 'Gypi2')]
        degenerate = [Circuit([full_layers[0], full_layers[1]], line_labels=(0, 1, 2)),
                      Circuit([full_layers[1], full_layers[0], full_layers[1]], line_labels=(0, 1, 2)),
                      Circuit([full_layers[0]], line_labels=(0, 1, 2)),
                      Circuit([full_layers[1], full_layers[0], full_layers[0]], line_labels=(0, 1, 2))]
        designs = lgst.create_design_matrix_list(degenerate, params, return_info=True)
        directions = lgst.unidentifiable_directions(designs, params)
        self.assertEqual(len(directions), 1)
        self.assertEqual(set(directions[0].keys()), {params[1], params[2]})
        self.assertAlmostEqual(directions[0][params[1]] + directions[0][params[2]], 0.0)
        # same result from a stacked array
        self.assertEqual(len(lgst.unidentifiable_directions(np.vstack(designs), params)), 1)
        # a circuit in which qubit 2 idles (and the crosstalk is subsequently observable) restores identifiability
        diverse = degenerate + [Circuit([Label([Label('Gxpi2', 0), Label('Gypi2', 1)]), full_layers[0]], line_labels=(0, 1, 2))]
        self.assertEqual(lgst.unidentifiable_directions(lgst.create_design_matrix_list(diverse, params), params), [])
        with self.assertRaises(ValueError):
            lgst.unidentifiable_directions(np.zeros((3, 2)), params)

    @pytest.mark.skipif(importlib.util.find_spec('pathos') is None, reason="pathos not installed")
    def test_parallel_pathos(self):
        serial = lgst.create_design_matrix_list(self.circuits[:6], self.params)
        parallel = lgst.create_design_matrix_list_parallel_pathos(self.circuits[:6], self.params, do_parallel=True,
                                                                  num_workers=2)
        for a, b in zip(serial, parallel):
            self.assertArraysAlmostEqual(a, b)


class MCMDesignMatrixTester(BaseCase):
    """Linearized GST with mid-circuit measurements modelled by the virtual-qubit gadget."""

    def setUp(self):
        self.rng = np.random.default_rng(7)
        self.pspec = QubitProcessorSpec(2, ['Gxpi2', 'Gypi2', 'Gcphase'], geometry='line')
        self.ansatz = _gate_ansatz(self.rng)
        fomgi_names = list(mg.fomgi_representatives().keys())
        self.fomgi_truth = {nm: (self.rng.uniform(-0.01, 0.01) if mg.FOMGI_SECTORS[nm] == 'H' else self.rng.uniform(0, 0.002))
                            for nm in fomgi_names}
        self.ansatz[('Iz', 0)] = mg.fomgi_ansatz(rates=self.fomgi_truth)
        self.params, rates = lgst.build_model_parameter_indexing(self.ansatz, 2)
        self.true_rates = np.array(rates)
        self.circuits = _random_circuits(self.rng, 50, 5, mcm_qubit=0, mcm_every=2)

    def test_mcm_design_matrix_and_recovery(self):
        designs = lgst.create_design_matrix_list(self.circuits, self.params, return_info=True)
        n_mcm_circuits = sum(1 for d in designs if d.num_mcms == 1)
        self.assertEqual(n_mcm_circuits, 25)
        for d in designs:
            self.assertEqual(d.shape[0], 6 if d.num_mcms == 1 else 3)  # weight <= 2 Z-type Paulis on 3 (2) qubits
            self.assertTrue(set(np.round(d.ideal_expectations, 8)) <= {-1.0, 0.0, 1.0})
        D = np.vstack([d.design_matrix for d in designs])
        self.assertEqual(np.linalg.matrix_rank(D), len(self.params))

        obs, ideal = _simulate(self.pspec, self.ansatz, self.circuits, self.params)
        for d, i in zip(designs, ideal):
            self.assertArraysAlmostEqual(lgst.compute_pauli_expectations(i, d.paulis), d.ideal_expectations)
        shifts, _ = lgst.observed_expectation_shifts(designs, [obs, ideal])
        resid1 = np.abs(shifts - D @ self.true_rates).max()
        self.assertLess(resid1, 0.1 * np.abs(shifts).max())
        obs_half, _ = _simulate(self.pspec, self.ansatz, self.circuits, self.params, rates=self.true_rates / 2)
        shifts_half, _ = lgst.observed_expectation_shifts(designs, [obs_half, None])
        resid2 = np.abs(shifts_half - D @ (self.true_rates / 2)).max()
        self.assertGreater(resid1 / resid2, 3.0)
        self.assertLess(resid1 / resid2, 5.5)

        rates, _ = lgst.estimate_error_rates(designs, [obs, None], None, self.params)
        self.assertLess(np.abs(rates - self.true_rates).max(), 1e-3)
        fomgi = lgst.mcm_fomgi_estimates(self.params, rates)[('Iz', 0)]
        self.assertEqual(list(fomgi.keys()), list(self.fomgi_truth.keys()))
        for name, value in self.fomgi_truth.items():
            self.assertAlmostEqual(fomgi[name], value, delta=1e-3)

    def test_mcm_gauge_directions_have_no_sensitivity(self):
        # add gadget error generators that are pure MCM gauge (or gauge-equivalent pairs) to the MCM ansatz
        ansatz = dict(self.ansatz)
        ansatz[('Iz', 0)] = dict(self.ansatz[('Iz', 0)])
        ansatz[('Iz', 0)].update({('S', 'ZI'): 0.0, ('S', 'IZ'): 0.0, ('S', 'ZZ'): 0.0, ('H', 'ZI'): 0.0, ('H', 'IZ'): 0.0,
                                  ('H', 'ZZ'): 0.0, ('H', 'YY'): 0.0, ('S', 'IY'): 0.0})
        params, _ = lgst.build_model_parameter_indexing(ansatz, 2)
        D = np.vstack(lgst.create_design_matrix_list(self.circuits, params))
        col = {(p.errorgen_type, str(p.errorgen.basis_element_labels[0])[1:].replace('_', 'I')): j
               for j, p in enumerate(params) if p.key == ('Iz', 0)}
        for eg in [('S', 'IIZ'), ('S', 'ZII'), ('S', 'ZIZ'), ('H', 'ZII'), ('H', 'IIZ'), ('H', 'ZIZ')]:
            self.assertLess(np.abs(D[:, col[eg]]).max(), 1e-12)   # exactly zero columns
        self.assertLess(np.abs(D[:, col[('H', 'XIX')]] + D[:, col[('H', 'YIY')]]).max(), 1e-12)  # H_XX + H_YY is gauge
        self.assertArraysAlmostEqual(D[:, col[('S', 'IIX')]], D[:, col[('S', 'IIY')]])       # S_IX ~ S_IY
        self.assertGreater(np.abs(D[:, col[('S', 'IIX')]]).max(), 0.1)
        # the FOMGI ansatz itself is free of MCM-gauge redundancy
        F = mg.mcm_crunch_matrix(list(self.ansatz[('Iz', 0)].keys()))
        self.assertEqual(np.linalg.matrix_rank(F, tol=1e-9), 13)

    def test_multiple_and_parallel_mcms(self):
        ansatz = dict(self.ansatz)
        ansatz[('Iz', 1)] = {('S', 'IX'): 0.003, ('H', 'ZY'): 0.006, ('S', 'XX:0,v'): 0.001}  # incl. crosstalk onto qubit 0
        params, rates = lgst.build_model_parameter_indexing(ansatz, 2)
        true_rates = np.array(rates)
        circuits = [
            Circuit([Label('Gxpi2', 0), Label('Iz', (0, 1)), Label('Gcphase', (0, 1)), Label('Gypi2', 1)], line_labels=(0, 1)),
            Circuit([Label('Gypi2', 0), Label('Iz', 0), Label('Gxpi2', 0), Label('Iz', 0), Label('Gcphase', (0, 1))], line_labels=(0, 1)),
            Circuit([Label([Label('Iz', 1), Label('Gxpi2', 0)]), Label('Gcphase', (0, 1)), Label([Label('Gypi2', 0), Label('Gypi2', 1)])], line_labels=(0, 1)),
            Circuit([Label('Iz', 1), Label('Gxpi2', 1), Label('Iz', 1)], line_labels=(0, 1)),
        ]
        designs = lgst.create_design_matrix_list(circuits, params, return_info=True, max_pauli_weight=3)
        self.assertEqual([d.num_mcms for d in designs], [2, 2, 1, 2])
        self.assertEqual([d.shape[0] for d in designs], [14, 14, 7, 14])
        obs, ideal = _simulate(self.pspec, ansatz, circuits, params, multiqubit_mcm_labels=[(0, 1)])
        self.assertIn(('01', '00'), obs[0])       # multi-qubit instrument outcome
        self.assertIn(('0', '1', '00'), obs[1])   # two sequential MCMs
        for d, i in zip(designs, ideal):
            self.assertArraysAlmostEqual(lgst.compute_pauli_expectations(i, d.paulis), d.ideal_expectations)
        shifts, _ = lgst.observed_expectation_shifts(designs, [obs, ideal])
        D = np.vstack([d.design_matrix for d in designs])
        self.assertGreater(np.abs(shifts).max(), 0.005)
        self.assertLess(np.abs(shifts - D @ true_rates).max(), 0.15 * np.abs(shifts).max())

    def test_full_hsca_error_model(self):
        """Pauli-correlation ('C') and active ('A') errors on gates and on the MCM, with a physical (CPTP) true model."""
        def relaxation(P, Q, gamma):  # amplitude damping (decay probability gamma) in elementary-errorgen form
            return {('S', P): gamma / 4, ('S', Q): gamma / 4, ('A', P, Q): -gamma / 4}

        true_model = {k: dict(v) for k, v in self.ansatz.items()}
        for q in (0, 1):
            for gate, axis in [('Gxpi2', 'X'), ('Gypi2', 'Y')]:
                d = true_model[(gate, q)]
                for k, v in relaxation('X', 'Y', 0.004).items():
                    d[k] = d.get(k, 0.0) + v
                d[('C', 'X', 'Y')] = 0.0004 if axis == 'X' else 0.0  # tilted dephasing axis (C^2 <= S_X S_Y)
        d = true_model[('Gcphase', 0, 1)]
        d.update({('S', 'ZI'): 0.001, ('S', 'IZ'): 0.001, ('C', 'ZI', 'IZ'): 0.0008})  # correlated dephasing
        # MCM: FOMGI H+S representatives plus relaxation before/after the record is formed and an asymmetric readout
        gadget = dict(mg.fomgi_ansatz(rates=self.fomgi_truth))
        for (P, Q, gamma) in [('XX', 'YX', 0.004), ('XI', 'YI', 0.004), ('IX', 'IY', 0.003)]:
            for k, v in relaxation(P, Q, gamma).items():
                gadget[k] = gadget.get(k, 0.0) + v
        true_model[('Iz', 0)] = gadget

        # the ansatz used for the fit: same gate terms, all 28 FOMGI quantities for the MCM
        ansatz = {k: dict(v) for k, v in true_model.items()}
        ansatz[('Iz', 0)] = mg.fomgi_ansatz(sectors='all')
        params, _ = lgst.build_model_parameter_indexing(ansatz, 2)
        types = np.array(params.errorgen_types())
        self.assertEqual(len(params.indices_for_key(('Iz', 0))), 28)
        self.assertTrue((types == 'C').sum() >= 13 and (types == 'A').sum() >= 6)

        # truth expressed in the coordinates of the ansatz (MCM part via the FOMGI projection of the gadget model)
        fomgi_true = mg.fomgi_quantities(gadget, sectors='all')
        self.assertAlmostEqual(fomgi_true['a_prep'], -0.001)
        self.assertAlmostEqual(fomgi_true['a_read'], -0.00075)
        self.assertAlmostEqual(fomgi_true['a_meas'], -0.001)
        self.assertAlmostEqual(fomgi_true['s_prep'], self.fomgi_truth['s_prep'] + 0.002)  # S_YI ~ S_XI in the crunch map
        truth = {k: dict(v) for k, v in true_model.items()}
        truth[('Iz', 0)] = mg.fomgi_ansatz(sectors='all', rates=fomgi_true)
        params_t, true_rates = lgst.build_model_parameter_indexing(truth, 2)
        self.assertEqual(list(params_t), list(params))
        true_rates = np.array(true_rates)

        circuits = _random_circuits(self.rng, 200, 5, mcm_qubit=0, mcm_every=2)
        designs = lgst.create_design_matrix_list(circuits, params, return_info=True)
        self.assertEqual(lgst.unidentifiable_directions(designs, params), [])

        # rate_to_model realizes C/A terms exactly (mixed-support pairs, either Pauli order, non-CP estimates)
        noisy = lgst.rate_to_model(self.pspec, true_model)
        swapped = {k: dict(v) for k, v in true_model.items()}
        swapped[('Gcphase', 0, 1)] = {(k[0], k[2], k[1]) if k[0] == 'C' else k: v for k, v in d.items()}
        noisy_swapped = lgst.rate_to_model(self.pspec, swapped)
        obs = [dict(noisy.probabilities(c)) for c in circuits]
        for c, p in zip(circuits[:10], obs[:10]):
            self.assertGreaterEqual(min(p.values()), -1e-12)
            for outcome, prob in noisy_swapped.probabilities(c).items():
                self.assertAlmostEqual(prob, p[outcome], places=10)
        rates, _ = lgst.estimate_error_rates(designs, [obs, None], None, params)
        for typ in 'HSCA':
            self.assertLess(np.abs(rates - true_rates)[types == typ].max(), 6e-4, msg="sector %s" % typ)
        self.assertTrue(np.all(rates[types == 'S'] >= 0))  # stochastic rates stay non-negative in the joint solve
        fomgi = lgst.mcm_fomgi_estimates(params, rates)[('Iz', 0)]
        self.assertEqual(len(fomgi), 28)
        for name in ('a_prep', 'a_read', 'a_meas', 's_prep', 's_read', 'rt_meas_x', 'wt0'):
            self.assertAlmostEqual(fomgi[name], fomgi_true[name], delta=6e-4)
        lgst.rate_to_model(self.pspec, ansatz, rate_ests=rates)  # estimated (not necessarily CP) rates are accepted

    def test_user_supplied_paulis_are_padded(self):
        circ = self.circuits[0]  # contains an MCM
        design = lgst.create_circuit_design_matrix(circ, self.params, pauli_measurements=[stim.PauliString('ZI')],
                                                   return_info=True)
        self.assertEqual(design.shape[0], 1)
        self.assertEqual(str(design.paulis[0]), '+Z__')
        design2 = lgst.create_circuit_design_matrix(circ, self.params, pauli_measurements=lambda n, m: [stim.PauliString('I' * n + 'Z' * m)],
                                                    return_info=True)
        self.assertEqual(str(design2.paulis[0]), '+__Z')
        self.assertGreater(np.abs(design2.design_matrix).max(), 0.1)  # the MCM outcome is sensitive to readout error

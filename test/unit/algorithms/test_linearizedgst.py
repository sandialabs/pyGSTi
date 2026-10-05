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

    def test_predicted_error_bars(self):
        designs = lgst.create_design_matrix_list(self.circuits, self.params, return_info=True)
        eb = lgst.predicted_error_bars(designs, shots=1000)
        self.assertEqual(eb.shape, (len(self.params),))
        self.assertTrue(np.all(np.isfinite(eb)) and np.all(eb > 0))
        self.assertArraysAlmostEqual(lgst.predicted_error_bars(designs, shots=4000), eb / 2)   # 1/sqrt(shots)
        # bare arrays: independent unit-variance observables, i.e. sqrt(diag((D^T D)^-1) / shots)
        D = np.vstack(designs)
        eb_plain = lgst.predicted_error_bars(D, shots=1000)
        self.assertArraysAlmostEqual(eb_plain, np.sqrt(np.diag(np.linalg.inv(D.T @ D)) / 1000))
        eb_arrays = lgst.predicted_error_bars([d.design_matrix for d in designs], shots=1000)
        self.assertArraysAlmostEqual(eb_arrays, eb_plain)
        # the ideal-state covariance accounts for correlated observables (Z0, Z1 vs Z0Z1 from the same shots)
        self.assertFalse(np.allclose(eb, eb_plain))
        # rank-deficient designs give infinite error bars for the parameters that are not identifiable
        eb_few = lgst.predicted_error_bars(designs[:2], shots=1000)
        self.assertTrue(np.any(np.isinf(eb_few)))
        # the Hamiltonian predictions agree with a shot-noise bootstrap (the stochastic ones are upper bounds)
        obs, _ = _simulate(self.pspec, self.ansatz, self.circuits, self.params)
        _, boot = lgst.estimate_error_rates(designs, [obs, None], None, self.params,
                                            error_bars='bootstrap', error_bar_params=[100, 1000], seed=5)
        types = np.array(self.params.errorgen_types())
        ratio = boot[types == 'H'] / eb[types == 'H']
        self.assertLess(np.abs(np.log(ratio)).max(), np.log(1.5))
        self.assertTrue(np.all(boot[types == 'S'] <= eb[types == 'S']))

    def test_select_circuits(self):
        designs = lgst.create_design_matrix_list(self.circuits, self.params, return_info=True)
        P, n = len(self.params), len(designs)
        gram = [d.design_matrix.T @ d.design_matrix for d in designs]

        def trace_inv(idx):
            return np.trace(np.linalg.inv(sum(gram[i] for i in idx)))

        selected = lgst.select_circuits(designs, 12)
        self.assertEqual(len(selected), 12)
        self.assertEqual(len(set(selected)), 12)
        self.assertEqual(np.linalg.matrix_rank(np.vstack([designs[i] for i in selected])), P)
        self.assertEqual(lgst.unidentifiable_directions([designs[i] for i in selected], self.params), [])
        # better (A-criterion) than the first 12 random circuits, which are not even full rank here
        self.assertLess(np.linalg.matrix_rank(np.vstack(designs[:12])), P)
        self.assertLess(trace_inv(selected), trace_inv(range(n)) * 4)

        # forward selection is nested, and every step is a greedy-optimal choice (checked against direct
        # recomputation of the criterion; circuits with identical information content tie exactly, so only the
        # criterion values -- not the indices -- are compared)
        self.assertEqual(lgst.select_circuits(designs, 8), selected[:8])
        criteria = {'A': lambda M: -np.trace(np.linalg.inv(M)), 'D': lambda M: np.linalg.slogdet(M)[1],
                    'G': lambda M: -np.max(np.diag(np.linalg.inv(M))), 'E': lambda M: np.linalg.eigvalsh(M)[0]}
        ridge = 1e-6 * np.eye(P)
        for criterion, crit in criteria.items():
            sel = lgst.select_circuits(designs, 12, criterion=criterion)
            M = ridge.copy()
            for step, j in enumerate(sel):
                best = max(crit(M + gram[i]) for i in range(n) if i not in sel[:step])
                self.assertAlmostEqual(crit(M + gram[j]), best, delta=1e-9 * max(1.0, abs(best)))
                M = M + gram[j]
            # backward elimination reaches the same criterion value as a brute-force greedy elimination
            sel_b = lgst.select_circuits(designs, 12, criterion=criterion, method='backward')
            self.assertEqual(len(sel_b), 12)
            self.assertEqual(np.linalg.matrix_rank(np.vstack([designs[i] for i in sel_b])), P)
            remaining, M = list(range(n)), ridge + sum(gram)
            while len(remaining) > 12:
                j = remaining[int(np.argmax([crit(M - gram[i]) for i in remaining]))]
                remaining.remove(j)
                M = M - gram[j]
            self.assertAlmostEqual(crit(ridge + sum(gram[i] for i in sel_b)), crit(M),
                                   delta=1e-9 * max(1.0, abs(crit(M))))

        # backward elimination
        backward = lgst.select_circuits(designs, 12, method='backward')
        self.assertEqual(len(backward), 12)
        self.assertEqual(backward, sorted(backward))
        self.assertEqual(np.linalg.matrix_rank(np.vstack([designs[i] for i in backward])), P)
        self.assertEqual(lgst.select_circuits(designs, n, method='backward'), list(range(n)))
        self.assertEqual(lgst.select_circuits(designs, 0), [])

        # stopping at a target precision
        target = 1.5 * lgst.predicted_error_bars(designs, shots=1000).max()
        sel = lgst.select_circuits(designs, max_error_bar=target, shots=1000)
        self.assertLess(len(sel), n)
        self.assertLessEqual(lgst.predicted_error_bars([designs[i] for i in sel], shots=1000).max(), target)
        self.assertGreater(lgst.predicted_error_bars([designs[i] for i in sel[:-1]], shots=1000).max(), target)
        sel_b = lgst.select_circuits(designs, max_error_bar=target, shots=1000, method='backward')
        self.assertLess(len(sel_b), n)
        self.assertLessEqual(lgst.predicted_error_bars([designs[i] for i in sel_b], shots=1000).max(), target)
        with self.assertWarns(UserWarning):   # the cap stops the search before the target is reached
            self.assertEqual(len(lgst.select_circuits(designs, num_circuits=5, max_error_bar=target, shots=1000)), 5)

        # warm start / protected circuits
        warm = lgst.select_circuits(designs, 12, include=[5, 3, 5])
        self.assertEqual(warm[:2], [5, 3])
        self.assertEqual(len(warm), 12)
        M = ridge + sum(gram[i] for i in warm[:2])
        for step in range(2, 12):
            best = max(criteria['A'](M + gram[i]) for i in range(n) if i not in warm[:step])
            self.assertAlmostEqual(criteria['A'](M + gram[warm[step]]), best, delta=1e-9 * abs(best))
            M = M + gram[warm[step]]
        kept = lgst.select_circuits(designs, 12, method='backward', include=[0, 1])
        self.assertTrue({0, 1} <= set(kept) and len(kept) == 12)
        with self.assertRaises(ValueError):
            lgst.select_circuits(designs, 1, include=[0, 1])
        with self.assertRaises(ValueError):
            lgst.select_circuits(designs, 5, include=[n])

        with self.assertRaises(ValueError):   # unreachable target
            lgst.select_circuits(designs, max_error_bar=target / 100, shots=1000)
        with self.assertRaises(ValueError):
            lgst.select_circuits(designs, max_error_bar=target / 100, shots=1000, method='backward')
        with self.assertRaises(ValueError):   # neither a number of circuits nor a target
            lgst.select_circuits(designs)
        with self.assertRaises(ValueError):
            lgst.select_circuits(designs, 12, criterion='Z')
        with self.assertRaises(ValueError):
            lgst.select_circuits(designs, 12, method='sideways')
        with self.assertRaises(ValueError):
            lgst.select_circuits(designs, n + 1)

    @pytest.mark.skipif(importlib.util.find_spec('pathos') is None, reason="pathos not installed")
    def test_parallel_pathos(self):
        serial = lgst.create_design_matrix_list(self.circuits[:6], self.params)
        parallel = lgst.create_design_matrix_list_parallel_pathos(self.circuits[:6], self.params, do_parallel=True,
                                                                  num_workers=2)
        for a, b in zip(serial, parallel):
            self.assertArraysAlmostEqual(a, b)


_HAS_CVXPY = importlib.util.find_spec('cvxpy') is not None


class ShotAllocationTester(BaseCase):
    """Allocating a fixed shot budget across a candidate pool, at possibly unequal numbers per circuit."""

    def setUp(self):
        self.rng = np.random.default_rng(2025)
        self.ansatz = _gate_ansatz(self.rng)
        self.params, rates = lgst.build_model_parameter_indexing(self.ansatz, 2)
        self.true_rates = np.array(rates)
        self.circuits = _random_circuits(self.rng, 60, 4)
        self.designs = lgst.create_design_matrix_list(self.circuits, self.params, return_info=True)
        self.budget = 60000.0

    def test_neyman_shot_allocation(self):
        P, n = len(self.params), len(self.designs)
        shots, objective = lgst.neyman_shot_allocation(self.designs, self.budget)
        self.assertEqual(shots.shape, (n,))
        self.assertTrue(np.all(shots >= 0))
        self.assertAlmostEqual(shots.sum(), self.budget, delta=1e-6 * self.budget)

        # the returned objective is exactly the achieved sum of predicted variances (on the nonzero-shot
        # circuits -- a zero allocation means "drop this circuit", not "infinite noise")
        active = shots > 1e-6
        achieved = np.sum(lgst.predicted_error_bars([self.designs[i] for i in range(n) if active[i]],
                                                    shots=shots[active]) ** 2)
        self.assertAlmostEqual(achieved, objective, delta=1e-9 * objective)

        # it is at least as good as equal shots (equality only in degenerate cases), for the same estimator
        equal_objective = np.sum(lgst.predicted_error_bars(self.designs, shots=self.budget / n) ** 2)
        self.assertLess(objective, equal_objective)

        # closed-form check on an orthogonal 2-circuit toy problem: two scalar parameters, each measured by
        # its own circuit with per-shot sensitivity (information) f1, f2; Cauchy-Schwarz gives N1 ∝ sqrt(f2)
        D1, D2 = np.array([[3.0, 0.0]]), np.array([[0.0, 1.0]])
        f1, f2 = 9.0, 1.0
        shots2, obj2 = lgst.neyman_shot_allocation([D1, D2], self.budget)
        n1_analytic = self.budget * np.sqrt(f2) / (np.sqrt(f1) + np.sqrt(f2))
        self.assertAlmostEqual(shots2[0], n1_analytic, delta=1e-6 * self.budget)
        self.assertAlmostEqual(obj2, (1 / np.sqrt(f1) + 1 / np.sqrt(f2)) ** 2 / self.budget, delta=1e-12)

        # a weight matrix that only scores one parameter concentrates the budget on circuits that inform it
        weight = np.zeros((P, P))
        weight[0, 0] = 1.0
        shots_w, obj_w = lgst.neyman_shot_allocation(self.designs, self.budget, weight=weight)
        self.assertAlmostEqual(shots_w.sum(), self.budget, delta=1e-6 * self.budget)
        var0_weighted = lgst.predicted_error_bars([self.designs[i] for i in range(n) if shots_w[i] > 1e-6],
                                                   shots=shots_w[shots_w > 1e-6])[0] ** 2
        self.assertAlmostEqual(var0_weighted, obj_w, delta=1e-9 * max(obj_w, 1e-30))
        self.assertLessEqual(var0_weighted, achieved)  # at least as good at the one thing it was asked to optimize

        with self.assertRaises(ValueError):
            lgst.neyman_shot_allocation(self.designs, 0)
        with self.assertRaises(ValueError):  # rank-deficient support
            lgst.neyman_shot_allocation(self.designs[:2], self.budget)

    @pytest.mark.skipif(not _HAS_CVXPY, reason="cvxpy not installed")
    def test_optimal_shot_allocation(self):
        P, n = len(self.params), len(self.designs)

        def info(shots_per_circuit):
            return sum(s * (d.design_matrix.T @ d.design_matrix) for s, d in zip(shots_per_circuit, self.designs))

        equal = np.full(n, self.budget / n)
        equal_info = info(equal)

        shots, objective = lgst.optimal_shot_allocation(self.designs, self.budget, criterion='A')
        self.assertEqual(shots.shape, (n,))
        self.assertTrue(np.all(shots >= -1e-6))
        self.assertAlmostEqual(shots.sum(), self.budget, delta=1e-3 * self.budget)
        self.assertAlmostEqual(np.trace(np.linalg.inv(info(shots))), objective, delta=1e-5 * objective)
        self.assertLess(objective, np.trace(np.linalg.inv(equal_info)))  # beats equal shots on its own criterion

        shots_d, objective_d = lgst.optimal_shot_allocation(self.designs, self.budget, criterion='D')
        self.assertAlmostEqual(np.linalg.slogdet(info(shots_d))[1], objective_d, delta=1e-4)
        self.assertGreater(objective_d, np.linalg.slogdet(equal_info)[1])

        shots_e, objective_e = lgst.optimal_shot_allocation(self.designs, self.budget, criterion='E')
        self.assertAlmostEqual(np.linalg.eigvalsh(info(shots_e))[0], objective_e, delta=1e-4 * objective_e)
        self.assertGreater(objective_e, np.linalg.eigvalsh(equal_info)[0])

        # closed-form check: same orthogonal 2-circuit toy as the Neyman test, now via the convex solver
        D1, D2 = np.array([[3.0, 0.0]]), np.array([[0.0, 1.0]])
        f1, f2 = 9.0, 1.0
        shots2, obj2 = lgst.optimal_shot_allocation([D1, D2], self.budget, criterion='A')
        n1_analytic = self.budget * np.sqrt(f2) / (np.sqrt(f1) + np.sqrt(f2))
        self.assertAlmostEqual(shots2[0], n1_analytic, delta=1e-3 * self.budget)
        self.assertAlmostEqual(obj2, (1 / np.sqrt(f1) + 1 / np.sqrt(f2)) ** 2 / self.budget, delta=1e-4 * obj2)

        # a weight matrix restricted to one parameter does at least as well on that parameter as unweighted A
        weight = np.zeros((P, P))
        weight[0, 0] = 1.0
        shots_w, objective_w = lgst.optimal_shot_allocation(self.designs, self.budget, criterion='A', weight=weight)
        self.assertLessEqual(objective_w, 1.0 / info(equal)[0, 0])
        self.assertLessEqual(objective_w, objective)  # focusing the budget can only help that one parameter

        # min_shots: every circuit that is used clears the floor, and the objective is barely worse
        shots_pruned, objective_pruned = lgst.optimal_shot_allocation(self.designs, self.budget, criterion='A',
                                                                      min_shots=200)
        active = shots_pruned > 0
        self.assertTrue(np.all(shots_pruned[active] >= 200 - 1e-6))
        self.assertGreaterEqual(objective_pruned, objective)
        self.assertLess(objective_pruned, 1.1 * objective)

        with self.assertRaises(ValueError):
            lgst.optimal_shot_allocation(self.designs, self.budget, criterion='Z')
        with self.assertRaises(ValueError):
            lgst.optimal_shot_allocation(self.designs, 0)
        with self.assertWarns(UserWarning):   # `weight` is ignored for 'D'/'E'
            lgst.optimal_shot_allocation(self.designs, self.budget, criterion='D', weight=np.eye(P))
        with self.assertRaises(ValueError):   # the whole pool can't individually clear an impossible floor
            lgst.optimal_shot_allocation(self.designs, self.budget, criterion='A', min_shots=self.budget)

    @pytest.mark.skipif(_HAS_CVXPY, reason="only meaningful without cvxpy installed")
    def test_optimal_shot_allocation_requires_cvxpy(self):
        with self.assertRaises(ImportError):
            lgst.optimal_shot_allocation(self.designs, self.budget)


class NominalDesignTester(BaseCase):
    """Experiment design under a nominal error model: exact shot-noise statistics, Fisher information, optimal
    allocation with an optimality certificate, and the matched generalized-least-squares estimator -- on a
    2-qubit model with crosstalk, SPAM errors and a mid-circuit measurement."""

    def setUp(self):
        self.rng = np.random.default_rng(11)
        self.pspec = QubitProcessorSpec(2, ['Gxpi2', 'Gypi2', 'Gcphase'], geometry='line')
        self.ansatz = _gate_ansatz(self.rng)
        self.ansatz['prep'] = {('S', 'X:0'): 0.002, ('S', 'X:1'): 0.001}
        self.ansatz['povm'] = {('S', 'X:0'): 0.003, ('S', 'X:1'): 0.0015}
        self.ansatz[('Iz', 0)] = mg.fomgi_ansatz(rates={'s_read': 0.01, 's_prep': 0.002, 'r_meas_y': 0.005, 'w0': -0.004})
        self.params, rates = lgst.build_model_parameter_indexing(self.ansatz, 2)
        self.true_rates = np.array(rates)
        self.types = np.array(self.params.errorgen_types())
        self.circuits = _random_circuits(self.rng, 80, 4, mcm_qubit=0, mcm_every=2)
        self.designs = lgst.create_design_matrix_list(self.circuits, self.params, return_info=True)
        self.moments = lgst.create_design_moments(self.designs, self.params)
        self.budget = 2.0e6

    def test_moments_reproduce_the_multinomial_covariance(self):
        obs, _ = _simulate(self.pspec, self.ansatz, self.circuits[:12], self.params)
        for m, p in zip(self.moments[:12], obs):
            self.assertIsInstance(m, lgst.CircuitMoments)
            k = m.num_observables
            self.assertEqual(m.pair_index.shape, (k, k))
            self.assertTrue(np.all(np.diag(m.pair_index) == -1))
            # exact covariance of the Z-type observable estimates from the exact outcome distribution
            paulis = list(m.design.paulis)
            mexact = lgst.compute_pauli_expectations(p, paulis)
            Mexact = np.eye(k)
            for a in range(k):
                for b in range(a + 1, k):
                    Mexact[a, b] = Mexact[b, a] = lgst.compute_pauli_expectations(p, [paulis[a] * paulis[b]])[0]
            Sigma_exact = Mexact - np.outer(mexact, mexact)
            Sigma_nom = m.covariance(self.true_rates, min_variance=0.0)
            self.assertLess(np.abs(Sigma_nom - Sigma_exact).max(), 3e-3)   # first order in the rates
            # observables with ideal value ±1 have a variance of the order of the rates, not 1
            pm1 = np.abs(m.design.ideal_expectations) > 0.5
            if pm1.any():
                self.assertLess(np.diag(Sigma_nom)[pm1].max(), 0.2)
                self.assertGreater(np.diag(Sigma_nom)[pm1].min(), 0.0)
            # the floor makes the matrix positive definite
            floored = lgst.nominal_observable_covariance(m, self.true_rates, min_variance=1e-3)
            self.assertGreaterEqual(np.linalg.eigvalsh(floored)[0], 1e-3 - 1e-12)
        # moments can also be built directly from circuits
        m2 = lgst.create_design_moments(self.circuits[:2], self.params)
        self.assertArraysAlmostEqual(m2[0].design.design_matrix, self.moments[0].design.design_matrix)
        self.assertArraysAlmostEqual(m2[0].product_design_matrix, self.moments[0].product_design_matrix)
        nominal = lgst.nominal_rates_by_type(self.params, {'H': 0.0, 'S': 1e-3})
        self.assertTrue(np.all(nominal[self.types == 'S'] == 1e-3) and np.all(nominal[self.types == 'H'] == 0))

    def test_information_and_predicted_covariance(self):
        info = lgst.circuit_information(self.moments, self.true_rates, min_variance=1e-3)
        self.assertIsInstance(info, lgst.DesignInformation)
        self.assertEqual(len(info), len(self.moments))
        P = len(self.params)
        shots = self.rng.uniform(0, 1, len(info)) * self.budget
        # the information of an allocation is the shot-weighted sum of D_cᵀ Σ_c⁻¹ D_c
        M_direct = np.zeros((P, P))
        for m, s in zip(self.moments, shots):
            D = m.design.design_matrix
            M_direct += s * D.T @ np.linalg.solve(m.covariance(self.true_rates, 1e-3), D)
        self.assertTrue(np.allclose(info.information(shots), M_direct, rtol=1e-9, atol=1e-9 * np.abs(M_direct).max()))
        self.assertArraysAlmostEqual(info.matrices()[3], info.blocks[3].T @ info.blocks[3])
        cov = lgst.predicted_covariance(info, shots)
        self.assertArraysAlmostEqual(cov @ M_direct, np.eye(P))
        self.assertTrue(np.allclose(lgst.predicted_covariance(self.moments, shots, self.true_rates, 'gls'), cov,
                                    rtol=1e-9, atol=1e-12 * np.abs(cov).max()))
        eb = lgst.predicted_error_bars_nominal(info, shots)
        self.assertArraysAlmostEqual(eb, np.sqrt(np.diag(cov)))
        # Gauss-Markov: the plain pseudo-inverse estimator is never more precise than GLS, in any weight
        cov_ols = lgst.predicted_covariance(self.moments, shots, self.true_rates, 'ols')
        self.assertGreaterEqual(np.linalg.eigvalsh(cov_ols - cov)[0], -1e-9 * np.trace(cov))
        self.assertLess(np.trace(cov), np.trace(cov_ols))
        # and the old worst-case prediction is a (loose) upper bound for the stochastic rates
        old = lgst.predicted_error_bars(self.designs, shots=shots)
        self.assertTrue(np.all(old[self.types == 'S'] > np.sqrt(np.diag(cov_ols))[self.types == 'S']))
        # criteria
        self.assertAlmostEqual(lgst.design_criterion(info, shots, 'A'), np.trace(cov), delta=1e-9 * np.trace(cov))
        self.assertAlmostEqual(lgst.design_criterion(info, shots, 'D'), np.linalg.slogdet(M_direct)[1], delta=1e-6)
        self.assertAlmostEqual(lgst.design_criterion(info, shots, 'E'), np.linalg.eigvalsh(M_direct)[0],
                               delta=1e-6 * np.linalg.eigvalsh(M_direct)[0])
        # a subset keeps the blocks and costs
        sub = info.subset([5, 1])
        self.assertArraysAlmostEqual(sub.blocks[1], info.blocks[1])
        # the full-distribution information is at least as large as that of the weight-<=2 observables
        prob_designs = lgst.create_design_matrix_list(self.circuits[:5], self.params, usePaulis=False, return_info=True)
        pinfo = lgst.probability_information(prob_designs, self.true_rates, min_probability=5e-4)
        oinfo = lgst.circuit_information(self.moments[:5], self.true_rates, min_variance=1e-3)
        gap = pinfo.information(1.0) - oinfo.information(1.0)
        self.assertGreater(np.linalg.eigvalsh(gap)[0], -1e-6 * np.trace(oinfo.information(1.0)))
        with self.assertRaises(ValueError):
            lgst.probability_information(self.designs[:2], self.true_rates)
        with self.assertRaises(ValueError):
            lgst.circuit_information(self.moments, self.true_rates[:3])

    def test_optimal_allocation(self):
        info = lgst.circuit_information(self.moments, self.true_rates)
        n, P = len(info), len(self.params)
        equal = np.full(n, self.budget / n)
        equal_obj = lgst.design_criterion(info, equal, 'A')

        shots, res = lgst.optimal_allocation(info, self.budget, criterion='A', tol=1e-3)
        self.assertEqual(shots.shape, (n,))
        self.assertTrue(np.all(shots >= 0))
        self.assertAlmostEqual(shots.sum(), self.budget, delta=1e-6 * self.budget)
        self.assertEqual(res.method, 'multiplicative')
        self.assertLessEqual(res.gap, 1e-3)                     # equivalence-theorem certificate
        self.assertEqual(res.support_size, int((shots > 0).sum()))
        self.assertLess(res.support_size, n)                    # some candidates are not worth running
        self.assertAlmostEqual(res.objective, lgst.design_criterion(info, shots, 'A'), delta=1e-9 * res.objective)
        self.assertLess(res.objective, 0.9 * equal_obj)
        # the certificate is a true bound: a tighter solve cannot beat the certified value
        shots_tight, res_tight = lgst.optimal_allocation(info, self.budget, criterion='A', tol=1e-5, max_iter=50000)
        self.assertGreaterEqual(res.objective * (1 + 1e-12), res_tight.objective)
        self.assertLessEqual(res.objective, res_tight.objective * (1 + 1e-3))
        # the equivalence theorem holds at the optimum: no candidate has a larger directional derivative
        M = info.information(shots_tight)
        Minv = np.linalg.inv(M)
        scores = np.array([np.trace(Minv @ Minv @ F) for F in info.matrices()])
        self.assertLessEqual(scores.max() * self.budget, np.trace(Minv) * (1 + 1e-4))

        shots_d, res_d = lgst.optimal_allocation(info, self.budget, criterion='D')
        self.assertAlmostEqual(res_d.objective, np.linalg.slogdet(info.information(shots_d))[1], delta=1e-6)
        self.assertGreater(res_d.objective, lgst.design_criterion(info, equal, 'D'))

        # closed-form 2-circuit toy: information f1, f2 on two orthogonal parameters -> N1 ∝ sqrt(f2)
        toy = lgst.DesignInformation([np.array([[3.0, 0.0]]), np.array([[0.0, 1.0]])])
        shots2, res2 = lgst.optimal_allocation(toy, self.budget, criterion='A', tol=1e-8)
        f1, f2 = 9.0, 1.0
        self.assertAlmostEqual(shots2[0], self.budget * np.sqrt(f2) / (np.sqrt(f1) + np.sqrt(f2)), delta=1e-4 * self.budget)
        self.assertAlmostEqual(res2.objective, (1 / np.sqrt(f1) + 1 / np.sqrt(f2)) ** 2 / self.budget, delta=1e-6 * res2.objective)

        # a weight restricted to one parameter does at least as well on that parameter
        weight = np.zeros(P)
        weight[0] = 1.0
        shots_w, res_w = lgst.optimal_allocation(info, self.budget, criterion='A', weight=weight)
        self.assertLessEqual(res_w.objective, lgst.predicted_covariance(info, shots)[0, 0] * (1 + 1e-6))

        # costs: the budget is spent in cost units
        cost = self.rng.uniform(1, 3, n)
        info_cost = lgst.circuit_information(self.moments, self.true_rates, cost=cost)
        shots_c, res_c = lgst.optimal_allocation(info_cost, self.budget, criterion='A')
        self.assertAlmostEqual(np.dot(cost, shots_c), self.budget, delta=1e-6 * self.budget)
        self.assertAlmostEqual(res_c.objective, lgst.design_criterion(info, shots_c, 'A'), delta=1e-9 * res_c.objective)

        # min_shots: every used circuit clears the floor at a small price
        shots_m, res_m = lgst.optimal_allocation(info, self.budget, criterion='A', min_shots=5000)
        self.assertTrue(np.all(shots_m[shots_m > 0] >= 5000 - 1e-6))
        self.assertLess(res_m.objective, 1.2 * res.objective)

        # rounding
        rounded = lgst.round_shots(shots, self.budget)
        self.assertEqual(rounded.dtype.kind, 'i')
        self.assertEqual(rounded.sum(), int(self.budget))
        self.assertTrue(np.all(rounded[shots == 0] == 0))
        self.assertTrue(np.all(np.abs(rounded - shots) < 1))

        with self.assertRaises(ValueError):
            lgst.optimal_allocation(info, self.budget, criterion='Z')
        with self.assertRaises(ValueError):
            lgst.optimal_allocation(info, 0)
        with self.assertRaises(ValueError):   # 'E' needs cvxpy
            lgst.optimal_allocation(info, self.budget, criterion='E', method='multiplicative')
        with self.assertRaises(TypeError):
            lgst.optimal_allocation(self.designs, self.budget)
        with self.assertRaises(ValueError):   # rank-deficient pool
            lgst.optimal_allocation(info.subset([0, 1]), self.budget)

    @pytest.mark.skipif(not _HAS_CVXPY, reason="cvxpy not installed")
    def test_optimal_allocation_agrees_with_cvxpy(self):
        info = lgst.circuit_information(self.moments, self.true_rates)
        shots, res = lgst.optimal_allocation(info, self.budget, criterion='A', tol=1e-4)
        shots_c, res_c = lgst.optimal_allocation(info, self.budget, criterion='A', method='cvxpy')
        self.assertEqual(res_c.method, 'cvxpy')
        self.assertAlmostEqual(res_c.objective, res.objective, delta=5e-3 * res.objective)
        shots_e, res_e = lgst.optimal_allocation(info, self.budget, criterion='E')
        self.assertAlmostEqual(res_e.objective, np.linalg.eigvalsh(info.information(shots_e))[0],
                               delta=1e-4 * res_e.objective)
        self.assertGreater(res_e.objective, lgst.design_criterion(info, self.budget / len(info), 'E'))
        shots_cap, _ = lgst.optimal_allocation(info, self.budget, criterion='A', method='cvxpy',
                                               max_shots=0.1 * self.budget)
        self.assertLessEqual(shots_cap.max(), 0.1 * self.budget * (1 + 1e-4))

    def test_gls_estimator(self):
        obs, _ = _simulate(self.pspec, self.ansatz, self.circuits, self.params)
        shots = np.full(len(self.circuits), self.budget / len(self.circuits))
        # exact data: the GLS estimate reproduces the rates up to the linearization error
        rates, eb = lgst.estimate_error_rates_gls(self.moments, [obs, None], self.params, shots)
        self.assertLess(np.abs(rates - self.true_rates).max(), 6e-4)
        self.assertTrue(np.all(rates[self.types == 'S'] >= 0))
        self.assertEqual(eb.shape, rates.shape)
        info = lgst.circuit_information(self.moments, rates)
        self.assertArraysAlmostEqual(eb, lgst.predicted_error_bars_nominal(info, shots), places=6)
        rates_inv, _ = lgst.estimate_error_rates_gls(self.moments, [obs, None], self.params, shots,
                                                     stochastic_solver='inversion', error_bars=None)
        self.assertLess(np.abs(rates_inv - self.true_rates).max(), 6e-4)
        # sampled data: realized errors are consistent with the asymptotic error bars, and GLS beats OLS
        freqs = lgst.resample_frequencies(obs, shots, seed=5)
        nominal = lgst.nominal_rates_by_type(self.params, {'S': 1e-3})
        rates_s, eb_s = lgst.estimate_error_rates_gls(self.moments, [freqs, None], self.params, shots,
                                                      nominal_rates=nominal)
        z = np.abs(rates_s - self.true_rates) / eb_s
        self.assertLess(np.median(z), 2.0)
        self.assertLess(np.abs(rates_s - self.true_rates).mean(), 3 * eb_s.mean())
        rates_ols, _ = lgst.estimate_error_rates(self.designs, [freqs, None], None, self.params)
        err_gls = np.sqrt(np.mean((rates_s - self.true_rates) ** 2))
        err_ols = np.sqrt(np.mean((rates_ols - self.true_rates) ** 2))
        self.assertLess(err_gls, 1.1 * err_ols)
        # unequal shots, including circuits without any
        shots_u = shots.copy()
        shots_u[::3] = 0.0
        shots_u[1::3] *= 2.5
        freqs_u = lgst.resample_frequencies(obs, shots_u, seed=6)
        self.assertEqual(freqs_u[0], {})
        rates_u, eb_u = lgst.estimate_error_rates_gls(self.moments, [freqs_u, None], self.params, shots_u)
        self.assertLess(np.abs(rates_u - self.true_rates).max(), 10 * eb_u.max())
        # bootstrap error bars agree with the asymptotic ones to within a factor of ~2 on average
        rates_b, eb_b = lgst.estimate_error_rates_gls(self.moments, [freqs, None], self.params, shots,
                                                      nominal_rates=nominal, error_bars='bootstrap',
                                                      bootstrap_trials=15, seed=2)
        self.assertArraysAlmostEqual(rates_b, rates_s)
        self.assertLess(0.5, (eb_b / eb_s).mean())
        self.assertLess((eb_b / eb_s).mean(), 2.0)
        with self.assertRaises(ValueError):
            lgst.estimate_error_rates_gls(self.moments, [freqs, None], self.params, shots, error_bars='magic')
        with self.assertRaises(ValueError):
            lgst.estimate_error_rates_gls(self.moments, [freqs, None], self.params, np.zeros(len(shots)))

    def test_prediction_weight_and_targeted_design(self):
        app = self.designs[:10]
        W0 = lgst.prediction_weight(app, ridge=0.0)
        P = len(self.params)
        self.assertEqual(W0.shape, (P, P))
        self.assertGreaterEqual(np.linalg.eigvalsh(W0)[0], -1e-12)
        D = np.vstack([d.design_matrix for d in app])
        self.assertArraysAlmostEqual(W0, D.T @ D / D.shape[0])
        W = lgst.prediction_weight(app)   # default: a small relative ridge keeps every parameter weakly wanted
        self.assertArraysAlmostEqual(W, W0 + 1e-3 * np.mean(np.diag(W0)) * np.eye(P))
        self.assertGreater(np.linalg.eigvalsh(W)[0], 0)
        info = lgst.circuit_information(self.moments, self.true_rates)
        shots_w, res_w = lgst.optimal_allocation(info, self.budget, criterion='A', weight=W)
        shots_i, res_i = lgst.optimal_allocation(info, self.budget, criterion='A')
        self.assertTrue(np.all(np.isfinite(lgst.predicted_error_bars_nominal(info, shots_w))))
        # the targeted design predicts the application at least as well as the general-purpose one
        self.assertLessEqual(res_w.objective, lgst.design_criterion(info, shots_i, 'A', weight=W) * (1 + 1e-6))
        # and its objective is the mean squared prediction error over the application's observables (+ ridge)
        cov = lgst.predicted_covariance(info, shots_w)
        self.assertAlmostEqual(res_w.objective, np.mean(np.einsum('ri,ij,rj->r', D, cov, D)) + np.trace(cov) * (W[0, 0] - W0[0, 0]),
                               delta=1e-6 * res_w.objective)

    def test_full_hsca_mcm_spam_design_and_estimation(self):
        """The whole pipeline on the most general ansatz: H+S+C+A gate errors, SPAM, all 28 FOMGI MCM quantities."""
        ansatz = {k: dict(v) for k, v in self.ansatz.items()}
        ansatz[('Gcphase', 0, 1)].update({('S', 'ZI'): 0.001, ('S', 'IZ'): 0.001, ('C', 'ZI', 'IZ'): 0.0008})
        ansatz[('Gxpi2', 0)].update({('S', 'Y'): 0.001, ('A', 'X', 'Y'): -0.0008})
        ansatz[('Iz', 0)] = mg.fomgi_ansatz(sectors='all', rates={'s_read': 0.01, 's_prep': 0.003, 'a_prep': -0.001,
                                                                   'r_meas_y': 0.005, 'wt0': 0.001})
        params, rates = lgst.build_model_parameter_indexing(ansatz, 2)
        true_rates = np.array(rates)
        types = np.array(params.errorgen_types())
        self.assertTrue((types == 'C').any() and (types == 'A').any())
        circuits = _random_circuits(self.rng, 160, 5, mcm_qubit=0, mcm_every=2)
        moments = lgst.create_design_moments(circuits, params)
        self.assertEqual(lgst.unidentifiable_directions([m.design for m in moments], params), [])
        info = lgst.circuit_information(moments, lgst.nominal_rates_by_type(params, {'S': 1e-3}))
        budget = 4.0e6
        shots, res = lgst.optimal_allocation(info, budget, criterion='A')
        self.assertLessEqual(res.gap, 1e-3)
        self.assertLess(res.objective, lgst.design_criterion(info, budget / len(info), 'A'))
        used = np.nonzero(shots > 0)[0]
        self.assertTrue(any(moments[i].design.num_mcms > 0 for i in used))
        obs, _ = _simulate(self.pspec, ansatz, [circuits[i] for i in used], params)
        est, eb = lgst.estimate_error_rates_gls([moments[i] for i in used], [obs, None], params, shots[used])
        for typ in 'HSCA':
            self.assertLess(np.abs(est - true_rates)[types == typ].max(), 8e-4, msg="sector %s" % typ)
        self.assertTrue(np.all(est[types == 'S'] >= 0))
        self.assertTrue(np.all(np.isfinite(eb)))
        fomgi = lgst.mcm_fomgi_estimates(params, est)[('Iz', 0)]
        self.assertAlmostEqual(fomgi['s_read'], 0.01, delta=8e-4)
        self.assertAlmostEqual(fomgi['a_prep'], -0.001, delta=8e-4)


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

    @pytest.mark.skipif(not _HAS_CVXPY, reason="cvxpy not installed")
    def test_shot_allocation_with_full_hsca_mcm_and_spam(self):
        """`neyman_shot_allocation`/`optimal_shot_allocation` run on the most general supported ansatz: H+S+C+A
        gate errors, SPAM, and an MCM with all 28 FOMGI quantities -- some circuits have two MCMs, whose
        design matrices generally have exactly-correlated (rank-deficient) observable pairs."""
        ansatz = {k: dict(v) for k, v in self.ansatz.items()}
        ansatz[('Gcphase', 0, 1)][('C', 'ZI', 'IZ')] = 0.0005
        ansatz[('Gxpi2', 0)][('A', 'X', 'Y')] = 0.0008
        ansatz['prep'] = {('S', 'X:0'): 0.002, ('S', 'X:1'): 0.001}
        ansatz['povm'] = {('S', 'X:0'): 0.003, ('S', 'X:1'): 0.0015}
        ansatz[('Iz', 0)] = mg.fomgi_ansatz(sectors='all', rates=self.fomgi_truth)
        params, _ = lgst.build_model_parameter_indexing(ansatz, 2)
        types = np.array(params.errorgen_types())
        self.assertTrue((types == 'C').any() and (types == 'A').any())

        # same recipe as `test_full_hsca_error_model`, which establishes that it is fully identifiable
        circuits = _random_circuits(self.rng, 200, 5, mcm_qubit=0, mcm_every=2)
        designs = lgst.create_design_matrix_list(circuits, params, return_info=True)
        self.assertTrue(any(d.num_mcms >= 1 for d in designs))
        self.assertEqual(lgst.unidentifiable_directions(designs, params), [])
        budget = len(designs) * 2000.0

        equal_obj = np.sum(lgst.predicted_error_bars(designs, shots=budget / len(designs)) ** 2)
        neyman_shots, neyman_obj = lgst.neyman_shot_allocation(designs, budget)
        self.assertAlmostEqual(neyman_shots.sum(), budget, delta=1e-6 * budget)
        self.assertLessEqual(neyman_obj, equal_obj)

        opt_shots, opt_obj = lgst.optimal_shot_allocation(designs, budget, criterion='A')
        self.assertAlmostEqual(opt_shots.sum(), budget, delta=1e-3 * budget)
        self.assertTrue(np.all(opt_shots >= -1e-6))

        for criterion in ('D', 'E'):  # just check these run to completion on the full ansatz without error
            shots_c, obj_c = lgst.optimal_shot_allocation(designs, budget, criterion=criterion)
            self.assertAlmostEqual(shots_c.sum(), budget, delta=1e-3 * budget)
            self.assertTrue(np.isfinite(obj_c))

        # a weight restricted to the MCM's FOMGI parameters concentrates the budget on MCM-sensitive circuits
        mcm_cols = params.indices_for_key(('Iz', 0))
        weight = np.zeros((len(params), len(params)))
        for j in mcm_cols:
            weight[j, j] = 1.0
        mcm_shots, _ = lgst.optimal_shot_allocation(designs, budget, criterion='A', weight=weight)
        has_mcm = np.array([d.num_mcms > 0 for d in designs])
        self.assertGreater(mcm_shots[has_mcm].sum(), 0.9 * budget)  # most of the budget goes to MCM circuits
        self.assertGreater(mcm_shots[has_mcm].sum(), opt_shots[has_mcm].sum())  # more than the unweighted design

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

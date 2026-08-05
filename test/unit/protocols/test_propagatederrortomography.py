"""
Tests for pygsti.protocols.propagatederrortomography: the error-generator-rate coordinate
enumeration and the first-order design matrix built from it.
"""
import numpy as np
from numpy.testing import assert_array_equal

from pygsti.algorithms.randomcircuit import create_random_circuit
from pygsti.errorgenpropagation.errorpropagator import ErrorGeneratorPropagator
from pygsti.models import create_cloud_crosstalk_model
from pygsti.processors import QubitProcessorSpec
from pygsti.protocols.propagatederrortomography import (
    error_generator_rates,
    _circuit_design_block,
    _design_matrix,
    _hamiltonian_column_mask,
    _split_rows,
    _z_type_observables,
)
from pygsti.tools import errgenpolytools as _errgenpolytools
from pygsti.tools import errgenproptools as _errgenproptools

from ..util import BaseCase

_GATE_NAMES = ('Gxpi2', 'Gypi2', 'Gzpi2', 'Gcphase')
_AXIS_OF_GATE = {'Gxpi2': 'X', 'Gypi2': 'Y', 'Gzpi2': 'Z'}


def _paper_shaped_ansatz(num_qubits, seed=0, scale=1.0, with_spam=True):
    """
    A cloud-crosstalk model shaped like the paper's Sec. IV A ansatz: per-gate local H/S on the
    gate's target, a coherent Z on every spectator qubit, coherent ZZ on connected pairs for the
    two-qubit gate, and (if `with_spam`) stochastic X errors on `prep` and `povm`.
    """
    ring = [(i, (i + 1) % num_qubits) for i in range(num_qubits)] if num_qubits > 2 else [(0, 1)]
    pspec = QubitProcessorSpec(num_qubits, list(_GATE_NAMES), availability={'Gcphase': ring},
                               qubit_labels=list(range(num_qubits)))
    rng = np.random.default_rng(seed)
    h1, s1 = 1e-2 * scale, 1e-3 * scale
    h2, s2 = 2e-3 * scale, (1. / 6) * 1e-2 * scale
    coeffs = {}
    for gate_name in _GATE_NAMES:
        if gate_name == 'Gcphase':
            continue
        axis = _AXIS_OF_GATE[gate_name]
        for q in range(num_qubits):
            entry = {('H', f'{axis}:{q}'): rng.uniform(-h1, h1),
                     ('S', f'{axis}:{q}'): rng.uniform(0, s1)}
            for spectator in range(num_qubits):
                if spectator != q:
                    entry[('H', f'Z:{spectator}')] = rng.uniform(-h1, h1)
            coeffs[(gate_name, q)] = entry
    for (a, b) in ring:
        entry = {('H', f'ZZ:{a},{b}'): rng.uniform(-h2, h2),
                 ('S', f'ZZ:{a},{b}'): rng.uniform(0, s2)}
        for q in range(num_qubits):
            entry[('H', f'Z:{q}')] = rng.uniform(-h2, h2)
        coeffs[('Gcphase', a, b)] = entry
    if with_spam:
        coeffs['prep'] = {('S', f'X:{q}'): rng.uniform(0, 1e-3 * scale) for q in range(num_qubits)}
        coeffs['povm'] = {('S', f'X:{q}'): rng.uniform(0, 1e-3 * scale) for q in range(num_qubits)}
    model = create_cloud_crosstalk_model(pspec, lindblad_error_coeffs=coeffs,
                                         lindblad_parameterization='GLND', errcomp_type='errorgens',
                                         independent_gates=True)
    return pspec, model


def _sample_circuits(pspec, depth, num_circuits, seed=0):
    return [create_random_circuit(pspec, depth, sampler='edgegrab', samplerargs=[0.5],
                                  rand_state=seed + i)
            for i in range(num_circuits)]


def _order1_reference(propagator, circuit, observables):
    """pyGSTi's existing BCH order-1 forward model for `<Q>`, via BCH-propagated numeric rates."""
    propagated = propagator.propagate_errorgens_bch(circuit, bch_order=1)
    tableau = circuit.convert_to_stim_tableau()
    return np.array([_errgenproptools.approximate_stabilizer_pauli_expectation(
        propagated, tableau, obs, order=1) for obs in observables])


def _exact_expectations(model, circuit, observables):
    """`<Q>` for each observable, computed from the model's exact outcome-probability distribution."""
    probs = model.probabilities(circuit)
    out = np.zeros(len(observables))
    for i, pauli in enumerate(observables):
        support = [j for j, p in enumerate(pauli) if p != 0]
        sign = 1.0 if pauli.sign == 1 else -1.0
        total = 0.0
        for bits, prob in probs.items():
            bits = bits[0] if isinstance(bits, tuple) else bits
            parity = sum(int(bits[j]) for j in support) % 2
            total += prob * (1 - 2 * parity)
        out[i] = sign * total
    return out


def _multi_contributor_variable_count(model, propagator, circuit):
    """Number of polynomial variables attributed to more than one gate contributor."""
    tmap = propagator.errorgen_transform_map(circuit, include_spam=True)
    _, var_to_errorgen = _errgenpolytools.error_generator_to_polynomial_variable_maps(
        tmap, return_reverse=True)
    count = 0
    for errorgen, layer_idx in var_to_errorgen.values():
        contributors = _errgenpolytools.errorgen_gate_contributors(
            model, errorgen, circuit, layer_idx, include_spam=True)
        if len(contributors) > 1:
            count += 1
    return count


class ErrorGeneratorRatesTester(BaseCase):
    def test_rate_coordinates_round_trip(self):
        """`error_generator_rates` keys are unique, and its values round-trip through each
        operator's `set_errorgen_coefficients`."""
        _, model = _paper_shaped_ansatz(2, with_spam=True)
        rates = error_generator_rates(model)

        labels = (list(model.primitive_prep_labels) + list(model.primitive_op_labels)
                  + list(model.primitive_povm_labels))
        num_pairs_generated = 0
        for lbl in labels:
            op = model.circuit_layer_operator(lbl)
            if hasattr(op, 'errorgen_coefficients'):
                num_pairs_generated += len(op.errorgen_coefficients(label_type='local'))
        self.assertEqual(len(rates), num_pairs_generated)

        by_op = {}
        for (lbl, eglbl), rate in rates.items():
            by_op.setdefault(lbl, {})[eglbl] = rate
        for lbl, local_rates in by_op.items():
            op = model.circuit_layer_operator(lbl)
            op.set_errorgen_coefficients(local_rates)
            readback = op.errorgen_coefficients(label_type='local')
            for eglbl, rate in local_rates.items():
                self.assertLess(abs(readback[eglbl] - rate), 1e-15)


class DesignMatrixTester(BaseCase):
    def test_agrees_with_order1_forward_model(self):
        """`D @ rates + ideal` reproduces pyGSTi's existing BCH order-1 forward model, at 2 and 3
        qubits with SPAM. This validates the whole chain, including multi-contributor attribution."""
        for num_qubits in (2, 3):
            pspec, model = _paper_shaped_ansatz(num_qubits, seed=1, with_spam=True)
            rates = error_generator_rates(model)
            rate_labels = list(rates)
            observables = _z_type_observables(list(range(num_qubits)), max_weight=2)
            circuits = _sample_circuits(pspec, depth=4, num_circuits=3, seed=10)
            D, ideal = _design_matrix(model, circuits, observables, rate_labels)

            propagator = ErrorGeneratorPropagator(model.copy())
            reference = np.concatenate([_order1_reference(propagator, c, observables)
                                        for c in circuits])
            predicted = ideal + D @ np.array(list(rates.values()))
            self.assertLess(np.max(np.abs(predicted - reference)), 1e-12)

    def test_first_order_convergence(self):
        """Shrinking every rate by 10x shrinks the residual against dense simulation by a factor
        between 50 and 200 -- the signature of a correct first-order model."""
        num_qubits = 2
        observables = _z_type_observables(list(range(num_qubits)), max_weight=2)
        residuals = {}
        for scale in (1.0, 0.1):
            pspec, model = _paper_shaped_ansatz(num_qubits, seed=2, scale=scale, with_spam=True)
            rates = error_generator_rates(model)
            rate_labels = list(rates)
            circuits = _sample_circuits(pspec, depth=4, num_circuits=3, seed=20)
            D, ideal = _design_matrix(model, circuits, observables, rate_labels)
            predicted = ideal + D @ np.array(list(rates.values()))
            exact = np.concatenate([_exact_expectations(model, c, observables) for c in circuits])
            residuals[scale] = np.max(np.abs(predicted - exact))
        ratio = residuals[1.0] / residuals[0.1]
        self.assertGreater(ratio, 50)
        self.assertLess(ratio, 200)

    def test_row_classification(self):
        """Every ideal expectation is exactly 0, +1 or -1; H-rows are zero on S-columns and
        S-rows are zero on H-columns."""
        num_qubits = 3
        pspec, model = _paper_shaped_ansatz(num_qubits, seed=3, with_spam=True)
        rates = error_generator_rates(model)
        rate_labels = list(rates)
        observables = _z_type_observables(list(range(num_qubits)), max_weight=2)
        circuits = _sample_circuits(pspec, depth=4, num_circuits=3, seed=30)
        D, ideal = _design_matrix(model, circuits, observables, rate_labels)

        h_rows, s_rows = _split_rows(ideal)
        is_h_col = _hamiltonian_column_mask(rate_labels)
        assert_array_equal(D[np.ix_(h_rows, ~is_h_col)], 0)
        assert_array_equal(D[np.ix_(s_rows, is_h_col)], 0)

    def test_spam_columns_present_iff_include_spam(self):
        """SPAM rate coordinates (`rho0`, `Mdefault`) appear with nonzero design-matrix columns
        for a SPAM-carrying ansatz, and are absent for a SPAM-free ansatz."""
        num_qubits = 2
        observables = _z_type_observables(list(range(num_qubits)), max_weight=2)

        spam_pspec, spam_model = _paper_shaped_ansatz(num_qubits, seed=4, with_spam=True)
        spam_rates = error_generator_rates(spam_model)
        spam_labels = list(spam_rates)
        spam_gate_names = {str(lbl) for lbl, _ in spam_labels}
        self.assertIn('rho0', spam_gate_names)
        self.assertIn('Mdefault', spam_gate_names)
        circuits = _sample_circuits(spam_pspec, depth=4, num_circuits=3, seed=40)
        D, _ = _design_matrix(spam_model, circuits, observables, spam_labels)
        rho0_cols = [i for i, (lbl, _) in enumerate(spam_labels) if str(lbl) == 'rho0']
        povm_cols = [i for i, (lbl, _) in enumerate(spam_labels) if str(lbl) == 'Mdefault']
        self.assertTrue(np.any(D[:, rho0_cols] != 0))
        self.assertTrue(np.any(D[:, povm_cols] != 0))

        _, bare_model = _paper_shaped_ansatz(num_qubits, seed=4, with_spam=False)
        bare_gate_names = {str(lbl) for lbl, _ in error_generator_rates(bare_model)}
        self.assertNotIn('rho0', bare_gate_names)
        self.assertNotIn('Mdefault', bare_gate_names)

    def test_multi_contributor_attribution(self):
        """At least one polynomial variable has more than one gate contributor on a 3-qubit
        ansatz, and forward-model agreement still holds in that regime."""
        num_qubits = 3
        pspec, model = _paper_shaped_ansatz(num_qubits, seed=5, with_spam=True)
        rates = error_generator_rates(model)
        rate_labels = list(rates)
        observables = _z_type_observables(list(range(num_qubits)), max_weight=2)
        circuits = _sample_circuits(pspec, depth=4, num_circuits=3, seed=50)
        propagator = ErrorGeneratorPropagator(model.copy())

        multi_contributor_total = sum(_multi_contributor_variable_count(model, propagator, c)
                                      for c in circuits)
        self.assertGreater(multi_contributor_total, 0)

        D, ideal = _design_matrix(model, circuits, observables, rate_labels)
        reference = np.concatenate([_order1_reference(propagator, c, observables)
                                    for c in circuits])
        predicted = ideal + D @ np.array(list(rates.values()))
        self.assertLess(np.max(np.abs(predicted - reference)), 1e-12)

    def test_design_matrix_row_ordering(self):
        """`_design_matrix` over `[c0, c1]` equals the vertical stack of `_circuit_design_block`
        applied to each circuit separately (circuit-major row order)."""
        num_qubits = 2
        pspec, model = _paper_shaped_ansatz(num_qubits, seed=6, with_spam=True)
        rates = error_generator_rates(model)
        rate_labels = list(rates)
        rate_index = {label: i for i, label in enumerate(rate_labels)}
        observables = _z_type_observables(list(range(num_qubits)), max_weight=2)
        c0, c1 = _sample_circuits(pspec, depth=4, num_circuits=2, seed=60)

        propagator = ErrorGeneratorPropagator(model.copy())
        block0, ideal0 = _circuit_design_block(model, propagator, c0, observables, rate_index)
        block1, ideal1 = _circuit_design_block(model, propagator, c1, observables, rate_index)

        D, ideal = _design_matrix(model, [c0, c1], observables, rate_labels)
        assert_array_equal(D, np.vstack([block0, block1]))
        assert_array_equal(ideal, np.concatenate([ideal0, ideal1]))

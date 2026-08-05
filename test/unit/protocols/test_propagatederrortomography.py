"""
Tests for pygsti.protocols.propagatederrortomography: the error-generator-rate coordinate
enumeration and the first-order design matrix built from it.
"""
import tempfile

import numpy as np
from numpy.testing import assert_array_equal

from pygsti.algorithms.randomcircuit import create_random_circuit
from pygsti.errorgenpropagation.errorpropagator import ErrorGeneratorPropagator
from pygsti.models import create_cloud_crosstalk_model
from pygsti.processors import QubitProcessorSpec
from pygsti.protocols.propagatederrortomography import (
    error_generator_rates,
    design_matrix_rank_diagnostics,
    sample_full_rank_design,
    PropagatedErrorTomographyDesign,
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


def _paper_shaped_ansatz(num_qubits, seed=0, scale=1.0, with_spam=True, spectators=True):
    """
    A cloud-crosstalk model shaped like the paper's Sec. IV A ansatz: per-gate local H/S on the
    gate's target, coherent ZZ on connected pairs for the two-qubit gate, (if `with_spam`)
    stochastic X errors on `prep` and `povm`, and (if `spectators`) a coherent Z on every
    spectator qubit -- the ingredient responsible for the ansatz's n(n-2)-dimensional gauge
    freedom under every in-built pyGSTi sampler.
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
            if spectators:
                for spectator in range(num_qubits):
                    if spectator != q:
                        entry[('H', f'Z:{spectator}')] = rng.uniform(-h1, h1)
            coeffs[(gate_name, q)] = entry
    for (a, b) in ring:
        entry = {('H', f'ZZ:{a},{b}'): rng.uniform(-h2, h2),
                 ('S', f'ZZ:{a},{b}'): rng.uniform(0, s2)}
        if spectators:
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

    def test_empty_circuit_list_raises(self):
        """`_design_matrix([])` raises `ValueError` naming the problem, not numpy's bare
        concatenate error."""
        pspec, model = _paper_shaped_ansatz(2, with_spam=True)
        rate_labels = list(error_generator_rates(model))
        observables = _z_type_observables(list(range(2)), max_weight=2)
        with self.assertRaisesRegex(ValueError, 'empty circuit list'):
            _design_matrix(model, [], observables, rate_labels)


def _bare_sampler(pspec, qubit_labels=None, rand_state=None):
    """A minimal callable sampler: a fixed one-qubit-gate layer, ignoring `rand_state`."""
    from pygsti.baseobjs.label import Label
    labels = qubit_labels if qubit_labels is not None else pspec.qubit_labels
    return [Label('Gxpi2', q) for q in labels]


class PropagatedErrorTomographyDesignTester(BaseCase):
    def test_serialization_round_trip(self):
        """`write` then `from_dir` reproduces `depth`, `num_circuits`, `sampler`, `samplerargs`,
        `seed`, `descriptor` and the circuit list, for both a string sampler and a callable one
        (which is stored as `'function'`)."""
        pspec, _ = _paper_shaped_ansatz(2, with_spam=True)
        design = PropagatedErrorTomographyDesign(pspec, depth=3, num_circuits=4,
                                                 sampler='edgegrab', samplerargs=[0.5], seed=11,
                                                 descriptor='a design')
        with tempfile.TemporaryDirectory() as tmpdir:
            design.write(tmpdir)
            reloaded = PropagatedErrorTomographyDesign.from_dir(tmpdir)
        for attr in ('depth', 'num_circuits', 'sampler', 'samplerargs', 'seed', 'descriptor'):
            self.assertEqual(getattr(design, attr), getattr(reloaded, attr))
        self.assertEqual(list(design.all_circuits_needing_data),
                         list(reloaded.all_circuits_needing_data))

        callable_design = PropagatedErrorTomographyDesign(pspec, depth=2, num_circuits=2,
                                                          sampler=_bare_sampler, samplerargs=[],
                                                          seed=12)
        self.assertEqual(callable_design.sampler, 'function')
        with tempfile.TemporaryDirectory() as tmpdir:
            callable_design.write(tmpdir)
            reloaded_callable = PropagatedErrorTomographyDesign.from_dir(tmpdir)
        self.assertEqual(reloaded_callable.sampler, 'function')

    def test_from_circuits(self):
        """`from_circuits` produces a design whose circuits are the ones passed, with `depth`,
        `sampler`, `samplerargs` and `seed` all `None` and `num_circuits == len(circuits)`."""
        pspec, _ = _paper_shaped_ansatz(2, with_spam=True)
        circuits = _sample_circuits(pspec, depth=3, num_circuits=5, seed=70)
        design = PropagatedErrorTomographyDesign.from_circuits(circuits, descriptor='external')
        self.assertEqual(list(design.all_circuits_needing_data), circuits)
        self.assertIsNone(design.depth)
        self.assertIsNone(design.sampler)
        self.assertIsNone(design.samplerargs)
        self.assertIsNone(design.seed)
        self.assertEqual(design.num_circuits, len(circuits))
        self.assertEqual(design.descriptor, 'external')

    def test_reproducibility(self):
        """Two designs built with the same seed have identical circuits; different seeds do
        not."""
        pspec, _ = _paper_shaped_ansatz(2, with_spam=True)
        same_a = PropagatedErrorTomographyDesign(pspec, depth=3, num_circuits=5,
                                                 samplerargs=[0.5], seed=21)
        same_b = PropagatedErrorTomographyDesign(pspec, depth=3, num_circuits=5,
                                                 samplerargs=[0.5], seed=21)
        different = PropagatedErrorTomographyDesign(pspec, depth=3, num_circuits=5,
                                                    samplerargs=[0.5], seed=22)
        self.assertEqual(list(same_a.all_circuits_needing_data),
                         list(same_b.all_circuits_needing_data))
        self.assertNotEqual(list(same_a.all_circuits_needing_data),
                            list(different.all_circuits_needing_data))

    def test_bad_seed_raises(self):
        """A `seed` that is neither `None` nor an `int` raises `ValueError`."""
        pspec, _ = _paper_shaped_ansatz(2, with_spam=True)
        with self.assertRaises(ValueError):
            PropagatedErrorTomographyDesign(pspec, depth=3, num_circuits=2, samplerargs=[0.5],
                                            seed=1.5)


class DesignMatrixRankDiagnosticsTester(BaseCase):
    def test_reproduces_phase0_gauge_freedom(self):
        """On the paper-shaped ansatz at n = 4 and n = 5, `design_matrix_rank_diagnostics`
        reports an H-block deficit of exactly n * (n - 2) and a full-rank S block."""
        for num_qubits in (4, 5):
            pspec, model = _paper_shaped_ansatz(num_qubits, seed=1, with_spam=True)
            circuits = _sample_circuits(pspec, depth=6, num_circuits=150, seed=80)
            diagnostics = design_matrix_rank_diagnostics(circuits, model)
            self.assertEqual(diagnostics['hamiltonian']['deficit'], num_qubits * (num_qubits - 2))
            self.assertEqual(diagnostics['stochastic']['deficit'], 0)

    def test_qubit_ordering_guard(self):
        """A circuit whose `line_labels` disagree with the model's qubit ordering raises
        `ValueError`."""
        pspec, model = _paper_shaped_ansatz(2, with_spam=True)
        circuits = _sample_circuits(pspec, depth=3, num_circuits=2, seed=81)
        relabeled = [c.copy() for c in circuits]
        relabeled[0] = relabeled[0].map_state_space_labels({0: 'a', 1: 'b'})
        with self.assertRaises(ValueError):
            design_matrix_rank_diagnostics(relabeled, model)


class SampleFullRankDesignTester(BaseCase):
    def test_default_samplerargs_works_with_edgegrab(self):
        """Omitting `samplerargs` samples successfully and records the `[0.25]` default, rather
        than failing inside `create_random_circuit`, which requires a two-qubit gate density."""
        pspec, _ = _paper_shaped_ansatz(2, with_spam=True)
        design = PropagatedErrorTomographyDesign(pspec, depth=3, num_circuits=2, seed=1)
        self.assertEqual(design.samplerargs, [0.25])
        self.assertEqual(len(design.all_circuits_needing_data), 2)

    def test_provenance_equivalence(self):
        """A design from the constructor and one from `sample_full_rank_design`, given the same
        `depth`, `sampler`, `samplerargs` and `seed`, are equal attribute by attribute except for
        their circuits and `num_circuits`. This is the separation-of-concerns claim: the returned
        design carries no trace of the `model` that guided the sampling."""
        pspec, model = _paper_shaped_ansatz(3, seed=2, with_spam=True, spectators=False)
        sampled, _ = sample_full_rank_design(pspec, model, depth=4, batch_size=10,
                                             sampler='edgegrab', samplerargs=[0.5], seed=123)
        constructed = PropagatedErrorTomographyDesign(pspec, depth=4, num_circuits=7,
                                                      sampler='edgegrab', samplerargs=[0.5],
                                                      seed=123)
        for attr in ('depth', 'sampler', 'samplerargs', 'seed', 'descriptor', 'qubit_labels'):
            self.assertEqual(getattr(sampled, attr), getattr(constructed, attr))

    def test_reaches_full_rank_on_gauge_free_ansatz(self):
        """On a gauge-free ansatz, `sample_full_rank_design` returns a full-rank design,
        `diagnostics['full_rank']` is `True`, `diagnostics['saturated']` is `False`, and
        `rank_history` is non-decreasing in both blocks."""
        pspec, model = _paper_shaped_ansatz(3, seed=2, with_spam=True, spectators=False)
        design, diagnostics = sample_full_rank_design(pspec, model, depth=4, batch_size=10,
                                                      samplerargs=[0.5], seed=123)
        self.assertTrue(diagnostics['full_rank'])
        self.assertFalse(diagnostics['saturated'])
        self.assertEqual(diagnostics['hamiltonian']['deficit'], 0)
        self.assertEqual(diagnostics['stochastic']['deficit'], 0)
        self.assertEqual(len(design.all_circuits_needing_data), design.num_circuits)
        h_ranks = [entry[1] for entry in diagnostics['rank_history']]
        s_ranks = [entry[2] for entry in diagnostics['rank_history']]
        self.assertTrue(all(a <= b for a, b in zip(h_ranks, h_ranks[1:])))
        self.assertTrue(all(a <= b for a, b in zip(s_ranks, s_ranks[1:])))

    def test_saturation_and_max_circuits_errors_are_distinguishable(self):
        """On the paper-shaped ansatz with `max_circuits=None`, `sample_full_rank_design` raises
        `ValueError` attributing a below-full-rank deficit to a gauge freedom. Separately, with a
        small `max_circuits` on a gauge-free ansatz that has not yet saturated, it raises
        `ValueError` with the other message. The two messages are distinguishable."""
        gauge_pspec, gauge_model = _paper_shaped_ansatz(4, seed=3, with_spam=True,
                                                        spectators=True)
        with self.assertRaises(ValueError) as saturation_ctx:
            sample_full_rank_design(gauge_pspec, gauge_model, depth=6, batch_size=25,
                                    samplerargs=[0.5], max_circuits=None, seed=5)
        saturation_message = str(saturation_ctx.exception)
        self.assertIn('gauge freedom', saturation_message)

        free_pspec, free_model = _paper_shaped_ansatz(3, seed=4, with_spam=True,
                                                      spectators=False)
        with self.assertRaises(ValueError) as max_circuits_ctx:
            sample_full_rank_design(free_pspec, free_model, depth=4, batch_size=10,
                                    samplerargs=[0.5], max_circuits=15, seed=7)
        max_circuits_message = str(max_circuits_ctx.exception)
        self.assertIn('max_circuits', max_circuits_message)
        self.assertNotIn('gauge freedom', max_circuits_message)
        self.assertNotEqual(saturation_message, max_circuits_message)

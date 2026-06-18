import numpy as np
import stim

from itertools import product

from pygsti.algorithms.randomcircuit import create_random_circuit
from pygsti.baseobjs.polynomial import Polynomial
from pygsti.errorgenpropagation.errorpropagator import ErrorGeneratorPropagator
from pygsti.errorgenpropagation.localstimerrorgen import LocalStimErrorgenLabel as _LSE
from pygsti.models.modelconstruction import create_crosstalk_free_model
from pygsti.processors import QubitProcessorSpec
from pygsti.tools import errgenproptools as _eprop
from pygsti.tools import errgenpolytools as _epoly
from pygsti.tools.lindbladtools import random_CPTP_error_generator_rates

from ..util import BaseCase
from .test_errgenproptools import sample_error_rates_dict


class ErrgenPolyToolsTester(BaseCase):

    def setUp(self):
        num_qubits = 4
        gate_names = ['Gcphase', 'Gxpi2', 'Gypi2']
        availability = {'Gcphase': [(0, 1), (1, 2), (2, 3), (3, 0)]}
        self.pspec = QubitProcessorSpec(num_qubits, gate_names, availability=availability)

        max_strengths = {
            1: {'S': 0.0005, 'H': 0.0001},
            2: {'S': 0.0015, 'H': 0.0003}
        }
        error_rates_dict = sample_error_rates_dict(self.pspec, max_strengths, seed=12345)
        self.error_model = create_crosstalk_free_model(self.pspec, lindblad_error_coeffs=error_rates_dict)
        self.errorgen_propagator = ErrorGeneratorPropagator(self.error_model.copy())

        error_rates_dict_shared = {
            'Gcphase': random_CPTP_error_generator_rates(2, errorgen_types=('H', 'S'), label_type='local', seed=1234),
            'Gxpi2': random_CPTP_error_generator_rates(1, errorgen_types=('H', 'S'), label_type='local', seed=1235),
            'Gypi2': random_CPTP_error_generator_rates(1, errorgen_types=('H', 'S'), label_type='local', seed=1236)
        }
        self.error_model_shared = create_crosstalk_free_model(self.pspec, lindblad_error_coeffs=error_rates_dict_shared)
        self.errorgen_propagator_shared = ErrorGeneratorPropagator(self.error_model_shared.copy())

        self.circuits_4Q = [
            create_random_circuit(self.pspec, 4, sampler='edgegrab', samplerargs=[0.4], rand_state=1235 + i)
            for i in range(3)
        ]
        self.test_ckt = self.circuits_4Q[0]
        self.test_tableau = self.test_ckt.convert_to_stim_tableau()

        self.errorgen_phases = self.errorgen_propagator.errorgen_transform_map(self.test_ckt)
        self.errorgen_phases_no_spam = self.errorgen_propagator.errorgen_transform_map(self.test_ckt, include_spam=False)        
        self.errorgen_phases_by_layer = self.errorgen_propagator.errorgen_transform_maps(self.test_ckt)
        self.propagated_errorgens_bch1 = self.errorgen_propagator.propagate_errorgens_bch(self.test_ckt, bch_order=1)
        self.propagated_errorgens_bch2 = self.errorgen_propagator.propagate_errorgens_bch(self.test_ckt, bch_order=2)
        self.propagated_errorgens_bch1_shared = self.errorgen_propagator_shared.propagate_errorgens_bch(self.test_ckt, bch_order=1)

        self.errorgen_phases_shared = self.errorgen_propagator_shared.errorgen_transform_map(self.test_ckt)
        self.errorgen_phases_shared_no_spam = self.errorgen_propagator_shared.errorgen_transform_map(self.test_ckt, include_spam=False)
        self.errorgen_phases_by_layer_shared = self.errorgen_propagator_shared.errorgen_transform_maps(self.test_ckt)

        self.errgen_to_var_map, self.var_to_errgen_map = _epoly.error_generator_to_polynomial_variable_maps(
            self.errorgen_phases, return_reverse=True
        )

        self.errgen_to_var_map_no_spam, self.var_to_errgen_map_no_spam = _epoly.error_generator_to_polynomial_variable_maps(
            self.errorgen_phases_no_spam, return_reverse=True, 
        )

        self.errgen_to_var_map_shared, self.var_to_errgen_map_shared = _epoly.error_generator_to_polynomial_variable_maps(
            self.errorgen_phases_shared, return_reverse=True
        )

        self.errgen_to_var_map_shared_no_spam, self.var_to_errgen_map_shared_no_spam = _epoly.error_generator_to_polynomial_variable_maps(
            self.errorgen_phases_shared_no_spam, return_reverse=True
        )

        self.errgen_to_var_map_gate_noshared, self.var_to_errgen_map_gate_noshared = \
            _epoly.error_generator_to_polynomial_variable_maps_by_gate(
                self.error_model, self.errgen_to_var_map, self.test_ckt,
                include_spam=True, aggregate_shared_parameter_gates=False, return_reverse=True
            )

        self.errgen_to_var_map_gate_shared_noagg, self.var_to_errgen_map_gate_shared_noagg = \
            _epoly.error_generator_to_polynomial_variable_maps_by_gate(
                self.error_model_shared, self.errgen_to_var_map_shared, self.test_ckt,
                include_spam=True, aggregate_shared_parameter_gates=False, return_reverse=True
            )

        self.errgen_to_var_map_gate_shared_agg, self.var_to_errgen_map_gate_shared_agg = \
            _epoly.error_generator_to_polynomial_variable_maps_by_gate(
                self.error_model_shared, self.errgen_to_var_map_shared, self.test_ckt,
                include_spam=True, aggregate_shared_parameter_gates=True, return_reverse=True
            )

        self.paramvec_unaggregated = _epoly.construct_polynomial_parameter_vector_from_propagator(
            self.errorgen_propagator, self.var_to_errgen_map, self.test_ckt
        )
        self.paramvec_shared_unaggregated = _epoly.construct_polynomial_parameter_vector_from_propagator(
            self.errorgen_propagator_shared, self.var_to_errgen_map_shared, self.test_ckt
        )
        self.paramvec_shared_aggregated = _epoly.construct_polynomial_parameter_vector_from_propagator(
            self.errorgen_propagator_shared, self.var_to_errgen_map_gate_shared_agg, self.test_ckt
        )

        self.first_order_magnus_polys = _epoly.magnus_symbolic_polynomial(
            self.errorgen_phases_by_layer, self.errgen_to_var_map, magnus_order=1
        )
        self.second_order_magnus_polys = _epoly.magnus_symbolic_polynomial(
            self.errorgen_phases_by_layer, self.errgen_to_var_map, magnus_order=2
        )

        self.first_order_magnus_polys_shared_agg = _epoly.magnus_symbolic_polynomial(
            self.errorgen_phases_by_layer_shared, self.errgen_to_var_map_gate_shared_agg, magnus_order=1
        )

        self.four_qubit_bitstrings = np.array([''.join(bs) for bs in product(['0', '1'], repeat=4)], dtype=object)
        self.four_qubit_paulis = np.fromiter(stim.PauliString.iter_all(num_qubits=4, min_weight=1), dtype=object)
        rng = np.random.default_rng(1234)
        self.random_paulis = rng.choice(self.four_qubit_paulis, 2, replace=False).tolist()
        self.random_bitstrings = rng.choice(self.four_qubit_bitstrings, 2, replace=False).tolist()

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _indexed_errorgen_layers(self, propagator, circuit, include_spam=True):
        errorgen_layers = propagator.construct_errorgen_layers(circuit, len(circuit.line_labels), include_spam=include_spam)
        indexed = {}
        for i, layer in enumerate(errorgen_layers):
            for errorgen, rate in layer.items():
                indexed[(errorgen, i)] = rate
        return indexed

    def _assert_poly_dict_matches_numeric_dict(self, poly_dict, numeric_dict, paramvec, places=12):
        for key, poly in poly_dict.items():
            self.assertAlmostEqual(poly.evaluate(paramvec), numeric_dict[key], places=places)

    # ------------------------------------------------------------------
    # helper / mapping tests
    # ------------------------------------------------------------------

    def test_truncate_lse_support_without_locality_validation(self):
        err = _LSE('H', (stim.PauliString('IXYZ'),))
        self.assertEqual(
            _epoly._truncate_lse_support(err, [0, 1], validate_locality=False),
            _LSE('H', (stim.PauliString('IX'),))
        )
        self.assertEqual(
            _epoly._truncate_lse_support(err, [0, 3], validate_locality=False),
            _LSE('H', (stim.PauliString('IZ'),))
        )

    def test_truncate_lse_support_with_locality_validation_raises(self):
        err = _LSE('H', (stim.PauliString('IXYZ'),))
        with self.assertRaises(RuntimeError):
            _epoly._truncate_lse_support(err, [0, 3], validate_locality=True)

    def test_error_generator_to_polynomial_variable_maps_roundtrip(self):
        forward, reverse = _epoly.error_generator_to_polynomial_variable_maps(self.errorgen_phases, return_reverse=True)
        for k, v in forward.items():
            self.assertEqual(reverse[v], k)

    def test_construct_polynomial_parameter_vector_from_propagator_unaggregated(self):
        indexed_errorgen_layers = self._indexed_errorgen_layers(self.errorgen_propagator_shared, self.test_ckt)
        expected = np.zeros(len(self.var_to_errgen_map_shared))
        for lbl, rate in indexed_errorgen_layers.items():
            expected[self.errgen_to_var_map_shared[lbl]] = rate

        actual = _epoly.construct_polynomial_parameter_vector_from_propagator(
            self.errorgen_propagator_shared, self.var_to_errgen_map_shared, self.test_ckt
        )
        self.assertAlmostEqual(np.linalg.norm(expected - actual), 0.0)

    def test_construct_polynomial_parameter_vector_from_propagator_aggregated(self):
        actual = _epoly.construct_polynomial_parameter_vector_from_propagator(
            self.errorgen_propagator_shared, self.var_to_errgen_map_gate_shared_agg, self.test_ckt
        )
        self.assertEqual(len(actual), len(self.var_to_errgen_map_gate_shared_agg))

    def test_construct_polynomial_parameter_vector_aggregated_classes_have_equal_values(self):
        indexed_errorgen_layers = self._indexed_errorgen_layers(self.errorgen_propagator_shared, self.test_ckt)
        for _, errgen_pairs in self.var_to_errgen_map_gate_shared_agg.items():
            vals = [indexed_errorgen_layers[pair] for pair in errgen_pairs]
            for v in vals[1:]:
                self.assertAlmostEqual(v, vals[0])

    # ------------------------------------------------------------------
    # gate aggregation tests
    # ------------------------------------------------------------------

    def test_error_generator_to_polynomial_variable_maps_by_gate_shared_params_no_cross_gate_aggregation(self):
        mapped_nonshared = _epoly.error_generator_to_polynomial_variable_maps_by_gate(
            self.error_model, self.errgen_to_var_map, self.test_ckt,
            include_spam=True, aggregate_shared_parameter_gates=False
        )
        mapped_shared = _epoly.error_generator_to_polynomial_variable_maps_by_gate(
            self.error_model_shared, self.errgen_to_var_map, self.test_ckt,
            include_spam=True, aggregate_shared_parameter_gates=False
        )
        self.assertEqual(mapped_nonshared, mapped_shared)

    def test_error_generator_to_polynomial_variable_maps_by_gate_shared_params_with_cross_gate_aggregation(self):
        nvars_noagg = len(set(self.errgen_to_var_map_gate_shared_noagg.values()))
        nvars_agg = len(set(self.errgen_to_var_map_gate_shared_agg.values()))
        self.assertTrue(nvars_agg < nvars_noagg)

    def test_error_generator_to_polynomial_variable_maps_by_gate_reverse_map_consistency(self):
        for errgen_pair, var_idx in self.errgen_to_var_map_gate_shared_agg.items():
            self.assertIn(errgen_pair, self.var_to_errgen_map_gate_shared_agg[var_idx])

    def test_construct_polynomial_parameter_vector_include_spam_false(self):
        _, reverse = _epoly.error_generator_to_polynomial_variable_maps_by_gate(
            self.error_model, self.errgen_to_var_map_no_spam, self.test_ckt,
            include_spam=False, return_reverse=True
        )
        actual = _epoly.construct_polynomial_parameter_vector_from_propagator(
            self.errorgen_propagator, reverse, self.test_ckt, include_spam=False
        )
        self.assertEqual(len(actual), len(reverse))

    # ------------------------------------------------------------------
    # validation tests
    # ------------------------------------------------------------------

    def test_error_generator_to_polynomial_variable_maps_by_gate_invalid_model_type_raises(self):
        with self.assertRaises(ValueError):
            _epoly.error_generator_to_polynomial_variable_maps_by_gate(
                model="not_a_model",
                errorgen_var_map=self.errgen_to_var_map,
                circuit=self.test_ckt
            )

    def test_errorgen_gate_contributors_return_operators_consistency(self):
        some_key = next(iter(self.errgen_to_var_map))
        errorgen, layer_idx = some_key
        gates, ops = _epoly.errorgen_gate_contributors(
            self.error_model, errorgen, self.test_ckt, layer_idx,
            include_spam=True, return_operators=True
        )
        self.assertEqual(len(gates), len(ops))


    # ------------------------------------------------------------------
    # magnus polynomial correctness
    # ------------------------------------------------------------------

    def test_magnus_symbolic_polynomial_first_order_matches_bch(self):
        self._assert_poly_dict_matches_numeric_dict(
            self.first_order_magnus_polys,
            self.propagated_errorgens_bch1,
            self.paramvec_unaggregated,
            places=12
        )

    def test_magnus_symbolic_polynomial_second_order_matches_bch(self):
        self._assert_poly_dict_matches_numeric_dict(
            self.second_order_magnus_polys,
            self.propagated_errorgens_bch2,
            self.paramvec_unaggregated,
            places=12
        )

    def test_magnus_symbolic_polynomial_first_order_shared_aggregated_matches_bch(self):
        self._assert_poly_dict_matches_numeric_dict(
            self.first_order_magnus_polys_shared_agg,
            self.propagated_errorgens_bch1_shared,
            self.paramvec_shared_aggregated,
            places=12
        )

    # ------------------------------------------------------------------
    # taylor polynomial correctness
    # ------------------------------------------------------------------

    def test_taylor_expansion_symbolic_polynomial_first_order_matches_numeric(self):
        taylor_polys = _epoly.error_generator_taylor_expansion_symbolic_polynomial(
            self.first_order_magnus_polys, self.errgen_to_var_map, order=1
        )[0]
        taylor_numeric = _eprop.error_generator_taylor_expansion(self.propagated_errorgens_bch1, order=1)[0]
        self._assert_poly_dict_matches_numeric_dict(taylor_polys, taylor_numeric, self.paramvec_unaggregated, places=12)

    def test_taylor_expansion_symbolic_polynomial_second_order_matches_numeric(self):
        taylor_polys = _epoly.error_generator_taylor_expansion_symbolic_polynomial(
            self.first_order_magnus_polys, self.errgen_to_var_map, order=2
        )[1]
        taylor_numeric = _eprop.error_generator_taylor_expansion(
            self.propagated_errorgens_bch1, order=2, truncation_threshold=-1
        )[1]
        self._assert_poly_dict_matches_numeric_dict(taylor_polys, taylor_numeric, self.paramvec_unaggregated, places=12)

    def test_combined_taylor_expansion_polynomial_single_order_fast_path(self):
        combined = _epoly._combined_taylor_expansion_polynomial(
            [self.first_order_magnus_polys], self.errgen_to_var_map
        )
        self.assertEqual(combined, self.first_order_magnus_polys)

    
    # ------------------------------------------------------------------
    # probability correction polynomial correctness
    # ------------------------------------------------------------------

    def test_stabilizer_probability_correction_symbolic_polynomial_first_order_matches_numeric(self):
        poly = _epoly.stabilizer_probability_correction_symbolic_polynomial(
            self.first_order_magnus_polys, self.errgen_to_var_map, self.test_tableau, '0000', order=1
        )
        numeric = _eprop.stabilizer_probability_correction(
            self.propagated_errorgens_bch1, self.test_tableau, '0000', order=1
        )
        self.assertAlmostEqual(poly.evaluate(self.paramvec_unaggregated), numeric, places=12)

    def test_stabilizer_probability_correction_symbolic_polynomial_second_order_matches_numeric(self):
        poly = _epoly.stabilizer_probability_correction_symbolic_polynomial(
            self.first_order_magnus_polys, self.errgen_to_var_map, self.test_tableau, '0000', order=2
        )
        numeric = _eprop.stabilizer_probability_correction(
            self.propagated_errorgens_bch1, self.test_tableau, '0000', order=2
        )
        self.assertAlmostEqual(poly.evaluate(self.paramvec_unaggregated), numeric, places=10)

    def test_stabilizer_probability_correction_symbolic_polynomial_first_order_shared_aggregated_matches_numeric(self):
        poly = _epoly.stabilizer_probability_correction_symbolic_polynomial(
            self.first_order_magnus_polys_shared_agg,
            self.errgen_to_var_map_gate_shared_agg,
            self.test_tableau,
            '0000',
            order=1
        )
        numeric = _eprop.stabilizer_probability_correction(
            self.propagated_errorgens_bch1_shared,
            self.test_tableau,
            '0000',
            order=1
        )
        self.assertAlmostEqual(poly.evaluate(self.paramvec_shared_aggregated), numeric, places=12)

    # ------------------------------------------------------------------
    # pauli expectation correction polynomial correctness
    # ------------------------------------------------------------------

    def test_stabilizer_pauli_expectation_correction_symbolic_polynomial_first_order_matches_numeric(self):
        pauli = stim.PauliString('ZZZZ')
        poly = _epoly.stabilizer_pauli_expectation_correction_symbolic_polynomial(
            self.first_order_magnus_polys, self.errgen_to_var_map, self.test_tableau, pauli, order=1
        )
        numeric = _eprop.stabilizer_pauli_expectation_correction(
            self.propagated_errorgens_bch1, self.test_tableau, pauli, order=1
        )
        self.assertAlmostEqual(poly.evaluate(self.paramvec_unaggregated), numeric, places=12)

    def test_stabilizer_pauli_expectation_correction_symbolic_polynomial_second_order_matches_numeric(self):
        pauli = stim.PauliString('ZZZZ')
        poly = _epoly.stabilizer_pauli_expectation_correction_symbolic_polynomial(
            self.first_order_magnus_polys, self.errgen_to_var_map, self.test_tableau, pauli, order=2
        )
        numeric = _eprop.stabilizer_pauli_expectation_correction(
            self.propagated_errorgens_bch1, self.test_tableau, pauli, order=2
        )
        self.assertAlmostEqual(poly.evaluate(self.paramvec_unaggregated), numeric, places=10)

    def test_stabilizer_pauli_expectation_correction_symbolic_polynomial_first_order_shared_aggregated_matches_numeric(self):
        pauli = stim.PauliString('ZZZZ')
        poly = _epoly.stabilizer_pauli_expectation_correction_symbolic_polynomial(
            self.first_order_magnus_polys_shared_agg,
            self.errgen_to_var_map_gate_shared_agg,
            self.test_tableau,
            pauli,
            order=1
        )
        numeric = _eprop.stabilizer_pauli_expectation_correction(
            self.propagated_errorgens_bch1_shared,
            self.test_tableau,
            pauli,
            order=1
        )
        self.assertAlmostEqual(poly.evaluate(self.paramvec_shared_aggregated), numeric, places=12)

    # ------------------------------------------------------------------
    # bulk probability polynomial tests
    # ------------------------------------------------------------------

    def test_bulk_stabilizer_probability_correction_symbolic_polynomial_matches_scalar_version_order1(self):
        bulk = _epoly.bulk_stabilizer_probability_correction_symbolic_polynomial(
            self.first_order_magnus_polys, self.errgen_to_var_map, self.test_tableau,
            self.random_bitstrings, order=1
        )
        scalar = [
            _epoly.stabilizer_probability_correction_symbolic_polynomial(
                self.first_order_magnus_polys, self.errgen_to_var_map, self.test_tableau, bs, order=1
            )
            for bs in self.random_bitstrings
        ]
        self.assertEqual(bulk, scalar)

    def test_bulk_stabilizer_probability_correction_symbolic_polynomial_matches_scalar_version_order2(self):
        bulk = _epoly.bulk_stabilizer_probability_correction_symbolic_polynomial(
            self.first_order_magnus_polys, self.errgen_to_var_map, self.test_tableau,
            self.random_bitstrings, order=2
        )
        scalar = [
            _epoly.stabilizer_probability_correction_symbolic_polynomial(
                self.first_order_magnus_polys, self.errgen_to_var_map, self.test_tableau, bs, order=2
            )
            for bs in self.random_bitstrings
        ]
        self.assertEqual(bulk, scalar)

    # ------------------------------------------------------------------
    # bulk expectation polynomial tests
    # ------------------------------------------------------------------

    def test_bulk_stabilizer_pauli_expectation_correction_symbolic_polynomial_matches_scalar_version_order1(self):
        bulk = _epoly.bulk_stabilizer_pauli_expectation_correction_symbolic_polynomial(
            self.first_order_magnus_polys, self.errgen_to_var_map, self.test_tableau,
            self.random_paulis, order=1
        )
        scalar = [
            _epoly.stabilizer_pauli_expectation_correction_symbolic_polynomial(
                self.first_order_magnus_polys, self.errgen_to_var_map, self.test_tableau, p, order=1
            )
            for p in self.random_paulis
        ]
        self.assertEqual(bulk, scalar)

    def test_bulk_stabilizer_pauli_expectation_correction_symbolic_polynomial_matches_scalar_version_order2(self):
        bulk = _epoly.bulk_stabilizer_pauli_expectation_correction_symbolic_polynomial(
            self.first_order_magnus_polys, self.errgen_to_var_map, self.test_tableau,
            self.random_paulis, order=2
        )
        scalar = [
            _epoly.stabilizer_pauli_expectation_correction_symbolic_polynomial(
                self.first_order_magnus_polys, self.errgen_to_var_map, self.test_tableau, p, order=2
            )
            for p in self.random_paulis
        ]
        self.assertEqual(bulk, scalar)

    def test_bulk_stabilizer_pauli_expectation_correction_symbolic_polynomial_zero_correction_identity_pauli(self):
        pauli = stim.PauliString('IIII')
        bulk = _epoly.bulk_stabilizer_pauli_expectation_correction_symbolic_polynomial(
            self.first_order_magnus_polys, self.errgen_to_var_map, self.test_tableau,
            [pauli], order=1
        )
        zero_poly = Polynomial({}, max_num_vars=next(iter(self.first_order_magnus_polys.values())).max_num_vars)
        self.assertEqual(bulk[0], zero_poly)

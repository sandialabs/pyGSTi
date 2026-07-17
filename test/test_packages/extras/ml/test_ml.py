import unittest
from typing import Any, cast
import numpy as np
import stim
import tensorflow as tf
import keras

import pygsti
from pygsti.processors.processorspec import QubitProcessorSpec as _ProcessorSpec
from pygsti.circuits import Circuit
from pygsti.data import DataSet
from pygsti.extras.ml import errgentools, encoding, snippers, legacy, customlayers, qpanns

class MLSubpackageTester(unittest.TestCase):
    def test_errgentools(self):
        # test base conversion
        self.assertEqual(errgentools.numberToBase(0, 4), [0])
        self.assertEqual(errgentools.numberToBase(13, 4), [3, 1])
        
        # test paulistring_to_index and index_to_paulistring
        self.assertEqual(errgentools.paulistring_to_index('IX', 2), 1)
        self.assertEqual(errgentools.paulistring_to_index('ZZ', 2), 15)
        self.assertEqual(errgentools.index_to_paulistring(1, 2), 'IX')
        self.assertEqual(errgentools.index_to_paulistring(15, 2), 'ZZ')
        
        # test up_to_weight_k_paulis
        paulis = errgentools.up_to_weight_k_paulis(1, 2)
        self.assertIn('IX', paulis)
        self.assertIn('XI', paulis)
        self.assertNotIn('XX', paulis)
        
        # test error_generator_index and index_to_error_gen
        idx = errgentools.error_generator_index('H', ('IX',))
        self.assertEqual(idx, 1)
        eg = cast(Any, errgentools.index_to_error_gen(1, 2))
        self.assertEqual(eg[0], 'H')
        self.assertEqual(eg[1], ('IX',))
        
        # test up_to_weight_k_error_gens
        egs = errgentools.up_to_weight_k_error_gens(1, 2, ['H', 'S'])
        self.assertTrue(len(egs) > 0)
        self.assertEqual(egs[0][0], 'H')
        
        # test up_to_weight_k_error_gens_from_qubit_graph
        laplacian = np.array([[1, -1], [-1, 1]])
        egs_graph = errgentools.up_to_weight_k_error_gens_from_qubit_graph(1, 2, laplacian, 1)
        self.assertTrue(len(egs_graph) > 0)

    def test_encoding(self):
        nonstd_gate_unitaries = {}
        availability = {}
        pspec = _ProcessorSpec(2, ['{idle}', 'Gx', 'Gy'], nonstd_gate_unitaries, availability, geometry="line")
        
        encoder = encoding.StandardCircuitEncoder(pspec)
        self.assertEqual(encoder.length, len(encoder.gate_indexing))
        
        layer = encoder.layer_encoding(None)
        self.assertEqual(len(layer), encoder.length)
        self.assertTrue(all(val == 0.0 for val in layer))
        
        circ = Circuit(cast(Any, [('Gx', 0)]))
        encoded = encoder(circ)
        self.assertEqual(encoded.shape, (circ.depth + encoder.initialization_encoding_depth() + encoder.measurement_encoding_depth(), encoder.length))
        
        circs = [Circuit(cast(Any, [('Gx', 0)])), Circuit(cast(Any, [('Gy', 1)]))]
        tensor = encoding.circuits_to_tensor(circs, encoder)
        self.assertEqual(tensor.shape[0], 2)
        
        paulis = encoding.make_paulis(2, 1)
        self.assertTrue(len(paulis) > 0)

    def test_snippers(self):
        adj = snippers.undirected_adjacency_matrix_from_edges([(0, 1)], [0, 1])
        np.testing.assert_array_equal(adj, np.array([[0, 1], [1, 0]]))
        
        nonstd_gate_unitaries = {}
        availability = {}
        pspec = _ProcessorSpec(2, ['{idle}', 'Gx', 'Gy'], nonstd_gate_unitaries, availability, geometry="line")
        encoder = encoding.StandardCircuitEncoder(pspec)
        error_generators = [('H', ('IX',)), ('S', ('ZZ',))]
        snipper = snippers.layer_snipper_from_qubit_graph(error_generators, encoder, adj, 1)
        self.assertEqual(len(snipper), len(error_generators))

    def test_legacy(self):
        nonstd_gate_unitaries = {}
        availability = {'Gcnot': [(0, 1)]}
        pspec = _ProcessorSpec(2, ['{idle}', 'Gx', 'Gy', 'Gcnot'], nonstd_gate_unitaries, availability, geometry="line")
        specmodel, error_dict = legacy.create_spec_model(pspec)
        self.assertIsNotNone(specmodel)
        self.assertIsNotNone(error_dict)
        
        circuits = [Circuit(cast(Any, [('Gx', 0)])), Circuit(cast(Any, [('Gy', 1)]))]
        preds = legacy.batch_prediction(specmodel, circuits)
        self.assertEqual(preds.shape, (2,))

    def test_qpanns_and_customlayers(self):
        layer = customlayers.SelectiveDense(units=5, input_indices=[[0, 1], [1, 2]])
        layer.build((None, 3))
        self.assertEqual(len(layer.kernels), 2)
        
        model = qpanns.QPANN(encoding_length=10, modelled_error_generators=[('H', ('IX',))], snipper=[[0, 1]])
        self.assertEqual(model.encoding_length, 10)
        self.assertEqual(model.probability_computation, 'concise')
        config = model.get_config()
        self.assertEqual(config['encoding_length'], 10)

    def test_customdense_forward(self):
        # Regression test for a Keras-3 incompatibility: `CustomDense` used to subclass
        # `keras.layers.Dense`, whose `kernel` attribute is a read-only `@property` under
        # Keras 3 (added to support LoRA), so `build()`'s `self.kernel = self.add_weight(...)`
        # raised `AttributeError: property 'kernel' of 'CustomDense' object has no setter`.
        # `CustomDense` now subclasses `keras.layers.Layer` directly (like `SelectiveDense`
        # elsewhere in this same module), which has no such property. This bug only manifested
        # when the layer was actually called (forward pass), not at construction time, so a
        # test that never calls the layer (as `test_qpanns_and_customlayers` above does not)
        # would not catch it.
        num_errorgens = 3
        layer = customlayers.CustomDense(units=4, num_errorgens=num_errorgens, activation='linear')
        inputs = tf.random.normal((5, num_errorgens, 2))  # (batch, num_errorgens, input_dim)
        outputs = layer(inputs)
        self.assertEqual(tuple(outputs.shape), (5, num_errorgens, 4))
        self.assertTrue(len(layer.trainable_variables) > 0)
        self.assertTrue(np.all(np.isfinite(outputs.numpy())))

    def test_qpann_forward_and_fit(self):
        # Regression test: a QPANN must be not just constructible (as in
        # test_qpanns_and_customlayers above) but actually callable and *trainable*. This
        # would have caught two Keras-3-specific bugs that the old test did not exercise:
        #   1. The `CustomDense`/`Dense.kernel` property conflict (see test_customdense_forward
        #      above) -- raised on the very first forward pass.
        #   2. `QPANN`/`CircuitToErrorRatesEinSum` used to cache `self.stochastic_mask` and
        #      `self.hamiltonian_mask` as `tf.constant(...)`, created eagerly at construction
        #      time. Keras 3's `Model.fit` wraps `train_step` in nested `tf.function`s, and
        #      referencing a `tf.constant` from a different (already-closed) graph context
        #      inside one of those raised `InaccessibleTensorError: ... is out of scope`. This
        #      only manifested during `.fit()` (not a bare forward pass), so it required both
        #      fixes to be applied together in order to write a passing end-to-end test. The
        #      masks are now plain numpy arrays.
        pspec = _ProcessorSpec(2, ['Gxpi2', 'Gypi2'], {}, {}, geometry="line", qubit_labels=[0, 1])
        circuits = [Circuit('[Gxpi2:0Gypi2:1]@(0,1)'), Circuit('[Gypi2:0][Gxpi2:1]@(0,1)')]
        modelled_error_generators = [('H', ('XI',)), ('S', ('IX',))]

        tensors = encoding.error_generator_tensors(circuits, modelled_error_generators, pspec,
                                                     alpha_representation='concise')
        probabilities, alphas = tensors['probabilities'], tensors['alphas']

        encoder = encoding.StandardCircuitEncoder(pspec)
        circuits_tensor = encoding.circuits_to_tensor(circuits, encoder)
        adjacency_matrix = snippers.undirected_adjacency_matrix_from_edges([(0, 1)], [0, 1])
        snipper = snippers.layer_snipper_from_qubit_graph(modelled_error_generators, encoder,
                                                            adjacency_matrix, hops=1)

        model = qpanns.QPANN(encoder.length, modelled_error_generators, snipper)
        x = [circuits_tensor, alphas, probabilities]

        # A bare forward pass -- would have raised AttributeError pre-fix (bug 1 above).
        output = model(x)
        self.assertEqual(tuple(output.shape), (len(circuits), 2 ** pspec.num_qubits))
        self.assertTrue(np.all(np.isfinite(output.numpy())))

        # Actually train -- would have raised InaccessibleTensorError pre-fix (bug 2 above),
        # even with bug 1 already fixed.
        initial_weights = [w.numpy().copy() for w in model.trainable_variables]
        self.assertTrue(len(initial_weights) > 0)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-1), loss='mse')
        model.fit(x, probabilities + 0.01, epochs=2, verbose=0)

        # At least one weight should have actually changed, proving gradients flowed through
        # training (not just that .fit() ran without crashing).
        changed = any(not np.allclose(w0, w1.numpy())
                      for w0, w1 in zip(initial_weights, model.trainable_variables))
        self.assertTrue(changed)

    def test_definite_outcome_hamiltonian_alpha_is_zero(self):
        # Regression test for a dead-code bug in `encoding._circuit_loop_probs`: it used to
        # contain
        #     if egtype == 'H' and (isclose(probabilities[l], 0.) or isclose(probabilities[l], 1.)):
        #         alpha = 0
        #     alpha = scale * alpha_coefficient(...)   # <- ran unconditionally, clobbering the above
        # so the special case never took effect. The special case is physically motivated: a
        # bitstring whose ideal probability is already exactly 0 or 1 (a "definite outcome")
        # must have zero first-order sensitivity to a Hamiltonian ('H'-type) error, because a
        # Hamiltonian rate can have either sign and the probability is bounded to [0, 1] -- if
        # the derivative were nonzero, the probability would leave [0, 1] for one sign of the
        # rate. (This argument doesn't apply to 'S'-type/stochastic rates, which are physically
        # constrained to be non-negative, so the fixed code only special-cases 'H'.)
        #
        # It was checked empirically (see git history/PR discussion) that `alpha_coefficient`
        # already independently evaluates to exactly 0 in this regime -- so this fix does not
        # change any previously-computed alpha values -- but it is still a real, worthwhile fix:
        # it removes confusing dead code, and it skips a real (and, for near-deterministic
        # circuits, often substantial) amount of otherwise-unnecessary computation, since
        # `alpha_coefficient` is not cheap and definite-outcome bitstrings are extremely common
        # for the high-fidelity circuits this codebase targets.
        pspec = _ProcessorSpec(2, ['Gxpi2', 'Gypi2'], {}, {}, geometry="line", qubit_labels=[0, 1])
        # Two full-pi rotations on each qubit: a deterministic bit flip, |00> -> |11> w.p. 1.
        circuit = Circuit('[Gxpi2:0Gxpi2:1][Gxpi2:0Gxpi2:1]@(0,1)')
        modelled_error_generators = [('H', ('XI',)), ('H', ('IX',)), ('S', ('XI',))]

        tensors = encoding.error_generator_tensors([circuit], modelled_error_generators, pspec,
                                                     alpha_representation='concise')
        probabilities, alphas = tensors['probabilities'], tensors['alphas']
        np.testing.assert_allclose(probabilities, [[0., 0., 0., 1.]])

        # Every bitstring here is a definite outcome (probability exactly 0 or 1), so every
        # 'H'-type generator's alpha should be exactly 0 for every bitstring and every layer.
        hamiltonian_columns = [j for j, eg in enumerate(modelled_error_generators) if eg[0] == 'H']
        for l in range(probabilities.shape[1]):
            for j in hamiltonian_columns:
                np.testing.assert_array_equal(alphas[0, l, :, j], 0.0)

        # Confirm the short-circuit is real (not just numerically inconsequential): with every
        # bitstring at a definite outcome, `alpha_coefficient` should never actually be called
        # for the 'H'-type generators.
        nbit_strings = ['00', '01', '10', '11']
        call_indices = []
        original_alpha_coefficient = encoding.alpha_coefficient

        def _counting_alpha_coefficient(i, *args, **kwargs):
            call_indices.append(i)
            return original_alpha_coefficient(i, *args, **kwargs)

        encoding.alpha_coefficient = _counting_alpha_coefficient
        try:
            encoding._circuit_loop_probs(circuit, tensors['indices'][0], nbit_strings, 2)
        finally:
            encoding.alpha_coefficient = original_alpha_coefficient

        hamiltonian_indices = {errgentools.error_generator_index(*eg)
                                for eg in modelled_error_generators if eg[0] == 'H'}
        self.assertFalse(any(i in hamiltonian_indices for i in call_indices))

if __name__ == '__main__':
    unittest.main()

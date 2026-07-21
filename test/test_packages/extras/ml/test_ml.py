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

    def test_errgentools_pauli_pairs(self):
        # 'C' (Pauli-correlation) and 'A' (active) type error generators are indexed by an
        # UNORDERED pair of two DISTINCT, non-identity Paulis (see "A Taxonomy of Small Errors",
        # Blume-Kohout et al., Sec. V.C-V.D), unlike 'H'/'S' which are each indexed by a single
        # Pauli. This tests the pair-indexing utilities added to support 'C'/'A'.

        # canonical_pauli_pair: sorts into lexicographic order, tracks whether a swap occurred.
        P, Q, swapped = errgentools.canonical_pauli_pair('ZZ', 'XY')
        self.assertEqual((P, Q, swapped), ('XY', 'ZZ', True))
        P, Q, swapped = errgentools.canonical_pauli_pair('XY', 'ZZ')
        self.assertEqual((P, Q, swapped), ('XY', 'ZZ', False))
        with self.assertRaises(ValueError):
            errgentools.canonical_pauli_pair('XY', 'XY')  # P == Q is disallowed

        # error_generator_canonicalization_sign: nontrivial (+-1) ONLY for 'A', since
        # A_{P,Q} = -A_{Q,P} (antisymmetric), whereas C_{P,Q} = C_{Q,P} (symmetric) and 'H'/'S'
        # have no ordering ambiguity (single Pauli).
        self.assertEqual(errgentools.error_generator_canonicalization_sign('C', ('ZZ', 'XY')), 1)
        self.assertEqual(errgentools.error_generator_canonicalization_sign('A', ('XY', 'ZZ')), 1)
        self.assertEqual(errgentools.error_generator_canonicalization_sign('A', ('ZZ', 'XY')), -1)
        self.assertEqual(errgentools.error_generator_canonicalization_sign('H', ('X',)), 1)

        # num_pauli_pairs: matches "A Taxonomy of Small Errors"'s own stated total of 105 for n=2
        # (Sec. V.G: "There are 105 linearly independent two-qubit Pauli-correlation generators").
        self.assertEqual(errgentools.num_pauli_pairs(1), 3)
        self.assertEqual(errgentools.num_pauli_pairs(2), 105)

        # pauli_pair_to_index / index_to_pauli_pair: exhaustive round-trip for small n, order
        # invariance, and full coverage of [0, num_pauli_pairs(n)) with no duplicates.
        import itertools
        for n in [1, 2, 3]:
            nonident = [errgentools.index_to_paulistring(i, n) for i in range(1, 4**n)]
            M = errgentools.num_pauli_pairs(n)
            seen = set()
            for p1, p2 in itertools.combinations(nonident, 2):
                idx = errgentools.pauli_pair_to_index(p1, p2, n)
                self.assertEqual(idx, errgentools.pauli_pair_to_index(p2, p1, n))  # order-invariant
                self.assertNotIn(idx, seen)
                seen.add(idx)
                self.assertEqual(errgentools.index_to_pauli_pair(idx, n), tuple(sorted((p1, p2))))
            self.assertEqual(seen, set(range(M)))

        # up_to_weight_k_pauli_pairs: cross-validate against a slow/naive reference for small n,k.
        def naive_pauli_pairs(k, n):
            nonident = [errgentools.index_to_paulistring(i, n) for i in range(1, 4**n)]
            pairs = set()
            for p1, p2 in itertools.combinations(nonident, 2):
                support = set(i for i, c in enumerate(p1) if c != 'I') | set(i for i, c in enumerate(p2) if c != 'I')
                if len(support) <= k:
                    pairs.add(tuple(sorted((p1, p2))))
            return pairs

        for n in [1, 2, 3]:
            for k in range(1, n + 1):
                fast = set(errgentools.up_to_weight_k_pauli_pairs(k, n))
                self.assertEqual(fast, naive_pauli_pairs(k, n))
                self.assertTrue(all(p1 < p2 for p1, p2 in fast))  # canonical order

        # up_to_weight_k_pauli_pairs_from_qubit_graph: a weight-1 pair should always be allowed
        # (single-qubit support is trivially "connected"); a weight-2 pair split across two
        # UNCONNECTED qubits (no edge, and no path within num_hops) should be excluded, but
        # included once num_hops is large enough to connect them.
        n = 3
        line_laplacian = np.array([[1, -1, 0], [-1, 2, -1], [0, -1, 1]])  # 0-1-2 line graph
        pairs_hop1 = errgentools.up_to_weight_k_pauli_pairs_from_qubit_graph(2, n, line_laplacian, 1)
        # A pair spanning qubits {0,2} (not adjacent in the line graph) should NOT appear at hops=1.
        self.assertFalse(any(
            (set(i for i, c in enumerate(p1) if c != 'I') | set(i for i, c in enumerate(p2) if c != 'I')) == {0, 2}
            for p1, p2 in pairs_hop1
        ))
        pairs_hop2 = errgentools.up_to_weight_k_pauli_pairs_from_qubit_graph(2, n, line_laplacian, 2)
        self.assertTrue(any(
            (set(i for i, c in enumerate(p1) if c != 'I') | set(i for i, c in enumerate(p2) if c != 'I')) == {0, 2}
            for p1, p2 in pairs_hop2
        ))

    def test_errgentools_error_generator_index_ca(self):
        # 'H'/'S' backward compatibility: exact same index values as before 'C'/'A' were added.
        self.assertEqual(errgentools.error_generator_index('H', ('IX',)), 1)
        self.assertEqual(errgentools.error_generator_index('H', ('ZZ',)), 15)
        self.assertEqual(errgentools.error_generator_index('S', ('IX',)), 17)
        self.assertEqual(errgentools.index_to_error_gen(1, 2), ('H', ('IX',)))
        self.assertEqual(errgentools.index_to_error_gen(17, 2), ('S', ('IX',)))

        # 'C'/'A': index is invariant to input order (internally canonicalized), and
        # index_to_error_gen's round trip always returns the canonical (sorted) order.
        for typ in ['C', 'A']:
            idx1 = errgentools.error_generator_index(typ, ('XY', 'ZZ'))
            idx2 = errgentools.error_generator_index(typ, ('ZZ', 'XY'))
            self.assertEqual(idx1, idx2)
            self.assertEqual(errgentools.index_to_error_gen(idx1, 2), (typ, ('XY', 'ZZ')))

        # Full index range [0, num_error_generators(n)) round-trips exactly and without overlap
        # between the H/S/C/A sub-ranges, for a small n.
        n = 2
        total = errgentools.num_error_generators(n)
        self.assertEqual(total, 2 * 4**n + 2 * errgentools.num_pauli_pairs(n))
        seen_indices = set()
        for i in range(total):
            typ, paulis = errgentools.index_to_error_gen(i, n)
            self.assertEqual(errgentools.error_generator_index(typ, paulis), i)
            self.assertNotIn(i, seen_indices)
            seen_indices.add(i)
        self.assertEqual(seen_indices, set(range(total)))
        with self.assertRaises(ValueError):
            errgentools.index_to_error_gen(total, n)  # one past the end

        # Validity checks: 'C'/'A' with P==Q or an identity Pauli must raise.
        with self.assertRaises(ValueError):
            errgentools.error_generator_index('C', ('XY', 'XY'))
        with self.assertRaises(ValueError):
            errgentools.error_generator_index('A', ('II', 'XY'))
        with self.assertRaises(ValueError):
            errgentools.error_generator_index('Q', ('XY',))  # unknown type

    def test_errgentools_up_to_weight_k_error_gens_ca(self):
        n = 3
        laplacian = np.array([[1, -1, 0], [-1, 2, -1], [0, -1, 1]])

        # Mixed H/S/C/A dispatch: correct counts and correct tuple "shape" per type.
        egs = errgentools.up_to_weight_k_error_gens(2, n, egtypes=['H', 'S', 'C', 'A'])
        h = [eg for eg in egs if eg[0] == 'H']
        s = [eg for eg in egs if eg[0] == 'S']
        c = [eg for eg in egs if eg[0] == 'C']
        a = [eg for eg in egs if eg[0] == 'A']
        self.assertEqual(len(h), len(errgentools.up_to_weight_k_paulis(2, n)))
        self.assertEqual(len(s), len(h))
        self.assertEqual(len(c), len(errgentools.up_to_weight_k_pauli_pairs(2, n)))
        self.assertEqual(len(a), len(c))
        self.assertTrue(all(len(eg[1]) == 1 for eg in h + s))
        self.assertTrue(all(len(eg[1]) == 2 for eg in c + a))

        # Graph-restricted dispatch, single type ('C' only).
        egs_c_only = errgentools.up_to_weight_k_error_gens_from_qubit_graph(2, n, laplacian, 1, egtypes=['C'])
        self.assertTrue(all(eg[0] == 'C' for eg in egs_c_only))
        self.assertEqual(len(egs_c_only), len(errgentools.up_to_weight_k_pauli_pairs_from_qubit_graph(2, n, laplacian, 1)))

        # Backward compatibility: default egtypes=['H','S'] is unaffected by the 'C'/'A' additions.
        egs_default = errgentools.up_to_weight_k_error_gens(2, n)
        self.assertTrue(all(eg[0] in ('H', 'S') for eg in egs_default))

        # Unknown types raise a clear error (both dispatch functions).
        with self.assertRaises(ValueError):
            errgentools.up_to_weight_k_error_gens(2, n, egtypes=['H', 'Q'])
        with self.assertRaises(ValueError):
            errgentools.up_to_weight_k_error_gens_from_qubit_graph(2, n, laplacian, 1, egtypes=['Q'])

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

    def test_encoding_active_generator_canonicalization_sign(self):
        # Regression/correctness test for a subtlety in `circuit_error_propagation_matrices`
        # specific to 'A'-type (active) error generators: A_{P,Q} = -A_{Q,P} (antisymmetric
        # under swapping its two indexing Paulis -- see "A Taxonomy of Small Errors", Sec. V.D,
        # Eq. 16), so if Clifford propagation happens to produce a non-canonically-ordered pair
        # (Q,P) with Q > P, reindexing it into the canonical index for (P,Q) requires ALSO
        # flipping the sign of the propagated rate to compensate -- otherwise the wrong sign
        # would be used. ('C'-type generators need no such correction, since C_{P,Q}=C_{Q,P}.)
        #
        # Hand-derived example: propagating A_{X,Y} through a single-qubit "Gzpi2" layer (whose
        # stim tableau is exactly the S/phase gate: X->Y with sign +1, Y->-X with sign -1) gives
        # propagated basis labels (Y, X) [i.e. bel_to_strings() = ('Y','X')] with raw propagated
        # weightmod = sign(X->Y) * sign(Y->-X) = (+1)*(-1) = -1. Since (Y,X) is NOT in canonical
        # order (X < Y), canonicalizing it to (X,Y) requires an additional sign flip of -1
        # (A_{Y,X} = -A_{X,Y}). Combined: (-1) * (-1) = +1.
        pspec = _ProcessorSpec(1, ['Gzpi2', 'Gh'], {}, {}, geometry="line", qubit_labels=[0])
        # 2-layer circuit: the layer-0 error generator propagates through layer 1 (Gzpi2).
        circuit = Circuit('[Gh:0][Gzpi2:0]@(0)')
        indices, signs = encoding.circuit_error_propagation_matrices(circuit, [('A', ('X', 'Y'))])
        expected_index = errgentools.error_generator_index('A', ('X', 'Y'))
        self.assertEqual(indices[0, 0], expected_index)
        self.assertEqual(signs[0, 0], 1)

        # Cross-check against a DIRECT, independent computation of the physical alpha value
        # (using the RAW, non-canonicalized propagated label) to confirm the canonicalized
        # index+sign gives the exact same physically-meaningful result -- not just that the
        # sign bookkeeping is internally self-consistent.
        from pygsti.errorgenpropagation.errorpropagator import ErrorGeneratorPropagator
        from pygsti.errorgenpropagation.localstimerrorgen import LocalStimErrorgenLabel as LSE

        prop = ErrorGeneratorPropagator(None)
        stim_layers = prop.construct_stim_layers(circuit, drop_first_layer=True)
        propagation_layers = prop.construct_propagation_layers(stim_layers)
        lse = LSE('A', [stim.PauliString('X'), stim.PauliString('Y')])
        raw_propagated, raw_weightmod = lse.propagate_error_gen_tableau(propagation_layers[0], 1.0)

        tableau = circuit.convert_to_stim_tableau()
        canonical_label = errgentools.index_to_error_gen(indices[0, 0], 1, as_label=True)
        for bs in ['0', '1']:
            from pygsti.tools import errgenproptools as ep
            raw_alpha = raw_weightmod * ep.alpha(raw_propagated, tableau, bs).real
            canonical_alpha = signs[0, 0] * ep.alpha(canonical_label, tableau, bs).real
            self.assertAlmostEqual(raw_alpha, canonical_alpha)

        # A parallel 'C'-type check: same propagation math (weightmod), but NO canonicalization
        # sign correction should ever be applied (C is symmetric under swapping P,Q).
        indices_c, signs_c = encoding.circuit_error_propagation_matrices(circuit, [('C', ('X', 'Y'))])
        self.assertEqual(indices_c[0, 0], errgentools.error_generator_index('C', ('X', 'Y')))
        self.assertEqual(signs_c[0, 0], -1)  # the raw weightmod itself, uncorrected

    def test_encoding_error_generator_tensors_with_ca(self):
        # End-to-end test of the default ('concise') `error_generator_tensors` pipeline with a
        # mix of all four error generator types, including weight-2 'C'/'A' pairs whose two
        # Paulis act on different qubits. Cross-validates every entry of the resulting alpha
        # tensor against a direct, independent `alpha_coefficient` computation.
        pspec = _ProcessorSpec(2, ['Gxpi2', 'Gypi2', 'Gcphase'], {}, {'Gcphase': [(0, 1)]},
                                geometry="line", qubit_labels=[0, 1])
        circuits = [
            Circuit('[Gxpi2:0][Gcphase:0:1]@(0,1)'),
            Circuit('[Gypi2:0Gxpi2:1][Gcphase:0:1]@(0,1)'),
        ]
        modelled_error_generators = [('H', ('XI',)), ('S', ('IX',)), ('C', ('XI', 'YZ')), ('A', ('XI', 'YZ'))]

        tensors = encoding.error_generator_tensors(circuits, modelled_error_generators, pspec,
                                                     alpha_representation='concise')
        probabilities, alphas = tensors['probabilities'], tensors['alphas']
        indices, signs = tensors['indices'], tensors['signs']
        nbit_strings = ['00', '01', '10', '11']

        for c_idx, circuit in enumerate(circuits):
            tableau = circuit.convert_to_stim_tableau()
            scale = 1 / 2 ** encoding._egptools.random_support(tableau)
            for l, bs in enumerate(nbit_strings):
                for layer in range(circuit.depth):
                    for j in range(len(modelled_error_generators)):
                        idx = indices[c_idx, layer, j]
                        sign = signs[c_idx, layer, j]
                        expected = sign * scale * encoding.alpha_coefficient(idx, 2, tableau, bs)
                        self.assertAlmostEqual(alphas[c_idx, l, layer, j], expected, places=10)

    def test_encoding_matrix_representation_rejects_ca(self):
        # The dense ('matrix'/'expanded') alpha representation only supports 'H'/'S' (its fixed
        # `2*4**n`-wide array would need to grow by `2*num_pauli_pairs(n)` -- which is O(16**n)
        # -- to accommodate 'C'/'A'; see `dense_alpha_matrix`'s docstring). It should raise a
        # clear, early `NotImplementedError` if asked to include 'C'/'A' generators, rather than
        # silently producing wrong results or an opaque IndexError from array overflow.
        pspec = _ProcessorSpec(2, ['Gxpi2', 'Gypi2', 'Gcphase'], {}, {'Gcphase': [(0, 1)]},
                                geometry="line", qubit_labels=[0, 1])
        circuits = [Circuit('[Gxpi2:0][Gcphase:0:1]@(0,1)')]

        with self.assertRaises(NotImplementedError):
            encoding.error_generator_tensors(circuits, [('C', ('XI', 'YZ'))], pspec, alpha_representation='matrix')
        with self.assertRaises(NotImplementedError):
            encoding.error_generator_tensors(circuits, [('A', ('XI', 'YZ'))], pspec, alpha_representation='matrix')

        # 'H'/'S'-only 'matrix' usage must still work (backward compatibility).
        result = encoding.error_generator_tensors(circuits, [('H', ('XI',)), ('S', ('IZ',))], pspec,
                                                    alpha_representation='matrix')
        self.assertEqual(result['alphas'].shape, (1, 4, 2 * 4**2))

        # dense_alpha_matrix itself should also raise directly if given an out-of-range (C/A) index.
        tableau = circuits[0].convert_to_stim_tableau()
        with self.assertRaises(NotImplementedError):
            encoding.dense_alpha_matrix(tableau, 2, populate_for_error_generators=[2 * 4**2])

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

    def test_snippers_ca_union_support(self):
        # For 'C'/'A' error generators (indexed by a pair of two Paulis), the "support" a
        # snipper should look at is the UNION of the qubits acted on by BOTH Paulis in the pair
        # (see "A Taxonomy of Small Errors", Sec. VIII: "The support of a generator C_{P,Q} or
        # A_{P,Q} is the union of the supports of P and Q"), not just the first Pauli's support
        # (which is all that the pre-'C'/'A' implementation looked at).
        nonstd_gate_unitaries = {}
        availability = {}
        pspec = _ProcessorSpec(4, ['{idle}', 'Gx', 'Gy'], nonstd_gate_unitaries, availability, geometry="line")
        encoder = encoding.StandardCircuitEncoder(pspec)
        adj = snippers.undirected_adjacency_matrix_from_edges([(0, 1), (1, 2), (2, 3)], [0, 1, 2, 3])

        # 'IIIX' touches qubit 3 only; 'IYII' touches qubit 1 only. Union support = {1, 3}.
        error_generators_ca = [('C', ('IIIX', 'IYII'))]
        snip_ca = snippers.layer_snipper_from_qubit_graph(error_generators_ca, encoder, adj, hops=0)
        expected = encoder.indices_for_qubits([1, 3])
        self.assertEqual(snip_ca[0], expected)

        # With hops=1, should pick up neighbors of BOTH qubits 1 and 3 (i.e. also 0, 2 -- the
        # full line graph), not just neighbors of one of them.
        snip_ca_hops1 = snippers.layer_snipper_from_qubit_graph(error_generators_ca, encoder, adj, hops=1)
        expected_hops1 = encoder.indices_for_qubits([0, 1, 2, 3])
        self.assertEqual(snip_ca_hops1[0], expected_hops1)

        # A same-qubit 'A' pair (both Paulis on qubit 3 only) should behave like a weight-1
        # single-Pauli generator on that qubit.
        error_generators_a_1q = [('A', ('IIIX', 'IIIY'))]
        snip_a_1q = snippers.layer_snipper_from_qubit_graph(error_generators_a_1q, encoder, adj, hops=0)
        self.assertEqual(snip_a_1q[0], encoder.indices_for_qubits([3]))

        # A mix of H/S (1-tuple) and C/A (2-tuple) generators in the same call should still work
        # (regression check that the H/S code path is unaffected by the 2-tuple support).
        mixed = [('H', ('IIIX',)), ('C', ('IIIX', 'IYII'))]
        snip_mixed = snippers.layer_snipper_from_qubit_graph(mixed, encoder, adj, hops=0)
        self.assertEqual(snip_mixed[0], encoder.indices_for_qubits([3]))
        self.assertEqual(snip_mixed[1], encoder.indices_for_qubits([1, 3]))

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

    def test_qpann_forward_and_fit_with_ca_generators(self):
        # Full end-to-end test that the entire pipeline (encoding -> tensors -> QPANN -> forward
        # pass + training) works correctly with 'C' (Pauli-correlation) and 'A' (active) type
        # error generators mixed in with 'H'/'S', including weight-2 'C'/'A' pairs whose two
        # Paulis act on DIFFERENT qubits (exercising the snipper's union-support logic and
        # encoding's canonicalization-sign logic together, in the full model context).
        pspec = _ProcessorSpec(2, ['Gxpi2', 'Gypi2', 'Gcphase'], {}, {'Gcphase': [(0, 1)]},
                                geometry="line", qubit_labels=[0, 1])
        circuits = [
            Circuit('[Gxpi2:0Gypi2:1]Gcphase:0:1[Gxpi2:1Gypi2:0]@(0,1)'),
            Circuit('[Gypi2:0][Gcphase:0:1][Gxpi2:1]@(0,1)'),
            Circuit('[Gxpi2:0Gxpi2:1]Gcphase:0:1@(0,1)'),
        ]
        modelled_error_generators = [
            ('H', ('XI',)), ('S', ('IX',)),
            ('C', ('XI', 'YZ')), ('A', ('XI', 'YZ')),  # weight-2 pair spanning both qubits
        ]

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

        # Forward pass.
        output = model(x)
        self.assertEqual(tuple(output.shape), (len(circuits), 2 ** pspec.num_qubits))
        self.assertTrue(np.all(np.isfinite(output.numpy())))

        # 'C'/'A' should be treated as unconstrained/linear (like 'H'), NOT squared (like 'S').
        self.assertEqual(list(model.stochastic_mask), [False, True, False, False])

        # Train.
        initial_weights = [w.numpy().copy() for w in model.trainable_variables]
        self.assertTrue(len(initial_weights) > 0)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-1), loss='mse')
        model.fit(x, probabilities + 0.01, epochs=2, verbose=0)
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

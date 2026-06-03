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

if __name__ == '__main__':
    unittest.main()

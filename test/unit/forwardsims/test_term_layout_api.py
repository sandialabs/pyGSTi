"""Term layouts accept the shared forward-simulator derivative keyword."""
import unittest
import numpy as np

from pygsti.circuits import Circuit
from pygsti.forwardsims import TermForwardSimulator
from pygsti.modelpacks import smq1Q_XYI
from pygsti.objectivefns import ModelDatasetCircuitsStore


class TermLayoutAPITester(unittest.TestCase):
    def setUp(self):
        self.model = smq1Q_XYI.target_model('static unitary', evotype='statevec')
        self.model.set_all_parameterizations('H+S')
        self.model.from_vector(np.full(self.model.num_params, 1e-4))
        self.model.sim = TermForwardSimulator(mode='taylor-order', max_order=1)
        self.circuits = [Circuit([('Gxpi2', 0)], line_labels=(0,))]

    def test_default_objective_store_constructs_real_term_layout(self):
        store = ModelDatasetCircuitsStore(self.model, None, self.circuits,
                                         array_types=('e',))
        probabilities = store.layout.allocate_local_array('e', 'd')
        try:
            self.model.sim.bulk_fill_probs(probabilities, store.layout)
            self.assertTrue(np.all(np.isfinite(probabilities)))
            self.assertAlmostEqual(float(probabilities.sum()), 1., places=6)
        finally:
            store.layout.free_local_array(probabilities)

    def test_plural_tuple_and_legacy_integer_dimensions_agree(self):
        n = self.model.num_params
        simulator = self.model.sim
        layouts = [simulator.create_layout(self.circuits, array_types=('e', 'ep'),
                                            derivative_dimensions=(n,)),
                   simulator.create_layout(self.circuits, array_types=('e', 'ep'),
                                            derivative_dimensions=(n, n)),
                   simulator.create_layout(self.circuits, array_types=('e', 'ep'),
                                            derivative_dimension=n),
                   simulator.create_layout(self.circuits, None, None, ('e', 'ep'), n)]
        for layout in layouts:
            derivative = layout.allocate_local_array('ep', 'd')
            try:
                simulator.bulk_fill_dprobs(derivative, layout)
                self.assertEqual(derivative.shape[1], n)
                self.assertTrue(np.all(np.isfinite(derivative)))
            finally:
                layout.free_local_array(derivative)

    def test_empty_dimensions_create_value_only_layout(self):
        layout = self.model.sim.create_layout(self.circuits, array_types=('e',),
                                              derivative_dimensions=())
        probabilities = layout.allocate_local_array('e', 'd')
        try:
            self.model.sim.bulk_fill_probs(probabilities, layout)
            self.assertTrue(np.all(np.isfinite(probabilities)))
            self.assertAlmostEqual(float(probabilities.sum()), 1., places=6)
        finally:
            layout.free_local_array(probabilities)

    def test_unequal_hessian_axes_and_conflicting_alias_rejected(self):
        n = self.model.num_params
        with self.assertRaisesRegex(ValueError, 'unequal Hessian'):
            self.model.sim.create_layout(self.circuits, derivative_dimensions=(n, n + 1))
        with self.assertRaisesRegex(ValueError, 'only one'):
            self.model.sim.create_layout(self.circuits, derivative_dimensions=(n,), derivative_dimension=n)

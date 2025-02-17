import unittest

from pygsti.forwardsims.mapforwardsim import MapForwardSimulator

import pygsti
from pygsti.modelpacks import smq1Q_XY
from ..testutils import BaseTestCase


class LayoutTestCase(BaseTestCase):

    def setUp(self):
        super(LayoutTestCase, self).setUp()
        self.circuits = pygsti.circuits.to_circuits(["Gxpi2:0", "Gypi2:0", "Gxpi2:0Gxpi2:0",
                                                         "Gypi2:0Gypi2:0", "Gxpi2:0Gypi2:0"])
        self.model = smq1Q_XY.target_model()
        model_matrix = self.model.copy()
        model_matrix.sim = 'map'
        self.model_matrix = model_matrix

    def _test_layout(self, layout):
        self.assertEqual(layout.num_elements, len(self.circuits) * 2)  # 2 outcomes per circuit
        self.assertEqual(layout.num_elements, len(layout))
        self.assertEqual(layout.num_circuits, len(self.circuits))
        for i, c in enumerate(self.circuits):
            print("Circuit%d: " % i, c)
            indices = layout.indices(c)
            outcomes = layout.outcomes(c)
            self.assertEqual(pygsti.tools.slicetools.length(indices), 2)
            self.assertEqual(outcomes, (('0',), ('1',)))
            if isinstance(indices, slice):
                self.assertEqual(layout.indices_for_index(i), indices)
            else:  # indices is an array
                self.assertArraysEqual(layout.indices_for_index(i), indices)
            self.assertEqual(layout.outcomes_for_index(i), outcomes)
            self.assertEqual(layout.indices_and_outcomes(c), (indices, outcomes))
            self.assertEqual(layout.indices_and_outcomes_for_index(i), (indices, outcomes))

        circuits_seen = set()
        for indices, c, outcomes in layout.iter_unique_circuits():
            self.assertFalse(c in circuits_seen)
            circuits_seen.add(c)

            if isinstance(indices, slice):
                self.assertEqual(indices, layout.indices(c))
            else:
                self.assertArraysEqual(indices, layout.indices(c))
            self.assertEqual(outcomes, layout.outcomes(c))

        layout_copy = layout.copy()
        self.assertEqual(layout.circuits, layout_copy.circuits)

    def test_base_layout(self):
        self._test_layout(pygsti.layouts.copalayout.CircuitOutcomeProbabilityArrayLayout.create_from(self.circuits[:], self.model))

    def test_map_layout(self):
        self._test_layout(pygsti.layouts.maplayout.MapCOPALayout(self.circuits[:], self.model))
        #TODO: test split layouts

    def test_matrix_layout(self):
        self._test_layout(pygsti.layouts.matrixlayout.MatrixCOPALayout(self.circuits[:], self.model_matrix))

if __name__ == '__main__':
    unittest.main(verbosity=2)

from pygsti.modelpacks.legacy import std1Q_XYI as std
from ..testutils         import BaseTestCase


class AlgorithmTestCase(BaseTestCase):
    def setUp(self):
        super(AlgorithmTestCase, self).setUp()
        self.mdl_target_noisy = std.target_model().randomize_with_unitary(0.001, seed=1234)

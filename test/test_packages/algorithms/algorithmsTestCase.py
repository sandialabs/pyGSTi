from pygsti.construction import std1Q_XYI as std
from ..testutils         import BaseTestCase, compare_files, temp_files

class AlgorithmTestCase(BaseTestCase):
    def setUp(self):
        super(AlgorithmTestCase, self).setUp()
        self.gs_target_noisy = std.gs_target.randomize_with_unitary(0.001, seed=1234)

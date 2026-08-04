from unittest import mock

import numpy as np

from pygsti.modelpacks.legacy import std1Q_XYI as std
from pygsti.tools import chi2fns
from . import fixtures as pkg
from ..util import BaseCase


class Chi2LogLTester(BaseCase):
    def setUp(self):
        self.dataset = pkg.dataset.copy()

    def test_chi2(self):
        # TODO rather than faking expensive calls this should really use a simpler dataset
        with mock.patch('pygsti.forwardsims.matrixforwardsim.MatrixForwardSimulator._compute_hproduct_cache') as mock_hproduct_cache:
            mock_hproduct_cache.return_value = np.zeros((868, 60, 60, 4, 4))
            # TODO assert correctness
            chi2fns.chi2(std.target_model(), self.dataset)
            chi2fns.chi2(std.target_model(), self.dataset, mem_limit=1000000000)
            # NOTE: this Chi2LogLTester.test_chi2 function used to invoke deprecated
            # functions without inspecting their output. Those invocations have been
            # removed. Invocations of non-deprecated functions remain, without 
            # checks for correctness.
        return

    def test_chi2_raises_on_out_of_memory(self):
        with self.assertRaises(MemoryError):
            chi2fns.chi2(std.target_model(), self.dataset, mem_limit=1)  # No memory for you

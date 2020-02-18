import numpy as np
from unittest import mock

from ..util import BaseCase
from . import fixtures as pkg

from pygsti.modelpacks.legacy import std1Q_XYI as std
from pygsti.objects.dataset import DataSet
from pygsti.tools import chi2fns


class Chi2LogLTester(BaseCase):
    def setUp(self):
        self.dataset = pkg.dataset.copy()

    def test_chi2_terms(self):
        mdl = pkg.mdl_lsgst_go.copy()
        terms = chi2fns.chi2_terms(mdl, self.dataset)
        # TODO assert correctness

    def test_chi2_fn(self):
        # TODO rather than faking expensive calls this should really use a simpler dataset
        with mock.patch('pygsti.objects.matrixforwardsim.MatrixForwardSimulator._compute_hproduct_cache') as mock_hproduct_cache:
            mock_hproduct_cache.return_value = np.zeros((868, 60, 60, 4, 4))

            chi2, grad = chi2fns.chi2(std.target_model(), self.dataset, returnGradient=True)
            chi2fns.chi2(std.target_model(), self.dataset, returnHessian=True)

            chi2fns.chi2fn_2outcome(N=100, p=0.5, f=0.6)
            chi2fns.chi2fn_2outcome_wfreqs(N=100, p=0.5, f=0.6)
            chi2fns.chi2fn(N=100, p=0.5, f=0.6)
            chi2fns.chi2fn_wfreqs(N=100, p=0.5, f=0.6)
            chi2fns.chi2(std.target_model(), self.dataset, memLimit=100000)
            # TODO assert correctness

            with self.assertRaises(ValueError):
                chi2fns.chi2(std.target_model(), self.dataset, useFreqWeightedChiSq=True)  # no impl yet

    def test_chi2_raises_on_out_of_memory(self):
        with self.assertRaises(MemoryError):
            chi2fns.chi2(std.target_model(), self.dataset, memLimit=0)  # No memory for you

import pytest

from pygsti import algorithms as alg, circuits as pc
from pygsti.drivers import bootstrap as bs
from . import fixtures as pkg
from ..util import BaseCase


class BootstrapBase(BaseCase):
    @classmethod
    def setUpClass(cls):
        cls.opLabels = pkg.opLabels
        cls.prep_fids = pkg.prep_fids
        cls.meas_fids = pkg.meas_fids
        cls.germs = pkg.germs
        cls.ds = pkg.dataset
        cls.full_target = pkg.model.copy()
        cls.mdl = alg.run_lgst(
            cls.ds, cls.prep_fids, cls.meas_fids, target_model=cls.full_target, svd_truncate_to=4, verbosity=0
        )

    def setUp(self):
        self.ds = self.ds.copy()
        self.mdl = self.mdl.copy()

class BootstrapDatasetTester(BootstrapBase):
    def test_make_bootstrap_dataset_parametric(self):
        bootds_p = bs.create_bootstrap_dataset(self.ds, 'parametric', self.mdl, seed=1234)
        # TODO assert correctness

    def test_make_bootstrap_dataset_nonparametric(self):
        bootds_np = bs.create_bootstrap_dataset(self.ds, 'nonparametric', seed=1234)
        # TODO assert correctness

    def test_make_bootstrap_dataset_raises_on_bad_generation_method(self):
        with self.assertRaises(ValueError):
            bs.create_bootstrap_dataset(self.ds, 'foobar', seed=1)

    def test_make_bootstrap_dataset_raises_on_no_parametric_model(self):
        with self.assertRaises(ValueError):
            bs.create_bootstrap_dataset(self.ds, 'parametric', seed=1)

    def test_make_bootstrap_dataset_raises_on_nonparametric_with_model(self):
        with self.assertRaises(ValueError):
            bs.create_bootstrap_dataset(self.ds, 'nonparametric', self.mdl, seed=1)

@pytest.mark.filterwarnings('ignore:Setting the first element of a max-length list to zero') # Explicitly using this to build LGST only
class BootstrapModelTester(BootstrapBase):
    def setUp(self):
        super(BootstrapModelTester, self).setUp()
        self.maxLengths = [0]  # just do LGST strings to make this fast...

    def test_make_bootstrap_models_parametric(self):
        # TODO optimize
        bootgs_p = bs.create_bootstrap_models(
            2, self.ds, 'parametric', self.prep_fids, self.meas_fids,
            self.germs, self.maxLengths, input_model=self.mdl, target_model=self.full_target,
            return_data=False
        )
        # TODO assert correctness

    def test_make_bootstrap_models_with_list(self):
        # TODO optimize
        custom_strs = pc.create_lsgst_circuit_lists(
            self.full_target, self.prep_fids, self.meas_fids, self.germs, [1]
        )
        bootgs_p_custom = bs.create_bootstrap_models(
            2, self.ds, 'parametric', None, None, None, None,
            lsgst_lists=custom_strs, input_model=self.mdl, target_model=self.full_target,
            return_data=False
        )
        # TODO assert correctness

    def test_make_bootstrap_models_nonparametric(self):
        # TODO optimize
        bootgs_np, bootds_np2 = bs.create_bootstrap_models(
            2, self.ds, 'nonparametric', self.prep_fids, self.meas_fids,
            self.germs, self.maxLengths, target_model=self.full_target,
            return_data=True
        )
        # TODO assert correctness

    def test_make_bootstrap_models_raises_on_no_model(self):
        with self.assertRaises(ValueError):
            bs.create_bootstrap_models(
                2, self.ds, 'parametric', self.prep_fids, self.meas_fids,
                self.germs, self.maxLengths, return_data=False
            )

@pytest.mark.filterwarnings('ignore:Setting the first element of a max-length list to zero') # Explicitly using this to build LGST only
class BootstrapUtilityTester(BootstrapBase):
    @classmethod
    def setUpClass(cls):
        super(BootstrapUtilityTester, cls).setUpClass()
        maxLengths = [0]
        cls.bootgs_p = bs.create_bootstrap_models(
            2, cls.ds, 'parametric', cls.prep_fids, cls.meas_fids,
            cls.germs, maxLengths, input_model=cls.mdl, target_model=cls.full_target,
            return_data=False
        )

    def setUp(self):
        super(BootstrapUtilityTester, self).setUp()
        self.bootgs_p = self.bootgs_p[:]  # python2-compatible copy

    def test_gauge_optimize_model_list(self):
        bs.gauge_optimize_models(
            self.bootgs_p, self.full_target, gate_metric='frobenius',
            spam_metric='frobenius', plot=False
        )
        # TODO assert correctness

    def test_gauge_optimize_model_list_with_plot(self):
        with self.assertRaises(NotImplementedError):
            bs.gauge_optimize_models(
                self.bootgs_p, self.full_target, gate_metric='frobenius',
                spam_metric='frobenius', plot=True)

    def test_bootstrap_utilities(self):
        #Test utility functions -- just make sure they run for now...
        def gsFn(mdl):
            return mdl.dim
        bs._model_stdev(gsFn, self.bootgs_p)
        bs._model_mean(gsFn, self.bootgs_p)

        bs._to_mean_model(self.bootgs_p, self.full_target)
        bs._to_std_model(self.bootgs_p, self.full_target)
        bs._to_rms_model(self.bootgs_p, self.full_target)
        # TODO assert correctness

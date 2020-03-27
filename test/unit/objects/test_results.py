import pickle

from ..util import BaseCase, Namespace
from . import fixtures as pkg

from pygsti.objects import results, estimate
from pygsti.modelpacks.legacy import std1Q_XYI as std


class ResultsBase(object):
    def setUp(self):
        # prepare a results object
        self.model = pkg.mdl_lsgst_go
        self.ds = pkg.dataset
        self.maxLengthList = pkg.maxLengthList
        self.gss = pkg.lsgstStructs

        # Construct results but do not init
        self.res = results.Results()


class BareResultsTester(ResultsBase, BaseCase):
    def test_init_dataset(self):
        self.res.init_dataset(self.ds)
        # TODO assert correctness

    def test_init_circuits(self):
        self.res.init_dataset(self.ds)
        self.res.init_circuits(self.gss)
        # TODO assert correctness

    def test_add_estimate_raises_before_initialization(self):
        # Raise before initializing dataset
        with self.assertRaises(ValueError):
            self.res.add_estimate(None, None, None, None)
        self.res.init_dataset(self.ds)

        # Raise before initializing circuits
        with self.assertRaises(ValueError):
            self.res.add_estimate(None, None, None, None)


class ResultsTester(ResultsBase, BaseCase):
    def setUp(self):
        super(ResultsTester, self).setUp()
        # Init results before tests
        self.res.init_dataset(self.ds)
        self.res.init_circuits(self.gss)

    def test_init_warns_on_reinitialization(self):
        with self.assertWarns(Warning):
            self.res.init_dataset(self.ds)  # usually don't want to re-init
        with self.assertWarns(Warning):
            self.res.init_circuits(self.gss)  # usually don't want to re-init

    def test_init_circuits_raises_on_bad_arg(self):
        # XXX Don't test this unless init_circuits can actually take a string  EGN: we can remove this test.
        # (and if it can, document it)
        with self.assertRaises(ValueError):
            self.res.init_circuits("foobar")

    def test_add_estimate(self):
        # add estimates
        self.res.add_estimate(
            std.target_model(), std.target_model(),
            [self.model] * len(self.maxLengthList), parameters={'objective': 'logl'},
            estimate_key="default"
        )
        # TODO assert correctness


class PopulatedResultsTester(ResultsBase, BaseCase):
    def setUp(self):
        super(PopulatedResultsTester, self).setUp()
        self.res.init_dataset(self.ds)
        self.res.init_circuits(self.gss)
        self.res.add_estimate(
            std.target_model(), std.target_model(),
            [self.model] * len(self.maxLengthList), parameters={'objective': 'logl'},
            estimate_key="default"
        )

    def test_add_estimate_warns_on_overwrite(self):
        with self.assertWarns(Warning):
            self.res.add_estimate(
                std.target_model(), std.target_model(),
                [self.model] * len(self.maxLengthList), parameters={'objective': 'logl'},
                estimate_key="default"
            )  # re-init existing estimate

    def test_to_string(self):
        s = str(self.res)
        # TODO assert correctness

    def test_view(self):
        #Results views
        rview = self.res.view(['default'])
        rview2 = self.res.view('default')  # this works too
        # TODO assert correctness

    def test_add_estimate_from_results(self):
        # add_estimates from other results
        res2 = results.Results()
        res2.init_dataset(self.ds)
        res2.init_circuits(self.gss)

        res2.add_estimates(self.res, ['default'])

    def test_rename_estimate(self):
        # rename estimates
        self.res.rename_estimate('default', 'renamed_default')

    def test_rename_estimate_raises_on_missing_key(self):
        with self.assertRaises(KeyError):
            self.res.rename_estimate('foobar', 'renamed_foobar')

    def test_add_model_test(self):
        # add estimate from model test
        mdl_guess = std.target_model().depolarize(op_noise=0.07, spam_noise=0.03)
        self.res.add_model_test(std.target_model(), mdl_guess, estimate_key='Test', gauge_opt_keys="auto")
        # TODO assert correctness

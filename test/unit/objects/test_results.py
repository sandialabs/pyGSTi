import pickle

from ..util import BaseCase, Namespace
from . import fixtures as pkg

from pygsti.protocols.gst import ModelEstimateResults
from pygsti.protocols import Protocol, ProtocolData, CircuitListsDesign
from pygsti.objects import BulkCircuitList
from pygsti.modelpacks.legacy import std1Q_XYI as std
from pygsti.protocols import estimate


class ResultsBase(object):
    def setUp(self):
        # prepare a results object
        self.model = pkg.mdl_lsgst_go
        self.ds = pkg.dataset
        self.maxLengthList = pkg.maxLengthList
        self.gss = pkg.lsgstStructs

        # Construct results
        edesign = CircuitListsDesign([BulkCircuitList(circuit_struct)
                                      for circuit_struct in pkg.lsgstStructs])
        data = ProtocolData(edesign, pkg.dataset)
        self.res = ModelEstimateResults(data, Protocol("test-protocol"))


class ResultsTester(ResultsBase, BaseCase):

    def test_add_estimate(self):
        self.res.add_estimate(
            estimate.Estimate.gst_init(
                self.res, std.target_model(), std.target_model(),
                [self.model] * len(self.maxLengthList), parameters={'objective': 'logl'}),
            estimate_key="default"
        )
        # TODO assert correctness


class PopulatedResultsTester(ResultsBase, BaseCase):

    def setUp(self):
        super().setUp()

        # add an estimate
        self.res.add_estimate(
            estimate.Estimate.gst_init(
                self.res, std.target_model(), std.target_model(),
                [self.model] * len(self.maxLengthList), parameters={'objective': 'logl'}),
            estimate_key="default"
        )
    
    def test_add_estimate_warns_on_overwrite(self):
        with self.assertWarns(Warning):
            self.res.add_estimate(
                estimate.Estimate.gst_init(
                    self.res, std.target_model(), std.target_model(),
                    [self.model] * len(self.maxLengthList), parameters={'objective': 'logl'}),
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
        edesign = CircuitListsDesign([BulkCircuitList(circuit_struct)
                                      for circuit_struct in pkg.lsgstStructs])
        data = ProtocolData(edesign, pkg.dataset)
        res2 = ModelEstimateResults(data, Protocol("test-protocol2"))

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

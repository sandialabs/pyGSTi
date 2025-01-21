import pickle

from pygsti.modelpacks.legacy import std1Q_XYI as std
from pygsti.circuits import CircuitList
from pygsti.protocols import Protocol, ProtocolData, CircuitListsDesign
from pygsti.protocols import estimate
from pygsti.protocols.gst import ModelEstimateResults
from . import fixtures as pkg
from ..util import BaseCase


class EstimateBase(object):
    @classmethod
    def setUpClass(cls):
        cls.model = pkg.mdl_lsgst_go
        cls.maxLengthList = pkg.maxLengthList

        edesign = CircuitListsDesign([CircuitList(circuit_struct)
                                      for circuit_struct in pkg.lsgstStrings])
        data = ProtocolData(edesign, pkg.dataset)
        cls.res = ModelEstimateResults(data, Protocol("test-protocol"))

    def setUp(self):
        self.model = self.model.copy()
        self.res = self.res.copy()

    def test_get_effective_dataset(self):
        # Get effective estimate dataset
        effds = self.est.create_effective_dataset()
        effds, subMxs = self.est.create_effective_dataset(return_submxs=True)
        # TODO assert correctness

    def test_view(self):
        #Estimate views
        est_view = self.est.view(None)
        est_view = self.est.view(['test'])
        # TODO assert correctness

    def test_to_string(self):
        #Estimate & results render as str
        s = str(self.est)
        # TODO assert correctness

    def test_pickle(self):
        s = pickle.dumps(self.est)
        est_pickled = pickle.loads(s)
        # TODO assert correctness


class ResultsEstimateTester(EstimateBase, BaseCase):
    def setUp(self):
        super(ResultsEstimateTester, self).setUp()
        self.res.add_estimate(
            estimate.Estimate.create_gst_estimate(
                self.res, std.target_model(), std.target_model(),
                [self.model] * len(self.maxLengthList),
                parameters={'objective': 'logl'}),
            estimate_key="default"
        )
        self.est = self.res.estimates['default']

    def test_add_gaugeoptimized(self):
        # TODO optimize
        goparams = {'item_weights': {'gates': 1.0, 'spam': 0.1},
                    'method': 'BFGS'}  # method so we don't need a legit comm
        self.est.add_gaugeoptimized(goparams, label="test", comm=None, verbosity=None)
        # TODO assert correctness


class EmptyEstimateTester(EstimateBase, BaseCase):
    def setUp(self):
        super(EmptyEstimateTester, self).setUp()
        self.est = estimate.Estimate(self.res)

    def test_add_gaugeoptimized_raises_on_no_model(self):
        with self.assertRaises(ValueError):
            goparams = {'item_weights': {'gates': 1.0, 'spam': 0.1}, 'target_model': self.model}
            self.est.add_gaugeoptimized(goparams, label="test", comm=None, verbosity=None)  # goparams must have 'model'

    def test_add_gaugeoptimized_raises_on_no_target_model(self):
        with self.assertRaises(ValueError):
            goparams = {'item_weights': {'gates': 1.0, 'spam': 0.1}, 'model': self.model}
            self.est.add_gaugeoptimized(goparams, label="test", comm=None,
                                        verbosity=None)  # goparams must have 'target_model'

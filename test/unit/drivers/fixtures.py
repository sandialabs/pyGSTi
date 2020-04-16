"""Shared test fixtures for pygsti.drivers unit tests"""
from ..util import Namespace
from pygsti.modelpacks.legacy import std1Q_XYI as std
import pygsti.construction as pc

ns = Namespace()
ns.model = std.target_model()
ns.opLabels = list(ns.model.operations.keys())
ns.fiducials = std.fiducials
ns.germs = std.germs
ns.maxLengthList = [1, 2, 4]


@ns.memo
def datagen_gateset(self):
    return self.model.depolarize(op_noise=0.05, spam_noise=0.1)


@ns.memo
def lsgstStrings(self):
    return pc.make_lsgst_lists(
        self.opLabels, self.fiducials, self.fiducials,
        self.germs, self.maxLengthList
    )


@ns.memo
def dataset(self):
    return pc.generate_fake_data(
        self.datagen_gateset, self.lsgstStrings[-1],
        n_samples=1000, sample_error='binomial', seed=100
    )

"""Shared test fixtures for pygsti.algorithms unit tests"""
from ..util import Namespace

from pygsti.modelpacks.legacy import std1Q_XYI as std
import pygsti.construction as pc
import pygsti.algorithms as alg

ns = Namespace()
ns.model = std.target_model()
ns.opLabels = list(ns.model.operations.keys())
ns.fiducials = std.fiducials
ns.germs = std.germs
ns.maxLengthList = [0, 1, 2]


_SEED = 1234

@ns.memo
def datagen_gateset(self):
    return self.model.depolarize(op_noise=0.05, spam_noise=0.1)


@ns.memo
def lgstStrings(self):
    return pc.list_lgst_circuits(
        self.fiducials, self.fiducials, self.opLabels
    )


@ns.memo
def elgstStrings(self):
    return pc.make_elgst_lists(
        self.opLabels, self.germs, self.maxLengthList
    )


@ns.memo
def lsgstStrings(self):
    return pc.make_lsgst_lists(
        self.opLabels, self.fiducials, self.fiducials,
        self.germs, self.maxLengthList
    )


@ns.memo
def ds(self):
    expList = pc.make_lsgst_experiment_list(
        self.opLabels, self.fiducials, self.fiducials,
        self.germs, self.maxLengthList
    )
    return pc.generate_fake_data(
        self.datagen_gateset, expList,
        nSamples=1000, sampleError='binomial', seed=_SEED
    )


@ns.memo
def ds_lgst(self):
    return pc.generate_fake_data(
        self.datagen_gateset, self.lgstStrings,
        nSamples=10000, sampleError='binomial', seed=_SEED
    )


@ns.memo
def mdl_lgst(self):
    return alg.do_lgst(
        self.ds, self.fiducials, self.fiducials, self.model,
        svdTruncateTo=4, verbosity=0
    )


@ns.memo
def mdl_lgst_go(self):
    return alg.gaugeopt_to_target(
        self.mdl_lgst, self.model, {'spam': 1.0, 'gates': 1.0}, checkJac=True
    )


@ns.memo
def mdl_clgst(self):
    return alg.contract(self.mdl_lgst_go, 'CPTP')


@ns.memo
def mdl_target_noisy(self):
    return self.model.randomize_with_unitary(0.001, seed=_SEED)

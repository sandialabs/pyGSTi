"""Shared test fixtures for pygsti.algorithms unit tests"""
import pygsti.algorithms as alg
import pygsti.circuits as circuits
import pygsti.data as data
from pygsti.modelpacks.legacy import std1Q_XYI as std
from ..util import Namespace

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
    return circuits.create_lgst_circuits(
        self.fiducials, self.fiducials, self.opLabels
    )


@ns.memo
def elgstStrings(self):
    return circuits.create_elgst_lists(
        self.opLabels, self.germs, self.maxLengthList
    )


@ns.memo
def lsgstStrings(self):
    return circuits.create_lsgst_circuit_lists(
        self.opLabels, self.fiducials, self.fiducials,
        self.germs, self.maxLengthList
    )


@ns.memo
def ds(self):
    expList = circuits.create_lsgst_circuits(
        self.opLabels, self.fiducials, self.fiducials,
        self.germs, self.maxLengthList
    )
    return data.simulate_data(
        self.datagen_gateset, expList,
        num_samples=1000, sample_error='binomial', seed=_SEED
    )


@ns.memo
def ds_lgst(self):
    return data.simulate_data(
        self.datagen_gateset, self.lgstStrings,
        num_samples=10000, sample_error='binomial', seed=_SEED
    )


@ns.memo
def mdl_lgst(self):
    return alg.run_lgst(
        self.ds, self.fiducials, self.fiducials, self.model,
        svd_truncate_to=4, verbosity=0
    )


@ns.memo
def mdl_lgst_go(self):
    return alg.gaugeopt_to_target(
        self.mdl_lgst, self.model, {'spam': 1.0, 'gates': 1.0}, check_jac=True
    )


@ns.memo
def mdl_clgst(self):
    return alg.contract(self.mdl_lgst_go, 'CPTP')


@ns.memo
def mdl_target_noisy(self):
    return self.model.randomize_with_unitary(0.001, seed=_SEED)

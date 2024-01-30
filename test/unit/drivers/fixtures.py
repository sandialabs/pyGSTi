"""Shared test fixtures for pygsti.drivers unit tests"""
import pygsti.circuits as pc
import pygsti.data as pdata
from pygsti.modelpacks import smq1Q_XY as std
from ..util import Namespace

ns = Namespace()
ns.model = std.target_model()
ns.pspec = std.processor_spec()
ns.opLabels = list(ns.model.operations.keys())
ns.prep_fids = std.prep_fiducials()
ns.meas_fids = std.meas_fiducials()
ns.germs = std.germs(lite=True)
ns.maxLengthList = [1, 2, 4]


@ns.memo
def datagen_gateset(self):
    return self.model.depolarize(op_noise=0.05, spam_noise=0.1)


@ns.memo
def lsgstStrings(self):
    return pc.create_lsgst_circuit_lists(
        self.opLabels, self.prep_fids, self.meas_fids,
        self.germs, self.maxLengthList
    )


@ns.memo
def dataset(self):
    return pdata.simulate_data(
        self.datagen_gateset, self.lsgstStrings[-1],
        num_samples=1000, sample_error='binomial', seed=100
    )

"""Shared test fixtures for pygsti.tools unit tests"""
from ..util import Namespace

import pygsti
from pygsti.modelpacks.legacy import std1Q_XYI as std

ns = Namespace()
ns.model = std.target_model()
ns.opLabels = list(ns.model.operations.keys())
ns.fiducials = std.fiducials
ns.germs = std.germs
ns.maxLengthList = [0, 1, 2, 4, 8]
ns.CM = pygsti.objects.profiler._get_mem_usage()


@ns.memo
def datagen_gateset(self):
    return self.model.depolarize(op_noise=0.05, spam_noise=0.1)


@ns.memo
def expList(self):
    return pygsti.construction.make_lsgst_experiment_list(
        self.opLabels, self.fiducials, self.fiducials, self.germs, self.maxLengthList)


@ns.memo
def dataset(self):
    # Was previously written to disk as 'analysis.dataset'
    return pygsti.construction.generate_fake_data(
        self.datagen_gateset, self.expList, n_samples=10000,
        sample_error='binomial', seed=100
    )


@ns.memo
def mdl_lgst(self):
    return pygsti.do_lgst(self.dataset, self.fiducials, self.fiducials, self.model, svdTruncateTo=4, verbosity=0)


@ns.memo
def mdl_lgst_go(self):
    return pygsti.gaugeopt_to_target(self.mdl_lgst, self.model, {'spam': 1.0, 'gates': 1.0}, check_jac=True)


@ns.memo
def mdl_clgst(self):
    return pygsti.contract(self.mdl_lgst_go, "CPTP")


@ns.memo
def lsgstStrings(self):
    return pygsti.construction.make_lsgst_lists(
        self.opLabels, self.fiducials, self.fiducials, self.germs,
        self.maxLengthList
    )


@ns.memo
def mdl_lsgst(self):
    return pygsti.do_iterative_mc2gst(
        self.dataset, self.mdl_clgst, self.lsgstStrings, verbosity=0,
        minProbClipForWeighting=1e-6, probClipInterval=(-1e6, 1e6),
        memLimit=self.CM + 10*1024**3
    )


@ns.memo
def mdl_lsgst_go(self):
    # Was previously written to disk as 'analysis.model'
    return pygsti.gaugeopt_to_target(self.mdl_lsgst, self.model, {'spam': 1.0})

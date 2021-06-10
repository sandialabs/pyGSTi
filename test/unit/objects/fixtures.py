"""Shared test fixtures for pygsti.objects unit tests"""
import pygsti
from pygsti.modelpacks.legacy import std1Q_XYI as std
from ..util import Namespace

ns = Namespace()
ns.model = std.target_model()
ns.opLabels = list(ns.model.operations.keys())
ns.fiducials = std.fiducials
ns.germs = std.germs
ns.maxLengthList = [1, 2]
ns.CM = pygsti.baseobjs.profiler._get_mem_usage()


@ns.memo
def datagen_gateset(self):
    return self.model.depolarize(op_noise=0.05, spam_noise=0.1)


@ns.memo
def expList(self):
    return pygsti.construction.create_lsgst_circuits(
        self.opLabels, self.fiducials, self.fiducials, self.germs, self.maxLengthList)


@ns.memo
def dataset(self):
    # Was previously written to disk as 'analysis.dataset'
    return pygsti.construction.simulate_data(
        self.datagen_gateset, self.expList, num_samples=10000,
        sample_error='binomial', seed=100
    )


@ns.memo
def mdl_lgst(self):
    return pygsti.run_lgst(self.dataset, self.fiducials, self.fiducials, self.model, svd_truncate_to=4, verbosity=0)


@ns.memo
def mdl_lgst_go(self):
    return pygsti.gaugeopt_to_target(self.mdl_lgst, self.model, {'spam': 1.0, 'gates': 1.0}, check_jac=True)


@ns.memo
def mdl_clgst(self):
    return pygsti.contract(self.mdl_lgst_go, "CPTP")


@ns.memo
def lsgstStrings(self):
    return pygsti.construction.create_lsgst_circuit_lists(
        self.opLabels, self.fiducials, self.fiducials, self.germs,
        self.maxLengthList
    )


@ns.memo
def lsgstStructs(self):
    return pygsti.construction.make_lsgst_structs(
        self.opLabels, self.fiducials, self.fiducials, self.germs,
        self.maxLengthList
    )


@ns.memo
def mdl_lsgst(self):
    chi2_builder = pygsti.objectivefns.Chi2Function.builder(
        regularization={'min_prob_clip_for_weighting': 1e-6},
        penalties={'prob_clip_interval': (-1e6, 1e6)})
    models, _, _ = pygsti.algorithms.core.run_iterative_gst(
        self.dataset, self.mdl_clgst, self.lsgstStrings,
        optimizer=None,
        iteration_objfn_builders=[chi2_builder],
        final_objfn_builders=[],
        resource_alloc={'mem_limit': self.CM + 1024**3},
        verbosity=0
    )
    return models[-1]


@ns.memo
def mdl_lsgst_go(self):
    # Was previously written to disk as 'analysis.model'
    return pygsti.gaugeopt_to_target(self.mdl_lsgst, self.model, {'spam': 1.0})

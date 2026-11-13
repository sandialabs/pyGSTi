"""Fit on every rank, then do follow-up work on rank 0 alone (issue #964).

Run under ``mpiexec -n 2`` by ``test_rank0_followup.py``. Rank 0 prints one JSON line;
any exception or hang fails the calling test.
"""
import json
import sys

import numpy as np
from mpi4py import MPI
from mpi4py.util.sync import Mutex

import pygsti
from pygsti.modelpacks import smq1Q_XY

comm = MPI.COMM_WORLD

edesign = smq1Q_XY.create_gst_experiment_design(2)
datagen = smq1Q_XY.target_model().depolarize(op_noise=0.01, spam_noise=0.02)
dataset = pygsti.data.simulate_data(datagen, edesign, 1000, seed=2026)
data = pygsti.protocols.ProtocolData(edesign, dataset)

gst = pygsti.protocols.GateSetTomography(
    smq1Q_XY.target_model('full TP'), gaugeopt_suite='stdgaugeopt', verbosity=0,
    optimizer={'maxiter': 10})
gst_results = gst.run(data, comm=comm, disable_checkpointing=True)
mt = pygsti.protocols.ModelTest(datagen, verbosity=0)
mt_results = mt.run(data, comm=comm, disable_checkpointing=True)

# The caller gets back `hessians_agree` and `stores_invalidated` for all ranks
gst_results.add_hessians(compute_hessian_kwargs={'comm': comm})
est = gst_results.estimates['GateSetTomography']
crf = est.confidence_region_factories[('stdgaugeopt', 'final')]
inv_hessian = crf.inv_hessian_projections['intrinsic error']
hessians_agree = all(np.array_equal(inv_hessian, h) for h in comm.allgather(inv_hessian))
stores_invalidated = {
    '_final_objfn'         : est._final_objfn        is None,
    '_final_mdc_store'     : est._final_mdc_store    is None,
    '_per_iter_mdc_store'  : est._per_iter_mdc_store is None
}


# By calling misfit_sigma with no argument, as happens in report generation,
# the calculation happens as though no MPI were enabled.
# 
# In order for those calculations to be correct we need to have invalidated
# the estimate's caches from the calls to MPI-enabled `GateSetTomography.run`
# and `ModelTest.run`.
#
ns_e = est.misfit_sigma(resource_alloc=None)
ns_t = mt_results.estimates['ModelTest'].misfit_sigma(resource_alloc=None)

with Mutex(comm=comm):
    out = {
        'hessians_agree': hessians_agree,
        'stores_invalidated': stores_invalidated,
        'nsigma_e': ns_e,
        'nsigma_t': ns_t
    }
    print(json.dumps(out))
    sys.stdout.flush()

comm.Barrier()

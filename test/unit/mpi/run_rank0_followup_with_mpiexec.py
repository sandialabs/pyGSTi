"""Fit on every rank, then do follow-up work on rank 0 alone (issue #964).

Run under ``mpiexec -n 2`` by ``test_rank0_followup.py``. Rank 0 prints one JSON line;
any exception or hang fails the calling test.
"""
import json
import sys

import numpy as np
from mpi4py import MPI

import pygsti
from pygsti.modelpacks import smq1Q_XY

comm = MPI.COMM_WORLD

edesign = smq1Q_XY.create_gst_experiment_design(2)
datagen = smq1Q_XY.target_model().depolarize(op_noise=0.05, spam_noise=0.02)
dataset = pygsti.data.simulate_data(datagen, edesign, 1000, seed=2026)
data = pygsti.protocols.ProtocolData(edesign, dataset)

gst = pygsti.protocols.GateSetTomography(
    smq1Q_XY.target_model('full TP'), gaugeopt_suite='stdgaugeopt', verbosity=0,
    optimizer={'maxiter': 100})
gst_results = gst.run(data, comm=comm, disable_checkpointing=True)
mt = pygsti.protocols.ModelTest(datagen, verbosity=0)
mt_results = mt.run(data, comm=comm, disable_checkpointing=True)

# Bug 1: add_hessians on every rank, with the comm.
gst_results.add_hessians(compute_hessian_kwargs={'comm': comm})
crf = gst_results.estimates['GateSetTomography'].confidence_region_factories[('stdgaugeopt', 'final')]
inv_hessian = crf.inv_hessian_projections['intrinsic error']
hessians_agree = all(np.array_equal(inv_hessian, h) for h in comm.allgather(inv_hessian))

# Bug 2: follow-up calls on rank 0 alone.
if comm.rank == 0:
    out = {
        'hessians_agree': hessians_agree,
        'gst_misfit_sigma': float(gst_results.estimates['GateSetTomography'].misfit_sigma()),
        'mt_misfit_sigma': float(mt_results.estimates['ModelTest'].misfit_sigma()),
    }
    print(json.dumps(out))
    sys.stdout.flush()
comm.Barrier()

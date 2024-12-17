#!/usr/bin/env python

import cProfile
import os
import pickle

from mpi4py import MPI

import pygsti
from pygsti.modelpacks import smq2Q_XYICNOT as std

comm = MPI.COMM_WORLD
resource_alloc = pygsti.baseobjs.ResourceAllocation(comm)
mdl = std.target_model()

exp_design = std.get_gst_experiment_design(64)

mdl_datagen = mdl.depolarize(op_noise=0.01, spam_noise=0.01)

# First time running through, generate reference dataset
#if comm.rank == 0:
#    ds = pygsti.construction.simulate_data(mdl_datagen, exp_design, 1000, seed=1234, comm=resource_alloc.comm)
#    pickle.dump(ds, open('reference_ds.pkl','wb'))
#sys.exit(0)
ds_ref = pickle.load(open('reference_ds.pkl','rb'))
ds = ds_ref

MINCLIP = 1e-4
chi2_builder = pygsti.objects.Chi2Function.builder(
        'chi2', regularization={'min_prob_clip_for_weighting': MINCLIP}, penalties={'cptp_penalty_factor': 0.0})
mle_builder = pygsti.objects.PoissonPicDeltaLogLFunction.builder(
        'logl', regularization={'min_prob_clip': MINCLIP, 'radius': MINCLIP})
iteration_builders = [chi2_builder]; final_builders = [mle_builder]
builders = pygsti.protocols.GSTObjFnBuilders(iteration_builders, final_builders)

tol = 1e-6

opt = None  # default

#GST TEST
data = pygsti.protocols.ProtocolData(exp_design, ds)
#mdl.sim = pygsti.baseobjs.MatrixForwardSimulator(num_atoms=1)
mdl.sim = pygsti.objects.MapForwardSimulator(num_atoms=1, max_cache_size=0)
gst = pygsti.protocols.GateSetTomography(mdl, gaugeopt_suite=False,  # 'randomizeStart': 0e-6,
                                         objfn_builders=builders, optimizer=opt, verbosity=4)

profiler = cProfile.Profile()
profiler.enable()

results = gst.run(data, comm=comm)

profiler.disable()

num_procs = comm.Get_size()
num_procs_host = os.environ.get('PYGSTI_MAX_HOST_PROCS', num_procs)
os.makedirs(f'{num_procs}_{num_procs_host}.profile', exist_ok=True)
profiler.dump_stats(f'{num_procs}_{num_procs_host}.profile/{comm.rank}.prof')

results = None  # Needed to cause shared mem to be freed by garbage collection *before* python shuts down shared mem "system"

comm.barrier()
if comm.rank == 0: print("DONE")

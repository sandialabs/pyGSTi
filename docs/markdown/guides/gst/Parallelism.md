---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.19.3
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Running GST in parallel with MPI

pyGSTi's core GST routines are written to spread work across processors using MPI. Long fits are the reason you'd care: the circuit-by-circuit likelihood evaluations that dominate the runtime divide cleanly across ranks, and the per-rank memory footprint drops with them.

There are two ways to get there. `Protocol.run_mpi` launches the worker processes for you, waits for them, and hands back a results object, so you never leave the notebook. The lower-level route is to write a script that imports `mpi4py`, passes an `MPI.Comm` to `Protocol.run`, and launch it yourself under `mpiexec`. Use the first unless you need control over how the job is launched. The page closes with checkpointing — parallel or not, a fit long enough to care about is long enough to lose to a crash, and checkpoints let it resume instead of restart.

Both require the `mpi4py` Python package and an MPI library underneath it.

```{note}
**This page will not run without a working MPI installation.** The `run_mpi` call
below shells out to a launcher (`mpiexec` or `mpirun`) that has to be on your
`PATH`, and the `mpi4py` package alone does not provide one: it supplies the
Python bindings, not the launcher or the MPI runtime underneath it. Without a
launcher you will see `FileNotFoundError: resolve_mpiexec: could not find an MPI
launcher on PATH`.

Any of these will get you one:

- conda: `conda install -c conda-forge openmpi mpi4py`, which installs the launcher
  into the active environment and builds `mpi4py` against it.
- Debian/Ubuntu: `apt install openmpi-bin`.
- An HPC cluster: `module load openmpi`.
```

## Setup

Both routes fit the same simulated single-qubit data, built here from the `smq1Q_XYI` model pack. Generate it once, serially, before any MPI process starts: the counts come from a random number generator, and generating them inside a parallel job would hand every rank a different dataset. This is only a concern for simulated data. Experimental counts get read from disk and are the same everywhere.

```{code-cell} ipython3
from pygsti.modelpacks import smq1Q_XYI as mp
from pygsti.protocols import ProtocolData, StandardGST
from pygsti.data import simulate_data

exp_design  = mp.create_gst_experiment_design(max_max_length=32)  # type: ignore
mdl_ideal   = mp.target_model()                                   # type: ignore
mdl_datagen = mdl_ideal.depolarize(op_noise=0.1, spam_noise=0.001)

data  = simulate_data(mdl_datagen, exp_design.all_circuits_needing_data, num_samples=1000, seed=2020)
pdata = ProtocolData(exp_design, data)
```

## Letting pyGSTi launch the workers

`run_mpi` serializes the protocol and the data to a working directory, writes a runner script, launches `num_ranks` copies of it under an MPI launcher, and reads the results back when the subprocess exits. With `num_ranks=1` it skips all of that and calls `run` directly.

The call below passes two extra arguments for robustness across MPI setups and machines with few cores:

- `env={'FI_PROVIDER': 'sockets'}` can be needed on some MPI distributions. Try without it as well, and omit it unless you need it.
- `extra_mpi_args`, carrying `--oversubscribe`, lets the launcher start more ranks than the machine has cores. This flag is only needed when using Open MPI and you ask for more MPI ranks than available cores (without this flag, using Open MPI with more ranks than cores leads `run_mpi` to raise a `CalledProcessError`).

We inspect the environment to decide whether we need to pass `--oversubscribe`. We need that inspection because other MPI distributions (e.g., MPICH and Intel MPI) neither need nor recognize this argument.

```{code-cell} ipython3
import shutil

launcher = shutil.which('mpiexec') or shutil.which('mpirun')
# `ompi_info` ships with Open MPI and with no other MPI distribution, which makes it
# a more dependable test than the launcher's --version banner: invoked as `mpiexec`,
# Open MPI 4.x brands itself "OpenRTE" and never prints the string "Open MPI" at all.
is_openmpi = shutil.which('ompi_info') is not None
extra_args = ['--oversubscribe'] if is_openmpi else []

print("launcher:", launcher)
print("extra_mpi_args:", extra_args)
```

```{code-cell} ipython3
protocol = StandardGST(verbosity=2)
results = protocol.run_mpi(pdata,
    num_ranks=3, mpiexec='auto', env={'FI_PROVIDER': 'sockets'},
    extra_mpi_args=extra_args
)
```

Notice that the progress output isn't duplicated three times. Only the root rank writes to stdout, so a parallel fit reads exactly like a serial one.

From here, post-processing is the same as for any other `ModelEstimateResults` object.

```{code-cell} ipython3
from pygsti.report import construct_standard_report

report = construct_standard_report(
    results, title="MPI Example Report", verbosity=0
)
report.write_html('../../../example_files/mpi_example_brief', connected=True, auto_open=False)
```

Open the <a href="../../../reports/mpi_example_brief/main.html">report</a>.

## Driving MPI yourself

Sometimes you need the launch to be yours: a scheduler submits the job, a hostfile picks the nodes, or the environment isn't one `run_mpi` can reproduce from a notebook. In that case, write the script. The pattern is three lines of MPI and one changed keyword argument: get `MPI.COMM_WORLD`, load the data, and pass the comm to `run`.

`mpi4py` doesn't coexist well with Jupyter kernels, so the script has to be a separate file rather than a cell. Start by putting the data somewhere the workers can read it.

```{code-cell} ipython3
pdata.write("../../../example_files/mpi_gst_example")
```

Two things in the script below go beyond the bare comm. `memlimit` sets a rough per-processor memory budget in bytes, which tells pyGSTi how finely to partition its computations; it matters for the large fits that motivate running in parallel in the first place. Setting `results` to `None` at the end releases the shared-memory buffers before the interpreter tears down garbage collection.

Only the root rank writes results to disk. Every rank holds the same answer, so writing from all of them would just be three processes racing for the same files.

```{code-cell} ipython3
mpiScript = """
import pygsti

#get MPI comm
from mpi4py import MPI
comm = MPI.COMM_WORLD

if comm.Get_rank() == 0:
    print("Running on %d ranks" % comm.Get_size(), flush=True)

#load in data
data = pygsti.io.read_data_from_dir("../../../example_files/mpi_gst_example")

#Specify a per-core memory limit (useful for larger GST calculations)
memLim = 2.1*(1024)**3  # 2.1 GB

#Perform TP-constrained GST
protocol = pygsti.protocols.StandardGST("full TP")
results = protocol.run(data, memlimit=memLim, comm=comm)

if comm.Get_rank() == 0:
    results.write()  #write results (within same directory as data was loaded from)

results=None # needed to free shared memory before garbage collection is torn down
"""
with open("../../../example_files/mpi_example_script.py","w") as f:
    f.write(mpiScript)
```

Now run it on 3 processors. In a notebook, prefix a line with `!` to execute it as a shell command, and wrap any Python value you want to splice into the command line in braces. `FI_PROVIDER=sockets` and `extra_args` play the same roles they did for `run_mpi`; on a non-Open-MPI launcher `extra_args` is empty and the command is just `mpiexec -n 3 python3 ...`.

A `!` line that fails does not raise, so watch its output. If the launcher rejects the command, the error you actually see will be a missing-results error in the next code cell.

If `mpiexec` doesn't exist on your system, try `mpirun`; one or the other came with your MPI distribution.

```{code-cell} ipython3
oversub = ' '.join(extra_args)
!env FI_PROVIDER=sockets mpiexec -n 3 {oversub} python3 "../../../example_files/mpi_example_script.py"
```

The results are now on disk next to the data. Read them back and analyze them as usual.

```{code-cell} ipython3
import pygsti

results_from_disk = pygsti.io.read_results_from_dir("../../../example_files/mpi_gst_example",
                                                    name="StandardGST")
print(results_from_disk.estimates.keys())
```

## Clusters and schedulers

`run_mpi` assumes it can launch the job right now, which a login node usually won't let you do. Two escape hatches exist. Pass `dry_run=True` along with `persistent_dir` and pyGSTi writes the data and runner script to that directory, then returns `None` instead of launching anything, leaving you to submit the job however your site expects. It prints the command it would have run only when the protocol's `verbosity` is positive or unset, so a protocol built with `verbosity=0` writes the files silently. For SLURM specifically, `stage_slurm` writes the same working files plus a batch script ready for `sbatch`.

## Checkpointing

The `GateSetTomography` and `StandardGST` protocols both support checkpointing to enable resuming GST analysis after an unexpected failure, such as an out-of-memory error, or an unexpected timeout in resource limited compute environments (clusters etc.), or for whatever other reason. Checkpointing works the same with or without MPI, and it is enabled by default, so no additional changes are needed in order to have these generated.

Each protocol has a corresponding checkpoint object, `GateSetTomographyCheckpoint` and `StandardGSTCheckpoint`, which are saved to disk over the course of an iterative fit in serialized json format. By default checkpoint files associated with a `GateSetTomographyCheckpoint` object are saved to a new directory located in whichever current working directory the protocol is being run from named 'gst_checkpoints'. A new file is written to disk after each iteration with default naming of the form `GateSetTomography_iteration_{i}.json` where i is the index of the completed GST iteration associated with that checkpoint. Similarly, for a `StandardGSTCheckpoint` object the checkpoints are by default saved to a directory named 'standard_gst_checkpoints' with default file names of the form `StandardGST_{mode}_iteration_{i}` where mode corresponds to the current parameterized fit or model test associated with that file (including checkpoint information for all previously completed modes prior to the currently running one) and i is the index of the completed iteration within that current mode.

Below we fit the data from the setup section one more time — serially, with a TP-constrained model, and with checkpointing enabled (as is the default).

```{code-cell} ipython3
target_model_TP = mp.target_model("full TP")
proto = pygsti.protocols.GateSetTomography(target_model_TP)
results_TP = proto.run(pdata, checkpoint_path='../../../example_files/gst_checkpoints/GateSetTomography')
```

Note that in the example above we have specified a value for an additional kwarg called `checkpoint_path`. This allows for overriding the default behavior for the save location and naming of checkpoint files. The expected format is `{path}/{name}` where path is the directory to save the checkpoint files to (that directory is created if it does not exist) and where name is the stem of the checkpoint file names `{name}_iteration_{i}.json`. Inspecting the contents of the directory we just specified, we can see that it is now populated by 6 new checkpoint files, one per iteration.

```{code-cell} ipython3
import os
sorted(os.listdir('../../../example_files/gst_checkpoints/'))
```

Suppose hypothetically that a GST fit had failed partway through iteration 4 and we wanted to restart from that point without redoing all of the previous iterations from scratch again. We'll call this warmstarting. We can do so by reading in the last completed iteration's serialized checkpoint object using the `read` class method of `GateSetTomographyCheckpoint` and passing that now loaded checkpoint object in for the `checkpoint` kwarg of `run`.

```{code-cell} ipython3
from pygsti.protocols import GateSetTomographyCheckpoint
gst_iter_3_checkpoint = GateSetTomographyCheckpoint.read('../../../example_files/gst_checkpoints/GateSetTomography_iteration_3.json')
results_TP_from_iter_3 = proto.run(pdata, checkpoint=gst_iter_3_checkpoint, checkpoint_path='../../../example_files/gst_checkpoints/GateSetTomography')
```

We can see from the output that we indeed started from iteration 4 (note the output log indexes from 1 instead of 0, so it prints "Iter 5 of 6"). Moreover we can see that we've indeed produced the same output as before without warmstarting, as we would expect/hope:

```{code-cell} ipython3
all(results_TP.estimates['GateSetTomography'].models['final iteration estimate'].to_vector() == \
results_TP_from_iter_3.estimates['GateSetTomography'].models['final iteration estimate'].to_vector())
```

The checkpoint object itself contains information that could be useful for diagnostics or debugging, including the current list of models associated each iterative fit, the last completed iteration it is associated with, and the list of circuits for the last completed iteration it is associated with.

Checkpointing with the `StandardGST` protocol works similarly:

```{code-cell} ipython3
:tags: [output_scroll]

proto_standard_gst = pygsti.protocols.StandardGST(modes=['full TP', 'CPTPLND', 'Target'], verbosity=4)
results_stdprac = proto_standard_gst.run(pdata, checkpoint_path='../../../example_files/standard_gst_checkpoints/StandardGST')
```

Except this time we have significantly more files saved, as during the course of the StandardGST protocol we're actually running three subprotocols:

```{code-cell} ipython3
sorted(os.listdir('../../../example_files/standard_gst_checkpoints/'))
```

Note that the StandardGST protocol runs the subprotocols in the order listed in the `modes` argument, and checkpoint objects labeled with a given model label additionally contain the checkpointing information for the final iterations of any preceding modes which have been completed. i.e. the CPTPLND checkpoint objects contain the information required for full TP. Likewise, checkpoints for Target contain the information required for the full TP and CPTPLND modes. As before, imagine that our fitting failed for whatever reason during iteration 5 of CPTPLND, we can warmstart the protocol by loading in the checkpoint object associated with iteration 4 as below:

```{code-cell} ipython3
from pygsti.protocols import StandardGSTCheckpoint
standard_gst_checkpoint = StandardGSTCheckpoint.read('../../../example_files/standard_gst_checkpoints/StandardGST_CPTPLND_iteration_4.json')
results_stdprac_warmstart = proto_standard_gst.run(pdata, checkpoint=standard_gst_checkpoint, checkpoint_path='../../../example_files/standard_gst_checkpoints/StandardGST')
```

Notice that we've indeed skipped past the previously completed full TP mode and jumped straight to the 6th iteration of the CPTPLND fit as expected. 

As for the GateSetTomographyCheckpoint object described above, the `StandardGSTCheckpoint` can often be useful to inspect as a debugging/diagnostic tool. `StandardGSTCheckpoints` are essentially structured as container object that hold a set of child `GateSetTomographyCheckpoint` and `ModelTestCheckpoint` (more on these in the [model testing tutorial](../analysis/ModelTesting)) objects for each of the modes being run (and potentially more types of child checkpoints in the future as we add additional functionality). These children can be accessed using the `children` attribute of a `StandardGSTCheckpoint` instance which is a dictionary with keys given by the mode names contained therein.

```{code-cell} ipython3
print(standard_gst_checkpoint.children['CPTPLND'])
```

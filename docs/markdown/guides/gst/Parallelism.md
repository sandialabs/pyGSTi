---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.19.4
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Running GST in parallel with MPI

pyGSTi's core GST routines are written to spread work across processors using MPI. Long fits are the reason you'd care: the circuit-by-circuit likelihood evaluations that dominate the runtime divide cleanly across ranks, and the per-rank memory footprint drops with them.

There are two ways to get there. `Protocol.run_mpi` launches the worker processes for you, waits for them, and hands back a results object, so you never leave the notebook. The lower-level route is to write a script that imports `mpi4py`, passes an `MPI.Comm` to `Protocol.run`, and launch it yourself under `mpiexec`. Use the first unless you need control over how the job is launched.

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
import shutil, subprocess

launcher = shutil.which('mpiexec') or shutil.which('mpirun')
banner = subprocess.run([launcher, '--version'], capture_output=True, text=True) if launcher else None
is_openmpi = banner is not None and 'Open MPI' in (banner.stdout + banner.stderr)
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
report.write_html('../../../example_files/mpi_example_brief', auto_open=False)
```

Open the [report](../../../example_files/mpi_example_brief/main.html).

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
import time
import pygsti

#get MPI comm
from mpi4py import MPI
comm = MPI.COMM_WORLD

print("Rank %d started" % comm.Get_rank())

#load in data
data = pygsti.io.read_data_from_dir("../../../example_files/mpi_gst_example")

#Specify a per-core memory limit (useful for larger GST calculations)
memLim = 2.1*(1024)**3  # 2.1 GB

#Perform TP-constrained GST
protocol = pygsti.protocols.StandardGST("full TP")
start = time.time()
results = protocol.run(data, memlimit=memLim, comm=comm)
end = time.time()

print("Rank %d finished in %.1fs" % (comm.Get_rank(), end-start))
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

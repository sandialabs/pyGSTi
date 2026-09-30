---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.19.5
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# "Automagic" parallel GST using MPI
This tutorial demonstrates how to compute GST estimates in parallel using MPI. 
This requires the `mpi4py` python package, which must be able to connect to an underlying MPI library (say, openmpi).

```{note}
**This notebook requires a working system MPI launcher and is expected to fail
in environments that do not have one.** The `run_mpi` call below shells out to
an MPI launcher (`mpiexec`/`mpirun`/`mpiexec.hydra`) that must be discoverable on
`PATH`. Installing the `mpi4py` Python package alone is *not* sufficient: it
provides the Python bindings but not the launcher binary or the underlying MPI
runtime.

When run under the notebook regression tests (`pytest --nbval-lax`) on a machine
without an MPI distribution, this notebook fails with
`FileNotFoundError: resolve_mpiexec: could not find an MPI launcher on PATH`.
This is a **known, accepted, environment-limited failure** and does not indicate
a bug in pyGSTi. To run this notebook successfully, install a system MPI
distribution (e.g. `openmpi-bin` on Debian/Ubuntu, or `module load openmpi` on
an HPC cluster) so that `mpiexec` is on `PATH`. pyGSTi's CI does exactly this via
its `before_install.sh` setup script.
```

We'll start by setting the stage.

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

In this demo we invoke `run_mpi` with two extra arguments for robustness across MPI setups and machines with few cores:

- `env={'FI_PROVIDER': 'sockets'}` can be needed on some MPI distributions. Try without it as well, and omit it unless you need it.
- `extra_mpi_args=['--oversubscribe']` lets the launcher start more ranks than the machine has cores. We request `num_ranks=3`, and many machines (including CI runners) have fewer than three cores; without this flag Open MPI refuses to launch and `run_mpi` raises a `CalledProcessError`. Note that `--oversubscribe` is an Open MPI flag — on MPICH or Intel MPI, omit it (they neither recognize nor require it).

```{code-cell} ipython3
protocol = StandardGST(verbosity=2)
results = protocol.run_mpi(pdata,
    num_ranks=3, mpiexec='auto', env={'FI_PROVIDER': 'sockets'},
    extra_mpi_args=['--oversubscribe']
)
```

```{code-cell} ipython3
from pygsti.report import construct_standard_report

report = construct_standard_report(
    results, title="MPI Example Report", verbosity=0
)
report.write_html('../../example_files/mpi_example_brief', auto_open=False)
```

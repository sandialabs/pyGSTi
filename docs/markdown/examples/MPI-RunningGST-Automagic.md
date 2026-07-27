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

# "Automagic" parallel GST using MPI
This tutorial demonstrates how to compute GST estimates in parallel using MPI. 
This requires the `mpi4py` python package, which must be able to connect to an underlying MPI library (say, openmpi).

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

```{code-cell} ipython3
protocol = StandardGST(verbosity=2)
results = protocol.run_mpi(pdata, num_ranks=3, mpiexec='auto')
```

```{code-cell} ipython3
from pygsti.report import construct_standard_report

report = construct_standard_report(
    results, title="MPI Example Report", verbosity=0
)
report.write_html('example_files/mpi_example_brief', auto_open=False)
```

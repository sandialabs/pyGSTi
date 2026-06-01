---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.17.3
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Model Fitting
The purpose of this tutorial is to demonstrate how to compute GST estimates in parallel (using multiple CPUs or "processors").  The core PyGSTi computational routines are written to take advantage of multiple processors via the MPI communication framework, and so one must have a version of MPI and the `mpi4py` python package installed in order use run pyGSTi calculations in parallel.  

Since `mpi4py` doesn't play nicely with Jupyter notebooks, this tutorial is a bit more clunky than the others.  In it, we will create a standalone Python script that imports `mpi4py` and execute it.

We will use as an example the same "standard" single-qubit model of the first tutorial.  We'll first create a dataset, and then a script to be run in parallel which loads the data.  The creation of a simulated data is performed in the same way as the first tutorial.   Since *random* numbers are generated and used as simulated counts within the call to `generate_fake_data`, it is important that this is *not* done in a parallel environment, or different CPUs may get different data sets.  (This isn't an issue in the typical situation when the data is obtained experimentally.)

```{code-cell} ipython3
#Import pyGSTi and the "stardard 1-qubit quantities for a model with X(pi/2), Y(pi/2), and idle gates"
import pygsti
from pygsti.modelpacks import smq1Q_XYI

#Create experiment design
exp_design = smq1Q_XYI.create_gst_experiment_design(max_max_length=32)
pygsti.io.write_empty_protocol_data("../../example_files/mpi_gst_example", exp_design, clobber_ok=True)

#Simulate taking data
mdl_datagen  = smq1Q_XYI.target_model().depolarize(op_noise=0.1, spam_noise=0.001)
pygsti.io.fill_in_empty_dataset_with_fake_data("../../example_files/mpi_gst_example/data/dataset.txt",
                                               mdl_datagen, num_samples=1000, seed=2020)
```

Next, we'll write a Python script that will load in the just-created `DataSet`, run GST on it, and write the output to a file.  The only major difference between the contents of this script and previous examples is that the script imports `mpi4py` and passes a MPI comm object (`comm`) to the `run_long_sequence_gst` function.  Since parallel computing is best used for computationaly intensive GST calculations, we also demonstrate how to set a per-processor memory limit to tell pyGSTi to partition its computations so as to not exceed this memory usage.  Lastly, note the use of the `gaugeOptParams` argument of `run_long_sequence_gst`, which can be used to weight different model members differently during gauge optimization.

```{code-cell} ipython3
mpiScript = """
import time
import pygsti

#get MPI comm
from mpi4py import MPI
comm = MPI.COMM_WORLD

print("Rank %d started" % comm.Get_rank())

#load in data
data = pygsti.io.read_data_from_dir("../../example_files/mpi_gst_example")

#Specify a per-core memory limit (useful for larger GST calculations)
memLim = 2.1*(1024)**3  # 2.1 GB

#Perform TP-constrained GST
protocol = pygsti.protocols.StandardGST("full TP")
start = time.time()
results = protocol.run(data, memlimit=memLim, comm=comm)
end = time.time()

print("Rank %d finished in %.1fs" % (comm.Get_rank(), end-start))
if comm.Get_rank() == 0:
    results.write()  #write results (within same diretory as data was loaded from)
    
results=None # needed to free shared memory before garbage collection is torn down
"""
with open("../../example_files/mpi_example_script.py","w") as f:
    f.write(mpiScript)
```

Next, we run the script with 3 processors using `mpiexec`.  The `mpiexec` executable should have been installed with your MPI distribution -- if it doesn't exist, try replacing `mpiexec` with `mpirun`.

```{code-cell} bash
:tags: [nbval-skip]

mpiexec -n 3 python3 "../../example_files/mpi_example_script.py"
```

Notice in the above that output within `StandardGST.run` is not duplicated (only the first processor outputs to stdout) so that the output looks identical to running on a single processor.  Finally, we just need to read the saved `ModelEstimateResults` object from file and proceed with any post-processing analysis.  In this case, we'll just create a  report.

```{code-cell} ipython3
:tags: [nbval-skip]

results = pygsti.io.read_results_from_dir("../../example_files/mpi_gst_example", name="StandardGST")
pygsti.report.construct_standard_report(
    results, title="MPI Example Report", verbosity=2
).write_html('../../example_files/mpi_example_brief', auto_open=True)
```

Open the [report](../../example_files/mpi_example_brief/main.html).



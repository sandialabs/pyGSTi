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

# 2Q-GST Fitting

```{attention} Is this redundant with the Overview and MPI-RunningGST tutorials?```

This example gives an overview of the typical steps used to perform an end-to-end (i.e. experimental-data-to-report) Gate Set Tomography analysis on a 2-qubit system.  The steps are very similar to the single-qubit case described in the tutorials, but we thought 2Q-GST is an important enough topic to deserve a separate example.

```{code-cell} ipython3
import pygsti
```

### Step 1: Construct the desired 2-qubit model
Since the purpose of this example is to show how to *run* 2Q-GST, we'll just use a built-in "standard" 2-qubit model.  (Another example covers how to create a custom 2-qubit model.)

```{code-cell} ipython3
from pygsti.modelpacks import smq2Q_XY
target_model = smq2Q_XY.target_model('full TP')
```

### Step 2: create an experiment design
An experiment design is a object containing all the information needed to perform and later interpret the data from a set of circuits.  In the case of GST, lists of fiducial and germ sub-circuits are the building blocks of the circuits performed in the experiment. Typically, these lists are either provided by pyGSTi because you're using a "standard" model (as we are here), or computed using the "fiducial selection" and "germ selection" algorithms which are a part of pyGSTi and covered in the tutorials.  As an additional input, we'll need a list of lengths indicating the maximum length circuits to use on each successive GST iteration.  Since 2Q-GST can take a while, only use short sequences (`max_max_lengths=1`) with fiducial-pair reduction (`fpr=True`) to demonstrate 2Q-GST more quickly (because we know you have important stuff to do).

```{code-cell} ipython3
exp_design = smq2Q_XY.create_gst_experiment_design(max_max_length=2, fpr=True)
```

### Step 3: Data generation
Now that we have an experment design we can generate the list of experiments needed to run GST, just like in the 1-qubit case.

```{code-cell} ipython3
#Create an empty dataset file at example_files/My2QExample/data/dataset.txt, which stores the
# list of experiments and zerod-out columns where data should be inserted.
pygsti.io.write_empty_protocol_data("example_files/My2QExample", exp_design, clobber_ok=True)
```

```{code-cell} ipython3
#Generate some "fake" (simulated) data based on a depolarized version of the target model.  In actual
# situations, you'd fill in dataset.txt with real data.
mdl_datagen = target_model.depolarize(op_noise=0.1, spam_noise=0.01)
pygsti.io.fill_in_empty_dataset_with_fake_data("example_files/My2QExample/data/dataset.txt",
                                               mdl_datagen, num_samples=1000, seed=2020)

# ---- NOTE: you can stop and restart the python session at this point; everything you need is saved to disk ---- 

#Load in the "data object" which packages together the dataset and experiment design
data = pygsti.io.read_data_from_dir("example_files/My2QExample")
```

### Step 4: Run GST
Just like for 1-qubit GST, we use the `StandardGST` protocol to compute the GST estimates.  Usually for two qubits this could take a long time (hours on a single cpu) based on the number of operation sequences used, and running on multiple processors is a good idea (see the MPI example).  Here, we set the tolerance to a high value ($10^{-3}$) so that it only takes around 30 minutes to run.

Some notes about the options/arguments here that are particularly relevant to 2-qubit GST:
  - `memlimit` gives an estimate of how much memory is available to use on your system (in bytes).  This is currently *not* a hard limit, and pyGSTi may require slightly more memory than this "limit".  So you'll need to be conservative in the value you place here: if your machine has 10GB of RAM, set this to 6 or 8 GB initially and increase it as you see how much memory is actually used using a separate OS performance monitor tool.  If you're running on multiple processors, this should be the memory available *per processor*.
  - `verbosity` tells the routine how much detail to print to stdout.  If you don't mind waiting a while without getting any output, you can leave this at its default value (2).  If you can't standing wondering whether GST is still running or has locked up, set this to 3.

```{code-cell} ipython3
:tags: [nbval-skip]

import time
start = time.time()
protocol = pygsti.protocols.StandardGST("CPTPLND", optimizer={'tol': 1e-3}, verbosity=2)
results = protocol.run(data, memlimit=5*(1024)**3)
end = time.time()
print("Total time=%f hours" % ((end - start) / 3600.0))
```

### Step 5: Create report(s) using the returned `ModelEstimateResults` object
The `ModelEstimateResults` object returned from `run` can be used to generate a "general" HTML report, just as in the 1-qubit case:

```{code-cell} ipython3
:tags: [nbval-skip]

report = pygsti.report.construct_standard_report(
    results, title="Example 2Q-GST Report", verbosity=2)
report.write_html('example_files/easy_2q_report', verbosity=2)
```

Now open [example_files/easy_2q_report/main.html](example_files/easy_2q_report/main.html) to see the results.  You've run 2-qubit GST!

You can save the `ModelEstimateResults` object to the same directory as the data and experiment design:

```{code-cell} ipython3
:tags: [nbval-skip]

results.write()
```

```{code-cell} ipython3

```

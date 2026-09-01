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

# 2Q-GST fitting

This example performs an end-to-end (i.e. experimental-data-to-report) gate set tomography analysis on a 2-qubit system.  The steps are the same as in the single-qubit [GST overview tutorial](../../start/FirstGST); this page focuses on what changes when you add a second qubit.

```{code-cell} ipython3
import pygsti
```

## Step 1: construct the desired 2-qubit model
Since the purpose of this example is to show how to *run* 2Q-GST, we'll just use a built-in "standard" 2-qubit model.  (The [explicit models tutorial](../models/Models) covers creating a custom 2-qubit model.)

```{code-cell} ipython3
from pygsti.modelpacks import smq2Q_XY
target_model = smq2Q_XY.target_model('full TP')
```

## Step 2: create an experiment design
Experiment designs, and the fiducial and germ circuits GST builds them from, are covered in the [GST circuits tutorial](GSTCircuits).  What changes with a second qubit is scale: a full-power design for this model (`max_max_length=32`) contains just over 8,000 circuits, so the expensive part of 2Q-GST is collecting the data, not fitting it.  Fiducial-pair reduction (`fpr=True`) is the main tool for shrinking a design; at the short `max_max_length=4` used here it cuts 2,295 circuits down to 837.

```{code-cell} ipython3
exp_design = smq2Q_XY.create_gst_experiment_design(max_max_length=4, fpr=True)
```

## Step 3: generate data
Now that we have an experiment design we can generate the list of experiments needed to run GST, just like in the 1-qubit case.

```{code-cell} ipython3
#Create an empty dataset file at ../../../example_files/My2QExample/data/dataset.txt, which stores the
# list of experiments and zerod-out columns where data should be inserted.
pygsti.io.write_empty_protocol_data("../../../example_files/My2QExample", exp_design, clobber_ok=True)
```

```{code-cell} ipython3
#Generate some "fake" (simulated) data based on a depolarized version of the target model.  In actual
# situations, you'd fill in dataset.txt with real data.
mdl_datagen = target_model.depolarize(op_noise=0.1, spam_noise=0.01)
pygsti.io.fill_in_empty_dataset_with_fake_data("../../../example_files/My2QExample/data/dataset.txt",
                                               mdl_datagen, num_samples=1000, seed=2020)

# ---- NOTE: you can stop and restart the python session at this point; everything you need is saved to disk ---- 

#Load in the "data object" which packages together the dataset and experiment design
data = pygsti.io.read_data_from_dir("../../../example_files/My2QExample")
```

## Step 4: run GST
Just like for 1-qubit GST, we use the `StandardGST` protocol to compute the GST estimates.  We loosen the optimizer's convergence tolerance (`optimizer={'tol': 1e-3}`) to keep this demonstration quick — the fit below finishes in well under a minute, versus several minutes at the default tolerance — but you should drop the `optimizer` argument for a real analysis.  Runtime materially increases with sequence length; if your fits are taking too long, pyGSTi supports MPI-acceleration — see the [Parallelism tutorial](Parallelism).

Some notes about the options/arguments here that are particularly relevant to 2-qubit GST:
  - `memlimit` gives an estimate of how much memory is available to use on your system (in bytes).  This is currently *not* a hard limit, and pyGSTi may require slightly more memory than this "limit".  So you'll need to be conservative in the value you place here: if your machine has 10GB of RAM, set this to 6 or 8 GB initially. Then, use some standalone OS performance monitor tool to see how much memory is actually used when you run.  If you're running on multiple processors, this should be the memory available *per processor*.
  - `verbosity` tells the routine how much detail to print to stdout. The default value is 2. Increase this value if you're worried that pyGSTi is stuck and you want evidence to the contrary.

```{code-cell} ipython3
:tags: [nbval-ignore-output]

import time
start = time.time()
protocol = pygsti.protocols.StandardGST("CPTPLND", optimizer={'tol': 1e-3}, verbosity=2)
results = protocol.run(data, memlimit=5*(1024)**3)
end = time.time()
print("Total time=%.1f seconds" % (end - start))
```

## Step 5: create reports
The returned `ModelEstimateResults` object (see the [Results tutorial](../analysis/Results)) generates an HTML report just as in the 1-qubit case.  Building and writing the report costs about as much as the fit itself — both finish in well under a minute here — and prints little while it runs:

```{code-cell} ipython3
report = pygsti.report.construct_standard_report(
    results, title="Example 2Q-GST Report", verbosity=2)
report.write_html('../../../example_files/easy_2q_report', connected=True, verbosity=2)
```

The report is served with these docs: <a href="../../../reports/easy_2q_report/main.html">easy 2Q report</a>.  You've run 2-qubit GST!

You can save the `ModelEstimateResults` object to the same directory as the data and experiment design:

```{code-cell} ipython3
results.write()
```

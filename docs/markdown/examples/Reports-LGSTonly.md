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

# Create An LGST-Only Report
This example shows how you can create a HTML report from just the results of running *linear GST* (LGST).  This can be useful when you want to get a rough estimate of your gates quickly, as LGST is takes substantially less data and computation time compared with long-sequence GST.  This example is modeled after Tutorial 0.

```{code-cell} ipython3
#Import the pygsti module (always do this) and the standard XYI model
import pygsti
from pygsti.modelpacks import smq1Q_XYI

#Get experiment design (for now, just max_max_length=1 GST sequences)
exp_design = smq1Q_XYI.create_gst_experiment_design(max_max_length=1)
pygsti.io.write_empty_protocol_data("../../example_files/lgst_only_example", exp_design, clobber_ok=True)
print("Only %d sequences are required!" % len(exp_design.all_circuits_needing_data))

#Simulate taking the data (here you'd really fill in dataset.txt with actual data)
mdl_datagen = smq1Q_XYI.target_model().depolarize(op_noise=0.1, spam_noise=0.001)
pygsti.io.fill_in_empty_dataset_with_fake_data("../../example_files/lgst_only_example/data/dataset.txt",
                                               mdl_datagen, num_samples=1000, seed=2020)

#load in the data
data = pygsti.io.read_data_from_dir("../../example_files/lgst_only_example")
```

```{code-cell} ipython3
#Run LGST and create a report
# You can also eliminate gauge optimization step by setting gaugeOptParams=False
results = pygsti.protocols.LGST(smq1Q_XYI.target_model()).run(data)
```

```{code-cell} ipython3
pygsti.report.construct_standard_report(
    results, title="LGST-only Example Report", verbosity=2
).write_html('../../example_files/LGSTonlyReport', verbosity=2)
```

Click to open the file [../../example_files/LGSTonlyReport/main.html](../../example_files/LGSTonlyReport/main.html) in your browser to view the report.

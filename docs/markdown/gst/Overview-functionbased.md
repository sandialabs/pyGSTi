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

# Gate set tomography (function-based)

This tutorial shows you how to run gate set tomography using the older-style function-centric API.  This can be useful in certain scenarios, but most of the time it's easier to use the newer, object-oriented approach outlined in the main [GST overview tutorial](Overview).  For more details and options for running GST, see the [GST circuits tutorial](CircuitConstruction) and the [tutorial covering the different pyGSTi functions for running GST](Driverfunctions).

To run GST, we need three inputs:
1. a "**target model**" which describes the desired, or ideal, operations we want our experimental hardware to perform.  In the example below, we use one of pyGSTi's build-in "model packs" (see the [tutorial on model packs](../objects/ModelPacks)) - which acts on a single qubit with the following operations:
    - two gates: $\pi/2$ rotations around the $x$- and $y$-axes.
    - a single state preparation in the $|0\rangle$ state.
    - a 2-outcome measurement with the label "0" associated with measuring $|0\rangle$ and "1" with measuring $|1\rangle$.
    
2. a list of circuits tailored to the target model; essentially a list of what experiments we need to run.  Using a standard model makes things especially straightforward here, since the building blocks, called *germ* and *fiducial* circuits, needed to make good GST circuits have already been computed (see the [tutorial on GST circuits](CircuitConstruction)).

3. data, in the form of experimental outcome counts, for each of the required sequences.  In this example we'll generate "fake" or "simulated" data from a depolarized version of our ideal model.  For more information about `DataSet` objects, see the [tutorial on DataSets](../objects/DataSet).

```{code-cell} ipython3
#Import the pygsti module (always do this) and the XYI model pack
import pygsti
from pygsti.modelpacks import smq1Q_XY

# 1) get the target Model
target_model = smq1Q_XY.target_model()

# 2) get the building blocks needed to specify which operation sequences are needed
prep_fiducials, meas_fiducials = smq1Q_XY.prep_fiducials(), smq1Q_XY.meas_fiducials()
germs = smq1Q_XY.germs()
maxLengths = [1,2,4,8,16] # roughly gives the length of the sequences used by GST

# 3) generate "fake" data from a depolarized version of target_model
mdl_datagen = target_model.depolarize(op_noise=0.01, spam_noise=0.001)
listOfExperiments = pygsti.circuits.create_lsgst_circuits(
    target_model, prep_fiducials, meas_fiducials, germs, maxLengths)
ds = pygsti.data.simulate_data(mdl_datagen, listOfExperiments, num_samples=1000,
                                            sample_error="binomial", seed=1234)

#Note: from listOfExperiments we can also create an empty dataset file
# which has columns of zeros where actual data should go. 
pygsti.io.write_empty_dataset("../../tutorial_files/GettingStartedDataTemplate.txt", listOfExperiments,
                              "## Columns = 0 count, 1 count")
# After replacing the zeros with actual data, the data set can be 
# loaded back into pyGSTi using the line below and used in the rest
# of this tutorial. 
#ds = pygsti.io.load_dataset("tutorial_files/GettingStartedDataTemplate.txt")
```

Now that we have all of the inputs, we can run GST in a standard way using the `run_stdpractice_gst` function.  For more information about this and related functions, see the [GST methods tutorial](Driverfunctions).  This returns a `pygsti.report.Results` object (see the [Results tutorial](../objects/Results)), from which we can generate a report giving us a summary of the analysis.

```{code-cell} ipython3
#Run GST and create a report
results = pygsti.run_stdpractice_gst(ds, target_model, prep_fiducials, meas_fiducials, 
                                    germs, maxLengths, verbosity=3)

pygsti.report.construct_standard_report(
    results, title="GST Overview Tutorial Example Report", verbosity=2
).write_html("../../tutorial_files/gettingStartedReport", verbosity=2)
```

You can now open the file [../../tutorial_files/gettingStartedReport/main.html](../../tutorial_files/gettingStartedReport/main.html) in your browser (Firefox works best) to view the report.  **That's it!  You've just run GST!** 

In the cell above, `results` is a `Results` object, which is used to generate a HTML report.  For more information see the [Results object tutorial](../objects/Results) and [report generation tutorial](../reporting/ReportGeneration).

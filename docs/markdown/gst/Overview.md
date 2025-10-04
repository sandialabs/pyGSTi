---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.17.3
kernelspec:
  display_name: random_pygsti_debugging
  language: python
  name: random_pygsti_debugging
---

# Gate Set Tomography

The `pygsti` package provides multiple levels of abstraction over the core Gate Set Tomography (GST) algorithms.  This tutorial will show you how to run Gate Set Tomography on some simulated (generated) data, hopefully giving you an overall sense of what it takes (and how easy it is!) to run GST.  For more details and options for running GST, see the [GST circuits tutorial](GSTCircuitConstruction) and the [tutorial covering the different protocols for running GST](GST-Protocols).

There are three basic steps to running protocols in pyGSTi:

## Step 1: create an experiment design
The first step is creating an object that specifies what data (from the quantum processor) will be needed to perform GST, and how it should be taken.  This is called an "experiment design" in pyGSTi.

To run GST, we need the following three inputs:
1. a "**target model**" which describes the desired, or ideal, operations we want our experimental hardware to perform.  In the example below, we use the target model from one of pyGSTi's build-in "model packs" (see the [tutorial on model packs](../objects/ModelPacks)) - which acts on a single qubit with the following operations:
    - two gates: $\pi/2$ rotations around the $x$- and $y$-axes.
    - a single state preparation in the $|0\rangle$ state.
    - a 2-outcome measurement with the label "0" associated with measuring $|0\rangle$ and "1" with measuring $|1\rangle$.
    
2. a list of circuits tailored to the target model; essentially a list of what experiments we need to run.  Using a standard model makes things especially straightforward here, since the building blocks, called *germ* and *fiducial* circuits, needed to make good GST circuits have already been computed (see the [tutorial on GST circuits](GSTCircuitConstruction)).  In the example below, the model pack also provides the necessary germ and fiducial lists, so that all that is needed is a list of "maximum lengths" describing how long (deep) the circuits should be.

3. data, in the form of experimental outcome counts, for each of the required sequences.  In this example we'll generate "fake" or "simulated" data from a depolarized version of our ideal model.  For more information about `DataSet` objects, see the [tutorial on DataSets](../objects/DataSet).

The first two inputs form an "experiment design", as they describe the experiment that must be performed on a quantum processor (usually running some prescribed set of circuits) in order to run the GST protocol.  The third input - the data counts - is packaged with the experiment design to create a `ProtocolData`, or "data" object.  As we will see later, a data object serves as the input to the GST protocol.

**The cell below creates an experiment design for running standard GST on the 1-qubit quantum process described by the gates above using circuits whose depth is at most 32.**

```{code-cell} ipython3
import pygsti
from pygsti.modelpacks import smq1Q_XY

#Step 1: create an "experiment design" for doing GST on the std1Q_XYI gate set
target_model = smq1Q_XY.target_model()      # a Model object
prep_fiducials = smq1Q_XY.prep_fiducials()  # a list of Circuit objects
meas_fiducials = smq1Q_XY.meas_fiducials()  # a list of Circuit objects
germs = smq1Q_XY.germs()                    # a list of Circuit objects
maxLengths = [1,2,4,8,16]
exp_design = pygsti.protocols.StandardGSTDesign(target_model, prep_fiducials, meas_fiducials,
                                                germs, maxLengths)
```

**Pro tip:** the contents of the cell above (except the imports) could be replaced by the single line:

```exp_design = smq1Q_XYI.create_gst_experiment_design(max_max_length=32)```


## Step 2: collect data as specified by the experiment design
Next, we just follow the instructions in the experiment design to collect data from the quantum processor (or the portion of the processor we're characterizing).  In this example, we'll generate the data using a depolarizing noise model since we don't have a real quantum processor lying around.  The call to `simulate_taking_data` should be replaced with the user filling out the empty "template" data set file with real data.  Note also that we set `clobber_ok=True`; this is so the tutorial can be run multiple times without having to manually remove the dataset.txt file - we recommend you leave this set to False (the default) when using it in your own scripts.

```{code-cell} ipython3
def simulate_taking_data(data_template_filename):
    """Simulate taking 1-qubit data and filling the results into a template dataset.txt file"""
    datagen_model = smq1Q_XY.target_model().depolarize(op_noise=0.01, spam_noise=0.001)
    pygsti.io.fill_in_empty_dataset_with_fake_data(data_template_filename, datagen_model, num_samples=1000, seed=1234)
```

```{code-cell} ipython3
pygsti.io.write_empty_protocol_data('../tutorial_files/test_gst_dir', exp_design, clobber_ok=True)

# -- fill in the dataset file in tutorial_files/test_gst_dir/data/dataset.txt --
simulate_taking_data("../tutorial_files/test_gst_dir/data/dataset.txt")  # REPLACE with actual data-taking

data = pygsti.io.read_data_from_dir('../tutorial_files/test_gst_dir')
```

## Step 3: Run the GST protocol and create a report
Now we just instantiate a `StandardGST` protocol and `.run` it on our data object.  This returns a results object that can be used to create a report.

```{code-cell} ipython3
#run the GST protocol and create a report 
gst_protocol = pygsti.protocols.StandardGST(['full TP','CPTPLND','Target'])
results = gst_protocol.run(data)

report = pygsti.report.construct_standard_report(
    results, title="GST Overview Tutorial Example Report", verbosity=2)
report.write_html("../tutorial_files/gettingStartedReport", verbosity=2)
```

You can now open the file [../tutorial_files/gettingStartedReport/main.html](../tutorial_files/gettingStartedReport/main.html) in your browser (Firefox works best) to view the report.  **That's it!  You've just run GST!** 

In the cell above, `results` is a `ModelEstimateResults` object, which is used to generate a HTML report.  For more information see the [Results object tutorial](../objects/Results) and [report generation tutorial](../reporting/ReportGeneration).


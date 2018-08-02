
# coding: utf-8

# # Getting started with pyGSTi and Gate Set Tomography

# The `pygsti` package provides multiple levels of abstraction over the core Gate Set Tomography (GST) algorithms.  This initial tutorial will show you how to run Gate Set Tomography on some simulated (generated) data, hopefully giving you an overall sense of what it takes (and how easy it is!) to run GST.  Subsequent tutorials will delve into the details of `pygsti` objects and algorithms, and how to use them in detail.
#
# To run GST, we need three inputs:
# 1. a "**target gate set**" which describes the desired, or ideal, operations we want our experimental hardware to perform.  In the example below, we use one of pyGSTi's "standard" gate sets - the on acting on a single qubit with the following operations:
#     - three gates: the identity, and $\pi/2$ rotations around the $x$- and $y$-axes.
#     - a single state preparation in the $|0\rangle$ state.
#     - a 2-outcome measurement with the label "0" associated with measuring $|0\rangle$ and "1" with measuring $|1\rangle$.
#
# 2. a list of GST sequences corresponding to the target gate set; essentially a list of what experiments (= gate sequences) we need to run.  Using a standard gate set makes things especially straightforward here, since the building blocks, called *germ* and *fiducial* sequences needed to make good GST sequences have already been computed.
#
# 3. data, in the form of experimental outcome counts, for each of the required sequences.  In this example we'll generate "fake" or "simulated" data from a depolarized version of our ideal gate set.
#

# In[3]:


#Make print statements compatible with Python 2 and 3
from __future__ import print_function

#Import the pygsti module (always do this) and the standard XYI gate set
import pygsti
from pygsti.construction import std1Q_XYI

# 1) get the target GateSet
gs_target = std1Q_XYI.gs_target

# 2) get the building blocks needed to specify which gate sequences are needed
prep_fiducials, meas_fiducials = std1Q_XYI.prepStrs, std1Q_XYI.effectStrs
germs = std1Q_XYI.germs
maxLengths = [1,2,4,8,16,32] # roughly gives the length of the sequences used by GST

# 3) generate "fake" data from a depolarized version of gs_target
gs_datagen = gs_target.depolarize(gate_noise=0.1, spam_noise=0.001)
listOfExperiments = pygsti.construction.make_lsgst_experiment_list(
    gs_target, prep_fiducials, meas_fiducials, germs, maxLengths)
ds = pygsti.construction.generate_fake_data(gs_datagen, listOfExperiments, nSamples=1000,
                                            sampleError="binomial", seed=1234)

#Note: from listOfExperiments we can also create an empty dataset file
# which has columns of zeros where actual data should go.
pygsti.io.write_empty_dataset("tutorial_files/GettingStartedDataTemplate.txt", listOfExperiments,
                              "## Columns = 1 count, count total")
# After replacing the zeros with actual data, the data set can be
# loaded back into pyGSTi using the line below and used in the rest
# of this tutorial.
#ds = pygsti.io.load_dataset("tutorial_files/GettingStartedDataTemplate.txt")


# Now that we have all of the inputs, we can run GST in a standard way using the `do_stdpractice_gst` high-level driver function.  This returns a `pygsti.report.Results` object, from which we can generate a report giving us a summary of the analysis.

# In[4]:


#Run GST and create a report
results = pygsti.do_stdpractice_gst(ds, gs_target, prep_fiducials, meas_fiducials, germs, maxLengths)
pygsti.report.create_standard_report(results, filename="tutorial_files/gettingStartedReport",
                                    title="Tutorial0 Example Report", verbosity=2)


# You can now open the file [tutorial_files/gettingStartedReport/main.html](tutorial_files/gettingStartedReport/main.html) in your browser to view the report.  **That's it!  You've just run GST!**
#
# The other tutorials in this directory will explain how to use the various objects and algorithms that comprise pyGSTi. These **tutorial notebooks are meant to be fairly pedagogical** and include details about the inner workings of and design choices within pyGSTi.  In contrast, the **"FAQ" directory contains notebooks with attempt to address specific questions as quickly and directly as possible, with little or no explanation of related topics or broader context**.

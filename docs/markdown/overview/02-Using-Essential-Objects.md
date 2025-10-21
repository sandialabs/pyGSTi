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

# Using Essential Objects
Now that we've covered what [circuits, models, and data sets](01-Essential-Objects) are, let's see what we can do with them!  This tutorial is intended to be an overview of the things that pyGSTi is able to do, with links to more detailed explanations and demonstrations as is appropriate.  We begin with the simpler applications and proceed to the more complex ones.  Here's a table of contents to give you a sense of what's here and so you can skip around if you'd like.  Each of the sections here can more-or-less stand on its own.

We'll begin by setting up a `Workspace` so we can display pretty interactive figures inline (see the [intro to Workspaces tutorial](../reporting/Workspace) for more details).

```{code-cell} ipython3
import pygsti
import numpy as np
ws = pygsti.report.Workspace()
ws.init_notebook_mode(autodisplay=True)
```

## Computing circuit outcome probabilities
One of the simplest uses of pyGSTi is to construct a `Model` and use it to compute the outcome probabilities of one or more `Circuit` objects.  This is generally accomplished using the `.probabilities` method of a `Model` object as shown below (this was also demonstrated in the [essential objects tutorial](01-Essential-Objects)).  The real work  is in constructing the `Circuit` and `Model` objects, which is covered in more detail in the [circuits tutorial](../objects/Circuit) and in the [explicit-model](../objects/ExplicitModel) (best for 1-2 qubits) and [implicit-model](../objects/ImplicitModel) (best for 3+ qubits) tutorials.  For more information on circuit simulation, see the [circuit simulation tutorial](../simulation/CircuitSimulation).

```{code-cell} ipython3
pspec = pygsti.processors.QubitProcessorSpec(num_qubits=2, gate_names=['Gx', 'Gy', 'Gcnot'],
                                             availability={'Gx': [(0,), (1,)],
                                                           'Gy': [(0,), (1,)],
                                                           'Gcnot': [(0,1)]})
mdl = pygsti.models.create_explicit_model(pspec) 
c = pygsti.circuits.Circuit([('Gx',0),('Gcnot',0,1),('Gy',1)] , line_labels=[0,1])
print("mdl will simulate probabilities using a '%s' forward simulator." % str(mdl.sim))
mdl.probabilities(c) # Compute the outcome probabilities of circuit `c`
```

## Simulating observed data based on a model
Only slightly more complex than computing circuit-outcome probabilities is generating simulated data (outcome *counts* rather than probabilities).  This is performed by the `generate_fake_data` function, which just samples the circuit-outcome probability distribution.  You supply a list of `Circuits`, a number of samples, and often times a seed to initialize the random sampler.  This is an easy way to create a `DataSet` to test other pyGSTi functions or to use independently.  

The default behavior is to sample the multinomial distribution associated with the given outcome probabilities and number of samples.  It's possible to turn off finite-sample error altogether and make the data-set counts *exactly equal* the probability values multiplied by the number of samples by setting `sample_error='none'`, as demonstrated below.

```{code-cell} ipython3
circuit_list = pygsti.circuits.to_circuits([ (), 
                                             (('Gx',0),),
                                             (('Gx',0),('Gy',1)),
                                             (('Gx',0),)*4,
                                             (('Gx',0),('Gcnot',0,1)) ], line_labels=(0,1))
ds_fake = pygsti.data.simulate_data(mdl, circuit_list, num_samples=100,
                                                 sample_error='multinomial', seed=1234)
print("Normal:")
print(ds_fake)

ds_nosampleerr = pygsti.data.simulate_data(mdl, circuit_list, num_samples=100,
                                                 sample_error='none')
print("Without any sample error:")
print(ds_nosampleerr)
```

## Testing how well a model describes a set of data
The above section showed how the circuit-outcome probabilities computed by a `Model` object can be used to generate data.  We can also compare these probabilities with the outcome counts in an existing `DataSet`, that is, ask the question: "For each circuit, how well do the frequencies of the outcomes (in the data) align with the probabilities predicted by the model?".  There are several common statistics for this purpose; the two used most often in pyGSTi are the $\chi^2$ and log-likelihood ($\log\mathcal{L}$) statistics.  If you're not sure what these are, the Methods section of [this paper](https://www.nature.com/articles/ncomms14485) provides some details and here are a few practical considerations:
- the larger $\log\mathcal{L}$ is, and the smaller $\chi^2$ is, the better the model agrees with the data.
- the value of $\log\mathcal{L}$ doesn't mean anything in isolation - only when compared to other $\log\mathcal{L}$ values.
- one can compute the $\log\mathcal{L}$ of a "maximal model" that agrees with the data exactly.  We call this value $\log\mathcal{L}_{max}$.
- in the limit of many samples, $\chi^2 \approx 2(\log\mathcal{L}_{max}-\log\mathcal{L})$, and we denote the latter $2\Delta\log\mathcal{L}$.  
- let $k$ be the the number of independent degrees of freedom in the data (e.g. each row of 4 (00, 01, 10, and 11) counts in the example below contributes $3$ degrees of freedom because the counts are constrained to add to 100, so $k=5*3=15$). If the model is "valid" (i.e. it *could* have generated the data) then $2\Delta\log\mathcal{L}$ should have come from a $\chi^2_k$ distribution, i.e. it has expectation value $k$ and standard deviation $\sqrt{2k}$.

Here's how we compute the $\chi^2$ and $2\Delta\log\mathcal{L}$ between some data and a model:

```{code-cell} ipython3
import numpy as np
dataset_txt = \
"""## Columns = 00 count, 01 count, 10 count, 11 count
{}@(0,1)            100   0   0   0
Gx:0@(0,1)           55   5  40   0
Gx:0Gy:1@(0,1)       20  27  23  30
Gx:0^4@(0,1)         85   3  10   2
Gx:0Gcnot:0:1@(0,1)  45   1   4  50
"""
with open("../../tutorial_files/Example_Short_Dataset.txt","w") as f:
    f.write(dataset_txt)
ds = pygsti.io.read_dataset("../../tutorial_files/Example_Short_Dataset.txt")

def compare(prefix, model, dataset):
    chi2 = pygsti.tools.chi2(model, dataset)
    logl = pygsti.tools.logl(model, dataset, min_prob_clip=1e-16, radius=1e-16)  # technical note: need these regularization args ~ 0 because we compare with noise-free data
    max_logl = pygsti.tools.logl_max(model, dataset)
    k = dataset.degrees_of_freedom()
    Nsigma = (2*(max_logl-logl) - k)/np.sqrt(2*k)
    print(prefix, "\n    chi^2 = ",chi2,"\n    2DeltaLogL = ", 2*(max_logl-logl),
          "\n    #std-deviations away from expected (%g) = " % k,Nsigma,"\n")
    
print("\nModel compared with:")
compare("1. Hand-chosen data (doesn't agree): ",mdl,ds)
compare("2. Model-generated data (agrees): ",mdl,ds_fake)
compare("3. Model-generated data w/no sample err (agrees *exactly*): ",mdl,ds_nosampleerr)
```

We can also look at these values on a per-circuit basis:

```{code-cell} ipython3
logl_percircuit = pygsti.tools.logl_per_circuit(mdl, ds)
max_logl_percircuit = pygsti.tools.logl_max_per_circuit(mdl, ds)
print("2DeltaLogL per circuit = ", 2*(max_logl_percircuit - logl_percircuit))

#ws.ColorBoxPlot('logl', pygsti.obj.LsGermsSerialStructure([0],)) TODO
```

It's possible to display model testing results within figure and HTML reports too.  For more information on model testing, especially alongside GST, see the [tutorial on model testing](../utilities/ModelTesting)(using protocol objects) and the [functions for model testing](../utilities/ModelTesting-functions).

## Randomized Benchmarking (RB)
PyGSTi is able to perform two types of randomized benchmarking (RB).  First, there is the [standard Clifford-circuit-based RB](http://journals.aps.org/prl/abstract/10.1103/PhysRevLett.106.180504) protocol first defined by Magesan et al. Second, there is ["Direct RB"](https://arxiv.org/abs/1807.07975), which is particularly suited to multi-qubit benchmarking.  More details on using these protocols (e.g. how to generate a set of RB sequences) see the separate [RB overview tutorial](../rb/Overview) and related tutorials.

## Robust Phase Estimation (RPE)
The Robust Phase Estimation (RPE) protocol is designed to efficiently estimate a few specific parameters of certain single-qubit models.  Below we demonstrate how to run RPE with the single-qubit model containing $X(\pi/2)$ and $Y(\pi/2)$ gates.  The list of requisite circuits is given by `make_rpe_angle_string_list_dict` and simulated noisy data is analyzed using `analyze_rpe_data`.  For more information on running RPE see the [RPE tutorial](../protocols/RobustPhaseEstimation).

```{code-cell} ipython3
from pygsti.extras import rpe
from pygsti.modelpacks.legacy import std1Q_XY
import numpy as np

#Declare the particular RPE instance we are interested in (X and Y pi/2 rotations)
# Note: Prep and measurement are for the |0> state.
rpeconfig_inst = rpe.rpeconfig_GxPi2_GyPi2_00
stringListsRPE = rpe.rpeconstruction.create_rpe_angle_circuits_dict(10,rpeconfig_inst)

angleList = ['alpha','epsilon','theta']
numStrsD = {'RPE' : [6*i for i in np.arange(1,12)] }

#Create fake noisy model
print(stringListsRPE['totalStrList'][0])
mdl_real = std1Q_XY.target_model().randomize_with_unitary(.01,seed=0)
ds_rpe = pygsti.data.simulate_data(mdl_real,stringListsRPE['totalStrList'],
                                           num_samples=1000,sample_error='binomial',seed=1)

#Run RPE protocol
resultsRPE = rpe.analyze_rpe_data(ds_rpe,mdl_real,stringListsRPE,rpeconfig_inst)

print('alpha_true - alpha_est_final =',resultsRPE['alphaErrorList'][-1])
print('epsilon_true - epsilon_est_final =',resultsRPE['epsilonErrorList'][-1])
print('theta_true - theta_est_final =',resultsRPE['thetaErrorList'][-1])
```

## Data set comparison tests
The `DataComparator` object is designed to check multiple `DataSet` objects for consistency.  This procedure essentially answers the question: "Is it better to describe `DataSet`s $A$ and $B$ as having been generated by the *same* set of probabilities or *different* sets?".  This quick test is useful for detecting drift in experimental setups from one round of data-taking to the next, and doesn't require constructing any `Model` objects.  Below, we generate three `DataSet` objects - two from the same underlying model and one from a different model - and show that we can detect this difference.  For more information, see the [tutorial on data set comparison](../utilities/DatasetComparison).

```{code-cell} ipython3
from pygsti.modelpacks import smq1Q_XYI

#Generate data from two different models
mdlA = smq1Q_XYI.target_model().randomize_with_unitary(.01,seed=0)
mdlB = smq1Q_XYI.target_model().randomize_with_unitary(.01,seed=1)

circuits = pygsti.circuits.create_lsgst_circuits(
    smq1Q_XYI.target_model(),smq1Q_XYI.prep_fiducials(),
    smq1Q_XYI.meas_fiducials(),smq1Q_XYI.germs(),[1,2,4,8])

#Generate the data for the two datasets, using the same model, and one with a different model
dsA1 = pygsti.data.simulate_data(mdlA,circuits,100,'binomial',seed=10)
dsA2 = pygsti.data.simulate_data(mdlA,circuits,100,'binomial',seed=20)
dsB  = pygsti.data.simulate_data(mdlB,circuits,100,'binomial',seed=30)

#Let's compare the two datasets.
print("Compare two *consistent* DataSets (generated from the same underlying model)")
comparator_A1_A2 = pygsti.data.DataComparator([dsA1,dsA2])
comparator_A1_A2.run(significance=0.05)

print("\nCompare two *inconsistent* DataSets (generated from different model)")
comparator_A1_B = pygsti.data.DataComparator([dsA1,dsB])
comparator_A1_B.run(significance=0.05)

#Plots of consistent (top) and inconsistent (bottom) cases
ws.DatasetComparisonHistogramPlot(comparator_A1_A2, log=True, display='pvalue', scale=0.8)
ws.DatasetComparisonHistogramPlot(comparator_A1_B, log=True, display='pvalue', scale=0.8)
```

## Gate Set Tomography (GST)
Gate set tomography (GST) is a protocol designed to solve the inverse of "use this model to simulate observed data"; its goal is to *infer a model based on actual observed data*.  From a functional perspective, GST can be viewed as an inverse of the `generate_fake_data` function we've used a bunch above: it takes a `DataSet` and produces a `Model`.

Because this inverse problem traverses some technical challenges, GST also requires a *structured* set of `Circuits` to work reliably and efficiently.  Here enters the concepts of "fiducial" and "germ" circuits, as well as a list of "maximum-repeated-germ-lengths" or just "max-lengths".  For details, see the [tutorial on the structure of GST circuits](../gst/CircuitConstruction) and the [tutorial on fiducial and germ selection](../gst/FiducialAndGermSelection).  The important takeaway is that the GST circuits are described below by the 4 variables: `prep_fiducials`, `meas_fiducials`, `germs`, and `maxLengths`.

Below, we generate a set of GST circuits and simulate them using a noisy (slightly depolarized) model of the some ideal operations to get a `DataSet`.

```{code-cell} ipython3
from pygsti.modelpacks import smq1Q_XYI

# 1) get the target Model
mdl_ideal = smq1Q_XYI.target_model()

# 2) get the building blocks needed to specify which circuits are needed
prep_fiducials, meas_fiducials = smq1Q_XYI.prep_fiducials(), smq1Q_XYI.meas_fiducials()
germs = smq1Q_XYI.germs()
maxLengths = [1,2,4] # roughly gives the length of the sequences used by GST

# 3) generate "fake" data from a depolarized version of mdl_ideal
mdl_true = mdl_ideal.depolarize(op_noise=0.01, spam_noise=0.001)
listOfExperiments = pygsti.circuits.create_lsgst_circuits(
    mdl_ideal, prep_fiducials, meas_fiducials, germs, maxLengths)
ds = pygsti.data.simulate_data(mdl_true, listOfExperiments, num_samples=1000,
                                            sample_error="binomial", seed=1234)

#Run GST
results = pygsti.run_stdpractice_gst(ds, mdl_ideal, prep_fiducials, meas_fiducials, 
                                    germs, maxLengths, modes="full TP,Target", verbosity=1)

mdl_estimate = results.estimates['full TP'].models['stdgaugeopt']
print("2DeltaLogL(estimate, data): ", pygsti.tools.two_delta_logl(mdl_estimate, ds))
print("2DeltaLogL(true, data): ", pygsti.tools.two_delta_logl(mdl_true, ds))
print("2DeltaLogL(ideal, data): ", pygsti.tools.two_delta_logl(mdl_ideal, ds))
```

Recall that a lower $2\Delta\log\mathcal{L}$ means a better agreement between model and data. Note that the GST estimate fits the data best (even slightly better than the *true* model, because it agrees better with the finite sample noise), and the GST estimate is much better than the ideal model.

GST is essentially an automated model-tester that keeps modifying and testing models until it finds one the agrees with the data as well any model of the specified type can. In modern ``pyGSTi``, GST is typically run in an object-oriented way using `Protocol` objects as described in the [GST overview tutorial](../gst/Overview). To learn more about how to run GST by directly using functions that act on essential objects see the [function-based GST overview tutorial](../gst/Overview-functionbased).

The output of GST is an entire `Model` (contrasted with the one or several numbers of RB and RPE), there are many ways to assess and understand the performance of a QIP based on GST results.  The `ModelEstimateResults` object in pyGSTi is responsible for holding the GST and other model-based protocol results.  The structure and use of a `ModelEstimateResults` object is explained in the [Results tutorial](../objects/Results).  A common use for results objects is to generate "reports".  PyGSTi has the ability to generate HTML reports (a directory of files) whose goal is to display relevant model vs. data metrics such as $2\Delta\log\mathcal{L}$ as well as model vs. model metrics like process fidelity and diamond distance. To learn more about generating these "model-explaining" reports see the [report generation tutorial](../reporting/ReportGeneration).

Here's an example of how to generate a report (it will auto-open in a new tab; if it doesn't display **try it in FireFox**):

```{code-cell} ipython3
pygsti.report.construct_standard_report(
    results, title="Example GST Report", verbosity=1
).write_html("tutorial_files/myFirstGSTReport", auto_open=False, verbosity=1)
```

## Idle tomography
Idle tomography estimates the error rates of an $n$-qubit idle operation using relatively few sequences.  To learn more about how to use it, see the [idle tomography tutorial](../protocols/IdleTomography).

## Drift Characterization
Time-series data can be analyzed for significant indications of drift (time variance in circuit outcome probabilities).  See the [tutorial on drift characterization](../protocols/DriftCharacterization) for more details.

## Time-dependent gate set tomography
pyGSTi has recently added support for time-dependent models and data sets, allowing the GST to be performed in a time-dependent fashion.  See the [time-dependent GST tutorial](../gst/TimeDependent) for more details.

## What's next?
This concludes our overview of what can be done with pyGSTi. The rest of this documentation is split into in-depth tutorials on: the available protocols; various objects, including the aforementioned "essential" objects and other lower-level objects; details and options for forward simulators; generating human-readable reports; serialization, deserialization, and file I/O; and, last but not least, some end-to-end examples for common workflows.

---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.17.3
kernelspec:
  display_name: api_updates
  language: python
  name: api_updates
---

# Drift Characterization

This tutorial shows how to implement instability ("drift") detection and characterization on time-stamped data. This data can be from *any* quantum circuits, on *any* number of qubits, but we require around 100+ time-stamps per circuit (perhaps fewer if there are multiple measurement outcomes per time-stamp). If you only have data that is binned into a few different time periods then consider instead using the `DataComparator` object demonstrated in the [DataSetComparison](../algorithms/DatasetComparison.ipynb) tutorial.

Currently the gap between data collection times for each circuit is required to be approximately constant, both across the data collection times for each circuit, and across circuits. If this is not the case the code should still work, but the analysis it performs may be significantly sub-optimal, and interpretting the results is more complicated. There is beta-level capabilities within the functions used below to properly analyze unequally-spaced data, but it is untested and will not be used with the default options in the analysis code. This limitation will be addressed in a future release of pyGSTi.

This notebook is an introduction to these tools, and it will be augmented with further notebooks at a later date.

```{code-cell} ipython3
# Importing the drift module is essential
from pygsti.extras import drift

# Importing all of pyGSTi is optional, but often useful.
import pygsti
```

## Quick and Easy Analysis
First we import some *time-stamped* data. For more information on the mechanics of using time-stamped `DataSets` see the [TimestampedDataSets](../objects/advanced/TimestampedDataSets.ipynb) tutorial. The data we are importing is from long-sequence GST on $G_x$, and $G_y$ with time-dependent coherent errors on the gates.

We load the time-dependent data from the `timestamped_dataset.txt` file included with pyGSTi, and then build a `ProtocolData` object out of it so it can be used as input for `Protocol` objects.  We can pass `None` as the experiment design when constructing `data` because the stability analysis doesn't require any special structure to the circuits - it just requires the data to have timestamps.

```{code-cell} ipython3
# Initialize the circuit structure details of the imported data.

# Construct a basic ExplicitModel for the experiment design
model = pygsti.models.create_explicit_model_from_expressions(
    ['Q0'], ['Gx','Gy'],
    [ "X(pi/2,Q0)", "Y(pi/2,Q0)"] )

# This manually specifies the germ and fiducial structure for the imported data.
prep_fiducials = ['{}','Gx','Gy','GxGx']
meas_fiducials = ['{}','Gx','Gy']

germ_strs = ['Gx','Gy','GxGy','GxGxGyGxGyGy']
log2maxL = 7 # log2 of the maximum germ power

# Below we use the maxlength, germ and fiducial lists to create the GST structures needed for box plots.
prep_fiducials = [pygsti.circuits.Circuit(fs) for fs in prep_fiducials]
meas_fiducials = [pygsti.circuits.Circuit(fs) for fs in meas_fiducials]
germs = [pygsti.circuits.Circuit(g) for g in germ_strs]
max_lengths = [2**i for i in range(0,log2maxL+1)]
exp_design = pygsti.protocols.StandardGSTDesign(model, prep_fiducials, meas_fiducials, germs, max_lengths)

ds = pygsti.io.load_dataset("../tutorial_files/timestamped_dataset.txt")  # a DataSet
ds = ds.truncate(list(exp_design.all_circuits_needing_data))
data = pygsti.protocols.ProtocolData(exp_design, ds)
```

We then simply create a `StabilityAnalysis` protocol, and run it on the data.

```{code-cell} ipython3
%%time
protocol = pygsti.protocols.StabilityAnalysis()
results = protocol.run(data)
```

**Note** the `StabilityAnalysis` protocol has a variety of optional arguments that can be used to optimize the tool to different circumstances. In this tutorial we won't discuss the full range of analyzes that can be performed using the `drift` module.

+++

## Inspecting the results

Everything has been calculated, and we can now look at the results. If we print the returned results object (currently, essentially a container `StabilityAnalyzer` object), it will tell us whether instability was detected. If no instability is detected, then there is little else to do: the circuits are, as far as we can tell, stable, and most of the other results contained in a `StabilityAnalyzer` will not be very interesting. However, here instability is detected:

```{code-cell} ipython3
print(results.stabilityanalyzer)
```

NOTE: In notebook display of report figures does not work in jupyterlab due to restrictions on running javascript. Please use classic jupyter notebook if this is desired.

```{code-cell} ipython3
# Create a workspace to show plots
w = pygsti.report.Workspace()
w.init_notebook_mode(connected=False, autodisplay=True) 
```

### 1. Instability Detection Results : Power Spectra and the Frequencies of Instabilities

The analysis is based on generating power spectra from the data. If the data is sampled from a time-invariant probability distribution, the powers in these spectra have an expected value of 1 and a known distribution (it is $ \frac{1}{k}\chi^2_k$ with $k$ depending on exactly what spectrum we're looking at). So we can test for violations of this, by looking for powers that are too high to be consistent with the time-invariant probability distribution hypothesis.

A power spectrum is obtained for each circuit, and these can be averaged to obtain a single "global" power spectrum. This is plotted below:

```{code-cell} ipython3
w.PowerSpectraPlot(results)
```

Frequencies with power above the threshold in a spectrum are almost certainly components in the underlying (perhaps) time-varying probability for the corresponding circuit. We can access the frequencies above the threshold in the global power spectrum (which can't be assigned to a particular circuit, but *are* components in the probabilities for one or more circuits) as follows:

```{code-cell} ipython3
print(results.instability_frequencies())
```

To get the power spectrum, and detected significant frequencies, for a particular circuit we add an optional argument to the above functions:

```{code-cell} ipython3
spectrumlabel = {'circuit':pygsti.circuits.Circuit('(Gx)^128')}
print("significant frequencies: ", results.instability_frequencies(spectrumlabel))
w.PowerSpectraPlot(results, spectrumlabel)
```

Note that all frequencies are in 1/units where "units" are the units of the time stamps in the `DataSet`. Some of the plotting functions display frequencies as Hertz, which is based on the assumption that these time stamps are in seconds. (in the future we will allow the user to specify the units of the time stamps).

We can access a dictionary of all the circuits that we have detected as being unstable, with the values the detected frequencies of the instabilities.

```{code-cell} ipython3
unstablecircuits = results.unstable_circuits()
# We only display the first 10 circuits and frequencies, as there are a lot of them!
for ind, (circuit, freqs) in enumerate(unstablecircuits.items()):
    if ind < 10: print(circuit.str, freqs)
```

We can jointly plot the power spectra for any set of circuits, by handing a dictionary of circuits to the `PowerSpectraPlot` function.

```{code-cell} ipython3
circuits = {L: pygsti.circuits.Circuit(None,stringrep='Gx(Gx)^'+str(L)+'Gx') for L in [1,2,4,16,64,128]}
w.PowerSpectraPlot(results, {'circuit':circuits}, showlegend=True)
```

### 2. Instability Characterization Results : Probability Trajectories

The tools also estimate the probability trajectories for each circuit, i.e., the probabilities to obtain each possible circuit outcome as a function of time. We can plot the estimated probability trajectory for any circuit of interest (or a selection of circuits, if we hand the plotting function a dictionary or list of circuits instead of a single circuit).

```{code-cell} ipython3
circuit = pygsti.circuits.Circuit(None, stringrep= 'Gx(Gx)^128Gx')
w.ProbTrajectoriesPlot(results.stabilityanalyzer, circuit, ('1',))
```

If you simply want to access the time-varying distribution, use the `probability_trajectory()` method.

The size of the instability in a circuit can be summarized by the amplitudes in front of the non-constant basis functions in our estimate of the probability trajectories. By summing these all up (and dividing by 2), we can get an upper bound on the maximum TVD between the instaneous probability distribution (over circuit outcomes) and the mean of this time-varying probability distribution, with this maximization over all times.

```{code-cell} ipython3
results.maximum_tvd_bound(circuit)
```

If you want to access this quantity for all unstable circuits, you can set `getmaxtvd = True` in the `unstable_circuits()` method. We can also access it's maximum over all the circuits as

```{code-cell} ipython3
results.maxmax_tvd_bound()
```

### 3. Further plotting for data from structured circuits (e.g., GST circuits)

If the data is from GST experiments, or anything with a GST-like structure of germs and fudicials (such as Ramsey circuits), we can create some extra plots, and a drift report.

We can plot all of the power spectra, and probability trajectories, for amy (preparation fiducial, germ, measurement fiducial) triple. This shows how any instability changes, for this triple, as the germ power is increased.

```{code-cell} ipython3
w.GermFiducialPowerSpectraPlot(results, 'Gy', 'Gx', 'Gx', showlegend=True)
w.GermFiducialProbTrajectoriesPlot(results, 'Gy', 'Gx', 'Gx', ('0',), showlegend=True)
```

We can make a boxplot that shows $\lambda = -\log_{10}(p)$ for each circuit, where $p$ is the p-value of the maximum power in the spectrum for that circuit. This statistic is a good summary for the evidence for instability in the circuit. Note that, for technical reasons, $\lambda$ is truncated above at 16.

```{code-cell} ipython3
circuits128 = exp_design.circuit_lists[-1] # Pull out circuits up to max L
w.ColorBoxPlot('driftdetector', circuits128, None, None, stabilityanalyzer=results.stabilityanalyzer)
```

The $\lambda = -\log_{10}(p)$ values do not *directly* tell us anything about the size of any detected instabilities. The $\lambda$ statistics summarizes how certain we are that there is some instability, but if enough data is taken then even tiny instabilities will become obvious (and so we would have $\lambda \to \infty$, except that the code truncates it to 16). 

The boxplot below summarizes the **size** of any detected instability with the bound on the maximal instantaneous TVD for each circuit, introduced above. Here this is zero for most circuits, as we did not detect any instability in those circuits - so our estimate for the probability trajectory is constant. The gradient of the colored boxes in this plot fairly closely mirror those in the plot above, but this is not always guaranteed to happen (e.g., if there is a lot more data for some of the circuits than others).

```{code-cell} ipython3
# Create a boxplot of the maximum power in the power spectra for each sequence.
w.ColorBoxPlot('driftsize', circuits128, None, None, stabilityanalyzer=results.stabilityanalyzer)
```

We can also create a report that contains all of these plots, as well as a few other things. But note that creating this report is currently fairly slow. Moreover, all the plots it contains have been demonstrated above, and everything else it contains can be accessed directly from the `StabilityAnalyzer` object. To explore all the things that are recorded in the `StabilityAnalyzer` object take a look at its `get` methods.

```{code-cell} ipython3
report = pygsti.report.create_drift_report(results, title='Example Drift Report')
report.write_html('../tutorial_files/DriftReport')
```

You can now open the file [../tutorial_files/DriftReport/main.html](../tutorial_files/DriftReport/main.html) in your browser (Firefox works best) to view the report.

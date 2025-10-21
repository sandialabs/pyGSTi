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

# Volumetric Benchmarking
This tutorial demonstrates how to compute volumetric benchmarks in pyGSTi.  Volumetric benchmarks map a (*width*, *depth*) pair to a test suite of circuits with (at least approximately) the given width and depth, and define an overall success measure that lies between 0 and 1 (1 indicating better performance on the test suite).  Thus, by collecting the success measures for many (width, depth) pairs, one can explore, in addition to the overall processor performance, the tradeoff between a quantum processor's ability to perform deep vs. wide circuits.  For more information on the theory and motivation for volumetric benchmarks, see [this paper](https://arxiv.org/abs/1904.05546).

We'll begin by importing pyGSTi as usual, and making `pp` a shorthand for `pygsti.protocols` since we'll be using it a lot.

```{code-cell} ipython3
import pygsti
import pygsti.protocols as pp
from pygsti.processors import QubitProcessorSpec as QPS
from pygsti.processors import CliffordCompilationRules as CCR
```

## Define the processor
Next, we define the processor that we're benchmarking.  For this we use a `QubitProcessorSpec` object (see the [tutorial on processor specs](../objects/ProcessorSpec)) to define a ring of 4 qubits.  Each qubit has 4 single-qubit gates: $X(\pm\pi/2)$ and $Y(\pm\pi/2)$, and CPHASE gates are allowed between neighbors.

```{code-cell} ipython3
n_qubits = 4
qubit_labels = ['Q0', 'Q1', 'Q2', 'Q3']
gate_names = ['Gxpi2', 'Gxmpi2', 'Gypi2', 'Gympi2', 'Gcphase']
availability = {'Gcphase': [('Q0', 'Q1'), ('Q1', 'Q2'), ('Q2', 'Q3'), ('Q3', 'Q0')]}

# A ProcessorSpec for available gates
pspec = pygsti.processors.QubitProcessorSpec(n_qubits, gate_names, availability=availability, 
                                             qubit_labels=qubit_labels, geometry='line')

# Rules for how to compile native gates into Cliffors
compilations = {'absolute': CCR.create_standard(pspec, 'absolute', ('paulis', '1Qcliffords'), verbosity=0),            
                'paulieq': CCR.create_standard(pspec, 'paulieq', ('1Qcliffords', 'allcnots'), verbosity=0)}
```

## Step 1: create an experiment design

There are many types of volumetric benchmarks.  In this example, we'll use associate a suite of random circuits with various (width, depth) pairs by using direct-RB (DRB) circuits (a family of test suites for different *depths*) on different portions of the processor and therefore for different *widths*.

We create `DirectRBDesign` experiment design objects, giving a different `qubit_labels` argument each time.  To each design we add a `ByDepthSummaryStatistics` "default protocol" to make it easier to run the protocols later (see step 3).  We decide to save some time (perhaps at the expense of increased crosstalk error) by performing some of these experiment designs simultaneously.  This is achieved by using multiple "sub-designs" to construct a `SimultaneousExperimentDesign`.  We create two simultaneous designs (one which performs two 2-qubit DRB test suites at the same time, the other which performs four 1-qubit DRB suites), and combine these using a `CombinedExperimentDesign`.  (Note that if we didn't want to run any of the suites simultaneously, we could have just combined all six DRB designs under a `CombinedExperimentDesign`.)

In this way, the cell below defines the entire experiment we want to perform, and all the circuits we need to run are nicely bundled into a single experiment design (`entire_design`), which we write to disk and await data collection.

```{code-cell} ipython3
depths = [0, 3]#, 10, 15, 20]
circuits_per_depth = 1#30

VB_design01 = pp.DirectRBDesign(pspec, compilations, depths, circuits_per_depth, qubit_labels=['Q0', 'Q1'])
VB_design23 = pp.DirectRBDesign(pspec, compilations, depths, circuits_per_depth, qubit_labels=['Q2', 'Q3'])
VB_design01.add_default_protocol(pp.ByDepthSummaryStatistics(statistics_to_compute=('polarization',)))
VB_design23.add_default_protocol(pp.ByDepthSummaryStatistics(statistics_to_compute=('polarization',)))

designS1 = pp.SimultaneousExperimentDesign([VB_design01, VB_design23], qubit_labels=qubit_labels)

VB_design0 = pp.DirectRBDesign(pspec, compilations, depths, circuits_per_depth, qubit_labels=['Q0'])
VB_design1 = pp.DirectRBDesign(pspec, compilations, depths, circuits_per_depth, qubit_labels=['Q1'])
VB_design2 = pp.DirectRBDesign(pspec, compilations, depths, circuits_per_depth, qubit_labels=['Q2'])
VB_design3 = pp.DirectRBDesign(pspec, compilations, depths, circuits_per_depth, qubit_labels=['Q3'])
VB_design0.add_default_protocol(pp.ByDepthSummaryStatistics(statistics_to_compute=('polarization',)))
VB_design1.add_default_protocol(pp.ByDepthSummaryStatistics(statistics_to_compute=('polarization',)))
VB_design2.add_default_protocol(pp.ByDepthSummaryStatistics(statistics_to_compute=('polarization',)))
VB_design3.add_default_protocol(pp.ByDepthSummaryStatistics(statistics_to_compute=('polarization',)))

designS2 = pp.SimultaneousExperimentDesign([VB_design0, VB_design1, VB_design2, VB_design3], qubit_labels=qubit_labels)

entire_design = pp.CombinedExperimentDesign({"specA": designS1, "specB": designS2})
try:
    import shutil; shutil.rmtree('../../tutorial_files/vb_example')  # start with a clean directory - stale files can be problematic
except FileNotFoundError:
    pass
pygsti.io.write_empty_protocol_data("../../tutorial_files/vb_example", entire_design)
```

## Step 2: collect data as specified by the experiment design
Next, we just follow the instructions in the experiment design to collect data from the quantum processor.  In this example, we'll generate the data using a depolarizing noise model since we don't have a real quantum processor lying around.  The cell below simulates taking the data, and would be replaced with the user filling out the empty "template" data set file with real data.

```{code-cell} ipython3
mdl_datagen = pygsti.models.create_crosstalk_free_model(pspec, ideal_gate_type='full TP')
for gate in mdl_datagen.operation_blks['gates'].values(): gate.depolarize(0.01)
pygsti.io.fill_in_empty_dataset_with_fake_data("../../tutorial_files/vb_example/data/dataset.txt", mdl_datagen,
                                               num_samples=1000, seed=2020)
```

Now that the template file has been filled in (or just replaced with one containing data), we read it in from the same root directory we saved the data to above.  This loads in both the `dataset.txt` we simulated above and the experiment design (saved in the `.../vb_example/edesign` directory).

```{code-cell} ipython3
data = pygsti.io.read_data_from_dir('../../tutorial_files/vb_example')
```

## Step 3: Run the volumetric benchmark protocol on each DRB experiment design
Now that we have the data, we'd like extract our volumetric-benchmarking metric for each circuit depth within each of the sub-designs defined above.  This is done by running a `ByDepthSummaryStatistics` protocol on each of the DRB sub-designs.  The `ByDepthSummaryStatistics` protocol is able to compute many different "summary" metrics given a set of definite-outcome circuit data (like DRB data).  We supplied `ByDepthSummaryStatistics` instances as the default protocol to use for each experiment design, which allows us to conveniently run the protocols via the function `run_default_protocols` and have PyGSTi's protocol-object framework keep track of where each experiment designs sits within the nested hierarchy of designs. (This function walks through the tree of experiment designs and runs any and all default protocols.)  When creating the protocols, we specified that they should compute the single `'polarizaton'` metric, which we will utilize later on.

```{code-cell} ipython3
results = pp.run_default_protocols(data)
```

## Look at the results
Volumetric benchmarks (VB) are fairly new to pyGSTi, and we don't have a nice built-in plot for displaying collections of VB data.  Below we demonstrate how the returned results object can be converted into a [Pandas](https://pandas.pydata.org) data-frame, which allows the user to easily slice the data and create plots using their favorite plotting tools.  We demonstrate how this is done with pyGSTi's preferred plotting library, [Plotly](https://plot.ly/python).

```{code-cell} ipython3
df = results.to_dataframe()  # you'll need the 'pandas' python package for this
```

```{code-cell} ipython3
df.head()  # The raw data
```

```{code-cell} ipython3
# Filter the data
df_relevant_cols = df.loc[:, ['Value', 'ValueName', 'Depth', 'Width']]
df_vb = df_relevant_cols[ df_relevant_cols['ValueName'] == 'polarization' ].loc[:, ['Value', 'Depth', 'Width']]
df_vb.head()
```

```{code-cell} ipython3
#Get the data to plot
widths = sorted(df_vb.Width.unique())
depths = sorted(df_vb.Depth.unique())
vals = [ [ df_vb[(df_vb['Depth'] == d) & (df_vb['Width'] == w) ]['Value'].mean()
            for d in depths ]  for w in widths ]    
```

```{code-cell} ipython3
#Make the plot (you'll need the 'plotly' python package for this)
import plotly.graph_objects as go 
fig = go.Figure(data=go.Heatmap(z=vals, x=depths, y=widths, colorscale='Bluered_r'))
fig.update_layout(title='Volumetric benchmarking example',
                  xaxis={'title': 'Depth'}, yaxis={'title': 'Width'},
                  height=300, width=400)
fig.show()
```

```{code-cell} ipython3

```

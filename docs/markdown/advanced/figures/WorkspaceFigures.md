---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.19.4
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Workspace tables and plots

A `Workspace` is a container and factory for pyGSTi's figures. You hand it objects you've already computed or loaded (a `Model`, a `DataSet`, a `ModelEstimateResults`) and it hands back a table or a plot. Every figure in an automated pyGSTi report comes from a `Workspace`, so anything you see in a report you can also build one piece at a time in a notebook.

The figures are HTML: native HTML tables and [Plotly](https://plot.ly/python/) plots, rather than the more traditional LaTeX tables and [matplotlib](https://matplotlib.org) plots. Three reasons for that choice:

1. Interactivity. HTML plots and tables can respond to the mouse; with LaTeX that's impossible and with matplotlib it's painful.
2. Integration. HTML drops into web pages (a nicer report than a many-page PDF) and into Jupyter notebooks.
3. Portability. Plotly figures are HTML and JS, so they store and travel more robustly than matplotlib `Figure` objects, which are hard even to pickle.

The rest of this page is a gallery. Skim it for the figure you want, then read the docstring for the arguments.

## Creating a workspace

```{code-cell} ipython3
import numpy as np
import pygsti

w = pygsti.report.Workspace()
w.init_notebook_mode(connected=False, autodisplay=True)
```

`init_notebook_mode` injects the HTML and JavaScript that make figures display. Run it once, near the top of your notebook. If it worked you'll see a green **Notebook Initialization Complete** message. A blue **Loading...** message that never resolves means one of two things: the notebook isn't "Trusted" (check the upper right corner of the Jupyter window), or you asked for web-hosted resources without a working internet connection. Fix whichever it is and reload with your *browser's* reload button, not Jupyter's.

The `connected` argument controls where those resources come from. With `connected=True` they're loaded from a CDN, which keeps the notebook file small if you save it as HTML. With `connected=False` pyGSTi supplies everything except MathJax itself and writes an `offline` directory alongside your notebook. That directory has to tag along with the notebook, and with any saved-as-HTML version of it, or nothing renders.

`autodisplay=True` means a figure appears as soon as you create it. Leave it `False` and you have to capture the returned object and call its `.display()` method.

Figures are member functions of the workspace. Type `w.` and hit TAB to see the somewhat-descriptive names of everything it can build; hit SHIFT-TAB after the opening parenthesis, say after typing `w.GatesVsTargetTable(`, to bring up the signature in Jupyter's help window. Displayed figures have a resize handle in their lower right corner.

Two quick ones, to show the shape of the thing:

```{code-cell} ipython3
w.MatrixPlot(np.array([[1, 2], [3, 4]], 'd'), color_min=0, color_max=4)
```

```{code-cell} ipython3
from pygsti.modelpacks import smq1Q_XYI
w.GatesTable(smq1Q_XYI.target_model())
```

## Getting some results

Most of the interesting figures want GST output, so run gate set tomography on the standard 1-qubit model to get something to play with. Generate a few `DataSet` objects, then call `run_long_sequence_gst` on each to get `ModelEstimateResults` objects. For the details, see the [GST overview tutorial](../../start/FirstGST) and the [tutorial on the ModelEstimateResults object](../../guides/analysis/Results).

```{code-cell} ipython3
#The usual GST setup: we're going to run GST on the standard XYI 1-qubit model
target_model = smq1Q_XYI.target_model()
prep_fiducials = smq1Q_XYI.prep_fiducials()
meas_fiducials = smq1Q_XYI.meas_fiducials()
germs = smq1Q_XYI.germs()
maxLengths = [1,2]
listOfExperiments = pygsti.circuits.create_lsgst_circuits(
    target_model.operations.keys(), prep_fiducials, meas_fiducials, germs, maxLengths)
```

```{code-cell} ipython3
#Create some datasets for analysis
mdl_datagen1 = target_model.depolarize(op_noise=0.1, spam_noise=0.02)
mdl_datagen2 = target_model.depolarize(op_noise=0.05, spam_noise=0.01).rotate(rotate=(0.01,0.01,0.01))

ds1 = pygsti.data.simulate_data(mdl_datagen1, listOfExperiments, num_samples=1000,
                                            sample_error="binomial", seed=1234)
ds2 = pygsti.data.simulate_data(mdl_datagen2, listOfExperiments, num_samples=1000,
                                            sample_error="binomial", seed=1234)
ds3 = ds1.copy_nonstatic(); ds3.add_counts_from_dataset(ds2); ds3.done_adding_data()
```

```{code-cell} ipython3
#Run GST on all three datasets
target_model.set_all_parameterizations("full TP")
results1 = pygsti.run_long_sequence_gst(ds1, target_model, prep_fiducials, meas_fiducials, germs, maxLengths, verbosity=0)
results2 = pygsti.run_long_sequence_gst(ds2, target_model, prep_fiducials, meas_fiducials, germs, maxLengths, verbosity=0)
results3 = pygsti.run_long_sequence_gst(ds3, target_model, prep_fiducials, meas_fiducials, germs, maxLengths, verbosity=0)

#make some shorthand variable names for later
tgt = results1.estimates['GateSetTomography'].models['target']

ds1 = results1.dataset
ds2 = results2.dataset
ds3 = results3.dataset

mdl1 = results1.estimates['GateSetTomography'].models['stdgaugeopt']
mdl2 = results2.estimates['GateSetTomography'].models['stdgaugeopt']
mdl3 = results3.estimates['GateSetTomography'].models['stdgaugeopt']

gss = results1.circuit_lists['final']
```

## Gallery

```{code-cell} ipython3
w.ColorBoxPlot(("logl",), gss, ds1, mdl1, typ='scatter')
w.ColorBoxPlot(("logl",), gss, ds1, mdl1, typ='boxes')
w.ColorBoxPlot(("logl",), gss, ds1, mdl1, typ='histogram')
```

```{code-cell} ipython3
iteration_estimates = [results1.estimates['GateSetTomography'].models['iteration %d estimate' % i]
                       for i in range(results1.estimates['GateSetTomography'].num_iterations)]
w.FitComparisonBarPlot(gss, results1.circuit_lists['iteration'],iteration_estimates, ds1)
```

```{code-cell} ipython3
w.GramMatrixBarPlot(ds1,tgt)
```

```{code-cell} ipython3
w.GatesVsTargetTable(mdl1, tgt)
```

```{code-cell} ipython3
w.SpamVsTargetTable(mdl2, tgt)
```

```{code-cell} ipython3
w.ColorBoxPlot(("chi2","logl"), gss, ds1, mdl1, box_labels=True)
  #Notice how long it takes to switch between "chi2" and "logl".  This 
  # is due to drawing all of the box labels (box_labels=True).
```

```{code-cell} ipython3
#This one requires knowing that each Results object holds a list of models
# from each GST iteration along with the corresponding circuit lists that were used.
iteration_estimates = [results1.estimates['GateSetTomography'].models['iteration %d estimate' % i]
                       for i in range(results1.estimates['GateSetTomography'].num_iterations)]
w.FitComparisonTable(gss, results1.circuit_lists['iteration'], iteration_estimates, ds1)
```

```{code-cell} ipython3
# We can reuse 'gss' for all three since the operation sequences are the same.
w.FitComparisonTable(["GS1","GS2","GS3"], [gss, gss, gss], [mdl1,mdl2,mdl3], ds1, x_label="Model")
```

```{code-cell} ipython3
w.ChoiTable(mdl3, display=('matrix','barplot'))
```

```{code-cell} ipython3
w.GateMatrixPlot(mdl1[('Gxpi2',0)].to_dense(),scale=1.0, box_labels=True,ylabel="hello")
w.GateMatrixPlot(pygsti.tools.error_generator(mdl1[('Gxpi2',0)].to_dense(), tgt[('Gxpi2',0)].to_dense(), 'pp'), scale=1.5)
```

```{code-cell} ipython3
from pygsti.modelpacks import smq2Q_XYCNOT
w.GateMatrixPlot(smq2Q_XYCNOT.target_model()[('Gxpi2',0)].to_dense(),scale=1.0, box_labels=False,ylabel="hello",mx_basis_x="pp")
```

```{code-cell} ipython3
mx = np.array( 
[[ 7.3380823,   8.28446943,  7.4593754,   3.91256384,  0.68631199],
 [ 3.36139818,  7.42955114,  6.78516082,  0.35863173,  5.57713093],
 [ 2.61489939,  3.40182958,  6.77389064,  9.29736475,  0.33824271],
 [ 9.64258149,  9.45928809,  6.91516602,  5.61423854,  0.56480777],
 [ 2.15195669,  9.37588783,  5.1781991,   7.20087591,  1.46096288]], 'd')
cMap = pygsti.report.colormaps.LinlogColormap(vmin=0, vmax=10, num_boxes=25, pcntle=0.55, dof_per_box=1, color='blue')
w.MatrixPlot(mx, colormap=cMap, colorbar=False)
```

```{code-cell} ipython3
mx = np.identity(3,'d')
mx[0,1] = 2.1
mx[2,2] = 4.0
mx[2,0] = 3.0
mx[0,2] = 7.0
mx[2,1] = 10.0
mx[0,0] = np.nan
cMap = pygsti.report.colormaps.PiecewiseLinearColormap(
            [[0,(0,0.5,0)],[1,(0,1.0,0)],[2,(1.0,1.0,0)],
             [4,(1.0,0.5,0)],[10,(1.0,0,0)]])
#print(cMap.colorscale())
w.MatrixPlot(mx, colormap=cMap, colorbar=False, grid="white:1", box_labels=True, prec=2,
             xlabels=('full TP',"CPTPLND","full"),ylabels=("DS0","DS1","DS2"))
```

```{code-cell} ipython3
w.ErrgenTable(mdl3,tgt)
```

```{code-cell} ipython3
w.PolarEigenvaluePlot([np.linalg.eigvals(mdl2[('Gxpi2',0)].to_dense())],["purple"],scale=1.5)
```

```{code-cell} ipython3
w.GateEigenvalueTable(mdl2, display=('evals','polar'))
```

```{code-cell} ipython3
w.GateDecompTable(mdl1,target_model)
#w.old_GateDecompTable(gs1) #historical; 1Q only
```

```{code-cell} ipython3
#Note 2Q angle decompositions
from pygsti.modelpacks import smq2Q_XXYYII
from pygsti.modelpacks import smq2Q_XYCNOT

w.GateDecompTable(smq2Q_XXYYII.target_model(), smq2Q_XXYYII.target_model())

import scipy
I = np.array([[1,0],[0,1]],'complex')
X = np.array([[0,1],[1,0]],'complex')
Y = np.array([[0,1j],[-1j,0]],'complex')
XX = np.kron(X,X)
YY = np.kron(Y,Y)
IX = np.kron(I,X)
XI = np.kron(X,I)
testU = scipy.linalg.expm(-1j*np.pi/2*XX)
testS = pygsti.unitary_to_process_mx(testU)
testS = pygsti.change_basis(testS,"std","pp")

#mdl_decomp = std2Q_XYCNOT.target_model()
#mdl_decomp.operations['Gtest'] = testS
#w.GateDecompTable(mdl_decomp, mdl_decomp)
```

```{code-cell} ipython3
dsLabels = ["A","B","C"]
datasets = [ds1, ds2, ds3]
dscmps = {}
for i,ds_a in enumerate(datasets):
    for j,ds_b in enumerate(datasets[i+1:],start=i+1):
        dscmps[(i,j)] = pygsti.data.DataComparator([ds_a, ds_b])

w.DatasetComparisonSummaryPlot(dsLabels, dscmps)
```

```{code-cell} ipython3
w.DatasetComparisonHistogramPlot(dscmps[(1,2)])
```

## Saving figures to file

Tables and plots have a `saveas` method. The output format comes from the file extension:

- `pdf`: Adobe portable document format
- `tex`: LaTeX source, uncompiled, *tables only*
- `pkl`: Python pickle, of a pandas `DataFrame` for tables and a dict for plots
- `html`: a stand-alone HTML document

```{code-cell} ipython3
import os
if not os.path.exists("../../../tutorial_files/tempTest"):
    os.mkdir("../../../tutorial_files/tempTest")

obj = w.GatesVsTargetTable(mdl1, tgt)
#obj = w.ErrgenTable(mdl3,tgt)
#obj = w.ColorBoxPlot(("logl",), gss, ds1, mdl1, typ='boxes')

obj.saveas("../../../tutorial_files/tempTest/testSave.tex")
obj.saveas("../../../tutorial_files/tempTest/testSave.pkl")
obj.saveas("../../../tutorial_files/tempTest/testSave.html")
```

Saving a *table* as a pdf requires `pdflatex` installed and on the system path: tables render to LaTeX first, then get compiled. Plots take a different route, converting the Plotly figure to a matplotlib one, so plot pdfs need no LaTeX.

```{code-cell} ipython3
:tags: [nbval-skip]

obj.saveas("../../../tutorial_files/tempTest/testSave.pdf")
```

## Exporting notebooks to HTML

You can save a figure-containing notebook like this one as an HTML file with **File => Download As => HTML** in the Jupyter menu. The plots stay interactive, so long as the file sits in a directory with an `offline` folder, which is why we passed `connected=False` above.

## Where to go next

Screen real estate runs out fast once you start stacking figures. The [switchboard tutorial](Switchboards) shows how to put dropdowns, buttons, and sliders in front of a set of workspace figures so you can flip between them instead of scrolling. The [report generation tutorial](../../guides/analysis/Reports) shows the other end of the same machinery: pyGSTi driving a `Workspace` for you to produce a standalone HTML report.

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

## Tutorial showing use of a `Workspace` object
### Part 2: Switchboards

"This tutorial introduces the `Switchboard` workspace object and demonstrates its use.  You may have gotten the sense from the last tutorial that screen real estate can quickly be taken up by plots and tables.  Wouldn't it me nice if we could interactively switch between plots or figures using buttons or sliders instead of having to scroll through endless pages of plots?  `Switchboard` to the rescue!

First though, let's run GST on the standard 1Q model to get some results (the same ones as the first tutorial).

```{code-cell} ipython3
import numpy as np
import pygsti
from pygsti.modelpacks import smq1Q_XYI

#The usual GST setup: we're going to run GST on the standard XYI 1-qubit model
target_model = smq1Q_XYI.target_model()
prep_fiducials = smq1Q_XYI.prep_fiducials()
meas_fiducials = smq1Q_XYI.meas_fiducials()
germs = smq1Q_XYI.germs()
maxLengths = [1,2,4,8]
listOfExperiments = pygsti.circuits.create_lsgst_circuits(
    target_model.operations.keys(), prep_fiducials, meas_fiducials, germs, maxLengths)

#Create some datasets for analysis
mdl_datagen1 = target_model.depolarize(op_noise=0.1, spam_noise=0.001)
mdl_datagen2 = target_model.depolarize(op_noise=0.05, spam_noise=0.01).rotate(rotate=(0.01,0,0))

ds1 = pygsti.data.simulate_data(mdl_datagen1, listOfExperiments, num_samples=1000,
                                            sample_error="binomial", seed=1234)
ds2 = pygsti.data.simulate_data(mdl_datagen2, listOfExperiments, num_samples=1000,
                                            sample_error="binomial", seed=1234)
ds3 = ds1.copy_nonstatic(); ds3.add_counts_from_dataset(ds2); ds3.done_adding_data()

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

mdl1 = results1.estimates['GateSetTomography'].models['go0']
mdl2 = results2.estimates['GateSetTomography'].models['go0']
mdl3 = results3.estimates['GateSetTomography'].models['go0']

circuits = results1.circuit_lists['final']
```

Next we create the workspace, as before.  This time, we'll leave `autodisplay=False` (the default), to demonstrate how this gives us more control over when workspace items are displayed.  In particular, we'll build up a several workspace objects and display them all at once.  **NOTE that setting `connected=True` means you need to have an internet connection!**

```{code-cell} ipython3
w = pygsti.report.Workspace()  #create a new workspace
w.init_notebook_mode(connected=False) # and initialize it so it works within a notebook
```

Note that if we create a table it doesn't get displayed automatically.

```{code-cell} ipython3
tbl1 = w.GatesVsTargetTable(mdl1, tgt)
```

To see it, we need to call `display()`:

```{code-cell} ipython3
tbl1.display()
```

### Switchboards
A `Switchboard` is essentially a collection of one or more switches along with a dictionary of "values" which depend on some or all of the switch positions.  Each value looks like a NumPy `ndarray` whose axes correspond to the switches that value depends upon.  The array can hold whatever you want: `Model`s, `DataSet`s, `float`s, etc., and from the perspective of the plot and table workspace objects the value looks like the thing contained in its array (e.g. a *single* `Model`, `DataSet`, or `float`, etc.).  

Let's start off simple and create a switchboard with a single switch named "My Switch" that has two positions "On" and "Off":

```{code-cell} ipython3
switchbd = w.Switchboard(["My Switch"],[["On","Off"]],["buttons"])
```

Next, add a "value" to the switchboard called "mdl" (for "model"), with is dependent on the 0-th (and only) switch of the switchboard:

```{code-cell} ipython3
switchbd.add("mdl", [0])
```

Now `switchbd` has a member, `mdl`, which looks like a 1-dimensional Numpy array (since `mdl` only depends on a single switch) of length 2 (because that single switch has 2 positions).

```{code-cell} ipython3
switchbd.mdl.shape
```

We'll use `switchbd.mdl` to switch between the models `mdl1` and `mdl2`.  We associate the "On" position with `mdl1` and the "Off" position with `mdl2` by simply assigning them to the corresponding locations of the array.  Note that we can use NumPy's fancy indexing to make this a breeze.

```{code-cell} ipython3
switchbd.mdl[:] = [mdl1,mdl2]
```

Ok, now here's the magical part: even though `switchbd.mdl` is really an array holding `Model` objects, when you provide it as an input to create a workspace item such as a plot or a table, it *behaves* like a single `Model` and can thus be used for any `Model`-type argument.  We'll use it as the first argument to `GatesVsTargetTable`.

```{code-cell} ipython3
tbl2 = w.GatesVsTargetTable(switchbd.mdl, tgt)
```

Note the the second argument (`tgt`, the target model) in the above call is just a plain old `Model`, just like it's always been up to this point.  The above line creates a table, `tbl2`, that is *connected* to the switchboard `switchbd`.  Let's display both the switchboard and the table together.

```{code-cell} ipython3
switchbd.display()
tbl2.display()
```

My pressing the "On" or "Off" button the table changes between displaying metrics for `mdl1` vs. `tgt` and `mdl2` vs. `tgt`, as expected.  In this simple example there was one switch controlling on table.  It is possible to have any number of switches controlling any number of tables and/or plots, and also to have multiple switchboards controlling a single plot or table.  In the following cells, more sophisticated uses of switchboards are demonstrated.

```{code-cell} ipython3
# Create a switchboard with straighforward dataset and model dropdown switches
switchbd2 = w.Switchboard(["dataset","model"], [["DS1","DS2","DS3"],["MODEL1","MODEL2","MODEL3"]], ["dropdown","dropdown"])
switchbd2.add("ds",(0,))
switchbd2.add("mdl",(1,))
switchbd2.ds[:] = [ds1, ds2, ds3]
switchbd2.mdl[:] = [mdl1, mdl2, mdl3]

#Then create a chi2 plot that can show the goodness-of-fit between any model-dataset pair
chi2plot = w.ColorBoxPlot(("chi2",), circuits, switchbd2.ds, switchbd2.mdl, scale=0.75)

# Can also truncate circuits to only a subset of the germs and depths
circuits2 = circuits.truncate(xs_to_keep=[1,2], ys_to_keep=circuits.ys[1:4])
chi2plot2 = w.ColorBoxPlot(("chi2",), circuits2, switchbd2.ds, switchbd2.mdl, scale=0.75)

switchbd2.display()
chi2plot.display()
chi2plot2.display()
```

```{code-cell} ipython3
#Perform gauge optimizations of gs1 using different spam weights
spamWts = np.linspace(0.0,1.0,20)
mdl_gaugeopts = [ pygsti.gaugeopt_to_target(mdl1, tgt,{'gates': 1, 'spam': x}) for x in spamWts]
```

```{code-cell} ipython3
# Create a switchboard with a slider that controls the spam-weight used in gauge optimization
switchbd3 = w.Switchboard(["spam-weight"], [["%.2f" % x for x in spamWts]], ["slider"])
switchbd3.add("mdlGO",(0,))
switchbd3.mdlGO[:] = mdl_gaugeopts

#Then create a comparison vs. target tables
tbl3 = w.GatesVsTargetTable(switchbd3.mdlGO, tgt)
tbl4 = w.SpamVsTargetTable(switchbd3.mdlGO, tgt)

switchbd3.display()
tbl3.display()
tbl4.display()
```

```{code-cell} ipython3
print(results1.estimates['GateSetTomography'])
```

```{code-cell} ipython3
# Create a slider showing the color box plot at different GST iterations
switchbd4 = w.Switchboard(["max(L)"], [list(map(str,circuits.xs))], ["slider"])
switchbd4.add("mdl",(0,))
switchbd4.add("circuits",(0,))
switchbd4.mdl[:] = [results1.estimates['GateSetTomography'].models['iteration '+ str(i)+ ' estimate' ] for i in range(len(maxLengths))]
switchbd4.circuits[:] = results1.circuit_lists['iteration']
            

#Then create a logl plot that can show the goodness-of-fit at different iterations
logLProgress = w.ColorBoxPlot(("logl",), switchbd4.circuits, ds1, switchbd4.mdl, scale=0.75)

logLProgress.display()
switchbd4.display()
```

### Switchboard Views
If you want to duplicate a switch board in order to have the same switches accessible at different (multiple) location in a page, you need to create switchboard *views*.  These are somewhat like NumPy array views in that they are windows into some base data - in this case the original `Switchboard` object.  Let's create a view of the `Switchboard` above.

```{code-cell} ipython3
sbv = switchbd4.view()
sbv.display()
```

Note that when you move one slider, the other moves with it.  This is because there's really only *one* switch.

Views don't need to contain *all* of the switches of the base `Switchboard` either.  Here's an example where each view only shows only a subset of the switches.  We also demonstrate here how the *initial positions* of each switch can be set via the `initial_pos` argument.

```{code-cell} ipython3
parent = w.Switchboard(["My Buttons","My Dropdown", "My Slider"],
                         [["On","Off"],["A","B","C"],["0","0.5","0.8","1.0"]],
                         ["buttons","dropdown","slider"], initial_pos=[0,1,2])
parent.display()
```

```{code-cell} ipython3
buttonsView = parent.view(["My Buttons"])
buttonsView.display()
```

```{code-cell} ipython3
otherView = parent.view(["My Dropdown","My Slider"])
otherView.display()
```

### Exporting to HTML
Again, you can save this notebook as an HTML file by going to **File => Download As => HTML** in the Jupyter menu.  The resulting file will retain all of the plot *and switch* interactivity, and in this case doesn't need the `offline` folder (because we set `connected=True` in `init_notebook_mode` above) but does need an internet connection.

```{code-cell} ipython3

```

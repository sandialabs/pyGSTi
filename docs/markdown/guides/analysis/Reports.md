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

# Report generation

PyGSTi constructs polished report documents that give both high-level summaries and detailed analyses of results, gate set tomography (GST) and model-testing results in particular.  Reports are meant to be a quick and easy way of analyzing `Model`-type estimates, and pyGSTi's report generation functions are designed to work with the `ModelEstimateResults` object produced by pyGSTi's GST protocols (see, for example, the [GST overview](../../start/FirstGST)).  A report generation function takes one or more results objects as input and produces an HTML file as output.  The HTML format lets reports include **interactive plots** and **switches** (see the [workspace switchboard guide](../../advanced/figures/Switchboards)), which makes it easy to compare different types of analysis or different data sets.

PyGSTi's reports are stand-alone HTML documents that cannot run Python.  Everything displayed in a report is pre-computed.  If you find yourself wanting to fiddle with things and feel that these reports are too static, use a `Workspace` object (see [Workspace tables and plots](../../advanced/figures/WorkspaceFigures)) inside a Jupyter notebook, where you can intermix report tables/plots and Python.  Internally, functions like `construct_standard_report` are simple factories for `Report` objects, which are in turn little more than a wrapper around a `Workspace` object plus a set of instructions for how to generate output in different formats.

## Get some `ModelEstimateResults`

Start by performing GST to create a `ModelEstimateResults` object (you could also just load one from file).  The calls below use the protocol-object API — an experiment design paired with data and run through a protocol — described in [migrating from the function-based API](../../advanced/migration/FromFunctionAPI).

```{code-cell} ipython3
import pygsti
from pygsti.modelpacks import smq1Q_XYI

target_model = smq1Q_XYI.target_model()
prep_fiducials = smq1Q_XYI.prep_fiducials()
meas_fiducials = smq1Q_XYI.meas_fiducials()
germs = smq1Q_XYI.germs()
maxLengths = [1,2,4,8,16]
ds = pygsti.io.read_dataset("../../../tutorial_files/Example_Dataset.txt", cache=True)

#Run GST
target_model.set_all_parameterizations("full TP") #TP-constrained
edesign = pygsti.protocols.StandardGSTDesign(target_model.create_processor_spec(), prep_fiducials,
                                             meas_fiducials, germs, maxLengths)
data = pygsti.protocols.ProtocolData(edesign, ds)

results = pygsti.protocols.GateSetTomography(target_model, verbosity=3).run(data)
```

## Make a report

Now that we have `results`, use `construct_standard_report` within `pygsti.report` to generate a `Report`.  `pygsti.report.construct_standard_report` is the most commonly used report factory function in pyGSTi; it's appropriate for smaller models (1- and 2-qubit) whose *operations are, or can be represented as, dense matrices and/or vectors*.

Once constructed, a `Report` object can write itself out as an HTML document, a PDF, or a notebook.  To open an HTML-format report, open the `main.html` file inside the report's folder.  Setting `auto_open=True` makes the finished report open in your web browser automatically.

```{code-cell} ipython3
report = pygsti.report.construct_standard_report(results, title="GST Example Report", verbosity=1)
#HTML
report.write_html("../../../tutorial_files/exampleReport", auto_open=False, verbosity=1)
```

```{code-cell} ipython3
:tags: [nbval-skip]

#PDF
report.write_pdf("../../../tutorial_files/exampleReport.pdf", auto_open=False, verbosity=1)
```

Several remarks about these reports are worth noting:

1. The **HTML reports are the primary report type in pyGSTi**, and are much more flexible.  The PDF reports are more limited (they can only display a *single* estimate and gauge optimization), and essentially contain a subset of the information and descriptive text of an HTML report.  So, if you can, use the HTML reports.  The PDF report's strength is its portability: PDFs are easily displayed by many devices, and they embed all that they need neatly into a single file.  **If you need to generate a PDF report** from `Results` objects that have multiple estimates and/or gauge optimizations, consider using the `Results` object's `view` method to single out the estimate and gauge optimization you're after.
2. It's best to use **Firefox** when opening the HTML reports.  (If there's a problem with your browser's capabilities it will be shown on the screen when you try to load the report.)
3. You'll need **`pdflatex`** on your system to compile PDF reports.
4. To familiarize yourself with the layout of an HTML report, click on the gray **"Help" link** on the black sidebar.

## Multiple estimates in a single report

Next, analyze the same data two different ways: with and without the TP constraint (that is, whether the gates *must* be trace-preserving), gauge optimizing each case using several different SPAM weights.  In each case we run `GateSetTomography` with `gaugeopt_suite=None`, so that no gauge optimization is done, then perform several gauge optimizations separately and add these to the `Results` object via its `add_gaugeoptimized` method.  Each run gets a distinct `name` so the two estimates can later live side by side in a single `Results` object.  Both cases fit the same circuits collected in `ds`, so both reuse the `edesign`/`data` pair built above.

```{code-cell} ipython3
#Case1: TP-constrained GST
tpTarget = target_model.copy()
tpTarget.set_all_parameterizations("full TP")
results_tp = pygsti.protocols.GateSetTomography(tpTarget, gaugeopt_suite=None, name='full TP',
                                               verbosity=1).run(data)

#Gauge optimize
est = results_tp.estimates['full TP']
mdlFinal = est.models['final iteration estimate']
mdlTarget = est.models['target']
for spamWt in [1e-4,1e-2,1.0]:
    mdl = pygsti.gaugeopt_to_target(mdlFinal,mdlTarget,{'gates':1, 'spam':spamWt})
    est.add_gaugeoptimized({'item_weights': {'gates':1, 'spam':spamWt}}, mdl, "Spam %g" % spamWt)
```

```{code-cell} ipython3
#Case2: "Full" GST
fullTarget = target_model.copy()
fullTarget.set_all_parameterizations("full")
results_full = pygsti.protocols.GateSetTomography(fullTarget, gaugeopt_suite=None, name='Full',
                                                 verbosity=1).run(data)

#Gauge optimize
est = results_full.estimates['Full']
mdlFinal = est.models['final iteration estimate']
mdlTarget = est.models['target']
for spamWt in [1e-4,1e-2,1.0]:
    mdl = pygsti.gaugeopt_to_target(mdlFinal,mdlTarget,{'gates':1, 'spam':spamWt})
    est.add_gaugeoptimized({'item_weights': {'gates':1, 'spam':spamWt}}, mdl, "Spam %g" % spamWt)
```

Now call the *same* `construct_standard_report` function, but instead of passing a single `Results` object as the first argument pass a *dictionary* of them.  The result is an **HTML report that includes switches** for selecting which case ("TP" or "Full") and which gauge optimization to display output quantities for.  PDF reports cannot support this interactivity, so **if you try to generate a PDF report you'll get an error**.

```{code-cell} ipython3
ws = pygsti.report.Workspace()
report = pygsti.report.construct_standard_report(
    {'full TP': results_tp, "Full": results_full}, title="Example Multi-Estimate Report", ws=ws, verbosity=2)
report.write_html("../../../tutorial_files/exampleMultiEstimateReport", auto_open=False, verbosity=2)
```

The call above constructs `ws`, a `Workspace` object.  PyGSTi's `Workspace` objects are both a factory for figures and tables and a smart cache for computed values.  A `Workspace` object can optionally be passed to `construct_standard_report`, where it is used to create all the figures in the report.  As an intended side effect, each of those figures is cached, along with some of the intermediate results used to create it.  Passing a preconstructed `Workspace` object to `construct_standard_report` lets it reuse previously cached quantities.

**Another way**: because `results_tp` and `results_full` used the same dataset and operation sequences, they could have been combined as two estimates in a single `ModelEstimateResults` object (see [Results](Results) for the structure of those objects).  Add the estimate within `results_full` to the estimates already contained in `results_tp`:

```{code-cell} ipython3
results_both = results_tp.copy() #copy just for neatness
results_both.add_estimates(results_full, estimates_to_add=['Full'])
```

Creating a report from `results_both` gives the same report we just generated.  We'll do it anyway, this time supplying `construct_standard_report` with the same `Workspace` used before.  That tells the constructed `Report` to use any cached values in the given *input* `Workspace` to expedite report generation.  Since our workspace has exactly the quantities we need cached in it, you'll notice a significant speedup.  Note that even though there's just a single `Results` object, you **still can't generate a PDF report** from it, because it contains multiple estimates.

```{code-cell} ipython3
pygsti.report.construct_standard_report(
    results_both,
    title="Example Multi-Estimate Report (v2)", 
    ws=ws, verbosity=2
).write_html("../../../tutorial_files/exampleMultiEstimateReport2", auto_open=False, verbosity=2)
```

## Multiple estimates and `StandardGST`

It's no coincidence that a `Results` object containing multiple estimates from the same data is precisely what the `StandardGST` protocol returns.  It runs GST several times, creating different "standard" estimates and gauge optimizations, so you can plot them all in a single HTML report.

```{code-cell} ipython3
results_std = pygsti.protocols.StandardGST(modes=('full TP', 'CPTPLND', 'Target'),
                                           gaugeopt_suite=('stdgaugeopt','toggleValidSpam'),
                                           target_model=target_model, verbosity=4).run(data)

# Generate a report with "TP", "CPTP", and "Target" estimates
pygsti.report.construct_standard_report(
    results_std, title="Post StdPractice Report", verbosity=1
).write_html("../../../tutorial_files/exampleStdReport", auto_open=False, verbosity=1)
```

## Reports with confidence regions

To display confidence intervals for reported quantities, you must do two things:

1. specify the `confidence_level` argument to `construct_standard_report`.
2. give the estimate(s) being reported a valid confidence-region factory.

Constructing a factory often means computing a Hessian, which can be time consuming, so it isn't done automatically.  Here's how to construct a valid factory for the "Spam 0.001" gauge optimization of the "CPTP" estimate, by computing and then projecting the Hessian of the likelihood function.

```{code-cell} ipython3
#Construct and initialize a "confidence region factory" for the CPTP estimate
crfact = results_std.estimates["CPTPLND"].add_confidence_region_factory('Spam 0.001', 'final')
crfact.compute_hessian(comm=None) #we could use more processors
crfact.project_hessian('intrinsic error')

pygsti.report.construct_standard_report(
    results_std, title="Post StdPractice Report (w/CIs on CPTP)",
    confidence_level=95, verbosity=1
).write_html("../../../tutorial_files/exampleStdReport2", auto_open=False, verbosity=1)
```

## Reports with multiple *different* data sets

We've already seen that `construct_standard_report` can be given a dictionary of `Results` objects instead of a single one.  That also allows reports containing estimates for different `DataSet`s, since each `Results` object only holds estimates for a single `DataSet`.  When the data sets have the same operation sequences, they're compared within a tab of the HTML report.

Below, we generate a new data set with the same sequences as the one loaded at the beginning of this page, run standard-practice GST on it, and create a report of those results alongside the original data set's.  Look at the **"Data Comparison" tab** within the gauge-invariant error metrics category.

```{code-cell} ipython3
#Make another dataset & estimates
target_model = smq1Q_XYI.target_model('full TP')
depol_gateset = target_model.depolarize(op_noise=0.1)
datagen_gateset = depol_gateset.rotate((0.05,0,0.03))

#Compute the sequences needed to perform long-sequence GST on this Model,
# using the same maxLengths as the fit below so the data covers what GST asks for
circuit_list = pygsti.circuits.create_lsgst_circuits(
    smq1Q_XYI.target_model(), smq1Q_XYI.prep_fiducials(), smq1Q_XYI.meas_fiducials(),
    smq1Q_XYI.germs(), maxLengths)
ds2 = pygsti.data.simulate_data(datagen_gateset, circuit_list, num_samples=1000,
                                             sample_error='binomial', seed=2018)

#Same circuits as `edesign`, just paired with the new dataset
data2 = pygsti.protocols.ProtocolData(edesign, ds2)
results_std2 = pygsti.protocols.StandardGST(modes=('full TP', 'Target'),
                                            gaugeopt_suite=('stdgaugeopt','toggleValidSpam'),
                                            target_model=target_model, verbosity=3).run(data2)

pygsti.report.construct_standard_report(
    {'DS1': results_std, 'DS2': results_std2},
    title="Example Multi-Dataset Report", verbosity=1
).write_html("../../../tutorial_files/exampleMultiDataSetReport", auto_open=False, verbosity=1)
```

## Reports from LGST alone

Reports aren't restricted to long-sequence GST.  *Linear* GST (LGST) takes substantially less data and computation time, so when a rough estimate of your gates is all you're after, it's worth knowing that its output feeds the same report machinery.  The experiment design below uses `max_max_length=1`, which is all LGST requires.

This workflow differs from the ones above in how the data arrives: instead of pairing an existing `DataSet` with an experiment design, write an empty data directory from the experiment design, fill it in (standing in for actually collecting data), read the completed directory back, then run the protocol.

```{code-cell} ipython3
#Get experiment design (for now, just max_max_length=1 GST sequences)
exp_design = smq1Q_XYI.create_gst_experiment_design(max_max_length=1)
pygsti.io.write_empty_protocol_data("../../../example_files/lgst_only_example", exp_design, clobber_ok=True)
print("Only %d sequences are required!" % len(exp_design.all_circuits_needing_data))

#Simulate taking the data (here you'd really fill in dataset.txt with actual data)
mdl_datagen = smq1Q_XYI.target_model().depolarize(op_noise=0.1, spam_noise=0.001)
pygsti.io.fill_in_empty_dataset_with_fake_data("../../../example_files/lgst_only_example/data/dataset.txt",
                                               mdl_datagen, num_samples=1000, seed=2020)

#load in the data
lgst_data = pygsti.io.read_data_from_dir("../../../example_files/lgst_only_example")
```

```{code-cell} ipython3
#Run LGST on the data written above.
results_lgst = pygsti.protocols.LGST(smq1Q_XYI.target_model()).run(lgst_data)
```

```{code-cell} ipython3
pygsti.report.construct_standard_report(
    results_lgst, title="LGST-only Example Report", verbosity=2
).write_html('../../../example_files/LGSTonlyReport', auto_open=False, verbosity=2)
```

Open [../../../example_files/LGSTonlyReport/main.html](../../../example_files/LGSTonlyReport/main.html) in your browser to view that report.

## Other `Report` tricks

A few additional arguments to the `Report` output methods give further control over what ends up in the generated report.

- Setting the `link_to` argument to a tuple of `'pkl'`, `'tex'`, and/or `'pdf'` creates hyperlinks within the plots or below the tables of the HTML, pointing at Python pickle, LaTeX source, and PDF versions of the content.  The pickle files for tables contain pickled pandas `DataFrame` objects; those for plots contain ordinary Python dictionaries of the plotted data.  Applies to HTML reports only.

- Setting the `brevity` argument to an integer higher than $0$ (the default) reduces the amount of information included in the report (for what's included at each value, see the doc string).  Using `brevity > 0` cuts the time required to create, and later load, the report, along with the output file/folder size.  This applies to both HTML and PDF reports.

Below we demonstrate both options in a very brief (`brevity=4`) report with links to pickle and LaTeX files.  Note that generating `'pdf'` links requires `pdflatex`.

```{code-cell} ipython3
pygsti.report.construct_standard_report(
    results_std, title="Example Brief Report", verbosity=1
).write_html("../../../tutorial_files/exampleBriefReport", auto_open=False, verbosity=1,
             brevity=4, link_to=('pkl', 'tex'))
```

## Report notebooks: `Report.write_notebook`

Besides the standard HTML-page reports demonstrated above, pyGSTi can generate a Jupyter notebook containing the Python commands that create the figures and tables of a general report.  `Workspace` objects, being factories for figures and tables, make this possible.  Calling `Report.write_notebook` dumps all of the relevant `Workspace` initialization and calls to a new notebook file, which you can then run fully or partially at your convenience.  The advantage is that you can insert Python code amidst the figure and table generation calls to inspect or modify what's displayed.  The disadvantages: report notebooks require a running Jupyter server, and nothing is displayed until you run the notebook.

```{warning}
Interactive cells in report notebooks require JavaScript, and therefore do not work with JupyterLab.  To track this issue, see https://github.com/pyGSTio/pyGSTi/issues/205.
```

The line below creates a report notebook with `write_notebook`.  The argument list is very similar to the other `Report` output methods.

```{code-cell} ipython3
pygsti.report.construct_standard_report(
    results, title="GST Example Report Notebook", confidence_level=None, verbosity=3
).write_notebook("../../../tutorial_files/exampleReport.ipynb", auto_open=False, connected=False, verbosity=3)
```

## Multi-qubit reports

The density matrix space of more than 2 qubits gets quite large, and models for 3+ qubits rarely let every element of the operation process matrices vary independently.  Many of the figures generated by `construct_standard_report` are therefore both unwieldy (a $64 \times 64$ grid of colored boxes for each operation) and unhelpful (you don't often care what each element of an operation matrix is).  For this case we are developing a report that doesn't just dump out and analyze operation matrices as a whole, but looks at a `Model`'s structure to decide how best to report quantities.  This "n-qubit report" is invoked using `pygsti.report.construct_nqnoise_report`, and takes arguments similar to `construct_standard_report`.  It is, however, <b style="color:red">still under development</b>, and while you're welcome to try it out, it may crash or fail in other weird ways.

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

# Introduction to `Workspace` objects
This tutorial briefly explains how pyGSTi visualizes data using `Workspace` objects.  Understanding the basics covered here will help demystify the creation of plots and tables in other tutorials.

PyGSTi prefers to use HTML-based figures (native HTML tables and [Plotly](https://plot.ly/python/) plots) over the perhaps more traditional LaTeX tables and [matplotlib](https://matplotlib.org) plots.  There are several reasons for this:
1. Interactivity - HTML allows plots and tables to be interactive; with LaTeX this is impossible and with matplotlib it's painful.
2. HTML's ability to be integrated into web pages (making nicer reports than a many-page PDF) and into Jupyter notebooks.
3. Portability - Plotly figures (HTML and JS) can be more robustly stored and transported (e.g. over the web) than matplotlib `Figure` objects, which are difficult even to pickle with Python.

The creation of (HTML) figures, both tables and plots, is handled by the `pygsti.report.Workspace` factory object.

```{code-cell} ipython3
import pygsti
ws = pygsti.report.Workspace()
```

Within an IPython notebook like this one, we can create figures in notebook cells by calling (once, usually at the beginning of the notebook):

```{code-cell} ipython3
ws.init_notebook_mode(autodisplay=True)
```

This injects necessary HTML and JavaScript into the notebook so that plots and tables display properly.  If everything works properly, you'll see a GREEN "**Notebook Initialization Complete**" message.  If instead you see a BLUE "**Loading...**" message, then 1) check that this notebook is "Trusted" in the upper right corner of this window and 2) check that you have a working internet connection.  You will need to reload this notebook using your *browser's* reload button after fixing either of these issues.

Setting `autodisplay=True` means that figures will be displayed as soon as they're created (otherwise we'd have to capture the returned object and call `.display()` on it).  By typing `ws.` and then hitting TAB you can see the somewhat-descriptive names of the figures that can be created.  Here are a few examples (for more, see the [Workspace examples tutorial](WorkspaceExamples.ipynb)):

```{code-cell} ipython3
import numpy as np
ws.MatrixPlot( np.array([[1,2],[3,4]],'d'), color_min=0, color_max=4 )
```

```{code-cell} ipython3
from pygsti.modelpacks import smq1Q_XYI
ws.GatesTable( smq1Q_XYI.target_model() )
```

Thats covers the basics!  The [Workspace examples tutorial](WorkspaceExamples.ipynb) shows a **gallery** of many of the tables and plots a `Workspace` can create, and the [Workspace switchboard tutorial](advanced/WorkspaceSwitchboards.ipynb) shows how to integrate workspace figures with *switches* (dropdown boxes, buttons, and sliders).  `Workspace` objects are used internally when generating HTML reports. The [report generation tutorial](ReportGeneration.ipynb) demonstrates how the automated use of a `Workspace` can lead to a standalone HTML report.

+++

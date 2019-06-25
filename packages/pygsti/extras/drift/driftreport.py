""" Drift reporting and plotting functions """
from __future__ import division, print_function, absolute_import, unicode_literals

import time as _time
import numpy as _np
import itertools as _itertools
import collections as _collections
import warnings as _warnings
import os as _os

from ... import _version
from ...baseobjs import VerbosityPrinter as _VerbosityPrinter
from ...objects import Circuit as _Circuit
from ...objects import DataComparator as _DataComparator
from ...report import workspace as _ws
from ...report import workspaceplots as _wp
from ...report import table as _reporttable
from ...report import figure as _reportfigure
from ...report import merge_helpers as _merge
from ...report import autotitle as _autotitle
from ...tools import timed_block as _timed_block

import plotly.graph_objs as go


class DriftSampleTable(_ws.WorkspaceTable):
    """
    TODO: docstrings in this entire module
    """

    def __init__(self, ws, item1, item2):
        super(DriftSampleTable, self).__init__(
            ws, self._create, item1, item2)

    def _create(self, item1, item2):
        colHeadings = ['Column 1', 'Column 2', ]
        table = _reporttable.ReportTable(colHeadings, (None,) * len(colHeadings))
        table.addrow([item1, item2], [None,None])
        table.finish()
        return table


class DriftSamplePlot(_ws.WorkspacePlot):
    def __init__(self, ws, title):
        super(DriftSamplePlot, self).__init__(ws, self._create, title)

    def _create(self, title):
        traces = []
        traces.append(go.Scatter(
            x=[0, 1, 2],
            y=[1.0, 2.0, 1.0],
            name='data'))

        layout = go.Layout(
            width=700,
            height=400,
            title=title,
            font=dict(size=10),
            xaxis=dict(title="X label"),
            yaxis=dict(title="Y label"),
        )

        pythonVal = {} # metadata
        return _reportfigure.ReportFigure(
            go.Figure(data=traces, layout=layout),
            None, pythonVal)


#Note: SAME function as in report/factory.py (copied)
def _add_new_labels(running_lbls, current_lbls):
    """
    Simple routine to add current-labels to a list of
    running-labels without introducing duplicates and
    preserving order as best we can.
    """
    if running_lbls is None:
        return current_lbls[:]  # copy!
    elif running_lbls != current_lbls:
        for lbl in current_lbls:
            if lbl not in running_lbls:
                running_lbls.append(lbl)
    return running_lbls


def _create_switchboard(ws, results_dict):
    """
    Creates the switchboard used by the drift report
    """

    if isinstance(results_dict, _collections.OrderedDict):
        dataset_labels = list(results_dict.keys())
    else:
        dataset_labels = sorted(list(results_dict.keys()))

    multidataset = bool(len(dataset_labels) > 1)

    switchBd = ws.Switchboard(
        ["Dataset"],
        [dataset_labels],
        ["dropdown"], [0],
        show=[multidataset]  # only show dataset dropdown (for sidebar)
    )

    switchBd.add("results", (0,))
    for d, dslbl in enumerate(dataset_labels):
        switchBd.results[d] = results_dict[dslbl]

    return switchBd, dataset_labels


def create_drift_report(results, filename, title="auto",
                        ws=None, auto_open=False, link_to=None,
                        brevity=0, advancedOptions=None, verbosity=1):
    """
    Creates a Drift report.
    """
    tStart = _time.time()
    printer = _VerbosityPrinter.build_printer(verbosity)  # , comm=comm)

    if advancedOptions is None: advancedOptions = {}
    precision = advancedOptions.get('precision', None)
    cachefile = advancedOptions.get('cachefile', None)
    connected = advancedOptions.get('connected', False)
    resizable = advancedOptions.get('resizable', True)
    autosize = advancedOptions.get('autosize', 'initial')
    mdl_sim = advancedOptions.get('simulator', None)  # a model

    if filename and filename.endswith(".pdf"):
        fmt = "latex"
    else:
        fmt = "html"

    printer.log('*** Creating workspace ***')
    if ws is None: ws = _ws.Workspace(cachefile)

    if title is None or title == "auto":
        if filename is not None:
            autoname = _autotitle.generate_name()
            title = "Drift Report for " + autoname
            _warnings.warn(("You should really specify `title=` when generating reports,"
                            " as this makes it much easier to identify them later on.  "
                            "Since you didn't, pyGSTi has generated a random one"
                            " for you: '{}'.").format(autoname))
        else:
            title = "N/A"  # No title - but it doesn't matter since filename is None

    results_dict = results if isinstance(results, dict) else {"unique": results}

    renderMath = True

    qtys = {}  # stores strings to be inserted into report template

    def addqty(b, name, fn, *args, **kwargs):
        """Adds an item to the qtys dict within a timed block"""
        if b is None or brevity < b:
            with _timed_block(name, formatStr='{:45}', printer=printer, verbosity=2):
                qtys[name] = fn(*args, **kwargs)

    qtys['title'] = title
    qtys['date'] = _time.strftime("%B %d, %Y")

    pdfInfo = [('Author', 'pyGSTi'), ('Title', title),
               ('Keywords', 'GST'), ('pyGSTi Version', _version.__version__)]
    qtys['pdfinfo'] = _merge.to_pdfinfo(pdfInfo)

    # Generate Switchboard
    printer.log("*** Generating switchboard ***")

    #Create master switchboard
    switchBd, dataset_labels = \
        _create_switchboard(ws, results_dict)
    if fmt == "latex" and (len(dataset_labels) > 1):
        raise ValueError("PDF reports can only show a *single* dataset,"
                         " estimate, and gauge optimization.")

    # Generate Tables
    printer.log("*** Generating tables ***")

    if fmt == "html":
        qtys['topSwitchboard'] = switchBd

    results = switchBd.results
    A = None  # no brevity restriction: always display

    #ADD TABLES HERE
    addqty(A, 'sampleTable', ws.DriftSampleTable, "Item A", "Item B")


    # Generate plots
    printer.log("*** Generating plots ***")

    #ADD PLOTS HERE
    addqty(A, 'samplePlot', ws.DriftSamplePlot, "Title of Plot")

    toggles = {}
    toggles['CompareDatasets'] = False  # not comparable by default

    if filename is not None:
        if True:  # comm is None or comm.Get_rank() == 0:
            # 3) populate template file => report file
            printer.log("*** Merging into template file ***")

            if fmt == "html":
                templateDir = "drift_html_report"
                _merge.merge_html_template_dir(
                    qtys, templateDir, filename, auto_open, precision, link_to,
                    connected=connected, toggles=toggles, renderMath=renderMath,
                    resizable=resizable, autosize=autosize, verbosity=printer)

            elif fmt == "latex":
                raise NotImplementedError("No PDF version of this report is available yet.")
                templateFile = "drift_pdf_report.tex"
                base = _os.path.splitext(filename)[0]  # no extension
                _merge.merge_latex_template(qtys, templateFile, base + ".tex", toggles,
                                            precision, printer)

                # compile report latex file into PDF
                cmd = _ws.WorkspaceOutput.default_render_options.get('latex_cmd', None)
                flags = _ws.WorkspaceOutput.default_render_options.get('latex_flags', [])
                assert(cmd), "Cannot render PDF documents: no `latex_cmd` render option."
                printer.log("Latex file(s) successfully generated.  Attempting to compile with %s..." % cmd)
                _merge.compile_latex_report(base, [cmd] + flags, printer, auto_open)
            else:
                raise ValueError("Unrecognized format: %s" % fmt)
    else:
        printer.log("*** NOT Merging into template file (filename is None) ***")
    printer.log("*** Report Generation Complete!  Total time %gs ***" % (_time.time() - tStart))

    return ws

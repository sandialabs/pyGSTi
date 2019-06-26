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

from . import signal as _sig

import plotly.graph_objs as go
import seaborn as _sns


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


class PowerSpectraPlot(_ws.WorkspacePlot):
    """ Plot of time-series data power spectrum """

    def __init__(self, ws, stabilityanalyzer, spectrumlabel={}, detectorkey=None,
                 showlegend=False, title=None, scale=1.0):
        """
        Plot RB decay curve, as a function of sequence length.  Optionally
        includes a fitted exponential decay.

        Parameters
        ----------

        Returns
        -------
        None
        """
        super(PowerSpectraPlot, self).__init__(ws, self._create, stabilityanalyzer,
                                               spectrumlabel, detectorkey, showlegend, title, scale)

    def _create(self, stabilityanalyzer, spectrumlabel, detectorkey, showlegend, title, scale):

        circuits = spectrumlabel.get('circuit', None)

        # If we're plotting spectra for more than one circuit.
        if isinstance(circuits, dict) or isinstance(circuits, list):

            threshold, thresholdtype = stabilityanalyzer.get_power_threshold(test=tuple(spectrumlabel.keys()), detectorkey=detectorkey)
            data = []
            ymax = threshold
            xmax = 0

            if isinstance(circuits, list):
                circuits = {c.str: c for c in circuits}

            colors = ['rgb' + str(tuple(i)) for i in _sns.color_palette("coolwarm", len(circuits))]

            for ind, (circlabel, circ) in enumerate(circuits.items()):

                spectrumlabel['circuit'] = circ
                freqs, powers = stabilityanalyzer.get_spectrum(spectrumlabel, returnfrequencies=True, checklevel=2)

                xdata = _np.array(freqs)
                ydata = _np.array(powers)

                insig_xdata = xdata[ydata <= threshold]
                insig_ydata = ydata[ydata <= threshold]
                sig_xdata = xdata[ydata > threshold]
                sig_ydata = ydata[ydata > threshold]

                xmax = max(max(xdata), xmax)
                ymax = max(max(ydata), ymax)

                data.append(go.Scatter(x=insig_xdata, y=insig_ydata, mode='markers', marker=dict(color=colors[ind], size=4), name=circlabel, showlegend=showlegend))
                data.append(go.Scatter(x=sig_xdata, y=sig_ydata, mode='markers', marker=dict(color=colors[ind], size=8),
                                       name=circlabel, showlegend=False))

        # If we're plotting a single spectrum.
        else:

            freqs, powers = stabilityanalyzer.get_spectrum(spectrumlabel, returnfrequencies=True, checklevel=2)
            threshold, thresholdtype = stabilityanalyzer.get_power_threshold(test=tuple(spectrumlabel.keys()), detectorkey=detectorkey)

            xdata = _np.array(freqs)
            ydata = _np.array(powers)

            insig_xdata = xdata[ydata <= threshold]
            insig_ydata = ydata[ydata <= threshold]
            sig_xdata = xdata[ydata > threshold]
            sig_ydata = ydata[ydata > threshold]

            data = []  # list of traces
            data.append(go.Scatter(x=insig_xdata, y=insig_ydata, mode='markers', marker=dict(color="#2ecc71", size=4),
                                   name='Insignificant Data', showlegend=showlegend))
            data.append(go.Scatter(x=sig_xdata, y=sig_ydata, mode='markers', marker=dict(color='#2ecc71', size=8),
                                   name='Significant Data', showlegend=showlegend))

            xmax = max(xdata)
            ymax = max(max(ydata), threshold)

        ylim = [0, ymax * 1.1]
        xlim = [-0.05 * xmax, xmax * 1.05]

        text = go.Scatter(x=[0.85*(xlim[1]-xlim[0]) + xlim[0], 0.85*(xlim[1]-xlim[0]) + xlim[0]],
                          y=[threshold + 0.05*(ylim[1]-ylim[0]) + ylim[0], 1 - 0.05*(ylim[1]-ylim[0]) + ylim[0]],
                          # Todo.
                         text=['{}% Significance Threshold'.format(stabilityanalyzer.get_statistical_significance(detectorkey)*100),
                               'Expected Shot-Noise Level'],
                         mode='text',
                         showlegend=False
                         )

        data.append(text)

        layout = go.Layout(width=800 * scale, height=400 * scale, title=title, titlefont=dict(size=16),
                           xaxis=dict(title="Frequency (Hz)", titlefont=dict(size=14), range=xlim,),
                           yaxis=dict(title="Spectral Power", titlefont=dict(size=14), range=ylim,),
                           legend=dict(
                    x=0.56,
                    y=1.1,
                    traceorder='normal',
                    font=dict(
                        size=10,
                        color='#000'
                    ),
                    bgcolor='#ecf0f1',
                    bordercolor='#bdc3c7',
                    borderwidth=2,
                    orientation="h"
                ),
                 shapes= [{
            'type': 'line',
            'x0': xlim[0],
            'y0': threshold,
            'x1': xlim[1],
            'y1': threshold,
            'line': {
                'color': '#3498db',
                'width': 2,
                'dash': 'dot',
                    },
             },
             {
            'type': 'line',
            'x0': xlim[0],
            'y0': 1,
            'x1': xlim[1],
            'y1': 1,
            'line': {
                'color': '#f1c40f',
                'width': 2,
                'dash': 'dashdot',
                    },
             },
            ],
            showlegend=showlegend, 
                           )

        pythonVal = {}
        for i, tr in enumerate(data):
            if 'x0' in tr: continue  # don't put boxes in python val for now
            key = tr['name'] if ("name" in tr) else "trace%d" % i
            pythonVal[key] = {'x': tr['x'], 'y': tr['y']}

        return  _reportfigure.ReportFigure(go.Figure(data=list(data), layout=layout), None, pythonVal)


class ProbTrajectoriesPlot(_ws.WorkspacePlot):
    """ Plot of time-series data power spectrum """

    def __init__(self, ws, stabilityanalyzer, circuits, outcome, times=None, dskey=None, estimatekey=None, estimator=None, showlegend=True, scale=1.0):
        """
        """
        super(ProbTrajectoriesPlot, self).__init__(ws, self._create, stabilityanalyzer, circuits, outcome,
                                                        times, dskey, estimatekey, estimator, showlegend, scale)

    def _create(self, stabilityanalyzer, circuits, outcome, times, dskey, estimatekey, estimator, showlegend, scale):

        # If we're plotting probability trajectories for multiple circuits.
        if isinstance(circuits, dict) or isinstance(circuits, list):

            if isinstance(circuits, list):
                circuits = {c.str: c for c in circuits}

            if dskey is None:
                assert(len(stabilityanalyzer.data.keys()) == 1), "There is more than one DataSet, so must specify the `dskey`!"
                dskey = list(stabilityanalyzer.data.keys())[0]

            colors = ['rgb' + str(tuple(i)) for i in _sns.color_palette("coolwarm", len(circuits))]
            data = []
            if times is None:
                mintime = min(stabilityanalyzer.data[dskey].timeData)
                maxtime = max(stabilityanalyzer.data[dskey].timeData)
                times = _np.linspace(mintime, maxtime, 5000)
            xdata = _np.asarray(times)

            for ind, (label, circuit) in enumerate(circuits.items()):
                probsdict = stabilityanalyzer.get_probability_trajectory(circuit, times, dskey, estimatekey, estimator)
                ydata = _np.asarray(probsdict[outcome])

                # list of traces
                print(label)
                data.append(go.Scatter(x=xdata, y=ydata, mode='lines', line=dict(width=2, color=colors[ind]),
                                       name=label, showlegend=True))

            ylim = [-0.1, 1.1]
            xlim = [min(xdata), max(xdata)]

            layout = go.Layout(width=800 * scale, height=400 * scale, title=None, titlefont=dict(size=16),
                               xaxis=dict(title="Time (seconds)", titlefont=dict(size=14), range=xlim, rangeslider=dict(visible = True)),
                               yaxis=dict(title="Probability", titlefont=dict(size=14), range=ylim),
                               legend=dict(
                    x=0.05,
                    y=1.05,
                    traceorder='normal',
                    font=dict(
                        size=10,
                        color='#000'
                    ),
                    bgcolor='#ecf0f1',
                    bordercolor='#bdc3c7',
                    borderwidth=2,
                    orientation="h"
                ), showlegend=showlegend)

        # If we're plotting probability trajectories for a single circuit.
        else:

            circuit = circuits

            if dskey is None:
                assert(len(stabilityanalyzer.data.keys()) == 1), "There is more than one DataSet, so must specify the `dskey`!"
                dskey = list(stabilityanalyzer.data.keys())[0]
            dtimes, data = stabilityanalyzer.data[dskey][circuit].get_timeseries_for_outcomes()
            if times is None:
                times = _np.linspace(min(dtimes), max(dtimes), 5000)
            p = stabilityanalyzer.get_probability_trajectory(circuit, times=times, dskey=dskey, estimatekey=estimatekey, estimator=estimator)[outcome]
            lowpass = _sig.moving_average(data[outcome], width=100)

            trace_pt = go.Scatter(x=times, y=p, name="Probability Trajectory", line=dict(color='#e74c3c'),
                                  opacity=1.)
            trace_lowpass = go.Scatter(x=dtimes, y=lowpass, name="Moving average", line=dict(color='#7F7F7F'),
                                       opacity=0.8)

            data = [trace_pt, trace_lowpass]

            updatemenus = list([
                dict(active=0,
                     buttons=list([dict(label='Probability trajectory',
                                        method='update',
                                        args=[{'visible': [True, False]}, ]),
                                   dict(label='Moving average',
                                        method='update',
                                        args=[{'visible': [False, True]}, ]),
                                   dict(label='Both',
                                        method='update',
                                        args=[{'visible': [True, True]},]),
                                   ]),
                     xanchor='left',
                     yanchor='top',
                     x=0.02,
                     y = 1.2, #y=0.98,
                     showactive=True
                     )
            ])

            layout = dict(width=800 * scale, height=500 * scale,
                #title='Probability Trajectory',
                xaxis=dict(title="Time (seconds)",
                    rangeslider=dict(visible = True),
                ),
                yaxis=dict(title="Probability", titlefont=dict(size=14),range=[0,1]), 
                updatemenus=updatemenus,
                legend=dict(
                    x=0.5,
                    y=1.05,
                    traceorder='normal',
                    font=dict(
                        size=12,
                        color='#000'
                    ),
                    bgcolor='#ecf0f1',
                    bordercolor='#bdc3c7',
                    borderwidth=2,
                    orientation="h"
                ), 
                showlegend=showlegend
            )

        pythonVal = {}
        for i, tr in enumerate(data):
            if 'x0' in tr: continue  # don't put boxes in python val for now
            key = tr['name'] if ("name" in tr) else "trace%d" % i
            pythonVal[key] = {'x': tr['x'], 'y': tr['y']}

        return _reportfigure.ReportFigure(go.Figure(data=list(data), layout=layout), None, pythonVal)

# class GermProbTrajectoriesPlot(_ws.WorkspacePlot):

#     def __init__(self, ws, stabilityanalyzer, gss, germ, outcome, minL=1, times=None, dskey=None, estimatekey=None,
#                  estimator=None, showlegend=False, scale=1.0):
#         """
#         gss : CircuitStructure
#             Specifies the set of operation sequences along with their structure, e.g. fiducials, germs,
#             and maximum lengths.
#         """
#         super(GermProbTrajectoriesPlot, self).__init__(ws, self._create, stabilityanalyzer, gss, germ, outcome, minL, times,
#                                                        dskey, estimatekey, estimator, showlegend, scale)

#     def _create(self, stabilityanalyzer, gss, germ, outcome, minL, times, dskey, estimatekey, estimator, showlegend, scale):

#         if isinstance(germ, tuple):
#             germ = _Circuit(germ)
#         elif isinstance(germ, str):
#             germ = _Circuit(None, stringrep=germ)

#         if dskey is None:
#             assert(len(stabilityanalyzer.data.keys()) == 1), "There is more than one DataSet, so must specify the `dskey`!"
#             dskey = list(stabilityanalyzer.data.keys())[0]

#         if times is None:
#             mintime = min(stabilityanalyzer.data[dskey].timeData)
#             maxtime = max(stabilityanalyzer.data[dskey].timeData)
#             times = _np.linspace(mintime, maxtime, 1000)

#         prepind = []
#         measind = []
#         datapt = []

#         truncatedL = []
#         for L in gss.Ls:
#             if L >= minL:
#                 truncatedL.append(L)

#         numL = len(gss.Ls)
#         #colors = ['rgb' + str(tuple(i)) for i in _sns.cubehelix_palette(n_colors=numL, rot=-.4)]
#         colors = ['rgb' + str(tuple(i)) for i in _sns.color_palette("coolwarm", numL )]

#         for Lind, L in enumerate(truncatedL):
#             for j, k, circuit in gss.get_plaquette(L, germ):

#                 prepind.append(j)
#                 measind.append(k)
#                 p = stabilityanalyzer.get_probability_trajectory(circuit, times=times, dskey=dskey,
#                                                                  estimatekey=estimatekey,
#                                                                  estimator=estimator)[outcome]

#                 trace_pt = go.Scatter(x=times, y=p, name="{}".format(L), line=dict(color=colors[Lind], width=2),
#                                       opacity=1.)

#                 datapt.append(trace_pt)

#         data = datapt
#         prepind = _np.array(prepind)
#         measind = _np.array(measind)

#         updatemenus = list([
#             dict(active=1,
#                  buttons=list([dict(label=str(gss.prepStrs[i].str) + '()' + str(gss.effectStrs[j].str),
#                                     method='update',
#                                     args=[{'visible': [x and y for x, y in zip(prepind == i, measind == j)]}])
#                                           for i, j in _itertools.product(range(len(gss.prepStrs)), range(len(gss.effectStrs)))]),
#                  xanchor='left',
#                  yanchor='top',
#                  x=0.02,
#                  y = 1.2,
#                  showactive=True
#                  )
#         ])

#         layout = dict(width=800 * scale, height=500 * scale,
#                       #title='Probability Trajectory for the {} germ'.format(germ.str),
#                        xaxis=dict(title="Time (seconds)",
#                                   rangeslider=dict(visible = True),
#             ),
#             yaxis=dict(title="Probability for {}".format(outcome), titlefont=dict(size=14),range=[0,1]), 
#             updatemenus=updatemenus,
#             legend=dict(
#                 x=0.5,
#                 y=1.05,
#                 traceorder='normal',
#                 font=dict(
#                     size=10,
#                     color='#000'
#                 ),
#                 bgcolor='#ecf0f1',
#                 bordercolor='#bdc3c7',
#                 borderwidth=2,
#                 orientation="h"
#             ),
#             showlegend=showlegend
#         )

#         pythonVal = {}
#         for i, tr in enumerate(data):
#             if 'x0' in tr: continue  # don't put boxes in python val for now
#             key = tr['name'] if ("name" in tr) else "trace%d" % i
#             pythonVal[key] = {'x': tr['x'], 'y': tr['y']}

#         return _reportfigure.ReportFigure(go.Figure(data=list(data), layout=layout), None, pythonVal)


class GermFiducialProbTrajectoriesPlot(_ws.WorkspacePlot):

    def __init__(self, ws, stabilityanalyzer, gss, prep, germ, meas, outcome, minL=1, times=None, dskey=None, estimatekey=None,
                 estimator=None, showlegend=False, scale=1.0):
        """
        gss : CircuitStructure
            Specifies the set of operation sequences along with their structure, e.g. fiducials, germs,
            and maximum lengths.
        """
        super(GermFiducialProbTrajectoriesPlot, self).__init__(ws, self._create, stabilityanalyzer, gss, prep, germ, meas, outcome, minL, times,
                                                       dskey, estimatekey, estimator, showlegend, scale)

    def _create(self, stabilityanalyzer, gss, prep, germ, meas, outcome, minL, times, dskey, estimatekey, estimator, showlegend, scale):

        if isinstance(germ, str):
            germ = _Circuit(None, stringrep=germ)
        if isinstance(prep, str):
            prep = _Circuit(None, stringrep=prep)
        if isinstance(meas, str):
            meas = _Circuit(None, stringrep=meas)

        if dskey is None:
            assert(len(stabilityanalyzer.data.keys()) == 1), "There is more than one DataSet, so must specify the `dskey`!"
            dskey = list(stabilityanalyzer.data.keys())[0]

        if times is None:
            mintime = min(stabilityanalyzer.data[dskey].timeData)
            maxtime = max(stabilityanalyzer.data[dskey].timeData)
            times = _np.linspace(mintime, maxtime, 1000)

        prepind = gss.prepStrs.index(prep)
        measind = gss.prepStrs.index(meas)
        datapt = []

        truncatedL = []
        for L in gss.Ls:
            if L >= minL:
                truncatedL.append(L)

        numL = len(gss.Ls)
        colors = ['rgb' + str(tuple(i)) for i in _sns.color_palette("coolwarm", numL)]

        for Lind, L in enumerate(gss.Ls):
            if L >= minL:
                for j, k, circuit in gss.get_plaquette(L, germ):               
                    if j == prepind:
                        if k == measind:
                            p = stabilityanalyzer.get_probability_trajectory(circuit, times=times, dskey=dskey,
                                                                     estimatekey=estimatekey,
                                                                     estimator=estimator)[outcome]

                            trace_pt = go.Scatter(x=times, y=p, name="{}".format(L), line=dict(color=colors[Lind], width=2),
                                      opacity=1.)

                datapt.append(trace_pt)

        data = datapt

        layout = dict(width=800 * scale, height=500 * scale,
                      #title='Probability Trajectory for the {} germ'.format(germ.str),
                       xaxis=dict(title="Time (seconds)",
                                  rangeslider=dict(visible = True),
            ),
            yaxis=dict(title="Probability for {}".format(outcome), titlefont=dict(size=14),range=[0,1]), 
            legend=dict(
                x=0.5,
                y=1.05,
                traceorder='normal',
                font=dict(
                    size=10,
                    color='#000'
                ),
                bgcolor='#ecf0f1',
                bordercolor='#bdc3c7',
                borderwidth=2,
                orientation="h"
            ),
            showlegend=showlegend
        )

        pythonVal = {}
        for i, tr in enumerate(data):
            if 'x0' in tr: continue  # don't put boxes in python val for now
            key = tr['name'] if ("name" in tr) else "trace%d" % i
            pythonVal[key] = {'x': tr['x'], 'y': tr['y']}

        return _reportfigure.ReportFigure(go.Figure(data=list(data), layout=layout), None, pythonVal)


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

def _create_drift_switchboard(ws, results, gss):

    if len(results.data.keys()) > 1:  # multidataset
        drift_switchBd = ws.Switchboard(
            ["Dataset", "Germ", "Preperation Fiducial", "Measurement Fiducial", "Outcome"], [list(results.data.keys()), [c.str for c in gss.germs], [c.str for c in(gss.prepStrs)], 
                                                                    [c.str for c in gss.effectStrs],
                                                                 [i.str for i in results.data.get_outcome_labels()]],
            ["dropdown", "dropdown", "dropdown", "dropdown", "dropdown"], [0, 0, 0, 0, 0], show=[True, True, True, True, True])
        drift_switchBd.add("dataset", (0,))
        drift_switchBd.add("germ", (1,))
        drift_switchBd.add("prep", (2,))
        drift_switchBd.add("meas", (3,))
        drift_switchBd.add("outcome", (4,))

    else:
        drift_switchBd = ws.Switchboard(
            ["Germ", "Preperation Fiducial", "Measurement Fiducial", "Outcome"], [[c.str for c in gss.germs], [c.str for c in(gss.prepStrs)], 
            [c.str for c in gss.effectStrs], [str(o) for o in results.data.get_outcome_labels()]],
            ["dropdown", "dropdown", "dropdown", "dropdown"], [0, 0, 0, 0], show=[True, True, True, True])
        drift_switchBd.add("germ", (0,))
        drift_switchBd.add("prep", (1,))
        drift_switchBd.add("meas", (2,))
        drift_switchBd.add("outcome", (3,))

        drift_switchBd.germs = gss.germs
        drift_switchBd.prep = gss.prepStrs
        drift_switchBd.meas = gss.effectStrs
        drift_switchBd.outcome = results.data.get_outcome_labels()

    return drift_switchBd


def create_drift_report(results, gss, filename, title="auto",
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

    drift_switchBd = _create_drift_switchboard(ws, results, gss)
    qtys = {}  # stores strings to be inserted into report template
    qtys['drift_switchBd'] = drift_switchBd

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
    #addqty(A, 'driftPvalueBoxPlot', ws.ColorBoxPlot, 'driftdetector', gss, None, None, stabilityanalyzer=results)
    addqty(A, 'driftdetectorColorBoxPlot', ws.ColorBoxPlot, 'driftdetector', gss, None, None, False, False, True, False,
           'compact', .05, 1e-4, None, None, results)
    addqty(A, 'driftsizeColorBoxPlot', ws.ColorBoxPlot, 'driftsize', gss, None, None, False, False, True, False,
           'compact', .05, 1e-4, None, None, results)
    #addqty(A, 'samplePlot', ws.DriftSamplePlot, "Title of Plot")
    germ = gss.germs[0]
    if len(results.data.keys()) > 1:
        dskeysetter = drift_switchBd.dataset
    else:
        dskeysetter = None
    addqty(A, 'GermFiducialProbTrajectoriesPlot', GermFiducialProbTrajectoriesPlot, results, gss, drift_switchBd.prep, drift_switchBd.germ, drift_switchBd.meas, 
        drift_switchBd.outcome, 1,)

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

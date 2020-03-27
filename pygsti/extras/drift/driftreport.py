#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************
""" Drift reporting and plotting functions """

import time as _time
import numpy as _np
import itertools as _itertools
import collections as _collections
import warnings as _warnings
import os as _os

from ... import _version
from ...objects.verbosityprinter import VerbosityPrinter as _VerbosityPrinter
from ...objects import Circuit as _Circuit
from ...objects import DataComparator as _DataComparator
from ...report import workspace as _ws
from ...report import workspaceplots as _wp
from ...report import table as _reporttable
from ...report import figure as _reportfigure
from ...report import merge_helpers as _merge
from ...report import autotitle as _autotitle
from ...report import colormaps as _cmaps
from ...tools import timed_block as _timed_block

from . import signal as _sig
from . import stabilityanalyzer as _sa

import plotly.graph_objs as go
#import seaborn as _sns

#We don't want to import seaborn just for a colorscale, so pulled this matplotlib source (derived from _cm.py):
plotly_coolwarm_colorscale = [
    (0.0, 'rgb(58,76,192)'), (0.03125, 'rgb(68,90,205)'), (0.0625, 'rgb(77,104,216)'),
    (0.09375, 'rgb(87,117,226)'), (0.125, 'rgb(98,130,234)'), (0.15625, 'rgb(108,142,242)'),
    (0.1875, 'rgb(119,154,247)'), (0.21875, 'rgb(130,165,252)'), (0.25, 'rgb(141,176,254)'),
    (0.28125, 'rgb(152,185,255)'), (0.3125, 'rgb(163,194,255)'), (0.34375, 'rgb(174,201,253)'),
    (0.375, 'rgb(184,208,250)'), (0.40625, 'rgb(194,213,244)'), (0.4375, 'rgb(204,217,238)'),
    (0.46875, 'rgb(213,220,230)'), (0.5, 'rgb(221,221,221)'), (0.53125, 'rgb(229,217,210)'),
    (0.5625, 'rgb(236,211,198)'), (0.59375, 'rgb(241,205,186)'), (0.625, 'rgb(245,197,173)'),
    (0.65625, 'rgb(247,187,160)'), (0.6875, 'rgb(248,177,148)'), (0.71875, 'rgb(247,166,135)'),
    (0.75, 'rgb(245,154,123)'), (0.78125, 'rgb(241,141,111)'), (0.8125, 'rgb(236,127,99)'),
    (0.84375, 'rgb(230,112,87)'), (0.875, 'rgb(222,96,76)'), (0.90625, 'rgb(213,80,66)'),
    (0.9375, 'rgb(203,61,56)'), (0.96875, 'rgb(192,40,47)'), (1.0, 'rgb(180,3,38)')]


class DriftSummaryTable(_ws.WorkspaceTable):
    """
    todo
    """

    def __init__(self, ws, stabilityanalyzer, dskey=None, detectorkey=None, estimatekey=None):
        """
        todo
        """
        super(DriftSummaryTable, self).__init__(ws, self._create, stabilityanalyzer, dskey, detectorkey, estimatekey)

    def _create(self, stabilityanalyzer, dskey, detectorkey, estimatekey):
        colHeadings = ['', '', ]
        table = _reporttable.ReportTable(colHeadings, (None,) * len(colHeadings))
        table.addrow(['Global statistical significance level',
                      stabilityanalyzer.get_statistical_significance(detectorkey=detectorkey)], [None, None])
        table.addrow(['Instability detected', stabilityanalyzer.instability_detected(
            detectorkey=detectorkey)], [None, None])
        table.addrow(['Instability size', stabilityanalyzer.get_maxmax_tvd_bound(
            dskey=dskey, estimatekey=estimatekey)], [None, None])
        table.finish()
        return table


class DriftDetailsTable(_ws.WorkspaceTable):
    """
    todo
    """

    def __init__(self, ws, stabilityanalyzer, detectorkey=None, estimatekey=None):
        """
        todo
        """
        super(DriftDetailsTable, self).__init__(ws, self._create, stabilityanalyzer, detectorkey, estimatekey)

    def _create(self, stabilityanalyzer, detectorkey, estimatekey):
        if detectorkey is None:
            detectorkey = stabilityanalyzer._def_detection
        if estimatekey is None:
            estimatekey = stabilityanalyzer._def_probtrajectories
        colHeadings = ['', '', ]
        table = _reporttable.ReportTable(colHeadings, (None,) * len(colHeadings))
        table.addrow(['Transform', stabilityanalyzer.transform], [None, None])
        table.addrow(['Single detector in the results', len(stabilityanalyzer._driftdetectors) == 1], [None, None])
        table.addrow(['Name of detector', detectorkey], [None, None])
        string_condtestsrun = ''
        for test in stabilityanalyzer._condtests[detectorkey]: string_condtestsrun += str(test) + ', '
        string_estimatekey = ''
        for detail in estimatekey: string_estimatekey += str(detail) + ', '
        table.addrow(['Tests run for detector', string_condtestsrun], [None, None])
        table.addrow(['Type of estimator', string_estimatekey], [None, None])
        table.finish()
        return table


class PowerSpectraPlot(_ws.WorkspacePlot):
    """
    Plot of time-series data power spectrum
    """

    def __init__(self, ws, stabilityanalyzer, spectrumlabel={}, detectorkey=None,
                 showlegend=False, scale=1.0):
        """
        todo
        """
        super(PowerSpectraPlot, self).__init__(ws, self._create, stabilityanalyzer,
                                               spectrumlabel, detectorkey, showlegend, scale)

    def _create(self, stabilityanalyzer, spectrumlabel, detectorkey, showlegend, scale):

        circuits = spectrumlabel.get('circuit', None)

        # If we're plotting spectra for more than one circuit.
        if isinstance(circuits, dict) or isinstance(circuits, list):

            threshold, thresholdtype = stabilityanalyzer.get_power_threshold(
                test=tuple(spectrumlabel.keys()), detectorkey=detectorkey)
            data = []
            ymax = threshold
            xmax = 0

            if isinstance(circuits, list):
                circuits = {c.str: c for c in circuits}

            #colors = ['rgb' + str(tuple(i)) for i in _sns.color_palette("coolwarm", len(circuits))]
            colors = [_cmaps.interpolate_plotly_colorscale(plotly_coolwarm_colorscale, x)
                      for x in _np.linspace(0.0, 1.0, len(circuits))]

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

                data.append(go.Scatter(x=insig_xdata, y=insig_ydata, mode='markers', marker=dict(
                    color=colors[ind], size=4), name=circlabel, showlegend=showlegend))
                data.append(go.Scatter(x=sig_xdata, y=sig_ydata, mode='markers', marker=dict(color=colors[ind], size=8),
                                       name=circlabel, showlegend=False))

        # If we're plotting a single spectrum.
        else:

            freqs, powers = stabilityanalyzer.get_spectrum(spectrumlabel, returnfrequencies=True, checklevel=2)
            threshold, thresholdtype = stabilityanalyzer.get_power_threshold(
                test=tuple(spectrumlabel.keys()), detectorkey=detectorkey)

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

        text = go.Scatter(x=[0.85 * (xlim[1] - xlim[0]) + xlim[0], 0.85 * (xlim[1] - xlim[0]) + xlim[0]],
                          y=[threshold + 0.05 * (ylim[1] - ylim[0]) + ylim[0],
                             1 - 0.05 * (ylim[1] - ylim[0]) + ylim[0]],
                          # Todo.
                          text=['{}% Significance Threshold'.format(
                              stabilityanalyzer.get_statistical_significance(detectorkey) * 100),
                              'Expected Shot-Noise Level'],
                          mode='text',
                          showlegend=False
                          )

        data.append(text)

        layout = go.Layout(width=800 * scale, height=400 * scale,
                           xaxis=dict(title="Frequency (Hz)", titlefont=dict(size=14), range=xlim,),
                           yaxis=dict(title="Spectral Power", titlefont=dict(size=14), range=ylim,),
                           legend=dict(
                               traceorder='normal',
                               font=dict(
                                   size=10,
                                   color='#000'
                               ),
                               bgcolor='#ecf0f1',
                               bordercolor='#bdc3c7',
                               borderwidth=2,
                               orientation="v"
                           ),
                           shapes=[{
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

        return _reportfigure.ReportFigure(go.Figure(data=list(data), layout=layout), None, pythonVal)


class GermFiducialPowerSpectraPlot(_ws.WorkspacePlot):
    """
    Plot of time-series data power spectrum
    """

    def __init__(self, ws, stabilityanalyzer, gss, prep, germ, meas, dskey=None, detectorkey=None,
                 showlegend=False, scale=1.0):
        """
        todo
        """
        super(GermFiducialPowerSpectraPlot, self).__init__(ws, self._create, stabilityanalyzer, gss, prep, germ, meas,
                                                           dskey, detectorkey, showlegend, scale)

    def _create(self, stabilityanalyzer, gss, prep, germ, meas, dskey, detectorkey, showlegend, scale):

        if isinstance(germ, str):
            germ = _Circuit(None, stringrep=germ)
        if isinstance(prep, str):
            prep = _Circuit(None, stringrep=prep)
        if isinstance(meas, str):
            meas = _Circuit(None, stringrep=meas)

        if dskey is None:
            assert(len(stabilityanalyzer.data.keys()) == 1), \
                "There is more than one DataSet, so must specify the `dskey`!"
            dskey = list(stabilityanalyzer.data.keys())[0]

        prepind = gss.prepStrs.index(prep)
        measind = gss.prepStrs.index(meas)
        circuitdict = {}

        #UNUSED: numL = len(gss.Ls)
        #UNUSED: colors = ['rgb' + str(tuple(i)) for i in _sns.color_palette("coolwarm", numL)]
        for Lind, L in enumerate(gss.Ls):
            for j, k, circuit in gss.get_plaquette(L, germ):
                if j == prepind:
                    if k == measind:
                        circuitdict[L] = circuit

        spectrumlabel = {'dataset': dskey, 'circuit': circuitdict}

        psp = PowerSpectraPlot(self.ws, stabilityanalyzer, spectrumlabel, detectorkey,
                               showlegend, scale)
        assert(len(psp.figs) == 1), "Only one figure should have been created!"
        return psp.figs[0]


class ProbTrajectoriesPlot(_ws.WorkspacePlot):
    """
    todo
    """

    def __init__(self, ws, stabilityanalyzer, circuits, outcome, times=None, dskey=None, estimatekey=None,
                 estimator=None, showlegend=True, scale=1.0):
        """
        todo
        """
        super(ProbTrajectoriesPlot, self).__init__(ws, self._create, stabilityanalyzer, circuits, outcome,
                                                   times, dskey, estimatekey, estimator, showlegend, scale)

    def _create(self, stabilityanalyzer, circuits, outcome, times, dskey, estimatekey, estimator, showlegend, scale):

        # If we're plotting probability trajectories for multiple circuits.
        if isinstance(circuits, dict) or isinstance(circuits, list):

            if isinstance(circuits, list):
                circuits = {c.str: c for c in circuits}

            if dskey is None:
                assert(len(stabilityanalyzer.data.keys()) == 1), \
                    "There is more than one DataSet, so must specify the `dskey`!"
                dskey = list(stabilityanalyzer.data.keys())[0]

            #colors = ['rgb' + str(tuple(i)) for i in _sns.color_palette("coolwarm", len(circuits))]
            colors = [_cmaps.interpolate_plotly_colorscale(plotly_coolwarm_colorscale, x)
                      for x in _np.linspace(0.0, 1.0, len(circuits))]

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
                data.append(go.Scatter(x=xdata, y=ydata, mode='lines', line=dict(width=2, color=colors[ind]),
                                       name=label, showlegend=True))

            ylim = [-0.1, 1.1]
            xlim = [min(xdata), max(xdata)]

            layout = go.Layout(width=800 * scale, height=400 * scale, title=None, titlefont=dict(size=16),
                               # , rangeslider=dict(visible = True)),
                               xaxis=dict(title="Time (seconds)", titlefont=dict(size=14), range=xlim),
                               yaxis=dict(title="Probability", titlefont=dict(size=14), range=ylim),
                               legend=dict(
                #                    x=0.05,
                #                    y=1.05,
                traceorder='normal',
                font=dict(
                    size=10,
                    color='#000'
                ),
                bgcolor='#ecf0f1',
                bordercolor='#bdc3c7',
                borderwidth=2,
                orientation="v"
            ), showlegend=showlegend)

        # If we're plotting probability trajectories for a single circuit.
        else:

            circuit = circuits

            if dskey is None:
                assert(len(stabilityanalyzer.data.keys()) == 1), \
                    "There is more than one DataSet, so must specify the `dskey`!"
                dskey = list(stabilityanalyzer.data.keys())[0]
            dtimes, data = stabilityanalyzer.data[dskey][circuit].get_timeseries_for_outcomes()
            if times is None:
                times = _np.linspace(min(dtimes), max(dtimes), 5000)
            p = stabilityanalyzer.get_probability_trajectory(
                circuit, times=times, dskey=dskey, estimatekey=estimatekey, estimator=estimator)[outcome]
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
                                        args=[{'visible': [True, True]}, ]),
                                   ]),
                     xanchor='left',
                     yanchor='top',
                     x=0.02,
                     y=1.2,  # y=0.98,
                     showactive=True
                     )
            ])

            layout = dict(width=800 * scale, height=500 * scale,
                          #title='Probability Trajectory',
                          xaxis=dict(title="Time (seconds)",),
                          #                    rangeslider=dict(visible = True),
                          #               ),
                          yaxis=dict(title="Probability", titlefont=dict(size=14), range=[0, 1]),
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


class GermFiducialProbTrajectoriesPlot(_ws.WorkspacePlot):
    """
    todo
    """

    def __init__(self, ws, stabilityanalyzer, gss, prep, germ, meas, outcome, min_length=1, times=None, dskey=None,
                 estimatekey=None, estimator=None, showlegend=False, scale=1.0):
        """
        todo

        gss : CircuitStructure
            Specifies the set of operation sequences along with their structure, e.g. fiducials, germs,
            and maximum lengths.
        """
        super(GermFiducialProbTrajectoriesPlot, self).__init__(ws, self._create, stabilityanalyzer, gss, prep, germ,
                                                               meas, outcome, min_length, times, dskey, estimatekey,
                                                               estimator, showlegend, scale)

    def _create(self, stabilityanalyzer, gss, prep, germ, meas, outcome, min_length, times, dskey, estimatekey,
                estimator, showlegend, scale):

        if isinstance(germ, str):
            germ = _Circuit(None, stringrep=germ)
        if isinstance(prep, str):
            prep = _Circuit(None, stringrep=prep)
        if isinstance(meas, str):
            meas = _Circuit(None, stringrep=meas)

        prepind = gss.prepStrs.index(prep)
        measind = gss.prepStrs.index(meas)
        # data = []
        circuitsdict = {}

        truncatedL = []
        for L in gss.Ls:
            if L >= min_length:
                truncatedL.append(L)

        #numL = len(gss.Ls)
        for Lind, L in enumerate(gss.Ls):
            if L >= min_length:
                #trace_pt = None
                for j, k, circuit in gss.get_plaquette(L, germ):
                    if j == prepind:
                        if k == measind:
                            circuitsdict[L] = circuit

        pjp = ProbTrajectoriesPlot(self.ws, stabilityanalyzer, circuitsdict, outcome, times, dskey, estimatekey,
                                   estimator, showlegend, scale)
        assert(len(pjp.figs) == 1), "Only one figure should have been created!"
        return pjp.figs[0]


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
    """
    todo
    """
    if len(results.data.keys()) > 1:  # multidataset
        drift_switchBd = ws.Switchboard(
            ["Dataset              ", "Germ                 ", "Preparation Fiducial ", "Measurement Fiducial",
             "Outcome             "],
            [list(results.data.keys()), [c.str for c in gss.germs], [c.str for c in(gss.prepStrs)],
             [c.str for c in gss.effectStrs],
             [i.str for i in results.data.get_outcome_labels()]],
            ["dropdown", "dropdown", "dropdown", "dropdown", "dropdown"], [0, 1, 0, 0, 0],
            show=[True, True, True, True, True])
        drift_switchBd.add("dataset", (0,))
        drift_switchBd.add("germ", (1,))
        drift_switchBd.add("prep", (2,))
        drift_switchBd.add("meas", (3,))
        drift_switchBd.add("outcome", (4,))

    else:
        drift_switchBd = ws.Switchboard(
            ["Germ", "Preperation Fiducial", "Measurement Fiducial", "Outcome"],
            [[c.str for c in gss.germs], [c.str for c in(gss.prepStrs)],
             [c.str for c in gss.effectStrs], [str(o) for o in results.data.get_outcome_labels()]],
            ["dropdown", "dropdown", "dropdown", "dropdown"], [0, 0, 0, 0], show=[True, True, True, True])
        drift_switchBd.add("germs", (0,))
        drift_switchBd.add("prepStrs", (1,))
        drift_switchBd.add("effectStrs", (2,))
        drift_switchBd.add("outcomes", (3,))

        drift_switchBd.germs[:] = gss.germs
        drift_switchBd.prepStrs[:] = gss.prepStrs
        drift_switchBd.effectStrs[:] = gss.effectStrs
        drift_switchBd.outcomes[:] = results.data.get_outcome_labels()

    return drift_switchBd


# TODO deprecate in favor of `report.factory.construct_drift_report`
def create_drift_report(results, gss, filename, title="auto",
                        ws=None, auto_open=False, link_to=None,
                        brevity=0, advanced_options=None, verbosity=1):
    """
    Creates a Drift report.
    """
    from pygsti.report.factory import construct_drift_report
    # Wrap a call to the new factory method
    advanced_options = advanced_options or {}
    ws = ws or _ws.Workspace(advanced_options.get('cachefile', None))

    report = construct_drift_report(
        results, gss, title, ws, verbosity
    )

    advanced_options = advanced_options or {}
    precision = advanced_options.get('precision', None)

    if filename is not None:
        if filename.endswith(".pdf"):
            report.write_pdf(
                filename, build_options=advanced_options,
                brevity=brevity, precision=precision, auto_open=auto_open,
                verbosity=verbosity
            )
        else:
            resizable = advanced_options.get('resizable', True)
            autosize = advanced_options.get('autosize', 'initial')
            connected = advanced_options.get('connected', False)
            single_file = filename.endswith(".html")

            report.write_html(
                filename, auto_open=auto_open, link_to=link_to,
                connected=connected, build_options=advanced_options,
                brevity=brevity, precision=precision,
                resizable=resizable, autosize=autosize,
                single_file=single_file, verbosity=verbosity
            )

    return ws

#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************
""" Idle Tomography reporting and plotting functions """

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
from ...tools import timed_block as _timed_block
from . import pauliobjs as _pobjs

import plotly.graph_objs as go


class IdleTomographyObservedRatesTable(_ws.WorkspaceTable):
    """
    A table of the largest N (in absolute value) observed error rates.
    """

    def __init__(self, ws, idtresults, threshold=1.0, mdl_simulator=None):
        """
        Create a IdleTomographyObservedRatesTable object.

        Parameters
        ----------
        idtresults : IdleTomographyResults
            The idle tomography results object from which to extract
            observed-rate data.

        threshold : int or float
            Specifies how many observed error rates to display.
            If an integer, display the top `threshold` rates.
            If a float, display the top `threshold` fraction of all the rates
            (e.g. 0.2 will show the to 20%).

        mdl_simulator : Model, optional
            If not None, use this Model to simulate the observed data
            points and plot these simulated values alongside the data.

        Returns
        -------
        ReportTable
        """
        super(IdleTomographyObservedRatesTable, self).__init__(
            ws, self._create, idtresults, threshold, mdl_simulator)

    def _create(self, idtresults, threshold, mdl_simulator):
        colHeadings = ['Observable Rate', 'Relation to intrinsic rates', ]

        # compute rate_threshold, so we know what to display
        all_obs_rates = []
        for typ in idtresults.pauli_fidpairs:
            for dict_of_infos in idtresults.observed_rate_infos[typ]:
                for info_dict in dict_of_infos.values():
                    all_obs_rates.append(abs(info_dict['rate']))
        all_obs_rates.sort(reverse=True)

        if isinstance(threshold, float):
            i = int(round(len(all_obs_rates) * threshold))
        elif isinstance(threshold, int):
            i = threshold
        else:
            raise ValueError("Invalid `threshold` value: %s" % str(threshold))

        if 0 <= i < len(all_obs_rates):
            rate_threshold = all_obs_rates[i]  # only display rates above this value
        else:
            rate_threshold = -1e100  # include everything

        #if typ in ('stochastic','affine') and \
        #        'stochastic/affine' in idtresults.pauli_fidpairs:
        #    typ = 'stochastic/affine' # for intrinsic stochastic and affine types
        #    if typ == "affine":  # affine columns follow all stochastic columns in jacobian
        #        intrinsicIndx += len(idtresults.error_list)

        #get "specs" tuple for all the observable rates that we'll display
        obs_rate_specs = []; nBelowThreshold = 0
        for typ in idtresults.pauli_fidpairs:  # keys == "types" of observed rates
            for fidpair, dict_of_infos in zip(idtresults.pauli_fidpairs[typ],
                                              idtresults.observed_rate_infos[typ]):
                for obsORoutcome, info_dict in dict_of_infos.items():
                    jac_row = info_dict['jacobian row']
                    if 'affine jacobian row' in info_dict:
                        jac_row = _np.concatenate((jac_row, info_dict['affine jacobian row']))
                    rate = info_dict['rate']
                    if abs(rate) > rate_threshold:
                        obs_rate_specs.append((typ, fidpair, obsORoutcome, jac_row, rate))
                    else:
                        nBelowThreshold += 1

        #sort obs_rate_specs by rate
        obs_rate_specs.sort(key=lambda x: x[4], reverse=True)

        errlst = idtresults.error_list  # shorthand
        Ne = len(idtresults.error_list)
        # number of intrinsic rates for each type (ham, sto, aff)

        table = _reporttable.ReportTable(colHeadings, (None,) * len(colHeadings))
        for typ, fidpair, obsOrOutcome, jac_row, _ in obs_rate_specs:
            fig = IdleTomographyObservedRatePlot(self.ws, idtresults, typ,
                                                 fidpair, obsOrOutcome, title="auto",
                                                 mdl_simulator=mdl_simulator)
            intrinsic_reln = ""
            for i, el in enumerate(jac_row):
                if abs(el) > 1e-6:
                    # get intrinsic name `iname` for i-th element:
                    if typ == "diffbasis":
                        if i < Ne: iname = "H(%s)" % str(errlst[i]).strip()
                        else: iname = "A(%s)" % str(errlst[i - Ne]).strip()
                    else:  # typ == "samebasis"
                        if i < Ne: iname = "S(%s)" % str(errlst[i]).strip()
                        else: iname = "A(%s)" % str(errlst[i - Ne]).strip()

                    if len(intrinsic_reln) == 0:
                        if el == 1.0: elstr = ""
                        elif el == -1.0: elstr = "-"
                        else: elstr = "%g" % el
                    else:
                        elstr = " + " if el >= 0 else " - "
                        elstr += "" if abs(el) == 1.0 else "%g" % abs(el)
                    intrinsic_reln += elstr + iname

            row_data = [fig, intrinsic_reln]
            row_formatters = ['Figure', None]
            table.addrow(row_data, row_formatters)

        if nBelowThreshold > 0:
            table.addrow(["%d observed rates below %g" % (nBelowThreshold, rate_threshold), ""],
                         [None, None])

        table.finish()
        return table


class IdleTomographyObservedRatesForIntrinsicRateTable(_ws.WorkspaceTable):
    """
    A table showing the observed error rates relevant for determining a
    particular intrinsic rate.  Output can be limited to just the largest
    observed rates.
    """

    def __init__(self, ws, idtresults, typ, errorOp, threshold=1.0,
                 mdl_simulator=None):
        """
        Create a IdleTomographyObservedRatesForIntrinsicRateTable.

        Parameters
        ----------
        idtresults : IdleTomographyResults
            The idle tomography results object from which to extract
            observed-rate data.

        typ : {"hamiltonian", "stochastic", "affine"}
            The type of the intrinsic rate to target.

        errorOp : NQPauliOp
            The intrinsic error (of the given `typ`), specified as
            a N-qubit Pauli operator.

        threshold : int or float
            Specifies how many observed error rates to consider.
            If an integer, display the top `threshold` rates of *all* the
            observed rates.  For example, if `threshold=10` and none of the
            top 10 rates are applicable to the given `typ`,`errorOp` error,
            then nothing is displayed.  If a float, display the top `threshold`
            fraction, again of *all* the rates (e.g. 0.2 means the top 20%).

        mdl_simulator : Model, optional
            If not None, use this Model to simulate the observed data
            points and plot these simulated values alongside the data.

        Returns
        -------
        ReportTable
        """
        super(IdleTomographyObservedRatesForIntrinsicRateTable, self).__init__(
            ws, self._create, idtresults, typ, errorOp, threshold,
            mdl_simulator)

    def _create(self, idtresults, typ, errorOp, threshold, mdl_simulator):
        colHeadings = ['Jacobian El', 'Observable Rate']

        if not isinstance(errorOp, _pobjs.NQPauliOp):
            errorOp = _pobjs.NQPauliOp(errorOp)  # try to init w/whatever we've been given

        intrinsicIndx = idtresults.error_list.index(errorOp)

        if typ in ('stochastic', 'affine') and \
                'stochastic/affine' in idtresults.pauli_fidpairs:
            typ = 'stochastic/affine'  # for intrinsic stochastic and affine types
            if typ == "affine":  # affine columns follow all stochastic columns in jacobian
                intrinsicIndx += len(idtresults.error_list)

        #thresholding:
        all_obs_rates = []
        for dict_of_infos in idtresults.observed_rate_infos[typ]:
            for info_dict in dict_of_infos.values():
                all_obs_rates.append(abs(info_dict['rate']))
        all_obs_rates.sort(reverse=True)

        if isinstance(threshold, float):
            i = int(round(len(all_obs_rates) * threshold))
        elif isinstance(threshold, int):
            i = threshold
        else:
            raise ValueError("Invalid `threshold` value: %s" % str(threshold))

        if 0 <= i < len(all_obs_rates):
            rate_threshold = all_obs_rates[i]  # only display rates above this value
        else:
            rate_threshold = -1e100  # include everything

        #get all the observable rates that contribute to the intrinsic
        # rate specified by `typ` and `errorOp`
        obs_rate_specs = []; nBelowThreshold = 0
        #print("DB: err list = ",idtresults.error_list, " LEN=",len(idtresults.error_list))
        #print("DB: Intrinsic index = ",intrinsicIndx)
        for fidpair, dict_of_infos in zip(idtresults.pauli_fidpairs[typ],
                                          idtresults.observed_rate_infos[typ]):
            for obsORoutcome, info_dict in dict_of_infos.items():
                jac_element = info_dict['jacobian row'][intrinsicIndx]
                rate = info_dict['rate']
                if abs(jac_element) > 0:
                    #print("DB: found in Jrow=",info_dict['jacobian row'], " LEN=",len(info_dict['jacobian row']))
                    #print("   (fidpair = ",fidpair[0],fidpair[1]," o=",obsORoutcome)
                    if abs(rate) > rate_threshold:
                        obs_rate_specs.append((fidpair, obsORoutcome, jac_element, rate))
                    else:
                        nBelowThreshold += 1

        #sort obs_rate_specs by rate
        obs_rate_specs.sort(key=lambda x: x[3], reverse=True)

        table = _reporttable.ReportTable(colHeadings, (None,) * len(colHeadings))
        for fidpair, obsOrOutcome, jac_element, _ in obs_rate_specs:
            fig = IdleTomographyObservedRatePlot(self.ws, idtresults, typ,
                                                 fidpair, obsOrOutcome, title="auto",
                                                 mdl_simulator=mdl_simulator)
            row_data = [str(jac_element), fig]
            row_formatters = [None, 'Figure']
            table.addrow(row_data, row_formatters)

        if nBelowThreshold > 0:
            table.addrow(["", "%d observed rates below %g" % (nBelowThreshold, rate_threshold)],
                         [None, None])

        table.finish()
        return table


class IdleTomographyObservedRatePlot(_ws.WorkspacePlot):
    """
    A plot showing how an observed error rate is obtained by fitting a sequence
    of observed data to a simple polynomial.
    """

    def __init__(self, ws, idtresults, typ, fidpair, obsORoutcome, title="auto",
                 scale=1.0, mdl_simulator=None):
        """
        Create a IdleTomographyObservedRatePlot.

        Parameters
        ----------
        idtresults : IdleTomographyResults
            The idle tomography results object from which to extract
            observed-rate data.

        typ : {"samebasis","diffbasis"}
            The type of observed-rate: same-basis or definite-outcome rates
            prepare and measure in the same Pauli basis.  Other rates prepare
            and measure in different bases, and so have non-definite-outcomes.

        fidpair : tuple
            A `(prep,measure)` 2-tuple of :class:`NQPauliState` objects specifying
            the fiducial pair (a constant) for the data used to obtain the
            observed rate being plotted.

        obsORoutcome : NQPauliOp or NQOutcome
            The observable (if `typ` == "diffbasis") or outcome (if `typ`
            == "samebasis") identifying the observed rate to plot.

        title : str, optional
            The plot title to use.  If `"auto"`, then one is created based on
            the parameters.

        scale : float, optional
            Scaling factor to adjust the size of the final figure.

        mdl_simulator : Model, optional
            If not None, use this Model to simulate the observed data
            points and plot these simulated values alongside the data.
        """
        super(IdleTomographyObservedRatePlot, self).__init__(
            ws, self._create, idtresults, typ, fidpair, obsORoutcome,
            title, scale, mdl_simulator)

    def _create(self, idtresults, typ, fidpair, obsORoutcome,
                title, scale, mdl_simulator):

        maxLens = idtresults.max_lengths
        GiStr = _Circuit(idtresults.idle_str)
        prepStr = fidpair[0].to_circuit(idtresults.prep_basis_strs)
        measStr = fidpair[1].to_circuit(idtresults.meas_basis_strs)

        ifidpair = idtresults.pauli_fidpairs[typ].index(fidpair)
        info_dict = idtresults.observed_rate_infos[typ][ifidpair][obsORoutcome]
        obs_rate = info_dict['rate']
        data_pts = info_dict['data']
        errorbars = info_dict['errbars']
        fitCoeffs = info_dict['fitCoeffs']
        fitOrder = info_dict['fitOrder']
        if idtresults.predicted_obs_rates is not None:
            predictedRate = idtresults.predicted_obs_rates[typ][fidpair][obsORoutcome]
        else:
            predictedRate = None

        if title == "auto":
            title = "Prep: %s (%s), Meas: %s (%s)" % (prepStr.str, str(fidpair[0]),
                                                      measStr.str, str(fidpair[1]))
        xlabel = "Length"
        if typ == "diffbasis":
            ylabel = "<" + str(obsORoutcome).strip() + ">"  # Expectation value
        else:
            ylabel = "Prob(" + str(obsORoutcome).strip() + ")"  # Outcome probability

        traces = []
        x = _np.linspace(maxLens[0], maxLens[-1], 50)

        traces.append(go.Scatter(
            x=maxLens,
            y=data_pts,
            error_y=dict(
                type='data',
                array=errorbars,
                visible=True,
                color='#000000',
                thickness=1,
                width=2
            ),
            mode="markers",
            marker=dict(
                color='black',
                size=10),
            name='observed data'))

        if mdl_simulator:
            circuits = [prepStr + GiStr * L + measStr for L in maxLens]
            probs = mdl_simulator.bulk_probs(circuits)
            sim_data = []
            for opstr in circuits:
                ps = probs[opstr]

                #Expectation value - assume weight at most 2 for now
                if typ == "diffbasis":
                    obs_indices = [i for i, letter in enumerate(obsORoutcome.rep) if letter != 'I']
                    minus_sign = _np.prod([fidpair[1].signs[i] for i in obs_indices])

                    # <Z> = p0 - p1 (* minus_sign)
                    if len(obs_indices) == 1:
                        i = obs_indices[0]  # the qubit we care about
                        p0 = p1 = 0
                        for outcome, p in ps.items():
                            if outcome[0][i] == '0': p0 += p  # [0] b/c outcomes are actually 1-tuples
                            else: p1 += p
                        exptn = p0 - p1

                    # <ZZ> = p00 - p01 - p10 + p11 (* minus_sign)
                    elif len(obs_indices) == 2:
                        i, j = obs_indices  # the qubits we care about
                        p_even = p_odd = 0
                        for outcome, p in ps.items():
                            if outcome[0][i] == outcome[0][j]: p_even += p
                            else: p_odd += p
                            exptn = p_even - p_odd
                    else:
                        raise NotImplementedError("Expectation values of weight > 2 observables are not implemented!")
                    val = minus_sign * exptn

                #Outcome probability
                else:
                    outcomeStr = str(obsORoutcome)
                    val = ps[outcomeStr]
                sim_data.append(val)

            traces.append(go.Scatter(
                x=maxLens,
                y=sim_data,
                mode="markers",
                marker=dict(
                    color='#DD00DD',
                    size=5),
                name='simulated'))

        if len(fitCoeffs) == 2:  # 1st order fit
            assert(_np.isclose(fitCoeffs[0], obs_rate))
            fit = fitCoeffs[0] * x + fitCoeffs[1]
            fit_line = None
        elif len(fitCoeffs) == 3:
            fit = fitCoeffs[0] * x**2 + fitCoeffs[1] * x + fitCoeffs[2]
            #OLD: assert(_np.isclose(fitCoeffs[1], obs_rate))
            #OLD: fit_line = fitCoeffs[1]*x + (fitCoeffs[0]*x[0]**2 + fitCoeffs[2])
            det = fitCoeffs[1]**2 - 4 * fitCoeffs[2] * fitCoeffs[0]
            slope = -_np.sign(fitCoeffs[0]) * _np.sqrt(det) if det >= 0 else fitCoeffs[1]
            fit_line = slope * x + (fit[0] - slope * x[0])
            assert(_np.isclose(slope, obs_rate))
        else:
            #print("DB: ",fitCoeffs)
            raise NotImplementedError("Only up to order 2 fits!")

        traces.append(go.Scatter(
            x=x,
            y=fit,
            mode="lines",  # dashed? "markers"?
            marker=dict(
                color='rgba(0,0,255,0.8)',
                line=dict(
                    width=2,
                )),
            name='o(%d) fit (slope=%.2g)' % (fitOrder, obs_rate)))

        if fit_line is not None:
            traces.append(go.Scatter(
                x=x,
                y=fit_line,
                mode="lines",
                marker=dict(
                    color='rgba(0,0,280,0.8)'),
                line=dict(
                    width=1,
                    dash='dash'),
                name='o(%d) fit line' % fitOrder,
                showlegend=False))

        if predictedRate is not None:
            traces.append(go.Scatter(
                x=x,
                y=(fit[0] - predictedRate * x[0]) + predictedRate * x,
                mode="lines",  # dashed? "markers"?
                marker=dict(
                    color='rgba(0,280,0,0.8)',  # black?
                    line=dict(
                        width=2,
                    )),
                name='predicted rate = %g' % predictedRate))

        layout = go.Layout(
            width=700 * scale,
            height=400 * scale,
            title=title,
            font=dict(size=10),
            xaxis=dict(
                title=xlabel,
            ),
            yaxis=dict(
                title=ylabel,
            ),
        )

        pythonVal = {}  # TODO
        return _reportfigure.ReportFigure(
            go.Figure(data=traces, layout=layout),
            None, pythonVal)


class IdleTomographyIntrinsicErrorsTable(_ws.WorkspaceTable):
    """
    A table of all the intrinsic rates found by idle tomography.
    """

    def __init__(self, ws, idtresults,
                 display=("H", "S", "A"), display_as="boxes"):
        """
        Create a IdleTomographyIntrinsicErrorsTable.

        Parameters
        ----------
        idtresults : IdleTomographyResults
            The idle tomography results object from which to extract
            observed-rate data.

        display : tuple of {"H","S","A"}
            Specifes which columns to include: the intrinsic Hamiltonian,
            Stochastic, and/or Affine errors.  Note that if an error type
            is not included in `idtresults` it's column will not be displayed
            regardless of the value of `display`.

        display_as : {"numbers", "boxes"}, optional
            How to display the matrices, as either numerical
            grids (fine for small matrices) or as a plot of colored
            boxes (space-conserving and better for large matrices).

        Returns
        -------
        ReportTable
        """
        super(IdleTomographyIntrinsicErrorsTable, self).__init__(
            ws, self._create, idtresults, display, display_as)

    def _create(self, idtresults, display, display_as):
        colHeadings = ['Qubits']

        irname = {'H': 'hamiltonian', 'S': 'stochastic', 'A': 'affine'}
        display = [disp for disp in display
                   if irname[disp] in idtresults.intrinsic_rates]

        for disp in display:
            if disp == "H":
                colHeadings.append('Hamiltonian')
            elif disp == "S":
                colHeadings.append('Stochastic')
            elif disp == "A":
                colHeadings.append('Affine')
            else: raise ValueError("Invalid display element: %s" % disp)

        assert(display_as == "boxes" or display_as == "numbers")
        table = _reporttable.ReportTable(colHeadings, (None,) * len(colHeadings))

        def process_rates(typ):
            """Process list of intrinsic rates, binning into rates for different sets of qubits"""
            rates = _collections.defaultdict(dict)
            for err, value in zip(idtresults.error_list,
                                  idtresults.intrinsic_rates[typ]):
                qubits = [i for i, P in enumerate(err.rep) if P != 'I']  # (in sorted order)
                op = _pobjs.NQPauliOp(''.join([P for P in err.rep if P != 'I']))
                rates[tuple(qubits)][op] = value
            return rates

        M = 0; all_keys = set()
        ham_rates = sto_rates = aff_rates = {}  # defaults
        if 'H' in display:
            ham_rates = process_rates('hamiltonian')
            M = max(M, max(_np.abs(idtresults.intrinsic_rates['hamiltonian'])))
            all_keys.update(ham_rates.keys())
        if 'S' in display:
            sto_rates = process_rates('stochastic')
            M = max(M, max(_np.abs(idtresults.intrinsic_rates['stochastic'])))
            all_keys.update(sto_rates.keys())
        if 'A' in display:
            aff_rates = process_rates('affine')
            M = max(M, max(_np.abs(idtresults.intrinsic_rates['affine'])))
            all_keys.update(aff_rates.keys())

        #min/max
        m = -M

        def get_plot_info(qubits, rate_dict):
            wt = len(qubits)  # the weight of the errors
            basisLblLookup = {_pobjs.NQPauliOp(''.join(tup)): i for i, tup in
                              enumerate(_itertools.product(["X", "Y", "Z"], repeat=wt))}
            #print("DB: ",list(basisLblLookup.keys()))
            #print("DB: ",list(rate_dict.keys()))
            values = _np.zeros(len(basisLblLookup), 'd')
            for op, val in rate_dict.items():
                values[basisLblLookup[op]] = val
            if wt == 2:
                xlabels = ["X", "Y", "Z"]
                ylabels = ["X", "Y", "Z"]
                values = values.reshape((3, 3))
            else:
                xlabels = list(_itertools.product(["X", "Y", "Z"], repeat=wt))
                ylabels = [""]
                values = values.reshape((1, len(values)))
            return values, xlabels, ylabels

        sorted_keys = sorted(list(all_keys), key=lambda x: (len(x),) + x)

        #Create rows with plots
        for ky in sorted_keys:
            row_data = [str(ky)]
            row_formatters = [None]

            for disp in display:
                if disp == "H" and ky in ham_rates:
                    values, xlabels, ylabels = get_plot_info(ky, ham_rates[ky])
                    if display_as == "boxes":
                        fig = _wp.MatrixPlot(
                            self.ws, values, m, M, xlabels, ylabels,
                            boxLabels=True, prec="compacthp")
                        row_data.append(fig)
                        row_formatters.append('Figure')
                    else:
                        row_data.append(values)
                        row_formatters.append('Brackets')

                if disp == "S" and ky in sto_rates:
                    values, xlabels, ylabels = get_plot_info(ky, sto_rates[ky])
                    if display_as == "boxes":
                        fig = _wp.MatrixPlot(
                            self.ws, values, m, M, xlabels, ylabels,
                            boxLabels=True, prec="compacthp")
                        row_data.append(fig)
                        row_formatters.append('Figure')
                    else:
                        row_data.append(values)
                        row_formatters.append('Brackets')

                if disp == "A" and ky in aff_rates:
                    values, xlabels, ylabels = get_plot_info(ky, aff_rates[ky])
                    if display_as == "boxes":
                        fig = _wp.MatrixPlot(
                            self.ws, values, m, M, xlabels, ylabels,
                            boxLabels=True, prec="compacthp")
                        row_data.append(fig)
                        row_formatters.append('Figure')
                    else:
                        row_data.append(values)
                        row_formatters.append('Brackets')

            table.addrow(row_data, row_formatters)

        table.finish()
        return table

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
    Creates the switchboard used by the idle tomography report
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

    #OLD TODO REMOVE
    # errortype_labels = None
    # errorop_labels = None
    # for results in results_dict.values():
    #     errorop_labels = _add_new_labels(errorop_labels, [str(e).strip() for e in results.error_list])
    #     errortype_labels   = _add_new_labels(errortype_labels, list(results.intrinsic_rates.keys()))
    # errortype_labels = list(sorted(errortype_labels))
    #
    # multidataset = bool(len(dataset_labels) > 1)
    #
    # switchBd = ws.Switchboard(
    #     ["Dataset","ErrorType","ErrorOp"],
    #     [dataset_labels,errortype_labels,errorop_labels],
    #     ["dropdown","dropdown","dropdown"], [0,0,0],
    #     show=[multidataset,False,False] # only show dataset dropdown (for sidebar)
    # )
    #
    # switchBd.add("results",(0,))
    # switchBd.add("errortype",(1,))
    # switchBd.add("errorop",(2,))
    #
    # for d,dslbl in enumerate(dataset_labels):
    #     switchBd.results[d] = results_dict[dslbl]
    #
    # for i,etyp in enumerate(errortype_labels):
    #     switchBd.errortype[i] = etyp
    #
    # for i,eop in enumerate(errorop_labels):
    #     switchBd.errorop[i] = eop

    return switchBd, dataset_labels


def create_idletomography_report(results, filename, title="auto",
                                 ws=None, auto_open=False, link_to=None,
                                 brevity=0, advancedOptions=None, verbosity=1):
    """
    Creates an Idle Tomography report, summarizing the results of running
    idle tomography on a data set.

    Parameters
    ----------
    results : IdleTomographyResults
        An object which represents the set of results from an idle tomography
        run, typically obtained from running :func:`do_idle_tomography` OR a
        dictionary of such objects, representing multiple idle tomography runs
        to be compared (typically all with *different* data sets). The keys of
        this dictionary are used to label different data sets that are
        selectable in the report.

    filename : string, optional
       The output filename where the report file(s) will be saved.  If
       None, then no output file is produced (but returned Workspace
       still caches all intermediate results).

    title : string, optional
       The title of the report.  "auto" causes a random title to be
       generated (which you may or may not like).

    ws : Workspace, optional
        The workspace used as a scratch space for performing the calculations
        and visualizations required for this report.  If you're creating
        multiple reports with similar tables, plots, etc., it may boost
        performance to use a single Workspace for all the report generation.

    auto_open : bool, optional
        If True, automatically open the report in a web browser after it
        has been generated.

    link_to : list, optional
        If not None, a list of one or more items from the set
        {"tex", "pdf", "pkl"} indicating whether or not to
        create and include links to Latex, PDF, and Python pickle
        files, respectively.  "tex" creates latex source files for
        tables; "pdf" renders PDFs of tables and plots ; "pkl" creates
        Python versions of plots (pickled python data) and tables (pickled
        pandas DataFrams).

    advancedOptions : dict, optional
        A dictionary of advanced options for which the default values aer usually
        are fine.  Here are the possible keys of `advancedOptions`:

        - connected : bool, optional
            Whether output HTML should assume an active internet connection.  If
            True, then the resulting HTML file size will be reduced because it
            will link to web resources (e.g. CDN libraries) instead of embedding
            them.

        - cachefile : str, optional
            filename with cached workspace results

        - precision : int or dict, optional
            The amount of precision to display.  A dictionary with keys
            "polar", "sci", and "normal" can separately specify the
            precision for complex angles, numbers in scientific notation, and
            everything else, respectively.  If an integer is given, it this
            same value is taken for all precision types.  If None, then
            `{'normal': 6, 'polar': 3, 'sci': 0}` is used.

        - resizable : bool, optional
            Whether plots and tables are made with resize handles and can be
            resized within the report.

        - autosize : {'none', 'initial', 'continual'}
            Whether tables and plots should be resized, either initially --
            i.e. just upon first rendering (`"initial"`) -- or whenever
            the browser window is resized (`"continual"`).

    verbosity : int, optional
       How much detail to send to stdout.

    Returns
    -------
    Workspace
        The workspace object used to create the report
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
            title = "Idle Tomography Report for " + autoname
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

    multidataset = bool(len(dataset_labels) > 1)
    #REM intErrView = [False,True,True]

    if fmt == "html":
        qtys['topSwitchboard'] = switchBd
        #REM qtys['intrinsicErrSwitchboard'] = switchBd.view(intErrView,"v1")

    results = switchBd.results
    #REM errortype = switchBd.errortype
    #REM errorop = switchBd.errorop
    A = None  # no brevity restriction: always display; for "Summary"- & "Help"-tab figs

    #Brevity key:
    # TODO - everything is always displayed for now

    addqty(A, 'intrinsicErrorsTable', ws.IdleTomographyIntrinsicErrorsTable, results)
    addqty(A, 'observedRatesTable', ws.IdleTomographyObservedRatesTable, results,
           20, mdl_sim)  # HARDCODED - show only top 20 rates
    # errortype, errorop,

    # Generate plots
    printer.log("*** Generating plots ***")

    toggles = {}
    toggles['CompareDatasets'] = False  # not comparable by default
    if multidataset:
        #check if data sets are comparable (if they have the same sequences)
        comparable = True
        gstrCmpList = list(results_dict[dataset_labels[0]].dataset.keys())  # maybe use circuit_lists['final']??
        for dslbl in dataset_labels:
            if list(results_dict[dslbl].dataset.keys()) != gstrCmpList:
                _warnings.warn("Not all data sets are comparable - no comparisions will be made.")
                comparable = False; break

        if comparable:
            #initialize a new "dataset comparison switchboard"
            dscmp_switchBd = ws.Switchboard(
                ["Dataset1", "Dataset2"],
                [dataset_labels, dataset_labels],
                ["buttons", "buttons"], [0, 1]
            )
            dscmp_switchBd.add("dscmp", (0, 1))
            dscmp_switchBd.add("dscmp_gss", (0,))
            dscmp_switchBd.add("refds", (0,))

            for d1, dslbl1 in enumerate(dataset_labels):
                dscmp_switchBd.dscmp_gss[d1] = results_dict[dslbl1].circuit_structs['final']
                dscmp_switchBd.refds[d1] = results_dict[dslbl1].dataset  # only used for #of spam labels below

            # dsComp = dict()
            all_dsComps = dict()
            indices = []
            for i in range(len(dataset_labels)):
                for j in range(len(dataset_labels)):
                    indices.append((i, j))

            #REMOVE (for using comm)
            #if comm is not None:
            #    _, indexDict, _ = _distribute_indices(indices, comm)
            #    rank = comm.Get_rank()
            #    for k, v in indexDict.items():
            #        if v == rank:
            #            d1, d2 = k
            #            dslbl1 = dataset_labels[d1]
            #            dslbl2 = dataset_labels[d2]
            #
            #            ds1 = results_dict[dslbl1].dataset
            #            ds2 = results_dict[dslbl2].dataset
            #            dsComp[(d1, d2)] = _DataComparator(
            #                [ds1, ds2], DS_names=[dslbl1, dslbl2])
            #    dicts = comm.gather(dsComp, root=0)
            #    if rank == 0:
            #        for d in dicts:
            #            for k, v in d.items():
            #                d1, d2 = k
            #                dscmp_switchBd.dscmp[d1, d2] = v
            #                all_dsComps[(d1,d2)] = v
            #else:
            for d1, d2 in indices:
                dslbl1 = dataset_labels[d1]
                dslbl2 = dataset_labels[d2]
                ds1 = results_dict[dslbl1].dataset
                ds2 = results_dict[dslbl2].dataset
                all_dsComps[(d1, d2)] = _DataComparator([ds1, ds2], DS_names=[dslbl1, dslbl2])
                dscmp_switchBd.dscmp[d1, d2] = all_dsComps[(d1, d2)]

            qtys['dscmpSwitchboard'] = dscmp_switchBd
            addqty(4, 'dsComparisonSummary', ws.DatasetComparisonSummaryPlot, dataset_labels, all_dsComps)
            #addqty('dsComparisonHistogram', ws.DatasetComparisonHistogramPlot, dscmp_switchBd.dscmp, display='pvalue')
            addqty(4, 'dsComparisonHistogram', ws.ColorBoxPlot,
                   'dscmp', dscmp_switchBd.dscmp_gss, dscmp_switchBd.refds, None,
                   dscomparator=dscmp_switchBd.dscmp, typ="histogram")
            addqty(1, 'dsComparisonBoxPlot', ws.ColorBoxPlot, 'dscmp', dscmp_switchBd.dscmp_gss,
                   dscmp_switchBd.refds, None, dscomparator=dscmp_switchBd.dscmp)
            toggles['CompareDatasets'] = True
        else:
            toggles['CompareDatasets'] = False  # not comparable!
    else:
        toggles['CompareDatasets'] = False

    if filename is not None:
        if True:  # comm is None or comm.Get_rank() == 0:
            # 3) populate template file => report file
            printer.log("*** Merging into template file ***")

            if fmt == "html":
                if filename.endswith(".html"):
                    _merge.merge_jinja_template(
                        qtys, filename, templateDir='~idletomography_html_report',
                        auto_open=auto_open, precision=precision, link_to=link_to,
                        connected=connected, toggles=toggles, renderMath=renderMath,
                        resizable=resizable, autosize=autosize, verbosity=printer
                    )
                else:
                    _merge.merge_jinja_template_dir(
                        qtys, filename, templateDir='~idletomography_html_report',
                        auto_open=auto_open, precision=precision, link_to=link_to,
                        connected=connected, toggles=toggles, renderMath=renderMath,
                        resizable=resizable, autosize=autosize, verbosity=printer
                    )

            elif fmt == "latex":
                raise NotImplementedError("No PDF version of this report is available yet.")
                templateFile = "idletomography_pdf_report.tex"
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

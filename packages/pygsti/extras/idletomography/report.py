""" Idle Tomography reporting and plotting functions """
from __future__ import division, print_function, absolute_import, unicode_literals

import numpy as _np
import itertools as _itertools
import collections as _collections

from ...report import workspace as _workspace
from ...report import workspaceplots as _wp
from ...report import table as _reporttable
from ...report import figure as _reportfigure
from . import pauliobjs as _pobjs

import plotly.graph_objs as go


#HERE - need to create this table of intrinsic values, then
# - map each intrinsic value to a set of observable rates via jacobians,
#   maybe as a list of (typ, fidpair, obs/outcome, jac_element) tuples?
# - create plots for a/each observable rate, i.e., for any such tuple above,
#   and maybe allow multiple idtresults as input...
# - create another workspace table that displays all the above such plots
#   that affect a given intrinsic rate.
# - report/tab will show intrinsic-rates table and a switchboard that allows the
#   user to select a given intrinsic rate and displays the corresponding table of
#   observable rate plots.


class IdleTomographyObservedRatesTable(_workspace.WorkspaceTable):
    """ 
    TODO: docstring
    """
    def __init__(self, ws, idtresult, typ, errorOp):
        """
        TODO: docstring
        idtresult may be a list or results too? titles?

        Returns
        -------
        ReportTable
        """
        super(IdleTomographyObservedRatesTable,self).__init__(
            ws, self._create, idtresult, typ, errorOp)

    def _create(self, idtresult, typ, errorOp):
        colHeadings = ['Jacobian El', 'Observable Rate']

        if not isinstance(errorOp, _pobjs.NQPauliOp):
            errorOp = _pobjs.NQPauliOp(errorOp) # try to init w/whatever we've been given

        intrinsicIndx = idtresult.error_list.index(errorOp)

        if typ in ('stochastic','affine') and \
                'stochastic/affine' in idtresult.pauli_fidpairs: 
            typ = 'stochastic/affine' # for intrinsic stochastic and affine types
            if typ == "affine":  # affine columns follow all stochastic columns in jacobian
                intrinsicIndx += len(idtresult.error_list)

        #get all the observable rates that contribute to the intrinsic
        # rate specified by `typ` and `errorOp`
        obs_rate_specs = []
        #print("DB: err list = ",idtresult.error_list, " LEN=",len(idtresult.error_list))
        #print("DB: Intrinsic index = ",intrinsicIndx)
        for fidpair,dict_of_infos in zip(idtresult.pauli_fidpairs[typ],
                                         idtresult.observed_rate_infos[typ]):
            for obsORoutcome,info_dict in dict_of_infos.items():
                jac_element = info_dict['jacobian row'][intrinsicIndx]
                if abs(jac_element) > 0:
                    #print("DB: found in Jrow=",info_dict['jacobian row'], " LEN=",len(info_dict['jacobian row']))
                    #print("   (fidpair = ",fidpair[0],fidpair[1]," o=",obsORoutcome)
                    obs_rate_specs.append( (fidpair, obsORoutcome, jac_element) )

        #TODO: sort obs_rate_specs in some sensible way
        
        table = _reporttable.ReportTable(colHeadings, (None,)*len(colHeadings))
        for fidpair, obsOrOutcome, jac_element in obs_rate_specs:
            fig = IdleTomographyObservedRatePlot(self.ws, idtresult, typ, 
                                                 fidpair, obsOrOutcome, title="auto")
            row_data = [str(jac_element), fig]
            row_formatters = [None, 'Figure']
            table.addrow(row_data, row_formatters)

        table.finish()
        return table
        



class IdleTomographyObservedRatePlot(_workspace.WorkspacePlot):
    """ TODO """
    def __init__(self, ws, idtresult, typ, fidpair, obsORoutcome, title="auto",
                 true_rate=None, scale=1.0):
        super(IdleTomographyObservedRatePlot,self).__init__(
            ws, self._create, idtresult, typ, fidpair, obsORoutcome,
                 title, true_rate, scale)
        
    def _create(self, idtresult, typ, fidpair, obsORoutcome,
                 title, true_rate, scale):

        if title == "auto":
            title = typ + " fidpair=%s,%s" % (fidpair[0],fidpair[1])
            if typ == "hamiltonian":
                title += " observable="+str(obsORoutcome)
            else:
                title += " outcome="+str(obsORoutcome)   

        xlabel = "Length"
        if typ == "hamiltonian": 
            ylabel =  "Expectation value"
        else:
            ylabel = "Outcome probability"
    
        maxLens = idtresult.max_lengths
        ifidpair = idtresult.pauli_fidpairs[typ].index(fidpair)
        info_dict = idtresult.observed_rate_infos[typ][ifidpair][obsORoutcome]
        obs_rate = info_dict['rate']
        data_pts = info_dict['data']
        weights = info_dict['weights']
        fitCoeffs = info_dict['fitCoeffs']
        fitOrder = info_dict['fitOrder']
    
        traces = []
        traces.append( go.Scatter(
            x=maxLens,
            y=data_pts,
            mode="markers",
            marker=dict(
                color = 'black',
                size=10),
            name='observed data' ))
    
        x = _np.linspace(maxLens[0],maxLens[-1],50)
        if len(fitCoeffs) == 2: # 1st order fit
            assert(_np.isclose(fitCoeffs[0], obs_rate))
            fit = fitCoeffs[0]*x + fitCoeffs[1]
            fit_line = None
        elif len(fitCoeffs) == 3:
            assert(_np.isclose(fitCoeffs[1], obs_rate))
            fit = fitCoeffs[0]*x**2 + fitCoeffs[1]*x + fitCoeffs[2]
            fit_line = fitCoeffs[1]*x + (fitCoeffs[0]*x[0]**2 + fitCoeffs[2])
        else:
            #print("DB: ",fitCoeffs)
            raise NotImplementedError("Only up to order 2 fits!")
    
        traces.append( go.Scatter(
            x=x,
            y=fit,
            mode="lines", #dashed? "markers"? 
            marker=dict(
                color = 'rgba(0,0,255,0.8)',
                line = dict(
                    width = 2,
                    )),
            name='o(%d) fit (slope=%.2g)' % (fitOrder,obs_rate)))
    
        if fit_line:
            traces.append( go.Scatter(
                x=x,
                y=fit_line,
                mode="lines",
                marker=dict(
                    color = 'rgba(0,0,200,0.8)',
                    line = dict(
                        width = 1,
                        )),
                name='o(%d) fit slope' % fitOrder))
    
        if true_rate:
            traces.append( go.Scatter(
                x=x,
                y=(fit[0]-true_rate*x[0])+true_rate*x,
                mode="lines", #dashed? "markers"? 
                marker=dict(
                    color = 'rgba(0,0,255,0.8)', # black?
                    line = dict(
                        width = 2,
                        )),
                name='true rate = %g' % true_rate))
    
        layout = go.Layout(
            width=700*scale,
            height=400*scale,
            title=title,
            font=dict(size=10),
            xaxis=dict(
                title=xlabel,
                ),
            yaxis=dict(
                    title=ylabel,
                ),
            )
    
        pythonVal = {} # TODO
        return _reportfigure.ReportFigure(
            go.Figure(data=traces, layout=layout),
            None, pythonVal)

    

class IdleTomographyIntrinsicErrorsTable(_workspace.WorkspaceTable):
    """ 
    TODO: docstring
    """
    def __init__(self, ws, idtresults, 
                 display=("H","S","A"), display_as="boxes"):

        """
        TODO: docstring
        idtresults may be a list or results too? titles?

        Returns
        -------
        ReportTable
        """
        super(IdleTomographyIntrinsicErrorsTable,self).__init__(
            ws, self._create, idtresults, display, display_as)

    def _create(self, idtresults, display, display_as):
        colHeadings = ['Qubits']

        for disp in display:
            if disp == "H":
                colHeadings.append('Hamiltonian')
            elif disp == "S":
                colHeadings.append('Stochastic')
            elif disp == "A":
                colHeadings.append('Affine')
            else: raise ValueError("Invalid display element: %s" % disp)

        assert(display_as == "boxes" or display_as == "numbers")
        table = _reporttable.ReportTable(colHeadings, (None,)*len(colHeadings))

        #Process list of intrinsic rates, binning into rates for different sets of qubits
        def process_rates(typ):
            rates = _collections.defaultdict(dict)
            for err, value in zip(idtresults.error_list,
                                  idtresults.intrinsic_rates[typ]):
                qubits = [i for i,P in enumerate(err.rep) if P != 'I'] # (in sorted order)
                op    = _pobjs.NQPauliOp(''.join([P for P in err.rep if P != 'I']))
                rates[tuple(qubits)][op] = value
            return rates

        M = 0; all_keys = set()
        ham_rates = sto_rates = aff_rates = {} # defaults
        if 'H' in display:
            ham_rates = process_rates('hamiltonian')
            M = max(M,max(_np.abs(idtresults.intrinsic_rates['hamiltonian'])))
            all_keys.update(ham_rates.keys())
        if 'S' in display:
            sto_rates = process_rates('stochastic')
            M = max(M,max(_np.abs(idtresults.intrinsic_rates['stochastic'])))
            all_keys.update(sto_rates.keys())
        if 'A' in display:
            aff_rates = process_rates('affine')
            M = max(M,max(_np.abs(idtresults.intrinsic_rates['affine'])))
            all_keys.update(aff_rates.keys())

        #min/max
        m = -M


        def get_plot_info(qubits, rate_dict):
            wt = len(qubits) # the weight of the errors
            basisLblLookup = { _pobjs.NQPauliOp(''.join(tup)):i for i,tup in 
                               enumerate(_itertools.product(["X","Y","Z"],repeat=wt)) }
            #print("DB: ",list(basisLblLookup.keys()))
            #print("DB: ",list(rate_dict.keys()))
            values = _np.zeros(len(basisLblLookup),'d')
            for op,val in rate_dict.items():
                values[basisLblLookup[op]] = val
            if wt == 2:
                xlabels = ["X","Y","Z"]
                ylabels = ["X","Y","Z"]
                values = values.reshape((3,3))
            else:
                xlabels = list(_itertools.product(["X","Y","Z"],repeat=wt))
                ylabels = [""]
                values = values.reshape((1,len(values)))
            return values, xlabels, ylabels
                                    
        sorted_keys = sorted(list(all_keys), key=lambda x: (len(x),)+x)

        #Create rows with plots
        for ky in sorted_keys:
            row_data = [str(ky)]
            row_formatters = [None]

            for disp in display:
                if disp == "H" and ky in ham_rates:
                    values, xlabels, ylabels = get_plot_info(ky,ham_rates[ky])
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
                    values, xlabels, ylabels = get_plot_info(ky,sto_rates[ky])
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
                    values, xlabels, ylabels = get_plot_info(ky,aff_rates[ky])
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

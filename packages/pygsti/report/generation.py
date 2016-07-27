from __future__ import division, print_function, absolute_import, unicode_literals
#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************
""" Functions for generating GST reports (PDF or HTML)."""

import warnings           as _warnings
import numpy              as _np
import scipy.stats        as _stats

from .. import algorithms as _alg
from .. import tools      as _tools
from .. import objects    as _objs

from . import reportables as _cr
from . import plotting    as _plotting

from .table import ReportTable as _ReportTable



def get_blank_table():
    """ Create a blank table as a placeholder."""
    table = _ReportTable(['Blank'], [None])
    table.finish()
    return table


def get_gateset_spam_table(gateset, confidenceRegionInfo=None,
                           includeHSVec=True):
    """
    Create a table for gateset's SPAM vectors.

    Parameters
    ----------
    gateset : GateSet
        The GateSet

    confidenceRegionInfo : ConfidenceRegion, optional
        If not None, specifies a confidence-region
        used to display error intervals.

    includeHSVec : boolean, optional
        Whether or not to include the Hilbert-Schmidt
        vector representation column in the table.

    Returns
    -------
    ReportTable
    """

    mxBasis    = gateset.get_basis_name()
    mxBasisDim = gateset.get_basis_dimension()
    basisNm    = _tools.basis_longname(mxBasis, mxBasisDim)

    if confidenceRegionInfo is None:
        if includeHSVec:
            colHeadings = ('Operator','Hilbert-Schmidt vector (%s basis)' % basisNm,'Matrix')
            formatters  = (None,None,None)
        else:
            colHeadings = ('Operator','Matrix')
            formatters  = (None,None)

    else:
        if includeHSVec:
            colHeadings = ('Operator',
                           'Hilbert-Schmidt vector (%s basis)' % basisNm,
                           '%g%% C.I. half-width' % confidenceRegionInfo.level,
                           'Matrix')
            formatters  = (None,None,'Conversion',None)
        else:
            colHeadings = ('Operator',
                           'Matrix')
            formatters = (None,None)


    table = _ReportTable(colHeadings, formatters)

    for lbl,rhoVec in gateset.preps.items():
        rhoMx = _tools.vec_to_stdmx(rhoVec, mxBasis)

        if includeHSVec:
            if confidenceRegionInfo is None:
                table.addrow((lbl, rhoVec, rhoMx), ('Rho','Normal','Brackets'))
            else:
                intervalVec = confidenceRegionInfo.get_profile_likelihood_confidence_intervals(lbl)[:,None]
                if intervalVec.shape[0] == gateset.get_dimension()-1: #TP constrained, so pad with zero top row
                    intervalVec = _np.concatenate( (_np.zeros((1,1),'d'),intervalVec), axis=0 )
                table.addrow((lbl, rhoVec, intervalVec, rhoMx), ('Rho','Normal','Normal','Brackets'))
        else:
            #no dependence on confidence region (yet) when HS vector is not shown...
            table.addrow((lbl, rhoMx), ('Rho','Brackets'))


    for lbl,EVec in gateset.effects.items():
        EMx = _tools.vec_to_stdmx(EVec, mxBasis)

        if includeHSVec:
            if confidenceRegionInfo is None:
                table.addrow((lbl, EVec, EMx), ('Effect', 'Normal', 'Brackets'))
            else:
                intervalVec = confidenceRegionInfo.get_profile_likelihood_confidence_intervals(lbl)[:,None]
                table.addrow((lbl, EVec, intervalVec, EMx), ('Effect','Normal','Normal','Brackets'))
        else:
            #no dependence on confidence region (yet) when HS vector is not shown...
            table.addrow((lbl, EMx), ('Effect','Brackets'))

    table.finish()
    return table



def get_gateset_spam_parameters_table(gateset, confidenceRegionInfo=None):
    """
    Create a table for gateset's "SPAM parameters", that is, the
    dot products of prep-vectors and effect-vectors.

    Parameters
    ----------
    gateset : GateSet
        The GateSet

    confidenceRegionInfo : ConfidenceRegion, optional
        If not None, specifies a confidence-region
        used to display error intervals.

    Returns
    -------
    ReportTable
    """
    colHeadings = [''] + list(gateset.get_effect_labels())
    formatters  = [None] + [ 'Effect' ]*len(gateset.get_effect_labels())

    table       = _ReportTable(colHeadings, formatters)

    spamDotProdsQty = _cr.compute_gateset_qty("Spam DotProds", gateset, confidenceRegionInfo)
    DPs, DPEBs      = spamDotProdsQty.get_value_and_err_bar()

    formatters      = [ 'Rho' ] + [ 'ErrorBars' ]*len(gateset.get_effect_labels()) #for rows below

    for ii,prepLabel in enumerate(gateset.get_prep_labels()): # ii enumerates rhoLabels to index DPs
        rowData = [prepLabel]
        for jj,_ in enumerate(gateset.get_effect_labels()): # jj enumerates eLabels to index DPs
            if confidenceRegionInfo is None:
                rowData.append((DPs[ii,jj],None))
            else:
                rowData.append((DPs[ii,jj],DPEBs[ii,jj]))
        table.addrow(rowData, formatters)

    table.finish()
    return table


def get_gateset_gates_table(gateset, confidenceRegionInfo=None):
    """
    Create a table for gateset's gates.

    Parameters
    ----------
    gateset : GateSet
        The GateSet

    confidenceRegionInfo : ConfidenceRegion, optional
        If not None, specifies a confidence-region
        used to display error intervals.


    Returns
    -------
    ReportTable
    """
    gateLabels = list(gateset.gates.keys())  # gate labels
    mxBasis    = gateset.get_basis_name()
    mxBasisDim = gateset.get_basis_dimension()
    basisNm    = _tools.basis_longname(mxBasis, mxBasisDim)

    if confidenceRegionInfo is None:
        colHeadings = ('Gate','Superoperator (%s basis)' % basisNm)
        formatters  = (None,None)
    else:
        colHeadings = ('Gate','Superoperator (%s basis)' % basisNm,
                       '%g%% C.I. half-width' % confidenceRegionInfo.level)
        formatters  = (None,None,'Conversion')

    table = _ReportTable(colHeadings, formatters)

    for gl in gateLabels:
        if confidenceRegionInfo is None:
            table.addrow((gl, gateset.gates[gl]), (None,'Brackets'))
        else:
            intervalVec    = confidenceRegionInfo.get_profile_likelihood_confidence_intervals(gl)[:,None]
            if isinstance(gateset.gates[gl], _objs.FullyParameterizedGate): #then we know how to reshape into a matrix
                gate_dim   = gateset.get_dimension()
                intervalMx = intervalVec.reshape(gate_dim,gate_dim)
            elif isinstance(gateset.gates[gl], _objs.TPParameterizedGate): #then we know how to reshape into a matrix
                gate_dim   = gateset.get_dimension()
                intervalMx = _np.concatenate( ( _np.zeros((1,gate_dim),'d'),
                                                intervalVec.reshape(gate_dim-1,gate_dim)), axis=0 )
            else:
                intervalMx = intervalVec # we don't know how best to reshape vector of parameter intervals, so don't
            table.addrow((gl, gateset.gates[gl], intervalMx), (None,'Brackets','Brackets'))

    table.finish()
    return table


def get_unitary_gateset_gates_table(gateset, confidenceRegionInfo=None):
    """
    Create a table for gateset's gates assuming they're unitary.

    Parameters
    ----------
    gateset : GateSet
        The GateSet

    confidenceRegionInfo : ConfidenceRegion, optional
        If not None, specifies a confidence-region
        used to display error intervals.


    Returns
    -------
    ReportTable
    """
    gateLabels = list(gateset.gates.keys())  # gate labels
    mxBasis    = gateset.get_basis_name()
    mxBasisDim = gateset.get_basis_dimension()
    basisNm    = _tools.basis_longname(mxBasis, mxBasisDim)

    qtys_to_compute = [ ('%s decomposition' % gl) for gl in gateLabels ]
    qtys = _cr.compute_gateset_qtys(qtys_to_compute, gateset, confidenceRegionInfo)

    if confidenceRegionInfo is None:
        colHeadings = ('Gate','Superoperator (%s basis)' % basisNm,'Rotation axis','Angle')
        formatters  = (None,None,None,None)
    else:
        colHeadings = ('Gate','Superoperator (%s basis)' % basisNm,
                       '%g%% C.I. half-width' % confidenceRegionInfo.level,
                       'Rotation axis','Angle')
        formatters  = (None,None,'Conversion',None,None)

    table = _ReportTable(colHeadings, formatters)

    for gl in gateLabels:
        decomp, decompEB = qtys['%s decomposition' % gl].get_value_and_err_bar()
        if confidenceRegionInfo is None:
            table.addrow(
                        (gl, gateset.gates[gl],decomp.get('axis of rotation','X'),decomp.get('pi rotations','X')),
                        (None, 'Brackets', 'Normal', 'Pi') )
        else:
            intervalVec = confidenceRegionInfo.get_profile_likelihood_confidence_intervals(gl)[:,None]
            if isinstance(gateset.gates[gl], _objs.FullyParameterizedGate): #then we know how to reshape into a matrix
                gate_dim   = gateset.get_dimension()
                intervalMx = intervalVec.reshape(gate_dim,gate_dim)
            elif isinstance(gateset.gates[gl], _objs.TPParameterizedGate): #then we know how to reshape into a matrix
                gate_dim   = gateset.get_dimension()
                intervalMx = _np.concatenate( ( _np.zeros((1,gate_dim),'d'),
                                                intervalVec.reshape(gate_dim-1,gate_dim)), axis=0 )
            else:
                intervalMx = intervalVec # we don't know how best to reshape vector of parameter intervals, so don't

            table.addrow(
                        (gl, gateset.gates[gl],decomp.get('axis of rotation','X'),
                         (decomp.get('pi rotations','X'), decompEB.get('pi rotations','X')) ),
                        (None, 'Brackets', 'Normal', 'PiErrorBars') )

    table.finish()
    return table


def get_gateset_choi_table(gateset, confidenceRegionInfo=None):
    """
    Create a table for the Choi matrices of a gateset's gates.

    Parameters
    ----------
    gateset : GateSet
        The GateSet

    confidenceRegionInfo : ConfidenceRegion, optional
        If not None, specifies a confidence-region
        used to display error intervals.

    Returns
    -------
    ReportTable
    """
    gateLabels = list(gateset.gates.keys())  # gate labels

    qtys_to_compute = []
    qtys_to_compute += [ ('%s choi eigenvalues' % gl) for gl in gateLabels ]
    qtys_to_compute += [ ('%s choi matrix' % gl) for gl in gateLabels ]

    mxBasis    = gateset.get_basis_name()
    mxBasisDim = gateset.get_basis_dimension()
    qtys       = _cr.compute_gateset_qtys(qtys_to_compute, gateset, confidenceRegionInfo)
    basisNm    = _tools.basis_longname(mxBasis, mxBasisDim)

    colHeadings = ('Gate','Choi matrix (%s basis)' % basisNm,'Eigenvalues')
    formatters  = (None,None,None)

    table = _ReportTable(colHeadings, formatters)

    for gl in gateLabels:
        evals, evalsEB = qtys['%s choi eigenvalues' % gl].get_value_and_err_bar()

        choiMx, _ = qtys['%s choi matrix' % gl].get_value_and_err_bar()
        if confidenceRegionInfo is None:
            table.addrow((gl, choiMx, evals), (None, 'Brackets', 'Normal'))
        else:
            table.addrow((gl, choiMx, (evals,evalsEB)), (None, 'Brackets', 'VecErrorBars'))

    table.finish()
    return table


def get_gates_vs_target_table(gateset, targetGateset,
                              confidenceRegionInfo=None):
    """
    Create a table comparing a gateset's gates to a target gateset.

    Parameters
    ----------
    gateset, targetGateset : GateSet
        The gate sets to compare

    confidenceRegionInfo : ConfidenceRegion, optional
        If not None, specifies a confidence-region
        used to display error intervals.


    Returns
    -------
    ReportTable
    """
    gateLabels  = list(gateset.gates.keys())  # gate labels

    colHeadings = ('Gate', "Process|Infidelity", "1/2 Trace|Distance", "1/2 Diamond-Norm") #, "Frobenius|Distance"
    formatters  = (None,'Conversion','Conversion','Conversion') # ,'Conversion'

    qtyNames        = ('infidelity','Jamiolkowski trace dist','diamond norm') #,'Frobenius diff'
    qtys_to_compute = [ '%s %s' % (gl,qty) for qty in qtyNames for gl in gateLabels ]
    qtys            = _cr.compute_gateset_gateset_qtys(qtys_to_compute, gateset, targetGateset,
                                            confidenceRegionInfo)

    table = _ReportTable(colHeadings, formatters)

    formatters = [None] + [ 'ErrorBars' ]*len(qtyNames)

    for gl in gateLabels:
        if confidenceRegionInfo is None:
            rowData = [gl] + [ (qtys['%s %s' % (gl,qty)].get_value(),None) for qty in qtyNames ]
        else:
            rowData = [gl] + [ qtys['%s %s' % (gl,qty)].get_value_and_err_bar() for qty in qtyNames ]
        table.addrow(rowData, formatters)

    table.finish()
    return table


def get_spam_vs_target_table(gateset, targetGateset,
                             confidenceRegionInfo=None):
    """
    Create a table comparing a gateset's SPAM operations to a target gateset.

    Parameters
    ----------
    gateset, targetGateset : GateSet
        The gate sets to compare

    confidenceRegionInfo : ConfidenceRegion, optional
        If not None, specifies a confidence-region
        used to display error intervals.


    Returns
    -------
    ReportTable
    """
    prepLabels   = gateset.get_prep_labels()
    effectLabels = gateset.get_effect_labels()

    colHeadings  = ('Prep/POVM', "State|Infidelity", "1/2 Trace|Distance")
    formatters   = (None,'Conversion','Conversion')

    table = _ReportTable(colHeadings, formatters)

    qtyNames = ('state infidelity','trace dist')

    formatters = [ 'Rho' ] + [ 'ErrorBars' ]*len(qtyNames)
    qtys_to_compute = [ '%s prep %s' % (l,qty) for qty in qtyNames for l in prepLabels ]
    qtys = _cr.compute_gateset_gateset_qtys(qtys_to_compute, gateset, targetGateset,
                                            confidenceRegionInfo)
    for l in prepLabels:
        if confidenceRegionInfo is None:
            rowData = [l] + [ (qtys['%s prep %s' % (l,qty)].get_value(),None) for qty in qtyNames ]
        else:
            rowData = [l] + [ qtys['%s prep %s' % (l,qty)].get_value_and_err_bar() for qty in qtyNames ]
        table.addrow(rowData, formatters)

    formatters = [ 'Effect' ] + [ 'ErrorBars' ]*len(qtyNames)
    qtys_to_compute = [ '%s effect %s' % (l,qty) for qty in qtyNames for l in effectLabels ]
    qtys = _cr.compute_gateset_gateset_qtys(qtys_to_compute, gateset, targetGateset,
                                            confidenceRegionInfo)
    for l in effectLabels:
        if confidenceRegionInfo is None:
            rowData = [l] + [ (qtys['%s effect %s' % (l,qty)].get_value(),None) for qty in qtyNames ]
        else:
            rowData = [l] + [ qtys['%s effect %s' % (l,qty)].get_value_and_err_bar() for qty in qtyNames ]
        table.addrow(rowData, formatters)

    table.finish()
    return table



def get_gates_vs_target_err_gen_table(gateset, targetGateset,
                                        confidenceRegionInfo=None):
    """
    Create a table listing the error generators obtained by
    comparing a gateset's gates to a target gateset.

    Parameters
    ----------
    gateset, targetGateset : GateSet
        The gate sets to compare

    confidenceRegionInfo : ConfidenceRegion, optional
        If not None, specifies a confidence-region
        used to display error intervals.

    Returns
    -------
    ReportTable
    """
    gateLabels  = list(gateset.gates.keys())  # gate labels
    colHeadings = ('Gate','Error Generator')

    table = _ReportTable(colHeadings, (None,None))

    for gl in gateLabels:
        table.addrow((gl, _tools.error_generator(gateset.gates[gl],
                                                 targetGateset.gates[gl])),
                     (None, 'Brackets'))
    table.finish()
    return table



def get_gates_vs_target_angles_table(gateset, targetGateset,
                                     confidenceRegionInfo=None):
    """
    Create a table comparing a gateset to a target gateset.

    Parameters
    ----------
    gateset, targetGateset : GateSet
        The gate sets to compare

    confidenceRegionInfo : ConfidenceRegion, optional
        If not None, specifies a confidence-region
        used to display error intervals.


    Returns
    -------
    ReportTable
    """
    gateLabels  = list(gateset.gates.keys())  # gate labels

    colHeadings = ('Gate', "Angle between|rotation axes")
    formatters  = (None,'Conversion')

    qtyNames        = ('angle btwn rotn axes',)
    qtys_to_compute = [ '%s %s' % (gl,qty) for qty in qtyNames for gl in gateLabels ]
    qtys            = _cr.compute_gateset_gateset_qtys(qtys_to_compute, gateset, targetGateset,
                                            confidenceRegionInfo)

    table = _ReportTable(colHeadings, formatters)

    formatters = [None] + [ 'PiErrorBars' ]*len(qtyNames)

    for gl in gateLabels:
        if confidenceRegionInfo is None:
            rowData = [gl] + [ (qtys['%s %s' % (gl,qty)].get_value(),None) for qty in qtyNames ]
        else:
            rowData = [gl] + [ qtys['%s %s' % (gl,qty)].get_value_and_err_bar() for qty in qtyNames ]
        table.addrow(rowData, formatters)

    table.finish()
    return table


def get_gateset_closest_unitary_table(gateset, confidenceRegionInfo=None):
    """
    Create a table for gateset that contains closest-unitary gates.

    Parameters
    ----------
    gateset : GateSet
        The GateSet

    confidenceRegionInfo : ConfidenceRegion, optional
        If not None, specifies a confidence-region
        used to display error intervals.

    Returns
    -------
    ReportTable
    """

    gateLabels  = list(gateset.gates.keys())  # gate labels
    colHeadings = ('Gate','Process|Infidelity','1/2 Trace|Distance','Rotation|Axis','Rotation|Angle','Sanity Check')
    formatters  = (None,'Conversion','Conversion','Conversion','Conversion','Conversion')

    if gateset.get_dimension() != 4:
        table = _ReportTable(colHeadings, formatters)

        table.finish()
        return table

    qtyNames = ('max fidelity with unitary', 'max trace dist with unitary',
                'closest unitary decomposition', 'upper bound on fidelity with unitary')
    qtys_to_compute = [ '%s %s' % (gl,qty) for qty in qtyNames for gl in gateLabels ]
    qtys            = _cr.compute_gateset_qtys(qtys_to_compute, gateset, confidenceRegionInfo)
    decompNames = ('axis of rotation','pi rotations')
    #Other possible qtyName: 'closest unitary choi matrix'

    table = _ReportTable(colHeadings, formatters)

    formatters = [None, 'ErrorBars', 'ErrorBars', 'VecErrorBars', 'PiErrorBars', 'Normal' ] # Note len(decompNames)==2, 2nd el is rotn angle

    for gl in gateLabels:
        fUB, _ = qtys['%s upper bound on fidelity with unitary' % gl].get_value_and_err_bar()
        fLB, fLB_EB = qtys['%s max fidelity with unitary' % gl].get_value_and_err_bar()
        td, td_EB = qtys['%s max trace dist with unitary' % gl].get_value_and_err_bar()
        sanity = (1.0-fLB)/(1.0-fUB) - 1.0 #Robin's sanity check metric (0=good, >1=bad)
        decomp, decompEB = qtys['%s closest unitary decomposition' % gl].get_value_and_err_bar()

        if confidenceRegionInfo is None:
            rowData = [gl, (1.0-fLB,None), (td,None)] + [(decomp.get(x,'X'),None) for x in decompNames]
        else:
            rowData = [gl, (1.0-fLB,fLB_EB), (td,td_EB)] + [(decomp.get(x,'X'),decompEB.get(x,None)) for x in decompNames]
        rowData.append(sanity)

        table.addrow(rowData, formatters)

    table.finish()
    return table


def get_gateset_decomp_table(gateset, confidenceRegionInfo=None):
    """
    Create table for decomposing a gateset's gates.

    Parameters
    ----------
    gateset : GateSet
        The GateSet

    confidenceRegionInfo : ConfidenceRegion, optional
        If not None, specifies a confidence-region
        used to display error intervals.

    Returns
    -------
    ReportTable
    """
    gateLabels = list(gateset.gates.keys())  # gate labels
    colHeadings = ('Gate','Eigenvalues','Fixed pt','Rotn. axis','Diag. decay','Off-diag. decay')
    formatters = [None]*6

    qtyNames = ('eigenvalues','decomposition')
    qtys_to_compute = [ '%s %s' % (gl,qty) for qty in qtyNames for gl in gateLabels ]
    qtys = _cr.compute_gateset_qtys(qtys_to_compute, gateset, confidenceRegionInfo)
    decompNames = ('fixed point',
                   'axis of rotation',
                   'decay of diagonal rotation terms',
                   'decay of off diagonal rotation terms')

    table = _ReportTable(colHeadings, formatters)

    formatters = (None, 'VecErrorBars', 'Normal', 'Normal', 'ErrorBars', 'ErrorBars')

    for gl in gateLabels:
        decomp, decompEB = qtys['%s decomposition' % gl].get_value_and_err_bar()

        if confidenceRegionInfo is None or decompEB is None: #decompEB is None when gate decomp failed
            evals = qtys['%s eigenvalues' % gl].get_value()
            rowData = [gl, (evals,None)] + [decomp.get(x,'X') for x in decompNames[0:2] ] + \
                [(decomp.get(x,'X'),None) for x in decompNames[2:4] ]
        else:
            evals, evalsEB = qtys['%s eigenvalues' % gl].get_value_and_err_bar()
            rowData = [gl, (evals,evalsEB)] + [decomp.get(x,'X') for x in decompNames[0:2] ] + \
                [(decomp.get(x,'X'),decompEB.get(x,'X')) for x in decompNames[2:4] ]

        table.addrow(rowData, formatters)

    table.finish()
    return table


def get_gateset_rotn_axis_table(gateset, confidenceRegionInfo=None,
                                showAxisAngleErrBars=True):
    """
    Create a table of the angle between a gate rotation axes for
     gates belonging to a gateset

    Parameters
    ----------
    gateset : GateSet
        The GateSet

    confidenceRegionInfo : ConfidenceRegion, optional
        If not None, specifies a confidence-region
        used to display error intervals.

    showAxisAngleErrBars : bool, optional
        Whether or not table should include error bars on the angles
        between rotation axes (doing so makes the table take up more
        space).

    Returns
    -------
    ReportTable
    """
    gateLabels = list(gateset.gates.keys())

    qtys_to_compute = [ '%s decomposition' % gl for gl in gateLabels ] + ['Gateset Axis Angles']
    qtys = _cr.compute_gateset_qtys(qtys_to_compute, gateset, confidenceRegionInfo)

    colHeadings = ("Gate","Angle") + tuple( [ "RAAW(%s)" % gl for gl in gateLabels] )
    nCols = len(colHeadings)
    formatters = [None] * nCols

    table = "tabular"
    latex_head =  "\\begin{%s}[l]{%s}\n\hline\n" % (table, "|c" * nCols + "|")
    latex_head += "\\multirow{2}{*}{Gate} & \\multirow{2}{*}{Angle} & " + \
                  "\\multicolumn{%d}{c|}{Angle between Rotation Axes} \\\\ \cline{3-%d}\n" % (len(gateLabels),nCols)
    latex_head += " & & %s \\\\ \hline\n" % (" & ".join(gateLabels))

    table = _ReportTable(colHeadings, formatters,
                         customHeader={'latex': latex_head} )

    formatters = [None, 'PiErrorBars'] + [ 'PiErrorBars' ] * len(gateLabels)

    rotnAxisAngles, rotnAxisAnglesEB = qtys['Gateset Axis Angles'].get_value_and_err_bar()
    rotnAngles = [ qtys['%s decomposition' % gl].get_value().get('pi rotations','X') \
                       for gl in gateLabels ]

    for i,gl in enumerate(gateLabels):
        decomp, decompEB = qtys['%s decomposition' % gl].get_value_and_err_bar()
        rotnAngle = decomp.get('pi rotations','X')

        angles_btwn_rotn_axes = []
        for j,gl_other in enumerate(gateLabels):
            decomp_other, _ = qtys['%s decomposition' % gl_other].get_value_and_err_bar()
            rotnAngle_other = decomp_other.get('pi rotations','X')

            if gl_other == gl:
                angles_btwn_rotn_axes.append( ("",None) )
            elif str(rotnAngle) == 'X' or abs(rotnAngle) < 1e-4 or \
                 str(rotnAngle_other) == 'X' or abs(rotnAngle_other) < 1e-4:
                angles_btwn_rotn_axes.append( ("--",None) )
            elif not _np.isnan(rotnAxisAngles[i,j]):
                if showAxisAngleErrBars and rotnAxisAnglesEB is not None:
                    angles_btwn_rotn_axes.append( (rotnAxisAngles[i,j], rotnAxisAnglesEB[i,j]) )
                else:
                    angles_btwn_rotn_axes.append( (rotnAxisAngles[i,j], None) )
            else:
                angles_btwn_rotn_axes.append( ("X",None) )

        if confidenceRegionInfo is None or decompEB is None: #decompEB is None when gate decomp failed
            rowData = [gl, (rotnAngle,None)] + angles_btwn_rotn_axes
        else:
            rowData = [gl, (rotnAngle,decompEB.get('pi rotations','X'))] + angles_btwn_rotn_axes
        table.addrow(rowData, formatters)

    table.finish()
    return table


def get_gateset_eigenval_table(gateset, targetGateset,
                               figFilePrefix,
                               maxWidth=6.5, maxHeight=8.0,
                               confidenceRegionInfo=None):
    """
    Create table which lists and plots the eigenvalues of a
    gateset's gates.

    Parameters
    ----------
    gateset : GateSet
        The GateSet

    targetGateset : GateSet
        The target gate set.

    figFilePrefix : str
        A filename prefix (not including any directories!) to use
        when rendering figures as a part of rendering this table.

    maxWidth : float
        The maximum width (in inches) of the entire figure.

    maxHeight : float
        The maximum height (in inches) of the entire figure.

    confidenceRegionInfo : ConfidenceRegion, optional
        If not None, specifies a confidence-region
        used to display error intervals.


    Returns
    -------
    ReportTable
    """
    gateLabels = list(gateset.gates.keys())  # gate labels

    colHeadings = ('Gate','Eigenvalues','Polar Plot') # ,'Hamiltonian'
    formatters = [None]*3

    qtyNames = ('eigenvalues',)
    qtys_to_compute = [ '%s %s' % (gl,qty) for qty in qtyNames for gl in gateLabels ]
    qtys = _cr.compute_gateset_qtys(qtys_to_compute, gateset, confidenceRegionInfo)

    table = _ReportTable(colHeadings, formatters)

    formatters = (None, 'VecErrorBars', 'Figure')
    nRows = len(gateLabels)

    for gl in gateLabels:

        gate = gateset.gates[gl]
        targetGate = targetGateset.gates[gl]

        fig = _plotting.polar_eigenval_plot(
            gate, targetGate, title=gl, save_to="",
            showNormal=True, showRelative=False)

        sz = min(0.95*(maxHeight/nRows), 0.95*0.75*(maxWidth - 0.5))
        sz = min(sz, 2.0)
        nm = figFilePrefix + "_" + gl
        figInfo = (fig,nm,sz,sz)

        if confidenceRegionInfo is None:
            evals = qtys['%s eigenvalues' % gl].get_value()
            evals = evals.reshape(evals.size//2, 2) #assumes len(evals) is even!
            rowData = [gl, (evals,None), figInfo]
        else:
            evals, evalsEB = qtys['%s eigenvalues' % gl].get_value_and_err_bar()
            evals = evals.reshape(evals.size//2, 2) #assumes len(evals) is even!
            rowData = [gl, (evals,evalsEB), figInfo]

        table.addrow(rowData, formatters)

    table.finish()
    return table


def get_gateset_relative_eigenval_table(gateset, targetGateset,
                                        figFilePrefix,
                                        maxWidth=6.5, maxHeight=8.0,
                                        confidenceRegionInfo=None):
    """
    Create table which lists and plots the *relative* eigenvalues of a
    gateset's gates.

    Relative eigenvalues are defined as the eigenvalues of
    inv(G_target) * G.

    Parameters
    ----------
    gateset : GateSet
        The GateSet

    targetGateset : GateSet
        The target gate set used to compute eigenvalues of
        gate*inv(target_gate).

    figFilePrefix : str
        A filename prefix (not including any directories!) to use
        when rendering figures as a part of rendering this table.

    maxWidth : float
        The maximum width (in inches) of the entire figure.

    maxHeight : float
        The maximum height (in inches) of the entire figure.

    confidenceRegionInfo : ConfidenceRegion, optional
        If not None, specifies a confidence-region
        used to display error intervals.


    Returns
    -------
    ReportTable
    """
    gateLabels = list(gateset.gates.keys())  # gate labels

    colHeadings = ('Gate','Relative Evals','Polar Plot') # ,'Hamiltonian'
    formatters = [None]*3

    qtyNames = ('relative eigenvalues',)
    qtys_to_compute = [ '%s %s' % (gl,qty) for qty in qtyNames for gl in gateLabels ]
    qtys = _cr.compute_gateset_gateset_qtys(qtys_to_compute, gateset, targetGateset,
                                            confidenceRegionInfo)

    table = _ReportTable(colHeadings, formatters)

    formatters = (None, 'VecErrorBars', 'Figure')
    nRows = len(gateLabels)

    for gl in gateLabels:
        gate = gateset.gates[gl]
        targetGate = targetGateset.gates[gl]

        fig = _plotting.polar_eigenval_plot(
            gate, targetGate, title=gl, save_to="",
            showNormal=False, showRelative=True)

        sz = min(0.95*(maxHeight/nRows), 0.95*0.75*(maxWidth - 0.5))
        sz = min(sz, 2.0)
        nm = figFilePrefix + "_" + gl
        figInfo = (fig,nm,sz,sz)

        if confidenceRegionInfo is None:
            rel_evals = qtys['%s relative eigenvalues' % gl].get_value()
            rel_evals = rel_evals.reshape(rel_evals.size//2, 2)
            rowData = [gl, (rel_evals,None), figInfo]
        else:
            rel_evals, rel_evalsEB = qtys['%s relative eigenvalues' % gl].get_value_and_err_bar()
            rel_evals = rel_evals.reshape(rel_evals.size//2, 2)
            rowData = [gl, (rel_evals,rel_evalsEB), figInfo]

        table.addrow(rowData, formatters)

    table.finish()
    return table


def get_gateset_choi_eigenval_table(gateset, figFilePrefix,
                                    maxWidth=6.5, maxHeight=8.0,
                                    confidenceRegionInfo=None):
    """
    Create a table for the Choi matrices of a gateset's gates.

    Parameters
    ----------
    gateset : GateSet
        The GateSet

    figFilePrefix : str
        A filename prefix (not including any directories!) to use
        when rendering figures as a part of rendering this table.

    maxWidth : float
        The maximum width (in inches) of the entire figure.

    maxHeight : float
        The maximum height (in inches) of the entire figure.

    confidenceRegionInfo : ConfidenceRegion, optional
        If not None, specifies a confidence-region
        used to display error intervals.

    Returns
    -------
    ReportTable
    """
    gateLabels = list(gateset.gates.keys())  # gate labels

    qtys_to_compute = []
    qtys_to_compute += [ ('%s choi eigenvalues' % gl) for gl in gateLabels ]

    mxBasis = gateset.get_basis_name()
    mxBasisDim = gateset.get_basis_dimension()
    qtys = _cr.compute_gateset_qtys(qtys_to_compute, gateset, confidenceRegionInfo)

    colHeadings = ('Gate','Eigenvalues','Eigenvalue Magnitudes')
    formatters = (None,None,None)
    table = _ReportTable(colHeadings, formatters)

    nRows = len(gateLabels)
    sz = min(0.95*(maxHeight/nRows), 0.95*(maxWidth - 3.0))

    for gl in gateLabels:

        evals, evalsEB = qtys['%s choi eigenvalues' % gl].get_value_and_err_bar()
        evals = evals.reshape(evals.size//4, 4) #assumes len(evals) is multiple of 4!
        nm = figFilePrefix + "_" + gl

        if confidenceRegionInfo is None:
            fig = _plotting.choi_eigenvalue_barplot(evals, ylabel="")
            figInfo = (fig,nm,sz,sz)
            table.addrow((gl, evals, figInfo), (None, 'Normal', 'Figure'))
        else:
            evalsEB = evalsEB.reshape(evalsEB.size//4, 4)
            fig = _plotting.choi_eigenvalue_barplot(evals, evalsEB, ylabel="")
            figInfo = (fig,nm,sz,sz)
            table.addrow((gl, (evals,evalsEB), figInfo), (None, 'VecErrorBars', 'Figure'))

    table.finish()
    return table


def get_dataset_overview_table(dataset, target, maxlen=10, fixedLists=None,
                               maxLengthList=None):
    """
    Create a table overviewing a data set.

    Parameters
    ----------
    dataset : DataSet
        The DataSet

    target : GateSet
        A target gateset which is used for it's mapping of SPAM labels to
        SPAM specifiers.

    maxlen : integer, optional
        The maximum length string used when searching for the
        maximal (best) Gram matrix.  It's useful to make this
        at least twice the maximum length fiducial sequence.

    fixedLists : (prepStrs, effectStrs), optional
        2-tuple of gate string lists, specifying the preparation and
        measurement fiducials to use when constructing the Gram matrix,
        and thereby bypassing the search for such lists.

    maxLengthList : list of ints, optional
        A list of the maximum lengths used, if available.

    Returns
    -------
    ReportTable
    """
    colHeadings = ('Quantity','Value')
    formatters = (None,None)
    _, svals, target_svals = _alg.max_gram_rank_and_evals( dataset, maxlen, target, fixedLists=fixedLists )
    svals = _np.sort(_np.abs(svals)).reshape(-1,1)
    target_svals = _np.sort(_np.abs(target_svals)).reshape(-1,1)
    svals_2col = _np.concatenate( (svals,target_svals), axis=1 )

    table = _ReportTable(colHeadings, formatters)

    minN = round(min([ row.total() for row in dataset.itervalues()]))
    maxN = round(max([ row.total() for row in dataset.itervalues()]))
    cntStr = "[%d,%d]" % (minN,maxN) if (minN != maxN) else "%d" % round(minN)

    table.addrow(("Number of strings", str(len(dataset))), (None,None))
    table.addrow(("Gate labels", ", ".join(dataset.get_gate_labels()) ), (None,None))
    table.addrow(("SPAM labels",  ", ".join(dataset.get_spam_labels()) ), (None,None))
    table.addrow(("Counts per string", cntStr  ), (None,None))
    table.addrow(("Gram singular values| (right column gives the values|when using the target gate set)",
                  svals_2col), ('Conversion','Small'))
    if maxLengthList is not None:
        table.addrow(("Max. Lengths", ", ".join(map(str,maxLengthList)) ), (None,None))

    table.finish()
    return table


def get_chi2_progress_table(Ls, gatesetsByL, gateStringsByL, dataset):
    """
    Create a table showing how Chi2 changes with GST iteration.

    Parameters
    ----------
    Ls : list of integers
        List of L-values (typically maximum lengths or exponents) used to
        construct the gate string lists for different iterations of GST.

    gatesetsByL : list of GateSets
        The GateSet corresponding to each iteration of GST.

    gateStringsByL : list of lists of GateStrings
        The list of gate strings used at each iteration of GST.

    dataset : DataSet
        The data set used in the GST iterations.

    Returns
    -------
    ReportTable
    """
    colHeadings = { 'latex': ('L','$\\chi^2$','$k$','$\\chi^2-k$','$\sqrt{2k}$','$p$','$N_s$','$N_p$', 'Rating'),
                    'html': ('L','&chi;<sup>2</sup>','k','&chi;<sup>2</sup>-k',
                             '&radic;<span style="text-decoration:overline;">2k</span>',
                             'p','N<sub>s</sub>','N<sub>p</sub>', 'Rating'),
                    'text': ('L','chi^2','k','chi^2-k','sqrt{2k}','p','N_s','N_p', 'Rating'),
                    'ppt': ('L','chi^2','k','chi^2-k','sqrt{2k}','p','N_s','N_p', 'Rating')
                  }

    table = _ReportTable(colHeadings, None)

    for L,gs,gstrs in zip(Ls,gatesetsByL,gateStringsByL):
        chi2 = _tools.chi2( dataset, gs, gstrs,
                                     minProbClipForWeighting=1e-4)
        Ns = len(gstrs)
        Np = gs.num_nongauge_params()

        k = max(Ns-Np,0) #expected chi^2 mean
        pv = 1.0 - _stats.chi2.cdf(chi2,k) # reject GST model if p-value < threshold (~0.05?)

        if   (chi2-k) < _np.sqrt(2*k): rating = 5
        elif (chi2-k) < 2*k: rating = 4
        elif (chi2-k) < 5*k: rating = 3
        elif (chi2-k) < 10*k: rating = 2
        else: rating = 1
        table.addrow(
                    (str(L),chi2,k,chi2-k,_np.sqrt(2*k),pv,Ns,Np,"<STAR>"*rating),
                    (None,'Normal','Normal','Normal','Normal','Rounded','Normal','Normal','Conversion'))

    table.finish()
    return table


def get_logl_progress_table(Ls, gatesetsByL, gateStringsByL, dataset):
    """
    Create a table showing how the log-likelihood changes with GST iteration.

    Parameters
    ----------
    Ls : list of integers
        List of L-values (typically maximum lengths or exponents) used to
        construct the gate string lists for different iterations of GST.

    gatesetsByL : list of GateSets
        The GateSet corresponding to each iteration of GST.

    gateStringsByL : list of lists of GateStrings
        The list of gate strings used at each iteration of GST.

    dataset : DataSet
        The data set used in the GST iterations.

    Returns
    -------
    ReportTable
    """
    colHeadings = { 'latex': ('L','$2\Delta\\log(\\mathcal{L})$','$k$','$2\Delta\\log(\\mathcal{L})-k$',
                              '$\sqrt{2k}$','$p$','$N_s$','$N_p$', 'Rating'),
                    'html': ('L','2&Delta;(log L)','k','2&Delta;(log L)-k',
                             '&radic;<span style="text-decoration:overline;">2k</span>',
                             'p','N<sub>s</sub>','N<sub>p</sub>', 'Rating'),
                    'text': ('L','2*Delta(log L)','k','2*Delta(log L)-k','sqrt{2k}','p','N_s','N_p', 'Rating'),
                    'ppt': ('L','2*Delta(log L)','k','2*Delta(log L)-k','sqrt{2k}','p','N_s','N_p', 'Rating')
                  }
    table = _ReportTable(colHeadings, None)

    for L,gs,gstrs in zip(Ls,gatesetsByL,gateStringsByL):
        logL_upperbound = _tools.logl_max(dataset, gstrs)
        logl = _tools.logl( gs, dataset, gstrs )
        if(logL_upperbound < logl):
            raise ValueError("LogL upper bound = %g but logl = %g!!" % (logL_upperbound, logl))
        Ns = len(gstrs)*(len(dataset.get_spam_labels())-1) #number of independent parameters in dataset
        Np = gs.num_nongauge_params()

        k = max(Ns-Np,0) #expected 2*(logL_ub-logl) mean
        twoDeltaLogL = 2*(logL_upperbound - logl)
        pv = 1.0 - _stats.chi2.cdf(twoDeltaLogL,k) # reject GST model if p-value < threshold (~0.05?)

        if   (twoDeltaLogL-k) < _np.sqrt(2*k): rating = 5
        elif (twoDeltaLogL-k) < 2*k: rating = 4
        elif (twoDeltaLogL-k) < 5*k: rating = 3
        elif (twoDeltaLogL-k) < 10*k: rating = 2
        else: rating = 1

        table.addrow(
                    (str(L),twoDeltaLogL,k,twoDeltaLogL-k,_np.sqrt(2*k),pv,Ns,Np,"<STAR>"*rating),
                    (None,'Normal','Normal','Normal','Normal','Rounded','Normal','Normal','Conversion'))

    table.finish()
    return table


def get_gatestring_table(gsList, title, nCols=1):
    """
    Creates a 2*nCols-column table enumerating a list of gate strings.

    Parameters
    ----------
    gsList : list of GateStrings
        List of gate strings to put in table.

    title : string
        The title for the table column containing the strings.

    nCols : int, optional
        The number of *data* columns, i.e. those containing
        gate strings.  Actual number of columns is twice this
        due to columns containing enumeration indices.

    Returns
    -------
    ReportTable
    """
    colHeadings = ('#',title)*nCols
    formatters = ('Conversion', 'Normal')*nCols

    table = _ReportTable(colHeadings, formatters)

    nRows = (len(gsList)+(nCols-1)) // nCols

    for i in range(nRows):
        formatters = ('Normal','GateString')*nCols
        rowdata = []
        for k in range(nCols):
            l = i+nRows*k #index of gatestring
            rowdata.extend( [l+1, gsList[l] if l<len(gsList) else "" ] )
        table.addrow(rowdata, formatters)

    table.finish()
    return table


def get_gatestring_multi_table(gsLists, titles, commonTitle=None):
    """
    Creates an N-column table enumerating a N-1 lists of gate strings.

    Parameters
    ----------
    gsLists : list of GateString lists
        List of gate strings to put in table.

    titles : list of strings
        The titles for the table columns containing the strings.

    commonTitle : string, optional
        A single title string to place in a cell spanning across
        all the gate string columns.

    Returns
    -------
    ReportTable
    """
    colHeadings = ('#',) + tuple(titles)
    formatters = ('Conversion',) + ('Normal',)*len(titles)

    if commonTitle is None:
        table = _ReportTable(colHeadings, formatters)
    else:
        table = "tabular"
        colHeadings = ('\\#',) + tuple(titles)
        latex_head  = "\\begin{%s}[l]{%s}\n\hline\n" % (table, "|c" * len(colHeadings) + "|")
        latex_head += " & \multicolumn{%d}{c|}{%s} \\\\ \hline\n" % (len(titles),commonTitle)
        latex_head += "%s \\\\ \hline\n" % (" & ".join(colHeadings))

        html_head = "<table><thead>"
        html_head += '<tr><th></th><th colspan="%d">%s</th></tr>\n' % (len(titles),commonTitle)
        html_head += "<tr><th> %s </th></tr>" % (" </th><th> ".join(colHeadings))
        html_head += "</thead><tbody>"
        table = _ReportTable(colHeadings, formatters,
                             customHeader={'latex': latex_head,
                                           'html': html_head})

    formatters = ('Normal',) + ('GateString',)*len(gsLists)

    for i in range( max([len(gsl) for gsl in gsLists]) ):
        rowData = [i+1]
        for gsList in gsLists:
            if i < len(gsList):
                rowData.append( gsList[i] )
            else:
                rowData.append( None ) #empty string
        table.addrow(rowData, formatters)

    table.finish()
    return table



def get_gateset_gate_boxes_table(gateset, figFilePrefix, maxWidth=6.5,
                                maxHeight=8.0, confidenceRegionInfo=None):
    """
    Create a table for a gateset's gates, where each gate is a grid of boxes.

    Similar to get_gateset_gates_table(...), except the gates are displayed
    as grids of colored boxes instead of printing the actual numerical elements.
    This is useful for displaying large gate matrices.

    Parameters
    ----------
    gateset : GateSet
        The GateSet

    figFilePrefix : str
        A filename prefix (not including any directories!) to use
        when rendering figures as a part of rendering this table.

    maxWidth : float
        The maximum width (in inches) of the entire figure.

    maxHeight : float
        The maximum height (in inches) of the entire figure.

    confidenceRegionInfo : ConfidenceRegion, optional
        If not None, specifies a confidence-region
        used to display error intervals.


    Returns
    -------
    ReportTable
    """
    gateLabels = list(gateset.gates.keys())  # gate labels
    basisNm = gateset.get_basis_name()
    basisDims = gateset.get_basis_dimension()
    basisLongNm = _tools.basis_longname(basisNm, basisDims)

    if confidenceRegionInfo is None:
        colHeadings = ('Gate','Superoperator (%s basis)' % basisLongNm)
        formatters = (None,None)
    else:
        colHeadings = ('Gate','Superoperator (%s basis)' % basisLongNm,
                       '%g%% C.I. half-width' % confidenceRegionInfo.level)
        formatters = (None,None,'Conversion')

    table = _ReportTable(colHeadings, formatters)
    nRows = len(gateset.gates)

    for gl in gateLabels:
        #Note: currently, we don't use confidence region...
        fig = _plotting.gate_matrix_boxplot(
            gateset.gates[gl], save_to="",
            mxBasis=basisNm, mxBasisDims=basisDims)

        maxFigSz = min(0.95*(maxHeight/nRows), 0.95*(maxWidth - 1.0))
        sz = min(gateset.gates[gl].shape[0] * 0.5, maxFigSz)
        nm = figFilePrefix + "_" + gl

        figInfo = (fig,nm,sz,sz)
        table.addrow((gl, figInfo ), (None,'Figure'))

    table.finish()
    return table


def get_gates_vs_target_err_gen_boxes_table(gateset, targetGateset,
                                            figFilePrefix, maxWidth=6.5,
                                            maxHeight=8.0,
                                            confidenceRegionInfo=None):
    """
    Create a table of gate error generators, where each is shown as grid of boxes.

    Parameters
    ----------
    gateset, targetGateset : GateSet
        The gate sets to compare

    figFilePrefix : str
        A filename prefix (not including any directories!) to use
        when rendering figures as a part of rendering this table.

    maxWidth : float
        The maximum width (in inches) of the entire figure.

    maxHeight : float
        The maximum height (in inches) of the entire figure.

    confidenceRegionInfo : ConfidenceRegion, optional
        If not None, specifies a confidence-region
        used to display error intervals.


    Returns
    -------
    ReportTable
    """
    gateLabels = list(gateset.gates.keys())  # gate labels
    basisNm = gateset.get_basis_name()
    basisDims = gateset.get_basis_dimension()

    if basisNm != targetGateset.get_basis_name():
        raise ValueError("Basis mismatch between gateset (%s) and target (%s)!"\
                             % (basisNm, targetGateset.get_basis_name()))

    colHeadings = ('Gate','Error Generator','Pauli projections')

    table = _ReportTable(colHeadings, (None,None,None))
    nRows = len(gateset.gates)
    #nCols = len(colHeadings)

    for gl in gateLabels:
        gate = gateset.gates[gl]
        targetGate = targetGateset.gates[gl]

        errgen_fig = _plotting.gate_matrix_errgen_boxplot(
            gate, targetGate, save_to="", mxBasis=basisNm,
            mxBasisDims=basisDims)

        hamdecomp_fig = _plotting.pauliprod_hamiltonian_boxplot(
            gate, targetGate, save_to="", mxBasis=basisNm, boxLabels=True)

        maxFigSz = min(0.95*(maxHeight/nRows), 0.95*(2./3.)*(maxWidth-1.0))
        sz = min(gateset.gates[gl].shape[0] * 0.5, maxFigSz)
        nm = figFilePrefix + "_" + gl + "_errgen"
        errgen_figInfo = (errgen_fig,nm,sz,sz)

        maxFigSz = min(0.95*(maxHeight/nRows), 0.95*(1./3.)*(maxWidth-1.0))
        sz = min( (gateset.gates[gl].size/4) * 0.5, maxFigSz)
        nm = figFilePrefix + "_" + gl + "_hamdecomp"
        hamdecomp_figInfo = (hamdecomp_fig,nm,sz,sz)

        table.addrow((gl, errgen_figInfo, hamdecomp_figInfo),
                     (None, 'Figure', 'Figure'))
    table.finish()
    return table



def get_gaugeopt_params_table(gaugeOptArgs):
    """
    Create a table displaying a list of gauge
    optimzation parameters.

    Parameters
    ----------
    gaugeOptArgs : dict
        A dictionary of specifying values for zero or more
        of the *arguments* of pyGSTi's optimize_gauge
        function.

    Returns
    -------
    ReportTable
    """
    colHeadings = ('Quantity','Value')
    formatters = ('Bold','Bold')

    table = _ReportTable(colHeadings, formatters)

    if 'toGetTo' in gaugeOptArgs:
        table.addrow(("Gauge optimize to", gaugeOptArgs['toGetTo']), (None,None))
    if 'method' in gaugeOptArgs:
        table.addrow(("Method", str(gaugeOptArgs['method'])), (None,None))
    if 'constrainToTP' in gaugeOptArgs:
        table.addrow(("TP constrained", str(gaugeOptArgs['constrainToTP'])), (None,None))
    if 'constrainToCP' in gaugeOptArgs:
        table.addrow(("CP constrained", str(gaugeOptArgs['constrainToCP'])), (None,None))
    if 'constrainToValidSpam' in gaugeOptArgs:
        table.addrow(("Valid-SPAM constrained", str(gaugeOptArgs['constrainToValidSpam'])), (None,None))
    if 'targetFactor' in gaugeOptArgs:
        table.addrow(("Target weighting", str(gaugeOptArgs['targetFactor'])), (None,None))
    if 'targetGatesMetric' in gaugeOptArgs:
        table.addrow(("Metric for gate-to-target", str(gaugeOptArgs['targetGatesMetric'])), (None,None))
    if 'targetSpamMetric' in gaugeOptArgs:
        table.addrow(("Metric for SPAM-to-target", str(gaugeOptArgs['targetSpamMetric'])), (None,None))
    if 'gateWeight' in gaugeOptArgs:
        table.addrow(("Gate weighting", str(gaugeOptArgs['gateWeight'])), (None,None))
    if 'spamWeight' in gaugeOptArgs:
        table.addrow(("SPAM weighting", str(gaugeOptArgs['spamWeight'])), (None,None))

    table.finish()
    return table




def get_logl_confidence_region(gateset, dataset, confidenceLevel,
                               gatestring_list=None, probClipInterval=(-1e6,1e6),
                               minProbClip=1e-4, radius=1e-4, hessianProjection="std",
                               regionType="std", comm=None, memLimit=None):

    """
    Constructs a ConfidenceRegion given a gateset and dataset using the log-likelihood Hessian.
    (Internally, this evaluates the log-likelihood Hessian.)

    Parameters
    ----------
    gateset : GateSet
        the gate set point estimate that maximizes the logl or minimizes
        the chi2, and marks the point in gateset-space where the Hessian
        has been evaluated.

    dataset : DataSet
        Probability data

    confidenceLevel : float
        If not None, then the confidence level (between 0 and 100) used in
        the computation of confidence regions/intervals. If None, no
        confidence regions or intervals are computed.

    gatestring_list : list of (tuples or GateStrings), optional
        Each element specifies a gate string to include in the log-likelihood
        sum.  Default value of None implies all the gate strings in dataset
        should be used.

    probClipInterval : 2-tuple or None, optional
        (min,max) values used to clip the probabilities predicted by gateset. Defaults to no clipping.

    minProbClip : float, optional
        The minimum probability treated normally in the evaluation of the log-likelihood.
        A penalty function replaces the true log-likelihood for probabilities that lie
        below this threshold so that the log-likelihood never becomes undefined.

    radius : float, optional
        Specifies the severity of rounding used to "patch" the zero-frequency
        terms of the log-likelihood.

    hessianProjection : string, optional
        Specifies how (and whether) to project the given hessian matrix
        onto a non-gauge space.  Allowed values are:

        - 'std' -- standard projection onto the space perpendicular to the
          gauge space.
        - 'none' -- no projection is performed.  Useful if the supplied
          hessian has already been projected.
        - 'optimal gate CIs' -- a lengthier projection process in which a
          numerical optimization is performed to find the non-gauge space
          which minimizes the (average) size of the confidence intervals
          corresponding to gate (as opposed to SPAM vector) parameters.

    regionType : {'std', 'non-markovian'}, optional
        The type of confidence region to create.  'std' creates a standard
        confidence region, while 'non-markovian' creates a region which
        attempts to account for the non-markovian-ness of the data.

    comm : mpi4py.MPI.Comm, optional
        When not None, an MPI communicator for distributing the computation
        across multiple processors.

    memLimit : int, optional
        A rough memory limit in bytes which restricts the amount of intermediate
        values that are computed and stored.


    Returns
    -------
    ConfidenceRegion
    """
    if gatestring_list is None:
        gatestring_list = list(dataset.keys())

    #Compute appropriate Hessian
    hessian = _tools.logl_hessian(gateset, dataset, gatestring_list,
                                  minProbClip, probClipInterval, radius,
                                  comm=comm, memLimit=memLimit)

    #Compute the non-Markovian "radius" if required
    if regionType == "std":
        nonMarkRadiusSq = 0.0
    elif regionType == "non-markovian":
        nGateStrings = len(gatestring_list)
        nModelParams = gateset.num_nongauge_params()
        nDataParams  = nGateStrings*(len(dataset.get_spam_labels())-1)
          #number of independent parameters in dataset (max. model # of params)

        MIN_NON_MARK_RADIUS = 1e-8 #must be >= 0
        nonMarkRadiusSq = max( 2*(_tools.logl_max(dataset)
                                  - _tools.logl(gateset, dataset)) \
                                   - (nDataParams-nModelParams),
                               MIN_NON_MARK_RADIUS )
    else:
        raise ValueError("Invalid confidence region type: %s" % regionType)


    cri = _objs.ConfidenceRegion(gateset, hessian, confidenceLevel,
                                 hessianProjection,
                                 nonMarkRadiusSq=nonMarkRadiusSq)

    #Check that number of gauge parameters reported by gateset is consistent with confidence region
    # since the parameter number computed this way is used in chi2 or logl progress tables
    Np_check =  gateset.num_nongauge_params()
    if(Np_check != cri.nNonGaugeParams):
        _warnings.warn("Number of non-gauge parameters in gateset and confidence region do "
                       + " not match.  This indicates an internal logic error.")

    return cri



def get_chi2_confidence_region(gateset, dataset, confidenceLevel,
                               gatestring_list=None, probClipInterval=(-1e6,1e6),
                               minProbClipForWeighting=1e-4, hessianProjection="std",
                               regionType='std', comm=None, memLimit=None):

    """
    Constructs a ConfidenceRegion given a gateset and dataset using the Chi2 Hessian.
    (Internally, this evaluates the Chi2 Hessian.)

    Parameters
    ----------
    gateset : GateSet
        the gate set point estimate that maximizes the logl or minimizes
        the chi2, and marks the point in gateset-space where the Hessian
        has been evaluated.

    dataset : DataSet
        Probability data

    confidenceLevel : float
        If not None, then the confidence level (between 0 and 100) used in
        the computation of confidence regions/intervals. If None, no
        confidence regions or intervals are computed.

    gatestring_list : list of (tuples or GateStrings), optional
        Each element specifies a gate string to include in the log-likelihood
        sum.  Default value of None implies all the gate strings in dataset
        should be used.

    probClipInterval : 2-tuple or None, optional
        (min,max) values used to clip the probabilities predicted by gateset. Defaults to no clipping.

    minProbClipForWeighting : float, optional
        Sets the minimum and maximum probability p allowed in the chi^2 weights: N/(p*(1-p))
        by clipping probability p values to lie within the interval
        [ minProbClipForWeighting, 1-minProbClipForWeighting ].

    hessianProjection : string, optional
        Specifies how (and whether) to project the given hessian matrix
        onto a non-gauge space.  Allowed values are:

        - 'std' -- standard projection onto the space perpendicular to the
          gauge space.
        - 'none' -- no projection is performed.  Useful if the supplied
          hessian has already been projected.
        - 'optimal gate CIs' -- a lengthier projection process in which a
          numerical optimization is performed to find the non-gauge space
          which minimizes the (average) size of the confidence intervals
          corresponding to gate (as opposed to SPAM vector) parameters.

    regionType : {'std', 'non-markovian'}, optional
        The type of confidence region to create.  'std' creates a standard
        confidence region, while 'non-markovian' creates a region which
        attempts to account for the non-markovian-ness of the data.

    comm : mpi4py.MPI.Comm, optional
        When not None, an MPI communicator for distributing the computation
        across multiple processors.

    memLimit : int, optional
        A rough memory limit in bytes which restricts the amount of intermediate
        values that are computed and stored.


    Returns
    -------
    ConfidenceRegion
    """
    if gatestring_list is None:
        gatestring_list = list(dataset.keys())

    #Compute appropriate Hessian
    chi2, hessian = _tools.chi2(dataset, gateset, gatestring_list,
                                False, True, minProbClipForWeighting,
                                probClipInterval, memLimit=memLimit)

    #Compute the non-Markovian "radius" if required
    if regionType == "std":
        nonMarkRadiusSq = 0.0
    elif regionType == "non-markovian":
        nGateStrings = len(gatestring_list)
        nModelParams = gateset.num_nongauge_params()
        nDataParams  = nGateStrings*(len(dataset.get_spam_labels())-1)
          #number of independent parameters in dataset (max. model # of params)

        MIN_NON_MARK_RADIUS = 1e-8 #must be >= 0
        nonMarkRadiusSq = max(chi2 - (nDataParams-nModelParams), MIN_NON_MARK_RADIUS)
    else:
        raise ValueError("Invalid confidence region type: %s" % regionType)


    cri = _objs.ConfidenceRegion(gateset, hessian, confidenceLevel,
                                 hessianProjection,
                                 nonMarkRadiusSq=nonMarkRadiusSq)

    #Check that number of gauge parameters reported by gateset is consistent with confidence region
    # since the parameter number computed this way is used in chi2 or logl progress tables
    Np_check =  gateset.num_nongauge_params()
    if(Np_check != cri.nNonGaugeParams):
        _warnings.warn("Number of non-gauge parameters in gateset and confidence region do "
                       + " not match.  This indicates an internal logic error.")

    return cri

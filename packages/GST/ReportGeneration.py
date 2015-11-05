""" Functions for generating GST reports (PDF or HTML)."""
import os as _os
import re as _re
import collections as _collections
import warnings as _warnings
import numpy as _np
import scipy.linalg as _spl
import scipy.optimize as _spo
import scipy.stats as _stats
import sys as _sys
import matplotlib as _matplotlib
import LatexUtil as _LU
import HtmlUtil as _HU
import ComputeReportables as _CR
import GramMatrix as _GM
import AnalysisTools as _AT
import BasisTools as _BT
import JamiolkowskiOps as _JOps
import LikelihoodFunctions as _LF
import Gate as _Gate
import ReportTableFormat as _F
from Core import getRhoAndESpecs as _getRhoAndESpecs
from Core import getRhoAndEStrs as _getRhoAndEStrs
from Core import optimizeGauge as _optimizeGauge
from confidenceregion import ConfidenceRegion as _ConfidenceRegion


def getBlankTable(formats):
    """ Create a blank table as a placeholder witht the given formats """
    tables = {}
    _F.CreateTable(formats, tables, ['Blank'], [None], "", False)
    _F.FinishTable(formats, tables, False)
    return tables
    

def getGatesetSPAMTable(gateset, formats, tableclass, longtable, confidenceRegionInfo=None):
    """ 
    Create an HTML table for gateset's SPAM vectors.
    
    Parameters
    ----------
    gateset : GateSet
        The GateSet

    tableclass : string
        CSS class to apply to the HTML table.

    confidenceRegionInfo : ConfidenceRegion, optional
        If not None, specifies a confidence-region
        used to display error intervals.

    Returns
    -------
    string
        HTML string output.
    """

    if confidenceRegionInfo is None:
        colHeadings = ('Operator','Hilbert-Schmidt vector (Pauli basis)','Matrix')
        formatters = (None,None,None)
    else:
        colHeadings = ('Operator',
                       'Hilbert-Schmidt vector (Pauli basis)',
                       '%g%% C.I. half-width' % confidenceRegionInfo.level,
                       'Matrix')
        formatters = (None,None,_F.TxtCnv,None)

    tables = {}
    _F.CreateTable(formats, tables, colHeadings, formatters, tableclass, longtable)

    for i,rhoVec in enumerate(gateset.rhoVecs):
        if confidenceRegionInfo is None:
            _F.AddTableRow(formats, tables, (i, rhoVec, _BT.pauliProdVectorToMatrixInStdBasis(rhoVec)),
                        (_F.Rho,_F.Nml,_F.Brk))
        else:
            intervalVec = confidenceRegionInfo.getProfileLikelihoodConfidenceIntervals("rho%d" % i)[:,None]
            if intervalVec.shape[0] == gateset.get_dimension()-1: #TP constrained, so pad with zero top row
                intervalVec = _np.concatenate( (_np.zeros((1,1),'d'),intervalVec), axis=0 )

            _F.AddTableRow(formats, tables, (i, rhoVec, intervalVec, _BT.pauliProdVectorToMatrixInStdBasis(rhoVec)),
                        (_F.Rho,_F.Nml,_F.Nml,_F.Brk))

    for i,EVec in enumerate(gateset.EVecs):
        if confidenceRegionInfo is None:
            _F.AddTableRow(formats, tables, (i, EVec, _BT.pauliProdVectorToMatrixInStdBasis(EVec)),
                        (_F.E,_F.Nml,_F.Brk))
        else:
            intervalVec = confidenceRegionInfo.getProfileLikelihoodConfidenceIntervals("E%d" % i)[:,None]
            _F.AddTableRow(formats, tables, (i, EVec, intervalVec, _BT.pauliProdVectorToMatrixInStdBasis(EVec)),
                        (_F.Rho,_F.Nml,_F.Nml,_F.Brk))
            
    _F.FinishTable(formats, tables, longtable)
    return tables



def getGatesetSPAMParametersTable(gateset, formats, tableclass, longtable, confidenceRegionInfo=None):
    """ 
    Create latex table for gateset's "SPAM parameters", that is, the
    dot products of rho-vectors and E-vectors.
    
    Parameters
    ----------
    gateset : GateSet
        The GateSet

    longtable : bool
        Whether table should be a latex longtable or not.

    confidenceRegionInfo : ConfidenceRegion, optional
        If not None, specifies a confidence-region
        used to display error intervals.

    Returns
    -------
    string
        Latex string output.
    """
    colHeadings = [''] + [ i for i in gateset.getEVecIndices() ]
    formatters = [None] + [ _F.E ]*len(gateset.getEVecIndices())

    tables = {}
    _F.CreateTable(formats, tables, colHeadings, formatters, tableclass, longtable)

    spamDotProdsQty = _CR.compute_GateSet_Quantity("Spam DotProds", gateset, confidenceRegionInfo)
    DPs, DPEBs = spamDotProdsQty.getValueAndErrBar()

    formatters = [ _F.Rho ] + [ _F.EB ]*len(gateset.getEVecIndices()) #for rows below

    for ii,i in enumerate(gateset.getRhoVecIndices()): # i is rho index into gateset, ii enumerates these to index DPs
        rowData = [i]
        for jj,j in enumerate(gateset.getEVecIndices()): # j is E index into gateset, jj enumerates these to index DPs
            if confidenceRegionInfo is None:
                rowData.append((DPs[ii,jj],None))
            else:
                rowData.append((DPs[ii,jj],DPEBs[ii,jj]))
        _F.AddTableRow(formats, tables, rowData, formatters)

    _F.FinishTable(formats, tables, longtable)
    return tables


def getGatesetGatesTable(gateset, formats, tableclass, longtable, confidenceRegionInfo=None):
    """ 
    Create latex table for gateset's gates.
    
    Parameters
    ----------
    gateset : GateSet
        The GateSet

    longtable : bool
        Whether table should be a latex longtable or not.

    confidenceRegionInfo : ConfidenceRegion, optional
        If not None, specifies a confidence-region
        used to display error intervals.

    Returns
    -------
    string
        Latex string output.
    """
    gateLabels = gateset.keys()  # gate labels

    if confidenceRegionInfo is None:    
        colHeadings = ('Gate','Superoperator (Pauli basis)')
        formatters = (None,None)
    else:
        colHeadings = ('Gate','Superoperator (Pauli basis)','%g%% C.I. half-width' % confidenceRegionInfo.level)
        formatters = (None,None,_F.TxtCnv)

    tables = {}
    _F.CreateTable(formats, tables, colHeadings, formatters, tableclass, longtable)

    for gl in gateLabels:
        if confidenceRegionInfo is None:
            _F.AddTableRow(formats, tables, (gl, gateset[gl]), (None,_F.Brk))
        else:
            intervalVec = confidenceRegionInfo.getProfileLikelihoodConfidenceIntervals(gl)[:,None]
            if isinstance(gateset.get_gate(gl), _Gate.FullyParameterizedGate): #then we know how to reshape into a matrix
                nCols = gateset.get_dimension(); nRows = intervalVec.size / nCols
                intervalMx = intervalVec.reshape(nRows,nCols)
                if nRows == (nCols-1): #TP constrained, so pad with zero top row
                    intervalMx = _np.concatenate( (_np.zeros((1,nCols),'d'),intervalMx), axis=0 )
            else: 
                intervalMx = intervalVec # we don't know how best to reshape vector of parameter intervals, so don't
            _F.AddTableRow(formats, tables, (gl, gateset[gl], intervalMx), (None,_F.Brk,_F.Brk))

    _F.FinishTable(formats, tables, longtable)
    return tables


def getUnitaryGatesetGatesTable(gateset, formats, tableclass, longtable, confidenceRegionInfo=None):
    """ 
    Create latex table for gateset's gates assuming they're unitary.
    
    Parameters
    ----------
    gateset : GateSet
        The GateSet

    longtable : bool
        Whether table should be a latex longtable or not.

    confidenceRegionInfo : ConfidenceRegion, optional
        If not None, specifies a confidence-region
        used to display error intervals.

    Returns
    -------
    string
        Latex string output.
    """
    gateLabels = gateset.keys()  # gate labels

    qtys_to_compute = [ ('%s decomposition' % gl) for gl in gateLabels ]
    qtys = _CR.compute_GateSet_Quantities(qtys_to_compute, gateset, confidenceRegionInfo)

    if confidenceRegionInfo is None:
        colHeadings = ('Gate','Superoperator (Pauli basis)','Rotation axis','Angle')
        formatters = (None,None,None,None)
    else:
        colHeadings = ('Gate','Superoperator (Pauli basis)',
                       '%g%% C.I. half-width' % confidenceRegionInfo.level,
                       'Rotation axis','Angle')
        formatters = (None,None,_F.TxtCnv,None,None)
    
    tables = {}
    _F.CreateTable(formats, tables, colHeadings, formatters, tableclass, longtable)

    for gl in gateLabels:
        decomp, decompEB = qtys['%s decomposition' % gl].getValueAndErrBar()
        if confidenceRegionInfo is None:
            _F.AddTableRow(formats, tables, 
                        (gl, gateset[gl],decomp.get('axis of rotation','X'),decomp.get('pi rotations','X')),
                        (None, _F.Brk, _F.Nml, _F.Pi) )
        else:
            intervalVec = confidenceRegionInfo.getProfileLikelihoodConfidenceIntervals(gl)[:,None]
            if isinstance(gateset.get_gate(gl), _Gate.FullyParameterizedGate): #then we know how to reshape into a matrix
                nCols = gateset.get_dimension(); nRows = intervalVec.size / nCols
                intervalMx = intervalVec.reshape(nRows,nCols)
                if nRows == (nCols-1): #TP constrained, so pad with zero top row
                    intervalMx = _np.concatenate( (_np.zeros((1,nCols),'d'),intervalMx), axis=0 )
            else: 
                intervalMx = intervalVec # we don't know how best to reshape vector of parameter intervals, so don't

            _F.AddTableRow(formats, tables, 
                        (gl, gateset[gl],decomp.get('axis of rotation','X'), 
                         (decomp.get('pi rotations','X'), decompEB.get('pi rotations','X')) ),
                        (None, _F.Brk, _F.Nml, _F.EBPi) )

    _F.FinishTable(formats, tables, longtable)
    return tables


def getGatesetChoiTable(gateset, formats, tableclass, longtable, confidenceRegionInfo=None):
    """ 
    Create latex table for the Choi matrices of a gateset's gates.
    
    Parameters
    ----------
    gateset : GateSet
        The GateSet

    longtable : bool
        Whether table should be a latex longtable or not.

    confidenceRegionInfo : ConfidenceRegion, optional
        If not None, specifies a confidence-region
        used to display error intervals.

    Returns
    -------
    string
        Latex string output.
    """
    gateLabels = gateset.keys()  # gate labels

    qtys_to_compute = [ ('%s choi matrix in pauli basis' % gl) for gl in gateLabels ]
    qtys_to_compute += [ ('%s choi eigenvalues' % gl) for gl in gateLabels ]
    qtys = _CR.compute_GateSet_Quantities(qtys_to_compute, gateset, confidenceRegionInfo)

    if confidenceRegionInfo is None:
        colHeadings = ('Gate','Choi matrix (Pauli basis)','Eigenvalues')
        formatters = (None,None,None)        
    else:
        colHeadings = ('Gate','Choi matrix (Pauli basis)','Eigenvalues') # 'Confidence Intervals',
        formatters = (None,None,None)

    tables = {}
    _F.CreateTable(formats, tables, colHeadings, formatters, tableclass, longtable)

    for gl in gateLabels:
        choiMx,choiEB = qtys['%s choi matrix in pauli basis' % gl].getValueAndErrBar()
        evals, evalsEB = qtys['%s choi eigenvalues' % gl].getValueAndErrBar()
    
        if confidenceRegionInfo is None:
            _F.AddTableRow(formats, tables, (gl, choiMx, evals), (None, _F.Brk, _F.Nml))
        else:
            _F.AddTableRow(formats, tables, (gl, choiMx, (evals,evalsEB)), (None, _F.Brk, _F.EBvec))

    _F.FinishTable(formats, tables, longtable)
    return tables


def getGatesetVsTargetTable(gateset, targetGateset, formats, tableclass, longtable, confidenceRegionInfo=None):
    """ 
    Create latex table comparing a gateset to a target gateset.
    
    Parameters
    ----------
    gateset, targetGateset : GateSet
        The gate sets to compare

    longtable : bool
        Whether table should be a latex longtable or not.

    confidenceRegionInfo : ConfidenceRegion, optional
        If not None, specifies a confidence-region
        used to display error intervals.

    Returns
    -------
    string
        Latex string output.
    """
    gateLabels = gateset.keys()  # gate labels

    colHeadings = ('Gate', "Process|Infidelity", "1/2 Trace|Distance", "1/2 Diamond-Norm", "Frobenius|Distance")
    formatters = (None,_F.TxtCnv,_F.TxtCnv,_F.TxtCnv,_F.TxtCnv)

    qtyNames = ('infidelity','Jamiolkowski trace dist','diamond norm','Frobenius diff')
    qtys_to_compute = [ '%s %s' % (gl,qty) for qty in qtyNames for gl in gateLabels ]
    qtys = _CR.compute_GateSet_GateSet_Quantities(qtys_to_compute, gateset, targetGateset, confidenceRegionInfo)

    tables = {}
    _F.CreateTable(formats, tables, colHeadings, formatters, tableclass, longtable)

    formatters = [None] + [ _F.EB ]*len(qtyNames)
    
    for gl in gateLabels:
        if confidenceRegionInfo is None:
            rowData = [gl] + [ (qtys['%s %s' % (gl,qty)].getValue(),None) for qty in qtyNames ]
        else:
            rowData = [gl] + [ qtys['%s %s' % (gl,qty)].getValueAndErrBar() for qty in qtyNames ]
        _F.AddTableRow(formats, tables, rowData, formatters)

    _F.FinishTable(formats, tables, longtable)
    return tables


def getGatesetVsTargetErrGenTable(gateset, targetGateset, formats, tableclass, longtable, confidenceRegionInfo=None):
    """ 
    Create latex table listing the error generators obtained by 
    comparing a gateset to a target gateset.
    
    Parameters
    ----------
    gateset, targetGateset : GateSet
        The gate sets to compare

    longtable : bool
        Whether table should be a latex longtable or not.

    confidenceRegionInfo : ConfidenceRegion, optional
        If not None, specifies a confidence-region
        used to display error intervals.

    Returns
    -------
    string
        Latex string output.
    """
    gateLabels = gateset.keys()  # gate labels
    colHeadings = ('Gate','Error Generator')

    tables = {}
    _F.CreateTable(formats, tables, colHeadings, (None,None), tableclass, longtable)
    for gl in gateLabels:
        _F.AddTableRow(formats, tables, (gl, _spl.logm(_np.dot(_np.linalg.inv(targetGateset[gl]),gateset[gl]))),
                    (None, _F.Brk))
    _F.FinishTable(formats, tables, longtable)
    return tables


def getGatesetClosestUnitaryTable(gateset, formats, tableclass, longtable, confidenceRegionInfo=None):
    """ 
    Create latex table for gateset that contains closest-unitary gates.
    
    Parameters
    ----------
    gateset : GateSet
        The GateSet

    longtable : bool
        Whether table should be a latex longtable or not.

    confidenceRegionInfo : ConfidenceRegion, optional
        If not None, specifies a confidence-region
        used to display error intervals.

    Returns
    -------
    string
        Latex string output.
    """

    gateLabels = gateset.keys()  # gate labels
    colHeadings = ('Gate','Process|Infidelity','1/2 Trace|Distance','Rotation|Axis','Rotation|Angle','Sanity Check')
    formatters = (None,_F.TxtCnv,_F.TxtCnv,_F.TxtCnv,_F.TxtCnv,_F.TxtCnv)

    qtyNames = ('max fidelity with unitary', 'max trace dist with unitary',
                'closest unitary decomposition', 'upper bound on fidelity with unitary')
    qtys_to_compute = [ '%s %s' % (gl,qty) for qty in qtyNames for gl in gateLabels ]
    qtys = _CR.compute_GateSet_Quantities(qtys_to_compute, gateset, confidenceRegionInfo)
    decompNames = ('axis of rotation','pi rotations')
    #Other possible qtyName: 'closest unitary choi matrix in pauli basis'

    tables = {}
    _F.CreateTable(formats, tables, colHeadings, formatters, tableclass, longtable)

    formatters = [None, _F.EB, _F.EB, _F.EBvec, _F.EBPi, _F.Nml ] # Note len(decompNames)==2, 2nd el is rotn angle

    for gl in gateLabels:
        fUB,fUB_EB = qtys['%s upper bound on fidelity with unitary' % gl].getValueAndErrBar()
        fLB,fLB_EB = qtys['%s max fidelity with unitary' % gl].getValueAndErrBar()
        td, td_EB = qtys['%s max trace dist with unitary' % gl].getValueAndErrBar()
        sanity = (1.0-fLB)/(1.0-fUB) - 1.0 #Robin's sanity check metric (0=good, >1=bad)
        decomp, decompEB = qtys['%s closest unitary decomposition' % gl].getValueAndErrBar()
        
        if confidenceRegionInfo is None:        
            rowData = [gl, (1.0-fLB,None), (td,None)] + [(decomp.get(x,'X'),None) for x in decompNames]
        else:
            rowData = [gl, (1.0-fLB,fLB_EB), (td,td_EB)] + [(decomp.get(x,'X'),decompEB.get(x,None)) for x in decompNames]
        rowData.append(sanity)

        _F.AddTableRow(formats, tables, rowData, formatters)

    _F.FinishTable(formats, tables, longtable)
    return tables


def getGatesetDecompTable(gateset, formats, tableclass, longtable, confidenceRegionInfo=None):
    """ 
    Create latex tables for decomposing a gateset's gates.
    
    Parameters
    ----------
    gateset : GateSet
        The GateSet

    longtable : bool
        Whether table should be a latex longtable or not.

    confidenceRegionInfo : ConfidenceRegion, optional
        If not None, specifies a confidence-region
        used to display error intervals.

    Returns
    -------
    string
        Latex string output.
    """
    gateLabels = gateset.keys()  # gate labels
    colHeadings = ('Gate','Eigenvalues','Fixed pt','Rotn. axis','Diag. decay','Off-diag. decay')
    formatters = [None]*6

    qtyNames = ('eigenvalues','decomposition')
    qtys_to_compute = [ '%s %s' % (gl,qty) for qty in qtyNames for gl in gateLabels ]
    qtys = _CR.compute_GateSet_Quantities(qtys_to_compute, gateset, confidenceRegionInfo)
    decompNames = ('fixed point',
                   'axis of rotation',
                   'decay of diagonal rotation terms',
                   'decay of off diagonal rotation terms')

    tables = {}
    _F.CreateTable(formats, tables, colHeadings, formatters, tableclass, longtable)

    formatters = (None, _F.EBvec, _F.Nml, _F.Nml, _F.EB, _F.EB)

    for gl in gateLabels:
        decomp, decompEB = qtys['%s decomposition' % gl].getValueAndErrBar()

        if confidenceRegionInfo is None or decompEB is None: #decompEB is None when gate decomp failed
            evals = qtys['%s eigenvalues' % gl].getValue()
            rowData = [gl, (evals,None)] + [decomp.get(x,'X') for x in decompNames[0:2] ] + \
                [(decomp.get(x,'X'),None) for x in decompNames[2:4] ]
        else:
            evals, evalsEB = qtys['%s eigenvalues' % gl].getValueAndErrBar()
            rowData = [gl, (evals,evalsEB)] + [decomp.get(x,'X') for x in decompNames[0:2] ] + \
                [(decomp.get(x,'X'),decompEB.get(x,'X')) for x in decompNames[2:4] ]

        _F.AddTableRow(formats, tables, rowData, formatters)

    _F.FinishTable(formats, tables, longtable)
    return tables


def getGatesetRotnAxisTable(gateset, formats, tableclass, longtable, confidenceRegionInfo=None):
    """ 
    Create latex table of the angle between a gate rotation axes for 
     gates belonging to a gateset
    
    Parameters
    ----------
    gateset : GateSet
        The GateSet

    longtable : bool
        Whether table should be a latex longtable or not.

    confidenceRegionInfo : ConfidenceRegion, optional
        If not None, specifies a confidence-region
        used to display error intervals.

    Returns
    -------
    string
        Latex string output.
    """
    gateLabels = gateset.keys()  # gate labels

    qtyNames = ('decomposition',)
    qtys_to_compute = [ '%s %s' % (gl,qty) for qty in qtyNames for gl in gateLabels ]
    qtys = _CR.compute_GateSet_Quantities(qtys_to_compute, gateset, confidenceRegionInfo)

    colHeadings = ("Gate","Angle") + tuple( [ "RAAW(%s)" % gl for gl in gateLabels] )
    nCols = len(colHeadings)
    formatters = [None] * nCols

    table = "longtable" if longtable else "tabular"
    latex_head =  "\\begin{%s}[l]{%s}\n\hline\n" % (table, "|c" * nCols + "|")
    latex_head += "\\multirow{2}{*}{Gate} & \\multirow{2}{*}{Angle} & " + \
                  "\\multicolumn{%d}{c|}{Angle between Rotation Axes} \\\\ \cline{3-%d}\n" % (len(gateLabels),nCols)
    latex_head += " & & %s \\\\ \hline\n" % (" & ".join(gateLabels))

    tables = {}
    _F.CreateTable(formats, tables, colHeadings, formatters, tableclass, longtable, customHeader={'latex': latex_head} )

    formatters = [None, _F.EB] + [ _F.Pi ] * len(gateLabels)

    for gl in gateLabels:
        decomp, decompEB = qtys['%s decomposition' % gl].getValueAndErrBar()
        rotnAngle = decomp.get('pi rotations','X')

        angles_btwn_rotn_axes = []
        axisOfRotn = decomp.get('axis of rotation',None)
        for gl_other in gateLabels:

            if gl_other == gl:  
                angles_btwn_rotn_axes.append( "" )
            else:
                decomp_other, decompEB_other = qtys['%s decomposition' % gl_other].getValueAndErrBar()
                rotnAngle_other = decomp_other.get('pi rotations','X')                
                if rotnAngle == 'X' or abs(rotnAngle) < 1e-4 or \
                   rotnAngle_other == 'X' or abs(rotnAngle_other) < 1e-4:
                    angles_btwn_rotn_axes.append( "--" )
                else:
                    axisOfRotn_other = decomp_other.get('axis of rotation',None)
                    if axisOfRotn is not None and axisOfRotn_other is not None:
                        real_dot =  _np.clip( _np.real(_np.dot(axisOfRotn,axisOfRotn_other)), 0.0, 1.0)
                        angles_btwn_rotn_axes.append( _np.arccos( real_dot ) / _np.pi )
                    else: 
                        angles_btwn_rotn_axes.append( "X" )
        
        if confidenceRegionInfo is None or decompEB is None: #decompEB is None when gate decomp failed
            rowData = [gl, (rotnAngle,None)] + angles_btwn_rotn_axes
        else:
            rowData = [gl, (rotnAngle,decompEB.get('pi rotations','X'))] + angles_btwn_rotn_axes
        _F.AddTableRow(formats, tables, rowData, formatters)

    _F.FinishTable(formats, tables, longtable)
    return tables


def getDatasetOverviewTable(dataset, formats, tableclass, longtable):
    """ 
    Create latex table overviewing a data set.
    
    Parameters
    ----------
    dataset : DataSet
        The DataSet

    longtable : bool
        Whether table should be a latex longtable or not.

    Returns
    -------
    string
        Latex string output.
    """
    colHeadings = ('Quantity','Value')
    formatters = (None,None)
    rank,evals = _GM.maxGramRankAndEvals( dataset )

    tables = {}
    _F.CreateTable(formats, tables, colHeadings, formatters, tableclass, longtable)

    _F.AddTableRow(formats, tables, ("Number of strings", str(len(dataset))), (None,None))
    _F.AddTableRow(formats, tables, ("Gate labels", ", ".join(dataset.getGateLabels()) ), (None,None))
    _F.AddTableRow(formats, tables, ("SPAM labels",  ", ".join(dataset.getSpamLabels()) ), (None,None))
    _F.AddTableRow(formats, tables, ("Gram singular vals", _np.sort(abs(evals)).reshape(-1,1) ), (None,_F.Sml))

    _F.FinishTable(formats, tables, longtable)
    return tables


def getChi2ProgressTable(Ls, gatesetsByL, gateStringsByL, dataset, TPconstrained, formats, tableclass, longtable):
    """ 
    Create latex table showing how Chi2 changes with GST iteration.
    
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

    TPconstrained : bool
        Whether gatesetsByL were optimized under a TP constraint.

    longtable : bool
        Whether table should be a latex longtable or not.

    Returns
    -------
    string
        Latex string output.
    """
    colHeadings = { 'latex': ('L','$\\chi^2$','$k$','$\\chi^2-k$','$\sqrt{2k}$','$P$','$N_s$','$N_p$', 'Rating'),
                    'html': ('L','&chi;<sup>2</sup>','k','&chi;<sup>2</sup>-k',
                             '&radic;<span style="text-decoration:overline;">2k</span>',
                             'P','N<sub>s</sub>','N<sub>p</sub>', 'Rating'),
                    'py': ('L','chi^2','k','chi^2-k','sqrt{2k}','P','N_s','N_p', 'Rating'),
                    'ppt': ('L','chi^2','k','chi^2-k','sqrt{2k}','P','N_s','N_p', 'Rating')
                  }
    tables = {}
    _F.CreateTable_preformatted(formats, tables, colHeadings, tableclass, longtable)

    for L,gs,gstrs in zip(Ls,gatesetsByL,gateStringsByL):
        chi2 = _AT.TotalChiSquared( dataset, gs, gstrs, minProbClipForWeighting=1e-4)
        Ns = len(gstrs)

        #Get number of gateset parameters - this should match algorithm used
        if TPconstrained:
            Np = gs.getNumNonGaugeParams(gates=True,G0=False,SPAM=True,SP0=False) 
        else:
            Np = gs.getNumNonGaugeParams() #include everything

        k = Ns-Np #expected chi^2 mean
        pv = 1.0 - _stats.chi2.cdf(chi2,k) # reject GST model if p-value < threshold (~0.05?)

        if   (chi2-k) < _np.sqrt(2*k): rating = 5
        elif (chi2-k) < 2*k: rating = 4
        elif (chi2-k) < 5*k: rating = 3
        elif (chi2-k) < 10*k: rating = 2
        else: rating = 1
        _F.AddTableRow(formats, tables, 
                    (str(L),chi2,k,chi2-k,_np.sqrt(2*k),pv,Ns,Np,"<STAR>"*rating),
                    (None,_F.Nml,_F.Nml,_F.Nml,_F.Nml,_F.Nml2,_F.Nml,_F.Nml,_F.TxtCnv))

    _F.FinishTable(formats, tables, longtable)
    return tables


def getLogLProgressTable(Ls, gatesetsByL, gateStringsByL, dataset, TPconstrained, formats, tableclass, longtable):
    """ 
    Create latex table showing how the log-likelihood changes with GST iteration.
    
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

    TPconstrained : bool
        Whether gatesetsByL were optimized under a TP constraint.

    longtable : bool
        Whether table should be a latex longtable or not.

    Returns
    -------
    string
        Latex string output.
    """
    colHeadings = { 'latex': ('L','$2\Delta\\log(\\mathcal{L})$','$k$','$2\Delta\\log(\\mathcal{L})-k$',
                              '$\sqrt{2k}$','$P$','$N_s$','$N_p$', 'Rating'),
                    'html': ('L','2&Delta;(log L)','k','2&Delta;(log L)-k',
                             '&radic;<span style="text-decoration:overline;">2k</span>',
                             'P','N<sub>s</sub>','N<sub>p</sub>', 'Rating'),
                    'py': ('L','2*Delta(log L)','k','2*Delta(log L)-k','sqrt{2k}','P','N_s','N_p', 'Rating'),
                    'ppt': ('L','2*Delta(log L)','k','2*Delta(log L)-k','sqrt{2k}','P','N_s','N_p', 'Rating')
                  }
    tables = {}
    _F.CreateTable_preformatted(formats, tables, colHeadings, tableclass, longtable)

    for L,gs,gstrs in zip(Ls,gatesetsByL,gateStringsByL):
        logL_upperbound = _LF.logL_max(dataset, gstrs)
        logL = _LF.logL( gs, dataset, gstrs )
        if(logL_upperbound < logL):
            raise ValueError("LogL upper bound = %g but logL = %g!!" % (logL_upperbound, logL))
        Ns = len(gstrs)

        #Get number of gateset parameters - this should match algorithm used
        if TPconstrained:
            Np = gs.getNumNonGaugeParams(gates=True,G0=False,SPAM=True,SP0=False) 
        else:
            Np = gs.getNumNonGaugeParams() #include everything

        k = Ns-Np #expected 2*(logL_ub-logL) mean
        twoDeltaLogL = 2*(logL_upperbound - logL)
        pv = 1.0 - _stats.chi2.cdf(twoDeltaLogL,k) # reject GST model if p-value < threshold (~0.05?)

        if   (twoDeltaLogL-k) < _np.sqrt(2*k): rating = 5
        elif (twoDeltaLogL-k) < 2*k: rating = 4
        elif (twoDeltaLogL-k) < 5*k: rating = 3
        elif (twoDeltaLogL-k) < 10*k: rating = 2
        else: rating = 1

        _F.AddTableRow(formats, tables, 
                    (str(L),twoDeltaLogL,k,twoDeltaLogL-k,_np.sqrt(2*k),pv,Ns,Np,"<STAR>"*rating),
                    (None,_F.Nml,_F.Nml,_F.Nml,_F.Nml,_F.Nml2,_F.Nml,_F.Nml,_F.TxtCnv))

    _F.FinishTable(formats, tables, longtable)
    return tables
    

def getGatestringTable(gsList, title, formats, tableclass, longtable):
    """ 
    Creates a 2-column latex table enumerating a list of gate strings.
    
    Parameters
    ----------
    gsList : list of GateStrings
        List of gate strings to put in table.
        
    title : string
        The title for the table column containing the strings.

    longtable : bool
        Whether table should be a latex longtable or not.

    Returns
    -------
    string
        Latex string output.
    """
    colHeadings = ('#',title)
    formatters = (_F.TxtCnv,_F.Nml)

    tables = {}
    _F.CreateTable(formats, tables, colHeadings, formatters, tableclass, longtable)

    for i,gstr in enumerate(gsList,start=1):
        _F.AddTableRow(formats, tables, (i, gstr), (_F.Nml,_F.GStr) )

    _F.FinishTable(formats, tables, longtable)
    return tables


def getGatestringMultiTable(gsLists, titles, formats, tableclass, longtable, commonTitle=None):
    """ 
    Creates an N-column latex table enumerating a N-1 lists of gate strings.
    
    Parameters
    ----------
    gsLists : list of GateString lists
        List of gate strings to put in table.
        
    titles : list of strings
        The titles for the table columns containing the strings.

    longtable : bool
        Whether table should be a latex longtable or not.

    commonTitle : string, optional
        A single title string to place in a cell spanning across
        all the gate string columns.

    Returns
    -------
    string
        Latex string output.
    """
    colHeadings = ('#',) + tuple(titles)
    formatters = (_F.TxtCnv,) + (_F.Nml,)*len(titles)

    tables = {}

    if commonTitle is not None:
        _F.CreateTable(formats, tables, colHeadings, formatters, tableclass, longtable)
    else:
        colHeadings = ('\\#',) + tuple(titles)
        latex_head  = "\\begin{%s}[l]{%s}\n\hline\n" % (table, "|c" * len(colHeadings) + "|")
        latex_head += " & \multicolumn{%d}{c|}{%s} \\\\ \hline\n" % (len(titles),commonTitle)
        latex_head += "%s \\\\ \hline\n" % (" & ".join(colHeadings))

        html_head = "<table class=%s><thead>" % tableclass
        html_head += '<tr><th></th><th colspan="%d">%s</th></tr>\n' % (len(titles),commonTitle)
        html_head += "<tr><th> %s </th></tr>" % (" </th><th> ".join(colHeadings))
        html_head += "</thead><tbody>"

        _F.CreateTable(formats, tables, colHeadings, formatters, tableclass, longtable, 
                    customHeader={'latex': latex_head, 'html': html_head})

    formatters = (_F.Nml,) + (_F.GStr,)*len(gsLists)

    for i in range( max([len(gsl) for gsl in gsLists]) ):
        rowData = [i+1]
        for gsList in gsLists:
            if i < len(gsList):
                rowData.append( gsList[i] )
            else:
                rowData.append( None ) #empty string
        _F.AddTableRow(formats, tables, rowData, formatters)

    _F.FinishTable(formats, tables, longtable)
    return tables




def constructLogLConfidenceRegion(gateset, dataset, confidenceLevel, TPconstrained=True,
                                  gateStringList=None, probClipInterval=(-1e6,1e6),
                                  minProbClip=1e-4, radius=1e-4, hessianProjection="std"):

    """ 
    Constructs a ConfidenceRegion given a gateset and dataset using the log-likelihood Hessian.
    (Internally, this evaluates the log-likelihood Hessian.)

    Parameters
    ----------
    gateset : GateSet
        the gate set point estimate that maximizes the logL or minimizes 
        the chi2, and marks the point in gateset-space where the Hessian
        has been evaluated.

    dataset : DataSet
        Probability data

    confidenceLevel : float
        If not None, then the confidence level (between 0 and 100) used in
        the computation of confidence regions/intervals. If None, no 
        confidence regions or intervals are computed.

    TPconstrained : bool, optional
        Whether to constrain GST to trace-preserving gatesets.

    gateStringList : list of (tuples or GateStrings), optional
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

    Returns
    -------
    ConfidenceRegion
    """

    gates = G0 = SPAM = SP0 = True    
    if TPconstrained:
        G0 = SP0 = False

    if gateStringList is None:
        gateStringList = dataset.keys()
        
    #Compute appropriate Hessian
    hessian = _LF.logL_hessian(gateset, dataset, gateStringList, gates, G0, SPAM, SP0, 
                               minProbClip, probClipInterval, radius) 

    cri = _ConfidenceRegion(gateset, hessian, confidenceLevel, gates, G0, SPAM, SP0, hessianProjection)

    #Check that number of gauge parameters reported by gateset is consistent with confidence region
    # since the parameter number computed this way is used in chi2 or logL progress tables
    if TPconstrained:
        Np_check = gateset.getNumNonGaugeParams(G0=False,SP0=False)
    else: Np_check =  gateset.getNumNonGaugeParams()
    if(Np_check != cri.nNonGaugeParams):
        _warnings.warn("Number of non-gauge parameters in gateset and confidence region do " 
                       + " not match.  This indicates an internal logic error.")            
    
    return cri



def constructChi2ConfidenceRegion(gateset, dataset, confidenceLevel, TPconstrained=True,
                                  gateStringList=None, probClipInterval=(-1e6,1e6),
                                  minProbClipForWeighting=1e-4, hessianProjection="std"):

    """ 
    Constructs a ConfidenceRegion given a gateset and dataset using the Chi2 Hessian.
    (Internally, this evaluates the Chi2 Hessian.)

    Parameters
    ----------
    gateset : GateSet
        the gate set point estimate that maximizes the logL or minimizes 
        the chi2, and marks the point in gateset-space where the Hessian
        has been evaluated.

    dataset : DataSet
        Probability data

    confidenceLevel : float
        If not None, then the confidence level (between 0 and 100) used in
        the computation of confidence regions/intervals. If None, no 
        confidence regions or intervals are computed.

    TPconstrained : bool, optional
        Whether to constrain GST to trace-preserving gatesets.

    gateStringList : list of (tuples or GateStrings), optional
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


    Returns
    -------
    ConfidenceRegion
    """

    gates = G0 = SPAM = SP0 = True    
    if TPconstrained:
        G0 = SP0 = False

    if gateStringList is None:
        gateStringList = dataset.keys()
        
    #Compute appropriate Hessian
    dummy_chi2, hessian = _AT.TotalChiSquared(dataset, gateset, gateStringList, False, True,
                                              G0, SP0, SPAM, gates, minProbClipForWeighting,
                                              probClipInterval)

    cri = _ConfidenceRegion(gateset, hessian, confidenceLevel, gates, G0, SPAM, SP0, hessianProjection)

    #Check that number of gauge parameters reported by gateset is consistent with confidence region
    # since the parameter number computed this way is used in chi2 or logL progress tables
    if TPconstrained:
        Np_check = gateset.getNumNonGaugeParams(G0=False,SP0=False)
    else: Np_check =  gateset.getNumNonGaugeParams()
    if(Np_check != cri.nNonGaugeParams):
        _warnings.warn("Number of non-gauge parameters in gateset and confidence region do " 
                       + " not match.  This indicates an internal logic error.")            
    
    return cri

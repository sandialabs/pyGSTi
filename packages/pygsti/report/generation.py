#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation              
#    This Software is released under the GPL license detailed    
#    in the file "license.txt" in the top-level pyGSTi directory 
#*****************************************************************
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

from .. import algorithms as _alg
from .. import tools as _tools
from .. import objects as _objs

import reportables as _cr
import tableformat as _tf



def get_blank_table(formats):
    """ Create a blank table as a placeholder with the given formats """
    tables = {}
    _tf.create_table(formats, tables, ['Blank'], [None], "", False)
    _tf.finish_table(formats, tables, False)
    return tables
    

def get_gateset_spam_table(gateset, formats, tableclass, longtable, 
                           confidenceRegionInfo=None, mxBasis="gm"):
    """
    Create a table for gateset's SPAM vectors.
    
    Parameters
    ----------
    gateset : GateSet
        The GateSet

    formats : list
        List of formats to include in returned tables dictionary. Allowed
        formats are 'latex', 'html', 'py', and 'ppt'.

    tableclass : string
        CSS class to apply to the HTML table.

    longtable : bool
        Whether table should be a latex longtable or not.

    confidenceRegionInfo : ConfidenceRegion, optional
        If not None, specifies a confidence-region
        used to display error intervals.

    mxBasis : {'std', 'gm','pp'}
        Which basis the gateset is represented in.  Allowed
        options are Matrix-unit (std), Gell-Mann (gm) and
        Pauli-product (pp).

    Returns
    -------
    dict
        Dictionary with keys equal to the requested formats (e.g. 'latex'),
        and values equal to the table data in the corresponding format.
    """
    
    basisNm = _tools.basis_longname(mxBasis, gateset.get_dimension())

    if confidenceRegionInfo is None:
        colHeadings = ('Operator','Hilbert-Schmidt vector (%s basis)' % basisNm,'Matrix')
        formatters = (None,None,None)
    else:
        colHeadings = ('Operator',
                       'Hilbert-Schmidt vector (%s basis)' % basisNm,
                       '%g%% C.I. half-width' % confidenceRegionInfo.level,
                       'Matrix')
        formatters = (None,None,_tf.TxtCnv,None)

    tables = {}
    _tf.create_table(formats, tables, colHeadings, formatters, tableclass, longtable)

    for lbl,rhoVec in gateset.preps.iteritems():
        if mxBasis == "pp":   rhoMx = _tools.ppvec_to_stdmx(rhoVec)
        elif mxBasis == "gm": rhoMx = _tools.gmvec_to_stdmx(rhoVec)
        elif mxBasis == "std": rhoMx = _tools.stdvec_to_stdmx(rhoVec)
        else: raise ValueError("Invalid basis specifier: %s" % mxBasis)

        if confidenceRegionInfo is None:
            _tf.add_table_row(formats, tables, (lbl, rhoVec, rhoMx), (_tf.Rho,_tf.Nml,_tf.Brk))
        else:
            intervalVec = confidenceRegionInfo.get_profile_likelihood_confidence_intervals(lbl)[:,None]
            if intervalVec.shape[0] == gateset.get_dimension()-1: #TP constrained, so pad with zero top row
                intervalVec = _np.concatenate( (_np.zeros((1,1),'d'),intervalVec), axis=0 )
            _tf.add_table_row(formats, tables, (lbl, rhoVec, intervalVec, rhoMx), (_tf.Rho,_tf.Nml,_tf.Nml,_tf.Brk))

    for lbl,EVec in gateset.effects.iteritems():
        if mxBasis == "pp":    EMx = _tools.ppvec_to_stdmx(EVec)
        elif mxBasis == "gm":  EMx = _tools.gmvec_to_stdmx(EVec)
        elif mxBasis == "std": EMx = _tools.stdvec_to_stdmx(EVec)
        else: raise ValueError("Invalid basis specifier: %s" % mxBasis)

        if confidenceRegionInfo is None:
            _tf.add_table_row(formats, tables, (lbl, EVec, EMx), (_tf.E,_tf.Nml,_tf.Brk))
        else:
            intervalVec = confidenceRegionInfo.get_profile_likelihood_confidence_intervals(lbl)[:,None]
            _tf.add_table_row(formats, tables, (lbl, EVec, intervalVec, EMx), (_tf.E,_tf.Nml,_tf.Nml,_tf.Brk))
            
    _tf.finish_table(formats, tables, longtable)
    return tables



def get_gateset_spam_parameters_table(gateset, formats, tableclass, longtable, confidenceRegionInfo=None):
    """ 
    Create a table for gateset's "SPAM parameters", that is, the
    dot products of prep-vectors and effect-vectors.
    
    Parameters
    ----------
    gateset : GateSet
        The GateSet

    formats : list
        List of formats to include in returned tables dictionary. Allowed
        formats are 'latex', 'html', 'py', and 'ppt'.

    tableclass : string
        CSS class to apply to the HTML table.

    longtable : bool
        Whether table should be a latex longtable or not.

    confidenceRegionInfo : ConfidenceRegion, optional
        If not None, specifies a confidence-region
        used to display error intervals.

    Returns
    -------
    dict
        Dictionary with keys equal to the requested formats (e.g. 'latex'),
        and values equal to the table data in the corresponding format.
    """
    colHeadings = [''] + list(gateset.get_effect_labels())
    formatters = [None] + [ _tf.E ]*len(gateset.get_effect_labels())

    tables = {}
    _tf.create_table(formats, tables, colHeadings, formatters, tableclass, longtable)

    spamDotProdsQty = _cr.compute_gateset_qty("Spam DotProds", gateset, confidenceRegionInfo)
    DPs, DPEBs = spamDotProdsQty.get_value_and_err_bar()

    formatters = [ _tf.Rho ] + [ _tf.EB ]*len(gateset.get_effect_labels()) #for rows below

    for ii,prepLabel in enumerate(gateset.get_prep_labels()): # ii enumerates rhoLabels to index DPs
        rowData = [prepLabel]
        for jj,effectLabel in enumerate(gateset.get_effect_labels()): # jj enumerates eLabels to index DPs
            if confidenceRegionInfo is None:
                rowData.append((DPs[ii,jj],None))
            else:
                rowData.append((DPs[ii,jj],DPEBs[ii,jj]))
        _tf.add_table_row(formats, tables, rowData, formatters)

    _tf.finish_table(formats, tables, longtable)
    return tables


def get_gateset_gates_table(gateset, formats, tableclass, longtable,
                            confidenceRegionInfo=None, mxBasis="gm"):
    """ 
    Create a table for gateset's gates.
    
    Parameters
    ----------
    gateset : GateSet
        The GateSet

    formats : list
        List of formats to include in returned tables dictionary. Allowed
        formats are 'latex', 'html', 'py', and 'ppt'.

    tableclass : string
        CSS class to apply to the HTML table.

    longtable : bool
        Whether table should be a latex longtable or not.

    confidenceRegionInfo : ConfidenceRegion, optional
        If not None, specifies a confidence-region
        used to display error intervals.

    mxBasis : {'std', 'gm','pp'}
        Which basis the gateset is represented in.  Allowed
        options are Matrix-unit (std), Gell-Mann (gm) and
        Pauli-product (pp).


    Returns
    -------
    dict
        Dictionary with keys equal to the requested formats (e.g. 'latex'),
        and values equal to the table data in the corresponding format.
    """
    gateLabels = gateset.gates.keys()  # gate labels
    basisNm = _tools.basis_longname(mxBasis, gateset.get_dimension())

    if confidenceRegionInfo is None:    
        colHeadings = ('Gate','Superoperator (%s basis)' % basisNm)
        formatters = (None,None)
    else:
        colHeadings = ('Gate','Superoperator (%s basis)' % basisNm,
                       '%g%% C.I. half-width' % confidenceRegionInfo.level)
        formatters = (None,None,_tf.TxtCnv)

    tables = {}
    _tf.create_table(formats, tables, colHeadings, formatters, tableclass, longtable)

    for gl in gateLabels:
        if confidenceRegionInfo is None:
            _tf.add_table_row(formats, tables, (gl, gateset.gates[gl]), (None,_tf.Brk))
        else:
            intervalVec = confidenceRegionInfo.get_profile_likelihood_confidence_intervals(gl)[:,None]
            if isinstance(gateset.gates[gl], _objs.FullyParameterizedGate): #then we know how to reshape into a matrix
                gate_dim = gateset.get_dimension()
                intervalMx = intervalVec.reshape(gate_dim,gate_dim)
            elif isinstance(gateset.gates[gl], _objs.TPParameterizedGate): #then we know how to reshape into a matrix
                gate_dim = gateset.get_dimension()
                intervalMx = _np.concatenate( ( _np.zeros((1,gate_dim),'d'),
                                                intervalVec.reshape(gate_dim-1,gate_dim)), axis=0 )
            else: 
                intervalMx = intervalVec # we don't know how best to reshape vector of parameter intervals, so don't
            _tf.add_table_row(formats, tables, (gl, gateset.gates[gl], intervalMx), (None,_tf.Brk,_tf.Brk))

    _tf.finish_table(formats, tables, longtable)
    return tables


def get_unitary_gateset_gates_table(gateset, formats, tableclass, longtable, 
                                    confidenceRegionInfo=None, mxBasis="gm"):
    """ 
    Create a table for gateset's gates assuming they're unitary.
    
    Parameters
    ----------
    gateset : GateSet
        The GateSet

    formats : list
        List of formats to include in returned tables dictionary. Allowed
        formats are 'latex', 'html', 'py', and 'ppt'.

    tableclass : string
        CSS class to apply to the HTML table.

    longtable : bool
        Whether table should be a latex longtable or not.

    confidenceRegionInfo : ConfidenceRegion, optional
        If not None, specifies a confidence-region
        used to display error intervals.

    mxBasis : {'std', 'gm','pp'}
        Which basis the gateset is represented in.  Allowed
        options are Matrix-unit (std), Gell-Mann (gm) and
        Pauli-product (pp).


    Returns
    -------
    dict
        Dictionary with keys equal to the requested formats (e.g. 'latex'),
        and values equal to the table data in the corresponding format.
    """
    gateLabels = gateset.gates.keys()  # gate labels
    basisNm = _tools.basis_longname(mxBasis, gateset.get_dimension())

    qtys_to_compute = [ ('%s decomposition' % gl) for gl in gateLabels ]
    qtys = _cr.compute_gateset_qtys(qtys_to_compute, gateset, confidenceRegionInfo)

    if confidenceRegionInfo is None:
        colHeadings = ('Gate','Superoperator (%s basis)' % basisNm,'Rotation axis','Angle')
        formatters = (None,None,None,None)
    else:
        colHeadings = ('Gate','Superoperator (%s basis)' % basisNm,
                       '%g%% C.I. half-width' % confidenceRegionInfo.level,
                       'Rotation axis','Angle')
        formatters = (None,None,_tf.TxtCnv,None,None)
    
    tables = {}
    _tf.create_table(formats, tables, colHeadings, formatters, tableclass, longtable)

    for gl in gateLabels:
        decomp, decompEB = qtys['%s decomposition' % gl].get_value_and_err_bar()
        if confidenceRegionInfo is None:
            _tf.add_table_row(formats, tables, 
                        (gl, gateset.gates[gl],decomp.get('axis of rotation','X'),decomp.get('pi rotations','X')),
                        (None, _tf.Brk, _tf.Nml, _tf.Pi) )
        else:
            intervalVec = confidenceRegionInfo.get_profile_likelihood_confidence_intervals(gl)[:,None]
            if isinstance(gateset.gates[gl], _objs.FullyParameterizedGate): #then we know how to reshape into a matrix
                nCols = gateset.get_dimension(); nRows = intervalVec.size / nCols
                intervalMx = intervalVec.reshape(nRows,nCols)
                if nRows == (nCols-1): #TP constrained, so pad with zero top row
                    intervalMx = _np.concatenate( (_np.zeros((1,nCols),'d'),intervalMx), axis=0 )
            else: 
                intervalMx = intervalVec # we don't know how best to reshape vector of parameter intervals, so don't

            _tf.add_table_row(formats, tables, 
                        (gl, gateset.gates[gl],decomp.get('axis of rotation','X'), 
                         (decomp.get('pi rotations','X'), decompEB.get('pi rotations','X')) ),
                        (None, _tf.Brk, _tf.Nml, _tf.EBPi) )

    _tf.finish_table(formats, tables, longtable)
    return tables


def get_gateset_choi_table(gateset, formats, tableclass, longtable,
                           confidenceRegionInfo=None, mxBasis="gm"):
    """ 
    Create a table for the Choi matrices of a gateset's gates.
    
    Parameters
    ----------
    gateset : GateSet
        The GateSet

    formats : list
        List of formats to include in returned tables dictionary. Allowed
        formats are 'latex', 'html', 'py', and 'ppt'.

    tableclass : string
        CSS class to apply to the HTML table.

    longtable : bool
        Whether table should be a latex longtable or not.

    confidenceRegionInfo : ConfidenceRegion, optional
        If not None, specifies a confidence-region
        used to display error intervals.

    mxBasis : {'std', 'gm','pp'}
        Which basis the gateset is represented in.  Allowed
        options are Matrix-unit (std), Gell-Mann (gm) and
        Pauli-product (pp).


    Returns
    -------
    dict
        Dictionary with keys equal to the requested formats (e.g. 'latex'),
        and values equal to the table data in the corresponding format.
    """
    gateLabels = gateset.gates.keys()  # gate labels

    qtys_to_compute = [ ('%s choi matrix' % gl) for gl in gateLabels ]
    qtys_to_compute += [ ('%s choi eigenvalues' % gl) for gl in gateLabels ]
    qtys = _cr.compute_gateset_qtys(qtys_to_compute, gateset, confidenceRegionInfo, mxBasis)
    basisNm = _tools.basis_longname(mxBasis, gateset.get_dimension())
    
    if confidenceRegionInfo is None:
        colHeadings = ('Gate','Choi matrix (%s basis)' % basisNm,'Eigenvalues')
        formatters = (None,None,None)        
    else:
        colHeadings = ('Gate','Choi matrix (%s basis)' % basisNm,'Eigenvalues') # 'Confidence Intervals',
        formatters = (None,None,None)

    tables = {}
    _tf.create_table(formats, tables, colHeadings, formatters, tableclass, longtable)

    for gl in gateLabels:
        choiMx,choiEB = qtys['%s choi matrix' % gl].get_value_and_err_bar()
        evals, evalsEB = qtys['%s choi eigenvalues' % gl].get_value_and_err_bar()
    
        if confidenceRegionInfo is None:
            _tf.add_table_row(formats, tables, (gl, choiMx, evals), (None, _tf.Brk, _tf.Nml))
        else:
            _tf.add_table_row(formats, tables, (gl, choiMx, (evals,evalsEB)), (None, _tf.Brk, _tf.EBvec))

    _tf.finish_table(formats, tables, longtable)
    return tables


def get_gateset_vs_target_table(gateset, targetGateset, formats, tableclass, longtable, 
                                confidenceRegionInfo=None, mxBasis="gm"):
    """ 
    Create a table comparing a gateset to a target gateset.
    
    Parameters
    ----------
    gateset, targetGateset : GateSet
        The gate sets to compare

    formats : list
        List of formats to include in returned tables dictionary. Allowed
        formats are 'latex', 'html', 'py', and 'ppt'.

    tableclass : string
        CSS class to apply to the HTML table.

    longtable : bool
        Whether table should be a latex longtable or not.

    confidenceRegionInfo : ConfidenceRegion, optional
        If not None, specifies a confidence-region
        used to display error intervals.

    mxBasis : {'std', 'gm','pp'}
        Which basis the gateset is represented in.  Allowed
        options are Matrix-unit (std), Gell-Mann (gm) and
        Pauli-product (pp).


    Returns
    -------
    dict
        Dictionary with keys equal to the requested formats (e.g. 'latex'),
        and values equal to the table data in the corresponding format.
    """
    gateLabels = gateset.gates.keys()  # gate labels

    colHeadings = ('Gate', "Process|Infidelity", "1/2 Trace|Distance", "1/2 Diamond-Norm", "Frobenius|Distance")
    formatters = (None,_tf.TxtCnv,_tf.TxtCnv,_tf.TxtCnv,_tf.TxtCnv)

    qtyNames = ('infidelity','Jamiolkowski trace dist','diamond norm','Frobenius diff')
    qtys_to_compute = [ '%s %s' % (gl,qty) for qty in qtyNames for gl in gateLabels ]
    qtys = _cr.compute_gateset_gateset_qtys(qtys_to_compute, gateset, targetGateset,
                                            confidenceRegionInfo, mxBasis)

    tables = {}
    _tf.create_table(formats, tables, colHeadings, formatters, tableclass, longtable)

    formatters = [None] + [ _tf.EB ]*len(qtyNames)
    
    for gl in gateLabels:
        if confidenceRegionInfo is None:
            rowData = [gl] + [ (qtys['%s %s' % (gl,qty)].get_value(),None) for qty in qtyNames ]
        else:
            rowData = [gl] + [ qtys['%s %s' % (gl,qty)].get_value_and_err_bar() for qty in qtyNames ]
        _tf.add_table_row(formats, tables, rowData, formatters)

    _tf.finish_table(formats, tables, longtable)
    return tables


def get_gateset_vs_target_err_gen_table(gateset, targetGateset, formats, tableclass, longtable, confidenceRegionInfo=None):
    """ 
    Create a table listing the error generators obtained by 
    comparing a gateset to a target gateset.
    
    Parameters
    ----------
    gateset, targetGateset : GateSet
        The gate sets to compare

    formats : list
        List of formats to include in returned tables dictionary. Allowed
        formats are 'latex', 'html', 'py', and 'ppt'.

    tableclass : string
        CSS class to apply to the HTML table.

    longtable : bool
        Whether table should be a latex longtable or not.

    confidenceRegionInfo : ConfidenceRegion, optional
        If not None, specifies a confidence-region
        used to display error intervals.

    Returns
    -------
    dict
        Dictionary with keys equal to the requested formats (e.g. 'latex'),
        and values equal to the table data in the corresponding format.
    """
    gateLabels = gateset.gates.keys()  # gate labels
    colHeadings = ('Gate','Error Generator')

    tables = {}
    _tf.create_table(formats, tables, colHeadings, (None,None), tableclass, longtable)
    for gl in gateLabels:
        _tf.add_table_row(formats, tables, (gl, _spl.logm(_np.dot(_np.linalg.inv(
                            targetGateset.gates[gl]),gateset.gates[gl]))),
                    (None, _tf.Brk))
    _tf.finish_table(formats, tables, longtable)
    return tables



def get_gateset_vs_target_angles_table(gateset, targetGateset, formats, tableclass, longtable, 
                                       confidenceRegionInfo=None, mxBasis="gm"):
    """ 
    Create a table comparing a gateset to a target gateset.
    
    Parameters
    ----------
    gateset, targetGateset : GateSet
        The gate sets to compare

    formats : list
        List of formats to include in returned tables dictionary. Allowed
        formats are 'latex', 'html', 'py', and 'ppt'.

    tableclass : string
        CSS class to apply to the HTML table.

    longtable : bool
        Whether table should be a latex longtable or not.

    confidenceRegionInfo : ConfidenceRegion, optional
        If not None, specifies a confidence-region
        used to display error intervals.

    mxBasis : {'std', 'gm','pp'}
        Which basis the gateset is represented in.  Allowed
        options are Matrix-unit (std), Gell-Mann (gm) and
        Pauli-product (pp).


    Returns
    -------
    dict
        Dictionary with keys equal to the requested formats (e.g. 'latex'),
        and values equal to the table data in the corresponding format.
    """
    gateLabels = gateset.gates.keys()  # gate labels

    colHeadings = ('Gate', "Angle between|rotation axes")
    formatters = (None,_tf.TxtCnv)

    qtyNames = ('angle btwn rotn axes',)
    qtys_to_compute = [ '%s %s' % (gl,qty) for qty in qtyNames for gl in gateLabels ]
    qtys = _cr.compute_gateset_gateset_qtys(qtys_to_compute, gateset, targetGateset,
                                            confidenceRegionInfo, mxBasis)

    tables = {}
    _tf.create_table(formats, tables, colHeadings, formatters, tableclass, longtable)

    formatters = [None] + [ _tf.EBPi ]*len(qtyNames)
    
    for gl in gateLabels:
        if confidenceRegionInfo is None:
            rowData = [gl] + [ (qtys['%s %s' % (gl,qty)].get_value(),None) for qty in qtyNames ]
        else:
            rowData = [gl] + [ qtys['%s %s' % (gl,qty)].get_value_and_err_bar() for qty in qtyNames ]
        _tf.add_table_row(formats, tables, rowData, formatters)

    _tf.finish_table(formats, tables, longtable)
    return tables


def get_gateset_closest_unitary_table(gateset, formats, tableclass, longtable, confidenceRegionInfo=None):
    """ 
    Create a table for gateset that contains closest-unitary gates.
    
    Parameters
    ----------
    gateset : GateSet
        The GateSet

    formats : list
        List of formats to include in returned tables dictionary. Allowed
        formats are 'latex', 'html', 'py', and 'ppt'.

    tableclass : string
        CSS class to apply to the HTML table.

    longtable : bool
        Whether table should be a latex longtable or not.

    confidenceRegionInfo : ConfidenceRegion, optional
        If not None, specifies a confidence-region
        used to display error intervals.

    Returns
    -------
    dict
        Dictionary with keys equal to the requested formats (e.g. 'latex'),
        and values equal to the table data in the corresponding format.
    """

    gateLabels = gateset.gates.keys()  # gate labels
    colHeadings = ('Gate','Process|Infidelity','1/2 Trace|Distance','Rotation|Axis','Rotation|Angle','Sanity Check')
    formatters = (None,_tf.TxtCnv,_tf.TxtCnv,_tf.TxtCnv,_tf.TxtCnv,_tf.TxtCnv)

    if gateset.get_dimension() != 4:
        tables = {}
        _tf.create_table(formats, tables, colHeadings, formatters, tableclass, longtable)
        _tf.finish_table(formats, tables, longtable)
        return tables

    qtyNames = ('max fidelity with unitary', 'max trace dist with unitary',
                'closest unitary decomposition', 'upper bound on fidelity with unitary')
    qtys_to_compute = [ '%s %s' % (gl,qty) for qty in qtyNames for gl in gateLabels ]
    qtys = _cr.compute_gateset_qtys(qtys_to_compute, gateset, confidenceRegionInfo)
    decompNames = ('axis of rotation','pi rotations')
    #Other possible qtyName: 'closest unitary choi matrix'

    tables = {}
    _tf.create_table(formats, tables, colHeadings, formatters, tableclass, longtable)

    formatters = [None, _tf.EB, _tf.EB, _tf.EBvec, _tf.EBPi, _tf.Nml ] # Note len(decompNames)==2, 2nd el is rotn angle

    for gl in gateLabels:
        fUB,fUB_EB = qtys['%s upper bound on fidelity with unitary' % gl].get_value_and_err_bar()
        fLB,fLB_EB = qtys['%s max fidelity with unitary' % gl].get_value_and_err_bar()
        td, td_EB = qtys['%s max trace dist with unitary' % gl].get_value_and_err_bar()
        sanity = (1.0-fLB)/(1.0-fUB) - 1.0 #Robin's sanity check metric (0=good, >1=bad)
        decomp, decompEB = qtys['%s closest unitary decomposition' % gl].get_value_and_err_bar()
        
        if confidenceRegionInfo is None:        
            rowData = [gl, (1.0-fLB,None), (td,None)] + [(decomp.get(x,'X'),None) for x in decompNames]
        else:
            rowData = [gl, (1.0-fLB,fLB_EB), (td,td_EB)] + [(decomp.get(x,'X'),decompEB.get(x,None)) for x in decompNames]
        rowData.append(sanity)

        _tf.add_table_row(formats, tables, rowData, formatters)

    _tf.finish_table(formats, tables, longtable)
    return tables


def get_gateset_decomp_table(gateset, formats, tableclass, longtable, confidenceRegionInfo=None):
    """ 
    Create tables for decomposing a gateset's gates.
    
    Parameters
    ----------
    gateset : GateSet
        The GateSet

    formats : list
        List of formats to include in returned tables dictionary. Allowed
        formats are 'latex', 'html', 'py', and 'ppt'.

    tableclass : string
        CSS class to apply to the HTML table.

    longtable : bool
        Whether table should be a latex longtable or not.

    confidenceRegionInfo : ConfidenceRegion, optional
        If not None, specifies a confidence-region
        used to display error intervals.

    Returns
    -------
    dict
        Dictionary with keys equal to the requested formats (e.g. 'latex'),
        and values equal to the table data in the corresponding format.
    """
    gateLabels = gateset.gates.keys()  # gate labels
    colHeadings = ('Gate','Eigenvalues','Fixed pt','Rotn. axis','Diag. decay','Off-diag. decay')
    formatters = [None]*6

    qtyNames = ('eigenvalues','decomposition')
    qtys_to_compute = [ '%s %s' % (gl,qty) for qty in qtyNames for gl in gateLabels ]
    qtys = _cr.compute_gateset_qtys(qtys_to_compute, gateset, confidenceRegionInfo)
    decompNames = ('fixed point',
                   'axis of rotation',
                   'decay of diagonal rotation terms',
                   'decay of off diagonal rotation terms')

    tables = {}
    _tf.create_table(formats, tables, colHeadings, formatters, tableclass, longtable)

    formatters = (None, _tf.EBvec, _tf.Nml, _tf.Nml, _tf.EB, _tf.EB)

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

        _tf.add_table_row(formats, tables, rowData, formatters)

    _tf.finish_table(formats, tables, longtable)
    return tables


def get_gateset_rotn_axis_table(gateset, formats, tableclass, longtable, 
                                confidenceRegionInfo=None, showAxisAngleErrBars=True):
    """ 
    Create a table of the angle between a gate rotation axes for 
     gates belonging to a gateset
    
    Parameters
    ----------
    gateset : GateSet
        The GateSet

    formats : list
        List of formats to include in returned tables dictionary. Allowed
        formats are 'latex', 'html', 'py', and 'ppt'.

    tableclass : string
        CSS class to apply to the HTML table.

    longtable : bool
        Whether table should be a latex longtable or not.

    confidenceRegionInfo : ConfidenceRegion, optional
        If not None, specifies a confidence-region
        used to display error intervals.

    showAxisAngleErrBars : bool, optional
        Whether or not table should include error bars on the angles
        between rotation axes (doing so makes the table take up more
        space).

    Returns
    -------
    dict
        Dictionary with keys equal to the requested formats (e.g. 'latex'),
        and values equal to the table data in the corresponding format.
    """
    gateLabels = gateset.gates.keys()

    qtys_to_compute = [ '%s decomposition' % gl for gl in gateLabels ] + ['Gateset Axis Angles']
    qtys = _cr.compute_gateset_qtys(qtys_to_compute, gateset, confidenceRegionInfo)

    colHeadings = ("Gate","Angle") + tuple( [ "RAAW(%s)" % gl for gl in gateLabels] )
    nCols = len(colHeadings)
    formatters = [None] * nCols

    table = "longtable" if longtable else "tabular"
    latex_head =  "\\begin{%s}[l]{%s}\n\hline\n" % (table, "|c" * nCols + "|")
    latex_head += "\\multirow{2}{*}{Gate} & \\multirow{2}{*}{Angle} & " + \
                  "\\multicolumn{%d}{c|}{Angle between Rotation Axes} \\\\ \cline{3-%d}\n" % (len(gateLabels),nCols)
    latex_head += " & & %s \\\\ \hline\n" % (" & ".join(gateLabels))

    tables = {}
    _tf.create_table(formats, tables, colHeadings, formatters, tableclass, longtable, customHeader={'latex': latex_head} )

    formatters = [None, _tf.EBPi] + [ _tf.EBPi ] * len(gateLabels)

    rotnAxisAngles, rotnAxisAnglesEB = qtys['Gateset Axis Angles'].get_value_and_err_bar()
    rotnAngles = [ qtys['%s decomposition' % gl].get_value().get('pi rotations','X') \
                       for gl in gateLabels ]

    for i,gl in enumerate(gateLabels):
        decomp, decompEB = qtys['%s decomposition' % gl].get_value_and_err_bar()
        rotnAngle = decomp.get('pi rotations','X')

        angles_btwn_rotn_axes = []
        for j,gl_other in enumerate(gateLabels):
            decomp_other, decompEB_other = qtys['%s decomposition' % gl_other].get_value_and_err_bar()
            rotnAngle_other = decomp_other.get('pi rotations','X')

            if gl_other == gl:
                angles_btwn_rotn_axes.append( ("",None) )
            elif rotnAngle == 'X' or abs(rotnAngle) < 1e-4 or \
                 rotnAngle_other == 'X' or abs(rotnAngle_other) < 1e-4:
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
        _tf.add_table_row(formats, tables, rowData, formatters)

    _tf.finish_table(formats, tables, longtable)
    return tables


def get_dataset_overview_table(dataset, target, formats, tableclass, longtable,
                               maxlen=10, fixedLists=None):
    """ 
    Create a table overviewing a data set.
    
    Parameters
    ----------
    dataset : DataSet
        The DataSet

    target : GateSet
        A target gateset which is used for it's mapping of SPAM labels to
        SPAM specifiers.

    formats : list
        List of formats to include in returned tables dictionary. Allowed
        formats are 'latex', 'html', 'py', and 'ppt'.

    tableclass : string
        CSS class to apply to the HTML table.

    longtable : bool
        Whether table should be a latex longtable or not.

    maxlen : integer, optional
        The maximum length string used when searching for the 
        maximal (best) Gram matrix.  It's useful to make this
        at least twice the maximum length fiducial sequence.

    fixedLists : (prepStrs, effectStrs), optional
      2-tuple of gate string lists, specifying the preparation and
      measurement fiducials to use when constructing the Gram matrix,
      and thereby bypassing the search for such lists.


    Returns
    -------
    dict
        Dictionary with keys equal to the requested formats (e.g. 'latex'),
        and values equal to the table data in the corresponding format.
    """
    colHeadings = ('Quantity','Value')
    formatters = (None,None)
    rank,evals = _alg.max_gram_rank_and_evals( dataset, maxlen, target, fixedLists=fixedLists )

    tables = {}
    _tf.create_table(formats, tables, colHeadings, formatters, tableclass, longtable)

    _tf.add_table_row(formats, tables, ("Number of strings", str(len(dataset))), (None,None))
    _tf.add_table_row(formats, tables, ("Gate labels", ", ".join(dataset.get_gate_labels()) ), (None,None))
    _tf.add_table_row(formats, tables, ("SPAM labels",  ", ".join(dataset.get_spam_labels()) ), (None,None))
    _tf.add_table_row(formats, tables, ("Gram singular vals", _np.sort(abs(evals)).reshape(-1,1) ), (None,_tf.Sml))

    _tf.finish_table(formats, tables, longtable)
    return tables


def get_chi2_progress_table(Ls, gatesetsByL, gateStringsByL, dataset, formats, tableclass, longtable):
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

    formats : list
        List of formats to include in returned tables dictionary. Allowed
        formats are 'latex', 'html', 'py', and 'ppt'.

    tableclass : string
        CSS class to apply to the HTML table.

    longtable : bool
        Whether table should be a latex longtable or not.

    Returns
    -------
    dict
        Dictionary with keys equal to the requested formats (e.g. 'latex'),
        and values equal to the table data in the corresponding format.
    """
    colHeadings = { 'latex': ('L','$\\chi^2$','$k$','$\\chi^2-k$','$\sqrt{2k}$','$P$','$N_s$','$N_p$', 'Rating'),
                    'html': ('L','&chi;<sup>2</sup>','k','&chi;<sup>2</sup>-k',
                             '&radic;<span style="text-decoration:overline;">2k</span>',
                             'P','N<sub>s</sub>','N<sub>p</sub>', 'Rating'),
                    'py': ('L','chi^2','k','chi^2-k','sqrt{2k}','P','N_s','N_p', 'Rating'),
                    'ppt': ('L','chi^2','k','chi^2-k','sqrt{2k}','P','N_s','N_p', 'Rating')
                  }
    tables = {}
    _tf.create_table_preformatted(formats, tables, colHeadings, tableclass, longtable)

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
        _tf.add_table_row(formats, tables, 
                    (str(L),chi2,k,chi2-k,_np.sqrt(2*k),pv,Ns,Np,"<STAR>"*rating),
                    (None,_tf.Nml,_tf.Nml,_tf.Nml,_tf.Nml,_tf.Nml2,_tf.Nml,_tf.Nml,_tf.TxtCnv))

    _tf.finish_table(formats, tables, longtable)
    return tables


def get_logl_progress_table(Ls, gatesetsByL, gateStringsByL, dataset, formats, tableclass, longtable):
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

    formats : list
        List of formats to include in returned tables dictionary. Allowed
        formats are 'latex', 'html', 'py', and 'ppt'.

    tableclass : string
        CSS class to apply to the HTML table.

    longtable : bool
        Whether table should be a latex longtable or not.

    Returns
    -------
    dict
        Dictionary with keys equal to the requested formats (e.g. 'latex'),
        and values equal to the table data in the corresponding format.
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
    _tf.create_table_preformatted(formats, tables, colHeadings, tableclass, longtable)

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

        _tf.add_table_row(formats, tables, 
                    (str(L),twoDeltaLogL,k,twoDeltaLogL-k,_np.sqrt(2*k),pv,Ns,Np,"<STAR>"*rating),
                    (None,_tf.Nml,_tf.Nml,_tf.Nml,_tf.Nml,_tf.Nml2,_tf.Nml,_tf.Nml,_tf.TxtCnv))

    _tf.finish_table(formats, tables, longtable)
    return tables
    

def get_gatestring_table(gsList, title, formats, tableclass, longtable):
    """ 
    Creates a 2-column table enumerating a list of gate strings.
    
    Parameters
    ----------
    gsList : list of GateStrings
        List of gate strings to put in table.
        
    title : string
        The title for the table column containing the strings.

    formats : list
        List of formats to include in returned tables dictionary. Allowed
        formats are 'latex', 'html', 'py', and 'ppt'.

    tableclass : string
        CSS class to apply to the HTML table.

    longtable : bool
        Whether table should be a latex longtable or not.

    Returns
    -------
    dict
        Dictionary with keys equal to the requested formats (e.g. 'latex'),
        and values equal to the table data in the corresponding format.
    """
    colHeadings = ('#',title)
    formatters = (_tf.TxtCnv,_tf.Nml)

    tables = {}
    _tf.create_table(formats, tables, colHeadings, formatters, tableclass, longtable)

    for i,gstr in enumerate(gsList,start=1):
        _tf.add_table_row(formats, tables, (i, gstr), (_tf.Nml,_tf.GStr) )

    _tf.finish_table(formats, tables, longtable)
    return tables


def get_gatestring_multi_table(gsLists, titles, formats, tableclass, longtable, commonTitle=None):
    """ 
    Creates an N-column table enumerating a N-1 lists of gate strings.
    
    Parameters
    ----------
    gsLists : list of GateString lists
        List of gate strings to put in table.
        
    titles : list of strings
        The titles for the table columns containing the strings.

    formats : list
        List of formats to include in returned tables dictionary. Allowed
        formats are 'latex', 'html', 'py', and 'ppt'.

    tableclass : string
        CSS class to apply to the HTML table.

    longtable : bool
        Whether table should be a latex longtable or not.

    commonTitle : string, optional
        A single title string to place in a cell spanning across
        all the gate string columns.

    Returns
    -------
    dict
        Dictionary with keys equal to the requested formats (e.g. 'latex'),
        and values equal to the table data in the corresponding format.
    """
    colHeadings = ('#',) + tuple(titles)
    formatters = (_tf.TxtCnv,) + (_tf.Nml,)*len(titles)

    tables = {}

    if commonTitle is None:
        _tf.create_table(formats, tables, colHeadings, formatters, tableclass, longtable)
    else:
        table = "longtable" if longtable else "tabular"
        colHeadings = ('\\#',) + tuple(titles)
        latex_head  = "\\begin{%s}[l]{%s}\n\hline\n" % (table, "|c" * len(colHeadings) + "|")
        latex_head += " & \multicolumn{%d}{c|}{%s} \\\\ \hline\n" % (len(titles),commonTitle)
        latex_head += "%s \\\\ \hline\n" % (" & ".join(colHeadings))

        html_head = "<table class=%s><thead>" % tableclass
        html_head += '<tr><th></th><th colspan="%d">%s</th></tr>\n' % (len(titles),commonTitle)
        html_head += "<tr><th> %s </th></tr>" % (" </th><th> ".join(colHeadings))
        html_head += "</thead><tbody>"

        _tf.create_table(formats, tables, colHeadings, formatters, tableclass, longtable, 
                    customHeader={'latex': latex_head, 'html': html_head})

    formatters = (_tf.Nml,) + (_tf.GStr,)*len(gsLists)

    for i in range( max([len(gsl) for gsl in gsLists]) ):
        rowData = [i+1]
        for gsList in gsLists:
            if i < len(gsList):
                rowData.append( gsList[i] )
            else:
                rowData.append( None ) #empty string
        _tf.add_table_row(formats, tables, rowData, formatters)

    _tf.finish_table(formats, tables, longtable)
    return tables




def get_logl_confidence_region(gateset, dataset, confidenceLevel,
                               gatestring_list=None, probClipInterval=(-1e6,1e6),
                               minProbClip=1e-4, radius=1e-4, hessianProjection="std"):

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

    Returns
    -------
    ConfidenceRegion
    """
    if gatestring_list is None:
        gatestring_list = dataset.keys()
        
    #Compute appropriate Hessian
    hessian = _tools.logl_hessian(gateset, dataset, gatestring_list,
                                  minProbClip, probClipInterval, radius) 

    cri = _objs.ConfidenceRegion(gateset, hessian, confidenceLevel,
                                 hessianProjection)

    #Check that number of gauge parameters reported by gateset is consistent with confidence region
    # since the parameter number computed this way is used in chi2 or logl progress tables
    Np_check =  gateset.num_nongauge_params()
    if(Np_check != cri.nNonGaugeParams):
        _warnings.warn("Number of non-gauge parameters in gateset and confidence region do " 
                       + " not match.  This indicates an internal logic error.")            
    
    return cri



def get_chi2_confidence_region(gateset, dataset, confidenceLevel,
                                  gatestring_list=None, probClipInterval=(-1e6,1e6),
                                  minProbClipForWeighting=1e-4, hessianProjection="std"):

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


    Returns
    -------
    ConfidenceRegion
    """
    if gatestring_list is None:
        gatestring_list = dataset.keys()
        
    #Compute appropriate Hessian
    dummy_chi2, hessian = _tools.chi2(dataset, gateset, gatestring_list,
                                      False, True, minProbClipForWeighting,
                                      probClipInterval)

    cri = _objs.ConfidenceRegion(gateset, hessian, confidenceLevel,
                                 hessianProjection)

    #Check that number of gauge parameters reported by gateset is consistent with confidence region
    # since the parameter number computed this way is used in chi2 or logl progress tables
    Np_check =  gateset.num_nongauge_params()
    if(Np_check != cri.nNonGaugeParams):
        _warnings.warn("Number of non-gauge parameters in gateset and confidence region do " 
                       + " not match.  This indicates an internal logic error.")            
    
    return cri

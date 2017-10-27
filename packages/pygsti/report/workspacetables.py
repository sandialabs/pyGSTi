from __future__ import division, print_function, absolute_import, unicode_literals
#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************
""" Classes corresponding to tables within a Workspace context."""

import warnings           as _warnings
import numpy              as _np
import scipy.stats        as _stats
import scipy.linalg       as _spl

from .. import algorithms as _alg
from .. import construction as _cnst
from .. import tools      as _tools
from .. import objects    as _objs
from . import reportables as _reportables
from .reportables import evaluate as _ev

from .table import ReportTable as _ReportTable

from .workspace import WorkspaceTable
from . import workspaceplots as _wp

class BlankTable(WorkspaceTable):
    def __init__(self, ws):
        """A completely blank placeholder table."""
        super(BlankTable,self).__init__(ws, self._create)

    def _create(self):
        table = _ReportTable(['Blank'], [None])
        table.finish()
        return table
    
class SpamTable(WorkspaceTable):
    def __init__(self, ws, gatesets, titles=None,
                 display_as="boxes", confidenceRegionInfo=None,
                 includeHSVec=True):
        """
        A table of one or more gateset's SPAM elements.
    
        Parameters
        ----------
        gatesets : GateSet or list of GateSets
            The GateSet(s) whose SPAM elements should be displayed. If
            multiple GateSets are given, they should have the same SPAM
            elements..
    
        titles : list of strs, optional
            Titles correponding to elements of `gatesets`, e.g. `"Target"`.

        display_as : {"numbers", "boxes"}, optional
            How to display the SPAM matrices, as either numerical
            grids (fine for small matrices) or as a plot of colored
            boxes (space-conserving and better for large matrices).
    
        confidenceRegionInfo : ConfidenceRegion, optional
            If not None, specifies a confidence-region
            used to display error intervals.
    
        includeHSVec : boolean, optional
            Whether or not to include Hilbert-Schmidt
            vector representation columns in the table.    
        """
        super(SpamTable,self).__init__(ws, self._create, gatesets,
                                       titles, display_as, confidenceRegionInfo,
                                       includeHSVec)

    def _create(self, gatesets, titles, display_as, confidenceRegionInfo,
                includeHSVec):
        
        if isinstance(gatesets, _objs.GateSet):
            gatesets = [gatesets]

        rhoLabels = list(gatesets[0].get_prep_labels()) #use labels of 1st gateset
        ELabels = list(gatesets[0].get_effect_labels()) #use labels of 1st gateset
            
        if titles is None:
            titles = ['']*len(gatesets)

        colHeadings = ['Operator']
        for gateset,title in zip(gatesets,titles):
            colHeadings.append( '%sMatrix' % (title+' ' if title else '') )
        for gateset,title in zip(gatesets,titles):
            colHeadings.append( '%sEigenvals' % (title+' ' if title else '') )

        formatters = [None]*len(colHeadings)

        if includeHSVec:
            gateset = gatesets[-1] #only show HSVec for last gateset
            basisNm    = _objs.basis_longname(gateset.basis.name)
            colHeadings.append( 'Hilbert-Schmidt vector (%s basis)' % basisNm )
            formatters.append( None )
            
            if confidenceRegionInfo is not None:
                colHeadings.append('%g%% C.I. half-width' % confidenceRegionInfo.level)
                formatters.append( 'Conversion' )

                
        table = _ReportTable(colHeadings, formatters, confidenceRegionInfo=confidenceRegionInfo)
    
        for lbl in rhoLabels:
            rowData = [lbl]; rowFormatters = ['Rho']

            for gateset in gatesets:
                rhoMx = _ev(_reportables.Vec_as_stdmx(gateset, lbl, "prep"))
                            # confidenceRegionInfo) #don't put CIs on matrices for now
                if display_as == "numbers":
                    rowData.append( rhoMx )
                    rowFormatters.append('Brackets')
                elif display_as == "boxes":
                    rhoMx_real = rhoMx.hermitian_to_real()
                    v = rhoMx_real.get_value()
                    fig = _wp.GateMatrixPlot(self.ws, v, colorbar=False, 
                                             boxLabels=True, prec='compacthp',
                                             mxBasis=None) #no basis labels 
                    rowData.append( fig )
                    rowFormatters.append('Figure')
                else:
                    raise ValueError("Invalid 'display_as' argument: %s" % display_as)


            for gateset in gatesets:
                cri = confidenceRegionInfo if confidenceRegionInfo and \
                      (confidenceRegionInfo.gateset.frobeniusdist(gateset) < 1e-6) else None
                evals = _ev(_reportables.Vec_as_stdmx_eigenvalues(gateset, lbl, "prep"),
                            cri)
                rowData.append( evals )
                rowFormatters.append('Brackets')
                
            if includeHSVec:
                rowData.append( gatesets[-1].preps[lbl] )
                rowFormatters.append('Normal')
    
                if confidenceRegionInfo is not None:
                    intervalVec = confidenceRegionInfo.get_profile_likelihood_confidence_intervals(lbl)[:,None]
                    if intervalVec.shape[0] == gatesets[-1].get_dimension()-1:
                        #TP constrained, so pad with zero top row
                        intervalVec = _np.concatenate( (_np.zeros((1,1),'d'),intervalVec), axis=0 )
                    rowData.append( intervalVec ); rowFormatters.append('Normal')
    
            #Note: no dependence on confidence region (yet) when HS vector is not shown...
            table.addrow(rowData, rowFormatters)


        for lbl in ELabels:
            rowData = [lbl]; rowFormatters = ['Effect']

            for gateset in gatesets:
                EMx = _ev(_reportables.Vec_as_stdmx(gateset, lbl, "effect"))
                          #confidenceRegionInfo) #don't put CIs on matrices for now
                if display_as == "numbers":
                    rowData.append( EMx )
                    rowFormatters.append('Brackets')
                elif display_as == "boxes":
                    EMx_real = EMx.hermitian_to_real()
                    v = EMx_real.get_value()
                    fig = _wp.GateMatrixPlot(self.ws, v, colorbar=False,
                                             boxLabels=True, prec='compacthp',
                                             mxBasis=None) #no basis labels 
                    rowData.append( fig )
                    rowFormatters.append('Figure')
                else:
                    raise ValueError("Invalid 'display_as' argument: %s" % display_as)

            for gateset in gatesets:
                cri = confidenceRegionInfo if confidenceRegionInfo and \
                      (confidenceRegionInfo.gateset.frobeniusdist(gateset) < 1e-6) else None
                evals = _ev(_reportables.Vec_as_stdmx_eigenvalues(gateset, lbl, "effect"),
                            cri)
                rowData.append( evals )
                rowFormatters.append('Brackets')
    
            if includeHSVec:
                rowData.append( gatesets[-1].effects[lbl] )
                rowFormatters.append('Normal')
    
                if confidenceRegionInfo is not None:
                    intervalVec = confidenceRegionInfo.get_profile_likelihood_confidence_intervals(lbl)[:,None]
                    rowData.append( intervalVec ); rowFormatters.append('Normal')
    
            #Note: no dependence on confidence region (yet) when HS vector is not shown...
            table.addrow(rowData, rowFormatters)
    
        table.finish()
        return table



#def get_gateset_spam_parameters_table(gateset, confidenceRegionInfo=None):
class SpamParametersTable(WorkspaceTable):
    def __init__(self, ws, gatesets, titles=None, confidenceRegionInfo=None):
        """
        Create a table for gateset's "SPAM parameters", that is, the
        dot products of prep-vectors and effect-vectors.
    
        Parameters
        ----------
        gatesets : GateSet or list of GateSets
            The GateSet(s) whose SPAM parameters should be displayed. If
            multiple GateSets are given, they should have the same gates.

        titles : list of strs, optional
            Titles correponding to elements of `gatesets`, e.g. `"Target"`.
    
        confidenceRegionInfo : ConfidenceRegion, optional
            If not None, specifies a confidence-region
            used to display error intervals.
    
        Returns
        -------
        ReportTable
        """
        super(SpamParametersTable,self).__init__(ws, self._create, gatesets, titles, confidenceRegionInfo)

    def _create(self, gatesets, titles, confidenceRegionInfo):
        
        if isinstance(gatesets, _objs.GateSet):
            gatesets = [gatesets]
        if titles is None:
            titles = ['']*len(gatesets)

        colHeadings = [''] + list(gatesets[0].get_effect_labels())
        formatters  = [None] + [ 'Effect' ]*len(gatesets[0].get_effect_labels())
    
        table       = _ReportTable(colHeadings, formatters, confidenceRegionInfo=confidenceRegionInfo)

        for gstitle, gateset in zip(titles,gatesets):
            cri = confidenceRegionInfo if (confidenceRegionInfo and
                                           confidenceRegionInfo.gateset.frobeniusdist(gateset) < 1e-6) else None
            spamDotProdsQty = _ev( _reportables.Spam_dotprods(gateset), cri)
            DPs, DPEBs      = spamDotProdsQty.get_value_and_err_bar()
        
            formatters      = [ 'Rho' ] + [ 'Normal' ]*len(gateset.get_effect_labels()) #for rows below
        
            for ii,prepLabel in enumerate(gateset.get_prep_labels()): # ii enumerates rhoLabels to index DPs
                prefix = gstitle + " " if len(gstitle) else ""
                rowData = [prefix + prepLabel]
                for jj,_ in enumerate(gateset.get_effect_labels()): # jj enumerates eLabels to index DPs
                    if cri is None:
                        rowData.append((DPs[ii,jj],None))
                    else:
                        rowData.append((DPs[ii,jj],DPEBs[ii,jj]))
                table.addrow(rowData, formatters)
    
        table.finish()
        return table
    
    
#def get_gateset_gates_table(gateset, confidenceRegionInfo=None):
class GatesTable(WorkspaceTable):
    def __init__(self, ws, gatesets, titles=None, display_as="boxes",
                 confidenceRegionInfo=None):
        """
        Create a table showing a gateset's raw gates.
    
        Parameters
        ----------
        gatesets : GateSet or list of GateSets
            The GateSet(s) whose gates should be displayed.  If multiple
            GateSets are given, they should have the same gate labels.

        titles : list of strings, optional
            A list of titles corresponding to the gate sets, used to 
            prefix the column(s) for that gate set. E.g. `"Target"`.
    
        display_as : {"numbers", "boxes"}, optional
            How to display the gate matrices, as either numerical
            grids (fine for small matrices) or as a plot of colored
            boxes (space-conserving and better for large matrices).

        confidenceRegionInfo : ConfidenceRegion, optional
            If not None, specifies a confidence-region
            used to display error intervals for the *final*
            element of `gatesets`.
    
        Returns
        -------
        ReportTable
        """
        super(GatesTable,self).__init__(ws, self._create, gatesets, titles,
                                        display_as, confidenceRegionInfo)

        
    def _create(self, gatesets, titles, display_as, confidenceRegionInfo):

        if isinstance(gatesets, _objs.GateSet):
            gatesets = [gatesets]

        gateLabels = list(gatesets[0].gates.keys()) #use labels of 1st gateset

        if titles is None:
            titles = ['']*len(gatesets)

        colHeadings = ['Gate']
        for gateset,title in zip(gatesets,titles):
            basisLongNm = _objs.basis_longname(gateset.basis.name)
            pre = (title+' ' if title else '')
            colHeadings.append('%sSuperoperator (%s basis)' % (pre,basisLongNm))
        formatters = [None]*len(colHeadings)

        if confidenceRegionInfo is not None:
            #Only use confidence region for the *final* gateset.
            colHeadings.append('%g%% C.I. half-width' % confidenceRegionInfo.level)
            formatters.append('Conversion')
    
        table = _ReportTable(colHeadings, formatters, confidenceRegionInfo=confidenceRegionInfo)

        for gl in gateLabels:
            #Note: currently, we don't use confidence region...
            row_data = [gl]
            row_formatters = [None]
    
            for gateset in gatesets:
                basis = gateset.basis

                if display_as == "numbers":
                    row_data.append(gateset.gates[gl])
                    row_formatters.append('Brackets')
                elif display_as == "boxes":
                    fig = _wp.GateMatrixPlot(self.ws, gateset.gates[gl],
                                             colorbar=False,
                                             mxBasis=basis)

                    row_data.append( fig )
                    row_formatters.append('Figure')
                else:
                    raise ValueError("Invalid 'display_as' argument: %s" % display_as)

            if confidenceRegionInfo is not None:
                intervalVec = confidenceRegionInfo.get_profile_likelihood_confidence_intervals(gl)[:,None]
                if isinstance(gatesets[-1].gates[gl], _objs.FullyParameterizedGate):
                    #then we know how to reshape into a matrix
                    gate_dim   = gatesets[-1].get_dimension()
                    basis = gatesets[-1].basis
                    intervalMx = intervalVec.reshape(gate_dim,gate_dim)
                elif isinstance(gatesets[-1].gates[gl], _objs.TPParameterizedGate):
                    #then we know how to reshape into a matrix
                    gate_dim   = gatesets[-1].get_dimension()
                    basis = gatesets[-1].basis
                    intervalMx = _np.concatenate( ( _np.zeros((1,gate_dim),'d'),
                                                    intervalVec.reshape(gate_dim-1,gate_dim)), axis=0 )
                else:
                    # we don't know how best to reshape interval matrix for gate, so
                    # use derivative
                    gate_dim   = gatesets[-1].get_dimension()
                    basis = gatesets[-1].basis
                    gate_deriv = gatesets[-1].gates[gl].deriv_wrt_params()
                    intervalMx = _np.abs(_np.dot(gate_deriv, intervalVec).reshape(gate_dim,gate_dim))
                    
                    # vector of parameter intervals
                    #OLD: intervalMx = intervalVec.reshape(len(intervalVec),1) #col of boxes
                    
                if display_as == "numbers":
                    row_data.append(intervalMx)
                    row_formatters.append('Brackets')
                    
                elif display_as == "boxes":
                    maxAbsVal = _np.max(_np.abs(intervalMx))
                    fig = _wp.GateMatrixPlot(self.ws, intervalMx,
                                             m=-maxAbsVal, M=maxAbsVal,
                                             colorbar=False,
                                             mxBasis=basis)
                    row_data.append( fig )
                    row_formatters.append('Figure')
                else:
                    assert(False)
                
            table.addrow(row_data, row_formatters)
    
        table.finish()
        return table
        
    
#    def get_gateset_choi_table(gateset, confidenceRegionInfo=None):
class ChoiTable(WorkspaceTable):
    def __init__(self, ws, gatesets, titles=None,
                 confidenceRegionInfo=None,
                 display=("matrix","eigenvalues","barplot")):
        """
        Create a table of the Choi matrices and/or their eigenvalues of
        a gateset's gates.
    
        Parameters
        ----------
        gatesets : GateSet or list of GateSets
            The GateSet(s) whose Choi info should be displayed.  If multiple
            GateSets are given, they should have the same gate labels.

        titles : list of strings, optional
            A list of titles corresponding to the gate sets, used to 
            prefix the column(s) for that gate set. E.g. `"Target"`.
    
        confidenceRegionInfo : ConfidenceRegion, optional
            If not None, specifies a confidence-region
            used to display eigenvalue error intervals for the
            *final* GateSet in `gatesets`.

        display : tuple or list of {"matrices","eigenvalues","barplot"}
            Which columns to display: the Choi matrices (as numerical grids),
            the Choi matrix eigenvalues (as a numerical list), and/or the
            eigenvalues on a bar plot.

    
        Returns
        -------
        ReportTable
        """
        super(ChoiTable,self).__init__(ws, self._create, gatesets, titles,
                                       confidenceRegionInfo, display)
        
    def _create(self, gatesets, titles, confidenceRegionInfo, display):
        if isinstance(gatesets, _objs.GateSet):
            gatesets = [gatesets]

        gateLabels = list(gatesets[0].gates.keys()) #use labels of 1st gateset

        if titles is None:
            titles = ['']*len(gatesets)
        
        qtysList = []
        for gateset in gatesets:
            gateLabels = list(gateset.gates.keys()) # gate labels
            #qtys_to_compute = []
            if 'matrix' in display:
                choiMxs = [_ev(_reportables.Choi_matrix(gateset,gl)) for gl in gateLabels]
            else:
                choiMxs = None
            if 'eigenvalues' in display or 'barplot' in display:
                evals   = [_ev(_reportables.Choi_evals(gateset,gl), confidenceRegionInfo) for gl in gateLabels]
            else:
                evals = None
            qtysList.append((choiMxs, evals))
        colHeadings = ['Gate']
        for disp in display:
            if disp == "matrix":
                for gateset,title in zip(gatesets,titles):
                    basisLongNm = _objs.basis_longname(gateset.basis.name)
                    pre = (title+' ' if title else '')
                    colHeadings.append('%sChoi matrix (%s basis)' % (pre,basisLongNm))
            elif disp == "eigenvalues":
                for gateset,title in zip(gatesets,titles):
                    pre = (title+' ' if title else '')
                    colHeadings.append('%sEigenvalues' % pre)
            elif disp == "barplot":
                for gateset,title in zip(gatesets,titles):
                    pre = (title+' ' if title else '')
                    colHeadings.append('%sEigenvalue Magnitudes' % pre)
            else:
                raise ValueError("Invalid element of `display`: %s" % disp)                    
        formatters = [None]*len(colHeadings)

        
        table = _ReportTable(colHeadings, formatters, confidenceRegionInfo=confidenceRegionInfo)

        for i, gl in enumerate(gateLabels):
            #Note: currently, we don't use confidence region...
            row_data = [gl]
            row_formatters = [None]

            for disp in display:
                if disp == "matrix":
                    for gateset, (choiMxs, _) in zip(gatesets, qtysList):
                        row_data.append(choiMxs[i])
                        row_formatters.append('Brackets')
        
                elif disp == "eigenvalues":
                    for gateset, (_, evals) in zip(gatesets, qtysList):
                        try:
                            evals[i] = evals[i].reshape(evals[i].size//4, 4)
                            #assumes len(evals) is multiple of 4!
                        except: # if it isn't try 3 (qutrits)
                            evals[i] = evals[i].reshape(evals[i].size//3, 3)
                            #assumes len(evals) is multiple of 3!
                        row_data.append(evals[i])                        
                        row_formatters.append('Normal')
                            
                elif disp == "barplot":
                    for gateset in gatesets:
                        for gateset, (_, evals) in zip(gatesets,qtysList):
                            evs, evsEB = evals[i].get_value_and_err_bar()
                            fig = _wp.ChoiEigenvalueBarPlot(self.ws, evs, evsEB)
                            row_data.append(fig)
                            row_formatters.append('Figure')
                            
            table.addrow(row_data, row_formatters)
        table.finish()
        return table


class GatesetVsTargetTable(WorkspaceTable):
    def __init__(self, ws, gateset, targetGateset, clifford_compilation, confidenceRegionInfo=None):
        """
        Create a table comparing a gateset (as a whole) to a target gateset
        using metrics that can be evaluatd for an entire gate set.
    
        Parameters
        ----------
        gateset, targetGateset : GateSet
            The gate sets to compare

        clifford_compilation : dict
            A dictionary of gate sequences, one for each Clifford operation
            in the Clifford group relevant to the gate set Hilbert space.  If 
            None, then rows requiring a clifford compilation are omitted.
    
        confidenceRegionInfo : ConfidenceRegion, optional
            If not None, specifies a confidence-region
            used to display error intervals.    
    
        Returns
        -------
        ReportTable
        """
        super(GatesetVsTargetTable,self).__init__(ws, self._create, gateset,
                                                  targetGateset, clifford_compilation,
                                                  confidenceRegionInfo)
    
    def _create(self, gateset, targetGateset, clifford_compilation, confidenceRegionInfo):
    
        colHeadings = ('Metric', "Value")
        formatters  = (None,None)
        
        tooltips = colHeadings
        table = _ReportTable(colHeadings, formatters, colHeadingLabels=tooltips, confidenceRegionInfo=confidenceRegionInfo)

        #Leave this off for now, as it's primary use is to compare with RB and the predicted RB number is better for this.
        #pAGsI = _ev(_reportables.Average_gateset_infidelity(gateset, targetGateset), confidenceRegionInfo)
        #table.addrow(("Avg. primitive gate set infidelity", pAGsI), (None, 'Normal') )

        pRBnum = _ev(_reportables.Predicted_rb_number(gateset, targetGateset), confidenceRegionInfo)
        table.addrow(("Predicted primitive RB number", pRBnum), (None, 'Normal') )

        if clifford_compilation:
            clifford_gateset = _cnst.build_alias_gateset(gateset,clifford_compilation)
            clifford_targetGateset = _cnst.build_alias_gateset(targetGateset,clifford_compilation)

            ##For clifford versions we don't have a confidence region - so no error bars
            #AGsI = _ev(_reportables.Average_gateset_infidelity(clifford_gateset, clifford_targetGateset))
            #table.addrow(("Avg. clifford gate set infidelity", AGsI), (None, 'Normal') )
    
            RBnum = _ev(_reportables.Predicted_rb_number(clifford_gateset, clifford_targetGateset))
            table.addrow(("Predicted RB number", RBnum), (None, 'Normal') )

        table.finish()
        return table

    
#    def get_gates_vs_target_table(gateset, targetGateset, confidenceRegionInfo=None):
class GatesVsTargetTable(WorkspaceTable):
    def __init__(self, ws, gateset, targetGateset, confidenceRegionInfo=None,
                 display=('inf','agi','trace','diamond','nuinf','nuagi'),
                 virtual_gates=None):
        """
        Create a table comparing a gateset's gates to a target gateset using
        metrics such as the  infidelity, diamond-norm distance, and trace distance.
    
        Parameters
        ----------
        gateset, targetGateset : GateSet
            The gate sets to compare

        confidenceRegionInfo : ConfidenceRegion, optional
            If not None, specifies a confidence-region
            used to display error intervals.    

        display : tuple, optional
            A tuple of one or more of the allowed options (see below) which
            specify which columns are displayed in the table.

            - "inf" :     entanglement infidelity
            - "agi" :     average gate infidelity
            - "trace" :   1/2 trace distance
            - "diamond" : 1/2 diamond norm distance
            - "nuinf" :   non-unitary entanglement infidelity
            - "nuagi" :   non-unitary entanglement infidelity
            - "evinf" :     eigenvalue entanglement infidelity
            - "evagi" :     eigenvalue average gate infidelity
            - "evnuinf" :   eigenvalue non-unitary entanglement infidelity
            - "evnuagi" :   eigenvalue non-unitary entanglement infidelity
            - "evdiamond" : eigenvalue 1/2 diamond norm distance
            - "evnudiamond" : eigenvalue non-unitary 1/2 diamond norm distance
            - "frob" :    frobenius distance

        virtual_gates : list, optional
            If not None, a list of `GateString` objects specifying additional "gates"
            (i.e. processes) to compute eigenvalues of.  Length-1 gate strings are
            automatically discarded so they are not displayed twice.    
    
        Returns
        -------
        ReportTable
        """
        super(GatesVsTargetTable,self).__init__(ws, self._create, gateset,
                                                targetGateset, confidenceRegionInfo,
                                                display, virtual_gates)
    
    def _create(self, gateset, targetGateset, confidenceRegionInfo,
                display, virtual_gates):
    
        gateLabels  = list(gateset.gates.keys())  # gate labels

        colHeadings = ['Gate'] if (virtual_gates is None) else ['Gate or Germ']
        tooltips    = ['Gate'] if (virtual_gates is None) else ['Gate or Germ']
        for disp in display:
            if disp == "inf":
                colHeadings.append("Entanglement|Infidelity")
                tooltips.append("1.0 - <psi| 1 x Lambda(psi) |psi>")
            elif disp == "agi":
                colHeadings.append("Avg. Gate|Infidelity")
                tooltips.append("d/(d+1) (entanglement infidelity)")
            elif disp == "trace":
                colHeadings.append("1/2 Trace|Distance")
                tooltips.append("0.5 | Chi(A) - Chi(B) |_tr")
            elif disp == "diamond":
                colHeadings.append( "1/2 Diamond-Dist")
                tooltips.append("0.5 sup | (1 x (A-B))(rho) |_tr")
            elif disp == "nuinf":
                colHeadings.append("Non-unitary|Ent. Infidelity")
                tooltips.append("(d^2-1)/d^2 [1 - sqrt( unitarity(A B^-1) )]")
            elif disp == "nuagi":
                colHeadings.append("Non-unitary|Avg. Gate Infidelity")
                tooltips.append("(d-1)/d [1 - sqrt( unitarity(A B^-1) )]")
            elif disp == "evinf":
                colHeadings.append("Eigenvalue|Ent. Infidelity")
                tooltips.append("min_P 1 - (lambda P lambda^dag)/d^2  [P = permutation, lambda = eigenvalues]")
            elif disp == "evagi":
                colHeadings.append("Eigenvalue|Avg. Gate Infidelity")
                tooltips.append("min_P (d^2 - lambda P lambda^dag)/d(d+1)  [P = permutation, lambda = eigenvalues]")
            elif disp == "evnuinf":
                colHeadings.append("Eigenvalue Non-U.|Ent. Infidelity")
                tooltips.append("(d^2-1)/d^2 [1 - sqrt( eigenvalue_unitarity(A B^-1) )]")
            elif disp == "evnuagi":
                colHeadings.append("Eigenvalue Non-U.|Avg. Gate Infidelity")
                tooltips.append("(d-1)/d [1 - sqrt( eigenvalue_unitarity(A B^-1) )]")
            elif disp == "evdiamond":
                colHeadings.append("Eigenvalue|1/2 Diamond-Dist")
                tooltips.append("(d^2-1)/d^2 max_i { |a_i - b_i| } where (a_i,b_i) are corresponding eigenvalues of A and B.")
            elif disp == "evnudiamond":
                colHeadings.append("Eigenvalue Non-U.|1/2 Diamond-Dist")
                tooltips.append("(d^2-1)/d^2 max_i { | |a_i| - |b_i| | } where (a_i,b_i) are corresponding eigenvalues of A and B.")
            elif disp == "frob":
                colHeadings.append("Frobenius|Distance")
                tooltips.append("sqrt( sum( (A_ij - B_ij)^2 ) )")
            else: raise ValueError("Invalid display column name: %s" % disp)

        formatters  = (None,) + ('Conversion',) * (len(colHeadings)-1)

        table = _ReportTable(colHeadings, formatters, colHeadingLabels=tooltips,
                             confidenceRegionInfo=confidenceRegionInfo)

        formatters = (None,) + ('Normal',) * (len(colHeadings) - 1)

        if virtual_gates is None:
            iterOver = gateLabels
        else:
            iterOver = gateLabels + [v for v in virtual_gates if len(v) > 1]

        for gl in iterOver:
            #Note: gl may be a gate label (a string) or a GateString
            row_data = [ str(gl) ]
            b = bool(_tools.isstr(gl)) #whether this is a gate label or a string

            for disp in display:
                if disp == "inf":
                    fn = _reportables.Entanglement_infidelity if b else \
                         _reportables.Gatestring_entanglement_infidelity
                elif disp == "agi":
                    fn = _reportables.Avg_gate_infidelity if b else \
                         _reportables.Gatestring_avg_gate_infidelity
                elif disp == "trace":
                    fn = _reportables.Jt_diff if b else \
                         _reportables.Gatestring_jt_diff
                elif disp == "diamond":
                    fn = _reportables.Half_diamond_norm if b else \
                         _reportables.Gatestring_half_diamond_norm
                elif disp == "nuinf":
                    fn = _reportables.Nonunitary_entanglement_infidelity if b else \
                         _reportables.Gatestring_nonunitary_entanglement_infidelity
                elif disp == "nuagi":
                    fn = _reportables.Nonunitary_avg_gate_infidelity if b else \
                         _reportables.Gatestring_nonunitary_avg_gate_infidelity
                elif disp == "evinf":
                    fn = _reportables.Eigenvalue_entanglement_infidelity if b else \
                         _reportables.Gatestring_eigenvalue_entanglement_infidelity
                elif disp == "evagi":
                    fn = _reportables.Eigenvalue_avg_gate_infidelity if b else \
                         _reportables.Gatestring_eigenvalue_avg_gate_infidelity
                elif disp == "evnuinf":
                    fn = _reportables.Eigenvalue_nonunitary_entanglement_infidelity if b else \
                         _reportables.Gatestring_eigenvalue_nonunitary_entanglement_infidelity
                elif disp == "evnuagi":
                    fn = _reportables.Eigenvalue_nonunitary_avg_gate_infidelity if b else \
                         _reportables.Gatestring_eigenvalue_nonunitary_avg_gate_infidelity
                elif disp == "evdiamond":
                    fn = _reportables.Eigenvalue_diamondnorm if b else \
                         _reportables.Gatestring_eigenvalue_diamondnorm
                elif disp == "evnudiamond":
                    fn = _reportables.Eigenvalue_nonunitary_diamondnorm if b else \
                         _reportables.Gatestring_eigenvalue_nonunitary_diamondnorm
                elif disp == "frob":
                    fn = _reportables.Fro_diff if b else \
                         _reportables.Gatestring_fro_diff

                #import time as _time #DEBUG
                #tStart = _time.time() #DEBUG
                qty = _ev( fn(gateset, targetGateset, gl), confidenceRegionInfo)
                #tm = _time.time()-tStart #DEBUG
                #if tm > 0.01: print("DB: Evaluated %s in %gs" % (disp, tm)) #DEBUG
                row_data.append( qty )

            table.addrow(row_data, formatters)
        table.finish()
        return table
        
    
#    def get_spam_vs_target_table(gateset, targetGateset, confidenceRegionInfo=None):
class SpamVsTargetTable(WorkspaceTable):
    def __init__(self, ws, gateset, targetGateset, confidenceRegionInfo=None):
        """
        Create a table comparing a gateset's SPAM operations to a target gateset
        using state infidelity and trace distance.
    
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
        super(SpamVsTargetTable,self).__init__(ws, self._create, gateset,
                                               targetGateset, confidenceRegionInfo)
    
    def _create(self, gateset, targetGateset, confidenceRegionInfo):
    
        prepLabels   = gateset.get_prep_labels()
        effectLabels = gateset.get_effect_labels()
    
        colHeadings  = ('Prep/POVM', "Infidelity", "1/2 Trace|Distance", "1/2 Diamond-Dist") #HERE
        formatters   = (None,'Conversion','Conversion','Conversion')
        tooltips = ('','State infidelity or entanglement infidelity of POVM map',
                    'Trace distance between states (preps) or Jamiolkowski states of POVM maps',
                    'Half-diamond-norm distance between POVM maps')
        table = _ReportTable(colHeadings, formatters, colHeadingLabels=tooltips,
                             confidenceRegionInfo=confidenceRegionInfo)
    
        formatters = [ 'Rho' ] + [ 'Normal' ] * (len(colHeadings) - 1)
        prepInfidelities = [_ev(_reportables.Vec_infidelity(gateset, targetGateset, l,
                                                            'prep'), confidenceRegionInfo)
                            for l in prepLabels]
        prepTraceDists   = [_ev(_reportables.Vec_tr_diff(gateset, targetGateset, l,
                                                        'prep'), confidenceRegionInfo)
                            for l in prepLabels]
        prepDiamondDists = [ _objs.reportableqty.ReportableQty(_np.nan) ] * len(prepLabels)
        for rowData in _reportables.labeled_data_rows(prepLabels, confidenceRegionInfo,
                                                      prepInfidelities, prepTraceDists,
                                                      prepDiamondDists):
            table.addrow(rowData, formatters)


        #OLD - when per-effect metrics were displayed
        #formatters = [ 'Effect' ] + [ 'Normal' ] * (len(colHeadings) - 1)
        #effectInfidelities = [_ev(_reportables.Vec_infidelity(gateset, targetGateset, l,
        #                                                'effect'), confidenceRegionInfo)
        #                    for l in effectLabels]
        #effectTraceDists   = [_ev(_reportables.Vec_tr_diff(gateset, targetGateset, l,
        #                                                'effect'), confidenceRegionInfo)
        #                    for l in effectLabels]
        #for rowData in _reportables.labeled_data_rows(effectLabels, confidenceRegionInfo, 
        #                                              effectInfidelities, effectTraceDists):
        #    table.addrow(rowData, formatters)

        formatters = [ None ] + [ 'Normal' ] * (len(colHeadings) - 1)
        povmInfidelity = _ev(_reportables.POVM_entanglement_infidelity(
            gateset, targetGateset), confidenceRegionInfo)                 
        povmTraceDist = _ev(_reportables.POVM_jt_diff(
            gateset, targetGateset), confidenceRegionInfo) 
        povmDiamondDist = _ev(_reportables.POVM_half_diamond_norm(
            gateset, targetGateset), confidenceRegionInfo)
        table.addrow(['POVM', povmInfidelity, povmTraceDist, povmDiamondDist],
                     formatters)
    
        table.finish()
        return table
    
    

#    def get_gates_vs_target_err_gen_table(gateset, targetGateset, confidenceRegionInfo=None, genType="logG-logT"):
class ErrgenTable(WorkspaceTable):
    def __init__(self, ws, gateset, targetGateset, confidenceRegionInfo=None,
                 display=("errgen","H","S","A"), display_as="boxes",
                 genType="logTiG"):
                 
        """
        Create a table listing the error generators obtained by
        comparing a gateset's gates to a target gateset.
    
        Parameters
        ----------
        gateset, targetGateset : GateSet
            The gate sets to compare

        display : tuple of {"errgen","H","S","A"}
            Specifes which columns to include: the error generator itself
            and the projections of the generator onto Hamiltoian-type error
            (generators), Stochastic-type errors, and Affine-type errors.

        display_as : {"numbers", "boxes"}, optional
            How to display the requested matrices, as either numerical
            grids (fine for small matrices) or as a plot of colored boxes
            (space-conserving and better for large matrices).
    
        confidenceRegionInfo : ConfidenceRegion, optional
            If not None, specifies a confidence-region
            used to display error intervals.
    
        genType : {"logG-logT", "logTiG"}
            The type of error generator to compute.  Allowed values are:
            
            - "logG-logT" : errgen = log(gate) - log(target_gate)
            - "logTiG" : errgen = log( dot(inv(target_gate), gate) )
        
        Returns
        -------
        ReportTable
        """
        super(ErrgenTable,self).__init__(ws, self._create, gateset,
                                         targetGateset, confidenceRegionInfo,
                                         display, display_as, genType)
    
    def _create(self, gateset, targetGateset, 
                confidenceRegionInfo, display, display_as, genType):
    
        gateLabels  = list(gateset.gates.keys())  # gate labels
        basis = gateset.basis
        basisPrefix = ""
        if basis.name == "pp": basisPrefix = "Pauli "
        elif basis.name == "qt": basisPrefix = "Qutrit "
        elif basis.name == "gm": basisPrefix = "GM "
        elif basis.name == "std": basisPrefix = "Mx unit "
        
        colHeadings = ['Gate']

        for disp in display:
            if disp == "errgen":
                colHeadings.append('Error Generator')
            elif disp == "H":
                colHeadings.append('%sHamiltonian Projections' % basisPrefix)
            elif disp == "S":
                colHeadings.append('%sStochastic Projections' % basisPrefix)
            elif disp == "A":
                colHeadings.append('%sAffine Projections' % basisPrefix)
            else: raise ValueError("Invalid display element: %s" % disp)

        assert(display_as == "boxes" or display_as == "numbers")
        table = _ReportTable(colHeadings, (None,)*len(colHeadings),
                             confidenceRegionInfo=confidenceRegionInfo)

        errgenAndProjs = { }
        errgensM = []
        hamProjsM = []
        stoProjsM = []
        affProjsM = []

        def getMinMax(max_lst, M):
            #return a [min,max] already in list if there's one within an order of magnitude
            M = max(M, ABS_THRESHOLD) 
            for mx in max_lst:
                if (abs(M) >= 1e-6 and 0.9999 < mx/M < 10) or (abs(mx)<1e-6 and abs(M)<1e-6):
                    return -mx,mx
            return None

        ABS_THRESHOLD = 1e-6 #don't let color scales run from 0 to 0: at least this much!
        def addMax(max_lst, M):
            M = max(M, ABS_THRESHOLD) 
            if not getMinMax(max_lst,M):
                max_lst.append(M)
    
        #Do computation, so shared color scales can be computed
        for gl in gateLabels:
            gate = gateset.gates[gl]
            targetGate = targetGateset.gates[gl]

            if genType == "logG-logT":
                info = _ev(_reportables.LogGmlogT_and_projections(
                    gateset, targetGateset, gl), confidenceRegionInfo)
            elif genType == "logTiG":
                info = _ev(_reportables.LogTiG_and_projections(
                    gateset, targetGateset, gl), confidenceRegionInfo)
            else: raise ValueError("Invalid generator type: %s" % genType)
            errgenAndProjs[gl] = info
            
            errgen = info['error generator'].get_value()
            absMax = _np.max(_np.abs(errgen))
            addMax(errgensM, absMax)

            if "H" in display:
                absMax = _np.max(_np.abs(info['hamiltonian projections'].get_value()))
                addMax(hamProjsM, absMax)

            if "S" in display:
                absMax = _np.max(_np.abs(info['stochastic projections'].get_value()))
                addMax(stoProjsM, absMax)

            if "A" in display:
                absMax = _np.max(_np.abs(info['affine projections'].get_value()))
                addMax(affProjsM, absMax)

                
        #Do plotting
        for gl in gateLabels:
            row_data = [gl]
            row_formatters = [None]
            info = errgenAndProjs[gl]
            
            for disp in display:
                if disp == "errgen":
                    if display_as == "boxes":
                        errgen, EB = info['error generator'].get_value_and_err_bar()
                        m,M = getMinMax(errgensM, _np.max(_np.abs(errgen)))
                        errgen_fig =  _wp.GateMatrixPlot(self.ws, errgen, m,M,
                                                         basis, EBmatrix=EB)
                        row_data.append(errgen_fig)
                        row_formatters.append('Figure')
                    else:
                        row_data.append(info['error generator'])
                        row_formatters.append('Brackets')

                elif disp == "H":
                    if display_as == "boxes":
                        T = "Power %.2g" % info['hamiltonian projection power'].get_value()
                        hamProjs, EB = info['hamiltonian projections'].get_value_and_err_bar()
                        m,M = getMinMax(hamProjsM,_np.max(_np.abs(hamProjs)))
                        hamdecomp_fig = _wp.ProjectionsBoxPlot(
                            self.ws, hamProjs, basis.name, m, M,
                            boxLabels=True, EBmatrix=EB, title=T) # basis.name because projector dim is not the same as gate dim
                        row_data.append(hamdecomp_fig)
                        row_formatters.append('Figure')
                    else:
                        row_data.append(info['hamiltonian projections'])
                        row_formatters.append('Brackets')


                elif disp == "S":
                    if display_as == "boxes":
                        T = "Power %.2g" % info['stochastic projection power'].get_value()
                        stoProjs, EB = info['stochastic projections'].get_value_and_err_bar()
                        m,M = getMinMax(stoProjsM,_np.max(_np.abs(stoProjs)))
                        stodecomp_fig = _wp.ProjectionsBoxPlot(
                            self.ws, stoProjs, basis.name, m, M,
                            boxLabels=True, EBmatrix=EB, title=T) # basis.name because projector dim is not the same as gate dim
                        row_data.append(stodecomp_fig)
                        row_formatters.append('Figure')
                    else:
                        row_data.append(info['stochastic projections'])
                        row_formatters.append('Brackets')

                elif disp == "A":
                    if display_as == "boxes":
                        T = "Power %.2g" % info['affine projection power'].get_value()
                        affProjs, EB = info['affine projections'].get_value_and_err_bar()
                        m,M = getMinMax(affProjsM,_np.max(_np.abs(affProjs)))
                        affdecomp_fig = _wp.ProjectionsBoxPlot(
                            self.ws, affProjs, basis.name, m, M,
                            boxLabels=True, EBmatrix=EB, title=T) # basis.name because projector dim is not the same as gate dim
                        row_data.append(affdecomp_fig)
                        row_formatters.append('Figure')
                    else:
                        row_data.append(info['affine projections'])
                        row_formatters.append('Brackets')

            table.addrow(row_data, row_formatters)
    
        table.finish()
        return table
    
    
    
#    def get_gates_vs_target_angles_table(gateset, targetGateset, confidenceRegionInfo=None):
class old_RotationAxisVsTargetTable(WorkspaceTable):
    def __init__(self, ws, gateset, targetGateset, confidenceRegionInfo=None):
        """
        Create a table comparing the rotation axes of the single-qubit gates in
        `gateset` with those in `targetGateset`.  Differences are shown as
        angles between the rotation axes of corresponding gates.
    
        Parameters
        ----------
        gateset, targetGateset : GateSet
            The gate sets to compare.  Must be single-qubit.
    
        confidenceRegionInfo : ConfidenceRegion, optional
            If not None, specifies a confidence-region
            used to display error intervals.
    
        Returns
        -------
        ReportTable
        """
        super(RotationAxisVsTargetTable,self).__init__(
            ws, self._create, gateset, targetGateset, confidenceRegionInfo)

        
    def _create(self, gateset, targetGateset, confidenceRegionInfo):
    
        gateLabels  = list(gateset.gates.keys())  # gate labels
    
        colHeadings = ('Gate', "Angle between|rotation axes")
        formatters  = (None,'Conversion')
    
        anglesList = [_ev(_reportables.Gateset_gateset_angles_btwn_axes(
            gateset, targetGateset, gl), confidenceRegionInfo) for gl in gateLabels]

        table = _ReportTable(colHeadings, formatters, confidenceRegionInfo=confidenceRegionInfo)
    
        formatters = [None] + ['Pi']
    
        for gl, angle in zip(gateLabels, anglesList):
            rowData = [gl] + [angle]
            table.addrow(rowData, formatters)
    
        table.finish()
        return table
        
    
#    def get_gateset_decomp_table(gateset, confidenceRegionInfo=None):
class GateDecompTable(WorkspaceTable):
    def __init__(self, ws, gateset, targetGateset, confidenceRegionInfo=None):
        """
        Create table for decomposing a gateset's gates.

        This table interprets the Hamiltonian projection of the log
        of the gate matrix to extract a rotation angle and axis.

        Parameters
        ----------
        gateset : GateSet
            The estimated gate set.

        targetGateset : GateSet
            The target gate set, used to help disambiguate the matrix
            logarithms that are used in the decomposition.
    
        confidenceRegionInfo : ConfidenceRegion, optional
            If not None, specifies a confidence-region
            used to display error intervals.
    
        Returns
        -------
        ReportTable
        """
        super(GateDecompTable,self).__init__(ws, self._create, gateset,
                                             targetGateset, confidenceRegionInfo)

        
    def _create(self, gateset, targetGateset, confidenceRegionInfo):
        gateLabels = list(gateset.gates.keys())  # gate labels

        colHeadings = ('Gate','Ham. Evals.','Rotn. angle','Rotn. axis','Log Error') \
                      + tuple( [ "Axis angle w/%s" % gl for gl in gateLabels] )
        tooltips = ('Gate','Hamiltonian Eigenvalues','Rotation angle','Rotation axis',
                    'Taking the log of a gate may be performed approximately.  This is ' +
                    'error in that estimate, i.e. norm(G - exp(approxLogG)).') + \
                    tuple( [ "Angle between the rotation axis of %s and the gate of the current row" % gl for gl in gateLabels] )
        formatters = [None]*len(colHeadings)

        table = _ReportTable(colHeadings, formatters, 
                             colHeadingLabels=tooltips, confidenceRegionInfo=confidenceRegionInfo)
        formatters = (None, 'Pi','Pi', 'Figure', 'Normal') + ('Pi',)*len(gateLabels)

        decomp = _ev(_reportables.General_decomposition(
            gateset, targetGateset), confidenceRegionInfo)

        for gl in gateLabels:
            axis, axisEB = decomp[gl + ' axis'].get_value_and_err_bar()
            axisFig = _wp.ProjectionsBoxPlot(self.ws, axis, gateset.basis.name, -1.0,1.0,
                                             boxLabels=True, EBmatrix=axisEB)
            decomp[gl + ' hamiltonian eigenvalues'].scale( 1.0/_np.pi ) #scale evals to units of pi
            rowData = [gl, decomp[gl + ' hamiltonian eigenvalues'],
                       decomp[gl + ' angle'], axisFig,
                       decomp[gl + ' log inexactness'] ]

            for j,gl_other in enumerate(gateLabels):
                rotnAngle = decomp[gl + ' angle'].get_value()
                rotnAngle_other = decomp[gl_other + ' angle'].get_value()
    
                if gl_other == gl:
                    rowData.append( "" )
                elif abs(rotnAngle) < 1e-4 or abs(rotnAngle_other) < 1e-4:
                    rowData.append( "--" )
                else:
                    rowData.append(decomp[gl + ',' + gl_other + ' axis angle'])

            table.addrow(rowData, formatters)
    
        table.finish()
        return table
    

class old_GateDecompTable(WorkspaceTable):
    def __init__(self, ws, gateset, confidenceRegionInfo=None):
        """
        Create table for decomposing a single-qubit gateset's gates.

        This table interprets the eigenvectors and eigenvalues of the
        gates to extract a rotation angle, axis, and various decay
        coefficients.
    
        Parameters
        ----------
        gateset : GateSet
            A single-qubit `GateSet`.
    
        confidenceRegionInfo : ConfidenceRegion, optional
            If not None, specifies a confidence-region
            used to display error intervals.
    
        Returns
        -------
        ReportTable
        """
        super(old_GateDecompTable,self).__init__(ws, self._create, gateset, confidenceRegionInfo)

        
    def _create(self, gateset, confidenceRegionInfo):

        gateLabels = list(gateset.gates.keys())  # gate labels
        colHeadings = ('Gate','Eigenvalues','Fixed pt','Rotn. axis','Diag. decay','Off-diag. decay')
        formatters = [None]*6
    
        decomps = [_reportables.decomposition(gateset.gates[gl]) for gl in gateLabels]
        decompNames = ('fixed point',
                       'axis of rotation',
                       'decay of diagonal rotation terms',
                       'decay of off diagonal rotation terms')

        table = _ReportTable(colHeadings, formatters, confidenceRegionInfo=confidenceRegionInfo)
    
        formatters = (None, 'Vec', 'Normal', 'Normal', 'Normal', 'Normal')

        for decomp, gl in zip(decomps, gateLabels):
            evals = _ev(_reportables.Gate_eigenvalues(gateset,gl))
            decomp, decompEB = decomp.get_value_and_err_bar() #OLD
    
            rowData = [gl, evals] + [decomp.get(x,'X') for x in decompNames[0:2] ] + \
                [(decomp.get(x,'X'),decompEB) for x in decompNames[2:4] ]
    
            table.addrow(rowData, formatters)
    
        table.finish()
        return table

    
#    def get_gateset_rotn_axis_table(gateset, confidenceRegionInfo=None, showAxisAngleErrBars=True):
class old_RotationAxisTable(WorkspaceTable):
    def __init__(self, ws, gateset, confidenceRegionInfo=None, showAxisAngleErrBars=True):
        """
        Create a table of the angle between a gate rotation axes for
        gates belonging to a single-qubit gateset.
    
        Parameters
        ----------
        gateset : GateSet
            A single-qubit `GateSet`.
    
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
        super(RotationAxisTable,self).__init__(ws, self._create, gateset, confidenceRegionInfo, showAxisAngleErrBars)

        
    def _create(self, gateset, confidenceRegionInfo, showAxisAngleErrBars):
    
        gateLabels = list(gateset.gates.keys())
    
        decomps = [_reportables.decomposition(gateset.gates[gl]) for gl in gateLabels]
    
        colHeadings = ("Gate","Angle") + tuple( [ "RAAW(%s)" % gl for gl in gateLabels] )
        nCols = len(colHeadings)
        formatters = [None] * nCols

        table = "tabular"
        latex_head =  "\\begin{%s}[l]{%s}\n\hline\n" % (table, "|c" * nCols + "|")
        latex_head += "\\multirow{2}{*}{Gate} & \\multirow{2}{*}{Angle} & " + \
                      "\\multicolumn{%d}{c|}{Angle between Rotation Axes} \\\\ \cline{3-%d}\n" % (len(gateLabels),nCols)
        latex_head += " & & %s \\\\ \hline\n" % (" & ".join(gateLabels))
    
        table = _ReportTable(colHeadings, formatters,
                             customHeader={'latex': latex_head}, confidenceRegionInfo=confidenceRegionInfo)
    
        formatters = [None, 'Pi'] + ['Pi'] * len(gateLabels)
    
        rotnAxisAngles, rotnAxisAnglesEB = _ev(_reportables.Angles_btwn_rotn_axes(gateset),
                                               confidenceRegionInfo)
        rotnAngles = [ qtys['%s decomposition' % gl].get_value().get('pi rotations','X') \
                           for gl in gateLabels ] #OLD
    
        for i,gl in enumerate(gateLabels):
            decomp, decompEB = decomps[i].get_value_and_err_bar() #OLD
            rotnAngle = decomp.get('pi rotations','X')
    
            angles_btwn_rotn_axes = []
            for j,gl_other in enumerate(gateLabels):
                decomp_other, _ = decomps[j].get_value_and_err_bar() #OLD
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
    
    
#    def get_gateset_eigenval_table(gateset, targetGateset, figFilePrefix, maxWidth=6.5, maxHeight=8.0,
#                                   confidenceRegionInfo=None):
class GateEigenvalueTable(WorkspaceTable):
    def __init__(self, ws, gateset, targetGateset=None,
                 confidenceRegionInfo=None,
                 display=('evals','rel','log-evals','log-rel','polar','relpolar'),
                 virtual_gates=None):
        """
        Create table which lists and displays (using a polar plot)
        the eigenvalues of a gateset's gates.
    
        Parameters
        ----------
        gateset : GateSet
            The GateSet
    
        targetGateset : GateSet, optional
            The target gate set.  If given, the target's eigenvalue will
            be plotted alongside `gateset`'s gate eigenvalue, the
            "relative eigenvalues".
        
        confidenceRegionInfo : ConfidenceRegion, optional
            If not None, specifies a confidence-region
            used to display error intervals.

        display : tuple
            A tuple of one or more of the allowed options (see below) which
            specify which columns are displayed in the table.  If 
            `targetGateset` is None, then `"target"`, `"rel"`, `"log-rel"`
            `"relpolar"`, `"gidm"`, and `"giinf"` will be silently ignored.
        
            - "evals" : the gate eigenvalues
            - "target" : the target gate eigenvalues
            - "rel" : the relative-gate eigenvalues
            - "log-evals" : the (complex) logarithm of the eigenvalues
            - "log-rel" : the (complex) logarithm of the relative eigenvalues
            - "polar": a polar plot of the gate eigenvalues
            - "relpolar" : a polar plot of the relative-gate eigenvalues
            - "absdiff-evals" : absolute difference w/target eigenvalues
            - "infdiff-evals" : 1-Re(z0.C*z) difference w/target eigenvalues
            - "absdiff-log-evals" : Re & Im differences in eigenvalue logarithms
            - "gidm" : the gauge-invariant diamond norm metric
            - "giinf" : the gauge-invariant infidelity metric

        virtual_gates : list, optional
            If not None, a list of `GateString` objects specifying additional "gates"
            (i.e. processes) to compute eigenvalues of.  Length-1 gate strings are
            automatically discarded so they are not displayed twice.
    
        Returns
        -------
        ReportTable
        """
        super(GateEigenvalueTable,self).__init__(ws, self._create, gateset,
                                                 targetGateset,
                                                 confidenceRegionInfo, display,
                                                 virtual_gates)
        
    def _create(self, gateset, targetGateset,               
                confidenceRegionInfo, display,
                virtual_gates):
        
        gateLabels = list(gateset.gates.keys())  # gate labels
        colHeadings = ['Gate'] if (virtual_gates is None) else ['Gate or Germ']
        for disp in display:
            if disp == "evals":
                colHeadings.append('Eigenvalues (E)')
            elif disp == "target":
                colHeadings.append('Target Evals. (T)')
            elif disp == "rel":
                if(targetGateset is not None): #silently ignore
                    colHeadings.append('Rel. Evals (R)')
            elif disp == "log-evals":
                colHeadings.append('Re log(E)')
                colHeadings.append('Im log(E)')
            elif disp == "log-rel":
                colHeadings.append('Re log(R)')
                colHeadings.append('Im log(R)')
            elif disp == "polar":
                colHeadings.append('Eigenvalues')
            elif disp == "relpolar":
                if(targetGateset is not None): #silently ignore
                    colHeadings.append('Rel. Evals')
            elif disp == "absdiff-evals":
                if(targetGateset is not None): #silently ignore
                    colHeadings.append('|E - T|')
            elif disp == "infdiff-evals":
                if(targetGateset is not None): #silently ignore
                    colHeadings.append('1.0 - Re(T.C*E)')
            elif disp == "absdiff-log-evals":
                if(targetGateset is not None): #silently ignore
                    colHeadings.append('|Re(log E) - Re(log T)|')
                    colHeadings.append('|Im(log E) - Im(log T)|')
            elif disp == "gidm":
                if(targetGateset is not None): #silently ignore
                    colHeadings.append('Gauge-inv. diamond norm')
            elif disp == "giinf":
                if(targetGateset is not None): #silently ignore
                    colHeadings.append('Gauge-inv. infidelity')
            else:
                raise ValueError("Invalid display element: %s" % disp)

        formatters = [None]*len(colHeadings)
    
        table = _ReportTable(colHeadings, formatters, confidenceRegionInfo=confidenceRegionInfo)

        if virtual_gates is None:
            iterOver = gateLabels
        else:
            iterOver = gateLabels + [v for v in virtual_gates if len(v) > 1]
    
        for gl in iterOver:
            #Note: gl may be a gate label (a string) or a GateString
            row_data = [ str(gl) ]
            row_formatters = [None]

            #import time as _time #DEBUG
            #tStart = _time.time() #DEBUG
            fn = _reportables.Gate_eigenvalues if _tools.isstr(gl) else \
                 _reportables.Gatestring_eigenvalues
            evals = _ev(fn(gateset,gl), confidenceRegionInfo)
            #tm = _time.time() - tStart #DEBUG
            #if tm > 0.01: print("DB: Gate eigenvalues in %gs" % tm) #DEBUG
                
            evals = evals.reshape(evals.size, 1)
            #OLD: format to 2-columns - but polar plots are big, so just stick to 1col now
            #try: evals = evals.reshape(evals.size//2, 2) #assumes len(evals) is even!
            #except: evals = evals.reshape(evals.size, 1)

            if targetGateset is not None:
                #TODO: move this to a reportable qty to get error bars?
                
                if _tools.isstr(gl):
                    target_evals = _np.linalg.eigvals( targetGateset.gates[gl] ) #no error bars
                else:
                    target_evals = _np.linalg.eigvals( targetGateset.product(gl) ) #no error bars

                if any([(x in display) for x in ('rel','log-rel','relpolar')]):
                    if _tools.isstr(gl):
                        rel_evals = _ev(_reportables.Rel_gate_eigenvalues(gateset, targetGateset, gl), confidenceRegionInfo)
                    else:
                        rel_evals = _ev(_reportables.Rel_gatestring_eigenvalues(gateset, targetGateset, gl), confidenceRegionInfo)

                # permute target eigenvalues according to min-weight matching
                _, pairs = _tools.minweight_match( evals.get_value(), target_evals, lambda x,y: abs(x-y) )
                matched_target_evals = target_evals.copy()
                for i,j in pairs:
                    matched_target_evals[i] = target_evals[j]
                target_evals = matched_target_evals
                target_evals = target_evals.reshape(evals.value.shape)
                   # b/c evals have shape (x,1) and targets (x,),
                   # which causes problems when we try to subtract them

            for disp in display:
                if disp == "evals":
                    row_data.append( evals )
                    row_formatters.append('Normal')

                elif disp == "target" and targetGateset is not None:
                    row_data.append( target_evals )
                    row_formatters.append('Normal')

                elif disp == "rel" and targetGateset is not None:
                    row_data.append( rel_evals )
                    row_formatters.append('Normal')

                elif disp == "log-evals":
                    logevals = evals.log()
                    row_data.append( logevals.real() )
                    row_data.append( logevals.imag()/_np.pi )
                    row_formatters.append('Normal')
                    row_formatters.append('Pi')

                elif disp == "log-rel":
                    log_relevals = rel_evals.log()
                    row_data.append( log_relevals.real() )
                    row_data.append( log_relevals.imag()/_np.pi )
                    row_formatters.append('Vec') 
                    row_formatters.append('Pi')

                elif disp == "absdiff-evals":
                    absdiff_evals = evals.absdiff(target_evals)
                    row_data.append( absdiff_evals )
                    row_formatters.append('Vec')
                    
                elif disp == "infdiff-evals":
                    infdiff_evals = evals.infidelity_diff(target_evals)
                    row_data.append( infdiff_evals )
                    row_formatters.append('Vec')

                elif disp == "absdiff-log-evals":
                    log_evals = evals.log()
                    re_diff, im_diff = log_evals.absdiff( _np.log(target_evals.astype(complex)), separate_re_im=True )
                    row_data.append( re_diff )
                    row_data.append( (im_diff/_np.pi).mod(2.0) )
                    row_formatters.append('Vec')
                    row_formatters.append('Pi')

                elif disp == "gidm":
                    if targetGateset is None:
                        fn = _reportables.gaugeinv_diamondnorm if _tools.isstr(gl) else \
                             _reportables.gatestring_gaugeinv_diamondnorm
                        gidm = _ev(fn(gateset, targetGateset, gl), confidenceRegionInfo)
                        row_data.append( gidm )
                        row_formatters.append('Normal')
                
                elif disp == "giinf":
                    if targetGateset is None:
                        fn = _reportables.gaugeinv_infidelity if _tools.isstr(gl) else \
                             _reportables.gatestring_gaugeinv_infidelity
                        giinf = _ev(fn(gateset, targetGateset, gl), confidenceRegionInfo)
                        row_data.append( giinf )
                        row_formatters.append('Normal')
                    
                elif disp == "polar":
                    evals_val = evals.get_value()
                    if targetGateset is None:
                        fig = _wp.PolarEigenvaluePlot(
                            self.ws,[evals_val],["blue"],centerText=str(gl))
                    else:
                        fig = _wp.PolarEigenvaluePlot(
                            self.ws,[target_evals,evals_val],
                            ["black","blue"],["target","gate"], centerText=str(gl))
                    row_data.append( fig )
                    row_formatters.append('Figure')

                elif disp == "relpolar" and targetGateset is not None:
                    rel_evals_val = rel_evals.get_value()
                    fig = _wp.PolarEigenvaluePlot(
                        self.ws,[rel_evals_val],["red"],["rel"],centerText=str(gl))
                    row_data.append( fig )
                    row_formatters.append('Figure')
            table.addrow(row_data, row_formatters)
        table.finish()
        return table
    
        
    
#    def get_dataset_overview_table(dataset, target, maxlen=10, fixedLists=None,
#                                   maxLengthList=None):
class DataSetOverviewTable(WorkspaceTable):
    def __init__(self, ws, dataset, maxLengthList=None):
        """
        Create a table that gives a summary of the properties of `dataset`.
    
        Parameters
        ----------
        dataset : DataSet
            The DataSet
        
        maxLengthList : list of ints, optional
            A list of the maximum lengths used, if available.
    
        Returns
        -------
        ReportTable
        """
        super(DataSetOverviewTable,self).__init__(ws, self._create, dataset, maxLengthList)
    
    def _create(self, dataset, maxLengthList):
    
        colHeadings = ('Quantity','Value')
        formatters = (None,None)
    
        table = _ReportTable(colHeadings, formatters)
    
        minN = round(min([ row.total() for row in dataset.itervalues()]))
        maxN = round(max([ row.total() for row in dataset.itervalues()]))
        cntStr = "[%d,%d]" % (minN,maxN) if (minN != maxN) else "%d" % round(minN)
    
        table.addrow(("Number of strings", str(len(dataset))), (None,None))
        table.addrow(("Gate labels", ", ".join(dataset.get_gate_labels()) ), (None,None))
        table.addrow(("SPAM labels",  ", ".join(dataset.get_spam_labels()) ), (None,None))
        table.addrow(("Counts per string", cntStr  ), (None,None))

        if maxLengthList is not None:
            table.addrow(("Max. Lengths", ", ".join(map(str,maxLengthList)) ), (None,None))
        if hasattr(dataset,'comment') and dataset.comment is not None:
            commentLines = dataset.comment.split('\n')
            for i,commentLine in enumerate(commentLines,start=1):
                table.addrow(("User comment %d" % i, commentLine  ), (None,'Verbatim'))
    
        table.finish()
        return table
    
    
#    def get_chi2_progress_table(Ls, gatesetsByL, gateStringsByL, dataset,
#                                gateLabelAliases=None):
class FitComparisonTable(WorkspaceTable):
    def __init__(self, ws, Xs, gssByX, gatesetByX, dataset, objective="logl",
                 Xlabel='L', NpByX=None):
        """
        Create a table showing how the chi^2 or log-likelihood changed with
        successive GST iterations.
    
        Parameters
        ----------
        Xs : list of integers
            List of X-values. Typically these are the maximum lengths or
            exponents used to index the different iterations of GST.

        gssByX : list of LsGermsStructure
            Specifies the set (& structure) of the gate strings used at each X.

        gatesetByX : list of GateSets
            `GateSet`s corresponding to each X value.
    
        dataset : DataSet
            The data set to compare each gate set against.
    
        objective : {"logl", "chi2"}, optional
            Whether to use log-likelihood or chi^2 values.

        Xlabel : str, optional
            A label for the 'X' variable which indexes the different gate sets.
            This string will be the header of the first table column.

        NpByX : list of ints, optional
            A list of parameter counts to use for each X.  If None, then
            the number of non-gauge parameters for each gate set is used.
        
        
        Returns
        -------
        ReportTable
        """
        super(FitComparisonTable,self).__init__(ws, self._create, Xs, gssByX, gatesetByX,
                                                dataset, objective, Xlabel, NpByX)
    
    def _create(self, Xs, gssByX, gatesetByX, dataset, objective, Xlabel, NpByX):

        if objective == "chi2":
            colHeadings = {
                'latex': (Xlabel,'$\\chi^2$','$k$','$\\chi^2-k$','$\sqrt{2k}$',
                          '$N_\\sigma$','$N_s$','$N_p$', 'Rating'),
                'html': (Xlabel,'&chi;<sup>2</sup>','k','&chi;<sup>2</sup>-k',
                         '&radic;<span style="text-decoration:overline;">2k</span>',
                         'N<sub>sigma</sub>','N<sub>s</sub>','N<sub>p</sub>', 'Rating'),
                'python': (Xlabel,'chi^2','k','chi^2-k','sqrt{2k}','N_{sigma}','N_s','N_p', 'Rating')
                }
                
        elif objective == "logl":
            colHeadings = {
                'latex': (Xlabel,'$2\Delta\\log(\\mathcal{L})$','$k$','$2\Delta\\log(\\mathcal{L})-k$',
                          '$\sqrt{2k}$','$N_\\sigma$','$N_s$','$N_p$', 'Rating'),
                'html': (Xlabel,'2&Delta;(log L)','k','2&Delta;(log L)-k',
                         '&radic;<span style="text-decoration:overline;">2k</span>',
                         'N<sub>sigma</sub>','N<sub>s</sub>','N<sub>p</sub>', 'Rating'),
                'python': (Xlabel,'2*Delta(log L)','k','2*Delta(log L)-k','sqrt{2k}',
                                               'N_{sigma}','N_s','N_p', 'Rating')
                }
        else:
            raise ValueError("Invalid `objective` argument: %s" % objective)

        if NpByX is None:
            NpByX = [ gs.num_nongauge_params() for gs in gatesetByX ]

        tooltips = ('', 'Difference in logL', 'number of degrees of freedom',
                    'difference between observed logl and expected mean',
                    'std deviation', 'number of std deviation', 'dataset dof',
                    'number of gateset parameters', '1-5 star rating (like Netflix)')
        table = _ReportTable(colHeadings, None, colHeadingLabels=tooltips)
        
        for X,gs,gss,Np in zip(Xs,gatesetByX,gssByX,NpByX):
            gstrs = gss.allstrs
            
            if objective == "chi2":
                fitQty = _tools.chi2( dataset, gs, gstrs,
                                    minProbClipForWeighting=1e-4,
                                    gateLabelAliases=gss.aliases )
            elif objective == "logl":
                logL_upperbound = _tools.logl_max(dataset, gstrs, gateLabelAliases=gss.aliases)
                logl = _tools.logl( gs, dataset, gstrs, gateLabelAliases=gss.aliases)
                fitQty = 2*(logL_upperbound - logl) # twoDeltaLogL
                if(logL_upperbound < logl):
                    raise ValueError("LogL upper bound = %g but logl = %g!!" % (logL_upperbound, logl))

            Ns = len(gstrs)*(len(dataset.get_spam_labels())-1) #number of independent parameters in dataset
            k = max(Ns-Np,1) #expected chi^2 or 2*(logL_ub-logl) mean
            Nsig = (fitQty-k)/_np.sqrt(2*k)
            if Ns <= Np: _warnings.warn("Max-model params (%d) <= gate set params (%d)!  Using k == 1." % (Ns,Np))
            #pv = 1.0 - _stats.chi2.cdf(chi2,k) # reject GST model if p-value < threshold (~0.05?)
    
            if   (fitQty-k) < _np.sqrt(2*k): rating = 5
            elif (fitQty-k) < 2*k: rating = 4
            elif (fitQty-k) < 5*k: rating = 3
            elif (fitQty-k) < 10*k: rating = 2
            else: rating = 1
            table.addrow(
                        (str(X),fitQty,k,fitQty-k,_np.sqrt(2*k),Nsig,Ns,Np,"<STAR>"*rating),
                        (None,'Normal','Normal','Normal','Normal','Rounded','Normal','Normal','Conversion'))
    
        table.finish()
        return table
        
    
#    def get_gatestring_multi_table(gsLists, titles, commonTitle=None):
class GatestringTable(WorkspaceTable):
    def __init__(self, ws, gsLists, titles, nCols=1, commonTitle=None):
        """
        Creates a table of enumerating one or more sets of gate strings.
    
        Parameters
        ----------
        gsLists : GateString list or list of GateString lists
            List(s) of gate strings to put in table.
    
        titles : string or list of strings
            The title(s) for the different string lists.  These are displayed in
            the relevant table columns containing the strings.

        nCols : int, optional
            The number of *data* columns, i.e. those containing
            gate strings, for each string list.
    
        commonTitle : string, optional
            A single title string to place in a cell spanning across
            all the other column headers.
    
        Returns
        -------
        ReportTable
        """
        super(GatestringTable,self).__init__(ws, self._create, gsLists, titles,
                                             nCols, commonTitle)

        
    def _create(self, gsLists, titles, nCols, commonTitle):

        if isinstance(gsLists[0], _objs.GateString) or \
           (isinstance(gsLists[0], tuple) and _tools.isstr(gsLists[0][0])):
            gsLists = [ gsLists ]

        if _tools.isstr(titles): titles = [ titles ]*len(gsLists)
            
        colHeadings = (('#',) + tuple(titles))*nCols
        formatters = (('Conversion',) + ('Normal',)*len(titles))*nCols
    
        if commonTitle is None:
            table = _ReportTable(colHeadings, formatters)
        else:
            table = "tabular"
            colHeadings = ('\\#',) + tuple(titles)
            latex_head  = "\\begin{%s}[l]{%s}\n\hline\n" % (table, "|c" * len(colHeadings) + "|")
            latex_head += " & \multicolumn{%d}{c|}{%s} \\\\ \hline\n" % (len(colHeadings)-1,commonTitle)
            latex_head += "%s \\\\ \hline\n" % (" & ".join(colHeadings))

            colHeadings = ('#',) + tuple(titles)
            html_head = '<table class="%(tableclass)s" id="%(tableid)s" ><thead>'
            html_head += '<tr><th></th><th colspan="%d">%s</th></tr>\n' % (len(colHeadings)-1,commonTitle)
            html_head += "<tr><th> %s </th></tr>" % (" </th><th> ".join(colHeadings))
            html_head += "</thead><tbody>"
            table = _ReportTable(colHeadings, formatters,
                                 customHeader={'latex': latex_head,
                                               'html': html_head })
    
        formatters = (('Normal',) + ('GateString',)*len(gsLists))*nCols

        maxListLength = max(list(map(len,gsLists)))
        nRows = (maxListLength+(nCols-1)) // nCols #ceiling
    
        #for i in range( max([len(gsl) for gsl in gsLists]) ):
        for i in range(nRows):
            rowData = []
            for k in range(nCols):
                l = i+nRows*k #index of gatestring
                rowData.append( l+1 )
                for gsList in gsLists:
                    if l < len(gsList):
                        rowData.append( gsList[l] )
                    else:
                        rowData.append( None ) #empty string
            table.addrow(rowData, formatters)
    
        table.finish()
        return table
        
    
#    def get_projected_err_gen_comparison_table(gateset, targetGateset,
#                                               compare_with="target",
#                                               genType="logG-logT"):
class GatesSingleMetricTable(WorkspaceTable):
    def __init__(self, ws, gatesets, titles, targetGateset,
                 metric="infidelity"):
        """
        Create a table comparing the gates of various gate sets (`gatesets`) to
        those of `targetGateset` using the metric named by `metric`.
    
        Parameters
        ----------
        gatesets : list of GateSets
            The gate sets to compare with `targetGateset`

        titles : list of strs
            A list of titles used to describe each element of `gatesets`.

        targetGateset : GateSet
            The gate set to compare against.
    
        metric : {"infidelity","diamond","jtrace"}
            Specifies which metric to compute and display.
    
        Returns
        -------
        ReportTable
        """
        super(GatesSingleMetricTable,self).__init__(
            ws, self._create, gatesets, titles, targetGateset, metric)
    
    def _create(self, gatesets, titles, targetGateset, metric):
    
        gateLabels = list(targetGateset.gates.keys())  # use target's gate labels
        basis = targetGateset.basis

        #Check that all gatesets are in the same basis as targetGateset
        for title,gateset in zip(titles,gatesets):
            if basis.name != gateset.basis.name:
                raise ValueError("Basis mismatch between '%s' gateset (%s) and target (%s)!"\
                                 % (title, gateset.basis.name, basis.name))

        #Do computation first
        metricVals = [] #one element per row (gate label)
        for gl in gateLabels:
            cmpGate = targetGateset.gates[gl]
            dct = {}
            for title,gateset in zip(titles,gatesets):
                gate = gateset.gates[gl]
                if metric == "infidelity":
                    dct[title] = 1-_tools.process_fidelity(gate, cmpGate, basis)
                elif metric == "diamond":
                    dct[title] = _tools.jtracedist(gate, cmpGate, basis)
                elif metric == "jtrace":
                    dct[title] = 0.5 * _tools.diamonddist(gate, cmpGate, basis)
                else: raise ValueError("Invalid `metric` argument: %s" % metric)
            metricVals.append(dct)

            
        if metric == "infidelity":
            niceNm = "Process Infidelity"
        elif metric == "diamond":
            niceNm = "1/2 Diamond-Norm"
        elif metric == "jtrace":
            niceNm = "1/2 Trace Distance"
        else: raise ValueError("Invalid `metric` argument: %s" % metric)

        def mknice(x):
            if x == "H": return "$\mathcal{H}$"
            if x == "S": return "$\mathcal{H}$"
            if x in ("H + S","H+S"): return "$\mathcal{H} + \mathcal{S}$"
            return x
        
        colHeadings = ("Gate",) + \
                      tuple( [ "%s(%s)" % (niceNm,title) for title in titles] )
        nCols = len(colHeadings)
        formatters = [None] + ['GatesetType']*(nCols-1)

        latex_head =  "\\begin{tabular}[l]{%s}\n\hline\n" % ("|c" * nCols + "|")
        latex_head += "\\multirow{2}{*}{Gate} & " + \
                      "\\multicolumn{%d}{c|}{%s} \\\\ \cline{2-%d}\n" % (len(titles),niceNm,nCols)
        latex_head += " & " + " & ".join([mknice(t) for t in titles]) + "\\\\ \hline\n"

        html_head = '<table class="%(tableclass)s" id="%(tableid)s" ><thead>'
        html_head += '<tr><th rowspan="2"></th>' + \
                     '<th colspan="%d">%s</th></tr>\n' % (len(titles),niceNm)
        html_head += "<tr><th>" +  " </th><th> ".join([mknice(t) for t in titles]) + "</th></tr>\n"
        html_head += "</thead><tbody>"
    
        table = _ReportTable(colHeadings, formatters,
                             customHeader={'latex': latex_head,
                                           'html': html_head} )

        for gl,dct in zip(gateLabels,metricVals):
            row_data = [gl] + [ dct[t] for t in titles ]
            row_formatters = [None] + ['Normal']*len(titles)
            table.addrow(row_data, row_formatters)
    
        table.finish()
        return table
    
    
#    def get_err_gen_projector_boxes_table(gateset_dim, projection_type,
#                                          projection_basis, figFilePrefix,
#                                          maxWidth=6.5, maxHeight=8.0):
class StandardErrgenTable(WorkspaceTable):
    def __init__(self, ws, gateset_dim, projection_type,
                 projection_basis):
        """
        Create a table of the "standard" gate error generators, such as those
        which correspond to Hamiltonian or Stochastic errors.  Each generator
        is shown as grid of colored boxes.
    
        Parameters
        ----------
        gateset_dim : int
            The dimension of the gate set, which equals the number of 
            rows (or columns) in a gate matrix (e.g., 4 for a single qubit).
    
        projection_type : {"hamiltonian", "stochastic"}
            The type of error generator projectors to create a table for.
            If "hamiltonian", then use the Hamiltonian generators which take a
            density matrix rho -> -i*[ H, rho ] for basis matrix H.
            If "stochastic", then use the Stochastic error generators which take
            rho -> P*rho*P for basis matrix P (recall P is self adjoint).
    
        projection_basis : {'std', 'gm', 'pp', 'qt'}
          Which basis is used to construct the error generators.  Allowed
          values are Matrix-unit (std), Gell-Mann (gm), Pauli-product (pp)
          and Qutrit (qt).
        
        Returns
        -------
        ReportTable
        """
        super(StandardErrgenTable,self).__init__(
            ws, self._create, gateset_dim, projection_type,
            projection_basis)
    

    def _create(self,  gateset_dim, projection_type,
                projection_basis):
    
        d2 = gateset_dim # number of projections == dim of gate
        d = int(_np.sqrt(d2)) # dim of density matrix
        nQubits = _np.log2(d)
    
        #Get a list of the d2 generators (in corresspondence with the
        #  given basis matrices)
        lindbladMxs = _tools.std_error_generators(d2, projection_type,
                                                  projection_basis) # in std basis
    
        if not _np.isclose(round(nQubits),nQubits): 
            #Non-integral # of qubits, so just show as a single row
            yd,xd = 1,d
            xlabel = ""; ylabel = ""        
        if nQubits == 1:
            yd,xd = 1,2 # y and x pauli-prod *basis* dimensions
            xlabel = "Q1"; ylabel = ""
        elif nQubits == 2:
            yd,xd = 2,2
            xlabel = "Q2"; ylabel="Q1"
        else:
            yd,xd = 2,d/2
            xlabel = "Q*"; ylabel="Q1"
    
        topright = "%s \\ %s" % (ylabel,xlabel) if (len(ylabel) > 0) else ""
        colHeadings=[topright] + \
            [("%s" % x) if len(x) else "" \
                 for x in _tools.basis_element_labels(projection_basis,xd)]
        rowLabels=[("%s" % x) if len(x) else "" \
                     for x in _tools.basis_element_labels(projection_basis,yd)]
    
        xLabels = _tools.basis_element_labels(projection_basis,xd)
        yLabels = _tools.basis_element_labels(projection_basis,yd)
    
        table = _ReportTable(colHeadings,["Conversion"]+[None]*(len(colHeadings)-1))
            
        iCur = 0
        for i,ylabel  in enumerate(yLabels):
            rowData = [rowLabels[i]]
            rowFormatters = [None]
    
            for j,xlabel in enumerate(xLabels):
                projector = lindbladMxs[iCur]; iCur += 1
                projector = _tools.change_basis(projector,"std",projection_basis)
                m,M = -_np.max(_np.abs(projector)), _np.max(_np.abs(projector))
                fig = _wp.GateMatrixPlot(self.ws, projector, m,M,
                                         projection_basis, d)
                rowData.append(fig)
                rowFormatters.append('Figure')
    
            table.addrow(rowData, rowFormatters)
    
        table.finish()
        return table
    
    
    
#    def get_gaugeopt_params_table(gaugeOptArgs):
class GaugeOptParamsTable(WorkspaceTable):
    def __init__(self, ws, gaugeOptArgs):
        """
        Create a table displaying a list of gauge
        optimzation parameters.
    
        Parameters
        ----------
        gaugeOptArgs : dict or list
            A dictionary or list of dictionaries specifying values for
            zero or more of the *arguments* of pyGSTi's
            :func:`gaugeopt_to_target` function.
    
        Returns
        -------
        ReportTable
        """
        super(GaugeOptParamsTable,self).__init__(ws, self._create, gaugeOptArgs)
    
    def _create(self, gaugeOptArgs):
        
        colHeadings = ('G-Opt Param','Value')
        formatters = ('Bold','Bold')

        if gaugeOptArgs == False: #signals *no* gauge optimization
            goargs_list = [ {'Method': "No gauge optimization was performed" } ]
        else:
            goargs_list = [gaugeOptArgs] if hasattr(gaugeOptArgs,'keys') \
                            else gaugeOptArgs
            
        table = _ReportTable(colHeadings, formatters)

        for i,goargs in enumerate(goargs_list):
            pre = ("%d: " % i) if len(goargs_list) > 1 else ""
            if 'method' in goargs:
                table.addrow(("%sMethod" % pre, str(goargs['method'])), (None,None))
            #if 'TPpenalty' in goargs: #REMOVED
            #    table.addrow(("%sTP penalty factor" % pre, str(goargs['TPpenalty'])), (None,None))
            if 'cptp_penalty_factor' in goargs and goargs['cptp_penalty_factor'] != 0:
                table.addrow(("%sCP penalty factor" % pre, str(goargs['cptp_penalty_factor'])), (None,None))
            if 'spam_penalty_factor' in goargs and goargs['spam_penalty_factor'] != 0:
                table.addrow(("%sSPAM penalty factor" % pre, str(goargs['spam_penalty_factor'])), (None,None))
            if 'gatesMetric' in goargs:
                table.addrow(("%sMetric for gate-to-target" % pre, str(goargs['gatesMetric'])), (None,None))
            if 'spamMetric' in goargs:
                table.addrow(("%sMetric for SPAM-to-target" % pre, str(goargs['spamMetric'])), (None,None))
            if 'itemWeights' in goargs:
                if goargs['itemWeights']:
                    table.addrow(("%sItem weights" % pre, ", ".join([("%s=%.2g" % (k,v)) 
                                   for k,v in goargs['itemWeights'].items()])), (None,None))
            if 'gauge_group' in goargs:
                table.addrow( ("%sGauge group" % pre, goargs['gauge_group'].name) , (None,None))
    
        table.finish()
        return table
    
    
    
#    def get_metadata_table(gateset, result_options, result_params):
class MetadataTable(WorkspaceTable):
    def __init__(self, ws, gateset, params):
        """
        Create a table of parameters and options from a `Results` object.
    
        Parameters
        ----------
        gateset : GateSet
            The gateset (usually the final estimate of a GST computation) to 
            show information for (e.g. the types of its gates).
    
        params: dict
            A parameter dictionary to display
    
        Returns
        -------
        ReportTable
        """
        super(MetadataTable,self).__init__(ws, self._create, gateset, params)
    
    def _create(self, gateset, params_dict):
    
        colHeadings = ('Quantity','Value')
        formatters = ('Bold','Bold')
    
        #custom latex header for maximum width imposed on 2nd col
        latex_head =  "\\begin{tabular}[l]{|c|p{3in}|}\n\hline\n"
        latex_head += "\\textbf{Quantity} & \\textbf{Value} \\\\ \hline\n"
        table = _ReportTable(colHeadings, formatters,
                             customHeader={'latex': latex_head} )
        
        for key in sorted(list(params_dict.keys())):
            if key in ['L,germ tuple base string dict', 'profiler']: continue #skip these
            if key == 'gaugeOptParams':
                if isinstance(params_dict[key],dict):
                    val = params_dict[key].copy()
                    if 'targetGateset' in val:
                        del val['targetGateset'] #don't print this!
                                
                elif isinstance(params_dict[key],list):
                    val = []
                    for go_param_dict in params_dict[key]:
                        if isinstance(go_param_dict,dict): #to ensure .copy() exists
                            val.append(go_param_dict.copy())
                            if 'targetGateset' in val[-1]:
                                del val[-1]['targetGateset'] #don't print this!
            else:
                val = params_dict[key]
            table.addrow((key, str(val)), (None,'Verbatim'))
    
    
        for lbl,vec in gateset.preps.items():
            if isinstance(vec, _objs.StaticSPAMVec): paramTyp = "static"
            elif isinstance(vec, _objs.FullyParameterizedSPAMVec): paramTyp = "full"
            elif isinstance(vec, _objs.TPParameterizedSPAMVec): paramTyp = "TP"
            else: paramTyp = "unknown"
            table.addrow((lbl + " parameterization", paramTyp), (None,'Verbatim'))
    
        for lbl,vec in gateset.effects.items():
            if isinstance(vec, _objs.StaticSPAMVec): paramTyp = "static"
            elif isinstance(vec, _objs.FullyParameterizedSPAMVec): paramTyp = "full"
            elif isinstance(vec, _objs.TPParameterizedSPAMVec): paramTyp = "TP"
            else: paramTyp = "unknown"
            table.addrow((lbl + " parameterization", paramTyp), (None,'Verbatim'))
    
        #Not displayed since the POVM identity is always fully parameterized,
        # even through it doesn't contribute to the gate set parameters (a special case)
        #if gateset.povm_identity is not None:
        #    vec = gateset.povm_identity
        #    if isinstance(vec, _objs.StaticSPAMVec): paramTyp = "static"
        #    elif isinstance(vec, _objs.FullyParameterizedSPAMVec): paramTyp = "full"
        #    elif isinstance(vec, _objs.TPParameterizedSPAMVec): paramTyp = "TP"
        #    else: paramTyp = "unknown"
        #    table.addrow(("POVM identity parameterization", paramTyp), (None,'Verbatim'))
    
        for gl,gate in gateset.gates.items():
            if isinstance(gate, _objs.StaticGate): paramTyp = "static"
            elif isinstance(gate, _objs.FullyParameterizedGate): paramTyp = "full"
            elif isinstance(gate, _objs.TPParameterizedGate): paramTyp = "TP"
            elif isinstance(gate, _objs.LinearlyParameterizedGate): paramTyp = "linear"
            elif isinstance(gate, _objs.EigenvalueParameterizedGate): paramTyp = "eigenvalue"
            elif isinstance(gate, _objs.LindbladParameterizedGate):
                paramTyp = "Lindblad"
                if gate.cptp: paramTyp += " CPTP "
                paramTyp += "(%d, %d params)" % (gate.ham_basis_size, gate.other_basis_size)
            else: paramTyp = "unknown"
            table.addrow((gl + " parameterization", paramTyp), (None,'Verbatim'))
            
        
        table.finish()
        return table
    
    
#    def get_software_environment_table():
class SoftwareEnvTable(WorkspaceTable):
    def __init__(self, ws):
        """
        Create a table displaying the software environment relevant to pyGSTi.
    
        Returns
        -------
        ReportTable
        """
        super(SoftwareEnvTable,self).__init__(ws, self._create)
    
    def _create(self):
    
        import platform
        
        def get_version(moduleName):
            if moduleName == "cvxopt":
                #special case b/c cvxopt can be weird...
                try:
                    mod = __import__("cvxopt.info")
                    return str(mod.info.version)
                except Exception: pass #try the normal way below
    
            try:
                mod = __import__(moduleName)
                return str(mod.__version__)
            except ImportError:
                return "missing"
            except AttributeError:
                return "ver?"
            except Exception:
                return "???"
            
        colHeadings = ('Quantity','Value')
        formatters = ('Bold','Bold')
    
        #custom latex header for maximum width imposed on 2nd col
        latex_head =  "\\begin{tabular}[l]{|c|p{3in}|}\n\hline\n"
        latex_head += "\\textbf{Quantity} & \\textbf{Value} \\\\ \hline\n"
        table = _ReportTable(colHeadings, formatters, 
                             customHeader={'latex': latex_head} )
        
        #Python package information
        from .._version import __version__ as pyGSTi_version
        table.addrow(("pyGSTi version", str(pyGSTi_version)), (None,'Verbatim'))
    
        packages = ['numpy','scipy','matplotlib','ply','cvxopt','cvxpy',
                    'nose','PIL','psutil']
        for pkg in packages:
            table.addrow((pkg, get_version(pkg)), (None,'Verbatim'))
    
        #Python information
        table.addrow(("Python version", str(platform.python_version())), (None,'Verbatim'))
        table.addrow(("Python type", str(platform.python_implementation())), (None,'Verbatim'))
        table.addrow(("Python compiler", str(platform.python_compiler())), (None,'Verbatim'))
        table.addrow(("Python build", str(platform.python_build())), (None,'Verbatim'))
        table.addrow(("Python branch", str(platform.python_branch())), (None,'Verbatim'))
        table.addrow(("Python revision", str(platform.python_revision())), (None,'Verbatim'))
    
        #Platform information
        (system, node, release, version, machine, processor) = platform.uname()
        table.addrow(("Platform summary", str(platform.platform())), (None,'Verbatim'))
        table.addrow(("System", str(system)), (None,'Verbatim'))
        #table.addrow(("Sys Node", str(node)), (None,'Verbatim')) #seems unnecessary
        table.addrow(("Sys Release", str(release)), (None,'Verbatim'))
        table.addrow(("Sys Version", str(version)), (None,'Verbatim'))
        table.addrow(("Machine", str(machine)), (None,'Verbatim'))
        table.addrow(("Processor", str(processor)), (None,'Verbatim'))
    
        table.finish()
        return table


class ExampleTable(WorkspaceTable):
    def __init__(self, ws):
        """A table showing how to use table features."""
        super(ExampleTable,self).__init__(ws, self._create)

    def _create(self):
        colHeadings = ["Hover over me...","And me!","Click the pig"]
        tooltips = ["This tooltip can give more information about what this column displays",
                    "Unfortunately, we can't show nicely formatted math in these tooltips (yet)",
                    "Click on the pyGSTi logo below to create the non-automatically-generated plot;" +
                    " then hover over the colored boxes."]
        example_mx = _np.array( [[ 1.0,  1/3, -1/3, -1.0],
                                 [ 1/3,  1.0,  0.0, -1/5],
                                 [-1/3,  0.0,  1.0,  1/6],
                                 [-1.0, -1/5,  1/6,  1.0]] )
        example_ebmx = _np.abs(example_mx) * 0.05
        example_fig =  _wp.GateMatrixPlot(self.ws, example_mx, -1.0,1.0,
                                          "pp", EBmatrix=example_ebmx)
        
        table = _ReportTable(colHeadings, None, colHeadingLabels=tooltips)
        table.addrow(("Pi",_np.pi, example_fig), ('Normal','Normal','Figure'))
        table.finish()
        return table

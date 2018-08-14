""" Classes corresponding to tables within a Workspace context."""
from __future__ import division, print_function, absolute_import, unicode_literals
#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************

import warnings           as _warnings
import numpy              as _np

from .. import construction as _cnst
from .. import tools      as _tools
from .. import objects    as _objs
from . import reportables as _reportables
from .reportables import evaluate as _ev

from .table import ReportTable as _ReportTable

from .workspace import WorkspaceTable
from . import workspaceplots as _wp
from . import plothelpers as _ph

class BlankTable(WorkspaceTable):
    """A completely blank placeholder table."""
    def __init__(self, ws):
        """A completely blank placeholder table."""
        super(BlankTable,self).__init__(ws, self._create)

    def _create(self):
        table = _ReportTable(['Blank'], [None])
        table.finish()
        return table

class SpamTable(WorkspaceTable):
    """ A table of one or more gateset's SPAM elements. """
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

        rhoLabels = list(gatesets[0].preps.keys()) #use labels of 1st gateset
        povmLabels = list(gatesets[0].povms.keys()) #use labels of 1st gateset

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
            basisNm    = _tools.basis_longname(gateset.basis.name)
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


        for povmlbl in povmLabels:
            for lbl in gatesets[0].povms[povmlbl].keys():
                povmAndELbl = str(povmlbl) + ":" + lbl # format for GateSetFunction objs
                rowData = [lbl] if (len(povmLabels) == 1) else [povmAndELbl] #show POVM name if there's more than one of them
                rowFormatters = ['Effect']

                for gateset in gatesets:
                    EMx = _ev(_reportables.Vec_as_stdmx(gateset, povmAndELbl, "effect"))
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
                        raise ValueError("Invalid 'display_as' argument: %s" % display_as) # pragma: no cover

                for gateset in gatesets:
                    cri = confidenceRegionInfo if confidenceRegionInfo and \
                          (confidenceRegionInfo.gateset.frobeniusdist(gateset) < 1e-6) else None
                    evals = _ev(_reportables.Vec_as_stdmx_eigenvalues(gateset, povmAndELbl, "effect"),
                                cri)
                    rowData.append( evals )
                    rowFormatters.append('Brackets')

                if includeHSVec:
                    rowData.append( gatesets[-1].povms[povmlbl][lbl] )
                    rowFormatters.append('Normal')

                    if confidenceRegionInfo is not None:
                        intervalVec = confidenceRegionInfo.get_profile_likelihood_confidence_intervals(povmlbl)[:,None] #for all povm params
                        intervalVec = intervalVec[gatesets[-1].povms[povmlbl][lbl].gpindices] #specific to this effect
                        rowData.append( intervalVec ); rowFormatters.append('Normal')

                #Note: no dependence on confidence region (yet) when HS vector is not shown...
                table.addrow(rowData, rowFormatters)

        table.finish()
        return table



class SpamParametersTable(WorkspaceTable):
    """ A table for "SPAM parameters" (dot products of SPAM vectors)"""
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

        if len(gatesets[0].povms) == 1:
            povmKey = list(gatesets[0].povms.keys())[0]
            effectLbls = [ eLbl for eLbl in gatesets[0].povms[povmKey] ]
        else:
            effectLbls = [ povmLbl + "." + eLbl
                           for povmLbl,povm in gatesets[0].povms.items()
                           for eLbl in povm.keys() ]

        colHeadings = [''] + effectLbls
        formatters  = [None] + [ 'Effect' ]*len(effectLbls)

        table       = _ReportTable(colHeadings, formatters, confidenceRegionInfo=confidenceRegionInfo)

        for gstitle, gateset in zip(titles,gatesets):
            cri = confidenceRegionInfo if (confidenceRegionInfo and
                                           confidenceRegionInfo.gateset.frobeniusdist(gateset) < 1e-6) else None
            spamDotProdsQty = _ev( _reportables.Spam_dotprods(gateset), cri)
            DPs, DPEBs      = spamDotProdsQty.get_value_and_err_bar()
            assert(DPs.shape[1] == len(effectLbls)), \
                "Gatesets must have the same number of POVMs & effects"

            formatters      = [ 'Rho' ] + [ 'Normal' ]*len(effectLbls) #for rows below

            for ii,prepLabel in enumerate(gateset.preps.keys()): # ii enumerates rhoLabels to index DPs
                prefix = gstitle + " " if len(gstitle) else ""
                rowData = [prefix + str(prepLabel)]
                for jj,_ in enumerate(effectLbls): # jj enumerates eLabels to index DPs
                    if cri is None:
                        rowData.append((DPs[ii,jj],None))
                    else:
                        rowData.append((DPs[ii,jj],DPEBs[ii,jj]))
                table.addrow(rowData, formatters)

        table.finish()
        return table


class GatesTable(WorkspaceTable):
    """ Create a table showing a gateset's raw gates. """
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
            basisLongNm = _tools.basis_longname(gateset.basis.name)
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
                    assert(False) # pragma: no cover

            table.addrow(row_data, row_formatters)

        table.finish()
        return table


class ChoiTable(WorkspaceTable):
    """A table of the Choi representations of a GateSet's gates"""
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

        display : tuple/list of {"matrices","eigenvalues","barplot","boxplot"}
            Which columns to display: the Choi matrices (as numerical grids),
            the Choi matrix eigenvalues (as a numerical list), the eigenvalues
            on a bar plot, and/or the matrix as a plot of colored boxes.


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
            if 'matrix' in display or 'boxplot' in display:
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
                    basisLongNm = _tools.basis_longname(gateset.basis.name)
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
            elif disp == "boxplot":
                for gateset,title in zip(gatesets,titles):
                    basisLongNm = _tools.basis_longname(gateset.basis.name)
                    pre = (title+' ' if title else '')
                    colHeadings.append('%sChoi matrix (%s basis)' % (pre,basisLongNm))
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
                    for gateset, (_, evals) in zip(gatesets,qtysList):
                        evs, evsEB = evals[i].get_value_and_err_bar()
                        fig = _wp.ChoiEigenvalueBarPlot(self.ws, evs, evsEB)
                        row_data.append(fig)
                        row_formatters.append('Figure')

                elif disp == "boxplot":
                    for gateset, (choiMxs, _) in zip(gatesets, qtysList):
                        choiMx_real = choiMxs[i].hermitian_to_real()
                        choiMx, EB = choiMx_real.get_value_and_err_bar()
                        fig = _wp.GateMatrixPlot(self.ws, choiMx,
                                                 colorbar=False,
                                                 mxBasis=gateset.basis,
                                                 EBmatrix=EB)
                        row_data.append( fig )
                        row_formatters.append('Figure')

            table.addrow(row_data, row_formatters)
        table.finish()
        return table


class GatesetVsTargetTable(WorkspaceTable):
    """ Table comparing a GateSet (as a whole) to a target """
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
            table.addrow(("Predicted Clifford RB number", RBnum), (None, 'Normal') )

        table.finish()
        return table


class GatesVsTargetTable(WorkspaceTable):
    """ Table comparing a GateSet's gates to those of a target gate set """
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
            try:
                heading, tooltip = _reportables.info_of_gatefn_by_name(disp)
            except ValueError:
                raise ValueError("Invalid display column name: %s" % disp)
            colHeadings.append(heading)
            tooltips.append(tooltip)

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

            for disp in display:
                #import time as _time #DEBUG
                #tStart = _time.time() #DEBUG
                qty = _reportables.evaluate_gatefn_by_name(
                    disp, gateset, targetGateset, gl, confidenceRegionInfo)
                #tm = _time.time()-tStart #DEBUG
                #if tm > 0.01: print("DB: Evaluated %s in %gs" % (disp, tm)) #DEBUG
                row_data.append( qty )

            table.addrow(row_data, formatters)
        table.finish()
        return table


class SpamVsTargetTable(WorkspaceTable):
    """ Table comparing a GateSet's SPAM vectors to those of a target """
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

        prepLabels = list(gateset.preps.keys())
        povmLabels = list(gateset.povms.keys())

        colHeadings  = ('Prep/POVM', "Infidelity", "1/2 Trace|Distance", "1/2 Diamond-Dist")
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
        for rowData in zip(prepLabels, prepInfidelities, prepTraceDists,
                           prepDiamondDists):
            table.addrow(rowData, formatters)


        formatters = [ 'Normal' ] + [ 'Normal' ] * (len(colHeadings) - 1)
        povmInfidelities = [_ev(_reportables.POVM_entanglement_infidelity(
                             gateset, targetGateset, l), confidenceRegionInfo)
                            for l in povmLabels]
        povmTraceDists   = [_ev(_reportables.POVM_jt_diff(
                             gateset, targetGateset, l), confidenceRegionInfo)
                            for l in povmLabels]
        povmDiamondDists = [_ev(_reportables.POVM_half_diamond_norm(
                             gateset, targetGateset, l), confidenceRegionInfo)
                            for l in povmLabels]

        for rowData in zip(povmLabels, povmInfidelities, povmTraceDists,
                           povmDiamondDists):
            table.addrow(rowData, formatters)


        table.finish()
        return table



class ErrgenTable(WorkspaceTable):
    """ Table displaying the error generators of a GateSet's gates as well
        as their projections onto spaces of standard generators """
    def __init__(self, ws, gateset, targetGateset, confidenceRegionInfo=None,
                 display=("errgen","H","S","A"), display_as="boxes",
                 genType="logGTi"):

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

        genType : {"logG-logT", "logTiG", "logGTi"}
            The type of error generator to compute.  Allowed values are:

            - "logG-logT" : errgen = log(gate) - log(target_gate)
            - "logTiG" : errgen = log( dot(inv(target_gate), gate) )
            - "logTiG" : errgen = log( dot(gate, inv(target_gate)) )

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
            """return a [min,max] already in list if there's one within an
               order of magnitude"""
            M = max(M, ABS_THRESHOLD)
            for mx in max_lst:
                if (abs(M) >= 1e-6 and 0.9999 < mx/M < 10) or (abs(mx)<1e-6 and abs(M)<1e-6):
                    return -mx,mx
            return None

        ABS_THRESHOLD = 1e-6 #don't let color scales run from 0 to 0: at least this much!
        def addMax(max_lst, M):
            """add `M` to a list of maximas if it's different enough from
               existing elements"""
            M = max(M, ABS_THRESHOLD)
            if not getMinMax(max_lst,M):
                max_lst.append(M)

        #Do computation, so shared color scales can be computed
        for gl in gateLabels:
            if genType == "logG-logT":
                info = _ev(_reportables.LogGmlogT_and_projections(
                    gateset, targetGateset, gl), confidenceRegionInfo)
            elif genType == "logTiG":
                info = _ev(_reportables.LogTiG_and_projections(
                    gateset, targetGateset, gl), confidenceRegionInfo)
            elif genType == "logGTi":
                info = _ev(_reportables.LogGTi_and_projections(
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


class GaugeRobustErrgenTable(WorkspaceTable):
    """ Table displaying the first-order gauge invariant ("gauge robust")
        linear combinations of standard error generator coefficients for
        the gates in a gate set.
    """
    def __init__(self, ws, gateset, targetGateset, confidenceRegionInfo=None,
                 genType="logGTi"):

        """
        Create a table listing the first-order gauge invariant ("gauge robust")
        linear combinations of standard error generator coefficients for
        the gates in `gateset`.  This table identifies, through the use of
        "synthetic idle tomography", which combinations of standard-error-
        generator coefficients are robust (to first-order) to gauge variations.

        Parameters
        ----------
        gateset, targetGateset : GateSet
            The gate sets to compare

        confidenceRegionInfo : ConfidenceRegion, optional
            If not None, specifies a confidence-region
            used to display error intervals.

        genType : {"logG-logT", "logTiG", "logGTi"}
            The type of error generator to compute.  Allowed values are:

            - "logG-logT" : errgen = log(gate) - log(target_gate)
            - "logTiG" : errgen = log( dot(inv(target_gate), gate) )
            - "logTiG" : errgen = log( dot(gate, inv(target_gate)) )

        Returns
        -------
        ReportTable
        """
        super(GaugeRobustErrgenTable,self).__init__(ws, self._create, gateset,
                                                    targetGateset, confidenceRegionInfo,
                                                    genType)

    def _create(self, gateset, targetGateset, confidenceRegionInfo, genType):

        gateLabels  = list(gateset.gates.keys())  # gate labels
        colHeadings = ['Error rates', 'Value']

        table = _ReportTable(colHeadings, (None,)*len(colHeadings),
                             confidenceRegionInfo=confidenceRegionInfo)

        assert(genType == "logGTi"), "Only `genType == \"logGTI\"` is supported when `gaugeRobust` is True"
        syntheticIdleStrs = []

        ## Construct synthetic idles
        maxPower = 4; maxLen = 6; Id = _np.identity(targetGateset.dim,'d')
        baseStrs = _cnst.list_all_gatestrings_without_powers_and_cycles(list(gateset.gates.keys()), maxLen)
        for s in baseStrs:
            for i in range(1,maxPower):
                if len(s**i) > 1 and _np.linalg.norm(targetGateset.product( s**i ) - Id) < 1e-6:
                    syntheticIdleStrs.append( s**i ); break
        #syntheticIdleStrs = _cnst.gatestring_list([ ('Gx',)*4, ('Gy',)*4 ] ) #DEBUG!!!
        #syntheticIdleStrs = _cnst.gatestring_list([ ('Gx',)*4, ('Gy',)*4, ('Gy','Gx','Gx')*2] ) #DEBUG!!!
        print("Using synthetic idles: \n",'\n'.join([str(gstr) for gstr in syntheticIdleStrs]))

        gaugeRobust_info = _ev(_reportables.Robust_LogGTi_and_projections(
            gateset, targetGateset, syntheticIdleStrs), confidenceRegionInfo)

        for linear_combo_lbl, val in gaugeRobust_info.items():
            row_data = [linear_combo_lbl, val]
            row_formatters = [None, 'Normal']
            table.addrow(row_data, row_formatters)

        table.finish()
        return table



class old_RotationAxisVsTargetTable(WorkspaceTable):
    """ Old 1-qubit-only gate rotation axis table """
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
        super(old_RotationAxisVsTargetTable,self).__init__(
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


class GateDecompTable(WorkspaceTable):
    """ Table of angle & axis decompositions of a GateSet's gates """
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
            gl = str(gl) # Label -> str for decomp-dict keys
            axis, axisEB = decomp[gl + ' axis'].get_value_and_err_bar()
            axisFig = _wp.ProjectionsBoxPlot(self.ws, axis, gateset.basis.name, -1.0,1.0,
                                             boxLabels=True, EBmatrix=axisEB)
            decomp[gl + ' hamiltonian eigenvalues'].scale( 1.0/_np.pi ) #scale evals to units of pi
            rowData = [gl, decomp[gl + ' hamiltonian eigenvalues'],
                       decomp[gl + ' angle'], axisFig,
                       decomp[gl + ' log inexactness'] ]

            for gl_other in gateLabels:
                gl_other = str(gl_other)
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
    """ 1-qubit-only table of gate decompositions """
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


class old_RotationAxisTable(WorkspaceTable):
    """ 1-qubit-only table of gate rotation angles and axes """
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
        super(old_RotationAxisTable,self).__init__(ws, self._create, gateset, confidenceRegionInfo, showAxisAngleErrBars)


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
        latex_head += " & & %s \\\\ \hline\n" % (" & ".join(map(str,gateLabels)))
    
        table = _ReportTable(colHeadings, formatters,
                             customHeader={'latex': latex_head}, confidenceRegionInfo=confidenceRegionInfo)

        formatters = [None, 'Pi'] + ['Pi'] * len(gateLabels)

        rotnAxisAnglesQty = _ev(_reportables.Angles_btwn_rotn_axes(gateset),
                                confidenceRegionInfo)
        rotnAxisAngles, rotnAxisAnglesEB = rotnAxisAnglesQty.get_value_and_err_bar()

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


class GateEigenvalueTable(WorkspaceTable):
    """ Table displaying, in a variety of ways, the eigenvalues of a
        GateSet's gates """
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
            - "evdm" : the gauge-invariant "eigenvalue diamond norm" metric
            - "evinf" : the gauge-invariant "eigenvalue infidelity" metric

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
        formatters = [None]
        for disp in display:
            if disp == "evals":
                colHeadings.append('Eigenvalues ($E$)')
                formatters.append(None)

            elif disp == "target":
                colHeadings.append('Target Evals. ($T$)')
                formatters.append(None)

            elif disp == "rel":
                if(targetGateset is not None): #silently ignore
                    colHeadings.append('Rel. Evals ($R$)')
                    formatters.append(None)

            elif disp == "log-evals":
                colHeadings.append('Re log(E)')
                colHeadings.append('Im log(E)')
                formatters.append('MathText')
                formatters.append('MathText')

            elif disp == "log-rel":
                colHeadings.append('Re log(R)')
                colHeadings.append('Im log(R)')
                formatters.append('MathText')
                formatters.append('MathText')

            elif disp == "polar":
                colHeadings.append('Eigenvalues') #Note: make sure header is *distinct* for pandas conversion
                formatters.append(None)

            elif disp == "relpolar":
                if(targetGateset is not None): #silently ignore
                    colHeadings.append('Rel. Evals')  #Note: make sure header is *distinct* for pandas conversion
                    formatters.append(None)

            elif disp == "absdiff-evals":
                if(targetGateset is not None): #silently ignore
                    colHeadings.append('|E - T|')
                    formatters.append('MathText')

            elif disp == "infdiff-evals":
                if(targetGateset is not None): #silently ignore
                    colHeadings.append('1.0 - Re(\\bar{T}*E)')
                    formatters.append('MathText')

            elif disp == "absdiff-log-evals":
                if(targetGateset is not None): #silently ignore
                    colHeadings.append('|Re(log E) - Re(log T)|')
                    colHeadings.append('|Im(log E) - Im(log T)|')
                    formatters.append('MathText')
                    formatters.append('MathText')

            elif disp == "evdm":
                if(targetGateset is not None): #silently ignore
                    colHeadings.append('Eigenvalue Diamond norm')
                    formatters.append('Conversion')

            elif disp == "evinf":
                if(targetGateset is not None): #silently ignore
                    colHeadings.append('Eigenvalue infidelity')
                    formatters.append(None)
            else:
                raise ValueError("Invalid display element: %s" % disp)

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

                elif disp == "evdm":
                    if targetGateset is not None:
                        fn = _reportables.Eigenvalue_diamondnorm if _tools.isstr(gl) else \
                             _reportables.Gatestring_eigenvalue_diamondnorm
                        gidm = _ev(fn(gateset, targetGateset, gl), confidenceRegionInfo)
                        row_data.append( gidm )
                        row_formatters.append('Normal')

                elif disp == "evinf":
                    if targetGateset is not None:
                        fn = _reportables.Eigenvalue_entanglement_infidelity if _tools.isstr(gl) else \
                             _reportables.Gatestring_eigenvalue_entanglement_infidelity
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



class DataSetOverviewTable(WorkspaceTable):
    """ Table giving a summary of the properties of `dataset`. """
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

        minN = round(min([ row.total for row in dataset.values()]))
        maxN = round(max([ row.total for row in dataset.values()]))
        cntStr = "[%d,%d]" % (minN,maxN) if (minN != maxN) else "%d" % round(minN)

        table.addrow(("Number of strings", str(len(dataset))), (None,None))
        table.addrow(("Gate labels", ", ".join([str(gl) for gl in dataset.get_gate_labels()]) ), (None,None))
        table.addrow(("Outcome labels",  ", ".join(map(str,dataset.get_outcome_labels())) ), (None,None))
        table.addrow(("Counts per string", cntStr  ), (None,None))

        if maxLengthList is not None:
            table.addrow(("Max. Lengths", ", ".join(map(str,maxLengthList)) ), (None,None))
        if hasattr(dataset,'comment') and dataset.comment is not None:
            commentLines = dataset.comment.split('\n')
            for i,commentLine in enumerate(commentLines,start=1):
                table.addrow(("User comment %d" % i, commentLine  ), (None,'Verbatim'))

        table.finish()
        return table


class FitComparisonTable(WorkspaceTable):
    """ Table showing how the goodness-of-fit evolved over GST iterations """
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
            try:
                NpByX = [ gs.num_nongauge_params() for gs in gatesetByX ]
            except _np.linalg.LinAlgError:
                _warnings.warn(("LinAlgError when trying to compute the number"
                                " of non-gauge parameters.  Using total"
                                " parameters instead."))
                NpByX = [ gs.num_params() for gs in gatesetByX ]

        tooltips = ('', 'Difference in logL', 'number of degrees of freedom',
                    'difference between observed logl and expected mean',
                    'std deviation', 'number of std deviation', 'dataset dof',
                    'number of gateset parameters', '1-5 star rating (like Netflix)')
        table = _ReportTable(colHeadings, None, colHeadingLabels=tooltips)

        for X,gs,gss,Np in zip(Xs,gatesetByX,gssByX,NpByX):
            Nsig, rating, fitQty, k, Ns, Np = _ph.ratedNsigma(dataset, gs, gss,
                                                              objective, Np, returnAll=True)
            table.addrow((str(X),fitQty,k,fitQty-k,_np.sqrt(2*k),Nsig,Ns,Np,"<STAR>"*rating),
                         (None,'Normal','Normal','Normal','Normal','Rounded','Normal','Normal','Conversion'))

        table.finish()
        return table


class GatestringTable(WorkspaceTable):
    """ Table which simply displays list(s) of gate strings """
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


class GatesSingleMetricTable(WorkspaceTable):
    """ Table that compares the gates of many GateSets which share the same gate
        labels to target GateSets using a single metric, so that the GateSet
        titles can be used as the row and column headers."""
    def __init__(self, ws, metric, gatesets, targetGatesets, titles,
                 rowtitles=None, tableTitle=None, gateLabel=None,
                 confidenceRegionInfo=None):
        """
        Create a table comparing the gates of various gate sets (`gatesets`) to
        those of `targetGatesets` using the metric named by `metric`.

        If `gatesets` and `targetGatesets` are 1D lists, then `rowtitles` and
        `gateLabel` should be left as their default values so that the
        gate labels are used as row headers.

        If `gatesets` and `targetGatesets` are 2D (nested) lists, then
        `rowtitles` should specify the row-titles corresponding to the outer list
        elements and `gateLabel` should specify a single gate label that names
        the gate being compared throughout the entire table.

        Parameters
        ----------
        metric : str
            The abbreviation for the metric to use.  Allowed values are:

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

        gatesets : list
            A list or nested list-of-lists of gate sets to compare with
            corresponding elements of `targetGatesets`.

        targetGatesets : list
            A list or nested list-of-lists of gate sets to compare with
            corresponding elements of `gatesets`.

        titles : list of strs
            A list of column titles used to describe elements of the
            innermost list(s) in `gatesets`.

        rowtitles : list of strs, optional
            A list of row titles used to describe elements of the
            outer list in `gatesets`.  If None, then the gate labels
            are used.

        tableTitle : str, optional
            If not None, text to place in a top header cell which spans all the
            columns of the table.

        gateLabel : str, optional
            If not None, the single gate label to use for all comparisons
            computed in this table.  This should be set when (and only when)
            `gatesets` and `targetGatesets` are 2D (nested) lists.

        confidenceRegionInfo : ConfidenceRegion, optional
            If not None, specifies a confidence-region
            used to display error intervals.

        Returns
        -------
        ReportTable
        """
        super(GatesSingleMetricTable,self).__init__(
            ws, self._create, metric, gatesets, targetGatesets, titles,
            rowtitles, tableTitle, gateLabel, confidenceRegionInfo)

    def _create(self, metric, gatesets, targetGatesets, titles,
                rowtitles, tableTitle, gateLabel, confidenceRegionInfo):

        if rowtitles is None:
            assert(gateLabel is None), "`gateLabel` must be None when `rowtitles` is"
            colHeadings = ("Gate",) + tuple(titles)
        else:
            colHeadings = ("",) + tuple(titles)

        nCols = len(colHeadings)
        formatters = [None]*nCols #[None] + ['GatesetType']*(nCols-1)

        #latex_head =  "\\begin{tabular}[l]{%s}\n\hline\n" % ("|c" * nCols + "|")
        #latex_head += "\\multirow{2}{*}{Gate} & " + \
        #              "\\multicolumn{%d}{c|}{%s} \\\\ \cline{2-%d}\n" % (len(titles),niceNm,nCols)
        #latex_head += " & " + " & ".join([mknice(t) for t in titles]) + "\\\\ \hline\n"
        #
        #html_head = '<table class="%(tableclass)s" id="%(tableid)s" ><thead>'
        #html_head += '<tr><th rowspan="2"></th>' + \
        #             '<th colspan="%d">%s</th></tr>\n' % (len(titles),niceNm)
        #html_head += "<tr><th>" +  " </th><th> ".join([mknice(t) for t in titles]) + "</th></tr>\n"
        #html_head += "</thead><tbody>"

        if tableTitle:
            latex_head =  "\\begin{tabular}[l]{%s}\n\hline\n" % ("|c" * nCols + "|")
            latex_head += "\\multicolumn{%d}{c|}{%s} \\\\ \cline{1-%d}\n" % (nCols,tableTitle,nCols)
            latex_head += " & ".join(colHeadings) + "\\\\ \hline\n"

            html_head = '<table class="%(tableclass)s" id="%(tableid)s" ><thead>'
            html_head += '<tr><th colspan="%d">%s</th></tr>\n' % (nCols,tableTitle)
            html_head += "<tr><th>" +  " </th><th> ".join(colHeadings) + "</th></tr>\n"
            html_head += "</thead><tbody>"

            table = _ReportTable(colHeadings, formatters,
                                 customHeader={'latex': latex_head,
                                               'html': html_head} )
        else:
            table = _ReportTable(colHeadings, formatters)

        row_formatters = [None] + ['Normal']*len(titles)

        if rowtitles is None:
            for gl in targetGatesets[0].gates: # use first target's gate labels
                row_data = [gl]
                for gs,gsTarget in zip(gatesets, targetGatesets):
                    if gs is None or gsTarget is None:
                        qty = _objs.reportableqty.ReportableQty(_np.nan)
                    else:
                        qty = _reportables.evaluate_gatefn_by_name(
                            metric, gs, gsTarget, gl, confidenceRegionInfo)
                    row_data.append( qty )
                table.addrow(row_data, row_formatters)
        else:
            for rowtitle,gsList,tgsList in zip(rowtitles,gatesets,targetGatesets):
                row_data = [rowtitle]
                for gs,gsTarget in zip(gsList, tgsList):
                    if gs is None or gsTarget is None:
                        qty = _objs.reportableqty.ReportableQty(_np.nan)
                    else:
                        qty = _reportables.evaluate_gatefn_by_name(
                            metric, gs, gsTarget, gateLabel, confidenceRegionInfo)
                    row_data.append( qty )
                table.addrow(row_data, row_formatters)

        table.finish()
        return table


class StandardErrgenTable(WorkspaceTable):
    """ A table showing what the standard error generators' superoperator
        matrices look like."""
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
        elif nQubits == 1:
            yd,xd = 1,2 # y and x pauli-prod *basis* dimensions
            xlabel = "Q1"; ylabel = ""
        elif nQubits == 2:
            yd,xd = 2,2
            xlabel = "Q2"; ylabel="Q1"
        else:
            assert(d%2 == 0)
            yd,xd = 2,d//2
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

            for xlabel in xLabels:
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



class GaugeOptParamsTable(WorkspaceTable):
    """ Table of gauge optimization parameters """
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



class MetadataTable(WorkspaceTable):
    """ Table of raw parameters, often taken directly from a `Results` object"""
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
            if key in ['L,germ tuple base string dict', 'weights', 'profiler']: continue #skip these
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
            elif isinstance(vec, _objs.ComplementSPAMVec): paramTyp = "Comp"
            else: paramTyp = "unknown" # pragma: no cover
            table.addrow((lbl + " parameterization", paramTyp), (None,'Verbatim'))

        for povmlbl, povm in gateset.povms.items():
            if isinstance(povm, _objs.UnconstrainedPOVM): paramTyp = "unconstrained"
            elif isinstance(povm, _objs.TPPOVM): paramTyp = "TP"
            elif isinstance(povm, _objs.TensorProdPOVM): paramTyp = "TensorProd"
            else: paramTyp = "unknown" # pragma: no cover
            table.addrow((povmlbl + " parameterization", paramTyp), (None,'Verbatim'))

            for lbl,vec in povm.items():
                if isinstance(vec, _objs.StaticSPAMVec): paramTyp = "static"
                elif isinstance(vec, _objs.FullyParameterizedSPAMVec): paramTyp = "full"
                elif isinstance(vec, _objs.TPParameterizedSPAMVec): paramTyp = "TP"
                elif isinstance(vec, _objs.ComplementSPAMVec): paramTyp = "Comp"
                else: paramTyp = "unknown" # pragma: no cover
                table.addrow(("> " + lbl + " parameterization", paramTyp), (None,'Verbatim'))

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
            else: paramTyp = "unknown" # pragma: no cover
            table.addrow((gl + " parameterization", paramTyp), (None,'Verbatim'))


        table.finish()
        return table


class SoftwareEnvTable(WorkspaceTable):
    """ Table showing details about the current software environment """
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
            """ Extract the current version of a python module """
            if moduleName == "cvxopt":
                #special case b/c cvxopt can be weird...
                try:
                    mod = __import__("cvxopt.info")
                    return str(mod.info.version)
                except Exception: pass #try the normal way below

            try:
                mod = __import__(moduleName)
                return str(mod.__version__)
            except ImportError:     # pragma: no cover
                return "missing"    # pragma: no cover
            except AttributeError:  # pragma: no cover
                return "ver?"       # pragma: no cover
            except Exception:       # pragma: no cover
                return "???"        # pragma: no cover

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
        (system, _, release, version, machine, processor) = platform.uname()
        table.addrow(("Platform summary", str(platform.platform())), (None,'Verbatim'))
        table.addrow(("System", str(system)), (None,'Verbatim'))
        table.addrow(("Sys Release", str(release)), (None,'Verbatim'))
        table.addrow(("Sys Version", str(version)), (None,'Verbatim'))
        table.addrow(("Machine", str(machine)), (None,'Verbatim'))
        table.addrow(("Processor", str(processor)), (None,'Verbatim'))

        table.finish()
        return table


class ProfilerTable(WorkspaceTable):
    """ Table of profiler timing information """
    def __init__(self, ws, profiler, sortBy="time"):
        """
        Create a table of profiler timing information.

        Parameters
        ----------
        profiler : Profiler
            The profiler object to extract timings from.

        sortBy : {"time", "name"}
            What the timer values should be sorted by.
        """
        super(ProfilerTable,self).__init__(ws, self._create, profiler, sortBy)

    def _create(self, profiler, sortBy):

        colHeadings = ('Label','Time (sec)')
        formatters = ('Bold','Bold')

        #custom latex header for maximum width imposed on 2nd col
        latex_head =  "\\begin{tabular}[l]{|c|p{3in}|}\n\hline\n"
        latex_head += "\\textbf{Label} & \\textbf{Time} (sec) \\\\ \hline\n"
        table = _ReportTable(colHeadings, formatters,
                             customHeader={'latex': latex_head} )

        if profiler is not None:
            if sortBy == "name":
                timerNames = sorted(list(profiler.timers.keys()))
            elif sortBy == "time":
                timerNames = sorted(list(profiler.timers.keys()),
                                    key=lambda x: -profiler.timers[x])
            else:
                raise ValueError("Invalid 'sortBy' argument: %s" % sortBy)

            for nm in timerNames:
                table.addrow((nm, profiler.timers[nm]), (None,None))

        table.finish()
        return table



class ExampleTable(WorkspaceTable):
    """ Table used just as an example of what tables can do/look like for use
        within the "Help" section of reports. """
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

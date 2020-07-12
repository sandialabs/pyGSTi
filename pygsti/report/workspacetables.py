"""
Classes corresponding to tables within a Workspace context.
"""
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

import warnings as _warnings
import numpy as _np
import scipy.sparse as _sps

from .. import construction as _cnst
from .. import tools as _tools
from .. import objects as _objs
from . import reportables as _reportables
from .reportables import evaluate as _ev
from ..objects.label import Label as _Lbl
from ..objects.basis import DirectSumBasis as _DirectSumBasis
from ..objects import objectivefns as _objfns
from ..objects import MatrixForwardSimulator as _MatrixFSim
from ..algorithms import gaugeopt as _gopt

from .table import ReportTable as _ReportTable

from .workspace import WorkspaceTable
from . import workspaceplots as _wp
from . import plothelpers as _ph


class BlankTable(WorkspaceTable):
    """
    A completely blank placeholder table.

    Parameters
    ----------
    ws : Workspace
        The containing (parent) workspace.
    """

    def __init__(self, ws):
        """A completely blank placeholder table."""
        super(BlankTable, self).__init__(ws, self._create)

    def _create(self):
        table = _ReportTable(['Blank'], [None])
        table.finish()
        return table


class SpamTable(WorkspaceTable):
    """
    A table of one or more model's SPAM elements.

    Parameters
    ----------
    ws : Workspace
        The containing (parent) workspace.

    models : Model or list of Models
        The Model(s) whose SPAM elements should be displayed. If
        multiple Models are given, they should have the same SPAM
        elements..

    titles : list of strs, optional
        Titles correponding to elements of `models`, e.g. `"Target"`.

    display_as : {"numbers", "boxes"}, optional
        How to display the SPAM matrices, as either numerical
        grids (fine for small matrices) or as a plot of colored
        boxes (space-conserving and better for large matrices).

    confidence_region_info : ConfidenceRegion, optional
        If not None, specifies a confidence-region
        used to display error intervals.

    include_hs_vec : boolean, optional
        Whether or not to include Hilbert-Schmidt
        vector representation columns in the table.
    """

    def __init__(self, ws, models, titles=None,
                 display_as="boxes", confidence_region_info=None,
                 include_hs_vec=True):
        """
        A table of one or more model's SPAM elements.

        Parameters
        ----------
        models : Model or list of Models
            The Model(s) whose SPAM elements should be displayed. If
            multiple Models are given, they should have the same SPAM
            elements..

        titles : list of strs, optional
            Titles correponding to elements of `models`, e.g. `"Target"`.

        display_as : {"numbers", "boxes"}, optional
            How to display the SPAM matrices, as either numerical
            grids (fine for small matrices) or as a plot of colored
            boxes (space-conserving and better for large matrices).

        confidence_region_info : ConfidenceRegion, optional
            If not None, specifies a confidence-region
            used to display error intervals.

        include_hs_vec : boolean, optional
            Whether or not to include Hilbert-Schmidt
            vector representation columns in the table.
        """
        super(SpamTable, self).__init__(ws, self._create, models,
                                        titles, display_as, confidence_region_info,
                                        include_hs_vec)

    def _create(self, models, titles, display_as, confidence_region_info,
                include_hs_vec):

        if isinstance(models, _objs.Model):
            models = [models]

        rhoLabels = list(models[0].preps.keys())  # use labels of 1st model
        povmLabels = list(models[0].povms.keys())  # use labels of 1st model

        if titles is None:
            titles = [''] * len(models)

        colHeadings = ['Operator']
        for model, title in zip(models, titles):
            colHeadings.append('%sMatrix' % (title + ' ' if title else ''))
        for model, title in zip(models, titles):
            colHeadings.append('%sEigenvals' % (title + ' ' if title else ''))

        formatters = [None] * len(colHeadings)

        if include_hs_vec:
            model = models[-1]  # only show HSVec for last model
            basisNm = _tools.basis_longname(model.basis)
            colHeadings.append('Hilbert-Schmidt vector (%s basis)' % basisNm)
            formatters.append(None)

            if confidence_region_info is not None:
                colHeadings.append('%g%% C.I. half-width' % confidence_region_info.level)
                formatters.append('Conversion')

        table = _ReportTable(colHeadings, formatters, confidence_region_info=confidence_region_info)

        for lbl in rhoLabels:
            rowData = [lbl]; rowFormatters = ['Rho']

            for model in models:
                rhoMx = _ev(_reportables.Vec_as_stdmx(model, lbl, "prep"))
                # confidence_region_info) #don't put CIs on matrices for now
                if display_as == "numbers":
                    rowData.append(rhoMx)
                    rowFormatters.append('Brackets')
                elif display_as == "boxes":
                    rhoMx_real = rhoMx.hermitian_to_real()
                    v = rhoMx_real.value()
                    fig = _wp.GateMatrixPlot(self.ws, v, colorbar=False,
                                             box_labels=True, prec='compacthp',
                                             mx_basis=None)  # no basis labels
                    rowData.append(fig)
                    rowFormatters.append('Figure')
                else:
                    raise ValueError("Invalid 'display_as' argument: %s" % display_as)

            for model in models:
                cri = confidence_region_info if confidence_region_info and \
                    (confidence_region_info.model.frobeniusdist(model) < 1e-6) else None
                evals = _ev(_reportables.Vec_as_stdmx_eigenvalues(model, lbl, "prep"),
                            cri)
                rowData.append(evals)
                rowFormatters.append('Brackets')

            if include_hs_vec:
                rowData.append(models[-1].preps[lbl])
                rowFormatters.append('Normal')

                if confidence_region_info is not None:
                    intervalVec = confidence_region_info.retrieve_profile_likelihood_confidence_intervals(lbl)[:, None]
                    if intervalVec.shape[0] == models[-1].dim - 1:
                        #TP constrained, so pad with zero top row
                        intervalVec = _np.concatenate((_np.zeros((1, 1), 'd'), intervalVec), axis=0)
                    rowData.append(intervalVec); rowFormatters.append('Normal')

            #Note: no dependence on confidence region (yet) when HS vector is not shown...
            table.add_row(rowData, rowFormatters)

        for povmlbl in povmLabels:
            for lbl in models[0].povms[povmlbl].keys():
                povmAndELbl = str(povmlbl) + ":" + lbl  # format for ModelFunction objs
                # show POVM name if there's more than one of them
                rowData = [lbl] if (len(povmLabels) == 1) else [povmAndELbl]
                rowFormatters = ['Effect']

                for model in models:
                    EMx = _ev(_reportables.Vec_as_stdmx(model, povmAndELbl, "effect"))
                    #confidence_region_info) #don't put CIs on matrices for now
                    if display_as == "numbers":
                        rowData.append(EMx)
                        rowFormatters.append('Brackets')
                    elif display_as == "boxes":
                        EMx_real = EMx.hermitian_to_real()
                        v = EMx_real.value()
                        fig = _wp.GateMatrixPlot(self.ws, v, colorbar=False,
                                                 box_labels=True, prec='compacthp',
                                                 mx_basis=None)  # no basis labels
                        rowData.append(fig)
                        rowFormatters.append('Figure')
                    else:
                        raise ValueError("Invalid 'display_as' argument: %s" % display_as)  # pragma: no cover

                for model in models:
                    cri = confidence_region_info if confidence_region_info and \
                        (confidence_region_info.model.frobeniusdist(model) < 1e-6) else None
                    evals = _ev(_reportables.Vec_as_stdmx_eigenvalues(model, povmAndELbl, "effect"),
                                cri)
                    rowData.append(evals)
                    rowFormatters.append('Brackets')

                if include_hs_vec:
                    rowData.append(models[-1].povms[povmlbl][lbl])
                    rowFormatters.append('Normal')

                    if confidence_region_info is not None:
                        intervalVec = confidence_region_info.retrieve_profile_likelihood_confidence_intervals(povmlbl)[
                            :, None]  # for all povm params
                        intervalVec = intervalVec[models[-1].povms[povmlbl][lbl].gpindices]  # specific to this effect
                        rowData.append(intervalVec); rowFormatters.append('Normal')

                #Note: no dependence on confidence region (yet) when HS vector is not shown...
                table.add_row(rowData, rowFormatters)

        table.finish()
        return table


class SpamParametersTable(WorkspaceTable):
    """
    A table for "SPAM parameters" (dot products of SPAM vectors)

    Parameters
    ----------
    ws : Workspace
        The containing (parent) workspace.

    models : Model or list of Models
        The Model(s) whose SPAM parameters should be displayed. If
        multiple Models are given, they should have the same gates.

    titles : list of strs, optional
        Titles correponding to elements of `models`, e.g. `"Target"`.

    confidence_region_info : ConfidenceRegion, optional
        If not None, specifies a confidence-region
        used to display error intervals.
    """

    def __init__(self, ws, models, titles=None, confidence_region_info=None):
        """
        Create a table for model's "SPAM parameters", that is, the
        dot products of prep-vectors and effect-vectors.

        Parameters
        ----------
        models : Model or list of Models
            The Model(s) whose SPAM parameters should be displayed. If
            multiple Models are given, they should have the same gates.

        titles : list of strs, optional
            Titles correponding to elements of `models`, e.g. `"Target"`.

        confidence_region_info : ConfidenceRegion, optional
            If not None, specifies a confidence-region
            used to display error intervals.

        Returns
        -------
        ReportTable
        """
        super(SpamParametersTable, self).__init__(ws, self._create, models, titles, confidence_region_info)

    def _create(self, models, titles, confidence_region_info):

        if isinstance(models, _objs.Model):
            models = [models]
        if titles is None:
            titles = [''] * len(models)

        if len(models[0].povms) == 1:
            povmKey = list(models[0].povms.keys())[0]
            effectLbls = [eLbl for eLbl in models[0].povms[povmKey]]
        else:
            effectLbls = [povmLbl + "." + eLbl
                          for povmLbl, povm in models[0].povms.items()
                          for eLbl in povm.keys()]

        colHeadings = [''] + effectLbls
        formatters = [None] + ['Effect'] * len(effectLbls)

        table = _ReportTable(colHeadings, formatters, confidence_region_info=confidence_region_info)

        for gstitle, model in zip(titles, models):
            cri = confidence_region_info if (confidence_region_info
                                             and confidence_region_info.model.frobeniusdist(model) < 1e-6) else None
            spamDotProdsQty = _ev(_reportables.Spam_dotprods(model), cri)
            DPs, DPEBs = spamDotProdsQty.value_and_errorbar()
            assert(DPs.shape[1] == len(effectLbls)), \
                "Models must have the same number of POVMs & effects"

            formatters = ['Rho'] + ['Normal'] * len(effectLbls)  # for rows below

            for ii, prepLabel in enumerate(model.preps.keys()):  # ii enumerates rhoLabels to index DPs
                prefix = gstitle + " " if len(gstitle) else ""
                rowData = [prefix + str(prepLabel)]
                for jj, _ in enumerate(effectLbls):  # jj enumerates eLabels to index DPs
                    if cri is None:
                        rowData.append((DPs[ii, jj], None))
                    else:
                        rowData.append((DPs[ii, jj], DPEBs[ii, jj]))
                table.add_row(rowData, formatters)

        table.finish()
        return table


class GatesTable(WorkspaceTable):
    """
    Create a table showing a model's raw gates.

    Parameters
    ----------
    ws : Workspace
        The containing (parent) workspace.

    models : Model or list of Models
        The Model(s) whose gates should be displayed.  If multiple
        Models are given, they should have the same operation labels.

    titles : list of strings, optional
        A list of titles corresponding to the models, used to
        prefix the column(s) for that model. E.g. `"Target"`.

    display_as : {"numbers", "boxes"}, optional
        How to display the operation matrices, as either numerical
        grids (fine for small matrices) or as a plot of colored
        boxes (space-conserving and better for large matrices).

    confidence_region_info : ConfidenceRegion, optional
        If not None, specifies a confidence-region
        used to display error intervals for the *final*
        element of `models`.
    """

    def __init__(self, ws, models, titles=None, display_as="boxes",
                 confidence_region_info=None):
        """
        Create a table showing a model's raw gates.

        Parameters
        ----------
        models : Model or list of Models
            The Model(s) whose gates should be displayed.  If multiple
            Models are given, they should have the same operation labels.

        titles : list of strings, optional
            A list of titles corresponding to the models, used to
            prefix the column(s) for that model. E.g. `"Target"`.

        display_as : {"numbers", "boxes"}, optional
            How to display the operation matrices, as either numerical
            grids (fine for small matrices) or as a plot of colored
            boxes (space-conserving and better for large matrices).

        confidence_region_info : ConfidenceRegion, optional
            If not None, specifies a confidence-region
            used to display error intervals for the *final*
            element of `models`.

        Returns
        -------
        ReportTable
        """
        super(GatesTable, self).__init__(ws, self._create, models, titles,
                                         display_as, confidence_region_info)

    def _create(self, models, titles, display_as, confidence_region_info):

        if isinstance(models, _objs.Model):
            models = [models]

        opLabels = models[0].primitive_op_labels  # use labels of 1st model
        instLabels = list(models[0].instruments.keys())  # requires an explicit model!
        assert(isinstance(models[0], _objs.ExplicitOpModel)), "%s only works with explicit models" % str(type(self))

        if titles is None:
            titles = [''] * len(models)

        colHeadings = ['Gate']
        for model, title in zip(models, titles):
            basisLongNm = _tools.basis_longname(model.basis)
            pre = (title + ' ' if title else '')
            colHeadings.append('%sSuperoperator (%s basis)' % (pre, basisLongNm))
        formatters = [None] * len(colHeadings)

        if confidence_region_info is not None:
            #Only use confidence region for the *final* model.
            colHeadings.append('%g%% C.I. half-width' % confidence_region_info.level)
            formatters.append('Conversion')

        table = _ReportTable(colHeadings, formatters, confidence_region_info=confidence_region_info)

        #Create list of labels and gate-like objects, allowing instruments to be included:
        label_op_tups = []
        for gl in opLabels:
            # may want to gracefully handle index error here?
            tup_of_ops = tuple([model.operations[gl] for model in models])
            label_op_tups.append((gl, tup_of_ops))
        for il in instLabels:
            for comp_lbl in models[0].instruments[il].keys():
                tup_of_ops = tuple([model.instruments[il][comp_lbl] for model in models]
                                   )  # may want to gracefully handle index error here?
                label_op_tups.append((il + "." + comp_lbl, tup_of_ops))

        for lbl, per_model_ops in label_op_tups:
            row_data = [lbl]
            row_formatters = [None]

            for model, op in zip(models, per_model_ops):
                basis = model.basis

                if display_as == "numbers":
                    row_data.append(op)
                    row_formatters.append('Brackets')
                elif display_as == "boxes":
                    fig = _wp.GateMatrixPlot(self.ws, op.to_dense(),
                                             colorbar=False,
                                             mx_basis=basis)

                    row_data.append(fig)
                    row_formatters.append('Figure')
                else:
                    raise ValueError("Invalid 'display_as' argument: %s" % display_as)

            if confidence_region_info is not None:
                intervalVec = confidence_region_info.retrieve_profile_likelihood_confidence_intervals(
                    lbl)[:, None]  # TODO: won't work for instruments
                if isinstance(per_model_ops[-1], _objs.FullDenseOp):
                    #then we know how to reshape into a matrix
                    op_dim = models[-1].dim
                    basis = models[-1].basis
                    intervalMx = intervalVec.reshape(op_dim, op_dim)
                elif isinstance(per_model_ops[-1], _objs.TPDenseOp):
                    #then we know how to reshape into a matrix
                    op_dim = models[-1].dim
                    basis = models[-1].basis
                    intervalMx = _np.concatenate((_np.zeros((1, op_dim), 'd'),
                                                  intervalVec.reshape(op_dim - 1, op_dim)), axis=0)
                else:
                    # we don't know how best to reshape interval matrix for gate, so
                    # use derivative
                    op_dim = models[-1].dim
                    basis = models[-1].basis
                    op_deriv = per_model_ops[-1].deriv_wrt_params()
                    intervalMx = _np.abs(_np.dot(op_deriv, intervalVec).reshape(op_dim, op_dim))

                if display_as == "numbers":
                    row_data.append(intervalMx)
                    row_formatters.append('Brackets')

                elif display_as == "boxes":
                    maxAbsVal = _np.max(_np.abs(intervalMx))
                    fig = _wp.GateMatrixPlot(self.ws, intervalMx,
                                             color_min=-maxAbsVal, color_max=maxAbsVal,
                                             colorbar=False,
                                             mx_basis=basis)
                    row_data.append(fig)
                    row_formatters.append('Figure')
                else:
                    assert(False)  # pragma: no cover

            table.add_row(row_data, row_formatters)

        table.finish()
        return table


class ChoiTable(WorkspaceTable):
    """
    A table of the Choi representations of a Model's gates

    Parameters
    ----------
    ws : Workspace
        The containing (parent) workspace.

    models : Model or list of Models
        The Model(s) whose Choi info should be displayed.  If multiple
        Models are given, they should have the same operation labels.

    titles : list of strings, optional
        A list of titles corresponding to the models, used to
        prefix the column(s) for that model. E.g. `"Target"`.

    confidence_region_info : ConfidenceRegion, optional
        If not None, specifies a confidence-region
        used to display eigenvalue error intervals for the
        *final* Model in `models`.

    display : tuple/list of {"matrices","eigenvalues","barplot","boxplot"}
        Which columns to display: the Choi matrices (as numerical grids),
        the Choi matrix eigenvalues (as a numerical list), the eigenvalues
        on a bar plot, and/or the matrix as a plot of colored boxes.
    """

    def __init__(self, ws, models, titles=None,
                 confidence_region_info=None,
                 display=("matrix", "eigenvalues", "barplot")):
        """
        Create a table of the Choi matrices and/or their eigenvalues of
        a model's gates.

        Parameters
        ----------
        models : Model or list of Models
            The Model(s) whose Choi info should be displayed.  If multiple
            Models are given, they should have the same operation labels.

        titles : list of strings, optional
            A list of titles corresponding to the models, used to
            prefix the column(s) for that model. E.g. `"Target"`.

        confidence_region_info : ConfidenceRegion, optional
            If not None, specifies a confidence-region
            used to display eigenvalue error intervals for the
            *final* Model in `models`.

        display : tuple/list of {"matrices","eigenvalues","barplot","boxplot"}
            Which columns to display: the Choi matrices (as numerical grids),
            the Choi matrix eigenvalues (as a numerical list), the eigenvalues
            on a bar plot, and/or the matrix as a plot of colored boxes.

        Returns
        -------
        ReportTable
        """
        super(ChoiTable, self).__init__(ws, self._create, models, titles,
                                        confidence_region_info, display)

    def _create(self, models, titles, confidence_region_info, display):
        if isinstance(models, _objs.Model):
            models = [models]

        opLabels = models[0].primitive_op_labels  # use labels of 1st model
        assert(isinstance(models[0], _objs.ExplicitOpModel)), "%s only works with explicit models" % str(type(self))

        if titles is None:
            titles = [''] * len(models)

        qtysList = []
        for model in models:
            opLabels = model.primitive_op_labels  # operation labels
            #qtys_to_compute = []
            if 'matrix' in display or 'boxplot' in display:
                choiMxs = [_ev(_reportables.Choi_matrix(model, gl)) for gl in opLabels]
            else:
                choiMxs = None
            if 'eigenvalues' in display or 'barplot' in display:
                evals = [_ev(_reportables.Choi_evals(model, gl), confidence_region_info) for gl in opLabels]
            else:
                evals = None
            qtysList.append((choiMxs, evals))
        colHeadings = ['Gate']
        for disp in display:
            if disp == "matrix":
                for model, title in zip(models, titles):
                    basisLongNm = _tools.basis_longname(model.basis)
                    pre = (title + ' ' if title else '')
                    colHeadings.append('%sChoi matrix (%s basis)' % (pre, basisLongNm))
            elif disp == "eigenvalues":
                for model, title in zip(models, titles):
                    pre = (title + ' ' if title else '')
                    colHeadings.append('%sEigenvalues' % pre)
            elif disp == "barplot":
                for model, title in zip(models, titles):
                    pre = (title + ' ' if title else '')
                    colHeadings.append('%sEigenvalue Magnitudes' % pre)
            elif disp == "boxplot":
                for model, title in zip(models, titles):
                    basisLongNm = _tools.basis_longname(model.basis)
                    pre = (title + ' ' if title else '')
                    colHeadings.append('%sChoi matrix (%s basis)' % (pre, basisLongNm))
            else:
                raise ValueError("Invalid element of `display`: %s" % disp)
        formatters = [None] * len(colHeadings)

        table = _ReportTable(colHeadings, formatters, confidence_region_info=confidence_region_info)

        for i, gl in enumerate(opLabels):
            #Note: currently, we don't use confidence region...
            row_data = [gl]
            row_formatters = [None]

            for disp in display:
                if disp == "matrix":
                    for model, (choiMxs, _) in zip(models, qtysList):
                        row_data.append(choiMxs[i])
                        row_formatters.append('Brackets')

                elif disp == "eigenvalues":
                    for model, (_, evals) in zip(models, qtysList):
                        try:
                            evals[i] = evals[i].reshape(evals[i].size // 4, 4)
                            #assumes len(evals) is multiple of 4!
                        except:  # if it isn't try 3 (qutrits)
                            evals[i] = evals[i].reshape(evals[i].size // 3, 3)
                            #assumes len(evals) is multiple of 3!
                        row_data.append(evals[i])
                        row_formatters.append('Normal')

                elif disp == "barplot":
                    for model, (_, evals) in zip(models, qtysList):
                        evs, evsEB = evals[i].value_and_errorbar()
                        fig = _wp.ChoiEigenvalueBarPlot(self.ws, evs, evsEB)
                        row_data.append(fig)
                        row_formatters.append('Figure')

                elif disp == "boxplot":
                    for model, (choiMxs, _) in zip(models, qtysList):
                        choiMx_real = choiMxs[i].hermitian_to_real()
                        choiMx, EB = choiMx_real.value_and_errorbar()
                        fig = _wp.GateMatrixPlot(self.ws, choiMx,
                                                 colorbar=False,
                                                 mx_basis=model.basis,
                                                 eb_matrix=EB)
                        row_data.append(fig)
                        row_formatters.append('Figure')

            table.add_row(row_data, row_formatters)
        table.finish()
        return table


class GaugeRobustModelTable(WorkspaceTable):
    """
    Create a table showing a model in a gauge-robust representation.

    Parameters
    ----------
    ws : Workspace
        The containing (parent) workspace.

    model : Model
        The Model to display.

    target_model : Model
        The (usually ideal) reference model to compute gauge-invariant
        quantities with respect to.

    display_as : {"numbers", "boxes"}, optional
        How to display the operation matrices, as either numerical
        grids (fine for small matrices) or as a plot of colored
        boxes (space-conserving and better for large matrices).

    confidence_region_info : ConfidenceRegion, optional
        If not None, specifies a confidence-region
        used to display error intervals.
    """

    def __init__(self, ws, model, target_model, display_as="boxes",
                 confidence_region_info=None):
        """
        Create a table showing a gauge-invariant representation of a model.

        Parameters
        ----------
        model : Model
            The Model to display.

        target_model : Model
            The (usually ideal) reference model to compute gauge-invariant
            quantities with respect to.

        display_as : {"numbers", "boxes"}, optional
            How to display the operation matrices, as either numerical
            grids (fine for small matrices) or as a plot of colored
            boxes (space-conserving and better for large matrices).

        confidence_region_info : ConfidenceRegion, optional
            If not None, specifies a confidence-region
            used to display error intervals.

        Returns
        -------
        ReportTable
        """
        super(GaugeRobustModelTable, self).__init__(ws, self._create, model, target_model,
                                                    display_as, confidence_region_info)

    def _create(self, model, target_model, display_as, confidence_region_info):

        assert(isinstance(model, _objs.ExplicitOpModel)), "%s only works with explicit models" % str(type(self))
        opLabels = model.primitive_op_labels  # use labels of 1st model

        colHeadings = ['Gate', 'M - I'] + ['FinvF(%s) - I' % str(lbl) for lbl in opLabels]
        formatters = [None] * len(colHeadings)
        confidence_region_info = None  # Don't deal with CIs yet...

        def get_gig_decomp(mx, tmx):  # "Gauge invariant gateset" decomposition
            G0, G = tmx, mx
            #ev0, U0 = _tools.sorted_eig(G0)
            #ev, U = _tools.sorted_eig(G)
            #U0inv = _np.linalg.inv(U0)
            #Uinv = _np.linalg.inv(U)

            _, U, U0, ev0 = _tools.compute_best_case_gauge_transform(G, G0, return_all=True)
            U0inv = _np.linalg.inv(U0)
            Uinv = _np.linalg.inv(U)
            kite = _tools.compute_kite(ev0)

            F = _tools.find_zero_communtant_connection(U, Uinv, U0, U0inv, kite)  # Uinv * F * U0 is block diag
            Finv = _np.linalg.inv(F)
            # if G0 = U0 * E0 * U0inv then
            # Uinv * F * G0 * Finv * U = D * E0 * Dinv = E0 b/c D is block diagonal w/E0's degenercies
            # so F * G0 * Finv = U * E0 * Uinv = Gp ==> Finv * G * F = M * G0
            M = _np.dot(Finv, _np.dot(G, _np.dot(F, _np.linalg.inv(G0))))
            assert(_np.linalg.norm(M.imag) < 1e-8)

            M0 = _np.dot(U0inv, _np.dot(M, U0))  # M in G0's eigenbasis
            assert(_np.linalg.norm(_tools.project_onto_antikite(M0, kite)) < 1e-8)  # should be block diagonal
            assert(_np.allclose(G, _np.dot(F, _np.dot(M, _np.dot(G0, Finv)))))  # this is desired decomp
            assert(_np.linalg.norm(M.imag) < 1e-6 and _np.linalg.norm(F.imag) < 1e-6)  # and everthing should be real
            return F, M, Finv

        table = _ReportTable(colHeadings, formatters, confidence_region_info=confidence_region_info)
        I = _np.identity(model.dim, 'd')

        M = 0.0  # max abs for colorscale
        op_decomps = {}
        for gl in opLabels:
            try:
                op_decomps[gl] = get_gig_decomp(model.operations[gl].to_dense(),
                                                target_model.operations[gl].to_dense())
                M = max(M, max(_np.abs((op_decomps[gl][1] - I).flat)))  # update max
            except Exception as e:
                _warnings.warn("Failed gauge-robust decomposition of %s op:\n%s" % (gl, str(e)))

        for i, lbl in enumerate(opLabels):
            if lbl not in op_decomps: continue
            for j, lbl2 in enumerate(opLabels):
                if lbl2 not in op_decomps: continue
                if i == j: continue
                val = _np.dot(op_decomps[lbl][2], op_decomps[lbl2][0]) - I  # value plotted below
                M = max(M, max(_np.abs(val).flat))  # update max

        #FUTURE: instruments too?
        for i, lbl in enumerate(opLabels):
            row_data = [lbl]
            row_formatters = [None]
            if lbl in op_decomps:
                Fi, Mi, Finvi = op_decomps[lbl]

                #Print "M" matrix
                if display_as == "numbers":
                    row_data.append(Mi - I)
                    row_formatters.append('Brackets')
                elif display_as == "boxes":
                    fig = _wp.GateMatrixPlot(self.ws, Mi - I, -M, M, colorbar=False)
                    row_data.append(fig)
                    row_formatters.append('Figure')
                else:
                    raise ValueError("Invalid 'display_as' argument: %s" % display_as)
            else:
                row_data.append(_objs.reportableqty.ReportableQty(_np.nan))
                row_formatters.append('Normal')

            for j, lbl2 in enumerate(opLabels):
                if i == j:
                    row_data.append("0")
                    row_formatters.append(None)
                elif (lbl in op_decomps and lbl2 in op_decomps):
                    val = _np.dot(Finvi, op_decomps[lbl2][0])

                    #Print "Finv*F" matrix
                    if display_as == "numbers":
                        row_data.append(val - I)
                        row_formatters.append('Brackets')
                    elif display_as == "boxes":
                        fig = _wp.GateMatrixPlot(self.ws, val - I, -M, M, colorbar=False)
                        row_data.append(fig)
                        row_formatters.append('Figure')
                    else:
                        raise ValueError("Invalid 'display_as' argument: %s" % display_as)
                else:
                    row_data.append(_objs.reportableqty.ReportableQty(_np.nan))
                    row_formatters.append('Normal')

            table.add_row(row_data, row_formatters)

        table.finish()
        return table


class GaugeRobustMetricTable(WorkspaceTable):
    """
    Create a table showing a standard metric in a gauge-robust way.

    Parameters
    ----------
    ws : Workspace
        The containing (parent) workspace.

    model : Model
        The Model to display.

    target_model : Model
        The (usually ideal) reference model to compute gauge-invariant
        quantities with respect to.

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

    confidence_region_info : ConfidenceRegion, optional
        If not None, specifies a confidence-region
        used to display error intervals.
    """

    def __init__(self, ws, model, target_model, metric,
                 confidence_region_info=None):
        """
        Create a table showing a standard metric in a gauge-robust way.

        Parameters
        ----------
        model : Model
            The Model to display.

        target_model : Model
            The (usually ideal) reference model to compute gauge-invariant
            quantities with respect to.

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

        confidence_region_info : ConfidenceRegion, optional
            If not None, specifies a confidence-region
            used to display error intervals.

        Returns
        -------
        ReportTable
        """
        super(GaugeRobustMetricTable, self).__init__(ws, self._create, model, target_model,
                                                     metric, confidence_region_info)

    def _create(self, model, target_model, metric, confidence_region_info):

        assert(isinstance(model, _objs.ExplicitOpModel)), "%s only works with explicit models" % str(type(self))
        opLabels = model.primitive_op_labels

        colHeadings = [''] + ['%s' % str(lbl) for lbl in opLabels]
        formatters = [None] * len(colHeadings)
        confidence_region_info = None  # Don't deal with CIs yet...

        # Table will essentially be a matrix whose diagonal elements are
        # --> metric(GateA_in_As_best_gauge, TargetA)
        #     where a "best gauge" of a gate is one where it is co-diagonal with its target (same evecs can diag both).
        # Off-diagonal elements are given by:
        # --> min( metric(TargetA_in_Bs_best_gauge, TargetA), metric(TargetB_in_As_best_gauge, TargetB) )
        #
        # Thus, the diagonal elements tell us how much worse a (target) gate gets when just it's eigenvalues are
        # replaced with those of the actual estimated gate, and the off-diagonal elements tell us the least amount of
        # damage that must be done to a pair of (target) gates when just changing their eigenvectors to be consistent
        # with the actual estimated gates.

        table = _ReportTable(colHeadings, formatters, confidence_region_info=confidence_region_info)

        orig_model = model.copy()
        orig_model.set_all_parameterizations("full")  # so we can freely gauge transform this
        orig_target = target_model.copy()
        orig_target.set_all_parameterizations("full")  # so we can freely gauge transform this

        # ** A first attempt at fixing the gauge optimization issues. ** -- "frobeniustt" should replace this.
        #if metric in ("inf", "agi", "nuinf", "nuagi", "evinf", "evagi", "evnuinf", "evnuagi"):
        #    gmetric = "fidelity"
        #elif metric in ("trace", "diamond", "evdiamond", "evnudiamond"):
        #    gmetric = "tracedist"
        #else:
        #    gmetric = "frobenius"
        gmetric = "frobeniustt"

        mdl_in_best_gauge = []
        target_mdl_in_best_gauge = []
        for lbl in opLabels:
            gate_mx = orig_model.operations[lbl].to_dense()
            target_gate_mx = target_model.operations[lbl].to_dense()
            Ugauge = _tools.compute_best_case_gauge_transform(gate_mx, target_gate_mx)
            Ugg = _objs.FullGaugeGroupElement(_np.linalg.inv(Ugauge))  # transforms gates as Ugauge * gate * Ugauge_inv

            mdl = orig_model.copy()
            mdl.transform_inplace(Ugg)

            #DEBUG statements for trying to figure out why we get negative off-diagonals so often.
            #print("----- ",lbl,"--------")
            #print("PT1:\n",mdl.strdiff(target_model))
            #print("PT1b:\n",mdl.strdiff(target_model, 'inf'))
            try:
                _, Ugg_addl, mdl = _gopt.gaugeopt_to_target(mdl, orig_target, gates_metric=gmetric, spam_metric=gmetric,
                                                            item_weights={'spam': 0, 'gates': 1e-4, lbl: 1.0},
                                                            return_all=True, tol=1e-5, maxiter=100)  # ADDITIONAL GOPT
            except Exception as e:
                _warnings.warn(("GaugeRobustMetricTable gauge opt failed for %s label - "
                                "falling back to frobenius metric! Error was:\n%s") % (lbl, str(e)))
                _, Ugg_addl, mdl = _gopt.gaugeopt_to_target(mdl, orig_target, gates_metric="frobenius",
                                                            spam_metric="frobenius",
                                                            item_weights={'spam': 0, 'gates': 1e-4, lbl: 1.0},
                                                            return_all=True, tol=1e-5, maxiter=100)  # ADDITIONAL GOPT

            #print("PT2:\n",mdl.strdiff(target_model))
            #print("PT2b:\n",mdl.strdiff(target_model, 'inf'))
            mdl_in_best_gauge.append(mdl)

            target_mdl = orig_target.copy()
            target_mdl.transform_inplace(Ugg)
            target_mdl.transform_inplace(Ugg_addl)  # ADDITIONAL GOPT
            target_mdl_in_best_gauge.append(target_mdl)

        #FUTURE: instruments too?
        for i, lbl in enumerate(opLabels):
            row_data = [lbl]
            row_formatters = [None]

            for j, lbl2 in enumerate(opLabels):
                if i > j:  # leave lower diagonal blank
                    el = _objs.reportableqty.ReportableQty(_np.nan)
                elif i == j:  # diagonal element
                    try:
                        el = _reportables.evaluate_opfn_by_name(
                            metric, mdl_in_best_gauge[i], target_model, lbl, confidence_region_info)
                    except Exception:
                        _warnings.warn("Error computing %s for %s op in gauge-robust metrics table!" % (metric, lbl))
                        el = _objs.reportableqty.ReportableQty(_np.nan)
                else:  # off-diagonal element
                    try:
                        el1 = _reportables.evaluate_opfn_by_name(
                            metric, target_mdl_in_best_gauge[i], target_mdl_in_best_gauge[j], lbl2,
                            confidence_region_info)
                        el2 = _reportables.evaluate_opfn_by_name(
                            metric, target_mdl_in_best_gauge[i], target_mdl_in_best_gauge[j], lbl,
                            confidence_region_info)
                        el = _objs.reportableqty.minimum(el1, el2)
                    except Exception:
                        _warnings.warn("Error computing %s for %s,%s ops in gauge-robust metrics table!" %
                                       (metric, lbl, lbl2))
                        el = _objs.reportableqty.ReportableQty(_np.nan)

                row_data.append(el)
                row_formatters.append('Normal')

            table.add_row(row_data, row_formatters)

        table.finish()
        return table


class ModelVsTargetTable(WorkspaceTable):
    """
    Table comparing a Model (as a whole) to a target

    Parameters
    ----------
    ws : Workspace
        The containing (parent) workspace.

    model : Model
        The model to compare with `target_model`.

    target_model : Model
        The target model to compare with.

    clifford_compilation : dict
        A dictionary of circuits, one for each Clifford operation
        in the Clifford group relevant to the model Hilbert space.  If
        None, then rows requiring a clifford compilation are omitted.

    confidence_region_info : ConfidenceRegion, optional
        If not None, specifies a confidence-region
        used to display error intervals.
    """

    def __init__(self, ws, model, target_model, clifford_compilation, confidence_region_info=None):
        """
        Create a table comparing a model (as a whole) to a target model
        using metrics that can be evaluatd for an entire model.

        Parameters
        ----------
        model, target_model : Model
            The models to compare

        clifford_compilation : dict
            A dictionary of circuits, one for each Clifford operation
            in the Clifford group relevant to the model Hilbert space.  If
            None, then rows requiring a clifford compilation are omitted.

        confidence_region_info : ConfidenceRegion, optional
            If not None, specifies a confidence-region
            used to display error intervals.

        Returns
        -------
        ReportTable
        """
        super(ModelVsTargetTable, self).__init__(ws, self._create, model,
                                                 target_model, clifford_compilation,
                                                 confidence_region_info)

    def _create(self, model, target_model, clifford_compilation, confidence_region_info):

        colHeadings = ('Metric', "Value")
        formatters = (None, None)

        tooltips = colHeadings
        table = _ReportTable(colHeadings, formatters, col_heading_labels=tooltips,
                             confidence_region_info=confidence_region_info)

        #Leave this off for now, as it's primary use is to compare with RB and the predicted RB number is better
        #for this.
        #pAGsI = _ev(_reportables.Average_gateset_infidelity(model, target_model), confidence_region_info)
        #table.add_row(("Avg. primitive model infidelity", pAGsI), (None, 'Normal') )

        pRBnum = _ev(_reportables.Predicted_rb_number(model, target_model), confidence_region_info)
        table.add_row(("Predicted primitive RB number", pRBnum), (None, 'Normal'))

        if clifford_compilation and isinstance(model.sim, _MatrixFSim):
            clifford_model = _cnst.create_explicit_alias_model(model, clifford_compilation)
            clifford_targetModel = _cnst.create_explicit_alias_model(target_model, clifford_compilation)

            ##For clifford versions we don't have a confidence region - so no error bars
            #AGsI = _ev(_reportables.Average_gateset_infidelity(clifford_model, clifford_targetModel))
            #table.add_row(("Avg. clifford model infidelity", AGsI), (None, 'Normal') )

            RBnum = _ev(_reportables.Predicted_rb_number(clifford_model, clifford_targetModel))
            table.add_row(("Predicted Clifford RB number", RBnum), (None, 'Normal'))

        table.finish()
        return table


class GatesVsTargetTable(WorkspaceTable):
    """
    Table comparing a Model's gates to those of a target model

    Parameters
    ----------
    ws : Workspace
        The containing (parent) workspace.

    model : Model
        The model to compare to `target_model`.

    target_model : model
        The model to compare with.

    confidence_region_info : ConfidenceRegion, optional
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
        - "unmodeled" : unmodeled "wildcard" budget

    virtual_ops : list, optional
        If not None, a list of `Circuit` objects specifying additional "gates"
        (i.e. processes) to compute eigenvalues of.  Length-1 circuits are
        automatically discarded so they are not displayed twice.

    wildcard: PrimitiveOpsWildcardBudget
        A wildcard budget with a `budget_for` method that is used to
        fill in the "unmodeled" error column when it is requested.
    """

    def __init__(self, ws, model, target_model, confidence_region_info=None,
                 display=('inf', 'agi', 'trace', 'diamond', 'nuinf', 'nuagi'),
                 virtual_ops=None, wildcard=None):
        """
        Create a table comparing a model's gates to a target model using
        metrics such as the  infidelity, diamond-norm distance, and trace distance.

        Parameters
        ----------
        model, target_model : Model
            The models to compare

        confidence_region_info : ConfidenceRegion, optional
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
            - "unmodeled" : unmodeled "wildcard" budget

        virtual_ops : list, optional
            If not None, a list of `Circuit` objects specifying additional "gates"
            (i.e. processes) to compute eigenvalues of.  Length-1 circuits are
            automatically discarded so they are not displayed twice.

        wildcard: PrimitiveOpsWildcardBudget
            A wildcard budget with a `budget_for` method that is used to
            fill in the "unmodeled" error column when it is requested.

        Returns
        -------
        ReportTable
        """
        super(GatesVsTargetTable, self).__init__(ws, self._create, model,
                                                 target_model, confidence_region_info,
                                                 display, virtual_ops, wildcard)

    def _create(self, model, target_model, confidence_region_info,
                display, virtual_ops, wildcard):

        opLabels = model.primitive_op_labels  # operation labels
        instLabels = list(model.instruments.keys())  # requires an explicit model!
        assert(isinstance(model, _objs.ExplicitOpModel)), "%s only works with explicit models" % str(type(self))

        colHeadings = ['Gate'] if (virtual_ops is None) else ['Gate or Germ']
        tooltips = ['Gate'] if (virtual_ops is None) else ['Gate or Germ']
        for disp in display:
            if disp == "unmodeled" and not wildcard: continue  # skip wildcard column if there is no wilcard info
            try:
                heading, tooltip = _reportables.info_of_opfn_by_name(disp)
            except ValueError:
                raise ValueError("Invalid display column name: %s" % disp)
            colHeadings.append(heading)
            tooltips.append(tooltip)

        formatters = (None,) + ('Conversion',) * (len(colHeadings) - 1)

        table = _ReportTable(colHeadings, formatters, col_heading_labels=tooltips,
                             confidence_region_info=confidence_region_info)

        formatters = (None,) + ('Normal',) * (len(colHeadings) - 1)

        if virtual_ops is None:
            iterOver = opLabels
        else:
            iterOver = opLabels + tuple((v for v in virtual_ops if len(v) > 1))

        for gl in iterOver:
            #Note: gl may be a operation label (a string) or a Circuit
            row_data = [str(gl)]

            for disp in display:
                if disp == "unmodeled":  # a special case for now
                    if wildcard:
                        row_data.append(_objs.reportableqty.ReportableQty(
                            wildcard.budget_for(gl)))
                    continue  # Note: don't append anything if 'not wildcard'

                #import time as _time #DEBUG
                #tStart = _time.time() #DEBUG
                if target_model is None:
                    qty = _objs.reportableqty.ReportableQty(_np.nan)
                else:
                    qty = _reportables.evaluate_opfn_by_name(
                        disp, model, target_model, gl, confidence_region_info)
                #tm = _time.time()-tStart #DEBUG
                #if tm > 0.01: print("DB: Evaluated %s in %gs" % (disp, tm)) #DEBUG
                row_data.append(qty)

            table.add_row(row_data, formatters)

        #Iterate over instruments
        for il in instLabels:
            row_data = [str(il)]
            inst = model.instruments[il]
            tinst = target_model.instruments[il]
            basis = model.basis

            #Note: could move this to a reportables function in future for easier
            # confidence region support - for now, no CI support:
            for disp in display:
                if disp == "unmodeled":  # a special case for now
                    if wildcard:
                        row_data.append(_objs.reportableqty.ReportableQty(
                            wildcard.budget_for(il)))
                    continue  # Note: don't append anything if 'not wildcard'

                if disp == "inf":
                    sqrt_component_fidelities = [_np.sqrt(_reportables.entanglement_fidelity(inst[l], tinst[l], basis))
                                                 for l in inst.keys()]
                    qty = 1 - sum(sqrt_component_fidelities)**2
                    row_data.append(_objs.reportableqty.ReportableQty(qty))

                elif disp == "diamond":
                    nComps = len(inst.keys())
                    tpbasis = _DirectSumBasis([basis] * nComps)
                    composite_op = _np.zeros((inst.dim * nComps, inst.dim * nComps), 'd')
                    composite_top = _np.zeros((inst.dim * nComps, inst.dim * nComps), 'd')
                    for i, clbl in enumerate(inst.keys()):
                        a, b = i * inst.dim, (i + 1) * inst.dim
                        composite_op[a:b, a:b] = inst[clbl].to_dense()
                        composite_top[a:b, a:b] = tinst[clbl].to_dense()
                        qty = _reportables.half_diamond_norm(composite_op, composite_top, tpbasis)
                    row_data.append(_objs.reportableqty.ReportableQty(qty))

                else:
                    row_data.append(_objs.reportableqty.ReportableQty(_np.nan))

            table.add_row(row_data, formatters)

        table.finish()
        return table


class SpamVsTargetTable(WorkspaceTable):
    """
    Table comparing a Model's SPAM vectors to those of a target

    Parameters
    ----------
    ws : Workspace
        The containing (parent) workspace.

    model : Model
        The model to compare to `target_model`.

    target_model : model
        The model to compare with.

    confidence_region_info : ConfidenceRegion, optional
        If not None, specifies a confidence-region
        used to display error intervals.
    """

    def __init__(self, ws, model, target_model, confidence_region_info=None):
        """
        Create a table comparing a model's SPAM operations to a target model
        using state infidelity and trace distance.

        Parameters
        ----------
        model, target_model : Model
            The models to compare

        confidence_region_info : ConfidenceRegion, optional
            If not None, specifies a confidence-region
            used to display error intervals.

        Returns
        -------
        ReportTable
        """
        super(SpamVsTargetTable, self).__init__(ws, self._create, model,
                                                target_model, confidence_region_info)

    def _create(self, model, target_model, confidence_region_info):

        prepLabels = list(model.preps.keys())
        povmLabels = list(model.povms.keys())

        colHeadings = ('Prep/POVM', "Infidelity", "1/2 Trace|Distance", "1/2 Diamond-Dist")
        formatters = (None, 'Conversion', 'Conversion', 'Conversion')
        tooltips = ('', 'State infidelity or entanglement infidelity of POVM map',
                    'Trace distance between states (preps) or Jamiolkowski states of POVM maps',
                    'Half-diamond-norm distance between POVM maps')
        table = _ReportTable(colHeadings, formatters, col_heading_labels=tooltips,
                             confidence_region_info=confidence_region_info)

        formatters = ['Rho'] + ['Normal'] * (len(colHeadings) - 1)
        prepInfidelities = [_ev(_reportables.Vec_infidelity(model, target_model, l,
                                                            'prep'), confidence_region_info)
                            for l in prepLabels]
        prepTraceDists = [_ev(_reportables.Vec_tr_diff(model, target_model, l,
                                                       'prep'), confidence_region_info)
                          for l in prepLabels]
        prepDiamondDists = [_objs.reportableqty.ReportableQty(_np.nan)] * len(prepLabels)
        for rowData in zip(prepLabels, prepInfidelities, prepTraceDists,
                           prepDiamondDists):
            table.add_row(rowData, formatters)

        formatters = ['Normal'] + ['Normal'] * (len(colHeadings) - 1)
        povmInfidelities = [_ev(_reportables.POVM_entanglement_infidelity(
            model, target_model, l), confidence_region_info)
            for l in povmLabels]
        povmTraceDists = [_ev(_reportables.POVM_jt_diff(
            model, target_model, l), confidence_region_info)
            for l in povmLabels]
        povmDiamondDists = [_ev(_reportables.POVM_half_diamond_norm(
            model, target_model, l), confidence_region_info)
            for l in povmLabels]

        for rowData in zip(povmLabels, povmInfidelities, povmTraceDists,
                           povmDiamondDists):
            table.add_row(rowData, formatters)

        table.finish()
        return table


class ErrgenTable(WorkspaceTable):
    """
    Table displaying the error generators of a Model's gates and their projections.

    Projections are given onto spaces of standard generators.

    Parameters
    ----------
    ws : Workspace
        The containing (parent) workspace.

    model : Model
        The model to compare to `target_model`.

    target_model : model
        The model to compare with.

    confidence_region_info : ConfidenceRegion, optional
        If not None, specifies a confidence-region
        used to display error intervals.

    display : tuple of {"errgen","H","S","A"}
        Specifes which columns to include: the error generator itself
        and the projections of the generator onto Hamiltoian-type error
        (generators), Stochastic-type errors, and Affine-type errors.

    display_as : {"numbers", "boxes"}, optional
        How to display the requested matrices, as either numerical
        grids (fine for small matrices) or as a plot of colored boxes
        (space-conserving and better for large matrices).

    gen_type : {"logG-logT", "logTiG", "logGTi"}
        The type of error generator to compute.  Allowed values are:

        - "logG-logT" : errgen = log(gate) - log(target_op)
        - "logTiG" : errgen = log( dot(inv(target_op), gate) )
        - "logTiG" : errgen = log( dot(gate, inv(target_op)) )
    """

    def __init__(self, ws, model, target_model, confidence_region_info=None,
                 display=("errgen", "H", "S", "A"), display_as="boxes",
                 gen_type="logGTi"):
        """
        Create a table listing the error generators obtained by
        comparing a model's gates to a target model.

        Parameters
        ----------
        model, target_model : Model
            The models to compare

        display : tuple of {"errgen","H","S","A"}
            Specifes which columns to include: the error generator itself
            and the projections of the generator onto Hamiltoian-type error
            (generators), Stochastic-type errors, and Affine-type errors.

        display_as : {"numbers", "boxes"}, optional
            How to display the requested matrices, as either numerical
            grids (fine for small matrices) or as a plot of colored boxes
            (space-conserving and better for large matrices).

        confidence_region_info : ConfidenceRegion, optional
            If not None, specifies a confidence-region
            used to display error intervals.

        gen_type : {"logG-logT", "logTiG", "logGTi"}
            The type of error generator to compute.  Allowed values are:

            - "logG-logT" : errgen = log(gate) - log(target_op)
            - "logTiG" : errgen = log( dot(inv(target_op), gate) )
            - "logTiG" : errgen = log( dot(gate, inv(target_op)) )

        Returns
        -------
        ReportTable
        """
        super(ErrgenTable, self).__init__(ws, self._create, model,
                                          target_model, confidence_region_info,
                                          display, display_as, gen_type)

    def _create(self, model, target_model,
                confidence_region_info, display, display_as, gen_type):

        opLabels = model.primitive_op_labels  # operation labels
        basis = model.basis
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
        table = _ReportTable(colHeadings, (None,) * len(colHeadings),
                             confidence_region_info=confidence_region_info)

        errgenAndProjs = {}
        errgensM = []
        hamProjsM = []
        stoProjsM = []
        affProjsM = []

        def get_min_max(max_lst, m):
            """return a [min,max] already in list if there's one within an
               order of magnitude"""
            m = max(m, ABS_THRESHOLD)
            for mx in max_lst:
                if (abs(m) >= 1e-6 and 0.9999 < mx / m < 10) or (abs(mx) < 1e-6 and abs(m) < 1e-6):
                    return -mx, mx
            return None

        ABS_THRESHOLD = 1e-6  # don't let color scales run from 0 to 0: at least this much!

        def add_max(max_lst, m):
            """add `m` to a list of maximas if it's different enough from
               existing elements"""
            m = max(m, ABS_THRESHOLD)
            if not get_min_max(max_lst, m):
                max_lst.append(m)

        #Do computation, so shared color scales can be computed
        for gl in opLabels:
            if gen_type == "logG-logT":
                info = _ev(_reportables.LogGmlogT_and_projections(
                    model, target_model, gl), confidence_region_info)
            elif gen_type == "logTiG":
                info = _ev(_reportables.LogTiG_and_projections(
                    model, target_model, gl), confidence_region_info)
            elif gen_type == "logGTi":
                info = _ev(_reportables.LogGTi_and_projections(
                    model, target_model, gl), confidence_region_info)
            else: raise ValueError("Invalid generator type: %s" % gen_type)
            errgenAndProjs[gl] = info

            errgen = info['error generator'].value()
            absMax = _np.max(_np.abs(errgen))
            add_max(errgensM, absMax)

            if "H" in display:
                absMax = _np.max(_np.abs(info['hamiltonian projections'].value()))
                add_max(hamProjsM, absMax)

            if "S" in display:
                absMax = _np.max(_np.abs(info['stochastic projections'].value()))
                add_max(stoProjsM, absMax)

            if "A" in display:
                absMax = _np.max(_np.abs(info['affine projections'].value()))
                add_max(affProjsM, absMax)

        #Do plotting
        for gl in opLabels:
            row_data = [gl]
            row_formatters = [None]
            info = errgenAndProjs[gl]

            for disp in display:
                if disp == "errgen":
                    if display_as == "boxes":
                        errgen, EB = info['error generator'].value_and_errorbar()
                        m, M = get_min_max(errgensM, _np.max(_np.abs(errgen)))
                        errgen_fig = _wp.GateMatrixPlot(self.ws, errgen, m, M,
                                                        basis, eb_matrix=EB)
                        row_data.append(errgen_fig)
                        row_formatters.append('Figure')
                    else:
                        row_data.append(info['error generator'])
                        row_formatters.append('Brackets')

                elif disp == "H":
                    if display_as == "boxes":
                        T = "Power %.2g" % info['hamiltonian projection power'].value()
                        hamProjs, EB = info['hamiltonian projections'].value_and_errorbar()
                        m, M = get_min_max(hamProjsM, _np.max(_np.abs(hamProjs)))
                        hamdecomp_fig = _wp.ProjectionsBoxPlot(
                            self.ws, hamProjs, basis, m, M,
                            box_labels=True, eb_matrix=EB, title=T)
                        row_data.append(hamdecomp_fig)
                        row_formatters.append('Figure')
                    else:
                        row_data.append(info['hamiltonian projections'])
                        row_formatters.append('Brackets')

                elif disp == "S":
                    if display_as == "boxes":
                        T = "Power %.2g" % info['stochastic projection power'].value()
                        stoProjs, EB = info['stochastic projections'].value_and_errorbar()
                        m, M = get_min_max(stoProjsM, _np.max(_np.abs(stoProjs)))
                        stodecomp_fig = _wp.ProjectionsBoxPlot(
                            self.ws, stoProjs, basis, m, M,
                            box_labels=True, eb_matrix=EB, title=T)
                        row_data.append(stodecomp_fig)
                        row_formatters.append('Figure')
                    else:
                        row_data.append(info['stochastic projections'])
                        row_formatters.append('Brackets')

                elif disp == "A":
                    if display_as == "boxes":
                        T = "Power %.2g" % info['affine projection power'].value()
                        affProjs, EB = info['affine projections'].value_and_errorbar()
                        m, M = get_min_max(affProjsM, _np.max(_np.abs(affProjs)))
                        affdecomp_fig = _wp.ProjectionsBoxPlot(
                            self.ws, affProjs, basis, m, M,
                            box_labels=True, eb_matrix=EB, title=T)
                        row_data.append(affdecomp_fig)
                        row_formatters.append('Figure')
                    else:
                        row_data.append(info['affine projections'])
                        row_formatters.append('Brackets')

            table.add_row(row_data, row_formatters)

        table.finish()
        return table


class GaugeRobustErrgenTable(WorkspaceTable):
    """
    Table of gauge-robust error generators.

    A table displaying the first-order gauge invariant ("gauge robust")
    linear combinations of standard error generator coefficients for
    the gates in a model.

    Parameters
    ----------
    ws : Workspace
        The containing (parent) workspace.

    model : Model
        The model to compare to `target_model`.

    target_model : model
        The model to compare with.

    confidence_region_info : ConfidenceRegion, optional
        If not None, specifies a confidence-region
        used to display error intervals.

    gen_type : {"logG-logT", "logTiG", "logGTi"}
        The type of error generator to compute.  Allowed values are:

        - "logG-logT" : errgen = log(gate) - log(target_op)
        - "logTiG" : errgen = log( dot(inv(target_op), gate) )
        - "logTiG" : errgen = log( dot(gate, inv(target_op)) )
    """

    def __init__(self, ws, model, target_model, confidence_region_info=None,
                 gen_type="logGTi"):
        """
        Create a table listing the first-order gauge invariant ("gauge robust")
        linear combinations of standard error generator coefficients for
        the gates in `model`.  This table identifies, through the use of
        "synthetic idle tomography", which combinations of standard-error-
        generator coefficients are robust (to first-order) to gauge variations.

        Parameters
        ----------
        model, target_model : Model
            The models to compare

        confidence_region_info : ConfidenceRegion, optional
            If not None, specifies a confidence-region
            used to display error intervals.

        gen_type : {"logG-logT", "logTiG", "logGTi"}
            The type of error generator to compute.  Allowed values are:

            - "logG-logT" : errgen = log(gate) - log(target_op)
            - "logTiG" : errgen = log( dot(inv(target_op), gate) )
            - "logTiG" : errgen = log( dot(gate, inv(target_op)) )

        Returns
        -------
        ReportTable
        """
        super(GaugeRobustErrgenTable, self).__init__(ws, self._create, model,
                                                     target_model, confidence_region_info,
                                                     gen_type)

    def _create(self, model, target_model, confidence_region_info, gen_type):
        assert(isinstance(model, _objs.ExplicitOpModel)), "%s only works with explicit models" % str(type(self))

        colHeadings = ['Error rates', 'Value']

        table = _ReportTable(colHeadings, (None,) * len(colHeadings),
                             confidence_region_info=confidence_region_info)

        assert(gen_type == "logGTi"), "Only `gen_type == \"logGTI\"` is supported when `gaugeRobust` is True"
        syntheticIdleStrs = []

        ## Construct synthetic idles
        maxPower = 4; maxLen = 6; Id = _np.identity(target_model.dim, 'd')
        baseStrs = _cnst.list_all_circuits_without_powers_and_cycles(list(model.operations.keys()), maxLen)
        for s in baseStrs:
            for i in range(1, maxPower):
                if len(s**i) > 1 and _np.linalg.norm(target_model.sim.product(s**i) - Id) < 1e-6:
                    syntheticIdleStrs.append(s**i); break
        #syntheticIdleStrs = _cnst.to_circuits([ ('Gx',)*4, ('Gy',)*4 ] ) #DEBUG!!!
        #syntheticIdleStrs = _cnst.to_circuits([ ('Gx',)*4, ('Gy',)*4, ('Gy','Gx','Gx')*2] ) #DEBUG!!!
        print("Using synthetic idles: \n", '\n'.join([str(opstr) for opstr in syntheticIdleStrs]))

        gaugeRobust_info = _ev(_reportables.Robust_LogGTi_and_projections(
            model, target_model, syntheticIdleStrs), confidence_region_info)

        for linear_combo_lbl, val in gaugeRobust_info.items():
            row_data = [linear_combo_lbl, val]
            row_formatters = [None, 'Normal']
            table.add_row(row_data, row_formatters)

        table.finish()
        return table


class NQubitErrgenTable(WorkspaceTable):
    """
    Table displaying the error rates (coefficients of error generators) of a Model's gates.

    The gates are assumed to have a particular structure.

    Specifically, gates must be :class:`LindbladOp` or
    :class:`StaticDenseOp` objects wrapped within :class:`EmbeddedOp` and/or
    :class:`ComposedOp` objects (this is consistent with the operation
    blocks of a :class:`CloudNoiseModel`).  As such, error rates
    are read directly from the gate objects rather than being computed by
    projecting dense gate representations onto a "basis" of fixed error
    generators (e.g. H+S+A generators).

    Parameters
    ----------
    ws : Workspace
        The containing (parent) workspace.

    model : Model
        The model to analyze.

    confidence_region_info : ConfidenceRegion, optional
        If not None, specifies a confidence-region
        used to display error intervals.

    display : tuple of {"H","S","A"}
        Specifes which columns to include: Hamiltoian-type,
        Pauli-Stochastic-type, and Affine-type rates, respectively.

    display_as : {"numbers", "boxes"}, optional
        How to display the requested matrices, as either numerical
        grids (fine for small matrices) or as a plot of colored boxes
        (space-conserving and better for large matrices).
    """

    def __init__(self, ws, model, confidence_region_info=None,
                 display=("H", "S", "A"), display_as="boxes"):
        """
        Create a table listing the error rates of the gates in `model`.

        The gates in `model` are assumed to have a particular structure,
        namely: they must be :class:`LindbladOp` or
        :class:`StaticDenseOp` objects wrapped within :class:`EmbeddedOp`
        and/or :class:`ComposedOp` objects.

        Error rates are organized by order of composition and which qubits
        the corresponding error generators act upon.

        Parameters
        ----------
        model : Model
            The model to analyze.

        confidence_region_info : ConfidenceRegion, optional
            If not None, specifies a confidence-region
            used to display error intervals.

        display : tuple of {"H","S","A"}
            Specifes which columns to include: Hamiltoian-type,
            Pauli-Stochastic-type, and Affine-type rates, respectively.

        display_as : {"numbers", "boxes"}, optional
            How to display the requested matrices, as either numerical
            grids (fine for small matrices) or as a plot of colored boxes
            (space-conserving and better for large matrices).

        Returns
        -------
        ReportTable
        """
        super(NQubitErrgenTable, self).__init__(ws, self._create, model,
                                                confidence_region_info,
                                                display, display_as)

    def _create(self, model, confidence_region_info, display, display_as):
        opLabels = model.primitive_op_labels  # operation labels

        #basis = model.basis
        #basisPrefix = ""
        #if basis.name == "pp": basisPrefix = "Pauli "
        #elif basis.name == "qt": basisPrefix = "Qutrit "
        #elif basis.name == "gm": basisPrefix = "GM "
        #elif basis.name == "std": basisPrefix = "Mx unit "

        colHeadings = ['Gate', 'Compos', 'SSLbls']

        for disp in display:
            #if disp == "errgen":
            #    colHeadings.append('Error Generator')
            if disp == "H":
                colHeadings.append('Hamiltonian Coeffs')
            elif disp == "S":
                colHeadings.append('Stochastic Coeffs')
            elif disp == "A":
                colHeadings.append('Affine Coeffs')
            else: raise ValueError("Invalid display element: %s" % disp)

        assert(display_as == "boxes" or display_as == "numbers")
        table = _ReportTable(colHeadings, (None,) * len(colHeadings),
                             confidence_region_info=confidence_region_info)

        def get_min_max(max_lst, m):
            """return a [min,max] already in list if there's one within an
               order of magnitude"""
            m = max(m, ABS_THRESHOLD)
            for mx in max_lst:
                if (abs(m) >= 1e-6 and 0.9999 < mx / m < 10) or (abs(mx) < 1e-6 and abs(m) < 1e-6):
                    return -mx, mx
            return None

        ABS_THRESHOLD = 1e-6  # don't let color scales run from 0 to 0: at least this much!

        def add_max(max_lst, m):
            """add `m` to a list of maximas if it's different enough from
               existing elements"""
            m = max(m, ABS_THRESHOLD)
            if not get_min_max(max_lst, m):
                max_lst.append(m)

        pre_rows = []; displayed_params = set()

        def process_gate(lbl, gate, comppos_prefix, sslbls):
            if isinstance(gate, _objs.ComposedOp):
                for i, fgate in enumerate(gate.factorops):
                    process_gate(lbl, fgate, comppos_prefix + (i,), sslbls)
            elif isinstance(gate, _objs.EmbeddedOp):
                process_gate(lbl, gate.embedded_op, comppos_prefix, gate.targetLabels)
            elif isinstance(gate, _objs.StaticDenseOp):
                pass  # no error coefficients associated w/static gates
            elif isinstance(gate, _objs.LindbladOp):

                # Only display coeffs for gates that correspond to *new*
                # (not yet displayed) parameters.
                params = set(gate.gpindices_as_array())
                if not params.issubset(displayed_params):
                    displayed_params.update(params)

                    Ldict, basis = gate.errorgen_coefficients(return_basis=True)
                    sparse = basis.sparse

                    #Try to find good labels for these basis elements
                    # (so far, just try to match with "pp" basis els)
                    ref_basis = _objs.BuiltinBasis("pp", gate.dim, sparse=sparse)
                    basisLbls = {}
                    for bl1, mx in zip(basis.labels, basis.elements):
                        for bl2, mx2 in zip(ref_basis.labels, ref_basis.elements):
                            if (sparse and _tools.sparse_equal(mx, mx2)) or (not sparse and _np.allclose(mx, mx2)):
                                basisLbls[bl1] = bl2; break
                        else:
                            basisLbls[bl1] = bl1

                    pre_rows.append((lbl, comppos_prefix, sslbls, Ldict, basisLbls))
            else:
                raise ValueError("Unknown gate type for NQubitErrgenTable: %s" % str(type(gate)))

        def get_plot_info(lindblad_dict, basis_lbls, typ):
            # for now just make a 1D plot - can get fancy later...
            ylabels = [""]
            xlabels = []
            coeffs = []
            for termInfo, coeff in lindblad_dict.items():
                termtyp = termInfo[0]
                if termtyp not in ("H", "S", "A"): raise ValueError("Unknown terminfo: ", termInfo)
                if (termtyp == "H" and typ == "hamiltonian") or \
                   (termtyp == "S" and typ == "stochastic") or \
                   (termtyp == "A" and typ == "affine"):
                    assert(len(termInfo) == 2), "Non-diagonal terms not suppoted (yet)!"
                    xlabels.append(basis_lbls[termInfo[1]])
                    coeffs.append(coeff)
            return _np.array([coeffs]), xlabels, ylabels

        #Do computation, so shared color scales can be computed
        if isinstance(model, _objs.ExplicitOpModel):
            for gl in opLabels:
                process_gate(gl, model.operations[gl], (), None)
        elif isinstance(model, _objs.LocalNoiseModel):  # process primitive op error
            for gl in opLabels:
                process_gate(gl, model.operation_blks['layers'][gl], (), None)
        elif isinstance(model, _objs.CloudNoiseModel):  # process primitive op error
            for gl in opLabels:
                process_gate(gl, model.operation_blks['cloudnoise'][gl], (), None)
        else:
            raise ValueError("Unrecognized type of model: %s" % str(type(model)))

        #get min/max
        if len(pre_rows) > 0:
            M = max((max(map(abs, Ldict.values())) for _, _, _, Ldict, _ in pre_rows))
            m = -M
        else:
            M = m = 0

        #Now pre_rows is filled, so we just need to create the plots:
        for gl, comppos, sslbls, Ldict, basisLbls in pre_rows:
            row_data = [gl, str(comppos), str(sslbls)]
            row_formatters = [None, None, None]

            for disp in display:
                if disp == "H":
                    hamCoeffs, xlabels, ylabels = get_plot_info(Ldict, basisLbls, "hamiltonian")
                    if display_as == "boxes":
                        #m,M = get_min_max(coeffsM,_np.max(_np.abs(hamCoeffs)))
                        # May need to add EB code and/or title to MatrixPlot in FUTURE
                        hamCoeffs_fig = _wp.MatrixPlot(
                            self.ws, hamCoeffs, m, M, xlabels, ylabels,
                            box_labels=True, prec="compacthp")
                        row_data.append(hamCoeffs_fig)
                        row_formatters.append('Figure')
                    else:
                        row_data.append(hamCoeffs)
                        row_formatters.append('Brackets')

                if disp == "S":
                    stoCoeffs, xlabels, ylabels = get_plot_info(Ldict, basisLbls, "stochastic")
                    if display_as == "boxes":
                        #m,M = get_min_max(coeffsM,_np.max(_np.abs(stoCoeffs)))
                        # May need to add EB code and/or title to MatrixPlot in FUTURE
                        stoCoeffs_fig = _wp.MatrixPlot(
                            self.ws, stoCoeffs, m, M, xlabels, ylabels,
                            box_labels=True, prec="compacthp")
                        row_data.append(stoCoeffs_fig)
                        row_formatters.append('Figure')
                    else:
                        row_data.append(stoCoeffs)
                        row_formatters.append('Brackets')

                if disp == "A":
                    affCoeffs, xlabels, ylabels = get_plot_info(Ldict, basisLbls, "affine")
                    if display_as == "boxes":
                        #m,M = get_min_max(coeffsM,_np.max(_np.abs(effCoeffs)))
                        # May need to add EB code and/or title to MatrixPlot in FUTURE
                        affCoeffs_fig = _wp.MatrixPlot(
                            self.ws, affCoeffs, m, M, xlabels, ylabels,
                            box_labels=True, prec="compacthp")
                        row_data.append(affCoeffs_fig)
                        row_formatters.append('Figure')
                    else:
                        row_data.append(affCoeffs)
                        row_formatters.append('Brackets')

            table.add_row(row_data, row_formatters)

        table.finish()
        return table


class OldRotationAxisVsTargetTable(WorkspaceTable):
    """
    Old 1-qubit-only gate rotation axis table

    Parameters
    ----------
    ws : Workspace
        The containing (parent) workspace.

    model : Model
        The model to compare to `target_model`. Must be single qubit.

    target_model : model
        The model to compare with.  Must be single qubit.

    confidence_region_info : ConfidenceRegion, optional
        If not None, specifies a confidence-region
        used to display error intervals.
    """

    def __init__(self, ws, model, target_model, confidence_region_info=None):
        """
        Create a table comparing the rotation axes of the single-qubit gates in
        `model` with those in `target_model`.  Differences are shown as
        angles between the rotation axes of corresponding gates.

        Parameters
        ----------
        model, target_model : Model
            The models to compare.  Must be single-qubit.

        confidence_region_info : ConfidenceRegion, optional
            If not None, specifies a confidence-region
            used to display error intervals.

        Returns
        -------
        ReportTable
        """
        super(OldRotationAxisVsTargetTable, self).__init__(
            ws, self._create, model, target_model, confidence_region_info)

    def _create(self, model, target_model, confidence_region_info):

        opLabels = model.primitive_op_labels  # operation labels

        colHeadings = ('Gate', "Angle between|rotation axes")
        formatters = (None, 'Conversion')

        anglesList = [_ev(_reportables.Model_model_angles_btwn_axes(
            model, target_model, gl), confidence_region_info) for gl in opLabels]

        table = _ReportTable(colHeadings, formatters, confidence_region_info=confidence_region_info)

        formatters = [None] + ['Pi']

        for gl, angle in zip(opLabels, anglesList):
            rowData = [gl] + [angle]
            table.add_row(rowData, formatters)

        table.finish()
        return table


class GateDecompTable(WorkspaceTable):
    """
    Table of angle & axis decompositions of a Model's gates

    Parameters
    ----------
    ws : Workspace
        The containing (parent) workspace.

    model : Model
        The estimated model.

    target_model : Model
        The target model, used to help disambiguate the matrix
        logarithms that are used in the decomposition.

    confidence_region_info : ConfidenceRegion, optional
        If not None, specifies a confidence-region
        used to display error intervals.
    """

    def __init__(self, ws, model, target_model, confidence_region_info=None):
        """
        Create table for decomposing a model's gates.

        This table interprets the Hamiltonian projection of the log
        of the operation matrix to extract a rotation angle and axis.

        Parameters
        ----------
        model : Model
            The estimated model.

        target_model : Model
            The target model, used to help disambiguate the matrix
            logarithms that are used in the decomposition.

        confidence_region_info : ConfidenceRegion, optional
            If not None, specifies a confidence-region
            used to display error intervals.

        Returns
        -------
        ReportTable
        """
        super(GateDecompTable, self).__init__(ws, self._create, model,
                                              target_model, confidence_region_info)

    def _create(self, model, target_model, confidence_region_info):
        opLabels = model.primitive_op_labels  # operation labels

        colHeadings = ('Gate', 'Ham. Evals.', 'Rotn. angle', 'Rotn. axis', 'Log Error') \
            + tuple(["Axis angle w/%s" % str(gl) for gl in opLabels])
        tooltips = (
            'Gate', 'Hamiltonian Eigenvalues', 'Rotation angle', 'Rotation axis',
            'Taking the log of a gate may be performed approximately.  This is '
            'error in that estimate, i.e. norm(G - exp(approxLogG)).'
        ) + tuple(["Angle between the rotation axis of %s and the gate of the current row"
                   % str(gl) for gl in opLabels])
        formatters = [None] * len(colHeadings)

        table = _ReportTable(colHeadings, formatters,
                             col_heading_labels=tooltips, confidence_region_info=confidence_region_info)
        formatters = (None, 'Pi', 'Pi', 'Figure', 'Normal') + ('Pi',) * len(opLabels)

        decomp = _ev(_reportables.General_decomposition(
            model, target_model), confidence_region_info)

        for gl in opLabels:
            gl = str(gl)  # Label -> str for decomp-dict keys
            axis, axisEB = decomp[gl + ' axis'].value_and_errorbar()
            axisFig = _wp.ProjectionsBoxPlot(self.ws, axis, model.basis, -1.0, 1.0,
                                             box_labels=True, eb_matrix=axisEB)
            decomp[gl + ' hamiltonian eigenvalues'].scale_inplace(1.0 / _np.pi)  # scale evals to units of pi
            rowData = [gl, decomp[gl + ' hamiltonian eigenvalues'],
                       decomp[gl + ' angle'], axisFig,
                       decomp[gl + ' log inexactness']]

            for gl_other in opLabels:
                gl_other = str(gl_other)
                rotnAngle = decomp[gl + ' angle'].value()
                rotnAngle_other = decomp[gl_other + ' angle'].value()

                if gl_other == gl:
                    rowData.append("")
                elif abs(rotnAngle) < 1e-4 or abs(rotnAngle_other) < 1e-4:
                    rowData.append("--")
                else:
                    rowData.append(decomp[gl + ',' + gl_other + ' axis angle'])

            table.add_row(rowData, formatters)

        table.finish()
        return table


class OldGateDecompTable(WorkspaceTable):
    """
    1-qubit-only table of gate decompositions

    Parameters
    ----------
    ws : Workspace
        The containing (parent) workspace.

    model : Model
        A single-qubit `Model`.

    confidence_region_info : ConfidenceRegion, optional
        If not None, specifies a confidence-region
        used to display error intervals.
    """

    def __init__(self, ws, model, confidence_region_info=None):
        """
        Create table for decomposing a single-qubit model's gates.

        This table interprets the eigenvectors and eigenvalues of the
        gates to extract a rotation angle, axis, and various decay
        coefficients.

        Parameters
        ----------
        model : Model
            A single-qubit `Model`.

        confidence_region_info : ConfidenceRegion, optional
            If not None, specifies a confidence-region
            used to display error intervals.

        Returns
        -------
        ReportTable
        """
        super(OldGateDecompTable, self).__init__(ws, self._create, model, confidence_region_info)

    def _create(self, model, confidence_region_info):

        opLabels = model.primitive_op_labels  # operation labels
        colHeadings = ('Gate', 'Eigenvalues', 'Fixed pt', 'Rotn. axis', 'Diag. decay', 'Off-diag. decay')
        formatters = [None] * 6

        assert(isinstance(model, _objs.ExplicitOpModel)), "OldGateDecompTable only works with explicit models"
        decomps = [_reportables.decomposition(model.operations[gl]) for gl in opLabels]
        decompNames = ('fixed point',
                       'axis of rotation',
                       'decay of diagonal rotation terms',
                       'decay of off diagonal rotation terms')

        table = _ReportTable(colHeadings, formatters, confidence_region_info=confidence_region_info)

        formatters = (None, 'Vec', 'Normal', 'Normal', 'Normal', 'Normal')

        for decomp, gl in zip(decomps, opLabels):
            evals = _ev(_reportables.GateEigenvalues(model, gl))
            decomp, decompEB = decomp.value_and_errorbar()  # OLD

            rowData = [gl, evals] + [decomp.get(x, 'X') for x in decompNames[0:2]] + \
                [(decomp.get(x, 'X'), decompEB) for x in decompNames[2:4]]

            table.add_row(rowData, formatters)

        table.finish()
        return table


class OldRotationAxisTable(WorkspaceTable):
    """
    1-qubit-only table of gate rotation angles and axes

    Parameters
    ----------
    ws : Workspace
        The containing (parent) workspace.

    model : Model
        A single-qubit `Model`.

    confidence_region_info : ConfidenceRegion, optional
        If not None, specifies a confidence-region
        used to display error intervals.

    show_axis_angle_err_bars : bool, optional
        Whether or not table should include error bars on the angles
        between rotation axes (doing so makes the table take up more
        space).
    """

    def __init__(self, ws, model, confidence_region_info=None, show_axis_angle_err_bars=True):
        """
        Create a table of the angle between a gate rotation axes for
        gates belonging to a single-qubit model.

        Parameters
        ----------
        model : Model
            A single-qubit `Model`.

        confidence_region_info : ConfidenceRegion, optional
            If not None, specifies a confidence-region
            used to display error intervals.

        show_axis_angle_err_bars : bool, optional
            Whether or not table should include error bars on the angles
            between rotation axes (doing so makes the table take up more
            space).

        Returns
        -------
        ReportTable
        """
        super(OldRotationAxisTable, self).__init__(ws, self._create, model, confidence_region_info,
                                                   show_axis_angle_err_bars)

    def _create(self, model, confidence_region_info, show_axis_angle_err_bars):

        opLabels = model.primitive_op_labels

        assert(isinstance(model, _objs.ExplicitOpModel)), "OldRotationAxisTable only works with explicit models"
        decomps = [_reportables.decomposition(model.operations[gl]) for gl in opLabels]

        colHeadings = ("Gate", "Angle") + tuple(["RAAW(%s)" % gl for gl in opLabels])
        nCols = len(colHeadings)
        formatters = [None] * nCols

        table = "tabular"
        latex_head = "\\begin{%s}[l]{%s}\n\hline\n" % (table, "|c" * nCols + "|")
        latex_head += "\\multirow{2}{*}{Gate} & \\multirow{2}{*}{Angle} & " + \
                      "\\multicolumn{%d}{c|}{Angle between Rotation Axes} \\\\ \cline{3-%d}\n" % (len(opLabels), nCols)
        latex_head += " & & %s \\\\ \hline\n" % (" & ".join(map(str, opLabels)))

        table = _ReportTable(colHeadings, formatters,
                             custom_header={'latex': latex_head}, confidence_region_info=confidence_region_info)

        formatters = [None, 'Pi'] + ['Pi'] * len(opLabels)

        rotnAxisAnglesQty = _ev(_reportables.Angles_btwn_rotn_axes(model),
                                confidence_region_info)
        rotnAxisAngles, rotnAxisAnglesEB = rotnAxisAnglesQty.value_and_errorbar()

        for i, gl in enumerate(opLabels):
            decomp, decompEB = decomps[i].value_and_errorbar()  # OLD
            rotnAngle = decomp.get('pi rotations', 'X')

            angles_btwn_rotn_axes = []
            for j, gl_other in enumerate(opLabels):
                decomp_other, _ = decomps[j].value_and_errorbar()  # OLD
                rotnAngle_other = decomp_other.get('pi rotations', 'X')

                if gl_other == gl:
                    angles_btwn_rotn_axes.append(("", None))
                elif str(rotnAngle) == 'X' or abs(rotnAngle) < 1e-4 or \
                        str(rotnAngle_other) == 'X' or abs(rotnAngle_other) < 1e-4:
                    angles_btwn_rotn_axes.append(("--", None))
                elif not _np.isnan(rotnAxisAngles[i, j]):
                    if show_axis_angle_err_bars and rotnAxisAnglesEB is not None:
                        angles_btwn_rotn_axes.append((rotnAxisAngles[i, j], rotnAxisAnglesEB[i, j]))
                    else:
                        angles_btwn_rotn_axes.append((rotnAxisAngles[i, j], None))
                else:
                    angles_btwn_rotn_axes.append(("X", None))

            if confidence_region_info is None or decompEB is None:  # decompEB is None when gate decomp failed
                rowData = [gl, (rotnAngle, None)] + angles_btwn_rotn_axes
            else:
                rowData = [gl, (rotnAngle, decompEB.get('pi rotations', 'X'))] + angles_btwn_rotn_axes
            table.add_row(rowData, formatters)

        table.finish()
        return table


class GateEigenvalueTable(WorkspaceTable):
    """
    Table displaying, in a variety of ways, the eigenvalues of a Model's gates.

    Parameters
    ----------
    ws : Workspace
        The containing (parent) workspace.

    model : Model
        The Model

    target_model : Model, optional
        The target model.  If given, the target's eigenvalue will
        be plotted alongside `model`'s gate eigenvalue, the
        "relative eigenvalues".

    confidence_region_info : ConfidenceRegion, optional
        If not None, specifies a confidence-region
        used to display error intervals.

    display : tuple
        A tuple of one or more of the allowed options (see below) which
        specify which columns are displayed in the table.  If
        `target_model` is None, then `"target"`, `"rel"`, `"log-rel"`
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

    virtual_ops : list, optional
        If not None, a list of `Circuit` objects specifying additional "gates"
        (i.e. processes) to compute eigenvalues of.  Length-1 circuits are
        automatically discarded so they are not displayed twice.
    """

    def __init__(self, ws, model, target_model=None,
                 confidence_region_info=None,
                 display=('evals', 'rel', 'log-evals', 'log-rel', 'polar', 'relpolar'),
                 virtual_ops=None):
        """
        Create table which lists and displays (using a polar plot)
        the eigenvalues of a model's gates.

        Parameters
        ----------
        model : Model
            The Model

        target_model : Model, optional
            The target model.  If given, the target's eigenvalue will
            be plotted alongside `model`'s gate eigenvalue, the
            "relative eigenvalues".

        confidence_region_info : ConfidenceRegion, optional
            If not None, specifies a confidence-region
            used to display error intervals.

        display : tuple
            A tuple of one or more of the allowed options (see below) which
            specify which columns are displayed in the table.  If
            `target_model` is None, then `"target"`, `"rel"`, `"log-rel"`
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

        virtual_ops : list, optional
            If not None, a list of `Circuit` objects specifying additional "gates"
            (i.e. processes) to compute eigenvalues of.  Length-1 circuits are
            automatically discarded so they are not displayed twice.

        Returns
        -------
        ReportTable
        """
        super(GateEigenvalueTable, self).__init__(ws, self._create, model,
                                                  target_model,
                                                  confidence_region_info, display,
                                                  virtual_ops)

    def _create(self, model, target_model,
                confidence_region_info, display,
                virtual_ops):

        opLabels = model.primitive_op_labels  # operation labels
        assert(isinstance(model, _objs.ExplicitOpModel)), "GateEigenvalueTable only works with explicit models"

        colHeadings = ['Gate'] if (virtual_ops is None) else ['Gate or Germ']
        formatters = [None]
        for disp in display:
            if disp == "evals":
                colHeadings.append('Eigenvalues ($E$)')
                formatters.append(None)

            elif disp == "target":
                if target_model is not None:  # silently ignore
                    colHeadings.append('Target Evals. ($T$)')
                    formatters.append(None)

            elif disp == "rel":
                if target_model is not None:  # silently ignore
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
                colHeadings.append('Eigenvalues')  # Note: make sure header is *distinct* for pandas conversion
                formatters.append(None)

            elif disp == "relpolar":
                if(target_model is not None):  # silently ignore
                    colHeadings.append('Rel. Evals')  # Note: make sure header is *distinct* for pandas conversion
                    formatters.append(None)

            elif disp == "absdiff-evals":
                if(target_model is not None):  # silently ignore
                    colHeadings.append('|E - T|')
                    formatters.append('MathText')

            elif disp == "infdiff-evals":
                if(target_model is not None):  # silently ignore
                    colHeadings.append('1.0 - Re(\\bar{T}*E)')
                    formatters.append('MathText')

            elif disp == "absdiff-log-evals":
                if(target_model is not None):  # silently ignore
                    colHeadings.append('|Re(log E) - Re(log T)|')
                    colHeadings.append('|Im(log E) - Im(log T)|')
                    formatters.append('MathText')
                    formatters.append('MathText')

            elif disp == "evdm":
                if(target_model is not None):  # silently ignore
                    colHeadings.append('Eigenvalue Diamond norm')
                    formatters.append('Conversion')

            elif disp == "evinf":
                if(target_model is not None):  # silently ignore
                    colHeadings.append('Eigenvalue infidelity')
                    formatters.append(None)
            else:
                raise ValueError("Invalid display element: %s" % disp)

        table = _ReportTable(colHeadings, formatters, confidence_region_info=confidence_region_info)

        if virtual_ops is None:
            iterOver = opLabels
        else:
            iterOver = opLabels + tuple((v for v in virtual_ops if len(v) > 1))

        for gl in iterOver:
            #Note: gl may be a operation label (a string) or a Circuit
            row_data = [str(gl)]
            row_formatters = [None]

            #import time as _time #DEBUG
            #tStart = _time.time() #DEBUG
            fn = _reportables.GateEigenvalues if \
                isinstance(gl, _objs.Label) or isinstance(gl, str) else \
                _reportables.CircuitEigenvalues
            evals = _ev(fn(model, gl), confidence_region_info)
            #tm = _time.time() - tStart #DEBUG
            #if tm > 0.01: print("DB: Gate eigenvalues in %gs" % tm) #DEBUG

            evals = evals.reshape(evals.size, 1)
            #OLD: format to 2-columns - but polar plots are big, so just stick to 1col now
            #try: evals = evals.reshape(evals.size//2, 2) #assumes len(evals) is even!
            #except: evals = evals.reshape(evals.size, 1)

            if target_model is not None:
                #TODO: move this to a reportable qty to get error bars?

                if isinstance(gl, _objs.Label) or isinstance(gl, str):
                    target_evals = _np.linalg.eigvals(target_model.operations[gl].to_dense())  # no error bars
                else:
                    target_evals = _np.linalg.eigvals(target_model.sim.product(gl))  # no error bars

                if any([(x in display) for x in ('rel', 'log-rel', 'relpolar')]):
                    if isinstance(gl, _objs.Label) or isinstance(gl, str):
                        rel_evals = _ev(_reportables.Rel_gate_eigenvalues(model, target_model, gl),
                                        confidence_region_info)
                    else:
                        rel_evals = _ev(_reportables.Rel_circuit_eigenvalues(
                            model, target_model, gl), confidence_region_info)

                # permute target eigenvalues according to min-weight matching
                _, pairs = _tools.minweight_match(evals.value(), target_evals, lambda x, y: abs(x - y))
                matched_target_evals = target_evals.copy()
                for i, j in pairs:
                    matched_target_evals[i] = target_evals[j]
                target_evals = matched_target_evals
                target_evals = target_evals.reshape(evals.value().shape)
                # b/c evals have shape (x,1) and targets (x,),
                # which causes problems when we try to subtract them

            for disp in display:
                if disp == "evals":
                    row_data.append(evals)
                    row_formatters.append('Normal')

                elif disp == "target" and target_model is not None:
                    row_data.append(target_evals)
                    row_formatters.append('Normal')

                elif disp == "rel" and target_model is not None:
                    row_data.append(rel_evals)
                    row_formatters.append('Normal')

                elif disp == "log-evals":
                    logevals = evals.log()
                    row_data.append(logevals.real())
                    row_data.append(logevals.imag() / _np.pi)
                    row_formatters.append('Normal')
                    row_formatters.append('Pi')

                elif disp == "log-rel":
                    log_relevals = rel_evals.log()
                    row_data.append(log_relevals.real())
                    row_data.append(log_relevals.imag() / _np.pi)
                    row_formatters.append('Vec')
                    row_formatters.append('Pi')

                elif disp == "absdiff-evals" and target_model is not None:
                    absdiff_evals = evals.absdiff(target_evals)
                    row_data.append(absdiff_evals)
                    row_formatters.append('Vec')

                elif disp == "infdiff-evals" and target_model is not None:
                    infdiff_evals = evals.infidelity_diff(target_evals)
                    row_data.append(infdiff_evals)
                    row_formatters.append('Vec')

                elif disp == "absdiff-log-evals" and target_model is not None:
                    log_evals = evals.log()
                    re_diff, im_diff = log_evals.absdiff(_np.log(target_evals.astype(complex)), separate_re_im=True)
                    row_data.append(re_diff)
                    row_data.append((im_diff / _np.pi).mod(2.0))
                    row_formatters.append('Vec')
                    row_formatters.append('Pi')

                elif disp == "evdm":
                    if target_model is not None:
                        fn = _reportables.Eigenvalue_diamondnorm if \
                            isinstance(gl, _objs.Label) or isinstance(gl, str) else \
                            _reportables.Circuit_eigenvalue_diamondnorm
                        gidm = _ev(fn(model, target_model, gl), confidence_region_info)
                        row_data.append(gidm)
                        row_formatters.append('Normal')

                elif disp == "evinf":
                    if target_model is not None:
                        fn = _reportables.Eigenvalue_entanglement_infidelity if \
                            isinstance(gl, _objs.Label) or isinstance(gl, str) else \
                            _reportables.Circuit_eigenvalue_entanglement_infidelity
                        giinf = _ev(fn(model, target_model, gl), confidence_region_info)
                        row_data.append(giinf)
                        row_formatters.append('Normal')

                elif disp == "polar":
                    evals_val = evals.value()
                    if target_model is None:
                        fig = _wp.PolarEigenvaluePlot(
                            self.ws, [evals_val], ["blue"], center_text=str(gl))
                    else:
                        fig = _wp.PolarEigenvaluePlot(
                            self.ws, [target_evals, evals_val],
                            ["black", "blue"], ["target", "gate"], center_text=str(gl))
                    row_data.append(fig)
                    row_formatters.append('Figure')

                elif disp == "relpolar" and target_model is not None:
                    rel_evals_val = rel_evals.value()
                    fig = _wp.PolarEigenvaluePlot(
                        self.ws, [rel_evals_val], ["red"], ["rel"], center_text=str(gl))
                    row_data.append(fig)
                    row_formatters.append('Figure')
            table.add_row(row_data, row_formatters)

        #Iterate over instruments
        for il, inst in model.instruments.items():
            tinst = target_model.instruments[il]
            for comp_lbl, comp in inst.items():
                tcomp = tinst[comp_lbl]

                row_data = [il + "." + comp_lbl]
                row_formatters = [None]

                #FUTURE: use reportables to get instrument eigenvalues
                evals = _objs.reportableqty.ReportableQty(_np.linalg.eigvals(comp.to_dense()))
                evals = evals.reshape(evals.size, 1)

                if target_model is not None:
                    target_evals = _np.linalg.eigvals(tcomp.to_dense())  # no error bars
                    #Note: no support for relative eigenvalues of instruments (yet)

                    # permute target eigenvalues according to min-weight matching
                    _, pairs = _tools.minweight_match(evals.value(), target_evals, lambda x, y: abs(x - y))
                    matched_target_evals = target_evals.copy()
                    for i, j in pairs:
                        matched_target_evals[i] = target_evals[j]
                    target_evals = matched_target_evals
                    target_evals = target_evals.reshape(evals.value().shape)
                    # b/c evals have shape (x,1) and targets (x,),
                    # which causes problems when we try to subtract them

                for disp in display:
                    if disp == "evals":
                        row_data.append(evals)
                        row_formatters.append('Normal')

                    elif disp == "target" and target_model is not None:
                        row_data.append(target_evals)
                        row_formatters.append('Normal')

                    elif disp == "rel" and target_model is not None:
                        row_data.append(_np.nan)
                        row_formatters.append('Normal')

                    elif disp == "log-evals":
                        logevals = evals.log()
                        row_data.append(logevals.real())
                        row_data.append(logevals.imag() / _np.pi)
                        row_formatters.append('Normal')
                        row_formatters.append('Pi')

                    elif disp == "log-rel":
                        row_data.append(_np.nan)
                        row_formatters.append('Normal')

                    elif disp == "absdiff-evals":
                        absdiff_evals = evals.absdiff(target_evals)
                        row_data.append(absdiff_evals)
                        row_formatters.append('Vec')

                    elif disp == "infdiff-evals":
                        infdiff_evals = evals.infidelity_diff(target_evals)
                        row_data.append(infdiff_evals)
                        row_formatters.append('Vec')

                    elif disp == "absdiff-log-evals":
                        log_evals = evals.log()
                        re_diff, im_diff = log_evals.absdiff(_np.log(target_evals.astype(complex)), separate_re_im=True)
                        row_data.append(re_diff)
                        row_data.append((im_diff / _np.pi).mod(2.0))
                        row_formatters.append('Vec')
                        row_formatters.append('Pi')

                    elif disp == "evdm":
                        row_data.append(_np.nan)
                        row_formatters.append('Normal')

                    elif disp == "evinf":
                        row_data.append(_np.nan)
                        row_formatters.append('Normal')

                    elif disp == "polar":
                        evals_val = evals.value()
                        if target_model is None:
                            fig = _wp.PolarEigenvaluePlot(
                                self.ws, [evals_val], ["blue"], center_text=str(gl))
                        else:
                            fig = _wp.PolarEigenvaluePlot(
                                self.ws, [target_evals, evals_val],
                                ["black", "blue"], ["target", "gate"], center_text=str(gl))
                        row_data.append(fig)
                        row_formatters.append('Figure')

                    elif disp == "relpolar" and target_model is not None:
                        row_data.append(_np.nan)
                        row_formatters.append('Normal')
                        row_formatters.append('Figure')

                table.add_row(row_data, row_formatters)

        table.finish()
        return table


class DataSetOverviewTable(WorkspaceTable):
    """
    Table giving a summary of the properties of `dataset`.

    Parameters
    ----------
    ws : Workspace
        The containing (parent) workspace.

    dataset : DataSet
        The DataSet

    max_length_list : list of ints, optional
        A list of the maximum lengths used, if available.
    """

    def __init__(self, ws, dataset, max_length_list=None):
        """
        Create a table that gives a summary of the properties of `dataset`.

        Parameters
        ----------
        dataset : DataSet
            The DataSet

        max_length_list : list of ints, optional
            A list of the maximum lengths used, if available.

        Returns
        -------
        ReportTable
        """
        super(DataSetOverviewTable, self).__init__(ws, self._create, dataset, max_length_list)

    def _create(self, dataset, max_length_list):

        colHeadings = ('Quantity', 'Value')
        formatters = (None, None)

        table = _ReportTable(colHeadings, formatters)

        minN = round(min([row.total for row in dataset.values()]))
        maxN = round(max([row.total for row in dataset.values()]))
        cntStr = "[%d,%d]" % (minN, maxN) if (minN != maxN) else "%d" % round(minN)

        table.add_row(("Number of strings", str(len(dataset))), (None, None))
        table.add_row(("Gate labels", ", ".join([str(gl) for gl in dataset.gate_labels()])), (None, None))
        table.add_row(("Outcome labels", ", ".join(map(str, dataset.outcome_labels()))), (None, None))
        table.add_row(("Counts per string", cntStr), (None, None))

        if max_length_list is not None:
            table.add_row(("Max. Lengths", ", ".join(map(str, max_length_list))), (None, None))
        if hasattr(dataset, 'comment') and dataset.comment is not None:
            commentLines = dataset.comment.split('\n')
            for i, commentLine in enumerate(commentLines, start=1):
                table.add_row(("User comment %d" % i, commentLine), (None, 'Verbatim'))

        table.finish()
        return table


class FitComparisonTable(WorkspaceTable):
    """
    Table showing how the goodness-of-fit evolved over GST iterations

    Parameters
    ----------
    ws : Workspace
        The containing (parent) workspace.

    xs : list of integers
        List of X-values. Typically these are the maximum lengths or
        exponents used to index the different iterations of GST.

    circuits_by_x : list of (CircuitLists or lists of Circuits)
        Specifies the set of circuits used at each X.

    model_by_x : list of Models
        `Model`s corresponding to each X value.

    dataset : DataSet
        The data set to compare each model against.

    objfn_builder : ObjectiveFunctionBuilder or {"logl", "chi2"}, optional
        The objective function to use, or one of the given strings
        to use a defaut log-likelihood or chi^2 function.

    x_label : str, optional
        A label for the 'X' variable which indexes the different models.
        This string will be the header of the first table column.

    np_by_x : list of ints, optional
        A list of parameter counts to use for each X.  If None, then
        the number of non-gauge parameters for each model is used.

    comm : mpi4py.MPI.Comm, optional
        When not None, an MPI communicator for distributing the computation
        across multiple processors.

    wildcard : WildcardBudget
        A wildcard budget to apply to the objective function (`objective`),
        which increases the goodness of fit by adjusting (by an amount
        measured in TVD) the probabilities produced by a model before
        comparing with the frequencies in `dataset`.  Currently, this
        functionality is only supported for `objective == "logl"`.
    """

    def __init__(self, ws, xs, circuits_by_x, model_by_x, dataset, objfn_builder='logl',
                 x_label='L', np_by_x=None, comm=None, wildcard=None):
        """
        Create a table showing how the chi^2 or log-likelihood changed with
        successive GST iterations.

        Parameters
        ----------
        xs : list of integers
            List of X-values. Typically these are the maximum lengths or
            exponents used to index the different iterations of GST.

        circuits_by_x : list of (CircuitLists or lists of Circuits)
            Specifies the set of circuits used at each X.

        model_by_x : list of Models
            `Model`s corresponding to each X value.

        dataset : DataSet
            The data set to compare each model against.

        objfn_builder : ObjectiveFunctionBuilder or {"logl", "chi2"}, optional
            The objective function to use, or one of the given strings
            to use a defaut log-likelihood or chi^2 function.

        x_label : str, optional
            A label for the 'X' variable which indexes the different models.
            This string will be the header of the first table column.

        np_by_x : list of ints, optional
            A list of parameter counts to use for each X.  If None, then
            the number of non-gauge parameters for each model is used.

        comm : mpi4py.MPI.Comm, optional
            When not None, an MPI communicator for distributing the computation
            across multiple processors.

        wildcard : WildcardBudget
            A wildcard budget to apply to the objective function (`objective`),
            which increases the goodness of fit by adjusting (by an amount
            measured in TVD) the probabilities produced by a model before
            comparing with the frequencies in `dataset`.  Currently, this
            functionality is only supported for `objective == "logl"`.

        Returns
        -------
        ReportTable
        """
        super(FitComparisonTable, self).__init__(ws, self._create, xs, circuits_by_x, model_by_x,
                                                 dataset, objfn_builder, x_label, np_by_x, comm,
                                                 wildcard)

    def _create(self, xs, circuits_by_x, model_by_x, dataset, objfn_builder, x_label, np_by_x, comm, wildcard):

        if objfn_builder == "chi2" or (isinstance(objfn_builder, _objfns.ObjectiveFunctionBuilder)
                                       and objfn_builder.cls_to_build == _objfns.Chi2Function):
            colHeadings = {
                'latex': (x_label, '$\\chi^2$', '$k$', '$\\chi^2-k$', '$\sqrt{2k}$',
                          '$N_\\sigma$', '$N_s$', '$N_p$', 'Rating'),
                'html': (x_label, '&chi;<sup>2</sup>', 'k', '&chi;<sup>2</sup>-k',
                         '&radic;<span style="text-decoration:overline;">2k</span>',
                         'N<sub>sigma</sub>', 'N<sub>s</sub>', 'N<sub>p</sub>', 'Rating'),
                'python': (x_label, 'chi^2', 'k', 'chi^2-k', 'sqrt{2k}', 'N_{sigma}', 'N_s', 'N_p', 'Rating')
            }

        elif objfn_builder == "logl" or (isinstance(objfn_builder, _objfns.ObjectiveFunctionBuilder)
                                         and objfn_builder.cls_to_build == _objfns.PoissonPicDeltaLogLFunction):
            colHeadings = {
                'latex': (x_label, '$2\Delta\\log(\\mathcal{L})$', '$k$', '$2\Delta\\log(\\mathcal{L})-k$',
                          '$\sqrt{2k}$', '$N_\\sigma$', '$N_s$', '$N_p$', 'Rating'),
                'html': (x_label, '2&Delta;(log L)', 'k', '2&Delta;(log L)-k',
                         '&radic;<span style="text-decoration:overline;">2k</span>',
                         'N<sub>sigma</sub>', 'N<sub>s</sub>', 'N<sub>p</sub>', 'Rating'),
                'python': (x_label, '2*Delta(log L)', 'k', '2*Delta(log L)-k', 'sqrt{2k}',
                           'N_{sigma}', 'N_s', 'N_p', 'Rating')
            }
        else:
            raise ValueError("Invalid `objfn_builder` argument: %s" % str(objfn_builder))

        if np_by_x is None:
            try:
                np_by_x = [mdl.num_nongauge_params() for mdl in model_by_x]
            except _np.linalg.LinAlgError:
                _warnings.warn(("LinAlgError when trying to compute the number"
                                " of non-gauge parameters.  Using total"
                                " parameters instead."))
                np_by_x = [mdl.num_params() for mdl in model_by_x]
            except (NotImplementedError, AttributeError):
                _warnings.warn(("FitComparisonTable could not obtain number of"
                                "*non-gauge* parameters - using total params instead"))
                np_by_x = [mdl.num_params() for mdl in model_by_x]

        tooltips = ('', 'Difference in logL', 'number of degrees of freedom',
                    'difference between observed logl and expected mean',
                    'std deviation', 'number of std deviation', 'dataset dof',
                    'number of model parameters', '1-5 star rating (like Netflix)')
        table = _ReportTable(colHeadings, None, col_heading_labels=tooltips)

        for X, mdl, circuit_list, Np in zip(xs, model_by_x, circuits_by_x, np_by_x):
            Nsig, rating, fitQty, k, Ns, Np = self._ccompute(
                _ph.rated_n_sigma, dataset, mdl, circuit_list,
                objfn_builder, Np, wildcard, return_all=True,
                comm=comm)  # self.ws.smartCache derived?
            table.add_row((str(X), fitQty, k, fitQty - k, _np.sqrt(2 * k), Nsig, Ns, Np, "<STAR>" * rating),
                          (None, 'Normal', 'Normal', 'Normal', 'Normal', 'Rounded', 'Normal', 'Normal', 'Conversion'))

        table.finish()
        return table


class CircuitTable(WorkspaceTable):
    """
    Table which simply displays list(s) of circuits

    Parameters
    ----------
    ws : Workspace
        The containing (parent) workspace.

    circuit_lists : Circuit list or list of Circuit lists
        List(s) of circuits to put in table.

    titles : string or list of strings
        The title(s) for the different string lists.  These are displayed in
        the relevant table columns containing the strings.

    n_cols : int, optional
        The number of *data* columns, i.e. those containing
        circuits, for each string list.

    common_title : string, optional
        A single title string to place in a cell spanning across
        all the other column headers.
    """

    def __init__(self, ws, circuit_lists, titles, n_cols=1, common_title=None):
        """
        Creates a table of enumerating one or more sets of circuits.

        Parameters
        ----------
        circuit_lists : Circuit list or list of Circuit lists
            List(s) of circuits to put in table.

        titles : string or list of strings
            The title(s) for the different string lists.  These are displayed in
            the relevant table columns containing the strings.

        n_cols : int, optional
            The number of *data* columns, i.e. those containing
            circuits, for each string list.

        common_title : string, optional
            A single title string to place in a cell spanning across
            all the other column headers.

        Returns
        -------
        ReportTable
        """
        super(CircuitTable, self).__init__(ws, self._create, circuit_lists, titles,
                                           n_cols, common_title)

    def _create(self, circuit_lists, titles, n_cols, common_title):

        if len(circuit_lists) == 0:
            circuit_lists = [[]]
        elif isinstance(circuit_lists[0], _objs.Circuit) or \
                (isinstance(circuit_lists[0], tuple) and isinstance(circuit_lists[0][0], str)):
            circuit_lists = [circuit_lists]

        if isinstance(titles, str): titles = [titles] * len(circuit_lists)

        colHeadings = (('#',) + tuple(titles)) * n_cols
        formatters = (('Conversion',) + ('Normal',) * len(titles)) * n_cols

        if common_title is None:
            table = _ReportTable(colHeadings, formatters)
        else:
            table = "tabular"
            colHeadings = ('\\#',) + tuple(titles)
            latex_head = "\\begin{%s}[l]{%s}\n\hline\n" % (table, "|c" * len(colHeadings) + "|")
            latex_head += " & \multicolumn{%d}{c|}{%s} \\\\ \hline\n" % (len(colHeadings) - 1, common_title)
            latex_head += "%s \\\\ \hline\n" % (" & ".join(colHeadings))

            colHeadings = ('#',) + tuple(titles)
            html_head = '<table class="%(tableclass)s" id="%(tableid)s" ><thead>'
            html_head += '<tr><th></th><th colspan="%d">%s</th></tr>\n' % (len(colHeadings) - 1, common_title)
            html_head += "<tr><th> %s </th></tr>" % (" </th><th> ".join(colHeadings))
            html_head += "</thead><tbody>"
            table = _ReportTable(colHeadings, formatters,
                                 custom_header={'latex': latex_head,
                                                'html': html_head})

        formatters = (('Normal',) + ('Circuit',) * len(circuit_lists)) * n_cols

        maxListLength = max(list(map(len, circuit_lists)))
        nRows = (maxListLength + (n_cols - 1)) // n_cols  # ceiling

        #for i in range( max([len(gsl) for gsl in circuit_lists]) ):
        for i in range(nRows):
            rowData = []
            for k in range(n_cols):
                l = i + nRows * k  # index of circuit
                rowData.append(l + 1)
                for gsList in circuit_lists:
                    if l < len(gsList):
                        rowData.append(gsList[l])
                    else:
                        rowData.append(None)  # empty string
            table.add_row(rowData, formatters)

        table.finish()
        return table


class GatesSingleMetricTable(WorkspaceTable):
    """
    Table that compares the gates of many models to target models using a single metric (`metric`).

    This allows the model titles to be used as the row and column headers. The models
    must share the same gate labels.

    If `models` and `target_models` are 1D lists, then `rowtitles` and
    `op_label` should be left as their default values so that the
    operation labels are used as row headers.

    If `models` and `target_models` are 2D (nested) lists, then
    `rowtitles` should specify the row-titles corresponding to the outer list
    elements and `op_label` should specify a single operation label that names
    the gate being compared throughout the entire table.

    Parameters
    ----------
    ws : Workspace
        The containing (parent) workspace.

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

    models : list
        A list or nested list-of-lists of models to compare with
        corresponding elements of `target_models`.

    target_models : list
        A list or nested list-of-lists of models to compare with
        corresponding elements of `models`.

    titles : list of strs
        A list of column titles used to describe elements of the
        innermost list(s) in `models`.

    rowtitles : list of strs, optional
        A list of row titles used to describe elements of the
        outer list in `models`.  If None, then the operation labels
        are used.

    table_title : str, optional
        If not None, text to place in a top header cell which spans all the
        columns of the table.

    op_label : str, optional
        If not None, the single operation label to use for all comparisons
        computed in this table.  This should be set when (and only when)
        `models` and `target_models` are 2D (nested) lists.

    confidence_region_info : ConfidenceRegion, optional
        If not None, specifies a confidence-region
        used to display error intervals.
    """

    def __init__(self, ws, metric, models, target_models, titles,
                 rowtitles=None, table_title=None, op_label=None,
                 confidence_region_info=None):
        """
        Create a table comparing the gates of various models (`models`) to
        those of `target_models` using the metric named by `metric`.

        If `models` and `target_models` are 1D lists, then `rowtitles` and
        `op_label` should be left as their default values so that the
        operation labels are used as row headers.

        If `models` and `target_models` are 2D (nested) lists, then
        `rowtitles` should specify the row-titles corresponding to the outer list
        elements and `op_label` should specify a single operation label that names
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

        models : list
            A list or nested list-of-lists of models to compare with
            corresponding elements of `target_models`.

        target_models : list
            A list or nested list-of-lists of models to compare with
            corresponding elements of `models`.

        titles : list of strs
            A list of column titles used to describe elements of the
            innermost list(s) in `models`.

        rowtitles : list of strs, optional
            A list of row titles used to describe elements of the
            outer list in `models`.  If None, then the operation labels
            are used.

        table_title : str, optional
            If not None, text to place in a top header cell which spans all the
            columns of the table.

        op_label : str, optional
            If not None, the single operation label to use for all comparisons
            computed in this table.  This should be set when (and only when)
            `models` and `target_models` are 2D (nested) lists.

        confidence_region_info : ConfidenceRegion, optional
            If not None, specifies a confidence-region
            used to display error intervals.

        Returns
        -------
        ReportTable
        """
        super(GatesSingleMetricTable, self).__init__(
            ws, self._create, metric, models, target_models, titles,
            rowtitles, table_title, op_label, confidence_region_info)

    def _create(self, metric, models, target_models, titles,
                rowtitles, table_title, op_label, confidence_region_info):

        if rowtitles is None:
            assert(op_label is None), "`op_label` must be None when `rowtitles` is"
            colHeadings = ("Gate",) + tuple(titles)
        else:
            colHeadings = ("",) + tuple(titles)

        nCols = len(colHeadings)
        formatters = [None] * nCols  # [None] + ['ModelType']*(nCols-1)

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

        if table_title:
            latex_head = "\\begin{tabular}[l]{%s}\n\hline\n" % ("|c" * nCols + "|")
            latex_head += "\\multicolumn{%d}{c|}{%s} \\\\ \cline{1-%d}\n" % (nCols, table_title, nCols)
            latex_head += " & ".join(colHeadings) + "\\\\ \hline\n"

            html_head = '<table class="%(tableclass)s" id="%(tableid)s" ><thead>'
            html_head += '<tr><th colspan="%d">%s</th></tr>\n' % (nCols, table_title)
            html_head += "<tr><th>" + " </th><th> ".join(colHeadings) + "</th></tr>\n"
            html_head += "</thead><tbody>"

            table = _ReportTable(colHeadings, formatters,
                                 custom_header={'latex': latex_head,
                                                'html': html_head})
        else:
            table = _ReportTable(colHeadings, formatters)

        row_formatters = [None] + ['Normal'] * len(titles)

        if rowtitles is None:
            assert(isinstance(target_models[0], _objs.ExplicitOpModel)
                   ), "%s only works with explicit models" % str(type(self))
            for gl in target_models[0].operations:  # use first target's operation labels
                row_data = [gl]
                for mdl, gsTarget in zip(models, target_models):
                    if mdl is None or gsTarget is None:
                        qty = _objs.reportableqty.ReportableQty(_np.nan)
                    else:
                        qty = _reportables.evaluate_opfn_by_name(
                            metric, mdl, gsTarget, gl, confidence_region_info)
                    row_data.append(qty)
                table.add_row(row_data, row_formatters)
        else:
            for rowtitle, gsList, tgsList in zip(rowtitles, models, target_models):
                row_data = [rowtitle]
                for mdl, gsTarget in zip(gsList, tgsList):
                    if mdl is None or gsTarget is None:
                        qty = _objs.reportableqty.ReportableQty(_np.nan)
                    else:
                        qty = _reportables.evaluate_opfn_by_name(
                            metric, mdl, gsTarget, op_label, confidence_region_info)
                    row_data.append(qty)
                table.add_row(row_data, row_formatters)

        table.finish()
        return table


class StandardErrgenTable(WorkspaceTable):
    """
    A table showing what the standard error generators' superoperator matrices look like.

    Parameters
    ----------
    ws : Workspace
        The containing (parent) workspace.

    model_dim : int
        The dimension of the model, which equals the number of
        rows (or columns) in a operation matrix (e.g., 4 for a single qubit).

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
    """

    def __init__(self, ws, model_dim, projection_type,
                 projection_basis):
        """
        Create a table of the "standard" gate error generators, such as those
        which correspond to Hamiltonian or Stochastic errors.  Each generator
        is shown as grid of colored boxes.

        Parameters
        ----------
        model_dim : int
            The dimension of the model, which equals the number of
            rows (or columns) in a operation matrix (e.g., 4 for a single qubit).

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
        super(StandardErrgenTable, self).__init__(
            ws, self._create, model_dim, projection_type,
            projection_basis)

    def _create(self, model_dim, projection_type,
                projection_basis):

        d2 = model_dim  # number of projections == dim of gate
        d = int(_np.sqrt(d2))  # dim of density matrix
        nQubits = _np.log2(d)

        #Get a list of the d2 generators (in corresspondence with the
        #  given basis matrices)
        lindbladMxs = _tools.std_error_generators(d2, projection_type,
                                                  projection_basis)  # in std basis

        if not _np.isclose(round(nQubits), nQubits):
            #Non-integral # of qubits, so just show as a single row
            yd, xd = 1, d
            xlabel = ""; ylabel = ""
        elif nQubits == 1:
            yd, xd = 1, 2  # y and x pauli-prod *basis* dimensions
            xlabel = "Q1"; ylabel = ""
        elif nQubits == 2:
            yd, xd = 2, 2
            xlabel = "Q2"; ylabel = "Q1"
        else:
            assert(d % 2 == 0)
            yd, xd = 2, d // 2
            xlabel = "Q*"; ylabel = "Q1"

        topright = "%s \\ %s" % (ylabel, xlabel) if (len(ylabel) > 0) else ""
        colHeadings = [topright] + \
            [("%s" % x) if len(x) else ""
             for x in _tools.basis_element_labels(projection_basis, xd**2)]
        rowLabels = [("%s" % x) if len(x) else ""
                     for x in _tools.basis_element_labels(projection_basis, yd**2)]

        xLabels = _tools.basis_element_labels(projection_basis, xd**2)
        yLabels = _tools.basis_element_labels(projection_basis, yd**2)

        table = _ReportTable(colHeadings, ["Conversion"] + [None] * (len(colHeadings) - 1))

        iCur = 0
        for i, ylabel in enumerate(yLabels):
            rowData = [rowLabels[i]]
            rowFormatters = [None]

            for xlabel in xLabels:
                projector = lindbladMxs[iCur]; iCur += 1
                projector = _tools.change_basis(projector, "std", projection_basis)
                m, M = -_np.max(_np.abs(projector)), _np.max(_np.abs(projector))
                fig = _wp.GateMatrixPlot(self.ws, projector, m, M,
                                         projection_basis, d)
                rowData.append(fig)
                rowFormatters.append('Figure')

            table.add_row(rowData, rowFormatters)

        table.finish()
        return table


class GaugeOptParamsTable(WorkspaceTable):
    """
    Table of gauge optimization parameters

    Parameters
    ----------
    ws : Workspace
        The containing (parent) workspace.

    gauge_opt_args : dict or list
        A dictionary or list of dictionaries specifying values for
        zero or more of the *arguments* of pyGSTi's
        :func:`gaugeopt_to_target` function.
    """

    def __init__(self, ws, gauge_opt_args):
        """
        Create a table displaying a list of gauge
        optimzation parameters.

        Parameters
        ----------
        gauge_opt_args : dict or list
            A dictionary or list of dictionaries specifying values for
            zero or more of the *arguments* of pyGSTi's
            :func:`gaugeopt_to_target` function.

        Returns
        -------
        ReportTable
        """
        super(GaugeOptParamsTable, self).__init__(ws, self._create, gauge_opt_args)

    def _create(self, gauge_opt_args):

        colHeadings = ('G-Opt Param', 'Value')
        formatters = ('Bold', 'Bold')

        if gauge_opt_args is False:  # signals *no* gauge optimization
            goargs_list = [{'Method': "No gauge optimization was performed"}]
        else:
            goargs_list = [gauge_opt_args] if hasattr(gauge_opt_args, 'keys') \
                else gauge_opt_args

        table = _ReportTable(colHeadings, formatters)

        for i, goargs in enumerate(goargs_list):
            pre = ("%d: " % i) if len(goargs_list) > 1 else ""
            if 'method' in goargs:
                table.add_row(("%sMethod" % pre, str(goargs['method'])), (None, None))
            if 'cptp_penalty_factor' in goargs and goargs['cptp_penalty_factor'] != 0:
                table.add_row(("%sCP penalty factor" % pre, str(goargs['cptp_penalty_factor'])), (None, None))
            if 'spam_penalty_factor' in goargs and goargs['spam_penalty_factor'] != 0:
                table.add_row(("%sSPAM penalty factor" % pre, str(goargs['spam_penalty_factor'])), (None, None))
            if 'gates_metric' in goargs:
                table.add_row(("%sMetric for gate-to-target" % pre, str(goargs['gates_metric'])), (None, None))
            if 'spam_metric' in goargs:
                table.add_row(("%sMetric for SPAM-to-target" % pre, str(goargs['spam_metric'])), (None, None))
            if 'item_weights' in goargs:
                if goargs['item_weights']:
                    table.add_row(
                        ("%sItem weights" % pre,
                         ", ".join([("%s=%.2g" % (k, v)) for k, v in goargs['item_weights'].items()])), (None, None))
            if 'gauge_group' in goargs:
                table.add_row(("%sGauge group" % pre, goargs['gauge_group'].name), (None, None))

        table.finish()
        return table


class MetadataTable(WorkspaceTable):
    """
    Table of raw parameters, often taken directly from a `Results` object

    Parameters
    ----------
    ws : Workspace
        The containing (parent) workspace.

    model : Model
        The model (usually the final estimate of a GST computation) to
        show information for (e.g. the types of its gates).

    params: dict
        A parameter dictionary to display
    """

    def __init__(self, ws, model, params):
        """
        Create a table of parameters and options from a `Results` object.

        Parameters
        ----------
        model : Model
            The model (usually the final estimate of a GST computation) to
            show information for (e.g. the types of its gates).

        params: dict
            A parameter dictionary to display

        Returns
        -------
        ReportTable
        """
        super(MetadataTable, self).__init__(ws, self._create, model, params)

    def _create(self, model, params_dict):

        colHeadings = ('Quantity', 'Value')
        formatters = ('Bold', 'Bold')

        #custom latex header for maximum width imposed on 2nd col
        latex_head = "\\begin{tabular}[l]{|c|p{3in}|}\n\hline\n"
        latex_head += "\\textbf{Quantity} & \\textbf{Value} \\\\ \hline\n"
        table = _ReportTable(colHeadings, formatters,
                             custom_header={'latex': latex_head})

        for key in sorted(list(params_dict.keys())):
            if key in ['L,germ tuple base string dict', 'weights', 'profiler']: continue  # skip these
            if key == 'gaugeOptParams':
                if isinstance(params_dict[key], dict):
                    val = params_dict[key].copy()
                    if 'targetModel' in val:
                        del val['targetModel']  # don't print this!

                elif isinstance(params_dict[key], list):
                    val = []
                    for go_param_dict in params_dict[key]:
                        if isinstance(go_param_dict, dict):  # to ensure .copy() exists
                            val.append(go_param_dict.copy())
                            if 'targetModel' in val[-1]:
                                del val[-1]['targetModel']  # don't print this!
            else:
                val = params_dict[key]
            table.add_row((key, str(val)), (None, 'Verbatim'))

        if isinstance(self, _objs.ExplicitOpModel):
            for lbl, vec in model.preps.items():
                if isinstance(vec, _objs.StaticSPAMVec): paramTyp = "static"
                elif isinstance(vec, _objs.FullSPAMVec): paramTyp = "full"
                elif isinstance(vec, _objs.TPSPAMVec): paramTyp = "TP"
                elif isinstance(vec, _objs.ComplementSPAMVec): paramTyp = "Comp"
                else: paramTyp = "unknown"  # pragma: no cover
                table.add_row((lbl + " parameterization", paramTyp), (None, 'Verbatim'))

            for povmlbl, povm in model.povms.items():
                if isinstance(povm, _objs.UnconstrainedPOVM): paramTyp = "unconstrained"
                elif isinstance(povm, _objs.TPPOVM): paramTyp = "TP"
                elif isinstance(povm, _objs.TensorProdPOVM): paramTyp = "TensorProd"
                else: paramTyp = "unknown"  # pragma: no cover
                table.add_row((povmlbl + " parameterization", paramTyp), (None, 'Verbatim'))

                for lbl, vec in povm.items():
                    if isinstance(vec, _objs.StaticSPAMVec): paramTyp = "static"
                    elif isinstance(vec, _objs.FullSPAMVec): paramTyp = "full"
                    elif isinstance(vec, _objs.TPSPAMVec): paramTyp = "TP"
                    elif isinstance(vec, _objs.ComplementSPAMVec): paramTyp = "Comp"
                    else: paramTyp = "unknown"  # pragma: no cover
                    table.add_row(("> " + lbl + " parameterization", paramTyp), (None, 'Verbatim'))

            for gl, gate in model.operations.items():
                if isinstance(gate, _objs.StaticDenseOp): paramTyp = "static"
                elif isinstance(gate, _objs.FullDenseOp): paramTyp = "full"
                elif isinstance(gate, _objs.TPDenseOp): paramTyp = "TP"
                elif isinstance(gate, _objs.LinearlyParamDenseOp): paramTyp = "linear"
                elif isinstance(gate, _objs.EigenvalueParamDenseOp): paramTyp = "eigenvalue"
                elif isinstance(gate, _objs.LindbladDenseOp):
                    paramTyp = "Lindblad"
                    if gate.errorgen.param_mode == "cptp": paramTyp += " CPTP "
                    paramTyp += "(%d, %d params)" % (gate.errorgen.ham_basis_size, gate.errorgen.other_basis_size)
                else: paramTyp = "unknown"  # pragma: no cover
                table.add_row((gl + " parameterization", paramTyp), (None, 'Verbatim'))

        table.finish()
        return table


class SoftwareEnvTable(WorkspaceTable):
    """
    Table showing details about the current software environment.

    Parameters
    ----------
    ws : Workspace
        The containing (parent) workspace.
    """

    def __init__(self, ws):
        """
        Create a table displaying the software environment relevant to pyGSTi.

        Returns
        -------
        ReportTable
        """
        super(SoftwareEnvTable, self).__init__(ws, self._create)

    def _create(self):

        import platform

        def get_version(module_name):
            """ Extract the current version of a python module """
            if module_name == "cvxopt":
                #special case b/c cvxopt can be weird...
                try:
                    mod = __import__("cvxopt.info")
                    return str(mod.info.version)
                except Exception: pass  # try the normal way below

            try:
                mod = __import__(module_name)
                return str(mod.__version__)
            except ImportError:     # pragma: no cover
                return "missing"    # pragma: no cover
            except AttributeError:  # pragma: no cover
                return "ver?"       # pragma: no cover
            except Exception:       # pragma: no cover
                return "???"        # pragma: no cover

        colHeadings = ('Quantity', 'Value')
        formatters = ('Bold', 'Bold')

        #custom latex header for maximum width imposed on 2nd col
        latex_head = "\\begin{tabular}[l]{|c|p{3in}|}\n\hline\n"
        latex_head += "\\textbf{Quantity} & \\textbf{Value} \\\\ \hline\n"
        table = _ReportTable(colHeadings, formatters,
                             custom_header={'latex': latex_head})

        #Python package information
        from .._version import __version__ as pygsti_version
        table.add_row(("pyGSTi version", str(pygsti_version)), (None, 'Verbatim'))

        packages = ['numpy', 'scipy', 'matplotlib', 'ply', 'cvxopt', 'cvxpy',
                    'nose', 'PIL', 'psutil']
        for pkg in packages:
            table.add_row((pkg, get_version(pkg)), (None, 'Verbatim'))

        #Python information
        table.add_row(("Python version", str(platform.python_version())), (None, 'Verbatim'))
        table.add_row(("Python type", str(platform.python_implementation())), (None, 'Verbatim'))
        table.add_row(("Python compiler", str(platform.python_compiler())), (None, 'Verbatim'))
        table.add_row(("Python build", str(platform.python_build())), (None, 'Verbatim'))
        table.add_row(("Python branch", str(platform.python_branch())), (None, 'Verbatim'))
        table.add_row(("Python revision", str(platform.python_revision())), (None, 'Verbatim'))

        #Platform information
        (system, _, release, version, machine, processor) = platform.uname()
        table.add_row(("Platform summary", str(platform.platform())), (None, 'Verbatim'))
        table.add_row(("System", str(system)), (None, 'Verbatim'))
        table.add_row(("Sys Release", str(release)), (None, 'Verbatim'))
        table.add_row(("Sys Version", str(version)), (None, 'Verbatim'))
        table.add_row(("Machine", str(machine)), (None, 'Verbatim'))
        table.add_row(("Processor", str(processor)), (None, 'Verbatim'))

        table.finish()
        return table


class ProfilerTable(WorkspaceTable):
    """
    Table of profiler timing information

    Parameters
    ----------
    ws : Workspace
        The containing (parent) workspace.

    profiler : Profiler
        The profiler object to extract timings from.

    sort_by : {"time", "name"}
        What the timer values should be sorted by.
    """

    def __init__(self, ws, profiler, sort_by="time"):
        """
        Create a table of profiler timing information.

        Parameters
        ----------
        profiler : Profiler
            The profiler object to extract timings from.

        sort_by : {"time", "name"}
            What the timer values should be sorted by.
        """
        super(ProfilerTable, self).__init__(ws, self._create, profiler, sort_by)

    def _create(self, profiler, sort_by):

        colHeadings = ('Label', 'Time (sec)')
        formatters = ('Bold', 'Bold')

        #custom latex header for maximum width imposed on 2nd col
        latex_head = "\\begin{tabular}[l]{|c|p{3in}|}\n\hline\n"
        latex_head += "\\textbf{Label} & \\textbf{Time} (sec) \\\\ \hline\n"
        table = _ReportTable(colHeadings, formatters,
                             custom_header={'latex': latex_head})

        if profiler is not None:
            if sort_by == "name":
                timerNames = sorted(list(profiler.timers.keys()))
            elif sort_by == "time":
                timerNames = sorted(list(profiler.timers.keys()),
                                    key=lambda x: -profiler.timers[x])
            else:
                raise ValueError("Invalid 'sort_by' argument: %s" % sort_by)

            for nm in timerNames:
                table.add_row((nm, profiler.timers[nm]), (None, None))

        table.finish()
        return table


class WildcardBudgetTable(WorkspaceTable):
    """
    Table of wildcard budget information.

    Parameters
    ----------
    ws : Workspace
        The containing (parent) workspace.

    budget : WildcardBudget
        The wildcard budget object to extract timings from.
    """

    def __init__(self, ws, budget):
        """
        Create a table of wildcard budget information.

        Parameters
        ----------
        budget : WildcardBudget
            The wildcard budget object to extract timings from.
        """
        super(WildcardBudgetTable, self).__init__(ws, self._create, budget)

    def _create(self, budget):

        colHeadings = ('Element', 'Description', 'Budget')
        formatters = ('Bold', 'Bold', 'Bold')

        #custom latex header for maximum width imposed on 2nd col
        table = _ReportTable(colHeadings, formatters)

        if budget is not None:
            for nm, (desc, val) in budget.description().items():
                table.add_row((nm, desc, val), (None, None, None))

        table.finish()
        return table


class ExampleTable(WorkspaceTable):
    """
    Table used just as an example of what tables can do/look like for use within the "Help" section of reports.

    Parameters
    ----------
    ws : Workspace
        The containing (parent) workspace.
    """

    def __init__(self, ws):
        """A table showing how to use table features."""
        super(ExampleTable, self).__init__(ws, self._create)

    def _create(self):
        colHeadings = ["Hover over me...", "And me!", "Click the pig"]
        tooltips = ["This tooltip can give more information about what this column displays",
                    "Unfortunately, we can't show nicely formatted math in these tooltips (yet)",
                    "Click on the pyGSTi logo below to create the non-automatically-generated plot; "
                    "then hover over the colored boxes."]
        example_mx = _np.array([[1.0, 1 / 3, -1 / 3, -1.0],
                                [1 / 3, 1.0, 0.0, -1 / 5],
                                [-1 / 3, 0.0, 1.0, 1 / 6],
                                [-1.0, -1 / 5, 1 / 6, 1.0]])
        example_ebmx = _np.abs(example_mx) * 0.05
        example_fig = _wp.GateMatrixPlot(self.ws, example_mx, -1.0, 1.0,
                                         "pp", eb_matrix=example_ebmx)

        table = _ReportTable(colHeadings, None, col_heading_labels=tooltips)
        table.add_row(("Pi", _np.pi, example_fig), ('Normal', 'Normal', 'Figure'))
        table.finish()
        return table

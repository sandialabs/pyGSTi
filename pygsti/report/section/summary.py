""" Summary section """
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

from pygsti.report.section import Section as _Section

from pygsti.report import workspace as _ws


class SummarySection(_Section):
    _HTML_TEMPLATE = 'tabs/Summary.html'

    @_Section.figure_factory()
    def final_model_fit_progress_bar_plot_sum(workspace, switchboard=None, max_lengths=None, comm=None, **kwargs):
        return workspace.FitComparisonBarPlot(
            max_lengths, switchboard.circuits_all, switchboard.mdl_all_modvi,
            switchboard.modvi_ds, switchboard.objfn_builder_modvi,
            'L', comm=comm
        )

    @_Section.figure_factory()
    def final_model_fit_histogram(workspace, switchboard=None, linlog_percentile=5, comm=None, bgcolor='white',
                                  **kwargs):
        return workspace.ColorBoxPlot(
            switchboard.objfn_builder_modvi,  # NOTE: this should objfun_builder_modvi
            switchboard.circuits_final,
            switchboard.modvi_ds, switchboard.mdl_current_modvi,
            linlg_pcntle=linlog_percentile / 100,
            typ='histogram', comm=comm, bgcolor=bgcolor
        )

    @_Section.figure_factory()
    def final_gates_vs_target_table_insummary(workspace, switchboard=None, confidence_level=None, ci_brevity=1,
                                              show_unmodeled_error=False, **kwargs):
        summary_display = ('inf', 'trace', 'diamond', 'evinf', 'evdiamond')
        wildcardBudget = None
        if show_unmodeled_error:
            summary_display += ('unmodeled',)
            wildcardBudget = switchboard.wildcard_budget_optional

        if confidence_level is not None and ci_brevity <= 1:
            cri = switchboard.cri
        else:
            cri = None

        return workspace.GatesVsTargetTable(
            switchboard.mdl_final, switchboard.mdl_target, cri,
            summary_display, None, wildcardBudget
        )

    @_Section.figure_factory()
    def final_fits_comparison_plot(workspace, switchboard=None, est_labels=None, dataset_labels=None, comm=None,
                                   **kwargs):
        # Build final_fits_comparison_plot
        # Don't display "Target" in model violation summary, as it's often
        # huge and messes up the plot scale.
        est_inds_mt = [i for i, l in enumerate(est_labels) if l != "Target"]
        est_lbls_mt = [est_labels[i] for i in est_inds_mt]  # "minus target"
        Nd = len(dataset_labels)
        Ne = len(est_inds_mt)
        grid_objfn_builder = switchboard.objfn_builder_modvi[0, 0]  # just take first one for now

        def na_to_none(x):
            return None if isinstance(x, _ws.NotApplicable) else x

        if len(dataset_labels) > 1:
            dsGrid = [[na_to_none(switchboard.modvi_ds[d, i]) for i in est_inds_mt]
                      for d in range(Nd)]
            circuitsGrid = [[na_to_none(switchboard.circuits_final[i])] * Ne for i in range(Nd)]
            mdlGrid = [[na_to_none(switchboard.mdl_current_modvi[d, i, -1]) for i in est_inds_mt]
                       for d in range(Nd)]
            return workspace.FitComparisonBoxPlot(
                est_lbls_mt, dataset_labels, circuitsGrid, mdlGrid, dsGrid, grid_objfn_builder,
                comm=comm
            )
        else:
            dsGrid = [na_to_none(switchboard.modvi_ds[0, i]) for i in est_inds_mt]
            circuitsGrid = [na_to_none(switchboard.circuits_final[0])] * Ne
            if switchboard.mdl_current_modvi.shape[2] == 0:  # can't use -1 index on length-0 array
                mdlGrid = [None for i in est_inds_mt]
            else:
                mdlGrid = [na_to_none(switchboard.mdl_current_modvi[0, i, -1]) for i in est_inds_mt]
            return workspace.FitComparisonBarPlot(
                est_lbls_mt, circuitsGrid, mdlGrid, dsGrid, grid_objfn_builder, 'Estimate',
                comm=comm
            )

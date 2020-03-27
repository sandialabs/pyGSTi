""" Summary section """
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

from . import Section as _Section

from .. import workspace as _ws


class SummarySection(_Section):
    _HTML_TEMPLATE = 'tabs/Summary.html'

    @_Section.figure_factory()
    def final_model_fit_progress_bar_plot_sum(workspace, switchboard=None, max_lengths=None, comm=None, **kwargs):
        return workspace.FitComparisonBarPlot(
            max_lengths, switchboard.gssAllL, switchboard.gsAllL_modvi,
            switchboard.modvi_ds, switchboard.objective_modvi,
            'L', comm=comm, min_prob_clip=switchboard.mpc_modvi
        )

    @_Section.figure_factory()
    def final_model_fit_histogram(workspace, switchboard=None, linlog_percentile=5, comm=None, bgcolor='white',
                                  **kwargs):
        return workspace.ColorBoxPlot(
            switchboard.objective, switchboard.gss,
            switchboard.modvi_ds, switchboard.gsL_modvi,
            linlg_pcntle=linlog_percentile / 100,
            min_prob_clip_for_weighting=switchboard.mpc_modvi,
            typ='histogram', comm=comm, bgcolor=bgcolor
        )

    @_Section.figure_factory()
    def final_gates_vs_target_table_insummary(workspace, switchboard=None, confidence_level=None, ci_brevity=1,
                                              show_unmodeled_error=False, **kwargs):
        summary_display = ('inf', 'trace', 'diamond', 'evinf', 'evdiamond')
        wildcardBudget = None
        if show_unmodeled_error:
            summary_display += ('unmodeled',)
            wildcardBudget = switchboard.wildcardBudgetOptional

        if confidence_level is not None and ci_brevity <= 1:
            cri = switchboard.cri
        else:
            cri = None

        return workspace.GatesVsTargetTable(
            switchboard.gsFinal, switchboard.gsTarget, cri,
            summary_display, wildcardBudget
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
        grid_objective = switchboard.objective_modvi[0, 0]  # just take first one for now

        def na_to_none(x):
            return None if isinstance(x, _ws.NotApplicable) else x

        if len(dataset_labels) > 1:
            dsGrid = [[na_to_none(switchboard.modvi_ds[d, i]) for i in est_inds_mt]
                      for d in range(Nd)]
            gssGrid = [[na_to_none(switchboard.gssFinal[i])] * Ne for i in range(Nd)]
            gsGrid = [[na_to_none(switchboard.gsL_modvi[d, i, -1]) for i in est_inds_mt]
                      for d in range(Nd)]
            return workspace.FitComparisonBoxPlot(
                est_lbls_mt, dataset_labels, gssGrid, gsGrid, dsGrid, grid_objective,
                comm=comm, min_prob_clip=switchboard.mpc_modvi
            )
        else:
            dsGrid = [na_to_none(switchboard.modvi_ds[0, i]) for i in est_inds_mt]
            gssGrid = [na_to_none(switchboard.gssFinal[0])] * Ne
            if switchboard.gsL_modvi.shape[2] == 0:  # can't use -1 index on length-0 array
                gsGrid = [None for i in est_inds_mt]
            else:
                gsGrid = [na_to_none(switchboard.gsL_modvi[0, i, -1]) for i in est_inds_mt]
            return workspace.FitComparisonBarPlot(
                est_lbls_mt, gssGrid, gsGrid, dsGrid, grid_objective, 'Estimate',
                comm=comm, min_prob_clip=switchboard.mpc_modvi
            )

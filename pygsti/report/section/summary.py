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

    def __init__(self, workspace, dataset_labels, estimate_labels, Ls,
                 switchboard, cri, linlog_percentile,
                 show_unmodeled_error, bgcolor, comm=None):
        # TODO bgcolor really should not be needed until render time...

        # Build finalFitComparePlot
        # Don't display "Target" in model violation summary, as it's often
        # huge and messes up the plot scale.
        est_inds_mt = [i for i, l in enumerate(estimate_labels) if l != "Target"]
        est_lbls_mt = [estimate_labels[i] for i in est_inds_mt]  # "minus target"
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
            final_fit_comparison = workspace.FitComparisonBoxPlot(
                est_lbls_mt, gssGrid, gsGrid, dsGrid, grid_objective, comm=comm
            )
        else:
            dsGrid = [na_to_none(switchboard.modvi_ds[0, i]) for i in est_inds_mt]
            gssGrid = [na_to_none(switchboard.gssFinal[0])] * Ne
            gsGrid = [na_to_none(switchboard.gsL_modvi[0, i, -1]) for i in est_inds_mt]
            final_fit_comparison = workspace.FitComparisonBarPlot(
                est_lbls_mt, gssGrid, gsGrid, dsGrid, grid_objective, 'Estimate', comm=comm
            )

        summary_display = ('inf', 'trace', 'diamond', 'evinf', 'evdiamond'); wildcardBudget = None
        if show_unmodeled_error:
            summary_display += ('unmodeled',)
            wildcardBudget = switchboard.wildcardBudget
        best_gates_vs_target = workspace.GatesVsTargetTable(
            switchboard.gsFinal, switchboard.gsTarget, cri,
            summary_display, wildcardBudget
        )

        super().__init__({
            'finalFitComparePlot': final_fit_comparison,
            'progressBarPlot_sum': workspace.FitComparisonBarPlot(
                Ls, switchboard.gssAllL, switchboard.gsAllL_modvi,
                switchboard.modvi_ds, switchboard.objective_modvi,
                'L', comm=comm
            ),
            'bestEstimateColorHistogram': workspace.ColorBoxPlot(
                switchboard.objective, switchboard.gss,
                switchboard.modvi_ds, switchboard.gsL_modvi,
                linlog_percentile / 100,
                minProbClipForWeighting=switchboard.mpc_modvi,
                typ='histogram', comm=comm, bgcolor=bgcolor
            ),
            'bestGatesVsTargetTable_sum': best_gates_vs_target
        })

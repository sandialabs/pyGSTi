""" Goodness sections """
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

from . import Section as _Section


class GoodnessSection(_Section):
    _HTML_TEMPLATE = 'tabs/Goodness.html'

    @_Section.figure_factory(1)
    def final_model_fit_colorscatter_plot(workspace, switchboard=None, linlog_percentile=5, comm=None, bgcolor='white',
                                          **kwargs):
        return workspace.ColorBoxPlot(
            switchboard.objective, switchboard.gss,
            switchboard.modvi_ds, switchboard.gsL_modvi,
            linlg_pcntle=linlog_percentile / 100,
            min_prob_clip_for_weighting=switchboard.mpc_modvi,
            typ="scatter", comm=comm, bgcolor=bgcolor
        )

    @_Section.figure_factory(4)
    def final_model_fit_progress_table(workspace, switchboard=None, max_lengths=None, comm=None, **kwargs):
        return workspace.FitComparisonTable(
            max_lengths, switchboard.gssAllL, switchboard.gsAllL_modvi,
            switchboard.modvi_ds, switchboard.objective_modvi, 'L',
            comm=comm, min_prob_clip=switchboard.mpc_modvi
        )

    @_Section.figure_factory(4)
    def final_model_fit_progress_bar_plot(workspace, switchboard=None, max_lengths=None, comm=None, **kwargs):
        return workspace.FitComparisonBarPlot(
            max_lengths, switchboard.gssAllL, switchboard.gsAllL_modvi,
            switchboard.modvi_ds, switchboard.objective_modvi, 'L',
            comm=comm, min_prob_clip=switchboard.mpc_modvi
        )


class GoodnessColorBoxPlotSection(_Section):
    _HTML_TEMPLATE = 'tabs/Goodness_colorboxplot.html'

    @_Section.figure_factory(1)
    def final_model_fit_colorbox_plot(workspace, switchboard=None, linlog_percentile=5, brevity=0, comm=None,
                                      bgcolor='white', **kwargs):
        qty = workspace.ColorBoxPlot(
            switchboard.objective, switchboard.gss,
            switchboard.modvi_ds, switchboard.gsL_modvi,
            linlg_pcntle=linlog_percentile / 100,
            min_prob_clip_for_weighting=switchboard.mpc_modvi, comm=comm,
            bgcolor=bgcolor
        )
        if brevity < 1:
            qty.set_render_options(click_to_display=False, valign='bottom')
        return qty

    @_Section.figure_factory(1)
    def final_model_tvd_colorbox_plot(workspace, switchboard=None, brevity=0, comm=None, bgcolor='white', **kwargs):
        qty = workspace.ColorBoxPlot(
            'tvd', switchboard.gss, switchboard.modvi_ds, switchboard.gsL_modvi, comm=comm, bgcolor=bgcolor
        )
        if brevity < 1:
            qty.set_render_options(click_to_display=False, valign='bottom')
        return qty

    @_Section.figure_factory()
    def maxlength_switchboard1(workspace, switchboard=None, switchbd_maxlengths=None, **kwargs):
        maxLView = [False, False, False, len(switchbd_maxlengths) > 1]
        return switchboard.view(maxLView, 'v6')


class GoodnessScalingSection(_Section):
    _HTML_TEMPLATE = 'tabs/Goodness_scaling.html'

    @_Section.figure_factory(1)
    def final_model_fit_colorscatter_plot_scl(workspace, switchboard=None, linlog_percentile=5, comm=None,
                                              bgcolor='white', **kwargs):
        return workspace.ColorBoxPlot(
            switchboard.objective, switchboard.gss,
            switchboard.eff_ds, switchboard.gsL,
            linlg_pcntle=linlog_percentile / 100,
            min_prob_clip_for_weighting=switchboard.mpc,
            typ="scatter", comm=comm, bgcolor=bgcolor
        )

    @_Section.figure_factory(1)
    def final_model_fit_colorbox_plot_scl(workspace, switchboard=None, linlog_percentile=5, brevity=0, comm=None,
                                          bgcolor='white', **kwargs):
        qty = workspace.ColorBoxPlot(
            switchboard.objective, switchboard.gss,
            switchboard.eff_ds, switchboard.gsL,
            linlg_pcntle=linlog_percentile / 100,
            min_prob_clip_for_weighting=switchboard.mpc, comm=comm,
            bgcolor=bgcolor
        )
        if brevity < 1:
            qty.set_render_options(click_to_display=False, valign='bottom')
        return qty

    @_Section.figure_factory()
    def final_model_fit_histogram_scl(workspace, switchboard=None, linlog_percentile=5, comm=None, bgcolor='white',
                                      **kwargs):
        return workspace.ColorBoxPlot(
            switchboard.objective, switchboard.gss,
            switchboard.eff_ds, switchboard.gsL,
            linlg_pcntle=linlog_percentile / 100,
            min_prob_clip_for_weighting=switchboard.mpc,
            typ="histogram", comm=comm, bgcolor=bgcolor
        )

    @_Section.figure_factory(4)
    def final_model_fit_progress_table_scl(workspace, switchboard=None, max_lengths=None, comm=None, **kwargs):
        return workspace.FitComparisonTable(
            max_lengths, switchboard.gssAllL, switchboard.gsAllL,
            switchboard.eff_ds, switchboard.objective, 'L', comm=comm, min_prob_clip=switchboard.mpc
        )

    @_Section.figure_factory(4)
    def final_model_fit_progress_bar_plot_scl(workspace, switchboard=None, max_lengths=None, comm=None, **kwargs):
        return workspace.FitComparisonBarPlot(
            max_lengths, switchboard.gssAllL, switchboard.gsAllL,
            switchboard.eff_ds, switchboard.objective, 'L', comm=comm, min_prob_clip=switchboard.mpc
        )

    @_Section.figure_factory(1)
    def data_scaling_colorbox_plot(workspace, switchboard=None, comm=None, bgcolor='white', **kwargs):
        return workspace.ColorBoxPlot(
            'scaling', switchboard.gssFinal, switchboard.eff_ds, None,
            submatrices=switchboard.scaledSubMxsDict, comm=comm,
            bgcolor=bgcolor
        )


class GoodnessUnmodeledSection(_Section):
    _HTML_TEMPLATE = 'tabs/Goodness_unmodeled.html'

    @_Section.figure_factory(1)
    def unmodeled_error_budget_table(workspace, switchboard=None, **kwargs):
        return workspace.WildcardBudgetTable(switchboard.wildcardBudget)

    @_Section.figure_factory(4)
    def final_model_fit_progress_bar_plot_ume(workspace, switchboard=None, max_lengths=None, comm=None, **kwargs):
        return workspace.FitComparisonBarPlot(
            max_lengths, switchboard.gssAllL, switchboard.gsAllL,
            switchboard.modvi_ds, switchboard.objective, 'L',
            wildcard=switchboard.wildcardBudget, comm=comm, min_prob_clip=switchboard.mpc_modvi
        )

    @_Section.figure_factory(4)
    def final_model_fit_progress_table_ume(workspace, switchboard=None, max_lengths=None, comm=None, **kwargs):
        return workspace.FitComparisonTable(
            max_lengths, switchboard.gssAllL, switchboard.gsAllL,
            switchboard.modvi_ds, switchboard.objective, 'L',
            wildcard=switchboard.wildcardBudget, comm=comm, min_prob_clip=switchboard.mpc_modvi
        )

    @_Section.figure_factory()
    def final_model_fit_histogram_ume(workspace, switchboard=None, linlog_percentile=5, comm=None, bgcolor='white',
                                      **kwargs):
        return workspace.ColorBoxPlot(
            switchboard.objective, switchboard.gss,
            switchboard.modvi_ds, switchboard.gsL,
            linlg_pcntle=linlog_percentile / 100,
            min_prob_clip_for_weighting=switchboard.mpc, typ="histogram",
            wildcard=switchboard.wildcardBudget, comm=comm,
            bgcolor=bgcolor
        )

    @_Section.figure_factory(1)
    def final_model_fit_colorbox_plot_ume(workspace, switchboard=None, linlog_percentile=5, brevity=0, comm=None,
                                          bgcolor='white', **kwargs):
        qty = workspace.ColorBoxPlot(
            switchboard.objective, switchboard.gss,
            switchboard.modvi_ds, switchboard.gsL,
            linlg_pcntle=linlog_percentile / 100,
            min_prob_clip_for_weighting=switchboard.mpc,
            wildcard=switchboard.wildcardBudget, comm=comm,
            bgcolor=bgcolor
        )
        if brevity < 1:
            qty.set_render_options(click_to_display=False, valign='bottom')
        return qty

    @_Section.figure_factory(1)
    def final_model_fit_colorscatter_plot_ume(workspace, switchboard=None, linlog_percentile=5, comm=None,
                                              bgcolor='white', **kwargs):
        return workspace.ColorBoxPlot(
            switchboard.objective, switchboard.gss,
            switchboard.modvi_ds, switchboard.gsL,
            linlg_pcntle=linlog_percentile / 100,
            min_prob_clip_for_weighting=switchboard.mpc, typ="scatter",
            wildcard=switchboard.wildcardBudget, comm=comm,
            bgcolor=bgcolor
        )

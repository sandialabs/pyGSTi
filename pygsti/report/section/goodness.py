""" Goodness sections """
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

from pygsti.report.section import Section as _Section


class GoodnessSection(_Section):
    _HTML_TEMPLATE = 'tabs/Goodness.html'

    @_Section.figure_factory(1)
    def final_model_fit_colorscatter_plot(workspace, switchboard=None, linlog_percentile=5, comm=None, bgcolor='white',
                                          **kwargs):
        return workspace.ColorBoxPlot(
            switchboard.objfn_builder_modvi, switchboard.circuits_final,
            switchboard.modvi_ds, switchboard.mdl_current_modvi, linlg_pcntle=linlog_percentile / 100,
            typ="scatter", comm=comm, bgcolor=bgcolor, mdc_store= switchboard.final_mdc_store
        )

    @_Section.figure_factory(4)
    def final_model_fit_progress_table(workspace, switchboard=None, max_lengths=None, comm=None, **kwargs):
        return workspace.FitComparisonTable(
            max_lengths, switchboard.circuits_all, switchboard.mdl_all_modvi,
            switchboard.modvi_ds, switchboard.objfn_builder_modvi, 'L', comm=comm
        )

    @_Section.figure_factory(4)
    def final_model_fit_progress_bar_plot(workspace, switchboard=None, max_lengths=None, comm=None, **kwargs):
        return workspace.FitComparisonBarPlot(
            max_lengths, switchboard.circuits_all, switchboard.mdl_all_modvi,
            switchboard.modvi_ds, switchboard.objfn_builder_modvi, 'L', comm=comm
        )


class GoodnessColorBoxPlotSection(_Section):
    _HTML_TEMPLATE = 'tabs/Goodness_colorboxplot.html'

#    @_Section.figure_factory(1)
#    def final_model_fit_colorbox_plot(workspace, switchboard=None, linlog_percentile=5, brevity=0, comm=None,
#                                      bgcolor='white', **kwargs):
#        qty = workspace.ColorBoxPlot(
#            switchboard.objfn_builder_modvi, switchboard.circuits_current,
#            switchboard.modvi_ds, switchboard.mdl_current_modvi,
#            linlg_pcntle=linlog_percentile / 100, comm=comm, bgcolor=bgcolor
#        )
#        if brevity < 1:
#            qty.set_render_options(click_to_display=False, valign='bottom')
#        return qty
#
#    @_Section.figure_factory(1)
#    def final_model_tvd_colorbox_plot(workspace, switchboard=None, brevity=0, comm=None, bgcolor='white', **kwargs):
#        qty = workspace.ColorBoxPlot(
#            'tvd', switchboard.circuits_current, switchboard.modvi_ds, switchboard.mdl_current_modvi,
#            comm=comm, bgcolor=bgcolor
#        )
#        if brevity < 1:
#            qty.set_render_options(click_to_display=False, valign='bottom')
#        return qty
#
#    @_Section.figure_factory()
#    def maxlength_switchboard1(workspace, switchboard=None, switchbd_maxlengths=None, **kwargs):
#        maxLView = [False, False, False, len(switchbd_maxlengths) > 1]
#        return switchboard.view(maxLView, 'v6')
        
    @_Section.figure_factory(1)
    def final_model_fit_colorbox_plot(workspace, switchboard=None, linlog_percentile=5, brevity=0, comm=None,
                                      bgcolor='white', **kwargs):
        qty = workspace.ColorBoxPlot(switchboard.objfn_builder_modvi, switchboard.circuits_final, switchboard.modvi_ds, switchboard.mdl_final_modvi,
            linlg_pcntle=linlog_percentile / 100, comm=comm, bgcolor=bgcolor,
            mdc_store= switchboard.final_mdc_store
        )
        if brevity < 1:
            qty.set_render_options(click_to_display=False, valign='bottom')
        return qty

    @_Section.figure_factory(1)
    def final_model_tvd_colorbox_plot(workspace, switchboard=None, brevity=0, comm=None, bgcolor='white', **kwargs):
        qty = workspace.ColorBoxPlot(
            'tvd', switchboard.circuits_final, switchboard.modvi_ds, switchboard.mdl_final_modvi, comm=comm, bgcolor=bgcolor, mdc_store= switchboard.final_mdc_store
        )
        if brevity < 1:
            qty.set_render_options(click_to_display=False, valign='bottom')
        return qty


class GoodnessScalingSection(_Section):
    _HTML_TEMPLATE = 'tabs/Goodness_scaling.html'

    @_Section.figure_factory(1)
    def final_model_fit_colorscatter_plot_scl(workspace, switchboard=None, linlog_percentile=5, comm=None,
                                              bgcolor='white', **kwargs):
        return workspace.ColorBoxPlot(
            switchboard.objfn_builder, switchboard.circuits_current,
            switchboard.eff_ds, switchboard.mdl_current,
            linlg_pcntle=linlog_percentile / 100,
            typ="scatter", comm=comm, bgcolor=bgcolor
        )

    @_Section.figure_factory(1)
    def final_model_fit_colorbox_plot_scl(workspace, switchboard=None, linlog_percentile=5, brevity=0, comm=None,
                                          bgcolor='white', **kwargs):
        qty = workspace.ColorBoxPlot(
            switchboard.objfn_builder, switchboard.circuits_current,
            switchboard.eff_ds, switchboard.mdl_current,
            linlg_pcntle=linlog_percentile / 100,
            comm=comm, bgcolor=bgcolor
        )
        if brevity < 1:
            qty.set_render_options(click_to_display=False, valign='bottom')
        return qty

    @_Section.figure_factory()
    def final_model_fit_histogram_scl(workspace, switchboard=None, linlog_percentile=5, comm=None, bgcolor='white',
                                      **kwargs):
        return workspace.ColorBoxPlot(
            switchboard.objfn_builder, switchboard.circuits_current,
            switchboard.eff_ds, switchboard.mdl_current,
            linlg_pcntle=linlog_percentile / 100,
            typ="histogram", comm=comm, bgcolor=bgcolor
        )

    @_Section.figure_factory(4)
    def final_model_fit_progress_table_scl(workspace, switchboard=None, max_lengths=None, comm=None, **kwargs):
        return workspace.FitComparisonTable(
            max_lengths, switchboard.circuits_all, switchboard.mdl_all,
            switchboard.eff_ds, switchboard.objfn_builder, 'L', comm=comm
        )

    @_Section.figure_factory(4)
    def final_model_fit_progress_bar_plot_scl(workspace, switchboard=None, max_lengths=None, comm=None, **kwargs):
        return workspace.FitComparisonBarPlot(
            max_lengths, switchboard.circuits_all, switchboard.mdl_all,
            switchboard.eff_ds, switchboard.objfn_builder, 'L', comm=comm
        )

    @_Section.figure_factory(1)
    def data_scaling_colorbox_plot(workspace, switchboard=None, comm=None, bgcolor='white', **kwargs):
        return workspace.ColorBoxPlot(
            'scaling', switchboard.circuits_final, switchboard.eff_ds, None,
            submatrices=switchboard.scaled_submxs_dict, comm=comm,
            bgcolor=bgcolor
        )


class GoodnessUnmodeledSection(_Section):
    _HTML_TEMPLATE = 'tabs/Goodness_unmodeled.html'

    @_Section.figure_factory(1)
    def unmodeled_error_budget_table(workspace, switchboard=None, **kwargs):
        return workspace.WildcardBudgetTable(switchboard.wildcard_budget)

    @_Section.figure_factory(4)
    def final_model_fit_progress_bar_plot_ume(workspace, switchboard=None, max_lengths=None, comm=None, **kwargs):
        return workspace.FitComparisonBarPlot(
            max_lengths, switchboard.circuits_all, switchboard.mdl_all,
            switchboard.modvi_ds, switchboard.objfn_builder_modvi, 'L',
            wildcard=switchboard.wildcard_budget, comm=comm
        )

    @_Section.figure_factory(4)
    def final_model_fit_progress_table_ume(workspace, switchboard=None, max_lengths=None, comm=None, **kwargs):
        return workspace.FitComparisonTable(
            max_lengths, switchboard.circuits_all, switchboard.mdl_all,
            switchboard.modvi_ds, switchboard.objfn_builder_modvi, 'L',
            wildcard=switchboard.wildcard_budget, comm=comm
        )

    @_Section.figure_factory()
    def final_model_fit_histogram_ume(workspace, switchboard=None, linlog_percentile=5, comm=None, bgcolor='white',
                                      **kwargs):
        return workspace.ColorBoxPlot(
            switchboard.objfn_builder_modvi, switchboard.circuits_current,
            switchboard.modvi_ds, switchboard.mdl_current,
            linlg_pcntle=linlog_percentile / 100,
            typ="histogram", wildcard=switchboard.wildcard_budget,
            comm=comm, bgcolor=bgcolor
        )

    @_Section.figure_factory(1)
    def final_model_fit_colorbox_plot_ume(workspace, switchboard=None, linlog_percentile=5, brevity=0, comm=None,
                                          bgcolor='white', **kwargs):
        qty = workspace.ColorBoxPlot(
            switchboard.objfn_builder_modvi, switchboard.circuits_current,
            switchboard.modvi_ds, switchboard.mdl_current,
            linlg_pcntle=linlog_percentile / 100,
            wildcard=switchboard.wildcard_budget, comm=comm,
            bgcolor=bgcolor
        )
        if brevity < 1:
            qty.set_render_options(click_to_display=False, valign='bottom')
        return qty

    @_Section.figure_factory(1)
    def final_model_fit_colorscatter_plot_ume(workspace, switchboard=None, linlog_percentile=5, comm=None,
                                              bgcolor='white', **kwargs):
        return workspace.ColorBoxPlot(
            switchboard.objfn_builder_modvi, switchboard.circuits_current,
            switchboard.modvi_ds, switchboard.mdl_current,
            linlg_pcntle=linlog_percentile / 100, typ="scatter",
            wildcard=switchboard.wildcard_budget, comm=comm,
            bgcolor=bgcolor
        )

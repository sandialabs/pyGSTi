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
    def bestEstimateColorScatterPlot(workspace, switchboard=None, linlog_percentile=5, comm=None, bgcolor='white',
                                     **kwargs):
        return workspace.ColorBoxPlot(
            switchboard.objective, switchboard.gss,
            switchboard.modvi_ds, switchboard.gsL_modvi,
            linlg_pcntle=linlog_percentile / 100,
            min_prob_clip_for_weighting=switchboard.mpc_modvi,
            typ="scatter", comm=comm, bgcolor=bgcolor
        )

    @_Section.figure_factory(4)
    def progressTable(workspace, switchboard=None, Ls=None, comm=None, **kwargs):
        return workspace.FitComparisonTable(
            Ls, switchboard.gssAllL, switchboard.gsAllL_modvi,
            switchboard.modvi_ds, switchboard.objective_modvi, 'L',
            comm=comm, min_prob_clip=switchboard.mpc_modvi
        )

    @_Section.figure_factory(4)
    def progressBarPlot(workspace, switchboard=None, Ls=None, comm=None, **kwargs):
        return workspace.FitComparisonBarPlot(
            Ls, switchboard.gssAllL, switchboard.gsAllL_modvi,
            switchboard.modvi_ds, switchboard.objective_modvi, 'L',
            comm=comm, min_prob_clip=switchboard.mpc_modvi
        )


class GoodnessColorBoxPlotSection(_Section):
    _HTML_TEMPLATE = 'tabs/Goodness_colorboxplot.html'

    @_Section.figure_factory(1)
    def bestEstimateColorBoxPlot(workspace, switchboard=None, linlog_percentile=5, brevity=0, comm=None,
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
    def bestEstimateTVDColorBoxPlot(workspace, switchboard=None, brevity=0, comm=None, bgcolor='white', **kwargs):
        qty = workspace.ColorBoxPlot(
            'tvd', switchboard.gss, switchboard.modvi_ds, switchboard.gsL_modvi, comm=comm, bgcolor=bgcolor
        )
        if brevity < 1:
            qty.set_render_options(click_to_display=False, valign='bottom')
        return qty

    @_Section.figure_factory()
    def maxLSwitchboard1(workspace, switchboard=None, swLs=None, **kwargs):
        maxLView = [False, False, False, len(swLs) > 1]
        return switchboard.view(maxLView, 'v6')


class GoodnessScalingSection(_Section):
    _HTML_TEMPLATE = 'tabs/Goodness_scaling.html'

    @_Section.figure_factory(1)
    def bestEstimateColorScatterPlot_scl(workspace, switchboard=None, linlog_percentile=5, comm=None, bgcolor='white',
                                         **kwargs):
        return workspace.ColorBoxPlot(
            switchboard.objective, switchboard.gss,
            switchboard.eff_ds, switchboard.gsL,
            linlg_pcntle=linlog_percentile / 100,
            min_prob_clip_for_weighting=switchboard.mpc,
            typ="scatter", comm=comm, bgcolor=bgcolor
        )

    @_Section.figure_factory(1)
    def bestEstimateColorBoxPlot_scl(workspace, switchboard=None, linlog_percentile=5, brevity=0, comm=None,
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
    def bestEstimateColorHistogram_scl(workspace, switchboard=None, linlog_percentile=5, comm=None, bgcolor='white',
                                       **kwargs):
        return workspace.ColorBoxPlot(
            switchboard.objective, switchboard.gss,
            switchboard.eff_ds, switchboard.gsL,
            linlg_pcntle=linlog_percentile / 100,
            min_prob_clip_for_weighting=switchboard.mpc,
            typ="histogram", comm=comm, bgcolor=bgcolor
        )

    @_Section.figure_factory(4)
    def progressTable_scl(workspace, switchboard=None, Ls=None, comm=None, **kwargs):
        return workspace.FitComparisonTable(
            Ls, switchboard.gssAllL, switchboard.gsAllL,
            switchboard.eff_ds, switchboard.objective, 'L', comm=comm, min_prob_clip=switchboard.mpc
        )

    @_Section.figure_factory(4)
    def progressBarPlot_scl(workspace, switchboard=None, Ls=None, comm=None, **kwargs):
        return workspace.FitComparisonBarPlot(
            Ls, switchboard.gssAllL, switchboard.gsAllL,
            switchboard.eff_ds, switchboard.objective, 'L', comm=comm, min_prob_clip=switchboard.mpc
        )

    @_Section.figure_factory(1)
    def dataScalingColorBoxPlot(workspace, switchboard=None, comm=None, bgcolor='white', **kwargs):
        return workspace.ColorBoxPlot(
            'scaling', switchboard.gssFinal, switchboard.eff_ds, None,
            submatrices=switchboard.scaledSubMxsDict, comm=comm,
            bgcolor=bgcolor
        )


class GoodnessUnmodeledSection(_Section):
    _HTML_TEMPLATE = 'tabs/Goodness_unmodeled.html'

    @_Section.figure_factory(1)
    def unmodeledErrorBudgetTable(workspace, switchboard=None, **kwargs):
        return workspace.WildcardBudgetTable(switchboard.wildcardBudget)

    @_Section.figure_factory(4)
    def progressBarPlot_ume(workspace, switchboard=None, Ls=None, comm=None, **kwargs):
        return workspace.FitComparisonBarPlot(
            Ls, switchboard.gssAllL, switchboard.gsAllL,
            switchboard.modvi_ds, switchboard.objective, 'L',
            wildcard=switchboard.wildcardBudget, comm=comm, min_prob_clip=switchboard.mpc_modvi
        )

    @_Section.figure_factory(4)
    def progressTable_ume(workspace, switchboard=None, Ls=None, comm=None, **kwargs):
        return workspace.FitComparisonTable(
            Ls, switchboard.gssAllL, switchboard.gsAllL,
            switchboard.modvi_ds, switchboard.objective, 'L',
            wildcard=switchboard.wildcardBudget, comm=comm, min_prob_clip=switchboard.mpc_modvi
        )

    @_Section.figure_factory()
    def bestEstimateColorHistogram_ume(workspace, switchboard=None, linlog_percentile=5, comm=None, bgcolor='white',
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
    def bestEstimateColorBoxPlot_ume(workspace, switchboard=None, linlog_percentile=5, brevity=0, comm=None,
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
    def bestEstimateColorScatterPlot_ume(workspace, switchboard=None, linlog_percentile=5, comm=None, bgcolor='white',
                                         **kwargs):
        return workspace.ColorBoxPlot(
            switchboard.objective, switchboard.gss,
            switchboard.modvi_ds, switchboard.gsL,
            linlg_pcntle=linlog_percentile / 100,
            min_prob_clip_for_weighting=switchboard.mpc, typ="scatter",
            wildcard=switchboard.wildcardBudget, comm=comm,
            bgcolor=bgcolor
        )

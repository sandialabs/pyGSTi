""" Gauge-invariant and -dependent sections """
#***************************************************************************************************
# Copyright 2015, 2019, 2025 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

from pygsti.report.section import Section as _Section
from pygsti.report import reportables as _reportables
from pygsti.report.workspace import NotApplicable as _NotApplicable


class GaugeInvariantsGatesSection(_Section):
    _HTML_TEMPLATE = 'tabs/GaugeInvariants_gates.html'

    def render(self, workspace, results=None, dataset_labels=None, est_labels=None, embed_figures=True, **kwargs):
        # This section's figures depend on switchboards, which must be rendered in advance:
        gi_switchboard = _create_single_metric_switchboard(
            workspace, results, True, dataset_labels, est_labels, embed_figures
        )
        gr_switchboard = _create_single_metric_switchboard(
            workspace, {}, False, [], embed_figures=embed_figures
        )

        return {
            'metricSwitchboard_gi': gi_switchboard,
            'metricSwitchboard_gr': gr_switchboard,
            **super().render(
                workspace, gr_switchboard=gr_switchboard,
                gi_switchboard=gi_switchboard,
                dataset_labels=dataset_labels, est_labels=est_labels,
                embed_figures=embed_figures, **kwargs
            )
        }

    @_Section.figure_factory(4)
    def final_model_spam_parameters_table(workspace, switchboard=None, confidence_level=None, ci_brevity=1, **kwargs):
        return workspace.SpamParametersTable(
            switchboard.mdl_target_and_final, ['Target', 'Estimated'], _cri(1, switchboard,
                                                                            confidence_level, ci_brevity)
        )

    @_Section.figure_factory(4)
    def final_model_eigenvalue_table(workspace, switchboard=None, confidence_level=None, ci_brevity=1, **kwargs):
        return workspace.GateEigenvalueTable(
            switchboard.mdl_gaugeinv, switchboard.mdl_target, _cri_gauge_inv(1, switchboard,
                                                                             confidence_level, ci_brevity),
            display=('evals', 'target', 'absdiff-evals', 'infdiff-evals', 'log-evals', 'absdiff-log-evals')
        )

    @_Section.figure_factory(4)
    def final_model_predicted_RB_table(workspace, switchboard=None, confidence_level=None, ci_brevity=1, **kwargs):
        return workspace.ModelVsTargetTable(
            switchboard.mdl_gaugeinv, switchboard.mdl_target, switchboard.clifford_compilation,
            _cri(1, switchboard, confidence_level, ci_brevity)
        )

    @_Section.figure_factory(4)
    def final_gates_vs_target_table_gauge_inv(workspace, switchboard=None, confidence_level=None,
                                              ci_brevity=1, **kwargs):
        return workspace.GatesVsTargetTable(
            switchboard.mdl_gaugeinv, switchboard.mdl_target, _cri_gauge_inv(0, switchboard,
                                                                             confidence_level, ci_brevity),
            display=('evinf', 'evagi', 'evnuinf', 'evnuagi', 'evdiamond', 'evnudiamond')
        )

    @_Section.figure_factory(4)
    def final_gauge_inv_model_table(workspace, switchboard=None, confidence_level=None, ci_brevity=1, **kwargs):
        #return workspace.BlankTable()  # this table is slow, uncomment this to disable it temporarily
        return workspace.GaugeRobustModelTable(
            switchboard.mdl_gaugeinv, switchboard.mdl_target, 'boxes', _cri(1, switchboard, confidence_level, ci_brevity)
        )

    @_Section.figure_factory(4)
    def single_metric_table_gauge_inv(workspace, switchboard=None, dataset_labels=None, est_labels=None,
                                      gi_switchboard=None, **kwargs):
        if len(dataset_labels) > 1:
            # Multiple data
            return workspace.GatesSingleMetricTable(
                gi_switchboard.metric, switchboard.mdl_final_grid,
                switchboard.mdl_target_grid, est_labels, dataset_labels,
                gi_switchboard.cmp_table_title, gi_switchboard.op_label,
                confidence_region_info=None
            )
        else:
            return workspace.GatesSingleMetricTable(
                gi_switchboard.metric, switchboard.mdl_final_grid,
                switchboard.mdl_target_grid, est_labels, None,
                gi_switchboard.cmp_table_title,
                confidence_region_info=None
            )

    @_Section.figure_factory(4)
    def final_gauge_inv_metric_table(workspace, switchboard=None, confidence_level=None, ci_brevity=1,
                                     gr_switchboard=None, **kwargs):
        # Temporarily disable this table for > 1Q models, since it's so slow...
        first_model = None
        for mdl in switchboard.mdl_gaugeinv.base.flat:
            if not isinstance(mdl, _NotApplicable):
                first_model = mdl; break
        if first_model and first_model.dim > 4:
            return workspace.BlankTable()  # this table is slow, uncomment this to disable it temporarily

        return workspace.GaugeRobustMetricTable(
            switchboard.mdl_gaugeinv, switchboard.mdl_target, gr_switchboard.metric,
            _cri(1, switchboard, confidence_level, ci_brevity)
        )

    @_Section.figure_factory(4)
    def gram_bar_plot(workspace, switchboard=None, **kwargs):
        try:
            return workspace.GramMatrixBarPlot(switchboard.ds, switchboard.mdl_target, 10, switchboard.fiducials_tup)
        except KeyError:  # when we don't have LGST data, just ignore plot
            return workspace.BlankTable()


class GaugeInvariantsGermsSection(_Section):
    _HTML_TEMPLATE = 'tabs/GaugeInvariants_germs.html'

    @_Section.figure_factory(3)
    def final_gates_vs_target_table_gauge_invgerms(workspace, switchboard=None, confidence_level=None,
                                                   ci_brevity=1, **kwargs):
        return workspace.GatesVsTargetTable(
            switchboard.mdl_gaugeinv, switchboard.mdl_target,
            _cri_gauge_inv(0, switchboard, confidence_level, ci_brevity),
            display=('evdiamond', 'evnudiamond'), virtual_ops=switchboard.germs
        )

    @_Section.figure_factory(3)
    def germs_eigenvalue_table(workspace, switchboard=None, confidence_level=None, ci_brevity=1, **kwargs):
        return workspace.GateEigenvalueTable(
            switchboard.mdl_gaugeinv, switchboard.mdl_target,
            _cri_gauge_inv(1, switchboard, confidence_level, ci_brevity),
            display=('evals', 'target', 'absdiff-evals', 'infdiff-evals', 'log-evals', 'absdiff-log-evals'),
            virtual_ops=switchboard.germs
        )


class GaugeVariantSection(_Section):
    _HTML_TEMPLATE = 'tabs/GaugeVariants.html'

    def render(self, workspace, results=None, dataset_labels=None, est_labels=None, embed_figures=True, **kwargs):
        # This section's figures depend on switchboards, which must be rendered in advance:
        # XXX this is SO wack
        gv_switchboard = _create_single_metric_switchboard(
            workspace, results, False, dataset_labels, est_labels, embed_figures
        )

        return {
            'metricSwitchboard_gv': gv_switchboard,
            **super().render(
                workspace, gv_switchboard=gv_switchboard,
                dataset_labels=dataset_labels, est_labels=est_labels,
                embed_figures=embed_figures, **kwargs
            )
        }

    @_Section.figure_factory(4)
    def final_model_spam_vs_target_table(workspace, switchboard=None, confidence_level=None, ci_brevity=1, **kwargs):
        return workspace.SpamVsTargetTable(
            switchboard.mdl_final, switchboard.mdl_target, _cri(1, switchboard, confidence_level, ci_brevity)
        )

    @_Section.figure_factory(4)
    def final_gates_vs_target_table_gauge_var(workspace, switchboard=None, confidence_level=None,
                                              ci_brevity=1, **kwargs):

        if switchboard is not None and 'mdl_target_grid' in switchboard:
            basis_representative = switchboard['mdl_target_grid'][0].basis
        else:
            res_representative   = kwargs['results'][list(kwargs['results'])[0]]
            est_representative   = res_representative.estimates[list(res_representative.estimates)[0]]
            mdl_representative   = est_representative.models[list(est_representative.models)[-1]]
            basis_representative = mdl_representative.basis
        n_leak_default = None if basis_representative.implies_leakage_modeling else 0

        if kwargs.get('n_leak', n_leak_default) == 0:
            display = ('inf', 'agi', 'geni', 'trace', 'diamond', 'nuinf', 'nuagi')
        else:
            display = ('sub-inf', 'sub-trace', 'sub-diamond', 'plf-sub-diamond', 'leak-rate-max', 'leak-rate-min', 'seep-rate' )
        return workspace.GatesVsTargetTable(
            switchboard.mdl_final, switchboard.mdl_target, _cri(1, switchboard, confidence_level, ci_brevity),
            display=display
        )

    @_Section.figure_factory(3)
    def final_gates_vs_target_table_gauge_vargerms(workspace, switchboard=None, confidence_level=None,
                                                   ci_brevity=1, **kwargs):
        return workspace.GatesVsTargetTable(
            switchboard.mdl_final, switchboard.mdl_target, _cri(0, switchboard, confidence_level, ci_brevity),
            display=('inf', 'trace', 'geni', 'nuinf'), virtual_ops=switchboard.germs
        )

    @_Section.figure_factory(4)
    def single_metric_table_gauge_var(workspace, switchboard=None, dataset_labels=None, est_labels=None,
                                      gv_switchboard=None, **kwargs):
        if len(dataset_labels) > 1:
            # Multiple data
            return workspace.GatesSingleMetricTable(
                gv_switchboard.metric, switchboard.mdl_final_grid,
                switchboard.mdl_target_grid, est_labels, dataset_labels,
                gv_switchboard.cmp_table_title, gv_switchboard.op_label,
                confidence_region_info=None
            )
        else:
            return workspace.GatesSingleMetricTable(
                gv_switchboard.metric, switchboard.mdl_final_grid,
                switchboard.mdl_target_grid, est_labels, None,
                gv_switchboard.cmp_table_title,
                confidence_region_info=None
            )


class GaugeVariantsDecompSection(_Section):
    _HTML_TEMPLATE = 'tabs/GaugeVariants_decomp.html'

    @_Section.figure_factory(4)
    def final_model_choi_eigenvalue_table(workspace, switchboard=None, confidence_level=None, ci_brevity=1, **kwargs):
        return workspace.ChoiTable(
            switchboard.mdl_final, None, _cri(1, switchboard, confidence_level, ci_brevity),
            display=('boxplot', 'barplot')
        )

    @_Section.figure_factory(4)
    def final_model_decomposition_table(workspace, switchboard=None, confidence_level=None, ci_brevity=1, **kwargs):
        return workspace.GateDecompTable(
            switchboard.mdl_final, switchboard.mdl_target, _cri(0, switchboard, confidence_level, ci_brevity)
        )


class GaugeVariantsErrorGenSection(_Section):
    _HTML_TEMPLATE = 'tabs/GaugeVariants_errgen.html'

    @_Section.figure_factory(4)
    def final_model_errorgen_box_table(workspace, switchboard=None, confidence_level=None, ci_brevity=1,
                                       errgen_type='logGTi', **kwargs):
        return workspace.ErrgenTable(
            switchboard.mdl_final, switchboard.mdl_target, _cri(1, switchboard, confidence_level, ci_brevity),
            ('errgen', 'H', 'S', 'CA'), 'boxes', errgen_type
        )

    @_Section.figure_factory(4)
    def errorgen_type(workspace, errgen_type='logGTi', **kwargs):
        # Not a figure, but who cares?
        return errgen_type


class GaugeVariantsErrorGenNQubitSection(GaugeVariantsErrorGenSection):
    @_Section.figure_factory(4)
    def final_model_errorgen_box_table(workspace, switchboard=None, confidence_level=None, ci_brevity=1,
                                       errgen_type='logGTi', **kwargs):
        return workspace.NQubitErrgenTable(
            switchboard.mdl_gaugeinv, _cri(1, switchboard, confidence_level, ci_brevity),
            ('H', 'S'), 'boxes'
        )  # 'errgen' not allowed - 'A'?


class GaugeVariantsRawSection(_Section):
    _HTML_TEMPLATE = 'tabs/GaugeVariants_raw.html'

    @_Section.figure_factory(4)
    def final_gates_box_table(workspace, switchboard=None, confidence_level=None, ci_brevity=1, **kwargs):
        return workspace.GatesTable(
            switchboard.mdl_target_and_final, ['Target', 'Estimated'], 'boxes',
            _cri_target_and_final(1, switchboard, confidence_level, ci_brevity)
        )

    @_Section.figure_factory(4)
    def final_model_brief_spam_table(workspace, switchboard=None, confidence_level=None, ci_brevity=1, **kwargs):
        return workspace.SpamTable(
            switchboard.mdl_target_and_final, ['Target', 'Estimated'], 'boxes',
            _cri_target_and_final(1, switchboard, confidence_level, ci_brevity), include_hs_vec=True
        )


# Helper functions
def _cri(el, switchboard, confidence_level, ci_brevity):
    return switchboard.cri if confidence_level is not None and ci_brevity <= el else None

def _cri_target_and_final(el, switchboard, confidence_level, ci_brevity):
    return switchboard.cri_target_and_final if confidence_level is not None and ci_brevity <= el else [None, None]

def _cri_gauge_inv(el, switchboard, confidence_level, ci_brevity):
    return switchboard.cri_gaugeinv if confidence_level is not None and ci_brevity <= el else None


def _create_single_metric_switchboard(ws, results_dict, b_gauge_inv,
                                      dataset_labels, est_labels=None, embed_figures=True):
    op_labels = []
    for results in results_dict.values():
        for est in results.estimates.values():
            if 'target' in est.models:
                # append non-duplicate labels
                op_labels.extend([op for op in est.models['target'].operations.keys() if op not in op_labels])

    if b_gauge_inv:
        metric_abbrevs = ["evinf", "evagi", "evnuinf", "evnuagi", "evdiamond",
                          "evnudiamond"]
    else:
        metric_abbrevs = ["inf", "agi", "geni", "trace", "diamond", "nuinf", "nuagi",
                          "frob"]
    metric_names = [_reportables.info_of_opfn_by_name(abbrev)[0].replace('|', ' ')
                    for abbrev in metric_abbrevs]

    if len(dataset_labels) > 1:  # multidataset
        metric_switchBd = ws.Switchboard(
            ["Metric", "Operation"], [metric_names, op_labels],
            ["dropdown", "dropdown"], [0, 0], show=[True, True],
            use_loadable_items=embed_figures)
        metric_switchBd.add("op_label", (1,))
        metric_switchBd.add("metric", (0,))
        metric_switchBd.add("cmp_table_title", (0, 1))

        metric_switchBd.op_label[:] = op_labels
        for i, gl in enumerate(op_labels):
            metric_switchBd.cmp_table_title[:, i] = ["%s %s" % (gl, nm) for nm in metric_names]

    else:
        metric_switchBd = ws.Switchboard(
            ["Metric"], [metric_names],
            ["dropdown"], [0], show=[True],
            use_loadable_items=embed_figures)
        metric_switchBd.add("metric", (0,))
        metric_switchBd.add("cmp_table_title", (0,))
        metric_switchBd.cmp_table_title[:] = metric_names

    metric_switchBd.metric[:] = metric_abbrevs

    return metric_switchBd

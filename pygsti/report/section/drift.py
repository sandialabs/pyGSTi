""" Drift report sections """
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

from . import Section as _Section


class DriftSection(_Section):
    _HTML_TEMPLATE = 'tabs/Drift.html'

    @_Section.figure_factory()
    def drift_summary_table(workspace, results=None, dskey=None, **kwargs):
        return workspace.DriftSummaryTable(results, dskey)

    @_Section.figure_factory()
    def drift_details_table(workspace, results=None, **kwargs):
        return workspace.DriftDetailsTable(results)

    @_Section.figure_factory()
    def global_power_spectra_plot(workspace, results=None, dskey=None, **kwargs):
        return workspace.PowerSpectraPlot(results, {'dataset': dskey})

    @_Section.figure_factory()
    def germ_fiducial_power_spectra_plot(workspace, results=None, gss=None, switchboard=None, dskey=None, **kwargs):
        return workspace.germ_fiducial_power_spectra_plot(
            results, gss, switchboard.prepStrs, switchboard.germs,
            switchboard.effectStrs, dskey, None, True
        )

    @_Section.figure_factory()
    def germ_fiducial_prob_trajectories_plot(workspace, results=None, gss=None, switchboard=None, dskey=None, **kwargs):
        return workspace.germ_fiducial_prob_trajectories_plot(
            results, gss, switchboard.prepStrs, switchboard.germs,
            switchboard.effectStrs, switchboard.outcomes, 1, None,
            dskey, None, None, True
        )

    @_Section.figure_factory()
    def drift_detector_colorbox_plot(workspace, results=None, gss=None, **kwargs):
        return workspace.ColorBoxPlot(
            'driftdetector', gss, None, None, False, False, True,
            False, 'compact', .05, 1e-4, None, None, results
        )

    @_Section.figure_factory()
    def drift_size_colorbox_plot(workspace, results=None, gss=None, **kwargs):
        return workspace.ColorBoxPlot(
            'driftsize', gss, None, None, False, False, True, False,
            'compact', .05, 1e-4, None, None, results
        )

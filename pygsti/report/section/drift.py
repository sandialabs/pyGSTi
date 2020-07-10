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
    def germ_fiducial_power_spectra_plot(workspace, results=None, circuit_list=None, switchboard=None,
                                         dskey=None, **kwargs):
        return workspace.GermFiducialPowerSpectraPlot(
            results, circuit_list, switchboard.prep_fiducials, switchboard.germs,
            switchboard.meas_fiducials, dskey, None, True
        )

    @_Section.figure_factory()
    def germ_fiducial_prob_trajectories_plot(workspace, results=None, circuit_list=None, switchboard=None,
                                             dskey=None, **kwargs):
        return workspace.GermFiducialProbTrajectoriesPlot(
            results, circuit_list, switchboard.prep_fiducials, switchboard.germs,
            switchboard.meas_fiducials, switchboard.outcomes, 1, None,
            dskey, None, None, True
        )

    @_Section.figure_factory()
    def drift_detector_colorbox_plot(workspace, results=None, circuit_list=None, **kwargs):
        return workspace.ColorBoxPlot(
            'driftdetector', circuit_list, None, None, linlg_pcntle=.05, stability_analyzer=results
        )

    @_Section.figure_factory()
    def drift_size_colorbox_plot(workspace, results=None, circuit_list=None, **kwargs):
        return workspace.ColorBoxPlot(
            'driftsize', circuit_list, None, None, linlg_pcntle=.05, stability_analyzer=results
        )

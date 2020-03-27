""" Metadata sections """
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

from . import Section as _Section


class InputSection(_Section):
    _HTML_TEMPLATE = 'tabs/Input.html'

    @_Section.figure_factory(2)
    def fiducial_list_table(workspace, switchboard=None, **kwargs):
        return workspace.CircuitTable(
            switchboard.strs, ["Prep.", "Measure"], common_title="Fiducials"
        )

    @_Section.figure_factory(2)
    def germ_list_2col_table(workspace, switchboard=None, **kwargs):
        return workspace.CircuitTable(switchboard.germs, "Germ", n_cols=2)

    @_Section.figure_factory(2)
    def dataset_overview_table(workspace, switchboard=None, **kwargs):
        return workspace.DataSetOverviewTable(switchboard.ds)

    @_Section.figure_factory(2)
    def target_gates_box_table(workspace, switchboard=None, **kwargs):
        return workspace.GatesTable(switchboard.gsTarget, display_as="boxes")

    @_Section.figure_factory(2)
    def target_spam_brief_table(workspace, switchboard=None, **kwargs):
        return workspace.SpamTable(
            switchboard.gsTarget, None, display_as='boxes', include_hs_vec=False
        )


class MetaSection(_Section):
    _HTML_TEMPLATE = 'tabs/Meta.html'

    @_Section.figure_factory(2)
    def metadata_table(workspace, switchboard=None, **kwargs):
        return workspace.MetadataTable(switchboard.gsFinal, switchboard.params)

    @_Section.figure_factory(2)
    def stdout_block(workspace, switchboard=None, **kwargs):
        return workspace.StdoutText(switchboard.meta_stdout)

    @_Section.figure_factory(2)
    def profiler_table(workspace, switchboard=None, **kwargs):
        return workspace.ProfilerTable(switchboard.profiler)

    @_Section.figure_factory(2)
    def software_environment_table(workspace, **kwargs):
        return workspace.SoftwareEnvTable()

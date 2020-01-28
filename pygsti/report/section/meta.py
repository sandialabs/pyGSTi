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
    def fiducialListTable(workspace, switchboard=None, **kwargs):
        return workspace.CircuitTable(
            switchboard.strs, ["Prep.", "Measure"], commonTitle="Fiducials"
        )

    @_Section.figure_factory(2)
    def germList2ColTable(workspace, switchboard=None, **kwargs):
        return workspace.CircuitTable(switchboard.germs, "Germ", nCols=2)

    @_Section.figure_factory(2)
    def datasetOverviewTable(workspace, switchboard=None, **kwargs):
        return workspace.DataSetOverviewTable(switchboard.ds)

    @_Section.figure_factory(2)
    def targetGatesBoxTable(workspace, switchboard=None, **kwargs):
        return workspace.GatesTable(switchboard.gsTarget, display_as="boxes")

    @_Section.figure_factory(2)
    def targetSpamBriefTable(workspace, switchboard=None, **kwargs):
        return workspace.SpamTable(
            switchboard.gsTarget, None, display_as='boxes', includeHSVec=False
        )


class MetaSection(_Section):
    _HTML_TEMPLATE = 'tabs/Meta.html'

    @_Section.figure_factory(2)
    def metadataTable(workspace, switchboard=None, **kwargs):
        return workspace.MetadataTable(switchboard.gsFinal, switchboard.params)

    @_Section.figure_factory(2)
    def stdoutBlock(workspace, switchboard=None, **kwargs):
        return workspace.StdoutText(switchboard.meta_stdout)

    @_Section.figure_factory(2)
    def profilerTable(workspace, switchboard=None, **kwargs):
        return workspace.ProfilerTable(switchboard.profiler)

    @_Section.figure_factory(2)
    def softwareEnvTable(workspace, **kwargs):
        return workspace.SoftwareEnvTable()

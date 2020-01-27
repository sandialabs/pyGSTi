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

    def __init__(self, workspace, germs, dataset):
        super().__init__({
            'germList2ColTable': workspace.CircuitTable(germs, "Germ", nCols=2),
            'datasetOverviewTable': workspace.DataSetOverviewTable(dataset),
        })

    def with_fiducial_list(self, workspace, fiducials):
        self._quantities['fiducialListTable'] = workspace.CircuitTable(
            fiducials, ["Prep.", "Measure"], commonTitle="Fiducials"
        )
        return self

    def with_target_gates_and_spam(self, workspace, target):
        self._quantities['targetGatesBoxTable'] = workspace.GatesTable(target, display_as="boxes")
        self._quantities['targetSpamBriefTable'] = workspace.SpamTable(
            target, None, display_as='boxes', includeHSVec=False
        )
        return self


class MetaSection(_Section):
    _HTML_TEMPLATE = 'tabs/Meta.html'

    def __init__(self, workspace, final_model, params, stdout, profiler):
        super().__init__({
            'metadataTable': workspace.MetadataTable(final_model, params),
            'stdoutBlock': workspace.StdoutText(stdout),
            'profilerTable': workspace.ProfilerTable(profiler),
            'softwareEnvTable': workspace.SoftwareEnvTable()
        })

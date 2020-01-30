""" Idle Tomography section """
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

from . import Section as _Section


class IdleTomographySection(_Section):
    _HTML_TEMPLATE = 'tabs/IdleTomography.html'

    @_Section.figure_factory()
    def idtIntrinsicErrorsTable(workspace, switchboard=None, **kwargs):
        return workspace.IdleTomographyIntrinsicErrorsTable(switchboard.idtresults)

    @_Section.figure_factory(3)
    def idtObservedRatesTable(workspace, switchboard=None, **kwargs):
        # HARDCODED - show only top 20 rates
        return workspace.IdleTomographyObservedRatesTable(
            switchboard.idtresults, 20, switchboard.gsGIRep
        )

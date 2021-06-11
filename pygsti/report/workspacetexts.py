"""
Classes corresponding to text blocks within a Workspace context.
"""
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

from .textblock import ReportText as _ReportText
from .workspace import WorkspaceText


class StdoutText(WorkspaceText):
    """
    A text block showing standard output recorded using VerbosityPrinter objects.

    Parameters
    ----------
    ws : Workspace
        The containing (parent) workspace.

    vb_recorded_output : list
        A list of `(type,level,message)` tuples, one per line/message
        as returned by :method:`VerbosityPrinter.stop_recording`.
    """

    def __init__(self, ws, vb_recorded_output):
        """
        A text block of standard output.

        Parameters
        ----------
        vb_recorded_output : list
            A list of `(type,level,message)` tuples, one per line/message
            as returned by :method:`VerbosityPrinter.stop_recording`.
        """
        super(StdoutText, self).__init__(ws, self._create, vb_recorded_output)

    def _create(self, vb_recorded_output):
        return _ReportText(vb_recorded_output, "VerbosityPrinter")

""" Classes corresponding to text blocks within a Workspace context."""
from __future__ import division, print_function, absolute_import, unicode_literals
#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************

from .workspace import WorkspaceText
from .textblock import ReportText as _ReportText

class StdoutText(WorkspaceText):
    """A text block showing standard output recorded using
       VerbosityPrinter objects"""
    def __init__(self, ws, vbRecordedOutput):
        """
        A text block of standard output.

        Parameters
        ----------
        vbRecordedOutput : list
            A list of `(type,level,message)` tuples, one per line/message
            as returned by :method:`VerbosityPrinter.stop_recording`.
        """
        super(StdoutText,self).__init__(ws, self._create, vbRecordedOutput)

    def _create(self, vbRecordedOutput):
        return _ReportText(vbRecordedOutput, "VerbosityPrinter")

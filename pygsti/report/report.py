""" Internal model of a report during generation """
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

import time as _time
import warnings as _warnings

from . import autotitle as _autotitle
from . import merge_helpers as _merge
from .. import _version


class Report:
    """ The internal model of a report.

    This class should never be instantiated directly. instead, users
    should use the appropriate factory method in
    `pygsti.report.factory`.

    """
    def __init__(self):
        self.sections = []
        self.qtys = {}

    def _finalize(self):
        """ Finish construction of the underlying report """

        if 'date' not in self.qtys:
            self.qtys['date'] = _time.strftime("%B %d, %Y")
        if 'title' not in self.qtys:
            autoname = _autotitle.generate_name()
            _warnings.warn(("You should really specify `title=` when generating reports,"
                            " as this makes it much easier to identify them later on.  "
                            "Since you didn't, pyGSTi has generated a random one"
                            " for you: '{}'.").format(autoname))
            self.qtys['title'] = "GST Report for {}".format(autoname)
        if 'pdfinfo' not in self.qtys:
            pdfInfo = [('Author', 'pyGSTi'), ('Title', self.qtys['title']),
                       ('Keywords', 'GST'), ('pyGSTi Version', _version.__version__)]
            self.qtys['pdfinfo'] = _merge.to_pdfinfo(pdfInfo)

    def write_html(self, path):
        """ Write this report to the disk as a collection of HTML documents.

        Parameters
        ----------
        path : path-like object
            The filesystem path of a directory to write the report
            to. If the specified directory does not exist, it will be
            created automatically.
        """
        self._finalize()
        # TODO

    def write_notebook(self, path):
        """ Write this report to the disk as an IPython notebook

        Parameters
        ----------
        path : path-like object
            The filesystem path to write the report to. By convention,
            this should use the `.ipynb` file extension.
        """
        self._finalize()
        # TODO

    def write_pdf(self, path):
        """ Write this report to the disk as a PDF document.

        Parameters
        ----------
        path : path-like object
            The filesystem path to write the report to. By convention,
            this should use the `.pdf` file extension.
        """
        self._finalize()
        # TODO

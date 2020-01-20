""" Internal model of a report during generation """
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

from . import workspace as _ws


class Report:
    """ The internal model of a report

    TODO docstring
    """
    def __init__(self, cache_file=None):
        self._sections = []
        self._globals = {}
        self._workspace = _ws.Workspace(cache_file)

    def write_html(self, path):
        """ Write this report to the disk as a collection of HTML documents.

        Parameters
        ----------
        path : path-like object
            The filesystem path of a directory to write the report
            to. If the specified directory does not exist, it will be
            created automatically.
        """
        pass  # TODO

    def write_notebook(self, path):
        """ Write this report to the disk as an IPython notebook

        Parameters
        ----------
        path : path-like object
            The filesystem path to write the report to. By convention,
            this should use the `.ipynb` file extension.
        """
        pass  # TODO

    def write_pdf(self, path):
        """ Write this report to the disk as a PDF document.

        Parameters
        ----------
        path : path-like object
            The filesystem path to write the report to. By convention,
            this should use the `.pdf` file extension.
        """
        pass  # TODO

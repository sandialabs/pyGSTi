""" Internal model of a section of a generated report """
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************


class Section:
    """ Abstract base class for report sections.

    Derived classes encapsulate the structure of data within the
    respective section of the report, and provide methods for
    rendering the section to various output formats.
    """
    _HTML_TEMPLATE = None

    def __init__(self, quantities):
        self._quantities = quantities

    @_abstractmethod
    def render_html(self, global_qtys):
        """ Render this section to HTML

        Parameters
        ----------
        global_qtys: {WorkspaceOutput}
            A dictionary of reportable quantities global to the report
        """
        pass  # TODO

    @_abstractmethod
    def render_latex(self, global_qtys):
        """ Render this section to LaTeX

        Parameters
        ----------
        global_qtys: {WorkspaceOutput}
            A dictionary of reportable quantities global to the report

        Returns
        -------
        str : The generated LaTeX source for this section
        """
        pass  # TODO

    @_abstractmethod
    def render_notebook(self, global_qtys, notebook):
        """ Render this section to an IPython notebook

        Parameters
        ----------
        global_qtys: {WorkspaceOutput}
            A dictionary of reportable quantities global to the report
        notebook: :class:`Notebook`
            The IPython notebook to extend with this section
        """
        pass  # TODO

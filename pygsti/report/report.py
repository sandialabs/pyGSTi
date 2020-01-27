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
from pathlib import Path as _Path
import shutil as _shutil
from collections import defaultdict as _dd

from . import autotitle as _autotitle
from . import merge_helpers as _merge
from .. import _version
from ..objects import VerbosityPrinter as _VerbosityPrinter


class Report:
    """ The internal model of a report.

    This class should never be instantiated directly. instead, users
    should use the appropriate factory method in
    `pygsti.report.factory`.

    """
    def __init__(self):
        self.sections = []
        self.qtys = {}
        self._toggles = _dd(lambda: False)

    def set_toggle(self, key, value=True):
        """ Set the value of a configuration toggle for this report. """
        self._toggles[key] = value

    def write_html(self, path, template_dir='~standard_html_report',
                   template_name='main.html', auto_open=False,
                   link_to=None, embed_figures=True,
                   render_options=None, single_file=False,
                   verbosity=0):
        """ Write this report to the disk as a collection of HTML documents.

        Parameters
        ----------
        path : path-like object
            The filesystem path of a directory to write the report
            to. If the specified directory does not exist, it will be
            created automatically.

        TODO docstring
        """

        # Render sections
        # TODO actually dispatch rendering to sections
        # as a quick stand-in we can just treat local quantities as global
        global_qtys = self.qtys.copy()
        for section in self.sections:
            section.render_html(global_qtys)

        render_options = render_options or {}
        precision = render_options.get('precision', None)
        resizable = render_options.get('resizable', True)
        autosize = render_options.get('autosize', 'initial')

        # TODO refactor all rendering into this method and section rendering methods
        # XXX FWIW single-file mode is a bad idea
        template_fn = _merge.merge_jinja_template if single_file else _merge.merge_jinja_template_dir
        template_fn(
            global_qtys, path, templateDir=template_dir,
            templateName=template_name, auto_open=auto_open,
            precision=precision, link_to=link_to, connected=False,
            toggles=self._toggles, renderMath=True, resizable=resizable,
            autosize=autosize, verbosity=verbosity
        )

    def write_notebook(self, path, template_file=None):
        """ Write this report to the disk as an IPython notebook

        Parameters
        ----------
        path : path-like object
            The filesystem path to write the report to. By convention,
            this should use the `.ipynb` file extension.
        """
        pass
        # TODO

    def write_pdf(self, path, template="standard_pdf_report.tex",
                  precision=None, latex_cmd='pdflatex', latex_flags=None,
                  auto_open=False, comm=None, verbosity=0):
        """ Write this report to the disk as a PDF document.

        Parameters
        ----------
        path : path-like object
            The filesystem path to write the report to. By convention,
            this should use the `.pdf` file extension.

        TODO docstring
        """

        printer = _VerbosityPrinter.build_printer(verbosity, comm=comm)
        path = _Path(path)
        latex_flags = latex_flags or ["-interaction=nonstopmode", "-halt-on-error", "-shell-escape"]

        # Render sections
        # TODO actually dispatch rendering to sections
        # as a quick stand-in we can just treat local quantities as global
        global_qtys = self.qtys.copy()
        for section in self.sections:
            section.render_latex(global_qtys)

        # Remove switchboards
        # TODO these should be generated when rendering HTML, making this redundant
        from .workspace import SwitchboardView, Switchboard
        to_del = []
        for k, v in global_qtys.items():
            if isinstance(v, SwitchboardView) or isinstance(v, Switchboard):
                to_del.append(k)
        for k in to_del:
            del global_qtys[k]

        printer.log("Generating LaTeX source...")
        _merge.merge_latex_template(
            global_qtys, template, str(path.with_suffix('.tex')),
            self._toggles, precision, printer
        )

        printer.log("Compiling with `{} {}`".format(latex_cmd, ' '.join(latex_flags)))
        _merge.compile_latex_report(str(path.parent / path.stem), [latex_cmd] + latex_flags, printer, auto_open)

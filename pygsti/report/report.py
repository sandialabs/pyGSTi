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

    This class should never be instantiated directly. Instead, users
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
                   link_to=None, connected=False, precision=None,
                   resizable=True, autosize='initial',
                   embed_figures=True, single_file=False,
                   verbosity=0):
        """ Write this report to the disk as a collection of HTML documents.

        Parameters
        ----------
        path : str or path-like object
            The filesystem path of a directory to write the report
            to. If the specified directory does not exist, it will be
            created automatically

        templateDir : str, optional
            Path to look for templates, relative to pyGSTi's `templates` directory.

        templateName : str, optional
            The entry-point template filename, relative to pyGSTi's `templates` directory.

        auto_open : bool, optional
            Whether the output file should be automatically opened in a web browser.

        link_to : list, optional
            If not None, a list of one or more items from the set
            {"tex", "pdf", "pkl"} indicating whether or not to
            create and include links to Latex, PDF, and Python pickle
            files, respectively.

        connected : bool, optional
            Whether output HTML should assume an active internet connection.  If
            True, then the resulting HTML file size will be reduced because it
            will link to web resources (e.g. CDN libraries) instead of embedding
            them.

        precision : int or dict, optional
            The amount of precision to display.  A dictionary with keys
            "polar", "sci", and "normal" can separately specify the
            precision for complex angles, numbers in scientific notation, and
            everything else, respectively.  If an integer is given, it this
            same value is taken for all precision types.  If None, then
            `{'normal': 6, 'polar': 3, 'sci': 0}` is used.

        resizable : bool, optional
            Whether plots and tables are made with resize handles and can be
            resized within the report.

        autosize : {'none', 'initial', 'continual'}
            Whether tables and plots should be resized, either initially --
            i.e. just upon first rendering (`"initial"`) -- or whenever
            the browser window is resized (`"continual"`).

        embed_figures : bool, optional
            Whether figures should be embedded in the generated report. If
            False, figures will be written to a 'figures' directory in the
            output directory, and will be loaded dynamically via
            AJAX. This may be useful for reducing the size of the
            generated report if the report includes relatively many
            figures.
            Note that all major web browsers will block AJAX requests from
            an HTML document loaded from the filesystem. If this option is
            set to False, the generated report will fail to load from the
            filesystem, and must be served up by a webserver. Python's
            `http.server` module works fine.

        single_file : bool, optional
            If true, the report will be written to a single HTML
            document, with external dependencies baked-in. This mode
            is not recommended for large reports, because this file
            can grow large enough that major web browsers may struggle
            to render it.

        verbosity : int, optional
            Amount of detail to print to stdout.
        """

        # Render sections
        # TODO actually dispatch rendering to sections
        # as a quick stand-in we can just treat local quantities as global
        global_qtys = self.qtys.copy()
        for section in self.sections:
            section.render_html(global_qtys)

        # TODO refactor all rendering into this method and section rendering methods
        if single_file:
            _merge.merge_jinja_template(
                global_qtys, path, templateDir=template_dir,
                templateName=template_name, auto_open=auto_open,
                precision=precision, link_to=link_to, connected=connected,
                toggles=self._toggles, renderMath=True, resizable=resizable,
                autosize=autosize, verbosity=verbosity
            )
        else:
            _merge.merge_jinja_template_dir(
                global_qtys, path, templateDir=template_dir,
                templateName=template_name, auto_open=auto_open,
                precision=precision, link_to=link_to, connected=connected,
                toggles=self._toggles, renderMath=True, resizable=resizable,
                autosize=autosize, embed_figures=embed_figures, verbosity=verbosity
            )

    def write_notebook(self, path, template_file=None):
        """ Write this report to the disk as an IPython notebook

        Parameters
        ----------
        path : str or path-like object
            The filesystem path to write the report to. By convention,
            this should use the `.ipynb` file extension.
        """
        pass
        # TODO

    def write_pdf(self, path, template="standard_pdf_report.tex",
                  latex_cmd='pdflatex', latex_flags=None, precision=None,
                  auto_open=False, comm=None, verbosity=0):
        """ Write this report to the disk as a PDF document.

        Parameters
        ----------
        path : str or path-like object
            The filesystem path to write the report to. By convention,
            this should use the `.pdf` file extension.

        template : str or path-like object, optional
            The filesystem path of the LaTeX template to
            fill. Relative to the pygsti template directoy.

        latex_cmd : str, optional
            Shell command to run to compile a PDF document from the
            generated LaTeX source.

        latex_flags : [str], optional
            List of flags to pass when calling `latex_cmd`.

        precision : int or dict, optional
            The amount of precision to display.  A dictionary with keys
            "polar", "sci", and "normal" can separately specify the
            precision for complex angles, numbers in scientific notation, and
            everything else, respectively.  If an integer is given, it this
            same value is taken for all precision types.  If None, then
            `{'normal': 6, 'polar': 3, 'sci': 0}` is used.

        auto_open : bool, optional
            Whether the output file should be automatically opened in a web browser.

        comm : mpi4py.MPI.Comm, optional
            When not None, an MPI communicator for distributing the computation
            across multiple processors.

        verbosity : int, optional
            Amount of detail to print to stdout.
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
        del global_qtys['topSwitchboard']
        del global_qtys['maxLSwitchboard1']

        printer.log("Generating LaTeX source...")
        _merge.merge_latex_template(
            global_qtys, template, str(path.with_suffix('.tex')),
            self._toggles, precision, printer
        )

        printer.log("Compiling with `{} {}`".format(latex_cmd, ' '.join(latex_flags)))
        _merge.compile_latex_report(str(path.parent / path.stem), [latex_cmd] + latex_flags, printer, auto_open)

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
from collections import defaultdict as _defaultdict, OrderedDict as _OrderedDict

from . import autotitle as _autotitle
from . import merge_helpers as _merge
from .. import _version, tools as _tools
from ..objects import VerbosityPrinter as _VerbosityPrinter, ExplicitOpModel as _ExplicitOpModel
from . import workspace as _ws

ROBUST_SUFFIX_LIST = [".robust", ".Robust", ".robust+", ".Robust+", ".wildcard"]


class Report:
    """ The internal model of a report.

    This class should never be instantiated directly. Instead, users
    should use the appropriate factory method in
    `pygsti.report.factory`.

    """
    def __init__(self, templates, results, sections, flags, global_qtys, report_params, workspace=None):
        self._templates = templates
        self._results = results
        self._sections = sections
        self._flags = flags
        self._global_qtys = global_qtys
        self._report_params = report_params
        self._workspace = workspace or _ws.Workspace()

    def _build(self, build_params=None):
        """ Render all sections to a map of report elements for templating """
        build_params = build_params or {}
        full_params = {
            'results': self._results,
            **self._report_params,
            **build_params,
        }
        qtys = self._global_qtys.copy()
        for section in self._sections:
            qtys.update(section.render(self._workspace, **full_params))

        return qtys

    def write_html(self, path, auto_open=False, link_to=None,
                   connected=False, brevity=0, errgen_type='logGTi',
                   ci_brevity=1, bgcolor='white', precision=None,
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

        brevity : int, optional
            Amount of detail to include in the report.  Larger values mean smaller
            "more briefr" reports, which reduce generation time, load time, and
            disk space consumption.  In particular:

            - 1: Plots showing per-sequences quantities disappear at brevity=1
            - 2: Reference sections disappear at brevity=2
            - 3: Germ-level estimate tables disappear at brevity=3
            - 4: Everything but summary figures disappears at brevity=4

        errgen_type: {"logG-logT", "logTiG", "logGTi"}, optional
            The type of error generator to compute.  Allowed values are:

            - "logG-logT" : errgen = log(gate) - log(target_op)
            - "logTiG" : errgen = log( dot(inv(target_op), gate) )
            - "logGTi" : errgen = log( dot(gate, inv(target_op)) )

        ci_brevity : int, optional
            Roughly specifies how many figures will have confidence intervals
            (when applicable). Defaults to '1'.  Smaller values mean more
            tables will get confidence intervals (and reports will take longer
            to generate).

        bgcolor : str, optional
            Background color for the color box plots in this report.  Can be common
            color names, e.g. `"black"`, or string RGB values, e.g. `"rgb(255,128,0)"`.

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

        toggles = _defaultdict(lambda: False)
        toggles.update(
            {k: True for k in self._flags}
        )
        for k in range(brevity, 4):
            toggles['BrevityLT' + str(k + 1)] = True

        # Render sections
        qtys = self._build(build_params=dict(
            brevity=brevity,
            errgen_type=errgen_type,
            ci_brevity=ci_brevity,
            bgcolor=bgcolor,
            embed_figures=embed_figures
        ))

        if single_file:
            _merge.merge_jinja_template(
                qtys, path, templateDir=self._templates['html'],
                auto_open=auto_open, precision=precision,
                link_to=link_to, connected=connected, toggles=toggles,
                renderMath=True, resizable=resizable,
                autosize=autosize, verbosity=verbosity
            )
        else:
            _merge.merge_jinja_template_dir(
                qtys, path, templateDir=self._templates['html'],
                auto_open=auto_open, precision=precision,
                link_to=link_to, connected=connected, toggles=toggles,
                renderMath=True, resizable=resizable,
                autosize=autosize, embed_figures=embed_figures,
                verbosity=verbosity
            )

    def write_notebook(self, path):
        """ Write this report to the disk as an IPython notebook

        Parameters
        ----------
        path : str or path-like object
            The filesystem path to write the report to. By convention,
            this should use the `.ipynb` file extension.
        """
        pass  # TODO

    def write_pdf(self, path, latex_cmd='pdflatex', latex_flags=None,
                  brevity=0, errgen_type='logGTi', ci_brevity=1,
                  bgcolor='white', precision=None, auto_open=False,
                  comm=None, verbosity=0):
        """ Write this report to the disk as a PDF document.

        Parameters
        ----------
        path : str or path-like object
            The filesystem path to write the report to. By convention,
            this should use the `.pdf` file extension.

        latex_cmd : str, optional
            Shell command to run to compile a PDF document from the
            generated LaTeX source.

        latex_flags : [str], optional
            List of flags to pass when calling `latex_cmd`.

        brevity : int, optional
            Amount of detail to include in the report.  Larger values mean smaller
            "more briefr" reports, which reduce generation time, load time, and
            disk space consumption.  In particular:

            - 1: Plots showing per-sequences quantities disappear at brevity=1
            - 2: Reference sections disappear at brevity=2
            - 3: Germ-level estimate tables disappear at brevity=3
            - 4: Everything but summary figures disappears at brevity=4

        errgen_type: {"logG-logT", "logTiG", "logGTi"}, optional
            The type of error generator to compute.  Allowed values are:

            - "logG-logT" : errgen = log(gate) - log(target_op)
            - "logTiG" : errgen = log( dot(inv(target_op), gate) )
            - "logGTi" : errgen = log( dot(gate, inv(target_op)) )

        ci_brevity : int, optional
            Roughly specifies how many figures will have confidence intervals
            (when applicable). Defaults to '1'.  Smaller values mean more
            tables will get confidence intervals (and reports will take longer
            to generate).

        bgcolor : str, optional
            Background color for the color box plots in this report.  Can be common
            color names, e.g. `"black"`, or string RGB values, e.g. `"rgb(255,128,0)"`.

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

        if len(self._results) > 1:
            raise ValueError("PDF reports cannot be generated for multiple result objects")

        toggles = _defaultdict(lambda: False)
        toggles.update(
            {k: True for k in self._flags}
        )
        for k in range(brevity, 4):
            toggles['BrevityLT' + str(k + 1)] = True

        printer = _VerbosityPrinter.build_printer(verbosity, comm=comm)
        path = _Path(path)
        latex_flags = latex_flags or ["-interaction=nonstopmode", "-halt-on-error", "-shell-escape"]

        # Render sections
        qtys = self._build(build_params=dict(
            brevity=brevity,
            errgen_type=errgen_type,
            ci_brevity=ci_brevity,
            bgcolor=bgcolor
        ))
        # TODO: filter while generating plots to remove need for sanitization
        qtys = {k: v for k, v in qtys.items()
                if not(isinstance(v, _ws.Switchboard) or isinstance(v, _ws.SwitchboardView))}

        printer.log("Generating LaTeX source...")
        _merge.merge_latex_template(
            qtys, self._templates['pdf'], str(path.with_suffix('.tex')),
            toggles, precision, printer
        )

        printer.log("Compiling with `{} {}`".format(latex_cmd, ' '.join(latex_flags)))
        _merge.compile_latex_report(str(path.parent / path.stem), [latex_cmd] + latex_flags, printer, auto_open)

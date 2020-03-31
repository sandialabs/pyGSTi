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
import pickle as _pickle

from . import autotitle as _autotitle
from . import merge_helpers as _merge
from .. import _version, tools as _tools
from ..objects import VerbosityPrinter as _VerbosityPrinter, ExplicitOpModel as _ExplicitOpModel
from . import workspace as _ws
from .notebook import Notebook as _Notebook


# TODO this whole thing needs to be rewritten with different reports as derived classes
class Report:
    """ The internal model of a report.

    This class should never be instantiated directly. Instead, users
    should use the appropriate factory method in
    `pygsti.report.factory`.

    """
    def __init__(self, templates, results, sections, flags,
                 global_qtys, report_params, build_defaults=None,
                 pdf_available=True, workspace=None):
        self._templates = templates
        self._results = results
        self._sections = sections
        self._flags = flags
        self._global_qtys = global_qtys
        self._report_params = report_params
        self._workspace = workspace or _ws.Workspace()
        self._build_defaults = build_defaults or {}
        self._pdf_available = pdf_available

    def _build(self, build_options=None):
        """ Render all sections to a map of report elements for templating """
        full_params = {
            'results': self._results,
            **self._report_params
        }
        full_params.update(self._build_defaults)
        full_params.update(build_options or {})
        qtys = self._global_qtys.copy()
        for section in self._sections:
            qtys.update(section.render(self._workspace, **full_params))

        return qtys

    def write_html(self, path, auto_open=False, link_to=None,
                   connected=False, build_options=None, brevity=0,
                   precision=None, resizable=True, autosize='initial',
                   single_file=False, verbosity=0):
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

        build_options : dict
           Dict of options for building plots. Expected values are
           defined during construction of this report object.

        brevity : int, optional
            Amount of detail to include in the report.  Larger values mean smaller
            "more briefr" reports, which reduce generation time, load time, and
            disk space consumption.  In particular:

            - 1: Plots showing per-sequences quantities disappear at brevity=1
            - 2: Reference sections disappear at brevity=2
            - 3: Germ-level estimate tables disappear at brevity=3
            - 4: Everything but summary figures disappears at brevity=4

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

        single_file : bool, optional
            If true, the report will be written to a single HTML
            document, with external dependencies baked-in. This mode
            is not recommended for large reports, because this file
            can grow large enough that major web browsers may struggle
            to render it.

        verbosity : int, optional
            Amount of detail to print to stdout.
        """

        build_options = build_options or {}

        toggles = _defaultdict(lambda: False)
        toggles.update(
            {k: True for k in self._flags}
        )
        for k in range(brevity, 4):
            toggles['BrevityLT' + str(k + 1)] = True

        # Render sections
        qtys = self._build(build_options)

        # TODO this really should be a parameter of this method
        embed_figures = self._report_params.get('embed_figures', True)

        if single_file:
            assert(embed_figures), \
                "Single-file mode requires `embed_figures` to be True"
            _merge.merge_jinja_template(
                qtys, path, template_dir=self._templates['html'],
                auto_open=auto_open, precision=precision,
                link_to=link_to, connected=connected, toggles=toggles,
                render_math=True, resizable=resizable,
                autosize=autosize, verbosity=verbosity
            )
        else:
            _merge.merge_jinja_template_dir(
                qtys, path, template_dir=self._templates['html'],
                auto_open=auto_open, precision=precision,
                link_to=link_to, connected=connected, toggles=toggles,
                render_math=True, resizable=resizable,
                autosize=autosize, embed_figures=embed_figures,
                verbosity=verbosity
            )

    def write_notebook(self, path, auto_open=False, connected=False, verbosity=0):
        """ Write this report to the disk as an IPython notebook

        A notebook report allows the user to interact more flexibly with the data
        underlying the figures, and to easily generate customized variants on the
        figures.  As such, this type of report will be most useful for experts
        who want to tinker with the standard analysis presented in the static
        HTML or LaTeX format reports.

        Parameters
        ----------
        path : str or path-like object
            The filesystem path to write the report to. By convention,
            this should use the `.ipynb` file extension.
        auto_open : bool, optional
            If True, automatically open the report in a web browser after it
            has been generated.

        connected : bool, optional
            Whether output notebook should assume an active internet connection.  If
            True, then the resulting file size will be reduced because it will link
            to web resources (e.g. CDN libraries) instead of embedding them.

        verbosity : int, optional
           How much detail to send to stdout.
        """

        # TODO this only applies to standard reports; rewrite generally
        title = self._global_qtys['title']
        confidenceLevel = self._report_params['confidence_level']

        path = _Path(path)
        printer = _VerbosityPrinter.build_printer(verbosity)
        templatePath = _Path(__file__).parent / 'templates' / self._templates['notebook']
        outputDir = path.parent

        #Copy offline directory into position
        if not connected:
            _merge.rsync_offline_dir(outputDir)

        #Save results to file
        # basename = _os.path.splitext(_os.path.basename(filename))[0]
        basename = path.stem
        results_file_base = basename + '_results.pkl'
        results_file = outputDir / results_file_base
        with open(str(results_file), 'wb') as f:
            _pickle.dump(self._results, f)

        nb = _Notebook()
        nb.add_markdown('# {title}\n(Created on {date})'.format(
            title=title, date=_time.strftime("%B %d, %Y")))

        nb.add_code("""\
            import pickle
            import pygsti""")

        dsKeys = list(self._results.keys())
        results = self._results[dsKeys[0]]
        #Note: `results` is always a single Results obj from here down

        nb.add_code("""\
        #Load results dictionary
        with open('{infile}', 'rb') as infile:
            results_dict = pickle.load(infile)
        print("Available dataset keys: ", ', '.join(results_dict.keys()))\
        """.format(infile=results_file_base))

        nb.add_code("""\
        #Set which dataset should be used below
        results = results_dict['{dsKey}']
        print("Available estimates: ", ', '.join(results.estimates.keys()))\
        """.format(dsKey=dsKeys[0]))

        estLabels = list(results.estimates.keys())
        estimate = results.estimates[estLabels[0]]
        nb.add_code("""\
        #Set which estimate is to be used below
        estimate = results.estimates['{estLabel}']
        print("Available gauge opts: ", ', '.join(estimate.goparameters.keys()))\
        """.format(estLabel=estLabels[0]))

        goLabels = list(estimate.goparameters.keys())
        nb.add_code("""\
            gopt      = '{goLabel}'
            ds        = results.dataset

            circuits_final = results.circuit_lists['final']
            circuits_per_iter = results.circuit_lists['iteration']  # All L-values
            if isinstance(circuit_final, pygsti.objs.BulkCircuitList):
                Ls = circuits_final.circuit_structure.Ls

            prep_fiducials = results.circuit_lists['prep fiducials']
            meas_fiducials = results.circuit_lists['meas fiducials']
            germs = results.circuit_lists['germs']

            params = estimate.parameters
            objfn_builder = estimate.parameters.get('final_objfn_builder', 'logl')
            clifford_compilation = estimate.parameters.get('clifford_compilation',None)
            effective_ds, scale_submxs = estimate.get_effective_dataset(True)

            models        = estimate.models
            mdl           = models[gopt]  # final, gauge-optimized estimate
            mdl_final     = models['final iteration estimate'] # final estimate before gauge-opt
            target_model  = models['target']
            mdl_per_iter  = models['iteration estimates']

            mdl_eigenspace_projected = pygsti.tools.project_to_target_eigenspace(mdl, target_model)

            goparams = estimate.goparameters[gopt]

            confidenceLevel = {CL}
            if confidenceLevel is None:
                cri = None
            else:
                crfactory = estimate.get_confidence_region_factory(gopt)
                region_type = "normal" if confidenceLevel >= 0 else "non-markovian"
                cri = crfactory.view(abs(confidenceLevel), region_type)\
        """.format(goLabel=goLabels[0], CL=confidenceLevel))

        nb.add_code("""\
            from pygsti.report import Workspace
            ws = Workspace()
            ws.init_notebook_mode(connected={conn}, autodisplay=True)\
            """.format(conn=str(connected)))

        nb.add_notebook_text_files([
            templatePath / 'summary.txt',
            templatePath / 'goodness.txt',
            templatePath / 'gauge_invariant.txt',
            templatePath / 'gauge_variant.txt'])

        #Insert multi-dataset specific analysis
        if len(dsKeys) > 1:
            nb.add_markdown(('# Dataset comparisons\n'
                             'This report contains information for more than one data set.'
                             'This page shows comparisons between different data sets.'))

            nb.add_code("""\
            dslbl1 = '{dsLbl1}'
            dslbl2 = '{dsLbl2}'
            dscmp_circuits = results_dict[dslbl1].circuit_lists['final']
            ds1 = results_dict[dslbl1].dataset
            ds2 = results_dict[dslbl2].dataset
            dscmp = pygsti.obj.DataComparator([ds1, ds2], ds_names=[dslbl1, dslbl2])
            """.format(dsLbl1=dsKeys[0], dsLbl2=dsKeys[1]))
            nb.add_notebook_text_files([
                templatePath / 'data_comparison.txt'])

        #Add reference material
        nb.add_notebook_text_files([
            templatePath / 'input.txt',
            templatePath / 'meta.txt'])

        printer.log("Report Notebook created as %s" % path)

        if auto_open:
            port = "auto" if auto_open is True else int(auto_open)
            nb.launch(str(path), port=port)
        else:
            nb.save_to(str(path))

    def write_pdf(self, path, latex_cmd='pdflatex', latex_flags=None,
                  build_options=None,
                  brevity=0, precision=None, auto_open=False,
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

        build_options : dict
           Dict of options for building plots. Expected values are
           defined during construction of this report object.

        brevity : int, optional
            Amount of detail to include in the report.  Larger values mean smaller
            "more briefr" reports, which reduce generation time, load time, and
            disk space consumption.  In particular:

            - 1: Plots showing per-sequences quantities disappear at brevity=1
            - 2: Reference sections disappear at brevity=2
            - 3: Germ-level estimate tables disappear at brevity=3
            - 4: Everything but summary figures disappears at brevity=4

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

        if not self._pdf_available:
            raise ValueError(("PDF output unavailable.  (Usually this is because this report"
                              " has multiple gauge optimizations and/or datasets.)"))

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
        qtys = self._build(build_options)
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

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
    def __init__(self, results, sections, flags, global_qtys, report_params):
        self._results = results
        self._sections = sections
        self._flags = flags
        self._global_qtys = global_qtys
        self._report_params = report_params

    def _build(self, workspace, build_params=None):
        """ Render all sections to a map of report elements for templating """
        build_params = build_params or {}
        full_params = {
            'results': self._results,
            **self._report_params,
            **build_params,
        }
        qtys = self._global_qtys.copy()
        for section in self._sections:
            qtys.update(section.render(workspace, **full_params))

        pdfInfo = [('Author', 'pyGSTi'), ('Title', qtys.get('title', 'untitled')),
                   ('Keywords', 'GST'), ('pyGSTi Version', _version.__version__)]
        qtys['pdfinfo'] = _merge.to_pdfinfo(pdfInfo)

        return qtys

    def write_html(self, path, template_dir='~standard_html_report',
                   template_name='main.html', auto_open=False,
                   link_to=None, connected=False, brevity=0,
                   errgen_type='logGTi', ci_brevity=1,
                   bgcolor='white', precision=None, resizable=True,
                   autosize='initial', embed_figures=True,
                   single_file=False, workspace=None, verbosity=0):
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

        workspace : Workspace, optional
            The workspace used as a scratch space for performing the calculations
            and visualizations required for this report.  If you're creating
            multiple reports with similar tables, plots, etc., it may boost
            performance to use a single Workspace for all the report generation.

        verbosity : int, optional
            Amount of detail to print to stdout.
        """

        workspace = workspace or _ws.Workspace()

        toggles = _defaultdict(lambda: False)
        toggles.update(
            {k: True for k in self._flags}
        )
        for k in range(brevity, 4):
            toggles['BrevityLT' + str(k + 1)] = True


        # Render sections
        qtys = self._build(workspace, build_params=dict(
            brevity=brevity,
            errgen_type=errgen_type,
            ci_brevity=ci_brevity,
            bgcolor=bgcolor,
            embed_figures=embed_figures
        ))

        # TODO refactor all rendering into this method and section rendering methods
        if single_file:
            _merge.merge_jinja_template(
                qtys, path, templateDir=template_dir,
                templateName=template_name, auto_open=auto_open,
                precision=precision, link_to=link_to, connected=connected,
                toggles=toggles, renderMath=True, resizable=resizable,
                autosize=autosize, verbosity=verbosity
            )
        else:
            _merge.merge_jinja_template_dir(
                qtys, path, templateDir=template_dir,
                templateName=template_name, auto_open=auto_open,
                precision=precision, link_to=link_to, connected=connected,
                toggles=toggles, renderMath=True, resizable=resizable,
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
        self.prepare_notebook()
        # TODO

    def write_pdf(self, path, template="standard_pdf_report.tex",
                  latex_cmd='pdflatex', latex_flags=None, brevity=0,
                  errgen_type='logGTi', ci_brevity=1, bgcolor='white',
                  precision=None, auto_open=False, comm=None,
                  workspace=None, verbosity=0):
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

        workspace : Workspace, optional
            The workspace used as a scratch space for performing the calculations
            and visualizations required for this report.  If you're creating
            multiple reports with similar tables, plots, etc., it may boost
            performance to use a single Workspace for all the report generation.

        verbosity : int, optional
            Amount of detail to print to stdout.
        """

        if len(self._results) > 1:
            raise ValueError("PDF reports cannot be generated for multiple result objects")

        workspace = workspace or _ws.Workspace()

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
        qtys = self._build(workspace, build_params=dict(
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
            qtys, template, str(path.with_suffix('.tex')),
            toggles, precision, printer
        )

        printer.log("Compiling with `{} {}`".format(latex_cmd, ' '.join(latex_flags)))
        _merge.compile_latex_report(str(path.parent / path.stem), [latex_cmd] + latex_flags, printer, auto_open)


def _add_new_labels(running_lbls, current_lbls):
    """
    Simple routine to add current-labels to a list of
    running-labels without introducing duplicates and
    preserving order as best we can.
    """
    if running_lbls is None:
        return current_lbls[:]  # copy!
    elif running_lbls != current_lbls:
        for lbl in current_lbls:
            if lbl not in running_lbls:
                running_lbls.append(lbl)
    return running_lbls


def _add_new_estimate_labels(running_lbls, estimates, combine_robust):
    """
    Like _add_new_labels but perform robust-suffix processing.

    In particular, if `combine_robust == True` then do not add
    labels which have a ".robust" counterpart.
    """
    current_lbls = list(estimates.keys())

    def _add_lbl(lst, lbl):
        if combine_robust and any([(lbl + suffix in current_lbls)
                                   for suffix in ROBUST_SUFFIX_LIST]):
            return  # don't add label
        lst.append(lbl)  # add label

    if running_lbls is None:
        running_lbls = []

    if running_lbls != current_lbls:
        for lbl in current_lbls:
            if lbl not in running_lbls:
                _add_lbl(running_lbls, lbl)

    return running_lbls


def _get_viewable_crf(est, est_lbl, mdl_lbl, verbosity=0):
    printer = _VerbosityPrinter.build_printer(verbosity)

    if est.has_confidence_region_factory(mdl_lbl, 'final'):
        crf = est.get_confidence_region_factory(mdl_lbl, 'final')
        if crf.can_construct_views():
            return crf
        else:
            printer.log(
                ("Note: Confidence interval factory for {estlbl}.{gslbl} "
                 "model exists but cannot create views.  This could be "
                 "because you forgot to create a Hessian *projection*"
                 ).format(estlbl=est_lbl, gslbl=mdl_lbl))
    else:
        printer.log(
            ("Note: no factory to compute confidence "
             "intervals for the '{estlbl}.{gslbl}' model."
             ).format(estlbl=est_lbl, gslbl=mdl_lbl))

    return None


def _find_std_clifford_compilation(model, verbosity=0):
    """
    Returns the standard Clifford compilation for `model`, if
    one exists.  Otherwise returns None.

    Parameters
    ----------
    model : Model
        The ideal (target) model of primitive gates.

    verbosity : int, optional
        How much detail to send to stdout.

    Returns
    -------
    dict or None
        The Clifford compilation dictionary (if one can be found).
    """
    printer = _VerbosityPrinter.build_printer(verbosity)
    if not isinstance(model, _ExplicitOpModel):
        return None  # only match explicit models

    std_modules = ("std1Q_XY",
                   "std1Q_XYI",
                   "std1Q_XYZI",
                   "std1Q_XZ",
                   "std1Q_ZN",
                   "std1Q_pi4_pi2_XZ",
                   "std2Q_XXII",
                   "std2Q_XXYYII",
                   "std2Q_XY",
                   "std2Q_XYCNOT",
                   "std2Q_XYCPHASE",
                   "std2Q_XYI",
                   "std2Q_XYI1",
                   "std2Q_XYI2",
                   "std2Q_XYICNOT",
                   "std2Q_XYICPHASE",
                   "std2Q_XYZICNOT")
    import importlib
    for module_name in std_modules:
        mod = importlib.import_module("pygsti.construction." + module_name)
        target_model = mod.target_model()
        if target_model.dim == model.dim and \
           set(target_model.operations.keys()) == set(model.operations.keys()) and \
           set(target_model.preps.keys()) == set(model.preps.keys()) and \
           set(target_model.povms.keys()) == set(model.povms.keys()):
            if target_model.frobeniusdist(model) < 1e-6:
                if hasattr(mod, "clifford_compilation"):
                    printer.log("Found standard clifford compilation from %s" % module_name)
                    return mod.clifford_compilation
    return None


def _create_master_switchboard(ws, results_dict, confidenceLevel,
                               nmthreshold, printer, fmt,
                               combine_robust, idt_results_dict=None, embed_figures=True):
    """
    Creates the "master switchboard" used by several of the reports
    """

    if isinstance(results_dict, _OrderedDict):
        dataset_labels = list(results_dict.keys())
    else:
        dataset_labels = sorted(list(results_dict.keys()))

    est_labels = None
    gauge_opt_labels = None
    Ls = None

    for results in results_dict.values():
        est_labels = _add_new_estimate_labels(est_labels, results.estimates,
                                              combine_robust)
        Ls = _add_new_labels(Ls, results.circuit_structs['final'].Ls)
        for est in results.estimates.values():
            gauge_opt_labels = _add_new_labels(gauge_opt_labels,
                                               list(est.goparameters.keys()))

    Ls = list(sorted(Ls))  # make sure Ls are sorted in increasing order

    # XXX i suspect it's not actually this easy
    # if fmt == "latex" and len(Ls) > 0:
    #     swLs = [Ls[-1]]  # "switched Ls" = just take the single largest L
    # else:
    #     swLs = Ls  # switch over all Ls
    swLs = Ls

    multidataset = bool(len(dataset_labels) > 1)
    multiest = bool(len(est_labels) > 1)
    multiGO = bool(len(gauge_opt_labels) > 1)
    #multiL = bool(len(swLs) > 1)

    switchBd = ws.Switchboard(
        ["Dataset", "Estimate", "Gauge-Opt", "max(L)"],
        [dataset_labels, est_labels, gauge_opt_labels, list(map(str, swLs))],
        ["dropdown", "dropdown", "buttons", "slider"], [0, 0, 0, len(swLs) - 1],
        show=[multidataset, multiest, multiGO, False],  # "global" switches only + gauge-opt (OK if doesn't apply)
        use_loadable_items=embed_figures
    )

    switchBd.add("ds", (0,))
    switchBd.add("prepStrs", (0,))
    switchBd.add("effectStrs", (0,))
    switchBd.add("strs", (0,))
    switchBd.add("germs", (0,))

    switchBd.add("eff_ds", (0, 1))
    switchBd.add("modvi_ds", (0, 1))
    switchBd.add("wildcardBudget", (0, 1))
    switchBd.add("scaledSubMxsDict", (0, 1))
    switchBd.add("gsTarget", (0, 1))
    switchBd.add("params", (0, 1))
    switchBd.add("objective", (0, 1))
    switchBd.add("objective_tvd_tuple", (0, 1))
    switchBd.add("objective_modvi", (0, 1))
    switchBd.add("mpc", (0, 1))
    switchBd.add("mpc_modvi", (0, 1))
    switchBd.add("clifford_compilation", (0, 1))
    switchBd.add("meta_stdout", (0, 1))
    switchBd.add("profiler", (0, 1))

    switchBd.add("gsGIRep", (0, 1))
    switchBd.add("gsGIRepEP", (0, 1))
    switchBd.add("gsFinal", (0, 1, 2))
    switchBd.add("gsEvalProjected", (0, 1, 2))
    switchBd.add("gsTargetAndFinal", (0, 1, 2))  # general only!
    switchBd.add("goparams", (0, 1, 2))
    switchBd.add("gsL", (0, 1, 3))
    switchBd.add("gsL_modvi", (0, 1, 3))
    switchBd.add("gss", (0, 3))
    switchBd.add("gssFinal", (0,))
    switchBd.add("gsAllL", (0, 1))
    switchBd.add("gsAllL_modvi", (0, 1))
    switchBd.add("gssAllL", (0,))
    switchBd.add("gsFinalGrid", (2,))

    switchBd.add("idtresults", (0,))

    if confidenceLevel is not None:
        switchBd.add("cri", (0, 1, 2))
        switchBd.add("criGIRep", (0, 1))

    for d, dslbl in enumerate(dataset_labels):
        results = results_dict[dslbl]

        switchBd.ds[d] = results.dataset
        switchBd.prepStrs[d] = results.circuit_lists['prep fiducials']
        switchBd.effectStrs[d] = results.circuit_lists['effect fiducials']
        switchBd.strs[d] = (results.circuit_lists['prep fiducials'],
                            results.circuit_lists['effect fiducials'])
        switchBd.germs[d] = results.circuit_lists['germs']

        switchBd.gssFinal[d] = results.circuit_structs['final']
        for iL, L in enumerate(swLs):  # allow different results to have different Ls
            if L in results.circuit_structs['final'].Ls:
                k = results.circuit_structs['final'].Ls.index(L)
                switchBd.gss[d, iL] = results.circuit_structs['iteration'][k]
        switchBd.gssAllL[d] = results.circuit_structs['iteration']

        if idt_results_dict is not None:
            switchBd.idtresults[d] = idt_results_dict.get(dslbl, None)

        for i, lbl in enumerate(est_labels):
            est = results.estimates.get(lbl, None)
            if est is None: continue

            for suffix in ROBUST_SUFFIX_LIST:
                if combine_robust and lbl.endswith(suffix):
                    est_modvi = results.estimates.get(lbl[:-len(suffix)], est)
                    break
            else:
                est_modvi = est

            def rpt_objective(opt_objective):
                """ If optimized using just LGST, compute logl values """
                if opt_objective == "lgst": return "logl"
                else: return opt_objective

            switchBd.params[d, i] = est.parameters
            switchBd.objective[d, i] = rpt_objective(est.parameters['objective'])
            switchBd.objective_tvd_tuple[d, i] = (rpt_objective(est.parameters['objective']), 'tvd')
            switchBd.objective_modvi[d, i] = rpt_objective(est_modvi.parameters['objective'])
            if est.parameters['objective'] == "logl":
                switchBd.mpc[d, i] = est.parameters['minProbClip']
                switchBd.mpc_modvi[d, i] = est_modvi.parameters['minProbClip']
            elif est.parameters['objective'] == "chi2":
                switchBd.mpc[d, i] = est.parameters['minProbClipForWeighting']
                switchBd.mpc_modvi[d, i] = est_modvi.parameters['minProbClipForWeighting']
            else:  # "lgst" - just use defaults for logl
                switchBd.mpc[d, i] = 1e-4
                switchBd.mpc_modvi[d, i] = 1e-4
            switchBd.clifford_compilation[d, i] = est.parameters.get("clifford compilation", 'auto')
            if switchBd.clifford_compilation[d, i] == 'auto':
                switchBd.clifford_compilation[d, i] = _find_std_clifford_compilation(
                    est.models['target'], printer)

            switchBd.profiler[d, i] = est_modvi.parameters.get('profiler', None)
            switchBd.meta_stdout[d, i] = est_modvi.meta.get('stdout', [('LOG', 1, "No standard output recorded")])

            GIRepLbl = 'final iteration estimate'  # replace with a gauge-opt label if it has a CI factory
            if confidenceLevel is not None:
                if _get_viewable_crf(est, lbl, GIRepLbl) is None:
                    for l in gauge_opt_labels:
                        if _get_viewable_crf(est, lbl, l) is not None:
                            GIRepLbl = l; break

            # NOTE on modvi_ds (the dataset used in model violation plots)
            # if combine_robust is True, modvi_ds is the unscaled dataset.
            # if combine_robust is False, modvi_ds is the effective dataset
            #     for the estimate (potentially just the unscaled one)

            #if this estimate uses robust scaling or wildcard budget
            NA = ws.NotApplicable()
            if est.parameters.get("weights", None):
                effds, scale_subMxs = est.get_effective_dataset(True)
                switchBd.eff_ds[d, i] = effds
                switchBd.scaledSubMxsDict[d, i] = {'scaling': scale_subMxs, 'scaling.colormap': "revseq"}
                switchBd.modvi_ds[d, i] = results.dataset if combine_robust else effds
            else:
                switchBd.modvi_ds[d, i] = results.dataset
                switchBd.eff_ds[d, i] = NA
                switchBd.scaledSubMxsDict[d, i] = NA

            if est.parameters.get("unmodeled_error", None):
                switchBd.wildcardBudget[d, i] = est.parameters['unmodeled_error']
            else:
                switchBd.wildcardBudget[d, i] = NA

            switchBd.gsTarget[d, i] = est.models['target']
            switchBd.gsGIRep[d, i] = est.models[GIRepLbl]
            try:
                switchBd.gsGIRepEP[d, i] = _tools.project_to_target_eigenspace(est.models[GIRepLbl],
                                                                               est.models['target'])
            except AttributeError:  # Implicit models don't support everything, like set_all_parameterizations
                switchBd.gsGIRepEP[d, i] = None
            except AssertionError:  # if target is badly off, this can fail with an imaginary part assertion
                switchBd.gsGIRepEP[d, i] = None

            switchBd.gsFinal[d, i, :] = [est.models.get(l, NA) for l in gauge_opt_labels]
            switchBd.gsTargetAndFinal[d, i, :] = \
                [[est.models['target'], est.models[l]] if (l in est.models) else NA
                 for l in gauge_opt_labels]
            switchBd.goparams[d, i, :] = [est.goparameters.get(l, NA) for l in gauge_opt_labels]

            for iL, L in enumerate(swLs):  # allow different results to have different Ls
                if L in results.circuit_structs['final'].Ls:
                    k = results.circuit_structs['final'].Ls.index(L)
                    switchBd.gsL[d, i, iL] = est.models['iteration estimates'][k]
                    switchBd.gsL_modvi[d, i, iL] = est_modvi.models['iteration estimates'][k]
            switchBd.gsAllL[d, i] = est.models['iteration estimates']
            switchBd.gsAllL_modvi[d, i] = est_modvi.models['iteration estimates']

            if confidenceLevel is not None:
                misfit_sigma = est.misfit_sigma(use_accurate_Np=True)

                for il, l in enumerate(gauge_opt_labels):
                    if l in est.models:
                        switchBd.cri[d, i, il] = None  # default
                        crf = _get_viewable_crf(est, lbl, l, printer - 2)

                        if crf is not None:
                            #Check whether we should use non-Markovian error bars:
                            # If fit is bad, check if any reduced fits were computed
                            # that we can use with in-model error bars.  If not, use
                            # experimental non-markovian error bars.
                            region_type = "normal" if misfit_sigma <= nmthreshold \
                                          else "non-markovian"
                            switchBd.cri[d, i, il] = crf.view(confidenceLevel, region_type)

                    else: switchBd.cri[d, i, il] = NA

                # "Gauge Invariant Representation" model
                # If we can't compute CIs for this, ignore SILENTLY, since any
                #  relevant warnings/notes should have been given above.
                switchBd.criGIRep[d, i] = None  # default
                crf = _get_viewable_crf(est, lbl, GIRepLbl)
                if crf is not None:
                    region_type = "normal" if misfit_sigma <= nmthreshold \
                                  else "non-markovian"
                    switchBd.criGIRep[d, i] = crf.view(confidenceLevel, region_type)

    results_list = [results_dict[dslbl] for dslbl in dataset_labels]
    for i, gokey in enumerate(gauge_opt_labels):
        if multidataset:
            switchBd.gsFinalGrid[i] = [
                [(res.estimates[el].models.get(gokey, None)
                  if el in res.estimates else None) for el in est_labels]
                for res in results_list]
        else:
            switchBd.gsFinalGrid[i] = [
                (results_list[0].estimates[el].models.get(gokey, None)
                 if el in results_list[0].estimates else None) for el in est_labels]

    if multidataset:
        switchBd.add_unswitched('gsTargetGrid', [
            [(res.estimates[el].models.get('target', None)
              if el in res.estimates else None) for el in est_labels]
            for res in results_list])
    else:
        switchBd.add_unswitched('gsTargetGrid', [
            (results_list[0].estimates[el].models.get('target', None)
             if el in results_list[0].estimates else None) for el in est_labels])

    return switchBd, dataset_labels, est_labels, gauge_opt_labels, Ls, swLs

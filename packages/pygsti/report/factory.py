""" Report generation functions. """
from __future__ import division, print_function, absolute_import, unicode_literals
#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************

import pickle as _pickle
import os  as _os
import time as _time
import collections as _collections
import warnings as _warnings
import zipfile as _zipfile
import numpy as _np

from ..baseobjs import VerbosityPrinter as _VerbosityPrinter
from ..objects import DataComparator as _DataComparator
from ..tools   import timed_block as _timed_block

from ..tools.mpitools import distribute_indices as _distribute_indices

from .. import tools as _tools
from .. import _version

from . import workspace as _ws
from . import autotitle as _autotitle
from . import merge_helpers as _merge
from . import reportables as _reportables
from .notebook import Notebook as _Notebook

#maybe import these from drivers.longsequence so they stay synced?
ROBUST_SUFFIX_LIST = [".robust", ".Robust", ".robust+", ".Robust+"]
DEFAULT_BAD_FIT_THRESHOLD = 2.0

def _errgen_formula(errgen_type, typ):
    assert(typ in ('html','latex'))

    notDuringTxt = """This is <em>not</em> the Lindblad-type generator that would produce this noise if it acted continuously <em>during</em> the gate (i.e., simultaneously with a Hamiltonian that generates the ideal gate).  This choice is explicit; the authors of pyGSTi are concerned that reporting the continuous-time-generator would encourage a false sense of understanding the physics behind the noise, which is explicitly invalid if the gates were produced by anything other than a simple pulse."""

    if errgen_type == "logTiG": # G = T*exp(L) (pre-error)
        gen = '<span class="math">G = G_0 e^{\mathbb{L}}</span>'
        desc = ('<em>pre-gate</em> generator, so it answers the question '
                '"If all the noise occurred <em>before</em> the ideal gate,'
                ' what Lindbladian would generate it?" ') + notDuringTxt
    elif errgen_type == "logGTi": # G = exp(L)*T (post-error)
        gen = '<span class="math">G = e^{\mathbb{L}} G_0</span>'
        desc = ('<em>post-gate</em> generator, so it answers the question '
                '"If all the noise occurred <em>after</em> the ideal gate,'
                ' what Lindbladian would generate it?" ') + notDuringTxt
    elif errgen_type == "logG-logT":
        gen = '<span class="math">G = e^{\mathbb{L} + \log G_0}</span>'
        desc = ('<em>during-gate</em> generator, so it answers the question '
                '"What Lindblad-type generate would produce this noise if it'
                ' acted continuously <em>during</em> the gate?"  Note that '
                'this does <em>not necessarily</em> give insight into physics'
                ' producing the noise.')
    else:
        gen = desc = "???"

    if typ == "latex": #minor modifications for latex version
        gen = gen.replace('<span class="math">','$')
        gen = gen.replace('</span>','$')
        desc = desc.replace('<em>','\\emph{')
        desc = desc.replace('</em>','}')

    return gen, desc

def _add_new_labels(running_lbls, current_lbls):
    """
    Simple routine to add current-labels to a list of
    running-labels without introducing duplicates and
    preserving order as best we can.
    """
    if running_lbls is None:
        return current_lbls[:] #copy!
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
        if combine_robust and any([(lbl+suffix in current_lbls)
                                   for suffix in ROBUST_SUFFIX_LIST]):
            return #don't add label
        lst.append(lbl) #add label

    if running_lbls is None:
        running_lbls = []

    if running_lbls != current_lbls:
        for lbl in current_lbls:
            if lbl not in running_lbls:
                _add_lbl(running_lbls, lbl)

    return running_lbls


#def _robust_estimate_has_same_gatesets(estimates, est_lbl):
#    lbl_robust = est_lbl+ROBUST_SUFFIX
#    if lbl_robust not in estimates: return False #no robust estimate
#
#    for gs_lbl in list(estimates[est_lbl].goparameters.keys()) \
#        + ['final iteration estimate']:
#        if gs_lbl not in estimates[lbl_robust].gatesets:
#            return False #robust estimate is missing gs_lbl!
#
#        gs = estimates[lbl_robust].gatesets[gs_lbl]
#        if estimates[est_lbl].gatesets[gs_lbl].frobeniusdist(gs) > 1e-8:
#            return False #gateset mismatch!
#
#    return True

def _get_viewable_crf(est, est_lbl, gs_lbl, verbosity=0):
    printer = _VerbosityPrinter.build_printer(verbosity)

    if est.has_confidence_region_factory(gs_lbl, 'final'):
        crf = est.get_confidence_region_factory(gs_lbl,'final')
        if crf.can_construct_views():
            return crf
        else:
            printer.log(
                ("Note: Confidence interval factory for {estlbl}.{gslbl} "
                 "gate set exists but cannot create views.  This could be "
                 "because you forgot to create a Hessian *projection*"
                ).format(estlbl=est_lbl,gslbl=gs_lbl))
    else:
        printer.log(
            ("Note: no factory to compute confidence "
             "intervals for the '{estlbl}.{gslbl}' gate set."
            ).format(estlbl=est_lbl,gslbl=gs_lbl))

    return None



def create_offline_zip(outputDir="."):
    """
    Creates a zip file containing the a directory ("offline") of files
    need to display "offline" reports (generated with `connected=False`).

    For offline reports to display, the "offline" folder must be placed
    in the same directory as the report's HTML file.  This function can
    be used to easily obtain a copy of the offline folder for the purpose
    of sharing offline reports with other people.  If you're just creating
    your own offline reports using pyGSTi, the offline folder is
    automatically copied into it's proper position - so you don't need
    to call this function.

    Parameters
    ----------
    outputDir : str, optional
        The directory in which "offline.zip" should be place.

    Returns
    -------
    None
    """
    templatePath = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                                "templates")

    zipFName = _os.path.join(outputDir, "offline.zip")
    zipHandle = _zipfile.ZipFile(zipFName, 'w', _zipfile.ZIP_DEFLATED)
    for root, _, files in _os.walk(_os.path.join(templatePath,"offline")):
        for f in files:
            fullPath = _os.path.join(root, f)
            zipHandle.write(fullPath, _os.path.relpath(fullPath,templatePath))
    zipHandle.close()

def _set_toggles(results_dict, brevity, combine_robust):
    #Determine when to get gatestring weight (scaling) values and show via
    # ColorBoxPlots below by checking whether any estimate has "weights"
    # parameter (a dict) with > 0 entries.
    toggles = { }

    toggles["ShowScaling"] = False
    for res in results_dict.values():
        for est in res.estimates.values():
            weights = est.parameters.get("weights",None)
            if weights is not None and len(weights) > 0:
                toggles["ShowScaling"] = True

    toggles['BrevityLT1'] = bool(brevity < 1)
    toggles['BrevityLT2'] = bool(brevity < 2)
    toggles['BrevityLT3'] = bool(brevity < 3)
    toggles['BrevityLT4'] = bool(brevity < 4)

    toggles['CombineRobust'] = bool(combine_robust)
    return toggles

def _create_master_switchboard(ws, results_dict, confidenceLevel,
                               nmthreshold, printer, fmt,
                               combine_robust):
    """
    Creates the "master switchboard" used by several of the reports
    """

    if isinstance(results_dict, _collections.OrderedDict):
        dataset_labels = list(results_dict.keys())
    else:
        dataset_labels = sorted(list(results_dict.keys()))

    est_labels = None
    gauge_opt_labels = None
    Ls = None

    for results in results_dict.values():
        est_labels = _add_new_estimate_labels(est_labels, results.estimates,
                                              combine_robust)
        Ls = _add_new_labels(Ls, results.gatestring_structs['final'].Ls)
        for est in results.estimates.values():
            gauge_opt_labels = _add_new_labels(gauge_opt_labels,
                                               list(est.goparameters.keys()))

    Ls = list(sorted(Ls)) #make sure Ls are sorted in increasing order
    if fmt == "latex" and len(Ls) > 0:
        swLs = [ Ls[-1] ] # "switched Ls" = just take the single largest L
    else:
        swLs = Ls #switch over all Ls


    multidataset = bool(len(dataset_labels) > 1)
    multiest = bool(len(est_labels) > 1)
    multiGO = bool(len(gauge_opt_labels) > 1)
    #multiL = bool(len(swLs) > 1)

    switchBd = ws.Switchboard(
        ["Dataset","Estimate","Gauge-Opt","max(L)"],
        [dataset_labels, est_labels, gauge_opt_labels, list(map(str,swLs))],
        ["dropdown","dropdown", "buttons", "slider"], [0,0,0,len(swLs)-1],
        show=[multidataset,multiest,multiGO,False] # "global" switches only + gauge-opt (OK if doesn't apply)
    )

    switchBd.add("ds",(0,))
    switchBd.add("prepStrs",(0,))
    switchBd.add("effectStrs",(0,))
    switchBd.add("strs",(0,))
    switchBd.add("germs",(0,))

    switchBd.add("eff_ds",(0,1))
    switchBd.add("modvi_ds",(0,1))
    switchBd.add("scaledSubMxsDict",(0,1))
    switchBd.add("gsTarget",(0,1))
    switchBd.add("params",(0,1))
    switchBd.add("objective",(0,1))
    switchBd.add("objective_tvd_tuple",(0,1))
    switchBd.add("objective_modvi",(0,1))
    switchBd.add("mpc",(0,1))
    switchBd.add("mpc_modvi",(0,1))
    switchBd.add("clifford_compilation",(0,1))
    switchBd.add("meta_stdout",(0,1))
    switchBd.add("profiler",(0,1))

    switchBd.add("gsGIRep",(0,1))
    switchBd.add("gsGIRepEP",(0,1))
    switchBd.add("gsFinal",(0,1,2))
    switchBd.add("gsEvalProjected",(0,1,2))
    switchBd.add("gsTargetAndFinal",(0,1,2)) #general only!
    switchBd.add("goparams",(0,1,2))
    switchBd.add("gsL",(0,1,3))
    switchBd.add("gsL_modvi",(0,1,3))
    switchBd.add("gss",(0,3))
    switchBd.add("gssFinal",(0,))
    switchBd.add("gsAllL",(0,1))
    switchBd.add("gsAllL_modvi",(0,1))
    switchBd.add("gssAllL",(0,))
    switchBd.add("gsFinalGrid",(2,))

    if confidenceLevel is not None:
        switchBd.add("cri",(0,1,2))
        switchBd.add("criGIRep",(0,1))

    for d,dslbl in enumerate(dataset_labels):
        results = results_dict[dslbl]

        switchBd.ds[d] = results.dataset
        switchBd.prepStrs[d] = results.gatestring_lists['prep fiducials']
        switchBd.effectStrs[d] = results.gatestring_lists['effect fiducials']
        switchBd.strs[d] = (results.gatestring_lists['prep fiducials'],
                            results.gatestring_lists['effect fiducials'])
        switchBd.germs[d] = results.gatestring_lists['germs']

        switchBd.gssFinal[d] = results.gatestring_structs['final']
        for iL,L in enumerate(swLs): #allow different results to have different Ls
            if L in results.gatestring_structs['final'].Ls:
                k = results.gatestring_structs['final'].Ls.index(L)
                switchBd.gss[d,iL] = results.gatestring_structs['iteration'][k]
        switchBd.gssAllL[d] = results.gatestring_structs['iteration']

        for i,lbl in enumerate(est_labels):
            est = results.estimates.get(lbl,None)
            if est is None: continue

            for suffix in ROBUST_SUFFIX_LIST:
                if combine_robust and lbl.endswith(suffix):
                    est_modvi = results.estimates.get(lbl[:-len(suffix)],est)
                    break
            else:
                est_modvi = est

            def rpt_objective(opt_objective):
                """ If optimized using just LGST, compute logl values """
                if opt_objective == "lgst": return "logl"
                else: return opt_objective

            switchBd.params[d,i] = est.parameters
            switchBd.objective[d,i] = rpt_objective(est.parameters['objective'])
            switchBd.objective_tvd_tuple[d,i] = (rpt_objective(est.parameters['objective']),'tvd')
            switchBd.objective_modvi[d,i] = rpt_objective(est_modvi.parameters['objective'])
            if est.parameters['objective'] == "logl":
                switchBd.mpc[d,i] = est.parameters['minProbClip']
                switchBd.mpc_modvi[d,i] = est_modvi.parameters['minProbClip']
            elif est.parameters['objective'] == "chi2":
                switchBd.mpc[d,i] = est.parameters['minProbClipForWeighting']
                switchBd.mpc_modvi[d,i] = est_modvi.parameters['minProbClipForWeighting']
            else: # "lgst" - just use defaults for logl
                switchBd.mpc[d,i] = 1e-4
                switchBd.mpc_modvi[d,i] = 1e-4
            switchBd.clifford_compilation[d,i] = est.parameters.get("clifford compilation",'auto')
            if switchBd.clifford_compilation[d,i] == 'auto':
                switchBd.clifford_compilation[d,i] = find_std_clifford_compilation(
                    est.gatesets['target'],printer)

            switchBd.profiler[d,i] = est_modvi.parameters.get('profiler',None)
            switchBd.meta_stdout[d,i] = est_modvi.meta.get('stdout',[('LOG',1,"No standard output recorded")])

            GIRepLbl = 'final iteration estimate' #replace with a gauge-opt label if it has a CI factory
            if confidenceLevel is not None:
                if _get_viewable_crf(est, lbl, GIRepLbl) is None:
                    for l in gauge_opt_labels:
                        if _get_viewable_crf(est, lbl, l) is not None:
                            GIRepLbl = l; break

            # NOTE on modvi_ds (the dataset used in model violation plots)
            # if combine_robust is True, modvi_ds is the unscaled dataset.
            # if combine_robust is False, modvi_ds is the effective dataset
            #     for the estimate (potentially just the unscaled one)

            NA = ws.NotApplicable()
            if est.parameters.get("weights",None): #if this estimate uses robust scaling
                effds, scale_subMxs = est.get_effective_dataset(True)
                switchBd.eff_ds[d,i] = effds
                switchBd.scaledSubMxsDict[d,i] = {'scaling': scale_subMxs, 'scaling.colormap': "revseq"}
                switchBd.modvi_ds[d,i] = results.dataset if combine_robust else effds
            else:
                switchBd.modvi_ds[d,i] = results.dataset
                switchBd.eff_ds[d,i] = NA
                switchBd.scaledSubMxsDict[d,i] = NA

            switchBd.gsTarget[d,i] = est.gatesets['target']
            switchBd.gsGIRep[d,i] = est.gatesets[GIRepLbl]
            switchBd.gsGIRepEP[d,i] = _tools.project_to_target_eigenspace(est.gatesets[GIRepLbl],
                                                                          est.gatesets['target'])
            switchBd.gsFinal[d,i,:] = [ est.gatesets.get(l,NA) for l in gauge_opt_labels ]
            switchBd.gsTargetAndFinal[d,i,:] = \
                        [ [est.gatesets['target'], est.gatesets[l]] if (l in est.gatesets) else NA
                          for l in gauge_opt_labels ]
            switchBd.goparams[d,i,:] = [ est.goparameters.get(l,NA) for l in gauge_opt_labels]

            for iL,L in enumerate(swLs): #allow different results to have different Ls
                if L in results.gatestring_structs['final'].Ls:
                    k = results.gatestring_structs['final'].Ls.index(L)
                    switchBd.gsL[d,i,iL] = est.gatesets['iteration estimates'][k]
                    switchBd.gsL_modvi[d,i,iL] = est_modvi.gatesets['iteration estimates'][k]
            switchBd.gsAllL[d,i] = est.gatesets['iteration estimates']
            switchBd.gsAllL_modvi[d,i] = est_modvi.gatesets['iteration estimates']

            if confidenceLevel is not None:
                misfit_sigma = est.misfit_sigma(use_accurate_Np=True)

                for il,l in enumerate(gauge_opt_labels):
                    if l in est.gatesets:
                        switchBd.cri[d,i,il] = None #default
                        crf = _get_viewable_crf(est, lbl, l, printer-2)

                        if crf is not None:
                            #Check whether we should use non-Markovian error bars:
                            # If fit is bad, check if any reduced fits were computed
                            # that we can use with in-model error bars.  If not, use
                            # experimental non-markovian error bars.
                            region_type = "normal" if misfit_sigma <= nmthreshold \
                                          else "non-markovian"
                            switchBd.cri[d,i,il] = crf.view(confidenceLevel, region_type)

                    else: switchBd.cri[d,i,il] = NA

                # "Gauge Invariant Representation" gateset
                # If we can't compute CIs for this, ignore SILENTLY, since any
                #  relevant warnings/notes should have been given above.
                switchBd.criGIRep[d,i] = None #default
                crf = _get_viewable_crf(est, lbl, GIRepLbl)
                if crf is not None:
                    region_type = "normal" if misfit_sigma <= nmthreshold \
                                  else "non-markovian"
                    switchBd.criGIRep[d,i] = crf.view(confidenceLevel, region_type)

    results_list = [results_dict[dslbl] for dslbl in dataset_labels]
    for i,gokey in enumerate(gauge_opt_labels):
        if multidataset:
            switchBd.gsFinalGrid[i] = [
                [ (res.estimates[el].gatesets.get(gokey,None)
                   if el in res.estimates else None) for el in est_labels ]
                for res in results_list ]
        else:
            switchBd.gsFinalGrid[i] = [
                (results_list[0].estimates[el].gatesets.get(gokey,None)
                   if el in results_list[0].estimates else None) for el in est_labels ]

    if multidataset:
        switchBd.add_unswitched('gsTargetGrid', [
            [ (res.estimates[el].gatesets.get('target',None)
               if el in res.estimates else None) for el in est_labels ]
            for res in results_list ])
    else:
        switchBd.add_unswitched('gsTargetGrid', [
            (results_list[0].estimates[el].gatesets.get('target',None)
             if el in results_list[0].estimates else None) for el in est_labels])

    return switchBd, dataset_labels, est_labels, gauge_opt_labels, Ls, swLs


def _create_single_metric_switchboard(ws, results_dict, bGaugeInv,
                                      dataset_labels, est_labels):
    gate_labels = None
    for results in results_dict.values():
        for est in results.estimates.values():
            if 'target' in est.gatesets:
                gate_labels = _add_new_labels(gate_labels,
                                              list(est.gatesets['target'].gates.keys()))

    if bGaugeInv:
        metric_abbrevs = ["evinf", "evagi","evnuinf","evnuagi","evdiamond",
                          "evnudiamond"]
    else:
        metric_abbrevs = ["inf","agi","trace","diamond","nuinf","nuagi",
                          "frob"]
    metric_names = [ _reportables.info_of_gatefn_by_name(abbrev)[0].replace('|',' ')
                     for abbrev in metric_abbrevs ]

    if len(dataset_labels) > 1: # multidataset
        metric_switchBd = ws.Switchboard(
            ["Metric", "Gate Label"], [metric_names, gate_labels],
            ["dropdown", "dropdown"], [0,0], show=[True,True] )
        metric_switchBd.add("gateLabel",(1,))
        metric_switchBd.add("metric",(0,))
        metric_switchBd.add("cmpTableTitle",(0,1))

        metric_switchBd.gateLabel[:] = gate_labels
        for i,gl in enumerate(gate_labels):
            metric_switchBd.cmpTableTitle[:,i] = ["%s %s" % (gl,nm) for nm in metric_names]

    else:
        metric_switchBd = ws.Switchboard(
            ["Metric"], [metric_names],
            ["dropdown"], [0], show=[True] )
        metric_switchBd.add("metric",(0,))
        metric_switchBd.add("cmpTableTitle",(0,))
        metric_switchBd.cmpTableTitle[:] = metric_names

    metric_switchBd.metric[:] = metric_abbrevs

    return metric_switchBd



def create_general_report(results, filename, title="auto",
                          confidenceLevel=None,
                          linlogPercentile=5, errgen_type="logGTi",
                          nmthreshold=50, precision=None,
                          comm=None, ws=None, auto_open=False,
                          cachefile=None, brief=False, connected=False,
                          link_to=None, resizable=True, autosize='initial',
                          verbosity=1):
    """ DEPRECATED: use pygsti.report.create_standard_report(...) """
    _warnings.warn(
            ('create_general_report(...) will be removed from pyGSTi.\n'
             '  This function only ever existed in beta versions and will\n'
             '  be removed completely soon.  Please update this call with:\n'
             '  pygsti.report.create_standard_report(...)\n'))


def create_standard_report(results, filename, title="auto",
                            confidenceLevel=None, comm=None, ws=None,
                            auto_open=False, link_to=None, brevity=0,
                            advancedOptions=None, verbosity=1):
    """
    Create a "standard" GST report, containing details about each estimate
    in `results` individually.

    Either a PDF or HTML report is generated, based on whether `filename` ends
    in ".pdf" or not.  In the richer HTML-mode, switches (drop-down boxes,
    buttons, etc.) allow the viewer to choose which estimate is displayed.  The
    estimates in multiple :class:`Results` objects can be viewed by providing
    a dictionary of `Results` objects as the `results` argument.  Note that
    when comparing many estimates it is often more convenient to view the report
    generated by :func:`create_comparison_report`, which is organized for this
    purpose.

    In PDF-mode this interactivity is not possible and so `results` may contain
    just a *single* estimate.  The chief advantage of this more limited mode
    is that is produces a highly-portable and self-contained PDF file.


    Parameters
    ----------
    results : Results
        An object which represents the set of results from one *or more* GST
        estimation runs, typically obtained from running
        :func:`do_long_sequence_gst` or :func:`do_stdpractice_gst`, OR a
        dictionary of such objects, representing multiple GST runs to be
        compared (typically all with *different* data sets). The keys of this
        dictionary are used to label different data sets that are selectable
        in the report.

    filename : string, optional
       The output filename where the report file(s) will be saved.  If
       None, then no output file is produced (but returned Workspace
       still caches all intermediate results).

    title : string, optional
       The title of the report.  "auto" causes a random title to be
       generated (which you may or may not like).

    confidenceLevel : int, optional
       If not None, then the confidence level (between 0 and 100) used in
       the computation of confidence regions/intervals. If None, no
       confidence regions or intervals are computed.

    comm : mpi4py.MPI.Comm, optional
        When not None, an MPI communicator for distributing the computation
        across multiple processors.

    ws : Workspace, optional
        The workspace used as a scratch space for performing the calculations
        and visualizations required for this report.  If you're creating
        multiple reports with similar tables, plots, etc., it may boost
        performance to use a single Workspace for all the report generation.

    auto_open : bool, optional
        If True, automatically open the report in a web browser after it
        has been generated.

    link_to : list, optional
        If not None, a list of one or more items from the set
        {"tex", "pdf", "pkl"} indicating whether or not to
        create and include links to Latex, PDF, and Python pickle
        files, respectively.  "tex" creates latex source files for
        tables; "pdf" renders PDFs of tables and plots ; "pkl" creates
        Python versions of plots (pickled python data) and tables (pickled
        pandas DataFrams).

    brevity : int, optional
        Amount of detail to include in the report.  Larger values mean smaller
        "more briefr" reports, which reduce generation time, load time, and
        disk space consumption.  In particular:

        - 1: Plots showing per-sequences quantities disappear at brevity=1
        - 2: Reference sections disappear at brevity=2
        - 3: Germ-level estimate tables disappear at brevity=3
        - 4: Everything but summary figures disappears at brevity=4

    advancedOptions : dict, optional
        A dictionary of advanced options for which the default values aer usually
        are fine.  Here are the possible keys of `advancedOptions`:

        - connected : bool, optional
            Whether output HTML should assume an active internet connection.  If
            True, then the resulting HTML file size will be reduced because it
            will link to web resources (e.g. CDN libraries) instead of embedding
            them.

        - cachefile : str, optional
            filename with cached workspace results

        - linlogPercentile : float, optional
            Specifies the colorscale transition point for any logL or chi2 color
            box plots.  The lower `(100 - linlogPercentile)` percentile of the
            expected chi2 distribution is shown in a linear grayscale, and the
            top `linlogPercentile` is shown on a logarithmic colored scale.

        - errgen_type: {"logG-logT", "logTiG", "logGTi"}
            The type of error generator to compute.  Allowed values are:

            - "logG-logT" : errgen = log(gate) - log(target_gate)
            - "logTiG" : errgen = log( dot(inv(target_gate), gate) )
            - "logGTi" : errgen = log( dot(gate, inv(target_gate)) )

        - nmthreshold : float, optional
            The threshold, in units of standard deviations, that triggers the
            usage of non-Markovian error bars.  If None, then non-Markovian
            error bars are never computed.

        - precision : int or dict, optional
            The amount of precision to display.  A dictionary with keys
            "polar", "sci", and "normal" can separately specify the
            precision for complex angles, numbers in scientific notation, and
            everything else, respectively.  If an integer is given, it this
            same value is taken for all precision types.  If None, then
            `{'normal': 6, 'polar': 3, 'sci': 0}` is used.

        - resizable : bool, optional
            Whether plots and tables are made with resize handles and can be
            resized within the report.

        - autosize : {'none', 'initial', 'continual'}
            Whether tables and plots should be resized, either initially --
            i.e. just upon first rendering (`"initial"`) -- or whenever
            the browser window is resized (`"continual"`).

        - combine_robust : bool, optional
            Whether robust estimates should automatically be combined with
            their non-robust counterpart when displayed in reports. (default
            is True).

        - confidence_interval_brevity : int, optional
            Roughly specifies how many figures will have confidence intervals
            (when applicable). Defaults to '1'.  Smaller values mean more
            tables will get confidence intervals (and reports will take longer
            to generate).


    verbosity : int, optional
       How much detail to send to stdout.


    Returns
    -------
    Workspace
        The workspace object used to create the report
    """
    tStart = _time.time()
    printer = _VerbosityPrinter.build_printer(verbosity, comm=comm)

    if advancedOptions is None: advancedOptions = {}
    linlogPercentile = advancedOptions.get('linlog percentile',5)
    errgen_type = advancedOptions.get('error generator type', "logGTi")
    nmthreshold = advancedOptions.get('nm threshold',DEFAULT_BAD_FIT_THRESHOLD)
    precision = advancedOptions.get('precision', None)
    cachefile = advancedOptions.get('cachefile',None)
    connected = advancedOptions.get('connected',False)
    resizable = advancedOptions.get('resizable',True)
    autosize = advancedOptions.get('autosize','initial')
    combine_robust = advancedOptions.get('combine_robust',True)
    ci_brevity = advancedOptions.get('confidence_interval_brevity',1)

    if filename and filename.endswith(".pdf"):
        fmt = "latex"
    else:
        fmt = "html"
        
    printer.log('*** Creating workspace ***')
    if ws is None: ws = _ws.Workspace(cachefile)

    if isinstance(title,int): #to catch backward compatibility issues
        raise ValueError(("'title' argument must be a string.  You may be accidentally"
                          " specifying an int here because in older versions of pyGSTi"
                          " the third argument to create_general_report was the"
                          " confidence interval - please note the updated function signature"))

    if title is None or title == "auto":
        autoname = _autotitle.generate_name()
        title = "GST Report for " + autoname
        _warnings.warn( ("You should really specify `title=` when generating reports,"
                         "as this makes it much easier to identify them later on.  "
                         "Since you didn't, pyGSTi has generated a random one"
                         " for you: '{}'.").format(autoname))

    results_dict = results if isinstance(results, dict) else {"unique": results}
    toggles = _set_toggles(results_dict, brevity, combine_robust)

    #DEBUG
    renderMath = True
    #_ws.WorkspaceOutput.default_render_options['click_to_display'] = True #don't render any plots until they're clicked
    #_ws.WorkspaceOutput.default_render_options['render_math'] = renderMath #don't render any math

    qtys = {} # stores strings to be inserted into report template
    def addqty(b, name, fn, *args, **kwargs):
        """Adds an item to the qtys dict within a timed block"""
        if b is None or brevity < b:
            with _timed_block(name, formatStr='{:45}', printer=printer, verbosity=2):
                qtys[name] = fn(*args, **kwargs)

    qtys['title'] = title
    qtys['date'] = _time.strftime("%B %d, %Y")
    qtys['confidenceLevel'] = "%d" % \
        confidenceLevel if confidenceLevel is not None else "NOT-SET"
    qtys['linlg_pcntle'] = "%d" % round(linlogPercentile) #to nearest %
    qtys['linlg_pcntle_inv'] = "%d" % (100 - int(round(linlogPercentile)))
    qtys['errorgenformula'], qtys['errorgendescription'] = _errgen_formula(errgen_type, fmt)

    pdfInfo = [('Author','pyGSTi'), ('Title', title),
               ('Keywords', 'GST'), ('pyGSTi Version',_version.__version__)]
    qtys['pdfinfo'] = _merge.to_pdfinfo(pdfInfo)

    # Generate Switchboard
    printer.log("*** Generating switchboard ***")

    #Create master switchboard
    switchBd, dataset_labels, est_labels, gauge_opt_labels, Ls, swLs = \
            _create_master_switchboard(ws, results_dict, confidenceLevel,
                                       nmthreshold, printer, fmt,
                                       combine_robust)
    if fmt == "latex" and (len(dataset_labels) > 1 or len(est_labels) > 1 or
                         len(gauge_opt_labels) > 1 or len(swLs) > 1):
        raise ValueError("PDF reports can only show a *single* dataset," +
                         " estimate, and gauge optimization.")

    if len(Ls) > 0 and Ls[0] == 0:
        _warnings.warn(("Setting the first 'max-length' to zero, e.g. using"
                        " [0,1,2,4] instead of [1,2,4], is deprecated and"
                        " may cause 'no data to plot' errors when creating"
                        " this report.  Please remove this leading zero."))

    # Generate Tables
    printer.log("*** Generating tables ***")

    if confidenceLevel is not None:
        #TODO: make plain text fields which update based on switchboards?
        for some_cri in switchBd.cri.flat: #can have only some confidence regions
            if some_cri is not None and not isinstance(some_cri, _ws.NotApplicable):
                qtys['confidenceIntervalScaleFctr'] = "%.3g" % some_cri.intervalScaling
                qtys['confidenceIntervalNumNonGaugeParams'] = "%d" % some_cri.nNonGaugeParams

    multidataset = bool(len(dataset_labels) > 1)
    #multiest = bool(len(est_labels) > 1)
    #multiGO = bool(len(gauge_opt_labels) > 1)
    multiL = bool(len(swLs) > 1)

    ##goView = [multidataset,multiest,multiGO,False]
    ##maxLView = [multidataset,multiest,False,multiL]
    #goView = [False,False,multiGO,False]
    maxLView = [False,False,False,multiL]

    if fmt == "html":
        qtys['topSwitchboard'] = switchBd
        #qtys['goSwitchboard1'] = switchBd.view(goView,"v1")
        #qtys['goSwitchboard2'] = switchBd.view(goView,"v2")
        qtys['maxLSwitchboard1'] = switchBd.view(maxLView,"v6")

    gsTgt = switchBd.gsTarget
    ds = switchBd.ds
    eff_ds = switchBd.eff_ds
    modvi_ds = switchBd.modvi_ds
    prepStrs = switchBd.prepStrs
    effectStrs = switchBd.effectStrs
    germs = switchBd.germs
    strs = switchBd.strs
    cliffcomp = switchBd.clifford_compilation
    A = None # no brevity restriction: always display; for "Summary"- & "Help"-tab figs

    #Brevity key:
    # 1: Plots involving quantities for individual sequences disappear at brevity=1
    # 2: Input & Meta reference tables disappear at brevity=2
    # 3: Germ estimate tables disappear at brevity=3
    # 4: Everything but summary figs disappear at brevity=4

    addqty(2,'targetSpamBriefTable', ws.SpamTable, gsTgt, None, display_as='boxes', includeHSVec=False)
    addqty(2,'targetGatesBoxTable', ws.GatesTable, gsTgt, display_as="boxes")
    addqty(2,'datasetOverviewTable', ws.DataSetOverviewTable, ds)

    gsFinal = switchBd.gsFinal
    gsGIRep = switchBd.gsGIRep
    gsEP = switchBd.gsGIRepEP
    cri_base = switchBd.cri if (confidenceLevel is not None) else None
    criGIRep_base = switchBd.criGIRep if (confidenceLevel is not None) else None
    cri      = lambda l: cri_base if ci_brevity <= l else None
    criGIRep = lambda l: criGIRep_base if ci_brevity <= l else None

    # Non-summary gate estimates
    # Germ
    addqty(4,'bestGatesetSpamParametersTable', ws.SpamParametersTable, switchBd.gsTargetAndFinal,
           ['Target','Estimated'], cri(1) )
    addqty(4,'bestGatesetSpamBriefTable', ws.SpamTable, switchBd.gsTargetAndFinal,
           ['Target','Estimated'], 'boxes', cri(1), includeHSVec=False)
    addqty(4,'bestGatesetSpamVsTargetTable', ws.SpamVsTargetTable, gsFinal, gsTgt, cri(1))
    addqty(A,'bestGatesetGaugeOptParamsTable', ws.GaugeOptParamsTable, switchBd.goparams)
    addqty(4,'bestGatesetGatesBoxTable', ws.GatesTable, switchBd.gsTargetAndFinal,
                                                     ['Target','Estimated'], "boxes", cri(1))
    addqty(4,'bestGatesetChoiEvalTable', ws.ChoiTable, gsFinal, None, cri(1), display=("boxplot","barplot"))
    addqty(4,'bestGatesetDecompTable', ws.GateDecompTable, gsFinal, gsTgt, cri(0))
    addqty(4,'bestGatesetEvalTable', ws.GateEigenvalueTable, gsGIRep, gsTgt, criGIRep(1),
           display=('evals','target','absdiff-evals','infdiff-evals','log-evals','absdiff-log-evals'))
    addqty(3,'bestGermsEvalTable', ws.GateEigenvalueTable, gsGIRep, gsEP, criGIRep(1),
           display=('evals','target','absdiff-evals','infdiff-evals','log-evals','absdiff-log-evals'),
           virtual_gates=germs)
    #addqty('bestGatesetRelEvalTable', ws.GateEigenvalueTable, gsFinal, gsTgt, cri(1), display=('rel','log-rel'))
    addqty(4,'bestGatesetVsTargetTable', ws.GatesetVsTargetTable, gsFinal, gsTgt, cliffcomp, cri(1))
    addqty(4,'bestGatesVsTargetTable_gv', ws.GatesVsTargetTable, gsFinal, gsTgt, cri(1),
                                        display=('inf','agi','trace','diamond','nuinf','nuagi'))
    addqty(3,'bestGatesVsTargetTable_gvgerms', ws.GatesVsTargetTable, gsFinal, gsTgt, cri(0),
                                        display=('inf','trace','nuinf'), virtual_gates=germs)
    addqty(4,'bestGatesVsTargetTable_gi', ws.GatesVsTargetTable, gsGIRep, gsTgt, criGIRep(1),
                                        display=('evinf','evagi','evnuinf','evnuagi','evdiamond','evnudiamond'))
    addqty(3,'bestGatesVsTargetTable_gigerms', ws.GatesVsTargetTable, gsGIRep, gsEP, criGIRep(0),
                                        display=('evdiamond','evnudiamond'), virtual_gates=germs)
    addqty(A,'bestGatesVsTargetTable_sum', ws.GatesVsTargetTable, gsFinal, gsTgt, cri(1),
                                         display=('inf','trace','diamond','evinf','evdiamond'))

    addqty(4,'bestGatesetErrGenBoxTable', ws.ErrgenTable, gsFinal, gsTgt, cri(1), ("errgen","H","S","A"),
                                                           "boxes", errgen_type)
    addqty(2,'metadataTable', ws.MetadataTable, gsFinal, switchBd.params)
    addqty(2,'stdoutBlock', ws.StdoutText, switchBd.meta_stdout)
    addqty(2,'profilerTable', ws.ProfilerTable, switchBd.profiler)
    addqty(2,'softwareEnvTable', ws.SoftwareEnvTable)
    addqty(A,'exampleTable', ws.ExampleTable)
    qtys['exampleTable'].set_render_options(click_to_display=True)

      # single-metric comparison tables
    gvmetric_switchBd = _create_single_metric_switchboard(ws, results_dict, False,
                                                        dataset_labels, est_labels)
    gimetric_switchBd = _create_single_metric_switchboard(ws, results_dict, True,
                                                        dataset_labels, est_labels)
    qtys['metricSwitchboard_gv'] = gvmetric_switchBd
    qtys['metricSwitchboard_gi'] = gimetric_switchBd
    if multidataset:
        addqty(4,'singleMetricTable_gv', ws.GatesSingleMetricTable, gvmetric_switchBd.metric,
               switchBd.gsFinalGrid, switchBd.gsTargetGrid, est_labels, dataset_labels,
               gvmetric_switchBd.cmpTableTitle, gvmetric_switchBd.gateLabel, confidenceRegionInfo=None)
        addqty(4,'singleMetricTable_gi', ws.GatesSingleMetricTable, gimetric_switchBd.metric,
               switchBd.gsFinalGrid, switchBd.gsTargetGrid, est_labels, dataset_labels,
               gimetric_switchBd.cmpTableTitle, gimetric_switchBd.gateLabel, confidenceRegionInfo=None)

    else:
        addqty(4,'singleMetricTable_gv', ws.GatesSingleMetricTable, gvmetric_switchBd.metric,
               switchBd.gsFinalGrid, switchBd.gsTargetGrid, est_labels, None,
               gvmetric_switchBd.cmpTableTitle, confidenceRegionInfo=None)
        addqty(4,'singleMetricTable_gi', ws.GatesSingleMetricTable, gimetric_switchBd.metric,
               switchBd.gsFinalGrid, switchBd.gsTargetGrid, est_labels, None,
               gimetric_switchBd.cmpTableTitle, confidenceRegionInfo=None)

    #Ls and Germs specific
    gss = switchBd.gss
    gsL = switchBd.gsL
    gsL_modvi = switchBd.gsL_modvi
    gssAllL = switchBd.gssAllL
    addqty(2,'fiducialListTable', ws.GatestringTable, strs,["Prep.","Measure"], commonTitle="Fiducials")
    addqty(2,'prepStrListTable', ws.GatestringTable, prepStrs,"Preparation Fiducials")
    addqty(2,'effectStrListTable', ws.GatestringTable, effectStrs,"Measurement Fiducials")
    addqty(1,'colorBoxPlotKeyPlot', ws.BoxKeyPlot, prepStrs, effectStrs)
    addqty(2,'germList2ColTable', ws.GatestringTable, germs, "Germ", nCols=2)
    addqty(4,'progressTable', ws.FitComparisonTable,
           Ls, gssAllL, switchBd.gsAllL_modvi, ds, switchBd.objective_modvi, 'L')

    # Generate plots
    printer.log("*** Generating plots ***")

    addqty(4,'gramBarPlot', ws.GramMatrixBarPlot, ds,gsTgt,10,strs)


    addqty(4,'progressBarPlot', ws.FitComparisonBarPlot,
           Ls, gssAllL, switchBd.gsAllL_modvi, modvi_ds, switchBd.objective_modvi, 'L')
    addqty(A,'progressBarPlot_sum', ws.FitComparisonBarPlot,
           Ls, gssAllL, switchBd.gsAllL_modvi, modvi_ds, switchBd.objective_modvi, 'L') #just duplicate for now

    # Don't display "Target" in model violation summary, as it's often
    # huge and messes up the plot scale.
    est_inds_mt = [ i for i,l in enumerate(est_labels) if l != "Target" ]
    est_lbls_mt = [ est_labels[i] for i in est_inds_mt ] # "minus target"
    Nd = len(dataset_labels)
    Ne = len(est_inds_mt)
    grid_objective = switchBd.objective_modvi[0,0] #just take first one for now
    def na_to_none(x):
        return None if isinstance(x, _ws.NotApplicable) else x

    if multidataset:
        dsGrid = [ [ na_to_none(switchBd.modvi_ds[d,i]) for i in est_inds_mt]
                   for d in range(Nd)]
        gssGrid = [ [na_to_none(switchBd.gssFinal[i])]*Ne for i in range(Nd) ]
        gsGrid = [ [ na_to_none(switchBd.gsL_modvi[d,i,-1]) for i in est_inds_mt]
                   for d in range(Nd)]
        addqty(A,'finalFitComparePlot', ws.FitComparisonBoxPlot,
               est_lbls_mt, dataset_labels,
               gssGrid, gsGrid, dsGrid, grid_objective)
    else:
        dsGrid = [ na_to_none(switchBd.modvi_ds[0,i]) for i in est_inds_mt ]
        gssGrid =[ na_to_none(switchBd.gssFinal[0])]*Ne
        gsGrid = [ na_to_none(switchBd.gsL_modvi[0,i,-1]) for i in est_inds_mt]
        addqty(A,'finalFitComparePlot', ws.FitComparisonBarPlot,
               est_lbls_mt, gssGrid, gsGrid, dsGrid, grid_objective, 'Estimate')

    addqty(1,'bestEstimateColorBoxPlot', ws.ColorBoxPlot,
           switchBd.objective, gss, modvi_ds, gsL_modvi,
           linlg_pcntle=float(linlogPercentile) / 100,
           minProbClipForWeighting=switchBd.mpc_modvi)
    if brevity < 1: qtys['bestEstimateColorBoxPlot'].set_render_options(
            click_to_display=False, valign='bottom')

    addqty(1,'bestEstimateTVDColorBoxPlot', ws.ColorBoxPlot,
           'tvd', gss, modvi_ds, gsL_modvi)
    if brevity < 1: qtys['bestEstimateTVDColorBoxPlot'].set_render_options(
            click_to_display=False, valign='bottom')

    addqty(1,'bestEstimateColorScatterPlot', ws.ColorBoxPlot,
        switchBd.objective, gss, modvi_ds, gsL_modvi,
        linlg_pcntle=float(linlogPercentile) / 100,
        minProbClipForWeighting=switchBd.mpc_modvi, typ="scatter") #TODO: L-switchboard on modvi overview page?
    ##qtys['bestEstimateColorScatterPlot'].set_render_options(click_to_display=True)
    ##  Fast enough now thanks to scattergl, but webgl render issues so need to delay creation

    addqty(A,'bestEstimateColorHistogram', ws.ColorBoxPlot,
        switchBd.objective, gss, modvi_ds, gsL_modvi,
        linlg_pcntle=float(linlogPercentile) / 100,
        minProbClipForWeighting=switchBd.mpc_modvi, typ="histogram") #TODO: L-switchboard on summary page?


    if combine_robust:
        # model-violation (using _modvi variables) plots show pre-scaling
        # violation, so we create# additional _scl plots to separately show
        # post-scaling violation (using eff_ds and non-_modvi variables).
        # Note that 'eff_ds' is NA for estimates that have no scaling, so that
        # duplicate plots (for estiamtes without scaling) are avoided.

        addqty(4,'progressTable_scl', ws.FitComparisonTable,
               Ls, gssAllL, switchBd.gsAllL, eff_ds, switchBd.objective, 'L')

        addqty(4,'progressBarPlot_scl', ws.FitComparisonBarPlot,
               Ls, gssAllL, switchBd.gsAllL, eff_ds, switchBd.objective, 'L') # robust-scaled version

        #Not pagniated currently... just set to same full plot
        addqty(1,'bestEstimateColorBoxPlot_scl', ws.ColorBoxPlot,
            switchBd.objective, gss, eff_ds, gsL,
            linlg_pcntle=float(linlogPercentile) / 100,
            minProbClipForWeighting=switchBd.mpc)
        if brevity < 1: qtys['bestEstimateColorBoxPlot_scl'].set_render_options(
                click_to_display=False, valign='bottom')

        addqty(1,'bestEstimateColorScatterPlot_scl', ws.ColorBoxPlot,
               switchBd.objective, gss, eff_ds, gsL,
               linlg_pcntle=float(linlogPercentile) / 100,
               minProbClipForWeighting=switchBd.mpc, typ="scatter")

        addqty(A,'bestEstimateColorHistogram_scl', ws.ColorBoxPlot,
               switchBd.objective, gss, eff_ds, gsL,
               linlg_pcntle=float(linlogPercentile) / 100,
               minProbClipForWeighting=switchBd.mpc, typ="histogram")


    #Note: this is the only plot that uses eff_ds (and is on robust-scaling
    #  page) that is created when combine_robust == False
    addqty(1,'dataScalingColorBoxPlot', ws.ColorBoxPlot,
           "scaling", switchBd.gssFinal, eff_ds, None,
           submatrices=switchBd.scaledSubMxsDict)


    if multidataset:
        #check if data sets are comparable (if they have the same sequences)
        comparable = True
        gstrCmpList = list(results_dict[ dataset_labels[0] ].dataset.keys()) #maybe use gatestring_lists['final']??
        for dslbl in dataset_labels:
            if list(results_dict[dslbl].dataset.keys()) != gstrCmpList:
                _warnings.warn("Not all data sets are comparable - no comparisions will be made.")
                comparable=False; break

        if comparable:
            #initialize a new "dataset comparison switchboard"
            dscmp_switchBd = ws.Switchboard(
                ["Dataset1","Dataset2"],
                [dataset_labels, dataset_labels],
                ["buttons","buttons"], [0,1]
            )
            dscmp_switchBd.add("dscmp",(0,1))
            dscmp_switchBd.add("dscmp_gss",(0,))
            dscmp_switchBd.add("refds",(0,))

            for d1, dslbl1 in enumerate(dataset_labels):
                dscmp_switchBd.dscmp_gss[d1] = results_dict[dslbl1].gatestring_structs['final']
                dscmp_switchBd.refds[d1] = results_dict[dslbl1].dataset #only used for #of spam labels below

            dsComp = dict()
            all_dsComps = dict()
            indices = []
            for i in range(len(dataset_labels)):
                for j in range(len(dataset_labels)):
                    indices.append((i, j))
            if comm is not None:
                _, indexDict, _ = _distribute_indices(indices, comm)
                rank = comm.Get_rank()
                for k, v in indexDict.items():
                    if v == rank:
                        d1, d2 = k
                        dslbl1 = dataset_labels[d1]
                        dslbl2 = dataset_labels[d2]

                        ds1 = results_dict[dslbl1].dataset
                        ds2 = results_dict[dslbl2].dataset
                        dsComp[(d1, d2)] = _DataComparator(
                            [ds1, ds2], DS_names=[dslbl1, dslbl2])
                dicts = comm.gather(dsComp, root=0)
                if rank == 0:
                    for d in dicts:
                        for k, v in d.items():
                            d1, d2 = k
                            dscmp_switchBd.dscmp[d1, d2] = v
                            all_dsComps[(d1,d2)] = v
            else:
                for d1, d2 in indices:
                    dslbl1 = dataset_labels[d1]
                    dslbl2 = dataset_labels[d2]
                    ds1 = results_dict[dslbl1].dataset
                    ds2 = results_dict[dslbl2].dataset
                    all_dsComps[(d1,d2)] =  _DataComparator([ds1, ds2], DS_names=[dslbl1,dslbl2])
                    dscmp_switchBd.dscmp[d1, d2] = all_dsComps[(d1,d2)]

            qtys['dscmpSwitchboard'] = dscmp_switchBd
            addqty(4,'dsComparisonSummary', ws.DatasetComparisonSummaryPlot, dataset_labels, all_dsComps)
            #addqty('dsComparisonHistogram', ws.DatasetComparisonHistogramPlot, dscmp_switchBd.dscmp, display='pvalue')
            addqty(4,'dsComparisonHistogram', ws.ColorBoxPlot,
                   'dscmp', dscmp_switchBd.dscmp_gss, dscmp_switchBd.refds, None,
                   dscomparator=dscmp_switchBd.dscmp, typ="histogram")
            addqty(1,'dsComparisonBoxPlot', ws.ColorBoxPlot, 'dscmp', dscmp_switchBd.dscmp_gss,
                   dscmp_switchBd.refds, None, dscomparator=dscmp_switchBd.dscmp)
            toggles['CompareDatasets'] = True
        else:
            toggles['CompareDatasets'] = False # not comparable!
    else:
        toggles['CompareDatasets'] = False


    if filename is not None:
        if comm is None or comm.Get_rank() == 0:
            # 3) populate template file => report file
            printer.log("*** Merging into template file ***")

            if fmt == "html":
                templateDir = "standard_html_report"
                _merge.merge_html_template_dir(
                    qtys, templateDir, filename, auto_open, precision, link_to,
                    connected=connected, toggles=toggles, renderMath=renderMath,
                    resizable=resizable, autosize=autosize, verbosity=printer)

            elif fmt == "latex":
                templateFile = "standard_pdf_report.tex"
                base = _os.path.splitext(filename)[0] # no extension
                _merge.merge_latex_template(qtys, templateFile, base+".tex", toggles,
                                            precision, printer)

                # compile report latex file into PDF
                cmd = _ws.WorkspaceOutput.default_render_options.get('latex_cmd',None)
                flags = _ws.WorkspaceOutput.default_render_options.get('latex_flags',[])
                assert(cmd), "Cannot render PDF documents: no `latex_cmd` render option."
                printer.log("Latex file(s) successfully generated.  Attempting to compile with %s..." % cmd)
                _merge.compile_latex_report(base, [cmd] + flags, printer, auto_open)
            else:
                raise ValueError("Unrecognized format: %s" % fmt)

            #SmartCache.global_status(printer)
    else:
        printer.log("*** NOT Merging into template file (filename is None) ***")
    printer.log("*** Report Generation Complete!  Total time %gs ***" % (_time.time()-tStart))

    return ws


def create_report_notebook(results, filename, title="auto",
                           confidenceLevel=None,
                           auto_open=False, connected=False, verbosity=0):
    """
    Create a "report notebook": a Jupyter ipython notebook file which, when its
    cells are executed, will generate similar figures to those contained in an
    html report (via :func:`create_standard_report`).

    A notebook report allows the user to interact more flexibly with the data
    underlying the figures, and to easily generate customized variants on the
    figures.  As such, this type of report will be most useful for experts
    who want to tinker with the standard analysis presented in the static
    HTML or LaTeX format reports.

    Parameters
    ----------
    results : Results
        An object which represents the set of results from one *or more* GST
        estimation runs, typically obtained from running
        :func:`do_long_sequence_gst` or :func:`do_stdpractice_gst`, OR a
        dictionary of such objects, representing multiple GST runs to be
        compared (typically all with *different* data sets). The keys of this
        dictionary are used to label different data sets that are selectable
        (via setting Python variables) in the report.

    filename : string, optional
       The output filename where the report file(s) will be saved.  Must end
       in ".ipynb".

    title : string, optional
       The title of the report.  "auto" causes a random title to be
       generated (which you may or may not like).

    confidenceLevel : int, optional
       If not None, then the confidence level (between 0 and 100) used in
       the computation of confidence regions/intervals. If None, no
       confidence regions or intervals are computed.

    auto_open : bool, optional
        If True, automatically open the report in a web browser after it
        has been generated.

    connected : bool, optional
        Whether output notebook should assume an active internet connection.  If
        True, then the resulting file size will be reduced because it will link
        to web resources (e.g. CDN libraries) instead of embedding them.

    verbosity : int, optional
       How much detail to send to stdout.


    Returns
    -------
    None
    """
    printer = _VerbosityPrinter.build_printer(verbosity)
    templatePath = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                                 "templates","report_notebook")
    assert(_os.path.splitext(filename)[1] == '.ipynb'), 'Output file extension must be .ipynb'
    outputDir = _os.path.dirname(filename)

    #Copy offline directory into position
    if not connected:
        _merge.rsync_offline_dir(outputDir)

    #Save results to file
    basename = _os.path.splitext(_os.path.basename(filename))[0]
    results_file_base = basename + '_results.pkl'
    results_file = _os.path.join(outputDir, results_file_base)
    with open(results_file,'wb') as f:
        _pickle.dump(results, f)

    if title is None or title == "auto":
        autoname = _autotitle.generate_name()
        title = "GST Report for " + autoname
        _warnings.warn( ("You should really specify `title=` when generating reports,"
                         "as this makes it much easier to identify them later on.  "
                         "Since you didn't, pyGSTi will has generated a random one"
                         " for you: '{}'.").format(autoname))

    nb = _Notebook()
    nb.add_markdown('# {title}\n(Created on {date})'.format(
        title=title, date=_time.strftime("%B %d, %Y")))

    nb.add_code("""\
        from __future__ import print_function
        import pickle
        import pygsti""")

    if isinstance(results, dict):
        dsKeys = list(results.keys())
        results = results[dsKeys[0]]
          #Note: `results` is always a single Results obj from here down

        nb.add_code("""\
        #Load results dictionary
        with open('{infile}', 'rb') as infile:
            results_dict = pickle.load(infile)
        print("Available dataset keys: ", ', '.join(results_dict.keys()))\
        """.format(infile = results_file_base))

        nb.add_code("""\
        #Set which dataset should be used below
        results = results_dict['{dsKey}']
        print("Available estimates: ", ', '.join(results.estimates.keys()))\
        """.format(dsKey=dsKeys[0]))

    else:
        dsKeys = []
        nb.add_code("""\
        #Load results
        with open('{infile}', 'rb') as infile:
            results = pickle.load(infile)
        print("Available estimates: ", ', '.join(results.estimates.keys()))\
        """.format(infile = results_file_base))

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

        gssFinal  = results.gatestring_structs['final']
        Ls        = results.gatestring_structs['final'].Ls
        gssPerIter = results.gatestring_structs['iteration'] #ALL_L

        prepStrs = results.gatestring_lists['prep fiducials']
        effectStrs = results.gatestring_lists['effect fiducials']
        germs = results.gatestring_lists['germs']
        strs = (prepStrs, effectStrs)

        params = estimate.parameters
        objective = estimate.parameters['objective']
        if objective == "logl":
            mpc = estimate.parameters['minProbClip']
        else:
            mpc = estimate.parameters['minProbClipForWeighting']
        clifford_compilation = estimate.parameters.get('clifford_compilation',None)

        effective_ds, scale_subMxs = estimate.get_effective_dataset(True)
        scaledSubMxsDict = {{'scaling': scale_subMxs, 'scaling.colormap': "revseq"}}

        gatesets   = estimate.gatesets
        gs         = gatesets[gopt] #FINAL
        gs_final   = gatesets['final iteration estimate'] #ITER
        gs_target  = gatesets['target']
        gsPerIter  = gatesets['iteration estimates']

        gs_eigenspace_projected = \
            pygsti.tools.project_to_target_eigenspace(gs, gs_target)

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
        _os.path.join(templatePath,'summary.txt'),
        _os.path.join(templatePath,'goodness.txt'),
        _os.path.join(templatePath,'gauge_invariant.txt'),
        _os.path.join(templatePath,'gauge_variant.txt')])

    #Insert multi-dataset specific analysis
    if len(dsKeys) > 1:
        nb.add_markdown( ('# Dataset comparisons\n'
                          'This report contains information for more than one data set.'
                          'This page shows comparisons between different data sets.') )

        nb.add_code("""\
        dslbl1 = '{dsLbl1}'
        dslbl2 = '{dsLbl2}'
        dscmp_gss = results_dict[dslbl1].gatestring_structs['final']
        ds1 = results_dict[dslbl1].dataset
        ds2 = results_dict[dslbl2].dataset
        dscmp = pygsti.obj.DataComparator([ds1, ds2], DS_names=[dslbl1, dslbl2])
        """.format(dsLbl1=dsKeys[0], dsLbl2=dsKeys[1]))
        nb.add_notebook_text_files([
            _os.path.join(templatePath,'data_comparison.txt')])

    #Add reference material
    nb.add_notebook_text_files([
        _os.path.join(templatePath,'input.txt'),
        _os.path.join(templatePath,'meta.txt')])

    printer.log("Report Notebook created as %s" % filename)

    if auto_open:
        port = "auto" if auto_open == True else int(auto_open)
        nb.launch(filename, port=port)
    else:
        nb.save_to(filename)


def find_std_clifford_compilation(gateset, verbosity=0):
    """
    Returns the standard Clifford compilation for `gateset`, if
    one exists.  Otherwise returns None.

    Parameters
    ----------
    gateset : GateSet
        The ideal (target) gate set of primitive gates.

    verbosity : int, optional
        How much detail to send to stdout.

    Returns
    -------
    dict or None
        The Clifford compilation dictionary (if one can be found).
    """
    printer = _VerbosityPrinter.build_printer(verbosity)
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
        if set(mod.gs_target.gates.keys()) == set(gateset.gates.keys()) and \
           set(mod.gs_target.preps.keys()) == set(gateset.preps.keys()) and \
           set(mod.gs_target.povms.keys()) == set(gateset.povms.keys()):
            if mod.gs_target.frobeniusdist(gateset) < 1e-6:
                if hasattr(mod,"clifford_compilation"):
                    printer.log("Found standard clifford compilation from %s" % module_name)
                    return mod.clifford_compilation
    return None

##Scratch: SAVE!!! this code generates "projected" gatesets which can be sent to
## FitComparisonTable (with the same gss for each) to make a nice comparison plot.
#        gateLabels = list(gateset.gates.keys())  # gate labels
#        basis = gateset.basis
#
#        if basis.name != targetGateset.basis.name:
#            raise ValueError("Basis mismatch between gateset (%s) and target (%s)!"\
#                                 % (basis.name, targetGateset.basis.name))
#
#        #Do computation first
#        # Note: set to "full" parameterization so we can set the gates below
#        #  regardless of what to fo parameterization the original gateset had.
#        gsH = gateset.copy(); gsH.set_all_parameterizations("full"); Np_H = 0
#        gsS = gateset.copy(); gsS.set_all_parameterizations("full"); Np_S = 0
#        gsHS = gateset.copy(); gsHS.set_all_parameterizations("full"); Np_HS = 0
#        gsLND = gateset.copy(); gsLND.set_all_parameterizations("full"); Np_LND = 0
#        #gsHSCP = gateset.copy()
#        gsLNDCP = gateset.copy(); gsLNDCP.set_all_parameterizations("full")
#        for gl in gateLabels:
#            gate = gateset.gates[gl]
#            targetGate = targetGateset.gates[gl]
#
#            errgen = _tools.error_generator(gate, targetGate, genType)
#            hamProj, hamGens = _tools.std_errgen_projections(
#                errgen, "hamiltonian", basis.name, basis, True)
#            stoProj, stoGens = _tools.std_errgen_projections(
#                errgen, "stochastic", basis.name, basis, True)
#            HProj, OProj, HGens, OGens = \
#                _tools.lindblad_errgen_projections(
#                    errgen, basis, basis, basis, normalize=False,
#                    return_generators=True)
#                #Note: return values *can* be None if an empty/None basis is given
#
#            ham_error_gen = _np.einsum('i,ijk', hamProj, hamGens)
#            sto_error_gen = _np.einsum('i,ijk', stoProj, stoGens)
#            lnd_error_gen = _np.einsum('i,ijk', HProj, HGens) + \
#                _np.einsum('ij,ijkl', OProj, OGens)
#
#            ham_error_gen = _tools.change_basis(ham_error_gen,"std",basis)
#            sto_error_gen = _tools.change_basis(sto_error_gen,"std",basis)
#            lnd_error_gen = _tools.change_basis(lnd_error_gen,"std",basis)
#
#            gsH.gates[gl]  = _tools.gate_from_error_generator(
#                ham_error_gen, targetGate, genType)
#            gsS.gates[gl]  = _tools.gate_from_error_generator(
#                sto_error_gen, targetGate, genType)
#            gsHS.gates[gl] = _tools.gate_from_error_generator(
#                ham_error_gen+sto_error_gen, targetGate, genType)
#            gsLND.gates[gl] = _tools.gate_from_error_generator(
#                lnd_error_gen, targetGate, genType)
#
#            #CPTP projection
#
#            #Removed attempt to contract H+S to CPTP by removing positive stochastic projections,
#            # but this doesn't always return the gate to being CPTP (maybe b/c of normalization)...
#            #sto_error_gen_cp = _np.einsum('i,ijk', stoProj.clip(None,0), stoGens) #only negative stochastic projections OK
#            #sto_error_gen_cp = _tools.std_to_pp(sto_error_gen_cp)
#            #gsHSCP.gates[gl] = _tools.gate_from_error_generator(
#            #    ham_error_gen, targetGate, genType) #+sto_error_gen_cp
#
#            evals,U = _np.linalg.eig(OProj)
#            pos_evals = evals.clip(0,1e100) #clip negative eigenvalues to 0
#            OProj_cp = _np.dot(U,_np.dot(_np.diag(pos_evals),_np.linalg.inv(U))) #OProj_cp is now a pos-def matrix
#            lnd_error_gen_cp = _np.einsum('i,ijk', HProj, HGens) + \
#                _np.einsum('ij,ijkl', OProj_cp, OGens)
#            lnd_error_gen_cp = _tools.change_basis(lnd_error_gen_cp,"std",basis)
#
#            gsLNDCP.gates[gl] = _tools.gate_from_error_generator(
#                lnd_error_gen_cp, targetGate, genType)
#
#            Np_H += len(hamProj)
#            Np_S += len(stoProj)
#            Np_HS += len(hamProj) + len(stoProj)
#            Np_LND += HProj.size + OProj.size
#
#        #DEBUG!!!
#        #print("DEBUG: BEST sum neg evals = ",_tools.sum_of_negative_choi_evals(gateset))
#        #print("DEBUG: LNDCP sum neg evals = ",_tools.sum_of_negative_choi_evals(gsLNDCP))
#
#        #Check for CPTP where expected
#        #assert(_tools.sum_of_negative_choi_evals(gsHSCP) < 1e-6)
#        assert(_tools.sum_of_negative_choi_evals(gsLNDCP) < 1e-6)
#
#        # ...
#        gatesets = (gateset, gsHS, gsH, gsS, gsLND, cptpGateset, gsLNDCP, gsHSCPTP)
#        gatesetTyps = ("Full","H + S","H","S","LND","CPTP","LND CPTP","H + S CPTP")
#        Nps = (Nng, Np_HS, Np_H, Np_S, Np_LND, Nng, Np_LND, Np_HS)



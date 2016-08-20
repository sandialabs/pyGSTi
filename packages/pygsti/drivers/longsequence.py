from __future__ import division, print_function, absolute_import, unicode_literals
#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************
""" End-to-end functions for performing long-sequence GST """

import os as _os
import warnings as _warnings
import numpy as _np
import sys as _sys
import time as _time
import collections as _collections

from .. import report as _report
from .. import algorithms as _alg
from .. import construction as _construction
from .. import objects as _objs
from .. import io as _io

def do_long_sequence_gst(dataFilenameOrSet, targetGateFilenameOrSet,
                         prepStrsListOrFilename, effectStrsListOrFilename,
                         germsListOrFilename, maxLengths, gateLabels=None,
                         weightsDict=None, fidPairs=None, constrainToTP=True,
                         gaugeOptToCPTP=False, gaugeOptRatio=0.001,
                         gaugeOptItemWeights=None, objective="logl",
                         advancedOptions={}, lsgstLists=None,
                         truncScheme="whole germ powers", comm=None,
                         verbosity=2):
    """
    Perform end-to-end GST analysis using Ls and germs, with L as a maximum
    length.

    Constructs gate strings by repeating germ strings an integer number of
    times such that the length of the repeated germ is less than or equal to
    the maximum length set in maxLengths.  The LGST estimate of the gates is
    computed, gauge optimized, and then used as the seed for either LSGST or
    MLEGST.

    LSGST is iterated ``len(maxLengths)`` times with successively larger sets
    of gate strings.  On the i-th iteration, the repeated germs sequences
    limited by ``maxLengths[i]`` are included in the growing set of strings
    used by LSGST.  The final iteration will use MLEGST when ``objective ==
    "logl"`` to maximize the true log-likelihood instead of minimizing the
    chi-squared function.

    Once computed, the gate set estimates are gauge optimized to the CPTP space
    (if ``gaugeOptToCPTP == True``) and then to the target gate set (using
    `gaugeOptRatio` and `gaugeOptItemWeights`). A :class:`~pygsti.report.Results`
    object is returned, which encapsulates the input and outputs of this GST
    analysis, and can generate final end-user output such as reports and
    presentations.

    Parameters
    ----------
    dataFilenameOrSet : DataSet or string
        The data set object to use for the analysis, specified either directly
        or by the filename of a dataset file (in text format).

    targetGateFilenameOrSet : GateSet or string
        The target gate set, specified either directly or by the filename of a
        gateset file (text format).

    prepStrsListOrFilename : (list of GateStrings) or string
        The state preparation fiducial gate strings, specified either directly
        or by the filename of a gate string list file (text format).

    effectStrsListOrFilename : (list of GateStrings) or string or None
        The measurement fiducial gate strings, specified either directly or by
        the filename of a gate string list file (text format).  If ``None``,
        then use the same strings as specified by prepStrsListOrFilename.

    germsListOrFilename : (list of GateStrings) or string
        The germ gate strings, specified either directly or by the filename of a
        gate string list file (text format).

    maxLengths : list of ints
        List of integers, one per LSGST iteration, which set truncation lengths
        for repeated germ strings.  The list of gate strings for the i-th LSGST
        iteration includes the repeated germs truncated to the L-values *up to*
        and including the i-th one.

    gateLabels : list or tuple
        A list or tuple of the gate labels to use when generating the sets of
        gate strings used in LSGST iterations.  If ``None``, then the gate
        labels of the target gateset will be used.  This option is useful if
        you only want to include a *subset* of the available gates in the LSGST
        strings (e.g. leaving out the identity gate).

    weightsDict : dict, optional
        A dictionary with ``keys == gate strings`` and ``values ==
        multiplicative`` scaling factor for the corresponding gate string. The
        default is no weight scaling at all.

    fidPairs : list of 2-tuples, optional
        Specifies a subset of all prepStr,effectStr string pairs to be used in
        this analysis.  Each element of `fidPairs` is a ``(iRhoStr, iEStr)``
        2-tuple of integers, which index a string within the state preparation
        and measurement fiducial strings respectively.

    constrainToTP : bool, optional
        Whether to constrain GST to trace-preserving gatesets.

    gaugeOptToCPTP : bool, optional
        If ``True``, resulting gate sets are first optimized to CPTP and then
        to the target.  If ``False``, gate sets are only optimized to the
        target gate set.

    gaugeOptRatio : float, optional
        The ratio spamWeight/gateWeight used for gauge optimizing to the target
        gate set.

    gaugeOptItemWeights : dict, optional
       Dictionary of weighting factors for individual gates and spam operators
       used during gauge optimization.   Keys can be gate, state preparation,
       POVM effect, or spam labels.  Values are floating point numbers.  By
       default, gate weights are set to 1.0 and spam weights to gaugeOptRatio.

    objective : {'chi2', 'logl'}, optional
        Specifies which final objective function is used: the chi-squared or
        the log-likelihood.

    advancedOptions : dict, optional
        Specifies advanced options most of which deal with numerical details of
        the objective function.

    lsgstLists : list of gate string lists, optional
        Provides explicit list of gate string lists to be used in analysis; to
        be given if the dataset uses "incomplete" or "reduced" sets of gate
        string.  Default is ``None``.

    truncScheme : str, optional
        Truncation scheme used to interpret what the list of maximum lengths
        means. If unsure, leave as default. Allowed values are:

        - ``'whole germ powers'`` -- germs are repeated an integer number of
          times such that the length is less than or equal to the max.
        - ``'truncated germ powers'`` -- repeated germ string is truncated
          to be exactly equal to the max (partial germ at end is ok).
        - ``'length as exponent'`` -- max. length is instead interpreted
          as the germ exponent (the number of germ repetitions).

    comm : mpi4py.MPI.Comm, optional
        When not ``None``, an MPI communicator for distributing the computation
        across multiple processors.

    verbosity : int, optional
       The 'verbosity' option is an integer specifying the level of 
       detail printed to stdout during the calculation.


    Returns
    -------
    Results
    """

    tRef = _time.time(); times_list = []
    if 'verbosity' in advancedOptions: #for backward compatibility
        _warnings.warn("'verbosity' as an advanced option is deprecated." +
                       " Please use the 'verbosity' argument directly.")
        verbosity = advancedOptions['verbosity'] 
    printer = _objs.VerbosityPrinter.build_printer(verbosity, comm)

    #Get target gateset
    if isinstance(targetGateFilenameOrSet, str):
        gs_target = _io.load_gateset(targetGateFilenameOrSet)
    else:
        gs_target = targetGateFilenameOrSet #assume a GateSet object

    #Get dataset
    if isinstance(dataFilenameOrSet, str):
        ds = _io.load_dataset(dataFilenameOrSet, True, verbosity) #can't take a printer...
        default_dir = _os.path.dirname(dataFilenameOrSet) #default directory for reports, etc
        default_base = _os.path.splitext( _os.path.basename(dataFilenameOrSet) )[0]
    else:
        ds = dataFilenameOrSet #assume a Dataset object
        default_dir = default_base = None

    if isinstance(prepStrsListOrFilename, str):
        prepStrs = _io.load_gatestring_list(prepStrsListOrFilename)
    else: prepStrs = prepStrsListOrFilename

    if effectStrsListOrFilename is None:
        effectStrs = prepStrs #use same strings for effectStrs if effectStrsListOrFilename is None
    else:
        if isinstance(effectStrsListOrFilename, str):
            effectStrs = _io.load_gatestring_list(effectStrsListOrFilename)
        else: effectStrs = effectStrsListOrFilename

    if isinstance(germsListOrFilename, str):
        germs = _io.load_gatestring_list(germsListOrFilename)
    else: germs = germsListOrFilename
    if lsgstLists is None:

        #Get gate strings and labels
        if gateLabels is None:
            gateLabels = list(gs_target.gates.keys())

        nest = advancedOptions.get('nestedGateStringLists',True)
        lsgstLists = _construction.stdlists.make_lsgst_lists(
            gateLabels, prepStrs, effectStrs, germs, maxLengths, fidPairs,
            truncScheme, nest)

    tNxt = _time.time()
    times_list.append( ('Loading',tNxt-tRef) ); tRef=tNxt

    #Starting Point = LGST
    gate_dim = gs_target.get_dimension()
    specs = _construction.build_spam_specs(prepStrs=prepStrs, effectStrs=effectStrs,
                                           prep_labels=gs_target.get_prep_labels(),
                                           effect_labels=gs_target.get_effect_labels())
    gs_lgst = _alg.do_lgst(ds, specs, gs_target, svdTruncateTo=gate_dim,
                           verbosity=printer)

    tNxt = _time.time()
    times_list.append( ('LGST',tNxt-tRef) ); tRef=tNxt

    if constrainToTP: #gauge optimize (and contract if needed) to TP, then lock down first basis element as the identity
        #TODO: instead contract to vSPAM? (this could do more than just alter the 1st element...)
        gs_lgst.set_all_parameterizations("full") #make sure we can do gauge optimization
        minPenalty, _, gs_in_TP = _alg.optimize_gauge(
            gs_lgst, "TP",  returnAll=True, spamWeight=1.0,
            gateWeight=1.0, verbosity=printer)
            #Note: no  itemWeights=gaugeOptItemWeights here (LGST)

        if minPenalty > 0:
            gs_in_TP = _alg.contract(gs_in_TP, "TP")
            if minPenalty > 1e-5:
                _warnings.warn("Could not gauge optimize to TP (penalty=%g), so contracted LGST gateset to TP" % minPenalty)

        gs_after_gauge_opt = _alg.optimize_gauge(
            gs_in_TP, "target", targetGateset=gs_target, constrainToTP=True,
            spamWeight=1.0, gateWeight=1.0)
            #Note: no  itemWeights=gaugeOptItemWeights here (LGST)

        firstElIdentityVec = _np.zeros( (gate_dim,1) )
        firstElIdentityVec[0] = gate_dim**0.25 # first basis el is assumed = sqrt(gate_dim)-dimensional identity density matrix
        gs_after_gauge_opt.povm_identity = firstElIdentityVec # declare that this basis has the identity as its first element

    else: # no TP constraint
        gs_after_gauge_opt = _alg.optimize_gauge(
            gs_lgst, "target", targetGateset=gs_target,
            spamWeight=1.0, gateWeight=1.0)
            #Note: no  itemWeights=gaugeOptItemWeights here (LGST)

        #TODO: set identity vector, or leave as is, which assumes LGST had the right one and contraction doesn't change it ??
        # Really, should we even allow use of the identity vector when doing a non-TP-constrained optimization?

    #Advanced Options can specify further manipulation of LGST seed
    if advancedOptions.get('contractLGSTtoCPTP',False):
        gs_after_gauge_opt = _alg.contract(gs_after_gauge_opt, "CPTP")
    if advancedOptions.get('depolarizeLGST',0) > 0:
        gs_after_gauge_opt = gs_after_gauge_opt.depolarize(gate_noise=advancedOptions['depolarizeLGST'])

    if constrainToTP:
        gs_after_gauge_opt.set_all_parameterizations("TP")

    tNxt = _time.time()
    times_list.append( ('Prep LGST seed',tNxt-tRef) ); tRef=tNxt

    #Run LSGST on data
    if objective == "chi2":
        gs_lsgst_list = _alg.do_iterative_mc2gst(
            ds, gs_after_gauge_opt, lsgstLists,
            minProbClipForWeighting=advancedOptions.get(
                'minProbClipForWeighting',1e-4),
            probClipInterval = advancedOptions.get(
                'probClipInterval',(-1e6,1e6)),
            returnAll=True, gatestringWeightsDict=weightsDict,
            verbosity=printer,
            memLimit=advancedOptions.get('memoryLimitInBytes',None),
            useFreqWeightedChiSq=advancedOptions.get(
                'useFreqWeightedChiSq',False), times=times_list,
            comm=comm, distributeMethod=advancedOptions.get(
                'distributeMethod',"gatestrings") )
    elif objective == "logl":
        gs_lsgst_list = _alg.do_iterative_mlgst(
          ds, gs_after_gauge_opt, lsgstLists,
          minProbClip = advancedOptions.get('minProbClip',1e-4),
          probClipInterval = advancedOptions.get('probClipInterval',(-1e6,1e6)),
          radius=advancedOptions.get('radius',1e-4),
          returnAll=True, verbosity=printer,
          memLimit=advancedOptions.get('memoryLimitInBytes',None),
          useFreqWeightedChiSq=advancedOptions.get(
                'useFreqWeightedChiSq',False), times=times_list,
          comm=comm, distributeMethod=advancedOptions.get(
                'distributeMethod',"gatestrings"))
    else:
        raise ValueError("Invalid longSequenceObjective: %s" % objective)

    tNxt = _time.time()
    times_list.append( ('Total long-seq. opt.',tNxt-tRef) ); tRef=tNxt

    #Run the gatesets through gauge optimization, first to CPTP then to target
    #   so fidelity and frobenius distance w/targets is more meaningful
    if gaugeOptToCPTP:
        printer.log("\nGauge Optimizing to CPTP...",2)
        go_gs_lsgst_list = [_alg.optimize_gauge(
                gs,'CPTP',constrainToTP=constrainToTP) for gs in gs_lsgst_list]
        #Note: don't set itemWeights in optimize_gauge (doesn't apply to CPTP)

        tNxt = _time.time()
        times_list.append( ('Gauge opt to CPTP',tNxt-tRef) ); tRef=tNxt
    else:
        go_gs_lsgst_list = gs_lsgst_list

    #Note: we used to make constrainToCP contingent on whether each
    # gateset was already in CP, i.e. only constrain if
    # _tools.sum_of_negative_choi_evals(gs) < 1e-8.  But we rarely use
    # CP constraints, and this complicates the logic -- so now when
    # gaugeOptToCPTP == True, always constrain to CP.
    go_params = _collections.OrderedDict([
            ('toGetTo', 'target'),
            ('constrainToTP', constrainToTP),
            ('constrainToCP', gaugeOptToCPTP),
            ('gateWeight', 1.0),
            ('spamWeight', gaugeOptRatio),
            ('targetGatesMetric',"frobenius"),
            ('targetSpamMetric',"frobenius"),
            ('itemWeights', gaugeOptItemWeights) ])

    for i, gs in enumerate(go_gs_lsgst_list):
        args = go_params.copy()
        args['gateset'] = gs
        args['targetGateset'] = gs_target
        go_gs_lsgst_list[i] = _alg.optimize_gauge(**args)

    tNxt = _time.time()
    times_list.append( ('Gauge opt to target',tNxt-tRef) ); tRef=tNxt

    truncFn = _construction.stdlists._getTruncFunction(truncScheme)

    ret = _report.Results()
    ret.init_Ls_and_germs(objective, gs_target, ds,
                        gs_after_gauge_opt, maxLengths, germs,
                        go_gs_lsgst_list, lsgstLists, prepStrs, effectStrs,
                        truncFn,  constrainToTP, fidPairs, gs_lsgst_list)
    ret.parameters['minProbClip'] = \
        advancedOptions.get('minProbClip',1e-4)
    ret.parameters['minProbClipForWeighting'] = \
        advancedOptions.get('minProbClipForWeighting',1e-4)
    ret.parameters['probClipInterval'] = \
        advancedOptions.get('probClipInterval',(-1e6,1e6))
    ret.parameters['radius'] = advancedOptions.get('radius',1e-4)
    ret.parameters['weights'] = weightsDict
    ret.parameters['defaultDirectory'] = default_dir
    ret.parameters['defaultBasename'] = default_base
    ret.parameters['memLimit'] = advancedOptions.get('memoryLimitInBytes',None)
    ret.parameters['gaugeOptParams'] = go_params

    times_list.append( ('Results initialization',_time.time()-tRef) )
    ret.parameters['times'] = times_list

    assert( len(maxLengths) == len(lsgstLists) == len(go_gs_lsgst_list) )
    return ret

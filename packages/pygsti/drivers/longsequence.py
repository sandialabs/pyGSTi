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
                         weightsDict=None, fidPairs=None,
                         gaugeOptRatio=0.001,
                         gaugeOptItemWeights=None, objective="logl",
                         advancedOptions={}, lsgstLists=None,
                         truncScheme="whole germ powers", comm=None,
                         profile=1,verbosity=2):
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

    Once computed, the gate set estimates are optionally gauge optimized to
    the CPTP space and then to the target gate set (using `gaugeOptRatio`
    and `gaugeOptItemWeights`). A :class:`~pygsti.report.Results`
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

    fidPairs : list of 2-tuples or dict, optional
        Specifies a subset of all prepStr,effectStr string pairs to be used in
        this analysis.  If `fidPairs` is a list, each element of `fidPairs` is a
        ``(iRhoStr, iEStr)`` 2-tuple of integers, which index a string within
        the state preparation and measurement fiducial strings respectively. If
        `fidPairs` is a dict, then the keys must be germ strings and values are
        lists of 2-tuples as in the previous case.

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

    profile : int, optional
        Whether or not to perform lightweight timing and memory profiling.
        Allowed values are:
        
        - 0 -- no profiling is performed
        - 1 -- profiling enabled, but don't print anything in-line
        - 2 -- profiling enabled, and print memory usage at checkpoints

    verbosity : int, optional
       The 'verbosity' option is an integer specifying the level of 
       detail printed to stdout during the calculation.

       - 0 -- prints nothing
       - 1 -- shows progress bar for entire iterative GST
       - 2 -- show summary details about each individual iteration
       - 3 -- also shows outer iterations of LM algorithm
       - 4 -- also shows inner iterations of LM algorithm
       - 5 -- also shows detailed info from within jacobian
              and objective function calls.

    Returns
    -------
    Results
    """

    tRef = _time.time()

    if profile == 0: profiler = _objs.DummyProfiler()
    elif profile == 1: profiler = _objs.Profiler(comm,False)
    elif profile == 2: profiler = _objs.Profiler(comm,True)
    else: raise ValueError("Invalid value for 'profile' argument (%s)"%profile)

    if 'verbosity' in advancedOptions: #for backward compatibility
        _warnings.warn("'verbosity' as an advanced option is deprecated." +
                       " Please use the 'verbosity' argument directly.")
        verbosity = advancedOptions['verbosity'] 
    printer = _objs.VerbosityPrinter.build_printer(verbosity, comm)

    #Get/load target gateset
    if isinstance(targetGateFilenameOrSet, str):
        gs_target = _io.load_gateset(targetGateFilenameOrSet)
    else:
        gs_target = targetGateFilenameOrSet #assume a GateSet object

    #Get/load dataset
    if isinstance(dataFilenameOrSet, str):
        ds = _io.load_dataset(dataFilenameOrSet, True, "aggregate", printer)
        default_dir = _os.path.dirname(dataFilenameOrSet) #default directory for reports, etc
        default_base = _os.path.splitext( _os.path.basename(dataFilenameOrSet) )[0]
    else:
        ds = dataFilenameOrSet #assume a Dataset object
        default_dir = default_base = None

    #Get/load fiducials
    if isinstance(prepStrsListOrFilename, str):
        prepStrs = _io.load_gatestring_list(prepStrsListOrFilename)
    else: prepStrs = prepStrsListOrFilename

    if effectStrsListOrFilename is None:
        effectStrs = prepStrs #use same strings for effectStrs if effectStrsListOrFilename is None
    else:
        if isinstance(effectStrsListOrFilename, str):
            effectStrs = _io.load_gatestring_list(effectStrsListOrFilename)
        else: effectStrs = effectStrsListOrFilename

    #Get/load germs
    if isinstance(germsListOrFilename, str):
        germs = _io.load_gatestring_list(germsListOrFilename)
    else: germs = germsListOrFilename

    #Construct gate sequences
    if lsgstLists is None:

        #Get gate strings and labels
        if gateLabels is None:
            gateLabels = list(gs_target.gates.keys())

        nest = advancedOptions.get('nestedGateStringLists',True)
        lsgstLists = _construction.stdlists.make_lsgst_lists(
            gateLabels, prepStrs, effectStrs, germs, maxLengths, fidPairs,
            truncScheme, nest)

    tNxt = _time.time()
    profiler.add_time('do_long_sequence_gst: loading',tRef); tRef=tNxt

    #Starting Point - compute on rank 0 and distribute
    startingPt = advancedOptions.get('starting point',"LGST")
    gate_dim = gs_target.get_dimension()
    if comm is None or comm.Get_rank() == 0:

        #Compute starting point
        if startingPt == "LGST":
            specs = _construction.build_spam_specs(prepStrs=prepStrs, effectStrs=effectStrs,
                                                   prep_labels=gs_target.get_prep_labels(),
                                                   effect_labels=gs_target.get_effect_labels())
            gs_start = _alg.do_lgst(ds, specs, gs_target, svdTruncateTo=gate_dim,
                                    verbosity=printer) # returns a gateset with the *same*
                                                       # parameterizations as gs_target
        elif startingPt == "target":
            gs_start = gs_target.copy()
        else:
            raise ValueError("Invalid starting point: %s" % startingPt)
        
        tNxt = _time.time()
        profiler.add_time('do_long_sequence_gst: Starting Point (%s)'
                          % startingPt,tRef); tRef=tNxt

        #Gauge optimimize starting point to the target (historical, and
        # sometimes seems to help in practice, since it's gauge optimizing
        # to something in the realm of physical gates (e.g. is CPTP)
        gs_after_gauge_opt = _alg.gaugeopt_to_target(gs_start, gs_target)
          #use all *default* weights when optimizing the starting point

        if startingPt == "LGST":
            # reset the POVM identity to that of the target after gauge optimizing
            # to the target if LGST was used.  Essentially, we declare that this 
            # basis (gauge) has the same identity as the target (typically the
            # first basis element).
            gs_after_gauge_opt.povm_identity = gs_target.povm_identity.copy()
                


#        if constrainToTP: #gauge optimize (and contract if needed) to TP, then lock down first basis element as the identity
#            #TODO: instead contract to vSPAM? (this could do more than just alter the 1st element...)
#            gs_lgst.set_all_parameterizations("full") #make sure we can do gauge optimization
#            minPenalty, _, gs_in_TP = _alg.optimize_gauge(
#                gs_lgst, "TP",  returnAll=True, spamWeight=1.0,
#                gateWeight=1.0, verbosity=printer)
#                #Note: no  itemWeights=gaugeOptItemWeights here (LGST)
#        
#            if minPenalty > 0:
#                gs_in_TP = _alg.contract(gs_in_TP, "TP")
#                if minPenalty > 1e-5:
#                    _warnings.warn("Could not gauge optimize to TP (penalty=%g), so contracted LGST gateset to TP" % minPenalty)
#        
#            if startingPt == "LGST": #only need to gauge optmize LGST result
#                gs_after_gauge_opt = _alg.optimize_gauge(
#                    gs_in_TP, "target", targetGateset=gs_target, constrainToTP=True,
#                    spamWeight=1.0, gateWeight=1.0)
#                    #Note: no  itemWeights=gaugeOptItemWeights here (LGST)
#    
#                firstElIdentityVec = _np.zeros( (gate_dim,1) )
#                firstElIdentityVec[0] = gate_dim**0.25 # first basis el is assumed = sqrt(gate_dim)-dimensional identity density matrix
#                gs_after_gauge_opt.povm_identity = firstElIdentityVec # declare that this basis has the identity as its first element
#            else:
#                gs_after_gauge_opt = gs_in_TP.copy()
#    
#        else: # no TP constraint, just gauge optimize
#            if startingPt == "LGST": #don't need to gauge optimize 'target' starting pt
#                try:
#                    printer.log("\nGauge Optimizing without constraints...",2)
#                    gs_after_gauge_opt = _alg.optimize_gauge(
#                        gs_lgst, "target", targetGateset=gs_target,
#                        spamWeight=1.0, gateWeight=1.0)
#                    printer.log("Success!",2)
#                    #Note: no  itemWeights=gaugeOptItemWeights here (LGST)
#                except:
#                    try:
#                        printer.log("Failed! Trying with TP constraint...",2)
#                        gs_after_gauge_opt = _alg.optimize_gauge(
#                            gs_lgst, "target", targetGateset=gs_target, 
#                            constrainToTP=True, spamWeight=1.0, gateWeight=1.0)
#                        printer.log("Success!",2)
#                        #Note: no  itemWeights=gaugeOptItemWeights here (LGST)
#                    except:
#                        printer.log("Still Failed! No gauge optimization " +
#                                    "performed on LGST estimate.",2)
#                        gs_after_gauge_opt = gs_lgst.copy()
#            else:
#                gs_after_gauge_opt = gs_lgst.copy()
    
            #TODO: set identity vector, or leave as is, which assumes LGST had the right one and contraction doesn't change it ??
            # Really, should we even allow use of the identity vector when doing a non-TP-constrained optimization?
    
        #Advanced Options can specify further manipulation of Initial seed
        if advancedOptions.get('contractInitialToCPTP',False):
            gs_after_gauge_opt = _alg.contract(gs_after_gauge_opt, "CPTP")
        if advancedOptions.get('depolarizeInitial',0) > 0:
            gs_after_gauge_opt = gs_after_gauge_opt.depolarize(gate_noise=advancedOptions['depolarizeInitial'])
    
#        if constrainToTP:
#            gs_after_gauge_opt.set_all_parameterizations("TP")

        if comm is not None: #broadcast starting gate set
            comm.bcast(gs_after_gauge_opt, root=0)
    else:
        gs_after_gauge_opt = comm.bcast(None, root=0)


    tNxt = _time.time()
    profiler.add_time('do_long_sequence_gst: Prep Initial seed',tRef); tRef=tNxt

    #Run LSGST on data
    if objective == "chi2":
        gs_lsgst_list = _alg.do_iterative_mc2gst(
            ds, gs_after_gauge_opt, lsgstLists,
            tol = advancedOptions.get('tolerance',1e-6),
            maxiter = advancedOptions.get('maxIterations',100000),
            minProbClipForWeighting=advancedOptions.get(
                'minProbClipForWeighting',1e-4),
            probClipInterval = advancedOptions.get(
                'probClipInterval',(-1e6,1e6)),
            returnAll=True, gatestringWeightsDict=weightsDict,
            verbosity=printer,
            memLimit=advancedOptions.get('memoryLimitInBytes',None),
            useFreqWeightedChiSq=advancedOptions.get(
                'useFreqWeightedChiSq',False), profiler=profiler,
            comm=comm, distributeMethod=advancedOptions.get(
                'distributeMethod',"gatestrings") )
    elif objective == "logl":
        gs_lsgst_list = _alg.do_iterative_mlgst(
          ds, gs_after_gauge_opt, lsgstLists,
          tol = advancedOptions.get('tolerance',1e-6),
          maxiter = advancedOptions.get('maxIterations',100000),
          minProbClip = advancedOptions.get('minProbClip',1e-4),
          probClipInterval = advancedOptions.get('probClipInterval',(-1e6,1e6)),
          radius=advancedOptions.get('radius',1e-4),
          returnAll=True, verbosity=printer,
          memLimit=advancedOptions.get('memoryLimitInBytes',None),
          useFreqWeightedChiSq=advancedOptions.get(
                'useFreqWeightedChiSq',False), profiler=profiler,
          comm=comm, distributeMethod=advancedOptions.get(
                'distributeMethod',"gatestrings"))
    else:
        raise ValueError("Invalid longSequenceObjective: %s" % objective)

    tNxt = _time.time()
    profiler.add_time('do_long_sequence_gst: total long-seq. opt.',tRef); tRef=tNxt

    #Do final gauge optimization
    gaugeOptType = advancedOptions.get("gauge optimization", "target")
    go_gs_lsgst_list = gs_lsgst_list
    go_params_list = []

    if gaugeOptType in ("CPTP then target", "target", "CPTP"):
        if "CPTP" in gaugeOptType:
            printer.log("\nGauge Optimizing to CPTP...",2)
            go_params = _collections.OrderedDict([
                    ('CPpenalty', 1.0),
                    ('TPpenalty', 1.0),
                    ('validSpamPenalty', 1.0) ])
            go_gs_lsgst_list = [
                _alg.gaugeopt_to_target(gs, None, **go_params) 
                for gs in go_gs_lsgst_list]
            tNxt = _time.time()
            profiler.add_time('do_long_sequence_gst: gauge opt to CPTP',tRef); tRef=tNxt
            go_params_list.append(go_params)

        if "target" in gaugeOptType:
            printer.log("\nGauge Optimizing to target...",2)
            CPTPpenalty = 1.0 if ("CPTP" in gaugeOptType) else 0
            itemWeights = { 'gates': 1.0, 'spam': gaugeOptRatio }
            if gaugeOptItemWeights is not None:
                itemWeights.update(gaugeOptItemWeights)
                            
            go_params = _collections.OrderedDict([
                    ('itemWeights', itemWeights),
                    ('CPpenalty', CPTPpenalty),
                    ('TPpenalty', CPTPpenalty),
                    ('validSpamPenalty', CPTPpenalty),
                    ('gatesMetric',"frobenius"),
                    ('spamMetric',"frobenius") ])
            go_gs_lsgst_list = [
                _alg.gaugeopt_to_target(gs, gs_target, **go_params) 
                for gs in go_gs_lsgst_list]

            tNxt = _time.time()
            profiler.add_time('do_long_sequence_gst: gauge opt to target',tRef); tRef=tNxt
            go_params_list.append(go_params)

    elif gaugeOptType == "none":
        go_params_list.append( _collections.OrderedDict() )
    else:
        raise ValueError("Invalid gauge optmization type: %s" % gaugeOptType)

    truncFn = _construction.stdlists._getTruncFunction(truncScheme)

    ret = _report.Results()
    ret.init_Ls_and_germs(objective, gs_target, ds,
                        gs_after_gauge_opt, maxLengths, germs,
                        go_gs_lsgst_list, lsgstLists, prepStrs, effectStrs,
                        truncFn, fidPairs, gs_lsgst_list)
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
    ret.parameters['gaugeOptParams'] = go_params_list

    profiler.add_time('do_long_sequence_gst: results initialization',tRef)
    ret.parameters['profiler'] = profiler

    assert( len(maxLengths) == len(lsgstLists) == len(go_gs_lsgst_list) )
    return ret

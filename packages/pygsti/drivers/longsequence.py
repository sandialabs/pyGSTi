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
                         germsListOrFilename, maxLengths, gaugeOptParams=None,
                         objective="logl", fidPairs=None, lsgstLists=None,
                         advancedOptions=None, comm=None, memLimit=None,
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
        
    gaugeOptParams : dict, optional
        A dictionary of arguments to :func:`gaugeopt_to_target`, specifying
        how the final gauge optimization should be performed.  The keys and
        values of this dictionary may correspond to any of the arguments
        of :func:`gaugeopt_to_target` *except* for the first `gateset` 
        argument, which is specified internally.  The `targetGateset` argument,
        *can* be set, but is specified internally when it isn't.  If `None`,
        then the dictionary `{'itemWeights': {'gates':1.0, 'spam':0.001}}`
        is used.  If `False`, then then *no* gauge optimization is performed.

    objective : {'chi2', 'logl'}, optional
        Specifies which final objective function is used: the chi-squared or
        the log-likelihood.

    fidPairs : list of 2-tuples or dict, optional
        Specifies a subset of all prepStr,effectStr string pairs to be used in
        this analysis.  If `fidPairs` is a list, each element of `fidPairs` is a
        ``(iRhoStr, iEStr)`` 2-tuple of integers, which index a string within
        the state preparation and measurement fiducial strings respectively. If
        `fidPairs` is a dict, then the keys must be germ strings and values are
        lists of 2-tuples as in the previous case.

    lsgstLists : list of gate string lists, optional
        Provides explicit list of gate string lists to be used in analysis; to
        be given if the dataset uses "incomplete" or "reduced" sets of gate
        string.  Default is ``None``.

    advancedOptions : dict, optional
        Specifies advanced options most of which deal with numerical details of
        the objective function or expert-level functionality.  The allowed keys
        and values include:

        - gateLabels = list of strings
        - gsWeights = dict or None
        - starting point = "LGST" (default) or  "target" or GateSet
        - depolarizeStart = float (default == 0)
        - contractStartToCPTP = True / False (default)
        - cptpPenaltyFactor = float (default = 0)
        - tolerance = float
        - maxIterations = int
        - minProbClip = float
        - minProbClipForWeighting = float (default == 1e-4)
        - probClipInterval = tuple (default == (-1e6,1e6)
        - radius = float (default == 1e-4)
        - useFreqWeightedChiSq = True / False (default)
        - nestedGateStringLists = True (default) / False
        - distributeMethod = "gatestrings" or "deriv" (default)
        - profile = int (default == 1)
        - check = True / False (default)
        - truncScheme = "whole germ powers" (default) or "truncated germ powers"
                        or "length as exponent"

    comm : mpi4py.MPI.Comm, optional
        When not ``None``, an MPI communicator for distributing the computation
        across multiple processors.

    memLimit : int or None, optional
        A rough memory limit in bytes which restricts the amount of memory 
        used (per core when run on multi-CPUs).

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

    #gateLabels : list or tuple
    #    A list or tuple of the gate labels to use when generating the sets of
    #    gate strings used in LSGST iterations.  If ``None``, then the gate
    #    labels of the target gateset will be used.  This option is useful if
    #    you only want to include a *subset* of the available gates in the LSGST
    #    strings (e.g. leaving out the identity gate).
    #
    #weightsDict : dict, optional
    #    A dictionary with ``keys == gate strings`` and ``values ==
    #    multiplicative`` scaling factor for the corresponding gate string. The
    #    default is no weight scaling at all.
    #
    #gaugeOptRatio : float, optional
    #    The ratio spamWeight/gateWeight used for gauge optimizing to the target
    #    gate set.
    #
    #gaugeOptItemWeights : dict, optional
    #   Dictionary of weighting factors for individual gates and spam operators
    #   used during gauge optimization.   Keys can be gate, state preparation,
    #   POVM effect, or spam labels.  Values are floating point numbers.  By
    #   default, gate weights are set to 1.0 and spam weights to gaugeOptRatio.
    #profile : int, optional
    #    Whether or not to perform lightweight timing and memory profiling.
    #    Allowed values are:
    #    
    #    - 0 -- no profiling is performed
    #    - 1 -- profiling enabled, but don't print anything in-line
    #    - 2 -- profiling enabled, and print memory usage at checkpoints
    #truncScheme : str, optional
    #    Truncation scheme used to interpret what the list of maximum lengths
    #    means. If unsure, leave as default. Allowed values are:
    #
    #    - ``'whole germ powers'`` -- germs are repeated an integer number of
    #      times such that the length is less than or equal to the max.
    #    - ``'truncated germ powers'`` -- repeated germ string is truncated
    #      to be exactly equal to the max (partial germ at end is ok).
    #    - ``'length as exponent'`` -- max. length is instead interpreted
    #      as the germ exponent (the number of germ repetitions).





    tRef = _time.time()

    #Note: *don't* specify default dictionary arguments, as this is dangerous
    # because they are mutable objects
    if advancedOptions is None: advancedOptions = {}
    if gaugeOptParams is None: 
        gaugeOptParams = {'itemWeights': {'gates':1.0, 'spam':0.001}}

    profile = advancedOptions.get('profile',1)
    truncScheme = advancedOptions.get('truncScheme',"whole germ powers")

    if profile == 0: profiler = _objs.DummyProfiler()
    elif profile == 1: profiler = _objs.Profiler(comm,False)
    elif profile == 2: profiler = _objs.Profiler(comm,True)
    else: raise ValueError("Invalid value for 'profile' argument (%s)"%profile)

    if 'verbosity' in advancedOptions: #for backward compatibility
        _warnings.warn("'verbosity' as an advanced option is deprecated." +
                       " Please use the 'verbosity' argument directly.")
        verbosity = advancedOptions['verbosity'] 
    if 'memoryLimitInBytes' in advancedOptions: #for backward compatibility
        _warnings.warn("'memoryLimitInBytes' as an advanced option is deprecated." +
                       " Please use the 'memLimit' argument directly.")
        memLimit = advancedOptions['memoryLimitInBytes']

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
        gateLabels = advancedOptions.get(
            'gateLabels', list(gs_target.gates.keys()))
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

            #In LGST case, gauge optimimize starting point to the target
            # (historical; sometimes seems to help in practice, since it's gauge
            # optimizing to physical gates (e.g. something in CPTP)
            gs_start = _alg.gaugeopt_to_target(gs_start, gs_target)
              #Note: use *default* gauge-opt params when optimizing

            # Also reset the POVM identity to that of the target.  Essentially,
            # we declare that this basis (gauge) has the same identity as the
            # target (typically the first basis element).
            gs_start.povm_identity = gs_target.povm_identity.copy()

        elif startingPt == "target":
            gs_start = gs_target.copy()
        elif isinstance(startingPt, _objs.GateSet):
            gs_start = startingPt
            startingPt = "User-supplied-GateSet" #for profiler log below
        else:
            raise ValueError("Invalid starting point: %s" % startingPt)
        
        tNxt = _time.time()
        profiler.add_time('do_long_sequence_gst: Starting Point (%s)'
                          % startingPt,tRef); tRef=tNxt                
    
        #Advanced Options can specify further manipulation of starting gate set
        if advancedOptions.get('contractStartToCPTP',False):
            gs_start = _alg.contract(gs_start, "CPTP")
        if advancedOptions.get('depolarizeStart',0) > 0:
            gs_start = gs_start.depolarize(gate_noise=advancedOptions['depolarizeStart'])
    
        if comm is not None: #broadcast starting gate set
            comm.bcast(gs_start, root=0)
    else:
        gs_start = comm.bcast(None, root=0)

    tNxt = _time.time()
    profiler.add_time('do_long_sequence_gst: Prep Initial seed',tRef); tRef=tNxt

    #Run Long-sequence GST on data
    if objective == "chi2":
        gs_lsgst_list = _alg.do_iterative_mc2gst(
            ds, gs_start, lsgstLists,
            tol = advancedOptions.get('tolerance',1e-6),
            cptp_penalty_factor = advancedOptions.get('cptpPenaltyFactor',0),
            maxiter = advancedOptions.get('maxIterations',100000),
            minProbClipForWeighting=advancedOptions.get(
                'minProbClipForWeighting',1e-4),
            probClipInterval = advancedOptions.get(
                'probClipInterval',(-1e6,1e6)),
            returnAll=True, 
            gatestringWeightsDict=advancedOptions.get('gsWeights',None),
            verbosity=printer,
            memLimit=memLimit,
            useFreqWeightedChiSq=advancedOptions.get(
                'useFreqWeightedChiSq',False), profiler=profiler,
            comm=comm, distributeMethod=advancedOptions.get(
                'distributeMethod',"deriv"),
            check_jacobian=advancedOptions.get('check',False),
            check=advancedOptions.get('check',False))

    elif objective == "logl":
        gs_lsgst_list = _alg.do_iterative_mlgst(
          ds, gs_start, lsgstLists,
          tol = advancedOptions.get('tolerance',1e-6),
          cptp_penalty_factor = advancedOptions.get('cptpPenaltyFactor',0),
          maxiter = advancedOptions.get('maxIterations',100000),
          minProbClip = advancedOptions.get('minProbClip',1e-4),
          probClipInterval = advancedOptions.get('probClipInterval',(-1e6,1e6)),
          radius=advancedOptions.get('radius',1e-4),
          returnAll=True, verbosity=printer,
          memLimit=memLimit, profiler=profiler, comm=comm,
          useFreqWeightedChiSq=advancedOptions.get(
                'useFreqWeightedChiSq',False), 
          distributeMethod=advancedOptions.get(
                'distributeMethod',"deriv"),
          check=advancedOptions.get('check',False))
    else:
        raise ValueError("Invalid longSequenceObjective: %s" % objective)

    tNxt = _time.time()
    profiler.add_time('do_long_sequence_gst: total long-seq. opt.',tRef); tRef=tNxt

    #Do final gauge optimization.  gaugeOptParams can be a dict or a list of
    # dicts, each specifying a successive "stage" of gauge optimization.
    go_gs_lsgst_list = gs_lsgst_list
    if gaugeOptParams != False:
        if hasattr(gaugeOptParams,"keys"):
            go_params_list = [gaugeOptParams]
        else: go_params_list = gaugeOptParams

        ordered_go_params_list = []
        for go_params in go_params_list:
            if "targetGateset" not in go_params:
                go_params["targetGateset"] = gs_target

            ordered_go_params_list.append( _collections.OrderedDict( 
                [(k,go_params[k]) for k in sorted(list(go_params.keys()))]))

            go_gs_lsgst_list = [ _alg.gaugeopt_to_target(gs,**go_params)
                                 for gs in go_gs_lsgst_list]

        tNxt = _time.time()
        profiler.add_time('do_long_sequence_gst: gauge optimization',tRef); tRef=tNxt

    truncFn = _construction.stdlists._getTruncFunction(truncScheme)

    ret = _report.Results()
    ret.init_Ls_and_germs(objective, gs_target, ds,
                        gs_start, maxLengths, germs,
                        go_gs_lsgst_list, lsgstLists, prepStrs, effectStrs,
                        truncFn, fidPairs, gs_lsgst_list)
    ret.parameters['minProbClip'] = \
        advancedOptions.get('minProbClip',1e-4)
    ret.parameters['minProbClipForWeighting'] = \
        advancedOptions.get('minProbClipForWeighting',1e-4)
    ret.parameters['probClipInterval'] = \
        advancedOptions.get('probClipInterval',(-1e6,1e6))
    ret.parameters['radius'] = advancedOptions.get('radius',1e-4)
    ret.parameters['weights'] = advancedOptions.get('gsWeights',None)
    ret.parameters['defaultDirectory'] = default_dir
    ret.parameters['defaultBasename'] = default_base
    ret.parameters['memLimit'] = memLimit
    ret.parameters['gaugeOptParams'] = ordered_go_params_list
    ret.parameters['cptpPenaltyFactor'] = advancedOptions.get('cptpPenaltyFactor',0)
    ret.parameters['distributeMethod'] = advancedOptions.get('distributeMethod','deriv')

    profiler.add_time('do_long_sequence_gst: results initialization',tRef)
    ret.parameters['profiler'] = profiler

    assert( len(maxLengths) == len(lsgstLists) == len(go_gs_lsgst_list) )
    return ret

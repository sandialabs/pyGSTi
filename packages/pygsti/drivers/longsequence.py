""" End-to-end functions for performing long-sequence GST """
from __future__ import division, print_function, absolute_import, unicode_literals
#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************

import os as _os
import warnings as _warnings
import numpy as _np
import time as _time
import collections as _collections
import pickle as _pickle
from scipy.stats import chi2 as _chi2

from .. import algorithms as _alg
from .. import construction as _construction
from .. import objects as _objs
from .. import io as _io
from .. import tools as _tools
from ..tools import compattools as _compat
from ..baseobjs import DummyProfiler as _DummyProfiler

ROBUST_SUFFIX_LIST = [".robust", ".Robust", ".robust+", ".Robust+"]
DEFAULT_BAD_FIT_THRESHOLD = 2.0

def do_model_test(modelGateFilenameOrSet,
                  dataFilenameOrSet, targetGateFilenameOrSet,
                  prepStrsListOrFilename, effectStrsListOrFilename,
                  germsListOrFilename, maxLengths, gaugeOptParams=None,
                  advancedOptions=None, comm=None, memLimit=None,
                  output_pkl=None, verbosity=2):
    """
    Tests a GateSet model against a DataSet using a specific set of structured
    gate sequences (given by fiducials, maxLengths and germs).

    Constructs gate strings by repeating germ strings an integer number of
    times such that the length of the repeated germ is less than or equal to
    the maximum length set in maxLengths.  Each string thus constructed is
    sandwiched between all pairs of (prep, effect) fiducial sequences.

    `modelGateset` is used directly (without any optimization) as the
    the gate set estimate at each maximum-length "iteration".  The gate set
    is given a trivial `default_gauge_group` so that it is not altered
    during any gauge optimization step.

    A :class:`~pygsti.report.Results` object is returned, which encapsulates
    the gate set estimate and related parameters, and can be used with
    report-generation routines.
    Parameters
    ----------
    modelGateFilenameOrSet : GateSet or string
        The model gate set, specified either directly or by the filename of a
        gateset file (text format).

    dataFilenameOrSet : DataSet or string
        The data set object to use for the analysis, specified either directly
        or by the filename of a dataset file (assumed to be a pickled `DataSet`
        if extension is 'pkl' otherwise assumed to be in pyGSTi's text format).

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

    advancedOptions : dict, optional
        Specifies advanced options most of which deal with numerical details of
        the objective function or expert-level functionality.

    comm : mpi4py.MPI.Comm, optional
        When not ``None``, an MPI communicator for distributing the computation
        across multiple processors.

    memLimit : int or None, optional
        A rough memory limit in bytes which restricts the amount of memory
        used (per core when run on multi-CPUs).

    output_pkl : str or file, optional
        If not None, a file(name) to `pickle.dump` the returned `Results` object
        to (only the rank 0 process performs the dump when `comm` is not None).

    verbosity : int, optional
       The 'verbosity' option is an integer specifying the level of
       detail printed to stdout during the calculation.

    Returns
    -------
    Results
    """

    #Get/load target & model gatesets
    gs_model = _load_gateset(modelGateFilenameOrSet)
    gs_target = _load_gateset(targetGateFilenameOrSet)

    #Get/load fiducials and germs
    prepStrs, effectStrs, germs = _load_fiducials_and_germs(
        prepStrsListOrFilename,
        effectStrsListOrFilename,
        germsListOrFilename)

    #Get/load dataset
    ds = _load_dataset(dataFilenameOrSet, comm, verbosity)

    #Construct GateString lists
    lsgstLists = _get_lsgst_lists(ds, gs_target, prepStrs, effectStrs, germs,
                                  maxLengths, advancedOptions, verbosity)

    if gaugeOptParams is None: gaugeOptParams = {}
    if advancedOptions is None: advancedOptions = {}
    if advancedOptions.get('set trivial gauge group',True):
        gs_model = gs_model.copy()
        gs_model.default_gauge_group = _objs.TrivialGaugeGroup(gs_model.dim) #so no gauge opt is done
    gs_lsgst_list = [ gs_model ]*len(maxLengths)

    #    #Starting Point - compute on rank 0 and distribute
    #LGSTcompatibleGates = all([(isinstance(g,_objs.FullyParameterizedGate) or
    #                            isinstance(g,_objs.TPParameterizedGate))
    #                           for g in gs_target.gates.values()])
    #if isinstance(lsgstLists[0],_objs.LsGermsStructure) and LGSTcompatibleGates:
    #    startingPt = advancedOptions.get('starting point',"LGST")
    #else:
    #    startingPt = advancedOptions.get('starting point',"target")

    #Create profiler
    profile = advancedOptions.get('profile',1)
    if profile == 0: profiler = _DummyProfiler()
    elif profile == 1: profiler = _objs.Profiler(comm,False)
    elif profile == 2: profiler = _objs.Profiler(comm,True)
    else: raise ValueError("Invalid value for 'profile' argument (%s)"%profile)

    parameters = _collections.OrderedDict()
    parameters['objective'] = advancedOptions.get('objective','logl')
    if parameters['objective'] == 'logl':
        parameters['minProbClip'] = advancedOptions.get('minProbClip',1e-4)
        parameters['radius'] = advancedOptions.get('radius',1e-4)
    elif parameters['objective'] == 'chi2':
        parameters['minProbClipForWeighting'] = advancedOptions.get(
            'minProbClipForWeighting',1e-4)
    else:
        raise ValueError("Invalid objective: %s" % parameters['objective'])

    parameters['profiler'] = profiler
    parameters['gateLabelAliases'] = advancedOptions.get('gateLabelAliases',None)
    parameters['truncScheme'] = advancedOptions.get('truncScheme', "whole germ powers")
    parameters['weights'] = None

    #Set a different default for onBadFit: don't do anything
    if 'onBadFit' not in advancedOptions:
        advancedOptions['onBadFit'] = [] # empty list => 'do nothing'

    return _post_opt_processing('do_model_test', ds, gs_target, gs_model,
                                lsgstLists, parameters, None, gs_lsgst_list,
                                gaugeOptParams, advancedOptions, comm, memLimit,
                                output_pkl, verbosity, profiler)


def do_linear_gst(dataFilenameOrSet, targetGateFilenameOrSet,
                  prepStrsListOrFilename, effectStrsListOrFilename,
                  gaugeOptParams=None, advancedOptions=None, comm=None,
                  memLimit=None, output_pkl=None, verbosity=2):
    """
    Perform Linear Gate Set Tomography (LGST).

    This function differs from the lower level :function:`do_lgst` function
    in that it may perform a post-LGST gauge optimization and this routine
    returns a :class:`Results` object containing the LGST estimate.

    Overall, this is a high-level driver routine which can be used similarly
    to :function:`do_long_sequence_gst`  whereas `do_lgst` is a low-level
    routine used when building your own algorithms.


    Parameters
    ----------
    dataFilenameOrSet : DataSet or string
        The data set object to use for the analysis, specified either directly
        or by the filename of a dataset file (assumed to be a pickled `DataSet`
        if extension is 'pkl' otherwise assumed to be in pyGSTi's text format).

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

    gaugeOptParams : dict, optional
        A dictionary of arguments to :func:`gaugeopt_to_target`, specifying
        how the final gauge optimization should be performed.  The keys and
        values of this dictionary may correspond to any of the arguments
        of :func:`gaugeopt_to_target` *except* for the first `gateset`
        argument, which is specified internally.  The `targetGateset` argument,
        *can* be set, but is specified internally when it isn't.  If `None`,
        then the dictionary `{'itemWeights': {'gates':1.0, 'spam':0.001}}`
        is used.  If `False`, then then *no* gauge optimization is performed.

    advancedOptions : dict, optional
        Specifies advanced options most of which deal with numerical details of
        the objective function or expert-level functionality.  See
        :function:`do_long_sequence_gst`.

    comm : mpi4py.MPI.Comm, optional
        When not ``None``, an MPI communicator for distributing the computation
        across multiple processors.  In this LGST case, this is just the gauge
        optimization.

    memLimit : int or None, optional
        A rough memory limit in bytes which restricts the amount of memory
        used (per core when run on multi-CPUs).

    output_pkl : str or file, optional
        If not None, a file(name) to `pickle.dump` the returned `Results` object
        to (only the rank 0 process performs the dump when `comm` is not None).

    verbosity : int, optional
       The 'verbosity' option is an integer specifying the level of
       detail printed to stdout during the calculation.

    Returns
    -------
    Results
    """
    gs_target = _load_gateset(targetGateFilenameOrSet)
    germs = _construction.gatestring_list([()] + [(gl,) for gl in gs_target.gates.keys()]) # just the single gates
    maxLengths = [1] # we only need maxLength == 1 when doing LGST
    
    defAdvOptions = {'onBadFit': [], 'estimateLabel': 'LGST'}
    if advancedOptions is None: advancedOptions = {}
    advancedOptions.update(defAdvOptions)
    advancedOptions['objective'] = 'lgst' # not override-able

    return do_long_sequence_gst(dataFilenameOrSet, gs_target,
                                prepStrsListOrFilename,effectStrsListOrFilename,
                                germs, maxLengths, gaugeOptParams,
                                advancedOptions, comm, memLimit,
                                output_pkl, verbosity)


def do_long_sequence_gst(dataFilenameOrSet, targetGateFilenameOrSet,
                         prepStrsListOrFilename, effectStrsListOrFilename,
                         germsListOrFilename, maxLengths, gaugeOptParams=None,
                         advancedOptions=None, comm=None, memLimit=None,
                         output_pkl=None, verbosity=2):
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
        or by the filename of a dataset file (assumed to be a pickled `DataSet`
        if extension is 'pkl' otherwise assumed to be in pyGSTi's text format).

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

    advancedOptions : dict, optional
        Specifies advanced options most of which deal with numerical details of
        the objective function or expert-level functionality.  The allowed keys
        and values include:
        - objective = {'chi2', 'logl'}
        - gateLabels = list of strings
        - gsWeights = dict or None
        - starting point = "LGST" (default) or  "target" or GateSet
        - depolarizeStart = float (default == 0)
        - contractStartToCPTP = True / False (default)
        - cptpPenaltyFactor = float (default = 0)
        - tolerance = float or dict w/'relx','relf','f','jac' keys
        - maxIterations = int
        - minProbClip = float
        - minProbClipForWeighting = float (default == 1e-4)
        - probClipInterval = tuple (default == (-1e6,1e6)
        - radius = float (default == 1e-4)
        - useFreqWeightedChiSq = True / False (default)
        - nestedGateStringLists = True (default) / False
        - includeLGST = True / False (default is True)
        - distributeMethod = "default", "gatestrings" or "deriv"
        - profile = int (default == 1)
        - check = True / False (default)
        - gateLabelAliases = dict (default = None)
        - alwaysPerformMLE = bool (default = False)
        - truncScheme = "whole germ powers" (default) or "truncated germ powers"
                        or "length as exponent"
        - appendTo = Results (default = None)
        - estimateLabel = str (default = "default")
        - missingDataAction = {'drop','raise'} (default = 'drop')
        - stringManipRules = list of (find,replace) tuples
        - germLengthLimits = dict of form {germ: maxlength}
        - recordOutput = bool (default = True)

    comm : mpi4py.MPI.Comm, optional
        When not ``None``, an MPI communicator for distributing the computation
        across multiple processors.

    memLimit : int or None, optional
        A rough memory limit in bytes which restricts the amount of memory
        used (per core when run on multi-CPUs).

    output_pkl : str or file, optional
        If not None, a file(name) to `pickle.dump` the returned `Results` object
        to (only the rank 0 process performs the dump when `comm` is not None).

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

    #Now advanced options:
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

    printer = _objs.VerbosityPrinter.build_printer(verbosity, comm)
    if advancedOptions is None: advancedOptions = {}
    if advancedOptions.get('recordOutput',True) and not printer.is_recording():
        printer.start_recording()

    #Get/load target gateset
    gs_target = _load_gateset(targetGateFilenameOrSet)

    #Get/load fiducials and germs
    prepStrs, effectStrs, germs = _load_fiducials_and_germs(
        prepStrsListOrFilename,
        effectStrsListOrFilename,
        germsListOrFilename)

    #Get/load dataset
    ds = _load_dataset(dataFilenameOrSet, comm, printer)

    #Construct GateString lists
    lsgstLists = _get_lsgst_lists(ds, gs_target, prepStrs, effectStrs, germs,
                                  maxLengths, advancedOptions, printer)

    return do_long_sequence_gst_base(ds, gs_target, lsgstLists, gaugeOptParams,
                                     advancedOptions, comm, memLimit,
                                     output_pkl, printer)


def do_long_sequence_gst_base(dataFilenameOrSet, targetGateFilenameOrSet,
                              lsgstLists, gaugeOptParams=None,
                              advancedOptions=None, comm=None, memLimit=None,
                              output_pkl=None, verbosity=2):
    """
    A more fundamental interface for performing end-to-end GST.

    Similar to :func:`do_long_sequence_gst` except this function takes
    `lsgstLists`, a list of either raw gate string lists or of `LsGermsStruct`
    gate-string-structure objects to define which gate seqences are used on
    each GST iteration.

    Parameters
    ----------
    dataFilenameOrSet : DataSet or string
        The data set object to use for the analysis, specified either directly
        or by the filename of a dataset file (assumed to be a pickled `DataSet`
        if extension is 'pkl' otherwise assumed to be in pyGSTi's text format).

    targetGateFilenameOrSet : GateSet or string
        The target gate set, specified either directly or by the filename of a
        gateset file (text format).

    lsgstLists : list of lists or of LsGermsStructs
        An explicit list of either the raw gate string lists to be used in
        the analysis or of LsGermsStruct objects, which additionally contain
        the max-L, germ, and fiducial pair structure of a set of gate strings.

    gaugeOptParams : dict, optional
        A dictionary of arguments to :func:`gaugeopt_to_target`, specifying
        how the final gauge optimization should be performed.  The keys and
        values of this dictionary may correspond to any of the arguments
        of :func:`gaugeopt_to_target` *except* for the first `gateset`
        argument, which is specified internally.  The `targetGateset` argument,
        *can* be set, but is specified internally when it isn't.  If `None`,
        then the dictionary `{'itemWeights': {'gates':1.0, 'spam':0.001}}`
        is used.  If `False`, then then *no* gauge optimization is performed.

    advancedOptions : dict, optional
        Specifies advanced options most of which deal with numerical details of
        the objective function or expert-level functionality.  See
        :func:`do_long_sequence_gst` for a list of the allowed keys, with the
        exception  "nestedGateStringLists", "gateLabelAliases",
        "includeLGST", and "truncScheme".

    comm : mpi4py.MPI.Comm, optional
        When not ``None``, an MPI communicator for distributing the computation
        across multiple processors.

    memLimit : int or None, optional
        A rough memory limit in bytes which restricts the amount of memory
        used (per core when run on multi-CPUs).

    output_pkl : str or file, optional
        If not None, a file(name) to `pickle.dump` the returned `Results` object
        to (only the rank 0 process performs the dump when `comm` is not None).

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

    #Note: *don't* specify default dictionary arguments, as this is dangerous
    # because they are mutable objects
    if advancedOptions is None: advancedOptions = {}
    if gaugeOptParams is None:
        gaugeOptParams = {'itemWeights': {'gates':1.0, 'spam':0.001}}

    profile = advancedOptions.get('profile',1)

    if profile == 0: profiler = _DummyProfiler()
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
    if advancedOptions.get('recordOutput',True) and not printer.is_recording():
        printer.start_recording()


    #Get/load target gateset
    gs_target = _load_gateset(targetGateFilenameOrSet)

    #Get/load dataset
    ds = _load_dataset(dataFilenameOrSet, comm, printer)

    gate_dim = gs_target.get_dimension()

    tNxt = _time.time()
    profiler.add_time('do_long_sequence_gst: loading',tRef); tRef=tNxt


    #Starting Point - compute on rank 0 and distribute
    LGSTcompatibleGates = all([(isinstance(g,_objs.FullyParameterizedGate) or
                                isinstance(g,_objs.TPParameterizedGate))
                               for g in gs_target.gates.values()])
    if isinstance(lsgstLists[0],_objs.LsGermsStructure) and LGSTcompatibleGates:
        startingPt = advancedOptions.get('starting point',"LGST")
    else:
        startingPt = advancedOptions.get('starting point',"target")

    #Compute starting point
    if startingPt == "LGST":
        assert(isinstance(lsgstLists[0], _objs.LsGermsStructure)), \
               "Cannot run LGST: fiducials not specified!"
        gateLabels = advancedOptions.get('gateLabels',
                                         list(gs_target.gates.keys()) +
                                         list(gs_target.instruments.keys()) )
        gs_start = _alg.do_lgst(ds, lsgstLists[0].prepStrs, lsgstLists[0].effectStrs, gs_target,
                                gateLabels, svdTruncateTo=gate_dim,
                                gateLabelAliases=lsgstLists[0].aliases,
                                verbosity=printer) # returns a gateset with the *same*
                                                   # parameterizations as gs_target

        #In LGST case, gauge optimimize starting point to the target
        # (historical; sometimes seems to help in practice, since it's gauge
        # optimizing to physical gates (e.g. something in CPTP)
        tol = gaugeOptParams.get('tol',1e-8) if gaugeOptParams else 1e-8
        gs_start = _alg.gaugeopt_to_target(gs_start, gs_target, tol=tol, comm=comm)
          #Note: use *default* gauge-opt params when optimizing

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

    #Post-processing gs_start : done only on root proc in case there is any nondeterminism.
    if comm is None or comm.Get_rank() == 0:
        #Advanced Options can specify further manipulation of starting gate set
        if advancedOptions.get('contractStartToCPTP',False):
            gs_start = _alg.contract(gs_start, "CPTP")
        if advancedOptions.get('depolarizeStart',0) > 0:
            gs_start = gs_start.depolarize(gate_noise=advancedOptions.get('depolarizeStart',0))

        if comm is not None: #broadcast starting gate set
            comm.bcast(gs_start, root=0)
    else:
        gs_start = comm.bcast(None, root=0)

    tNxt = _time.time()
    profiler.add_time('do_long_sequence_gst: Prep Initial seed',tRef); tRef=tNxt

    # lsgstLists can hold either gatestring lists or structures - get
    # just the lists for calling core gst routines (structure is used only
    # for LGST and post-analysis).
    rawLists = [ l.allstrs if isinstance(l,_objs.LsGermsStructure) else l
                 for l in lsgstLists ]

    aliases = lsgstLists[-1].aliases if isinstance(
        lsgstLists[-1], _objs.LsGermsStructure) else None
    aliases = advancedOptions.get('gateLabelAliases',aliases)

    #Run Long-sequence GST on data
    objective = advancedOptions.get('objective', 'logl')

    args = dict(
        dataset=ds,
        startGateset=gs_start,
        gateStringSetsToUseInEstimation=rawLists,
        tol = advancedOptions.get('tolerance',1e-6),
        cptp_penalty_factor = advancedOptions.get('cptpPenaltyFactor',0),
        spam_penalty_factor = advancedOptions.get('spamPenaltyFactor',0),
        maxiter = advancedOptions.get('maxIterations',100000),
        probClipInterval = advancedOptions.get('probClipInterval',(-1e6,1e6)),
        returnAll=True,
        gatestringWeightsDict=advancedOptions.get('gsWeights',None),
        gateLabelAliases=aliases,
        verbosity=printer,
        memLimit=memLimit,
        profiler=profiler,
        comm=comm, distributeMethod=advancedOptions.get(
            'distributeMethod',"default"),
        check=advancedOptions.get('check',False),
        evaltree_cache={} )

    if objective == "chi2":
        args['useFreqWeightedChiSq'] = advancedOptions.get(
            'useFreqWeightedChiSq',False)
        args['minProbClipForWeighting'] = advancedOptions.get(
            'minProbClipForWeighting',1e-4)
        args['check_jacobian'] = advancedOptions.get('check',False)
        gs_lsgst_list = _alg.do_iterative_mc2gst(**args)

    elif objective == "logl":
        args['minProbClip'] = advancedOptions.get('minProbClip',1e-4)
        args['radius'] = advancedOptions.get('radius',1e-4)
        args['alwaysPerformMLE'] = advancedOptions.get('alwaysPerformMLE',False)
        gs_lsgst_list = _alg.do_iterative_mlgst(**args)
    elif objective == "lgst":
        assert(startingPt == "LGST"), "Can only set objective=\"lgst\" for parameterizations compatible with LGST"
        assert(len(lsgstLists) == 1), "Can only set objective=\"lgst\" with number if lists/max-lengths == 1"
        gs_lsgst_list = [args['startGateset']]
    else:
        raise ValueError("Invalid objective: %s" % objective)

    tNxt = _time.time()
    profiler.add_time('do_long_sequence_gst: total long-seq. opt.',tRef); tRef=tNxt

      #set parameters
    parameters = _collections.OrderedDict()
    parameters['objective'] = objective
    parameters['memLimit'] = memLimit
    parameters['starting point'] = startingPt
    parameters['profiler'] = profiler

      #from advanced options
    parameters['minProbClip'] = \
        advancedOptions.get('minProbClip',1e-4)
    parameters['minProbClipForWeighting'] = \
        advancedOptions.get('minProbClipForWeighting',1e-4)
    parameters['probClipInterval'] = \
        advancedOptions.get('probClipInterval',(-1e6,1e6))
    parameters['radius'] = advancedOptions.get('radius',1e-4)
    parameters['weights'] = advancedOptions.get('gsWeights',None)
    parameters['cptpPenaltyFactor'] = advancedOptions.get('cptpPenaltyFactor',0)
    parameters['spamPenaltyFactor'] = advancedOptions.get('spamPenaltyFactor',0)
    parameters['distributeMethod'] = advancedOptions.get('distributeMethod','default')
    parameters['depolarizeStart'] = advancedOptions.get('depolarizeStart',0)
    parameters['contractStartToCPTP'] = advancedOptions.get('contractStartToCPTP',False)
    parameters['tolerance'] = advancedOptions.get('tolerance',1e-6)
    parameters['maxIterations'] = advancedOptions.get('maxIterations',100000)
    parameters['useFreqWeightedChiSq'] = advancedOptions.get('useFreqWeightedChiSq',False)
    parameters['nestedGateStringLists'] = advancedOptions.get('nestedGateStringLists',True)
    parameters['profile'] = advancedOptions.get('profile',1)
    parameters['check'] = advancedOptions.get('check',False)
    parameters['truncScheme'] = advancedOptions.get('truncScheme', "whole germ powers")
    parameters['gateLabelAliases'] = advancedOptions.get('gateLabelAliases',None)
    parameters['includeLGST'] = advancedOptions.get('includeLGST', True)

    return _post_opt_processing('do_long_sequence_gst', ds, gs_target, gs_start,
                                lsgstLists, parameters, args, gs_lsgst_list,
                                gaugeOptParams, advancedOptions, comm, memLimit,
                                output_pkl, printer, profiler, args['evaltree_cache'])


def do_stdpractice_gst(dataFilenameOrSet,targetGateFilenameOrSet,
                       prepStrsListOrFilename, effectStrsListOrFilename,
                       germsListOrFilename, maxLengths, modes="TP,CPTP,Target",
                       gaugeOptSuite=('single','unreliable2Q'),
                       gaugeOptTarget=None, modelsToTest=None, comm=None, memLimit=None,
                       advancedOptions=None, output_pkl=None, verbosity=2):

    """
    Perform end-to-end GST analysis using standard practices.

    This routines is an even higher-level driver than
    :func:`do_long_sequence_gst`.  It performs bottled, typically-useful,
    runs of long sequence GST on a dataset.  This essentially boils down
    to running :func:`do_long_sequence_gst` one or more times using different
    gate set parameterizations, and performing commonly-useful gauge
    optimizations, based only on the high-level `modes` argument.

    Parameters
    ----------
    dataFilenameOrSet : DataSet or string
        The data set object to use for the analysis, specified either directly
        or by the filename of a dataset file (assumed to be a pickled `DataSet`
        if extension is 'pkl' otherwise assumed to be in pyGSTi's text format).

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

    modes : str, optional
        A comma-separated list of modes which dictate what types of analyses
        are performed.  Currently, these correspond to different types of
        parameterizations/constraints to apply to the estimated gate set.
        The default value is usually fine.  Allowed values are:

        - "full" : full (completely unconstrained)
        - "TP"   : TP-constrained
        - "CPTP" : Lindbladian CPTP-constrained
        - "H+S"  : Only Hamiltonian + Stochastic errors allowed (CPTP)
        - "S"    : Only Stochastic errors allowed (CPTP)
        - "Target" : use the target (ideal) gates as the estimate
        - <model> : any key in the `modelsToTest` argument

    gaugeOptSuite : str or list or dict, optional
        Specifies which gauge optimizations to perform on each estimate.  A
        string or list of strings (see below) specifies built-in sets of gauge
        optimizations, otherwise `gaugeOptSuite` should be a dictionary of
        gauge-optimization parameter dictionaries, as specified by the
        `gaugeOptParams` argument of :func:`do_long_sequence_gst`.  The key
        names of `gaugeOptSuite` then label the gauge optimizations within
        the resuling `Estimate` objects.  The built-in suites are:

          - "single" : performs only a single "best guess" gauge optimization.
          - "varySpam" : varies spam weight and toggles SPAM penalty (0 or 1).
          - "varySpamWt" : varies spam weight but no SPAM penalty.
          - "varyValidSpamWt" : varies spam weight with SPAM penalty == 1.
          - "toggleValidSpam" : toggles spame penalty (0 or 1); fixed SPAM wt.
          - "unreliable2Q" : adds branch to a spam suite that weights 2Q gates less
          - "none" : no gauge optimizations are performed.

    gaugeOptTarget : GateSet, optional
        If not None, a gate set to be used as the "target" for gauge-
        optimization (only).  This argument is useful when you want to
        gauge optimize toward something other than the *ideal* target gates
        given by `targetGateFilenameOrSet`, which are used as the default when
        `gaugeOptTarget` is None.

    modelsToTest : dict, optional
        A dictionary of GateSet objects representing (gate-set) models to
        test against the data.  These GateSets are essentially hypotheses for
        which (if any) model generated the data.  The keys of this dictionary
        can (and must, to actually test the models) be used within the comma-
        separate list given by the `modes` argument.

    comm : mpi4py.MPI.Comm, optional
        When not ``None``, an MPI communicator for distributing the computation
        across multiple processors.

    memLimit : int or None, optional
        A rough memory limit in bytes which restricts the amount of memory
        used (per core when run on multi-CPUs).

    advancedOptions : dict, optional
        Specifies advanced options most of which deal with numerical details of
        the objective function or expert-level functionality.  Keys of this
        dictionary can be any of the modes being computed (see the `modes`
        argument) or 'all', which applies to all modes.  Values are
        dictionaries of advanced arguements - see :func:`do_long_sequence_gst`
        for a list of the allowed keys for each such dictionary.

    output_pkl : str or file, optional
        If not None, a file(name) to `pickle.dump` the returned `Results` object
        to (only the rank 0 process performs the dump when `comm` is not None).

    verbosity : int, optional
       The 'verbosity' option is an integer specifying the level of
       detail printed to stdout during the calculation.

    Returns
    -------
    Results
    """
    printer = _objs.VerbosityPrinter.build_printer(verbosity, comm)
    if modelsToTest is None: modelsToTest = {}

    #Get/load target gateset
    gs_target = _load_gateset(targetGateFilenameOrSet)

    #Get/load fiducials and germs
    prepStrs, effectStrs, germs = _load_fiducials_and_germs(
        prepStrsListOrFilename,
        effectStrsListOrFilename,
        germsListOrFilename)

    #Get/load dataset
    ds = _load_dataset(dataFilenameOrSet, comm, printer)

    ret = None
    modes = modes.split(",")
    with printer.progress_logging(1):
        for i,mode in enumerate(modes):
            printer.show_progress(i, len(modes), prefix='-- Std Practice: ', suffix=' (%s) --' % mode)

            #prepare advanced options dictionary
            if advancedOptions is not None:
                advanced = advancedOptions.get('all',{})
                advanced.update( advancedOptions.get(mode,{}) )
            else: advanced = {}

            if mode == "Target":
                est_label = mode
                tgt = gs_target.copy() #no parameterization change
                tgt.default_gauge_group = _objs.TrivialGaugeGroup(tgt.dim) #so no gauge opt is done
                advanced.update( {'appendTo': ret, 'estimateLabel': est_label,
                                  'onBadFit': []} )
                ret = do_model_test(gs_target, ds, tgt, prepStrs,
                                    effectStrs, germs, maxLengths, False, advanced,
                                    comm, memLimit, None, printer-1)

            elif mode in ('full','TP','CPTP','H+S','S','static'): # mode is a parameterization
                est_label = parameterization = mode #for now, 1-1 correspondence
                tgt = gs_target.copy(); tgt.set_all_parameterizations(parameterization)
                advanced.update( {'appendTo': ret, 'estimateLabel': est_label } )
                ret = do_long_sequence_gst(ds, tgt, prepStrs, effectStrs, germs,
                                           maxLengths, False, advanced, comm, memLimit,
                                           None, printer-1)
            elif mode in modelsToTest:
                est_label = mode
                tgt = gs_target.copy() #no parameterization change
                tgt.default_gauge_group = _objs.TrivialGaugeGroup(tgt.dim) #so no gauge opt is done
                advanced.update( {'appendTo': ret, 'estimateLabel': est_label } )
                ret = do_model_test(modelsToTest[mode], ds, tgt, prepStrs,
                                    effectStrs, germs, maxLengths, False, advanced,
                                    comm, memLimit, None, printer-1)
            else:
                raise ValueError("Invalid item in 'modes' argument: %s" % mode)

            #Get gauge optimization dictionary
            assert(not printer.is_recording()); printer.start_recording()
            gaugeOptSuite_dict = gaugeopt_suite_to_dictionary(gaugeOptSuite, tgt,
                                                              advancedOptions, printer-1)

            if gaugeOptTarget is not None:
                assert(isinstance(gaugeOptTarget,_objs.GateSet)),"`gaugeOptTarget` must be None or a GateSet"
                for goparams in gaugeOptSuite_dict.values():
                    goparams_list = [goparams] if hasattr(goparams,'keys') else goparams
                    for goparams_dict in goparams_list:
                        if 'targetGateset' in goparams_dict:
                            _warnings.warn(("`gaugeOptTarget` argument is overriding"
                                            "user-defined targetGateset in gauge opt"
                                            "param dict(s)"))
                        goparams_dict.update( {'targetGateset': gaugeOptTarget } )

            #Gauge optimize to list of gauge optimization parameters
            for goLabel,goparams in gaugeOptSuite_dict.items():

                printer.log("-- Performing '%s' gauge optimization on %s estimate --" % (goLabel,est_label),2)
                gsStart = ret.estimates[est_label].get_start_gateset(goparams)
                ret.estimates[est_label].add_gaugeoptimized(goparams, None, goLabel, comm, printer-3)

                #Gauge optimize data-scaled estimate also
                for suffix in ROBUST_SUFFIX_LIST:
                    if est_label + suffix in ret.estimates:
                        gsStart_robust = ret.estimates[est_label+suffix].get_start_gateset(goparams)
                        if gsStart_robust.frobeniusdist(gsStart) < 1e-8:
                            printer.log("-- Conveying '%s' gauge optimization to %s estimate --" % (goLabel,est_label+suffix),2)
                            params = ret.estimates[est_label].goparameters[goLabel] #no need to copy here
                            gsopt = ret.estimates[est_label].gatesets[goLabel].copy()
                            ret.estimates[est_label + suffix].add_gaugeoptimized(params, gsopt, goLabel, comm, printer-3)
                        else:
                            printer.log("-- Performing '%s' gauge optimization on %s estimate --" % (goLabel,est_label+suffix),2)
                            ret.estimates[est_label + suffix].add_gaugeoptimized(goparams, None, goLabel, comm, printer-3)

            # Add gauge optimizations to end of any existing "stdout" meta info
            if 'stdout' in ret.estimates[est_label].meta:
                ret.estimates[est_label].meta['stdout'].extend(printer.stop_recording())
            else:
                ret.estimates[est_label].meta['stdout'] = printer.stop_recording()

    #Write results to a pickle file if desired
    if output_pkl and (comm is None or comm.Get_rank() == 0):
        if _compat.isstr(output_pkl):
            with open(output_pkl, 'wb') as pklfile:
                _pickle.dump(ret, pklfile)
        else:
            _pickle.dump(ret, output_pkl)

    return ret


def gaugeopt_suite_to_dictionary(gaugeOptSuite, gs_target, advancedOptions=None, verbosity=0):
    """
    Constructs a dictionary of gauge-optimization parameter dictionaries based
    on "gauge optimization suite" name(s).

    This is primarily a helper function for :func:`do_stdpractice_gst`, but can
    be useful in its own right for constructing the would-be gauge optimization
    dictionary used in :func:`do_stdpractice_gst` and modifying it slightly before
    before passing it in (`do_stdpractice_gst` will accept a raw dictionary too).

    Parameters
    ----------
    gaugeOptSuite : str or dict, optional
        Specifies which gauge optimizations to perform on each estimate.  An
        string (see below) specifies a built-in set of gauge optimizations,
        otherwise `gaugeOptSuite` should be a dictionary of gauge-optimization
        parameter dictionaries, as specified by the `gaugeOptParams` argument
        of :func:`do_long_sequence_gst`.  The key names of `gaugeOptSuite` then
        label the gauge optimizations within the resuling `Estimate` objects.
        The built-in gauge optmization suites are:

          - "single" : performs only a single "best guess" gauge optimization.
          - "varySpam" : varies spam weight and toggles SPAM penalty (0 or 1).
          - "varySpamWt" : varies spam weight but no SPAM penalty.
          - "varyValidSpamWt" : varies spam weight with SPAM penalty == 1.
          - "toggleValidSpam" : toggles spame penalty (0 or 1); fixed SPAM wt.
          - "unreliable2Q" : adds branch to a spam suite that weights 2Q gates less
          - "none" : no gauge optimizations are performed.

    gs_target : GateSet
        A gate set which specifies the dimension (i.e. parameterization) of the
        gauge-optimization and the basis.  Usually this is set to the *ideal*
        `target gate set` for the gate set being gauge optimized.

    advancedOptions : dict, optional
        A dictionary of advanced options for internal use.

    verbosity : int
        The verbosity to attach to the various gauge optimization parameter
        dictionaries.

    Returns
    -------
    dict
        A dictionary whose keys are the labels of the different gauge
        optimizations to perform and whose values are the corresponding
        dictionaries of arguments to :func:`gaugeopt_to_target` (or lists
        of such dictionaries for a multi-stage gauge optimization).
    """
    printer = _objs.VerbosityPrinter.build_printer(verbosity)

    #Build ordered dict of gauge optimization parameters
    if isinstance(gaugeOptSuite, dict):
        gaugeOptSuite_dict = _collections.OrderedDict()
        for lbl, goparams in gaugeOptSuite.items():
            if hasattr(goparams,'keys'):
                gaugeOptSuite_dict[lbl] = goparams.copy()
                gaugeOptSuite_dict[lbl].update( {'verbosity': printer } )
            else:
                assert(isinstance(goparams, list)), "If not a dictionary, gauge opt params should be a list of dicts!"
                gaugeOptSuite_dict[lbl] = []
                for goparams_stage in goparams:
                    dct = goparams_stage.copy()
                    dct.update( {'verbosity': printer } )
                    gaugeOptSuite_dict[lbl].append( dct )

    else:
        gaugeOptSuite_dict = _collections.OrderedDict()
        if _compat.isstr(gaugeOptSuite):
            gaugeOptSuites = [gaugeOptSuite]
        else:
            gaugeOptSuites = gaugeOptSuite[:] #assumes gaugeOptSuite is a list/tuple of strs

        for suiteName in gaugeOptSuites:
            if suiteName == "single":

                stages = [ ] #multi-stage gauge opt
                gg = gs_target.default_gauge_group
                if isinstance(gg, _objs.TrivialGaugeGroup):
                    #just do a single-stage "trivial" gauge opts using default group
                    gaugeOptSuite_dict['single'] =  { 'verbosity': printer }
                    if "unreliable2Q" in gaugeOptSuites and gs_target.dim == 16:
                        gaugeOptSuite_dict['single-2QUR'] = { 'verbosity': printer }

                elif gg is not None:

                    #Stage 1: plain vanilla gauge opt to get into "right ballpark"
                    if gg.name in ("Full", "TP"):
                        stages.append(
                            {
                                'itemWeights': {'gates': 1.0, 'spam': 1.0},
                                'verbosity': printer
                            })

                    #Stage 2: unitary gauge opt that tries to nail down gates (at
                    #         expense of spam if needed)
                    stages.append(
                        {
                            'itemWeights': {'gates': 1.0, 'spam': 0.0},
                            'gauge_group': _objs.UnitaryGaugeGroup(gs_target.dim, gs_target.basis),
                            'verbosity': printer
                        })

                    #Stage 3: spam gauge opt that fixes spam scaling at expense of
                    #         non-unital parts of gates (but shouldn't affect these
                    #         elements much since they should be small from Stage 2).
                    s3gg = _objs.SpamGaugeGroup if (gg.name == "Full") else \
                           _objs.TPSpamGaugeGroup
                    stages.append(
                        {
                            'itemWeights': {'gates': 0.0, 'spam': 1.0},
                            'spam_penalty_factor': 1.0,
                            'gauge_group': s3gg(gs_target.dim),
                            'verbosity': printer
                        })

                    gaugeOptSuite_dict['single'] = stages #can be a list of stage dictionaries

                    if "unreliable2Q" in gaugeOptSuites and gs_target.dim == 16:
                        if advancedOptions is not None:
                            advanced = advancedOptions.get('all',{}) #'unreliableGates' can only be specified in 'all' options
                        else: advanced = {}
                        unreliableGates = advanced.get('unreliableGates',['Gcnot','Gcphase','Gms','Gcn','Gcx','Gcz'])
                        if any([gl in gs_target.gates.keys() for gl in unreliableGates]):
                            stage2_item_weights = {'gates': 1, 'spam': 0.0}
                            for gl in unreliableGates:
                                if gl in gs_target.gates.keys(): stage2_item_weights[gl] = 0.01
                            stages_2QUR = [stage.copy() for stage in stages] # ~deep copy of stages
                            iStage2 = 1 if gg.name in ("Full", "TP") else 0
                            stages_2QUR[iStage2]['itemWeights'] = stage2_item_weights
                            gaugeOptSuite_dict['single-2QUR'] = stages_2QUR #add additional gauge opt

            elif suiteName in ("varySpam", "varySpamWt", "varyValidSpamWt", "toggleValidSpam"):

                itemWeights_bases = _collections.OrderedDict()
                itemWeights_bases[""] = {'gates': 1}

                if "unreliable2Q" in gaugeOptSuites and gs_target.dim == 16:
                    if advancedOptions is not None:
                        advanced = advancedOptions.get('all',{}) #'unreliableGates' can only be specified in 'all' options
                    else: advanced = {}
                    unreliableGates = advanced.get('unreliableGates',['Gcnot','Gcphase','Gms','Gcn','Gcx','Gcz'])
                    if any([gl in gs_target.gates.keys() for gl in unreliableGates]):
                        base = {'gates': 1}
                        for gl in unreliableGates:
                            if gl in gs_target.gates.keys(): base[gl] = 0.01
                        itemWeights_bases["-2QUR"] = base

                if suiteName == "varySpam":
                    vSpam_range = [0,1]; spamWt_range = [1e-4,1e-1]
                elif suiteName == "varySpamWt":
                    vSpam_range = [0]; spamWt_range = [1e-4,1e-1]
                elif suiteName == "varyValidSpamWt":
                    vSpam_range = [1]; spamWt_range = [1e-4,1e-1]
                elif suiteName == "toggleValidSpam":
                    vSpam_range = [0,1]; spamWt_range = [1e-3]

                for postfix,baseWts in itemWeights_bases.items():
                    for vSpam in vSpam_range:
                        for spamWt in spamWt_range:
                            lbl = "Spam %g%s%s" % (spamWt, "+v" if vSpam else "", postfix)
                            itemWeights = baseWts.copy()
                            itemWeights['spam'] = spamWt
                            gaugeOptSuite_dict[lbl] = {
                                'itemWeights': itemWeights,
                                'spam_penalty_factor': vSpam, 'verbosity': printer }

            elif suiteName == "unreliable2Q":
                assert( any([nm in ("single", "varySpam", "varySpamWt", "varyValidSpamWt", "toggleValidSpam")
                            for nm in gaugeOptSuite])), "'unreliable2Q' suite must be used with a spam or single suite."

            elif suiteName == "none":
                pass #add nothing

            else:
                raise ValueError("Invalid `gaugeOptSuite` argument - unknown suite '%s'" % suiteName)

    return gaugeOptSuite_dict



# ------------------ HELPER FUNCTIONS -----------------------------------

def _load_gateset(gatesetFilenameOrObj):
    if _compat.isstr(gatesetFilenameOrObj):
        return _io.load_gateset(gatesetFilenameOrObj)
    else:
        return gatesetFilenameOrObj #assume a GateSet object

def _load_fiducials_and_germs(prepStrsListOrFilename,
                              effectStrsListOrFilename,
                              germsListOrFilename):

    if _compat.isstr(prepStrsListOrFilename):
        prepStrs = _io.load_gatestring_list(prepStrsListOrFilename)
    else: prepStrs = prepStrsListOrFilename

    if effectStrsListOrFilename is None:
        effectStrs = prepStrs #use same strings for effectStrs if effectStrsListOrFilename is None
    else:
        if _compat.isstr(effectStrsListOrFilename):
            effectStrs = _io.load_gatestring_list(effectStrsListOrFilename)
        else: effectStrs = effectStrsListOrFilename

    #Get/load germs
    if _compat.isstr(germsListOrFilename):
        germs = _io.load_gatestring_list(germsListOrFilename)
    else: germs = germsListOrFilename

    return prepStrs, effectStrs, germs


def _load_dataset(dataFilenameOrSet, comm, verbosity):
    """Loads a DataSet from the dataFilenameOrSet argument of functions in this module."""
    printer = _objs.VerbosityPrinter.build_printer(verbosity, comm)
    if _compat.isstr(dataFilenameOrSet):
        if comm is None or comm.Get_rank() == 0:
            if _os.path.splitext(dataFilenameOrSet)[1] == ".pkl":
                with open(dataFilenameOrSet,'rb') as pklfile:
                    ds = _pickle.load(pklfile)
            else:
                ds = _io.load_dataset(dataFilenameOrSet, True, "aggregate", printer)
            if comm is not None: comm.bcast(ds, root=0)
        else:
            ds = comm.bcast(None, root=0)
    else:
        ds = dataFilenameOrSet #assume a Dataset object

    return ds


def _get_lsgst_lists(dschk, gs_target, prepStrs, effectStrs, germs,
                     maxLengths, advancedOptions, verbosity):
    """
    Sequence construction logic, fatctored into this separate
    function because it's shared do_long_sequence_gst and
    do_model_evaluation.
    """
    if advancedOptions is None: advancedOptions = {}

    #Update: now always include LGST strings unless advanced options says otherwise
    #Get starting point (so we know whether to include LGST strings)
    #LGSTcompatibleGates = all([(isinstance(g,_objs.FullyParameterizedGate) or
    #                            isinstance(g,_objs.TPParameterizedGate))
    #                           for g in gs_target.gates.values()])
    #if  LGSTcompatibleGates:
    #    startingPt = advancedOptions.get('starting point',"LGST")
    #else:
    #    startingPt = advancedOptions.get('starting point',"target")

    #Construct gate sequences
    actionIfMissing = advancedOptions.get('missingDataAction','drop')
    gateLabels = advancedOptions.get(
        'gateLabels', list(gs_target.gates.keys()))
    lsgstLists = _construction.stdlists.make_lsgst_structs(
        gateLabels, prepStrs, effectStrs, germs, maxLengths,
        truncScheme = advancedOptions.get('truncScheme',"whole germ powers"),
        nest = advancedOptions.get('nestedGateStringLists',True),
        includeLGST = advancedOptions.get('includeLGST', True),
        gateLabelAliases = advancedOptions.get('gateLabelAliases',None),
        sequenceRules = advancedOptions.get('stringManipRules',None),
        dscheck=dschk, actionIfMissing=actionIfMissing,
        germLengthLimits=advancedOptions.get('germLengthLimits',None),
        verbosity=verbosity)
    assert(len(maxLengths) == len(lsgstLists))

    return lsgstLists

def _post_opt_processing(callerName, ds, gs_target, gs_start, lsgstLists,
                         parameters, opt_args, gs_lsgst_list, gaugeOptParams,
                         advancedOptions, comm, memLimit, output_pkl, verbosity,
                         profiler, evaltree_cache=None):
    """
    Performs all of the post-optimization processing common to
    do_long_sequence_gst and do_model_evaluation.

    Creates a Results object to be returned from do_long_sequence_gst
    and do_model_evaluation (passed in as 'callerName').  Performs
    gauge optimization, and robust data scaling (with re-optimization
    if needed and opt_args is not None - i.e. only for
    do_long_sequence_gst).
    """
    printer = _objs.VerbosityPrinter.build_printer(verbosity, comm)
    if advancedOptions is None: advancedOptions = {}
    tRef = _time.time()

    ret = advancedOptions.get('appendTo',None)
    if ret is None:
        ret = _objs.Results()
        ret.init_dataset(ds)
        ret.init_gatestrings(lsgstLists)
    else:
        assert(ret.dataset is ds), "DataSet inconsistency: cannot append!"
        assert(len(lsgstLists) == len(ret.gatestring_structs['iteration'])), \
            "Iteration count inconsistency: cannot append!"
        for i,(a,b) in enumerate(zip(lsgstLists,ret.gatestring_structs['iteration'])):
            assert(len(a.allstrs)==len(b.allstrs)), \
                "Iteration %d should have %d of sequences but has %d" % (i,len(b.allstrs),len(a.allstrs))
            assert(set(a.allstrs)==set(b.allstrs)), \
                "Iteration %d: sequences don't match existing Results object!" % i

    #add estimate to Results
    estlbl = advancedOptions.get('estimateLabel','default')
    ret.add_estimate(gs_target, gs_start, gs_lsgst_list, parameters, estlbl)

    #Do final gauge optimization to *final* iteration result only
    if gaugeOptParams != False:
        gaugeOptParams = gaugeOptParams.copy() #so we don't modify the caller's dict
        if "targetGateset" not in gaugeOptParams:
            gaugeOptParams["targetGateset"] = gs_target

        # somewhat redundant given add_gaugeoptimized behavior - but
        #  if not here won't do parallel gaugeopt_to_target below
        if "comm" not in gaugeOptParams:
            gaugeOptParams["comm"] = comm

        gaugeOptParams['returnAll'] = True # so we get gaugeEl to save
        gaugeOptParams['gateset'] = gs_lsgst_list[-1] #starting gate set
        _, gaugeEl, go_gs_final = _alg.gaugeopt_to_target(**gaugeOptParams)
        gaugeOptParams['_gaugeGroupEl'] = gaugeEl #store gaugeopt el
        ret.estimates[estlbl].add_gaugeoptimized(gaugeOptParams, go_gs_final,
                                                 None, comm, printer-1)        

        tNxt = _time.time()
        profiler.add_time('%s: gauge optimization' % callerName,tRef); tRef=tNxt

    #Perform extra analysis if a bad fit was obtained
    rawLists = [ l.allstrs if isinstance(l,_objs.LsGermsStructure) else l
                 for l in lsgstLists ]
    objective = advancedOptions.get('objective', 'logl')
    badFitThreshold = advancedOptions.get('badFitThreshold',DEFAULT_BAD_FIT_THRESHOLD)
    if ret.estimates[estlbl].misfit_sigma(evaltree_cache=evaltree_cache, comm=comm) > badFitThreshold:
        onBadFit = advancedOptions.get('onBadFit',["Robust+"]) # empty list => 'do nothing'
        
        if len(onBadFit) > 0 and parameters.get('weights',None) is None:

            # Get by-sequence goodness of fit
            if objective == "chi2":
                fitQty = _tools.chi2_terms(gs_lsgst_list[-1], ds, rawLists[-1],
                                           advancedOptions.get('minProbClipForWeighting',1e-4),
                                           advancedOptions.get('probClipInterval',(-1e6,1e6)),
                                           False, False, memLimit,
                                           advancedOptions.get('gateLabelAliases',None),
                                           evaltree_cache=evaltree_cache, comm=comm)
            else: # "logl" or "lgst"
                maxLogL = _tools.logl_max_terms(gs_lsgst_list[-1], ds, rawLists[-1],
                                                gateLabelAliases=advancedOptions.get(
                                                    'gateLabelAliases',None),
                                                evaltree_cache=evaltree_cache)

                logL = _tools.logl_terms(gs_lsgst_list[-1], ds, rawLists[-1],
                                         advancedOptions.get('minProbClip',1e-4),
                                         advancedOptions.get('probClipInterval',(-1e6,1e6)),
                                         advancedOptions.get('radius',1e-4),
                                         gateLabelAliases=advancedOptions.get('gateLabelAliases',None),
                                         evaltree_cache=evaltree_cache, comm=comm)
                fitQty = 2*(maxLogL - logL)

            #Note: fitQty[iGateString] gives fit quantity for a single gate
            # string, aggregated over outcomes.
            expected = (len(ds.get_outcome_labels())-1) # == "k"
            dof_per_box = expected; nboxes = len(rawLists[-1])
            pc = 0.95 #hardcoded confidence level for now -- make into advanced option w/default

            for scale_typ in onBadFit:
                gsWeights = {}
                if scale_typ in ("robust","Robust"):
                    # Robust scaling V1: drastically scale down weights of especially bad sequences
                    threshold = _np.ceil(_chi2.ppf(1 - pc/nboxes, dof_per_box))
                    for i,gstr in enumerate(rawLists[-1]):
                        if fitQty[i] > threshold:
                            gsWeights[gstr] = expected/fitQty[i] #scaling factor
                    reopt = bool(scale_typ == "Robust")

                elif scale_typ in ("robust+","Robust+"):
                    # Robust scaling V2: V1 + rescale to desired chi2 distribution without reordering
                    threshold = _np.ceil(_chi2.ppf(1 - pc/nboxes, dof_per_box))
                    scaled_fitQty = fitQty.copy()
                    for i,gstr in enumerate(rawLists[-1]):
                        if fitQty[i] > threshold:
                            gsWeights[gstr] = expected/fitQty[i] #scaling factor
                            scaled_fitQty[i] = expected # (fitQty[i]*gsWeights[gstr])

                    N = len(fitQty)
                    percentiles = [ _chi2.ppf((i+1)/(N+1), dof_per_box) for i in range(N) ]
                    for iBin,i in enumerate(_np.argsort(scaled_fitQty)):
                        gstr = rawLists[-1][i]
                        fit, expected = scaled_fitQty[i], percentiles[iBin]
                        if fit > expected:
                            if gstr in gsWeights: gsWeights[gstr] *= expected/fit
                            else: gsWeights[gstr] = expected/fit

                    reopt = bool(scale_typ == "Robust+")

                elif scale_typ == "do nothing":
                    continue #go to next on-bad-fit directive
                else:
                    raise ValueError("Invalid on-bad-fit directive: %s" % scale_typ)

                tNxt = _time.time()
                profiler.add_time('%s: robust data scaling init' % callerName,tRef); tRef=tNxt

                scale_params = parameters.copy()
                scale_params['weights'] = gsWeights

                if reopt and (opt_args is not None):
                    #convert weights dict to an array for do_XXX methods below
                    gsWeightsArray = _np.ones( len(rawLists[-1]), 'd')
                    gsindx = { gatestr:i for i,gatestr in enumerate(rawLists[-1]) }
                    for gatestr, weight in gsWeights.items():
                        gsWeightsArray[ gsindx[gatestr] ] = weight
                    
                    reopt_args = dict(dataset=ds,
                                      startGateset=gs_lsgst_list[-1],
                                      gateStringsToUse=rawLists[-1],
                                      gatestringWeights=gsWeightsArray,
                                      verbosity=printer-1)
                    for x in ('maxiter', 'tol', 'cptp_penalty_factor', 'spam_penalty_factor',
                              'probClipInterval', 'check', 'gateLabelAliases',
                              'memLimit', 'comm', 'evaltree_cache', 'distributeMethod', 'profiler'):
                        reopt_args[x] = opt_args[x]

                    printer.log("--- Re-optimizing %s after robust data scaling ---" % objective)
                    if objective == "chi2":
                        reopt_args['useFreqWeightedChiSq'] = opt_args['useFreqWeightedChiSq']
                        reopt_args['minProbClipForWeighting'] = opt_args['minProbClipForWeighting']
                        reopt_args['check_jacobian'] = opt_args['check_jacobian']
                        _, gs_reopt = _alg.do_mc2gst(**reopt_args)

                    elif objective == "logl":
                        reopt_args['minProbClip'] = opt_args['minProbClip']
                        reopt_args['radius'] = opt_args['radius']
                        _, gs_reopt = _alg.do_mlgst(**reopt_args)

                    else: raise ValueError("Invalid objective '%s' for robust data scaling reopt" % objective)

                    tNxt = _time.time()
                    profiler.add_time('%s: robust data scaling re-opt' % callerName,tRef); tRef=tNxt

                    # Re-run final iteration of GST with weights computed above,
                    # and just keep (?) old estimates of all prior iterations (or use "blank"
                    # sentinel once this is supported).
                    ret.add_estimate(gs_target, gs_start, gs_lsgst_list[0:-1] + [gs_reopt],
                                     scale_params, estlbl+"."+scale_typ)
                else:
                    ret.add_estimate(gs_target, gs_start, gs_lsgst_list,
                                     scale_params, estlbl+"."+scale_typ)

                #Do final gauge optimization to data-scaled estimate also
                if gaugeOptParams != False:
                    if reopt and (opt_args is not None):
                        gaugeOptParams = gaugeOptParams.copy() #so we don't modify the caller's dict
                        if '_gaugeGroupEl' in gaugeOptParams: del gaugeOptParams['_gaugeGroupEl']

                        if "targetGateset" not in gaugeOptParams:
                            gaugeOptParams["targetGateset"] = gs_target
                        if "comm" not in gaugeOptParams:
                            gaugeOptParams["comm"] = comm

                        gaugeOptParams['returnAll'] = True # so we get gaugeEl to save
                        gaugeOptParams['gateset'] = gs_reopt #starting gate set
                        _, gaugeEl, go_gs_reopt = _alg.gaugeopt_to_target(**gaugeOptParams)
                        gaugeOptParams['_gaugeGroupEl'] = gaugeEl #store gaugeopt el

                        tNxt = _time.time()
                        profiler.add_time('%s: robust data scaling gauge-opt' % callerName,tRef); tRef=tNxt

                        # add new gauge-re-optimized result as above
                        ret.estimates[estlbl+'.'+scale_typ].add_gaugeoptimized(
                            gaugeOptParams, go_gs_reopt, None, comm, printer-1)
                    else:
                        # add same gauge-optimized result as above
                        ret.estimates[estlbl+'.'+scale_typ].add_gaugeoptimized(
                            gaugeOptParams.copy(), go_gs_final, None, comm, printer-1)


    profiler.add_time('%s: results initialization' % callerName,tRef)

    #Add recorded info (even robust-related info) to the *base*
    #   estimate label's "stdout" meta information
    if printer.is_recording():
        ret.estimates[estlbl].meta['stdout'] = printer.stop_recording()

    #Write results to a pickle file if desired
    if output_pkl and (comm is None or comm.Get_rank() == 0):
        if _compat.isstr(output_pkl):
            with open(output_pkl, 'wb') as pklfile:
                _pickle.dump(ret, pklfile)
        else:
            _pickle.dump(ret, output_pkl)

    return ret





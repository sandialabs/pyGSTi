""" End-to-end functions for performing long-sequence GST """
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

import os as _os
import warnings as _warnings
import numpy as _np
import time as _time
import collections as _collections
import pickle as _pickle
import scipy.optimize as _spo
from scipy.stats import chi2 as _chi2

from .. import protocols as _proto
from .. import algorithms as _alg
from .. import construction as _construction
from .. import objects as _objs
from .. import io as _io
from .. import tools as _tools
from ..objects import wildcardbudget as _wild
from ..objects.profiler import DummyProfiler as _DummyProfiler
from ..objects import objectivefns as _objfns

ROBUST_SUFFIX_LIST = [".robust", ".Robust", ".robust+", ".Robust+"]
DEFAULT_BAD_FIT_THRESHOLD = 2.0


def do_model_test(modelFilenameOrObj,
                  dataFilenameOrSet, targetModelFilenameOrObj,
                  prepStrsListOrFilename, effectStrsListOrFilename,
                  germsListOrFilename, maxLengths, gaugeOptParams=None,
                  advancedOptions=None, comm=None, memLimit=None,
                  output_pkl=None, verbosity=2):
    """
    Tests a Model model against a DataSet using a specific set of structured
    operation sequences (given by fiducials, maxLengths and germs).

    Constructs operation sequences by repeating germ strings an integer number of
    times such that the length of the repeated germ is less than or equal to
    the maximum length set in maxLengths.  Each string thus constructed is
    sandwiched between all pairs of (prep, effect) fiducial sequences.

    `themodel` is used directly (without any optimization) as the
    the model estimate at each maximum-length "iteration".  The model
    is given a trivial `default_gauge_group` so that it is not altered
    during any gauge optimization step.

    A :class:`~pygsti.report.Results` object is returned, which encapsulates
    the model estimate and related parameters, and can be used with
    report-generation routines.
    Parameters
    ----------
    modelFilenameOrObj : Model or string
        The model model, specified either directly or by the filename of a
        model file (text format).

    dataFilenameOrSet : DataSet or string
        The data set object to use for the analysis, specified either directly
        or by the filename of a dataset file (assumed to be a pickled `DataSet`
        if extension is 'pkl' otherwise assumed to be in pyGSTi's text format).

    targetModelFilenameOrObj : Model or string
        The target model, specified either directly or by the filename of a
        model file (text format).

    prepStrsListOrFilename : (list of Circuits) or string
        The state preparation fiducial operation sequences, specified either directly
        or by the filename of a operation sequence list file (text format).

    effectStrsListOrFilename : (list of Circuits) or string or None
        The measurement fiducial operation sequences, specified either directly or by
        the filename of a operation sequence list file (text format).  If ``None``,
        then use the same strings as specified by prepStrsListOrFilename.

    germsListOrFilename : (list of Circuits) or string
        The germ operation sequences, specified either directly or by the filename of a
        operation sequence list file (text format).

    maxLengths : list of ints
        List of integers, one per LSGST iteration, which set truncation lengths
        for repeated germ strings.  The list of operation sequences for the i-th LSGST
        iteration includes the repeated germs truncated to the L-values *up to*
        and including the i-th one.

    gaugeOptParams : dict, optional
        A dictionary of arguments to :func:`gaugeopt_to_target`, specifying
        how the final gauge optimization should be performed.  The keys and
        values of this dictionary may correspond to any of the arguments
        of :func:`gaugeopt_to_target` *except* for the first `model`
        argument, which is specified internally.  The `targetModel` argument,
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
    printer = _objs.VerbosityPrinter.build_printer(verbosity, comm)
    ds = _load_dataset(dataFilenameOrSet, comm, printer)
    advancedOptions = advancedOptions or {}

    exp_design = _proto.StandardGSTDesign(targetModelFilenameOrObj,
                                          prepStrsListOrFilename, effectStrsListOrFilename,
                                          germsListOrFilename, maxLengths,
                                          advancedOptions.get('germLengthLimits', None),
                                          None, 1, None,  # fidPairs, keepFraction, keepSeed
                                          advancedOptions.get('includeLGST', True),
                                          advancedOptions.get('nestedCircuitLists', True),
                                          advancedOptions.get('stringManipRules', None),
                                          advancedOptions.get('opLabelAliases', None),
                                          ds, 'drop', verbosity=printer)
    # Note: no advancedOptions['truncScheme'] support anymore

    data = _proto.ProtocolData(exp_design, ds)

    gopt_suite = {'go0': gaugeOptParams} if gaugeOptParams else None
    builder = _objfns.ObjectiveFunctionBuilder.simple(advancedOptions.get('objective', 'logl'),
                                                      advancedOptions.get('useFreqWeightedChiSq', False))
    _update_objfn_builders([builder], advancedOptions)

    #Create the protocol
    proto = _proto.ModelTest(_load_model(modelFilenameOrObj), None, gopt_suite, None,
                             builder, _get_badfit_options(advancedOptions),
                             advancedOptions.get('set trivial gauge group', True), printer)

    #Set more advanced options
    proto.profile = advancedOptions.get('profile', 1)
    proto.oplabel_aliases = advancedOptions.get('opLabelAliases', None)
    proto.circuit_weights = advancedOptions.get('circuitWeights', None)
    proto.unreliable_ops = advancedOptions.get('unreliableOps', ['Gcnot', 'Gcphase', 'Gms', 'Gcn', 'Gcx', 'Gcz'])

    results = proto.run(data, memLimit, comm)
    _output_to_pickle(results, output_pkl, comm)
    return results


def do_linear_gst(dataFilenameOrSet, targetModelFilenameOrObj,
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

    targetModelFilenameOrObj : Model or string
        The target model, specified either directly or by the filename of a
        model file (text format).

    prepStrsListOrFilename : (list of Circuits) or string
        The state preparation fiducial operation sequences, specified either directly
        or by the filename of a operation sequence list file (text format).

    effectStrsListOrFilename : (list of Circuits) or string or None
        The measurement fiducial operation sequences, specified either directly or by
        the filename of a operation sequence list file (text format).  If ``None``,
        then use the same strings as specified by prepStrsListOrFilename.

    gaugeOptParams : dict, optional
        A dictionary of arguments to :func:`gaugeopt_to_target`, specifying
        how the final gauge optimization should be performed.  The keys and
        values of this dictionary may correspond to any of the arguments
        of :func:`gaugeopt_to_target` *except* for the first `model`
        argument, which is specified internally.  The `targetModel` argument,
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
    printer = _objs.VerbosityPrinter.build_printer(verbosity, comm)
    advancedOptions = advancedOptions or {}
    ds = _load_dataset(dataFilenameOrSet, comm, printer)

    target_model = _load_model(targetModelFilenameOrObj)
    germs = _construction.circuit_list([()] + [(gl,) for gl in target_model.operations.keys()])  # just the single gates
    maxLengths = [1]  # we only need maxLength == 1 when doing LGST

    exp_design = _proto.StandardGSTDesign(target_model, prepStrsListOrFilename, effectStrsListOrFilename,
                                          germs, maxLengths,
                                          sequenceRules=advancedOptions.get('stringManipRules', None),
                                          opLabelAliases=advancedOptions.get('opLabelAliases', None),
                                          dscheck=ds, actionIfMissing='raise', verbosity=printer)

    data = _proto.ProtocolData(exp_design, ds)

    if gaugeOptParams is None:
        gaugeOptParams = {'itemWeights': {'gates': 1.0, 'spam': 0.001}}
    gopt_suite = {'go0': gaugeOptParams} if gaugeOptParams else None

    proto = _proto.LinearGateSetTomography(target_model, gopt_suite, None,
                                           _get_badfit_options(advancedOptions), printer)
    proto.profile = advancedOptions.get('profile', 1)
    proto.record_output = advancedOptions.get('recordOutput', 1)
    proto.oplabels = advancedOptions.get('opLabels', 'default')
    proto.oplabel_aliases = advancedOptions.get('opLabelAliases', None)
    proto.unreliable_ops = advancedOptions.get('unreliableOps', ['Gcnot', 'Gcphase', 'Gms', 'Gcn', 'Gcx', 'Gcz'])

    results = proto.run(data, memLimit, comm)
    _output_to_pickle(results, output_pkl, comm)
    return results


def do_long_sequence_gst(dataFilenameOrSet, targetModelFilenameOrObj,
                         prepStrsListOrFilename, effectStrsListOrFilename,
                         germsListOrFilename, maxLengths, gaugeOptParams=None,
                         advancedOptions=None, comm=None, memLimit=None,
                         output_pkl=None, verbosity=2):
    """
    Perform end-to-end GST analysis using Ls and germs, with L as a maximum
    length.

    Constructs operation sequences by repeating germ strings an integer number of
    times such that the length of the repeated germ is less than or equal to
    the maximum length set in maxLengths.  The LGST estimate of the gates is
    computed, gauge optimized, and then used as the seed for either LSGST or
    MLEGST.
    LSGST is iterated ``len(maxLengths)`` times with successively larger sets
    of operation sequences.  On the i-th iteration, the repeated germs sequences
    limited by ``maxLengths[i]`` are included in the growing set of strings
    used by LSGST.  The final iteration will use MLEGST when ``objective ==
    "logl"`` to maximize the true log-likelihood instead of minimizing the
    chi-squared function.
    Once computed, the model estimates are optionally gauge optimized to
    the CPTP space and then to the target model (using `gaugeOptRatio`
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

    targetModelFilenameOrObj : Model or string
        The target model, specified either directly or by the filename of a
        model file (text format).

    prepStrsListOrFilename : (list of Circuits) or string
        The state preparation fiducial operation sequences, specified either directly
        or by the filename of a operation sequence list file (text format).

    effectStrsListOrFilename : (list of Circuits) or string or None
        The measurement fiducial operation sequences, specified either directly or by
        the filename of a operation sequence list file (text format).  If ``None``,
        then use the same strings as specified by prepStrsListOrFilename.

    germsListOrFilename : (list of Circuits) or string
        The germ operation sequences, specified either directly or by the filename of a
        operation sequence list file (text format).

    maxLengths : list of ints
        List of integers, one per LSGST iteration, which set truncation lengths
        for repeated germ strings.  The list of operation sequences for the i-th LSGST
        iteration includes the repeated germs truncated to the L-values *up to*
        and including the i-th one.

    gaugeOptParams : dict, optional
        A dictionary of arguments to :func:`gaugeopt_to_target`, specifying
        how the final gauge optimization should be performed.  The keys and
        values of this dictionary may correspond to any of the arguments
        of :func:`gaugeopt_to_target` *except* for the first `model`
        argument, which is specified internally.  The `targetModel` argument,
        *can* be set, but is specified internally when it isn't.  If `None`,
        then the dictionary `{'itemWeights': {'gates':1.0, 'spam':0.001}}`
        is used.  If `False`, then then *no* gauge optimization is performed.

    advancedOptions : dict, optional
        Specifies advanced options most of which deal with numerical details of
        the objective function or expert-level functionality.  The allowed keys
        and values include:
        - objective = {'chi2', 'logl'}
        - opLabels = list of strings
        - circuitWeights = dict or None
        - starting point = "LGST-if-possible" (default), "LGST", or "target"
        - depolarizeStart = float (default == 0)
        - randomizeStart = float (default == 0)
        - contractStartToCPTP = True / False (default)
        - cptpPenaltyFactor = float (default = 0)
        - tolerance = float or dict w/'relx','relf','f','jac','maxdx' keys
        - maxIterations = int
        - fdIterations = int
        - minProbClip = float
        - minProbClipForWeighting = float (default == 1e-4)
        - probClipInterval = tuple (default == (-1e6,1e6)
        - radius = float (default == 1e-4)
        - useFreqWeightedChiSq = True / False (default)
        - XX nestedCircuitLists = True (default) / False
        - XX includeLGST = True / False (default is True)
        - distributeMethod = "default", "circuits" or "deriv"
        - profile = int (default == 1)
        - check = True / False (default)
        - XX opLabelAliases = dict (default = None)
        - alwaysPerformMLE = bool (default = False)
        - onlyPerformMLE = bool (default = False)
        - XX truncScheme = "whole germ powers" (default) or "truncated germ powers"
                          or "length as exponent"
        - appendTo = Results (default = None)
        - estimateLabel = str (default = "default")
        - XX missingDataAction = {'drop','raise'} (default = 'drop')
        - XX stringManipRules = list of (find,replace) tuples
        - germLengthLimits = dict of form {germ: maxlength}
        - recordOutput = bool (default = True)
        - timeDependent = bool (default = False)

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
    printer = _objs.VerbosityPrinter.build_printer(verbosity, comm)
    advancedOptions = advancedOptions or {}
    ds = _load_dataset(dataFilenameOrSet, comm, printer)

    exp_design = _proto.StandardGSTDesign(targetModelFilenameOrObj,
                                          prepStrsListOrFilename, effectStrsListOrFilename,
                                          germsListOrFilename, maxLengths,
                                          advancedOptions.get('germLengthLimits', None),
                                          None, 1, None,  # fidPairs, keepFraction, keepSeed
                                          advancedOptions.get('includeLGST', True),
                                          advancedOptions.get('nestedCircuitLists', True),
                                          advancedOptions.get('stringManipRules', None),
                                          advancedOptions.get('opLabelAliases', None),
                                          ds, 'drop', verbosity=printer)

    data = _proto.ProtocolData(exp_design, ds)

    if gaugeOptParams is None:
        gaugeOptParams = {'itemWeights': {'gates': 1.0, 'spam': 0.001}}
    gopt_suite = {'go0': gaugeOptParams} if gaugeOptParams else None
    proto = _proto.GateSetTomography(_get_gst_initial_model(advancedOptions), gopt_suite, None,
                                     _get_gst_builders(advancedOptions),
                                     _get_optimizer(advancedOptions, exp_design),
                                     _get_badfit_options(advancedOptions), printer)

    proto.profile = advancedOptions.get('profile', 1)
    proto.record_output = advancedOptions.get('recordOutput', 1)
    proto.distribute_method = advancedOptions.get('distributeMethod', "default")
    proto.oplabel_aliases = advancedOptions.get('opLabelAliases', None)
    proto.circuit_weights = advancedOptions.get('circuitWeights', None)
    proto.unreliable_ops = advancedOptions.get('unreliableOps', ['Gcnot', 'Gcphase', 'Gms', 'Gcn', 'Gcx', 'Gcz'])

    results = proto.run(data, memLimit, comm)
    _output_to_pickle(results, output_pkl, comm)
    return results


def do_long_sequence_gst_base(dataFilenameOrSet, targetModelFilenameOrObj,
                              lsgstLists, gaugeOptParams=None,
                              advancedOptions=None, comm=None, memLimit=None,
                              output_pkl=None, verbosity=2):
    """
    A more fundamental interface for performing end-to-end GST.

    Similar to :func:`do_long_sequence_gst` except this function takes
    `lsgstLists`, a list of either raw operation sequence lists or of `LsGermsStruct`
    gate-string-structure objects to define which gate seqences are used on
    each GST iteration.

    Parameters
    ----------
    dataFilenameOrSet : DataSet or string
        The data set object to use for the analysis, specified either directly
        or by the filename of a dataset file (assumed to be a pickled `DataSet`
        if extension is 'pkl' otherwise assumed to be in pyGSTi's text format).

    targetModelFilenameOrObj : Model or string
        The target model, specified either directly or by the filename of a
        model file (text format).

    lsgstLists : list of lists or LsGermsStruct(s)
        An explicit list of either the raw operation sequence lists to be used in
        the analysis or of LsGermsStruct objects, which additionally contain
        the max-L, germ, and fiducial pair structure of a set of operation sequences.
        A single LsGermsStruct object can also be given, which is equivalent
        to passing a list of successive L-value truncations of this object
        (e.g. if the object has `Ls = [1,2,4]` then this is like passing
         a list of three LsGermsStructs w/truncations `[1]`, `[1,2]`, and
         `[1,2,4]`).

    gaugeOptParams : dict, optional
        A dictionary of arguments to :func:`gaugeopt_to_target`, specifying
        how the final gauge optimization should be performed.  The keys and
        values of this dictionary may correspond to any of the arguments
        of :func:`gaugeopt_to_target` *except* for the first `model`
        argument, which is specified internally.  The `targetModel` argument,
        *can* be set, but is specified internally when it isn't.  If `None`,
        then the dictionary `{'itemWeights': {'gates':1.0, 'spam':0.001}}`
        is used.  If `False`, then then *no* gauge optimization is performed.

    advancedOptions : dict, optional
        Specifies advanced options most of which deal with numerical details of
        the objective function or expert-level functionality.  See
        :func:`do_long_sequence_gst` for a list of the allowed keys, with the
        exception  "nestedCircuitLists", "opLabelAliases",
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
    printer = _objs.VerbosityPrinter.build_printer(verbosity, comm)

    validStructTypes = (_objs.LsGermsStructure, _objs.LsGermsSerialStructure)
    if isinstance(lsgstLists, validStructTypes) or isinstance(lsgstLists[0], validStructTypes):
        exp_design = _proto.StructuredGSTDesign(targetModelFilenameOrObj, lsgstLists)
    else:
        exp_design = _proto.GateSetTomographyDesign(targetModelFilenameOrObj, lsgstLists)

    ds = _load_dataset(dataFilenameOrSet, comm, printer)
    data = _proto.ProtocolData(exp_design, ds)

    if gaugeOptParams is None:
        gaugeOptParams = {'itemWeights': {'gates': 1.0, 'spam': 0.001}}
    gopt_suite = {'go0': gaugeOptParams} if gaugeOptParams else None
    proto = _proto.GateSetTomography(_get_gst_initial_model(advancedOptions), gopt_suite, None,
                                     _get_gst_builders(advancedOptions),
                                     _get_optimizer(advancedOptions, exp_design),
                                     _get_badfit_options(advancedOptions), printer)

    proto.profile = advancedOptions.get('profile', 1)
    proto.record_output = advancedOptions.get('recordOutput', 1)
    proto.distribute_method = advancedOptions.get('distributeMethod', "default")
    proto.oplabel_aliases = advancedOptions.get('opLabelAliases', None)
    proto.circuit_weights = advancedOptions.get('circuitWeights', None)
    proto.unreliable_ops = advancedOptions.get('unreliableOps', ['Gcnot', 'Gcphase', 'Gms', 'Gcn', 'Gcx', 'Gcz'])

    results = proto.run(data, memLimit, comm)
    _output_to_pickle(results, output_pkl, comm)
    return results


def do_stdpractice_gst(dataFilenameOrSet, targetModelFilenameOrObj,
                       prepStrsListOrFilename, effectStrsListOrFilename,
                       germsListOrFilename, maxLengths, modes="TP,CPTP,Target",
                       gaugeOptSuite='stdgaugeopt',
                       gaugeOptTarget=None, modelsToTest=None, comm=None, memLimit=None,
                       advancedOptions=None, output_pkl=None, verbosity=2):
    """
    Perform end-to-end GST analysis using standard practices.

    This routines is an even higher-level driver than
    :func:`do_long_sequence_gst`.  It performs bottled, typically-useful,
    runs of long sequence GST on a dataset.  This essentially boils down
    to running :func:`do_long_sequence_gst` one or more times using different
    model parameterizations, and performing commonly-useful gauge
    optimizations, based only on the high-level `modes` argument.

    Parameters
    ----------
    dataFilenameOrSet : DataSet or string
        The data set object to use for the analysis, specified either directly
        or by the filename of a dataset file (assumed to be a pickled `DataSet`
        if extension is 'pkl' otherwise assumed to be in pyGSTi's text format).

    targetModelFilenameOrObj : Model or string
        The target model, specified either directly or by the filename of a
        model file (text format).

    prepStrsListOrFilename : (list of Circuits) or string
        The state preparation fiducial operation sequences, specified either directly
        or by the filename of a operation sequence list file (text format).

    effectStrsListOrFilename : (list of Circuits) or string or None
        The measurement fiducial operation sequences, specified either directly or by
        the filename of a operation sequence list file (text format).  If ``None``,
        then use the same strings as specified by prepStrsListOrFilename.

    germsListOrFilename : (list of Circuits) or string
        The germ operation sequences, specified either directly or by the filename of a
        operation sequence list file (text format).

    maxLengths : list of ints
        List of integers, one per LSGST iteration, which set truncation lengths
        for repeated germ strings.  The list of operation sequences for the i-th LSGST
        iteration includes the repeated germs truncated to the L-values *up to*
        and including the i-th one.

    modes : str, optional
        A comma-separated list of modes which dictate what types of analyses
        are performed.  Currently, these correspond to different types of
        parameterizations/constraints to apply to the estimated model.
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

    gaugeOptTarget : Model, optional
        If not None, a model to be used as the "target" for gauge-
        optimization (only).  This argument is useful when you want to
        gauge optimize toward something other than the *ideal* target gates
        given by `targetModelFilenameOrObj`, which are used as the default when
        `gaugeOptTarget` is None.

    modelsToTest : dict, optional
        A dictionary of Model objects representing (gate-set) models to
        test against the data.  These Models are essentially hypotheses for
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
    advancedOptions = advancedOptions or {}
    all_advanced = advancedOptions.get('all', {})
    ds = _load_dataset(dataFilenameOrSet, comm, printer)

    exp_design = _proto.StandardGSTDesign(targetModelFilenameOrObj,
                                          prepStrsListOrFilename, effectStrsListOrFilename,
                                          germsListOrFilename, maxLengths,
                                          all_advanced.get('germLengthLimits', None),
                                          None, 1, None,  # fidPairs, keepFraction, keepSeed
                                          all_advanced.get('includeLGST', True),
                                          all_advanced.get('nestedCircuitLists', True),
                                          all_advanced.get('stringManipRules', None),
                                          all_advanced.get('opLabelAliases', None),
                                          ds, 'drop', verbosity=printer)

    ds = _load_dataset(dataFilenameOrSet, comm, printer)
    data = _proto.ProtocolData(exp_design, ds)
    proto = _proto.StandardGST(modes, gaugeOptSuite, gaugeOptTarget, modelsToTest,
                               _get_gst_builders(advancedOptions),
                               _get_optimizer(advancedOptions, exp_design),
                               _get_badfit_options(advancedOptions), printer)

    results = proto.run(data, memLimit, comm)
    _output_to_pickle(results, output_pkl, comm)
    return results


# --- Helper functions ---

def _load_model(modelFilenameOrObj):
    if isinstance(modelFilenameOrObj, str):
        return _io.load_model(modelFilenameOrObj)
    else:
        return modelFilenameOrObj  # assume a Model object


def _load_dataset(dataFilenameOrSet, comm, verbosity):
    """Loads a DataSet from the dataFilenameOrSet argument of functions in this module."""
    printer = _objs.VerbosityPrinter.build_printer(verbosity, comm)
    if isinstance(dataFilenameOrSet, str):
        if comm is None or comm.Get_rank() == 0:
            if _os.path.splitext(dataFilenameOrSet)[1] == ".pkl":
                with open(dataFilenameOrSet, 'rb') as pklfile:
                    ds = _pickle.load(pklfile)
            else:
                ds = _io.load_dataset(dataFilenameOrSet, True, "aggregate", printer)
            if comm is not None: comm.bcast(ds, root=0)
        else:
            ds = comm.bcast(None, root=0)
    else:
        ds = dataFilenameOrSet  # assume a Dataset object

    return ds


def _update_objfn_builders(builders, advancedOptions):
    def _update_regularization(builder, nm):
        if builder.regularization and nm in builder.regularization and nm in advancedOptions:
            builder.regularization[nm] = advancedOptions[nm]

    def _update_penalty(builder, nm):
        if builder.penalties and nm in builder.penalties and nm in advancedOptions:
            builder.penalties[nm] = advancedOptions[nm]

    for builder in builders:
        _update_regularization(builder, 'probClipInterval')
        _update_regularization(builder, 'minProbClip')
        _update_regularization(builder, 'radius')
        _update_regularization(builder, 'minProbClipForWeighting')
        _update_penalty(builder, 'cptp_penalty_factor')
        _update_penalty(builder, 'spam_penalty_factor')


def _get_badfit_options(advancedOptions):
    old_badfit_options = advancedOptions.get('badFitOptions', {})
    return _proto.GSTBadFitOptions(advancedOptions.get('badFitThreshold', DEFAULT_BAD_FIT_THRESHOLD),
                                   advancedOptions.get('onBadFit', []),
                                   old_badfit_options.get('wildcard_budget_includes_spam', True),
                                   old_badfit_options.get('wildcard_smart_init', True))


def _output_to_pickle(obj, output_pkl, comm):
    if output_pkl and (comm is None or comm.Get_rank() == 0):
        if isinstance(output_pkl, str):
            with open(output_pkl, 'wb') as pklfile:
                _pickle.dump(obj, pklfile)
        else:
            _pickle.dump(obj, output_pkl)


def _get_gst_initial_model(advancedOptions):
    if advancedOptions.get("starting point", None) is None:
        advancedOptions["starting point"] = "LGST-if-possible"  # to keep backward compatibility
    return _proto.GSTInitialModel(None, advancedOptions.get("starting point", None),
                                  advancedOptions.get('depolarizeStart', 0),
                                  advancedOptions.get('randomizeStart', 0),
                                  advancedOptions.get('lgst_gaugeopt_tol', 1e-6),
                                  advancedOptions.get('contractStartToCPTP', 0))


def _get_gst_builders(advancedOptions):
    objfn_builders = _proto.GSTObjFnBuilders.init_simple(
        advancedOptions.get('objective', 'logl'),
        advancedOptions.get('useFreqWeightedChiSq', False),
        advancedOptions.get('alwaysPerformMLE', False),
        advancedOptions.get('onlyPerformMLE', False))
    _update_objfn_builders(objfn_builders.iteration_builders, advancedOptions)
    _update_objfn_builders(objfn_builders.final_builders, advancedOptions)
    return objfn_builders


def _get_optimizer(advancedOptions, exp_design):
    default_fditer = 0 if exp_design.target_model.simtype in ("termorder", "termgap") else 1
    optimizer = {'maxiter': advancedOptions.get('maxIterations', 100000),
                 'tol': advancedOptions.get('tolerance', 1e-6),
                 'fditer': advancedOptions.get('fdIterations', default_fditer)}
    optimizer.update(advancedOptions.get('extra_lm_opts', {}))



# --------------------------------------------------------------------------------------------------
# --- BELOW HERE ARE OLD "LEGACY" FUNCTIONS THAT ARE SOLELY FOR LIMITED BACKWARD COMPATIBILITY -----
# --------------------------------------------------------------------------------------------------

def legacy_do_model_test(modelFilenameOrObj,
                         dataFilenameOrSet, targetModelFilenameOrObj,
                         prepStrsListOrFilename, effectStrsListOrFilename,
                         germsListOrFilename, maxLengths, gaugeOptParams=None,
                         advancedOptions=None, comm=None, memLimit=None,
                         output_pkl=None, verbosity=2):
    """
    Tests a Model model against a DataSet using a specific set of structured
    operation sequences (given by fiducials, maxLengths and germs).

    Constructs operation sequences by repeating germ strings an integer number of
    times such that the length of the repeated germ is less than or equal to
    the maximum length set in maxLengths.  Each string thus constructed is
    sandwiched between all pairs of (prep, effect) fiducial sequences.

    `themodel` is used directly (without any optimization) as the
    the model estimate at each maximum-length "iteration".  The model
    is given a trivial `default_gauge_group` so that it is not altered
    during any gauge optimization step.

    A :class:`~pygsti.report.Results` object is returned, which encapsulates
    the model estimate and related parameters, and can be used with
    report-generation routines.
    Parameters
    ----------
    modelFilenameOrObj : Model or string
        The model model, specified either directly or by the filename of a
        model file (text format).

    dataFilenameOrSet : DataSet or string
        The data set object to use for the analysis, specified either directly
        or by the filename of a dataset file (assumed to be a pickled `DataSet`
        if extension is 'pkl' otherwise assumed to be in pyGSTi's text format).

    targetModelFilenameOrObj : Model or string
        The target model, specified either directly or by the filename of a
        model file (text format).

    prepStrsListOrFilename : (list of Circuits) or string
        The state preparation fiducial operation sequences, specified either directly
        or by the filename of a operation sequence list file (text format).

    effectStrsListOrFilename : (list of Circuits) or string or None
        The measurement fiducial operation sequences, specified either directly or by
        the filename of a operation sequence list file (text format).  If ``None``,
        then use the same strings as specified by prepStrsListOrFilename.

    germsListOrFilename : (list of Circuits) or string
        The germ operation sequences, specified either directly or by the filename of a
        operation sequence list file (text format).

    maxLengths : list of ints
        List of integers, one per LSGST iteration, which set truncation lengths
        for repeated germ strings.  The list of operation sequences for the i-th LSGST
        iteration includes the repeated germs truncated to the L-values *up to*
        and including the i-th one.

    gaugeOptParams : dict, optional
        A dictionary of arguments to :func:`gaugeopt_to_target`, specifying
        how the final gauge optimization should be performed.  The keys and
        values of this dictionary may correspond to any of the arguments
        of :func:`gaugeopt_to_target` *except* for the first `model`
        argument, which is specified internally.  The `targetModel` argument,
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

    #Get/load target & model models
    the_model = _load_model(modelFilenameOrObj)
    target_model = _load_model(targetModelFilenameOrObj)

    #Get/load fiducials and germs
    prepStrs, effectStrs, germs = _load_fiducials_and_germs(
        prepStrsListOrFilename,
        effectStrsListOrFilename,
        germsListOrFilename)

    #Get/load dataset
    ds = _load_dataset(dataFilenameOrSet, comm, verbosity)

    #Construct Circuit lists
    lsgstLists = _get_lsgst_lists(ds, target_model, prepStrs, effectStrs, germs,
                                  maxLengths, advancedOptions, verbosity)

    if gaugeOptParams is None: gaugeOptParams = {}
    if advancedOptions is None: advancedOptions = {}
    if advancedOptions.get('set trivial gauge group', True):
        the_model = the_model.copy()
        the_model.default_gauge_group = _objs.TrivialGaugeGroup(the_model.dim)  # so no gauge opt is done
    mdl_lsgst_list = [the_model] * len(maxLengths)

    #    #Starting Point - compute on rank 0 and distribute
    #LGSTcompatibleOps = all([(isinstance(g,_objs.FullDenseOp) or
    #                            isinstance(g,_objs.TPDenseOp))
    #                           for g in target_model.operations.values()])
    #if isinstance(lsgstLists[0],_objs.LsGermsStructure) and LGSTcompatibleOps:
    #    startingPt = advancedOptions.get('starting point',"LGST")
    #else:
    #    startingPt = advancedOptions.get('starting point',"target")

    #Create profiler
    profile = advancedOptions.get('profile', 1)
    if profile == 0: profiler = _DummyProfiler()
    elif profile == 1: profiler = _objs.Profiler(comm, False)
    elif profile == 2: profiler = _objs.Profiler(comm, True)
    else: raise ValueError("Invalid value for 'profile' argument (%s)" % profile)

    parameters = _collections.OrderedDict()
    parameters['objective'] = advancedOptions.get('objective', 'logl')
    if parameters['objective'] == 'logl':
        parameters['minProbClip'] = advancedOptions.get('minProbClip', 1e-4)
        parameters['radius'] = advancedOptions.get('radius', 1e-4)
    elif parameters['objective'] == 'chi2':
        parameters['minProbClipForWeighting'] = advancedOptions.get(
            'minProbClipForWeighting', 1e-4)
    else:
        raise ValueError("Invalid objective: %s" % parameters['objective'])

    parameters['profiler'] = profiler
    parameters['opLabelAliases'] = advancedOptions.get('opLabelAliases', None)
    parameters['truncScheme'] = advancedOptions.get('truncScheme', "whole germ powers")
    parameters['weights'] = None

    #Set a different default for onBadFit: don't do anything
    if 'onBadFit' not in advancedOptions:
        advancedOptions['onBadFit'] = []  # empty list => 'do nothing'

    return _package_into_results('do_model_test', ds, target_model, the_model,
                                 lsgstLists, parameters, None, mdl_lsgst_list,
                                 gaugeOptParams, advancedOptions, comm, memLimit,
                                 output_pkl, verbosity, profiler)


def legacy_do_linear_gst(dataFilenameOrSet, targetModelFilenameOrObj,
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

    targetModelFilenameOrObj : Model or string
        The target model, specified either directly or by the filename of a
        model file (text format).

    prepStrsListOrFilename : (list of Circuits) or string
        The state preparation fiducial operation sequences, specified either directly
        or by the filename of a operation sequence list file (text format).

    effectStrsListOrFilename : (list of Circuits) or string or None
        The measurement fiducial operation sequences, specified either directly or by
        the filename of a operation sequence list file (text format).  If ``None``,
        then use the same strings as specified by prepStrsListOrFilename.

    gaugeOptParams : dict, optional
        A dictionary of arguments to :func:`gaugeopt_to_target`, specifying
        how the final gauge optimization should be performed.  The keys and
        values of this dictionary may correspond to any of the arguments
        of :func:`gaugeopt_to_target` *except* for the first `model`
        argument, which is specified internally.  The `targetModel` argument,
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
    target_model = _load_model(targetModelFilenameOrObj)
    germs = _construction.circuit_list([()] + [(gl,) for gl in target_model.operations.keys()])  # just the single gates
    maxLengths = [1]  # we only need maxLength == 1 when doing LGST

    defAdvOptions = {'onBadFit': [], 'estimateLabel': 'LGST'}
    if advancedOptions is None: advancedOptions = {}
    advancedOptions.update(defAdvOptions)
    advancedOptions['objective'] = 'lgst'  # not override-able

    return legacy_do_long_sequence_gst(dataFilenameOrSet, target_model,
                                       prepStrsListOrFilename, effectStrsListOrFilename,
                                       germs, maxLengths, gaugeOptParams,
                                       advancedOptions, comm, memLimit,
                                       output_pkl, verbosity)


def legacy_do_long_sequence_gst(dataFilenameOrSet, targetModelFilenameOrObj,
                                prepStrsListOrFilename, effectStrsListOrFilename,
                                germsListOrFilename, maxLengths, gaugeOptParams=None,
                                advancedOptions=None, comm=None, memLimit=None,
                                output_pkl=None, verbosity=2):
    """
    Perform end-to-end GST analysis using Ls and germs, with L as a maximum
    length.

    Constructs operation sequences by repeating germ strings an integer number of
    times such that the length of the repeated germ is less than or equal to
    the maximum length set in maxLengths.  The LGST estimate of the gates is
    computed, gauge optimized, and then used as the seed for either LSGST or
    MLEGST.
    LSGST is iterated ``len(maxLengths)`` times with successively larger sets
    of operation sequences.  On the i-th iteration, the repeated germs sequences
    limited by ``maxLengths[i]`` are included in the growing set of strings
    used by LSGST.  The final iteration will use MLEGST when ``objective ==
    "logl"`` to maximize the true log-likelihood instead of minimizing the
    chi-squared function.
    Once computed, the model estimates are optionally gauge optimized to
    the CPTP space and then to the target model (using `gaugeOptRatio`
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

    targetModelFilenameOrObj : Model or string
        The target model, specified either directly or by the filename of a
        model file (text format).

    prepStrsListOrFilename : (list of Circuits) or string
        The state preparation fiducial operation sequences, specified either directly
        or by the filename of a operation sequence list file (text format).

    effectStrsListOrFilename : (list of Circuits) or string or None
        The measurement fiducial operation sequences, specified either directly or by
        the filename of a operation sequence list file (text format).  If ``None``,
        then use the same strings as specified by prepStrsListOrFilename.

    germsListOrFilename : (list of Circuits) or string
        The germ operation sequences, specified either directly or by the filename of a
        operation sequence list file (text format).

    maxLengths : list of ints
        List of integers, one per LSGST iteration, which set truncation lengths
        for repeated germ strings.  The list of operation sequences for the i-th LSGST
        iteration includes the repeated germs truncated to the L-values *up to*
        and including the i-th one.

    gaugeOptParams : dict, optional
        A dictionary of arguments to :func:`gaugeopt_to_target`, specifying
        how the final gauge optimization should be performed.  The keys and
        values of this dictionary may correspond to any of the arguments
        of :func:`gaugeopt_to_target` *except* for the first `model`
        argument, which is specified internally.  The `targetModel` argument,
        *can* be set, but is specified internally when it isn't.  If `None`,
        then the dictionary `{'itemWeights': {'gates':1.0, 'spam':0.001}}`
        is used.  If `False`, then then *no* gauge optimization is performed.

    advancedOptions : dict, optional
        Specifies advanced options most of which deal with numerical details of
        the objective function or expert-level functionality.  The allowed keys
        and values include:
        - objective = {'chi2', 'logl'}
        - opLabels = list of strings
        - circuitWeights = dict or None
        - starting point = "LGST-if-possible" (default), "LGST", or "target"
        - depolarizeStart = float (default == 0)
        - randomizeStart = float (default == 0)
        - contractStartToCPTP = True / False (default)
        - cptpPenaltyFactor = float (default = 0)
        - tolerance = float or dict w/'relx','relf','f','jac','maxdx' keys
        - maxIterations = int
        - fdIterations = int
        - minProbClip = float
        - minProbClipForWeighting = float (default == 1e-4)
        - probClipInterval = tuple (default == (-1e6,1e6)
        - radius = float (default == 1e-4)
        - useFreqWeightedChiSq = True / False (default)
        - nestedCircuitLists = True (default) / False
        - includeLGST = True / False (default is True)
        - distributeMethod = "default", "circuits" or "deriv"
        - profile = int (default == 1)
        - check = True / False (default)
        - opLabelAliases = dict (default = None)
        - alwaysPerformMLE = bool (default = False)
        - onlyPerformMLE = bool (default = False)
        - truncScheme = "whole germ powers" (default) or "truncated germ powers"
                        or "length as exponent"
        - appendTo = Results (default = None)
        - estimateLabel = str (default = "default")
        - missingDataAction = {'drop','raise'} (default = 'drop')
        - stringManipRules = list of (find,replace) tuples
        - germLengthLimits = dict of form {germ: maxlength}
        - recordOutput = bool (default = True)
        - timeDependent = bool (default = False)

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
    #opLabels : list or tuple
    #    A list or tuple of the operation labels to use when generating the sets of
    #    operation sequences used in LSGST iterations.  If ``None``, then the gate
    #    labels of the target model will be used.  This option is useful if
    #    you only want to include a *subset* of the available gates in the LSGST
    #    strings (e.g. leaving out the identity gate).
    #
    #weightsDict : dict, optional
    #    A dictionary with ``keys == operation sequences`` and ``values ==
    #    multiplicative`` scaling factor for the corresponding operation sequence. The
    #    default is no weight scaling at all.
    #
    #gaugeOptRatio : float, optional
    #    The ratio spamWeight/opWeight used for gauge optimizing to the target
    #    model.
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
    if advancedOptions.get('recordOutput', True) and not printer.is_recording():
        printer.start_recording()

    #Get/load target model
    target_model = _load_model(targetModelFilenameOrObj)

    #Get/load fiducials and germs
    prepStrs, effectStrs, germs = _load_fiducials_and_germs(
        prepStrsListOrFilename,
        effectStrsListOrFilename,
        germsListOrFilename)

    #Get/load dataset
    ds = _load_dataset(dataFilenameOrSet, comm, printer)

    #Construct Circuit lists
    lsgstLists = _get_lsgst_lists(ds, target_model, prepStrs, effectStrs, germs,
                                  maxLengths, advancedOptions, printer)

    return legacy_do_long_sequence_gst_base(ds, target_model, lsgstLists, gaugeOptParams,
                                            advancedOptions, comm, memLimit,
                                            output_pkl, printer)


def legacy_do_long_sequence_gst_base(dataFilenameOrSet, targetModelFilenameOrObj,
                                     lsgstLists, gaugeOptParams=None,
                                     advancedOptions=None, comm=None, memLimit=None,
                                     output_pkl=None, verbosity=2):
    """
    A more fundamental interface for performing end-to-end GST.

    Similar to :func:`do_long_sequence_gst` except this function takes
    `lsgstLists`, a list of either raw operation sequence lists or of `LsGermsStruct`
    gate-string-structure objects to define which gate seqences are used on
    each GST iteration.

    Parameters
    ----------
    dataFilenameOrSet : DataSet or string
        The data set object to use for the analysis, specified either directly
        or by the filename of a dataset file (assumed to be a pickled `DataSet`
        if extension is 'pkl' otherwise assumed to be in pyGSTi's text format).

    targetModelFilenameOrObj : Model or string
        The target model, specified either directly or by the filename of a
        model file (text format).

    lsgstLists : list of lists or LsGermsStruct(s)
        An explicit list of either the raw operation sequence lists to be used in
        the analysis or of LsGermsStruct objects, which additionally contain
        the max-L, germ, and fiducial pair structure of a set of operation sequences.
        A single LsGermsStruct object can also be given, which is equivalent
        to passing a list of successive L-value truncations of this object
        (e.g. if the object has `Ls = [1,2,4]` then this is like passing
         a list of three LsGermsStructs w/truncations `[1]`, `[1,2]`, and
         `[1,2,4]`).

    gaugeOptParams : dict, optional
        A dictionary of arguments to :func:`gaugeopt_to_target`, specifying
        how the final gauge optimization should be performed.  The keys and
        values of this dictionary may correspond to any of the arguments
        of :func:`gaugeopt_to_target` *except* for the first `model`
        argument, which is specified internally.  The `targetModel` argument,
        *can* be set, but is specified internally when it isn't.  If `None`,
        then the dictionary `{'itemWeights': {'gates':1.0, 'spam':0.001}}`
        is used.  If `False`, then then *no* gauge optimization is performed.

    advancedOptions : dict, optional
        Specifies advanced options most of which deal with numerical details of
        the objective function or expert-level functionality.  See
        :func:`do_long_sequence_gst` for a list of the allowed keys, with the
        exception  "nestedCircuitLists", "opLabelAliases",
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
        gaugeOptParams = {'itemWeights': {'gates': 1.0, 'spam': 0.001}}

    profile = advancedOptions.get('profile', 1)

    if profile == 0: profiler = _DummyProfiler()
    elif profile == 1: profiler = _objs.Profiler(comm, False)
    elif profile == 2: profiler = _objs.Profiler(comm, True)
    else: raise ValueError("Invalid value for 'profile' argument (%s)" % profile)

    if 'verbosity' in advancedOptions:  # for backward compatibility
        _warnings.warn("'verbosity' as an advanced option is deprecated."
                       " Please use the 'verbosity' argument directly.")
        verbosity = advancedOptions['verbosity']
    if 'memoryLimitInBytes' in advancedOptions:  # for backward compatibility
        _warnings.warn("'memoryLimitInBytes' as an advanced option is deprecated."
                       " Please use the 'memLimit' argument directly.")
        memLimit = advancedOptions['memoryLimitInBytes']

    printer = _objs.VerbosityPrinter.build_printer(verbosity, comm)
    if advancedOptions.get('recordOutput', True) and not printer.is_recording():
        printer.start_recording()

    #Get/load target model
    target_model = _load_model(targetModelFilenameOrObj)

    #Get/load dataset
    ds = _load_dataset(dataFilenameOrSet, comm, printer)

    op_dim = target_model.get_dimension()

    tNxt = _time.time()
    profiler.add_time('do_long_sequence_gst: loading', tRef); tRef = tNxt

    #Convert a single LsGermsStruct to a list if needed:
    validStructTypes = (_objs.LsGermsStructure, _objs.LsGermsSerialStructure)
    if isinstance(lsgstLists, validStructTypes):
        master = lsgstLists
        lsgstLists = [master.truncate(Ls=master.Ls[0:i + 1])
                      for i in range(len(master.Ls))]

    #Starting Point - compute on rank 0 and distribute
    if isinstance(target_model, _objs.ExplicitOpModel):
        LGSTcompatibleOps = all([(isinstance(g, _objs.FullDenseOp)
                                  or isinstance(g, _objs.TPDenseOp))
                                 for g in target_model.operations.values()])
    else:
        LGSTcompatibleOps = False

    if isinstance(lsgstLists[0], validStructTypes) and LGSTcompatibleOps:
        startingPt = advancedOptions.get('starting point', "LGST")
    else:
        startingPt = advancedOptions.get('starting point', "target")

    #Compute starting point
    if startingPt == "LGST":
        assert(isinstance(lsgstLists[0], validStructTypes)), \
            "Cannot run LGST: fiducials not specified!"
        opLabels = advancedOptions.get('opLabels',
                                       list(target_model.operations.keys())
                                       + list(target_model.instruments.keys()))
        mdl_start = _alg.do_lgst(ds, lsgstLists[0].prepStrs, lsgstLists[0].effectStrs, target_model,
                                 opLabels, svdTruncateTo=op_dim,
                                 opLabelAliases=lsgstLists[0].aliases,
                                 verbosity=printer)  # returns a model with the *same*
        # parameterizations as target_model

        #In LGST case, gauge optimimize starting point to the target
        # (historical; sometimes seems to help in practice, since it's gauge
        # optimizing to physical gates (e.g. something in CPTP)
        tol = gaugeOptParams.get('tol', 1e-8) if gaugeOptParams else 1e-8
        mdl_start = _alg.gaugeopt_to_target(mdl_start, target_model, tol=tol, comm=comm)
        #Note: use *default* gauge-opt params when optimizing

    elif startingPt == "target":
        mdl_start = target_model.copy()
    elif isinstance(startingPt, _objs.Model):
        mdl_start = startingPt
        startingPt = "User-supplied-Model"  # for profiler log below
    else:
        raise ValueError("Invalid starting point: %s" % startingPt)

    tNxt = _time.time()
    profiler.add_time('do_long_sequence_gst: Starting Point (%s)'
                      % startingPt, tRef); tRef = tNxt

    #Post-processing mdl_start : done only on root proc in case there is any nondeterminism.
    if comm is None or comm.Get_rank() == 0:
        #Advanced Options can specify further manipulation of starting model
        if advancedOptions.get('contractStartToCPTP', False):
            mdl_start = _alg.contract(mdl_start, "CPTP")
            raise ValueError("'contractStartToCPTP' has been removed b/c it can change the parameterization of a model")
        if advancedOptions.get('depolarizeStart', 0) > 0:
            mdl_start = mdl_start.depolarize(op_noise=advancedOptions.get('depolarizeStart', 0))
        if advancedOptions.get('randomizeStart', 0) > 0:
            v = mdl_start.to_vector()
            vrand = 2 * (_np.random.random(len(v)) - 0.5) * advancedOptions.get('randomizeStart', 0)
            mdl_start.from_vector(v + vrand)

        if comm is not None:  # broadcast starting model
            #OLD: comm.bcast(mdl_start, root=0)
            # just broadcast *vector* to avoid huge pickles (if cached calcs!)
            comm.bcast(mdl_start.to_vector(), root=0)
    else:
        #OLD: mdl_start = comm.bcast(None, root=0)
        v = comm.bcast(None, root=0)
        mdl_start.from_vector(v)

    tNxt = _time.time()
    profiler.add_time('do_long_sequence_gst: Prep Initial seed', tRef); tRef = tNxt

    # lsgstLists can hold either circuit lists or structures - get
    # just the lists for calling core gst routines (structure is used only
    # for LGST and post-analysis).
    rawLists = [l.allstrs if isinstance(l, validStructTypes) else l
                for l in lsgstLists]

    aliases = lsgstLists[-1].aliases if isinstance(
        lsgstLists[-1], validStructTypes) else None
    aliases = advancedOptions.get('opLabelAliases', aliases)

    #Run Long-sequence GST on data
    objective = advancedOptions.get('objective', 'logl')
    default_fditer = 0 if mdl_start.simtype in ("termorder", "termgap") else 1

    args = dict(
        dataset=ds,
        startModel=mdl_start,
        circuitSetsToUseInEstimation=rawLists,
        tol=advancedOptions.get('tolerance', 1e-6),
        extra_lm_opts=advancedOptions.get('extra_lm_opts', None),
        cptp_penalty_factor=advancedOptions.get('cptpPenaltyFactor', 0),
        spam_penalty_factor=advancedOptions.get('spamPenaltyFactor', 0),
        maxiter=advancedOptions.get('maxIterations', 100000),
        fditer=advancedOptions.get('fdIterations', default_fditer),
        probClipInterval=advancedOptions.get('probClipInterval', (-1e6, 1e6)),
        returnAll=True,
        circuitWeightsDict=advancedOptions.get('circuitWeights', None),
        opLabelAliases=aliases,
        verbosity=printer,
        memLimit=memLimit,
        profiler=profiler,
        comm=comm, distributeMethod=advancedOptions.get(
            'distributeMethod', "default"),
        check=advancedOptions.get('check', False),
        evaltree_cache={},
        time_dependent=advancedOptions.get('timeDependent', False))

    if objective == "chi2":
        args['useFreqWeightedChiSq'] = advancedOptions.get(
            'useFreqWeightedChiSq', False)
        args['minProbClipForWeighting'] = advancedOptions.get(
            'minProbClipForWeighting', 1e-4)
        args['check_jacobian'] = advancedOptions.get('check', False)
        mdl_lsgst_list = _alg.do_iterative_mc2gst(**args)

    elif objective == "logl":
        args['minProbClip'] = advancedOptions.get('minProbClip', 1e-4)
        args['radius'] = advancedOptions.get('radius', 1e-4)
        args['alwaysPerformMLE'] = advancedOptions.get('alwaysPerformMLE', False)
        args['onlyPerformMLE'] = advancedOptions.get('onlyPerformMLE', False)
        mdl_lsgst_list = _alg.do_iterative_mlgst(**args)

    elif objective == "lgst":
        assert(startingPt == "LGST"), "Can only set objective=\"lgst\" for parameterizations compatible with LGST"
        assert(len(lsgstLists) == 1), "Can only set objective=\"lgst\" with number if lists/max-lengths == 1"
        mdl_lsgst_list = [args['startModel']]
    else:
        raise ValueError("Invalid objective: %s" % objective)

    tNxt = _time.time()
    profiler.add_time('do_long_sequence_gst: total long-seq. opt.', tRef); tRef = tNxt

    #set parameters
    parameters = _collections.OrderedDict()
    parameters['objective'] = objective
    parameters['memLimit'] = memLimit
    parameters['starting point'] = startingPt
    parameters['profiler'] = profiler

    #from advanced options
    parameters['minProbClip'] = \
        advancedOptions.get('minProbClip', 1e-4)
    parameters['minProbClipForWeighting'] = \
        advancedOptions.get('minProbClipForWeighting', 1e-4)
    parameters['probClipInterval'] = \
        advancedOptions.get('probClipInterval', (-1e6, 1e6))
    parameters['radius'] = advancedOptions.get('radius', 1e-4)
    parameters['weights'] = advancedOptions.get('circuitWeights', None)
    parameters['cptpPenaltyFactor'] = advancedOptions.get('cptpPenaltyFactor', 0)
    parameters['spamPenaltyFactor'] = advancedOptions.get('spamPenaltyFactor', 0)
    parameters['distributeMethod'] = advancedOptions.get('distributeMethod', 'default')
    parameters['depolarizeStart'] = advancedOptions.get('depolarizeStart', 0)
    parameters['randomizeStart'] = advancedOptions.get('randomizeStart', 0)
    parameters['contractStartToCPTP'] = advancedOptions.get('contractStartToCPTP', False)
    parameters['tolerance'] = advancedOptions.get('tolerance', 1e-6)
    parameters['maxIterations'] = advancedOptions.get('maxIterations', 100000)
    parameters['useFreqWeightedChiSq'] = advancedOptions.get('useFreqWeightedChiSq', False)
    parameters['nestedCircuitLists'] = advancedOptions.get('nestedCircuitLists', True)
    parameters['profile'] = advancedOptions.get('profile', 1)
    parameters['check'] = advancedOptions.get('check', False)
    parameters['truncScheme'] = advancedOptions.get('truncScheme', "whole germ powers")
    parameters['opLabelAliases'] = advancedOptions.get('opLabelAliases', None)
    parameters['includeLGST'] = advancedOptions.get('includeLGST', True)

    return _package_into_results('do_long_sequence_gst', ds, target_model, mdl_start,
                                 lsgstLists, parameters, args, mdl_lsgst_list,
                                 gaugeOptParams, advancedOptions, comm, memLimit,
                                 output_pkl, printer, profiler, args['evaltree_cache'])


def legacy_do_stdpractice_gst(dataFilenameOrSet, targetModelFilenameOrObj,
                              prepStrsListOrFilename, effectStrsListOrFilename,
                              germsListOrFilename, maxLengths, modes="TP,CPTP,Target",
                              gaugeOptSuite=('single', 'unreliable2Q'),
                              gaugeOptTarget=None, modelsToTest=None, comm=None, memLimit=None,
                              advancedOptions=None, output_pkl=None, verbosity=2):
    """
    Perform end-to-end GST analysis using standard practices.

    This routines is an even higher-level driver than
    :func:`do_long_sequence_gst`.  It performs bottled, typically-useful,
    runs of long sequence GST on a dataset.  This essentially boils down
    to running :func:`do_long_sequence_gst` one or more times using different
    model parameterizations, and performing commonly-useful gauge
    optimizations, based only on the high-level `modes` argument.

    Parameters
    ----------
    dataFilenameOrSet : DataSet or string
        The data set object to use for the analysis, specified either directly
        or by the filename of a dataset file (assumed to be a pickled `DataSet`
        if extension is 'pkl' otherwise assumed to be in pyGSTi's text format).

    targetModelFilenameOrObj : Model or string
        The target model, specified either directly or by the filename of a
        model file (text format).

    prepStrsListOrFilename : (list of Circuits) or string
        The state preparation fiducial operation sequences, specified either directly
        or by the filename of a operation sequence list file (text format).

    effectStrsListOrFilename : (list of Circuits) or string or None
        The measurement fiducial operation sequences, specified either directly or by
        the filename of a operation sequence list file (text format).  If ``None``,
        then use the same strings as specified by prepStrsListOrFilename.

    germsListOrFilename : (list of Circuits) or string
        The germ operation sequences, specified either directly or by the filename of a
        operation sequence list file (text format).

    maxLengths : list of ints
        List of integers, one per LSGST iteration, which set truncation lengths
        for repeated germ strings.  The list of operation sequences for the i-th LSGST
        iteration includes the repeated germs truncated to the L-values *up to*
        and including the i-th one.

    modes : str, optional
        A comma-separated list of modes which dictate what types of analyses
        are performed.  Currently, these correspond to different types of
        parameterizations/constraints to apply to the estimated model.
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

    gaugeOptTarget : Model, optional
        If not None, a model to be used as the "target" for gauge-
        optimization (only).  This argument is useful when you want to
        gauge optimize toward something other than the *ideal* target gates
        given by `targetModelFilenameOrObj`, which are used as the default when
        `gaugeOptTarget` is None.

    modelsToTest : dict, optional
        A dictionary of Model objects representing (gate-set) models to
        test against the data.  These Models are essentially hypotheses for
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

    #Get/load target model
    target_model = _load_model(targetModelFilenameOrObj)

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
        for i, mode in enumerate(modes):
            printer.show_progress(i, len(modes), prefix='-- Std Practice: ', suffix=' (%s) --' % mode)

            #prepare advanced options dictionary
            if advancedOptions is not None:
                advanced = advancedOptions.get('all', {})
                advanced.update(advancedOptions.get(mode, {}))
            else: advanced = {}

            if mode == "Target":
                est_label = mode
                tgt = target_model.copy()  # no parameterization change
                tgt.default_gauge_group = _objs.TrivialGaugeGroup(tgt.dim)  # so no gauge opt is done
                advanced.update({'appendTo': ret, 'estimateLabel': est_label,
                                 'onBadFit': []})
                ret = legacy_do_model_test(target_model, ds, tgt, prepStrs,
                                           effectStrs, germs, maxLengths, False, advanced,
                                           comm, memLimit, None, printer - 1)

            elif mode in ('full', 'TP', 'CPTP', 'H+S', 'S', 'static'):  # mode is a parameterization
                est_label = parameterization = mode  # for now, 1-1 correspondence
                tgt = target_model.copy(); tgt.set_all_parameterizations(parameterization)
                advanced.update({'appendTo': ret, 'estimateLabel': est_label})
                ret = legacy_do_long_sequence_gst(ds, tgt, prepStrs, effectStrs, germs,
                                                  maxLengths, False, advanced, comm, memLimit,
                                                  None, printer - 1)
            elif mode in modelsToTest:
                est_label = mode
                tgt = target_model.copy()  # no parameterization change
                tgt.default_gauge_group = _objs.TrivialGaugeGroup(tgt.dim)  # so no gauge opt is done
                advanced.update({'appendTo': ret, 'estimateLabel': est_label})
                ret = legacy_do_model_test(modelsToTest[mode], ds, tgt, prepStrs,
                                           effectStrs, germs, maxLengths, False, advanced,
                                           comm, memLimit, None, printer - 1)
            else:
                raise ValueError("Invalid item in 'modes' argument: %s" % mode)

            #Get gauge optimization dictionary
            assert(not printer.is_recording()); printer.start_recording()
            gaugeOptSuite_dict = gaugeopt_suite_to_dictionary(gaugeOptSuite, tgt,
                                                              advancedOptions, printer - 1)

            if gaugeOptTarget is not None:
                assert(isinstance(gaugeOptTarget, _objs.Model)), "`gaugeOptTarget` must be None or a Model"
                for goparams in gaugeOptSuite_dict.values():
                    goparams_list = [goparams] if hasattr(goparams, 'keys') else goparams
                    for goparams_dict in goparams_list:
                        if 'targetModel' in goparams_dict:
                            _warnings.warn(("`gaugeOptTarget` argument is overriding"
                                            "user-defined targetModel in gauge opt"
                                            "param dict(s)"))
                        goparams_dict.update({'targetModel': gaugeOptTarget})

            #Gauge optimize to list of gauge optimization parameters
            for goLabel, goparams in gaugeOptSuite_dict.items():

                printer.log("-- Performing '%s' gauge optimization on %s estimate --" % (goLabel, est_label), 2)
                gsStart = ret.estimates[est_label].get_start_model(goparams)
                ret.estimates[est_label].add_gaugeoptimized(goparams, None, goLabel, comm, printer - 3)

                #Gauge optimize data-scaled estimate also
                for suffix in ROBUST_SUFFIX_LIST:
                    if est_label + suffix in ret.estimates:
                        gsStart_robust = ret.estimates[est_label + suffix].get_start_model(goparams)
                        if gsStart_robust.frobeniusdist(gsStart) < 1e-8:
                            printer.log("-- Conveying '%s' gauge optimization to %s estimate --" %
                                        (goLabel, est_label + suffix), 2)
                            params = ret.estimates[est_label].goparameters[goLabel]  # no need to copy here
                            gsopt = ret.estimates[est_label].models[goLabel].copy()
                            ret.estimates[est_label + suffix].add_gaugeoptimized(params, gsopt, goLabel, comm,
                                                                                 printer - 3)
                        else:
                            printer.log("-- Performing '%s' gauge optimization on %s estimate --" %
                                        (goLabel, est_label + suffix), 2)
                            ret.estimates[est_label + suffix].add_gaugeoptimized(goparams, None, goLabel, comm,
                                                                                 printer - 3)

            # Add gauge optimizations to end of any existing "stdout" meta info
            if 'stdout' in ret.estimates[est_label].meta:
                ret.estimates[est_label].meta['stdout'].extend(printer.stop_recording())
            else:
                ret.estimates[est_label].meta['stdout'] = printer.stop_recording()

    #Write results to a pickle file if desired
    if output_pkl and (comm is None or comm.Get_rank() == 0):
        if isinstance(output_pkl, str):
            with open(output_pkl, 'wb') as pklfile:
                _pickle.dump(ret, pklfile)
        else:
            _pickle.dump(ret, output_pkl)

    return ret


def gaugeopt_suite_to_dictionary(gaugeOptSuite, target_model, advancedOptions=None, verbosity=0):
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

    target_model : Model
        A model which specifies the dimension (i.e. parameterization) of the
        gauge-optimization and the basis.  Usually this is set to the *ideal*
        `target model` for the model being gauge optimized.

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
            if hasattr(goparams, 'keys'):
                gaugeOptSuite_dict[lbl] = goparams.copy()
                gaugeOptSuite_dict[lbl].update({'verbosity': printer})
            else:
                assert(isinstance(goparams, list)), "If not a dictionary, gauge opt params should be a list of dicts!"
                gaugeOptSuite_dict[lbl] = []
                for goparams_stage in goparams:
                    dct = goparams_stage.copy()
                    dct.update({'verbosity': printer})
                    gaugeOptSuite_dict[lbl].append(dct)

    else:
        gaugeOptSuite_dict = _collections.OrderedDict()
        if isinstance(gaugeOptSuite, str):
            gaugeOptSuites = [gaugeOptSuite]
        else:
            gaugeOptSuites = gaugeOptSuite[:]  # assumes gaugeOptSuite is a list/tuple of strs

        for suiteName in gaugeOptSuites:
            if suiteName == "single":

                stages = []  # multi-stage gauge opt
                gg = target_model.default_gauge_group
                if isinstance(gg, _objs.TrivialGaugeGroup):
                    #just do a single-stage "trivial" gauge opts using default group
                    gaugeOptSuite_dict['single'] = {'verbosity': printer}

                    if "unreliable2Q" in gaugeOptSuites and target_model.dim == 16:
                        if advancedOptions is not None:
                            # 'unreliableOps' can only be specified in 'all' options
                            advanced = advancedOptions.get('all', {})
                        else: advanced = {}
                        unreliableOps = advanced.get('unreliableOps', ['Gcnot', 'Gcphase', 'Gms', 'Gcn', 'Gcx', 'Gcz'])
                        if any([gl in target_model.operations.keys() for gl in unreliableOps]):
                            gaugeOptSuite_dict['single-2QUR'] = {'verbosity': printer}

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
                            'gauge_group': _objs.UnitaryGaugeGroup(target_model.dim, target_model.basis),
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
                            'gauge_group': s3gg(target_model.dim),
                            'verbosity': printer
                        })

                    gaugeOptSuite_dict['single'] = stages  # can be a list of stage dictionaries

                    if "unreliable2Q" in gaugeOptSuites and target_model.dim == 16:
                        if advancedOptions is not None:
                            # 'unreliableOps' can only be specified in 'all' options
                            advanced = advancedOptions.get('all', {})
                        else: advanced = {}
                        unreliableOps = advanced.get('unreliableOps', ['Gcnot', 'Gcphase', 'Gms', 'Gcn', 'Gcx', 'Gcz'])
                        if any([gl in target_model.operations.keys() for gl in unreliableOps]):
                            stage2_item_weights = {'gates': 1, 'spam': 0.0}
                            for gl in unreliableOps:
                                if gl in target_model.operations.keys(): stage2_item_weights[gl] = 0.01
                            stages_2QUR = [stage.copy() for stage in stages]  # ~deep copy of stages
                            iStage2 = 1 if gg.name in ("Full", "TP") else 0
                            stages_2QUR[iStage2]['itemWeights'] = stage2_item_weights
                            gaugeOptSuite_dict['single-2QUR'] = stages_2QUR  # add additional gauge opt
                        else:
                            _warnings.warn(("`unreliable2Q` was given as a gauge opt suite, but none of the"
                                            " gate names in advancedOptions['all']['unreliableOps'], i.e., %s,"
                                            " are present in the target model.  Omitting 'single-2QUR' gauge opt.")
                                           % (", ".join(unreliableOps)))

            elif suiteName in ("varySpam", "varySpamWt", "varyValidSpamWt", "toggleValidSpam"):

                itemWeights_bases = _collections.OrderedDict()
                itemWeights_bases[""] = {'gates': 1}

                if "unreliable2Q" in gaugeOptSuites and target_model.dim == 16:
                    if advancedOptions is not None:
                        # 'unreliableOps' can only be specified in 'all' options
                        advanced = advancedOptions.get('all', {})
                    else: advanced = {}
                    unreliableOps = advanced.get('unreliableOps', ['Gcnot', 'Gcphase', 'Gms', 'Gcn', 'Gcx', 'Gcz'])
                    if any([gl in target_model.operations.keys() for gl in unreliableOps]):
                        base = {'gates': 1}
                        for gl in unreliableOps:
                            if gl in target_model.operations.keys(): base[gl] = 0.01
                        itemWeights_bases["-2QUR"] = base

                if suiteName == "varySpam":
                    vSpam_range = [0, 1]; spamWt_range = [1e-4, 1e-1]
                elif suiteName == "varySpamWt":
                    vSpam_range = [0]; spamWt_range = [1e-4, 1e-1]
                elif suiteName == "varyValidSpamWt":
                    vSpam_range = [1]; spamWt_range = [1e-4, 1e-1]
                elif suiteName == "toggleValidSpam":
                    vSpam_range = [0, 1]; spamWt_range = [1e-3]

                for postfix, baseWts in itemWeights_bases.items():
                    for vSpam in vSpam_range:
                        for spamWt in spamWt_range:
                            lbl = "Spam %g%s%s" % (spamWt, "+v" if vSpam else "", postfix)
                            itemWeights = baseWts.copy()
                            itemWeights['spam'] = spamWt
                            gaugeOptSuite_dict[lbl] = {
                                'itemWeights': itemWeights,
                                'spam_penalty_factor': vSpam, 'verbosity': printer}

            elif suiteName == "unreliable2Q":
                assert(any([nm in ("single", "varySpam", "varySpamWt", "varyValidSpamWt", "toggleValidSpam")
                            for nm in gaugeOptSuite])), "'unreliable2Q' suite must be used with a spam or single suite."

            elif suiteName == "none":
                pass  # add nothing

            else:
                raise ValueError("Invalid `gaugeOptSuite` argument - unknown suite '%s'" % suiteName)

    return gaugeOptSuite_dict


# ------------------ HELPER FUNCTIONS -----------------------------------


def _load_fiducials_and_germs(prepStrsListOrFilename,
                              effectStrsListOrFilename,
                              germsListOrFilename):

    if isinstance(prepStrsListOrFilename, str):
        prepStrs = _io.load_circuit_list(prepStrsListOrFilename)
    else: prepStrs = prepStrsListOrFilename

    if effectStrsListOrFilename is None:
        effectStrs = prepStrs  # use same strings for effectStrs if effectStrsListOrFilename is None
    else:
        if isinstance(effectStrsListOrFilename, str):
            effectStrs = _io.load_circuit_list(effectStrsListOrFilename)
        else: effectStrs = effectStrsListOrFilename

    #Get/load germs
    if isinstance(germsListOrFilename, str):
        germs = _io.load_circuit_list(germsListOrFilename)
    else: germs = germsListOrFilename

    return prepStrs, effectStrs, germs


def _get_lsgst_lists(dschk, target_model, prepStrs, effectStrs, germs,
                     maxLengths, advancedOptions, verbosity):
    """
    Sequence construction logic, fatctored into this separate
    function because it's shared do_long_sequence_gst and
    do_model_evaluation.
    """
    if advancedOptions is None: advancedOptions = {}

    #Update: now always include LGST strings unless advanced options says otherwise
    #Get starting point (so we know whether to include LGST strings)
    #LGSTcompatibleOps = all([(isinstance(g,_objs.FullDenseOp) or
    #                            isinstance(g,_objs.TPDenseOp))
    #                           for g in target_model.operations.values()])
    #if  LGSTcompatibleOps:
    #    startingPt = advancedOptions.get('starting point',"LGST")
    #else:
    #    startingPt = advancedOptions.get('starting point',"target")

    #Construct operation sequences
    actionIfMissing = advancedOptions.get('missingDataAction', 'drop')
    opLabels = advancedOptions.get(
        'opLabels', list(target_model.get_primitive_op_labels()))
    lsgstLists = _construction.stdlists.make_lsgst_structs(
        opLabels, prepStrs, effectStrs, germs, maxLengths,
        truncScheme=advancedOptions.get('truncScheme', "whole germ powers"),
        nest=advancedOptions.get('nestedCircuitLists', True),
        includeLGST=advancedOptions.get('includeLGST', True),
        opLabelAliases=advancedOptions.get('opLabelAliases', None),
        sequenceRules=advancedOptions.get('stringManipRules', None),
        dscheck=dschk, actionIfMissing=actionIfMissing,
        germLengthLimits=advancedOptions.get('germLengthLimits', None),
        verbosity=verbosity)
    assert(len(maxLengths) == len(lsgstLists))

    return lsgstLists


def _package_into_results(callerName, ds, target_model, mdl_start, lsgstLists,
                          parameters, opt_args, mdl_lsgst_list, gaugeOptParams,
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

    ret = advancedOptions.get('appendTo', None)
    if ret is None:
        ret = _objs.Results()
        ret.init_dataset(ds)
        ret.init_circuits(lsgstLists)
    else:
        assert(ret.dataset is ds), "DataSet inconsistency: cannot append!"
        assert(len(lsgstLists) == len(ret.circuit_structs['iteration'])), \
            "Iteration count inconsistency: cannot append!"
        for i, (a, b) in enumerate(zip(lsgstLists, ret.circuit_structs['iteration'])):
            assert(len(a.allstrs) == len(b.allstrs)), \
                "Iteration %d should have %d of sequences but has %d" % (i, len(b.allstrs), len(a.allstrs))
            assert(set(a.allstrs) == set(b.allstrs)), \
                "Iteration %d: sequences don't match existing Results object!" % i

    #add estimate to Results
    estlbl = advancedOptions.get('estimateLabel', 'default')
    ret.add_estimate(target_model, mdl_start, mdl_lsgst_list, parameters, estlbl)
    profiler.add_time('%s: results initialization' % callerName, tRef); tRef = _time.time()

    #Do final gauge optimization to *final* iteration result only
    if gaugeOptParams:
        add_gauge_opt(ret.estimates[estlbl], gaugeOptParams, target_model,
                      mdl_lsgst_list[-1], comm, printer - 1)
        profiler.add_time('%s: gauge optimization' % callerName, tRef)

    #Perform extra analysis if a bad fit was obtained
    badFitThreshold = advancedOptions.get('badFitThreshold', DEFAULT_BAD_FIT_THRESHOLD)
    onBadFit = advancedOptions.get('onBadFit', [])  # ["wildcard"]) #["Robust+"]) # empty list => 'do nothing'
    add_badfit_estimates(ret, estlbl, onBadFit, badFitThreshold, opt_args, evaltree_cache, comm, memLimit, printer - 1)
    profiler.add_time('%s: add badfit estimates' % callerName, tRef); tRef = _time.time()

    #Add recorded info (even robust-related info) to the *base*
    #   estimate label's "stdout" meta information
    if printer.is_recording():
        ret.estimates[estlbl].meta['stdout'] = printer.stop_recording()

    #Write results to a pickle file if desired
    if output_pkl and (comm is None or comm.Get_rank() == 0):
        if isinstance(output_pkl, str):
            with open(output_pkl, 'wb') as pklfile:
                _pickle.dump(ret, pklfile)
        else:
            _pickle.dump(ret, output_pkl)

    return ret


def add_gauge_opt(estimate, gaugeOptParams, target_model, starting_model,
                  comm=None, verbosity=0):
    """
    Add a gauge optimization to an estimate.
    TODO: docstring - more details
    """
    gaugeOptParams = gaugeOptParams.copy()  # so we don't modify the caller's dict
    if '_gaugeGroupEl' in gaugeOptParams: del gaugeOptParams['_gaugeGroupEl']

    if "targetModel" not in gaugeOptParams:
        gaugeOptParams["targetModel"] = target_model

    # somewhat redundant given add_gaugeoptimized behavior - but
    #  if not here won't do parallel gaugeopt_to_target below
    if "comm" not in gaugeOptParams:
        gaugeOptParams["comm"] = comm

    gaugeOptParams['returnAll'] = True  # so we get gaugeEl to save
    gaugeOptParams['model'] = starting_model

    if isinstance(gaugeOptParams['model'], _objs.ExplicitOpModel):
        #only explicit models can be gauge optimized
        _, gaugeEl, go_gs_final = _alg.gaugeopt_to_target(**gaugeOptParams)
    else:
        #but still fill in results for other models (?)
        gaugeEl, go_gs_final = None, gaugeOptParams['model'].copy()

    gaugeOptParams['_gaugeGroupEl'] = gaugeEl  # store gaugeopt el
    estimate.add_gaugeoptimized(gaugeOptParams, go_gs_final,
                                None, comm, verbosity)


def add_badfit_estimates(results, base_estimate_label="default", estimate_types=('wildcard',),
                         badFitThreshold=None, opt_args=None, evaltree_cache=None,
                         comm=None, memLimit=None, verbosity=0):
    """
    Add any and all "bad fit" estimates to `results`.
    TODO: docstring
    """
    printer = _objs.VerbosityPrinter.build_printer(verbosity, comm)
    base_estimate = results.estimates[base_estimate_label]
    lsgstLists = results.circuit_structs['iteration']
    mdl_lsgst_list = base_estimate.models['iteration estimates']
    mdl_start = base_estimate.models['seed']
    target_model = base_estimate.models['target']
    ds = results.dataset
    parameters = base_estimate.parameters
    if evaltree_cache is None: evaltree_cache = {}  # so tree gets cached

    if badFitThreshold is not None and \
       base_estimate.misfit_sigma(evaltree_cache=evaltree_cache, use_accurate_Np=True, comm=comm) <= badFitThreshold:
        return  # fit is good enough - no need to add any estimates

    objective = parameters.get('objective', 'logl')
    validStructTypes = (_objs.LsGermsStructure, _objs.LsGermsSerialStructure)
    rawLists = [l.allstrs if isinstance(l, validStructTypes) else l
                for l in lsgstLists]
    circuitList = rawLists[-1]  # use final circuit list
    mdl = mdl_lsgst_list[-1]    # and model

    assert(parameters.get('weights', None) is None), \
        "Cannot perform bad-fit scaling when weights are already given!"

    for badfit_typ in estimate_types:
        new_params = parameters.copy()
        new_final_model = None

        if badfit_typ in ("robust", "Robust", "robust+", "Robust+"):
            new_params['weights'] = get_robust_scaling(badfit_typ, mdl, ds, circuitList,
                                                       parameters, evaltree_cache, comm, memLimit)
            if badfit_typ in ("Robust", "Robust+") and (opt_args is not None):
                mdl_reopt = reoptimize_with_weights(mdl, ds, circuitList, new_params['weights'],
                                                    objective, opt_args, printer - 1)
                new_final_model = mdl_reopt

        elif badfit_typ == "wildcard":
            new_params['unmodeled_error'] = get_wildcard_budget(mdl, ds, circuitList, parameters,
                                                                evaltree_cache, comm, memLimit, printer - 1)

        elif badfit_typ == "do nothing":
            continue  # go to next on-bad-fit directive

        else:
            raise ValueError("Invalid on-bad-fit directive: %s" % badfit_typ)

        # In case we've computed an updated final model, Just keep (?) old estimates of all
        # prior iterations (or use "blank" sentinel once this is supported).
        models_by_iter = mdl_lsgst_list[:] if (new_final_model is None) \
            else mdl_lsgst_list[0:-1] + [new_final_model]

        results.add_estimate(target_model, mdl_start, models_by_iter,
                             new_params, base_estimate_label + "." + badfit_typ)

        #Add gauge optimizations to the new estimate
        for gokey, gaugeOptParams in base_estimate.goparameters.items():
            if new_final_model is not None:
                add_gauge_opt(results.estimates[base_estimate_label + '.' + badfit_typ], gaugeOptParams,
                              target_model, new_final_model, comm, printer - 1)
            else:
                # add same gauge-optimized result as above
                go_gs_final = base_estimate.models[gokey]
                results.estimates[base_estimate_label + '.' + badfit_typ].add_gaugeoptimized(
                    gaugeOptParams.copy(), go_gs_final, None, comm, printer - 1)


def _get_fit_qty(model, ds, circuitList, parameters, evaltree_cache, comm, memLimit):
    # Get by-sequence goodness of fit
    objective = parameters.get('objective', 'logl')

    if objective == "chi2":
        fitQty = _tools.chi2_terms(model, ds, circuitList,
                                   parameters.get('minProbClipForWeighting', 1e-4),
                                   parameters.get('probClipInterval', (-1e6, 1e6)),
                                   False, False, memLimit,
                                   parameters.get('opLabelAliases', None),
                                   evaltree_cache=evaltree_cache, comm=comm)
    else:  # "logl" or "lgst"
        maxLogL = _tools.logl_max_terms(model, ds, circuitList,
                                        opLabelAliases=parameters.get(
                                            'opLabelAliases', None),
                                        evaltree_cache=evaltree_cache)

        logL = _tools.logl_terms(model, ds, circuitList,
                                 parameters.get('minProbClip', 1e-4),
                                 parameters.get('probClipInterval', (-1e6, 1e6)),
                                 parameters.get('radius', 1e-4),
                                 opLabelAliases=parameters.get('opLabelAliases', None),
                                 evaltree_cache=evaltree_cache, comm=comm)
        fitQty = 2 * (maxLogL - logL)
    return fitQty


def get_robust_scaling(scale_typ, model, ds, circuitList, parameters, evaltree_cache, comm, memLimit):
    """
    Get the per-circuit data scaling ("weights") for a given type of robust-data-scaling.
    TODO: docstring - more details
    """

    fitQty = _get_fit_qty(model, ds, circuitList, parameters, evaltree_cache, comm, memLimit)
    #Note: fitQty[iCircuit] gives fit quantity for a single circuit, aggregated over outcomes.

    expected = (len(ds.get_outcome_labels()) - 1)  # == "k"
    dof_per_box = expected; nboxes = len(circuitList)
    pc = 0.05  # hardcoded (1 - confidence level) for now -- make into advanced option w/default

    circuitWeights = {}
    if scale_typ in ("robust", "Robust"):
        # Robust scaling V1: drastically scale down weights of especially bad sequences
        threshold = _np.ceil(_chi2.ppf(1 - pc / nboxes, dof_per_box))
        for i, opstr in enumerate(circuitList):
            if fitQty[i] > threshold:
                circuitWeights[opstr] = expected / fitQty[i]  # scaling factor

    elif scale_typ in ("robust+", "Robust+"):
        # Robust scaling V2: V1 + rescale to desired chi2 distribution without reordering
        threshold = _np.ceil(_chi2.ppf(1 - pc / nboxes, dof_per_box))
        scaled_fitQty = fitQty.copy()
        for i, opstr in enumerate(circuitList):
            if fitQty[i] > threshold:
                circuitWeights[opstr] = expected / fitQty[i]  # scaling factor
                scaled_fitQty[i] = expected  # (fitQty[i]*circuitWeights[opstr])

        N = len(fitQty)
        percentiles = [_chi2.ppf((i + 1) / (N + 1), dof_per_box) for i in range(N)]
        for iBin, i in enumerate(_np.argsort(scaled_fitQty)):
            opstr = circuitList[i]
            fit, expected = scaled_fitQty[i], percentiles[iBin]
            if fit > expected:
                if opstr in circuitWeights: circuitWeights[opstr] *= expected / fit
                else: circuitWeights[opstr] = expected / fit

    return circuitWeights


def get_wildcard_budget(model, ds, circuitsToUse, parameters, evaltree_cache, comm, memLimit, verbosity):
    printer = _objs.VerbosityPrinter.build_printer(verbosity, comm)
    fitQty = _get_fit_qty(model, ds, circuitsToUse, parameters, evaltree_cache, comm, memLimit)

    printer.log("******************* Adding Wildcard Budget **************************")

    # Approach: we create an objective function that, for a given Wvec, computes:
    # (amt_of_2DLogL over threshold) + (amt of "red-box": per-outcome 2DlogL over threshold) + eta*|Wvec|_1                                     # noqa
    # and minimize this for different eta (binary search) to find that largest eta for which the
    # first two terms is are zero.  This Wvec is our keeper.
    if evaltree_cache and 'evTree' in evaltree_cache:
        #use cache dictionary to speed multiple calls which use
        # the same model, operation sequences, comm, memlim, etc.
        evTree = evaltree_cache['evTree']
    else:
        # Note: simplify_circuits doesn't support aliased dataset (yet)
        dstree = ds if (parameters.get('opLabelAliases', None) is None) else None
        evTree, _, _, lookup, outcomes_lookup = \
            model.bulk_evaltree_from_resources(
                circuitsToUse, None, memLimit, "deriv", ['bulk_fill_probs'], dstree)

        #Fill cache dict if one was given
        if evaltree_cache is not None:
            evaltree_cache['evTree'] = evTree
            evaltree_cache['lookup'] = lookup
            evaltree_cache['outcomes_lookup'] = outcomes_lookup

    nDataParams = ds.get_degrees_of_freedom(circuitsToUse)  # number of independent parameters
    # in dataset (max. model # of params)
    nModelParams = model.num_params()  # just use total number of params
    percentile = 0.05; nBoxes = len(circuitsToUse)
    twoDeltaLogL_threshold = _chi2.ppf(1 - percentile, nDataParams - nModelParams)
    redbox_threshold = _chi2.ppf(1 - percentile / nBoxes, 1)
    eta = 10.0  # some default starting value - this *shouldn't* really matter
    #print("DB2: ",twoDeltaLogL_threshold,redbox_threshold)

    objective = parameters.get('objective', 'logl')
    assert(objective == "logl"), "Can only use wildcard scaling with 'logl' objective!"
    twoDeltaLogL_terms = fitQty
    twoDeltaLogL = sum(twoDeltaLogL_terms)

    budget = _wild.PrimitiveOpsWildcardBudget(model.get_primitive_op_labels(), add_spam=True,
                                              start_budget=0.0)

    if twoDeltaLogL <= twoDeltaLogL_threshold \
       and sum(_np.clip(twoDeltaLogL_terms - redbox_threshold, 0, None)) < 1e-6:
        printer.log("No need to add budget!")
        Wvec = _np.zeros(len(model.get_primitive_op_labels()), 'd')
    else:
        pci = parameters.get('probClipInterval', (-1e6, 1e6))
        min_p = parameters.get('minProbClip', 1e-4)
        a = parameters.get('radius', 1e-4)

        loglFn = _objfns.LogLFunction.simple_init(model, ds, circuitsToUse, min_p, pci, a,
                                                  poissonPicture=True, evaltree_cache=evaltree_cache,
                                                  comm=comm)
        sqrt_dlogl_elements = loglFn.fn(model.to_vector())  # must evaluate loglFn before using it to init loglWCFn
        loglWCFn = _objfns.LogLWildcardFunction(loglFn, model.to_vector(), budget)

        nCircuits = len(circuitsToUse)
        dlogl_terms = _np.empty(nCircuits, 'd')
        # b/c loglFn gives sqrt of terms (for use in leastsq optimizer)
        dlogl_elements = sqrt_dlogl_elements**2
        for i in range(nCircuits):
            dlogl_terms[i] = _np.sum(dlogl_elements[loglFn.lookup[i]], axis=0)
        print("INITIAL 2DLogL (before any wildcard) = ", sum(2 * dlogl_terms), max(2 * dlogl_terms))
        print("THRESHOLDS = ", twoDeltaLogL_threshold, redbox_threshold, nBoxes)

        def _wildcard_objective_firstTerms(Wv):
            dlogl_elements = loglWCFn.fn(Wv)**2  # b/c loglWCFn gives sqrt of terms (for use in leastsq optimizer)
            for i in range(nCircuits):
                dlogl_terms[i] = _np.sum(dlogl_elements[loglFn.lookup[i]], axis=0)

            twoDLogL_terms = 2 * dlogl_terms
            twoDLogL = sum(twoDLogL_terms)
            return max(0, twoDLogL - twoDeltaLogL_threshold) \
                + sum(_np.clip(twoDLogL_terms - redbox_threshold, 0, None))

        nIters = 0
        Wvec_init = budget.to_vector()
        print("INITIAL Wildcard budget = ", str(budget))

        # Find a value of eta that is small enough that the "first terms" are 0.
        while nIters < 10:
            printer.log("  Iter %d: trying eta = %g" % (nIters, eta))

            def _wildcard_objective(Wv):
                return _wildcard_objective_firstTerms(Wv) + eta * _np.linalg.norm(Wv, ord=1)

            #TODO REMOVE
            #import bpdb; bpdb.set_trace()
            #Wvec_init[:] = 0.0; print("TEST budget 0\n", _wildcard_objective(Wvec_init))
            #Wvec_init[:] = 1e-5; print("TEST budget 1e-5\n", _wildcard_objective(Wvec_init))
            #Wvec_init[:] = 0.1; print("TEST budget 0.1\n", _wildcard_objective(Wvec_init))
            #Wvec_init[:] = 1.0; print("TEST budget 1.0\n", _wildcard_objective(Wvec_init))

            if printer.verbosity > 1:
                printer.log(("NOTE: optimizing wildcard budget with verbose progress messages"
                             " - this *increases* the runtime significantly."), 2)

                def callbackF(Wv):
                    a, b = _wildcard_objective_firstTerms(Wv), eta * _np.linalg.norm(Wv, ord=1)
                    printer.log('wildcard: misfit + L1_reg = %.3g + %.3g = %.3g Wvec=%s' % (a, b, a + b, str(Wv)), 2)
            else:
                callbackF = None
            soln = _spo.minimize(_wildcard_objective, Wvec_init,
                                 method='Nelder-Mead', callback=callbackF, tol=1e-6)
            if not soln.success:
                _warnings.warn("Nelder-Mead optimization failed to converge!")
            Wvec = soln.x
            firstTerms = _wildcard_objective_firstTerms(Wvec)
            #printer.log("  Firstterms value = %g" % firstTerms)
            meets_conditions = bool(firstTerms < 1e-4)  # some zero-tolerance here
            if meets_conditions:  # try larger eta
                break
            else:  # nonzero objective => take Wvec as new starting point; try smaller eta
                Wvec_init = Wvec
                eta /= 10

            printer.log("  Trying eta = %g" % eta)
            nIters += 1

    budget.from_vector(Wvec)
    #print("Wildcard budget found for Wvec = ",Wvec)
    #print("FINAL Wildcard budget = ", str(budget))
    printer.log(str(budget))
    return budget


def reoptimize_with_weights(model, ds, circuitList, circuitWeights, objective, opt_args, verbosity):
    """
    TODO: docstring
    """
    printer = _objs.VerbosityPrinter.build_printer(verbosity)

    #convert weights dict to an array for do_XXX methods below
    circuitWeightsArray = _np.ones(len(circuitList), 'd')
    gsindx = {opstr: i for i, opstr in enumerate(circuitList)}
    for opstr, weight in circuitWeights.items():
        circuitWeightsArray[gsindx[opstr]] = weight

    reopt_args = dict(dataset=ds,
                      startModel=model,
                      circuitsToUse=circuitList,
                      circuitWeights=circuitWeightsArray,
                      verbosity=printer - 1)
    for x in ('maxiter', 'tol', 'cptp_penalty_factor', 'spam_penalty_factor',
              'probClipInterval', 'check', 'opLabelAliases',
              'memLimit', 'comm', 'evaltree_cache', 'distributeMethod', 'profiler'):
        reopt_args[x] = opt_args[x]

    printer.log("--- Re-optimizing %s after robust data scaling ---" % objective)
    if objective == "chi2":
        reopt_args['useFreqWeightedChiSq'] = opt_args['useFreqWeightedChiSq']
        reopt_args['minProbClipForWeighting'] = opt_args['minProbClipForWeighting']
        reopt_args['check_jacobian'] = opt_args['check_jacobian']
        _, mdl_reopt = _alg.do_mc2gst(**reopt_args)

    elif objective == "logl":
        reopt_args['minProbClip'] = opt_args['minProbClip']
        reopt_args['radius'] = opt_args['radius']
        _, mdl_reopt = _alg.do_mlgst(**reopt_args)

    else: raise ValueError("Invalid objective '%s' for robust data scaling reopt" % objective)

    return mdl_reopt

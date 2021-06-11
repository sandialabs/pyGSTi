"""
End-to-end functions for performing long-sequence GST
"""
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

import os as _os
import pickle as _pickle

from .. import construction as _construction
from .. import io as _io
from .. import baseobjs as _baseobjs
from .. import protocols as _proto
from ..forwardsims.matrixforwardsim import MatrixForwardSimulator as _MatrixFSim
from ..objectivefns import objectivefns as _objfns
from ..baseobjs.advancedoptions import GSTAdvancedOptions as _GSTAdvancedOptions

ROBUST_SUFFIX_LIST = [".robust", ".Robust", ".robust+", ".Robust+"]
DEFAULT_BAD_FIT_THRESHOLD = 2.0


def run_model_test(model_filename_or_object,
                   data_filename_or_set, target_model_filename_or_object,
                   prep_fiducial_list_or_filename, meas_fiducial_list_or_filename,
                   germs_list_or_filename, max_lengths, gauge_opt_params=None,
                   advanced_options=None, comm=None, mem_limit=None,
                   output_pkl=None, verbosity=2):
    """
    Compares a :class:`Model`'s predictions to a `DataSet` using GST-like circuits.

    This routine tests a Model model against a DataSet using a specific set of
    structured, GST-like circuits (given by fiducials, max_lengths and germs).
    In particular, circuits are constructed by repeating germ strings an integer
    number of times such that the length of the repeated germ is less than or equal to
    the maximum length set in max_lengths.  Each string thus constructed is
    sandwiched between all pairs of (preparation, measurement) fiducial sequences.

    `model_filename_or_object` is used directly (without any optimization) as the
    the model estimate at each maximum-length "iteration".  The model
    is given a trivial `default_gauge_group` so that it is not altered
    during any gauge optimization step.

    A :class:`~pygsti.protocols.ModelEstimateResults` object is returned, which
    encapsulates the model estimate and related parameters, and can be used with
    report-generation routines.

    Parameters
    ----------
    model_filename_or_object : Model or string
        The model model, specified either directly or by the filename of a
        model file (text format).

    data_filename_or_set : DataSet or string
        The data set object to use for the analysis, specified either directly
        or by the filename of a dataset file (assumed to be a pickled `DataSet`
        if extension is 'pkl' otherwise assumed to be in pyGSTi's text format).

    target_model_filename_or_object : Model or string
        The target model, specified either directly or by the filename of a
        model file (text format).

    prep_fiducial_list_or_filename : (list of Circuits) or string
        The state preparation fiducial circuits, specified either directly
        or by the filename of a circuit list file (text format).

    meas_fiducial_list_or_filename : (list of Circuits) or string or None
        The measurement fiducial circuits, specified either directly or by
        the filename of a circuit list file (text format).  If ``None``,
        then use the same strings as specified by prep_fiducial_list_or_filename.

    germs_list_or_filename : (list of Circuits) or string
        The germ circuits, specified either directly or by the filename of a
        circuit list file (text format).

    max_lengths : list of ints
        List of integers, one per LSGST iteration, which set truncation lengths
        for repeated germ strings.  The list of circuits for the i-th LSGST
        iteration includes the repeated germs truncated to the L-values *up to*
        and including the i-th one.

    gauge_opt_params : dict, optional
        A dictionary of arguments to :func:`gaugeopt_to_target`, specifying
        how the final gauge optimization should be performed.  The keys and
        values of this dictionary may correspond to any of the arguments
        of :func:`gaugeopt_to_target` *except* for the first `model`
        argument, which is specified internally.  The `target_model` argument,
        *can* be set, but is specified internally when it isn't.  If `None`,
        then the dictionary `{'item_weights': {'gates':1.0, 'spam':0.001}}`
        is used.  If `False`, then then *no* gauge optimization is performed.

    advanced_options : dict, optional
        Specifies advanced options most of which deal with numerical details of
        the objective function or expert-level functionality.

    comm : mpi4py.MPI.Comm, optional
        When not ``None``, an MPI communicator for distributing the computation
        across multiple processors.

    mem_limit : int or None, optional
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
    printer = _baseobjs.VerbosityPrinter.create_printer(verbosity, comm)
    ds = _load_dataset(data_filename_or_set, comm, printer)
    advanced_options = _GSTAdvancedOptions(advanced_options or {})

    exp_design = _proto.StandardGSTDesign(target_model_filename_or_object,
                                          prep_fiducial_list_or_filename, meas_fiducial_list_or_filename,
                                          germs_list_or_filename, max_lengths,
                                          advanced_options.get('germ_length_limits', None),
                                          None, 1, None,  # fidPairs, keepFraction, keepSeed
                                          advanced_options.get('include_lgst', True),
                                          advanced_options.get('nested_circuit_lists', True),
                                          advanced_options.get('string_manipulation_rules', None),
                                          advanced_options.get('op_label_aliases', None),
                                          ds, 'drop', verbosity=printer)
    # Note: no advancedOptions['truncScheme'] support anymore

    data = _proto.ProtocolData(exp_design, ds)

    gopt_suite = {'go0': gauge_opt_params} if gauge_opt_params else None
    builder = _objfns.ObjectiveFunctionBuilder.create_from(advanced_options.get('objective', 'logl'),
                                                           advanced_options.get('use_freq_weighted_chi2', False))
    _update_objfn_builders([builder], advanced_options)

    #Create the protocol
    proto = _proto.ModelTest(_load_model(model_filename_or_object), None, gopt_suite, None,
                             builder, _get_badfit_options(advanced_options),
                             advanced_options.get('set trivial gauge group', True), printer,
                             name=advanced_options.get('estimate_label', None))

    #Set more advanced options
    proto.profile = advanced_options.get('profile', 1)
    proto.oplabel_aliases = advanced_options.get('op_label_aliases', None)
    proto.circuit_weights = advanced_options.get('circuit_weights', None)
    proto.unreliable_ops = advanced_options.get('unreliable_ops', ['Gcnot', 'Gcphase', 'Gms', 'Gcn', 'Gcx', 'Gcz'])

    results = proto.run(data, mem_limit, comm)
    _output_to_pickle(results, output_pkl, comm)
    return results


def run_linear_gst(data_filename_or_set, target_model_filename_or_object,
                   prep_fiducial_list_or_filename, meas_fiducial_list_or_filename,
                   gauge_opt_params=None, advanced_options=None, comm=None,
                   mem_limit=None, output_pkl=None, verbosity=2):
    """
    Perform Linear Gate Set Tomography (LGST).

    This function differs from the lower level :function:`run_lgst` function
    in that it may perform a post-LGST gauge optimization and this routine
    returns a :class:`Results` object containing the LGST estimate.

    Overall, this is a high-level driver routine which can be used similarly
    to :function:`run_long_sequence_gst`  whereas `run_lgst` is a low-level
    routine used when building your own algorithms.

    Parameters
    ----------
    data_filename_or_set : DataSet or string
        The data set object to use for the analysis, specified either directly
        or by the filename of a dataset file (assumed to be a pickled `DataSet`
        if extension is 'pkl' otherwise assumed to be in pyGSTi's text format).

    target_model_filename_or_object : Model or string
        The target model, specified either directly or by the filename of a
        model file (text format).

    prep_fiducial_list_or_filename : (list of Circuits) or string
        The state preparation fiducial circuits, specified either directly
        or by the filename of a circuit list file (text format).

    meas_fiducial_list_or_filename : (list of Circuits) or string or None
        The measurement fiducial circuits, specified either directly or by
        the filename of a circuit list file (text format).  If ``None``,
        then use the same strings as specified by prep_fiducial_list_or_filename.

    gauge_opt_params : dict, optional
        A dictionary of arguments to :func:`gaugeopt_to_target`, specifying
        how the final gauge optimization should be performed.  The keys and
        values of this dictionary may correspond to any of the arguments
        of :func:`gaugeopt_to_target` *except* for the first `model`
        argument, which is specified internally.  The `target_model` argument,
        *can* be set, but is specified internally when it isn't.  If `None`,
        then the dictionary `{'item_weights': {'gates':1.0, 'spam':0.001}}`
        is used.  If `False`, then then *no* gauge optimization is performed.

    advanced_options : dict, optional
        Specifies advanced options most of which deal with numerical details of
        the objective function or expert-level functionality.  See
        :function:`run_long_sequence_gst`.

    comm : mpi4py.MPI.Comm, optional
        When not ``None``, an MPI communicator for distributing the computation
        across multiple processors.  In this LGST case, this is just the gauge
        optimization.

    mem_limit : int or None, optional
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
    printer = _baseobjs.VerbosityPrinter.create_printer(verbosity, comm)
    advanced_options = _GSTAdvancedOptions(advanced_options or {})
    ds = _load_dataset(data_filename_or_set, comm, printer)

    target_model = _load_model(target_model_filename_or_object)
    germs = _construction.to_circuits([()] + [(gl,) for gl in target_model.operations.keys()])  # just the single gates
    max_lengths = [1]  # we only need maxLength == 1 when doing LGST

    exp_design = _proto.StandardGSTDesign(target_model, prep_fiducial_list_or_filename, meas_fiducial_list_or_filename,
                                          germs, max_lengths,
                                          sequenceRules=advanced_options.get('string_manipulation_rules', None),
                                          op_label_aliases=advanced_options.get('op_label_aliases', None),
                                          dscheck=ds, actionIfMissing='raise', verbosity=printer)

    data = _proto.ProtocolData(exp_design, ds)

    if gauge_opt_params is None:
        gauge_opt_params = {'item_weights': {'gates': 1.0, 'spam': 0.001}}
    gopt_suite = {'go0': gauge_opt_params} if gauge_opt_params else None

    proto = _proto.LinearGateSetTomography(target_model, gopt_suite, None,
                                           _get_badfit_options(advanced_options), printer,
                                           name=advanced_options.get('estimate_label', None))
    proto.profile = advanced_options.get('profile', 1)
    proto.record_output = advanced_options.get('record_output', 1)
    proto.oplabels = advanced_options.get('op_labels', 'default')
    proto.oplabel_aliases = advanced_options.get('op_label_aliases', None)
    proto.unreliable_ops = advanced_options.get('unreliable_ops', ['Gcnot', 'Gcphase', 'Gms', 'Gcn', 'Gcx', 'Gcz'])

    results = proto.run(data, mem_limit, comm)
    _output_to_pickle(results, output_pkl, comm)
    return results


def run_long_sequence_gst(data_filename_or_set, target_model_filename_or_object,
                          prep_fiducial_list_or_filename, meas_fiducial_list_or_filename,
                          germs_list_or_filename, max_lengths, gauge_opt_params=None,
                          advanced_options=None, comm=None, mem_limit=None,
                          output_pkl=None, verbosity=2):
    """
    Perform long-sequence GST (LSGST).

    This analysis fits a model (`target_model_filename_or_object`) to data
    (`data_filename_or_set`) using the outcomes from periodic GST circuits
    constructed by repeating germ strings an integer number of times such that
    the length of the repeated germ is less than or equal to the maximum length
    set in `max_lengths`.  When LGST is applicable (i.e. for explicit models
    with full or TP parameterizations), the LGST estimate of the gates is computed,
    gauge optimized, and used as a starting seed for the remaining optimizations.

    LSGST iterates ``len(max_lengths)`` times, optimizing the chi2 using successively
    larger sets of circuits.  On the i-th iteration, the repeated germs sequences
    limited by ``max_lengths[i]`` are included in the growing set of circuits
    used by LSGST.  The final iteration maximizes the log-likelihood.

    Once computed, the model estimates are optionally gauge optimized as
    directed by `gauge_opt_params`.  A :class:`~pygsti.protocols.ModelEstimateResults`
    object is returned, which encapsulates the input and outputs of this GST
    analysis, and can generate final end-user output such as reports and
    presentations.

    Parameters
    ----------
    data_filename_or_set : DataSet or string
        The data set object to use for the analysis, specified either directly
        or by the filename of a dataset file (assumed to be a pickled `DataSet`
        if extension is 'pkl' otherwise assumed to be in pyGSTi's text format).

    target_model_filename_or_object : Model or string
        The target model, specified either directly or by the filename of a
        model file (text format).

    prep_fiducial_list_or_filename : (list of Circuits) or string
        The state preparation fiducial circuits, specified either directly
        or by the filename of a circuit list file (text format).

    meas_fiducial_list_or_filename : (list of Circuits) or string or None
        The measurement fiducial circuits, specified either directly or by
        the filename of a circuit list file (text format).  If ``None``,
        then use the same strings as specified by prep_fiducial_list_or_filename.

    germs_list_or_filename : (list of Circuits) or string
        The germ circuits, specified either directly or by the filename of a
        circuit list file (text format).

    max_lengths : list of ints
        List of integers, one per LSGST iteration, which set truncation lengths
        for repeated germ strings.  The list of circuits for the i-th LSGST
        iteration includes the repeated germs truncated to the L-values *up to*
        and including the i-th one.

    gauge_opt_params : dict, optional
        A dictionary of arguments to :func:`gaugeopt_to_target`, specifying
        how the final gauge optimization should be performed.  The keys and
        values of this dictionary may correspond to any of the arguments
        of :func:`gaugeopt_to_target` *except* for the first `model`
        argument, which is specified internally.  The `target_model` argument,
        *can* be set, but is specified internally when it isn't.  If `None`,
        then the dictionary `{'item_weights': {'gates':1.0, 'spam':0.001}}`
        is used.  If `False`, then then *no* gauge optimization is performed.

    advanced_options : dict, optional
        Specifies advanced options most of which deal with numerical details of
        the objective function or expert-level functionality.  The allowed keys
        and values include:
        - objective = {'chi2', 'logl'}
        - op_labels = list of strings
        - circuit_weights = dict or None
        - starting_point = "LGST-if-possible" (default), "LGST", or "target"
        - depolarize_start = float (default == 0)
        - randomize_start = float (default == 0)
        - contract_start_to_cptp = True / False (default)
        - cptpPenaltyFactor = float (default = 0)
        - tolerance = float or dict w/'relx','relf','f','jac','maxdx' keys
        - max_iterations = int
        - finitediff_iterations = int
        - min_prob_clip = float
        - min_prob_clip_for_weighting = float (default == 1e-4)
        - prob_clip_interval = tuple (default == (-1e6,1e6)
        - radius = float (default == 1e-4)
        - use_freq_weighted_chi2 = True / False (default)
        - XX nested_circuit_lists = True (default) / False
        - XX include_lgst = True / False (default is True)
        - distribute_method = "default", "circuits" or "deriv"
        - profile = int (default == 1)
        - check = True / False (default)
        - XX op_label_aliases = dict (default = None)
        - always_perform_mle = bool (default = False)
        - only_perform_mle = bool (default = False)
        - XX truncScheme = "whole germ powers" (default) or "truncated germ powers"
                          or "length as exponent"
        - appendTo = Results (default = None)
        - estimateLabel = str (default = "default")
        - XX missingDataAction = {'drop','raise'} (default = 'drop')
        - XX string_manipulation_rules = list of (find,replace) tuples
        - germ_length_limits = dict of form {germ: maxlength}
        - record_output = bool (default = True)
        - timeDependent = bool (default = False)

    comm : mpi4py.MPI.Comm, optional
        When not ``None``, an MPI communicator for distributing the computation
        across multiple processors.

    mem_limit : int or None, optional
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
    printer = _baseobjs.VerbosityPrinter.create_printer(verbosity, comm)
    advanced_options = _GSTAdvancedOptions(advanced_options or {})
    ds = _load_dataset(data_filename_or_set, comm, printer)

    exp_design = _proto.StandardGSTDesign(target_model_filename_or_object,
                                          prep_fiducial_list_or_filename, meas_fiducial_list_or_filename,
                                          germs_list_or_filename, max_lengths,
                                          advanced_options.get('germ_length_limits', None),
                                          None, 1, None,  # fidPairs, keepFraction, keepSeed
                                          advanced_options.get('include_lgst', True),
                                          advanced_options.get('nested_circuit_lists', True),
                                          advanced_options.get('string_manipulation_rules', None),
                                          advanced_options.get('op_label_aliases', None),
                                          ds, 'drop', verbosity=printer)

    data = _proto.ProtocolData(exp_design, ds)

    if gauge_opt_params is None:
        gauge_opt_params = {'item_weights': {'gates': 1.0, 'spam': 0.001}}
    gopt_suite = {'go0': gauge_opt_params} if gauge_opt_params else None
    proto = _proto.GateSetTomography(_get_gst_initial_model(advanced_options), gopt_suite, None,
                                     _get_gst_builders(advanced_options),
                                     _get_optimizer(advanced_options, exp_design),
                                     _get_badfit_options(advanced_options), printer)

    proto.profile = advanced_options.get('profile', 1)
    proto.record_output = advanced_options.get('record_output', 1)
    proto.distribute_method = advanced_options.get('distribute_method', "default")
    proto.oplabel_aliases = advanced_options.get('op_label_aliases', None)
    proto.circuit_weights = advanced_options.get('circuit_weights', None)
    proto.unreliable_ops = advanced_options.get('unreliable_ops', ['Gcnot', 'Gcphase', 'Gms', 'Gcn', 'Gcx', 'Gcz'])

    results = proto.run(data, mem_limit, comm)
    _output_to_pickle(results, output_pkl, comm)
    return results


def run_long_sequence_gst_base(data_filename_or_set, target_model_filename_or_object,
                               lsgst_lists, gauge_opt_params=None,
                               advanced_options=None, comm=None, mem_limit=None,
                               output_pkl=None, verbosity=2):
    """
    A more fundamental interface for performing end-to-end GST.

    Similar to :func:`run_long_sequence_gst` except this function takes
    `lsgst_lists`, a list of either raw circuit lists or of
    :class:`PlaquetteGridCircuitStructure` objects to define which circuits
    are used on each GST iteration.

    Parameters
    ----------
    data_filename_or_set : DataSet or string
        The data set object to use for the analysis, specified either directly
        or by the filename of a dataset file (assumed to be a pickled `DataSet`
        if extension is 'pkl' otherwise assumed to be in pyGSTi's text format).

    target_model_filename_or_object : Model or string
        The target model, specified either directly or by the filename of a
        model file (text format).

    lsgst_lists : list of lists or PlaquetteGridCircuitStructure(s)
        An explicit list of either the raw circuit lists to be used in
        the analysis or of :class:`PlaquetteGridCircuitStructure` objects,
        which additionally contain the structure of a set of circuits.
        A single `PlaquetteGridCircuitStructure` object can also be given,
        which is equivalent to passing a list of successive L-value truncations
        of this object (e.g. if the object has `Ls = [1,2,4]` then this is like
        passing a list of three `PlaquetteGridCircuitStructure` objects w/truncations
        `[1]`, `[1,2]`, and `[1,2,4]`).

    gauge_opt_params : dict, optional
        A dictionary of arguments to :func:`gaugeopt_to_target`, specifying
        how the final gauge optimization should be performed.  The keys and
        values of this dictionary may correspond to any of the arguments
        of :func:`gaugeopt_to_target` *except* for the first `model`
        argument, which is specified internally.  The `target_model` argument,
        *can* be set, but is specified internally when it isn't.  If `None`,
        then the dictionary `{'item_weights': {'gates':1.0, 'spam':0.001}}`
        is used.  If `False`, then then *no* gauge optimization is performed.

    advanced_options : dict, optional
        Specifies advanced options most of which deal with numerical details of
        the objective function or expert-level functionality.  See
        :func:`run_long_sequence_gst` for a list of the allowed keys, with the
        exception  "nested_circuit_lists", "op_label_aliases",
        "include_lgst", and "truncScheme".

    comm : mpi4py.MPI.Comm, optional
        When not ``None``, an MPI communicator for distributing the computation
        across multiple processors.

    mem_limit : int or None, optional
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
    printer = _baseobjs.VerbosityPrinter.create_printer(verbosity, comm)
    advanced_options = advanced_options or {}

    exp_design = _proto.GateSetTomographyDesign(target_model_filename_or_object, lsgst_lists)

    ds = _load_dataset(data_filename_or_set, comm, printer)
    data = _proto.ProtocolData(exp_design, ds)

    if gauge_opt_params is None:
        gauge_opt_params = {'item_weights': {'gates': 1.0, 'spam': 0.001}}
    gopt_suite = {'go0': gauge_opt_params} if gauge_opt_params else None

    proto = _proto.GateSetTomography(_get_gst_initial_model(advanced_options), gopt_suite, None,
                                     _get_gst_builders(advanced_options),
                                     _get_optimizer(advanced_options, exp_design),
                                     _get_badfit_options(advanced_options), printer,
                                     name=advanced_options.get('estimate_label', None))

    proto.profile = advanced_options.get('profile', 1)
    proto.record_output = advanced_options.get('record_output', 1)
    proto.distribute_method = advanced_options.get('distribute_method', "default")
    proto.oplabel_aliases = advanced_options.get('op_label_aliases', None)
    proto.circuit_weights = advanced_options.get('circuit_weights', None)
    proto.unreliable_ops = advanced_options.get('unreliable_ops', ['Gcnot', 'Gcphase', 'Gms', 'Gcn', 'Gcx', 'Gcz'])

    results = proto.run(data, mem_limit, comm)
    _output_to_pickle(results, output_pkl, comm)
    return results


def run_stdpractice_gst(data_filename_or_set, target_model_filename_or_object,
                        prep_fiducial_list_or_filename, meas_fiducial_list_or_filename,
                        germs_list_or_filename, max_lengths, modes="TP,CPTP,Target",
                        gaugeopt_suite='stdgaugeopt',
                        gaugeopt_target=None, models_to_test=None, comm=None, mem_limit=None,
                        advanced_options=None, output_pkl=None, verbosity=2):
    """
    Perform end-to-end GST analysis using standard practices.

    This routines is an even higher-level driver than
    :func:`run_long_sequence_gst`.  It performs bottled, typically-useful,
    runs of long sequence GST on a dataset.  This essentially boils down
    to running :func:`run_long_sequence_gst` one or more times using different
    model parameterizations, and performing commonly-useful gauge
    optimizations, based only on the high-level `modes` argument.

    Parameters
    ----------
    data_filename_or_set : DataSet or string
        The data set object to use for the analysis, specified either directly
        or by the filename of a dataset file (assumed to be a pickled `DataSet`
        if extension is 'pkl' otherwise assumed to be in pyGSTi's text format).

    target_model_filename_or_object : Model or string
        The target model, specified either directly or by the filename of a
        model file (text format).

    prep_fiducial_list_or_filename : (list of Circuits) or string
        The state preparation fiducial circuits, specified either directly
        or by the filename of a circuit list file (text format).

    meas_fiducial_list_or_filename : (list of Circuits) or string or None
        The measurement fiducial circuits, specified either directly or by
        the filename of a circuit list file (text format).  If ``None``,
        then use the same strings as specified by prep_fiducial_list_or_filename.

    germs_list_or_filename : (list of Circuits) or string
        The germ circuits, specified either directly or by the filename of a
        circuit list file (text format).

    max_lengths : list of ints
        List of integers, one per LSGST iteration, which set truncation lengths
        for repeated germ strings.  The list of circuits for the i-th LSGST
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
        - <model> : any key in the `models_to_test` argument

    gaugeopt_suite : str or list or dict, optional
        Specifies which gauge optimizations to perform on each estimate.  A
        string or list of strings (see below) specifies built-in sets of gauge
        optimizations, otherwise `gaugeopt_suite` should be a dictionary of
        gauge-optimization parameter dictionaries, as specified by the
        `gauge_opt_params` argument of :func:`run_long_sequence_gst`.  The key
        names of `gaugeopt_suite` then label the gauge optimizations within
        the resuling `Estimate` objects.  The built-in suites are:

          - "single" : performs only a single "best guess" gauge optimization.
          - "varySpam" : varies spam weight and toggles SPAM penalty (0 or 1).
          - "varySpamWt" : varies spam weight but no SPAM penalty.
          - "varyValidSpamWt" : varies spam weight with SPAM penalty == 1.
          - "toggleValidSpam" : toggles spame penalty (0 or 1); fixed SPAM wt.
          - "unreliable2Q" : adds branch to a spam suite that weights 2Q gates less
          - "none" : no gauge optimizations are performed.

    gaugeopt_target : Model, optional
        If not None, a model to be used as the "target" for gauge-
        optimization (only).  This argument is useful when you want to
        gauge optimize toward something other than the *ideal* target gates
        given by `target_model_filename_or_object`, which are used as the default when
        `gaugeopt_target` is None.

    models_to_test : dict, optional
        A dictionary of Model objects representing (gate-set) models to
        test against the data.  These Models are essentially hypotheses for
        which (if any) model generated the data.  The keys of this dictionary
        can (and must, to actually test the models) be used within the comma-
        separate list given by the `modes` argument.

    comm : mpi4py.MPI.Comm, optional
        When not ``None``, an MPI communicator for distributing the computation
        across multiple processors.

    mem_limit : int or None, optional
        A rough memory limit in bytes which restricts the amount of memory
        used (per core when run on multi-CPUs).

    advanced_options : dict, optional
        Specifies advanced options most of which deal with numerical details of the
        objective function or expert-level functionality. See :func:`run_long_sequence_gst`
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
    printer = _baseobjs.VerbosityPrinter.create_printer(verbosity, comm)
    if advanced_options and 'all' in advanced_options and len(advanced_options) == 1:
        advanced_options = advanced_options['all']  # backward compatibility
    advanced_options = _GSTAdvancedOptions(advanced_options or {})
    ds = _load_dataset(data_filename_or_set, comm, printer)

    exp_design = _proto.StandardGSTDesign(target_model_filename_or_object,
                                          prep_fiducial_list_or_filename, meas_fiducial_list_or_filename,
                                          germs_list_or_filename, max_lengths,
                                          advanced_options.get('germ_length_limits', None),
                                          None, 1, None,  # fidPairs, keepFraction, keepSeed
                                          advanced_options.get('include_lgst', True),
                                          advanced_options.get('nested_circuit_lists', True),
                                          advanced_options.get('string_manipulation_rules', None),
                                          advanced_options.get('op_label_aliases', None),
                                          ds, 'drop', verbosity=printer)

    ds = _load_dataset(data_filename_or_set, comm, printer)
    data = _proto.ProtocolData(exp_design, ds)
    proto = _proto.StandardGST(modes, gaugeopt_suite, gaugeopt_target, models_to_test,
                               _get_gst_builders(advanced_options),
                               _get_optimizer(advanced_options, exp_design),
                               _get_badfit_options(advanced_options), printer,
                               name=advanced_options.get('estimate_label', None))

    results = proto.run(data, mem_limit, comm)
    _output_to_pickle(results, output_pkl, comm)
    return results


# --- Helper functions ---

def _load_model(model_filename_or_object):
    if isinstance(model_filename_or_object, str):
        return _io.load_model(model_filename_or_object)
    else:
        return model_filename_or_object  # assume a Model object


def _load_dataset(data_filename_or_set, comm, verbosity):
    """Loads a DataSet from the data_filename_or_set argument of functions in this module."""
    printer = _baseobjs.VerbosityPrinter.create_printer(verbosity, comm)
    if isinstance(data_filename_or_set, str):
        if comm is None or comm.Get_rank() == 0:
            if _os.path.splitext(data_filename_or_set)[1] == ".pkl":
                with open(data_filename_or_set, 'rb') as pklfile:
                    ds = _pickle.load(pklfile)
            else:
                ds = _io.load_dataset(data_filename_or_set, True, "aggregate", printer)
            if comm is not None: comm.bcast(ds, root=0)
        else:
            ds = comm.bcast(None, root=0)
    else:
        ds = data_filename_or_set  # assume a Dataset object

    return ds


def _update_objfn_builders(builders, advanced_options):
    def _update_regularization(builder, nm):
        if builder.regularization and nm in builder.regularization and nm in advanced_options:
            builder.regularization[nm] = advanced_options[nm]

    def _update_penalty(builder, nm):
        if builder.penalties and nm in builder.penalties and nm in advanced_options:
            builder.penalties[nm] = advanced_options[nm]

    for builder in builders:
        _update_regularization(builder, 'prob_clip_interval')
        _update_regularization(builder, 'min_prob_clip')
        _update_regularization(builder, 'radius')
        _update_regularization(builder, 'min_prob_clip_for_weighting')
        _update_penalty(builder, 'cptp_penalty_factor')
        _update_penalty(builder, 'spam_penalty_factor')


def _get_badfit_options(advanced_options):
    advanced_options = advanced_options or {}
    old_badfit_options = advanced_options.get('badFitOptions', {})
    assert(set(old_badfit_options.keys()).issubset(('wildcard_budget_includes_spam', 'wildcard_smart_init'))), \
        "Invalid keys in badFitOptions sub-dictionary!"
    return _proto.GSTBadFitOptions(advanced_options.get('bad_fit_threshold', DEFAULT_BAD_FIT_THRESHOLD),
                                   advanced_options.get('on_bad_fit', []),
                                   old_badfit_options.get('wildcard_budget_includes_spam', True),
                                   old_badfit_options.get('wildcard_smart_init', True))


def _output_to_pickle(obj, output_pkl, comm):
    if output_pkl and (comm is None or comm.Get_rank() == 0):
        if isinstance(output_pkl, str):
            with open(output_pkl, 'wb') as pklfile:
                _pickle.dump(obj, pklfile)
        else:
            _pickle.dump(obj, output_pkl)


def _get_gst_initial_model(advanced_options):
    advanced_options = advanced_options or {}
    if advanced_options.get("starting_point", None) is None:
        advanced_options["starting_point"] = "LGST-if-possible"  # to keep backward compatibility
    return _proto.GSTInitialModel(None, advanced_options.get("starting_point", None),
                                  advanced_options.get('depolarize_start', 0),
                                  advanced_options.get('randomize_start', 0),
                                  advanced_options.get('lgst_gaugeopt_tol', 1e-6),
                                  advanced_options.get('contract_start_to_cptp', 0))


def _get_gst_builders(advanced_options):
    advanced_options = advanced_options or {}
    objfn_builders = _proto.GSTObjFnBuilders.create_from(
        advanced_options.get('objective', 'logl'),
        advanced_options.get('use_freq_weighted_chi2', False),
        advanced_options.get('always_perform_mle', False),
        advanced_options.get('only_perform_mle', False))
    _update_objfn_builders(objfn_builders.iteration_builders, advanced_options)
    _update_objfn_builders(objfn_builders.final_builders, advanced_options)
    return objfn_builders


def _get_optimizer(advanced_options, exp_design):
    advanced_options = advanced_options or {}
    default_fditer = 1 if isinstance(exp_design.target_model.sim, _MatrixFSim) else 0
    optimizer = {'maxiter': advanced_options.get('max_iterations', 100000),
                 'tol': advanced_options.get('tolerance', 1e-6),
                 'fditer': advanced_options.get('finitediff_iterations', default_fditer)}
    optimizer.update(advanced_options.get('extra_lm_opts', {}))
    return optimizer

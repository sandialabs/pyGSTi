"""
Functions for generating Direct-(LGST, MC2GST, MLGST) models
"""
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************


from .. import tools as _tools
from .. import construction as _construction
from .. import objects as _objs
from ..modelmembers.operations import FullArbitraryOp as _FullArbitraryOp
from . import core as _core


def model_with_lgst_circuit_estimates(
        circuits_to_estimate, dataset, prep_fiducials, meas_fiducials,
        target_model, include_target_ops=True, op_label_aliases=None,
        guess_model_for_gauge=None, circuit_labels=None, svd_truncate_to=None,
        verbosity=0):
    """
    Constructs a model that contains LGST estimates for `circuits_to_estimate`.

    For each circuit in `circuits_to_estimate`, the constructed model
    contains the LGST estimate for s as separate gate, labeled either by
    the corresponding element of circuit_labels or by the tuple of s itself.

    Parameters
    ----------
    circuits_to_estimate : list of Circuits or tuples
        The circuits to estimate using LGST

    dataset : DataSet
        The data to use for LGST

    prep_fiducials : list of Circuits
       Fiducial circuits used to construct an informationally complete
       effective preparation.

    meas_fiducials : list of Circuits
       Fiducial circuits used to construct an informationally complete
       effective measurement.

    target_model : Model
        A model used by LGST to specify which operation labels should be estimated,
        a guess for which gauge these estimates should be returned in, and
        used to simplify circuits.

    include_target_ops : bool, optional
        If True, the operation labels in target_model will be included in the
        returned model.

    op_label_aliases : dictionary, optional
        Dictionary whose keys are operation label "aliases" and whose values are tuples
        corresponding to what that operation label should be expanded into before querying
        the dataset. Defaults to the empty dictionary (no aliases defined)
        e.g. opLabelAliases['Gx^3'] = ('Gx','Gx','Gx')

    guess_model_for_gauge : Model, optional
        A model used to compute a gauge transformation that is applied to
        the LGST estimates.  This gauge transformation is computed such that
        if the estimated gates matched the model given, then the gate
        matrices would match, i.e. the gauge would be the same as
        the model supplied. Defaults to the target_model.

    circuit_labels : list of strings, optional
        A list of labels in one-to-one correspondence with the
        circuit in `circuits_to_estimate`.  These labels are
        the keys to access the operation matrices in the returned
        Model, i.e. op_matrix = returned_model[op_label]

    svd_truncate_to : int, optional
        The Hilbert space dimension to truncate the operation matrices to using
        a SVD to keep only the largest svdToTruncateTo singular values of
        the I_tildle LGST matrix. Zero means no truncation.
        Defaults to dimension of `target_model`.

    verbosity : int, optional
        Verbosity value to send to run_lgst(...) call.

    Returns
    -------
    Model
        A model containing LGST estimates for all the requested
        circuits and possibly the gates in target_model.
    """
    opLabels = []  # list of operation labels for LGST to estimate
    if op_label_aliases is None: aliases = {}
    else: aliases = op_label_aliases.copy()

    #Add circuits to estimate as aliases
    if circuit_labels is not None:
        assert(len(circuit_labels) == len(circuits_to_estimate))
        for opLabel, opStr in zip(circuit_labels, circuits_to_estimate):
            aliases[opLabel] = opStr.replace_layers_with_aliases(op_label_aliases)
            opLabels.append(opLabel)
    else:
        for opStr in circuits_to_estimate:
            newLabel = 'G' + '.'.join(map(str, tuple(opStr)))
            aliases[newLabel] = opStr.replace_layers_with_aliases(op_label_aliases)  # use circuit tuple as label
            opLabels.append(newLabel)

    #Add target model labels (not aliased) if requested
    if include_target_ops and target_model is not None:
        for targetOpLabel in target_model.operations:
            if targetOpLabel not in opLabels:  # very unlikely that this is false
                opLabels.append(targetOpLabel)

    return _core.run_lgst(dataset, prep_fiducials, meas_fiducials, target_model,
                          opLabels, aliases, guess_model_for_gauge,
                          svd_truncate_to, verbosity)


def direct_lgst_model(circuit_to_estimate, circuit_label, dataset,
                      prep_fiducials, meas_fiducials, target_model,
                      op_label_aliases=None, svd_truncate_to=None, verbosity=0):
    """
    Constructs a model of LGST estimates for target gates and circuit_to_estimate.

    Parameters
    ----------
    circuit_to_estimate : Circuit or tuple
        The single circuit to estimate using LGST

    circuit_label : string
        The label for the estimate of `circuit_to_estimate`.
        i.e. op_matrix = returned_model[op_label]

    dataset : DataSet
        The data to use for LGST

    prep_fiducials : list of Circuits
        Fiducial circuits used to construct an informationally complete
        effective preparation.

    meas_fiducials : list of Circuits
        Fiducial circuits used to construct an informationally complete
        effective measurement.

    target_model : Model
        The target model used by LGST to extract operation labels and an initial gauge

    op_label_aliases : dictionary, optional
        Dictionary whose keys are operation label "aliases" and whose values are tuples
        corresponding to what that operation label should be expanded into before querying
        the dataset. Defaults to the empty dictionary (no aliases defined)
        e.g. opLabelAliases['Gx^3'] = ('Gx','Gx','Gx')

    svd_truncate_to : int, optional
        The Hilbert space dimension to truncate the operation matrices to using
        a SVD to keep only the largest svdToTruncateTo singular values of
        the I_tildle LGST matrix.  Zero means no truncation.
        Defaults to dimension of `target_model`.

    verbosity : int, optional
        Verbosity value to send to run_lgst(...) call.

    Returns
    -------
    Model
        A model containing LGST estimates of `circuit_to_estimate`
        and the gates of `target_model`.
    """
    return model_with_lgst_circuit_estimates(
        [circuit_to_estimate], dataset, prep_fiducials, meas_fiducials, target_model,
        True, op_label_aliases, None, [circuit_label], svd_truncate_to,
        verbosity)


def direct_lgst_models(circuits, dataset, prep_fiducials, meas_fiducials, target_model,
                       op_label_aliases=None, svd_truncate_to=None, verbosity=0):
    """
    Constructs a dictionary with keys == circuits and values == Direct-LGST Models.

    Parameters
    ----------
    circuits : list of Circuit or tuple objects
        The circuits to estimate using LGST.  The elements of this list
        are the keys of the returned dictionary.

    dataset : DataSet
        The data to use for all LGST estimates.

    prep_fiducials : list of Circuits
        Fiducial circuits used to construct an informationally complete
        effective preparation.

    meas_fiducials : list of Circuits
        Fiducial circuits used to construct an informationally complete
        effective measurement.

    target_model : Model
        The target model used by LGST to extract operation labels and an initial gauge

    op_label_aliases : dictionary, optional
        Dictionary whose keys are operation label "aliases" and whose values are tuples
        corresponding to what that operation label should be expanded into before querying
        the dataset. Defaults to the empty dictionary (no aliases defined)
        e.g. opLabelAliases['Gx^3'] = ('Gx','Gx','Gx')

    svd_truncate_to : int, optional
        The Hilbert space dimension to truncate the operation matrices to using
        a SVD to keep only the largest svdToTruncateTo singular values of
        the I_tildle LGST matrix. Zero means no truncation.
        Defaults to dimension of `target_model`.

    verbosity : int, optional
        Verbosity value to send to run_lgst(...) call.

    Returns
    -------
    dict
        A dictionary that relates each circuit to a Model containing the LGST
        estimate of that circuit's action (as a SPAM-less operation sequence)
        stored under the operation label "GsigmaLbl", along with LGST estimates
        of the gates in `target_model`.
    """
    printer = _objs.VerbosityPrinter.create_printer(verbosity)

    directLGSTmodels = {}
    printer.log("--- Direct LGST precomputation ---")
    with printer.progress_logging(1):
        for i, sigma in enumerate(circuits):
            printer.show_progress(i, len(circuits), prefix="--- Computing model for string -", suffix='---')
            directLGSTmodels[sigma] = direct_lgst_model(
                sigma, "GsigmaLbl", dataset, prep_fiducials, meas_fiducials, target_model,
                op_label_aliases, svd_truncate_to, verbosity)
    return directLGSTmodels


def direct_mc2gst_model(circuit_to_estimate, circuit_label, dataset,
                        prep_fiducials, meas_fiducials, target_model,
                        op_label_aliases=None, svd_truncate_to=None,
                        min_prob_clip_for_weighting=1e-4,
                        prob_clip_interval=(-1e6, 1e6), verbosity=0):
    """
    Constructs a model of LSGST estimates for target gates and circuit_to_estimate.

    Starting with a Direct-LGST estimate for circuit_to_estimate, runs LSGST
    using the same strings that LGST would have used to estimate circuit_to_estimate
    and each of the target gates.  That is, LSGST is run with strings of the form:

    1. prep_fiducial
    2. meas_fiducial
    3. prep_fiducial + meas_fiducial
    4. prep_fiducial + single_gate + meas_fiducial
    5. prep_fiducial + circuit_to_estimate + meas_fiducial

    and the resulting Model estimate is returned.

    Parameters
    ----------
    circuit_to_estimate : Circuit
        The single circuit to estimate using LSGST

    circuit_label : string
        The label for the estimate of `circuit_to_estimate`.
        i.e. op_matrix = returned_mode[op_label]

    dataset : DataSet
        The data to use for LGST

    prep_fiducials : list of Circuits
        Fiducial circuits used to construct an informationally complete
        effective preparation.

    meas_fiducials : list of Circuits
        Fiducial circuits used to construct an informationally complete
        effective measurement.

    target_model : Model
        The target model used by LGST to extract operation labels and an initial gauge

    op_label_aliases : dictionary, optional
        Dictionary whose keys are operation label "aliases" and whose values are tuples
        corresponding to what that operation label should be expanded into before querying
        the dataset. Defaults to the empty dictionary (no aliases defined)
        e.g. opLabelAliases['Gx^3'] = ('Gx','Gx','Gx')

    svd_truncate_to : int, optional
        The Hilbert space dimension to truncate the operation matrices to using
        a SVD to keep only the largest svdToTruncateTo singular values of
        the I_tildle LGST matrix. Zero means no truncation.
        Defaults to dimension of `target_model`.

    min_prob_clip_for_weighting : float, optional
        defines the clipping interval for the statistical weight used
        within the chi^2 function (see chi2fn).

    prob_clip_interval : 2-tuple, optional
        (min,max) to clip probabilities to within Model probability
        computation routines (see Model.bulk_fill_probs)

    verbosity : int, optional
        Verbosity value to send to run_lgst(...) and do_mc2gst(...) calls.

    Returns
    -------
    Model
        A model containing LSGST estimates of `circuit_to_estimate`
        and the gates of `target_model`.
    """
    direct_lgst = model_with_lgst_circuit_estimates(
        [circuit_to_estimate], dataset, prep_fiducials, meas_fiducials, target_model,
        True, op_label_aliases, None, [circuit_label], svd_truncate_to, verbosity)

    # LEXICOGRAPHICAL VS MATRIX ORDER
    circuits = prep_fiducials + meas_fiducials + [prepC + measC for prepC in prep_fiducials
                                                  for measC in meas_fiducials]
    for opLabel in direct_lgst.operations:
        circuits.extend([prepC + _objs.Circuit((opLabel,)) + measC
                         for prepC in prep_fiducials for measC in meas_fiducials])

    aliases = {} if (op_label_aliases is None) else op_label_aliases.copy()
    aliases[circuit_label] = circuit_to_estimate.replace_layers_with_aliases(op_label_aliases)

    obuilder = _objs.Chi2Function.builder(regularization={'min_prob_clip_for_weighting': min_prob_clip_for_weighting},
                                          penalties={'prob_clip_interval': prob_clip_interval})
    bulk_circuits = _objs.CircuitList(circuits, aliases)
    _, direct_lsgst = _core.run_gst_fit_simple(dataset, direct_lgst, bulk_circuits, optimizer=None,
                                               objective_function_builder=obuilder, resource_alloc=None,
                                               verbosity=verbosity)

    return direct_lsgst


def direct_mc2gst_models(circuits, dataset, prep_fiducials, meas_fiducials,
                         target_model, op_label_aliases=None,
                         svd_truncate_to=None, min_prob_clip_for_weighting=1e-4,
                         prob_clip_interval=(-1e6, 1e6), verbosity=0):
    """
    Constructs a dictionary with keys == circuits and values == Direct-LSGST Models.

    Parameters
    ----------
    circuits : list of Circuit or tuple objects
        The circuits to estimate using LSGST.  The elements of this list
        are the keys of the returned dictionary.

    dataset : DataSet
        The data to use for all LGST and LSGST estimates.

    prep_fiducials : list of Circuits
        Fiducial circuits used to construct an informationally complete
        effective preparation.

    meas_fiducials : list of Circuits
        Fiducial circuits used to construct an informationally complete
        effective measurement.

    target_model : Model
        The target model used by LGST to extract operation labels and an initial gauge

    op_label_aliases : dictionary, optional
        Dictionary whose keys are operation label "aliases" and whose values are tuples
        corresponding to what that operation label should be expanded into before querying
        the dataset. Defaults to the empty dictionary (no aliases defined)
        e.g. opLabelAliases['Gx^3'] = ('Gx','Gx','Gx')

    svd_truncate_to : int, optional
        The Hilbert space dimension to truncate the operation matrices to using
        a SVD to keep only the largest svdToTruncateTo singular values of
        the I_tildle LGST matrix. Zero means no truncation.
        Defaults to dimension of `target_model`.

    min_prob_clip_for_weighting : float, optional
        defines the clipping interval for the statistical weight used
        within the chi^2 function (see chi2fn).

    prob_clip_interval : 2-tuple, optional
        (min,max) to clip probabilities to within Model probability
        computation routines (see Model.bulk_fill_probs)

    verbosity : int, optional
        Verbosity value to send to run_lgst(...) and do_mc2gst(...) calls.

    Returns
    -------
    dict
        A dictionary that relates each circuit to a Model containing the LGST
        estimate of that circuit's action (as a SPAM-less operation sequence)
        stored under the operation label "GsigmaLbl", along with LSGST estimates
        of the gates in `target_model`.
    """
    printer = _objs.VerbosityPrinter.create_printer(verbosity)
    directLSGSTmodels = {}
    printer.log("--- Direct LSGST precomputation ---")
    with printer.progress_logging(1):
        for i, sigma in enumerate(circuits):
            printer.show_progress(i, len(circuits), prefix="--- Computing model for string-", suffix='---')
            directLSGSTmodels[sigma] = direct_mc2gst_model(
                sigma, "GsigmaLbl", dataset, prep_fiducials, meas_fiducials, target_model,
                op_label_aliases, svd_truncate_to, min_prob_clip_for_weighting,
                prob_clip_interval, verbosity)

    return directLSGSTmodels


def direct_mlgst_model(circuit_to_estimate, circuit_label, dataset,
                       prep_fiducials, meas_fiducials, target_model,
                       op_label_aliases=None, svd_truncate_to=None, min_prob_clip=1e-6,
                       prob_clip_interval=(-1e6, 1e6), verbosity=0):
    """
    Constructs a model of MLEGST estimates for target gates and circuit_to_estimate.

    Starting with a Direct-LGST estimate for circuit_to_estimate, runs MLEGST
    using the same strings that LGST would have used to estimate circuit_to_estimate
    and each of the target gates.  That is, MLEGST is run with strings of the form:

    1. prep_fiducial
    2. meas_fiducial
    3. prep_fiducial + meas_fiducial
    4. prep_fiducial + singleGate + meas_fiducial
    5. prep_fiducial + circuit_to_estimate + meas_fiducial

    and the resulting Model estimate is returned.

    Parameters
    ----------
    circuit_to_estimate : Circuit or tuple
        The single circuit to estimate using LSGST

    circuit_label : string
        The label for the estimate of `circuit_to_estimate`.
        i.e. `op_matrix = returned_model[op_label]`

    dataset : DataSet
        The data to use for LGST

    prep_fiducials : list of Circuits
        Fiducial circuits used to construct an informationally complete
        effective preparation.

    meas_fiducials : list of Circuits
        Fiducial circuits used to construct an informationally complete
        effective measurement.

    target_model : Model
        The target model used by LGST to extract operation labels and an initial gauge

    op_label_aliases : dictionary, optional
        Dictionary whose keys are operation label "aliases" and whose values are tuples
        corresponding to what that operation label should be expanded into before querying
        the dataset. Defaults to the empty dictionary (no aliases defined)
        e.g. opLabelAliases['Gx^3'] = ('Gx','Gx','Gx')

    svd_truncate_to : int, optional
        The Hilbert space dimension to truncate the operation matrices to using
        a SVD to keep only the largest svdToTruncateTo singular values of
        the I_tildle LGST matrix. Zero means no truncation.
        Defaults to dimension of `target_model`.

    min_prob_clip : float, optional
        defines the minimum probability "patch point" used
        within the logl function.

    prob_clip_interval : 2-tuple, optional
        (min,max) to clip probabilities to within Model probability
        computation routines (see Model.bulk_fill_probs)

    verbosity : int, optional
        Verbosity value to send to run_lgst(...) and do_mlgst(...) calls.

    Returns
    -------
    Model
        A model containing MLEGST estimates of `circuit_to_estimate`
        and the gates of `target_model`.
    """
    direct_lgst = model_with_lgst_circuit_estimates(
        [circuit_to_estimate], dataset, prep_fiducials, meas_fiducials, target_model,
        True, op_label_aliases, None, [circuit_label], svd_truncate_to, verbosity)

    # LEXICOGRAPHICAL VS MATRIX ORDER
    circuits = prep_fiducials + meas_fiducials + [prepC + measC for prepC in prep_fiducials
                                                  for measC in meas_fiducials]
    for opLabel in direct_lgst.operations:
        circuits.extend([prepC + _objs.Circuit((opLabel,)) + measC
                         for prepC in prep_fiducials for measC in meas_fiducials])

    aliases = {} if (op_label_aliases is None) else op_label_aliases.copy()
    aliases[circuit_label] = circuit_to_estimate.replace_layers_with_aliases(op_label_aliases)

    obuilder = _objs.PoissonPicDeltaLogLFunction.builder(regularization={'min_prob_clip': min_prob_clip},
                                                         penalties={'prob_clip_interval': prob_clip_interval})
    bulk_circuits = _objs.CircuitList(circuits, aliases)
    _, direct_mlegst = _core.run_gst_fit_simple(dataset, direct_lgst, bulk_circuits, optimizer=None,
                                                objective_function_builder=obuilder, resource_alloc=None,
                                                verbosity=verbosity)

    return direct_mlegst


def direct_mlgst_models(circuits, dataset, prep_fiducials, meas_fiducials, target_model,
                        op_label_aliases=None, svd_truncate_to=None, min_prob_clip=1e-6,
                        prob_clip_interval=(-1e6, 1e6), verbosity=0):
    """
    Constructs a dictionary with keys == circuits and values == Direct-MLEGST Models.

    Parameters
    ----------
    circuits : list of Circuit or tuple objects
        The circuits to estimate using MLEGST.  The elements of this list
        are the keys of the returned dictionary.

    dataset : DataSet
        The data to use for all LGST and LSGST estimates.

    prep_fiducials : list of Circuits
        Fiducial circuits used to construct an informationally complete
        effective preparation.

    meas_fiducials : list of Circuits
        Fiducial circuits used to construct an informationally complete
        effective measurement.

    target_model : Model
        The target model used by LGST to extract operation labels and an initial gauge

    op_label_aliases : dictionary, optional
        Dictionary whose keys are operation label "aliases" and whose values are tuples
        corresponding to what that operation label should be expanded into before querying
        the dataset. Defaults to the empty dictionary (no aliases defined)
        e.g. opLabelAliases['Gx^3'] = ('Gx','Gx','Gx')

    svd_truncate_to : int, optional
        The Hilbert space dimension to truncate the operation matrices to using
        a SVD to keep only the largest svdToTruncateTo singular values of
        the I_tildle LGST matrix. Zero means no truncation.
        Defaults to dimension of `target_model`.

    min_prob_clip : float, optional
        defines the minimum probability "patch point" used
        within the logl function.

    prob_clip_interval : 2-tuple, optional
        (min,max) to clip probabilities to within Model probability
        computation routines (see Model.bulk_fill_probs)

    verbosity : int, optional
        Verbosity value to send to run_lgst(...) and do_mlgst(...) calls.

    Returns
    -------
    dict
        A dictionary that relates each circuit to a Model containing the LGST
        estimate of that circuit's action (as a SPAM-less operation sequence)
        stored under the operation label "GsigmaLbl", along with MLEGST estimates
        of the gates in `target_model`.
    """
    printer = _objs.VerbosityPrinter.create_printer(verbosity)
    directMLEGSTmodels = {}
    printer.log("--- Direct MLEGST precomputation ---")
    with printer.progress_logging(1):
        for i, sigma in enumerate(circuits):
            printer.show_progress(i, len(circuits), prefix="--- Computing model for string ", suffix="---")
            directMLEGSTmodels[sigma] = direct_mlgst_model(
                sigma, "GsigmaLbl", dataset, prep_fiducials, meas_fiducials, target_model,
                op_label_aliases, svd_truncate_to, min_prob_clip,
                prob_clip_interval, verbosity)

    return directMLEGSTmodels


def focused_mc2gst_model(circuit_to_estimate, circuit_label, dataset,
                         prep_fiducials, meas_fiducials, start_model,
                         op_label_aliases=None, min_prob_clip_for_weighting=1e-4,
                         prob_clip_interval=(-1e6, 1e6), verbosity=0):
    """
    Constructs a model containing a single LSGST estimate of `circuit_to_estimate`.

    Starting with `start_model`, run LSGST with the same circuits that LGST
    would use to estimate `circuit_to_estimate`.  That is, LSGST is run with
    strings of the form:  prep_fiducial + circuit_to_estimate + meas_fiducial
    and return the resulting Model.

    Parameters
    ----------
    circuit_to_estimate : Circuit or tuple
        The single circuit to estimate using LSGST

    circuit_label : string
        The label for the estimate of `circuit_to_estimate`.
        i.e. `op_matrix = returned_model[op_label]`

    dataset : DataSet
        The data to use for LGST

    prep_fiducials : list of Circuits
        Fiducial circuits used to construct an informationally complete
        effective preparation.

    meas_fiducials : list of Circuits
        Fiducial circuits used to construct an informationally complete
        effective measurement.

    start_model : Model
        The model to seed LSGST with. Often times obtained via LGST.

    op_label_aliases : dictionary, optional
        Dictionary whose keys are operation label "aliases" and whose values are tuples
        corresponding to what that operation label should be expanded into before querying
        the dataset. Defaults to the empty dictionary (no aliases defined)
        e.g. opLabelAliases['Gx^3'] = ('Gx','Gx','Gx')

    min_prob_clip_for_weighting : float, optional
        defines the clipping interval for the statistical weight used
        within the chi^2 function (see chi2fn).

    prob_clip_interval : 2-tuple, optional
        (min,max) to clip probabilities to within Model probability
        computation routines (see Model.bulk_fill_probs)

    verbosity : int, optional
        Verbosity value to send do_mc2gst(...) call.

    Returns
    -------
    Model
        A model containing LSGST estimate of `circuit_to_estimate`.
    """
    circuits = [prepC + circuit_to_estimate + measC for prepC in prep_fiducials for measC in meas_fiducials]

    obuilder = _objs.Chi2Function.builder(regularization={'min_prob_clip_for_weighting': min_prob_clip_for_weighting},
                                          penalties={'prob_clip_interval': prob_clip_interval})
    bulk_circuits = _objs.CircuitList(circuits, op_label_aliases)
    _, focused_lsgst = _core.run_gst_fit_simple(dataset, start_model, bulk_circuits, optimizer=None,
                                                objective_function_builder=obuilder, resource_alloc=None,
                                                verbosity=verbosity)

    focused_lsgst.operations[circuit_label] = _FullArbitraryOp(
        focused_lsgst.sim.product(circuit_to_estimate))  # add desired string as a separate labeled gate
    return focused_lsgst


def focused_mc2gst_models(circuits, dataset, prep_fiducials, meas_fiducials,
                          start_model, op_label_aliases=None,
                          min_prob_clip_for_weighting=1e-4,
                          prob_clip_interval=(-1e6, 1e6), verbosity=0):
    """
    Constructs a dictionary with keys == circuits and values == Focused-LSGST Models.

    Parameters
    ----------
    circuits : list of Circuit or tuple objects
        The circuits to estimate using LSGST.  The elements of this list
        are the keys of the returned dictionary.

    dataset : DataSet
        The data to use for all LGST and LSGST estimates.

    prep_fiducials : list of Circuits
        Fiducial circuits used to construct an informationally complete
        effective preparation.

    meas_fiducials : list of Circuits
        Fiducial circuits used to construct an informationally complete
        effective measurement.

    start_model : Model
        The model to seed LSGST with. Often times obtained via LGST.

    op_label_aliases : dictionary, optional
        Dictionary whose keys are operation label "aliases" and whose values are tuples
        corresponding to what that operation label should be expanded into before querying
        the dataset. Defaults to the empty dictionary (no aliases defined)
        e.g. opLabelAliases['Gx^3'] = ('Gx','Gx','Gx')

    min_prob_clip_for_weighting : float, optional
        defines the clipping interval for the statistical weight used
        within the chi^2 function (see chi2fn).

    prob_clip_interval : 2-tuple, optional
        (min,max) to clip probabilities to within Model probability
        computation routines (see Model.bulk_fill_probs)

    verbosity : int, optional
        Verbosity value to send to do_mc2gst(...) call.

    Returns
    -------
    dict
        A dictionary that relates each circuit to a Model containing the
        LSGST estimate of that circuit's action, stored under the
        operation label "GsigmaLbl".
    """

    printer = _objs.VerbosityPrinter.create_printer(verbosity)
    focusedLSGSTmodels = {}
    printer.log("--- Focused LSGST precomputation ---")
    with printer.progress_logging(1):
        for i, sigma in enumerate(circuits):
            printer.show_progress(i, len(circuits), prefix="--- Computing model for string", suffix='---')
            focusedLSGSTmodels[sigma] = focused_mc2gst_model(
                sigma, "GsigmaLbl", dataset, prep_fiducials, meas_fiducials, start_model,
                op_label_aliases, min_prob_clip_for_weighting, prob_clip_interval, verbosity)
    return focusedLSGSTmodels

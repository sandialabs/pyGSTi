""" Functions for generating Direct-(LGST, MC2GST, MLGST) models """
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
from . import core as _core


def model_with_lgst_circuit_estimates(
        circuitsToEstimate, dataset, prepStrs, effectStrs,
        targetModel, includeTargetOps=True, opLabelAliases=None,
        guessModelForGauge=None, circuitLabels=None, svdTruncateTo=None,
        verbosity=0):
    """
    Constructs a model that contains LGST estimates for circuitsToEstimate.

    For each operation sequence s in circuitsToEstimate, the constructed model
    contains the LGST estimate for s as separate gate, labeled either by
    the corresponding element of circuitLabels or by the tuple of s itself.

    Parameters
    ----------
    circuitsToEstimate : list of Circuits or tuples
        The operation sequences to estimate using LGST

    dataset : DataSet
        The data to use for LGST

    prepStrs,effectStrs : list of Circuits
        Fiducial Circuit lists used to construct a informationally complete
        preparation and measurement.

    targetModel : Model
        A model used by LGST to specify which operation labels should be estimated,
        a guess for which gauge these estimates should be returned in, and
        used to simplify operation sequences.

    includeTargetOps : bool, optional
        If True, the operation labels in targetModel will be included in the
        returned model.

    opLabelAliases : dictionary, optional
        Dictionary whose keys are operation label "aliases" and whose values are tuples
        corresponding to what that operation label should be expanded into before querying
        the dataset. Defaults to the empty dictionary (no aliases defined)
        e.g. opLabelAliases['Gx^3'] = ('Gx','Gx','Gx')

    guessModelForGauge : Model, optional
        A model used to compute a gauge transformation that is applied to
        the LGST estimates.  This gauge transformation is computed such that
        if the estimated gates matched the model given, then the gate
        matrices would match, i.e. the gauge would be the same as
        the model supplied. Defaults to the targetModel.

    circuitLabels : list of strings, optional
        A list of labels in one-to-one correspondence with the
        operation sequence in circuitsToEstimate.  These labels are
        the keys to access the operation matrices in the returned
        Model, i.e. op_matrix = returned_model[op_label]

    svdTruncateTo : int, optional
        The Hilbert space dimension to truncate the operation matrices to using
        a SVD to keep only the largest svdToTruncateTo singular values of
        the I_tildle LGST matrix. Zero means no truncation.
        Defaults to dimension of `targetModel`.

    verbosity : int, optional
        Verbosity value to send to do_lgst(...) call.

    Returns
    -------
    Model
        A model containing LGST estimates for all the requested
        operation sequences and possibly the gates in targetModel.
    """
    opLabels = []  # list of operation labels for LGST to estimate
    if opLabelAliases is None: aliases = {}
    else: aliases = opLabelAliases.copy()

    #Add operation sequences to estimate as aliases
    if circuitLabels is not None:
        assert(len(circuitLabels) == len(circuitsToEstimate))
        for opLabel, opStr in zip(circuitLabels, circuitsToEstimate):
            aliases[opLabel] = opStr.replace_layers_with_aliases(opLabelAliases)
            opLabels.append(opLabel)
    else:
        for opStr in circuitsToEstimate:
            newLabel = 'G' + '.'.join(map(str, tuple(opStr)))
            aliases[newLabel] = opStr.replace_layers_with_aliases(opLabelAliases)  # use circuit tuple as label
            opLabels.append(newLabel)

    #Add target model labels (not aliased) if requested
    if includeTargetOps and targetModel is not None:
        for targetOpLabel in targetModel.operations:
            if targetOpLabel not in opLabels:  # very unlikely that this is false
                opLabels.append(targetOpLabel)

    return _core.do_lgst(dataset, prepStrs, effectStrs, targetModel,
                         opLabels, aliases, guessModelForGauge,
                         svdTruncateTo, verbosity)


def direct_lgst_model(circuitToEstimate, circuitLabel, dataset,
                      prepStrs, effectStrs, targetModel,
                      opLabelAliases=None, svdTruncateTo=None, verbosity=0):
    """
    Constructs a model of LGST estimates for target gates and circuitToEstimate.

    Parameters
    ----------
    circuitToEstimate : Circuit or tuple
        The single operation sequence to estimate using LGST

    circuitLabel : string
        The label for the estimate of circuitToEstimate.
        i.e. op_matrix = returned_model[op_label]

    dataset : DataSet
        The data to use for LGST

    prepStrs,effectStrs : list of Circuits
        Fiducial Circuit lists used to construct a informationally complete
        preparation and measurement.

    targetModel : Model
        The target model used by LGST to extract operation labels and an initial gauge

    opLabelAliases : dictionary, optional
        Dictionary whose keys are operation label "aliases" and whose values are tuples
        corresponding to what that operation label should be expanded into before querying
        the dataset. Defaults to the empty dictionary (no aliases defined)
        e.g. opLabelAliases['Gx^3'] = ('Gx','Gx','Gx')

    svdTruncateTo : int, optional
        The Hilbert space dimension to truncate the operation matrices to using
        a SVD to keep only the largest svdToTruncateTo singular values of
        the I_tildle LGST matrix.  Zero means no truncation.
        Defaults to dimension of `targetModel`.

    verbosity : int, optional
        Verbosity value to send to do_lgst(...) call.

    Returns
    -------
    Model
        A model containing LGST estimates of circuitToEstimate
        and the gates of targetModel.
    """
    return model_with_lgst_circuit_estimates(
        [circuitToEstimate], dataset, prepStrs, effectStrs, targetModel,
        True, opLabelAliases, None, [circuitLabel], svdTruncateTo,
        verbosity)


def direct_lgst_models(circuits, dataset, prepStrs, effectStrs, targetModel,
                       opLabelAliases=None, svdTruncateTo=None, verbosity=0):
    """
    Constructs a dictionary with keys == operation sequences and values == Direct-LGST Models.

    Parameters
    ----------
    circuits : list of Circuit or tuple objects
        The operation sequences to estimate using LGST.  The elements of this list
        are the keys of the returned dictionary.

    dataset : DataSet
        The data to use for all LGST estimates.

    prepStrs,effectStrs : list of Circuits
        Fiducial Circuit lists used to construct a informationally complete
        preparation and measurement.

    targetModel : Model
        The target model used by LGST to extract operation labels and an initial gauge

    opLabelAliases : dictionary, optional
        Dictionary whose keys are operation label "aliases" and whose values are tuples
        corresponding to what that operation label should be expanded into before querying
        the dataset. Defaults to the empty dictionary (no aliases defined)
        e.g. opLabelAliases['Gx^3'] = ('Gx','Gx','Gx')

    svdTruncateTo : int, optional
        The Hilbert space dimension to truncate the operation matrices to using
        a SVD to keep only the largest svdToTruncateTo singular values of
        the I_tildle LGST matrix. Zero means no truncation.
        Defaults to dimension of `targetModel`.

    verbosity : int, optional
        Verbosity value to send to do_lgst(...) call.

    Returns
    -------
    dict
        A dictionary that relates each operation sequence of circuits to a
        Model containing the LGST estimate of that operation sequence stored under
        the operation label "GsigmaLbl", along with LGST estimates of the gates in
        targetModel.
    """
    printer = _objs.VerbosityPrinter.build_printer(verbosity)

    directLGSTmodels = {}
    printer.log("--- Direct LGST precomputation ---")
    with printer.progress_logging(1):
        for i, sigma in enumerate(circuits):
            printer.show_progress(i, len(circuits), prefix="--- Computing model for string -", suffix='---')
            directLGSTmodels[sigma] = direct_lgst_model(
                sigma, "GsigmaLbl", dataset, prepStrs, effectStrs, targetModel,
                opLabelAliases, svdTruncateTo, verbosity)
    return directLGSTmodels


def direct_mc2gst_model(circuitToEstimate, circuitLabel, dataset,
                        prepStrs, effectStrs, targetModel,
                        opLabelAliases=None, svdTruncateTo=None,
                        minProbClipForWeighting=1e-4,
                        probClipInterval=(-1e6, 1e6), verbosity=0):
    """
    Constructs a model of LSGST estimates for target gates and circuitToEstimate.

    Starting with a Direct-LGST estimate for circuitToEstimate, runs LSGST
    using the same strings that LGST would have used to estimate circuitToEstimate
    and each of the target gates.  That is, LSGST is run with strings of the form:

    1. prepStr
    2. effectStr
    3. prepStr + effectStr
    4. prepStr + singleGate + effectStr
    5. prepStr + circuitToEstimate + effectStr

    and the resulting Model estimate is returned.

    Parameters
    ----------
    circuitToEstimate : Circuit
        The single operation sequence to estimate using LSGST

    circuitLabel : string
        The label for the estimate of circuitToEstimate.
        i.e. op_matrix = returned_mode[op_label]

    dataset : DataSet
        The data to use for LGST

    prepStrs,effectStrs : list of Circuits
        Fiducial Circuit lists used to construct a informationally complete
        preparation and measurement.

    targetModel : Model
        The target model used by LGST to extract operation labels and an initial gauge

    opLabelAliases : dictionary, optional
        Dictionary whose keys are operation label "aliases" and whose values are tuples
        corresponding to what that operation label should be expanded into before querying
        the dataset. Defaults to the empty dictionary (no aliases defined)
        e.g. opLabelAliases['Gx^3'] = ('Gx','Gx','Gx')

    svdTruncateTo : int, optional
        The Hilbert space dimension to truncate the operation matrices to using
        a SVD to keep only the largest svdToTruncateTo singular values of
        the I_tildle LGST matrix. Zero means no truncation.
        Defaults to dimension of `targetModel`.

    minProbClipForWeighting : float, optional
        defines the clipping interval for the statistical weight used
        within the chi^2 function (see chi2fn).

    probClipInterval : 2-tuple, optional
        (min,max) to clip probabilities to within Model probability
        computation routines (see Model.bulk_fill_probs)

    verbosity : int, optional
        Verbosity value to send to do_lgst(...) and do_mc2gst(...) calls.

    Returns
    -------
    Model
        A model containing LSGST estimates of circuitToEstimate
        and the gates of targetModel.
    """
    direct_lgst = model_with_lgst_circuit_estimates(
        [circuitToEstimate], dataset, prepStrs, effectStrs, targetModel,
        True, opLabelAliases, None, [circuitLabel], svdTruncateTo, verbosity)

    # LEXICOGRAPHICAL VS MATRIX ORDER
    circuits = prepStrs + effectStrs + [prepStr + effectStr for prepStr in prepStrs for effectStr in effectStrs]
    for opLabel in direct_lgst.operations:
        circuits.extend([prepStr + _objs.Circuit((opLabel,)) + effectStr
                         for prepStr in prepStrs for effectStr in effectStrs])

    aliases = {} if (opLabelAliases is None) else opLabelAliases.copy()
    aliases[circuitLabel] = circuitToEstimate.replace_layers_with_aliases(opLabelAliases)

    _, direct_lsgst = _core.do_mc2gst(
        dataset, direct_lgst, circuits,
        minProbClipForWeighting=minProbClipForWeighting,
        probClipInterval=probClipInterval, verbosity=verbosity,
        opLabelAliases=aliases)

    return direct_lsgst


def direct_mc2gst_models(circuits, dataset, prepStrs, effectStrs,
                         targetModel, opLabelAliases=None,
                         svdTruncateTo=None, minProbClipForWeighting=1e-4,
                         probClipInterval=(-1e6, 1e6), verbosity=0):
    """
    Constructs a dictionary with keys == operation sequences and values == Direct-LSGST Models.

    Parameters
    ----------
    circuits : list of Circuit or tuple objects
        The operation sequences to estimate using LSGST.  The elements of this list
        are the keys of the returned dictionary.

    dataset : DataSet
        The data to use for all LGST and LSGST estimates.

    prepStrs,effectStrs : list of Circuits
        Fiducial Circuit lists used to construct a informationally complete
        preparation and measurement.

    targetModel : Model
        The target model used by LGST to extract operation labels and an initial gauge

    opLabelAliases : dictionary, optional
        Dictionary whose keys are operation label "aliases" and whose values are tuples
        corresponding to what that operation label should be expanded into before querying
        the dataset. Defaults to the empty dictionary (no aliases defined)
        e.g. opLabelAliases['Gx^3'] = ('Gx','Gx','Gx')

    svdTruncateTo : int, optional
        The Hilbert space dimension to truncate the operation matrices to using
        a SVD to keep only the largest svdToTruncateTo singular values of
        the I_tildle LGST matrix. Zero means no truncation.
        Defaults to dimension of `targetModel`.

    minProbClipForWeighting : float, optional
        defines the clipping interval for the statistical weight used
        within the chi^2 function (see chi2fn).

    probClipInterval : 2-tuple, optional
        (min,max) to clip probabilities to within Model probability
        computation routines (see Model.bulk_fill_probs)

    verbosity : int, optional
        Verbosity value to send to do_lgst(...) and do_mc2gst(...) calls.

    Returns
    -------
    dict
        A dictionary that relates each operation sequence of circuits to a
        Model containing the LSGST estimate of that operation sequence stored under
        the operation label "GsigmaLbl", along with LSGST estimates of the gates in
        targetModel.
    """
    printer = _objs.VerbosityPrinter.build_printer(verbosity)
    directLSGSTmodels = {}
    printer.log("--- Direct LSGST precomputation ---")
    with printer.progress_logging(1):
        for i, sigma in enumerate(circuits):
            printer.show_progress(i, len(circuits), prefix="--- Computing model for string-", suffix='---')
            directLSGSTmodels[sigma] = direct_mc2gst_model(
                sigma, "GsigmaLbl", dataset, prepStrs, effectStrs, targetModel,
                opLabelAliases, svdTruncateTo, minProbClipForWeighting,
                probClipInterval, verbosity)

    return directLSGSTmodels


def direct_mlgst_model(circuitToEstimate, circuitLabel, dataset,
                       prepStrs, effectStrs, targetModel,
                       opLabelAliases=None, svdTruncateTo=None, minProbClip=1e-6,
                       probClipInterval=(-1e6, 1e6), verbosity=0):
    """
    Constructs a model of MLEGST estimates for target gates and circuitToEstimate.

    Starting with a Direct-LGST estimate for circuitToEstimate, runs MLEGST
    using the same strings that LGST would have used to estimate circuitToEstimate
    and each of the target gates.  That is, MLEGST is run with strings of the form:

    1. prepStr
    2. effectStr
    3. prepStr + effectStr
    4. prepStr + singleGate + effectStr
    5. prepStr + circuitToEstimate + effectStr

    and the resulting Model estimate is returned.

    Parameters
    ----------
    circuitToEstimate : Circuit or tuple
        The single operation sequence to estimate using LSGST

    circuitLabel : string
        The label for the estimate of circuitToEstimate.
        i.e. op_matrix = returned_model[op_label]

    dataset : DataSet
        The data to use for LGST

    prepStrs,effectStrs : list of Circuits
        Fiducial Circuit lists used to construct a informationally complete
        preparation and measurement.

    targetModel : Model
        The target model used by LGST to extract operation labels and an initial gauge

    opLabelAliases : dictionary, optional
        Dictionary whose keys are operation label "aliases" and whose values are tuples
        corresponding to what that operation label should be expanded into before querying
        the dataset. Defaults to the empty dictionary (no aliases defined)
        e.g. opLabelAliases['Gx^3'] = ('Gx','Gx','Gx')

    svdTruncateTo : int, optional
        The Hilbert space dimension to truncate the operation matrices to using
        a SVD to keep only the largest svdToTruncateTo singular values of
        the I_tildle LGST matrix. Zero means no truncation.
        Defaults to dimension of `targetModel`.

    minProbClip : float, optional
        defines the minimum probability "patch point" used
        within the logl function.

    probClipInterval : 2-tuple, optional
        (min,max) to clip probabilities to within Model probability
        computation routines (see Model.bulk_fill_probs)

    verbosity : int, optional
        Verbosity value to send to do_lgst(...) and do_mlgst(...) calls.

    Returns
    -------
    Model
        A model containing MLEGST estimates of circuitToEstimate
        and the gates of targetModel.
    """
    direct_lgst = model_with_lgst_circuit_estimates(
        [circuitToEstimate], dataset, prepStrs, effectStrs, targetModel,
        True, opLabelAliases, None, [circuitLabel], svdTruncateTo, verbosity)

    # LEXICOGRAPHICAL VS MATRIX ORDER
    circuits = prepStrs + effectStrs + [prepStr + effectStr for prepStr in prepStrs for effectStr in effectStrs]
    for opLabel in direct_lgst.operations:
        circuits.extend([prepStr + _objs.Circuit((opLabel,)) + effectStr
                         for prepStr in prepStrs for effectStr in effectStrs])

    aliases = {} if (opLabelAliases is None) else opLabelAliases.copy()
    aliases[circuitLabel] = circuitToEstimate.replace_layers_with_aliases(opLabelAliases)

    _, direct_mlegst = _core.do_mlgst(
        dataset, direct_lgst, circuits, minProbClip=minProbClip,
        probClipInterval=probClipInterval, verbosity=verbosity,
        opLabelAliases=aliases)
    return direct_mlegst


def direct_mlgst_models(circuits, dataset, prepStrs, effectStrs, targetModel,
                        opLabelAliases=None, svdTruncateTo=None, minProbClip=1e-6,
                        probClipInterval=(-1e6, 1e6), verbosity=0):
    """
    Constructs a dictionary with keys == operation sequences and values == Direct-MLEGST Models.

    Parameters
    ----------
    circuits : list of Circuit or tuple objects
        The operation sequences to estimate using MLEGST.  The elements of this list
        are the keys of the returned dictionary.

    dataset : DataSet
        The data to use for all LGST and LSGST estimates.

    prepStrs,effectStrs : list of Circuits
        Fiducial Circuit lists used to construct a informationally complete
        preparation and measurement.

    targetModel : Model
        The target model used by LGST to extract operation labels and an initial gauge

    opLabelAliases : dictionary, optional
        Dictionary whose keys are operation label "aliases" and whose values are tuples
        corresponding to what that operation label should be expanded into before querying
        the dataset. Defaults to the empty dictionary (no aliases defined)
        e.g. opLabelAliases['Gx^3'] = ('Gx','Gx','Gx')

    svdTruncateTo : int, optional
        The Hilbert space dimension to truncate the operation matrices to using
        a SVD to keep only the largest svdToTruncateTo singular values of
        the I_tildle LGST matrix. Zero means no truncation.
        Defaults to dimension of `targetModel`.

    minProbClip : float, optional
        defines the minimum probability "patch point" used
        within the logl function.

    probClipInterval : 2-tuple, optional
        (min,max) to clip probabilities to within Model probability
        computation routines (see Model.bulk_fill_probs)

    verbosity : int, optional
        Verbosity value to send to do_lgst(...) and do_mlgst(...) calls.

    Returns
    -------
    dict
        A dictionary that relates each operation sequence of circuits to a
        Model containing the MLEGST estimate of that operation sequence stored under
        the operation label "GsigmaLbl", along with MLEGST estimates of the gates in
        targetModel.
    """
    printer = _objs.VerbosityPrinter.build_printer(verbosity)
    directMLEGSTmodels = {}
    printer.log("--- Direct MLEGST precomputation ---")
    with printer.progress_logging(1):
        for i, sigma in enumerate(circuits):
            printer.show_progress(i, len(circuits), prefix="--- Computing model for string ", suffix="---")
            directMLEGSTmodels[sigma] = direct_mlgst_model(
                sigma, "GsigmaLbl", dataset, prepStrs, effectStrs, targetModel,
                opLabelAliases, svdTruncateTo, minProbClip,
                probClipInterval, verbosity)

    return directMLEGSTmodels


def focused_mc2gst_model(circuitToEstimate, circuitLabel, dataset,
                         prepStrs, effectStrs, startModel,
                         opLabelAliases=None, minProbClipForWeighting=1e-4,
                         probClipInterval=(-1e6, 1e6), verbosity=0):
    """
    Constructs a model containing a single LSGST estimate of circuitToEstimate.

    Starting with startModel, run LSGST with the same operation sequences that LGST
    would use to estimate circuitToEstimate.  That is, LSGST is run with
    strings of the form:  prepStr + circuitToEstimate + effectStr
    and return the resulting Model.

    Parameters
    ----------
    circuitToEstimate : Circuit or tuple
        The single operation sequence to estimate using LSGST

    circuitLabel : string
        The label for the estimate of circuitToEstimate.
        i.e. op_matrix = returned_model[op_label]

    dataset : DataSet
        The data to use for LGST

    prepStrs,effectStrs : list of Circuits
        Fiducial Circuit lists used to construct a informationally complete
        preparation and measurement.

    startModel : Model
        The model to seed LSGST with. Often times obtained via LGST.

    opLabelAliases : dictionary, optional
        Dictionary whose keys are operation label "aliases" and whose values are tuples
        corresponding to what that operation label should be expanded into before querying
        the dataset. Defaults to the empty dictionary (no aliases defined)
        e.g. opLabelAliases['Gx^3'] = ('Gx','Gx','Gx')

    minProbClipForWeighting : float, optional
        defines the clipping interval for the statistical weight used
        within the chi^2 function (see chi2fn).

    probClipInterval : 2-tuple, optional
        (min,max) to clip probabilities to within Model probability
        computation routines (see Model.bulk_fill_probs)

    verbosity : int, optional
        Verbosity value to send do_mc2gst(...) call.

    Returns
    -------
    Model
        A model containing LSGST estimate of circuitToEstimate.
    """
    circuits = [prepStr + circuitToEstimate + effectStr for prepStr in prepStrs for effectStr in effectStrs]

    _, focused_lsgst = _core.do_mc2gst(
        dataset, startModel, circuits,
        minProbClipForWeighting=minProbClipForWeighting,
        probClipInterval=probClipInterval,
        opLabelAliases=opLabelAliases,
        verbosity=verbosity)

    focused_lsgst.operations[circuitLabel] = _objs.FullDenseOp(
        focused_lsgst.product(circuitToEstimate))  # add desired string as a separate labeled gate
    return focused_lsgst


def focused_mc2gst_models(circuits, dataset, prepStrs, effectStrs,
                          startModel, opLabelAliases=None,
                          minProbClipForWeighting=1e-4,
                          probClipInterval=(-1e6, 1e6), verbosity=0):
    """
    Constructs a dictionary with keys == operation sequences and values == Focused-LSGST Models.

    Parameters
    ----------
    circuits : list of Circuit or tuple objects
        The operation sequences to estimate using LSGST.  The elements of this list
        are the keys of the returned dictionary.

    dataset : DataSet
        The data to use for all LGST and LSGST estimates.

    prepStrs,effectStrs : list of Circuits
        Fiducial Circuit lists used to construct a informationally complete
        preparation and measurement.

    startModel : Model
        The model to seed LSGST with. Often times obtained via LGST.

    opLabelAliases : dictionary, optional
        Dictionary whose keys are operation label "aliases" and whose values are tuples
        corresponding to what that operation label should be expanded into before querying
        the dataset. Defaults to the empty dictionary (no aliases defined)
        e.g. opLabelAliases['Gx^3'] = ('Gx','Gx','Gx')

    minProbClipForWeighting : float, optional
        defines the clipping interval for the statistical weight used
        within the chi^2 function (see chi2fn).

    probClipInterval : 2-tuple, optional
        (min,max) to clip probabilities to within Model probability
        computation routines (see Model.bulk_fill_probs)

    verbosity : int, optional
        Verbosity value to send to do_mc2gst(...) call.

    Returns
    -------
    dict
        A dictionary that relates each operation sequence of circuits to a
        Model containing the LSGST estimate of that operation sequence stored under
        the operation label "GsigmaLbl".
    """

    printer = _objs.VerbosityPrinter.build_printer(verbosity)
    focusedLSGSTmodels = {}
    printer.log("--- Focused LSGST precomputation ---")
    with printer.progress_logging(1):
        for i, sigma in enumerate(circuits):
            printer.show_progress(i, len(circuits), prefix="--- Computing model for string", suffix='---')
            focusedLSGSTmodels[sigma] = focused_mc2gst_model(
                sigma, "GsigmaLbl", dataset, prepStrs, effectStrs, startModel,
                opLabelAliases, minProbClipForWeighting, probClipInterval, verbosity)
    return focusedLSGSTmodels

""" Functions for generating Direct-(LGST, MC2GST, MLGST) gatesets """
from __future__ import division, print_function, absolute_import, unicode_literals
#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************


from .. import tools        as _tools
from .. import construction as _construction
from .. import objects      as _objs
from .  import core         as _core


def gateset_with_lgst_gatestring_estimates(
        gateStringsToEstimate, dataset, prepStrs, effectStrs,
        targetGateset, includeTargetGates=True, gateLabelAliases=None, 
        guessGatesetForGauge=None, gateStringLabels=None, svdTruncateTo=None,
        verbosity=0 ):
    """
    Constructs a gateset that contains LGST estimates for gateStringsToEstimate.

    For each gate string s in gateStringsToEstimate, the constructed gateset
    contains the LGST estimate for s as separate gate, labeled either by
    the corresponding element of gateStringLabels or by the tuple of s itself.

    Parameters
    ----------
    gateStringsToEstimate : list of GateStrings or tuples
        The gate strings to estimate using LGST

    dataset : DataSet
        The data to use for LGST

    prepStrs,effectStrs : list of GateStrings
        Fiducial GateString lists used to construct a informationally complete
        preparation and measurement.

    targetGateset : GateSet
        A gateset used by LGST to specify which gate labels should be estimated,
        a guess for which gauge these estimates should be returned in, and 
        used to compile gate sequences.

    includeTargetGates : bool, optional
        If True, the gate labels in targetGateset will be included in the
        returned gate set.

    gateLabelAliases : dictionary, optional
        Dictionary whose keys are gate label "aliases" and whose values are tuples
        corresponding to what that gate label should be expanded into before querying
        the dataset. Defaults to the empty dictionary (no aliases defined)
        e.g. gateLabelAliases['Gx^3'] = ('Gx','Gx','Gx')

    guessGatesetForGauge : GateSet, optional
        A gateset used to compute a gauge transformation that is applied to
        the LGST estimates.  This gauge transformation is computed such that
        if the estimated gates matched the gateset given, then the gate
        matrices would match, i.e. the gauge would be the same as
        the gateset supplied. Defaults to the targetGateset.

    gateStringLabels : list of strings, optional
        A list of labels in one-to-one correspondence with the
        gate string in gateStringsToEstimate.  These labels are
        the keys to access the gate matrices in the returned
        GateSet, i.e. gate_matrix = returned_gateset[gate_label]

    svdTruncateTo : int, optional
        The Hilbert space dimension to truncate the gate matrices to using
        a SVD to keep only the largest svdToTruncateTo singular values of
        the I_tildle LGST matrix. Zero means no truncation.
        Defaults to dimension of `targetGateset`.

    verbosity : int, optional
        Verbosity value to send to do_lgst(...) call.

    Returns
    -------
    Gateset
        A gateset containing LGST estimates for all the requested
        gate strings and possibly the gates in targetGateset.
    """
    gateLabels = [] #list of gate labels for LGST to estimate
    if gateLabelAliases is None: aliases = { }
    else: aliases = gateLabelAliases.copy()
    
    #Add gate strings to estimate as aliases
    if gateStringLabels is not None:
        assert(len(gateStringLabels) == len(gateStringsToEstimate))
        for gateLabel,gateStr in zip(gateStringLabels,gateStringsToEstimate):
            aliases[gateLabel] = _tools.find_replace_tuple(gateStr,gateLabelAliases)
            gateLabels.append(gateLabel)
    else:
        for gateStr in gateStringsToEstimate:
            newLabel = 'G'+'.'.join(map(str,tuple(gateStr)))
            aliases[newLabel] = _tools.find_replace_tuple(gateStr,gateLabelAliases) #use gatestring tuple as label
            gateLabels.append(newLabel)

    #Add target gateset labels (not aliased) if requested
    if includeTargetGates and targetGateset is not None:
        for targetGateLabel in targetGateset.gates:
            if targetGateLabel not in gateLabels: #very unlikely that this is false
                gateLabels.append(targetGateLabel)

    return _core.do_lgst( dataset, prepStrs, effectStrs, targetGateset,
                          gateLabels, aliases, guessGatesetForGauge,
                          svdTruncateTo, verbosity )

def direct_lgst_gateset(gateStringToEstimate, gateStringLabel, dataset,
                        prepStrs, effectStrs, targetGateset,
                        gateLabelAliases=None, svdTruncateTo=None, verbosity=0):
    """
    Constructs a gateset of LGST estimates for target gates and gateStringToEstimate.

    Parameters
    ----------
    gateStringToEstimate : GateString or tuple
        The single gate string to estimate using LGST

    gateStringLabel : string
        The label for the estimate of gateStringToEstimate.
        i.e. gate_matrix = returned_gateset[gate_label]

    dataset : DataSet
        The data to use for LGST

    prepStrs,effectStrs : list of GateStrings
        Fiducial GateString lists used to construct a informationally complete
        preparation and measurement.

    targetGateset : GateSet
        The target gate set used by LGST to extract gate labels and an initial gauge

    gateLabelAliases : dictionary, optional
        Dictionary whose keys are gate label "aliases" and whose values are tuples
        corresponding to what that gate label should be expanded into before querying
        the dataset. Defaults to the empty dictionary (no aliases defined)
        e.g. gateLabelAliases['Gx^3'] = ('Gx','Gx','Gx')

    svdTruncateTo : int, optional
        The Hilbert space dimension to truncate the gate matrices to using
        a SVD to keep only the largest svdToTruncateTo singular values of
        the I_tildle LGST matrix.  Zero means no truncation.
        Defaults to dimension of `targetGateset`.

    verbosity : int, optional
        Verbosity value to send to do_lgst(...) call.

    Returns
    -------
    Gateset
        A gateset containing LGST estimates of gateStringToEstimate
        and the gates of targetGateset.
    """
    return gateset_with_lgst_gatestring_estimates(
        [gateStringToEstimate], dataset, prepStrs, effectStrs, targetGateset,
        True, gateLabelAliases, None, [gateStringLabel], svdTruncateTo,
        verbosity )


def direct_lgst_gatesets(gateStrings, dataset, prepStrs, effectStrs, targetGateset,
                         gateLabelAliases=None, svdTruncateTo=None, verbosity=0):
    """
    Constructs a dictionary with keys == gate strings and values == Direct-LGST GateSets.

    Parameters
    ----------
    gateStrings : list of GateString or tuple objects
        The gate strings to estimate using LGST.  The elements of this list
        are the keys of the returned dictionary.

    dataset : DataSet
        The data to use for all LGST estimates.

    prepStrs,effectStrs : list of GateStrings
        Fiducial GateString lists used to construct a informationally complete
        preparation and measurement.

    targetGateset : GateSet
        The target gate set used by LGST to extract gate labels and an initial gauge

    gateLabelAliases : dictionary, optional
        Dictionary whose keys are gate label "aliases" and whose values are tuples
        corresponding to what that gate label should be expanded into before querying
        the dataset. Defaults to the empty dictionary (no aliases defined)
        e.g. gateLabelAliases['Gx^3'] = ('Gx','Gx','Gx')

    svdTruncateTo : int, optional
        The Hilbert space dimension to truncate the gate matrices to using
        a SVD to keep only the largest svdToTruncateTo singular values of
        the I_tildle LGST matrix. Zero means no truncation.
        Defaults to dimension of `targetGateset`.

    verbosity : int, optional
        Verbosity value to send to do_lgst(...) call.

    Returns
    -------
    dict
        A dictionary that relates each gate string of gateStrings to a
        GateSet containing the LGST estimate of that gate string stored under
        the gate label "GsigmaLbl", along with LGST estimates of the gates in
        targetGateset.
    """
    printer = _objs.VerbosityPrinter.build_printer(verbosity)

    directLGSTgatesets = {}
    printer.log("--- Direct LGST precomputation ---")
    with printer.progress_logging(1):
        for i,sigma in enumerate(gateStrings):
            printer.show_progress(i, len(gateStrings), prefix="--- Computing gateset for string -", suffix='---' )
            directLGSTgatesets[sigma] = direct_lgst_gateset(
                sigma, "GsigmaLbl", dataset, prepStrs, effectStrs, targetGateset,
                gateLabelAliases, svdTruncateTo, verbosity)
    return directLGSTgatesets



def direct_mc2gst_gateset( gateStringToEstimate, gateStringLabel, dataset,
                           prepStrs, effectStrs, targetGateset,
                           gateLabelAliases=None, svdTruncateTo=None,
                           minProbClipForWeighting=1e-4,
                           probClipInterval=(-1e6,1e6), verbosity=0 ):
    """
    Constructs a gateset of LSGST estimates for target gates and gateStringToEstimate.

    Starting with a Direct-LGST estimate for gateStringToEstimate, runs LSGST
    using the same strings that LGST would have used to estimate gateStringToEstimate
    and each of the target gates.  That is, LSGST is run with strings of the form:

    1. prepStr
    2. effectStr
    3. prepStr + effectStr
    4. prepStr + singleGate + effectStr
    5. prepStr + gateStringToEstimate + effectStr

    and the resulting Gateset estimate is returned.

    Parameters
    ----------
    gateStringToEstimate : GateString or tuple
        The single gate string to estimate using LSGST

    gateStringLabel : string
        The label for the estimate of gateStringToEstimate.
        i.e. gate_matrix = returned_gateset[gate_label]

    dataset : DataSet
        The data to use for LGST

    prepStrs,effectStrs : list of GateStrings
        Fiducial GateString lists used to construct a informationally complete
        preparation and measurement.

    targetGateset : GateSet
        The target gate set used by LGST to extract gate labels and an initial gauge

    gateLabelAliases : dictionary, optional
        Dictionary whose keys are gate label "aliases" and whose values are tuples
        corresponding to what that gate label should be expanded into before querying
        the dataset. Defaults to the empty dictionary (no aliases defined)
        e.g. gateLabelAliases['Gx^3'] = ('Gx','Gx','Gx')

    svdTruncateTo : int, optional
        The Hilbert space dimension to truncate the gate matrices to using
        a SVD to keep only the largest svdToTruncateTo singular values of
        the I_tildle LGST matrix. Zero means no truncation.
        Defaults to dimension of `targetGateset`.

    minProbClipForWeighting : float, optional
        defines the clipping interval for the statistical weight used
        within the chi^2 function (see chi2fn).

    probClipInterval : 2-tuple, optional
        (min,max) to clip probabilities to within GateSet probability
        computation routines (see GateSet.bulk_fill_probs)

    verbosity : int, optional
        Verbosity value to send to do_lgst(...) and do_mc2gst(...) calls.

    Returns
    -------
    Gateset
        A gateset containing LSGST estimates of gateStringToEstimate
        and the gates of targetGateset.
    """
    direct_lgst = gateset_with_lgst_gatestring_estimates(
        [gateStringToEstimate], dataset, prepStrs, effectStrs, targetGateset,
        True, gateLabelAliases, None, [gateStringLabel], svdTruncateTo, verbosity)

    # LEXICOGRAPHICAL VS MATRIX ORDER
    gatestrings = prepStrs + effectStrs + [ prepStr + effectStr for prepStr in prepStrs for effectStr in effectStrs ]
    for gateLabel in direct_lgst.gates:
        gatestrings.extend( [ prepStr + _objs.GateString( (gateLabel,), bCheck=False) + effectStr
                              for prepStr in prepStrs for effectStr in effectStrs ] )

    aliases = {} if (gateLabelAliases is None) else gateLabelAliases.copy()
    aliases[gateStringLabel] = _tools.find_replace_tuple(gateStringToEstimate,gateLabelAliases)
    
    _, direct_lsgst = _core.do_mc2gst(
        dataset, direct_lgst, gatestrings,
        minProbClipForWeighting=minProbClipForWeighting,
        probClipInterval=probClipInterval, verbosity=verbosity,
        gateLabelAliases=aliases )

    return direct_lsgst


def direct_mc2gst_gatesets(gateStrings, dataset, prepStrs, effectStrs,
                           targetGateset, gateLabelAliases=None,
                           svdTruncateTo=None, minProbClipForWeighting=1e-4,
                           probClipInterval=(-1e6,1e6), verbosity=0):
    """
    Constructs a dictionary with keys == gate strings and values == Direct-LSGST GateSets.

    Parameters
    ----------
    gateStrings : list of GateString or tuple objects
        The gate strings to estimate using LSGST.  The elements of this list
        are the keys of the returned dictionary.

    dataset : DataSet
        The data to use for all LGST and LSGST estimates.

    prepStrs,effectStrs : list of GateStrings
        Fiducial GateString lists used to construct a informationally complete
        preparation and measurement.

    targetGateset : GateSet
        The target gate set used by LGST to extract gate labels and an initial gauge

    gateLabelAliases : dictionary, optional
        Dictionary whose keys are gate label "aliases" and whose values are tuples
        corresponding to what that gate label should be expanded into before querying
        the dataset. Defaults to the empty dictionary (no aliases defined)
        e.g. gateLabelAliases['Gx^3'] = ('Gx','Gx','Gx')

    svdTruncateTo : int, optional
        The Hilbert space dimension to truncate the gate matrices to using
        a SVD to keep only the largest svdToTruncateTo singular values of
        the I_tildle LGST matrix. Zero means no truncation.
        Defaults to dimension of `targetGateset`.

    minProbClipForWeighting : float, optional
        defines the clipping interval for the statistical weight used
        within the chi^2 function (see chi2fn).

    probClipInterval : 2-tuple, optional
        (min,max) to clip probabilities to within GateSet probability
        computation routines (see GateSet.bulk_fill_probs)

    verbosity : int, optional
        Verbosity value to send to do_lgst(...) and do_mc2gst(...) calls.

    Returns
    -------
    dict
        A dictionary that relates each gate string of gateStrings to a
        GateSet containing the LSGST estimate of that gate string stored under
        the gate label "GsigmaLbl", along with LSGST estimates of the gates in
        targetGateset.
    """
    printer = _objs.VerbosityPrinter.build_printer(verbosity)
    directLSGSTgatesets = {}
    printer.log("--- Direct LSGST precomputation ---")
    with printer.progress_logging(1):
        for i,sigma in enumerate(gateStrings):
            printer.show_progress(i, len(gateStrings), prefix="--- Computing gateset for string-", suffix='---')
            directLSGSTgatesets[sigma] = direct_mc2gst_gateset(
                sigma, "GsigmaLbl", dataset, prepStrs, effectStrs, targetGateset,
                gateLabelAliases, svdTruncateTo, minProbClipForWeighting,
                probClipInterval, verbosity)
            
    return directLSGSTgatesets


def direct_mlgst_gateset( gateStringToEstimate, gateStringLabel, dataset,
                          prepStrs, effectStrs, targetGateset,
                          gateLabelAliases=None, svdTruncateTo=None, minProbClip=1e-6,
                          probClipInterval=(-1e6,1e6), verbosity=0 ):
    """
    Constructs a gateset of MLEGST estimates for target gates and gateStringToEstimate.

    Starting with a Direct-LGST estimate for gateStringToEstimate, runs MLEGST
    using the same strings that LGST would have used to estimate gateStringToEstimate
    and each of the target gates.  That is, MLEGST is run with strings of the form:

    1. prepStr
    2. effectStr
    3. prepStr + effectStr
    4. prepStr + singleGate + effectStr
    5. prepStr + gateStringToEstimate + effectStr

    and the resulting Gateset estimate is returned.

    Parameters
    ----------
    gateStringToEstimate : GateString or tuple
        The single gate string to estimate using LSGST

    gateStringLabel : string
        The label for the estimate of gateStringToEstimate.
        i.e. gate_matrix = returned_gateset[gate_label]

    dataset : DataSet
        The data to use for LGST

    prepStrs,effectStrs : list of GateStrings
        Fiducial GateString lists used to construct a informationally complete
        preparation and measurement.

    targetGateset : GateSet
        The target gate set used by LGST to extract gate labels and an initial gauge

    gateLabelAliases : dictionary, optional
        Dictionary whose keys are gate label "aliases" and whose values are tuples
        corresponding to what that gate label should be expanded into before querying
        the dataset. Defaults to the empty dictionary (no aliases defined)
        e.g. gateLabelAliases['Gx^3'] = ('Gx','Gx','Gx')

    svdTruncateTo : int, optional
        The Hilbert space dimension to truncate the gate matrices to using
        a SVD to keep only the largest svdToTruncateTo singular values of
        the I_tildle LGST matrix. Zero means no truncation.
        Defaults to dimension of `targetGateset`.

    minProbClip : float, optional
        defines the minimum probability "patch point" used
        within the logl function.

    probClipInterval : 2-tuple, optional
        (min,max) to clip probabilities to within GateSet probability
        computation routines (see GateSet.bulk_fill_probs)

    verbosity : int, optional
        Verbosity value to send to do_lgst(...) and do_mlgst(...) calls.

    Returns
    -------
    Gateset
        A gateset containing MLEGST estimates of gateStringToEstimate
        and the gates of targetGateset.
    """
    direct_lgst = gateset_with_lgst_gatestring_estimates(
        [gateStringToEstimate], dataset, prepStrs, effectStrs, targetGateset,
        True, gateLabelAliases, None, [gateStringLabel], svdTruncateTo, verbosity)

    # LEXICOGRAPHICAL VS MATRIX ORDER
    gatestrings = prepStrs + effectStrs + [ prepStr + effectStr for prepStr in prepStrs for effectStr in effectStrs ]
    for gateLabel in direct_lgst.gates:
        gatestrings.extend( [ prepStr + _objs.GateString( (gateLabel,), bCheck=False) + effectStr
                              for prepStr in prepStrs for effectStr in effectStrs ] )

    aliases = {} if (gateLabelAliases is None) else gateLabelAliases.copy()
    aliases[gateStringLabel] = _tools.find_replace_tuple(gateStringToEstimate,gateLabelAliases)

    _, direct_mlegst = _core.do_mlgst(
        dataset, direct_lgst, gatestrings, minProbClip=minProbClip,
        probClipInterval=probClipInterval, verbosity=verbosity,
        gateLabelAliases=aliases )
    return direct_mlegst


def direct_mlgst_gatesets(gateStrings, dataset, prepStrs, effectStrs, targetGateset,
                          gateLabelAliases=None, svdTruncateTo=None, minProbClip=1e-6,
                          probClipInterval=(-1e6,1e6), verbosity=0):
    """
    Constructs a dictionary with keys == gate strings and values == Direct-MLEGST GateSets.

    Parameters
    ----------
    gateStrings : list of GateString or tuple objects
        The gate strings to estimate using MLEGST.  The elements of this list
        are the keys of the returned dictionary.

    dataset : DataSet
        The data to use for all LGST and LSGST estimates.

    prepStrs,effectStrs : list of GateStrings
        Fiducial GateString lists used to construct a informationally complete
        preparation and measurement.

    targetGateset : GateSet
        The target gate set used by LGST to extract gate labels and an initial gauge

    gateLabelAliases : dictionary, optional
        Dictionary whose keys are gate label "aliases" and whose values are tuples
        corresponding to what that gate label should be expanded into before querying
        the dataset. Defaults to the empty dictionary (no aliases defined)
        e.g. gateLabelAliases['Gx^3'] = ('Gx','Gx','Gx')

    svdTruncateTo : int, optional
        The Hilbert space dimension to truncate the gate matrices to using
        a SVD to keep only the largest svdToTruncateTo singular values of
        the I_tildle LGST matrix. Zero means no truncation.
        Defaults to dimension of `targetGateset`.

    minProbClip : float, optional
        defines the minimum probability "patch point" used
        within the logl function.

    probClipInterval : 2-tuple, optional
        (min,max) to clip probabilities to within GateSet probability
        computation routines (see GateSet.bulk_fill_probs)

    verbosity : int, optional
        Verbosity value to send to do_lgst(...) and do_mlgst(...) calls.

    Returns
    -------
    dict
        A dictionary that relates each gate string of gateStrings to a
        GateSet containing the MLEGST estimate of that gate string stored under
        the gate label "GsigmaLbl", along with MLEGST estimates of the gates in
        targetGateset.
    """
    printer = _objs.VerbosityPrinter.build_printer(verbosity)
    directMLEGSTgatesets = {}
    printer.log("--- Direct MLEGST precomputation ---")
    with printer.progress_logging(1):
        for i,sigma in enumerate(gateStrings):
            printer.show_progress(i, len(gateStrings), prefix="--- Computing gateset for string ", suffix="---")
            directMLEGSTgatesets[sigma] = direct_mlgst_gateset(
                sigma, "GsigmaLbl", dataset, prepStrs, effectStrs, targetGateset,
                gateLabelAliases, svdTruncateTo, minProbClip,
                probClipInterval, verbosity)
            
    return directMLEGSTgatesets


def focused_mc2gst_gateset( gateStringToEstimate, gateStringLabel, dataset,
                            prepStrs, effectStrs, startGateset,
                            gateLabelAliases=None, minProbClipForWeighting=1e-4,
                            probClipInterval=(-1e6,1e6), verbosity=0 ):
    """
    Constructs a gateset containing a single LSGST estimate of gateStringToEstimate.

    Starting with startGateset, run LSGST with the same gate strings that LGST
    would use to estimate gateStringToEstimate.  That is, LSGST is run with
    strings of the form:  prepStr + gateStringToEstimate + effectStr
    and return the resulting Gateset.

    Parameters
    ----------
    gateStringToEstimate : GateString or tuple
        The single gate string to estimate using LSGST

    gateStringLabel : string
        The label for the estimate of gateStringToEstimate.
        i.e. gate_matrix = returned_gateset[gate_label]

    dataset : DataSet
        The data to use for LGST

    prepStrs,effectStrs : list of GateStrings
        Fiducial GateString lists used to construct a informationally complete
        preparation and measurement.

    startGateset : GateSet
        The gate set to seed LSGST with. Often times obtained via LGST.

    gateLabelAliases : dictionary, optional
        Dictionary whose keys are gate label "aliases" and whose values are tuples
        corresponding to what that gate label should be expanded into before querying
        the dataset. Defaults to the empty dictionary (no aliases defined)
        e.g. gateLabelAliases['Gx^3'] = ('Gx','Gx','Gx')

    minProbClipForWeighting : float, optional
        defines the clipping interval for the statistical weight used
        within the chi^2 function (see chi2fn).

    probClipInterval : 2-tuple, optional
        (min,max) to clip probabilities to within GateSet probability
        computation routines (see GateSet.bulk_fill_probs)

    verbosity : int, optional
        Verbosity value to send do_mc2gst(...) call.

    Returns
    -------
    Gateset
        A gateset containing LSGST estimate of gateStringToEstimate.
    """
    gatestrings = [ prepStr + gateStringToEstimate + effectStr for prepStr in prepStrs for effectStr in effectStrs ]

    _, focused_lsgst = _core.do_mc2gst(
        dataset, startGateset, gatestrings,
        minProbClipForWeighting=minProbClipForWeighting,
        probClipInterval=probClipInterval,
        gateLabelAliases=gateLabelAliases,
        verbosity=verbosity)

    focused_lsgst.gates[gateStringLabel] = _objs.FullyParameterizedGate(
            focused_lsgst.product(gateStringToEstimate)) #add desired string as a separate labeled gate
    return focused_lsgst


def focused_mc2gst_gatesets(gateStrings, dataset, prepStrs, effectStrs,
                            startGateset, gateLabelAliases=None,
                            minProbClipForWeighting=1e-4,
                            probClipInterval=(-1e6,1e6), verbosity=0):
    """
    Constructs a dictionary with keys == gate strings and values == Focused-LSGST GateSets.

    Parameters
    ----------
    gateStrings : list of GateString or tuple objects
        The gate strings to estimate using LSGST.  The elements of this list
        are the keys of the returned dictionary.

    dataset : DataSet
        The data to use for all LGST and LSGST estimates.

    prepStrs,effectStrs : list of GateStrings
        Fiducial GateString lists used to construct a informationally complete
        preparation and measurement.

    startGateset : GateSet
        The gate set to seed LSGST with. Often times obtained via LGST.

    gateLabelAliases : dictionary, optional
        Dictionary whose keys are gate label "aliases" and whose values are tuples
        corresponding to what that gate label should be expanded into before querying
        the dataset. Defaults to the empty dictionary (no aliases defined)
        e.g. gateLabelAliases['Gx^3'] = ('Gx','Gx','Gx')

    minProbClipForWeighting : float, optional
        defines the clipping interval for the statistical weight used
        within the chi^2 function (see chi2fn).

    probClipInterval : 2-tuple, optional
        (min,max) to clip probabilities to within GateSet probability
        computation routines (see GateSet.bulk_fill_probs)

    verbosity : int, optional
        Verbosity value to send to do_mc2gst(...) call.

    Returns
    -------
    dict
        A dictionary that relates each gate string of gateStrings to a
        GateSet containing the LSGST estimate of that gate string stored under
        the gate label "GsigmaLbl".
    """

    printer = _objs.VerbosityPrinter.build_printer(verbosity)
    focusedLSGSTgatesets = {}
    printer.log("--- Focused LSGST precomputation ---")
    with printer.progress_logging(1):
        for i,sigma in enumerate(gateStrings):
            printer.show_progress(i, len(gateStrings), prefix="--- Computing gateset for string", suffix='---')
            focusedLSGSTgatesets[sigma] = focused_mc2gst_gateset(
                sigma, "GsigmaLbl", dataset, prepStrs, effectStrs, startGateset,
                gateLabelAliases, minProbClipForWeighting, probClipInterval, verbosity)
    return focusedLSGSTgatesets

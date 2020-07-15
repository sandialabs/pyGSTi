"""
Helper Functions for generating plots
"""
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

import numpy as _np
import warnings as _warnings

from .. import tools as _tools
from .. import objects as _objs
from ..objects import objectivefns as _objfns
from ..objects.smartcache import smart_cached
from ..objects.circuitlist import CircuitList as _CircuitList

#TODO REMOVE
#def total_count_matrix(gsplaq, dataset):
#    """
#    Computes the total count matrix for a base circuit.
#
#    Parameters
#    ----------
#    gsplaq : CircuitPlaquette
#        Obtained via :method:`CircuitStructure.get_plaquette`, this object
#        specifies which matrix indices should be computed and which circuits
#        they correspond to.
#
#    dataset : DataSet
#        The data used to specify the counts
#
#    Returns
#    -------
#    numpy array of shape (M,N)
#        total count values (sum of count values for each SPAM label)
#        corresponding to circuits where circuit is sandwiched
#        between the specified set of N prep-fiducial and M effect-fiducial
#        circuits.
#    """
#    ret = _np.nan * _np.ones(gsplaq.num_simplified_elements, 'd')
#    for i, j, opstr, elIndices, outcomes in gsplaq.iter_simplified():
#        ret[elIndices] = dataset[opstr].total
#        # OR should it sum only over outcomes, i.e.
#        # = sum([dataset[opstr][ol] for ol in outcomes])
#    return ret
#
#
#def count_matrices(gsplaq, dataset):
#    """
#    Computes spamLabel's count matrix for a base circuit.
#
#    Parameters
#    ----------
#    gsplaq : CircuitPlaquette
#        Obtained via :method:`CircuitStructure.get_plaquette`, this object
#        specifies which matrix indices should be computed and which circuits
#        they correspond to.
#
#    dataset : DataSet
#        The data used to specify the counts
#
#    spamlabels : list of strings
#        The spam labels to extract counts for, e.g. ['plus']
#
#    Returns
#    -------
#    numpy array of shape ( len(spamlabels), len(effect_strs), len(prep_strs) )
#        count values corresponding to spamLabel and circuits
#        where circuit is sandwiched between the each prep-fiducial and
#        effect-fiducial pair.
#    """
#    ret = _np.nan * _np.ones(gsplaq.num_simplified_elements, 'd')
#    for i, j, opstr, elIndices, outcomes in gsplaq.iter_simplified():
#        datarow = dataset[opstr]
#        ret[elIndices] = [datarow[ol] for ol in outcomes]
#    return ret
#
#
#def frequency_matrices(gsplaq, dataset):
#    """
#    Computes spamLabel's frequency matrix for a base circuit.
#
#    Parameters
#    ----------
#    gsplaq : CircuitPlaquette
#        Obtained via :method:`CircuitStructure.get_plaquette`, this object
#        specifies which matrix indices should be computed and which circuits
#        they correspond to.
#
#    dataset : DataSet
#        The data used to specify the frequencies
#
#    spamlabels : list of strings
#        The spam labels to extract frequencies for, e.g. ['plus']
#
#
#    Returns
#    -------
#    numpy array of shape ( len(spamlabels), len(effect_strs), len(prep_strs) )
#        frequency values corresponding to spamLabel and circuits
#        where circuit is sandwiched between the each prep-fiducial,
#        effect-fiducial pair.
#    """
#    return count_matrices(gsplaq, dataset) \
#        / total_count_matrix(gsplaq, dataset)
#
#
#def probability_matrices(gsplaq, model,
#                         probs_precomp_dict=None):
#    """
#    Computes spamLabel's probability matrix for a base circuit.
#
#    Parameters
#    ----------
#    gsplaq : CircuitPlaquette
#        Obtained via :method:`CircuitStructure.get_plaquette`, this object
#        specifies which matrix indices should be computed and which circuits
#        they correspond to.
#
#    model : Model
#        The model used to specify the probabilities
#
#    spamlabels : list of strings
#        The spam labels to extract probabilities for, e.g. ['plus']
#
#    probs_precomp_dict : dict, optional
#        A dictionary of precomputed probabilities.  Keys are circuits
#        and values are prob-dictionaries (as returned from Model.probabilities)
#        corresponding to each circuit.
#
#    Returns
#    -------
#    numpy array of shape ( len(spamlabels), len(effect_strs), len(prep_strs) )
#        probability values corresponding to spamLabel and circuits
#        where circuit is sandwiched between the each prep-fiducial,
#        effect-fiducial pair.
#    """
#    ret = _np.nan * _np.ones(gsplaq.num_simplified_elements, 'd')
#    if probs_precomp_dict is None:
#        if model is not None:
#            for i, j, opstr, elIndices, outcomes in gsplaq.iter_simplified():
#                probs = model.probs(opstr)
#                ret[elIndices] = [probs[ol] for ol in outcomes]
#    else:
#        for i, j, opstr, elIndices, _ in gsplaq.iter_simplified():
#            ret[elIndices] = probs_precomp_dict[opstr]  # precomp is already in element-array form
#    return ret
#
#
#@smart_cached
#def chi2_matrix(gsplaq, dataset, model, min_prob_clip_for_weighting=1e-4,
#                probs_precomp_dict=None):
#    """
#    Computes the chi^2 matrix for a base circuit.
#
#    Parameters
#    ----------
#    gsplaq : CircuitPlaquette
#        Obtained via :method:`CircuitStructure.get_plaquette`, this object
#        specifies which matrix indices should be computed and which circuits
#        they correspond to.
#
#    dataset : DataSet
#        The data used to specify frequencies and counts
#
#    model : Model
#        The model used to specify the probabilities and SPAM labels
#
#    min_prob_clip_for_weighting : float, optional
#        defines the clipping interval for the statistical weight (see chi2fn).
#
#    probs_precomp_dict : dict, optional
#        A dictionary of precomputed probabilities.  Keys are circuits
#        and values are prob-dictionaries (as returned from Model.probabilities)
#        corresponding to each circuit.
#
#    Returns
#    -------
#    numpy array of shape ( len(effect_strs), len(prep_strs) )
#        chi^2 values corresponding to circuits where
#        circuit is sandwiched between the each prep-fiducial,
#        effect-fiducial pair.
#    """
#    gsplaq_ds = gsplaq.expand_aliases(dataset, circuit_simplifier=model)
#    cnts = total_count_matrix(gsplaq_ds, dataset)
#    probs = probability_matrices(gsplaq, model,
#                                 probs_precomp_dict)
#    freqs = frequency_matrices(gsplaq_ds, dataset)
#
#    ret = _np.nan * _np.ones((gsplaq.rows, gsplaq.cols), 'd')
#    for (i, j, opstr, elIndices, _), (_, _, _, elIndices_ds, _) in zip(
#            gsplaq.iter_simplified(), gsplaq_ds.iter_simplified()):
#        chiSqs = _tools.chi2fn(cnts[elIndices_ds], probs[elIndices],
#                               freqs[elIndices_ds], min_prob_clip_for_weighting)
#        ret[i, j] = sum(chiSqs)  # sum all elements for each (i,j) pair
#    return ret
#
#
#@smart_cached
#def logl_matrix(gsplaq, dataset, model, min_prob_clip=1e-6,
#                probs_precomp_dict=None):
#    """
#    Computes the log-likelihood matrix of 2*( log(L)_upperbound - log(L) )
#    values for a base circuit.
#
#    Parameters
#    ----------
#    gsplaq : CircuitPlaquette
#        Obtained via :method:`CircuitStructure.get_plaquette`, this object
#        specifies which matrix indices should be computed and which circuits
#        they correspond to.
#
#    dataset : DataSet
#        The data used to specify frequencies and counts
#
#    model : Model
#        The model used to specify the probabilities and SPAM labels
#
#    min_prob_clip : float, optional
#        defines the minimum probability "patch-point" of the log-likelihood function.
#
#    probs_precomp_dict : dict, optional
#        A dictionary of precomputed probabilities.  Keys are circuits
#        and values are prob-dictionaries (as returned from Model.probabilities)
#        corresponding to each circuit.
#
#
#    Returns
#    -------
#    numpy array of shape ( len(effect_strs), len(prep_strs) )
#        logl values corresponding to circuits where
#        circuit is sandwiched between the each prep-fiducial,
#        effect-fiducial pair.
#    """
#    gsplaq_ds = gsplaq.expand_aliases(dataset, circuit_simplifier=model)
#
#    cnts = total_count_matrix(gsplaq_ds, dataset)
#    probs = probability_matrices(gsplaq, model,
#                                 probs_precomp_dict)
#    freqs = frequency_matrices(gsplaq_ds, dataset)
#
#    ret = _np.nan * _np.ones((gsplaq.rows, gsplaq.cols), 'd')
#    for (i, j, opstr, elIndices, _), (_, _, _, elIndices_ds, _) in zip(
#            gsplaq.iter_simplified(), gsplaq_ds.iter_simplified()):
#        logLs = _tools.two_delta_logl_term(cnts[elIndices_ds], probs[elIndices],
#                                        freqs[elIndices_ds], min_prob_clip)
#        ret[i, j] = sum(logLs)  # sum all elements for each (i,j) pair
#    return ret
#
#
#@smart_cached
#def tvd_matrix(gsplaq, dataset, model, probs_precomp_dict=None):
#    """
#    Computes the total-variational distance matrix of `0.5 * |p-f|`
#    values for a base circuit.
#
#    Parameters
#    ----------
#    gsplaq : CircuitPlaquette
#        Obtained via :method:`CircuitStructure.get_plaquette`, this object
#        specifies which matrix indices should be computed and which circuits
#        they correspond to.
#
#    dataset : DataSet
#        The data used to specify frequencies and counts
#
#    model : Model
#        The model used to specify the probabilities and SPAM labels
#
#    probs_precomp_dict : dict, optional
#        A dictionary of precomputed probabilities.  Keys are circuits
#        and values are prob-dictionaries (as returned from Model.probabilities)
#        corresponding to each circuit.
#
#
#    Returns
#    -------
#    numpy array of shape ( len(effect_strs), len(prep_strs) )
#        logl values corresponding to circuits where
#        circuit is sandwiched between the each prep-fiducial,
#        effect-fiducial pair.
#    """
#    gsplaq_ds = gsplaq.expand_aliases(dataset, circuit_simplifier=model)
#
#    probs = probability_matrices(gsplaq, model,
#                                 probs_precomp_dict)
#    freqs = frequency_matrices(gsplaq_ds, dataset)
#
#    ret = _np.nan * _np.ones((gsplaq.rows, gsplaq.cols), 'd')
#    for (i, j, opstr, elIndices, _), (_, _, _, elIndices_ds, _) in zip(
#            gsplaq.iter_simplified(), gsplaq_ds.iter_simplified()):
#        TVDs = 0.5 * _np.abs(probs[elIndices] - freqs[elIndices_ds])
#        ret[i, j] = sum(TVDs)  # sum all elements for each (i,j) pair
#    return ret


def small_eigenvalue_err_rate(sigma, direct_gst_models):
    """
    Compute per-gate error rate.

    The per-gate error rate, extrapolated from the smallest eigvalue
    of the Direct GST estimate of the given circuit sigma.

    Parameters
    ----------
    sigma : Circuit or tuple of operation labels
        The gate sequence that is used to estimate the error rate

    direct_gst_models : dictionary of Models
        A dictionary with keys = circuits and
        values = Models.

    Returns
    -------
    float
        the approximate per-gate error rate.
    """
    if sigma is None: return _np.nan  # in plot processing, "None" circuits = no plot output = nan values
    mdl_direct = direct_gst_models[sigma]
    minEigval = min(abs(_np.linalg.eigvals(mdl_direct.operations["GsigmaLbl"])))
    # (approximate) per-gate error rate; max averts divide by zero error
    return 1.0 - minEigval**(1.0 / max(len(sigma), 1))


def _eformat(f, prec):
    """
    Formatting routine for writing compact representations of
    numbers in plot boxes
    """
    if _np.isnan(f): return ""  # show NAN as blanks
    if prec == 'compact' or prec == 'compacthp':
        if f < 0:
            ef = _eformat(-f, prec)
            return "-" + ef if (ef != "0") else "0"

        if prec == 'compacthp':
            if f <= 0.5e-9:  # can't fit in 3 digits; 1e-9 = "1m9" is the smallest 3-digit (not counting minus signs)
                return "0"
            if f < 0.005:  # then need scientific notation since 3-digit float would be 0.00...
                s = "%.0e" % f
                try:
                    mantissa, exp = s.split('e')
                    exp = int(exp); assert(exp < 0)
                    if exp < -9: return "0"  # should have been caugth above, but just in case
                    return "%sm%d" % (mantissa, -exp)
                except:
                    return "?"
            if f < 1:
                z = "%.2f" % f  # print first two decimal places
                if z.startswith("0."): return z[1:]  # fails for '1.00'; then thunk down to next f<10 case
            if f < 10:
                return "%.1f" % f  # print whole number and tenths

        if f < 100:
            return "%.0f" % f  # print nearest whole number if only 1 or 2 digits

        #if f >= 100, minimal scientific notation, such as "4e7", not "4e+07"
        s = "%.0e" % f
        try:
            mantissa, exp = s.split('e')
            exp = int(exp)
            if exp >= 100: return "B"  # if number is too big to print
            if exp >= 10: return "*%d" % exp
            return "%se%d" % (mantissa, exp)
        except:
            return str(s)[0:3]

    elif type(prec) == int:
        if prec >= 0:
            return "%.*f" % (prec, f)
        else:
            return "%.*g" % (-prec, f)
    else:
        return "%g" % f  # fallback to general format


def _num_non_nan(array):
    ixs = _np.where(_np.isnan(_np.array(array).flatten()) == False)[0]  # noqa: E712
    return int(len(ixs))


def _all_same(items):
    return all(x == items[0] for x in items)


def _compute_num_boxes_dof(sub_mxs, sum_up, element_dof):
    """
    A helper function to compute the number of boxes, and corresponding
    number of degrees of freedom, for the GST chi2/logl boxplots.

    """
    if sum_up:
        s = _np.shape(sub_mxs)
        # Reshape the sub_mxs into a "flattened" form (as opposed to a
        # two-dimensional one)
        reshape_subMxs = _np.array(_np.reshape(sub_mxs, (s[0] * s[1], s[2], s[3])))

        #Get all the boxes where the entries are not all NaN
        non_all_NaN = reshape_subMxs[_np.where(_np.array([_np.isnan(k).all() for k in reshape_subMxs]) == False)]  # noqa: E712,E501
        s = _np.shape(non_all_NaN)
        dof_each_box = [_num_non_nan(k) * element_dof for k in non_all_NaN]

        # Don't assert this anymore -- just use average below
        if not _all_same(dof_each_box):
            _warnings.warn('Number of degrees of freedom different for different boxes!')

        # The number of boxes is equal to the number of rows in non_all_NaN
        n_boxes = s[0]

        if n_boxes > 0:
            # Each box is a chi2_(sum) random variable
            dof_per_box = _np.average(dof_each_box)
        else:
            dof_per_box = None  # unknown, since there are no boxes
    else:
        # Each box is a chi2_m random variable currently dictated by the number of
        # dataset degrees of freedom.
        dof_per_box = element_dof

        # Gets all the non-NaN boxes, flattens the resulting
        # array, and does the sum.
        n_boxes = _np.sum(~_np.isnan(sub_mxs).flatten())

    return n_boxes, dof_per_box


#TODO REMOVE
#def _compute_probabilities(gss, model, dataset, prob_clip_interval=(-1e6, 1e6),
#                           check=False, op_label_aliases=None,
#                           comm=None, smartc=None, wildcard=None):
#    """
#    Returns a dictionary of probabilities for each gate sequence in
#    CircuitStructure `gss`.
#    """
#    def smart(fn, *args, **kwargs):
#        if smartc:
#            return smartc.cached_compute(fn, args, kwargs)[1]
#        else:
#            if '_filledarrays' in kwargs: del kwargs['_filledarrays']
#            return fn(*args, **kwargs)
#
#    circuitList = gss.allstrs
#
#    #compute probabilities
#    #OLD: evt,lookup,_ = smart(model.bulk_evaltree, circuitList, dataset=dataset)
#    evt, _, _, lookup, outcomes_lookup = smart(model.bulk_evaltree_from_resources,
#                                               circuitList, comm, dataset=dataset)
#
#    # _np.empty(evt.num_final_elements(), 'd') - .zeros b/c of caching
#    bulk_probs = _np.zeros(evt.num_final_elements(), 'd')
#    smart(model.bulk_fill_probs, bulk_probs, evt, prob_clip_interval, check, comm, _filledarrays=(0,))
#    # bulk_probs indexed by [element_index]
#
#    if wildcard:
#        freqs = _np.empty(evt.num_final_elements(), 'd')
#        #ds_circuit_list = _tools.find_replace_tuple_list(circuitList, op_label_aliases)
#        ds_circuit_list = _tools.apply_aliases_to_circuits(circuitList, op_label_aliases)
#        for (i, opStr) in enumerate(ds_circuit_list):
#            cnts = dataset[opStr].counts; total = sum(cnts.values())
#            freqs[lookup[i]] = [cnts.get(x, 0) / total for x in outcomes_lookup[i]]
#
#        probs_in = bulk_probs.copy()
#        wildcard.update_probs(probs_in, bulk_probs, freqs, circuitList, lookup)
#
#    probs_dict = \
#        {circuitList[i]: bulk_probs.take(_tools.to_array(lookup[i]))
#         for i in range(len(circuitList))}
#    return probs_dict


#@smart_cached
def _compute_sub_mxs(gss, model, sub_mx_creation_fn, dataset=None, sub_mx_creation_fn_extra_arg=None):
    subMxs = [[sub_mx_creation_fn(gss.plaquette(x, y, True), x, y, sub_mx_creation_fn_extra_arg)
               for x in gss.used_xs] for y in gss.used_ys]
    #Note: subMxs[y-index][x-index] is proper usage
    return subMxs


#TODO REMOVE
#@smart_cached
#def direct_chi2_matrix(gsplaq, gss, dataset, direct_model,
#                       min_prob_clip_for_weighting=1e-4):
#    """
#    Computes the Direct-X chi^2 matrix for a base circuit sigma.
#
#    Similar to chi2_matrix, except the probabilities used to compute
#    chi^2 values come from using the "composite gate" of directModels[sigma],
#    a Model assumed to contain some estimate of sigma stored under the
#    operation label "GsigmaLbl".
#
#    Parameters
#    ----------
#    gsplaq : CircuitPlaquette
#        Obtained via :method:`CircuitStructure.get_plaquette`, this object
#        specifies which matrix indices should be computed and which circuits
#        (for accessing the dataset) they correspond to.
#
#    gss : CircuitStructure
#        The circuit structure object containing `gsplaq`.  The structure is
#        neede to create a special plaquette for computing probabilities from the
#        direct model containing a "GsigmaLbl" gate.
#
#    dataset : DataSet
#        The data used to specify frequencies and counts
#
#    direct_model : Model
#        Model which contains an estimate of sigma stored
#        under the operation label "GsigmaLbl".
#
#    min_prob_clip_for_weighting : float, optional
#        defines the clipping interval for the statistical weight (see chi2fn).
#
#
#    Returns
#    -------
#    numpy array of shape ( len(effect_strs), len(prep_strs) )
#        Direct-X chi^2 values corresponding to circuits where
#        circuit is sandwiched between the each (effectStr,prepStr) pair.
#    """
#    if len(gsplaq) > 0:  # skip cases with no strings
#        plaq_ds = gsplaq.expand_aliases(dataset, circuit_simplifier=direct_model)
#        plaq_pr = gss.create_plaquette(_objs.Circuit(("GsigmaLbl",)))
#        plaq_pr.simplify_circuits(direct_model)
#
#        cnts = total_count_matrix(plaq_ds, dataset)
#        probs = probability_matrices(plaq_pr, direct_model)  # no probs_precomp_dict
#        freqs = frequency_matrices(plaq_ds, dataset)
#
#        ret = _np.empty((plaq_ds.rows, plaq_ds.cols), 'd')
#        for (i, j, opstr, elIndices, _), (_, _, _, elIndices_ds, _) in zip(
#                plaq_pr.iter_simplified(), plaq_ds.iter_simplified()):
#            chiSqs = _tools.chi2fn(cnts[elIndices_ds], probs[elIndices],
#                                   freqs[elIndices_ds], min_prob_clip_for_weighting)
#            ret[i, j] = sum(chiSqs)  # sum all elements for each (i,j) pair
#
#        return ret
#    else:
#        return _np.nan * _np.ones((gsplaq.rows, gsplaq.cols), 'd')
#
#
#@smart_cached
#def direct_logl_matrix(gsplaq, gss, dataset, direct_model,
#                       min_prob_clip=1e-6):
#    """
#    Computes the Direct-X log-likelihood matrix, containing the values
#     of 2*( log(L)_upperbound - log(L) ) for a base circuit sigma.
#
#    Similar to logl_matrix, except the probabilities used to compute
#    LogL values come from using the "composite gate" of directModels[sigma],
#    a Model assumed to contain some estimate of sigma stored under the
#    operation label "GsigmaLbl".
#
#    Parameters
#    ----------
#    gsplaq : CircuitPlaquette
#        Obtained via :method:`CircuitStructure.get_plaquette`, this object
#        specifies which matrix indices should be computed and which circuits
#        (for accessing the dataset) they correspond to.
#
#    gss : CircuitStructure
#        The circuit structure object containing `gsplaq`.  The structure is
#        neede to create a special plaquette for computing probabilities from the
#        direct model containing a "GsigmaLbl" gate.
#
#    dataset : DataSet
#        The data used to specify frequencies and counts
#
#    direct_model : Model
#        Model which contains an estimate of sigma stored
#        under the operation label "GsigmaLbl".
#
#    min_prob_clip : float, optional
#        defines the minimum probability clipping.
#
#    Returns
#    -------
#    numpy array of shape ( len(effect_strs), len(prep_strs) )
#        Direct-X logL values corresponding to circuits where
#        circuit is sandwiched between the each (effectStr,prepStr) pair.
#    """
#    if len(gsplaq) > 0:  # skip cases with no strings
#        plaq_ds = gsplaq.expand_aliases(dataset, circuit_simplifier=direct_model)
#        plaq_pr = gss.create_plaquette(_objs.Circuit(("GsigmaLbl",)))
#        plaq_pr.simplify_circuits(direct_model)
#
#        cnts = total_count_matrix(plaq_ds, dataset)
#        probs = probability_matrices(plaq_pr, direct_model)  # no probs_precomp_dict
#        freqs = frequency_matrices(plaq_ds, dataset)
#
#        ret = _np.empty((plaq_ds.rows, plaq_ds.cols), 'd')
#        for (i, j, opstr, elIndices, _), (_, _, _, elIndices_ds, _) in zip(
#                plaq_pr.iter_simplified(), plaq_ds.iter_simplified()):
#            logLs = _tools.two_delta_logl_term(cnts[elIndices_ds], probs[elIndices],
#                                            freqs[elIndices_ds], min_prob_clip)
#            ret[i, j] = sum(logLs)  # sum all elements for each (i,j) pair
#        return ret
#    else:
#        return _np.nan * _np.ones((gsplaq.rows, gsplaq.cols), 'd')


@smart_cached
def dscompare_llr_matrices(gsplaq, dscomparator):
    """
    Computes matrix of 2*log-likelihood-ratios comparing the datasets of `dscomparator`.

    Parameters
    ----------
    gsplaq : CircuitPlaquette
        Obtained via :method:`CircuitStructure.get_plaquette`, this object
        specifies which matrix indices should be computed and which circuits
        they correspond to.

    dscomparator : DataComparator
        The object specifying the data to be compared.

    Returns
    -------
    numpy array of shape ( len(effect_strs), len(prep_strs) )
        log-likelihood-ratio values corresponding to the circuits
        where a base circuit is sandwiched between the each prep-fiducial and
        effect-fiducial pair.
    """
    ret = _np.nan * _np.ones((gsplaq.num_rows, gsplaq.num_cols), 'd')
    for i, j, opstr in gsplaq:
        ret[i, j] = dscomparator.llrs[opstr]
    return ret


@smart_cached
def drift_neglog10pvalue_matrices(gsplaq, drifttuple):
    """
    Computes matrix of -log10(pvalues) for testing the stable-circuit ("no drift") null hypothesis in each circuit.

    This uses the "max power in spectra" test.

    Parameters
    ----------
    gsplaq : CircuitPlaquette
        Obtained via :method:`CircuitStructure.get_plaquette`, this object
        specifies which matrix indices should be computed and which circuits
        they correspond to.

    drifttuple : 2-tuple
        The first element of the tuple is a StabilityAnalyzer. The second element is a
        tuple that specifies the hypothesis test(s) from which to extract the p-values.
        This can be None, and then the default is used.

    Returns
    -------
    numpy array of shape ( len(effect_strs), len(prep_strs) )
        -log10(pvalues) for testing the "no drift" null hypothesis, using the "max power in
        spectra" test, on the relevant sequences. This circuits correspond to the
        circuits where a base circuit is sandwiched between the each prep-fiducial
        and effect-fiducial pair.
    """
    ret = _np.nan * _np.ones((gsplaq.num_rows, gsplaq.num_cols), 'd')
    stabilityanalyzer = drifttuple[0]
    dictlabel = drifttuple[1]
    assert(dictlabel == ('circuit',)), "Currently can only create these matrices for this single type of test!"
    for i, j, opstr in gsplaq:
        try:
            pval = stabilityanalyzer.get_pvalue(dictlabel={'circuit': opstr}, cutoff=1e-16)
            ret[i, j] = -1 * _np.log10(pval)
        except:
            pass
    return ret


@smart_cached
def drift_maxtvd_matrices(gsplaq, drifttuple):
    """
    Computes matrix of max-tvds for quantifying the size of any detected drift.

    Parameters
    ----------
    gsplaq : CircuitPlaquette
        Obtained via :method:`CircuitStructure.get_plaquette`, this object
        specifies which matrix indices should be computed and which circuits
        they correspond to.

    drifttuple : 2-tuple
        The first element of the tuple is a StabilityAnalyzer. The second element is a
        tuple that specifies the estimatorkey, and the third element is an estimator
        name, that specifies the estimates to use (both can be None, and then the
        default is used).

    Returns
    -------
    numpy array of shape ( len(effect_strs), len(prep_strs) )
        The max tvd for quantifying deviations from the data mean. This
        circuits correspond to the circuits where a base circuit
        is sandwiched between the each prep-fiducial and effect-fiducial pair.
    """
    ret = _np.nan * _np.ones((gsplaq.num_rows, gsplaq.num_cols), 'd')
    stabilityanalyzer = drifttuple[0]
    estimatekey = drifttuple[1]
    estimator = drifttuple[2]
    for i, j, opstr in gsplaq:
        try:
            ret[i, j] = stabilityanalyzer.get_max_tvd_bound(opstr, dskey=None,
                                                            estimatekey=estimatekey, estimator=estimator)
        except:
            pass
    return ret


# future: delete this if we decide not to add this option back in. REMOVE
# @smart_cached
# def drift_maxpower_matrices(gsplaq, driftresults):
#     """
#     Computes matrix of max powers in the time-series power spectra. This
#     value is a reasonable proxy for how "drifty" the sequence appears
#     to be.

#     Parameters
#     ----------
#     gsplaq : CircuitPlaquette
#         Obtained via :method:`CircuitStructure.get_plaquette`, this object
#         specifies which matrix indices should be computed and which circuits
#         they correspond to.

#     driftresults : BasicDriftResults
#         The drift analysis results.

#     Returns
#     -------
#     numpy array of shape ( len(effect_strs), len(prep_strs) )
#         Matrix of max powers in the time-series power spectra forthe circuits where a
#         base circuit is sandwiched between the each prep-fiducial and effect-fiducial pair.

#     """
#     ret = _np.nan * _np.ones((gsplaq.rows, gsplaq.cols), 'd')
#     for i, j, opstr in gsplaq:
#         try:
#             ret[i, j] = driftresults.get_maxpower(sequence=opstr)
#         except:
#             pass
#     return ret


def rated_n_sigma(dataset, model, circuits, objfn_builder, np=None, wildcard=None, return_all=False, comm=None):
    """
    Computes the number of standard deviations of model violation between `model` and `data`.

    Function compares the data in `dataset` with the `model` model at the "points" (circuits)
    specified by `circuits`.

    Parameters
    ----------
    dataset : DataSet
        The data set.

    model : Model
        The model (model).

    circuits : CircuitList or list of Circuits
        The circuits to use when computing the model violation.  A
        :class:`CircuitList` object may be given to include additional information
        (e.g. aliases) along with the list of circuits.

    objfn_builder : ObjectiveFunctionBuilder
        Builds the objective function to be used to compute the model violation.

    np : int, optional
        The number of free parameters in the model.  If None, then
        `model.num_nongauge_params()` is used.

    wildcard : WildcardBudget
        A wildcard budget to apply to the objective function (`objective`),
        which increases the goodness of fit by adjusting (by an amount measured
        in TVD) the probabilities produced by `model` before comparing with
        the frequencies in `dataset`.  Currently, this functionality is only
        supported for `objective == "logl"`.

    return_all : bool, optional
        Returns additional information such as the raw and expected model
        violation (see below).

    comm : mpi4py.MPI.Comm, optional
        When not None, an MPI communicator for distributing the computation
        across multiple processors.

    Returns
    -------
    Nsig : float
        The number of sigma of model violaition
    rating : int
        A 1-5 rating (e.g. "number of stars") used to indicate the rough
        abililty of the model to fit the data (better fit = higher rating).
    modelViolation : float
        The raw value of the objective function.  Only returned when
        `return_all==True`.
    expectedViolation : float
        The expected value of the objective function.  Only returned when
        `return_all==True`.
    Ns, np : int
        The number of dataset and model parameters, respectively. Only
        returned when `return_all==True`.
    """
    if isinstance(objfn_builder, str):
        objfn_builder = _objfns.ObjectiveFunctionBuilder.create_from(objfn_builder)

    objfn = objfn_builder.build(model, dataset, circuits, {'comm': comm})
    if wildcard:
        objfn.terms()  # objfn used within wildcard objective fn must be pre-evaluated
        objfn = _objfns.LogLWildcardFunction(objfn, model.to_vector(), wildcard)
    fitqty = objfn.chi2k_distributed_qty(objfn.fn())

    aliases = circuits.op_label_aliases if isinstance(circuits, _CircuitList) else None
    ds_gstrs = _tools.apply_aliases_to_circuits(circuits, aliases)

    if hasattr(model, 'num_nongauge_params'):
        np = model.num_nongauge_params()
    else:
        np = model.num_params()
    Ns = dataset.degrees_of_freedom(ds_gstrs)  # number of independent parameters in dataset
    k = max(Ns - np, 1)  # expected chi^2 or 2*(logL_ub-logl) mean
    Nsig = (fitqty - k) / _np.sqrt(2 * k)
    if Ns <= np: _warnings.warn("Max-model params (%d) <= model params (%d)!  Using k == 1." % (Ns, np))
    #pv = 1.0 - _stats.chi2.cdf(chi2,k) # reject GST model if p-value < threshold (~0.05?)

    if Nsig <= 2: rating = 5
    elif Nsig <= 20: rating = 4
    elif Nsig <= 100: rating = 3
    elif Nsig <= 500: rating = 2
    else: rating = 1

    if return_all:
        return Nsig, rating, fitqty, k, Ns, np
    else:
        return Nsig, rating
